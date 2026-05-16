import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

# --- PHYSICAL CONSTANTS ---
# These match the original event camera resolution
ORIGINAL_WIDTH = 640
ORIGINAL_HEIGHT = 480

def generate_gaussian_heatmaps(coords, out_h=64, out_w=64, orig_w=640, orig_h=480, sigma=3.0):
    """
    Converts 640x480 (X, Y) coordinates into 64x64 Gaussian heatmaps.
    """
    batch_size, seq_len, _ = coords.shape
    device = coords.device
    
    x_c = (coords[..., 0] * out_w).view(batch_size, seq_len, 1, 1)
    y_c = (coords[..., 1] * out_h).view(batch_size, seq_len, 1, 1)
    
    y_grid = torch.arange(out_h, device=device).view(1, 1, out_h, 1).float()
    x_grid = torch.arange(out_w, device=device).view(1, 1, 1, out_w).float()
    
    squared_dist = (x_grid - x_c)**2 + (y_grid - y_c)**2
    return torch.exp(-squared_dist / (2 * sigma**2))

def extract_coordinates(heatmaps, orig_w=640, orig_h=480):
    """
    Extracts continuous (X, Y) coordinates using Center of Mass (Soft-Argmax).
    """
    B, T, H, W = heatmaps.shape
    device = heatmaps.device

    heatmaps_clean = torch.pow(heatmaps, 3.0)
    
    # Normalize probabilities to sum to 1
    heatmaps_flat = heatmaps.view(B, T, -1)
    heatmaps_norm = heatmaps_flat / (heatmaps_flat.sum(dim=-1, keepdim=True) + 1e-8)
    heatmaps_norm = heatmaps_norm.view(B, T, H, W)
    
    x_grid = torch.arange(W, device=device).float()
    y_grid = torch.arange(H, device=device).float()
    
    pred_x = (heatmaps_norm.sum(dim=2) * x_grid).sum(dim=2)
    pred_y = (heatmaps_norm.sum(dim=3) * y_grid).sum(dim=2)
    
    # Scale continuous coordinates back to original sensor resolution
    pred_x = pred_x * (orig_w / float(W))
    pred_y = pred_y * (orig_h / float(H))
    
    return torch.stack([pred_x, pred_y], dim=-1)


def plot_full_video(pt_file, txt_file, epoch, device, model, plot_dir='plots'):
    """
    Plots an entire raw continuous video sequence.
    Extracts spatial standard deviation from the SNN heatmaps to render uncertainty shadows.
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # 1. Load the raw continuous video directly
    frames = torch.load(pt_file, map_location=device, weights_only=True)
    raw_targets = np.loadtxt(txt_file)
    
    images = frames.unsqueeze(0).to(device) # Shape: [1, T, C, H, W]
    
    # Physical ground truth coordinates
    t_x = raw_targets[:, 0]
    t_y = raw_targets[:, 1]
    t_blink = raw_targets[:, 2]

    # 2. DDP Safety: Unwrap the model for pure forward inference
    base_model = model.module if hasattr(model, 'module') else model
    base_model.eval()

    with torch.no_grad():
        base_model.reset_states()
        outputs_heatmaps, _ = base_model(images, reset=True)
        
        # 3. Extract the predicted center coordinates
        pred_coords = extract_coordinates(outputs_heatmaps)
        
        outputs_numpy = pred_coords.cpu().numpy()[0]
        o_x = outputs_numpy[:, 0]
        o_y = outputs_numpy[:, 1]
        
        # 4. Mathematically extract uncertainty (Spatial Standard Deviation)
        _, T, H_map, W_map = outputs_heatmaps.shape
        
        # Normalize the raw spikes into a spatial probability distribution
        probs = outputs_heatmaps.view(1, T, H_map * W_map)
        sharpening_factor = 2.0 
        probs = probs ** sharpening_factor
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        probs = probs.view(1, T, H_map, W_map)
        
        # Get 1D marginal probabilities
        prob_x = probs.sum(dim=2) 
        prob_y = probs.sum(dim=3) 
        
        # Generate spatial coordinate grids
        grid_x = torch.linspace(0, 1, W_map, device=device).view(1, 1, W_map)
        grid_y = torch.linspace(0, 1, H_map, device=device).view(1, 1, H_map)
        
        # Calculate the Mean (Center of Mass)
        mu_x = (prob_x * grid_x).sum(dim=-1, keepdim=True)
        mu_y = (prob_y * grid_y).sum(dim=-1, keepdim=True)
        
        # Calculate the Standard Deviation (Spread of the probability mass)
        std_x = torch.sqrt((prob_x * (grid_x - mu_x)**2).sum(dim=-1)).cpu().numpy()[0] * ORIGINAL_WIDTH
        std_y = torch.sqrt((prob_y * (grid_y - mu_y)**2).sum(dim=-1)).cpu().numpy()[0] * ORIGINAL_HEIGHT

    # Truncate arrays to match the shortest length in case of truncation
    eval_len = min(len(t_x), len(o_x))
    time_axis = np.arange(eval_len)
    
    t_x, t_y = t_x[:eval_len], t_y[:eval_len]
    o_x, o_y = o_x[:eval_len], o_y[:eval_len]
    std_x, std_y = std_x[:eval_len], std_y[:eval_len]

    t_blink = t_blink[:eval_len]

    # 5. Render the plots with uncertainty bounds
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Subplot X
    ax1.plot(time_axis, t_x, label='Target X', color='#2ecc71', linewidth=2.0)
    ax1.plot(time_axis, o_x, label='Predicted X', color='#e74c3c', linestyle='-', linewidth=1.5)
    ax1.fill_between(time_axis, o_x - std_x, o_x + std_x, color='#e74c3c', alpha=0.25, label=r'$\pm 1\sigma$ Uncertainty')
    ax1.set_ylabel('Horizontal Position (px)')
    ax1.set_title(f'Continuous SNN Tracking & Uncertainty | Epoch {epoch}')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_ylim(0, ORIGINAL_WIDTH)

    ax1.fill_between(time_axis, 0, ORIGINAL_WIDTH, where=(t_blink == 1), color='#7f8c8d', alpha=0.3, label='Blink Blackout')

    # Subplot Y
    ax2.plot(time_axis, t_y, label='Target Y', color='#2ecc71', linewidth=2.0)
    ax2.plot(time_axis, o_y, label='Predicted Y', color='#e74c3c', linestyle='-', linewidth=1.5)
    ax2.fill_between(time_axis, o_y - std_y, o_y + std_y, color='#e74c3c', alpha=0.25, label=r'$\pm 1\sigma$ Uncertainty')
    ax2.set_ylabel('Vertical Position (px)')
    ax2.set_xlabel('Frame Index (Sequence Progression)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.set_ylim(0, ORIGINAL_HEIGHT)

    ax2.fill_between(time_axis, 0, ORIGINAL_HEIGHT, where=(t_blink == 1), color='#7f8c8d', alpha=0.3, label='Blink Blackout')
    
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f'full_trajectory_epoch_{epoch}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"--> Full video diagnostic plot generated: {save_path}")


def save_debug_video(images, targets, outputs, epoch, out_dir):
    """Generates a 2x2 grid montage video blending event frames with predicted heatmaps."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"validation_epoch_{epoch}.mp4")

    if os.path.exists(out_path):
        try: os.remove(out_path)
        except OSError: pass
    
    # This outputs coordinates in the original 640x480 space
    pred_coords = extract_coordinates(outputs, orig_w=ORIGINAL_WIDTH, orig_h=ORIGINAL_HEIGHT)
    
    # Select up to 4 samples from the batch
    indices = [0, 1, 2, 3] if images.shape[0] >= 4 else list(range(images.shape[0]))
    B, T, C, H, W = images.shape
    scale = 4 
    tile_w, tile_h = W * scale, H * scale
    
    grid_cols = 2 if len(indices) >= 2 else 1
    grid_rows = math.ceil(len(indices) / 2)
    vid_w, vid_h = tile_w * grid_cols, tile_h * grid_rows
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (vid_w, vid_h))
    
    # Pre-process tensors
    batch_imgs = images[indices].cpu().numpy()
    batch_targs = targets[indices].cpu().numpy()
    batch_outs = outputs[indices].cpu().detach().numpy()      # Heatmaps: (B, T, 64, 64)
    batch_preds = pred_coords[indices].cpu().detach().numpy() # Coords: (B, T, 2)

    for t in range(T):
        tiles = []
        for i in range(len(indices)):
            # 1. Base Event Image (Blue = Negative, Red = Positive)
            event_img = np.zeros((H, W, 3), dtype=np.uint8)
            event_img[..., 2] = np.clip(batch_imgs[i, t, 1] * 255, 0, 255)
            event_img[..., 0] = np.clip(batch_imgs[i, t, 0] * 255, 0, 255)
            
            # 2. Heatmap Image overlay
            heatmap_val = batch_outs[i, t]
            hm_max = heatmap_val.max()
            if hm_max > 0:
                # Normalize probability to 0-255 for OpenCV
                heatmap_norm = (heatmap_val / hm_max * 255).astype(np.uint8)
            else:
                heatmap_norm = np.zeros_like(heatmap_val, dtype=np.uint8)
                
            heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            
            # 3. Blend Event Data and Heatmap (60% Events, 40% Heatmap)
            blended = cv2.addWeighted(event_img, 0.9, heatmap_color, 0.1, 0)
            
            # 4. Resize to visibility
            tile = cv2.resize(blended, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
            
            # 5. Draw Ground Truth Target (Green Circle)
            tx = int(batch_targs[i, t, 0] * tile_w)
            ty = int(batch_targs[i, t, 1] * tile_h)
            cv2.circle(tile, (tx, ty), 4, (0, 255, 0), -1)
            
            # 6. Draw Predicted Soft-Argmax Coordinate (White Crosshair)
            px = int((batch_preds[i, t, 0] / ORIGINAL_WIDTH) * tile_w)
            py = int((batch_preds[i, t, 1] / ORIGINAL_HEIGHT) * tile_h)
            cv2.drawMarker(tile, (px, py), (255, 255, 255), cv2.MARKER_CROSS, 8, 2)
            
            cv2.putText(tile, f"ID:{indices[i]}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            tiles.append(tile)
        
        # Fill empty slots if batch has fewer than 4 items
        while len(tiles) < (grid_cols * grid_rows):
            tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
            
        # Assemble 2x2 grid
        if len(tiles) == 4:
            top_row = np.hstack((tiles[0], tiles[1]))
            bottom_row = np.hstack((tiles[2], tiles[3]))
            full_frame = np.vstack((top_row, bottom_row))
        elif len(tiles) == 2:
            full_frame = np.hstack((tiles[0], tiles[1]))
        else:
            full_frame = tiles[0]
        
        out.write(full_frame)
        
    out.release()
