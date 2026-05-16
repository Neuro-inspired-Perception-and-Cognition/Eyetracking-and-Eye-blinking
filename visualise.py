import argparse
import h5py
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import re
import scipy.interpolate
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# --- IMPORT YOUR MODELS ---
from model_blink import SNN_Model as BlinkModelBase
from model_track import EyeTrackingSNN

EXTRACTED_MEAN = -0.10029278639580454
EXTRACTED_VAR = 0.19582512047632403

class FrozenTemporalNorm(torch.nn.Module):
    def __init__(self, gn_layer, fixed_mean, fixed_var):
        super().__init__()
        # Steal the trained affine weights from the GroupNorm
        self.weight = gn_layer.weight.detach().clone()
        self.bias = gn_layer.bias.detach().clone()
        self.eps = gn_layer.eps
        
        # Lock in the mathematical statistics
        self.fixed_mean = fixed_mean
        self.fixed_var = fixed_var

    def forward(self, x):
        # x shape: (B, C, Time)
        x_norm = (x - self.fixed_mean) / torch.sqrt(torch.tensor(self.fixed_var + self.eps, device=x.device))
        return x_norm * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)


# --- HELPER: ROBUST WEIGHT LOADER ---
def load_weights_clean(model, path, device):
    print(f"Loading {path}...")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint,
                                                              dict) and "model_state_dict" in checkpoint else checkpoint
    # Strip DDP 'module.' prefixes and drop v_mem states
    clean_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if "v_mem" not in k}
    model.load_state_dict(clean_dict, strict=False)


# --- UTILS ---
def load_labels(label_path):
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            match = re.search(r'\((\d+),\s*(\d+),\s*(\d+)\)', line)
            if match:
                labels.append([int(match.group(1)), int(match.group(2)), int(match.group(3))])
    return np.array(labels)


def upsample_labels(labels, target_timestamps_us):
    t_original = np.arange(len(labels)) * 10000.0
    interp_func = scipy.interpolate.interp1d(t_original, labels, axis=0, kind='nearest', fill_value="extrapolate")
    return interp_func(target_timestamps_us)


def events_to_frame(x, y, p, H, W):
    img = np.zeros((2, H, W), dtype=np.float32)
    mask_on = (p == 1) & (x < W) & (y < H)
    mask_off = (p == 0) & (x < W) & (y < H)
    np.add.at(img[1], (y[mask_on], x[mask_on]), 1)
    np.add.at(img[0], (y[mask_off], x[mask_off]), 1)
    return torch.from_numpy(img)


# --- MAIN ---
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Initialize Base Model directly (No wrapper needed)
    blink_model = BlinkModelBase(num_classes=4).to(device)
    load_weights_clean(blink_model, args.blink_weights, device)
    blink_model.eval()

    # Overwrite the dynamic GroupNorm with our frozen layer
    blink_model.temporal_bn = FrozenTemporalNorm(blink_model.temporal_bn, EXTRACTED_MEAN, EXTRACTED_VAR)

    blink_model.reset_states()  # Initialize clean membrane potentials

    track_model = EyeTrackingSNN(input_dim=2).to(device)
    load_weights_clean(track_model, args.track_weights, device)
    track_model.eval()
    track_model.reset_states()

    with h5py.File(args.events, 'r') as f:
        ev_t, ev_x, ev_y, ev_p = f['events']['t'][:], f['events']['x'][:], f['events']['y'][:], f['events']['p'][:]
    if args.labels is not None:
        labels = load_labels(args.labels)
    else:
        labels = None

    dt = 3000
    t_min, t_max = ev_t[0], ev_t[-1]
    num_frames = int((t_max - t_min) / dt)
    frame_times = t_min + np.arange(num_frames) * dt

    if labels is not None:
        gt_data = upsample_labels(labels, frame_times - t_min)
        gt_x = gt_data[:, 0]
        gt_y = gt_data[:, 1]
        gt_blink = gt_data[:, 2]

    indices = np.searchsorted(ev_t, frame_times)
    indices = np.append(indices, len(ev_t))

    W_sensor = int(np.max(ev_x)) + 1
    H_sensor = int(np.max(ev_y)) + 1
    H_blink, W_blink = 64, 64


    CLASS_NAMES = {0: "OPEN", 1: "CLOSING", 2: "CLOSED", 3: "OPENING"}

    # INFERENCE
    blink_preds = []
    track_preds = []

    blink_temporal_buffer = np.zeros((2, H_blink, W_blink), dtype=np.float32)
    decay_rate = args.decay_rate

    print(f"Running Inference on {num_frames} frames...")
    with torch.no_grad():
        for i in tqdm(range(num_frames)):
            start, end = indices[i], indices[i + 1]
            
            blink_temporal_buffer *= decay_rate

            if start < end:
                ix, iy, ip = ev_x[start:end], ev_y[start:end], ev_p[start:end]
                
                # 1. Full resolution frame for Tracking Model
                frame_raw = events_to_frame(ix, iy, ip, H_sensor, W_sensor)
                
                # 2. Downscale coordinates directly for Blink Model
                shift_x = max(1, W_sensor // W_blink)
                shift_y = max(1, H_sensor // H_blink)
                
                ix_scaled = ix // shift_x
                iy_scaled = iy // shift_y
                
                mask_on = (ip == 1) & (ix_scaled < W_blink) & (iy_scaled < H_blink)
                mask_off = (ip == 0) & (ix_scaled < W_blink) & (iy_scaled < H_blink)
                
                current_frame_np = np.zeros((2, H_blink, W_blink), dtype=np.float32)
                np.add.at(current_frame_np[1], (iy_scaled[mask_on], ix_scaled[mask_on]), 1)
                np.add.at(current_frame_np[0], (iy_scaled[mask_off], ix_scaled[mask_off]), 1)
                
                blink_temporal_buffer += current_frame_np
                
                density_scalar = (128 * 128) / (W_sensor * H_sensor)
                frame_sum_np = blink_temporal_buffer * (density_scalar * args.calib_gain)
            else:
                frame_raw = torch.zeros((2, H_sensor, W_sensor), dtype=torch.float32)
                frame_sum_np = blink_temporal_buffer * 1.0

            frame_avg = F.interpolate(frame_raw.unsqueeze(0), size=(H_blink, W_blink), mode='area').to(device)
            sequential_input_track = frame_avg.unsqueeze(1).float()
            
            # Global sub-threshold bias to simulate missing noise floor
            frame_sum_np += args.calib_bias
            
            frame_sum = torch.from_numpy(frame_sum_np).to(device)
            sequential_input_blink = frame_sum.unsqueeze(0).unsqueeze(1).float()

            out_b, _ = blink_model(sequential_input_blink)
            pred_class = torch.argmax(out_b, dim=2).item()
            blink_preds.append(pred_class)

            track_out, _ = track_model(sequential_input_track, reset=False)
            heatmap = track_out[0, 0].cpu().numpy()
            track_preds.append(heatmap)

    blink_preds = np.array(blink_preds)
    
    unique, counts = np.unique(blink_preds, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"DISTRIBUTION -> Open: {dist.get(0, 0)} | Closing: {dist.get(1, 0)} | Closed: {dist.get(2, 0)} | Opening: {dist.get(3, 0)}")

    # --- METRICS (Definition: Anything != 0 is Blink) ---
    if labels is not None:
    
        y_true = (gt_blink != 0).astype(int)
        y_pred = (blink_preds != 0).astype(int)

        majority_class_count = np.max([np.sum(y_true == 0), np.sum(y_true == 1)])
        null_accuracy = majority_class_count / len(y_true)

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        except ValueError:
            tn, fp, fn, tp = 0, 0, 0, 0

        print("\n" + "=" * 60)
        print("BLINK METRIC REPORT (Definition: Blink = Closing/Closed/Opening)")
        print("=" * 60)
        print(f"Total Frames:       {len(y_true)}")
        print(f"True Blinks (GT):   {np.sum(y_true)} ({(np.sum(y_true) / len(y_true)) * 100:.1f}% of data)")
        print("-" * 30)
        print(f"Model Accuracy:     {np.mean(y_true == y_pred) * 100:.2f}%")
        print(f"Null Accuracy:      {null_accuracy * 100:.2f}% (Baseline)")
        print("-" * 30)
        print(f"Recall:             {recall * 100:.2f}%")
        print(f"Precision:          {precision * 100:.2f}%")
        print(f"F1 Score:           {f1:.4f}")
        print("-" * 30)
        print("CONFUSION MATRIX:")
        print(f"                 Pred Open    Pred Blink (1,2,3)")
        print(f"Actual Open      {tn:<9}    {fp:<9}")
        print(f"Actual Blink     {fn:<9}    {tp:<9}")

        # --- TRACKING METRICS ---
        pred_centers = []
        scale_x = W_sensor / 64.0
        scale_y = H_sensor / 64.0

        for heatmap in track_preds:
            h_idx, w_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            pred_centers.append((w_idx * scale_x, h_idx * scale_y))

        pred_centers = np.array(pred_centers)
        t_original = t_min + np.arange(len(labels)) * 10000.0

        spatial_errors = []

        for j, gt_time in enumerate(t_original):
            gt_x, gt_y, gt_b = labels[j]
            if gt_b != 0: continue

            closest_pred_idx = np.argmin(np.abs(frame_times - gt_time))

            if closest_pred_idx < len(pred_centers):
                pred_x, pred_y = pred_centers[closest_pred_idx]
                dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
                spatial_errors.append(dist)

        spatial_errors = np.array(spatial_errors)

        if len(spatial_errors) > 0:
            mee = np.mean(spatial_errors)
            pck_5 = np.mean(spatial_errors <= 5.0) * 100
            pck_10 = np.mean(spatial_errors <= 10.0) * 100
            pck_15 = np.mean(spatial_errors <= 15.0) * 100
        else:
            mee = pck_5 = pck_10 = pck_15 = 0.0

        print("\n" + "=" * 60)
        print("TRACKING METRIC REPORT (Evaluated at 100Hz GT)")
        print("=" * 60)
        print(f"Valid Open-Eye Frames: {len(spatial_errors)}")
        print(f"Mean Euclidean Error:  {mee:.2f} pixels")
        print("-" * 30)
        print(f"PCK @ 5 pixels:        {pck_5:.2f}%")
        print(f"PCK @ 10 pixels:       {pck_10:.2f}% (Standard Tolerance)")
        print(f"PCK @ 15 pixels:       {pck_15:.2f}%")
        print("=" * 60)

    # VIDEO GENERATION
    if args.visualise:
        fps = 60
        print(f"\nGenerating {fps} FPS video: {args.output} at {args.slow_down}x slow motion")
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W_sensor, H_sensor))

        vis_buffer = np.zeros((H_sensor, W_sensor, 3), dtype=np.float32)
        vis_decay_factor = 0.9  # Visual decay for aesthetic event trailing
        
        video_frames_to_write = 0.0
        smoothed_heatmap = np.zeros((H_sensor, W_sensor), dtype=np.float32)

        for i in tqdm(range(num_frames)):
            vis_buffer *= vis_decay_factor
            
            start, end = indices[i], indices[i + 1]
            if start < end:
                ix, iy, ip = ev_x[start:end], ev_y[start:end], ev_p[start:end]
                vis_buffer[iy[ip == 1], ix[ip == 1]] = [0, 0, 255]
                vis_buffer[iy[ip == 0], ix[ip == 0]] = [255, 0, 0]
            
            img = vis_buffer.astype(np.uint8)

            pred_class = blink_preds[i]
            heatmap = track_preds[i]
            heatmap_resized = cv2.resize(heatmap, (W_sensor, H_sensor))

            smoothed_heatmap = (heatmap_resized * 0.05) + (smoothed_heatmap * 0.95)

            h_min, h_max = np.min(smoothed_heatmap), np.max(smoothed_heatmap)
            heatmap_norm = (smoothed_heatmap - h_min) / (h_max - h_min + 1e-5)

            mask = heatmap_norm > 0.3
            heatmap_colored = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            img[mask] = cv2.addWeighted(img, 0.7, heatmap_colored, 0.3, 0)[mask]

            _, _, _, max_loc = cv2.minMaxLoc(heatmap_norm)
            pred_x, pred_y = max_loc
            cv2.drawMarker(img, (pred_x, pred_y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

            if args.gt_blink and args.labels is not None:
                show_box = (y_true[i] == 1)
                status = "GT BLINK"
            else:
                show_box = (pred_class != 0)
                status = f"Pred: {CLASS_NAMES.get(pred_class, '?')} ({pred_class})"

            if show_box:
                cv2.rectangle(img, (0, 0), (W_sensor - 1, H_sensor - 1), (0, 255, 0), 6)
                cv2.putText(img, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Frame rate decoupling logic
            video_time_step = (dt / 1_000_000.0) * args.slow_down
            video_frames_to_write += video_time_step * fps
            
            while video_frames_to_write >= 1.0:
                out.write(img)
                video_frames_to_write -= 1.0
                
        out.release()
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="DV008.h5")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--blink_weights", default="blink.pth")
    parser.add_argument("--track_weights", default="track.pth")
    parser.add_argument("--visualise", action="store_true")
    parser.add_argument("--gt_blink", action="store_true")
    parser.add_argument("--output", default="DV008.mp4")
    parser.add_argument("--slow_down", type=int, default=3)
    parser.add_argument("--calib_gain", type=float, default=2.0)
    parser.add_argument("--calib_bias", type=float, default=0.0)
    parser.add_argument("--decay_rate", type=float, default=0.5)
    args = parser.parse_args()
    run(args)