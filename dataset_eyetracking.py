import torch
from torch.utils.data import Dataset
import numpy as np
from functools import lru_cache

class EyeTrackingDataset(Dataset):
    def __init__(self, pt_files, label_files, seq_len=90, stride=1, augment=False):
        self.pt_files = sorted(pt_files)
        self.label_files = sorted(label_files)
        self.seq_len = seq_len
        self.stride = stride
        self.augment = augment
        
        if len(self.pt_files) != len(self.label_files):
            raise ValueError("Mismatch between tensor files and label files.")
            
        self.samples = []
        self.labels_cache = {}
        
        # Build a mathematically precise index of available sequences
        for file_idx, label_path in enumerate(self.label_files):
            raw_labels = np.loadtxt(label_path)
            num_frames = len(raw_labels)
            
            # Scale labels once per file
            scaled_labels = np.zeros_like(raw_labels[:, :2])
            scaled_labels[:, 0] = raw_labels[:, 0] / 640
            scaled_labels[:, 1] = raw_labels[:, 1] / 480
            self.labels_cache[file_idx] = scaled_labels
            
            # Calculate exact valid starting indices to prevent DataLoader truncation crashes
            max_start_idx = num_frames - self.seq_len
            for start_idx in range(0, max_start_idx + 1, self.stride):
                self.samples.append((file_idx, start_idx))

    def __len__(self):
        return len(self.samples)

    # Use lru_cache to prevent loading the same .pt file hundreds of times sequentially
    @lru_cache(maxsize=4)
    def load_tensor(self, file_idx):
        return torch.load(self.pt_files[file_idx], map_location='cpu', weights_only=True)

    def __getitem__(self, index):
        file_idx, start_idx = self.samples[index]
        end_idx = start_idx + self.seq_len

        # Pull from RAM cache if recently accessed
        frames = self.load_tensor(file_idx)
        
        seq_frames = frames[start_idx:end_idx]
        seq_labels = torch.from_numpy(self.labels_cache[file_idx][start_idx:end_idx]).float()

        if self.augment:
            seq_frames = seq_frames.clone()
            seq_labels = seq_labels.clone()
            if torch.rand(1) < 0.5:
                drop_mask = torch.rand_like(seq_frames) > 0.10 
                seq_frames = seq_frames * drop_mask
            
            if torch.rand(1) < 0.5:
                # Limit shift to a maximum of 4 pixels
                max_shift = 4
                tx = torch.randint(-max_shift, max_shift + 1, (1,)).item()
                ty = torch.randint(-max_shift, max_shift + 1, (1,)).item()

                T, C, H, W = seq_frames.shape
                
                # Create a zero-filled tensor to act as the canvas
                shifted_frames = torch.zeros_like(seq_frames)
                
                # Calculate the mathematically valid bounding boxes for the slice
                src_y1, src_y2 = max(0, -ty), min(H, H - ty)
                dst_y1, dst_y2 = max(0, ty), min(H, H + ty)
                
                src_x1, src_x2 = max(0, -tx), min(W, W - tx)
                dst_x1, dst_x2 = max(0, tx), min(W, W + tx)
                
                # Copy the cropped original into the shifted destination coordinates
                shifted_frames[:, :, dst_y1:dst_y2, dst_x1:dst_x2] = seq_frames[:, :, src_y1:src_y2, src_x1:src_x2]
                seq_frames = shifted_frames

                # Mathematically translate the [0, 1] normalized target coordinates
                seq_labels[:, 0] = torch.clamp(seq_labels[:, 0] + (tx / float(W)), 0.0, 1.0)
                seq_labels[:, 1] = torch.clamp(seq_labels[:, 1] + (ty / float(H)), 0.0, 1.0)

        return seq_frames, seq_labels