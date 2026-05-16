import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
import random
from tqdm import tqdm
import os
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np


class BlinkTensorDataset(Dataset):
    def __init__(self, root_dir, split="train", val_split=0.2, augment=False, cache_to_ram=True, target_res=(64, 64), map_to_3class=False):
        """
        Universal Loader for .pt tensors (RGB or SNN).
        """
        self.target_res = target_res
        self.root_dir = Path(root_dir)
        self.augment = augment
        self.cache_to_ram = cache_to_ram
        self.map_to_3class = map_to_3class

        all_files = sorted(list(self.root_dir.glob("*.pt")))
        if len(all_files) == 0:
            raise RuntimeError(f"No .pt files found in {root_dir}")

        random.seed(42)
        random.shuffle(all_files)

        split_idx = int(len(all_files) * (1 - val_split))

        if split == "train":
            self.files = all_files[:split_idx]
        else:
            self.files = all_files[split_idx:]

        print(f"[{split.upper()}] Loaded {len(self.files)} samples from {self.root_dir.name}")
        if self.map_to_3class:
            print(f"  -> BRIDGE ENABLED: Mapping 4 Classes [0,1,2,3] -> 3 Classes [0,1,2]")

        self.data_cache = []
        if self.cache_to_ram:
            print(f"  -> Loading {len(self.files)} files into RAM...")
            for f in tqdm(self.files, desc="Caching to RAM"):
                data = torch.load(f)
                self.data_cache.append(data)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.cache_to_ram:
            data = self.data_cache[idx]
        else:
            data = torch.load(self.files[idx])

        frames = data["frames"]
        labels = data["labels"].long()

        if frames.is_sparse:
            frames = frames.to_dense()


        if self.map_to_3class:
            labels = labels.clone()
            labels[labels == 3] = 1

        t, c, h, w = frames.shape

        if (h, w) != self.target_res:
            frames = F.interpolate(frames, size=self.target_res, mode='bilinear', align_corners=False)

        if self.augment:
            if random.random() > 0.5:
                frames = torch.flip(frames, dims=[-1])

            dx = random.randint(-4, 4)
            dy = random.randint(-4, 4)
            if dx != 0 or dy != 0:
                frames = F.pad(frames, (4, 4, 4, 4), value=0)
                start_y = 4 - dy
                start_x = 4 - dx
                frames = frames[..., start_y : start_y + 64, start_x : start_x + 64]

            if random.random() > 0.5:
                drop_prob = random.uniform(0.1, 0.3)
                mask = torch.empty_like(frames).bernoulli_(1 - drop_prob)
                frames = frames * mask

        return frames, labels

    def get_sample_weights(self):
        print("Scanning dataset to calculate sample weights...")
        is_blink_file = []
        iterable = self.data_cache if self.cache_to_ram else self.files

        for item in iterable:
            if self.cache_to_ram:
                labels = item["labels"]
                has_closed = (labels >= 1).any().item()
            else:
                has_closed = "NonBlinks" not in str(item)
                
            is_blink_file.append(has_closed)

        count_full = sum(is_blink_file)
        count_bg = len(is_blink_file) - count_full

        print(f"  -> Found: {count_full} Full Blink Videos")
        print(f"  -> Found: {count_bg} Background Videos")

        target_blink_ratio = 0.9

        if count_full > 0:
            weight_per_blink = target_blink_ratio / count_full
        else:
            weight_per_blink = 0

        if count_bg > 0:
            weight_per_bg = (1.0 - target_blink_ratio) / count_bg
        else:
            weight_per_bg = 0

        weights = []
        for is_blink in is_blink_file:
            if is_blink:
                weights.append(weight_per_blink)
            else:
                weights.append(weight_per_bg)

        print(f"  -> Weights assigned. Blink samples will constitute {target_blink_ratio * 100}% of the batch.")

        return torch.tensor(weights).double()


def validate(model, loader, device, is_snn=False, loss_fn=None, num_classes=4):
    """
    Validation loop that handles both 3-class and 4-class evaluation dynamically.
    """
    model.eval()
    all_preds = []
    all_targets = []
    val_loss = 0.0
    steps = 0

    with torch.no_grad():
        for frames, labels in loader:
            if is_snn and hasattr(model, 'reset_states'):
                model.reset_states()

            frames, labels = frames.to(device), labels.to(device)

            output = model(frames)

            if loss_fn:
                loss = loss_fn(output.reshape(-1, num_classes), labels.reshape(-1))
                val_loss += loss.item()
                steps += 1

            preds = output.argmax(dim=2)

            valid_mask = labels != -100
            
            all_preds.extend(preds[valid_mask].cpu().numpy().flatten())
            all_targets.extend(labels[valid_mask].cpu().numpy().flatten())

    avg_loss = val_loss / steps if steps > 0 else 0.0

    target_labels = list(range(num_classes))  # [0, 1, 2] or [0, 1, 2, 3]

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average=None, labels=target_labels, zero_division=0
    )

    acc = (np.array(all_preds) == np.array(all_targets)).mean() * 100

    cm = confusion_matrix(all_targets, all_preds, labels=target_labels)

    return val_loss, acc, precision, recall, f1, cm