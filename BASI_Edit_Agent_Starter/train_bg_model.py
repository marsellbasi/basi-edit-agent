import argparse
import os
import json
from typing import List

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# -----------------------------
# Dataset
# -----------------------------

class BgDataset(Dataset):
    """
    Generic pair dataset for:
        root / dataset_version / split / before
        root / dataset_version / split / after
    where filenames match between before/after.
    """

    def __init__(
        self,
        root: str,
        dataset_version: str = "bg_v1",
        split: str = "train",
        max_side: int = 640,
        patch_size: int = 192,
        center_crop: bool = False,
    ):
        super().__init__()
        self.before_dir = os.path.join(root, dataset_version, split, "before")
        self.after_dir = os.path.join(root, dataset_version, split, "after")

        if not os.path.isdir(self.before_dir):
            raise FileNotFoundError(f"Missing before_dir: {self.before_dir}")
        if not os.path.isdir(self.after_dir):
            raise FileNotFoundError(f"Missing after_dir: {self.after_dir}")

        all_files = sorted(os.listdir(self.before_dir))
        exts = (".jpg", ".jpeg", ".png")
        files: List[str] = [f for f in all_files if f.lower().endswith(exts)]

        # Keep only files that have a matching "after"
        self.files = [
            f for f in files
            if os.path.exists(os.path.join(self.after_dir, f))
        ]

        if len(self.files) == 0:
            raise RuntimeError(
                f"No matching before/after pairs found in {self.before_dir} and {self.after_dir}"
            )

        self.max_side = max_side
        self.patch_size = patch_size
        self.center_crop = center_crop
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def _load_and_resize(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        max_side = max(w, h)
        if self.max_side is not None and max_side > self.max_side:
            scale = self.max_side / max_side
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = img.resize((new_w, new_h), Image.BICUBIC)
        return img

    def __getitem__(self, idx):
        fname = self.files[idx]
        before_path = os.path.join(self.before_dir, fname)
        after_path = os.path.join(self.after_dir, fname)

        before_img = self._load_and_resize(before_path)
        after_img = self._load_and_resize(after_path)

        # Ensure we can crop a patch
        if self.patch_size is not None and self.patch_size > 0:
            bw, bh = before_img.size
            if bw < self.patch_size or bh < self.patch_size:
                # scale up just enough so min side >= patch_size
                scale = self.patch_size / min(bw, bh)
                new_w = int(round(bw * scale))
                new_h = int(round(bh * scale))
                before_img = before_img.resize((new_w, new_h), Image.BICUBIC)
                after_img = after_img.resize((new_w, new_h), Image.BICUBIC)
                bw, bh = before_img.size

            if self.center_crop:
                left = (bw - self.patch_size) // 2
                top = (bh - self.patch_size) // 2
            else:
                import random
                left = random.randint(0, bw - self.patch_size)
                top = random.randint(0, bh - self.patch_size)

            right = left + self.patch_size
            bottom = top + self.patch_size

            before_img = before_img.crop((left, top, right, bottom))
            after_img = after_img.crop((left, top, right, bottom))

        before_t = self.to_tensor(before_img)  # [3, H, W], in [0,1]
        after_t = self.to_tensor(after_img)

        return before_t, after_t, fname


# -----------------------------
# Model
# -----------------------------

class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(x + y)


class BgCleanerNet(nn.Module):
    """
    Small residual CNN:
      output = x + delta(x)
    so it learns to make small corrections to Color_v1 output.
    """

    def __init__(self, base_channels: int = 64, num_blocks: int = 8):
        super().__init__()
        self.entry = nn.Conv2d(3, base_channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            *[ResBlock(base_channels) for _ in range(num_blocks)]
        )
        self.exit = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.entry(x)
        feat = self.blocks(feat)
        delta = self.exit(feat)
        out = torch.clamp(x + delta, 0.0, 1.0)
        return out


# -----------------------------
# Training / Eval
# -----------------------------

def train_epoch(model, loader, opt, device):
    model.train()
    running = 0.0
    n = 0

    for i, (xb, yb, _) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        opt.zero_grad()
        pred = model(xb)
        loss = F.l1_loss(pred, yb)
        loss.backward()
        opt.step()

        running += loss.item() * xb.size(0)
        n += xb.size(0)

        if (i + 1) % 50 == 0:
            print(f"  [train] step {i+1}/{len(loader)}  L1={loss.item():.4f}", flush=True)

    return running / max(n, 1)


def eval_epoch(model, loader, device):
    model.eval()
    running = 0.0
    n = 0

    with torch.no_grad():
        for xb, yb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = F.l1_loss(pred, yb)
            running += loss.item() * xb.size(0)
            n += xb.size(0)

    return running / max(n, 1)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Train BASI Background Cleaner (Stage 2)")
    parser.add_argument("--data_root", type=str, default="BASI_EDIT_AGENT")
    parser.add_argument("--dataset_version", type=str, default="bg_v1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_side", type=int, default=640)
    parser.add_argument("--patch_size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--models_dir", type=str, default="BASI_ARCHIVE/models/bg_v0")
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Datasets
    train_ds = BgDataset(
        root=args.data_root,
        dataset_version=args.dataset_version,
        split="train",
        max_side=args.max_side,
        patch_size=args.patch_size,
        center_crop=False,
    )
    val_ds = BgDataset(
        root=args.data_root,
        dataset_version=args.dataset_version,
        split="val",
        max_side=args.max_side,
        patch_size=args.patch_size,
        center_crop=True,
    )

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model + optimizer
    model = BgCleanerNet(
        base_channels=args.base_channels,
        num_blocks=args.num_blocks,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = None

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_l1 = train_epoch(model, train_loader, opt, device)
        val_l1 = eval_epoch(model, val_loader, device)
        print(f"  Train L1: {train_l1:.4f} | Val L1: {val_l1:.4f}")

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "train_l1": train_l1,
            "val_l1": val_l1,
            "args": vars(args),
        }

        # Always save "latest"
        latest_path = os.path.join(args.models_dir, "bg_cleaner_latest.pt")
        torch.save(ckpt, latest_path)

        # Save best
        if best_val is None or val_l1 < best_val:
            best_val = val_l1
            best_path = os.path.join(args.models_dir, "bg_cleaner_best.pt")
            torch.save(ckpt, best_path)
            print(f"  👉 New best checkpoint saved (val L1={best_val:.4f})")


if __name__ == "__main__":
    main()
