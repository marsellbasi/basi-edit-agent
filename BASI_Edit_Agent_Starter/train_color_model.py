import argparse
import os
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml


# -----------------------
# Utility: reproducibility
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------
# Global color model
# -----------------------
class GlobalColorModel(nn.Module):
    """
    Simple global color transform:
        x -> affine RGB -> per-channel monotonic tone curves.
    Operates on full images (no spatial conv).
    """

    def __init__(self, n_curve_points: int = 256):
        super().__init__()
        # 3x3 matrix + 3-d bias
        self.M = nn.Parameter(torch.eye(3))
        self.b = nn.Parameter(torch.zeros(3))

        # Learnable per-channel tone curves, stored as positive diffs
        # shape: [3, n_curve_points-1]; we prepend a 0 and do cumsum
        self.n_curve_points = n_curve_points
        self.curve_diffs = nn.Parameter(torch.zeros(3, n_curve_points - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W] in [0, 1]
        returns same shape.
        """
        B, C, H, W = x.shape

        # Flatten to pixels: [B*H*W, 3]
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, 3)

        # Global affine color transform
        y = x_flat @ self.M.t() + self.b  # [Npix, 3]
        y = y.clamp(0.0, 1.0)

        # Build monotonic tone curves
        # curve_diffs -> positive increments -> cumulative -> normalized 0..1
        diffs = F.softplus(self.curve_diffs)  # [3, n_pts-1] >= 0
        zeros = torch.zeros(3, 1, device=x.device, dtype=x.dtype)
        curves = torch.cat([zeros, diffs], dim=1).cumsum(dim=1)  # [3, n_pts]
        curves = curves / curves[:, -1:].clamp(min=1e-6)

        # Map y through tone curves via per-channel linear interpolation
        n_pts = self.n_curve_points
        y_clamped = y.clamp(0.0, 1.0)
        pos = y_clamped * (n_pts - 1)  # [Npix, 3] in [0, n_pts-1]

        idx_lo = torch.floor(pos).long()               # [Npix, 3]
        idx_hi = torch.clamp(idx_lo + 1, max=n_pts-1)  # [Npix, 3]
        w_hi = (pos - idx_lo.float())                  # [Npix, 3]
        w_lo = 1.0 - w_hi

        # curves: [3, n_pts]; we want [Npix, 3]
        # Use advanced indexing: for each pixel/channel, pick the curve value
        c_idx = torch.arange(3, device=x.device).view(1, 3).expand_as(idx_lo)  # [Npix,3]

        lo_vals = curves[c_idx, idx_lo]  # [Npix, 3]
        hi_vals = curves[c_idx, idx_hi]  # [Npix, 3]

        y_tone = w_lo * lo_vals + w_hi * hi_vals  # [Npix, 3]

        # Reshape back to image using real B,H,W
        y_img = y_tone.view(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
        return y_img.clamp(0.0, 1.0)


IMG_OK = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# -----------------------
# Dataset loading
# -----------------------
def _load_image(path: Path, max_side: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h)) if max_side is not None else 1.0
    if scale != 1.0:
        img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3]
    t = torch.from_numpy(arr).permute(2, 0, 1)       # [3, H, W]
    return t


def load_pairs(ds_root: Path, dataset_version: str, split: str, max_side: int):
    """
    ds_root: root that contains dataset_v1/
    dataset_version: e.g. "dataset_v1"
    split: "train" or "val"
    Returns a list of (before, after) tensors.
    """
    ds_dir = ds_root / dataset_version
    split_dir = ds_dir / split
    before_dir = split_dir / "before"
    after_dir = split_dir / "after"

    pairs = []
    if not before_dir.exists():
        raise FileNotFoundError(f"Before dir not found: {before_dir}")
    if not after_dir.exists():
        raise FileNotFoundError(f"After dir not found: {after_dir}")

    for bf_path in sorted(before_dir.iterdir()):
        if not bf_path.is_file() or bf_path.suffix.lower() not in IMG_OK:
            continue
        af_path = after_dir / bf_path.name
        if not af_path.exists():
            # unmatched pair, skip
            continue

        xb = _load_image(bf_path, max_side)
        yb = _load_image(af_path, max_side)

        # Ensure same spatial size
        h = min(xb.shape[1], yb.shape[1])
        w = min(xb.shape[2], yb.shape[2])
        xb_crop = xb[:, :h, :w]
        yb_crop = yb[:, :h, :w]

        pairs.append((xb_crop, yb_crop))

    return pairs

def midtone_weighted_l1(pred, target, mid=0.6, width=0.25, floor=0.5, gain=1.0):
    """
    L1 loss that up-weights pixels whose ground-truth luminance is in the midtones.
    This tends to emphasize faces / skin / midtones without needing a skin mask.
    pred, target: [B, 3, H, W], values in [0, 1].
    """
    # Compute luminance from the *target* (your final edit)
    with torch.no_grad():
        r = target[:, 0:1]
        g = target[:, 1:2]
        b = target[:, 2:3]
        Y = 0.299 * r + 0.587 * g + 0.114 * b  # [B,1,H,W]

        # Gaussian weight centered on `mid`
        w = torch.exp(-((Y - mid) ** 2) / (2.0 * width * width))  # [B,1,H,W]
        # Keep a non-zero floor so shadows/highlights still matter
        w = floor + gain * w  # roughly in [floor, floor+gain]

    # Plain L1 per-pixel, averaged across channels
    l1 = torch.abs(pred - target).mean(dim=1, keepdim=True)  # [B,1,H,W]

    # Weighted mean
    return (w * l1).mean()

def save_color_model_json(model, out_path):
    """
    Save the learned color model parameters to a JSON file.

    This is what the apply script will later load from BASI_ARCHIVE/models/color_v0/color_model.json.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with torch.no_grad():
        data = {
            "M": model.M.detach().cpu().tolist(),
            "b": model.b.detach().cpu().tolist(),
            "curve_diffs": model.curve_diffs.detach().cpu().tolist(),
        }

    with open(out_path, "w") as f:
        json.dump(data, f)

# -----------------------
# Training
# -----------------------
def train(args):
    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg.get("dataset_root", "./BASI_EDIT_AGENT"))
    output_root = Path(cfg.get("output_root", "./BASI_EDIT_AGENT"))
    random_seed = cfg.get("random_seed", 42)

    set_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load data
    ds_root = dataset_root
    print("Dataset root:", ds_root)
    train_pairs = load_pairs(ds_root, args.dataset_version, "train", args.max_side)
    val_pairs = load_pairs(ds_root, args.dataset_version, "val", args.max_side)

    print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

    train_loader = DataLoader(train_pairs, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_pairs, batch_size=1, shuffle=False, num_workers=0)

    # Model + optimizer
    model = GlobalColorModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Checkpoint path
    ckpt_path = Path("BASI_ARCHIVE/models/color_v0/color_model.json")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if exists
    if ckpt_path.exists():
        print(f"Loading checkpoint from {ckpt_path} ...")
        with open(ckpt_path, "r") as f:
            ckpt = json.load(f)

        with torch.no_grad():
            M_ckpt = torch.tensor(ckpt["M"], dtype=torch.float32, device=device)
            b_ckpt = torch.tensor(ckpt["b"], dtype=torch.float32, device=device)
            model.M.copy_(M_ckpt)
            model.b.copy_(b_ckpt)

            if "curve_diffs" in ckpt:
                cd_ckpt = torch.tensor(ckpt["curve_diffs"], dtype=torch.float32, device=device)
                if cd_ckpt.shape == model.curve_diffs.shape:
                    model.curve_diffs.copy_(cd_ckpt)
                    print(f"Loaded curve_diffs with shape {cd_ckpt.shape}.")
                else:
                    print(
                        f"Skipping curve_diffs load: ckpt {cd_ckpt.shape} vs model {model.curve_diffs.shape}"
                    )
        print("Checkpoint loaded (M/b + optional curves).")
    else:
        print(f"No checkpoint found at {ckpt_path}, starting from scratch.")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        num_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            yhat = model(xb)

            # Plain global L1
            base_l1 = F.l1_loss(yhat, yb)

            # Midtone-weighted L1 (emphasize faces / skin / midtones)
            mt_l1 = midtone_weighted_l1(yhat, yb)

            # Combine them: 70% plain L1, 30% midtone-weighted
            loss = 0.7 * base_l1 + 0.3 * mt_l1

            loss.backward()
            opt.step()

            train_loss += loss.item() * xb.size(0)
            num_train += xb.size(0)

        train_l1 = train_loss / max(1, num_train)

        # Validation
        model.eval()
        val_loss = 0.0
        num_val = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                yhat = model(xb)

                base_l1 = F.l1_loss(yhat, yb)
                mt_l1 = midtone_weighted_l1(yhat, yb)
                val_loss_batch = 0.5 * base_l1 + 0.5 * mt_l1

                val_loss += val_loss_batch.item() * xb.size(0)
                num_val += xb.size(0)

        val_l1 = val_loss / max(1, num_val)

        print(f"Epoch {epoch}/{args.epochs}: train L1={train_l1:.4f} | val L1={val_l1:.4f}")

        # Save latest color model (JSON)
        checkpoint_path = "BASI_ARCHIVE/models/color_v0/color_model.json"
        save_color_model_json(model, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset_version", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_side", type=int, default=640)
    parser.add_argument("--patch_size", type=int, default=192)  # unused but kept for CLI compat

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
