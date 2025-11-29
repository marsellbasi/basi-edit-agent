"""
Training script for HDRNet-based BASI Color v1 (Stage 1 global color model).

This script trains an HDRNet-style color/tone model using the same dataset
and training patterns as the baseline color model, but with PyTorch checkpointing.
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

# Import HDRNet model
from models.hdrnet_color_model import build_hdrnet_color_model_from_config

# Reuse dataset loading from train_color_model
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_color_model import (
    set_seed,
    load_pairs,
    midtone_weighted_l1,
    IMG_OK,
)


# -----------------------
# Training
# -----------------------
def train(args):
    """Main training function for HDRNet color model."""
    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg.get("dataset_root", "./BASI_EDIT_AGENT"))
    output_root = Path(cfg.get("output_root", "./BASI_EDIT_AGENT"))
    random_seed = cfg.get("random_seed", 42)

    set_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load HDRNet config
    hdrnet_cfg = cfg.get("color_model", {}).get("hdrnet", {})
    training_cfg = cfg.get("training", {}).get("hdrnet_color", {})

    # Load data (reuse existing dataset loading)
    ds_root = dataset_root
    print("Dataset root:", ds_root)
    train_pairs = load_pairs(ds_root, args.dataset_version, "train", args.max_side)
    val_pairs = load_pairs(ds_root, args.dataset_version, "val", args.max_side)

    print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

    # DataLoaders
    batch_size = training_cfg.get("batch_size", args.batch_size if hasattr(args, 'batch_size') else 1)
    num_workers = training_cfg.get("num_workers", 0)
    train_loader = DataLoader(train_pairs, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_pairs, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Build model
    model = build_hdrnet_color_model_from_config(hdrnet_cfg)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    lr = training_cfg.get("lr", 1.0e-4)
    weight_decay = training_cfg.get("weight_decay", 0.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loss weights
    loss_cfg = training_cfg.get("loss", {})
    l1_weight = loss_cfg.get("l1_weight", 1.0)
    ssim_weight = loss_cfg.get("ssim_weight", 0.0)  # SSIM not implemented yet, can add later

    # Checkpoint directory
    ckpt_dir = Path("checkpoints/hdrnet_color")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = ckpt_dir / "latest.pt"

    # Resume from checkpoint if specified or if latest exists
    start_epoch = 1
    if args.resume_from:
        ckpt_path = Path(args.resume_from)
    elif latest_ckpt.exists() and args.resume:
        ckpt_path = latest_ckpt
    else:
        ckpt_path = None

    if ckpt_path and ckpt_path.exists():
        print(f"Loading checkpoint from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 1) + 1
        print(f"Resumed from epoch {start_epoch - 1}")
    else:
        print("Starting training from scratch.")

    # Training loop
    epochs = args.epochs if args.epochs else training_cfg.get("epochs", 20)
    save_every = training_cfg.get("save_every", 1)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        train_l1 = 0.0
        num_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            yhat = model(xb)

            # L1 loss
            base_l1 = F.l1_loss(yhat, yb)

            # Midtone-weighted L1 (reuse from train_color_model)
            mt_l1 = midtone_weighted_l1(yhat, yb)

            # Combine losses
            loss = l1_weight * (0.7 * base_l1 + 0.3 * mt_l1)

            loss.backward()
            opt.step()

            train_loss += loss.item() * xb.size(0)
            train_l1 += base_l1.item() * xb.size(0)
            num_train += xb.size(0)

        avg_train_loss = train_loss / max(1, num_train)
        avg_train_l1 = train_l1 / max(1, num_train)

        # Validation
        model.eval()
        val_loss = 0.0
        val_l1 = 0.0
        num_val = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                yhat = model(xb)

                base_l1 = F.l1_loss(yhat, yb)
                mt_l1 = midtone_weighted_l1(yhat, yb)
                val_loss_batch = l1_weight * (0.5 * base_l1 + 0.5 * mt_l1)

                val_loss += val_loss_batch.item() * xb.size(0)
                val_l1 += base_l1.item() * xb.size(0)
                num_val += xb.size(0)

        avg_val_loss = val_loss / max(1, num_val)
        avg_val_l1 = val_l1 / max(1, num_val)

        print(
            f"Epoch {epoch}/{epochs}: "
            f"train_loss={avg_train_loss:.4f} (L1={avg_train_l1:.4f}) | "
            f"val_loss={avg_val_loss:.4f} (L1={avg_val_l1:.4f})"
        )

        # Save checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            # Save epoch checkpoint
            epoch_ckpt = ckpt_dir / f"epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "config": hdrnet_cfg,
                },
                epoch_ckpt,
            )

            # Update latest checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "config": hdrnet_cfg,
                },
                latest_ckpt,
            )
            print(f"Saved checkpoint to {epoch_ckpt} and {latest_ckpt}")

    print(f"\nTraining complete! Checkpoints saved to {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train HDRNet-based BASI Color v1 (Stage 1 global color model)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--dataset_version",
        type=str,
        required=True,
        help="Dataset version (e.g., dataset_v1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config if provided)",
    )
    parser.add_argument(
        "--max_side",
        type=int,
        default=640,
        help="Maximum side length for image resizing",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config if provided)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint if it exists",
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

