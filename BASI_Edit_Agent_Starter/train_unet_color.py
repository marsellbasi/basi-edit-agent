"""
Training script for U-Net-based BASI Color (Stage 1 global color model).

This script trains a U-Net color/tone model using the same dataset
and training patterns as the baseline and HDRNet color models.
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

# Import UNet model
from models.unet_color_model import build_unet_color_model_from_config

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
    """Main training function for U-Net color model."""
    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg.get("dataset_root", "./BASI_EDIT_AGENT"))
    output_root = Path(cfg.get("output_root", "./BASI_EDIT_AGENT"))
    random_seed = cfg.get("random_seed", 42)

    set_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load UNet config
    unet_cfg = cfg.get("color_model", {}).get("unet", {})
    training_cfg = cfg.get("training", {}).get("unet_color", {})

    # Cast config values to correct types (YAML may read them as strings)
    lr = float(training_cfg.get("lr", 1.0e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    max_side = int(training_cfg.get("max_side", 1024)) if not args.max_side else args.max_side
    num_workers = int(training_cfg.get("num_workers", 0))
    epochs = int(training_cfg.get("epochs", 20)) if not args.epochs else args.epochs
    save_every = int(training_cfg.get("save_every", 1))
    
    # Loss weights
    loss_cfg = training_cfg.get("loss", {})
    l1_weight = float(loss_cfg.get("l1_weight", 1.0))

    # Ignore any batch_size from config/args for now.
    # Use batch_size=1 to avoid size mismatch errors when images have different dimensions.
    train_batch_size = 1
    val_batch_size = 1

    # Load data (reuse existing dataset loading)
    ds_root = dataset_root
    print("Dataset root:", ds_root)
    train_pairs = load_pairs(ds_root, args.dataset_version, "train", max_side)
    val_pairs = load_pairs(ds_root, args.dataset_version, "val", max_side)

    print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

    # DataLoaders
    train_loader = DataLoader(
        train_pairs,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_pairs,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Build model
    model = build_unet_color_model_from_config({"color_model": {"unet": unet_cfg}})
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Checkpoint directory
    model_dir = args.model_dir if args.model_dir else training_cfg.get("model_dir", "checkpoints/unet_color")
    ckpt_dir = Path(model_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = ckpt_dir / "latest.pt"

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume_ckpt:
        ckpt_path = Path(args.resume_ckpt)
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
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        train_l1 = 0.0
        num_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            residual = model(xb)  # Model returns residual only
            yhat = xb + residual  # Add residual to input to get final output
            yhat = torch.clamp(yhat, 0.0, 1.0)  # Clamp to valid range

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
                residual = model(xb)  # Model returns residual only
                yhat = xb + residual  # Add residual to input to get final output
                yhat = torch.clamp(yhat, 0.0, 1.0)  # Clamp to valid range

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
                    "config": unet_cfg,
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
                    "config": unet_cfg,
                },
                latest_ckpt,
            )
            print(f"Saved checkpoint to {epoch_ckpt} and {latest_ckpt}")

    print(f"\nTraining complete! Checkpoints saved to {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train U-Net-based BASI Color (Stage 1 global color model)"
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
        default=None,
        help="Maximum side length for image resizing (overrides config if provided)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config if provided)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Model checkpoint directory (overrides config if provided)",
    )
    parser.add_argument(
        "--resume_ckpt",
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

