"""
Training script for Stage 3 Skin Residual Model.

This script trains a residual UNet model for skin/face retouching.
The model predicts a residual that is added to the input image.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.residual_unet import SkinResidualNet
from training_utils import BeforeAfterDataset


def _center_crop(img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    img: (C, H, W)
    Returns a center-cropped tensor of shape (C, target_h, target_w).
    """
    _, h, w = img.shape
    top = max(0, (h - target_h) // 2)
    left = max(0, (w - target_w) // 2)
    return img[:, top:top + target_h, left:left + target_w]


def collate_fn(batch):
    """
    batch: list of (before_tensor, after_tensor)
    returns:
      xb: (B, 3, H, W)
      yb: (B, 3, H, W)
    We center-crop all images in the batch to the same (min_h, min_w)
    so torch.stack() will not fail on tiny mismatches like 426 vs 427.
    """
    befores, afters = zip(*batch)
    
    # Compute global minimum H and W across both before and after tensors
    all_h = [b.shape[1] for b in befores] + [a.shape[1] for a in afters]
    all_w = [b.shape[2] for b in befores] + [a.shape[2] for a in afters]
    target_h = min(all_h)
    target_w = min(all_w)
    
    cropped_befores = [
        _center_crop(b.contiguous(), target_h, target_w) for b in befores
    ]
    cropped_afters = [
        _center_crop(a.contiguous(), target_h, target_w) for a in afters
    ]
    
    xb = torch.stack(cropped_befores, dim=0)
    yb = torch.stack(cropped_afters, dim=0)
    
    return xb, yb


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ----- Logging startup info -----
    print("=" * 60)
    print("BASI Skin Residual Model Training (Stage 3)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dataset version: {args.dataset_version}")
    print(f"Train before glob: {args.train_before_glob}")
    print(f"Train after glob: {args.train_after_glob}")
    print(f"Val before glob: {args.val_before_glob}")
    print(f"Val after glob: {args.val_after_glob}")
    print(f"Max side: {args.max_side}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Identity weight: {args.identity_weight}")
    print(f"Model directory: {args.model_dir}")
    print("=" * 60)
    
    # ----- datasets -----
    train_ds = BeforeAfterDataset(
        before_glob=args.train_before_glob,
        after_glob=args.train_after_glob,
        max_side=args.max_side,
        is_train=True,
    )
    val_ds = BeforeAfterDataset(
        before_glob=args.val_before_glob,
        after_glob=args.val_after_glob,
        max_side=args.max_side,
        is_train=False,
    )
    
    print(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")
    
    # ----- dataloaders -----
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # ----- create model / optimizer -----
    model = SkinResidualNet(in_ch=3, base_ch=32, use_mask=False).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Checkpoints will be saved to: {args.model_dir}")
    print("=" * 60)
    
    # Cast config values to correct types (YAML may read them as strings)
    lr = float(args.lr)
    weight_decay = float(args.weight_decay)
    print(f"[train_skin_model] Using lr={lr} weight_decay={weight_decay}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_val = float("inf")
    start_epoch = 1
    
    # Handle resume logic
    if args.resume:
        if args.resume_ckpt:
            ckpt_path = args.resume_ckpt
        else:
            # Auto-resume from latest.pt in model_dir
            ckpt_path = os.path.join(args.model_dir, "latest.pt")
        
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 1) + 1
            best_val = ckpt.get("best_val", ckpt.get("val_l1", float("inf")))
            print(f"Resuming from epoch {start_epoch}, best_val={best_val:.4f}")
        else:
            if args.resume_ckpt:
                print(f"Warning: checkpoint {ckpt_path} not found, starting fresh")
            else:
                print(f"Warning: auto-resume checkpoint {ckpt_path} not found, starting fresh")
    
    print()  # Empty line before training starts
    
    identity_weight = args.identity_weight
    print(f"[train_skin_model] identity_weight={identity_weight}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_total = 0.0
        running_l1 = 0.0
        running_identity = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            residual = model(xb)
            # Scale residual to be reasonably small at the start
            residual = torch.tanh(residual) * 0.3
            pred = torch.clamp(xb + residual, 0.0, 1.0)
            
            # Main loss: L1 between prediction and target
            loss_l1 = F.l1_loss(pred, yb)
            
            # Identity loss: L1 between prediction and input (regularization)
            loss_identity = F.l1_loss(pred, xb)
            
            # Total loss
            loss = loss_l1 + identity_weight * loss_identity
            
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            batch_size = xb.size(0)
            running_total += loss.item() * batch_size
            running_l1 += loss_l1.item() * batch_size
            running_identity += loss_identity.item() * batch_size
            pbar.set_postfix(
                {"l1": f"{loss_l1.item():.4f}", "id": f"{loss_identity.item():.4f}"}
            )
        
        train_total = running_total / len(train_ds)
        train_l1 = running_l1 / len(train_ds)
        train_id = running_identity / len(train_ds)
        
        # --- validation ---
        model.eval()
        val_running = 0.0
        val_l1 = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                
                residual = model(xb)
                residual = torch.tanh(residual) * 0.3
                pred = torch.clamp(xb + residual, 0.0, 1.0)
                
                loss_l1 = F.l1_loss(pred, yb)
                val_running += loss_l1.item() * xb.size(0)
                val_l1 += loss_l1.item() * xb.size(0)
        
        val_l1 = val_running / len(val_ds)
        print(
            f"Epoch {epoch} done | Train L1: {train_l1:.4f} "
            f"(total={train_total:.4f}, id={train_id:.4f}) | Val L1: {val_l1:.4f}"
        )
        
        # Save checkpoints
        os.makedirs(args.model_dir, exist_ok=True)
        last_ckpt_path = os.path.join(args.model_dir, "latest.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_l1": val_l1,
                "best_val": best_val,
            },
            last_ckpt_path,
        )
        
        # Also save as epoch checkpoint
        epoch_ckpt_path = os.path.join(args.model_dir, f"epoch_{epoch:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_l1": val_l1,
                "best_val": best_val,
            },
            epoch_ckpt_path,
        )
        
        if val_l1 < best_val:
            best_val = val_l1
            ckpt_path = os.path.join(args.model_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_l1": val_l1,
                    "best_val": best_val,
                },
                ckpt_path,
            )
            print(f"✅ New best val L1={val_l1:.4f} — checkpoint saved to {ckpt_path}")
    
    print(f"\nTraining complete! Checkpoints saved to {args.model_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Train BASI Skin Residual Model (Stage 3)")
    
    p.add_argument("--config", type=str, default="config.yaml",
                   help="Path to config.yaml file")
    p.add_argument("--dataset_version", type=str, default=None,
                   help="Dataset version (e.g., skin_v1)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Number of epochs (overrides config)")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Batch size (overrides config)")
    p.add_argument("--max_side", type=int, default=None,
                   help="Max side length (overrides config)")
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate (overrides config)")
    p.add_argument("--weight_decay", type=float, default=None,
                   help="Weight decay (overrides config)")
    p.add_argument("--model_dir", type=str, default=None,
                   help="Model directory (overrides config)")
    p.add_argument("--identity_weight", type=float, default=None,
                   help="Identity weight (overrides config)")
    
    p.add_argument("--train_before_glob", type=str, default=None,
                   help="Glob for training before images (overrides dataset_version)")
    p.add_argument("--train_after_glob", type=str, default=None,
                   help="Glob for training after images (overrides dataset_version)")
    p.add_argument("--val_before_glob", type=str, default=None,
                   help="Glob for validation before images (overrides dataset_version)")
    p.add_argument("--val_after_glob", type=str, default=None,
                   help="Glob for validation after images (overrides dataset_version)")
    
    p.add_argument("--resume", action="store_true",
                   help="Resume training from checkpoint")
    p.add_argument("--resume_ckpt", type=str, default=None,
                   help="Path to checkpoint file. If --resume is set but no path given, auto-resumes from latest.pt in model_dir")
    
    args = p.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    skin_cfg = config.get("training", {}).get("skin_residual", {})
    dataset_root = config.get("dataset_root", "BASI_EDIT_AGENT")
    datasets_cfg = config.get("datasets", {})
    
    # Set defaults from config if not provided via CLI
    if args.dataset_version is None:
        args.dataset_version = skin_cfg.get("dataset_version", "skin_v1")
    
    # Get dataset paths from datasets config
    dataset_cfg = datasets_cfg.get(args.dataset_version, {})
    
    if args.train_before_glob is None:
        args.train_before_glob = dataset_cfg.get("train_before_glob", 
            os.path.join(dataset_root, args.dataset_version, "train", "before", "*.jpg"))
    if args.train_after_glob is None:
        args.train_after_glob = dataset_cfg.get("train_after_glob",
            os.path.join(dataset_root, args.dataset_version, "train", "after", "*.jpg"))
    if args.val_before_glob is None:
        args.val_before_glob = dataset_cfg.get("val_before_glob",
            os.path.join(dataset_root, args.dataset_version, "val", "before", "*.jpg"))
    if args.val_after_glob is None:
        args.val_after_glob = dataset_cfg.get("val_after_glob",
            os.path.join(dataset_root, args.dataset_version, "val", "after", "*.jpg"))
    
    if args.epochs is None:
        args.epochs = skin_cfg.get("epochs", 20)
    if args.batch_size is None:
        args.batch_size = skin_cfg.get("batch_size", 4)
    if args.max_side is None:
        args.max_side = skin_cfg.get("max_side", 1024)
    if args.lr is None:
        args.lr = skin_cfg.get("lr", 1e-4)
    if args.weight_decay is None:
        args.weight_decay = skin_cfg.get("weight_decay", 0.0)
    if args.model_dir is None:
        args.model_dir = skin_cfg.get("model_dir", "checkpoints/skin_residual")
    if args.identity_weight is None:
        args.identity_weight = skin_cfg.get("identity_weight", 0.3)
    
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)


# Example:
# python train_skin_model.py \
#   --config config.yaml \
#   --dataset_version skin_v1 \
#   --epochs 20 \
#   --batch_size 4 \
#   --max_side 1024

