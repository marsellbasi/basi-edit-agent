import argparse
import json
import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import yaml

IMG_OK = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def load_baseline_model(model_ckpt: str):
    """Load baseline color model from JSON checkpoint."""
    with open(model_ckpt, "r") as f:
        ckpt = json.load(f)
    M = np.array(ckpt["M"], dtype=np.float32)
    b = np.array(ckpt["b"], dtype=np.float32)
    diffs = np.array(ckpt["curve_diffs"], dtype=np.float32)
    # Reconstruct curves (softplus inverse at inference)
    curves = np.concatenate(
        [np.zeros((3, 1), dtype=np.float32), np.log1p(np.exp(diffs))], axis=1
    ).cumsum(axis=1)
    curves = curves / np.maximum(curves[:, -1:], 1e-6)
    return M, b, curves


def load_hdrnet_model(model_ckpt: str, config_path: str = None):
    """Load HDRNet color model from PyTorch checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config if provided
    if config_path:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        hdrnet_cfg = cfg.get("color_model", {}).get("hdrnet", {})
    else:
        # Try to load config from checkpoint
        hdrnet_cfg = {}
    
    # Import HDRNet model
    from models.hdrnet_color_model import build_hdrnet_color_model_from_config
    
    # Load checkpoint
    ckpt = torch.load(model_ckpt, map_location=device)
    
    # Get config from checkpoint if available
    if "config" in ckpt:
        hdrnet_cfg = ckpt["config"]
    
    # Build model
    model = build_hdrnet_color_model_from_config(hdrnet_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, device


def apply_baseline_transform(arr, M, b, curves, max_side=None):
    """Apply baseline color transform to image array."""
    # Resize if max_side is specified
    h, w = arr.shape[:2]
    if max_side is not None:
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            img = Image.fromarray(arr)
            new_size = (int(round(w * scale)), int(round(h * scale)))
            img = img.resize(new_size, Image.LANCZOS)
            arr = np.array(img)

    x = arr.astype(np.float32) / 255.0
    y = x @ M.T + b.reshape(1, 1, 3)
    y = np.clip(y, 0, 1)
    y_idx = (y * 255.0).clip(0, 255 - 1e-6)
    lo = np.floor(y_idx).astype(np.int32)
    hi = np.minimum(lo + 1, 255)
    w = y_idx - lo
    out = np.empty_like(y)
    for ch in range(3):
        c = curves[ch]
        y_lo = c[lo[..., ch]]
        y_hi = c[hi[..., ch]]
        out[..., ch] = (1 - w[..., ch]) * y_lo + w[..., ch] * y_hi
    out = (out * 255.0 + 0.5).astype(np.uint8)
    return out


def apply_hdrnet_transform(arr, model, device, max_side=None):
    """Apply HDRNet color transform to image array."""
    # Resize if max_side is specified
    h, w = arr.shape[:2]
    if max_side is not None:
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            img = Image.fromarray(arr)
            new_size = (int(round(w * scale)), int(round(h * scale)))
            img = img.resize(new_size, Image.LANCZOS)
            arr = np.array(img)
            h, w = arr.shape[:2]

    # Convert to tensor: [H, W, 3] -> [1, 3, H, W]
    x = arr.astype(np.float32) / 255.0
    x_tensor = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Apply model
    with torch.no_grad():
        y_tensor = model(x_tensor)
    
    # Convert back to numpy: [1, 3, H, W] -> [H, W, 3]
    y = y_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    y = np.clip(y, 0, 1)
    out = (y * 255.0 + 0.5).astype(np.uint8)
    return out


def load_unet_model(model_ckpt: str, config_path: str = None):
    """Load U-Net color model from PyTorch checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config if provided
    if config_path:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        unet_cfg = cfg.get("color_model", {}).get("unet", {})
    else:
        unet_cfg = {}
    
    # Import UNet model
    from models.unet_color_model import build_unet_color_model_from_config
    
    # Load checkpoint
    ckpt = torch.load(model_ckpt, map_location=device)
    
    # Get config from checkpoint if available
    if "config" in ckpt:
        unet_cfg = ckpt["config"]
    
    # Build model
    model = build_unet_color_model_from_config({"color_model": {"unet": unet_cfg}})
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, device


def apply_unet_transform(arr, model, device, residual_scale=0.35, max_side=None):
    """
    Apply U-Net color transform to image array.
    
    Args:
        arr: Input image array [H, W, 3] in uint8
        model: UNet model that predicts residual
        device: Torch device
        residual_scale: Scale factor for residual (0.0-1.0, lower = more subtle)
        max_side: Optional max side for resizing
    
    Returns:
        Output image array [H, W, 3] in uint8
    """
    # Resize if max_side is specified
    h, w = arr.shape[:2]
    if max_side is not None:
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            img = Image.fromarray(arr)
            new_size = (int(round(w * scale)), int(round(h * scale)))
            img = img.resize(new_size, Image.LANCZOS)
            arr = np.array(img)
            h, w = arr.shape[:2]

    # Convert to tensor: [H, W, 3] -> [1, 3, H, W]
    x = arr.astype(np.float32) / 255.0
    x_tensor = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Apply model to get residual
    with torch.no_grad():
        residual_tensor = model(x_tensor)  # [1, 3, H, W]
    
    # Blend: y = x + residual_scale * residual
    y_tensor = x_tensor + residual_scale * residual_tensor
    
    # Convert back to numpy: [1, 3, H, W] -> [H, W, 3]
    y = y_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    y = np.clip(y, 0, 1)
    out = (y * 255.0 + 0.5).astype(np.uint8)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Apply Stage 1 global color model to images."
    )
    parser.add_argument(
        "--input_glob",
        required=True,
        type=str,
        help="Glob pattern for input images (e.g., BASI_EDIT_AGENT/bg_v1/val/before/*.jpg)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory to save color-corrected outputs",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default=None,
        help="Path to color model checkpoint (JSON for baseline, .pt for HDRNet). If not provided, uses config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (required if using HDRNet or if model_ckpt not provided)",
    )
    parser.add_argument(
        "--max_side",
        type=int,
        default=None,
        help="Optional max side for resizing (default: use original size)",
    )

    args = parser.parse_args()

    files = sorted([Path(p) for p in glob.glob(args.input_glob)])
    files = [p for p in files if p.suffix.lower() in IMG_OK]

    if not files:
        raise RuntimeError(f"No files matched input_glob: {args.input_glob}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load config once if provided
    cfg = None
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    
    # Determine model type from config or checkpoint extension
    model_type = None
    if cfg:
        model_type = cfg.get("color_model", {}).get("type", "unet")  # Default to unet
    
    # If model_ckpt provided and no config, infer type from extension
    if args.model_ckpt and model_type is None:
        if args.model_ckpt.endswith(".pt") or args.model_ckpt.endswith(".pth"):
            # Can't distinguish between hdrnet and unet from extension alone
            # Default to unet if config not provided
            model_type = "unet"
        elif args.model_ckpt.endswith(".json"):
            model_type = "baseline"
    
    # Default to unet if not determined
    if model_type is None:
        model_type = "unet"
    
    print(f"Using model type: {model_type}")
    
    # Load model
    baseline_model = None
    hdrnet_model = None
    unet_model = None
    device = None
    residual_scale = 0.35  # Default for UNet (will be overridden from config if available)
    
    if model_type == "baseline":
        if not args.model_ckpt:
            # Try to find default baseline checkpoint
            default_ckpt = "BASI_ARCHIVE/models/color_v0/color_model.json"
            if os.path.exists(default_ckpt):
                args.model_ckpt = default_ckpt
            else:
                raise ValueError("Baseline model requires --model_ckpt or config must specify checkpoint path")
        
        print(f"Loading baseline model from: {args.model_ckpt}")
        M, b, curves = load_baseline_model(args.model_ckpt)
        baseline_model = (M, b, curves)
    
    elif model_type == "hdrnet":
        if not args.model_ckpt:
            # Try to find default HDRNet checkpoint
            default_ckpt = "checkpoints/hdrnet_color/latest.pt"
            if os.path.exists(default_ckpt):
                args.model_ckpt = default_ckpt
            else:
                raise ValueError("HDRNet model requires --model_ckpt or config must specify checkpoint path")
        
        print(f"Loading HDRNet model from: {args.model_ckpt}")
        hdrnet_model, device = load_hdrnet_model(args.model_ckpt, args.config)
    
    elif model_type == "unet":
        if not args.model_ckpt:
            # Try to find default UNet checkpoint
            default_ckpt = "checkpoints/unet_color/latest.pt"
            if os.path.exists(default_ckpt):
                args.model_ckpt = default_ckpt
            else:
                raise ValueError("UNet model requires --model_ckpt or config must specify checkpoint path")
        
        print(f"Loading UNet model from: {args.model_ckpt}")
        unet_model, device = load_unet_model(args.model_ckpt, args.config)
        
        # Read residual_scale from config for UNet (allows tuning strength at inference)
        # Lower values (e.g., 0.35) produce more subtle effects
        # Tune this via color_model.unet.residual_scale in config.yaml
        if cfg:
            unet_cfg = cfg.get("color_model", {}).get("unet", {})
            residual_scale = float(unet_cfg.get("residual_scale", 0.35))
        print(f"Using residual_scale: {residual_scale} (tune via color_model.unet.residual_scale in config.yaml)")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    processed = 0
    for src in tqdm(files, desc="Applying color model"):
        try:
            img = Image.open(src).convert("RGB")
            arr = np.array(img)
            
            if model_type == "baseline":
                y = apply_baseline_transform(arr, baseline_model[0], baseline_model[1], baseline_model[2], max_side=args.max_side)
            elif model_type == "hdrnet":
                y = apply_hdrnet_transform(arr, hdrnet_model, device, max_side=args.max_side)
            elif model_type == "unet":
                y = apply_unet_transform(arr, unet_model, device, residual_scale=residual_scale, max_side=args.max_side)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Save with same basename
            fname = src.name
            out_path = os.path.join(args.output_dir, fname)
            Image.fromarray(y).save(out_path, quality=95)
            processed += 1
        except Exception as e:
            print(f"Failed to process {src}: {e}")

    print(f"Processed {processed} images. Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
