"""
Apply Stage 3 Skin Residual Model to images.

This script loads a trained SkinResidualNet checkpoint and applies it to input images.
"""

import argparse
import glob
import os
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from models.residual_unet import SkinResidualNet

IMG_OK = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def load_model(model_ckpt: str, device: torch.device) -> torch.nn.Module:
    """Load SkinResidualNet from checkpoint."""
    model = SkinResidualNet(in_ch=3, base_ch=32, use_mask=False).to(device)
    ckpt = torch.load(model_ckpt, map_location=device)
    
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Apply SkinResidualNet to images.")
    parser.add_argument(
        "--input_glob",
        required=True,
        type=str,
        help="Glob pattern for input images (e.g., /path/to/*.jpg)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory to save processed images",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default=None,
        help="Path to skin residual checkpoint (.pt). If not provided, uses config.",
    )
    parser.add_argument(
        "--residual_scale",
        type=float,
        default=None,
        help="Scale factor for residual before adding to input (e.g., 0.5 for milder effect). If not provided, uses 0.5.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu). If not provided, auto-detects.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    cfg = None
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    
    # Determine model checkpoint path
    if args.model_ckpt is None:
        if cfg:
            skin_cfg = cfg.get("training", {}).get("skin_residual", {})
            model_dir = skin_cfg.get("model_dir", "checkpoints/skin_residual")
            args.model_ckpt = os.path.join(model_dir, "best.pt")
        else:
            args.model_ckpt = "checkpoints/skin_residual/best.pt"
    
    # Determine residual scale (from config or default)
    if args.residual_scale is None:
        if cfg:
            # Try to get from config, but skin_residual doesn't have residual_scale in training config
            # Use default 0.5 for skin model (gentle retouching)
            args.residual_scale = 0.5
        else:
            args.residual_scale = 0.5  # Default for skin model
    
    # Determine device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    files = sorted([Path(p) for p in glob.glob(args.input_glob)])
    files = [p for p in files if p.suffix.lower() in IMG_OK]
    
    if not files:
        raise RuntimeError(f"No files matched input_glob: {args.input_glob}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_ckpt}")
    print(f"Residual scale: {args.residual_scale}")
    print(f"Processing {len(files)} images...")
    
    model = load_model(args.model_ckpt, device)
    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()
    
    processed = 0
    for src in tqdm(files, desc="Applying SkinResidualNet"):
        try:
            img = Image.open(src).convert("RGB")
            xb = to_tensor(img).unsqueeze(0).to(device)  # [1, 3, H, W]
            
            with torch.no_grad():
                # Forward pass
                residual = model(xb)
                
                # Scale residual (matching training scaling)
                residual = torch.tanh(residual) * 0.3  # Match training scaling
                residual_scaled = args.residual_scale * residual
                pred = torch.clamp(xb + residual_scaled, 0.0, 1.0)
            
            pred_img = to_pil(pred.squeeze(0).cpu())
            fname = src.name
            out_path = os.path.join(args.output_dir, fname)
            pred_img.save(out_path, quality=95)
            processed += 1
        except Exception as e:
            print(f"Failed to process {src}: {e}")
    
    print(f"Processed {processed}/{len(files)} images. Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()

