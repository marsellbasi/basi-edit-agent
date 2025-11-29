import argparse
import glob
import os
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from train_bg_model_residual import BgResidualNet

IMG_OK = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def load_model(model_ckpt: str, device: torch.device, use_mask: bool = False) -> torch.nn.Module:
    """Load BgResidualNet from checkpoint."""
    model = BgResidualNet(in_ch=3, base_ch=32, use_mask=use_mask).to(device)
    ckpt = torch.load(model_ckpt, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


def load_mask(mask_path: str, img_size: tuple) -> torch.Tensor:
    """Load subject mask and resize to match image size."""
    if not os.path.exists(mask_path):
        return None
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(img_size, Image.NEAREST)
    mask_tensor = T.ToTensor()(mask)  # [1, H, W] in [0, 1]
    # Binarize: > 0.5 -> 1.0, else 0.0
    mask_tensor = (mask_tensor > 0.5).float()
    return mask_tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Apply BgResidualNet to images.")
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
        help="Directory to save cleaned images",
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
        help="Path to background residual checkpoint (.pt). If not provided, uses config.",
    )
    parser.add_argument(
        "--mask_glob",
        type=str,
        default=None,
        help="Optional glob pattern for subject masks (e.g., /path/to/masks/*.png). If provided, masks will be used as 4th channel input.",
    )
    parser.add_argument(
        "--residual_scale",
        type=float,
        default=None,
        help="Scale factor for residual before adding to input (e.g., 0.3 for milder effect). If not provided, uses 0.3.",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        help="Use subject mask as 4th channel input (requires --mask_glob)",
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
            bg_cfg = cfg.get("training", {}).get("bg_residual", {})
            model_dir = bg_cfg.get("model_dir", "checkpoints/bg_residual")
            args.model_ckpt = os.path.join(model_dir, "latest.pt")
        else:
            args.model_ckpt = "checkpoints/bg_residual/latest.pt"
    
    # Determine residual scale
    if args.residual_scale is None:
        args.residual_scale = 0.3  # Default
    
    # Check if masks should be used
    use_mask = args.use_mask and args.mask_glob is not None

    files = sorted([Path(p) for p in glob.glob(args.input_glob)])
    files = [p for p in files if p.suffix.lower() in IMG_OK]

    if not files:
        raise RuntimeError(f"No files matched input_glob: {args.input_glob}")

    # Load mask paths if provided
    mask_paths = {}
    if args.mask_glob:
        mask_files = sorted([Path(p) for p in glob.glob(args.mask_glob)])
        mask_files = [p for p in mask_files if p.suffix.lower() in IMG_OK]
        # Build mapping from input basename to mask path
        for mask_file in mask_files:
            # Match by basename (without extension)
            mask_paths[mask_file.stem] = mask_file
        if use_mask and len(mask_paths) == 0:
            print(f"Warning: --use_mask specified but no masks found for glob: {args.mask_glob}")
            use_mask = False

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_ckpt}")
    print(f"Residual scale: {args.residual_scale}")
    print(f"Use mask: {use_mask}")
    if use_mask:
        print(f"Mask glob: {args.mask_glob}")
        print(f"Found {len(mask_paths)} masks")
    print(f"Processing {len(files)} images...")

    model = load_model(args.model_ckpt, device, use_mask=use_mask)
    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()

    processed = 0
    for src in tqdm(files, desc="Applying BgResidualNet"):
        try:
            img = Image.open(src).convert("RGB")
            xb = to_tensor(img).unsqueeze(0).to(device)  # [1, 3, H, W]

            # Load mask if available
            mask = None
            if use_mask:
                mask_stem = src.stem
                if mask_stem in mask_paths:
                    mask_path = mask_paths[mask_stem]
                    mask = load_mask(str(mask_path), img.size)  # [1, H, W]
                    if mask is not None:
                        mask = mask.unsqueeze(0).to(device)  # [1, 1, H, W]

            with torch.no_grad():
                # Forward pass
                if use_mask and mask is not None:
                    residual = model.forward_with_mask(xb, mask)
                else:
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
