import argparse
import glob
import os

import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from train_bg_model_residual import BgResidualNet


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
        "--model_ckpt",
        type=str,
        default="checkpoints/bg_v1_residual_e10/bg_residual_best.pt",
        help="Path to residual checkpoint (.pt)",
    )
    parser.add_argument(
        "--residual_scale",
        type=float,
        default=0.6,
        help="Scale factor for residual before adding to input (e.g., 0.5 for milder effect)",
    )
    return parser.parse_args()


def load_model(model_ckpt: str, device: torch.device) -> torch.nn.Module:
    model = BgResidualNet(in_ch=3, base_ch=32).to(device)
    ckpt = torch.load(model_ckpt, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


def main():
    args = parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise RuntimeError(f"No files matched input_glob: {args.input_glob}")

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.model_ckpt, device)
    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()

    for path in tqdm(files, desc="Applying BgResidualNet"):
        img = Image.open(path).convert("RGB")
        xb = to_tensor(img).unsqueeze(0).to(device)

        with torch.no_grad():
            residual = model(xb)
            residual = torch.tanh(residual) * 0.3  # match training scaling
            residual_scaled = args.residual_scale * residual
            pred = torch.clamp(xb + residual_scaled, 0.0, 1.0)

        pred_img = to_pil(pred.squeeze(0).cpu())
        fname = os.path.basename(path)
        out_path = os.path.join(args.output_dir, fname)
        pred_img.save(out_path, quality=95)

    print(f"Processed {len(files)} images. Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()

