import argparse
import glob
import os
from typing import Dict, List

import torch
from PIL import Image
from torchvision import transforms as T

from BASI_Edit_Agent_Starter.train_bg_model_residual import BgResidualNet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create before/model/after triplet previews."
    )
    parser.add_argument("--before_glob", required=True, type=str)
    parser.add_argument("--after_glob", required=True, type=str)
    parser.add_argument("--model_ckpt", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    return parser.parse_args()


def collect_pairs(before_glob: str, after_glob: str) -> List[str]:
    before_paths = {os.path.basename(p): p for p in glob.glob(before_glob)}
    after_paths = {os.path.basename(p): p for p in glob.glob(after_glob)}

    common_names = sorted(set(before_paths.keys()) & set(after_paths.keys()))
    return [(name, before_paths[name], after_paths[name]) for name in common_names]


def resize_to_height(img: Image.Image, target_h: int) -> Image.Image:
    if img.height == target_h:
        return img
    width = int(round(img.width * target_h / img.height))
    return img.resize((width, target_h), Image.LANCZOS)


def load_model(model_ckpt: str, device: torch.device) -> torch.nn.Module:
    model = BgResidualNet(in_ch=3, base_ch=32).to(device)
    checkpoint = torch.load(model_ckpt, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = collect_pairs(args.before_glob, args.after_glob)
    if not pairs:
        print(
            f"No pairs found for before_glob={args.before_glob}, after_glob={args.after_glob}"
        )
        return

    os.makedirs(args.out_dir, exist_ok=True)

    model = load_model(args.model_ckpt, device)
    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()

    saved = 0
    for name, before_path, after_path in pairs:
        try:
            before_img = Image.open(before_path).convert("RGB")
            after_img = Image.open(after_path).convert("RGB")
        except Exception as exc:
            print(f"Skipping {name} due to error: {exc}")
            continue

        before_tensor = to_tensor(before_img).to(device)
        after_tensor = to_tensor(after_img).to(device)
        xb = before_tensor.unsqueeze(0)

        with torch.no_grad():
            residual = model(xb)
            model_img_tensor = torch.clamp(xb + residual, 0.0, 1.0).squeeze(0).cpu()

        model_pil = to_pil(model_img_tensor)
        before_pil = to_pil(before_tensor.cpu())
        after_pil = to_pil(after_tensor.cpu())

        target_h = max(before_pil.height, model_pil.height, after_pil.height)
        before_resized = resize_to_height(before_pil, target_h)
        model_resized = resize_to_height(model_pil, target_h)
        after_resized = resize_to_height(after_pil, target_h)

        total_w = before_resized.width + model_resized.width + after_resized.width
        triplet = Image.new("RGB", (total_w, target_h))

        x = 0
        for img in (before_resized, model_resized, after_resized):
            triplet.paste(img, (x, 0))
            x += img.width

        stem, _ = os.path.splitext(name)
        out_path = os.path.join(args.out_dir, f"{stem}_triplet.jpg")
        triplet.save(out_path, quality=95)
        saved += 1

    print(f"Saved {saved} triplet previews to {args.out_dir}")


if __name__ == "__main__":
    main()
