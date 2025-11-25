import argparse
import glob
import os
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms as T

from BASI_Edit_Agent_Starter.train_bg_model_residual import BgResidualNet


def parse_args():
    parser = argparse.ArgumentParser(description="Create before/model/after triplets.")
    parser.add_argument("--before_glob", type=str, required=True)
    parser.add_argument("--after_glob", type=str, required=True)
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    return parser.parse_args()


def resize_to_height(img: Image.Image, target_h: int) -> Image.Image:
    if img.height == target_h:
        return img
    w = int(img.width * target_h / img.height)
    return img.resize((w, target_h), Image.LANCZOS)


def load_pairs(before_glob: str, after_glob: str) -> List[Tuple[str, str]]:
    before_paths = sorted(glob.glob(before_glob))
    after_paths = sorted(glob.glob(after_glob))
    n = min(len(before_paths), len(after_paths))
    return list(zip(before_paths[:n], after_paths[:n])), n


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

    pairs, n = load_pairs(args.before_glob, args.after_glob)
    if n == 0:
        print(
            f"No pairs found for before_glob={args.before_glob}, after_glob={args.after_glob}"
        )
        return

    os.makedirs(args.out_dir, exist_ok=True)

    model = load_model(args.model_ckpt, device)
    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()

    saved = 0
    for before_path, after_path in pairs:
        try:
            before_img = Image.open(before_path).convert("RGB")
            after_img = Image.open(after_path).convert("RGB")
        except Exception as exc:
            print(f"Skipping pair ({before_path}, {after_path}) due to error: {exc}")
            continue

        xb = to_tensor(before_img).unsqueeze(0).to(device)

        with torch.no_grad():
            residual = model(xb)
            model_img_tensor = torch.clamp(xb + residual, 0.0, 1.0)

        model_img = to_pil(model_img_tensor.squeeze(0).cpu())

        target_h = max(before_img.height, model_img.height, after_img.height)
        before_resized = resize_to_height(before_img, target_h)
        model_resized = resize_to_height(model_img, target_h)
        after_resized = resize_to_height(after_img, target_h)

        total_w = (
            before_resized.width + model_resized.width + after_resized.width
        )
        triplet = Image.new("RGB", (total_w, target_h), (0, 0, 0))

        x = 0
        for img in (before_resized, model_resized, after_resized):
            triplet.paste(img, (x, 0))
            x += img.width

        base = os.path.splitext(os.path.basename(before_path))[0]
        out_path = os.path.join(args.out_dir, f"{base}_triplet.jpg")
        triplet.save(out_path, quality=95)
        saved += 1

    print(f"Saved {saved} triplet previews to {args.out_dir}")


if __name__ == "__main__":
    main()
