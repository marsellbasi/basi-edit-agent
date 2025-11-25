import argparse
import json
import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_OK = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def load_model(model_ckpt: str):
    """Load color model from JSON checkpoint."""
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


def apply_transform(arr, M, b, curves, max_side=None):
    """Apply color transform to image array."""
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
        required=True,
        type=str,
        help="Path to color model JSON checkpoint (e.g., checkpoints/color_v1_e20/color_model.json or BASI_ARCHIVE/models/color_v0/color_model.json)",
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

    print(f"Loading color model from: {args.model_ckpt}")
    M, b, curves = load_model(args.model_ckpt)

    processed = 0
    for src in tqdm(files, desc="Applying color model"):
        try:
            img = Image.open(src).convert("RGB")
            arr = np.array(img)
            y = apply_transform(arr, M, b, curves, max_side=args.max_side)

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
