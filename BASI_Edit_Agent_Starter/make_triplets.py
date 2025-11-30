#!/usr/bin/env python3

"""
Make side-by-side triplet previews:

    [ BEFORE | MODEL OUTPUT | AFTER ]


General-purpose triplet preview generator for BASI Edit Agent.
Given three folders of images (before / model_output / after),
generates triplet previews laid out horizontally for easy side-by-side review.


Example usage:

python make_triplets.py \
  --before_glob "BASI_EDIT_AGENT/dataset_v1/val/before/*.jpg" \
  --model_glob  "BASI_Edit_Agent_Starter/test_outputs_unet_scaled/*.jpg" \
  --after_glob  "BASI_EDIT_AGENT/dataset_v1/val/after/*.jpg" \
  --output_dir  "triplets_unet_v1" \
  --max_triplets 50 \
  --resize_long_edge 1600
"""

import argparse
import glob
import os
import random
from pathlib import Path
from typing import Dict, List

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create before/model/after triplet previews."
    )
    parser.add_argument(
        "--before_glob",
        type=str,
        required=True,
        help="Glob pattern for original 'before' images.",
    )
    parser.add_argument(
        "--model_glob",
        type=str,
        required=True,
        help="Glob pattern for model output images.",
    )
    parser.add_argument(
        "--after_glob",
        type=str,
        required=True,
        help="Glob pattern for hand-edited 'after' images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save triplet JPEGs.",
    )
    parser.add_argument(
        "--max_triplets",
        type=int,
        default=0,
        help="Maximum number of triplets to generate (0 = no limit).",
    )
    parser.add_argument(
        "--resize_long_edge",
        type=int,
        default=1600,
        help="Resize each image so its long edge equals this size (preserves aspect ratio).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomly shuffle matched stems before picking triplets.",
    )
    return parser.parse_args()


def get_stem(path: str) -> str:
    """Extract stem (filename without extension) from a path."""
    return Path(path).stem


def index_by_stem(paths: List[str]) -> Dict[str, str]:
    """Build a dictionary mapping stem (filename without extension) to full path."""
    out: Dict[str, str] = {}
    for p in paths:
        stem = get_stem(p)
        if stem in out:
            # Warn if duplicate stems found
            print(f"Warning: duplicate stem '{stem}' found. Using: {p}")
        out[stem] = p
    return out


def load_and_resize(path: str, max_long_edge: int) -> Image.Image:
    """Load image and resize so long edge equals max_long_edge, preserving aspect ratio."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    long_edge = max(w, h)
    
    if long_edge <= max_long_edge:
        return img
    
    scale = max_long_edge / float(long_edge)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return img.resize((new_w, new_h), resample=Image.BICUBIC)


def make_triplet(
    before_path: str,
    model_path: str,
    after_path: str,
    out_path: str,
    max_long_edge: int,
) -> None:
    """Create a horizontal triplet image from before, model, and after images."""
    # Load and resize each image
    before = load_and_resize(before_path, max_long_edge)
    model = load_and_resize(model_path, max_long_edge)
    after = load_and_resize(after_path, max_long_edge)
    
    # Compute target height (max of all three)
    target_h = max(before.height, model.height, after.height)
    
    # Create a canvas with target height
    total_w = before.width + model.width + after.width
    canvas = Image.new("RGB", (total_w, target_h), (0, 0, 0))
    
    # Paste images horizontally, centering vertically if needed
    x = 0
    for img in (before, model, after):
        # Center vertically if image is shorter than target_h
        y_offset = (target_h - img.height) // 2
        canvas.paste(img, (x, y_offset))
        x += img.width
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path, "JPEG", quality=95)


def main() -> None:
    args = parse_args()
    
    # Load all images matching each glob
    before_paths = sorted(glob.glob(args.before_glob))
    model_paths = sorted(glob.glob(args.model_glob))
    after_paths = sorted(glob.glob(args.after_glob))
    
    # Check for empty globs
    if not before_paths:
        raise RuntimeError(f"No BEFORE images found for glob: {args.before_glob}")
    if not model_paths:
        raise RuntimeError(f"No MODEL images found for glob: {args.model_glob}")
    if not after_paths:
        raise RuntimeError(f"No AFTER images found for glob: {args.after_glob}")
    
    # Build dictionaries keyed by stem
    before_map = index_by_stem(before_paths)
    model_map = index_by_stem(model_paths)
    after_map = index_by_stem(after_paths)
    
    # Compute intersection of stems present in all three sets
    common_stems = set(before_map.keys()) & set(model_map.keys()) & set(after_map.keys())
    
    if not common_stems:
        raise RuntimeError(
            "No matching stems found across before/model/after sets. "
            "Make sure filenames (without extensions) match across all three directories."
        )
    
    # Convert to sorted list for reproducibility
    common_stems = sorted(common_stems)
    
    # Shuffle if requested
    if args.shuffle:
        random.shuffle(common_stems)
    
    # Limit to max_triplets if specified
    if args.max_triplets > 0:
        common_stems = common_stems[:args.max_triplets]
    
    # Print summary stats
    print(f"Found {len(before_paths)} BEFORE, {len(model_paths)} MODEL, {len(after_paths)} AFTER")
    print(f"Common stems: {len(common_stems)}")
    print(f"Writing {len(common_stems)} triplets to {args.output_dir}")
    
    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate triplets
    for i, stem in enumerate(common_stems, start=1):
        before_path = before_map[stem]
        model_path = model_map[stem]
        after_path = after_map[stem]
        
        out_name = f"{stem}_triplet.jpg"
        out_path = out_dir / out_name
        
        try:
            make_triplet(
                before_path=before_path,
                model_path=model_path,
                after_path=after_path,
                out_path=str(out_path),
                max_long_edge=args.resize_long_edge,
            )
            print(f"[{i}/{len(common_stems)}] Saved triplet: {out_path}")
        except Exception as exc:
            print(f"[{i}/{len(common_stems)}] Error processing {stem}: {exc}")
            continue
    
    print(f"\nDone. Triplets saved to: {out_dir}")


if __name__ == "__main__":
    main()


# Example: Generate triplets for UNet color model on dataset_v1
#
# python make_triplets.py \
#   --before_glob "BASI_EDIT_AGENT/dataset_v1/val/before/*.jpg" \
#   --model_glob  "BASI_Edit_Agent_Starter/test_outputs_unet_scaled/*.jpg" \
#   --after_glob  "BASI_EDIT_AGENT/dataset_v1/val/after/*.jpg" \
#   --output_dir  "triplets_unet_v1" \
#   --max_triplets 50 \
#   --resize_long_edge 1600
