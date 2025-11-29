#!/usr/bin/env python3

"""
Make side-by-side triplet previews:

    [ BEFORE | MODEL OUTPUT | AFTER ]



Designed for BASI Edit Agent UNet color tests, but generic enough

to reuse for HDRNet or future models.



Example:

python make_triplet_previews.py \

  --before_glob "BASI_Edit_Agent_Starter/test_inputs_unet/*.jpg" \

  --model_glob  "BASI_Edit_Agent_Starter/test_outputs_unet_scaled/*.jpg" \

  --after_glob  "BASI_Edit_Agent_Starter/test_targets_unet/*.jpg" \

  --output_dir  "BASI_Edit_Agent_Starter/triplets_unet_v1" \

  --max_triplets 50 \

  --resize_long_edge 1600

"""



import argparse

import glob

import os

from pathlib import Path

from typing import Dict, List, Tuple



from PIL import Image





def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Create before/model/after triplet previews.")

    parser.add_argument("--before_glob", type=str, required=True,

                        help="Glob for BEFORE images (originals).")

    parser.add_argument("--model_glob", type=str, required=True,

                        help="Glob for MODEL OUTPUT images.")

    parser.add_argument("--after_glob", type=str, required=True,

                        help="Glob for AFTER / target images.")

    parser.add_argument("--output_dir", type=str, required=True,

                        help="Directory to save triplet JPEGs.")

    parser.add_argument("--max_triplets", type=int, default=50,

                        help="Maximum number of triplets to generate.")

    parser.add_argument("--resize_long_edge", type=int, default=1600,

                        help="Resize so the longest edge of each image is at most this many pixels.")

    return parser.parse_args()





def index_by_basename(paths: List[str]) -> Dict[str, str]:

    out: Dict[str, str] = {}

    for p in paths:

        name = os.path.basename(p)

        out[name] = p

    return out





def load_and_resize(path: str, max_long_edge: int) -> Image.Image:

    img = Image.open(path).convert("RGB")

    w, h = img.size

    long_edge = max(w, h)

    if long_edge <= max_long_edge:

        return img



    scale = max_long_edge / float(long_edge)

    new_w = int(round(w * scale))

    new_h = int(round(h * scale))

    return img.resize((new_w, new_h), resample=Image.LANCZOS)





def make_triplet(

    before_path: str,

    model_path: str,

    after_path: str,

    out_path: str,

    max_long_edge: int,

) -> None:

    before = load_and_resize(before_path, max_long_edge)

    model = load_and_resize(model_path, max_long_edge)

    after = load_and_resize(after_path, max_long_edge)



    # Resize all three to the same height (using min height to avoid upsampling)

    heights = [before.height, model.height, after.height]

    target_h = min(heights)



    def resize_to_height(img: Image.Image, h: int) -> Image.Image:

        w, old_h = img.size

        if old_h == h:

            return img

        scale = h / float(old_h)

        new_w = int(round(w * scale))

        return img.resize((new_w, h), resample=Image.LANCZOS)



    before = resize_to_height(before, target_h)

    model = resize_to_height(model, target_h)

    after = resize_to_height(after, target_h)



    total_w = before.width + model.width + after.width

    canvas = Image.new("RGB", (total_w, target_h), (0, 0, 0))



    x = 0

    for img in (before, model, after):

        canvas.paste(img, (x, 0))

        x += img.width



    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    canvas.save(out_path, "JPEG", quality=95)





def main() -> None:

    args = parse_args()



    before_paths = sorted(glob.glob(args.before_glob))

    model_paths = sorted(glob.glob(args.model_glob))

    after_paths = sorted(glob.glob(args.after_glob))



    if not before_paths:

        raise RuntimeError(f"No BEFORE images found for glob: {args.before_glob}")

    if not model_paths:

        raise RuntimeError(f"No MODEL images found for glob: {args.model_glob}")

    if not after_paths:

        raise RuntimeError(f"No AFTER images found for glob: {args.after_glob}")



    before_map = index_by_basename(before_paths)

    model_map = index_by_basename(model_paths)

    after_map = index_by_basename(after_paths)



    common_names = sorted(set(before_map.keys()) & set(model_map.keys()) & set(after_map.keys()))

    if not common_names:

        raise RuntimeError("No matching filenames found across before/model/after sets.")



    if args.max_triplets > 0:

        common_names = common_names[: args.max_triplets]



    print(f"Found {len(common_names)} matching triplets.")

    out_dir = Path(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)



    for i, name in enumerate(common_names, start=1):

        before_path = before_map[name]

        model_path = model_map[name]

        after_path = after_map[name]



        stem = Path(name).stem

        out_path = out_dir / f"{stem}_triplet.jpg"



        print(f"[{i}/{len(common_names)}] {name} -> {out_path}")

        make_triplet(

            before_path=before_path,

            model_path=model_path,

            after_path=after_path,

            out_path=str(out_path),

            max_long_edge=args.resize_long_edge,

        )



    print(f"Done. Triplets saved to: {out_dir}")





if __name__ == "__main__":

    main()
