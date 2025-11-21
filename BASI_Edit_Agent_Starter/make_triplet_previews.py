from pathlib import Path
from PIL import Image

# Base paths (relative to this script)
BASE = Path(__file__).parent

before_dir = BASE / "BASI_EDIT_AGENT" / "dataset_v1" / "val" / "before"
after_dir  = BASE / "BASI_EDIT_AGENT" / "dataset_v1" / "val" / "after"
model_dir  = BASE / "BASI_ARCHIVE" / "previews" / "color_v0_h100_e10"

out_dir    = BASE / "BASI_ARCHIVE" / "previews" / "triplets_color_v0_h100_e10"
out_dir.mkdir(parents=True, exist_ok=True)

def resize_to_height(img, target_h=512):
    w = int(round(img.width * target_h / img.height))
    return img.resize((w, target_h), Image.LANCZOS)

def make_triplets(max_samples=None):
    count = 0
    for bf_path in sorted(before_dir.iterdir()):
        if not bf_path.is_file():
            continue

        name = bf_path.name
        af_path = after_dir / name
        md_path = model_dir / name

        if not af_path.exists() or not md_path.exists():
            # skip if we don't have all three
            continue

        before_img = Image.open(bf_path).convert("RGB")
        after_img  = Image.open(af_path).convert("RGB")
        model_img  = Image.open(md_path).convert("RGB")

        # Resize all to same height
        target_h = 512
        before_img = resize_to_height(before_img, target_h)
        model_img  = resize_to_height(model_img, target_h)
        after_img  = resize_to_height(after_img, target_h)

        w_total = before_img.width + model_img.width + after_img.width
        triplet = Image.new("RGB", (w_total, target_h))

        x = 0
        triplet.paste(before_img, (x, 0)); x += before_img.width
        triplet.paste(model_img,  (x, 0)); x += model_img.width
        triplet.paste(after_img,  (x, 0))

        out_path = out_dir / name
        triplet.save(out_path, quality=95)

        count += 1
        if max_samples is not None and count >= max_samples:
            break

    print(f"Saved {count} triplet previews to {out_dir}")

if __name__ == "__main__":
    make_triplets(max_samples=None)  # set e.g. 40 if you want a smaller subset
