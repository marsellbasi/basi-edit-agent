from pathlib import Path
from PIL import Image

# Paths (relative to BASI_Edit_Agent_Starter)
before_dir = Path("BASI_EDIT_AGENT/dataset_v1/val/before")
after_dir = Path("BASI_EDIT_AGENT/dataset_v1/val/after")
model_dir = Path("BASI_ARCHIVE/previews/color_v0_h100_e20")
out_dir = Path("BASI_ARCHIVE/previews/triplets_color_v0_h100_e20")

out_dir.mkdir(parents=True, exist_ok=True)

def resize_to_height(img, h):
    if img.height == h:
        return img
    w = int(img.width * h / img.height)
    return img.resize((w, h), Image.LANCZOS)

for before_path in sorted(before_dir.glob("*.jpg")):
    name = before_path.name
    model_path = model_dir / name
    after_path = after_dir / name

    if not model_path.exists() or not after_path.exists():
        # Skip if we don't have all three versions
        continue

    before_img = Image.open(before_path).convert("RGB")
    model_img = Image.open(model_path).convert("RGB")
    after_img = Image.open(after_path).convert("RGB")

    # Match heights
    target_h = max(before_img.height, model_img.height, after_img.height)
    before_img = resize_to_height(before_img, target_h)
    model_img = resize_to_height(model_img, target_h)
    after_img = resize_to_height(after_img, target_h)

    total_w = before_img.width + model_img.width + after_img.width
    triplet = Image.new("RGB", (total_w, target_h), (0, 0, 0))

    x = 0
    for img in (before_img, model_img, after_img):
        triplet.paste(img, (x, 0))
        x += img.width

    triplet.save(out_dir / name, quality=95)
