import argparse
import glob
import os

from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm


def build_model(device: torch.device):
    """
    Load a pretrained DeepLabV3 model for person segmentation.
    We assume COCO-style labels where class 15 corresponds to 'person'.
    """
    model = deeplabv3_resnet50(weights="COCO_WITH_VOC_LABELS_V1")
    model.to(device)
    model.eval()
    return model


def build_preprocess():
    """
    Standard ImageNet normalization; we can feed full-res images,
    but we cap the long side to keep memory in check.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def run_inference(model, preprocess, img: Image.Image, device: torch.device):
    """
    Run segmentation and return a binary PIL mask (0 background, 1 subject).
    We downscale for speed, run the model, then upsample the mask
    back to the original size.
    """
    orig_w, orig_h = img.size

    # Resize to keep the larger side <= 640 for efficiency
    max_side = 640
    scale = min(max_side / max(orig_w, orig_h), 1.0)
    if scale < 1.0:
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img_small = img.resize((new_w, new_h), Image.BILINEAR)
    else:
        img_small = img

    tensor = preprocess(img_small).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)["out"]  # [1, C, H, W]
        # DeepLabV3 trained on VOC-style labels: class 15 == 'person'
        person_class = 15
        pred = out.argmax(dim=1)  # [1, H, W]
        mask_small = (pred == person_class).float().squeeze(0).cpu()  # [H, W]

    # Upsample to original size
    mask_np = mask_small.numpy()
    mask_img = Image.fromarray((mask_np * 255).astype("uint8"))
    if mask_img.size != (orig_w, orig_h):
        mask_img = mask_img.resize((orig_w, orig_h), Image.NEAREST)

    # Convert to binary 0 / 1 for training convenience
    mask_bin = mask_img.point(lambda p: 1 if p >= 128 else 0)
    return mask_bin


def main():
    parser = argparse.ArgumentParser(description="Generate subject (person) masks for BASI bg model.")
    parser.add_argument("--before_glob", type=str, required=True,
                        help="Glob for BEFORE images, e.g. 'BASI_EDIT_AGENT/bg_v1/train/before/*.jpg'")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save mask PNGs, e.g. 'BASI_EDIT_AGENT/bg_v1/train/masks'")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(device)
    preprocess = build_preprocess()

    before_paths = sorted(glob.glob(args.before_glob))
    print(f"Found {len(before_paths)} images")

    for before_path in tqdm(before_paths):
        stem = os.path.splitext(os.path.basename(before_path))[0]
        out_path = os.path.join(args.output_dir, f"{stem}.png")

        if os.path.exists(out_path):
            continue  # already done

        img = Image.open(before_path).convert("RGB")
        mask = run_inference(model, preprocess, img, device)
        mask.save(out_path)

    print("Done. Masks written to", args.output_dir)


if __name__ == "__main__":
    main()

