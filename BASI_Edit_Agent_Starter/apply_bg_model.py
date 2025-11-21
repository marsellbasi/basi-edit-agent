import argparse
import os
import glob

from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms


# -----------------------------
# Model (must match train_bg_model.py)
# -----------------------------

class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(x + y)


class BgCleanerNet(nn.Module):
    """
    Same architecture as in train_bg_model.py:
      output = x + delta(x)
    """

    def __init__(self, base_channels: int = 64, num_blocks: int = 8):
        super().__init__()
        self.entry = nn.Conv2d(3, base_channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            *[ResBlock(base_channels) for _ in range(num_blocks)]
        )
        self.exit = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.entry(x)
        feat = self.blocks(feat)
        delta = self.exit(feat)
        out = torch.clamp(x + delta, 0.0, 1.0)
        return out


# -----------------------------
# Utils
# -----------------------------

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()


def load_model(checkpoint_path: str, device: torch.device) -> BgCleanerNet:
    ckpt = torch.load(checkpoint_path, map_location=device)
    args_ckpt = ckpt.get("args", {})

    base_channels = args_ckpt.get("base_channels", 64)
    num_blocks = args_ckpt.get("num_blocks", 8)

    model = BgCleanerNet(
        base_channels=base_channels,
        num_blocks=num_blocks,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def make_triplet(input_img: Image.Image,
                 cleaned_img: Image.Image,
                 target_img: Image.Image) -> Image.Image:
    """
    Build [Color_v1 | BgCleaner | Final Edit] strip
    using a common HEIGHT but preserving each image's aspect ratio,
    so nothing gets squished.
    """
    # Use the cleaned image height as our reference
    _, h_clean = cleaned_img.size
    target_h = h_clean

    def resize_to_height(img: Image.Image, h: int) -> Image.Image:
        w, h0 = img.size
        new_w = int(round(w * (h / float(h0))))
        return img.resize((new_w, h), Image.BICUBIC)

    input_resized = resize_to_height(input_img, target_h)
    cleaned_resized = resize_to_height(cleaned_img, target_h)
    target_resized = resize_to_height(target_img, target_h)

    w1, _ = input_resized.size
    w2, _ = cleaned_resized.size
    w3, _ = target_resized.size

    total_w = w1 + w2 + w3
    trip = Image.new("RGB", (total_w, target_h))

    x = 0
    trip.paste(input_resized, (x, 0)); x += w1
    trip.paste(cleaned_resized, (x, 0)); x += w2
    trip.paste(target_resized, (x, 0))

    return trip

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Apply BASI Background Cleaner (Stage 2)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to bg_cleaner .pt checkpoint")
    parser.add_argument("--input_glob", type=str, required=True,
                        help="Glob pattern for input images (Color_v1 outputs)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save cleaned images")
    parser.add_argument("--target_dir", type=str, default=None,
                        help="Directory with final 'after' images (for triplets)")
    parser.add_argument("--triplets", action="store_true",
                        help="If set, also save [input | cleaned | target] triplets")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    triplet_dir = os.path.join(args.out_dir, "triplets")
    if args.triplets:
        os.makedirs(triplet_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = load_model(args.checkpoint, device)

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise RuntimeError(f"No files matched input_glob: {args.input_glob}")

    print(f"Found {len(files)} input images")

    for idx, in_path in enumerate(files, 1):
        fname = os.path.basename(in_path)
        print(f"[{idx}/{len(files)}] {fname}", flush=True)

        input_img = Image.open(in_path).convert("RGB")
        x = to_tensor(input_img).unsqueeze(0).to(device)

        with torch.no_grad():
            y = model(x)[0].cpu().clamp(0.0, 1.0)

        cleaned_img = to_pil(y)

        # Save cleaned output
        out_path = os.path.join(args.out_dir, fname)
        cleaned_img.save(out_path, quality=95)

        # Optional triplet
        if args.triplets and args.target_dir is not None:
            target_path = os.path.join(args.target_dir, fname)
            if os.path.exists(target_path):
                target_img = Image.open(target_path).convert("RGB")
                trip = make_triplet(input_img, cleaned_img, target_img)
                trip_path = os.path.join(triplet_dir, fname)
                trip.save(trip_path, quality=95)
            else:
                print(f"  [warn] No target found for {fname} in {args.target_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
