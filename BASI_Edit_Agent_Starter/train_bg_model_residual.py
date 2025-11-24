# scroll to "class BgDataset(Dataset):" and replace that whole class
# with the version above, then:
#   Ctrl+O, Enter to save
#   Ctrl+X to exit
import argparse, os, glob
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from tqdm import tqdm

# ...

def collate_pairs(batch):
    """
    batch: list of (before_tensor, after_tensor)
    returns:
      xb: (B, 3, H, W)
      yb: (B, 3, H, W)
    """
    befores, afters = zip(*batch)  # unzip list of tuples
    xb = torch.stack(befores, dim=0)
    yb = torch.stack(afters, dim=0)
    return xb, yb


# -----------------------------
# Dataset
# -----------------------------

class BgDataset(Dataset):
    """
    Background cleanup dataset.

    before_glob: pattern for 'before' images
    after_glob:  pattern for 'after' images
    """

    def __init__(self, before_glob, after_glob, max_side=1024, train=True):
        self.max_side = max_side
        self.train = train

        # Grab file lists
        self.before_paths = sorted(glob.glob(before_glob))
        self.after_paths = sorted(glob.glob(after_glob))

        if not self.before_paths:
            raise ValueError(f"No 'before' files found for pattern: {before_glob}")
        if not self.after_paths:
            raise ValueError(f"No 'after' files found for pattern: {after_glob}")

        # Pair by index after sorting
        n = min(len(self.before_paths), len(self.after_paths))
        if n == 0:
            raise ValueError("No overlapping before/after files (n=0).")

        if len(self.before_paths) != len(self.after_paths):
            print(
                f"[BgDataset] WARNING: len(before)={len(self.before_paths)} "
                f"len(after)={len(self.after_paths)} -> trimming to {n}"
            )
            self.before_paths = self.before_paths[:n]
            self.after_paths = self.after_paths[:n]

    def __len__(self):
        return len(self.before_paths)

    def _load_and_preprocess(self, path):
        img = Image.open(path).convert("RGB")
        w, h = img.size

        # Simple max-side resize
        scale = min(self.max_side / max(w, h), 1.0)
        if scale < 1.0:
            new_size = (int(round(w * scale)), int(round(h * scale)))
            img = img.resize(new_size, Image.BICUBIC)

        return T.ToTensor()(img)

    def __getitem__(self, idx):
        xb = self._load_and_preprocess(self.before_paths[idx])
        yb = self._load_and_preprocess(self.after_paths[idx])
        return xb, yb

# -----------------------------
# Small UNet-like residual model
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class BgResidualNet(nn.Module):
    """
    Fully convolutional, predicts a residual image R, then we use:
        y_hat = clamp(x + R, 0, 1)
    """
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()

        self.down1 = ConvBlock(in_ch, base_ch)
        self.down2 = ConvBlock(base_ch, base_ch * 2)
        self.down3 = ConvBlock(base_ch * 2, base_ch * 4)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 4)

        self.up3 = ConvBlock(base_ch * 4 + base_ch * 4, base_ch * 2)
        self.up2 = ConvBlock(base_ch * 2 + base_ch * 2, base_ch)
        self.up1 = ConvBlock(base_ch + base_ch, base_ch)

        self.out_conv = nn.Conv2d(base_ch, in_ch, 1)

    def _align_and_cat(self, up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Align the spatial size of `up` to match `skip` before concatenation.
        This prevents off-by-one H/W mismatches coming from upsampling.
        """
        if up.shape[2:] != skip.shape[2:]:
            up = F.interpolate(
                up,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        return torch.cat([up, skip], dim=1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        p1 = self.pool(d1)

        d2 = self.down2(p1)
        p2 = self.pool(d2)

        d3 = self.down3(p2)
        p3 = self.pool(d3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        u3 = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        u3 = self._align_and_cat(u3, d3)
        u3 = self.up3(u3)

        u2 = F.interpolate(u3, scale_factor=2, mode="bilinear", align_corners=False)
        u2 = self._align_and_cat(u2, d2)
        u2 = self.up2(u2)

        u1 = F.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False)
        u1 = self._align_and_cat(u1, d1)
        u1 = self.up1(u1)

        # Residual (unconstrained); we clamp after adding back to x
        residual = self.out_conv(u1)
        return residual

def collate_pairs(batch):
    """
    batch: list of (before_tensor, after_tensor)
    returns:
      xb: (B, 3, H, W)
      yb: (B, 3, H, W)
    """
    befores, afters = zip(*batch)  # unzip
    xb = torch.stack(befores, dim=0)
    yb = torch.stack(afters, dim=0)
    return xb, yb


# -----------------------------
# Training loop
# -----------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----- datasets -----
    train_ds = BgDataset(
        before_glob=args.train_before_glob,
        after_glob=args.train_after_glob,
        max_side=args.max_side,
        train=True,
    )
    val_ds = BgDataset(
        before_glob=args.val_before_glob,
        after_glob=args.val_after_glob,
        max_side=args.max_side,
        train=False,
    )

    # Debug: how many pairs we actually have
    print(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")

    # Create model

    # custom collate: stack befores and afters into tensors
    def collate_fn(batch):
        """
        batch: list of (before_tensor, after_tensor)
        Returns:
          xb: (B, 3, H, W)
          yb: (B, 3, H, W)

        We handle small spatial mismatches (e.g. 640x426 vs 640x427)
        by center-cropping every tensor in the batch down to the
        minimal height and width observed over the batch.
        """
        befores, afters = zip(*batch)  # unzip

        # Ensure contiguous tensors
        befores = [b.contiguous() for b in befores]
        afters = [a.contiguous() for a in afters]

        # Compute minimal H, W across before+after tensors
        min_h = min(t.shape[1] for t in (*befores, *afters))
        min_w = min(t.shape[2] for t in (*befores, *afters))

        def center_crop(t: torch.Tensor) -> torch.Tensor:
            _, h, w = t.shape
            top = max((h - min_h) // 2, 0)
            left = max((w - min_w) // 2, 0)
            return t[:, top:top + min_h, left:left + min_w]

        befores = [center_crop(b) for b in befores]
        afters = [center_crop(a) for a in afters]

        xb = torch.stack(befores, dim=0)
        yb = torch.stack(afters, dim=0)
        return xb, yb

    # ----- dataloaders -----
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,        # start with 0 to avoid worker weirdness
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ----- create model / optimizer exactly like you had before -----
    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    model = BgResidualNet(in_ch=3, base_ch=32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()

            residual = model(xb)
            # scale residual to be reasonably small at the start
            residual = torch.tanh(residual) * 0.3
            y_hat = torch.clamp(xb + residual, 0.0, 1.0)

            loss = F.l1_loss(y_hat, yb)
            loss.backward()
            opt.step()

            running += loss.item() * xb.size(0)
            pbar.set_postfix({"L1": f"{loss.item():.4f}"})

        train_l1 = running / len(train_ds)

        # --- validation ---
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                residual = model(xb)
                residual = torch.tanh(residual) * 0.3
                y_hat = torch.clamp(xb + residual, 0.0, 1.0)

                loss = F.l1_loss(y_hat, yb)
                val_running += loss.item() * xb.size(0)

        val_l1 = val_running / len(val_ds)
        print(f"Epoch {epoch} done | Train L1: {train_l1:.4f} | Val L1: {val_l1:.4f}")

        if val_l1 < best_val:
            best_val = val_l1
            torch.save(model.state_dict(), best_path)
            print(f"👍 New best checkpoint saved to {best_path} (val L1 {best_val:.4f})")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_side", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--model_dir", type=str, default="BASI_ARCHIVE/models/bg_v2")

    p.add_argument("--train_before_glob", type=str,
                   default="BASI_EDIT_AGENT/bg_v1/train/before/*.jpg")
    p.add_argument("--train_after_glob", type=str,
                   default="BASI_EDIT_AGENT/bg_v1/train/after/*.jpg")
    p.add_argument("--val_before_glob", type=str,
                   default="BASI_EDIT_AGENT/bg_v1/val/before/*.jpg")
    p.add_argument("--val_after_glob", type=str,
                   default="BASI_EDIT_AGENT/bg_v1/val/after/*.jpg")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
