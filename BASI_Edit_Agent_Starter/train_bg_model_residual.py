# scroll to "class BgDataset(Dataset):" and replace that whole class
# with the version above, then:
#   Ctrl+O, Enter to save
#   Ctrl+X to exit
import argparse, os, glob, subprocess
from PIL import Image

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
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
# GCS Backup
# -----------------------------
def backup_to_gcs(model_dir, gcs_backup_dir):
    """Copy model_dir to GCS using gsutil. Returns True on success, False on failure."""
    if not gcs_backup_dir:
        return False
    
    try:
        cmd = ["gsutil", "-m", "cp", "-r", model_dir, gcs_backup_dir]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Successfully backed up {model_dir} to {gcs_backup_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ GCS backup failed: {e.stderr.strip()}")
        return False
    except Exception as e:
        print(f"❌ GCS backup error: {str(e)}")
        return False


# -----------------------------
# Preview generation
# -----------------------------
def generate_previews(model, val_ds, device, output_dir, max_samples=6):
    """Generate triplet previews for first N validation pairs."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    count = 0
    with torch.no_grad():
        for i in range(min(max_samples, len(val_ds))):
            xb, yb = val_ds[i]
            xb = xb.unsqueeze(0).to(device)  # (1, 3, H, W)
            yb = yb.unsqueeze(0).to(device)
            
            residual = model(xb)
            residual = torch.tanh(residual) * 0.3
            y_hat = torch.clamp(xb + residual, 0.0, 1.0)
            
            # Convert to PIL
            before_img = TF.to_pil_image(xb[0].cpu())
            model_img = TF.to_pil_image(y_hat[0].cpu())
            after_img = TF.to_pil_image(yb[0].cpu())
            
            # Resize to same height
            target_h = max(before_img.height, model_img.height, after_img.height)
            if before_img.height != target_h:
                w = int(before_img.width * target_h / before_img.height)
                before_img = before_img.resize((w, target_h), Image.LANCZOS)
            if model_img.height != target_h:
                w = int(model_img.width * target_h / model_img.height)
                model_img = model_img.resize((w, target_h), Image.LANCZOS)
            if after_img.height != target_h:
                w = int(after_img.width * target_h / after_img.height)
                after_img = after_img.resize((w, target_h), Image.LANCZOS)
            
            # Create triplet
            total_w = before_img.width + model_img.width + after_img.width
            triplet = Image.new("RGB", (total_w, target_h))
            x = 0
            triplet.paste(before_img, (x, 0))
            x += before_img.width
            triplet.paste(model_img, (x, 0))
            x += model_img.width
            triplet.paste(after_img, (x, 0))
            
            # Save
            out_path = os.path.join(output_dir, f"{i:03d}_triplet.jpg")
            triplet.save(out_path, quality=95)
            count += 1
    
    return count


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

    def _center_crop(img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        img: (C, H, W)
        Returns a center-cropped tensor of shape (C, target_h, target_w).
        """
        _, h, w = img.shape
        top = max(0, (h - target_h) // 2)
        left = max(0, (w - target_w) // 2)
        return img[:, top:top + target_h, left:left + target_w]

    # custom collate: stack befores and afters into tensors
    def collate_fn(batch):
        """
        batch: list of (before_tensor, after_tensor)
        returns:
          xb: (B, 3, H, W)
          yb: (B, 3, H, W)
        We center-crop all images in the batch to the same (min_h, min_w)
        so torch.stack() will not fail on tiny mismatches like 426 vs 427.
        """
        befores, afters = zip(*batch)  # unzip

        # Compute global minimum H and W across both before and after tensors
        all_h = [b.shape[1] for b in befores] + [a.shape[1] for a in afters]
        all_w = [b.shape[2] for b in befores] + [a.shape[2] for a in afters]
        target_h = min(all_h)
        target_w = min(all_w)

        cropped_befores = [
            _center_crop(b.contiguous(), target_h, target_w) for b in befores
        ]
        cropped_afters = [
            _center_crop(a.contiguous(), target_h, target_w) for a in afters
        ]

        xb = torch.stack(cropped_befores, dim=0)
        yb = torch.stack(cropped_afters, dim=0)
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
    best_val = float("inf")
    start_epoch = 1

    # Handle resume logic
    if args.resume:
        if args.resume_ckpt:
            ckpt_path = args.resume_ckpt
        else:
            # Auto-resume from bg_residual_last.pt in model_dir
            ckpt_path = os.path.join(args.model_dir, "bg_residual_last.pt")
        
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 1) + 1
            best_val = ckpt.get("best_val", ckpt.get("val_l1", float("inf")))
            print(f"Resuming from epoch {start_epoch}, best_val={best_val:.4f}")
        else:
            if args.resume_ckpt:
                print(f"Warning: checkpoint {ckpt_path} not found, starting fresh")
            else:
                print(f"Warning: auto-resume checkpoint {ckpt_path} not found, starting fresh")

    identity_weight = args.identity_weight

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_total = 0.0
        running_bg = 0.0
        running_identity = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            residual = model(xb)
            # scale residual to be reasonably small at the start
            residual = torch.tanh(residual) * 0.3
            pred = torch.clamp(xb + residual, 0.0, 1.0)

            loss_bg = F.l1_loss(pred, yb)
            loss_identity = F.l1_loss(pred, xb)
            loss = loss_bg + identity_weight * loss_identity
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = xb.size(0)
            running_total += loss.item() * batch_size
            running_bg += loss_bg.item() * batch_size
            running_identity += loss_identity.item() * batch_size
            pbar.set_postfix(
                {"bg": f"{loss_bg.item():.4f}", "id": f"{loss_identity.item():.4f}"}
            )

        train_total = running_total / len(train_ds)
        train_bg = running_bg / len(train_ds)
        train_id = running_identity / len(train_ds)

        # --- validation ---
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                residual = model(xb)
                residual = torch.tanh(residual) * 0.3
                pred = torch.clamp(xb + residual, 0.0, 1.0)

                loss = F.l1_loss(pred, yb)
                val_running += loss.item() * xb.size(0)

        val_l1 = val_running / len(val_ds)
        print(
            f"Epoch {epoch} done | Train L1: {train_total:.4f} "
            f"(bg={train_bg:.4f}, id={train_id:.4f}) | Val L1: {val_l1:.4f}"
        )

        # Generate previews if enabled
        if args.preview_every > 0 and epoch % args.preview_every == 0:
            preview_dir = os.path.join(args.model_dir, "previews", f"epoch_{epoch:03d}")
            count = generate_previews(model, val_ds, device, preview_dir, max_samples=6)
            print(f"  Saved {count} preview triplets to {preview_dir}")

        # Save last checkpoint
        os.makedirs(args.model_dir, exist_ok=True)
        last_ckpt_path = os.path.join(args.model_dir, "bg_residual_last.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_l1": val_l1,
                "best_val": best_val,
            },
            last_ckpt_path,
        )

        if val_l1 < best_val:
            best_val = val_l1
            ckpt_path = os.path.join(args.model_dir, "bg_residual_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_l1": val_l1,
                    "best_val": best_val,
                },
                ckpt_path,
            )
            print(f"✅ New best val L1={val_l1:.4f} — checkpoint saved to {ckpt_path}")
            # Backup to GCS if enabled
            if args.gcs_backup_dir:
                backup_to_gcs(args.model_dir, args.gcs_backup_dir)

    # Final backup after training completes
    if args.gcs_backup_dir:
        print(f"\nBacking up final model directory to GCS...")
        backup_to_gcs(args.model_dir, args.gcs_backup_dir)


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

    p.add_argument("--resume", action="store_true",
                   help="Resume training from checkpoint")
    p.add_argument("--resume_ckpt", type=str, default=None,
                   help="Path to checkpoint file. If --resume is set but no path given, auto-resumes from bg_residual_last.pt in model_dir")
    p.add_argument("--preview_every", type=int, default=0,
                   help="Generate preview triplets every N epochs (0 to disable)")
    p.add_argument("--gcs_backup_dir", type=str, default="",
                   help="GCS path to backup model_dir (e.g., gs://bucket/path). Empty to disable.")
    p.add_argument(
        "--identity_weight",
        type=float,
        default=0.3,
        help="Weight for identity (input-preservation) loss term",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
