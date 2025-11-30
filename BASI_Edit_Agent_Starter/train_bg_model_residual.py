import argparse, os, glob, subprocess, sys
import yaml
from PIL import Image

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from tqdm import tqdm

from models.residual_unet import BgResidualNet, BgResidualNetV2

# ...

# -----------------------------
# Dataset
# -----------------------------

class BgDataset(Dataset):
    def __init__(self, before_glob, after_glob, max_side=640, is_train=True, use_mask=False):
        self.before_paths = sorted(glob.glob(before_glob))
        self.after_paths = sorted(glob.glob(after_glob))

        assert len(self.before_paths) == len(self.after_paths), \
            f"len(before)={len(self.before_paths)} len(after)={len(self.after_paths)} mismatch"

        self.max_side = max_side
        self.is_train = is_train
        self.use_mask = use_mask

    def __len__(self):
        return len(self.before_paths)

    def _load_mask(self, before_path: str):
        """
        Load a subject mask PNG if it exists.
        We expect something like:

        before: BASI_EDIT_AGENT/bg_v1/train/before/abc.jpg
        mask:   BASI_EDIT_AGENT/bg_v1/train/masks/abc.png
        """
        mask_path = before_path.replace("/before/", "/masks/")
        mask_path = os.path.splitext(mask_path)[0] + ".png"
        if os.path.exists(mask_path):
            m = Image.open(mask_path).convert("L")  # grayscale
            return m
        return None

    def _resize(self, img: Image.Image, max_side: int):
        w, h = img.size
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)
        return img

    def _to_tensor(self, img: Image.Image):
        import torchvision.transforms as T
        tfm = T.ToTensor()  # [0,1], CxHxW
        return tfm(img)

    def __getitem__(self, idx):
        before_path = self.before_paths[idx]
        after_path = self.after_paths[idx]

        x_img = Image.open(before_path).convert("RGB")
        y_img = Image.open(after_path).convert("RGB")
        m_img = self._load_mask(before_path)

        # If use_mask is True, require mask to exist
        if self.use_mask and m_img is None:
            raise RuntimeError(
                f"use_mask=True but mask not found for: {before_path}\n"
                f"Expected mask at: {before_path.replace('/before/', '/masks/').replace('.jpg', '.png')}"
            )

        # Basic resize to max_side for all
        x_img = self._resize(x_img, self.max_side)
        y_img = self._resize(y_img, self.max_side)
        if m_img is not None:
            m_img = m_img.resize(x_img.size, Image.NEAREST)

        # TODO: if you already have more advanced data augmentation in this file
        # (random crops, flips, etc.), make sure to apply the SAME ops to x_img,
        # y_img, and m_img here. For now we keep it simple and only resize.

        x = self._to_tensor(x_img)  # [3,H,W]
        y = self._to_tensor(y_img)  # [3,H,W]

        if m_img is not None:
            import numpy as np
            # Normalize mask to [0, 1] range (values may be 0-255)
            m_np = np.array(m_img).astype("float32")
            if m_np.max() > 1.0:
                m_np = m_np / 255.0
            m = torch.from_numpy(m_np)  # [H,W], values in [0, 1]
        else:
            m = None

        return x, y, m

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
    
    Supports optional mask channel: if use_mask=True, expects 4-channel input [R, G, B, mask].
    If use_mask=False, expects 3-channel RGB input.
    """
    def __init__(self, in_ch=3, base_ch=32, use_mask=False):
        super().__init__()
        self.use_mask = use_mask
        # If using mask, input is 4 channels (RGB + mask), otherwise 3
        actual_in_ch = 4 if use_mask else in_ch

        self.down1 = ConvBlock(actual_in_ch, base_ch)
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
    
    def forward_with_mask(self, x, mask):
        """
        Forward pass with mask channel concatenated.
        x: [B, 3, H, W] RGB image
        mask: [B, 1, H, W] or [B, H, W] subject mask (1.0 on subject, 0.0 on background)
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        x_with_mask = torch.cat([x, mask], dim=1)  # [B, 4, H, W]
        return self.forward(x_with_mask)

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
    
    # ----- Logging startup info -----
    print("=" * 60)
    print("BASI Background Residual Model Training (Stage 2)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dataset version: {args.dataset_version}")
    print(f"Train before glob: {args.train_before_glob}")
    print(f"Train after glob: {args.train_after_glob}")
    print(f"Val before glob: {args.val_before_glob}")
    print(f"Val after glob: {args.val_after_glob}")
    print(f"Max side: {args.max_side}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Identity weight: {args.identity_weight}")
    print(f"Model directory: {args.model_dir}")
    print(f"Use mask: {args.use_mask}")
    print("=" * 60)

    # ----- datasets -----
    train_ds = BgDataset(
        before_glob=args.train_before_glob,
        after_glob=args.train_after_glob,
        max_side=args.max_side,
        is_train=True,
        use_mask=args.use_mask,
    )
    val_ds = BgDataset(
        before_glob=args.val_before_glob,
        after_glob=args.val_after_glob,
        max_side=args.max_side,
        is_train=False,
        use_mask=args.use_mask,
    )

    print(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")
    if args.use_mask:
        print("Mask support: Enabled (masks required for loss computation)")
    else:
        print("Mask support: Disabled (using fallback loss computation)")
    
    # Determine whether we should use masks in the model
    has_masks = bool(args.use_mask)
    
    # Determine model version (from args, set in parse_args)
    bg_model_version = args.bg_model_version
    print(f"BG Model Version: {bg_model_version}")

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

    # custom collate: stack befores, afters, and masks into tensors
    def collate_fn(batch):
        """
        batch: list of (before_tensor, after_tensor, mask_tensor or None)
        returns:
          xb: (B, 3, H, W)
          yb: (B, 3, H, W)
          mb: (B, H, W) or None
        We center-crop all images in the batch to the same (min_h, min_w)
        so torch.stack() will not fail on tiny mismatches like 426 vs 427.
        """
        items = list(zip(*batch))  # unzip: (befores, afters, masks)
        befores = items[0]
        afters = items[1]
        masks = items[2] if len(items) > 2 else [None] * len(befores)

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

        # Handle masks: crop and stack if they exist
        mb_list = []
        has_masks = any(m is not None for m in masks)
        if has_masks:
            for m in masks:
                if m is not None:
                    m_cropped = _center_crop(m.unsqueeze(0).contiguous(), target_h, target_w).squeeze(0)
                    mb_list.append(m_cropped)
                else:
                    # Create a dummy mask (all ones) if missing
                    mb_list.append(torch.ones(target_h, target_w, dtype=torch.float32))
            mb = torch.stack(mb_list, dim=0)
        else:
            mb = None

        return xb, yb, mb

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

    # ----- create model / optimizer -----
    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    # Select model version: v1 (original) or v2 (stronger background edits with mask-weighted residuals)
    if bg_model_version == "v2":
        # v2: Enhanced model with mask-weighted residual application
        base_ch = getattr(args, 'base_ch', 48)
        bg_residual_scale = getattr(args, 'bg_residual_scale', 1.0)
        subj_residual_scale = getattr(args, 'subj_residual_scale', 0.1)
        
        if not args.use_mask:
            print("Warning: bg_model_version=v2 requires use_mask=True. Enabling mask support.")
            args.use_mask = True
        
        model = BgResidualNetV2(
            in_ch=3,
            base_ch=base_ch,
            use_mask=True,  # v2 always uses masks
            bg_residual_scale=bg_residual_scale,
            subj_residual_scale=subj_residual_scale
        ).to(device)
        print(f"[train_bg_model_residual] Using BgResidualNetV2: base_ch={base_ch}, "
              f"bg_scale={bg_residual_scale}, subj_scale={subj_residual_scale}")
    else:
        # v1: Original model
        model = BgResidualNet(in_ch=3, base_ch=32, use_mask=args.use_mask and has_masks).to(device)
        print(f"[train_bg_model_residual] Using BgResidualNet v1")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Checkpoints will be saved to: {args.model_dir}")
    print("=" * 60)

    # Cast config values to correct types (YAML may read them as strings)
    lr = float(args.lr)
    weight_decay = float(args.weight_decay)
    print(f"[train_bg_model_residual] Using lr={lr} weight_decay={weight_decay}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_val = float("inf")
    start_epoch = 1

    # Handle resume logic
    if args.resume:
        if args.resume_ckpt:
            ckpt_path = args.resume_ckpt
        else:
            # Auto-resume from latest.pt in model_dir
            ckpt_path = os.path.join(args.model_dir, "latest.pt")
        
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
    
    print()  # Empty line before training starts

    identity_weight = args.identity_weight
    print(f"[train_bg_model_residual] identity_weight={identity_weight}")
    print(f"[train_bg_model_residual] use_mask={args.use_mask}")
    print(f"[train_bg_model_residual] bg_model_version={bg_model_version}")

    def compute_loss_with_mask(pred, target_bg, target_identity, mask, identity_weight):
        """
        Compute loss using subject masks.
        
        pred: [B,3,H,W] - model prediction
        target_bg: [B,3,H,W] - target for background (after image)
        target_identity: [B,3,H,W] - target for identity (before image)
        mask: [B,H,W] or [B,1,H,W] - subject mask (1 = subject, 0 = background)
        identity_weight: float - weight for identity loss
        
        Returns:
            loss: scalar tensor
            loss_bg: scalar tensor (background L1)
            loss_identity: scalar tensor (identity L1)
        """
        # Ensure mask is [B,1,H,W]
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]
        
        # Broadcast mask to 3 channels: [B,1,H,W] -> [B,3,H,W]
        mask_3ch = mask.expand_as(pred)
        
        # Background mask: 1 on background, 0 on subject
        bg_mask = 1.0 - mask_3ch  # [B,3,H,W]
        
        # Subject mask: 1 on subject, 0 on background
        subj_mask = mask_3ch  # [B,3,H,W]
        
        # Background loss: L1 only on background pixels
        bg_diff = bg_mask * torch.abs(pred - target_bg)
        bg_mask_sum = bg_mask.sum() + 1e-8  # avoid division by zero
        loss_bg = bg_diff.sum() / bg_mask_sum
        
        # Identity loss: L1 only on subject pixels
        id_diff = subj_mask * torch.abs(pred - target_identity)
        subj_mask_sum = subj_mask.sum() + 1e-8  # avoid division by zero
        loss_identity = id_diff.sum() / subj_mask_sum
        
        # Total loss
        loss = loss_bg + identity_weight * loss_identity
        
        return loss, loss_bg, loss_identity

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_total = 0.0
        running_bg = 0.0
        running_identity = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for xb, yb, mb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            # Forward pass: use mask if model supports it
            if bg_model_version == "v2" and mb is not None:
                # v2: Use mask-weighted residual application
                mb = mb.to(device)
                residual = model.forward_with_mask_weighted(xb, mb)
                # v2: No tanh scaling - let the model learn larger residuals, mask weighting handles suppression
                # Apply a gentle tanh to prevent extreme values, but allow larger range
                residual = torch.tanh(residual) * 0.6  # v2: 0.6 vs v1's 0.3 for stronger residuals
            elif model.use_mask and mb is not None:
                # v1: Standard mask channel input
                mb = mb.to(device)
                residual = model.forward_with_mask(xb, mb)
                # v1: Scale residual to be reasonably small
                residual = torch.tanh(residual) * 0.3
            else:
                # No mask: standard forward pass
                residual = model(xb)
                residual = torch.tanh(residual) * 0.3
            
            pred = torch.clamp(xb + residual, 0.0, 1.0)

            # Compute loss
            if args.use_mask and mb is not None:
                # Use mask-based loss computation
                mb = mb.to(device)
                loss, loss_bg, loss_identity = compute_loss_with_mask(
                    pred, yb, xb, mb, identity_weight
                )
            else:
                # Fallback: old behavior when use_mask is False
                # Treat whole image as both subject & bg
                bg_mask = torch.ones_like(xb[:, :1, :, :])
                subject_mask = torch.ones_like(xb[:, :1, :, :])
                
                # Background should match the edited AFTER image
                bg_diff = bg_mask * torch.abs(pred - yb)
                loss_bg = bg_diff.sum() / (bg_mask.sum() * pred.shape[1] + 1e-8)
                
                # Subject should stay close to the original BEFORE image
                id_diff = subject_mask * torch.abs(pred - xb)
                loss_identity = id_diff.sum() / (subject_mask.sum() * pred.shape[1] + 1e-8)
                
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
            for xb, yb, mb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                # Forward pass: use mask if model supports it
                if bg_model_version == "v2" and mb is not None:
                    # v2: Use mask-weighted residual application
                    mb = mb.to(device)
                    residual = model.forward_with_mask_weighted(xb, mb)
                    residual = torch.tanh(residual) * 0.6  # v2: stronger residuals
                elif model.use_mask and mb is not None:
                    # v1: Standard mask channel input
                    mb = mb.to(device)
                    residual = model.forward_with_mask(xb, mb)
                    residual = torch.tanh(residual) * 0.3
                else:
                    # No mask: standard forward pass
                    residual = model(xb)
                    residual = torch.tanh(residual) * 0.3
                
                pred = torch.clamp(xb + residual, 0.0, 1.0)

                # Compute validation loss (background L1 only)
                if args.use_mask and mb is not None:
                    mb = mb.to(device)
                    # For validation, we only compute background L1
                    if mb.dim() == 3:
                        mb = mb.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]
                    bg_mask = (1.0 - mb).expand_as(pred)  # [B,3,H,W]
                    bg_diff = bg_mask * torch.abs(pred - yb)
                    bg_mask_sum = bg_mask.sum() + 1e-8
                    loss = bg_diff.sum() / bg_mask_sum
                else:
                    # Fallback: old behavior
                    bg_mask = torch.ones_like(xb[:, :1, :, :])
                    bg_diff = bg_mask * torch.abs(pred - yb)
                    loss = bg_diff.sum() / (bg_mask.sum() * pred.shape[1] + 1e-8)
                
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
        last_ckpt_path = os.path.join(args.model_dir, "latest.pt")
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
        
        # Also save as epoch checkpoint
        epoch_ckpt_path = os.path.join(args.model_dir, f"epoch_{epoch:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_l1": val_l1,
                "best_val": best_val,
            },
            epoch_ckpt_path,
        )

        if val_l1 < best_val:
            best_val = val_l1
            ckpt_path = os.path.join(args.model_dir, "best.pt")
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
    p = argparse.ArgumentParser(description="Train BASI Background Residual Model (Stage 2)")
    
    p.add_argument("--config", type=str, default="config.yaml",
                   help="Path to config.yaml file")
    p.add_argument("--dataset_version", type=str, default=None,
                   help="Dataset version (overrides config)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Number of epochs (overrides config)")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Batch size (overrides config)")
    p.add_argument("--max_side", type=int, default=None,
                   help="Max side length (overrides config)")
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate (overrides config)")
    p.add_argument("--weight_decay", type=float, default=None,
                   help="Weight decay (overrides config)")
    p.add_argument("--model_dir", type=str, default=None,
                   help="Model directory (overrides config)")
    p.add_argument("--identity_weight", type=float, default=None,
                   help="Identity weight (overrides config)")

    p.add_argument("--train_before_glob", type=str, default=None,
                   help="Glob for training before images (overrides dataset_version)")
    p.add_argument("--train_after_glob", type=str, default=None,
                   help="Glob for training after images (overrides dataset_version)")
    p.add_argument("--val_before_glob", type=str, default=None,
                   help="Glob for validation before images (overrides dataset_version)")
    p.add_argument("--val_after_glob", type=str, default=None,
                   help="Glob for validation after images (overrides dataset_version)")

    p.add_argument("--resume", action="store_true",
                   help="Resume training from checkpoint")
    p.add_argument("--resume_ckpt", type=str, default=None,
                   help="Path to checkpoint file. If --resume is set but no path given, auto-resumes from bg_residual_last.pt in model_dir")
    p.add_argument("--preview_every", type=int, default=0,
                   help="Generate preview triplets every N epochs (0 to disable)")
    p.add_argument("--gcs_backup_dir", type=str, default="",
                   help="GCS path to backup model_dir (e.g., gs://bucket/path). Empty to disable.")
    p.add_argument("--use_mask", action="store_true",
                   help="Use subject mask for loss computation (masks required)")

    args = p.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    bg_cfg = config.get("training", {}).get("bg_residual", {})
    dataset_root = config.get("dataset_root", "BASI_EDIT_AGENT")
    
    # Set defaults from config if not provided via CLI
    if args.dataset_version is None:
        args.dataset_version = bg_cfg.get("dataset_version", "bg_v1")
    if args.epochs is None:
        args.epochs = bg_cfg.get("epochs", 20)
    if args.batch_size is None:
        args.batch_size = bg_cfg.get("batch_size", 2)
    if args.max_side is None:
        args.max_side = bg_cfg.get("max_side", 1024)
    if args.lr is None:
        args.lr = bg_cfg.get("lr", 1e-4)
    if args.weight_decay is None:
        args.weight_decay = bg_cfg.get("weight_decay", 0.0)
    if args.model_dir is None:
        args.model_dir = bg_cfg.get("model_dir", "checkpoints/bg_residual")
    if args.identity_weight is None:
        args.identity_weight = bg_cfg.get("identity_weight", 0.3)
    # use_mask: If CLI flag was set (--use_mask in sys.argv), it's True. Otherwise use config value.
    if "--use_mask" not in sys.argv:
        args.use_mask = bg_cfg.get("use_mask", False)
    
    # bg_model_version: v1 (original) or v2 (stronger background edits with mask-weighted residuals)
    args.bg_model_version = bg_cfg.get("bg_model_version", "v1")
    # v2-specific settings
    args.base_ch = bg_cfg.get("base_ch", 48)
    args.bg_residual_scale = bg_cfg.get("bg_residual_scale", 1.0)
    args.subj_residual_scale = bg_cfg.get("subj_residual_scale", 0.1)
    
    # Build globs from dataset_version if not provided
    if args.train_before_glob is None:
        args.train_before_glob = os.path.join(dataset_root, args.dataset_version, "train", "before", "*.jpg")
    if args.train_after_glob is None:
        args.train_after_glob = os.path.join(dataset_root, args.dataset_version, "train", "after", "*.jpg")
    if args.val_before_glob is None:
        args.val_before_glob = os.path.join(dataset_root, args.dataset_version, "val", "before", "*.jpg")
    if args.val_after_glob is None:
        args.val_after_glob = os.path.join(dataset_root, args.dataset_version, "val", "after", "*.jpg")
    
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
