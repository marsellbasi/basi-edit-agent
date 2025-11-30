"""
Shared Residual UNet model for bg_residual and skin_residual training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ResidualUNet(nn.Module):
    """
    Fully convolutional UNet-style network that predicts a residual image R.
    Usage: y_hat = clamp(x + R, 0, 1)
    
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


# Aliases for backward compatibility and clarity
BgResidualNet = ResidualUNet
SkinResidualNet = ResidualUNet


class BgResidualNetV2(ResidualUNet):
    """
    Enhanced BgResidualNet v2 for stronger background edits.
    
    Key improvements over v1:
    - Increased model capacity (base_ch=48 by default, vs 32 in v1)
    - Mask-weighted residual application: suppresses residuals on subject, allows larger residuals on background
    - Stronger residual scaling for background regions (0.6 vs 0.3 in v1)
    
    How v2's mask-weighted residuals work:
    - The model predicts a raw residual R
    - In background regions (mask=0): residual is scaled by bg_residual_scale (default 1.0 = full strength)
    - In subject regions (mask=1): residual is scaled by subj_residual_scale (default 0.1 = strongly suppressed)
    - This allows the model to learn larger background changes while protecting subject identity
    
    This model is specifically designed for Stage 2 background editing with subject masks.
    It should produce more visible background changes while preserving subject identity.
    
    Usage:
        model = BgResidualNetV2(in_ch=3, base_ch=48, use_mask=True)
        residual = model.forward_with_mask_weighted(x, mask)  # Use this for v2
        output = torch.clamp(x + residual, 0, 1)
    """
    def __init__(self, in_ch=3, base_ch=48, use_mask=True, bg_residual_scale=1.0, subj_residual_scale=0.1):
        """
        Args:
            in_ch: Input channels (3 for RGB)
            base_ch: Base number of channels (default 48 for v2, vs 32 for v1)
            use_mask: Whether to use mask channel (should be True for v2)
            bg_residual_scale: Scale factor for residuals in background regions (default 1.0 = full strength)
            subj_residual_scale: Scale factor for residuals in subject regions (default 0.1 = strongly suppressed)
        """
        super().__init__(in_ch=in_ch, base_ch=base_ch, use_mask=use_mask)
        self.bg_residual_scale = bg_residual_scale
        self.subj_residual_scale = subj_residual_scale
    
    def forward_with_mask_weighted(self, x, mask):
        """
        Forward pass with mask-weighted residual application.
        
        This is the key v2 feature: residuals are scaled differently based on mask.
        - Background regions (mask=0): get full residual strength
        - Subject regions (mask=1): get suppressed residual strength
        
        Args:
            x: [B, 3, H, W] RGB image
            mask: [B, 1, H, W] or [B, H, W] subject mask (1.0 on subject, 0.0 on background)
        
        Returns:
            residual: [B, 3, H, W] mask-weighted residual
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        
        # Get raw residual from model
        residual_raw = self.forward_with_mask(x, mask)  # [B, 3, H, W]
        
        # Create weighted mask: 1.0 on background, subj_residual_scale on subject
        # mask is [B, 1, H, W] with 1.0 on subject, 0.0 on background
        bg_mask = 1.0 - mask  # [B, 1, H, W], 1.0 on bg, 0.0 on subject
        subj_mask = mask  # [B, 1, H, W], 1.0 on subject, 0.0 on bg
        
        # Broadcast to 3 channels
        bg_mask_3ch = bg_mask.expand_as(residual_raw)  # [B, 3, H, W]
        subj_mask_3ch = subj_mask.expand_as(residual_raw)  # [B, 3, H, W]
        
        # Apply weighted scaling: background gets bg_residual_scale, subject gets subj_residual_scale
        residual_weighted = (
            residual_raw * bg_mask_3ch * self.bg_residual_scale +
            residual_raw * subj_mask_3ch * self.subj_residual_scale
        )
        
        return residual_weighted

