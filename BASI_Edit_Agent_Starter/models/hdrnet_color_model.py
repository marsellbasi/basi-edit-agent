"""
HDRNet-style color/tone model for BASI Edit Agent.

This module implements a PyTorch-based HDRNet model following the architecture
from "Deep Bilateral Learning for Real-Time Image Enhancement" (SIGGRAPH 2017).

The model uses a bilateral grid to predict local affine color transforms,
enabling both global and local color/tone adjustments while preserving edges.
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_affine_coeffs(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Apply per-pixel 3x4 affine color transforms to an RGB image.

    Args:
        x:      [B, 3, H, W] tensor in [0, 1]
        coeffs: [B, 12, H, W] tensor, representing a 3x4 matrix per pixel
                flattened in row-major order:
                    [a00, a01, a02, a03,
                     a10, a11, a12, a13,
                     a20, a21, a22, a23]
                so that:
                    [r', g', b']^T = A @ [r, g, b, 1]^T

    Returns:
        y: [B, 3, H, W] tensor in [0, 1]
    """
    assert x.dim() == 4 and x.size(1) == 3, f"Expected x [B, 3, H, W], got {x.shape}"
    assert coeffs.dim() == 4 and coeffs.size(1) == 12, f"Expected coeffs [B, 12, H, W], got {coeffs.shape}"

    B, _, H, W = x.shape

    # [B, 12, H, W] -> [B, H, W, 12]
    coeffs_hw = coeffs.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 12]

    # [B, H, W, 12] -> [B, H*W, 3, 4]
    coeffs_mat = coeffs_hw.view(B, H * W, 3, 4)          # [B, HW, 3, 4]

    # Split into weights and bias
    weights = coeffs_mat[..., :3]  # [B, HW, 3, 3]
    bias    = coeffs_mat[..., 3]   # [B, HW, 3]

    # Flatten image to [B, HW, 3]
    x_flat = x.view(B, 3, H * W).permute(0, 2, 1)        # [B, HW, 3]

    # Apply affine: y = W * x + b
    x_expanded = x_flat.unsqueeze(-1)                    # [B, HW, 3, 1]
    y_expanded = torch.matmul(weights, x_expanded)       # [B, HW, 3, 1]
    y_flat = y_expanded.squeeze(-1) + bias               # [B, HW, 3]

    # Back to [B, 3, H, W]
    y = y_flat.permute(0, 2, 1).view(B, 3, H, W)

    # Clamp to valid range
    y = torch.clamp(y, 0.0, 1.0)

    return y


class BilateralSlice(nn.Module):
    """
    Bilateral slicing using torch.nn.functional.grid_sample.

    Grid layout: [B, C, Dg, Hg, Wg]
    Guide: [B, 1, H, W] in [0, 1]
    Output: [B, C, H, W]
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        grid: torch.Tensor,
        guide: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        B, C, Dg, Hg, Wg = grid.shape
        Bg, _, H, W = guide.shape
        assert B == Bg, f"Batch mismatch: grid B={B}, guide B={Bg}"
        assert output_size == (H, W), f"Expected output_size {(H, W)}, got {output_size}"

        device = grid.device
        dtype = grid.dtype

        # Normalize spatial coordinates to [-1, 1]
        # y: 0..Hg-1 -> -1..1, x: 0..Wg-1 -> -1..1
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]

        yy = yy.unsqueeze(0).expand(B, H, W)  # [B, H, W]
        xx = xx.unsqueeze(0).expand(B, H, W)  # [B, H, W]

        # Depth from guide: map [0,1] -> [-1,1]
        z = guide.squeeze(1).clamp(0.0, 1.0)  # [B, H, W]
        zz = 2.0 * z - 1.0                    # [-1, 1]

        # Stack into grid_sample coordinates
        # For 5D input [B, C, Dg, Hg, Wg], grid_sample expects grid [B, D_out, H_out, W_out, 3]
        # We need to add a D dimension (depth) to the grid
        sample_grid = torch.stack([xx, yy, zz], dim=-1)  # [B, H, W, 3]
        # Reshape to [B, 1, H, W, 3] for 5D grid_sample
        sample_grid = sample_grid.unsqueeze(1)  # [B, 1, H, W, 3]

        # grid_sample for 5D input: expects input [B, C, Dg, Hg, Wg] and grid [B, D_out, H_out, W_out, 3]
        # and returns [B, C, D_out, H_out, W_out]
        sliced = F.grid_sample(
            grid,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # [B, C, 1, H, W]

        # Squeeze the D dimension: [B, C, 1, H, W] -> [B, C, H, W]
        sliced = sliced.squeeze(2)  # [B, C, H, W]

        return sliced


class HDRNetColorModel(nn.Module):
    """
    HDRNet-style color/tone model for BASI Edit Agent.
    
    Architecture:
    1. Low-resolution encoder processes downsampled input
    2. Predicts a bilateral grid of affine coefficients
    3. Slices the grid to full resolution using guide image (luminance)
    4. Applies local affine color transform to full-res input
    
    Input: [B, 3, H, W] RGB tensor in [0, 1]
    Output: [B, 3, H, W] RGB tensor in [0, 1]
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        bilateral_grid_size: Tuple[int, int, int] = (16, 16, 8),
        lowres_channels: int = 16,
        hidden_dim: int = 64,
        num_affine_params: int = 12,  # 3x4 affine for RGB (3 channels * 4 params: 3 weights + 1 bias)
        input_downsample: int = 256,
    ):
        """
        Initialize HDRNet color model.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            bilateral_grid_size: Size of bilateral grid (H_grid, W_grid, D_grid)
            lowres_channels: Number of channels in low-res encoder
            hidden_dim: Hidden dimension for coefficient prediction network
            num_affine_params: Number of affine parameters per pixel (12 for 3x4)
            input_downsample: Target side length for low-res processing
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.grid_size = bilateral_grid_size
        self.lowres_channels = lowres_channels
        self.hidden_dim = hidden_dim
        self.num_affine_params = num_affine_params
        self.input_downsample = input_downsample
        
        H_grid, W_grid, D_grid = bilateral_grid_size
        grid_channels = num_affine_params
        
        # Low-resolution encoder
        # Simple CNN to process downsampled input
        self.lowres_encoder = nn.Sequential(
            nn.Conv2d(in_channels, lowres_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(lowres_channels, lowres_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(lowres_channels * 2, lowres_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Coefficient prediction network
        # Maps low-res features to bilateral grid
        encoder_out_channels = lowres_channels * 2
        self.coeff_predictor = nn.Sequential(
            nn.Conv2d(encoder_out_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, grid_channels * D_grid, kernel_size=3, padding=1),
        )
        
        # Bilateral slicing module
        self.bilateral_slice = BilateralSlice()
        
    def compute_luminance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute luminance guide image from RGB."""
        # Standard RGB to luminance weights
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        return y.clamp(0.0, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image [B, 3, H, W] in [0, 1]
        
        Returns:
            Output image [B, 3, H, W] in [0, 1]
        """
        # Sanity check input shape
        assert x.dim() == 4 and x.size(1) == 3, \
            f"Expected x [B, 3, H, W], got {x.shape}"
        
        B, C, H, W = x.shape
        
        # Compute luminance guide for bilateral slicing
        guide = self.compute_luminance(x)  # [B, 1, H, W]
        
        # Downsample input for low-res processing
        target_size = self.input_downsample
        scale = min(1.0, target_size / max(H, W))
        if scale < 1.0:
            H_low = int(H * scale)
            W_low = int(W * scale)
            x_low = F.interpolate(x, size=(H_low, W_low), mode='bilinear', align_corners=False)
        else:
            x_low = x
            H_low, W_low = H, W
        
        # Encode low-res input
        features = self.lowres_encoder(x_low)  # [B, lowres_channels*2, H_low, W_low]
        
        # Predict bilateral grid coefficients
        # Output: [B, grid_channels * D_grid, H_low, W_low]
        coeffs_flat = self.coeff_predictor(features)
        
        # Reshape to bilateral grid
        H_grid, W_grid, D_grid = self.grid_size
        # Resize to grid spatial dimensions
        # coeffs_flat: [B, num_affine_params * D_grid, H_grid, W_grid]
        coeffs_flat = F.interpolate(
            coeffs_flat,
            size=(H_grid, W_grid),
            mode="bilinear",
            align_corners=False,
        )
        
        # Reshape to [B, C, Dg, Hg, Wg]
        B_flat, cd, Hg, Wg = coeffs_flat.shape
        C = self.num_affine_params
        Dg = cd // C
        assert cd == C * Dg, f"Expected cd to be divisible by C, got cd={cd}, C={C}"
        
        coeffs_grid = coeffs_flat.view(B, C, Dg, Hg, Wg)  # [B, C, Dg, Hg, Wg]
        
        # Slice grid to full resolution
        # guide: [B, 1, H, W]
        sliced_coeffs = self.bilateral_slice(coeffs_grid, guide, output_size=(H, W))
        
        # Sanity check: sliced_coeffs should be [B, 12, H, W]
        assert sliced_coeffs.shape == (B, self.num_affine_params, H, W), \
            f"Expected sliced_coeffs shape {(B, self.num_affine_params, H, W)}, got {sliced_coeffs.shape}"
        
        # Apply affine coefficients to input image
        y = apply_affine_coeffs(x, sliced_coeffs)
        
        return y


def build_hdrnet_color_model_from_config(cfg: Dict) -> HDRNetColorModel:
    """
    Build HDRNet color model from configuration dictionary.
    
    Args:
        cfg: Configuration dict with keys:
            - bilateral_grid_size: [H, W, D] or tuple
            - lowres_channels: int
            - hidden_dim: int
            - num_affine_params: int (default 12)
            - input_downsample: int (default 256)
    
    Returns:
        HDRNetColorModel instance
    """
    grid_size = cfg.get("bilateral_grid_size", [16, 16, 8])
    if isinstance(grid_size, list):
        grid_size = tuple(grid_size)
    
    model = HDRNetColorModel(
        in_channels=3,
        bilateral_grid_size=grid_size,
        lowres_channels=cfg.get("lowres_channels", 16),
        hidden_dim=cfg.get("hidden_dim", 64),
        num_affine_params=cfg.get("num_affine_params", 12),
        input_downsample=cfg.get("input_downsample", 256),
    )
    
    return model


def _test_identity_affine():
    """Test that identity affine coefficients produce the input image."""
    B, H, W = 2, 32, 48
    x = torch.rand(B, 3, H, W)

    # Build identity affine: y = x
    coeffs = torch.zeros(B, 12, H, W)
    # Row 0: [1, 0, 0, 0] maps R -> R
    coeffs[:, 0, :, :] = 1.0  # a00
    # Row 1: [0, 1, 0, 0] maps G -> G
    coeffs[:, 5, :, :] = 1.0  # a11
    # Row 2: [0, 0, 1, 0] maps B -> B
    coeffs[:, 10, :, :] = 1.0 # a22

    y = apply_affine_coeffs(x, coeffs)
    max_diff = (x - y).abs().max().item()
    print("Max diff (identity test):", max_diff)


def _test_full_identity_pipeline():
    """Test that a constant identity grid + bilateral slice + apply_affine_coeffs reproduces x."""
    B, H, W = 2, 64, 80
    x = torch.rand(B, 3, H, W)

    # Make guide arbitrary (but valid)
    guide = torch.rand(B, 1, H, W)

    C = 12
    Dg, Hg, Wg = 4, 8, 8

    # Build a grid that is identity everywhere
    grid = torch.zeros(B, C, Dg, Hg, Wg)
    # Row 0: [1, 0, 0, 0]
    grid[:, 0, :, :, :] = 1.0  # a00
    # Row 1: [0, 1, 0, 0]
    grid[:, 5, :, :, :] = 1.0  # a11
    # Row 2: [0, 0, 1, 0]
    grid[:, 10, :, :, :] = 1.0 # a22

    slicer = BilateralSlice()
    coeffs = slicer(grid, guide, output_size=(H, W))  # [B, 12, H, W]

    y = apply_affine_coeffs(x, coeffs)

    max_diff = (x - y).abs().max().item()
    print("Max diff (full identity pipeline):", max_diff)


if __name__ == "__main__":
    _test_identity_affine()
    _test_full_identity_pipeline()

