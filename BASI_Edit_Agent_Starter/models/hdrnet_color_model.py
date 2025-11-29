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


class BilateralSlice(nn.Module):
    """
    Bilateral grid slicing operation.
    
    Takes a bilateral grid of affine coefficients and slices it to full resolution
    using trilinear interpolation based on pixel coordinates and luminance.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        grid: torch.Tensor,
        guide: torch.Tensor,
        input_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Slice the bilateral grid to full resolution.
        
        Args:
            grid: Bilateral grid of shape [B, C, H_grid, W_grid, D_grid]
            guide: Guide image (luminance) of shape [B, 1, H, W] in [0, 1]
            input_image: Full-res input image [B, 3, H, W] for shape reference
        
        Returns:
            Sliced coefficients of shape [B, C, H, W]
        """
        B, C, H_grid, W_grid, D_grid = grid.shape
        _, _, H, W = input_image.shape
        
        # Normalize guide to grid depth coordinates [0, D_grid-1]
        guide_flat = guide.squeeze(1)  # [B, H, W]
        d_coords = guide_flat * (D_grid - 1)  # [B, H, W] in [0, D_grid-1]
        
        # Get depth indices for trilinear interpolation
        d_lo = torch.floor(d_coords).long().clamp(0, D_grid - 1)  # [B, H, W]
        d_hi = torch.clamp(d_lo + 1, max=D_grid - 1)  # [B, H, W]
        d_w = (d_coords - d_lo.float()).clamp(0, 1)  # [B, H, W]
        
        # Create spatial coordinate grids normalized to [-1, 1]
        y_coords = torch.linspace(-1, 1, H, device=grid.device, dtype=grid.dtype)
        x_coords = torch.linspace(-1, 1, W, device=grid.device, dtype=grid.dtype)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        y_grid = y_grid.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
        x_grid = x_grid.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
        
        # Stack for 2D grid_sample: [B, H, W, 2]
        coords_2d = torch.stack([y_grid, x_grid], dim=-1)  # [B, H, W, 2]
        
        # Sample from each depth level and interpolate
        output_slices = []
        for d_idx in [d_lo, d_hi]:
            # For each batch and pixel, get the depth slice
            # We need to sample per-pixel depth, so we'll use a loop or advanced indexing
            # More efficient: sample all depth slices and use advanced indexing
            
            # Sample each depth slice using 2D grid_sample
            depth_outputs = []
            for d in range(D_grid):
                depth_slice = grid[:, :, :, :, d]  # [B, C, H_grid, W_grid]
                sampled = F.grid_sample(
                    depth_slice,
                    coords_2d,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=True
                )  # [B, C, H, W]
                depth_outputs.append(sampled)
            
            # Stack all depth slices: [B, C, H, W, D_grid]
            all_depths = torch.stack(depth_outputs, dim=-1)
            
            # Use advanced indexing to select per-pixel depth
            # d_idx: [B, H, W], we need to index into last dimension
            batch_idx = torch.arange(B, device=grid.device).view(B, 1, 1).expand(-1, H, W)
            h_idx = torch.arange(H, device=grid.device).view(1, H, 1).expand(B, -1, W)
            w_idx = torch.arange(W, device=grid.device).view(1, 1, W).expand(B, H, -1)
            
            # Select: all_depths[b, c, h, w, d_idx[b, h, w]]
            # Reshape for indexing: [B, C, H*W, D_grid]
            all_depths_flat = all_depths.view(B, C, H * W, D_grid)
            d_idx_flat = d_idx.view(B, H * W)  # [B, H*W]
            
            # Use gather to select the right depth per pixel
            d_idx_expanded = d_idx_flat.unsqueeze(1).expand(-1, C, -1)  # [B, C, H*W]
            d_idx_expanded = d_idx_expanded.unsqueeze(-1)  # [B, C, H*W, 1]
            selected = torch.gather(all_depths_flat, dim=3, index=d_idx_expanded).squeeze(-1)  # [B, C, H*W]
            selected = selected.view(B, C, H, W)  # [B, C, H, W]
            
            output_slices.append(selected)
        
        # Interpolate between depth levels
        d_w_expanded = d_w.unsqueeze(1)  # [B, 1, H, W]
        output = (1 - d_w_expanded) * output_slices[0] + d_w_expanded * output_slices[1]
        
        return output


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
        coeffs_flat = F.interpolate(
            coeffs_flat,
            size=(H_grid, W_grid),
            mode='bilinear',
            align_corners=False
        )  # [B, grid_channels * D_grid, H_grid, W_grid]
        
        # Reshape to grid format: [B, grid_channels, H_grid, W_grid, D_grid]
        coeffs_grid = coeffs_flat.view(B, self.num_affine_params, D_grid, H_grid, W_grid)
        coeffs_grid = coeffs_grid.permute(0, 1, 3, 4, 2).contiguous()
        # Now: [B, grid_channels, H_grid, W_grid, D_grid]
        
        # Slice grid to full resolution
        # Output: [B, grid_channels, H, W]
        sliced_coeffs = self.bilateral_slice(coeffs_grid, guide, x)
        
        # Apply local affine transform
        # sliced_coeffs: [B, 12, H, W] where 12 = 3 channels * 4 params (3 weights + 1 bias)
        # For each pixel, we have a 3x4 affine matrix
        
        # Reshape for matrix multiplication
        # x: [B, 3, H, W] -> [B, 3, H*W]
        x_flat = x.view(B, 3, H * W)  # [B, 3, H*W]
        
        # Reshape coefficients: [B, 12, H, W] -> [B, 3, 4, H*W]
        coeffs_reshaped = sliced_coeffs.view(B, 3, 4, H * W)  # [B, 3, 4, H*W]
        
        # Extract weights and bias
        weights = coeffs_reshaped[:, :, :3, :]  # [B, 3, 3, H*W]
        bias = coeffs_reshaped[:, :, 3:4, :]  # [B, 3, 1, H*W]
        
        # Apply affine transform per pixel
        # weights: [B, 3, 3, H*W], x_flat: [B, 3, H*W]
        # We need to do batched matrix-vector multiplication
        # For each pixel: y = W @ x + b
        
        # Reshape for batched matmul
        weights = weights.permute(0, 3, 1, 2)  # [B, H*W, 3, 3]
        x_flat_t = x_flat.permute(0, 2, 1).unsqueeze(-1)  # [B, H*W, 3, 1]
        bias_t = bias.permute(0, 3, 1, 2)  # [B, H*W, 3, 1]
        
        # Batched matmul: [B, H*W, 3, 3] @ [B, H*W, 3, 1] -> [B, H*W, 3, 1]
        y_flat = torch.bmm(weights, x_flat_t) + bias_t  # [B, H*W, 3, 1]
        y_flat = y_flat.squeeze(-1)  # [B, H*W, 3]
        
        # Reshape back to image
        y = y_flat.permute(0, 2, 1).view(B, 3, H, W)  # [B, 3, H, W]
        
        # Clamp to valid range
        return y.clamp(0.0, 1.0)


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


# Simple test function
if __name__ == "__main__":
    # Test that the model works
    model = HDRNetColorModel()
    x = torch.rand(2, 3, 512, 512)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
    assert y.shape == x.shape, "Output shape should match input"
    assert y.min() >= 0.0 and y.max() <= 1.0, "Output should be in [0, 1]"
    print("✓ HDRNet model test passed!")

