"""
U-Net-based color/tone model for BASI Edit Agent.

This module implements a lightweight U-Net that predicts residual color adjustments,
providing an alternative to the HDRNet and baseline models for Stage 1 color grading.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block with optional batch normalization."""
    
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetColorModel(nn.Module):
    """
    Lightweight U-Net for predicting color/tone residuals.
    
    Architecture:
    - Encoder: Downsampling via stride-2 convolutions
    - Bottleneck: Feature processing
    - Decoder: Upsampling with skip connections
    - Output: 3-channel residual to add to input
    
    Input: [B, 3, H, W] RGB tensor in [0, 1]
    Output: [B, 3, H, W] RGB tensor in [0, 1]
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_down: int = 4,
        use_batchnorm: bool = True,
        residual: bool = True,
    ):
        """
        Initialize U-Net color model.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            base_channels: Base number of channels (doubles at each downsampling)
            num_down: Number of downsampling levels
            use_batchnorm: Whether to use batch normalization
            residual: If True, model predicts residual; if False, predicts output directly
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_down = num_down
        self.residual = residual
        
        # Encoder
        self.encoder = nn.ModuleList()
        in_ch = in_channels
        out_ch = base_channels
        
        for i in range(num_down):
            self.encoder.append(DoubleConv(in_ch, out_ch, use_batchnorm))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 256)  # Cap at 256 channels
        
        # Bottleneck
        self.bottleneck = DoubleConv(in_ch, in_ch, use_batchnorm)
        
        # Track channels for decoder (reverse of encoder)
        encoder_channels = []
        in_ch = in_channels
        out_ch = base_channels
        for i in range(num_down):
            encoder_channels.append(out_ch)
            in_ch = out_ch
            out_ch = min(out_ch * 2, 256)
        
        # Decoder (reverse order of encoder)
        self.decoder = nn.ModuleList()
        decoder_in_ch = in_ch  # Start from bottleneck output channels
        
        for i in range(num_down - 1, -1, -1):
            skip_ch = encoder_channels[i]  # Skip connection channels
            decoder_out_ch = encoder_channels[i]  # Output channels after upsampling
            
            # Upsample: [B, decoder_in_ch, H, W] -> [B, decoder_out_ch, 2H, 2W]
            self.decoder.append(nn.ConvTranspose2d(decoder_in_ch, decoder_out_ch, kernel_size=2, stride=2))
            # DoubleConv after concatenation with skip: [B, decoder_out_ch + skip_ch, 2H, 2W] -> [B, decoder_out_ch, 2H, 2W]
            self.decoder.append(DoubleConv(decoder_out_ch + skip_ch, decoder_out_ch, use_batchnorm))
            
            decoder_in_ch = decoder_out_ch
        
        # Final output layer
        self.final_conv = nn.Conv2d(decoder_out_ch, 3, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image [B, 3, H, W] in [0, 1]
        
        Returns:
            Output image [B, 3, H, W] in [0, 1]
        """
        # Encoder with skip connections
        skip_connections = []
        encoder_out = x
        
        for i, encoder_block in enumerate(self.encoder):
            encoder_out = encoder_block(encoder_out)
            skip_connections.append(encoder_out)
            # Downsample (except for last encoder block)
            if i < len(self.encoder) - 1:
                encoder_out = F.max_pool2d(encoder_out, kernel_size=2, stride=2)
        
        # Bottleneck
        bottleneck_out = self.bottleneck(encoder_out)
        
        # Decoder with skip connections
        decoder_out = bottleneck_out
        skip_idx = len(skip_connections) - 1
        
        for i in range(0, len(self.decoder), 2):
            # Upsample
            upsample = self.decoder[i]
            decoder_out = upsample(decoder_out)
            
            # Concatenate skip connection
            if skip_idx >= 0:
                skip = skip_connections[skip_idx]
                # Handle size mismatch (can happen due to odd dimensions)
                if decoder_out.shape[2:] != skip.shape[2:]:
                    decoder_out = F.interpolate(decoder_out, size=skip.shape[2:], mode='bilinear', align_corners=False)
                decoder_out = torch.cat([decoder_out, skip], dim=1)
                skip_idx -= 1
            
            # DoubleConv
            double_conv = self.decoder[i + 1]
            decoder_out = double_conv(decoder_out)
        
        # Final output
        residual = self.final_conv(decoder_out)
        
        # Handle size mismatch with input
        if residual.shape[2:] != x.shape[2:]:
            residual = F.interpolate(residual, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Add residual to input
        if self.residual:
            y = x + residual
        else:
            y = residual
        
        return torch.clamp(y, 0.0, 1.0)


def build_unet_color_model_from_config(cfg: Dict[str, Any]) -> nn.Module:
    """
    Build U-Net color model from configuration dictionary.
    
    Args:
        cfg: Configuration dict with keys:
            - base_channels: int (default 32)
            - num_down: int (default 4)
            - use_batchnorm: bool (default True)
            - residual: bool (default True)
    
    Returns:
        UNetColorModel instance
    """
    unet_cfg = cfg.get("color_model", {}).get("unet", {})
    
    model = UNetColorModel(
        in_channels=3,
        base_channels=unet_cfg.get("base_channels", 32),
        num_down=unet_cfg.get("num_down", 4),
        use_batchnorm=unet_cfg.get("use_batchnorm", True),
        residual=unet_cfg.get("residual", True),
    )
    
    return model


# Simple test function
if __name__ == "__main__":
    # Test that the model works
    model = UNetColorModel()
    x = torch.rand(2, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
    assert y.shape == x.shape, "Output shape should match input"
    assert y.min() >= 0.0 and y.max() <= 1.0, "Output should be in [0, 1]"
    print("✓ U-Net model test passed!")

