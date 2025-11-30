"""
Shared utilities for training scripts (bg_residual, skin_residual, etc.)
"""

import glob
import os
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class BeforeAfterDataset(Dataset):
    """
    Generic dataset for paired before/after images.
    
    Args:
        before_glob: Glob pattern for "before" images
        after_glob: Glob pattern for "after" images
        max_side: Maximum side length for resizing (preserves aspect ratio)
        is_train: Whether this is training data (for potential augmentation)
    """
    def __init__(self, before_glob: str, after_glob: str, max_side: int = 1024, is_train: bool = True):
        self.before_paths = sorted(glob.glob(before_glob))
        self.after_paths = sorted(glob.glob(after_glob))
        
        assert len(self.before_paths) == len(self.after_paths), \
            f"len(before)={len(self.before_paths)} len(after)={len(self.after_paths)} mismatch"
        
        self.max_side = max_side
        self.is_train = is_train
    
    def __len__(self):
        return len(self.before_paths)
    
    def _resize(self, img: Image.Image, max_side: int) -> Image.Image:
        """Resize image so longest side equals max_side, preserving aspect ratio."""
        w, h = img.size
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)
        return img
    
    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor [C, H, W] in [0, 1] range."""
        tfm = T.ToTensor()
        return tfm(img)
    
    def __getitem__(self, idx):
        before_path = self.before_paths[idx]
        after_path = self.after_paths[idx]
        
        x_img = Image.open(before_path).convert("RGB")
        y_img = Image.open(after_path).convert("RGB")
        
        # Resize to max_side
        x_img = self._resize(x_img, self.max_side)
        y_img = self._resize(y_img, self.max_side)
        
        # Convert to tensors
        x = self._to_tensor(x_img)  # [3, H, W]
        y = self._to_tensor(y_img)  # [3, H, W]
        
        return x, y


def load_image(path: str) -> Image.Image:
    """Load and convert image to RGB."""
    return Image.open(path).convert("RGB")


def resize_with_aspect(img: Image.Image, max_side: int) -> Image.Image:
    """Resize image so longest side equals max_side, preserving aspect ratio."""
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
    return img

