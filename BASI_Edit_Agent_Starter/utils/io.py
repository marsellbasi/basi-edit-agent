from pathlib import Path
from PIL import Image
import numpy as np

IMG_OK = {'.jpg','.jpeg','.png','.tif','.tiff'}

def load_image_rgb(path: Path):
    im = Image.open(path).convert("RGB")
    return np.array(im, dtype=np.float32) / 255.0

def save_image_rgb(arr, path: Path, quality=95):
    arr = (arr.clip(0,1) * 255.0 + 0.5).astype("uint8")
    im = Image.fromarray(arr, mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=quality)
