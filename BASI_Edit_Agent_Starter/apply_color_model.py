import argparse, json
from pathlib import Path
import yaml, numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm

IMG_OK = {'.jpg','.jpeg','.png','.tif','.tiff'}

def load_model(model_dir: Path):
    with open(model_dir/"color_model.json","r") as f:
        ckpt = json.load(f)
    import numpy as np
    M = np.array(ckpt["M"], dtype=np.float32)
    b = np.array(ckpt["b"], dtype=np.float32)
    diffs = np.array(ckpt["curve_diffs"], dtype=np.float32)
    # reconstruct curves (softplus inverse at inference)
    curves = np.concatenate([np.zeros((3,1), dtype=np.float32), np.log1p(np.exp(diffs))], axis=1).cumsum(axis=1)
    curves = curves / np.maximum(curves[:,-1:], 1e-6)
    return M, b, curves

def apply_transform(arr, M, b, curves):
    x = arr.astype(np.float32)/255.0
    y = x @ M.T + b.reshape(1,1,3)
    y = np.clip(y, 0, 1)
    y_idx = (y*255.0).clip(0,255-1e-6)
    lo = np.floor(y_idx).astype(np.int32); hi = np.minimum(lo+1, 255); w = y_idx - lo
    out = np.empty_like(y)
    for ch in range(3):
        c = curves[ch]
        y_lo = c[lo[...,ch]]; y_hi = c[hi[...,ch]]
        out[...,ch] = (1-w[...,ch])*y_lo + w[...,ch]*y_hi
    out = (out*255.0 + 0.5).astype(np.uint8)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input_glob", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    with open(args.config,'r') as f: cfg = yaml.safe_load(f)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    M,b,curves = load_model(Path(args.model_dir))

    files = [Path(p) for p in glob(args.input_glob, recursive=True)]
    files = [p for p in files if p.suffix.lower() in IMG_OK]
    for src in tqdm(files, desc="Applying"):
        try:
            im = Image.open(src).convert("RGB")
            arr = np.array(im)
            y = apply_transform(arr, M, b, curves)
            (out_dir / src.name).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(y).save(out_dir / src.name, quality=95)
        except Exception as e:
            print("Failed:", src, e)

if __name__ == "__main__":
    main()
