import argparse, os, re, shutil, csv, sys, random
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
from PIL import Image
from tqdm import tqdm

# Extensions we consider images for pairing; RAW included but resizing may be skipped for them.
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp', '.cr2', '.cr3', '.nef', '.raf', '.arw', '.dng'}

NORMALIZE_SUFFIXES = [
    r'_edit$', r'-edit$', r'_after$', r'-after$', r'_final$', r'-final$',
    r'_retouch$', r'-retouch$', r'_color$', r'-color$', r'_graded$', r'-graded$',
    r'_v\d+$', r'-v\d+$'
]

def load_config(p):
    with open(p, 'r') as f:
        return yaml.safe_load(f)

def list_files(root: Path, pattern: str) -> List[Path]:
    return [p for p in root.glob(pattern) if p.suffix.lower() in IMG_EXTS]

def norm_key(stem: str) -> str:
    s = stem.lower()
    for suf in NORMALIZE_SUFFIXES:
        s = re.sub(suf, '', s)
    return s

def image_size(path: Path):
    try:
        with Image.open(path) as im:
            return im.size  # (W,H)
    except Exception:
        return (0,0)

def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    shutil.copy2(src, dst)

def resize_copy(src: Path, dst: Path, max_side: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src) as im:
            im = im.convert('RGB')
            w, h = im.size
            scale = max_side / max(w, h)
            if scale < 1.0:
                im = im.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            im.save(dst, quality=95)
            return True
    except Exception:
        # Not a format Pillow can open (likely RAW) -> fallback to copy
        shutil.copy2(src, dst)
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--before_glob", required=True, help="e.g. '*/before/**/*.*'")
    ap.add_argument("--after_glob", required=True,  help="e.g. '*/after/**/*.*'")
    ap.add_argument("--dataset_version", default="dataset_v1")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--auto_match", action="store_true")
    ap.add_argument("--resize", type=int, default=0, help="if >0, resize copies to max side length")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset_root = Path(cfg["dataset_root"]).expanduser()
    output_root = Path(cfg["output_root"]).expanduser()
    random_seed = int(cfg.get("random_seed", 42))
    random.seed(random_seed)

    before_files = list_files(dataset_root, args.before_glob)
    after_files  = list_files(dataset_root, args.after_glob)

    if not before_files or not after_files:
        print("No files found. Check your globs and dataset_root in config.yaml")
        sys.exit(1)

    before_map: Dict[str, List[Path]] = {}
    after_map: Dict[str, List[Path]]  = {}

    for p in before_files:
        key = norm_key(p.stem) if args.auto_match else p.stem.lower()
        before_map.setdefault(key, []).append(p)

    for p in after_files:
        key = norm_key(p.stem) if args.auto_match else p.stem.lower()
        after_map.setdefault(key, []).append(p)

    pairs: List[Tuple[Path, Path]] = []
    misses_before, misses_after = [], []

    keys = set(before_map.keys()) | set(after_map.keys())
    for k in sorted(keys):
        b = before_map.get(k, [])
        a = after_map.get(k, [])
        if len(b)==1 and len(a)==1:
            pairs.append((b[0], a[0]))
        else:
            if not b and a:
                for x in a: misses_after.append(x)
            if not a and b:
                for x in b: misses_before.append(x)
            if len(b)>1 or len(a)>1:
                for x in b: misses_before.append(x)
                for x in a: misses_after.append(x)

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs)*args.val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    ds_dir = output_root / args.dataset_version
    train_b = ds_dir / "train/before"
    train_a = ds_dir / "train/after"
    val_b   = ds_dir / "val/before"
    val_a   = ds_dir / "val/after"
    for d in [train_b, train_a, val_b, val_a]:
        d.mkdir(parents=True, exist_ok=True)

    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_csv = reports_dir / "dataset_report.csv"

    with open(report_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set","before_path","after_path","before_w","before_h","after_w","after_h","note"])
        # write pairs
        for split, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
            for b, a in tqdm(split_pairs, desc=f"Writing {split}"):
                bw, bh = image_size(b)
                aw, ah = image_size(a)
                note = ""
                relname = b.stem
                bdst = (train_b if split=="train" else val_b) / f"{relname}{b.suffix.lower()}"
                adst = (train_a if split=="train" else val_a) / f"{relname}{a.suffix.lower()}"
                try:
                    if args.resize>0:
                        # Resize when possible, copy otherwise (RAW)
                        resized_b = resize_copy(b, bdst, args.resize)
                        resized_a = resize_copy(a, adst, args.resize)
                        if not resized_b or not resized_a:
                            note = "raw_copied_no_resize"
                    else:
                        copy_file(b, bdst)
                        copy_file(a, adst)
                except Exception as e:
                    note = f"copy_error:{e}"
                w.writerow([split, str(b), str(a), bw, bh, aw, ah, note])

        for m in misses_before:
            bw, bh = image_size(m)
            w.writerow(["unmatched_before", str(m), "", bw, bh, "", "", "no_after_match"])
        for m in misses_after:
            aw, ah = image_size(m)
            w.writerow(["unmatched_after", "", str(m), "", "", aw, ah, "no_before_match"])

    print(f"Pairs: total={len(pairs)} | train={len(train_pairs)} | val={len(val_pairs)}")
    print(f"Report: {report_csv}")
    print(f"Organized dataset at: {ds_dir}")

if __name__ == "__main__":
    main()
