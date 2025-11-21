# BASI Edit Agent — Starter Kit (Windows/macOS)

This kit gives you:
1) **Dataset Checker** — pairs your before/after images into a clean train/val set and writes a CSV report.
2) **Color Model v0** — trains a global color transform (matrix + tone curves) that mimics your "after" look.
3) **Apply Script** — applies the trained transform to full‑res images.
4) **Watch Agent** — a simple watch‑folder runner (drop files → get outputs).

## 0) Create the conda environment
```bash
conda env create -f environment.yml
conda activate basi-edit-agent
```

## 1) Edit `config.yaml`
Set the two paths:
- `dataset_root`: where your original folders live on your external drive
- `output_root`: where the organized dataset & outputs should be written

**Windows example**
```yaml
dataset_root: "E:/Photos/BASI_ARCHIVE"
output_root:  "E:/BASI_EDIT_AGENT"
```

**macOS example**
```yaml
dataset_root: "/Volumes/Basi M2/Photos/BASI_ARCHIVE"
output_root:  "/Volumes/Basi M2/BASI_EDIT_AGENT"
```

## 2) Run the Dataset Checker
This scans `dataset_root` using your globs, pairs files, and writes clean copies into `<output_root>/<dataset_version>/...`

```bash
python dataset_checker.py --config config.yaml   --before_glob "*/before/**/*.*"   --after_glob  "*/after/**/*.*"   --auto_match   --val_ratio 0.1   --resize 1536
```

- Use `--resize 0` if you want 1:1 copies.
- RAW files that can't be opened by Pillow will be copied (not resized).
- Report CSV: `<output_root>/reports/dataset_report.csv`

## 3) Train the Color Model (v0)
```bash
python train_color_model.py --config config.yaml   --dataset_version dataset_v1   --epochs 10   --patch_size 256   --max_side 768
```
Outputs:
- `<output_root>/models/color_v0/color_model.json`

## 4) Apply the model to full‑res images
```bash
python apply_color_model.py --config config.yaml   --model_dir "<output_root>/models/color_v0"   --input_glob "<some path>/**/*.jpg"   --out_dir "<output_root>/inference/color_v0"
```

## 5) Watch‑folder agent
```bash
python watch_agent.py --config config.yaml --sleep 10
```
Drop JPG/PNG/TIF images into `<output_root>/agent/inbox`; outputs land in `outbox/`. Done markers are written to `done/`.

---
If you get stuck, open the CSV report and check unmatched rows.
