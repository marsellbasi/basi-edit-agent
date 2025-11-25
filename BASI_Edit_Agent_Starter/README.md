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

## RunPod Bootstrap (for Stage 2 Background Training)

For training the background residual model on RunPod pods:

### First-time pod setup
```bash
cd /workspace/code
git clone https://github.com/marsellbasi/basi-edit-agent.git
cd basi-edit-agent
bash basi_pod_bootstrap_v2.sh
```

This script will:
- Install Google Cloud SDK (gcloud + gsutil) if missing
- Ensure gcloud/gsutil are on PATH for current and future shells
- Activate service account and set project
- Clone or update the basi-edit-agent repo
- Sync Stage 2 bg_residual checkpoints from GCS
- Install Python dependencies (tqdm, etc.)

### New shell setup
If you open a new shell and need gcloud/gsutil on PATH:
```bash
bash refresh_gcloud_env.sh
```

Or rely on `~/.bashrc` sourcing (automatically added by bootstrap).

---

## Stage 1 – Global Color Model

### Train Stage 1 color model
```bash
cd /workspace/code/basi-edit-agent/BASI_Edit_Agent_Starter
chmod +x train_stage1_color.sh
./train_stage1_color.sh        # default 20 epochs
# or:
EPOCHS=10 ./train_stage1_color.sh
# or:
./train_stage1_color.sh 10
```

The script will automatically resume if it finds an existing checkpoint at `BASI_ARCHIVE/models/color_v0/color_model.json`.

### Apply Stage 1 to a folder
```bash
python3 apply_color_model.py \
  --input_glob "BASI_EDIT_AGENT/bg_v1/val/before/*.jpg" \
  --output_dir "BASI_EDIT_AGENT/stage1_color_only/val_e20" \
  --model_ckpt "checkpoints/color_v1_e20/color_model.json"
```

Or use the default checkpoint location:
```bash
python3 apply_color_model.py \
  --input_glob "BASI_EDIT_AGENT/bg_v1/val/before/*.jpg" \
  --output_dir "BASI_EDIT_AGENT/stage1_color_only/val_e20" \
  --model_ckpt "BASI_ARCHIVE/models/color_v0/color_model.json"
```

### Combine with Stage 2 triplets
After running Stage 1, you can create triplets showing the progression:
```bash
# Create triplets with Stage 1 outputs as "after" images
python make_triplet_previews.py \
  --before_glob "BASI_EDIT_AGENT/bg_v1/val/before/*.jpg" \
  --after_glob "BASI_EDIT_AGENT/stage1_color_only/val_e20/*.jpg" \
  --model_ckpt "checkpoints/bg_v1_residual_e10/bg_residual_best.pt" \
  --out_dir "BASI_EDIT_AGENT/bg_v1/val/stage1_stage2_triplets" \
  --residual_scale 0.3
```

---

## Stage 2 – Background Residual Model

### Training Stage 2 background model
```bash
cd BASI_Edit_Agent_Starter
python train_bg_model_residual.py \
  --train_before_glob "BASI_EDIT_AGENT/bg_v1/train/before/*.jpg" \
  --train_after_glob "BASI_EDIT_AGENT/bg_v1/train/after/*.jpg" \
  --val_before_glob "BASI_EDIT_AGENT/bg_v1/val/before/*.jpg" \
  --val_after_glob "BASI_EDIT_AGENT/bg_v1/val/after/*.jpg" \
  --model_dir "checkpoints/bg_v1_residual_e10" \
  --resume \
  --epochs 20 \
  --identity_weight 0.3
```

### Generating triplet previews
```bash
python make_triplet_previews.py \
  --before_glob "BASI_EDIT_AGENT/bg_v1/val/before/*.jpg" \
  --after_glob "BASI_EDIT_AGENT/bg_v1/val/after/*.jpg" \
  --model_ckpt "checkpoints/bg_v1_residual_e10/bg_residual_best.pt" \
  --out_dir "BASI_EDIT_AGENT/bg_v1/val/bg_v1_residual_e20_triplets" \
  --residual_scale 0.3
```

### Stage 2 wrapper script
For a complete pipeline (train + backup + apply + triplets) in one go:
```bash
cd /workspace/code/basi-edit-agent/BASI_Edit_Agent_Starter
chmod +x train_stage2_bg_residual.sh
bash train_stage2_bg_residual.sh 20
```

The script accepts positional arguments:
- `$1`: epochs (default: 20)
- `$2`: batch_size (default: 2)
- `$3`: max_side (default: 640)
- `$4`: identity_weight (default: 0.3)

It will:
1. Train BgResidualNet (auto-resumes if checkpoints exist)
2. Backup checkpoints to GCS (if gsutil is available)
3. Apply the model to validation "before" images
4. Build before|pred|after triplet strips

---

If you get stuck, open the CSV report and check unmatched rows.
