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

## Stage 1 – HDRNet Color Model (BASI Color v1)

The HDRNet-based color model is an advanced alternative to the baseline global color model. It uses a bilateral grid architecture to predict local affine color transforms, enabling both global and local color/tone adjustments while preserving edges.

### What is HDRNet?

HDRNet (Deep Bilateral Learning for Real-Time Image Enhancement) uses:
- A **low-resolution encoder** that processes a downsampled input
- A **bilateral grid** of affine coefficients predicted from the low-res features
- **Bilateral slicing** that maps coefficients to full resolution using a luminance guide
- **Local affine color transforms** applied per-pixel to the full-resolution image

This architecture allows the model to learn spatially-varying color adjustments while maintaining edge-preserving properties.

### Training the HDRNet Color Model

```bash
cd BASI_Edit_Agent_Starter
python train_hdrnet_color.py \
  --config config.yaml \
  --dataset_version dataset_v1 \
  --epochs 20 \
  --max_side 640
```

**Command-line options:**
- `--config`: Path to YAML config file (required)
- `--dataset_version`: Dataset version (e.g., `dataset_v1`)
- `--epochs`: Number of training epochs (overrides config if provided)
- `--max_side`: Maximum side length for image resizing (default: 640)
- `--batch_size`: Batch size (overrides config if provided)
- `--resume_from`: Path to checkpoint to resume from
- `--resume`: Resume from latest checkpoint if it exists

**Checkpoints:**
- Saved to `checkpoints/hdrnet_color/epoch_{:03d}.pt`
- Latest checkpoint: `checkpoints/hdrnet_color/latest.pt`
- Each checkpoint includes model state, optimizer state, epoch number, and config

**Resuming training:**
```bash
# Resume from latest checkpoint
python train_hdrnet_color.py \
  --config config.yaml \
  --dataset_version dataset_v1 \
  --epochs 20 \
  --resume

# Resume from specific checkpoint
python train_hdrnet_color.py \
  --config config.yaml \
  --dataset_version dataset_v1 \
  --epochs 20 \
  --resume_from checkpoints/hdrnet_color/epoch_010.pt
```

### Running Inference with HDRNet

**Using config file (recommended):**
1. Set `color_model.type: "hdrnet"` in `config.yaml`
2. Run inference:
```bash
python apply_color_model.py \
  --config config.yaml \
  --input_glob "BASI_EDIT_AGENT/bg_v1/val/before/*.jpg" \
  --output_dir "BASI_EDIT_AGENT/stage1_hdrnet/val_e20" \
  --model_ckpt "checkpoints/hdrnet_color/latest.pt"
```

**Direct checkpoint path:**
```bash
python apply_color_model.py \
  --input_glob "BASI_EDIT_AGENT/bg_v1/val/before/*.jpg" \
  --output_dir "BASI_EDIT_AGENT/stage1_hdrnet/val_e20" \
  --model_ckpt "checkpoints/hdrnet_color/epoch_020.pt" \
  --config config.yaml
```

The script automatically detects HDRNet models by the `.pt` extension and uses the appropriate inference path.

### Configuration

HDRNet model parameters are configured in `config.yaml`:

```yaml
color_model:
  type: "hdrnet"  # or "baseline" for the original model
  hdrnet:
    bilateral_grid_size: [16, 16, 8]  # [H_grid, W_grid, D_grid]
    lowres_channels: 16
    hidden_dim: 64
    num_affine_params: 12  # 3 channels * 4 params (3 weights + 1 bias)
    input_downsample: 256  # Target side length for low-res processing

training:
  hdrnet_color:
    batch_size: 1
    num_workers: 0
    lr: 1.0e-4
    weight_decay: 0.0
    epochs: 20
    save_every: 1
    loss:
      l1_weight: 1.0
      ssim_weight: 0.0
```

### Switching Between Models

To switch between baseline and HDRNet models, simply change `color_model.type` in `config.yaml`:
- `"baseline"`: Uses the original global color model (JSON checkpoint)
- `"hdrnet"`: Uses the HDRNet model (PyTorch checkpoint)

The `apply_color_model.py` script automatically selects the correct model type based on the config or checkpoint file extension.

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
