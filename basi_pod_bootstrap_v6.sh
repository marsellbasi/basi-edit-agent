#!/usr/bin/env bash
set -e

echo "=== BASI Pod Bootstrap v6 (UNet color + BG residual) ==="

# Always work from /workspace/code
cd /workspace/code

# ---------------- [1/6] Install Google Cloud SDK (gcloud + gsutil) ----------------
if ! command -v gcloud &>/dev/null; then
  echo "[1/6] Installing Google Cloud SDK..."

  if [ ! -f "google-cloud-cli-489.0.0-linux-x86_64.tar.gz" ]; then
    curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-489.0.0-linux-x86_64.tar.gz
  fi

  tar -xf google-cloud-cli-489.0.0-linux-x86_64.tar.gz
  ./google-cloud-sdk/install.sh -q
fi

# Make sure gcloud / gsutil are on PATH for this shell
if [ -f "/workspace/code/google-cloud-sdk/path.bash.inc" ]; then
  # shellcheck disable=SC1091
  source "/workspace/code/google-cloud-sdk/path.bash.inc"
fi

echo "[1/6] gcloud: $(command -v gcloud || echo 'NOT FOUND')"
echo "[1/6] gsutil: $(command -v gsutil || echo 'NOT FOUND')"

# ---------------- [2/6] Authenticate service account (if key is present) ---------
SA_KEY="/workspace/code/basi-joan-ai-sa.json"
GCP_PROJECT="basi-joan-ai"

if [ -f "$SA_KEY" ]; then
  echo "[2/6] Activating service account and setting project..."
  gcloud auth activate-service-account --key-file="$SA_KEY"
  gcloud config set project "$GCP_PROJECT"
else
  echo "[2/6] WARNING: Service account key not found at $SA_KEY"
  echo "       gsutil commands will fail unless you authenticate manually."
fi

# ---------------- [3/6] Clone or update basi-edit-agent repo ---------------------
if [ ! -d "basi-edit-agent" ]; then
  echo "[3/6] Cloning basi-edit-agent repo..."
  git clone https://github.com/marsellbasi/basi-edit-agent.git
fi

cd basi-edit-agent
echo "[3/6] Updating repo from origin/main..."
git fetch origin main
git checkout main
git pull --rebase origin main

# ---------------- [4/6] Python environment / dependencies ------------------------
echo "[4/6] Installing Python dependencies..."

python3 -m pip install --upgrade pip
python3 -m pip install \
  tqdm \
  einops \
  kornia \
  lpips \
  pyyaml \
  opencv-python \
  Pillow

# (PyTorch and CUDA are assumed to be preinstalled on the RunPod image.)

# ---------------- [5/6] Sync datasets + checkpoints from GCS ---------------------

DATA_DIR="./BASI_EDIT_AGENT"
CKPT_ROOT="./BASI_Edit_Agent_Starter/checkpoints"

mkdir -p "$DATA_DIR"
mkdir -p "$CKPT_ROOT/unet_color" "$CKPT_ROOT/hdrnet_color" "$CKPT_ROOT/bg_residual"

if command -v gsutil &>/dev/null; then
  echo "[5/6] Syncing BASI_EDIT_AGENT datasets from GCS (bg_v1 + others)..."
  gsutil -m rsync -r "gs://basi-joan-ai/BASI_EDIT_AGENT" "$DATA_DIR" || echo "[5/6] Dataset rsync skipped/failed."

  echo "[5/6] Syncing UNet color checkpoints from GCS..."
  gsutil -m rsync -r "gs://basi-joan-ai/checkpoints/unet_color" "$CKPT_ROOT/unet_color" || echo "[5/6] UNet rsync skipped/failed."

  echo "[5/6] Syncing HDRNet color checkpoints from GCS (if present)..."
  gsutil -m rsync -r "gs://basi-joan-ai/checkpoints/hdrnet_color" "$CKPT_ROOT/hdrnet_color" || echo "[5/6] HDRNet rsync skipped/failed."

  echo "[5/6] Syncing BG residual checkpoints from GCS (if present)..."
  gsutil -m rsync -r "gs://basi-joan-ai/checkpoints/bg_residual" "$CKPT_ROOT/bg_residual" || echo "[5/6] BG residual rsync skipped/failed."
else
  echo "[5/6] WARNING: gsutil not available; skipping dataset/checkpoint sync."
fi

# ---------------- [6/6] Start or resume BG residual training ---------------------

cd BASI_Edit_Agent_Starter
echo "[6/6] Starting BASI Background Residual training (Stage 2)..."

# Common args that reflect our current setup
COMMON_ARGS=(
  --config config.yaml
  --dataset_version bg_v1
  --epochs 20
  --batch_size 2
  --max_side 1024
)

if [ -f "checkpoints/bg_residual/latest.pt" ]; then
  echo "[6/6] Found checkpoints/bg_residual/latest.pt — resuming training."
  python3 train_bg_model_residual.py "${COMMON_ARGS[@]}" --resume
else
  echo "[6/6] No existing bg_residual checkpoint — starting from scratch."
  python3 train_bg_model_residual.py "${COMMON_ARGS[@]}"
fi

echo "=== BASI Pod Bootstrap v6 complete ==="

