#!/usr/bin/env bash
set -e

echo "=== BASI Pod Bootstrap v5 (HDRNet Color + bg_v2 checkpoints) ==="

# Always do everything from /workspace/code
cd /workspace/code

# -------- [1/6] Google Cloud SDK (gcloud + gsutil) --------
if ! command -v gcloud &>/dev/null; then
  echo "[1/6] Installing Google Cloud SDK (gcloud + gsutil)..."

  # Download SDK tarball if it doesn't exist yet
  if [ ! -f "google-cloud-cli-489.0.0-linux-x86_64.tar.gz" ]; then
    curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-489.0.0-linux-x86_64.tar.gz
  fi

  # Extract under /workspace/code/google-cloud-sdk
  if [ ! -d "google-cloud-sdk" ]; then
    tar -xf google-cloud-cli-489.0.0-linux-x86_64.tar.gz
  fi
else
  echo "[1/6] gcloud already available in this container."
fi

# Put gcloud + gsutil on PATH for this shell
if [ -f "/workspace/code/google-cloud-sdk/path.bash.inc" ]; then
  # shellcheck source=/dev/null
  source /workspace/code/google-cloud-sdk/path.bash.inc
fi

# Persist PATH so new shells automatically have gcloud/gsutil as well
if [ -f "/workspace/code/google-cloud-sdk/path.bash.inc" ]; then
  if ! grep -q "google-cloud-sdk/path.bash.inc" "${HOME}/.bashrc" 2>/dev/null; then
    echo "source /workspace/code/google-cloud-sdk/path.bash.inc" >> "${HOME}/.bashrc"
  fi
fi

echo "[1/6] gcloud version: $(gcloud --version | head -n 1 || echo 'unknown')"
echo "[1/6] gsutil version: $(gsutil --version 2>/dev/null || echo 'unknown')"

# -------- [2/6] Python deps (tqdm, etc.) --------
echo "[2/6] Ensuring required Python packages are installed..."

python -m pip install --upgrade pip
# Add more deps here as the project grows
python -m pip install --no-cache-dir tqdm

echo "[2/6] Python deps installed."

# -------- [3/6] Service account auth + project --------
echo "[3/6] Activating service account & setting project..."

SA_KEY="/workspace/code/beaming-theorem-475904-f5-d6369aa7655d.json"
# Make sure this JSON is uploaded to that path on the pod.

if [ ! -f "$SA_KEY" ]; then
  echo "ERROR: Service account JSON not found at: $SA_KEY"
  echo "Upload it to that path on the pod, then re-run this script."
  exit 1
fi

gcloud auth activate-service-account \
  --key-file="$SA_KEY"

gcloud config set project beaming-theorem-475904-f5

# -------- [4/6] Clone / update basi-edit-agent repo --------
echo "[4/6] Cloning / updating basi-edit-agent repo..."

cd /workspace/code

if [ ! -d "basi-edit-agent" ]; then
  git clone https://github.com/marsellbasi/basi-edit-agent.git
fi

cd /workspace/code/basi-edit-agent
git pull origin main || true

# -------- [5/6] Sync checkpoints from GCS (bg_v2_residual + hdrnet_color) --------
echo "[5/6] Syncing checkpoints from GCS..."

# Stage 2 background model checkpoints
BG_MODEL_DIR="/workspace/code/basi-edit-agent/BASI_Edit_Agent_Starter/checkpoints/bg_v2_residual"
mkdir -p "$BG_MODEL_DIR"

gsutil -m rsync -r \
  gs://basi-joan-ai/checkpoints/bg_v2_residual \
  "$BG_MODEL_DIR" || true

echo "[5/6] bg_v2_residual checkpoints synced to: $BG_MODEL_DIR"

# Stage 1 HDRNet color model checkpoints (if they exist yet)
HDRNET_MODEL_DIR="/workspace/code/basi-edit-agent/BASI_Edit_Agent_Starter/checkpoints/hdrnet_color"
mkdir -p "$HDRNET_MODEL_DIR"

gsutil -m rsync -r \
  gs://basi-joan-ai/checkpoints/hdrnet_color \
  "$HDRNET_MODEL_DIR" || true

echo "[5/6] hdrnet_color checkpoints synced to: $HDRNET_MODEL_DIR"

# -------- [6/6] Start Stage 1 HDRNet color training --------
echo "[6/6] Starting Stage 1 HDRNet color training..."

cd /workspace/code/basi-edit-agent/BASI_Edit_Agent_Starter

python3 train_hdrnet_color.py \
  --config config.yaml \
  --dataset_version dataset_v1 \
  --epochs 20

echo
echo "=== BASI Pod Bootstrap v5 complete ✅ ==="
echo "Repo:              /workspace/code/basi-edit-agent"
echo "BG checkpoints:    $BG_MODEL_DIR"
echo "HDRNet checkpoints:$HDRNET_MODEL_DIR"
echo "Training:          HDRNet color model (Stage 1) just ran with the args above."
