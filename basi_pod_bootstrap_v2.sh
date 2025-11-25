#!/usr/bin/env bash
set -e

echo "=== BASI Pod Bootstrap v3 ==="

# Always do everything from /workspace/code
cd /workspace/code

# -------- [1/5] Google Cloud SDK (gcloud + gsutil) --------
if ! command -v gcloud &>/dev/null; then
  echo "[1/5] Installing Google Cloud SDK (gcloud + gsutil)..."

  # Download SDK tarball if it doesn't exist yet
  if [ ! -f "google-cloud-cli-489.0.0-linux-x86_64.tar.gz" ]; then
    curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-489.0.0-linux-x86_64.tar.gz
  fi

  # Extract under /workspace/code/google-cloud-sdk
  if [ ! -d "google-cloud-sdk" ]; then
    tar -xf google-cloud-cli-489.0.0-linux-x86_64.tar.gz
  fi
else
  echo "[1/5] gcloud already available in this container."
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

echo "[1/5] gcloud version: $(gcloud --version | head -n 1 || echo 'unknown')"
echo "[1/5] gsutil version: $(gsutil --version 2>/dev/null || echo 'unknown')"

# -------- [2/5] Python deps (tqdm, etc.) --------
echo "[2/5] Ensuring required Python packages are installed..."

python -m pip install --upgrade pip
# Add more deps here as the project grows
python -m pip install --no-cache-dir tqdm

echo "[2/5] Python deps installed."

# -------- [3/5] Service account auth + project --------
echo "[3/5] Activating service account & setting project..."

SA_KEY="/workspace/code/beaming-theorem-475904-f5-d6369aa7655d.json"

if [ ! -f "$SA_KEY" ]; then
  echo "ERROR: Service account JSON not found at: $SA_KEY"
  echo "Upload it to that path on the pod, then re-run this script."
  exit 1
fi

gcloud auth activate-service-account \
  --key-file="$SA_KEY"

gcloud config set project beaming-theorem-475904-f5

# -------- [4/5] Clone / update basi-edit-agent repo --------
echo "[4/5] Cloning / updating basi-edit-agent repo..."

cd /workspace/code

if [ ! -d "basi-edit-agent" ]; then
  git clone https://github.com/marsellbasi/basi-edit-agent.git
fi

cd /workspace/code/basi-edit-agent
git pull origin main || true

# -------- [5/5] Sync Stage 2 bg_residual checkpoints from GCS --------
echo "[5/5] Syncing Stage 2 bg_residual checkpoints from GCS..."

MODEL_DIR="/workspace/code/basi-edit-agent/BASI_Edit_Agent_Starter/checkpoints/bg_v1_residual_e10"
mkdir -p "$MODEL_DIR"

# These objects already exist in the basi-joan-ai bucket.
gsutil -m cp -n \
  gs://basi-joan-ai/checkpoints/bg_v1_residual_e10/bg_residual_best.pt \
  "$MODEL_DIR" || true

gsutil -m cp -n \
  gs://basi-joan-ai/checkpoints/bg_v1_residual_e10/bg_residual_last.pt \
  "$MODEL_DIR" || true

gsutil -m cp -n \
  gs://basi-joan-ai/checkpoints/bg_v1_residual_e10/bg_v1_residual_latest.pt \
  "$MODEL_DIR" || true

echo
echo "=== BASI Pod Bootstrap v3 complete ✅ ==="
echo "Repo:        /workspace/code/basi-edit-agent"
echo "Checkpoints: $MODEL_DIR"
echo "Tip: open a new shell so ~/.bashrc sourcing for gcloud/gsutil kicks in."
