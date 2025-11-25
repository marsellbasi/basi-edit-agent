#!/usr/bin/env bash
set -e

echo "=== BASI Pod Bootstrap v2 ==="

# 1) Go to workspace root
cd /workspace/code

# -------- GCloud / gsutil --------
if ! command -v gcloud >/dev/null 2>&1; then
  echo "[1/4] Installing Google Cloud SDK..."
  if [ ! -d "google-cloud-sdk" ]; then
    curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-489.0.0-linux-x86_64.tar.gz
    tar -xf google-cloud-cli-489.0.0-linux-x86_64.tar.gz
  fi
  # Add to PATH for this shell
  source /workspace/code/google-cloud-sdk/path.bash.inc
  echo "gcloud/gsutil now on PATH."
else
  echo "[1/4] gcloud already available."
fi

# Ensure gsutil is available (comes with gcloud)
if ! command -v gsutil >/dev/null 2>&1; then
  echo "ERROR: gsutil not found even after SDK install."
  exit 1
fi

# -------- Service account auth --------
echo "[2/4] Activating service account..."
SA_KEY="/workspace/code/beaming-theorem-475904-f5-d6369aa7655d.json"

if [ ! -f "$SA_KEY" ]; then
  echo "ERROR: Service account JSON not found at: $SA_KEY"
  echo "Upload it to that path, then re-run this script."
  exit 1
fi

gcloud auth activate-service-account \
  --key-file="$SA_KEY"

gcloud config set project beaming-theorem-475904-f5

# -------- Repo setup --------
echo "[3/4] Cloning / updating basi-edit-agent repo..."
cd /workspace/code
if [ ! -d "basi-edit-agent" ]; then
  git clone https://github.com/marsellbasi/basi-edit-agent.git
fi

cd /workspace/code/basi-edit-agent
git pull origin main || true

# -------- Stage 2 checkpoints from GCS --------
echo "[4/4] Ensuring bg residual checkpoints are present..."

MODEL_DIR="/workspace/code/basi-edit-agent/BASI_Edit_Agent_Starter/checkpoints/bg_v1_residual_e10"
mkdir -p "$MODEL_DIR"

gsutil cp -n gs://basi-joan-ai/checkpoints/bg_v1_residual_e10/bg_residual_best.pt "$MODEL_DIR" || true
gsutil cp -n gs://basi-joan-ai/checkpoints/bg_v1_residual_e10/bg_residual_last.pt "$MODEL_DIR" || true
gsutil cp -n gs://basi-joan-ai/checkpoints/bg_v1_residual_latest.pt "$MODEL_DIR" || true

echo "Bootstrap complete ✅"
echo "Repo: /workspace/code/basi-edit-agent"
echo "Checkpoints: $MODEL_DIR"
