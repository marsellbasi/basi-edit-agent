#!/usr/bin/env bash
# Stage 1 global color training wrapper

set -e

echo "=== Stage 1: Global color training ==="

# Change into the starter directory
cd "$(dirname "$0")"

# -------- Configurable knobs --------
EPOCHS="${1:-${EPOCHS:-20}}"          # allow CLI arg or env var, default 20
MAX_SIDE="${MAX_SIDE:-640}"
PATCH_SIZE="${PATCH_SIZE:-192}"
DATASET_VERSION="${DATASET_VERSION:-dataset_v1}"

MODEL_DIR="checkpoints/color_v1_e${EPOCHS}"

# Checkpoint path (train_color_model.py saves to JSON)
CKPT_JSON="BASI_ARCHIVE/models/color_v0/color_model.json"

echo "Epochs:          ${EPOCHS}"
echo "Max side:        ${MAX_SIDE}"
echo "Patch size:      ${PATCH_SIZE}"
echo "Dataset version: ${DATASET_VERSION}"
echo "Model dir:       ${MODEL_DIR}"
echo

# Create model directory
mkdir -p "${MODEL_DIR}"

# -------- Auto-resume logic --------
# train_color_model.py automatically loads from CKPT_JSON if it exists
if [ -f "${CKPT_JSON}" ]; then
  echo "[Stage1] Found existing checkpoint at ${CKPT_JSON} – will resume."
else
  echo "[Stage1] No existing checkpoint found – starting fresh."
fi

# -------- Train color model --------
echo
echo "=== Training GlobalColorModel ==="
echo

python3 train_color_model.py \
  --config config.yaml \
  --dataset_version "${DATASET_VERSION}" \
  --epochs "${EPOCHS}" \
  --max_side "${MAX_SIDE}" \
  --patch_size "${PATCH_SIZE}"

# Copy checkpoint to organized model directory
if [ -f "${CKPT_JSON}" ]; then
  mkdir -p "${MODEL_DIR}"
  cp "${CKPT_JSON}" "${MODEL_DIR}/color_model.json"
  echo "Copied checkpoint to ${MODEL_DIR}/color_model.json"
fi

echo
echo "=== Stage 1 color training complete ✅ ==="
echo "Checkpoints saved under: ${CKPT_JSON}"
echo "Also copied to: ${MODEL_DIR}/color_model.json"

