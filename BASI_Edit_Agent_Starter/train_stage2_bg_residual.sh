#!/usr/bin/env bash
# Stage 2 background residual training + backup + previews

set -e

echo "=== BASI Stage 2: BgResidualNet pipeline ==="

# Always run from the starter folder
cd /workspace/code/basi-edit-agent/BASI_Edit_Agent_Starter

# -------- Configurable knobs --------
EPOCHS="${1:-20}"          # total epochs for this run
BATCH_SIZE="${2:-2}"
MAX_SIDE="${3:-640}"
IDENTITY_WEIGHT="${4:-0.3}"

DATA_ROOT="BASI_EDIT_AGENT/bg_v1"
TRAIN_BEFORE_GLOB="${DATA_ROOT}/train/before/*.jpg"
TRAIN_AFTER_GLOB="${DATA_ROOT}/train/after/*.jpg"
VAL_BEFORE_GLOB="${DATA_ROOT}/val/before/*.jpg"
VAL_AFTER_GLOB="${DATA_ROOT}/val/after/*.jpg"

# We keep using this directory for Stage 2 bg model
MODEL_DIR="checkpoints/bg_v1_residual_e10"

# Where to drop val predictions + triplet strips for this run
VAL_PRED_DIR="${DATA_ROOT}/val/bg_clean_e${EPOCHS}"
TRIPLET_DIR="${DATA_ROOT}/val/bg_v1_residual_e${EPOCHS}_triplets"

mkdir -p "${MODEL_DIR}"
mkdir -p "${VAL_PRED_DIR}"
mkdir -p "${TRIPLET_DIR}"

# -------- Auto-resume logic --------
RESUME_FLAGS=()

if [ -f "${MODEL_DIR}/bg_residual_last.pt" ]; then
  echo "[Stage2] Found existing checkpoint at ${MODEL_DIR}/bg_residual_last.pt – will resume."
  RESUME_FLAGS=(--resume --resume_ckpt "${MODEL_DIR}/bg_residual_last.pt")
else
  echo "[Stage2] No existing checkpoint found – training from scratch."
fi

# -------- 1) Train BgResidualNet --------
echo
echo "=== [1/4] Training BgResidualNet ==="
echo "Epochs:       ${EPOCHS}"
echo "Batch size:   ${BATCH_SIZE}"
echo "Max side:     ${MAX_SIDE}"
echo "Model dir:    ${MODEL_DIR}"
echo "Identity wgt: ${IDENTITY_WEIGHT}"
echo

python3 train_bg_model_residual.py \
  --train_before_glob "${TRAIN_BEFORE_GLOB}" \
  --train_after_glob  "${TRAIN_AFTER_GLOB}" \
  --val_before_glob   "${VAL_BEFORE_GLOB}" \
  --val_after_glob    "${VAL_AFTER_GLOB}" \
  --epochs            "${EPOCHS}" \
  --batch_size        "${BATCH_SIZE}" \
  --max_side          "${MAX_SIDE}" \
  --model_dir         "${MODEL_DIR}" \
  --identity_weight   "${IDENTITY_WEIGHT}" \
  "${RESUME_FLAGS[@]}"

echo
echo "=== [1/4] Training complete. Latest checkpoints: ==="
ls -lh "${MODEL_DIR}" || true

# -------- 2) Backup checkpoints to GCS (if gsutil available) --------
echo
echo "=== [2/4] Backing up checkpoints to GCS (if gsutil is available) ==="

if command -v gsutil &>/dev/null; then
  BUCKET_DIR="gs://basi-joan-ai/checkpoints/bg_v1_residual_e10"

  for fname in bg_residual_best.pt bg_residual_last.pt bg_v1_residual_latest.pt; do
    if [ -f "${MODEL_DIR}/${fname}" ]; then
      echo "Uploading ${fname} -> ${BUCKET_DIR}/${fname}"
      gsutil -m cp "${MODEL_DIR}/${fname}" "${BUCKET_DIR}/${fname}"
    else
      echo "Skipping ${fname} (not found locally)."
    fi
  done
else
  echo "gsutil not on PATH – skipping GCS backup. Run basi_pod_bootstrap_v2.sh or refresh_gcloud_env.sh first."
fi

# -------- 3) Apply bg model to val set --------
echo
echo "=== [3/4] Applying BgResidualNet to val set ==="

python3 apply_bg_model_residual.py \
  --input_glob   "${VAL_BEFORE_GLOB}" \
  --output_dir   "${VAL_PRED_DIR}" \
  --model_ckpt   "${MODEL_DIR}/bg_residual_best.pt" \
  --residual_scale 0.3

echo "Val predictions written to: ${VAL_PRED_DIR}"

# -------- 4) Build before|pred|after triplet strips --------
echo
echo "=== [4/4] Building triplet previews ==="

python3 make_triplet_previews.py \
  --before_glob   "${VAL_BEFORE_GLOB}" \
  --after_glob    "${VAL_AFTER_GLOB}" \
  --output_dir    "${TRIPLET_DIR}" \
  --model_ckpt    "${MODEL_DIR}/bg_residual_best.pt" \
  --residual_scale 0.3

echo "Triplet previews written to: ${TRIPLET_DIR}"
echo
echo "=== BASI Stage 2 pipeline complete ✅ ==="
echo "Checkpoints:  ${MODEL_DIR}"
echo "Val outputs:  ${VAL_PRED_DIR}"
echo "Triplets:     ${TRIPLET_DIR}"

