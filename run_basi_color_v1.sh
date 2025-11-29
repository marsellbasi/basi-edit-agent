#!/usr/bin/env bash
set -e

# Get script directory and cd into BASI_Edit_Agent_Starter
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/BASI_Edit_Agent_Starter"

# Parse optional arguments
INPUT_GLOB="${1:-test_inputs_unet/*.jpg}"
OUTPUT_DIR="${2:-outputs_basi_color_v1}"

# Run BASI Color v1 (UNet) inference
python3 apply_color_model.py \
  --config config.yaml \
  --input_glob "$INPUT_GLOB" \
  --output_dir "$OUTPUT_DIR" \
  --model_ckpt "checkpoints/unet_color/latest.pt"

# Print summary
echo "BASI Color v1 (UNet) complete."
echo "Input:  $INPUT_GLOB"
echo "Output: $OUTPUT_DIR"

