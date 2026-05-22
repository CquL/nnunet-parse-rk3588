#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage:"
  echo "  bash run_aligned32_inference.sh INPUT.nii.gz OUTPUT.nii.gz [extra args]"
  echo
  echo "Examples:"
  echo "  bash run_aligned32_inference.sh ../test_images/PA000005_0000.nii.gz ../outputs/PA000005_rknn_aligned32_mask.nii.gz"
  echo "  bash run_aligned32_inference.sh ../test_images/PA000005_0000.nii.gz ../outputs/PA000005_fast_mask.nii.gz --tile-step-size 1.0 --no-gaussian"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="$1"
OUTPUT="$2"
shift 2

python3 "$SCRIPT_DIR/device_infer_nii_rknn_nnunetv2_aligned.py" \
  -i "$INPUT" \
  -o "$OUTPUT" \
  --config "$SCRIPT_DIR/model_config_parse_current_32x64x64.json" \
  "$@"
