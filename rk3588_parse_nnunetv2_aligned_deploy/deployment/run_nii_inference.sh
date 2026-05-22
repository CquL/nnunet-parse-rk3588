#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage:"
  echo "  bash run_nii_inference.sh INPUT.nii.gz OUTPUT.nii.gz [--center-patch-only]"
  echo
  echo "Examples:"
  echo "  bash run_nii_inference.sh ../PA000005_0000.nii.gz ../PA000005_rknn_mask.nii.gz --center-patch-only"
  echo "  bash run_nii_inference.sh ../PA000005_0000.nii.gz ../PA000005_rknn_mask.nii.gz"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="$1"
OUTPUT="$2"
shift 2

python3 "$SCRIPT_DIR/device_infer_nii_rknn.py" \
  -i "$INPUT" \
  -o "$OUTPUT" \
  "$@"
