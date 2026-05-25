#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CHECKPOINT="nnunetv2_PARSE_model_minimal/Dataset501_PARSE/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
OUTPUT="deployment/parse_3d_fullres_patch_64x128x128.onnx"
LOG="deployment/export_64x128x128_$(date '+%Y%m%d_%H%M%S').log"

if [ ! -f "$CHECKPOINT" ]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  echo "Copy checkpoint_best.pth into this package first." >&2
  exit 2
fi

mkdir -p deployment
echo "[$(date -Iseconds)] Exporting ONNX to $OUTPUT" | tee "$LOG"
python export_nnunet_patch_onnx.py \
  --patch-size 64x128x128 \
  --output "$OUTPUT" 2>&1 | tee -a "$LOG"
echo "[$(date -Iseconds)] Done: $OUTPUT" | tee -a "$LOG"

