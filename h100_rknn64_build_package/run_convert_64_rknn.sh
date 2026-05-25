#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ONNX="deployment/parse_3d_fullres_patch_64x128x128.onnx"
RKNN="deployment/parse_3d_fullres_patch_64x128x128.rknn"
LOG="deployment/convert_64x128x128_$(date '+%Y%m%d_%H%M%S').log"

if [ ! -f "$ONNX" ]; then
  echo "Missing ONNX: $ONNX" >&2
  echo "Run bash run_export_64_rknn.sh first." >&2
  exit 2
fi

echo "[$(date -Iseconds)] Converting ONNX to RKNN: $RKNN" | tee "$LOG"
python convert_onnx_to_rknn.py \
  --onnx "$ONNX" \
  --rknn "$RKNN" 2>&1 | tee -a "$LOG"

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$RKNN" | tee "$RKNN.sha256"
fi
ls -lh "$RKNN" | tee -a "$LOG"
echo "[$(date -Iseconds)] Done: $RKNN" | tee -a "$LOG"

