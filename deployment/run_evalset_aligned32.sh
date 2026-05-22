#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage:"
  echo "  bash run_evalset_aligned32.sh EVALSET_DIR [OUTPUT_DIR] [extra inference args]"
  echo
  echo "Examples:"
  echo "  bash run_evalset_aligned32.sh ../../nnunetv2_PARSE_fold0_evalset ../outputs/evalset_aligned32"
  echo "  bash run_evalset_aligned32.sh ../../nnunetv2_PARSE_fold0_evalset ../outputs/evalset_fast --tile-step-size 1.0 --no-gaussian"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALSET_DIR="$1"
OUTPUT_DIR="${2:-../outputs/evalset_aligned32}"
shift || true
if [ "$#" -gt 0 ]; then
  shift || true
fi

IMAGE_DIR="$EVALSET_DIR/images"
LABEL_DIR="$EVALSET_DIR/labels"
mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/evalset_$(date '+%Y%m%d_%H%M%S').log"

if [ ! -d "$IMAGE_DIR" ]; then
  echo "Missing image directory: $IMAGE_DIR" >&2
  exit 2
fi

{
  echo "start_time=$(date -Iseconds)"
  echo "evalset_dir=$EVALSET_DIR"
  echo "output_dir=$OUTPUT_DIR"
  echo "extra_args=$*"
} | tee -a "$RUN_LOG"

for image in "$IMAGE_DIR"/*_0000.nii.gz; do
  case_name="$(basename "$image" _0000.nii.gz)"
  output="$OUTPUT_DIR/${case_name}.nii.gz"
  echo "[$(date '+%F %T')] Running $case_name" | tee -a "$RUN_LOG"
  bash "$SCRIPT_DIR/run_aligned32_inference.sh" "$image" "$output" "$@"
done

if [ -d "$LABEL_DIR" ]; then
  python3 "$SCRIPT_DIR/evaluate_segmentation_dice.py" \
    --pred-dir "$OUTPUT_DIR" \
    --label-dir "$LABEL_DIR" \
    --csv "$OUTPUT_DIR/metrics_eval.csv" \
    --summary-json "$OUTPUT_DIR/metrics_summary.json" | tee -a "$RUN_LOG"
else
  echo "Label directory not found, skipped Dice: $LABEL_DIR" | tee -a "$RUN_LOG"
fi

echo "end_time=$(date -Iseconds)" | tee -a "$RUN_LOG"
