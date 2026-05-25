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
CONFIG_PATH="${CONFIG_PATH:-$SCRIPT_DIR/model_config_parse_current_32x64x64.json}"

OUTPUT_PARENT="$(dirname "$OUTPUT")"
mkdir -p "$OUTPUT_PARENT"
OUTPUT_DIR="$(cd "$OUTPUT_PARENT" && pwd)"
CASE_NAME="$(basename "$OUTPUT" .nii.gz)"
LOG_DIR="${MONITOR_LOG_DIR:-$OUTPUT_DIR/logs}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-10}"
mkdir -p "$LOG_DIR"

RUN_LOG="$LOG_DIR/${CASE_NAME}_infer.log"
MONITOR_LOG="$LOG_DIR/${CASE_NAME}_monitor.log"
SUMMARY_TSV="$LOG_DIR/inference_runs.tsv"
METRICS_JSONL="${METRICS_JSONL:-$OUTPUT_DIR/metrics_infer.jsonl}"

start_epoch="$(date +%s)"
start_iso="$(date '+%F %T %Z')"
echo "[$start_iso] start input=$INPUT output=$OUTPUT args=$*" | tee -a "$RUN_LOG"
if [ ! -f "$SUMMARY_TSV" ]; then
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "start_time" "finish_time" "case" "elapsed_seconds" "input" "output" "extra_args" >> "$SUMMARY_TSV"
fi

"$SCRIPT_DIR/runtime_monitor.sh" snapshot 0 "$MONITOR_LOG" || true
monitor_pid=""
if [ "$MONITOR_INTERVAL" != "0" ]; then
  "$SCRIPT_DIR/runtime_monitor.sh" watch "$MONITOR_INTERVAL" "$MONITOR_LOG" &
  monitor_pid="$!"
fi

cleanup_monitor() {
  if [ -n "$monitor_pid" ]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
}
trap cleanup_monitor EXIT

cmd=(
  python3 "$SCRIPT_DIR/device_infer_nii_rknn_nnunetv2_aligned.py"
  -i "$INPUT"
  -o "$OUTPUT"
  --config "$CONFIG_PATH"
  --metrics-jsonl "$METRICS_JSONL"
  "$@"
)

if command -v /usr/bin/time >/dev/null 2>&1; then
  /usr/bin/time -v "${cmd[@]}" 2>&1 | tee -a "$RUN_LOG"
else
  "${cmd[@]}" 2>&1 | tee -a "$RUN_LOG"
fi

cleanup_monitor
trap - EXIT
"$SCRIPT_DIR/runtime_monitor.sh" snapshot 0 "$MONITOR_LOG" || true

end_epoch="$(date +%s)"
end_iso="$(date '+%F %T %Z')"
elapsed="$((end_epoch - start_epoch))"
echo "[$end_iso] finished case=$CASE_NAME elapsed_seconds=$elapsed output=$OUTPUT" | tee -a "$RUN_LOG"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$start_iso" "$end_iso" "$CASE_NAME" "$elapsed" "$INPUT" "$OUTPUT" "$*" >> "$SUMMARY_TSV"
