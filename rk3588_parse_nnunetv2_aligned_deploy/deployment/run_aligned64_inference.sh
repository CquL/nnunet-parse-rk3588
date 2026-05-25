#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_PATH="${CONFIG_PATH:-$SCRIPT_DIR/model_config_parse_mid_64x128x128.json}"
exec bash "$SCRIPT_DIR/run_aligned32_inference.sh" "$@"

