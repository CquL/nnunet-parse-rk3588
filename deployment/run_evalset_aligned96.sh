#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_PATH="${CONFIG_PATH:-$SCRIPT_DIR/model_config_parse_true_96x160x160.json}"
exec bash "$SCRIPT_DIR/run_evalset_aligned32.sh" "$@"
