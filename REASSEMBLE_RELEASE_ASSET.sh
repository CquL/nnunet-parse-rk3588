#!/usr/bin/env bash
set -euo pipefail

output="${1:-nnunet_parse_rk3588_full_package.zip}"

rm -f "$output"
cat nnunet_parse_rk3588_full_package.zip.part* > "$output"

echo "Reassembled: $output"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$output"
fi
