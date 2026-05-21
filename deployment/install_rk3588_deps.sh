#!/usr/bin/env bash
set -e

echo "Installing basic Python dependencies for RK3588 NPU inference..."

sudo apt-get update
sudo apt-get install -y python3-pip python3-venv libgomp1

python3 -m pip install --user --upgrade pip
python3 -m pip install --user numpy SimpleITK rknn-toolkit-lite2

echo
echo "Checking imports..."
python3 - <<'PY'
import numpy
import SimpleITK
from rknnlite.api import RKNNLite

print("numpy:", numpy.__version__)
print("SimpleITK: OK")
print("RKNNLite: OK")
PY

echo
echo "Done. If rknnlite import still fails, the board image may be missing Rockchip RKNN runtime libraries."
