#!/usr/bin/env bash
set -e

echo "== System =="
uname -a
echo

echo "== CPU arch =="
uname -m
echo

echo "== Device tree compatible =="
cat /proc/device-tree/compatible 2>/dev/null | tr '\0' '\n' || true
echo

echo "== Memory =="
free -h
echo

echo "== RKNN runtime libraries =="
find /usr /lib /opt -name 'librknnrt.so*' 2>/dev/null || true
echo

echo "== Python RKNNLite import =="
python3 - <<'PY'
try:
    import numpy
    print("numpy import: OK", numpy.__version__)
except Exception as exc:
    print("numpy import: FAILED")
    print(repr(exc))

try:
    import SimpleITK as sitk
    print("SimpleITK import: OK", sitk.Version_VersionString())
except Exception as exc:
    print("SimpleITK import: FAILED")
    print(repr(exc))

try:
    import scipy
    print("scipy import: OK", scipy.__version__)
except Exception as exc:
    print("scipy import: OPTIONAL_MISSING")
    print(repr(exc))

try:
    import skimage
    print("scikit-image import: OK", skimage.__version__)
except Exception as exc:
    print("scikit-image import: OPTIONAL_MISSING")
    print(repr(exc))

try:
    from rknnlite.api import RKNNLite
    print("rknnlite import: OK")
    print("RKNNLite:", RKNNLite)
except Exception as exc:
    print("rknnlite import: FAILED")
    print(repr(exc))
PY
