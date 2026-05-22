#!/usr/bin/env bash
set -e

echo "== System =="
uname -a
if [ -f /etc/os-release ]; then
  . /etc/os-release
  echo "os=${PRETTY_NAME:-unknown}"
fi
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

echo "== Disk =="
df -h .
echo

echo "== Limits =="
ulimit -a
echo

echo "== RKNN runtime libraries =="
find /usr /lib /opt -name 'librknnrt.so*' 2>/dev/null || true
for lib in $(find /usr /lib /opt -name 'librknnrt.so*' 2>/dev/null); do
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$lib" || true
  fi
done
echo

echo "== RKNPU debug/status =="
if [ -r /sys/kernel/debug/rknpu/load ]; then
  cat /sys/kernel/debug/rknpu/load || true
elif ls /sys/kernel/debug/rknpu*/load >/dev/null 2>&1; then
  cat /sys/kernel/debug/rknpu*/load || true
else
  echo "No readable rknpu load file found. Try running with sudo or mount debugfs:"
  echo "  sudo mount -t debugfs none /sys/kernel/debug"
fi
find /sys/kernel/debug -maxdepth 3 -iname '*rknpu*' 2>/dev/null || true
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
    for name in dir(RKNNLite):
        if "NPU_CORE" in name:
            print(f"{name}={getattr(RKNNLite, name)}")
except Exception as exc:
    print("rknnlite import: FAILED")
    print(repr(exc))
PY
