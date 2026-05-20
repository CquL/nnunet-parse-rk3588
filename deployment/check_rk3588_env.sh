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
    from rknnlite.api import RKNNLite
    print("rknnlite import: OK")
    print("RKNNLite:", RKNNLite)
except Exception as exc:
    print("rknnlite import: FAILED")
    print(repr(exc))
PY
