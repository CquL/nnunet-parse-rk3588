#!/usr/bin/env bash
set -euo pipefail

mode="${1:-snapshot}"
interval="${2:-5}"
log_file="${3:-/dev/stdout}"

timestamp() {
  date '+%F %T %Z'
}

read_rknpu_load() {
  if [ -r /sys/kernel/debug/rknpu/load ]; then
    cat /sys/kernel/debug/rknpu/load
    return
  fi
  for path in /sys/kernel/debug/rknpu*/load; do
    if [ -r "$path" ]; then
      echo "== $path =="
      cat "$path"
    fi
  done
}

snapshot() {
  {
    echo "===== runtime snapshot: $(timestamp) ====="
    echo "== uptime/load =="
    uptime || true
    echo
    echo "== memory =="
    free -h || true
    echo
    echo "== disk =="
    df -h . /tmp 2>/dev/null || df -h || true
    echo
    echo "== python processes =="
    ps -eo pid,ppid,stat,%cpu,%mem,rss,vsz,etime,cmd --sort=-rss | head -20 || true
    echo
    echo "== rknpu load =="
    read_rknpu_load 2>/dev/null || echo "rknpu load not available"
    echo
    echo "== rknpu debug paths =="
    find /sys/kernel/debug -maxdepth 3 -iname '*rknpu*' 2>/dev/null | sort || true
    echo
  } >> "$log_file"
}

case "$mode" in
  snapshot)
    snapshot
    ;;
  watch)
    while true; do
      snapshot
      sleep "$interval"
    done
    ;;
  *)
    echo "Usage: $0 snapshot [interval] [log_file]" >&2
    echo "       $0 watch INTERVAL LOG_FILE" >&2
    exit 2
    ;;
esac
