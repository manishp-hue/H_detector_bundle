#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo " NEWSPACE AIR"
echo "=========================================="

if [[ -r /run/newspace-camera.status ]]; then
    echo "[CAMERA] Boot check: $(cat /run/newspace-camera.status)"
else
    echo "[CAMERA] Boot check: UNKNOWN"
fi

echo "[AIR] Camera opens only on Detection ON"
echo "[AIR] Camera releases on Detection OFF"
echo

exec "$ROOT/run_core.sh" "$@"
