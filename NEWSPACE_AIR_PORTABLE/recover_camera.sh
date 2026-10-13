#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[RECOVERY] stopping NEWSPACE AIR publisher if present..."
pkill -TERM -f "$ROOT/build/publisher_node" 2>/dev/null || true
sleep 1

echo "[RECOVERY] stopping raw nvarguscamerasrc gst-launch tests if present..."
pkill -TERM -f 'gst-launch-1.0.*nvarguscamerasrc' 2>/dev/null || true
sleep 1

echo "[RECOVERY] restarting NVIDIA Argus daemon..."
sudo systemctl restart nvargus-daemon
sleep 2

echo
echo "Argus recovery complete. Test raw CSI with:"
echo "gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=60/1,format=NV12' ! nvvidconv ! queue ! nv3dsink sync=false"
