#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
missing=0

for cmd in cmake g++ pkg-config gst-launch-1.0; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "MISSING command: $cmd"
        missing=1
    fi
done

if command -v pkg-config >/dev/null 2>&1; then
    for pc in \
      opencv4 \
      gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0 \
      gstreamer-rtsp-server-1.0 gstreamer-rtsp-1.0 gio-2.0; do
        if ! pkg-config --exists "$pc" 2>/dev/null; then
            echo "MISSING pkg-config module: $pc"
            missing=1
        fi
    done
fi

if [[ ! -f /usr/include/aarch64-linux-gnu/NvInfer.h && \
      ! -f /usr/include/NvInfer.h && \
      ! -f /usr/local/cuda/include/NvInfer.h ]]; then
    echo "MISSING TensorRT development header: NvInfer.h"
    missing=1
fi

LDCONFIG_TEXT="$(ldconfig -p 2>/dev/null || true)"
if ! grep -F 'libnvinfer.so' <<<"$LDCONFIG_TEXT" >/dev/null 2>&1 && \
   ! compgen -G '/usr/lib/aarch64-linux-gnu/libnvinfer.so*' >/dev/null && \
   ! compgen -G '/usr/lib/libnvinfer.so*' >/dev/null; then
    echo "MISSING TensorRT runtime library: libnvinfer.so"
    missing=1
fi

if [[ $missing -ne 0 ]]; then
    cat <<'EOM'

Dependency check FAILED.

For a normal JetPack Jetson, install missing build dependencies as needed:

  sudo apt update
  sudo apt install -y build-essential cmake pkg-config libopencv-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgstrtspserver-1.0-dev libglib2.0-dev

TensorRT/CUDA must match the NVIDIA JetPack installation on this Jetson.
EOM
    exit 2
fi

echo "Dependency check: PASS"
echo "Rebuilding publisher_node for this Jetson..."
rm -rf "$ROOT/build"
JOBS="${JOBS:-2}" "$ROOT/build.sh"
echo
echo "Setup complete."
echo "Production/headless: ./run.sh"
echo "Bench + local view: DISPLAY=:1 ./run_local_display.sh"
