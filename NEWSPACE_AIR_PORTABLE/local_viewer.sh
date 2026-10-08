#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RTSP_PORT="${RTSP_PORT:-8554}"
URL="rtsp://127.0.0.1:${RTSP_PORT}/main.264"
export DISPLAY="${DISPLAY:-:1}"

stop=0
trap 'stop=1' INT TERM

echo "[LOCAL VIEWER] DISPLAY=$DISPLAY"
echo "[LOCAL VIEWER] $URL"
echo "[LOCAL VIEWER] waits until Detection ON and a final annotated frame is ready"

while [[ $stop -eq 0 ]]; do
    if gst-inspect-1.0 nvv4l2decoder >/dev/null 2>&1; then
        gst-launch-1.0 -q \
          rtspsrc location="$URL" protocols=tcp latency=0 buffer-mode=none drop-on-latency=true tcp-timeout=3000000 \
          ! rtph264depay \
          ! h264parse \
          ! nvv4l2decoder \
          ! nvvidconv \
          ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream \
          ! nv3dsink sync=false 2>/dev/null || true
    else
        gst-launch-1.0 -q \
          rtspsrc location="$URL" protocols=tcp latency=0 buffer-mode=none drop-on-latency=true tcp-timeout=3000000 \
          ! rtph264depay \
          ! h264parse \
          ! avdec_h264 max-threads=2 \
          ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream \
          ! videoconvert \
          ! autovideosink sync=false 2>/dev/null || true
    fi

    [[ $stop -ne 0 ]] && break
    sleep 1

done
