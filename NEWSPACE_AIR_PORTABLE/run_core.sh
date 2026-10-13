#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DINO="$ROOT/models/gdino/grounding_dino_fixed.plan"
SAM="$ROOT/models/sam2_box.plan"
BIN="$ROOT/build/publisher_node"

[[ -f "$DINO" ]] || { echo "ERROR: missing $DINO" >&2; exit 3; }
[[ -f "$SAM" ]] || { echo "ERROR: missing $SAM" >&2; exit 3; }

if [[ ! -x "$BIN" ]]; then
    echo "[PORTABLE] publisher_node is not built; rebuilding for this Jetson..."
    JOBS="${JOBS:-2}" "$ROOT/build.sh"
fi

# Do not compete with a raw CSI gst-launch test. A second Argus client can
# destabilize the camera session. The publisher itself also has a C++ singleton lock.
if pgrep -af 'gst-launch-1.0.*nvarguscamerasrc' >/dev/null 2>&1; then
    echo "ERROR: another nvarguscamerasrc gst-launch process is already using CSI." >&2
    pgrep -af 'gst-launch-1.0.*nvarguscamerasrc' >&2 || true
    echo "Stop the raw camera test before starting NEWSPACE AIR." >&2
    exit 20
fi

# IMPORTANT: publisher_node itself is always headless.  Use local_viewer.sh or
# run_local_display.sh for a Jetson-local final detection window.  This prevents
# an X11 failure from killing control, CSI, detection and RTSP.
exec "$BIN" \
  --sensor-id "${SENSOR_ID:-0}" \
  --width "${WIDTH:-1920}" \
  --height "${HEIGHT:-1080}" \
  --fps "${FPS:-60}" \
  --stream-width "${STREAM_WIDTH:-1280}" \
  --stream-height "${STREAM_HEIGHT:-720}" \
  --stream-fps "${STREAM_FPS:-30}" \
  --bitrate "${VIDEO_BITRATE_KBPS:-3000}" \
  --control-port "${CONTROL_PORT:-5701}" \
  --rtsp-port "${RTSP_PORT:-8554}" \
  --local-display off \
  --dino-plan "$DINO" \
  --sam-plan "$SAM"
