#!/usr/bin/env bash
# NEWSPACE_ALWAYS_ON_IMX477_V2 - Newspace uses the existing camera owner.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE="$ROOT/run_core.sh"
cd "$ROOT"
UNIT="imx477-camera.service"
STATE="/run/newspace-imx477/status.json"
SOCKET="/run/newspace-imx477/frames.sock"

if [[ ! -x "$CORE" ]]; then
  echo "[NEWSPACE] ERROR: missing original launcher: $CORE" >&2
  exit 21
fi

# V5 publishes 1080p/60 I420. Air stream output settings remain independent.
if [[ "${SENSOR_ID:-0}" != "0" || "${WIDTH:-1920}" != "1920" || \
      "${HEIGHT:-1080}" != "1080" || "${FPS:-60}" != "60" ]]; then
  echo "[NEWSPACE] ERROR: V5 requires SENSOR_ID=0 WIDTH=1920 HEIGHT=1080 FPS=60" >&2
  exit 22
fi

# A single, long-lived systemd owner must be active. NEVER restart Argus here.
if ! systemctl is-active --quiet "$UNIT"; then
  echo "[NEWSPACE] ERROR: $UNIT is not active. Check imx477ctl status" >&2
  exit 23
fi

# Camera-first check, bounded to 5 seconds. The camera owner already runs at boot.
ready=0
for _ in 1 2 3 4 5; do
  if /usr/bin/python3 - "$STATE" "$SOCKET" <<'PY_CHECK'
import json, os, pathlib, stat, sys, time
p=pathlib.Path(sys.argv[1]); sock=pathlib.Path(sys.argv[2])
try:
    obj=json.loads(p.read_text())
    age=time.time()-float(obj['updated_unix'])
    if not (0 <= age <= 5 and obj['status']=='healthy'
            and float(obj.get('capture_fps',0)) >= 1
            and float(obj.get('output_fps',0)) >= 1
            and sock.is_socket() and os.access(sock,os.R_OK|os.W_OK)):
        sys.exit(1)
except (OSError, ValueError, TypeError, KeyError):
    sys.exit(1)
PY_CHECK
  then ready=1; break; fi
  sleep 1
done

if [[ "$ready" != "1" ]]; then
  echo "[NEWSPACE] ERROR: V5 is not healthy / its SHM socket is unavailable." >&2
  imx477ctl status || true
  echo "[NEWSPACE] Do NOT start another nvarguscamerasrc. Check the V5 service." >&2
  exit 24
fi

# Detection ON/OFF now opens/closes ONLY an SHM reader. V5 CSI stays alive.
echo "[NEWSPACE] INPUT: always-on IMX477 V5, SHM $SOCKET (I420 1080p60)"
echo "[NEWSPACE] CAMERA OWNER: $UNIT - no Argus restart or CSI re-probe"
echo "[NEWSPACE] AIR controls/AI/tracker/RTSP remain unchanged"
exec "$CORE" "$@"
