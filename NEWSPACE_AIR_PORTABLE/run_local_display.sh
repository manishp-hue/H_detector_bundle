#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DISPLAY="${DISPLAY:-:1}"

PUB_PID=""
VIEW_PID=""
cleaned=0
cleanup(){
    [[ $cleaned -eq 1 ]] && return
    cleaned=1
    trap - INT TERM EXIT
    if [[ -n "$VIEW_PID" ]]; then kill -TERM "$VIEW_PID" 2>/dev/null || true; fi
    if [[ -n "$PUB_PID" ]]; then kill -INT "$PUB_PID" 2>/dev/null || true; fi
    [[ -n "$VIEW_PID" ]] && wait "$VIEW_PID" 2>/dev/null || true
    [[ -n "$PUB_PID" ]] && wait "$PUB_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "[PORTABLE] starting AIR publisher headless..."
"$ROOT/run.sh" &
PUB_PID=$!

sleep 1
if ! kill -0 "$PUB_PID" 2>/dev/null; then
    wait "$PUB_PID"
    exit $?
fi

echo "[PORTABLE] starting independent Jetson local final-frame viewer..."
"$ROOT/local_viewer.sh" &
VIEW_PID=$!

# Keep this wrapper alive until the publisher exits or Ctrl+C is pressed.
while kill -0 "$PUB_PID" 2>/dev/null; do sleep 1; done
wait "$PUB_PID"
rc=$?
cleanup
exit "$rc"
