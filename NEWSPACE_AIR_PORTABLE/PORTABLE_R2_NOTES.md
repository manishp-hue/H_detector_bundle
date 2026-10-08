# Portable R2 safety fix

Two issues were corrected:

1. **Jetson local X11/HighGUI can terminate the whole publisher process.**
   A detached C++ thread does not protect against an Xlib fatal I/O error because
   Xlib terminates the process. `publisher_node` now runs permanently headless.
   Local viewing is a separate localhost RTSP client (`local_viewer.sh`). If that
   viewer crashes, control/detection/RTSP remain alive.

2. **False `libnvinfer.so` failure in `setup_once.sh`.**
   The old `ldconfig -p | grep -q` test was used together with `pipefail`; after a
   successful early grep match, `ldconfig` could receive SIGPIPE and make the
   pipeline look failed. The check now captures `ldconfig` output first and also
   checks the normal Jetson library path.

## Production run

```bash
cd ~/NEWSPACE_AIR_PORTABLE
./run.sh
```

The ground station controls Detection and AI. No X11 process is involved.

## Bench run with Jetson local final detection video

```bash
cd ~/NEWSPACE_AIR_PORTABLE
DISPLAY=:1 ./run_local_display.sh
```

This starts:

- `publisher_node` headless (authoritative AIR process)
- one independent local RTSP viewer on `127.0.0.1:8554/main.264`

The local window appears only after the ground station turns Detection ON and the
first final annotated frame is ready.

## Stop

Press `Ctrl+C` in the `run_local_display.sh` terminal. The wrapper terminates the
viewer and sends SIGINT to `publisher_node`.
