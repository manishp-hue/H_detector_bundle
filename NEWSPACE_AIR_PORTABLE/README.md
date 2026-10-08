# NEWSPACE AIR PORTABLE

Portable single-folder Jetson AIR runtime.

## What is inside

Only the airborne files required by the current Fresh/R4 architecture are kept:

- `src/publisher_node.cpp` - one-process AIR publisher/control/RTSP runtime.
- `tracker/` - the copied V15.57/CoreLock detector sources and required headers.
- `models/gdino/grounding_dino_fixed.plan` - GDINO TensorRT engine.
- `models/sam2_box.plan` - SAM2 TensorRT engine.
- `build.sh` - rebuilds `publisher_node` on the target Jetson.
- `run.sh` - normal launcher.
- `run_local_display.sh` - final Jetson detection window enabled.
- `run_headless.sh` - no Jetson display.
- `setup_once.sh` - dependency check + clean target-side rebuild.
- `verify_bundle.sh` / `MANIFEST.sha256` - integrity check.

There is no VisionGateway, no intermediate video publisher/subscriber, no image SHM transport, and no ZMQ video bridge in this bundle.

## Runtime behavior

`publisher_node` is the only Newspace AIR application process.

- Startup: control TCP listener only.
- Ground `Detection ON`: opens CSI 1920x1080@60, creates detector, then produces final annotated frames.
- The exact final annotated frame is available to the optional Jetson local window and the RTSP/TCP stream.
- Ground `AI ON/OFF`: gates AI assistance without restarting the camera/video path.
- Ground `Detection OFF`: closes camera/detector/AI/RTSP media; control listener remains available.
- X/Y comes from detector guidance authority, not a second ground-side calculation.

Default ports:

- Control + X/Y telemetry TCP: `5701`
- RTSP/TCP video: `8554`
- URL: `rtsp://JETSON_IP:8554/main.264`

## First use on another Jetson

The target should be the same class of Jetson and should preferably use the same JetPack/L4T, CUDA and TensorRT versions as the source Jetson. TensorRT `.plan` engines are not universally portable across different Jetson models or TensorRT/CUDA versions.

Copy/extract this folder to the target Jetson, then:

```bash
cd ~/NEWSPACE_AIR_PORTABLE
./verify_bundle.sh
./setup_once.sh
```

`setup_once.sh` performs a clean compile against the target Jetson's installed libraries.

## Run with local final detection display

```bash
cd ~/NEWSPACE_AIR_PORTABLE
DISPLAY=:1 LOCAL_DISPLAY=on VIDEO_BITRATE_KBPS=3000 ./run.sh
```

Equivalent convenience command:

```bash
cd ~/NEWSPACE_AIR_PORTABLE
./run_local_display.sh
```

The local display is diagnostic only and uses the final annotated detector frame. The R4 design keeps HighGUI/X11 off the main vision/control thread, so the display cannot back-pressure the camera/detector/RTSP path.

## Run headless

```bash
cd ~/NEWSPACE_AIR_PORTABLE
LOCAL_DISPLAY=off VIDEO_BITRATE_KBPS=3000 ./run.sh
```

or:

```bash
./run_headless.sh
```

## Default camera/stream parameters

```text
CSI sensor       0
Detection input  1920x1080 @ 60 FPS
Ground stream    1280x720 @ max 30 FPS
Video bitrate    3000 kbps
RTSP transport   TCP
Control port     5701/TCP
RTSP port        8554/TCP
```

Override without editing code, for example:

```bash
SENSOR_ID=0 \
WIDTH=1920 HEIGHT=1080 FPS=60 \
STREAM_WIDTH=1280 STREAM_HEIGHT=720 STREAM_FPS=30 \
VIDEO_BITRATE_KBPS=3000 \
CONTROL_PORT=5701 RTSP_PORT=8554 \
LOCAL_DISPLAY=off \
./run.sh
```

## Expected startup

Before Detection ON:

```text
[AIR] publisher_node ready. Detection starts only after DET 1.
[AIR] RTSP rtsp://<jetson>:8554/main.264 (TCP only)
[CONTROL] TCP listening on 5701
```

After Detection ON, the important AIR checkpoints are:

```text
[VISION] DET ON: opening CSI and detector
[AI] engines READY
[VISION] capture loop START; waiting for first CSI frame
[VISION] first CSI frame RECEIVED
[VISION] first detector result READY
[VISION] first FINAL annotated frame COMPOSED
[RTSP] first final frame READY; remote RTSP may connect now
```

## Stop

Use `Ctrl+C` in the terminal running `run.sh`.

This portable bundle intentionally does not install a boot service. Manual launch is the default deployment policy.
