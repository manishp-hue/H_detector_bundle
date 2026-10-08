# H_detector_bundle

Portable Air + Ground setup for live helipad detection, AI verification, telemetry and annotated video streaming.

## 1. Functionality and Usability

### Air / Jetson
- Captures CSI camera at **1920x1080 @ 60 FPS**
- Runs **V15.57 / CoreLock05** helipad detection
- Optional **TensorRT AI** verification controlled from Ground
- Sends **X/Y error, state and FPS telemetry** over TCP
- Streams the **final annotated detection video** over RTSP/TCP
- Air publisher runs headless; optional local Jetson viewer is isolated from the main pipeline

### Ground Station
- Connects to the Air publisher
- Controls **Detection ON/OFF**
- Controls **AI ON/OFF**
- Displays final annotated video
- Displays X/Y error, detection state, AI state and FPS
- Includes a simple bandwidth monitor

## 2. Hardware Details

| Device | Details |
|---|---|
| Air computer | NVIDIA Jetson Orin Nano 8GB |
| Camera | ArduCam B0273 CSI |
| Air RF module | PMDDL 2550AES256 |
| Ground RF module | PMDDL 2550AES256 |
| Ground computer | Ubuntu / WSL Ubuntu 24.04 laptop |
| Jetson IP | `192.168.144.16` |
| Air module IP | `192.168.144.105` |
| Ground module IP | `192.168.144.106` |
| Ground laptop IP | `192.168.144.4` |

## 3. Software Details

### Air Package
```text
~/NEWSPACE_AIR_PORTABLE
```

Main functions:
- `publisher_node` - camera, detection, AI, telemetry and RTSP publisher
- `run.sh` - normal headless production run
- `run_local_display.sh` - optional Jetson-local final detection display
- `setup_once.sh` - dependency check and rebuild
- `verify_bundle.sh` - portable bundle verification

Communication:
```text
Control / Telemetry : TCP 5701
RTSP Video          : TCP 8554
RTSP Path           : /main.264
```

### Ground Package
```text
~/NEWSPACE_GROUND_PORTABLE
```

Main functions:
- `subscriber_node` - GUI, control, telemetry and RTSP receiver
- `run.sh` - start Ground Station
- `setup_once.sh` - dependency check and rebuild
- `verify_bundle.sh` - portable bundle verification
- `monitor_bandwidth.sh` - live RX/TX bandwidth display

## 4. Setup Air and Ground Station

### Air / Jetson

Extract:
```bash
tar -xzf ~/NEWSPACE_AIR_PORTABLE_R2.tar.gz -C ~
cd ~/NEWSPACE_AIR_PORTABLE
```

Verify and build:
```bash
./verify_bundle.sh
./setup_once.sh
```

Run:
```bash
./run.sh
```

Optional Jetson-local view:
```bash
DISPLAY=:1 ./run_local_display.sh
```

> TensorRT engine plans should be used on the same Jetson class with a matching JetPack/L4T/CUDA/TensorRT environment.

### Ground Station

Extract:
```bash
tar -xzf ~/NEWSPACE_GROUND_PORTABLE_FINAL.tar.gz -C ~
cd ~/NEWSPACE_GROUND_PORTABLE
```

Verify and build:
```bash
./verify_bundle.sh
./setup_once.sh
```

Run:
```bash
./run.sh
```

## 5. Final Run / Test Procedure

1. Connect camera -> Jetson -> PMDDL Air -> PMDDL Ground -> Ground laptop.
2. Start Air:
   ```bash
   cd ~/NEWSPACE_AIR_PORTABLE
   ./run.sh
   ```
3. Start Ground:
   ```bash
   cd ~/NEWSPACE_GROUND_PORTABLE
   ./run.sh
   ```
4. Confirm **CONTROL LINK LIVE**.
5. Turn **Detection ON**.
6. Confirm live annotated video is received.
7. Verify X/Y error, Detection state, Processed/Source FPS and live RTSP video.
8. Turn **AI ON** and confirm AI state changes without restarting video.
9. Test **Detection OFF -> ON** again.
10. Check network load if required:
    ```bash
    ./monitor_bandwidth.sh
    ```

## Expected Operating Flow

```text
CSI Camera
    |
    v
Jetson publisher_node
    |
    +-- V15.57 / CoreLock05 Detection
    +-- TensorRT AI
    +-- X/Y + State -> TCP 5701
    +-- Final Annotated Video -> RTSP/TCP 8554
                              |
                              v
                    Ground subscriber_node
                              |
                              +-- Video
                              +-- Detection ON/OFF
                              +-- AI ON/OFF
                              +-- X/Y / FPS / State
```