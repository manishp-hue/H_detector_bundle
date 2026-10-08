# NEWSPACE GROUND PORTABLE

Portable ground-station package made from the currently working
`~/newspace_ground_fresh` project.

## Purpose

This folder is the complete ground-side application for the fresh
Newspace Air/Ground architecture.

Ground responsibilities:

- connect to Jetson control TCP port `5701`
- send Detection ON/OFF
- send AI ON/OFF
- receive X/Y and state telemetry
- receive final annotated video from Jetson by RTSP/TCP
- show the Newspace Ground GUI
- keep control handling independent from RTSP recovery

Default Jetson:

- IP: `192.168.144.16`
- control: TCP `5701`
- RTSP: TCP `8554`
- path: `/main.264`

The ground station performs no helipad detection and contains no AI models.

## Folder contents

The bundle contains the tested C++/Qt/GStreamer source, CMake project,
runtime scripts, README, checksum manifest, and optionally the binary from
the original laptop under `prebuilt/`.

Generated build directories, old backups, logs and temporary files are not
included.

## Recommended setup on another Ubuntu laptop / WSL

Extract:

```bash
tar -xzf NEWSPACE_GROUND_PORTABLE_R1.tar.gz -C ~
cd ~/NEWSPACE_GROUND_PORTABLE
```

Verify:

```bash
./verify_bundle.sh
```

Install/check dependencies and rebuild:

```bash
./setup_once.sh
```

Run:

```bash
./run.sh
```

## Normal operating sequence

1. Start the Jetson Air publisher.
2. Start this ground GUI.
3. Wait for `CONTROL LINK LIVE`.
4. Press Detection ON.
5. Air opens CSI, detector starts, and first final annotated frame becomes ready.
6. Ground RTSP receiver connects automatically.
7. AI may then be toggled independently.
8. Detection OFF stops Air vision/video.

## Bandwidth monitor

```bash
./monitor_bandwidth.sh
```

Override Jetson IP if necessary:

```bash
JETSON_IP=192.168.144.16 ./monitor_bandwidth.sh
```

## Important

`prebuilt/subscriber_node`, when present, is only a convenience copy from
the original laptop. On another machine, prefer `./setup_once.sh` so the
application is rebuilt against that machine's Qt and GStreamer libraries.

This portable ground package does not contain any detector model or TensorRT
engine. Those belong only to the Air/Jetson bundle.
