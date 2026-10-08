# NEWSPACE AIR PORTABLE R2

The portable runtime is designed so the **AIR publisher never touches X11**.
This matters because an Xlib fatal I/O error terminates the complete process;
moving OpenCV HighGUI to a C++ thread is not sufficient isolation.

## Production

```bash
cd ~/NEWSPACE_AIR_PORTABLE
./run.sh
```

`publisher_node` stays headless and provides:

- CSI capture
- V15.57/CoreLock detection
- optional TensorRT AI controlled from ground
- X/Y telemetry on TCP 5701
- final annotated RTSP/TCP video on 8554 `/main.264`

## Jetson-local final detection display

```bash
cd ~/NEWSPACE_AIR_PORTABLE
DISPLAY=:1 ./run_local_display.sh
```

This keeps `publisher_node` headless and starts an **independent local RTSP viewer**
on `127.0.0.1:8554/main.264`. The local viewer shows the same final annotated
stream sent to ground. If X11 or the viewer fails, the AIR publisher continues.

The local viewer waits/retries while Detection is OFF. The window appears after
the ground station turns Detection ON and the first final frame becomes ready.

## First setup/rebuild on another same-platform Jetson

```bash
cd ~/NEWSPACE_AIR_PORTABLE
./verify_bundle.sh
./setup_once.sh
```

Then run either production or bench mode above.

`setup_once.sh` contains a reliable TensorRT library check and no longer uses the
`ldconfig | grep -q` pattern that could report a false failure under `pipefail`.

## Transfer

After repairing the source Jetson bundle:

```bash
tar -C ~ -czf ~/NEWSPACE_AIR_PORTABLE_R2.tar.gz NEWSPACE_AIR_PORTABLE
scp ~/NEWSPACE_AIR_PORTABLE_R2.tar.gz <user>@<new-jetson-ip>:~/
```

Target:

```bash
tar -xzf ~/NEWSPACE_AIR_PORTABLE_R2.tar.gz -C ~
cd ~/NEWSPACE_AIR_PORTABLE
./verify_bundle.sh
./setup_once.sh
./run.sh
```

TensorRT engine plans should be used on the same Jetson class and matching
JetPack/L4T/CUDA/TensorRT stack.
