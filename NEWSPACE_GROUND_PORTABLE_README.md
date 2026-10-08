# NEWSPACE GROUND PORTABLE

This bundle is generated from the **currently working** `~/newspace_ground_fresh`
directory and is intended to be transferable to another Ubuntu/WSL laptop.

It contains the tested ground C++/Qt/GStreamer source plus build/run/setup
scripts. It does **not** contain Air-side detector code or TensorRT models.

Normal destination:

```text
~/NEWSPACE_GROUND_PORTABLE/
```

Portable archive:

```text
~/NEWSPACE_GROUND_PORTABLE_R1.tar.gz
```

On a new laptop:

```bash
tar -xzf ~/NEWSPACE_GROUND_PORTABLE_R1.tar.gz -C ~
cd ~/NEWSPACE_GROUND_PORTABLE
./verify_bundle.sh
./setup_once.sh
./run.sh
```

Default communication:

```text
Jetson IP    192.168.144.16
Control TCP  5701
RTSP/TCP     8554
RTSP path    /main.264
```

For bandwidth:

```bash
./monitor_bandwidth.sh
```
