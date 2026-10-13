# CAMERA GUARD R3

- only one publisher_node instance can run
- TensorRT engine loading completes before CSI opens
- CSI opens only once per Detection session
- CSI open/read failure latches Detection OFF instead of auto-reopening
- no forced RTSP client reset at first-frame readiness
- recover_camera.sh can restart Argus without rebooting the Jetson
