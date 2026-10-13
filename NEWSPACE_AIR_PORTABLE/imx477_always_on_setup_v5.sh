#!/usr/bin/env bash
# One-file installer: Newspace IMX477 always-on camera monitor + systemd.
# Does not start or restart your existing camera session during installation.
set -euo pipefail
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
cat > "$scratch/imx477_camera_service.py" <<'END_IMX477_PYTHON_577'
#!/usr/bin/env python3
"""Newspace IMX477: one long-lived, headless camera owner for systemd.

Jetson Orin Nano, JetPack/L4T R36.5.2, IMX477-A on CAM0.
One owner; 1080p60; GStreamer shared-memory preview is a separate client.
Never restarts Argus automatically after a capture error.

This code has been syntax-checked off-device; test on the Jetson before flight use.
"""

import fcntl
import json
import logging
import os
from pathlib import Path
import signal
import socket
import sys
import time

RUN = Path('/run/newspace-imx477')
STATUS = RUN / 'status.json'
SOCKET = RUN / 'frames.sock'
LOCK = Path('/home/gw_jetson_01/.local/state/newspace-imx477/systemd_owner.lock')
DEVICE = Path('/dev/video0')
PREVIOUS_LOCKS = [
    Path('/home/gw_jetson_01/.local/state/newspace-imx477/guard_v3.lock'),
    Path('/home/gw_jetson_01/.local/state/newspace-imx477/guard_v4.lock'),
]
FRAME_TIMEOUT = 3.0  # seconds since last buffer at BOTH probes

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
LOG = logging.getLogger('imx477-camera')


def sd_notify(message):
    """Send READY/STATUS to systemd, without an external Python dependency."""
    target = os.getenv('NOTIFY_SOCKET')
    if not target:
        return
    if target.startswith('@'):
        target = '\0' + target[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as connection:
            connection.sendto(message.encode('utf-8'), target)
    except OSError as exc:
        LOG.warning('systemd notification failed: %s', exc)


def claim_legacy_guard_locks():
    """Own both V3/V4 guard locks so old launchers cannot open CSI in parallel."""
    held = []
    try:
        for path in PREVIOUS_LOCKS:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                raise RuntimeError('Existing V3/V4 camera owner active: %s' % path)
            held.append(fd)
        return held
    except Exception:
        for fd in held:
            os.close(fd)
        raise


def check_for_other_owner():
    # Old V3/V4 guards are excluded by holding their lock files in this process.
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            argv = (process / 'cmdline').read_bytes().replace(b'\0', b' ')
        except (OSError, PermissionError):
            continue
        if b'gst-launch-1.0' in argv and b'nvarguscamerasrc' in argv:
            raise RuntimeError('Another gst-launch process uses nvarguscamerasrc (PID %s)' % process.name)


class CameraOwner:
    def __init__(self):
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import GLib, Gst

        self.GLib = GLib
        self.Gst = Gst
        Gst.init(None)
        self.loop = GLib.MainLoop()
        self.pipeline = None
        self.bus = None
        self.shutting_down = False
        self.cleanup_done = False
        self.error = ''
        self.capture_frames = 0
        self.output_frames = 0
        self.capture_time = None
        self.output_time = None
        self.last_tick = time.monotonic()
        self.last_capture_frames = 0
        self.last_output_frames = 0
        self.fps = 0.0
        self.output_fps = 0.0
        self.previous_health = None
        self.started_at = time.time()
        self.exit_code = 0

    def capture_probe(self, _pad, _info):
        # Keep pad-probe callbacks extremely short: they execute in a streaming thread.
        self.capture_frames += 1
        self.capture_time = time.monotonic()
        return self.Gst.PadProbeReturn.OK

    def output_probe(self, _pad, _info):
        self.output_frames += 1
        self.output_time = time.monotonic()
        return self.Gst.PadProbeReturn.OK

    def create_pipeline(self):
        gst = self.Gst
        for name in ('nvarguscamerasrc', 'identity', 'queue', 'nvvidconv', 'shmsink'):
            if not gst.ElementFactory.find(name):
                raise RuntimeError('Missing GStreamer element: %s' % name)
        # Crucial: no X11, DISPLAY, nv3dsink or preview in the camera owner.
        desc = (
            'nvarguscamerasrc sensor-id=0 sensor-mode=1 ! '
            'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=60/1,format=NV12 ! '
            'identity name=capture_tap silent=true ! '
            'queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! '
            'nvvidconv ! '
            'video/x-raw,width=1920,height=1080,framerate=60/1,format=I420 ! '
            'queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! '
            'shmsink name=output_sink socket-path=%s shm-size=67108864 wait-for-connection=false sync=false'
        ) % SOCKET
        self.pipeline = gst.parse_launch(desc)
        capture = self.pipeline.get_by_name('capture_tap')
        output = self.pipeline.get_by_name('output_sink')
        if not capture or not output:
            raise RuntimeError('Cannot attach camera frame probes')
        capture.get_static_pad('sink').add_probe(gst.PadProbeType.BUFFER, self.capture_probe)
        output.get_static_pad('sink').add_probe(gst.PadProbeType.BUFFER, self.output_probe)
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect('message', self.on_bus)
        changed = self.pipeline.set_state(gst.State.PLAYING)
        if changed == gst.StateChangeReturn.FAILURE:
            raise RuntimeError('GStreamer refused the PLAYING state')
        LOG.info('Headless Argus session started; 1920x1080 @ 60 FPS, CAM0 IMX477-A')

    def on_bus(self, _bus, msg):
        gst = self.Gst
        if msg.type == gst.MessageType.ERROR:
            err, details = msg.parse_error()
            self.error = '%s: %s%s' % (msg.src.get_name(), err.message,
                                       ' (' + details[:150] + ')' if details else '')
            LOG.error('Camera pipeline error: %s', self.error)
            # Do not reopen the camera. Keep the monitor process alive and report BAD.
            # Only explicit systemctl restart should close and reopen this session.
        elif msg.type == gst.MessageType.EOS:
            if self.shutting_down:
                LOG.info('EOS acknowledged; releasing camera')
                self.finish_shutdown()
            else:
                self.error = 'Unexpected EOS from camera'
                LOG.error(self.error)
        elif msg.type == gst.MessageType.WARNING:
            warn, details = msg.parse_warning()
            LOG.warning('%s: %s; %s', msg.src.get_name(), warn.message, details)

    def write_health(self):
        now = time.monotonic()
        elapsed = max(0.001, now - self.last_tick)
        self.fps = (self.capture_frames - self.last_capture_frames) / elapsed
        self.output_fps = (self.output_frames - self.last_output_frames) / elapsed
        self.last_tick = now
        self.last_capture_frames = self.capture_frames
        self.last_output_frames = self.output_frames

        camera_present = DEVICE.exists()
        source_age = None if self.capture_time is None else round(now - self.capture_time, 2)
        output_age = None if self.output_time is None else round(now - self.output_time, 2)
        if not camera_present:
            health = 'disconnected'
            reason = '/dev/video0 is absent; check camera enumeration'
        elif (source_age is not None and source_age <= FRAME_TIMEOUT and
              output_age is not None and output_age <= FRAME_TIMEOUT):
            health = 'healthy'
            reason = 'fresh camera frames and output frames detected'
        else:
            health = 'bad-pipeline'
            reason = self.error or (
                'waiting for frames' if self.capture_time is None
                else 'stale capture/output frames; check Argus and CSI logs'
            )

        if self.shutting_down:
            reason = 'controlled camera shutdown in progress'

        data = {
            'status': health,
            'reason': reason,
            'video_device_present': camera_present,
            'capture_fps': round(self.fps, 1),
            'output_fps': round(self.output_fps, 1),
            'last_capture_age_seconds': source_age,
            'last_output_age_seconds': output_age,
            'capture_frames': self.capture_frames,
            'output_frames': self.output_frames,
            'last_error': self.error,
            'pid': os.getpid(),
            'updated_unix': time.time(),
            'uptime_seconds': round(time.time() - self.started_at, 1),
        }
        # Status is written atomically; CLI never sees half-written JSON.
        try:
            temporary = STATUS.with_suffix('.tmp')
            temporary.write_text(json.dumps(data, sort_keys=True) + '\n')
            os.replace(temporary, STATUS)
        except OSError as exc:
            LOG.error('Could not write camera status: %s', exc)

        age_text = 'never' if source_age is None else '%.1fs' % source_age
        guidance = {
            'healthy': 'continue operation',
            'bad-pipeline': 'check journal; one controlled restart if needed',
            'disconnected': 'power off before checking CSI hardware',
        }[health]
        sd_notify('STATUS=%s | input %.1f FPS | output %.1f FPS | age %s | %s' % (
            health, self.fps, self.output_fps, age_text, guidance))
        if health != self.previous_health:
            LOG.warning('CAMERA STATUS: %s | %s', health, reason)
            self.previous_health = health
        return not self.shutting_down

    def request_shutdown(self, reason):
        if self.shutting_down:
            return False
        self.shutting_down = True
        LOG.info('Shutdown requested (%s): try EOS, then NULL; no force kill', reason)
        sd_notify('STOPPING=1\nSTATUS=stopping gracefully: EOS then NULL')
        if self.pipeline:
            try:
                sent = self.pipeline.send_event(self.Gst.Event.new_eos())
                LOG.info('EOS request accepted: %s', sent)
            except Exception as exc:
                LOG.warning('EOS request failed: %s', exc)
            self.GLib.timeout_add(1500, self.finish_shutdown)
        else:
            self.finish_shutdown()
        return False

    def finish_shutdown(self):
        if self.cleanup_done:
            return False
        self.cleanup_done = True
        LOG.info('Setting camera pipeline to NULL (releases Argus resources)')
        if self.pipeline:
            try:
                changed = self.pipeline.set_state(self.Gst.State.NULL)
                state_result, state, _ = self.pipeline.get_state(8 * self.Gst.SECOND)
                LOG.info('Shutdown result=%s state=%s initial=%s',
                         state_result.value_nick, state.value_nick, changed.value_nick)
                if state != self.Gst.State.NULL:
                    self.exit_code = 8
                    LOG.error('GStreamer did NOT confirm NULL; manual inspection needed')
            except Exception:
                self.exit_code = 8
                LOG.exception('Exception releasing camera')
            if self.bus:
                try:
                    self.bus.remove_signal_watch()
                except Exception:
                    pass
            self.pipeline = None
            self.bus = None
        LOG.info('Camera-owner cleanup completed; systemd waits 5s before restart')
        self.loop.quit()
        return False

    def run(self):
        self.GLib.timeout_add_seconds(1, self.write_health)

        def stop_signal(sig, _frame):
            # systemd KillSignal=SIGINT is the same logical stop as Ctrl+C.
            self.GLib.idle_add(self.request_shutdown, signal.Signals(sig).name)

        signal.signal(signal.SIGINT, stop_signal)
        signal.signal(signal.SIGTERM, stop_signal)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

        try:
            if not DEVICE.exists():
                self.error = 'IMX477 video device /dev/video0 not detected'
                LOG.error(self.error)
            else:
                try:
                    self.create_pipeline()
                except Exception as exc:
                    self.error = str(exc)
                    LOG.exception('Camera startup failed; staying online for diagnostics')
            sd_notify('READY=1\nSTATUS=bad-pipeline | starting / waiting for frames')
            self.write_health()
            self.loop.run()
        finally:
            if not self.cleanup_done:
                self.finish_shutdown()


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(LOCK), os.O_RDWR | os.O_CREAT, 0o600)
    legacy_lock_fds = []
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            LOG.error('Camera owner lock is held; refusing duplicate CSI capture')
            return 2
        legacy_lock_fds = claim_legacy_guard_locks()
        check_for_other_owner()
        owner = CameraOwner()
        owner.run()
        return owner.exit_code
    except Exception:
        LOG.exception('Camera service could not initialize')
        return 3
    finally:
        for fd in legacy_lock_fds:
            os.close(fd)
        os.close(lock_fd)


if __name__ == '__main__':
    sys.exit(main())
END_IMX477_PYTHON_577
cat > "$scratch/imx477-camera.service" <<'END_IMX477_UNIT_577'
[Unit]
Description=Newspace IMX477 CAM0 Always-On Headless Camera
Documentation=https://gstreamer.freedesktop.org/documentation/
After=nvargus-daemon.service
Wants=nvargus-daemon.service

[Service]
Type=notify
NotifyAccess=main
User=gw_jetson_01
WorkingDirectory=/home/gw_jetson_01
RuntimeDirectory=newspace-imx477
RuntimeDirectoryMode=0700
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 -u /opt/newspace-imx477/imx477_camera_service.py
# systemd sends SIGINT, just like Ctrl+C. Python handles SIGINT and SIGTERM.
KillSignal=SIGINT
# Allow EOS -> NULL -> resource release; do not kill during clean shutdown.
TimeoutStopSec=35s
SendSIGKILL=no
# On an explicit restart, wait after the old owner exits before reopening Argus.
ExecStopPost=/usr/bin/sleep 5
# Deliberately do not auto-restart a bad Argus session.
Restart=no

[Install]
WantedBy=multi-user.target
END_IMX477_UNIT_577
cat > "$scratch/imx477ctl" <<'END_IMX477_CTL_577'
#!/usr/bin/env bash
# Simple operator interface to the always-on Newspace IMX477 camera.
set -euo pipefail
SVC=imx477-camera.service
STATE=/run/newspace-imx477/status.json
FRAMES=/run/newspace-imx477/frames.sock
cmd="${1:-status}"
case "$cmd" in
  status)
    /usr/bin/python3 - "$STATE" "$SVC" <<'PY'
import json
import pathlib
import subprocess
import sys
import time
status_path=pathlib.Path(sys.argv[1]); unit=sys.argv[2]
p=subprocess.run(['systemctl','is-active',unit], capture_output=True, text=True)
active=p.stdout.strip() or 'unknown'
print('service:',active)
if not status_path.is_file() or active != 'active':
    print('status: service-stopped' if active != 'active' else 'status: bad-pipeline')
    print('suggestion: Check systemctl status %s and journalctl -u %s.' % (unit,unit))
    sys.exit(1)
try:
    data=json.loads(status_path.read_text())
except (OSError,ValueError) as e:
    print('status: bad-pipeline')
    print('suggestion: Status data could not be read:',e)
    sys.exit(2)
if time.time()-data.get('updated_unix',0)>5:
    print('status: bad-pipeline')
    print('reason: camera-owner heartbeat is stale')
    print('suggestion: Check service logs; do not repeatedly restart Argus.')
    sys.exit(2)
state=data.get('status','bad-pipeline')
print('status:',state)
print('reason:',data.get('reason','unknown'))
print('input FPS: %.1f | output FPS: %.1f' % (data.get('capture_fps',0),data.get('output_fps',0)))
print('last capture frame age:',data.get('last_capture_age_seconds'),'seconds')
if data.get('last_error'):
    print('last pipeline error:',data['last_error'])
if state=='healthy':
    print('suggestion: No action needed. Keep the owner running; preview can be opened or closed.')
elif state=='disconnected':
    print('suggestion: Sensor is not enumerated. Shut down and remove power before checking the CSI ribbon.')
else:
    print('suggestion: Check sudo journalctl -u nvargus-daemon -n 60 and')
    print('            sudo journalctl -u imx477-camera -n 60.')
    print('            If necessary, try ONE sudo systemctl restart imx477-camera.')
    print('            Persistent PIX_SHORT/EMB_SHORT may need a cold power cycle or hardware diagnosis.')
PY
    ;;
  preview)
    if [ ! -S "$FRAMES" ]; then
      echo "Camera SHM socket missing. Run: imx477ctl status" >&2
      exit 2
    fi
    export DISPLAY="${DISPLAY:-:1}"
    echo "Preview on DISPLAY=$DISPLAY. Closing this window will NOT stop the camera owner."
    exec gst-launch-1.0 -f -e \
      shmsrc socket-path="$FRAMES" is-live=true do-timestamp=true ! \
      'video/x-raw,width=1920,height=1080,framerate=60/1,format=I420' ! \
      queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! \
      nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! \
      nv3dsink sync=false
    ;;
  logs)
    sudo journalctl -u "$SVC" -n 80 --no-pager -l
    ;;
  restart)
    echo 'Requesting graceful SIGINT -> EOS -> NULL, then 5-second cooldown and restart.'
    sudo systemctl restart "$SVC"
    ;;
  *)
    echo "Usage: imx477ctl {status|preview|logs|restart}" >&2
    exit 2
    ;;
esac
END_IMX477_CTL_577
missing=()
for pkg in python3-gi python3-gst-1.0 gir1.2-gstreamer-1.0; do
  if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q 'install ok installed'; then
    missing+=("$pkg")
  fi
done
if ((${#missing[@]})); then
  sudo apt update
  sudo apt install -y "${missing[@]}"
fi
sudo install -d -m 0755 /opt/newspace-imx477
sudo install -m 0755 "$scratch/imx477_camera_service.py" /opt/newspace-imx477/imx477_camera_service.py
sudo install -m 0755 "$scratch/imx477ctl" /usr/local/bin/imx477ctl
sudo install -m 0644 "$scratch/imx477-camera.service" /etc/systemd/system/imx477-camera.service
sudo systemctl daemon-reload
sudo systemctl enable imx477-camera.service
printf '
=== IMX477 V5 INSTALLED ===
'
printf 'Auto-start is ENABLED for subsequent boots. It was NOT started now.
'
printf 'Keep old camera owner V4 and new service from running simultaneously.
'
printf 'Recommended clean transition: sudo poweroff, disconnect/reconnect Jetson power.
'
printf 'After power-on: imx477ctl status
'
printf 'Inspect: sudo systemctl status imx477-camera
'
printf 'Preview: DISPLAY=:1 imx477ctl preview
'
printf 'Restart (graceful release + 5s cooldown): imx477ctl restart
'
