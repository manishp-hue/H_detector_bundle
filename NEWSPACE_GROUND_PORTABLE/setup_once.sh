#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "NEWSPACE GROUND portable setup"
echo "Directory: $ROOT"
echo

missing=0

for cmd in cmake pkg-config g++ gst-launch-1.0; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "MISSING command: $cmd"
        missing=1
    fi
done

for pkg in Qt6Widgets Qt6Network gstreamer-1.0 gstreamer-app-1.0; do
    if ! pkg-config --exists "$pkg" 2>/dev/null; then
        echo "MISSING development package: $pkg"
        missing=1
    fi
done

if [[ "$missing" -ne 0 ]]; then
    cat <<'MSG'

Install the normal Ubuntu ground-station dependencies:

  sudo apt update
  sudo apt install -y \
    build-essential cmake pkg-config \
    qt6-base-dev qt6-base-dev-tools \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-libav

Then run:

  ./setup_once.sh

MSG
    exit 2
fi

echo "Dependency check: PASS"
echo

"$ROOT/build.sh"

echo
echo "Setup complete."
echo "Run:"
echo "  ./run.sh"
