#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

command -v cmake >/dev/null 2>&1 || { echo "ERROR: cmake is missing. Run ./setup_once.sh" >&2; exit 2; }
command -v pkg-config >/dev/null 2>&1 || { echo "ERROR: pkg-config is missing. Run ./setup_once.sh" >&2; exit 2; }

cmake -S "$ROOT" -B "$ROOT/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_TENSORRT=ON
cmake --build "$ROOT/build" -j"${JOBS:-2}"
echo "Built: $ROOT/build/publisher_node"
