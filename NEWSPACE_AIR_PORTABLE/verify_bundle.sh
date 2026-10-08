#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

required=(
  CMakeLists.txt
  src/publisher_node.cpp
  tracker/include/tracker.hpp
  tracker/include/vision.hpp
  tracker/include/trt_runtime.hpp
  tracker/src/vision.cpp
  tracker/src/tracker.cpp
  tracker/src/trt_runtime.cpp
  models/gdino/grounding_dino_fixed.plan
  models/sam2_box.plan
  build.sh
  run.sh
)

for f in "${required[@]}"; do
    [[ -e "$f" ]] || { echo "FAIL missing: $f"; exit 2; }
done

if [[ -f MANIFEST.sha256 ]]; then
    sha256sum -c MANIFEST.sha256
else
    echo "WARNING: MANIFEST.sha256 is absent"
fi

echo "Portable AIR bundle verification: PASS"
