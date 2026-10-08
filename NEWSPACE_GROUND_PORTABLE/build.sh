#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS="${JOBS:-$(nproc)}"

cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build" -j"$JOBS"

BIN=""
for candidate in \
    "$ROOT/build/subscriber_node" \
    "$ROOT/build/newspace_ground_gui"
do
    if [[ -x "$candidate" ]]; then
        BIN="$candidate"
        break
    fi
done

if [[ -z "$BIN" ]]; then
    echo "ERROR: build finished but no supported ground executable was found." >&2
    echo "Expected one of:" >&2
    echo "  $ROOT/build/subscriber_node" >&2
    echo "  $ROOT/build/newspace_ground_gui" >&2
    exit 1
fi

echo
echo "Built: $BIN"
