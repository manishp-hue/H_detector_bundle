#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for candidate in \
    "$ROOT/build/subscriber_node" \
    "$ROOT/build/newspace_ground_gui"
do
    if [[ -x "$candidate" ]]; then
        exec "$candidate" "$@"
    fi
done

echo "Ground executable is not built yet."
echo "Run:"
echo "  ./setup_once.sh"
exit 1
