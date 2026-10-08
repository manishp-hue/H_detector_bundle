#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Verifying NEWSPACE GROUND bundle..."
echo

fail=0

req() {
    if [[ -e "$ROOT/$1" ]]; then
        printf "PASS  %s\n" "$1"
    else
        printf "FAIL  %s\n" "$1"
        fail=1
    fi
}

req CMakeLists.txt
req build.sh
req setup_once.sh
req run.sh
req README.md

# At least one C++ source must exist.
if find "$ROOT" -path "$ROOT/build" -prune -o -type f \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \) -print -quit | grep -q .; then
    echo "PASS  C++ source"
else
    echo "FAIL  no C++ source found"
    fail=1
fi

# Print the detected source tree for easy audit.
echo
echo "Portable source files:"
find "$ROOT" \
    -path "$ROOT/build" -prune -o \
    -path "$ROOT/prebuilt" -prune -o \
    -type f -print | sed "s#^$ROOT/##" | sort

echo
if [[ "$fail" -ne 0 ]]; then
    echo "Bundle verification: FAIL"
    exit 1
fi
echo "Bundle verification: PASS"
