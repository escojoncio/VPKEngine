#!/usr/bin/env bash
# Compiles every golden .metal with Apple's Metal compiler for visionOS. macOS only: on other
# systems it reports a skip and exits 0 (the goldens are still checked textually by CTest).
set -euo pipefail
dir="$(cd "$(dirname "$0")" && pwd)"
if [[ "$(uname)" != Darwin ]]; then
  echo "compile_all: skipped (needs xcrun metal, macOS only)"
  exit 0
fi
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
status=0
for f in "$dir"/*.metal; do
  if xcrun -sdk xros metal -std=metal3.1 -c "$f" -o "$out/$(basename "$f" .metal).air"; then
    echo "ok    $(basename "$f")"
  else
    echo "FAIL  $(basename "$f")"
    status=1
  fi
done
exit $status
