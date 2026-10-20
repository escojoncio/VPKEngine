#!/usr/bin/env bash
# Compiles every golden .metal with Apple's Metal compiler for the visionOS device SDK, then links
# them all into one .metallib. Needs xcrun (a macOS host); elsewhere it reports a skip and exits 0
# (the goldens are still checked textually by CTest).
# Usage: compile_all.sh [out_dir]   (out_dir keeps goldens.metallib; default: a temp dir)
set -euo pipefail
dir="$(cd "$(dirname "$0")" && pwd)"
if [[ "$(uname)" != Darwin ]]; then
  echo "compile_all: skipped (needs xcrun metal on a macOS host)"
  exit 0
fi
if [[ $# -gt 0 ]]; then
  out="$1"
  mkdir -p "$out"
else
  out="$(mktemp -d)"
  trap 'rm -rf "$out"' EXIT
fi
status=0
airs=()
# The MSL goldens, plus the GCN lifter's (gpu/shader/gcn/tests/lift), prefixed to keep names apart.
lift="$dir/../../gcn/tests/lift"
# And the presenter's own shaders (gpu/visionos/shaders: VPKTAA...), entry points with their own names.
present="$dir/../../../visionos/shaders"
for f in "$dir"/*.metal "$lift"/*.metal "$present"/*.metal; do
  [[ -e "$f" ]] || continue
  n="$(basename "$f" .metal)"
  [[ "$f" == "$lift"/* ]] && n="lift_$n"
  # Every golden names its entry point vp_main: rename per file so they can share one library.
  if xcrun -sdk xros metal -std=metal3.1 -Dvp_main="vp_$n" -c "$f" -o "$out/$n.air"; then
    echo "ok    $n.metal"
    airs+=("$out/$n.air")
  else
    echo "FAIL  $n.metal"
    status=1
  fi
done
if [[ $status -eq 0 ]]; then
  if xcrun -sdk xros metallib "${airs[@]}" -o "$out/goldens.metallib"; then
    echo "ok    goldens.metallib (${#airs[@]} shaders)"
  else
    echo "FAIL  metallib link"
    status=1
  fi
fi
exit $status
