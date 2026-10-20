#!/usr/bin/env bash
# The MSL of every differential program (diff/programs*.txt, lifted by a host build of vpgcn_diff)
# compiled with Apple's Metal compiler for the headset: the emitter on random op combinations and
# control flow. Needs xcrun (macOS); elsewhere it reports a skip.
# Usage: compile_msl.sh path/to/vpgcn_diff [out_dir]
set -euo pipefail
dir="$(cd "$(dirname "$0")" && pwd)"
diff_bin="$1"
if [[ "$(uname)" != Darwin ]]; then
  echo "compile_msl: skipped (needs xcrun metal on a macOS host)"
  exit 0
fi
out="${2:-$(mktemp -d)}"
xcrun -sdk xros metal --version | head -1
mkdir -p "$out"
total=0
failed=0
for set in programs programs_cf; do
  d="$out/$set"
  rm -rf "$d" && mkdir -p "$d"
  "$diff_bin" "$dir/$set.txt" --emit-msl "$d" | tail -1
  # Parallel compiles; a failure leaves NAME.err.
  # (no -I: macOS xargs limits its replacement size; each file is $1 of its own sh)
  find "$d" -name '*.metal' -print0 | xargs -0 -n 1 -P "$(sysctl -n hw.ncpu)" sh -c \
    'xcrun -sdk xros metal -std=metal3.1 -c "$1" -o "$1.air" 2> "$1.log" || mv "$1.log" "$1.err"; exit 0' _ \
    || echo "compile_msl: xargs exited with $?"
  n=$(find "$d" -name '*.metal' | wc -l | tr -d ' ')
  f=$(find "$d" -name '*.err' | wc -l | tr -d ' ')
  echo "$set: $n programs, $f Metal errors"
  total=$((total + n))
  failed=$((failed + f))
  for e in $(find "$d" -name '*.err' | sort | head -5); do
    echo "--- $(basename "$e" .metal.err)"
    head -20 "$e"
  done
done
echo "compile_msl: $total programs, $failed failed"
[[ $failed -eq 0 ]]
