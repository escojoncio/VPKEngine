#!/bin/sh
# Real binaries at scale: the system's libstdc++ (about a million instructions, thousands of
# landing pads, jump tables, hot/cold split functions) translated whole as a --pic module split
# into units, and every unit compiled. Any C error (a missing label, a cut line, a bad macro)
# fails. usage: run.sh [vpaot] [library]
set -e
D=$(cd "$(dirname "$0")" && pwd); R=$D/../../..; VPAOT=${1:-$R/build/vpaot/vpaot}; B=$R/build/realbin; mkdir -p "$B"
LIB=${2:-$(ls /usr/lib/x86_64-linux-gnu/libstdc++.so.6 2>/dev/null || true)}
[ -n "$LIB" ] || { echo "no x86-64 libstdc++ here; skipped"; exit 0; }
CC=${CC:-cc}
rm -f "$B"/x_*.c
"$VPAOT" --elf "$LIB" --pic --module x --split 400 --out "$B/x.c" --stats "$B/stats.json"
ls "$B"/x_*.c | xargs -P "$(nproc 2>/dev/null || echo 2)" -n 1 $CC -fsyntax-only -w -frounding-math -I "$R/runtime"
echo "$(ls "$B"/x_*.c | wc -l) units of $(basename "$LIB") compile"
