#!/bin/sh
# Missing entry points: translate, run (the runtime logs the entry it lacked), translate again
# with --roots on that log, run again. usage: run.sh [vpaot]
set -e
D=$(cd "$(dirname "$0")" && pwd); R=$D/../../..; VPAOT=${1:-$R/build/vpaot/vpaot}; B=$R/build/roots; mkdir -p "$B"
AS=${AS:-as}; LD=${LD:-ld}; CC=${CC:-cc}; XCC=${XCC:-gcc}
$XCC -O2 -fno-pie -fno-asynchronous-unwind-tables -fcf-protection=none -fno-stack-protector -ffreestanding -fno-builtin \
     -mno-red-zone -msse4.2 -c -o "$B/prog.o" "$D/prog.c"
$AS --64 -o "$B/start.o" "$R/tests/aot/program/start.s"
$LD -static -s -e _start -Ttext-segment=0x400000 -o "$B/prog.elf" "$B/start.o" "$B/prog.o"
$CC -O2 -I "$R/runtime" -Dvp_main=vp_main_native -c -o "$B/native.o" "$D/prog.c"
rm -f "$B/missing.txt"
"$VPAOT" --elf "$B/prog.elf" --module prog --out "$B/prog1.c"
$CC -O2 -frounding-math -I "$R/runtime" -o "$B/host1" "$R/runtime/vp_host.c" "$R/tests/aot/program/host.c" "$B/prog1.c" "$B/native.o" -lm
if VPENGINE_MISSING_LOG="$B/missing.txt" "$B/host1" "$B/prog.elf"; then echo "expected the first translation to miss the hidden function"; exit 1; fi
echo "missing entries logged:"; cat "$B/missing.txt"
"$VPAOT" --elf "$B/prog.elf" --module prog --out "$B/prog2.c" --roots "$B/missing.txt"
$CC -O2 -frounding-math -I "$R/runtime" -o "$B/host2" "$R/runtime/vp_host.c" "$R/tests/aot/program/host.c" "$B/prog2.c" "$B/native.o" -lm
"$B/host2" "$B/prog.elf"
