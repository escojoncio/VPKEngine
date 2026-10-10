#!/bin/sh
# Translation on demand (vpaot --fragment): the function the static analysis cannot find is
# translated, compiled and loaded when the run first reaches it. usage: run.sh [vpaot]
set -e
D=$(cd "$(dirname "$0")" && pwd); R=$D/../../..; VPAOT=${1:-$R/build/vpaot/vpaot}; B=$R/build/ondemand; mkdir -p "$B"
AS=${AS:-as}; LD=${LD:-ld}; CC=${CC:-cc}; XCC=${XCC:-gcc}
rm -f "$B"/prog_ondemand_*
$XCC -O2 -fno-pie -fno-asynchronous-unwind-tables -fcf-protection=none -fno-stack-protector -ffreestanding -fno-builtin \
     -mno-red-zone -msse4.2 -c -o "$B/prog.o" "$R/tests/aot/roots/prog.c"
$AS --64 -o "$B/start.o" "$R/tests/aot/program/start.s"
$LD -static -s -e _start -Ttext-segment=0x400000 -o "$B/prog.elf" "$B/start.o" "$B/prog.o"
$CC -O2 -I "$R/runtime" -Dvp_main=vp_main_native -c -o "$B/native.o" "$R/tests/aot/roots/prog.c"
"$VPAOT" --elf "$B/prog.elf" --module prog --out "$B/prog.c"
$CC -O2 -frounding-math -rdynamic -I "$R/runtime" -o "$B/host" "$R/runtime/vp_host.c" "$D/host.c" "$B/prog.c" "$B/native.o" -ldl -lm
"$B/host" "$B/prog.elf" "$VPAOT" "$B" "$R/runtime"
