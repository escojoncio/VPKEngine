#!/bin/sh
# Whole-program test through the ELF path. usage: run.sh [vpaot]
set -e
D=$(cd "$(dirname "$0")" && pwd); R=$D/../../..; VPAOT=${1:-$R/build/vpaot/vpaot}; B=$R/build/program; mkdir -p "$B"
AS=${AS:-as}; LD=${LD:-ld}; CC=${CC:-cc}; XCC=${XCC:-gcc}
# The guest program: static, non-PIE, no libc, Jaguar-class ISA.
$XCC -O2 -fno-pie -fno-asynchronous-unwind-tables -fcf-protection=none -fno-stack-protector -ffreestanding -fno-builtin \
     -mno-red-zone -msse4.2 -mpopcnt -c -o "$B/prog.o" "$D/prog.c"
$AS --64 -o "$B/start.o" "$D/start.s"
$LD -static -e _start -Ttext-segment=0x400000 -o "$B/prog.elf" "$B/start.o" "$B/prog.o"
"$VPAOT" --elf "$B/prog.elf" --out "$B/prog.c" --stats "$B/stats.json" --scan-data
# The host: the translation + the same program compiled natively for the expected value.
$CC -O2 -I "$R/runtime" -Dvp_main=vp_main_native -c -o "$B/native.o" "$D/prog.c"
$CC -O2 -I "$R/runtime" -o "$B/host" "$R/runtime/vp_host.c" "$D/host.c" "$B/prog.c" "$B/native.o" -lm
"$B/host" "$B/prog.elf"
