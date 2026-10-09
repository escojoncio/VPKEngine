#!/bin/sh
# Locked operations under contention (8 threads), aligned and split. usage: run.sh [vpaot]
set -e
D=$(cd "$(dirname "$0")" && pwd); R=$D/../../..; VPAOT=${1:-$R/build/vpaot/vpaot}; B=$R/build/atomics; mkdir -p "$B"
AS=${AS:-as}; LD=${LD:-ld}; OBJCOPY=${OBJCOPY:-objcopy}; CC=${CC:-cc}
$AS --64 -o "$B/contend.o" "$D/contend.s"
$LD -static -Ttext=0x400000 -e _start -o "$B/contend.elf" "$B/contend.o"
$OBJCOPY -O binary "$B/contend.elf" "$B/contend.bin"
"$VPAOT" --raw "$B/contend.bin" --base 0x400000 --entry 0x400000 --out "$B/contend.c"
$CC -O2 -frounding-math -I "$R/runtime" -o "$B/host" "$R/runtime/vp_host.c" "$D/host.c" "$B/contend.c" -lpthread -lm
"$B/host"
