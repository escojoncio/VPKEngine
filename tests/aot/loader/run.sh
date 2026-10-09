#!/bin/sh
# Loader + imports test: a static PIE with undefined symbols, loaded by vp_loader with natives.
set -e
D=$(cd "$(dirname "$0")" && pwd); R=$D/../../..; VPAOT=${1:-$R/build/vpaot/vpaot}; B=$R/build/loader; mkdir -p "$B"
AS=${AS:-as}; LD=${LD:-ld}; CC=${CC:-cc}; XCC=${XCC:-gcc}
$XCC -O2 -fPIE -fno-asynchronous-unwind-tables -fcf-protection=none -fno-stack-protector -ffreestanding -fno-builtin \
     -mno-red-zone -msse4.2 -fno-plt -c -o "$B/prog2.o" "$D/prog2.c"
$AS --64 -o "$B/start.o" "$R/tests/aot/program/start.s"
$LD -shared -e _start -Ttext-segment=0x200000000 -o "$B/prog2.elf" "$B/start.o" "$B/prog2.o"
"$VPAOT" --elf "$B/prog2.elf" --out "$B/prog2.c" --stats "$B/stats.json" --pic
$CC -O2 -I "$R/runtime" -Dvp_main=vp_main_native -c -o "$B/native.o" "$D/prog2.c"
$CC -O2 -I "$R/runtime" -o "$B/host2" "$R/runtime/vp_host.c" "$R/runtime/vp_loader.c" "$D/host2.c" "$B/prog2.c" "$B/native.o" -lm
"$B/host2" "$B/prog2.elf"
# The same translation placed 1 GB higher: relocated by the loader, --pic code follows it.
VP_LOAD_AT=0x240000000 "$B/host2" "$B/prog2.elf"
# The same image on 8 threads at once.
$CC -O2 -I "$R/runtime" -o "$B/threads" "$R/runtime/vp_host.c" "$R/runtime/vp_loader.c" "$D/threads.c" "$B/prog2.c" "$B/native.o" -lm -lpthread
"$B/threads" "$B/prog2.elf"
