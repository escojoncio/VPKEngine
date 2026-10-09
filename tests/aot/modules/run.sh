#!/bin/sh
# Two guest modules translated separately (--module, --pic), loaded at addresses of their own,
# calling each other; then attached by fingerprint.
set -e
D=$(cd "$(dirname "$0")" && pwd); R=$D/../../..; VPAOT=${1:-$R/build/vpaot/vpaot}; B=$R/build/modules; mkdir -p "$B"
AS=${AS:-as}; LD=${LD:-ld}; CC=${CC:-cc}; XCC=${XCC:-gcc}; NM=${NM:-nm}
F="-O2 -fPIC -fno-asynchronous-unwind-tables -fcf-protection=none -fno-stack-protector -ffreestanding -fno-builtin -mno-red-zone -msse4.2 -fno-plt"
$XCC $F -c -o "$B/lib.o" "$D/lib.c"
$XCC $F -c -o "$B/main.o" "$D/main.c"
$AS --64 -o "$B/start.o" "$R/tests/aot/program/start.s"
$LD -shared -Ttext-segment=0x200000000 -o "$B/lib.so" "$B/lib.o"
$LD -shared -e _start -Ttext-segment=0x210000000 -o "$B/main.so" "$B/start.o" "$B/main.o"
"$VPAOT" --elf "$B/lib.so" --module lib --pic --out "$B/lib.c" --entry 0x0 2>/dev/null || "$VPAOT" --elf "$B/lib.so" --module lib --pic --out "$B/lib.c"
"$VPAOT" --elf "$B/main.so" --module main --pic --out "$B/main_t.c"
LIBMIX=0x$($NM "$B/lib.so" | awk '$3=="lib_mix"{print $1}')
$CC -O2 -I "$R/runtime" -Dvp_main=vp_main_native -c -o "$B/native_main.o" "$D/main.c"
$CC -O2 -I "$R/runtime" -c -o "$B/native_lib.o" "$D/lib.c"
$CC -O2 -I "$R/runtime" -o "$B/host" "$R/runtime/vp_host.c" "$R/runtime/vp_loader.c" "$D/host.c" "$B/lib.c" "$B/main_t.c" "$B/native_main.o" "$B/native_lib.o" -lm
"$B/host" "$B/lib.so" "$B/main.so" "$LIBMIX"
