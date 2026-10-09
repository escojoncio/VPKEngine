#!/bin/sh
# A game pack (tools/scripts/make_game_pack): the translated modules built as a shared library of
# their own, with only the freestanding headers (no platform SDK), linked against the runtime as
# a shared library, and loaded at run time with dlopen. Same guest as tests/aot/modules.
set -e
D=$(cd "$(dirname "$0")" && pwd); R=$D/../../..; VPAOT=${1:-$R/build/vpaot/vpaot}; B=$R/build/pack; mkdir -p "$B"
AS=${AS:-as}; LD=${LD:-ld}; CC=${CC:-cc}; XCC=${XCC:-gcc}; NM=${NM:-nm}
M=$R/tests/aot/modules
F="-O2 -fPIC -fno-asynchronous-unwind-tables -fcf-protection=none -fno-stack-protector -ffreestanding -fno-builtin -mno-red-zone -msse4.2 -fno-plt"
$XCC $F -c -o "$B/lib.o" "$M/lib.c"
$XCC $F -c -o "$B/main.o" "$M/main.c"
$AS --64 -o "$B/start.o" "$R/tests/aot/program/start.s"
$LD -shared -Ttext-segment=0x200000000 -o "$B/lib.so" "$B/lib.o"
$LD -shared -e _start -Ttext-segment=0x210000000 -o "$B/main.so" "$B/start.o" "$B/main.o"
"$VPAOT" --elf "$B/lib.so" --module lib --pic --split 2 --out "$B/lib.c"
"$VPAOT" --elf "$B/main.so" --module main --pic --out "$B/main_t.c"
"$VPAOT" --registry "$B/registry.c" --pack "TEST00001" lib main
# The runtime, as the app's VPRuntime.
$CC -O2 -frounding-math -fPIC -shared -I "$R/runtime" -o "$B/libvpruntime.so" "$R/runtime/vp_host.c" -lm -lpthread
# The pack: only what the pack scripts give the compiler (freestanding headers, no libc).
PF="-O2 -frounding-math -fno-math-errno -fPIC -ffreestanding -fno-stack-protector -isystem $R/runtime/freestanding -I $R/runtime"
OBJS=""
LIBSRC="$B/lib.c"; [ -f "$B/lib_files.txt" ] && LIBSRC=$(cat "$B/lib_files.txt")
for f in $LIBSRC "$B/main_t.c" "$B/registry.c"; do
    case "$f" in /*) src=$f ;; *) src=$B/$f ;; esac
    o="$B/pack_$(basename "$src" .c).o"
    $CC $PF -c -o "$o" "$src"
    OBJS="$OBJS $o"
done
# shellcheck disable=SC2086
$CC -shared -nostdlib -o "$B/game.so" $OBJS -L "$B" -lvpruntime -Wl,--no-undefined -Wl,--allow-shlib-undefined
# Nothing but the runtime (and memcpy/memset) may be left for the loader to find.
UNDEF=$($NM -D --undefined-only "$B/game.so" | awk '{print $2}' | grep -v '^vp_' | grep -vE '^(memcpy|memset|memmove|memcmp)(@.*)?$' || true)
[ -z "$UNDEF" ] || { echo "the pack needs more than the runtime: $UNDEF"; exit 1; }
LIBMIX=0x$($NM "$B/lib.so" | awk '$3=="lib_mix"{print $1}')
$CC -O2 -I "$R/runtime" -Dvp_main=vp_main_native -c -o "$B/native_main.o" "$M/main.c"
$CC -O2 -I "$R/runtime" -c -o "$B/native_lib.o" "$M/lib.c"
$CC -O2 -I "$R/runtime" -o "$B/host" "$R/runtime/vp_loader.c" "$D/host.c" "$B/native_main.o" "$B/native_lib.o" \
    -L "$B" -lvpruntime -Wl,-rpath,"$B" -ldl
"$B/host" "$B/game.so" "$B/lib.so" "$B/main.so" "$LIBMIX"
