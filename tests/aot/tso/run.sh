#!/bin/sh
# x86 memory order on the host: message passing between two translated threads, built with VP_TSO=1 (the default off
# x86; forced here too: on x86 a plain spin-loop load may be hoisted by the C compiler). usage: run.sh [vpaot]. On ARM it is built for LRCPC (LDAPR, what clang emits for Apple silicon) when
# the compiler can, and with clang for Apple's target it checks that guest loads are LDAPR and stores STLR.
set -e
D=$(cd "$(dirname "$0")" && pwd); R=$D/../../..; VPAOT=${1:-$R/build/vpaot/vpaot}; B=$R/build/tso; mkdir -p "$B"
AS=${AS:-as}; LD=${LD:-ld}; OBJCOPY=${OBJCOPY:-objcopy}; CC=${CC:-cc}
$AS --64 -o "$B/mp.o" "$D/mp.s"
$LD -static -Ttext=0x400000 -e _start -o "$B/mp.elf" "$B/mp.o"
$OBJCOPY -O binary "$B/mp.elf" "$B/mp.bin"
"$VPAOT" --raw "$B/mp.bin" --base 0x400000 --entry 0x400000 --out "$B/mp.c"
ARCH=""
if [ "$(uname -m)" = aarch64 ] && echo 'int x;' | $CC -march=armv8.3-a+rcpc -x c -c -o /dev/null - 2>/dev/null; then
    ARCH="-march=armv8.3-a+rcpc"
fi
$CC -O2 -frounding-math -DVP_TSO=1 $ARCH -I "$R/runtime" -o "$B/host" "$R/runtime/vp_host.c" "$D/host.c" "$B/mp.c" -lpthread -lm
echo "built with: $CC $ARCH"
"$B/host"
if command -v clang >/dev/null 2>&1 && clang --target=arm64-apple-xros2.0 -O2 -ffreestanding -fno-stack-protector -frounding-math -w \
        -isystem "$R/runtime/freestanding" -I "$R/runtime" -S -o "$B/mp-apple.s" "$B/mp.c" 2>/dev/null; then
    grep -q ldapr "$B/mp-apple.s" && grep -q stlr "$B/mp-apple.s" || { echo "arm64-apple codegen: no LDAPR/STLR"; exit 1; }
    echo "arm64-apple codegen: guest loads LDAPR, stores STLR"
fi
