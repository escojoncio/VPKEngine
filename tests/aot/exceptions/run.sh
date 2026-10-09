#!/bin/sh
# C++ exceptions end to end: a static image with its own libsupc++ + libgcc_eh (unwinder, .eh_frame,
# LSDA landing pads), translated whole and compared with the same C++ compiled natively.
# usage: run.sh [vpaot]   (XCXX/XCC/LD: an x86-64 C++ toolchain with libsupc++ and libgcc_eh)
set -e
D=$(cd "$(dirname "$0")" && pwd); R=$D/../../..; VPAOT=${1:-$R/build/vpaot/vpaot}; B=$R/build/exceptions; mkdir -p "$B"
AS=${AS:-as}; CC=${CC:-cc}; CXX=${CXX:-c++}; XCXX=${XCXX:-g++}
XCXX=$XCXX "$D/build.sh" "$B"
$AS --64 -o "$B/start.o" "$R/tests/aot/program/start.s"
LIBS="$($XCXX -print-file-name=libsupc++.a) $($XCXX -print-libgcc-file-name) $($XCXX -print-file-name=libgcc_eh.a)"
# shellcheck disable=SC2086
$XCXX -static -nostdlib -no-pie -Wl,-e,_start -Wl,-Ttext-segment=0x400000 -Wl,--eh-frame-hdr -o "$B/prog.elf" \
    "$B/start.o" "$B/prog.o" "$B/shim.o" -Wl,--start-group $LIBS -Wl,--end-group 2>/dev/null
"$VPAOT" --elf "$B/prog.elf" --out "$B/prog.c" --stats "$B/stats.json"
$CXX -O2 -Dvp_main=vp_main_native -c -o "$B/native.o" "$D/prog.cpp"
$CC -O2 -frounding-math -I "$R/runtime" -c -o "$B/host.o" "$D/host.c"
$CC -O2 -frounding-math -I "$R/runtime" -c -o "$B/vp_host.o" "$R/runtime/vp_host.c"
$CC -O1 -frounding-math -w -I "$R/runtime" -c -o "$B/prog_t.o" "$B/prog.c"
$CXX -o "$B/host" "$B/host.o" "$B/vp_host.o" "$B/prog_t.o" "$B/native.o" -lm
"$B/host" "$B/prog.elf"
