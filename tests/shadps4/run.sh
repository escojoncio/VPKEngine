#!/bin/sh
# The AOT guest engine (integrations/shadps4/aot_guest_engine.cpp) driven the way AstroVisionPro's
# shadPS4 drives FEX's: veneers, HLE bridge, nested guest calls, guest threads, TLS through fs.
set -e
D=$(cd "$(dirname "$0")" && pwd); R=$D/../..; VPAOT=${1:-$R/build/vpaot/vpaot}; B=$R/build/shadps4; mkdir -p "$B"
LD=${LD:-ld}; CC=${CC:-cc}; CXX=${CXX:-c++}; XCC=${XCC:-gcc}; NM=${NM:-nm}
F="-O2 -fPIC -fno-asynchronous-unwind-tables -fcf-protection=none -fno-stack-protector -ffreestanding -fno-builtin -mno-red-zone -msse4.2 -fno-plt"
$XCC $F -c -o "$B/guest.o" "$D/guest.c"
$LD -shared -Ttext-segment=0x400000000 -o "$B/guest.so" "$B/guest.o"
"$VPAOT" --elf "$B/guest.so" --module game --pic --out "$B/guest_t.c" --stats "$B/stats.json"
$XCC $F -c -o "$B/lib.o" "$D/lib.c"
$LD -shared -Ttext-segment=0x400000000 -o "$B/lib.so" "$B/lib.o"
"$VPAOT" --elf "$B/lib.so" --module lib --pic --out "$B/lib_t.c"
$CC -O2 -frounding-math -I "$R/runtime" -c -o "$B/lib_t.o" "$B/lib_t.c"
$CC -O2 -DNATIVE -Dguest_main=guest_main_native -Don_signal=on_signal_native -Dfail_test=fail_test_native \
    -Dfail_out=fail_out_native -Dsignals_seen=signals_seen_native -Dsignal_sink=signal_sink_native -c -o "$B/native.o" "$D/guest.c"
$CC -O2 -Dlib_fn=lib_fn_native -c -o "$B/lib_native.o" "$D/lib.c"
$CC -O2 -frounding-math -I "$R/runtime" -c -o "$B/guest_t.o" "$B/guest_t.c"
$CC -O2 -I "$R/runtime" -c -o "$B/vp_host.o" "$R/runtime/vp_host.c"
$CC -O2 -I "$R/runtime" -c -o "$B/vp_loader.o" "$R/runtime/vp_loader.c"
$CXX -std=c++20 -O2 -I "$D/include" -I "$R/runtime" -c -o "$B/engine.o" "$R/integrations/shadps4/aot_guest_engine.cpp"
$CXX -std=c++20 -O2 -I "$D/include" -I "$R/runtime" -c -o "$B/host.o" "$D/host.cpp"
# As on visionOS, the engine and the translations are a static library: only what is referenced
# gets linked, so the modules come in through the registry vpaot writes (lib first: the game's is
# then not the first registered module).
"$VPAOT" --registry "$B/registry.c" lib game
$CC -O2 -I "$R/runtime" -c -o "$B/registry.o" "$B/registry.c"
rm -f "$B/libshadps4.a"
${AR:-ar} rcs "$B/libshadps4.a" "$B/engine.o" "$B/registry.o" "$B/lib_t.o" "$B/guest_t.o" "$B/vp_host.o" "$B/vp_loader.o"
$CXX -o "$B/host" "$B/host.o" "$B/native.o" "$B/lib_native.o" "$B/libshadps4.a" -lpthread -lm
sym() { echo "0x$($NM "$1" | awk -v s="$2" '$3==s{print $1}')"; }
"$B/host" "$B/guest.so" "$(sym "$B/guest.so" guest_main)" "$(sym "$B/guest.so" on_signal)" "$B/lib.so" \
    "$(sym "$B/lib.so" lib_fn)" "$(sym "$B/guest.so" fail_test)" "$(sym "$B/guest.so" fail_out)"
