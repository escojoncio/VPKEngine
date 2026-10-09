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
$CC -O2 -DNATIVE -Dguest_main=guest_main_native -c -o "$B/native.o" "$D/guest.c"
$CC -O2 -I "$R/runtime" -c -o "$B/guest_t.o" "$B/guest_t.c"
$CC -O2 -I "$R/runtime" -c -o "$B/vp_host.o" "$R/runtime/vp_host.c"
$CC -O2 -I "$R/runtime" -c -o "$B/vp_loader.o" "$R/runtime/vp_loader.c"
$CXX -std=c++20 -O2 -I "$D/include" -I "$R/runtime" -c -o "$B/engine.o" "$R/integrations/shadps4/aot_guest_engine.cpp"
$CXX -std=c++20 -O2 -I "$D/include" -I "$R/runtime" -c -o "$B/host.o" "$D/host.cpp"
$CXX -o "$B/host" "$B/host.o" "$B/engine.o" "$B/guest_t.o" "$B/vp_host.o" "$B/vp_loader.o" "$B/native.o" -lpthread -lm
"$B/host" "$B/guest.so" "0x$($NM "$B/guest.so" | awk '$3=="guest_main"{print $1}')"
