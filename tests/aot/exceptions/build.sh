#!/bin/sh
set -e
D=$(cd "$(dirname "$0")" && pwd); B=$1
XCXX=${XCXX:-g++}; XCC=${XCC:-gcc}
F="-O2 -fno-pie -no-pie -fcf-protection=none -fno-stack-protector -mno-red-zone -msse4.2"
$XCXX $F -fexceptions -c -o "$B/prog.o" "$D/prog.cpp"
$XCC $F -ffreestanding -fno-builtin -c -o "$B/shim.o" "$D/shim.c"
