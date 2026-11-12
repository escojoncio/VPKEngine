#!/bin/sh
# usage: run.sh [vpaot [workdir]]
# A valid SELF whose ELF has a segment far into the image because of alignment padding (ld -z max-page-size=0x200000: LOAD at
# file offset 0x200000, 2 bytes long) must be accepted, with the same limits as runtime/vp_loader.c (runtime/vp_self.h).
# The refused counterparts are in ../self_malformed.
D=$(cd "$(dirname "$0")" && pwd); R=$D/../../..; VPAOT=${1:-$R/build/vpaot/vpaot}; B=${2:-$R/build/self_valid}; mkdir -p "$B"
printf '.text\n.globl _start\n_start:\n nop\n ret\n' > "$B/a.s" || exit 1
as --64 -o "$B/a.o" "$B/a.s" && ld -static -z max-page-size=0x200000 -z separate-code -o "$B/a.elf" "$B/a.o" || { echo "FAIL: cannot build the ELF"; exit 1; }
python3 -I "$D/../self_wrap.py" "$B/a.elf" "$B/a.self" || exit 1
ulimit -v 600000 2> /dev/null
if "$VPAOT" --elf "$B/a.self" --out "$B/o.c" > /dev/null 2> "$B/a.err"; then echo "ok   dense_2m_align"; else echo "FAIL dense_2m_align: $(head -n1 "$B/a.err")"; exit 1; fi
