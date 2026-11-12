#!/bin/sh
# usage: run.sh [vpaot [workdir]]
# Valid SELF containers must be converted (exit 0, no error at all), with the same limits as runtime/vp_loader.c (runtime/vp_self.h):
#  - dense_2m_align: an ELF linked with ld -z max-page-size=0x200000 (LOAD segments at file offsets 0 and 0x200000), wrapped by
#    ../self_wrap.py: the SELF is small, the ELF inside has the alignment padding;
#  - ladder_4x2m: four LOAD segments at file offsets 0, 2, 4 and 6 MiB (ladder.ld), a SELF under 1 KB for a 6 MiB ELF;
#  - edge_ok (gen.py): the farthest segment end the limit allows (32 backed headers: N + 66 MiB); the refused twin, one byte
#    more, is edge_plus1 in ../self_malformed.
# The refused counterparts are in ../self_malformed.
D=$(cd "$(dirname "$0")" && pwd); R=$D/../../..; VPAOT=${1:-$R/build/vpaot/vpaot}; B=${2:-$R/build/self_valid}; mkdir -p "$B"
cases=$(python3 -I "$D/gen.py" "$B") || { echo "FAIL: gen.py failed"; exit 1; }
[ -n "$cases" ] || { echo "FAIL: gen.py made no cases (nothing was tested)"; exit 1; }
ulimit -v 600000 2> /dev/null
bad=0
check() { # name file
  if "$VPAOT" --elf "$2" --out "$B/o.c" > /dev/null 2> "$B/$1.err"; then echo "ok   $1"; else echo "FAIL $1: $(head -n1 "$B/$1.err")"; bad=1; fi
}
printf '.text\n.globl _start\n_start:\n nop\n ret\n' > "$B/a.s"
printf '.section .t1,"a"\n .byte 0\n.text\n.globl _start\n_start:\n nop\n ret\n.section .rodata,"a"\n .byte 1\n.data\n .byte 2\n' > "$B/b.s"
as --64 -o "$B/a.o" "$B/a.s" && ld -static -z max-page-size=0x200000 -z separate-code -o "$B/a.elf" "$B/a.o" || { echo "FAIL: cannot build the ELF a"; exit 1; }
as --64 -o "$B/b.o" "$B/b.s" && ld -static -z max-page-size=0x200000 -T "$D/ladder.ld" -o "$B/b.elf" "$B/b.o" || { echo "FAIL: cannot build the ELF b"; exit 1; }
for c in a b; do python3 -I "$D/../self_wrap.py" "$B/$c.elf" "$B/$c.self" || exit 1; done
readelf -lW "$B/b.elf" | grep LOAD | awk '{print $2}' | tr '\n' ' ' | grep -q '^0x000000 0x200000 0x400000 0x600000 $' || { echo "FAIL: the ladder ELF is not at 0, 2, 4, 6 MiB"; exit 1; }
[ "$(wc -c < "$B/b.self")" -lt 1000 ] || { echo "FAIL: ladder SELF is not small"; exit 1; }
check dense_2m_align "$B/a.self"
check ladder_4x2m "$B/b.self"
for c in $cases; do check "$c" "$B/$c.self"; done
exit $bad
