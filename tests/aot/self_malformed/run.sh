#!/bin/sh
# usage: run.sh [vpaot [workdir]]
# A malformed SELF (bad program header table, sizes that overflow, segment past the file) is refused with a
# "vpaot: ..." error and exit status 1; it must never crash (vpaot runs inside vpconvert), nor try to reserve memory
# the file does not justify (the address space is capped, so a case that does ends in "bad_alloc", which fails).
D=$(cd "$(dirname "$0")" && pwd); R=$D/../../..; VPAOT=${1:-$R/build/vpaot/vpaot}; B=${2:-$R/build/self_malformed}; mkdir -p "$B"
cases=$(python3 -I "$D/gen.py" "$B")
[ -n "$cases" ] || { echo "FAIL: gen.py made no cases (nothing was tested)"; exit 1; }
bad=0
ulimit -v 600000 2> /dev/null
for c in $cases; do
  "$VPAOT" --elf "$B/$c.self" --out "$B/o.c" > /dev/null 2> "$B/$c.err"
  rc=$?
  if [ "$rc" -ne 1 ] || ! grep -q '^vpaot: ' "$B/$c.err" || grep -q bad_alloc "$B/$c.err"; then echo "FAIL $c: exit $rc"; bad=1; else echo "ok   $c: $(head -n1 "$B/$c.err")"; fi
done
exit $bad
