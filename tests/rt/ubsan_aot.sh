#!/bin/sh
# The translated C of every tests/aot/cases/*.s (or the cases named) compiled with UBSan (signed
# overflow, shifts, float-to-int casts, ...) and run on its own, without comparing with the host.
#   [CASES_DIR=dir] ubsan_aot.sh VPAOT OUT_DIR [case...]
set -u
VPAOT=$1; OUT=$2; shift 2
D=$(cd "$(dirname "$0")" && pwd); R=$D/../..
CASES=${CASES_DIR:-$R/tests/aot/cases}
mkdir -p "$OUT"
cases=${*:-$(cd "$CASES" && ls *.s | sed 's/\.s$//')}
SAN="-fsanitize=undefined,float-cast-overflow -fno-sanitize-recover=all"
fail=0; n=0; skipped=0
for c in $cases; do
    w=$OUT/$c; mkdir -p "$w"; n=$((n + 1))
    as --64 -o "$w/code.o" "$CASES/$c.s" &&
    ld -static -T "$R/tests/aot/link.ld" -o "$w/code.elf" "$w/code.o" &&
    objcopy -O binary "$w/code.elf" "$w/code.bin" || { echo "SKIP $c (assembler)"; skipped=$((skipped + 1)); continue; }
    nat=$(sed -n 's/^# native: *//p' "$CASES/$c.s" | head -1)
    extra=""
    [ -n "$nat" ] && extra="--native 0x$(nm "$w/code.elf" | awk -v s="$nat" '$3==s{print $1}')"
    "$VPAOT" --raw "$w/code.bin" --base 0x400000 --entry 0x400000 --out "$w/code.c" $extra >/dev/null 2>&1 || { echo "FAIL $c (vpaot cannot translate it)"; fail=$((fail + 1)); continue; }
    cc -O1 -g -std=gnu11 -frounding-math -w $SAN -I "$R/runtime" -o "$w/h" "$R/runtime/vp_host.c" "$R/tests/aot/harness.c" "$w/code.c" \
       "$R/tests/aot/native_x86.c" -lm -latomic 2>"$w/cc.log" || { echo "FAIL $c (cc: $(head -c 200 "$w/cc.log"))"; fail=$((fail + 1)); continue; }
    if ! env VP_NO_NATIVE=1 ${nat:+VP_NATIVE=0x$(nm "$w/code.elf" | awk -v s="$nat" '$3==s{print $1}')} "$w/h" "$w/code.bin" >"$w/out.txt" 2>"$w/err.txt"; then
        echo "FAIL $c: $(grep -m1 'runtime error' "$w/err.txt" || head -c 200 "$w/err.txt")"; fail=$((fail + 1))
    elif grep -q 'runtime error' "$w/err.txt"; then
        echo "FAIL $c: $(grep -m1 'runtime error' "$w/err.txt")"; fail=$((fail + 1))
    fi
done
echo "ubsan_aot: $n cases, $skipped skipped (no assembler), $fail failed"
[ "$fail" = 0 ] && [ "$skipped" -lt "$n" ]
