#!/bin/sh
# Times f() natively (x86-64 only) and translated, same code. usage: run_bench.sh [vpaot]
set -e
D=$(cd "$(dirname "$0")" && pwd); R=$D/../../..; VPAOT=${1:-$R/build/vpaot/vpaot}; B=$R/build/bench; mkdir -p "$B"
gcc -O2 -fPIE -fno-asynchronous-unwind-tables -fcf-protection=none -fno-stack-protector -ffreestanding -fno-builtin -mno-red-zone -msse4.2 -S -o "$B/bench.s" "$D/bench.c"
{ printf '.text\n.globl _start\n_start:\n    sub $8, %%rsp\n    mov $37, %%rdi\n    mov $0x10000000, %%rsi\n    call f\n    add $8, %%rsp\n    ret\n'
  grep -v -E '^\s*\.(file|ident|section\s+\.note|globl|type|size|p2align|text)\b' "$B/bench.s" | sed -E 's/^\s*\.section\s+\.rodata.*$/.section .rodata/'; } > "$B/case.s"
as --64 -o "$B/case.o" "$B/case.s"; ld -static -T "$D/../link.ld" -o "$B/case.elf" "$B/case.o"; objcopy -O binary "$B/case.elf" "$B/code.bin"
"$VPAOT" --raw "$B/code.bin" --base 0x400000 --entry 0x400000 --out "$B/code.c" ${VPFLAGS:-} 2>/dev/null
${CC:-gcc} ${OPT:--O3} -I "$R/runtime" -o "$B/bench" "$R/runtime/vp_host.c" "$D/bench_main.c" "$D/../native_x86.c" "$B/code.c" -lm
"$B/bench" "$B/code.bin"
