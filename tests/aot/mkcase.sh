#!/bin/sh
# Turns a C file with `unsigned long f(unsigned long x, unsigned long* mem)` into a test case:
# _start calls f(37, scratch) and returns its value in rax.  usage: mkcase.sh file.c name
set -e
src=$1; name=$2; dir=$(dirname "$0")
gcc -O2 -fPIE -fno-asynchronous-unwind-tables -fcf-protection=none -fno-stack-protector -ffreestanding \
    -fno-builtin -mno-red-zone -mpopcnt -mlzcnt -mbmi -msse4.2 ${VP_CFLAGS:-} -S -o /tmp/vp_case.s "$src"
{
  printf '.text\n.globl _start\n_start:\n    sub $8, %%rsp\n    mov $37, %%rdi\n    mov $0x600000, %%rsi\n    call f\n    add $8, %%rsp\n    ret\n'
  grep -v -E '^\s*\.(file|ident|section\s+\.note|globl|type|size|p2align|text)\b' /tmp/vp_case.s \
    | sed -E 's/^\s*\.section\s+\.rodata.*$/.section .rodata/'
} > "$dir/cases/$name.s"
