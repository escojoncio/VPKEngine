# Adversarial review of the AVX change: case kept as a regression test (dp).
.text
.globl _start
_start:
    lea fa(%rip), %rax
    vmovups (%rax), %ymm0
    vmovups 32(%rax), %ymm1
    vdpps $0xff, %ymm1, %ymm0, %ymm2
    vdpps $0xf1, %xmm1, %xmm0, %xmm3
    ret
.section .rodata
.balign 32
fa: .float 1e8, 1, -1e8, 1, 1, 1e8, 1, -1e8
    .float 1, 1, 1, 1, 1, 1, 1, 1
    .fill 128,1,0xc3
