# Adversarial review of the AVX change: case kept as a regression test (cmp).
.text
.globl _start
_start:
    lea fa(%rip), %rax
    lea fb(%rip), %rcx
    vmovups (%rax), %ymm0
    vmovups (%rcx), %ymm1
    vcmpps $12, %ymm1, %ymm0, %ymm2
    vcmpps $30, %ymm1, %ymm0, %ymm3
    vcmpps $11, %ymm1, %ymm0, %ymm4
    vcmpps $15, %ymm1, %ymm0, %ymm5
    vcmpps $8, %ymm1, %ymm0, %ymm6
    vcmpps $13, %xmm1, %xmm0, %xmm7
    ret
.section .rodata
.balign 32
fa: .long 0x7fc00000, 0x3f800000, 0x40000000, 0x7fc00000, 0x3f800000, 0x40000000, 0x40400000, 0xbf800000
fb: .long 0x3f800000, 0x7fc00000, 0x3f800000, 0x7fc00000, 0x3f800000, 0x40400000, 0x40000000, 0xbf800000
