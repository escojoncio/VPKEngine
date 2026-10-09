# Adversarial review of the AVX change: case kept as a regression test (rc).
.text
.globl _start
_start:
    lea fa(%rip), %rax
    vmovups (%rax), %ymm0
    vmovupd 32(%rax), %ymm1
    movl $0x5f80, (%rdi)
    ldmxcsr (%rdi)
    vroundps $0, %ymm0, %ymm2
    vroundps $4, %ymm0, %ymm3
    vcvtps2dq %ymm0, %ymm4
    vcvtpd2dq %ymm1, %xmm5
    vcvtpd2ps %ymm1, %xmm6
    vcvtps2ph $4, %ymm0, %xmm7
    vcvttps2dq %ymm0, %ymm8
    vcvtdq2ps 64(%rax), %ymm9
    vmaxps %ymm1, %ymm0, %ymm10
    vminpd %ymm0, %ymm1, %ymm11
    movl $0x1f80, (%rdi)
    ldmxcsr (%rdi)
    ret
.section .rodata
.balign 32
fa: .float 1.5, -2.5, 2.5, 0.7, 1e10, -1e10, 3.3, -0.0
    .double 1.5, -2.5, 3.0000001, 1e300
    .long 0x7fffffff, 0x01000001, -3, 16777217, 0x7ffffff1, 5, 6, 7
