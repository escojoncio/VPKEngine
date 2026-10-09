# AVX (VEX.256) and F16C as Jaguar runs them: lane-split float ops, the immediates with per-half
# bits, cross-lane moves, VEX.128 zeroing bits 128..255 while legacy SSE keeps them, masked
# moves, flag-setting tests, width-changing conversions, half-precision conversions.
.text
.globl _start
_start:
    mov $0x600000, %rdi
    lea fa(%rip), %rax
    lea fb(%rip), %rcx
    vmovups (%rax), %ymm0
    vmovups (%rcx), %ymm1
    # legacy SSE keeps the upper half of ymm2 (initial state); VEX.128 zeroes ymm3's
    addps %xmm0, %xmm2
    vaddps %xmm0, %xmm1, %xmm3
    # lane-split arithmetic and logic
    vaddps %ymm1, %ymm0, %ymm4
    vmulps (%rcx), %ymm0, %ymm5
    vdivps %ymm0, %ymm1, %ymm6
    vsqrtps %ymm1, %ymm7
    vminps %ymm0, %ymm1, %ymm8
    vandnps %ymm1, %ymm0, %ymm9
    vxorps %ymm10, %ymm10, %ymm10
    vhaddps %ymm1, %ymm0, %ymm11
    vdpps $0xf1, %ymm1, %ymm0, %ymm12
    vroundps $1, %ymm6, %ymm13
    vmovddup %ymm1, %ymm14
    vunpckhps %ymm1, %ymm0, %ymm15
    vmovups %ymm4, 0x100(%rdi)
    vmovups %ymm11, 0x120(%rdi)
    vmovups %ymm12, 0x140(%rdi)
    # immediates with separate bits per half
    vshufps $0x1b, %ymm1, %ymm0, %ymm4
    vshufpd $0x9, %ymm1, %ymm0, %ymm5
    vblendps $0xa5, %ymm1, %ymm0, %ymm6
    vblendpd $0x6, %ymm1, %ymm0, %ymm7
    vcmpltps %ymm1, %ymm0, %ymm8
    vblendvps %ymm8, %ymm1, %ymm0, %ymm9
    vmovups %ymm4, 0x160(%rdi)
    vmovups %ymm5, 0x180(%rdi)
    vmovups %ymm6, 0x1a0(%rdi)
    vmovups %ymm7, 0x1c0(%rdi)
    vmovups %ymm9, 0x1e0(%rdi)
    # permutes and cross-lane moves
    vpermilps $0x4e, %ymm0, %ymm4
    lea ctl(%rip), %rdx
    vpermilps (%rdx), %ymm1, %ymm5
    vpermilpd $0x5, %ymm0, %ymm6
    vpermilpd 32(%rdx), %ymm1, %ymm7
    vperm2f128 $0x31, %ymm1, %ymm0, %ymm8
    vperm2f128 $0x82, %ymm1, %ymm0, %ymm9
    vinsertf128 $1, %xmm1, %ymm0, %ymm10
    vinsertf128 $0, 16(%rcx), %ymm1, %ymm11
    vextractf128 $1, %ymm0, %xmm12
    vextractf128 $1, %ymm1, 0x200(%rdi)
    vbroadcastss 4(%rax), %ymm13
    vbroadcastsd 8(%rcx), %ymm14
    vbroadcastf128 16(%rax), %ymm15
    vmovups %ymm4, 0x220(%rdi)
    vmovups %ymm5, 0x240(%rdi)
    vmovups %ymm6, 0x260(%rdi)
    vmovups %ymm7, 0x280(%rdi)
    vmovups %ymm8, 0x2a0(%rdi)
    vmovups %ymm9, 0x2c0(%rdi)
    vmovups %ymm10, 0x2e0(%rdi)
    vmovups %ymm11, 0x300(%rdi)
    vmovups %ymm13, 0x320(%rdi)
    vmovups %ymm14, 0x340(%rdi)
    vmovups %ymm15, 0x360(%rdi)
    # masks, tests, conversions
    vcmpltps %ymm1, %ymm0, %ymm2
    vmovmskps %ymm2, %r8d
    vmovmskpd %ymm1, %r9d
    vtestps %ymm2, %ymm0
    setz %r10b
    setc %r11b
    vptest %ymm1, %ymm1
    setz %r12b
    vmaskmovps (%rax), %ymm2, %ymm3
    vmaskmovps %ymm1, %ymm2, 0x380(%rdi)
    vmaskmovpd 0x3a0(%rdi), %xmm2, %xmm4
    vcvtps2pd %xmm0, %ymm5
    vcvtpd2ps %ymm5, %xmm6
    vcvttpd2dq %ymm5, %xmm7
    vcvtpd2dqy (%rcx), %xmm8
    vcvtdq2pd 0x10(%rdi), %ymm9
    vmovups %ymm5, 0x3c0(%rdi)
    vmovups %ymm9, 0x3e0(%rdi)
    # F16C: every rounding mode, from the immediate and from MXCSR
    vcvtps2ph $0, %ymm0, %xmm10
    vcvtps2ph $1, %ymm1, 0x400(%rdi)
    vcvtps2ph $2, %xmm0, %xmm11
    vcvtps2ph $3, %xmm1, 0x410(%rdi)
    vcvtps2ph $4, %ymm6, %xmm12
    vcvtph2ps %xmm10, %ymm13
    vcvtph2ps 0x400(%rdi), %xmm14
    vcvtph2ps (%rdx), %ymm15
    vmovups %ymm13, 0x420(%rdi)
    vmovups %ymm15, 0x440(%rdi)
    # vzeroupper, then a legacy op and a 256-bit op again
    vzeroupper
    mulps %xmm1, %xmm0
    vsubps %ymm1, %ymm1, %ymm3
    ret

.section .rodata
.balign 32
fa: .float 1.5, -2.25, 3.0, 0.125, 100.0, -7.5, 65504.0, 1.0e-5
fb: .float 2.0, 4.5, -1.0, 8.0, 0.5, 3.25, -65520.0, 7.0e-8
ctl: .long 3, 0, 2, 1, 1, 1, 0, 3
     .quad 2, 0, 0, 2
