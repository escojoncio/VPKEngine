# Adversarial review of the x87 change: pseudo-NaN operands.
.text
.globl _start
_start:
    mov $0x600000, %rdi
    lea k(%rip), %rsi
    fldt 16(%rsi)
    fldt 48(%rsi)
    fadd %st(1), %st
    fnstsw 0(%rdi)
    fstpt 64(%rdi)
    fnclex
    fninit
    fldt 16(%rsi)
    fldt 48(%rsi)
    fmul %st(1), %st
    fnstsw 2(%rdi)
    fstpt 80(%rdi)
    fnclex
    fninit
    fldt 16(%rsi)
    fldt 48(%rsi)
    fsqrt
    fnstsw 4(%rdi)
    fstpt 96(%rdi)
    fnclex
    fninit
    fldt 16(%rsi)
    fldt 48(%rsi)
    frndint
    fnstsw 6(%rdi)
    fstpt 112(%rdi)
    fnclex
    fninit
    fldt 16(%rsi)
    fldt 48(%rsi)
    fxam
    fnstsw 8(%rdi)
    fstpt 128(%rdi)
    fnclex
    fninit
    fldt 16(%rsi)
    fldt 48(%rsi)
    ftst
    fnstsw 10(%rdi)
    fstpt 144(%rdi)
    fnclex
    fninit
    fldt 16(%rsi)
    fldt 48(%rsi)
    fcom %st(1)
    fnstsw 12(%rdi)
    fstpt 160(%rdi)
    fnclex
    fninit
    fldt 16(%rsi)
    fldt 48(%rsi)
    fucom %st(1)
    fnstsw 14(%rdi)
    fstpt 176(%rdi)
    fnclex
    fninit
    fldt 16(%rsi)
    fldt 48(%rsi)
    fprem
    fnstsw 16(%rdi)
    fstpt 192(%rdi)
    fnclex
    fninit
    fldt 16(%rsi)
    fldt 48(%rsi)
    fscale
    fnstsw 18(%rdi)
    fstpt 208(%rdi)
    fnclex
    fninit
    fldt 16(%rsi)
    fldt 48(%rsi)
    fxtract
    fnstsw 20(%rdi)
    fstpt 224(%rdi)
    fnclex
    fninit
    fldt 16(%rsi)
    fldt 48(%rsi)
    fchs
    fnstsw 22(%rdi)
    fstpt 240(%rdi)
    fnclex
    fninit
    fldt 16(%rsi)
    fldt 48(%rsi)
    fabs
    fnstsw 24(%rdi)
    fstpt 256(%rdi)
    fnclex
    fninit
    fldt 48(%rsi)
    fstps 0x200(%rdi)
    fnstsw 516(%rdi)
    fnclex
    fldt 48(%rsi)
    fistpl 0x208(%rdi)
    fnstsw 524(%rdi)
    fnclex
    fldt 48(%rsi)
    fstpl 0x210(%rdi)
    fnstsw 536(%rdi)
    fnclex
    fldt 16(%rsi)
    fldt 48(%rsi)
    fcomi %st(1), %st
    setb 0x220(%rdi)
    setz 0x221(%rdi)
    setp 0x222(%rdi)
    fnstsw 548(%rdi)
    fninit
    fldt 48(%rsi)
    fldt 16(%rsi)
    fprem
    fnstsw 550(%rdi)
    fstpt 0x230(%rdi)
    fninit
    fldt 48(%rsi)
    fldt 16(%rsi)
    fscale
    fnstsw 576(%rdi)
    fstpt 0x250(%rdi)
    fninit

    ret

.section .rodata
.balign 16
k:
    .quad 0x8000000000000000, 0x3fff        # 0: 1.0
    .quad 0xC000000000000000, 0x4000        # 16: 3.0
    .quad 0x4000000000000000, 0x4000        # 32: unnormal (exp 0x4000, int bit 0)
    .quad 0x4000000000000000, 0x7fff        # 48: pseudo-NaN
    .quad 0x0000000000000000, 0x7fff        # 64: pseudo-infinity
    .quad 0x8000000000000001, 0x0000        # 80: pseudo-denormal
    .quad 0x0000000000000123, 0x0000        # 96: denormal
    .quad 0xA000000000000000, 0x4000        # 112: 5.0
    .quad 0x8000000000000001, 0x7ff0        # 128: huge
    .quad 0xC000000000000000, 0x403e        # 144: 3*2^63
    .quad 0x8000000000000000, 0x0001        # 160: min normal
    .quad 0x8000000000000000, 0x407f        # 176: 2^128
    .quad 0xFFFFFFFFFFFFFFFF, 0x3ffe        # 192: just below 1
    .short 300, -300, 32767, -32768         # 208
    .long 0x7f800001                        # 216 f32 SNaN
    .long 0x00000001                        # 220 f32 denormal
    .quad 0x0000000000001234, 0x8000        # 224: negative denormal
    .byte 0x21,0x43,0x65,0x87,0x09,0x21,0x43,0x65,0x87,0x80   # 240: bcd -876543210987654321
