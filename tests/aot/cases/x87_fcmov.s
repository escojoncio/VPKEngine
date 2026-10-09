# Adversarial review of the x87 change: fcmovcc stack underflow whatever the condition.
.text
.globl _start
_start:
    mov $0x600000, %rdi
    lea k(%rip), %rsi
    fld1
    clc
    fcmovb %st(1), %st
    fnstsw 0(%rdi)
    stc
    fcmovb %st(1), %st
    fnstsw 2(%rdi)
    fninit
    fld1
    fld1
    ffree %st(0)
    stc
    fcmovb %st(1), %st
    fnstsw 4(%rdi)
    fninit
    fld1
    fld1
    ffree %st(0)
    clc
    fcmovb %st(1), %st
    fnstsw 6(%rdi)

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
