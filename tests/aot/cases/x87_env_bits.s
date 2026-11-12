# x87: reserved control word bits, C1 after ffree/ffreep, the 16-bit environment forms, fbstp of an empty register or an unnormal.
.text
.globl _start
_start:
    mov $0x600000, %rdi
    lea k(%rip), %rsi
    fninit
    movw $0xffff, 0(%rdi)
    fldcw 0(%rdi)
    fnstcw 2(%rdi)
    movw $0x1234, 0(%rdi)
    fldcw 0(%rdi)
    fnstcw 4(%rdi)
    fninit
    .byte 0x66; fnstenv 16(%rdi)
    fninit
    fnstenv 32(%rdi)
    fninit
    .byte 0x66; fnsave 128(%rdi)
    .byte 0x66; frstor 128(%rdi)
    .byte 0x66; fnstenv 240(%rdi)
    fninit
    fld1; fld1; fld1; fld1; fld1; fld1; fld1; fld1
    fld1
    ffree %st(1)
    fnstsw 6(%rdi)
    fninit
    fld1; fld1; fld1; fld1; fld1; fld1; fld1; fld1
    fld1
    ffreep %st(1)
    fnstsw 8(%rdi)
    fninit
    fbstp 64(%rdi)
    fnstsw 10(%rdi)
    fninit
    fldt 0(%rsi)
    fbstp 80(%rdi)
    fnstsw 12(%rdi)
    fninit
    ret
.balign 8
k:
    .quad 0x0000000000000001
    .word 0x4000
