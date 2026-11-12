# x87: a signaling NaN in a m32/m64 operand is not quieted before the NaN pick; a push overflow gives no DE.
.text
.globl _start
_start:
    mov $0x600000, %rdi
    lea k(%rip), %rsi
    fninit
    flds 0(%rsi)            # QNaN
    fadds 4(%rsi)           # SNaN, smaller significand: the QNaN wins
    fnstsw 0(%rdi)
    fstpt 16(%rdi)
    fnclex
    fninit
    fldl 8(%rsi)            # QNaN
    faddl 16(%rsi)          # SNaN
    fnstsw 2(%rdi)
    fstpt 32(%rdi)
    fnclex
    fninit
    fldl 8(%rsi)
    fmull 24(%rsi)          # SNaN with the bit below the quiet bit set
    fnstsw 4(%rdi)
    fstpt 48(%rdi)
    fnclex
    fninit
    flds 4(%rsi)            # fld of an SNaN: IE and the quieted value
    fnstsw 6(%rdi)
    fstpt 64(%rdi)
    fnclex
    fninit
    fld1; fld1; fld1; fld1; fld1; fld1; fld1; fld1
    flds 32(%rsi)           # push on a full stack with a denormal: IE, SF, C1, no DE
    fnstsw 8(%rdi)
    fnclex
    fninit
    ret
.balign 8
k:
    .long 0x7fc00001, 0x7f800123
    .quad 0x7ff8000000000001, 0x7ff0000000000123, 0x7ff0000000000001
    .long 0x00000001
