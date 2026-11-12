# x87: zero + zero reaches SoftFloat's normalisation with a zero significand (a shift by 64 in C).
.text
.globl _start
_start:
    mov $0x600000, %rdi
    fninit
    fldz
    fldz
    faddp %st, %st(1)
    fstpt 0(%rdi)
    fldz
    fchs
    fldz
    fchs
    faddp %st, %st(1)
    fstpt 16(%rdi)
    fldz
    fldz
    fchs
    faddp %st, %st(1)
    fstpt 32(%rdi)
    fnstsw 48(%rdi)
    fninit
    ret
