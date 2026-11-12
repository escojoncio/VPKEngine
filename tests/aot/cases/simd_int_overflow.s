# Lane arithmetic at the signed extremes (pmullw, pmaddwd, psignd, phaddd): no signed overflow in the emitted C
# (checked with VPCFLAGS="-fsanitize=undefined -fno-sanitize-recover=all"), same results as the hardware.
.text
.globl _start
_start:
    movabs $0x8000ffff80007fff, %rax
    movq %rax, %xmm0
    punpcklqdq %xmm0, %xmm0
    movdqa %xmm0, %xmm1
    pmullw %xmm0, %xmm1
    movdqa %xmm0, %xmm2
    pmaddwd %xmm0, %xmm2
    movabs $0x8000000080000000, %rax
    movq %rax, %xmm3
    punpcklqdq %xmm3, %xmm3
    movdqa %xmm3, %xmm4
    psignd %xmm3, %xmm4
    movabs $0x7fffffff80000000, %rax
    movq %rax, %xmm5
    punpcklqdq %xmm5, %xmm5
    phaddd %xmm5, %xmm3
    phaddd %xmm5, %xmm5
    mov $0x600000, %rdi
    movdqu %xmm1, (%rdi)
    movdqu %xmm2, 16(%rdi)
    movdqu %xmm4, 32(%rdi)
    movdqu %xmm3, 48(%rdi)
    movdqu %xmm5, 64(%rdi)
    ret
