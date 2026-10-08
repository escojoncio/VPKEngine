# Scalar and packed SSE arithmetic, moves, conversions and comparisons.
.text
.globl _start
_start:
    cvtsi2ss %eax, %xmm0
    cvtsi2sd %rcx, %xmm1
    movss %xmm0, (%rdi)
    movsd %xmm1, 8(%rdi)
    addss (%rdi), %xmm0
    mulsd 8(%rdi), %xmm1
    subss %xmm0, %xmm2
    divsd %xmm1, %xmm3
    sqrtsd %xmm1, %xmm4
    cvtss2sd %xmm0, %xmm5
    cvtsd2ss %xmm1, %xmm6
    cvttss2si %xmm0, %eax
    cvttsd2si %xmm1, %rcx
    ucomiss %xmm0, %xmm2
    seta %dl
    setp %bl
    comisd %xmm1, %xmm3
    setb %r8b
    xorps %xmm7, %xmm7
    movaps %xmm0, %xmm8
    andps %xmm8, %xmm9
    orps %xmm1, %xmm10
    pxor %xmm11, %xmm11
    movups 0x10(%rdi), %xmm12
    movaps %xmm12, 0x40(%rdi)
    addps %xmm12, %xmm13
    mulps 0x40(%rdi), %xmm13
    shufps $0x1b, %xmm13, %xmm13
    pshufd $0x4e, %xmm12, %xmm14
    unpcklps %xmm12, %xmm15
    minss %xmm0, %xmm2
    maxsd %xmm1, %xmm3
    movd %eax, %xmm7
    movq %xmm1, %r9
    movq %xmm12, 0x50(%rdi)
    movhps 0x58(%rdi), %xmm12
    cvtdq2ps %xmm14, %xmm14
    ret
