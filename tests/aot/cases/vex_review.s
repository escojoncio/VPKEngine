# Adversarial VEX cases: shift-by-immediate (3-operand VEX, 2-operand legacy) and
# register-form moves where the destination is also the second source.
.text
.globl _start
_start:
    mov $0x600000, %rdi
    movabs $0x0123456789abcdef, %rax
    movabs $0xfedcba9876543210, %rcx
    mov %rax, (%rdi)
    mov %rcx, 8(%rdi)
    mov %rcx, 16(%rdi)
    mov %rax, 24(%rdi)
    vmovdqu (%rdi), %xmm1
    vmovdqu 16(%rdi), %xmm2
    # shift by immediate: dst != src
    vpslldq $3, %xmm1, %xmm3
    vpsrldq $5, %xmm2, %xmm4
    vpslld $7, %xmm1, %xmm5
    vpsrlq $9, %xmm2, %xmm6
    vpsraw $2, %xmm1, %xmm7
    # shift by xmm count
    vmovq %rax, %xmm8
    mov $3, %rdx
    vmovq %rdx, %xmm9
    vpslld %xmm9, %xmm1, %xmm10
    # dst == src2, src1 differs
    vmovaps %xmm1, %xmm11
    vmovaps %xmm2, %xmm12
    vmovss %xmm12, %xmm11, %xmm12
    vmovaps %xmm1, %xmm13
    vmovaps %xmm2, %xmm14
    vmovsd %xmm14, %xmm13, %xmm14
    vmovaps %xmm1, %xmm15
    vmovaps %xmm2, %xmm0
    vmovlhps %xmm0, %xmm15, %xmm0
    vmovaps %xmm1, %xmm11
    vmovaps %xmm2, %xmm13
    vmovhlps %xmm13, %xmm11, %xmm13
    # sanity: 3/4-operand forms with dst == src2
    vmovaps %xmm1, %xmm11
    vaddps %xmm11, %xmm2, %xmm11
    vshufps $0x1b, %xmm1, %xmm2, %xmm1
    vpinsrq $1, %rcx, %xmm2, %xmm2
    vpextrq $1, %xmm1, %r8
    vcvtsi2ss %r8d, %xmm2, %xmm2
    vroundss $1, %xmm2, %xmm1, %xmm2
    vmovss 4(%rdi), %xmm9
    vmovd %r8d, %xmm11
    vinsertps $0x90, %xmm2, %xmm1, %xmm2
    ret
