# requires: sse4a
# SSE4a (AMD only; the PS4's Jaguar has it): extrq / insertq in both forms, movntss / movntsd.
# On an Intel host this case is checked against the golden file written on the AMD CI runner.
.text
.globl _start
_start:
    mov $0x600000, %rdi
    movdqu 0(%rdi), %xmm1
    movdqu 16(%rdi), %xmm2
    movdqa %xmm1, %xmm3
    extrq $8, $4, %xmm3
    movdqa %xmm1, %xmm4
    extrq $0, $0, %xmm4
    movdqa %xmm1, %xmm5
    extrq $40, $24, %xmm5
    movq $0x0c05, %rax
    movq %rax, %xmm6
    movdqa %xmm1, %xmm7
    extrq %xmm6, %xmm7
    movdqa %xmm1, %xmm8
    insertq $12, $20, %xmm2, %xmm8
    movdqa %xmm1, %xmm9
    insertq $0, $0, %xmm2, %xmm9
    movdqa %xmm2, %xmm10
    movq $0x1810, %rax
    pinsrq $1, %rax, %xmm10
    movdqa %xmm1, %xmm11
    insertq %xmm10, %xmm11
    movntss %xmm1, 0x100(%rdi)
    movntsd %xmm2, 0x108(%rdi)
    ret
