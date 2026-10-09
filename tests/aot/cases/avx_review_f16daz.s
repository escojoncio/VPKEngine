# Adversarial review of the AVX change: case kept as a regression test (f16daz).
.text
.globl _start
_start:
    movl $0x1fc0, (%rdi)
    ldmxcsr (%rdi)
    xor %ebx, %ebx
    xor %ecx, %ecx
2:  vmovd %ecx, %xmm0
    vcvtps2ph $2, %xmm0, %xmm1
    vmovd %xmm1, %eax
    xor %rax, %rbx
    rol $5, %rbx
    vcvtps2ph $4, %xmm0, %xmm1
    vmovd %xmm1, %eax
    xor %rax, %rbx
    rol $5, %rbx
    inc %ecx
    cmp $0x2000, %ecx
    jne 2b
    mov %rbx, %r9
    vmovd %ecx, %xmm5
    mov $1, %ecx
    vmovd %ecx, %xmm0
    vcvtps2ph $2, %xmm0, %xmm2
    mov $0x0001, %ecx
    vmovd %ecx, %xmm0
    vcvtph2ps %xmm0, %xmm3
    movl $0x1f80, (%rdi)
    ldmxcsr (%rdi)
    ret
