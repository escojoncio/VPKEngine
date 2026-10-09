# Adversarial review regression (r4_shl).
.text
.globl _start
_start:
    mov $0x600000, %rdi
    mov $5, %eax
    cmp $7, %eax
    mov $0, %ecx
    shl %cl, %rbx
    pushfq
    pop %r8
    and $0x8d5, %r8
    ret
