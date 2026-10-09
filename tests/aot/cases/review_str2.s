# Adversarial review regression (r4_str2).
.text
.globl _start
_start:
    mov $0x600000, %rdi
    mov $0x600400, %rsi
    mov $5, %eax
    cmp $7, %eax
    mov $0, %ecx
    repe cmpsb
    pushfq
    pop %r8
    and $0x8d5, %r8
    ret
