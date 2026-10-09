# Adversarial review regression (r4_a32).
.text
.globl _start
_start:
    movabs $0xffffffff00600800, %rsi
    addr32 lodsb
    mov %rsi, %r8
    movabs $0xffffffff00600800, %rdi
    mov $3, %ecx
    addr32 repne scasb
    mov %rdi, %r9
    mov $0x600000, %rdi
    ret
