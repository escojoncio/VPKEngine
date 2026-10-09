# Adversarial review regression (r4_flags).
.text
.globl _start
_start:
    mov $0x600000, %rdi
    pushfq
    pop %rax
    xor $0x200000, %rax       # ID
    push %rax
    popfq
    pushfq
    pop %rbx
    and $0x200000, %rbx
    push %rax
    xor $0x200000, %rax
    push %rax
    popfq
    popfq
    mov $0x1f, %ecx
    mov $0x0203, %edx
    movabs $0x123456789abcdef1, %rsi
    bextr %rdx, %rsi, %r8
    mov $0x4010, %edx
    bextr %rdx, %rsi, %r9
    mov $0xff3f, %edx
    bextr %rdx, %rsi, %r10
    mov $0xff1f, %edx
    bextr %edx, %esi, %r11d
    mov $0xff20, %edx
    bextr %edx, %esi, %r12d
    mov $0x0000, %edx
    bextr %rdx, %rsi, %r13
    cmp %r13, %r12            # bextr leaves PF/SF/AF undefined (Intel and AMD differ): define them
    ret
