# Shifts, rotates, multiplication and division.
.text
.globl _start
_start:
    mov $5, %cl
    shl %cl, %rax
    shr $3, %edx
    sar $1, %bx
    rol $7, %r8d
    ror %cl, %r9
    shl $0, %r10d
    mov $0x12345678, %eax
    mov $-7, %ecx
    imul %ecx, %eax
    imul $100, %rdx, %rdx
    mov $0x8000000000000000, %rax
    mov $3, %rcx
    mul %rcx
    mov $-1000, %rax
    cqo
    mov $7, %rcx
    idiv %rcx
    mov $0xffffffff, %eax
    xor %edx, %edx
    mov $16, %ecx
    div %ecx
    mov $0x80, %al
    imul %r11b
    bsf %r12, %r13
    bsr %r12d, %r14d
    bswap %r15
    popcnt %r12, %rbx
    tzcnt %r8w, %r9w
    bt $5, %r12
    setc %al
    bts $63, %r13
    ret
