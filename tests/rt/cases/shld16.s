# SHLD / SHRD on 16 bits with a count above 16 (the result is architecturally undefined; the
# translation must still be well defined C). Checked by tests/rt/ubsan_aot.sh, not against the CPU.
.text
.globl _start
_start:
    mov $0x600000, %rdi
    mov $20, %ecx
    mov $0x1234, %eax
    mov $0x8765, %ebx
    shld %cl, %bx, %ax
    mov %ax, 0(%rdi)
    shrd %cl, %bx, %ax
    mov %ax, 2(%rdi)
    mov $31, %ecx
    shld %cl, %bx, %ax
    shrd %cl, %bx, %ax
    mov %ax, 4(%rdi)
    mov $5, %ecx
    shld %cl, %bx, %ax
    mov %ax, 6(%rdi)
    mov $17, %ecx
    shl %cl, %bl
    shl %cl, %ax
    ret
