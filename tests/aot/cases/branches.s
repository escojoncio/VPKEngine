# Conditional branches, loops, cmov, a tail call.
.text
.globl _start
_start:
    xor %eax, %eax
    mov $10, %ecx
1:
    add %ecx, %eax
    dec %ecx
    jnz 1b
    cmp $55, %eax
    jne fail
    mov $3, %r8
    mov $4, %r9
    cmp %r8, %r9
    cmovg %r9, %r8
    cmovl %r9, %r10
    test %r11, %r11
    js neg
    mov $1, %ebx
    jmp done
neg:
    mov $2, %ebx
done:
    mov $5, %ecx
2:
    loop 2b
    jmp tail
fail:
    mov $0xbad, %eax
    ret
tail:
    inc %rax
    ret
