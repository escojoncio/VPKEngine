# Reading segment registers: the selectors user-mode code sees (cs 0x33, ss 0x2b), zero-extended
# into a 32/64-bit register, the low 16 bits only with a 16-bit operand, 16 bits into memory.
.text
.globl _start
_start:
    mov $-1, %rax
    mov %cs, %eax
    mov $-1, %rcx
    mov %ss, %rcx
    mov $-1, %rbx
    mov %cs, %bx
    movq $-1, (%rdi)
    mov %cs, (%rdi)
    mov %ss, 2(%rdi)
    ret
