# Loads and stores of every width, push/pop, rep movs/stos, calls and returns.
.text
.globl _start
_start:
    push %rbx
    push %r12
    movq %rax, (%rdi)
    movl %ecx, 8(%rdi)
    movw %dx, 12(%rdi)
    movb %bl, 14(%rdi)
    movzbl 14(%rdi), %eax
    movswq 12(%rdi), %rcx
    movsbl 14(%rdi), %edx
    movslq 8(%rdi), %rbx
    addq $0x10, 0x20(%rdi)
    subl %eax, 0x28(%rdi)
    incw 0x2c(%rdi)
    xchg %rax, 0x30(%rdi)
    call helper
    mov %rax, %r12
    mov $0x40, %ecx
    cld
    rep movsb
    mov $0x11, %al
    mov $0x20, %ecx
    rep stosb
    mov $0x8, %ecx
    rep stosq
    mov %r12, %rax
    pop %r12
    pop %rbx
    ret
helper:
    push %rbp
    mov %rsp, %rbp
    sub $0x20, %rsp
    mov %rdi, -8(%rbp)
    mov -8(%rbp), %rax
    add $1, %rax
    leave
    ret
