# Adversarial review regression: fibers switched by jmp (boost.context style), 200000 switches:
# the host stack must not grow with each switch.
# A user-level context switch (as fiber libraries and coroutines do): each side saves its
# callee-saved registers and stack pointer and loads the other's, then `ret`s into the other
# context. In the translation the `ret` lands on a return address whose host frames are gone
# (they belong to the other context): it is a resume point that vp_run re-enters.
.text
.globl _start
_start:
    push %rbx
    push %rbp
    push %r12
    lea 0x8000(%rdi), %rax         # the coroutine's stack
    lea co_entry(%rip), %rcx
    mov %rcx, -8(%rax)             # where its first switch-in returns to
    sub $56, %rax                  # six saved registers below it
    movq $0, 0(%rax)
    movq $0, 8(%rax)
    movq $0, 16(%rax)
    movq $0, 24(%rax)
    movq $0, 32(%rax)
    movq $0, 40(%rax)
    mov %rax, 0x100(%rdi)
    movq $0, 0x200(%rdi)
    mov $1, %r12
    mov $200000, %ebx
1:
    addq $100, 0x200(%rdi)
    call to_co
    imul $3, %r12, %r12            # main's own registers survive the round trip
    add 0x200(%rdi), %r12
    dec %ebx
    jnz 1b
    mov %r12, %rax
    pop %r12
    pop %rbp
    pop %rbx
    ret

co_entry:
    mov $11, %r13
2:
    addq $3, 0x200(%rdi)
    add %r13, 0x208(%rdi)
    inc %r13
    call to_main
    jmp 2b

to_co:
    push %rbx
    push %rbp
    push %r12
    push %r13
    push %r14
    push %r15
    mov %rsp, 0x108(%rdi)
    mov 0x100(%rdi), %rsp
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbp
    pop %rbx
    pop %r8
    jmp *%r8

to_main:
    push %rbx
    push %rbp
    push %r12
    push %r13
    push %r14
    push %r15
    mov %rsp, 0x100(%rdi)
    mov 0x108(%rdi), %rsp
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbp
    pop %rbx
    pop %r8
    jmp *%r8
