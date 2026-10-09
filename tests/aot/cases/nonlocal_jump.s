# A jump into the middle of an outer function from a callee (what an exception unwinder or
# longjmp does): B resets the stack to A's frame and jumps to A's landing code L. The translated
# C frames of B and A then unwind by the return-address check until _start's frame matches.
.text
.globl _start
_start:
    sub $8, %rsp
    mov $10, %rbx
    call A
    add %rax, %rbx
    call A
    add %rax, %rbx
    add $8, %rsp
    ret
A:
    push %rbp
    mov %rsp, %rbp
    push %r12
    mov $7, %r12
    mov %rsp, (%rdi)          # the frame to come back to
    call B
    mov $1, %rax              # never reached
    pop %r12
    pop %rbp
    ret
L:                            # the landing code
    mov (%rdi), %rsp
    lea 100(%r12), %rax
    pop %r12
    pop %rbp
    ret
B:
    push %r13
    sub $40, %rsp
    lea L(%rip), %rax
    mov (%rdi), %rsp
    jmp *%rax
