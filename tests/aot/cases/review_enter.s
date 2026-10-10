.text
.globl _start
_start:
    mov %rsp,%r15
    mov %rbp,%r14
    # fill some "frames" on stack
    lea -0x200(%rsp),%rsp
    mov %rsp,%rbp
    lea 0x100(%rbp),%rbp
    movq $0x1111,-8(%rbp)
    movq $0x2222,-16(%rbp)
    movq $0x3333,-24(%rbp)
    enter $0x20,$0
    mov %rbp,0(%rdi)
    mov %rsp,%rax
    sub %rbp,%rax
    mov %rax,8(%rdi)
    leave
    enter $0x20,$1
    mov %rbp,%rax; sub %rsp,%rax; mov %rax,16(%rdi)
    mov (%rbp),%rax; mov %rax,24(%rdi)
    mov -8(%rbp),%rax; mov %rax,32(%rdi)
    leave
    enter $0x10,$4
    mov %rbp,%rax; sub %rsp,%rax; mov %rax,40(%rdi)
    mov -8(%rbp),%rax; mov %rax,48(%rdi)
    mov -16(%rbp),%rax; mov %rax,56(%rdi)
    mov -24(%rbp),%rax; mov %rax,64(%rdi)
    mov -32(%rbp),%rax; mov %rax,72(%rdi)
    mov (%rbp),%rax; mov %rax,80(%rdi)
    mov %rbp,%rax; sub %rsp,%rax; mov %rax,88(%rdi)
    leave
    enter $0,$31
    mov %rbp,%rax; sub %rsp,%rax; mov %rax,96(%rdi)
    mov %r15,%rsp
    mov %r14,%rbp
    ret
