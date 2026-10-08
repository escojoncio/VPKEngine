# An import: `ext` is real x86 for the native run (rax = rdi * 2 + 1); the translated run
# replaces it with a host function registered at the same address (see harness.c).
# native: ext
.text
.globl _start
_start:
    sub $8, %rsp
    mov $20, %rdi
    call ext
    mov %rax, %rbx
    lea 1(%rbx), %rdi
    call ext
    add %rbx, %rax
    mov %rax, %rdi
    add $8, %rsp
    jmp ext
ext:
    lea 1(%rdi,%rdi), %rax
    ret
