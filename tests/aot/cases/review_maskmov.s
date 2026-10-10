.text
.globl _start
_start:
    movq 0x40(%rdi),%mm0
    movq 0x48(%rdi),%mm1
    maskmovq %mm1,%mm0
    movdqa 0x50(%rdi),%xmm1
    movdqa 0x60(%rdi),%xmm2
    mov %rdi,%r13
    maskmovdqu %xmm2,%xmm1
    vmaskmovdqu %xmm2,%xmm1
    mov %rdi,%rax
    shl $32,%rax
    add $0,%rax
    emms
    ret
