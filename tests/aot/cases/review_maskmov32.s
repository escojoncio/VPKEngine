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
    # address-size override: edi
    mov %rdi,%r12
    movabs $0x1100600000,%rdi
    .byte 0x67
    maskmovdqu %xmm2,%xmm1
    .byte 0x67
    maskmovq %mm1,%mm0
    mov %r12,%rdi
    emms
    ret
