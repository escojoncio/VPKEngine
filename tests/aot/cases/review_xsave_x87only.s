.text
.globl _start
_start:
    movaps (%rdi),%xmm3
    movaps %xmm3,0x1a0(%rdi)
    mov $1,%eax
    xor %edx,%edx
    xsave 0x100(%rdi)
    ret
