.text
.globl _start
_start:
    movaps (%rdi),%xmm3
    movaps %xmm3,0x1a0(%rdi)
    mov $1,%eax
    xor %edx,%edx
    xsave 0x100(%rdi)
    # XSTATE_BV bit 0 for an x87 in its initial state: AMD writes 0 (not in use), Intel 1; both allowed
    andb $0xfe, 0x300(%rdi)
    ret
