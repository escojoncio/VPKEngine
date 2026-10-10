.text
.globl _start
_start:
    movq $0,0x600(%rdi)
    movq $0,1544(%rdi)
    movq $0,1552(%rdi)
    movq $0,1560(%rdi)
    movq $0,1568(%rdi)
    movq $0,1576(%rdi)
    movq $0,1584(%rdi)
    movq $0,1592(%rdi)
    mov $7,%eax
    xor %edx,%edx
    fld1
    vbroadcastss (%rdi),%ymm14
    vbroadcastss 4(%rdi),%ymm15
    xsave 0x400(%rdi)
    # FIP/FDP (not modeled) and MXCSR_MASK (Intel 0xFFFF, AMD 0x2FFFF) depend on the CPU
    movq $0, 0x408(%rdi)
    movq $0, 0x410(%rdi)
    movl $0, 0x41c(%rdi)
    vzeroall
    fninit
    xrstor 0x400(%rdi)
    fstpl 0x20(%rdi)
    mov $6,%eax
    vzeroall
    xrstor 0x400(%rdi)
    mov $4,%eax
    vzeroall
    xrstor 0x400(%rdi)
    mov $7,%eax
    xrstor 0x400(%rdi)
    ret
