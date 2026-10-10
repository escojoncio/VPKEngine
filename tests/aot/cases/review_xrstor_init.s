.text
.globl _start
_start:
    fld1
    fld1
    movaps (%rdi),%xmm3
    vbroadcastss (%rdi),%ymm3
    movq $0,0x600(%rdi)
    movq $0,0x608(%rdi)
    movq $0,0x610(%rdi)
    movq $0,0x618(%rdi)
    movq $0,0x620(%rdi)
    movq $0,0x628(%rdi)
    movq $0,0x630(%rdi)
    movq $0,0x638(%rdi)
    movl $0x1f80,0x418(%rdi)
    movl $0xffff,0x41c(%rdi)
    movq $0x03ff,0x400(%rdi)
    mov $7,%eax
    xor %edx,%edx
    xrstor 0x400(%rdi)
    fnstcw 0x10(%rdi)
    fnstsw 0x12(%rdi)
    fnstenv 0x40(%rdi)
    stmxcsr 0x14(%rdi)
    ret
