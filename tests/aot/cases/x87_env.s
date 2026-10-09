# Adversarial review of the x87 change: fnstenv/fldenv/fxsave/fxrstor control and status words
# (the hardware instruction/data pointers are cleared after each save).
.text
.globl _start
_start:
    mov $0x600000, %rdi
    lea k(%rip), %rsi
    fldz
    fld1
    fdivp
    fstp %st(0)
    movw $0x037b, 0x200(%rdi)
    fldcw 0x200(%rdi)
    fnstsw 0(%rdi)
    fnstcw 2(%rdi)
    fnstenv 0x10(%rdi)
    movq $0, 28(%rdi)
    movl $0, 36(%rdi)
    movw $0, 40(%rdi)
    fnstcw 4(%rdi)
    fnstsw 6(%rdi)
    fnclex
    fninit
    fnstenv 0x40(%rdi)
    movq $0, 76(%rdi)
    movl $0, 84(%rdi)
    movw $0, 88(%rdi)
    movw $0x0080, 0x44(%rdi)
    fldenv 0x40(%rdi)
    fnstsw 8(%rdi)
    fninit
    fnstenv 0x60(%rdi)
    movq $0, 108(%rdi)
    movl $0, 116(%rdi)
    movw $0, 120(%rdi)
    movw $0x0004, 0x64(%rdi)
    movw $0x037b, 0x60(%rdi)
    fldenv 0x60(%rdi)
    fnstsw 10(%rdi)
    fnstcw 12(%rdi)
    fnclex
    fninit
    fxsave 0x100(%rdi)
    movl $0, 284(%rdi)          # MXCSR_MASK: 0xFFFF on Intel, 0x2FFFF on AMD
    movw $0, 262(%rdi)
    movq $0, 264(%rdi)
    movq $0, 272(%rdi)
    movw $0x0000, 0x100(%rdi)
    fxrstor 0x100(%rdi)
    fnstcw 14(%rdi)
    fninit
    fxsave 0x300(%rdi)
    movl $0, 796(%rdi)          # MXCSR_MASK: 0xFFFF on Intel, 0x2FFFF on AMD
    movw $0, 774(%rdi)
    movq $0, 776(%rdi)
    movq $0, 784(%rdi)
    movw $0x0080, 0x302(%rdi)
    fxrstor 0x300(%rdi)
    fnstsw 16(%rdi)
    fninit
    fxsave 0x500(%rdi)
    movl $0, 1308(%rdi)          # MXCSR_MASK: 0xFFFF on Intel, 0x2FFFF on AMD
    movw $0, 1286(%rdi)
    movq $0, 1288(%rdi)
    movq $0, 1296(%rdi)
    movw $0x0001, 0x502(%rdi)
    movw $0x037e, 0x500(%rdi)
    fxrstor 0x500(%rdi)
    fnstsw 18(%rdi)
    fnclex
    fninit
    movw $0x0000, 0x20(%rdi)
    fldcw 0x20(%rdi)
    fnstcw 0x22(%rdi)
    fninit
    fxsave 0x700(%rdi)
    movl $0, 1820(%rdi)          # MXCSR_MASK: 0xFFFF on Intel, 0x2FFFF on AMD
    movw $0, 1798(%rdi)
    movq $0, 1800(%rdi)
    movq $0, 1808(%rdi)
    movw $0,0x706(%rdi)
    movq $0,0x708(%rdi)
    movq $0,0x710(%rdi)
    movl $0,0x8a0(%rdi)

    ret

.section .rodata
.balign 16
k:
    .quad 0x8000000000000000, 0x3fff        # 0: 1.0
    .quad 0xC000000000000000, 0x4000        # 16: 3.0
    .quad 0x4000000000000000, 0x4000        # 32: unnormal (exp 0x4000, int bit 0)
    .quad 0x4000000000000000, 0x7fff        # 48: pseudo-NaN
    .quad 0x0000000000000000, 0x7fff        # 64: pseudo-infinity
    .quad 0x8000000000000001, 0x0000        # 80: pseudo-denormal
    .quad 0x0000000000000123, 0x0000        # 96: denormal
    .quad 0xA000000000000000, 0x4000        # 112: 5.0
    .quad 0x8000000000000001, 0x7ff0        # 128: huge
    .quad 0xC000000000000000, 0x403e        # 144: 3*2^63
    .quad 0x8000000000000000, 0x0001        # 160: min normal
    .quad 0x8000000000000000, 0x407f        # 176: 2^128
    .quad 0xFFFFFFFFFFFFFFFF, 0x3ffe        # 192: just below 1
    .short 300, -300, 32767, -32768         # 208
    .long 0x7f800001                        # 216 f32 SNaN
    .long 0x00000001                        # 220 f32 denormal
    .quad 0x0000000000001234, 0x8000        # 224: negative denormal
    .byte 0x21,0x43,0x65,0x87,0x09,0x21,0x43,0x65,0x87,0x80   # 240: bcd -876543210987654321
