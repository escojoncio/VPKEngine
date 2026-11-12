# requires: undefined_isa
# 16-bit shld/shrd and 8/16-bit shl with a count above the operand width (flags or result undefined in the ISA, so no
# native run: golden only). The emitted C and the runtime flag helpers must not shift by a negative amount
# (VPCFLAGS="-fsanitize=undefined -fno-sanitize-recover=all").
.text
.globl _start
_start:
    mov $0x1234, %ax
    mov $0xabcd, %dx
    shrd $20, %dx, %ax
    mov $0x8001, %cx
    shrd $17, %dx, %cx
    mov $0x4321, %bx
    shrd $16, %dx, %bx
    mov $0x1234, %si
    shld $20, %dx, %si
    mov $0x8001, %di
    shld $31, %dx, %di
    mov $0x81, %r8b
    shl $20, %r8b
    mov $0x8001, %r9w
    shl $18, %r9w
    ret
