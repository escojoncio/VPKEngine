# requires: intel
# 16-bit shld/shrd and 8/16-bit shl with a count above the operand width (the ISA leaves them undefined; the
# Intel hardware does a fixed thing: shld/shrd work on the 48 bits x:y:x), then every count 0..31 against the
# native result, CF, PF, ZF and SF (OF is in shd16_of.s). Intel only (measured there; other vendors differ): on another
# CPU the case is checked against its golden file alone, and run.py never writes a golden file from such a run.
# The counts 0..16, defined in the ISA, are in shd16_defined.s for every vendor. The emitted C and the runtime flag
# helpers must not shift by a negative amount (VPCFLAGS="-fsanitize=undefined -fno-sanitize-recover=all").
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

    # every count 0..31, shrd and shld, three operand pairs: result and CF/PF/ZF/SF, each next to the native result
    mov $0x600000, %r10d
    xor %ebx, %ebx
1:
    mov %ebx, %ecx
    mov $0x1234, %ax
    mov $0xabcd, %dx
    xor %r8d, %r8d
    shrd %cl, %dx, %ax
    pushfq
    pop %r8
    and $0xc5, %r8d
    mov %ax, (%r10)
    mov %r8w, 2(%r10)
    add $4, %r10
    mov $0x1234, %ax
    mov $0xabcd, %dx
    xor %r8d, %r8d
    shld %cl, %dx, %ax
    pushfq
    pop %r8
    and $0xc5, %r8d
    mov %ax, (%r10)
    mov %r8w, 2(%r10)
    add $4, %r10
    mov $0x8001, %ax
    mov $0x7fff, %dx
    xor %r8d, %r8d
    shrd %cl, %dx, %ax
    pushfq
    pop %r8
    and $0xc5, %r8d
    mov %ax, (%r10)
    mov %r8w, 2(%r10)
    add $4, %r10
    mov $0x8001, %ax
    mov $0x7fff, %dx
    xor %r8d, %r8d
    shld %cl, %dx, %ax
    pushfq
    pop %r8
    and $0xc5, %r8d
    mov %ax, (%r10)
    mov %r8w, 2(%r10)
    add $4, %r10
    mov $0xffff, %ax
    mov $0x0000, %dx
    xor %r8d, %r8d
    shrd %cl, %dx, %ax
    pushfq
    pop %r8
    and $0xc5, %r8d
    mov %ax, (%r10)
    mov %r8w, 2(%r10)
    add $4, %r10
    mov $0xffff, %ax
    mov $0x0000, %dx
    xor %r8d, %r8d
    shld %cl, %dx, %ax
    pushfq
    pop %r8
    and $0xc5, %r8d
    mov %ax, (%r10)
    mov %r8w, 2(%r10)
    add $4, %r10
    inc %ebx
    cmp $32, %ebx
    jb 1b
    ret
