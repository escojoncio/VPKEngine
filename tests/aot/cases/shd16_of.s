# requires: intel
# OF of 16-bit shld/shrd for every count 0..31 (undefined in the ISA for counts above 1; the Intel hardware gives the 1-bit shift formula,
# whatever the count). Intel only: other vendors differ, the case is then checked against its golden file alone.
.text
.globl _start
_start:
    # every count 0..31, shrd and shld, three operand pairs: result and OF, each next to the native result
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
    and $0x800, %r8d
    mov %ax, (%r10)
    mov %r8w, 2(%r10)
    add $4, %r10
    mov $0x1234, %ax
    mov $0xabcd, %dx
    xor %r8d, %r8d
    shld %cl, %dx, %ax
    pushfq
    pop %r8
    and $0x800, %r8d
    mov %ax, (%r10)
    mov %r8w, 2(%r10)
    add $4, %r10
    mov $0x8001, %ax
    mov $0x7fff, %dx
    xor %r8d, %r8d
    shrd %cl, %dx, %ax
    pushfq
    pop %r8
    and $0x800, %r8d
    mov %ax, (%r10)
    mov %r8w, 2(%r10)
    add $4, %r10
    mov $0x8001, %ax
    mov $0x7fff, %dx
    xor %r8d, %r8d
    shld %cl, %dx, %ax
    pushfq
    pop %r8
    and $0x800, %r8d
    mov %ax, (%r10)
    mov %r8w, 2(%r10)
    add $4, %r10
    mov $0xffff, %ax
    mov $0x0000, %dx
    xor %r8d, %r8d
    shrd %cl, %dx, %ax
    pushfq
    pop %r8
    and $0x800, %r8d
    mov %ax, (%r10)
    mov %r8w, 2(%r10)
    add $4, %r10
    mov $0xffff, %ax
    mov $0x0000, %dx
    xor %r8d, %r8d
    shld %cl, %dx, %ax
    pushfq
    pop %r8
    and $0x800, %r8d
    mov %ax, (%r10)
    mov %r8w, 2(%r10)
    add $4, %r10
    inc %ebx
    cmp $32, %ebx
    jb 1b
    ret
