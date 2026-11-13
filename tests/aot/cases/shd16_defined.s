# 16-bit shld/shrd with the counts 0..16, which the ISA defines (result, CF, PF, ZF, SF; OF only for count 1, in
# shift_sar_shrd_flags.s): the same on every vendor, so no `requires`. The counts above 16 are in shrd16_wide_count.s.
.text
.globl _start
_start:
    # every count 0..16
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
    movq $0, -8(%rsp)   # scrub the pushed rflags (AF, OF are undefined here and differ between vendors)
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
    movq $0, -8(%rsp)   # scrub the pushed rflags (AF, OF are undefined here and differ between vendors)
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
    movq $0, -8(%rsp)   # scrub the pushed rflags (AF, OF are undefined here and differ between vendors)
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
    movq $0, -8(%rsp)   # scrub the pushed rflags (AF, OF are undefined here and differ between vendors)
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
    movq $0, -8(%rsp)   # scrub the pushed rflags (AF, OF are undefined here and differ between vendors)
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
    movq $0, -8(%rsp)   # scrub the pushed rflags (AF, OF are undefined here and differ between vendors)
    and $0xc5, %r8d
    mov %ax, (%r10)
    mov %r8w, 2(%r10)
    add $4, %r10
    inc %ebx
    cmp $17, %ebx
    jb 1b
    ret
