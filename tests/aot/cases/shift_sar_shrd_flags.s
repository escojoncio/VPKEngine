# SAR CF with a count at or above the width of an 8/16-bit operand, and SHRD OF with a count of one.
.text
.globl _start
_start:
    mov $0x80, %al
    sar $15, %al
    setc 0(%rdi)
    mov $0x8000, %bx
    sar $17, %bx
    setc 1(%rdi)
    mov $0x7f, %cl
    sar $9, %cl
    setc 2(%rdi)
    mov $0x80, %dl
    mov $8, %cl
    sar %cl, %dl
    setc 3(%rdi)
    mov $0x0001, %ax
    mov $0x0001, %dx
    shrd $1, %dx, %ax
    seto 4(%rdi)
    mov $0x00000001, %eax
    mov $0x00000001, %edx
    shrd $1, %edx, %eax
    seto 5(%rdi)
    mov $0x80000000, %r8d
    xor %r9d, %r9d
    shrd $1, %r9d, %r8d
    seto 6(%rdi)
    mov $0x0000000100000000, %r10
    mov $0x1, %r11
    shrd $1, %r11, %r10
    seto 7(%rdi)
    mov $0x1, %r10
    mov $0x1, %r11
    shrd $1, %r11, %r10
    seto 8(%rdi)
    ret
