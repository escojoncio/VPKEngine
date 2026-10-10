.text
.globl _start
_start:
    mov $0x7f,%al
    add $1,%al
    mov $1,%bl
    rcl $9,%bl
    seto 0(%rdi)
    setc 1(%rdi)
    mov $0x7f,%al
    add $1,%al
    mov $1,%bx
    rcr $17,%bx
    seto 2(%rdi)
    stc
    mov $0x80,%cl
    rcl $1,%cl
    seto 3(%rdi)
    ret
