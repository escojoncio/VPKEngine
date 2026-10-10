# rcl/rcr of 8/16 bits by an immediate that is a multiple of 9/17 rotate by 0 through CF: CF must stay the
# previous producer's (lazy flags must keep computing it). OF there is undefined (AMD writes it, Intel not).
.text
.globl _start
_start:
    mov $0xff,%al
    add $1,%al
    mov $1,%bl
    rcl $9,%bl
    setc 0(%rdi)
    mov %bl,1(%rdi)
    mov $0xff,%al
    add $1,%al
    mov $1,%bx
    rcr $17,%bx
    setc 2(%rdi)
    mov %bx,4(%rdi)
    stc
    mov $0x80,%cl
    rcl $1,%cl
    seto 3(%rdi)
    ret
