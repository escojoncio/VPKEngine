# Integer ALU: add/sub/and/or/xor/cmp/test/inc/dec/neg/not with every width and partial registers.
.text
.globl _start
_start:
    add %rcx, %rax
    sub $0x1234, %ecx
    and $0x0f0f, %dx
    or  $0x5a, %bl
    xor %ah, %ch
    inc %r8
    dec %r9d
    neg %r10w
    not %r11b
    cmp %r12, %r13
    setb %al
    seto %cl
    setz %dl
    setle (%rdi)
    test $0x80, %r14b
    sets 1(%rdi)
    add $-1, %r15
    adc $0, %r8
    sbb %r9, %r10
    mov $0x7fffffff, %eax
    add $1, %eax
    seto 2(%rdi)
    lea 3(%rax,%rcx,4), %rdx
    lea -8(%rsp), %rbx
    ret
