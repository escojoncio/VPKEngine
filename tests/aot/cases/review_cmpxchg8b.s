.text
.globl _start
_start:
    mov %rbx,%r15
    mov $0x1122334455667788,%rax; mov %rax,(%rdi)
    mov $0x55667788,%eax
    mov $0x11223344,%edx
    mov $0xaabbccdd,%ebx
    mov $0x99887766,%ecx
    cmpxchg8b (%rdi)
    setz 16(%rdi)
    mov $0x1,%eax
    mov $0x2,%edx
    cmpxchg8b 1(%rdi)
    setz 17(%rdi)
    mov %rax,24(%rdi)
    mov %rdx,32(%rdi)
    mov %r15,%rbx
    lock cmpxchg8b (%rdi)
    ret
