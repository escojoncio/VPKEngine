# Message passing (rdi = block, rsi = rounds, rdx = role): the writer stores data then flag with plain movs,
# the reader loads flag then data. x86 (TSO) never lets the reader see a flag newer than the data; a weakly
# ordered host does unless the translation orders the accesses (VP_TSO). rax = rounds where it did (reader).
.text
.globl _start
_start:
    test %rdx, %rdx
    jnz reader
    xor %eax, %eax
1:  inc %rax
    mov %rax, 0(%rdi)       # data
    mov %rax, 64(%rdi)      # flag, another cache line
    cmp %rsi, %rax
    jne 1b
    xor %eax, %eax
    ret
reader:
    xor %ecx, %ecx
2:  mov 64(%rdi), %r8       # flag
    mov 0(%rdi), %r9        # data: must be >= flag
    cmp %r8, %r9
    jae 3f
    inc %rcx
3:  cmp %rsi, %r8
    jne 2b
    mov %rcx, %rax
    ret
