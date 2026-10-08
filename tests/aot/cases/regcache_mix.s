# Register cache: locals vs struct around helper bodies (calls, jump tables, mul/div, cpuid),
# stack ops on rsp itself, loop/jrcxz, rep movs, partial-register writes.
.text
.globl _start
_start:
    sub $8, %rsp
    # call: registers changed before/after must survive the write-back/reload.
    mov $0x1122334455667788, %rbx
    mov $7, %rdi
    call f_mul
    add %rbx, %rax
    mov %rax, 0x10(%rsi)
    # mul/div helpers (write rax/rdx in the struct) then local use of the result.
    mov $0xfffffffffffffff1, %rax
    mov $0x12345, %rcx
    mul %rcx
    add %rdx, %rax
    xor %edx, %edx
    mov $0x1234567890abcdef, %rax
    mov $0x9876, %rcx
    div %rcx
    lea (%rax,%rdx,4), %r9
    # 8-bit / 16-bit partial writes on locals.
    mov $0xdeadbeefcafebabe, %r10
    mov %r9b, %r10b
    mov %r9w, %r11w
    mov %al, %ah
    movb $0x5a, %dh
    movw $0x1234, %r10w
    # cpuid: leaf 0 (deterministic vendor string).
    xor %eax, %eax
    cpuid
    # (the host identity differs from the fixed one: only the register traffic is checked)
    xor %eax, %eax
    xor %ebx, %ebx
    xor %ecx, %ecx
    xor %edx, %edx
    # Stack ops that touch rsp itself.
    push %rsp
    pop %r12
    sub %rsp, %r12
    mov %rsp, %r13
    push %r13
    pop %rsp
    sub %rsp, %r13
    lea -16(%rsp), %rsp
    push $0x77
    pop 8(%rsp)
    pop %r14
    pop %r15
    # jump table via indirect call (register copy must reach the callee).
    mov $3, %rdi
    lea tbl_fn(%rip), %r8
    call *%r8
    mov %rax, 0x40(%rsi)
    # loop / jrcxz on locals.
    mov $5, %rcx
    xor %eax, %eax
1:  add %rcx, %rax
    loop 1b
    jrcxz 2f
    inc %rax
2:  mov %rax, 0x48(%rsi)
    # rep movsb with DF set, then cleared.
    lea 0x100(%rsi), %rdi
    lea 0x200(%rsi), %rdx
    push %rsi
    mov %rdx, %rsi
    mov $0x40, %rcx
    std
    rep movsb
    cld
    mov $0x20, %rcx
    rep movsb
    pop %rsi
    # xchg, cmov, setcc, shifts on partial regs.
    mov $-5, %eax
    mov $9, %ebx
    xchg %eax, %ebx
    cmp %eax, %ebx
    cmovl %eax, %ebx
    setg %bl
    sar $3, %eax
    shl $1, %bh
    rol $4, %r9w
    # Recursive call with flags live across it.
    mov $6, %rdi
    call fact
    mov %rax, 0x50(%rsi)
    add $8, %rsp
    ret

f_mul:
    imul %rdi, %rbx
    lea 1(%rbx,%rdi,2), %rax
    ret

fact:
    cmp $1, %rdi
    jbe 1f
    push %rdi
    dec %rdi
    call fact
    pop %rdi
    imul %rdi, %rax
    ret
1:  mov $1, %eax
    ret

tbl_fn:
    and $3, %rdi
    lea jt(%rip), %r8
    movslq (%r8,%rdi,4), %rax
    add %r8, %rax
    jmp *%rax
c0: mov $0x100, %rax
    ret
c1: mov $0x200, %rax
    ret
c2: mov $0x300, %rax
    ret
c3: lea 0x400(%rdi), %rax
    add %r9, %rax
    ret
.section .rodata
    .align 4
jt:
    .long c0-jt
    .long c1-jt
    .long c2-jt
    .long c3-jt
