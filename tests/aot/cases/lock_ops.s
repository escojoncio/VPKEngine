# LOCK-prefixed read-modify-writes: the ones without a single atomic counterpart run as a
# compare-and-swap loop; misaligned addresses (split locks) take the runtime's locked path.
# Also bt/bts/btr/btc with a register bit offset into memory (signed, outside the operand).
.text
.globl _start
_start:
    mov $0x600000, %rdi
    movq $5, 0x100(%rdi)
    lock incq 0x100(%rdi)
    lock decl 0x104(%rdi)
    lock negw 0x102(%rdi)
    lock notb 0x101(%rdi)
    lock btsq $40, 0x108(%rdi)
    setc %al
    lock btrl $3, 0x110(%rdi)
    setc %bl
    lock btcw $9, 0x118(%rdi)
    setc %cl
    stc
    lock adcl $7, 0x120(%rdi)
    lock sbbq %rdx, 0x128(%rdi)
    lock andl $0xf0f0f0f0, 0x130(%rdi)
    lock orq %rsi, 0x138(%rdi)
    lock xorw $0x5a5a, 0x140(%rdi)
    lock addl $100, 0x144(%rdi)
    lock subq %r8, 0x148(%rdi)
    # misaligned: split locks
    lock incq 0x203(%rdi)
    lock xaddl %r9d, 0x20d(%rdi)
    mov 0x215(%rdi), %eax
    lock cmpxchgl %r10d, 0x215(%rdi)
    lock cmpxchgq %r11, 0x223(%rdi)
    xchg %r12, 0x231(%rdi)
    lock andq %r13, 0x23b(%rdi)
    lock addw $3, 0x24f(%rdi)
    # bit offsets in registers, positive and negative
    mov $77, %rdx
    btsq %rdx, 0x300(%rdi)
    setc %r14b
    mov $-13, %rdx
    btrq %rdx, 0x340(%rdi)
    mov $-1000, %edx
    lock btcl %edx, 0x400(%rdi)
    setc %r15b
    mov $35, %dx
    btw %dx, 0x380(%rdi)
    adc $0, %r14
    ret
