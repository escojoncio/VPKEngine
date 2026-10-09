# contend(rdi = shared block, rsi = iterations): every kind of locked update on shared words,
# aligned and misaligned; run on many threads at once, the totals must be exact.
.text
.globl _start
_start:
1:  lock incq (%rdi)                 # aligned add
    lock addl $2, 0x13(%rdi)         # misaligned (split lock) add
    mov $3, %eax
    lock xaddq %rax, 0x20(%rdi)      # aligned xadd
    lock incl 0x2d(%rdi)             # misaligned inc: CAS loop on the split path
    lock negq 0x40(%rdi)             # twice per iteration below: back to the start value
    lock negq 0x40(%rdi)
    lock btcq $5, 0x48(%rdi)         # toggled twice: unchanged
    lock btcq $5, 0x48(%rdi)
    lock adcq $0, 0x50(%rdi)         # CF from the btc above: +0 or +1 (counted by the host)
    mov 0x58(%rdi), %rax             # an increment through cmpxchg
2:  lea 1(%rax), %rcx
    lock cmpxchgq %rcx, 0x58(%rdi)
    jnz 2b
    lock orq $1, 0x60(%rdi)
    lock xorq $0xff, 0x68(%rdi)
    lock xorq $0xff, 0x68(%rdi)
    dec %rsi
    jnz 1b
    ret
