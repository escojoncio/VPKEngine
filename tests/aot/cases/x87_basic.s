# The x87 FPU (exact parts): loads and stores of every format, arithmetic in every form, the
# precision and rounding controls, compares and condition codes, stack faults, exceptions,
# fprem/fprem1/fscale/fxtract, constants, environment save/restore. Compared bit for bit with the
# hardware: registers, tags, TOP, status and control words, and memory.
.text
.globl _start
_start:
    mov $0x600000, %rdi
    lea k(%rip), %rsi
    # loads of each format, integer loads
    flds 0(%rsi)              # 1.5f
    fldl 8(%rsi)              # 0.1
    fldt 16(%rsi)             # an 80-bit value with low bits set
    filds 32(%rsi)            # -1234
    fildl 36(%rsi)            # 100000007
    fildll 40(%rsi)           # big
    # arithmetic, every form
    fadd %st(1), %st
    fsub %st, %st(2)
    fmulp %st, %st(3)
    fdivr %st(2), %st
    fsubrp %st, %st(1)
    fdivp %st, %st(1)
    fadds 0(%rsi)
    fmull 8(%rsi)
    fisubl 36(%rsi)
    fidivrs 32(%rsi)
    fstpt 0x00(%rdi)
    fstpt 0x10(%rdi)
    fstpt 0x20(%rdi)
    # precision control 24 and 53 bits, every rounding mode
    fnstcw 0x30(%rdi)
    movw $0x007f, 0x32(%rdi)      # PC=24, RC=nearest
    fldcw 0x32(%rdi)
    fldt 16(%rsi)
    fldl 8(%rsi)
    fdivr %st(1), %st
    fstpt 0x40(%rdi)
    movw $0x0e7f, 0x34(%rdi)      # PC=53, RC=toward zero
    fldcw 0x34(%rdi)
    fldl 8(%rsi)
    fdivr %st(1), %st
    fstpt 0x50(%rdi)
    movw $0x0b7f, 0x36(%rdi)      # PC=64, RC=up
    fldcw 0x36(%rdi)
    fldl 8(%rsi)
    fdiv %st(1), %st
    fsts 0x60(%rdi)
    fstl 0x68(%rdi)
    fistl 0x70(%rdi)
    fistps 0x74(%rdi)
    movw $0x077f, 0x38(%rdi)      # RC=down
    fldcw 0x38(%rdi)
    fldl 8(%rsi)
    fchs
    fistpll 0x78(%rdi)
    fldpi
    fldl2e
    fldl2t
    fldlg2
    fldln2
    fstpt 0x80(%rdi)
    fstpt 0x90(%rdi)
    fstpt 0xa0(%rdi)
    fstpt 0xb0(%rdi)
    fstpt 0xc0(%rdi)
    fldcw 0x30(%rdi)              # back to the default
    fstp %st(0)
    # integer stores out of range, fisttp
    fldl 48(%rsi)                 # 1e30
    fistl 0xd0(%rdi)
    fists 0xd4(%rdi)
    fisttpll 0xd8(%rdi)
    fldl 8(%rsi)
    fchs
    fisttpl 0xe0(%rdi)
    fnstsw 0xe4(%rdi)
    fnclex
    # sqrt, rndint, chs, abs
    fldt 16(%rsi)
    fsqrt
    fld %st(0)
    frndint
    fabs
    fchs
    fxch %st(1)
    fstpt 0x100(%rdi)
    fstpt 0x110(%rdi)
    # compares
    fld1
    fldz
    fcom %st(1)
    fnstsw %ax
    mov %ax, 0x120(%rdi)
    fcomi %st(1), %st
    seta 0x122(%rdi)
    setb 0x123(%rdi)
    setz 0x124(%rdi)
    fucomip %st(1), %st
    setp 0x125(%rdi)
    fcoms 0(%rsi)
    fnstsw 0x126(%rdi)
    ficoml 36(%rsi)
    fnstsw 0x128(%rdi)
    ftst
    fnstsw 0x12a(%rdi)
    fxam
    fnstsw 0x12c(%rdi)
    fcmovb %st(0), %st
    stc
    fldz
    fcmovb %st(1), %st
    fcmovne %st(1), %st
    fstpt 0x130(%rdi)
    fcompp
    fnstsw 0x140(%rdi)
    # NaN compares: invalid only for fcom (and signalling NaNs for fucom)
    flds 56(%rsi)                 # quiet NaN
    fld1
    fucom %st(1)
    fnstsw 0x142(%rdi)
    fcom %st(1)
    fnstsw 0x144(%rdi)
    fnclex
    fucompp
    # exceptions: zero divide, invalid, overflow, underflow, denormal operand
    fldz
    fld1
    fdivp %st, %st(1)
    fnstsw 0x146(%rdi)
    fstpt 0x150(%rdi)
    fnclex
    fld1
    fchs
    fsqrt
    fnstsw 0x160(%rdi)
    fstpt 0x170(%rdi)
    fnclex
    fldt 64(%rsi)                 # huge
    fmul %st(0), %st
    fnstsw 0x180(%rdi)
    fstpt 0x190(%rdi)
    fnclex
    fldt 80(%rsi)                 # tiny
    fmul %st(0), %st
    fnstsw 0x1a0(%rdi)
    fstpt 0x1b0(%rdi)
    fnclex
    flds 60(%rsi)                 # a float denormal
    fnstsw 0x1c0(%rdi)
    fstpt 0x1d0(%rdi)
    fnclex
    # fprem / fprem1 (quotient bits in C0 C3 C1), fscale, fxtract
    fldl 96(%rsi)                 # 7.5
    fldl 104(%rsi)                # 100.25
    fprem
    fnstsw 0x1e0(%rdi)
    fstpt 0x1f0(%rdi)
    fldl 104(%rsi)
    fprem1
    fnstsw 0x200(%rdi)
    fstpt 0x210(%rdi)
    fldl 112(%rsi)                # -1e10
    fprem1
    fnstsw 0x220(%rdi)
    fstpt 0x230(%rdi)
    fstp %st(0)
    fildl 36(%rsi)
    fldl 104(%rsi)
    fscale
    fstpt 0x240(%rdi)
    fstp %st(0)
    fldl 112(%rsi)
    fxtract
    fstpt 0x250(%rdi)
    fstpt 0x260(%rdi)
    # stack overflow and underflow
    fld1
    fld1
    fld1
    fld1
    fld1
    fld1
    fld1
    fld1
    fld1                          # ninth: overflow
    fnstsw 0x270(%rdi)
    fnclex
    ffree %st(3)
    fincstp
    fincstp
    fincstp
    fadd %st(0), %st              # ST(0) now empty: underflow
    fnstsw 0x272(%rdi)
    fnclex
    fdecstp
    # environment and full save, with the hardware's instruction/data pointers cleared
    fnstenv 0x300(%rdi)
    movq $0, 0x30c(%rdi)
    movq $0, 0x314(%rdi)
    movl $0, 0x31c(%rdi)
    fldenv 0x300(%rdi)
    fnsave 0x340(%rdi)
    movq $0, 0x34c(%rdi)
    movq $0, 0x354(%rdi)
    movl $0, 0x35c(%rdi)
    fld1
    frstor 0x340(%rdi)
    fstpt 0x3c0(%rdi)
    fxsave 0x400(%rdi)
    movw $0, 0x406(%rdi)
    movq $0, 0x408(%rdi)
    movq $0, 0x410(%rdi)
    movl $0, 0x41c(%rdi)
    movl $0, 0x5a0(%rdi)          # the reserved area hardware may leave untouched
    ret

.section .rodata
.balign 16
k:
    .float 1.5                       # 0
    .float 0                         # 4
    .double 0.1                      # 8
    .quad 0xd3a4b2c1f0e9a817         # 16: 80-bit, significand
    .short 0x4003, 0, 0, 0           # 24
    .short -1234                     # 32
    .short 0
    .long 100000007                  # 36
    .quad 0x123456789abcdef          # 40
    .double 1e30                     # 48
    .long 0x7fc00000                 # 56: quiet NaN
    .long 0x00000005                 # 60: float denormal
    .quad 0x8000000000000001         # 64: huge 80-bit
    .short 0x7ff0, 0, 0, 0
    .quad 0x8000000000000001         # 80: tiny 80-bit
    .short 0x0010, 0, 0, 0
    .double 7.5                      # 96
    .double 100.25                   # 104
    .double -1e10                    # 112
