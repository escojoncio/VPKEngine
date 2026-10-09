# SSE4.1 widening moves (all twelve, register and 2/4/8-byte memory sources), saturating
# adds and subtracts, cvtpd2dq and movntdqa, legacy and VEX: compared with the hardware.
.text
.globl _start
_start:
    mov $0x600000, %rdi
    pmovzxbw %xmm10, %xmm4
    pmovzxbw 3232(%rdi), %xmm1
    vpmovzxbw %xmm10, %xmm2
    vpmovzxbw 6720(%rdi), %xmm3
    pmovzxbd %xmm11, %xmm1
    pmovzxbd 7448(%rdi), %xmm6
    vpmovzxbd %xmm11, %xmm1
    vpmovzxbd 704(%rdi), %xmm13
    pmovzxbq %xmm13, %xmm2
    pmovzxbq 1968(%rdi), %xmm2
    vpmovzxbq %xmm13, %xmm13
    vpmovzxbq 480(%rdi), %xmm3
    pmovzxwd %xmm7, %xmm1
    pmovzxwd 4720(%rdi), %xmm12
    vpmovzxwd %xmm7, %xmm1
    vpmovzxwd 7992(%rdi), %xmm7
    pmovzxwq %xmm1, %xmm4
    pmovzxwq 2368(%rdi), %xmm13
    vpmovzxwq %xmm1, %xmm4
    vpmovzxwq 4424(%rdi), %xmm3
    pmovzxdq %xmm9, %xmm5
    pmovzxdq 840(%rdi), %xmm6
    vpmovzxdq %xmm9, %xmm11
    vpmovzxdq 792(%rdi), %xmm2
    pmovsxbw %xmm1, %xmm6
    pmovsxbw 4064(%rdi), %xmm13
    vpmovsxbw %xmm1, %xmm10
    vpmovsxbw 3808(%rdi), %xmm14
    pmovsxbd %xmm11, %xmm9
    pmovsxbd 2032(%rdi), %xmm5
    vpmovsxbd %xmm11, %xmm7
    vpmovsxbd 664(%rdi), %xmm9
    pmovsxbq %xmm15, %xmm10
    pmovsxbq 5968(%rdi), %xmm14
    vpmovsxbq %xmm15, %xmm9
    vpmovsxbq 4984(%rdi), %xmm2
    pmovsxwd %xmm3, %xmm13
    pmovsxwd 1344(%rdi), %xmm10
    vpmovsxwd %xmm3, %xmm4
    vpmovsxwd 7640(%rdi), %xmm15
    pmovsxwq %xmm13, %xmm1
    pmovsxwq 7880(%rdi), %xmm2
    vpmovsxwq %xmm13, %xmm10
    vpmovsxwq 2784(%rdi), %xmm11
    pmovsxdq %xmm15, %xmm14
    pmovsxdq 560(%rdi), %xmm2
    vpmovsxdq %xmm15, %xmm8
    vpmovsxdq 3880(%rdi), %xmm2
    paddsb %xmm1, %xmm9
    paddsb 5296(%rdi), %xmm14
    vpaddsb %xmm9, %xmm12, %xmm11
    paddsw %xmm0, %xmm14
    paddsw 2896(%rdi), %xmm5
    vpaddsw %xmm3, %xmm15, %xmm1
    paddusb %xmm6, %xmm9
    paddusb 1056(%rdi), %xmm7
    vpaddusb %xmm12, %xmm12, %xmm15
    paddusw %xmm2, %xmm5
    paddusw 3664(%rdi), %xmm12
    vpaddusw %xmm8, %xmm4, %xmm13
    psubsb %xmm8, %xmm13
    psubsb 2928(%rdi), %xmm12
    vpsubsb %xmm7, %xmm4, %xmm2
    psubsw %xmm5, %xmm4
    psubsw 1888(%rdi), %xmm7
    vpsubsw %xmm0, %xmm15, %xmm5
    psubusb %xmm8, %xmm9
    psubusb 32(%rdi), %xmm4
    vpsubusb %xmm13, %xmm11, %xmm10
    psubusw %xmm4, %xmm1
    psubusw 3728(%rdi), %xmm12
    vpsubusw %xmm12, %xmm12, %xmm12
    pmovzxbw %xmm3, %xmm15
    pmovzxbw 5192(%rdi), %xmm12
    vpmovzxbw %xmm3, %xmm1
    vpmovzxbw 1560(%rdi), %xmm2
    pmovzxbd %xmm6, %xmm14
    pmovzxbd 1328(%rdi), %xmm3
    vpmovzxbd %xmm6, %xmm10
    vpmovzxbd 4920(%rdi), %xmm1
    pmovzxbq %xmm3, %xmm0
    pmovzxbq 4640(%rdi), %xmm4
    vpmovzxbq %xmm3, %xmm3
    vpmovzxbq 7768(%rdi), %xmm11
    pmovzxwd %xmm0, %xmm2
    pmovzxwd 7160(%rdi), %xmm6
    vpmovzxwd %xmm0, %xmm12
    vpmovzxwd 1216(%rdi), %xmm8
    pmovzxwq %xmm11, %xmm11
    pmovzxwq 3880(%rdi), %xmm3
    vpmovzxwq %xmm11, %xmm3
    vpmovzxwq 6952(%rdi), %xmm15
    pmovzxdq %xmm14, %xmm15
    pmovzxdq 3960(%rdi), %xmm9
    vpmovzxdq %xmm14, %xmm2
    vpmovzxdq 1176(%rdi), %xmm3
    pmovsxbw %xmm10, %xmm8
    pmovsxbw 3920(%rdi), %xmm5
    vpmovsxbw %xmm10, %xmm0
    vpmovsxbw 1680(%rdi), %xmm11
    pmovsxbd %xmm4, %xmm0
    pmovsxbd 6208(%rdi), %xmm9
    vpmovsxbd %xmm4, %xmm2
    vpmovsxbd 5696(%rdi), %xmm8
    pmovsxbq %xmm11, %xmm5
    pmovsxbq 2912(%rdi), %xmm7
    vpmovsxbq %xmm11, %xmm10
    vpmovsxbq 5208(%rdi), %xmm7
    pmovsxwd %xmm6, %xmm7
    pmovsxwd 6696(%rdi), %xmm12
    vpmovsxwd %xmm6, %xmm7
    vpmovsxwd 1632(%rdi), %xmm15
    pmovsxwq %xmm11, %xmm0
    pmovsxwq 224(%rdi), %xmm8
    vpmovsxwq %xmm11, %xmm15
    vpmovsxwq 2120(%rdi), %xmm6
    pmovsxdq %xmm11, %xmm14
    pmovsxdq 6616(%rdi), %xmm11
    vpmovsxdq %xmm11, %xmm11
    vpmovsxdq 656(%rdi), %xmm7
    paddsb %xmm3, %xmm7
    paddsb 3840(%rdi), %xmm6
    vpaddsb %xmm10, %xmm6, %xmm15
    paddsw %xmm0, %xmm15
    paddsw 7440(%rdi), %xmm11
    vpaddsw %xmm2, %xmm3, %xmm12
    paddusb %xmm6, %xmm15
    paddusb 7280(%rdi), %xmm5
    vpaddusb %xmm13, %xmm10, %xmm2
    paddusw %xmm12, %xmm14
    paddusw 3280(%rdi), %xmm2
    vpaddusw %xmm5, %xmm5, %xmm4
    psubsb %xmm0, %xmm4
    psubsb 4832(%rdi), %xmm14
    vpsubsb %xmm4, %xmm15, %xmm11
    psubsw %xmm4, %xmm4
    psubsw 160(%rdi), %xmm0
    vpsubsw %xmm3, %xmm4, %xmm13
    psubusb %xmm6, %xmm6
    psubusb 224(%rdi), %xmm8
    vpsubusb %xmm6, %xmm9, %xmm7
    psubusw %xmm10, %xmm8
    psubusw 4448(%rdi), %xmm13
    vpsubusw %xmm4, %xmm1, %xmm11
    pmovzxbw %xmm14, %xmm13
    pmovzxbw 6768(%rdi), %xmm4
    vpmovzxbw %xmm14, %xmm4
    vpmovzxbw 4288(%rdi), %xmm0
    pmovzxbd %xmm14, %xmm5
    pmovzxbd 4984(%rdi), %xmm0
    vpmovzxbd %xmm14, %xmm4
    vpmovzxbd 1408(%rdi), %xmm4
    pmovzxbq %xmm15, %xmm3
    pmovzxbq 4552(%rdi), %xmm1
    vpmovzxbq %xmm15, %xmm10
    vpmovzxbq 5584(%rdi), %xmm15
    pmovzxwd %xmm3, %xmm1
    pmovzxwd 2032(%rdi), %xmm6
    vpmovzxwd %xmm3, %xmm8
    vpmovzxwd 344(%rdi), %xmm3
    pmovzxwq %xmm14, %xmm0
    pmovzxwq 6224(%rdi), %xmm2
    vpmovzxwq %xmm14, %xmm14
    vpmovzxwq 2664(%rdi), %xmm6
    pmovzxdq %xmm8, %xmm14
    pmovzxdq 4160(%rdi), %xmm15
    vpmovzxdq %xmm8, %xmm7
    vpmovzxdq 5720(%rdi), %xmm8
    pmovsxbw %xmm6, %xmm14
    pmovsxbw 1120(%rdi), %xmm13
    vpmovsxbw %xmm6, %xmm3
    vpmovsxbw 3208(%rdi), %xmm14
    pmovsxbd %xmm10, %xmm2
    pmovsxbd 5496(%rdi), %xmm7
    vpmovsxbd %xmm10, %xmm13
    vpmovsxbd 592(%rdi), %xmm6
    pmovsxbq %xmm9, %xmm3
    pmovsxbq 7344(%rdi), %xmm4
    vpmovsxbq %xmm9, %xmm11
    vpmovsxbq 1168(%rdi), %xmm8
    pmovsxwd %xmm4, %xmm14
    pmovsxwd 1792(%rdi), %xmm3
    vpmovsxwd %xmm4, %xmm12
    vpmovsxwd 7248(%rdi), %xmm15
    pmovsxwq %xmm5, %xmm7
    pmovsxwq 1320(%rdi), %xmm13
    vpmovsxwq %xmm5, %xmm12
    vpmovsxwq 2776(%rdi), %xmm13
    pmovsxdq %xmm6, %xmm11
    pmovsxdq 2608(%rdi), %xmm2
    vpmovsxdq %xmm6, %xmm11
    vpmovsxdq 152(%rdi), %xmm10
    paddsb %xmm14, %xmm14
    paddsb 5760(%rdi), %xmm0
    vpaddsb %xmm12, %xmm10, %xmm9
    paddsw %xmm2, %xmm3
    paddsw 7520(%rdi), %xmm7
    vpaddsw %xmm3, %xmm2, %xmm8
    paddusb %xmm8, %xmm1
    paddusb 7408(%rdi), %xmm5
    vpaddusb %xmm8, %xmm4, %xmm13
    paddusw %xmm8, %xmm12
    paddusw 1216(%rdi), %xmm15
    vpaddusw %xmm10, %xmm2, %xmm8
    psubsb %xmm1, %xmm5
    psubsb 3472(%rdi), %xmm2
    vpsubsb %xmm8, %xmm0, %xmm2
    psubsw %xmm8, %xmm2
    psubsw 4976(%rdi), %xmm7
    vpsubsw %xmm2, %xmm8, %xmm3
    psubusb %xmm14, %xmm0
    psubusb 2768(%rdi), %xmm13
    vpsubusb %xmm8, %xmm4, %xmm1
    psubusw %xmm7, %xmm3
    psubusw 7936(%rdi), %xmm5
    vpsubusw %xmm8, %xmm1, %xmm5
    mov $123456789, %rax
    cvtsi2sd %rax, %xmm3
    mov $-98765, %rax
    cvtsi2sd %rax, %xmm4
    unpcklpd %xmm4, %xmm3
    mov $3, %rax
    cvtsi2sd %rax, %xmm5
    mov $2, %rax
    cvtsi2sd %rax, %xmm6
    divsd %xmm6, %xmm5
    movlhps %xmm5, %xmm5
    cvtpd2dq %xmm3, %xmm7
    cvtpd2dq %xmm5, %xmm8
    vcvtpd2dq %xmm3, %xmm9
    movdqa %xmm3, 0x2000(%rdi)
    cvtpd2dq 0x2000(%rdi), %xmm10
    movntdqa 0x40(%rdi), %xmm11
    vmovntdqa 0x80(%rdi), %xmm12
    ret
