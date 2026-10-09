# SSSE3/SSE4.1/SSE4.2/AES/PCLMUL/CRC32/BMI1 and the string instructions, on the harness's
# random data (rdi) and on short strings: every result compared with the hardware.
.text
.globl _start
_start:
    mov $0x600000, %rdi
    aesimc 192(%rdi), %xmm2
    pmaddubsw 848(%rdi), %xmm8
    vpmaddubsw %xmm3, %xmm8, %xmm1
    pblendw $114, %xmm13, %xmm0
    pclmulqdq $0, %xmm0, %xmm9
    mov $6, %eax
    mov $-11, %edx
    pcmpestri $34, %xmm8, %xmm0
    pushfq
    pop %r8
    mov %r8, 8192(%rdi)
    mov %rcx, 8200(%rdi)
    mpsadbw $1, %xmm4, %xmm9
    aeskeygenassist $49, %xmm3, %xmm10
    pblendw $105, %xmm9, %xmm1
    pcmpgtq %xmm6, %xmm8
    phaddsw %xmm12, %xmm3
    vphaddsw %xmm12, %xmm3, %xmm9
    pcmpgtq %xmm14, %xmm7
    vpcmpgtq %xmm14, %xmm7, %xmm1
    pcmpgtq 992(%rdi), %xmm6
    mov $2, %eax
    mov $18, %edx
    pcmpestri $80, %xmm12, %xmm1
    pushfq
    pop %r8
    mov %r8, 8208(%rdi)
    mov %rcx, 8216(%rdi)
    phaddw %xmm7, %xmm12
    pmaddubsw %xmm1, %xmm10
    pclmulqdq $16, %xmm13, %xmm10
    aeskeygenassist $11, %xmm10, %xmm14
    mov $-17, %eax
    mov $-7, %edx
    pcmpestrm $29, %xmm2, %xmm5
    pushfq
    pop %r8
    mov %r8, 8224(%rdi)
    mov %rcx, 8232(%rdi)
    mov $5, %eax
    mov $11, %edx
    pcmpestrm $63, %xmm11, %xmm2
    pushfq
    pop %r8
    mov %r8, 8240(%rdi)
    mov %rcx, 8248(%rdi)
    aesenclast 272(%rdi), %xmm7
    mov $4, %eax
    mov $-6, %edx
    pcmpestri $106, %xmm11, %xmm4
    pushfq
    pop %r8
    mov %r8, 8256(%rdi)
    mov %rcx, 8264(%rdi)
    psignb %xmm2, %xmm2
    vpsignb %xmm2, %xmm2, %xmm13
    pblendw $74, %xmm4, %xmm4
    aesdec %xmm9, %xmm5
    vaesdec %xmm9, %xmm5, %xmm13
    pblendw $233, %xmm10, %xmm9
    mov $5, %eax
    mov $5, %edx
    pcmpestrm $100, %xmm13, %xmm12
    pushfq
    pop %r8
    mov %r8, 8272(%rdi)
    mov %rcx, 8280(%rdi)
    pmaddubsw 416(%rdi), %xmm10
    phsubw 304(%rdi), %xmm9
    pcmpistrm $18, %xmm0, %xmm9
    pushfq
    pop %r8
    mov %r8, 8288(%rdi)
    mov %rcx, 8296(%rdi)
    aeskeygenassist $177, %xmm10, %xmm2
    pblendw $249, %xmm1, %xmm7
    mov $-15, %eax
    mov $-11, %edx
    pcmpestri $123, %xmm7, %xmm7
    pushfq
    pop %r8
    mov %r8, 8304(%rdi)
    mov %rcx, 8312(%rdi)
    psignw %xmm11, %xmm5
    aeskeygenassist $75, %xmm8, %xmm3
    aeskeygenassist $46, %xmm0, %xmm14
    aeskeygenassist $85, %xmm8, %xmm4
    aesenclast %xmm8, %xmm3
    phsubsw %xmm12, %xmm12
    pclmulqdq $16, %xmm3, %xmm3
    aeskeygenassist $241, %xmm12, %xmm0
    pminsb %xmm9, %xmm11
    pcmpistri $56, %xmm1, %xmm5
    pushfq
    pop %r8
    mov %r8, 8320(%rdi)
    mov %rcx, 8328(%rdi)
    phsubsw %xmm5, %xmm3
    aeskeygenassist $43, %xmm7, %xmm0
    pcmpistrm $99, %xmm14, %xmm1
    pushfq
    pop %r8
    mov %r8, 8336(%rdi)
    mov %rcx, 8344(%rdi)
    aesimc %xmm6, %xmm2
    pcmpistri $102, %xmm7, %xmm6
    pushfq
    pop %r8
    mov %r8, 8352(%rdi)
    mov %rcx, 8360(%rdi)
    pblendw $77, %xmm2, %xmm2
    mpsadbw $7, %xmm12, %xmm7
    mpsadbw $0, %xmm2, %xmm5
    phsubw %xmm10, %xmm11
    phsubsw 432(%rdi), %xmm13
    vphsubsw %xmm3, %xmm13, %xmm3
    pcmpistrm $107, %xmm4, %xmm5
    pushfq
    pop %r8
    mov %r8, 8368(%rdi)
    mov %rcx, 8376(%rdi)
    pmaxuw %xmm5, %xmm11
    mpsadbw $2, %xmm14, %xmm13
    pclmulqdq $1, %xmm13, %xmm0
    mpsadbw $2, %xmm12, %xmm12
    phsubw %xmm11, %xmm9
    pblendw $29, %xmm7, %xmm8
    phsubw %xmm0, %xmm4
    mov $19, %eax
    mov $12, %edx
    pcmpestri $113, %xmm1, %xmm14
    pushfq
    pop %r8
    mov %r8, 8384(%rdi)
    mov %rcx, 8392(%rdi)
    aeskeygenassist $231, %xmm11, %xmm3
    mpsadbw $4, %xmm7, %xmm12
    pcmpistrm $114, %xmm3, %xmm14
    pushfq
    pop %r8
    mov %r8, 8400(%rdi)
    mov %rcx, 8408(%rdi)
    pcmpgtq 480(%rdi), %xmm6
    phsubw %xmm12, %xmm4
    vphsubw %xmm12, %xmm4, %xmm11
    aeskeygenassist $70, %xmm2, %xmm5
    mov $11, %eax
    mov $-10, %edx
    pcmpestrm $24, %xmm11, %xmm3
    pushfq
    pop %r8
    mov %r8, 8416(%rdi)
    mov %rcx, 8424(%rdi)
    mov $12, %eax
    mov $5, %edx
    pcmpestrm $41, %xmm3, %xmm13
    pushfq
    pop %r8
    mov %r8, 8432(%rdi)
    mov %rcx, 8440(%rdi)
    pcmpgtq 736(%rdi), %xmm3
    vpcmpgtq %xmm5, %xmm3, %xmm8
    pmaxsb %xmm0, %xmm11
    mpsadbw $1, %xmm1, %xmm1
    phaddsw 256(%rdi), %xmm4
    mov $-11, %eax
    mov $14, %edx
    pcmpestrm $66, %xmm13, %xmm10
    pushfq
    pop %r8
    mov %r8, 8448(%rdi)
    mov %rcx, 8456(%rdi)
    pcmpistri $83, %xmm7, %xmm9
    pushfq
    pop %r8
    mov %r8, 8464(%rdi)
    mov %rcx, 8472(%rdi)
    phaddsw %xmm11, %xmm12
    vphaddsw %xmm11, %xmm12, %xmm0
    pblendw $113, %xmm4, %xmm12
    pmaxuw 848(%rdi), %xmm13
    aesenc %xmm0, %xmm2
    pcmpistrm $46, %xmm0, %xmm4
    pushfq
    pop %r8
    mov %r8, 8480(%rdi)
    mov %rcx, 8488(%rdi)
    mov $8, %eax
    mov $12, %edx
    pcmpestri $52, %xmm4, %xmm10
    pushfq
    pop %r8
    mov %r8, 8496(%rdi)
    mov %rcx, 8504(%rdi)
    pblendw $128, %xmm5, %xmm4
    aesenc %xmm11, %xmm0
    vaesenc %xmm11, %xmm0, %xmm7
    aesimc %xmm1, %xmm7
    pcmpistrm $78, %xmm8, %xmm6
    pushfq
    pop %r8
    mov %r8, 8512(%rdi)
    mov %rcx, 8520(%rdi)
    mov $2, %eax
    mov $-17, %edx
    pcmpestrm $35, %xmm3, %xmm5
    pushfq
    pop %r8
    mov %r8, 8528(%rdi)
    mov %rcx, 8536(%rdi)
    mov $-10, %eax
    mov $-17, %edx
    pcmpestrm $65, %xmm1, %xmm0
    pushfq
    pop %r8
    mov %r8, 8544(%rdi)
    mov %rcx, 8552(%rdi)
    aesenc %xmm6, %xmm13
    vaesenc %xmm6, %xmm13, %xmm3
    mpsadbw $2, %xmm7, %xmm0
    pminsb %xmm4, %xmm0
    aeskeygenassist $111, %xmm0, %xmm3
    pmaxsb 560(%rdi), %xmm0
    pmulhrsw 176(%rdi), %xmm8
    vpmulhrsw %xmm12, %xmm8, %xmm9
    pmuldq %xmm4, %xmm0
    vpmuldq %xmm4, %xmm0, %xmm8
    mov $11, %eax
    mov $-11, %edx
    pcmpestri $99, %xmm10, %xmm2
    pushfq
    pop %r8
    mov %r8, 8560(%rdi)
    mov %rcx, 8568(%rdi)
    phsubd 864(%rdi), %xmm9
    pcmpistrm $4, %xmm14, %xmm2
    pushfq
    pop %r8
    mov %r8, 8576(%rdi)
    mov %rcx, 8584(%rdi)
    aesimc %xmm2, %xmm0
    vaesimc %xmm2, %xmm0
    pcmpistrm $4, %xmm0, %xmm8
    pushfq
    pop %r8
    mov %r8, 8592(%rdi)
    mov %rcx, 8600(%rdi)
    phaddw %xmm7, %xmm0
    pblendw $242, %xmm8, %xmm10
    psignw 416(%rdi), %xmm1
    vpsignw %xmm13, %xmm1, %xmm10
    pcmpistri $97, %xmm13, %xmm7
    pushfq
    pop %r8
    mov %r8, 8608(%rdi)
    mov %rcx, 8616(%rdi)
    pmaddubsw %xmm4, %xmm10
    psignw %xmm5, %xmm2
    pclmulqdq $0, %xmm0, %xmm2
    phsubsw %xmm1, %xmm10
    vphsubsw %xmm1, %xmm10, %xmm8
    phsubw %xmm7, %xmm7
    phminposuw 928(%rdi), %xmm1
    vphminposuw %xmm14, %xmm1
    pclmulqdq $1, %xmm4, %xmm7
    pcmpistri $19, %xmm3, %xmm14
    pushfq
    pop %r8
    mov %r8, 8624(%rdi)
    mov %rcx, 8632(%rdi)
    pminsb 560(%rdi), %xmm8
    pclmulqdq $17, %xmm7, %xmm3
    pmaxuw %xmm7, %xmm0
    phsubw %xmm5, %xmm6
    vphsubw %xmm5, %xmm6, %xmm12
    phsubsw %xmm1, %xmm6
    pmaxsb %xmm1, %xmm5
    psignw %xmm6, %xmm14
    vpsignw %xmm6, %xmm14, %xmm0
    pcmpistrm $38, %xmm10, %xmm4
    pushfq
    pop %r8
    mov %r8, 8640(%rdi)
    mov %rcx, 8648(%rdi)
    pcmpistrm $80, %xmm8, %xmm6
    pushfq
    pop %r8
    mov %r8, 8656(%rdi)
    mov %rcx, 8664(%rdi)
    movq $0x0041424300444546, %rax
    mov %rax, 0x3000(%rdi)
    movq $0x4142434445460047, %rax
    mov %rax, 0x3008(%rdi)
    movdqu 0x3000(%rdi), %xmm1
    movdqu 0x3010(%rdi), %xmm2
    pcmpistri $0, %xmm2, %xmm1
    pushfq
    pop %r8
    mov %r8, 10720(%rdi)
    mov %ecx, 10728(%rdi)
    pcmpistrm $0, %xmm1, %xmm2
    movdqu %xmm0, 10736(%rdi)
    pcmpistri $4, %xmm2, %xmm1
    pushfq
    pop %r8
    mov %r8, 10752(%rdi)
    mov %ecx, 10760(%rdi)
    pcmpistrm $4, %xmm1, %xmm2
    movdqu %xmm0, 10768(%rdi)
    pcmpistri $8, %xmm2, %xmm1
    pushfq
    pop %r8
    mov %r8, 10784(%rdi)
    mov %ecx, 10792(%rdi)
    pcmpistrm $8, %xmm1, %xmm2
    movdqu %xmm0, 10800(%rdi)
    pcmpistri $12, %xmm2, %xmm1
    pushfq
    pop %r8
    mov %r8, 10816(%rdi)
    mov %ecx, 10824(%rdi)
    pcmpistrm $12, %xmm1, %xmm2
    movdqu %xmm0, 10832(%rdi)
    pcmpistri $20, %xmm2, %xmm1
    pushfq
    pop %r8
    mov %r8, 10848(%rdi)
    mov %ecx, 10856(%rdi)
    pcmpistrm $20, %xmm1, %xmm2
    movdqu %xmm0, 10864(%rdi)
    pcmpistri $24, %xmm2, %xmm1
    pushfq
    pop %r8
    mov %r8, 10880(%rdi)
    mov %ecx, 10888(%rdi)
    pcmpistrm $24, %xmm1, %xmm2
    movdqu %xmm0, 10896(%rdi)
    pcmpistri $28, %xmm2, %xmm1
    pushfq
    pop %r8
    mov %r8, 10912(%rdi)
    mov %ecx, 10920(%rdi)
    pcmpistrm $28, %xmm1, %xmm2
    movdqu %xmm0, 10928(%rdi)
    pcmpistri $48, %xmm2, %xmm1
    pushfq
    pop %r8
    mov %r8, 10944(%rdi)
    mov %ecx, 10952(%rdi)
    pcmpistrm $48, %xmm1, %xmm2
    movdqu %xmm0, 10960(%rdi)
    pcmpistri $56, %xmm2, %xmm1
    pushfq
    pop %r8
    mov %r8, 10976(%rdi)
    mov %ecx, 10984(%rdi)
    pcmpistrm $56, %xmm1, %xmm2
    movdqu %xmm0, 10992(%rdi)
    pcmpistri $64, %xmm2, %xmm1
    pushfq
    pop %r8
    mov %r8, 11008(%rdi)
    mov %ecx, 11016(%rdi)
    pcmpistrm $64, %xmm1, %xmm2
    movdqu %xmm0, 11024(%rdi)
    pcmpistri $68, %xmm2, %xmm1
    pushfq
    pop %r8
    mov %r8, 11040(%rdi)
    mov %ecx, 11048(%rdi)
    pcmpistrm $68, %xmm1, %xmm2
    movdqu %xmm0, 11056(%rdi)
    pcmpistri $72, %xmm2, %xmm1
    pushfq
    pop %r8
    mov %r8, 11072(%rdi)
    mov %ecx, 11080(%rdi)
    pcmpistrm $72, %xmm1, %xmm2
    movdqu %xmm0, 11088(%rdi)
    pcmpistri $76, %xmm2, %xmm1
    pushfq
    pop %r8
    mov %r8, 11104(%rdi)
    mov %ecx, 11112(%rdi)
    pcmpistrm $76, %xmm1, %xmm2
    movdqu %xmm0, 11120(%rdi)
    pcmpistri $1, %xmm2, %xmm1
    pushfq
    pop %r8
    mov %r8, 11136(%rdi)
    mov %ecx, 11144(%rdi)
    pcmpistrm $1, %xmm1, %xmm2
    movdqu %xmm0, 11152(%rdi)
    pcmpistri $13, %xmm2, %xmm1
    pushfq
    pop %r8
    mov %r8, 11168(%rdi)
    mov %ecx, 11176(%rdi)
    pcmpistrm $13, %xmm1, %xmm2
    movdqu %xmm0, 11184(%rdi)
    pcmpistri $60, %xmm2, %xmm1
    pushfq
    pop %r8
    mov %r8, 11200(%rdi)
    mov %ecx, 11208(%rdi)
    pcmpistrm $60, %xmm1, %xmm2
    movdqu %xmm0, 11216(%rdi)
    pcmpistri $124, %xmm2, %xmm1
    pushfq
    pop %r8
    mov %r8, 11232(%rdi)
    mov %ecx, 11240(%rdi)
    pcmpistrm $124, %xmm1, %xmm2
    movdqu %xmm0, 11248(%rdi)
    mov (%rdi), %rax
    mov 8(%rdi), %rbx
    crc32b %bl, %eax
    crc32w %bx, %eax
    crc32l %ebx, %eax
    crc32q %rbx, %rax
    crc32b 16(%rdi), %rax
    mov $0x0c04, %ecx
    bextr %ecx, %ebx, %edx
    mov $0x2810, %ecx
    bextr %rcx, %rbx, %rsi
    pushfq
    pop %r9
    lahf
    mov %ah, %dl
    mov %dl, %r10b
    mov $0xd5, %ah
    sahf
    pushfq
    pop %r11
    lea 0x100(%rdi), %rbx
    mov $0x37, %al
    xlat
    mov $0x600000, %rdi
    lea 0x400(%rdi), %rsi
    lea 0x408(%rdi), %rdi
    cld
    mov $5, %ecx
    repe cmpsb
    pushfq
    pop %r12
    mov %rcx, %r13
    mov $0x600000, %rdi
    movb $0x5a, 0x520(%rdi)
    lea 0x500(%rdi), %rdi
    mov $0x5a, %al
    mov $64, %ecx
    repne scasb
    mov %rcx, %r14
    pushfq
    pop %r15
    mov $0x600000, %rdi
    lea 0x600(%rdi), %rsi
    lodsq
    lodsw
    std
    lodsb
    cld
    mov %rsi, %rdx
    mov $0x600000, %rdi
    lea 0x700(%rdi), %rsi
    lea 0x780(%rdi), %rdi
    mov $3, %ecx
    repne cmpsq
    pushfq
    pop %rbp
    mov $0x600000, %rdi
    ret
