.text
.globl _start
_start:
    sub $8, %rsp
    mov $37, %rdi
    mov $0x600000, %rsi
    call f
    add $8, %rsp
    ret
f:
	vmovsd	.LC0(%rip), %xmm3
	vmovss	.LC1(%rip), %xmm1
	movq	%rdi, %r10
	movq	%rsi, %r9
	vmovss	.LC2(%rip), %xmm13
	vmovss	.LC3(%rip), %xmm12
	vxorps	%xmm2, %xmm2, %xmm2
	leaq	(%rdi,%rdi,2), %rdi
	vmovss	.LC6(%rip), %xmm11
	xorl	%esi, %esi
	vxorps	%xmm5, %xmm5, %xmm5
	vmovss	.LC7(%rip), %xmm10
	vmovsd	.LC9(%rip), %xmm9
	vmovq	.LC12(%rip), %xmm6
	vxorpd	%xmm4, %xmm4, %xmm4
	movabsq	$8083404706456994529, %r11
	vmovss	.LC5(%rip), %xmm7
.L9:
	movq	%rdi, %rax
	movl	%esi, %r8d
	mulq	%r11
	movq	%rdi, %rax
	subq	%rdx, %rax
	shrq	%rax
	addq	%rax, %rdx
	shrq	$6, %rdx
	leaq	(%rdx,%rdx,4), %rax
	leaq	(%rdx,%rax,2), %rax
	leaq	(%rdx,%rax,8), %rdx
	movq	%rdi, %rax
	subq	%rdx, %rax
	vcvtsi2ssl	%eax, %xmm2, %xmm0
	vsubss	%xmm13, %xmm0, %xmm0
	vmulss	%xmm0, %xmm0, %xmm15
	vxorps	%xmm7, %xmm0, %xmm8
	vmulss	%xmm12, %xmm15, %xmm14
	vmovaps	%xmm0, %xmm15
	vcmpltss	%xmm5, %xmm0, %xmm0
	vblendvps	%xmm0, %xmm8, %xmm15, %xmm0
#APP
# 1 "/tmp/c_avx.c" 1
	vsqrtss %xmm0, %xmm0, %xmm15
# 0 "" 2
#NO_APP
	vcvtsi2ssl	%esi, %xmm2, %xmm0
	vsubss	%xmm15, %xmm14, %xmm15
	vaddss	%xmm11, %xmm0, %xmm0
	vdivss	%xmm0, %xmm1, %xmm0
	vaddss	%xmm15, %xmm0, %xmm15
	vcomiss	%xmm15, %xmm10
	vmaxss	%xmm1, %xmm15, %xmm1
	vmovss	%xmm15, (%r9,%rsi,4)
	jbe	.L5
	vsubss	.LC8(%rip), %xmm1, %xmm1
.L5:
	movl	%r8d, %ecx
	movq	%r10, %rax
	vcvtss2sd	%xmm15, %xmm15, %xmm15
	addq	$11, %rdi
	vmulsd	%xmm9, %xmm15, %xmm15
	andl	$15, %ecx
	shrq	%cl, %rax
	vcvtsi2sdq	%rax, %xmm2, %xmm0
	vaddsd	%xmm15, %xmm0, %xmm0
	vdivsd	.LC10(%rip), %xmm0, %xmm0
	vcvtsi2sdl	%r8d, %xmm2, %xmm15
	vsubsd	%xmm15, %xmm0, %xmm0
	vmovsd	%xmm0, 2048(%r9,%rsi,8)
	vmovsd	%xmm0, %xmm0, %xmm15
	vxorpd	%xmm6, %xmm0, %xmm14
	addq	$1, %rsi
	vcmpltsd	%xmm4, %xmm0, %xmm0
	vblendvpd	%xmm0, %xmm14, %xmm15, %xmm0
	vaddsd	%xmm0, %xmm3, %xmm3
	cmpq	$48, %rsi
	jne	.L9
	vcvttss2sil	%xmm1, %eax
	vaddss	%xmm1, %xmm1, %xmm0
	xorl	%edx, %edx
	vcomisd	.LC13(%rip), %xmm3
	cltq
	seta	%dl
	movq	%rax, 4800(%r9)
	vcvttsd2siq	%xmm3, %rax
	movq	%rax, 4808(%r9)
	vcvttss2siq	%xmm0, %rax
	vmovss	.LC14(%rip), %xmm0
	movl	%eax, %eax
	movq	%rax, 4816(%r9)
	xorl	%eax, %eax
	vcomiss	%xmm1, %xmm0
	seta	%al
	leal	(%rax,%rdx,2), %eax
	xorl	%edx, %edx
	vucomiss	%xmm1, %xmm1
	vmulss	.LC15(%rip), %xmm1, %xmm1
	setp	%dl
	vcomiss	.LC16(%rip), %xmm1
	leal	(%rax,%rdx,4), %eax
	cltq
	movq	%rax, 4824(%r9)
	jnb	.L11
	vcvttss2siq	%xmm1, %rax
.L12:
	vmulsd	.LC17(%rip), %xmm3, %xmm3
	vmovsd	.LC18(%rip), %xmm0
	vcomisd	%xmm0, %xmm3
	jnb	.L13
	vcvttsd2siq	%xmm3, %rdx
	xorq	%rdx, %rax
	ret
.L11:
	vsubss	.LC16(%rip), %xmm1, %xmm1
	vcvttss2siq	%xmm1, %rax
	btcq	$63, %rax
	jmp	.L12
.L13:
	vsubsd	%xmm0, %xmm3, %xmm3
	vcvttsd2siq	%xmm3, %rdx
	btcq	$63, %rdx
	xorq	%rdx, %rax
	ret
.section .rodata
	.align 8
.LC0:
	.long	0
	.long	1074003968
.section .rodata
	.align 4
.LC1:
	.long	1061158912
	.align 4
.LC2:
	.long	1109393408
	.align 4
.LC3:
	.long	1056964608
.section .rodata
	.align 16
.LC5:
	.long	-2147483648
	.long	0
	.long	0
	.long	0
.section .rodata
	.align 4
.LC6:
	.long	1073741824
	.align 4
.LC7:
	.long	-1063256064
	.align 4
.LC8:
	.long	1065353216
.section .rodata
	.align 8
.LC9:
	.long	0
	.long	1072955392
	.align 8
.LC10:
	.long	0
	.long	1075576832
.section .rodata
	.align 16
.LC12:
	.long	0
	.long	-2147483648
	.long	0
	.long	0
.section .rodata
	.align 8
.LC13:
	.long	0
	.long	1072693248
.section .rodata
	.align 4
.LC14:
	.long	1140457472
	.align 4
.LC15:
	.long	1120403456
	.align 4
.LC16:
	.long	1593835520
.section .rodata
	.align 8
.LC17:
	.long	0
	.long	1076101120
	.align 8
.LC18:
	.long	0
	.long	1138753536
