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
	vmovss	.LC1(%rip), %xmm3
	pushq	%rbp
	vmovss	.LC2(%rip), %xmm2
	movq	%rsp, %rbp
	vmovss	.LC3(%rip), %xmm1
	pushq	%rbx
	movq	%rdi, %r10
	movq	%rsi, %rbx
	xorl	%ecx, %ecx
	leaq	(%rdi,%rdi,2), %rsi
	leaq	a(%rip), %r8
	xorl	%edi, %edi
	leaq	b(%rip), %r9
	movabsq	$5895351198814392785, %r11
.L2:
	movq	%rsi, %rax
	mulq	%r11
	movq	%rsi, %rax
	subq	%rdx, %rax
	shrq	%rax
	addq	%rax, %rdx
	movq	%rsi, %rax
	shrq	$6, %rdx
	addq	%r10, %rsi
	imulq	$97, %rdx, %rdx
	subq	%rdx, %rax
	movl	%ecx, %edx
	vcvtsi2ssq	%rax, %xmm0, %xmm0
	movl	%ecx, %eax
	imulq	$1321528399, %rax, %rax
	vmulss	%xmm3, %xmm0, %xmm0
	addl	$7, %ecx
	vsubss	%xmm2, %xmm0, %xmm0
	shrq	$34, %rax
	imull	$13, %eax, %eax
	vmovss	%xmm0, (%r8,%rdi)
	subl	%eax, %edx
	vcvtsi2ssl	%edx, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	vmovss	%xmm0, (%r9,%rdi)
	addq	$4, %rdi
	cmpl	$448, %ecx
	jne	.L2
	xorl	%eax, %eax
	leaq	c(%rip), %rdx
.L3:
	vmovaps	(%r8,%rax), %ymm0
	vmovaps	(%r9,%rax), %ymm1
	vmulps	%ymm0, %ymm1, %ymm2
	vmaxps	%ymm1, %ymm0, %ymm0
	vaddps	%ymm2, %ymm0, %ymm0
	vmovaps	%ymm0, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$256, %rax
	jne	.L3
	leaq	c(%rip), %rax
	leaq	h(%rip), %rdx
	leaq	256(%rax), %rcx
.L4:
	vmovaps	(%rax), %ymm0
	addq	$32, %rax
	addq	$16, %rdx
	vcvtps2ph	$0, %ymm0, %xmm1
	vmovdqa	%xmm1, -16(%rdx)
	vcvtph2ps	%xmm1, %ymm1
	vsubps	%ymm1, %ymm0, %ymm0
	vmovaps	%ymm0, -32(%rax)
	cmpq	%rcx, %rax
	jne	.L4
	movl	$1, %eax
	leaq	-4+c(%rip), %rsi
	leaq	-2+h(%rip), %rcx
	vxorpd	%xmm1, %xmm1, %xmm1
.L5:
	vcvtsi2sdl	%eax, %xmm2, %xmm2
	movzwl	(%rcx,%rax,2), %edx
	vcvtss2sd	(%rsi,%rax,4), %xmm0, %xmm0
	vmulsd	%xmm2, %xmm0, %xmm0
	incq	%rax
	cmpq	$65, %rax
	vcvtsi2sdl	%edx, %xmm2, %xmm2
	vaddsd	%xmm2, %xmm0, %xmm0
	vaddsd	%xmm0, %xmm1, %xmm1
	jne	.L5
	vmulsd	.LC5(%rip), %xmm1, %xmm2
	vxorpd	.LC4(%rip), %xmm1, %xmm0
	vmovsd	.LC6(%rip), %xmm4
	vmovsd	.LC9(%rip), %xmm3
	vunpcklpd	%xmm0, %xmm4, %xmm0
	vunpcklpd	%xmm1, %xmm2, %xmm2
	vmulsd	.LC10(%rip), %xmm1, %xmm1
	vinsertf128	$0x1, %xmm2, %ymm0, %ymm0
	vbroadcastsd	.LC8(%rip), %ymm2
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vmulpd	%ymm2, %ymm0, %ymm0
	vmulsd	%xmm3, %xmm0, %xmm2
	vextractf128	$0x1, %ymm0, %xmm0
	vunpckhpd	%xmm0, %xmm0, %xmm0
	vmulsd	%xmm3, %xmm0, %xmm0
	vcvttsd2siq	%xmm2, %rdx
	movq	%rdx, (%rbx)
	vcvttsd2siq	%xmm0, %rax
	movq	%rax, 8(%rbx)
	vcvttsd2siq	%xmm1, %rax
	xorq	%rdx, %rax
	vzeroupper
	movq	-8(%rbp), %rbx
	leave
	ret
	.local	h
	.comm	h,128,32
	.local	c
	.comm	c,256,32
	.local	b
	.comm	b,256,32
	.local	a
	.comm	a,256,32
.section .rodata
	.align 4
.LC1:
	.long	1048576000
	.align 4
.LC2:
	.long	1086324736
	.align 4
.LC3:
	.long	1056964608
.section .rodata
	.align 16
.LC4:
	.long	0
	.long	-2147483648
	.long	0
	.long	0
.section .rodata
	.align 8
.LC5:
	.long	0
	.long	1071644672
	.align 8
.LC6:
	.long	0
	.long	1074266112
	.align 8
.LC8:
	.long	0
	.long	1073217536
	.align 8
.LC9:
	.long	0
	.long	1083129856
	.align 8
.LC10:
	.long	0
	.long	1085276160
