.text
.globl _start
_start:
    mov $37, %rdi
    mov $0x600000, %rsi
    call f
    ret
f:
	movsd	.LC0(%rip), %xmm3
	movss	.LC1(%rip), %xmm2
	movq	%rsi, %r9
	movq	%rdi, %r10
	movss	.LC2(%rip), %xmm7
	xorl	%esi, %esi
	pxor	%xmm11, %xmm11
	movss	.LC5(%rip), %xmm6
	movsd	.LC6(%rip), %xmm5
	pxor	%xmm10, %xmm10
	movsd	.LC7(%rip), %xmm4
	movabsq	$5895351198814392785, %r11
	movq	.LC9(%rip), %xmm8
	movss	.LC4(%rip), %xmm9
.L7:
	movq	%rdi, %rax
	pxor	%xmm0, %xmm0
	movl	%esi, %r8d
	mulq	%r11
	movq	%rdi, %rax
	subq	%rdx, %rax
	shrq	%rax
	addq	%rax, %rdx
	shrq	$6, %rdx
	leaq	(%rdx,%rdx,2), %rax
	salq	$5, %rax
	addq	%rdx, %rax
	movq	%rdi, %rdx
	subq	%rax, %rdx
	cvtsi2ssl	%edx, %xmm0
	subss	%xmm7, %xmm0
	comiss	%xmm0, %xmm11
	movaps	%xmm0, %xmm12
	mulss	%xmm0, %xmm12
	jbe	.L2
	xorps	%xmm9, %xmm0
.L2:
	pxor	%xmm13, %xmm13
	movaps	%xmm2, %xmm1
	movl	%r8d, %ecx
	movq	%r10, %rax
	cvtsi2ssl	%r8d, %xmm13
	andl	$31, %ecx
#APP
# 1 "/tmp/c_float.c" 1
	sqrtss %xmm0, %xmm0
# 0 "" 2
#NO_APP
	subss	%xmm0, %xmm12
	shrq	%cl, %rax
	addss	%xmm6, %xmm13
	divss	%xmm13, %xmm1
	addss	%xmm12, %xmm1
	movaps	%xmm1, %xmm0
	movss	%xmm1, (%r9,%rsi,4)
	cvtss2sd	%xmm1, %xmm1
	maxss	%xmm2, %xmm0
	mulsd	%xmm5, %xmm1
	movaps	%xmm0, %xmm2
	pxor	%xmm0, %xmm0
	cvtsi2sdq	%rax, %xmm0
	addsd	%xmm1, %xmm0
	pxor	%xmm1, %xmm1
	cvtsi2sdl	%r8d, %xmm1
	divsd	%xmm4, %xmm0
	subsd	%xmm1, %xmm0
	movsd	%xmm0, 2048(%r9,%rsi,8)
	comisd	%xmm0, %xmm10
	jbe	.L5
	xorpd	%xmm8, %xmm0
.L5:
	addq	$1, %rsi
	addsd	%xmm0, %xmm3
	addq	$7, %rdi
	cmpq	$64, %rsi
	jne	.L7
	cvttss2sil	%xmm2, %eax
	movss	.LC10(%rip), %xmm0
	xorl	%edx, %edx
	comisd	.LC11(%rip), %xmm3
	mulss	%xmm2, %xmm0
	cltq
	seta	%dl
	movq	%rax, 4800(%r9)
	cvttsd2siq	%xmm3, %rax
	movq	%rax, 4808(%r9)
	cvttss2siq	%xmm0, %rax
	movss	.LC12(%rip), %xmm0
	movl	%eax, %eax
	movq	%rax, 4816(%r9)
	xorl	%eax, %eax
	comiss	%xmm2, %xmm0
	seta	%al
	leal	(%rax,%rdx,2), %eax
	xorl	%edx, %edx
	ucomiss	%xmm2, %xmm2
	mulss	%xmm0, %xmm2
	setnp	%dl
	comiss	.LC13(%rip), %xmm2
	leal	(%rax,%rdx,4), %eax
	cltq
	movq	%rax, 4824(%r9)
	jnb	.L9
	cvttss2siq	%xmm2, %rax
.L10:
	mulsd	.LC14(%rip), %xmm3
	movsd	.LC15(%rip), %xmm0
	comisd	%xmm0, %xmm3
	jnb	.L11
	cvttsd2siq	%xmm3, %rdx
	xorq	%rdx, %rax
	ret
.L9:
	subss	.LC13(%rip), %xmm2
	cvttss2siq	%xmm2, %rax
	btcq	$63, %rax
	jmp	.L10
.L11:
	subsd	%xmm0, %xmm3
	cvttsd2siq	%xmm3, %rdx
	btcq	$63, %rdx
	xorq	%rdx, %rax
	ret
.section .rodata
	.align 8
.LC0:
	.long	0
	.long	1072955392
.section .rodata
	.align 4
.LC1:
	.long	1056964608
	.align 4
.LC2:
	.long	1111490560
.section .rodata
	.align 16
.LC4:
	.long	-2147483648
	.long	0
	.long	0
	.long	0
.section .rodata
	.align 4
.LC5:
	.long	1065353216
.section .rodata
	.align 8
.LC6:
	.long	0
	.long	1073217536
	.align 8
.LC7:
	.long	0
	.long	1074266112
.section .rodata
	.align 16
.LC9:
	.long	0
	.long	-2147483648
	.long	0
	.long	0
.section .rodata
	.align 4
.LC10:
	.long	1077936128
.section .rodata
	.align 8
.LC11:
	.long	0
	.long	1073741824
.section .rodata
	.align 4
.LC12:
	.long	1148846080
	.align 4
.LC13:
	.long	1593835520
.section .rodata
	.align 8
.LC14:
	.long	0
	.long	1076101120
	.align 8
.LC15:
	.long	0
	.long	1138753536
