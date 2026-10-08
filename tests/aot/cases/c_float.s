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
	movsd	.LC0(%rip), %xmm2
	movq	%rsi, %r8
	movq	%rdi, %r9
	movss	.LC1(%rip), %xmm1
	movss	.LC2(%rip), %xmm10
	xorl	%esi, %esi
	pxor	%xmm4, %xmm4
	movss	.LC5(%rip), %xmm9
	movsd	.LC6(%rip), %xmm8
	pxor	%xmm3, %xmm3
	movsd	.LC7(%rip), %xmm7
	movabsq	$5895351198814392785, %r10
	movq	.LC9(%rip), %xmm5
	movss	.LC4(%rip), %xmm6
.L7:
	movq	%rdi, %rax
	pxor	%xmm0, %xmm0
	movl	%esi, %ecx
	mulq	%r10
	movq	%rdi, %rax
	andl	$31, %ecx
	subq	%rdx, %rax
	shrq	%rax
	addq	%rax, %rdx
	shrq	$6, %rdx
	leaq	(%rdx,%rdx,2), %rax
	salq	$5, %rax
	addq	%rdx, %rax
	movq	%rdi, %rdx
	addq	$7, %rdi
	subq	%rax, %rdx
	movq	%r9, %rax
	cvtsi2ssl	%edx, %xmm0
	shrq	%cl, %rax
	subss	%xmm10, %xmm0
	movaps	%xmm0, %xmm12
	movaps	%xmm0, %xmm11
	movaps	%xmm0, %xmm13
	mulss	%xmm0, %xmm12
	cmpltss	%xmm4, %xmm0
	xorps	%xmm6, %xmm13
	blendvps	%xmm0, %xmm13, %xmm11
	pxor	%xmm13, %xmm13
	cvtsi2ssl	%esi, %xmm13
#APP
# 1 "/tmp/c_float.c" 1
	sqrtss %xmm11, %xmm0
# 0 "" 2
#NO_APP
	movaps	%xmm1, %xmm11
	subss	%xmm0, %xmm12
	addss	%xmm9, %xmm13
	divss	%xmm13, %xmm11
	addss	%xmm12, %xmm11
	movaps	%xmm11, %xmm0
	movss	%xmm11, (%r8,%rsi,4)
	cvtss2sd	%xmm11, %xmm11
	maxss	%xmm1, %xmm0
	mulsd	%xmm8, %xmm11
	movaps	%xmm0, %xmm1
	pxor	%xmm0, %xmm0
	cvtsi2sdq	%rax, %xmm0
	addsd	%xmm11, %xmm0
	pxor	%xmm11, %xmm11
	cvtsi2sdl	%esi, %xmm11
	divsd	%xmm7, %xmm0
	subsd	%xmm11, %xmm0
	movsd	%xmm0, 2048(%r8,%rsi,8)
	movapd	%xmm0, %xmm11
	movapd	%xmm0, %xmm12
	addq	$1, %rsi
	cmpltsd	%xmm3, %xmm0
	xorpd	%xmm5, %xmm12
	blendvpd	%xmm0, %xmm12, %xmm11
	addsd	%xmm11, %xmm2
	cmpq	$64, %rsi
	jne	.L7
	cvttss2sil	%xmm1, %eax
	movss	.LC10(%rip), %xmm0
	xorl	%edx, %edx
	comisd	.LC11(%rip), %xmm2
	mulss	%xmm1, %xmm0
	cltq
	seta	%dl
	movq	%rax, 4800(%r8)
	cvttsd2siq	%xmm2, %rax
	movq	%rax, 4808(%r8)
	cvttss2siq	%xmm0, %rax
	movss	.LC12(%rip), %xmm0
	movl	%eax, %eax
	movq	%rax, 4816(%r8)
	xorl	%eax, %eax
	comiss	%xmm1, %xmm0
	seta	%al
	leal	(%rax,%rdx,2), %eax
	xorl	%edx, %edx
	ucomiss	%xmm1, %xmm1
	mulss	%xmm0, %xmm1
	setnp	%dl
	comiss	.LC13(%rip), %xmm1
	leal	(%rax,%rdx,4), %eax
	cltq
	movq	%rax, 4824(%r8)
	jnb	.L9
	cvttss2siq	%xmm1, %rax
.L10:
	mulsd	.LC14(%rip), %xmm2
	movsd	.LC15(%rip), %xmm0
	comisd	%xmm0, %xmm2
	jnb	.L11
	cvttsd2siq	%xmm2, %rdx
	xorq	%rdx, %rax
	ret
.L9:
	subss	.LC13(%rip), %xmm1
	cvttss2siq	%xmm1, %rax
	btcq	$63, %rax
	jmp	.L10
.L11:
	subsd	%xmm0, %xmm2
	cvttsd2siq	%xmm2, %rdx
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
