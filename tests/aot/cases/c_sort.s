.text
.globl _start
_start:
    sub $8, %rsp
    mov $37, %rdi
    mov $0x600000, %rsi
    call f
    add $8, %rsp
    ret
qs:
	cmpq	%rdx, %rsi
	jge	.L24
	pushq	%r14
	movq	%rdi, %r14
	pushq	%r13
	movq	%rsi, %r13
	pushq	%r12
	movq	%rdx, %r12
	pushq	%rbp
	pushq	%rbx
.L2:
	leaq	0(%r13,%r12), %rdx
	movq	%r12, %rbx
	movq	%r13, %rbp
	movq	%rdx, %rax
	shrq	$63, %rax
	addq	%rdx, %rax
	sarq	%rax
	movq	(%r14,%rax,8), %rcx
.L7:
	cmpq	%rbx, %rbp
	jg	.L8
.L29:
	leaq	0(,%rbp,8), %rax
	leaq	0(,%rbx,8), %r9
	leaq	(%r14,%rax), %rdi
	leaq	(%r14,%r9), %r8
	movq	(%rdi), %rdx
	movq	(%r8), %rsi
	leaq	8(%r14,%rax), %rax
	cmpq	%rcx, %rdx
	jnb	.L28
.L4:
	movq	%rax, %rdi
	movq	(%rax), %rdx
	addq	$8, %rax
	addq	$1, %rbp
	cmpq	%rcx, %rdx
	jb	.L4
	cmpq	%rsi, %rcx
	jnb	.L5
.L14:
	leaq	-8(%r14,%r9), %rax
.L6:
	movq	%rax, %r8
	movq	(%rax), %rsi
	subq	$8, %rax
	subq	$1, %rbx
	cmpq	%rsi, %rcx
	jb	.L6
.L5:
	cmpq	%rbp, %rbx
	jl	.L7
	movq	%rsi, (%rdi)
	addq	$1, %rbp
	subq	$1, %rbx
	movq	%rdx, (%r8)
.L30:
	cmpq	%rbx, %rbp
	jle	.L29
.L8:
	movq	%rbx, %rdx
	movq	%r12, %rax
	subq	%r13, %rdx
	subq	%rbp, %rax
	cmpq	%rax, %rdx
	jge	.L11
	movq	%r13, %rsi
	movq	%rbx, %rdx
	movq	%r14, %rdi
	movq	%rbp, %r13
	call	qs
.L12:
	cmpq	%r13, %r12
	jg	.L2
	popq	%rbx
	popq	%rbp
	popq	%r12
	popq	%r13
	popq	%r14
	ret
.L28:
	cmpq	%rsi, %rcx
	jb	.L14
	movq	%rsi, (%rdi)
	addq	$1, %rbp
	subq	$1, %rbx
	movq	%rdx, (%r8)
	jmp	.L30
.L11:
	movq	%r12, %rdx
	movq	%rbp, %rsi
	movq	%r14, %rdi
	movq	%rbx, %r12
	call	qs
	jmp	.L12
.L24:
	ret
f:
	pushq	%rbx
	movq	%rdi, %rcx
	movq	%rsi, %rbx
	movq	%rsi, %r10
	leaq	1600(%rsi), %r11
	movabsq	$2361183241434822607, %rdi
.L32:
	movq	%rcx, %rdx
	addq	$8, %rsi
	salq	$13, %rdx
	xorq	%rcx, %rdx
	movq	%rdx, %rax
	shrq	$7, %rax
	xorq	%rdx, %rax
	movq	%rax, %rcx
	salq	$17, %rcx
	xorq	%rax, %rcx
	movq	%rcx, %rdx
	shrq	$3, %rdx
	movq	%rdx, %rax
	mulq	%rdi
	movq	%rcx, %rax
	shrq	$4, %rdx
	imulq	$1000, %rdx, %rdx
	subq	%rdx, %rax
	movq	%rax, -8(%rsi)
	cmpq	%rsi, %r11
	jne	.L32
	movl	$199, %edx
	xorl	%esi, %esi
	movq	%rbx, %rdi
	call	qs
	movabsq	$1469598103934665603, %rax
	movabsq	$1099511628211, %rdx
.L33:
	xorq	(%r10), %rax
	addq	$8, %r10
	imulq	%rdx, %rax
	cmpq	%r10, %r11
	jne	.L33
	movq	1592(%rbx), %rdi
	cmpq	%rdi, (%rbx)
	adcq	$0, %rax
	popq	%rbx
	ret
