.text
.globl _start
_start:
    sub $8, %rsp
    mov $37, %rdi
    mov $0x600000, %rsi
    call f
    add $8, %rsp
    ret
add:
	leaq	(%rdi,%rsi), %rax
	ret
mul:
	imulq	%rsi, %rdi
	leaq	1(%rdi), %rax
	ret
xr:
	xorq	%rsi, %rdi
	leaq	(%rdi,%rdi), %rax
	ret
fib:
	movq	%rdi, %rax
	cmpq	$1, %rdi
	je	.L67
	pushq	%r15
	leaq	-1(%rdi), %rdx
	subq	$4, %rax
	pushq	%r14
	leaq	-2(%rdi), %r14
	andq	$-2, %rdx
	pushq	%r13
	pushq	%r12
	movq	%r14, %r12
	pushq	%rbp
	subq	%rdx, %r12
	movq	%r14, %rdx
	xorl	%ebp, %ebp
	pushq	%rbx
	andq	$-2, %rdx
	subq	%rdx, %rax
	subq	$184, %rsp
	movq	%rax, 112(%rsp)
	cmpq	%r14, %r12
	je	.L7
.L72:
	movq	%r12, 72(%rsp)
	xorl	%eax, %eax
	movq	%r14, %rbx
	movq	%r14, 80(%rsp)
	movq	%rbp, 64(%rsp)
	movq	%rax, %rbp
.L36:
	cmpq	$1, %rbx
	je	.L65
	leaq	-1(%rbx), %rcx
	xorl	%eax, %eax
	movq	%rbp, 88(%rsp)
	movq	%rbx, %rbp
	movq	%rax, %r13
	movq	%rcx, %rsi
	movq	%rcx, %r14
.L33:
	cmpq	$1, %r14
	je	.L64
	movq	%r14, 104(%rsp)
	xorl	%eax, %eax
	leaq	-1(%r14), %r8
	movq	%r13, 96(%rsp)
	movq	%r8, %rbx
	movq	%rax, %r13
	movq	%rbp, %rax
.L30:
	cmpq	$1, %rbx
	je	.L63
	leaq	-1(%rbx), %r11
	xorl	%r15d, %r15d
	movq	%rsi, %r9
	movq	%rax, %r10
	movq	%r13, %rsi
	movq	%r11, %r14
	movq	%r15, %r13
	movq	%rbx, %rbp
.L27:
	cmpq	$1, %r14
	je	.L62
	leaq	-1(%r14), %r15
	xorl	%edx, %edx
	movq	%rsi, %r12
	movq	%r15, %rbx
.L24:
	cmpq	$1, %rbx
	je	.L61
	leaq	-1(%rbx), %rcx
	leaq	-2(%rbx), %rsi
	movq	%rbx, %rdi
	movq	%r10, 8(%rsp)
	movq	%rcx, %rax
	movq	%rsi, 16(%rsp)
	andq	$-2, %rax
	subq	%rax, %rdi
	xorl	%eax, %eax
	movq	%rax, %r10
	movq	8(%rsp), %rax
	movq	%rdi, 56(%rsp)
	movq	%rdx, %rdi
	movq	%r9, %rdx
.L21:
	movq	56(%rsp), %r9
	cmpq	%r9, %rbx
	je	.L13
	movq	16(%rsp), %r9
	movq	%rdi, 24(%rsp)
	movq	$0, 8(%rsp)
	movq	%r10, 32(%rsp)
	movq	%rax, 40(%rsp)
	movq	%rdx, 48(%rsp)
	cmpq	$1, %r9
	je	.L59
.L70:
	movq	%r9, %r10
	xorl	%edx, %edx
.L15:
	leaq	-1(%r10), %rdi
	movq	%r9, 168(%rsp)
	movq	%rsi, 160(%rsp)
	movq	%rcx, 152(%rsp)
	movq	%r11, 144(%rsp)
	movq	%r8, 136(%rsp)
	movq	%rdx, 128(%rsp)
	movq	%r10, 120(%rsp)
	call	fib
	movq	120(%rsp), %r10
	movq	128(%rsp), %rdx
	movq	136(%rsp), %r8
	movq	144(%rsp), %r11
	subq	$2, %r10
	addq	%rax, %rdx
	movq	152(%rsp), %rcx
	movq	160(%rsp), %rsi
	cmpq	$1, %r10
	movq	168(%rsp), %r9
	ja	.L15
	movq	8(%rsp), %rax
	leaq	-2(%r9), %rdi
	movq	%rdi, %r10
	andq	$-2, %r10
	leaq	-2(%rax,%r9), %rax
	subq	%r10, %rax
	addq	%rdx, %rax
	movq	%rax, 8(%rsp)
	cmpq	$2, %r9
	je	.L59
	movq	%rdi, %r9
	cmpq	$1, %r9
	jne	.L70
.L59:
	movq	8(%rsp), %r9
	movq	32(%rsp), %r10
	subq	$2, %rbx
	subq	$2, 16(%rsp)
	movq	24(%rsp), %rdi
	addq	$1, %r9
	movq	40(%rsp), %rax
	movq	48(%rsp), %rdx
	addq	%r9, %r10
	cmpq	$1, %rbx
	jne	.L21
	movq	%rax, %rbx
	movq	%r10, %rax
	movq	%rdx, %r9
	movq	%rdi, %rdx
	addq	$1, %rax
	movq	%rbx, %r10
	addq	%rax, %rdx
	cmpq	$1, %rcx
	jne	.L40
.L61:
	movq	%r12, %rsi
	movq	%rdx, %r12
	subq	$2, %r14
	addq	$1, %r12
	addq	%r12, %r13
	cmpq	$1, %r15
	jne	.L27
.L62:
	movq	%r13, %r15
	movq	%rbp, %rbx
	movq	%rsi, %r13
	movq	%r10, %rax
	addq	$1, %r15
	movq	%r9, %rsi
	subq	$2, %rbx
	addq	%r15, %r13
	cmpq	$1, %r11
	jne	.L30
.L63:
	movq	104(%rsp), %r14
	movq	%rax, %rbp
	movq	%r13, %rax
	movq	96(%rsp), %r13
	addq	$1, %rax
	addq	%rax, %r13
	subq	$2, %r14
	cmpq	$1, %r8
	jne	.L33
.L64:
	movq	%rbp, %rbx
	movq	%r13, %rax
	movq	88(%rsp), %rbp
	addq	$1, %rax
	subq	$2, %rbx
	addq	%rax, %rbp
	cmpq	$1, %rsi
	jne	.L36
.L65:
	movq	%rbp, %rax
	movq	80(%rsp), %r14
	movq	64(%rsp), %rbp
	addq	$1, %rax
	movq	72(%rsp), %r12
	addq	%rax, %rbp
	leaq	-2(%r14), %rax
	cmpq	%rax, 112(%rsp)
	je	.L71
	movq	%rax, %r14
	cmpq	%r14, %r12
	jne	.L72
.L7:
	leaq	1(%rbp,%r12), %rax
.L6:
	addq	$184, %rsp
	popq	%rbx
	popq	%rbp
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	ret
.L13:
	movq	%rax, 8(%rsp)
	movq	%r10, %rax
	movq	%rdx, %r9
	movq	%rdi, %rdx
	leaq	-1(%rbx,%rax), %rax
	movq	8(%rsp), %r10
	addq	%rax, %rdx
	cmpq	$1, %rcx
	je	.L61
.L40:
	movq	%rsi, %rbx
	jmp	.L24
.L67:
	ret
.L71:
	leaq	0(%rbp,%r14), %rax
	jmp	.L6
f:
	pushq	%r15
	leaq	add(%rip), %rcx
	leaq	mul(%rip), %rax
	pushq	%r14
	movq	%rcx, %xmm0
	pushq	%r13
	pinsrq	$1, %rax, %xmm0
	leaq	xr(%rip), %rax
	pushq	%r12
	xorl	%r12d, %r12d
	pushq	%rbp
	movq	%rdi, %rbp
	pushq	%rbx
	subq	$56, %rsp
	movq	$1, (%rsi)
	movq	%rax, 32(%rsp)
	movaps	%xmm0, 16(%rsp)
.L75:
	leaq	(%r12,%rbp), %rcx
	movq	%rbp, %rdi
	andl	$7, %ebp
	movabsq	$-6148914691236517205, %rax
	mulq	%rcx
	leaq	0(,%r12,8), %rsi
	xorl	%r15d, %r15d
	subq	%r12, %rsi
	leaq	5(%rbp), %r14
	movq	%r14, %rbx
	movq	%rdx, %rax
	andq	$-2, %rdx
	shrq	%rax
	addq	%rax, %rdx
	subq	%rdx, %rcx
	call	*16(%rsp,%rcx,8)
	movq	%rax, %r13
.L74:
	leaq	-1(%rbx), %rdi
	subq	$2, %rbx
	call	fib
	addq	%rax, %r15
	cmpq	$1, %rbx
	ja	.L74
	addq	$3, %rbp
	leaq	-2(%r13,%r14), %rbx
	addq	$1, %r12
	andl	$14, %ebp
	subq	%rbp, %rbx
	leaq	(%rbx,%r15), %rbp
	cmpq	$30, %r12
	jne	.L75
	testb	$1, %bpl
	leaq	add(%rip), %rax
	movq	%rbp, %rdi
	movl	$99, %esi
	leaq	mul(%rip), %rdx
	cmovne	%rdx, %rax
	movq	%rax, 8(%rsp)
	movq	8(%rsp), %rax
	addq	$56, %rsp
	popq	%rbx
	popq	%rbp
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	jmp	*%rax
