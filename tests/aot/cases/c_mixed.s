.text
.globl _start
_start:
    mov $37, %rdi
    mov $0x600000, %rsi
    call f
    ret
f:
	pushq	%r15
	movq	%rsi, %r10
	movq	%rsi, %rax
	xorl	%r11d, %r11d
	pushq	%r14
	movl	$-20000, %r9d
	movl	%edi, %r14d
	xorl	%edx, %edx
	pushq	%r13
	movl	$-100, %r8d
	movl	%edi, %r13d
	xorl	%ecx, %ecx
	pushq	%r12
	movl	%edi, %r12d
	pushq	%rbp
	movabsq	$-7046029254386353131, %rbp
	pushq	%rbx
	imulq	%rdi, %rbp
.L2:
	leal	(%r14,%rcx), %ebx
	movw	%dx, 2(%rax)
	addq	$24, %rax
	addl	%r13d, %edx
	movb	%bl, -24(%rax)
	movq	%rdi, %rbx
	shrq	%cl, %rbx
	movb	%r8b, -8(%rax)
	addl	$13, %r8d
	movl	%ebx, -20(%rax)
	leaq	0(%rbp,%rcx), %rbx
	addq	$1, %rcx
	movq	%rbx, -16(%rax)
	movl	%r11d, %ebx
	addl	$77777, %r11d
	xorl	%r12d, %ebx
	movw	%r9w, -6(%rax)
	addw	$1000, %r9w
	movl	%ebx, -4(%rax)
	cmpq	$40, %rcx
	jne	.L2
	leaq	960(%r10), %r11
	xorl	%r13d, %r13d
	movabsq	$7905747460161236407, %r9
	movabsq	$-5270498306774157605, %r8
	movabsq	$2635249153387078801, %rcx
	movabsq	$896011011859258473, %rbp
	movabsq	$-6148914691236517205, %rbx
	jmp	.L5
.L3:
	testl	%r12d, %r12d
	jns	.L4
	negq	%rdx
	movq	%rdx, %rax
	movq	%rdx, %r12
	mulq	%rbp
	movq	%r12, %rax
	subq	%rdx, %rax
	shrq	%rax
	addq	%rax, %rdx
	shrq	$19, %rdx
	imulq	$1000003, %rdx, %rax
	movq	%r12, %rdx
	subq	%rax, %rdx
.L4:
	movq	%rdx, %r12
	movq	%rdx, %r13
	xorl	%eax, %eax
	addq	$24, %rsi
	orq	$256, %r12
	orq	$1, %r13
	popcntq	%rdx, %rax
	addq	%rax, %rdx
	tzcntq	%r12, %r12
	lzcntq	%r13, %r13
	addq	%r12, %r13
	addq	%rdx, %r13
	cmpq	%rsi, %r11
	je	.L10
.L5:
	movq	8(%rsi), %rax
	movsbq	16(%rsi), %rdx
	movswq	18(%rsi), %r14
	movl	20(%rsi), %r12d
	leaq	(%rdx,%rdx,2), %rdx
	movzbl	(%rsi), %r15d
	addq	%rax, %r14
	imulq	%r9, %rax
	addq	%r14, %rdx
	movslq	%r12d, %r14
	subq	%r14, %rdx
	movzwl	2(%rsi), %r14d
	addq	%r8, %rax
	addl	%r15d, %r14d
	addl	4(%rsi), %r14d
	addq	%r14, %rdx
	addq	%r13, %rdx
	rolq	$5, %rdx
	cmpq	%rax, %rcx
	jb	.L3
	movq	%rdx, %rax
	mulq	%rbx
	shrq	%rdx
	jmp	.L4
.L10:
	movq	%r13, %rax
	popq	%rbx
	popq	%rbp
	mulq	%rdi
	orq	$1, %rdi
	popq	%r12
	movq	%rax, %rcx
	movq	%rdx, 7200(%r10)
	movq	%r13, %rax
	xorl	%edx, %edx
	divq	%rdi
	movq	%rax, 7208(%r10)
	movq	%r13, %rax
	movq	%rdx, 7216(%r10)
	cqto
	idivq	%rdi
	movq	%rax, 7224(%r10)
	movq	%r13, %rax
	popq	%r13
	xorq	%rcx, %rax
	popq	%r14
	popq	%r15
	ret
