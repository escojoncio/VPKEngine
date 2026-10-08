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
	pushq	%r15
	movq	%rdi, %r15
	movq	%rdi, %rax
	movq	%rdi, %r10
	pushq	%r14
	xorl	%r11d, %r11d
	xorl	%ecx, %ecx
	negq	%r15
	pushq	%r13
	pushq	%r12
	pushq	%rbp
	leaq	800(%rsi), %rbp
	pushq	%rbx
	movq	%rsi, %rbx
	leaq	1600(%rsi), %rsi
	leaq	2480(%rbx), %r12
	leaq	2488(%rbx), %r13
	leaq	2496(%rbx), %r14
.L2:
	movl	%ecx, %edi
	lock xaddl	%edi, 0(%rbp)
	movslq	%edi, %rdi
	addq	%rax, %rdi
	movq	%r15, %rax
	lock xaddq	%rax, (%rsi)
	xorq	%rax, %rdi
	movq	1600(%rbx), %rax
	lock cmpxchgq	%rdi, (%rsi)
	movl	%ecx, %edx
	andl	$3, %edx
	leaq	300(%rdx), %r8
	leaq	(%rax,%rdi), %rdx
	movq	%rdx, %rax
	xchgq	(%rbx,%r8,8), %rax
	leaq	1(%rax), %rdx
	movzbl	%dl, %eax
	lock orq	%rax, (%r12)
	movq	%rdx, %rax
	notq	%rax
	lock andq	%rax, 0(%r13)
	lock xorq	%rdx, (%r14)
	xorl	%r8d, %r8d
	xorl	%edi, %edi
	movq	%rdx, %r9
	orq	%r10, %r8
	orq	%r11, %r9
	shrdq	%r9, %r8
	shrq	%cl, %r9
	testb	$64, %cl
	cmovne	%r9, %r8
	cmovne	%rdi, %r9
	addl	$1, %ecx
	movq	%r9, %rax
	xorq	%r8, %rax
	addq	%rdx, %rax
	rolq	$7, %rax
	cmpl	$20, %ecx
	jne	.L2
	popq	%rbx
	popq	%rbp
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	ret
