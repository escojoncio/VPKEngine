.text
.globl _start
_start:
    mov $37, %rdi
    mov $0x600000, %rsi
    call f
    ret
f:
	leaq	(%rdi,%rdi,2), %rcx
	xorl	%edx, %edx
	leaq	.L4(%rip), %r8
.L17:
	leaq	(%rdi,%rdx), %rax
	andl	$15, %eax
	cmpq	$11, %rax
	ja	.L2
	movslq	(%r8,%rax,4), %rax
	addq	%r8, %rax
	jmp	*%rax
.section .rodata
	.align 4
	.align 4
.L4:
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
.L5:
	leaq	(%rcx,%rcx,4), %rax
	leaq	(%rcx,%rax,2), %rcx
.L16:
	addq	$1, %rdx
	cmpq	$12, %rdx
	jne	.L17
	movq	%rcx, %rax
	ret
.L6:
	shrq	%rcx
	jmp	.L16
.L8:
	notq	%rcx
	jmp	.L16
.L9:
	orb	$1, %ch
	jmp	.L16
.L7:
	addq	8(%rsi), %rcx
	jmp	.L16
.L10:
	salq	$3, %rcx
	jmp	.L16
.L11:
	subq	(%rsi), %rcx
	jmp	.L16
.L12:
	movq	%rcx, (%rsi,%rdx,8)
	jmp	.L16
.L13:
	leaq	0(,%rcx,8), %rax
	subq	%rcx, %rax
	leaq	1(%rax), %rcx
	jmp	.L16
.L14:
	xorq	$85, %rcx
	jmp	.L16
.L15:
	addq	$1, %rcx
	jmp	.L16
.L3:
	xorq	%rdx, %rcx
	jmp	.L16
.L2:
	addq	%rdx, %rcx
	jmp	.L16
