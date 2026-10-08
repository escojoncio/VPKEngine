/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * Runs a snippet natively on an x86-64 host from a VpCpu state and stores the final state back.
 * The offsets below are checked against the struct at compile time.
 */
#include "vp_cpu.h"

#include <stddef.h>

_Static_assert(offsetof(VpCpu, r) == 0, "r");
_Static_assert(offsetof(VpCpu, cf) == 136, "cf");
_Static_assert(offsetof(VpCpu, pf) == 137, "pf");
_Static_assert(offsetof(VpCpu, af) == 138, "af");
_Static_assert(offsetof(VpCpu, zf) == 139, "zf");
_Static_assert(offsetof(VpCpu, sf) == 140, "sf");
_Static_assert(offsetof(VpCpu, of) == 141, "of");
_Static_assert(offsetof(VpCpu, df) == 142, "df");
_Static_assert(offsetof(VpCpu, mxcsr) == 144, "mxcsr");
_Static_assert(offsetof(VpCpu, xmm) == 168, "xmm");

uint64_t vp_native_exit_address(void);
void vp_native_run(VpCpu* cpu, uint64_t code);

__asm__(
    ".text\n"
    ".globl vp_native_run\n"
    ".type vp_native_run,@function\n"
    "vp_native_run:\n"
    "  push %rbx\n push %rbp\n push %r12\n push %r13\n push %r14\n push %r15\n"
    "  mov %rsp, vp_saved_rsp(%rip)\n"
    "  mov %rdi, vp_saved_cpu(%rip)\n"
    "  mov %rsi, vp_saved_code(%rip)\n"
    "  ldmxcsr 144(%rdi)\n"
    "  movdqu 168(%rdi), %xmm0\n  movdqu 184(%rdi), %xmm1\n  movdqu 200(%rdi), %xmm2\n  movdqu 216(%rdi), %xmm3\n"
    "  movdqu 232(%rdi), %xmm4\n  movdqu 248(%rdi), %xmm5\n  movdqu 264(%rdi), %xmm6\n  movdqu 280(%rdi), %xmm7\n"
    "  movdqu 296(%rdi), %xmm8\n  movdqu 312(%rdi), %xmm9\n  movdqu 328(%rdi), %xmm10\n movdqu 344(%rdi), %xmm11\n"
    "  movdqu 360(%rdi), %xmm12\n movdqu 376(%rdi), %xmm13\n movdqu 392(%rdi), %xmm14\n movdqu 408(%rdi), %xmm15\n"
    "  push $0x202\n popfq\n"
    "  mov 0(%rdi), %rax\n  mov 8(%rdi), %rcx\n  mov 16(%rdi), %rdx\n  mov 24(%rdi), %rbx\n"
    "  mov 40(%rdi), %rbp\n  mov 48(%rdi), %rsi\n"
    "  mov 64(%rdi), %r8\n  mov 72(%rdi), %r9\n  mov 80(%rdi), %r10\n  mov 88(%rdi), %r11\n"
    "  mov 96(%rdi), %r12\n  mov 104(%rdi), %r13\n  mov 112(%rdi), %r14\n  mov 120(%rdi), %r15\n"
    "  mov 32(%rdi), %rsp\n"
    "  mov 56(%rdi), %rdi\n"
    "  jmp *vp_saved_code(%rip)\n"
    ".globl vp_native_exit\n"
    "vp_native_exit:\n"
    "  mov %rax, vp_saved_rax(%rip)\n"
    "  mov vp_saved_cpu(%rip), %rax\n"
    "  mov %rcx, 8(%rax)\n  mov %rdx, 16(%rax)\n  mov %rbx, 24(%rax)\n  mov %rsp, 32(%rax)\n"
    "  mov %rbp, 40(%rax)\n  mov %rsi, 48(%rax)\n  mov %rdi, 56(%rax)\n"
    "  mov %r8, 64(%rax)\n  mov %r9, 72(%rax)\n  mov %r10, 80(%rax)\n  mov %r11, 88(%rax)\n"
    "  mov %r12, 96(%rax)\n  mov %r13, 104(%rax)\n  mov %r14, 112(%rax)\n  mov %r15, 120(%rax)\n"
    "  mov vp_saved_rax(%rip), %rcx\n  mov %rcx, 0(%rax)\n"
    "  pushfq\n  pop %rcx\n"
    "  mov %rcx, %rdx\n  and $1, %rdx\n  mov %dl, 136(%rax)\n"
    "  mov %rcx, %rdx\n  shr $2, %rdx\n  and $1, %rdx\n  mov %dl, 137(%rax)\n"
    "  mov %rcx, %rdx\n  shr $4, %rdx\n  and $1, %rdx\n  mov %dl, 138(%rax)\n"
    "  mov %rcx, %rdx\n  shr $6, %rdx\n  and $1, %rdx\n  mov %dl, 139(%rax)\n"
    "  mov %rcx, %rdx\n  shr $7, %rdx\n  and $1, %rdx\n  mov %dl, 140(%rax)\n"
    "  mov %rcx, %rdx\n  shr $11, %rdx\n  and $1, %rdx\n  mov %dl, 141(%rax)\n"
    "  mov %rcx, %rdx\n  shr $10, %rdx\n  and $1, %rdx\n  mov %dl, 142(%rax)\n"
    "  stmxcsr 144(%rax)\n"
    "  movdqu %xmm0, 168(%rax)\n  movdqu %xmm1, 184(%rax)\n  movdqu %xmm2, 200(%rax)\n  movdqu %xmm3, 216(%rax)\n"
    "  movdqu %xmm4, 232(%rax)\n  movdqu %xmm5, 248(%rax)\n  movdqu %xmm6, 264(%rax)\n  movdqu %xmm7, 280(%rax)\n"
    "  movdqu %xmm8, 296(%rax)\n  movdqu %xmm9, 312(%rax)\n  movdqu %xmm10, 328(%rax)\n movdqu %xmm11, 344(%rax)\n"
    "  movdqu %xmm12, 360(%rax)\n movdqu %xmm13, 376(%rax)\n movdqu %xmm14, 392(%rax)\n movdqu %xmm15, 408(%rax)\n"
    "  cld\n"
    "  mov vp_saved_rsp(%rip), %rsp\n"
    "  pop %r15\n pop %r14\n pop %r13\n pop %r12\n pop %rbp\n pop %rbx\n"
    "  ret\n"
    ".globl vp_native_exit_address\n"
    "vp_native_exit_address:\n"
    "  lea vp_native_exit(%rip), %rax\n"
    "  ret\n"
    ".data\n"
    "vp_saved_rsp: .quad 0\n"
    "vp_saved_cpu: .quad 0\n"
    "vp_saved_code: .quad 0\n"
    "vp_saved_rax: .quad 0\n"
    ".text\n");
