/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * Like tests/aot/native_x86.c, but the flags (including DF) start as the VpCpu state says, not all clear.
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
_Static_assert(offsetof(VpCpu, ymmh) == 424, "ymmh");

uint64_t vp_native_exit_address(void);
void vp_native_run_raw(VpCpu* cpu, uint64_t code);
/* The x87 state goes in and out through FXRSTOR/FXSAVE of this area, in the layout the runtime's
 * vp_x87_fxsave/fxrstor use. */
__attribute__((aligned(64))) unsigned char vp_native_fx[512];

void vp_native_run(VpCpu* cpu, uint64_t code) {
    vp_x87_fxsave(cpu, (uint64_t)(uintptr_t)vp_native_fx);
    vp_native_run_raw(cpu, code);
    vp_x87_fxrstor(cpu, (uint64_t)(uintptr_t)vp_native_fx);
}

__asm__(
    ".text\n"
    ".globl vp_native_run_raw\n"
    ".type vp_native_run_raw,@function\n"
    "vp_native_run_raw:\n"
    "  push %rbx\n push %rbp\n push %r12\n push %r13\n push %r14\n push %r15\n"
    "  mov %rsp, vp_saved_rsp(%rip)\n"
    "  mov %rdi, vp_saved_cpu(%rip)\n"
    "  mov %rsi, vp_saved_code(%rip)\n"
    "  fxrstor vp_native_fx(%rip)\n"
    "  ldmxcsr 144(%rdi)\n"
    "  vmovdqu 168(%rdi), %xmm0\n  vinsertf128 $1, 424(%rdi), %ymm0, %ymm0\n  vmovdqu 184(%rdi), %xmm1\n  vinsertf128 $1, 440(%rdi), %ymm1, %ymm1\n  vmovdqu 200(%rdi), %xmm2\n  vinsertf128 $1, 456(%rdi), %ymm2, %ymm2\n  vmovdqu 216(%rdi), %xmm3\n  vinsertf128 $1, 472(%rdi), %ymm3, %ymm3\n  vmovdqu 232(%rdi), %xmm4\n  vinsertf128 $1, 488(%rdi), %ymm4, %ymm4\n  vmovdqu 248(%rdi), %xmm5\n  vinsertf128 $1, 504(%rdi), %ymm5, %ymm5\n  vmovdqu 264(%rdi), %xmm6\n  vinsertf128 $1, 520(%rdi), %ymm6, %ymm6\n  vmovdqu 280(%rdi), %xmm7\n  vinsertf128 $1, 536(%rdi), %ymm7, %ymm7\n  vmovdqu 296(%rdi), %xmm8\n  vinsertf128 $1, 552(%rdi), %ymm8, %ymm8\n  vmovdqu 312(%rdi), %xmm9\n  vinsertf128 $1, 568(%rdi), %ymm9, %ymm9\n  vmovdqu 328(%rdi), %xmm10\n  vinsertf128 $1, 584(%rdi), %ymm10, %ymm10\n  vmovdqu 344(%rdi), %xmm11\n  vinsertf128 $1, 600(%rdi), %ymm11, %ymm11\n  vmovdqu 360(%rdi), %xmm12\n  vinsertf128 $1, 616(%rdi), %ymm12, %ymm12\n  vmovdqu 376(%rdi), %xmm13\n  vinsertf128 $1, 632(%rdi), %ymm13, %ymm13\n  vmovdqu 392(%rdi), %xmm14\n  vinsertf128 $1, 648(%rdi), %ymm14, %ymm14\n  vmovdqu 408(%rdi), %xmm15\n  vinsertf128 $1, 664(%rdi), %ymm15, %ymm15\n"
    "  movzbl 136(%rdi), %eax\n"
    "  movzbl 137(%rdi), %ecx\n  shl $2, %ecx\n  or %ecx, %eax\n"
    "  movzbl 138(%rdi), %ecx\n  shl $4, %ecx\n  or %ecx, %eax\n"
    "  movzbl 139(%rdi), %ecx\n  shl $6, %ecx\n  or %ecx, %eax\n"
    "  movzbl 140(%rdi), %ecx\n  shl $7, %ecx\n  or %ecx, %eax\n"
    "  movzbl 142(%rdi), %ecx\n  shl $10, %ecx\n  or %ecx, %eax\n"
    "  movzbl 141(%rdi), %ecx\n  shl $11, %ecx\n  or %ecx, %eax\n"
    "  or $0x202, %eax\n  push %rax\n popfq\n"
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
    "  fxsave vp_native_fx(%rip)\n"
    "  stmxcsr 144(%rax)\n"
    "  vmovdqu %xmm0, 168(%rax)\n  vextractf128 $1, %ymm0, 424(%rax)\n  vmovdqu %xmm1, 184(%rax)\n  vextractf128 $1, %ymm1, 440(%rax)\n  vmovdqu %xmm2, 200(%rax)\n  vextractf128 $1, %ymm2, 456(%rax)\n  vmovdqu %xmm3, 216(%rax)\n  vextractf128 $1, %ymm3, 472(%rax)\n  vmovdqu %xmm4, 232(%rax)\n  vextractf128 $1, %ymm4, 488(%rax)\n  vmovdqu %xmm5, 248(%rax)\n  vextractf128 $1, %ymm5, 504(%rax)\n  vmovdqu %xmm6, 264(%rax)\n  vextractf128 $1, %ymm6, 520(%rax)\n  vmovdqu %xmm7, 280(%rax)\n  vextractf128 $1, %ymm7, 536(%rax)\n  vmovdqu %xmm8, 296(%rax)\n  vextractf128 $1, %ymm8, 552(%rax)\n  vmovdqu %xmm9, 312(%rax)\n  vextractf128 $1, %ymm9, 568(%rax)\n  vmovdqu %xmm10, 328(%rax)\n  vextractf128 $1, %ymm10, 584(%rax)\n  vmovdqu %xmm11, 344(%rax)\n  vextractf128 $1, %ymm11, 600(%rax)\n  vmovdqu %xmm12, 360(%rax)\n  vextractf128 $1, %ymm12, 616(%rax)\n  vmovdqu %xmm13, 376(%rax)\n  vextractf128 $1, %ymm13, 632(%rax)\n  vmovdqu %xmm14, 392(%rax)\n  vextractf128 $1, %ymm14, 648(%rax)\n  vmovdqu %xmm15, 408(%rax)\n  vextractf128 $1, %ymm15, 664(%rax)\n"
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
