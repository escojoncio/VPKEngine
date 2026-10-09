/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * A minimal host for translated code: enters it at a guest address, resolves indirect calls and
 * jumps through the entry table, and reports what the translation could not do. The game
 * runtimes replace vp_cpuid/vp_rdtsc and the fault handlers with their own.
 */
#include "vp_emit.h"

#include <fenv.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Per thread: a game runs translated code on many threads, each with its own VpCpu. */
/* Per thread, and nested: a native the guest called may run guest code again (callbacks), so
 * every vp_run keeps its own exit point and restores the outer one when it returns. */
static _Thread_local jmp_buf* vp_exit_jump;
static _Thread_local VpCpu* vp_run_cpu; /* the state the innermost vp_run was given */
#define vp_exit_armed (vp_exit_jump != NULL)

static const VpEntry* vp_find(uint64_t guest) {
    size_t lo = 0, hi = vp_entry_count;
    if (vp_tables_relative) guest -= vp_image_base;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (vp_entries[mid].guest == guest) return &vp_entries[mid];
        if (vp_entries[mid].guest < guest) lo = mid + 1; else hi = mid;
    }
    return NULL;
}

#define VP_MAX_NATIVES 4096
static struct { uint64_t guest; VpNative fn; } vp_natives[VP_MAX_NATIVES];
static size_t vp_native_count;

/* Registration happens before the threads start; the table is kept sorted for the lookups. */
void vp_register_native(uint64_t guest, VpNative fn) {
    size_t i;
    for (i = 0; i < vp_native_count; ++i) {
        if (vp_natives[i].guest == guest) { vp_natives[i].fn = fn; return; }
        if (vp_natives[i].guest > guest) break;
    }
    if (vp_native_count >= VP_MAX_NATIVES) return;
    for (size_t k = vp_native_count; k > i; --k) vp_natives[k] = vp_natives[k - 1];
    vp_natives[i].guest = guest;
    vp_natives[i].fn = fn;
    vp_native_count++;
}

static VpNative vp_find_native(uint64_t guest) {
    size_t lo = 0, hi = vp_native_count;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (vp_natives[mid].guest == guest) return vp_natives[mid].fn;
        if (vp_natives[mid].guest < guest) lo = mid + 1; else hi = mid;
    }
    return NULL;
}

void vp_call_native(VpCpu* c, uint64_t guest) {
    VpNative fn = vp_find_native(guest);
    if (!fn) { vp_unsupported(c, guest, "native not registered"); return; }
    fn(c);
}

void vp_dispatch(VpCpu* c, uint64_t target) {
    const VpEntry* e = vp_find(target);
    if (e) {
        e->function(c, 0);
        return;
    }
    if (vp_find_native(target)) {
        vp_call_native(c, target);
        return;
    }
    const uint64_t key = vp_tables_relative ? target - vp_image_base : target;
    for (size_t i = 0; i < vp_extra_entry_count; ++i) {
        if (vp_extra_entries[i].guest == key) {
            vp_extra_entries[i].function(c, vp_extra_entries[i].entry);
            return;
        }
    }
    /* vp_host_exit (the trampoline the tests use as a return address) ends the run. */
    if (target == VP_HOST_EXIT_ADDRESS) {
        c->rip = target;
        if (vp_exit_armed) { if (vp_run_cpu && vp_run_cpu != c) *vp_run_cpu = *c; longjmp(*vp_exit_jump, 1); }
        return;
    }
    c->fault_rip = target;
    c->fault_what = "no translation for this address";
    if (vp_exit_armed) { if (vp_run_cpu && vp_run_cpu != c) *vp_run_cpu = *c; longjmp(*vp_exit_jump, 2); }
    fprintf(stderr, "vp: no translation for %#llx\n", (unsigned long long)target);
    abort();
}

void vp_unsupported(VpCpu* c, uint64_t rip, const char* what) {
    c->fault_rip = rip;
    c->fault_what = what;
    if (vp_exit_armed) { if (vp_run_cpu && vp_run_cpu != c) *vp_run_cpu = *c; longjmp(*vp_exit_jump, 3); }
    fprintf(stderr, "vp: unsupported instruction %s at %#llx\n", what, (unsigned long long)rip);
    abort();
}

void vp_divide_error(VpCpu* c, uint64_t rip) { vp_unsupported(c, rip, "#DE"); }

void vp_cpuid(VpCpu* c) {
    /* A fixed, conservative identity: SSE4.2 and below, no AVX. The game runtimes refine it. */
    const uint32_t leaf = (uint32_t)c->r[VP_RAX];
    uint32_t a = 0, b = 0, cc = 0, d = 0;
    if (leaf == 0) { a = 7; b = 0x756e6547; d = 0x49656e69; cc = 0x6c65746e; }
    else if (leaf == 1) { a = 0x000306a9; b = 0x00100800; cc = 0x00c00000 | 0x00180201 | 0x00000200; d = 0x178bfbff; }
    c->r[VP_RAX] = a; c->r[VP_RBX] = b; c->r[VP_RCX] = cc; c->r[VP_RDX] = d;
}

__attribute__((weak)) void vp_syscall(VpCpu* c) {
    vp_unsupported(c, c->rip, "syscall");
}

uint64_t vp_rdtsc(VpCpu* c) {
    (void)c;
    static uint64_t t;
    return t += 1000;
}

void vp_apply_mxcsr(VpCpu* c) {
    switch ((c->mxcsr & VP_MXCSR_RC_MASK) >> 13) {
    case 0: fesetround(FE_TONEAREST); break;
    case 1: fesetround(FE_DOWNWARD); break;
    case 2: fesetround(FE_UPWARD); break;
    default: fesetround(FE_TOWARDZERO); break;
    }
}

__attribute__((weak)) void vp_trace(VpCpu* cpu, uint64_t rip) { (void)cpu; (void)rip; }

/* Runs translated code from `entry` until it returns to VP_HOST_EXIT_ADDRESS (pushed by the
 * caller as the return address) or faults. Returns 0 on a clean exit, else the fault kind. */
static int vp_known(uint64_t target) {
    if (vp_find(target)) return 1;
    if (vp_find_native(target)) return 1;
    const uint64_t key = vp_tables_relative ? target - vp_image_base : target;
    for (size_t i = 0; i < vp_extra_entry_count; ++i) if (vp_extra_entries[i].guest == key) return 1;
    return 0;
}

int vp_run(VpCpu* c, uint64_t entry) {
    jmp_buf here;
    jmp_buf* const outer = vp_exit_jump;
    VpCpu* const outer_cpu = vp_run_cpu;
    volatile int r;
    if (!vp_known(entry)) {
        c->fault_rip = entry;
        c->fault_what = "entry not translated";
        return 2;
    }
    vp_exit_jump = &here;
    vp_run_cpu = c;
    r = setjmp(here);
    if (r == 0) {
        vp_dispatch(c, entry);
        /* Returned normally: the function's `ret` left rip = the return address. */
        r = (c->rip == VP_HOST_EXIT_ADDRESS) ? 0 : 2;
        if (r) { c->fault_rip = c->rip; c->fault_what = "returned to an untranslated address"; }
    } else if (r == 1) {
        r = 0;
    }
    vp_exit_jump = outer;
    vp_run_cpu = outer_cpu;
    return r;
}

void vp_run_reset(void) {
    vp_exit_jump = NULL;
    vp_run_cpu = NULL;
}

uint64_t vp_call_guest(VpCpu* c, uint64_t fn) {
    /* The caller set the arguments and the stack (its return address slot is pushed here). */
    uint64_t saved[16];
    const uint64_t saved_rip = c->rip;
    memcpy(saved, c->r, sizeof saved);
    c->r[VP_RSP] -= 8;
    vp_st64(c->r[VP_RSP], VP_HOST_EXIT_ADDRESS);
    const int fault = vp_run(c, fn);
    const uint64_t result = c->r[VP_RAX];
    if (fault) {
        fprintf(stderr, "vp: guest callback %#llx failed: %s at %#llx\n", (unsigned long long)fn,
                c->fault_what ? c->fault_what : "?", (unsigned long long)c->fault_rip);
        abort();
    }
    /* Callee-saved registers come back by the ABI; the rest are restored so that the native that
     * made the call sees its own state again. */
    memcpy(c->r, saved, sizeof saved);
    c->r[VP_RAX] = result;
    c->rip = saved_rip;
    return result;
}
