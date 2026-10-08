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

static jmp_buf vp_exit_jump;
static int vp_exit_armed;
static VpCpu* vp_run_cpu; /* the state vp_run was given: faults are reported there */

static const VpEntry* vp_find(uint64_t guest) {
    size_t lo = 0, hi = vp_entry_count;
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

void vp_register_native(uint64_t guest, VpNative fn) {
    if (vp_native_count < VP_MAX_NATIVES) {
        vp_natives[vp_native_count].guest = guest;
        vp_natives[vp_native_count].fn = fn;
        vp_native_count++;
    }
}

static VpNative vp_find_native(uint64_t guest) {
    for (size_t i = 0; i < vp_native_count; ++i) if (vp_natives[i].guest == guest) return vp_natives[i].fn;
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
    /* vp_host_exit (the trampoline the tests use as a return address) ends the run. */
    if (target == VP_HOST_EXIT_ADDRESS) {
        c->rip = target;
        if (vp_exit_armed) { if (vp_run_cpu && vp_run_cpu != c) *vp_run_cpu = *c; longjmp(vp_exit_jump, 1); }
        return;
    }
    c->fault_rip = target;
    c->fault_what = "no translation for this address";
    if (vp_exit_armed) { if (vp_run_cpu && vp_run_cpu != c) *vp_run_cpu = *c; longjmp(vp_exit_jump, 2); }
    fprintf(stderr, "vp: no translation for %#llx\n", (unsigned long long)target);
    abort();
}

void vp_unsupported(VpCpu* c, uint64_t rip, const char* what) {
    c->fault_rip = rip;
    c->fault_what = what;
    if (vp_exit_armed) { if (vp_run_cpu && vp_run_cpu != c) *vp_run_cpu = *c; longjmp(vp_exit_jump, 3); }
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
int vp_run(VpCpu* c, uint64_t entry) {
    const VpEntry* e = vp_find(entry);
    int r;
    if (!e) {
        c->fault_rip = entry;
        c->fault_what = "entry not translated";
        return 2;
    }
    vp_exit_armed = 1;
    vp_run_cpu = c;
    r = setjmp(vp_exit_jump);
    if (r == 0) {
        e->function(c, 0);
        /* Returned normally: the function's `ret` left rip = the return address. */
        r = (c->rip == VP_HOST_EXIT_ADDRESS) ? 0 : 2;
        if (r) { c->fault_rip = c->rip; c->fault_what = "returned to an untranslated address"; }
    } else if (r == 1) {
        r = 0;
    }
    vp_exit_armed = 0;
    return r;
}
