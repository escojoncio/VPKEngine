/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * A minimal host for translated code: enters it at a guest address, resolves indirect calls and
 * jumps through the entry table, and reports what the translation could not do. The game
 * runtimes replace vp_cpuid/vp_rdtsc and the fault handlers with their own.
 */
#include "vp_emit.h"

#include <fenv.h>
#include <pthread.h>
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

/* The registered modules. Writers (registration, attach, detach: before threads start, or when
 * a runtime loads a module while the game runs) take the lock; dispatch reads without it, so
 * the list head and each module's base/attached are published with release stores and read with
 * acquire loads. Modules are never freed. */
static VpModule* vp_modules;
static pthread_mutex_t vp_modules_lock = PTHREAD_MUTEX_INITIALIZER;

static VpModule* vp_modules_head(void) { return __atomic_load_n(&vp_modules, __ATOMIC_ACQUIRE); }

void vp_register_module(VpModule* m) {
    pthread_mutex_lock(&vp_modules_lock);
    for (VpModule* x = vp_modules; x; x = x->next) {
        if (x == m) { pthread_mutex_unlock(&vp_modules_lock); return; }
        if (!strcmp(x->name, m->name)) {
            fprintf(stderr, "vp: two translated modules are named %s (give each its own --module)\n", m->name);
            abort();
        }
    }
    m->next = vp_modules;
    __atomic_store_n(&vp_modules, m, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&vp_modules_lock);
}

VpModule* vp_module_by_name(const char* name) {
    for (VpModule* m = vp_modules_head(); m; m = m->next) if (!strcmp(m->name, name)) return m;
    return NULL;
}

VpModule* vp_module_list(void) { return vp_modules_head(); }

VpModule* vp_first_module(void) {
    /* The list is in reverse registration order: the first registered is the last. */
    VpModule* m = vp_modules_head();
    while (m && m->next) m = m->next;
    return m;
}

VpModule* vp_module_at(uint64_t address) {
    for (VpModule* m = vp_modules_head(); m; m = m->next) {
        if (!__atomic_load_n(&m->attached, __ATOMIC_ACQUIRE)) continue;
        const uint64_t base = __atomic_load_n(&m->base, __ATOMIC_RELAXED);
        if (address >= base && address - base < m->size) return m;
    }
    return NULL;
}

void vp_module_set_base(VpModule* m, uint64_t base) {
    pthread_mutex_lock(&vp_modules_lock);
    __atomic_store_n(&m->attached, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&m->base, base, __ATOMIC_RELAXED);
    __atomic_store_n(&m->attached, 1, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&vp_modules_lock);
}

void vp_detach_module(VpModule* m) {
    pthread_mutex_lock(&vp_modules_lock);
    __atomic_store_n(&m->attached, 0, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&vp_modules_lock);
}

uint64_t vp_fingerprint(uint64_t base, const VpRange* code, size_t code_count, const uint64_t* reloc_sites, size_t reloc_count) {
    uint64_t h = 1469598103934665603ull;
    size_t r = 0;
    for (size_t i = 0; i < code_count; ++i) {
        const unsigned char* p = (const unsigned char*)(uintptr_t)(base + code[i].start);
        for (uint64_t k = 0; k < code[i].size; ++k) {
            const uint64_t off = code[i].start + k;
            while (r < reloc_count && reloc_sites[r] + 8 <= off) ++r;
            const int in_reloc = r < reloc_count && reloc_sites[r] <= off && off < reloc_sites[r] + 8;
            h = (h ^ (in_reloc ? 0 : p[k])) * 1099511628211ull;
        }
    }
    return h;
}

VpModule* vp_attach_module(uint64_t base, uint64_t size) {
    for (VpModule* m = vp_modules_head(); m; m = m->next) {
        if (m->size != size) continue;
        if (base != m->link_base && !m->relative) continue;
        if (vp_fingerprint(base, m->code, m->code_count, m->reloc_sites, m->reloc_site_count) != m->fingerprint) continue;
        vp_module_set_base(m, base);
        return m;
    }
    return NULL;
}

uint64_t vp_module_fingerprint_now(const VpModule* m) {
    return vp_fingerprint(m->base, m->code, m->code_count, m->reloc_sites, m->reloc_site_count);
}

static const VpEntry* vp_find_in(const VpModule* m, uint64_t guest) {
    size_t lo = 0, hi = m->entry_count;
    if (m->relative) guest -= __atomic_load_n(&m->base, __ATOMIC_RELAXED);
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (m->entries[mid].guest == guest) return &m->entries[mid];
        if (m->entries[mid].guest < guest) lo = mid + 1; else hi = mid;
    }
    return NULL;
}

static const VpExtraEntry* vp_find_extra_in(const VpModule* m, uint64_t guest) {
    const uint64_t key = m->relative ? guest - __atomic_load_n(&m->base, __ATOMIC_RELAXED) : guest;
    for (size_t i = 0; i < m->extra_count; ++i) if (m->extra[i].guest == key) return &m->extra[i];
    return NULL;
}

/* Exit ranges: a dispatch into one ends the innermost vp_run (an embedder's return pages). */
#define VP_MAX_EXIT_RANGES 8
static struct { uint64_t start, size; } vp_exit_ranges[VP_MAX_EXIT_RANGES];
static int vp_exit_range_count;

int vp_add_exit_range(uint64_t start, uint64_t size) {
    int ok = 0;
    pthread_mutex_lock(&vp_modules_lock);
    for (int i = 0; i < vp_exit_range_count; ++i) {
        if (vp_exit_ranges[i].start == start && vp_exit_ranges[i].size == size) ok = 1; /* already there */
    }
    if (!ok && vp_exit_range_count < VP_MAX_EXIT_RANGES) {
        vp_exit_ranges[vp_exit_range_count].start = start;
        vp_exit_ranges[vp_exit_range_count].size = size;
        __atomic_store_n(&vp_exit_range_count, vp_exit_range_count + 1, __ATOMIC_RELEASE);
        ok = 1;
    }
    pthread_mutex_unlock(&vp_modules_lock);
    return ok ? 0 : -1;
}

int vp_is_exit(uint64_t target) {
    if (target == VP_HOST_EXIT_ADDRESS) return 1;
    const int n = __atomic_load_n(&vp_exit_range_count, __ATOMIC_ACQUIRE);
    for (int i = 0; i < n; ++i) {
        if (target >= vp_exit_ranges[i].start && target - vp_exit_ranges[i].start < vp_exit_ranges[i].size) return 1;
    }
    return 0;
}

__attribute__((weak)) int vp_dispatch_miss(VpCpu* cpu, uint64_t target) { (void)cpu; (void)target; return 0; }
/* Whether vp_dispatch_miss may know `target` (an embedder overrides it with vp_dispatch_miss). */
__attribute__((weak)) int vp_dispatch_miss_possible(uint64_t target) { (void)target; return 0; }

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
    for (VpModule* m = vp_modules_head(); m; m = m->next) {
        if (!__atomic_load_n(&m->attached, __ATOMIC_ACQUIRE)) continue;
        const uint64_t base = __atomic_load_n(&m->base, __ATOMIC_RELAXED);
        if (target < base || target - base >= m->size) continue;
        const VpEntry* e = vp_find_in(m, target);
        if (e) {
            e->function(c, 0);
            return;
        }
        const VpExtraEntry* x = vp_find_extra_in(m, target);
        if (x) {
            x->function(c, x->entry);
            return;
        }
    }
    if (vp_find_native(target)) {
        vp_call_native(c, target);
        return;
    }
    /* An exit address (the trampoline the tests use, an embedder's return page) ends the run. */
    if (vp_is_exit(target)) {
        c->rip = target;
        if (vp_exit_armed) { if (vp_run_cpu && vp_run_cpu != c) *vp_run_cpu = *c; longjmp(*vp_exit_jump, 1); }
        return;
    }
    if (vp_dispatch_miss(c, target)) return;
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
    /* The PS4's CPU (AMD Jaguar, family 16h model 30h), restricted to what the translator runs:
     * SSE3/SSSE3/SSE4.1/SSE4.2, CX16, MOVBE, POPCNT, XSAVE+OSXSAVE, AVX, F16C; BMI1 in leaf 7. Code
     * that checks before using AVX takes the same paths as on the console. */
    const uint32_t leaf = (uint32_t)c->r[VP_RAX], sub = (uint32_t)c->r[VP_RCX];
    uint32_t a = 0, b = 0, cc = 0, d = 0;
    if (leaf == 0) { a = 0xd; b = 0x68747541; d = 0x69746e65; cc = 0x444d4163; } /* AuthenticAMD */
    else if (leaf == 1) {
        a = 0x00730f01;
        b = 0x00080800;
        cc = (1u << 0) | (1u << 9) | (1u << 13) | (1u << 19) | (1u << 20) | (1u << 22) | (1u << 23) |
             (1u << 26) | (1u << 27) | (1u << 28) | (1u << 29);
        d = 0x178bfbff;
    } else if (leaf == 7 && sub == 0) {
        b = 1u << 3; /* BMI1 */
    } else if (leaf == 0xd && sub == 0) {
        a = 7; b = 0x340; cc = 0x340; /* x87, SSE, AVX state; 832-byte XSAVE area */
    } else if (leaf == 0x80000000) {
        a = 0x8000001e; b = 0x68747541; d = 0x69746e65; cc = 0x444d4163;
    } else if (leaf == 0x80000001) {
        a = 0x00730f01; cc = 1u << 5; d = 0x2fd3fbff; /* ABM (lzcnt) */
    }
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
    for (VpModule* m = vp_modules_head(); m; m = m->next) {
        if (!__atomic_load_n(&m->attached, __ATOMIC_ACQUIRE)) continue;
        const uint64_t base = __atomic_load_n(&m->base, __ATOMIC_RELAXED);
        if (target < base || target - base >= m->size) continue;
        if (vp_find_in(m, target) || vp_find_extra_in(m, target)) return 1;
    }
    if (vp_find_native(target)) return 1;
    return 0;
}

int vp_run(VpCpu* c, uint64_t entry) {
    jmp_buf here;
    jmp_buf* const outer = vp_exit_jump;
    VpCpu* const outer_cpu = vp_run_cpu;
    volatile int r;
    if (!vp_known(entry) && !vp_dispatch_miss_possible(entry)) {
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
        r = vp_is_exit(c->rip) ? 0 : 2;
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
