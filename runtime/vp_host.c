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

/* Bumped whenever a module is registered, moved or detached: the per-thread dispatch caches below
 * drop every entry made before. */
static unsigned vp_dispatch_gen = 1;
static void vp_dispatch_changed(void) { __atomic_add_fetch(&vp_dispatch_gen, 1, __ATOMIC_RELEASE); }

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
    vp_dispatch_changed();
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
    vp_dispatch_changed();
    pthread_mutex_unlock(&vp_modules_lock);
}

void vp_detach_module(VpModule* m) {
    pthread_mutex_lock(&vp_modules_lock);
    __atomic_store_n(&m->attached, 0, __ATOMIC_RELEASE);
    vp_dispatch_changed();
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
    /* Sorted by address (landing pads and every call's return address: many). */
    const uint64_t key = m->relative ? guest - __atomic_load_n(&m->base, __ATOMIC_RELAXED) : guest;
    size_t lo = 0, hi = m->extra_count;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (m->extra[mid].guest == key) return &m->extra[mid];
        if (m->extra[mid].guest < key) lo = mid + 1; else hi = mid;
    }
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
        vp_dispatch_changed();
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

/* The embedder's handlers (vp_set_embedder_hooks). Set once, before the game's threads start. */
static VpEmbedderHooks vp_hooks;

void vp_set_embedder_hooks(const VpEmbedderHooks* hooks) {
    VpEmbedderHooks h = {0};
    if (hooks) h = *hooks;
    vp_hooks = h;
    vp_dispatch_changed(); /* cached embedder targets were the old handler's */
}

int vp_runtime_abi(void) { return VP_RUNTIME_ABI; }

__attribute__((weak)) int vp_dispatch_miss(VpCpu* cpu, uint64_t target) {
    return vp_hooks.dispatch_miss ? vp_hooks.dispatch_miss(cpu, target) : 0;
}
/* Whether vp_dispatch_miss may know `target` (an embedder overrides it with vp_dispatch_miss). */
__attribute__((weak)) int vp_dispatch_miss_possible(uint64_t target) {
    return vp_hooks.dispatch_miss_possible ? vp_hooks.dispatch_miss_possible(target) : 0;
}

#define VP_MAX_NATIVES 262144 /* 4 MB of zero pages: only the used part is ever touched */
static struct { uint64_t guest; VpNative fn; } vp_natives[VP_MAX_NATIVES];
static size_t vp_native_count;
/* Natives are registered while other threads dispatch (a runtime that loads a module while the
 * game runs). Writers take the lock and bracket the rewrite with an odd sequence number; the
 * lookups read the table with relaxed atomic loads and run again if a rewrite overlapped. */
static pthread_mutex_t vp_natives_lock = PTHREAD_MUTEX_INITIALIZER;
static unsigned vp_native_seq;

#define NAT_GUEST(i) __atomic_load_n(&vp_natives[i].guest, __ATOMIC_RELAXED)
#define NAT_FN(i) __atomic_load_n(&vp_natives[i].fn, __ATOMIC_RELAXED)
#define NAT_SET(i, g, f) (__atomic_store_n(&vp_natives[i].guest, (g), __ATOMIC_RELAXED), __atomic_store_n(&vp_natives[i].fn, (f), __ATOMIC_RELAXED))

int vp_register_native(uint64_t guest, VpNative fn) {
    pthread_mutex_lock(&vp_natives_lock);
    size_t lo = 0, hi = vp_native_count;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (NAT_GUEST(mid) < guest) lo = mid + 1; else hi = mid;
    }
    const size_t i = lo;
    if (i < vp_native_count && NAT_GUEST(i) == guest) {
        __atomic_store_n(&vp_natives[i].fn, fn, __ATOMIC_RELAXED);
    } else if (vp_native_count >= VP_MAX_NATIVES) {
        pthread_mutex_unlock(&vp_natives_lock);
        fprintf(stderr, "vp: the table of %d natives is full: %#llx is not registered\n", VP_MAX_NATIVES, (unsigned long long)guest);
        return -1;
    } else {
        __atomic_store_n(&vp_native_seq, vp_native_seq + 1, __ATOMIC_RELAXED);
        __atomic_thread_fence(__ATOMIC_RELEASE);
        for (size_t k = vp_native_count; k > i; --k) NAT_SET(k, NAT_GUEST(k - 1), NAT_FN(k - 1));
        NAT_SET(i, guest, fn);
        __atomic_store_n(&vp_native_count, vp_native_count + 1, __ATOMIC_RELAXED);
        __atomic_store_n(&vp_native_seq, vp_native_seq + 1, __ATOMIC_RELEASE);
    }
    pthread_mutex_unlock(&vp_natives_lock);
    vp_dispatch_changed(); /* cached embedder targets must not shadow a new native */
    return 0;
}

static VpNative vp_find_native(uint64_t guest) {
    for (;;) {
        const unsigned before = __atomic_load_n(&vp_native_seq, __ATOMIC_ACQUIRE);
        if (before & 1) continue; /* a registration is rewriting the table */
        VpNative found = NULL;
        size_t lo = 0, hi = __atomic_load_n(&vp_native_count, __ATOMIC_RELAXED);
        while (lo < hi) {
            const size_t mid = (lo + hi) / 2;
            const uint64_t g = NAT_GUEST(mid);
            if (g == guest) { found = NAT_FN(mid); break; }
            if (g < guest) lo = mid + 1; else hi = mid;
        }
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        if (__atomic_load_n(&vp_native_seq, __ATOMIC_RELAXED) == before) return found;
    }
}

void vp_call_native(VpCpu* c, uint64_t guest) {
    VpNative fn = vp_find_native(guest);
    if (!fn) { vp_unsupported(c, guest, "native not registered"); return; }
    fn(c);
}

/* Missing entry points: an address inside a translated module that the translation has no entry
 * for (code reached only through a pointer the static analysis did not find). Each one is logged
 * once as "module+0xOFFSET" to the missing-entries file, which `vpaot --roots` reads back. */
static pthread_mutex_t vp_missing_lock = PTHREAD_MUTEX_INITIALIZER;
static char vp_missing_path[1024];
static uint64_t vp_missing_seen[4096];
static size_t vp_missing_count;

void vp_set_missing_log(const char* path) {
    pthread_mutex_lock(&vp_missing_lock);
    snprintf(vp_missing_path, sizeof vp_missing_path, "%s", path ? path : "");
    pthread_mutex_unlock(&vp_missing_lock);
}

static void vp_note_missing(uint64_t target) {
    const VpModule* m = NULL;
    for (VpModule* x = vp_modules_head(); x; x = x->next) {
        if (!__atomic_load_n(&x->attached, __ATOMIC_ACQUIRE)) continue;
        const uint64_t base = __atomic_load_n(&x->base, __ATOMIC_RELAXED);
        if (target >= base && target - base < x->size) { m = x; break; }
    }
    if (!m) return;
    const uint64_t off = target - __atomic_load_n(&m->base, __ATOMIC_RELAXED) + (m->relative ? 0 : m->base - m->link_base);
    pthread_mutex_lock(&vp_missing_lock);
    for (size_t i = 0; i < vp_missing_count; ++i) {
        if (vp_missing_seen[i] == target) { pthread_mutex_unlock(&vp_missing_lock); return; }
    }
    if (vp_missing_count >= sizeof vp_missing_seen / sizeof vp_missing_seen[0]) { /* full: no more lines, or the file grows without limit */
        pthread_mutex_unlock(&vp_missing_lock);
        return;
    }
    vp_missing_seen[vp_missing_count++] = target;
    const char* path = vp_missing_path[0] ? vp_missing_path : getenv("VPKENGINE_MISSING_LOG");
    FILE* f = path && *path ? fopen(path, "a+") : NULL;
    char entry[256];
    snprintf(entry, sizeof entry, "%s+%#llx\n", m->name, (unsigned long long)off);
    /* The file outlives the process: an earlier session (or the app, after translating the entry
     * on demand) may have written it already. */
    int known = 0;
    if (f) {
        char line[512];
        rewind(f);
        while (!known && fgets(line, sizeof line, f)) known = !strcmp(line, entry);
        fseek(f, 0, SEEK_END); /* between reading and writing */
    }
    if (!known) {
        if (f) fputs(entry, f);
        else fprintf(stderr, "VPKENGINE: missing entry %s", entry);
    }
    if (f) fclose(f);
    pthread_mutex_unlock(&vp_missing_lock);
}

/* Per-thread cache of resolved dispatch targets (indirect calls and jumps, returns to unknown
 * addresses): direct-mapped, tagged with the generation it was filled in. */
#define VP_DISPATCH_CACHE 1024
static _Thread_local struct { uint64_t target; VpFunction fn; uint32_t entry; unsigned gen; } vp_dcache[VP_DISPATCH_CACHE];

/* Set by vp_run right before it dispatches: that dispatch may enter mid-function directly. */
static _Thread_local int vp_top_dispatch;

void vp_dispatch(VpCpu* c, uint64_t target) {
    const int top = vp_top_dispatch;
    vp_top_dispatch = 0;
    const unsigned gen = __atomic_load_n(&vp_dispatch_gen, __ATOMIC_ACQUIRE);
    const size_t slot = (size_t)((target >> 2) ^ (target >> 12)) & (VP_DISPATCH_CACHE - 1);
    int miss_tried = 0;
    if (vp_dcache[slot].target == target && vp_dcache[slot].gen == gen) {
        if (vp_dcache[slot].fn) { vp_dcache[slot].fn(c, vp_dcache[slot].entry); return; }
        if (vp_dispatch_miss(c, target)) return; /* the embedder's (an HLE veneer): no module scan */
        miss_tried = 1; /* not again below: the handler may already have run part of a stub */
    }
    for (VpModule* m = vp_modules_head(); m; m = m->next) {
        if (!__atomic_load_n(&m->attached, __ATOMIC_ACQUIRE)) continue;
        const uint64_t base = __atomic_load_n(&m->base, __ATOMIC_RELAXED);
        if (target < base || target - base >= m->size) continue;
        const VpEntry* e = vp_find_in(m, target);
        if (e) {
            vp_dcache[slot].target = target; vp_dcache[slot].fn = e->function; vp_dcache[slot].entry = 0; vp_dcache[slot].gen = gen;
            e->function(c, 0);
            return;
        }
        const VpExtraEntry* x = vp_find_extra_in(m, target);
        if (x) {
            /* Mid-function (a landing pad or a resume point): reached by a jump that never comes
             * back (an unwinder, longjmp, a fiber switched by `jmp`). Entering it on top of the
             * current host frames would grow the host stack on every switch: drop them first and
             * let vp_run enter it. The state is all in `c` (VP_OUT ran before the dispatch). */
            if (!top && vp_exit_armed) {
                c->rip = target;
                if (vp_run_cpu && vp_run_cpu != c) *vp_run_cpu = *c;
                longjmp(*vp_exit_jump, 4);
            }
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
    if (!miss_tried && vp_dispatch_miss(c, target)) {
        /* Remembered only if no module changed meanwhile (the handler may have attached one). */
        if (__atomic_load_n(&vp_dispatch_gen, __ATOMIC_ACQUIRE) == gen) {
            vp_dcache[slot].target = target; vp_dcache[slot].fn = NULL; vp_dcache[slot].entry = 0; vp_dcache[slot].gen = gen;
        }
        return;
    }
    vp_note_missing(target);
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

/* Locked operations on a misaligned address (x86 allows them; ARM atomics do not): serialized by
 * one lock. They are atomic against each other, not against aligned atomics on overlapping bytes,
 * which x86 code does not mix in practice (split locks are slow on x86 too). */
static pthread_mutex_t vp_split_lock_mutex = PTHREAD_MUTEX_INITIALIZER;
int vp_cas_split(uint64_t a, void* expected, const void* desired, unsigned bytes) {
    __atomic_thread_fence(__ATOMIC_SEQ_CST); /* a locked op is a full barrier on both sides */
    pthread_mutex_lock(&vp_split_lock_mutex);
    int ok = memcmp((const void*)(uintptr_t)a, expected, bytes) == 0;
    if (ok) memcpy((void*)(uintptr_t)a, desired, bytes);
    else memcpy(expected, (const void*)(uintptr_t)a, bytes);
    pthread_mutex_unlock(&vp_split_lock_mutex);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    return ok;
}

void vp_cpuid(VpCpu* c) {
    /* The PS4's CPU (AMD Jaguar, family 16h model 30h), restricted to what the translator runs:
     * SSE3/SSSE3/SSE4.1/SSE4.2, PCLMULQDQ, AES, CX16, MOVBE, POPCNT, XSAVE+OSXSAVE, AVX, F16C; BMI1 in leaf 7. Code
     * that checks before using AVX takes the same paths as on the console. */
    const uint32_t leaf = (uint32_t)c->r[VP_RAX], sub = (uint32_t)c->r[VP_RCX];
    uint32_t a = 0, b = 0, cc = 0, d = 0;
    if (leaf == 0) { a = 0xd; b = 0x68747541; d = 0x69746e65; cc = 0x444d4163; } /* AuthenticAMD */
    else if (leaf == 1) {
        a = 0x00730f01;
        b = 0x00080800;
        cc = (1u << 0) | (1u << 1) | (1u << 9) | (1u << 13) | (1u << 19) | (1u << 20) | (1u << 22) | (1u << 23) |
             (1u << 25) | (1u << 26) | (1u << 27) | (1u << 28) | (1u << 29); /* + PCLMULQDQ, AES */
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
    if (vp_hooks.syscall) { vp_hooks.syscall(c); return; }
    vp_unsupported(c, c->rip, "syscall");
}

uint64_t vp_rdtsc(VpCpu* c) {
    (void)c;
    static uint64_t t;
    return __atomic_add_fetch(&t, 1000, __ATOMIC_RELAXED);
}

void vp_apply_mxcsr(VpCpu* c) {
    switch ((c->mxcsr & VP_MXCSR_RC_MASK) >> 13) {
    case 0: fesetround(FE_TONEAREST); break;
    case 1: fesetround(FE_DOWNWARD); break;
    case 2: fesetround(FE_UPWARD); break;
    default: fesetround(FE_TOWARDZERO); break;
    }
    /* DAZ (bit 6) and FTZ (bit 15) are applied to the host's own state, so the translated float code
     * (which compiles to the hardware instructions) flushes denormals as the guest asked. */
#if defined(__x86_64__)
    __builtin_ia32_ldmxcsr((__builtin_ia32_stmxcsr() & ~0x8040u) | (c->mxcsr & 0x8040u));
#elif defined(__aarch64__)
    /* FPCR.FZ flushes inputs and results together: there is no separate DAZ / FTZ, so either bit sets it. */
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    fpcr = (fpcr & ~(UINT64_C(1) << 24)) | ((c->mxcsr & 0x8040u) ? UINT64_C(1) << 24 : 0);
    __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
#endif
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
    if (r == 0 || r == 4) {
        if (r == 0) { vp_top_dispatch = 1; vp_dispatch(c, entry); }
        /* Returned: the last `ret` left rip = the return address. It is the exit, or (after a
         * context switch: fibers, coroutines, a longjmp to another stack) a resume point inside a
         * function whose host frames are gone: continue there. */
        while (!vp_is_exit(c->rip)) {
            if (!vp_known(c->rip) && !vp_dispatch_miss_possible(c->rip)) {
                c->fault_rip = c->rip;
                c->fault_what = "returned to an untranslated address";
                vp_note_missing(c->rip);
                break;
            }
            vp_top_dispatch = 1;
            vp_dispatch(c, c->rip);
        }
        r = vp_is_exit(c->rip) ? 0 : 2;
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

/* The x87 FPU (and SoftFloat behind it) in this same unit: no build needs another file. */
#include "vp_x87.c"
