/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * Differential test harness. The same x86-64 snippet (assembled from tests/aot/cases, placed at
 * VP_CODE_BASE) is run twice from the same initial state: natively on an x86-64 host, and as the
 * C that vpaot produced. The final registers, flags and the scratch memory must match.
 *
 * On a non-x86 host (the arm64 CI job) only the translated run happens and its result is compared
 * with the golden state recorded by the x86 job (tests/aot/golden/<case>.txt).
 */
#include "vp_emit.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define VP_CODE_BASE  UINT64_C(0x400000)
#define VP_CODE_SIZE  UINT64_C(0x10000)
#define VP_STACK_BASE UINT64_C(0x7f000000)
#define VP_STACK_SIZE UINT64_C(0x10000)
#define VP_DATA_BASE  UINT64_C(0x600000)
#define VP_DATA_SIZE  UINT64_C(0x10000)

#if defined(__x86_64__)
extern void vp_native_run(VpCpu* cpu, uint64_t code);
extern uint64_t vp_native_exit_address(void);
#endif

static void* map_fixed(uint64_t at, uint64_t size, int prot) {
    void* p = mmap((void*)(uintptr_t)at, size, prot, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (p == MAP_FAILED || (uintptr_t)p != at) {
        fprintf(stderr, "cannot map %#" PRIx64 "\n", at);
        exit(3);
    }
    return p;
}

/* A deterministic, non-trivial initial state: every register and the scratch memory are filled
 * from a small generator, so that a wrong read shows up somewhere. */
static void init_state(VpCpu* c) {
    uint64_t x = 0x9e3779b97f4a7c15ull;
    memset(c, 0, sizeof *c);
    for (int i = 0; i < 16; ++i) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        c->r[i] = x;
        c->xmm[i].u64[0] = x * 0x2545f4914f6cdd1dull;
        c->xmm[i].u64[1] = x ^ 0x5555555555555555ull;
        c->ymmh[i].u64[0] = x * 0x9e3779b97f4a7c15ull;
        c->ymmh[i].u64[1] = ~x;
    }
    c->r[VP_RSP] = VP_STACK_BASE + VP_STACK_SIZE - 0x100;
    c->r[VP_RBP] = c->r[VP_RSP];
    /* Pointers the snippets use: rdi = scratch data, rsi = scratch data + 0x800. */
    c->r[VP_RDI] = VP_DATA_BASE;
    c->r[VP_RSI] = VP_DATA_BASE + 0x800;
    c->mxcsr = 0x1f80;
    c->fcw = 0x037f; /* FNINIT state: empty stack */
    uint8_t* d = (uint8_t*)(uintptr_t)VP_DATA_BASE;
    for (uint64_t i = 0; i < VP_DATA_SIZE; ++i) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        d[i] = (uint8_t)x;
    }
    memset((void*)(uintptr_t)VP_STACK_BASE, 0, VP_STACK_SIZE);
}

/* VP_DUMP=<prefix>: the data region of each run goes to <prefix>.translated / <prefix>.native,
 * to find which bytes differ. */
static void dump_data(const char* which) {
    const char* prefix = getenv("VP_DUMP");
    if (!prefix) return;
    char path[512];
    snprintf(path, sizeof path, "%s.%s", prefix, which);
    FILE* f = fopen(path, "wb");
    if (f) { fwrite((const void*)(uintptr_t)VP_DATA_BASE, 1, VP_DATA_SIZE, f); fclose(f); }
}

static uint64_t hash_data(void) {
    /* The return-address slot differs between the two runs by design. */
    vp_st64(VP_STACK_BASE + VP_STACK_SIZE - 0x100 - 8, 0);
    const uint8_t* d = (const uint8_t*)(uintptr_t)VP_DATA_BASE;
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < VP_DATA_SIZE; ++i) h = (h ^ d[i]) * 1099511628211ull;
    const uint8_t* s = (const uint8_t*)(uintptr_t)VP_STACK_BASE;
    for (uint64_t i = 0; i < VP_STACK_SIZE; ++i) h = (h ^ s[i]) * 1099511628211ull;
    return h;
}

static void print_state(FILE* f, const VpCpu* c, uint64_t data_hash) {
    static const char* names[16] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
                                    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"};
    for (int i = 0; i < 16; ++i) fprintf(f, "%s=%016" PRIx64 "\n", names[i], c->r[i]);
    fprintf(f, "flags=cf%d pf%d zf%d sf%d of%d df%d\n", c->cf, c->pf, c->zf, c->sf, c->of, c->df);
    for (int i = 0; i < 16; ++i) fprintf(f, "xmm%d=%016" PRIx64 "%016" PRIx64 "\n", i, c->xmm[i].u64[1], c->xmm[i].u64[0]);
    for (int i = 0; i < 16; ++i) fprintf(f, "ymmh%d=%016" PRIx64 "%016" PRIx64 "\n", i, c->ymmh[i].u64[1], c->ymmh[i].u64[0]);
    fprintf(f, "x87 fcw=%04x fsw=%04x tag=%02x\n", c->fcw, vp_x87_fnstsw((VpCpu*)c), c->ftag);
    for (int i = 0; i < 8; ++i) fprintf(f, "r%d=%04x%016" PRIx64 "\n", i, c->st[i].se, c->st[i].m);
    fprintf(f, "memory=%016" PRIx64 "\n", data_hash);
}

static int states_equal(const VpCpu* a, const VpCpu* b, int compare_af) {
    (void)compare_af; /* AF is undefined after most instructions; never compared. */
    if (memcmp(a->r, b->r, sizeof a->r)) return 0;
    if (a->cf != b->cf || a->pf != b->pf || a->zf != b->zf || a->sf != b->sf || a->of != b->of || a->df != b->df) return 0;
    if (memcmp(a->xmm, b->xmm, sizeof a->xmm)) return 0;
    if (memcmp(a->ymmh, b->ymmh, sizeof a->ymmh)) return 0;
    if (a->fcw != b->fcw || a->fsw != b->fsw || a->ftop != b->ftop || a->ftag != b->ftag) return 0;
    for (int i = 0; i < 8; ++i) if (a->st[i].m != b->st[i].m || a->st[i].se != b->st[i].se) return 0;
    return 1;
}

/* VP_NATIVE=<guest address>: the test image has a stub there (real x86 code for the native run)
 * that the translated run replaces with this host function: rax = rdi * 2 + 1. */
static void native_double_plus_one(VpCpu* c) {
    c->r[VP_RAX] = c->r[VP_RDI] * 2 + 1;
    c->rip = vp_pop64(c);
}

int main(int argc, char** argv) {
    { VpModule* m = vp_first_module(); if (m && !m->attached) vp_module_set_base(m, m->link_base); } /* a --pic translation runs at its link base here */
    if (getenv("VP_NATIVE")) vp_register_native(strtoull(getenv("VP_NATIVE"), NULL, 0), native_double_plus_one);
    if (argc < 2) {
        fprintf(stderr, "usage: harness CODE.bin [--golden FILE | --write-golden FILE]\n");
        return 2;
    }
    map_fixed(VP_CODE_BASE, VP_CODE_SIZE, PROT_READ | PROT_WRITE | PROT_EXEC);
    map_fixed(VP_STACK_BASE, VP_STACK_SIZE, PROT_READ | PROT_WRITE);
    map_fixed(VP_DATA_BASE, VP_DATA_SIZE, PROT_READ | PROT_WRITE);
    FILE* cf = fopen(argv[1], "rb");
    if (!cf) { perror(argv[1]); return 2; }
    size_t code_size = fread((void*)(uintptr_t)VP_CODE_BASE, 1, VP_CODE_SIZE, cf);
    fclose(cf);
    if (!code_size) { fprintf(stderr, "empty code\n"); return 2; }

    VpCpu translated, native;
    uint64_t translated_hash = 0, native_hash = 0;

    /* Translated run. */
    init_state(&translated);
    translated.r[VP_RSP] -= 8;
    vp_st64(translated.r[VP_RSP], VP_HOST_EXIT_ADDRESS);
    const int fault = vp_run(&translated, VP_CODE_BASE);
    if (fault) {
        fprintf(stderr, "translated: %s at %#" PRIx64 "\n", translated.fault_what, translated.fault_rip);
        return 1;
    }
    translated_hash = hash_data();
    dump_data("translated");

    const char* golden = (argc >= 4 && !strcmp(argv[2], "--golden")) ? argv[3] : NULL;
    const char* write_golden = (argc >= 4 && !strcmp(argv[2], "--write-golden")) ? argv[3] : NULL;

#if defined(__x86_64__)
    /* Native run, from the same state and memory; the return address is the native exit stub. */
    init_state(&native);
    native.r[VP_RSP] -= 8;
    vp_st64(native.r[VP_RSP], vp_native_exit_address());
    vp_native_run(&native, VP_CODE_BASE);
    native_hash = hash_data();
    dump_data("native");
    if (!states_equal(&translated, &native, 0) || translated_hash != native_hash) {
        fprintf(stderr, "MISMATCH native vs translated\n--- native\n");
        print_state(stderr, &native, native_hash);
        fprintf(stderr, "--- translated\n");
        print_state(stderr, &translated, translated_hash);
        return 1;
    }
    if (write_golden) {
        FILE* g = fopen(write_golden, "w");
        if (!g) { perror(write_golden); return 2; }
        print_state(g, &native, native_hash);
        fclose(g);
    }
#else
    (void)native; (void)native_hash; (void)write_golden;
#endif
    if (golden) {
        FILE* g = fopen(golden, "r");
        if (!g) { perror(golden); return 2; }
        char expected[8192], actual[8192];
        const size_t n = fread(expected, 1, sizeof expected - 1, g);
        expected[n] = 0;
        fclose(g);
        FILE* m = fmemopen(actual, sizeof actual, "w");
        print_state(m, &translated, translated_hash);
        fclose(m);
        if (strcmp(expected, actual)) {
            fprintf(stderr, "MISMATCH golden vs translated\n--- golden\n%s--- translated\n%s", expected, actual);
            return 1;
        }
    }
    print_state(stdout, &translated, translated_hash);
    return 0;
}
