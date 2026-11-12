/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * The same import in several modules: with the same native it gets one stub (the table of natives
 * grows with the distinct imports, not with the modules loaded: a game has dozens of modules that
 * import the same libc), with another native it gets its own, and the same module loaded again
 * reuses its stubs, also when the modules load on several threads at once (build with
 * -fsanitize=thread to see the races). Each stub calls its own native.
 *   loader_shared DIR
 */
#include "vp_loader.h"
#include "elfgen.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

static void native_a(VpCpu* c) { c->r[VP_RAX] = 0xa; c->rip = 0; }
static void native_b(VpCpu* c) { c->r[VP_RAX] = 0xb; c->rip = 0; }

static int resolve(const char* name, int function, VpNative* native, uint64_t* data, void* user) {
    (void)name; (void)function; (void)data;
    *native = user; /* the native this module's loader wants for every import */
    return 1;
}

/* Loads module number `idx` (at its own base) whose one import "shared_symbol" gets `native`; returns its stub. */
static uint64_t load(const char* dir, unsigned idx, uint64_t base, VpNative native) {
    char path[512];
    snprintf(path, sizeof path, "%s/shared%u.elf", dir, idx);
    const TestRela r = {base + 0x600, ((uint64_t)1 << 32) | 6, 0};
    if (elfgen(path, base, &r, 1, sizeof r, NULL, 0)) exit(2);
    static VpImport imports[16];
    static VpModule modules[16];
    static char names[16][16];
    imports[idx] = (VpImport){"shared_symbol", base + 0x600, 1};
    snprintf(names[idx], 16, "shared%u", idx);
    VpModule* m = &modules[idx];
    m->name = names[idx];
    m->link_base = base;
    m->size = TEST_IMG;
    m->imports = &imports[idx];
    m->import_count = 1;
    m->fingerprint = vp_fingerprint(0, NULL, 0, NULL, 0);
    vp_register_module(m);
    VpLoadedImage out;
    if (vp_load_module(path, m, 0, resolve, native, &out)) { printf("load %s: %s\n", names[idx], out.error); exit(2); }
    uint64_t stub;
    memcpy(&stub, (void*)(uintptr_t)(base + 0x600), 8);
    return stub;
}

typedef struct { const char* dir; unsigned idx; uint64_t stub; } Job;
static void* job(void* arg) {
    Job* j = arg;
    j->stub = load(j->dir, j->idx, UINT64_C(0x540000000) + (uint64_t)j->idx * 0x10000000, native_a);
    return NULL;
}

static uint64_t call(uint64_t stub) {
    VpCpu c;
    memset(&c, 0, sizeof c);
    vp_call_native(&c, stub);
    return c.r[VP_RAX];
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    const uint64_t s1 = load(argv[1], 0, UINT64_C(0x500000000), native_a);
    const uint64_t s2 = load(argv[1], 1, UINT64_C(0x510000000), native_a); /* another module, same import, same native */
    const uint64_t s3 = load(argv[1], 2, UINT64_C(0x520000000), native_b); /* same name, another native */
    int bad = 0;
    Job jobs[4];
    pthread_t th[4];
    for (unsigned i = 0; i < 4; ++i) { jobs[i] = (Job){argv[1], 3 + i, 0}; pthread_create(&th[i], NULL, job, &jobs[i]); }
    for (unsigned i = 0; i < 4; ++i) pthread_join(th[i], NULL);
    for (unsigned i = 0; i < 4; ++i) {
        if (jobs[i].stub != s1) { printf("FAIL: module %u loaded on a thread got another stub\n", 3 + i); bad = 1; }
    }
    if (s1 != s2) { printf("FAIL: the same import and native got two stubs (%#llx, %#llx)\n", (unsigned long long)s1, (unsigned long long)s2); bad = 1; }
    if (s3 == s1) { printf("FAIL: another native shares the first one's stub\n"); bad = 1; }
    if (call(s1) != 0xa || call(s2) != 0xa || call(s3) != 0xb) { printf("FAIL: a stub reaches the wrong native\n"); bad = 1; }
    if (!bad) printf("ok   shared stubs\n");
    return bad;
}
