/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * Two modules loaded one after the other, each importing a different native function: the stub
 * addresses the loader hands out must differ, so that each call reaches its own native.
 *   loader_stubs DIR
 */
#include "vp_loader.h"
#include "elfgen.h"
#include <stdio.h>
#include <stdlib.h>

#define BASE_A UINT64_C(0x500000000)
#define BASE_B UINT64_C(0x510000000)

static void native_a(VpCpu* c) { c->r[VP_RAX] = 0xa; c->rip = 0; }
static void native_b(VpCpu* c) { c->r[VP_RAX] = 0xb; c->rip = 0; }

static int resolve(const char* name, int function, VpNative* native, uint64_t* data, void* user) {
    (void)function; (void)data; (void)user;
    if (!strcmp(name, "fa")) { *native = native_a; return 1; }
    if (!strcmp(name, "fb")) { *native = native_b; return 1; }
    return 0;
}

static int load(const char* dir, const char* name, const char* symbol, uint64_t base, uint64_t* stub) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s.elf", dir, name);
    const TestRela r = {base + 0x600, ((uint64_t)1 << 32) | 6, 0}; /* GLOB_DAT, symbol 1 */
    if (elfgen(path, base, &r, 1, sizeof r, NULL, 0)) return 2;
    static VpImport imports[2];
    static VpModule modules[2];
    static int used;
    VpImport* im = &imports[used];
    *im = (VpImport){symbol, base + 0x600, 1};
    VpModule* m = &modules[used++];
    m->name = name;
    m->link_base = base;
    m->size = TEST_IMG;
    m->imports = im;
    m->import_count = 1;
    m->fingerprint = vp_fingerprint(0, NULL, 0, NULL, 0);
    vp_register_module(m);
    VpLoadedImage out;
    if (vp_load_module(path, m, 0, resolve, NULL, &out)) { printf("load %s: %s\n", name, out.error); return 2; }
    memcpy(stub, (void*)(uintptr_t)(base + 0x600), 8);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    uint64_t a = 0, b = 0;
    if (load(argv[1], "ma", "fa", BASE_A, &a) || load(argv[1], "mb", "fb", BASE_B, &b)) return 2;
    VpCpu c;
    memset(&c, 0, sizeof c);
    vp_call_native(&c, a);
    const uint64_t ra = c.r[VP_RAX];
    vp_call_native(&c, b);
    const uint64_t rb = c.r[VP_RAX];
    printf("stubs %#llx %#llx, calls %#llx %#llx\n", (unsigned long long)a, (unsigned long long)b, (unsigned long long)ra, (unsigned long long)rb);
    if (a == b || ra != 0xa || rb != 0xb) { printf("FAIL: the second module's import replaced the first's\n"); return 1; }
    printf("ok   distinct stubs\n");
    return 0;
}
