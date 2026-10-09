/* Two separately translated modules in one process, each placed by the loader at an address of
 * its own; the main module's import of lib_mix is bound to the library's guest code; the library's
 * import of vp_ext_counter to a native. Then the same images are attached by fingerprint, as a
 * runtime that loads images itself would do. */
#include "vp_loader.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef unsigned long u64;
static u64 counter;
static uint64_t lib_mix_address;
static void ext_counter(VpCpu* c) { c->r[VP_RAX] = ++counter; c->rip = vp_pop64(c); }
static int resolve(const char* name, int function, VpNative* native, uint64_t* data, void* user) {
    (void)user; (void)function;
    if (!strcmp(name, "vp_ext_counter")) { *native = ext_counter; return 1; }
    if (!strcmp(name, "lib_mix") && lib_mix_address) { *data = lib_mix_address; return 1; }
    return 0;
}
u64 vp_ext_counter(void) { return ++counter; }
u64 vp_main_native(u64* mem);
int main(int argc, char** argv) {
    VpModule* lib = vp_module_by_name("lib");
    VpModule* mainm = vp_module_by_name("main");
    if (!lib || !mainm) { fprintf(stderr, "modules not registered\n"); return 1; }
    VpLoadedImage li, mi;
    if (vp_load_module(argv[1], lib, 0x260000000ull, resolve, NULL, &li)) { fprintf(stderr, "lib: %s\n", li.error); return 1; }
    lib_mix_address = li.base + strtoull(argv[3], NULL, 0) - lib->link_base;
    if (vp_load_module(argv[2], mainm, 0x280000000ull, resolve, NULL, &mi)) { fprintf(stderr, "main: %s\n", mi.error); return 1; }
    if (vp_map_fixed(0x2f0000000ull, 0x100000) || vp_map_fixed(0x300000000ull, 1 << 20)) return 2;
    VpCpu c; memset(&c, 0, sizeof c); c.mxcsr = 0x1f80;
    c.r[VP_RSP] = 0x2f0000000ull + 0x100000 - 0x100 - 8; vp_st64(c.r[VP_RSP], VP_HOST_EXIT_ADDRESS);
    c.r[VP_RDI] = 0x300000000ull;
    counter = 0;
    if (vp_run(&c, mi.entry)) { fprintf(stderr, "translated: %s at %#" PRIx64 "\n", c.fault_what, c.fault_rip); return 1; }
    u64 mem2[8] = {0};
    counter = 0;
    const u64 expected = vp_main_native(mem2);
    printf("two modules: translated %016" PRIx64 " native %016" PRIx64 " %s\n", c.r[VP_RAX], expected, c.r[VP_RAX] == expected ? "OK" : "MISMATCH");
    /* Attach by fingerprint: forget the bases, find the modules again from memory. */
    vp_module_set_base(lib, 0); vp_module_set_base(mainm, 0);
    VpModule* a = vp_attach_module(li.base, li.end - li.base);
    VpModule* b = vp_attach_module(mi.base, mi.end - mi.base);
    printf("attach by fingerprint: %s, %s\n", a == lib ? "lib OK" : "lib FAILED", b == mainm ? "main OK" : "main FAILED");
    /* A modified image must not attach. */
    unsigned char* code = (unsigned char*)(uintptr_t)(mi.entry);
    code[0] ^= 0xff;
    VpModule* bad = vp_attach_module(mi.base, mi.end - mi.base);
    printf("modified image: %s\n", bad ? "ATTACHED (bug)" : "rejected OK");
    return (c.r[VP_RAX] == expected && a == lib && b == mainm && !bad) ? 0 : 1;
}
