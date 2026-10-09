/* Host for the loader test: provides the two imports and checks the program's result. */
#include "vp_loader.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef unsigned long u64;
static u64 counter;
static void ext_double(VpCpu* c) { c->r[VP_RAX] = c->r[VP_RDI] * 2; c->rip = vp_pop64(c); }
static void ext_counter(VpCpu* c) { c->r[VP_RAX] = ++counter; c->rip = vp_pop64(c); }
static int resolve(const char* name, int function, VpNative* native, uint64_t* data, void* user) {
    (void)user; (void)function; (void)data;
    if (!strcmp(name, "vp_ext_double")) { *native = ext_double; return 1; }
    if (!strcmp(name, "vp_ext_counter")) { *native = ext_counter; return 1; }
    return 0;
}
/* The same program natively, with the same imports. */
u64 vp_ext_double(u64 x) { return x * 2; }
u64 vp_ext_counter(void) { return ++counter; }
u64 vp_main_native(u64* mem);
int main(int argc, char** argv) {
    VpLoadedImage img;
    const char* at = getenv("VP_LOAD_AT");
    if (vp_load_image_at(argv[1], at ? strtoull(at, NULL, 0) : 0, resolve, NULL, &img)) { fprintf(stderr, "load: %s\n", img.error); return 1; }
    printf("image %#" PRIx64 "..%#" PRIx64 ", entry %#" PRIx64 ", imports resolved %zu missing %zu\n", img.base, img.end, img.entry, img.imports_resolved, img.imports_missing);
    if (vp_map_fixed(0x2f0000000ull, 0x100000) || vp_map_fixed(0x300000000ull, 1 << 20)) return 2;
    VpCpu c; memset(&c, 0, sizeof c); c.mxcsr = 0x1f80;
    c.r[VP_RSP] = 0x2f0000000ull + 0x100000 - 0x100 - 8; vp_st64(c.r[VP_RSP], VP_HOST_EXIT_ADDRESS);
    c.r[VP_RDI] = 0x300000000ull;
    counter = 0;
    if (vp_run(&c, img.entry)) { fprintf(stderr, "translated: %s at %#" PRIx64 "\n", c.fault_what, c.fault_rip); return 1; }
    u64 mem2[64] = {0};
    counter = 0;
    const u64 expected = vp_main_native(mem2);
    printf("translated %016" PRIx64 " native %016" PRIx64 " %s\n", c.r[VP_RAX], expected, c.r[VP_RAX] == expected ? "OK" : "MISMATCH");
    return c.r[VP_RAX] == expected ? 0 : 1;
}
