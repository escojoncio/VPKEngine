/* A game pack on Linux: the runtime in its own shared library (as VPRuntime in the app), the
 * translated modules in another (the pack) that only links against the runtime, loaded with
 * dlopen. The host checks the pack's vp_pack_info, runs the same two modules as tests/aot/modules
 * and checks that the embedder hooks set at run time reach the runtime. */
#include "vp_loader.h"
#include <dlfcn.h>
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

#define HOOKED_ADDRESS UINT64_C(0x5550000)
static int misses;
static int miss(VpCpu* c, uint64_t target) {
    if (target != HOOKED_ADDRESS) return 0;
    ++misses;
    c->r[VP_RAX] = 0x600d;
    c->rip = vp_pop64(c); /* as a `ret` would */
    return 1;
}
static int miss_possible(uint64_t target) { return target == HOOKED_ADDRESS; }

int main(int argc, char** argv) {
    if (vp_module_list()) { fprintf(stderr, "modules registered before the pack was loaded\n"); return 1; }
    void* pack = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!pack) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    const VpPackInfo* info = (const VpPackInfo*)dlsym(pack, "vp_pack_info");
    if (!info || info->magic != VP_PACK_MAGIC || info->abi != (uint32_t)vp_runtime_abi() || info->module_count != 2) {
        fprintf(stderr, "bad vp_pack_info\n");
        return 1;
    }
    printf("pack %s: %zu modules, abi %u\n", info->title, info->module_count, info->abi);
    VpModule* lib = vp_module_by_name("lib");
    VpModule* mainm = vp_module_by_name("main");
    if (!lib || !mainm) { fprintf(stderr, "modules not registered by loading the pack\n"); return 1; }
    VpLoadedImage li, mi;
    if (vp_load_module(argv[2], lib, 0x260000000ull, resolve, NULL, &li)) { fprintf(stderr, "lib: %s\n", li.error); return 1; }
    lib_mix_address = li.base + strtoull(argv[4], NULL, 0) - lib->link_base;
    if (vp_load_module(argv[3], mainm, 0x280000000ull, resolve, NULL, &mi)) { fprintf(stderr, "main: %s\n", mi.error); return 1; }
    if (vp_map_fixed(0x2f0000000ull, 0x100000) || vp_map_fixed(0x300000000ull, 1 << 20)) return 2;
    VpCpu c; memset(&c, 0, sizeof c); c.mxcsr = 0x1f80;
    c.r[VP_RSP] = 0x2f0000000ull + 0x100000 - 0x100 - 8; vp_st64(c.r[VP_RSP], VP_HOST_EXIT_ADDRESS);
    c.r[VP_RDI] = 0x300000000ull;
    counter = 0;
    if (vp_run(&c, mi.entry)) { fprintf(stderr, "translated: %s at %#" PRIx64 "\n", c.fault_what, c.fault_rip); return 1; }
    u64 mem2[8] = {0};
    counter = 0;
    const u64 expected = vp_main_native(mem2);
    printf("modules from the pack: translated %016" PRIx64 " native %016" PRIx64 " %s\n", c.r[VP_RAX], expected,
           c.r[VP_RAX] == expected ? "OK" : "MISMATCH");
    /* Hooks: an address nothing knows, handled by the embedder. */
    const VpEmbedderHooks hooks = {miss, miss_possible, NULL};
    vp_set_embedder_hooks(&hooks);
    VpCpu h; memset(&h, 0, sizeof h); h.mxcsr = 0x1f80;
    h.r[VP_RSP] = 0x2f0000000ull + 0x80000; vp_st64(h.r[VP_RSP], VP_HOST_EXIT_ADDRESS);
    const int r = vp_run(&h, HOOKED_ADDRESS);
    const int hooks_ok = r == 0 && misses == 1 && h.r[VP_RAX] == 0x600d;
    printf("embedder hooks: %s\n", hooks_ok ? "OK" : "FAILED");
    return (c.r[VP_RAX] == expected && hooks_ok) ? 0 : 1;
}
