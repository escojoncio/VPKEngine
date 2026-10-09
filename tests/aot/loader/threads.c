/* Runs the translated program on several threads at once, each with its own state, stack and
 * scratch memory, and checks every result: the translated code and the host keep no hidden
 * shared state. */
#include "vp_emit.h"
#include "vp_loader.h"
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
typedef unsigned long u64;
u64 vp_main_native(u64* mem);
/* The imports of tests/aot/loader/prog2.c, per thread. */
static _Thread_local u64 counter;
static void ext_double(VpCpu* c) { c->r[VP_RAX] = c->r[VP_RDI] * 2; c->rip = vp_pop64(c); }
static void ext_counter(VpCpu* c) { c->r[VP_RAX] = ++counter; c->rip = vp_pop64(c); }
static int resolve(const char* name, int function, VpNative* native, uint64_t* data, void* user) {
    (void)user; (void)function; (void)data;
    if (!strcmp(name, "vp_ext_double")) { *native = ext_double; return 1; }
    if (!strcmp(name, "vp_ext_counter")) { *native = ext_counter; return 1; }
    return 0;
}
u64 vp_ext_double(u64 x) { return x * 2; }
u64 vp_ext_counter(void) { return ++counter; }
static uint64_t entry;
static u64 expected;
static int failures;
static void* worker(void* arg) {
    const int n = (int)(intptr_t)arg;
    const uint64_t stack = 0x2f0000000ull + (uint64_t)n * 0x200000, data = 0x300000000ull + (uint64_t)n * 0x200000;
    if (vp_map_fixed(stack, 0x100000) || vp_map_fixed(data, 1 << 20)) { __atomic_fetch_add(&failures, 1, __ATOMIC_SEQ_CST); return NULL; }
    for (int round = 0; round < 20; ++round) {
        VpCpu c; memset(&c, 0, sizeof c); c.mxcsr = 0x1f80;
        memset((void*)(uintptr_t)data, 0, 1 << 20);
        c.r[VP_RSP] = stack + 0x100000 - 0x100 - 8; vp_st64(c.r[VP_RSP], VP_HOST_EXIT_ADDRESS);
        c.r[VP_RDI] = data;
        counter = 0;
        if (vp_run(&c, entry) || c.r[VP_RAX] != expected) { __atomic_fetch_add(&failures, 1, __ATOMIC_SEQ_CST); return NULL; }
    }
    return NULL;
}
int main(int argc, char** argv) {
    VpLoadedImage img;
    if (vp_load_image(argv[1], resolve, NULL, &img)) { fprintf(stderr, "load: %s\n", img.error); return 1; }
    entry = img.entry;
    u64 mem2[64]; memset(mem2, 0, sizeof mem2);
    counter = 0;
    expected = vp_main_native(mem2);
    pthread_t t[8];
    for (int i = 0; i < 8; ++i) pthread_create(&t[i], NULL, worker, (void*)(intptr_t)i);
    for (int i = 0; i < 8; ++i) pthread_join(t[i], NULL);
    printf("8 threads x 20 runs: %s\n", failures ? "FAILURES" : "all matched");
    return failures != 0;
}
