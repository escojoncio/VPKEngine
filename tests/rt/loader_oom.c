/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * vp_loader.c when an allocation fails at any point of a load that is refused afterwards (the code is not the translated one):
 * whatever fails, the refused image must not stay mapped (abandon_image / release_part_locked unmap it even without memory to
 * restore the embedder's reservation) and nothing may crash. Each run arms the failure of the K-th malloc/calloc/realloc of the
 * load, for K = 1.. until a load no longer reaches it.   loader_oom DIR   (GNU ld: -Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc)
 */
#include "vp_loader.h"
#include "elfgen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

void* __real_malloc(size_t);
void* __real_calloc(size_t, size_t);
void* __real_realloc(void*, size_t);
static volatile long countdown;  /* > 0: the countdown-th allocation from now fails */
static volatile long reached;    /* allocations seen while armed */
static volatile int latch, latched;  /* latch: once the K-th fails, every later allocation fails too (until disarmed) */
static int take(void) {
  if (latched) return 1;
  if (countdown > 0) { ++reached; if (--countdown == 0) { latched = latch; return 1; } }
  return 0;
}
void* __wrap_malloc(size_t n) { return take() ? NULL : __real_malloc(n); }
void* __wrap_calloc(size_t a, size_t b) { return take() ? NULL : __real_calloc(a, b); }
void* __wrap_realloc(void* p, size_t n) { return take() ? NULL : __real_realloc(p, n); }

static int pipefd[2];
static int readable(uint64_t at) { return write(pipefd[1], (const void*)(uintptr_t)at, 1) == 1 && read(pipefd[0], &(char){0}, 1) == 1; }

static VpModule mods[512];
static char names[512][16];
static VpImport imps[512];  /* (registered modules keep pointing at these) */

static void native_a(VpCpu* c) { c->rip = 0; }
static int resolve(const char* name, int function, VpNative* native, uint64_t* data, void* user) {
    (void)name; (void)function; (void)data; (void)user;
    *native = native_a;
    return 1;
}

int main(int argc, char** argv) {
    if (argc < 2 || pipe(pipefd)) return 2;
    int failures = 0;
    const uint64_t R = UINT64_C(0x700000000);
    if (mmap((void*)(uintptr_t)R, UINT64_C(1) << 30, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0) == MAP_FAILED) return 2;
    static const unsigned char zeros[0x200];
    const VpRange code = {0x100, 0x10};
    for (unsigned iter = 2; iter < 400; ++iter) {
        const unsigned k = iter / 2;  /* each K twice: only the K-th allocation fails, then every one from the K-th on */
        latch = iter & 1;
        const uint64_t base = R + (uint64_t)iter * 0x100000;
        char path[512];
        snprintf(path, sizeof path, "%s/oom%u.elf", argv[1], iter);
        const TestRela r = {base + 0x600, ((uint64_t)1 << 32) | 6, 0};
        if (elfgen(path, base, &r, 1, sizeof r, NULL, 0)) return 2;
        imps[iter] = (VpImport){"sym", base + 0x600, 1};
        VpModule* m = &mods[iter];
        snprintf(names[iter], 16, "oom%u", iter);
        m->name = names[iter]; m->imports = &imps[iter]; m->import_count = 1; m->link_base = base; m->size = TEST_IMG;
        m->code = &code; m->code_count = 1;
        m->fingerprint = vp_fingerprint((uint64_t)(uintptr_t)zeros, &code, 1, NULL, 0) ^ 1; /* never matches */
        vp_register_module(m);
        if (vp_map_fixed_over(base, TEST_IMG)) { printf("FAIL prepare\n"); return 1; } /* the embedder's prepared range */
        VpLoadedImage out;
        reached = 0; countdown = k;
        const int rc = vp_load_module(path, m, 0, resolve, NULL, &out);
        const long seen = reached;
        countdown = 0; latched = 0;
        remove(path);
        if (rc == 0) { printf("FAIL k=%u: a load with the wrong fingerprint worked\n", k); ++failures; }
        if (readable(base) && memcmp((const void*)(uintptr_t)base, "\x7f" "ELF", 4) == 0) {  /* (a range that was only prepared reads as zeros) */ printf("FAIL k=%u: the refused image stays mapped (allocation %u failed%s)\n", k, k, latch ? " and every later one" : ""); ++failures; }
        if (vp_module_at(base + 0x100) != NULL) { printf("FAIL k=%u: the refused module stays attached\n", k); ++failures; }
        vp_unmap_fixed(base, TEST_IMG);
        if (seen < (long)k && (iter & 1)) break; /* the load did not reach the K-th allocation: all of them were covered */
    }
    printf(failures ? "loader_oom: %d FAILED\n" : "loader_oom: ok\n", failures);
    return failures != 0;
}
