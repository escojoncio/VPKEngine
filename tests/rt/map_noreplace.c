/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * vp_map_fixed and the loader never take over memory that is already mapped (MAP_FIXED would
 * replace it silently): a mapping that overlaps is refused, and what was there is left as it was.
 *   map_noreplace DIR   (UBSan; ASan cannot map at these addresses)
 */
#include "vp_loader.h"
#include "elfgen.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#define BASE UINT64_C(0x500000000)
static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL "); printf(__VA_ARGS__); printf("\n"); ++failures; } } while (0)

static void* sentinel(uint64_t at, uint64_t size) {
    void* p = mmap((void*)(uintptr_t)at, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (p == MAP_FAILED) { printf("cannot set up the test at %#llx\n", (unsigned long long)at); exit(2); }
    memset(p, 0x5a, size);
    return p;
}

static int intact(uint64_t at, uint64_t size) {
    const unsigned char* p = (const unsigned char*)(uintptr_t)at;
    for (uint64_t i = 0; i < size; i += 4096) if (p[i] != 0x5a) return 0;
    return 1;
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    /* 1. a mapping over a mapping */
    const uint64_t a = UINT64_C(0x520000000);
    sentinel(a + 0x10000, 0x10000);
    CHECK(vp_map_fixed(a, 0x30000) != 0, "vp_map_fixed over an existing mapping succeeded");
    CHECK(intact(a + 0x10000, 0x10000), "vp_map_fixed replaced the memory that was there");
    CHECK(msync((void*)(uintptr_t)a, 4096, MS_ASYNC) != 0, "a refused vp_map_fixed left part of its range mapped");
    /* 2. twice the same address; after unmapping it works again */
    const uint64_t b = UINT64_C(0x530000000);
    CHECK(vp_map_fixed(b, 0x20000) == 0, "vp_map_fixed on free memory failed");
    CHECK(vp_map_fixed(b, 0x20000) != 0, "the same range mapped twice");
    munmap((void*)(uintptr_t)b, 0x20000);
    CHECK(vp_map_fixed(b, 0x20000) == 0, "vp_map_fixed after the range was freed failed");
    /* 3. a module whose image would land on something */
    char path[512];
    snprintf(path, sizeof path, "%s/noreplace.elf", argv[1]);
    if (elfgen(path, BASE, NULL, 0, 0, NULL, 0)) return 2;
    sentinel(BASE, 0x10000);
    VpModule m;
    memset(&m, 0, sizeof m);
    m.name = "noreplace";
    m.link_base = BASE;
    m.size = TEST_IMG;
    m.fingerprint = vp_fingerprint(0, NULL, 0, NULL, 0);
    vp_register_module(&m);
    VpLoadedImage out;
    CHECK(vp_load_module(path, &m, 0, NULL, NULL, &out) != 0, "a module loaded over an existing mapping");
    CHECK(intact(BASE, 0x10000), "the loader replaced the memory that was there");
    remove(path);
    printf(failures ? "map_noreplace: %d FAILED\n" : "map_noreplace: ok\n", failures);
    return failures != 0;
}
