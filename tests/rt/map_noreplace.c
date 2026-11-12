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
#include <sys/wait.h>
#include <unistd.h>

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
    CHECK(vp_unmap_fixed(b, 0x20000) == 0, "vp_unmap_fixed of a vp_map_fixed range failed");
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
    /* 4. vp_map_fixed never takes a range an embedder prepared (vp_map_fixed_over) */
    const uint64_t x = UINT64_C(0x540000000);
    CHECK(vp_map_fixed_over(x, 0x10000) == 0, "vp_map_fixed_over on free memory failed");
    *(volatile unsigned char*)(uintptr_t)x = 0x77;
    CHECK(vp_map_fixed(x - 0x10000, 0x30000) != 0, "vp_map_fixed replaced a prepared range");
    CHECK(*(volatile unsigned char*)(uintptr_t)x == 0x77, "vp_map_fixed wiped a prepared range");
    CHECK(msync((void*)(uintptr_t)(x - 0x10000), 4096, MS_ASYNC) != 0, "a refused vp_map_fixed left part of its range mapped");
    CHECK(vp_unmap_fixed(x, 0x10000) == 0, "vp_unmap_fixed of a prepared range failed");
    /* 5. empty and wrapping ranges are refused by all three */
    const uint64_t z = UINT64_C(0x550000000);
    CHECK(vp_map_fixed(z, 0) != 0, "vp_map_fixed with size 0 succeeded");
    CHECK(vp_map_fixed_over(z, 0) != 0, "vp_map_fixed_over with size 0 succeeded");
    CHECK(vp_unmap_fixed(z, 0) != 0, "vp_unmap_fixed with size 0 succeeded");
    CHECK(vp_map_fixed(UINT64_MAX - 0x1000, 0x10000) != 0, "vp_map_fixed wrapping around succeeded");
    CHECK(vp_map_fixed_over(UINT64_MAX - 0x1000, 0x10000) != 0, "vp_map_fixed_over wrapping around succeeded");
    CHECK(vp_unmap_fixed(UINT64_MAX - 0x1000, 0x10000) != 0, "vp_unmap_fixed wrapping around succeeded");
    CHECK(vp_map_fixed(z, UINT64_MAX) != 0, "vp_map_fixed with a huge size succeeded");
    /* 6. a load that fails inside a prepared range leaves PROT_NONE there, not a hole */
    {
        const uint64_t base2 = UINT64_C(0x560000000);
        snprintf(path, sizeof path, "%s/noreplace2.elf", argv[1]);
        if (elfgen(path, base2, NULL, 0, 0, NULL, 0)) return 2;
        void* r = mmap((void*)(uintptr_t)base2, TEST_IMG, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
        if (r == MAP_FAILED) return 2;
        CHECK(vp_map_fixed_over(base2, TEST_IMG) == 0, "vp_map_fixed_over of the reservation failed");
        VpRange code = {0x100, 0x10};
        VpModule m2;
        memset(&m2, 0, sizeof m2);
        m2.name = "noreplace2";
        m2.link_base = base2;
        m2.size = TEST_IMG;
        m2.code = &code;
        m2.code_count = 1;
        m2.fingerprint = 12345; /* not the hash of the code in the file */
        vp_register_module(&m2);
        VpLoadedImage out2;
        CHECK(vp_load_module(path, &m2, 0, NULL, NULL, &out2) != 0, "a module with the wrong code loaded");
        CHECK(msync((void*)(uintptr_t)base2, 4096, MS_ASYNC) == 0, "the failed load left a hole in the prepared range");
        fflush(stdout);
        const pid_t pid = fork();
        if (pid == 0) {
            if (!freopen("/dev/null", "w", stderr)) _exit(2);
            volatile unsigned char c = *(volatile unsigned char*)(uintptr_t)base2;
            (void)c;
            _exit(0); /* readable: not PROT_NONE */
        }
        int status = 0;
        waitpid(pid, &status, 0);
        CHECK(!(WIFEXITED(status) && WEXITSTATUS(status) == 0), "the failed load left the prepared range readable (not PROT_NONE)");
        CHECK(vp_map_fixed(base2, TEST_IMG) != 0, "vp_map_fixed took the range given back as prepared");
        CHECK(vp_map_fixed_over(base2, TEST_IMG) == 0, "the range cannot be prepared again after the failed load");
        CHECK(vp_unmap_fixed(base2, TEST_IMG) == 0, "vp_unmap_fixed of the given-back range failed");
        remove(path);
    }
    /* 7. a reload that fails inside a prepared range: it was prepared before the first (good) load */
    {
        const uint64_t base3 = UINT64_C(0x570000000);
        snprintf(path, sizeof path, "%s/noreplace3.elf", argv[1]);
        if (elfgen(path, base3, NULL, 0, 0, NULL, 0)) return 2;
        if (mmap((void*)(uintptr_t)base3, TEST_IMG, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == MAP_FAILED) return 2;
        CHECK(vp_map_fixed_over(base3, TEST_IMG) == 0, "vp_map_fixed_over of the reservation failed");
        VpRange code = {0x100, 0x10};
        VpModule m3;
        memset(&m3, 0, sizeof m3);
        m3.name = "noreplace3";
        m3.link_base = base3;
        m3.size = TEST_IMG;
        m3.fingerprint = vp_fingerprint(0, NULL, 0, NULL, 0);
        vp_register_module(&m3);
        VpLoadedImage out3;
        CHECK(vp_load_module(path, &m3, 0, NULL, NULL, &out3) == 0, "the first load into the prepared range failed: %s", out3.error);
        m3.code = &code;
        m3.code_count = 1;
        m3.fingerprint = 12345;
        CHECK(vp_load_module(path, &m3, 0, NULL, NULL, &out3) != 0, "the reload with the wrong code worked");
        CHECK(msync((void*)(uintptr_t)base3, 4096, MS_ASYNC) == 0, "the failed reload left a hole in the reservation");
        fflush(stdout);
        const pid_t pid = fork();
        if (pid == 0) {
            if (!freopen("/dev/null", "w", stderr)) _exit(2);
            volatile unsigned char c = *(volatile unsigned char*)(uintptr_t)base3;
            (void)c;
            _exit(0);
        }
        int status = 0;
        waitpid(pid, &status, 0);
        CHECK(!(WIFEXITED(status) && WEXITSTATUS(status) == 0), "the failed reload left the range readable (not PROT_NONE)");
        CHECK(vp_unmap_fixed(base3, TEST_IMG) == 0, "vp_unmap_fixed after the failed reload failed");
        remove(path);
    }
    /* 8. a load whose 64 KB rounding wraps around the address space is refused, not crashed on */
    {
        snprintf(path, sizeof path, "%s/noreplace4.elf", argv[1]);
        const uint64_t high = UINT64_C(0xFFFFFFFFFFFFE000);
        if (elfgen(path, high, NULL, 0, 0, NULL, 0)) return 2;
        VpModule m4;
        memset(&m4, 0, sizeof m4);
        m4.name = "noreplace4";
        m4.link_base = high;
        m4.size = TEST_IMG;
        m4.fingerprint = vp_fingerprint(0, NULL, 0, NULL, 0);
        vp_register_module(&m4);
        VpLoadedImage out4;
        CHECK(vp_load_module(path, &m4, 0, NULL, NULL, &out4) != 0, "a load that wraps around the address space worked");
        /* --pic placed so that it wraps */
        snprintf(path, sizeof path, "%s/noreplace5.elf", argv[1]);
        if (elfgen(path, UINT64_C(0x580000000), NULL, 0, 0, NULL, 0)) return 2;
        VpModule m5;
        memset(&m5, 0, sizeof m5);
        m5.name = "noreplace5";
        m5.link_base = UINT64_C(0x580000000);
        m5.size = TEST_IMG;
        m5.relative = 1;
        m5.fingerprint = vp_fingerprint(0, NULL, 0, NULL, 0);
        vp_register_module(&m5);
        CHECK(vp_load_module(path, &m5, UINT64_C(0xFFFFFFFFFFFF0000), NULL, NULL, &out4) != 0, "a --pic load that wraps around the address space worked");
        remove(path);
        snprintf(path, sizeof path, "%s/noreplace4.elf", argv[1]);
        remove(path);
    }
    snprintf(path, sizeof path, "%s/noreplace.elf", argv[1]);
    remove(path);
    printf(failures ? "map_noreplace: %d FAILED\n" : "map_noreplace: ok\n", failures);
    return failures != 0;
}
