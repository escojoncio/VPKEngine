/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * The loader's memory contract (runtime/vp_loader.h): a module loaded again replaces its own
 * earlier image (the game restarted in the process, or vp_detach_module + load), an image can go
 * into a range the embedder prepared inside its PROT_NONE reservation with vp_map_fixed_over, and
 * nothing else is ever replaced: not the app's memory, not a reservation nobody prepared, not
 * another module that shares a 64 KB block. vp_unmap_fixed gives back what the loader mapped.
 *   loader_reload DIR   (UBSan; ASan cannot map at these addresses)
 */
#include "vp_loader.h"
#include "elfgen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define A UINT64_C(0x500000000)
static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL "); printf(__VA_ARGS__); printf("\n"); ++failures; } } while (0)

static void native_a(VpCpu* c) { c->r[VP_RAX] = 0xa; c->rip = 0; }
static int resolve(const char* name, int function, VpNative* native, uint64_t* data, void* user) {
    (void)name; (void)function; (void)data; (void)user;
    *native = native_a;
    return 1;
}

static const char* dir;
static VpModule mods[8];
static VpImport imps[8];
static char names[8][16];

/* Module `idx` with its one import, linked at `base`; loaded at `load_at` (0: at `base`). */
static int load(unsigned idx, uint64_t base, uint64_t load_at, uint64_t fingerprint_xor, VpLoadedImage* out) {
    char path[512];
    snprintf(path, sizeof path, "%s/reload%u.elf", dir, idx);
    const TestRela r = {base + 0x600, ((uint64_t)1 << 32) | 6, 0};
    if (elfgen(path, base, &r, 1, sizeof r, NULL, 0)) exit(2);
    VpModule* m = &mods[idx];
    if (!m->name) {
        snprintf(names[idx], 16, "reload%u", idx);
        imps[idx] = (VpImport){"sym", load_at ? 0x600 : base + 0x600, 1};
        m->name = names[idx];
        m->imports = &imps[idx];
        m->import_count = 1;
        m->link_base = base;
        m->size = TEST_IMG;
        m->relative = load_at != 0;
        m->fingerprint = vp_fingerprint(0, NULL, 0, NULL, 0);
        vp_register_module(m);
    }
    VpRange code = {0x100, 0x10};
    m->code = fingerprint_xor ? &code : NULL; /* a code range makes the fingerprint check bite */
    m->code_count = fingerprint_xor ? 1 : 0;
    static const unsigned char zeros[0x200];
    m->fingerprint = fingerprint_xor ? vp_fingerprint((uint64_t)(uintptr_t)zeros, &code, 1, NULL, 0) ^ (fingerprint_xor == 1 ? 0 : 1) : vp_fingerprint(0, NULL, 0, NULL, 0);
    const int rc = vp_load_module(path, m, load_at, resolve, NULL, out);
    remove(path);
    return rc;
}


/* Like load(), but the image is `blocks` 64 KB blocks long (the segment's memsz; the file stays 4 KB) and relocatable. */
static int load_big(unsigned idx, uint64_t base, uint64_t load_at, uint64_t blocks, VpLoadedImage* out) {
    char path[512];
    snprintf(path, sizeof path, "%s/big%u.elf", dir, idx);
    const TestRela r = {base + 0x600, ((uint64_t)1 << 32) | 6, 0};
    if (elfgen(path, base, &r, 1, sizeof r, NULL, 0)) exit(2);
    const uint64_t memsz = blocks * 0x10000;
    FILE* f = fopen(path, "r+b");
    if (!f || fseek(f, 64 + 40, SEEK_SET) || fwrite(&memsz, 8, 1, f) != 1) exit(2); /* program header 0: p_memsz */
    fclose(f);
    VpModule* m = &mods[idx];
    if (!m->name) {
        snprintf(names[idx], 16, "big%u", idx);
        imps[idx] = (VpImport){"sym", 0x600, 1};
        m->name = names[idx]; m->imports = &imps[idx]; m->import_count = 1; m->link_base = base; m->size = memsz; m->relative = 1;
        m->fingerprint = vp_fingerprint(0, NULL, 0, NULL, 0);
        vp_register_module(m);
    }
    m->link_base = base; m->size = memsz; /* the same module can come with another size and base */
    const int rc = vp_load_module(path, m, load_at, resolve, NULL, out);
    remove(path);
    return rc;
}

static int pipefd[2];
static int readable(uint64_t at) { return write(pipefd[1], (const void*)(uintptr_t)at, 1) == 1 && read(pipefd[0], &(char){0}, 1) == 1; }
static void mark(uint64_t at) { *(volatile unsigned char*)(uintptr_t)at = 0x77; }
static int marked(uint64_t at) { return *(volatile unsigned char*)(uintptr_t)at == 0x77; }

#ifdef __linux__
/* The number of PROT_NONE mappings that start inside [lo,hi), from /proc/self/maps (those separated by a gap do not merge). */
static unsigned prot_none_in(uint64_t lo, uint64_t hi) {
    FILE* f = fopen("/proc/self/maps", "r");
    unsigned n = 0;
    char line[512];
    while (f && fgets(line, sizeof line, f)) {
        unsigned long long a, b;
        char perms[8];
        if (sscanf(line, "%llx-%llx %7s", &a, &b, perms) == 3 && a >= lo && a < hi && !strncmp(perms, "---", 3)) ++n;
    }
    if (f) fclose(f);
    return n;
}
#endif

int main(int argc, char** argv) {
    if (argc < 2 || pipe(pipefd)) return 2;
    dir = argv[1];
    VpLoadedImage out;

    /* 1. the same module loaded again in the same range (game restarted, or detach + load) */
    CHECK(load(0, A, 0, 0, &out) == 0, "first load: %s", out.error);
    mark(A + 0x700);
    CHECK(load(0, A, 0, 0, &out) == 0, "the same module loaded again in its range: %s", out.error);
    CHECK(!marked(A + 0x700), "the reload did not replace the earlier image");
    vp_detach_module(&mods[0]);
    CHECK(load(0, A, 0, 0, &out) == 0, "load after vp_detach_module: %s", out.error);
    CHECK(vp_module_at(A + 0x100) == &mods[0], "the reloaded module is not attached");

    /* 2. a second module in the same 64 KB block fails cleanly and leaves the first alone */
    mark(A + 0x700);
    CHECK(load(1, A + 0x4000, 0, 0, &out) != 0, "two modules in one 64 KB block both loaded");
    CHECK(marked(A + 0x700) && vp_module_at(A + 0x100) == &mods[0], "the refused load damaged the module already there");
    CHECK(!readable(A + 0x4000 + 0x10000), "the refused load left something mapped");

    /* 3. the app's own mapping and a PROT_NONE reservation are never replaced by default; the
     * embedder prepares the range explicitly */
    const uint64_t R = UINT64_C(0x600000000);
    CHECK(mmap((void*)(uintptr_t)R, 1 << 24, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0) != MAP_FAILED, "cannot reserve");
    CHECK(load(2, R + 0x10000, 0, 0, &out) != 0, "a module loaded into a reservation nobody prepared");
    CHECK(!readable(R + 0x10000), "the refused load replaced the reservation");
    CHECK(vp_map_fixed(R + 0x10000, 0x1000) != 0, "vp_map_fixed replaced a reservation");
    CHECK(vp_map_fixed_over(R + 0x10000, TEST_IMG) == 0, "vp_map_fixed_over failed in a reservation");
    CHECK(readable(R + 0x10000) && !readable(R + 0x20000) && !readable(R), "vp_map_fixed_over touched more than its 64 KB block");
    CHECK(load(2, R + 0x10000, 0, 0, &out) == 0, "load into the prepared range: %s", out.error);
    CHECK(vp_map_fixed_over(R + 0x10000, TEST_IMG) != 0, "vp_map_fixed_over replaced an image");
    CHECK(vp_map_fixed_over(A, 0x1000) != 0, "vp_map_fixed_over replaced another module's image");
    CHECK(load(2, R + 0x10000, 0, 0, &out) == 0, "reload in the reservation: %s", out.error);

    /* 4. what the loader maps, it can unmap (the image's own range, a part of a range); nothing else */
    const uint64_t P = UINT64_C(0x620000000);
    CHECK(vp_map_fixed(P, 0x30000) == 0, "vp_map_fixed");
    CHECK(vp_map_fixed_over(P, 0x10000) != 0, "vp_map_fixed_over replaced a vp_map_fixed mapping");
    CHECK(vp_unmap_fixed(P + 0x10000, 0x10000) == 0, "unmap of the middle of a mapping");
    CHECK(readable(P) && !readable(P + 0x10000) && readable(P + 0x20000), "the middle was not the only part unmapped");
    CHECK(vp_map_fixed(P + 0x10000, 0x10000) == 0, "the freed middle cannot be mapped again");
    CHECK(vp_unmap_fixed(P, 0x30000) == 0, "unmap of the whole");
    CHECK(!readable(P) && !readable(P + 0x10000) && !readable(P + 0x20000), "the unmapped range is still there");
    void* mine = mmap((void*)(uintptr_t)(P + 0x40000), 0x10000, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    CHECK(mine != MAP_FAILED && vp_unmap_fixed(P + 0x40000, 0x10000) != 0 && readable(P + 0x40000), "vp_unmap_fixed unmapped memory that is not the loader's");
    CHECK(vp_unmap_fixed(out.base, out.end - out.base) == 0 && !readable(out.base), "unmap of the image");
    CHECK(vp_module_at(R + 0x10100) == NULL, "an unmapped image's module stays attached");
    CHECK(load(2, R + 0x10000, 0, 0, &out) == 0, "load after the image was unmapped: %s", out.error);

    /* 5. a reload that fails (the code is not the translated one) leaves the range unmapped and the
     * module detached; the next good load works */
    CHECK(load(3, A + 0x100000, 0, 2, &out) != 0, "a load with the wrong fingerprint worked");
    CHECK(!readable(A + 0x100000), "the failed load left its image mapped");
    CHECK(load(3, A + 0x100000, 0, 1, &out) == 0, "a good load after a failed one: %s", out.error);
    CHECK(load(3, A + 0x100000, 0, 2, &out) != 0, "a reload with the wrong fingerprint worked");
    CHECK(!readable(A + 0x100000) && vp_module_at(A + 0x100100) == NULL, "the failed reload left the old image mapped or attached");
    CHECK(load(3, A + 0x100000, 0, 1, &out) == 0, "a good load after a failed reload: %s", out.error);

    /* 6. a position-independent module loaded again somewhere else releases its first place */
    const uint64_t W1 = UINT64_C(0x640000000), W2 = UINT64_C(0x641000000);
    CHECK(load(4, A + 0x200000, W1, 0, &out) == 0, "pic load: %s", out.error);
    CHECK(readable(W1) && load(4, A + 0x200000, W2, 0, &out) == 0, "pic load elsewhere: %s", out.error);
    CHECK(readable(W2) && !readable(W1), "the earlier place of a reloaded module stays mapped");

    /* 7. a position-independent module loaded into a range the embedder prepared and then again somewhere else gives the first
     * range back as prepared (PROT_NONE, still the embedder's reservation), not as a hole */
    const uint64_t V = UINT64_C(0x680000000);
    CHECK(mmap((void*)(uintptr_t)V, 1 << 24, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0) != MAP_FAILED, "cannot reserve V");
    CHECK(vp_map_fixed_over(V + 0x10000, TEST_IMG) == 0 && vp_map_fixed_over(V + 0x100000, TEST_IMG) == 0, "prepare two ranges");
    CHECK(load(5, A + 0x300000, V + 0x10000, 0, &out) == 0, "pic load into a prepared range: %s", out.error);
    CHECK(readable(V + 0x10000), "the image is not readable");
    CHECK(load(5, A + 0x300000, V + 0x100000, 0, &out) == 0, "pic load into the other prepared range: %s", out.error);
    CHECK(readable(V + 0x100000) && !readable(V + 0x10000), "the first place of the reloaded module stays readable");
    /* (Without MAP_FIXED_NOREPLACE, e.g. on macOS, the address is only a hint: an occupied one gives another address.) */
    int probe = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_FIXED_NOREPLACE
    probe |= MAP_FIXED_NOREPLACE;
#endif
    void* hole = mmap((void*)(uintptr_t)(V + 0x10000), 0x10000, PROT_READ, probe, -1, 0);
    CHECK(hole == MAP_FAILED || (uint64_t)(uintptr_t)hole != V + 0x10000, "the first place was unmapped: a hole in the embedder's reservation");
    if (hole != MAP_FAILED) munmap(hole, 0x10000);
    CHECK(vp_map_fixed_over(V + 0x10000, TEST_IMG) == 0, "the first place is not the embedder's range again");
    CHECK(load(5, A + 0x300000, V + 0x10000, 0, &out) == 0, "pic load back into the first range: %s", out.error);

    /* 8. a very fragmented image (2050 pieces, after the embedder unmapped every other 64 KB block) reloaded somewhere else leaves
     * nothing behind: every piece is released (no bound on the number of entries), and a neighbour that is not the module's
     * (a vp_map_fixed mapping between two of the pieces) is not touched */
    {
        const uint64_t F1 = UINT64_C(0x6c0000000), F2 = UINT64_C(0x6e0000000), blocks = 4100;
        CHECK(load_big(6, F1, F1, blocks, &out) == 0, "big pic load: %s", out.error);
        for (uint64_t b = 1; b < blocks; b += 2) CHECK(vp_unmap_fixed(F1 + b * 0x10000, 0x10000) == 0, "unmap of block %llu", (unsigned long long)b);
        CHECK(vp_map_fixed(F1 + 1 * 0x10000, 0x10000) == 0, "vp_map_fixed in a gap of the image");
        mark(F1 + 0x10000);
        CHECK(load_big(6, F1, F2, blocks, &out) == 0, "big pic load elsewhere: %s", out.error);
        unsigned left = 0;
        for (uint64_t b = 0; b < blocks; b += 2) left += readable(F1 + b * 0x10000);
        CHECK(left == 0, "%u pieces of the earlier image stay mapped", left);
        CHECK(readable(F2) && readable(F2 + (blocks - 1) * 0x10000), "the new image is not all there");
        CHECK(marked(F1 + 0x10000), "the release touched a mapping that is not the module's");
        CHECK(vp_unmap_fixed(F1 + 0x10000, 0x10000) == 0, "unmap of the neighbour");
        CHECK(vp_unmap_fixed(F2, blocks * 0x10000) == 0, "unmap of the new image");
    }

    /* 9. an image loaded into a prepared range, fragmented by the embedder (every other block unmapped with vp_unmap_fixed), and
     * reloaded somewhere else: the prepared range comes back as exactly one PROT_NONE mapping per remaining piece (n+1 for n
     * unmapped blocks; no piece of the reservation is lost however many there are), and nothing stale stays registered */
    {
        static const unsigned counts[] = {1, 7, 8, 20, 2000};
        for (unsigned t = 0; t < sizeof counts / sizeof *counts; ++t) {
            const uint64_t n = counts[t], blocks = 2 * n + 1, len = blocks * 0x10000;
            const uint64_t R = UINT64_C(0x7000000000) + (uint64_t)t * UINT64_C(0x100000000), F2 = R + UINT64_C(0x80000000);
            CHECK(mmap((void*)(uintptr_t)R, len, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0) != MAP_FAILED, "n=%llu: cannot reserve", (unsigned long long)n);
            CHECK(vp_map_fixed_over(R, len) == 0, "n=%llu: prepare", (unsigned long long)n);
            CHECK(load_big(7, R, R, blocks, &out) == 0, "n=%llu: load into the prepared range: %s", (unsigned long long)n, out.error);
            for (uint64_t b = 1; b < blocks; b += 2) CHECK(vp_unmap_fixed(R + b * 0x10000, 0x10000) == 0, "n=%llu: unmap of block %llu", (unsigned long long)n, (unsigned long long)b);
            CHECK(load_big(7, R, F2, blocks, &out) == 0, "n=%llu: reload elsewhere: %s", (unsigned long long)n, out.error);
#ifdef __linux__
            const unsigned none = prot_none_in(R, R + len);
            CHECK(none == n + 1, "n=%llu: %u PROT_NONE mappings came back, %llu expected", (unsigned long long)n, none, (unsigned long long)(n + 1));
#endif
            /* no entry is stale: a gap block takes a vp_map_fixed, a restored block can be unmapped and mapped again */
            CHECK(vp_map_fixed(R + 0x10000, 0x10000) == 0, "n=%llu: vp_map_fixed in a gap", (unsigned long long)n);
            CHECK(vp_unmap_fixed(R + 0x10000, 0x10000) == 0, "n=%llu: unmap of it", (unsigned long long)n);
            CHECK(vp_unmap_fixed(R, 0x10000) == 0 && vp_map_fixed(R, 0x10000) == 0, "n=%llu: the first restored block is not the loader's", (unsigned long long)n);
            CHECK(vp_unmap_fixed(R, 0x10000) == 0, "n=%llu: unmap of it", (unsigned long long)n);
            for (uint64_t b = 2; b < blocks; b += 2) CHECK(vp_unmap_fixed(R + b * 0x10000, 0x10000) == 0, "n=%llu: unmap of restored block %llu", (unsigned long long)n, (unsigned long long)b);
            CHECK(vp_unmap_fixed(F2, len) == 0, "n=%llu: unmap of the new image", (unsigned long long)n);
        }
    }

    printf(failures ? "loader_reload: %d FAILED\n" : "loader_reload: ok\n", failures);
    return failures != 0;
}
