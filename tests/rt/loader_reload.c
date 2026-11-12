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

static int pipefd[2];
static int readable(uint64_t at) { return write(pipefd[1], (const void*)(uintptr_t)at, 1) == 1 && read(pipefd[0], &(char){0}, 1) == 1; }
static void mark(uint64_t at) { *(volatile unsigned char*)(uintptr_t)at = 0x77; }
static int marked(uint64_t at) { return *(volatile unsigned char*)(uintptr_t)at == 0x77; }

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

    printf(failures ? "loader_reload: %d FAILED\n" : "loader_reload: ok\n", failures);
    return failures != 0;
}
