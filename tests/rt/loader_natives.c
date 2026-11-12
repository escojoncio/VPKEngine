/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * The tables behind the stubs of the imports have no small fixed limit:
 *  - the registry that makes the same import of two modules share a stub keeps sharing after
 *    12288 distinct names (it used to stop there, and every later module then registered natives
 *    of its own, without bound);
 *  - when the table of natives is full vp_register_native says so, and a load whose stubs it
 *    cannot register fails (and gives its mapping back) instead of returning modules whose imports
 *    reach no native.
 *   loader_natives DIR   (UBSan; ASan cannot map at these addresses)
 */
#include "vp_loader.h"
#include "elfgen.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#define IMG 0x200000u
#define SLOTS 0x80000u
#define PER 4096u
#define NAT_BASE UINT64_C(0x7fff00000000) /* above the stubs: each registration appends, none shifts the table */
static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL "); printf(__VA_ARGS__); printf("\n"); ++failures; } } while (0)

static void native_a(VpCpu* c) { c->r[VP_RAX] = 0xa; c->rip = 0; }
static int resolve(const char* name, int function, VpNative* native, uint64_t* data, void* user) {
    (void)name; (void)data; (void)user;
    if (function) *native = native_a;
    return 1;
}

/* A module linked at `base` whose slot i imports "sym_<first + i>"; loaded; the stubs go to `stubs`. */
static int load_names(const char* dir, unsigned idx, unsigned first, unsigned count, uint64_t* stubs, VpLoadedImage* out) {
    static VpModule mods[16];
    static char modnames[16][16];
    const uint64_t base = UINT64_C(0x500000000) + (uint64_t)idx * 0x10000000;
    uint8_t* f = calloc(1, IMG);
    const uint16_t type = 3, machine = 62, phs = 56, phn = 2;
    const uint64_t phoff = 64, entry = base + 0x100;
    memcpy(f, "\x7f" "ELF\x02\x01\x01", 7);
    memcpy(f + 16, &type, 2); memcpy(f + 18, &machine, 2); memcpy(f + 24, &entry, 8); memcpy(f + 32, &phoff, 8);
    memcpy(f + 54, &phs, 2); memcpy(f + 56, &phn, 2);
    const TestPhdr ph[2] = {{1, 7, 0, base, base, IMG, IMG, 0x1000}, {2, 6, 0x400, base + 0x400, base + 0x400, 0x40, 0x40, 8}};
    memcpy(f + 64, ph, sizeof ph);
    const uint64_t dyn[6] = {7, base + 0x1000, 8, (uint64_t)count * 24, 0, 0};
    memcpy(f + 0x400, dyn, sizeof dyn);
    VpImport* im = calloc(count, sizeof *im);
    char (*names)[16] = calloc(count, 16);
    for (unsigned i = 0; i < count; ++i) {
        const uint64_t r[3] = {base + SLOTS + 8 * i, ((uint64_t)1 << 32) | 6, 0};
        memcpy(f + 0x1000 + 24 * i, r, 24);
        snprintf(names[i], 16, "sym_%u", first + i);
        im[i] = (VpImport){names[i], base + SLOTS + 8 * i, 1};
    }
    char path[512];
    snprintf(path, sizeof path, "%s/natives%u.elf", dir, idx);
    FILE* o = fopen(path, "wb");
    if (!o || fwrite(f, 1, IMG, o) != IMG) exit(2);
    fclose(o);
    free(f);
    VpModule* m = &mods[idx];
    snprintf(modnames[idx], 16, "natives%u", idx);
    m->name = modnames[idx];
    m->link_base = base; m->size = IMG; m->imports = im; m->import_count = count;
    m->fingerprint = vp_fingerprint(0, NULL, 0, NULL, 0);
    vp_register_module(m);
    const int rc = vp_load_module(path, m, 0, resolve, NULL, out);
    remove(path);
    if (rc == 0 && stubs) memcpy(stubs, (void*)(uintptr_t)(base + SLOTS), 8 * (size_t)count);
    return rc;
}

static int shared_past_12288(const char* dir) {
    static uint64_t stubs[4][PER], again[2][PER];
    VpLoadedImage out;
    for (unsigned k = 0; k < 4; ++k) {
        if (load_names(dir, k, k * PER, PER, stubs[k], &out)) { printf("FAIL load %u: %s\n", k, out.error); return 1; }
    }
    /* 16384 distinct names so far; the last 4096 (names 12288..16383) and the first again */
    if (load_names(dir, 4, 3 * PER, PER, again[0], &out) || load_names(dir, 5, 0, PER, again[1], &out)) { printf("FAIL reload: %s\n", out.error); return 1; }
    unsigned late = 0, early = 0;
    for (unsigned i = 0; i < PER; ++i) { late += again[0][i] != stubs[3][i]; early += again[1][i] != stubs[0][i]; }
    CHECK(late == 0, "%u of the %u imports named past the 12288th got a stub of their own", late, PER);
    CHECK(early == 0, "%u of the %u first imports got a stub of their own", early, PER);
    return 0;
}

static int table_full(const char* dir) {
    unsigned n = 0;
    while (vp_register_native(NAT_BASE + (uint64_t)n * 16, native_a) == 0) {
        if (++n > (1u << 22)) { printf("FAIL the table of natives never fills\n"); return 1; }
    }
    CHECK(vp_register_native(NAT_BASE, native_a) == 0, "re-registering an address of a full table failed");
    VpLoadedImage out;
    uint64_t stubs[1];
    CHECK(load_names(dir, 6, 100000, 1, stubs, &out) != 0, "a module whose stub cannot be registered loaded");
    CHECK(msync((void*)(uintptr_t)(UINT64_C(0x500000000) + 6 * 0x10000000), 4096, MS_ASYNC) != 0, "the failed load left its image mapped");
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    shared_past_12288(argv[1]);
    fflush(stdout);
    const pid_t pid = fork(); /* the full table is for good: in a child */
    if (pid == 0) { if (!freopen("/dev/null", "w", stderr)) _exit(2); table_full(argv[1]); _exit(failures != 0); }
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status)) ++failures;
    printf(failures ? "loader_natives: %d FAILED\n" : "loader_natives: ok\n", failures);
    return failures != 0;
}
