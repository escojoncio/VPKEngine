/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * vp_loader.c with thousands of imports and relocations (a PS4 libc has ~1500 imports): every slot
 * that names the same symbol gets the same stub, different symbols get different stubs, a symbol
 * past the limit is refused and gives the mapping back, and the load takes a time that does not
 * depend on imports x relocations (the time is printed; it is not a pass condition).
 *   loader_many
 */
#include "vp_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

typedef struct { uint32_t type, flags; uint64_t offset, vaddr, paddr, filesz, memsz, align; } Ph;
#define IMG 0x200000u
#define BASE UINT64_C(0x500000000)
#define SLOTS 0x80000u /* offset of the slots in the image */

static unsigned resolved;
static void native(VpCpu* c) { c->rip = VP_HOST_EXIT_ADDRESS; }
static int resolve(const char* n, int function, VpNative* fn, uint64_t* data, void* user) {
    (void)n; (void)data; (void)user;
    ++resolved;
    if (function) *fn = native;
    return 1;
}

/* `nslots` slots, slot i naming symbol (i % nsyms). */
static int run(const char* what, unsigned nslots, unsigned nsyms, int expect_ok) {
    uint8_t* f = calloc(1, IMG);
    const uint16_t type = 3, machine = 62, phs = 56, phn = 2;
    const uint64_t phoff = 64, entry = BASE + 0x100;
    memcpy(f, "\x7f" "ELF\x02\x01\x01", 7);
    memcpy(f + 16, &type, 2); memcpy(f + 18, &machine, 2); memcpy(f + 24, &entry, 8); memcpy(f + 32, &phoff, 8);
    memcpy(f + 54, &phs, 2); memcpy(f + 56, &phn, 2);
    const Ph ph[2] = {{1, 7, 0, BASE, BASE, IMG, IMG, 0x1000}, {2, 6, 0x400, BASE + 0x400, BASE + 0x400, 0x40, 0x40, 8}};
    memcpy(f + 64, ph, sizeof ph);
    const uint64_t dyn[6] = {7, BASE + 0x1000, 8, (uint64_t)nslots * 24, 0, 0};
    memcpy(f + 0x400, dyn, sizeof dyn);
    for (unsigned i = 0; i < nslots; ++i) {
        const uint64_t r[3] = {BASE + SLOTS + 8 * i, ((uint64_t)1 << 32) | 6, 0};
        memcpy(f + 0x1000 + 24 * i, r, 24);
    }
    /* A directory of its own (mkdtemp): a fixed name in /tmp is shared by parallel runs and by other users. */
    char dir[512], path[600];
    const char* tmp = getenv("TMPDIR");
    snprintf(dir, sizeof dir, "%s/vp_loader_many_XXXXXX", tmp && *tmp ? tmp : "/tmp");
    if (!mkdtemp(dir)) return 2;
    snprintf(path, sizeof path, "%s/image.elf", dir);
    FILE* o = fopen(path, "wb");
    if (!o || fwrite(f, 1, IMG, o) != IMG) { rmdir(dir); return 2; }
    fclose(o);
    free(f);

    VpImport* im = calloc(nslots, sizeof *im);
    char (*names)[16] = calloc(nslots, 16);
    for (unsigned i = 0; i < nslots; ++i) {
        snprintf(names[i], 16, "sym_%u", i % nsyms);
        im[i] = (VpImport){names[i], SLOTS + 8 * i, 1}; /* the module's slots are offsets from the link base */
    }
    static VpModule mods[4]; /* registered for good: one per call */
    static char modname[4][16];
    static int used;
    VpModule* m = &mods[used];
    snprintf(modname[used], 16, "many%d", used);
    m->name = modname[used++];
    m->link_base = BASE; m->size = IMG; m->imports = im; m->import_count = nslots; m->relative = 1;
    m->fingerprint = vp_fingerprint(0, NULL, 0, NULL, 0);
    vp_register_module(m);

    VpLoadedImage out;
    resolved = 0;
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    const int rc = vp_load_module(path, m, 0, resolve, NULL, &out);
    remove(path);
    rmdir(dir);
    clock_gettime(CLOCK_MONOTONIC, &b);
    const double ms = (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6;
    int bad = 0;
    if ((rc == 0) != expect_ok) { printf("FAIL %s: load %s (%s)\n", what, rc ? "refused" : "accepted", out.error); bad = 1; }
    if (rc == 0) {
        uint64_t* slot = (uint64_t*)(uintptr_t)(BASE + SLOTS);
        for (unsigned i = 0; i < nslots && !bad; ++i) {
            if (slot[i] != slot[i % nsyms]) { printf("FAIL %s: slot %u has another stub than slot %u (same symbol)\n", what, i, i % nsyms); bad = 1; }
            if (i < nsyms && i && slot[i] == slot[i - 1]) { printf("FAIL %s: symbols %u and %u share a stub\n", what, i - 1, i); bad = 1; }
        }
        if (!bad && resolved != nsyms) { printf("FAIL %s: %u symbols asked of the resolver, %u expected\n", what, resolved, nsyms); bad = 1; }
        munmap((void*)(uintptr_t)BASE, IMG);
    } else if (!bad && msync((void*)(uintptr_t)BASE, 4096, MS_ASYNC) == 0) {
        printf("FAIL %s: the refused image stays mapped\n", what);
        bad = 1;
    }
    if (!bad) printf("ok   %s (%u slots, %u symbols, %.1f ms)\n", what, nslots, nsyms, ms);
    free(im); free(names);
    return bad;
}

int main(void) {
    int bad = 0;
    bad += run("shared_symbols", 20000, 3000, 1);
    bad += run("all_distinct", 4000, 4000, 1);
    bad += run("past_the_limit", 4100, 4100, 0);
    return bad ? 1 : 0;
}
