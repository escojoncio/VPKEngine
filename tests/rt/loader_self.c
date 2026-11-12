/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * The SELF unwrapper of vp_loader.c and the limits it shares with tools/vpaot/image.cpp (runtime/vp_self.h): a SELF whose ELF
 * has its segment far into the image because of alignment (-z max-page-size=2M) is accepted ("bad SELF" is the unwrapper's
 * answer; what the loader says later about the image does not matter here), one that puts a few bytes at 2 GiB is refused
 * before the image is reserved.   loader_self DIR
 */
#include "vp_loader.h"
#include "vp_self.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint32_t type, flags; uint64_t offset, vaddr, paddr, filesz, memsz, align; } Ph;

/* A SELF with one data segment: program header `p`, its `p.filesz` bytes in the SELF. */
static int write_self(const char* path, Ph p) {
    uint8_t f[512];
    memset(f, 0, sizeof f);
    memcpy(f, "O\x15=\x1d", 4);
    const uint16_t count = 1;
    memcpy(f + 24, &count, 2);
    const uint64_t flags = 0x800, size = p.filesz, base = 32 + 32, data = base + 64 + 56 + 16;
    memcpy(f + 32, &flags, 8); memcpy(f + 40, &data, 8); memcpy(f + 48, &size, 8); memcpy(f + 56, &size, 8);
    const uint64_t phoff = 64, entry = 0;
    const uint16_t phnum = 1, machine = 62;
    memcpy(f + base, "\x7f" "ELF\x02\x01\x01", 7);
    memcpy(f + base + 18, &machine, 2); memcpy(f + base + 24, &entry, 8); memcpy(f + base + 32, &phoff, 8);
    memcpy(f + base + 56, &phnum, 2);
    memcpy(f + base + phoff, &p, 56);
    memset(f + data, 0x90, p.filesz);
    FILE* o = fopen(path, "wb");
    if (!o) return -1;
    const size_t n = data + p.filesz;
    const int ok = fwrite(f, 1, n, o) == n;
    return fclose(o) == 0 && ok ? 0 : -1;
}

static int check(const char* dir, const char* name, Ph p, int accepted) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s.self", dir, name);
    if (write_self(path, p)) { printf("FAIL %s: cannot write\n", name); return 1; }
    static VpModule mods[32];  /* the registry keeps the pointers */
    static int nmods;
    VpModule* mp = &mods[nmods++];
    mp->name = name; mp->link_base = UINT64_C(0x500000000); mp->size = 0x1000;
    vp_register_module(mp);
    VpLoadedImage out;
    vp_load_module(path, mp, 0, NULL, NULL, &out);
    const int refused = strcmp(out.error, "bad SELF") == 0;
    if (refused == accepted) { printf("FAIL %s: %s (%s)\n", name, refused ? "refused" : "accepted", out.error); return 1; }
    printf("ok   %s\n", name);
    return 0;
}


/* n program headers of one byte at offset 0 (align `align`), then one of 16 bytes at `far`; `backed`: 1 a SELF segment for each of the n, 2 the same without the 0x800 flag. */
static int write_many(const char* path, unsigned n, uint64_t align, uint64_t far, int backed, uint64_t step) {
    const unsigned count = backed ? n : 0;
    const uint64_t base = 32 + (uint64_t)count * 32, phoff = 64, tail = base + phoff + (uint64_t)(n + 1) * 56;
    const size_t total = tail + 16;
    uint8_t* f = calloc(1, total);
    if (!f) return -1;
    memcpy(f, "O\x15=\x1d", 4);
    const uint16_t c16 = (uint16_t)count, phnum = (uint16_t)(n + 1), machine = 62;
    memcpy(f + 24, &c16, 2);
    for (unsigned i = 0; i < count; ++i) {
        const uint64_t e[4] = {(backed == 2 ? 0 : 0x800) | ((uint64_t)i << 20), 0, 1, 1};  /* backed 2: entries without the blocked flag */
        memcpy(f + 32 + i * 32, e, 32);
    }
    memcpy(f + base, "\x7f" "ELF\x02\x01\x01", 7);
    memcpy(f + base + 18, &machine, 2); memcpy(f + base + 32, &phoff, 8); memcpy(f + base + 56, &phnum, 2);
    for (unsigned i = 0; i <= n; ++i) {
        const Ph p = i < n ? (Ph){1, 5, (uint64_t)i * step, 0, 0, 1, 1, align} : (Ph){1, 5, far, 0, 0, 16, 16, 0x1000};
        memcpy(f + base + phoff + (size_t)i * 56, &p, 56);
    }
    FILE* o = fopen(path, "wb");
    if (!o) { free(f); return -1; }
    const int ok = fwrite(f, 1, total, o) == total;
    free(f);
    return fclose(o) == 0 && ok ? 0 : -1;
}

/* accepted: the unwrapper let it pass (what the loader says about the image later does not matter) */
static int check_many(const char* dir, const char* name, unsigned n, uint64_t align, uint64_t far, int backed, uint64_t step, int accepted) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s.self", dir, name);
    if (write_many(path, n, align, far, backed, step)) { printf("FAIL %s: cannot write\n", name); return 1; }
    static VpModule mods[32];  /* the registry keeps the pointers */
    static int nmods;
    VpModule* mp = &mods[nmods++];
    mp->name = name; mp->link_base = UINT64_C(0x500000000); mp->size = 0x1000;
    vp_register_module(mp);
    VpLoadedImage out;
    vp_load_module(path, mp, 0, NULL, NULL, &out);
    if ((strcmp(out.error, "bad SELF") != 0) != accepted) { printf("FAIL %s: %s (%s)\n", name, accepted ? "refused" : "accepted", out.error); return 1; }
    printf("ok   %s\n", name);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    int bad = 0;
    const uint64_t M2 = UINT64_C(1) << 21;
    /* the same layout as ld -z max-page-size=0x200000 gives a 0x14-byte LOAD: offset 2 MiB, align 2 MiB */
    bad += check(argv[1], "dense_2m_align", (Ph){1, 5, M2, M2, M2, 0x14, 0x14, M2}, 1);
    bad += check(argv[1], "dense_16k_align", (Ph){1, 5, 0x4000, 0x4000, 0x4000, 0x14, 0x14, 0x4000}, 1);
    /* the padding is bounded by the alignment: an offset past size + one page is not justified by the file */
    bad += check(argv[1], "offset_past_align", (Ph){1, 5, 3 * M2, 3 * M2, 3 * M2, 0x14, 0x14, 0x1000}, 0);
    bad += check(argv[1], "big_offset", (Ph){1, 5, (UINT64_C(1) << 31) - 16, 0, 0, 16, 16, 0x1000}, 0);
    bad += check(argv[1], "big_offset_huge_align", (Ph){1, 5, (UINT64_C(1) << 31) - 16, 0, 0, 16, 16, UINT64_C(1) << 40}, 0);
    bad += check(argv[1], "offset_wrap", (Ph){1, 5, UINT64_C(0xFFFFFFFFFFFFFFF0), 0, 0, 0x20, 0x20, 0x1000}, 0);
    /* many headers do not add up to a justification: padding counts only for backed headers, at most 64 MiB in all */
    bad += check_many(argv[1], "many_unbacked", 1024, M2, (UINT64_C(1) << 31) - 16, 0, 0, 0);
    bad += check_many(argv[1], "many_backed", 1024, M2, (UINT64_C(1) << 31) - 16, 1, 0, 0);
    bad += check_many(argv[1], "many_400m", 200, M2, UINT64_C(400) << 20, 0, 0, 0);
    bad += check_many(argv[1], "many_align0", 60000, 0, UINT64_C(234) << 20, 0, 0, 0);
    bad += check_many(argv[1], "many_backed_cap", 1024, M2, UINT64_C(100) << 20, 1, 0, 0);
    /* four 2 MiB-aligned segments 2 MiB apart (a SELF of ~1 KB with a 6 MiB ELF): accepted; the mutation pad_sum = 0 refuses it */
    bad += check_many(argv[1], "ladder_4x2m", 3, M2, 3 * M2, 1, M2, 1);
    /* 8 headers of 2 MiB alignment without a SELF segment count for nothing: a segment at 10 MiB is refused */
    bad += check_many(argv[1], "unbacked_8x2m", 8, M2, UINT64_C(10) << 20, 0, 0, 0);
    /* the exact edge: 32 backed headers (64 MiB of padding), the far segment ends at N + 66 MiB / one byte later */
    {
        const uint64_t n = 32, total = 32 + 32 * n + 64 + (n + 1) * 56 + 16, edge = total + (66u << 20);
        bad += check_many(argv[1], "edge_ok", 32, M2, edge - 16, 1, 0, 1);
        bad += check_many(argv[1], "edge_plus1", 32, M2, edge - 15, 1, 0, 0);
        bad += check_many(argv[1], "unbacked_32x2m_edge", 32, M2, (UINT64_C(66) << 20) - 16, 0, 0, 0);
        /* fewer than 32 backed headers (the 64 MiB cap not reached): N + 17 * 2 MiB */
        const uint64_t n16 = 16, total16 = 32 + 32 * n16 + 64 + (n16 + 1) * 56 + 16, edge16 = total16 + ((n16 + 1) << 21);
        bad += check_many(argv[1], "edge16_ok", 16, M2, edge16 - 16, 1, 0, 1);
        bad += check_many(argv[1], "edge16_plus1", 16, M2, edge16 - 15, 1, 0, 0);
        bad += check_many(argv[1], "noflag_8x2m", 8, M2, UINT64_C(10) << 20, 2, 0, 0);
    }
    /* no program headers at all: refused early */
    {
        char path[512];
        snprintf(path, sizeof path, "%s/phnum_zero.self", argv[1]);
        uint8_t f[160] = {0};
        memcpy(f, "O\x15=\x1d", 4);
        memcpy(f + 32, "\x7f" "ELF\x02\x01\x01", 7);  /* count 0, ELF header at 32, phnum 0 */
        const uint64_t phoff = 64;
        memcpy(f + 32 + 32, &phoff, 8);
        FILE* o = fopen(path, "wb");
        if (!o || fwrite(f, 1, sizeof f, o) != sizeof f) { printf("FAIL phnum_zero: cannot write\n"); bad++; }
        else {
            fclose(o);
            static VpModule pm;
            pm.name = "phnum_zero"; pm.link_base = UINT64_C(0x500000000); pm.size = 0x1000;
            vp_register_module(&pm);
            VpLoadedImage out;
            vp_load_module(path, &pm, 0, NULL, NULL, &out);
            if (strcmp(out.error, "bad SELF") != 0) { printf("FAIL phnum_zero: %s\n", out.error); bad++; } else printf("ok   phnum_zero\n");
        }
    }
    /* the helpers themselves */
    if (vp_self_end_limit(100, UINT64_C(1) << 40) != 100 + VP_SELF_MAX_PAD_SUM + VP_SELF_MAX_PAD) { printf("FAIL pad_sum cap\n"); bad++; }
    if (vp_self_end_limit(100, 0) > 100 + VP_SELF_MAX_PAD || vp_self_end_limit(UINT64_MAX, 0) != VP_SELF_MAX_ELF ||
        vp_self_end_limit(100, UINT64_MAX) != 100 + VP_SELF_MAX_PAD_SUM + VP_SELF_MAX_PAD) { printf("FAIL vp_self_end_limit\n"); bad++; }
    printf(bad ? "loader_self: %d failed\n" : "loader_self: ok\n", bad);
    return bad != 0;
}
