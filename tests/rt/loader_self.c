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
    VpModule m;
    memset(&m, 0, sizeof m);
    m.name = name; m.link_base = UINT64_C(0x500000000); m.size = 0x1000;
    vp_register_module(&m);
    VpLoadedImage out;
    vp_load_module(path, &m, 0, NULL, NULL, &out);
    const int refused = strcmp(out.error, "bad SELF") == 0;
    if (refused == accepted) { printf("FAIL %s: %s (%s)\n", name, refused ? "refused" : "accepted", out.error); return 1; }
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
    /* the helpers themselves */
    if (vp_self_end_limit(100, 0) > 100 + VP_SELF_MAX_PAD || vp_self_end_limit(UINT64_MAX, 0) != VP_SELF_MAX_ELF ||
        vp_self_end_limit(100, UINT64_MAX) != VP_SELF_MAX_ELF) { printf("FAIL vp_self_end_limit\n"); bad++; }
    printf(bad ? "loader_self: %d failed\n" : "loader_self: ok\n", bad);
    return bad != 0;
}
