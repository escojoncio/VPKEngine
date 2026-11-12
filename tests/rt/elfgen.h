/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * Hand-made x86-64 ELF images for the loader tests: one 4 KB LOAD segment at `base` (the file is
 * the segment), a relocation table at +0x400 and a PT_DYNAMIC at +0x800 that points at it.
 */
#ifndef VP_TEST_ELFGEN_H
#define VP_TEST_ELFGEN_H
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct { uint32_t type, flags; uint64_t offset, vaddr, paddr, filesz, memsz, align; } TestPhdr;
typedef struct { uint64_t target, info; int64_t addend; } TestRela;

#define TEST_IMG 0x1000u

/* extra_phdrs: further program headers (may be NULL). rela_size: DT_RELASZ to write. */
static inline int elfgen(const char* path, uint64_t base, const TestRela* rela, unsigned nrela, uint64_t rela_size,
                         const TestPhdr* extra, unsigned nextra) {
    uint8_t f[TEST_IMG];
    memset(f, 0, sizeof f);
    memcpy(f, "\x7f" "ELF\x02\x01\x01", 7);
    const uint16_t type = 3, machine = 62, phsize = 56, phnum = (uint16_t)(2 + nextra);
    const uint64_t phoff = 64, entry = base + 0x100;
    memcpy(f + 16, &type, 2); memcpy(f + 18, &machine, 2);
    memcpy(f + 24, &entry, 8); memcpy(f + 32, &phoff, 8);
    memcpy(f + 54, &phsize, 2); memcpy(f + 56, &phnum, 2);
    TestPhdr ph[8];
    memset(ph, 0, sizeof ph);
    ph[0] = (TestPhdr){1, 7, 0, base, base, TEST_IMG, TEST_IMG, 0x1000};
    ph[1] = (TestPhdr){2, 6, 0x800, base + 0x800, base + 0x800, 0x40, 0x40, 8};
    for (unsigned i = 0; i < nextra; ++i) ph[2 + i] = extra[i];
    memcpy(f + 64, ph, sizeof(TestPhdr) * phnum);
    if (nrela) memcpy(f + 0x400, rela, sizeof(TestRela) * nrela);
    const uint64_t dyn[6] = {7, base + 0x400, 8, rela_size, 0, 0};
    memcpy(f + 0x800, dyn, sizeof dyn);
    FILE* o = fopen(path, "wb");
    if (!o) return -1;
    const int ok = fwrite(f, 1, sizeof f, o) == sizeof f;
    return (fclose(o) == 0 && ok) ? 0 : -1;
}
#endif
