/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 */
#include "vp_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define VP_MAX_IMPORT_STUBS 4096

int vp_map_fixed(uint64_t at, uint64_t size) {
    const uint64_t page = 65536; /* covers 4K, 16K and 64K hosts */
    const uint64_t start = at & ~(page - 1), end = (at + size + page - 1) & ~(page - 1);
    void* p = mmap((void*)(uintptr_t)start, end - start, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    return p == MAP_FAILED ? -1 : 0;
}

static int fail(VpLoadedImage* out, const char* what) {
    snprintf(out->error, sizeof out->error, "%s", what);
    return -1;
}

typedef struct { uint32_t type, flags; uint64_t offset, vaddr, paddr, filesz, memsz, align; } Phdr;

/* PS4 SELF -> the ELF it wraps, with the segment data at the program headers' offsets. */
static unsigned char* unwrap_self(const unsigned char* self, size_t size, size_t* out_size) {
    if (size < 32) return NULL;
    uint16_t count; memcpy(&count, self + 24, 2);
    const size_t base = 32 + (size_t)count * 32;
    if (base + 64 > size || memcmp(self + base, "\x7f" "ELF", 4) != 0) return NULL;
    uint64_t phoff; uint16_t phnum;
    memcpy(&phoff, self + base + 32, 8); memcpy(&phnum, self + base + 56, 2);
    if (phoff > size - base || (uint64_t)phnum * 56 > size - base - phoff) return NULL;
    const size_t header_end = (size_t)phoff + (size_t)phnum * 56;
    size_t end = header_end;
    for (uint16_t i = 0; i < phnum; ++i) {
        Phdr p; memcpy(&p, self + base + phoff + i * 56, 56);
        if (p.filesz > (UINT64_C(1) << 31) || p.offset > (UINT64_C(1) << 31)) return NULL;
        if (p.offset + p.filesz > end) end = p.offset + p.filesz;
    }
    unsigned char* elf = calloc(1, end);
    if (!elf) return NULL;
    memcpy(elf, self + base, header_end);
    for (uint16_t i = 0; i < count; ++i) {
        uint64_t flags, off, sz; memcpy(&flags, self + 32 + i * 32, 8); memcpy(&off, self + 40 + i * 32, 8); memcpy(&sz, self + 48 + i * 32, 8);
        if (!(flags & 0x800)) continue;
        if (flags & 10) { free(elf); return NULL; } /* encrypted or compressed: not a usable dump */
        const size_t index = (flags >> 20) & 4095;
        if (index >= phnum) { free(elf); return NULL; }
        Phdr p; memcpy(&p, self + base + phoff + index * 56, 56);
        if (sz != p.filesz || sz > size || off > size - sz || p.offset + sz > end) { free(elf); return NULL; }
        memcpy(elf + p.offset, self + off, sz);
    }
    *out_size = end;
    return elf;
}

static uint64_t vp_dyn_tag(const unsigned char* d, uint64_t size, uint64_t wanted) {
    for (uint64_t pos = 0; pos + 16 <= size; pos += 16) {
        uint64_t tag, val; memcpy(&tag, d + pos, 8); memcpy(&val, d + pos + 8, 8);
        if (!tag) break;
        if (tag == wanted) return val;
    }
    return 0;
}

static const VpImport* find_import(uint64_t slot) {
    for (size_t i = 0; i < vp_import_count; ++i) if (vp_imports[i].slot == slot) return &vp_imports[i];
    return NULL;
}

int vp_load_image(const char* path, VpImportResolver resolve, void* user, VpLoadedImage* out) {
    memset(out, 0, sizeof *out);
    FILE* f = fopen(path, "rb");
    if (!f) return fail(out, "cannot open the image");
    fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char* file = malloc(fsize);
    if (!file || fread(file, 1, fsize, f) != (size_t)fsize) { fclose(f); free(file); return fail(out, "cannot read the image"); }
    fclose(f);
    unsigned char* elf = file;
    size_t elf_size = (size_t)fsize;
    if (fsize >= 4 && memcmp(file, "O\x15=\x1d", 4) == 0) {
        elf = unwrap_self(file, fsize, &elf_size);
        free(file);
        if (!elf) return fail(out, "bad SELF");
        file = elf;
    }
    if (elf_size < 64 || memcmp(elf, "\x7f" "ELF\x02\x01", 6) != 0) { free(file); return fail(out, "not an x86-64 ELF"); }
    uint64_t phoff; uint16_t phnum;
    memcpy(&out->entry, elf + 24, 8); memcpy(&phoff, elf + 32, 8); memcpy(&phnum, elf + 56, 2);
    uint64_t lo = UINT64_MAX, hi = 0;
    const Phdr* dyn = NULL; const Phdr* dynlib = NULL;
    if (phoff > elf_size || (uint64_t)phnum * 56 > elf_size - phoff) { free(file); return fail(out, "bad program headers"); }
    Phdr* ph = malloc(sizeof(Phdr) * phnum);
    for (uint16_t i = 0; i < phnum; ++i) {
        memcpy(&ph[i], elf + phoff + i * 56, 56);
        if (ph[i].filesz > ph[i].memsz || ph[i].offset > elf_size || ph[i].filesz > elf_size - ph[i].offset) { free(ph); free(file); return fail(out, "bad segment"); }
        if (ph[i].type == 1 || ph[i].type == 0x61000010) {
            if (ph[i].vaddr < lo) lo = ph[i].vaddr;
            if (ph[i].vaddr + ph[i].memsz > hi) hi = ph[i].vaddr + ph[i].memsz;
        }
        if (ph[i].type == 2) dyn = &ph[i];
        if (ph[i].type == 0x61000000) dynlib = &ph[i];
    }
    if (lo >= hi || hi - lo > (UINT64_C(1) << 31)) { free(ph); free(file); return fail(out, "no loadable segments, or the image is larger than 2 GB"); }
    if (vp_map_fixed(lo, hi - lo)) { free(ph); free(file); return fail(out, "cannot map the image at its addresses"); }
    for (uint16_t i = 0; i < phnum; ++i) {
        if ((ph[i].type == 1 || ph[i].type == 0x61000010) && ph[i].filesz && ph[i].offset + ph[i].filesz <= elf_size) {
            memcpy((void*)(uintptr_t)ph[i].vaddr, elf + ph[i].offset, ph[i].filesz);
        }
    }
    out->base = lo; out->end = hi;

    /* Relocations. The image is loaded at its link address, so RELATIVE slots hold the addend;
     * the symbolic ones against imports get a stub address registered as the native. */
    if (dyn && dyn->filesz <= elf_size && dyn->offset <= elf_size - dyn->filesz) {
        const unsigned char* dyn_data = elf + dyn->offset;
        const uint64_t dyn_size = dyn->filesz;
        #define TAG(t) vp_dyn_tag(dyn_data, dyn_size, (t))
        const unsigned char* tables[2] = {NULL, NULL}; uint64_t sizes[2] = {0, 0};
        const unsigned char* symtab = NULL; uint64_t symtab_size = 0;
        if (dynlib) {
            const uint64_t so = TAG(0x61000039), ss = TAG(0x6100003f);
            if (so + ss <= dynlib->filesz) { symtab = elf + dynlib->offset + so; symtab_size = ss; }
        } else {
            const uint64_t so = TAG(6), to = TAG(5);
            if (so >= lo && so < hi) { symtab = (const unsigned char*)(uintptr_t)so; symtab_size = (to > so && to <= hi) ? to - so : hi - so; }
        }
        if (dynlib) {
            const uint64_t rela = TAG(0x61000029), relasz = TAG(0x6100002d), jmprel = TAG(0x6100002f), pltsz = TAG(0x61000031);
            if (rela + relasz <= dynlib->filesz) { tables[0] = elf + dynlib->offset + rela; sizes[0] = relasz; }
            if (jmprel + pltsz <= dynlib->filesz) { tables[1] = elf + dynlib->offset + jmprel; sizes[1] = pltsz; }
        } else {
            const uint64_t rela = TAG(7), relasz = TAG(8), jmprel = TAG(23), pltsz = TAG(2);
            if (rela >= lo && rela + relasz <= hi) { tables[0] = (const unsigned char*)(uintptr_t)rela; sizes[0] = relasz; }
            if (jmprel >= lo && jmprel + pltsz <= hi) { tables[1] = (const unsigned char*)(uintptr_t)jmprel; sizes[1] = pltsz; }
        }
        #undef TAG
        /* One stub per imported symbol, whatever the number of relocations that name it. */
        const char* stub_names[VP_MAX_IMPORT_STUBS]; uint64_t stub_values[VP_MAX_IMPORT_STUBS]; size_t stubs = 0;
        for (int t = 0; t < 2; ++t) {
            for (uint64_t pos = 0; tables[t] && pos + 24 <= sizes[t]; pos += 24) {
                uint64_t target, info; int64_t addend;
                memcpy(&target, tables[t] + pos, 8); memcpy(&info, tables[t] + pos + 8, 8); memcpy(&addend, tables[t] + pos + 16, 8);
                const uint32_t kind = (uint32_t)info;
                if (target < lo || target + 8 > hi) continue;
                const uint32_t sym = (uint32_t)(info >> 32);
                if (kind == 8 || (kind == 1 && sym == 0)) { /* RELATIVE, or an absolute 64-bit value: the link address is the load address */
                    memcpy((void*)(uintptr_t)target, &addend, 8);
                } else if (kind == 1 || kind == 6 || kind == 7) {
                    const VpImport* im = find_import(target);
                    if (!im) {
                        /* A defined symbol: its value (plus the addend) goes into the slot. */
                        if (symtab && (uint64_t)sym * 24 + 24 <= symtab_size) {
                            uint64_t value; uint16_t shndx;
                            memcpy(&value, symtab + (uint64_t)sym * 24 + 8, 8); memcpy(&shndx, symtab + (uint64_t)sym * 24 + 6, 2);
                            if (shndx) { value += (kind == 1) ? (uint64_t)addend : 0; memcpy((void*)(uintptr_t)target, &value, 8); }
                        }
                        continue;
                    }
                    size_t k;
                    for (k = 0; k < stubs; ++k) if (!strcmp(stub_names[k], im->name)) break;
                    if (k == stubs) {
                        if (stubs == VP_MAX_IMPORT_STUBS) { free(ph); free(file); return fail(out, "too many imports"); }
                        VpNative fn = NULL; uint64_t data = 0;
                        const int provided = resolve ? resolve(im->name, im->function, &fn, &data, user) : 0;
                        stub_names[k] = im->name;
                        if (im->function) {
                            stub_values[k] = VP_IMPORT_STUB_BASE + (uint64_t)k * VP_IMPORT_STUB_STRIDE;
                            if (provided && fn) vp_register_native(stub_values[k], fn);
                        } else {
                            stub_values[k] = provided ? data : 0;
                        }
                        if (provided) out->imports_resolved++;
                        else {
                            if (out->imports_missing < 64) out->missing[out->imports_missing] = im->name;
                            out->imports_missing++;
                        }
                        stubs++;
                    }
                    uint64_t value = stub_values[k] + (kind == 1 ? (uint64_t)addend : 0);
                    memcpy((void*)(uintptr_t)target, &value, 8);
                }
            }
        }
    }
    free(ph);
    free(file);
    return 0;
}
