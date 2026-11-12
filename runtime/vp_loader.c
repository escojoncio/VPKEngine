/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 */
#include "vp_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <sys/mman.h>

#define VP_MAX_IMPORT_STUBS 4096
#define STUB_HASH_SIZE (2 * VP_MAX_IMPORT_STUBS) /* a power of two: at most half full */

/* Stub addresses are handed out process-wide: two modules loaded one after the other must not
 * get the same address for different imports (the second registration would replace the first's
 * native). */
static uint64_t vp_next_stub;

/* Exactly at `at`, and only on free memory: MAP_FIXED would replace whatever is mapped there (the
 * app's own memory, another module). The address is a hint (plus MAP_FIXED_NOREPLACE where there
 * is one); a mapping the kernel put elsewhere, or that an older kernel placed by ignoring that
 * flag, is given back. */
int vp_map_fixed(uint64_t at, uint64_t size) {
    const uint64_t page = 65536; /* covers 4K, 16K and 64K hosts */
    const uint64_t start = at & ~(page - 1), end = (at + size + page - 1) & ~(page - 1);
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_FIXED_NOREPLACE
    flags |= MAP_FIXED_NOREPLACE;
#endif
    void* p = mmap((void*)(uintptr_t)start, end - start, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (p == MAP_FAILED) return -1;
    if ((uint64_t)(uintptr_t)p != start) { munmap(p, end - start); return -1; }
    return 0;
}

/* Gives back what vp_map_fixed mapped (a load that failed after it). */
static void unmap_fixed(uint64_t at, uint64_t size) {
    const uint64_t page = 65536;
    const uint64_t start = at & ~(page - 1), end = (at + size + page - 1) & ~(page - 1);
    munmap((void*)(uintptr_t)start, end - start);
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

/* The module's imports by slot: an open-addressing table of index + 1 (0 is empty), so that each
 * relocation finds its import without a pass over all of them. The first of equal slots wins. */
static uint64_t slot_hash(uint64_t x) { x *= UINT64_C(0x9E3779B97F4A7C15); return x ^ (x >> 32); }

static uint32_t* build_slot_index(const VpModule* m, size_t* mask) {
    size_t cap = 16;
    while (cap < m->import_count * 2) cap <<= 1;
    uint32_t* t = calloc(cap, sizeof *t);
    if (!t) return NULL;
    *mask = cap - 1;
    for (size_t i = 0; i < m->import_count; ++i) {
        size_t h = slot_hash(m->imports[i].slot) & *mask;
        while (t[h] && m->imports[t[h] - 1].slot != m->imports[i].slot) h = (h + 1) & *mask;
        if (!t[h]) t[h] = (uint32_t)(i + 1);
    }
    return t;
}

static const VpImport* find_import(const VpModule* m, const uint32_t* index, size_t mask, uint64_t base, uint64_t slot) {
    const uint64_t key = m->relative ? slot - base : slot;
    for (size_t h = slot_hash(key) & mask; index[h]; h = (h + 1) & mask) {
        if (m->imports[index[h] - 1].slot == key) return &m->imports[index[h] - 1];
    }
    return NULL;
}

static size_t name_hash(const char* s) {
    uint64_t h = UINT64_C(1469598103934665603);
    for (; *s; ++s) h = (h ^ (unsigned char)*s) * UINT64_C(1099511628211);
    return (size_t)(h ^ (h >> 32));
}

/* The stubs handed out so far, by import name and native: the same import of another module (or of
 * this one loaded again) gets the stub it already has, so that the table of natives grows with the
 * distinct imports and not with the modules loaded. Never freed: a stub address stays valid. */
#define STUB_REGISTRY 16384 /* a power of two; filled to three quarters, then every import gets a stub of its own */
static struct { char* name; VpNative fn; uint64_t stub; } stub_registry[STUB_REGISTRY];
static size_t stub_registry_count;
static pthread_mutex_t stub_registry_lock = PTHREAD_MUTEX_INITIALIZER;

/* The stub for an import: the one the same (name, native) got before, else a new address. A new
 * stub is not registered here but by stub_commit, once the load cannot fail any more: a load that
 * fails (too many imports, the code is not the translated one) leaves nothing in the tables (it
 * only used up some addresses). */
static uint64_t stub_alloc(const char* name, VpNative fn) {
    uint64_t stub;
    pthread_mutex_lock(&stub_registry_lock);
    size_t h = (name_hash(name) ^ ((uintptr_t)fn >> 4)) & (STUB_REGISTRY - 1);
    while (stub_registry[h].name && (stub_registry[h].fn != fn || strcmp(stub_registry[h].name, name))) h = (h + 1) & (STUB_REGISTRY - 1);
    stub = stub_registry[h].name ? stub_registry[h].stub : VP_IMPORT_STUB_BASE + vp_next_stub++ * VP_IMPORT_STUB_STRIDE;
    pthread_mutex_unlock(&stub_registry_lock);
    return stub;
}

/* Makes the stub stub_alloc gave a module real: its native registered and, if it is new, in the
 * registry. When another load committed the same (name, native) first, under another address, this
 * module's own stub still gets its native. */
static void stub_commit(const char* name, VpNative fn, uint64_t stub) {
    pthread_mutex_lock(&stub_registry_lock);
    size_t h = (name_hash(name) ^ ((uintptr_t)fn >> 4)) & (STUB_REGISTRY - 1);
    while (stub_registry[h].name && (stub_registry[h].fn != fn || strcmp(stub_registry[h].name, name))) h = (h + 1) & (STUB_REGISTRY - 1);
    if (!stub_registry[h].name) {
        if (fn) vp_register_native(stub, fn);
        if (stub_registry_count < STUB_REGISTRY / 4 * 3 && (stub_registry[h].name = strdup(name))) {
            stub_registry[h].fn = fn;
            stub_registry[h].stub = stub;
            stub_registry_count++;
        }
    } else if (stub_registry[h].stub != stub && fn) {
        vp_register_native(stub, fn);
    }
    pthread_mutex_unlock(&stub_registry_lock);
}

int vp_load_image(const char* path, VpImportResolver resolve, void* user, VpLoadedImage* out) {
    return vp_load_module(path, NULL, 0, resolve, user, out);
}

int vp_load_image_at(const char* path, uint64_t load_at, VpImportResolver resolve, void* user, VpLoadedImage* out) {
    return vp_load_module(path, NULL, load_at, resolve, user, out);
}

/* Link-time address `a` where the image is now. */
#define AT(a) ((uint64_t)(a) + delta)

int vp_load_module(const char* path, VpModule* module, uint64_t load_at, VpImportResolver resolve, void* user, VpLoadedImage* out) {
    memset(out, 0, sizeof *out);
    if (!module) module = vp_first_module();
    if (!module) return fail(out, "no translated module is registered");
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
    Phdr* ph = malloc(sizeof(Phdr) * (phnum ? phnum : 1));
    if (!ph) { free(file); return fail(out, "out of memory"); }
    for (uint16_t i = 0; i < phnum; ++i) {
        memcpy(&ph[i], elf + phoff + i * 56, 56);
        if (ph[i].filesz > ph[i].memsz || ph[i].offset > elf_size || ph[i].filesz > elf_size - ph[i].offset) { free(ph); free(file); return fail(out, "bad segment"); }
        if ((ph[i].type == 1 || ph[i].type == 0x61000010) && ph[i].memsz > UINT64_MAX - ph[i].vaddr) { free(ph); free(file); return fail(out, "bad segment"); }
        if (ph[i].type == 1 || ph[i].type == 0x61000010) {
            if (ph[i].vaddr < lo) lo = ph[i].vaddr;
            if (ph[i].vaddr + ph[i].memsz > hi) hi = ph[i].vaddr + ph[i].memsz;
        }
        if (ph[i].type == 2) dyn = &ph[i];
        if (ph[i].type == 0x61000000) dynlib = &ph[i];
    }
    if (lo >= hi || hi - lo > (UINT64_C(1) << 31)) { free(ph); free(file); return fail(out, "no loadable segments, or the image is larger than 2 GB"); }
    /* Elsewhere than the link address only when the translation allows it (--pic). The mapping
     * granularity is 64 KB, so the new place keeps the image's offset within 64 KB. */
    const uint64_t delta = load_at ? (load_at & ~UINT64_C(0xffff)) + (lo & 0xffff) - lo : 0;
    if (delta && !module->relative) { free(ph); free(file); return fail(out, "the translation was not made with --pic: it can only run at its link address"); }
    /* Everything that can be checked before touching memory: the image must be the one the
     * module was translated from (link base, size). */
    if (lo != module->link_base || hi - lo != module->size) { free(ph); free(file); return fail(out, "this image is not the one the module was translated from (link base or size differs)"); }
    if (vp_map_fixed(AT(lo), hi - lo)) { free(ph); free(file); return fail(out, "cannot map the image"); }
    for (uint16_t i = 0; i < phnum; ++i) {
        if ((ph[i].type == 1 || ph[i].type == 0x61000010) && ph[i].filesz && ph[i].offset + ph[i].filesz <= elf_size) {
            memcpy((void*)(uintptr_t)AT(ph[i].vaddr), elf + ph[i].offset, ph[i].filesz);
        }
    }
    out->base = AT(lo); out->end = AT(hi); out->entry = AT(out->entry);


    /* Relocations. The image is loaded at its link address, so RELATIVE slots hold the addend;
     * the symbolic ones against imports get a stub address registered as the native. */
    /* The stubs the module's imports got (stub_fresh: made by this load, to be committed). */
    const char* stub_names[VP_MAX_IMPORT_STUBS]; uint64_t stub_values[VP_MAX_IMPORT_STUBS]; VpNative stub_natives[VP_MAX_IMPORT_STUBS];
    unsigned char stub_fresh[VP_MAX_IMPORT_STUBS]; size_t stubs = 0;
    if (dyn && dyn->filesz <= elf_size && dyn->offset <= elf_size - dyn->filesz) {
        const unsigned char* dyn_data = elf + dyn->offset;
        const uint64_t dyn_size = dyn->filesz;
        #define TAG(t) vp_dyn_tag(dyn_data, dyn_size, (t))
        const unsigned char* tables[2] = {NULL, NULL}; uint64_t sizes[2] = {0, 0};
        const unsigned char* symtab = NULL; uint64_t symtab_size = 0;
        if (dynlib) {
            const uint64_t so = TAG(0x61000039), ss = TAG(0x6100003f);
            if (so <= dynlib->filesz && ss <= dynlib->filesz - so) { symtab = elf + dynlib->offset + so; symtab_size = ss; }
        } else {
            const uint64_t so = TAG(6), to = TAG(5);
            if (so >= lo && so < hi) { symtab = (const unsigned char*)(uintptr_t)AT(so); symtab_size = (to > so && to <= hi) ? to - so : hi - so; }
        }
        if (dynlib) {
            const uint64_t rela = TAG(0x61000029), relasz = TAG(0x6100002d), jmprel = TAG(0x6100002f), pltsz = TAG(0x61000031);
            if (rela <= dynlib->filesz && relasz <= dynlib->filesz - rela) { tables[0] = elf + dynlib->offset + rela; sizes[0] = relasz; }
            if (jmprel <= dynlib->filesz && pltsz <= dynlib->filesz - jmprel) { tables[1] = elf + dynlib->offset + jmprel; sizes[1] = pltsz; }
        } else {
            const uint64_t rela = TAG(7), relasz = TAG(8), jmprel = TAG(23), pltsz = TAG(2);
            if (rela >= lo && rela <= hi && relasz <= hi - rela) { tables[0] = (const unsigned char*)(uintptr_t)AT(rela); sizes[0] = relasz; }
            if (jmprel >= lo && jmprel <= hi && pltsz <= hi - jmprel) { tables[1] = (const unsigned char*)(uintptr_t)AT(jmprel); sizes[1] = pltsz; }
        }
        #undef TAG
        /* One stub per imported symbol, whatever the number of relocations that name it. */
        uint16_t stub_hash[STUB_HASH_SIZE]; /* stub index + 1 by the hash of the name; 0 is empty */
        memset(stub_hash, 0, sizeof stub_hash);
        size_t slot_mask = 0;
        uint32_t* const slot_index = build_slot_index(module, &slot_mask);
        if (!slot_index) { free(ph); free(file); unmap_fixed(AT(lo), hi - lo); return fail(out, "out of memory"); }
        for (int t = 0; t < 2; ++t) {
            for (uint64_t pos = 0; tables[t] && pos + 24 <= sizes[t]; pos += 24) {
                uint64_t target, info; int64_t addend;
                memcpy(&target, tables[t] + pos, 8); memcpy(&info, tables[t] + pos + 8, 8); memcpy(&addend, tables[t] + pos + 16, 8);
                const uint32_t kind = (uint32_t)info;
                if (target < lo || target > hi || hi - target < 8) continue;
                void* const slot = (void*)(uintptr_t)AT(target);
                const uint32_t sym = (uint32_t)(info >> 32);
                if (kind == 8) { /* RELATIVE: base + addend */
                    const uint64_t value = AT(addend);
                    memcpy(slot, &value, 8);
                } else if (kind == 1 && sym == 0) { /* an absolute value */
                    memcpy(slot, &addend, 8);
                } else if (kind == 1 || kind == 6 || kind == 7) {
                    const VpImport* im = find_import(module, slot_index, slot_mask, AT(lo), AT(target));
                    if (!im) {
                        /* A defined symbol: its value (plus the addend) goes into the slot. */
                        if (symtab && (uint64_t)sym * 24 + 24 <= symtab_size) {
                            uint64_t value; uint16_t shndx;
                            memcpy(&value, symtab + (uint64_t)sym * 24 + 8, 8); memcpy(&shndx, symtab + (uint64_t)sym * 24 + 6, 2);
                            if (shndx) { value = AT(value) + ((kind == 1) ? (uint64_t)addend : 0); memcpy(slot, &value, 8); }
                        }
                        continue;
                    }
                    size_t hs = name_hash(im->name) & (STUB_HASH_SIZE - 1), k;
                    while (stub_hash[hs] && strcmp(stub_names[stub_hash[hs] - 1], im->name)) hs = (hs + 1) & (STUB_HASH_SIZE - 1);
                    k = stub_hash[hs] ? (size_t)stub_hash[hs] - 1 : stubs;
                    if (k == stubs) {
                        if (stubs == VP_MAX_IMPORT_STUBS) { free(slot_index); free(ph); free(file); unmap_fixed(AT(lo), hi - lo); return fail(out, "too many imports"); }
                        stub_hash[hs] = (uint16_t)(stubs + 1);
                        VpNative fn = NULL; uint64_t data = 0;
                        const int provided = resolve ? resolve(im->name, im->function, &fn, &data, user) : 0;
                        stub_names[k] = im->name;
                        stub_fresh[k] = 0;
                        if (im->function && provided && !fn && data) {
                            stub_values[k] = data; /* another guest module's function */
                        } else if (im->function) {
                            stub_natives[k] = provided ? fn : NULL;
                            stub_values[k] = stub_alloc(im->name, stub_natives[k]);
                            stub_fresh[k] = 1;
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
                    memcpy(slot, &value, 8);
                }
            }
        }
        free(slot_index);
    }
    free(ph);
    free(file);
    /* The code now in memory must be the code that was translated; only then does dispatch use
     * the module at its new place. */
    if (vp_fingerprint(AT(lo), module->code, module->code_count, module->reloc_sites, module->reloc_site_count) != module->fingerprint) {
        unmap_fixed(AT(lo), hi - lo);
        return fail(out, "the code of this image differs from the translation (fingerprint)");
    }
    for (size_t k = 0; k < stubs; ++k) {
        if (stub_fresh[k]) stub_commit(stub_names[k], stub_natives[k], stub_values[k]);
    }
    vp_module_set_base(module, AT(lo));
    return 0;
}
