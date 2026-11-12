/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 */
#include "vp_loader.h"
#include "vp_self.h"

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

/* The ranges the loader mapped itself, so that it can replace and release them (and only them):
 * an image's own (owner: its VpModule), those vp_map_fixed made (OWNER_PLAIN) and the ones an
 * embedder committed with vp_map_fixed_over (OWNER_OVER). All 64 KB aligned and disjoint. */
typedef struct { uint64_t start, end; const void* owner; } Mapped;
static Mapped* mapped;
static size_t mapped_count, mapped_cap;
static pthread_mutex_t mapped_lock = PTHREAD_MUTEX_INITIALIZER;
static const char owner_plain_tag, owner_over_tag;
#define OWNER_PLAIN ((const void*)&owner_plain_tag)
#define OWNER_OVER ((const void*)&owner_over_tag)

#define MAP_PAGE UINT64_C(65536) /* covers 4K, 16K and 64K hosts */
static uint64_t round_down(uint64_t a) { return a & ~(MAP_PAGE - 1); }
static uint64_t round_up(uint64_t a) { return (a + MAP_PAGE - 1) & ~(MAP_PAGE - 1); }

/* All of these run with mapped_lock held. */
static int mapped_reserve(size_t extra) {
    if (mapped_count + extra <= mapped_cap) return 0;
    size_t cap = mapped_cap ? mapped_cap * 2 : 16;
    while (cap < mapped_count + extra) cap *= 2;
    Mapped* m = realloc(mapped, cap * sizeof *m);
    if (!m) return -1;
    mapped = m;
    mapped_cap = cap;
    return 0;
}

/* Forgets [start,end) (needs room for one more entry: it may split one). A module that loses its
 * image this way is detached: its code is no longer there. */
static void mapped_forget(uint64_t start, uint64_t end) {
    for (size_t i = 0; i < mapped_count;) {
        Mapped* e = &mapped[i];
        if (e->end <= start || e->start >= end) { ++i; continue; }
        if (e->owner != OWNER_PLAIN && e->owner != OWNER_OVER) vp_detach_module((VpModule*)(uintptr_t)e->owner);
        if (e->start < start && e->end > end) { /* the middle goes: two entries remain */
            mapped[mapped_count++] = (Mapped){end, e->end, e->owner};
            e->end = start;
            ++i;
        } else if (e->start < start) { e->end = start; ++i; }
        else if (e->end > end) { e->start = end; ++i; }
        else { *e = mapped[--mapped_count]; }
    }
}

/* Whether the entry may be replaced when `owner` maps: an image replaces its own earlier image and
 * the ranges an embedder prepared for it (OWNER_OVER); vp_map_fixed (OWNER_PLAIN) replaces nothing,
 * not even a prepared range; nothing replaces anything else. */
static int replaceable(const Mapped* e, const void* owner) {
    return owner != OWNER_PLAIN && owner != OWNER_OVER && (e->owner == OWNER_OVER || e->owner == owner);
}

/* Rounds [at,at+size) out to 64 KB blocks; -1 for an empty range or one that wraps around. */
static int range_of(uint64_t at, uint64_t size, uint64_t* start, uint64_t* end) {
    if (size == 0 || size > UINT64_MAX - at || at + size > UINT64_MAX - (MAP_PAGE - 1)) return -1;
    *start = round_down(at);
    *end = round_up(at + size);
    return 0;
}

/* Where prepared (vp_map_fixed_over) ranges were replaced by an image: a failed load or reload
 * gives those parts back as prepared, instead of leaving a hole. Sorted by nothing, disjoint. All
 * under mapped_lock. Losing an entry (out of memory) only costs that restoration. */
typedef struct { uint64_t start, end; } Span;
static Span* origin;
static size_t origin_count, origin_cap;

static void origin_sub(uint64_t start, uint64_t end) {
    for (size_t i = 0; i < origin_count;) {
        Span* o = &origin[i];
        if (o->end <= start || o->start >= end) { ++i; continue; }
        if (o->start < start && o->end > end) {
            if (origin_count < origin_cap) origin[origin_count++] = (Span){end, o->end};
            o->end = start;
            ++i;
        } else if (o->start < start) { o->end = start; ++i; }
        else if (o->end > end) { o->start = end; ++i; }
        else { *o = origin[--origin_count]; }
    }
}

static void origin_add(uint64_t start, uint64_t end) {
    origin_sub(start, end);
    if (origin_count + 2 > origin_cap) {
        const size_t cap = origin_cap ? origin_cap * 2 : 8;
        Span* n = realloc(origin, cap * sizeof *n);
        if (!n) return;
        origin = n;
        origin_cap = cap;
    }
    origin[origin_count++] = (Span){start, end};
}

/* The replaceable entry's part of [from,end) that starts first. */
static int next_replaceable(const void* owner, uint64_t from, uint64_t end, uint64_t* s, uint64_t* e, const void** who) {
    int found = 0;
    for (size_t i = 0; i < mapped_count; ++i) {
        if (!replaceable(&mapped[i], owner)) continue;
        const uint64_t lo = mapped[i].start > from ? mapped[i].start : from, hi = mapped[i].end < end ? mapped[i].end : end;
        if (lo >= hi || (found && lo >= *s)) continue;
        *s = lo; *e = hi; *who = mapped[i].owner; found = 1;
    }
    return found;
}

/* Only on free memory: the address is a hint (plus MAP_FIXED_NOREPLACE where there is one); a
 * mapping the kernel put elsewhere, or that an older kernel placed by ignoring that flag, is given
 * back. */
static int map_free(uint64_t start, uint64_t end) {
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_FIXED_NOREPLACE
    flags |= MAP_FIXED_NOREPLACE;
#endif
    void* p = mmap((void*)(uintptr_t)start, end - start, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (p == MAP_FAILED) return -1;
    /* Without MAP_FIXED_NOREPLACE (Darwin, old kernels) the address is only a hint: whatever the
     * kernel returned other than exactly `start` is not what was asked for, and goes back. */
    if ((uint64_t)(uintptr_t)p != start) { munmap(p, end - start); return -1; }
    return 0;
}

/* The parts of [start,end) no replaceable entry covers: mapped (free memory only) when `map`, else
 * given back up to `stop`. Mapping stops at the first part that cannot be, which is returned in
 * *failed_at (UINT64_MAX when none). */
static void gaps(const void* owner, uint64_t start, uint64_t end, int map, uint64_t stop, uint64_t* failed_at) {
    uint64_t cursor = start, s, e;
    const void* who;
    *failed_at = UINT64_MAX;
    while (cursor < end && (map || cursor < stop)) {
        const int more = next_replaceable(owner, cursor, end, &s, &e, &who);
        const uint64_t gap_end = more ? s : end;
        if (gap_end > cursor) {
            if (map) { if (map_free(cursor, gap_end)) { *failed_at = cursor; return; } }
            else munmap((void*)(uintptr_t)cursor, (gap_end < stop ? gap_end : stop) - cursor);
        }
        cursor = more ? e : end;
    }
}

/* Maps [start,end) read/write for `owner` (a VpModule, OWNER_PLAIN or OWNER_OVER) and records it.
 * Free memory is mapped without replacing anything; what the owner may replace (see `replaceable`)
 * is replaced. Nothing is mapped on failure, except that a replaced range stays replaced. */
static int map_range(uint64_t start, uint64_t end, const void* owner) {
    int rc = -1;
    if (start >= end) return -1;
    pthread_mutex_lock(&mapped_lock);
    if (mapped_reserve(2)) goto out;
    uint64_t failed;
    gaps(owner, start, end, 1, 0, &failed);
    if (failed != UINT64_MAX) { gaps(owner, start, end, 0, failed, &failed); goto out; }
    uint64_t cursor = start, s, e;
    const void* who;
    while (cursor < end && next_replaceable(owner, cursor, end, &s, &e, &who)) {
        if (who != OWNER_OVER) vp_detach_module((VpModule*)(uintptr_t)who); /* its image is about to go */
        else if (owner != OWNER_PLAIN) origin_add(s, e); /* a failed load puts it back as prepared */
        if (mmap((void*)(uintptr_t)s, e - s, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == MAP_FAILED) {
            gaps(owner, start, end, 0, UINT64_MAX, &failed); /* the free parts are given back */
            goto out;
        }
        cursor = e;
    }
    mapped_forget(start, end);
    mapped[mapped_count++] = (Mapped){start, end, owner};
    if (owner != OWNER_PLAIN && owner != OWNER_OVER) { /* a module has one image: an earlier one elsewhere goes */
        for (size_t i = 0; i < mapped_count;) {
            if (mapped[i].owner == owner && mapped[i].start != start) {
                vp_detach_module((VpModule*)(uintptr_t)owner);
                munmap((void*)(uintptr_t)mapped[i].start, mapped[i].end - mapped[i].start);
                origin_sub(mapped[i].start, mapped[i].end);
                mapped[i] = mapped[--mapped_count];
            } else ++i;
        }
    }
    rc = 0;
out:
    pthread_mutex_unlock(&mapped_lock);
    return rc;
}

int vp_map_fixed(uint64_t at, uint64_t size) {
    uint64_t start, end;
    if (range_of(at, size, &start, &end)) return -1;
    return map_range(start, end, OWNER_PLAIN);
}

/* Maps over whatever is there (an embedder's reservation), unless it is an image or a
 * vp_map_fixed mapping: those are refused. The range is recorded as the embedder's. */
int vp_map_fixed_over(uint64_t at, uint64_t size) {
    uint64_t start, end;
    if (range_of(at, size, &start, &end)) return -1;
    int rc = -1;
    pthread_mutex_lock(&mapped_lock);
    if (mapped_reserve(2)) goto out;
    for (size_t i = 0; i < mapped_count; ++i) {
        if (mapped[i].owner != OWNER_OVER && mapped[i].start < end && mapped[i].end > start) goto out;
    }
    if (mmap((void*)(uintptr_t)start, end - start, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == MAP_FAILED) goto out;
    mapped_forget(start, end);
    mapped[mapped_count++] = (Mapped){start, end, OWNER_OVER};
    rc = 0;
out:
    pthread_mutex_unlock(&mapped_lock);
    return rc;
}

/* With mapped_lock held and room for one more entry: unmaps [start,end) if all of it is the
 * loader's. */
static int unmap_locked(uint64_t start, uint64_t end) {
    for (uint64_t cursor = start; cursor < end;) { /* all of it must be the loader's */
        uint64_t next = cursor;
        for (size_t i = 0; i < mapped_count; ++i) if (mapped[i].start <= cursor && cursor < mapped[i].end) next = mapped[i].end;
        if (next == cursor) return -1;
        cursor = next;
    }
    munmap((void*)(uintptr_t)start, end - start);
    mapped_forget(start, end);
    origin_sub(start, end);
    return 0;
}

int vp_unmap_fixed(uint64_t at, uint64_t size) {
    uint64_t start, end;
    if (range_of(at, size, &start, &end)) return -1;
    int rc = -1;
    pthread_mutex_lock(&mapped_lock);
    if (mapped_reserve(1)) goto out;
    rc = unmap_locked(start, end);
out:
    pthread_mutex_unlock(&mapped_lock);
    return rc;
}

/* A load that failed after mapping: the image goes, and what an embedder had prepared there goes
 * back to being prepared (registered as vp_map_fixed_over, but PROT_NONE: nothing committed), so
 * that there is no hole and the load can be retried. One lock hold, so nothing can map in between.
 * Best effort for the restoration (memory that is no longer free is left alone). */
static void abandon_image(uint64_t start, uint64_t end) {
    pthread_mutex_lock(&mapped_lock);
    size_t n = 0;
    for (size_t i = 0; i < origin_count; ++i) if (origin[i].start < end && origin[i].end > start) ++n;
    if (mapped_reserve(1 + n)) {
        n = 0;
        if (mapped_reserve(1)) { pthread_mutex_unlock(&mapped_lock); return; }
    }
    Span* sp = n ? malloc(n * sizeof *sp) : NULL;
    if (!sp) n = 0;
    size_t k = 0;
    for (size_t i = 0; i < origin_count && k < n; ++i) {
        if (origin[i].start < end && origin[i].end > start) sp[k++] = (Span){origin[i].start > start ? origin[i].start : start, origin[i].end < end ? origin[i].end : end};
    }
    if (unmap_locked(start, end) == 0) {
        for (size_t i = 0; i < k; ++i) {
            int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
            flags |= MAP_NORESERVE;
#endif
#ifdef MAP_FIXED_NOREPLACE
            flags |= MAP_FIXED_NOREPLACE;
#endif
            void* p = mmap((void*)(uintptr_t)sp[i].start, sp[i].end - sp[i].start, PROT_NONE, flags, -1, 0);
            if (p == MAP_FAILED) continue;
            if ((uint64_t)(uintptr_t)p != sp[i].start) { munmap(p, sp[i].end - sp[i].start); continue; }
            mapped[mapped_count++] = (Mapped){sp[i].start, sp[i].end, OWNER_OVER};
            origin_add(sp[i].start, sp[i].end);
        }
    }
    free(sp);
    pthread_mutex_unlock(&mapped_lock);
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
    if (phnum == 0) return NULL; /* no program headers: nothing to rebuild (and calloc(1, 0) would be unspecified) */
    const size_t header_end = (size_t)phoff + (size_t)phnum * 56;
    size_t end = header_end;
    /* The same limit as tools/vpaot/image.cpp (vp_self.h): the end of a segment is in proportion to the file. */
    unsigned char* backed = calloc(phnum, 1); /* program headers whose data a SELF segment carries, counted once */
    if (!backed) return NULL;
    for (uint16_t i = 0; i < count; ++i) {
        uint64_t flags; memcpy(&flags, self + 32 + i * 32, 8);
        const size_t index = (flags >> 20) & 4095;
        if ((flags & 0x800) && index < phnum) backed[index] = 1;
    }
    uint64_t pad_sum = 0;
    for (uint16_t i = 0; i < phnum; ++i) {
        Phdr p; memcpy(&p, self + base + phoff + i * 56, 56);
        if (backed[i] && p.filesz) pad_sum += vp_self_pad(p.align);
    }
    free(backed);
    const uint64_t cap = vp_self_end_limit(size, pad_sum);
    for (uint16_t i = 0; i < phnum; ++i) {
        Phdr p; memcpy(&p, self + base + phoff + i * 56, 56);
        if (p.filesz > size || p.filesz > cap || p.offset > cap - p.filesz) return NULL;
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

/* What one load keeps about the stubs of its imports (fresh: made by this load, to be committed). */
typedef struct {
    const char* names[VP_MAX_IMPORT_STUBS];
    uint64_t values[VP_MAX_IMPORT_STUBS];
    VpNative natives[VP_MAX_IMPORT_STUBS];
    unsigned char fresh[VP_MAX_IMPORT_STUBS];
    uint16_t hash[STUB_HASH_SIZE]; /* stub index + 1 by the hash of the name; 0 is empty */
} StubSet;

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
 * distinct imports and not with the modules loaded. Never freed: a stub address stays valid. An
 * open-addressing table that doubles when it is three quarters full. */
typedef struct { char* name; VpNative fn; uint64_t stub; } StubEntry;
static StubEntry* stub_registry;
static size_t stub_registry_cap, stub_registry_count; /* cap: 0 or a power of two */
static pthread_mutex_t stub_registry_lock = PTHREAD_MUTEX_INITIALIZER;

static size_t stub_slot(const StubEntry* table, size_t cap, const char* name, VpNative fn) {
    size_t h = (name_hash(name) ^ ((uintptr_t)fn >> 4)) & (cap - 1);
    while (table[h].name && (table[h].fn != fn || strcmp(table[h].name, name))) h = (h + 1) & (cap - 1);
    return h;
}

/* Room for one more entry; 0 when there is none (the stub then simply is not shared). */
static int stub_registry_room(void) {
    if ((stub_registry_count + 1) * 4 <= stub_registry_cap * 3) return 1;
    const size_t cap = stub_registry_cap ? stub_registry_cap * 2 : 1024;
    StubEntry* table = calloc(cap, sizeof *table);
    if (!table) return 0;
    for (size_t i = 0; i < stub_registry_cap; ++i) {
        if (stub_registry[i].name) table[stub_slot(table, cap, stub_registry[i].name, stub_registry[i].fn)] = stub_registry[i];
    }
    free(stub_registry);
    stub_registry = table;
    stub_registry_cap = cap;
    return 1;
}

/* The stub for an import: the one the same (name, native) got before, else a new address. A new
 * stub is not registered here but by stub_commit, once the load cannot fail any more: a load that
 * fails (too many imports, the code is not the translated one) leaves nothing in the tables (it
 * only used up some addresses). */
static uint64_t stub_alloc(const char* name, VpNative fn) {
    uint64_t stub;
    pthread_mutex_lock(&stub_registry_lock);
    const size_t h = stub_registry_cap ? stub_slot(stub_registry, stub_registry_cap, name, fn) : 0;
    stub = stub_registry_cap && stub_registry[h].name ? stub_registry[h].stub : VP_IMPORT_STUB_BASE + vp_next_stub++ * VP_IMPORT_STUB_STRIDE;
    pthread_mutex_unlock(&stub_registry_lock);
    return stub;
}

/* Makes the stub stub_alloc gave a module real: its native registered and, if it is new, in the
 * registry. When another load committed the same (name, native) first, under another address, this
 * module's own stub still gets its native. Returns -1 when the table of natives is full. */
static int stub_commit(const char* name, VpNative fn, uint64_t stub) {
    int rc = 0;
    pthread_mutex_lock(&stub_registry_lock);
    size_t h = stub_registry_cap ? stub_slot(stub_registry, stub_registry_cap, name, fn) : 0;
    if (!stub_registry_cap || !stub_registry[h].name) {
        if (fn) rc = vp_register_native(stub, fn);
        char* copy = rc == 0 && stub_registry_room() ? strdup(name) : NULL;
        if (copy) {
            h = stub_slot(stub_registry, stub_registry_cap, name, fn); /* the table may have moved */
            stub_registry[h] = (StubEntry){copy, fn, stub};
            stub_registry_count++;
        }
    } else if (stub_registry[h].stub != stub && fn) {
        rc = vp_register_native(stub, fn);
    }
    pthread_mutex_unlock(&stub_registry_lock);
    return rc;
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
    uint64_t map_start, map_end;
    if (range_of(AT(lo), hi - lo, &map_start, &map_end) || map_range(map_start, map_end, module)) { free(ph); free(file); return fail(out, "cannot map the image"); }
    for (uint16_t i = 0; i < phnum; ++i) {
        if ((ph[i].type == 1 || ph[i].type == 0x61000010) && ph[i].filesz && ph[i].offset + ph[i].filesz <= elf_size) {
            memcpy((void*)(uintptr_t)AT(ph[i].vaddr), elf + ph[i].offset, ph[i].filesz);
        }
    }
    out->base = AT(lo); out->end = AT(hi); out->entry = AT(out->entry);


    /* Relocations. The image is loaded at its link address, so RELATIVE slots hold the addend;
     * the symbolic ones against imports get a stub address registered as the native. */
    /* On the heap: ~118 KB are too much for the stack of a loader thread. */
    StubSet* st = NULL; size_t stubs = 0;
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
        size_t slot_mask = 0;
        st = calloc(1, sizeof *st);
        uint32_t* const slot_index = st ? build_slot_index(module, &slot_mask) : NULL;
        if (!slot_index) { free(st); free(ph); free(file); abandon_image(map_start, map_end); return fail(out, "out of memory"); }
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
                    while (st->hash[hs] && strcmp(st->names[st->hash[hs] - 1], im->name)) hs = (hs + 1) & (STUB_HASH_SIZE - 1);
                    k = st->hash[hs] ? (size_t)st->hash[hs] - 1 : stubs;
                    if (k == stubs) {
                        if (stubs == VP_MAX_IMPORT_STUBS) { free(slot_index); free(st); free(ph); free(file); abandon_image(map_start, map_end); return fail(out, "too many imports"); }
                        st->hash[hs] = (uint16_t)(stubs + 1);
                        VpNative fn = NULL; uint64_t data = 0;
                        const int provided = resolve ? resolve(im->name, im->function, &fn, &data, user) : 0;
                        st->names[k] = im->name;
                        st->fresh[k] = 0;
                        if (im->function && provided && !fn && data) {
                            st->values[k] = data; /* another guest module's function */
                        } else if (im->function) {
                            st->natives[k] = provided ? fn : NULL;
                            st->values[k] = stub_alloc(im->name, st->natives[k]);
                            st->fresh[k] = 1;
                        } else {
                            st->values[k] = provided ? data : 0;
                        }
                        if (provided) out->imports_resolved++;
                        else {
                            if (out->imports_missing < 64) out->missing[out->imports_missing] = im->name;
                            out->imports_missing++;
                        }
                        stubs++;
                    }
                    uint64_t value = st->values[k] + (kind == 1 ? (uint64_t)addend : 0);
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
        free(st);
        abandon_image(map_start, map_end);
        return fail(out, "the code of this image differs from the translation (fingerprint)");
    }
    int committed = 0;
    for (size_t k = 0; k < stubs && committed == 0; ++k) {
        if (st->fresh[k]) committed = stub_commit(st->names[k], st->natives[k], st->values[k]);
    }
    free(st);
    if (committed) {
        abandon_image(map_start, map_end);
        return fail(out, "the table of natives is full");
    }
    vp_module_set_base(module, AT(lo));
    return 0;
}
