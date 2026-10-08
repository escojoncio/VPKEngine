/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * The demo's host: maps the guest program's segments at their addresses (above 4 GB, outside the
 * app's page-zero region), runs its translation from the entry point, and compares the result
 * with the same program compiled natively. */
#include "demo.h"
#include "vp_emit.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
typedef unsigned long u64;
u64 vp_main_native(u64* mem);
#define STACK 0x2f0000000ull
#define DATA  0x300000000ull
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int fixed(uint64_t at, uint64_t size) {
    const uint64_t page = 16384;
    const uint64_t start = at & ~(page - 1), end = (at + size + page - 1) & ~(page - 1);
    void* p = mmap((void*)(uintptr_t)start, end - start, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    return p == MAP_FAILED ? -1 : 0;
}
vp_demo_result vp_demo_run(const char* elf_path) {
    vp_demo_result r;
    memset(&r, 0, sizeof r);
    FILE* f = fopen(elf_path, "rb");
    if (!f) { snprintf(r.message, sizeof r.message, "cannot open %s", elf_path); return r; }
    fseek(f, 0, SEEK_END); long size = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char* elf = malloc(size);
    if (fread(elf, 1, size, f) != (size_t)size) { fclose(f); snprintf(r.message, sizeof r.message, "short read"); return r; }
    fclose(f);
    uint64_t entry, phoff; uint16_t phnum;
    memcpy(&entry, elf + 24, 8); memcpy(&phoff, elf + 32, 8); memcpy(&phnum, elf + 56, 2);
    /* One mapping for the whole image (the segments are 4 KB apart; the device's pages are 16 KB). */
    uint64_t lo = UINT64_MAX, hi = 0;
    for (uint16_t i = 0; i < phnum; ++i) {
        unsigned char* ph = elf + phoff + i * 56; uint32_t type; uint64_t vaddr, memsz;
        memcpy(&type, ph, 4); memcpy(&vaddr, ph + 16, 8); memcpy(&memsz, ph + 40, 8);
        if (type != 1 || !memsz) continue;
        if (vaddr < lo) lo = vaddr;
        if (vaddr + memsz > hi) hi = vaddr + memsz;
    }
    if (lo >= hi || fixed(lo, hi - lo)) { snprintf(r.message, sizeof r.message, "cannot map the guest at %#llx", (unsigned long long)lo); return r; }
    for (uint16_t i = 0; i < phnum; ++i) {
        unsigned char* ph = elf + phoff + i * 56; uint32_t type; uint64_t off, vaddr, filesz, memsz;
        memcpy(&type, ph, 4); memcpy(&off, ph + 8, 8); memcpy(&vaddr, ph + 16, 8); memcpy(&filesz, ph + 32, 8); memcpy(&memsz, ph + 40, 8);
        if (type != 1 || !memsz) continue;
        memcpy((void*)(uintptr_t)vaddr, elf + off, filesz);
    }
    if (fixed(STACK, 0x100000) || fixed(DATA, 1 << 20)) { snprintf(r.message, sizeof r.message, "cannot map the stack or data"); return r; }
    VpCpu c; memset(&c, 0, sizeof c); c.mxcsr = 0x1f80;
    c.r[VP_RSP] = STACK + 0x100000 - 0x100 - 8; vp_st64(c.r[VP_RSP], VP_HOST_EXIT_ADDRESS);
    c.r[VP_RDI] = DATA;
    double t0 = now_ms();
    const int fault = vp_run(&c, entry);
    r.translated_ms = now_ms() - t0;
    if (fault) { snprintf(r.message, sizeof r.message, "translated code faulted: %s at %#llx", c.fault_what, (unsigned long long)c.fault_rip); return r; }
    r.translated = c.r[VP_RAX];
    u64* mem2 = calloc(1 << 17, 8);
    t0 = now_ms();
    r.native = vp_main_native(mem2);
    r.native_ms = now_ms() - t0;
    free(mem2); free(elf);
    r.ok = r.translated == r.native;
    snprintf(r.message, sizeof r.message, r.ok ? "x86-64 program translated ahead of time and run natively: result %016llx, identical to the arm64 build"
                                              : "MISMATCH: translated %016llx, native %016llx", (unsigned long long)r.translated, (unsigned long long)r.native);
    return r;
}
