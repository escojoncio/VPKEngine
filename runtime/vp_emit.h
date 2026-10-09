/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 * What the C that vpaot writes needs besides vp_cpu.h: the entry table and bit casts.
 */
#ifndef VP_EMIT_H
#define VP_EMIT_H

#include "vp_cpu.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*VpFunction)(VpCpu* cpu, uint32_t entry);

typedef struct VpEntry {
    uint64_t guest;     /* guest address of the function's first instruction */
    VpFunction function;
} VpEntry;

/* Entries into the middle of a function (C++ landing pads): grouped by function, not sorted. */
typedef struct VpExtraEntry {
    uint64_t guest;
    VpFunction function;
    uint32_t entry; /* the offset passed as `entry` */
} VpExtraEntry;

/* Imports of the image: symbol name and the slot (GOT entry) the guest reads the address from.
 * The runtime writes a unique guest address per import into the slot and registers its native
 * implementation there with vp_register_native. */
typedef struct VpImport {
    const char* name;
    uint64_t slot;
    int function;
} VpImport;

/* A range of the image (offsets from its link base). */
typedef struct VpRange {
    uint64_t start, size;
} VpRange;

/* One translated image (an executable, a PS4 module, a DLL). The generated file defines one and
 * registers it before main(); a loader that places the image elsewhere than its link base calls
 * vp_module_set_base (only --pic output allows that). Guest addresses in the tables are offsets
 * from the base when `relative`, else absolute. */
typedef struct VpModule {
    const char* name;
    uint64_t base;          /* where the image is now */
    uint64_t link_base;     /* where it was linked (the translation's addresses) */
    uint64_t size;          /* bytes from the base that belong to it */
    int relative;
    const VpEntry* entries; /* sorted by guest address */
    size_t entry_count;
    const VpExtraEntry* extra;
    size_t extra_count;
    const VpImport* imports;
    size_t import_count;
    const VpRange* code;    /* executable ranges, for the fingerprint */
    size_t code_count;
    const uint64_t* reloc_sites; /* sorted offsets of relocated 8-byte fields: left out of it */
    size_t reloc_site_count;
    uint64_t fingerprint;   /* FNV-1a 64 of the code bytes the translation was made from */
    int attached;           /* the image is in memory at `base` (only then does dispatch use it);
                               a non --pic module starts attached at its link base */
    struct VpModule* next;
} VpModule;

void vp_register_module(VpModule* m);
VpModule* vp_module_by_name(const char* name);
VpModule* vp_module_at(uint64_t address);
VpModule* vp_first_module(void);
/* Moves a module to `base` and marks it attached. */
void vp_module_set_base(VpModule* m, uint64_t base);
/* The image was unmapped: dispatch stops using the module. */
void vp_detach_module(VpModule* m);
/* For a runtime that loads images itself (shadPS4): the registered module whose translation is
 * the image now at [base, base + size) (same size, same code fingerprint), moved there. NULL
 * when none matches: that image runs without a translation. */
VpModule* vp_attach_module(uint64_t base, uint64_t size);
/* The fingerprint of the code now at m->base: equal to m->fingerprint when the loaded image is
 * the one that was translated. */
uint64_t vp_module_fingerprint_now(const VpModule* m);
/* FNV-1a 64 over the code ranges of an image at `base` (reloc sites hashed as zero). */
uint64_t vp_fingerprint(uint64_t base, const VpRange* code, size_t code_count, const uint64_t* reloc_sites, size_t reloc_count);

static inline float vp_bits_f32(uint32_t v) { float f; memcpy(&f, &v, 4); return f; }
static inline double vp_bits_f64(uint64_t v) { double d; memcpy(&d, &v, 8); return d; }

/* The guest address the host pushes as a return address: a `ret` to it ends the run. */
#define VP_HOST_EXIT_ADDRESS UINT64_C(0x7fff0000)

/* Natives: guest addresses (imports, stubs) the host implements. A native reads its arguments
 * from the registers (System V ABI), writes rax, and ends with `cpu->rip = vp_pop64(cpu)`: it
 * pops the return address that the translated `call` pushed, exactly as a guest `ret` would. */
typedef void (*VpNative)(VpCpu* cpu);
void vp_register_native(uint64_t guest, VpNative fn);
void vp_call_native(VpCpu* cpu, uint64_t guest);

/* runtime/vp_host.c: runs translated code from `entry` (0 = clean exit, else a fault). Nested
 * calls are allowed (a native the guest called may run guest code again). */
int vp_run(VpCpu* cpu, uint64_t entry);
/* Forgets every vp_run of this thread (after a longjmp past them, as for process exit). */
void vp_run_reset(void);
/* Calls guest function `fn` from a native: the caller has set the argument registers and made
 * room on the guest stack; the return address is pushed here. Returns rax; the other registers
 * are restored. A fault aborts the process with a report. */
uint64_t vp_call_guest(VpCpu* cpu, uint64_t fn);

/* A dispatch to an address no module or native knows: the embedding runtime may handle it
 * (stubs it generated at run time, a fallback CPU) and return 1, after which the dispatch is
 * considered done (cpu->rip set by the handler as a `ret` would). Default: 0. */
int vp_dispatch_miss(VpCpu* cpu, uint64_t target);
int vp_dispatch_miss_possible(uint64_t target);

/* Optional per-instruction trace hook (vpaot --trace). */
void vp_trace(VpCpu* cpu, uint64_t rip);

#ifdef __cplusplus
}
#endif
#endif
