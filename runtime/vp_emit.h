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

/* Where the translated image is in memory. The generated file defines it as the link base; a
 * loader that places the image elsewhere sets it before running (only --pic output allows that).
 * With vp_tables_relative, the guest addresses in the tables below are offsets from it. */
extern uint64_t vp_image_base;
extern const int vp_tables_relative;

/* Sorted by guest address; produced at the end of the generated file. */
extern const VpEntry vp_entries[];
extern const size_t vp_entry_count;

/* Entries into the middle of a function (C++ landing pads): grouped by function, not sorted. */
typedef struct VpExtraEntry {
    uint64_t guest;
    VpFunction function;
    uint32_t entry; /* the offset passed as `entry` */
} VpExtraEntry;
extern const VpExtraEntry vp_extra_entries[];
extern const size_t vp_extra_entry_count;

/* Imports of the image: symbol name and the slot (GOT entry) the guest reads the address from.
 * The runtime writes a unique guest address per import into the slot and registers its native
 * implementation there with vp_register_native. */
typedef struct VpImport {
    const char* name;
    uint64_t slot;
    int function;
} VpImport;
extern const VpImport vp_imports[];
extern const size_t vp_import_count;

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

/* Optional per-instruction trace hook (vpaot --trace). */
void vp_trace(VpCpu* cpu, uint64_t rip);

#ifdef __cplusplus
}
#endif
#endif
