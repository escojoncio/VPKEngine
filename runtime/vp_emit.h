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

/* Sorted by guest address; produced at the end of the generated file. */
extern const VpEntry vp_entries[];
extern const size_t vp_entry_count;

static inline float vp_bits_f32(uint32_t v) { float f; memcpy(&f, &v, 4); return f; }
static inline double vp_bits_f64(uint64_t v) { double d; memcpy(&d, &v, 8); return d; }

/* The guest address the host pushes as a return address: a `ret` to it ends the run. */
#define VP_HOST_EXIT_ADDRESS UINT64_C(0x7fff0000)

/* runtime/vp_host.c: runs translated code from `entry` (0 = clean exit, else a fault). */
int vp_run(VpCpu* cpu, uint64_t entry);

/* Optional per-instruction trace hook (vpaot --trace). */
void vp_trace(VpCpu* cpu, uint64_t rip);

#ifdef __cplusplus
}
#endif
#endif
