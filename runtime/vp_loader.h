/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * Loads the guest program the translation was made from: maps its segments at their own
 * addresses, applies the relocations, and connects its imports to native implementations.
 * Works for a plain x86-64 ELF and for a PS4 SELF (decrypted dump) wrapping one.
 */
#ifndef VP_LOADER_H
#define VP_LOADER_H

#include "vp_emit.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VpLoadedImage {
    uint64_t base;      /* lowest mapped address */
    uint64_t end;       /* highest mapped address + 1 */
    uint64_t entry;
    size_t imports_resolved;
    size_t imports_missing;
    char error[256];
} VpLoadedImage;

/* The runtime's implementation of an import, by the symbol's name as the image spells it
 * (PS4: "NID#LIB#MOD" style names). NULL: not provided. */
typedef VpNative (*VpImportResolver)(const char* name, void* user);

/* Guest addresses handed out for imported functions, one per import: calls to them reach the
 * resolver's natives through vp_dispatch / vp_call_native. */
#define VP_IMPORT_STUB_BASE UINT64_C(0x7ffe000000)
#define VP_IMPORT_STUB_STRIDE 16

/* Maps the image from the file (ELF or SELF) at its link addresses (the translation hard-codes
 * them), applies R_X86_64_RELATIVE/64/GLOB_DAT/JUMP_SLOT, and fills every import slot listed in
 * vp_imports[] with a stub address registered as the resolver's native. Returns 0 on success. */
int vp_load_image(const char* path, VpImportResolver resolve, void* user, VpLoadedImage* out);

/* Maps `size` bytes read/write at exactly `at` (guest addresses are host addresses). */
int vp_map_fixed(uint64_t at, uint64_t size);

#ifdef __cplusplus
}
#endif
#endif
