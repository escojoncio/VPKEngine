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
    const char* missing[64]; /* the first missing import names (static storage of vp_imports) */
    char error[256];
} VpLoadedImage;

/* The runtime's implementation of an import, by the symbol's name as the image spells it
 * (PS4: "NID#LIB#MOD" style names). For a function import, `*native` is the implementation
 * (the loader hands out the guest address of its stub); for a data import (`function == 0`),
 * `*data` is the guest address of the object. A function import may also be another guest
 * module's function: `*native` left NULL and `*data` its guest address. Returns 0 when the import
 * is not provided. */
typedef int (*VpImportResolver)(const char* name, int function, VpNative* native, uint64_t* data, void* user);

/* Guest addresses handed out for imported functions, one per import: calls to them reach the
 * resolver's natives through vp_dispatch / vp_call_native. */
#define VP_IMPORT_STUB_BASE UINT64_C(0x7ffe000000)
#define VP_IMPORT_STUB_STRIDE 16

/* Maps the image from the file (ELF or SELF) at its link addresses (the translation hard-codes
 * them), applies R_X86_64_RELATIVE/64/GLOB_DAT/JUMP_SLOT, and fills every import slot listed in
 * vp_imports[] with a stub address registered as the resolver's native. Returns 0 on success. */
int vp_load_image(const char* path, VpImportResolver resolve, void* user, VpLoadedImage* out);
/* The same, placed near `load_at` instead (the translation must be --pic). */
int vp_load_image_at(const char* path, uint64_t load_at, VpImportResolver resolve, void* user, VpLoadedImage* out);

/* The same for a given translated module (NULL: the first registered), checking that the image
 * is the one it was translated from (link base and code fingerprint). */
int vp_load_module(const char* path, VpModule* module, uint64_t load_at, VpImportResolver resolve, void* user, VpLoadedImage* out);

/* Maps `size` bytes read/write at exactly `at` (guest addresses are host addresses). */
int vp_map_fixed(uint64_t at, uint64_t size);

#ifdef __cplusplus
}
#endif
#endif
