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

/* Memory contract. Guest addresses are host addresses, mapped in blocks of 64 KB (every range
 * below is rounded out to them). The loader keeps a registry (under a lock) of the ranges it
 * mapped itself: images, vp_map_fixed and vp_map_fixed_over. It never replaces a mapping that is
 * not in its registry (the app's memory, a PROT_NONE reservation), and never replaces another
 * module's image or a vp_map_fixed mapping either: two modules whose images share a 64 KB block
 * cannot both be loaded (the second load fails, the first is untouched).
 *  - Loading a module again (restarting the game in the process, or after vp_detach_module)
 *    replaces its own earlier image, wherever that was; the module is detached while its memory is
 *    replaced. Nothing may be running the module then. If that load fails, the image is unmapped
 *    and the module detached (parts that were prepared, see below, are given back as prepared).
 *  - A range an embedder wants an image to go into, inside its own PROT_NONE reservation for
 *    instance, is prepared with vp_map_fixed_over; the load then replaces it, but only within the
 *    image's own 64 KB blocks: the rest of a larger prepared range stays as it was. vp_map_fixed
 *    never replaces a prepared range. If a load or reload fails, the parts of the image that were
 *    prepared (also those of an earlier successful load) are not left as a hole: they are mapped
 *    again PROT_NONE and registered as prepared (vp_map_fixed_over), so the load can be retried
 *    and vp_unmap_fixed releases them. This is best effort (memory that is no longer free is left
 *    alone). Note that if the range had been free memory and not an embedder's reservation, it
 *    stays as a PROT_NONE mapping owned by the registry: release it with vp_unmap_fixed.
 *  - Release a mapping made through this API with vp_unmap_fixed, never with munmap: the registry
 *    would keep a stale entry, and a later vp_map_fixed or load there would be refused (or a
 *    load would take a module's image that is no longer there).
 *    For an image: vp_unmap_fixed(img.base, img.end - img.base); the module is detached.
 *  - Every call takes the registry lock; a load holds it while it maps and while it gives a failed
 *    image back, but not while it copies and relocates, so another thread must not touch the
 *    image's range meanwhile.
 *  - Where there is no MAP_FIXED_NOREPLACE (Darwin, old kernels) free memory is mapped with the
 *    address as a hint, and a mapping the kernel put anywhere else is unmapped and refused.
 * A failed load gives its mapping back. */

/* Maps `size` bytes read/write at exactly `at`. Returns -1, mapping nothing, when any of that
 * range (rounded to 64 KB) is already mapped, whoever mapped it (a vp_map_fixed_over range
 * included), when size is 0, or when at + size wraps around. */
int vp_map_fixed(uint64_t at, uint64_t size);
/* The same, replacing whatever is mapped there, for a range of the embedder's own (it commits
 * pages of a PROT_NONE reservation for an image). Returns -1, changing nothing, where an image or
 * a vp_map_fixed mapping is: vp_unmap_fixed that first, and for size 0 or a wrapping range. */
int vp_map_fixed_over(uint64_t at, uint64_t size);
/* Unmaps a range given by vp_map_fixed, vp_map_fixed_over or a load (all of it, in whole or in
 * parts; it may be a part of one). Returns -1, unmapping nothing, when any of it is not the
 * loader's, or when the range is empty (size 0) or wraps around. */
int vp_unmap_fixed(uint64_t at, uint64_t size);

#ifdef __cplusplus
}
#endif
#endif
