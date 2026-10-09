/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * What an app needs to load a game pack (vp_emit.h has the rest of the runtime's interface):
 * small and free of the CPU state, so that Swift can import it.
 */
#ifndef VP_PACK_H
#define VP_PACK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct VpModule;

/* The binary interface between translated code and the runtime: VpCpu's layout, VpModule's, and
 * the functions below. A game pack (translated code built apart from the app, loaded at run
 * time) is only used by a runtime with the same number. Bump it when any of them changes. */
#define VP_RUNTIME_ABI 1
int vp_runtime_abi(void);

/* A game pack: the translated modules of one game, built as a shared library (tools/scripts/
 * make_game_pack) and loaded by the app from the game's folder. It exports one VpPackInfo named
 * vp_pack_info; loading it registers its modules (each module registers itself). */
#define VP_PACK_MAGIC 0x4b505056u /* "VPPK" */
typedef struct VpPackInfo {
    uint32_t magic;          /* VP_PACK_MAGIC */
    uint32_t abi;            /* VP_RUNTIME_ABI the pack was built against */
    const char* title;       /* what the pack was made from (the game's title ID or folder name) */
    size_t module_count;
    struct VpModule* const* modules;
} VpPackInfo;

#ifdef __cplusplus
}
#endif
#endif
