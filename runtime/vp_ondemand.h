/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * Translation on demand: what an embedder's engine (integrations/shadps4/aot_guest_engine.cpp)
 * asks the app for when the game reaches code of a translated module that the translation lacks
 * (found by no static analysis: a pointer computed at run time, code the data filter dropped).
 * The app translates that entry (tools/vpconvert: vp_convert_fragment), signs and loads the
 * library, and returns 0; the engine then attaches it and the game goes on. Called on the game's
 * thread that reached the code, which waits; one at a time.
 *
 *   module   the module's name (eboot, libc_prx…)
 *   offset   the entry, from the module's base
 *   known    offsets the loaded translation of that module already has (its entries), sorted
 *
 * Not part of the game pack SDK (packs never include it). */
#ifndef VP_ONDEMAND_H
#define VP_ONDEMAND_H

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*VpOnDemandHandler)(void* user, const char* module, unsigned long long offset,
                                 const unsigned long long* known, unsigned long known_count);

/* Set by the app once (NULL: no translation on demand; a missing entry then ends the game). */
void vpk_engine_set_on_demand(VpOnDemandHandler handler, void* user);

#ifdef __cplusplus
}
#endif
#endif
