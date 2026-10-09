/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * vpconvert: a PS4 game's code (eboot.bin and the .prx files in sce_module) turned into a game pack on the
 * device that runs it, with no PC: vpaot translates each module to C, clang (linked in, run in
 * this process) compiles the pieces in parallel, lld (also linked in) links them into one Mach-O
 * library that the app then signs and loads (see platform/visionos/Sources/VPGamePack.swift).
 *
 * Resumable: everything lives in `work_dir`; a module translated before is not translated again,
 * a piece compiled before is not compiled again, and each piece's C is deleted once compiled. A
 * stop request (should_stop) ends the run after the pieces being compiled, returning 1.
 */
#ifndef VPCONVERT_H
#define VPCONVERT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VpConvertConfig {
    const char* game_dir;   /* the folder with eboot.bin (and sce_module/) */
    const char* work_dir;   /* where the conversion keeps its state */
    const char* output;     /* the library to write */
    /* Files the compiler and linker need, as the app bundles them:
     *   runtime/            vp_cpu.h vp_emit.h vp_regs.h, runtime/freestanding/ string.h math.h
     *   clang/include/      clang's own headers (stddef.h, stdint.h, ...)
     *   tbd/                libVPRuntime.tbd libSystem.tbd                                    */
    const char* sdk_dir;
    const char* title;           /* recorded in the pack (NULL: the game folder's name) */
    const char* missing_log;     /* entries the engine logged as missing (vpaot --roots), or NULL */
    int jobs;                    /* pieces compiled at once (0: 4) */
    int split;                   /* functions per piece of C (0: 300) */
    const char* triple;          /* NULL: arm64-apple-xros2.0 */
    const char* platform;        /* lld -platform_version: NULL "xros 2.0 26.0" */
    const char* opt_level;       /* NULL: "-O2" */
} VpConvertConfig;

enum {
    VP_CONVERT_TRANSLATE = 0,
    VP_CONVERT_COMPILE = 1,
    VP_CONVERT_LINK = 2,
};

typedef struct VpConvertCallbacks {
    void* user;
    /* A line for conversion.log (no time stamp: the caller adds its own). Any thread. */
    void (*log)(void* user, const char* line);
    /* Work done in the current phase: `done` of `total` units (modules, pieces, 0/1). Any thread. */
    void (*progress)(void* user, int phase, long done, long total);
    /* Non-zero: stop after what is running now (the app is about to be suspended). Any thread. */
    int (*should_stop)(void* user);
    /* The whole conversion done so far, 0 to 1 (translation ~5 %, compilation by MB of C ~92 %,
     * the link the rest). Optional (NULL). Any thread. */
    void (*overall)(void* user, double fraction);
    /* Pieces that may compile at once right now (heat): fewer than `jobs` makes the others wait.
     * Optional (NULL: always `jobs`). Any thread, every few seconds. */
    int (*max_jobs)(void* user);
    /* The embedder's own changes to a module's code, as it will make them when it loads the
     * module (an emulator's patches of a known title): `image` is the module as loaded at its
     * link base (`size` bytes, offset 0 = the lowest address); change it in place. The module is
     * translated as changed, so its fingerprint matches the changed image at run time. Must give
     * the same changes for the same settings. Optional (NULL). Called on the converting thread. */
    void (*patch_image)(void* user, const char* module, unsigned char* image, unsigned long size);
} VpConvertCallbacks;

/* 0: the library is at `output`; 1: stopped on request (call again to continue);
 * negative: failed (the reason was logged). */
int vp_convert(const VpConvertConfig* config, const VpConvertCallbacks* callbacks);

#ifdef __cplusplus
}
#endif
#endif
