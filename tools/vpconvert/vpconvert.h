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
} VpConvertCallbacks;

/* 0: the library is at `output`; 1: stopped on request (call again to continue);
 * negative: failed (the reason was logged). */
int vp_convert(const VpConvertConfig* config, const VpConvertCallbacks* callbacks);

#ifdef __cplusplus
}
#endif
#endif
