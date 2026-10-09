/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * VPWin: runs a translated Windows x86-64 program (a PE that vpaot turned into C). The PE is
 * placed in memory, its imports are bound to native implementations of the Win32 API and the C
 * runtime (kernel32, msvcrt / ucrt, vcruntime), the process and thread structures Windows code
 * reads through gs: (TEB, PEB, TLS) are built, and the entry point runs.
 *
 * Natives follow the Windows x64 calling convention on the guest side: integer arguments in
 * rcx, rdx, r8, r9, then on the stack above a 32-byte shadow area; floating point ones in
 * xmm0..xmm3 by position; the result in rax or xmm0.
 */
#ifndef VP_WIN_H
#define VP_WIN_H

#include "../vp_emit.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VpWinOptions {
    uint64_t load_at;       /* 0: the PE's preferred ImageBase (or anywhere if that fails and the
                               translation is --pic) */
    const char* root;       /* the folder that "C:\" stands for (NULL: the current folder) */
    int trace_imports;      /* print every native call to stderr */
} VpWinOptions;

/* Loads the PE the linked translation was made from and runs it with `argv` (argv[0] is the
 * program's name as Windows code sees it). Returns the process exit code; -1 when it could not
 * start (the reason is printed). */
int vp_win_run(const char* exe_path, int argc, char** argv, const VpWinOptions* options);

/* ---- for the API implementations ----------------------------------------------------------- */

/* Argument `i` (0-based) of the native being called, Windows x64 convention. */
static inline uint64_t vp_win_arg(const VpCpu* c, int i) {
    return i < 4 ? c->r[i == 0 ? VP_RCX : i == 1 ? VP_RDX : i == 2 ? VP_R8 : VP_R9]
                 : vp_ld64(c->r[VP_RSP] + 8 + 8 * (uint64_t)i);
}
static inline double vp_win_argf64(const VpCpu* c, int i) { return i < 4 ? c->xmm[i].f64[0] : vp_bits_f64(vp_win_arg(c, i)); }
static inline float vp_win_argf32(const VpCpu* c, int i) { return i < 4 ? c->xmm[i].f32[0] : vp_bits_f32((uint32_t)vp_win_arg(c, i)); }
/* Returns from the native: result in rax, back to the guest's return address. */
static inline void vp_win_ret(VpCpu* c, uint64_t value) { c->r[VP_RAX] = value; c->rip = vp_pop64(c); }
static inline void vp_win_retf64(VpCpu* c, double v) { memset(&c->xmm[0], 0, 16); c->xmm[0].f64[0] = v; c->rip = vp_pop64(c); }
static inline void vp_win_retf32(VpCpu* c, float v) { memset(&c->xmm[0], 0, 16); c->xmm[0].f32[0] = v; c->rip = vp_pop64(c); }

/* Calls guest function `fn` with up to 8 integer arguments (Windows x64 convention). */
uint64_t vp_win_call(VpCpu* c, uint64_t fn, int count, const uint64_t* args);

/* The guest state of the calling host thread. */
VpCpu* vp_win_cpu(void);

/* Ends the process (ExitProcess, exit). */
_Noreturn void vp_win_exit(int code);

/* Last error (TEB). */
void vp_win_set_last_error(uint32_t error);
uint32_t vp_win_get_last_error(void);

/* A native implementation, by name ("WriteFile"): NULL if not provided. */
typedef struct VpWinApi {
    const char* name;
    VpNative fn;
} VpWinApi;
extern const VpWinApi vp_win_kernel32[];
extern const VpWinApi vp_win_crt[];

/* A guest address that calls `fn` (for GetProcAddress, callbacks the host hands out). */
uint64_t vp_win_stub(const char* name, VpNative fn);

/* UTF-16 <-> UTF-8 between guest and host memory. */
size_t vp_win_wcslen(uint64_t s);
char* vp_win_utf16_to_utf8(uint64_t s, int64_t count); /* count < 0: up to the terminator; malloc'd */
size_t vp_win_utf8_to_utf16(const char* s, uint16_t* out, size_t cap); /* returns units written, with the terminator */

/* Windows path (guest memory, narrow or wide) to a host path under the root. malloc'd. */
char* vp_win_host_path(const char* windows_path);

/* printf-family formatting with msvcrt rules. The arguments come from the native's arguments
 * starting at `first` (va == 0) or from a Windows va_list pointer (va != 0). */
char* vp_win_format(VpCpu* c, const char* fmt, int first, uint64_t va, int wide, size_t* length);

/* Handles. */
enum { VP_H_NONE, VP_H_FILE, VP_H_EVENT, VP_H_MUTEX, VP_H_SEMAPHORE, VP_H_THREAD, VP_H_FIND };
typedef struct VpWinHandle VpWinHandle;
uint64_t vp_win_handle_new(int type, void* object);
void* vp_win_handle_get(uint64_t handle, int type);
int vp_win_handle_type(uint64_t handle);
int vp_win_handle_close(uint64_t handle);
int vp_win_file_fd(uint64_t handle); /* -1 if not a file */

/* The process. */
typedef struct VpWinProcess {
    uint64_t image_base, image_size, entry;
    uint64_t peb;
    uint64_t heap;              /* the process heap handle */
    char* command_line_a;       /* guest memory */
    uint16_t* command_line_w;   /* guest memory */
    char* exe_path_windows;     /* "C:\...\name.exe" */
    const char* root;
    int argc;
    char** argv;                /* host strings */
    uint64_t tls_index_address, tls_start, tls_end, tls_zero_fill, tls_callbacks;
    int trace_imports;
} VpWinProcess;
extern VpWinProcess vp_win_process;

/* Builds the TEB and static TLS of a new guest thread with a stack of `stack_size` bytes, and
 * sets the CPU's gs base and stack pointer. */
int vp_win_thread_setup(VpCpu* c, size_t stack_size);
void vp_win_thread_teardown(VpCpu* c);
/* Runs the PE's TLS callbacks with `reason` (1 process attach, 2 thread attach, 3 thread detach). */
void vp_win_tls_callbacks(VpCpu* c, uint32_t reason);

#ifdef __cplusplus
}
#endif
#endif
