/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 */
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct vp_demo_result {
    int ok;                 /* 1: the translated program ran and matched the native build */
    uint64_t translated;    /* rax of the translated run */
    uint64_t native;        /* result of the same program compiled natively for this device */
    double translated_ms;
    double native_ms;
    char message[512];      /* what happened, for the screen */
} vp_demo_result;
/* Runs the translated test program whose ELF (for its data segments) is at `elf_path`. */
vp_demo_result vp_demo_run(const char* elf_path);
#ifdef __cplusplus
}
#endif
