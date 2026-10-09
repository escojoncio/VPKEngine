/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * The part of <string.h> the translated C uses, for building a game pack without any platform
 * SDK (clang -ffreestanding -isystem runtime/freestanding, e.g. on Windows for visionOS). The
 * calls that remain after inlining bind to the system's libSystem (tools/sdk/libSystem.tbd).
 */
#ifndef VP_FREESTANDING_STRING_H
#define VP_FREESTANDING_STRING_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
void* memcpy(void* dst, const void* src, size_t n);
void* memmove(void* dst, const void* src, size_t n);
void* memset(void* dst, int c, size_t n);
int memcmp(const void* a, const void* b, size_t n);
#ifdef __cplusplus
}
#endif
/* -ffreestanding turns off the compiler's knowledge of these: give it back so that small fixed
 * copies (bit casts, 16-byte vectors) become loads and stores instead of calls. */
#define memcpy(d, s, n) __builtin_memcpy((d), (s), (n))
#define memmove(d, s, n) __builtin_memmove((d), (s), (n))
#define memset(d, c, n) __builtin_memset((d), (c), (n))
#define memcmp(a, b, n) __builtin_memcmp((a), (b), (n))
#endif
