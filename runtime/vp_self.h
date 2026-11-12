/* Limits shared by the two SELF unwrappers (runtime/vp_loader.c and tools/vpaot/image.cpp), so that both accept and refuse the
 * same files. Header only, plain C (also included from C++). */
#pragma once
#include <stdint.h>

#define VP_SELF_MAX_ELF (UINT64_C(1) << 31)  /* the rebuilt ELF never reaches this size */
#define VP_SELF_MAX_PAD (UINT64_C(2) << 20)  /* alignment padding counted per segment, at most (a 2 MiB page) */
#define VP_SELF_MAX_PAD_SUM (UINT64_C(64) << 20)  /* ... and in all: the rebuilt ELF is at most size + 64 MiB + 2 MiB */

/* What one data-carrying program header (one backed by a blocked SELF segment, each counted once) may add to the rebuilt ELF besides its data: the padding that puts the segment at
 * an offset congruent to its address modulo p_align (a linker with -z max-page-size=2M leaves up to 2 MiB). */
static inline uint64_t vp_self_pad(uint64_t align) {
    return align < 4096 ? 4096 : align > VP_SELF_MAX_PAD ? VP_SELF_MAX_PAD : align;
}

/* The highest end (offset + filesz) a segment may have in the ELF rebuilt from a SELF of `self_size` bytes, whose
 * data-carrying program headers have vp_self_pad() summing to `pad_sum`: the data is in the SELF, the rest is padding (plus
 * one pad for the header area). `pad_sum` is capped at VP_SELF_MAX_PAD_SUM, so a file of N bytes never justifies more than N + 66 MiB
 * however many program headers it has. A few bytes "at 2 GiB" are refused before the image is reserved. */
static inline uint64_t vp_self_end_limit(uint64_t self_size, uint64_t pad_sum) {
    pad_sum *= 2;
    if (pad_sum > VP_SELF_MAX_PAD_SUM) pad_sum = VP_SELF_MAX_PAD_SUM;
    if (self_size >= VP_SELF_MAX_ELF) return VP_SELF_MAX_ELF;
    const uint64_t l = self_size + pad_sum + VP_SELF_MAX_PAD;
    return l > VP_SELF_MAX_ELF ? VP_SELF_MAX_ELF : l;
}
