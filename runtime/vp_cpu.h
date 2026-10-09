/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * VPEngine AOT runtime: the x86-64 machine state and the instruction semantics that the C code
 * produced by tools/vpaot calls. Header-only so that the host compiler (clang for arm64) inlines
 * everything and removes the flag computations whose results nothing reads.
 *
 * Memory model: guest addresses are host addresses (the console's memory is mapped flat at the
 * addresses the game expects). Every access goes through the vp_ld / vp_st functions so that the memory order
 * (x86 TSO on ARM) and tracing can be changed in one place.
 */
#ifndef VP_CPU_H
#define VP_CPU_H

#include <stdint.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Register indices, in the encoding order (the same as Zydis's). */
enum { VP_RAX, VP_RCX, VP_RDX, VP_RBX, VP_RSP, VP_RBP, VP_RSI, VP_RDI,
       VP_R8, VP_R9, VP_R10, VP_R11, VP_R12, VP_R13, VP_R14, VP_R15 };

typedef union VpXmm {
    uint8_t  u8[16];
    uint16_t u16[8];
    uint32_t u32[4];
    uint64_t u64[2];
    int32_t  i32[4];
    int64_t  i64[2];
    float    f32[4];
    double   f64[2];
} VpXmm;

typedef struct VpCpu {
    uint64_t r[16];
    uint64_t rip;
    /* Flags, one byte each, always 0 or 1 (AF is kept for DAA-class instructions only). */
    uint8_t cf, pf, af, zf, sf, of, df;
    uint8_t _pad;
    uint32_t mxcsr;
    uint64_t fs_base, gs_base;
    VpXmm xmm[16];
    VpXmm ymmh[16]; /* bits 128..255 of ymm0-15 (AVX): kept by legacy SSE, zeroed by VEX.128 */
    /* Set by vp_unsupported / vp_dispatch before leaving the translated code. */
    uint64_t fault_rip;
    const char* fault_what;
    void* user;
} VpCpu;

/* ---- Memory ------------------------------------------------------------------------------- */

#if defined(VP_TRACE_MEMORY)
void vp_trace_load(uint64_t address, unsigned bytes);
void vp_trace_store(uint64_t address, unsigned bytes);
#define VP_TRACE_LD(a, n) vp_trace_load((a), (n))
#define VP_TRACE_ST(a, n) vp_trace_store((a), (n))
#else
#define VP_TRACE_LD(a, n) ((void)0)
#define VP_TRACE_ST(a, n) ((void)0)
#endif

/* x86 is TSO: on ARM, VP_TSO=1 turns the plain accesses into acquire loads and release stores
 * (LRCPC on Apple silicon) for the aligned sizes; unaligned accesses stay plain, as FEX's
 * "half barrier" mode does. Off by default: it is a per-game choice (docs/PLAN.md). */
#ifndef VP_TSO
#define VP_TSO 0
#endif

#define VP_DEF_LD(bits, type)                                                            \
    static inline type vp_ld##bits(uint64_t a) {                                         \
        type v;                                                                          \
        VP_TRACE_LD(a, bits / 8);                                                        \
        if (VP_TSO && (a & (bits / 8 - 1)) == 0) {                                       \
            v = __atomic_load_n((const type*)(uintptr_t)a, __ATOMIC_ACQUIRE);            \
        } else {                                                                         \
            memcpy(&v, (const void*)(uintptr_t)a, bits / 8);                             \
        }                                                                                \
        return v;                                                                        \
    }
#define VP_DEF_ST(bits, type)                                                            \
    static inline void vp_st##bits(uint64_t a, type v) {                                 \
        VP_TRACE_ST(a, bits / 8);                                                        \
        if (VP_TSO && (a & (bits / 8 - 1)) == 0) {                                       \
            __atomic_store_n((type*)(uintptr_t)a, v, __ATOMIC_RELEASE);                  \
        } else {                                                                         \
            memcpy((void*)(uintptr_t)a, &v, bits / 8);                                   \
        }                                                                                \
    }
VP_DEF_LD(8, uint8_t) VP_DEF_LD(16, uint16_t) VP_DEF_LD(32, uint32_t) VP_DEF_LD(64, uint64_t)
VP_DEF_ST(8, uint8_t) VP_DEF_ST(16, uint16_t) VP_DEF_ST(32, uint32_t) VP_DEF_ST(64, uint64_t)
#undef VP_DEF_LD
#undef VP_DEF_ST

static inline VpXmm vp_ld128(uint64_t a) {
    VpXmm v;
    VP_TRACE_LD(a, 16);
    memcpy(&v, (const void*)(uintptr_t)a, 16);
    return v;
}
static inline void vp_st128(uint64_t a, VpXmm v) {
    VP_TRACE_ST(a, 16);
    memcpy((void*)(uintptr_t)a, &v, 16);
}

/* ---- Atomics (LOCK prefix, xchg) ----------------------------------------------------------- */

/* x86 makes locked operations atomic at any alignment; ARM's atomics fault on a misaligned
 * address. Aligned (the normal case): the native atomic. Misaligned (a "split lock", rare): one
 * process-wide lock around a plain compare-and-store (runtime/vp_host.c). */
int vp_cas_split(uint64_t a, void* expected, const void* desired, unsigned bytes);

#define VP_DEF_ATOMIC(bits, type)                                                                      \
    static inline int vp_cas##bits(uint64_t a, type* expected, type desired) {                          \
        if (__builtin_expect((a & (bits / 8 - 1)) == 0, 1))                                             \
            return __atomic_compare_exchange_n((type*)(uintptr_t)a, expected, desired, 0,               \
                                               __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);                     \
        return vp_cas_split(a, expected, &desired, bits / 8);                                           \
    }                                                                                                  \
    static inline type vp_atomic_ld##bits(uint64_t a) {                                                \
        type v;                                                                                        \
        if (__builtin_expect((a & (bits / 8 - 1)) == 0, 1))                                             \
            return __atomic_load_n((const type*)(uintptr_t)a, __ATOMIC_RELAXED);                       \
        memcpy(&v, (const void*)(uintptr_t)a, bits / 8);                                               \
        return v;                                                                                      \
    }                                                                                                  \
    static inline type vp_xchg##bits(uint64_t a, type v) {                                             \
        if (__builtin_expect((a & (bits / 8 - 1)) == 0, 1))                                             \
            return __atomic_exchange_n((type*)(uintptr_t)a, v, __ATOMIC_SEQ_CST);                      \
        type old = vp_atomic_ld##bits(a);                                                              \
        while (!vp_cas##bits(a, &old, v)) {}                                                           \
        return old;                                                                                    \
    }                                                                                                  \
    static inline type vp_fetch_add##bits(uint64_t a, type v) {                                        \
        if (__builtin_expect((a & (bits / 8 - 1)) == 0, 1))                                             \
            return __atomic_fetch_add((type*)(uintptr_t)a, v, __ATOMIC_SEQ_CST);                       \
        type old = vp_atomic_ld##bits(a);                                                              \
        while (!vp_cas##bits(a, &old, (type)(old + v))) {}                                             \
        return old;                                                                                    \
    }
VP_DEF_ATOMIC(8, uint8_t) VP_DEF_ATOMIC(16, uint16_t) VP_DEF_ATOMIC(32, uint32_t) VP_DEF_ATOMIC(64, uint64_t)

/* ---- Registers ---------------------------------------------------------------------------- */

static inline uint64_t vp_r64(const VpCpu* c, int i) { return c->r[i]; }
static inline uint32_t vp_r32(const VpCpu* c, int i) { return (uint32_t)c->r[i]; }
static inline uint16_t vp_r16(const VpCpu* c, int i) { return (uint16_t)c->r[i]; }
static inline uint8_t vp_r8(const VpCpu* c, int i) { return (uint8_t)c->r[i]; }
/* AH, CH, DH, BH: bits 8..15 of RAX..RBX. */
static inline uint8_t vp_r8h(const VpCpu* c, int i) { return (uint8_t)(c->r[i] >> 8); }

static inline void vp_w64(VpCpu* c, int i, uint64_t v) { c->r[i] = v; }
/* A 32-bit write zeroes the upper half; 8- and 16-bit writes keep the rest. */
static inline void vp_w32(VpCpu* c, int i, uint32_t v) { c->r[i] = v; }
static inline void vp_w16(VpCpu* c, int i, uint16_t v) { c->r[i] = (c->r[i] & ~UINT64_C(0xffff)) | v; }
static inline void vp_w8(VpCpu* c, int i, uint8_t v) { c->r[i] = (c->r[i] & ~UINT64_C(0xff)) | v; }
static inline void vp_w8h(VpCpu* c, int i, uint8_t v) {
    c->r[i] = (c->r[i] & ~UINT64_C(0xff00)) | ((uint64_t)v << 8);
}

/* ---- Flags -------------------------------------------------------------------------------- */

static inline uint8_t vp_parity(uint64_t r) { return (uint8_t)!__builtin_parity((unsigned)(r & 0xff)); }

#define VP_SIGN(bits, v) ((uint8_t)(((v) >> ((bits) - 1)) & 1))
#define VP_MASK(bits) ((bits) == 64 ? ~UINT64_C(0) : ((UINT64_C(1) << (bits)) - 1))

/* Flag bits (Zydis's numbering) of the mask the generated code passes: only the flags that a
 * later instruction reads are written, so the compiler drops the rest of the computation. The
 * generated code defines VP_FLAG_MASK per instruction; the default writes everything. */
#define VP_F_CF 0x001u
#define VP_F_PF 0x004u
#define VP_F_AF 0x010u
#define VP_F_ZF 0x040u
#define VP_F_SF 0x080u
#define VP_F_OF 0x800u
#define VP_F_ALL 0x8d5u
#ifndef VP_FLAG_MASK
#define VP_FLAG_MASK VP_F_ALL
#endif

/* Result flags shared by every arithmetic instruction (ZF, SF, PF). */
static inline void vp_flags_result_m(VpCpu* c, int bits, uint64_t r, unsigned mk) {
    r &= VP_MASK(bits);
    if (mk & VP_F_ZF) c->zf = (r == 0);
    if (mk & VP_F_SF) c->sf = VP_SIGN(bits, r);
    if (mk & VP_F_PF) c->pf = vp_parity(r);
}

static inline void vp_flags_add_m(VpCpu* c, int bits, uint64_t a, uint64_t b, uint64_t r, unsigned mk) {
    const uint64_t m = VP_MASK(bits);
    a &= m; b &= m; r &= m;
    vp_flags_result_m(c, bits, r, mk);
    if (mk & VP_F_CF) c->cf = (r < a);
    if (mk & VP_F_AF) c->af = ((a ^ b ^ r) >> 4) & 1;
    if (mk & VP_F_OF) c->of = VP_SIGN(bits, (a ^ r) & (b ^ r));
}

/* ADC: a + b + carry_in. */
static inline void vp_flags_adc_m(VpCpu* c, int bits, uint64_t a, uint64_t b, uint64_t cin, uint64_t r, unsigned mk) {
    const uint64_t m = VP_MASK(bits);
    a &= m; b &= m; r &= m;
    vp_flags_result_m(c, bits, r, mk);
    if (mk & VP_F_CF) c->cf = cin ? (r <= a) : (r < a);
    if (mk & VP_F_AF) c->af = ((a ^ b ^ r) >> 4) & 1;
    if (mk & VP_F_OF) c->of = VP_SIGN(bits, (a ^ r) & (b ^ r));
}

static inline void vp_flags_sub_m(VpCpu* c, int bits, uint64_t a, uint64_t b, uint64_t r, unsigned mk) {
    const uint64_t m = VP_MASK(bits);
    a &= m; b &= m; r &= m;
    vp_flags_result_m(c, bits, r, mk);
    if (mk & VP_F_CF) c->cf = (a < b);
    if (mk & VP_F_AF) c->af = ((a ^ b ^ r) >> 4) & 1;
    if (mk & VP_F_OF) c->of = VP_SIGN(bits, (a ^ b) & (a ^ r));
}

/* SBB: a - b - borrow_in. */
static inline void vp_flags_sbb_m(VpCpu* c, int bits, uint64_t a, uint64_t b, uint64_t bin, uint64_t r, unsigned mk) {
    const uint64_t m = VP_MASK(bits);
    a &= m; b &= m; r &= m;
    vp_flags_result_m(c, bits, r, mk);
    if (mk & VP_F_CF) c->cf = bin ? (a <= b) : (a < b);
    if (mk & VP_F_AF) c->af = ((a ^ b ^ r) >> 4) & 1;
    if (mk & VP_F_OF) c->of = VP_SIGN(bits, (a ^ b) & (a ^ r));
}

/* AND, OR, XOR, TEST: CF = OF = 0; AF is undefined and left 0. */
static inline void vp_flags_logic_m(VpCpu* c, int bits, uint64_t r, unsigned mk) {
    vp_flags_result_m(c, bits, r, mk);
    if (mk & VP_F_CF) c->cf = 0;
    if (mk & VP_F_OF) c->of = 0;
    if (mk & VP_F_AF) c->af = 0;
}

/* INC / DEC leave CF alone. */
static inline void vp_flags_inc_m(VpCpu* c, int bits, uint64_t a, uint64_t r, unsigned mk) {
    vp_flags_add_m(c, bits, a, 1, r, mk & ~VP_F_CF);
}
static inline void vp_flags_dec_m(VpCpu* c, int bits, uint64_t a, uint64_t r, unsigned mk) {
    vp_flags_sub_m(c, bits, a, 1, r, mk & ~VP_F_CF);
}

/* SHL/SHR/SAR with the count already masked (5 or 6 bits) and known non-zero: the caller skips
 * the flag update for a zero count, as the hardware does. */
static inline void vp_flags_shl_m(VpCpu* c, int bits, uint64_t a, unsigned n, uint64_t r, unsigned mk) {
    vp_flags_result_m(c, bits, r, mk);
    const uint8_t cf = (uint8_t)((a >> (bits - n)) & 1);
    if (mk & VP_F_CF) c->cf = cf;
    if (mk & VP_F_OF) c->of = (n == 1) ? (uint8_t)(cf ^ VP_SIGN(bits, r)) : 0; /* undefined for n > 1 */
    if (mk & VP_F_AF) c->af = 0;
}
static inline void vp_flags_shr_m(VpCpu* c, int bits, uint64_t a, unsigned n, uint64_t r, unsigned mk) {
    vp_flags_result_m(c, bits, r, mk);
    if (mk & VP_F_CF) c->cf = (uint8_t)((a >> (n - 1)) & 1);
    if (mk & VP_F_OF) c->of = (n == 1) ? VP_SIGN(bits, a) : 0;
    if (mk & VP_F_AF) c->af = 0;
}
static inline void vp_flags_sar_m(VpCpu* c, int bits, uint64_t a, unsigned n, uint64_t r, unsigned mk) {
    vp_flags_result_m(c, bits, r, mk);
    if (mk & VP_F_CF) c->cf = (uint8_t)((a >> (n - 1)) & 1);
    if (mk & VP_F_OF) c->of = 0;
    if (mk & VP_F_AF) c->af = 0;
}

/* The unmasked names, as macros so that the mask in force at the use site applies. */
#define vp_flags_result(c, b, r) vp_flags_result_m(c, b, r, VP_FLAG_MASK)
#define vp_flags_add(c, b, x, y, r) vp_flags_add_m(c, b, x, y, r, VP_FLAG_MASK)
#define vp_flags_adc(c, b, x, y, ci, r) vp_flags_adc_m(c, b, x, y, ci, r, VP_FLAG_MASK)
#define vp_flags_sub(c, b, x, y, r) vp_flags_sub_m(c, b, x, y, r, VP_FLAG_MASK)
#define vp_flags_sbb(c, b, x, y, bi, r) vp_flags_sbb_m(c, b, x, y, bi, r, VP_FLAG_MASK)
#define vp_flags_logic(c, b, r) vp_flags_logic_m(c, b, r, VP_FLAG_MASK)
#define vp_flags_inc(c, b, x, r) vp_flags_inc_m(c, b, x, r, VP_FLAG_MASK)
#define vp_flags_dec(c, b, x, r) vp_flags_dec_m(c, b, x, r, VP_FLAG_MASK)
#define vp_flags_shl(c, b, x, n, r) vp_flags_shl_m(c, b, x, n, r, VP_FLAG_MASK)
#define vp_flags_shr(c, b, x, n, r) vp_flags_shr_m(c, b, x, n, r, VP_FLAG_MASK)
#define vp_flags_sar(c, b, x, n, r) vp_flags_sar_m(c, b, x, n, r, VP_FLAG_MASK)

/* Sign extension of the low `bits` of v to 64. */
static inline int64_t vp_sext(int bits, uint64_t v) {
    switch (bits) {
    case 8: return (int8_t)v;
    case 16: return (int16_t)v;
    case 32: return (int32_t)v;
    default: return (int64_t)v;
    }
}

/* Arithmetic right shift that is well defined for every count below the width. */
static inline uint64_t vp_sar(int bits, uint64_t a, unsigned n) {
    return (uint64_t)(vp_sext(bits, a) >> n) & VP_MASK(bits);
}

static inline uint64_t vp_rol(int bits, uint64_t a, unsigned n) {
    a &= VP_MASK(bits);
    n %= (unsigned)bits;
    return n ? ((a << n) | (a >> (bits - n))) & VP_MASK(bits) : a;
}
static inline uint64_t vp_ror(int bits, uint64_t a, unsigned n) {
    a &= VP_MASK(bits);
    n %= (unsigned)bits;
    return n ? ((a >> n) | (a << (bits - n))) & VP_MASK(bits) : a;
}

/* Two-operand IMUL: result truncated; CF = OF = whether the full product fits the width. */
static inline uint64_t vp_imul(VpCpu* c, int bits, uint64_t a, uint64_t b) {
    const __int128 p = (__int128)vp_sext(bits, a) * vp_sext(bits, b);
    const uint64_t r = (uint64_t)p & VP_MASK(bits);
    const int fits = (p == (__int128)vp_sext(bits, r));
    c->cf = c->of = (uint8_t)!fits;
    c->sf = VP_SIGN(bits, r); /* SF, ZF, PF, AF are undefined; SF of the result is what
                                 hardware produces in practice and what tests compare. */
    c->zf = (r == 0);
    c->pf = vp_parity(r);
    return r;
}

/* One-operand MUL / IMUL: RDX:RAX (DX:AX, AH:AL) = RAX * src. */
static inline void vp_mul1(VpCpu* c, int bits, uint64_t src, int is_signed) {
    const uint64_t a = c->r[VP_RAX] & VP_MASK(bits);
    unsigned __int128 p;
    uint64_t lo, hi;
    src &= VP_MASK(bits);
    if (is_signed) {
        p = (unsigned __int128)((__int128)vp_sext(bits, a) * vp_sext(bits, src));
    } else {
        p = (unsigned __int128)a * src;
    }
    lo = (uint64_t)p & VP_MASK(bits);
    hi = (uint64_t)(p >> bits) & VP_MASK(bits);
    if (bits == 8) {
        vp_w16(c, VP_RAX, (uint16_t)(lo | (hi << 8)));
    } else if (bits == 16) {
        vp_w16(c, VP_RAX, (uint16_t)lo);
        vp_w16(c, VP_RDX, (uint16_t)hi);
    } else if (bits == 32) {
        vp_w32(c, VP_RAX, (uint32_t)lo);
        vp_w32(c, VP_RDX, (uint32_t)hi);
    } else {
        c->r[VP_RAX] = lo;
        c->r[VP_RDX] = hi;
    }
    if (is_signed) {
        c->cf = c->of = (uint8_t)((__int128)p != (__int128)vp_sext(bits, lo));
    } else {
        c->cf = c->of = (uint8_t)(hi != 0);
    }
}

/* Returns 0 on success, 1 on a #DE (division by zero or quotient overflow). */
static inline int vp_div1(VpCpu* c, int bits, uint64_t src, int is_signed) {
    const uint64_t m = VP_MASK(bits);
    unsigned __int128 n;
    uint64_t q, rem;
    src &= m;
    if (src == 0) {
        return 1;
    }
    if (bits == 8) {
        n = c->r[VP_RAX] & 0xffff;
    } else {
        n = ((unsigned __int128)(c->r[VP_RDX] & m) << bits) | (c->r[VP_RAX] & m);
    }
    if (is_signed) {
        const __int128 sn = (bits == 8) ? (__int128)(int16_t)n
                          : (bits == 16) ? (__int128)(int32_t)n
                          : (bits == 32) ? (__int128)(int64_t)n
                          : (__int128)n;
        const __int128 d = vp_sext(bits, src);
        __int128 sq;
        if (d == -1 && sn == -((__int128)1 << (2 * bits - 1))) {
            return 1;
        }
        sq = sn / d;
        if (sq != (__int128)vp_sext(bits, (uint64_t)sq & m)) {
            return 1;
        }
        q = (uint64_t)sq & m;
        rem = (uint64_t)(sn % d) & m;
    } else {
        const unsigned __int128 uq = n / src;
        if (uq > m) {
            return 1;
        }
        q = (uint64_t)uq;
        rem = (uint64_t)(n % src);
    }
    if (bits == 8) {
        vp_w16(c, VP_RAX, (uint16_t)(q | (rem << 8)));
    } else if (bits == 16) {
        vp_w16(c, VP_RAX, (uint16_t)q);
        vp_w16(c, VP_RDX, (uint16_t)rem);
    } else if (bits == 32) {
        vp_w32(c, VP_RAX, (uint32_t)q);
        vp_w32(c, VP_RDX, (uint32_t)rem);
    } else {
        c->r[VP_RAX] = q;
        c->r[VP_RDX] = rem;
    }
    return 0;
}

/* ---- Conditions (the Jcc / SETcc / CMOVcc encodings 0..15) -------------------------------- */

static inline int vp_cc(const VpCpu* c, int cc) {
    int v;
    switch (cc >> 1) {
    case 0: v = c->of; break;                  /* O  / NO */
    case 1: v = c->cf; break;                  /* B  / AE */
    case 2: v = c->zf; break;                  /* E  / NE */
    case 3: v = c->cf | c->zf; break;          /* BE / A  */
    case 4: v = c->sf; break;                  /* S  / NS */
    case 5: v = c->pf; break;                  /* P  / NP */
    case 6: v = c->sf != c->of; break;         /* L  / GE */
    default: v = c->zf | (c->sf != c->of);     /* LE / G  */
    }
    return (cc & 1) ? !v : v;
}

/* ---- Stack -------------------------------------------------------------------------------- */

static inline void vp_push64(VpCpu* c, uint64_t v) {
    c->r[VP_RSP] -= 8;
    vp_st64(c->r[VP_RSP], v);
}
static inline uint64_t vp_pop64(VpCpu* c) {
    const uint64_t v = vp_ld64(c->r[VP_RSP]);
    c->r[VP_RSP] += 8;
    return v;
}

/* ---- SSE scalar / packed helpers ------------------------------------------------------------ */

/* MXCSR bits used by the scalar conversions. */
#define VP_MXCSR_RC_MASK 0x6000u
#define VP_MXCSR_FTZ 0x8000u
#define VP_MXCSR_DAZ 0x0040u

/* CVTT*: truncation toward zero; an out-of-range value or NaN gives the "integer indefinite". */
static inline int32_t vp_cvtt_f32_i32(float f) {
    if (!(f > -2147483904.0f && f < 2147483648.0f)) return INT32_MIN;
    return (int32_t)f;
}
static inline int64_t vp_cvtt_f32_i64(float f) {
    if (!(f > -9223373136366403584.0f && f < 9223372036854775808.0f)) return INT64_MIN;
    return (int64_t)f;
}
static inline int32_t vp_cvtt_f64_i32(double d) {
    if (!(d > -2147483649.0 && d < 2147483648.0)) return INT32_MIN;
    return (int32_t)d;
}
static inline int64_t vp_cvtt_f64_i64(double d) {
    if (!(d > -9223372036854777856.0 && d < 9223372036854775808.0)) return INT64_MIN;
    return (int64_t)d;
}
/* CVT* (rounding by MXCSR.RC, nearest-even by default): rint follows the current host rounding
 * mode, which vp_apply_mxcsr sets from MXCSR. */
static inline int32_t vp_cvt_f32_i32(float f) {
    if (!(f > -2147483904.0f && f < 2147483648.0f)) return INT32_MIN;
    return (int32_t)rintf(f);
}
static inline int64_t vp_cvt_f32_i64(float f) {
    if (!(f > -9223373136366403584.0f && f < 9223372036854775808.0f)) return INT64_MIN;
    return (int64_t)rintf(f);
}
static inline int32_t vp_cvt_f64_i32(double d) {
    if (!(d > -2147483649.0 && d < 2147483648.0)) return INT32_MIN;
    return (int32_t)rint(d);
}
static inline int64_t vp_cvt_f64_i64(double d) {
    if (!(d > -9223372036854777856.0 && d < 9223372036854775808.0)) return INT64_MIN;
    return (int64_t)rint(d);
}

/* COMISS / UCOMISS result flags: ZF, PF, CF; OF, SF, AF cleared. */
static inline void vp_flags_fcmp(VpCpu* c, int unordered, int less, int equal) {
    c->zf = (uint8_t)(unordered | equal);
    c->pf = (uint8_t)unordered;
    c->cf = (uint8_t)(unordered | less);
    c->of = c->sf = c->af = 0;
}
static inline void vp_comiss(VpCpu* c, float a, float b) {
    vp_flags_fcmp(c, isunordered(a, b), a < b, a == b);
}
static inline void vp_comisd(VpCpu* c, double a, double b) {
    vp_flags_fcmp(c, isunordered(a, b), a < b, a == b);
}

/* MINSS/MAXSS semantics: if either is NaN, or both are zero, the second operand is returned. */
static inline float vp_minss(float a, float b) { return (a < b) ? a : b; }
static inline float vp_maxss(float a, float b) { return (a > b) ? a : b; }
static inline double vp_minsd(double a, double b) { return (a < b) ? a : b; }
static inline double vp_maxsd(double a, double b) { return (a > b) ? a : b; }

/* CMPSS / CMPPS predicates: 0..7 (SSE) and the VEX 8..31. Bits 0-2 pick the relation, bit 3
 * flips how an unordered pair (a NaN) comes out, bit 4 only says whether a quiet NaN signals. */
static inline int vp_fcmp_pred(int pred, double a, double b) {
    switch (pred & 15) {
    case 0: return a == b;                       /* EQ_OQ */
    case 1: return a < b;                        /* LT_OS */
    case 2: return a <= b;                       /* LE_OS */
    case 3: return isunordered(a, b);            /* UNORD_Q */
    case 4: return !(a == b);                    /* NEQ_UQ */
    case 5: return !(a < b);                     /* NLT_US */
    case 6: return !(a <= b);                    /* NLE_US */
    case 7: return !isunordered(a, b);           /* ORD_Q */
    case 8: return a == b || isunordered(a, b);  /* EQ_UQ */
    case 9: return !(a >= b);                    /* NGE_US */
    case 10: return !(a > b);                    /* NGT_US */
    case 11: return 0;                           /* FALSE_OQ */
    case 12: return a < b || a > b;              /* NEQ_OQ */
    case 13: return a >= b;                      /* GE_OS */
    case 14: return a > b;                       /* GT_OS */
    default: return 1;                           /* TRUE_UQ */
    }
}

/* ---- Leaving the translated code ----------------------------------------------------------- */

/* Both are provided by the host (runtime/vp_host.c in the tests): an indirect call or jump to an
 * address whose translation is not known, and an instruction the translator did not handle. */
void vp_dispatch(VpCpu* c, uint64_t target);
void vp_unsupported(VpCpu* c, uint64_t rip, const char* what);
void vp_divide_error(VpCpu* c, uint64_t rip);
/* Hooks for instructions that need the host: CPUID, RDTSC, syscall-class instructions. */
void vp_cpuid(VpCpu* c);
/* A `syscall` instruction: number in rax, arguments in rdi, rsi, rdx, r10, r8, r9; result in rax. */
void vp_syscall(VpCpu* c);
uint64_t vp_rdtsc(VpCpu* c);
void vp_apply_mxcsr(VpCpu* c);

/* ---- F16C (half precision) ----------------------------------------------------------------- */

/* vcvtph2ps: exact (every half is a float); a signalling NaN comes back quiet. */
static inline float vp_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ffu;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) bits = sign;
        else { float f = (float)m * 5.9604644775390625e-08f; /* m * 2^-24, exact */ memcpy(&bits, &f, 4); bits |= sign; }
    } else if (e == 31) {
        bits = sign | 0x7f800000u | (m ? 0x400000u | (m << 13) : 0);
    } else {
        bits = sign | ((e + 112) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* vcvtps2ph with rounding `rc` (0 nearest even, 1 down, 2 up, 3 toward zero) and MXCSR.DAZ:
 * integer rounding, independent of the host's rounding mode (FTZ does not apply, as on hardware). */
static inline uint16_t vp_f32_to_f16(float f, unsigned rc, unsigned daz) {
    uint32_t b;
    memcpy(&b, &f, 4);
    const uint16_t sign = (uint16_t)((b >> 16) & 0x8000u);
    const uint32_t E = (b >> 23) & 0xff, frac = b & 0x7fffffu;
    if (E == 255) return (uint16_t)(sign | 0x7c00u | (frac ? 0x200u | (frac >> 13) : 0));
    if (E == 0 && (frac == 0 || daz)) return sign; /* MXCSR.DAZ: a denormal input is zero */
    const uint32_t M = E ? (frac | 0x800000u) : frac;
    const int Ee = E ? (int)E : 1;
    const int eh = Ee - 112; /* the half's biased exponent if normal */
    const int shift = eh >= 1 ? 13 : 126 - Ee;
    uint32_t r, rem, half;
    if (shift >= 32) { r = 0; rem = M; half = 0xffffffffu; /* far below the halfway point */ }
    else { r = M >> shift; rem = M & ((1u << shift) - 1); half = 1u << (shift - 1); }
    const int neg = sign != 0;
    if (rem) {
        if (rc == 0) { if (rem > half || (rem == half && (r & 1))) ++r; }
        else if (rc == 2) { if (!neg) ++r; }
        else if (rc == 1) { if (neg) ++r; }
    }
    uint32_t v = eh >= 1 ? ((uint32_t)eh << 10) + (r - 0x400u) : r;
    if (v >= 0x7c00u) {
        const int to_inf = rc == 0 || (rc == 2 && !neg) || (rc == 1 && neg);
        v = to_inf ? 0x7c00u : 0x7bffu;
    }
    return (uint16_t)(sign | v);
}

#ifdef __cplusplus
}
#endif
#endif
