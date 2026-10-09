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

/* An x87 register: 64-bit significand (explicit integer bit) and sign + 15-bit exponent. */
typedef struct VpF80 {
    uint64_t m;
    uint16_t se;
    uint16_t pad[3];
} VpF80;

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
    /* x87 (runtime/vp_x87.c): physical registers R0-R7 (ST(i) is R[(ftop + i) & 7]), control
     * word, status word without TOP, TOP, and the abridged tag (bit i: R_i holds a value). */
    VpF80 st[8];
    uint16_t fcw, fsw;
    uint8_t ftop, ftag;
    uint32_t rflags_ac_id; /* RFLAGS.AC and .ID as popfq left them (pushfq gives them back) */
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

static inline int16_t vp_sat16(int32_t v) { return (int16_t)(v < -32768 ? -32768 : v > 32767 ? 32767 : v); }

/* ---- SSE4.2 / AES / PCLMUL / CRC32 helpers ---------------------------------------------------- */

/* crc32 (SSE4.2): CRC-32C, reflected, no pre/post inversion (the instruction's raw step). */
static inline uint32_t vp_crc32c(uint32_t crc, uint64_t data, int bytes) {
    for (int k = 0; k < bytes; ++k) {
        crc ^= (uint8_t)(data >> (8 * k));
        for (int b = 0; b < 8; ++b) crc = (crc >> 1) ^ (0x82F63B78u & (0u - (crc & 1u)));
    }
    return crc;
}

/* pclmulqdq: carry-less 64 x 64 -> 128. */
static inline VpXmm vp_clmul(uint64_t a, uint64_t b) {
    uint64_t lo = 0, hi = 0;
    for (int i = 0; i < 64; ++i) {
        if ((b >> i) & 1) { lo ^= a << i; if (i) hi ^= a >> (64 - i); }
    }
    VpXmm r; r.u64[0] = lo; r.u64[1] = hi;
    return r;
}

static const uint8_t vp_aes_sbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16};
static const uint8_t vp_aes_isbox[256] = {
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d};

static inline uint8_t vp_aes_xt(uint8_t a) { return (uint8_t)((a << 1) ^ ((a & 0x80) ? 0x1b : 0)); }
static inline uint8_t vp_aes_mul(uint8_t a, uint8_t b) {
    uint8_t p = 0;
    for (int i = 0; i < 8; ++i) { if (b & 1) p ^= a; a = vp_aes_xt(a); b >>= 1; }
    return p;
}
/* The state is the register's 16 bytes, byte i at row i % 4, column i / 4. */
static inline VpXmm vp_aes_shift_rows(VpXmm s, int inverse) {
    VpXmm r;
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row)
            r.u8[row + 4 * c] = s.u8[row + 4 * ((inverse ? c - row + 4 : c + row) % 4)];
    return r;
}
static inline VpXmm vp_aes_sub_bytes(VpXmm s, int inverse) {
    for (int i = 0; i < 16; ++i) s.u8[i] = inverse ? vp_aes_isbox[s.u8[i]] : vp_aes_sbox[s.u8[i]];
    return s;
}
static inline VpXmm vp_aes_mix_columns(VpXmm s, int inverse) {
    VpXmm r;
    static const uint8_t m[2][4] = {{2, 3, 1, 1}, {14, 11, 13, 9}};
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) {
            uint8_t v = 0;
            for (int k = 0; k < 4; ++k) v ^= vp_aes_mul(s.u8[k + 4 * c], m[inverse][(k - row + 4) % 4]);
            r.u8[row + 4 * c] = v;
        }
    return r;
}
static inline VpXmm vp_xor128(VpXmm a, VpXmm b) { a.u64[0] ^= b.u64[0]; a.u64[1] ^= b.u64[1]; return a; }
/* aesenc / aesenclast / aesdec / aesdeclast: one round of state `s` with round key `k`. */
static inline VpXmm vp_aesenc(VpXmm s, VpXmm k, int last) {
    s = vp_aes_sub_bytes(vp_aes_shift_rows(s, 0), 0);
    if (!last) s = vp_aes_mix_columns(s, 0);
    return vp_xor128(s, k);
}
static inline VpXmm vp_aesdec(VpXmm s, VpXmm k, int last) {
    s = vp_aes_sub_bytes(vp_aes_shift_rows(s, 1), 1);
    if (!last) s = vp_aes_mix_columns(s, 1);
    return vp_xor128(s, k);
}
static inline VpXmm vp_aesimc(VpXmm s) { return vp_aes_mix_columns(s, 1); }
static inline uint32_t vp_aes_subword(uint32_t w) {
    return (uint32_t)vp_aes_sbox[w & 0xff] | (uint32_t)vp_aes_sbox[(w >> 8) & 0xff] << 8 |
           (uint32_t)vp_aes_sbox[(w >> 16) & 0xff] << 16 | (uint32_t)vp_aes_sbox[w >> 24] << 24;
}
static inline VpXmm vp_aeskeygenassist(VpXmm s, uint32_t rcon) {
    const uint32_t x1 = vp_aes_subword(s.u32[1]), x3 = vp_aes_subword(s.u32[3]);
    VpXmm r;
    r.u32[0] = x1;
    r.u32[1] = ((x1 >> 8) | (x1 << 24)) ^ rcon;
    r.u32[2] = x3;
    r.u32[3] = ((x3 >> 8) | (x3 << 24)) ^ rcon;
    return r;
}

/* pcmpestri/pcmpestrm/pcmpistri/pcmpistrm: IntRes2 of the SSE4.2 string compare, with CF ZF SF OF
 * (AF PF cleared) in `f`. `la`/`lb` < 0: implicit lengths (up to the first zero element). */
static inline uint32_t vp_pcmpstr(VpXmm a, VpXmm b, int64_t la, int64_t lb, unsigned imm, VpCpu* f) {
    const int words = imm & 1, sgn = (imm >> 1) & 1, n = words ? 8 : 16, agg = (imm >> 2) & 3, pol = (imm >> 4) & 3;
    int32_t ea[16], eb[16];
    for (int i = 0; i < n; ++i) {
        ea[i] = words ? (sgn ? (int32_t)(int16_t)a.u16[i] : (int32_t)a.u16[i]) : (sgn ? (int32_t)(int8_t)a.u8[i] : (int32_t)a.u8[i]);
        eb[i] = words ? (sgn ? (int32_t)(int16_t)b.u16[i] : (int32_t)b.u16[i]) : (sgn ? (int32_t)(int8_t)b.u8[i] : (int32_t)b.u8[i]);
    }
    if (la < 0) { la = n; for (int i = 0; i < n; ++i) if (ea[i] == 0) { la = i; break; } }
    if (lb < 0) { lb = n; for (int i = 0; i < n; ++i) if (eb[i] == 0) { lb = i; break; } }
    if (la > n) la = n;
    if (lb > n) lb = n;
    uint32_t r1 = 0;
    for (int j = 0; j < n; ++j) {
        int bit = 0;
        const int vb = j < lb;
        switch (agg) {
        case 0: /* equal any */
            if (vb) for (int i = 0; i < la; ++i) if (ea[i] == eb[j]) { bit = 1; break; }
            break;
        case 1: /* ranges: pairs (lower, upper) of a, both valid */
            if (vb) for (int i = 0; i + 1 < la; i += 2) if (eb[j] >= ea[i] && eb[j] <= ea[i + 1]) { bit = 1; break; }
            break;
        case 2: { /* equal each */
            const int va = j < la;
            bit = (va && vb) ? ea[j] == eb[j] : (!va && !vb);
            break;
        }
        default: /* equal ordered: a occurs in b at j */
            bit = 1;
            for (int k = 0; k < n - j; ++k) {
                const int va = k < la, vbk = (j + k) < lb;
                if (!va) break;                     /* the rest of a is past its end: matches */
                if (!vbk || ea[k] != eb[j + k]) { bit = 0; break; }
            }
            break;
        }
        r1 |= (uint32_t)bit << j;
    }
    const uint32_t all = (n == 16) ? 0xFFFFu : 0xFFu;
    uint32_t r2 = r1;
    if (pol == 1) r2 = ~r1 & all;
    else if (pol == 3) r2 = r1 ^ (uint32_t)(((1u << lb) - 1) & all);
    f->cf = r2 != 0;
    f->zf = lb < n;
    f->sf = la < n;
    f->of = r2 & 1;
    f->af = f->pf = 0;
    return r2;
}
static inline uint32_t vp_pcmpstr_index(uint32_t r2, unsigned imm) {
    const int n = (imm & 1) ? 8 : 16;
    if (!r2) return (uint32_t)n;
    return (imm & 0x40) ? (uint32_t)(31 - __builtin_clz(r2)) : (uint32_t)__builtin_ctz(r2);
}
static inline VpXmm vp_pcmpstr_mask(uint32_t r2, unsigned imm) {
    VpXmm m;
    memset(&m, 0, sizeof m);
    if (!(imm & 0x40)) { m.u32[0] = r2; return m; }
    if (imm & 1) { for (int i = 0; i < 8; ++i) m.u16[i] = ((r2 >> i) & 1) ? 0xFFFF : 0; }
    else { for (int i = 0; i < 16; ++i) m.u8[i] = ((r2 >> i) & 1) ? 0xFF : 0; }
    return m;
}

/* ---- x87 (runtime/vp_x87.c) ----------------------------------------------------------------- */

void vp_x87_fld_mem(VpCpu* c, uint64_t a, int kind);       /* kind: 0 f32, 1 f64, 2 f80, 3 i16, 4 i32, 5 i64 */
void vp_x87_fld_reg(VpCpu* c, int i);
void vp_x87_fld_const(VpCpu* c, int which);                 /* 1, l2t, l2e, pi, lg2, ln2, 0 */
void vp_x87_fst_reg(VpCpu* c, int i, int pop);
void vp_x87_fst_mem(VpCpu* c, uint64_t a, int kind, int pop, int truncate);
void vp_x87_arith_mem(VpCpu* c, int op, uint64_t a, int kind); /* op: add sub subr mul div divr */
void vp_x87_arith_reg(VpCpu* c, int op, int dst, int src, int pop);
void vp_x87_unary(VpCpu* c, int op);                        /* chs abs sqrt rndint */
void vp_x87_fcom_reg(VpCpu* c, int i, int pops, int quiet);
void vp_x87_fcom_mem(VpCpu* c, uint64_t a, int kind, int pop);
void vp_x87_ftst(VpCpu* c);
void vp_x87_fcomi(VpCpu* c, VpCpu* flags, int i, int pop, int quiet);
void vp_x87_fxam(VpCpu* c);
void vp_x87_fxch(VpCpu* c, int i);
void vp_x87_fcmov(VpCpu* c, int i, int cond);
void vp_x87_ffree(VpCpu* c, int i);
void vp_x87_fincstp(VpCpu* c);
void vp_x87_fdecstp(VpCpu* c);
void vp_x87_fstp_discard(VpCpu* c);
void vp_x87_fninit(VpCpu* c);
void vp_x87_fnclex(VpCpu* c);
uint16_t vp_x87_fnstsw(VpCpu* c);
uint16_t vp_x87_fnstcw(VpCpu* c);
void vp_x87_fldcw(VpCpu* c, uint16_t cw);
void vp_x87_fnstenv(VpCpu* c, uint64_t a);
void vp_x87_fldenv(VpCpu* c, uint64_t a);
void vp_x87_fnsave(VpCpu* c, uint64_t a);
void vp_x87_frstor(VpCpu* c, uint64_t a);
void vp_x87_fxsave(VpCpu* c, uint64_t a);
void vp_x87_fxrstor(VpCpu* c, uint64_t a);
void vp_x87_fprem(VpCpu* c, int ieee);
void vp_x87_fscale(VpCpu* c);
void vp_x87_fxtract(VpCpu* c);
void vp_x87_transcendental(VpCpu* c, int op);               /* sin cos sincos ptan patan f2xm1 yl2x yl2xp1 */
void vp_x87_fbld(VpCpu* c, uint64_t a);
void vp_x87_fbstp(VpCpu* c, uint64_t a);

#ifdef __cplusplus
}
#endif
#endif
