/* SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
 *
 * The x87 FPU for translated code: 80-bit extended precision computed exactly with Berkeley
 * SoftFloat 3e (runtime/softfloat), including the control word's precision and rounding
 * control, the status word's exception flags, condition codes and stack faults, and the tag
 * word. The transcendental instructions (fsin, fcos, fsincos, fptan, fpatan, f2xm1, fyl2x, fyl2xp1)
 * are computed in binary128 (SoftFloat's f128) and rounded once: within one unit of the last place of
 * the Intel hardware, bit-exact for 92-99.9% of the arguments (tests/aot/fuzz/x87_trans_diff.c).
 *
 * Included by runtime/vp_host.c (one translation unit, no extra build flags).
 */
#include "vp_cpu.h"

#include <math.h>
#include <string.h>

#include "softfloat/vp_softfloat.c"

/* Status word bits. */
#define FSW_IE 0x0001
#define FSW_DE 0x0002
#define FSW_ZE 0x0004
#define FSW_OE 0x0008
#define FSW_UE 0x0010
#define FSW_PE 0x0020
#define FSW_SF 0x0040
#define FSW_ES 0x0080
#define FSW_C0 0x0100
#define FSW_C1 0x0200
#define FSW_C2 0x0400
#define FSW_C3 0x4000
#define FSW_B 0x8000
#define FSW_CC (FSW_C0 | FSW_C1 | FSW_C2 | FSW_C3)

static const extFloat80_t vp_f80_indefinite = {UINT64_C(0xC000000000000000), 0xFFFF};

static inline extFloat80_t to_sf(const VpF80* r) { extFloat80_t v; v.signif = r->m; v.signExp = r->se; return v; }
static inline void from_sf(VpF80* r, extFloat80_t v) { r->m = v.signif; r->se = v.signExp; r->pad[0] = r->pad[1] = r->pad[2] = 0; }

/* A zeroed VpCpu (memset by an embedder) has fcw 0, which no x87 code uses: it means the
 * power-on state, FNINIT's 0x037F. */
static inline void vp_x87_fix_cw(VpCpu* c) { if (c->fcw == 0) c->fcw = 0x037F; }

/* SoftFloat's modes from the control word, flags cleared. */
static void vp_x87_begin(VpCpu* c) {
    vp_x87_fix_cw(c);
    static const uint_fast8_t rc[4] = {softfloat_round_near_even, softfloat_round_min, softfloat_round_max, softfloat_round_minMag};
    static const uint_fast8_t pc[4] = {32, 80, 64, 80}; /* 01 is reserved: treated as extended */
    softfloat_roundingMode = rc[(c->fcw >> 10) & 3];
    extF80_roundingPrecision = pc[(c->fcw >> 8) & 3];
    softfloat_detectTininess = softfloat_tininess_afterRounding;
    softfloat_exceptionFlags = 0;
}

/* SoftFloat's flags into the status word (sticky); ES and B when an unmasked one is set. */
static void vp_x87_end(VpCpu* c, uint16_t extra) {
    uint16_t f = extra;
    const uint_fast8_t e = softfloat_exceptionFlags;
    /* Invalid and zero-divide outrank the denormal-operand exception: then no DE. */
    if (e & (softfloat_flag_invalid | softfloat_flag_infinite)) f &= (uint16_t)~FSW_DE;
    if (e & softfloat_flag_invalid) f |= FSW_IE;
    if (e & softfloat_flag_infinite) f |= FSW_ZE;
    if (e & softfloat_flag_overflow) f |= FSW_OE;
    if (e & softfloat_flag_underflow) f |= FSW_UE;
    if (e & softfloat_flag_inexact) f |= FSW_PE;
    c->fsw |= f;
    if (c->fsw & ~c->fcw & 0x3F) c->fsw |= FSW_ES | FSW_B;
}

static inline int is_denormal(extFloat80_t v) { return (v.signExp & 0x7FFF) == 0 && v.signif != 0; }
static inline int is_nan(extFloat80_t v) { return (v.signExp & 0x7FFF) == 0x7FFF && (v.signif & UINT64_C(0x7FFFFFFFFFFFFFFF)); }
static inline int is_inf(extFloat80_t v) { return (v.signExp & 0x7FFF) == 0x7FFF && !(v.signif & UINT64_C(0x7FFFFFFFFFFFFFFF)); }
static inline int is_zero(extFloat80_t v) { return (v.signExp & 0x7FFF) == 0 && v.signif == 0; }
/* Unnormals, pseudo-NaNs and pseudo-infinities (exponent not 0, integer bit clear): the hardware
 * rejects them as operands (IE, the indefinite). Only fldt of garbage memory produces them. */
static inline int is_unsupported(extFloat80_t v) { return (v.signExp & 0x7FFF) != 0 && !(v.signif >> 63); }

/* ---- the stack ------------------------------------------------------------------------------- */

static inline int phys(const VpCpu* c, int i) { return (c->ftop + i) & 7; }
static inline int valid(const VpCpu* c, int i) { return (c->ftag >> phys(c, i)) & 1; }

/* Reads ST(i); an empty register is a stack underflow (IE, SF, C1 = 0) giving the indefinite. */
static int st_get(VpCpu* c, int i, extFloat80_t* v) {
    if (!valid(c, i)) {
        c->fsw = (uint16_t)((c->fsw & ~FSW_C1) | FSW_IE | FSW_SF);
        *v = vp_f80_indefinite;
        return 0;
    }
    *v = to_sf(&c->st[phys(c, i)]);
    return 1;
}
static void st_set(VpCpu* c, int i, extFloat80_t v) {
    from_sf(&c->st[phys(c, i)], v);
    c->ftag |= (uint8_t)(1u << phys(c, i));
}
/* Push: overflow (the new top is in use) gives IE, SF, C1 = 1 and the indefinite. */
static int st_push(VpCpu* c, extFloat80_t v) {
    int overflow = 0;
    c->ftop = (c->ftop - 1) & 7;
    if ((c->ftag >> c->ftop) & 1) {
        c->fsw |= FSW_IE | FSW_SF | FSW_C1;
        v = vp_f80_indefinite;
        overflow = 1;
    } else {
        c->fsw &= ~FSW_C1;
    }
    st_set(c, 0, v);
    return overflow;
}
/* fxtract, fsincos, fptan: replace ST(0) and push. On a stack overflow the hardware leaves the
 * indefinite in both (the old ST(0) included). */
static void st_replace_push(VpCpu* c, extFloat80_t st0, extFloat80_t pushed) {
    if ((c->ftag >> ((c->ftop - 1) & 7)) & 1) st0 = pushed = vp_f80_indefinite;
    st_set(c, 0, st0);
    st_push(c, pushed);
}
static void st_pop(VpCpu* c) {
    c->ftag &= (uint8_t)~(1u << c->ftop);
    c->ftop = (c->ftop + 1) & 7;
}
static void vp_x87_flags_es(VpCpu* c) { if (c->fsw & ~c->fcw & 0x3F) c->fsw |= FSW_ES | FSW_B; }

/* ---- conversions from memory ------------------------------------------------------------------ */

enum { VP_X87_F32, VP_X87_F64, VP_X87_F80, VP_X87_I16, VP_X87_I32, VP_X87_I64 };

static extFloat80_t load_kind(VpCpu* c, uint64_t a, int kind, uint16_t* extra) {
    extFloat80_t v;
    switch (kind) {
    case VP_X87_F32: {
        float32_t f; f.v = vp_ld32(a);
        if ((f.v & 0x7F800000u) == 0 && (f.v & 0x7FFFFFu)) *extra |= FSW_DE;
        if ((f.v & 0x7F800000u) == 0x7F800000u && (f.v & 0x7FFFFFu)) { /* NaN: kept as is, a signaling one is quieted by the user */
            v.signExp = (uint16_t)(((f.v >> 16) & 0x8000u) | 0x7FFFu);
            v.signif = UINT64_C(0x8000000000000000) | ((uint64_t)(f.v & 0x7FFFFFu) << 40);
        } else v = f32_to_extF80(f);
        break;
    }
    case VP_X87_F64: {
        float64_t f; f.v = vp_ld64(a);
        if ((f.v & UINT64_C(0x7FF0000000000000)) == 0 && (f.v & UINT64_C(0xFFFFFFFFFFFFF))) *extra |= FSW_DE;
        if ((f.v & UINT64_C(0x7FF0000000000000)) == UINT64_C(0x7FF0000000000000) && (f.v & UINT64_C(0xFFFFFFFFFFFFF))) {
            v.signExp = (uint16_t)(((f.v >> 48) & 0x8000u) | 0x7FFFu);
            v.signif = UINT64_C(0x8000000000000000) | ((f.v & UINT64_C(0xFFFFFFFFFFFFF)) << 11);
        } else v = f64_to_extF80(f);
        break;
    }
    case VP_X87_F80: v.signif = vp_ld64(a); v.signExp = vp_ld16(a + 8); break;
    case VP_X87_I16: v = i32_to_extF80((int16_t)vp_ld16(a)); break;
    case VP_X87_I32: v = i32_to_extF80((int32_t)vp_ld32(a)); break;
    default: v = i64_to_extF80((int64_t)vp_ld64(a)); break;
    }
    (void)c;
    return v;
}

/* ---- loads and stores ------------------------------------------------------------------------- */

void vp_x87_fld_mem(VpCpu* c, uint64_t a, int kind) {
    vp_x87_begin(c);
    uint16_t extra = 0;
    extFloat80_t v = load_kind(c, a, kind, &extra);
    if (kind <= VP_X87_F64 && extF80_isSignalingNaN(v)) { softfloat_exceptionFlags |= softfloat_flag_invalid; v.signif |= UINT64_C(0x4000000000000000); }
    if (st_push(c, v)) extra &= (uint16_t)~FSW_DE; /* a stack overflow is an invalid operation: no DE */
    vp_x87_end(c, extra);
}

void vp_x87_fld_reg(VpCpu* c, int i) {
    vp_x87_begin(c);
    extFloat80_t v;
    st_get(c, i, &v);
    st_push(c, v);
    vp_x87_end(c, 0);
}

/* fld1, fldl2t, fldl2e, fldpi, fldlg2, fldln2, fldz: the 66-bit constants rounded by RC. */
void vp_x87_fld_const(VpCpu* c, int which) {
    vp_x87_fix_cw(c);
    static const struct { uint64_t m; uint16_t se; int round_up_if; } k[7] = {
        {UINT64_C(0x8000000000000000), 0x3FFF, 0},  /* 1 */
        {UINT64_C(0xD49A784BCD1B8AFE), 0x4000, 2},  /* log2(10): rounds up only toward +inf */
        {UINT64_C(0xB8AA3B295C17F0BB), 0x3FFF, 1},  /* log2(e): nearest and up give ...BC */
        {UINT64_C(0xC90FDAA22168C234), 0x4000, 1},  /* pi */
        {UINT64_C(0x9A209A84FBCFF798), 0x3FFD, 1},  /* log10(2) */
        {UINT64_C(0xB17217F7D1CF79AB), 0x3FFE, 1},  /* ln(2) */
        {0, 0, 0},                                  /* 0 */
    };
    const int rc = (c->fcw >> 10) & 3;
    extFloat80_t v;
    v.signif = k[which].m;
    v.signExp = k[which].se;
    /* The table holds the truncated values; nearest rounds up for the ones marked 1 (their
     * dropped bits are >= half) and up (rc 2) rounds every inexact one up. */
    if (k[which].round_up_if == 1 && (rc == 0 || rc == 2)) v.signif += 1;
    if (k[which].round_up_if == 2 && rc == 2) v.signif += 1;
    st_push(c, v);
    vp_x87_flags_es(c);
}

/* The undocumented D9 D8+i (FSTPNCE): FSTP, but an empty ST(0) is only popped, with no exception. */
void vp_x87_fstpnce(VpCpu* c, int i) {
    if (!valid(c, 0)) { st_pop(c); c->fsw &= ~FSW_C1; return; }
    vp_x87_fst_reg(c, i, 1);
}
void vp_x87_fst_reg(VpCpu* c, int i, int pop) {
    vp_x87_begin(c);
    c->fsw &= ~FSW_C1;
    extFloat80_t v;
    st_get(c, 0, &v);
    st_set(c, i, v);
    if (pop) st_pop(c);
    vp_x87_end(c, 0);
}

/* Integer store of a value that does not fit: the integer indefinite, IE. */
static uint16_t fist16(extFloat80_t v, uint_fast8_t rm) {
    const int_fast32_t r = extF80_to_i32(v, rm, true);
    if (softfloat_exceptionFlags & softfloat_flag_invalid) return 0x8000;
    if (r < -32768 || r > 32767) { softfloat_exceptionFlags = (softfloat_exceptionFlags & ~softfloat_flag_inexact) | softfloat_flag_invalid; return 0x8000; }
    return (uint16_t)r;
}

/* fst/fstp m32/m64/m80, fist/fistp m16/m32/m64, fisttp (truncate = 1). */
void vp_x87_fst_mem(VpCpu* c, uint64_t a, int kind, int pop, int truncate) {
    vp_x87_begin(c);
    extFloat80_t v;
    st_get(c, 0, &v);
    const uint_fast8_t rm = truncate ? softfloat_round_minMag : softfloat_roundingMode;
    uint16_t extra = 0;
    c->fsw &= ~FSW_C1;
    /* No DE here: fst/fist do not report denormal sources. An unsupported encoding stores the
     * indefinite of the destination format (fstp m80 copies it as it is). */
    if (kind != VP_X87_F80 && is_unsupported(v)) { v = vp_f80_indefinite; softfloat_exceptionFlags |= softfloat_flag_invalid; }
    switch (kind) {
    case VP_X87_F32: {
        const uint32_t r = extF80_to_f32(v).v;
        if ((softfloat_exceptionFlags & softfloat_flag_inexact) && !is_nan(v)) { /* C1: rounded up in magnitude */
            const uint_fast8_t m = softfloat_roundingMode, f = softfloat_exceptionFlags;
            softfloat_roundingMode = softfloat_round_minMag;
            if (extF80_to_f32(v).v != r) extra |= FSW_C1;
            softfloat_roundingMode = m; softfloat_exceptionFlags = f;
        }
        vp_st32(a, r);
        break;
    }
    case VP_X87_F64: {
        const uint64_t r = extF80_to_f64(v).v;
        if ((softfloat_exceptionFlags & softfloat_flag_inexact) && !is_nan(v)) {
            const uint_fast8_t m = softfloat_roundingMode, f = softfloat_exceptionFlags;
            softfloat_roundingMode = softfloat_round_minMag;
            if (extF80_to_f64(v).v != r) extra |= FSW_C1;
            softfloat_roundingMode = m; softfloat_exceptionFlags = f;
        }
        vp_st64(a, r);
        break;
    }
    case VP_X87_F80: vp_st64(a, v.signif); vp_st16(a + 8, v.signExp); break;
    case VP_X87_I16: vp_st16(a, fist16(v, rm)); break;
    case VP_X87_I32: {
        const int_fast32_t r = extF80_to_i32(v, rm, true);
        vp_st32(a, (softfloat_exceptionFlags & softfloat_flag_invalid) ? 0x80000000u : (uint32_t)r);
        break;
    }
    default: {
        const int_fast64_t r = extF80_to_i64(v, rm, true);
        vp_st64(a, (softfloat_exceptionFlags & softfloat_flag_invalid) ? UINT64_C(0x8000000000000000) : (uint64_t)r);
        break;
    }
    }
    if ((kind == VP_X87_I16 || kind == VP_X87_I32 || kind == VP_X87_I64) &&
        (softfloat_exceptionFlags & (softfloat_flag_inexact | softfloat_flag_invalid)) == softfloat_flag_inexact) {
        /* C1: the integer was rounded up in magnitude (it differs from the truncated one). */
        const uint_fast8_t f = softfloat_exceptionFlags;
        if (extF80_to_i64(v, rm, false) != extF80_to_i64(v, softfloat_round_minMag, false)) extra |= FSW_C1;
        softfloat_exceptionFlags = f;
    }
    if (pop) st_pop(c);
    vp_x87_end(c, extra);
}

/* ---- arithmetic -------------------------------------------------------------------------------- */

enum { VP_X87_ADD, VP_X87_SUB, VP_X87_SUBR, VP_X87_MUL, VP_X87_DIV, VP_X87_DIVR };

static extFloat80_t arith(int op, extFloat80_t a, extFloat80_t b) {
    switch (op) {
    case VP_X87_ADD: return extF80_add(a, b);
    case VP_X87_SUB: return extF80_sub(a, b);
    case VP_X87_SUBR: return extF80_sub(b, a);
    case VP_X87_MUL: return extF80_mul(a, b);
    case VP_X87_DIV: return extF80_div(a, b);
    default: return extF80_div(b, a);
    }
}

/* C1 after an inexact result: set when it was rounded away from zero (up in magnitude). */
static uint16_t round_up_c1(int op, extFloat80_t a, extFloat80_t b, extFloat80_t r) {
    if (!(softfloat_exceptionFlags & softfloat_flag_inexact) || is_nan(r)) return 0; /* overflow to inf counts as up */
    const uint_fast8_t saved_mode = softfloat_roundingMode, saved_flags = softfloat_exceptionFlags;
    softfloat_roundingMode = softfloat_round_minMag;
    const extFloat80_t t = arith(op, a, b);
    softfloat_roundingMode = saved_mode;
    softfloat_exceptionFlags = saved_flags;
    return (t.signif != r.signif || t.signExp != r.signExp) ? FSW_C1 : 0;
}

/* DE: a denormal operand, unless a NaN operand (or a stack fault) decides the result first. */
static uint16_t de_rule(uint16_t extra, extFloat80_t a, extFloat80_t b, int ok) {
    if (!ok || is_nan(a) || is_nan(b)) return (uint16_t)(extra & ~FSW_DE);
    if (is_denormal(a) || is_denormal(b)) extra |= FSW_DE;
    return extra;
}

static void arith_into(VpCpu* c, int op, int dst, extFloat80_t a, extFloat80_t b, uint16_t extra, int ok) {
    /* A NaN operand decides the result before any denormal is looked at: no DE then. */
    if (is_nan(a) || is_nan(b) || !ok) extra &= ~FSW_DE;
    else if (is_denormal(a) || is_denormal(b)) extra |= FSW_DE;
    c->fsw &= ~FSW_C1;
    if (ok && (is_unsupported(a) || is_unsupported(b))) { softfloat_exceptionFlags |= softfloat_flag_invalid; extra &= ~FSW_DE; ok = 0; }
    extFloat80_t r = ok ? arith(op, a, b) : vp_f80_indefinite;
    if (ok) extra |= round_up_c1(op, a, b, r);
    st_set(c, dst, r);
    vp_x87_end(c, extra);
}

/* ST(0) = ST(0) op m32/m64/m16int/m32int. */
void vp_x87_arith_mem(VpCpu* c, int op, uint64_t a, int kind) {
    vp_x87_begin(c);
    uint16_t extra = 0;
    const extFloat80_t m = load_kind(c, a, kind, &extra);
    extFloat80_t s;
    const int ok = st_get(c, 0, &s);
    arith_into(c, op, 0, s, m, extra, ok);
}

/* ST(dst) = ST(dst) op ST(src), then pop if asked (faddp & co.). */
void vp_x87_arith_reg(VpCpu* c, int op, int dst, int src, int pop) {
    vp_x87_begin(c);
    extFloat80_t d, s;
    const int ok = st_get(c, dst, &d) & st_get(c, src, &s);
    arith_into(c, op, dst, d, s, 0, ok);
    if (pop) st_pop(c);
}

/* fchs, fabs, fsqrt, frndint. */
enum { VP_X87_CHS, VP_X87_ABS, VP_X87_SQRT, VP_X87_RNDINT };
void vp_x87_unary(VpCpu* c, int op) {
    vp_x87_begin(c);
    extFloat80_t v;
    const int ok = st_get(c, 0, &v);
    uint16_t extra = 0;
    c->fsw &= ~FSW_C1;
    if (ok) {
        switch (op) {
        case VP_X87_CHS: v.signExp ^= 0x8000; break;
        case VP_X87_ABS: v.signExp &= 0x7FFF; break;
        case VP_X87_SQRT: {
            if (is_unsupported(v)) { softfloat_exceptionFlags |= softfloat_flag_invalid; v = vp_f80_indefinite; break; }
            if (is_denormal(v)) extra |= FSW_DE;
            const extFloat80_t r = extF80_sqrt(v);
            if ((softfloat_exceptionFlags & softfloat_flag_inexact) && !is_nan(r)) {
                const uint_fast8_t m = softfloat_roundingMode, f = softfloat_exceptionFlags;
                softfloat_roundingMode = softfloat_round_minMag;
                const extFloat80_t t = extF80_sqrt(v);
                softfloat_roundingMode = m;
                softfloat_exceptionFlags = f;
                if (t.signif != r.signif || t.signExp != r.signExp) extra |= FSW_C1;
            }
            v = r;
            break;
        }
        default: {
            if (is_unsupported(v)) { softfloat_exceptionFlags |= softfloat_flag_invalid; v = vp_f80_indefinite; break; }
            if (is_denormal(v)) extra |= FSW_DE;
            /* frndint rounds to an integer in RC, independent of the precision control. */
            const uint_fast8_t p = extF80_roundingPrecision;
            extF80_roundingPrecision = 80;
            const extFloat80_t r = extF80_roundToInt(v, softfloat_roundingMode, true);
            if ((softfloat_exceptionFlags & softfloat_flag_inexact) && !is_nan(v)) {
                const uint_fast8_t f = softfloat_exceptionFlags;
                const extFloat80_t t = extF80_roundToInt(v, softfloat_round_minMag, false);
                softfloat_exceptionFlags = f;
                if (t.signif != r.signif || t.signExp != r.signExp) extra |= FSW_C1;
            }
            v = r;
            extF80_roundingPrecision = p;
            break;
        }
        }
    }
    st_set(c, 0, v);
    vp_x87_end(c, extra);
}

/* ---- comparisons ------------------------------------------------------------------------------ */

/* 0: less, 1: equal, 2: greater, 3: unordered. `quiet`: only a signalling NaN raises IE. */
static int compare(extFloat80_t a, extFloat80_t b, int quiet) {
    if (is_unsupported(a) || is_unsupported(b)) { softfloat_exceptionFlags |= softfloat_flag_invalid; return 3; }
    if (is_nan(a) || is_nan(b)) {
        if (!quiet || extF80_isSignalingNaN(a) || extF80_isSignalingNaN(b)) softfloat_exceptionFlags |= softfloat_flag_invalid;
        return 3;
    }
    if (extF80_eq(a, b)) return 1;
    return extF80_lt(a, b) ? 0 : 2;
}

static void set_cc(VpCpu* c, int r) {
    c->fsw &= ~FSW_CC;
    if (r == 0) c->fsw |= FSW_C0;
    else if (r == 1) c->fsw |= FSW_C3;
    else if (r == 3) c->fsw |= FSW_C0 | FSW_C2 | FSW_C3;
}

/* fcom/fcomp/fucom.. with a register (pops: 0, 1 or 2) or memory (kind), ftst (with zero). */
void vp_x87_fcom_reg(VpCpu* c, int i, int pops, int quiet) {
    vp_x87_begin(c);
    extFloat80_t a, b;
    const int ok = st_get(c, 0, &a) & st_get(c, i, &b);
    uint16_t extra = de_rule(0, a, b, ok);
    const int r = ok ? compare(a, b, quiet) : 3;
    if (!ok) softfloat_exceptionFlags |= softfloat_flag_invalid;
    set_cc(c, r);
    while (pops-- > 0) st_pop(c);
    vp_x87_end(c, extra);
}
void vp_x87_fcom_mem(VpCpu* c, uint64_t a, int kind, int pop) {
    vp_x87_begin(c);
    uint16_t extra = 0;
    const extFloat80_t m = load_kind(c, a, kind, &extra);
    extFloat80_t s;
    const int ok = st_get(c, 0, &s);
    extra = de_rule(extra, s, m, ok);
    const int r = ok ? compare(s, m, 0) : 3;
    if (!ok) softfloat_exceptionFlags |= softfloat_flag_invalid;
    set_cc(c, r);
    if (pop) st_pop(c);
    vp_x87_end(c, extra);
}
void vp_x87_ftst(VpCpu* c) {
    vp_x87_begin(c);
    extFloat80_t s;
    const extFloat80_t z = {0, 0};
    const int ok = st_get(c, 0, &s);
    const int r = ok ? compare(s, z, 0) : 3;
    if (!ok) softfloat_exceptionFlags |= softfloat_flag_invalid;
    set_cc(c, r);
    vp_x87_end(c, de_rule(0, s, z, ok));
}
/* fcomi/fcomip/fucomi/fucomip: ZF, PF, CF in `flags` (the translated code's flag state). */
void vp_x87_fcomi(VpCpu* c, VpCpu* flags, int i, int pop, int quiet) {
    vp_x87_begin(c);
    extFloat80_t a, b;
    const int ok = st_get(c, 0, &a) & st_get(c, i, &b);
    const int r = ok ? compare(a, b, quiet) : 3;
    if (!ok) softfloat_exceptionFlags |= softfloat_flag_invalid;
    /* C0, C2, C3 are not touched; C1 only by a stack underflow (st_get). */
    flags->zf = r == 1 || r == 3;
    flags->pf = r == 3;
    flags->cf = r == 0 || r == 3;
    flags->of = flags->sf = flags->af = 0;
    if (pop) st_pop(c);
    vp_x87_end(c, de_rule(0, a, b, ok));
}

/* fxam: the class of ST(0) in C3, C2, C0 and its sign in C1 (empty included). */
void vp_x87_fxam(VpCpu* c) {
    vp_x87_fix_cw(c);
    const extFloat80_t v = to_sf(&c->st[c->ftop]);
    uint16_t cc;
    const int e = v.signExp & 0x7FFF;
    if (!valid(c, 0)) cc = FSW_C3 | FSW_C0;
    else if (is_unsupported(v)) cc = 0; /* unnormal, pseudo-NaN, pseudo-infinity: "unsupported" */
    else if (e == 0x7FFF) cc = (v.signif & UINT64_C(0x7FFFFFFFFFFFFFFF)) ? FSW_C0 : (FSW_C2 | FSW_C0);
    else if (e == 0) cc = v.signif ? (FSW_C3 | FSW_C2) : FSW_C3;
    else cc = (v.signif >> 63) ? FSW_C2 : 0; /* an unnormal (integer bit clear) is "unsupported" */
    if (v.signExp & 0x8000) cc |= FSW_C1;
    c->fsw = (uint16_t)((c->fsw & ~FSW_CC) | cc);
}

/* ---- the rest of the stack ---------------------------------------------------------------------- */

void vp_x87_fxch(VpCpu* c, int i) {
    vp_x87_begin(c);
    extFloat80_t a, b;
    st_get(c, 0, &a);
    st_get(c, i, &b);
    st_set(c, 0, b);
    st_set(c, i, a);
    c->fsw &= ~FSW_C1;
    vp_x87_end(c, 0);
}
/* fcmovcc: ST(0) = ST(i) when `cond`. Both registers are checked whatever the condition: an
 * empty one is a stack underflow that leaves the indefinite in ST(0). */
void vp_x87_fcmov(VpCpu* c, int i, int cond) {
    vp_x87_begin(c);
    if (!valid(c, 0) || !valid(c, i)) {
        c->fsw = (uint16_t)((c->fsw & ~FSW_C1) | FSW_IE | FSW_SF);
        st_set(c, 0, vp_f80_indefinite);
    } else if (cond) {
        st_set(c, 0, to_sf(&c->st[phys(c, i)]));
    }
    vp_x87_end(c, 0);
}
void vp_x87_ffree(VpCpu* c, int i) { c->ftag &= (uint8_t)~(1u << phys(c, i)); c->fsw &= ~FSW_C1; } /* C1 is cleared (Intel) */
void vp_x87_fincstp(VpCpu* c) { c->ftop = (c->ftop + 1) & 7; c->fsw &= ~FSW_C1; }
void vp_x87_fdecstp(VpCpu* c) { c->ftop = (c->ftop - 1) & 7; c->fsw &= ~FSW_C1; }
void vp_x87_fstp_discard(VpCpu* c) { st_pop(c); } /* ffreep */

/* ---- control ------------------------------------------------------------------------------------ */

void vp_x87_fninit(VpCpu* c) { c->fcw = 0x037F; c->fsw = 0; c->ftop = 0; c->ftag = 0; }
void vp_x87_fnclex(VpCpu* c) { vp_x87_fix_cw(c); c->fsw &= 0x7F00; }
uint16_t vp_x87_fnstsw(VpCpu* c) { return (uint16_t)((c->fsw & ~0x3800) | ((c->ftop & 7) << 11)); }
uint16_t vp_x87_fnstcw(VpCpu* c) { vp_x87_fix_cw(c); return c->fcw; }
/* Unmasked x87 exceptions (a cleared mask bit) are not modelled: results are always the masked
 * ones and no #MF is raised. PS4 code runs masked; say it once if a game unmasks anything. */
static void vp_x87_check_masks(const VpCpu* c) {
    static int told;
    if ((c->fcw & 0x3F) != 0x3F && !__atomic_exchange_n(&told, 1, __ATOMIC_RELAXED))
        fprintf(stderr, "VPKENGINE: x87 control word %#x unmasks exceptions; results stay the masked ones\n", c->fcw);
}

/* The control word bits that exist: IC (12), RC, PC, the masks (0-5); bit 6 reads as 1, 7 and 13-15 as 0 (Intel). */
#define VP_FCW_MASK 0x1F3F
void vp_x87_fldcw(VpCpu* c, uint16_t cw) {
    c->fcw = (uint16_t)((cw & VP_FCW_MASK) | 0x0040); /* bit 6 reads as 1 */
    c->fsw &= ~(FSW_ES | FSW_B);
    vp_x87_flags_es(c);
    vp_x87_check_masks(c);
}

/* The full tag word: 00 valid, 01 zero, 10 special, 11 empty, by physical register. */
static uint16_t full_tag(const VpCpu* c) {
    uint16_t t = 0;
    for (int r = 0; r < 8; ++r) {
        unsigned tag;
        if (!((c->ftag >> r) & 1)) tag = 3;
        else {
            const extFloat80_t v = to_sf(&c->st[r]);
            const int e = v.signExp & 0x7FFF;
            if (e == 0 && v.signif == 0) tag = 1;
            else if (e == 0x7FFF || e == 0 || !(v.signif >> 63)) tag = 2;
            else tag = 0;
        }
        t |= (uint16_t)(tag << (2 * r));
    }
    return t;
}

/* fnstenv / fldenv, 28-byte protected-mode layout (instruction and data pointers are 0); with the
 * operand-size prefix (the _16 forms) the 14-byte one: three words, then the four pointer words. */
static void env_store(VpCpu* c, uint64_t a, int size16) {
    vp_x87_fix_cw(c);
    uint8_t env[28];
    memset(env, 0, sizeof env);
    const uint16_t cw = c->fcw, sw = vp_x87_fnstsw(c), tw = full_tag(c);
    if (size16) {
        memcpy(env + 0, &cw, 2);
        memcpy(env + 2, &sw, 2);
        memcpy(env + 4, &tw, 2);
        memcpy((void*)(uintptr_t)a, env, 14);
    } else {
        memcpy(env + 0, &cw, 2);
        memcpy(env + 4, &sw, 2);
        memcpy(env + 8, &tw, 2);
        env[2] = env[3] = env[6] = env[7] = env[10] = env[11] = env[26] = env[27] = 0xFF; /* reserved halves read as ones */
        memcpy((void*)(uintptr_t)a, env, sizeof env);
    }
    c->fcw |= 0x3F; /* fnstenv masks every exception afterwards, which clears ES and B */
    c->fsw &= ~(FSW_ES | FSW_B);
}
void vp_x87_fnstenv(VpCpu* c, uint64_t a) { env_store(c, a, 0); }
void vp_x87_fnstenv_16(VpCpu* c, uint64_t a) { env_store(c, a, 1); }
static void env_load(VpCpu* c, uint64_t a, int size16) {
    uint16_t cw, sw, tw;
    memcpy(&cw, (const void*)(uintptr_t)a, 2);
    memcpy(&sw, (const void*)(uintptr_t)(a + (size16 ? 2 : 4)), 2);
    memcpy(&tw, (const void*)(uintptr_t)(a + (size16 ? 4 : 8)), 2);
    c->fcw = (uint16_t)((cw & VP_FCW_MASK) | 0x0040);
    c->ftop = (sw >> 11) & 7;
    c->fsw = sw & ~0x3800;
    c->ftag = 0;
    for (int r = 0; r < 8; ++r) if (((tw >> (2 * r)) & 3) != 3) c->ftag |= (uint8_t)(1u << r);
    c->fsw &= ~(FSW_ES | FSW_B);
    vp_x87_flags_es(c);
    vp_x87_check_masks(c);
}
void vp_x87_fldenv(VpCpu* c, uint64_t a) { env_load(c, a, 0); }
void vp_x87_fldenv_16(VpCpu* c, uint64_t a) { env_load(c, a, 1); }
/* fnsave / frstor: the environment, then the registers in stack order (ST0 first), 10 bytes each. */
static void save_regs(VpCpu* c, uint64_t a, int size16) {
    env_store(c, a, size16);
    const uint64_t r0 = a + (size16 ? 14 : 28);
    for (int i = 0; i < 8; ++i) {
        const VpF80* r = &c->st[phys(c, i)];
        memcpy((void*)(uintptr_t)(r0 + 10 * i), &r->m, 8);
        memcpy((void*)(uintptr_t)(r0 + 8 + 10 * i), &r->se, 2);
    }
    vp_x87_fninit(c);
}
static void restore_regs(VpCpu* c, uint64_t a, int size16) {
    env_load(c, a, size16);
    const uint64_t r0 = a + (size16 ? 14 : 28);
    for (int i = 0; i < 8; ++i) {
        VpF80* r = &c->st[phys(c, i)];
        memcpy(&r->m, (const void*)(uintptr_t)(r0 + 10 * i), 8);
        memcpy(&r->se, (const void*)(uintptr_t)(r0 + 8 + 10 * i), 2);
    }
}
void vp_x87_fnsave(VpCpu* c, uint64_t a) { save_regs(c, a, 0); }
void vp_x87_fnsave_16(VpCpu* c, uint64_t a) { save_regs(c, a, 1); }
void vp_x87_frstor(VpCpu* c, uint64_t a) { restore_regs(c, a, 0); }
void vp_x87_frstor_16(VpCpu* c, uint64_t a) { restore_regs(c, a, 1); }

/* fxsave / fxrstor (512 bytes): x87 + MXCSR + XMM0-15. */
void vp_x87_fxsave(VpCpu* c, uint64_t a) {
    vp_x87_fix_cw(c);
    uint8_t* p = (uint8_t*)(uintptr_t)a;
    memset(p, 0, 160);
    const uint16_t cw = c->fcw, sw = vp_x87_fnstsw(c);
    memcpy(p + 0, &cw, 2);
    memcpy(p + 2, &sw, 2);
    p[4] = c->ftag; /* abridged, by physical register */
    /* MXCSR_MASK: 0x2FFFF is a decision (tests/rt/x86_float.c fixes it): the target is the PS4's Jaguar, which reports it
     * (MM, the misaligned-SSE bit, included). Intel CPUs report 0xFFFF, so a native comparison on one differs in this byte. */
    const uint32_t mxcsr = c->mxcsr, mask = 0x2FFFF;
    memcpy(p + 24, &mxcsr, 4);
    memcpy(p + 28, &mask, 4);
    for (int i = 0; i < 8; ++i) {
        const VpF80* r = &c->st[phys(c, i)];
        memcpy(p + 32 + 16 * i, &r->m, 8);
        memcpy(p + 40 + 16 * i, &r->se, 2);
    }
    memcpy(p + 160, c->xmm, 16 * 16);
}
void vp_x87_fxrstor(VpCpu* c, uint64_t a) {
    const uint8_t* p = (const uint8_t*)(uintptr_t)a;
    uint16_t cw, sw;
    memcpy(&cw, p + 0, 2);
    memcpy(&sw, p + 2, 2);
    c->fcw = (uint16_t)((cw & VP_FCW_MASK) | 0x0040);
    c->ftop = (sw >> 11) & 7;
    c->fsw = sw & ~0x3800;
    c->fsw &= ~(FSW_ES | FSW_B); /* recomputed from the flags and the masks, as fldenv does */
    vp_x87_flags_es(c);
    vp_x87_check_masks(c);
    c->ftag = p[4];
    memcpy(&c->mxcsr, p + 24, 4);
    vp_apply_mxcsr(c);
    for (int i = 0; i < 8; ++i) {
        VpF80* r = &c->st[phys(c, i)];
        memcpy(&r->m, p + 32 + 16 * i, 8);
        memcpy(&r->se, p + 40 + 16 * i, 2);
    }
    memcpy(c->xmm, p + 160, 16 * 16);
}

/* ---- fprem, fprem1, fscale, fxtract ------------------------------------------------------------- */

/* fprem (truncating) / fprem1 (round to nearest): exact remainder with the quotient's low three
 * bits in C0, C3, C1; when the exponents are 64 or more apart, a partial remainder and C2 = 1. */
void vp_x87_fprem(VpCpu* c, int ieee) {
    vp_x87_begin(c);
    extFloat80_t a, b;
    const int ok = st_get(c, 0, &a) & st_get(c, 1, &b);
    uint16_t extra = de_rule(0, a, b, ok);
    /* A NaN or invalid result clears C1 and C2 only (the hardware leaves C0 and C3). */
    c->fsw &= ~(FSW_C1 | FSW_C2);
    if (!ok) { st_set(c, 0, vp_f80_indefinite); vp_x87_end(c, extra); return; }
    if (is_unsupported(a) || is_unsupported(b)) {
        softfloat_exceptionFlags |= softfloat_flag_invalid;
        st_set(c, 0, vp_f80_indefinite);
        vp_x87_end(c, 0);
        return;
    }
    if (is_nan(a) || is_nan(b)) {
        st_set(c, 0, extF80_add(a, b)); /* NaN propagation (and IE for a signalling one) */
        vp_x87_end(c, extra);
        return;
    }
    if (is_inf(a) || is_zero(b)) {
        softfloat_exceptionFlags |= softfloat_flag_invalid;
        st_set(c, 0, vp_f80_indefinite);
        vp_x87_end(c, extra);
        return;
    }
    c->fsw &= ~FSW_CC;
    if (is_zero(a) || is_inf(b)) { vp_x87_end(c, extra); return; } /* ST(0) unchanged */
    int ea = a.signExp & 0x7FFF, eb = b.signExp & 0x7FFF;
    uint64_t ma = a.signif, mb = b.signif;
    /* Normalize (denormals and unnormals). */
    if (!ea) ea = 1;
    if (!eb) eb = 1;
    while (!(ma >> 63)) { ma <<= 1; --ea; }
    while (!(mb >> 63)) { mb <<= 1; --eb; }
    const int sign = a.signExp & 0x8000;
    int d = ea - eb;
    if (d < 0 && !(ieee && d == -1)) { vp_x87_end(c, extra); return; } /* |a| < |b|: a, q = 0 */
    uint64_t q = 0;
    unsigned __int128 r;
    int partial = 0;
    int rexp = eb;
    if (d >= 64) {
        /* Partial remainder: reduce the exponent difference to a multiple of 32 below 64, as
         * the hardware does, by dividing by b scaled up by 2^(d - n). */
        const int n = 32 + (d % 32);
        partial = 1;
        rexp = ea - n;
        const unsigned __int128 num = (unsigned __int128)ma << n;
        r = num % mb;
        q = (uint64_t)(num / mb);
    } else if (d >= 0) {
        const unsigned __int128 num = (unsigned __int128)ma << d;
        r = num % mb;
        q = (uint64_t)(num / mb);
        if (ieee) {
            const unsigned __int128 twice = r << 1;
            if (twice > mb || (twice == mb && (q & 1))) { r = mb - r; q += 1; rexp = eb; goto flip; }
        }
    } else { /* d == -1, fprem1: a/b = ma / (2 mb), rounds to q = 1 above one half (a tie is even: 0) */
        r = ma;
        rexp = ea;
        if (ma > mb) { r = ((unsigned __int128)mb << 1) - ma; q = 1; goto flip; }
    }
    {
        extFloat80_t res;
        if (r == 0) { res.signif = 0; res.signExp = (uint16_t)sign; }
        else {
            uint64_t m = (uint64_t)r;
            int e = rexp;
            while (!(m >> 63)) { m <<= 1; --e; }
            if (e <= 0) { m >>= (1 - e); e = 0; }
            res.signif = m; res.signExp = (uint16_t)(sign | e);
        }
        st_set(c, 0, res);
        goto codes;
    }
flip: {
        extFloat80_t res;
        uint64_t m = (uint64_t)r;
        int e = rexp;
        if (m == 0) { res.signif = 0; res.signExp = (uint16_t)(sign ^ 0x8000); }
        else {
            while (!(m >> 63)) { m <<= 1; --e; }
            if (e <= 0) { m >>= (1 - e); e = 0; }
            res.signif = m; res.signExp = (uint16_t)((sign ^ 0x8000) | e);
        }
        st_set(c, 0, res);
    }
codes:
    if (partial) c->fsw |= FSW_C2;
    else {
        if (q & 4) c->fsw |= FSW_C0;
        if (q & 2) c->fsw |= FSW_C3;
        if (q & 1) c->fsw |= FSW_C1;
    }
    vp_x87_end(c, extra);
}

/* fscale: ST(0) * 2^trunc(ST(1)), rounded once (overflow and underflow as a multiply would). */
void vp_x87_fscale(VpCpu* c) {
    vp_x87_begin(c);
    extFloat80_t a, b;
    const int ok = st_get(c, 0, &a) & st_get(c, 1, &b);
    uint16_t extra = de_rule(0, a, b, ok);
    c->fsw &= ~FSW_C1;
    extFloat80_t r;
    if (!ok) r = vp_f80_indefinite;
    else if (is_unsupported(a) || is_unsupported(b)) { softfloat_exceptionFlags |= softfloat_flag_invalid; r = vp_f80_indefinite; extra = 0; }
    else if (is_nan(a) || is_nan(b)) r = extF80_add(a, b);
    else if (is_inf(b)) {
        const int neg = b.signExp & 0x8000;
        if ((neg && is_inf(a)) || (!neg && is_zero(a))) { softfloat_exceptionFlags |= softfloat_flag_invalid; r = vp_f80_indefinite; }
        else if (neg) { r.signif = 0; r.signExp = a.signExp & 0x8000; }
        else { r.signif = is_zero(a) ? 0 : UINT64_C(0x8000000000000000); r.signExp = (uint16_t)((a.signExp & 0x8000) | (is_zero(a) ? 0 : 0x7FFF)); }
    } else if (is_zero(a) || is_inf(a)) r = a;
    else {
        /* trunc(ST(1)), saturated far beyond any exponent (to_i64 of a huge value would give
         * the integer indefinite, which is negative). */
        int32_t n;
        if ((b.signExp & 0x7FFF) >= 0x3FFF + 30) n = (b.signExp & 0x8000) ? -100000 : 100000;
        else n = (int32_t)extF80_to_i32_r_minMag(b, false);
        int e = a.signExp & 0x7FFF;
        uint64_t m = a.signif;
        if (!e) { e = 1; while (!(m >> 63)) { m <<= 1; --e; } }
        r = softfloat_roundPackToExtF80(a.signExp >> 15, (int_fast32_t)e + n, m, 0, 80); /* fscale ignores PC */
        if (softfloat_exceptionFlags & softfloat_flag_inexact) { /* C1: rounded up in magnitude */
            const uint_fast8_t rm = softfloat_roundingMode, f = softfloat_exceptionFlags;
            softfloat_roundingMode = softfloat_round_minMag;
            const extFloat80_t t = softfloat_roundPackToExtF80(a.signExp >> 15, (int_fast32_t)e + n, m, 0, 80); /* fscale ignores PC */
            softfloat_roundingMode = rm;
            softfloat_exceptionFlags = f;
            if (t.signif != r.signif || t.signExp != r.signExp) extra |= FSW_C1;
        }
    }
    st_set(c, 0, r);
    vp_x87_end(c, extra);
}

/* fxtract: ST(0) = exponent (unbiased, as a value), then push the significand (exponent 0). */
void vp_x87_fxtract(VpCpu* c) {
    vp_x87_begin(c);
    extFloat80_t a;
    const int ok = st_get(c, 0, &a);
    uint16_t extra = de_rule(0, a, a, ok);
    extFloat80_t ex, sig;
    const int overflow = (c->ftag >> ((c->ftop - 1) & 7)) & 1;
    if (!ok || overflow) { ex = sig = vp_f80_indefinite; extra = 0; } /* a stack fault comes first */
    else if (is_unsupported(a)) { softfloat_exceptionFlags |= softfloat_flag_invalid; ex = sig = vp_f80_indefinite; extra = 0; }
    else if (is_nan(a)) { ex = sig = extF80_add(a, a); }
    else if (is_zero(a)) {
        softfloat_exceptionFlags |= softfloat_flag_infinite;
        ex.signif = UINT64_C(0x8000000000000000); ex.signExp = 0xFFFF; /* -inf */
        sig = a;
    } else if (is_inf(a)) {
        ex.signif = UINT64_C(0x8000000000000000); ex.signExp = 0x7FFF;
        sig = a;
    } else {
        int e = a.signExp & 0x7FFF;
        uint64_t m = a.signif;
        if (!e) { e = 1; while (!(m >> 63)) { m <<= 1; --e; } }
        ex = i32_to_extF80(e - 0x3FFF);
        sig.signif = m;
        sig.signExp = (uint16_t)((a.signExp & 0x8000) | 0x3FFF);
    }
    st_replace_push(c, ex, sig);
    vp_x87_end(c, extra);
}

static const extFloat80_t vp_f80_one = {UINT64_C(0x8000000000000000), 0x3FFF};

/* ---- transcendental functions ------------------------------------------------------------------- */

/* Computed in binary128 (SoftFloat's f128, 113 bits) and rounded once to the 64 bits of the result:
 * correctly rounded but for the rare cases where the 113-bit value is within 2^-50 ulp of a tie.
 * Intel's own results are within an ulp of these (measured with tests/aot/fuzz/x87_trans_diff.c). */
#include "vp_x87_trans.h"

typedef float128_t F128;
static const F128 TF_PI = {VP_F128_PI}, TF_PI_2 = {VP_F128_PI_2}, TF_PI_4 = {VP_F128_PI_4}, TF_3PI_4 = {VP_F128_3PI_4},
                  TF_LN2 = {VP_F128_LN2}, TF_LOG2E = {VP_F128_LOG2E};

static F128 tf_i(int64_t v) { return i64_to_f128(v); }
static F128 tf_f80(extFloat80_t v) { return extF80_to_f128(v); }
static F128 tf_add(F128 a, F128 b) { return f128_add(a, b); }
static F128 tf_sub(F128 a, F128 b) { return f128_sub(a, b); }
static F128 tf_mul(F128 a, F128 b) { return f128_mul(a, b); }
static F128 tf_div(F128 a, F128 b) { return f128_div(a, b); }
static int tf_exp(F128 a) { return (int)((a.v[1] >> 48) & 0x7FFF); }
static F128 tf_neg(F128 a) { a.v[1] ^= UINT64_C(0x8000000000000000); return a; }
static F128 tf_abs(F128 a) { a.v[1] &= ~UINT64_C(0x8000000000000000); return a; }
/* a * 2^n for a finite value whose scaled magnitude stays inside the normal range of binary128, |n| < 16000
 * (zero stays zero; a subnormal a, e.g. an extended denormal, is scaled by a multiplication). */
static F128 tf_scale(F128 a, int n) {
    if (!(a.v[1] & UINT64_C(0x7FFF000000000000))) {
        if (!a.v[0] && !(a.v[1] & UINT64_C(0x0000FFFFFFFFFFFF))) return a;
        F128 p; /* 2^n by the exponent field alone */
        p.v[0] = 0; p.v[1] = (uint64_t)(0x3FFF + n) << 48;
        return f128_mul(a, p);
    }
    a.v[1] += (uint64_t)(int64_t)n << 48;
    return a;
}

/* The rounding mode and flags of the caller are kept apart from the binary128 arithmetic, which runs
 * to nearest and whose inexact flags mean nothing. */
typedef struct { uint_fast8_t flags, mode; } TfSave;
static TfSave tf_enter(void) {
    const TfSave s = {softfloat_exceptionFlags, softfloat_roundingMode};
    softfloat_roundingMode = softfloat_round_near_even;
    return s;
}
static void tf_leave(TfSave s) { softfloat_exceptionFlags = s.flags; softfloat_roundingMode = s.mode; }

/* Rounds v to extended precision with the current mode (flags: inexact, underflow, overflow); C1 in
 * *extra when the magnitude was rounded up. */
static extFloat80_t tf_round(F128 v, uint16_t* extra) {
    extFloat80_t r = f128_to_extF80(v);
    if ((v.v[1] & UINT64_C(0x7FFF000000000000)) == UINT64_C(0x7FFF000000000000)) {
        /* The operands of every caller are finite: an infinite v is an overflow already in binary128, and the
         * extended result overflows in the caller's rounding mode (infinity or the largest finite value). */
        r = softfloat_roundPackToExtF80((bool)(v.v[1] >> 63), 0x7FFF, UINT64_C(0x8000000000000000), 0, 80);
        if (is_inf(r) && extra) *extra |= FSW_C1;
    } else if (is_inf(r)) softfloat_exceptionFlags |= softfloat_flag_overflow | softfloat_flag_inexact;
    /* The hardware flags every finite non-zero result inexact, an exact one (fyl2x of 2, 1) included,
     * and a denormal one also underflows. */
    if (!is_zero(r) && !is_inf(r) && !is_nan(r)) {
        softfloat_exceptionFlags |= softfloat_flag_inexact;
        if (!(r.signExp & 0x7FFF)) softfloat_exceptionFlags |= softfloat_flag_underflow;
    }
    if (extra) {
        const F128 back = extF80_to_f128(r);
        if (!f128_eq(back, v) && f128_lt(tf_abs(v), tf_abs(back))) *extra |= FSW_C1;
    }
    return r;
}

/* A result that underflowed to zero in binary128 itself (a true value too small even for it): inexact, and the rounding
 * mode decides between a zero and the smallest denormal of the sign (RU positive, RD negative: rounded up in magnitude, C1). */
static void tf_underflowed(extFloat80_t* r, uint16_t* extra) {
    softfloat_exceptionFlags |= softfloat_flag_inexact;
    const int neg = (r->signExp & 0x8000) != 0;
    if ((softfloat_roundingMode == softfloat_round_max && !neg) || (softfloat_roundingMode == softfloat_round_min && neg)) {
        r->signif = 1;
        *extra |= FSW_C1;
    }
}

/* sin and cos of r, |r| <= pi/4 (a little more is fine): Taylor series. */
static void tf_sincos_kernel(F128 r, F128* s, F128* c) {
    const F128 r2 = tf_mul(r, r);
    F128 ts = r, tc = tf_i(1), ss = r, sc = tf_i(1);
    for (int k = 1; k <= 18; ++k) {
        tc = tf_div(tf_mul(tc, r2), tf_i((int64_t)(2 * k - 1) * (2 * k)));
        ts = tf_div(tf_mul(ts, r2), tf_i((int64_t)(2 * k) * (2 * k + 1)));
        if (k & 1) { sc = tf_sub(sc, tc); ss = tf_sub(ss, ts); }
        else { sc = tf_add(sc, tc); ss = tf_add(ss, ts); }
        /* The terms fall by more than half each step: once both are 2^-120 of their sums (7 bits beyond the 113 of binary128)
         * the rest cannot change the result; a small |r| (the common case) needs 6 or so terms of the 18. */
        if (tf_exp(tc) + 120 < tf_exp(sc) && tf_exp(ts) + 120 < tf_exp(ss)) break;
    }
    *s = ss;
    *c = sc;
}

/* x = n * pi/2 + r with |r| <= pi/4, for a finite |x| < 2^63 (the sign is the caller's): returns n mod 4
 * and r in binary128. Like the hardware, the reduction uses pi/2 to 66 bits (0xC90FDAA22168C234C / 2^67),
 * not the exact value: that is what makes the results near multiples of pi/2 differ from the true ones,
 * and so from a correctly rounded library, in the same way. The product n * pi/2 and the subtractions are
 * exact in binary128. */
static int tf_reduce(extFloat80_t a, F128* r) {
    static const F128 pi_2_66 = {VP_F128_PI_2_66};
    const F128 x = tf_abs(tf_f80(a));
    if (f128_lt(x, TF_PI_4)) { *r = x; return 0; }
    const int64_t n = f128_to_i64(tf_div(x, TF_PI_2), softfloat_round_near_even, false);
    F128 v = x;
    if (n >> 32) v = tf_sub(v, tf_scale(tf_mul(tf_i(n >> 32), pi_2_66), 32));
    *r = tf_sub(v, tf_mul(tf_i(n & INT64_C(0xFFFFFFFF)), pi_2_66));
    return (int)(n & 3);
}

/* sin, cos of a finite extended-precision value with |a| < 2^63. */
static void tf_sincos(extFloat80_t a, F128* sin_out, F128* cos_out) {
    const TfSave sv = tf_enter();
    F128 r, s, c;
    const int n = tf_reduce(a, &r);
    tf_sincos_kernel(r, &s, &c);
    F128 sn, cs;
    switch (n) {
    case 0: sn = s; cs = c; break;
    case 1: sn = c; cs = tf_neg(s); break;
    case 2: sn = tf_neg(s); cs = tf_neg(c); break;
    default: sn = tf_neg(c); cs = s; break;
    }
    if (a.signExp & 0x8000) sn = tf_neg(sn);
    *sin_out = sn;
    *cos_out = cs;
    tf_leave(sv);
}

/* 2 * (z + z^3/3 + z^5/5 + ...) = ln((1 + z) / (1 - z)), |z| <= 0.172. */
static F128 tf_ln_series(F128 z) {
    const F128 z2 = tf_mul(z, z);
    F128 sum = z, pw = z;
    for (int k = 1; k <= 26; ++k) {
        pw = tf_mul(pw, z2);
        sum = tf_add(sum, tf_div(pw, tf_i(2 * k + 1)));
    }
    return tf_scale(sum, 1);
}
/* log2 of a positive finite value. */
static F128 tf_log2(F128 v) {
    int e = 0;
    if (!(v.v[1] & UINT64_C(0x7FFF000000000000))) { /* an extended denormal is subnormal in binary128 too */
        v = tf_mul(v, tf_scale(tf_i(1), 120));
        e = -120;
    }
    e += (int)((v.v[1] >> 48) & 0x7FFF) - 16383;
    v.v[1] = (v.v[1] & ~UINT64_C(0x7FFF000000000000)) | (UINT64_C(16383) << 48); /* m in [1, 2) */
    if (((v.v[1] >> 40) & 0xFF) >= 0x6A) { v = tf_scale(v, -1); ++e; } /* m in [0.707, 1.414) */
    const F128 one = tf_i(1);
    const F128 z = tf_div(tf_sub(v, one), tf_add(v, one));
    return tf_add(tf_i(e), tf_mul(tf_ln_series(z), TF_LOG2E));
}
/* y * log2(x) and y * log2(1 + x) for x > 0 / x > -1 (finite, y finite or infinite). */
static F128 tf_yl2x(extFloat80_t y, extFloat80_t x) {
    const TfSave sv = tf_enter();
    const F128 v = tf_mul(tf_f80(y), tf_log2(tf_f80(x)));
    tf_leave(sv);
    return v;
}
static F128 tf_yl2xp1(extFloat80_t y, extFloat80_t x) {
    const TfSave sv = tf_enter();
    const F128 xx = tf_f80(x), one = tf_i(1);
    if (!(xx.v[1] & UINT64_C(0x7FFF000000000000))) { /* denormal x: log2(1 + x) = x log2(e) to far more than 64 bits, computed on 2^120 x
                                                      * (binary128 is subnormal there) and scaled back with y */
        const F128 l = tf_mul(tf_scale(xx, 120), TF_LOG2E);
        const F128 v = tf_mul(tf_mul(tf_f80(y), tf_scale(one, -120)), l);
        tf_leave(sv);
        return v;
    }
    /* |x| < 1/4: ln(1 + x) = 2 atanh(x / (2 + x)) keeps the relative precision of a small x */
    const F128 l = f128_lt(tf_abs(xx), tf_scale(one, -2)) ? tf_mul(tf_ln_series(tf_div(xx, tf_add(tf_i(2), xx))), TF_LOG2E) : tf_log2(tf_add(one, xx));
    const F128 v = tf_mul(tf_f80(y), l);
    tf_leave(sv);
    return v;
}

/* 2^x - 1 */
static F128 tf_exp2m1(extFloat80_t a) {
    const TfSave sv = tf_enter();
    const F128 x = tf_f80(a);
    int n = (int)f128_to_i32(x, softfloat_round_near_even, false);
    if (n > 15000) n = 15000; /* the caller passes |x| <= 1; this only keeps tf_scale inside its range */
    else if (n < -15000) n = -15000;
    const F128 y = tf_mul(tf_sub(x, tf_i(n)), TF_LN2);
    F128 sum = y, term = y;
    for (int k = 2; k <= 28; ++k) {
        term = tf_div(tf_mul(term, y), tf_i(k));
        sum = tf_add(sum, term);
    }
    const F128 v = n == 0 ? sum : tf_add(tf_scale(sum, n), tf_sub(tf_scale(tf_i(1), n), tf_i(1)));
    tf_leave(sv);
    return v;
}

/* atan of t >= 0: t -> t / (1 + sqrt(1 + t^2)) three times (atan doubles each time back), series. */
static F128 tf_atan_pos(F128 t) {
    const F128 one = tf_i(1);
    int inv = 0;
    if (f128_lt(one, t)) { t = tf_div(one, t); inv = 1; }
    for (int i = 0; i < 3; ++i) t = tf_div(t, tf_add(one, f128_sqrt(tf_add(one, tf_mul(t, t)))));
    const F128 t2 = tf_mul(t, t);
    F128 sum = t, pw = t;
    for (int k = 1; k <= 22; ++k) {
        pw = tf_mul(pw, t2);
        const F128 term = tf_div(pw, tf_i(2 * k + 1));
        sum = (k & 1) ? tf_sub(sum, term) : tf_add(sum, term);
    }
    sum = tf_scale(sum, 3);
    return inv ? tf_sub(TF_PI_2, sum) : sum;
}
/* atan2(y, x) for finite non-zero y and x. */
static F128 tf_atan2(extFloat80_t y, extFloat80_t x) {
    const TfSave sv = tf_enter();
    F128 t = tf_atan_pos(tf_div(tf_abs(tf_f80(y)), tf_abs(tf_f80(x))));
    if (x.signExp & 0x8000) t = tf_sub(TF_PI, t);
    if (y.signExp & 0x8000) t = tf_neg(t);
    tf_leave(sv);
    return t;
}
/* atan2 with a zero or an infinity among the operands (not NaN). */
static F128 tf_atan2_special(extFloat80_t y, extFloat80_t x) {
    const int ys = (y.signExp & 0x8000) != 0, xs = (x.signExp & 0x8000) != 0;
    F128 t;
    if (is_zero(y)) t = xs ? TF_PI : tf_i(0);
    else if (is_zero(x)) t = TF_PI_2;
    else if (is_inf(y)) t = is_inf(x) ? (xs ? TF_3PI_4 : TF_PI_4) : TF_PI_2;
    else t = xs ? TF_PI : tf_i(0); /* x infinite, y finite */
    return ys ? tf_neg(t) : t;
}

/* The transcendental instructions. Special values are settled here; the rest goes to the functions above. */
enum { VP_X87_SIN, VP_X87_COS, VP_X87_SINCOS, VP_X87_PTAN, VP_X87_PATAN, VP_X87_F2XM1, VP_X87_YL2X, VP_X87_YL2XP1 };
void vp_x87_transcendental(VpCpu* c, int op) {
    vp_x87_begin(c);
    extFloat80_t a;
    int ok = st_get(c, 0, &a);
    c->fsw &= ~FSW_C1;
    if (op <= VP_X87_PTAN) c->fsw &= ~FSW_C2; /* the others leave C2 alone */
    uint16_t extra = 0;
    if (ok && is_unsupported(a)) { softfloat_exceptionFlags |= softfloat_flag_invalid; a = vp_f80_indefinite; ok = 0; }
    switch (op) {
    case VP_X87_SIN: case VP_X87_COS: case VP_X87_SINCOS: case VP_X87_PTAN: {
        if ((op == VP_X87_SINCOS || op == VP_X87_PTAN) && ((c->ftag >> ((c->ftop - 1) & 7)) & 1)) {
            st_replace_push(c, a, a); /* stack overflow: both indefinite, nothing computed */
            break;
        }
        extFloat80_t r1 = a, r2 = a;
        if (ok && is_nan(a)) r1 = r2 = extF80_add(a, a); /* quiets (IE for a signalling NaN) */
        else if (ok && is_inf(a)) { softfloat_exceptionFlags |= softfloat_flag_invalid; r1 = r2 = vp_f80_indefinite; }
        else if (ok && (a.signExp & 0x7FFF) >= 0x3FFF + 63) { c->fsw |= FSW_C2; break; } /* out of range: unchanged */
        else if (ok) {
            if (is_denormal(a)) extra |= FSW_DE;
            if (is_zero(a) || (a.signExp & 0x7FFF) < 0x3FFF - 33) {
                /* |x| < 2^-33: sin x = tan x = x and cos x = 1 to 64 bits */
                r1 = a;
                r2 = vp_f80_one;
                if (!is_zero(a)) softfloat_exceptionFlags |= softfloat_flag_inexact;
                if (op == VP_X87_COS) r1 = vp_f80_one;
            } else {
                F128 s, co;
                tf_sincos(a, &s, &co);
                const TfSave sv = tf_enter();
                const F128 tn = op == VP_X87_PTAN ? tf_div(s, co) : s; /* the quotient is correctly rounded in binary128 */
                tf_leave(sv);
/* C1 is the "rounded up" of the result the hardware reports it for: fsin, fcos, fptan the first, fsincos the cosine */
                r1 = tf_round(op == VP_X87_COS ? co : tn, op == VP_X87_SINCOS ? NULL : &extra);
                r2 = op == VP_X87_PTAN ? vp_f80_one : tf_round(co, op == VP_X87_SINCOS ? &extra : NULL);
            }
        }
        if (op == VP_X87_SIN || op == VP_X87_COS) st_set(c, 0, r1);
        else st_replace_push(c, r1, r2);
        break;
    }
    case VP_X87_F2XM1: {
        extFloat80_t r = a;
        if (ok && is_nan(a)) r = extF80_add(a, a);
        else if (ok && is_inf(a)) { if (a.signExp & 0x8000) { r.signif = UINT64_C(0x8000000000000000); r.signExp = 0xBFFF; } } /* -inf: -1 */
        else if (ok && !is_zero(a)) {
            if (is_denormal(a)) extra |= FSW_DE;
            /* The instruction is defined for |x| <= 1. Outside it the Intel hardware (measured from 1.5 to
             * +-16383.6, and just above 1) returns the operand unchanged, flagging only inexact. */
            if ((a.signExp & 0x7FFF) > 0x3FFF || ((a.signExp & 0x7FFF) == 0x3FFF && (a.signif << 1))) softfloat_exceptionFlags |= softfloat_flag_inexact;
            else r = tf_round(tf_exp2m1(a), &extra);
        }
        st_set(c, 0, r);
        break;
    }
    case VP_X87_PATAN: case VP_X87_YL2X: case VP_X87_YL2XP1: {
        extFloat80_t b, r;
        ok &= st_get(c, 1, &b);
        if (ok && is_unsupported(b)) { softfloat_exceptionFlags |= softfloat_flag_invalid; ok = 0; }
        if (!ok) r = vp_f80_indefinite;
        else if (is_nan(a) || is_nan(b)) r = extF80_add(b, a);
        else {
            if (is_denormal(a) || is_denormal(b)) extra |= FSW_DE;
            const int ysign = (b.signExp & 0x8000) != 0;
            if (op == VP_X87_PATAN) {
                int ex = (int)(a.signExp & 0x7FFF), ey = (int)(b.signExp & 0x7FFF);
                if (is_zero(a) || is_zero(b) || is_inf(a) || is_inf(b)) r = tf_round(tf_atan2_special(b, a), &extra);
                else {
                    if (ex == 0) ex = 1; /* denormals */
                    if (ey == 0) ey = 1;
                    if (ex - ey > 60 && !(a.signExp & 0x8000)) r = extF80_div(b, a); /* atan(y/x) = y/x to 64 bits */
                    else if (ey - ex > 66) { /* +-pi/2 to 64 bits (the rounded constant: up, so C1) */
                        r.signif = UINT64_C(0xC90FDAA22168C235); r.signExp = (uint16_t)(0x3FFF | (ysign << 15));
                        extra |= FSW_C1;
                    }
                    else r = tf_round(tf_atan2(b, a), &extra);
                }
                /* exact: zero y with a positive x, and a finite y with x = +inf (an exact 0) */
                if ((!is_zero(b) || (a.signExp & 0x8000)) && !(is_inf(a) && !(a.signExp & 0x8000) && !is_inf(b))) softfloat_exceptionFlags |= softfloat_flag_inexact;
            } else if (op == VP_X87_YL2X) {
                if (((a.signExp & 0x8000) && !is_zero(a)) || (is_zero(a) && is_zero(b)) || (is_inf(a) && is_zero(b)) ||
                    (a.signExp == 0x3FFF && a.signif == UINT64_C(0x8000000000000000) && is_inf(b))) {
                    softfloat_exceptionFlags |= softfloat_flag_invalid;
                    r = vp_f80_indefinite;
                } else if (is_zero(a)) { /* log2(0) = -inf: y * -inf (zero-divide only for a finite y) */
                    if (!is_inf(b)) softfloat_exceptionFlags |= softfloat_flag_infinite;
                    r.signif = UINT64_C(0x8000000000000000); r.signExp = (uint16_t)(ysign ? 0x7FFF : 0xFFFF);
                } else if (is_inf(a)) {
                    r.signif = UINT64_C(0x8000000000000000); r.signExp = (uint16_t)(ysign ? 0xFFFF : 0x7FFF);
                } else if (is_inf(b)) { /* infinite y, x neither 1 nor 0 nor infinite: the sign of y log2(x) */
                    r.signif = UINT64_C(0x8000000000000000); r.signExp = (uint16_t)(0x7FFF | ((ysign ^ ((a.signExp & 0x7FFF) < 0x3FFF)) << 15));
                } else {
                    r = tf_round(tf_yl2x(b, a), &extra);
                    /* y * log2(x) underflowed in binary128 itself: still a tiny inexact result (x = 1 gives an exact 0) */
                    if (is_zero(r) && !is_zero(b) && !(a.signExp == 0x3FFF && a.signif == UINT64_C(0x8000000000000000))) tf_underflowed(&r, &extra);
                }
            } else { /* fyl2xp1: y log2(1 + x) */
                if (is_zero(a)) {
                    if (is_inf(b)) { softfloat_exceptionFlags |= softfloat_flag_invalid; r = vp_f80_indefinite; }
                    else { r.signif = 0; r.signExp = (uint16_t)((a.signExp ^ b.signExp) & 0x8000); }
                } else if (is_inf(a)) {
                    if (is_zero(b) || (a.signExp & 0x8000)) { softfloat_exceptionFlags |= softfloat_flag_invalid; r = vp_f80_indefinite; } /* 0 * inf, and x = -inf */
                    else { r.signif = UINT64_C(0x8000000000000000); r.signExp = (uint16_t)(ysign ? 0xFFFF : 0x7FFF); }
                } else if ((a.signExp & 0x8000) && (a.signExp & 0x7FFF) >= 0x3FFF) { /* x <= -1: outside the domain; the hardware gives x, inexact
                                                                                      * (for a zero or infinite y: y with the sign flipped, exact) */
                    if (is_zero(b) || is_inf(b)) { r = b; r.signExp ^= 0x8000; }
                    else { r = a; softfloat_exceptionFlags |= softfloat_flag_inexact; }
                } else if (is_inf(b)) { /* the sign of y log2(1 + x) (x is not 0 here) */
                    r.signif = UINT64_C(0x8000000000000000); r.signExp = (uint16_t)(0x7FFF | (((a.signExp ^ b.signExp) & 0x8000)));
                } else {
                    r = tf_round(tf_yl2xp1(b, a), &extra);
                    if (is_zero(r) && !is_zero(b)) tf_underflowed(&r, &extra); /* underflowed in binary128 itself */
                }
            }
        }
        st_set(c, 1, r);
        st_pop(c);
        break;
    }
    }
    /* A tiny result (denormal or zero) that is inexact also underflows. */
    if (softfloat_exceptionFlags & softfloat_flag_inexact) {
        for (int i = 0; i < (op == VP_X87_SINCOS || op == VP_X87_PTAN ? 2 : 1); ++i)
            if (!(c->st[phys(c, i)].se & 0x7FFF)) softfloat_exceptionFlags |= softfloat_flag_underflow;
    }
    vp_x87_end(c, extra);
}

/* fbld / fbstp: 18-digit packed BCD. */
void vp_x87_fbld(VpCpu* c, uint64_t a) {
    vp_x87_begin(c);
    int64_t v = 0;
    for (int i = 8; i >= 0; --i) {
        const uint8_t b = vp_ld8(a + (uint64_t)i);
        v = v * 100 + (b >> 4) * 10 + (b & 15);
    }
    extFloat80_t r = i64_to_extF80(v);
    if (vp_ld8(a + 9) & 0x80) r.signExp |= 0x8000;
    st_push(c, r);
    vp_x87_end(c, 0);
}
void vp_x87_fbstp(VpCpu* c, uint64_t a) {
    vp_x87_begin(c);
    extFloat80_t v;
    const int ok = st_get(c, 0, &v);
    c->fsw &= ~FSW_C1;
    int_fast64_t n = ok ? extF80_to_i64(v, softfloat_roundingMode, true) : 0;
    uint16_t extra = 0;
    if (ok && !(softfloat_exceptionFlags & softfloat_flag_invalid) && n != extF80_to_i64(v, softfloat_round_minMag, false)) extra |= FSW_C1; /* rounded up */
    uint8_t out[10];
    if (!ok || is_unsupported(v) || (softfloat_exceptionFlags & softfloat_flag_invalid) || n > INT64_C(999999999999999999) || n < -INT64_C(999999999999999999)) {
        softfloat_exceptionFlags = softfloat_flag_invalid;
        extra = 0;
        memset(out, 0, 7); out[7] = 0xC0; out[8] = 0xFF; out[9] = 0xFF; /* the BCD indefinite */
    } else {
        const int neg = n < 0 || (n == 0 && (v.signExp & 0x8000));
        uint64_t u = n < 0 ? (uint64_t)(-n) : (uint64_t)n;
        for (int i = 0; i < 9; ++i) { const unsigned lo = u % 10; u /= 10; const unsigned hi = u % 10; u /= 10; out[i] = (uint8_t)(hi << 4 | lo); }
        out[9] = neg ? 0x80 : 0;
    }
    for (int i = 0; i < 10; ++i) vp_st8(a + (uint64_t)i, out[i]);
    st_pop(c);
    vp_x87_end(c, extra);
}
