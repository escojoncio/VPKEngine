// Soft-float IEEE-754 binary64 for GCN (gfx7) semantics.
//
// This file is compiled twice from the same text: as C++ (vp4ir::f64, the reference used by the VPIR
// evaluator and the tests) and as Metal Shading Language (emitted once per shader that contains a VPIR
// `f64` op, see vp4ir::f64MslSource()). It is therefore written in the common subset of both: no
// references, no templates, no operator overloading, u32/u64 integer arithmetic only. The prelude of each
// language defines u32, u64, i32, uint2 (members x = low dword, y = high dword), VP_CLZ64 and VP_TABLE.
//
// Values are binary64 bit patterns. `mode` is the GCN MODE register:
//   bits 3:2   FP_ROUND for double/half: 0 nearest even, 1 +infinity, 2 -infinity, 3 toward zero
//   bit  6     FP_DENORM double: 1 = denormal inputs are used (0 = flushed to a zero of the same sign)
//   bit  7     FP_DENORM double: 1 = denormal results are kept (0 = flushed to a zero of the same sign)
//   bit  8     DX10_CLAMP (NaN clamps to 0), bit 9 IEEE (min/max quiet signalling NaNs)
// The default mode 0x1c0 is round to nearest even, denormals kept.
//
// What is matched to the hardware, and what is not (documented in docs/isa-f64.md):
//   * add, mul, fma, ldexp, conversions, floor/ceil/trunc/rndne/fract, cmp, class: exact IEEE results in
//     every rounding mode (the hardware's fma is a single rounding too).
//   * rcp, sqrt: correctly rounded in every mode. The hardware's v_rcp_f64 / v_sqrt_f64 are approximations
//     good to about 1 ulp that compilers refine with fma and v_div_fixup; the refined results are identical.
//   * rsq: error below 0.502 ulp (correctly rounded in all but about 1 in 10000 cases), the hardware is not exact either.
//   * Invalid operations return the hardware's default NaN 0xFFF8000000000000 (negative quiet NaN); a NaN
//     operand is returned quieted (first NaN operand in source order).
//   * Exceptions and the exception flags are not modelled.

#define VP_U64(h, l) ((u64(h) << 32) | u64(l))
#define VPF_SIGN VP_U64(0x80000000u, 0u)
#define VPF_EXP VP_U64(0x7ff00000u, 0u)
#define VPF_FRAC VP_U64(0x000fffffu, 0xffffffffu)
#define VPF_QUIET VP_U64(0x00080000u, 0u)
#define VPF_DEFAULT_NAN VP_U64(0xfff80000u, 0u)
#define VPF_MAXFIN VP_U64(0x7fefffffu, 0xffffffffu)
#define VPF_ONE VP_U64(0x3ff00000u, 0u)

struct vpf_u128 {
    u64 hi;
    u64 lo;
};

inline vpf_u128 vpf_mk(u64 hi, u64 lo) {
    vpf_u128 r;
    r.hi = hi;
    r.lo = lo;
    return r;
}
inline vpf_u128 vpf_add128(vpf_u128 a, vpf_u128 b) {
    vpf_u128 r;
    r.lo = a.lo + b.lo;
    r.hi = a.hi + b.hi + (r.lo < a.lo ? u64(1) : u64(0));
    return r;
}
inline vpf_u128 vpf_sub128(vpf_u128 a, vpf_u128 b) {
    vpf_u128 r;
    r.lo = a.lo - b.lo;
    r.hi = a.hi - b.hi - (a.lo < b.lo ? u64(1) : u64(0));
    return r;
}
inline bool vpf_ge128(vpf_u128 a, vpf_u128 b) { return a.hi > b.hi || (a.hi == b.hi && a.lo >= b.lo); }
inline bool vpf_zero128(vpf_u128 a) { return (a.hi | a.lo) == u64(0); }
// n in 0..127
inline vpf_u128 vpf_shl128(vpf_u128 a, u32 n) {
    if (n == 0u) return a;
    if (n >= 64u) return vpf_mk(a.lo << (n - 64u), u64(0));
    return vpf_mk((a.hi << n) | (a.lo >> (64u - n)), a.lo << n);
}
// Right shift that ORs everything shifted out into the lowest bit (sticky).
inline vpf_u128 vpf_shr128j(vpf_u128 a, u32 n) {
    if (n == 0u) return a;
    if (n >= 128u) return vpf_mk(u64(0), (a.hi | a.lo) != u64(0) ? u64(1) : u64(0));
    if (n >= 64u) {
        u32 k = n - 64u;
        u64 lost = a.lo;
        if (k > 0u) lost |= a.hi & ((u64(1) << k) - u64(1));
        return vpf_mk(u64(0), (a.hi >> k) | (lost != u64(0) ? u64(1) : u64(0)));
    }
    u64 lost = a.lo & ((u64(1) << n) - u64(1));
    return vpf_mk(a.hi >> n, (a.lo >> n) | (a.hi << (64u - n)) | (lost != u64(0) ? u64(1) : u64(0)));
}
inline u32 vpf_clz128(vpf_u128 a) { return a.hi != u64(0) ? u32(VP_CLZ64(a.hi)) : 64u + u32(VP_CLZ64(a.lo)); }

inline vpf_u128 vpf_mul64(u64 a, u64 b) {
    u64 m32 = u64(0xffffffffu);
    u64 a0 = a & m32, a1 = a >> 32, b0 = b & m32, b1 = b >> 32;
    u64 p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    u64 mid = (p00 >> 32) + (p01 & m32) + (p10 & m32);
    return vpf_mk(p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32), (p00 & m32) | (mid << 32));
}

// ---- Classification and unpacking --------------------------------------------------------------

inline bool vpf_isnan(u64 a) { return (a & ~VPF_SIGN) > VPF_EXP; }
inline bool vpf_issnan(u64 a) { return vpf_isnan(a) && (a & VPF_QUIET) == u64(0); }
inline bool vpf_isinf(u64 a) { return (a & ~VPF_SIGN) == VPF_EXP; }
inline bool vpf_iszero(u64 a) { return (a & ~VPF_SIGN) == u64(0); }
inline u64 vpf_quiet(u64 a) { return a | VPF_QUIET; }
inline u32 vpf_round(u32 mode) { return (mode >> 2) & 3u; }

// Denormal inputs read as zero when MODE says so.
inline u64 vpf_in(u64 a, u32 mode) {
    if ((mode & 0x40u) == 0u && (a & VPF_EXP) == u64(0)) return a & VPF_SIGN;
    return a;
}

struct vpf_un {
    u32 cls;  // 0 zero, 1 finite non-zero, 2 infinity, 3 NaN
    u32 sign;
    i32 e;    // value = m * 2^(e - 63)
    u64 m;    // bit 63 set for finite non-zero
};

inline vpf_un vpf_unpack(u64 a) {
    vpf_un r;
    r.sign = u32(a >> 63);
    r.e = 0;
    r.m = u64(0);
    u32 ex = u32(a >> 52) & 0x7ffu;
    u64 f = a & VPF_FRAC;
    if (ex == 0x7ffu) {
        r.cls = f != u64(0) ? 3u : 2u;
        return r;
    }
    if (ex == 0u) {
        if (f == u64(0)) {
            r.cls = 0u;
            return r;
        }
        u32 sh = u32(VP_CLZ64(f));
        r.cls = 1u;
        r.m = f << sh;
        r.e = -1011 - i32(sh);
        return r;
    }
    r.cls = 1u;
    r.m = (f | (u64(1) << 52)) << 11;
    r.e = i32(ex) - 1023;
    return r;
}

// ---- Rounding and packing ------------------------------------------------------------------------

// Rounds (-1)^sign * (m + sticky * epsilon) * 2^(e - 63), m with bit 63 set, to a binary format with
// `fracBits` fraction bits (52 / 23) and exponent bias `bias`; returns the bit pattern (f32 in the low
// dword). Denormal results are kept when denormOut, else flushed to a zero of the same sign.
inline u64 vpf_pack(u32 sign, i32 e, u64 m, u32 sticky, u32 rm, u32 fracBits, i32 bias, bool denormOut) {
    i32 maxExp = (2 << (fracBits == 52u ? 10 : 7)) - 1;  // all-ones exponent field
    i32 be = e + bias;
    u64 signBit = u64(sign) << (fracBits == 52u ? 63u : 31u);
    u64 infBits = u64(maxExp) << fracBits;
    u64 maxFin = infBits - u64(1);
    bool towardInf = (rm == 1u && sign == 0u) || (rm == 2u && sign == 1u);
    if (be >= maxExp) {
        bool toInf = rm == 0u || towardInf;
        return signBit | (toInf ? infBits : maxFin);
    }
    u32 shift = 63u - fracBits;
    if (be <= 0) {
        i32 extra = 1 - be;
        shift = extra > 70 ? 140u : shift + u32(extra);
    }
    u64 kept = u64(0);
    bool roundBit = false;
    bool stk = sticky != 0u;
    if (shift <= 63u) {
        kept = m >> shift;
        u64 low = m & ((u64(1) << shift) - u64(1));
        u64 halfBit = u64(1) << (shift - 1u);
        roundBit = (low & halfBit) != u64(0);
        stk = stk || (low & (halfBit - u64(1))) != u64(0);
    } else if (shift == 64u) {
        roundBit = (m >> 63) != u64(0);
        stk = stk || (m << 1) != u64(0);
    } else {
        stk = true;
    }
    bool inexact = roundBit || stk;
    bool inc = false;
    if (rm == 0u) inc = roundBit && (stk || (kept & u64(1)) != u64(0));
    else if (towardInf) inc = inexact;
    kept += inc ? u64(1) : u64(0);
    u64 r = (be > 0 ? (u64(be - 1) << fracBits) : u64(0)) + kept;
    if (r >= infBits) {
        bool toInf = rm == 0u || towardInf;
        return signBit | (toInf ? infBits : maxFin);
    }
    if (!denormOut && (r >> fracBits) == u64(0)) r = u64(0);
    return signBit | r;
}

inline u64 vpf_pack64(u32 sign, i32 e, u64 m, u32 sticky, u32 mode) {
    return vpf_pack(sign, e, m, sticky, vpf_round(mode), 52u, 1023, (mode & 0x80u) != 0u);
}

// ---- Arithmetic ----------------------------------------------------------------------------------

// Zero result of an exact cancellation x + (-x): -0 only when rounding toward -infinity.
inline u64 vpf_cancel_zero(u32 mode) { return vpf_round(mode) == 2u ? VPF_SIGN : u64(0); }

// Denormal results are flushed to a zero of the same sign when MODE says so.
inline u64 vpf_out(u64 r, u32 mode) {
    if ((mode & 0x80u) == 0u && (r & VPF_EXP) == u64(0)) return r & VPF_SIGN;
    return r;
}

inline u64 vpf_add_in(u64 a, u64 b, u32 mode) {
    if (vpf_isnan(a)) return vpf_quiet(a);
    if (vpf_isnan(b)) return vpf_quiet(b);
    if (vpf_isinf(a)) {
        if (vpf_isinf(b) && ((a ^ b) & VPF_SIGN) != u64(0)) return VPF_DEFAULT_NAN;
        return a;
    }
    if (vpf_isinf(b)) return b;
    if (vpf_iszero(a) && vpf_iszero(b)) {
        if (((a ^ b) & VPF_SIGN) == u64(0)) return a;
        return vpf_cancel_zero(mode);
    }
    if (vpf_iszero(a)) return vpf_out(b, mode);
    if (vpf_iszero(b)) return vpf_out(a, mode);
    vpf_un x = vpf_unpack(a), y = vpf_unpack(b);
    if (y.e > x.e || (y.e == x.e && y.m > x.m)) {
        vpf_un t = x;
        x = y;
        y = t;
    }
    u32 d = u32(x.e - y.e);
    u64 X = x.m >> 1, Y = y.m >> 1;
    if (d > 0u) {
        if (d >= 63u) Y = u64(1);
        else Y = (Y >> d) | ((Y & ((u64(1) << d) - u64(1))) != u64(0) ? u64(1) : u64(0));
    }
    u64 S = x.sign == y.sign ? X + Y : X - Y;
    if (S == u64(0)) return vpf_cancel_zero(mode);
    u32 lz = u32(VP_CLZ64(S));
    return vpf_pack64(x.sign, x.e + 1 - i32(lz), S << lz, 0u, mode);
}

inline u64 vpf_add(u64 a, u64 b, u32 mode) { return vpf_add_in(vpf_in(a, mode), vpf_in(b, mode), mode); }

inline u64 vpf_mul(u64 a, u64 b, u32 mode) {
    a = vpf_in(a, mode);
    b = vpf_in(b, mode);
    if (vpf_isnan(a)) return vpf_quiet(a);
    if (vpf_isnan(b)) return vpf_quiet(b);
    u64 sign = (a ^ b) & VPF_SIGN;
    if (vpf_isinf(a) || vpf_isinf(b)) {
        if (vpf_iszero(a) || vpf_iszero(b)) return VPF_DEFAULT_NAN;
        return sign | VPF_EXP;
    }
    if (vpf_iszero(a) || vpf_iszero(b)) return sign;
    vpf_un x = vpf_unpack(a), y = vpf_unpack(b);
    vpf_u128 p = vpf_mul64(x.m, y.m);
    i32 e = x.e + y.e;
    if ((p.hi >> 63) != u64(0)) e += 1;
    else p = vpf_shl128(p, 1u);
    return vpf_pack64(u32(sign >> 63), e, p.hi, p.lo != u64(0) ? 1u : 0u, mode);
}

inline u64 vpf_ldexp(u64 a, u32 k, u32 mode);

// a * b + c rounded once, then scaled by 2^sc inside the same rounding (v_div_fmas).
inline u64 vpf_fma_in(u64 a, u64 b, u64 c, u32 mode, i32 sc) {
    if (vpf_isnan(a)) return vpf_quiet(a);
    if (vpf_isnan(b)) return vpf_quiet(b);
    if (vpf_isnan(c)) return vpf_quiet(c);
    u64 psign = (a ^ b) & VPF_SIGN;
    bool pinf = vpf_isinf(a) || vpf_isinf(b);
    if (pinf) {
        if (vpf_iszero(a) || vpf_iszero(b)) return VPF_DEFAULT_NAN;
        if (vpf_isinf(c) && ((c ^ psign) & VPF_SIGN) != u64(0)) return VPF_DEFAULT_NAN;
        return psign | VPF_EXP;
    }
    if (vpf_isinf(c)) return c;
    bool pzero = vpf_iszero(a) || vpf_iszero(b);
    if (pzero) {
        if (vpf_iszero(c)) {
            if (((psign ^ c) & VPF_SIGN) == u64(0)) return c;
            return vpf_cancel_zero(mode);
        }
        return sc == 0 ? vpf_out(c, mode) : vpf_ldexp(c, u32(sc), mode);
    }
    vpf_un x = vpf_unpack(a), y = vpf_unpack(b);
    vpf_u128 P = vpf_mul64(x.m, y.m);
    i32 ep = x.e + y.e + 1;
    if ((P.hi >> 63) == u64(0)) {
        P = vpf_shl128(P, 1u);
        ep -= 1;
    }
    // (P is now the product with bit 127 set: value = P * 2^(ep - 127))
    u32 psg = u32(psign >> 63);
    if (vpf_iszero(c)) return vpf_pack64(psg, ep + sc, P.hi, P.lo != u64(0) ? 1u : 0u, mode);
    vpf_un z = vpf_unpack(c);
    // Both as words with bit 126 as their top bit: value = word * 2^(E - 126).
    vpf_u128 A = vpf_mk(P.hi >> 1, (P.hi << 63) | (P.lo >> 1));
    vpf_u128 B = vpf_mk(z.m >> 1, z.m << 63);
    i32 EA = ep, EB = z.e;
    u32 sa = psg, sb = z.sign;
    if (EB > EA || (EB == EA && !vpf_ge128(A, B))) {
        vpf_u128 t = A;
        A = B;
        B = t;
        i32 te = EA;
        EA = EB;
        EB = te;
        u32 ts = sa;
        sa = sb;
        sb = ts;
    }
    u32 d = u32(EA - EB);
    if (d > 0u) B = vpf_shr128j(B, d > 200u ? 200u : d);
    vpf_u128 S = sa == sb ? vpf_add128(A, B) : vpf_sub128(A, B);
    if (vpf_zero128(S)) return vpf_cancel_zero(mode);
    u32 lz = vpf_clz128(S);  // bit position p = 127 - lz; e = EA + p - 126
    S = vpf_shl128(S, lz);
    return vpf_pack64(sa, EA + 1 - i32(lz) + sc, S.hi, S.lo != u64(0) ? 1u : 0u, mode);
}

inline u64 vpf_fma(u64 a, u64 b, u64 c, u32 mode) {
    return vpf_fma_in(vpf_in(a, mode), vpf_in(b, mode), vpf_in(c, mode), mode, 0);
}

// Quotient of two mantissas with the exponent difference: shared by division and rsq. num/den finite
// non-zero (m with bit 63 set).
inline u64 vpf_divmant(u32 sign, i32 en, u64 mn, i32 ed, u64 md, u32 stickyD, u32 mode) {
    u64 R = mn;
    u64 q0 = u64(0);
    if (R >= md) {
        R -= md;
        q0 = u64(1);
    }
    u64 F = u64(0);
    VP_NOUNROLL
    for (u32 i = 0u; i < 64u; ++i) {
        u64 carry = R >> 63;
        R <<= 1;
        F <<= 1;
        if (carry != u64(0) || R >= md) {
            R -= md;
            F |= u64(1);
        }
    }
    u32 sticky = (R != u64(0) || stickyD != 0u) ? 1u : 0u;
    i32 e = en - ed;
    u64 m;
    if (q0 != u64(0)) {
        sticky |= (F & u64(1)) != u64(0) ? 1u : 0u;
        m = (u64(1) << 63) | (F >> 1);
    } else {
        m = F;
        e -= 1;
    }
    return vpf_pack64(sign, e, m, sticky, mode);
}

inline u64 vpf_div_in(u64 a, u64 b, u32 mode) {
    if (vpf_isnan(a)) return vpf_quiet(a);
    if (vpf_isnan(b)) return vpf_quiet(b);
    u64 sign = (a ^ b) & VPF_SIGN;
    if (vpf_isinf(a)) return vpf_isinf(b) ? VPF_DEFAULT_NAN : (sign | VPF_EXP);
    if (vpf_isinf(b)) return sign;
    if (vpf_iszero(b)) return vpf_iszero(a) ? VPF_DEFAULT_NAN : (sign | VPF_EXP);
    if (vpf_iszero(a)) return sign;
    vpf_un x = vpf_unpack(a), y = vpf_unpack(b);
    return vpf_divmant(u32(sign >> 63), x.e, x.m, y.e, y.m, 0u, mode);
}
inline u64 vpf_div(u64 a, u64 b, u32 mode) { return vpf_div_in(vpf_in(a, mode), vpf_in(b, mode), mode); }

// Square root of a finite positive value: root mantissa (bit 63 set), exponent, sticky (inexact).
struct vpf_root {
    u64 m;
    i32 e;
    u32 sticky;
};
inline vpf_root vpf_isqrt(vpf_un x) {
    vpf_u128 N;
    vpf_root r;
    if ((x.e & 1) != 0) {
        N = vpf_mk(x.m, u64(0));
        r.e = (x.e - 1) / 2;
    } else {
        N = vpf_mk(x.m >> 1, x.m << 63);
        r.e = x.e / 2;
    }
    vpf_u128 rem = N;
    vpf_u128 root = vpf_mk(u64(0), u64(0));
    vpf_u128 bit = vpf_mk(u64(1) << 62, u64(0));  // 2^126
    VP_NOUNROLL
    for (u32 i = 0u; i < 64u; ++i) {
        vpf_u128 t = vpf_add128(root, bit);
        if (vpf_ge128(rem, t)) {
            rem = vpf_sub128(rem, t);
            root = vpf_add128(vpf_mk(root.hi >> 1, (root.lo >> 1) | (root.hi << 63)), bit);
        } else {
            root = vpf_mk(root.hi >> 1, (root.lo >> 1) | (root.hi << 63));
        }
        bit = vpf_mk(bit.hi >> 2, (bit.lo >> 2) | (bit.hi << 62));
    }
    r.m = root.lo;
    r.sticky = vpf_zero128(rem) ? 0u : 1u;
    return r;
}

inline u64 vpf_sqrt(u64 a, u32 mode) {
    a = vpf_in(a, mode);
    if (vpf_isnan(a)) return vpf_quiet(a);
    if (vpf_iszero(a)) return a;
    if ((a & VPF_SIGN) != u64(0)) return VPF_DEFAULT_NAN;
    if (vpf_isinf(a)) return a;
    vpf_un x = vpf_unpack(a);
    vpf_root r = vpf_isqrt(x);
    return vpf_pack64(0u, r.e, r.m, r.sticky, mode);
}

inline u64 vpf_rcp(u64 a, u32 mode) { return vpf_div(VPF_ONE, a, mode); }

inline u64 vpf_rsq(u64 a, u32 mode) {
    a = vpf_in(a, mode);
    if (vpf_isnan(a)) return vpf_quiet(a);
    if (vpf_iszero(a)) return (a & VPF_SIGN) | VPF_EXP;
    if ((a & VPF_SIGN) != u64(0)) return VPF_DEFAULT_NAN;
    if (vpf_isinf(a)) return u64(0);
    vpf_un x = vpf_unpack(a);
    vpf_root r = vpf_isqrt(x);
    return vpf_divmant(0u, 0, u64(1) << 63, r.e, r.m, r.sticky, mode);
}

// *_clamp_f64: infinities become the largest finite value of the same sign.
inline u64 vpf_clamp_inf(u64 r) { return vpf_isinf(r) ? ((r & VPF_SIGN) | VPF_MAXFIN) : r; }

inline u64 vpf_ldexp(u64 a, u32 k, u32 mode) {
    a = vpf_in(a, mode);
    if (vpf_isnan(a)) return vpf_quiet(a);
    if (vpf_isinf(a) || vpf_iszero(a)) return a;
    i32 ki = i32(k);
    if (ki > 4500) ki = 4500;
    if (ki < -4500) ki = -4500;
    vpf_un x = vpf_unpack(a);
    return vpf_pack64(x.sign, x.e + ki, x.m, 0u, mode);
}

// ---- min / max / compare ---------------------------------------------------------------------------

inline bool vpf_lt(u64 a, u64 b) {  // neither is NaN
    if (((a | b) & ~VPF_SIGN) == u64(0)) return false;
    u64 sa = a >> 63, sb = b >> 63;
    if (sa != sb) return sa > sb;
    return sa == u64(0) ? a < b : a > b;
}
inline bool vpf_eq(u64 a, u64 b) { return a == b || ((a | b) & ~VPF_SIGN) == u64(0); }

inline u64 vpf_min(u64 a, u64 b, u32 mode) {
    a = vpf_in(a, mode);
    b = vpf_in(b, mode);
    bool ieee = (mode & 0x200u) != 0u;
    if (ieee && vpf_issnan(a)) return vpf_quiet(a);
    if (ieee && vpf_issnan(b)) return vpf_quiet(b);
    if (vpf_isnan(a)) return b;
    if (vpf_isnan(b)) return a;
    if (((a | b) & ~VPF_SIGN) == u64(0)) return (a & VPF_SIGN) != u64(0) ? a : b;  // -0 orders below +0 (as the f32 forms)
    return vpf_lt(a, b) ? a : b;
}
inline u64 vpf_max(u64 a, u64 b, u32 mode) {
    a = vpf_in(a, mode);
    b = vpf_in(b, mode);
    bool ieee = (mode & 0x200u) != 0u;
    if (ieee && vpf_issnan(a)) return vpf_quiet(a);
    if (ieee && vpf_issnan(b)) return vpf_quiet(b);
    if (vpf_isnan(a)) return b;
    if (vpf_isnan(b)) return a;
    if (((a | b) & ~VPF_SIGN) == u64(0)) return (a & VPF_SIGN) != u64(0) ? b : a;  // +0 orders above -0
    return !vpf_lt(a, b) ? a : b;  // a >= b ? a : b
}

// v_cmp condition codes, in opcode order: 0 f, 1 lt, 2 eq, 3 le, 4 gt, 5 lg, 6 ge, 7 o, 8 u, 9 nge, 10 nlg,
// 11 ngt, 12 nle, 13 neq, 14 nlt, 15 tru.
inline bool vpf_cmp(u64 a, u64 b, u32 cond, u32 mode) {
    a = vpf_in(a, mode);
    b = vpf_in(b, mode);
    bool anyNan = vpf_isnan(a) || vpf_isnan(b);
    bool lt = !anyNan && vpf_lt(a, b), gt = !anyNan && vpf_lt(b, a), eq = !anyNan && vpf_eq(a, b);
    switch (cond) {
        case 0u: return false;
        case 1u: return lt;
        case 2u: return eq;
        case 3u: return lt || eq;
        case 4u: return gt;
        case 5u: return lt || gt;
        case 6u: return gt || eq;
        case 7u: return !anyNan;
        case 8u: return anyNan;
        case 9u: return !(gt || eq);
        case 10u: return !(lt || gt);
        case 11u: return !gt;
        case 12u: return !(lt || eq);
        case 13u: return !eq;
        case 14u: return !lt;
        default: return true;
    }
}

// v_cmp_class: bit 0 signalling NaN, 1 quiet NaN, 2 -inf, 3 -normal, 4 -denormal, 5 -0, 6 +0, 7 +denormal,
// 8 +normal, 9 +inf. No denormal flushing.
inline bool vpf_class(u64 a, u32 mask) {
    bool isNeg = (a & VPF_SIGN) != u64(0);
    u64 mag = a & ~VPF_SIGN;
    u32 cls;
    if (mag > VPF_EXP) cls = (mag & VPF_QUIET) != u64(0) ? 1u : 0u;
    else if (mag == VPF_EXP) cls = isNeg ? 2u : 9u;
    else if (mag == u64(0)) cls = isNeg ? 5u : 6u;
    else if (mag < (u64(1) << 52)) cls = isNeg ? 4u : 7u;
    else cls = isNeg ? 3u : 8u;
    return ((mask >> cls) & 1u) != 0u;
}

// ---- Rounding to integral values -----------------------------------------------------------------

inline u64 vpf_trunc(u64 a, u32 mode) {
    a = vpf_in(a, mode);
    if (vpf_isnan(a)) return vpf_quiet(a);
    u32 ex = u32(a >> 52) & 0x7ffu;
    if (ex < 1023u) return a & VPF_SIGN;
    if (ex >= 1075u) return a;
    u64 mask = VPF_FRAC >> (ex - 1023u);
    return a & ~mask;
}
inline u64 vpf_floor(u64 a, u32 mode) {
    u64 t = vpf_trunc(a, mode);
    if (vpf_isnan(t)) return t;
    a = vpf_in(a, mode);
    if ((a & VPF_SIGN) != u64(0) && t != a) return vpf_add_in(t, VPF_SIGN | VPF_ONE, 0x1c0u);
    return t;
}
inline u64 vpf_ceil(u64 a, u32 mode) {
    u64 t = vpf_trunc(a, mode);
    if (vpf_isnan(t)) return t;
    a = vpf_in(a, mode);
    if ((a & VPF_SIGN) == u64(0) && t != a) return vpf_add_in(t, VPF_ONE, 0x1c0u);
    return t;
}
inline u64 vpf_rndne(u64 a, u32 mode) {
    a = vpf_in(a, mode);
    if (vpf_isnan(a)) return vpf_quiet(a);
    u32 ex = u32(a >> 52) & 0x7ffu;
    if (ex >= 1075u) return a;
    u64 magic = VP_U64(0x43300000u, 0u);  // 2^52
    u64 r = vpf_add_in(vpf_add_in(a & ~VPF_SIGN, magic, 0x1c0u), VPF_SIGN | magic, 0x1c0u);
    return r | (a & VPF_SIGN);
}
inline u64 vpf_fract(u64 a, u32 mode) {
    a = vpf_in(a, mode);
    if (vpf_isnan(a)) return vpf_quiet(a);
    u64 fl = vpf_floor(a, mode);
    u64 r = vpf_add_in(a, fl ^ VPF_SIGN, mode);
    // The hardware clamps the result just below 1.0 (x - floor(x) rounds up to 1.0 for tiny negatives).
    u64 kBelowOne = VP_U64(0x3fefffffu, 0xffffffffu);
    if (!vpf_isnan(r) && r > kBelowOne && r < VPF_SIGN) return kBelowOne;
    return r;
}

inline u64 vpf_frexp_mant(u64 a, u32 mode) {
    a = vpf_in(a, mode);
    if (vpf_isnan(a) || vpf_isinf(a) || vpf_iszero(a)) return a;
    vpf_un x = vpf_unpack(a);
    return (u64(x.sign) << 63) | (u64(1022) << 52) | ((x.m << 1) >> 12);
}
inline u32 vpf_frexp_exp(u64 a, u32 mode) {
    a = vpf_in(a, mode);
    if (vpf_isnan(a) || vpf_isinf(a) || vpf_iszero(a)) return 0u;
    vpf_un x = vpf_unpack(a);
    return u32(x.e + 1);
}

// ---- Conversions -----------------------------------------------------------------------------------

inline u64 vpf_from_u32(u32 v, u32 mode) {
    if (v == 0u) return u64(0);
    u32 lz = u32(VP_CLZ64(u64(v)));
    return vpf_pack64(0u, 63 - i32(lz), u64(v) << lz, 0u, mode | 0x80u);
}
inline u64 vpf_from_i32(u32 v, u32 mode) {
    if ((v >> 31) == 0u) return vpf_from_u32(v, mode);
    u64 r = vpf_from_u32(0u - v, mode);
    return r | VPF_SIGN;
}
inline u64 vpf_from_f32(u32 f) {
    u64 sign = u64(f >> 31) << 63;
    u32 ex = (f >> 23) & 0xffu, fr = f & 0x7fffffu;
    if (ex == 0xffu) {
        if (fr == 0u) return sign | VPF_EXP;
        return sign | VPF_EXP | VPF_QUIET | (u64(fr) << 29);
    }
    if (ex == 0u) {
        if (fr == 0u) return sign;
        u32 sh = u32(VP_CLZ64(u64(fr)));  // normalise: value = fr * 2^-149
        return vpf_pack(f >> 31, 63 - i32(sh) - 149, u64(fr) << sh, 0u, 0u, 52u, 1023, true);
    }
    return sign | (u64(ex + 896u) << 52) | (u64(fr) << 29);
}
// f64 -> f32 with the single-precision rounding mode `rm` and denormal result handling.
inline u32 vpf_to_f32(u64 a, u32 mode, u32 rm, bool denormOut) {
    a = vpf_in(a, mode);
    u32 sign = u32(a >> 63);
    if (vpf_isnan(a)) return (sign << 31) | 0x7fc00000u | u32((a & VPF_FRAC) >> 29);
    if (vpf_isinf(a)) return (sign << 31) | 0x7f800000u;
    if (vpf_iszero(a)) return sign << 31;
    vpf_un x = vpf_unpack(a);
    return u32(vpf_pack(sign, x.e, x.m, 0u, rm, 23u, 127, denormOut));
}
inline u32 vpf_to_i32(u64 a, u32 mode) {
    a = vpf_in(a, mode);
    if (vpf_isnan(a) || vpf_iszero(a)) return 0u;
    bool isNeg = (a & VPF_SIGN) != u64(0);
    if (vpf_isinf(a)) return isNeg ? 0x80000000u : 0x7fffffffu;
    vpf_un x = vpf_unpack(a);
    if (x.e < 0) return 0u;
    if (x.e >= 31) return isNeg ? 0x80000000u : 0x7fffffffu;
    u32 mag = u32(x.m >> (63u - u32(x.e)));
    return isNeg ? 0u - mag : mag;
}
inline u32 vpf_to_u32(u64 a, u32 mode) {
    a = vpf_in(a, mode);
    if (vpf_isnan(a) || vpf_iszero(a)) return 0u;
    if ((a & VPF_SIGN) != u64(0)) return 0u;
    if (vpf_isinf(a)) return 0xffffffffu;
    vpf_un x = vpf_unpack(a);
    if (x.e < 0) return 0u;
    if (x.e >= 32) return 0xffffffffu;
    return u32(x.m >> (63u - u32(x.e)));
}

// ---- Division helpers (v_div_scale / v_div_fmas / v_div_fixup) -------------------------------------

inline bool vpf_isden(u64 a) { return (a & VPF_EXP) == u64(0) && (a & VPF_FRAC) != u64(0); }
inline i32 vpf_expf(u64 a) { return i32((a >> 52) & u64(0x7ff)); }

// Returns the scaled operand; *vcc (via the high bit of the second result) says whether it was scaled.
struct vpf_scale {
    u64 d;
    bool vcc;
};
inline vpf_scale vpf_div_scale(u64 s0, u64 s1, u64 s2) {
    vpf_scale r;
    r.d = s0;
    r.vcc = false;
    // Cases in the order of the ISA pseudo code (the denormal tests use exact division).
    if (vpf_iszero(s2) || vpf_iszero(s1)) {
        r.d = VPF_DEFAULT_NAN;
        return r;
    }
    bool eq1 = vpf_eq(s0, s1), eq2 = vpf_eq(s0, s2);
    if (vpf_expf(s2) - vpf_expf(s1) >= 768) {
        r.vcc = true;
        if (eq1) r.d = vpf_ldexp(s0, 128u, 0x1c0u);
        return r;
    }
    if (vpf_isden(s1)) {
        r.d = vpf_ldexp(s0, 128u, 0x1c0u);
        return r;
    }
    bool rdn = vpf_isden(vpf_div_in(VPF_ONE, s1, 0x1c0u)), qdn = vpf_isden(vpf_div_in(s2, s1, 0x1c0u));
    if (rdn && qdn) {
        // (denominator too large for a normal reciprocal, quotient denormal: shrink the denominator, leave the
        // numerator, the quotient estimate grows by 2^128 and v_div_fmas takes it back)
        r.vcc = true;
        if (eq1) r.d = vpf_ldexp(s0, 0u - 128u, 0x1c0u);
        return r;
    }
    if (rdn) {
        r.d = vpf_ldexp(s0, 0u - 128u, 0x1c0u);
        return r;
    }
    if (qdn) {
        r.vcc = true;
        if (eq2) r.d = vpf_ldexp(s0, 128u, 0x1c0u);
        return r;
    }
    if (vpf_expf(s2) <= 53) r.d = vpf_ldexp(s0, 128u, 0x1c0u);
    return r;
}

inline u64 vpf_div_fmas(u64 a, u64 b, u64 c, bool vcc, u32 mode) {
    i32 sc = 0;
    if (vcc) sc = vpf_expf(c) > 1023 ? 128 : -128;
    return vpf_fma_in(vpf_in(a, mode), vpf_in(b, mode), vpf_in(c, mode), mode, sc);
}

inline u64 vpf_div_fixup(u64 q, u64 den, u64 num) {
    u64 sign = (den ^ num) & VPF_SIGN;
    bool nanDen = vpf_isnan(den), nanNum = vpf_isnan(num);
    bool zeroDen = vpf_iszero(den), zeroNum = vpf_iszero(num);
    bool infDen = vpf_isinf(den), infNum = vpf_isinf(num);
    u64 r = (q & ~VPF_SIGN) | sign;
    if (infDen || zeroNum) r = sign;
    if (zeroDen || infNum) r = sign | VPF_EXP;
    if (infDen && infNum) r = VPF_DEFAULT_NAN;
    if (zeroDen && zeroNum) r = VPF_DEFAULT_NAN;
    // (an estimate that overflowed in the division sequence arrives as NaN: infinity or zero by the operands'
    // exponents)
    if (vpf_isnan(q) && !vpf_isinf(num) && !vpf_isinf(den) && !zeroNum && !zeroDen)
        r = sign | (vpf_expf(num) >= vpf_expf(den) ? VPF_EXP : u64(0));
    if (nanDen) r = den | VPF_QUIET;
    if (nanNum) r = num | VPF_QUIET;
    return r;
}

// clamp modifier: [0, 1], NaN to 0 with DX10_CLAMP (else NaN stays).
inline u64 vpf_saturate(u64 a, u32 mode) {
    if (vpf_isnan(a)) return (mode & 0x100u) != 0u ? u64(0) : vpf_quiet(a);
    if (vpf_lt(a, u64(0)) || vpf_iszero(a)) return u64(0);
    if (vpf_lt(VPF_ONE, a)) return VPF_ONE;
    return a;
}

// ---- v_trig_preop_f64 ------------------------------------------------------------------------------
// 2/pi as a binary fraction, most significant bits first, 32 per word (VPF_TWO_OVER_PI_WORDS words,
// then zero padding).
VP_TABLE u32 vpf_two_over_pi[VPF_TWO_OVER_PI_WORDS] = {VPF_TWO_OVER_PI_DATA};

inline u32 vpf_tw(u32 i) { return i < u32(VPF_TWO_OVER_PI_WORDS) ? vpf_two_over_pi[i] : 0u; }

// 53 bits of 2/pi starting at fraction bit index 53 * s1[4:0] (further shifted for large s0 exponents),
// scaled by the position of those bits.
inline u64 vpf_trig_preop(u64 a, u32 idx, u32 mode) {
    a = vpf_in(a, mode);
    if (vpf_isnan(a)) return vpf_quiet(a);
    i32 ex = vpf_expf(a);
    u32 shift = (idx & 31u) * 53u;
    if (ex > 1077) shift += u32(ex - 1077);
    u32 w = shift >> 5, off = shift & 31u;
    u64 hi = (u64(vpf_tw(w)) << 32) | u64(vpf_tw(w + 1u));
    u64 win = off == 0u ? hi : ((hi << off) | (u64(vpf_tw(w + 2u)) >> (32u - off)));
    u64 bits = win >> 11;  // the top 53 bits
    i32 scale = -53 - i32(shift);
    if (ex >= 1968) scale += 128;
    if (bits == u64(0)) return u64(0);
    u32 lz = u32(VP_CLZ64(bits));
    return vpf_pack64(0u, 63 + scale - i32(lz), bits << lz, 0u, mode | 0x80u);
}

// ---- Entry points of the VPIR `f64` op (one per vp4ir::F64Fn) ------------------------------------------
// Operands and results travel as uint2 {low dword, high dword}; `mode` is the MODE register.

inline u64 vpf_pk(uint2 v) { return (u64(v.y) << 32) | u64(v.x); }
inline uint2 vpf_up(u64 v) {
    uint2 r;
    r.x = u32(v);
    r.y = u32(v >> 32);
    return r;
}
inline uint2 vp_f64_add(uint2 a, uint2 b, u32 mode) { return vpf_up(vpf_add(vpf_pk(a), vpf_pk(b), mode)); }
inline uint2 vp_f64_mul(uint2 a, uint2 b, u32 mode) { return vpf_up(vpf_mul(vpf_pk(a), vpf_pk(b), mode)); }
inline uint2 vp_f64_fma(uint2 a, uint2 b, uint2 c, u32 mode) {
    return vpf_up(vpf_fma(vpf_pk(a), vpf_pk(b), vpf_pk(c), mode));
}
inline uint2 vp_f64_min(uint2 a, uint2 b, u32 mode) { return vpf_up(vpf_min(vpf_pk(a), vpf_pk(b), mode)); }
inline uint2 vp_f64_max(uint2 a, uint2 b, u32 mode) { return vpf_up(vpf_max(vpf_pk(a), vpf_pk(b), mode)); }
inline uint2 vp_f64_ldexp(uint2 a, u32 k, u32 mode) { return vpf_up(vpf_ldexp(vpf_pk(a), k, mode)); }
inline uint2 vp_f64_rcp(uint2 a, u32 mode) { return vpf_up(vpf_rcp(vpf_pk(a), mode)); }
inline uint2 vp_f64_rcp_clamp(uint2 a, u32 mode) { return vpf_up(vpf_clamp_inf(vpf_rcp(vpf_pk(a), mode))); }
inline uint2 vp_f64_rsq(uint2 a, u32 mode) { return vpf_up(vpf_rsq(vpf_pk(a), mode)); }
inline uint2 vp_f64_rsq_clamp(uint2 a, u32 mode) { return vpf_up(vpf_clamp_inf(vpf_rsq(vpf_pk(a), mode))); }
inline uint2 vp_f64_sqrt(uint2 a, u32 mode) { return vpf_up(vpf_sqrt(vpf_pk(a), mode)); }
inline uint2 vp_f64_trunc(uint2 a, u32 mode) { return vpf_up(vpf_trunc(vpf_pk(a), mode)); }
inline uint2 vp_f64_ceil(uint2 a, u32 mode) { return vpf_up(vpf_ceil(vpf_pk(a), mode)); }
inline uint2 vp_f64_rndne(uint2 a, u32 mode) { return vpf_up(vpf_rndne(vpf_pk(a), mode)); }
inline uint2 vp_f64_floor(uint2 a, u32 mode) { return vpf_up(vpf_floor(vpf_pk(a), mode)); }
inline uint2 vp_f64_fract(uint2 a, u32 mode) { return vpf_up(vpf_fract(vpf_pk(a), mode)); }
inline uint2 vp_f64_frexp_mant(uint2 a, u32 mode) { return vpf_up(vpf_frexp_mant(vpf_pk(a), mode)); }
inline u32 vp_f64_frexp_exp(uint2 a, u32 mode) { return vpf_frexp_exp(vpf_pk(a), mode); }
inline uint2 vp_f64_from_i32(u32 v, u32 mode) { return vpf_up(vpf_from_i32(v, mode)); }
inline uint2 vp_f64_from_u32(u32 v, u32 mode) { return vpf_up(vpf_from_u32(v, mode)); }
inline uint2 vp_f64_from_f32(u32 v, u32) { return vpf_up(vpf_from_f32(v)); }
// f32 result: rounding by MODE[1:0], denormal results kept by MODE bit 5.
inline u32 vp_f64_to_f32(uint2 a, u32 mode) { return vpf_to_f32(vpf_pk(a), mode, mode & 3u, (mode & 0x20u) != 0u); }
inline u32 vp_f64_to_i32(uint2 a, u32 mode) { return vpf_to_i32(vpf_pk(a), mode); }
inline u32 vp_f64_to_u32(uint2 a, u32 mode) { return vpf_to_u32(vpf_pk(a), mode); }
inline bool vp_f64_cmp(uint2 a, uint2 b, u32 cond, u32 mode) { return vpf_cmp(vpf_pk(a), vpf_pk(b), cond, mode); }
inline bool vp_f64_class(uint2 a, u32 mask, u32) { return vpf_class(vpf_pk(a), mask); }
// {result low, result high, VCC (0 / 1)}
inline uint3 vp_f64_div_scale(uint2 s0, uint2 s1, uint2 s2, u32) {
    vpf_scale r = vpf_div_scale(vpf_pk(s0), vpf_pk(s1), vpf_pk(s2));
    uint3 o;
    o.x = u32(r.d);
    o.y = u32(r.d >> 32);
    o.z = r.vcc ? 1u : 0u;
    return o;
}
inline uint2 vp_f64_div_fmas(uint2 a, uint2 b, uint2 c, u32 vcc, u32 mode) {
    return vpf_up(vpf_div_fmas(vpf_pk(a), vpf_pk(b), vpf_pk(c), vcc != 0u, mode));
}
inline uint2 vp_f64_div_fixup(uint2 q, uint2 den, uint2 num, u32) {
    return vpf_up(vpf_div_fixup(vpf_pk(q), vpf_pk(den), vpf_pk(num)));
}
inline uint2 vp_f64_trig_preop(uint2 a, u32 idx, u32 mode) { return vpf_up(vpf_trig_preop(vpf_pk(a), idx, mode)); }
inline uint2 vp_f64_sat(uint2 a, u32 mode) { return vpf_up(vpf_saturate(vpf_in(vpf_pk(a), mode), mode)); }
