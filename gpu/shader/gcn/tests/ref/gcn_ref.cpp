// Reference interpreter for GCN 2 SALU / VALU instructions. See gcn_ref.h for the modelling
// choices. Semantics follow AMD's "Sea Islands Series Instruction Set Architecture" reference.
#include "gcn_ref.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

namespace vpgcnref {
namespace {

using vpgcn::Encoding;
using vpgcn::Instruction;
using vpgcn::Op;
using vpgcn::Operand;
using vpgcn::OperandKind;

using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i32 = int32_t;
using i64 = int64_t;

float asF(u32 b) { return std::bit_cast<float>(b); }
u32 asU(float f) { return std::bit_cast<u32>(f); }
double asD(u64 b) { return std::bit_cast<double>(b); }
u64 asU(double d) { return std::bit_cast<u64>(d); }

// ---- Raw VOPC opcode of an Op (condition and type are encoded in it) ------------------------------
struct OpcEntry {
    Op op;
    u16 opc;
};
constexpr OpcEntry kVopc[] = {
#define VPGCNREF_X(name, opc, sig) {Op::name, opc},
    VPGCN_VOPC(VPGCNREF_X)
#undef VPGCNREF_X
};
int vopcOpcode(Op op) {
    for (const auto& e : kVopc)
        if (e.op == op) return e.opc;
    return -1;
}

// ---- Bit helpers ---------------------------------------------------------------------------------
template <class U>
U bfeU(U x, unsigned off, unsigned width) {
    constexpr unsigned N = sizeof(U) * 8;
    if (width == 0) return 0;
    if (off + width >= N) return U(x >> off);
    return U((x >> off) & ((U(1) << width) - 1));
}
template <class U>
U bfeS(U x, unsigned off, unsigned width) {
    using S = std::make_signed_t<U>;
    constexpr unsigned N = sizeof(U) * 8;
    if (width == 0) return 0;
    if (off + width >= N) return U(S(x) >> off);
    U f = U((x >> off) & ((U(1) << width) - 1));
    return U(S(U(f << (N - width))) >> (N - width));
}
u32 brev32(u32 x) {
    u32 r = 0;
    for (int i = 0; i < 32; ++i) r |= ((x >> i) & 1u) << (31 - i);
    return r;
}
u64 brev64(u64 x) { return (u64(brev32(u32(x))) << 32) | brev32(u32(x >> 32)); }
// First bit set counting from the LSB, -1 if none.
template <class U>
u32 ffbl(U x) {
    return x ? u32(std::countr_zero(x)) : 0xFFFFFFFFu;
}
// First bit set counting from the MSB, -1 if none.
template <class U>
u32 ffbh(U x) {
    return x ? u32(std::countl_zero(x)) : 0xFFFFFFFFu;
}
// First bit differing from the sign bit, counting from the MSB, -1 if all bits equal.
template <class U>
u32 ffbhSigned(U x) {
    constexpr unsigned N = sizeof(U) * 8;
    U y = (x >> (N - 1)) ? U(~x) : x;
    return y ? u32(std::countl_zero(y)) : 0xFFFFFFFFu;
}
u32 sext24(u32 x) { return u32(i32(x << 8) >> 8); }

// ---- Float helpers -------------------------------------------------------------------------------
// IEEE minNum / maxNum, -0 < +0.
template <class F>
F fminIeee(F a, F b) {
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    if (a < b) return a;
    if (b < a) return b;
    return std::signbit(a) ? a : b;
}
template <class F>
F fmaxIeee(F a, F b) {
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    if (a > b) return a;
    if (b > a) return b;
    return std::signbit(a) ? b : a;
}
// DX9 multiply: 0 times anything (inf, NaN included) is +0.
float mulLegacy(float a, float b) {
    if (a == 0.0f || b == 0.0f) return 0.0f;
    return a * b;
}
float med3f(float a, float b, float c) {
    if (std::isnan(a) || std::isnan(b) || std::isnan(c)) return fminIeee(fminIeee(a, b), c);
    float mx = fmaxIeee(fmaxIeee(a, b), c);
    if (mx == a) return fmaxIeee(b, c);
    if (mx == b) return fmaxIeee(a, c);
    return fmaxIeee(a, b);
}
template <class T>
T med3i(T a, T b, T c) {
    T mx = std::max(std::max(a, b), c);
    if (mx == a) return std::max(b, c);
    if (mx == b) return std::max(a, c);
    return std::max(a, b);
}

// f32 -> f16 bits, round to nearest even or toward zero. NaN stays a (quiet) NaN.
u16 f32ToF16(u32 x, bool rtz) {
    u32 sign = (x >> 16) & 0x8000u;
    u32 exp = (x >> 23) & 0xFFu;
    u32 man = x & 0x7FFFFFu;
    if (exp == 0xFF) return u16(man ? (sign | 0x7E00u | (man >> 13)) : (sign | 0x7C00u));
    if (exp == 0) return u16(sign);  // f32 denormals are far below the f16 denormal range
    int e = int(exp) - 127 + 15;
    if (e >= 31) return u16(rtz ? (sign | 0x7BFFu) : (sign | 0x7C00u));
    u32 m = man | 0x800000u;
    unsigned shift = e <= 0 ? unsigned(13 + 1 - e) : 13u;
    if (shift >= 25) return u16(sign);
    u32 q = m >> shift;
    u32 rem = m & ((1u << shift) - 1);
    u32 half = 1u << (shift - 1);
    if (!rtz && (rem > half || (rem == half && (q & 1)))) ++q;
    if (e <= 0) return u16(sign | q);
    u32 r = (u32(e) << 10) + q - 0x400u;
    if (r >= 0x7C00u) r = rtz ? 0x7BFFu : 0x7C00u;
    return u16(sign | r);
}
u32 f16ToF32(u16 h) {
    u32 sign = u32(h & 0x8000u) << 16;
    u32 exp = (h >> 10) & 0x1Fu;
    u32 man = h & 0x3FFu;
    if (exp == 0x1F) return sign | 0x7F800000u | (man << 13);
    if (exp == 0) {
        if (man == 0) return sign;
        // Denormal: value = man * 2^-24.
        float f = float(man) * 0x1p-24f;
        return sign | asU(f);
    }
    return sign | ((exp - 15 + 127) << 23) | (man << 13);
}

u32 cvtU32(double v) {
    if (std::isnan(v) || v <= 0.0) return 0;
    if (v >= 4294967296.0) return 0xFFFFFFFFu;
    return u32(v);
}
u32 cvtI32(double v) {
    if (std::isnan(v)) return 0;
    if (v <= -2147483648.0) return 0x80000000u;
    if (v >= 2147483647.0) return 0x7FFFFFFFu;
    return u32(i32(v));
}
// Float -> normalized 16-bit (round to nearest even), NaN -> 0. The scale is an f32 multiply, as the
// GPU's ALU does it (in double a product just off a .5 tie rounds the other way).
u32 snorm16(float f) {
    if (std::isnan(f)) return 0;
    float d = std::clamp(f, -1.0f, 1.0f);
    return u32(i32(std::nearbyint(d * 32767.0f))) & 0xFFFFu;
}
u32 unorm16(float f) {
    if (std::isnan(f)) return 0;
    float d = std::clamp(f, 0.0f, 1.0f);
    return u32(std::nearbyint(d * 65535.0f));
}
u32 u8OfF32(float f) {
    if (std::isnan(f)) return 0;
    double d = std::clamp(double(f), 0.0, 255.0);
    return u32(std::nearbyint(d));
}

template <class F>
F fract(F x) {
    if (std::isinf(x)) return std::numeric_limits<F>::quiet_NaN();
    F r = x - std::floor(x);
    F one_minus = std::nextafter(F(1), F(0));
    return std::isnan(r) ? r : std::min(r, one_minus);
}

// IEEE class bits of v_cmp_class: 0 sNaN, 1 qNaN, 2 -inf, 3 -normal, 4 -denormal, 5 -0, 6 +0,
// 7 +denormal, 8 +normal, 9 +inf.
// Classified from the bit pattern so a signalling NaN is never touched by an FP operation.
template <class U, unsigned kManBits>
unsigned classBit(U x) {
    constexpr unsigned N = sizeof(U) * 8;
    constexpr U kManMask = (U(1) << kManBits) - 1;
    constexpr U kExpMask = (U(1) << (N - 1 - kManBits)) - 1;
    bool neg = (x >> (N - 1)) != 0;
    U exp = (x >> kManBits) & kExpMask, man = x & kManMask;
    if (exp == kExpMask) {
        if (man == 0) return neg ? 2 : 9;
        return ((man >> (kManBits - 1)) & 1) ? 1 : 0;
    }
    if (exp != 0) return neg ? 3 : 8;
    if (man != 0) return neg ? 4 : 7;
    return neg ? 5 : 6;
}

template <class F>
bool cmpFloat(unsigned cond, F a, F b) {
    bool un = std::isnan(a) || std::isnan(b);
    switch (cond & 15) {
        case 0: return false;
        case 1: return a < b;
        case 2: return a == b;
        case 3: return a <= b;
        case 4: return a > b;
        case 5: return !un && a != b;  // lg
        case 6: return a >= b;
        case 7: return !un;            // o
        case 8: return un;             // u
        case 9: return !(a >= b);      // nge
        case 10: return un || a == b;  // nlg
        case 11: return !(a > b);      // ngt
        case 12: return !(a <= b);     // nle
        case 13: return !(a == b);     // neq
        case 14: return !(a < b);      // nlt
        default: return true;          // tru
    }
}
template <class T>
bool cmpInt(unsigned cond, T a, T b) {
    switch (cond & 7) {
        case 0: return false;
        case 1: return a < b;
        case 2: return a == b;
        case 3: return a <= b;
        case 4: return a > b;
        case 5: return a != b;
        case 6: return a >= b;
        default: return true;
    }
}

struct Cube {
    float id, sc, tc, ma;
};
Cube cube(float x, float y, float z) {
    Cube c{};
    float ax = std::fabs(x), ay = std::fabs(y), az = std::fabs(z);
    if (az >= ax && az >= ay) {
        bool n = z < 0.0f;
        c.id = n ? 5.0f : 4.0f;
        c.sc = n ? -x : x;
        c.tc = -y;
        c.ma = z;
    } else if (ay >= ax) {
        bool n = y < 0.0f;
        c.id = n ? 3.0f : 2.0f;
        c.sc = x;
        c.tc = n ? -z : z;
        c.ma = y;
    } else {
        bool n = x < 0.0f;
        c.id = n ? 1.0f : 0.0f;
        c.sc = n ? z : -z;
        c.tc = -y;
        c.ma = x;
    }
    c.ma = c.ma + c.ma;
    return c;
}

// ---- Executor ------------------------------------------------------------------------------------
class Exec {
public:
    Exec(const Instruction& in, const Wave& r, Wave& w, std::string& err) : in_(in), r_(r), w_(w), err_(err) {}

    bool run();

private:
    const Instruction& in_;
    const Wave& r_;  // state before the instruction (all reads)
    Wave& w_;        // state after (all writes)
    std::string& err_;
    bool ok_ = true;

    void fail(const std::string& m) {
        if (ok_) err_ = m;
        ok_ = false;
    }
    bool unsupported() {
        fail("unsupported " + std::string(vpgcn::mnemonic(in_.op)));
        return false;
    }
    bool isVop3() const { return in_.encoding == Encoding::VOP3; }

    // ---- scalar operand access ----
    u32 s32(const Operand& o) {
        switch (o.kind) {
            case OperandKind::Sgpr:
                if (o.index >= kNumSgprs) break;
                return r_.s[o.index];
            case OperandKind::Const:
            case OperandKind::Literal:
                return o.value;
            case OperandKind::Special:
                switch (o.code) {
                    case 106: return u32(r_.vcc);
                    case 107: return u32(r_.vcc >> 32);
                    case 124: return r_.m0;
                    case 126: return u32(r_.exec);
                    case 127: return u32(r_.exec >> 32);
                    case 251: return r_.vcc == 0;
                    case 252: return r_.exec == 0;
                    case 253: return r_.scc;
                    default: break;
                }
                break;
            default: break;
        }
        fail("unsupported scalar operand (kind " + std::to_string(int(o.kind)) + ", code " +
             std::to_string(o.code) + ")");
        return 0;
    }
    // 64-bit scalar operand. Inline constants: integers sign-extended, floats as f64. Literals:
    // high dword for f64, zero-extended for integers.
    u64 s64(const Operand& o, bool f64) {
        switch (o.kind) {
            case OperandKind::Sgpr:
                if (o.index + 1u >= kNumSgprs) break;
                return u64(r_.s[o.index]) | (u64(r_.s[o.index + 1]) << 32);
            case OperandKind::Const:
                if (o.code == 248) return 0x3FC45F306DC9C882ull;  // 1/(2*pi) as f64
                if (o.isFloat) return asU(double(asF(o.value)));
                return u64(i64(i32(o.value)));
            case OperandKind::Literal:
                return f64 ? u64(o.value) << 32 : u64(o.value);
            case OperandKind::Special:
                if (o.code == 106) return r_.vcc;
                if (o.code == 126) return r_.exec;
                if (o.code >= 251 && o.code <= 253) return s32(o);
                break;
            default: break;
        }
        fail("unsupported 64-bit scalar operand (kind " + std::to_string(int(o.kind)) + ", code " +
             std::to_string(o.code) + ")");
        return 0;
    }
    void ws32(const Operand& o, u32 v) {
        switch (o.kind) {
            case OperandKind::Sgpr:
                if (o.index >= kNumSgprs) break;
                w_.s[o.index] = v;
                return;
            case OperandKind::Special:
                switch (o.code) {
                    case 106: w_.vcc = (w_.vcc & ~0xFFFFFFFFull) | v; return;
                    case 107: w_.vcc = (w_.vcc & 0xFFFFFFFFull) | (u64(v) << 32); return;
                    case 124: w_.m0 = v; return;
                    case 126: w_.exec = (w_.exec & ~0xFFFFFFFFull) | v; return;
                    case 127: w_.exec = (w_.exec & 0xFFFFFFFFull) | (u64(v) << 32); return;
                    default: break;
                }
                break;
            default: break;
        }
        fail("unsupported scalar destination (code " + std::to_string(o.code) + ")");
    }
    void ws64(const Operand& o, u64 v) {
        switch (o.kind) {
            case OperandKind::Sgpr:
                if (o.index + 1u >= kNumSgprs) break;
                w_.s[o.index] = u32(v);
                w_.s[o.index + 1] = u32(v >> 32);
                return;
            case OperandKind::Special:
                if (o.code == 106) { w_.vcc = v; return; }
                if (o.code == 126) { w_.exec = v; return; }
                break;
            default: break;
        }
        fail("unsupported 64-bit scalar destination (code " + std::to_string(o.code) + ")");
    }

    // ---- vector operand access ----
    u32 v32(const Operand& o, unsigned lane) {
        if (o.kind == OperandKind::Vgpr) {
            if (o.index >= kNumVgprs) { fail("bad vgpr"); return 0; }
            return r_.v[o.index][lane];
        }
        return s32(o);
    }
    u64 v64(const Operand& o, unsigned lane, bool f64) {
        if (o.kind == OperandKind::Vgpr) {
            if (o.index + 1u >= kNumVgprs) { fail("bad vgpr"); return 0; }
            return u64(r_.v[o.index][lane]) | (u64(r_.v[o.index + 1][lane]) << 32);
        }
        return s64(o, f64);
    }
    void wv32(const Operand& o, unsigned lane, u32 v) {
        if (o.kind != OperandKind::Vgpr || o.index >= kNumVgprs) { fail("bad vector destination"); return; }
        w_.v[o.index][lane] = v;
    }
    void wv64(const Operand& o, unsigned lane, u64 v) {
        if (o.kind != OperandKind::Vgpr || o.index + 1u >= kNumVgprs) { fail("bad vector destination"); return; }
        w_.v[o.index][lane] = u32(v);
        w_.v[o.index + 1][lane] = u32(v >> 32);
    }
    bool active(unsigned lane) const { return (r_.exec >> lane) & 1; }

    // Source i as f32 bits / float, VOP3 abs then neg applied.
    u32 fbits(int i, unsigned lane) {
        u32 b = v32(in_.src[i], lane);
        if ((in_.abs >> i) & 1) b &= 0x7FFFFFFFu;
        if ((in_.neg >> i) & 1) b ^= 0x80000000u;
        return b;
    }
    float fs(int i, unsigned lane) { return asF(fbits(i, lane)); }
    u64 dbits(int i, unsigned lane) {
        u64 b = v64(in_.src[i], lane, true);
        if ((in_.abs >> i) & 1) b &= ~(1ull << 63);
        if ((in_.neg >> i) & 1) b ^= 1ull << 63;
        return b;
    }
    double ds(int i, unsigned lane) { return asD(dbits(i, lane)); }
    u32 us(int i, unsigned lane) { return v32(in_.src[i], lane); }
    u64 us64(int i, unsigned lane) { return v64(in_.src[i], lane, false); }

    // Output modifiers for float results: omod then clamp (NaN -> +0).
    template <class F>
    F omodClamp(F x) {
        switch (in_.omod) {
            case 1: x = x * F(2); break;
            case 2: x = x * F(4); break;
            case 3: x = x * F(0.5); break;
            default: break;
        }
        if (in_.clamp) {
            if (std::isnan(x)) return F(0);
            if (x <= F(0)) return F(0);
            if (x > F(1)) return F(1);
        }
        return x;
    }
    // NaN results are canonical (0x7fc00000): which input's payload a NaN carries differs between
    // GPUs, Metal and even host compilers (operand order), so the differential test fixes one.
    u32 outF(float x) {
        x = omodClamp(x);
        return std::isnan(x) ? 0x7fc00000u : asU(x);
    }
    u64 outD(double x) { return asU(omodClamp(x)); }

    template <class Fn>
    void lanes(Fn fn) {
        for (unsigned l = 0; l < kLanes && ok_; ++l)
            if (active(l)) fn(l);
    }
    template <class Fn>
    void vop(Fn fn) {  // 32-bit result
        lanes([&](unsigned l) { wv32(in_.dst, l, fn(l)); });
    }
    template <class Fn>
    void vopF(Fn fn) {  // f32 result with output modifiers
        lanes([&](unsigned l) { wv32(in_.dst, l, outF(fn(l))); });
    }
    template <class Fn>
    void vop64(Fn fn) {
        lanes([&](unsigned l) { wv64(in_.dst, l, fn(l)); });
    }
    template <class Fn>
    void vopD(Fn fn) {
        lanes([&](unsigned l) { wv64(in_.dst, l, outD(fn(l))); });
    }
    // Result + carry/borrow bit per lane; the mask goes to sdst (inactive lanes 0).
    template <class Fn>
    void vopCarry(Fn fn) {
        u64 mask = 0;
        lanes([&](unsigned l) {
            bool c = false;
            u32 d = fn(l, c);
            wv32(in_.dst, l, d);
            if (c) mask |= 1ull << l;
        });
        if (ok_) ws64(in_.sdst, mask);
    }
    template <class Fn>
    void vcmp(bool x, Fn fn) {
        u64 mask = 0;
        lanes([&](unsigned l) {
            if (fn(l)) mask |= 1ull << l;
        });
        if (!ok_) return;
        ws64(in_.sdst, mask);
        if (x) w_.exec = mask;
    }
    bool vopc();
    bool salu();
    bool valu();
};

bool Exec::run() {
    bool handled;
    switch (in_.encoding) {
        case Encoding::SOP1:
        case Encoding::SOP2:
        case Encoding::SOPC:
        case Encoding::SOPK:
        case Encoding::SOPP: handled = salu(); break;
        case Encoding::VOPC: handled = vopc(); break;
        case Encoding::VOP1:
        case Encoding::VOP2: handled = valu(); break;
        case Encoding::VOP3: handled = vopcOpcode(in_.op) >= 0 ? vopc() : valu(); break;
        default: return unsupported();
    }
    return handled && ok_;
}

bool Exec::vopc() {
    int opc = vopcOpcode(in_.op);
    if (opc < 0) return unsupported();
    unsigned c = unsigned(opc);
    bool x = (c & 16) != 0;
    if (c < 128) {
        bool f64 = (c & 32) != 0;  // 0-31 f32, 32-63 f64, 64-95 cmps f32, 96-127 cmps f64
        if (f64)
            vcmp(x, [&](unsigned l) { return cmpFloat(c, ds(0, l), ds(1, l)); });
        else
            vcmp(x, [&](unsigned l) { return cmpFloat(c, fs(0, l), fs(1, l)); });
        return true;
    }
    unsigned group = (c - 128) >> 5;  // 0 i32, 1 i64, 2 u32, 3 u64
    if ((c & 15) == 8) {
        if (group == 0)
            vcmp(x, [&](unsigned l) { return ((us(1, l) >> classBit<u32, 23>(fbits(0, l))) & 1) != 0; });
        else if (group == 1)
            vcmp(x, [&](unsigned l) { return ((us(1, l) >> classBit<u64, 52>(dbits(0, l))) & 1) != 0; });
        else
            return unsupported();
        return true;
    }
    if ((c & 15) > 8) return unsupported();
    switch (group) {
        case 0: vcmp(x, [&](unsigned l) { return cmpInt(c, i32(us(0, l)), i32(us(1, l))); }); break;
        case 1: vcmp(x, [&](unsigned l) { return cmpInt(c, i64(us64(0, l)), i64(us64(1, l))); }); break;
        case 2: vcmp(x, [&](unsigned l) { return cmpInt(c, us(0, l), us(1, l)); }); break;
        default: vcmp(x, [&](unsigned l) { return cmpInt(c, us64(0, l), us64(1, l)); }); break;
    }
    return true;
}

bool Exec::salu() {
    const Operand& D = in_.dst;
    auto S = [&](int i) { return s32(in_.src[i]); };
    auto S64 = [&](int i) { return s64(in_.src[i], false); };
    bool& scc = w_.scc;
    const bool sccIn = r_.scc;
    const u32 simm = in_.simm16;
    const u32 ksext = u32(i32(int16_t(simm)));

    switch (in_.op) {
        // ---- SOP2 ----
        case Op::s_add_u32: { u64 r = u64(S(0)) + S(1); ws32(D, u32(r)); scc = (r >> 32) != 0; break; }
        case Op::s_sub_u32: { u32 a = S(0), b = S(1); ws32(D, a - b); scc = b > a; break; }
        case Op::s_add_i32: {
            u32 a = S(0), b = S(1), d = a + b;
            ws32(D, d);
            scc = (((a ^ d) & (b ^ d)) >> 31) != 0;
            break;
        }
        case Op::s_sub_i32: {
            u32 a = S(0), b = S(1), d = a - b;
            ws32(D, d);
            scc = (((a ^ b) & (a ^ d)) >> 31) != 0;
            break;
        }
        case Op::s_addc_u32: {
            u64 r = u64(S(0)) + S(1) + (sccIn ? 1 : 0);
            ws32(D, u32(r));
            scc = (r >> 32) != 0;
            break;
        }
        case Op::s_subb_u32: {
            u32 a = S(0), b = S(1), c = sccIn ? 1 : 0;
            ws32(D, a - b - c);
            scc = u64(b) + c > a;
            break;
        }
        case Op::s_min_i32: { i32 a = i32(S(0)), b = i32(S(1)); ws32(D, u32(a < b ? a : b)); scc = a < b; break; }
        case Op::s_min_u32: { u32 a = S(0), b = S(1); ws32(D, a < b ? a : b); scc = a < b; break; }
        case Op::s_max_i32: { i32 a = i32(S(0)), b = i32(S(1)); ws32(D, u32(a > b ? a : b)); scc = a > b; break; }
        case Op::s_max_u32: { u32 a = S(0), b = S(1); ws32(D, a > b ? a : b); scc = a > b; break; }
        case Op::s_cselect_b32: { u32 a = S(0), b = S(1); ws32(D, sccIn ? a : b); break; }
        case Op::s_cselect_b64: { u64 a = S64(0), b = S64(1); ws64(D, sccIn ? a : b); break; }
#define VPGCNREF_LOGIC(name, expr)                                                     \
    case Op::name##_b32: { u32 a = S(0), b = S(1), d = (expr); ws32(D, d); scc = d != 0; break; } \
    case Op::name##_b64: { u64 a = S64(0), b = S64(1), d = (expr); ws64(D, d); scc = d != 0; break; }
        VPGCNREF_LOGIC(s_and, a & b)
        VPGCNREF_LOGIC(s_or, a | b)
        VPGCNREF_LOGIC(s_xor, a ^ b)
        VPGCNREF_LOGIC(s_andn2, a & ~b)
        VPGCNREF_LOGIC(s_orn2, a | ~b)
        VPGCNREF_LOGIC(s_nand, ~(a & b))
        VPGCNREF_LOGIC(s_nor, ~(a | b))
        VPGCNREF_LOGIC(s_xnor, ~(a ^ b))
#undef VPGCNREF_LOGIC
        case Op::s_lshl_b32: { u32 d = S(0) << (S(1) & 31); ws32(D, d); scc = d != 0; break; }
        case Op::s_lshl_b64: { u64 a = S64(0); u64 d = a << (S(1) & 63); ws64(D, d); scc = d != 0; break; }
        case Op::s_lshr_b32: { u32 d = S(0) >> (S(1) & 31); ws32(D, d); scc = d != 0; break; }
        case Op::s_lshr_b64: { u64 a = S64(0); u64 d = a >> (S(1) & 63); ws64(D, d); scc = d != 0; break; }
        case Op::s_ashr_i32: { u32 d = u32(i32(S(0)) >> (S(1) & 31)); ws32(D, d); scc = d != 0; break; }
        case Op::s_ashr_i64: { u64 a = S64(0); u64 d = u64(i64(a) >> (S(1) & 63)); ws64(D, d); scc = d != 0; break; }
        case Op::s_bfm_b32: { u32 a = S(0), b = S(1); ws32(D, ((1u << (a & 31)) - 1) << (b & 31)); break; }
        case Op::s_bfm_b64: { u32 a = S(0), b = S(1); ws64(D, ((1ull << (a & 63)) - 1) << (b & 63)); break; }
        case Op::s_mul_i32: { u32 a = S(0), b = S(1); ws32(D, a * b); break; }
        case Op::s_bfe_u32: { u32 a = S(0), b = S(1), d = bfeU(a, b & 31, (b >> 16) & 127); ws32(D, d); scc = d != 0; break; }
        case Op::s_bfe_i32: { u32 a = S(0), b = S(1), d = bfeS(a, b & 31, (b >> 16) & 127); ws32(D, d); scc = d != 0; break; }
        case Op::s_bfe_u64: { u64 a = S64(0); u32 b = S(1); u64 d = bfeU(a, b & 63, (b >> 16) & 127); ws64(D, d); scc = d != 0; break; }
        case Op::s_bfe_i64: { u64 a = S64(0); u32 b = S(1); u64 d = bfeS(a, b & 63, (b >> 16) & 127); ws64(D, d); scc = d != 0; break; }
        case Op::s_absdiff_i32: {
            u32 a = S(0), b = S(1);
            u32 diff = a - b;
            u32 d = i32(diff) < 0 ? 0u - diff : diff;
            ws32(D, d);
            scc = d != 0;
            break;
        }

        // ---- SOP1 ----
        case Op::s_mov_b32: ws32(D, S(0)); break;
        case Op::s_mov_b64: ws64(D, S64(0)); break;
        case Op::s_cmov_b32: { u32 a = S(0); if (sccIn) ws32(D, a); break; }
        case Op::s_cmov_b64: { u64 a = S64(0); if (sccIn) ws64(D, a); break; }
        case Op::s_not_b32: { u32 d = ~S(0); ws32(D, d); scc = d != 0; break; }
        case Op::s_not_b64: { u64 d = ~S64(0); ws64(D, d); scc = d != 0; break; }
        case Op::s_wqm_b32:
        case Op::s_wqm_b64: {
            bool b64 = in_.op == Op::s_wqm_b64;
            u64 a = b64 ? S64(0) : S(0), d = 0;
            for (unsigned q = 0; q < (b64 ? 16u : 8u); ++q)
                if ((a >> (4 * q)) & 0xF) d |= 0xFull << (4 * q);
            if (b64) ws64(D, d); else ws32(D, u32(d));
            scc = d != 0;
            break;
        }
        case Op::s_brev_b32: ws32(D, brev32(S(0))); break;
        case Op::s_brev_b64: ws64(D, brev64(S64(0))); break;
        case Op::s_bcnt0_i32_b32: { u32 d = u32(32 - std::popcount(S(0))); ws32(D, d); scc = d != 0; break; }
        case Op::s_bcnt0_i32_b64: { u32 d = u32(64 - std::popcount(S64(0))); ws32(D, d); scc = d != 0; break; }
        case Op::s_bcnt1_i32_b32: { u32 d = u32(std::popcount(S(0))); ws32(D, d); scc = d != 0; break; }
        case Op::s_bcnt1_i32_b64: { u32 d = u32(std::popcount(S64(0))); ws32(D, d); scc = d != 0; break; }
        case Op::s_ff0_i32_b32: ws32(D, ffbl(u32(~S(0)))); break;
        case Op::s_ff0_i32_b64: ws32(D, ffbl(u64(~S64(0)))); break;
        case Op::s_ff1_i32_b32: ws32(D, ffbl(S(0))); break;
        case Op::s_ff1_i32_b64: ws32(D, ffbl(S64(0))); break;
        case Op::s_flbit_i32_b32: ws32(D, ffbh(S(0))); break;
        case Op::s_flbit_i32_b64: ws32(D, ffbh(S64(0))); break;
        case Op::s_flbit_i32: ws32(D, ffbhSigned(S(0))); break;
        case Op::s_flbit_i32_i64: ws32(D, ffbhSigned(S64(0))); break;
        case Op::s_sext_i32_i8: ws32(D, u32(i32(int8_t(S(0))))); break;
        case Op::s_sext_i32_i16: ws32(D, u32(i32(int16_t(S(0))))); break;
        case Op::s_bitset0_b32: { u32 b = S(0), d = s32(D); ws32(D, d & ~(1u << (b & 31))); break; }
        case Op::s_bitset1_b32: { u32 b = S(0), d = s32(D); ws32(D, d | (1u << (b & 31))); break; }
        case Op::s_bitset0_b64: { u32 b = S(0); u64 d = s64(D, false); ws64(D, d & ~(1ull << (b & 63))); break; }
        case Op::s_bitset1_b64: { u32 b = S(0); u64 d = s64(D, false); ws64(D, d | (1ull << (b & 63))); break; }
        case Op::s_and_saveexec_b64:
        case Op::s_or_saveexec_b64:
        case Op::s_xor_saveexec_b64:
        case Op::s_andn2_saveexec_b64:
        case Op::s_orn2_saveexec_b64:
        case Op::s_nand_saveexec_b64:
        case Op::s_nor_saveexec_b64:
        case Op::s_xnor_saveexec_b64: {
            u64 a = S64(0), e = r_.exec, n = 0;
            switch (in_.op) {
                case Op::s_and_saveexec_b64: n = a & e; break;
                case Op::s_or_saveexec_b64: n = a | e; break;
                case Op::s_xor_saveexec_b64: n = a ^ e; break;
                case Op::s_andn2_saveexec_b64: n = a & ~e; break;
                case Op::s_orn2_saveexec_b64: n = a | ~e; break;
                case Op::s_nand_saveexec_b64: n = ~(a & e); break;
                case Op::s_nor_saveexec_b64: n = ~(a | e); break;
                default: n = ~(a ^ e); break;
            }
            ws64(D, e);
            w_.exec = n;
            scc = n != 0;
            break;
        }
        case Op::s_quadmask_b32:
        case Op::s_quadmask_b64: {
            bool b64 = in_.op == Op::s_quadmask_b64;
            u64 a = b64 ? S64(0) : S(0), d = 0;
            for (unsigned q = 0; q < (b64 ? 16u : 8u); ++q)
                if ((a >> (4 * q)) & 0xF) d |= 1ull << q;
            if (b64) ws64(D, d); else ws32(D, u32(d));
            scc = d != 0;
            break;
        }
        case Op::s_movrels_b32: {
            const Operand& s = in_.src[0];
            u32 idx = u32(s.index) + r_.m0;
            if (s.kind != OperandKind::Sgpr || idx >= kNumSgprs) { fail("s_movrels out of range"); break; }
            ws32(D, r_.s[idx]);
            break;
        }
        case Op::s_movreld_b32: {
            u32 v = S(0);
            u32 idx = u32(D.index) + r_.m0;
            if (D.kind != OperandKind::Sgpr || idx >= kNumSgprs) { fail("s_movreld out of range"); break; }
            w_.s[idx] = v;
            break;
        }
        case Op::s_abs_i32: {
            u32 a = S(0), d = i32(a) < 0 ? 0u - a : a;
            ws32(D, d);
            scc = d != 0;
            break;
        }

        // ---- SOPC ----
        case Op::s_cmp_eq_i32: scc = i32(S(0)) == i32(S(1)); break;
        case Op::s_cmp_lg_i32: scc = i32(S(0)) != i32(S(1)); break;
        case Op::s_cmp_gt_i32: scc = i32(S(0)) > i32(S(1)); break;
        case Op::s_cmp_ge_i32: scc = i32(S(0)) >= i32(S(1)); break;
        case Op::s_cmp_lt_i32: scc = i32(S(0)) < i32(S(1)); break;
        case Op::s_cmp_le_i32: scc = i32(S(0)) <= i32(S(1)); break;
        case Op::s_cmp_eq_u32: scc = S(0) == S(1); break;
        case Op::s_cmp_lg_u32: scc = S(0) != S(1); break;
        case Op::s_cmp_gt_u32: scc = S(0) > S(1); break;
        case Op::s_cmp_ge_u32: scc = S(0) >= S(1); break;
        case Op::s_cmp_lt_u32: scc = S(0) < S(1); break;
        case Op::s_cmp_le_u32: scc = S(0) <= S(1); break;
        case Op::s_bitcmp0_b32: scc = ((S(0) >> (S(1) & 31)) & 1) == 0; break;
        case Op::s_bitcmp1_b32: scc = ((S(0) >> (S(1) & 31)) & 1) != 0; break;
        case Op::s_bitcmp0_b64: { u64 a = S64(0); scc = ((a >> (S(1) & 63)) & 1) == 0; break; }
        case Op::s_bitcmp1_b64: { u64 a = S64(0); scc = ((a >> (S(1) & 63)) & 1) != 0; break; }

        // ---- SOPK (dst is also the compared / accumulated register) ----
        case Op::s_movk_i32: ws32(D, ksext); break;
        case Op::s_cmovk_i32: if (sccIn) ws32(D, ksext); break;
        case Op::s_cmpk_eq_i32: scc = i32(s32(D)) == i32(ksext); break;
        case Op::s_cmpk_lg_i32: scc = i32(s32(D)) != i32(ksext); break;
        case Op::s_cmpk_gt_i32: scc = i32(s32(D)) > i32(ksext); break;
        case Op::s_cmpk_ge_i32: scc = i32(s32(D)) >= i32(ksext); break;
        case Op::s_cmpk_lt_i32: scc = i32(s32(D)) < i32(ksext); break;
        case Op::s_cmpk_le_i32: scc = i32(s32(D)) <= i32(ksext); break;
        case Op::s_cmpk_eq_u32: scc = s32(D) == simm; break;
        case Op::s_cmpk_lg_u32: scc = s32(D) != simm; break;
        case Op::s_cmpk_gt_u32: scc = s32(D) > simm; break;
        case Op::s_cmpk_ge_u32: scc = s32(D) >= simm; break;
        case Op::s_cmpk_lt_u32: scc = s32(D) < simm; break;
        case Op::s_cmpk_le_u32: scc = s32(D) <= simm; break;
        case Op::s_addk_i32: {
            u32 a = s32(D), b = ksext, d = a + b;
            ws32(D, d);
            scc = (((a ^ d) & (b ^ d)) >> 31) != 0;
            break;
        }
        case Op::s_mulk_i32: ws32(D, s32(D) * ksext); break;

        // ---- SOPP: only the no-op ----
        case Op::s_nop: break;

        default: return unsupported();
    }
    return true;
}

bool Exec::valu() {
    const Operand& D = in_.dst;
    switch (in_.op) {
        case Op::v_nop:
        case Op::v_clrexcp: break;

        // ---- moves / lane ops ----
        case Op::v_mov_b32: vop([&](unsigned l) { return fbits(0, l); }); break;
        case Op::v_cndmask_b32: {
            u64 m = s64(in_.src[2], false);
            vop([&](unsigned l) { return ((m >> l) & 1) ? fbits(1, l) : fbits(0, l); });
            break;
        }
        case Op::v_readfirstlane_b32: {
            unsigned lane = r_.exec ? unsigned(std::countr_zero(r_.exec)) : 0;
            ws32(D, v32(in_.src[0], lane));
            break;
        }
        case Op::v_readlane_b32: {
            unsigned lane = s32(in_.src[1]) & 63;
            ws32(D, v32(in_.src[0], lane));
            break;
        }
        case Op::v_writelane_b32: {
            u32 v = s32(in_.src[0]);
            unsigned lane = s32(in_.src[1]) & 63;
            wv32(D, lane, v);
            break;
        }
        case Op::v_movreld_b32:
        case Op::v_movrels_b32:
        case Op::v_movrelsd_b32: {
            bool relD = in_.op != Op::v_movrels_b32, relS = in_.op != Op::v_movreld_b32;
            Operand d = D, s = in_.src[0];
            if (relD) d.index = u16(d.index + r_.m0);
            if (relS) {
                if (s.kind != OperandKind::Vgpr) { fail("v_movrels source not a vgpr"); break; }
                s.index = u16(s.index + r_.m0);
            }
            if (u32(D.index) + (relD ? r_.m0 : 0) >= kNumVgprs ||
                (relS && u32(in_.src[0].index) + r_.m0 >= kNumVgprs)) {
                fail("v_movrel out of range");
                break;
            }
            lanes([&](unsigned l) { wv32(d, l, v32(s, l)); });
            break;
        }

        // ---- f32 arithmetic ----
        case Op::v_add_f32: vopF([&](unsigned l) { return fs(0, l) + fs(1, l); }); break;
        case Op::v_sub_f32: vopF([&](unsigned l) { return fs(0, l) - fs(1, l); }); break;
        case Op::v_subrev_f32: vopF([&](unsigned l) { return fs(1, l) - fs(0, l); }); break;
        case Op::v_mul_f32: vopF([&](unsigned l) { return fs(0, l) * fs(1, l); }); break;
        case Op::v_mul_legacy_f32: vopF([&](unsigned l) { return mulLegacy(fs(0, l), fs(1, l)); }); break;
        case Op::v_mac_f32:
            vopF([&](unsigned l) {
                float p = fs(0, l) * fs(1, l);
                return p + asF(r_.v[D.index][l]);
            });
            break;
        case Op::v_mac_legacy_f32:
            vopF([&](unsigned l) {
                float p = mulLegacy(fs(0, l), fs(1, l));
                return p + asF(r_.v[D.index][l]);
            });
            break;
        case Op::v_mad_f32:
        case Op::v_madmk_f32:  // src order as decoded: src0 * src1 + src2 (K placed accordingly)
        case Op::v_madak_f32:
            vopF([&](unsigned l) {
                float p = fs(0, l) * fs(1, l);
                return p + fs(2, l);
            });
            break;
        case Op::v_mad_legacy_f32:
            vopF([&](unsigned l) {
                float p = mulLegacy(fs(0, l), fs(1, l));
                return p + fs(2, l);
            });
            break;
        case Op::v_fma_f32: vopF([&](unsigned l) { return std::fma(fs(0, l), fs(1, l), fs(2, l)); }); break;
        case Op::v_min_f32: vopF([&](unsigned l) { return fminIeee(fs(0, l), fs(1, l)); }); break;
        case Op::v_max_f32: vopF([&](unsigned l) { return fmaxIeee(fs(0, l), fs(1, l)); }); break;
        case Op::v_min_legacy_f32:
            vopF([&](unsigned l) { float a = fs(0, l), b = fs(1, l); return a < b ? a : b; });
            break;
        case Op::v_max_legacy_f32:
            vopF([&](unsigned l) { float a = fs(0, l), b = fs(1, l); return a >= b ? a : b; });
            break;
        case Op::v_min3_f32:
            vopF([&](unsigned l) { return fminIeee(fminIeee(fs(0, l), fs(1, l)), fs(2, l)); });
            break;
        case Op::v_max3_f32:
            vopF([&](unsigned l) { return fmaxIeee(fmaxIeee(fs(0, l), fs(1, l)), fs(2, l)); });
            break;
        case Op::v_med3_f32: vopF([&](unsigned l) { return med3f(fs(0, l), fs(1, l), fs(2, l)); }); break;
        case Op::v_ldexp_f32:
            vopF([&](unsigned l) { return std::ldexp(fs(0, l), int(i32(us(1, l)))); });
            break;
        case Op::v_cubeid_f32: vopF([&](unsigned l) { return cube(fs(0, l), fs(1, l), fs(2, l)).id; }); break;
        case Op::v_cubesc_f32: vopF([&](unsigned l) { return cube(fs(0, l), fs(1, l), fs(2, l)).sc; }); break;
        case Op::v_cubetc_f32: vopF([&](unsigned l) { return cube(fs(0, l), fs(1, l), fs(2, l)).tc; }); break;
        case Op::v_cubema_f32: vopF([&](unsigned l) { return cube(fs(0, l), fs(1, l), fs(2, l)).ma; }); break;

        // ---- f32 unary ----
        case Op::v_fract_f32: vopF([&](unsigned l) { return fract(fs(0, l)); }); break;
        case Op::v_trunc_f32: vopF([&](unsigned l) { return std::trunc(fs(0, l)); }); break;
        case Op::v_ceil_f32: vopF([&](unsigned l) { return std::ceil(fs(0, l)); }); break;
        case Op::v_floor_f32: vopF([&](unsigned l) { return std::floor(fs(0, l)); }); break;
        case Op::v_rndne_f32: vopF([&](unsigned l) { return std::nearbyint(fs(0, l)); }); break;
        case Op::v_frexp_mant_f32:
            vopF([&](unsigned l) {
                float a = fs(0, l);
                if (std::isnan(a) || std::isinf(a)) return a;
                int e;
                return std::frexp(a, &e);
            });
            break;
        case Op::v_frexp_exp_i32_f32:
            vop([&](unsigned l) {
                float a = fs(0, l);
                if (std::isnan(a) || std::isinf(a)) return 0u;
                int e = 0;
                std::frexp(a, &e);
                return u32(e);
            });
            break;
        // Transcendentals (approximate on hardware: compare with a tolerance).
        case Op::v_exp_f32: vopF([&](unsigned l) { return std::exp2(fs(0, l)); }); break;
        case Op::v_log_f32: vopF([&](unsigned l) { return std::log2(fs(0, l)); }); break;
        case Op::v_log_clamp_f32:
            vopF([&](unsigned l) {
                float r = std::log2(fs(0, l));
                return std::isinf(r) && r < 0 ? -std::numeric_limits<float>::max() : r;
            });
            break;
        case Op::v_rcp_f32:
        case Op::v_rcp_iflag_f32: vopF([&](unsigned l) { return 1.0f / fs(0, l); }); break;
        case Op::v_rcp_clamp_f32:
            vopF([&](unsigned l) {
                float r = 1.0f / fs(0, l);
                return std::isinf(r) ? std::copysign(std::numeric_limits<float>::max(), r) : r;
            });
            break;
        case Op::v_rsq_f32: vopF([&](unsigned l) { return 1.0f / std::sqrt(fs(0, l)); }); break;
        case Op::v_rsq_clamp_f32:
            vopF([&](unsigned l) {
                float r = 1.0f / std::sqrt(fs(0, l));
                return std::isinf(r) ? std::copysign(std::numeric_limits<float>::max(), r) : r;
            });
            break;
        case Op::v_sqrt_f32: vopF([&](unsigned l) { return std::sqrt(fs(0, l)); }); break;
        case Op::v_sin_f32:  // input in revolutions
            vopF([&](unsigned l) { return float(std::sin(double(fs(0, l)) * 6.283185307179586)); });
            break;
        case Op::v_cos_f32:
            vopF([&](unsigned l) { return float(std::cos(double(fs(0, l)) * 6.283185307179586)); });
            break;

        // ---- conversions ----
        case Op::v_cvt_f32_i32: vopF([&](unsigned l) { return float(i32(us(0, l))); }); break;
        case Op::v_cvt_f32_u32: vopF([&](unsigned l) { return float(us(0, l)); }); break;
        case Op::v_cvt_u32_f32: vop([&](unsigned l) { return cvtU32(std::trunc(double(fs(0, l)))); }); break;
        case Op::v_cvt_i32_f32: vop([&](unsigned l) { return cvtI32(std::trunc(double(fs(0, l)))); }); break;
        case Op::v_cvt_rpi_i32_f32:
            vop([&](unsigned l) { return cvtI32(double(std::floor(fs(0, l) + 0.5f))); });
            break;
        case Op::v_cvt_flr_i32_f32: vop([&](unsigned l) { return cvtI32(double(std::floor(fs(0, l)))); }); break;
        case Op::v_cvt_f16_f32:
            vop([&](unsigned l) { return u32(f32ToF16(asU(omodClamp(fs(0, l))), false)); });
            break;
        case Op::v_cvt_f32_f16:
            vopF([&](unsigned l) {
                u32 b = f16ToF32(u16(us(0, l)));
                if (in_.abs & 1) b &= 0x7FFFFFFFu;
                if (in_.neg & 1) b ^= 0x80000000u;
                return asF(b);
            });
            break;
        case Op::v_cvt_pkrtz_f16_f32:
            vop([&](unsigned l) {
                u32 lo = f32ToF16(asU(omodClamp(fs(0, l))), true);
                u32 hi = f32ToF16(asU(omodClamp(fs(1, l))), true);
                return lo | (hi << 16);
            });
            break;
        case Op::v_cvt_pknorm_i16_f32:
            vop([&](unsigned l) { return snorm16(fs(0, l)) | (snorm16(fs(1, l)) << 16); });
            break;
        case Op::v_cvt_pknorm_u16_f32:
            vop([&](unsigned l) { return unorm16(fs(0, l)) | (unorm16(fs(1, l)) << 16); });
            break;
        case Op::v_cvt_pk_u16_u32:
            vop([&](unsigned l) { return std::min(us(0, l), 0xFFFFu) | (std::min(us(1, l), 0xFFFFu) << 16); });
            break;
        case Op::v_cvt_pk_i16_i32:
            vop([&](unsigned l) {
                auto sat = [](u32 x) { return u32(std::clamp(i32(x), -32768, 32767)) & 0xFFFFu; };
                return sat(us(0, l)) | (sat(us(1, l)) << 16);
            });
            break;
        case Op::v_cvt_pk_u8_f32:
            vop([&](unsigned l) {
                unsigned sh = (us(1, l) & 3) * 8;
                return (us(2, l) & ~(0xFFu << sh)) | (u8OfF32(fs(0, l)) << sh);
            });
            break;
        case Op::v_cvt_f32_ubyte0: vopF([&](unsigned l) { return float(us(0, l) & 0xFF); }); break;
        case Op::v_cvt_f32_ubyte1: vopF([&](unsigned l) { return float((us(0, l) >> 8) & 0xFF); }); break;
        case Op::v_cvt_f32_ubyte2: vopF([&](unsigned l) { return float((us(0, l) >> 16) & 0xFF); }); break;
        case Op::v_cvt_f32_ubyte3: vopF([&](unsigned l) { return float(us(0, l) >> 24); }); break;
        case Op::v_cvt_off_f32_i4:
            vopF([&](unsigned l) { return float(i32(us(0, l) << 28) >> 28) * 0.0625f; });
            break;

        // ---- f64 ----
        case Op::v_add_f64: vopD([&](unsigned l) { return ds(0, l) + ds(1, l); }); break;
        case Op::v_mul_f64: vopD([&](unsigned l) { return ds(0, l) * ds(1, l); }); break;
        case Op::v_fma_f64: vopD([&](unsigned l) { return std::fma(ds(0, l), ds(1, l), ds(2, l)); }); break;
        case Op::v_min_f64: vopD([&](unsigned l) { return fminIeee(ds(0, l), ds(1, l)); }); break;
        case Op::v_max_f64: vopD([&](unsigned l) { return fmaxIeee(ds(0, l), ds(1, l)); }); break;
        case Op::v_ldexp_f64: vopD([&](unsigned l) { return std::ldexp(ds(0, l), int(i32(us(1, l)))); }); break;
        case Op::v_trunc_f64: vopD([&](unsigned l) { return std::trunc(ds(0, l)); }); break;
        case Op::v_ceil_f64: vopD([&](unsigned l) { return std::ceil(ds(0, l)); }); break;
        case Op::v_floor_f64: vopD([&](unsigned l) { return std::floor(ds(0, l)); }); break;
        case Op::v_rndne_f64: vopD([&](unsigned l) { return std::nearbyint(ds(0, l)); }); break;
        case Op::v_fract_f64: vopD([&](unsigned l) { return fract(ds(0, l)); }); break;
        case Op::v_rcp_f64: vopD([&](unsigned l) { return 1.0 / ds(0, l); }); break;
        case Op::v_rsq_f64: vopD([&](unsigned l) { return 1.0 / std::sqrt(ds(0, l)); }); break;
        case Op::v_sqrt_f64: vopD([&](unsigned l) { return std::sqrt(ds(0, l)); }); break;
        case Op::v_frexp_mant_f64:
            vopD([&](unsigned l) {
                double a = ds(0, l);
                if (std::isnan(a) || std::isinf(a)) return a;
                int e;
                return std::frexp(a, &e);
            });
            break;
        case Op::v_frexp_exp_i32_f64:
            vop([&](unsigned l) {
                double a = ds(0, l);
                if (std::isnan(a) || std::isinf(a)) return 0u;
                int e = 0;
                std::frexp(a, &e);
                return u32(e);
            });
            break;
        case Op::v_cvt_f64_f32: vopD([&](unsigned l) { return double(fs(0, l)); }); break;
        case Op::v_cvt_f32_f64: vopF([&](unsigned l) { return float(ds(0, l)); }); break;
        case Op::v_cvt_f64_i32: vopD([&](unsigned l) { return double(i32(us(0, l))); }); break;
        case Op::v_cvt_f64_u32: vopD([&](unsigned l) { return double(us(0, l)); }); break;
        case Op::v_cvt_i32_f64: vop([&](unsigned l) { return cvtI32(std::trunc(ds(0, l))); }); break;
        case Op::v_cvt_u32_f64: vop([&](unsigned l) { return cvtU32(std::trunc(ds(0, l))); }); break;

        // ---- integer ----
        case Op::v_add_i32:
            vopCarry([&](unsigned l, bool& c) { u64 r = u64(us(0, l)) + us(1, l); c = (r >> 32) != 0; return u32(r); });
            break;
        case Op::v_sub_i32:
            vopCarry([&](unsigned l, bool& c) { u32 a = us(0, l), b = us(1, l); c = b > a; return a - b; });
            break;
        case Op::v_subrev_i32:
            vopCarry([&](unsigned l, bool& c) { u32 a = us(1, l), b = us(0, l); c = b > a; return a - b; });
            break;
        case Op::v_addc_u32:
        case Op::v_subb_u32:
        case Op::v_subbrev_u32: {
            u64 cin = s64(in_.src[2], false);
            Op op = in_.op;
            vopCarry([&](unsigned l, bool& c) {
                u32 ci = (cin >> l) & 1;
                u32 a = us(0, l), b = us(1, l);
                if (op == Op::v_addc_u32) {
                    u64 r = u64(a) + b + ci;
                    c = (r >> 32) != 0;
                    return u32(r);
                }
                if (op == Op::v_subbrev_u32) std::swap(a, b);
                c = u64(b) + ci > a;
                return a - b - ci;
            });
            break;
        }
        case Op::v_mul_u32_u24: vop([&](unsigned l) { return (us(0, l) & 0xFFFFFF) * (us(1, l) & 0xFFFFFF); }); break;
        case Op::v_mul_hi_u32_u24:
            vop([&](unsigned l) { return u32((u64(us(0, l) & 0xFFFFFF) * (us(1, l) & 0xFFFFFF)) >> 32); });
            break;
        case Op::v_mul_i32_i24: vop([&](unsigned l) { return sext24(us(0, l)) * sext24(us(1, l)); }); break;
        case Op::v_mul_hi_i32_i24:
            vop([&](unsigned l) { return u32(u64(i64(i32(sext24(us(0, l)))) * i64(i32(sext24(us(1, l))))) >> 32); });
            break;
        case Op::v_mad_u32_u24:
            vop([&](unsigned l) { return (us(0, l) & 0xFFFFFF) * (us(1, l) & 0xFFFFFF) + us(2, l); });
            break;
        case Op::v_mad_i32_i24: vop([&](unsigned l) { return sext24(us(0, l)) * sext24(us(1, l)) + us(2, l); }); break;
        case Op::v_mul_lo_u32:
        case Op::v_mul_lo_i32: vop([&](unsigned l) { return us(0, l) * us(1, l); }); break;
        case Op::v_mul_hi_u32: vop([&](unsigned l) { return u32((u64(us(0, l)) * us(1, l)) >> 32); }); break;
        case Op::v_mul_hi_i32:
            vop([&](unsigned l) { return u32(u64(i64(i32(us(0, l))) * i64(i32(us(1, l)))) >> 32); });
            break;
        case Op::v_mad_u64_u32: {
            u64 mask = 0;
            lanes([&](unsigned l) {
                u64 p = u64(us(0, l)) * us(1, l);
                u64 c = us64(2, l);
                u64 r = p + c;
                if (r < p) mask |= 1ull << l;
                wv64(D, l, r);
            });
            if (ok_) ws64(in_.sdst, mask);
            break;
        }
        case Op::v_min_i32: vop([&](unsigned l) { return u32(std::min(i32(us(0, l)), i32(us(1, l)))); }); break;
        case Op::v_max_i32: vop([&](unsigned l) { return u32(std::max(i32(us(0, l)), i32(us(1, l)))); }); break;
        case Op::v_min_u32: vop([&](unsigned l) { return std::min(us(0, l), us(1, l)); }); break;
        case Op::v_max_u32: vop([&](unsigned l) { return std::max(us(0, l), us(1, l)); }); break;
        case Op::v_min3_i32:
            vop([&](unsigned l) { return u32(std::min({i32(us(0, l)), i32(us(1, l)), i32(us(2, l))})); });
            break;
        case Op::v_max3_i32:
            vop([&](unsigned l) { return u32(std::max({i32(us(0, l)), i32(us(1, l)), i32(us(2, l))})); });
            break;
        case Op::v_med3_i32: vop([&](unsigned l) { return u32(med3i(i32(us(0, l)), i32(us(1, l)), i32(us(2, l)))); }); break;
        case Op::v_min3_u32: vop([&](unsigned l) { return std::min({us(0, l), us(1, l), us(2, l)}); }); break;
        case Op::v_max3_u32: vop([&](unsigned l) { return std::max({us(0, l), us(1, l), us(2, l)}); }); break;
        case Op::v_med3_u32: vop([&](unsigned l) { return med3i(us(0, l), us(1, l), us(2, l)); }); break;
        case Op::v_lshr_b32: vop([&](unsigned l) { return us(0, l) >> (us(1, l) & 31); }); break;
        case Op::v_lshrrev_b32: vop([&](unsigned l) { return us(1, l) >> (us(0, l) & 31); }); break;
        case Op::v_ashr_i32: vop([&](unsigned l) { return u32(i32(us(0, l)) >> (us(1, l) & 31)); }); break;
        case Op::v_ashrrev_i32: vop([&](unsigned l) { return u32(i32(us(1, l)) >> (us(0, l) & 31)); }); break;
        case Op::v_lshl_b32: vop([&](unsigned l) { return us(0, l) << (us(1, l) & 31); }); break;
        case Op::v_lshlrev_b32: vop([&](unsigned l) { return us(1, l) << (us(0, l) & 31); }); break;
        case Op::v_lshl_b64: vop64([&](unsigned l) { return us64(0, l) << (us(1, l) & 63); }); break;
        case Op::v_lshr_b64: vop64([&](unsigned l) { return us64(0, l) >> (us(1, l) & 63); }); break;
        case Op::v_ashr_i64: vop64([&](unsigned l) { return u64(i64(us64(0, l)) >> (us(1, l) & 63)); }); break;
        case Op::v_and_b32: vop([&](unsigned l) { return us(0, l) & us(1, l); }); break;
        case Op::v_or_b32: vop([&](unsigned l) { return us(0, l) | us(1, l); }); break;
        case Op::v_xor_b32: vop([&](unsigned l) { return us(0, l) ^ us(1, l); }); break;
        case Op::v_not_b32: vop([&](unsigned l) { return ~us(0, l); }); break;
        case Op::v_bfm_b32: vop([&](unsigned l) { return ((1u << (us(0, l) & 31)) - 1) << (us(1, l) & 31); }); break;
        case Op::v_bfe_u32: vop([&](unsigned l) { return bfeU(us(0, l), us(1, l) & 31, us(2, l) & 31); }); break;
        case Op::v_bfe_i32: vop([&](unsigned l) { return bfeS(us(0, l), us(1, l) & 31, us(2, l) & 31); }); break;
        case Op::v_bfi_b32:
            vop([&](unsigned l) { u32 m = us(0, l); return (m & us(1, l)) | (~m & us(2, l)); });
            break;
        case Op::v_alignbit_b32:
            vop([&](unsigned l) { return u32(((u64(us(0, l)) << 32) | us(1, l)) >> (us(2, l) & 31)); });
            break;
        case Op::v_alignbyte_b32:
            vop([&](unsigned l) { return u32(((u64(us(0, l)) << 32) | us(1, l)) >> (8 * (us(2, l) & 3))); });
            break;
        case Op::v_bcnt_u32_b32: vop([&](unsigned l) { return u32(std::popcount(us(0, l))) + us(1, l); }); break;
        case Op::v_mbcnt_lo_u32_b32:
            vop([&](unsigned l) {
                u32 m = l >= 32 ? 0xFFFFFFFFu : ((1u << l) - 1);
                return u32(std::popcount(us(0, l) & m)) + us(1, l);
            });
            break;
        case Op::v_mbcnt_hi_u32_b32:
            vop([&](unsigned l) {
                u32 m = l < 32 ? 0u : ((1ull << (l - 32)) - 1);
                return u32(std::popcount(us(0, l) & m)) + us(1, l);
            });
            break;
        case Op::v_ffbl_b32: vop([&](unsigned l) { return ffbl(us(0, l)); }); break;
        case Op::v_ffbh_u32: vop([&](unsigned l) { return ffbh(us(0, l)); }); break;
        case Op::v_ffbh_i32: vop([&](unsigned l) { return ffbhSigned(us(0, l)); }); break;
        case Op::v_bfrev_b32: vop([&](unsigned l) { return brev32(us(0, l)); }); break;
        case Op::v_lerp_u8:
            vop([&](unsigned l) {
                u32 a = us(0, l), b = us(1, l), c = us(2, l), d = 0;
                for (unsigned i = 0; i < 32; i += 8)
                    d |= ((((a >> i) & 0xFF) + ((b >> i) & 0xFF) + ((c >> i) & 1)) >> 1) << i;
                return d;
            });
            break;
        case Op::v_sad_u8:
        case Op::v_sad_hi_u8:
        case Op::v_sad_u16: {
            unsigned w = in_.op == Op::v_sad_u16 ? 16 : 8;
            bool hi = in_.op == Op::v_sad_hi_u8;
            vop([&](unsigned l) {
                u32 a = us(0, l), b = us(1, l), s = 0, m = (1u << w) - 1;
                for (unsigned i = 0; i < 32; i += w) {
                    u32 x = (a >> i) & m, y = (b >> i) & m;
                    s += x > y ? x - y : y - x;
                }
                return (hi ? s << 16 : s) + us(2, l);
            });
            break;
        }
        case Op::v_sad_u32:
            vop([&](unsigned l) { u32 a = us(0, l), b = us(1, l); return (a > b ? a - b : b - a) + us(2, l); });
            break;

        default: return unsupported();
    }
    return true;
}

}  // namespace

bool step(const vpgcn::Instruction& inst, Wave& wave, std::string& err) {
    Wave next = wave;
    Exec ex(inst, wave, next, err);
    if (!ex.run()) return false;
    wave = next;
    return true;
}

}  // namespace vpgcnref
