#include "vpir_eval.h"

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace vpireval {
namespace {

using namespace vpir;

float f32(uint32_t b) { return std::bit_cast<float>(b); }
uint32_t bits(float f) { return std::bit_cast<uint32_t>(f); }

enum class Flow { Next, Break, Continue, Return, Discard, Error };

struct Machine {
    const Module& m;
    const Function& fn;
    Invocation& inv;
    std::string& error;
    std::vector<Value> values;
    std::vector<Value> vars;
    size_t steps = 0;

    Machine(const Module& mod, const Function& f, Invocation& i, std::string& e)
        : m(mod), fn(f), inv(i), error(e), values(f.valueTypes.size()), vars(f.vars.size()) {
        for (size_t k = 0; k < vars.size(); ++k) vars[k].type = f.vars[k].type;  // zero-initialised
    }

    bool fail(const std::string& what) {
        if (error.empty()) error = what;
        return false;
    }

    // Component-wise helpers. `per` gets component bits and returns result bits.
    template <class F>
    Value map1(const Value& a, Type rt, F per) {
        Value r;
        r.type = rt;
        for (unsigned c = 0; c < rt.width; ++c) r.bits[c] = per(a.bits[a.type.width == 1 ? 0 : c]);
        return r;
    }
    template <class F>
    Value map2(const Value& a, const Value& b, Type rt, F per) {
        Value r;
        r.type = rt;
        for (unsigned c = 0; c < rt.width; ++c)
            r.bits[c] = per(a.bits[a.type.width == 1 ? 0 : c], b.bits[b.type.width == 1 ? 0 : c]);
        return r;
    }
    template <class F>
    Value map3(const Value& a, const Value& b, const Value& d, Type rt, F per) {
        Value r;
        r.type = rt;
        for (unsigned c = 0; c < rt.width; ++c)
            r.bits[c] = per(a.bits[a.type.width == 1 ? 0 : c], b.bits[b.type.width == 1 ? 0 : c],
                            d.bits[d.type.width == 1 ? 0 : c]);
        return r;
    }

    // Float component as float (f16 widened).
    static float fl(Scalar s, uint32_t b) { return s == Scalar::F16 ? halfToFloat(uint16_t(b)) : f32(b); }
    static uint32_t fb(Scalar s, float v) { return s == Scalar::F16 ? floatToHalf(v) : bits(v); }

    // fmin/fmax as Metal's: any NaN operand, signaling too, yields the other one (libm quiets a
    // signaling NaN and returns it); -0 orders below +0, as IEEE 754-2008 minNum/maxNum on GPUs.
    static float minN(float x, float y) {
        if (std::isnan(x)) return y;
        if (std::isnan(y)) return x;
        return x == 0 && y == 0 ? (std::signbit(x) ? x : y) : std::fmin(x, y);
    }
    static float maxN(float x, float y) {
        if (std::isnan(x)) return y;
        if (std::isnan(y)) return x;
        return x == 0 && y == 0 ? (std::signbit(x) ? y : x) : std::fmax(x, y);
    }

    bool arith(const Inst& in, const std::vector<Value>& a, Value& out) {
        Type t = in.type;
        Scalar s = (a.empty() ? t : a[0].type).scalar;
        bool fl32 = s == Scalar::F32 || s == Scalar::F16;
        auto F1 = [&](auto op) { return map1(a[0], t, [&](uint32_t x) { return fb(s, op(fl(s, x))); }); };
        auto F2 = [&](auto op) {
            return map2(a[0], a[1], t, [&](uint32_t x, uint32_t y) { return fb(s, op(fl(s, x), fl(s, y))); });
        };
        auto I2 = [&](auto op) {
            return map2(a[0], a[1], t, [&](uint32_t x, uint32_t y) {
                if (s == Scalar::I32) return uint32_t(op(int32_t(x), int32_t(y)));
                return uint32_t(op(x, y));
            });
        };
        switch (in.op) {
            case Op::Neg:
                out = fl32 ? F1([](float x) { return -x; }) : map1(a[0], t, [](uint32_t x) { return uint32_t(0) - x; });
                return true;
            case Op::Abs:
                out = fl32 ? F1([](float x) { return std::fabs(x); })
                           : map1(a[0], t, [](uint32_t x) { return int32_t(x) < 0 ? uint32_t(0) - x : x; });
                return true;
            case Op::Not:
                out = map1(a[0], t, [&](uint32_t x) { return s == Scalar::Bool ? uint32_t(!x) : ~x; });
                return true;
            case Op::Add: out = fl32 ? F2([](float x, float y) { return x + y; }) : I2([](auto x, auto y) { return x + y; }); return true;
            case Op::Sub: out = fl32 ? F2([](float x, float y) { return x - y; }) : I2([](auto x, auto y) { return x - y; }); return true;
            case Op::Mul:
                out = fl32 ? F2([](float x, float y) { return x * y; })
                           : map2(a[0], a[1], t, [](uint32_t x, uint32_t y) { return x * y; });
                return true;
            case Op::Div:
                if (fl32) {
                    out = F2([](float x, float y) { return x / y; });
                    return true;
                }
                out = I2([](auto x, auto y) { return y == 0 ? decltype(x)(0) : (s_div(x, y)); });
                return true;
            case Op::Rem:
                out = fl32 ? F2([](float x, float y) { return std::fmod(x, y); })
                           : I2([](auto x, auto y) { return y == 0 ? decltype(x)(0) : s_rem(x, y); });
                return true;
            case Op::Min: out = fl32 ? F2(minN) : I2([](auto x, auto y) { return x < y ? x : y; }); return true;
            case Op::Max: out = fl32 ? F2(maxN) : I2([](auto x, auto y) { return x > y ? x : y; }); return true;
            case Op::And: out = map2(a[0], a[1], t, [](uint32_t x, uint32_t y) { return x & y; }); return true;
            case Op::Or: out = map2(a[0], a[1], t, [](uint32_t x, uint32_t y) { return x | y; }); return true;
            case Op::Xor: out = map2(a[0], a[1], t, [](uint32_t x, uint32_t y) { return x ^ y; }); return true;
            case Op::Shl: out = map2(a[0], a[1], t, [](uint32_t x, uint32_t y) { return y >= 32 ? 0u : x << y; }); return true;
            case Op::Shr:
                out = map2(a[0], a[1], t, [&](uint32_t x, uint32_t y) {
                    if (y >= 32) y = 31;
                    return s == Scalar::I32 ? uint32_t(int32_t(x) >> y) : x >> y;
                });
                return true;
            case Op::Fma:
                out = map3(a[0], a[1], a[2], t, [&](uint32_t x, uint32_t y, uint32_t z) {
                    return fb(s, std::fma(fl(s, x), fl(s, y), fl(s, z)));
                });
                return true;
            case Op::Clamp:
                out = map3(a[0], a[1], a[2], t, [&](uint32_t x, uint32_t lo, uint32_t hi) {
                    if (fl32) return fb(s, minN(maxN(fl(s, x), fl(s, lo)), fl(s, hi)));
                    if (s == Scalar::I32) return uint32_t(std::min(std::max(int32_t(x), int32_t(lo)), int32_t(hi)));
                    return std::min(std::max(x, lo), hi);
                });
                return true;
            case Op::Mix:
                out = map3(a[0], a[1], a[2], t, [&](uint32_t x, uint32_t y, uint32_t w) {
                    float fx = fl(s, x);
                    return fb(s, fx + (fl(s, y) - fx) * fl(s, w));
                });
                return true;
            default: return false;
        }
    }
    template <class T>
    static T s_div(T x, T y) { return x / y; }
    template <class T>
    static T s_rem(T x, T y) { return x % y; }

    bool compare(const Inst& in, const std::vector<Value>& a, Value& out) {
        Scalar s = a[0].type.scalar;
        auto cmp = [&](auto pred) {
            return map2(a[0], a[1], in.type, [&](uint32_t x, uint32_t y) -> uint32_t {
                if (s == Scalar::F32 || s == Scalar::F16) return pred(fl(s, x), fl(s, y));
                if (s == Scalar::I32) return pred(int32_t(x), int32_t(y));
                return pred(x, y);
            });
        };
        switch (in.op) {
            case Op::Eq: out = cmp([](auto x, auto y) { return x == y; }); return true;
            case Op::Ne: out = cmp([](auto x, auto y) { return x != y; }); return true;
            case Op::Lt: out = cmp([](auto x, auto y) { return x < y; }); return true;
            case Op::Le: out = cmp([](auto x, auto y) { return x <= y; }); return true;
            case Op::Gt: out = cmp([](auto x, auto y) { return x > y; }); return true;
            case Op::Ge: out = cmp([](auto x, auto y) { return x >= y; }); return true;
            case Op::IsNan: out = map1(a[0], in.type, [&](uint32_t x) { return uint32_t(std::isnan(fl(s, x))); }); return true;
            case Op::IsInf: out = map1(a[0], in.type, [&](uint32_t x) { return uint32_t(std::isinf(fl(s, x))); }); return true;
            default: return false;
        }
    }

    bool floatUnary(const Inst& in, const std::vector<Value>& a, Value& out) {
        Scalar s = in.type.scalar;
        auto F1 = [&](auto op) { out = map1(a[0], in.type, [&](uint32_t x) { return fb(s, op(fl(s, x))); }); return true; };
        switch (in.op) {
            case Op::Floor: return F1([](float x) { return std::floor(x); });
            case Op::Ceil: return F1([](float x) { return std::ceil(x); });
            case Op::Trunc: return F1([](float x) { return std::trunc(x); });
            case Op::Round: return F1([](float x) { return std::nearbyint(x); });
            case Op::Fract: return F1([](float x) { float f = x - std::floor(x); return std::isnan(f) ? f : minN(f, 0x1.fffffep-1f); });
            case Op::Sqrt: return F1([](float x) { return std::sqrt(x); });
            case Op::Rsq: return F1([](float x) { return 1.0f / std::sqrt(x); });
            case Op::Rcp: return F1([](float x) { return 1.0f / x; });
            case Op::Exp2: return F1([](float x) { return std::exp2(x); });
            case Op::Log2: return F1([](float x) { return std::log2(x); });
            case Op::Sin: return F1([](float x) { return std::sin(x); });
            case Op::Cos: return F1([](float x) { return std::cos(x); });
            case Op::Saturate: return F1([](float x) { return minN(maxN(x, 0.0f), 1.0f); });
            default: return false;
        }
    }

    bool convert(const Inst& in, const Value& a, Value& out) {
        Scalar from = a.type.scalar, to = in.type.scalar;
        out = map1(a, in.type, [&](uint32_t x) -> uint32_t {
            bool ff = from == Scalar::F32 || from == Scalar::F16;
            if (ff) {
                float v = fl(from, x);
                if (to == Scalar::F32 || to == Scalar::F16) return fb(to, v);
                if (std::isnan(v)) return 0u;
                double t = std::trunc(double(v));
                if (to == Scalar::I32) return uint32_t(int32_t(std::fmin(std::fmax(t, -2147483648.0), 2147483647.0)));
                if (to == Scalar::U32) return uint32_t(std::fmin(std::fmax(t, 0.0), 4294967295.0));
                return uint32_t(v != 0);
            }
            if (to == Scalar::F32 || to == Scalar::F16) return fb(to, from == Scalar::I32 ? float(int32_t(x)) : float(x));
            if (to == Scalar::Bool) return uint32_t(x != 0);
            return x;  // i32 <-> u32
        });
        return true;
    }

    static void toBytes(const Value& v, std::vector<uint8_t>& b) {
        unsigned esz = v.type.scalar == Scalar::F16 ? 2 : 4;
        for (unsigned c = 0; c < v.type.width; ++c)
            for (unsigned k = 0; k < esz; ++k) b.push_back(uint8_t(v.bits[c] >> (8 * k)));
    }
    static Value fromBytes(Type t, const std::vector<uint8_t>& b) {
        Value v;
        v.type = t;
        unsigned esz = t.scalar == Scalar::F16 ? 2 : 4;
        for (unsigned c = 0; c < t.width; ++c)
            for (unsigned k = 0; k < esz; ++k) v.bits[c] |= uint32_t(b[c * esz + k]) << (8 * k);
        return v;
    }

    bool eval(const Inst& in) {
        if (++steps > 20000000) return fail("step limit (an endless loop?)");
        std::vector<Value> a;
        for (ValueId id : in.args) a.push_back(values[id]);
        Value out;
        out.type = in.type;
        switch (in.op) {
            case Op::Const:
                for (unsigned c = 0; c < in.type.width; ++c) out.bits[c] = in.lits[c];
                break;
            case Op::LoadVar: out = vars[in.sym]; break;
            case Op::StoreVar: vars[in.sym] = a[0]; return true;
            case Op::LoadInput: {
                const IoVar& io = m.inputs[in.sym];
                if (io.builtin != Builtin::None) {
                    auto it = inv.builtins.find(io.builtin);
                    if (it != inv.builtins.end()) out = it->second, out.type = in.type;
                } else {
                    auto it = inv.locations.find(io.location);
                    if (it != inv.locations.end()) out = it->second, out.type = in.type;
                }
                break;
            }
            case Op::StoreOutput: inv.outputs[in.sym] = a[0]; return true;
            case Op::Select: {
                const Value& c = a[0];
                for (unsigned k = 0; k < in.type.width; ++k)
                    out.bits[k] = c.bits[c.type.width == 1 ? 0 : k] ? a[1].bits[k] : a[2].bits[k];
                break;
            }
            case Op::Convert: convert(in, a[0], out); break;
            case Op::Bitcast: {
                std::vector<uint8_t> b;
                toBytes(a[0], b);
                out = fromBytes(in.type, b);
                break;
            }
            case Op::BitfieldExtract:
                out = map3(a[0], a[1], a[2], in.type, [&](uint32_t v, uint32_t off, uint32_t cnt) -> uint32_t {
                    if (cnt == 0) return 0;
                    uint32_t f = (v >> off) & (cnt >= 32 ? ~0u : ((1u << cnt) - 1));
                    if (in.type.scalar == Scalar::I32 && cnt < 32 && (f >> (cnt - 1)) & 1) f |= ~0u << cnt;
                    return f;
                });
                break;
            case Op::BitfieldInsert:
                out = map2(a[0], a[1], in.type, [&](uint32_t base, uint32_t ins) {
                    uint32_t off = a[2].bits[0], cnt = a[3].bits[0];
                    uint32_t mask = (cnt >= 32 ? ~0u : ((1u << cnt) - 1)) << off;
                    return (base & ~mask) | ((ins << off) & mask);
                });
                break;
            case Op::BitCount: out = map1(a[0], in.type, [](uint32_t x) { return uint32_t(std::popcount(x)); }); break;
            case Op::BitReverse:
                out = map1(a[0], in.type, [](uint32_t x) {
                    uint32_t r = 0;
                    for (int k = 0; k < 32; ++k) r |= ((x >> k) & 1u) << (31 - k);
                    return r;
                });
                break;
            case Op::FindLsb: out = map1(a[0], in.type, [](uint32_t x) { return uint32_t(std::countr_zero(x)); }); break;
            case Op::FindMsb: out = map1(a[0], in.type, [](uint32_t x) { return x ? uint32_t(31 - std::countl_zero(x)) : ~0u; }); break;
            case Op::Pow: {
                Scalar s = in.type.scalar;
                out = map2(a[0], a[1], in.type, [&](uint32_t x, uint32_t y) { return fb(s, std::pow(fl(s, x), fl(s, y))); });
                break;
            }
            case Op::Dot: {
                Scalar s = a[0].type.scalar;
                float acc = 0;
                for (unsigned c = 0; c < a[0].type.width; ++c) acc += fl(s, a[0].bits[c]) * fl(s, a[1].bits[c]);
                out.bits[0] = fb(s, acc);
                break;
            }
            case Op::Construct: {
                unsigned k = 0;
                for (const Value& v : a)
                    for (unsigned c = 0; c < v.type.width; ++c) out.bits[k++] = v.bits[c];
                break;
            }
            case Op::Extract: out.bits[0] = a[0].bits[in.lits[0]]; break;
            case Op::Insert:
                out = a[0];
                out.bits[in.lits[0]] = a[1].bits[0];
                break;
            case Op::Shuffle:
                for (unsigned c = 0; c < in.type.width; ++c) {
                    uint32_t l = in.lits[c];
                    out.bits[c] = l < a[0].type.width ? a[0].bits[l] : a[1].bits[l - a[0].type.width];
                }
                break;
            case Op::Load:
            case Op::Store:
            case Op::AtomicAdd:
            case Op::AtomicSub:
            case Op::AtomicMin:
            case Op::AtomicMax:
            case Op::AtomicAnd:
            case Op::AtomicOr:
            case Op::AtomicXor:
            case Op::AtomicExchange:
            case Op::AtomicCompareExchange: {
                if (in.sym >= m.resources.size()) return fail("memory target other than a buffer");
                std::vector<uint8_t>& mem = inv.memory[in.sym];
                uint32_t addr = a[0].bits[0];
                auto rd = [&](uint32_t at) {
                    uint32_t w = 0;
                    if (at + 4 <= mem.size()) std::memcpy(&w, &mem[at], 4);
                    return w;
                };
                auto wr = [&](uint32_t at, uint32_t w) {
                    if (at + 4 > mem.size()) mem.resize(at + 4, 0);
                    std::memcpy(&mem[at], &w, 4);
                };
                if (in.op == Op::Load) {
                    for (unsigned c = 0; c < in.type.width; ++c) out.bits[c] = rd(addr + 4 * c);
                    break;
                }
                if (in.op == Op::Store) {
                    for (unsigned c = 0; c < a[1].type.width; ++c) wr(addr + 4 * c, a[1].bits[c]);
                    return true;
                }
                uint32_t old = rd(addr), v = a[1].bits[0], n = old;
                bool sgn = in.type.scalar == Scalar::I32;
                switch (in.op) {
                    case Op::AtomicAdd: n = old + v; break;
                    case Op::AtomicSub: n = old - v; break;
                    case Op::AtomicMin: n = sgn ? uint32_t(std::min(int32_t(old), int32_t(v))) : std::min(old, v); break;
                    case Op::AtomicMax: n = sgn ? uint32_t(std::max(int32_t(old), int32_t(v))) : std::max(old, v); break;
                    case Op::AtomicAnd: n = old & v; break;
                    case Op::AtomicOr: n = old | v; break;
                    case Op::AtomicXor: n = old ^ v; break;
                    case Op::AtomicExchange: n = v; break;
                    default: n = old == v ? a[2].bits[0] : old; break;  // compare (v), value
                }
                wr(addr, n);
                out.bits[0] = old;
                break;
            }
            case Op::Barrier:
            case Op::BarrierAll: return true;
            default: {
                if (arith(in, a, out) || compare(in, a, out) || floatUnary(in, a, out)) break;
                return fail(std::string("cannot evaluate ") + opInfo(in.op).name);
            }
        }
        if (in.result != kNoValue) values[in.result] = out;
        if (trace && in.result != kNoValue)
            std::fprintf(stderr, "%%%u = %08x %08x %08x %08x\n", unsigned(in.result), out.bits[0], out.bits[1], out.bits[2],
                         out.bits[3]);
        return true;
    }

    Flow block(const Block& b) {
        for (const Node& n : b) {
            switch (n.kind) {
                case NodeKind::Inst:
                    if (!eval(n.inst)) return Flow::Error;
                    break;
                case NodeKind::If: {
                    Flow f = block(values[n.value].bits[0] ? n.body : n.elseBody);
                    if (f != Flow::Next) return f;
                    break;
                }
                case NodeKind::Loop:
                    for (;;) {
                        Flow f = block(n.body);
                        if (f == Flow::Break) break;
                        if (f == Flow::Continue || f == Flow::Next) {
                            if (steps > 20000000) return fail("step limit"), Flow::Error;
                            continue;
                        }
                        return f;
                    }
                    break;
                case NodeKind::Break: return Flow::Break;
                case NodeKind::Continue: return Flow::Continue;
                case NodeKind::Return: return Flow::Return;
                case NodeKind::Discard: inv.discarded = true; return Flow::Discard;
            }
        }
        return Flow::Next;
    }
};

}  // namespace

bool run(const vpir::Module& m, Invocation& inv, std::string& error) {
    const vpir::Function* main = m.entry();
    if (!main) {
        error = "no entry point";
        return false;
    }
    if (main->body.empty()) return true;
    Machine machine(m, *main, inv, error);
    Flow f = machine.block(main->body);
    return f != Flow::Error;
}

}  // namespace vpireval
