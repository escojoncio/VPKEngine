#include <algorithm>
#include <bit>
#include <cmath>
#include <map>
#include <optional>
#include <tuple>

#include "pass_util.h"
#include "vpir/passes.h"

namespace vpir {

namespace {

bool hasEffect(Op op) { return op == Op::Call || (op >= Op::AtomicAdd && op <= Op::ImageAtomicCompareExchange); }

// Ops whose result depends only on their operands and on state that is fixed for the invocation, so
// a dominating copy can stand in. Subgroup ops and derivatives depend on which lanes are active,
// variables and device memory on earlier stores: those are never merged.
bool pure(const Module& m, const Inst& in) {
    if (in.result == kNoValue || hasEffect(in.op)) return false;
    switch (in.op) {
        case Op::LoadVar:
        case Op::Ddx:
        case Op::Ddy:
        case Op::Fwidth:
        case Op::ImageGather:
        case Op::ImageQueryLod: return false;  // (uses derivatives)
        case Op::ImageSample: return in.image & (kImageLod | kImageGrad);  // implicit lod uses derivatives
        case Op::Load: return in.sym < m.resources.size() && m.resources[in.sym].kind == ResourceKind::ConstantBuffer;
        case Op::ImageFetch:
        case Op::ImageSize:
        case Op::ImageLevels:
        case Op::ImageSamples: return in.sym < m.resources.size() && m.resources[in.sym].kind == ResourceKind::Texture;
        case Op::QuadShuffle:
        case Op::QuadAny: return false;  // (depend on the active lanes, like subgroup ops)
        default: return !(in.op >= Op::SubgroupBallot && in.op <= Op::SubgroupAll);
    }
}

using Key = std::tuple<Op, Scalar, uint8_t, bool, std::vector<ValueId>, uint32_t, uint32_t, std::vector<uint32_t>, uint8_t>;
Key keyOf(const Inst& in) {
    return {in.op, in.type.scalar, in.type.width, in.precise, in.args, in.sym, in.sampler, in.lits, in.image};
}

class Optimizer {
public:
    explicit Optimizer(Module& m) : m_(m) {}

    void run() {
        for (Function& f : m_.functions) {
            fn_ = &f;
            replace_.clear();
            consts_.clear();
            scopes_.assign(1, {});
            block(f.body);
            removeDeadValues(f);
        }
    }

private:
    ValueId resolve(ValueId v) const {
        auto it = replace_.find(v);
        return it == replace_.end() ? v : it->second;
    }

    void block(Block& b) {
        for (Node& n : b) {
            if (n.value != kNoValue) n.value = resolve(n.value);
            if (n.kind == NodeKind::Inst) {
                inst(n.inst);
                continue;
            }
            // A nested block sees the enclosing values; its own values die with it.
            for (Block* sub : {&n.body, &n.elseBody}) {
                scopes_.emplace_back();
                block(*sub);
                scopes_.pop_back();
            }
        }
    }

    void inst(Inst& in) {
        for (ValueId& a : in.args) a = resolve(a);
        if (in.result == kNoValue) return;
        if (ValueId same = simplify(in); same != kNoValue) {
            replace_[in.result] = same;
            return;
        }
        fold(in);
        if (in.op == Op::Const) consts_[in.result] = &in;
        if (!pure(m_, in)) return;
        Key k = keyOf(in);
        for (auto s = scopes_.rbegin(); s != scopes_.rend(); ++s)
            if (auto it = s->find(k); it != s->end()) {
                replace_[in.result] = it->second;
                return;
            }
        scopes_.back().emplace(std::move(k), in.result);
    }

    const Inst* constant(ValueId v) const {
        auto it = consts_.find(v);
        return it == consts_.end() ? nullptr : it->second;
    }

    // Component c of a constant, splatting scalars.
    static uint32_t lane(const Inst& c, unsigned i) { return c.lits.size() == 1 ? c.lits[0] : c.lits[i]; }

    bool allLanes(ValueId v, uint32_t bits) const {
        const Inst* c = constant(v);
        return c && std::all_of(c->lits.begin(), c->lits.end(), [&](uint32_t l) { return l == bits; });
    }

    // Exact identities that make an instruction a copy of an operand; returns that operand.
    ValueId simplify(const Inst& in) const {
        const Type t = in.type;
        auto& a = in.args;
        uint32_t one = t.scalar == Scalar::F32 ? 0x3F800000u : 1u;
        switch (in.op) {
            case Op::Add:
            case Op::Or:
            case Op::Xor:
                if (!t.isInt()) return kNoValue;  // float x + 0 is not exact for x = -0
                if (allLanes(a[1], 0)) return a[0];
                if (allLanes(a[0], 0)) return a[1];
                return kNoValue;
            case Op::Sub:
            case Op::Shl:
            case Op::Shr: return t.isInt() && allLanes(a[1], 0) ? a[0] : kNoValue;
            case Op::Mul:
                if (!t.isInt() && t.scalar != Scalar::F32) return kNoValue;
                if (allLanes(a[1], one)) return a[0];
                if (allLanes(a[0], one)) return a[1];
                return kNoValue;
            case Op::Select:
                if (a[1] == a[2]) return a[1];
                if (fn_->typeOf(a[0]).width == 1 && constant(a[0])) return constant(a[0])->lits[0] ? a[1] : a[2];
                return kNoValue;
            case Op::Min:
            case Op::Max: return a[0] == a[1] && t.isInt() ? a[0] : kNoValue;
            default: return kNoValue;
        }
    }

    // Replaces an instruction whose operands are all constants by its value.
    void fold(Inst& in) {
        if (in.op == Op::Const || in.args.empty() || in.image || in.precise) return;
        std::vector<const Inst*> c;
        for (ValueId a : in.args) {
            const Inst* k = constant(a);
            if (!k) return;
            c.push_back(k);
        }
        std::vector<uint32_t> out;
        switch (in.op) {
            case Op::Construct:
                for (const Inst* k : c) out.insert(out.end(), k->lits.begin(), k->lits.end());
                break;
            case Op::Extract: out = {c[0]->lits[in.lits[0]]}; break;
            case Op::Shuffle:
                for (uint32_t l : in.lits) {
                    unsigned w = unsigned(c[0]->lits.size());
                    out.push_back(l < w ? c[0]->lits[l] : c[1]->lits[l - w]);
                }
                break;
            case Op::Bitcast:
                if (c[0]->lits.size() != in.type.width || in.type.scalar == Scalar::F16 ||
                    c[0]->type.scalar == Scalar::F16)
                    return;
                out = c[0]->lits;
                break;
            default:
                for (unsigned i = 0; i < in.type.width; ++i) {
                    std::optional<uint32_t> r = lane(in, c, i);
                    if (!r) return;
                    out.push_back(*r);
                }
        }
        Type t = in.type;
        ValueId res = in.result;
        in = Inst{};
        in.op = Op::Const;
        in.type = t;
        in.result = res;
        in.lits = std::move(out);
    }

    // One component of a folded arithmetic op; nullopt when it is not folded.
    std::optional<uint32_t> lane(const Inst& in, const std::vector<const Inst*>& c, unsigned i) const {
        Scalar s = fn_->typeOf(in.args[0]).scalar;
        uint32_t x = lane(*c[0], i), y = c.size() > 1 ? lane(*c[1], i) : 0;
        auto bit = [](bool b) { return uint32_t(b); };
        if (s == Scalar::I32 || s == Scalar::U32 || s == Scalar::Bool) {
            bool sgn = s == Scalar::I32;
            int32_t xi = int32_t(x), yi = int32_t(y);
            switch (in.op) {
                case Op::Add: return x + y;
                case Op::Sub: return x - y;
                case Op::Mul: return x * y;
                case Op::And: return x & y;
                case Op::Or: return x | y;
                case Op::Xor: return x ^ y;
                case Op::Not: return s == Scalar::Bool ? !x : ~x;
                case Op::Neg: return 0u - x;
                case Op::Shl: return y < 32 ? std::optional(x << y) : std::nullopt;
                case Op::Shr: return y < 32 ? std::optional(sgn ? uint32_t(xi >> yi) : x >> y) : std::nullopt;
                case Op::Min: return sgn ? uint32_t(std::min(xi, yi)) : std::min(x, y);
                case Op::Max: return sgn ? uint32_t(std::max(xi, yi)) : std::max(x, y);
                case Op::Eq: return bit(x == y);
                case Op::Ne: return bit(x != y);
                case Op::Lt: return bit(sgn ? xi < yi : x < y);
                case Op::Le: return bit(sgn ? xi <= yi : x <= y);
                case Op::Gt: return bit(sgn ? xi > yi : x > y);
                case Op::Ge: return bit(sgn ? xi >= yi : x >= y);
                case Op::Select: return lane(*c[0], i) ? lane(*c[1], i) : lane(*c[2], i);
                case Op::Convert:
                    if (in.type.scalar == Scalar::F32) return std::bit_cast<uint32_t>(sgn ? float(xi) : float(x));
                    if (in.type.isBool()) return bit(x != 0);
                    if (in.type.isInt()) return x;  // i32 <-> u32 keeps the bits, bool is 0/1
                    return std::nullopt;
                default: return std::nullopt;
            }
        }
        if (s != Scalar::F32) return std::nullopt;  // f16 arithmetic is left to the GPU compiler
        float fx = std::bit_cast<float>(x), fy = std::bit_cast<float>(y);
        auto f = [](float v) { return std::optional(std::bit_cast<uint32_t>(v)); };
        switch (in.op) {
            case Op::Add: return f(fx + fy);
            case Op::Sub: return f(fx - fy);
            case Op::Mul: return f(fx * fy);
            case Op::Neg: return x ^ 0x80000000u;
            case Op::Abs: return x & 0x7FFFFFFFu;
            case Op::Eq: return bit(fx == fy);
            case Op::Ne: return bit(fx != fy);
            case Op::Lt: return bit(fx < fy);
            case Op::Le: return bit(fx <= fy);
            case Op::Gt: return bit(fx > fy);
            case Op::Ge: return bit(fx >= fy);
            case Op::Convert:
                // Out-of-range float-to-int conversions are undefined in MSL: leave them to run.
                if (in.type.scalar == Scalar::I32 && std::abs(fx) < 2147483648.0f) return uint32_t(int32_t(fx));
                if (in.type.scalar == Scalar::U32 && fx >= 0.0f && fx < 4294967296.0f) return uint32_t(fx);
                return std::nullopt;
            default: return std::nullopt;
        }
    }

    Module& m_;
    Function* fn_ = nullptr;
    std::unordered_map<ValueId, ValueId> replace_;
    std::unordered_map<ValueId, const Inst*> consts_;
    std::vector<std::map<Key, ValueId>> scopes_;
};

}  // namespace

void removeDeadValues(Function& f) {
    for (bool again = true; again;) {
        std::unordered_map<ValueId, unsigned> uses;
        auto count = [&](const Block& b, auto& self) -> void {
            for (const Node& n : b) {
                if (n.value != kNoValue) ++uses[n.value];
                for (ValueId a : n.inst.args) ++uses[a];
                self(n.body, self);
                self(n.elseBody, self);
            }
        };
        count(f.body, count);
        again = false;
        auto sweep = [&](Block& b, auto& self) -> void {
            std::erase_if(b, [&](const Node& n) {
                bool dead = n.kind == NodeKind::Inst && n.inst.result != kNoValue && !hasEffect(n.inst.op) &&
                            !uses.contains(n.inst.result);
                again |= dead;
                return dead;
            });
            for (Node& n : b) {
                self(n.body, self);
                self(n.elseBody, self);
            }
        };
        sweep(f.body, sweep);
    }
}

void optimize(Module& module) { Optimizer(module).run(); }

}  // namespace vpir
