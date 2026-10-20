// Wave-64 lowering: subgroup_* ops of a GCN module (Module::waveSize 64) in a compute workgroup of exactly 64
// invocations, rewritten for Apple's 32-lane SIMD groups. A wave of 64 is the two SIMD groups of the workgroup
// (invocation i is lane i & 63 of the wave; with a 64-invocation workgroup that is its local index). The groups
// exchange values through threadgroup memory appended after the module's own `shared` bytes, with
// barriers on both sides of every exchange (the second keeps the next exchange from overwriting
// values a slower lane has not read yet). All invocations of the workgroup must reach each op together: the
// GCN lifter's wave mode makes control flow wave-uniform, which is what a hardware wave gives for free.
//
//   ballot(c)      the 32-lane ballot of each group is stored; both words are read back: {lo, hi, 0, 0}.
//   any / all      from that ballot.
//   broadcast / shuffle / read_first   every lane stores its value in its own slot; each lane reads the slot of
//                  the lane it names.
//   subgroup_invocation / subgroup_size  local index (< 64) / 64.
#include <algorithm>
#include <functional>

#include "vp4ir/builder.h"
#include "vp4ir/passes.h"

namespace vp4ir {

namespace {

constexpr uint32_t kSlotBytes = 16;                 // one lane's value, up to vec4 of 32-bit components
constexpr uint32_t kBallotOffset = 64 * kSlotBytes;  // two words (one per SIMD group) after the lane slots
constexpr uint32_t kScratchBytes = kBallotOffset + 16;

bool isSubgroupOp(Op op) { return op >= Op::SubgroupBallot && op <= Op::SubgroupAll; }

bool usesWaveOps(const Block& b) {
    return std::any_of(b.begin(), b.end(), [](const Node& n) {
        return (n.kind == NodeKind::Inst && isSubgroupOp(n.inst.op)) || usesWaveOps(n.body) || usesWaveOps(n.elseBody);
    });
}

class Lowering {
public:
    Lowering(Module& m, uint32_t base, uint32_t indexInput) : m_(m), base_(base), indexInput_(indexInput) {}

    std::string run() {
        for (Function& f : m_.functions) {
            if (!usesWaveOps(f.body) && !loadsSubgroupBuiltin(f.body)) continue;
            fn_ = &f;
            // lane id and its slot address, at the function start (they dominate everything).
            Block seq;
            ValueId idx = emitInst(seq, Op::LoadInput, kU32, {}, indexInput_);
            lane_ = bin(seq, Op::And, idx, cu(seq, 63), kU32);
            ValueId slot = bin(seq, Op::Shl, lane_, cu(seq, 4), kU32);
            slot_ = bin(seq, Op::Add, slot, cu(seq, base_), kU32);
            simd_ = bin(seq, Op::Shr, lane_, cu(seq, 5), kU32);
            rewrite(f.body);
            f.body.insert(f.body.begin(), std::make_move_iterator(seq.begin()), std::make_move_iterator(seq.end()));
            if (!error_.empty()) return error_;
        }
        return error_;
    }

private:
    bool loadsSubgroupBuiltin(const Block& b) const {
        return std::any_of(b.begin(), b.end(), [&](const Node& n) {
            if (n.kind == NodeKind::Inst && n.inst.op == Op::LoadInput && n.inst.sym < m_.inputs.size() &&
                (m_.inputs[n.inst.sym].builtin == Builtin::SubgroupInvocation ||
                 m_.inputs[n.inst.sym].builtin == Builtin::SubgroupSize))
                return true;
            return loadsSubgroupBuiltin(n.body) || loadsSubgroupBuiltin(n.elseBody);
        });
    }

    ValueId fresh(Type t) {
        fn_->valueTypes.push_back(t);
        return ValueId(fn_->valueTypes.size() - 1);
    }
    ValueId emitInst(Block& seq, Op op, Type t, std::vector<ValueId> args, uint32_t sym = 0, std::vector<uint32_t> lits = {}) {
        Node n;
        n.kind = NodeKind::Inst;
        n.inst.op = op;
        n.inst.type = t;
        n.inst.result = t.isVoid() ? kNoValue : fresh(t);
        n.inst.args = std::move(args);
        n.inst.sym = sym;
        n.inst.lits = std::move(lits);
        ValueId r = n.inst.result;
        seq.push_back(std::move(n));
        return r;
    }
    ValueId cu(Block& seq, uint32_t v) { return emitInst(seq, Op::Const, kU32, {}, 0, {v}); }
    ValueId bin(Block& seq, Op op, ValueId a, ValueId b, Type t) { return emitInst(seq, op, t, {a, b}); }

    // The 64-lane ballot as two words.
    void ballot(Block& seq, ValueId cond, ValueId& lo, ValueId& hi) {
        ValueId v = emitInst(seq, Op::SubgroupBallot, Type(Scalar::U32, 4), {cond});
        ValueId word = emitInst(seq, Op::Extract, kU32, {v}, 0, {0});
        ValueId addr = bin(seq, Op::Add, bin(seq, Op::Shl, simd_, cu(seq, 2), kU32), cu(seq, base_ + kBallotOffset), kU32);
        emitInst(seq, Op::Store, kVoid, {addr, word}, kTargetShared);
        emitInst(seq, Op::Barrier, kVoid, {});
        lo = emitInst(seq, Op::Load, kU32, {cu(seq, base_ + kBallotOffset)}, kTargetShared);
        hi = emitInst(seq, Op::Load, kU32, {cu(seq, base_ + kBallotOffset + 4)}, kTargetShared);
        emitInst(seq, Op::Barrier, kVoid, {});
    }

    // Lane `src` (u32 value, < 64) reads the value of `v` held by lane `from`.
    ValueId exchange(Block& seq, ValueId v, Type t, ValueId from) {
        emitInst(seq, Op::Store, kVoid, {slot_, v}, kTargetShared);
        emitInst(seq, Op::Barrier, kVoid, {});
        ValueId lane = bin(seq, Op::And, from, cu(seq, 63), kU32);
        ValueId addr = bin(seq, Op::Add, bin(seq, Op::Shl, lane, cu(seq, 4), kU32), cu(seq, base_), kU32);
        ValueId r = emitInst(seq, Op::Load, t, {addr}, kTargetShared);
        emitInst(seq, Op::Barrier, kVoid, {});
        return r;
    }

    void rewrite(Block& b) {
        Block out;
        for (Node& n : b) {
            if (n.kind != NodeKind::Inst) {
                rewrite(n.body);
                rewrite(n.elseBody);
                out.push_back(std::move(n));
                continue;
            }
            Inst& in = n.inst;
            if (in.op == Op::LoadInput && in.sym < m_.inputs.size()) {
                Builtin bi = m_.inputs[in.sym].builtin;
                if (bi == Builtin::SubgroupInvocation) {
                    in.sym = indexInput_;
                } else if (bi == Builtin::SubgroupSize) {
                    in = Inst{Op::Const, kU32, in.result, false, {}, 0, 0, {64}, 0};
                }
                out.push_back(std::move(n));
                continue;
            }
            if (!isSubgroupOp(in.op)) {
                out.push_back(std::move(n));
                continue;
            }
            Block seq;
            ValueId orig = in.result, produced = kNoValue;
            switch (in.op) {
                case Op::SubgroupBallot: {
                    ValueId lo, hi;
                    ballot(seq, in.args[0], lo, hi);
                    ValueId zero = cu(seq, 0);
                    produced = emitInst(seq, Op::Construct, in.type, {lo, hi, zero, zero});
                    break;
                }
                case Op::SubgroupAny: {
                    ValueId lo, hi;
                    ballot(seq, in.args[0], lo, hi);
                    produced = emitInst(seq, Op::Ne, kBool, {bin(seq, Op::Or, lo, hi, kU32), cu(seq, 0)});
                    break;
                }
                case Op::SubgroupAll: {
                    ValueId lo, hi;
                    ballot(seq, in.args[0], lo, hi);
                    ValueId ones = cu(seq, 0xffffffffu);
                    produced = emitInst(seq, Op::And, kBool, {bin(seq, Op::Eq, lo, ones, kBool), bin(seq, Op::Eq, hi, ones, kBool)});
                    break;
                }
                case Op::SubgroupBroadcast:
                case Op::SubgroupShuffle:
                case Op::SubgroupReadFirst: {
                    Type t = in.type;
                    if (t.scalar == Scalar::F16 || t.scalar == Scalar::Bool || t.scalar == Scalar::Void) {
                        error_ = "wave 64 lowering: subgroup exchange of " + toString(t) + " is not supported";
                        out.push_back(std::move(n));
                        continue;
                    }
                    ValueId from = in.op == Op::SubgroupReadFirst ? cu(seq, 0) : in.args[1];
                    produced = exchange(seq, in.args[0], t, from);
                    break;
                }
                default: break;
            }
            // The instruction that computes the answer defines the original result id.
            if (orig != kNoValue && produced != kNoValue)
                for (Node& s : seq)
                    if (s.inst.result == produced) {
                        s.inst.result = orig;
                        fn_->valueTypes[produced] = kVoid;
                    }
            for (Node& s : seq) out.push_back(std::move(s));
        }
        b = std::move(out);
    }

    Module& m_;
    Function* fn_ = nullptr;
    uint32_t base_, indexInput_;
    ValueId lane_ = kNoValue, slot_ = kNoValue, simd_ = kNoValue;
    std::string error_;
};

}  // namespace

std::string lowerWave64(Module& m) {
    if (m.waveSize != 64) return {};
    bool uses = std::any_of(m.functions.begin(), m.functions.end(), [](const Function& f) { return usesWaveOps(f.body); }) ||
                std::any_of(m.inputs.begin(), m.inputs.end(), [](const IoVar& v) {
                    return v.builtin == Builtin::SubgroupInvocation || v.builtin == Builtin::SubgroupSize;
                });
    if (!uses) {
        m.waveSize = 32;
        return {};
    }
    if (m.stage != Stage::Compute || uint64_t(m.workgroupSize[0]) * m.workgroupSize[1] * m.workgroupSize[2] != 64)
        return "wave_size 64: subgroup ops need a compute workgroup of exactly 64 invocations";
    // The local index input.
    uint32_t index = uint32_t(m.inputs.size());
    for (uint32_t i = 0; i < m.inputs.size(); ++i)
        if (m.inputs[i].builtin == Builtin::LocalInvocationIndex) index = i;
    if (index == m.inputs.size()) {
        IoVar v;
        v.name = "vp_local_index";
        v.type = kU32;
        v.builtin = Builtin::LocalInvocationIndex;
        m.inputs.push_back(v);
    }
    uint32_t base = (m.sharedBytes + 15) & ~15u;
    m.sharedBytes = base + kScratchBytes;
    std::string err = Lowering(m, base, index).run();
    if (!err.empty()) return err;
    m.waveSize = 32;  // (what remains of the subgroup ops is the native 32-lane ballot)
    return {};
}

}  // namespace vp4ir
