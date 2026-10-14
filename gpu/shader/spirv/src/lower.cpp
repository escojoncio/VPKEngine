// SPIR-V reader, second half: one entry point of a parsed module -> one VPIR module.
//
// Shape of the translation
//   * Every function is inlined into VPIR's `main` (SPIR-V forbids recursion). A callee runs inside
//     a loop that executes once, so a `return` anywhere in it is a `break`.
//   * SPIR-V's structured control flow becomes VPIR's tree: a selection is an `if`, a loop a `loop`
//     whose body starts with the header block, a switch an `if` chain inside a once-loop. A branch
//     to the continue target emits the continue construct in place (it is small: the increment) and
//     then `continue`s. Exits that cross more than one VPIR loop go through `exit code` variable.
//   * SSA values used outside the block that defines them, and every OpPhi, live in VPIR
//     variables: VPIR values are scoped to their region.
//   * Memory (Function/Private/Workgroup variables, uniform/storage/push-constant blocks) is
//     byte-addressed: explicit-layout blocks follow their Offset/ArrayStride/MatrixStride
//     decorations, the rest a packed 4-byte layout. Bools are stored as u32, f16 as f32.
//   * Interface variables are flattened to one VPIR input/output per scalar or vector. Outputs are
//     shadowed in variables (SPIR-V may read them back) and written at every return.
//   * Vertex inputs are fetched from the vertex buffers the driver lists (no vertex descriptors).
//   * Vulkan clip space has +y down; Metal's has +y up: Position.y is negated on the way out.
//   * Multiview (viewCount > 1): the driver draws instances * viewCount; the vertex shader takes
//     view = instance_id % viewCount, instance = instance_id / viewCount and writes the view to the
//     render-target layer, which the fragment shader reads back as ViewIndex.
#include <algorithm>
#include <bit>
#include <iterator>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <set>
#include <stdexcept>
#include <unordered_set>

#include "module.h"
#include "vpir/builder.h"
#include "vpvk/spirv_frontend.h"

namespace spirv {

using namespace spv;
using vpir::FunctionBuilder;
using vpir::Inst;
using vpir::kNoValue;
using vpir::Op;
using vpir::Scalar;
using vpir::Type;
using vpir::ValueId;

namespace {

struct LowerError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
[[noreturn]] void fail(const std::string& why) { throw LowerError(why); }

// A SPIR-V value flattened to VPIR values: one per scalar or vector leaf, in declaration order
// (matrices by column, arrays by element, structs by member).
struct Val {
    std::vector<ValueId> v;
};

// Image/sampler handles are not values in VPIR: they name resources.
struct Handle {
    int texture = -1;   // index into vpir::Module::resources
    int sampler = -1;
    int buffer = -1;    // texel buffers: the Buffer resource
    vpgpu::Format texelFormat = vpgpu::Format::Undefined;
    uint32_t imageType = 0;  // OpTypeImage id
};

// A resolved pointer.
struct RPtr {
    enum Kind : uint8_t { Mem, Input, Output, Handles, ImageTexel } kind = Mem;
    uint32_t type = 0;          // pointee type id
    // Mem
    uint32_t target = 0;        // resource index or vpir::kTarget*
    ValueId dyn = kNoValue;     // dynamic byte offset (u32), added to `off`
    uint32_t off = 0;
    bool explicitLayout = false;
    uint32_t rowMajorStride = 0;  // the pointee is (part of) a row-major matrix with this MatrixStride
    uint32_t compStride = 4;      // vector component stride (row-major columns: the matrix stride)
    uint32_t matStride = 0;       // MatrixStride of the member the pointee came from (explicit layouts)
    // Input/Output: first leaf
    uint32_t leaf = 0;
    // Handles: resource binding base and element
    uint32_t global = 0;
    uint32_t element = 0;
    // ImageTexel
    Handle image;
    ValueId coord = kNoValue;
};

struct IoLeaf {
    enum Kind : uint8_t { Io, Dropped, VertexAttr, ViewIndex, Helper, InstanceIndex, VertexIndex,
                          SampleMaskIn, FrontFacing } kind = Io;
    int index = -1;       // vpir input/output index (Io)
    uint32_t location = 0;
    uint32_t type = 0;    // SPIR-V leaf type
};

struct IoVarInfo {
    bool output = false;
    std::vector<IoLeaf> leaves;
};

class Lowering;

// One inlined function body.
struct Scope {
    const FunctionDef* fn = nullptr;
    std::unordered_map<uint32_t, Val> vals;
    std::unordered_map<uint32_t, uint32_t> valBlock;   // id -> label of the block that defined it
    std::unordered_map<uint32_t, std::vector<uint32_t>> vars;  // spilled id -> VPIR variables
    std::unordered_map<uint32_t, Handle> handles;
    std::unordered_map<uint32_t, RPtr> params;   // pointer parameters (resolved at the call)
    std::unordered_map<uint32_t, uint32_t> locals;  // OpVariable id -> private slot target
    std::unordered_map<uint32_t, Instr> chains;  // access chains, resolved at use
    std::unordered_map<uint32_t, Instr> texelPointers;
    std::unordered_set<uint32_t> spilled;
    std::unordered_map<uint32_t, std::vector<uint32_t>> phiVars;  // phi id -> its variables
    uint32_t block = 0;             // label being emitted
    std::vector<uint32_t> retVars;  // callee: return value variables
    uint32_t returnType = 0;
};

enum class Action : uint8_t { Break, Continue, Header };

struct Ctx {
    enum Kind : uint8_t { Loop, Switch, Func } kind = Loop;
    uint32_t header = 0, merge = 0, cont = 0;
    Scope* scope = nullptr;
    struct Pending {
        uint32_t code;
        size_t target;
        Action action;
    };
    std::vector<Pending> pending;
};

uint32_t floatBits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}

class Lowering {
public:
    Lowering(const SpirvModule& m, const SpirvLowerOptions& o, vpir::Module& out, SpirvInterface& iface)
        : m_(m), o_(o), out_(out), iface_(iface) {}

    void run();

private:
    const SpirvModule& m_;
    const SpirvLowerOptions& o_;
    vpir::Module& out_;
    SpirvInterface& iface_;
    std::unique_ptr<FunctionBuilder> b_;
    const EntryPoint* entry_ = nullptr;
    vpir::Stage stage_ = vpir::Stage::Vertex;

    std::unordered_map<uint32_t, IoVarInfo> io_;          // interface variable -> leaves
    std::vector<uint32_t> outputShadow_;                  // vpir output index -> variable
    std::unordered_map<uint32_t, uint32_t> sharedOffset_; // Workgroup variable -> byte offset
    std::unordered_map<uint32_t, uint32_t> privateGlobal_;// Private global -> private slot target
    std::map<std::pair<int, uint32_t>, int> resIndex_;    // (kind, slot) -> resource index
    std::set<int> written_;                               // buffer/texture resources stored to
    std::map<int, std::pair<uint32_t, uint32_t>> resBinding_;  // resource index -> (set, binding)
    std::vector<Ctx> ctx_;
    std::vector<std::unique_ptr<Scope>> scopes_;
    Scope* scope_ = nullptr;
    uint32_t exitVar_ = UINT32_MAX;
    uint32_t nextCode_ = 1;
    int positionInput_ = -1, layerInput_ = -1, instanceInput_ = -1, vertexInput_ = -1;
    int layerOutput_ = -1;
    bool writesLayer_ = false;
    std::unordered_map<uint32_t, uint64_t> specValue_;

    // ---- types ---------------------------------------------------------------------------------
    const TypeInfo& T(uint32_t id) const {
        auto t = m_.type(id);
        if (!t) fail("unknown type %" + std::to_string(id));
        return *t;
    }
    uint32_t typeOf(uint32_t id) const {
        auto it = m_.resultType.find(id);
        if (it == m_.resultType.end()) fail("no type for %" + std::to_string(id));
        return it->second;
    }
    Scalar scalarOf(uint32_t typeId) const {
        const TypeInfo& t = T(typeId);
        switch (t.kind) {
            case TypeKind::Bool: return Scalar::Bool;
            case TypeKind::Int:
                if (t.width != 32) fail(std::to_string(t.width) + "-bit integers are not supported");
                return t.sign ? Scalar::I32 : Scalar::U32;
            case TypeKind::Float:
                if (t.width == 16) return Scalar::F16;
                if (t.width != 32) fail(std::to_string(t.width) + "-bit floats are not supported");
                return Scalar::F32;
            default: fail("not a scalar type");
        }
    }
    bool isLeaf(uint32_t typeId) const {
        auto k = T(typeId).kind;
        return k == TypeKind::Bool || k == TypeKind::Int || k == TypeKind::Float || k == TypeKind::Vector;
    }
    Type vt(uint32_t typeId) const {
        const TypeInfo& t = T(typeId);
        if (t.kind == TypeKind::Vector) return Type(scalarOf(t.elem), uint8_t(t.count));
        return Type(scalarOf(typeId));
    }
    uint32_t arrayLength(const TypeInfo& t) const {
        if (t.kind != TypeKind::Array) fail("not a sized array");
        auto c = m_.constants.find(t.lengthId);
        if (c == m_.constants.end()) fail("array length is not a constant");
        return uint32_t(constScalar(t.lengthId));
    }
    // Leaf types of a composite, in order.
    void leaves(uint32_t typeId, std::vector<uint32_t>& out) const {
        const TypeInfo& t = T(typeId);
        switch (t.kind) {
            case TypeKind::Matrix:
                for (uint32_t i = 0; i < t.count; ++i) out.push_back(t.elem);
                break;
            case TypeKind::Array: {
                uint32_t n = arrayLength(t);
                for (uint32_t i = 0; i < n; ++i) leaves(t.elem, out);
                break;
            }
            case TypeKind::Struct:
                for (uint32_t mt : t.members) leaves(mt, out);
                break;
            case TypeKind::RuntimeArray: fail("runtime arrays cannot be values");
            default: out.push_back(typeId);
        }
    }
    uint32_t leafCount(uint32_t typeId) const {
        std::vector<uint32_t> l;
        leaves(typeId, l);
        return uint32_t(l.size());
    }

    // ---- constants -----------------------------------------------------------------------------
    uint64_t constScalar(uint32_t id) const {
        auto it = m_.constants.find(id);
        if (it == m_.constants.end()) fail("%" + std::to_string(id) + " is not a constant");
        const ConstantDef& c = it->second;
        if (c.isNull) return 0;
        if (c.specOp.op) return foldSpecOp(id, c);
        if (c.composite) fail("composite used as a scalar constant");
        if (c.spec) {
            auto s = specValue_.find(c.specId);
            if (s != specValue_.end()) {
                if (T(c.type).kind == TypeKind::Bool) return s->second != 0;
                return s->second;
            }
        }
        return c.bits;
    }
    // OpSpecConstantOp on scalars: the operations GLSL front ends produce for array sizes and
    // workgroup sizes.
    uint64_t foldSpecOp(uint32_t id, const ConstantDef& c) const {
        const Instr& in = c.specOp;
        uint32_t op = m_.w(in, 3);
        auto a = [&](uint32_t i) -> uint64_t { return constScalar(m_.w(in, 4 + i)); };
        auto s32 = [&](uint32_t i) { return int32_t(uint32_t(a(i))); };
        switch (op) {
            case OpIAdd: return uint32_t(a(0) + a(1));
            case OpISub: return uint32_t(a(0) - a(1));
            case OpIMul: return uint32_t(a(0) * a(1));
            case OpUDiv: return a(1) ? uint32_t(a(0) / a(1)) : 0;
            case OpSDiv: return s32(1) ? uint32_t(s32(0) / s32(1)) : 0;
            case OpUMod: return a(1) ? uint32_t(a(0) % a(1)) : 0;
            case OpShiftLeftLogical: return uint32_t(a(0) << (a(1) & 31));
            case OpShiftRightLogical: return uint32_t(uint32_t(a(0)) >> (a(1) & 31));
            case OpBitwiseAnd: return a(0) & a(1);
            case OpBitwiseOr: return a(0) | a(1);
            case OpBitwiseXor: return a(0) ^ a(1);
            case OpLogicalAnd: return a(0) && a(1);
            case OpLogicalOr: return a(0) || a(1);
            case OpLogicalNot: return !a(0);
            case OpLogicalEqual: case OpIEqual: return a(0) == a(1);
            case OpLogicalNotEqual: case OpINotEqual: return a(0) != a(1);
            case OpULessThan: return a(0) < a(1);
            case OpSLessThan: return s32(0) < s32(1);
            case OpUGreaterThan: return a(0) > a(1);
            case OpSGreaterThan: return s32(0) > s32(1);
            case OpSelect: return a(0) ? a(1) : a(2);
            case OpSNegate: return uint32_t(-s32(0));
            case OpNot: return uint32_t(~a(0));
            case OpUConvert: case OpSConvert: return uint32_t(a(0));
            case OpCompositeExtract: {
                uint32_t comp = m_.w(in, 4);
                for (uint32_t i = 5; i < in.count; ++i) {
                    auto cc = m_.constants.find(comp);
                    if (cc == m_.constants.end() || !cc->second.composite) fail("bad spec composite extract");
                    const uint32_t k = m_.w(in, i);
                    if (k >= cc->second.parts.size()) fail("spec composite extract index " + std::to_string(k) + " out of range");
                    comp = cc->second.parts[k];
                }
                return constScalar(comp);
            }
            default: fail("unsupported OpSpecConstantOp " + std::to_string(op) + " (%" + std::to_string(id) + ")");
        }
    }
    Val constVal(uint32_t id) {
        auto it = m_.constants.find(id);
        const ConstantDef& c = it->second;
        uint32_t type = c.type;
        if (c.isNull || (c.composite && false)) return zero(type);
        if (c.composite) {
            const TypeInfo& t = T(type);
            if (t.kind == TypeKind::Vector) {
                std::vector<uint32_t> bits;
                for (uint32_t p : c.parts) bits.push_back(scalarBits(p));
                return Val{{constant(vt(type), bits)}};
            }
            Val r;
            for (uint32_t p : c.parts) {
                Val pv = constVal(p);
                r.v.insert(r.v.end(), pv.v.begin(), pv.v.end());
            }
            return r;
        }
        return Val{{constant(vt(type), {scalarBits(id)})}};
    }
    uint32_t scalarBits(uint32_t id) const {
        auto it = m_.constants.find(id);
        uint64_t v = constScalar(id);
        const TypeInfo& t = T(it->second.type);
        if (t.kind == TypeKind::Float && t.width == 64) {
            double d;
            std::memcpy(&d, &v, 8);
            return floatBits(float(d));
        }
        return uint32_t(v);
    }
    ValueId constant(Type t, const std::vector<uint32_t>& bits) {
        Inst in;
        in.op = Op::Const;
        in.type = t;
        in.lits = bits;
        if (t.isBool())
            for (auto& l : in.lits) l = l ? 1 : 0;
        return b_->emit(in);
    }
    ValueId splat(Type t, uint32_t bits) { return constant(t, std::vector<uint32_t>(t.width, bits)); }
    ValueId f32(float f) { return b_->constF32(f); }
    ValueId u32(uint32_t v) { return b_->constU32(v); }
    ValueId zeroOf(Type t) {
        if (t.isBool()) return splat(t, 0);
        return splat(t, 0);
    }
    Val zero(uint32_t typeId) {
        std::vector<uint32_t> l;
        leaves(typeId, l);
        Val r;
        for (uint32_t lt : l) r.v.push_back(zeroOf(vt(lt)));
        return r;
    }

    // ---- small VPIR helpers --------------------------------------------------------------------
    Type typeOfValue(ValueId v) { return b_fn().typeOf(v); }
    vpir::Function& b_fn() { return out_.functions.back(); }
    ValueId un(Op op, ValueId a, Type t) { return b_->unary(op, a, t); }
    ValueId bin(Op op, ValueId a, ValueId c, Type t) { return b_->binary(op, a, c, t); }
    ValueId tri(Op op, ValueId a, ValueId c, ValueId d, Type t) { return b_->ternary(op, a, c, d, t); }
    // `v` as type `t` when they differ only in how the bits are read (int signedness, or float vs
    // int of one width: SPIR-V composites carry exact types, VPIR values may be typed otherwise).
    ValueId sameBits(ValueId v, Type t) {
        Type vt0 = typeOfValue(v);
        if (vt0 == t || vt0.width != t.width || vt0.isBool() || t.isBool() || vt0.byteSize() != t.byteSize()) return v;
        return un(Op::Bitcast, v, t);
    }
    ValueId ext(ValueId v, uint32_t i) {
        Type t = typeOfValue(v);
        if (t.width == 1) return v;
        return b_->extract(v, i, t.element());
    }
    ValueId vecOf(const std::vector<ValueId>& parts, Type t) {
        if (t.width == 1) return parts.at(0);
        return b_->construct(parts, t);
    }
    ValueId splatValue(ValueId s, uint8_t width) {
        if (width == 1) return s;
        std::vector<ValueId> p(width, s);
        return b_->construct(p, typeOfValue(s).withWidth(width));
    }
    ValueId cast(ValueId v, Type to) {
        Type from = typeOfValue(v);
        if (from == to) return v;
        if (from.width != to.width) fail("internal: cast between widths");
        if (from.isBool()) return tri(Op::Select, v, splat(to, to.isFloat() ? floatBits(1.f) : 1), splat(to, 0), to);
        if (to.isBool()) return bin(Op::Ne, v, splat(from, 0), to);
        if (from.byteSize() == to.byteSize() && from.isInt() && to.isInt()) return un(Op::Bitcast, v, to);
        return un(Op::Convert, v, to);
    }
    ValueId bitcast(ValueId v, Type to) {
        Type from = typeOfValue(v);
        if (from == to) return v;
        return un(Op::Bitcast, v, to);
    }
    ValueId select(ValueId c, ValueId a, ValueId d) { return tri(Op::Select, c, a, d, typeOfValue(a)); }
    ValueId loadVar(uint32_t var) { return b_->loadVar(var); }
    uint32_t newVar(Type t) { return b_->variable(t); }
    uint32_t exitVar() {
        if (exitVar_ == UINT32_MAX) {
            exitVar_ = newVar(vpir::kU32);
            // Initialised at the top of main (see run()).
        }
        return exitVar_;
    }

    // ---- values ---------------------------------------------------------------------------------
    bool isConstant(uint32_t id) const { return m_.constants.count(id) != 0; }

    Val get(uint32_t id) {
        if (isConstant(id)) return constVal(id);
        Scope& s = *scope_;
        if (auto pv = s.phiVars.find(id); pv != s.phiVars.end()) {
            auto it = s.vals.find(id);
            if (it != s.vals.end() && s.valBlock[id] == s.block) return it->second;
            Val r;
            for (uint32_t var : pv->second) r.v.push_back(loadVar(var));
            return r;
        }
        auto it = s.vals.find(id);
        if (s.spilled.count(id)) {
            if (it != s.vals.end() && s.valBlock[id] == s.block) return it->second;
            auto vv = s.vars.find(id);
            if (vv == s.vars.end()) fail("value %" + std::to_string(id) + " used before its definition");
            Val r;
            for (uint32_t var : vv->second) r.v.push_back(loadVar(var));
            return r;
        }
        if (it == s.vals.end()) fail("value " + m_.nameOf(id) + " is not available");
        return it->second;
    }
    ValueId get1(uint32_t id) {
        Val v = get(id);
        if (v.v.size() != 1) fail("expected a scalar or vector for " + m_.nameOf(id));
        return v.v[0];
    }
    void define(uint32_t id, Val v) {
        Scope& s = *scope_;
        s.vals[id] = v;
        s.valBlock[id] = s.block;
        if (s.spilled.count(id)) {
            auto& vars = s.vars[id];
            if (vars.empty())
                for (ValueId x : v.v) vars.push_back(newVar(typeOfValue(x)));
            for (size_t i = 0; i < v.v.size(); ++i) b_->storeVar(vars[i], v.v[i]);
        }
    }
    void define1(uint32_t id, ValueId v) { define(id, Val{{v}}); }

    // Which results of `fn` are used outside their defining block.
    void analyse(Scope& s) {
        const FunctionDef& f = *s.fn;
        std::unordered_map<uint32_t, uint32_t> defBlock;
        for (const Block& bl : f.blocks)
            for (const Instr& in : bl.body)
                if (in.count >= 3 && m_.resultType.count(m_.w(in, 2)) && m_.resultType.at(m_.w(in, 2)) == m_.w(in, 1))
                    defBlock[m_.w(in, 2)] = bl.label;
        // Pointers are resolved where they are used, so a use of an access chain is a use of its
        // operands there.
        std::unordered_map<uint32_t, const Instr*> derived;
        for (const Block& bl : f.blocks)
            for (const Instr& in : bl.body)
                if ((in.op == OpAccessChain || in.op == OpInBoundsAccessChain || in.op == OpPtrAccessChain ||
                     in.op == OpInBoundsPtrAccessChain || in.op == OpImageTexelPointer) && in.count >= 4)
                    derived[m_.w(in, 2)] = &in;
        std::function<void(uint32_t, uint32_t, int)> use = [&](uint32_t id, uint32_t block, int depth) {
            auto d = defBlock.find(id);
            if (d != defBlock.end() && d->second != block) s.spilled.insert(id);
            auto c = derived.find(id);
            if (c != derived.end() && depth < 32)
                for (uint32_t i = 3; i < c->second->count; ++i) use(m_.w(*c->second, i), block, depth + 1);
        };
        for (const Block& bl : f.blocks) {
            for (const Instr& in : bl.body) {
                if (in.op == OpPhi) {
                    for (uint32_t i = 3; i + 1 < in.count; i += 2) use(m_.w(in, i), m_.w(in, i + 1), 0);
                    continue;
                }
                for (uint32_t i = 1; i < in.count; ++i) use(m_.w(in, i), bl.label, 0);
            }
            for (const Instr* in : {&bl.merge, &bl.terminator})
                if (in->op)
                    for (uint32_t i = 1; i < in->count; ++i) use(m_.w(*in, i), bl.label, 0);
        }
    }

    // ---- resources ------------------------------------------------------------------------------
    const SpirvBinding* binding(uint32_t global) const {
        auto d = m_.deco(global);
        uint32_t set = d ? d->get(Deco::DescriptorSet) : 0, bind = d ? d->get(Deco::Binding) : 0;
        for (const SpirvBinding& sb : o_.layout)
            if (sb.set == set && sb.binding == bind) return &sb;
        fail("descriptor set " + std::to_string(set) + " binding " + std::to_string(bind) + " (" + m_.nameOf(global) +
             ") is not in the pipeline layout");
    }
    void report(const SpirvBinding& sb, bool written) {
        for (auto& r : iface_.bindings)
            if (r.set == sb.set && r.binding == sb.binding) {
                r.written = r.written || written;
                return;
            }
        SpirvBinding c = sb;
        c.written = written;
        iface_.bindings.push_back(c);
    }
    int resource(vpir::ResourceKind kind, uint32_t slot, const std::function<void(vpir::Resource&)>& init) {
        auto key = std::make_pair(int(kind), slot);
        auto it = resIndex_.find(key);
        if (it != resIndex_.end()) return it->second;
        vpir::Resource r;
        r.kind = kind;
        r.index = slot;
        static const char* prefix[] = {"b", "c", "t", "s", "u", "bs"};  // by ResourceKind
        static_assert(std::size(prefix) == size_t(vpir::ResourceKind::BufferSize) + 1);
        r.name = std::string(prefix[int(kind)]) + std::to_string(slot);
        r.access = vpir::Access::ReadWrite;
        if (init) init(r);
        out_.resources.push_back(r);
        int idx = int(out_.resources.size() - 1);
        resIndex_[key] = idx;
        return idx;
    }

    Scalar imageElem(const TypeInfo& img) const {
        const TypeInfo& st = T(img.sampledType);
        if (st.kind == TypeKind::Int) return st.sign ? Scalar::I32 : Scalar::U32;
        if (st.kind == TypeKind::Float) return Scalar::F32;
        // Void sampled type: take it from the format.
        switch (img.format) {
            case ImgFmt::R32i: case ImgFmt::Rg32i: case ImgFmt::Rgba32i: case ImgFmt::R16i: case ImgFmt::Rgba16i:
            case ImgFmt::R8i: case ImgFmt::Rgba8i: case ImgFmt::Rg16i: case ImgFmt::Rg8i:
                return Scalar::I32;
            case ImgFmt::R32ui: case ImgFmt::Rg32ui: case ImgFmt::Rgba32ui: case ImgFmt::R16ui: case ImgFmt::Rgba16ui:
            case ImgFmt::R8ui: case ImgFmt::Rgba8ui: case ImgFmt::Rg16ui: case ImgFmt::Rg8ui: case ImgFmt::Rgb10a2ui:
                return Scalar::U32;
            default: return Scalar::F32;
        }
    }
    vpir::TexDim texDim(const TypeInfo& img) const {
        bool arr = img.arrayed != 0;
        switch (img.dim) {
            // 1D images are 2D textures one texel high on the GPU (Metal 1D textures have no
            // mips and no render targets): coordinates get y (oneD() call sites), queries drop it.
            case Dim::Dim1D: return arr ? vpir::TexDim::D2Array : vpir::TexDim::D2;
            case Dim::Dim2D:
            case Dim::Rect: return arr ? vpir::TexDim::D2Array : vpir::TexDim::D2;
            case Dim::SubpassData:
                // Multiview input attachments are bound as the 2D-array view of the multiview
                // attachment: declare texture2d_array and read the layer of this view.
                return (arr || multiview()) ? vpir::TexDim::D2Array : vpir::TexDim::D2;
            case Dim::Dim3D: return vpir::TexDim::D3;
            case Dim::Cube: return arr ? vpir::TexDim::CubeArray : vpir::TexDim::Cube;
            default: fail("unsupported image dimension " + std::to_string(uint32_t(img.dim)));
        }
    }
    bool usedWithDref(uint32_t global) const {
        // Shadow samplers declared with Depth = 0 still compare: scan the whole module once.
        static thread_local std::unordered_map<const SpirvModule*, std::set<uint32_t>> cache;
        (void)cache;
        return drefGlobals_.count(global) != 0;
    }
    std::set<uint32_t> drefGlobals_;

    Handle handleFor(uint32_t global, uint32_t element) {
        const GlobalVar& g = m_.globals.at(global);
        uint32_t t = T(g.type).elem;
        while (T(t).kind == TypeKind::Array || T(t).kind == TypeKind::RuntimeArray) t = T(t).elem;
        const SpirvBinding& sb = *binding(global);
        if (element >= sb.arraySize) fail("descriptor index out of range for " + m_.nameOf(global));
        report(sb, false);
        Handle h;
        const TypeInfo& tt = T(t);
        if (tt.kind == TypeKind::Sampler) {
            h.sampler = resource(vpir::ResourceKind::Sampler, sb.slot + element, nullptr);
            return h;
        }
        uint32_t imgType = tt.kind == TypeKind::SampledImage ? tt.elem : t;
        const TypeInfo& img = T(imgType);
        h.imageType = imgType;
        if (img.dim == Dim::Buffer) {
            h.buffer = resource(vpir::ResourceKind::Buffer, sb.slot + element, [&](vpir::Resource& r) {
                r.access = img.sampled == 2 ? vpir::Access::ReadWrite : vpir::Access::Read;
            });
            h.texelFormat = sb.texelFormat != vpgpu::Format::Undefined ? sb.texelFormat : formatOfImage(img.format);
            resBinding_[h.buffer] = {sb.set, sb.binding};
            return h;
        }
        // SubpassData always has Sampled=2 in SPIR-V but is read as a plain texture (input attachment).
        bool storage = sb.kind == vpgpu::BindKind::StorageTexture ||
                       (img.sampled == 2 && img.dim != Dim::SubpassData);
        bool depth = img.depth == 1 || drefGlobals_.count(global);
        h.texture = resource(storage ? vpir::ResourceKind::StorageTexture : vpir::ResourceKind::Texture, sb.slot + element,
                             [&](vpir::Resource& r) {
                                 r.dim = texDim(img);
                                 r.elem = imageElem(img);
                                 r.depth = depth && !storage;
                                 r.multisample = img.ms && img.dim == Dim::Dim2D && !storage && !o_.multisampleAsSingle;
                                 if (storage) {
                                     auto d = m_.deco(global);
                                     bool nr = d && d->has(Deco::NonReadable), nw = d && d->has(Deco::NonWritable);
                                     r.access = nr ? vpir::Access::Write : nw ? vpir::Access::Read : vpir::Access::ReadWrite;
                                 }
                             });
        resBinding_[h.texture] = {sb.set, sb.binding};
        if (tt.kind == TypeKind::SampledImage) {
            if (sb.samplerSlot == 0xFFFF) fail("combined image sampler " + m_.nameOf(global) + " has no sampler slot");
            h.sampler = resource(vpir::ResourceKind::Sampler, sb.samplerSlot + element, nullptr);
        }
        return h;
    }
    static vpgpu::Format formatOfImage(ImgFmt f) {
        switch (f) {
            case ImgFmt::Rgba32f: return vpgpu::Format::RGBA32Float;
            case ImgFmt::Rgba16f: return vpgpu::Format::RGBA16Float;
            case ImgFmt::R32f: return vpgpu::Format::R32Float;
            case ImgFmt::Rgba8: return vpgpu::Format::RGBA8Unorm;
            case ImgFmt::Rgba8Snorm: return vpgpu::Format::RGBA8Snorm;
            case ImgFmt::Rg32f: return vpgpu::Format::RG32Float;
            case ImgFmt::Rg16f: return vpgpu::Format::RG16Float;
            case ImgFmt::R16f: return vpgpu::Format::R16Float;
            case ImgFmt::Rgba32i: return vpgpu::Format::RGBA32Sint;
            case ImgFmt::Rgba32ui: return vpgpu::Format::RGBA32Uint;
            case ImgFmt::R32i: return vpgpu::Format::R32Sint;
            case ImgFmt::R32ui: return vpgpu::Format::R32Uint;
            case ImgFmt::Rg32ui: return vpgpu::Format::RG32Uint;
            case ImgFmt::Rgba8ui: return vpgpu::Format::RGBA8Uint;
            case ImgFmt::R8: return vpgpu::Format::R8Unorm;
            case ImgFmt::Rg8: return vpgpu::Format::RG8Unorm;
            default: return vpgpu::Format::R32Uint;
        }
    }

    // Buffer-backed global (Uniform/StorageBuffer/PushConstant) -> resource index.
    int bufferResource(uint32_t global, uint32_t element, bool write) {
        const GlobalVar& g = m_.globals.at(global);
        if (g.storage == Storage::PushConstant) {
            if (o_.pushConstantSlot < 0) fail("push constants used but the layout has none");
            iface_.usesPushConstants = true;
            return resource(vpir::ResourceKind::ConstantBuffer, uint32_t(o_.pushConstantSlot), [](vpir::Resource& r) {
                r.access = vpir::Access::Read;
            });
        }
        const SpirvBinding& sb = *binding(global);
        if (element >= sb.arraySize) fail("descriptor index out of range for " + m_.nameOf(global));
        report(sb, write);
        bool constant = sb.kind == vpgpu::BindKind::ConstantBuffer;
        if (constant && write) fail("store to the uniform block " + m_.nameOf(global));
        int r = resource(constant ? vpir::ResourceKind::ConstantBuffer : vpir::ResourceKind::Buffer, sb.slot + element,
                         [&](vpir::Resource& res) { res.access = constant ? vpir::Access::Read : vpir::Access::ReadWrite; });
        if (write) written_.insert(r);
        resBinding_[r] = {sb.set, sb.binding};
        return r;
    }

    // ---- layout ---------------------------------------------------------------------------------
    // One backend array per variable (see vpir::kTargetPrivateSlot0); privateBytes stays for the
    // single shared arena (kTargetPrivate) that other frontends use.
    uint32_t newPrivateSlot(uint32_t bytes) {
        uint32_t size = (std::max(bytes, 4u) + 15u) & ~15u;
        if (out_.privateSlots.size() >= vpir::kMaxPrivateSlots) fail("too many private variables");
        out_.privateSlots.push_back(size);
        return vpir::kTargetPrivateSlot0 + uint32_t(out_.privateSlots.size() - 1);
    }

    uint32_t naturalSize(uint32_t typeId) const {
        const TypeInfo& t = T(typeId);
        switch (t.kind) {
            case TypeKind::Bool: case TypeKind::Int: case TypeKind::Float: return 4;
            case TypeKind::Vector: return 4 * t.count;
            case TypeKind::Matrix: return t.count * naturalSize(t.elem);
            case TypeKind::Array: return arrayLength(t) * naturalSize(t.elem);
            case TypeKind::Struct: {
                uint32_t s = 0;
                for (uint32_t mt : t.members) s += naturalSize(mt);
                return s;
            }
            default: fail("type has no memory layout");
        }
    }
    uint32_t memberOffset(uint32_t structId, uint32_t member, bool explicitLayout) const {
        const TypeInfo& t = T(structId);
        if (explicitLayout) {
            auto d = m_.memberDeco(structId, member);
            if (!d || !d->has(Deco::Offset)) fail("member " + std::to_string(member) + " of " + m_.nameOf(structId) + " has no Offset");
            return d->get(Deco::Offset);
        }
        uint32_t off = 0;
        for (uint32_t i = 0; i < member; ++i) off += naturalSize(t.members[i]);
        return off;
    }
    uint32_t arrayStride(uint32_t arrayType, bool explicitLayout) const {
        if (explicitLayout) {
            auto d = m_.deco(arrayType);
            if (d && d->has(Deco::ArrayStride)) return d->get(Deco::ArrayStride);
        }
        return naturalSize(T(arrayType).elem);
    }

    ValueId address(const RPtr& p) {
        if (p.dyn == kNoValue) return u32(p.off);
        if (p.off == 0) return p.dyn;
        return bin(Op::Add, p.dyn, u32(p.off), vpir::kU32);
    }
    void addOffset(RPtr& p, ValueId index, uint32_t constIndex, bool isConst, uint32_t stride) {
        if (isConst) {
            p.off += constIndex * stride;
            return;
        }
        ValueId i = cast(index, vpir::kU32);
        ValueId scaled = stride == 1 ? i : bin(Op::Mul, i, u32(stride), vpir::kU32);
        p.dyn = p.dyn == kNoValue ? scaled : bin(Op::Add, p.dyn, scaled, vpir::kU32);
    }

    // ---- pointers -------------------------------------------------------------------------------
    RPtr rootPointer(uint32_t id, bool write) {
        Scope& s = *scope_;
        if (auto pp = s.params.find(id); pp != s.params.end()) return pp->second;
        if (auto l = s.locals.find(id); l != s.locals.end()) {
            RPtr p;
            p.kind = RPtr::Mem, p.target = l->second, p.off = 0, p.type = T(typeOf(id)).elem;
            return p;
        }
        auto g = m_.globals.find(id);
        if (g == m_.globals.end()) fail("pointer " + m_.nameOf(id) + " has no root variable");
        const GlobalVar& gv = g->second;
        RPtr p;
        p.type = T(gv.type).elem;
        switch (gv.storage) {
            case Storage::Input:
            case Storage::Output: {
                auto io = io_.find(id);
                if (io == io_.end()) fail(m_.nameOf(id) + " is not in the entry point's interface");
                p.kind = gv.storage == Storage::Input ? RPtr::Input : RPtr::Output;
                p.global = id;
                return p;
            }
            case Storage::Private:
                p.kind = RPtr::Mem, p.target = privateGlobal_.at(id), p.off = 0;
                return p;
            case Storage::Workgroup:
                p.kind = RPtr::Mem, p.target = vpir::kTargetShared, p.off = sharedOffset_.at(id);
                return p;
            case Storage::Uniform:
            case Storage::StorageBuffer:
            case Storage::PushConstant: {
                p.kind = RPtr::Mem, p.explicitLayout = true, p.global = id;
                // Descriptor arrays of blocks: the element is chosen by the first index (see chain()).
                const TypeInfo& pt = T(p.type);
                bool blockArray = (pt.kind == TypeKind::Array || pt.kind == TypeKind::RuntimeArray) &&
                                  T(pt.elem).kind == TypeKind::Struct && gv.storage != Storage::PushConstant;
                if (!blockArray) p.target = uint32_t(bufferResource(id, 0, write));
                else p.target = UINT32_MAX;  // resolved with the first index
                return p;
            }
            case Storage::UniformConstant:
                p.kind = RPtr::Handles, p.global = id;
                return p;
            default: fail("unsupported storage class " + std::to_string(uint32_t(gv.storage)) + " for " + m_.nameOf(id));
        }
    }

    RPtr resolve(uint32_t id, bool write) {
        Scope& s = *scope_;
        auto c = s.chains.find(id);
        if (c == s.chains.end()) return rootPointer(id, write);
        const Instr& in = c->second;
        RPtr p = resolve(m_.w(in, 3), write);
        uint32_t first = 4;
        if (in.op == OpPtrAccessChain || in.op == OpInBoundsPtrAccessChain) {
            // The element index steps over whole pointees: only zero is supported (used by some
            // front ends for plain access).
            if (!isConstant(m_.w(in, 4)) || constScalar(m_.w(in, 4)) != 0) fail("OpPtrAccessChain with a non-zero element");
            first = 5;
        }
        for (uint32_t i = first; i < in.count; ++i) step(p, m_.w(in, i), write);
        return p;
    }

    void step(RPtr& p, uint32_t indexId, bool write) {
        const bool isConst = isConstant(indexId);
        const uint32_t ci = isConst ? uint32_t(constScalar(indexId)) : 0;
        ValueId dyn = isConst ? kNoValue : get1(indexId);
        const TypeInfo& t = T(p.type);
        if (p.kind == RPtr::Handles) {
            if (t.kind != TypeKind::Array && t.kind != TypeKind::RuntimeArray) fail("index into a non-array handle");
            if (!isConst) fail("dynamic indexing of descriptor arrays is not supported (" + m_.nameOf(p.global) + ")");
            p.element += ci;
            p.type = t.elem;
            return;
        }
        if (p.kind == RPtr::Input || p.kind == RPtr::Output) {
            if (!isConst) fail("dynamic indexing of interface variables is not supported");
            if (t.kind == TypeKind::Struct) {
                if (ci >= t.members.size()) fail("struct member index " + std::to_string(ci) + " out of range");
                for (uint32_t i = 0; i < ci; ++i) p.leaf += leafCount(t.members[i]);
                p.type = t.members.at(ci);
            } else if (t.kind == TypeKind::Array || t.kind == TypeKind::Matrix) {
                p.leaf += ci * leafCount(t.elem);
                p.type = t.elem;
            } else if (t.kind == TypeKind::Vector) {
                // Component of an interface vector: kept as a component pointer.
                p.element = ci + 1;  // 1-based component marker
                p.type = t.elem;
            } else fail("bad interface access chain");
            return;
        }
        // Memory.
        if (p.target == UINT32_MAX) {  // descriptor array of blocks
            if (!isConst) fail("dynamic indexing of buffer descriptor arrays is not supported");
            p.target = uint32_t(bufferResource(p.global, ci, write));
            p.type = t.elem;
            return;
        }
        switch (t.kind) {
            case TypeKind::Struct: {
                if (!isConst) fail("dynamic struct member index");
                if (ci >= t.members.size()) fail("struct member index " + std::to_string(ci) + " out of range");
                p.off += memberOffset(p.type, ci, p.explicitLayout);
                uint32_t mt = t.members.at(ci);
                if (p.explicitLayout) {
                    if (auto d = m_.memberDeco(p.type, ci)) {
                        bool rowMajor = d->has(Deco::RowMajor);
                        uint32_t ms = d->get(Deco::MatrixStride, 0);
                        // Matrices (or arrays of them) inside this member use the stride and order.
                        p.rowMajorStride = rowMajor ? ms : 0;
                        p.compStride = 4;
                        p.matStride = ms;
                    }
                }
                p.type = mt;
                return;
            }
            case TypeKind::Array:
            case TypeKind::RuntimeArray:
                addOffset(p, dyn, ci, isConst, arrayStride(p.type, p.explicitLayout));
                p.type = t.elem;
                return;
            case TypeKind::Matrix: {
                uint32_t colStride = p.explicitLayout && p.matStride ? p.matStride : naturalSize(t.elem);
                if (p.rowMajorStride) {
                    addOffset(p, dyn, ci, isConst, 4);  // column c starts at c * 4
                    p.compStride = p.rowMajorStride;
                } else {
                    addOffset(p, dyn, ci, isConst, colStride);
                    p.compStride = 4;
                }
                p.type = t.elem;
                return;
            }
            case TypeKind::Vector:
                addOffset(p, dyn, ci, isConst, p.compStride);
                p.type = t.elem;
                return;
            default: fail("bad access chain");
        }
    }

    // ---- memory loads and stores ----------------------------------------------------------------
    Val loadMem(const RPtr& p) {
        const TypeInfo& t = T(p.type);
        Val r;
        switch (t.kind) {
            case TypeKind::Bool: case TypeKind::Int: case TypeKind::Float: {
                r.v.push_back(loadScalarVec(p, vt(p.type)));
                return r;
            }
            case TypeKind::Vector: {
                Type vtp = vt(p.type);
                if (p.compStride == 4) {
                    r.v.push_back(loadScalarVec(p, vtp));
                    return r;
                }
                std::vector<ValueId> parts;
                for (uint32_t i = 0; i < t.count; ++i) {
                    RPtr q = p;
                    q.off += i * p.compStride;
                    parts.push_back(loadScalarVec(q, vtp.element()));
                }
                r.v.push_back(vecOf(parts, vtp));
                return r;
            }
            case TypeKind::Matrix: {
                uint32_t colStride = p.explicitLayout && p.matStride ? p.matStride : naturalSize(t.elem);
                for (uint32_t c = 0; c < t.count; ++c) {
                    RPtr q = p;
                    q.type = t.elem;
                    if (p.rowMajorStride) q.off += c * 4, q.compStride = p.rowMajorStride;
                    else q.off += c * colStride, q.compStride = 4;
                    Val cv = loadMem(q);
                    r.v.insert(r.v.end(), cv.v.begin(), cv.v.end());
                }
                return r;
            }
            case TypeKind::Array: {
                uint32_t n = arrayLength(t), stride = arrayStride(p.type, p.explicitLayout);
                for (uint32_t i = 0; i < n; ++i) {
                    RPtr q = p;
                    q.type = t.elem, q.off += i * stride;
                    Val e = loadMem(q);
                    r.v.insert(r.v.end(), e.v.begin(), e.v.end());
                }
                return r;
            }
            case TypeKind::Struct: {
                for (uint32_t i = 0; i < t.members.size(); ++i) {
                    RPtr q = p;
                    q.type = t.members[i];
                    q.off += memberOffset(p.type, i, p.explicitLayout);
                    if (p.explicitLayout) {
                        if (auto d = m_.memberDeco(p.type, i)) {
                            q.rowMajorStride = d->has(Deco::RowMajor) ? d->get(Deco::MatrixStride) : 0;
                            q.matStride = d->get(Deco::MatrixStride, 0);
                        }
                    }
                    Val e = loadMem(q);
                    r.v.insert(r.v.end(), e.v.begin(), e.v.end());
                }
                return r;
            }
            default: fail("cannot load this type from memory");
        }
    }
    ValueId loadScalarVec(const RPtr& p, Type t) {
        Type mem = t;
        if (t.isBool()) mem = vpir::kU32.withWidth(t.width);
        if (t.scalar == Scalar::F16) mem = vpir::kF32.withWidth(t.width);
        ValueId v = b_->load(p.target, address(p), mem);
        if (t.isBool()) return bin(Op::Ne, v, splat(mem, 0), t);
        if (t.scalar == Scalar::F16) return un(Op::Convert, v, t);
        return v;
    }
    void storeMem(const RPtr& p, const Val& v, size_t& at) {
        const TypeInfo& t = T(p.type);
        switch (t.kind) {
            case TypeKind::Bool: case TypeKind::Int: case TypeKind::Float:
                storeScalarVec(p, v.v.at(at++));
                return;
            case TypeKind::Vector: {
                ValueId x = v.v.at(at++);
                if (p.compStride == 4) {
                    storeScalarVec(p, x);
                    return;
                }
                for (uint32_t i = 0; i < t.count; ++i) {
                    RPtr q = p;
                    q.off += i * p.compStride;
                    storeScalarVec(q, ext(x, i));
                }
                return;
            }
            case TypeKind::Matrix: {
                uint32_t colStride = p.explicitLayout && p.matStride ? p.matStride : naturalSize(t.elem);
                for (uint32_t c = 0; c < t.count; ++c) {
                    RPtr q = p;
                    q.type = t.elem;
                    if (p.rowMajorStride) q.off += c * 4, q.compStride = p.rowMajorStride;
                    else q.off += c * colStride, q.compStride = 4;
                    storeMem(q, v, at);
                }
                return;
            }
            case TypeKind::Array: {
                uint32_t n = arrayLength(t), stride = arrayStride(p.type, p.explicitLayout);
                for (uint32_t i = 0; i < n; ++i) {
                    RPtr q = p;
                    q.type = t.elem, q.off += i * stride;
                    storeMem(q, v, at);
                }
                return;
            }
            case TypeKind::Struct:
                for (uint32_t i = 0; i < t.members.size(); ++i) {
                    RPtr q = p;
                    q.type = t.members[i];
                    q.off += memberOffset(p.type, i, p.explicitLayout);
                    if (p.explicitLayout)
                        if (auto d = m_.memberDeco(p.type, i)) {
                            q.rowMajorStride = d->has(Deco::RowMajor) ? d->get(Deco::MatrixStride) : 0;
                            q.matStride = d->get(Deco::MatrixStride, 0);
                        }
                    storeMem(q, v, at);
                }
                return;
            default: fail("cannot store this type to memory");
        }
    }
    void storeScalarVec(const RPtr& p, ValueId x) {
        Type t = typeOfValue(x);
        if (t.isBool()) x = cast(x, vpir::kU32.withWidth(t.width));
        else if (t.scalar == Scalar::F16) x = un(Op::Convert, x, vpir::kF32.withWidth(t.width));
        if (p.target < out_.resources.size()) written_.insert(int(p.target));
        b_->store(p.target, address(p), x);
    }

    // ---- interface ------------------------------------------------------------------------------
    Val loadIo(const RPtr& p) {
        const IoVarInfo& io = io_.at(p.global);
        std::vector<uint32_t> lt;
        leaves(p.type, lt);
        Val r;
        if (p.element) {  // component of a vector
            const IoLeaf& leaf = io.leaves.at(p.leaf);
            ValueId v = p.kind == RPtr::Output ? outputValue(leaf) : inputValue(leaf);
            r.v.push_back(ext(v, p.element - 1));
            return r;
        }
        for (uint32_t i = 0; i < lt.size(); ++i) {
            const IoLeaf& leaf = io.leaves.at(p.leaf + i);
            r.v.push_back(p.kind == RPtr::Output ? outputValue(leaf) : inputValue(leaf));
        }
        return r;
    }
    ValueId outputValue(const IoLeaf& leaf) {
        if (leaf.kind != IoLeaf::Io) return zeroOf(vt(leaf.type));
        return loadVar(outputShadow_.at(leaf.index));
    }
    void storeIo(const RPtr& p, const Val& v) {
        if (p.kind != RPtr::Output) fail("store to an input");
        const IoVarInfo& io = io_.at(p.global);
        if (p.element) {
            const IoLeaf& leaf = io.leaves.at(p.leaf);
            if (leaf.kind != IoLeaf::Io) return;
            uint32_t var = outputShadow_.at(leaf.index);
            ValueId cur = loadVar(var);
            Inst in;
            in.op = Op::Insert;
            in.type = typeOfValue(cur);
            // (SPIR-V types match exactly; ours may differ in signedness: bit-cast to the element)
            in.args = {cur, sameBits(v.v.at(0), in.type.element())};
            in.lits = {p.element - 1};
            b_->storeVar(var, b_->emit(in));
            return;
        }
        for (size_t i = 0; i < v.v.size(); ++i) {
            const IoLeaf& leaf = io.leaves.at(p.leaf + i);
            if (leaf.kind != IoLeaf::Io) continue;
            uint32_t var = outputShadow_.at(leaf.index);
            b_->storeVar(var, cast(v.v[i], b_fn().vars[var].type));
        }
    }
    ValueId viewIndex() {
        if (stage_ == vpir::Stage::Fragment) {
            if (!multiview()) return u32(0);
            if (layerInput_ < 0) {
                // Declared on first use only: a fragment shader that never reads the view gives the
                // same MSL in multiview and single-view passes (one library, precompile hits).
                vpir::IoVar v;
                v.builtin = vpir::Builtin::Layer, v.name = "layer", v.type = vpir::kU32;
                v.interp = vpir::Interp::Flat;
                layerInput_ = addInput(v);
            }
            return b_->loadInput(uint32_t(layerInput_), vpir::kU32);
        }
        if (!multiview()) return u32(0);
        ValueId k = o_.viewCount <= 1 || instanceInput_ < 0
                        ? u32(0)
                        : bin(Op::Rem, b_->loadInput(uint32_t(instanceInput_), vpir::kU32), u32(o_.viewCount), vpir::kU32);
        return viewBit(k);
    }
    // Multiview: instances are drawn viewCount times; the k-th view of the mask renders to layer
    // (and is gl_ViewIndex) = the index of the mask's k-th set bit (k itself unless o_.viewMask).
    bool multiview() const { return o_.viewCount > 1 || o_.viewMask != 0; }
    ValueId viewBit(ValueId k) {
        if (!o_.viewMask) return k;
        ValueId v = u32(uint32_t(std::countr_zero(o_.viewMask)));
        uint32_t j = 0;
        for (uint32_t b = 0; b < 32; ++b) {
            if (!(o_.viewMask >> b & 1)) continue;
            if (j > 0) v = select(bin(Op::Eq, k, u32(j), vpir::kBool), u32(b), v);
            ++j;
        }
        return v;
    }
    ValueId instanceIndex() {
        ValueId i = b_->loadInput(uint32_t(instanceInput_), vpir::kU32);
        if (o_.viewCount > 1) i = bin(Op::Div, i, u32(o_.viewCount), vpir::kU32);
        return i;
    }
    ValueId inputValue(const IoLeaf& leaf) {
        Type want = vt(leaf.type);
        switch (leaf.kind) {
            case IoLeaf::Io: return cast(b_->loadInput(uint32_t(leaf.index), out_.inputs[leaf.index].type), want);
            case IoLeaf::Dropped: return zeroOf(want);
            case IoLeaf::ViewIndex: return cast(viewIndex(), want);
            case IoLeaf::Helper: return b_->constBool(false);
            case IoLeaf::InstanceIndex: return cast(instanceIndex(), want);
            case IoLeaf::VertexIndex: return cast(b_->loadInput(uint32_t(vertexInput_), vpir::kU32), want);
            case IoLeaf::SampleMaskIn: return cast(b_->loadInput(uint32_t(leaf.index), vpir::kU32), want);
            case IoLeaf::FrontFacing: return b_->loadInput(uint32_t(leaf.index), vpir::kBool);
            case IoLeaf::VertexAttr: return fetchAttribute(leaf.location, want);
        }
        fail("bad input leaf");
    }

    int addInput(vpir::IoVar v) {
        for (size_t i = 0; i < out_.inputs.size(); ++i)
            if (v.builtin != vpir::Builtin::None && out_.inputs[i].builtin == v.builtin) return int(i);
        out_.inputs.push_back(v);
        return int(out_.inputs.size() - 1);
    }
    int addOutput(vpir::IoVar v) {
        out_.outputs.push_back(v);
        outputShadow_.push_back(UINT32_MAX);
        return int(out_.outputs.size() - 1);
    }
    vpir::Builtin mapBuiltin(BuiltIn b, bool input, IoLeaf::Kind& special) {
        special = IoLeaf::Io;
        switch (b) {
            case BuiltIn::Position: return input ? vpir::Builtin::Position : vpir::Builtin::Position;
            case BuiltIn::FragCoord: return vpir::Builtin::Position;
            case BuiltIn::PointSize: return vpir::Builtin::PointSize;
            case BuiltIn::VertexIndex: special = IoLeaf::VertexIndex; return vpir::Builtin::VertexId;
            case BuiltIn::InstanceIndex: special = IoLeaf::InstanceIndex; return vpir::Builtin::InstanceId;
            case BuiltIn::FrontFacing: special = IoLeaf::FrontFacing; return vpir::Builtin::FrontFacing;
            case BuiltIn::SampleId: return vpir::Builtin::SampleId;
            case BuiltIn::SampleMask: if (input) special = IoLeaf::SampleMaskIn; return vpir::Builtin::SampleMask;
            case BuiltIn::FragDepth: return vpir::Builtin::FragDepth;
            case BuiltIn::PrimitiveId: return vpir::Builtin::PrimitiveId;
            case BuiltIn::Layer: return vpir::Builtin::Layer;
            case BuiltIn::ViewIndex: special = IoLeaf::ViewIndex; return vpir::Builtin::None;
            case BuiltIn::HelperInvocation: special = IoLeaf::Helper; return vpir::Builtin::None;
            case BuiltIn::GlobalInvocationId: return vpir::Builtin::GlobalInvocationId;
            case BuiltIn::LocalInvocationId: return vpir::Builtin::LocalInvocationId;
            case BuiltIn::WorkgroupId: return vpir::Builtin::WorkgroupId;
            case BuiltIn::NumWorkgroups: return vpir::Builtin::NumWorkgroups;
            case BuiltIn::LocalInvocationIndex: return vpir::Builtin::LocalInvocationIndex;
            case BuiltIn::SubgroupLocalInvocationId: return vpir::Builtin::SubgroupInvocation;
            case BuiltIn::SubgroupSize: return vpir::Builtin::SubgroupSize;
            // Clip and cull distances, and the rest: no VPIR builtin; writes are dropped.
            default: special = IoLeaf::Dropped; return vpir::Builtin::None;
        }
    }

    void declareIo() {
        for (uint32_t id : entry_->interface) {
            auto g = m_.globals.find(id);
            if (g == m_.globals.end()) continue;
            const GlobalVar& gv = g->second;
            if (gv.storage != Storage::Input && gv.storage != Storage::Output) continue;
            bool input = gv.storage == Storage::Input;
            IoVarInfo info;
            info.output = !input;
            uint32_t pointee = T(gv.type).elem;
            const Decorations* vd = m_.deco(id);
            uint32_t location = vd ? vd->get(Deco::Location, 0) : 0;
            if (vd && vd->has(Deco::Component) && vd->get(Deco::Component) != 0)
                fail("Component-packed interface variables are not supported (" + m_.nameOf(id) + ")");
            std::function<void(uint32_t, const Decorations*, uint32_t, uint32_t)> walk =
                [&](uint32_t typeId, const Decorations* d, uint32_t parentStruct, uint32_t member) {
                    const TypeInfo& t = T(typeId);
                    if (t.kind == TypeKind::Struct) {
                        for (uint32_t i = 0; i < t.members.size(); ++i) {
                            const Decorations* md = m_.memberDeco(typeId, i);
                            if (md && md->has(Deco::Location)) location = md->get(Deco::Location);
                            walk(t.members[i], md ? md : d, typeId, i);
                        }
                        return;
                    }
                    bool builtin = d && d->has(Deco::BuiltIn);
                    if (t.kind == TypeKind::Array || t.kind == TypeKind::Matrix) {
                        uint32_t n = t.kind == TypeKind::Array ? arrayLength(t) : t.count;
                        if (builtin) {  // ClipDistance[] / SampleMask[] / TessLevel…
                            BuiltIn bi = BuiltIn(d->get(Deco::BuiltIn));
                            for (uint32_t i = 0; i < n; ++i) {
                                IoLeaf leaf;
                                leaf.type = t.elem;
                                if (bi == BuiltIn::ClipDistance && !input && stage_ == vpir::Stage::Vertex && i < 8) {
                                    // [[clip_distance]] array element i (Metal has no cull distances).
                                    vpir::IoVar v;
                                    v.name = "clip_distance" + std::to_string(i);
                                    v.type = vpir::kF32;
                                    v.builtin = vpir::Builtin::ClipDistance;
                                    v.location = i;
                                    leaf.kind = IoLeaf::Io;
                                    leaf.index = addOutput(v);
                                } else if (bi == BuiltIn::SampleMask && i == 0) {
                                    vpir::IoVar v;
                                    v.name = input ? "sample_mask_in" : "sample_mask";
                                    v.type = vpir::kU32;
                                    v.builtin = vpir::Builtin::SampleMask;
                                    if (input) leaf.kind = IoLeaf::SampleMaskIn, leaf.index = addInput(v);
                                    else leaf.index = addOutput(v);
                                } else leaf.kind = IoLeaf::Dropped;
                                std::vector<uint32_t> sub;
                                leaves(t.elem, sub);
                                for (size_t k = 0; k < sub.size(); ++k) info.leaves.push_back(leaf);
                            }
                            return;
                        }
                        for (uint32_t i = 0; i < n; ++i) walk(t.elem, d, parentStruct, member);
                        return;
                    }
                    IoLeaf leaf;
                    leaf.type = typeId;
                    if (builtin) {
                        BuiltIn bi = BuiltIn(d->get(Deco::BuiltIn));
                        IoLeaf::Kind special;
                        vpir::Builtin vb = mapBuiltin(bi, input, special);
                        leaf.kind = special;
                        if (vb != vpir::Builtin::None && special != IoLeaf::Dropped) {
                            vpir::IoVar v;
                            v.builtin = vb;
                            v.name = std::string(vpir::toString(vb));
                            v.type = vpir::builtinInfo(vb, input).type;
                            if (input) {
                                leaf.index = addInput(v);
                                if (vb == vpir::Builtin::InstanceId) instanceInput_ = leaf.index;
                                if (vb == vpir::Builtin::VertexId) vertexInput_ = leaf.index;
                            } else {
                                leaf.index = addOutput(v);
                                if (vb == vpir::Builtin::Layer) writesLayer_ = true, layerOutput_ = leaf.index;
                            }
                            if (special == IoLeaf::Io || special == IoLeaf::FrontFacing || special == IoLeaf::SampleMaskIn) {
                                if (special != IoLeaf::Io) {}  // index set above
                            }
                        }
                        if (special == IoLeaf::InstanceIndex) leaf.kind = IoLeaf::InstanceIndex;
                        if (special == IoLeaf::VertexIndex) leaf.kind = IoLeaf::VertexIndex;
                        info.leaves.push_back(leaf);
                        return;
                    }
                    if (input && stage_ == vpir::Stage::Vertex) {
                        leaf.kind = IoLeaf::VertexAttr;
                        leaf.location = location++;
                        info.leaves.push_back(leaf);
                        return;
                    }
                    vpir::IoVar v;
                    v.type = vt(typeId);
                    if (v.type.isBool()) fail("bool interface variables are not supported");
                    v.location = location++;
                    v.name = (input ? "in" : "out") + std::to_string(v.location);
                    if (d && d->has(Deco::Flat)) v.interp = vpir::Interp::Flat;
                    else if (d && d->has(Deco::NoPerspective)) v.interp = vpir::Interp::Linear;
                    if (d && d->has(Deco::Centroid)) v.sampling = vpir::Sampling::Centroid;
                    if (d && d->has(Deco::Sample)) v.sampling = vpir::Sampling::Sample;
                    if (v.type.isInt()) v.interp = vpir::Interp::Flat;
                    if (stage_ == vpir::Stage::Fragment && !input) {
                        if (v.location > 7) fail("fragment output location > 7");
                        iface_.colorOutputs |= uint8_t(1u << v.location);
                        v.name = "color" + std::to_string(v.location);
                        // Dual-source blending: Index 1 is the second source of location 0.
                        if (d && d->has(Deco::Index) && d->get(Deco::Index) == 1) v.index = 1, v.name += "_1";
                    }
                    if (input) leaf.index = addInput(v);
                    else leaf.index = addOutput(v);
                    info.leaves.push_back(leaf);
                };
            walk(pointee, vd, 0, 0);
            io_[id] = std::move(info);
        }
        // What multiview and the input-attachment pattern need.
        if (stage_ == vpir::Stage::Vertex && (o_.viewCount > 1 || hasInstanceAttributes())) {
            if (instanceInput_ < 0) {
                vpir::IoVar v;
                v.builtin = vpir::Builtin::InstanceId, v.name = "instance_id", v.type = vpir::kU32;
                instanceInput_ = addInput(v);
            }
        }
        if (stage_ == vpir::Stage::Vertex && vertexInput_ < 0 && hasVertexAttributes()) {
            vpir::IoVar v;
            v.builtin = vpir::Builtin::VertexId, v.name = "vertex_id", v.type = vpir::kU32;
            vertexInput_ = addInput(v);
        }
        if (stage_ == vpir::Stage::Vertex && multiview() && !writesLayer_) {
            vpir::IoVar v;
            v.builtin = vpir::Builtin::Layer, v.name = "layer", v.type = vpir::kU32;
            layerOutput_ = addOutput(v);
        }
        for (size_t i = 0; i < out_.outputs.size(); ++i) {
            uint32_t var = newVar(out_.outputs[i].type);
            outputShadow_[i] = var;
            b_->storeVar(var, zeroOf(out_.outputs[i].type));
        }
    }
    bool hasVertexAttributes() const { return !o_.vertexAttributes.empty(); }
    bool hasInstanceAttributes() const {
        for (auto& vb : o_.vertexBuffers)
            if (vb.perInstance) return true;
        return false;
    }
    ValueId positionInput() {
        if (positionInput_ < 0) {
            for (size_t i = 0; i < out_.inputs.size(); ++i)
                if (out_.inputs[i].builtin == vpir::Builtin::Position) positionInput_ = int(i);
            if (positionInput_ < 0) {
                vpir::IoVar v;
                v.builtin = vpir::Builtin::Position, v.name = "position", v.type = vpir::kF32.withWidth(4);
                positionInput_ = addInput(v);
            }
        }
        return b_->loadInput(uint32_t(positionInput_), vpir::kF32.withWidth(4));
    }

    // Writes the shadowed outputs (at every return of main).
    void flushOutputs() {
        for (size_t i = 0; i < out_.outputs.size(); ++i) {
            ValueId v = loadVar(outputShadow_[i]);
            const vpir::IoVar& o = out_.outputs[i];
            if (o.builtin == vpir::Builtin::Position && stage_ == vpir::Stage::Vertex) {
                ValueId y = un(Op::Neg, ext(v, 1), vpir::kF32);
                Inst in;
                in.op = Op::Insert, in.type = o.type, in.args = {v, y}, in.lits = {1};
                v = b_->emit(in);
            }
            if (int(i) == layerOutput_ && !writesLayer_) v = viewIndex();
            b_->storeOutput(uint32_t(i), v);
        }
    }

    // ---- vertex fetch ---------------------------------------------------------------------------
    ValueId fetchAttribute(uint32_t location, Type want) {
        const VertexAttribute* a = nullptr;
        for (auto& x : o_.vertexAttributes)
            if (x.location == location) a = &x;
        auto defaults = [&](std::vector<ValueId> comps, Scalar sc) {
            // Missing components read as (0, 0, 0, 1).
            Type et{want.scalar};
            for (size_t i = 0; i < comps.size(); ++i) comps[i] = cast(comps[i], et);
            while (comps.size() < want.width)
                comps.push_back(comps.size() == 3 ? (et.isFloat() ? f32(1.f) : constant(et, {1})) : zeroOf(et));
            comps.resize(want.width);
            (void)sc;
            return vecOf(comps, want);
        };
        if (!a) return zeroOf(want);
        const VertexBufferBinding* vb = nullptr;
        for (auto& x : o_.vertexBuffers)
            if (x.binding == a->binding) vb = &x;
        if (!vb) fail("vertex attribute " + std::to_string(location) + " has no vertex buffer");
        int res = resource(vpir::ResourceKind::Buffer, vb->slot, [](vpir::Resource& r) { r.access = vpir::Access::Read; });
        ValueId index = vb->perInstance ? instanceIndex() : b_->loadInput(uint32_t(vertexInput_), vpir::kU32);
        ValueId base = bin(Op::Mul, index, u32(vb->stride), vpir::kU32);
        RPtr p;
        p.target = uint32_t(res);
        p.dyn = base;
        // Buffer reads are 4-byte aligned: an 8/16-bit attribute at an odd offset is read from its
        // word and shifted (strides are multiples of 4 in practice).
        p.off = a->offset & ~3u;
        std::vector<ValueId> comps = unpackFormat(p, a->format, a->components, (a->offset & 3u) * 8);
        return defaults(comps, want.scalar);
    }

    // Reads one texel of `fmt` at p (vertex attributes and texel buffers). Components as f32/i32/u32.
    std::vector<ValueId> unpackFormat(const RPtr& p, vpgpu::Format fmt, uint8_t limit = 0, uint32_t bitShift = 0) {
        using F = vpgpu::Format;
        auto word = [&](uint32_t byteOff) {
            RPtr q = p;
            q.off += byteOff;
            return b_->load(q.target, address(q), vpir::kU32);
        };
        auto loadN = [&](Scalar s, uint32_t n) {
            if (limit && limit < n) n = limit;
            std::vector<ValueId> c;
            ValueId v = b_->load(p.target, address(p), Type(s, uint8_t(n)));
            for (uint32_t i = 0; i < n; ++i) c.push_back(ext(v, i));
            return c;
        };
        auto bits = [&](ValueId w, uint32_t off, uint32_t count, bool sign) {
            Type t = sign ? vpir::kI32 : vpir::kU32;
            ValueId x = sign ? bitcast(w, vpir::kI32) : w;
            return b_->ternary(Op::BitfieldExtract, x, u32(off), u32(count), t);
        };
        auto norm = [&](ValueId x, uint32_t count, bool sign) {
            float scale = sign ? float((1u << (count - 1)) - 1) : float((1ull << count) - 1);
            ValueId f = un(Op::Convert, x, vpir::kF32);
            f = bin(Op::Mul, f, f32(1.f / scale), vpir::kF32);
            if (sign) f = bin(Op::Max, f, f32(-1.f), vpir::kF32);
            return f;
        };
        auto packed = [&](uint32_t n, uint32_t width, bool sign, bool normalized) {
            // n components of `width` bits from consecutive words.
            if (limit && limit < n) n = limit;
            std::vector<ValueId> c;
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t bit = bitShift + i * width;
                ValueId w = word((bit / 32) * 4);
                ValueId x = bits(w, bit % 32, width, sign);
                c.push_back(normalized ? norm(x, width, sign) : x);
            }
            return c;
        };
        auto halves = [&](uint32_t n) {
            if (limit && limit < n) n = limit;
            std::vector<ValueId> c;
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t bit = bitShift + i * 16;
                ValueId w = word((bit / 32) * 4);
                c.push_back(halfToFloat(bits(w, bit % 32, 16, false)));
            }
            return c;
        };
        switch (fmt) {
            case F::R32Float: return loadN(Scalar::F32, 1);
            case F::RG32Float: return loadN(Scalar::F32, 2);
            case F::RGBA32Float: return loadN(Scalar::F32, 4);
            case F::R32Uint: return loadN(Scalar::U32, 1);
            case F::RG32Uint: return loadN(Scalar::U32, 2);
            case F::RGBA32Uint: return loadN(Scalar::U32, 4);
            case F::R32Sint: return loadN(Scalar::I32, 1);
            case F::RG32Sint: return loadN(Scalar::I32, 2);
            case F::RGBA32Sint: return loadN(Scalar::I32, 4);
            case F::R8Unorm: return packed(1, 8, false, true);
            case F::RG8Unorm: return packed(2, 8, false, true);
            case F::RGBA8Unorm: case F::RGBA8Srgb: return packed(4, 8, false, true);
            case F::BGRA8Unorm: case F::BGRA8Srgb: {
                auto c = packed(4, 8, false, true);
                std::swap(c[0], c[2]);
                return c;
            }
            case F::R8Snorm: return packed(1, 8, true, true);
            case F::RG8Snorm: return packed(2, 8, true, true);
            case F::RGBA8Snorm: return packed(4, 8, true, true);
            case F::R8Uint: return packed(1, 8, false, false);
            case F::RG8Uint: return packed(2, 8, false, false);
            case F::RGBA8Uint: return packed(4, 8, false, false);
            case F::R8Sint: return packed(1, 8, true, false);
            case F::RG8Sint: return packed(2, 8, true, false);
            case F::RGBA8Sint: return packed(4, 8, true, false);
            case F::R16Unorm: return packed(1, 16, false, true);
            case F::RG16Unorm: return packed(2, 16, false, true);
            case F::RGBA16Unorm: return packed(4, 16, false, true);
            case F::R16Snorm: return packed(1, 16, true, true);
            case F::RG16Snorm: return packed(2, 16, true, true);
            case F::RGBA16Snorm: return packed(4, 16, true, true);
            case F::R16Uint: return packed(1, 16, false, false);
            case F::RG16Uint: return packed(2, 16, false, false);
            case F::RGBA16Uint: return packed(4, 16, false, false);
            case F::R16Sint: return packed(1, 16, true, false);
            case F::RG16Sint: return packed(2, 16, true, false);
            case F::RGBA16Sint: return packed(4, 16, true, false);
            case F::R16Float: return halves(1);
            case F::RG16Float: return halves(2);
            case F::RGBA16Float: return halves(4);
            case F::RGB10A2Unorm: {
                ValueId w = word(0);
                return {norm(bits(w, 0, 10, false), 10, false), norm(bits(w, 10, 10, false), 10, false),
                        norm(bits(w, 20, 10, false), 10, false), norm(bits(w, 30, 2, false), 2, false)};
            }
            case F::RGB10A2Uint: {
                ValueId w = word(0);
                return {bits(w, 0, 10, false), bits(w, 10, 10, false), bits(w, 20, 10, false), bits(w, 30, 2, false)};
            }
            default: fail(std::string("unsupported vertex/texel format ") + vpgpu::formatInfo(fmt).name);
        }
    }

    // IEEE half bits (u32, low 16) -> f32, exactly.
    ValueId halfToFloat(ValueId h) {
        ValueId sign = bin(Op::Shl, bin(Op::And, h, u32(0x8000), vpir::kU32), u32(16), vpir::kU32);
        ValueId expo = b_->ternary(Op::BitfieldExtract, h, u32(10), u32(5), vpir::kU32);
        ValueId mant = bin(Op::And, h, u32(0x3FF), vpir::kU32);
        // normal: (exp + 112) << 23 | mant << 13
        ValueId normal = bin(Op::Or, bin(Op::Shl, bin(Op::Add, expo, u32(112), vpir::kU32), u32(23), vpir::kU32),
                             bin(Op::Shl, mant, u32(13), vpir::kU32), vpir::kU32);
        ValueId inf = bin(Op::Or, u32(0x7F800000), bin(Op::Shl, mant, u32(13), vpir::kU32), vpir::kU32);
        // subnormal: mant * 2^-24
        ValueId subf = bin(Op::Mul, un(Op::Convert, mant, vpir::kF32), f32(5.9604644775390625e-8f), vpir::kF32);
        ValueId sub = bitcast(subf, vpir::kU32);
        ValueId isZeroExp = bin(Op::Eq, expo, u32(0), vpir::kBool);
        ValueId isMaxExp = bin(Op::Eq, expo, u32(31), vpir::kBool);
        ValueId mag = select(isZeroExp, sub, select(isMaxExp, inf, normal));
        return bitcast(bin(Op::Or, mag, sign, vpir::kU32), vpir::kF32);
    }
    // f32 -> half bits (u32), round to nearest even.
    ValueId floatToHalf(ValueId f) {
        // Through the native conversion: f32 -> f16 -> bits (as the low half of a u32).
        ValueId h = un(Op::Convert, f, vpir::kF16);
        // f16 has no 32-bit bitcast; pair it with zero and reinterpret the f16x2 as u32.
        ValueId pair = b_->construct(std::vector<ValueId>{h, splat(vpir::kF16, 0)}, vpir::kF16.withWidth(2));
        return un(Op::Bitcast, pair, vpir::kU32);
    }

    // ---- control flow ---------------------------------------------------------------------------
    const Block& blockOf(uint32_t label) const {
        auto it = scope_->fn->blockOf.find(label);
        if (it == scope_->fn->blockOf.end()) fail("unknown block %" + std::to_string(label));
        return scope_->fn->blocks[it->second];
    }

    void storePhis(uint32_t target, uint32_t from) {
        const Block& bl = blockOf(target);
        std::vector<std::pair<uint32_t, Val>> vals;
        for (const Instr& in : bl.body) {
            if (in.op != OpPhi) break;
            uint32_t id = m_.w(in, 2);
            for (uint32_t i = 3; i + 1 < in.count; i += 2)
                if (m_.w(in, i + 1) == from) vals.push_back({id, get(m_.w(in, i))});
        }
        for (auto& [id, v] : vals) {
            auto& vars = scope_->phiVars[id];
            for (size_t i = 0; i < v.v.size() && i < vars.size(); ++i) b_->storeVar(vars[i], v.v[i]);
        }
    }

    enum class Jump : uint8_t { Next, Done };

    // A branch from `from` to `to`; `stop` ends the current region. Next: emit `to` here.
    Jump jump(uint32_t from, uint32_t to, uint32_t stop) {
        if (from) storePhis(to, from);
        if (to == stop) return Jump::Done;
        for (size_t i = ctx_.size(); i-- > 0;) {
            Ctx& c = ctx_[i];
            if (c.scope != scope_) break;  // targets never leave the current function
            if (c.kind == Ctx::Loop) {
                if (to == c.header) return exitTo(i, Action::Header), Jump::Done;
                if (to == c.cont) return exitTo(i, Action::Continue), Jump::Done;
                if (to == c.merge) return exitTo(i, Action::Break), Jump::Done;
            } else if (c.kind == Ctx::Switch) {
                if (to == c.merge) return exitTo(i, Action::Break), Jump::Done;
            }
        }
        return Jump::Next;
    }

    void exitTo(size_t target, Action a) {
        if (target + 1 == ctx_.size()) return perform(target, a);
        uint32_t code = nextCode_++;
        for (size_t i = target + 1; i < ctx_.size(); ++i) ctx_[i].pending.push_back({code, target, a});
        b_->storeVar(exitVar(), u32(code));
        b_->breakLoop();
    }
    void perform(size_t target, Action a) {
        Ctx& c = ctx_[target];
        switch (a) {
            case Action::Break: b_->breakLoop(); return;
            case Action::Header: b_->continueLoop(); return;
            case Action::Continue:
                if (c.cont == c.header) {
                    b_->continueLoop();
                    return;
                }
                // The continue construct, in place; it ends by branching to the header.
                flow(c.cont, 0);
                return;
        }
    }
    // After the VPIR loop of ctx_.back() closed: route the exits that were passing through it.
    void popCtx() {
        Ctx c = std::move(ctx_.back());
        ctx_.pop_back();
        for (const Ctx::Pending& p : c.pending) {
            b_->beginIf(bin(Op::Eq, loadVar(exitVar()), u32(p.code), vpir::kBool));
            if (p.target + 1 == ctx_.size()) {
                b_->storeVar(exitVar(), u32(0));
                perform(p.target, p.action);
            } else {
                b_->breakLoop();
            }
            b_->end();
        }
    }

    bool activeLoop(uint32_t header) const {
        for (const Ctx& c : ctx_)
            if (c.kind == Ctx::Loop && c.header == header && c.scope == scope_) return true;
        return false;
    }

    // Emits blocks from `label` until the region ends (stop reached, or every path left it).
    void flow(uint32_t label, uint32_t stop) {
        uint32_t cur = label;
        bool enteringHeader = false;
        while (cur) {
            const Block& bl = blockOf(cur);
            if (bl.merge.op == OpLoopMerge && !activeLoop(cur) && !enteringHeader) {
                Ctx c;
                c.kind = Ctx::Loop, c.header = cur, c.merge = m_.w(bl.merge, 1), c.cont = m_.w(bl.merge, 2), c.scope = scope_;
                ctx_.push_back(c);
                b_->beginLoop();
                flowHeader(cur);
                b_->end();
                popCtx();
                uint32_t merge = c.merge;
                cur = proceed(merge, stop);
                continue;
            }
            enteringHeader = false;
            cur = emitBlock(bl, stop);
        }
    }
    void flowHeader(uint32_t header) {
        const Block& bl = blockOf(header);
        uint32_t next = emitBlock(bl, 0);
        if (next) flowNoHeader(next);
    }
    void flowNoHeader(uint32_t label) { flow(label, 0); }

    // After a construct: where emission continues (0 = the region is over).
    uint32_t proceed(uint32_t merge, uint32_t stop) {
        if (!merge) return 0;
        return jump(0, merge, stop) == Jump::Next ? merge : 0;
    }

    // Emits one block; returns the next block to emit in this region, or 0.
    uint32_t emitBlock(const Block& bl, uint32_t stop) {
        scope_->block = bl.label;
        for (const Instr& in : bl.body) instruction(in);
        const Instr& t = bl.terminator;
        cur_ = &t;
        switch (t.op) {
            case OpBranch: {
                uint32_t to = m_.w(t, 1);
                return jump(bl.label, to, stop) == Jump::Next ? to : 0;
            }
            case OpBranchConditional: {
                ValueId c = get1(m_.w(t, 1));
                uint32_t tt = m_.w(t, 2), ff = m_.w(t, 3);
                uint32_t merge = bl.merge.op == OpSelectionMerge ? m_.w(bl.merge, 1) : 0;
                uint32_t armStop = merge ? merge : stop;
                if (tt == ff) {
                    return jump(bl.label, tt, stop) == Jump::Next ? tt : 0;
                }
                b_->beginIf(c);
                arm(bl.label, tt, armStop);
                b_->beginElse();
                arm(bl.label, ff, armStop);
                b_->end();
                scope_->block = bl.label;
                if (!merge) return 0;
                return proceed(merge, stop);
            }
            case OpSwitch: return emitSwitch(bl, stop);
            case OpReturn:
            case OpReturnValue: {
                if (t.op == OpReturnValue) {
                    Val v = get(m_.w(t, 1));
                    for (size_t i = 0; i < v.v.size() && i < scope_->retVars.size(); ++i) b_->storeVar(scope_->retVars[i], v.v[i]);
                }
                if (isEntryScope()) {
                    flushOutputs();
                    b_->ret();
                } else {
                    // Out of the callee's once-loop.
                    size_t f = ctx_.size();
                    while (f-- > 0)
                        if (ctx_[f].kind == Ctx::Func && ctx_[f].scope == scope_) break;
                    exitTo(f, Action::Break);
                }
                return 0;
            }
            case OpKill:
            case OpTerminateInvocation:
                b_->discard();
                return 0;
            case OpUnreachable: return 0;
            default: fail("block without a known terminator");
        }
    }
    bool isEntryScope() const { return scope_ == scopes_.front().get(); }

    void arm(uint32_t from, uint32_t to, uint32_t stop) {
        if (jump(from, to, stop) == Jump::Next) flow(to, stop);
        scope_->block = from;
    }

    uint32_t emitSwitch(const Block& bl, uint32_t stop) {
        const Instr& t = bl.terminator;
        ValueId sel = get1(m_.w(t, 1));
        Type st = typeOfValue(sel);
        uint32_t def = m_.w(t, 2);
        uint32_t merge = bl.merge.op == OpSelectionMerge ? m_.w(bl.merge, 1) : 0;
        if (!merge) fail("OpSwitch without OpSelectionMerge");
        // Cases grouped by target, in first-appearance order.
        std::vector<std::pair<uint32_t, std::vector<uint32_t>>> cases;
        for (uint32_t i = 3; i + 1 < t.count; i += 2) {
            uint32_t lit = m_.w(t, i), target = m_.w(t, i + 1);
            auto it = std::find_if(cases.begin(), cases.end(), [&](auto& c) { return c.first == target; });
            if (it == cases.end()) cases.push_back({target, {lit}});
            else it->second.push_back(lit);
        }
        Ctx c;
        c.kind = Ctx::Switch, c.merge = merge, c.scope = scope_;
        ctx_.push_back(c);
        b_->beginLoop();
        size_t opened = 0;
        for (auto& [target, lits] : cases) {
            if (target == def) continue;  // handled by the default arm
            ValueId cond = kNoValue;
            for (uint32_t lit : lits) {
                ValueId e = bin(Op::Eq, sel, constant(st, {lit}), vpir::kBool);
                cond = cond == kNoValue ? e : bin(Op::Or, cond, e, vpir::kBool);
            }
            b_->beginIf(cond);
            arm(bl.label, target, merge);
            b_->beginElse();
            ++opened;
        }
        arm(bl.label, def, merge);
        for (size_t i = 0; i < opened; ++i) b_->end();
        b_->breakLoop();
        b_->end();
        popCtx();
        scope_->block = bl.label;
        return proceed(merge, stop);
    }

    // ---- function calls -------------------------------------------------------------------------
    Scope* newScope(const FunctionDef& f) {
        scopes_.push_back(std::make_unique<Scope>());
        Scope* s = scopes_.back().get();
        s->fn = &f;
        s->returnType = f.resultType;
        analyse(*s);
        for (const Block& bl : f.blocks)
            for (const Instr& in : bl.body)
                if (in.op == OpPhi) {
                    uint32_t id = m_.w(in, 2);
                    std::vector<uint32_t> lt;
                    leaves(m_.w(in, 1), lt);
                    for (uint32_t x : lt) s->phiVars[id].push_back(newVar(vt(x)));
                }
        return s;
    }

    Val call(const Instr& in) {
        uint32_t resultType = m_.w(in, 1);
        uint32_t fid = m_.w(in, 3);
        auto fit = m_.functions.find(fid);
        if (fit == m_.functions.end()) fail("call to an unknown function");
        const FunctionDef& f = fit->second;
        if (++callDepth_ > 64) fail("call depth exceeds 64 (recursion?)");
        Scope* caller = scope_;
        Scope* s = newScope(f);
        // Arguments, evaluated in the caller.
        for (size_t i = 0; i < f.params.size(); ++i) {
            uint32_t arg = m_.w(in, 4 + uint32_t(i));
            uint32_t pt = f.paramTypes[i];
            const TypeInfo& ptt = T(pt);
            if (ptt.kind == TypeKind::Pointer) {
                if (ptt.storage == Storage::UniformConstant) {
                    // A handle passed by pointer.
                    s->params[f.params[i]] = resolve(arg, false);
                } else {
                    s->params[f.params[i]] = resolve(arg, false);
                }
            } else if (ptt.kind == TypeKind::Image || ptt.kind == TypeKind::Sampler || ptt.kind == TypeKind::SampledImage) {
                s->handles[f.params[i]] = handle(arg);
            } else {
                Val v = get(arg);
                s->vals[f.params[i]] = v;
                s->valBlock[f.params[i]] = 0;
                // Parameters may be used in any block: keep them in variables when spilled.
                if (s->spilled.count(f.params[i])) {
                    Scope* keep = scope_;
                    scope_ = s;
                    s->block = 0;
                    define(f.params[i], v);
                    scope_ = keep;
                }
            }
        }
        std::vector<uint32_t> lt;
        if (T(resultType).kind != TypeKind::Void) {
            leaves(resultType, lt);
            for (uint32_t x : lt) s->retVars.push_back(newVar(vt(x)));
        }
        Ctx c;
        c.kind = Ctx::Func, c.scope = s;
        ctx_.push_back(c);
        b_->beginLoop();
        scope_ = s;
        flow(f.blocks.front().label, 0);
        scope_ = caller;
        b_->breakLoop();
        b_->end();
        popCtx();
        --callDepth_;
        Val r;
        for (uint32_t var : s->retVars) r.v.push_back(loadVar(var));
        return r;
    }
    int callDepth_ = 0;

    // ---- handles ---------------------------------------------------------------------------------
    Handle handle(uint32_t id) {
        auto h = scope_->handles.find(id);
        if (h != scope_->handles.end()) return h->second;
        // OpImageTexelPointer names the image variable itself.
        auto t = m_.resultType.find(id);
        if (t != m_.resultType.end() && T(t->second).kind == TypeKind::Pointer) return loadHandle(resolve(id, false));
        fail("image or sampler " + m_.nameOf(id) + " is not a known handle");
    }
    Handle loadHandle(const RPtr& p) {
        if (p.kind != RPtr::Handles) fail("handle load from a non-handle pointer");
        return handleFor(p.global, p.element);
    }

    // ---- instructions ---------------------------------------------------------------------------
    uint32_t w(const Instr& in, uint32_t i) const { return m_.w(in, i); }

    Type resultVt(const Instr& in) { return vt(w(in, 1)); }

    // Operands converted to the result's scalar kind (SPIR-V integer ops ignore signedness).
    ValueId opAs(uint32_t id, Type t) { return castBits(get1(id), t); }
    ValueId castBits(ValueId v, Type t) {
        Type from = typeOfValue(v);
        if (from == t) return v;
        if (from.isInt() && t.isInt() && from.width == t.width) return un(Op::Bitcast, v, t);
        if (from.width == t.width) return cast(v, t);
        fail("internal: operand shape mismatch");
    }
    ValueId signedOp(Op op, const Instr& in, bool sign) {
        Type rt = resultVt(in);
        Type it = Type(sign ? Scalar::I32 : Scalar::U32, rt.width);
        ValueId a = castBits(get1(w(in, 3)), it), c = castBits(get1(w(in, 4)), it);
        return castBits(bin(op, a, c, it), rt);
    }
    ValueId compare(Op op, const Instr& in, Scalar s) {
        ValueId a = get1(w(in, 3)), c = get1(w(in, 4));
        Type at = typeOfValue(a);
        if (s != Scalar::Void) {
            Type it(s, at.width);
            a = castBits(a, it), c = castBits(c, it);
        }
        return bin(op, a, c, resultVt(in));
    }

    void instruction(const Instr& in) {
        cur_ = &in;
        const uint16_t op = in.op;
        switch (op) {
            case OpNop: case OpLine: case OpNoLine: case OpSelectionMerge: case OpLoopMerge:
                return;
            case OpImageWrite: imageWrite(in); return;
            case OpAtomicStore: {
                if (auto tp = scope_->texelPointers.find(w(in, 1)); tp != scope_->texelPointers.end()) {
                    ValueId v = get1(w(in, 4));
                    Type t(typeOfValue(v).scalar);
                    texelAtomic(tp->second, Op::AtomicExchange, {v}, t, t);
                    return;
                }
                RPtr p = resolve(w(in, 1), true);
                storeScalarVec(p, get1(w(in, 4)));
                return;
            }
            case OpPhi: {
                uint32_t id = w(in, 2);
                Val r;
                for (uint32_t var : scope_->phiVars[id]) r.v.push_back(loadVar(var));
                scope_->vals[id] = r;
                scope_->valBlock[id] = scope_->block;
                return;
            }
            case OpUndef: define(w(in, 2), zero(w(in, 1))); return;
            case OpVariable: {
                uint32_t id = w(in, 2);
                const TypeInfo& pt = T(w(in, 1));
                if (pt.storage != Storage::Function) fail("non-function variable inside a function");
                uint32_t slot;
                auto it = scope_->locals.find(id);
                if (it == scope_->locals.end()) {
                    slot = newPrivateSlot(naturalSize(pt.elem));
                    scope_->locals[id] = slot;
                } else slot = it->second;
                if (in.count > 4) {
                    RPtr p;
                    p.target = slot, p.off = 0, p.type = pt.elem;
                    size_t at = 0;
                    Val v = get(w(in, 4));
                    storeMem(p, v, at);
                }
                return;
            }
            case OpAccessChain:
            case OpInBoundsAccessChain:
            case OpPtrAccessChain:
            case OpInBoundsPtrAccessChain:
                scope_->chains[w(in, 2)] = in;
                return;
            case OpLoad: {
                uint32_t type = w(in, 1), id = w(in, 2);
                const TypeInfo& tt = T(type);
                RPtr p = resolve(w(in, 3), false);
                if (tt.kind == TypeKind::Image || tt.kind == TypeKind::Sampler || tt.kind == TypeKind::SampledImage) {
                    scope_->handles[id] = loadHandle(p);
                    return;
                }
                if (p.kind == RPtr::Input || p.kind == RPtr::Output) return define(id, loadIo(p));
                if (p.kind != RPtr::Mem) fail("load through an unsupported pointer");
                define(id, loadMem(p));
                return;
            }
            case OpStore: {
                RPtr p = resolve(w(in, 1), true);
                Val v = get(w(in, 2));
                if (p.kind == RPtr::Output || p.kind == RPtr::Input) return storeIo(p, v);
                size_t at = 0;
                storeMem(p, v, at);
                return;
            }
            case OpCopyMemory: {
                RPtr dst = resolve(w(in, 1), true), src = resolve(w(in, 2), false);
                Val v = src.kind == RPtr::Mem ? loadMem(src) : loadIo(src);
                if (dst.kind == RPtr::Output) return storeIo(dst, v);
                size_t at = 0;
                storeMem(dst, v, at);
                return;
            }
            case OpCopyObject:
            case OpCopyLogical: {
                uint32_t src = w(in, 3);
                if (auto h = scope_->handles.find(src); h != scope_->handles.end()) {
                    scope_->handles[w(in, 2)] = h->second;
                    return;
                }
                if (scope_->chains.count(src) || m_.globals.count(src) || scope_->locals.count(src)) {
                    // Pointer copy: alias the chain.
                    Instr alias = in;
                    scope_->chains[w(in, 2)] = scope_->chains.count(src) ? scope_->chains[src] : alias;
                    if (!scope_->chains.count(src)) fail("pointer copy of a root variable is not supported");
                    return;
                }
                return define(w(in, 2), get(src));
            }
            case OpFunctionCall: {
                Val r = call(in);
                if (T(w(in, 1)).kind != TypeKind::Void) define(w(in, 2), r);
                return;
            }
            case OpSampledImage: {
                Handle img = handle(w(in, 3)), smp = handle(w(in, 4));
                img.sampler = smp.sampler;
                scope_->handles[w(in, 2)] = img;
                return;
            }
            case OpImage: {
                Handle h = handle(w(in, 3));
                h.sampler = -1;
                scope_->handles[w(in, 2)] = h;
                return;
            }
            case OpImageTexelPointer: {
                scope_->texelPointers[w(in, 2)] = in;
                return;
            }
            case OpControlBarrier: {
                uint32_t sem = uint32_t(constScalar(w(in, 3)));
                if (stage_ != vpir::Stage::Compute) return;  // fragment/vertex barriers: no-op
                Inst bi;
                bi.op = (sem & (MemSem::UniformMemory | MemSem::ImageMemory)) ? Op::BarrierAll : Op::Barrier;
                b_->emit(bi);
                return;
            }
            case OpMemoryBarrier: return;  // execution barriers carry the ordering VPIR expresses
            case OpDemoteToHelperInvocation: b_->discard(); return;
            case OpIsHelperInvocationEXT: define1(w(in, 2), b_->constBool(false)); return;
            case OpExtInst: return extInst(in);
            default: break;
        }
        if (in.count < 3) fail("unsupported instruction " + std::to_string(op));
        define(w(in, 2), value(in));
    }

    Val value(const Instr& in) {
        const uint16_t op = in.op;
        const uint32_t rtId = w(in, 1);
        auto one = [](ValueId v) { return Val{{v}}; };
        switch (op) {
            // ---- composites
            case OpCompositeConstruct: return compositeConstruct(in);
            case OpCompositeExtract: return compositeExtract(get(w(in, 3)), typeOf(w(in, 3)), in, 4);
            case OpCompositeInsert: return compositeInsert(in);
            case OpVectorShuffle: {
                ValueId a = get1(w(in, 3)), c = get1(w(in, 4));
                Type at = typeOfValue(a), ct = typeOfValue(c), rt = vt(rtId);
                std::vector<ValueId> comps;
                for (uint32_t i = 5; i < in.count; ++i) {
                    uint32_t k = w(in, i);
                    if (k == 0xFFFFFFFFu) comps.push_back(zeroOf(rt.element()));
                    else if (k < at.width) comps.push_back(ext(a, k));
                    else comps.push_back(ext(c, k - at.width));
                }
                (void)ct;
                return one(vecOf(comps, rt));
            }
            case OpVectorExtractDynamic: {
                ValueId v = get1(w(in, 3)), idx = cast(get1(w(in, 4)), vpir::kU32);
                Type t = typeOfValue(v);
                ValueId r = ext(v, 0);
                for (uint32_t i = 1; i < t.width; ++i) r = select(bin(Op::Eq, idx, u32(i), vpir::kBool), ext(v, i), r);
                return one(r);
            }
            case OpVectorInsertDynamic: {
                ValueId v = get1(w(in, 3)), s = get1(w(in, 4)), idx = cast(get1(w(in, 5)), vpir::kU32);
                Type t = typeOfValue(v);
                std::vector<ValueId> comps;
                for (uint32_t i = 0; i < t.width; ++i)
                    comps.push_back(select(bin(Op::Eq, idx, u32(i), vpir::kBool), s, ext(v, i)));
                return one(vecOf(comps, t));
            }
            case OpTranspose: {
                Val m = get(w(in, 3));
                const TypeInfo& rt = T(rtId);
                Val r;
                for (uint32_t c = 0; c < rt.count; ++c) {
                    std::vector<ValueId> col;
                    for (ValueId src : m.v) col.push_back(ext(src, c));
                    r.v.push_back(vecOf(col, vt(rt.elem)));
                }
                return r;
            }
            // ---- arithmetic
            case OpSNegate: {
                Type rt = vt(rtId), it(Scalar::I32, rt.width);
                return one(castBits(un(Op::Neg, castBits(get1(w(in, 3)), it), it), rt));
            }
            case OpFNegate: return one(un(Op::Neg, get1(w(in, 3)), vt(rtId)));
            case OpIAdd: return one(bin(Op::Add, opAs(w(in, 3), vt(rtId)), opAs(w(in, 4), vt(rtId)), vt(rtId)));
            case OpISub: return one(bin(Op::Sub, opAs(w(in, 3), vt(rtId)), opAs(w(in, 4), vt(rtId)), vt(rtId)));
            case OpIMul: return one(bin(Op::Mul, opAs(w(in, 3), vt(rtId)), opAs(w(in, 4), vt(rtId)), vt(rtId)));
            case OpFAdd: return one(bin(Op::Add, get1(w(in, 3)), get1(w(in, 4)), vt(rtId)));
            case OpFSub: return one(bin(Op::Sub, get1(w(in, 3)), get1(w(in, 4)), vt(rtId)));
            case OpFMul: return one(bin(Op::Mul, get1(w(in, 3)), get1(w(in, 4)), vt(rtId)));
            case OpFDiv: return one(bin(Op::Div, get1(w(in, 3)), get1(w(in, 4)), vt(rtId)));
            case OpFRem: return one(bin(Op::Rem, get1(w(in, 3)), get1(w(in, 4)), vt(rtId)));
            case OpFMod: {
                ValueId a = get1(w(in, 3)), c = get1(w(in, 4));
                Type t = vt(rtId);
                ValueId q = un(Op::Floor, bin(Op::Div, a, c, t), t);
                return one(bin(Op::Sub, a, bin(Op::Mul, c, q, t), t));
            }
            case OpUDiv: return one(signedOp(Op::Div, in, false));
            case OpSDiv: return one(signedOp(Op::Div, in, true));
            case OpUMod: return one(signedOp(Op::Rem, in, false));
            case OpSRem: return one(signedOp(Op::Rem, in, true));
            case OpSMod: {
                Type rt = vt(rtId), it(Scalar::I32, rt.width);
                ValueId a = castBits(get1(w(in, 3)), it), c = castBits(get1(w(in, 4)), it);
                ValueId r = bin(Op::Rem, a, c, it);
                ValueId z = splat(it, 0);
                Type bt(Scalar::Bool, rt.width);
                ValueId fix = bin(Op::And, bin(Op::Ne, r, z, bt), bin(Op::Ne, bin(Op::Lt, r, z, bt), bin(Op::Lt, c, z, bt), bt), bt);
                return one(castBits(select(fix, bin(Op::Add, r, c, it), r), rt));
            }
            case OpVectorTimesScalar: {
                ValueId v = get1(w(in, 3)), s = get1(w(in, 4));
                Type t = vt(rtId);
                return one(bin(Op::Mul, v, splatValue(s, t.width), t));
            }
            case OpMatrixTimesScalar: {
                Val m = get(w(in, 3));
                ValueId s = get1(w(in, 4));
                Val r;
                for (ValueId c : m.v) r.v.push_back(bin(Op::Mul, c, splatValue(s, typeOfValue(c).width), typeOfValue(c)));
                return r;
            }
            case OpMatrixTimesVector: {
                Val m = get(w(in, 3));
                ValueId v = get1(w(in, 4));
                Type t = vt(rtId);
                ValueId acc = kNoValue;
                for (size_t c = 0; c < m.v.size(); ++c) {
                    ValueId s = splatValue(ext(v, uint32_t(c)), t.width);
                    acc = acc == kNoValue ? bin(Op::Mul, m.v[c], s, t) : tri(Op::Fma, m.v[c], s, acc, t);
                }
                return one(acc);
            }
            case OpVectorTimesMatrix: {
                ValueId v = get1(w(in, 3));
                Val m = get(w(in, 4));
                Type t = vt(rtId);
                std::vector<ValueId> comps;
                for (ValueId c : m.v) comps.push_back(dot(v, c));
                return one(vecOf(comps, t));
            }
            case OpMatrixTimesMatrix: {
                Val a = get(w(in, 3)), c = get(w(in, 4));
                Type colT = typeOfValue(a.v[0]);
                Val r;
                for (ValueId col : c.v) {
                    ValueId acc = kNoValue;
                    for (size_t k = 0; k < a.v.size(); ++k) {
                        ValueId s = splatValue(ext(col, uint32_t(k)), colT.width);
                        acc = acc == kNoValue ? bin(Op::Mul, a.v[k], s, colT) : tri(Op::Fma, a.v[k], s, acc, colT);
                    }
                    r.v.push_back(acc);
                }
                return r;
            }
            case OpOuterProduct: {
                ValueId a = get1(w(in, 3)), c = get1(w(in, 4));
                Type at = typeOfValue(a);
                Val r;
                for (uint32_t j = 0; j < typeOfValue(c).width; ++j)
                    r.v.push_back(bin(Op::Mul, a, splatValue(ext(c, j), at.width), at));
                return r;
            }
            case OpDot: return one(dot(get1(w(in, 3)), get1(w(in, 4))));
            // ---- conversions
            case OpConvertFToU:
            case OpConvertFToS:
            case OpConvertSToF:
            case OpConvertUToF:
            case OpFConvert: {
                ValueId v = get1(w(in, 3));
                Type from = typeOfValue(v), to = vt(rtId);
                if (op == OpConvertSToF) v = castBits(v, Type(Scalar::I32, from.width));
                if (op == OpConvertUToF) v = castBits(v, Type(Scalar::U32, from.width));
                if (op == OpConvertFToS) {
                    ValueId r = un(Op::Convert, v, Type(Scalar::I32, to.width));
                    return one(castBits(r, to));
                }
                if (op == OpConvertFToU) {
                    ValueId r = un(Op::Convert, v, Type(Scalar::U32, to.width));
                    return one(castBits(r, to));
                }
                if (typeOfValue(v) == to) return one(v);
                return one(un(Op::Convert, v, to));
            }
            case OpUConvert:
            case OpSConvert: return one(castBits(get1(w(in, 3)), vt(rtId)));
            case OpBitcast: {
                ValueId v = get1(w(in, 3));
                Type to = vt(rtId);
                if (typeOfValue(v) == to) return one(v);
                return one(un(Op::Bitcast, v, to));
            }
            case OpQuantizeToF16: {
                Type t = vt(rtId), h(Scalar::F16, t.width);
                return one(un(Op::Convert, un(Op::Convert, get1(w(in, 3)), h), t));
            }
            // ---- logic and comparisons
            case OpLogicalAnd: return one(bin(Op::And, get1(w(in, 3)), get1(w(in, 4)), vt(rtId)));
            case OpLogicalOr: return one(bin(Op::Or, get1(w(in, 3)), get1(w(in, 4)), vt(rtId)));
            case OpLogicalNot: return one(un(Op::Not, get1(w(in, 3)), vt(rtId)));
            case OpLogicalEqual: return one(bin(Op::Eq, get1(w(in, 3)), get1(w(in, 4)), vt(rtId)));
            case OpLogicalNotEqual: return one(bin(Op::Ne, get1(w(in, 3)), get1(w(in, 4)), vt(rtId)));
            case OpSelect: {
                Val c = get(w(in, 3)), a = get(w(in, 4)), d = get(w(in, 5));
                Val r;
                for (size_t i = 0; i < a.v.size(); ++i) {
                    ValueId cond = c.v.size() == a.v.size() ? c.v[i] : c.v[0];
                    Type at = typeOfValue(a.v[i]);
                    Type ct = typeOfValue(cond);
                    if (ct.width > 1 && ct.width != at.width) cond = ext(cond, 0);
                    r.v.push_back(tri(Op::Select, cond, a.v[i], d.v[i], at));
                }
                return r;
            }
            case OpIEqual: return one(compare(Op::Eq, in, Scalar::U32));
            case OpINotEqual: return one(compare(Op::Ne, in, Scalar::U32));
            case OpUGreaterThan: return one(compare(Op::Gt, in, Scalar::U32));
            case OpUGreaterThanEqual: return one(compare(Op::Ge, in, Scalar::U32));
            case OpULessThan: return one(compare(Op::Lt, in, Scalar::U32));
            case OpULessThanEqual: return one(compare(Op::Le, in, Scalar::U32));
            case OpSGreaterThan: return one(compare(Op::Gt, in, Scalar::I32));
            case OpSGreaterThanEqual: return one(compare(Op::Ge, in, Scalar::I32));
            case OpSLessThan: return one(compare(Op::Lt, in, Scalar::I32));
            case OpSLessThanEqual: return one(compare(Op::Le, in, Scalar::I32));
            case OpFOrdEqual: case OpFUnordEqual: return one(compare(Op::Eq, in, Scalar::Void));
            case OpFOrdNotEqual: case OpFUnordNotEqual: return one(compare(Op::Ne, in, Scalar::Void));
            case OpFOrdLessThan: return one(compare(Op::Lt, in, Scalar::Void));
            case OpFOrdGreaterThan: return one(compare(Op::Gt, in, Scalar::Void));
            case OpFOrdLessThanEqual: return one(compare(Op::Le, in, Scalar::Void));
            case OpFOrdGreaterThanEqual: return one(compare(Op::Ge, in, Scalar::Void));
            case OpFUnordLessThan: return one(un(Op::Not, compare(Op::Ge, in, Scalar::Void), vt(rtId)));
            case OpFUnordGreaterThan: return one(un(Op::Not, compare(Op::Le, in, Scalar::Void), vt(rtId)));
            case OpFUnordLessThanEqual: return one(un(Op::Not, compare(Op::Gt, in, Scalar::Void), vt(rtId)));
            case OpFUnordGreaterThanEqual: return one(un(Op::Not, compare(Op::Lt, in, Scalar::Void), vt(rtId)));
            case OpIsNan: return one(un(Op::IsNan, get1(w(in, 3)), vt(rtId)));
            case OpIsInf: return one(un(Op::IsInf, get1(w(in, 3)), vt(rtId)));
            case OpAny:
            case OpAll: {
                ValueId v = get1(w(in, 3));
                Type t = typeOfValue(v);
                ValueId r = ext(v, 0);
                for (uint32_t i = 1; i < t.width; ++i) r = bin(op == OpAny ? Op::Or : Op::And, r, ext(v, i), vpir::kBool);
                return one(r);
            }
            // ---- bits
            case OpShiftLeftLogical:
            case OpShiftRightLogical:
            case OpShiftRightArithmetic: {
                Type rt = vt(rtId);
                Type it(op == OpShiftRightArithmetic ? Scalar::I32 : Scalar::U32, rt.width);
                ValueId a = castBits(get1(w(in, 3)), it);
                ValueId s = castBits(get1(w(in, 4)), Type(Scalar::U32, rt.width));
                s = bin(Op::And, s, splat(Type(Scalar::U32, rt.width), 31), Type(Scalar::U32, rt.width));
                s = castBits(s, it);
                return one(castBits(bin(op == OpShiftLeftLogical ? Op::Shl : Op::Shr, a, s, it), rt));
            }
            case OpBitwiseAnd: return one(bin(Op::And, opAs(w(in, 3), vt(rtId)), opAs(w(in, 4), vt(rtId)), vt(rtId)));
            case OpBitwiseOr: return one(bin(Op::Or, opAs(w(in, 3), vt(rtId)), opAs(w(in, 4), vt(rtId)), vt(rtId)));
            case OpBitwiseXor: return one(bin(Op::Xor, opAs(w(in, 3), vt(rtId)), opAs(w(in, 4), vt(rtId)), vt(rtId)));
            case OpNot: return one(un(Op::Not, opAs(w(in, 3), vt(rtId)), vt(rtId)));
            case OpBitFieldInsert: {
                Type t = vt(rtId);
                Inst bi;
                bi.op = Op::BitfieldInsert, bi.type = t;
                bi.args = {opAs(w(in, 3), t), opAs(w(in, 4), t), cast(get1(w(in, 5)), vpir::kU32), cast(get1(w(in, 6)), vpir::kU32)};
                return one(b_->emit(bi));
            }
            case OpBitFieldSExtract:
            case OpBitFieldUExtract: {
                Type rt = vt(rtId), it(op == OpBitFieldSExtract ? Scalar::I32 : Scalar::U32, rt.width);
                ValueId r = tri(Op::BitfieldExtract, castBits(get1(w(in, 3)), it), cast(get1(w(in, 4)), vpir::kU32),
                                cast(get1(w(in, 5)), vpir::kU32), it);
                return one(castBits(r, rt));
            }
            case OpBitReverse: return one(un(Op::BitReverse, opAs(w(in, 3), vt(rtId)), vt(rtId)));
            case OpBitCount: {
                ValueId v = get1(w(in, 3));
                Type vt0 = typeOfValue(v);
                return one(castBits(un(Op::BitCount, v, vt0), vt(rtId)));
            }
            // ---- derivatives
            case OpDPdx: case OpDPdxFine: case OpDPdxCoarse: return one(deriv(Op::Ddx, get1(w(in, 3)), vt(rtId)));
            case OpDPdy: case OpDPdyFine: case OpDPdyCoarse: return one(deriv(Op::Ddy, get1(w(in, 3)), vt(rtId)));
            case OpFwidth: case OpFwidthFine: case OpFwidthCoarse: return one(deriv(Op::Fwidth, get1(w(in, 3)), vt(rtId)));
            // ---- images
            case OpImageSampleImplicitLod: case OpImageSampleExplicitLod: case OpImageSampleDrefImplicitLod:
            case OpImageSampleDrefExplicitLod: case OpImageSampleProjImplicitLod: case OpImageSampleProjExplicitLod:
            case OpImageSampleProjDrefImplicitLod: case OpImageSampleProjDrefExplicitLod:
                return one(sample(in));
            case OpImageFetch: return one(fetch(in, false));
            case OpImageRead: return one(fetch(in, true));
            case OpImageGather:
            case OpImageDrefGather: return one(gather(in));
            case OpImageQuerySizeLod:
            case OpImageQuerySize: return one(querySize(in));
            case OpArrayLength: return one(arrayLength(in));
            case OpImageQueryLevels: {
                Handle h = handle(w(in, 3));
                Inst q;
                q.op = Op::ImageLevels, q.type = vpir::kU32, q.sym = uint32_t(h.texture);
                return one(castBits(b_->emit(q), vt(rtId)));
            }
            case OpImageQueryLod: {
                Handle h = handle(w(in, 3));
                Inst q;
                q.op = Op::ImageQueryLod, q.type = vpir::kF32.withWidth(2), q.sym = uint32_t(h.texture), q.sampler = uint32_t(h.sampler);
                q.args = {sampleCoord(h, get1(w(in, 4)), false)};
                return one(b_->emit(q));
            }
            case OpImageQuerySamples: {
                Handle h = handle(w(in, 3));
                if (h.texture < 0 || !resourceOf(h).multisample) return one(castBits(u32(1), vt(rtId)));
                Inst q;
                q.op = Op::ImageSamples, q.type = vpir::kU32, q.sym = uint32_t(h.texture);
                return one(castBits(b_->emit(q), vt(rtId)));
            }
            // ---- atomics
            case OpAtomicLoad: {
                if (auto tp = scope_->texelPointers.find(w(in, 3)); tp != scope_->texelPointers.end()) {
                    Type t = vt(rtId);  // read as an atomic or with 0
                    return Val{{texelAtomic(tp->second, Op::AtomicOr, {constant(t, {0})}, t, t)}};
                }
                RPtr p = resolve(w(in, 3), false);
                return Val{{loadScalarVec(p, vt(rtId))}};
            }
            case OpAtomicExchange: case OpAtomicCompareExchange: case OpAtomicIIncrement: case OpAtomicIDecrement:
            case OpAtomicIAdd: case OpAtomicISub: case OpAtomicSMin: case OpAtomicUMin: case OpAtomicSMax:
            case OpAtomicUMax: case OpAtomicAnd: case OpAtomicOr: case OpAtomicXor:
                return one(atomic(in));
            // ---- subgroups
            case OpGroupNonUniformElect: {
                ValueId ballot = un(Op::SubgroupBallot, b_->constBool(true), vpir::kU32.withWidth(4));
                ValueId first = un(Op::FindLsb, ext(ballot, 0), vpir::kU32);
                ValueId lane = laneId();
                return one(bin(Op::Eq, lane, first, vpir::kBool));
            }
            case OpGroupNonUniformBallot: return one(castBits(un(Op::SubgroupBallot, get1(w(in, 4)), vpir::kU32.withWidth(4)), vt(rtId)));
            // Ballot masks are uvec4 with the lanes in word 0 (Apple SIMD-groups have 32 lanes).
            case OpGroupNonUniformInverseBallot: {
                ValueId v = castBits(get1(w(in, 4)), vpir::kU32.withWidth(4));
                ValueId bit = bin(Op::And, bin(Op::Shr, ext(v, 0), laneId(), vpir::kU32), u32(1), vpir::kU32);
                return one(bin(Op::Ne, bit, u32(0), vpir::kBool));
            }
            case OpGroupNonUniformBallotBitExtract: {
                ValueId v = castBits(get1(w(in, 4)), vpir::kU32.withWidth(4));
                ValueId idx = castBits(get1(w(in, 5)), vpir::kU32);
                ValueId word = ext(v, 0);
                for (uint32_t k = 1; k < 4; ++k)
                    word = select(bin(Op::Eq, bin(Op::Shr, idx, u32(5), vpir::kU32), u32(k), vpir::kBool), ext(v, k), word);
                ValueId bit = bin(Op::And, bin(Op::Shr, word, bin(Op::And, idx, u32(31), vpir::kU32), vpir::kU32), u32(1),
                                  vpir::kU32);
                return one(bin(Op::Ne, bit, u32(0), vpir::kBool));
            }
            case OpGroupNonUniformBallotBitCount: {
                const uint32_t group = w(in, 4);  // 0 reduce, 1 inclusive scan, 2 exclusive scan
                ValueId v = castBits(get1(w(in, 5)), vpir::kU32.withWidth(4));
                ValueId word = ext(v, 0);
                if (group == 1 || group == 2) {
                    // Lanes below this one (exclusive), or up to it (inclusive): (1 << n) - 1, with
                    // n = lane (+1); n = 32 gives all ones (shifts stay < 32).
                    ValueId lane = laneId();
                    ValueId below = bin(Op::Sub, bin(Op::Shl, u32(1), lane, vpir::kU32), u32(1), vpir::kU32);
                    ValueId mask = below;
                    if (group == 1) mask = bin(Op::Or, below, bin(Op::Shl, u32(1), lane, vpir::kU32), vpir::kU32);
                    word = bin(Op::And, word, mask, vpir::kU32);
                    return one(castBits(un(Op::BitCount, word, vpir::kU32), vt(rtId)));
                }
                ValueId n = un(Op::BitCount, word, vpir::kU32);
                for (uint32_t k = 1; k < 4; ++k) n = bin(Op::Add, n, un(Op::BitCount, ext(v, k), vpir::kU32), vpir::kU32);
                return one(castBits(n, vt(rtId)));
            }
            case OpGroupNonUniformBallotFindLSB: {
                ValueId v = castBits(get1(w(in, 4)), vpir::kU32.withWidth(4));
                return one(castBits(un(Op::FindLsb, ext(v, 0), vpir::kU32), vt(rtId)));
            }
            case OpGroupNonUniformBallotFindMSB: {
                ValueId v = castBits(get1(w(in, 4)), vpir::kU32.withWidth(4));
                return one(castBits(un(Op::FindMsb, ext(v, 0), vpir::kU32), vt(rtId)));
            }
            case OpGroupNonUniformAll: return one(un(Op::SubgroupAll, get1(w(in, 4)), vpir::kBool));
            case OpGroupNonUniformAny: return one(un(Op::SubgroupAny, get1(w(in, 4)), vpir::kBool));
            case OpGroupNonUniformBroadcastFirst: return one(un(Op::SubgroupReadFirst, get1(w(in, 4)), vt(rtId)));
            case OpGroupNonUniformBroadcast:
                return one(bin(Op::SubgroupBroadcast, get1(w(in, 4)), cast(get1(w(in, 5)), vpir::kU32), vt(rtId)));
            case OpGroupNonUniformShuffle:
                return one(bin(Op::SubgroupShuffle, get1(w(in, 4)), cast(get1(w(in, 5)), vpir::kU32), vt(rtId)));
            case OpGroupNonUniformQuadBroadcast:
                return one(bin(Op::QuadShuffle, get1(w(in, 4)), cast(get1(w(in, 5)), vpir::kU32), vt(rtId)));
            case OpGroupNonUniformQuadSwap: {
                uint32_t dir = uint32_t(constScalar(w(in, 5)));
                ValueId lane = bin(Op::And, laneId(), u32(3), vpir::kU32);
                ValueId other = bin(Op::Xor, lane, u32(dir == 0 ? 1 : dir == 1 ? 2 : 3), vpir::kU32);
                return one(bin(Op::QuadShuffle, get1(w(in, 4)), other, vt(rtId)));
            }
            default: break;
        }
        fail("unsupported SPIR-V instruction " + std::to_string(op));
    }
    ValueId laneId() {
        vpir::IoVar v;
        v.builtin = vpir::Builtin::SubgroupInvocation, v.name = "lane", v.type = vpir::kU32;
        int idx = addInput(v);
        return b_->loadInput(uint32_t(idx), vpir::kU32);
    }
    ValueId deriv(Op op, ValueId v, Type t) {
        if (stage_ != vpir::Stage::Fragment) return zeroOf(t);
        return un(op, v, t);
    }
    ValueId dot(ValueId a, ValueId c) {
        Type t = typeOfValue(a);
        if (t.width == 1) return bin(Op::Mul, a, c, t);
        return bin(Op::Dot, a, c, t.element());
    }

    Val compositeConstruct(const Instr& in) {
        uint32_t rtId = w(in, 1);
        const TypeInfo& rt = T(rtId);
        if (rt.kind == TypeKind::Vector) {
            std::vector<ValueId> parts;
            for (uint32_t i = 3; i < in.count; ++i) parts.push_back(get1(w(in, i)));
            if (parts.size() == 1 && typeOfValue(parts[0]).width == rt.count) return Val{{parts[0]}};
            return Val{{b_->construct(parts, vt(rtId))}};
        }
        Val r;
        for (uint32_t i = 3; i < in.count; ++i) {
            Val p = get(w(in, i));
            r.v.insert(r.v.end(), p.v.begin(), p.v.end());
        }
        return r;
    }
    // Leaf range of member `index` of a composite type.
    std::pair<uint32_t, uint32_t> memberLeaves(uint32_t typeId, uint32_t index, uint32_t& memberType) {
        const TypeInfo& t = T(typeId);
        if (t.kind == TypeKind::Struct) {
            if (index >= t.members.size()) fail("struct member index " + std::to_string(index) + " out of range");
            uint32_t first = 0;
            for (uint32_t i = 0; i < index; ++i) first += leafCount(t.members[i]);
            memberType = t.members.at(index);
            return {first, leafCount(memberType)};
        }
        if (t.kind == TypeKind::Array || t.kind == TypeKind::Matrix) {
            memberType = t.elem;
            uint32_t n = leafCount(t.elem);
            return {index * n, n};
        }
        fail("bad composite index");
    }
    Val compositeExtract(Val v, uint32_t typeId, const Instr& in, uint32_t firstIndex) {
        uint32_t ty = typeId;
        uint32_t base = 0, count = uint32_t(v.v.size());
        for (uint32_t i = firstIndex; i < in.count; ++i) {
            uint32_t idx = w(in, i);
            if (T(ty).kind == TypeKind::Vector) {
                if (base >= v.v.size()) fail("composite extract past the value's leaves");
                return Val{{ext(v.v[base], idx)}};
            }
            uint32_t mt;
            auto [first, n] = memberLeaves(ty, idx, mt);
            base += first, count = n, ty = mt;
        }
        if (uint64_t(base) + count > v.v.size()) fail("composite extract past the value's leaves");
        Val r;
        r.v.assign(v.v.begin() + base, v.v.begin() + base + count);
        return r;
    }
    Val compositeInsert(const Instr& in) {
        Val obj = get(w(in, 3));
        Val comp = get(w(in, 4));
        uint32_t ty = w(in, 1);
        uint32_t base = 0;
        for (uint32_t i = 5; i < in.count; ++i) {
            uint32_t idx = w(in, i);
            if (T(ty).kind == TypeKind::Vector) {
                if (base >= obj.v.size() || comp.v.empty()) fail("composite insert past the value's leaves");
                Inst ins;
                ins.op = Op::Insert, ins.type = vt(ty);
                ins.args = {sameBits(obj.v[base], ins.type), sameBits(comp.v[0], ins.type.element())};
                ins.lits = {idx};
                obj.v[base] = b_->emit(ins);
                return obj;
            }
            uint32_t mt;
            auto [first, n] = memberLeaves(ty, idx, mt);
            base += first, ty = mt;
            (void)n;
        }
        if (uint64_t(base) + comp.v.size() > obj.v.size()) fail("composite insert past the value's leaves");
        for (size_t i = 0; i < comp.v.size(); ++i) obj.v[base + i] = comp.v[i];
        return obj;
    }

    // ---- image ops ------------------------------------------------------------------------------
    // Coordinate for sampling: f32 vector of the texture's coordinate width (+ layer).
    // The texture resource of a handle; a texel buffer or sampler here is a frontend gap, reported
    // as such instead of an out-of-range index.
    const vpir::Resource& resourceOf(const Handle& h, const char* what = "image op") const {
        if (h.texture < 0 || size_t(h.texture) >= out_.resources.size())
            fail(std::string(what) + " on a handle without a texture (" +
                 (h.buffer >= 0 ? "texel buffer" : h.sampler >= 0 ? "sampler" : "unresolved") + ")");
        return out_.resources[size_t(h.texture)];
    }
    // A 1D image (sampled as a 2D texture one texel high, see texDim).
    bool oneD(const Handle& h) const { return h.imageType && T(h.imageType).dim == Dim::Dim1D; }
    // 1D coordinate (x[, layer][, q]) as the 2D one (x, y[, layer][, q]): `y` is the texel row
    // (0.5 normalized: the centre, so filtering and wrap modes never leave it; 0 for integer).
    ValueId widen1D(ValueId c, ValueId y) {
        Type t = typeOfValue(c);
        std::vector<ValueId> parts{t.width == 1 ? c : ext(c, 0), y};
        for (unsigned i = 1; i < t.width; ++i) parts.push_back(ext(c, i));
        return vecOf(parts, t.withWidth(uint8_t(t.width + 1)));
    }
    ValueId sampleCoord(const Handle& h, ValueId c, bool proj) {
        if (oneD(h)) {
            Type t0 = typeOfValue(c);
            if (!t0.isFloat()) c = cast(c, vpir::kF32.withWidth(t0.width));
            if (proj) {  // (s, q): the row stays at the centre once divided by q
                ValueId q = ext(c, typeOfValue(c).width - 1);
                c = vecOf({ext(c, 0), bin(Op::Mul, f32(0.5f), q, vpir::kF32), q}, vpir::kF32.withWidth(3));
            } else {
                c = widen1D(c, f32(0.5f));
            }
        }
        const vpir::Resource& r = resourceOf(h);
        unsigned want = vpir::coordWidth(r.dim) + (vpir::isArrayed(r.dim) ? 1 : 0);
        Type t = typeOfValue(c);
        if (!t.isFloat()) c = cast(c, vpir::kF32.withWidth(t.width)), t = typeOfValue(c);
        if (proj) {
            ValueId q = ext(c, t.width - 1);
            std::vector<ValueId> parts;
            for (unsigned i = 0; i < want; ++i) parts.push_back(bin(Op::Div, ext(c, i), q, vpir::kF32));
            return vecOf(parts, vpir::kF32.withWidth(uint8_t(want)));
        }
        if (t.width == want) return c;
        std::vector<ValueId> parts;
        for (unsigned i = 0; i < want; ++i) parts.push_back(i < t.width ? ext(c, i) : f32(0));
        return vecOf(parts, vpir::kF32.withWidth(uint8_t(want)));
    }
    ValueId intCoord(const Handle& h, ValueId c) {
        if (oneD(h)) {
            Type t0 = typeOfValue(c);
            c = widen1D(castBits(c, Type(Scalar::U32, t0.width)), u32(0));
        }
        const vpir::Resource& r = resourceOf(h);
        unsigned want = vpir::coordWidth(r.dim) + (vpir::isArrayed(r.dim) ? 1 : 0);
        if (r.dim == vpir::TexDim::Cube || r.dim == vpir::TexDim::CubeArray) want = 3 + (vpir::isArrayed(r.dim) ? 1 : 0);
        Type t = typeOfValue(c);
        c = castBits(c, Type(Scalar::U32, t.width));
        if (t.width == want) return c;
        std::vector<ValueId> parts;
        for (unsigned i = 0; i < want; ++i) parts.push_back(i < t.width ? ext(c, i) : u32(0));
        return vecOf(parts, vpir::kU32.withWidth(uint8_t(want)));
    }
    struct ImageArgs {
        uint8_t mask = 0;
        std::vector<ValueId> extra;
    };
    // Optional SPIR-V image operands starting at word `at`, as VPIR extras in ImageOperand bit
    // order (lod, bias, grad, offset, compare). `allow` masks what the VPIR op accepts.
    ImageArgs imageOperands(const Instr& in, uint32_t at, const Handle& h, uint8_t allow, ValueId dref) {
        ImageArgs a;
        uint32_t mask = at < in.count ? w(in, at) : 0;
        uint32_t i = at + 1;
        ValueId lod = kNoValue, bias = kNoValue, gx = kNoValue, gy = kNoValue, offset = kNoValue;
        const vpir::Resource& r = resourceOf(h);
        unsigned cw = vpir::coordWidth(r.dim);
        if (mask & ImgOp::Bias) bias = cast(get1(w(in, i++)), vpir::kF32);
        if (mask & ImgOp::Lod) lod = cast(get1(w(in, i++)), vpir::kF32);
        if (mask & ImgOp::Grad) gx = get1(w(in, i++)), gy = get1(w(in, i++));
        if (mask & ImgOp::ConstOffset) offset = get1(w(in, i++));
        if (mask & ImgOp::Offset) offset = get1(w(in, i++));
        if (offset != kNoValue && oneD(h)) {  // 1D offset: (x, 0) on the 2D texture
            Type ot = typeOfValue(offset);
            ValueId ox = castBits(ot.width == 1 ? offset : ext(offset, 0), vpir::kI32);
            offset = vecOf({ox, constant(vpir::kI32, {0})}, vpir::kI32.withWidth(2));
        }
        if (mask & ImgOp::ConstOffsets) fail("ConstOffsets gathers are not supported");
        if (mask & ImgOp::Sample) fail("multisampled image reads are not supported");
        // MinLod and the rest are hints the target cannot express; ignored.
        const bool fragment = stage_ == vpir::Stage::Fragment;
        if (!(allow & vpir::kImageGrad) && gx != kNoValue) gx = gy = kNoValue, lod = lod == kNoValue && !fragment ? f32(0) : lod;
        if (!fragment || !(allow & vpir::kImageBias)) bias = kNoValue;
        if ((allow & vpir::kImageLod) && lod == kNoValue && gx == kNoValue && !fragment && r.dim != vpir::TexDim::D1)
            lod = f32(0);
        if (lod != kNoValue && (allow & vpir::kImageLod)) a.mask |= vpir::kImageLod, a.extra.push_back(lod);
        if (bias != kNoValue) a.mask |= vpir::kImageBias, a.extra.push_back(bias);
        if (gx != kNoValue) {
            Type g = vpir::kF32.withWidth(uint8_t(cw));
            a.mask |= vpir::kImageGrad, a.extra.push_back(fitF(gx, g)), a.extra.push_back(fitF(gy, g));
        }
        if (offset != kNoValue && (allow & vpir::kImageOffset) && r.dim != vpir::TexDim::Cube && r.dim != vpir::TexDim::CubeArray &&
            r.dim != vpir::TexDim::D1) {
            a.mask |= vpir::kImageOffset;
            a.extra.push_back(castBits(offset, Type(Scalar::I32, uint8_t(cw))));
        }
        if (dref != kNoValue) a.mask |= vpir::kImageCompare, a.extra.push_back(cast(dref, vpir::kF32));
        return a;
    }
    ValueId fitF(ValueId v, Type t) {
        Type vt0 = typeOfValue(v);
        if (vt0 == t) return v;
        std::vector<ValueId> parts;
        for (unsigned i = 0; i < t.width; ++i) parts.push_back(i < vt0.width ? ext(v, i) : f32(0));
        return vecOf(parts, t);
    }
    ValueId texelResult(const Handle& h, ValueId v, uint32_t rtId) {
        const vpir::Resource& r = resourceOf(h);
        Type rt = vt(rtId);
        Type got = typeOfValue(v);
        if (r.depth && got.width == 1 && rt.width > 1) {
            std::vector<ValueId> parts{v};
            while (parts.size() < rt.width) parts.push_back(parts.size() == 3 ? f32(1) : f32(0));
            v = vecOf(parts, vpir::kF32.withWidth(rt.width));
            got = typeOfValue(v);
        }
        if (got == rt) return v;
        if (got.width != rt.width) {
            if (rt.width == 1) return castBits(ext(v, 0), rt);
            std::vector<ValueId> parts;
            for (unsigned i = 0; i < rt.width; ++i) parts.push_back(ext(v, i));
            v = vecOf(parts, got.withWidth(rt.width));
        }
        Type cur = typeOfValue(v);
        if (cur.isInt() && rt.isInt()) return un(Op::Bitcast, v, rt);
        return cast(v, rt);
    }
    ValueId sample(const Instr& in) {
        const uint16_t op = in.op;
        Handle h = handle(w(in, 3));
        if (h.texture < 0 || h.sampler < 0) fail("sampling needs a texture and a sampler");
        bool proj = op == OpImageSampleProjImplicitLod || op == OpImageSampleProjExplicitLod ||
                    op == OpImageSampleProjDrefImplicitLod || op == OpImageSampleProjDrefExplicitLod;
        bool dref = op == OpImageSampleDrefImplicitLod || op == OpImageSampleDrefExplicitLod ||
                    op == OpImageSampleProjDrefImplicitLod || op == OpImageSampleProjDrefExplicitLod;
        ValueId coordRaw = get1(w(in, 4));
        ValueId coord = sampleCoord(h, coordRaw, proj);
        uint32_t at = dref ? 6 : 5;
        ValueId d = kNoValue;
        if (dref) {
            d = get1(w(in, 5));
            if (proj) d = bin(Op::Div, d, ext(coordRaw, typeOfValue(coordRaw).width - 1), vpir::kF32);
        }
        uint8_t allow = dref ? uint8_t(vpir::kImageLod | vpir::kImageOffset)
                             : uint8_t(vpir::kImageLod | vpir::kImageBias | vpir::kImageGrad | vpir::kImageOffset);
        ImageArgs a = imageOperands(in, at, h, allow, d);
        std::vector<ValueId> extra = a.extra;
        const vpir::Resource& r = resourceOf(h);
        Type t = r.depth ? vpir::kF32 : Type(r.elem, 4);
        ValueId v = b_->imageSample(uint32_t(h.texture), uint32_t(h.sampler), coord, a.mask, extra, t);
        return texelResult(h, v, w(in, 1));
    }
    ValueId fetch(const Instr& in, bool storage) {
        Handle h = handle(w(in, 3));
        if (h.buffer >= 0) return texelBufferRead(h, get1(w(in, 4)), w(in, 1));
        if (h.texture < 0) fail("fetch from a non-texture");
        ValueId coord;
        if (h.imageType && T(h.imageType).dim == Dim::SubpassData) {
            // subpassLoad: this fragment's own pixel (the driver turns it into a framebuffer fetch).
            ValueId pos = positionInput();
            ValueId xy = vecOf({ext(pos, 0), ext(pos, 1)}, vpir::kF32.withWidth(2));
            coord = un(Op::Convert, xy, vpir::kU32.withWidth(2));
            const vpir::Resource& r = resourceOf(h);
            if (r.dim == vpir::TexDim::D2Array)
                coord = vecOf({ext(coord, 0), ext(coord, 1), viewIndex()}, vpir::kU32.withWidth(3));
            Inst f;
            f.op = Op::ImageFetch, f.sym = uint32_t(h.texture), f.type = Type(r.elem, 4), f.args = {coord};
            return texelResult(h, b_->emit(f), w(in, 1));
        }
        coord = intCoord(h, get1(w(in, 4)));
        const vpir::Resource& r = resourceOf(h);
        Inst f;
        f.op = Op::ImageFetch;
        f.sym = uint32_t(h.texture);
        f.type = r.depth ? vpir::kF32 : Type(r.elem, 4);
        f.args = {coord};
        if (!storage && in.count > 5) {
            uint32_t mask = w(in, 5);
            uint32_t i = 6;
            if (mask & ImgOp::Lod) {
                ValueId l = castBits(get1(w(in, i++)), vpir::kU32);
                f.args.push_back(l);
                f.image = vpir::kImageLod;
            }
            // texelFetchOffset: Metal's read() takes no offset; add it to the texel coordinate
            // (x/y/z, never the array layer). Out-of-range results are undefined in Vulkan too.
            ValueId off = kNoValue;
            if (mask & ImgOp::ConstOffset) off = get1(w(in, i++));
            if (mask & ImgOp::Offset) off = get1(w(in, i++));
            if (mask & ImgOp::ConstOffsets) ++i;
            // texelFetch(sampler2DMS, p, sample): the sample index rides in the lod operand
            // (Metal's texture2d_ms::read(coord, sample)); a forced-1x image reads its one sample.
            if (mask & ImgOp::Sample) {
                ValueId smp = castBits(get1(w(in, i++)), vpir::kU32);
                if (r.multisample) f.args.push_back(smp), f.image = vpir::kImageLod;
            }
            if (off != kNoValue) {
                Type ot = typeOfValue(off);
                Type ct = typeOfValue(coord);
                const unsigned spatial = oneD(h) ? 1 : ot.width;
                std::vector<ValueId> parts;
                for (unsigned c = 0; c < ct.width; ++c) {
                    ValueId e = ext(coord, c);
                    if (c < spatial && c < ot.width)
                        e = bin(Op::Add, e, castBits(ext(off, c), vpir::kU32), vpir::kU32);
                    parts.push_back(e);
                }
                f.args[0] = vecOf(parts, ct);
            }
        }
        return texelResult(h, b_->emit(f), w(in, 1));
    }
    ValueId gather(const Instr& in) {
        Handle h = handle(w(in, 3));
        ValueId coord = sampleCoord(h, get1(w(in, 4)), false);
        bool dref = in.op == OpImageDrefGather;
        uint32_t comp = dref ? 0 : uint32_t(constScalar(w(in, 5)));
        ValueId d = dref ? get1(w(in, 5)) : kNoValue;
        ImageArgs a = imageOperands(in, 6, h, vpir::kImageOffset, d);
        std::vector<ValueId> extra = a.extra;
        const vpir::Resource& r = resourceOf(h);
        Inst g;
        g.op = Op::ImageGather;
        g.sym = uint32_t(h.texture), g.sampler = uint32_t(h.sampler);
        g.type = Type(r.depth ? Scalar::F32 : r.elem, 4);
        g.lits = {comp};
        g.args = {coord};
        g.args.insert(g.args.end(), extra.begin(), extra.end());
        g.image = a.mask;
        return texelResult(h, b_->emit(g), w(in, 1));
    }
    // OpArrayLength: elements of the runtime array that ends a storage block, from the size of the
    // range bound to the block's slot (a BufferSize argument the backend fills):
    // (max(size, offset) - offset) / stride.
    ValueId arrayLength(const Instr& in) {
        RPtr p = resolve(w(in, 3), false);
        if (p.kind != RPtr::Mem || !p.explicitLayout || p.target == UINT32_MAX || p.target >= out_.resources.size() ||
            out_.resources[p.target].kind != vpir::ResourceKind::Buffer)
            fail("OpArrayLength on something that is not a storage buffer block");
        if (p.dyn != kNoValue) fail("OpArrayLength on a dynamically offset block");
        const TypeInfo& st = T(p.type);
        const uint32_t member = w(in, 4);
        if (st.kind != TypeKind::Struct || member >= st.members.size()) fail("OpArrayLength: bad struct member");
        const uint32_t arr = st.members[member];
        if (T(arr).kind != TypeKind::RuntimeArray) fail("OpArrayLength: the member is not a runtime array");
        const uint32_t off = p.off + memberOffset(p.type, member, true);
        const uint32_t stride = arrayStride(arr, true);
        if (!stride) fail("OpArrayLength: runtime array without an ArrayStride");
        const int bs = resource(vpir::ResourceKind::BufferSize, out_.resources[p.target].index, nullptr);
        Inst q;
        q.op = Op::BufferSize;
        q.sym = uint32_t(bs);
        q.type = vpir::kU32;
        ValueId size = b_->emit(q);
        ValueId bytes = bin(Op::Sub, bin(Op::Max, size, u32(off), vpir::kU32), u32(off), vpir::kU32);
        return castBits(bin(Op::Div, bytes, u32(stride), vpir::kU32), vt(w(in, 1)));
    }
    ValueId querySize(const Instr& in) {
        Handle h = handle(w(in, 3));
        if (h.texture < 0) fail("size query on a texel buffer is not supported");
        const vpir::Resource& r = resourceOf(h);
        unsigned n = r.dim == vpir::TexDim::D1 ? 1 : (r.dim == vpir::TexDim::D2 || r.dim == vpir::TexDim::Cube) ? 2 : 3;
        const bool d1 = oneD(h);
        Inst q;
        q.op = Op::ImageSize;
        q.sym = uint32_t(h.texture);
        q.type = vpir::kU32.withWidth(uint8_t(n));
        if (in.op == OpImageQuerySizeLod && r.kind == vpir::ResourceKind::Texture) {
            q.args = {castBits(get1(w(in, 4)), vpir::kU32)};
            q.image = vpir::kImageLod;
        }
        ValueId v = b_->emit(q);
        if (d1)  // (width, 1[, layers]) -> width[, layers]
            v = n == 2 ? ext(v, 0) : vecOf({ext(v, 0), ext(v, 2)}, vpir::kU32.withWidth(2));
        return castBits(v, vt(w(in, 1)));
    }
    ValueId texelBufferRead(const Handle& h, ValueId index, uint32_t rtId) {
        uint32_t bytes = vpgpu::formatInfo(h.texelFormat).bytesPerBlock;
        RPtr p;
        p.target = uint32_t(h.buffer);
        p.dyn = bin(Op::Mul, castBits(index, vpir::kU32), u32(bytes), vpir::kU32);
        std::vector<ValueId> comps = unpackFormat(p, h.texelFormat);
        Type rt = vt(rtId);
        Type et = rt.element();
        for (auto& c : comps) c = typeOfValue(c).isInt() && et.isInt() ? castBits(c, et) : cast(c, et);
        while (comps.size() < rt.width) comps.push_back(comps.size() == 3 ? (et.isFloat() ? f32(1) : constant(et, {1})) : zeroOf(et));
        comps.resize(rt.width);
        return vecOf(comps, rt);
    }
    void imageWrite(const Instr& in) {
        Handle h = handle(w(in, 1));
        ValueId value = get1(w(in, 3));
        if (h.buffer >= 0) {
            // Texel buffer store: the 32-bit formats and RGBA8 unorm.
            uint32_t bytes = vpgpu::formatInfo(h.texelFormat).bytesPerBlock;
            ValueId addr = bin(Op::Mul, castBits(get1(w(in, 2)), vpir::kU32), u32(bytes), vpir::kU32);
            written_.insert(h.buffer);
            using F = vpgpu::Format;
            Type vt0 = typeOfValue(value);
            switch (h.texelFormat) {
                case F::R32Float: case F::R32Uint: case F::R32Sint:
                    b_->store(uint32_t(h.buffer), addr, ext(value, 0));
                    return;
                case F::RGBA32Float: case F::RGBA32Uint: case F::RGBA32Sint:
                    if (vt0.width == 4) b_->store(uint32_t(h.buffer), addr, value);
                    else fail("texel buffer store width");
                    return;
                case F::RGBA8Unorm: {
                    ValueId acc = u32(0);
                    for (uint32_t i = 0; i < 4; ++i) {
                        ValueId c = un(Op::Saturate, cast(ext(value, i), vpir::kF32), vpir::kF32);
                        c = un(Op::Convert, un(Op::Round, bin(Op::Mul, c, f32(255.f), vpir::kF32), vpir::kF32), vpir::kU32);
                        acc = bin(Op::Or, acc, bin(Op::Shl, c, u32(8 * i), vpir::kU32), vpir::kU32);
                    }
                    b_->store(uint32_t(h.buffer), addr, acc);
                    return;
                }
                default: fail("unsupported storage texel buffer format");
            }
        }
        const vpir::Resource& r = resourceOf(h);
        ValueId coord = intCoord(h, get1(w(in, 2)));
        Type tt(r.elem, 4);
        Type vt0 = typeOfValue(value);
        std::vector<ValueId> parts;
        for (uint32_t i = 0; i < 4; ++i) parts.push_back(i < vt0.width ? ext(value, i) : zeroOf(Type(vt0.scalar)));
        ValueId v4 = vecOf(parts, vt0.withWidth(4));
        if (typeOfValue(v4) != tt) v4 = typeOfValue(v4).isInt() && tt.isInt() ? un(Op::Bitcast, v4, tt) : cast(v4, tt);
        written_.insert(h.texture);
        b_->imageStore(uint32_t(h.texture), coord, v4);
    }
    ValueId atomic(const Instr& in) {
        Type rt = vt(w(in, 1));
        uint32_t ptrId = w(in, 3);
        auto tp = scope_->texelPointers.find(ptrId);
        Op op;
        switch (in.op) {
            case OpAtomicExchange: op = Op::AtomicExchange; break;
            case OpAtomicCompareExchange: op = Op::AtomicCompareExchange; break;
            case OpAtomicIIncrement: case OpAtomicIAdd: op = Op::AtomicAdd; break;
            case OpAtomicIDecrement: case OpAtomicISub: op = Op::AtomicSub; break;
            case OpAtomicSMin: case OpAtomicUMin: op = Op::AtomicMin; break;
            case OpAtomicSMax: case OpAtomicUMax: op = Op::AtomicMax; break;
            case OpAtomicAnd: op = Op::AtomicAnd; break;
            case OpAtomicOr: op = Op::AtomicOr; break;
            default: op = Op::AtomicXor; break;
        }
        Scalar s = (in.op == OpAtomicSMin || in.op == OpAtomicSMax) ? Scalar::I32
                   : (in.op == OpAtomicUMin || in.op == OpAtomicUMax) ? Scalar::U32 : rt.scalar;
        Type t(s);
        std::vector<ValueId> vals;
        if (in.op == OpAtomicIIncrement || in.op == OpAtomicIDecrement) vals.push_back(constant(t, {1}));
        else if (in.op == OpAtomicCompareExchange) {
            vals.push_back(castBits(get1(w(in, 8)), t));  // comparator
            vals.push_back(castBits(get1(w(in, 7)), t));  // value
        } else vals.push_back(castBits(get1(w(in, 6)), t));
        if (tp != scope_->texelPointers.end()) return texelAtomic(tp->second, op, vals, t, rt);
        RPtr p = resolve(ptrId, true);
        if (p.kind != RPtr::Mem) fail("atomic on a non-memory pointer");
        if (p.target < out_.resources.size()) written_.insert(int(p.target));
        return castBits(b_->atomic(op, p.target, address(p), vals, t), rt);
    }
    // Atomic through an OpImageTexelPointer `ti`: a storage image texel, or a word of a storage
    // texel buffer (imageAtomic* on imageBuffer, HLSL Interlocked* on RWBuffer: particle and
    // skinning counters). `vals` are typed `t`; the result is returned as `rt`.
    ValueId texelAtomic(const Instr& ti, Op op, std::vector<ValueId> vals, Type t, Type rt) {
        Handle h = handle(w(ti, 3));
        if (h.buffer >= 0) {
            if (vpgpu::formatInfo(h.texelFormat).bytesPerBlock != 4 || !t.isInt())
                fail("atomic on a texel buffer that is not 32-bit integer (" + m_.nameOf(w(ti, 3)) + ")");
            ValueId addr = bin(Op::Mul, castBits(get1(w(ti, 4)), vpir::kU32), u32(4), vpir::kU32);
            written_.insert(h.buffer);
            return castBits(b_->atomic(op, uint32_t(h.buffer), addr, vals, t), rt);
        }
        const vpir::Resource& r = resourceOf(h, "image atomic");
        ValueId coord = intCoord(h, get1(w(ti, 4)));
        Type et(r.elem);
        for (auto& v : vals) v = castBits(v, et);
        Op iop = Op(uint8_t(op) - uint8_t(Op::AtomicAdd) + uint8_t(Op::ImageAtomicAdd));
        Inst a;
        a.op = iop, a.type = et, a.sym = uint32_t(h.texture);
        a.args = {coord};
        a.args.insert(a.args.end(), vals.begin(), vals.end());
        written_.insert(h.texture);
        return castBits(b_->emit(a), rt);
    }

    // ---- GLSL.std.450 ---------------------------------------------------------------------------
    void extInst(const Instr& in) {
        if (w(in, 3) != m_.glslExt) fail("unsupported extended instruction set");
        uint32_t e = w(in, 4);
        uint32_t rtId = w(in, 1), id = w(in, 2);
        Type t = vt(rtId);
        auto a = [&](uint32_t i) { return get1(w(in, 5 + i)); };
        auto ai = [&](uint32_t i, Scalar s) { return castBits(a(i), Type(s, typeOfValue(a(i)).width)); };
        auto res = [&](ValueId v) { define1(id, castBits(v, t)); };
        auto sv = [&](float f) { return splat(t, floatBits(f)); };
        Type it(Scalar::I32, t.width), ut(Scalar::U32, t.width);
        switch (e) {
            case GLSL_Round: case GLSL_RoundEven: return res(un(Op::Round, a(0), t));
            case GLSL_Trunc: return res(un(Op::Trunc, a(0), t));
            case GLSL_FAbs: return res(un(Op::Abs, a(0), t));
            case GLSL_SAbs: return res(un(Op::Abs, ai(0, Scalar::I32), it));
            case GLSL_FSign: {
                Type bt(Scalar::Bool, t.width);
                ValueId x = a(0);
                ValueId pos = select(bin(Op::Gt, x, sv(0), bt), sv(1), sv(0));
                return res(select(bin(Op::Lt, x, sv(0), bt), sv(-1), pos));
            }
            case GLSL_SSign: {
                Type bt(Scalar::Bool, t.width);
                ValueId x = ai(0, Scalar::I32);
                ValueId z = splat(it, 0);
                ValueId pos = select(bin(Op::Gt, x, z, bt), splat(it, 1), z);
                return res(select(bin(Op::Lt, x, z, bt), splat(it, 0xFFFFFFFFu), pos));
            }
            case GLSL_Floor: return res(un(Op::Floor, a(0), t));
            case GLSL_Ceil: return res(un(Op::Ceil, a(0), t));
            case GLSL_Fract: return res(un(Op::Fract, a(0), t));
            case GLSL_Radians: return res(bin(Op::Mul, a(0), sv(0.017453292519943295f), t));
            case GLSL_Degrees: return res(bin(Op::Mul, a(0), sv(57.29577951308232f), t));
            case GLSL_Sin: return res(un(Op::Sin, a(0), t));
            case GLSL_Cos: return res(un(Op::Cos, a(0), t));
            case GLSL_Tan: return res(bin(Op::Div, un(Op::Sin, a(0), t), un(Op::Cos, a(0), t), t));
            case GLSL_Asin: return res(asinApprox(a(0), t));
            case GLSL_Acos: return res(bin(Op::Sub, sv(1.5707963267948966f), asinApprox(a(0), t), t));
            case GLSL_Atan: return res(atanApprox(a(0), t));
            case GLSL_Atan2: return res(atan2Approx(a(0), a(1), t));
            case GLSL_Sinh: case GLSL_Cosh: case GLSL_Tanh: {
                ValueId ex = expE(a(0), t), enx = expE(un(Op::Neg, a(0), t), t);
                ValueId s = bin(Op::Mul, bin(Op::Sub, ex, enx, t), sv(0.5f), t);
                ValueId c = bin(Op::Mul, bin(Op::Add, ex, enx, t), sv(0.5f), t);
                if (e == GLSL_Sinh) return res(s);
                if (e == GLSL_Cosh) return res(c);
                return res(bin(Op::Div, s, c, t));
            }
            case GLSL_Asinh: case GLSL_Acosh: case GLSL_Atanh: {
                ValueId x = a(0);
                if (e == GLSL_Atanh) {
                    ValueId q = bin(Op::Div, bin(Op::Add, sv(1), x, t), bin(Op::Sub, sv(1), x, t), t);
                    return res(bin(Op::Mul, logE(q, t), sv(0.5f), t));
                }
                ValueId r = un(Op::Sqrt, bin(Op::Add, bin(Op::Mul, x, x, t), sv(e == GLSL_Asinh ? 1.f : -1.f), t), t);
                return res(logE(bin(Op::Add, x, r, t), t));
            }
            case GLSL_Pow: return res(bin(Op::Pow, a(0), a(1), t));
            case GLSL_Exp: return res(expE(a(0), t));
            case GLSL_Log: return res(logE(a(0), t));
            case GLSL_Exp2: return res(un(Op::Exp2, a(0), t));
            case GLSL_Log2: return res(un(Op::Log2, a(0), t));
            case GLSL_Sqrt: return res(un(Op::Sqrt, a(0), t));
            case GLSL_InverseSqrt: return res(un(Op::Rsq, a(0), t));
            case GLSL_Determinant: return define1(id, determinant(get(w(in, 5)), typeOf(w(in, 5))));
            case GLSL_MatrixInverse: return define(id, inverse(get(w(in, 5)), typeOf(w(in, 5))));
            case GLSL_Modf: case GLSL_ModfStruct: {
                ValueId x = a(0);
                Type xt = typeOfValue(x);
                ValueId whole = un(Op::Trunc, x, xt);
                ValueId frac = bin(Op::Sub, x, whole, xt);
                if (e == GLSL_ModfStruct) return define(id, Val{{frac, whole}});
                RPtr p = resolve(w(in, 6), true);
                size_t at = 0;
                storeMem(p, Val{{whole}}, at);
                return define1(id, frac);
            }
            case GLSL_FMin: case GLSL_NMin: return res(bin(Op::Min, a(0), a(1), t));
            case GLSL_FMax: case GLSL_NMax: return res(bin(Op::Max, a(0), a(1), t));
            case GLSL_UMin: return res(bin(Op::Min, ai(0, Scalar::U32), ai(1, Scalar::U32), ut));
            case GLSL_UMax: return res(bin(Op::Max, ai(0, Scalar::U32), ai(1, Scalar::U32), ut));
            case GLSL_SMin: return res(bin(Op::Min, ai(0, Scalar::I32), ai(1, Scalar::I32), it));
            case GLSL_SMax: return res(bin(Op::Max, ai(0, Scalar::I32), ai(1, Scalar::I32), it));
            case GLSL_FClamp: case GLSL_NClamp: return res(tri(Op::Clamp, a(0), a(1), a(2), t));
            case GLSL_UClamp: return res(tri(Op::Clamp, ai(0, Scalar::U32), ai(1, Scalar::U32), ai(2, Scalar::U32), ut));
            case GLSL_SClamp: return res(tri(Op::Clamp, ai(0, Scalar::I32), ai(1, Scalar::I32), ai(2, Scalar::I32), it));
            case GLSL_FMix: {
                ValueId x = a(2);
                if (typeOfValue(x).width != t.width) x = splatValue(x, t.width);
                return res(tri(Op::Mix, a(0), a(1), x, t));
            }
            case GLSL_Step: {
                Type bt(Scalar::Bool, t.width);
                ValueId edge = a(0);
                if (typeOfValue(edge).width != t.width) edge = splatValue(edge, t.width);
                return res(select(bin(Op::Lt, a(1), edge, bt), sv(0), sv(1)));
            }
            case GLSL_SmoothStep: {
                ValueId e0 = a(0), e1 = a(1);
                if (typeOfValue(e0).width != t.width) e0 = splatValue(e0, t.width), e1 = splatValue(e1, t.width);
                ValueId x = un(Op::Saturate, bin(Op::Div, bin(Op::Sub, a(2), e0, t), bin(Op::Sub, e1, e0, t), t), t);
                ValueId k = bin(Op::Sub, sv(3), bin(Op::Mul, sv(2), x, t), t);
                return res(bin(Op::Mul, bin(Op::Mul, x, x, t), k, t));
            }
            case GLSL_Fma: return res(tri(Op::Fma, a(0), a(1), a(2), t));
            case GLSL_Ldexp: {
                ValueId ex = cast(ai(1, Scalar::I32), t);
                return res(bin(Op::Mul, a(0), un(Op::Exp2, ex, t), t));
            }
            case GLSL_Frexp: case GLSL_FrexpStruct: {
                ValueId x = a(0);
                ValueId ex = un(Op::Floor, un(Op::Log2, un(Op::Abs, x, t), t), t);
                ex = bin(Op::Add, ex, sv(1), t);
                Type bt(Scalar::Bool, t.width);
                ValueId zero = bin(Op::Eq, x, sv(0), bt);
                ex = select(zero, sv(0), ex);
                ValueId m = select(zero, sv(0), bin(Op::Mul, x, un(Op::Exp2, un(Op::Neg, ex, t), t), t));
                ValueId ei = un(Op::Convert, ex, it);
                if (e == GLSL_FrexpStruct) return define(id, Val{{m, ei}});
                RPtr p = resolve(w(in, 6), true);
                size_t at = 0;
                storeMem(p, Val{{castBits(ei, vt(T(typeOf(w(in, 6))).elem))}}, at);
                return define1(id, m);
            }
            case GLSL_PackHalf2x16: {
                ValueId v = a(0);
                ValueId lo = floatToHalf(ext(v, 0)), hi = floatToHalf(ext(v, 1));
                lo = bin(Op::And, lo, u32(0xFFFF), vpir::kU32);
                hi = bin(Op::Shl, bin(Op::And, hi, u32(0xFFFF), vpir::kU32), u32(16), vpir::kU32);
                return res(bin(Op::Or, lo, hi, vpir::kU32));
            }
            case GLSL_UnpackHalf2x16: {
                ValueId x = ai(0, Scalar::U32);
                ValueId lo = halfToFloat(bin(Op::And, x, u32(0xFFFF), vpir::kU32));
                ValueId hi = halfToFloat(bin(Op::Shr, x, u32(16), vpir::kU32));
                return res(vecOf({lo, hi}, vpir::kF32.withWidth(2)));
            }
            case GLSL_PackUnorm4x8: case GLSL_PackSnorm4x8: case GLSL_PackUnorm2x16: case GLSL_PackSnorm2x16: {
                bool snorm = e == GLSL_PackSnorm4x8 || e == GLSL_PackSnorm2x16;
                bool four = e == GLSL_PackUnorm4x8 || e == GLSL_PackSnorm4x8;
                uint32_t bitsN = four ? 8 : 16, n = four ? 4 : 2;
                float scale = snorm ? float((1u << (bitsN - 1)) - 1) : float((1u << bitsN) - 1);
                ValueId v = a(0);
                ValueId acc = u32(0);
                for (uint32_t i = 0; i < n; ++i) {
                    ValueId c = ext(v, i);
                    c = snorm ? tri(Op::Clamp, c, f32(-1), f32(1), vpir::kF32) : un(Op::Saturate, c, vpir::kF32);
                    c = un(Op::Round, bin(Op::Mul, c, f32(scale), vpir::kF32), vpir::kF32);
                    ValueId ci = snorm ? bitcast(un(Op::Convert, c, vpir::kI32), vpir::kU32) : un(Op::Convert, c, vpir::kU32);
                    ci = bin(Op::And, ci, u32((1u << bitsN) - 1), vpir::kU32);
                    acc = bin(Op::Or, acc, bin(Op::Shl, ci, u32(i * bitsN), vpir::kU32), vpir::kU32);
                }
                return res(acc);
            }
            case GLSL_UnpackUnorm4x8: case GLSL_UnpackSnorm4x8: case GLSL_UnpackUnorm2x16: case GLSL_UnpackSnorm2x16: {
                bool snorm = e == GLSL_UnpackSnorm4x8 || e == GLSL_UnpackSnorm2x16;
                bool four = e == GLSL_UnpackUnorm4x8 || e == GLSL_UnpackSnorm4x8;
                uint32_t bitsN = four ? 8 : 16, n = four ? 4 : 2;
                float scale = snorm ? float((1u << (bitsN - 1)) - 1) : float((1u << bitsN) - 1);
                ValueId x = ai(0, Scalar::U32);
                std::vector<ValueId> comps;
                for (uint32_t i = 0; i < n; ++i) {
                    Type st = snorm ? vpir::kI32 : vpir::kU32;
                    ValueId f = b_->ternary(Op::BitfieldExtract, castBits(x, st), u32(i * bitsN), u32(bitsN), st);
                    ValueId c = bin(Op::Mul, un(Op::Convert, f, vpir::kF32), f32(1.f / scale), vpir::kF32);
                    if (snorm) c = bin(Op::Max, c, f32(-1), vpir::kF32);
                    comps.push_back(c);
                }
                return res(vecOf(comps, vpir::kF32.withWidth(uint8_t(n))));
            }
            case GLSL_Length: {
                ValueId x = a(0);
                if (typeOfValue(x).width == 1) return res(un(Op::Abs, x, t));
                return res(un(Op::Length, x, t));
            }
            case GLSL_Distance: {
                Type xt = typeOfValue(a(0));
                ValueId d = bin(Op::Sub, a(0), a(1), xt);
                if (xt.width == 1) return res(un(Op::Abs, d, t));
                return res(un(Op::Length, d, t));
            }
            case GLSL_Cross: {
                ValueId x = a(0), y = a(1);
                auto c = [&](uint32_t i) { return ext(x, i); };
                auto d = [&](uint32_t i) { return ext(y, i); };
                auto m = [&](ValueId p, ValueId q) { return bin(Op::Mul, p, q, vpir::kF32); };
                ValueId r0 = bin(Op::Sub, m(c(1), d(2)), m(c(2), d(1)), vpir::kF32);
                ValueId r1 = bin(Op::Sub, m(c(2), d(0)), m(c(0), d(2)), vpir::kF32);
                ValueId r2 = bin(Op::Sub, m(c(0), d(1)), m(c(1), d(0)), vpir::kF32);
                return res(vecOf({r0, r1, r2}, t));
            }
            case GLSL_Normalize: {
                if (t.width == 1) return res(select(bin(Op::Lt, a(0), sv(0), vpir::kBool), sv(-1), sv(1)));
                return res(un(Op::Normalize, a(0), t));
            }
            case GLSL_FaceForward: {
                ValueId d = dot(a(2), a(1));
                return res(select(bin(Op::Lt, d, f32(0), vpir::kBool), a(0), un(Op::Neg, a(0), t)));
            }
            case GLSL_Reflect: {
                ValueId i0 = a(0), n = a(1);
                ValueId d = dot(n, i0);
                ValueId k = splatValue(bin(Op::Mul, d, f32(2), vpir::kF32), t.width);
                return res(bin(Op::Sub, i0, bin(Op::Mul, k, n, t), t));
            }
            case GLSL_Refract: {
                ValueId i0 = a(0), n = a(1), eta = a(2);
                ValueId d = dot(n, i0);
                ValueId k = bin(Op::Sub, f32(1), bin(Op::Mul, bin(Op::Mul, eta, eta, vpir::kF32),
                                                    bin(Op::Sub, f32(1), bin(Op::Mul, d, d, vpir::kF32), vpir::kF32), vpir::kF32), vpir::kF32);
                ValueId s = bin(Op::Add, bin(Op::Mul, eta, d, vpir::kF32), un(Op::Sqrt, k, vpir::kF32), vpir::kF32);
                ValueId r = bin(Op::Sub, bin(Op::Mul, splatValue(eta, t.width), i0, t), bin(Op::Mul, splatValue(s, t.width), n, t), t);
                return res(select(bin(Op::Lt, k, f32(0), vpir::kBool), zeroOf(t), r));
            }
            case GLSL_FindILsb: {
                ValueId x = ai(0, Scalar::U32);
                Type bt(Scalar::Bool, t.width);
                ValueId r = un(Op::FindLsb, x, ut);
                return res(select(bin(Op::Eq, x, splat(ut, 0), bt), splat(ut, 0xFFFFFFFFu), r));
            }
            case GLSL_FindUMsb: return res(un(Op::FindMsb, ai(0, Scalar::U32), ut));
            case GLSL_FindSMsb: {
                // Highest bit that differs from the sign bit.
                ValueId x = ai(0, Scalar::I32);
                Type bt(Scalar::Bool, t.width);
                ValueId neg = bin(Op::Lt, x, splat(it, 0), bt);
                ValueId v = select(neg, un(Op::Not, x, it), x);
                return res(castBits(un(Op::FindMsb, castBits(v, ut), ut), t));
            }
            case GLSL_InterpolateAtCentroid:
            case GLSL_InterpolateAtSample:
            case GLSL_InterpolateAtOffset: {
                RPtr p = resolve(w(in, 5), false);
                if (p.kind != RPtr::Input) fail("interpolation of a non-input");
                const IoLeaf& leaf = io_.at(p.global).leaves.at(p.leaf);
                if (leaf.kind != IoLeaf::Io) fail("interpolation of a builtin");
                Inst ii;
                ii.op = e == GLSL_InterpolateAtCentroid ? Op::InterpCentroid : e == GLSL_InterpolateAtSample ? Op::InterpSample : Op::InterpOffset;
                ii.sym = uint32_t(leaf.index);
                ii.type = out_.inputs[leaf.index].type;
                if (e == GLSL_InterpolateAtSample) ii.args = {castBits(a(1), vpir::kU32)};
                if (e == GLSL_InterpolateAtOffset) ii.args = {a(1)};
                ValueId v = b_->emit(ii);
                if (p.element) v = ext(v, p.element - 1);
                return res(v);
            }
            default: fail("unsupported GLSL.std.450 instruction " + std::to_string(e));
        }
    }
    ValueId expE(ValueId x, Type t) { return un(Op::Exp2, bin(Op::Mul, x, splat(t, floatBits(1.4426950408889634f)), t), t); }
    ValueId logE(ValueId x, Type t) { return bin(Op::Mul, un(Op::Log2, x, t), splat(t, floatBits(0.6931471805599453f)), t); }
    // Approximations with |error| < 1e-5 rad over the whole range (minimax polynomials).
    ValueId atanApprox(ValueId x, Type t) {
        auto sv = [&](float f) { return splat(t, floatBits(f)); };
        Type bt(Scalar::Bool, t.width);
        ValueId ax = un(Op::Abs, x, t);
        ValueId big = bin(Op::Gt, ax, sv(1), bt);
        ValueId z = select(big, un(Op::Rcp, ax, t), ax);
        ValueId z2 = bin(Op::Mul, z, z, t);
        // atan(z) for z in [0,1]
        ValueId p = sv(-0.0040540580f);
        p = tri(Op::Fma, p, z2, sv(0.0218612288f), t);
        p = tri(Op::Fma, p, z2, sv(-0.0559098861f), t);
        p = tri(Op::Fma, p, z2, sv(0.0964200441f), t);
        p = tri(Op::Fma, p, z2, sv(-0.1390853351f), t);
        p = tri(Op::Fma, p, z2, sv(0.1994653599f), t);
        p = tri(Op::Fma, p, z2, sv(-0.3332985605f), t);
        p = tri(Op::Fma, p, z2, sv(0.9999993329f), t);
        ValueId r = bin(Op::Mul, p, z, t);
        r = select(big, bin(Op::Sub, sv(1.5707963267948966f), r, t), r);
        return select(bin(Op::Lt, x, sv(0), bt), un(Op::Neg, r, t), r);
    }
    ValueId atan2Approx(ValueId y, ValueId x, Type t) {
        auto sv = [&](float f) { return splat(t, floatBits(f)); };
        Type bt(Scalar::Bool, t.width);
        ValueId base = atanApprox(bin(Op::Div, y, x, t), t);
        ValueId xneg = bin(Op::Lt, x, sv(0), bt);
        ValueId yneg = bin(Op::Lt, y, sv(0), bt);
        ValueId adj = select(yneg, bin(Op::Sub, base, sv(3.141592653589793f), t), bin(Op::Add, base, sv(3.141592653589793f), t));
        ValueId r = select(xneg, adj, base);
        ValueId xzero = bin(Op::Eq, x, sv(0), bt);
        ValueId vert = select(yneg, sv(-1.5707963267948966f), sv(1.5707963267948966f));
        return select(xzero, vert, r);
    }
    ValueId asinApprox(ValueId x, Type t) {
        auto sv = [&](float f) { return splat(t, floatBits(f)); };
        // asin(x) = atan2(x, sqrt(1 - x^2))
        ValueId c = un(Op::Sqrt, un(Op::Saturate, bin(Op::Sub, sv(1), bin(Op::Mul, x, x, t), t), t), t);
        return atan2Approx(x, c, t);
    }
    ValueId det2(ValueId a, ValueId b, ValueId c, ValueId d) {
        return bin(Op::Sub, bin(Op::Mul, a, d, vpir::kF32), bin(Op::Mul, b, c, vpir::kF32), vpir::kF32);
    }
    ValueId determinant(const Val& m, uint32_t typeId) {
        uint32_t n = T(typeId).count;
        auto e = [&](uint32_t c, uint32_t r) { return ext(m.v[c], r); };
        if (n == 2) return det2(e(0, 0), e(1, 0), e(0, 1), e(1, 1));
        if (n == 3) {
            ValueId c0 = bin(Op::Mul, e(0, 0), det2(e(1, 1), e(2, 1), e(1, 2), e(2, 2)), vpir::kF32);
            ValueId c1 = bin(Op::Mul, e(1, 0), det2(e(0, 1), e(2, 1), e(0, 2), e(2, 2)), vpir::kF32);
            ValueId c2 = bin(Op::Mul, e(2, 0), det2(e(0, 1), e(1, 1), e(0, 2), e(1, 2)), vpir::kF32);
            return bin(Op::Add, bin(Op::Sub, c0, c1, vpir::kF32), c2, vpir::kF32);
        }
        // 4x4: Laplace expansion along the first row.
        ValueId acc = kNoValue;
        for (uint32_t c = 0; c < 4; ++c) {
            uint32_t cols[3], k = 0;
            for (uint32_t i = 0; i < 4; ++i)
                if (i != c) cols[k++] = i;
            auto s = [&](uint32_t i, uint32_t r) { return e(cols[i], r); };
            ValueId minor = bin(Op::Add,
                                bin(Op::Sub, bin(Op::Mul, s(0, 1), det2(s(1, 2), s(2, 2), s(1, 3), s(2, 3)), vpir::kF32),
                                    bin(Op::Mul, s(1, 1), det2(s(0, 2), s(2, 2), s(0, 3), s(2, 3)), vpir::kF32), vpir::kF32),
                                bin(Op::Mul, s(2, 1), det2(s(0, 2), s(1, 2), s(0, 3), s(1, 3)), vpir::kF32), vpir::kF32);
            ValueId term = bin(Op::Mul, e(c, 0), minor, vpir::kF32);
            if (c & 1) term = un(Op::Neg, term, vpir::kF32);
            acc = acc == kNoValue ? term : bin(Op::Add, acc, term, vpir::kF32);
        }
        return acc;
    }
    Val inverse(const Val& m, uint32_t typeId) {
        uint32_t n = T(typeId).count;
        Type col = vpir::kF32.withWidth(uint8_t(n));
        auto e = [&](uint32_t c, uint32_t r) { return ext(m.v[c], r); };
        ValueId invDet = un(Op::Rcp, determinant(m, typeId), vpir::kF32);
        // Adjugate by cofactors: inv[c][r] = cof(r, c) / det.
        auto cofactor = [&](uint32_t row, uint32_t colIdx) -> ValueId {
            std::vector<uint32_t> rs, cs;
            for (uint32_t i = 0; i < n; ++i) {
                if (i != row) rs.push_back(i);
                if (i != colIdx) cs.push_back(i);
            }
            ValueId minor;
            if (n == 2) minor = e(cs[0], rs[0]);
            else if (n == 3) minor = det2(e(cs[0], rs[0]), e(cs[1], rs[0]), e(cs[0], rs[1]), e(cs[1], rs[1]));
            else {
                auto s = [&](uint32_t i, uint32_t r) { return e(cs[i], rs[r]); };
                minor = bin(Op::Add,
                            bin(Op::Sub, bin(Op::Mul, s(0, 0), det2(s(1, 1), s(2, 1), s(1, 2), s(2, 2)), vpir::kF32),
                                bin(Op::Mul, s(1, 0), det2(s(0, 1), s(2, 1), s(0, 2), s(2, 2)), vpir::kF32), vpir::kF32),
                            bin(Op::Mul, s(2, 0), det2(s(0, 1), s(1, 1), s(0, 2), s(1, 2)), vpir::kF32), vpir::kF32);
            }
            if ((row + colIdx) & 1) minor = un(Op::Neg, minor, vpir::kF32);
            return minor;
        };
        Val r;
        for (uint32_t c = 0; c < n; ++c) {
            std::vector<ValueId> comps;
            for (uint32_t rr = 0; rr < n; ++rr) comps.push_back(bin(Op::Mul, cofactor(c, rr), invDet, vpir::kF32));
            r.v.push_back(vecOf(comps, col));
        }
        return r;
    }

    // Where the lowering is, for error messages: the SPIR-V instruction being lowered (opcode,
    // result id, function, block and its first words), or the module-level phase.
    const Instr* cur_ = nullptr;
    const char* phase_ = "setup";

public:
    void lowerAll() { run(); }
    std::string where() const {
        std::string s = std::string("phase=") + phase_;
        if (!cur_ || !cur_->count || cur_->at + cur_->count > m_.words.size()) return s;
        const Instr& in = *cur_;
        s += " op=" + std::to_string(in.op) + " words=" + std::to_string(in.count);
        if (in.count >= 3 && in.op != OpStore && m_.resultType.count(m_.w(in, 2)) && m_.resultType.at(m_.w(in, 2)) == m_.w(in, 1))
            s += " result=" + m_.nameOf(m_.w(in, 2)) + " type=%" + std::to_string(m_.w(in, 1));
        if (scope_ && scope_->fn) s += " fn=" + m_.nameOf(scope_->fn->id) + " block=%" + std::to_string(scope_->block);
        s += " [";
        for (uint32_t i = 1; i < in.count && i <= 8; ++i) s += (i > 1 ? " " : "") + std::to_string(m_.w(in, i));
        if (in.count > 9) s += " ...";
        s += "]";
        return s;
    }
    friend struct RunAccess;

    void prepareSpec() {
        for (auto& [sid, bits] : o_.specConstants) specValue_[sid] = bits;
    }
    void scanDref() {
        // Images used with depth-compare ops are depth textures even when declared Depth = 0.
        std::unordered_map<uint32_t, uint32_t> origin;  // value id -> global variable
        for (auto& [fid, f] : m_.functions)
            for (const Block& bl : f.blocks)
                for (const Instr& in : bl.body) {
                    if (in.op == OpLoad && in.count >= 4 && m_.globals.count(m_.w(in, 3))) origin[m_.w(in, 2)] = m_.w(in, 3);
                    if (in.op == OpLoad && in.count >= 4) {
                        // through an access chain
                    }
                    if ((in.op == OpSampledImage || in.op == OpCopyObject) && in.count >= 4)
                        if (origin.count(m_.w(in, 3))) origin[m_.w(in, 2)] = origin[m_.w(in, 3)];
                    if (in.op == OpAccessChain && in.count >= 4 && m_.globals.count(m_.w(in, 3))) origin[m_.w(in, 2)] = m_.w(in, 3);
                    if (in.op == OpLoad && in.count >= 4 && origin.count(m_.w(in, 3))) origin[m_.w(in, 2)] = origin[m_.w(in, 3)];
                    bool dref = in.op == OpImageSampleDrefImplicitLod || in.op == OpImageSampleDrefExplicitLod ||
                                in.op == OpImageSampleProjDrefImplicitLod || in.op == OpImageSampleProjDrefExplicitLod ||
                                in.op == OpImageDrefGather;
                    if (dref && in.count >= 4 && origin.count(m_.w(in, 3))) drefGlobals_.insert(origin[m_.w(in, 3)]);
                }
    }
};

void Lowering::run() {
    for (const EntryPoint& e : m_.entries) {
        vpir::Stage st;
        switch (e.model) {
            case Model::Vertex: st = vpir::Stage::Vertex; break;
            case Model::Fragment: st = vpir::Stage::Fragment; break;
            case Model::GLCompute: st = vpir::Stage::Compute; break;
            default: continue;
        }
        if (st == o_.stage && (o_.entry.empty() || e.name == o_.entry)) {
            entry_ = &e;
            stage_ = st;
            break;
        }
    }
    if (!entry_) fail("no " + std::string(vpir::toString(o_.stage)) + " entry point named '" + o_.entry + "'");
    phase_ = "workgroup_size";
    prepareSpec();
    scanDref();
    out_ = vpir::Module{};
    out_.stage = stage_;
    out_.waveSize = 32;

    // Workgroup size: LocalSize / LocalSizeId modes, or a WorkgroupSize builtin constant.
    for (const Instr& md : entry_->modes) {
        Mode mode = Mode(m_.w(md, 2));
        if (mode == Mode::LocalSize && md.count >= 6)
            for (int i = 0; i < 3; ++i) out_.workgroupSize[i] = m_.w(md, 3 + i);
        if (mode == Mode::LocalSizeId && md.count >= 6)
            for (int i = 0; i < 3; ++i) out_.workgroupSize[i] = uint32_t(constScalar(m_.w(md, 3 + i)));
    }
    for (auto& [id, c] : m_.constants) {
        auto d = m_.deco(id);
        if (d && d->has(Deco::BuiltIn) && BuiltIn(d->get(Deco::BuiltIn)) == BuiltIn::WorkgroupSize && c.composite)
            for (int i = 0; i < 3 && i < int(c.parts.size()); ++i) out_.workgroupSize[i] = uint32_t(constScalar(c.parts[i]));
    }
    iface_.localSize = {out_.workgroupSize[0], out_.workgroupSize[1], out_.workgroupSize[2]};

    phase_ = "globals";
    b_ = std::make_unique<FunctionBuilder>(out_, "main");
    // Module-scope memory.
    for (uint32_t id : m_.globalOrder) {
        const GlobalVar& g = m_.globals.at(id);
        uint32_t pointee = T(g.type).elem;
        if (g.storage == Storage::Private) {
            privateGlobal_[id] = newPrivateSlot(naturalSize(pointee));
        } else if (g.storage == Storage::Workgroup) {
            sharedOffset_[id] = out_.sharedBytes;
            out_.sharedBytes += naturalSize(pointee);
        }
    }
    phase_ = "interface";
    declareIo();
    {
        // The exit-code variable, zeroed once.
        b_->storeVar(exitVar(), u32(0));
    }
    auto fit = m_.functions.find(entry_->function);
    if (fit == m_.functions.end()) fail("entry function missing");
    phase_ = "body";
    scope_ = newScope(fit->second);
    // Private globals with initializers.
    for (uint32_t id : m_.globalOrder) {
        const GlobalVar& g = m_.globals.at(id);
        if (g.storage == Storage::Private && g.initializer) {
            RPtr p;
            p.target = privateGlobal_[id], p.off = 0, p.type = T(g.type).elem;
            size_t at = 0;
            Val v = get(g.initializer);
            storeMem(p, v, at);
        }
    }
    flow(fit->second.blocks.front().label, 0);
    phase_ = "finish";
    cur_ = nullptr;
    b_->finish();

    // Buffers never stored to are read-only; storage textures keep the access their decorations
    // gave.
    for (size_t i = 0; i < out_.resources.size(); ++i) {
        auto& r = out_.resources[i];
        if (r.kind == vpir::ResourceKind::Buffer && !written_.count(int(i))) r.access = vpir::Access::Read;
    }
    // Stores through any path (storage images, texel buffers, image atomics, pointer parameters)
    // mark their binding written, so the frame compiler orders later reads after them.
    for (int r : written_) {
        auto rb = resBinding_.find(r);
        if (rb == resBinding_.end()) continue;
        for (auto& b : iface_.bindings)
            if (b.set == rb->second.first && b.binding == rb->second.second) b.written = true;
    }
    // The bindings the driver handed in for vertex buffers are reported as used.
    if (stage_ == vpir::Stage::Vertex)
        for (auto& vb : o_.vertexBuffers) {
            SpirvBinding sb;
            sb.set = kVertexBufferSet, sb.binding = vb.binding, sb.kind = vpgpu::BindKind::Buffer, sb.slot = vb.slot;
            bool used = false;
            for (auto& a : o_.vertexAttributes)
                if (a.binding == vb.binding) used = true;
            if (used) iface_.bindings.push_back(sb);
        }
}

}  // namespace

bool lower(const SpirvModule& m, const SpirvLowerOptions& options, vpir::Module& out, SpirvInterface& iface,
           std::string& error) {
    iface = SpirvInterface{};
    // Every failure names where it happened (see Lowering::where), so one log line is enough to
    // find the instruction in the game's SPIR-V.
    Lowering l(m, options, out, iface);
    try {
        l.lowerAll();
    } catch (const LowerError& e) {
        error = std::string(e.what()) + " at " + l.where();
        out = vpir::Module{};
        return false;
    } catch (const std::exception& e) {
        error = std::string("internal error: ") + e.what() + " at " + l.where();
        out = vpir::Module{};
        return false;
    }
    return true;
}

}  // namespace spirv
