// AST -> SPIR-V 1.3 for one stage of a linked program. The output is shaped for gpu/shader/spirv
// (lower.cpp): structured control flow, Function-storage pointer parameters, one OpVariable per
// GLSL interface variable, explicit std140 layout on uniform blocks, combined image samplers.
//
// Translation choices:
//  * User functions keep their names; the entry point `main` is a wrapper that runs global
//    initialisers, calls the user's main and (vertex) remaps clip z from GL [-w, w] to [0, w].
//  * Every parameter is a Function pointer: the caller copies `in`/`inout` arguments into a
//    temporary, calls, and copies `out`/`inout` back (GLSL's copy-in/copy-out). Samplers are passed
//    as UniformConstant pointers.
//  * `&&`, `||` and `?:` branch (GLSL evaluates one side only).
//  * `switch` becomes an if-chain inside a once-loop with a `matched` flag (fall-through and
//    `default` anywhere); a `continue` inside it is carried out of the once-loop by a flag.
//  * Statements after break/continue/return/discard in the same block are dead and not emitted.
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>

#include "builtins.h"
#include "emit.h"
#include "../../shader/spirv/src/spv_enums.h"

namespace vpglsl {

using namespace spirv::spv;
using SC = spirv::spv::Storage;

namespace {

struct EmitError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
[[noreturn]] void fail(const std::string& m) { throw EmitError(m); }

uint32_t floatBits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}

class Emitter {
public:
    Emitter(const Shader& sh, const ProgramLayout& layout) : sh_(sh), L_(layout) {}

    std::vector<uint32_t> run() {
        glsl_ = id();
        voidT_ = typeId(Type::make(Base::Void));
        boolT_ = typeId(Type::make(Base::Bool));
        declareGlobals();
        // Function ids first (calls may precede definitions).
        for (Function* f : sh_.functions)
            if (f->body) fnIds_[f] = id();
        for (Function* f : sh_.functions)
            if (f->body) function(*f);
        entryWrapper();
        return assemble();
    }

private:
    // ---- module sections ------------------------------------------------------------------------
    std::vector<uint32_t> caps_, names_, decos_, types_, code_;
    uint32_t next_ = 1;
    uint32_t id() { return next_++; }

    static void inst(std::vector<uint32_t>& out, uint16_t op, std::initializer_list<uint32_t> w) {
        out.push_back(uint32_t(w.size() + 1) << 16 | op);
        out.insert(out.end(), w.begin(), w.end());
    }
    static void instV(std::vector<uint32_t>& out, uint16_t op, const std::vector<uint32_t>& w) {
        out.push_back(uint32_t(w.size() + 1) << 16 | op);
        out.insert(out.end(), w.begin(), w.end());
    }
    static std::vector<uint32_t> str(const std::string& s) {
        std::vector<uint32_t> w((s.size() + 4) / 4, 0);
        std::memcpy(w.data(), s.data(), s.size());
        return w;
    }
    void name(uint32_t target, const std::string& n) {
        std::vector<uint32_t> w{target};
        auto s = str(n);
        w.insert(w.end(), s.begin(), s.end());
        instV(names_, OpName, w);
    }
    void deco(uint32_t target, Deco d, std::initializer_list<uint32_t> lits = {}) {
        std::vector<uint32_t> w{target, uint32_t(d)};
        w.insert(w.end(), lits.begin(), lits.end());
        instV(decos_, OpDecorate, w);
    }
    void memberDeco(uint32_t st, uint32_t m, Deco d, std::initializer_list<uint32_t> lits = {}) {
        std::vector<uint32_t> w{st, m, uint32_t(d)};
        w.insert(w.end(), lits.begin(), lits.end());
        instV(decos_, OpMemberDecorate, w);
    }

    // ---- types ----------------------------------------------------------------------------------
    std::map<std::string, uint32_t> typeCache_;
    uint32_t cached(const std::string& key, const std::function<uint32_t()>& make) {
        auto it = typeCache_.find(key);
        if (it != typeCache_.end()) return it->second;
        uint32_t r = make();
        typeCache_[key] = r;
        return r;
    }
    // `layout`: 0 = none (Function/Private/IO), 1 = std140 (decorated copies of arrays/structs).
    uint32_t typeId(const Type& t, int layout = 0, bool rowMajor = false) {
        if (!t.isArray() && t.base != Base::Struct) layout = 0, rowMajor = false;  // leaf types are shared
        const std::string key = t.str() + "#" + std::to_string(layout) + (t.st ? "#" + std::to_string(t.st->index) : "") +
                                (rowMajor ? "R" : "");
        return cached(key, [&]() -> uint32_t {
            if (t.isArray()) {
                uint32_t el = typeId(t.element(), layout, rowMajor);
                uint32_t len = constU(uint32_t(t.dims[0]));
                uint32_t r = id();
                inst(types_, OpTypeArray, {r, el, len});
                if (layout) deco(r, Deco::ArrayStride, {std140ArrayStride(t, rowMajor)});
                return r;
            }
            switch (t.base) {
                case Base::Void: { uint32_t r = id(); inst(types_, OpTypeVoid, {r}); return r; }
                case Base::Bool:
                case Base::Int:
                case Base::Uint:
                case Base::Float: {
                    if (t.cols) {
                        uint32_t col = typeId(t.columnType());
                        uint32_t r = id();
                        inst(types_, OpTypeMatrix, {r, col, t.cols});
                        return r;
                    }
                    if (t.vec > 1) {
                        uint32_t s = typeId(t.scalarType());
                        uint32_t r = id();
                        inst(types_, OpTypeVector, {r, s, t.vec});
                        return r;
                    }
                    uint32_t r = id();
                    if (t.base == Base::Bool) inst(types_, OpTypeBool, {r});
                    else if (t.base == Base::Float) inst(types_, OpTypeFloat, {r, 32});
                    else inst(types_, OpTypeInt, {r, 32, t.base == Base::Int ? 1u : 0u});
                    return r;
                }
                case Base::Sampler: {
                    uint32_t img = imageType(t);
                    uint32_t r = id();
                    inst(types_, OpTypeSampledImage, {r, img});
                    return r;
                }
                case Base::Struct: {
                    std::vector<uint32_t> w;
                    for (const Member& m : t.st->members) w.push_back(typeId(m.type, layout, m.rowMajor));
                    uint32_t r = id();
                    w.insert(w.begin(), r);
                    instV(types_, OpTypeStruct, w);
                    name(r, t.st->name);
                    if (layout) {
                        uint32_t size = 0;
                        auto off = std140Offsets(t.st->members, &size);
                        for (size_t i = 0; i < off.size(); ++i) layoutMember(r, uint32_t(i), t.st->members[i], off[i]);
                    }
                    return r;
                }
            }
            fail("unsupported type " + t.str());
        });
    }
    void layoutMember(uint32_t st, uint32_t i, const Member& m, uint32_t offset) {
        memberDeco(st, i, Deco::Offset, {offset});
        Type inner = m.type;
        while (inner.isArray()) inner = inner.element();
        if (inner.cols) {
            memberDeco(st, i, m.rowMajor ? Deco::RowMajor : Deco::ColMajor);
            memberDeco(st, i, Deco::MatrixStride, {16});
        }
    }
    uint32_t imageType(const Type& t) {
        const std::string key = "img" + std::to_string(int(t.dim)) + std::to_string(int(t.sampled)) +
                                std::to_string(t.shadow) + std::to_string(t.ms);
        return cached(key, [&] {
            uint32_t st = typeId(Type::make(t.sampled));
            uint32_t dim = t.dim == SDim::D3 ? uint32_t(Dim::Dim3D) : t.dim == SDim::Cube ? uint32_t(Dim::Cube) : uint32_t(Dim::Dim2D);
            uint32_t r = id();
            inst(types_, OpTypeImage, {r, st, dim, t.shadow ? 1u : 0u, t.dim == SDim::D2Array ? 1u : 0u, t.ms ? 1u : 0u, 1u, 0u});
            return r;
        });
    }
    uint32_t imageOf(const Type& sampler) { return imageType(sampler); }
    uint32_t ptrType(SC st, uint32_t pointee) {
        return cached("ptr" + std::to_string(uint32_t(st)) + "_" + std::to_string(pointee), [&] {
            uint32_t r = id();
            inst(types_, OpTypePointer, {r, uint32_t(st), pointee});
            return r;
        });
    }

    // ---- constants ------------------------------------------------------------------------------
    uint32_t constBits(const Type& scalar, uint32_t bits) {
        return cached("c" + scalar.str() + "_" + std::to_string(bits), [&] {
            uint32_t t = typeId(scalar);
            uint32_t r = id();
            if (scalar.base == Base::Bool) inst(types_, bits ? OpConstantTrue : OpConstantFalse, {t, r});
            else inst(types_, OpConstant, {t, r, bits});
            return r;
        });
    }
    uint32_t constU(uint32_t v) { return constBits(Type::make(Base::Uint), v); }
    uint32_t constI(int32_t v) { return constBits(Type::make(Base::Int), uint32_t(v)); }
    uint32_t constF(float v) { return constBits(Type::make(Base::Float), floatBits(v)); }
    uint32_t constB(bool v) { return constBits(Type::make(Base::Bool), v ? 1 : 0); }
    // Scalar constant of a numeric/bool scalar type from a double/int64.
    uint32_t constOf(const Type& s, double f, int64_t i) {
        switch (s.base) {
            case Base::Float: return constF(float(f));
            case Base::Int: return constI(int32_t(i));
            case Base::Uint: return constU(uint32_t(i));
            default: return constB(i != 0);
        }
    }
    uint32_t one(const Type& s) { return constOf(s.scalarType(), 1.0, 1); }
    uint32_t zero(const Type& s) { return constOf(s.scalarType(), 0.0, 0); }

    // ---- globals --------------------------------------------------------------------------------
    struct VarInfo {
        uint32_t ptr = 0;
        SC storage = SC::Function;
        Type type;
    };
    std::map<const Symbol*, VarInfo> vars_;
    std::vector<uint32_t> interface_;
    uint32_t defaultBlockVar_ = 0, defaultBlockType_ = 0;
    std::map<std::string, uint32_t> defaultIndex_;  // uniform name -> member index
    std::map<const UniformBlockDecl*, uint32_t> blockVars_;
    std::map<const UniformBlockDecl*, Type> blockTypes_;
    uint32_t fragDataVar_ = 0;
    std::vector<std::pair<const Symbol*, const Expr*>> initialisers_;
    uint32_t positionVar_ = 0;

    uint32_t globalVar(uint32_t type, SC st, const std::string& n) {
        uint32_t p = ptrType(st, type);
        uint32_t r = id();
        inst(types_, OpVariable, {p, r, uint32_t(st)});
        if (!n.empty()) name(r, n);
        if (st == SC::Input || st == SC::Output) interface_.push_back(r);
        return r;
    }

    BuiltIn spvBuiltin(BuiltinVar b) {
        switch (b) {
            case BuiltinVar::Position: return BuiltIn::Position;
            case BuiltinVar::PointSize: return BuiltIn::PointSize;
            case BuiltinVar::VertexID: return BuiltIn::VertexIndex;
            case BuiltinVar::InstanceID: return BuiltIn::InstanceIndex;
            case BuiltinVar::FragCoord: return BuiltIn::FragCoord;
            case BuiltinVar::FrontFacing: return BuiltIn::FrontFacing;
            case BuiltinVar::PointCoord: return BuiltIn::PointCoord;
            case BuiltinVar::FragDepth: return BuiltIn::FragDepth;
            case BuiltinVar::ViewID: return BuiltIn::ViewIndex;
            case BuiltinVar::LocalInvocationID: return BuiltIn::LocalInvocationId;
            case BuiltinVar::GlobalInvocationID: return BuiltIn::GlobalInvocationId;
            case BuiltinVar::WorkGroupID: return BuiltIn::WorkgroupId;
            case BuiltinVar::NumWorkGroups: return BuiltIn::NumWorkgroups;
            case BuiltinVar::LocalInvocationIndex: return BuiltIn::LocalInvocationIndex;
            default: fail("unsupported builtin");
        }
    }

    void declareGlobals() {
        // Default uniform block (shared by both stages, identical layout).
        if (!L_.defaultBlock.empty()) {
            StructDef* sd = &defaultStruct_;
            sd->name = "vpgl_Default";
            sd->index = 1 << 20;
            for (size_t i = 0; i < L_.defaultBlock.size(); ++i) {
                sd->members.push_back({L_.defaultBlock[i].name, L_.defaultBlock[i].type, false});
                defaultIndex_[L_.defaultBlock[i].name] = uint32_t(i);
            }
            Type t;
            t.base = Base::Struct, t.st = sd;
            defaultBlockType_ = typeId(t, 1);
            deco(defaultBlockType_, Deco::Block);
            defaultBlockVar_ = globalVar(defaultBlockType_, SC::Uniform, "vpgl_DefaultUniforms");
            deco(defaultBlockVar_, Deco::DescriptorSet, {0});
            deco(defaultBlockVar_, Deco::Binding, {kDefaultBlockBinding});
        }
        for (const UniformBlockDecl* b : sh_.blocks) {
            auto bit = L_.blockBinding.find(b->name);
            if (bit == L_.blockBinding.end()) continue;  // unused, dropped by the linker
            auto sd = std::make_unique<StructDef>();
            sd->name = b->name;
            sd->members = b->members;
            sd->index = (1 << 21) + int(blockStructs_.size());
            Type t;
            t.base = Base::Struct, t.st = sd.get();
            uint32_t st = typeId(t, 1);
            deco(st, Deco::Block);
            Type vt = t;
            vt.dims = b->dims;
            blockTypes_[b] = vt;
            uint32_t ty = vt.isArray() ? typeId(vt, 1) : st;
            uint32_t v = globalVar(ty, SC::Uniform, b->instance.empty() ? b->name : b->instance);
            deco(v, Deco::DescriptorSet, {0});
            deco(v, Deco::Binding, {bit->second});
            blockVars_[b] = v;
            blockStructs_.push_back(std::move(sd));
        }
        for (const GlobalDecl& g : sh_.globals) {
            const Symbol* s = g.sym;
            VarInfo vi;
            vi.type = s->type;
            switch (s->storage) {
                case vpglsl::Storage::Uniform: {
                    if (s->type.containsSampler()) {
                        if (s->type.base == Base::Struct) fail("samplers inside structs are not supported: '" + s->name + "'");
                        auto b = L_.samplerBinding.find(s->name);
                        if (b == L_.samplerBinding.end()) continue;
                        vi.storage = SC::UniformConstant;
                        vi.ptr = globalVar(typeId(s->type), SC::UniformConstant, s->name);
                        deco(vi.ptr, Deco::DescriptorSet, {0});
                        deco(vi.ptr, Deco::Binding, {b->second});
                    } else {
                        // Member of the default block: resolved by name at use.
                        vi.storage = SC::Uniform;
                        vi.ptr = 0;
                    }
                    break;
                }
                case vpglsl::Storage::In: {
                    if (!s->used) continue;
                    vi.storage = SC::Input;
                    vi.ptr = globalVar(typeId(s->type), SC::Input, s->name);
                    int loc = -1;
                    if (sh_.stage == Stage::Vertex) {
                        auto a = L_.attribLocation.find(s->name);
                        if (a != L_.attribLocation.end()) loc = a->second;
                    } else {
                        auto a = L_.varyingLocation.find(s->name);
                        if (a != L_.varyingLocation.end()) loc = a->second;
                    }
                    if (loc < 0) fail("no location for input '" + s->name + "'");
                    deco(vi.ptr, Deco::Location, {uint32_t(loc)});
                    if (sh_.stage == Stage::Fragment) interpDeco(vi.ptr, *s);
                    break;
                }
                case vpglsl::Storage::Out: {
                    vi.storage = SC::Output;
                    vi.ptr = globalVar(typeId(s->type), SC::Output, s->name);
                    int loc = -1;
                    if (sh_.stage == Stage::Vertex) {
                        auto a = L_.varyingLocation.find(s->name);
                        if (a != L_.varyingLocation.end()) loc = a->second;
                        else {
                            // Not read by the fragment shader: keep it as a private variable.
                            vi.storage = SC::Private;
                            interface_.pop_back();
                            vi.ptr = 0;
                        }
                    } else {
                        auto a = L_.outputLocation.find(s->name);
                        if (a != L_.outputLocation.end()) loc = a->second;
                    }
                    if (vi.storage == SC::Private) {
                        vi.ptr = globalVar(typeId(s->type), SC::Private, s->name + "_unused");
                        break;
                    }
                    if (loc < 0) fail("no location for output '" + s->name + "'");
                    deco(vi.ptr, Deco::Location, {uint32_t(loc)});
                    if (sh_.stage == Stage::Vertex) interpDeco(vi.ptr, *s);
                    break;
                }
                case vpglsl::Storage::Shared:
                    vi.storage = SC::Workgroup;
                    vi.ptr = globalVar(typeId(s->type), SC::Workgroup, s->name);
                    break;
                default:
                    vi.storage = SC::Private;
                    vi.ptr = globalVar(typeId(s->type), SC::Private, s->name);
                    if (g.init) initialisers_.push_back({s, g.init.get()});
                    break;
            }
            vars_[s] = vi;
        }
        // Builtins (only those referenced; gl_Position always in a vertex shader).
        std::vector<const Symbol*> builtins(sh_.builtins.begin(), sh_.builtins.end());
        for (const Symbol* s : builtins) declareBuiltin(s);
        if (sh_.stage == Stage::Vertex && !positionVar_) {
            Symbol tmp;
            tmp.name = "gl_Position", tmp.type = Type::make(Base::Float, 4), tmp.builtin = BuiltinVar::Position;
            positionVar_ = globalVar(typeId(tmp.type), SC::Output, "gl_Position");
            deco(positionVar_, Deco::BuiltIn, {uint32_t(BuiltIn::Position)});
        }
    }

    void interpDeco(uint32_t v, const Symbol& s) {
        Type inner = s.type;
        while (inner.isArray()) inner = inner.element();
        if (s.interp == Interp::Flat || inner.isIntegral()) deco(v, Deco::Flat);
        else if (s.interp == Interp::Centroid) deco(v, Deco::Centroid);
    }

    void declareBuiltin(const Symbol* s) {
        VarInfo vi;
        vi.type = s->type;
        switch (s->builtin) {
            case BuiltinVar::FragColor: {
                vi.storage = SC::Output;
                vi.ptr = globalVar(typeId(s->type), SC::Output, "gl_FragColor");
                deco(vi.ptr, Deco::Location, {0});
                break;
            }
            case BuiltinVar::FragData: {
                Type t = s->type;
                t.dims = {std::max(1, L_.fragDataCount)};
                vi.type = t;
                vi.storage = SC::Output;
                vi.ptr = globalVar(typeId(t), SC::Output, "gl_FragData");
                deco(vi.ptr, Deco::Location, {0});
                break;
            }
            case BuiltinVar::PointCoord: {
                // No VPIR builtin yet: (0.5, 0.5), the centre of the point (documented in bitacora).
                vi.storage = SC::Private;
                vi.ptr = globalVar(typeId(s->type), SC::Private, "gl_PointCoord");
                pointCoordVar_ = vi.ptr;
                break;
            }
            default: {
                const bool input = s->builtin != BuiltinVar::Position && s->builtin != BuiltinVar::PointSize &&
                                   s->builtin != BuiltinVar::FragDepth;
                Type t = s->type;
                // gl_VertexID/gl_InstanceID are int in GLSL, int in SPIR-V too.
                vi.storage = input ? SC::Input : SC::Output;
                vi.ptr = globalVar(typeId(t), vi.storage, s->name);
                deco(vi.ptr, Deco::BuiltIn, {uint32_t(spvBuiltin(s->builtin))});
                if (s->builtin == BuiltinVar::ViewID && sh_.stage == Stage::Fragment) deco(vi.ptr, Deco::Flat);
                if (s->builtin == BuiltinVar::Position) positionVar_ = vi.ptr;
                if (s->builtin == BuiltinVar::FragDepth) writesDepth_ = true;
                if (s->builtin == BuiltinVar::ViewID) usesMultiview_ = true;
            }
        }
        vars_[s] = vi;
    }

    // ---- functions ------------------------------------------------------------------------------
    std::map<const Function*, uint32_t> fnIds_;
    std::vector<uint32_t> fnLocals_;  // OpVariable instructions of the current function
    std::vector<uint32_t> body_;      // instructions after the entry label
    bool terminated_ = false;
    const Function* cur_ = nullptr;

    struct Loop {
        uint32_t cont, merge;
        bool isSwitch;
        uint32_t continueFlag;  // switch inside a loop: set by `continue`
    };
    std::vector<Loop> loops_;

    uint32_t localVar(uint32_t type, const std::string& n = {}) {
        uint32_t p = ptrType(SC::Function, type);
        uint32_t r = id();
        inst(fnLocals_, OpVariable, {p, r, uint32_t(SC::Function)});
        if (!n.empty()) name(r, n);
        return r;
    }
    void label(uint32_t l) {
        inst(body_, OpLabel, {l});
        terminated_ = false;
    }
    void branch(uint32_t to) {
        if (terminated_) return;
        inst(body_, OpBranch, {to});
        terminated_ = true;
    }
    uint32_t op(uint16_t o, uint32_t type, std::initializer_list<uint32_t> args) {
        uint32_t r = id();
        std::vector<uint32_t> w{type, r};
        w.insert(w.end(), args.begin(), args.end());
        instV(body_, o, w);
        return r;
    }
    uint32_t opV(uint16_t o, uint32_t type, const std::vector<uint32_t>& args) {
        uint32_t r = id();
        std::vector<uint32_t> w{type, r};
        w.insert(w.end(), args.begin(), args.end());
        instV(body_, o, w);
        return r;
    }
    uint32_t ext(uint32_t type, uint32_t e, std::initializer_list<uint32_t> args) {
        uint32_t r = id();
        std::vector<uint32_t> w{type, r, glsl_, e};
        w.insert(w.end(), args.begin(), args.end());
        instV(body_, OpExtInst, w);
        return r;
    }
    uint32_t load(const Type& t, uint32_t ptr) { return op(OpLoad, typeId(t), {ptr}); }
    void store(uint32_t ptr, uint32_t v) { inst(body_, OpStore, {ptr, v}); }

    uint32_t fnType(const Function& f) {
        std::vector<uint32_t> w{typeId(f.ret)};
        for (const Symbol* p : f.params)
            w.push_back(ptrType(p->type.isSampler() ? SC::UniformConstant : SC::Function, typeId(p->type)));
        std::string key = "fn";
        for (uint32_t x : w) key += "_" + std::to_string(x);
        return cached(key, [&] {
            uint32_t r = id();
            w.insert(w.begin(), r);
            instV(types_, OpTypeFunction, w);
            return r;
        });
    }

    void function(const Function& f) {
        cur_ = &f;
        fnLocals_.clear();
        body_.clear();
        std::vector<uint32_t> head;
        uint32_t fid = fnIds_.at(&f);
        name(fid, f.isMain() ? "vpgl_main" : f.name);
        inst(head, OpFunction, {typeId(f.ret), fid, 0, fnType(f)});
        for (const Symbol* p : f.params) {
            uint32_t pid = id();
            SC st = p->type.isSampler() ? SC::UniformConstant : SC::Function;
            inst(head, OpFunctionParameter, {ptrType(st, typeId(p->type)), pid});
            VarInfo vi;
            vi.ptr = pid, vi.storage = st, vi.type = p->type;
            vars_[p] = vi;
        }
        uint32_t entry = id();
        terminated_ = false;
        stmt(*f.body);
        if (!terminated_) {
            if (f.ret.base == Base::Void) inst(body_, OpReturn, {});
            else {
                // Falling off the end of a non-void function: undefined value (GLSL); return zero.
                inst(body_, OpReturnValue, {zeroValue(f.ret)});
            }
        }
        code_.insert(code_.end(), head.begin(), head.end());
        inst(code_, OpLabel, {entry});
        code_.insert(code_.end(), fnLocals_.begin(), fnLocals_.end());
        code_.insert(code_.end(), body_.begin(), body_.end());
        inst(code_, OpFunctionEnd, {});
    }

    uint32_t zeroValue(const Type& t) {
        if (t.isArray()) {
            std::vector<uint32_t> parts;
            for (int i = 0; i < t.dims[0]; ++i) parts.push_back(zeroValue(t.element()));
            return opV(OpCompositeConstruct, typeId(t), parts);
        }
        if (t.base == Base::Struct) {
            std::vector<uint32_t> parts;
            for (const Member& m : t.st->members) parts.push_back(zeroValue(m.type));
            return opV(OpCompositeConstruct, typeId(t), parts);
        }
        if (t.cols) {
            std::vector<uint32_t> parts(t.cols, zeroValue(t.columnType()));
            return opV(OpCompositeConstruct, typeId(t), parts);
        }
        if (t.vec > 1) return splat(zero(t), t);
        return zero(t);
    }

    void entryWrapper() {
        fnLocals_.clear();
        body_.clear();
        terminated_ = false;
        const Function* m = sh_.main();
        uint32_t fid = id();
        mainId_ = fid;
        name(fid, "main");
        std::vector<uint32_t> head;
        Function dummy;
        dummy.ret = Type::make(Base::Void);
        inst(head, OpFunction, {voidT_, fid, 0, fnType(dummy)});
        uint32_t entry = id();
        if (pointCoordVar_) store(pointCoordVar_, splat(constF(0.5f), Type::make(Base::Float, 2)));
        for (auto& [s, e] : initialisers_) store(vars_.at(s).ptr, value(*e));
        op(OpFunctionCall, voidT_, {fnIds_.at(m)});
        if (sh_.stage == Stage::Vertex && positionVar_) {
            // GL clip z in [-w, w] -> Vulkan [0, w]: z' = (z + w) / 2.
            Type v4 = Type::make(Base::Float, 4), f = Type::make(Base::Float);
            uint32_t p = load(v4, positionVar_);
            uint32_t z = op(OpCompositeExtract, typeId(f), {p, 2});
            uint32_t w = op(OpCompositeExtract, typeId(f), {p, 3});
            uint32_t s = op(OpFAdd, typeId(f), {z, w});
            uint32_t h = op(OpFMul, typeId(f), {s, constF(0.5f)});
            uint32_t r = op(OpCompositeInsert, typeId(v4), {h, p, 2});
            store(positionVar_, r);
        }
        inst(body_, OpReturn, {});
        code_.insert(code_.end(), head.begin(), head.end());
        inst(code_, OpLabel, {entry});
        code_.insert(code_.end(), fnLocals_.begin(), fnLocals_.end());
        code_.insert(code_.end(), body_.begin(), body_.end());
        inst(code_, OpFunctionEnd, {});
    }

    std::vector<uint32_t> assemble() {
        std::vector<uint32_t> out = {kMagic, 0x00010300u, 0x56504B00u /* generator: VPKEngine */, 0, 0};
        inst(out, OpCapability, {1 /* Shader */});
        if (usesMultiview_) inst(out, OpCapability, {4439 /* MultiView */});
        {
            std::vector<uint32_t> w{glsl_};
            auto s = str("GLSL.std.450");
            w.insert(w.end(), s.begin(), s.end());
            instV(out, OpExtInstImport, w);
        }
        inst(out, OpMemoryModel, {0 /* Logical */, 1 /* GLSL450 */});
        {
            Model model = sh_.stage == Stage::Vertex ? Model::Vertex : sh_.stage == Stage::Fragment ? Model::Fragment : Model::GLCompute;
            std::vector<uint32_t> w{uint32_t(model), mainId_};
            auto s = str("main");
            w.insert(w.end(), s.begin(), s.end());
            w.insert(w.end(), interface_.begin(), interface_.end());
            instV(out, OpEntryPoint, w);
        }
        if (sh_.stage == Stage::Fragment) {
            // GL keeps row 0 at the bottom; the GLES layer stores rows bottom-up, so upper-left in
            // memory terms is GL's lower-left in window terms (see glsl.h).
            inst(out, OpExecutionMode, {mainId_, uint32_t(Mode::OriginUpperLeft)});
            if (writesDepth_) inst(out, OpExecutionMode, {mainId_, uint32_t(Mode::DepthReplacing)});
            if (sh_.earlyFragmentTests) inst(out, OpExecutionMode, {mainId_, uint32_t(Mode::EarlyFragmentTests)});
        }
        if (sh_.stage == Stage::Compute)
            inst(out, OpExecutionMode, {mainId_, uint32_t(Mode::LocalSize), sh_.localSize[0], sh_.localSize[1], sh_.localSize[2]});
        out.insert(out.end(), names_.begin(), names_.end());
        out.insert(out.end(), decos_.begin(), decos_.end());
        out.insert(out.end(), types_.begin(), types_.end());
        out.insert(out.end(), code_.begin(), code_.end());
        out[3] = next_;
        return out;
    }

    // ---- statements -----------------------------------------------------------------------------
    void stmt(const Stmt& s) {
        if (terminated_) return;  // dead code after break/continue/return/discard
        switch (s.k) {
            case SK::Empty: return;
            case SK::Block:
                for (const StmtP& c : s.list) stmt(*c);
                return;
            case SK::Decl:
                for (const auto& [sym, init] : s.decls) {
                    uint32_t v = localVar(typeId(sym->type), sym->name);
                    VarInfo vi;
                    vi.ptr = v, vi.storage = SC::Function, vi.type = sym->type;
                    vars_[sym] = vi;
                    if (init) store(v, value(*init));
                }
                return;
            case SK::Expr: value(*s.e); return;
            case SK::If: {
                uint32_t c = value(*s.e);
                uint32_t t = id(), e = s.els ? id() : 0, m = id();
                inst(body_, OpSelectionMerge, {m, 0});
                inst(body_, OpBranchConditional, {c, t, e ? e : m});
                label(t);
                stmt(*s.then);
                branch(m);
                if (e) {
                    label(e);
                    stmt(*s.els);
                    branch(m);
                }
                label(m);
                return;
            }
            case SK::While:
            case SK::For: {
                if (s.k == SK::For && s.init) stmt(*s.init);
                uint32_t header = id(), condL = id(), bodyL = id(), cont = id(), merge = id();
                branch(header);
                label(header);
                inst(body_, OpLoopMerge, {merge, cont, 0});
                branch(condL);
                label(condL);
                if (s.e) {
                    uint32_t c = value(*s.e);
                    inst(body_, OpBranchConditional, {c, bodyL, merge});
                    terminated_ = true;
                } else {
                    branch(bodyL);
                }
                label(bodyL);
                loops_.push_back({cont, merge, false, 0});
                stmt(*s.body);
                loops_.pop_back();
                branch(cont);
                label(cont);
                if (s.inc) value(*s.inc);
                branch(header);
                label(merge);
                return;
            }
            case SK::Do: {
                uint32_t header = id(), bodyL = id(), cont = id(), merge = id();
                branch(header);
                label(header);
                inst(body_, OpLoopMerge, {merge, cont, 0});
                branch(bodyL);
                label(bodyL);
                loops_.push_back({cont, merge, false, 0});
                stmt(*s.body);
                loops_.pop_back();
                branch(cont);
                label(cont);
                uint32_t c = value(*s.e);
                inst(body_, OpBranchConditional, {c, header, merge});
                terminated_ = true;
                label(merge);
                return;
            }
            case SK::Return: {
                if (s.e) {
                    uint32_t v = value(*s.e);
                    inst(body_, OpReturnValue, {v});
                } else {
                    inst(body_, OpReturn, {});
                }
                terminated_ = true;
                return;
            }
            case SK::Discard:
                inst(body_, OpKill, {});
                terminated_ = true;
                return;
            case SK::Break: {
                if (loops_.empty()) fail("break outside a loop");
                branch(loops_.back().merge);
                return;
            }
            case SK::Continue: {
                // The innermost real loop; switches in between are left through their flag.
                size_t i = loops_.size();
                while (i-- > 0 && loops_[i].isSwitch) {}
                if (i == size_t(-1)) fail("continue outside a loop");
                if (i == loops_.size() - 1) {
                    branch(loops_.back().cont);
                    return;
                }
                // Inside switch(es): set every enclosing switch's flag and break out of the innermost.
                for (size_t k = i + 1; k < loops_.size(); ++k) store(loops_[k].continueFlag, constB(true));
                branch(loops_.back().merge);
                return;
            }
            case SK::Switch: return switchStmt(s);
            case SK::Case:
            case SK::Default: fail("case label outside a switch");
        }
    }

    void switchStmt(const Stmt& s) {
        const Type selT = s.e->type;
        uint32_t sel = value(*s.e);
        const Type b = Type::make(Base::Bool);
        // Sections: each label group followed by its statements.
        struct Section {
            std::vector<const Expr*> cases;
            bool isDefault = false;
            std::vector<const Stmt*> body;
        };
        std::vector<Section> secs;
        for (const StmtP& c : s.list) {
            if (c->k == SK::Case || c->k == SK::Default) {
                if (secs.empty() || !secs.back().body.empty()) secs.emplace_back();
                if (c->k == SK::Default) secs.back().isDefault = true;
                else secs.back().cases.push_back(c->e.get());
                continue;
            }
            if (secs.empty()) continue;  // statements before the first label never run
            secs.back().body.push_back(c.get());
        }
        auto eq = [&](const Expr* c) {
            uint32_t v = constOf(selT, c->constF, c->constI);
            return op(OpIEqual, typeId(b), {sel, v});
        };
        uint32_t anyMatch = constB(false);
        for (auto& sec : secs)
            for (const Expr* c : sec.cases) anyMatch = op(OpLogicalOr, typeId(b), {anyMatch, eq(c)});
        uint32_t matched = localVar(typeId(b), "vpgl_matched");
        store(matched, constB(false));
        uint32_t contFlag = 0;
        const bool inLoop = std::any_of(loops_.begin(), loops_.end(), [](const Loop& l) { return !l.isSwitch; });
        if (inLoop) {
            contFlag = localVar(typeId(b), "vpgl_continue");
            store(contFlag, constB(false));
        }
        // Once-loop.
        uint32_t header = id(), bodyL = id(), cont = id(), merge = id();
        branch(header);
        label(header);
        inst(body_, OpLoopMerge, {merge, cont, 0});
        branch(bodyL);
        label(bodyL);
        loops_.push_back({cont, merge, true, contFlag});
        for (auto& sec : secs) {
            if (terminated_) break;
            uint32_t m = load(b, matched);
            uint32_t c = constB(false);
            for (const Expr* e : sec.cases) c = op(OpLogicalOr, typeId(b), {c, eq(e)});
            if (sec.isDefault) c = op(OpLogicalOr, typeId(b), {c, op(OpLogicalNot, typeId(b), {anyMatch})});
            c = op(OpLogicalOr, typeId(b), {m, c});
            uint32_t t = id(), after = id();
            inst(body_, OpSelectionMerge, {after, 0});
            inst(body_, OpBranchConditional, {c, t, after});
            label(t);
            store(matched, constB(true));
            for (const Stmt* st : sec.body) stmt(*st);
            branch(after);
            label(after);
        }
        loops_.pop_back();
        branch(merge);
        label(cont);
        inst(body_, OpBranchConditional, {constB(false), header, merge});
        terminated_ = true;
        label(merge);
        if (contFlag) {
            // A `continue` inside the switch: continue the enclosing loop now.
            uint32_t f = load(b, contFlag);
            uint32_t t = id(), after = id();
            inst(body_, OpSelectionMerge, {after, 0});
            inst(body_, OpBranchConditional, {f, t, after});
            label(t);
            Stmt c;
            c.k = SK::Continue;
            stmt(c);
            branch(after);
            label(after);
        }
    }

    // ---- l-values -------------------------------------------------------------------------------
    struct Ptr {
        uint32_t ptr = 0;
        SC storage = SC::Function;
        Type type;
    };

    bool isDefaultUniform(const Symbol* s) const {
        return s->storage == vpglsl::Storage::Uniform && !s->type.containsSampler();
    }

    // Pointer to an addressable expression (Var/Index/Field chains over variables). Returns false
    // when the expression is not addressable (an r-value), so the caller spills it.
    bool pointer(const Expr& e, Ptr& out) {
        switch (e.k) {
            case EK::Var: {
                const Symbol* s = e.sym;
                if (s->storage == vpglsl::Storage::BlockInstance) {
                    out.ptr = blockVars_.at(s->block);
                    out.storage = SC::Uniform;
                    out.type = blockTypes_.at(s->block);
                    return true;
                }
                if (isDefaultUniform(s)) {
                    auto it = defaultIndex_.find(s->name);
                    if (it == defaultIndex_.end()) fail("uniform '" + s->name + "' missing from the default block");
                    out.storage = SC::Uniform;
                    out.type = s->type;
                    out.ptr = op(OpAccessChain, ptrType(SC::Uniform, typeId(s->type, 1)), {defaultBlockVar_, constU(it->second)});
                    return true;
                }
                auto it = vars_.find(s);
                if (it == vars_.end()) {
                    if (s->storage == vpglsl::Storage::Const && s->hasConst) return false;
                    fail("'" + s->name + "' has no storage");
                }
                out.ptr = it->second.ptr, out.storage = it->second.storage, out.type = it->second.type;
                return true;
            }
            case EK::Index: {
                const Type& bt = e.a[0]->type;
                // Vector components are not addressed (load + extract / insert instead).
                if (!bt.isArray() && !bt.cols) return false;
                Ptr base;
                if (!pointer(*e.a[0], base)) return false;
                uint32_t idx = value(*e.a[1]);
                out.type = e.type;
                if (bt.isArray() && base.type.isArray() && base.type.dims.size() > bt.dims.size()) out.type = e.type;
                out.storage = base.storage;
                out.ptr = op(OpAccessChain, ptrType(base.storage, typeId(e.type, layoutOf(base.storage))), {base.ptr, idx});
                return true;
            }
            case EK::Field: {
                Ptr base;
                if (!pointer(*e.a[0], base)) return false;
                out.type = e.type;
                out.storage = base.storage;
                out.ptr = op(OpAccessChain, ptrType(base.storage, typeId(e.type, layoutOf(base.storage), rowMajorOf(e))),
                             {base.ptr, constU(uint32_t(e.field))});
                return true;
            }
            default: return false;
        }
    }
    static int layoutOf(SC st) { return st == SC::Uniform ? 1 : 0; }
    bool rowMajorOf(const Expr& field) const {
        const Type& bt = field.a[0]->type;
        if (bt.base == Base::Struct && bt.st) return bt.st->members[size_t(field.field)].rowMajor;
        if (field.a[0]->k == EK::Var && field.a[0]->sym->block) return field.a[0]->sym->block->members[size_t(field.field)].rowMajor;
        if (field.a[0]->k == EK::Index && field.a[0]->a[0]->k == EK::Var && field.a[0]->a[0]->sym->block)
            return field.a[0]->a[0]->sym->block->members[size_t(field.field)].rowMajor;
        return false;
    }

    // Loads through a pointer whose pointee may carry the std140 layout: the loaded value is
    // rebuilt as the plain type (SPIR-V 1.3 has no OpCopyLogical-free way; lower.cpp reads
    // members/elements of layout types directly, so a load of the decorated type is fine as long
    // as the result type id matches the pointer's pointee).
    uint32_t loadPtr(const Ptr& p) {
        if (p.storage != SC::Uniform) return load(p.type, p.ptr);
        // Leaf types (scalars/vectors/matrices) are the same ids with or without layout.
        if (!p.type.isArray() && p.type.base != Base::Struct) return load(p.type, p.ptr);
        uint32_t laid = op(OpLoad, typeId(p.type, 1), {p.ptr});
        return relayout(laid, p.type);
    }
    // Rebuilds a value of a std140-decorated aggregate type as the undecorated type.
    uint32_t relayout(uint32_t v, const Type& t) {
        if (t.isArray()) {
            std::vector<uint32_t> parts;
            Type el = t.element();
            for (int i = 0; i < t.dims[0]; ++i) {
                uint32_t x = op(OpCompositeExtract, typeId(el, (el.isArray() || el.base == Base::Struct) ? 1 : 0), {v, uint32_t(i)});
                parts.push_back(el.isArray() || el.base == Base::Struct ? relayout(x, el) : x);
            }
            return opV(OpCompositeConstruct, typeId(t), parts);
        }
        if (t.base == Base::Struct) {
            std::vector<uint32_t> parts;
            for (size_t i = 0; i < t.st->members.size(); ++i) {
                const Type& mt = t.st->members[i].type;
                const bool agg = mt.isArray() || mt.base == Base::Struct;
                uint32_t x = op(OpCompositeExtract, typeId(mt, agg ? 1 : 0, t.st->members[i].rowMajor), {v, uint32_t(i)});
                parts.push_back(agg ? relayout(x, mt) : x);
            }
            return opV(OpCompositeConstruct, typeId(t), parts);
        }
        return v;
    }

    // Stores `v` (of e.type) into the l-value `e`.
    void assignTo(const Expr& e, uint32_t v) {
        if (e.k == EK::Swizzle) {
            const Expr& base = *e.a[0];
            uint32_t cur = value(base);
            const Type& bt = base.type;
            std::vector<uint32_t> comps(bt.vec);
            for (int i = 0; i < bt.vec; ++i) comps[size_t(i)] = uint32_t(i);
            for (int i = 0; i < e.nswz; ++i) comps[e.swz[i]] = uint32_t(bt.vec + i);
            uint32_t nv;
            if (e.nswz == 1) {
                nv = op(OpCompositeInsert, typeId(bt), {v, cur, e.swz[0]});
            } else {
                std::vector<uint32_t> w{cur, v};
                w.insert(w.end(), comps.begin(), comps.end());
                nv = opV(OpVectorShuffle, typeId(bt), w);
            }
            return assignTo(base, nv);
        }
        if (e.k == EK::Index && !e.a[0]->type.isArray() && !e.a[0]->type.cols) {
            // Vector component with a (possibly dynamic) index.
            uint32_t cur = value(*e.a[0]);
            uint32_t idx = value(*e.a[1]);
            uint32_t nv = op(OpVectorInsertDynamic, typeId(e.a[0]->type), {cur, v, idx});
            return assignTo(*e.a[0], nv);
        }
        Ptr p;
        if (!pointer(e, p)) fail("assignment to a non-l-value");
        store(p.ptr, v);
    }

    // ---- r-values -------------------------------------------------------------------------------
    uint32_t splat(uint32_t scalar, const Type& t) {
        if (t.vec == 1 && !t.cols) return scalar;
        std::vector<uint32_t> parts(t.vec, scalar);
        return opV(OpCompositeConstruct, typeId(t.columnType()), parts);
    }

    uint32_t value(const Expr& e) {
        if (e.hasConst && e.type.isScalar()) return constOf(e.type, e.constF, e.constI);
        switch (e.k) {
            case EK::IntLit: return constI(int32_t(uint32_t(e.i)));
            case EK::UintLit: return constU(uint32_t(e.i));
            case EK::FloatLit: return constF(float(e.f));
            case EK::BoolLit: return constB(e.i != 0);
            case EK::Var:
            case EK::Field:
            case EK::Index: {
                Ptr p;
                if (pointer(e, p)) return loadPtr(p);
                if (e.k == EK::Var) fail("cannot read '" + e.sym->name + "'");
                const Type& bt = e.a[0]->type;
                if (e.k == EK::Field) {
                    uint32_t b = value(*e.a[0]);
                    return op(OpCompositeExtract, typeId(e.type), {b, uint32_t(e.field)});
                }
                uint32_t b = value(*e.a[0]);
                if (!bt.isArray() && !bt.cols) {
                    if (e.a[1]->hasConst) return op(OpCompositeExtract, typeId(e.type), {b, uint32_t(e.a[1]->constI)});
                    return op(OpVectorExtractDynamic, typeId(e.type), {b, value(*e.a[1])});
                }
                if (e.a[1]->hasConst) return op(OpCompositeExtract, typeId(e.type), {b, uint32_t(e.a[1]->constI)});
                // Dynamic index into an r-value array/matrix: spill to a local.
                uint32_t tmp = localVar(typeId(bt));
                store(tmp, b);
                uint32_t pp = op(OpAccessChain, ptrType(SC::Function, typeId(e.type)), {tmp, value(*e.a[1])});
                return load(e.type, pp);
            }
            case EK::Swizzle: {
                uint32_t b = value(*e.a[0]);
                if (e.nswz == 1) return op(OpCompositeExtract, typeId(e.type), {b, e.swz[0]});
                std::vector<uint32_t> w{b, b};
                for (int i = 0; i < e.nswz; ++i) w.push_back(e.swz[i]);
                return opV(OpVectorShuffle, typeId(e.type), w);
            }
            case EK::Length: return constI(int32_t(e.constI));
            case EK::Comma:
                value(*e.a[0]);
                return value(*e.a[1]);
            case EK::Assign: {
                const Expr& l = *e.a[0];
                uint32_t v;
                if (e.op == "=") v = value(*e.a[1]);
                else {
                    std::string bop = e.op.substr(0, e.op.size() - 1);
                    v = arith(bop, value(l), l.type, value(*e.a[1]), e.a[1]->type, l.type);
                }
                assignTo(l, v);
                return v;
            }
            case EK::PreInc:
            case EK::PreDec:
            case EK::PostInc:
            case EK::PostDec: {
                const Expr& l = *e.a[0];
                uint32_t old = value(l);
                const bool inc = e.k == EK::PreInc || e.k == EK::PostInc;
                uint32_t nv = arith(inc ? "+" : "-", old, l.type, one(l.type), l.type.scalarType(), l.type);
                assignTo(l, nv);
                return e.k == EK::PreInc || e.k == EK::PreDec ? nv : old;
            }
            case EK::Unary: return unary(e);
            case EK::Binary: return binary(e);
            case EK::Ternary: {
                uint32_t c = value(*e.a[0]);
                uint32_t tmp = localVar(typeId(e.type));
                uint32_t t = id(), f = id(), m = id();
                inst(body_, OpSelectionMerge, {m, 0});
                inst(body_, OpBranchConditional, {c, t, f});
                label(t);
                store(tmp, value(*e.a[1]));
                branch(m);
                label(f);
                store(tmp, value(*e.a[2]));
                branch(m);
                label(m);
                return load(e.type, tmp);
            }
            case EK::Ctor: return construct(e);
            case EK::Call: return call(e);
            case EK::Builtin: return builtin(e);
        }
        fail("unsupported expression");
    }

    uint32_t unary(const Expr& e) {
        const Type& t = e.type;
        uint32_t v = value(*e.a[0]);
        if (e.op == "!") return op(OpLogicalNot, typeId(t), {v});
        if (e.op == "~") return op(OpNot, typeId(t), {v});
        // "-"
        if (t.cols) {
            std::vector<uint32_t> cols;
            for (int c = 0; c < t.cols; ++c)
                cols.push_back(op(OpFNegate, typeId(t.columnType()), {op(OpCompositeExtract, typeId(t.columnType()), {v, uint32_t(c)})}));
            return opV(OpCompositeConstruct, typeId(t), cols);
        }
        return op(t.base == Base::Float ? OpFNegate : OpSNegate, typeId(t), {v});
    }

    uint32_t binary(const Expr& e) {
        const std::string& o = e.op;
        if (o == "&&" || o == "||") {
            const Type b = Type::make(Base::Bool);
            uint32_t l = value(*e.a[0]);
            uint32_t tmp = localVar(typeId(b));
            store(tmp, l);
            uint32_t t = id(), m = id();
            inst(body_, OpSelectionMerge, {m, 0});
            if (o == "&&") inst(body_, OpBranchConditional, {l, t, m});
            else inst(body_, OpBranchConditional, {l, m, t});
            label(t);
            store(tmp, value(*e.a[1]));
            branch(m);
            label(m);
            return load(b, tmp);
        }
        uint32_t l = value(*e.a[0]);
        uint32_t r = value(*e.a[1]);
        return arith(o, l, e.a[0]->type, r, e.a[1]->type, e.type);
    }

    uint32_t columnOf(uint32_t m, const Type& mt, int c) {
        return op(OpCompositeExtract, typeId(mt.columnType()), {m, uint32_t(c)});
    }

    // Equality of whole values (any type): scalar bool.
    uint32_t equalAll(uint32_t l, uint32_t r, const Type& t) {
        const Type b = Type::make(Base::Bool);
        if (t.isArray()) {
            uint32_t acc = constB(true);
            Type el = t.element();
            for (int i = 0; i < t.dims[0]; ++i) {
                uint32_t x = op(OpCompositeExtract, typeId(el), {l, uint32_t(i)});
                uint32_t y = op(OpCompositeExtract, typeId(el), {r, uint32_t(i)});
                acc = op(OpLogicalAnd, typeId(b), {acc, equalAll(x, y, el)});
            }
            return acc;
        }
        if (t.base == Base::Struct) {
            uint32_t acc = constB(true);
            for (size_t i = 0; i < t.st->members.size(); ++i) {
                const Type& mt = t.st->members[i].type;
                uint32_t x = op(OpCompositeExtract, typeId(mt), {l, uint32_t(i)});
                uint32_t y = op(OpCompositeExtract, typeId(mt), {r, uint32_t(i)});
                acc = op(OpLogicalAnd, typeId(b), {acc, equalAll(x, y, mt)});
            }
            return acc;
        }
        if (t.cols) {
            uint32_t acc = constB(true);
            for (int c = 0; c < t.cols; ++c)
                acc = op(OpLogicalAnd, typeId(b), {acc, equalAll(columnOf(l, t, c), columnOf(r, t, c), t.columnType())});
            return acc;
        }
        uint16_t o = t.base == Base::Float ? OpFOrdEqual : t.base == Base::Bool ? OpLogicalEqual : OpIEqual;
        uint32_t c = op(o, typeId(Type::make(Base::Bool, t.vec)), {l, r});
        return t.vec > 1 ? op(OpAll, typeId(b), {c}) : c;
    }

    uint32_t arith(const std::string& o, uint32_t l, const Type& lt, uint32_t r, const Type& rt, const Type& res) {
        const Type b = Type::make(Base::Bool);
        if (o == "==") return equalAll(l, r, lt);
        if (o == "!=") return op(OpLogicalNot, typeId(b), {equalAll(l, r, lt)});
        if (o == "^^") return op(OpLogicalNotEqual, typeId(b), {l, r});
        if (o == "&&") return op(OpLogicalAnd, typeId(b), {l, r});
        if (o == "||") return op(OpLogicalOr, typeId(b), {l, r});
        const bool isF = lt.base == Base::Float, isU = lt.base == Base::Uint;
        if (o == "<" || o == ">" || o == "<=" || o == ">=") {
            uint16_t c;
            if (o == "<") c = isF ? OpFOrdLessThan : isU ? OpULessThan : OpSLessThan;
            else if (o == ">") c = isF ? OpFOrdGreaterThan : isU ? OpUGreaterThan : OpSGreaterThan;
            else if (o == "<=") c = isF ? OpFOrdLessThanEqual : isU ? OpULessThanEqual : OpSLessThanEqual;
            else c = isF ? OpFOrdGreaterThanEqual : isU ? OpUGreaterThanEqual : OpSGreaterThanEqual;
            return op(c, typeId(b), {l, r});
        }
        if (lt.cols || rt.cols) {
            if (o == "*") {
                if (lt.cols && rt.cols) return op(OpMatrixTimesMatrix, typeId(res), {l, r});
                if (lt.cols && rt.vec > 1) return op(OpMatrixTimesVector, typeId(res), {l, r});
                if (rt.cols && lt.vec > 1) return op(OpVectorTimesMatrix, typeId(res), {l, r});
                if (lt.cols) return op(OpMatrixTimesScalar, typeId(res), {l, r});
                return op(OpMatrixTimesScalar, typeId(res), {r, l});
            }
            // Component-wise per column (+, -, /, and matrix +/- scalar).
            std::vector<uint32_t> cols;
            const Type ct = res.columnType();
            for (int c = 0; c < res.cols; ++c) {
                uint32_t a = lt.cols ? columnOf(l, lt, c) : splat(l, ct);
                uint32_t d = rt.cols ? columnOf(r, rt, c) : splat(r, ct);
                cols.push_back(arith(o, a, ct, d, ct, ct));
            }
            return opV(OpCompositeConstruct, typeId(res), cols);
        }
        if (o == "<<" || o == ">>") {
            if (rt.vec != lt.vec) r = splat(r, lt);
            if (o == "<<") return op(OpShiftLeftLogical, typeId(res), {l, r});
            return op(isU ? OpShiftRightLogical : OpShiftRightArithmetic, typeId(res), {l, r});
        }
        // Scalar operands of vector operations are splat.
        if (lt.vec != res.vec) l = splat(l, res);
        if (rt.vec != res.vec) r = splat(r, res);
        uint16_t c = 0;
        if (o == "+") c = isF ? OpFAdd : OpIAdd;
        else if (o == "-") c = isF ? OpFSub : OpISub;
        else if (o == "*") c = isF ? OpFMul : OpIMul;
        else if (o == "/") c = isF ? OpFDiv : isU ? OpUDiv : OpSDiv;
        else if (o == "%") c = isU ? OpUMod : OpSRem;
        else if (o == "&") c = OpBitwiseAnd;
        else if (o == "|") c = OpBitwiseOr;
        else if (o == "^") c = OpBitwiseXor;
        else fail("operator '" + o + "'");
        return op(c, typeId(res), {l, r});
    }

    // Scalar conversion of `v` (scalar or vector of `from`'s base) to base `to` (same width).
    uint32_t convert(uint32_t v, const Type& from, Base to) {
        if (from.base == to) return v;
        Type t = Type::make(to, from.vec);
        const uint32_t tid = typeId(t);
        if (to == Base::Bool) {
            uint16_t c = from.base == Base::Float ? OpFUnordNotEqual : OpINotEqual;
            return op(c, tid, {v, from.vec > 1 ? splat(zero(from), from) : zero(from)});
        }
        if (from.base == Base::Bool) {
            uint32_t o1 = one(t), z = zero(t);
            if (t.vec > 1) o1 = splat(o1, t), z = splat(z, t);
            return op(OpSelect, tid, {v, o1, z});
        }
        if (to == Base::Float) return op(from.base == Base::Int ? OpConvertSToF : OpConvertUToF, tid, {v});
        if (from.base == Base::Float) return op(to == Base::Int ? OpConvertFToS : OpConvertFToU, tid, {v});
        return op(OpBitcast, tid, {v});  // int <-> uint
    }

    // Scalar components of a numeric/bool value, in GLSL order (matrices column-major).
    void components(uint32_t v, const Type& t, std::vector<std::pair<uint32_t, Base>>& out) {
        const Type s = t.scalarType();
        if (t.cols) {
            for (int c = 0; c < t.cols; ++c) components(columnOf(v, t, c), t.columnType(), out);
            return;
        }
        if (t.vec == 1) {
            out.push_back({v, t.base});
            return;
        }
        for (int i = 0; i < t.vec; ++i) out.push_back({op(OpCompositeExtract, typeId(s), {v, uint32_t(i)}), t.base});
    }

    uint32_t construct(const Expr& e) {
        const Type& t = e.type;
        if (t.isArray() || t.base == Base::Struct) {
            std::vector<uint32_t> parts;
            for (const ExprP& a : e.a) parts.push_back(value(*a));
            return opV(OpCompositeConstruct, typeId(t), parts);
        }
        const Type s = t.scalarType();
        // matN(matM)
        if (t.cols && e.a.size() == 1 && e.a[0]->type.cols) {
            const Type& mt = e.a[0]->type;
            uint32_t m = value(*e.a[0]);
            std::vector<uint32_t> cols;
            for (int c = 0; c < t.cols; ++c) {
                std::vector<uint32_t> comps;
                for (int r = 0; r < t.vec; ++r) {
                    if (c < mt.cols && r < mt.vec) comps.push_back(op(OpCompositeExtract, typeId(s), {m, uint32_t(c), uint32_t(r)}));
                    else comps.push_back(constF(c == r ? 1.0f : 0.0f));
                }
                cols.push_back(opV(OpCompositeConstruct, typeId(t.columnType()), comps));
            }
            return opV(OpCompositeConstruct, typeId(t), cols);
        }
        // Single scalar: splat (vectors) or diagonal (matrices).
        if (e.a.size() == 1 && e.a[0]->type.isScalar()) {
            uint32_t x = convert(value(*e.a[0]), e.a[0]->type, t.base);
            if (t.isScalar()) return x;
            if (!t.cols) return splat(x, t);
            std::vector<uint32_t> cols;
            for (int c = 0; c < t.cols; ++c) {
                std::vector<uint32_t> comps;
                for (int r = 0; r < t.vec; ++r) comps.push_back(c == r ? x : constF(0.0f));
                cols.push_back(opV(OpCompositeConstruct, typeId(t.columnType()), comps));
            }
            return opV(OpCompositeConstruct, typeId(t), cols);
        }
        // General: take components in order.
        std::vector<std::pair<uint32_t, Base>> comps;
        for (const ExprP& a : e.a) components(value(*a), a->type, comps);
        const int need = t.components();
        std::vector<uint32_t> conv;
        for (int i = 0; i < need; ++i) conv.push_back(convert(comps[size_t(i)].first, Type::make(comps[size_t(i)].second), t.base));
        if (t.isScalar()) return conv[0];
        if (!t.cols) return opV(OpCompositeConstruct, typeId(t), conv);
        std::vector<uint32_t> cols;
        for (int c = 0; c < t.cols; ++c) {
            std::vector<uint32_t> col(conv.begin() + c * t.vec, conv.begin() + (c + 1) * t.vec);
            cols.push_back(opV(OpCompositeConstruct, typeId(t.columnType()), col));
        }
        return opV(OpCompositeConstruct, typeId(t), cols);
    }

    uint32_t call(const Expr& e) {
        const Function* f = e.fn;
        if (!f->body) fail("'" + f->name + "' is declared but never defined");
        std::vector<uint32_t> args;
        std::vector<std::pair<const Expr*, uint32_t>> writeBack;
        for (size_t i = 0; i < f->params.size(); ++i) {
            const Symbol* p = f->params[i];
            const Expr& a = *e.a[i];
            if (p->type.isSampler()) {
                Ptr sp;
                if (!pointer(a, sp)) fail("sampler argument is not a uniform");
                args.push_back(sp.ptr);
                continue;
            }
            uint32_t tmp = localVar(typeId(p->type));
            if (p->dir != ParamDir::Out) store(tmp, value(a));
            if (p->dir != ParamDir::In) writeBack.push_back({&a, tmp});
            args.push_back(tmp);
        }
        std::vector<uint32_t> w{fnIds_.at(f)};
        w.insert(w.end(), args.begin(), args.end());
        uint32_t r = opV(OpFunctionCall, typeId(f->ret), w);
        for (auto& [a, tmp] : writeBack) assignTo(*a, load(a->type, tmp));
        return r;
    }

    // ---- built-in functions ---------------------------------------------------------------------
    uint32_t splatTo(uint32_t v, const Type& from, const Type& to) {
        return from.vec == to.vec ? v : splat(v, to);
    }

    uint32_t builtin(const Expr& e) {
        const std::string& n = e.op;
        Builtins::TexCall tc;
        if (Builtins::textureCall(n, tc)) return texture(e, tc);
        const Type& rt = e.type;
        const uint32_t rid = typeId(rt);
        std::vector<uint32_t> a;
        if (n != "modf")
            for (const ExprP& x : e.a) a.push_back(value(*x));
        const Type at = e.a.empty() ? Type{} : e.a[0]->type;
        const bool isF = at.base == Base::Float, isU = at.base == Base::Uint;
        static const std::map<std::string, uint32_t> kF1 = {
            {"radians", GLSL_Radians}, {"degrees", GLSL_Degrees}, {"sin", GLSL_Sin}, {"cos", GLSL_Cos},
            {"tan", GLSL_Tan}, {"asin", GLSL_Asin}, {"acos", GLSL_Acos}, {"sinh", GLSL_Sinh}, {"cosh", GLSL_Cosh},
            {"tanh", GLSL_Tanh}, {"asinh", GLSL_Asinh}, {"acosh", GLSL_Acosh}, {"atanh", GLSL_Atanh},
            {"exp", GLSL_Exp}, {"log", GLSL_Log}, {"exp2", GLSL_Exp2}, {"log2", GLSL_Log2}, {"sqrt", GLSL_Sqrt},
            {"inversesqrt", GLSL_InverseSqrt}, {"floor", GLSL_Floor}, {"ceil", GLSL_Ceil}, {"fract", GLSL_Fract},
            {"trunc", GLSL_Trunc}, {"round", GLSL_Round}, {"roundEven", GLSL_RoundEven}, {"normalize", GLSL_Normalize},
            {"length", GLSL_Length}, {"determinant", GLSL_Determinant}, {"inverse", GLSL_MatrixInverse},
            {"packSnorm2x16", GLSL_PackSnorm2x16}, {"packUnorm2x16", GLSL_PackUnorm2x16},
            {"packUnorm4x8", GLSL_PackUnorm4x8}, {"packSnorm4x8", GLSL_PackSnorm4x8},
            {"unpackSnorm2x16", GLSL_UnpackSnorm2x16}, {"unpackUnorm2x16", GLSL_UnpackUnorm2x16},
            {"unpackUnorm4x8", GLSL_UnpackUnorm4x8}, {"unpackSnorm4x8", GLSL_UnpackSnorm4x8},
            {"packHalf2x16", GLSL_PackHalf2x16}, {"unpackHalf2x16", GLSL_UnpackHalf2x16},
        };
        if (n == "atan" && a.size() == 2) return ext(rid, GLSL_Atan2, {a[0], a[1]});
        if (n == "atan") return ext(rid, GLSL_Atan, {a[0]});
        if (auto it = kF1.find(n); it != kF1.end()) return ext(rid, it->second, {a[0]});
        if (n == "abs") return ext(rid, isF ? GLSL_FAbs : GLSL_SAbs, {a[0]});
        if (n == "sign") return ext(rid, isF ? GLSL_FSign : GLSL_SSign, {a[0]});
        if (n == "pow") return ext(rid, GLSL_Pow, {a[0], a[1]});
        if (n == "mod") return op(OpFMod, rid, {a[0], splatTo(a[1], e.a[1]->type, rt)});
        if (n == "min" || n == "max") {
            uint32_t g = n == "min" ? (isF ? GLSL_FMin : isU ? GLSL_UMin : GLSL_SMin) : (isF ? GLSL_FMax : isU ? GLSL_UMax : GLSL_SMax);
            return ext(rid, g, {a[0], splatTo(a[1], e.a[1]->type, rt)});
        }
        if (n == "clamp")
            return ext(rid, isF ? GLSL_FClamp : isU ? GLSL_UClamp : GLSL_SClamp,
                       {a[0], splatTo(a[1], e.a[1]->type, rt), splatTo(a[2], e.a[2]->type, rt)});
        if (n == "mix") {
            if (e.a[2]->type.base == Base::Bool) return op(OpSelect, rid, {a[2], a[1], a[0]});
            return ext(rid, GLSL_FMix, {a[0], a[1], splatTo(a[2], e.a[2]->type, rt)});
        }
        if (n == "step") return ext(rid, GLSL_Step, {splatTo(a[0], e.a[0]->type, rt), a[1]});
        if (n == "smoothstep")
            return ext(rid, GLSL_SmoothStep, {splatTo(a[0], e.a[0]->type, rt), splatTo(a[1], e.a[1]->type, rt), a[2]});
        if (n == "fma") return ext(rid, GLSL_Fma, {a[0], a[1], a[2]});
        if (n == "distance") return ext(rid, GLSL_Distance, {a[0], a[1]});
        if (n == "dot") return at.vec == 1 ? op(OpFMul, rid, {a[0], a[1]}) : op(OpDot, rid, {a[0], a[1]});
        if (n == "cross") return ext(rid, GLSL_Cross, {a[0], a[1]});
        if (n == "faceforward") return ext(rid, GLSL_FaceForward, {a[0], a[1], a[2]});
        if (n == "reflect") return ext(rid, GLSL_Reflect, {a[0], a[1]});
        if (n == "refract") return ext(rid, GLSL_Refract, {a[0], a[1], a[2]});
        if (n == "matrixCompMult") {
            std::vector<uint32_t> cols;
            for (int c = 0; c < rt.cols; ++c)
                cols.push_back(op(OpFMul, typeId(rt.columnType()), {columnOf(a[0], rt, c), columnOf(a[1], rt, c)}));
            return opV(OpCompositeConstruct, rid, cols);
        }
        if (n == "outerProduct") return op(OpOuterProduct, rid, {a[0], a[1]});
        if (n == "transpose") return op(OpTranspose, rid, {a[0]});
        if (n == "lessThan" || n == "lessThanEqual" || n == "greaterThan" || n == "greaterThanEqual") {
            uint16_t c;
            if (n == "lessThan") c = isF ? OpFOrdLessThan : isU ? OpULessThan : OpSLessThan;
            else if (n == "lessThanEqual") c = isF ? OpFOrdLessThanEqual : isU ? OpULessThanEqual : OpSLessThanEqual;
            else if (n == "greaterThan") c = isF ? OpFOrdGreaterThan : isU ? OpUGreaterThan : OpSGreaterThan;
            else c = isF ? OpFOrdGreaterThanEqual : isU ? OpUGreaterThanEqual : OpSGreaterThanEqual;
            return op(c, rid, {a[0], a[1]});
        }
        if (n == "equal") return op(isF ? OpFOrdEqual : at.base == Base::Bool ? OpLogicalEqual : OpIEqual, rid, {a[0], a[1]});
        if (n == "notEqual") return op(isF ? OpFUnordNotEqual : at.base == Base::Bool ? OpLogicalNotEqual : OpINotEqual, rid, {a[0], a[1]});
        if (n == "any") return op(OpAny, rid, {a[0]});
        if (n == "all") return op(OpAll, rid, {a[0]});
        if (n == "not") return op(OpLogicalNot, rid, {a[0]});
        if (n == "isnan") return op(OpIsNan, rid, {a[0]});
        if (n == "isinf") return op(OpIsInf, rid, {a[0]});
        if (n == "floatBitsToInt" || n == "floatBitsToUint" || n == "intBitsToFloat" || n == "uintBitsToFloat")
            return op(OpBitcast, rid, {a[0]});
        if (n == "dFdx") return op(OpDPdx, rid, {a[0]});
        if (n == "dFdy") return op(OpDPdy, rid, {a[0]});
        if (n == "fwidth") return op(OpFwidth, rid, {a[0]});
        if (n == "bitCount") return op(OpBitCount, rid, {a[0]});
        if (n == "findLSB") return ext(rid, GLSL_FindILsb, {a[0]});
        if (n == "findMSB") return ext(rid, isU ? GLSL_FindUMsb : GLSL_FindSMsb, {a[0]});
        if (n == "bitfieldReverse") return op(OpBitReverse, rid, {a[0]});
        if (n == "bitfieldExtract") return op(isU ? OpBitFieldUExtract : OpBitFieldSExtract, rid, {a[0], a[1], a[2]});
        if (n == "bitfieldInsert") return op(OpBitFieldInsert, rid, {a[0], a[1], a[2], a[3]});
        if (n == "modf") {
            uint32_t x = value(*e.a[0]);
            uint32_t tmp = localVar(rid);
            uint32_t r = ext(rid, GLSL_Modf, {x, tmp});
            assignTo(*e.a[1], load(rt, tmp));
            return r;
        }
        fail("built-in '" + n + "' is not implemented");
    }

    uint32_t texture(const Expr& e, const Builtins::TexCall& tc) {
        const Type& st = e.a[0]->type;
        Ptr sp;
        if (!pointer(*e.a[0], sp)) fail("sampler is not a uniform");
        uint32_t si = load(st, sp.ptr);
        const Type rt = e.type;
        const uint32_t rid = typeId(rt);
        const Type f = Type::make(Base::Float);
        size_t i = 1;
        auto arg = [&]() { return value(*e.a[i++]); };
        if (tc.size) {
            uint32_t img = op(OpImage, imageOf(st), {si});
            if (st.ms) return op(OpImageQuerySize, rid, {img});
            uint32_t lod = arg();
            return op(OpImageQuerySizeLod, rid, {img, lod});
        }
        if (tc.fetch) {
            uint32_t img = op(OpImage, imageOf(st), {si});
            uint32_t coord = arg();
            uint32_t lodOrSample = arg();
            std::vector<uint32_t> w{img, coord};
            uint32_t mask = st.ms ? ImgOp::Sample : ImgOp::Lod;
            std::vector<uint32_t> ops{lodOrSample};
            if (tc.offset) {
                const Expr& o = *e.a[i];
                ops.push_back(value(o));
                mask |= constantExpr(o) ? ImgOp::ConstOffset : ImgOp::Offset;
                ++i;
            }
            w.push_back(mask);
            w.insert(w.end(), ops.begin(), ops.end());
            return opV(OpImageFetch, rid, w);
        }
        if (tc.gather) {
            uint32_t coord = arg();
            std::vector<uint32_t> w{si, coord};
            uint32_t dref = 0;
            if (st.shadow) dref = arg();
            uint32_t mask = 0;
            std::vector<uint32_t> ops;
            if (tc.offset) {
                const Expr& o = *e.a[i];
                ops.push_back(value(o));
                mask |= constantExpr(o) ? ImgOp::ConstOffset : ImgOp::Offset;
                ++i;
            }
            uint32_t comp = constI(0);
            if (!st.shadow && i < e.a.size()) comp = arg();
            if (st.shadow) w.push_back(dref);
            else w.push_back(comp);
            if (mask) {
                w.push_back(mask);
                w.insert(w.end(), ops.begin(), ops.end());
            }
            return opV(st.shadow ? OpImageDrefGather : OpImageGather, rid, w);
        }
        // Sampling.
        const Type& ct = e.a[1]->type;
        uint32_t coord = arg();
        uint32_t dref = 0;
        if (st.shadow) {
            // The reference is the last coordinate component (projective: the third, before q).
            if (tc.proj) {
                dref = op(OpCompositeExtract, typeId(f), {coord, 2});
                coord = opV(OpVectorShuffle, typeId(Type::make(Base::Float, 3)), {coord, coord, 0, 1, 3});
            } else {
                uint32_t last = uint32_t(ct.vec - 1);
                dref = op(OpCompositeExtract, typeId(f), {coord, last});
                Type c2 = Type::make(Base::Float, ct.vec - 1);
                std::vector<uint32_t> w{coord, coord};
                for (uint32_t k = 0; k < last; ++k) w.push_back(k);
                coord = opV(OpVectorShuffle, typeId(c2), w);
            }
        } else if (tc.proj && ct.vec == 4 && (st.dim == SDim::D2 || st.dim == SDim::External)) {
            coord = opV(OpVectorShuffle, typeId(Type::make(Base::Float, 3)), {coord, coord, 0, 1, 3});  // .xyw
        }
        uint32_t mask = 0;
        std::vector<uint32_t> ops;
        uint32_t lod = 0, gx = 0, gy = 0, offset = 0, bias = 0;
        bool constOffset = false;
        if (tc.lod) lod = arg();
        if (tc.grad) gx = arg(), gy = arg();
        if (tc.offset) {
            constOffset = constantExpr(*e.a[i]);
            offset = arg();
        }
        if (i < e.a.size()) bias = arg();
        // Implicit derivatives exist only in fragment shaders: elsewhere sample level 0.
        bool explicitLod = tc.lod || tc.grad || sh_.stage != Stage::Fragment;
        if (!tc.lod && !tc.grad && sh_.stage != Stage::Fragment) lod = constF(0.0f);
        if (bias && !explicitLod) mask |= ImgOp::Bias, ops.push_back(bias);
        if (lod) mask |= ImgOp::Lod, ops.push_back(lod);
        if (tc.grad) mask |= ImgOp::Grad, ops.push_back(gx), ops.push_back(gy);
        if (tc.offset) mask |= constOffset ? ImgOp::ConstOffset : ImgOp::Offset, ops.push_back(offset);
        uint16_t opc;
        if (st.shadow) opc = tc.proj ? (explicitLod ? OpImageSampleProjDrefExplicitLod : OpImageSampleProjDrefImplicitLod)
                                     : (explicitLod ? OpImageSampleDrefExplicitLod : OpImageSampleDrefImplicitLod);
        else opc = tc.proj ? (explicitLod ? OpImageSampleProjExplicitLod : OpImageSampleProjImplicitLod)
                           : (explicitLod ? OpImageSampleExplicitLod : OpImageSampleImplicitLod);
        std::vector<uint32_t> w{si, coord};
        if (st.shadow) w.push_back(dref);
        if (mask) {
            w.push_back(mask);
            w.insert(w.end(), ops.begin(), ops.end());
        }
        return opV(opc, rid, w);
    }

    static bool constantExpr(const Expr& e) {
        if (e.hasConst) return true;
        if (e.k == EK::Ctor) {
            for (const ExprP& a : e.a)
                if (!constantExpr(*a)) return false;
            return true;
        }
        return e.k == EK::IntLit || e.k == EK::UintLit || e.k == EK::FloatLit || e.k == EK::BoolLit;
    }

    const Shader& sh_;
    const ProgramLayout& L_;
    uint32_t glsl_ = 0, voidT_ = 0, boolT_ = 0, mainId_ = 0, pointCoordVar_ = 0;
    bool writesDepth_ = false, usesMultiview_ = false;
    StructDef defaultStruct_;
    std::vector<std::unique_ptr<StructDef>> blockStructs_;
};

}  // namespace

std::vector<uint32_t> emitSpirv(const Shader& sh, const ProgramLayout& layout, std::string& error) {
    try {
        Emitter e(sh, layout);
        return e.run();
    } catch (const std::exception& x) {
        error = x.what();
        return {};
    }
}

static void walkExpr(const Expr& e, const std::function<void(const Expr&)>& fn) {
    fn(e);
    for (const ExprP& a : e.a) walkExpr(*a, fn);
}
static void walkStmt(const Stmt& s, const std::function<void(const Expr&)>& fn) {
    if (s.e) walkExpr(*s.e, fn);
    if (s.inc) walkExpr(*s.inc, fn);
    for (const StmtP* c : {&s.init, &s.then, &s.els, &s.body})
        if (*c) walkStmt(**c, fn);
    for (const StmtP& c : s.list) walkStmt(*c, fn);
    for (const auto& [sym, init] : s.decls)
        if (init) walkExpr(*init, fn);
}

void forEachExpr(const Shader& sh, const std::function<void(const Expr&)>& fn) {
    for (const GlobalDecl& g : sh.globals)
        if (g.init) walkExpr(*g.init, fn);
    for (const Function* f : sh.functions)
        if (f->body) walkStmt(*f->body, fn);
}

}  // namespace vpglsl
