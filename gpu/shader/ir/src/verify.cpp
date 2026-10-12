#include "vpir/verify.h"

#include <algorithm>
#include <map>
#include <set>
#include <utility>

#include "text_common.h"

namespace vpir {

namespace {

constexpr unsigned bit(Stage s) { return 1u << unsigned(s); }
constexpr unsigned kGridStages = bit(Stage::Compute) | bit(Stage::Object) | bit(Stage::Mesh);
constexpr unsigned kSubgroupStages = kGridStages | bit(Stage::Fragment);

// Number of components returned by image_size for a dimension.
unsigned sizeWidth(TexDim d) {
    switch (d) {
        case TexDim::D1: return 1;
        case TexDim::D2:
        case TexDim::Cube: return 2;
        default: return 3;
    }
}

class Verifier {
public:
    explicit Verifier(const Module& m) : m_(m) {}

    std::vector<std::string> run() {
        checkInterface();
        for (fnIndex_ = 0; fnIndex_ < m_.functions.size(); ++fnIndex_) checkFunction();
        const Function* e = m_.entry();
        if (!e) error("module has no 'main' function");
        else if (!e->params.empty() || !e->returnType.isVoid()) error("'main' must take no parameters and return void");
        return std::move(errors_);
    }

private:
    template <typename... T>
    void error(const T&... parts) {
        std::string s;
        if (fn_) s = fn_->name + ": ";
        if (inst_) s += std::string(opInfo(inst_->op).name) + ": ";
        ((s += parts), ...);
        errors_.push_back(std::move(s));
    }
    bool require(bool ok, const char* what) {
        if (!ok) error(what);
        return ok;
    }

    bool stageIs(unsigned mask) const { return mask & bit(m_.stage); }

    // ---- module interface ----
    void checkIo(const IoVar& v, bool input, std::set<std::string>& names, std::set<uint32_t>& locs) {
        const char* dir = input ? "input " : "output ";
        if (!names.insert(v.name).second) error(dir, v.name, ": duplicate name");
        if (v.type.isVoid()) error(dir, v.name, ": void type");
        if (v.perPrimitive && (input || m_.stage != Stage::Mesh))
            error(dir, v.name, ": per_primitive is only for mesh outputs");
        if (v.builtin != Builtin::None) {
            BuiltinInfo b = builtinInfo(v.builtin, input);
            if (b.input != input || !stageIs(b.stages))
                error(dir, v.name, ": builtin ", toString(v.builtin), " not valid here");
            else if (b.type != v.type)
                error(dir, v.name, ": builtin ", toString(v.builtin), " must be ", toString(b.type));
            return;
        }
        if (m_.stage == Stage::Compute || m_.stage == Stage::Object || (input && m_.stage == Stage::Mesh))
            error(dir, v.name, ": this stage only has builtin ", input ? "inputs" : "outputs");
        if (v.type.isBool()) error(dir, v.name, ": bool cannot cross a stage boundary");
        if (!locs.insert(v.location).second) error(dir, v.name, ": duplicate location");
        bool interpolated = (input && m_.stage == Stage::Fragment) || (!input && m_.stage != Stage::Fragment);
        if (v.interp != Interp::Flat && interpolated && !v.type.isFloat())
            error(dir, v.name, ": integer varyings must be flat");
    }

    void checkInterface() {
        std::set<std::string> names;
        std::set<uint32_t> inLocs, outLocs;
        for (const IoVar& v : m_.inputs) checkIo(v, true, names, inLocs);
        names.clear();
        for (const IoVar& v : m_.outputs) checkIo(v, false, names, outLocs);
        bool rasterized = m_.stage == Stage::Mesh || (m_.stage == Stage::Vertex && !m_.outputs.empty());
        bool hasPosition = std::any_of(m_.outputs.begin(), m_.outputs.end(),
                                       [](const IoVar& v) { return v.builtin == Builtin::Position; });
        if (rasterized && !hasPosition) error("vertex and mesh outputs need a position builtin");

        names.clear();
        std::set<std::pair<ResourceKind, uint32_t>> slots;
        for (const Resource& r : m_.resources) {
            if (!names.insert(r.name).second) error("resource ", r.name, ": duplicate name");
            if (!slots.insert({r.kind, r.index}).second) error("resource ", r.name, ": duplicate index");
            if (r.kind == ResourceKind::StorageTexture && r.depth) error("resource ", r.name, ": storage textures cannot be depth");
            if (r.depth && r.elem != Scalar::F32) error("resource ", r.name, ": depth textures are f32");
            if (r.depth && (r.dim == TexDim::D1 || r.dim == TexDim::D3))
                error("resource ", r.name, ": depth textures are 2d, 2d_array, cube or cube_array");
            if ((r.kind == ResourceKind::Texture || r.kind == ResourceKind::StorageTexture) &&
                (r.elem == Scalar::Bool))
                error("resource ", r.name, ": bool texel type");
        }

        bool grid = stageIs(kGridStages);
        if (grid && (!m_.workgroupSize[0] || !m_.workgroupSize[1] || !m_.workgroupSize[2]))
            error("workgroup size must be non-zero");
        if (!grid && m_.sharedBytes) error("shared memory needs a compute, object or mesh stage");
        if (m_.sharedBytes % 4) error("shared_bytes must be a multiple of 4");
        if (m_.privateBytes % 4) error("private_bytes must be a multiple of 4");
        for (uint32_t b : m_.privateSlots)
            if (b == 0 || b % 16) error("private slot sizes must be non-zero multiples of 16");
        if (m_.payloadBytes && !stageIs(bit(Stage::Object) | bit(Stage::Mesh)))
            error("payload needs an object or mesh stage");
        if (m_.waveSize != 32 && m_.waveSize != 64) error("wave_size must be 32 or 64");
        if (m_.payloadBytes % 4) error("payload_bytes must be a multiple of 4");
        if (m_.stage == Stage::Mesh && (!m_.maxVertices || !m_.maxPrimitives))
            error("mesh stage needs max_vertices and max_primitives");
        // Metal's mesh limits; indices are stored as 8-bit values.
        if (m_.maxVertices > 256 || m_.maxPrimitives > 512) error("mesh limits are 256 vertices, 512 primitives");

        std::set<std::string> fnNames;
        for (const Function& f : m_.functions)
            if (!fnNames.insert(f.name).second) error("function ", f.name, ": duplicate name");
    }

    // ---- values ----
    // Scoped definitions: a value is visible from its definition to the end of its block.
    void define(ValueId id) {
        if (id >= fn_->valueTypes.size() || fn_->valueTypes[id].isVoid()) {
            error("%", std::to_string(id), " has no type");
            return;
        }
        if (everDefined_[id]) error("%", std::to_string(id), " defined twice");
        everDefined_[id] = visible_[id] = true;
        scopes_.back().push_back(id);
    }
    Type use(ValueId id) {
        if (id >= visible_.size() || !visible_[id]) {
            error("%", std::to_string(id), " used before definition or outside its scope");
            return kVoid;
        }
        return fn_->valueTypes[id];
    }

    void checkFunction() {
        fn_ = &m_.functions[fnIndex_];
        everDefined_.assign(fn_->valueTypes.size(), false);
        visible_.assign(fn_->valueTypes.size(), false);
        scopes_.assign(1, {});
        for (ValueId p : fn_->params) define(p);
        for (const Variable& v : fn_->vars)
            if (v.type.isVoid()) error("variable of type void");
        loopDepth_ = 0;
        checkBlock(fn_->body);
        fn_ = nullptr;
    }

    void checkBlock(const Block& b) {
        scopes_.emplace_back();
        for (const Node& n : b) checkNode(n);
        for (ValueId id : scopes_.back()) visible_[id] = false;
        scopes_.pop_back();
    }

    void checkNode(const Node& n) {
        switch (n.kind) {
            case NodeKind::Inst: checkInst(n.inst); break;
            case NodeKind::If:
                if (use(n.value) != kBool) error("if condition must be bool");
                checkBlock(n.body);
                checkBlock(n.elseBody);
                break;
            case NodeKind::Loop:
                ++loopDepth_;
                checkBlock(n.body);
                --loopDepth_;
                break;
            case NodeKind::Break:
            case NodeKind::Continue:
                if (!loopDepth_) error(n.kind == NodeKind::Break ? "break" : "continue", " outside a loop");
                break;
            case NodeKind::Discard:
                if (m_.stage != Stage::Fragment) error("discard outside a fragment shader");
                break;
            case NodeKind::Return: {
                Type t = n.value == kNoValue ? kVoid : use(n.value);
                if (t != fn_->returnType) error("return type does not match function");
                break;
            }
        }
    }

    // ---- instructions ----
    void checkInst(const Inst& in) {
        inst_ = &in;
        std::vector<Type> a;
        for (ValueId v : in.args) a.push_back(use(v));
        if (in.result != kNoValue) {
            define(in.result);
            if (in.result < fn_->valueTypes.size() && fn_->valueTypes[in.result] != in.type)
                error("result type disagrees with the function's value table");
        }
        // Type rules index operands by position, so they only run on well-formed operand lists.
        if (checkOperandCount(in) && !hasVoid(a)) checkTypes(in, a);
        inst_ = nullptr;
    }

    static bool hasVoid(const std::vector<Type>& a) {
        for (Type t : a)
            if (t.isVoid()) return true;
        return false;
    }

    bool checkOperandCount(const Inst& in) {
        unsigned values = 0, lits = 0;
        bool variadic = false, litList = false;
        for (const char* p = opInfo(in.op).signature; *p; ++p) {
            if (*p == 'v') ++values;
            if (*p == '*') variadic = true;
            if (*p == 'n') ++lits;
            if (*p == 'N' || *p == 'k') litList = true;
            if (*p == 'I')
                for (const ImageOperandSpec& s : kImageOperands)
                    if (in.image & s.bit) values += s.count;
        }
        bool ok = require(variadic ? in.args.size() >= values : in.args.size() == values, "wrong number of operands");
        ok &= require(litList ? in.lits.size() >= lits : in.lits.size() == lits, "wrong number of literals");
        ok &= require(!in.image || std::string_view(opInfo(in.op).signature).find('I') != std::string_view::npos, "unexpected image operands");
        return ok;
    }

    void expectResult(const Inst& in, bool has) {
        if ((in.result != kNoValue) != has) error(has ? "needs a result" : "has no result");
    }

    bool entryOnly() {
        if (fn_->name != "main") {
            error("stage interface access is only allowed in 'main'");
            return false;
        }
        return true;
    }

    // Checks a memory target and returns false if it is invalid.
    bool checkTarget(const Inst& in, bool write) {
        if (!entryOnly()) return false;
        if (in.sym == kTargetShared) return require(m_.sharedBytes > 0, "shared memory is not declared");
        if (in.sym == kTargetPrivate) return require(m_.privateBytes > 0, "private memory is not declared");
        if (isPrivateSlot(in.sym))
            return require(privateSlotIndex(in.sym) < m_.privateSlots.size(), "private slot is not declared");
        if (in.sym == kTargetPayload) {
            if (!require(m_.payloadBytes > 0, "payload is not declared")) return false;
            return require(!write || m_.stage == Stage::Object, "payload is read-only in mesh shaders");
        }
        if (in.sym >= m_.resources.size()) return require(false, "unknown memory target");
        if (m_.resources[in.sym].kind == ResourceKind::ConstantBuffer)
            return require(!write, "constant buffers are read-only");
        if (m_.resources[in.sym].kind != ResourceKind::Buffer) return require(false, "target is not a buffer");
        Access acc = m_.resources[in.sym].access;
        return require(write ? acc != Access::Read : acc != Access::Write, "buffer access mode forbids this");
    }

    const Resource* image(uint32_t sym, bool storage) {
        if (!entryOnly()) return nullptr;
        if (sym >= m_.resources.size()) return require(false, "unknown resource"), nullptr;
        const Resource& r = m_.resources[sym];
        ResourceKind want = storage ? ResourceKind::StorageTexture : ResourceKind::Texture;
        if (r.kind != want) return require(false, storage ? "expected a storage texture" : "expected a texture"), nullptr;
        return &r;
    }

    // Shape of the coordinate operand: layer (and cube face for integer access) is the last component.
    // For integer access to cubes the three components are x, y, face (then layer if arrayed).
    static Type coordType(const Resource& r, Scalar s) {
        return {s, uint8_t(coordWidth(r.dim) + (isArrayed(r.dim) ? 1 : 0))};
    }

    void checkSampleOperands(const Inst& in, const Resource& r, size_t first, const std::vector<Type>& a) {
        size_t i = first;
        unsigned cw = coordWidth(r.dim);
        bool cube = r.dim == TexDim::Cube || r.dim == TexDim::CubeArray;
        if (in.image & kImageLod) require(a[i++] == kF32, "lod must be f32");
        if (in.image & kImageBias) require(a[i++] == kF32, "bias must be f32");
        if (in.image & kImageGrad) {
            Type g{Scalar::F32, uint8_t(cw)};
            require(a[i] == g && a[i + 1] == g, "gradients must match the coordinate dimension");
            i += 2;
        }
        if (in.image & kImageOffset) {
            require(!cube && r.dim != TexDim::D1, "offsets need a 2d or 3d texture");
            require(a[i++] == Type(Scalar::I32, uint8_t(cw)), "offset must be an i32 vector of the coordinate dimension");
        }
        if (in.image & kImageCompare) {
            require(r.depth, "compare needs a depth texture");
            require(a[i++] == kF32, "compare reference must be f32");
            require(!(in.image & (kImageBias | kImageGrad)), "compare only allows lod");
        }
        unsigned lodModes = !!(in.image & kImageLod) + !!(in.image & kImageBias) + !!(in.image & kImageGrad);
        require(lodModes <= 1, "lod, bias and grad are mutually exclusive");
        require(r.dim != TexDim::D1 || !in.image, "1d textures take no image operands");
        if (!(in.image & (kImageLod | kImageGrad)) && r.dim != TexDim::D1)
            require(m_.stage == Stage::Fragment, "implicit-lod sampling is fragment-only");
    }

    Type texel(const Resource& r) const { return r.depth ? kF32 : Type(r.elem, 4); }

    void checkTypes(const Inst& in, const std::vector<Type>& a) {
        const Type t = in.type;
        auto sameAll = [&](Type want) {
            for (Type x : a)
                if (x != want) return false;
            return true;
        };
        auto unaryOrBinary = [&](bool ok, const char* what) {
            expectResult(in, true);
            require(ok && sameAll(t), what);
        };

        switch (in.op) {
            case Op::Const:
                expectResult(in, true);
                require(in.lits.size() == t.width, "needs one literal per component");
                if (t.isBool())
                    for (uint32_t l : in.lits) require(l <= 1, "bool literal must be 0 or 1");
                if (t.scalar == Scalar::F16)
                    for (uint32_t l : in.lits) require(l <= 0xFFFF, "f16 literal out of range");
                break;
            case Op::LoadVar:
                expectResult(in, true);
                if (require(in.sym < fn_->vars.size(), "unknown variable")) require(fn_->vars[in.sym].type == t, "type mismatch");
                break;
            case Op::StoreVar:
                expectResult(in, false);
                if (require(in.sym < fn_->vars.size(), "unknown variable")) require(fn_->vars[in.sym].type == a[0], "type mismatch");
                break;
            case Op::LoadInput:
                expectResult(in, true);
                if (entryOnly() && require(in.sym < m_.inputs.size(), "unknown input"))
                    require(m_.inputs[in.sym].type == t, "type mismatch");
                break;
            case Op::StoreOutput:
                expectResult(in, false);
                require(m_.stage != Stage::Mesh, "mesh shaders write outputs with mesh_vertex/mesh_primitive");
                if (entryOnly() && require(in.sym < m_.outputs.size(), "unknown output"))
                    require(m_.outputs[in.sym].type == a[0], "type mismatch");
                break;
            case Op::Call: {
                if (!require(in.sym < fnIndex_, "can only call functions declared earlier")) break;
                const Function& f = m_.functions[in.sym];
                expectResult(in, !f.returnType.isVoid());
                if (in.result != kNoValue) require(t == f.returnType, "result type differs from the callee's");
                bool ok = a.size() == f.params.size();
                for (size_t i = 0; ok && i < a.size(); ++i) ok = a[i] == f.typeOf(f.params[i]);
                require(ok, "arguments do not match the callee's parameters");
                break;
            }
            case Op::Neg:
            case Op::Abs: unaryOrBinary(t.isFloat() || t.scalar == Scalar::I32, "needs float or i32 operand"); break;
            case Op::Not: unaryOrBinary(t.isInt() || t.isBool(), "needs integer or bool operand"); break;
            case Op::Add:
            case Op::Sub:
            case Op::Mul:
            case Op::Div:
            case Op::Rem:
            case Op::Min:
            case Op::Max:
            case Op::Clamp: unaryOrBinary(t.isNumeric(), "needs matching numeric operands"); break;
            case Op::And:
            case Op::Or:
            case Op::Xor: unaryOrBinary(t.isInt() || t.isBool(), "needs matching integer or bool operands"); break;
            case Op::Shl:
            case Op::Shr:
            case Op::BitCount:
            case Op::BitReverse:
            case Op::FindLsb:
            case Op::FindMsb: unaryOrBinary(t.isInt(), "needs matching integer operands"); break;
            case Op::Fma:
            case Op::Mix:
            case Op::Pow:
            case Op::Floor:
            case Op::Ceil:
            case Op::Trunc:
            case Op::Round:
            case Op::Fract:
            case Op::Sqrt:
            case Op::Rsq:
            case Op::Rcp:
            case Op::Exp2:
            case Op::Log2:
            case Op::Sin:
            case Op::Cos:
            case Op::Saturate: unaryOrBinary(t.isFloat(), "needs matching float operands"); break;
            case Op::Normalize: unaryOrBinary(t.isFloat() && t.width > 1, "needs a float vector"); break;
            case Op::Eq:
            case Op::Ne:
            case Op::Lt:
            case Op::Le:
            case Op::Gt:
            case Op::Ge: {
                expectResult(in, true);
                bool eq = in.op == Op::Eq || in.op == Op::Ne;
                require(a[0] == a[1] && (a[0].isNumeric() || (eq && a[0].isBool())), "operands must match");
                require(t == kBool.withWidth(a[0].width), "result must be bool of the operand width");
                break;
            }
            case Op::IsNan:
            case Op::IsInf:
                expectResult(in, true);
                require(a[0].isFloat() && t == kBool.withWidth(a[0].width), "needs float operand and bool result");
                break;
            case Op::Select:
                expectResult(in, true);
                require(a[0].isBool() && (a[0].width == 1 || a[0].width == t.width), "condition must be bool");
                require(a[1] == t && a[2] == t, "operands must match the result");
                break;
            case Op::Convert:
                expectResult(in, true);
                require(a[0].width == t.width && a[0].scalar != t.scalar, "needs same width, different scalar");
                break;
            case Op::Bitcast:
                expectResult(in, true);
                require(!a[0].isBool() && !t.isBool() && a[0].byteSize() == t.byteSize() && a[0] != t,
                        "needs equal non-bool sizes");
                break;
            case Op::BitfieldExtract:
            case Op::BitfieldInsert: {
                expectResult(in, true);
                size_t n = in.op == Op::BitfieldInsert ? 2 : 1;
                bool ok = t.isInt();
                for (size_t i = 0; i < n; ++i) ok = ok && a[i] == t;
                ok = ok && a[n] == kU32 && a[n + 1] == kU32;
                require(ok, "needs integer values and u32 offset/count");
                break;
            }
            case Op::Dot:
                expectResult(in, true);
                require(a[0] == a[1] && a[0].isFloat() && a[0].width > 1 && t == a[0].element(), "needs float vectors");
                break;
            case Op::Length:
                expectResult(in, true);
                require(a[0].isFloat() && t == a[0].element(), "needs float operand, scalar result");
                break;
            case Op::Construct: {
                expectResult(in, true);
                unsigned w = 0;
                bool ok = t.width > 1;
                for (Type x : a) {
                    w += x.width;
                    ok = ok && x.scalar == t.scalar;
                }
                require(ok && w == t.width, "components must add up to the result vector");
                break;
            }
            case Op::Extract:
                expectResult(in, true);
                require(a[0].width > 1 && in.lits[0] < a[0].width && t == a[0].element(), "bad vector extract");
                break;
            case Op::Insert:
                expectResult(in, true);
                require(a[0] == t && t.width > 1 && a[1] == t.element() && in.lits[0] < t.width, "bad vector insert");
                break;
            case Op::Shuffle: {
                expectResult(in, true);
                bool ok = a[0].width > 1 && a[0] == a[1] && t.scalar == a[0].scalar && in.lits.size() == t.width && t.width > 1;
                for (uint32_t l : in.lits) ok = ok && l < 2u * a[0].width;
                require(ok, "bad shuffle");
                break;
            }
            case Op::Load:
                expectResult(in, true);
                checkTarget(in, false);
                require(a[0] == kU32, "address must be u32");
                require(t.byteSize() && t.scalar != Scalar::F16, "loads 32-bit scalars or vectors");
                break;
            case Op::Store:
                expectResult(in, false);
                checkTarget(in, true);
                require(a[0] == kU32, "address must be u32");
                require(a[1].byteSize() && a[1].scalar != Scalar::F16, "stores 32-bit scalars or vectors");
                break;
            case Op::AtomicAdd:
            case Op::AtomicSub:
            case Op::AtomicMin:
            case Op::AtomicMax:
            case Op::AtomicAnd:
            case Op::AtomicOr:
            case Op::AtomicXor:
            case Op::AtomicExchange:
            case Op::AtomicCompareExchange:
                expectResult(in, true);
                if (checkTarget(in, true)) {
                    require(in.sym != kTargetPayload, "atomics are not allowed on the payload");
                    require(in.sym != kTargetPrivate && !isPrivateSlot(in.sym), "atomics are not allowed on private memory");
                    if (in.sym < m_.resources.size())
                        require(m_.resources[in.sym].access == Access::ReadWrite, "atomics need a read_write buffer");
                }
                require(a[0] == kU32, "address must be u32");
                require(t.isScalar() && t.isInt(), "atomics operate on i32/u32");
                for (size_t i = 1; i < a.size(); ++i) require(a[i] == t, "value types must match the result");
                break;
            case Op::ImageSample:
            case Op::ImageGather: {
                expectResult(in, true);
                const Resource* r = image(in.sym, false);
                if (!r) break;
                if (!require(in.sampler < m_.resources.size() && m_.resources[in.sampler].kind == ResourceKind::Sampler,
                             "expected a sampler"))
                    break;
                size_t c = 0;
                if (in.op == Op::ImageGather) {
                    require(r->dim != TexDim::D1 && r->dim != TexDim::D3, "gather needs a 2d or cube texture");
                    require(in.lits[0] < 4 && (!r->depth || in.lits[0] == 0), "bad gather component");
                    require(!(in.image & (kImageLod | kImageBias | kImageGrad)), "gather takes no lod");
                    require(t == Type(r->depth ? Scalar::F32 : r->elem, 4), "result must be a 4-vector of texels");
                } else {
                    require(t == texel(*r), r->depth ? "depth sampling returns f32" : "result must be a vec4 of the texel type");
                }
                require(a[c] == coordType(*r, Scalar::F32), "coordinate type does not match the texture");
                checkSampleOperands(in, *r, c + 1, a);
                break;
            }
            case Op::ImageFetch: {
                expectResult(in, true);
                bool storage = in.sym < m_.resources.size() && m_.resources[in.sym].kind == ResourceKind::StorageTexture;
                const Resource* r = image(in.sym, storage);
                if (!r) break;
                require(!storage || r->access != Access::Write, "storage texture is write-only");
                require(in.image == 0 || in.image == kImageLod, "fetch only takes lod");
                require(!storage || in.image == 0, "storage reads take no lod");
                require(a[0] == coordType(*r, Scalar::U32), "coordinate must be u32 vector");
                if (in.image) require(a[1] == kU32, "lod must be u32");
                require(t == texel(*r), "result must be a vec4 of the texel type");
                break;
            }
            case Op::ImageStore: {
                expectResult(in, false);
                const Resource* r = image(in.sym, true);
                if (!r) break;
                require(r->access != Access::Read, "storage texture is read-only");
                require(a[0] == coordType(*r, Scalar::U32), "coordinate must be u32 vector");
                require(a[1] == texel(*r), "value must be a vec4 of the texel type");
                break;
            }
            case Op::ImageAtomicAdd:
            case Op::ImageAtomicSub:
            case Op::ImageAtomicMin:
            case Op::ImageAtomicMax:
            case Op::ImageAtomicAnd:
            case Op::ImageAtomicOr:
            case Op::ImageAtomicXor:
            case Op::ImageAtomicExchange:
            case Op::ImageAtomicCompareExchange: {
                expectResult(in, true);
                const Resource* r = image(in.sym, true);
                if (!r) break;
                require(r->access == Access::ReadWrite, "image atomics need a read_write storage texture");
                require(r->elem == Scalar::I32 || r->elem == Scalar::U32, "image atomics need i32/u32 texels");
                require(r->dim != TexDim::Cube && r->dim != TexDim::CubeArray, "image atomics need a 1d, 2d or 3d texture");
                require(a[0] == coordType(*r, Scalar::U32), "coordinate must be u32 vector");
                require(t == Type(r->elem, 1), "result must be the texel scalar");
                for (size_t i = 1; i < a.size(); ++i) require(a[i] == t, "value types must match the result");
                break;
            }
            case Op::ImageSize: {
                expectResult(in, true);
                bool storage = in.sym < m_.resources.size() && m_.resources[in.sym].kind == ResourceKind::StorageTexture;
                const Resource* r = image(in.sym, storage);
                if (!r) break;
                require(in.image == 0 || (in.image == kImageLod && !storage), "image_size only takes lod on textures");
                if (in.image) require(a[0] == kU32, "lod must be u32");
                require(t == Type(Scalar::U32, uint8_t(sizeWidth(r->dim))), "result must be a u32 vector of the size width");
                break;
            }
            case Op::ImageQueryLod: {
                // vec2(lod as sampled: clamped to the view, lod computed: unclamped), from derivatives
                expectResult(in, true);
                require(m_.stage == Stage::Fragment, "lod queries are fragment-only");
                const Resource* r = image(in.sym, false);
                if (!r) break;
                require(in.sampler < m_.resources.size() && m_.resources[in.sampler].kind == ResourceKind::Sampler, "expected a sampler");
                require(r->dim != TexDim::D1, "lod queries need a 2d, 3d or cube texture");
                require(a.size() == 1 && a[0] == coordType(*r, Scalar::F32), "coordinate type does not match the texture");
                require(t == Type(Scalar::F32, 2), "result must be vec2<f32>");
                break;
            }
            case Op::ImageLevels: {
                expectResult(in, true);
                if (image(in.sym, false)) require(t == kU32, "result must be u32");
                break;
            }
            case Op::InterpCentroid:
            case Op::InterpSample:
            case Op::InterpOffset: {
                expectResult(in, true);
                require(m_.stage == Stage::Fragment, "interpolation is fragment-only");
                if (!entryOnly() || !require(in.sym < m_.inputs.size(), "unknown input")) break;
                const IoVar& v = m_.inputs[in.sym];
                require(v.builtin == Builtin::None && v.interp != Interp::Flat && v.type.isFloat(),
                        "needs an interpolated float input");
                // Per-sample inputs are already evaluated at the sample; re-interpolating them is moot.
                require(v.sampling != Sampling::Sample, "input must use center or centroid sampling");
                require(v.type == t, "type mismatch");
                if (in.op == Op::InterpSample) require(a[0] == kU32, "sample index must be u32");
                if (in.op == Op::InterpOffset) require(a[0] == Type(Scalar::F32, 2), "offset must be vec2<f32>");
                break;
            }
            case Op::FramebufferFetch: {
                expectResult(in, true);
                require(m_.stage == Stage::Fragment, "framebuffer_fetch is fragment-only");
                entryOnly();
                if (!require(in.lits[0] < 8, "attachment index must be 0..7")) break;
                require(t.isNumeric(), "fetches colour attachments only: numeric scalar or vector "
                                       "(depth/stencil attachments cannot be read from tile memory)");
                auto [it, fresh] = fetchTypes_.emplace(in.lits[0], t);
                require(fresh || it->second == t, "every fetch of one attachment must have the same type");
                for (const IoVar& o : m_.outputs)
                    if (o.builtin == Builtin::None && o.location == in.lits[0])
                        require(o.type == t, "type differs from the output at the same location");
                break;
            }
            case Op::Ddx:
            case Op::Ddy:
            case Op::Fwidth:
                unaryOrBinary(t.isFloat(), "needs float operand");
                require(m_.stage == Stage::Fragment, "derivatives are fragment-only");
                break;
            case Op::Barrier:
            case Op::BarrierAll:
                expectResult(in, false);
                require(stageIs(kGridStages), "barriers need a compute, object or mesh stage");
                break;
            case Op::SubgroupBallot:
                expectResult(in, true);
                require(a[0] == kBool && t == Type(Scalar::U32, 4), "ballot takes bool and returns vec4<u32>");
                break;
            case Op::SubgroupBroadcast:
            case Op::SubgroupShuffle:
                expectResult(in, true);
                require(a[0] == t && !t.isBool() && a[1] == kU32, "needs a value and a u32 lane");
                break;
            case Op::SubgroupReadFirst:
                unaryOrBinary(!t.isBool(), "needs a non-bool value");
                break;
            case Op::SubgroupAny:
            case Op::SubgroupAll: unaryOrBinary(t == kBool, "needs bool"); break;
            case Op::QuadShuffle:
                expectResult(in, true);
                require(a[0] == t && !t.isBool() && a[1] == kU32, "needs a value and a u32 lane");
                require(stageIs(kSubgroupStages), "quad ops need a fragment, compute, object or mesh stage");
                break;
            case Op::QuadAny:
                unaryOrBinary(t == kBool, "needs bool");
                require(stageIs(kSubgroupStages), "quad ops need a fragment, compute, object or mesh stage");
                break;
            case Op::MeshDispatch:
                expectResult(in, false);
                require(m_.stage == Stage::Object, "mesh_dispatch is object-only");
                entryOnly();
                require(sameAll(kU32), "grid size must be u32");
                break;
            case Op::MeshVertex:
            case Op::MeshPrimitive: {
                expectResult(in, false);
                if (!require(m_.stage == Stage::Mesh, "mesh-only op") || !entryOnly()) break;
                bool prim = in.op == Op::MeshPrimitive;
                require(a[0] == kU32, "index must be u32");
                std::vector<Type> want;
                for (const IoVar& o : m_.outputs)
                    if (o.perPrimitive == prim) want.push_back(o.type);
                require(std::vector<Type>(a.begin() + 1, a.end()) == want,
                        "values must match the per-vertex/per-primitive outputs in order");
                break;
            }
            case Op::MeshIndex:
            case Op::MeshPrimitiveCount:
                expectResult(in, false);
                require(m_.stage == Stage::Mesh, "mesh-only op");
                entryOnly();
                require(sameAll(kU32), "operands must be u32");
                break;
        }
        if (in.op >= Op::SubgroupBallot && in.op <= Op::SubgroupAll)
            require(stageIs(kSubgroupStages), "subgroup ops need a fragment, compute, object or mesh stage");
    }

    const Module& m_;
    std::vector<std::string> errors_;
    const Function* fn_ = nullptr;
    uint32_t fnIndex_ = 0;
    const Inst* inst_ = nullptr;
    std::vector<bool> everDefined_, visible_;
    std::vector<std::vector<ValueId>> scopes_;
    unsigned loopDepth_ = 0;
    std::map<uint32_t, Type> fetchTypes_;  // framebuffer_fetch type per attachment
};

}  // namespace

std::vector<std::string> verify(const Module& module) { return Verifier(module).run(); }

}  // namespace vpir
