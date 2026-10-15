#include "vpmsl/msl.h"

#include <algorithm>
#include <iterator>
#include <bit>
#include <charconv>
#include <functional>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <string_view>
#include <unordered_map>

#include "vpir/passes.h"
#include "vpir/verify.h"

namespace vpmsl {

using namespace vpir;

namespace {

constexpr const char* kSwizzle = "xyzw";

std::string scalarName(Scalar s) {
    switch (s) {
        case Scalar::Bool: return "bool";
        case Scalar::I32: return "int";
        case Scalar::U32: return "uint";
        case Scalar::F16: return "half";
        case Scalar::F32: return "float";
        case Scalar::Void: break;
    }
    return "void";
}

std::string typeName(Type t) {
    std::string s = scalarName(t.scalar);
    return t.width > 1 ? s + char('0' + t.width) : s;
}

std::string hex(uint32_t v) {
    char buf[16];
    auto r = std::to_chars(buf, buf + sizeof buf, v, 16);
    return "0x" + std::string(buf, r.ptr);
}

std::string literal(Scalar s, uint32_t bits) {
    switch (s) {
        case Scalar::Bool: return bits ? "true" : "false";
        case Scalar::I32: return bits == 0x80000000u ? "int(0x80000000u)" : std::to_string(int32_t(bits));
        case Scalar::U32: return std::to_string(bits) + "u";
        case Scalar::F16:
        case Scalar::F32: {
            bool half = s == Scalar::F16;
            float f = half ? halfToFloat(uint16_t(bits)) : std::bit_cast<float>(bits);
            if (!std::isfinite(f))
                return half ? "as_type<half>(ushort(" + hex(bits) + "))" : "as_type<float>(" + hex(bits) + "u)";
            char buf[32];
            auto r = std::to_chars(buf, buf + sizeof buf, f);
            std::string t(buf, r.ptr);
            if (t.find_first_of(".e") == std::string::npos) t += ".0";
            return t + (half ? "h" : "f");
        }
        case Scalar::Void: break;
    }
    return {};
}

// User names keep their spelling unless they would clash with MSL words or the emitter's own
// names (vp_*, value names vN, local names lN).
std::string ident(const std::string& name) {
    static const std::set<std::string_view> reserved = {
        "auto", "bool", "break", "case", "char", "class", "const", "constant", "constexpr", "continue",
        "default", "delete", "device", "do", "double", "else", "enum", "extern", "false", "float",
        "for", "fragment", "goto", "half", "if", "inline", "int", "kernel", "long", "mesh", "metal",
        "namespace", "new", "object", "object_data", "operator", "private", "public", "ray_data",
        "return", "sampler", "short", "signed", "sizeof", "static", "struct", "switch", "template",
        "this", "thread", "threadgroup", "threadgroup_imageblock", "true", "typedef", "uchar", "uint",
        "ulong", "union", "unsigned", "ushort", "using", "vertex", "void", "volatile", "while"};
    auto numbered = [&](char c) {
        return name.size() > 1 && name[0] == c &&
               std::all_of(name.begin() + 1, name.end(), [](char d) { return d >= '0' && d <= '9'; });
    };
    bool vectorType = name.size() > 1 && name.back() >= '2' && name.back() <= '4' &&
                      reserved.contains(std::string_view(name).substr(0, name.size() - 1));
    if (reserved.contains(name) || vectorType || name.starts_with("vp_") || name.starts_with("__") ||
        numbered('v') || numbered('l'))
        return name + "_";
    return name;
}

const char* builtinAttr(Builtin b) {
    switch (b) {
        case Builtin::Position: return "position";
        case Builtin::PointSize: return "point_size";
        case Builtin::VertexId: return "vertex_id";
        case Builtin::InstanceId: return "instance_id";
        case Builtin::FrontFacing: return "front_facing";
        case Builtin::SampleId: return "sample_id";
        case Builtin::SampleMask: return "sample_mask";
        case Builtin::FragDepth: return "depth(any)";
        case Builtin::PrimitiveId: return "primitive_id";
        case Builtin::Layer: return "render_target_array_index";
        case Builtin::GlobalInvocationId: return "thread_position_in_grid";
        case Builtin::LocalInvocationId: return "thread_position_in_threadgroup";
        case Builtin::WorkgroupId: return "threadgroup_position_in_grid";
        case Builtin::NumWorkgroups: return "threadgroups_per_grid";
        case Builtin::LocalInvocationIndex: return "thread_index_in_threadgroup";
        case Builtin::SubgroupInvocation: return "thread_index_in_simdgroup";
        case Builtin::SubgroupSize: return "threads_per_simdgroup";
        case Builtin::ClipDistance: return "clip_distance";
        case Builtin::BaseVertex: return "base_vertex";
        case Builtin::BaseInstance: return "base_instance";
        case Builtin::DrawIndex: return "buffer(29)";  // a per-draw constant, not a Metal attribute
        case Builtin::None: break;
    }
    return "";
}

// Interpolation qualifier of a fragment input ("" = Metal's default, center_perspective).
std::string interpAttr(const IoVar& v) {
    if (v.interp == Interp::Flat) return "flat";
    if (v.interp == Interp::Perspective && v.sampling == Sampling::Center) return "";
    static const char* sampling[] = {"center", "centroid", "sample"};
    return std::string(sampling[unsigned(v.sampling)]) +
           (v.interp == Interp::Linear ? "_no_perspective" : "_perspective");
}

std::string textureType(const Resource& r) {
    static const char* color[] = {"texture1d", "texture2d", "texture2d_array", "texture3d", "texturecube",
                                  "texturecube_array"};
    static const char* depth[] = {"", "depth2d", "depth2d_array", "", "depthcube", "depthcube_array"};
    std::string t = r.depth ? depth[unsigned(r.dim)] : color[unsigned(r.dim)];
    if (r.multisample) t = std::string(r.depth ? "depth2d_ms" : "texture2d_ms") + (r.dim == TexDim::D2Array ? "_array" : "");
    t += "<" + scalarName(r.elem);
    if (r.kind == ResourceKind::StorageTexture) {
        static const char* access[] = {"read", "write", "read_write"};
        t += std::string(", access::") + access[unsigned(r.access)];
    }
    return t + ">";
}

class Emitter {
public:
    Emitter(const Module& m, uint32_t loopGuard) : m_(m), loopGuard_(loopGuard) {}

    MslShader run() {
        MslShader sh;
        if (m_.stage == Stage::Compute || m_.stage == Stage::Object || m_.stage == Stage::Mesh)
            sh.workgroupSize = {m_.workgroupSize[0], m_.workgroupSize[1], m_.workgroupSize[2]};

        scan(m_.entry()->body);
        // VP_PSZ: a vertex shader's PointSize output attribute. Vulkan lets any
        // pipeline write gl_PointSize; Metal rejects [[point_size]] once the
        // pipeline's inputPrimitiveTopology is not the point class. The backend
        // compiles non-point pipelines with VP_PSZ=user(vp_psz), turning the write
        // into an unconsumed varying; the default keeps the real built-in.
        out_ = "#include <metal_stdlib>\nusing namespace metal;\n"
               "#ifndef VP_PSZ\n#define VP_PSZ point_size\n#endif\n";
        // VP_NO_FDEPTH: Vulkan ignores FragDepth writes when the pass has no depth
        // attachment; Metal rejects a pipeline whose fragment function writes
        // [[depth]] without a depth pixel format. The backend compiles such
        // pipelines with VP_NO_FDEPTH defined, which removes the depth member and
        // every store to it. A fragment shader whose only output is FragDepth then
        // returns void (VP_FOUT / VP_FRET below), since an empty output struct is
        // not a valid fragment return type.
        if (depthOnlyOutput())
            out_ += "#ifdef VP_NO_FDEPTH\n#define VP_FOUT void\n#define VP_FRET return\n#else\n"
                    "#define VP_FOUT VpOut\n#define VP_FRET return vp_out\n#endif\n";
        argumentBuffer(sh);
        interfaceStructs();
        for (const Function& f : m_.functions)
            if (&f != m_.entry()) helper(f);
        entry();
        sh.source = std::move(out_);
        if (m_.stage == Stage::Fragment)
            for (const IoVar& v : m_.outputs)
                if (v.builtin == Builtin::None && v.location < 8) sh.colorOutputs |= uint8_t(1u << v.location);
        return sh;
    }

private:
    // ---- output helpers ----
    template <typename... T>
    void line(const T&... parts) {
        out_.append(depth_ * 4, ' ');
        ((out_ += parts), ...);
        out_ += '\n';
    }

    bool hasLocation(const std::vector<IoVar>& vars, bool perPrimitive = false) const {
        return std::any_of(vars.begin(), vars.end(), [&](const IoVar& v) {
            return v.builtin == Builtin::None && v.perPrimitive == perPrimitive;
        });
    }

    // ---- module-level declarations ----
    void argumentBuffer(MslShader& sh) {
        std::vector<uint32_t> order(m_.resources.size());
        for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            const Resource &x = m_.resources[a], &y = m_.resources[b];
            return std::pair(x.kind, x.index) < std::pair(y.kind, y.index);
        });
        if (order.empty()) return;
        out_ += "\nstruct VpArgs {\n";
        for (uint32_t i : order) {
            const Resource& r = m_.resources[i];
            sh.arguments.push_back({r.name, r.kind, r.index, uint32_t(8 * sh.arguments.size())});
            std::string decl;
            switch (r.kind) {
                case ResourceKind::Buffer:
                    decl = r.access == Access::Read ? "const device uchar*" : "device uchar*";
                    break;
                case ResourceKind::ConstantBuffer: decl = "constant uchar*"; break;
                case ResourceKind::Sampler: decl = "sampler"; break;
                case ResourceKind::Texture:
                case ResourceKind::StorageTexture: decl = textureType(r); break;
                case ResourceKind::BufferSize: decl = "ulong"; break;  // bytes of the bound range
            }
            out_ += "    " + decl + " " + ident(r.name) + ";\n";
        }
        out_ += "};\n";
    }

    void interfaceStructs() {
        Stage s = m_.stage;
        if ((s == Stage::Vertex || s == Stage::Fragment) && hasLocation(m_.inputs)) {
            out_ += "\nstruct VpIn {\n";
            for (const IoVar& v : m_.inputs) {
                if (v.builtin != Builtin::None) continue;
                // A packed varying (Component != 0) is its own user() name: loc<L>_<C>.
                std::string loc = std::to_string(v.location) +
                                  (v.component && s != Stage::Vertex ? "_" + std::to_string(v.component) : "");
                if (s == Stage::Vertex) {
                    out_ += "    " + typeName(v.type) + " " + ident(v.name) + " [[attribute(" + loc + ")]];\n";
                } else if (interpolated_.contains(&v - m_.inputs.data())) {
                    out_ += "    interpolant<" + typeName(v.type) + ", interpolation::" +
                            (v.interp == Interp::Linear ? "no_perspective" : "perspective") + "> " +
                            ident(v.name) + " [[user(loc" + loc + ")]];\n";
                } else {
                    std::string q = interpAttr(v);
                    out_ += "    " + typeName(v.type) + " " + ident(v.name) + " [[user(loc" + loc + ")" +
                            (q.empty() ? "" : ", " + q) + "]];\n";
                }
            }
            out_ += "};\n";
        }
        if (s == Stage::Vertex || s == Stage::Fragment) {
            if (!m_.outputs.empty()) outputStruct("VpOut", false);
        } else if (s == Stage::Mesh) {
            outputStruct("VpVertex", false);
            if (hasPrimitiveOutputs()) outputStruct("VpPrimitive", true);
        }
        if (m_.payloadBytes) out_ += "\nstruct VpPayload {\n    uint words[" + std::to_string(m_.payloadBytes / 4) + "];\n};\n";
    }

    // Fragment shader whose outputs are all FragDepth (depth-only pass).
    bool depthOnlyOutput() const {
        return m_.stage == Stage::Fragment && !m_.outputs.empty() &&
               std::all_of(m_.outputs.begin(), m_.outputs.end(),
                           [](const IoVar& v) { return v.builtin == Builtin::FragDepth; });
    }

    bool isFragDepth(const IoVar& v) const { return m_.stage == Stage::Fragment && v.builtin == Builtin::FragDepth; }
    bool paddedColor(const IoVar& v) const { return m_.stage == Stage::Fragment && v.builtin == Builtin::None; }

    bool hasPrimitiveOutputs() const {
        return std::any_of(m_.outputs.begin(), m_.outputs.end(), [](const IoVar& v) { return v.perPrimitive; });
    }

    void outputStruct(const char* name, bool perPrimitive) {
        out_ += std::string("\nstruct ") + name + " {\n";
        // Clip distances: one [[clip_distance]] array (StoreOutput indexes it by IoVar::location).
        uint32_t clips = 0;
        for (const IoVar& v : m_.outputs)
            if (v.builtin == Builtin::ClipDistance && !perPrimitive) clips = std::max(clips, v.location + 1);
        if (clips) out_ += "    float vp_clip [[clip_distance]] [" + std::to_string(clips) + "];\n";
        for (const IoVar& v : m_.outputs) {
            if (v.perPrimitive != perPrimitive) continue;
            if (v.builtin == Builtin::ClipDistance) continue;
            std::string attr;
            if (v.builtin == Builtin::PointSize && m_.stage == Stage::Vertex) attr = "VP_PSZ";
            else if (v.builtin != Builtin::None) attr = builtinAttr(v.builtin);
            else if (m_.stage == Stage::Fragment)
                attr = "color(" + std::to_string(v.location) + ")" + (v.index ? ", index(1)" : "");
            else attr = "user(loc" + std::to_string(v.location) + (v.component ? "_" + std::to_string(v.component) : "") + ")";
            // Fragment colour outputs are always 4-wide: Metal refuses a pipeline whose output has
            // fewer components than the attachment's format (Vulkan leaves the rest undefined).
            const Type mt = paddedColor(v) ? v.type.withWidth(4) : v.type;
            std::string member = "    " + typeName(mt) + " " + ident(v.name) + " [[" + attr + "]];\n";
            if (isFragDepth(v)) member = "#ifndef VP_NO_FDEPTH\n" + member + "#endif\n";
            out_ += member;
        }
        out_ += "};\n";
    }

    const char* returnOut() const { return depthOnlyOutput() ? "VP_FRET;" : "return vp_out;"; }

    // ---- functions ----
    void helper(const Function& f) {
        std::string sig = "static " + typeName(f.returnType) + " " + helperName(uint32_t(&f - m_.functions.data())) + "(";
        for (size_t i = 0; i < f.params.size(); ++i)
            sig += (i ? ", " : "") + typeName(f.typeOf(f.params[i])) + " v" + std::to_string(f.params[i]);
        out_ += "\n" + sig + ") {\n";
        body(f, false);
        out_ += "}\n";
    }

    void entry() {
        const Function& f = *m_.entry();
        Stage s = m_.stage;
        std::vector<std::string> params;
        bool vsfs = s == Stage::Vertex || s == Stage::Fragment;
        if (vsfs && hasLocation(m_.inputs)) params.push_back("VpIn vp_in [[stage_in]]");
        if (s == Stage::Object) {
            if (m_.payloadBytes) params.push_back("object_data VpPayload& vp_payload [[payload]]");
            params.push_back("mesh_grid_properties vp_grid");
        }
        if (s == Stage::Mesh) {
            static const char* topo[] = {"point", "line", "triangle"};
            params.push_back("metal::mesh<VpVertex, " + std::string(hasPrimitiveOutputs() ? "VpPrimitive" : "void") +
                             ", " + std::to_string(m_.maxVertices) + ", " + std::to_string(m_.maxPrimitives) +
                             ", topology::" + topo[unsigned(m_.topology)] + "> vp_mesh");
            if (m_.payloadBytes) params.push_back("const object_data VpPayload& vp_payload [[payload]]");
        }
        if (!m_.resources.empty()) params.push_back("constant VpArgs& vp_args [[buffer(0)]]");
        for (const IoVar& v : m_.inputs)
            if (v.builtin == Builtin::DrawIndex)
                params.push_back("constant uint& " + builtinParam(v) + " [[" + builtinAttr(v.builtin) + "]]");
            else if (v.builtin != Builtin::None)
                params.push_back(typeName(v.type) + " " + builtinParam(v) + " [[" + builtinAttr(v.builtin) + "]]");
        for (auto [n, t] : fetched_)
            params.push_back(typeName(t) + " vp_fb" + std::to_string(n) + " [[color(" + std::to_string(n) + ")]]");

        std::string head;
        switch (s) {
            case Stage::Vertex: head = std::string("vertex ") + (m_.outputs.empty() ? "void" : "VpOut"); break;
            case Stage::Fragment:
                head = std::string(m_.earlyFragmentTests ? "[[early_fragment_tests]] " : "") + "fragment " +
                       (m_.outputs.empty() ? "void" : depthOnlyOutput() ? "VP_FOUT" : "VpOut");
                break;
            case Stage::Compute:
                head = "[[max_total_threads_per_threadgroup(" +
                       std::to_string(m_.workgroupSize[0] * m_.workgroupSize[1] * m_.workgroupSize[2]) +
                       ")]] kernel void";
                break;
            case Stage::Object: head = "[[object]] void"; break;
            case Stage::Mesh: head = "[[mesh]] void"; break;
        }
        out_ += "\n" + head + " vp_main(";
        for (size_t i = 0; i < params.size(); ++i) out_ += (i ? ",\n        " : "") + params[i];
        out_ += ") {\n";
        body(f, true);
        out_ += "}\n";
    }

    // Inputs read through interp_* need an interpolant<> member; plain loads then interpolate too.
    // Fetched attachments become [[color(n)]] entry parameters.
    void scan(const Block& b) {
        for (const Node& n : b) {
            const Inst& in = n.inst;
            if (n.kind == NodeKind::Inst &&
                (in.op == Op::InterpCentroid || in.op == Op::InterpSample || in.op == Op::InterpOffset))
                interpolated_.insert(in.sym);
            if (n.kind == NodeKind::Inst && in.op == Op::FramebufferFetch) fetched_.emplace(in.lits[0], in.type);
            scan(n.body);
            scan(n.elseBody);
        }
    }

    void countUses(const Block& b) {
        auto use = [&](ValueId v) {
            ++uses_[v];
            useBlock_[v] = &b;
        };
        for (const Node& n : b) {
            if (n.kind == NodeKind::Inst) {
                defs_[n.inst.result] = &n.inst;
                defBlock_[n.inst.result] = &b;
                for (ValueId a : n.inst.args) use(a);
            } else if (n.value != kNoValue) {
                use(n.value);
            }
            countUses(n.body);
            countUses(n.elseBody);
        }
    }

    void body(const Function& f, bool isEntry) {
        fn_ = &f;
        entry_ = isEntry;
        expr_.assign(f.valueTypes.size(), {});
        for (ValueId p : f.params) expr_[p] = "v" + std::to_string(p);
        uses_.clear();
        defs_.clear();
        useBlock_.clear();
        defBlock_.clear();
        countUses(f.body);
        depth_ = 1;
        bool vsfsOut = isEntry && (m_.stage == Stage::Vertex || m_.stage == Stage::Fragment) && !m_.outputs.empty();
        if (vsfsOut) line("VpOut vp_out;");
        if (isEntry && m_.sharedBytes) line("threadgroup uint4 vp_shared[", std::to_string((m_.sharedBytes + 15) / 16), "];");
        if (isEntry && m_.privateBytes) line("uint4 vp_private[", std::to_string((m_.privateBytes + 15) / 16), "] = {};");
        // One array per private variable: Metal's compiler can keep each constant-indexed one in
        // registers; a dynamically indexed one goes to stack memory alone instead of dragging every
        // other local of the shader with it (the shared arena did, 6x slower on some materials).
        // Slots only ever touched at constant addresses with one type per cell go further: one
        // typed local per cell, which Metal's compiler handles much faster than a byte-cast array.
        if (isEntry) {
            planSlots(f.body);
            for (size_t i = 0; i < m_.privateSlots.size(); ++i) {
                auto cells = slotCells_.find(uint32_t(i));
                if (cells == slotCells_.end()) {
                    line("uint4 vp_p", std::to_string(i), "[", std::to_string(m_.privateSlots[i] / 16), "] = {};");
                    continue;
                }
                for (auto [off, ct] : cells->second) line(typeName(ct), " ", cellName(uint32_t(i), off), "{};");
            }
        }
        for (size_t i = 0; i < f.vars.size(); ++i) line(typeName(f.vars[i].type), " l", std::to_string(i), "{};");
        block(f.body);
        if (vsfsOut && (f.body.empty() || f.body.back().kind != NodeKind::Return)) line(returnOut());
    }

    // ---- private slot scalarization ----
    // Constant byte offset of an address value, if it is a Const.
    bool constAddress(ValueId addr, uint32_t& off) const {
        auto d = defs_.find(addr);
        if (d == defs_.end() || d->second->op != Op::Const || d->second->lits.empty()) return false;
        off = d->second->lits[0];
        return true;
    }

    // Collects the entry's private slot accesses. A slot qualifies for per-cell locals when every
    // access has a constant address and the accesses partition into non-overlapping cells: each
    // cell is the widest access at its offset, and any other access must lie entirely inside one
    // cell with the same scalar type at an element boundary (a component or sub-vector, emitted as
    // a swizzle). Anything else (dynamic index, reinterpreting a cell as another type, straddling
    // two cells) keeps the byte-addressed array.
    void planSlots(const Block& body) {
        slotCells_.clear();
        std::map<uint32_t, std::vector<std::pair<uint32_t, Type>>> accesses;
        std::set<uint32_t> rejected;
        std::function<void(const Block&)> walk = [&](const Block& b) {
            for (const Node& n : b) {
                const Inst& in = n.inst;
                if (n.kind == NodeKind::Inst && isPrivateSlot(in.sym) &&
                    (in.op == Op::Load || in.op == Op::Store)) {
                    uint32_t slot = privateSlotIndex(in.sym), off = 0;
                    Type t = in.op == Op::Load ? in.type : fn_->typeOf(in.args[1]);
                    if (!constAddress(in.args[0], off) || t.byteSize() == 0) rejected.insert(slot);
                    else accesses[slot].emplace_back(off, t);
                } else if (n.kind == NodeKind::Inst && isPrivateSlot(in.sym) &&
                           std::string_view(opInfo(in.op).signature).find('m') != std::string_view::npos) {
                    // Any other memory op on the slot (atomics): keep the byte array it addresses.
                    rejected.insert(privateSlotIndex(in.sym));
                }
                walk(n.body);
                walk(n.elseBody);
            }
        };
        walk(body);
        for (auto& [slot, list] : accesses) {
            if (rejected.contains(slot)) continue;
            // By offset, widest first: a cell is opened by the first access past the previous cell.
            std::sort(list.begin(), list.end(), [](const auto& x, const auto& y) {
                return x.first != y.first ? x.first < y.first : x.second.byteSize() > y.second.byteSize();
            });
            std::map<uint32_t, Type>& cells = slotCells_[slot];
            uint64_t start = 0, end = 0;
            Type cur;
            for (auto [off, t] : list) {
                if (off >= end) {
                    cells.emplace(off, t);
                    start = off;
                    end = uint64_t(off) + t.byteSize();
                    cur = t;
                    continue;
                }
                unsigned es = cur.element().byteSize();
                bool inside = t.scalar == cur.scalar && (off - start) % es == 0 && off + t.byteSize() <= end;
                if (!inside) {
                    rejected.insert(slot);
                    break;
                }
            }
        }
        for (uint32_t slot : rejected) slotCells_.erase(slot);
        // Slots with no access at all need no storage.
        for (uint32_t i = 0; i < m_.privateSlots.size(); ++i)
            if (!rejected.contains(i)) slotCells_.try_emplace(i);
    }

    static std::string cellName(uint32_t slot, uint32_t off) {
        return "vp_p" + std::to_string(slot) + "_" + std::to_string(off);
    }

    // Lvalue of a scalarized slot access (the cell, or a swizzle of it), or "" for array-backed
    // memory.
    std::string cell(uint32_t target, ValueId addr, Type t) const {
        if (!entry_ || !isPrivateSlot(target)) return {};
        auto cells = slotCells_.find(privateSlotIndex(target));
        uint32_t off = 0;
        if (cells == slotCells_.end() || !constAddress(addr, off)) return {};
        auto it = cells->second.upper_bound(off);
        if (it == cells->second.begin()) return {};  // (planSlots guarantees a containing cell)
        --it;
        std::string name = cellName(privateSlotIndex(target), it->first);
        if (it->second == t) return name;
        unsigned first = (off - it->first) / it->second.element().byteSize();
        name += '.';
        for (unsigned k = 0; k < t.width; ++k) name += kSwizzle[first + k];
        return name;
    }

    void block(const Block& b) {
        for (const Node& n : b) node(n);
    }

    void nested(const Block& b) {
        ++depth_;
        block(b);
        --depth_;
    }

    void node(const Node& n) {
        switch (n.kind) {
            case NodeKind::Inst: inst(n.inst); break;
            case NodeKind::If:
                line("if (", ex(n.value), ") {");
                nested(n.body);
                if (!n.elseBody.empty()) {
                    line("} else {");
                    nested(n.elseBody);
                }
                line("}");
                break;
            case NodeKind::Loop: {
                // Hang guard: a loop whose exit depends on data the translation got wrong (an empty
                // texture, a constant read from the wrong place) would spin until the GPU watchdog
                // resets the whole GPU (kIOGPUCommandBufferCallbackErrorHang), taking every other
                // client's work with it. Past loopGuard_ trips (a `continue` counts too) it breaks:
                // wrong pixels instead of a lost device.
                if (!loopGuard_) {
                    line("while (true) {");
                    nested(n.body);
                    line("}");
                    break;
                }
                const std::string guard = "vp_guard" + std::to_string(loops_++);
                line("uint ", guard, " = 0u;");
                line("while (true) {");
                ++depth_;
                line("if (++", guard, " > ", std::to_string(loopGuard_), "u) break;");
                --depth_;
                nested(n.body);
                line("}");
                break;
            }
            case NodeKind::Break: line("break;"); break;
            case NodeKind::Continue: line("continue;"); break;
            case NodeKind::Discard: line("discard_fragment();"); break;
            case NodeKind::Return:
                if (n.value != kNoValue) line("return ", ex(n.value), ";");
                else if (entry_ && (m_.stage == Stage::Vertex || m_.stage == Stage::Fragment) && !m_.outputs.empty())
                    line(returnOut());
                else line("return;");
                break;
        }
    }

    // ---- values ----
    const std::string& ex(ValueId v) const { return expr_[v]; }
    std::string name(ValueId v) const { return "v" + std::to_string(v); }

    void define(const Inst& in, const std::string& e) {
        // Dead values vanish; dead calls and atomics still run for their side effects.
        if (!uses_.contains(in.result)) {
            if (in.op == Op::Call || (in.op >= Op::AtomicAdd && in.op <= Op::AtomicExchange)) line(e, ";");
            return;
        }
        line("const ", typeName(in.type), " ", name(in.result), " = ", e, ";");
        expr_[in.result] = name(in.result);
    }

    // A name or member access: repeating it costs nothing.
    static bool simple(const std::string& e) {
        return std::all_of(e.begin(), e.end(), [](char c) { return std::isalnum(uint8_t(c)) || c == '_' || c == '.'; });
    }

    // Inlines cheap, pure expressions instead of naming them.
    void inlineAs(const Inst& in, std::string e) { expr_[in.result] = std::move(e); }

    std::string args(const Inst& in, size_t from = 0, size_t count = SIZE_MAX) const {
        std::string s;
        for (size_t i = from; i < in.args.size() && i - from < count; ++i) s += (i > from ? ", " : "") + ex(in.args[i]);
        return s;
    }

    std::string call(const char* fn, const Inst& in) const { return std::string(fn) + "(" + args(in) + ")"; }
    std::string bin(const char* op, const Inst& in) const { return ex(in.args[0]) + " " + op + " " + ex(in.args[1]); }
    std::string math(const char* fn, const Inst& in) const {
        return (in.precise ? std::string("precise::") : std::string()) + call(fn, in);
    }

    std::string one(Type t) const { return t.scalar == Scalar::F16 ? "1.0h" : "1.0f"; }

    // Highest power of two (capped at 16) known to divide a value; 1 when nothing is known.
    unsigned alignment(ValueId v) const {
        auto it = defs_.find(v);
        if (it == defs_.end()) return 1;
        const Inst& in = *it->second;
        auto lit = [&](ValueId a, uint32_t& out) {
            auto d = defs_.find(a);
            if (d == defs_.end() || d->second->op != Op::Const) return false;
            out = d->second->lits[0];
            return true;
        };
        uint32_t c = 0;
        switch (in.op) {
            case Op::Const: return in.lits[0] ? std::min(16u, in.lits[0] & -in.lits[0]) : 16;
            case Op::Add: return std::min(alignment(in.args[0]), alignment(in.args[1]));
            case Op::Mul: return std::min(16u, alignment(in.args[0]) * alignment(in.args[1]));
            case Op::Shl:
                if (lit(in.args[1], c)) return std::min(16u, alignment(in.args[0]) << std::min(c, 4u));
                return 1;
            default: return 1;
        }
    }

    struct Memory {
        std::string space;  // address space qualifier
        std::string base;   // byte pointer expression
    };

    Memory memory(uint32_t target) const {
        if (target == kTargetShared) return {"threadgroup", "(threadgroup uchar*)vp_shared"};
        if (target == kTargetPrivate) return {"thread", "(thread uchar*)vp_private"};
        if (isPrivateSlot(target)) return {"thread", "(thread uchar*)vp_p" + std::to_string(privateSlotIndex(target))};
        if (target == kTargetPayload)
            return m_.stage == Stage::Object ? Memory{"object_data", "(object_data uchar*)&vp_payload"}
                                             : Memory{"const object_data", "(const object_data uchar*)&vp_payload"};
        const Resource& r = m_.resources[target];
        std::string base = "vp_args." + ident(r.name);
        if (r.kind == ResourceKind::ConstantBuffer) return {"constant", base};
        return {r.access == Access::Read ? "const device" : "device", base};
    }

    // See the alignment contract in msl.h.
    unsigned baseAlignment(uint32_t target) const {
        if (target == kTargetShared || target == kTargetPayload || target == kTargetPrivate || isPrivateSlot(target))
            return 16;
        return m_.resources[target].kind == ResourceKind::ConstantBuffer ? 16 : 4;
    }

    std::string address(const Memory& mem, ValueId addr) const {
        auto d = defs_.find(addr);
        if (d != defs_.end() && d->second->op == Op::Const && d->second->lits[0] == 0) return mem.base;
        return mem.base + " + " + ex(addr);
    }

    // Dereferenceable lvalue for a typed access. Unaligned vectors go through packed types.
    std::string access(uint32_t target, ValueId addr, Type t, bool& packed) const {
        Memory mem = memory(target);
        // Addresses are 4-aligned by contract, whatever the analysis can prove.
        unsigned align = std::min(baseAlignment(target), std::max(4u, alignment(addr)));
        packed = t.width == 3 || (t.width > 1 && align < t.byteSize());
        std::string type = packed ? "packed_" + typeName(t) : typeName(t);
        return "*(" + mem.space + " " + type + "*)(" + address(mem, addr) + ")";
    }

    std::string atomicPtr(const Inst& in) const {
        Memory mem = memory(in.sym);
        return "(" + mem.space + " atomic_" + scalarName(in.type.scalar) + "*)(" + address(mem, in.args[0]) + ")";
    }

    // Texture coordinates split into Metal's (coord, [face], [array]) arguments.
    std::string coords(const Resource& r, ValueId v, bool integer) const {
        const std::string& c = ex(v);
        auto layer = [&](char comp) {
            std::string e = c + "." + comp;
            return integer ? e : "uint(rint(" + e + "))";
        };
        switch (r.dim) {
            case TexDim::D2Array: return c + ".xy, " + layer('z');
            case TexDim::Cube: return integer ? c + ".xy, " + c + ".z" : c;
            case TexDim::CubeArray:
                return integer ? c + ".xy, " + c + ".z, " + c + ".w" : c + ".xyz, " + layer('w');
            default: return c;
        }
    }

    std::string texture(uint32_t sym) const { return "vp_args." + ident(m_.resources[sym].name); }

    std::string sampleCall(const Inst& in) const {
        const Resource& r = m_.resources[in.sym];
        bool gather = in.op == Op::ImageGather;
        size_t i = 1;  // optional operands follow the coordinate
        std::string lod, offset, compare;
        static const char* grad[] = {"", "gradient2d", "gradient2d", "gradient3d", "gradientcube", "gradientcube"};
        if (in.image & kImageLod) lod = "level(" + ex(in.args[i++]) + ")";
        if (in.image & kImageBias) lod = "bias(" + ex(in.args[i++]) + ")";
        if (in.image & kImageGrad) {
            lod = std::string(grad[unsigned(r.dim)]) + "(" + ex(in.args[i]) + ", " + ex(in.args[i + 1]) + ")";
            i += 2;
        }
        if (in.image & kImageOffset) offset = ex(in.args[i++]);
        if (in.image & kImageCompare) compare = ex(in.args[i++]);

        std::vector<std::string> a = {"vp_args." + ident(m_.resources[in.sampler].name), coords(r, in.args[0], false)};
        if (!compare.empty()) a.push_back(compare);
        if (!lod.empty()) a.push_back(lod);
        bool cube = r.dim == TexDim::Cube || r.dim == TexDim::CubeArray;
        if (gather && !r.depth && !cube) {
            a.push_back(offset.empty() ? "int2(0)" : offset);  // Metal wants an offset before the component
            a.push_back(std::string("component::") + kSwizzle[in.lits[0]]);
        } else {
            if (!offset.empty()) a.push_back(offset);
            if (gather && !r.depth) a.push_back(std::string("component::") + kSwizzle[in.lits[0]]);
        }
        const char* fn = gather ? (compare.empty() ? "gather" : "gather_compare")
                                : (compare.empty() ? "sample" : "sample_compare");
        std::string s = texture(in.sym) + "." + fn + "(";
        for (size_t k = 0; k < a.size(); ++k) s += (k ? ", " : "") + a[k];
        return s + ")";
    }

    // Texture atomics (Metal 3.1) return the previous texel as a 4-vector: its x is the value. An
    // unused result is a plain statement.
    void imageAtomic(const char* fn, const Inst& in) {
        const Resource& r = m_.resources[in.sym];
        std::string call = texture(in.sym) + "." + fn + "(" + coords(r, in.args[0], true) + ", " + ex(in.args[1]) + ")";
        if (!uses_.contains(in.result)) return line(call, ";");
        define(in, call + ".x");
    }

    std::string imageSize(const Inst& in) const {
        const Resource& r = m_.resources[in.sym];
        std::string lod = in.image ? ex(in.args[0]) : "";
        std::string t = texture(in.sym);
        std::vector<std::string> parts = {t + ".get_width(" + lod + ")"};
        if (in.type.width > 1) parts.push_back(t + ".get_height(" + lod + ")");
        if (in.type.width > 2)
            parts.push_back(r.dim == TexDim::D3 ? t + ".get_depth(" + lod + ")" : t + ".get_array_size()");
        if (parts.size() == 1) return parts[0];
        std::string s = typeName(in.type) + "(";
        for (size_t k = 0; k < parts.size(); ++k) s += (k ? ", " : "") + parts[k];
        return s + ")";
    }

    // Builtin inputs become entry parameters and helpers free functions; the prefixes keep both
    // apart from each other, from MSL's functions and from the emitter's vp_* names.
    static std::string builtinParam(const IoVar& v) { return "in_" + v.name; }
    std::string helperName(uint32_t f) const { return "fn_" + m_.functions[f].name; }

    std::string inputExpr(uint32_t i) const {
        const IoVar& v = m_.inputs[i];
        if (v.builtin != Builtin::None) return builtinParam(v);
        std::string e = "vp_in." + ident(v.name);
        if (interpolated_.contains(i))
            return e + (v.sampling == Sampling::Centroid ? ".interpolate_at_centroid()" : ".interpolate_at_center()");
        return e;
    }

    std::string meshAggregate(const char* type, const Inst& in) const {
        return std::string(type) + "{" + args(in, 1) + "}";
    }

    void inst(const Inst& in) {
        const Type t = in.type;
        auto a = [&](size_t i) -> const std::string& { return ex(in.args[i]); };
        switch (in.op) {
            case Op::Const: {
                if (t.width == 1) return inlineAs(in, literal(t.scalar, in.lits[0]));
                bool splat = std::all_of(in.lits.begin(), in.lits.end(), [&](uint32_t l) { return l == in.lits[0]; });
                std::string s = typeName(t) + "(";
                for (size_t i = 0; i < (splat ? 1 : in.lits.size()); ++i)
                    s += (i ? ", " : "") + literal(t.scalar, in.lits[i]);
                s += ")";
                return uses_[in.result] > 1 ? define(in, s) : inlineAs(in, s);
            }
            case Op::LoadVar: return define(in, "l" + std::to_string(in.sym));
            case Op::StoreVar: return line("l", std::to_string(in.sym), " = ", a(0), ";");
            case Op::LoadInput: {
                std::string e = inputExpr(in.sym);
                return simple(e) ? inlineAs(in, e) : define(in, e);
            }
            case Op::StoreOutput:
                if (isFragDepth(m_.outputs[in.sym])) {
                    // Preprocessor lines start at column 0; the store keeps its indentation.
                    out_ += "#ifndef VP_NO_FDEPTH\n";
                    line("vp_out.", ident(m_.outputs[in.sym].name), " = ", a(0), ";");
                    out_ += "#endif\n";
                    return;
                }
                if (m_.outputs[in.sym].builtin == Builtin::ClipDistance)
                    return line("vp_out.vp_clip[", std::to_string(m_.outputs[in.sym].location), "] = ", a(0), ";");
                if (const IoVar& v = m_.outputs[in.sym]; paddedColor(v) && v.type.width < 4) {
                    // (x[, y[, z]]) -> (x, y, z, 1): zeros, alpha 1.
                    std::string pad;
                    for (unsigned i = v.type.width; i < 4; ++i)
                        pad += ", " + literal(v.type.scalar, i != 3 ? 0u
                                                       : v.type.scalar == Scalar::F16 ? 0x3C00u
                                                       : v.type.isFloat() ? std::bit_cast<uint32_t>(1.0f) : 1u);
                    return line("vp_out.", ident(v.name), " = ", typeName(v.type.withWidth(4)), "(", a(0), pad, ");");
                }
                return line("vp_out.", ident(m_.outputs[in.sym].name), " = ", a(0), ";");
            case Op::Call: {
                std::string c = helperName(in.sym) + "(" + args(in) + ")";
                return t.isVoid() ? line(c, ";") : define(in, c);
            }
            case Op::Neg: return define(in, a(0)[0] == '-' ? "-(" + a(0) + ")" : "-" + a(0));
            case Op::Abs: return define(in, call("abs", in));
            case Op::Not: return define(in, (t.isBool() ? "!" : "~") + a(0));
            case Op::Add: return define(in, bin("+", in));
            case Op::Sub: return define(in, bin("-", in));
            case Op::Mul: return define(in, bin("*", in));
            case Op::Div:
                return define(in, t.isFloat() && in.precise ? call("precise::divide", in) : bin("/", in));
            case Op::Rem: return define(in, t.isFloat() ? math("fmod", in) : bin("%", in));
            case Op::Min: return define(in, call("min", in));
            case Op::Max: return define(in, call("max", in));
            case Op::And: return define(in, bin(t.isBool() ? "&&" : "&", in));
            case Op::Or: return define(in, bin(t.isBool() ? "||" : "|", in));
            case Op::Xor: return define(in, bin(t.isBool() ? "!=" : "^", in));
            case Op::Shl: return define(in, bin("<<", in));
            case Op::Shr: return define(in, bin(">>", in));
            case Op::Fma: return define(in, call("fma", in));  // single rounding in either mode
            case Op::Clamp: return define(in, call("clamp", in));
            case Op::Mix: return define(in, call("mix", in));
            case Op::Eq: return define(in, bin("==", in));
            case Op::Ne: return define(in, bin("!=", in));
            case Op::Lt: return define(in, bin("<", in));
            case Op::Le: return define(in, bin("<=", in));
            case Op::Gt: return define(in, bin(">", in));
            case Op::Ge: return define(in, bin(">=", in));
            case Op::IsNan:
            case Op::IsInf: {
                // Bit tests: fast math lets the compiler assume no NaN/Inf and fold isnan()/isinf().
                Type src = fn_->typeOf(in.args[0]);
                std::string f32 = src.scalar == Scalar::F16 ? typeName(src.withScalar(Scalar::F32)) + "(" + a(0) + ")" : a(0);
                std::string bits = "(as_type<" + typeName(src.withScalar(Scalar::U32)) + ">(" + f32 + ") & 0x7fffffffu)";
                return define(in, bits + (in.op == Op::IsNan ? " > " : " == ") + "0x7f800000u");
            }
            case Op::Select:
                if (fn_->typeOf(in.args[0]).width == 1) return define(in, a(0) + " ? " + a(1) + " : " + a(2));
                return define(in, "select(" + a(2) + ", " + a(1) + ", " + a(0) + ")");
            case Op::Convert: return define(in, typeName(t) + "(" + a(0) + ")");
            case Op::Bitcast: return define(in, "as_type<" + typeName(t) + ">(" + a(0) + ")");
            case Op::BitfieldExtract: return define(in, call("extract_bits", in));
            case Op::BitfieldInsert: return define(in, call("insert_bits", in));
            case Op::BitCount: return define(in, call("popcount", in));
            case Op::BitReverse: return define(in, call("reverse_bits", in));
            case Op::FindLsb: return define(in, call("ctz", in));
            case Op::FindMsb: return define(in, "31" + std::string(t.scalar == Scalar::U32 ? "u" : "") + " - clz(" + a(0) + ")");
            case Op::Floor: return define(in, call("floor", in));
            case Op::Ceil: return define(in, call("ceil", in));
            case Op::Trunc: return define(in, call("trunc", in));
            case Op::Round: return define(in, call("rint", in));
            case Op::Fract: return define(in, math("fract", in));
            case Op::Sqrt: return define(in, math("sqrt", in));
            case Op::Rsq: return define(in, math("rsqrt", in));
            case Op::Rcp:
                return define(in, in.precise ? "precise::divide(" + one(t) + ", " + a(0) + ")" : one(t) + " / " + a(0));
            case Op::Exp2: return define(in, math("exp2", in));
            case Op::Log2: return define(in, math("log2", in));
            case Op::Sin: return define(in, math("sin", in));
            case Op::Cos: return define(in, math("cos", in));
            case Op::Saturate: return define(in, call("saturate", in));
            case Op::Pow: return define(in, math("pow", in));
            case Op::Dot: return define(in, call("dot", in));
            case Op::Length:
                if (fn_->typeOf(in.args[0]).width == 1) return define(in, call("abs", in));
                return define(in, in.precise ? "precise::sqrt(dot(" + a(0) + ", " + a(0) + "))" : call("length", in));
            case Op::Normalize:
                return define(in, in.precise ? a(0) + " * precise::rsqrt(dot(" + a(0) + ", " + a(0) + "))"
                                             : call("normalize", in));
            case Op::Construct: {
                bool splat = std::all_of(in.args.begin(), in.args.end(), [&](ValueId v) { return v == in.args[0]; }) &&
                             fn_->typeOf(in.args[0]).width == 1;
                return define(in, typeName(t) + "(" + (splat ? a(0) : args(in)) + ")");
            }
            case Op::Extract: {
                // Swizzles of names are free to repeat; swizzles of expressions are named once.
                std::string e = a(0) + "." + kSwizzle[in.lits[0]];
                return simple(a(0)) ? inlineAs(in, e) : define(in, e);
            }
            case Op::Insert:
                line(typeName(t), " ", name(in.result), " = ", a(0), ";");
                line(name(in.result), ".", std::string(1, kSwizzle[in.lits[0]]), " = ", a(1), ";");
                expr_[in.result] = name(in.result);
                return;
            case Op::Shuffle: {
                unsigned w = fn_->typeOf(in.args[0]).width;
                bool fromFirst = std::all_of(in.lits.begin(), in.lits.end(), [&](uint32_t l) { return l < w; });
                if (fromFirst) {
                    std::string sw;
                    for (uint32_t l : in.lits) sw += kSwizzle[l];
                    return define(in, a(0) + "." + sw);
                }
                std::string s = typeName(t) + "(";
                for (size_t i = 0; i < in.lits.size(); ++i) {
                    uint32_t l = in.lits[i];
                    s += (i ? ", " : "") + (l < w ? a(0) : a(1)) + "." + kSwizzle[l % w];
                }
                return define(in, s + ")");
            }
            case Op::Load: {
                if (std::string c = cell(in.sym, in.args[0], t); !c.empty()) return define(in, c);
                bool packed = false;
                std::string e = access(in.sym, in.args[0], t, packed);
                if (packed) e = typeName(t) + "(" + e + ")";
                // Constant-buffer reads are pure: fold a single use in the same block into its consumer
                // (never into a nested block, where it could land inside a loop).
                bool cb = in.sym < m_.resources.size() && m_.resources[in.sym].kind == ResourceKind::ConstantBuffer;
                if (cb && uses_[in.result] == 1 && useBlock_[in.result] == defBlock_[in.result])
                    return inlineAs(in, packed ? e : "(" + e + ")");
                return define(in, e);
            }
            case Op::Store: {
                if (std::string c = cell(in.sym, in.args[0], fn_->typeOf(in.args[1])); !c.empty())
                    return line(c, " = ", a(1), ";");
                bool packed = false;
                return line(access(in.sym, in.args[0], fn_->typeOf(in.args[1]), packed), " = ", a(1), ";");
            }
            case Op::AtomicAdd: return define(in, atomicOp("add", in));
            case Op::AtomicSub: return define(in, atomicOp("sub", in));
            case Op::AtomicMin: return define(in, atomicOp("min", in));
            case Op::AtomicMax: return define(in, atomicOp("max", in));
            case Op::AtomicAnd: return define(in, atomicOp("and", in));
            case Op::AtomicOr: return define(in, atomicOp("or", in));
            case Op::AtomicXor: return define(in, atomicOp("xor", in));
            case Op::AtomicExchange:
                return define(in, "atomic_exchange_explicit(" + atomicPtr(in) + ", " + a(1) + ", memory_order_relaxed)");
            case Op::AtomicCompareExchange: {
                // Metal only has the weak form; retry spurious failures so the result is exact.
                std::string v = name(in.result);
                line(typeName(t), " ", v, ";");
                line("do {");
                line("    ", v, " = ", a(1), ";");
                line("} while (!atomic_compare_exchange_weak_explicit(", atomicPtr(in), ", &", v, ", ", a(2),
                     ", memory_order_relaxed, memory_order_relaxed) && ", v, " == ", a(1), ");");
                expr_[in.result] = v;
                return;
            }
            case Op::ImageSample:
            case Op::ImageGather: return define(in, sampleCall(in));
            case Op::ImageFetch: {
                const Resource& r = m_.resources[in.sym];
                std::string s = texture(in.sym) + ".read(" + coords(r, in.args[0], true);
                if (in.image) s += ", " + a(1);
                return define(in, s + ")");
            }
            case Op::ImageStore:
                return line(texture(in.sym), ".write(", a(1), ", ", coords(m_.resources[in.sym], in.args[0], true), ");");
            case Op::ImageSize: return define(in, imageSize(in));
            case Op::ImageLevels: return define(in, texture(in.sym) + ".get_num_mip_levels()");
            case Op::ImageSamples: return define(in, texture(in.sym) + ".get_num_samples()");
            case Op::BufferSize: return define(in, "uint(" + texture(in.sym) + ")");
            case Op::ImageAtomicAdd: return imageAtomic("atomic_fetch_add", in);
            case Op::ImageAtomicSub: return imageAtomic("atomic_fetch_sub", in);
            case Op::ImageAtomicMin: return imageAtomic("atomic_fetch_min", in);
            case Op::ImageAtomicMax: return imageAtomic("atomic_fetch_max", in);
            case Op::ImageAtomicAnd: return imageAtomic("atomic_fetch_and", in);
            case Op::ImageAtomicOr: return imageAtomic("atomic_fetch_or", in);
            case Op::ImageAtomicXor: return imageAtomic("atomic_fetch_xor", in);
            case Op::ImageAtomicExchange: return imageAtomic("atomic_exchange", in);
            case Op::ImageAtomicCompareExchange: {
                // (texel-wide expected value; retried on spurious failures, as atomic_compare_exchange)
                std::string v = name(in.result), e = v + "_e";
                const Resource& r = m_.resources[in.sym];
                line(typeName(t), " ", v, ";");
                line("vec<", scalarName(r.elem), ", 4> ", e, ";");
                line("do {");
                line("    ", e, " = ", a(1), ";");
                line("} while (!", texture(in.sym), ".atomic_compare_exchange_weak(", coords(r, in.args[0], true), ", &", e,
                     ", ", a(2), ") && ", e, ".x == ", a(1), ");");
                line(v, " = ", e, ".x;");
                expr_[in.result] = v;
                return;
            }
            case Op::ImageQueryLod: {
                // (Metal takes the coordinate without the array layer)
                const Resource& r = m_.resources[in.sym];
                std::string c = a(0);
                if (r.dim == TexDim::D2Array) c += ".xy";
                if (r.dim == TexDim::CubeArray) c += ".xyz";
                std::string t = texture(in.sym), s = "vp_args." + ident(m_.resources[in.sampler].name);
                return define(in, "float2(" + t + ".calculate_clamped_lod(" + s + ", " + c + "), " + t +
                                      ".calculate_unclamped_lod(" + s + ", " + c + "))");
            }
            case Op::InterpCentroid: return define(in, "vp_in." + ident(m_.inputs[in.sym].name) + ".interpolate_at_centroid()");
            case Op::InterpSample:
                return define(in, "vp_in." + ident(m_.inputs[in.sym].name) + ".interpolate_at_sample(" + a(0) + ")");
            case Op::InterpOffset:
                // VPIR offsets are from the pixel centre; Metal's are from its top-left corner.
                return define(in, "vp_in." + ident(m_.inputs[in.sym].name) + ".interpolate_at_offset(" + a(0) +
                                      " + 0.5f)");
            case Op::FramebufferFetch: return inlineAs(in, "vp_fb" + std::to_string(in.lits[0]));
            case Op::Ddx: return define(in, call("dfdx", in));
            case Op::Ddy: return define(in, call("dfdy", in));
            case Op::Fwidth: return define(in, call("fwidth", in));
            case Op::Barrier: return line("threadgroup_barrier(mem_flags::mem_threadgroup);");
            case Op::BarrierAll:
                return line("threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture);");
            case Op::SubgroupBallot:
                return define(in, "uint4(as_type<uint2>(simd_vote::vote_t(simd_ballot(" + a(0) + "))), 0u, 0u)");
            case Op::SubgroupBroadcast: return define(in, "simd_broadcast(" + a(0) + ", ushort(" + a(1) + "))");
            case Op::SubgroupShuffle: return define(in, "simd_shuffle(" + a(0) + ", ushort(" + a(1) + "))");
            case Op::SubgroupReadFirst: return define(in, call("simd_broadcast_first", in));
            case Op::SubgroupAny: return define(in, call("simd_any", in));
            case Op::SubgroupAll: return define(in, call("simd_all", in));
            case Op::QuadShuffle: return define(in, "quad_shuffle(" + a(0) + ", ushort(" + a(1) + "))");
            case Op::QuadAny: return define(in, call("quad_any", in));
            case Op::MeshDispatch: return line("vp_grid.set_threadgroups_per_grid(uint3(", args(in), "));");
            case Op::MeshVertex: return line("vp_mesh.set_vertex(", a(0), ", ", meshAggregate("VpVertex", in), ");");
            case Op::MeshPrimitive:
                return line("vp_mesh.set_primitive(", a(0), ", ", meshAggregate("VpPrimitive", in), ");");
            case Op::MeshIndex: return line("vp_mesh.set_index(", a(0), ", uchar(", a(1), "));");
            case Op::MeshPrimitiveCount: return line("vp_mesh.set_primitive_count(", a(0), ");");
        }
    }

    std::string atomicOp(const char* op, const Inst& in) const {
        return std::string("atomic_fetch_") + op + "_explicit(" + atomicPtr(in) + ", " + ex(in.args[1]) +
               ", memory_order_relaxed)";
    }

    const Module& m_;
    std::string out_;
    unsigned depth_ = 0;
    unsigned loops_ = 0;  // loop guard counters emitted so far (names vp_guardN)
    // Iterations a loop may run per invocation before the hang guard breaks it (EmitOptions::loopGuard;
    // default 8192: far above real game loops — lights, kernel taps, bones; a dispatch loop counts one
    // per block run — and low enough that a runaway loop over a full-screen pass finishes before the
    // GPU watchdog fires). 0 = no guard.
    uint32_t loopGuard_ = 8192;
    const Function* fn_ = nullptr;
    bool entry_ = false;
    std::set<uint32_t> interpolated_;  // input indices read through interpolant<>
    std::map<uint32_t, Type> fetched_;  // framebuffer_fetch attachment -> type
    std::vector<std::string> expr_;    // per value: its name or inlined expression
    std::unordered_map<ValueId, unsigned> uses_;
    std::unordered_map<ValueId, const Inst*> defs_;
    std::unordered_map<ValueId, const Block*> defBlock_, useBlock_;
    std::map<uint32_t, std::map<uint32_t, Type>> slotCells_;  // scalarized slot -> offset -> cell type
};

bool usesSubgroups(const Block& b) {
    return std::any_of(b.begin(), b.end(), [](const Node& n) {
        bool op = n.kind == NodeKind::Inst && n.inst.op >= Op::SubgroupBallot && n.inst.op <= Op::SubgroupAll;
        return op || usesSubgroups(n.body) || usesSubgroups(n.elseBody);
    });
}

// Lane ids and ballots of a wave-64 module cannot be expressed on 32-wide hardware.
bool usesSubgroups(const Module& m) {
    bool builtin = std::any_of(m.inputs.begin(), m.inputs.end(), [](const IoVar& v) {
        return v.builtin == Builtin::SubgroupInvocation || v.builtin == Builtin::SubgroupSize;
    });
    return builtin || std::any_of(m.functions.begin(), m.functions.end(),
                                  [](const Function& f) { return usesSubgroups(f.body); });
}

}  // namespace

EmitResult emitMsl(const Module& module, const EmitOptions& options) {
    EmitResult r;
    r.errors = verify(module);
    if (module.waveSize != 32 && usesSubgroups(module))
        r.errors.push_back("wave_size 64: Apple SIMD groups are 32 lanes; lower subgroup ops to wave 32 first");
    if (!r.errors.empty()) return r;
    Module m = module;
    if (options.positionalNames) {
        // Same order as argumentBuffer(): (kind, slot).
        static const char* prefix[] = {"b", "c", "t", "s", "u", "bs"};  // by ResourceKind
        static_assert(std::size(prefix) == size_t(ResourceKind::BufferSize) + 1);
        std::vector<size_t> order(m.resources.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return std::pair(m.resources[a].kind, m.resources[a].index) < std::pair(m.resources[b].kind, m.resources[b].index);
        });
        uint32_t next[std::size(prefix)] = {};
        for (size_t i : order) {
            Resource& res = m.resources[i];
            res.name = std::string(prefix[int(res.kind)]) + std::to_string(next[int(res.kind)]++);
        }
    }
    optimize(m);
    for (const std::string& e : verify(m)) r.errors.push_back("after optimize: " + e);
    if (!r.errors.empty()) return r;
    r.shader = Emitter(m, options.loopGuard).run();
    return r;
}

std::string describeKey(const MslShader& s) {
    static const char* kind[] = {"b", "c", "t", "s", "u", "bs"};  // by ResourceKind
    static_assert(std::size(kind) == size_t(ResourceKind::BufferSize) + 1);
    char head[96];
    std::snprintf(head, sizeof head, "src=%016llx bytes=%zu args=", (unsigned long long)sourceHash(s.source), s.source.size());
    std::string out = head;
    for (size_t i = 0; i < s.arguments.size(); ++i)
        out += (i ? "," : "") + std::string(kind[int(s.arguments[i].kind)]) + std::to_string(s.arguments[i].index);
    if (s.arguments.empty()) out += "none";
    char tail[64];
    std::snprintf(tail, sizeof tail, " wg=%u,%u,%u colors=0x%x", s.workgroupSize[0], s.workgroupSize[1], s.workgroupSize[2],
                  unsigned(s.colorOutputs));
    return out + tail;
}

#ifndef VPMSL_TRANSLATOR_HASH
#define VPMSL_TRANSLATOR_HASH "unknown"
#endif
const char* translatorFingerprint() { return VPMSL_TRANSLATOR_HASH; }

}  // namespace vpmsl

// C entry for hosts (dlsym): a first-run precompile stamped with another fingerprint is stale.
extern "C" __attribute__((visibility("default"))) const char* vpmsl_translator_fingerprint() {
    return VPMSL_TRANSLATOR_HASH;
}
