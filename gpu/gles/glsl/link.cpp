// Program linking: interface matching, locations, the default uniform block and reflection.
#include <algorithm>
#include <functional>
#include <set>

#include "ast.h"
#include "emit.h"

namespace vpglsl {

namespace {

uint32_t stageBit(Stage s) { return 1u << uint32_t(s); }

// Expands a uniform of type `t` named `n` into GL-visible leaves (structs flattened, arrays of
// structs indexed), at std140 `offset`.
void flatten(const std::string& n, const Type& t, uint32_t offset, bool rowMajor, std::vector<Variable>& out) {
    if (t.base == Base::Struct) {
        if (t.isArray()) {
            const uint32_t stride = std140ArrayStride(t);
            for (int i = 0; i < t.dims[0]; ++i)
                flatten(n + "[" + std::to_string(i) + "]", t.element(), offset + stride * uint32_t(i), rowMajor, out);
            return;
        }
        uint32_t size = 0;
        auto off = std140Offsets(t.st->members, &size);
        for (size_t i = 0; i < off.size(); ++i)
            flatten(n + "." + t.st->members[i].name, t.st->members[i].type, offset + off[i], t.st->members[i].rowMajor, out);
        return;
    }
    if (t.isArray() && t.dims.size() > 1) {  // arrays of arrays: one entry per outer element
        const uint32_t stride = std140ArrayStride(t);
        for (int i = 0; i < t.dims[0]; ++i)
            flatten(n + "[" + std::to_string(i) + "]", t.element(), offset + stride * uint32_t(i), rowMajor, out);
        return;
    }
    Variable v;
    v.name = t.isArray() ? n + "[0]" : n;
    v.isArray = t.isArray();
    v.arraySize = t.isArray() ? t.dims[0] : 1;
    Type el = t.isArray() ? t.element() : t;
    v.glType = glTypeEnum(el);
    v.offset = offset;
    v.arrayStride = t.isArray() ? std140ArrayStride(t, rowMajor) : 0;
    v.matrixStride = el.cols ? 16 : 0;
    v.rowMajor = rowMajor;
    out.push_back(v);
}

bool usedAnywhere(const Shader& sh, const Symbol* s) { return s->used; }

}  // namespace

LinkResult link(const std::shared_ptr<Shader>& vs, const std::shared_ptr<Shader>& fs, const LinkOptions& options) {
    LinkResult r;
    auto error = [&](const std::string& m) {
        r.log += "ERROR: " + m + "\n";
        r.ok = false;
        return r;
    };
    if (!vs) return error("no shader attached");
    const bool compute = vs->stage == Stage::Compute;
    if (!compute) {
        if (vs->stage != Stage::Vertex) return error("missing vertex shader");
        if (!fs || fs->stage != Stage::Fragment) return error("missing fragment shader");
        if (vs->version != fs->version && (vs->version == 100 || fs->version == 100))
            return error("vertex and fragment shaders use different GLSL ES versions");
    }
    std::vector<Shader*> stages = {vs.get()};
    if (!compute) stages.push_back(fs.get());
    for (Shader* s : stages)
        if (!s->main()) return error("missing main()");

    ProgramLayout L;
    // ---- uniforms: union by name, types must agree --------------------------------------------
    struct U {
        const Symbol* sym;
        uint32_t stages;
    };
    std::vector<std::pair<std::string, U>> uniforms;
    auto findU = [&](const std::string& n) -> U* {
        for (auto& [k, u] : uniforms)
            if (k == n) return &u;
        return nullptr;
    };
    for (Shader* s : stages)
        for (const GlobalDecl& g : s->globals) {
            if (g.sym->storage != Storage::Uniform) continue;
            if (U* u = findU(g.sym->name)) {
                if (u->sym->type != g.sym->type) return error("uniform '" + g.sym->name + "' differs between stages");
                if (usedAnywhere(*s, g.sym)) u->stages |= stageBit(s->stage);
                continue;
            }
            uniforms.push_back({g.sym->name, {g.sym, usedAnywhere(*s, g.sym) ? stageBit(s->stage) : 0u}});
        }
    // Unused uniforms are inactive (GL): dropped from the layout and reflection.
    uint32_t samplerBinding = 1;
    for (auto& [n, u] : uniforms) {
        if (!u.stages) continue;
        if (u.sym->type.containsSampler()) {
            if (u.sym->type.base == Base::Struct) return error("samplers in structs are not supported ('" + n + "')");
            L.samplerBinding[n] = samplerBinding;
            Variable v;
            v.name = u.sym->type.isArray() ? n + "[0]" : n;
            v.isArray = u.sym->type.isArray();
            v.arraySize = v.isArray ? u.sym->type.dims[0] : 1;
            v.glType = glTypeEnum(v.isArray ? u.sym->type.element() : u.sym->type);
            v.binding = int(samplerBinding);
            v.stages = u.stages;
            r.reflection.uniforms.push_back(v);
            ++samplerBinding;
            continue;
        }
        L.defaultBlock.push_back({n, u.sym->type});
    }
    r.reflection.samplerCount = samplerBinding - 1;
    if (!L.defaultBlock.empty()) {
        std::vector<Member> members;
        for (auto& d : L.defaultBlock) members.push_back({d.name, d.type, false});
        uint32_t size = 0;
        auto off = std140Offsets(members, &size);
        r.reflection.defaultBlockSize = size;
        for (size_t i = 0; i < members.size(); ++i) {
            size_t first = r.reflection.uniforms.size();
            flatten(members[i].name, members[i].type, off[i], false, r.reflection.uniforms);
            U* u = findU(members[i].name);
            for (size_t k = first; k < r.reflection.uniforms.size(); ++k) r.reflection.uniforms[k].stages = u->stages;
        }
    }
    // ---- uniform blocks ------------------------------------------------------------------------
    uint32_t blockIndex = 0;
    std::map<std::string, const UniformBlockDecl*> blocks;
    for (Shader* s : stages)
        for (const UniformBlockDecl* b : s->blocks) {
            auto it = blocks.find(b->name);
            if (it != blocks.end()) {
                const UniformBlockDecl* o = it->second;
                bool same = o->members.size() == b->members.size() && o->dims == b->dims;
                for (size_t i = 0; same && i < b->members.size(); ++i)
                    same = o->members[i].name == b->members[i].name && o->members[i].type == b->members[i].type;
                if (!same) return error("uniform block '" + b->name + "' differs between stages");
                for (auto& rb : r.reflection.blocks)
                    if (rb.name == b->name || rb.name.rfind(b->name + "[", 0) == 0)
                        if (b->instanceSym->used) rb.stages |= stageBit(s->stage);
                continue;
            }
            blocks[b->name] = b;
            if (!b->instanceSym->used) {
                // Inactive in this stage; may become active in the other one.
                bool usedElsewhere = false;
                for (Shader* o : stages)
                    for (const UniformBlockDecl* ob : o->blocks)
                        if (ob->name == b->name && ob->instanceSym->used) usedElsewhere = true;
                if (!usedElsewhere) continue;
            }
            const uint32_t binding = kBlockBase + blockIndex++;
            L.blockBinding[b->name] = binding;
            uint32_t size = 0;
            auto off = std140Offsets(b->members, &size);
            const int count = b->dims.empty() ? 1 : b->dims[0];
            for (int e = 0; e < count; ++e) {
                UniformBlock ub;
                ub.name = b->dims.empty() ? b->name : b->name + "[" + std::to_string(e) + "]";
                ub.size = size;
                ub.binding = binding;
                ub.element = uint32_t(e);
                ub.stages = b->instanceSym->used ? stageBit(s->stage) : 0;
                const std::string prefix = b->instance.empty() ? "" : b->name + ".";
                for (size_t i = 0; i < off.size(); ++i)
                    flatten(prefix + b->members[i].name, b->members[i].type, off[i], b->members[i].rowMajor, ub.members);
                r.reflection.blocks.push_back(ub);
            }
        }
    // ---- vertex inputs ---------------------------------------------------------------------------
    if (!compute) {
        uint32_t taken = 0;
        auto slots = [](const Type& t) {
            int n = t.isArray() ? t.dims[0] : 1;
            Type el = t.isArray() ? t.element() : t;
            return n * (el.cols ? el.cols : 1);
        };
        std::vector<const Symbol*> pending;
        for (const GlobalDecl& g : vs->globals) {
            const Symbol* s = g.sym;
            if (s->storage != Storage::In || !s->used) continue;
            int loc = s->location;
            if (auto b = options.attribBindings.find(s->name); b != options.attribBindings.end() && loc < 0) loc = b->second;
            if (loc < 0) {
                pending.push_back(s);
                continue;
            }
            const int n = slots(s->type);
            for (int k = 0; k < n; ++k) {
                if (loc + k >= 16) return error("attribute '" + s->name + "' location out of range");
                taken |= 1u << (loc + k);
            }
            L.attribLocation[s->name] = loc;
        }
        for (const Symbol* s : pending) {
            const int n = slots(s->type);
            int loc = -1;
            for (int c = 0; c + n <= 16 && loc < 0; ++c) {
                bool free = true;
                for (int k = 0; k < n; ++k) free &= !(taken & (1u << (c + k)));
                if (free) loc = c;
            }
            if (loc < 0) return error("too many vertex attributes");
            for (int k = 0; k < n; ++k) taken |= 1u << (loc + k);
            L.attribLocation[s->name] = loc;
        }
        for (const GlobalDecl& g : vs->globals) {
            const Symbol* s = g.sym;
            auto it = L.attribLocation.find(s->name);
            if (s->storage != Storage::In || it == L.attribLocation.end()) continue;
            Variable v;
            v.name = s->name;
            v.glType = glTypeEnum(s->type.isArray() ? s->type.element() : s->type);
            v.isArray = s->type.isArray();
            v.arraySize = v.isArray ? s->type.dims[0] : 1;
            v.location = it->second;
            v.stages = stageBit(Stage::Vertex);
            r.reflection.attributes.push_back(v);
        }
        // ---- varyings: fragment inputs must be written by the vertex shader ------------------------
        int nextLoc = 0;
        for (const GlobalDecl& g : fs->globals) {
            const Symbol* in = g.sym;
            if (in->storage != Storage::In) continue;
            const Symbol* out = nullptr;
            for (const GlobalDecl& vg : vs->globals)
                if (vg.sym->storage == Storage::Out && vg.sym->name == in->name) out = vg.sym;
            if (!out) {
                if (in->used) return error("fragment input '" + in->name + "' is not written by the vertex shader");
                continue;
            }
            if (out->type != in->type) return error("varying '" + in->name + "' has different types in the two stages");
            if (fs->version >= 300 && out->interp != in->interp && (out->interp == Interp::Flat || in->interp == Interp::Flat))
                return error("varying '" + in->name + "' has different interpolation qualifiers");
            if (!in->used) continue;
            int loc = in->location >= 0 ? in->location : out->location;
            if (loc < 0) {
                loc = nextLoc;
            }
            Type el = in->type.isArray() ? in->type.element() : in->type;
            int n = (in->type.isArray() ? in->type.dims[0] : 1) * (el.cols ? el.cols : 1);
            if (el.base == Base::Struct) return error("struct varyings are not supported ('" + in->name + "')");
            L.varyingLocation[in->name] = loc;
            nextLoc = std::max(nextLoc, loc + n);
        }
        if (nextLoc > 31) return error("too many varyings");
        // ---- fragment outputs ------------------------------------------------------------------
        int autoLoc = 0;
        int outputs = 0;
        for (const GlobalDecl& g : fs->globals) {
            const Symbol* s = g.sym;
            if (s->storage != Storage::Out) continue;
            ++outputs;
            int loc = s->location >= 0 ? s->location : autoLoc;
            int n = s->type.isArray() ? s->type.dims[0] : 1;
            if (loc + n > 8) return error("fragment output '" + s->name + "' location out of range");
            L.outputLocation[s->name] = loc;
            autoLoc = loc + n;
            Variable v;
            v.name = s->name;
            v.glType = glTypeEnum(s->type.isArray() ? s->type.element() : s->type);
            v.location = loc;
            v.isArray = s->type.isArray();
            v.arraySize = n;
            r.reflection.outputs.push_back(v);
        }
        if (outputs > 1)
            for (const GlobalDecl& g : fs->globals)
                if (g.sym->storage == Storage::Out && g.sym->location < 0)
                    return error("with several fragment outputs each needs a layout(location)");
        // gl_FragData: emit as many elements as the shader indexes (constant indices), at least 1.
        int maxIdx = 0;
        forEachExpr(*fs, [&](const Expr& e) {
            if (e.k == EK::Index && e.a[0]->k == EK::Var && e.a[0]->sym->builtin == BuiltinVar::FragData) {
                if (e.a[1]->hasConst) maxIdx = std::max(maxIdx, int(e.a[1]->constI));
                else maxIdx = 7;
            }
        });
        L.fragDataCount = maxIdx + 1;
        r.reflection.numViews = uint32_t(std::max(1, vs->numViews));
    } else {
        r.reflection.localSize = vs->localSize;
    }
    // ---- emit -----------------------------------------------------------------------------------
    std::string err;
    if (compute) {
        r.computeSpirv = emitSpirv(*vs, L, err);
        if (r.computeSpirv.empty()) return error("compute: " + err);
    } else {
        r.vertexSpirv = emitSpirv(*vs, L, err);
        if (r.vertexSpirv.empty()) return error("vertex: " + err);
        r.fragmentSpirv = emitSpirv(*fs, L, err);
        if (r.fragmentSpirv.empty()) return error("fragment: " + err);
    }
    r.ok = true;
    return r;
}

}  // namespace vpglsl
