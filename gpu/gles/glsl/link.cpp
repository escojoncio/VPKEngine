// Program linking: interface matching, locations, the default uniform block and reflection.
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <set>

#include "ast.h"
#include "emit.h"

namespace vpglsl {

namespace {

uint32_t stageBit(Stage s) { return 1u << uint32_t(s); }

// Type equality across stages: struct types are separate per-shader declarations, so they compare by
// member names and types (GLSL ES 3.00 matches interface structs structurally), not by identity.
bool sameType(const Type& a, const Type& b) {
    if (a.base != Base::Struct || b.base != Base::Struct) return a == b;
    if (a.dims != b.dims) return false;
    if (a.st == b.st) return true;
    if (!a.st || !b.st || a.st->members.size() != b.st->members.size()) return false;
    for (size_t i = 0; i < a.st->members.size(); ++i)
        if (a.st->members[i].name != b.st->members[i].name || !sameType(a.st->members[i].type, b.st->members[i].type))
            return false;
    return true;
}

// Expands a uniform of type `t` named `n` into GL-visible leaves (structs flattened, arrays of
// structs indexed), at std140 `offset`.
void flatten(const std::string& n, const Type& t, uint32_t offset, bool rowMajor, std::vector<Variable>& out,
             bool s430 = false, bool storageTop = false) {
    if (storageTop && t.isArray() && (t.base == Base::Struct || t.dims.size() > 1)) {
        // The top-level array of a storage block member is reported once, through element 0, with its size and stride.
        const size_t first = out.size();
        flatten(n + "[0]", t.element(), offset, rowMajor, out, s430);
        const uint32_t stride = std140ArrayStride(t, rowMajor, s430);
        for (size_t i = first; i < out.size(); ++i) out[i].topLevelArraySize = t.dims[0], out[i].topLevelArrayStride = stride;
        return;
    }
    if (t.base == Base::Struct) {
        if (t.isArray()) {
            // An unsized array (last member of a storage block) is reported through its first element.
            const uint32_t stride = std140ArrayStride(t, rowMajor, s430);
            for (int i = 0; i < std::max(1, t.dims[0]); ++i)
                flatten(n + "[" + std::to_string(i) + "]", t.element(), offset + stride * uint32_t(i), rowMajor, out, s430);
            return;
        }
        uint32_t size = 0;
        auto off = std140Offsets(t.st->members, &size, s430, rowMajor);
        for (size_t i = 0; i < off.size(); ++i)
            flatten(n + "." + t.st->members[i].name, t.st->members[i].type, offset + off[i], t.st->members[i].rowMajor || rowMajor, out, s430);
        return;
    }
    if (t.isArray() && t.dims.size() > 1) {  // arrays of arrays: one entry per outer element
        const uint32_t stride = std140ArrayStride(t, rowMajor, s430);
        for (int i = 0; i < std::max(1, t.dims[0]); ++i)
            flatten(n + "[" + std::to_string(i) + "]", t.element(), offset + stride * uint32_t(i), rowMajor, out, s430);
        return;
    }
    Variable v;
    v.name = t.isArray() ? n + "[0]" : n;
    v.isArray = t.isArray();
    v.arraySize = t.isArray() ? t.dims[0] : 1;  // 0 = unsized
    Type el = t.isArray() ? t.element() : t;
    v.glType = glTypeEnum(el);
    v.offset = offset;
    v.arrayStride = t.isArray() ? std140ArrayStride(t, rowMajor, s430) : 0;
    v.matrixStride = el.cols ? std140MatrixStride(el, rowMajor, s430) : 0;
    v.rowMajor = el.cols != 0 && rowMajor;  // GL reports IS_ROW_MAJOR only for matrices
    out.push_back(v);
}

// Uniform locations a declaration takes: one per array element of every basic-type leaf.
int locationCount(const Type& t) {
    int n = 1;
    for (int d : t.dims) n *= std::max(1, d);
    if (t.base != Base::Struct) return n;
    int leaves = 0;
    for (const Member& m : t.st->members) leaves += locationCount(m.type);
    return n * leaves;
}

// Varying locations a type takes (Vulkan interface matching rules, as ES 3.00 4.3.4 counts them): a matrix one per column, an array one run per
// element, a struct the sum of its members.
int varyingSlots(const Type& t) {
    int n = 1;
    for (int d : t.dims) n *= std::max(1, d);
    if (t.base == Base::Struct) {
        int sum = 0;
        for (const Member& m : t.st->members) sum += varyingSlots(m.type);
        return n * sum;
    }
    return n * (t.cols ? t.cols : 1);
}

constexpr int kMaxUniformLocations = 1048576;  // GL_MAX_UNIFORM_LOCATIONS (gl_api.cpp)

bool usedAnywhere(const Shader&, const Symbol* s) { return s->used; }

// Declared precisions of two uniforms of the same (structurally equal) type must agree, member by member.
bool samePrecision(const Type& t, uint8_t a, uint8_t b, const Type& u) {
    if (t.base != Base::Struct) return a == b;
    for (size_t i = 0; i < t.st->members.size() && i < u.st->members.size(); ++i)
        if (!samePrecision(t.st->members[i].type, t.st->members[i].prec, u.st->members[i].prec, u.st->members[i].type) )
            return false;
    return true;
}

}  // namespace

int shaderVersion(const Shader& s) { return s.version; }

LinkResult link(const std::shared_ptr<Shader>& vs, const std::shared_ptr<Shader>& fs, const LinkOptions& options) {
    LinkResult r;
    auto error = [&](const std::string& m) {
        r.log += "ERROR: " + m + "\n";
        r.ok = false;
        return r;
    };
    // A rule of the language that a lenient driver lets through (see strictMode()): true when it is an error.
    auto strict = [&](const char* rule, const std::string& m) {
        if (strictMode()) {
            r.log += "ERROR: " + m + "\n";
            r.ok = false;
            return true;
        }
        if (std::find(r.lenient.begin(), r.lenient.end(), rule) == r.lenient.end()) r.lenient.push_back(rule);
        return false;
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
                if (!sameType(u->sym->type, g.sym->type) ||
                    (g.sym->type.base == Base::AtomicUint && (u->sym->offset != g.sym->offset || u->sym->binding != g.sym->binding)))
                    return error("uniform '" + g.sym->name + "' differs between stages");
                if (!samePrecision(u->sym->type, u->sym->prec, g.sym->prec, g.sym->type) &&
                    strict("uniform_precision", "uniform '" + g.sym->name + "' has different precisions in the two stages"))
                    return r;
                if (g.sym->type.base != Base::AtomicUint && u->sym->binding >= 0 && g.sym->binding >= 0 && u->sym->binding != g.sym->binding &&
                    strict("binding_contradictory", "uniform '" + g.sym->name + "' has different bindings in the two stages"))
                    return r;
                if (usedAnywhere(*s, g.sym)) u->stages |= stageBit(s->stage);
                if (u->sym->location < 0 && g.sym->location >= 0) u->sym = g.sym;  // layout(location) in one stage places the uniform for both
                continue;
            }
            uniforms.push_back({g.sym->name, {g.sym, usedAnywhere(*s, g.sym) ? stageBit(s->stage) : 0u}});
        }
    // Unused uniforms are inactive (GL): dropped from the layout and reflection.
    uint32_t samplerBinding = 1, imageIndex = 0;
    struct LocGroup {
        size_t first, last;  // reflection.uniforms[first, last)
        int declared;        // layout(location = N), -1 = none
    };
    std::vector<LocGroup> locGroups;
    std::map<int, uint32_t> atomicSize, atomicStages;  // GL binding point -> bytes / stage bits
    for (auto& [n, u] : uniforms) {
        if (!u.stages) continue;
        if (u.sym->type.base == Base::Image) {
            if (imageIndex >= 16) return error("too many image uniforms");
            const uint32_t binding = kImageBase + imageIndex++;
            L.imageBinding[n] = binding;
            Variable v;
            v.name = u.sym->type.isArray() ? n + "[0]" : n;
            v.isArray = u.sym->type.isArray();
            v.arraySize = v.isArray ? u.sym->type.dims[0] : 1;
            v.glType = glTypeEnum(v.isArray ? u.sym->type.element() : u.sym->type);
            v.binding = int(binding);
            v.unit = u.sym->binding;
            v.stages = u.stages;
            locGroups.push_back({r.reflection.uniforms.size(), r.reflection.uniforms.size() + 1, u.sym->location});
            r.reflection.uniforms.push_back(v);
            continue;
        }
        if (u.sym->type.base == Base::AtomicUint) {
            const int glBinding = u.sym->binding;
            if (glBinding < 0 || glBinding >= 16) return error("atomic counter binding out of range ('" + n + "')");
            L.atomicBinding[glBinding] = kAtomicBase + uint32_t(glBinding);
            const int count = u.sym->type.isArray() ? u.sym->type.dims[0] : 1;
            atomicSize[glBinding] = std::max(atomicSize[glBinding], uint32_t(u.sym->offset + 4 * count));
            atomicStages[glBinding] |= u.stages;
            Variable v;
            v.name = u.sym->type.isArray() ? n + "[0]" : n;
            v.isArray = u.sym->type.isArray();
            v.arraySize = count;
            v.glType = glTypeEnum(Type::make(Base::AtomicUint));
            v.offset = uint32_t(u.sym->offset);
            v.arrayStride = v.isArray ? 4 : 0;
            v.atomicBinding = glBinding;
            v.stages = u.stages;
            r.reflection.uniforms.push_back(v);
            continue;
        }
        if (u.sym->type.containsSampler()) {
            L.samplerBinding[n] = samplerBinding;
            Variable v;
            v.name = u.sym->type.isArray() ? n + "[0]" : n;
            v.isArray = u.sym->type.isArray();
            v.arraySize = v.isArray ? u.sym->type.dims[0] : 1;
            v.glType = glTypeEnum(v.isArray ? u.sym->type.element() : u.sym->type);
            v.binding = int(samplerBinding);
            v.unit = u.sym->binding;
            v.stages = u.stages;
            locGroups.push_back({r.reflection.uniforms.size(), r.reflection.uniforms.size() + 1, u.sym->location});
            r.reflection.uniforms.push_back(v);
            ++samplerBinding;
            continue;
        }
        L.defaultBlock.push_back({n, u.sym->type});
    }
    r.reflection.samplerCount = samplerBinding - 1;
    if (samplerBinding > kBlockBase) return error("too many sampler uniforms");
    for (auto& [b, size] : atomicSize) r.reflection.atomicBuffers.push_back({b, kAtomicBase + uint32_t(b), size, atomicStages[b]});
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
            locGroups.push_back({first, r.reflection.uniforms.size(), u->sym->location});
            for (size_t k = first; k < r.reflection.uniforms.size(); ++k) r.reflection.uniforms[k].stages = u->stages;
        }
    }
    // ---- uniform locations: layout(location = N) (even for inactive uniforms), the rest in the lowest free ones ----
    {
        std::vector<char> taken;
        auto take = [&](int at, int n) {
            if (size_t(at + n) > taken.size()) taken.resize(size_t(at + n), 0);
            for (int k = 0; k < n; ++k) taken[size_t(at + k)] = 1;
        };
        auto free = [&](int at, int n) {
            for (int k = 0; k < n && size_t(at + k) < taken.size(); ++k)
                if (taken[size_t(at + k)]) return false;
            return true;
        };
        std::map<std::string, int> explicitAt;  // uniform name -> its explicit location, when it was honoured
        for (auto& [n, u] : uniforms) {
            if (u.sym->location < 0 || u.sym->type.base == Base::AtomicUint) continue;
            const int count = locationCount(u.sym->type);
            if (u.sym->location + count > kMaxUniformLocations) {
                if (strict("uniform_location", "uniform '" + n + "' location is beyond MAX_UNIFORM_LOCATIONS")) return r;
                continue;
            }
            if (!free(u.sym->location, count)) {
                if (strict("uniform_location", "uniform '" + n + "' location overlaps another uniform")) return r;
                continue;  // lenient: it gets an implicit one
            }
            take(u.sym->location, count);
            explicitAt[n] = u.sym->location;
        }
        for (const LocGroup& g : locGroups) {
            if (g.first >= g.last) continue;
            int total = 0;
            for (size_t k = g.first; k < g.last; ++k) total += std::max(1, r.reflection.uniforms[k].arraySize);
            int at = -1;
            if (g.declared >= 0) {
                // The symbol name is the start of the first leaf ("name.member", "name[0]").
                std::string base = r.reflection.uniforms[g.first].name;
                base = base.substr(0, base.find_first_of(".["));
                auto it = explicitAt.find(base);
                if (it != explicitAt.end()) at = it->second;
            }
            if (at < 0) {
                at = 0;
                while (!free(at, total)) ++at;
                take(at, total);
            }
            for (size_t k = g.first; k < g.last; ++k) {
                Variable& v = r.reflection.uniforms[k];
                v.uniformLocation = at;
                at += std::max(1, v.arraySize);
            }
        }
        if (strictMode())
            for (const Variable& v : r.reflection.uniforms)
                if (v.uniformLocation >= kMaxUniformLocations) return error("too many uniform locations");
    }
    // ---- uniform and shader storage blocks -------------------------------------------------------
    uint32_t blockIndex = 0, storageIndex = 0;
    std::map<std::string, const UniformBlockDecl*> blocks;  // "U:name" / "S:name"
    std::map<std::string, std::string> unnamedMembers;      // member of an unnamed block -> its block
    for (Shader* s : stages)
        for (const UniformBlockDecl* b : s->blocks) {
            auto& list = b->buffer ? r.reflection.storageBlocks : r.reflection.blocks;
            const std::string key = (b->buffer ? "S:" : "U:") + b->name;
            auto it = blocks.find(key);
            if (it != blocks.end()) {
                const UniformBlockDecl* o = it->second;
                bool same = o->members.size() == b->members.size() && o->dims == b->dims && o->std430 == b->std430;
                for (size_t i = 0; same && i < b->members.size(); ++i)
                    same = o->members[i].name == b->members[i].name && sameType(o->members[i].type, b->members[i].type);
                if (!same) return error((b->buffer ? "shader storage block '" : "uniform block '") + b->name + "' differs between stages");
                if (o->instance.empty() != b->instance.empty() && strict("block_instance", "block '" + b->name + "' has an instance name in only one stage"))
                    return r;
                bool sameLayout = (o->packing ? o->packing : 1) == (b->packing ? b->packing : 1) && o->rowMajor == b->rowMajor;
                for (size_t i = 0; sameLayout && i < b->members.size(); ++i)
                    sameLayout = o->members[i].rowMajor == b->members[i].rowMajor;
                if (!sameLayout && strict("block_layout", "block '" + b->name + "' has different layout qualifiers in the two stages"))
                    return r;
                if (o->binding >= 0 && b->binding >= 0 && o->binding != b->binding &&
                    strict("binding_contradictory", "block '" + b->name + "' has different bindings in the two stages"))
                    return r;
                for (auto& rb : list)
                    if (rb.name == b->name || rb.name.rfind(b->name + "[", 0) == 0)
                        if (b->instanceSym->used) rb.stages |= stageBit(s->stage);
                continue;
            }
            blocks[key] = b;
            if (b->instance.empty())  // members of an unnamed block are global names of the whole program
                for (const Member& m : b->members) {
                    auto [at, fresh] = unnamedMembers.insert({m.name, b->name});
                    if (!fresh && at->second != b->name && strict("block_member_clash", "'" + m.name + "' is a member of blocks '" + at->second + "' and '" + b->name + "'"))
                        return r;
                }
            if (!b->instanceSym->used) {
                // Inactive in this stage; may become active in the other one.
                bool usedElsewhere = false;
                for (Shader* o : stages)
                    for (const UniformBlockDecl* ob : o->blocks)
                        if (ob->buffer == b->buffer && ob->name == b->name && ob->instanceSym->used) usedElsewhere = true;
                if (!usedElsewhere) continue;
            }
            uint32_t binding;
            if (b->buffer) {
                if (storageIndex >= 36) return error("too many shader storage blocks");
                binding = kStorageBase + storageIndex++;
                L.storageBinding[b->name] = binding;
            } else {
                if (blockIndex >= 84) return error("too many uniform blocks");
                binding = kBlockBase + blockIndex++;
                L.blockBinding[b->name] = binding;
            }
            uint32_t size = 0;
            auto off = std140Offsets(b->members, &size, b->std430);
            const int count = b->dims.empty() ? 1 : b->dims[0];
            for (int e = 0; e < count; ++e) {
                UniformBlock ub;
                ub.name = b->dims.empty() ? b->name : b->name + "[" + std::to_string(e) + "]";
                ub.size = size;
                ub.binding = binding;
                ub.initialBinding = b->binding >= 0 ? b->binding + e : -1;
                ub.element = uint32_t(e);
                ub.stages = b->instanceSym->used ? stageBit(s->stage) : 0;
                const std::string prefix = b->instance.empty() ? "" : b->name + ".";
                for (size_t i = 0; i < off.size(); ++i)
                    flatten(prefix + b->members[i].name, b->members[i].type, off[i], b->members[i].rowMajor, ub.members, b->std430, b->buffer);
                list.push_back(ub);
            }
        }
    // ---- vertex inputs ---------------------------------------------------------------------------
    if (!compute) {
        uint32_t taken = 0;
        auto slots = [](const Type& t) {
            int n = t.isArray() ? t.dims[0] : 1;
            return n * (t.cols ? t.cols : 1);  // an array element has the same column count as the array
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
                if (loc + k >= 32) return error("attribute '" + s->name + "' location out of range");
                if ((taken & (1u << (loc + k))) &&
                    strict("attribute_location_alias", "attribute '" + s->name + "' shares location " + std::to_string(loc + k) + " with another active attribute"))
                    return r;
                taken |= 1u << (loc + k);
            }
            L.attribLocation[s->name] = loc;
        }
        for (const Symbol* s : pending) {
            const int n = slots(s->type);
            int loc = -1;
            for (int c = 0; c + n <= 32 && loc < 0; ++c) {
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
        // ---- varyings: fragment inputs are paired with vertex outputs by name, else by explicit location --
        // An input no output feeds is not an error (real drivers read an undefined value): it reads as zero.
        std::string outsTxt, insTxt;
        auto tag = [](const Symbol* s) { return s->name + "@" + (s->location >= 0 ? std::to_string(s->location) : "-"); };
        for (const GlobalDecl& vg : vs->globals)
            if (vg.sym->storage == Storage::Out) outsTxt += (outsTxt.empty() ? "" : " ") + tag(vg.sym);
        auto note = [&](const Symbol* s, const char* why) {
            insTxt += (insTxt.empty() ? "" : " ") + tag(s) + why;
            r.varyingDiag = "vs_out=[" + outsTxt + "] fs_in=[" + insTxt + "]";
        };
        auto fragmentReads = [&](const std::string& n) {
            for (const GlobalDecl& fg : fs->globals)
                if (fg.sym->storage == Storage::In && fg.sym->name == n) return true;
            return false;
        };
        // Pass 1: pair every input. Pass 2: reserve the explicit locations (two varyings on one is an error), so the
        // automatic ones, which follow in declaration order, skip them.
        struct Pair {
            const Symbol *in, *out;
            int loc, n;
        };
        std::vector<Pair> pairs;
        for (const GlobalDecl& g : fs->globals) {
            const Symbol* in = g.sym;
            if (in->storage != Storage::In) continue;
            const Symbol* out = nullptr;
            for (const GlobalDecl& vg : vs->globals)
                if (vg.sym->storage == Storage::Out && vg.sym->name == in->name) out = vg.sym;
            if (!out && in->used && in->location >= 0)
                for (const GlobalDecl& vg : vs->globals) {
                    const Symbol* o = vg.sym;
                    if (o->storage == Storage::Out && o->location == in->location && sameType(o->type, in->type) && !fragmentReads(o->name))
                        out = o;
                }
            if (!out) {
                if (in->used) {
                    L.unwrittenInput.insert(in->name);
                    note(in, "!unwritten");
                    if (!options.separable &&
                        strict("varying_unmatched", "fragment input '" + in->name + "' is not an output of the vertex shader"))
                        return r;
                }
                continue;
            }
            if (!sameType(out->type, in->type)) {
                note(in, "!type");
                return error("varying '" + in->name + "' has different types in the two stages");
            }
            if (fs->version >= 300 && out->interp != in->interp && (out->interp == Interp::Flat || in->interp == Interp::Flat)) {
                note(in, "!interp");
                return error("varying '" + in->name + "' has different interpolation qualifiers");
            }
            if (!in->used) continue;
            const int n = varyingSlots(in->type);
            pairs.push_back({in, out, in->location >= 0 ? in->location : out->location, n});
            if (in->interp == Interp::Flat) r.reflection.flatVaryings = true;
        }
        // Between the program objects of a pipeline (ES 3.1 7.4.1) every declared output has an input and every declared input an
        // output: matched by explicit location when either has one, else by name, and at the same precision.
        if (options.pipeline) {
            auto matches = [](const Symbol* o, const Symbol* i) {
                return o->location >= 0 || i->location >= 0 ? o->location == i->location : o->name == i->name;
            };
            for (const GlobalDecl& vg : vs->globals) {
                if (vg.sym->storage != Storage::Out) continue;
                const Symbol* peer = nullptr;
                for (const GlobalDecl& fg : fs->globals)
                    if (fg.sym->storage == Storage::In && matches(vg.sym, fg.sym)) peer = fg.sym;
                if (!peer) {
                    if (strict("pipeline_output_unmatched", "vertex output '" + vg.sym->name + "' has no matching fragment input")) return r;
                } else if (peer->prec != vg.sym->prec &&
                           strict("pipeline_precision", "varying '" + vg.sym->name + "' has different precisions in the two stages")) {
                    return r;
                }
            }
            for (const GlobalDecl& fg : fs->globals) {
                if (fg.sym->storage != Storage::In) continue;
                bool found = false;
                for (const GlobalDecl& vg : vs->globals) found = found || (vg.sym->storage == Storage::Out && matches(vg.sym, fg.sym));
                if (!found && strict("pipeline_input_unmatched", "fragment input '" + fg.sym->name + "' has no matching vertex output")) return r;
            }
        }
        std::map<int, const Symbol*> slotOwner;  // varying slot -> the fragment input that owns it
        for (const Pair& p : pairs) {
            if (p.loc < 0) continue;
            for (int k = 0; k < p.n; ++k) {
                auto [it, fresh] = slotOwner.insert({p.loc + k, p.in});
                if (!fresh && it->second != p.in) {
                    note(p.in, "!location");
                    return error("varyings '" + it->second->name + "' and '" + p.in->name + "' use the same location " +
                                 std::to_string(p.loc + k));
                }
            }
        }
        int nextLoc = 0;
        for (Pair& p : pairs) {
            if (p.loc < 0) {
                p.loc = nextLoc;
                for (bool clash = true; clash;) {
                    clash = false;
                    for (int k = 0; k < p.n && !clash; ++k)
                        if (slotOwner.count(p.loc + k)) clash = true, p.loc += k + 1;
                }
                for (int k = 0; k < p.n; ++k) slotOwner[p.loc + k] = p.in;
            }
            L.varyingLocation[p.in->name] = p.loc;
            L.vertexOutLocation[p.out->name] = p.loc;
            nextLoc = std::max(nextLoc, p.loc + p.n);
        }
        if (nextLoc > 31) return error("too many varyings");
        // ---- fragment outputs ------------------------------------------------------------------
        int autoLoc = 0;
        int outputs = 0;
        uint32_t outTaken = 0;
        for (const GlobalDecl& g : fs->globals) {
            const Symbol* s = g.sym;
            if (s->storage != Storage::Out) continue;
            ++outputs;
            int loc = s->location >= 0 ? s->location : autoLoc;
            int n = s->type.isArray() ? s->type.dims[0] : 1;
            if (loc + n > 8) return error("fragment output '" + s->name + "' location out of range");
            {
                const uint32_t bits = ((n >= 32 ? 0u : 1u << n) - 1u) << loc;
                if ((outTaken & bits) &&
                    strict("output_location_alias", "fragment output '" + s->name + "' shares a location with another output"))
                    return r;
                outTaken |= bits;
            }
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
        // ---- framebuffer fetch: `inout` outputs, gl_LastFragData[i], gl_LastFragColorARM -----------
        uint32_t fetch = 0;
        for (const GlobalDecl& g : fs->globals) {
            if (g.sym->storage != Storage::Out || !g.sym->fbFetch) continue;
            const int n = g.sym->type.isArray() ? g.sym->type.dims[0] : 1;
            for (int k = 0; k < n; ++k) fetch |= 1u << (L.outputLocation.at(g.sym->name) + k);
        }
        int lastData = 0;
        for (const Symbol* b : fs->builtins) {
            if (b->builtin == BuiltinVar::LastFragColor) fetch |= 1u;
            if (b->builtin == BuiltinVar::LastFragData) lastData = 1;
        }
        if (lastData) {
            forEachExpr(*fs, [&](const Expr& e) {
                if (e.k == EK::Index && e.a[0]->k == EK::Var && e.a[0]->sym->builtin == BuiltinVar::LastFragData)
                    lastData = std::max(lastData, e.a[1]->hasConst ? int(e.a[1]->constI) + 1 : 8);
            });
            L.lastFragDataCount = lastData;
            for (int k = 0; k < lastData; ++k) fetch |= 1u << k;
        }
        for (int k = 0; k < 8; ++k)
            if (fetch & (1u << k)) L.fbFetchBinding[k] = kAttachmentBase + uint32_t(k);
        r.reflection.framebufferFetchMask = fetch;
        r.reflection.numViews = uint32_t(std::max(1, vs->numViews));
    } else {
        r.reflection.localSize = vs->localSize;
        // Workgroup memory: the shared variables at their std430 sizes (Metal threadgroup memory: 32 KiB).
        uint32_t bytes = 0;
        for (const GlobalDecl& g : vs->globals)
            if (g.sym->storage == Storage::Shared) bytes += (std140Size(g.sym->type, false, true) + 15) / 16 * 16;
        if (bytes > 32768) return error("shared variables use " + std::to_string(bytes) + " bytes (limit 32768)");
        r.reflection.sharedBytes = bytes;
    }
    // ---- transform feedback varyings -------------------------------------------------------------
    // The vertex outputs glTransformFeedbackVaryings names (and gl_Position / gl_PointSize as the shader wrote them): what one
    // captured vertex holds. A name that is no vertex output, repeated, or over the limits of the buffer mode fails the link.
    ProgramLayout LX;
    if (!compute && !options.xfbVaryings.empty()) {
        LX = L;
        std::map<std::string, std::vector<std::pair<int, int>>> taken;  // variable -> captured element ranges
        uint32_t offset = 0;
        for (const std::string& given : options.xfbVaryings) {
            std::string base = given;
            int index = -1;
            if (size_t b = given.find('['); b != std::string::npos) {
                base = given.substr(0, b);
                const std::string rest = given.substr(b + 1);
                if (rest.size() < 2 || rest.back() != ']' || rest.find_first_not_of("0123456789") != rest.size() - 1)
                    return error("transform feedback varying '" + given + "' is not a variable or an array element");
                index = std::atoi(rest.c_str());
            }
            Type el;
            int arraySize = 1;
            bool isArray = false;
            if (base == "gl_Position" && index < 0) el = Type::make(Base::Float, 4);
            else if (base == "gl_PointSize" && index < 0) el = Type::make(Base::Float);
            else {
                const Symbol* sym = nullptr;
                for (const GlobalDecl& g : vs->globals)
                    if (g.sym->storage == Storage::Out && g.sym->name == base) sym = g.sym;
                if (!sym) return error("transform feedback varying '" + given + "' is not an output of the vertex shader");
                if (sym->type.dims.size() > 1 || sym->type.base == Base::Struct)
                    return error("transform feedback varying '" + given + "' has a type that cannot be captured");
                isArray = sym->type.isArray();
                el = isArray ? sym->type.element() : sym->type;
                arraySize = isArray ? sym->type.dims[0] : 1;
                if (index >= 0 && (!isArray || index >= arraySize))
                    return error("transform feedback varying '" + given + "' is out of range");
            }
            const int first = index >= 0 ? index : 0, count = index >= 0 ? 1 : arraySize;
            for (auto [f, c] : taken[base])
                if (first < f + c && f < first + count) return error("transform feedback varying '" + given + "' is named twice");
            taken[base].push_back({first, count});
            Reflection::XfbVarying v;
            v.name = given;
            v.glType = glTypeEnum(el);
            v.arraySize = count;
            v.words = uint32_t(el.cols ? el.cols : 1) * el.vec;
            v.offset = offset;
            LX.xfb.push_back({base, first, count, isArray, offset});
            offset += v.words * uint32_t(count);
            r.reflection.xfb.push_back(v);
        }
        r.reflection.xfbRecordWords = LX.xfbRecordWords = offset;
        if (options.xfbSeparate) {
            if (r.reflection.xfb.size() > 4) return error("too many transform feedback varyings for separate attributes");
            for (const auto& v : r.reflection.xfb)
                if (v.words * uint32_t(v.arraySize) > 4)
                    return error("transform feedback varying '" + v.name + "' has more than 4 components for separate attributes");
        } else if (offset > 64) {
            return error("transform feedback varyings capture more than 64 components in interleaved mode");
        }
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
        if (!LX.xfb.empty()) {
            r.xfbVertexSpirv = emitSpirv(*vs, LX, err);
            if (r.xfbVertexSpirv.empty()) return error("transform feedback: " + err);
        }
    }
    r.ok = true;
    return r;
}

}  // namespace vpglsl
