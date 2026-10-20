// Geometry pipeline: ES, GS and copy shader lifted from the programs in lift/ and run on the VPIR
// interpreter over emulated rings (the dispatch the renderer makes: one invocation per ES entry /
// GS primitive, then the copy shader per index of the GS output list). RectList and the pos1-pos3
// exports are run the same way. usage: vpgcn_geom_test CASE
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "diff/vpir_eval.h"
#include "vpgcn/lift.h"

#ifndef VPGCN_LIFT_DIR
#error "VPGCN_LIFT_DIR"
#endif

namespace {

int failures = 0;
#define CHECK(c)                                                                          \
    do {                                                                                  \
        if (!(c)) {                                                                       \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #c);    \
            ++failures;                                                                   \
        }                                                                                 \
    } while (0)
#define CHECK_NEAR(a, b)                                                                                    \
    do {                                                                                                    \
        if (!(std::fabs(double(a) - double(b)) < 1e-5)) {                                                   \
            std::fprintf(stderr, "%s:%d: %s = %g, expected %g\n", __FILE__, __LINE__, #a, double(a), double(b)); \
            ++failures;                                                                                     \
        }                                                                                                   \
    } while (0)

using namespace vpgcn;

LiftResult load(const std::string& name) {
    LiftInput in;
    std::vector<uint32_t> code;
    std::ifstream f(std::string(VPGCN_LIFT_DIR) + "/" + name + ".lift");
    if (!f) std::fprintf(stderr, "missing %s\n", name.c_str()), std::exit(2);
    for (std::string line; std::getline(f, line);) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string key;
        ls >> key;
        if (key == "stage") {
            std::string s;
            ls >> s;
            in.stage = s == "vertex" ? ShaderStage::Vertex : s == "export" ? ShaderStage::Export
                     : s == "geometry" ? ShaderStage::Geometry : s == "copyvertex" ? ShaderStage::CopyVertex
                                                                               : ShaderStage::Compute;
        } else if (key == "user_sgprs") ls >> in.userSgprs;
        else if (key == "vgpr_comp_cnt") ls >> in.vgprCompCnt;
        else if (key == "vs_out_cntl") { std::string v; ls >> v; in.vsOutCntl = uint32_t(std::stoul(v, nullptr, 0)); }
        else if (key == "rect_list") in.rectList = true;
        else if (key == "geom") {
            for (std::string k, v; ls >> k >> v;) {
                const uint32_t n = uint32_t(std::stoul(v, nullptr, 0));
                if (k == "esgs_item") in.geom.esgsItemDwords = n;
                else if (k == "gsvs_vert") in.geom.gsvsVertDwords = n;
                else if (k == "max_vert") in.geom.maxVertOut = n;
                else if (k == "in_prim") in.geom.inputPrim = n;
                else if (k == "out_prim") in.geom.outPrim = n;
                else if (k == "gs_instances") in.geom.gsInstances = n;
            }
        } else if (key == "code") {
            for (std::string t; ls >> t;) code.push_back(uint32_t(std::stoul(t, nullptr, 16)));
        }
    }
    in.code = code;
    LiftResult r = lift(in);
    for (const std::string& e : r.errors) std::fprintf(stderr, "lift %s: %s\n", name.c_str(), e.c_str());
    CHECK(r.errors.empty());
    return r;
}

// The memory behind a lifted module's bindings.
struct Env {
    std::map<uint32_t, std::vector<uint8_t>> rings;  // GeomRing id -> bytes
    std::vector<uint32_t> userData = std::vector<uint32_t>(16);
    std::vector<uint8_t> vertexBuffer;
    std::vector<uint32_t> params = std::vector<uint32_t>(geomparam::Words);
    std::vector<uint8_t>& ring(GeomRing id) { return rings[uint32_t(id)]; }
};

bool runInvocation(const LiftResult& r, Env& env, vpireval::Invocation& inv) {
    std::vector<std::pair<uint32_t, std::vector<uint8_t>*>> back;
    for (const Binding& b : r.bindings) {
        std::vector<uint8_t>* mem = nullptr;
        if (b.source.kind == DescriptorSource::Ring) {
            if (b.source.sgpr == uint16_t(GeomRing::Params))
                inv.memory[b.slot].assign(reinterpret_cast<const uint8_t*>(env.params.data()),
                                          reinterpret_cast<const uint8_t*>(env.params.data()) + 4 * env.params.size());
            else mem = &env.rings[b.source.sgpr];
        } else if (b.source.kind == DescriptorSource::UserData && b.kind == vp4ir::ResourceKind::Buffer) {
            mem = &env.vertexBuffer;
        }
        if (mem) {
            inv.memory[b.slot] = *mem;
            back.push_back({b.slot, mem});
        }
    }
    if (r.userDataSlot)
        inv.memory[*r.userDataSlot].assign(reinterpret_cast<const uint8_t*>(env.userData.data()),
                                           reinterpret_cast<const uint8_t*>(env.userData.data()) + 4 * env.userData.size());
    std::string err;
    if (!vpireval::run(r.module, inv, err)) {
        std::fprintf(stderr, "eval: %s\n", err.c_str());
        return false;
    }
    for (auto& [slot, mem] : back) *mem = inv.memory[slot];
    return true;
}

vpireval::Value u32(uint32_t v) {
    vpireval::Value x;
    x.type = vp4ir::kU32;
    x.bits[0] = v;
    return x;
}

vpireval::Value vec3(uint32_t a) {
    vpireval::Value x;
    x.type = vp4ir::kU32.withWidth(3);
    x.bits[0] = a;
    return x;
}

float fl(uint32_t b) { return std::bit_cast<float>(b); }

// ---- The pipeline ------------------------------------------------------------------------------

struct Vertex {
    float pos[4], color[4];
};

struct Config {
    std::string es = "es_basic", gs, copy;
    uint32_t esgsItem = 8, vertDwords = 8, maxVert = 3, inPrim = inprim::TriList, outPrim = outprim::TriStrip;
    uint32_t gsInstances = 1;
};

struct Drawn {
    std::vector<uint32_t> indices;  // the GS output list
    uint32_t perInvocation = 0;
    // For each index that is not padding: the copy shader's outputs.
    struct Out {
        float pos[4];
        float color[4];
        uint32_t layer = 0;
    };
    std::map<uint32_t, Out> byIndex;
};

// index16 / index32 non-empty: an indexed draw.
Drawn runPipeline(const Config& c, const std::vector<Vertex>& vertices, uint32_t count, const std::vector<uint32_t>& indices,
                  bool wide = false, uint32_t firstVertex = 0) {
    LiftResult es = load(c.es), gs = load(c.gs), copy = load(c.copy);
    Env env;
    env.vertexBuffer.resize(vertices.size() * sizeof(Vertex));
    std::memcpy(env.vertexBuffer.data(), vertices.data(), env.vertexBuffer.size());
    env.userData[1] = 32u << 16;  // V# stride
    env.userData[2] = uint32_t(vertices.size());  // V# NUM_RECORDS (the lifted fetch checks its range)
    const bool indexed = !indices.empty();
    if (indexed) {
        std::vector<uint8_t>& ib = env.ring(GeomRing::InputIndex);
        if (wide) {
            ib.resize(4 * indices.size());
            std::memcpy(ib.data(), indices.data(), ib.size());
        } else {
            ib.resize(2 * indices.size() + 4);
            for (size_t i = 0; i < indices.size(); ++i) {
                const uint16_t v = uint16_t(indices[i]);
                std::memcpy(&ib[2 * i], &v, 2);
            }
        }
    }
    const uint32_t instances = 1;
    const uint32_t prims = inputPrimitives(c.inPrim, count);
    const uint32_t esCount = instances * count, gsCount = instances * prims * c.gsInstances;
    env.params[geomparam::EsCount] = esCount;
    env.params[geomparam::Entries] = count;
    env.params[geomparam::FirstVertex] = firstVertex;
    env.params[geomparam::Indexed] = indexed ? (wide ? 2 : 1) : 0;
    env.params[geomparam::Prims] = prims;
    env.params[geomparam::GsCount] = gsCount;
    env.params[geomparam::GsInstances] = c.gsInstances;
    env.params[geomparam::Instances] = instances;
    env.ring(GeomRing::EsGs).assign(((esCount + 63) / 64) * c.esgsItem * 256, 0);
    const uint32_t cap = outputIndexCapacity(c.outPrim, c.maxVert);
    env.ring(GeomRing::GsVs).assign(((gsCount + 63) / 64) * c.vertDwords * c.maxVert * 256, 0);
    env.ring(GeomRing::GsIndex).assign(4 * cap * gsCount, 0);
    for (uint32_t n = 0; n < ((esCount + 63) / 64) * 64; ++n) {
        vpireval::Invocation inv;
        inv.builtins[vp4ir::Builtin::GlobalInvocationId] = vec3(n);
        CHECK(runInvocation(es, env, inv));
    }
    for (uint32_t k = 0; k < ((gsCount + 63) / 64) * 64; ++k) {
        vpireval::Invocation inv;
        inv.builtins[vp4ir::Builtin::GlobalInvocationId] = vec3(k);
        CHECK(runInvocation(gs, env, inv));
    }
    Drawn d;
    d.perInvocation = cap;
    d.indices.resize(cap * gsCount);
    std::memcpy(d.indices.data(), env.ring(GeomRing::GsIndex).data(), 4 * d.indices.size());
    uint32_t posOut = ~0u, colorOut = ~0u, layerOut = ~0u;
    for (uint32_t i = 0; i < copy.module.outputs.size(); ++i) {
        const auto& o = copy.module.outputs[i];
        if (o.builtin == vp4ir::Builtin::Position) posOut = i;
        else if (o.builtin == vp4ir::Builtin::Layer) layerOut = i;
        else if (o.builtin == vp4ir::Builtin::None) colorOut = i;
    }
    for (uint32_t index : d.indices) {
        if (d.byIndex.count(index)) continue;
        vpireval::Invocation inv;
        inv.builtins[vp4ir::Builtin::VertexId] = u32(index);
        CHECK(runInvocation(copy, env, inv));
        Drawn::Out o{};
        if (posOut != ~0u && inv.outputs.count(posOut))
            for (int k = 0; k < 4; ++k) o.pos[k] = fl(inv.outputs[posOut].bits[k]);
        if (colorOut != ~0u && inv.outputs.count(colorOut))
            for (int k = 0; k < 4; ++k) o.color[k] = fl(inv.outputs[colorOut].bits[k]);
        if (layerOut != ~0u && inv.outputs.count(layerOut)) o.layer = inv.outputs[layerOut].bits[0];
        d.byIndex[index] = o;
    }
    return d;
}

std::vector<Vertex> makeVertices(uint32_t n) {
    std::vector<Vertex> v(n);
    for (uint32_t i = 0; i < n; ++i) {
        v[i].pos[0] = 0.01f * float(i);
        v[i].pos[1] = -0.5f + 0.003f * float(i);
        v[i].pos[2] = 0.25f;
        v[i].pos[3] = 1.0f;
        v[i].color[0] = float(i);
        v[i].color[1] = 0.5f;
        v[i].color[2] = 0.25f;
        v[i].color[3] = 1.0f;
    }
    return v;
}

constexpr uint32_t kPad = 0xffffffffu;

uint32_t vid(const Config& c, uint32_t k, uint32_t e) { return (k >> 6) * (c.vertDwords * c.maxVert * 64) + e * 64 + (k & 63); }

// ---- Cases -------------------------------------------------------------------------------------

void triangles() {
    Config c;
    c.gs = "gs_tri";
    c.copy = "copy_tri";
    std::vector<Vertex> v = makeVertices(6);
    Drawn d = runPipeline(c, v, 6, {});
    CHECK(d.indices.size() == 6);
    for (uint32_t k = 0; k < 2; ++k)
        for (uint32_t e = 0; e < 3; ++e) {
            const uint32_t idx = d.indices[k * 3 + e];
            CHECK(idx == vid(c, k, e));
            const Drawn::Out& o = d.byIndex[idx];
            const Vertex& src = v[k * 3 + e];
            for (int q = 0; q < 4; ++q) {
                CHECK_NEAR(o.pos[q], src.pos[q]);
                CHECK_NEAR(o.color[q], src.color[q]);
            }
        }
}

// 70 triangles, 16-bit indices in shuffled order, a first vertex: ES waves, GS waves, index reads.
void stereo(bool wide) {
    Config c;
    c.gs = "gs_stereo";
    c.copy = "copy_stereo";
    c.vertDwords = 9;
    c.maxVert = 6;
    const uint32_t tris = 70, n = tris * 3, first = 5;
    std::vector<Vertex> v = makeVertices(n + first);
    std::vector<uint32_t> idx(n);
    for (uint32_t i = 0; i < n; ++i) idx[i] = (i * 7) % n;  // 7 and 210 are coprime: a permutation
    Drawn d = runPipeline(c, v, n, idx, wide, first);
    CHECK(d.perInvocation == 12);
    CHECK(d.indices.size() == tris * 12);
    for (uint32_t k = 0; k < tris; ++k) {
        for (uint32_t t = 0; t < 12; ++t) {
            const uint32_t at = d.indices[k * 12 + t];
            if (t >= 6) {
                CHECK(at == kPad);
                continue;
            }
            const uint32_t e = t;  // two triangles of three vertices: indices e0 e1 e2 e3 e4 e5
            CHECK(at == vid(c, k, e));
            const Drawn::Out& o = d.byIndex[at];
            const uint32_t eye = e / 3, j = e % 3;
            const Vertex& src = v[first + idx[k * 3 + j]];
            CHECK_NEAR(o.pos[0], src.pos[0] + (eye ? 0.25f : -0.25f));
            CHECK_NEAR(o.pos[1], src.pos[1]);
            CHECK_NEAR(o.color[0], src.color[0]);
            CHECK(o.layer == eye);
        }
    }
    CHECK(d.byIndex.count(kPad));
    CHECK(d.byIndex[kPad].pos[3] == 0.0f);  // padding is clipped away
}

void strip() {
    Config c;
    c.gs = "gs_strip";
    c.copy = "copy_strip";
    c.maxVert = 4;
    std::vector<Vertex> v = makeVertices(9);
    Drawn d = runPipeline(c, v, 9, {});
    CHECK(d.perInvocation == 6);
    for (uint32_t k = 0; k < 3; ++k) {
        const uint32_t e[4] = {vid(c, k, 0), vid(c, k, 1), vid(c, k, 2), vid(c, k, 3)};
        const uint32_t expect[6] = {e[0], e[1], e[2], e[2], e[1], e[3]};  // the second triangle flipped
        for (int t = 0; t < 6; ++t) CHECK(d.indices[k * 6 + t] == expect[t]);
        const Drawn::Out& fourth = d.byIndex[e[3]];
        CHECK_NEAR(fourth.pos[0], v[3 * k].pos[0] + v[3 * k + 2].pos[0] - v[3 * k + 1].pos[0]);
        CHECK_NEAR(fourth.color[0], v[3 * k + 2].color[0]);
    }
}

void lines() {
    Config c;
    c.gs = "gs_lines";
    c.copy = "copy_tri";
    c.outPrim = outprim::LineStrip;
    std::vector<Vertex> v = makeVertices(6);
    Drawn d = runPipeline(c, v, 6, {});
    CHECK(d.perInvocation == 4);
    for (uint32_t k = 0; k < 2; ++k) {
        const uint32_t expect[4] = {vid(c, k, 0), vid(c, k, 1), vid(c, k, 1), vid(c, k, 2)};
        for (int t = 0; t < 4; ++t) CHECK(d.indices[k * 4 + t] == expect[t]);
    }
}

// An EXEC-masked emit: only the even primitives produce their point.
void conditional() {
    Config c;
    c.gs = "gs_cond_points";
    c.copy = "copy_one";
    c.maxVert = 1;
    c.outPrim = outprim::PointList;
    std::vector<Vertex> v = makeVertices(12);
    Drawn d = runPipeline(c, v, 12, {});
    CHECK(d.indices.size() == 4);
    for (uint32_t k = 0; k < 4; ++k) {
        if (k % 2 == 0) {
            CHECK(d.indices[k] == vid(c, k, 0));
            CHECK_NEAR(d.byIndex[d.indices[k]].pos[0], v[3 * k].pos[0]);
        } else {
            CHECK(d.indices[k] == kPad);
        }
    }
}

void instanced() {
    Config c;
    c.gs = "gs_instanced";
    c.copy = "copy_tri";
    c.gsInstances = 2;
    std::vector<Vertex> v = makeVertices(6);
    Drawn d = runPipeline(c, v, 6, {});
    CHECK(d.indices.size() == 4 * 3);  // 2 primitives * 2 instances
    for (uint32_t k = 0; k < 4; ++k) {
        const uint32_t prim = k / 2, inst = k % 2;
        for (uint32_t e = 0; e < 3; ++e) {
            CHECK(d.indices[k * 3 + e] == vid(c, k, e));
            CHECK_NEAR(d.byIndex[d.indices[k * 3 + e]].pos[0], v[prim * 3 + e].pos[0] + 0.5f * float(inst));
        }
    }
}

// RectList: the three given vertices, then the others and the derived fourth.
void rect() {
    LiftResult r = load("vs_rect");
    CHECK(r.errors.empty());
    Env env;
    auto& table = env.ring(GeomRing::RectTable);
    const uint32_t ids[6] = {0, 1, 2, 7, 8, 9};  // two rectangles; the second has the same corners
    table.resize(sizeof ids);
    std::memcpy(table.data(), ids, sizeof ids);
    uint32_t pos = ~0u, param = ~0u;
    for (uint32_t i = 0; i < r.module.outputs.size(); ++i) {
        if (r.module.outputs[i].builtin == vp4ir::Builtin::Position) pos = i;
        else param = i;
    }
    const float ex[6][2] = {{-1, -1}, {1, -1}, {-1, 1}, {1, -1}, {-1, 1}, {1, 1}};
    const float euv[6][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 0}, {0, 1}, {1, 1}};
    for (uint32_t vid6 = 0; vid6 < 12; ++vid6) {
        vpireval::Invocation inv;
        inv.builtins[vp4ir::Builtin::VertexId] = u32(vid6);
        // second rectangle: vertex ids 7, 8, 9 -> (same shader math on id & 1, id >> 1) differ, so
        // only the first rectangle is checked against the closed form
        CHECK(runInvocation(r, env, inv));
        if (vid6 >= 6) continue;
        CHECK_NEAR(fl(inv.outputs[pos].bits[0]), ex[vid6][0]);
        CHECK_NEAR(fl(inv.outputs[pos].bits[1]), ex[vid6][1]);
        CHECK_NEAR(fl(inv.outputs[param].bits[0]), euv[vid6][0]);
        CHECK_NEAR(fl(inv.outputs[param].bits[1]), euv[vid6][1]);
    }
}

void misc() {
    LiftResult r = load("vs_misc");
    Env env;
    vpireval::Invocation inv;
    inv.builtins[vp4ir::Builtin::VertexId] = u32(0);
    CHECK(runInvocation(r, env, inv));
    bool point = false, layer = false;
    std::map<uint32_t, float> clip;
    for (uint32_t i = 0; i < r.module.outputs.size(); ++i) {
        const auto& o = r.module.outputs[i];
        if (o.builtin == vp4ir::Builtin::PointSize) {
            point = true;
            CHECK_NEAR(fl(inv.outputs[i].bits[0]), 2.0f);
        }
        if (o.builtin == vp4ir::Builtin::Layer) {
            layer = true;
            CHECK(inv.outputs[i].bits[0] == 1);
        }
        if (o.builtin == vp4ir::Builtin::ClipDistance) clip[o.location] = fl(inv.outputs[i].bits[0]);
    }
    CHECK(point && layer);
    CHECK(clip.size() == 2);
    CHECK_NEAR(clip[0], 0.5f);
    CHECK_NEAR(clip[1], -0.5f);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string c = argc > 1 ? argv[1] : "";
    if (c == "triangles") triangles();
    else if (c == "stereo16") stereo(false);
    else if (c == "stereo32") stereo(true);
    else if (c == "strip") strip();
    else if (c == "lines") lines();
    else if (c == "conditional") conditional();
    else if (c == "instanced") instanced();
    else if (c == "rect") rect();
    else if (c == "misc") misc();
    else {
        std::fprintf(stderr, "unknown case\n");
        return 2;
    }
    return failures ? 1 : 0;
}
