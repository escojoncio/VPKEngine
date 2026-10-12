// SPIR-V -> VPIR -> MSL on our own shaders (tests/glsl, compiled with glslang; `opt` = after
// spirv-opt -O, which is what shipping titles carry: phis, merged blocks, folded constants).
// Each case lowers with a driver-style layout, verifies, emits MSL and checks what the lowering
// promises (bindings reported, multiview, the subpass-read pattern, workgroup size…).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <fstream>
#include <string>
#include <vector>

#include "vpir/passes.h"
#include "vpir/text.h"
#include "vpir/verify.h"
#include "vpmsl/msl.h"
#include "vpvk/spirv_frontend.h"

namespace {

int failures = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        if (!(c)) {                                                               \
            std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);  \
            ++failures;                                                           \
        }                                                                         \
    } while (0)

std::vector<uint32_t> readWords(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<uint32_t> w(bytes.size() / 4);
    std::memcpy(w.data(), bytes.data(), w.size() * 4);
    return w;
}

spirv::SpirvBinding bind(uint32_t set, uint32_t binding, vpgpu::BindKind kind, uint16_t slot, uint16_t samplerSlot = 0xFFFF,
                         vpgpu::Format texel = vpgpu::Format::Undefined) {
    spirv::SpirvBinding b;
    b.set = set, b.binding = binding, b.kind = kind, b.slot = slot, b.samplerSlot = samplerSlot, b.texelFormat = texel;
    return b;
}

struct Result {
    bool ok = false;
    vpir::Module module;
    spirv::SpirvInterface iface;
    vpmsl::MslShader msl;
};

Result run(const std::string& path, const spirv::SpirvLowerOptions& o, bool fetch = false) {
    Result r;
    std::vector<uint32_t> words = readWords(path);
    std::string error;
    spirv::SpirvModule* m = spirv::parse(words.data(), words.size(), error);
    if (!m) {
        std::fprintf(stderr, "  parse: %s\n", error.c_str());
        return r;
    }
    bool lowered = spirv::lower(*m, o, r.module, r.iface, error);
    spirv::release(m);
    if (!lowered) {
        std::fprintf(stderr, "  lower: %s\n", error.c_str());
        return r;
    }
    auto errs = vpir::verify(r.module);
    if (!errs.empty()) {
        std::fprintf(stderr, "%s\n", vpir::print(r.module).c_str());
        for (auto& e : errs) std::fprintf(stderr, "  verify: %s\n", e.c_str());
        return r;
    }
    if (fetch) {
        // The driver's step for input attachments: texture slot -> colour attachment.
        std::map<uint32_t, uint32_t> att{{0, 0}, {1, 1}};
        auto rej = vpir::toFramebufferFetch(r.module, att, {});
        for (auto& x : rej) std::fprintf(stderr, "  fbfetch rejected: %s\n", x.reason.c_str());
        CHECK(rej.empty());
    }
    vpmsl::EmitResult e = vpmsl::emitMsl(r.module);
    if (!e.errors.empty()) {
        for (auto& x : e.errors) std::fprintf(stderr, "  msl: %s\n", x.c_str());
        return r;
    }
    r.msl = e.shader;
    r.ok = true;
    if (std::getenv("VPSPIRV_PRINT")) std::printf("%s\n%s\n", vpir::print(r.module).c_str(), r.msl.source.c_str());
    return r;
}

const spirv::SpirvBinding* used(const Result& r, uint32_t set, uint32_t binding) {
    for (auto& b : r.iface.bindings)
        if (b.set == set && b.binding == binding) return &b;
    return nullptr;
}

void meshMultiview(const std::string& dir, const std::string& v) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Vertex;
    o.viewCount = 2;
    o.layout = {bind(0, 0, vpgpu::BindKind::ConstantBuffer, 0), bind(1, 0, vpgpu::BindKind::ConstantBuffer, 1)};
    o.pushConstantSlot = 2;
    o.vertexBuffers = {{0, 32, false, 3}, {1, 16, true, 4}};
    o.vertexAttributes = {{0, 0, 0, vpgpu::Format::RGBA32Float}, {1, 0, 12, vpgpu::Format::RGBA8Snorm},
                          {2, 0, 16, vpgpu::Format::RG16Float}, {3, 1, 0, vpgpu::Format::RGBA32Float}};
    o.vertexAttributes[0].format = vpgpu::Format::RGBA32Float;  // inPos reads xyz of a float4 slot
    Result r = run(dir + "/mesh_multiview.vert" + (v.empty() ? "" : ".opt") + ".spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    CHECK(used(r, 0, 0) && used(r, 1, 0));
    CHECK(r.iface.usesPushConstants);
    CHECK(used(r, spirv::kVertexBufferSet, 0) && used(r, spirv::kVertexBufferSet, 1));
    bool layer = false, instance = false;
    for (auto& out : r.module.outputs) layer |= out.builtin == vpir::Builtin::Layer;
    for (auto& in : r.module.inputs) instance |= in.builtin == vpir::Builtin::InstanceId;
    CHECK(layer && instance);
    CHECK(r.msl.source.find("render_target_array_index") != std::string::npos);
}

void lit(const std::string& dir, const std::string& v) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Fragment;
    o.viewCount = 2;
    o.layout = {bind(0, 1, vpgpu::BindKind::Texture, 0, 0), bind(0, 2, vpgpu::BindKind::Texture, 1),
                bind(0, 3, vpgpu::BindKind::Sampler, 1), bind(0, 4, vpgpu::BindKind::Texture, 2, 2),
                bind(0, 5, vpgpu::BindKind::Buffer, 0), bind(0, 6, vpgpu::BindKind::ConstantBuffer, 1)};
    Result r = run(dir + "/lit.frag" + (v.empty() ? "" : ".opt") + ".spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    for (uint32_t b : {1u, 2u, 3u, 4u, 5u, 6u}) CHECK(used(r, 0, b));
    CHECK(used(r, 0, 5) && !used(r, 0, 5)->written);
    CHECK(r.iface.colorOutputs == 1);
    bool depth = false;
    for (auto& res : r.module.resources) depth |= res.kind == vpir::ResourceKind::Texture && res.depth;
    CHECK(depth);  // the shadow array texture compares
    CHECK(r.msl.source.find("discard_fragment") != std::string::npos);
}

void cull(const std::string& dir, const std::string& v) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0), bind(0, 1, vpgpu::BindKind::Buffer, 1),
                bind(0, 2, vpgpu::BindKind::StorageTexture, 0),
                bind(0, 3, vpgpu::BindKind::Buffer, 2, 0xFFFF, vpgpu::Format::R32Float)};
    o.pushConstantSlot = 3;
    Result r = run(dir + "/cull.comp" + (v.empty() ? "" : ".opt") + ".spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    CHECK(r.iface.localSize[0] == 64 && r.module.workgroupSize[0] == 64);
    CHECK(r.module.sharedBytes == 4 + 64 * 4);
    CHECK(used(r, 0, 1) && used(r, 0, 1)->written);
    CHECK(r.msl.source.find("atomic") != std::string::npos);
}

void subpass(const std::string& dir, const std::string& v) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Fragment;
    o.layout = {bind(0, 0, vpgpu::BindKind::Texture, 0), bind(0, 1, vpgpu::BindKind::Texture, 1)};
    Result r = run(dir + "/subpass.frag" + (v.empty() ? "" : ".opt") + ".spv", o, true);
    CHECK(r.ok);
    if (!r.ok) return;
    CHECK(r.iface.colorOutputs == 3);
    CHECK(r.msl.source.find("[[color(0)]]") != std::string::npos);
    CHECK(r.msl.source.find("[[color(1)]]") != std::string::npos);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: vpspirv_tests <case> <plain|opt> <tests dir>\n");
        return 2;
    }
    std::string c = argv[1], v = std::string(argv[2]) == "opt" ? "opt" : "", dir = argv[3];
    if (c == "mesh_multiview") meshMultiview(dir, v);
    else if (c == "lit") lit(dir, v);
    else if (c == "cull") cull(dir, v);
    else if (c == "subpass") subpass(dir, v);
    else {
        std::fprintf(stderr, "unknown case %s\n", c.c_str());
        return 2;
    }
    if (failures) std::fprintf(stderr, "%d FAILED\n", failures);
    return failures ? 1 : 0;
}
