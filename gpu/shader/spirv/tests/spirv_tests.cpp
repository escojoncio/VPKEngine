// SPIR-V -> VPIR -> MSL on our own shaders (tests/glsl, compiled with glslang; `opt` = after
// spirv-opt -O, which is what shipping titles carry: phis, merged blocks, folded constants).
// Each case lowers with a driver-style layout, verifies, emits MSL and checks what the lowering
// promises (bindings reported, multiview, the subpass-read pattern, workgroup size…).
#include <pthread.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <fstream>
#include <string>
#include <vector>

#include "vpir/passes.h"
#include "vpir/text.h"
#include "vpir/verify.h"
#include "vpmsl/msl.h"
#include "vpvk/spirv_frontend.h"
#include "vpir_eval.h"

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

Result runWords(const std::vector<uint32_t>& words, const spirv::SpirvLowerOptions& o, bool fetch = false,
                const vpmsl::EmitOptions& eo = {});
Result run(const std::string& path, const spirv::SpirvLowerOptions& o, bool fetch = false,
           const vpmsl::EmitOptions& eo = {}) {
    return runWords(readWords(path), o, fetch, eo);
}
Result runWords(const std::vector<uint32_t>& words, const spirv::SpirvLowerOptions& o, bool fetch, const vpmsl::EmitOptions& eo) {
    Result r;
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
        // The driver's step for input attachments: texture slot -> colour attachment. The second input
        // (float) reads attachment 2: attachment 1 is written as uvec4, and reading it as float4 is a
        // real type conflict the verifier now rejects.
        std::map<uint32_t, uint32_t> att{{0, 0}, {1, 2}};
        auto rej = vpir::toFramebufferFetch(r.module, att, {});
        for (auto& x : rej) std::fprintf(stderr, "  fbfetch rejected: %s\n", x.reason.c_str());
        CHECK(rej.empty());
    }
    vpmsl::EmitResult e = vpmsl::emitMsl(r.module, eo);
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
    // Foveation: with viewportIndex the view is also the viewport index (one rate map layer / viewport per view);
    // without it the output does not exist (the default MSL is the one of before).
    CHECK(r.msl.source.find("viewport_array_index") == std::string::npos);
    o.viewportIndex = true;
    Result vpi = run(dir + "/mesh_multiview.vert" + (v.empty() ? "" : ".opt") + ".spv", o);
    CHECK(vpi.ok && vpi.msl.source.find("viewport_array_index") != std::string::npos);
    o.viewportIndex = false;
    // Views 0 and 2 (mask 0b101): view 1 renders to layer 2; one view at bit 2 (mask 0b100)
    // still writes the layer although only one instance per draw is made.
    o.viewMask = 0b101;
    Result g = run(dir + "/mesh_multiview.vert" + (v.empty() ? "" : ".opt") + ".spv", o);
    CHECK(g.ok && g.msl.source.find("2u") != std::string::npos && g.msl.source != r.msl.source);
    o.viewCount = 1, o.viewMask = 0b100;
    Result one = run(dir + "/mesh_multiview.vert" + (v.empty() ? "" : ".opt") + ".spv", o);
    CHECK(one.ok && one.msl.source.find("render_target_array_index") != std::string::npos);
    // VK_EXT_vertex_attribute_divisor: a divided instance binding reads base_instance; divisor 1 does not.
    o.viewCount = 1, o.viewMask = 0;
    Result plain = run(dir + "/mesh_multiview.vert" + (v.empty() ? "" : ".opt") + ".spv", o);
    CHECK(plain.ok && plain.msl.source.find("base_instance") == std::string::npos);
    for (uint32_t divisor : {4u, 0u}) {
        o.vertexBuffers[1].divisor = divisor;
        Result dv = run(dir + "/mesh_multiview.vert" + (v.empty() ? "" : ".opt") + ".spv", o);
        CHECK(dv.ok && dv.msl.source.find("base_instance") != std::string::npos && dv.msl.source != plain.msl.source);
    }
    o.vertexBuffers[1].divisor = 1;
}

// Vertex shader of a layered (not multiview) pass, VK_EXT_shader_viewport_index_layer: OpCapability ShaderViewportIndexLayerEXT;
// gl_Layer = gl_InstanceIndex; gl_Position = (0,0,0,1). What Unity emits for a vertex SV_RenderTargetArrayIndex.
const std::vector<uint32_t> kLayerFromInstanceVert = {
    0x07230203, 0x00010000, 0x00000000, 0x00000012, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x00001486, 0x000a000a, 0x5f565053, 0x5f545845, 0x64616873, 0x765f7265, 0x70776569, 0x5f74726f,
    0x65646e69, 0x616c5f78, 0x00726579, 0x0003000e, 0x00000000, 0x00000001, 0x0008000f, 0x00000000,
    0x0000000f, 0x6e69616d, 0x00000000, 0x00000006, 0x00000009, 0x0000000b, 0x00040047, 0x00000006,
    0x0000000b, 0x00000000, 0x00040047, 0x00000009, 0x0000000b, 0x00000009, 0x00040047, 0x0000000b,
    0x0000000b, 0x0000002b, 0x00020013, 0x00000001, 0x00030021, 0x00000002, 0x00000001, 0x00030016,
    0x00000003, 0x00000020, 0x00040017, 0x00000004, 0x00000003, 0x00000004, 0x00040020, 0x00000005,
    0x00000003, 0x00000004, 0x0004003b, 0x00000005, 0x00000006, 0x00000003, 0x00040015, 0x00000007,
    0x00000020, 0x00000001, 0x00040020, 0x00000008, 0x00000003, 0x00000007, 0x0004003b, 0x00000008,
    0x00000009, 0x00000003, 0x00040020, 0x0000000a, 0x00000001, 0x00000007, 0x0004003b, 0x0000000a,
    0x0000000b, 0x00000001, 0x0004002b, 0x00000003, 0x0000000c, 0x00000000, 0x0004002b, 0x00000003,
    0x0000000d, 0x3f800000, 0x0007002c, 0x00000004, 0x0000000e, 0x0000000c, 0x0000000c, 0x0000000c,
    0x0000000d, 0x00050036, 0x00000001, 0x0000000f, 0x00000000, 0x00000002, 0x000200f8, 0x00000010,
    0x0004003d, 0x00000007, 0x00000011, 0x0000000b, 0x0003003e, 0x00000009, 0x00000011, 0x0003003e,
    0x00000006, 0x0000000e, 0x000100fd, 0x00010038,
};

// Fragment shader writing gl_ViewIndex to a uint colour (tests/asm style, assembled with spvasm.py):
//   OpCapability MultiView; %vi = Input uint BuiltIn ViewIndex Flat; out uint Location 0; store load(%vi).
const std::vector<uint32_t> kViewIndexFrag = {
    0x07230203, 0x00010000, 0x00000000, 0x0000000b, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x00001157, 0x0003000e, 0x00000000, 0x00000001, 0x0007000f, 0x00000004, 0x00000001, 0x6e69616d,
    0x00000000, 0x00000002, 0x00000003, 0x00030010, 0x00000001, 0x00000007, 0x00040047, 0x00000002,
    0x0000000b, 0x00001158, 0x00030047, 0x00000002, 0x0000000e, 0x00040047, 0x00000003, 0x0000001e,
    0x00000000, 0x00020013, 0x00000004, 0x00030021, 0x00000005, 0x00000004, 0x00040015, 0x00000006,
    0x00000020, 0x00000000, 0x00040020, 0x00000007, 0x00000001, 0x00000006, 0x00040020, 0x00000008,
    0x00000003, 0x00000006, 0x0004003b, 0x00000007, 0x00000002, 0x00000001, 0x0004003b, 0x00000008,
    0x00000003, 0x00000003, 0x00050036, 0x00000004, 0x00000001, 0x00000000, 0x00000005, 0x000200f8,
    0x00000009, 0x0004003d, 0x00000006, 0x0000000a, 0x00000002, 0x0003003e, 0x00000003, 0x0000000a,
    0x000100fd, 0x00010038};

// Multiview vertex shaders writing gl_ClipDistance[0] from gl_ViewIndex (assembled with spvasm.py, gl_Position = attribute
// location 0): kClipViewData stores float(gl_ViewIndex); kClipViewBranch stores a Function variable set to 1 only under
// `if (gl_ViewIndex == 0)`.
const std::vector<uint32_t> kClipViewData = {
    0x07230203, 0x00010300, 0x00000000, 0x00000019, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x00000020, 0x00020011, 0x00001157, 0x0006000a, 0x5f565053, 0x5f52484b, 0x746c756d, 0x65697669,
    0x00000077, 0x0003000e, 0x00000000, 0x00000001, 0x0009000f, 0x00000000, 0x00000001, 0x6e69616d,
    0x00000000, 0x00000002, 0x00000003, 0x00000004, 0x00000005, 0x00040047, 0x00000002, 0x0000001e,
    0x00000000, 0x00040047, 0x00000003, 0x0000000b, 0x00000000, 0x00040047, 0x00000004, 0x0000000b,
    0x00000003, 0x00040047, 0x00000005, 0x0000000b, 0x00001158, 0x00020013, 0x00000006, 0x00030021,
    0x00000007, 0x00000006, 0x00030016, 0x00000008, 0x00000020, 0x00040017, 0x00000009, 0x00000008,
    0x00000004, 0x00040015, 0x0000000a, 0x00000020, 0x00000000, 0x00040015, 0x0000000b, 0x00000020,
    0x00000001, 0x0004002b, 0x0000000a, 0x0000000c, 0x00000001, 0x0004002b, 0x0000000b, 0x0000000d,
    0x00000000, 0x0004001c, 0x0000000e, 0x00000008, 0x0000000c, 0x00040020, 0x0000000f, 0x00000001,
    0x00000009, 0x00040020, 0x00000010, 0x00000001, 0x0000000a, 0x00040020, 0x00000011, 0x00000003,
    0x00000009, 0x00040020, 0x00000012, 0x00000003, 0x0000000e, 0x00040020, 0x00000013, 0x00000003,
    0x00000008, 0x0004003b, 0x0000000f, 0x00000002, 0x00000001, 0x0004003b, 0x00000010, 0x00000005,
    0x00000001, 0x0004003b, 0x00000011, 0x00000003, 0x00000003, 0x0004003b, 0x00000012, 0x00000004,
    0x00000003, 0x00050036, 0x00000006, 0x00000001, 0x00000000, 0x00000007, 0x000200f8, 0x00000014,
    0x0004003d, 0x00000009, 0x00000015, 0x00000002, 0x0003003e, 0x00000003, 0x00000015, 0x0004003d,
    0x0000000a, 0x00000016, 0x00000005, 0x00040070, 0x00000008, 0x00000017, 0x00000016, 0x00050041,
    0x00000013, 0x00000018, 0x00000004, 0x0000000d, 0x0003003e, 0x00000018, 0x00000017, 0x000100fd,
    0x00010038};
const std::vector<uint32_t> kClipViewBranch = {
    0x07230203, 0x00010300, 0x00000000, 0x00000022, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x00000020, 0x00020011, 0x00001157, 0x0006000a, 0x5f565053, 0x5f52484b, 0x746c756d, 0x65697669,
    0x00000077, 0x0003000e, 0x00000000, 0x00000001, 0x0009000f, 0x00000000, 0x00000001, 0x6e69616d,
    0x00000000, 0x00000002, 0x00000003, 0x00000004, 0x00000005, 0x00040047, 0x00000002, 0x0000001e,
    0x00000000, 0x00040047, 0x00000003, 0x0000000b, 0x00000000, 0x00040047, 0x00000004, 0x0000000b,
    0x00000003, 0x00040047, 0x00000005, 0x0000000b, 0x00001158, 0x00020013, 0x00000006, 0x00030021,
    0x00000007, 0x00000006, 0x00020014, 0x00000008, 0x00030016, 0x00000009, 0x00000020, 0x00040017,
    0x0000000a, 0x00000009, 0x00000004, 0x00040015, 0x0000000b, 0x00000020, 0x00000000, 0x00040015,
    0x0000000c, 0x00000020, 0x00000001, 0x0004002b, 0x0000000b, 0x0000000d, 0x00000000, 0x0004002b,
    0x0000000b, 0x0000000e, 0x00000001, 0x0004002b, 0x0000000c, 0x0000000f, 0x00000000, 0x0004002b,
    0x00000009, 0x00000010, 0x00000000, 0x0004002b, 0x00000009, 0x00000011, 0x3f800000, 0x0004001c,
    0x00000012, 0x00000009, 0x0000000e, 0x00040020, 0x00000013, 0x00000001, 0x0000000a, 0x00040020,
    0x00000014, 0x00000001, 0x0000000b, 0x00040020, 0x00000015, 0x00000003, 0x0000000a, 0x00040020,
    0x00000016, 0x00000003, 0x00000012, 0x00040020, 0x00000017, 0x00000003, 0x00000009, 0x00040020,
    0x00000018, 0x00000007, 0x00000009, 0x0004003b, 0x00000013, 0x00000002, 0x00000001, 0x0004003b,
    0x00000014, 0x00000005, 0x00000001, 0x0004003b, 0x00000015, 0x00000003, 0x00000003, 0x0004003b,
    0x00000016, 0x00000004, 0x00000003, 0x00050036, 0x00000006, 0x00000001, 0x00000000, 0x00000007,
    0x000200f8, 0x00000019, 0x0004003b, 0x00000018, 0x0000001a, 0x00000007, 0x0003003e, 0x0000001a,
    0x00000010, 0x0004003d, 0x0000000a, 0x0000001b, 0x00000002, 0x0003003e, 0x00000003, 0x0000001b,
    0x0004003d, 0x0000000b, 0x0000001c, 0x00000005, 0x000500aa, 0x00000008, 0x0000001d, 0x0000001c,
    0x0000000d, 0x000300f7, 0x0000001e, 0x00000000, 0x000400fa, 0x0000001d, 0x0000001f, 0x0000001e,
    0x000200f8, 0x0000001f, 0x0003003e, 0x0000001a, 0x00000011, 0x000200f9, 0x0000001e, 0x000200f8,
    0x0000001e, 0x0004003d, 0x00000009, 0x00000020, 0x0000001a, 0x00050041, 0x00000017, 0x00000021,
    0x00000004, 0x0000000f, 0x0003003e, 0x00000021, 0x00000020, 0x000100fd, 0x00010038};

// Translator modes (VPVulkan's translatorMode): multiview through Metal vertex amplification and the
// native half2 conversion of half vertex attributes.
void amplify(const std::string& dir, const std::string& v) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Vertex;
    o.viewCount = 2;
    o.layout = {bind(0, 0, vpgpu::BindKind::ConstantBuffer, 0), bind(1, 0, vpgpu::BindKind::ConstantBuffer, 1)};
    o.pushConstantSlot = 2;
    o.vertexBuffers = {{0, 32, false, 3}, {1, 16, true, 4}};
    o.vertexAttributes = {{0, 0, 0, vpgpu::Format::RGBA32Float}, {1, 0, 12, vpgpu::Format::RGBA8Snorm},
                          {2, 0, 16, vpgpu::Format::RG16Float}, {3, 1, 0, vpgpu::Format::RGBA32Float}};
    const std::string vs = dir + "/mesh_multiview.vert" + (v.empty() ? "" : ".opt") + ".spv";
    {  // modes off: instancing, bit-exact half expansion
        Result r = run(vs, o);
        CHECK(r.ok && !r.iface.amplified);
        CHECK(r.msl.source.find("amplification_id") == std::string::npos);
        CHECK(r.msl.source.find("as_type<half2>") == std::string::npos);
        CHECK(r.msl.source.find("% 2u") != std::string::npos && r.msl.source.find("/ 2u") != std::string::npos);
        CHECK(r.msl.source.find("user(loc63)") == std::string::npos);
        CHECK(r.msl.source.find("vp_view_base") == std::string::npos);
    }
    o.amplify = true, o.nativeFetch = true;
    {
        Result r = run(vs, o);
        CHECK(r.ok && r.iface.amplified);
        const std::string& s = r.msl.source;
        CHECK(s.find("ushort in_amplification_id [[amplification_id]]") != std::string::npos);
        CHECK(s.find("uint(in_amplification_id)") != std::string::npos);
        // The view is the amplification id plus the encoder's view base (0, or v in a view-split encoder): bound at buffer 27.
        CHECK(s.find("(uint(in_amplification_id) + vp_view_base)") != std::string::npos);
        CHECK(s.find("constant uint& vp_view_base [[buffer(27)]]") != std::string::npos);
        // The view no longer comes from the instance: no modulo / division by the view count.
        CHECK(s.find("% 2u") == std::string::npos && s.find("/ 2u") == std::string::npos);
        // The per-instance attribute still reads instance_id (one instance per Vulkan instance).
        CHECK(s.find("[[instance_id]]") != std::string::npos);
        // Layer index stays 0 (the view mappings add the view); the view varying is flat.
        CHECK(s.find("[[render_target_array_index]]") != std::string::npos);
        CHECK(s.find("vp_out.layer = 0u;") != std::string::npos);
        CHECK(s.find("viewport_array_index") == std::string::npos);  // no foveation: no viewport output
        CHECK(s.find("[[user(loc63)]]") != std::string::npos);
        CHECK(s.find("as_type<half2>") != std::string::npos);
        bool viewOut = false;
        for (auto& out : r.module.outputs)
            viewOut |= out.location == 63 && out.builtin == vpir::Builtin::None && out.interp == vpir::Interp::Flat;
        CHECK(viewOut);
        // Foveation: the viewport index stays 0 too (the view mappings add the view to it).
        spirv::SpirvLowerOptions f = o;
        f.viewportIndex = true;
        Result fr = run(vs, f);
        CHECK(fr.ok && fr.iface.amplified && fr.msl.source.find("vp_out.viewport_index = 0u;") != std::string::npos);
    }
    {  // one view: nothing to amplify
        spirv::SpirvLowerOptions one = o;
        one.viewCount = 1;
        Result r = run(vs, one);
        CHECK(r.ok && !r.iface.amplified && r.msl.source.find("amplification_id") == std::string::npos);
        CHECK(r.msl.source.find("user(loc63)") == std::string::npos);
        CHECK(r.msl.source.find("vp_view_base") == std::string::npos);
    }
    {  // an irregular view mask maps views to other layers: instancing stays
        spirv::SpirvLowerOptions m = o;
        m.viewMask = 0b101;
        Result r = run(vs, m);
        CHECK(r.ok && !r.iface.amplified && r.msl.source.find("amplification_id") == std::string::npos);
        CHECK(r.msl.source.find("vp_view_base") == std::string::npos);
    }
    {  // ClipDistance would be computed once for both views: this shader keeps instancing (still writes the varying)
        spirv::SpirvLowerOptions c;
        c.entry = "main", c.stage = vpir::Stage::Vertex, c.viewCount = 2, c.amplify = true;
        Result r = run(dir + "/clip_invariant.vert.spv", c);
        CHECK(r.ok && !r.iface.amplified);
        CHECK(r.msl.source.find("amplification_id") == std::string::npos);
        CHECK(r.msl.source.find("% 2u") != std::string::npos);
        CHECK(r.msl.source.find("[[user(loc63)]]") != std::string::npos);
        // ampClip: its clip distance does not depend on the view, so it stays amplified (Metal's one value is right for both).
        c.ampClip = true;
        Result a = run(dir + "/clip_invariant.vert.spv", c);
        CHECK(a.ok && a.iface.amplified);
        CHECK(a.msl.source.find("[[amplification_id]]") != std::string::npos && a.msl.source.find("% 2u") == std::string::npos);
        CHECK(a.msl.source.find("[[clip_distance]]") != std::string::npos);
        // ... but a clip distance that depends on the view, by value or by a branch, keeps instancing.
        for (const auto* words : {&kClipViewData, &kClipViewBranch}) {
            Result d = runWords(*words, c);
            CHECK(d.ok && !d.iface.amplified);
            CHECK(d.msl.source.find("amplification_id") == std::string::npos && d.msl.source.find("% 2u") != std::string::npos);
        }
    }
    {  // fragment ViewIndex: the flat view varying in this mode, the layer otherwise
        spirv::SpirvLowerOptions f;
        f.entry = "main", f.stage = vpir::Stage::Fragment, f.viewCount = 2, f.amplify = true;
        Result r = runWords(kViewIndexFrag, f);
        CHECK(r.ok);
        CHECK(r.msl.source.find("user(loc63), flat") != std::string::npos);
        CHECK(r.msl.source.find("render_target_array_index") == std::string::npos);
        f.amplify = false;
        Result l = runWords(kViewIndexFrag, f);
        CHECK(l.ok);
        CHECK(l.msl.source.find("render_target_array_index") != std::string::npos);
        CHECK(l.msl.source.find("loc63") == std::string::npos);
    }
    {  // gl_Layer written by the vertex stage of a pass with no view mask: the render target layer, from the instance
        spirv::SpirvLowerOptions v;
        v.entry = "main", v.stage = vpir::Stage::Vertex;
        Result r = runWords(kLayerFromInstanceVert, v);
        CHECK(r.ok && !r.iface.amplified);
        CHECK(r.msl.source.find("[[render_target_array_index]]") != std::string::npos);
        CHECK(r.msl.source.find("[[instance_id]]") != std::string::npos && r.msl.source.find("vp_out.layer = ") != std::string::npos);
    }
    {  // declaresLayer (why a multiview pipeline is not amplified): BuiltIn Layer (9) on a variable; ViewIndex (0x1158) is not it
        auto layer = [](bool withLayer) {
            std::vector<uint32_t> w = kViewIndexFrag;
            for (size_t i = 5; i + 3 < w.size(); ++i)
                if (w[i] == 0x00040047 && w[i + 2] == 0x0b && w[i + 3] == 0x1158 && withLayer) w[i + 3] = 9;
            std::string error;
            spirv::SpirvModule* m = spirv::parse(w.data(), w.size(), error);
            const bool declares = m && spirv::declaresLayer(*m);
            spirv::release(m);
            return declares;
        };
        CHECK(!layer(false) && layer(true));
    }
    {  // declaresSampleBuiltin (forced MSAA leaves a game MSAA pass with such a fragment shader alone): BuiltIn SampleId (18) / SampleMask (20)
        auto sample = [](uint32_t builtin) {
            std::vector<uint32_t> w = kViewIndexFrag;
            for (size_t i = 5; i + 3 < w.size(); ++i)
                if (w[i] == 0x00040047 && w[i + 2] == 0x0b && w[i + 3] == 0x1158 && builtin) w[i + 3] = builtin;
            std::string error;
            spirv::SpirvModule* m = spirv::parse(w.data(), w.size(), error);
            const bool declares = m && spirv::declaresSampleBuiltin(*m);
            spirv::release(m);
            return declares;
        };
        CHECK(!sample(0) && sample(18) && sample(20) && !sample(9));
    }
}

// unpackFetch: unorm/snorm attributes of whole 4-byte words decode with the unpack_* builtins; the other formats
// (and the mode off) keep the extract + convert + multiply path.
void unpackFetch(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Vertex;
    o.layout = {bind(0, 0, vpgpu::BindKind::ConstantBuffer, 0), bind(1, 0, vpgpu::BindKind::ConstantBuffer, 1)};
    o.pushConstantSlot = 2;
    o.vertexBuffers = {{0, 32, false, 3}, {1, 16, true, 4}};
    const std::string vs = dir + "/mesh_multiview.vert.spv";
    struct Case {
        vpgpu::Format format;
        const char* fn;  // nullptr: not a whole word, old path
    };
    for (const Case& c : {Case{vpgpu::Format::RGBA8Snorm, "unpack_snorm4x8_to_float("},
                          Case{vpgpu::Format::RGBA8Unorm, "unpack_unorm4x8_to_float("},
                          Case{vpgpu::Format::BGRA8Unorm, "unpack_unorm4x8_to_float("},
                          Case{vpgpu::Format::RG16Snorm, "unpack_snorm2x16_to_float("},
                          Case{vpgpu::Format::RG16Unorm, "unpack_unorm2x16_to_float("},
                          Case{vpgpu::Format::RGBA16Unorm, "unpack_unorm2x16_to_float("},
                          Case{vpgpu::Format::RG8Unorm, nullptr}, Case{vpgpu::Format::R16Snorm, nullptr},
                          Case{vpgpu::Format::RGBA8Uint, nullptr}}) {
        o.vertexAttributes = {{0, 0, 0, vpgpu::Format::RGBA32Float}, {1, 0, 12, c.format},
                              {2, 0, 16, vpgpu::Format::RG16Float}, {3, 1, 0, vpgpu::Format::RGBA32Float}};
        o.unpackFetch = false;
        Result off = run(vs, o);
        CHECK(off.ok && off.msl.source.find("unpack_") == std::string::npos);
        o.unpackFetch = true;
        Result on = run(vs, o);
        CHECK(on.ok);
        if (!off.ok || !on.ok) continue;
        const std::string& s = on.msl.source;
        if (c.fn) {
            CHECK(s.find(c.fn) != std::string::npos);
            CHECK(s != off.msl.source);
        } else {
            CHECK(s == off.msl.source);
        }
    }
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

    // VPKTAA UI mask: an extra float output at [[color(7)]] (R8 attachment). Without the option the
    // MSL is untouched; with it the output is written at the end (alpha of colour 0 x2 saturated, or 1).
    CHECK(r.msl.source.find("uimask") == std::string::npos && r.msl.colorOutputs == 1);
    spirv::SpirvLowerOptions um = o;
    um.uiMaskSlot = 7;
    Result m1 = run(dir + "/lit.frag" + (v.empty() ? "" : ".opt") + ".spv", um);
    um.uiMaskAlpha = true;
    Result ma = run(dir + "/lit.frag" + (v.empty() ? "" : ".opt") + ".spv", um);
    CHECK(m1.ok && ma.ok);
    if (m1.ok && ma.ok) {
        CHECK(m1.iface.colorOutputs == 0x81 && m1.msl.colorOutputs == 0x81 && ma.msl.colorOutputs == 0x81);
        CHECK(m1.msl.source.find("[[color(7)]]") != std::string::npos && ma.msl.source.find("[[color(7)]]") != std::string::npos);
        CHECK(m1.msl.source != r.msl.source && ma.msl.source != m1.msl.source);  // alpha x2 vs constant 1
        CHECK(ma.msl.source.find("saturate") != std::string::npos);
        if (std::getenv("VPVK_DUMP_UIMASK_MSL")) std::fprintf(stderr, "---- alpha\n%s\n---- const\n%s\n", ma.msl.source.c_str(), m1.msl.source.c_str());
    }
    um.uiMaskSlot = 0;  // collides with the shader's own colour 0
    Result bad = run(dir + "/lit.frag" + (v.empty() ? "" : ".opt") + ".spv", um);
    CHECK(!bad.ok);

    // Cache stability: the same module under a pipeline layout with other stages' bindings first
    // (shifted slots) and in a single-view pass gives the same MSL with positional names (the
    // first-run precompile sees the module alone); the ids still differ by the argument slots.
    vpmsl::EmitOptions pos;
    pos.positionalNames = true;
    Result a = run(dir + "/lit.frag" + (v.empty() ? "" : ".opt") + ".spv", o, false, pos);
    spirv::SpirvLowerOptions shifted = o;
    shifted.viewCount = 1;  // lit.frag never reads gl_ViewIndex
    for (auto& b : shifted.layout) {
        b.slot = uint16_t(b.slot + 3);
        if (b.samplerSlot != 0xFFFF) b.samplerSlot = uint16_t(b.samplerSlot + 3);
    }
    shifted.layout.push_back(bind(0, 9, vpgpu::BindKind::Texture, 0));  // another stage's binding
    Result c = run(dir + "/lit.frag" + (v.empty() ? "" : ".opt") + ".spv", shifted, false, pos);
    CHECK(a.ok && c.ok);
    if (!a.ok || !c.ok) return;
    CHECK(a.msl.source == c.msl.source);
    CHECK(vpmsl::sourceHash(a.msl.source) == vpmsl::sourceHash(c.msl.source));
    CHECK(vpmsl::shaderId(a.msl) != vpmsl::shaderId(c.msl));
    CHECK(r.msl.source != a.msl.source);  // default naming unchanged (absolute slots)
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
    CHECK(used(r, 0, 2) && used(r, 0, 2)->written);  // the storage image written by an image atomic
    CHECK(used(r, 0, 3) && !used(r, 0, 3)->written);
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

// Compute with a WorkgroupSize spec composite (local_size_x_id), shared memory sized by it, the
// compute builtins and an atomic on a storage texel buffer (the skinning/particle counter pattern
// that used to fail with "internal error: vector"). Hand-assembled: gen_wg_spec_atomic.py.
void wgSpecAtomic(const std::string& dir, const std::string& v) {
    for (uint32_t size : {64u, 128u}) {
        spirv::SpirvLowerOptions o;
        o.entry = "main";
        o.stage = vpir::Stage::Compute;
        o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0, 0xFFFF, vpgpu::Format::R32Uint),
                    bind(0, 1, vpgpu::BindKind::Buffer, 1)};
        if (size != 64) o.specConstants = {{0, size}};
        (void)v;
        Result r = run(dir + "/wg_spec_atomic.comp.spv", o);
        CHECK(r.ok);
        if (!r.ok) return;
        CHECK(r.iface.localSize[0] == size && r.module.workgroupSize[0] == size);
        CHECK(r.module.sharedBytes == size * 4);
        CHECK(used(r, 0, 0) && used(r, 0, 0)->written);
        CHECK(used(r, 0, 1) && used(r, 0, 1)->written);
        CHECK(r.msl.source.find("atomic_fetch_add_explicit") != std::string::npos);
    }
}

// OpArrayLength on the runtime array that ends a storage block (after a 16-byte header): the
// length is read from a BufferSize argument of the block's slot (the backend writes the bound
// range's size there) and computed as (size - 16) / 4. Hand-assembled: gen_array_length.py.
void arrayLength(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0)};
    Result r = run(dir + "/array_length.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    CHECK(used(r, 0, 0) && used(r, 0, 0)->written);
    bool sizeArg = false, bufArg = false;
    for (const auto& a : r.msl.arguments) {
        sizeArg |= a.kind == vpir::ResourceKind::BufferSize && a.index == 0;
        bufArg |= a.kind == vpir::ResourceKind::Buffer && a.index == 0;
    }
    CHECK(sizeArg && bufArg);
    CHECK(r.msl.source.find("ulong") != std::string::npos);
    CHECK(r.msl.source.find("uint(vp_args.") != std::string::npos);
    CHECK(vpmsl::describeKey(r.msl).find("bs0") != std::string::npos);
    // VPVulkan's naming (positional per kind) with a BufferSize resource.
    vpmsl::EmitOptions eo;
    eo.positionalNames = true;
    Result p = run(dir + "/array_length.comp.spv", o, false, eo);
    CHECK(p.ok && p.msl.source.find("ulong bs0;") != std::string::npos);
}

// 1D images (sampler1DArray / sampler1D) lower to 2D textures one texel high: coordinates gain
// y (0.5 normalized, 0 integer), offsets gain 0, size queries drop the height, projective
// coordinates keep the row centred. Hand-assembled: gen_tex1d.py.
void tex1d(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Fragment;
    o.layout = {bind(0, 0, vpgpu::BindKind::Texture, 0, 0), bind(0, 1, vpgpu::BindKind::Texture, 1, 1)};
    Result r = run(dir + "/tex1d.frag.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    CHECK(r.msl.source.find("texture1d") == std::string::npos);
    CHECK(r.msl.source.find("texture2d_array<float") != std::string::npos);
    CHECK(r.msl.source.find("texture2d<float") != std::string::npos);
    CHECK(r.msl.source.find("int2(1, 0)") != std::string::npos || r.msl.source.find("int2(1,0)") != std::string::npos);
    if (failures || std::getenv("VP_PRINT")) std::fprintf(stderr, "%s\n", r.msl.source.c_str());
}

// sampler2DMS: texture2d_ms, texelFetch's sample index as read()'s second argument and
// textureSamples() from get_num_samples(); with multisampleAsSingle (VPVulkan forcing MSAA off)
// a plain texture2d read at sample 0 and a count of 1.
void texMs(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Fragment;
    o.layout = {bind(0, 0, vpgpu::BindKind::Texture, 0, 0)};
    Result r = run(dir + "/tex_ms.frag.spv", o);
    CHECK(r.ok);
    if (r.ok) {
        CHECK(r.msl.source.find("texture2d_ms<int>") != std::string::npos);
        CHECK(r.msl.source.find("get_num_samples()") != std::string::npos);
        if (failures || std::getenv("VP_PRINT")) std::fprintf(stderr, "%s\n", r.msl.source.c_str());
    }
    o.multisampleAsSingle = true;
    Result s = run(dir + "/tex_ms.frag.spv", o);
    CHECK(s.ok);
    if (s.ok) {
        CHECK(s.msl.source.find("texture2d<int>") != std::string::npos);
        CHECK(s.msl.source.find("_ms") == std::string::npos);
    }
}

// gl_ClipDistance through gl_PerVertex: one [[clip_distance]] array of the declared size, the
// written element stored by index. Hand-assembled: gen_clip.py.
void clip(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Vertex;
    Result r = run(dir + "/clip.vert.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    CHECK(r.msl.source.find("float vp_clip [[clip_distance]] [2];") != std::string::npos);
    CHECK(r.msl.source.find("vp_out.vp_clip[1] = ") != std::string::npos);
    if (failures || std::getenv("VP_PRINT")) std::fprintf(stderr, "%s\n", r.msl.source.c_str());
}

// Vulkan CTS regressions (glsl/fixes.{frag,vert,comp}, glslang --target-env vulkan1.1).
void fixes(const std::string& dir) {
    spirv::SpirvLowerOptions f;
    f.entry = "main";
    f.stage = vpir::Stage::Fragment;
    f.layout = {bind(0, 0, vpgpu::BindKind::Texture, 0, 0), bind(0, 1, vpgpu::BindKind::Texture, 1, 1)};
    Result r = run(dir + "/fixes.frag.spv", f);
    CHECK(r.ok);
    if (r.ok) {
        const std::string& m = r.msl.source;
        CHECK(m.find(".z = vp_in.in0.w;") != std::string::npos);  // OpCompositeInsert: q into .z
        CHECK(m.find("sample_compare(") != std::string::npos && m.find("gradient2d(") != std::string::npos);
        CHECK(m.find("float4 color0 [[color(0)]];") != std::string::npos);  // three packed outputs, one colour
        CHECK(m.find("[[user(loc66)]]") != std::string::npos);              // gl_ClipDistance[2] read
        CHECK(m.find("[[early_fragment_tests]] fragment") != std::string::npos);
        CHECK(r.iface.colorOutputs == 1);
    }
    if (failures || std::getenv("VP_PRINT")) std::fprintf(stderr, "%s\n", r.msl.source.c_str());

    spirv::SpirvLowerOptions v;
    v.entry = "main";
    v.stage = vpir::Stage::Vertex;
    v.vertexBuffers = {{0, 32, false, 3}, {1, 1, false, 4}};
    v.vertexAttributes = {{0, 0, 0, vpgpu::Format::RGBA32Float}, {1, 0, 16, vpgpu::Format::RGBA32Float},
                          {2, 1, 0, vpgpu::Format::R8Uint}};
    Result vr = run(dir + "/fixes.vert.spv", v);
    CHECK(vr.ok);
    if (vr.ok) {
        const std::string& m = vr.msl.source;
        CHECK(m.find("float vp_clip [[clip_distance]] [4];") != std::string::npos);
        CHECK(m.find("clip_varying2 [[user(loc66)]]") != std::string::npos);
        CHECK(m.find("& 4294967292u") != std::string::npos);  // 1-byte stride: word of the byte address
    }
    if (failures || std::getenv("VP_PRINT")) std::fprintf(stderr, "%s\n", vr.msl.source.c_str());

    spirv::SpirvLowerOptions c;
    c.entry = "main";
    c.stage = vpir::Stage::Compute;
    c.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0)};
    Result cr = run(dir + "/fixes.comp.spv", c);
    CHECK(cr.ok);
    if (cr.ok) {
        const std::string& m = cr.msl.source;
        CHECK(m.find("log2") == std::string::npos && m.find("exp2") == std::string::npos);  // exact frexp/ldexp
        CHECK(m.find("== ctz(simd_or(1u << ") != std::string::npos && m.find("simd_all(") != std::string::npos);  // read-first
    }
    if (failures || std::getenv("VP_PRINT")) std::fprintf(stderr, "%s\n", cr.msl.source.c_str());
}

// A combined image sampler array indexed at run time: an if-chain over the four elements, each
// sampling its own texture/sampler slot. Hand-assembled: gen_dyn_index.py.
void dynIndex(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Fragment;
    spirv::SpirvBinding b = bind(0, 0, vpgpu::BindKind::Texture, 0, 0);
    b.arraySize = 4;
    o.layout = {b};
    Result r = run(dir + "/dyn_index.frag.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    for (int k = 0; k < 4; ++k) {
        const std::string t = "vp_args.t" + std::to_string(k) + ".sample(";
        CHECK(r.msl.source.find(t) != std::string::npos);
    }
    if (failures || std::getenv("VP_PRINT")) std::fprintf(stderr, "%s\n", r.msl.source.c_str());
}

// A 100-element combined image sampler array (more than the 64 an if-chain allows) indexed at run
// time: a balanced tree of `index < mid` tests, every element sampled from its own slot. Evaluated on
// the CPU for the last, first and a middle element. Hand-assembled: gen_dyn_index.py <out> 100.
void dynIndexBig(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Fragment;
    spirv::SpirvBinding b = bind(0, 0, vpgpu::BindKind::Texture, 0, 0);
    b.arraySize = 100;
    o.layout = {b};
    Result r = run(dir + "/dyn_index_big.frag.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    for (int k : {0, 1, 49, 50, 98, 99}) {
        const std::string t = "vp_args.t" + std::to_string(k) + ".sample(";
        CHECK(r.msl.source.find(t) != std::string::npos);
    }
    // An index past the end takes the last element: the tree's right-most leaf.
    CHECK(r.msl.source.find("vp_args.t100") == std::string::npos);
    // 99 branch points, not 99 nested levels: the deepest `if` nests log2(100) <= 7 levels.
    size_t depth = 0, maxDepth = 0;
    for (size_t i = 0; i < r.msl.source.size(); ++i) {
        if (r.msl.source[i] == '{') maxDepth = std::max(maxDepth, ++depth);
        else if (r.msl.source[i] == '}' && depth) --depth;
    }
    CHECK(maxDepth <= 14);
    if (failures || std::getenv("VP_PRINT")) std::fprintf(stderr, "%s\n", r.msl.source.c_str());
}

// A bool varying (vertex output, flat fragment input) travels as an unsigned integer and is
// converted back at every read. Hand-assembled: gen_bool_iface.py.
void boolIface(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Vertex;
    Result v = run(dir + "/bool_iface.vert.spv", o);
    CHECK(v.ok);
    if (v.ok) {
        int n = 0;
        for (const auto& out : v.module.outputs)
            if (out.builtin == vpir::Builtin::None) {
                CHECK(out.type == vpir::kU32);
                ++n;
            }
        CHECK(n == 1);
    }
    o.stage = vpir::Stage::Fragment;
    Result f = run(dir + "/bool_iface.frag.spv", o);
    CHECK(f.ok);
    if (f.ok) {
        int n = 0;
        for (const auto& in : f.module.inputs)
            if (in.builtin == vpir::Builtin::None) {
                CHECK(in.type == vpir::kU32 && in.interp == vpir::Interp::Flat);
                ++n;
            }
        CHECK(n == 1);
        CHECK(f.msl.source.find("bool") == std::string::npos || f.msl.source.find("uint") != std::string::npos);
    }
    if (failures || std::getenv("VP_PRINT")) {
        if (v.ok) std::fprintf(stderr, "%s\n", v.msl.source.c_str());
        if (f.ok) std::fprintf(stderr, "%s\n", f.msl.source.c_str());
    }
}

// A storage-buffer descriptor array indexed at run time: the store is unrolled over the two
// buffers. Hand-assembled: gen_dyn_buffer.py.
void dynBuffer(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    spirv::SpirvBinding b = bind(0, 0, vpgpu::BindKind::Buffer, 0);
    b.arraySize = 2;
    o.layout = {b};
    Result r = run(dir + "/dyn_buffer.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    CHECK(r.msl.source.find("vp_args.b0") != std::string::npos);
    CHECK(r.msl.source.find("vp_args.b1") != std::string::npos);
    if (failures || std::getenv("VP_PRINT")) std::fprintf(stderr, "%s\n", r.msl.source.c_str());
}

// Cube-array storage image texel coordinates split into (face, array) = (z % 6, z / 6); bool
// subgroup broadcast travels as an integer; subgroupAllEqual = all lanes equal the first lane's
// value. Hand-assembled: gen_subgroup_cube.py.
void subgroupCube(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::StorageTexture, 0), bind(0, 1, vpgpu::BindKind::Buffer, 0)};
    Result r = run(dir + "/subgroup_cube.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    const std::string& src = r.msl.source;
    // Storage cubes are texture2d_array (the driver binds a 2D-array view): slice = layer * 6 + face.
    CHECK(src.find("texture2d_array<uint") != std::string::npos && src.find("texturecube") == std::string::npos);
    CHECK(src.find(" % ") != std::string::npos && src.find("* 6u + ") != std::string::npos);
    CHECK(src.find("== ctz(simd_or(1u << ") != std::string::npos);  // read-first: simd_or of the first lane's bits
    CHECK(src.find("simd_all(") != std::string::npos);
    CHECK(src.find("simd_shuffle(true") == std::string::npos);
    if (failures || std::getenv("VP_PRINT")) std::fprintf(stderr, "%s\n", src.c_str());
}

// Relative shuffles (Xor / Up / Down) are a simd_shuffle of the computed lane, the quad broadcast a quad_shuffle.
// Hand-assembled: gen_subgroup_shuffle.py.
void subgroupShuffle(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0)};
    Result r = run(dir + "/subgroup_shuffle.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    const std::string& src = r.msl.source;
    size_t shuffles = 0;
    for (size_t at = src.find("simd_shuffle("); at != std::string::npos; at = src.find("simd_shuffle(", at + 1)) ++shuffles;
    CHECK(shuffles == 3);
    CHECK(src.find("quad_shuffle(") != std::string::npos);
    CHECK(src.find(" ^ ") != std::string::npos);
    if (failures || std::getenv("VP_PRINT")) std::fprintf(stderr, "%s\n", src.c_str());
}

// Arithmetic and clustered subgroup operations (vpir subgroup_arith): reductions and the sum / product scans are Metal's
// simd_* functions, the other scans and the clustered reductions a loop over the lanes of the SIMD group; a boolean
// LogicalAnd travels as an integer. Hand-assembled: gen_subgroup_arith.py.
void subgroupArith(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0)};
    Result r = run(dir + "/subgroup_arith.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    const std::string& src = r.msl.source;
    CHECK(src.find("simd_sum(") != std::string::npos);                    // IAdd reduce
    CHECK(src.find("simd_prefix_inclusive_sum(") != std::string::npos);   // FAdd inclusive scan
    CHECK(src.find("simd_xor(") != std::string::npos);                    // BitwiseXor reduce
    CHECK(src.find("simd_and(") != std::string::npos);                    // LogicalAnd reduce (0/1 integers)
    CHECK(src.find("simd_prefix_exclusive_") == std::string::npos);       // SMin exclusive: the loop, not a builtin
    // The active mask is a simd_or of one bit per lane (simd_active_threads_mask() / simd_ballot() are merged across divergent branches).
    CHECK(src.find("simd_or(1u << in_lane)") != std::string::npos && src.find("for (uint sk = 0u; sk < 32u; ++sk)") != std::string::npos);
    CHECK(src.find("simd_active_threads_mask") == std::string::npos && src.find("simd_ballot") == std::string::npos);
    CHECK(src.find("min(") != std::string::npos && src.find("max(") != std::string::npos);
    CHECK(src.find("(sk >> 2u) == (") != std::string::npos);              // clustered by 4
    CHECK(src.find("sk < ") != std::string::npos);                        // exclusive: lanes below
    CHECK(src.find("2147483647") != std::string::npos);                   // SMin identity
    if (failures || std::getenv("VP_PRINT")) std::fprintf(stderr, "%s\n", src.c_str());
}

// robustBufferAccess (EmitOptions::robustBuffers): every storage-buffer store / atomic is guarded by
// the BufferSize of its slot (added as an argument when the shader had none), so an out-of-range
// or unbound (size 0) buffer is never touched. Without the option the MSL has no guard.
void robust(const std::string& dir) {
    auto check = [&](const char* file, std::vector<spirv::SpirvBinding> layout, bool atomics) {
        spirv::SpirvLowerOptions o;
        o.entry = "main";
        o.stage = vpir::Stage::Compute;
        o.layout = layout;
        vpmsl::EmitOptions eo;
        eo.robustBuffers = true;
        Result r = run(dir + "/" + file, o, false, eo);
        CHECK(r.ok);
        if (!r.ok) return;
        const std::string& src = r.msl.source;
        CHECK(src.find("ul <= ((vp_args.bs") != std::string::npos);
        bool hasSize = false;
        for (auto& a : r.msl.arguments) hasSize |= a.kind == vpir::ResourceKind::BufferSize;
        CHECK(hasSize);
        if (atomics) CHECK(src.find(") atomic_fetch_") != std::string::npos || src.find(" = atomic_fetch_") != std::string::npos);
        Result plain = run(dir + "/" + file, o);
        CHECK(plain.ok && plain.msl.source.find("ul <= ((vp_args.bs") == std::string::npos);
        CHECK(src.find("// vp:int16") == std::string::npos);  // no 16-bit integers here
        if (failures || std::getenv("VP_PRINT")) std::fprintf(stderr, "%s\n", src.c_str());
    };
    spirv::SpirvBinding b = bind(0, 0, vpgpu::BindKind::Buffer, 0);
    b.arraySize = 2;
    check("dyn_buffer.comp.spv", {b}, false);
    check("array_length.comp.spv", {bind(0, 0, vpgpu::BindKind::Buffer, 0)}, false);
}

}  // namespace

// Uniform texel buffer of Unknown format: without a view format it is read as 32-bit words and
// reported as R32Uint; with the bound view's format (RGBA16Uint, R8Snorm…) it unpacks that format,
// sub-word texels from their aligned word shifted down. Hand-assembled: gen_texel_fetch.py.
void texelFetch(const std::string& dir) {
    for (vpgpu::Format f : {vpgpu::Format::Undefined, vpgpu::Format::RGBA16Uint, vpgpu::Format::RG8Sint,
                            vpgpu::Format::R16Snorm, vpgpu::Format::RG11B10Float, vpgpu::Format::RGB9E5Float,
                            vpgpu::Format::B5G6R5Unorm}) {
        spirv::SpirvLowerOptions o;
        o.entry = "main";
        o.stage = vpir::Stage::Compute;
        o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0), bind(0, 1, vpgpu::BindKind::Buffer, 1, 0xFFFF, f)};
        Result r = run(dir + "/texel_fetch.comp.spv", o);
        CHECK(r.ok);
        if (!r.ok) return;
        const spirv::SpirvBinding* t = used(r, 0, 1);
        CHECK(t && t->texelFormat == (f == vpgpu::Format::Undefined ? vpgpu::Format::R32Uint : f));
        const bool narrow = f == vpgpu::Format::RG8Sint || f == vpgpu::Format::R16Snorm || f == vpgpu::Format::B5G6R5Unorm;
        // Sub-word texels: a 4-aligned word address and a run-time shift.
        if (narrow) CHECK(r.msl.source.find("4294967292") != std::string::npos || r.msl.source.find("0xfffffffc") != std::string::npos);
        if (f == vpgpu::Format::RGBA16Uint) CHECK(r.msl.source.find("extract_bits") != std::string::npos);
    }
}

// shaderInt16: 16-bit integers wrap, extend and compare by their own width (int16.comp, from
// gen_int16_drawparams.py), checked by running the lowered VPIR in the interpreter.
void int16(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0)};
    Result r = run(dir + "/int16.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    CHECK(r.module.resources.size() == 1);
    CHECK(r.module.int16 && r.msl.source.find("\n// vp:int16\n") != std::string::npos);  // hang diagnostics marker
    vpireval::Invocation inv;
    std::string err;
    CHECK(vpireval::run(r.module, inv, err));
    if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
    const std::vector<uint8_t>& mem = inv.memory[0];
    CHECK(mem.size() >= 40);
    if (mem.size() < 40) return;
    int32_t got[10];
    std::memcpy(got, mem.data(), 40);
    // r[8], r[9]: an int16 spec constant whose literal is 0xFFFB (folded SConvert and a runtime one).
    const int32_t want[10] = {-25536, 40000, -1596, 2500, 1, 1, -2, 32767, -5, -5};
    for (int k = 0; k < 10; ++k) {
        if (got[k] != want[k]) std::fprintf(stderr, "  r[%d] = %d, want %d\n", k, got[k], want[k]);
        CHECK(got[k] == want[k]);
    }
}

// 16-bit storage: int16 / uint16 members 2 bytes apart in a buffer, read from the half of their
// word and stored with a half-word merge that keeps the neighbours (storage16.comp, from
// gen_storage16.py), run in the interpreter.
void storage16(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0)};
    Result r = run(dir + "/storage16.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    vpireval::Invocation inv;
    std::vector<uint8_t> mem(36, 0);
    const int16_t s[4] = {1, -2, 3, -4};
    const uint16_t v[2] = {65535, 7};
    const int16_t o0 = 0x1234;
    const uint32_t tail = 0xDEADBEEF;
    std::memcpy(mem.data(), s, 8);
    std::memcpy(mem.data() + 8, v, 4);
    std::memcpy(mem.data() + 28, &o0, 2);
    std::memcpy(mem.data() + 32, &tail, 4);
    inv.memory[0] = mem;
    std::string err;
    CHECK(vpireval::run(r.module, inv, err));
    if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
    const std::vector<uint8_t>& out = inv.memory[0];
    CHECK(out.size() >= 36);
    if (out.size() < 36) return;
    int32_t got[4];
    std::memcpy(got, out.data() + 12, 16);
    const int32_t want[4] = {-2, -4, 65535, 7};
    for (int k = 0; k < 4; ++k) {
        if (got[k] != want[k]) std::fprintf(stderr, "  r[%d] = %d, want %d\n", k, got[k], want[k]);
        CHECK(got[k] == want[k]);
    }
    int16_t oh[2];
    uint32_t t2 = 0;
    std::memcpy(oh, out.data() + 28, 4);
    std::memcpy(&t2, out.data() + 32, 4);
    CHECK(oh[0] == 0x1234 && oh[1] == 13);  // the low half (another element) kept
    CHECK(t2 == tail);
    CHECK(r.msl.source.find("atomic_fetch_and_explicit") != std::string::npos);
}

// shaderInt8 + 8-bit storage: bytes wrap and sign-extend by their own width and are stored with a
// byte merge that keeps the neighbours (int8.comp, from int8.comp.spvasm), run in the interpreter.
void int8(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0), bind(0, 1, vpgpu::BindKind::Buffer, 1)};
    Result r = run(dir + "/int8.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    vpireval::Invocation inv;
    inv.memory[0] = {200, 0xEE, 0xEE, 0xEE};
    inv.memory[1] = std::vector<uint8_t>(12, 0);
    inv.memory[1][9] = 0xAA;  // neighbour of the stored byte
    std::string err;
    CHECK(vpireval::run(r.module, inv, err));
    if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
    const std::vector<uint8_t>& out = inv.memory[1];
    CHECK(out.size() >= 12);
    if (out.size() < 12) return;
    int32_t got[2];
    std::memcpy(got, out.data(), 8);
    CHECK(got[0] == 144);   // 200 + 200 wraps at 8 bits (uint8)
    CHECK(got[1] == -112);  // int8(-56) * 2, sign-extended
    CHECK(out[8] == 144 && out[9] == 0xAA);
}

// shaderIntegerDotProduct: packed 4x8-bit OpSDot / OpUDot / OpSUDot and the saturating accumulate forms
// (dot.comp.spvasm), evaluated. a = bytes {1, 127, 128, 255}, b = bytes {2, 3, 4, 255}.
void dotProduct(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0), bind(0, 1, vpgpu::BindKind::Buffer, 1)};
    Result r = run(dir + "/dot.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    const uint32_t in[5] = {0xFF807F01u, 0xFF040302u, 0x80000010u, 0xFFFFFFF0u, 100u};
    vpireval::Invocation inv;
    inv.memory[0].resize(20);
    std::memcpy(inv.memory[0].data(), in, 20);
    inv.memory[1].assign(24, 0);
    std::string err;
    CHECK(vpireval::run(r.module, inv, err));
    if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
    uint32_t got[6] = {};
    if (inv.memory[1].size() >= 24) std::memcpy(got, inv.memory[1].data(), 24);
    CHECK(int32_t(got[0]) == -128);        // 2 + 381 - 512 + 1
    CHECK(got[1] == 65920u);               // 2 + 381 + 512 + 65025
    CHECK(int32_t(got[2]) == -384);        // signed a, unsigned b
    CHECK(got[3] == 0x80000000u);          // INT_MIN + 16 - 128 saturates
    CHECK(got[4] == 0xFFFFFFFFu);          // UINT_MAX - 15 + 65920 saturates
    CHECK(int32_t(got[5]) == -284);        // -384 + 100
}

// OpConvertFToS / OpConvertFToU (ftoi.comp.spvasm), evaluated: out-of-range values saturate, NaN gives 0.
// in = {NaN, 3e10, -3e10, -1.5, 200.5}.
void floatToInt(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0), bind(0, 1, vpgpu::BindKind::Buffer, 1)};
    Result r = run(dir + "/ftoi.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    // The evaluator's own conversion already saturates; what the lowering must emit for Metal is explicit.
    const std::string text = vpir::print(r.module);
    CHECK(text.find("isnan") != std::string::npos && text.find("min") != std::string::npos);
    const float in[5] = {std::nanf(""), 3e10f, -3e10f, -1.5f, 200.5f};
    vpireval::Invocation inv;
    inv.memory[0].resize(20);
    std::memcpy(inv.memory[0].data(), in, 20);
    inv.memory[1].assign(60, 0);
    std::string err;
    CHECK(vpireval::run(r.module, inv, err));
    if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
    uint32_t got[15] = {};
    if (inv.memory[1].size() >= 60) std::memcpy(got, inv.memory[1].data(), 60);
    const uint32_t want[15] = {0, 0x7FFFFFFFu, 0x80000000u, uint32_t(-1), 200,  //
                               0, 0xFFFFFFFFu, 0,           0,            200,  //
                               0, 255,         0,           0,            200};
    for (int i = 0; i < 15; ++i) {
        CHECK(got[i] == want[i]);
        if (got[i] != want[i]) std::fprintf(stderr, "  [%d] got %08x want %08x\n", i, got[i], want[i]);
    }
}

// OpConvertFToS / OpConvertFToU of values a clamp bounds (ftoi_bounded.comp.spvasm): converted directly, none of the NaN / overflow
// selects of the general case, same results. in = {3e10, -3e10, 200.25, -1.5}.
void floatToIntBounded(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0), bind(0, 1, vpgpu::BindKind::Buffer, 1)};
    Result r = run(dir + "/ftoi_bounded.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    CHECK(vpir::print(r.module).find("isnan") == std::string::npos);
    const float in[4] = {3e10f, -3e10f, 200.25f, -1.5f};
    vpireval::Invocation inv;
    inv.memory[0].resize(16);
    std::memcpy(inv.memory[0].data(), in, 16);
    inv.memory[1].assign(32, 0);
    std::string err;
    CHECK(vpireval::run(r.module, inv, err));
    uint32_t got[8] = {};
    if (inv.memory[1].size() >= 32) std::memcpy(got, inv.memory[1].data(), 32);
    const uint32_t want[8] = {255, 0, 200, 0, 100, uint32_t(-100), 100, uint32_t(-1)};
    for (int i = 0; i < 8; ++i) {
        CHECK(got[i] == want[i]);
        if (got[i] != want[i]) std::fprintf(stderr, "  [%d] got %08x want %08x\n", i, got[i], want[i]);
    }
}

// 16-bit float GLSL.std.450 functions (f16math.comp.spvasm), evaluated: their constants must be binary16 and their
// bit tricks / approximations run in 32 bits. in = {180, 1, -2, 11}.
void f16Math(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0), bind(0, 1, vpgpu::BindKind::Buffer, 1)};
    Result r = run(dir + "/f16math.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    const float in[4] = {180.f, 1.f, -2.f, 11.f};
    vpireval::Invocation inv;
    inv.memory[0].resize(16);
    std::memcpy(inv.memory[0].data(), in, 16);
    inv.memory[1].assign(24, 0);
    std::string err;
    CHECK(vpireval::run(r.module, inv, err));
    if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
    float got[6] = {};
    if (inv.memory[1].size() >= 24) std::memcpy(got, inv.memory[1].data(), 24);
    const float want[6] = {3.14159265f, 2.71828183f, -1.f, 29937.07f, 0.78539816f, 16.f};
    for (int i = 0; i < 6; ++i) {
        const bool good = std::fabs(got[i] - want[i]) <= 2e-3f * std::fabs(want[i]);  // half precision
        CHECK(good);
        if (!good) std::fprintf(stderr, "  [%d] got %g want %g\n", i, double(got[i]), double(want[i]));
    }
}

// 32-bit ldexp with exponents far outside the float range (ldexp.comp.spvasm), evaluated: the CTS draws them up to
// 2^30, and a large float scaled by a very negative exponent must flush to zero, not stop at a finite power.
void ldexpRange(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0), bind(0, 1, vpgpu::BindKind::Buffer, 1)};
    Result r = run(dir + "/ldexp.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    struct In { float x; int32_t e; };
    const In in[4] = {{3.0e38f, -1000000}, {1.5f, 3}, {std::ldexp(1.0f, 127), -252}, {1.25f, -140}};
    vpireval::Invocation inv;
    inv.memory[0].resize(sizeof in);
    std::memcpy(inv.memory[0].data(), in, sizeof in);
    inv.memory[1].assign(16, 0);
    std::string err;
    CHECK(vpireval::run(r.module, inv, err));
    if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
    float got[4] = {};
    if (inv.memory[1].size() >= 16) std::memcpy(got, inv.memory[1].data(), 16);
    const float want[4] = {0.f, 12.f, std::ldexp(1.0f, -125), std::ldexp(1.25f, -140)};
    for (int i = 0; i < 4; ++i) {
        CHECK(got[i] == want[i]);
        if (got[i] != want[i]) std::fprintf(stderr, "  [%d] got %g want %g\n", i, double(got[i]), double(want[i]));
    }
}

// An Output variable with an initializer (outinit.frag.spvasm: FragDepth = 0.2 never stored to, a lane terminated
// before the end): the output starts at the initializer, so the surviving lanes write it.
void outputInit(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Fragment;
    Result r = run(dir + "/outinit.frag.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    if (std::getenv("VP_PRINT")) std::fputs(r.msl.source.c_str(), stderr);
    CHECK(r.msl.source.find(" = 0.2f;") != std::string::npos);
}

// A variable pointer returned by a function (varptr_call.comp.spvasm: OpSelect of two pointers to an array of arrays of
// structs in one buffer, returned through OpFunctionCall, then indexed): the load reads the chosen member (a at byte 0,
// b at byte 256), evaluated.
void varPtrCall(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0), bind(0, 1, vpgpu::BindKind::Buffer, 1)};
    Result r = run(dir + "/varptr_call.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    float in[128];
    for (int i = 0; i < 64; ++i) in[i] = 4.f * float(i) / 255.f, in[i + 64] = (4.f * float(i) + 3.f) / 255.f;
    for (int pickA = 1; pickA >= 0; --pickA) {
        // The evaluator's memories follow the module's resource order (first use), not the binding numbers.
        const size_t ib = r.module.resources[0].index == 0 ? 0 : 1, ob = 1 - ib;
        vpireval::Invocation inv;
        inv.memory[ib].resize(sizeof in);
        std::memcpy(inv.memory[ib].data(), in, sizeof in);
        const float out[2] = {-1.f, float(pickA)};
        inv.memory[ob].resize(8);
        std::memcpy(inv.memory[ob].data(), out, 8);
        std::string err;
        CHECK(vpireval::run(r.module, inv, err));
        if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
        float got = -1.f;
        if (inv.memory[ob].size() >= 4) std::memcpy(&got, inv.memory[ob].data(), 4);
        const float want = in[(pickA ? 0 : 64) + 42];  // [1][0].y[0].z: byte 128 + 32 + 8 of the chosen member
        CHECK(got == want);
        if (got != want) std::fprintf(stderr, "  pickA=%d got %g want %g\n", pickA, double(got), double(want));
    }
}

// memoryBarrier() / memoryBarrierImage() without the Vulkan memory model (membar_image.comp.spvasm): the image class still
// fences (Metal needs it for storage-image writes to reach the rest of the threadgroup); a shared-only barrier adds none.
void memBarImage(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    Result r = run(dir + "/membar_image.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    int device = 0, group = 0;
    std::function<void(const vpir::Block&)> walk = [&](const vpir::Block& b) {
        for (const vpir::Node& n : b) {
            device += n.kind == vpir::NodeKind::Inst && n.inst.op == vpir::Op::FenceDevice;
            group += n.kind == vpir::NodeKind::Inst && n.inst.op == vpir::Op::FenceThreadgroup;
            walk(n.body);
            walk(n.elseBody);
        }
    };
    for (const vpir::Function& f : r.module.functions) walk(f.body);
    CHECK(device == 1 && group == 1);
}

// OpBitcast between a 32-bit word and a vector of 16-bit integers, both ways (bitcast16.comp.spvasm), evaluated.
void bitcast16(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0), bind(0, 1, vpgpu::BindKind::Buffer, 1)};
    Result r = run(dir + "/bitcast16.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    const uint32_t in = 0x8001FFFEu;
    vpireval::Invocation inv;
    inv.memory[0].resize(4);
    std::memcpy(inv.memory[0].data(), &in, 4);
    inv.memory[1].assign(12, 0);
    std::string err;
    CHECK(vpireval::run(r.module, inv, err));
    if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
    uint32_t got[3] = {};
    if (inv.memory[1].size() >= 12) std::memcpy(got, inv.memory[1].data(), 12);
    CHECK(int32_t(got[0]) == -2 && int32_t(got[1]) == -32767 && got[2] == in);
}

// CTS float16 OpCompositeInsert on a struct{v3f16} read-modify-written through a Uniform buffer (vertex stage), evaluated: the
// output starts as 0xffffffff words and ends as (0, 1, 2) plus the -1.0 pad.
void f16Insert(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Vertex;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0), bind(0, 1, vpgpu::BindKind::Buffer, 1)};
    Result r = run(dir + "/f16_insert.vert.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    const uint16_t in[4] = {0x0000, 0x3C00, 0x4000, 0xBC00};
    vpireval::Invocation inv;
    inv.memory[0].resize(8);
    std::memcpy(inv.memory[0].data(), in, 8);
    inv.memory[1].assign(8, 0xFF);
    std::string err;
    CHECK(vpireval::run(r.module, inv, err));
    if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
    uint16_t got[4] = {};
    if (inv.memory[1].size() >= 8) std::memcpy(got, inv.memory[1].data(), 8);
    CHECK(got[0] == 0x0000 && got[1] == 0x3C00 && got[2] == 0x4000 && got[3] == 0xBC00);
    if (failures || std::getenv("VP_PRINT")) std::fprintf(stderr, "%04x %04x %04x %04x\n%s\n", got[0], got[1], got[2], got[3], r.msl.source.c_str());
}

// gl_Subgroup{Eq,Ge,Gt,Le,Lt}Mask are derived from the lane and the subgroup size (submask.comp.spvasm).
void subgroupMasks(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0)};
    Result r = run(dir + "/submask.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    const std::string& src = r.msl.source;
    CHECK(src.find("thread_index_in_simdgroup") != std::string::npos && src.find("threads_per_simdgroup") != std::string::npos);
    if (failures || std::getenv("VP_PRINT")) std::fprintf(stderr, "%s\n", src.c_str());
}

// The pipeline's VkSampleMask: the fragment shader gets a [[sample_mask]] output holding it (solid.frag declares none); without
// a mask the MSL has none.
void sampleMask(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Fragment;
    Result plain = run(dir + "/solid.frag.spv", o);
    CHECK(plain.ok && plain.msl.source.find("[[sample_mask]]") == std::string::npos);
    o.sampleMask = 0x5u;
    Result r = run(dir + "/solid.frag.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    CHECK(r.msl.source.find("[[sample_mask]]") != std::string::npos);
    CHECK(r.msl.source.find("5u") != std::string::npos || r.msl.source.find("0x5") != std::string::npos);
    if (failures || std::getenv("VP_PRINT")) std::fprintf(stderr, "%s\n", r.msl.source.c_str());
}

// D24 emulation (SpirvLowerOptions::d24Emul): the fragment shader writes its depth rounded to 24 bits UNORM. solid.frag writes none: it
// gets [[depth(any)]] = FragCoord.z + the draw's bias (vp_dbias), rounded; depth_write.frag writes 0.001: the value is rounded, no bias.
// Without the option the MSL is exactly what it was (no depth output, no bias parameter).
// Depth in tile (VPVK_DEPTH_TILE): the writer's extra float output at [[color(6)]] holding the depth it writes (FragDepth, else the
// fragment's own), and the reader's multisampled depth input attachment turned into a per-sample fetch of that colour slot.
void depthTile(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Fragment;
    for (const char* name : {"/solid.frag.spv", "/depth_write.frag.spv"}) {
        Result off = run(dir + name, o);
        Result off2 = run(dir + name, o);
        CHECK(off.ok && off2.ok && off.msl.source == off2.msl.source);
        CHECK(off.msl.source.find("[[color(6)]]") == std::string::npos);
        spirv::SpirvLowerOptions w = o;
        w.depthShadowSlot = 6;
        Result on = run(dir + name, w);
        CHECK(on.ok);
        if (!on.ok) continue;
        CHECK(on.msl.source.find("[[color(6)]]") != std::string::npos);
        CHECK((on.msl.colorOutputs & 0x40) && (on.iface.colorOutputs & 0x40));
        CHECK(on.msl.colorOutputs == (off.msl.colorOutputs | 0x40));
        CHECK(on.msl.source != off.msl.source);
        w.depthShadowSlot = 0;  // collides with the shader's own colour 0
        CHECK(!run(dir + name, w).ok);
        if (failures || std::getenv("VP_PRINT")) std::fprintf(stderr, "---- %s\n%s\n", name, on.msl.source.c_str());
    }
    // The reader: the input attachment is a Texture binding (slot 0) until the fetch rewrite.
    o.layout = {bind(0, 3, vpgpu::BindKind::Texture, 0)};
    Result r = run(dir + "/tile/depth_input.frag.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    CHECK(used(r, 0, 3));
    CHECK(r.msl.source.find("[[color(6)]]") == std::string::npos);
    const auto bad = vpir::toDepthFramebufferFetch(r.module, {{5, 6}});  // another texture slot: nothing to rewrite
    CHECK(bad.empty());
    vpmsl::EmitResult none = vpmsl::emitMsl(r.module, {});
    CHECK(none.errors.empty() && none.shader.source == r.msl.source);
    const auto rej = vpir::toDepthFramebufferFetch(r.module, {{0, 6}});
    for (auto& x : rej) std::fprintf(stderr, "  depth fetch rejected: %s\n", x.reason.c_str());
    CHECK(rej.empty());
    for (const vpir::Resource& res : r.module.resources) CHECK(!(res.kind == vpir::ResourceKind::Texture && res.index == 0));
    CHECK(vpir::verify(r.module).empty());
    vpmsl::EmitResult e = vpmsl::emitMsl(r.module, {});
    CHECK(e.errors.empty());
    if (e.errors.empty()) {
        CHECK(e.shader.source.find("[[color(6)]]") != std::string::npos);
        CHECK(e.shader.source.find("[[sample_id]]") != std::string::npos);  // per-sample shading: the fetch reads this sample
        CHECK(e.shader.source.find("texture2d_ms") == std::string::npos && e.shader.source.find("read(") == std::string::npos);
        if (failures || std::getenv("VP_PRINT")) std::fprintf(stderr, "---- fetch\n%s\n", e.shader.source.c_str());
    }
}

void d24Emul(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Fragment;
    for (const char* name : {"/solid.frag.spv", "/depth_write.frag.spv"}) {
        const bool own = std::string(name) == "/solid.frag.spv";
        Result off = run(dir + name, o);
        o.d24Emul = false;
        Result off2 = run(dir + name, o);
        CHECK(off.ok && off2.ok && off.msl.source == off2.msl.source);
        CHECK(off.msl.source.find("vp_dbias") == std::string::npos && off.msl.source.find("16777215") == std::string::npos);
        CHECK(own == (off.msl.source.find("depth(any)") == std::string::npos));
        o.d24Emul = true;
        Result on = run(dir + name, o);
        o.d24Emul = false;
        CHECK(on.ok);
        if (!on.ok) continue;
        const std::string& s = on.msl.source;
        CHECK(s.find("[[depth(any)]]") != std::string::npos);
        CHECK(s.find("16777215") != std::string::npos && s.find("rint(") != std::string::npos && s.find("floor(") == std::string::npos);
        CHECK(s.find("precise::divide(") != std::string::npos);  // fast math must not turn the / 16777215 into a reciprocal
        CHECK(own == (s.find("constant float3& vp_dbias [[buffer(26)]]") != std::string::npos));
        CHECK(own == (s.find("dfdx(") != std::string::npos));
        CHECK(on.msl.source != off.msl.source);
        if (failures || std::getenv("VP_PRINT")) std::fprintf(stderr, "---- %s\n%s\n", name, s.c_str());
    }
    // Integer reference: round-to-nearest-even of z * (2^24 - 1) in exact (double) arithmetic, over 2^24 - 1.
    auto quantN = [](float z) { return std::nearbyint(double(std::min(std::max(z, 0.0f), 1.0f)) * 16777215.0); };
    auto quant = [&](float z) { return float(quantN(z)) / 16777215.0f; };
    auto bitsOf = [](float f) {
        uint32_t b;
        std::memcpy(&b, &f, 4);
        return b;
    };
    {
        o.d24Emul = true;
        Result r = run(dir + "/depth_write.frag.spv", o);
        o.d24Emul = false;
        CHECK(r.ok);
        if (r.ok) {
            vpireval::Invocation inv;
            std::string err;
            CHECK(vpireval::run(r.module, inv, err));
            bool found = false;
            for (size_t k = 0; k < r.module.outputs.size(); ++k) {
                if (r.module.outputs[k].builtin != vpir::Builtin::FragDepth) continue;
                auto it = inv.outputs.find(uint32_t(k));
                CHECK(it != inv.outputs.end());
                if (it == inv.outputs.end()) continue;
                found = true;
                float got;
                std::memcpy(&got, &it->second.bits[0], 4);
                CHECK(got == quant(0.001f) && got != 0.001f);
            }
            CHECK(found);
        }
    }
    // The module evaluated, not mirrored (solid.frag, own depth: FragCoord.z + bias, rounded): one invocation with the given
    // position.z, depth bias (constant, slope, clamp) and derivatives (the evaluator answers ddx / ddy from the test).
    {
        o.d24Emul = true;
        Result r = run(dir + "/solid.frag.spv", o);
        o.d24Emul = false;
        CHECK(r.ok);
        int depthOut = -1;
        for (size_t k = 0; r.ok && k < r.module.outputs.size(); ++k)
            if (r.module.outputs[k].builtin == vpir::Builtin::FragDepth) depthOut = int(k);
        CHECK(depthOut >= 0);
        auto eval = [&](float pz, float c, float sl, float cl, float dzdx, float dzdy, float& got) {
            vpireval::Invocation inv;
            vpireval::Value pos;
            pos.type = vpir::kF32.withWidth(4);
            pos.bits[0] = bitsOf(3.0f), pos.bits[1] = bitsOf(4.0f), pos.bits[2] = bitsOf(pz), pos.bits[3] = bitsOf(1.0f);
            inv.builtins[vpir::Builtin::Position] = pos;
            vpireval::Value bias;
            bias.type = vpir::kF32.withWidth(3);
            bias.bits[0] = bitsOf(c), bias.bits[1] = bitsOf(sl), bias.bits[2] = bitsOf(cl);
            inv.builtins[vpir::Builtin::DepthBias] = bias;
            inv.derivative = [&](const vpir::Inst&, bool y, const vpireval::Value&, vpireval::Value& out) {
                out.bits[0] = bitsOf(y ? dzdy : dzdx);
                return true;
            };
            std::string err;
            if (!vpireval::run(r.module, inv, err)) return false;
            auto it = inv.outputs.find(uint32_t(depthOut));
            if (it == inv.outputs.end()) return false;
            std::memcpy(&got, &it->second.bits[0], 4);
            return true;
        };
        if (r.ok && depthOut >= 0) {
            // (1) No bias: the rounding alone against the integer reference. z = 0, 1, 0.5 (a tie: to even), 1e-3, the ends, the
            // 2^-24 steps, out-of-range values (saturate), then a grid of the 2^24 steps and a wide pseudo-random sweep.
            unsigned bad = 0, over1 = 0, evalFail = 0, coarse = 0, total = 0;
            auto one = [&](float z, bool exactStrict) {
                float got = 0;
                ++total;
                if (!eval(z, 0, 0, 0, 0, 0, got)) return void(++evalFail);
                over1 += got > 1.0f || got < 0.0f;
                const double n = std::nearbyint(double(got) * 16777215.0);
                if (got != quant(z)) {
                    // (z below 0.5: float32 holds z * (2^24 - 1) in steps of 1/2 or finer, and rounding that again can land one step
                    // off the exact rounding; at or above 0.5 the product is an integer already and the result is exact)
                    ++coarse;
                    bad += exactStrict || std::fabs(n - quantN(z)) > 1.0;
                }
            };
            for (float z : {0.0f, 1.0f, 0.5f, 1e-3f, 0.25f, 0.75f, 0.999999f, std::nextafter(1.0f, 0.0f), std::nextafter(0.0f, 1.0f), 1e-9f,
                            std::ldexp(1.0f, -24), std::ldexp(1.0f, -25), std::ldexp(3.0f, -25), 8388607.5f / 16777215.0f,
                            std::nextafter(0.5f, 1.0f), std::nextafter(0.5f, 0.0f), -0.5f, 1.5f, -1e-30f, 1e30f})
                one(z, true);
            const unsigned stepsBefore = total;
            for (uint32_t n = 0; n < 16777216u; n += 61) {  // z = n / (2^24 - 1): the 24-bit value itself must come back, never above 1
                const float z = float(n) / 16777215.0f;
                float got = 0;
                ++total;
                if (!eval(z, 0, 0, 0, 0, 0, got)) {
                    ++evalFail;
                    continue;
                }
                over1 += got > 1.0f;
                bad += got != z;
            }
            CHECK(total > stepsBefore + 200000);
            uint32_t rng = 12345u;
            auto next = [&] { return rng = rng * 1664525u + 1013904223u; };
            for (int i = 0; i < 200000; ++i) {
                const float m = float(next() >> 8) / 16777216.0f;
                one(std::ldexp(m, -int(next() % 20)), false);
                one(1.0f - m * 0.5f, true);  // [0.5, 1]: exact
            }
            std::printf("  d24 rounding sweep: %u values, %u differ from the exact rounding (by one step, z < 0.5), %u bad, %u over 1\n",
                        total, coarse, bad, over1);
            CHECK(evalFail == 0 && bad == 0 && over1 == 0);
            CHECK(coarse < total / 4);
            // 1.0 gives exactly 1.0 (the old floor(z * K + 0.5) / K gave 1.00000012) and 0.5 gives 8388608 / K.
            float g = 0;
            CHECK(eval(1.0f, 0, 0, 0, 0, 0, g) && g == 1.0f);
            CHECK(eval(0.5f, 0, 0, 0, 0, 0, g) && g == 8388608.0f / 16777215.0f);
            CHECK(eval(0.0f, 0, 0, 0, 0, 0, g) && g == 0.0f);

            // (2) The FragCoord.z + bias path against the spec's formula, in CPU arithmetic: constant * 2^-24 (UNORM24) + slope *
            // max(|dz/dx|, |dz/dy|), limited by the clamp (positive: at most, negative: at least, 0: none), added to z, rounded.
            struct Case { float z, c, sl, cl, dx, dy; };
            const float u = std::ldexp(1.0f, -24);
            const Case cases[] = {{0.3f, 0, 0, 0, 1e-4f, 2e-4f},        // no bias: z alone
                                  {0.5f, 4, 0, 0, 0, 0},                 // constant only
                                  {0.25f, 0, 2.0f, 0, 1e-6f, -3e-6f},    // slope only: max(|dx|, |dy|) = 3e-6
                                  {0.75f, -3, 1.5f, 0, 2e-7f, 1e-7f},    // negative constant, slope
                                  {0.4f, 100, 0, 10 * u, 0, 0},          // positive clamp reached
                                  {0.4f, -100, 0, -10 * u, 0, 0},        // negative clamp reached
                                  {0.4f, -5, 0, 1e-3f, 0, 0},            // positive clamp, negative bias: unchanged
                                  {0.4f, 5, 0, -1e-3f, 0, 0},            // negative clamp, positive bias: unchanged
                                  {0.4f, 7, 3.0f, 1e-2f, -5e-6f, 4e-6f},  // clamp not reached
                                  {0.99999994f, 1000, 0, 0, 0, 0},       // above 1: saturates
                                  {0.0f, -1000, 1.0f, 0, 1e-3f, 1e-3f},  // below 0: saturates
                                  {1e-3f, 2, 0.5f, 0, 1e-6f, 1e-6f}};
            for (const Case& k : cases) {
                float got = 0;
                const bool ok = eval(k.z, k.c, k.sl, k.cl, k.dx, k.dy, got);
                CHECK(ok);
                if (!ok) continue;
                const float slope = std::max(std::fabs(k.dx), std::fabs(k.dy));
                const float t0 = k.c * u;
                const float t1 = k.sl * slope;
                float off = t0 + t1;
                if (k.cl > 0) off = std::min(off, k.cl);
                else if (k.cl < 0) off = std::max(off, k.cl);
                const float zb = k.z + off;
                // (the bias path is under test, not the rounding: the product in float32 as the shader has it, which may sit one step
                // from the exact rounding below 0.5, see above)
                const float sat = std::min(std::max(zb, 0.0f), 1.0f);
                const float prod = sat * 16777215.0f;
                const float want = std::nearbyint(prod) / 16777215.0f;
                CHECK(std::fabs(std::nearbyint(double(got) * 16777215.0) - quantN(zb)) <= 1.0);
                if (got != want) std::fprintf(stderr, "  bias case z=%g c=%g s=%g cl=%g: got %.9g want %.9g\n", k.z, k.c, k.sl, k.cl, got, want);
                CHECK(got == want);
            }
        }
    }
    // D24 emulation vs early fragment tests (inline modules: void main() with EarlyFragmentTests, then with a store into a storage
    // buffer): without resource writes the early tests go and the shader writes the rounded depth; with them it stays as it was
    // (d24EmulSkipped) unless the alpha test or a sample mask drops the early tests anyway (one decision, not two).
    {
        auto module = [](bool early, bool store, bool fragDepth = false) {
            std::vector<uint32_t> w = {0x07230203u, 0x00010300u, 0u, 20u, 0u};
            auto ins = [&](uint32_t op, std::initializer_list<uint32_t> ops) {
                w.push_back((uint32_t(ops.size() + 1) << 16) | op);
                w.insert(w.end(), ops);
            };
            ins(17, {1});                                    // OpCapability Shader
            ins(14, {0, 1});                                 // OpMemoryModel Logical GLSL450
            if (fragDepth) ins(15, {4, 1, 0x6e69616du, 0, 14});  // OpEntryPoint Fragment %1 "main" %14
            else ins(15, {4, 1, 0x6e69616du, 0});            // OpEntryPoint Fragment %1 "main"
            ins(16, {1, 7});                                 // OpExecutionMode %1 OriginUpperLeft
            if (early) ins(16, {1, 9});                      // OpExecutionMode %1 EarlyFragmentTests
            if (fragDepth) ins(16, {1, 12});                 // OpExecutionMode %1 DepthReplacing
            if (fragDepth) ins(71, {14, 11, 22});            // OpDecorate %14 BuiltIn FragDepth
            ins(71, {5, 2});                                 // OpDecorate %5 Block
            ins(72, {5, 0, 35, 0});                          // OpMemberDecorate %5 0 Offset 0
            ins(71, {7, 34, 0});                             // OpDecorate %7 DescriptorSet 0
            ins(71, {7, 33, 0});                             // OpDecorate %7 Binding 0
            ins(19, {2});                                    // %2 = OpTypeVoid
            ins(33, {3, 2});                                 // %3 = OpTypeFunction %2
            ins(21, {4, 32, 0});                             // %4 = OpTypeInt 32 0
            ins(30, {5, 4});                                 // %5 = OpTypeStruct %4
            ins(32, {6, 12, 5});                             // %6 = OpTypePointer StorageBuffer %5
            ins(59, {6, 7, 12});                             // %7 = OpVariable %6 StorageBuffer
            ins(32, {8, 12, 4});                             // %8 = OpTypePointer StorageBuffer %4
            ins(43, {4, 9, 0});                              // %9 = OpConstant %4 0
            ins(43, {4, 10, 1});                             // %10 = OpConstant %4 1
            if (fragDepth) {
                ins(22, {15, 32});                           // %15 = OpTypeFloat 32
                ins(32, {16, 3, 15});                        // %16 = OpTypePointer Output %15
                ins(59, {16, 14, 3});                        // %14 = OpVariable %16 Output
                ins(43, {15, 17, 0x3a83126fu});              // %17 = OpConstant %15 0.001
            }
            ins(54, {2, 1, 0, 3});                           // %1 = OpFunction %2 None %3
            ins(248, {11});                                  // %11 = OpLabel
            if (store) {
                ins(65, {8, 12, 7, 9});                      // %12 = OpAccessChain %8 %7 %9
                ins(62, {12, 10});                           // OpStore %12 %10
            }
            if (fragDepth) ins(62, {14, 17});                // OpStore %14 %17
            ins(253, {});                                    // OpReturn
            ins(56, {});                                     // OpFunctionEnd
            return w;
        };
        spirv::SpirvLowerOptions e;
        e.entry = "main";
        e.stage = vpir::Stage::Fragment;
        e.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0)};
        struct Variant { bool early, store, alpha; uint32_t mask; bool skipped; };
        const Variant variants[] = {{false, false, false, ~0u, false},   // no early tests: emulated
                                    {true, false, false, ~0u, false},    // early tests, no writes: they go, emulated
                                    {true, true, false, ~0u, true},      // early tests and a store: as it was
                                    {false, true, false, ~0u, false},    // a store, no early tests: emulated
                                    {true, true, true, ~0u, false},      // alpha test drops the early tests: emulated
                                    {true, true, false, 0x5u, false}};   // a sample mask drops them too: emulated
        for (const Variant& v : variants) {
            const std::vector<uint32_t> w = module(v.early, v.store);
            std::string error;
            std::shared_ptr<spirv::SpirvModule> m(spirv::parse(w.data(), w.size(), error), spirv::release);
            CHECK(m != nullptr);
            if (!m) continue;
            e.alphaTest = v.alpha, e.sampleMask = v.mask, e.d24Emul = true;
            CHECK(spirv::d24EmulSkipped(*m, e) == v.skipped);
            Result on = runWords(w, e);
            e.d24Emul = false;
            Result off = runWords(w, e);
            CHECK(on.ok && off.ok);
            if (!on.ok || !off.ok) continue;
            const bool early = on.msl.source.find("[[early_fragment_tests]]") != std::string::npos;
            const bool depth = on.msl.source.find("[[depth(any)]]") != std::string::npos;
            CHECK(!spirv::d24EmulSkipped(*m, e));  // (off: never)
            CHECK(depth == !v.skipped);
            CHECK(early == (v.skipped && v.early));
            CHECK(v.skipped == (on.msl.source == off.msl.source));
            if (failures || std::getenv("VP_PRINT")) std::fprintf(stderr, "---- early=%d store=%d alpha=%d mask=%x\n%s\n", v.early, v.store, v.alpha, v.mask, on.msl.source.c_str());
        }
        // A FragDepth write: with early tests Vulkan ignores it, so the depth is the invocation's FragCoord.z + the draw's bias (the
        // `own` path: vp_dbias, dfdx); without them the shader's own value is rounded (no bias).
        for (bool early : {true, false}) {
            const std::vector<uint32_t> w = module(early, false, true);
            e.alphaTest = false, e.sampleMask = ~0u, e.d24Emul = true;
            Result on = runWords(w, e);
            e.d24Emul = false;
            CHECK(on.ok);
            if (!on.ok) continue;
            const std::string& s = on.msl.source;
            CHECK(s.find("[[depth(any)]]") != std::string::npos);
            CHECK(s.find("[[early_fragment_tests]]") == std::string::npos);
            CHECK(early == (s.find("constant float3& vp_dbias [[buffer(26)]]") != std::string::npos));
            CHECK(early == (s.find("dfdx(") != std::string::npos));
            if (failures || std::getenv("VP_PRINT")) std::fprintf(stderr, "---- fragdepth early=%d\n%s\n", early, s.c_str());
        }
    }
    // Two coplanar triangles at z ~ 1e-3 whose depths differ by float32 rounding only (the plane evaluated in two orders, 4 ulps
    // apart at most): as float32 they differ at most pixels (>= comparisons flicker), rounded to 24 bits they tie, except where the
    // pair straddles a rounding boundary (probability |dz| / 2^-24 ~ 0.2 %); away from a boundary they tie exactly.
    {
        unsigned raw = 0, tied = 0, total = 0, straddle = 0, farFromBoundaryDiff = 0;
        for (int y = 0; y < 64; ++y)
            for (int x = 0; x < 64; ++x) {
                const float a = 1.3e-8f, b = -2.1e-8f, c = 1.0e-3f;
                const float z1 = c + a * float(x) + b * float(y);
                float z2 = (c + b * float(y)) + a * float(x);
                for (int k = 0; k < (x * 7 + y * 3) % 5; ++k) z2 = std::nextafter(z2, 1.0f);  // up to 4 ulps
                const float scaled = z1 * 16777215.0f + 0.5f;
                const float frac = scaled - std::floor(scaled);
                const bool nearBoundary = frac < 0.01f || frac > 0.99f;
                ++total, raw += z1 != z2, tied += quant(z1) == quant(z2), straddle += nearBoundary;
                farFromBoundaryDiff += !nearBoundary && quant(z1) != quant(z2);
            }
        CHECK(raw > total / 4);                  // float32 compares flicker
        CHECK(tied >= total - total / 50);       // 24-bit rounding ties (>= 98 %)
        CHECK(farFromBoundaryDiff == 0);
        (void)straddle;
    }
    // The flag is ignored by a vertex stage.
    spirv::SpirvLowerOptions v;
    v.entry = "main";
    v.stage = vpir::Stage::Vertex;
    v.d24Emul = true;
    Result vr = run(dir + "/pos.vert.spv", v);
    v.d24Emul = false;
    Result vr0 = run(dir + "/pos.vert.spv", v);
    CHECK(vr.ok && vr0.ok && vr.msl.source == vr0.msl.source);
}

// shaderZeroInitializeWorkgroupMemory: a Workgroup variable with an OpConstantNull initializer is cleared by a
// strided loop over the threadgroup array and a barrier before the body (zeroinit.comp.spvasm).
void zeroInit(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0)};
    Result r = run(dir + "/zeroinit.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    CHECK(r.module.sharedBytes == 160);
    const std::string text = vpir::print(r.module);
    CHECK(text.find("loop") != std::string::npos && text.find("barrier") != std::string::npos);
    vpireval::Invocation inv;
    inv.builtins[vpir::Builtin::LocalInvocationIndex] = {vpir::kU32, {2}};
    inv.memory[0].assign(16, 0xEE);
    std::string err;
    CHECK(vpireval::run(r.module, inv, err));
    if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
    uint32_t got[4] = {};
    std::memcpy(got, inv.memory[0].data(), 16);
    CHECK(got[2] == 7u);  // tile[i * 5] read after the clear
}

// VariablePointersStorageBuffer: OpSelect / OpPhi between pointers into two buffers, a store through
// the selected pointer and OpPtrAccessChain with a non-zero element (varptr.comp), run both ways.
void varPtr(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0), bind(0, 1, vpgpu::BindKind::Buffer, 1),
                bind(0, 2, vpgpu::BindKind::Buffer, 2)};
    Result r = run(dir + "/varptr.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    for (int cond = 1; cond >= 0; --cond) {
        vpireval::Invocation inv;
        const uint32_t a[4] = {uint32_t(cond), 10, 11, 12}, b[4] = {2, 20, 21, 22};
        inv.memory[0].resize(16), inv.memory[1].resize(16), inv.memory[2].assign(16, 0);
        std::memcpy(inv.memory[0].data(), a, 16);
        std::memcpy(inv.memory[1].data(), b, 16);
        std::string err;
        CHECK(vpireval::run(r.module, inv, err));
        if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
        uint32_t ga[4], gb[4], go[4];
        std::memcpy(ga, inv.memory[0].data(), 16);
        std::memcpy(gb, inv.memory[1].data(), 16);
        std::memcpy(go, inv.memory[2].data(), 16);
        CHECK(go[0] == (cond ? 10u : 21u));  // *sel before the store
        CHECK(go[1] == 12);                  // &A[1] + 2 elements
        CHECK(go[2] == 77);                  // the phi picks the same pointer, which now holds 77
        CHECK(ga[1] == (cond ? 77u : 10u) && gb[2] == (cond ? 21u : 77u));  // only the chosen buffer written
    }
}

// OpPtrAccessChain with a negative constant element (ptr_negative.comp): the step back is modular in 32 bits, as the dynamic
// index is; it used to be refused as "byte offset overflows 32 bits".
void ptrNegative(const std::string& dir, const char* file = "ptr_negative.comp.spv") {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0), bind(0, 1, vpgpu::BindKind::Buffer, 1)};
    Result r = run(dir + "/" + file, o);
    CHECK(r.ok);
    if (!r.ok) return;
    vpireval::Invocation inv;
    const uint32_t a[4] = {10, 11, 12, 13};
    inv.memory[0].resize(16), inv.memory[1].assign(12, 0);
    std::memcpy(inv.memory[0].data(), a, 16);
    std::string err;
    CHECK(vpireval::run(r.module, inv, err));
    if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
    uint32_t go[3];
    std::memcpy(go, inv.memory[1].data(), 12);
    CHECK(go[0] == 12u);  // (&A[3]) - 1
    CHECK(go[1] == 10u);  // (&A[3]) - 3
    CHECK(go[2] == 10u);  // ((&A[0]) - 1) + 1: below zero for a moment
}

// The same chains with UNSIGNED constants (0xFFFFFFFF, 0xFFFFFFFD): the element of an access chain is signed whatever its declared
// type, so the result is the same as ptr_negative's. It used to be refused as "byte offset overflows 32 bits".
void ptrUnsignedNeg(const std::string& dir) { ptrNegative(dir, "ptr_unsigned_neg.comp.spv"); }

// 16-bit elements, constant and run-time (ptr_narrow.comp): unsigned ones are zero-extended, signed ones sign-extended, both ways alike.
void ptrNarrow(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0), bind(0, 1, vpgpu::BindKind::Buffer, 1), bind(0, 2, vpgpu::BindKind::Buffer, 2)};
    Result r = run(dir + "/ptr_narrow.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    vpireval::Invocation inv;
    std::vector<uint32_t> a(32768 + 16);
    for (size_t i = 0; i < a.size(); ++i) a[i] = uint32_t(1000 + i);
    inv.memory[0].resize(a.size() * 4), inv.memory[1].assign(16, 0), inv.memory[2].resize(8);
    std::memcpy(inv.memory[0].data(), a.data(), a.size() * 4);
    const uint32_t pv[2] = {0x8000u, 0xFFFFFFFEu};
    std::memcpy(inv.memory[2].data(), pv, 8);
    std::string err;
    CHECK(vpireval::run(r.module, inv, err));
    if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
    uint32_t go[4];
    std::memcpy(go, inv.memory[1].data(), 16);
    CHECK(go[0] == 1000u + 10 + 32768);  // u16 0x8000, constant: forward
    CHECK(go[1] == 1000u + 8);           // i16 -2, constant: back
    CHECK(go[2] == go[0]);               // the same at run time
    CHECK(go[3] == go[1]);
}

// The UNSIGNED element 2^31 is the signed -2^31: a step of -2^33 bytes, 0 modulo 2^32 (as the main branch computes it).
void ptrUnsignedBig(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0), bind(0, 1, vpgpu::BindKind::Buffer, 1)};
    Result r = run(dir + "/ptr_unsigned_big.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    vpireval::Invocation inv;
    const uint32_t a[4] = {10, 11, 12, 13};
    inv.memory[0].resize(16), inv.memory[1].assign(12, 0);
    std::memcpy(inv.memory[0].data(), a, 16);
    std::string err;
    CHECK(vpireval::run(r.module, inv, err));
    if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
    uint32_t go[3];
    std::memcpy(go, inv.memory[1].data(), 12);
    CHECK(go[0] == 13u);  // &A[3] + 2^31 elements = &A[3]
    CHECK(go[1] == 10u);  // &A[3] - 3
    CHECK(go[2] == 11u);  // (&A[0] + 2^31 elements) + 1 = &A[1]
}

// VK_KHR_buffer_device_address: a PhysicalStorageBuffer pointer read from a buffer is a {low, high}
// address; loads / stores through it (and through chains into it) use the device target; pointers
// convert to and from uvec2 (bda.comp). The interpreter maps address high word h to memory[0x10000 + h].
void bda(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0), bind(0, 1, vpgpu::BindKind::Buffer, 1)};
    Result r = run(dir + "/bda.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    CHECK(r.msl.source.find("reinterpret_cast<device uchar*>(as_type<ulong>(") != std::string::npos);
    vpireval::Invocation inv;
    const uint32_t ptr[2] = {16, 1};  // region 1, offset 16
    inv.memory[0].resize(8);
    std::memcpy(inv.memory[0].data(), ptr, 8);
    inv.memory[1].assign(16, 0);
    inv.memory[0x10001].assign(32, 0);
    const uint32_t inner[2] = {111, 222};
    std::memcpy(inv.memory[0x10001].data() + 16, inner, 8);
    std::string err;
    CHECK(vpireval::run(r.module, inv, err));
    if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
    uint32_t out[4], now[2];
    std::memcpy(out, inv.memory[1].data(), 16);
    std::memcpy(now, inv.memory[0x10001].data() + 16, 8);
    CHECK(out[0] == 222 && out[1] == 16 && out[2] == 1 && out[3] == 111);
    CHECK(now[0] == 111 && now[1] == 1222);
}

// Vulkan memory model: release / acquire semantics, OpMemoryBarrier and MakePointerAvailable /
// MakePointerVisible become sequentially consistent fences (membar.comp: fences at device scope for
// the store with MakePointerAvailable, the release atomic store, the acquire load, the device barrier,
// the visible load; one at threadgroup scope for the workgroup barrier).
void memBar(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0)};
    Result r = run(dir + "/membar.comp.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    int device = 0, group = 0;
    std::function<void(const vpir::Block&)> walk = [&](const vpir::Block& b) {
        for (const vpir::Node& n : b) {
            device += n.kind == vpir::NodeKind::Inst && n.inst.op == vpir::Op::FenceDevice;
            group += n.kind == vpir::NodeKind::Inst && n.inst.op == vpir::Op::FenceThreadgroup;
            walk(n.body);
            walk(n.elseBody);
        }
    };
    for (const vpir::Function& f : r.module.functions) walk(f.body);
    CHECK(device == 5 && group == 1);
    CHECK(r.msl.source.find("atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture, "
                            "memory_order_seq_cst, thread_scope_device);") != std::string::npos);
    CHECK(r.msl.source.find("memory_order_seq_cst, thread_scope_threadgroup);") != std::string::npos);
}

// Reading texel buffers of every 8/16-bit format at run-time texel indices (fetch_{f,i,u}.comp: `out[0] = bits(texelFetch(t, gid.x))`): sign extension of sint, snorm scaling and clamping, unorm, half.
// The expected values follow the Vulkan format conversion rules.
void texelFormats(const std::string& dir) {
    using F = vpgpu::Format;
    struct Case {
        F f;
        unsigned comps, bytes;
        char kind;  // u unorm, s snorm, U uint, S sint, h half
    };
    const Case cases[] = {
        {F::R8Unorm, 1, 1, 'u'},   {F::RG8Unorm, 2, 2, 'u'},   {F::RGBA8Unorm, 4, 4, 'u'},
        {F::R8Snorm, 1, 1, 's'},   {F::RG8Snorm, 2, 2, 's'},   {F::RGBA8Snorm, 4, 4, 's'},
        {F::R8Uint, 1, 1, 'U'},    {F::RG8Uint, 2, 2, 'U'},    {F::RGBA8Uint, 4, 4, 'U'},
        {F::R8Sint, 1, 1, 'S'},    {F::RG8Sint, 2, 2, 'S'},    {F::RGBA8Sint, 4, 4, 'S'},
        {F::R16Unorm, 1, 2, 'u'},  {F::RG16Unorm, 2, 4, 'u'},  {F::RGBA16Unorm, 4, 8, 'u'},
        {F::R16Snorm, 1, 2, 's'},  {F::RG16Snorm, 2, 4, 's'},  {F::RGBA16Snorm, 4, 8, 's'},
        {F::R16Uint, 1, 2, 'U'},   {F::RG16Uint, 2, 4, 'U'},   {F::RGBA16Uint, 4, 8, 'U'},
        {F::R16Sint, 1, 2, 'S'},   {F::RG16Sint, 2, 4, 'S'},   {F::RGBA16Sint, 4, 8, 'S'},
    };
    for (const Case& c : cases) {
        spirv::SpirvLowerOptions o;
        o.entry = "main";
        o.stage = vpir::Stage::Compute;
        o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0), bind(0, 1, vpgpu::BindKind::Buffer, 1, 0xFFFF, c.f)};
        const char* shader = c.kind == 'U' ? "/fetch_u.comp.spv" : c.kind == 'S' ? "/fetch_i.comp.spv" : "/fetch_f.comp.spv";
        Result r = run(dir + shader, o);
        CHECK(r.ok);
        if (!r.ok) return;
        const unsigned bits = c.bytes / c.comps * 8;
        // Texels 0..3 with extreme and ordinary component values (a deterministic byte pattern).
        std::vector<uint8_t> data(c.bytes * 4);
        for (size_t k = 0; k < data.size(); ++k) data[k] = uint8_t((k * 37 + 0x80 * (k % 3 == 0) + 0x7F * (k % 5 == 0)) & 0xFF);
        data[0] = 0x80, data[c.bytes - 1] = 0x7F;  // most negative low component, positive high
        for (unsigned i = 0; i < 4; ++i) {
            // (the interpreter keys memory by position in module.resources)
            size_t outPos = 0, inPos = 0;
            for (size_t k = 0; k < r.module.resources.size(); ++k)
                (r.module.resources[k].index == 0 ? outPos : inPos) = k;
            vpireval::Invocation inv;
            inv.memory[uint32_t(inPos)] = data;
            inv.memory[uint32_t(outPos)].assign(16, 0);
            vpireval::Value gid;
            gid.type = vpir::Type(vpir::Scalar::U32, 3);
            gid.bits[0] = i;
            inv.builtins[vpir::Builtin::GlobalInvocationId] = gid;
            std::string err;
            CHECK(vpireval::run(r.module, inv, err));
            if (!err.empty()) std::fprintf(stderr, "  eval %s: %s\n", vpgpu::formatInfo(c.f).name, err.c_str());
            uint32_t got[4];
            std::memcpy(got, inv.memory[uint32_t(outPos)].data(), 16);
            for (unsigned k = 0; k < 4; ++k) {
                uint64_t raw = 0;
                if (k < c.comps) std::memcpy(&raw, &data[i * c.bytes + k * (bits / 8)], bits / 8);
                const int64_t sraw = int64_t(raw << (64 - bits)) >> (64 - bits);
                uint32_t want = 0;
                if (k >= c.comps) {
                    const bool floating = c.kind == 'u' || c.kind == 's';
                    want = k == 3 ? (floating ? 0x3F800000u : 1u) : 0u;
                } else if (c.kind == 'U') {
                    want = uint32_t(raw);
                } else if (c.kind == 'S') {
                    want = uint32_t(int32_t(sraw));
                } else {
                    const float fv = c.kind == 'u' ? float(raw) / float((1ull << bits) - 1)
                                                   : std::max(float(sraw) / float((1ull << (bits - 1)) - 1), -1.0f);
                    std::memcpy(&want, &fv, 4);
                }
                bool same = got[k] == want;
                if (!same && (c.kind == 'u' || c.kind == 's') && k < c.comps) {  // within 1 ulp
                    same = std::abs(int64_t(got[k]) - int64_t(want)) <= 1 && (got[k] >> 31) == (want >> 31);
                }
                if (!same)
                    std::fprintf(stderr, "  %s(%c) texel %u comp %u: got %08x want %08x\n", vpgpu::formatInfo(c.f).name, c.kind, i, k, got[k], want);
                CHECK(same);
            }
        }
    }
}

// Storage texel buffer stores (store_{f,i,u}.comp: `imageStore(t, gid.x, bits)`) of every 8/16-bit
// format: packing, rounding and two's complement, and sub-word merges that keep the neighbouring texels.
void texelStores(const std::string& dir) {
    using F = vpgpu::Format;
    struct Case {
        F f;
        unsigned comps, bytes;
        char kind;  // u unorm, s snorm, U uint, S sint
    };
    const Case cases[] = {
        {F::R8Unorm, 1, 1, 'u'},   {F::RG8Unorm, 2, 2, 'u'},   {F::RGBA8Unorm, 4, 4, 'u'},
        {F::R8Snorm, 1, 1, 's'},   {F::RG8Snorm, 2, 2, 's'},   {F::RGBA8Snorm, 4, 4, 's'},
        {F::R8Uint, 1, 1, 'U'},    {F::RG8Uint, 2, 2, 'U'},    {F::RGBA8Uint, 4, 4, 'U'},
        {F::R8Sint, 1, 1, 'S'},    {F::RG8Sint, 2, 2, 'S'},    {F::RGBA8Sint, 4, 4, 'S'},
        {F::R16Unorm, 1, 2, 'u'},  {F::RG16Unorm, 2, 4, 'u'},  {F::RGBA16Unorm, 4, 8, 'u'},
        {F::R16Snorm, 1, 2, 's'},  {F::RG16Snorm, 2, 4, 's'},  {F::RGBA16Snorm, 4, 8, 's'},
        {F::R16Uint, 1, 2, 'U'},   {F::RG16Uint, 2, 4, 'U'},   {F::RGBA16Uint, 4, 8, 'U'},
        {F::R16Sint, 1, 2, 'S'},   {F::RG16Sint, 2, 4, 'S'},   {F::RGBA16Sint, 4, 8, 'S'},
    };
    for (const Case& c : cases) {
        spirv::SpirvLowerOptions o;
        o.entry = "main";
        o.stage = vpir::Stage::Compute;
        o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0), bind(0, 1, vpgpu::BindKind::Buffer, 1, 0xFFFF, c.f)};
        const char* shader = c.kind == 'U' ? "/store_u.comp.spv" : c.kind == 'S' ? "/store_i.comp.spv" : "/store_f.comp.spv";
        Result r = run(dir + shader, o);
        CHECK(r.ok);
        if (!r.ok) return;
        const unsigned bits = c.bytes / c.comps * 8;
        const int64_t maxU = int64_t((1ull << bits) - 1), maxS = int64_t((1ull << (bits - 1)) - 1);
        // Raw component values to store (in range), and the matching shader inputs.
        int64_t raw[4];
        switch (c.kind) {
            case 'u': raw[0] = maxU / 15, raw[1] = maxU, raw[2] = 0, raw[3] = maxU / 2 - 1; break;
            case 's': raw[0] = -maxS / 3, raw[1] = maxS, raw[2] = -maxS, raw[3] = 5; break;
            case 'U': raw[0] = 1, raw[1] = maxU, raw[2] = 0, raw[3] = 0x33; break;
            default: raw[0] = -5, raw[1] = maxS, raw[2] = -maxS - 1, raw[3] = 7; break;
        }
        uint32_t in[4];
        for (unsigned k = 0; k < 4; ++k) {
            float fv;
            if (c.kind == 'u') fv = float(double(raw[k]) / double(maxU));
            else if (c.kind == 's') fv = float(double(raw[k]) / double(maxS));
            else {
                in[k] = uint32_t(int32_t(raw[k]));
                continue;
            }
            std::memcpy(&in[k], &fv, 4);
        }
        size_t outPos = 0, inPos = 0;  // (the interpreter keys memory by position in module.resources)
        for (size_t k = 0; k < r.module.resources.size(); ++k) (r.module.resources[k].index == 0 ? inPos : outPos) = k;
        std::vector<uint8_t> mem(c.bytes * 4, 0xA5);
        for (unsigned i : {1u, 2u}) {
            vpireval::Invocation inv;
            inv.memory[uint32_t(inPos)].resize(16);
            std::memcpy(inv.memory[uint32_t(inPos)].data(), in, 16);
            inv.memory[uint32_t(outPos)] = mem;
            vpireval::Value gid;
            gid.type = vpir::Type(vpir::Scalar::U32, 3);
            gid.bits[0] = i;
            inv.builtins[vpir::Builtin::GlobalInvocationId] = gid;
            std::string err;
            CHECK(vpireval::run(r.module, inv, err));
            if (!err.empty()) std::fprintf(stderr, "  eval %s: %s\n", vpgpu::formatInfo(c.f).name, err.c_str());
            mem = inv.memory[uint32_t(outPos)];
        }
        std::vector<uint8_t> want(c.bytes * 4, 0xA5);
        for (unsigned i : {1u, 2u})
            for (unsigned k = 0; k < c.comps; ++k)
                for (unsigned b = 0; b < bits / 8; ++b) want[i * c.bytes + k * (bits / 8) + b] = uint8_t((uint64_t(raw[k]) >> (8 * b)) & 0xFF);
        CHECK(mem.size() >= want.size());
        for (size_t k = 0; k < want.size() && k < mem.size(); ++k) {
            if (mem[k] != want[k])
                std::fprintf(stderr, "  %s(%c) byte %zu: got %02x want %02x\n", vpgpu::formatInfo(c.f).name, c.kind, k, mem[k], want[k]);
            CHECK(mem[k] == want[k]);
        }
    }
}

// VK_KHR_shader_draw_parameters: gl_BaseVertex / gl_BaseInstance are Metal builtins, gl_DrawID a
// per-draw constant at [[buffer(29)]]; the base instance of a multiview draw is divided back.
void drawParams(const std::string& dir) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Vertex;
    Result r = run(dir + "/draw_params.vert.spv", o);
    CHECK(r.ok);
    if (!r.ok) return;
    bool bv = false, bi = false, di = false;
    for (auto& in : r.module.inputs) {
        bv |= in.builtin == vpir::Builtin::BaseVertex;
        bi |= in.builtin == vpir::Builtin::BaseInstance;
        di |= in.builtin == vpir::Builtin::DrawIndex;
    }
    CHECK(bv && bi && di);
    CHECK(r.msl.source.find("[[base_vertex]]") != std::string::npos);
    CHECK(r.msl.source.find("[[base_instance]]") != std::string::npos);
    CHECK(r.msl.source.find(std::string(vpmsl::kDrawParamsDecl) + " [[buffer(29)]]") != std::string::npos);
    CHECK(r.msl.source.find("vp_draw.drawIndex") != std::string::npos);
    // gl_BaseVertex: Metal's only for indexed draws (metalBase), else the firstVertex the backend sets.
    CHECK(r.msl.source.find("vp_draw.metalBase != 0u ? in_base_vertex : vp_draw.baseVertex") != std::string::npos);
    // Run it: base vertex 7, base instance 6 drawn as 3 for two views, draw 2.
    o.viewCount = 2;
    Result m = run(dir + "/draw_params.vert.spv", o);
    CHECK(m.ok);
    if (!m.ok) return;
    vpireval::Invocation inv;
    inv.builtins[vpir::Builtin::BaseVertex] = {vpir::kU32, {7}};
    inv.builtins[vpir::Builtin::BaseInstance] = {vpir::kU32, {6}};
    inv.builtins[vpir::Builtin::DrawIndex] = {vpir::kU32, {2}};
    inv.builtins[vpir::Builtin::InstanceId] = {vpir::kU32, {6}};
    std::string err;
    CHECK(vpireval::run(m.module, inv, err));
    if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
    bool found = false;
    for (size_t k = 0; k < m.module.outputs.size(); ++k) {
        const auto& out = m.module.outputs[k];
        if (out.builtin != vpir::Builtin::None || out.location != 0) continue;
        auto it = inv.outputs.find(uint32_t(k));
        CHECK(it != inv.outputs.end());
        if (it == inv.outputs.end()) continue;
        found = true;
        CHECK(it->second.bits[0] == 7 && it->second.bits[1] == 3 && it->second.bits[2] == 2);
    }
    CHECK(found);
}

// OpDemoteToHelperInvocation + OpIsHelperInvocationEXT: the invocation keeps running (discard_fragment
// in MSL 2.3+ is a demote: derivatives stay valid, writes stop) and the query reads the helper state.
static void helper(const std::string& dir) {
  spirv::SpirvLowerOptions o;
  o.entry = "main";
  o.stage = vpir::Stage::Fragment;
  Result r = run(dir + "/helper.frag.spv", o);
  CHECK(r.ok);
  if (!r.ok) return;
  bool in = false;
  for (auto& v : r.module.inputs) in |= v.builtin == vpir::Builtin::HelperInvocation;
  CHECK(in);
  CHECK(r.msl.source.find("discard_fragment()") != std::string::npos);
  CHECK(r.msl.source.find("simd_is_helper_thread()") != std::string::npos);
}

// Translation correctness probes (hand-assembled, *.spvasm): an Invariant position is [[position, invariant]]
// next to the [[clip_distance]] array, a texel-fetch offset moves the read, OpKill keeps the lane as a helper
// outside loops (the texture read after clip() keeps its quad derivatives) and leaves the shader inside one,
// a storage image only written is access::write (DXC never decorates NonReadable).
static void correctness(const std::string& dir) {
    {
        spirv::SpirvLowerOptions o;
        o.entry = "main", o.stage = vpir::Stage::Vertex;
        Result r = run(dir + "/clip_invariant.vert.spv", o);
        CHECK(r.ok);
        CHECK(r.module.invariantPosition);
        CHECK(r.msl.source.find("[[position, invariant]]") != std::string::npos);
        CHECK(r.msl.source.find("float vp_clip [[clip_distance]] [1];") != std::string::npos);
        CHECK(r.msl.source.find("vp_out.vp_clip[0] = ") != std::string::npos);
    }
    {  // pos.vert has no Invariant: plain [[position]]
        spirv::SpirvLowerOptions o;
        o.entry = "main", o.stage = vpir::Stage::Vertex;
        Result r = run(dir + "/pos.vert.spv", o);
        CHECK(r.ok);
        CHECK(!r.module.invariantPosition);
        CHECK(r.msl.source.find("invariant") == std::string::npos);
        // SpirvLowerOptions::invariantPos (VPVK_INVARIANT_POS): the same shader with the position invariant anyway.
        o.invariantPos = true;
        Result ri = run(dir + "/pos.vert.spv", o);
        CHECK(ri.ok);
        CHECK(ri.module.invariantPosition);
        CHECK(ri.msl.source.find("[[position, invariant]]") != std::string::npos);
    }
    {  // only vertex stages: a fragment shader (FragCoord is a Position input) is untouched
        spirv::SpirvLowerOptions o;
        o.entry = "main", o.stage = vpir::Stage::Fragment;
        o.invariantPos = true;
        Result r = run(dir + "/outinit.frag.spv", o);
        CHECK(r.ok);
        CHECK(!r.module.invariantPosition);
        CHECK(r.msl.source.find("invariant") == std::string::npos);
    }
    spirv::SpirvLowerOptions f;
    f.entry = "main", f.stage = vpir::Stage::Fragment;
    f.layout = {bind(0, 1, vpgpu::BindKind::Texture, 0), bind(0, 2, vpgpu::BindKind::Sampler, 0)};
    {
        Result r = run(dir + "/fetch_offset_kill.frag.spv", f);
        CHECK(r.ok);
        const std::string& s = r.msl.source;
        CHECK(s.find("+ 4294967295u") != std::string::npos);  // ConstOffset (-1, +1) applied
        CHECK(s.find("+ 1u") != std::string::npos);
        const size_t d = s.find("discard_fragment();");
        CHECK(d != std::string::npos && s.find("return", d) > s.find(".sample(", d));  // no return before the sample
    }
    {
        Result r = run(dir + "/loop_kill.frag.spv", f);
        CHECK(r.ok);
        const std::string& s = r.msl.source;
        const size_t d = s.find("discard_fragment();");
        CHECK(d != std::string::npos && s.find("return vp_out;", d) < s.find("continue;", d));  // inside the loop: leaves
        f.killExit = true;  // a kill inside a loop already leaves: no group check is added
        Result k = run(dir + "/loop_kill.frag.spv", f);
        CHECK(k.ok && k.msl.source.find("simd_all(") == std::string::npos);
        f.killExit = false;
    }
    {  // killExit: the kill outside loops still keeps the lane as a helper, and once the branch has reconverged the
       // shader leaves when every lane of the SIMD-group is killed or a helper (before the sample that follows).
        f.killExit = true;
        Result r = run(dir + "/fetch_offset_kill.frag.spv", f);
        CHECK(r.ok);
        const std::string& s = r.msl.source;
        const size_t d = s.find("discard_fragment();"), g = s.find("simd_all(");
        CHECK(d != std::string::npos && g != std::string::npos && g > d && g < s.find(".sample(", d));
        CHECK(s.find("simd_is_helper_thread()") != std::string::npos);
        CHECK(s.find("return vp_out;", g) < s.find(".sample(", g));
        f.killExit = false;
        Result off = run(dir + "/fetch_offset_kill.frag.spv", f);
        CHECK(off.ok && off.msl.source.find("simd_all(") == std::string::npos);
    }
    {  // killExit: two kills in a row (a chain of kill-only branches) share one group check, after the last
        f.killExit = true;
        Result r = run(dir + "/two_kill.frag.spv", f);
        CHECK(r.ok);
        const std::string& s = r.msl.source;
        size_t discards = 0, checks = 0;
        for (size_t at = s.find("discard_fragment();"); at != std::string::npos; at = s.find("discard_fragment();", at + 1)) ++discards;
        for (size_t at = s.find("simd_all("); at != std::string::npos; at = s.find("simd_all(", at + 1)) ++checks;
        CHECK(discards == 2 && checks == 1);
        CHECK(s.find("simd_all(") > s.rfind("discard_fragment();"));
        f.killExit = false;
    }
    {  // a loop that always leaves in its first trip has no guard counter; a real loop keeps it
        Result once = run(dir + "/loop_once.frag.spv", f);
        CHECK(once.ok && once.msl.source.find("while (true)") != std::string::npos);
        CHECK(once.ok && once.msl.source.find("vp_guard") == std::string::npos);
        Result real = run(dir + "/loop_kill.frag.spv", f);
        CHECK(real.ok && real.msl.source.find("vp_guard0") != std::string::npos);
    }
    {  // row-major matrices (MatrixStride 16): one vector load per memory row, columns picked out of them
        spirv::SpirvLowerOptions c;
        c.entry = "main", c.stage = vpir::Stage::Compute;
        c.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0), bind(0, 1, vpgpu::BindKind::Buffer, 1)};
        Result r = run(dir + "/rowmajor.comp.spv", c);
        CHECK(r.ok);
        if (r.ok) {
            const std::string& s = r.msl.source;
            size_t loads = 0;
            for (size_t at = s.find("*(const device"); at != std::string::npos; at = s.find("*(const device", at + 1)) ++loads;
            CHECK(loads == 8);  // 4 rows of mat4 + 4 rows of mat3x4, not 28 scalar loads
            CHECK(s.find("packed_float3") != std::string::npos);
            // Same values as the per-component loads: column c, component r of a row-major matrix is at r * 16 + c * 4.
            size_t inPos = 0, outPos = 0;
            for (size_t k = 0; k < r.module.resources.size(); ++k) (r.module.resources[k].index == 0 ? inPos : outPos) = k;
            float mem[32];
            for (int k = 0; k < 32; ++k) mem[k] = float(k) * 1.5f - 7.f;
            vpireval::Invocation inv;
            inv.memory[uint32_t(inPos)].assign(reinterpret_cast<uint8_t*>(mem), reinterpret_cast<uint8_t*>(mem) + sizeof mem);
            inv.memory[uint32_t(outPos)].assign(48, 0);
            std::string err;
            CHECK(vpireval::run(r.module, inv, err));
            if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
            float got[12];
            std::memcpy(got, inv.memory[uint32_t(outPos)].data(), 48);
            for (int k = 0; k < 4; ++k) {
                CHECK(got[k] == mem[k * 4 + 1]);               // a[1]
                CHECK(got[4 + k] == mem[k * 4 + 3]);           // a[3]
                CHECK(got[8 + k] == mem[16 + k * 4 + 2]);      // b[2]
            }
        }
    }
    {
        spirv::SpirvLowerOptions c;
        c.entry = "main", c.stage = vpir::Stage::Compute;
        c.layout = {bind(0, 5, vpgpu::BindKind::StorageTexture, 0)};
        Result r = run(dir + "/storage_write.comp.spv", c);
        CHECK(r.ok);
        CHECK(r.msl.source.find("access::write>") != std::string::npos);
    }
}

// shaderFloat16: arithmetic stays in half (VPIR F16 / MSL half), conversions at the interface.
static void float16(const std::string& dir) {
  spirv::SpirvLowerOptions o;
  o.entry = "main";
  o.stage = vpir::Stage::Fragment;
  Result r = run(dir + "/f16.frag.spv", o);
  CHECK(r.ok);
  if (!r.ok) return;
  if (std::getenv("VP_DUMP")) std::fputs(r.msl.source.c_str(), stderr);
  CHECK(r.msl.source.find("half4") != std::string::npos);
  CHECK(r.msl.source.find("fma(") != std::string::npos);
  CHECK(r.msl.source.find("sqrt(") != std::string::npos);
}

// samplerYcbcrConversion: the constants of ycbcr.h (known values of the three models and both ranges),
// the reference conversion, and the lowering of a combined image sampler with a conversion: planes
// sampled from their own textures (slots), chroma coordinate moved onto the chroma samples
// (cosited), range expansion + matrix, checked against ycbcrConvert() by evaluating the VPIR.
static bool near(float a, float b, float eps = 2e-5f) { return std::fabs(a - b) <= eps; }
static float fbits(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
static uint32_t ubits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }

static void ycbcrMath() {
  using namespace spirv;
  const YcbcrCoeffs a = ycbcrCoeffs(YcbcrModel::Ycbcr709, YcbcrRange::Full);
  CHECK(a.matrix && near(a.rCr, 1.5748f, 1e-4f) && near(a.bCb, 1.8556f, 1e-4f));
  CHECK(near(a.gCb, -0.187324f, 1e-4f) && near(a.gCr, -0.468124f, 1e-4f));
  CHECK(a.yScale == 1.0f && a.yBias == 0.0f && a.cScale == 1.0f && near(a.cBias, -128.0f / 255.0f));
  const YcbcrCoeffs b = ycbcrCoeffs(YcbcrModel::Ycbcr601, YcbcrRange::Narrow);
  CHECK(near(b.rCr, 1.402f, 1e-4f) && near(b.bCb, 1.772f, 1e-4f) && near(b.gCb, -0.344136f, 1e-4f) && near(b.gCr, -0.714136f, 1e-4f));
  CHECK(near(b.yScale, 255.0f / 219.0f) && near(b.yBias, -16.0f / 219.0f) && near(b.cScale, 255.0f / 224.0f) && near(b.cBias, -128.0f / 224.0f));
  const YcbcrCoeffs c = ycbcrCoeffs(YcbcrModel::Ycbcr2020, YcbcrRange::Full);
  CHECK(near(c.rCr, 1.4746f, 1e-4f) && near(c.bCb, 1.8814f, 1e-4f) && near(c.gCb, -0.164553f, 1e-4f) && near(c.gCr, -0.571353f, 1e-4f));
  CHECK(!ycbcrCoeffs(YcbcrModel::YcbcrIdentity, YcbcrRange::Full).matrix && !ycbcrCoeffs(YcbcrModel::RgbIdentity, YcbcrRange::Full).matrix);
  // Narrow range 8-bit: black (16, 128, 128), white (235, 128, 128), saturated red of BT.709 (63, 102, 240).
  YcbcrBinding y;
  y.model = YcbcrModel::Ycbcr709, y.range = YcbcrRange::Narrow;
  float o[4];
  const float blk[4] = {128 / 255.f, 16 / 255.f, 128 / 255.f, 1}, wht[4] = {128 / 255.f, 235 / 255.f, 128 / 255.f, 1};
  ycbcrConvert(y, blk, o);
  CHECK(near(o[0], 0) && near(o[1], 0) && near(o[2], 0) && o[3] == 1);
  ycbcrConvert(y, wht, o);
  CHECK(near(o[0], 1) && near(o[1], 1) && near(o[2], 1));
  const float red[4] = {240 / 255.f, 63 / 255.f, 102 / 255.f, 1};
  ycbcrConvert(y, red, o);
  CHECK(o[0] > 0.99f && o[1] < 0.01f && o[2] < 0.01f);  // clamped to [0, 1]
  // Full range, 601: mid grey stays grey; the swizzle (B, G, R, A) of the components swaps R and B.
  y.model = YcbcrModel::Ycbcr601, y.range = YcbcrRange::Full;
  const float grey[4] = {128 / 255.f, 0.5f, 128 / 255.f, 1};
  ycbcrConvert(y, grey, o);
  CHECK(near(o[0], 0.5f) && near(o[1], 0.5f) && near(o[2], 0.5f));
  y.swizzle[0] = 1, y.swizzle[3] = 2;  // R = zero, A = one
  ycbcrConvert(y, grey, o);
  CHECK(near(o[0], 0.5f - 0.5f * 0.f, 0.6f) && o[3] == 1);
  // Shift of the chroma coordinate: a quarter of a chroma texel for cosited subsampled axes only.
  CHECK(ycbcrChromaShift(true, true) == 0.25f && ycbcrChromaShift(true, false) == 0 && ycbcrChromaShift(false, true) == 0);
}

static void ycbcr(const std::string& dir) {
  ycbcrMath();
  struct Case {
    spirv::YcbcrBinding y;
    float u, v;
    float luma, c1[4], c2;  // luma value, plane 1 texel, plane 2 value
  };
  std::vector<Case> cases;
  {
    Case k;  // 2 planes, BT.709 narrow, cosited on both axes
    k.y.planes = 2, k.y.model = spirv::YcbcrModel::Ycbcr709, k.y.range = spirv::YcbcrRange::Narrow;
    k.y.cositedX = k.y.cositedY = k.y.subX = k.y.subY = true;
    k.u = 0.375f, k.v = 0.625f, k.luma = 120 / 255.f, k.c1[0] = 90 / 255.f, k.c1[1] = 200 / 255.f, k.c2 = 0;
    cases.push_back(k);
  }
  {
    Case k;  // 3 planes, BT.601 full, midpoint (no shift), 4:2:2 (x only)
    k.y.planes = 3, k.y.model = spirv::YcbcrModel::Ycbcr601, k.y.range = spirv::YcbcrRange::Full;
    k.y.subX = true;
    k.u = 0.5f, k.v = 0.25f, k.luma = 0.6f, k.c1[0] = 0.4f, k.c2 = 0.7f;
    cases.push_back(k);
  }
  {
    Case k;  // 3 planes, BT.2020 narrow, swizzle (R, G, B, A) -> (B, G, R, one)
    k.y.planes = 3, k.y.model = spirv::YcbcrModel::Ycbcr2020, k.y.range = spirv::YcbcrRange::Narrow;
    k.y.subX = k.y.subY = true, k.y.cositedY = true;
    k.y.swizzle[0] = 5, k.y.swizzle[2] = 3, k.y.swizzle[3] = 2;
    k.u = 0.8f, k.v = 0.1f, k.luma = 0.5f, k.c1[0] = 0.45f, k.c2 = 0.55f;
    cases.push_back(k);
  }
  {
    Case k;  // identity models
    k.y.planes = 2, k.y.model = spirv::YcbcrModel::YcbcrIdentity, k.y.range = spirv::YcbcrRange::Full;
    k.u = 0.2f, k.v = 0.3f, k.luma = 0.3f, k.c1[0] = 0.6f, k.c1[1] = 0.8f, k.c2 = 0;
    cases.push_back(k);
    k.y.model = spirv::YcbcrModel::RgbIdentity, k.y.range = spirv::YcbcrRange::Narrow;
    cases.push_back(k);
  }
  for (Case& k : cases) {
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Fragment;
    spirv::SpirvBinding b = bind(0, 0, vpgpu::BindKind::Texture, 0, 0);
    b.ycbcr = k.y;
    b.ycbcr.planeSlot[0] = 1, b.ycbcr.planeSlot[1] = 2;
    o.layout = {b};
    Result r = run(dir + "/ycbcr.frag.spv", o);
    CHECK(r.ok);
    if (!r.ok) continue;
    // The planes are Texture resources at their slots.
    int tex[3] = {-1, -1, -1};
    for (size_t i = 0; i < r.module.resources.size(); ++i) {
      const vpir::Resource& res = r.module.resources[i];
      if (res.kind == vpir::ResourceKind::Texture && res.index < 3) tex[res.index] = int(i);
    }
    CHECK(tex[0] >= 0 && tex[1] >= 0 && (k.y.planes == 3) == (tex[2] >= 0));
    vpireval::Invocation inv;
    vpireval::Value uv;
    uv.type = vpir::kF32.withWidth(2);
    uv.bits[0] = ubits(k.u), uv.bits[1] = ubits(k.v);
    inv.locations[0] = uv;
    std::map<int, std::pair<float, float>> seen;  // resource -> coordinate it was sampled at
    inv.image = [&](const vpir::Inst& in, const std::vector<vpireval::Value>& a, vpireval::Value& out) {
      out.type = in.type;
      if (in.op == vpir::Op::ImageSize) {  // the chroma planes are 8 x 4 texels
        out.bits[0] = 8, out.bits[1] = 4;
        return true;
      }
      const uint32_t slot = r.module.resources[in.sym].index;
      seen[int(in.sym)] = {fbits(a[0].bits[0]), fbits(a[0].bits[1])};
      const float v = slot == 0 ? k.luma : slot == 1 ? k.c1[0] : k.c2;
      out.bits[0] = ubits(slot == 1 && k.y.planes == 2 ? k.c1[0] : v);
      out.bits[1] = ubits(slot == 1 ? k.c1[1] : 0.f);
      out.bits[2] = 0, out.bits[3] = ubits(1.f);
      return true;
    };
    std::string err;
    CHECK(vpireval::run(r.module, inv, err));
    if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
    // Reference.
    const float cr = k.y.planes == 3 ? k.c2 : k.c1[1];
    const float raw[4] = {cr, k.luma, k.c1[0], 1.f};
    float want[4];
    spirv::ycbcrConvert(k.y, raw, want);
    auto it = inv.outputs.find(0);
    CHECK(it != inv.outputs.end());
    if (it != inv.outputs.end())
      for (int c = 0; c < 4; ++c) {
        CHECK(near(fbits(it->second.bits[c]), want[c], 5e-5f));
        if (!near(fbits(it->second.bits[c]), want[c], 5e-5f))
          std::fprintf(stderr, "  component %d: got %f want %f\n", c, fbits(it->second.bits[c]), want[c]);
      }
    // Plane 0 is sampled at the coordinate itself, the chroma planes 1/4 chroma texel further (cosited).
    CHECK(seen[tex[0]].first == k.u && seen[tex[0]].second == k.v);
    const float sx = spirv::ycbcrChromaShift(k.y.subX, k.y.cositedX) / 8.f, sy = spirv::ycbcrChromaShift(k.y.subY, k.y.cositedY) / 4.f;
    CHECK(near(seen[tex[1]].first, k.u + sx, 1e-6f) && near(seen[tex[1]].second, k.v + sy, 1e-6f));
    if (k.y.planes == 3) CHECK(near(seen[tex[2]].first, k.u + sx, 1e-6f));
    CHECK(used(r, 0, 0) != nullptr);
    CHECK(r.msl.source.find("texture2d<float") != std::string::npos);
  }
  // Without a conversion the same shader samples plane 0 only and reports one texture.
  spirv::SpirvLowerOptions o;
  o.entry = "main";
  o.stage = vpir::Stage::Fragment;
  o.layout = {bind(0, 0, vpgpu::BindKind::Texture, 0, 0)};
  Result r = run(dir + "/ycbcr.frag.spv", o);
  CHECK(r.ok);
  if (r.ok) {
    int n = 0;
    for (const auto& res : r.module.resources) n += res.kind == vpir::ResourceKind::Texture;
    CHECK(n == 1);
  }
}

// Structural checks of the reader: modules a compiler never writes are refused with a message, and a minimal valid one
// passes. Each case is the header of a compute module (ids: 1 main, 2 void, 3 void(), bound 16) plus what the lambda adds.
void structure() {
    using Words = std::vector<uint32_t>;
    auto module = [](const std::function<void(Words&)>& rest) {
        Words w = {0x07230203u, 0x00010300u, 0, 16, 0};
        auto ins = [&](uint32_t op, std::initializer_list<uint32_t> ops) {
            w.push_back(uint32_t(ops.size() + 1) << 16 | op);
            w.insert(w.end(), ops.begin(), ops.end());
        };
        ins(17, {1});                    // OpCapability Shader
        ins(14, {0, 1});                 // OpMemoryModel Logical GLSL450
        ins(15, {5, 1, 0x6e69616du, 0}); // OpEntryPoint GLCompute %1 "main"
        ins(16, {1, 17, 1, 1, 1});       // OpExecutionMode %1 LocalSize 1 1 1
        ins(19, {2});                    // OpTypeVoid %2
        ins(33, {3, 2});                 // OpTypeFunction %3 %2
        rest(w);
        return w;
    };
    auto ins = [](Words& w, uint32_t op, std::initializer_list<uint32_t> ops) {
        w.push_back(uint32_t(ops.size() + 1) << 16 | op);
        w.insert(w.end(), ops.begin(), ops.end());
    };
    auto fn = [&](Words& w, uint32_t id, std::initializer_list<uint32_t> body) {  // void id() { body; return }
        ins(w, 54, {2, id, 0, 3});
        ins(w, 248, {id + 7});
        for (uint32_t x : body) w.push_back(x);
        ins(w, 253, {});
        ins(w, 56, {});
    };
    auto verdict = [&](const char* name, const Words& w, const char* expect) {
        std::string error;
        spirv::SpirvModule* m = spirv::parse(w.data(), w.size(), error);
        if (m) spirv::release(m);
        const bool ok = expect[0] == 0 ? m != nullptr : (!m && error.find(expect) != std::string::npos);
        if (!ok) std::fprintf(stderr, "  %s: expected \"%s\", got %s \"%s\"\n", name, expect, m ? "a module" : "an error", error.c_str());
        CHECK(ok);
    };
    verdict("valid", module([&](Words& w) { fn(w, 1, {}); }), "");
    verdict("branch to nothing", module([&](Words& w) {
                ins(w, 54, {2, 1, 0, 3});
                ins(w, 248, {4});
                ins(w, 249, {9});  // OpBranch %9
                ins(w, 56, {});
            }), "not one of its labels");
    verdict("recursion", module([&](Words& w) {
                ins(w, 54, {2, 1, 0, 3});
                ins(w, 248, {4});
                ins(w, 57, {2, 5, 1});  // %5 = OpFunctionCall %2 %1
                ins(w, 253, {});
                ins(w, 56, {});
            }), "recursion");
    verdict("mutual recursion", module([&](Words& w) {
                ins(w, 54, {2, 1, 0, 3});
                ins(w, 248, {4});
                ins(w, 57, {2, 5, 6});
                ins(w, 253, {});
                ins(w, 56, {});
                ins(w, 54, {2, 6, 0, 3});
                ins(w, 248, {7});
                ins(w, 57, {2, 8, 1});
                ins(w, 253, {});
                ins(w, 56, {});
            }), "recursion");
    verdict("call with an argument too many", module([&](Words& w) {
                fn(w, 6, {});
                ins(w, 54, {2, 1, 0, 3});
                ins(w, 248, {4});
                ins(w, 57, {2, 5, 6, 4});
                ins(w, 253, {});
                ins(w, 56, {});
            }), "arguments");
    verdict("call of a type", module([&](Words& w) {
                ins(w, 54, {2, 1, 0, 3});
                ins(w, 248, {4});
                ins(w, 57, {2, 5, 3});
                ins(w, 253, {});
                ins(w, 56, {});
            }), "not a function");
    verdict("id defined twice", module([&](Words& w) {
                ins(w, 22, {4, 32});       // %4 = OpTypeFloat 32
                ins(w, 43, {4, 4, 0});     // %4 = OpConstant %4 0
                fn(w, 1, {});
            }), "defined twice");
    verdict("id past the bound", module([&](Words& w) {
                ins(w, 22, {40, 32});      // %40 = OpTypeFloat 32, bound 16
                fn(w, 1, {});
            }), "id bound");
    verdict("entry point is not a function", module([&](Words& w) { ins(w, 22, {1, 32}); }), "not a function");
    verdict("pointer to nothing", module([&](Words& w) {
                ins(w, 32, {5, 7, 9});     // %5 = OpTypePointer Private %9, never declared
                fn(w, 1, {});
            }), "no declared pointee");
    verdict("merge block that is not a label", module([&](Words& w) {
                ins(w, 54, {2, 1, 0, 3});
                ins(w, 248, {4});
                ins(w, 247, {9, 0});       // OpSelectionMerge %9
                ins(w, 249, {4});
                ins(w, 56, {});
            }), "merge block");
}

// A vec4 member at byte 0xFFFFFFF4: its last component sits at byte 2^32. The offsets used to be summed in 32 bits and
// wrapped (a store to byte 0); the lowering now refuses the module.
void offsetOverflow(const std::string& dir) {
    const std::vector<uint32_t> words = readWords(dir + "/offset_overflow.comp.spv");
    std::string error;
    spirv::SpirvModule* m = spirv::parse(words.data(), words.size(), error);
    CHECK(m != nullptr);
    if (!m) return;
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0)};
    vpir::Module mod;
    spirv::SpirvInterface iface;
    const bool ok = spirv::lower(*m, o, mod, iface, error);
    spirv::release(m);
    CHECK(!ok);
    CHECK(error.find("overflows 32 bits") != std::string::npos);
    if (error.find("overflows 32 bits") == std::string::npos) std::fprintf(stderr, "  lower: %s\n", error.c_str());
}

// A compute module whose body is an else-if chain `levels` deep (each else holds the next if): `levels` + 1 nested control-flow
// regions, the shape of a long if / else if ladder. Hand-assembled; ids: 1 main, 2 void, 3 fn, 4 uint, 5 bool, 6 block,
// 7 ptr block, 8 buffer, 9 ptr uint, 10 uint 0, 11 .. the constants, labels and values.
std::vector<uint32_t> elseIfChain(uint32_t levels) {
    std::vector<uint32_t> w = {0x07230203u, 0x00010300u, 0, 0, 0};
    auto ins = [&](uint32_t op, std::initializer_list<uint32_t> ops) {
        w.push_back(uint32_t(ops.size() + 1) << 16 | op);
        w.insert(w.end(), ops.begin(), ops.end());
    };
    uint32_t next = 11;
    ins(17, {1});                     // OpCapability Shader
    ins(14, {0, 1});                  // OpMemoryModel Logical GLSL450
    ins(15, {5, 1, 0x6e69616du, 0});  // OpEntryPoint GLCompute %1 "main"
    ins(16, {1, 17, 1, 1, 1});        // OpExecutionMode LocalSize 1 1 1
    ins(71, {6, 2});                  // Block
    ins(72, {6, 0, 35, 0});           // member 0 Offset 0
    ins(71, {8, 34, 0});              // DescriptorSet 0
    ins(71, {8, 33, 0});              // Binding 0
    ins(19, {2});
    ins(33, {3, 2});
    ins(21, {4, 32, 0});
    ins(20, {5});
    ins(30, {6, 4});
    ins(32, {7, 12, 6});
    ins(59, {7, 8, 12});
    ins(32, {9, 12, 4});
    ins(43, {4, 10, 0});
    std::vector<uint32_t> k(levels), h(levels), t(levels), m(levels);
    for (uint32_t i = 0; i < levels; ++i) k[i] = next++, ins(43, {4, k[i], i + 1});
    for (uint32_t i = 0; i < levels; ++i) h[i] = next++, t[i] = next++, m[i] = next++;
    const uint32_t entry = next++, p = next++, v = next++, last = next++;
    ins(54, {2, 1, 0, 3});
    ins(248, {entry});
    ins(65, {9, p, 8, 10});  // OpAccessChain
    ins(61, {4, v, p});      // OpLoad
    ins(249, {h[0]});
    for (uint32_t i = 0; i < levels; ++i) {
        ins(248, {h[i]});
        const uint32_t c = next++;
        ins(172, {5, c, v, k[i]});  // OpUGreaterThan
        ins(247, {m[i], 0});
        ins(250, {c, t[i], i + 1 < levels ? h[i + 1] : last});
        ins(248, {t[i]});
        ins(62, {p, k[i]});
        ins(249, {m[i]});
    }
    ins(248, {last});
    ins(62, {p, k[0]});
    ins(249, {m[levels - 1]});
    for (int i = int(levels) - 1; i >= 0; --i) {
        ins(248, {m[i]});
        if (i) ins(249, {m[i - 1]});
        else ins(253, {});
    }
    ins(56, {});
    w[3] = next;
    return w;
}

struct StackJob {
    std::vector<uint32_t> words;
    bool parsed = false, lowered = false, skipped = false;
    std::string error;
};
void* stackJob(void* p) {
    StackJob& j = *static_cast<StackJob*>(p);
    spirv::SpirvModule* m = spirv::parse(j.words.data(), j.words.size(), j.error);
    if (!m) return nullptr;
    j.parsed = true;
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    o.layout = {bind(0, 0, vpgpu::BindKind::Buffer, 0)};
    vpir::Module mod;
    spirv::SpirvInterface iface;
    j.lowered = spirv::lower(*m, o, mod, iface, j.error);
    spirv::release(m);
    return nullptr;
}
StackJob lowerOnStack(uint32_t levels, size_t kb) {
    StackJob job;
    job.words = elseIfChain(levels);
    if (!kb) return stackJob(&job), job;  // this thread
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    const long pg = sysconf(_SC_PAGESIZE) > 0 ? sysconf(_SC_PAGESIZE) : 4096;  // (16 KB pages on macOS arm64: round up)
    const size_t bytes = (kb * 1024 + size_t(pg) - 1) / size_t(pg) * size_t(pg);
    if (pthread_attr_setstacksize(&attr, bytes) != 0) {  // refused: skip this size (no job: the caller treats it as not run)
        pthread_attr_destroy(&attr);
        job.skipped = true;
        return job;
    }
    pthread_t th;
    if (pthread_create(&th, &attr, stackJob, &job) != 0) std::abort();
    pthread_join(th, nullptr);
    pthread_attr_destroy(&attr);
    return job;
}

// The lowering recurses once per nested region and also measures the stack left in its thread: a worker thread with a small stack gets
// a clean error from a deep ladder, never a SIGSEGV. How many levels fit in a given stack depends on the frame sizes of the compiler
// and the sanitizers, so no threshold is assumed: over a wide range of depths and stack sizes every job must either lower or fail with
// a message (and the full stack of this thread lowers what a small one lowers).
void stackLimit() {
    int lowered = 0, refused = 0, skipped = 0, total = 0;
    for (size_t kb : {size_t(64), size_t(128), size_t(256), size_t(512)}) {
        for (uint32_t levels : {20u, 100u, 300u, 1000u, 3000u, 20000u}) {
            const StackJob j = lowerOnStack(levels, kb);
            ++total;
            if (j.skipped) {
                ++skipped;
                continue;
            }
            CHECK(j.parsed);
            if (j.lowered) {
                ++lowered;
                continue;
            }
            ++refused;
            CHECK(!j.error.empty());
            if (j.error.empty()) std::fprintf(stderr, "  %u levels on %zu KB: refused without a message\n", levels, kb);
        }
    }
    // The ends of the range: a short ladder on the largest stack lowers (even under a sanitizer), a ladder past the level cap does not.
    if (skipped == total) {  // (every stack size was refused by the system: nothing was tested)
        std::printf("stack_limit: skipped, the system refused every thread stack size\n");
        return;
    }
    CHECK(lowered > 0 && refused > 0);
    // A job that a small stack lowers lowers on the full stack of this thread too.
    const StackJob big = lowerOnStack(300, 0);
    CHECK(big.lowered);
    if (!big.lowered) std::fprintf(stderr, "  300 levels on the main thread: %s\n", big.error.c_str());
}

// A private structure of 40000 floats copied whole: each member's natural offset used to be summed from the first
// member (40000^2 / 2 lookups, about 10 s); the prefix sums are now built once per structure.
void wideStruct() {
    constexpr uint32_t kMembers = 40000;
    std::vector<uint32_t> w = {0x07230203u, 0x00010300u, 0, 0, 0};
    auto ins = [&](uint32_t op, std::initializer_list<uint32_t> ops) {
        w.push_back(uint32_t(ops.size() + 1) << 16 | op);
        w.insert(w.end(), ops.begin(), ops.end());
    };
    // ids: 1 main, 2 void, 3 fn, 4 f32, 5 struct, 6 ptr, 7 var, 8 value
    ins(17, {1});                             // OpCapability Shader
    ins(14, {0, 1});                          // OpMemoryModel Logical GLSL450
    ins(15, {5, 1, 0x6e69616du, 0});          // OpEntryPoint GLCompute %1 "main"
    ins(16, {1, 17, 1, 1, 1});                // OpExecutionMode LocalSize 1 1 1
    ins(19, {2});                             // OpTypeVoid
    ins(33, {3, 2});                          // OpTypeFunction
    ins(22, {4, 32});                         // OpTypeFloat 32
    w.push_back(uint32_t(kMembers + 2) << 16 | 30), w.push_back(5);  // OpTypeStruct %5 {%4 x kMembers}
    w.insert(w.end(), kMembers, 4);
    ins(32, {6, 6, 5});                       // OpTypePointer Private
    ins(59, {6, 7, 6});                       // OpVariable Private
    ins(54, {2, 1, 0, 3});                    // OpFunction
    ins(248, {9});                            // OpLabel
    ins(61, {5, 8, 7});                       // OpLoad %5 %8 %7
    ins(62, {7, 8});                          // OpStore
    ins(253, {});                             // OpReturn
    ins(56, {});                              // OpFunctionEnd
    w[3] = 10;
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    std::string error;
    spirv::SpirvModule* m = spirv::parse(w.data(), w.size(), error);
    CHECK(m != nullptr);
    if (!m) return std::fprintf(stderr, "  parse: %s\n", error.c_str()), void();
    const auto t0 = std::chrono::steady_clock::now();
    vpir::Module mod;
    spirv::SpirvInterface iface;
    const bool ok = spirv::lower(*m, o, mod, iface, error);
    spirv::release(m);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!ok) std::fprintf(stderr, "  lower: %s\n", error.c_str());
    CHECK(ok);
    if (ms > 4000) std::fprintf(stderr, "  lowering took %.0f ms\n", ms);
    CHECK(ms < 4000);
}

// tests/fuzz/*.spv: modules a mutation fuzzer (ASan/UBSan) broke the reader with - types and constants that contain
// themselves, instructions cut short at the end of the words, a function without blocks, access chains that loop,
// arrays of 2^32 elements, structures nested as a DAG, control flow that leads back into itself. Each is parsed and
// lowered for every stage under a few option sets: whatever the verdict (parse error, lowering error, a module), it
// must come back without crashing, without a sanitizer report, and within a few seconds.
void malformed(const std::string& dir) {
    namespace fs = std::filesystem;
    size_t files = 0;
    for (const auto& e : fs::directory_iterator(dir + "/fuzz")) {
        if (e.path().extension() != ".spv") continue;
        ++files;
        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<uint32_t> words = readWords(e.path().string());
        std::string error;
        spirv::SpirvModule* m = spirv::parse(words.data(), words.size(), error);
        if (!m) continue;
        for (uint32_t variant : {0u, 2u | 4u, 16u | 32u | 64u}) {
            for (vpir::Stage st : {vpir::Stage::Vertex, vpir::Stage::Fragment, vpir::Stage::Compute}) {
                spirv::SpirvLowerOptions o;
                o.entry = "main", o.stage = st, o.viewCount = (variant & 2) ? 2 : 1, o.amplify = variant & 4;
                if (variant & 16) o.uiMaskSlot = 3, o.alphaTest = true;
                o.d24Emul = variant & 32;
                o.viewportIndex = o.rateMapPosition = variant & 64;
                uint16_t slot = 0;
                for (uint32_t set = 0; set < 3; ++set)
                    for (uint32_t b = 0; b < 6; ++b, slot += 4) {
                        static const vpgpu::BindKind kinds[] = {vpgpu::BindKind::Buffer, vpgpu::BindKind::ConstantBuffer, vpgpu::BindKind::Texture,
                                                                 vpgpu::BindKind::StorageTexture, vpgpu::BindKind::Sampler};
                        const vpgpu::BindKind k = kinds[(set + b) % 5];
                        o.layout.push_back(bind(set, b, k, slot, k == vpgpu::BindKind::Texture ? uint16_t(slot + 1) : uint16_t(0xFFFF)));
                        o.layout.back().arraySize = 1 + b % 3;
                    }
                if (st == vpir::Stage::Vertex) {
                    spirv::VertexBufferBinding vb;
                    vb.binding = 0, vb.stride = 32, vb.slot = 40;
                    o.vertexBuffers.push_back(vb);
                    for (uint32_t l = 0; l < 4; ++l) {
                        spirv::VertexAttribute a;
                        a.location = l, a.offset = l * 4, a.format = vpgpu::Format::RGBA32Float;
                        o.vertexAttributes.push_back(a);
                    }
                }
                o.specConstants = {{0, 7}, {1, 0xFFFFFFFFu}};
                vpir::Module mod;
                spirv::SpirvInterface iface;
                if (spirv::lower(*m, o, mod, iface, error)) vpmsl::emitMsl(mod);
            }
        }
        spirv::release(m);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (ms > 10000) {
            std::fprintf(stderr, "  %s took %.0f ms\n", e.path().filename().string().c_str(), ms);
            ++failures;
        }
    }
    CHECK(files >= 10);
}

// A struct that holds a pointer to itself (buffer_reference lists): OpTypeForwardPointer names the pointer before the
// struct that contains it. The reader requires types to be declared before they are used, except the pointee of a pointer.
void forwardPointer() {
    auto op = [](std::vector<uint32_t>& w, uint32_t code, std::vector<uint32_t> operands) {
        w.push_back(uint32_t(operands.size() + 1) << 16 | code);
        w.insert(w.end(), operands.begin(), operands.end());
    };
    std::vector<uint32_t> w = {0x07230203, 0x00010500, 0, 16, 0};
    const uint32_t kPhysical = 5349;
    op(w, 17, {1});                                           // OpCapability Shader
    op(w, 14, {0, 1});                                        // OpMemoryModel Logical GLSL450
    op(w, 15, {5, 2, 0x6e69616d, 0});                         // OpEntryPoint GLCompute %2 "main"
    op(w, 19, {3});                                           // OpTypeVoid %3
    op(w, 33, {4, 3});                                        // OpTypeFunction %4 %3
    op(w, 21, {6, 32, 0});                                    // OpTypeInt %6 32 0
    op(w, 39, {7, kPhysical});                                // OpTypeForwardPointer %7
    op(w, 30, {8, 7, 6});                                     // OpTypeStruct %8 %7 %6
    op(w, 32, {7, kPhysical, 8});                             // OpTypePointer %7 PhysicalStorageBuffer %8
    op(w, 54, {3, 2, 0, 4});                                  // OpFunction %3 %2 None %4
    op(w, 248, {9});                                          // OpLabel %9
    op(w, 253, {});                                           // OpReturn
    op(w, 56, {});                                            // OpFunctionEnd
    std::string error;
    spirv::SpirvModule* m = spirv::parse(w.data(), w.size(), error);
    if (!m) std::fprintf(stderr, "  parse: %s\n", error.c_str());
    CHECK(m != nullptr);
    if (m) spirv::release(m);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: vpspirv_tests <case> <plain|opt> <tests dir>\n");
        return 2;
    }
    std::string c = argv[1], v = std::string(argv[2]) == "opt" ? "opt" : "", dir = argv[3];
    if (c == "mesh_multiview") meshMultiview(dir, v);
    else if (c == "amplify") amplify(dir, v);
    else if (c == "malformed") malformed(dir);
    else if (c == "forward_pointer") forwardPointer();
    else if (c == "offset_overflow") offsetOverflow(dir);
    else if (c == "stack_limit") stackLimit();
    else if (c == "structure") structure();
    else if (c == "wide_struct") wideStruct();
    else if (c == "unpack_fetch") unpackFetch(dir);
    else if (c == "lit") lit(dir, v);
    else if (c == "cull") cull(dir, v);
    else if (c == "subpass") subpass(dir, v);
    else if (c == "wg_spec_atomic") wgSpecAtomic(dir, v);
    else if (c == "array_length") arrayLength(dir);
    else if (c == "tex1d") tex1d(dir);
    else if (c == "tex_ms") texMs(dir);
    else if (c == "clip") clip(dir);
    else if (c == "fixes") fixes(dir);
    else if (c == "dyn_index") dynIndex(dir);
    else if (c == "dyn_index_big") dynIndexBig(dir);
    else if (c == "bool_iface") boolIface(dir);
    else if (c == "dyn_buffer") dynBuffer(dir);
    else if (c == "subgroup_cube") subgroupCube(dir);
    else if (c == "subgroup_shuffle") subgroupShuffle(dir);
    else if (c == "subgroup_arith") subgroupArith(dir);
    else if (c == "texel_fetch") texelFetch(dir);
    else if (c == "int16") int16(dir);
    else if (c == "storage16") storage16(dir);
    else if (c == "draw_params") drawParams(dir);
    else if (c == "helper") helper(dir);
    else if (c == "float16") float16(dir);
    else if (c == "texel_stores") texelStores(dir);
    else if (c == "texel_formats") texelFormats(dir);
    else if (c == "membar") memBar(dir);
    else if (c == "bda") bda(dir);
    else if (c == "varptr") varPtr(dir);
    else if (c == "ptr_negative") ptrNegative(dir);
    else if (c == "ptr_narrow") ptrNarrow(dir);
    else if (c == "ptr_unsigned_neg") ptrUnsignedNeg(dir);
    else if (c == "ptr_unsigned_big") ptrUnsignedBig(dir);
    else if (c == "int8") int8(dir);
    else if (c == "dot") dotProduct(dir);
    else if (c == "zero_init") zeroInit(dir);
    else if (c == "float_to_int") floatToInt(dir);
    else if (c == "float_to_int_bounded") floatToIntBounded(dir);
    else if (c == "f16_math") f16Math(dir);
    else if (c == "ldexp_range") ldexpRange(dir);
    else if (c == "output_init") outputInit(dir);
    else if (c == "varptr_call") varPtrCall(dir);
    else if (c == "membar_image") memBarImage(dir);
    else if (c == "bitcast16") bitcast16(dir);
    else if (c == "f16_insert") f16Insert(dir);
    else if (c == "submask") subgroupMasks(dir);
    else if (c == "sample_mask") sampleMask(dir);
    else if (c == "d24_emul") d24Emul(dir);
    else if (c == "depth_tile") depthTile(dir);
    else if (c == "ycbcr") ycbcr(dir);
    else if (c == "robust") robust(dir);
    else if (c == "correctness") correctness(dir);
    else {
        std::fprintf(stderr, "unknown case %s\n", c.c_str());
        return 2;
    }
    if (failures) std::fprintf(stderr, "%d FAILED\n", failures);
    return failures ? 1 : 0;
}
