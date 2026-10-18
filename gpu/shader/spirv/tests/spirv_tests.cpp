// SPIR-V -> VPIR -> MSL on our own shaders (tests/glsl, compiled with glslang; `opt` = after
// spirv-opt -O, which is what shipping titles carry: phis, merged blocks, folded constants).
// Each case lowers with a driver-style layout, verifies, emits MSL and checks what the lowering
// promises (bindings reported, multiview, the subpass-read pattern, workgroup size…).
#include <algorithm>
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

Result run(const std::string& path, const spirv::SpirvLowerOptions& o, bool fetch = false,
           const vpmsl::EmitOptions& eo = {}) {
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
    // Views 0 and 2 (mask 0b101): view 1 renders to layer 2; one view at bit 2 (mask 0b100)
    // still writes the layer although only one instance per draw is made.
    o.viewMask = 0b101;
    Result g = run(dir + "/mesh_multiview.vert" + (v.empty() ? "" : ".opt") + ".spv", o);
    CHECK(g.ok && g.msl.source.find("2u") != std::string::npos && g.msl.source != r.msl.source);
    o.viewCount = 1, o.viewMask = 0b100;
    Result one = run(dir + "/mesh_multiview.vert" + (v.empty() ? "" : ".opt") + ".spv", o);
    CHECK(one.ok && one.msl.source.find("render_target_array_index") != std::string::npos);
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
        CHECK(m.find("simd_broadcast_first(") != std::string::npos && m.find("simd_all(") != std::string::npos);
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
    CHECK(src.find("texturecube_array<uint") != std::string::npos);
    CHECK(src.find(" % ") != std::string::npos);
    CHECK(src.find("simd_broadcast_first(") != std::string::npos);
    CHECK(src.find("simd_all(") != std::string::npos);
    CHECK(src.find("simd_broadcast_first(true") == std::string::npos);
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
        CHECK(src.find("ul <= vp_args.bs") != std::string::npos);
        bool hasSize = false;
        for (auto& a : r.msl.arguments) hasSize |= a.kind == vpir::ResourceKind::BufferSize;
        CHECK(hasSize);
        if (atomics) CHECK(src.find(") atomic_fetch_") != std::string::npos || src.find(" = atomic_fetch_") != std::string::npos);
        Result plain = run(dir + "/" + file, o);
        CHECK(plain.ok && plain.msl.source.find("ul <= vp_args.bs") == std::string::npos);
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
    else if (c == "wg_spec_atomic") wgSpecAtomic(dir, v);
    else if (c == "array_length") arrayLength(dir);
    else if (c == "tex1d") tex1d(dir);
    else if (c == "tex_ms") texMs(dir);
    else if (c == "clip") clip(dir);
    else if (c == "fixes") fixes(dir);
    else if (c == "dyn_index") dynIndex(dir);
    else if (c == "dyn_buffer") dynBuffer(dir);
    else if (c == "subgroup_cube") subgroupCube(dir);
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
    else if (c == "int8") int8(dir);
    else if (c == "ycbcr") ycbcr(dir);
    else if (c == "robust") robust(dir);
    else {
        std::fprintf(stderr, "unknown case %s\n", c.c_str());
        return 2;
    }
    if (failures) std::fprintf(stderr, "%d FAILED\n", failures);
    return failures ? 1 : 0;
}
