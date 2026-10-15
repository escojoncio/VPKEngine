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
        // The driver's step for input attachments: texture slot -> colour attachment.
        std::map<uint32_t, uint32_t> att{{0, 0}, {1, 1}};
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
    vpireval::Invocation inv;
    std::string err;
    CHECK(vpireval::run(r.module, inv, err));
    if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
    const std::vector<uint8_t>& mem = inv.memory[0];
    CHECK(mem.size() >= 32);
    if (mem.size() < 32) return;
    int32_t got[8];
    std::memcpy(got, mem.data(), 32);
    const int32_t want[8] = {-25536, 40000, -1596, 2500, 1, 1, -2, 32767};
    for (int k = 0; k < 8; ++k) {
        if (got[k] != want[k]) std::fprintf(stderr, "  r[%d] = %d, want %d\n", k, got[k], want[k]);
        CHECK(got[k] == want[k]);
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
    CHECK(r.msl.source.find(std::string("constant uint& ") ) != std::string::npos);
    CHECK(r.msl.source.find(vpmsl::kDrawIndexBinding) != std::string::npos);
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
    else if (c == "draw_params") drawParams(dir);
    else {
        std::fprintf(stderr, "unknown case %s\n", c.c_str());
        return 2;
    }
    if (failures) std::fprintf(stderr, "%d FAILED\n", failures);
    return failures ? 1 : 0;
}
