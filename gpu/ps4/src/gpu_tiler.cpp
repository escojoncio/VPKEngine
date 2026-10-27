// GpuTiler (see gpu_tiler.h). The kernels are plain MSL in the Metal 3.1 dialect the backend
// compiles for the rest of the PS4 path: one VpArgs argument table (buffer(0)) in the order the
// backend writes it (Buffer slots, then ConstantBuffer slots), one thread per element.
#include "vpps4/gpu_tiler.h"

#include <algorithm>
#include <cstring>

extern const char* const kVpps4TilingAddrSource;  // tiling_addr.h as text (generated at configure time)

namespace vpps4 {
namespace {

constexpr uint32_t kGroup = 8;          // threads per threadgroup edge
constexpr uint64_t kAlign = 256;        // scratch pieces
static_assert(sizeof(VptParams) % 16 == 0);

uint64_t alignUp(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

std::string kernelSource(bool detile) {
    std::string s = "#include <metal_stdlib>\nusing namespace metal;\n\n";
    std::string addressing = kVpps4TilingAddrSource;
    if (const size_t once = addressing.find("#pragma once\n"); once != std::string::npos) addressing.erase(once, 13);
    s += addressing;
    s += "\nstruct VpArgs {\n";
    s += detile ? "    const device uchar* tiled;\n    device uchar* linear;\n" : "    const device uchar* linear;\n    device uchar* tiled;\n";
    s += "    constant uchar* params;\n};\n\n";
    s += "[[max_total_threads_per_threadgroup(64)]] kernel void vp_main(constant VpArgs& vp_args [[buffer(0)]],\n"
         "        uint3 gid [[thread_position_in_grid]]) {\n"
         "    constant VptParams& P = *(constant VptParams*)vp_args.params;\n"
         "    if (!vpt_in_range(P, gid.x, gid.y, gid.z)) return;\n"
         "    const uint t = vpt_tiled_offset(P, gid.x, gid.y, gid.z);\n"
         "    const uint l = vpt_linear_offset(P, gid.x, gid.y, gid.z);\n"
         "    const uint bpe = P.level.bpe;\n";
    if (detile)
        s += "    const device uchar* from = vp_args.tiled + t;\n    device uchar* to = vp_args.linear + l;\n";
    else
        s += "    const device uchar* from = vp_args.linear + l;\n    device uchar* to = vp_args.tiled + t;\n";
    s += "    if (bpe >= 4u) {\n"
         "        for (uint i = 0u; i < bpe / 4u; ++i) ((device uint*)to)[i] = ((const device uint*)from)[i];\n"
         "    } else {\n"
         "        for (uint i = 0u; i < bpe; ++i) to[i] = from[i];\n"
         "    }\n}\n";
    return s;
}

vpmsl::MslShader build(bool detile) {
    vpmsl::MslShader sh;
    sh.source = kernelSource(detile);
    sh.workgroupSize = {kGroup, kGroup, 1};
    // VpArgs: Buffer 0, Buffer 1, ConstantBuffer 0; 8 bytes each.
    sh.arguments.push_back({detile ? "tiled" : "linear", vpir::ResourceKind::Buffer, 0, 0});
    sh.arguments.push_back({detile ? "linear" : "tiled", vpir::ResourceKind::Buffer, 1, 8});
    sh.arguments.push_back({"params", vpir::ResourceKind::ConstantBuffer, 0, 16});
    return sh;
}

vpgpu::Access buffer(vpgpu::BufferId b, uint64_t offset, uint64_t size, bool write, vpgpu::BindKind kind, uint16_t slot) {
    vpgpu::Access a;
    a.range = vpgpu::ResourceRange::bytes(b, offset, size);
    a.stages = vpgpu::stage::compute;
    a.write = write;
    a.kind = kind;
    a.slot = slot;
    return a;
}

}  // namespace

vpmsl::MslShader GpuTiler::detileShader() { return build(true); }
vpmsl::MslShader GpuTiler::retileShader() { return build(false); }

GpuTiler::GpuTiler(vpgpu::Device& device, ShaderSink sink, ScratchAlloc scratch)
    : device_(device), sink_(std::move(sink)), scratch_(std::move(scratch)) {}

bool GpuTiler::kernelReady(Kernel& k, const char* label, vpgpu::PipelineKey& key, bool& failed) {
    failed = false;
    if (!k.id) {
        k.shader = build(label[0] == 'd');
        k.id = vpmsl::shaderId(k.shader);
    }
    if (!k.registered) {
        if (sink_) sink_(k.id, k.shader);
        k.registered = true;
    }
    vpgpu::PipelineState ps;
    ps.computeShader = k.id;
    ps.label = label;
    key = ps.key();
    const auto st = device_.pipelines().request(ps, vpgpu::Priority::Visible);
    failed = st == vpgpu::PipelineCache::Status::Failed;
    return st == vpgpu::PipelineCache::Status::Ready;
}

GpuTiler::Result GpuTiler::detile(vpgpu::Frame& cmd, BufferRange src, vpgpu::TextureId dst, const SurfaceLayout& desc,
                                  const Subresource& sub) {
    return run(true, cmd, src, dst, desc, sub);
}

GpuTiler::Result GpuTiler::retile(vpgpu::Frame& cmd, vpgpu::TextureId src, BufferRange dst, const SurfaceLayout& desc,
                                  const Subresource& sub) {
    return run(false, cmd, dst, src, desc, sub);
}

GpuTiler::Result GpuTiler::run(bool detile, vpgpu::Frame& cmd, BufferRange tiled, vpgpu::TextureId image,
                               const SurfaceLayout& desc, const Subresource& sub) {
    if (!desc.ok || !tiled.buffer || !image || !sub.levelCount) return Result::Unsupported;
    const vpgpu::TextureDesc* td = device_.resources().get(image);
    if (!td || td->samples != 1) return Result::Unsupported;
    if (uint64_t(sub.firstLevel) + sub.levelCount > desc.levels.size() || sub.firstLevel + sub.levelCount > td->mips ||
        sub.sample >= desc.surface.samples || tiled.size < desc.bytes)
        return Result::Unsupported;
    const bool volume = td->type == vpgpu::TextureType::Tex3D;
    if (volume != desc.surface.volume) return Result::Unsupported;
    const uint32_t bpe = desc.surface.bytesPerElement;

    // Plan and size the scratch: per level a parameter block and the level's linear data.
    struct Plan {
        uint32_t level, firstSlice, count;
        uint64_t params, data, dataBytes;
    };
    std::vector<Plan> plan;
    uint64_t total = 0;
    for (uint32_t i = 0; i < sub.levelCount; ++i) {
        const LevelLayout& l = desc.levels[sub.firstLevel + i];
        const uint32_t count = sub.sliceCount ? sub.sliceCount : l.slices - std::min(sub.firstSlice, l.slices);
        if (!count || uint64_t(sub.firstSlice) + count > l.slices) return Result::Unsupported;
        Plan p{sub.firstLevel + i, sub.firstSlice, count, 0, 0, uint64_t(l.width) * l.height * count * bpe};
        p.params = total;
        total += kAlign;
        p.data = total;
        total += alignUp(p.dataBytes, kAlign);
        plan.push_back(p);
    }
    vpgpu::PipelineKey key;
    bool failed = false;
    Kernel& k = detile ? detile_ : retile_;
    if (!kernelReady(k, detile ? "detile" : "retile", key, failed)) return failed ? Result::Unsupported : Result::NotReady;
    Scratch mem;
    if (!scratch_ || !scratch_(total, mem) || !mem.buffer || !mem.memory) return Result::Unsupported;

    for (const Plan& p : plan) {
        const LevelLayout& l = desc.levels[p.level];
        VptParams params{};
        params.level = l.hw;
        params.width = l.width;
        params.height = l.height;
        params.slices = p.count;
        params.firstSlice = p.firstSlice;
        params.sample = sub.sample;
        std::memcpy(mem.memory + p.params, &params, sizeof(params));

        // Copy between the image and the level's linear data.
        vpgpu::Copy c;
        const vpgpu::BufferSpan span{mem.buffer, mem.offset + p.data, p.dataBytes, l.width * bpe, l.height};
        vpgpu::TextureSpan& ts = detile ? c.dstTexture : c.srcTexture;
        ts.texture = image;
        ts.aspect = sub.aspect;
        ts.mip = uint16_t(p.level);
        if (volume) ts.z = p.firstSlice, ts.depth = p.count;
        else ts.layer = uint16_t(p.firstSlice), ts.layerCount = uint16_t(p.count);
        ts.width = std::max(td->width >> p.level, 1u);
        ts.height = std::max(td->height >> p.level, 1u);
        if (detile) c.kind = vpgpu::CopyKind::BufferToTexture, c.srcBuffer = span;
        else c.kind = vpgpu::CopyKind::TextureToBuffer, c.dstBuffer = span;
        c.label = std::string(detile ? "gpu detile mip " : "gpu retile mip ") + std::to_string(p.level);

        vpgpu::Dispatch d;
        d.pipeline = key;
        d.groups = {(l.width + kGroup - 1) / kGroup, (l.height + kGroup - 1) / kGroup, p.count};
        d.label = c.label;
        const vpgpu::BufferId lin = mem.buffer;
        const uint64_t tiledAt = tiled.offset + l.offset;
        if (detile) {
            d.bindings.push_back(buffer(tiled.buffer, tiledAt, l.bytes, false, vpgpu::BindKind::Buffer, 0));
            d.bindings.push_back(buffer(lin, span.offset, p.dataBytes, true, vpgpu::BindKind::Buffer, 1));
        } else {
            d.bindings.push_back(buffer(lin, span.offset, p.dataBytes, false, vpgpu::BindKind::Buffer, 0));
            d.bindings.push_back(buffer(tiled.buffer, tiledAt, l.bytes, true, vpgpu::BindKind::Buffer, 1));
        }
        d.bindings.push_back(buffer(lin, mem.offset + p.params, sizeof(VptParams), false, vpgpu::BindKind::ConstantBuffer, 0));
        if (detile) {
            cmd.dispatch(std::move(d));
            cmd.copy(std::move(c));
        } else {
            cmd.copy(std::move(c));
            cmd.dispatch(std::move(d));
        }
        ++dispatches_;
    }
    return Result::Recorded;
}

}  // namespace vpps4
