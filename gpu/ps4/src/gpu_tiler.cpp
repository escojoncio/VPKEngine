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
    s += kVpps4TilingAddrSource;
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

vp4msl::MslShader build(bool detile) {
    vp4msl::MslShader sh;
    sh.source = kernelSource(detile);
    sh.workgroupSize = {kGroup, kGroup, 1};
    // VpArgs: Buffer 0, Buffer 1, ConstantBuffer 0; 8 bytes each.
    sh.arguments.push_back({detile ? "tiled" : "linear", vp4ir::ResourceKind::Buffer, 0, 0});
    sh.arguments.push_back({detile ? "linear" : "tiled", vp4ir::ResourceKind::Buffer, 1, 8});
    sh.arguments.push_back({"params", vp4ir::ResourceKind::ConstantBuffer, 0, 16});
    return sh;
}

vp4gpu::Access buffer(vp4gpu::BufferId b, uint64_t offset, uint64_t size, bool write, vp4gpu::BindKind kind, uint16_t slot) {
    vp4gpu::Access a;
    a.range = vp4gpu::ResourceRange::bytes(b, offset, size);
    a.stages = vp4gpu::stage::compute;
    a.write = write;
    a.kind = kind;
    a.slot = slot;
    return a;
}

}  // namespace

vp4msl::MslShader GpuTiler::detileShader() { return build(true); }
vp4msl::MslShader GpuTiler::retileShader() { return build(false); }

GpuTiler::GpuTiler(vp4gpu::Device& device, ShaderSink sink, ScratchAlloc scratch)
    : device_(device), sink_(std::move(sink)), scratch_(std::move(scratch)) {}

bool GpuTiler::kernelReady(Kernel& k, const char* label, vp4gpu::PipelineKey& key, bool& failed) {
    failed = false;
    if (!k.id) {
        k.shader = build(label[0] == 'd');
        k.id = vp4msl::shaderId(k.shader);
    }
    if (!k.registered) {
        if (sink_) sink_(k.id, k.shader);
        k.registered = true;
    }
    vp4gpu::PipelineState ps;
    ps.computeShader = k.id;
    ps.label = label;
    key = ps.key();
    const auto st = device_.pipelines().request(ps, vp4gpu::Priority::Visible);
    failed = st == vp4gpu::PipelineCache::Status::Failed;
    return st == vp4gpu::PipelineCache::Status::Ready;
}

GpuTiler::Result GpuTiler::detile(vp4gpu::Frame& cmd, BufferRange src, vp4gpu::TextureId dst, const SurfaceLayout& desc,
                                  const Subresource& sub) {
    return run(true, cmd, src, dst, desc, sub);
}

GpuTiler::Result GpuTiler::retile(vp4gpu::Frame& cmd, vp4gpu::TextureId src, BufferRange dst, const SurfaceLayout& desc,
                                  const Subresource& sub) {
    return run(false, cmd, dst, src, desc, sub);
}

GpuTiler::Result GpuTiler::run(bool detile, vp4gpu::Frame& cmd, BufferRange tiled, vp4gpu::TextureId image,
                               const SurfaceLayout& desc, const Subresource& sub) {
    if (!desc.ok || !tiled.buffer || !image || !sub.levelCount) return Result::Unsupported;
    const vp4gpu::TextureDesc* td = device_.resources().get(image);
    if (!td || td->samples != 1) return Result::Unsupported;
    if (uint64_t(sub.firstLevel) + sub.levelCount > desc.levels.size() || sub.firstLevel + sub.levelCount > td->mips ||
        sub.sample >= desc.surface.samples || tiled.size < desc.bytes)
        return Result::Unsupported;
    const bool volume = td->type == vp4gpu::TextureType::Tex3D;
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
    vp4gpu::PipelineKey key;
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
        vp4gpu::Copy c;
        const vp4gpu::BufferSpan span{mem.buffer, mem.offset + p.data, p.dataBytes, l.width * bpe, l.height};
        vp4gpu::TextureSpan& ts = detile ? c.dstTexture : c.srcTexture;
        ts.texture = image;
        ts.mip = uint16_t(p.level);
        if (volume) ts.z = p.firstSlice, ts.depth = p.count;
        else ts.layer = uint16_t(p.firstSlice), ts.layerCount = uint16_t(p.count);
        ts.width = std::max(td->width >> p.level, 1u);
        ts.height = std::max(td->height >> p.level, 1u);
        if (detile) c.kind = vp4gpu::CopyKind::BufferToTexture, c.srcBuffer = span;
        else c.kind = vp4gpu::CopyKind::TextureToBuffer, c.dstBuffer = span;
        c.label = std::string(detile ? "gpu detile mip " : "gpu retile mip ") + std::to_string(p.level);

        vp4gpu::Dispatch d;
        d.pipeline = key;
        d.groups = {(l.width + kGroup - 1) / kGroup, (l.height + kGroup - 1) / kGroup, p.count};
        d.label = c.label;
        const vp4gpu::BufferId lin = mem.buffer;
        const uint64_t tiledAt = tiled.offset + l.offset;
        if (detile) {
            d.bindings.push_back(buffer(tiled.buffer, tiledAt, l.bytes, false, vp4gpu::BindKind::Buffer, 0));
            d.bindings.push_back(buffer(lin, span.offset, p.dataBytes, true, vp4gpu::BindKind::Buffer, 1));
        } else {
            d.bindings.push_back(buffer(lin, span.offset, p.dataBytes, false, vp4gpu::BindKind::Buffer, 0));
            d.bindings.push_back(buffer(tiled.buffer, tiledAt, l.bytes, true, vp4gpu::BindKind::Buffer, 1));
        }
        d.bindings.push_back(buffer(lin, mem.offset + p.params, sizeof(VptParams), false, vp4gpu::BindKind::ConstantBuffer, 0));
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
