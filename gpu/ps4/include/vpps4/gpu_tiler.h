// GpuTiler: detiling (guest tiled memory -> image) and tiling (image -> guest tiled memory) as
// Metal compute work on the GPU timeline, recorded into the frame next to the draws, so the CPU
// never walks a texture and a re-upload costs a dispatch instead of O(texels) of CPU time.
//
// One generic kernel per direction (MSL generated here, with the addressing of tiling_addr.h
// pasted in); per mip level one dispatch whose parameter block (VptParams) carries that level's
// layout. Detiling dispatches write a linear scratch buffer that a buffer->texture copy then moves
// into the image (block-compressed and sRGB textures cannot be storage-written); tiling does the
// reverse. A third kernel decodes block-compressed surfaces (BC1-BC7, bc_block.h) on hosts without BC: it reads the blocks
// from guest memory (detiling as it goes) and writes the decoded texels the same way, a linear buffer for a copy into the
// image. The CPU path (detileLevels/tileLevels in tiling.h) stays for tests and as the fallback
// while the kernels' pipelines are still compiling.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "vpgpu/core/device.h"
#include "vpmsl/msl.h"
#include "vpps4/tiling.h"

namespace vpps4 {

struct BufferRange {
    vpgpu::BufferId buffer;
    uint64_t offset = 0, size = 0;
};

// Which part of the surface to move: levels [firstLevel, firstLevel + levelCount), slices
// [firstSlice, firstSlice + sliceCount) of each (array layers / cube faces / volume depth;
// sliceCount 0 = all), one sample plane.
struct Subresource {
    uint32_t firstLevel = 0, levelCount = 1;
    uint32_t firstSlice = 0, sliceCount = 0;
    uint32_t sample = 0;
    vpgpu::Aspect aspect = vpgpu::Aspect::All;  // Depth for depth images
};

// Host-visible memory the GPU reads or writes during the frame (the renderer's upload ring):
// `bytes` long, 256-byte aligned inside `buffer` at `offset`, and `memory` is where the CPU sees it.
struct Scratch {
    vpgpu::BufferId buffer;
    uint64_t offset = 0;
    uint8_t* memory = nullptr;
};
using ScratchAlloc = std::function<bool(uint64_t bytes, Scratch& out)>;

// What the BC kernel decodes: `kind` is a bc::Kind value (bc_decode.h), `isSigned` the snorm / signed-float variants.
struct BcJob {
    uint32_t kind = 0;
    bool isSigned = false;
};

class GpuTiler {
public:
    // Registers a translated shader under its id with the backend's shader provider.
    using ShaderSink = std::function<void(uint64_t id, const vpmsl::MslShader&)>;

    enum class Result : uint8_t {
        Recorded,     // commands are in the frame
        NotReady,     // the kernel's pipeline is still compiling: use the CPU path this time
        Unsupported,  // layout not covered, or a range / scratch request that cannot be served
    };

    GpuTiler(vpgpu::Device&, ShaderSink, ScratchAlloc);

    // Tiled guest memory `src` (the surface base; layout.bytes readable) -> `dst`, mip levels and
    // slices of `sub`. `dst` must have the layout's mip count and slices.
    Result detile(vpgpu::Frame& cmd, BufferRange src, vpgpu::TextureId dst, const SurfaceLayout& desc, const Subresource& sub);
    // `src` -> tiled guest memory `dst` (layout.bytes writable).
    Result retile(vpgpu::Frame& cmd, vpgpu::TextureId src, BufferRange dst, const SurfaceLayout& desc, const Subresource& sub);

    // Block-compressed guest memory `src` (the surface base; layout.bytes readable, `desc` in blocks) -> the decoded texels
    // of `dst` (its host format; texel dimensions), levels and slices of `sub`. Same results as the CPU decoder.
    Result decodeBc(vpgpu::Frame& cmd, BufferRange src, vpgpu::TextureId dst, const SurfaceLayout& desc, const Subresource& sub,
                    const BcJob& job);

    // Starts compiling the BC kernel in the background (a host without BC calls it at start), so the first block-compressed
    // texture is already decoded on the GPU instead of by the CPU while it compiles.
    void prefetchBc();

    // The Metal source and argument layout of the kernels (for tests and the shader table).
    static vpmsl::MslShader detileShader();
    static vpmsl::MslShader retileShader();
    static vpmsl::MslShader bcShader();

    uint64_t dispatches() const { return dispatches_; }

private:
    struct Kernel {
        vpmsl::MslShader shader;
        uint64_t id = 0;
        bool registered = false;
    };
    Result run(bool detile, vpgpu::Frame& cmd, BufferRange tiled, vpgpu::TextureId image, const SurfaceLayout& desc,
               const Subresource& sub, const BcJob* bc = nullptr);
    bool kernelReady(Kernel&, const char* label, vpgpu::PipelineKey& key, bool& failed,
                     vpgpu::Priority priority = vpgpu::Priority::Visible);

    vpgpu::Device& device_;
    ShaderSink sink_;
    ScratchAlloc scratch_;
    Kernel detile_, retile_, bc_;
    uint64_t dispatches_ = 0;
};

}  // namespace vpps4
