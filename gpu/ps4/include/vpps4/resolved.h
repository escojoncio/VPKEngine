// What the renderer RESOLVED for one draw / dispatch / DMA, for a host that compares it with another
// renderer (Hooks::resolved, opt-in; nothing is built without it). Not the registers the host passed in:
// the targets with the host formats chosen, the size each buffer was bound with after the host clamped it
// to the memory that exists, the textures and samplers found, the viewport and fixed-function state.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "vpps4/formats.h"

namespace vpps4 {

struct ResolvedTarget {
    bool used = false;
    uint64_t address = 0;
    uint32_t width = 0, height = 0, pitch = 0;      // width / height of the target the renderer holds
    uint8_t format = 0, numberType = 0, compSwap = 0, tileMode = 0;
    uint16_t sliceStart = 0, sliceCount = 0;
    uint8_t samples = 0;                            // the title's colour fragments
    std::string hostFormat;                         // vpgpu::formatInfo(...).name
    bool cleared = false;                           // the draw's pass starts with a clear of the target (a fast clear that waited for it)
};

struct ResolvedDepth {
    bool used = false;
    uint64_t zAddress = 0, stencilAddress = 0;
    uint32_t width = 0, height = 0;
    uint8_t zFormat = 0, stencilFormat = 0, tileMode = 0, samples = 0;
    std::string hostFormat;
    bool depthClear = false, stencilClear = false;
};

// `stage` is a vpgcn::ShaderStage: 0 vertex, 1 pixel, 2 compute, 3 export (ES), 4 geometry, 5 copy shader.
// Only buffers at a guest address are listed (not user data, rings, GDS or the geometry scratch).
struct ResolvedBuffer {
    uint8_t stage = 0;
    uint64_t address = 0;
    uint64_t size = 0;      // bytes bound; 0: not bound for real (zeros)
    uint32_t stride = 0;
    bool write = false;
    bool pointer = false;   // reached by an address (s_load through a pointer, a program image), not through a V#
};

struct ResolvedTexture {
    uint8_t stage = 0;
    ImageDesc image;        // the T# as decoded
    bool storage = false;
    bool found = false;
    std::string hostFormat; // the host texture bound
    uint32_t hostWidth = 0, hostHeight = 0;
};

// Address modes: 0 repeat, 1 mirror, 2 clamp to edge, 3 mirror clamp to edge, 4 border. Filters: 0 nearest, 1 linear.
struct ResolvedSampler {
    uint8_t stage = 0;
    uint8_t addressU = 0, addressV = 0, addressW = 0;
    uint8_t magFilter = 0, minFilter = 0, mipFilter = 0;
    bool anisotropy = false, compare = false;
};

struct Resolved {
    enum class Kind : uint8_t { Draw, Dispatch, Fill, Copy };
    enum class Status : uint8_t { Drawn, Skipped, Handled };
    Kind kind = Kind::Draw;
    Status status = Status::Drawn;
    std::string reason;     // why skipped / what it was handled as
    std::array<ResolvedTarget, 8> color;
    ResolvedDepth depth;
    std::vector<ResolvedBuffer> buffers;
    std::vector<ResolvedTexture> textures;
    std::vector<ResolvedSampler> samplers;
    uint8_t topology = 0;   // host class of the input primitive: 0 points, 1 lines, 2 triangles
    bool viewportValid = false;
    float viewport[6]{};    // x, y, width, height, min depth, max depth
    bool scissorValid = false;
    int32_t scissor[4]{};   // x, y, width, height
    bool stateValid = false;
    uint8_t cull = 0;       // 0 none, 1 front, 2 back
    bool frontClockwise = false;
    bool depthTest = false, depthWrite = false;
    uint8_t depthFunc = 7;  // FRAG_* (7 always)
    bool stencilTest = false;
    std::array<bool, 8> blendEnable{};
};

}  // namespace vpps4
