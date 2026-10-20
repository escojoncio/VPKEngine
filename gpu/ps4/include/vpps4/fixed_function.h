// Fixed-function register semantics, as pure functions (the renderer applies their results; the
// tests check them without a GPU): polygon offset, blend factors, stencil operations, scissor
// combination, viewport depth range, polygon mode, index buffer preparation.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "vp4gpu/core/commands.h"
#include "vp4gpu/core/pipeline_cache.h"
#include "vpps4/state.h"

namespace vpps4 {

// What one draw hands to setDepthBias (Metal/Vulkan semantics: constant in units of the depth
// format's resolution, slope scale on the largest slope, clamp on the total).
struct DepthBias {
    bool enabled = false;
    float constant = 0, slope = 0, clamp = 0;
    bool approximated = false;  // front and back asked for different offsets and both faces are drawn
};
// `hostDepth16`: the host depth plane is 16-bit (Z_16 without stencil). Z_16 with stencil lives in a
// 32-bit float plane, whose resolution is 2^8 finer: the constant scales to match.
DepthBias resolveDepthBias(const Raster&, PrimitiveType, uint8_t zFormat, bool hasStencil);

struct BlendResolved {
    vp4gpu::BlendState state;
    bool exact = true;
    bool dualSource = false;  // a factor reads the second pixel shader output
};
// CB_BLENDn_CONTROL + CB_COLORn_*. `integer`: integer render target (blending does not apply).
BlendResolved resolveBlend(const Blend&, const ColorTarget&, bool integer);

struct StencilResolved {
    vp4gpu::StencilFace front, back;
    uint8_t refFront = 0, refBack = 0;
    bool exact = true;
    bool backDiffers = false;
};
StencilResolved resolveStencil(const DepthStencil&, bool readOnly);

// The scissor a draw runs with: screen, window and generic scissors (window offset applied to the
// last two), the viewport scissor when enabled.
Scissor combineScissor(const ScissorRegs&);

struct ViewportZ {
    float znear = 0, zfar = 1;
};
// Depth range of the host viewport for PA_CL_VPORT_Z*: Metal's clip z is 0..w. With a -w..w clip
// space the vertex shader maps z to (z + w) / 2 and the range spans zoffset -/+ zscale.
ViewportZ viewportDepthRange(const Viewport&, bool clipSpaceZeroToW);

// The host viewport rectangle (pixels) of PA_CL_VPORT_X/Y for a pass of `width` x `height`: GCN
// screen = ndc * scale + offset, so the rectangle starts at offset - |scale| and spans 2|scale|. A
// transform that is off (Viewport::xScreen / yScreen) maps the whole target with offset 0. `flipped`:
// positive y scale (the image comes out upside down on Metal). valid false: no usable scale.
struct ViewportRect {
    float x = 0, y = 0, width = 0, height = 0;
    bool flipped = false, valid = false, wholeTarget = false;
};
ViewportRect hostViewportRect(const Viewport&, uint32_t width, uint32_t height);

// Depth clamp instead of clipping z: PA_CL_CLIP_CNTL.ZCLIP_*_DISABLE or CLIP_DISABLE.
bool depthClamp(const Raster&, bool& approximated);

// Triangles drawn as edges: both visible faces in line mode.
struct PolygonMode {
    bool wireframe = false;
    bool pointMode = false;     // no host equivalent: drawn filled
};
PolygonMode resolvePolygonMode(const Raster&, PrimitiveType);

// An index buffer read by the CPU: bytes swapped as VGT_DMA_INDEX_TYPE.SWAP_MODE says, and the
// primitive restart index replaced by the host's (all ones) when restart is on and they differ.
// Returns false when nothing needs changing (the guest buffer can be used as it is).
bool indicesNeedPrep(const DrawState&);
void prepareIndices(const DrawState&, std::span<const uint8_t> in, std::vector<uint8_t>& out);

// Stencil-reference dependent register note for the census.
const char* stencilOpName(uint8_t gcn);

}  // namespace vpps4
