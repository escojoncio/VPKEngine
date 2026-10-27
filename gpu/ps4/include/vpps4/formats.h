// Liverpool register encodings -> gpu/core enums. Written from AMD's public register and ISA
// documentation (CB/DB/SQ formats, blend and compare encodings, sampler and image descriptors).
#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "vpgpu/core/commands.h"
#include "vpgpu/core/pipeline_cache.h"
#include "vpgpu/core/resources.h"
#include "vpgcn/formats.h"
#include "vpgcn/lift.h"

namespace vpps4 {

// What the host can do beyond the baseline (the host sets it from the device).
struct FormatCaps {
    bool bc = true;  // BC1-7 sampled natively (Metal supportsBCTextureCompression)
};
void setFormatCaps(const FormatCaps&);
const FormatCaps& formatCaps();

// ---- Textures -------------------------------------------------------------------------------------
//
// A T# (data format, number format) -> the host texture and what the shader must still do:
//   native      a Metal format that holds the texels as the hardware reads them;
//   converted   a Metal format of the same texel size plus TextureConv (scaled formats through the
//               normalised format with an affine, sRGB the host lacks decoded in the shader, the
//               channel order of packed 16-bit formats);
//   container   a raw unsigned-integer texture of the texel size (conv.manual): the shader fetches
//               and decodes any layout (nearest filtering, no mips unless the lod is explicit).
// Undefined: no way (reserved formats, BC without host support).
struct TexturePlan {
    vpgpu::Format format = vpgpu::Format::Undefined;
    vpgcn::TextureConv conv;          // select = the format's own channel order (identity when native)
    enum Elem : uint8_t { Float, Uint, Sint } elem = Float;  // what the shader declares the texture as
    uint8_t widthDivisor = 1;         // texels per element along x (the host texture is narrower)
    bool exact = true;                // false: an approximation was taken (`why` says which)
    const char* why = nullptr;
};
// `storage`: the texture is written by shaders (no sRGB / packed formats: containers instead).
TexturePlan texturePlan(uint8_t dataFormat, uint8_t numFormat, bool storage = false);
// The raw-container plan of a format (conv.manual); format Undefined if the layout has none.
TexturePlan containerTexturePlan(uint8_t dataFormat, uint8_t numFormat);
// Image descriptor IMG_DATA_FORMAT / IMG_NUM_FORMAT -> sampled texture format.
vpgpu::Format textureFormat(uint8_t dataFormat, uint8_t numFormat);
// Final component select of a texture read: T# DST_SEL (SQ_SEL_*) composed with the plan's own.
void composeSelect(const std::array<uint8_t, 4>& dstSel, const uint8_t planSelect[4], uint8_t out[4]);

// ---- Colour targets -----------------------------------------------------------------------------------
//
// CB_COLORn_INFO FORMAT / NUMBER_TYPE / COMP_SWAP -> render target format and the conversion of the
// shader's export to it (see vpgcn::ExportConv): normalised / float formats are converted by the
// host, the rest are raw integer containers the shader packs into.
struct ColorPlan {
    vpgpu::Format format = vpgpu::Format::Undefined;
    vpgcn::ExportConv conv;
    bool exact = true;
    const char* why = nullptr;
};
ColorPlan colorPlan(uint8_t format, uint8_t numberType, uint8_t compSwap);
vpgpu::Format colorFormat(uint8_t format, uint8_t numberType, uint8_t compSwap);

// ---- Depth / stencil ------------------------------------------------------------------------------------
//
// DB_Z_INFO.FORMAT and DB_STENCIL_INFO.FORMAT -> depth/stencil format. Apple GPUs have no 16-bit
// depth with stencil nor 24-bit depth: Z16+S8 and Z24 use the 32-bit float formats (more
// precision than the hardware; `exact` false).
vpgpu::Format depthFormat(uint8_t zFormat, uint8_t stencilFormat, bool* exact = nullptr);

// CB_BLENDn_CONTROL encodings; `exact` is cleared when the factor has no gpu/core equivalent
// (undefined encodings) and Zero is used.
vpgpu::BlendFactor blendFactor(uint8_t gcn, bool& exact);
vpgpu::BlendOp blendOp(uint8_t gcn);
// FRAG_* (depth) and REF_* (stencil) functions share the gpu/core order.
vpgpu::CompareOp compareOp(uint8_t gcn);
vpgpu::StencilOp stencilOp(uint8_t gcn, bool& exact);

// An image descriptor (T#, 8 dwords; a 4-dword one has the first four only).
struct ImageDesc {
    uint64_t address = 0;
    uint8_t dataFormat = 0, numFormat = 0;
    uint32_t width = 1, height = 1, depth = 1;  // depth: 3D depth or array layers (with cubes: faces)
    uint32_t pitch = 0;
    uint8_t baseLevel = 0, lastLevel = 0;
    uint8_t tilingIndex = 0;
    bool pow2Pad = false;                        // POW2_PAD: mip sizes and array layers rounded up to powers of two
    uint8_t type = 9;                            // SQ_RSRC_IMG_*: 8 1D, 9 2D, 10 3D, 11 cube, 12 1D array, 13 2D array, 14 2D MSAA, 15 2D MSAA array
    uint16_t baseArray = 0, lastArray = 0;
    std::array<uint8_t, 4> dstSel{4, 5, 6, 7};   // SQ_SEL_X..W = 4..7, 0 = 0, 1 = 1
};
ImageDesc decodeImage(std::span<const uint32_t> dwords);

// A buffer descriptor (V#, 4 dwords).
struct BufferDesc {
    uint64_t address = 0;
    uint32_t stride = 0, records = 0;
    uint8_t dataFormat = 0, numFormat = 0;
    bool swizzle = false, addTid = false;  // V#.swizzle_enable, ADD_TID_ENABLE
    uint64_t bytes() const { return stride ? uint64_t(stride) * records : records; }
};
BufferDesc decodeBuffer(std::span<const uint32_t> dwords);

// A sampler descriptor (S#, 4 dwords).
vpgpu::SamplerDesc decodeSampler(std::span<const uint32_t> dwords);

// Linear tiling indices (TILE_MODE_INDEX: linear aligned 8, linear general 31): no detiling needed.
inline bool linearTiling(uint8_t index) { return index == 8 || index == 31; }

}  // namespace vpps4
