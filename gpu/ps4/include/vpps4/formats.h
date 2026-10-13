// Liverpool register encodings -> gpu/core enums. Written from AMD's public register and ISA
// documentation (CB/DB/SQ formats, blend and compare encodings, sampler and image descriptors).
#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "vpgpu/core/commands.h"
#include "vpgpu/core/pipeline_cache.h"
#include "vpgpu/core/resources.h"

namespace vpps4 {

// CB_COLORn_INFO FORMAT / NUMBER_TYPE / COMP_SWAP -> render target format (Undefined: no Metal one).
vpgpu::Format colorFormat(uint8_t format, uint8_t numberType, uint8_t compSwap);
// DB_Z_INFO.FORMAT and DB_STENCIL_INFO.FORMAT -> depth/stencil format.
vpgpu::Format depthFormat(uint8_t zFormat, uint8_t stencilFormat);
// Image descriptor IMG_DATA_FORMAT / IMG_NUM_FORMAT -> sampled texture format.
vpgpu::Format textureFormat(uint8_t dataFormat, uint8_t numFormat);

// CB_BLENDn_CONTROL encodings; `exact` is cleared when the factor has no gpu/core equivalent
// (source-alpha saturate, dual-source) and an approximation is used.
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
    uint64_t bytes() const { return stride ? uint64_t(stride) * records : records; }
};
BufferDesc decodeBuffer(std::span<const uint32_t> dwords);

// A sampler descriptor (S#, 4 dwords).
vpgpu::SamplerDesc decodeSampler(std::span<const uint32_t> dwords);

// Linear tiling indices (TILE_MODE_INDEX: linear aligned 8, linear general 31): no detiling needed.
inline bool linearTiling(uint8_t index) { return index == 8 || index == 31; }

}  // namespace vpps4
