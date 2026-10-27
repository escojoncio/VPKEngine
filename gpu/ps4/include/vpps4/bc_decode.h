// CPU decoder of the block-compressed texture formats (BC1-BC7, the GCN data formats 35-41) for hosts without
// native support (visionOS): decoded to a plain host texture when the guest memory is uploaded.
// Written from the public BC1-BC7 format specifications (Microsoft Direct3D 10/11 texture compression docs).
#pragma once

#include <cstddef>
#include <cstdint>

#include "vpgpu/core/resources.h"

namespace vpps4::bc {

enum class Kind : uint8_t { BC1, BC2, BC3, BC4, BC5, BC6H, BC7 };

// GCN IMG_DATA_FORMAT 35..41 (BC1, BC2, BC3, BC4, BC5, BC6, BC7) -> kind; false for any other format.
bool kindOf(uint8_t dataFormat, Kind& out);
// The host texture a decoded format becomes: BC1/2/3/7 RGBA8 (sRGB when `srgb`), BC4 R8, BC5 RG8 (snorm when
// `isSigned`), BC6H RGBA16Float.
vpgpu::Format hostFormat(Kind, bool srgb, bool isSigned);
// The block-compressed format the guest memory holds (for the layout: 4x4 texels per block of 8 or 16 bytes).
vpgpu::Format layoutFormat(Kind, bool srgb, bool isSigned);
unsigned texelBytes(Kind);  // bytes per decoded texel

// One 4x4 block (8 or 16 bytes) -> 16 texels, row-major, texelBytes(kind) each. `isSigned`: BC4 / BC5 snorm, BC6H signed.
void decodeBlock(Kind, bool isSigned, const uint8_t* block, uint8_t* texels);
// A level of `width` x `height` texels: blocks are read from `src` (rows `srcRowBytes` apart, ceil(width / 4) blocks
// each) and the texels written to `dst` (rows `dstRowBytes` apart); a block that crosses the right or bottom edge
// contributes only the texels inside.
void decodeImage(Kind, bool isSigned, const uint8_t* src, size_t srcRowBytes, uint32_t width, uint32_t height, uint8_t* dst,
                 size_t dstRowBytes);

}  // namespace vpps4::bc
