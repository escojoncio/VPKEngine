#include "vpps4/formats.h"

#include <algorithm>

namespace vpps4 {

using vpgpu::Format;

namespace {
// Number formats of CB and image descriptors (same encoding for the shared values).
enum : uint8_t { Unorm = 0, Snorm = 1, Uscaled = 2, Sscaled = 3, Uint = 4, Sint = 5, SnormNz = 6, Float = 7, Srgb = 9 };
}  // namespace

static Format byLayout(uint8_t layout, uint8_t num, bool bgra) {
    switch (layout) {
        case 1:  // 8
            return num == Unorm ? Format::R8Unorm : num == Snorm ? Format::R8Snorm : num == Uint ? Format::R8Uint
                 : num == Sint  ? Format::R8Sint : Format::Undefined;
        case 2:  // 16
            return num == Unorm ? Format::R16Unorm : num == Snorm ? Format::R16Snorm : num == Uint ? Format::R16Uint
                 : num == Sint  ? Format::R16Sint : num == Float ? Format::R16Float : Format::Undefined;
        case 3:  // 8_8
            return num == Unorm ? Format::RG8Unorm : num == Snorm ? Format::RG8Snorm : num == Uint ? Format::RG8Uint
                 : num == Sint  ? Format::RG8Sint : Format::Undefined;
        case 4:  // 32
            return num == Uint ? Format::R32Uint : num == Sint ? Format::R32Sint : num == Float ? Format::R32Float
                                                                                               : Format::Undefined;
        case 5:  // 16_16
            return num == Unorm ? Format::RG16Unorm : num == Snorm ? Format::RG16Snorm : num == Uint ? Format::RG16Uint
                 : num == Sint  ? Format::RG16Sint : num == Float ? Format::RG16Float : Format::Undefined;
        case 6:  // 10_11_11 (B10G11R11: red in the low 11 bits)
            return num == Float ? Format::RG11B10Float : Format::Undefined;
        case 9:  // 2_10_10_10 (A2B10G10R10)
            return num == Unorm ? Format::RGB10A2Unorm : num == Uint ? Format::RGB10A2Uint : Format::Undefined;
        case 10:  // 8_8_8_8
            if (num == Unorm) return bgra ? Format::BGRA8Unorm : Format::RGBA8Unorm;
            if (num == Srgb) return bgra ? Format::BGRA8Srgb : Format::RGBA8Srgb;
            return num == Snorm ? Format::RGBA8Snorm : num == Uint ? Format::RGBA8Uint : num == Sint ? Format::RGBA8Sint
                                                                                                     : Format::Undefined;
        case 11:  // 32_32
            return num == Uint ? Format::RG32Uint : num == Sint ? Format::RG32Sint : num == Float ? Format::RG32Float
                                                                                               : Format::Undefined;
        case 12:  // 16_16_16_16
            return num == Unorm ? Format::RGBA16Unorm : num == Snorm ? Format::RGBA16Snorm : num == Uint ? Format::RGBA16Uint
                 : num == Sint  ? Format::RGBA16Sint : num == Float ? Format::RGBA16Float : Format::Undefined;
        case 14:  // 32_32_32_32
            return num == Uint ? Format::RGBA32Uint : num == Sint ? Format::RGBA32Sint : num == Float ? Format::RGBA32Float
                                                                                                   : Format::Undefined;
        case 16: return num == Unorm ? Format::B5G6R5Unorm : Format::Undefined;   // 5_6_5
        case 17:
        case 18: return num == Unorm ? Format::BGR5A1Unorm : Format::Undefined;   // 1_5_5_5 / 5_5_5_1
        case 19: return num == Unorm ? Format::ABGR4Unorm : Format::Undefined;    // 4_4_4_4
        default: return Format::Undefined;
    }
}

Format colorFormat(uint8_t format, uint8_t numberType, uint8_t compSwap) {
    // COMP_SWAP 1 (ALT) stores 4-component formats as BGRA.
    return byLayout(format, numberType, compSwap == 1);
}

Format depthFormat(uint8_t zFormat, uint8_t stencilFormat) {
    const bool stencil = stencilFormat != 0;
    switch (zFormat) {
        case 1: return stencil ? Format::D32FloatS8 : Format::D16Unorm;  // (no D16S8 on Apple GPUs)
        case 2:
        case 3: return stencil ? Format::D32FloatS8 : Format::D32Float;  // Z_24 as 32-bit float
        default: return stencil ? Format::S8Uint : Format::Undefined;
    }
}

Format textureFormat(uint8_t dataFormat, uint8_t numFormat) {
    const bool srgb = numFormat == Srgb;
    const bool snorm = numFormat == Snorm || numFormat == SnormNz;
    switch (dataFormat) {
        case 34: return Format::RGB9E5Float;                                    // 5_9_9_9
        case 35: return srgb ? Format::BC1Srgb : Format::BC1;
        case 36: return srgb ? Format::BC2Srgb : Format::BC2;
        case 37: return srgb ? Format::BC3Srgb : Format::BC3;
        case 38: return snorm ? Format::BC4Snorm : Format::BC4Unorm;
        case 39: return snorm ? Format::BC5Snorm : Format::BC5Unorm;
        case 40: return snorm ? Format::BC6HSfloat : Format::BC6HUfloat;
        case 41: return srgb ? Format::BC7Srgb : Format::BC7;
        default:
            // Scaled reads return the integer values as floats: the texture is the integer format's
            // float-sampleable neighbour where one exists.
            return byLayout(dataFormat, numFormat == Uscaled ? uint8_t(Unorm) : numFormat == Sscaled ? uint8_t(Snorm)
                                        : numFormat == SnormNz ? uint8_t(Snorm) : numFormat, false);
    }
}

vpgpu::BlendFactor blendFactor(uint8_t gcn, bool& exact) {
    using F = vpgpu::BlendFactor;
    switch (gcn) {
        case 0: return F::Zero;
        case 1: return F::One;
        case 2: return F::SrcColor;
        case 3: return F::OneMinusSrcColor;
        case 4: return F::SrcAlpha;
        case 5: return F::OneMinusSrcAlpha;
        case 6: return F::DstAlpha;
        case 7: return F::OneMinusDstAlpha;
        case 8: return F::DstColor;
        case 9: return F::OneMinusDstColor;
        case 13: return F::Constant;          // CONSTANT_COLOR
        case 14: return F::OneMinusConstant;  // ONE_MINUS_CONSTANT_COLOR
        case 19: return F::Constant;          // CONSTANT_ALPHA (the constant's alpha: approximated)
        case 20: return F::OneMinusConstant;
        default: exact = false; return gcn == 10 ? F::One : F::Zero;  // SRC_ALPHA_SATURATE, SRC1_* (dual source)
    }
}

vpgpu::BlendOp blendOp(uint8_t gcn) {
    using O = vpgpu::BlendOp;
    switch (gcn) {
        case 1: return O::Subtract;         // SRC_MINUS_DST
        case 2: return O::Min;
        case 3: return O::Max;
        case 4: return O::ReverseSubtract;  // DST_MINUS_SRC
        default: return O::Add;             // DST_PLUS_SRC
    }
}

vpgpu::CompareOp compareOp(uint8_t gcn) { return static_cast<vpgpu::CompareOp>(gcn & 7); }

vpgpu::StencilOp stencilOp(uint8_t gcn, bool& exact) {
    using S = vpgpu::StencilOp;
    switch (gcn) {
        case 0: return S::Keep;
        case 1: return S::Zero;
        case 3: return S::Replace;          // REPLACE_TEST (the reference value)
        case 5: return S::IncrementClamp;   // ADD_CLAMP
        case 6: return S::DecrementClamp;   // SUB_CLAMP
        case 7: return S::Invert;
        case 8: return S::IncrementWrap;
        case 9: return S::DecrementWrap;
        default: exact = false; return S::Keep;  // ONES, REPLACE_OP, logic ops
    }
}

static uint32_t bits(uint32_t v, unsigned lo, unsigned n) { return (v >> lo) & ((1u << n) - 1); }

ImageDesc decodeImage(std::span<const uint32_t> d) {
    ImageDesc r;
    if (d.size() < 4) return r;
    r.address = (uint64_t(d[0]) | uint64_t(bits(d[1], 0, 6)) << 32) << 8;
    r.dataFormat = uint8_t(bits(d[1], 20, 6));
    r.numFormat = uint8_t(bits(d[1], 26, 4));
    r.width = bits(d[2], 0, 14) + 1;
    r.height = bits(d[2], 14, 14) + 1;
    for (unsigned k = 0; k < 4; ++k) r.dstSel[k] = uint8_t(bits(d[3], 3 * k, 3));
    r.baseLevel = uint8_t(bits(d[3], 12, 4));
    r.lastLevel = uint8_t(bits(d[3], 16, 4));
    r.tilingIndex = uint8_t(bits(d[3], 20, 5));
    r.type = uint8_t(bits(d[3], 28, 4));
    if (d.size() >= 6) {
        r.depth = bits(d[4], 0, 13) + 1;
        r.pitch = bits(d[4], 13, 14) + 1;
        r.baseArray = uint16_t(bits(d[5], 0, 13));
        r.lastArray = uint16_t(bits(d[5], 13, 13));
    } else {
        r.pitch = r.width;
    }
    return r;
}

BufferDesc decodeBuffer(std::span<const uint32_t> d) {
    BufferDesc r;
    if (d.size() < 4) return r;
    r.address = uint64_t(d[0]) | uint64_t(bits(d[1], 0, 16)) << 32;
    r.stride = bits(d[1], 16, 14);
    r.records = d[2];
    r.numFormat = uint8_t(bits(d[3], 12, 3));
    r.dataFormat = uint8_t(bits(d[3], 15, 4));
    return r;
}

vpgpu::SamplerDesc decodeSampler(std::span<const uint32_t> d) {
    vpgpu::SamplerDesc s;
    if (d.size() < 4) return s;
    auto clamp = [](uint32_t c) {
        using A = vpgpu::AddressMode;
        switch (c) {
            case 0: return A::Repeat;
            case 1: return A::MirrorRepeat;
            case 2: return A::ClampToEdge;          // CLAMP_LAST_TEXEL
            case 3: return A::MirrorClampToEdge;    // MIRROR_ONCE_LAST_TEXEL
            case 6: return A::ClampToBorder;        // CLAMP_BORDER
            default: return A::ClampToEdge;         // half-border modes: nearest Metal one
        }
    };
    s.u = clamp(bits(d[0], 0, 3));
    s.v = clamp(bits(d[0], 3, 3));
    s.w = clamp(bits(d[0], 6, 3));
    s.maxAnisotropy = uint8_t(1u << std::min<uint32_t>(bits(d[0], 9, 3), 4));
    const uint32_t cmp = bits(d[0], 12, 3);
    if (cmp != 0) s.compare = static_cast<vpgpu::CompareOp>(cmp);
    s.lodMin = float(bits(d[1], 0, 12)) / 256.0f;
    s.lodMax = float(bits(d[1], 12, 12)) / 256.0f;
    const uint32_t mag = bits(d[2], 20, 2), min = bits(d[2], 22, 2), mip = bits(d[2], 26, 2);
    s.magFilter = (mag & 1) ? vpgpu::Filter::Linear : vpgpu::Filter::Nearest;
    s.minFilter = (min & 1) ? vpgpu::Filter::Linear : vpgpu::Filter::Nearest;
    if (min < 2 && mag < 2) s.maxAnisotropy = 1;  // (aniso only with the ANISO_* filters)
    s.mipFilter = mip == 0 ? vpgpu::MipFilter::None : mip == 1 ? vpgpu::MipFilter::Nearest : vpgpu::MipFilter::Linear;
    const uint32_t border = bits(d[3], 30, 2);
    s.border = border == 1 ? vpgpu::BorderColor::OpaqueBlack : border == 2 ? vpgpu::BorderColor::OpaqueWhite
                                                                           : vpgpu::BorderColor::TransparentBlack;
    return s;
}

}  // namespace vpps4
