#include "vpps4/formats.h"

#include <algorithm>

namespace vpps4 {

using vp4gpu::Format;

namespace {

FormatCaps gCaps;

// Metal formats per number class of a plain layout (Undefined: none of that class).
struct Row {
    Format unorm, snorm, uint, sint, flt, srgb;
    uint8_t compBits;  // width of one component (scaled formats rescale a normalised one)
};
const Row* rowFor(unsigned dfmt) {
    using F = Format;
    static const Row r8{F::R8Unorm, F::R8Snorm, F::R8Uint, F::R8Sint, F::Undefined, F::Undefined, 8};
    static const Row r88{F::RG8Unorm, F::RG8Snorm, F::RG8Uint, F::RG8Sint, F::Undefined, F::Undefined, 8};
    static const Row r8888{F::RGBA8Unorm, F::RGBA8Snorm, F::RGBA8Uint, F::RGBA8Sint, F::Undefined, F::RGBA8Srgb, 8};
    static const Row r16{F::R16Unorm, F::R16Snorm, F::R16Uint, F::R16Sint, F::R16Float, F::Undefined, 16};
    static const Row r1616{F::RG16Unorm, F::RG16Snorm, F::RG16Uint, F::RG16Sint, F::RG16Float, F::Undefined, 16};
    static const Row r16x4{F::RGBA16Unorm, F::RGBA16Snorm, F::RGBA16Uint, F::RGBA16Sint, F::RGBA16Float, F::Undefined, 16};
    static const Row r32{F::Undefined, F::Undefined, F::R32Uint, F::R32Sint, F::R32Float, F::Undefined, 32};
    static const Row r3232{F::Undefined, F::Undefined, F::RG32Uint, F::RG32Sint, F::RG32Float, F::Undefined, 32};
    static const Row r32x4{F::Undefined, F::Undefined, F::RGBA32Uint, F::RGBA32Sint, F::RGBA32Float, F::Undefined, 32};
    switch (dfmt) {
        case 1: return &r8;
        case 3: return &r88;
        case 10: return &r8888;
        case 2: return &r16;
        case 5: return &r1616;
        case 12: return &r16x4;
        case 4: return &r32;
        case 11: return &r3232;
        case 14: return &r32x4;
        default: return nullptr;
    }
}

Format containerFor(unsigned bytes) {
    switch (bytes) {
        case 1: return Format::R8Uint;
        case 2: return Format::R16Uint;
        case 4: return Format::R32Uint;
        case 8: return Format::RG32Uint;
        case 16: return Format::RGBA32Uint;
        default: return Format::Undefined;
    }
}

bool packedHostFormat(Format f) {
    return f == Format::RGB10A2Unorm || f == Format::RGB10A2Uint || f == Format::RG11B10Float || f == Format::RGB9E5Float ||
           f == Format::B5G6R5Unorm || f == Format::BGR5A1Unorm || f == Format::ABGR4Unorm;
}

Format unormOfSrgb(Format f) {
    switch (f) {
        case Format::RGBA8Srgb: return Format::RGBA8Unorm;
        case Format::BGRA8Srgb: return Format::BGRA8Unorm;
        default: return f;
    }
}

// The raw-container plan of a data format: the shader decodes (conv.manual). False if there is none.
bool containerPlan(unsigned dfmt, unsigned nfmt, TexturePlan& p) {
    const vpgcn::FormatLayout& l = vpgcn::layoutOf(dfmt);
    if (l.kind != vpgcn::FormatLayout::Plain && l.kind != vpgcn::FormatLayout::Subsampled && l.kind != vpgcn::FormatLayout::OneBit) return false;
    uint32_t probe[4] = {}, out[4];
    if (!vpgcn::decodeTexel(dfmt, nfmt, probe, 0, out)) return false;
    const Format f = containerFor(l.bytes);
    if (f == Format::Undefined) return false;
    p = TexturePlan{};
    p.format = f;
    p.elem = TexturePlan::Uint;
    p.conv.manual = true;
    p.conv.texelsPerElement = l.texelsPerElement;
    p.widthDivisor = l.texelsPerElement;
    return true;
}

void setSelect(vpgcn::TextureConv& c, uint8_t a, uint8_t b, uint8_t d, uint8_t e) {
    c.select[0] = a, c.select[1] = b, c.select[2] = d, c.select[3] = e;
}

// A host format that holds the texels of the layout with the number format, plus the conversion.
bool nativeTexturePlan(unsigned dfmt, unsigned nfmt, TexturePlan& p) {
    using F = Format;
    const vpgcn::FormatLayout& l = vpgcn::layoutOf(dfmt);
    p = TexturePlan{};
    if (dfmt == 34) {
        p.format = F::RGB9E5Float;
        return true;
    }
    if (l.kind != vpgcn::FormatLayout::Plain) return false;
    if (const Row* r = rowFor(dfmt)) {
        const unsigned w = r->compBits;
        auto affine = [&](float scale, float bias, unsigned comps) {
            p.conv.affine = true;
            for (unsigned k = 0; k < comps; ++k) p.conv.scale[k] = scale, p.conv.bias[k] = bias;
        };
        switch (nfmt) {
            case vpgcn::kUnorm: p.format = r->unorm; break;
            case vpgcn::kSnorm: p.format = r->snorm; break;
            case vpgcn::kUint: p.format = r->uint, p.elem = TexturePlan::Uint; break;
            case vpgcn::kSint: p.format = r->sint, p.elem = TexturePlan::Sint; break;
            case vpgcn::kFloat: p.format = r->flt; break;
            case vpgcn::kSrgb:
                if (r->srgb != F::Undefined) p.format = r->srgb;
                else p.format = r->unorm, p.conv.srgb = p.format != F::Undefined;
                break;
            case vpgcn::kUscaled:
                p.format = r->unorm;
                if (w < 32) affine(float((1u << w) - 1), 0.0f, l.comps);
                break;
            case vpgcn::kSscaled:
                p.format = r->snorm;
                if (w < 32) affine(float((1u << (w - 1)) - 1), 0.0f, l.comps);
                break;
            case vpgcn::kSnormNz: {
                p.format = r->snorm;
                const float maxU = float((1u << w) - 1), maxS = float((1u << (w - 1)) - 1);
                if (w < 32) affine(2.0f * maxS / maxU, 1.0f / maxU, l.comps);
                break;
            }
            case vpgcn::kUbnorm:
                p.format = r->unorm;
                affine(2.0f, -1.0f, l.comps);
                break;
            default: return false;
        }
        return p.format != F::Undefined;
    }
    switch (dfmt) {
        case 6:  // 10_11_11: B10G11R11, red in the low 11 bits
            if (nfmt == vpgcn::kFloat) return p.format = F::RG11B10Float, true;
            return false;
        case 9:  // 2_10_10_10: A2B10G10R10
            switch (nfmt) {
                case vpgcn::kUnorm: p.format = F::RGB10A2Unorm; return true;
                case vpgcn::kUint: p.format = F::RGB10A2Uint, p.elem = TexturePlan::Uint; return true;
                case vpgcn::kSrgb: p.format = F::RGB10A2Unorm, p.conv.srgb = true; return true;
                case vpgcn::kUscaled:
                    p.format = F::RGB10A2Unorm;
                    p.conv.affine = true;
                    p.conv.scale[0] = p.conv.scale[1] = p.conv.scale[2] = 1023.0f, p.conv.scale[3] = 3.0f;
                    return true;
                default: return false;
            }
        // Packed 16-bit formats: Metal names list from the least significant bit, the hardware
        // lists from the most significant, so channels come reversed.
        case 16: if (nfmt != vpgcn::kUnorm) return false; p.format = F::B5G6R5Unorm, setSelect(p.conv, 6, 5, 4, 7); return true;
        case 17: if (nfmt != vpgcn::kUnorm) return false; p.format = F::BGR5A1Unorm, setSelect(p.conv, 6, 5, 4, 7); return true;
        case 19: if (nfmt != vpgcn::kUnorm) return false; p.format = F::ABGR4Unorm, setSelect(p.conv, 7, 6, 5, 4); return true;
        default: return false;
    }
}

}  // namespace

void setFormatCaps(const FormatCaps& c) { gCaps = c; }
const FormatCaps& formatCaps() { return gCaps; }

TexturePlan texturePlan(uint8_t dfmt, uint8_t nfmt, bool storage) {
    using F = Format;
    TexturePlan p;
    const vpgcn::FormatLayout& l = vpgcn::layoutOf(dfmt);
    switch (l.kind) {
        case vpgcn::FormatLayout::Block: {
            if (!gCaps.bc) {
                p.why = "block-compressed format without host support";
                return p;
            }
            const bool srgb = nfmt == vpgcn::kSrgb, snorm = nfmt == vpgcn::kSnorm || nfmt == vpgcn::kSnormNz;
            switch (dfmt) {
                case 35: p.format = srgb ? F::BC1Srgb : F::BC1; break;
                case 36: p.format = srgb ? F::BC2Srgb : F::BC2; break;
                case 37: p.format = srgb ? F::BC3Srgb : F::BC3; break;
                case 38: p.format = snorm ? F::BC4Snorm : F::BC4Unorm; break;
                case 39: p.format = snorm ? F::BC5Snorm : F::BC5Unorm; break;
                case 40: p.format = snorm ? F::BC6HSfloat : F::BC6HUfloat; break;
                default: p.format = srgb ? F::BC7Srgb : F::BC7; break;
            }
            if (storage) p.format = F::Undefined;
            return p;
        }
        case vpgcn::FormatLayout::Fmask:
            p.format = containerFor(l.bytes);
            p.elem = TexturePlan::Uint;
            return p;
        case vpgcn::FormatLayout::None:
            p.why = dfmt == 0 ? "invalid data format" : "reserved data format";
            return p;
        default: break;
    }
    if (nfmt > vpgcn::kFloat && nfmt != vpgcn::kSrgb && nfmt != vpgcn::kUbnorm) {
        p.why = "number format without a defined conversion";
        return p;
    }
    if (!(l.kind == vpgcn::FormatLayout::Plain && dfmt != 13) && l.kind != vpgcn::FormatLayout::SharedExp &&
        l.kind != vpgcn::FormatLayout::Subsampled && l.kind != vpgcn::FormatLayout::OneBit) {
        p.why = "format without texture support";
        return p;
    }
    TexturePlan n;
    const bool haveNative = l.kind != vpgcn::FormatLayout::Subsampled && l.kind != vpgcn::FormatLayout::OneBit && dfmt != 13 &&
                            nativeTexturePlan(dfmt, nfmt, n);
    if (haveNative && storage) {
        // Shaders write these through the container instead of a format Metal cannot store to or
        // that needs a conversion on the way out.
        if (packedHostFormat(n.format) || n.conv.affine ||
            !(n.conv.select[0] == 4 && n.conv.select[1] == 5 && n.conv.select[2] == 6 && n.conv.select[3] == 7))
            return containerPlan(dfmt, nfmt, p) ? p : TexturePlan{};
        if (n.format == F::RGBA8Srgb || n.format == F::BGRA8Srgb) {
            n.format = unormOfSrgb(n.format);
            n.conv.srgb = true;
        }
    }
    if (haveNative) return n;
    if (containerPlan(dfmt, nfmt, p)) return p;
    p = TexturePlan{};
    p.why = "no host format and no shader decode for this format";
    return p;
}

TexturePlan containerTexturePlan(uint8_t dfmt, uint8_t nfmt) {
    TexturePlan p;
    if (!containerPlan(dfmt, nfmt, p)) p = TexturePlan{};
    return p;
}

Format textureFormat(uint8_t dataFormat, uint8_t numFormat) { return texturePlan(dataFormat, numFormat).format; }

void composeSelect(const std::array<uint8_t, 4>& dstSel, const uint8_t planSelect[4], uint8_t out[4]) {
    for (unsigned k = 0; k < 4; ++k) {
        const uint8_t s = dstSel[k];
        // SQ_SEL: 0 zero, 1 one, 4..7 x..w (2, 3 reserved: read as zero)
        out[k] = s >= 4 && s <= 7 ? planSelect[s - 4] : s == 1 ? uint8_t(1) : uint8_t(0);
    }
}

namespace {

// CB COMP_SWAP: the export component that feeds each storage channel of a format with `n` channels.
void swapTable(unsigned n, unsigned swap, uint8_t out[4], bool& exact) {
    static const uint8_t tables[4][5][4] = {
        // STD
        {{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}},
        // ALT (BGRA)
        {{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 3, 2, 3}, {2, 1, 0, 3}, {2, 1, 0, 3}},
        // STD_REV (ABGR)
        {{0, 1, 2, 3}, {0, 1, 2, 3}, {1, 0, 2, 3}, {2, 1, 0, 3}, {3, 2, 1, 0}},
        // ALT_REV (ARGB)
        {{0, 1, 2, 3}, {3, 1, 2, 3}, {3, 0, 2, 3}, {0, 1, 2, 3}, {3, 0, 1, 2}},
    };
    swap &= 3;
    for (unsigned k = 0; k < 4; ++k) out[k] = tables[swap][std::min(n, 4u)][k];
    if (swap == 3 && n == 3) exact = false;  // ALT_REV on three channels is not documented: kept in order
}

}  // namespace

ColorPlan colorPlan(uint8_t format, uint8_t numberType, uint8_t compSwap) {
    using F = Format;
    ColorPlan p;
    const unsigned nfmt = numberType == 6 ? unsigned(vpgcn::kSrgb) : unsigned(numberType);
    const vpgcn::FormatLayout& l = vpgcn::layoutOf(format);
    if (l.kind != vpgcn::FormatLayout::Plain || format == 13) {
        p.why = "colour format without a render target";
        return p;
    }
    if (nfmt > vpgcn::kFloat && nfmt != vpgcn::kSrgb) {
        p.why = "colour number type without a defined conversion";
        return p;
    }
    uint8_t swap[4];
    bool exact = true;
    swapTable(l.comps, compSwap, swap, exact);
    p.exact = exact;
    if (!exact) p.why = "colour component swap approximated";
    // Host channel j takes export component `final[j]`.
    uint8_t finalSel[4] = {swap[0], swap[1], swap[2], swap[3]};
    auto fieldBits = [&](vpgcn::ExportConv& c) {
        for (unsigned k = 0; k < l.comps; ++k) c.bits[k] = l.bits[k];
    };
    auto container = [&]() {
        p.format = containerFor(l.bytes);
        p.conv = vpgcn::ExportConv{};
        if (p.format == F::Undefined) {
            p.why = "colour format without a render target";
            return;
        }
        p.conv.kind = vpgcn::ExportConv::Packed;
        p.conv.dfmt = format;
        p.conv.nfmt = uint8_t(nfmt);
        for (unsigned k = 0; k < 4; ++k) p.conv.select[k] = swap[k];
    };
    if (const Row* r = rowFor(format)) {
        F f = F::Undefined;
        bool srgbShader = false;
        switch (nfmt) {
            case vpgcn::kUnorm: f = r->unorm; break;
            case vpgcn::kSnorm: f = r->snorm; break;
            case vpgcn::kUint: f = r->uint; break;
            case vpgcn::kSint: f = r->sint; break;
            case vpgcn::kFloat: f = r->flt; break;
            case vpgcn::kSrgb:
                if (r->srgb != F::Undefined) f = r->srgb;
                else f = r->unorm, srgbShader = f != F::Undefined;
                break;
            default: break;  // scaled / SNORM_NZ: the integer container
        }
        if (f == F::Undefined) {
            container();
            return p;
        }
        // The BGRA formats swap in the host: the channel order the present and CPU expect.
        if (format == 10 && compSwap == 1 && (f == F::RGBA8Unorm || f == F::RGBA8Srgb)) {
            f = f == F::RGBA8Unorm ? F::BGRA8Unorm : F::BGRA8Srgb;
            finalSel[0] = 0, finalSel[1] = 1, finalSel[2] = 2, finalSel[3] = 3;
        }
        p.format = f;
        for (unsigned k = 0; k < 4; ++k) p.conv.select[k] = finalSel[k];
        p.conv.srgb = srgbShader;
        if (nfmt == vpgcn::kUint) p.conv.kind = vpgcn::ExportConv::Uint, fieldBits(p.conv);
        if (nfmt == vpgcn::kSint) p.conv.kind = vpgcn::ExportConv::Sint, fieldBits(p.conv);
        return p;
    }
    switch (format) {
        case 6:
            if (nfmt == vpgcn::kFloat) {
                p.format = F::RG11B10Float;
                for (unsigned k = 0; k < 4; ++k) p.conv.select[k] = finalSel[k];
                return p;
            }
            break;
        case 9:
            if (nfmt == vpgcn::kUnorm || nfmt == vpgcn::kSrgb) {
                p.format = F::RGB10A2Unorm;
                p.conv.srgb = nfmt == vpgcn::kSrgb;
                for (unsigned k = 0; k < 4; ++k) p.conv.select[k] = finalSel[k];
                return p;
            }
            if (nfmt == vpgcn::kUint) {
                p.format = F::RGB10A2Uint;
                p.conv.kind = vpgcn::ExportConv::Uint;
                fieldBits(p.conv);
                for (unsigned k = 0; k < 4; ++k) p.conv.select[k] = finalSel[k];
                return p;
            }
            break;
        case 16:
        case 17:
        case 19: {
            if (nfmt != vpgcn::kUnorm) break;
            // Metal channel j holds hardware field perm[j] (see nativeTexturePlan).
            static const uint8_t perm565[4] = {2, 1, 0, 3}, perm4444[4] = {3, 2, 1, 0};
            const uint8_t* perm = format == 19 ? perm4444 : perm565;
            p.format = format == 16 ? F::B5G6R5Unorm : format == 17 ? F::BGR5A1Unorm : F::ABGR4Unorm;
            for (unsigned j = 0; j < 4; ++j) p.conv.select[j] = swap[perm[j]];
            return p;
        }
        default: break;
    }
    container();
    return p;
}

Format colorFormat(uint8_t format, uint8_t numberType, uint8_t compSwap) { return colorPlan(format, numberType, compSwap).format; }

Format depthFormat(uint8_t zFormat, uint8_t stencilFormat, bool* exact) {
    const bool stencil = stencilFormat != 0;
    if (exact) *exact = true;
    switch (zFormat) {
        case 1:  // Z_16
            if (stencil) {
                if (exact) *exact = false;  // no D16S8 on Apple GPUs: 32-bit float depth
                return Format::D32FloatS8;
            }
            return Format::D16Unorm;
        case 2:  // Z_24
        case 3:  // Z_32_FLOAT
            if (zFormat == 2 && exact) *exact = false;  // 24-bit depth held as 32-bit float
            return stencil ? Format::D32FloatS8 : Format::D32Float;
        default: return stencil ? Format::S8Uint : Format::Undefined;
    }
}

vp4gpu::BlendFactor blendFactor(uint8_t gcn, bool& exact) {
    using F = vp4gpu::BlendFactor;
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

vp4gpu::BlendOp blendOp(uint8_t gcn) {
    using O = vp4gpu::BlendOp;
    switch (gcn) {
        case 1: return O::Subtract;         // SRC_MINUS_DST
        case 2: return O::Min;
        case 3: return O::Max;
        case 4: return O::ReverseSubtract;  // DST_MINUS_SRC
        default: return O::Add;             // DST_PLUS_SRC
    }
}

vp4gpu::CompareOp compareOp(uint8_t gcn) { return static_cast<vp4gpu::CompareOp>(gcn & 7); }

vp4gpu::StencilOp stencilOp(uint8_t gcn, bool& exact) {
    using S = vp4gpu::StencilOp;
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
    r.pow2Pad = bits(d[3], 25, 1) != 0;
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

vp4gpu::SamplerDesc decodeSampler(std::span<const uint32_t> d) {
    vp4gpu::SamplerDesc s;
    if (d.size() < 4) return s;
    auto clamp = [](uint32_t c) {
        using A = vp4gpu::AddressMode;
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
    if (cmp != 0) s.compare = static_cast<vp4gpu::CompareOp>(cmp);
    s.lodMin = float(bits(d[1], 0, 12)) / 256.0f;
    s.lodMax = float(bits(d[1], 12, 12)) / 256.0f;
    const uint32_t mag = bits(d[2], 20, 2), min = bits(d[2], 22, 2), mip = bits(d[2], 26, 2);
    s.magFilter = (mag & 1) ? vp4gpu::Filter::Linear : vp4gpu::Filter::Nearest;
    s.minFilter = (min & 1) ? vp4gpu::Filter::Linear : vp4gpu::Filter::Nearest;
    if (min < 2 && mag < 2) s.maxAnisotropy = 1;  // (aniso only with the ANISO_* filters)
    s.mipFilter = mip == 0 ? vp4gpu::MipFilter::None : mip == 1 ? vp4gpu::MipFilter::Nearest : vp4gpu::MipFilter::Linear;
    const uint32_t border = bits(d[3], 30, 2);
    s.border = border == 1 ? vp4gpu::BorderColor::OpaqueBlack : border == 2 ? vp4gpu::BorderColor::OpaqueWhite
                                                                           : vp4gpu::BorderColor::TransparentBlack;
    return s;
}

}  // namespace vpps4
