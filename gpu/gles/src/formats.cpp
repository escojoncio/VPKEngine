#include "formats.h"

#include <cmath>
#include <cstring>

namespace vpgles {

namespace {

using namespace gl;
constexpr GLenum UNORM = 0x8C17, SNORM = 0x8F9C;

struct Row {
    GLenum internal;
    VkFormat vk;
    GLenum base;
    uint8_t bytes;
    uint8_t r, g, b, a, d, s;
    GLenum ctype;
    bool renderable;
};

const Row kRows[] = {
    {R8, VK_FORMAT_R8_UNORM, RED, 1, 8, 0, 0, 0, 0, 0, UNORM, true},
    {RG8, VK_FORMAT_R8G8_UNORM, RG, 2, 8, 8, 0, 0, 0, 0, UNORM, true},
    {RGB8, VK_FORMAT_R8G8B8A8_UNORM, RGB, 4, 8, 8, 8, 0, 0, 0, UNORM, true},
    {RGBA8, VK_FORMAT_R8G8B8A8_UNORM, RGBA, 4, 8, 8, 8, 8, 0, 0, UNORM, true},
    {SRGB8, VK_FORMAT_R8G8B8A8_SRGB, RGB, 4, 8, 8, 8, 0, 0, 0, UNORM, false},
    {SRGB8_ALPHA8, VK_FORMAT_R8G8B8A8_SRGB, RGBA, 4, 8, 8, 8, 8, 0, 0, UNORM, true},
    {BGRA8_EXT, VK_FORMAT_B8G8R8A8_UNORM, BGRA_EXT, 4, 8, 8, 8, 8, 0, 0, UNORM, true},
    {RGB565, VK_FORMAT_R5G6B5_UNORM_PACK16, RGB, 2, 5, 6, 5, 0, 0, 0, UNORM, true},
    {RGBA4, VK_FORMAT_R4G4B4A4_UNORM_PACK16, RGBA, 2, 4, 4, 4, 4, 0, 0, UNORM, true},
    {RGB5_A1, VK_FORMAT_A1R5G5B5_UNORM_PACK16, RGBA, 2, 5, 5, 5, 1, 0, 0, UNORM, true},
    {RGB10_A2, VK_FORMAT_A2B10G10R10_UNORM_PACK32, RGBA, 4, 10, 10, 10, 2, 0, 0, UNORM, true},
    {RGB10_A2UI, VK_FORMAT_A2B10G10R10_UINT_PACK32, RGBA_INTEGER, 4, 10, 10, 10, 2, 0, 0, UNSIGNED_INT, true},
    {R8_SNORM, VK_FORMAT_R8_SNORM, RED, 1, 8, 0, 0, 0, 0, 0, SNORM, false},
    {RG8_SNORM, VK_FORMAT_R8G8_SNORM, RG, 2, 8, 8, 0, 0, 0, 0, SNORM, false},
    {RGB8_SNORM, VK_FORMAT_R8G8B8A8_SNORM, RGB, 4, 8, 8, 8, 0, 0, 0, SNORM, false},
    {RGBA8_SNORM, VK_FORMAT_R8G8B8A8_SNORM, RGBA, 4, 8, 8, 8, 8, 0, 0, SNORM, false},
    {R16F, VK_FORMAT_R16_SFLOAT, RED, 2, 16, 0, 0, 0, 0, 0, FLOAT, true},
    {RG16F, VK_FORMAT_R16G16_SFLOAT, RG, 4, 16, 16, 0, 0, 0, 0, FLOAT, true},
    {RGB16F, VK_FORMAT_R16G16B16A16_SFLOAT, RGB, 8, 16, 16, 16, 0, 0, 0, FLOAT, true},
    {RGBA16F, VK_FORMAT_R16G16B16A16_SFLOAT, RGBA, 8, 16, 16, 16, 16, 0, 0, FLOAT, true},
    {R32F, VK_FORMAT_R32_SFLOAT, RED, 4, 32, 0, 0, 0, 0, 0, FLOAT, true},
    {RG32F, VK_FORMAT_R32G32_SFLOAT, RG, 8, 32, 32, 0, 0, 0, 0, FLOAT, true},
    {RGB32F, VK_FORMAT_R32G32B32A32_SFLOAT, RGB, 16, 32, 32, 32, 0, 0, 0, FLOAT, false},
    {RGBA32F, VK_FORMAT_R32G32B32A32_SFLOAT, RGBA, 16, 32, 32, 32, 32, 0, 0, FLOAT, true},
    {R11F_G11F_B10F, VK_FORMAT_B10G11R11_UFLOAT_PACK32, RGB, 4, 11, 11, 10, 0, 0, 0, FLOAT, true},
    {RGB9_E5, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, RGB, 4, 9, 9, 9, 0, 0, 0, FLOAT, false},
    {R16_EXT, VK_FORMAT_R16_UNORM, RED, 2, 16, 0, 0, 0, 0, 0, UNORM, true},
    {RG16_EXT, VK_FORMAT_R16G16_UNORM, RG, 4, 16, 16, 0, 0, 0, 0, UNORM, true},
    {RGBA16_EXT, VK_FORMAT_R16G16B16A16_UNORM, RGBA, 8, 16, 16, 16, 16, 0, 0, UNORM, true},
    {R8I, VK_FORMAT_R8_SINT, RED_INTEGER, 1, 8, 0, 0, 0, 0, 0, INT, true},
    {R8UI, VK_FORMAT_R8_UINT, RED_INTEGER, 1, 8, 0, 0, 0, 0, 0, UNSIGNED_INT, true},
    {R16I, VK_FORMAT_R16_SINT, RED_INTEGER, 2, 16, 0, 0, 0, 0, 0, INT, true},
    {R16UI, VK_FORMAT_R16_UINT, RED_INTEGER, 2, 16, 0, 0, 0, 0, 0, UNSIGNED_INT, true},
    {R32I, VK_FORMAT_R32_SINT, RED_INTEGER, 4, 32, 0, 0, 0, 0, 0, INT, true},
    {R32UI, VK_FORMAT_R32_UINT, RED_INTEGER, 4, 32, 0, 0, 0, 0, 0, UNSIGNED_INT, true},
    {RG8I, VK_FORMAT_R8G8_SINT, RG_INTEGER, 2, 8, 8, 0, 0, 0, 0, INT, true},
    {RG8UI, VK_FORMAT_R8G8_UINT, RG_INTEGER, 2, 8, 8, 0, 0, 0, 0, UNSIGNED_INT, true},
    {RG16I, VK_FORMAT_R16G16_SINT, RG_INTEGER, 4, 16, 16, 0, 0, 0, 0, INT, true},
    {RG16UI, VK_FORMAT_R16G16_UINT, RG_INTEGER, 4, 16, 16, 0, 0, 0, 0, UNSIGNED_INT, true},
    {RG32I, VK_FORMAT_R32G32_SINT, RG_INTEGER, 8, 32, 32, 0, 0, 0, 0, INT, true},
    {RG32UI, VK_FORMAT_R32G32_UINT, RG_INTEGER, 8, 32, 32, 0, 0, 0, 0, UNSIGNED_INT, true},
    {RGBA8I, VK_FORMAT_R8G8B8A8_SINT, RGBA_INTEGER, 4, 8, 8, 8, 8, 0, 0, INT, true},
    {RGBA8UI, VK_FORMAT_R8G8B8A8_UINT, RGBA_INTEGER, 4, 8, 8, 8, 8, 0, 0, UNSIGNED_INT, true},
    {RGBA16I, VK_FORMAT_R16G16B16A16_SINT, RGBA_INTEGER, 8, 16, 16, 16, 16, 0, 0, INT, true},
    {RGBA16UI, VK_FORMAT_R16G16B16A16_UINT, RGBA_INTEGER, 8, 16, 16, 16, 16, 0, 0, UNSIGNED_INT, true},
    {RGBA32I, VK_FORMAT_R32G32B32A32_SINT, RGBA_INTEGER, 16, 32, 32, 32, 32, 0, 0, INT, true},
    {RGBA32UI, VK_FORMAT_R32G32B32A32_UINT, RGBA_INTEGER, 16, 32, 32, 32, 32, 0, 0, UNSIGNED_INT, true},
    {RGB8I, VK_FORMAT_R8G8B8A8_SINT, RGB_INTEGER, 4, 8, 8, 8, 0, 0, 0, INT, false},
    {RGB8UI, VK_FORMAT_R8G8B8A8_UINT, RGB_INTEGER, 4, 8, 8, 8, 0, 0, 0, UNSIGNED_INT, false},
    {RGB16I, VK_FORMAT_R16G16B16A16_SINT, RGB_INTEGER, 8, 16, 16, 16, 0, 0, 0, INT, false},
    {RGB16UI, VK_FORMAT_R16G16B16A16_UINT, RGB_INTEGER, 8, 16, 16, 16, 0, 0, 0, UNSIGNED_INT, false},
    {RGB32I, VK_FORMAT_R32G32B32A32_SINT, RGB_INTEGER, 16, 32, 32, 32, 0, 0, 0, INT, false},
    {RGB32UI, VK_FORMAT_R32G32B32A32_UINT, RGB_INTEGER, 16, 32, 32, 32, 0, 0, 0, UNSIGNED_INT, false},
    {DEPTH_COMPONENT16, VK_FORMAT_D16_UNORM, DEPTH_COMPONENT, 2, 0, 0, 0, 0, 16, 0, UNORM, true},
    {DEPTH_COMPONENT24, VK_FORMAT_D24_UNORM_S8_UINT, DEPTH_COMPONENT, 4, 0, 0, 0, 0, 24, 0, UNORM, true},
    {DEPTH_COMPONENT32_OES, VK_FORMAT_D32_SFLOAT, DEPTH_COMPONENT, 4, 0, 0, 0, 0, 32, 0, FLOAT, true},
    {DEPTH_COMPONENT32F, VK_FORMAT_D32_SFLOAT, DEPTH_COMPONENT, 4, 0, 0, 0, 0, 32, 0, FLOAT, true},
    {DEPTH24_STENCIL8, VK_FORMAT_D24_UNORM_S8_UINT, DEPTH_STENCIL, 4, 0, 0, 0, 0, 24, 8, UNORM, true},
    {DEPTH32F_STENCIL8, VK_FORMAT_D32_SFLOAT_S8_UINT, DEPTH_STENCIL, 8, 0, 0, 0, 0, 32, 8, FLOAT, true},
    {STENCIL_INDEX8, VK_FORMAT_S8_UINT, STENCIL_INDEX, 1, 0, 0, 0, 0, 0, 8, UNSIGNED_INT, true},
};

struct CRow {
    GLenum internal;
    VkFormat vk;
    uint8_t bw, bh, bytes;
    bool srgb;
};
const CRow kCompressed[] = {
    {ETC1_RGB8_OES, VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 4, 4, 8, false},
    {COMPRESSED_RGB8_ETC2, VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 4, 4, 8, false},
    {COMPRESSED_SRGB8_ETC2, VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK, 4, 4, 8, true},
    {COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2, VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK, 4, 4, 8, false},
    {COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2, VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK, 4, 4, 8, true},
    {COMPRESSED_RGBA8_ETC2_EAC, VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, 4, 4, 16, false},
    {COMPRESSED_SRGB8_ALPHA8_ETC2_EAC, VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK, 4, 4, 16, true},
    {COMPRESSED_R11_EAC, VK_FORMAT_EAC_R11_UNORM_BLOCK, 4, 4, 8, false},
    {COMPRESSED_SIGNED_R11_EAC, VK_FORMAT_EAC_R11_SNORM_BLOCK, 4, 4, 8, false},
    {COMPRESSED_RG11_EAC, VK_FORMAT_EAC_R11G11_UNORM_BLOCK, 4, 4, 16, false},
    {COMPRESSED_SIGNED_RG11_EAC, VK_FORMAT_EAC_R11G11_SNORM_BLOCK, 4, 4, 16, false},
};
// ASTC: GL 0x93B0.. (4x4 5x4 5x5 6x5 6x6 8x5 8x6 8x8 10x5 10x6 10x8 10x10 12x10 12x12), sRGB at +0x20.
const uint8_t kAstcDims[14][2] = {{4, 4}, {5, 4}, {5, 5}, {6, 5}, {6, 6}, {8, 5}, {8, 6}, {8, 8},
                                  {10, 5}, {10, 6}, {10, 8}, {10, 10}, {12, 10}, {12, 12}};
// VK_FORMAT_ASTC_4x4_UNORM_BLOCK = 157, then (unorm, srgb) pairs in the same order.

float halfToFloat(uint16_t h);
uint16_t floatToHalf(float f);

std::vector<FormatInfo>& table() {
    static std::vector<FormatInfo> t = [] {
        std::vector<FormatInfo> v;
        for (const Row& r : kRows) {
            FormatInfo f;
            f.vk = r.vk, f.internal = r.internal, f.baseFormat = r.base, f.texelBytes = r.bytes;
            f.redBits = r.r, f.greenBits = r.g, f.blueBits = r.b, f.alphaBits = r.a, f.depthBits = r.d, f.stencilBits = r.s;
            f.depth = r.d != 0, f.stencil = r.s != 0;
            f.integer = r.base == RED_INTEGER || r.base == RG_INTEGER || r.base == RGB_INTEGER || r.base == RGBA_INTEGER;
            f.srgb = r.internal == SRGB8 || r.internal == SRGB8_ALPHA8;
            f.componentType = r.ctype;
            f.renderable = r.renderable;
            // RGB formats stored as RGBA: alpha reads 1.
            if (r.base == RGB || r.base == RGB_INTEGER)
                if (r.vk != VK_FORMAT_R5G6B5_UNORM_PACK16 && r.vk != VK_FORMAT_B10G11R11_UFLOAT_PACK32 &&
                    r.vk != VK_FORMAT_E5B9G9R9_UFLOAT_PACK32)
                    f.swz[3] = VK_COMPONENT_SWIZZLE_ONE;
            v.push_back(f);
        }
        for (const CRow& c : kCompressed) {
            FormatInfo f;
            f.vk = c.vk, f.internal = c.internal, f.baseFormat = RGBA, f.texelBytes = c.bytes;
            f.blockW = c.bw, f.blockH = c.bh, f.compressed = true, f.srgb = c.srgb, f.componentType = UNORM;
            f.redBits = f.greenBits = f.blueBits = 8;
            v.push_back(f);
        }
        for (int i = 0; i < 14; ++i)
            for (int s = 0; s < 2; ++s) {
                FormatInfo f;
                f.internal = GLenum((s ? COMPRESSED_SRGB8_ALPHA8_ASTC_4x4 : COMPRESSED_RGBA_ASTC_4x4) + i);
                f.vk = VkFormat(157 + 2 * i + s);
                f.baseFormat = RGBA, f.texelBytes = 16, f.blockW = kAstcDims[i][0], f.blockH = kAstcDims[i][1];
                f.compressed = true, f.srgb = s == 1, f.componentType = UNORM;
                f.redBits = f.greenBits = f.blueBits = f.alphaBits = 8;
                v.push_back(f);
            }
        // Unsized luminance/alpha formats (GLES 2): stored as R8 / RG8 with a sampling swizzle.
        auto lum = [&](GLenum internal, VkFormat vk, uint8_t bytes, VkComponentSwizzle r, VkComponentSwizzle g,
                       VkComponentSwizzle b, VkComponentSwizzle a, GLenum ctype, uint8_t floatBits = 16) {
            FormatInfo f;
            f.vk = vk, f.internal = internal, f.baseFormat = internal, f.texelBytes = bytes, f.componentType = ctype;
            f.swz[0] = r, f.swz[1] = g, f.swz[2] = b, f.swz[3] = a;
            if (ctype == FLOAT) f.redBits = floatBits;  // half float (OES_texture_half_float) or float (OES_texture_float)
            v.push_back(f);
        };
        lum(LUMINANCE, VK_FORMAT_R8_UNORM, 1, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R,
            VK_COMPONENT_SWIZZLE_ONE, UNORM);
        lum(ALPHA, VK_FORMAT_R8_UNORM, 1, VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO,
            VK_COMPONENT_SWIZZLE_R, UNORM);
        lum(LUMINANCE_ALPHA, VK_FORMAT_R8G8_UNORM, 2, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R,
            VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, UNORM);
        // The same with HALF_FLOAT_OES data (found by formatFor, after the 8-bit rows of the same internal format).
        lum(LUMINANCE, VK_FORMAT_R16_SFLOAT, 2, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R,
            VK_COMPONENT_SWIZZLE_ONE, FLOAT);
        lum(ALPHA, VK_FORMAT_R16_SFLOAT, 2, VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO,
            VK_COMPONENT_SWIZZLE_R, FLOAT);
        lum(LUMINANCE_ALPHA, VK_FORMAT_R16G16_SFLOAT, 4, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R,
            VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, FLOAT);
        // ... and with FLOAT data (OES_texture_float).
        lum(LUMINANCE, VK_FORMAT_R32_SFLOAT, 4, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R,
            VK_COMPONENT_SWIZZLE_ONE, FLOAT, 32);
        lum(ALPHA, VK_FORMAT_R32_SFLOAT, 4, VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO,
            VK_COMPONENT_SWIZZLE_R, FLOAT, 32);
        lum(LUMINANCE_ALPHA, VK_FORMAT_R32G32_SFLOAT, 8, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R,
            VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, FLOAT, 32);
        return v;
    }();
    return t;
}

const FormatInfo* bySized(GLenum internal) {
    for (const FormatInfo& f : table())
        if (f.internal == internal) return &f;
    return nullptr;
}

}  // namespace

const FormatInfo* formatFor(GLenum internal, GLenum format, GLenum type) {
    if (const FormatInfo* f = bySized(internal)) {
        if (internal == LUMINANCE || internal == ALPHA || internal == LUMINANCE_ALPHA) {
            // Half-float / float luminance and alpha (OES_texture_half_float, OES_texture_float) are R16F / RG16F / R32F / RG32F
            // with a swizzle.
            if (type == HALF_FLOAT_OES || type == FLOAT)
                for (const FormatInfo& x : table())
                    if (x.internal == internal && x.componentType == FLOAT && x.redBits == (type == FLOAT ? 32 : 16)) return &x;
            if (type != 0 && type != UNSIGNED_BYTE) return nullptr;
        }
        return f;
    }
    // Unsized (GLES 2 glTexImage2D(internalformat = format)).
    switch (internal) {
        case RGBA:
            if (type == UNSIGNED_SHORT_4_4_4_4) return bySized(RGBA4);
            if (type == UNSIGNED_SHORT_5_5_5_1) return bySized(RGB5_A1);
            if (type == HALF_FLOAT || type == HALF_FLOAT_OES) return bySized(RGBA16F);
            if (type == FLOAT) return bySized(RGBA32F);
            if (type == UNSIGNED_INT_2_10_10_10_REV) return bySized(RGB10_A2);
            return bySized(RGBA8);
        case RGB:
            if (type == UNSIGNED_SHORT_5_6_5) return bySized(RGB565);
            if (type == HALF_FLOAT || type == HALF_FLOAT_OES) return bySized(RGB16F);
            if (type == FLOAT) return bySized(RGB32F);
            if (type == UNSIGNED_INT_10F_11F_11F_REV) return bySized(R11F_G11F_B10F);
            return bySized(RGB8);
        case BGRA_EXT: return bySized(BGRA8_EXT);
        case RED: return type == FLOAT ? bySized(R32F) : (type == HALF_FLOAT || type == HALF_FLOAT_OES) ? bySized(R16F) : bySized(R8);
        case RG: return type == FLOAT ? bySized(RG32F) : (type == HALF_FLOAT || type == HALF_FLOAT_OES) ? bySized(RG16F) : bySized(RG8);
        case DEPTH_COMPONENT: return type == UNSIGNED_SHORT ? bySized(DEPTH_COMPONENT16) : type == FLOAT ? bySized(DEPTH_COMPONENT32F) : bySized(DEPTH_COMPONENT24);
        case DEPTH_STENCIL: return bySized(DEPTH24_STENCIL8);
        case SRGB: return bySized(SRGB8);
        case 0x8C42 /* SRGB_ALPHA_EXT */: return bySized(SRGB8_ALPHA8);
        default: break;
    }
    (void)format;
    return nullptr;
}

// Several GL formats share one Vulkan format (RGB8/RGBA8, SRGB8/SRGB8_ALPHA8: the RGB ones are stored as RGBA with
// alpha read as 1 and never written). An image the host wraps is a real RGBA image (the eye swapchain of a passthrough
// game): the first match used to be the RGB row, which forced the clear alpha to 1 and masked alpha writes, so the
// room never showed through. Prefer the row that has alpha.
const FormatInfo* formatByVk(VkFormat f) {
    const FormatInfo* first = nullptr;
    for (const FormatInfo& x : table()) {
        if (x.vk != f || x.compressed) continue;
        if (x.alphaBits || x.stencil) return &x;  // D24S8 is DEPTH24_STENCIL8 (not DEPTH_COMPONENT24: the window's stencil would vanish)
        if (!first) first = &x;
    }
    return first;
}

static uint32_t components(GLenum format) {
    switch (format) {
        case RED: case RED_INTEGER: case ALPHA: case LUMINANCE: case DEPTH_COMPONENT: case STENCIL_INDEX: return 1;
        case RG: case RG_INTEGER: case LUMINANCE_ALPHA: return 2;
        case RGB: case RGB_INTEGER: return 3;
        case RGBA: case RGBA_INTEGER: case BGRA_EXT: return 4;
        case DEPTH_STENCIL: return 1;
        default: return 0;
    }
}

// RGB floats into R11F_G11F_B10F (negative and NaN read 0; no sign bit, 6/6/5 mantissa bits truncated from half precision).
static uint32_t pack11f10f(const float c[3]) {
    auto uf = [&](float v) { return v != v || v <= 0 ? uint16_t(0) : floatToHalf(v); };
    return uint32_t(uf(c[0]) >> 4) | uint32_t(uf(c[1]) >> 4) << 11 | uint32_t(uf(c[2]) >> 5) << 22;
}

// RGB floats into RGB9_E5 (ES 3.0 / EXT_texture_shared_exponent 8.5.2 "Shared exponent texture color conversion").
static uint32_t packRgb9e5(const float c[3]) {
    const float kMax = 65408.0f;  // (511 / 512) * 2^(31 - 15)
    float r[3];
    for (int k = 0; k < 3; ++k) r[k] = c[k] != c[k] || c[k] < 0 ? 0.0f : c[k] > kMax ? kMax : c[k];
    const float maxc = std::fmax(r[0], std::fmax(r[1], r[2]));
    int e = maxc > 0 ? int(std::floor(std::log2(maxc))) : -16;
    if (e < -16) e = -16;
    int expShared = e + 16;
    double denom = std::ldexp(1.0, expShared - 15 - 9);
    if (std::floor(maxc / denom + 0.5) == 512.0) ++expShared, denom *= 2;
    uint32_t m[3];
    for (int k = 0; k < 3; ++k) m[k] = uint32_t(std::floor(r[k] / denom + 0.5));
    return m[0] | m[1] << 9 | m[2] << 18 | uint32_t(expShared) << 27;
}

GLenum transferError(const FormatInfo& f, GLenum format, GLenum type, bool sub) {
    if (format == 0x8C40) format = RGB;        // GL_SRGB_EXT / GL_SRGB_ALPHA_EXT (EXT_sRGB): the same data layout
    else if (format == 0x8C42) format = RGBA;
    switch (format) {
        case RED: case RG: case RGB: case RGBA: case RED_INTEGER: case RG_INTEGER: case RGB_INTEGER: case RGBA_INTEGER:
        case DEPTH_COMPONENT: case DEPTH_STENCIL: case LUMINANCE: case LUMINANCE_ALPHA: case ALPHA: case BGRA_EXT: break;
        default: return INVALID_ENUM;
    }
    switch (type) {
        case UNSIGNED_BYTE: case BYTE: case UNSIGNED_SHORT: case SHORT: case UNSIGNED_INT: case INT: case HALF_FLOAT:
        case HALF_FLOAT_OES: case FLOAT: case UNSIGNED_SHORT_5_6_5: case UNSIGNED_SHORT_4_4_4_4: case UNSIGNED_SHORT_5_5_5_1:
        case UNSIGNED_INT_2_10_10_10_REV: case UNSIGNED_INT_10F_11F_11F_REV: case UNSIGNED_INT_5_9_9_9_REV:
        case UNSIGNED_INT_24_8: case FLOAT_32_UNSIGNED_INT_24_8_REV: break;
        default: return INVALID_ENUM;
    }
    if (f.compressed) return INVALID_OPERATION;
    if (format != f.baseFormat) {
        const bool intFmt = format == RED_INTEGER || format == RG_INTEGER || format == RGB_INTEGER || format == RGBA_INTEGER;
        const bool depthFmt = format == DEPTH_COMPONENT || format == DEPTH_STENCIL;
        if (!sub || f.depth || f.stencil || depthFmt || intFmt != f.integer) return INVALID_OPERATION;
    }
    const bool half = type == HALF_FLOAT || type == HALF_FLOAT_OES;
    bool ok = false;
    switch (f.internal) {
        case RGB565: ok = type == UNSIGNED_BYTE || type == UNSIGNED_SHORT_5_6_5; break;
        case RGBA4: ok = type == UNSIGNED_BYTE || type == UNSIGNED_SHORT_4_4_4_4; break;
        case RGB5_A1: ok = type == UNSIGNED_BYTE || type == UNSIGNED_SHORT_5_5_5_1 || type == UNSIGNED_INT_2_10_10_10_REV; break;
        case RGB10_A2: case RGB10_A2UI: ok = type == UNSIGNED_INT_2_10_10_10_REV; break;
        case R11F_G11F_B10F: ok = type == UNSIGNED_INT_10F_11F_11F_REV || half || type == FLOAT; break;
        case RGB9_E5: ok = type == UNSIGNED_INT_5_9_9_9_REV || half || type == FLOAT; break;
        case DEPTH_COMPONENT16: ok = type == UNSIGNED_SHORT || type == UNSIGNED_INT; break;
        case DEPTH_COMPONENT24: ok = type == UNSIGNED_INT; break;
        case DEPTH_COMPONENT32F: case DEPTH_COMPONENT32_OES: ok = type == FLOAT; break;
        case DEPTH24_STENCIL8: ok = type == UNSIGNED_INT_24_8; break;
        case DEPTH32F_STENCIL8: ok = type == FLOAT_32_UNSIGNED_INT_24_8_REV; break;
        case STENCIL_INDEX8: ok = type == UNSIGNED_BYTE; break;
        default:
            if (f.integer) {
                const bool sgn = f.componentType == INT;
                ok = f.redBits == 8 ? type == (sgn ? BYTE : UNSIGNED_BYTE)
                     : f.redBits == 16 ? type == (sgn ? SHORT : UNSIGNED_SHORT) : type == (sgn ? INT : UNSIGNED_INT);
            } else if (f.componentType == FLOAT) {
                ok = f.redBits == 16 ? (half || type == FLOAT) : type == FLOAT;
            } else if (f.componentType == SNORM) {
                ok = type == BYTE;
            } else {
                ok = f.redBits == 16 ? type == UNSIGNED_SHORT : type == UNSIGNED_BYTE;
            }
    }
    return ok ? 0 : INVALID_OPERATION;
}

uint32_t clientPixelBytes(GLenum format, GLenum type) {
    switch (type) {
        case UNSIGNED_SHORT_5_6_5: case UNSIGNED_SHORT_4_4_4_4: case UNSIGNED_SHORT_5_5_5_1: return 2;
        case UNSIGNED_INT_2_10_10_10_REV: case UNSIGNED_INT_10F_11F_11F_REV: case UNSIGNED_INT_5_9_9_9_REV:
        case UNSIGNED_INT_24_8: return 4;
        case FLOAT_32_UNSIGNED_INT_24_8_REV: return 8;
        default: break;
    }
    uint32_t c = components(format);
    switch (type) {
        case UNSIGNED_BYTE: case BYTE: return c;
        case UNSIGNED_SHORT: case SHORT: case HALF_FLOAT: case HALF_FLOAT_OES: return 2 * c;
        case UNSIGNED_INT: case INT: case FLOAT: return 4 * c;
        default: return 0;
    }
}

uint32_t compressedSize(const FormatInfo& f, uint32_t w, uint32_t h) {
    uint32_t bx = (w + f.blockW - 1) / f.blockW, by = (h + f.blockH - 1) / f.blockH;
    return bx * by * f.texelBytes;
}

bool convertUpload(const FormatInfo& dst, GLenum format, GLenum type, const uint8_t* src, uint32_t n,
                   std::vector<uint8_t>& out) {
    if (!clientPixelBytes(format, type)) return false;
    out.resize(size_t(n) * dst.texelBytes);
    return convertUploadTo(dst, format, type, src, n, out.data());
}

bool convertUploadTo(const FormatInfo& dst, GLenum format, GLenum type, const uint8_t* src, uint32_t n, uint8_t* out) {
    const uint32_t inBytes = clientPixelBytes(format, type);
    if (!inBytes) return false;
    const size_t outBytes = size_t(n) * dst.texelBytes;
    // Same layout: copy.
    const bool rgbToRgba = (format == RGB || format == RGB_INTEGER) && dst.texelBytes == inBytes / 3 * 4 &&
                           (type == UNSIGNED_BYTE || type == BYTE || type == FLOAT || type == HALF_FLOAT ||
                            type == HALF_FLOAT_OES || type == UNSIGNED_SHORT || type == SHORT || type == INT ||
                            type == UNSIGNED_INT);
    if (rgbToRgba) {
        const uint32_t cb = inBytes / 3;
        for (uint32_t i = 0; i < n; ++i) {
            uint8_t* d = out + size_t(i) * dst.texelBytes;
            std::memcpy(d, src + size_t(i) * inBytes, inBytes);
            // alpha = 1 in the type's representation
            if (type == FLOAT) {
                float one = 1.0f;
                std::memcpy(d + 3 * cb, &one, 4);
            } else if (type == HALF_FLOAT || type == HALF_FLOAT_OES) {
                uint16_t one = 0x3C00;
                std::memcpy(d + 3 * cb, &one, 2);
            } else if (format == RGB_INTEGER) {
                uint32_t one = 1;
                std::memcpy(d + 3 * cb, &one, cb);
            } else {
                std::memset(d + 3 * cb, type == BYTE ? 0x7F : 0xFF, cb);
            }
        }
        return true;
    }
    // DEPTH_COMPONENT client data into D16 / D32F / D24. Before the same-layout copy: UNSIGNED_INT into D32F and
    // FLOAT into D24 have equal sizes but different values. Pairs already handled below / by the plain copy are skipped.
    const bool depthDst = dst.vk == VK_FORMAT_D16_UNORM || dst.vk == VK_FORMAT_D32_SFLOAT ||
                          dst.vk == VK_FORMAT_D24_UNORM_S8_UINT;
    const bool depthDone = (type == UNSIGNED_SHORT && dst.vk == VK_FORMAT_D16_UNORM) ||
                           (type == FLOAT && dst.vk == VK_FORMAT_D32_SFLOAT) ||
                           (type == UNSIGNED_INT && dst.vk == VK_FORMAT_D24_UNORM_S8_UINT);
    if (format == DEPTH_COMPONENT && depthDst && !depthDone && (type == UNSIGNED_SHORT || type == UNSIGNED_INT || type == FLOAT)) {
        for (uint32_t i = 0; i < n; ++i) {
            if (type == UNSIGNED_INT && dst.vk == VK_FORMAT_D16_UNORM) {
                uint32_t p;
                std::memcpy(&p, src + 4 * i, 4);
                const uint16_t q = uint16_t(p >> 16);
                std::memcpy(out + 2 * i, &q, 2);
                continue;
            }
            double v;  // normalized depth 0..1
            if (type == UNSIGNED_SHORT) {
                uint16_t p;
                std::memcpy(&p, src + 2 * i, 2);
                v = p / 65535.0;
            } else if (type == UNSIGNED_INT) {
                uint32_t p;
                std::memcpy(&p, src + 4 * i, 4);
                v = p / 4294967295.0;
            } else {
                float p;
                std::memcpy(&p, src + 4 * i, 4);
                v = !(p > 0) ? 0 : (p > 1 ? 1 : p);  // NaN -> 0
            }
            if (dst.vk == VK_FORMAT_D16_UNORM) {
                const uint16_t q = uint16_t(std::lround(v * 65535.0));
                std::memcpy(out + 2 * i, &q, 2);
            } else if (dst.vk == VK_FORMAT_D32_SFLOAT) {
                const float q = float(v);
                std::memcpy(out + 4 * i, &q, 4);
            } else {
                const uint32_t q = uint32_t(std::lround(v * 16777215.0));
                std::memcpy(out + 4 * i, &q, 4);
            }
        }
        return true;
    }
    // Before the same-size copy below, which would otherwise take it as a raw copy.
    if (type == UNSIGNED_INT && format == DEPTH_COMPONENT && dst.vk == VK_FORMAT_D24_UNORM_S8_UINT) {
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t p;
            std::memcpy(&p, src + 4 * i, 4);
            uint32_t q = p >> 8;
            std::memcpy(out + 4 * i, &q, 4);
        }
        return true;
    }
    // 32-bit float client data into half float storage (ES 3.0 table 3.2: R16F, RG16F, RGB16F, RGBA16F take FLOAT).
    if (type == FLOAT && !dst.compressed && dst.componentType == FLOAT && dst.redBits == 16 && !dst.depth) {
        const uint32_t sc = components(format), dc = dst.texelBytes / 2;
        if (!sc || sc > dc) return false;
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t k = 0; k < dc; ++k) {
                float v = 0;
                if (k < sc) std::memcpy(&v, src + (size_t(i) * sc + k) * 4, 4);
                else if (k == 3) v = 1.0f;
                const uint16_t h = floatToHalf(v);
                std::memcpy(&out[(size_t(i) * dc + k) * 2], &h, 2);
            }
        return true;
    }
    // HALF_FLOAT / FLOAT RGB data into the packed shared-exponent and 11/11/10 float formats.
    if (format == RGB && (type == FLOAT || type == HALF_FLOAT || type == HALF_FLOAT_OES) &&
        (dst.vk == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || dst.vk == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32)) {
        const bool f32 = type == FLOAT;
        for (uint32_t i = 0; i < n; ++i) {
            float c[3];
            for (int k = 0; k < 3; ++k) {
                if (f32) {
                    std::memcpy(&c[k], src + (size_t(i) * 3 + k) * 4, 4);
                } else {
                    uint16_t h;
                    std::memcpy(&h, src + (size_t(i) * 3 + k) * 2, 2);
                    c[k] = halfToFloat(h);
                }
            }
            const uint32_t q = dst.vk == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 ? packRgb9e5(c) : pack11f10f(c);
            std::memcpy(&out[4 * size_t(i)], &q, 4);
        }
        return true;
    }
    // 8-bit and 2-10-10-10 client data into RGB5_A1 (GL R 15..11, G 10..6, B 5..1, A 0 -> Vulkan A1R5G5B5).
    if (dst.vk == VK_FORMAT_A1R5G5B5_UNORM_PACK16 && format == RGBA && (type == UNSIGNED_BYTE || type == UNSIGNED_INT_2_10_10_10_REV)) {
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t r, g, b, a;
            if (type == UNSIGNED_BYTE) {
                const uint8_t* p = src + 4 * size_t(i);
                r = (p[0] * 31u + 127) / 255, g = (p[1] * 31u + 127) / 255, b = (p[2] * 31u + 127) / 255, a = p[3] >= 128;
            } else {
                uint32_t p;
                std::memcpy(&p, src + 4 * size_t(i), 4);
                r = ((p & 1023) * 31 + 511) / 1023, g = (((p >> 10) & 1023) * 31 + 511) / 1023, b = (((p >> 20) & 1023) * 31 + 511) / 1023;
                a = (p >> 30) >= 2;
            }
            const uint16_t q = uint16_t(a << 15 | r << 10 | g << 5 | b);
            std::memcpy(&out[2 * size_t(i)], &q, 2);
        }
        return true;
    }
    if (inBytes == dst.texelBytes) {
        if (type == UNSIGNED_SHORT_5_6_5 && dst.vk == VK_FORMAT_R5G6B5_UNORM_PACK16) {
            std::memcpy(out, src, outBytes);
            return true;
        }
        if (type == UNSIGNED_SHORT_4_4_4_4 && dst.vk == VK_FORMAT_R4G4B4A4_UNORM_PACK16) {
            std::memcpy(out, src, outBytes);  // same bit order (R in 15..12)
            return true;
        }
        if (type == UNSIGNED_SHORT_5_5_5_1 && dst.vk == VK_FORMAT_A1R5G5B5_UNORM_PACK16) {
            // GL: R 15..11 G 10..6 B 5..1 A 0 -> Vulkan A1R5G5B5: A 15, R 14..10, G 9..5, B 4..0.
            for (uint32_t i = 0; i < n; ++i) {
                uint16_t p;
                std::memcpy(&p, src + 2 * i, 2);
                uint16_t q = uint16_t((p & 1) << 15 | (p >> 1));
                std::memcpy(out + 2 * i, &q, 2);
            }
            return true;
        }
        if (type == UNSIGNED_INT_24_8 && dst.vk == VK_FORMAT_D24_UNORM_S8_UINT) {
            // GL: depth in 31..8, stencil 7..0; the null/Metal D24S8 upload takes depth in 23..0.
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t p;
                std::memcpy(&p, src + 4 * i, 4);
                uint32_t q = (p >> 8) | (p & 0xFF) << 24;
                std::memcpy(out + 4 * i, &q, 4);
            }
            return true;
        }
        std::memcpy(out, src, outBytes);
        return true;
    }
    // 8-bit RGBA into 16-bit packed targets (GLES 2 apps uploading RGBA bytes to an RGBA4 texture).
    if (type == UNSIGNED_BYTE && format == RGBA && dst.vk == VK_FORMAT_R4G4B4A4_UNORM_PACK16) {
        for (uint32_t i = 0; i < n; ++i) {
            const uint8_t* p = src + 4 * i;
            // unorm8 -> unorm4 rounds to nearest (ES 3.0 2.1.6.1): (v * 15 + 127) / 255, 255 -> 15, 8 -> 0 (not v >> 4)
            auto q4 = [](uint32_t v) { return (v * 15u + 127u) / 255u; };
            uint16_t q = uint16_t(q4(p[0]) << 12 | q4(p[1]) << 8 | q4(p[2]) << 4 | q4(p[3]));
            std::memcpy(out + 2 * i, &q, 2);
        }
        return true;
    }
    if (type == UNSIGNED_BYTE && format == RGB && dst.vk == VK_FORMAT_R5G6B5_UNORM_PACK16) {
        for (uint32_t i = 0; i < n; ++i) {
            const uint8_t* p = src + 3 * i;
            const uint32_t r = (p[0] * 31u + 127u) / 255u, g = (p[1] * 63u + 127u) / 255u, b = (p[2] * 31u + 127u) / 255u;  // rounded, not >> 3 / >> 2
            uint16_t q = uint16_t(r << 11 | g << 5 | b);
            std::memcpy(out + 2 * i, &q, 2);
        }
        return true;
    }
    return false;
}

namespace {

float halfToFloat(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? std::ldexp(float(m), -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp(float(m | 1024), int(e) - 25);
    return s ? -v : v;
}

uint16_t floatToHalf(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    const uint32_t sign = (b >> 16) & 0x8000, ab = b & 0x7FFFFFFF;
    if (ab > 0x7F800000) return uint16_t(sign | 0x7E00);        // NaN
    if (ab >= 0x477FF000) return uint16_t(sign | 0x7C00);       // overflow (and inf)
    if (ab < 0x33000001) return uint16_t(sign);                 // underflow
    const int e = int(ab >> 23) - 127;
    if (e < -14) {                                              // subnormal half
        const float scaled = std::ldexp(std::fabs(f), 24);
        return uint16_t(sign | uint32_t(std::nearbyint(scaled)));
    }
    uint32_t h = uint32_t(e + 15) << 10 | ((ab >> 13) & 0x3FF);
    const uint32_t tail = ab & 0x1FFF;
    if (tail > 0x1000 || (tail == 0x1000 && (h & 1))) ++h;      // round to nearest even (carries into the exponent)
    return uint16_t(sign | h);
}

// One texel of a colour format as four channels: normalized/float formats as their value, integer formats as the exact
// integer (a double holds a 32-bit one). Channels the format lacks read 0, 0, 0, 1 like glReadPixels.
struct Texel {
    double v[4] = {0, 0, 0, 1};
    bool isInt = false;
};

bool decodeTexel(const FormatInfo& src, const uint8_t* p, Texel& t) {
    auto u8 = [&](int i) { return double(p[i]); };
    auto s8 = [&](int i) { return double(int8_t(p[i])); };
    auto u16 = [&](int i) { uint16_t x; std::memcpy(&x, p + 2 * i, 2); return x; };
    auto u32 = [&](int i) { uint32_t x; std::memcpy(&x, p + 4 * i, 4); return x; };
    auto f32 = [&](int i) { float x; std::memcpy(&x, p + 4 * i, 4); return double(x); };
    auto nc = [](double x, double lo) { return x < lo ? lo : x; };
    int n = 0;
    switch (src.vk) {
        case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB:
            n = src.texelBytes;
            for (int i = 0; i < n; ++i) t.v[i] = u8(i) / 255.0;
            break;
        case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
            t.v[0] = u8(2) / 255.0, t.v[1] = u8(1) / 255.0, t.v[2] = u8(0) / 255.0, t.v[3] = u8(3) / 255.0;
            break;
        case VK_FORMAT_R8_SNORM: case VK_FORMAT_R8G8_SNORM: case VK_FORMAT_R8G8B8A8_SNORM:
            n = src.texelBytes;
            for (int i = 0; i < n; ++i) t.v[i] = nc(s8(i) / 127.0, -1.0);
            break;
        case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R16G16B16A16_UNORM:
            n = src.texelBytes / 2;
            for (int i = 0; i < n; ++i) t.v[i] = u16(i) / 65535.0;
            break;
        case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16B16A16_SFLOAT:
            n = src.texelBytes / 2;
            for (int i = 0; i < n; ++i) t.v[i] = halfToFloat(u16(i));
            break;
        case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32B32A32_SFLOAT:
            n = src.texelBytes / 4;
            for (int i = 0; i < n; ++i) t.v[i] = f32(i);
            break;
        case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8A8_UINT:
            t.isInt = true, n = src.texelBytes;
            for (int i = 0; i < n; ++i) t.v[i] = u8(i);
            break;
        case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8A8_SINT:
            t.isInt = true, n = src.texelBytes;
            for (int i = 0; i < n; ++i) t.v[i] = s8(i);
            break;
        case VK_FORMAT_R16_UINT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16B16A16_UINT:
            t.isInt = true, n = src.texelBytes / 2;
            for (int i = 0; i < n; ++i) t.v[i] = u16(i);
            break;
        case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16A16_SINT:
            t.isInt = true, n = src.texelBytes / 2;
            for (int i = 0; i < n; ++i) t.v[i] = double(int16_t(u16(i)));
            break;
        case VK_FORMAT_R32_UINT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32B32A32_UINT:
            t.isInt = true, n = src.texelBytes / 4;
            for (int i = 0; i < n; ++i) t.v[i] = u32(i);
            break;
        case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32B32A32_SINT:
            t.isInt = true, n = src.texelBytes / 4;
            for (int i = 0; i < n; ++i) t.v[i] = double(int32_t(u32(i)));
            break;
        case VK_FORMAT_R5G6B5_UNORM_PACK16: {
            const uint16_t q = u16(0);
            t.v[0] = ((q >> 11) & 31) / 31.0, t.v[1] = ((q >> 5) & 63) / 63.0, t.v[2] = (q & 31) / 31.0;
            return true;
        }
        case VK_FORMAT_R4G4B4A4_UNORM_PACK16: {
            const uint16_t q = u16(0);
            for (int c = 0; c < 4; ++c) t.v[c] = ((q >> (12 - 4 * c)) & 15) / 15.0;
            return true;
        }
        case VK_FORMAT_A1R5G5B5_UNORM_PACK16: {
            const uint16_t q = u16(0);
            t.v[0] = ((q >> 10) & 31) / 31.0, t.v[1] = ((q >> 5) & 31) / 31.0, t.v[2] = (q & 31) / 31.0, t.v[3] = (q >> 15) & 1;
            return true;
        }
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2B10G10R10_UINT_PACK32: {
            const uint32_t q = u32(0);
            const bool ui = src.vk == VK_FORMAT_A2B10G10R10_UINT_PACK32;
            const double d10 = ui ? 1.0 : 1023.0, d2 = ui ? 1.0 : 3.0;
            t.isInt = ui;
            t.v[0] = (q & 1023) / d10, t.v[1] = ((q >> 10) & 1023) / d10, t.v[2] = ((q >> 20) & 1023) / d10, t.v[3] = (q >> 30) / d2;
            return true;
        }
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32: {
            const uint32_t q = u32(0);
            t.v[0] = halfToFloat(uint16_t((q & 0x7FF) << 4)), t.v[1] = halfToFloat(uint16_t(((q >> 11) & 0x7FF) << 4)),
            t.v[2] = halfToFloat(uint16_t(((q >> 22) & 0x3FF) << 5));
            return true;
        }
        default: return false;
    }
    if (src.alphaBits == 0) t.v[3] = 1;          // RGB formats stored as RGBA: alpha reads 1
    return true;
}

// Stores `v` as a `type` component: normalized/float sources scale to the integer range, integer ones clamp to it.
bool putComponent(GLenum type, double v, bool isInt, uint8_t* o) {
    auto clampTo = [&](double x, double lo, double hi) { return x != x ? 0.0 : x < lo ? lo : x > hi ? hi : x; };
    switch (type) {
        case UNSIGNED_BYTE: *o = uint8_t(std::llround(isInt ? clampTo(v, 0, 255) : clampTo(v, 0, 1) * 255.0)); return true;
        case BYTE: *o = uint8_t(int8_t(std::llround(isInt ? clampTo(v, -128, 127) : clampTo(v, -1, 1) * 127.0))); return true;
        case UNSIGNED_SHORT: {
            const uint16_t x = uint16_t(std::llround(isInt ? clampTo(v, 0, 65535) : clampTo(v, 0, 1) * 65535.0));
            std::memcpy(o, &x, 2);
            return true;
        }
        case SHORT: {
            const int16_t x = int16_t(std::llround(isInt ? clampTo(v, -32768, 32767) : clampTo(v, -1, 1) * 32767.0));
            std::memcpy(o, &x, 2);
            return true;
        }
        case UNSIGNED_INT: {
            const uint32_t x = uint32_t(std::llround(isInt ? clampTo(v, 0, 4294967295.0) : clampTo(v, 0, 1) * 4294967295.0));
            std::memcpy(o, &x, 4);
            return true;
        }
        case INT: {
            const int32_t x = int32_t(std::llround(isInt ? clampTo(v, -2147483648.0, 2147483647.0) : clampTo(v, -1, 1) * 2147483647.0));
            std::memcpy(o, &x, 4);
            return true;
        }
        case FLOAT: {
            const float x = float(v);
            std::memcpy(o, &x, 4);
            return true;
        }
        case HALF_FLOAT: case HALF_FLOAT_OES: {
            const uint16_t x = floatToHalf(float(v));
            std::memcpy(o, &x, 2);
            return true;
        }
        default: return false;
    }
}

// Generic path of glReadPixels: decode the texel, encode it into the client (format, type).
bool convertReadbackGeneric(const FormatInfo& src, GLenum format, GLenum type, const uint8_t* in, uint32_t n, uint8_t* out) {
    const uint32_t outBytes = clientPixelBytes(format, type);
    const bool packed = type == UNSIGNED_SHORT_5_6_5 || type == UNSIGNED_SHORT_4_4_4_4 || type == UNSIGNED_SHORT_5_5_5_1 ||
                        type == UNSIGNED_INT_2_10_10_10_REV || type == UNSIGNED_INT_10F_11F_11F_REV;
    // glReadPixels formats (no luminance/alpha) and the packed types that go with each.
    switch (format) {
        case RED: case RG: case RGB: case RGBA: case BGRA_EXT: case RED_INTEGER: case RG_INTEGER: case RGB_INTEGER: case RGBA_INTEGER: break;
        default: return false;
    }
    if ((type == UNSIGNED_SHORT_5_6_5 || type == UNSIGNED_INT_10F_11F_11F_REV) && format != RGB) return false;
    if ((type == UNSIGNED_SHORT_4_4_4_4 || type == UNSIGNED_SHORT_5_5_5_1) && format != RGBA) return false;
    if (type == UNSIGNED_INT_2_10_10_10_REV && format != RGBA && format != RGBA_INTEGER) return false;
    const uint32_t comps = packed ? 0 : outBytes ? components(format) : 0;
    if (!packed && (!comps || format == DEPTH_COMPONENT || format == DEPTH_STENCIL || format == STENCIL_INDEX)) return false;
    const bool bgra = format == BGRA_EXT;
    for (uint32_t i = 0; i < n; ++i) {
        Texel t;
        if (!decodeTexel(src, in + size_t(i) * src.texelBytes, t)) return false;
        uint8_t* o = out + size_t(i) * outBytes;
        if (bgra) std::swap(t.v[0], t.v[2]);
        if (!packed) {
            const uint32_t cb = outBytes / comps;
            for (uint32_t c = 0; c < comps; ++c)
                if (!putComponent(type, t.v[c], t.isInt, o + c * cb)) return false;
            continue;
        }
        auto nrm = [&](double v, double max) { return v != v ? 0.0 : std::round((v < 0 ? 0 : v > 1 ? 1 : v) * max); };
        switch (type) {
            case UNSIGNED_SHORT_5_6_5: {
                const uint16_t q = uint16_t(uint32_t(nrm(t.v[0], 31)) << 11 | uint32_t(nrm(t.v[1], 63)) << 5 | uint32_t(nrm(t.v[2], 31)));
                std::memcpy(o, &q, 2);
                break;
            }
            case UNSIGNED_SHORT_4_4_4_4: {
                const uint16_t q = uint16_t(uint32_t(nrm(t.v[0], 15)) << 12 | uint32_t(nrm(t.v[1], 15)) << 8 |
                                            uint32_t(nrm(t.v[2], 15)) << 4 | uint32_t(nrm(t.v[3], 15)));
                std::memcpy(o, &q, 2);
                break;
            }
            case UNSIGNED_SHORT_5_5_5_1: {
                const uint16_t q = uint16_t(uint32_t(nrm(t.v[0], 31)) << 11 | uint32_t(nrm(t.v[1], 31)) << 6 |
                                            uint32_t(nrm(t.v[2], 31)) << 1 | uint32_t(nrm(t.v[3], 1)));
                std::memcpy(o, &q, 2);
                break;
            }
            case UNSIGNED_INT_2_10_10_10_REV: {
                auto ch = [&](int c, double max) {
                    return t.isInt ? uint32_t(t.v[c] < 0 ? 0 : t.v[c] > max ? max : t.v[c]) : uint32_t(nrm(t.v[c], max));
                };
                const uint32_t q = ch(0, 1023) | ch(1, 1023) << 10 | ch(2, 1023) << 20 | ch(3, 3) << 30;
                std::memcpy(o, &q, 4);
                break;
            }
            case UNSIGNED_INT_10F_11F_11F_REV: {
                auto uf = [&](double v) { return v != v || v <= 0 ? uint16_t(0) : floatToHalf(float(v)); };
                const uint32_t q = uint32_t(uf(t.v[0]) >> 4) | uint32_t(uf(t.v[1]) >> 4) << 11 | uint32_t(uf(t.v[2]) >> 5) << 22;
                std::memcpy(o, &q, 4);
                break;
            }
            default: return false;
        }
    }
    return true;
}

}  // namespace

bool readbackIsCopy(const FormatInfo& src, GLenum format, GLenum type) {
    return format == RGBA && type == UNSIGNED_BYTE && (src.vk == VK_FORMAT_R8G8B8A8_UNORM || src.vk == VK_FORMAT_R8G8B8A8_SRGB) &&
           src.swz[3] != VK_COMPONENT_SWIZZLE_ONE;  // (RGB8 is stored as RGBA: its alpha reads 1 whatever is in memory)
}

bool convertReadback(const FormatInfo& src, GLenum format, GLenum type, const uint8_t* in, uint32_t n, uint8_t* out) {
    const uint32_t outBytes = clientPixelBytes(format, type);
    if (!outBytes) return false;
    if (format == RGBA && type == UNSIGNED_BYTE) {
        switch (src.vk) {
            case VK_FORMAT_R8G8B8A8_UNORM:
            case VK_FORMAT_R8G8B8A8_SRGB:
                std::memcpy(out, in, size_t(n) * 4);
                // RGB8 / SRGB8 are stored as RGBA: the alpha they read is 1 whatever is in memory.
                if (src.swz[3] == VK_COMPONENT_SWIZZLE_ONE)
                    for (uint32_t i = 0; i < n; ++i) out[4 * i + 3] = 255;
                return true;
            case VK_FORMAT_B8G8R8A8_UNORM:
            case VK_FORMAT_B8G8R8A8_SRGB:
                for (uint32_t i = 0; i < n; ++i) {
                    out[4 * i] = in[4 * i + 2], out[4 * i + 1] = in[4 * i + 1], out[4 * i + 2] = in[4 * i], out[4 * i + 3] = in[4 * i + 3];
                }
                return true;
            case VK_FORMAT_R5G6B5_UNORM_PACK16:
                for (uint32_t i = 0; i < n; ++i) {
                    uint16_t p;
                    std::memcpy(&p, in + 2 * i, 2);
                    out[4 * i] = uint8_t((((p >> 11) & 31) * 255 + 15) / 31);  // unorm5/6 -> unorm8 rounds to nearest
                    out[4 * i + 1] = uint8_t((((p >> 5) & 63) * 255 + 31) / 63);
                    out[4 * i + 2] = uint8_t(((p & 31) * 255 + 15) / 31);
                    out[4 * i + 3] = 255;
                }
                return true;
            case VK_FORMAT_R4G4B4A4_UNORM_PACK16:
                for (uint32_t i = 0; i < n; ++i) {
                    uint16_t p;
                    std::memcpy(&p, in + 2 * i, 2);
                    for (int c = 0; c < 4; ++c) out[4 * i + c] = uint8_t(((p >> (12 - 4 * c)) & 15) * 17);
                }
                return true;
            case VK_FORMAT_R16G16B16A16_SFLOAT:
                for (uint32_t i = 0; i < n * 4; ++i) {
                    uint16_t h;
                    std::memcpy(&h, in + 2 * i, 2);
                    // half -> float -> unorm8
                    float v = halfToFloat(h);
                    v = !(v > 0) ? 0 : v > 1 ? 1 : v;                  // NaN and negatives read 0
                    out[i] = uint8_t(std::lround(v * 255.0f));
                }
                return true;
            case VK_FORMAT_R32G32B32A32_SFLOAT:
                for (uint32_t i = 0; i < n * 4; ++i) {
                    float v;
                    std::memcpy(&v, in + 4 * i, 4);
                    v = !(v > 0) ? 0 : v > 1 ? 1 : v;                  // NaN and negatives read 0
                    out[i] = uint8_t(std::lround(v * 255.0f));
                }
                return true;
            default: break;
        }
    }
    if (format == RGBA && type == FLOAT && src.vk == VK_FORMAT_R32G32B32A32_SFLOAT) {
        std::memcpy(out, in, size_t(n) * 16);
        return true;
    }
    if (format == RGBA_INTEGER && (type == INT || type == UNSIGNED_INT) && src.texelBytes == 16) {
        std::memcpy(out, in, size_t(n) * 16);
        return true;
    }
    if (convertReadbackGeneric(src, format, type, in, n, out)) return true;
    if (outBytes == src.texelBytes) {
        std::memcpy(out, in, size_t(n) * outBytes);
        return true;
    }
    return false;
}

}  // namespace vpgles
