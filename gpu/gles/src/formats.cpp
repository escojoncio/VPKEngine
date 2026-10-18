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
                       VkComponentSwizzle b, VkComponentSwizzle a, GLenum ctype) {
            FormatInfo f;
            f.vk = vk, f.internal = internal, f.baseFormat = internal, f.texelBytes = bytes, f.componentType = ctype;
            f.swz[0] = r, f.swz[1] = g, f.swz[2] = b, f.swz[3] = a;
            v.push_back(f);
        };
        lum(LUMINANCE, VK_FORMAT_R8_UNORM, 1, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R,
            VK_COMPONENT_SWIZZLE_ONE, UNORM);
        lum(ALPHA, VK_FORMAT_R8_UNORM, 1, VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO,
            VK_COMPONENT_SWIZZLE_R, UNORM);
        lum(LUMINANCE_ALPHA, VK_FORMAT_R8G8_UNORM, 2, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R,
            VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, UNORM);
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
            // Float luminance (OES_texture_float / half_float) is not emulated: report unknown.
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

const FormatInfo* formatByVk(VkFormat f) {
    for (const FormatInfo& x : table())
        if (x.vk == f && !x.compressed) return &x;
    return nullptr;
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
    const uint32_t inBytes = clientPixelBytes(format, type);
    if (!inBytes) return false;
    out.resize(size_t(n) * dst.texelBytes);
    // Same layout: copy.
    const bool rgbToRgba = (format == RGB || format == RGB_INTEGER) && dst.texelBytes == inBytes / 3 * 4 &&
                           (type == UNSIGNED_BYTE || type == BYTE || type == FLOAT || type == HALF_FLOAT ||
                            type == HALF_FLOAT_OES || type == UNSIGNED_SHORT || type == SHORT || type == INT ||
                            type == UNSIGNED_INT);
    if (rgbToRgba) {
        const uint32_t cb = inBytes / 3;
        for (uint32_t i = 0; i < n; ++i) {
            uint8_t* d = &out[size_t(i) * dst.texelBytes];
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
    if (inBytes == dst.texelBytes) {
        if (type == UNSIGNED_SHORT_5_6_5 && dst.vk == VK_FORMAT_R5G6B5_UNORM_PACK16) {
            std::memcpy(out.data(), src, out.size());
            return true;
        }
        if (type == UNSIGNED_SHORT_4_4_4_4 && dst.vk == VK_FORMAT_R4G4B4A4_UNORM_PACK16) {
            std::memcpy(out.data(), src, out.size());  // same bit order (R in 15..12)
            return true;
        }
        if (type == UNSIGNED_SHORT_5_5_5_1 && dst.vk == VK_FORMAT_A1R5G5B5_UNORM_PACK16) {
            // GL: R 15..11 G 10..6 B 5..1 A 0 -> Vulkan A1R5G5B5: A 15, R 14..10, G 9..5, B 4..0.
            for (uint32_t i = 0; i < n; ++i) {
                uint16_t p;
                std::memcpy(&p, src + 2 * i, 2);
                uint16_t q = uint16_t((p & 1) << 15 | (p >> 1));
                std::memcpy(&out[2 * i], &q, 2);
            }
            return true;
        }
        if (type == UNSIGNED_INT_24_8 && dst.vk == VK_FORMAT_D24_UNORM_S8_UINT) {
            // GL: depth in 31..8, stencil 7..0; the null/Metal D24S8 upload takes depth in 23..0.
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t p;
                std::memcpy(&p, src + 4 * i, 4);
                uint32_t q = (p >> 8) | (p & 0xFF) << 24;
                std::memcpy(&out[4 * i], &q, 4);
            }
            return true;
        }
        std::memcpy(out.data(), src, out.size());
        return true;
    }
    // 8-bit RGBA into 16-bit packed targets (GLES 2 apps uploading RGBA bytes to an RGBA4 texture).
    if (type == UNSIGNED_BYTE && format == RGBA && dst.vk == VK_FORMAT_R4G4B4A4_UNORM_PACK16) {
        for (uint32_t i = 0; i < n; ++i) {
            const uint8_t* p = src + 4 * i;
            uint16_t q = uint16_t((p[0] >> 4) << 12 | (p[1] >> 4) << 8 | (p[2] >> 4) << 4 | (p[3] >> 4));
            std::memcpy(&out[2 * i], &q, 2);
        }
        return true;
    }
    if (type == UNSIGNED_BYTE && format == RGB && dst.vk == VK_FORMAT_R5G6B5_UNORM_PACK16) {
        for (uint32_t i = 0; i < n; ++i) {
            const uint8_t* p = src + 3 * i;
            uint16_t q = uint16_t((p[0] >> 3) << 11 | (p[1] >> 2) << 5 | (p[2] >> 3));
            std::memcpy(&out[2 * i], &q, 2);
        }
        return true;
    }
    if (type == UNSIGNED_INT && format == DEPTH_COMPONENT && dst.vk == VK_FORMAT_D24_UNORM_S8_UINT) {
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t p;
            std::memcpy(&p, src + 4 * i, 4);
            uint32_t q = p >> 8;
            std::memcpy(&out[4 * i], &q, 4);
        }
        return true;
    }
    if (type == UNSIGNED_SHORT && format == DEPTH_COMPONENT && dst.vk == VK_FORMAT_D24_UNORM_S8_UINT) {
        for (uint32_t i = 0; i < n; ++i) {
            uint16_t p;
            std::memcpy(&p, src + 2 * i, 2);
            uint32_t q = uint32_t(p) * 0xFFFFFFu / 0xFFFFu;
            std::memcpy(&out[4 * i], &q, 4);
        }
        return true;
    }
    return false;
}

bool convertReadback(const FormatInfo& src, GLenum format, GLenum type, const uint8_t* in, uint32_t n, uint8_t* out) {
    const uint32_t outBytes = clientPixelBytes(format, type);
    if (!outBytes) return false;
    if (format == RGBA && type == UNSIGNED_BYTE) {
        switch (src.vk) {
            case VK_FORMAT_R8G8B8A8_UNORM:
            case VK_FORMAT_R8G8B8A8_SRGB: std::memcpy(out, in, size_t(n) * 4); return true;
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
                    out[4 * i] = uint8_t(((p >> 11) & 31) * 255 / 31);
                    out[4 * i + 1] = uint8_t(((p >> 5) & 63) * 255 / 63);
                    out[4 * i + 2] = uint8_t((p & 31) * 255 / 31);
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
                    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
                    float v = e == 0 ? std::ldexp(float(m), -24) : e == 31 ? 1e30f : std::ldexp(float(m | 1024), int(e) - 25);
                    if (s) v = -v;
                    v = v < 0 ? 0 : v > 1 ? 1 : v;
                    out[i] = uint8_t(std::lround(v * 255.0f));
                }
                return true;
            case VK_FORMAT_R32G32B32A32_SFLOAT:
                for (uint32_t i = 0; i < n * 4; ++i) {
                    float v;
                    std::memcpy(&v, in + 4 * i, 4);
                    v = v < 0 ? 0 : v > 1 ? 1 : v;
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
    if (outBytes == src.texelBytes) {
        std::memcpy(out, in, size_t(n) * outBytes);
        return true;
    }
    return false;
}

}  // namespace vpgles
