// GL internal formats / pixel transfer formats -> VkFormat, and the CPU conversions the transfer
// paths need (luminance/alpha expansion, 5-6-5 / 4-4-4-4 / 5-5-5-1 repacking, RGB8 -> RGBA8).
#pragma once

#include <cstdint>
#include <vector>

#include "gl_enums.h"
#include "vk.h"

namespace vpgles {

struct FormatInfo {
    VkFormat vk = VK_FORMAT_UNDEFINED;
    GLenum internal = 0;     // sized internal format
    GLenum baseFormat = 0;   // GL_RGBA, GL_DEPTH_COMPONENT...
    uint8_t texelBytes = 0;  // bytes per texel of `vk` (compressed: per block)
    uint8_t blockW = 1, blockH = 1;
    bool depth = false, stencil = false, integer = false, compressed = false, srgb = false, renderable = false;
    // Swizzle applied when sampling (luminance/alpha emulation): VkComponentSwizzle per r,g,b,a.
    VkComponentSwizzle swz[4] = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                 VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
    uint8_t redBits = 0, greenBits = 0, blueBits = 0, alphaBits = 0, depthBits = 0, stencilBits = 0;
    GLenum componentType = 0;  // GL_UNSIGNED_NORMALIZED (0x8C17), GL_FLOAT, GL_INT, GL_UNSIGNED_INT
};

// Sized or unsized internal format (unsized ones resolved with the transfer type, as GLES 2 does).
// Returns null for unknown formats.
const FormatInfo* formatFor(GLenum internalFormat, GLenum format = 0, GLenum type = 0);
const FormatInfo* formatByVk(VkFormat f);

// The error glTexImage / glTexSubImage raise for (format, type) against the texture's format `f`: 0 = none,
// GL_INVALID_ENUM = not a pixel format or type at all, GL_INVALID_OPERATION = a valid pair that does not suit `f` (ES 3.x
// table 3.2, plus the OES/EXT float, half float, BGRA and norm16 uploads vpgles takes). `sub`: a sub-image upload may use
// another colour format of the same class (RGB data into an RGBA8 texture), as drivers allow.
GLenum transferError(const FormatInfo& f, GLenum format, GLenum type, bool sub = false);

// Bytes of one pixel of client data in (format, type); 0 if the combination is invalid.
uint32_t clientPixelBytes(GLenum format, GLenum type);

// Converts `w*h*d` client pixels (rows already de-padded) of (format, type) to the layout of
// `dst` (tightly packed). Returns false when no conversion is known.
bool convertUpload(const FormatInfo& dst, GLenum format, GLenum type, const uint8_t* src, uint32_t pixels,
                   std::vector<uint8_t>& out);
// The same into `out`, which holds pixels * dst.texelBytes bytes (every one is written when this returns true).
bool convertUploadTo(const FormatInfo& dst, GLenum format, GLenum type, const uint8_t* src, uint32_t pixels, uint8_t* out);
// Whether convertReadback(src, format, type) leaves the texels as they are (the pixels are the bytes the image holds).
bool readbackIsCopy(const FormatInfo& src, GLenum format, GLenum type);
// Converts tightly packed texels of `src` into client (format, type) pixels (glReadPixels).
bool convertReadback(const FormatInfo& src, GLenum format, GLenum type, const uint8_t* in, uint32_t pixels,
                     uint8_t* out);

// Compressed formats: bytes for a w x h image.
uint32_t compressedSize(const FormatInfo& f, uint32_t w, uint32_t h);

}  // namespace vpgles
