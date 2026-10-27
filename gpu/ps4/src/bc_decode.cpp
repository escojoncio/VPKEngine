// CPU decoder of BC1-BC7 (see bc_decode.h): the per-block decoding is bc_block.h, the code the GPU kernel runs too.
#include "vpps4/bc_decode.h"

#include <algorithm>
#include <cstring>

#include "vpps4/bc_block.h"

namespace vpps4::bc {

using vpgpu::Format;

bool kindOf(uint8_t dataFormat, Kind& out) {
    if (dataFormat < 35 || dataFormat > 41) return false;
    out = Kind(dataFormat - 35);
    return true;
}

Format hostFormat(Kind k, bool srgb, bool isSigned) {
    switch (k) {
        case Kind::BC4: return isSigned ? Format::R8Snorm : Format::R8Unorm;
        case Kind::BC5: return isSigned ? Format::RG8Snorm : Format::RG8Unorm;
        case Kind::BC6H: return Format::RGBA16Float;
        default: return srgb ? Format::RGBA8Srgb : Format::RGBA8Unorm;
    }
}

Format layoutFormat(Kind k, bool srgb, bool isSigned) {
    switch (k) {
        case Kind::BC1: return srgb ? Format::BC1Srgb : Format::BC1;
        case Kind::BC2: return srgb ? Format::BC2Srgb : Format::BC2;
        case Kind::BC3: return srgb ? Format::BC3Srgb : Format::BC3;
        case Kind::BC4: return isSigned ? Format::BC4Snorm : Format::BC4Unorm;
        case Kind::BC5: return isSigned ? Format::BC5Snorm : Format::BC5Unorm;
        case Kind::BC6H: return isSigned ? Format::BC6HSfloat : Format::BC6HUfloat;
        default: return srgb ? Format::BC7Srgb : Format::BC7;
    }
}

unsigned texelBytes(Kind k) {
    switch (k) {
        case Kind::BC4: return 1;
        case Kind::BC5: return 2;
        case Kind::BC6H: return 8;
        default: return 4;
    }
}

void decodeBlock(Kind k, bool isSigned, const uint8_t* b, uint8_t* out) {
    const unsigned blockBytes = k == Kind::BC1 || k == Kind::BC4 ? 8 : 16, tb = texelBytes(k);
    uint32_t w[4] = {0, 0, 0, 0}, o[32];
    for (unsigned i = 0; i < blockBytes; ++i) w[i / 4] |= uint32_t(b[i]) << (8 * (i % 4));
    bc_decode_block(unsigned(k), isSigned ? 1u : 0u, w, o);
    for (unsigned t = 0; t < 16; ++t)
        for (unsigned j = 0; j < tb; ++j) out[t * tb + j] = uint8_t(o[2 * t + (j >> 2)] >> (8 * (j & 3)));
}

void decodeImage(Kind k, bool isSigned, const uint8_t* src, size_t srcRowBytes, uint32_t width, uint32_t height, uint8_t* dst,
                 size_t dstRowBytes) {
    const unsigned tb = texelBytes(k), blockBytes = k == Kind::BC1 || k == Kind::BC4 ? 8 : 16;
    uint8_t texels[16 * 8];
    for (uint32_t by = 0; by * 4 < height; ++by)
        for (uint32_t bx = 0; bx * 4 < width; ++bx) {
            decodeBlock(k, isSigned, src + by * srcRowBytes + bx * blockBytes, texels);
            const uint32_t w = std::min(4u, width - bx * 4), h = std::min(4u, height - by * 4);
            for (uint32_t y = 0; y < h; ++y)
                std::memcpy(dst + (by * 4 + y) * dstRowBytes + bx * 4 * tb, texels + y * 4 * tb, w * tb);
        }
}

}  // namespace vpps4::bc
