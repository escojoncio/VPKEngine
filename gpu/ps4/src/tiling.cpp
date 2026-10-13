// PS4 surface tiling (see tiling.h). The address of an element of a 2D tiled surface is built as
// the hardware does: an offset inside the element's 256-byte pipe interleave chunk, then the
// pipe bits, then the bank bits, then the offset of the chunk inside its pipe and bank.
#include "vpps4/tiling.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>

namespace vpps4 {
namespace {

enum class Array : uint8_t { Linear, Tiled1D, Tiled2D, Other };
enum class Micro : uint8_t { Display, Thin, Depth, Other };
enum class Pipes : uint8_t { P8_32x32_16x16, P8_32x32_8x16, P2 };

struct Mode {
    Array array;
    Micro micro;
    Pipes pipes;
    uint8_t sampleSplit;
    uint16_t tileSplitHw;  // bytes (depth micro tiles)
    const char* name;
};

// GB_TILE_MODE0..31 as the PS4 programs them (base mode).
constexpr Mode kModes[32] = {
    {Array::Tiled2D, Micro::Depth, Pipes::P8_32x32_16x16, 1, 64, "depth_2d_thin_64"},
    {Array::Tiled2D, Micro::Depth, Pipes::P8_32x32_16x16, 1, 128, "depth_2d_thin_128"},
    {Array::Tiled2D, Micro::Depth, Pipes::P8_32x32_16x16, 1, 256, "depth_2d_thin_256"},
    {Array::Tiled2D, Micro::Depth, Pipes::P8_32x32_16x16, 1, 512, "depth_2d_thin_512"},
    {Array::Tiled2D, Micro::Depth, Pipes::P8_32x32_16x16, 1, 1024, "depth_2d_thin_1k"},
    {Array::Tiled1D, Micro::Depth, Pipes::P8_32x32_16x16, 1, 64, "depth_1d_thin"},
    {Array::Other, Micro::Depth, Pipes::P8_32x32_16x16, 1, 256, "depth_2d_thin_prt_256"},
    {Array::Other, Micro::Depth, Pipes::P8_32x32_16x16, 1, 1024, "depth_2d_thin_prt_1k"},
    {Array::Linear, Micro::Display, Pipes::P8_32x32_16x16, 1, 64, "display_linear_aligned"},
    {Array::Tiled1D, Micro::Display, Pipes::P8_32x32_16x16, 1, 64, "display_1d_thin"},
    {Array::Tiled2D, Micro::Display, Pipes::P8_32x32_16x16, 2, 64, "display_2d_thin"},
    {Array::Other, Micro::Display, Pipes::P8_32x32_8x16, 2, 64, "display_thin_prt"},
    {Array::Other, Micro::Display, Pipes::P8_32x32_16x16, 2, 64, "display_2d_thin_prt"},
    {Array::Tiled1D, Micro::Thin, Pipes::P8_32x32_16x16, 1, 64, "thin_1d_thin"},
    {Array::Tiled2D, Micro::Thin, Pipes::P8_32x32_16x16, 2, 64, "thin_2d_thin"},
    {Array::Other, Micro::Thin, Pipes::P8_32x32_8x16, 2, 64, "thin_3d_thin"},
    {Array::Other, Micro::Thin, Pipes::P8_32x32_8x16, 2, 64, "thin_thin_prt"},
    {Array::Other, Micro::Thin, Pipes::P8_32x32_16x16, 2, 64, "thin_2d_thin_prt"},
    {Array::Other, Micro::Thin, Pipes::P8_32x32_16x16, 2, 64, "thin_3d_thin_prt"},
    {Array::Other, Micro::Other, Pipes::P8_32x32_16x16, 1, 64, "thick_1d_thick"},
    {Array::Other, Micro::Other, Pipes::P8_32x32_16x16, 1, 64, "thick_2d_thick"},
    {Array::Other, Micro::Other, Pipes::P8_32x32_8x16, 1, 64, "thick_3d_thick"},
    {Array::Other, Micro::Other, Pipes::P8_32x32_8x16, 1, 64, "thick_thick_prt"},
    {Array::Other, Micro::Other, Pipes::P8_32x32_16x16, 1, 64, "thick_2d_thick_prt"},
    {Array::Other, Micro::Other, Pipes::P8_32x32_8x16, 1, 64, "thick_3d_thick_prt"},
    {Array::Other, Micro::Other, Pipes::P8_32x32_16x16, 1, 64, "thick_2d_xthick"},
    {Array::Other, Micro::Other, Pipes::P8_32x32_8x16, 1, 64, "thick_3d_xthick"},
    {Array::Other, Micro::Other, Pipes::P2, 1, 64, "unused_27"},
    {Array::Other, Micro::Other, Pipes::P2, 1, 64, "unused_28"},
    {Array::Other, Micro::Other, Pipes::P2, 1, 64, "unused_29"},
    {Array::Other, Micro::Other, Pipes::P2, 1, 64, "unused_30"},
    {Array::Linear, Micro::Display, Pipes::P2, 1, 64, "display_linear_general"},
};

// GB_MACROTILE_MODE0..7 (base mode): bank height, banks, macro tile aspect; bank width is 1.
struct Banks {
    uint8_t height, count, aspect;
};
constexpr Banks kBanks[8] = {{4, 16, 4}, {2, 16, 2}, {1, 16, 2}, {1, 16, 2}, {1, 8, 1}, {1, 4, 1}, {1, 2, 1}, {1, 2, 1}};

constexpr uint32_t kMicro = 8;              // micro tile edge, elements
constexpr uint32_t kPipeInterleave = 256;   // bytes
constexpr uint32_t kPipeInterleaveBits = 8;
constexpr uint32_t kPipesCount = 8;
constexpr uint32_t kPipeBits = 3;

inline uint32_t bit(uint32_t v, unsigned b) { return (v >> b) & 1u; }

// Index of element (x, y) inside its 8x8 micro tile, by micro tile kind and element size.
uint32_t microIndex(Micro micro, uint32_t bpe, uint32_t x, uint32_t y) {
    const uint32_t x0 = bit(x, 0), x1 = bit(x, 1), x2 = bit(x, 2);
    const uint32_t y0 = bit(y, 0), y1 = bit(y, 1), y2 = bit(y, 2);
    std::array<uint32_t, 6> b{};
    if (micro == Micro::Display) {
        switch (bpe) {
            case 1: b = {x0, x1, x2, y1, y0, y2}; break;
            case 2: b = {x0, x1, x2, y0, y1, y2}; break;
            case 4: b = {x0, x1, y0, x2, y1, y2}; break;
            case 8: b = {x0, y0, x1, x2, y1, y2}; break;
            default: b = {y0, x0, x1, x2, y1, y2}; break;  // 16
        }
    } else {  // thin and depth order elements the same way (Z order)
        b = {x0, y0, x1, y1, x2, y2};
    }
    uint32_t i = 0;
    for (unsigned k = 0; k < 6; ++k) i |= b[k] << k;
    return i;
}

uint32_t pipeOf(Pipes p, uint32_t x, uint32_t y) {
    const uint32_t x3 = bit(x, 3), x4 = bit(x, 4), x5 = bit(x, 5);
    const uint32_t y3 = bit(y, 3), y4 = bit(y, 4), y5 = bit(y, 5);
    if (p == Pipes::P8_32x32_8x16) return (x4 ^ y3 ^ x5) | (x3 ^ y4) << 1 | (x5 ^ y5) << 2;
    return (x3 ^ y3 ^ x4) | (x4 ^ y4) << 1 | (x5 ^ y5) << 2;  // P8_32x32_16x16
}

uint32_t bankOf(uint32_t banks, uint32_t bankHeight, uint32_t x, uint32_t y) {
    const uint32_t tx = x / kMicro / kPipesCount;  // bank width 1
    const uint32_t ty = y / kMicro / bankHeight;
    const uint32_t x3 = bit(tx, 0), x4 = bit(tx, 1), x5 = bit(tx, 2), x6 = bit(tx, 3);
    const uint32_t y3 = bit(ty, 0), y4 = bit(ty, 1), y5 = bit(ty, 2), y6 = bit(ty, 3);
    switch (banks) {
        case 16: return (x3 ^ y6) | (x4 ^ y5 ^ y6) << 1 | (x5 ^ y4) << 2 | (x6 ^ y3) << 3;
        case 8: return (x3 ^ y5) | (x4 ^ y4 ^ y5) << 1 | (x5 ^ y3) << 2;
        case 4: return (x3 ^ y4) | (x4 ^ y3) << 1;
        default: return x3 ^ y3;
    }
}

// Everything about a surface's layout that does not depend on the element.
struct Layout {
    Array array = Array::Other;
    Micro micro = Micro::Other;
    Pipes pipes = Pipes::P8_32x32_16x16;
    uint32_t bpe = 0, pitch = 0, height = 0;  // padded
    uint64_t sliceBytes = 0;
    // 2D
    Banks banks{};
    uint32_t macroPitch = 0, macroHeight = 0, macroPerRow = 0;
    uint64_t macroBytes = 0;
    std::string why;
};

uint32_t alignUp(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

Layout layoutOf(const TiledSurface& s) {
    Layout l;
    if (s.tileIndex >= 32) {
        l.why = "tile mode index " + std::to_string(s.tileIndex) + " out of range";
        return l;
    }
    const Mode& m = kModes[s.tileIndex];
    const uint32_t bpe = s.bytesPerElement;
    if (bpe != 1 && bpe != 2 && bpe != 4 && bpe != 8 && bpe != 16) {
        l.why = std::to_string(bpe) + " bytes per element";
        return l;
    }
    if (m.array == Array::Other || m.micro == Micro::Other) {
        l.why = std::string("tile mode ") + m.name + " (thick, 3D or PRT) not covered";
        return l;
    }
    l.array = m.array;
    l.micro = m.micro;
    l.pipes = m.pipes;
    l.bpe = bpe;
    const uint32_t pitch = std::max(s.pitch, s.width);
    if (l.array == Array::Linear) {
        l.pitch = pitch;
        l.height = s.height;
        l.sliceBytes = uint64_t(l.pitch) * l.height * bpe;
        return l;
    }
    const uint32_t tileBytes = kMicro * kMicro * bpe;
    if (l.array == Array::Tiled2D) {
        uint32_t split = m.micro == Micro::Depth ? m.tileSplitHw : std::max(256u, m.sampleSplit * tileBytes);
        split = std::min(1024u, split);
        if (split < tileBytes) {
            l.why = std::string("tile mode ") + m.name + " splits a micro tile of " + std::to_string(tileBytes) +
                    " bytes (" + std::to_string(split) + "-byte split) - not covered";
            return l;
        }
        const unsigned mtm = unsigned(std::bit_width(std::min(split, tileBytes) / 64u)) - 1u;
        l.banks = kBanks[std::min(mtm, 7u)];
        l.macroPitch = kMicro * kPipesCount * l.banks.aspect;
        l.macroHeight = kMicro * l.banks.height * l.banks.count / l.banks.aspect;
        // Too small for one macro tile: laid out as 1D thin.
        if (pitch < l.macroPitch || s.height < l.macroHeight) l.array = Array::Tiled1D;
    }
    if (l.array == Array::Tiled1D) {
        l.pitch = alignUp(pitch, kMicro);
        l.height = alignUp(s.height, kMicro);
        l.sliceBytes = uint64_t(l.pitch) * l.height * bpe;
        return l;
    }
    l.pitch = alignUp(pitch, l.macroPitch);
    l.height = alignUp(s.height, l.macroHeight);
    l.macroPerRow = l.pitch / l.macroPitch;
    l.macroBytes = uint64_t(l.macroPitch) * l.macroHeight * bpe;
    l.sliceBytes = uint64_t(l.macroPerRow) * (l.height / l.macroHeight) * l.macroBytes;
    return l;
}

uint64_t offsetIn(const Layout& l, uint32_t x, uint32_t y, uint32_t slice) {
    const uint32_t bpe = l.bpe;
    if (l.array == Array::Linear) return slice * l.sliceBytes + (uint64_t(y) * l.pitch + x) * bpe;
    const uint64_t tileBytes = kMicro * kMicro * bpe;
    const uint64_t element = uint64_t(microIndex(l.micro, bpe, x % kMicro, y % kMicro)) * bpe;
    if (l.array == Array::Tiled1D) {
        const uint64_t tile = uint64_t(y / kMicro) * (l.pitch / kMicro) + x / kMicro;
        return slice * l.sliceBytes + tile * tileBytes + element;
    }
    // 2D thin. Offsets that cover all pipes and banks are divided among them; the micro tile's
    // place among those of its bank (bank width 1: a column of `bank height` tiles) and the
    // element's place in it are per pipe and bank.
    const uint64_t macro = uint64_t(y / l.macroHeight) * l.macroPerRow + x / l.macroPitch;
    const uint64_t spread = slice * l.sliceBytes + macro * l.macroBytes;
    const uint32_t tileRow = (y / kMicro) % l.banks.height;
    const uint64_t local = (spread >> (kPipeBits + std::countr_zero(uint32_t(l.banks.count)))) + tileRow * tileBytes + element;
    uint32_t bank = bankOf(l.banks.count, l.banks.height, x, y);
    // Slices of an array rotate their banks.
    if (slice) bank ^= (slice * (std::max(1u, uint32_t(l.banks.count) / 2u - 1u))) % l.banks.count;
    const uint32_t pipe = pipeOf(l.pipes, x, y);
    const unsigned bankBits = unsigned(std::countr_zero(uint32_t(l.banks.count)));
    return (local & (kPipeInterleave - 1)) | uint64_t(pipe) << kPipeInterleaveBits |
           uint64_t(bank) << (kPipeInterleaveBits + kPipeBits) |
           (local >> kPipeInterleaveBits) << (kPipeInterleaveBits + kPipeBits + bankBits);
}

template <bool kToLinear>
bool copy(const TiledSurface& s, const uint8_t* src, uint8_t* dst, std::string& why) {
    const Layout l = layoutOf(s);
    if (!l.why.empty()) {
        why = l.why;
        return false;
    }
    const uint32_t bpe = l.bpe;
    const uint64_t rowBytes = uint64_t(s.width) * bpe;
    for (uint32_t z = 0; z < s.slices; ++z)
        for (uint32_t y = 0; y < s.height; ++y) {
            uint8_t* linear = (kToLinear ? dst : const_cast<uint8_t*>(src)) + (uint64_t(z) * s.height + y) * rowBytes;
            if (l.array == Array::Linear) {
                const uint64_t o = offsetIn(l, 0, y, z);
                if constexpr (kToLinear) std::memcpy(linear, src + o, rowBytes);
                else std::memcpy(dst + o, linear, rowBytes);
                continue;
            }
            for (uint32_t x = 0; x < s.width; ++x) {
                const uint64_t o = offsetIn(l, x, y, z);
                if constexpr (kToLinear) std::memcpy(linear + uint64_t(x) * bpe, src + o, bpe);
                else std::memcpy(dst + o, linear + uint64_t(x) * bpe, bpe);
            }
        }
    return true;
}

}  // namespace

uint64_t tiledBytes(const TiledSurface& s, std::string& why) {
    const Layout l = layoutOf(s);
    if (!l.why.empty()) {
        why = l.why;
        return 0;
    }
    return l.sliceBytes * s.slices;
}

uint64_t tiledOffset(const TiledSurface& s, uint32_t x, uint32_t y, uint32_t slice) {
    const Layout l = layoutOf(s);
    return l.why.empty() ? offsetIn(l, x, y, slice) : UINT64_MAX;
}

bool detile(const TiledSurface& s, const uint8_t* src, uint8_t* dst, std::string& why) {
    return copy<true>(s, src, dst, why);
}

bool tile(const TiledSurface& s, const uint8_t* src, uint8_t* dst, std::string& why) {
    return copy<false>(s, src, dst, why);
}

}  // namespace vpps4
