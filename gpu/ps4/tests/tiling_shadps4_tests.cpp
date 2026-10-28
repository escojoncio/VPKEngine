// Tiling against shadPS4's own implementation (src/video_core of shadPS4, GPL-2.0-or-later), transcribed
// below into namespace sref as literally as C++ allows: the size and mip offsets of
// texture_cache/image_info.cpp ImageInfo::UpdateSize with texture_cache/tile.h ImageSizeLinearAligned /
// ImageSizeMicroTiled / ImageSizeMacroTiled (and its macro tile extent tables, shadps4_extents.inc),
// the mode helpers of amdgpu/tiling.cpp, and the address of host_shaders/tiling.comp (the detiler
// shader, one thread per texel of the guest span).
//
// What it checks, for every thin / display / depth tile mode that is linear, 1D or macro tiled (mode
// 18 and the thick modes are left out, see below), element sizes 1..16 bytes, block size 1 and 4,
// sizes from 4 to 256 (not power of two, below one macro tile on level 0, pitch beyond the width),
// 1..9 mip levels, 1 and 6 array slices (and volumes), POW2_PAD on and off:
//   * the pitch, the rows, the size and the offset of every mip level, and the total span;
//   * which levels are stored 1D tiled (shadPS4 only degrades mip levels > 0: level 0 of a surface smaller
//     than a macro tile is still macro tiled and padded to a whole macro tile);
//   * the byte offset of every element of every level: VPKEngine's against the shader's;
//   * detileLevels and tileLevels byte for byte (a pattern different in every byte of a 256-byte block)
//     against the same copy done with the shader's addresses, and the round trip.
// Known differences, deliberately not compared:
//   * shadPS4 maps mode 18 (Thin3DThinPrt) to the non-PRT 3D array mode but sizes it with the PRT row of its
//     extent table, so it contradicts itself; the thick modes size a layer array differently (shadPS4
//     rounds 1 slice up to the thickness and multiplies by the layers).
//   * shadPS4's shader addresses a degraded mip level (pitch below one macro tile) with the macro tiled
//     function of the whole image, which cannot work (zero macro tiles per row); the reference here uses the
//     shader's own 1D tiled function for those levels, which is what the size function describes.
//   * a level 0 pitch that the 1D / linear padding loop would grow: VPKEngine keeps the descriptor's pitch
//     (the hardware does), shadPS4 re-pads it. The descriptor pitches used here are first made stable under
//     that loop, as a real allocation's are.
#include <algorithm>
#include <bit>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "vpps4/tiling.h"

using namespace vpps4;

namespace sref {
using u32 = uint32_t;

#include "shadps4_extents.inc"

// ---- amdgpu/tiling.cpp ------------------------------------------------------------------------------
enum Array : u32 {
    LinearGeneral, LinearAligned, A1DThin, A1DThick, A2DThin, PrtThin, Prt2DThin, A2DThick, A2DXThick, PrtThick, Prt2DThick,
    Prt3DThin, A3DThin, A3DThick, A3DXThick, Prt3DThick
};
enum Micro : u32 { MDisplay = 0, MThin = 1, MDepth = 2, MRotated = 3, MThick = 4 };
constexpr u32 P2 = 0, P8_8x16 = 10, P8_16x16 = 12;  // ADDR_SURF_* of tiling.comp

inline Array arrayMode(u32 t) {  // GetArrayMode(TileMode)
    switch (t) {
        case 5: case 9: case 13: return A1DThin;
        case 0: case 1: case 2: case 3: case 4: case 10: case 14: return A2DThin;
        case 11: case 16: return PrtThin;
        case 6: case 7: case 12: case 17: return Prt2DThin;
        case 15: case 18: return A3DThin;
        case 19: return A1DThick;
        case 20: return A2DThick;
        case 21: return A3DThick;
        case 22: return PrtThick;
        case 23: return Prt2DThick;
        case 24: return Prt3DThick;
        case 25: return A2DXThick;
        case 26: return A3DXThick;
        case 8: return LinearAligned;
        default: return LinearGeneral;
    }
}
inline Micro microMode(u32 t) {  // GetMicroTileMode
    if (t <= 7) return MDepth;
    if (t <= 12 || t == 31) return MDisplay;
    if (t <= 18) return MThin;
    return MThick;
}
inline u32 pipeConfig(u32 t) {  // GetPipeConfig
    switch (t) {
        case 11: case 15: case 16: case 21: case 22: case 24: case 26: return P8_8x16;
        case 31: return P2;
        default: return P8_16x16;
    }
}
inline u32 sampleSplit(u32 t) {  // GetSampleSplit
    switch (t) {
        case 10: case 11: case 12: case 14: case 15: case 16: case 17: case 18: return 2;
        default: return 1;
    }
}
inline u32 tileSplitHw(u32 t) {  // GetTileSplitHw
    switch (t) {
        case 1: return 128;
        case 2: case 6: return 256;
        case 3: return 512;
        case 4: case 7: return 1024;
        default: return 64;
    }
}
inline u32 bankHeight(u32 m) {  // GetBankHeight(MacroTileMode)
    switch (m) {
        case 0: case 9: return 4;
        case 1: case 10: return 2;
        case 8: return 8;
        default: return 1;
    }
}
inline u32 numBanks(u32 m) {  // GetNumBanks
    switch (m) {
        case 6: case 7: case 14: case 15: return 2;
        case 5: case 13: return 4;
        case 4: case 12: return 8;
        default: return 16;
    }
}
inline u32 macroAspect(u32 m) {  // GetMacrotileAspect
    switch (m) {
        case 4: case 5: case 6: case 7: case 12: case 13: case 14: case 15: return 1;
        case 1: case 2: case 3: case 10: case 11: return 2;
        default: return 4;
    }
}
inline bool isMacro(Array a) { return a != LinearGeneral && a != LinearAligned && a != A1DThin && a != A1DThick; }
inline bool isPrt(Array a) { return a == PrtThin || a == PrtThick || a == Prt2DThin || a == Prt2DThick || a == Prt3DThin || a == Prt3DThick; }
inline u32 thickness(Array a) {  // GetMicroTileThickness
    switch (a) {
        case A1DThick: case A2DThick: case A3DThick: case PrtThick: case Prt2DThick: case Prt3DThick: return 4;
        case A2DXThick: case A3DXThick: return 8;
        default: return 1;
    }
}
inline u32 calcTileSplit(u32 t, Array a, Micro mm, u32 bpp) {  // CalculateTileSplit
    const u32 tileBytes1x = (bpp * 8 * 8 * thickness(a) + 7) / 8;
    const u32 color = std::max(256u, sampleSplit(t) * tileBytes1x);
    return std::min(1024u, mm == MDepth ? tileSplitHw(t) : color);
}
inline u32 macroTileMode(u32 t, u32 bpp, u32 samples) {  // CalculateMacrotileMode
    const Array a = arrayMode(t);
    const u32 tileBytes1x = bpp * 8 * 8 * thickness(a) / 8;
    const u32 tileBytes = std::min(calcTileSplit(t, a, microMode(t), bpp), samples * tileBytes1x);
    const u32 idx = uint32_t(std::bit_width(tileBytes / 64)) - 1;
    return isPrt(a) ? idx + 8 : idx;
}

// ---- texture_cache/tile.h ---------------------------------------------------------------------------
inline void macroExtents(u32 t, u32 bpp, u32 samples, u32& w, u32& h) {  // GetMacroTileExtents (not alt)
    const u32 log = samples == 1 ? 0 : samples == 2 ? 1 : samples == 4 ? 2 : 3;
    const u32 i = t * 5 + (uint32_t(std::bit_width(bpp)) - 4);
    const uint32_t (*tab)[2] = log == 0 ? kExtentsX1 : log == 1 ? kExtentsX2 : log == 2 ? kExtentsX4 : kExtentsX8;
    w = tab[i][0], h = tab[i][1];
}
struct Size3 { u32 pitch, height; size_t size; };
inline Size3 sizeLinearAligned(u32 pitch, u32 height, u32 bpp, u32 samples) {
    const u32 pitchAlign = std::max(8u, 64u / ((bpp + 7) / 8));
    u32 pa = (pitch + pitchAlign - 1) & ~(pitchAlign - 1);
    size_t log = size_t(pa) * height * samples;
    const u32 sliceAlign = std::max(64u, 256u / ((bpp + 7) / 8));
    while (log % sliceAlign) {
        pa += pitchAlign;
        log = size_t(pa) * height * samples;
    }
    return {pa, height, (log * bpp + 7) / 8};
}
inline Size3 sizeMicroTiled(u32 pitch, u32 height, u32 thick, u32 bpp, u32 samples) {
    u32 pa = (pitch + 7) & ~7u;
    const u32 ha = (height + 7) & ~7u;
    size_t log = (size_t(pa) * ha * bpp * samples + 7) / 8;
    while ((log * thick) % 256) {
        pa += 8;
        log = (size_t(pa) * ha * bpp * samples + 7) / 8;
    }
    return {pa, ha, log};
}
inline Size3 sizeMacroTiled(u32 pitch, u32 height, u32 thick, u32 bpp, u32 samples, u32 t, u32 mipN, bool& downgraded) {
    u32 pitchAlign, heightAlign;
    macroExtents(t, bpp, samples, pitchAlign, heightAlign);
    downgraded = false;
    if (mipN > 0) downgraded = pitch < pitchAlign || height < heightAlign;  // only mip levels > 0
    if (downgraded) return sizeMicroTiled(pitch, height, thick, bpp, samples);
    const u32 pa = (pitch + pitchAlign - 1) & ~(pitchAlign - 1);
    const u32 ha = (height + heightAlign - 1) & ~(heightAlign - 1);
    const size_t log = size_t(pa) * ha * samples;
    return {pa, ha, (log * bpp + 7) / 8};
}

// ---- texture_cache/image_info.cpp -------------------------------------------------------------------
struct Image {
    u32 tile = 14, pitch = 0, width = 1, height = 1, depth = 1, layers = 1, levels = 1, bits = 32, samples = 1, bankSwizzle = 0;
    bool block = false, pow2 = false, volume = false;
};
struct Mip {
    u32 size = 0, pitch = 0, height = 0, offset = 0;  // MipInfo (pitch, height before the block multiply)
    bool downgraded = false;
};
struct Chain {
    std::vector<Mip> mips;
    u32 guestSize = 0;
};
inline Chain updateSize(const Image& im) {  // ImageInfo::UpdateSize
    Chain c;
    const Array am = arrayMode(im.tile);
    for (u32 mip = 0; mip < im.levels; ++mip) {
        u32 mw = im.pitch >> mip, mh = im.height >> mip;
        if (im.block) mw = (mw + 3) / 4, mh = (mh + 3) / 4;
        mw = std::max(mw, 1u);
        mh = std::max(mh, 1u);
        u32 md = std::max((im.volume ? im.depth : 1u) >> mip, 1u);
        u32 thick = 1;
        if (im.pow2) mw = std::bit_ceil(mw), mh = std::bit_ceil(mh), md = std::bit_ceil(md);
        Mip m;
        switch (am) {
            case LinearAligned: {
                const Size3 s = sizeLinearAligned(mw, mh, im.bits, im.samples);
                m.pitch = s.pitch, m.height = s.height, m.size = u32(s.size);
                break;
            }
            case A1DThick:
                thick = 4;
                md += (0u - md) & (thick - 1);
                [[fallthrough]];
            case A1DThin: {
                const Size3 s = sizeMicroTiled(mw, mh, thick, im.bits, im.samples);
                m.pitch = s.pitch, m.height = s.height, m.size = u32(s.size);
                break;
            }
            default: {  // A2DThin and the other macro tiled thin modes (the real switch stops at 2D)
                const Size3 s = sizeMacroTiled(mw, mh, thick, im.bits, im.samples, im.tile, mip, m.downgraded);
                m.pitch = s.pitch, m.height = s.height, m.size = u32(s.size);
            }
        }
        m.size *= md * im.layers;
        m.offset = c.guestSize;
        c.guestSize += m.size;
        c.mips.push_back(m);
    }
    return c;
}

// ---- host_shaders/tiling.comp (one mip level's texel -> byte offset, sample 0) ------------------------
struct Defs {
    u32 bpp, samples, array, micro, thick, pipeCfg, bankH, banks, tileSplit, aspect, bankSwizzle;
};
inline u32 bitOf(u32 v, u32 b) { return (v >> b) & 1; }
inline u32 pixelIndex(const Defs& d, u32 x, u32 y, u32 z) {  // ComputePixelIndexWithinMicroTile
    u32 p[9] = {};
    const u32 x0 = bitOf(x, 0), x1 = bitOf(x, 1), x2 = bitOf(x, 2), y0 = bitOf(y, 0), y1 = bitOf(y, 1), y2 = bitOf(y, 2);
    const u32 z0 = bitOf(z, 0), z1 = bitOf(z, 1), z2 = bitOf(z, 2);
    if (d.micro == MDisplay) {
        switch (d.bpp) {
            case 8: p[0] = x0, p[1] = x1, p[2] = x2, p[3] = y1, p[4] = y0, p[5] = y2; break;
            case 16: p[0] = x0, p[1] = x1, p[2] = x2, p[3] = y0, p[4] = y1, p[5] = y2; break;
            case 32: p[0] = x0, p[1] = x1, p[2] = y0, p[3] = x2, p[4] = y1, p[5] = y2; break;
            case 64: p[0] = x0, p[1] = y0, p[2] = x1, p[3] = x2, p[4] = y1, p[5] = y2; break;
            default: p[0] = y0, p[1] = x0, p[2] = x1, p[3] = x2, p[4] = y1, p[5] = y2; break;
        }
    } else if (d.micro == MThin || d.micro == MDepth) {
        p[0] = x0, p[1] = y0, p[2] = x1, p[3] = y1, p[4] = x2, p[5] = y2;
    } else {
        if (d.bpp == 8 || d.bpp == 16) p[0] = x0, p[1] = y0, p[2] = x1, p[3] = y1, p[4] = z0, p[5] = z1;
        else if (d.bpp == 32) p[0] = x0, p[1] = y0, p[2] = x1, p[3] = z0, p[4] = y1, p[5] = z1;
        else p[0] = x0, p[1] = y0, p[2] = z0, p[3] = x1, p[4] = y1, p[5] = z1;
        p[6] = x2, p[7] = y2;
        if (d.thick == 8) p[8] = z2;
    }
    u32 r = 0;
    for (u32 i = 0; i < 9; ++i) r |= p[i] << i;
    return r;
}
inline u32 microTileBytes(const Defs& d) { return 8 * 8 * d.thick * d.bpp * d.samples / 8; }
inline u32 addrMicro(const Defs& d, u32 x, u32 y, u32 slice, u32 pitch, u32 height) {  // ComputeSurfaceAddrFromCoordMicroTiled
    const u32 sliceBytes = (pitch * height * d.thick * d.bpp * d.samples + 7) / 8;
    const u32 perRow = pitch / 8;
    const u32 sliceOffset = (slice / d.thick) * sliceBytes;
    const u32 tileOffset = ((y / 8) * perRow + (x / 8)) * microTileBytes(d);
    const u32 pixel = pixelIndex(d, x, y, slice);
    const u32 pixelOffset = d.micro == MDepth ? pixel * d.bpp * d.samples : pixel * d.bpp;
    return sliceOffset + tileOffset + pixelOffset / 8;  // (sample 0)
}
inline u32 addrMacro(const Defs& d, u32 x, u32 y, u32 slice, u32 pitch, u32 height) {  // ComputeSurfaceAddrFromCoordMacroTiled
    const u32 numPipes = d.pipeCfg == P2 ? 2 : 8, numPipeBits = d.pipeCfg == P2 ? 1 : 3;
    const u32 pixel = pixelIndex(d, x, y, slice);
    u32 element = (d.micro == MDepth ? pixel * d.bpp * d.samples : pixel * d.bpp) / 8;
    u32 slicesPerTile = 1, tileSplitSlice = 0, tileBytes = microTileBytes(d);
    if (tileBytes > d.tileSplit && d.thick == 1) {
        slicesPerTile = tileBytes / d.tileSplit;
        tileSplitSlice = element / d.tileSplit;
        element %= d.tileSplit;
        tileBytes = d.tileSplit;
    }
    const u32 macroPitch = (8 * 1 * numPipes) * d.aspect;
    const u32 macroHeight = (8 * d.bankH * d.banks) / d.aspect;
    const u32 macroBytes = tileBytes * (macroPitch / 8) * (macroHeight / 8) / (numPipes * d.banks);
    const u32 perRow = pitch / macroPitch;
    const u32 macroOffset = ((y / macroHeight) * perRow + x / macroPitch) * macroBytes;
    const u32 sliceBytes = perRow * (height / macroHeight) * macroBytes;
    const u32 sliceOffset = sliceBytes * (tileSplitSlice + slicesPerTile * (slice / d.thick));
    const u32 tileRow = (y / 8) % d.bankH, tileColumn = ((x / 8) / numPipes) % 1;
    const u32 tileOffset = (tileRow * 1 + tileColumn) * tileBytes;
    const u32 total = sliceOffset + macroOffset + element + tileOffset;
    const Array a = Array(d.array);
    if (isPrt(a)) x %= macroPitch, y %= macroHeight;

    // ComputePipeFromCoord
    const u32 tx = x / 8, ty = y / 8;
    const u32 x3 = bitOf(tx, 0), x4 = bitOf(tx, 1), x5 = bitOf(tx, 2), y3 = bitOf(ty, 0), y4 = bitOf(ty, 1), y5 = bitOf(ty, 2);
    u32 p0 = 0, p1 = 0, p2 = 0;
    if (d.pipeCfg == P2) p0 = x3 ^ y3;
    else if (d.pipeCfg == P8_8x16) p0 = x4 ^ y3 ^ x5, p1 = x3 ^ y4, p2 = x5 ^ y5;
    else p0 = x3 ^ y3 ^ x4, p1 = x4 ^ y4, p2 = x5 ^ y5;
    u32 pipe = p0 | (p1 << 1) | (p2 << 2);
    u32 pipeSwizzle = 0;
    if (a == A3DThin || a == A3DThick || a == A3DXThick) pipeSwizzle += std::max(1u, numPipes / 2 - 1) * (slice / d.thick);
    pipeSwizzle &= numPipes - 1;
    pipe ^= pipeSwizzle;

    // ComputeBankFromCoord
    const u32 btx = x / 8 / (1 * numPipes), bty = y / 8 / d.bankH;
    const u32 bx3 = bitOf(btx, 0), bx4 = bitOf(btx, 1), bx5 = bitOf(btx, 2), bx6 = bitOf(btx, 3);
    const u32 by3 = bitOf(bty, 0), by4 = bitOf(bty, 1), by5 = bitOf(bty, 2), by6 = bitOf(bty, 3);
    u32 b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    if (d.banks == 16) b0 = bx3 ^ by6, b1 = bx4 ^ by5 ^ by6, b2 = bx5 ^ by4, b3 = bx6 ^ by3;
    else if (d.banks == 8) b0 = bx3 ^ by5, b1 = bx4 ^ by4 ^ by5, b2 = bx5 ^ by3;
    else if (d.banks == 4) b0 = bx3 ^ by4, b1 = bx4 ^ by3;
    else if (d.banks == 2) b0 = bx3 ^ by3;
    u32 bank = b0 | (b1 << 1) | (b2 << 2) | (b3 << 3);
    u32 sliceRotation = 0, tileSplitRotation = 0;
    if (a == A2DThin || a == A2DThick || a == A2DXThick) sliceRotation = ((d.banks / 2) - 1) * (slice / d.thick);
    else if (a == A3DThin || a == A3DThick || a == A3DXThick) sliceRotation = std::max(1u, (numPipes / 2) - 1) * (slice / d.thick) / numPipes;
    if (a == A2DThin || a == A3DThin || a == Prt2DThin || a == Prt3DThin) tileSplitRotation = ((d.banks / 2) + 1) * tileSplitSlice;
    bank ^= d.bankSwizzle + sliceRotation;
    bank ^= tileSplitRotation;
    bank &= d.banks - 1;

    const u32 interleaveMask = (1u << 8) - 1;
    const u32 numBankBits = uint32_t(std::bit_width(d.banks)) - 1;
    u32 addr = total & interleaveMask;
    addr |= pipe << 8;
    addr |= bank << (8 + numPipeBits);
    addr |= (total >> 8) << (8 + numPipeBits + numBankBits);
    return addr;
}

// Shader defines of one image (TileManager::GetTilingPipeline), the offset of texel (x, y, slice) of
// `mip` in the guest span, as the shader main() computes it.
inline u32 shaderOffset(const Image& im, const Chain& ch, u32 mip, u32 x, u32 y, u32 slice) {
    const Array am = arrayMode(im.tile);
    const Mip& m = ch.mips[mip];
    // (UpdateSize multiplies a block image's pitch and rows by 4, at least 32, and DetileImage divides them
    // again: the identity for the 8-aligned pitch and rows of a tiled level.)
    const u32 pitch = m.pitch, height = m.height;
    Defs d{};
    d.bpp = im.bits, d.samples = im.samples, d.micro = microMode(im.tile), d.array = am, d.thick = thickness(am);
    d.bankSwizzle = im.bankSwizzle;
    if (isMacro(am) && !m.downgraded) {
        const u32 mtm = macroTileMode(im.tile, im.bits, im.samples);
        d.pipeCfg = pipeConfig(im.tile);
        d.bankH = bankHeight(mtm);
        d.banks = numBanks(mtm);
        d.tileSplit = calcTileSplit(im.tile, am, microMode(im.tile), im.bits);
        d.aspect = macroAspect(mtm);
        return m.offset + addrMacro(d, x, y, slice, pitch, height);
    }
    if (am == LinearAligned) return m.offset + (slice * pitch * height + y * pitch + x) * (im.bits / 8);
    if (m.downgraded) d.array = thickness(am) > 1 ? A1DThick : A1DThin, d.thick = thickness(am) > 1 ? 4 : 1;
    return m.offset + addrMicro(d, x, y, slice, pitch, height);
}

}  // namespace sref

// ---- the test -----------------------------------------------------------------------------------------

static int failures = 0, printed = 0;
static uint64_t surfacesChecked = 0, elementsChecked = 0, degradedLevels = 0, macroLevel0 = 0;
__attribute__((format(printf, 1, 2))) static void fail(const char* f, ...) {
    ++failures;
    if (printed++ >= 60) return;
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    std::printf("FAIL %s\n", buf);
}
#define CHECK(c)                                                    \
    do {                                                            \
        if (!(c)) {                                                 \
            std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                             \
        }                                                           \
    } while (0)

struct Shape {
    uint32_t w, h, pitch, layers, levels;
    bool pow2, volume;
    uint32_t bankSwizzle;
};

static std::string describe(const char* kind, uint32_t tile, uint32_t bpe, const Shape& sh, uint32_t block) {
    return std::string(kind) + " mode " + std::to_string(tile) + " bpe " + std::to_string(bpe) + " " + std::to_string(sh.w) + "x" +
           std::to_string(sh.h) + " pitch " + std::to_string(sh.pitch) + (sh.volume ? " depth " : " slices ") + std::to_string(sh.layers) +
           " levels " + std::to_string(sh.levels) + (sh.pow2 ? " pow2" : "") + (block > 1 ? " block4" : "");
}

// A pattern that differs from byte to byte inside any 256-byte block and from block to block.
static uint8_t pat(uint64_t i, uint32_t salt) { return uint8_t(i + (i >> 8) * 37 + (i >> 16) * 11 + salt * 101); }

static void checkOne(uint32_t tile, uint32_t bpe, uint32_t block, const Shape& sh) {
    const std::string what = describe("", tile, bpe, sh, block);
    sref::Image im;
    im.tile = tile, im.bits = bpe * 8, im.width = sh.w, im.height = sh.h, im.layers = sh.volume ? 1 : sh.layers;
    im.depth = sh.volume ? sh.layers : 1, im.levels = sh.levels, im.block = block > 1, im.pow2 = sh.pow2, im.volume = sh.volume;
    im.bankSwizzle = sh.bankSwizzle;
    im.pitch = sh.pitch ? sh.pitch : sh.w;
    // A pitch the descriptor of a real allocation would have: stable under the 1D / linear padding loop.
    const sref::Array am = sref::arrayMode(tile);
    if (sh.pitch && !sref::isMacro(am)) {
        for (int it = 0; it < 8; ++it) {
            const sref::Chain c = sref::updateSize(im);
            const uint32_t next = c.mips[0].pitch * block;
            if (next == im.pitch) break;
            im.pitch = next;
        }
    }
    const sref::Chain ch = sref::updateSize(im);

    TiledSurface s;
    s.tileIndex = uint8_t(tile), s.bytesPerElement = bpe, s.width = sh.w, s.height = sh.h, s.pitch = sh.pitch ? im.pitch : 0;
    s.slices = sh.layers, s.levels = sh.levels, s.samples = 1, s.blockSize = block, s.volume = sh.volume, s.pow2Pad = sh.pow2;
    s.bankSwizzle = sh.bankSwizzle;
    const SurfaceLayout lay = computeLayout(s);
    if (!lay.ok) return fail("%s: layout not ok", what.c_str());
    if (lay.levels.size() != ch.mips.size()) return fail("%s: level count", what.c_str());

    const int before = failures;
    for (size_t i = 0; i < ch.mips.size(); ++i) {
        const sref::Mip& m = ch.mips[i];
        const LevelLayout& l = lay.levels[i];
        if (l.pitch != m.pitch || l.paddedHeight != m.height || l.bytes != m.size || l.offset != m.offset ||
            l.degraded != m.downgraded) {
            fail("%s: level %zu pitch %u/%u rows %u/%u bytes %llu/%u offset %llu/%u degraded %d/%d (VPKEngine/shadPS4)", what.c_str(), i,
                 l.pitch, m.pitch, l.paddedHeight, m.height, (unsigned long long)l.bytes, m.size, (unsigned long long)l.offset, m.offset,
                 int(l.degraded), int(m.downgraded));
        }
    }
    if (lay.bytes != ch.guestSize) fail("%s: total %llu, shadPS4 %u", what.c_str(), (unsigned long long)lay.bytes, ch.guestSize);
    if (failures != before) return;  // the element comparison below needs the same geometry

    ++surfacesChecked;
    for (const LevelLayout& l : lay.levels) degradedLevels += l.degraded;
    macroLevel0 += sref::isMacro(am) && !lay.levels[0].degraded && (lay.levels[0].width < lay.levels[0].pitch || lay.levels[0].height < lay.levels[0].paddedHeight);
    // Element by element: the address of VPKEngine against the shader's.
    std::vector<uint32_t> want;
    for (uint32_t li = 0; li < lay.levels.size(); ++li) {
        const LevelLayout& l = lay.levels[li];
        for (uint32_t z = 0; z < l.slices; ++z)
            for (uint32_t y = 0; y < l.height; ++y)
                for (uint32_t x = 0; x < l.width; ++x) {
                    const uint32_t r = sref::shaderOffset(im, ch, li, x, y, z);
                    const uint64_t o = lay.offsetOf(li, x, y, z, 0);
                    if (o != r || r + bpe > ch.guestSize) {
                        return fail("%s: level %u (%u,%u,%u) at %llu, shader %u", what.c_str(), li, x, y, z, (unsigned long long)o, r);
                    }
                    want.push_back(r);
                    ++elementsChecked;
                }
    }

    // Byte for byte: detile a unique-pattern surface, tile a unique-pattern linear image.
    std::vector<uint8_t> tiled(lay.bytes), linear(lay.linearBytes), expLinear(lay.linearBytes, 0), expTiled(lay.bytes, 0);
    for (uint64_t i = 0; i < tiled.size(); ++i) tiled[i] = pat(i, 1);
    for (uint64_t i = 0; i < linear.size(); ++i) linear[i] = pat(i, 2);
    size_t k = 0;
    for (uint32_t li = 0; li < lay.levels.size(); ++li) {
        const LevelLayout& l = lay.levels[li];
        for (uint32_t z = 0; z < l.slices; ++z)
            for (uint32_t y = 0; y < l.height; ++y)
                for (uint32_t x = 0; x < l.width; ++x) {
                    const uint64_t lo = l.linearOffset + ((uint64_t(z) * l.height + y) * l.width + x) * bpe;
                    std::memcpy(&expLinear[lo], &tiled[want[k]], bpe);
                    std::memcpy(&expTiled[want[k]], &linear[lo], bpe);
                    ++k;
                }
    }
    std::vector<uint8_t> out(lay.linearBytes, 0);
    CHECK(detileLevels(lay, 0, uint32_t(lay.levels.size()), 0, tiled.data(), out.data()));
    if (out != expLinear) return fail("%s: detile differs from the shader's copy", what.c_str());
    std::vector<uint8_t> back(lay.bytes, 0);
    CHECK(tileLevels(lay, 0, uint32_t(lay.levels.size()), 0, linear.data(), back.data()));
    if (back != expTiled) return fail("%s: tile differs from the shader's copy", what.c_str());
    std::vector<uint8_t> round(lay.linearBytes, 0);
    CHECK(detileLevels(lay, 0, uint32_t(lay.levels.size()), 0, back.data(), round.data()));
    if (round != linear) fail("%s: round trip (tile then detile) differs", what.c_str());
}

int main() {
    // thin / display / depth: linear aligned (8), 1D (5, 9, 13), macro tiled (the rest). 18 and the thick modes: see above.
    const uint32_t modes[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    const Shape shapes[] = {
        {4, 4, 0, 1, 1, false, false, 0},        // far below one macro tile, level 0
        {17, 9, 0, 1, 5, false, false, 0},       // not a power of two, down to 1x1
        {33, 65, 40, 6, 7, false, false, 5},     // pitch beyond the width, 6 slices, bank swizzle
        {64, 64, 0, 1, 7, true, false, 0},       // POW2_PAD
        {100, 37, 128, 1, 8, true, false, 0},    // POW2_PAD with a pitch
        {130, 70, 0, 6, 8, false, false, 0},     // just above one macro tile
        {256, 256, 256, 1, 9, false, false, 0},  // 9 levels
        {200, 120, 256, 6, 9, true, false, 0},   // everything at once
        {40, 24, 0, 5, 6, false, true, 0},       // volume
        {37, 19, 48, 3, 6, true, true, 0},       // volume with POW2_PAD and a pitch
    };
    for (uint32_t m : modes)
        for (uint32_t bpe : {1u, 2u, 4u, 8u, 16u})
            for (uint32_t block : {1u, 4u})
                for (const Shape& sh : shapes) {
                    if (sh.volume && m != 8 && m != 9 && m != 10 && m != 13 && m != 14 && m != 15) continue;  // 3D in the 3D modes and a few others
                    checkOne(m, bpe, block, sh);
                }

    // The macro tile extents shadPS4 tabulates against the geometry VPKEngine derives, for every sample count.
    for (uint32_t m : modes) {
        if (!sref::isMacro(sref::arrayMode(m))) continue;
        for (uint32_t bpe : {1u, 2u, 4u, 8u, 16u})
            for (uint32_t samples : {1u, 2u, 4u, 8u}) {
                TiledSurface s;
                s.tileIndex = uint8_t(m), s.bytesPerElement = bpe, s.width = 1024, s.height = 1024, s.samples = samples;
                const SurfaceLayout l = computeLayout(s);
                uint32_t w = 0, h = 0;
                sref::macroExtents(m, bpe * 8, samples, w, h);
                if (!l.ok || l.levels[0].hw.macroPitch != w || l.levels[0].hw.macroHeight != h || l.levels[0].degraded ||
                    l.macroTileMode != sref::macroTileMode(m, bpe * 8, samples)) {
                    fail("extents mode %u bpe %u samples %u: %u x %u (macro mode %u), shadPS4 %u x %u (macro mode %u)", m, bpe, samples,
                         l.ok ? l.levels[0].hw.macroPitch : 0, l.ok ? l.levels[0].hw.macroHeight : 0, l.macroTileMode, w, h,
                         sref::macroTileMode(m, bpe * 8, samples));
                }
            }
    }
    // Mode 18 (PRT 3D thin): the extent table row of shadPS4 against the PRT macro tile VPKEngine derives.
    for (uint32_t bpe : {1u, 2u, 4u, 8u, 16u}) {
        TiledSurface s;
        s.tileIndex = 18, s.bytesPerElement = bpe, s.width = 1024, s.height = 1024;
        const SurfaceLayout l = computeLayout(s);
        uint32_t w = 0, h = 0;
        sref::macroExtents(18, bpe * 8, 1, w, h);
        CHECK(l.ok && l.levels[0].hw.macroPitch == w && l.levels[0].hw.macroHeight == h);
    }

    // The case from the captures: a 32-bit mode 14 texture smaller than its 128x64 macro tile is padded to a whole macro
    // tile on level 0 (not stored 1D); the smaller levels below it are.
    {
        TiledSurface s;
        s.tileIndex = 14, s.bytesPerElement = 4, s.width = 64, s.height = 32, s.levels = 3;
        const SurfaceLayout l = computeLayout(s);
        CHECK(l.ok && !l.levels[0].degraded && l.levels[0].pitch == 128 && l.levels[0].paddedHeight == 64);
        CHECK(l.ok && l.levels[1].degraded);
    }

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("tiling_shadps4: ok (%llu surfaces, %llu elements, %llu degraded mip levels, %llu padded macro level 0)\n", (unsigned long long)surfacesChecked,
                (unsigned long long)elementsChecked, (unsigned long long)degradedLevels, (unsigned long long)macroLevel0);
    return 0;
}
