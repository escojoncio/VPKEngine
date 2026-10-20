// PS4 surface tiling (see tiling.h): the tile mode tables, the mip chain layout and the CPU copies.
// The address of one element is vpt_offset() (tiling_addr.h), shared with the Metal kernels.
#include "vpps4/tiling.h"

#include <algorithm>
#include <bit>
#include <cstring>

namespace vpps4 {
namespace {

// GB_TILE_MODE0..31 as the PS4 programs them (base mode).
constexpr TileModeInfo kModes[32] = {
    {true, ArrayMode::Tiled2DThin, MicroMode::Depth, 1, 1, 64, "depth_2d_thin_64"},
    {true, ArrayMode::Tiled2DThin, MicroMode::Depth, 1, 1, 128, "depth_2d_thin_128"},
    {true, ArrayMode::Tiled2DThin, MicroMode::Depth, 1, 1, 256, "depth_2d_thin_256"},
    {true, ArrayMode::Tiled2DThin, MicroMode::Depth, 1, 1, 512, "depth_2d_thin_512"},
    {true, ArrayMode::Tiled2DThin, MicroMode::Depth, 1, 1, 1024, "depth_2d_thin_1k"},
    {true, ArrayMode::Tiled1DThin, MicroMode::Depth, 1, 1, 64, "depth_1d_thin"},
    {true, ArrayMode::Prt2DThin, MicroMode::Depth, 1, 1, 256, "depth_2d_thin_prt_256"},
    {true, ArrayMode::Prt2DThin, MicroMode::Depth, 1, 1, 1024, "depth_2d_thin_prt_1k"},
    {true, ArrayMode::LinearAligned, MicroMode::Display, 1, 1, 64, "display_linear_aligned"},
    {true, ArrayMode::Tiled1DThin, MicroMode::Display, 1, 1, 64, "display_1d_thin"},
    {true, ArrayMode::Tiled2DThin, MicroMode::Display, 1, 2, 64, "display_2d_thin"},
    {true, ArrayMode::PrtThin, MicroMode::Display, 2, 2, 64, "display_thin_prt"},
    {true, ArrayMode::Prt2DThin, MicroMode::Display, 1, 2, 64, "display_2d_thin_prt"},
    {true, ArrayMode::Tiled1DThin, MicroMode::Thin, 1, 1, 64, "thin_1d_thin"},
    {true, ArrayMode::Tiled2DThin, MicroMode::Thin, 1, 2, 64, "thin_2d_thin"},
    {true, ArrayMode::Tiled3DThin, MicroMode::Thin, 2, 2, 64, "thin_3d_thin"},
    {true, ArrayMode::PrtThin, MicroMode::Thin, 2, 2, 64, "thin_thin_prt"},
    {true, ArrayMode::Prt2DThin, MicroMode::Thin, 1, 2, 64, "thin_2d_thin_prt"},
    {true, ArrayMode::Prt3DThin, MicroMode::Thin, 1, 2, 64, "thin_3d_thin_prt"},
    {true, ArrayMode::Tiled1DThick, MicroMode::Thick, 1, 1, 64, "thick_1d_thick"},
    {true, ArrayMode::Tiled2DThick, MicroMode::Thick, 1, 1, 64, "thick_2d_thick"},
    {true, ArrayMode::Tiled3DThick, MicroMode::Thick, 2, 1, 64, "thick_3d_thick"},
    {true, ArrayMode::PrtThick, MicroMode::Thick, 2, 1, 64, "thick_thick_prt"},
    {true, ArrayMode::Prt2DThick, MicroMode::Thick, 1, 1, 64, "thick_2d_thick_prt"},
    {true, ArrayMode::Prt3DThick, MicroMode::Thick, 2, 1, 64, "thick_3d_thick_prt"},
    {true, ArrayMode::Tiled2DXThick, MicroMode::Thick, 1, 1, 64, "thick_2d_xthick"},
    {true, ArrayMode::Tiled3DXThick, MicroMode::Thick, 2, 1, 64, "thick_3d_xthick"},
    {false, ArrayMode::LinearAligned, MicroMode::Display, 1, 1, 64, "unused_27"},
    {false, ArrayMode::LinearAligned, MicroMode::Display, 1, 1, 64, "unused_28"},
    {false, ArrayMode::LinearAligned, MicroMode::Display, 1, 1, 64, "unused_29"},
    {false, ArrayMode::LinearAligned, MicroMode::Display, 1, 1, 64, "unused_30"},
    {true, ArrayMode::LinearGeneral, MicroMode::Display, 0, 1, 64, "display_linear_general"},
};
constexpr TileModeInfo kInvalid = {false, ArrayMode::LinearAligned, MicroMode::Display, 1, 1, 64, "out_of_range"};

// GB_MACROTILE_MODE0..15: bank width, bank height, banks, macro tile aspect.
constexpr MacroTileInfo kMacro[16] = {
    {1, 4, 16, 4}, {1, 2, 16, 2}, {1, 1, 16, 2}, {1, 1, 16, 2}, {1, 1, 8, 1}, {1, 1, 4, 1}, {1, 1, 2, 1}, {1, 1, 2, 1},
    {1, 8, 16, 4}, {1, 4, 16, 4}, {1, 2, 16, 2}, {1, 1, 16, 2}, {1, 1, 8, 1}, {1, 1, 4, 1}, {1, 1, 2, 1}, {1, 1, 2, 1},
};

uint32_t alignUp(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

bool isPrt(ArrayMode a) {
    return a == ArrayMode::PrtThin || a == ArrayMode::Prt2DThin || a == ArrayMode::PrtThick ||
           a == ArrayMode::Prt2DThick || a == ArrayMode::Prt3DThin || a == ArrayMode::Prt3DThick;
}

uint32_t flagsOf(ArrayMode a) {
    uint32_t f = 0;
    if (isPrt(a)) f |= VPT_F_PRT;
    if (a == ArrayMode::Tiled2DThin || a == ArrayMode::Tiled2DThick || a == ArrayMode::Tiled2DXThick) f |= VPT_F_ROT2D;
    if (a == ArrayMode::Tiled3DThin || a == ArrayMode::Tiled3DThick || a == ArrayMode::Tiled3DXThick) f |= VPT_F_ROT3D;
    if (a == ArrayMode::Tiled2DThin || a == ArrayMode::Tiled3DThin || a == ArrayMode::Prt2DThin || a == ArrayMode::Prt3DThin)
        f |= VPT_F_SPLITROT;
    return f;
}

}  // namespace

const TileModeInfo& tileModeInfo(uint32_t index) { return index < 32 ? kModes[index] : kInvalid; }

MacroTileInfo macroTileInfo(uint32_t mode) { return kMacro[mode & 15]; }

uint32_t microTileThickness(ArrayMode m) {
    switch (m) {
        case ArrayMode::Tiled1DThick:
        case ArrayMode::Tiled2DThick:
        case ArrayMode::Tiled3DThick:
        case ArrayMode::PrtThick:
        case ArrayMode::Prt2DThick:
        case ArrayMode::Prt3DThick: return 4;
        case ArrayMode::Tiled2DXThick:
        case ArrayMode::Tiled3DXThick: return 8;
        default: return 1;
    }
}

SurfaceLayout computeLayout(const TiledSurface& s) {
    SurfaceLayout out;
    out.surface = s;
    auto fail = [&](std::string why) {
        out.why = std::move(why);
        return out;
    };
    const TileModeInfo& m = tileModeInfo(s.tileIndex);
    if (!m.valid) return fail("tile mode index " + std::to_string(s.tileIndex) + " (" + m.name + ") has no layout");
    if (m.micro == MicroMode::Rotated) return fail(std::string("tile mode ") + m.name + ": rotated micro tiles not covered");
    const uint32_t bpe = s.bytesPerElement;
    if (bpe != 1 && bpe != 2 && bpe != 4 && bpe != 8 && bpe != 16) return fail(std::to_string(bpe) + " bytes per element");
    if (s.samples != 1 && s.samples != 2 && s.samples != 4 && s.samples != 8) return fail(std::to_string(s.samples) + " samples");
    if (!s.width || !s.height || !s.slices || !s.blockSize) return fail("empty surface");
    if (s.levels < 1 || s.levels > 16) return fail(std::to_string(s.levels) + " mip levels");
    if (isLinear(m.array) && s.samples > 1) return fail("multisampled linear surface");

    const uint32_t thickness = microTileThickness(m.array);
    const bool macro = isMacroTiled(m.array);
    out.thickness = thickness;
    MacroTileInfo mt{};
    uint32_t splitBytes = 0, splits = 1;
    const uint32_t microBytes1x = 64 * bpe * thickness;
    if (macro) {
        // The macro tile mode follows from the bytes of a micro tile that stay in one bank row.
        uint32_t tileSplit = m.micro == MicroMode::Depth ? m.tileSplit : std::max(256u, m.sampleSplit * microBytes1x);
        tileSplit = std::min(1024u, tileSplit);
        const uint32_t tileBytes = std::min(tileSplit, s.samples * microBytes1x);
        out.macroTileMode = uint32_t(std::bit_width(tileBytes / 64u)) - 1u + (isPrt(m.array) ? 8u : 0u);
        mt = macroTileInfo(out.macroTileMode);
        if (thickness == 1) {
            const uint32_t microBytes = microBytes1x * s.samples;
            if (microBytes > tileSplit) splitBytes = tileSplit, splits = microBytes / tileSplit;
        }
    }

    const uint32_t pitch0 = std::max(s.pitch ? s.pitch : s.width, s.width);
    const uint32_t pipeBits = m.pipeConfig == 0 ? 1 : 3;
    uint64_t offset = 0, linear = 0;
    for (uint32_t level = 0; level < s.levels; ++level) {
        LevelLayout l;
        uint32_t lw = std::max(pitch0 >> level, 1u), lh = std::max(s.height >> level, 1u);
        uint32_t vw = std::max(s.width >> level, 1u), vh = std::max(s.height >> level, 1u);
        const uint32_t vd = s.volume ? std::max(s.slices >> level, 1u) : s.slices;
        if (s.blockSize > 1) {
            const uint32_t b = s.blockSize;
            lw = std::max((lw + b - 1) / b, 1u), lh = std::max((lh + b - 1) / b, 1u);
            vw = std::max((vw + b - 1) / b, 1u), vh = std::max((vh + b - 1) / b, 1u);
        }
        if (s.pow2Pad) lw = std::bit_ceil(lw), lh = std::bit_ceil(lh);
        l.width = vw, l.height = vh, l.slices = vd;
        const uint32_t allocSlices = s.pow2Pad && s.volume ? std::bit_ceil(vd) : vd;
        // A level of a macro tiled mode smaller than one macro tile is stored 1D tiled.
        const uint32_t macroPitch = macro ? 8 * (1u << pipeBits) * mt.bankWidth * mt.aspect : 0;
        const uint32_t macroHeight = macro ? 8 * mt.bankHeight * mt.banks / mt.aspect : 0;
        ArrayMode array = m.array;
        uint32_t eThick = thickness;
        l.degraded = macro && (lw < macroPitch || lh < macroHeight);
        if (l.degraded) {
            eThick = thickness > 1 ? 4 : 1;
            array = thickness > 1 ? ArrayMode::Tiled1DThick : ArrayMode::Tiled1DThin;
        }
        l.array = array;
        // Level 0 keeps the descriptor's pitch; the others (and a surface without one) get the
        // library's padding, which grows the pitch until a slice is a whole number of 256-byte groups.
        const bool derivePitch = level > 0 || s.pitch == 0;
        VptLevel& hw = l.hw;
        hw = VptLevel{};
        hw.bpe = bpe;
        hw.thickness = eThick;
        hw.samples = s.samples;
        hw.micro = m.micro == MicroMode::Display ? VPT_DISPLAY : m.micro == MicroMode::Depth ? VPT_DEPTH : m.micro == MicroMode::Thick ? VPT_THICK : VPT_THIN;
        hw.bankSwizzle = s.bankSwizzle;
        hw.pipeSwizzle = s.pipeSwizzle;
        uint64_t sliceBytes;  // one slice, every sample
        if (isLinear(array)) {
            hw.kind = VPT_LINEAR;
            // Aligned rows are a multiple of 64 bytes (at least 8 elements); general linear has no padding.
            const uint32_t pitchAlign = array == ArrayMode::LinearGeneral ? 1u : std::max(8u, 64u / bpe);
            uint32_t p = alignUp(lw, pitchAlign);
            if (derivePitch && array != ArrayMode::LinearGeneral)
                while ((uint64_t(p) * lh) % std::max(64u, 256u / bpe)) p += pitchAlign;
            hw.pitch = p, hw.height = lh;
            sliceBytes = uint64_t(p) * lh * bpe;
        } else if (!macro || l.degraded) {
            hw.kind = VPT_MICRO;
            uint32_t p = alignUp(lw, 8);
            const uint32_t h = alignUp(lh, 8);
            if (derivePitch)
                while ((uint64_t(p) * h * bpe * s.samples * eThick) % 256) p += 8;
            hw.pitch = p, hw.height = h;
            sliceBytes = uint64_t(p) * h * bpe * s.samples;
        } else {
            hw.kind = VPT_MACRO;
            hw.pitch = alignUp(lw, macroPitch);
            hw.height = alignUp(lh, macroHeight);
            hw.pipeKind = m.pipeConfig;
            hw.pipeBits = pipeBits;
            hw.bankBits = uint32_t(std::countr_zero(uint32_t(mt.banks)));
            hw.bankHeight = mt.bankHeight;
            hw.macroPitch = macroPitch;
            hw.macroHeight = macroHeight;
            hw.macroPerRow = hw.pitch / macroPitch;
            hw.macroRows = hw.height / macroHeight;
            hw.splits = splits;
            hw.splitBytes = splits > 1 ? splitBytes : microBytes1x * s.samples;
            hw.flags = flagsOf(array);
            sliceBytes = uint64_t(hw.pitch) * hw.height * bpe * s.samples;
        }
        l.pitch = hw.pitch, l.paddedHeight = hw.height;
        l.bytes = sliceBytes * alignUp(allocSlices, eThick);
        if (l.bytes >= (1ull << 32)) return fail("a mip level of 4 GiB or more");
        l.offset = offset;
        offset += l.bytes;
        l.linearOffset = linear;
        l.linearBytes = uint64_t(vw) * vh * l.slices * bpe;
        linear += l.linearBytes;
        out.levels.push_back(l);
    }
    out.bytes = offset;
    out.linearBytes = linear;
    out.ok = true;
    return out;
}

uint64_t SurfaceLayout::offsetOf(uint32_t level, uint32_t x, uint32_t y, uint32_t slice, uint32_t sample) const {
    if (!ok || level >= levels.size() || sample >= surface.samples) return UINT64_MAX;
    const LevelLayout& l = levels[level];
    if (x >= l.pitch || y >= l.paddedHeight || slice >= l.slices) return UINT64_MAX;
    return l.offset + vpt_offset(l.hw, x, y, slice, sample);
}

TiledSurface fmaskSurface(uint8_t fmaskTileIndex, uint32_t width, uint32_t height, uint32_t pitch, uint32_t slices,
                          uint32_t samples, uint32_t fragments) {
    // Each of the pixel's samples names one of its fragments: ceil(log2(fragments)) bits (at least 1).
    const uint32_t perSample = std::max(1u, uint32_t(std::bit_width(std::max(fragments, 1u) - 1)));
    uint32_t bytes = std::bit_ceil(std::max(1u, (samples * perSample + 7) / 8));
    TiledSurface s;
    s.tileIndex = fmaskTileIndex;
    s.bytesPerElement = std::min(bytes, 16u);
    s.width = width, s.height = height, s.pitch = pitch, s.slices = slices;
    return s;
}

namespace {

template <bool kToLinear>
bool copyLevels(const SurfaceLayout& layout, uint32_t firstLevel, uint32_t levelCount, uint32_t sample, const uint8_t* src,
                uint8_t* dst) {
    if (!layout.ok || sample >= layout.surface.samples || firstLevel + uint64_t(levelCount) > layout.levels.size())
        return false;
    const uint32_t bpe = layout.surface.bytesPerElement;
    for (uint32_t i = firstLevel; i < firstLevel + levelCount; ++i) {
        const LevelLayout& l = layout.levels[i];
        uint8_t* linearBase = (kToLinear ? dst : const_cast<uint8_t*>(src)) + l.linearOffset;
        for (uint32_t z = 0; z < l.slices; ++z)
            for (uint32_t y = 0; y < l.height; ++y) {
                uint8_t* row = linearBase + (uint64_t(z) * l.height + y) * l.width * bpe;
                if (l.hw.kind == VPT_LINEAR && l.hw.samples == 1) {
                    const uint64_t o = l.offset + vpt_offset(l.hw, 0, y, z, 0);
                    if constexpr (kToLinear) std::memcpy(row, src + o, uint64_t(l.width) * bpe);
                    else std::memcpy(dst + o, row, uint64_t(l.width) * bpe);
                    continue;
                }
                for (uint32_t x = 0; x < l.width; ++x) {
                    const uint64_t o = l.offset + vpt_offset(l.hw, x, y, z, sample);
                    if constexpr (kToLinear) std::memcpy(row + uint64_t(x) * bpe, src + o, bpe);
                    else std::memcpy(dst + o, row + uint64_t(x) * bpe, bpe);
                }
            }
    }
    return true;
}

}  // namespace

bool detileLevels(const SurfaceLayout& layout, uint32_t firstLevel, uint32_t levelCount, uint32_t sample, const uint8_t* tiled,
                  uint8_t* linear) {
    return copyLevels<true>(layout, firstLevel, levelCount, sample, tiled, linear);
}

bool tileLevels(const SurfaceLayout& layout, uint32_t firstLevel, uint32_t levelCount, uint32_t sample, const uint8_t* linear,
                uint8_t* tiled) {
    return copyLevels<false>(layout, firstLevel, levelCount, sample, linear, tiled);
}

namespace {
SurfaceLayout mip0(const TiledSurface& s) {
    TiledSurface t = s;
    t.levels = 1;
    return computeLayout(t);
}
}  // namespace

uint64_t tiledBytes(const TiledSurface& s, std::string& why) {
    const SurfaceLayout l = mip0(s);
    if (!l.ok) {
        why = l.why;
        return 0;
    }
    return l.bytes;
}

uint64_t tiledOffset(const TiledSurface& s, uint32_t x, uint32_t y, uint32_t slice) {
    return mip0(s).offsetOf(0, x, y, slice, 0);
}

bool detile(const TiledSurface& s, const uint8_t* src, uint8_t* dst, std::string& why) {
    const SurfaceLayout l = mip0(s);
    if (!l.ok) {
        why = l.why;
        return false;
    }
    return detileLevels(l, 0, 1, 0, src, dst);
}

bool tile(const TiledSurface& s, const uint8_t* src, uint8_t* dst, std::string& why) {
    const SurfaceLayout l = mip0(s);
    if (!l.ok) {
        why = l.why;
        return false;
    }
    return tileLevels(l, 0, 1, 0, src, dst);
}

}  // namespace vpps4
