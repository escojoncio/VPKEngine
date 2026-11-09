// PS4 (Liverpool, gfx7 "Sea Islands") surface tiling: where an element of a tiled surface lives in
// memory, for every tile mode, mip level, array slice, volume slice and sample.
//
// A T# or CB names its layout by TILE_MODE_INDEX, an index into the GB_TILE_MODE table the PS4
// system software programs (array mode, micro tile mode, pipe config, tile split, sample split);
// macro tiled modes take their bank geometry from GB_MACROTILE_MODE, picked by the bytes of one
// micro tile. Both tables are fixed on the console and are kept here as constants.
//
// Layout model (the address library's, written from AMD's public register documentation):
//   * A surface is a chain of mip levels stored one after the other, each level holding all its
//     slices (array layers, cube faces, volume depth) back to back. Level 0 uses the pitch the
//     descriptor says; deeper levels are derived (pitch >> level, height >> level, pow2 padded when
//     the descriptor says so, in blocks for block-compressed formats).
//   * A mip level (level 0 never) whose padded size is below one macro tile is stored 1D tiled (the
//     hardware degrades it); level 0 of a surface smaller than a macro tile stays macro tiled, padded to
//     a whole macro tile. There are no packed mip tails on gfx7 outside PRT, which only changes the
//     addressing of bank and pipe bits.
//   * Within a level: element -> micro tile (8x8, thickness 1, 4 or 8) -> macro tile (pipes x
//     banks of micro tiles) -> pipe/bank bits above the 256-byte pipe interleave.
//   * Tile split: a micro tile bigger than the mode's tile split (depth surfaces, MSAA) is cut in
//     planes, stored apart like extra slices with a bank rotation of their own.
//   * Samples: depth micro tiles interleave samples per pixel; the others store one sample plane
//     after the other inside the micro tile.
//
// Elements are texels, or blocks for block-compressed formats (TiledSurface::blockSize).
//
// Not covered (reported through `why`, nothing written): rotated micro tiles (no entry of the PS4
// table uses them), the unused tile mode slots 27-30, 12-byte elements, the Neo (PS4 Pro) alternate
// pipe configuration, DCC/CMASK/HTILE metadata (FMASK is a plain surface: fmaskSurface()).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vpps4/tiling_addr.h"

namespace vpps4 {

// ---- Tile mode tables -----------------------------------------------------------------------

enum class ArrayMode : uint8_t {
    LinearGeneral, LinearAligned, Tiled1DThin, Tiled1DThick, Tiled2DThin, PrtThin, Prt2DThin, Tiled2DThick,
    Tiled2DXThick, PrtThick, Prt2DThick, Prt3DThin, Tiled3DThin, Tiled3DThick, Tiled3DXThick, Prt3DThick
};
enum class MicroMode : uint8_t { Display, Thin, Depth, Rotated, Thick };

struct TileModeInfo {
    bool valid;
    ArrayMode array;
    MicroMode micro;
    uint8_t pipeConfig;   // 0 P2, 1 P8_32x32_16x16, 2 P8_32x32_8x16
    uint8_t sampleSplit;  // colour tile split in micro tiles' worth of samples
    uint16_t tileSplit;   // depth tile split, bytes
    const char* name;
};
// GB_TILE_MODE[index]; index >= 32 gives an invalid entry.
const TileModeInfo& tileModeInfo(uint32_t index);

struct MacroTileInfo {
    uint8_t bankWidth, bankHeight, banks, aspect;
};
// GB_MACROTILE_MODE[0..15] (8..15: the PRT variants).
MacroTileInfo macroTileInfo(uint32_t mode);

inline bool isLinear(ArrayMode m) { return m == ArrayMode::LinearGeneral || m == ArrayMode::LinearAligned; }
inline bool isMacroTiled(ArrayMode m) { return m >= ArrayMode::Tiled2DThin; }
uint32_t microTileThickness(ArrayMode);

// ---- Surfaces -------------------------------------------------------------------------------

struct TiledSurface {
    uint8_t tileIndex = 8;           // TILE_MODE_INDEX
    uint32_t bytesPerElement = 4;    // 1, 2, 4, 8 or 16 (a block for block-compressed formats)
    uint32_t width = 1, height = 1;  // texels of level 0
    uint32_t pitch = 0;              // texels per row of level 0 as allocated (0: width)
    uint32_t rows = 0;               // rows of level 0 as allocated (0: height; a depth surface drawn smaller than DB_DEPTH_SIZE)
    uint32_t slices = 1;             // array layers (cube: 6 per cube), or the depth of a volume
    // Beyond mip 0 of a plain surface:
    uint32_t levels = 1;             // mip levels stored (T# last level - base level + 1)
    uint32_t samples = 1;            // 1, 2, 4 or 8
    uint32_t blockSize = 1;          // texels per element edge (4 for BCn)
    bool volume = false;             // `slices` is a depth that halves per level
    bool pow2Pad = false;            // T# POW2_PAD: level sizes rounded up to powers of two
    uint32_t bankSwizzle = 0;        // bank / pipe XOR from the T# (0 for ordinary surfaces)
    uint32_t pipeSwizzle = 0;
};

struct LevelLayout {
    uint32_t width = 0, height = 0, slices = 0;  // valid extent in elements (slices: layers or depth of this level)
    uint32_t pitch = 0, paddedHeight = 0;         // allocated, in elements / rows
    ArrayMode array = ArrayMode::LinearAligned;   // as stored (a degraded level reports 1D)
    bool degraded = false;                        // stored 1D tiled though the mode is macro tiled (levels > 0 only)
    uint64_t offset = 0;                          // bytes from the surface base
    uint64_t bytes = 0;                           // all its slices
    uint64_t linearOffset = 0;                    // in the concatenation of all levels, tightly packed
    uint64_t linearBytes = 0;                     // one sample plane, all slices
    VptLevel hw{};                                // what the addressing function takes (tiling_addr.h)
};

struct SurfaceLayout {
    bool ok = false;
    std::string why;                 // when !ok: what is missing
    TiledSurface surface;
    std::vector<LevelLayout> levels;
    uint64_t bytes = 0;              // the whole chain
    uint64_t linearBytes = 0;        // all levels, one sample plane
    uint32_t thickness = 1;
    uint32_t macroTileMode = 0;      // GB_MACROTILE_MODE used (macro tiled modes)

    // Byte offset of element (x, y, slice, sample) of `level` from the surface base. UINT64_MAX if
    // the layout is not ok or the coordinates are outside the level.
    uint64_t offsetOf(uint32_t level, uint32_t x, uint32_t y, uint32_t slice, uint32_t sample = 0) const;
};

SurfaceLayout computeLayout(const TiledSurface&);

// FMASK of a multisampled colour surface as a surface of its own: `fmaskTileIndex` from the CB,
// bits per pixel from samples and fragments (8 / 16 / 32). Elements are the fragment index tables.
TiledSurface fmaskSurface(uint8_t fmaskTileIndex, uint32_t width, uint32_t height, uint32_t pitch, uint32_t slices,
                          uint32_t samples, uint32_t fragments);

// ---- Copies ---------------------------------------------------------------------------------
// Linear side: levels concatenated at LevelLayout::linearOffset, inside a level slice after slice,
// rows of `width` elements, tightly packed. Tiled side: `tiled` is the surface base (layout.bytes
// readable). Each returns false (nothing touched) if the layout is not ok or a range is out of it.

bool detileLevels(const SurfaceLayout&, uint32_t firstLevel, uint32_t levelCount, uint32_t sample, const uint8_t* tiled,
                  uint8_t* linear);
bool tileLevels(const SurfaceLayout&, uint32_t firstLevel, uint32_t levelCount, uint32_t sample, const uint8_t* linear,
                uint8_t* tiled);

// One subresource (level, slice, sample 0) of the tiled surface against a host read of it (`packed`: rows of the level's
// `width` elements, tightly packed, as Backend::readTexture gives them). ok false: the layout is not ok, the coordinates are
// out of it, `tiled` (tiledBytes from the surface base) does not hold it or `packed` is short. Diagnostics (TEXVERIFY).
struct LevelDiff {
    bool ok = false;
    uint64_t elements = 0, differing = 0;
    uint32_t firstX = 0, firstY = 0;  // the first differing element (row order)
};
LevelDiff compareLevel(const SurfaceLayout&, uint32_t level, uint32_t slice, const uint8_t* tiled, uint64_t tiledBytes,
                       const uint8_t* packed, uint64_t packedBytes);

// ---- Mip 0 conveniences (the original API) ------------------------------------------------------

// Bytes of guest memory mip 0 spans (all slices), or 0 if the layout is not covered (`why` says).
uint64_t tiledBytes(const TiledSurface&, std::string& why);
// Element (x, y, slice)'s byte offset from the surface's base (sample 0, mip 0); UINT64_MAX if not covered.
uint64_t tiledOffset(const TiledSurface&, uint32_t x, uint32_t y, uint32_t slice);
// mip 0, sample 0: rows of `width` elements, tightly packed.
bool detile(const TiledSurface&, const uint8_t* src, uint8_t* dst, std::string& why);
bool tile(const TiledSurface&, const uint8_t* src, uint8_t* dst, std::string& why);

}  // namespace vpps4
