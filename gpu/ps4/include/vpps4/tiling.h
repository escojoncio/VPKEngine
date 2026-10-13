// PS4 (Liverpool, gfx7) surface tiling: where an element of a tiled surface lives in memory.
//
// A T# or CB names its layout by TILE_MODE_INDEX, an index into the GB_TILE_MODE table the PS4
// system software programs (array mode, micro tile mode, pipe config, tile split); for 2D tiled
// surfaces the bank geometry comes from GB_MACROTILE_MODE, picked by the bytes of one micro tile.
// detile() turns mip 0 of such a surface (all its slices) into rows of elements, which is what a
// texture upload wants. Elements are texels, or 4x4 blocks for block-compressed formats.
//
// Covered: linear (8, 31), 1D thin (5, 9, 13) and 2D thin (0-4, 10, 14) with display, thin and
// depth micro tiles, any slice count, 1 sample; 2D surfaces too small for one macro tile are laid
// out as 1D thin, as the address library does. Not covered (reported, left unfilled): thick and
// 3D modes, PRT modes, a tile split smaller than a micro tile (depth surfaces sampled from memory).
#pragma once

#include <cstdint>
#include <string>

namespace vpps4 {

struct TiledSurface {
    uint8_t tileIndex = 8;        // TILE_MODE_INDEX
    uint32_t bytesPerElement = 4; // 1, 2, 4, 8 or 16
    uint32_t width = 1, height = 1;  // elements
    uint32_t pitch = 0;           // elements per row as allocated (0: width)
    uint32_t slices = 1;
};

// Bytes of guest memory the surface's mip 0 spans (all slices), or 0 if its layout is not covered
// (`why` says what is missing).
uint64_t tiledBytes(const TiledSurface&, std::string& why);

// Element (x, y, slice)'s byte offset from the surface's base; UINT64_MAX if not covered.
uint64_t tiledOffset(const TiledSurface&, uint32_t x, uint32_t y, uint32_t slice);

// Copies mip 0 from `src` (tiledBytes() long) to `dst`: slice after slice, rows of `width`
// elements, tightly packed. False (dst untouched, `why` set) if the layout is not covered.
bool detile(const TiledSurface&, const uint8_t* src, uint8_t* dst, std::string& why);

// The inverse, for tests and for writing a render target back to guest memory.
bool tile(const TiledSurface&, const uint8_t* src, uint8_t* dst, std::string& why);

}  // namespace vpps4
