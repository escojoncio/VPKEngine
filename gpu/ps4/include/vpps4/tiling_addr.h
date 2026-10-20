// gfx7 surface addressing, in the subset of C++ that Metal Shading Language also accepts: this one
// header is compiled by the CPU path (tiling.cpp), by the tests, and pasted into the Metal kernels
// of GpuTiler (gpu_tiler.cpp), so all three compute an element's address from the same numbers.
//
// VptLevel holds what the address of an element of one mip level depends on, beyond the element's
// coordinates; the host fills it from the tile mode tables (computeLayout). Offsets are 32-bit
// inside a level (a level of 4 GiB or more is refused by the host).
#pragma once

#ifdef __METAL_VERSION__
#define VPT_CONST constant
typedef uint vpt_u32;
#else
#include <cstdint>
#define VPT_CONST const
using vpt_u32 = uint32_t;
#endif

#define VPT_LINEAR 0u
#define VPT_MICRO 1u  // 1D tiled: micro tiles in raster order
#define VPT_MACRO 2u  // 2D / 3D / PRT: micro tiles in macro tiles spread over pipes and banks

#define VPT_DISPLAY 0u
#define VPT_THIN 1u
#define VPT_DEPTH 2u
#define VPT_THICK 3u

// VptLevel::flags
#define VPT_F_PRT 1u         // pipe and bank bits repeat per macro tile
#define VPT_F_ROT2D 2u       // bank rotates with the slice (2D thin / thick / xthick)
#define VPT_F_ROT3D 4u       // pipe and bank rotate with the slice (3D modes)
#define VPT_F_SPLITROT 8u    // tile split planes rotate the bank (thin 2D / 3D modes)

struct VptLevel {
    vpt_u32 kind;        // VPT_LINEAR, VPT_MICRO, VPT_MACRO
    vpt_u32 micro;       // VPT_DISPLAY, VPT_THIN, VPT_DEPTH, VPT_THICK (element order inside a micro tile)
    vpt_u32 bpe;         // bytes per element: 1, 2, 4, 8, 16
    vpt_u32 thickness;   // slices per micro tile: 1, 4, 8
    vpt_u32 samples;     // 1, 2, 4, 8
    vpt_u32 pitch;       // allocated elements per row
    vpt_u32 height;      // allocated rows
    vpt_u32 pipeKind;    // 0 P2, 1 P8_32x32_16x16, 2 P8_32x32_8x16
    vpt_u32 pipeBits;    // log2 of the pipe count
    vpt_u32 bankBits;    // log2 of the bank count
    vpt_u32 bankHeight;  // micro tiles of a bank, vertically (bank width is 1 on the PS4)
    vpt_u32 macroPitch;  // macro tile extent, elements
    vpt_u32 macroHeight;
    vpt_u32 macroPerRow; // macro tiles per row of the level
    vpt_u32 macroRows;
    vpt_u32 splitBytes;  // bytes of one tile split plane of a micro tile (== the micro tile if no split)
    vpt_u32 splits;      // tile split planes per micro tile
    vpt_u32 flags;
    vpt_u32 bankSwizzle;
    vpt_u32 pipeSwizzle;
};

// Index of the element inside its 8x8(x thickness) micro tile.
inline vpt_u32 vpt_micro_index(vpt_u32 micro, vpt_u32 bpe, vpt_u32 thickness, vpt_u32 x, vpt_u32 y, vpt_u32 z) {
    const vpt_u32 x0 = x & 1u, x1 = (x >> 1) & 1u, x2 = (x >> 2) & 1u;
    const vpt_u32 y0 = y & 1u, y1 = (y >> 1) & 1u, y2 = (y >> 2) & 1u;
    vpt_u32 p0, p1, p2, p3, p4, p5, p6 = 0u, p7 = 0u, p8 = 0u;
    if (micro == VPT_DISPLAY) {
        if (bpe == 1u) {
            p0 = x0; p1 = x1; p2 = x2; p3 = y1; p4 = y0; p5 = y2;
        } else if (bpe == 2u) {
            p0 = x0; p1 = x1; p2 = x2; p3 = y0; p4 = y1; p5 = y2;
        } else if (bpe == 4u) {
            p0 = x0; p1 = x1; p2 = y0; p3 = x2; p4 = y1; p5 = y2;
        } else if (bpe == 8u) {
            p0 = x0; p1 = y0; p2 = x1; p3 = x2; p4 = y1; p5 = y2;
        } else {
            p0 = y0; p1 = x0; p2 = x1; p3 = x2; p4 = y1; p5 = y2;
        }
    } else if (micro == VPT_THICK) {
        const vpt_u32 z0 = z & 1u, z1 = (z >> 1) & 1u, z2 = (z >> 2) & 1u;
        if (bpe <= 2u) {
            p0 = x0; p1 = y0; p2 = x1; p3 = y1; p4 = z0; p5 = z1;
        } else if (bpe == 4u) {
            p0 = x0; p1 = y0; p2 = x1; p3 = z0; p4 = y1; p5 = z1;
        } else {
            p0 = x0; p1 = y0; p2 = z0; p3 = x1; p4 = y1; p5 = z1;
        }
        p6 = x2;
        p7 = y2;
        p8 = thickness == 8u ? z2 : 0u;
    } else {  // thin and depth: Z order
        p0 = x0; p1 = y0; p2 = x1; p3 = y1; p4 = x2; p5 = y2;
    }
    return p0 | (p1 << 1) | (p2 << 2) | (p3 << 3) | (p4 << 4) | (p5 << 5) | (p6 << 6) | (p7 << 7) | (p8 << 8);
}

inline vpt_u32 vpt_pipe(VPT_CONST VptLevel& L, vpt_u32 x, vpt_u32 y, vpt_u32 slice) {
    const vpt_u32 tx = x >> 3, ty = y >> 3;
    const vpt_u32 x3 = tx & 1u, x4 = (tx >> 1) & 1u, x5 = (tx >> 2) & 1u;
    const vpt_u32 y3 = ty & 1u, y4 = (ty >> 1) & 1u, y5 = (ty >> 2) & 1u;
    vpt_u32 pipe;
    if (L.pipeKind == 0u) {
        pipe = x3 ^ y3;
    } else if (L.pipeKind == 2u) {  // P8_32x32_8x16
        pipe = (x4 ^ y3 ^ x5) | ((x3 ^ y4) << 1) | ((x5 ^ y5) << 2);
    } else {                        // P8_32x32_16x16
        pipe = (x3 ^ y3 ^ x4) | ((x4 ^ y4) << 1) | ((x5 ^ y5) << 2);
    }
    const vpt_u32 pipes = 1u << L.pipeBits;
    vpt_u32 swizzle = L.pipeSwizzle;
    if (L.flags & VPT_F_ROT3D) {
        const vpt_u32 per = pipes / 2u - 1u;
        swizzle += (per > 1u ? per : 1u) * (slice / L.thickness);
    }
    return pipe ^ (swizzle & (pipes - 1u));
}

inline vpt_u32 vpt_bank(VPT_CONST VptLevel& L, vpt_u32 x, vpt_u32 y, vpt_u32 slice, vpt_u32 splitSlice) {
    const vpt_u32 tx = x >> (3u + L.pipeBits);  // bank width 1
    const vpt_u32 ty = (y >> 3) / L.bankHeight;
    const vpt_u32 x3 = tx & 1u, x4 = (tx >> 1) & 1u, x5 = (tx >> 2) & 1u, x6 = (tx >> 3) & 1u;
    const vpt_u32 y3 = ty & 1u, y4 = (ty >> 1) & 1u, y5 = (ty >> 2) & 1u, y6 = (ty >> 3) & 1u;
    const vpt_u32 banks = 1u << L.bankBits;
    vpt_u32 bank;
    if (banks == 16u) {
        bank = (x3 ^ y6) | ((x4 ^ y5 ^ y6) << 1) | ((x5 ^ y4) << 2) | ((x6 ^ y3) << 3);
    } else if (banks == 8u) {
        bank = (x3 ^ y5) | ((x4 ^ y4 ^ y5) << 1) | ((x5 ^ y3) << 2);
    } else if (banks == 4u) {
        bank = (x3 ^ y4) | ((x4 ^ y3) << 1);
    } else {
        bank = x3 ^ y3;
    }
    vpt_u32 rotation = 0u;
    if (L.flags & VPT_F_ROT2D) {
        rotation = (banks / 2u - 1u) * (slice / L.thickness);
    } else if (L.flags & VPT_F_ROT3D) {
        const vpt_u32 pipes = 1u << L.pipeBits;
        const vpt_u32 per = pipes / 2u - 1u;
        rotation = (per > 1u ? per : 1u) * (slice / L.thickness) / pipes;
    }
    bank ^= L.bankSwizzle + rotation;
    if (L.flags & VPT_F_SPLITROT) bank ^= (banks / 2u + 1u) * splitSlice;
    return bank & (banks - 1u);
}

// Byte offset of element (x, y) of `slice`, sample `sample`, from the start of the level.
inline vpt_u32 vpt_offset(VPT_CONST VptLevel& L, vpt_u32 x, vpt_u32 y, vpt_u32 slice, vpt_u32 sample) {
    if (L.kind == VPT_LINEAR) return slice * (L.pitch * L.height * L.bpe) + (y * L.pitch + x) * L.bpe;
    const vpt_u32 tileBytes = 64u * L.thickness * L.bpe * L.samples;  // a micro tile, every sample
    const vpt_u32 pixel = vpt_micro_index(L.micro, L.bpe, L.thickness, x & 7u, y & 7u, slice);
    vpt_u32 elem;
    if (L.micro == VPT_DEPTH) elem = pixel * L.bpe * L.samples + sample * L.bpe;  // samples interleaved per pixel
    else elem = sample * (tileBytes / L.samples) + pixel * L.bpe;                  // one plane per sample
    if (L.kind == VPT_MICRO) {
        const vpt_u32 sliceBytes = L.pitch * L.height * L.thickness * L.bpe * L.samples;
        return (slice / L.thickness) * sliceBytes + ((y >> 3) * (L.pitch >> 3) + (x >> 3)) * tileBytes + elem;
    }
    vpt_u32 splitSlice = 0u, planeBytes = tileBytes;
    if (L.splits > 1u) {
        splitSlice = elem / L.splitBytes;
        elem = elem % L.splitBytes;
        planeBytes = L.splitBytes;
    }
    // Sizes below are per pipe and bank: the macro tile's bytes shared among them.
    const vpt_u32 macroTileBytes = (planeBytes * ((L.macroPitch >> 3) * (L.macroHeight >> 3))) >> (L.pipeBits + L.bankBits);
    const vpt_u32 macroOffset = ((y / L.macroHeight) * L.macroPerRow + x / L.macroPitch) * macroTileBytes;
    const vpt_u32 sliceBytes = L.macroPerRow * L.macroRows * macroTileBytes;
    const vpt_u32 sliceOffset = sliceBytes * (splitSlice + L.splits * (slice / L.thickness));
    const vpt_u32 tileOffset = ((y >> 3) % L.bankHeight) * planeBytes;  // bank width 1: a column of micro tiles
    const vpt_u32 total = sliceOffset + macroOffset + elem + tileOffset;
    vpt_u32 px = x, py = y;
    if (L.flags & VPT_F_PRT) {
        px = x % L.macroPitch;
        py = y % L.macroHeight;
    }
    const vpt_u32 pipe = vpt_pipe(L, px, py, slice);
    const vpt_u32 bank = vpt_bank(L, px, py, slice, splitSlice);
    return (total & 255u) | (pipe << 8) | (bank << (8u + L.pipeBits)) | ((total >> 8) << (8u + L.pipeBits + L.bankBits));
}

// ---- The copy kernels' view: one thread per element ------------------------------------------
// The kernels (GpuTiler) run a grid of width x height x slices threads. A thread whose coordinates
// are inside the copy moves bpe bytes between the tiled byte vpt_tiled_offset() and the linear byte
// vpt_linear_offset(); detiling reads the first, writes the second, tiling the reverse.

struct VptParams {
    VptLevel level;
    vpt_u32 width, height, slices;  // extent of the copy: elements, rows, slices
    vpt_u32 firstSlice;             // slice of the level that grid z = 0 is
    vpt_u32 sample;
    vpt_u32 pad0, pad1, pad2;
};

inline bool vpt_in_range(VPT_CONST VptParams& P, vpt_u32 x, vpt_u32 y, vpt_u32 z) {
    return x < P.width && y < P.height && z < P.slices;
}
inline vpt_u32 vpt_tiled_offset(VPT_CONST VptParams& P, vpt_u32 x, vpt_u32 y, vpt_u32 z) {
    return vpt_offset(P.level, x, y, P.firstSlice + z, P.sample);
}
inline vpt_u32 vpt_linear_offset(VPT_CONST VptParams& P, vpt_u32 x, vpt_u32 y, vpt_u32 z) {
    return ((z * P.height + y) * P.width + x) * P.level.bpe;
}
