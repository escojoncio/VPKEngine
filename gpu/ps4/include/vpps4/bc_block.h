// Block-compressed texture decoding (BC1-BC7, the GCN data formats 35-41), one 4x4 block at a time, in the subset of C++
// that Metal Shading Language also accepts: this one header is compiled by the CPU decoder (bc_decode.cpp, the fallback and
// the tests' oracle) and pasted into the Metal kernel of GpuTiler (gpu_tiler.cpp), so both decode with the same code.
// Written from the public BC1-BC7 format specifications (Microsoft Direct3D 10/11 texture compression docs).
//
// A block goes in as four little-endian words (a block of 8 bytes: the last two words zero); the 16 texels come out as
// two words each (out[2 * t], out[2 * t + 1] = bytes 0..3 and 4..7 of the texel in memory order):
//   BC1 / BC2 / BC3 / BC7  RGBA8 (R in the lowest byte)        BC4  R8        BC5  RG8        BC6H  RGBA16F
#pragma once

#ifdef __METAL_VERSION__
#define BC_T thread
#define BC_CONST constant
#define BC_DEV device
typedef uint bc_u;
typedef int bc_i;
typedef uchar bc_u8;
// (GpuTiler pastes tiling_addr.h before this header)
#else
#include <cstdint>

#include "vpps4/tiling_addr.h"
#define BC_T
#define BC_CONST static const
#define BC_DEV
typedef uint32_t bc_u;
typedef int32_t bc_i;
typedef uint8_t bc_u8;
#endif

#define BC_BC1 0u
#define BC_BC2 1u
#define BC_BC3 2u
#define BC_BC4 3u
#define BC_BC5 4u
#define BC_BC6H 5u
#define BC_BC7 6u

// ---- Bit reading ---------------------------------------------------------------------------------

struct BcBits {
    bc_u w[4];
    bc_u pos;
};

inline bc_u bc_get(BC_T BcBits& s, bc_u n) {
    if (n == 0u) return 0u;
    const bc_u i = s.pos >> 5, o = s.pos & 31u;
    if (i > 3u) return 0u;
    bc_u v = s.w[i] >> o;
    if (o + n > 32u && i < 3u) v |= s.w[i + 1u] << (32u - o);
    s.pos += n;
    return n >= 32u ? v : (v & ((1u << n) - 1u));
}

inline bc_u bc_byte(BC_T const bc_u* b, bc_u i) { return (b[i >> 2] >> ((i & 3u) * 8u)) & 255u; }

// ---- BC1 / BC2 / BC3 / BC4 / BC5 -------------------------------------------------------------------

inline bc_u bc_expand5(bc_u v) { return (v << 3) | (v >> 2); }
inline bc_u bc_expand6(bc_u v) { return (v << 2) | (v >> 4); }

// A BC1 colour block at byte `off` of the block, 16 RGBA8 texels. `opaque`: always the four-colour mode (the colour half
// of BC2 / BC3).
inline void bc1_block(BC_T const bc_u* b, bc_u off, bc_u opaque, BC_T bc_u* out) {
    const bc_u c0 = bc_byte(b, off) | (bc_byte(b, off + 1u) << 8), c1 = bc_byte(b, off + 2u) | (bc_byte(b, off + 3u) << 8);
    bc_u col[4][4];
    col[0][0] = bc_expand5((c0 >> 11) & 31u), col[0][1] = bc_expand6((c0 >> 5) & 63u), col[0][2] = bc_expand5(c0 & 31u);
    col[1][0] = bc_expand5((c1 >> 11) & 31u), col[1][1] = bc_expand6((c1 >> 5) & 63u), col[1][2] = bc_expand5(c1 & 31u);
    col[0][3] = 255u, col[1][3] = 255u, col[2][3] = 255u;
    const bool four = c0 > c1 || opaque != 0u;
    for (bc_u c = 0u; c < 3u; ++c) {
        if (four) {
            col[2][c] = (2u * col[0][c] + col[1][c]) / 3u;
            col[3][c] = (col[0][c] + 2u * col[1][c]) / 3u;
        } else {
            col[2][c] = (col[0][c] + col[1][c]) / 2u;
            col[3][c] = 0u;
        }
    }
    col[3][3] = four ? 255u : 0u;
    bc_u idx = bc_byte(b, off + 4u) | (bc_byte(b, off + 5u) << 8) | (bc_byte(b, off + 6u) << 16) | (bc_byte(b, off + 7u) << 24);
    for (bc_u t = 0u; t < 16u; ++t, idx >>= 2) {
        const bc_u i = idx & 3u;
        out[t] = col[i][0] | (col[i][1] << 8) | (col[i][2] << 16) | (col[i][3] << 24);
    }
}

// A BC4 block (endpoints, then 16 x 3 bits of indices) at byte `off`: 16 bytes, unsigned or two's complement signed.
inline void bc4_block(BC_T const bc_u* b, bc_u off, bc_u isSigned, BC_T bc_u* v) {
    bc_i c[8];
    bc_i c0 = bc_i(bc_byte(b, off)), c1 = bc_i(bc_byte(b, off + 1u));
    if (isSigned != 0u) {
        if (c0 > 127) c0 -= 256;
        if (c1 > 127) c1 -= 256;
        if (c0 < -127) c0 = -127;
        if (c1 < -127) c1 = -127;
    }
    c[0] = c0, c[1] = c1;
    if (c0 > c1) {
        for (bc_i i = 1; i < 7; ++i) c[1 + i] = ((7 - i) * c0 + i * c1) / 7;
    } else {
        for (bc_i i = 1; i < 5; ++i) c[1 + i] = ((5 - i) * c0 + i * c1) / 5;
        c[6] = isSigned != 0u ? -127 : 0;
        c[7] = isSigned != 0u ? 127 : 255;
    }
    for (bc_u t = 0u; t < 16u; ++t) {
        const bc_u p = 3u * t, at = off + 2u + (p >> 3);
        const bc_u idx = ((bc_byte(b, at) | (bc_byte(b, at + 1u) << 8)) >> (p & 7u)) & 7u;
        v[t] = bc_u(c[idx]) & 255u;
    }
}

// ---- BC7 ------------------------------------------------------------------------------------------

// Partitions: one bit per texel (2 subsets) or two (3 subsets), texel 0 in the low bits. Anchors: the texel of
// subset 1 (2 subsets) and subsets 1 and 2 (3 subsets) whose index drops its top bit; subset 0 always has texel 0.
BC_CONST bc_u kBcPart2[64] = {
    0xcccc, 0x8888, 0xeeee, 0xecc8, 0xc880, 0xfeec, 0xfec8, 0xec80, 0xc800, 0xffec, 0xfe80, 0xe800, 0xffe8, 0xff00, 0xfff0, 0xf000,
    0xf710, 0x008e, 0x7100, 0x08ce, 0x008c, 0x7310, 0x3100, 0x8cce, 0x088c, 0x3110, 0x6666, 0x366c, 0x17e8, 0x0ff0, 0x718e, 0x399c,
    0xaaaa, 0xf0f0, 0x5a5a, 0x33cc, 0x3c3c, 0x55aa, 0x9696, 0xa55a, 0x73ce, 0x13c8, 0x324c, 0x3bdc, 0x6996, 0xc33c, 0x9966, 0x0660,
    0x0272, 0x04e4, 0x4e40, 0x2720, 0xc936, 0x936c, 0x39c6, 0x639c, 0x9336, 0x9cc6, 0x817e, 0xe718, 0xccf0, 0x0fcc, 0x7744, 0xee22,
};
BC_CONST bc_u kBcPart3[64] = {
    0xaa685050u, 0x6a5a5040u, 0x5a5a4200u, 0x5450a0a8u, 0xa5a50000u, 0xa0a05050u, 0x5555a0a0u, 0x5a5a5050u, 0xaa550000u, 0xaa555500u,
    0xaaaa5500u, 0x90909090u, 0x94949494u, 0xa4a4a4a4u, 0xa9a59450u, 0x2a0a4250u, 0xa5945040u, 0x0a425054u, 0xa5a5a500u, 0x55a0a0a0u,
    0xa8a85454u, 0x6a6a4040u, 0xa4a45000u, 0x1a1a0500u, 0x0050a4a4u, 0xaaa59090u, 0x14696914u, 0x69691400u, 0xa08585a0u, 0xaa821414u,
    0x50a4a450u, 0x6a5a0200u, 0xa9a58000u, 0x5090a0a8u, 0xa8a09050u, 0x24242424u, 0x00aa5500u, 0x24924924u, 0x24499224u, 0x50a50a50u,
    0x500aa550u, 0xaaaa4444u, 0x66660000u, 0xa5a0a5a0u, 0x50a050a0u, 0x69286928u, 0x44aaaa44u, 0x66666600u, 0xaa444444u, 0x54a854a8u,
    0x95809580u, 0x96969600u, 0xa85454a8u, 0x80959580u, 0xaa141414u, 0x96960000u, 0xaaaa1414u, 0xa05050a0u, 0xa0a5a5a0u, 0x96000000u,
    0x40804080u, 0xa9a8a9a8u, 0xaaaaaa44u, 0x2a4a5254u,
};
BC_CONST bc_u kBcAnchor2[64] = {
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2,  8,  2,  2,  8,  8,  15, 2,  8,  2,  2,  8,  8,  2,  2,
    15, 15, 6,  8,  2,  8,  15, 15, 2,  8,  2,  2,  2,  15, 15, 6,  6,  2,  6,  8,  15, 15, 2,  2,  15, 15, 15, 15, 15, 2,  2,  15,
};
BC_CONST bc_u kBcAnchor3a[64] = {
    3,  3,  15, 15, 8,  3,  15, 15, 8,  8,  6,  6,  6,  5,  3,  3,  3,  3,  8,  15, 3,  3,  6,  10, 5,  8,  8,  6,  8,  5,  15, 15,
    8,  15, 3,  5,  6,  10, 8,  15, 15, 3,  15, 5,  15, 15, 15, 15, 3,  15, 5,  5,  5,  8,  5,  10, 5,  10, 8,  13, 15, 12, 3,  3,
};
BC_CONST bc_u kBcAnchor3b[64] = {
    15, 8,  8,  3,  15, 15, 3,  8,  15, 15, 15, 15, 15, 15, 15, 8,  15, 8,  15, 3,  15, 8,  15, 8,  3,  15, 6,  10, 15, 15, 10, 8,
    15, 3,  15, 10, 10, 8,  9,  10, 6,  15, 8,  15, 3,  6,  6,  8,  15, 3,  15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3,  15, 15, 8,
};

BC_CONST bc_u kBcWeight2[4] = {0, 21, 43, 64};
BC_CONST bc_u kBcWeight3[8] = {0, 9, 18, 27, 37, 46, 55, 64};
BC_CONST bc_u kBcWeight4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};
inline bc_u bc_weight(bc_u bits, bc_u index) { return bits == 2u ? kBcWeight2[index] : bits == 3u ? kBcWeight3[index] : kBcWeight4[index]; }

// Per mode: subsets, partition bits, rotation bits, index selection bit, colour bits, alpha bits, endpoint P bit, shared P
// bit, index bits, second index bits.
BC_CONST bc_u kBc7Mode[80] = {
    3, 4, 0, 0, 4, 0, 1, 0, 3, 0,  2, 6, 0, 0, 6, 0, 0, 1, 3, 0,  3, 6, 0, 0, 5, 0, 0, 0, 2, 0,  2, 6, 0, 0, 7, 0, 1, 0, 2, 0,
    1, 0, 2, 1, 5, 6, 0, 0, 2, 3,  1, 0, 2, 0, 7, 8, 0, 0, 2, 2,  1, 0, 0, 0, 7, 7, 1, 0, 4, 0,  2, 6, 0, 0, 5, 5, 1, 0, 2, 0,
};

inline bc_u bc_expand_bits(bc_u v, bc_u bits) { return (v << (8u - bits)) | (v >> (2u * bits - 8u)); }

inline bc_u bc7_subset(bc_u subsets, bc_u part, bc_u t) {
    return subsets == 1u ? 0u : subsets == 2u ? ((kBcPart2[part] >> t) & 1u) : ((kBcPart3[part] >> (2u * t)) & 3u);
}
inline bool bc7_anchor(bc_u subsets, bc_u part, bc_u t, bc_u s) {
    return t == 0u || (s == 1u && t == (subsets == 2u ? kBcAnchor2[part] : kBcAnchor3a[part])) || (s == 2u && t == kBcAnchor3b[part]);
}

inline void bc7_block(BC_T const bc_u* block, BC_T bc_u* out) {
    BcBits br;
    for (bc_u i = 0u; i < 4u; ++i) br.w[i] = block[i];
    br.pos = 0u;
    bc_u mode = 0u;
    while (mode < 8u && bc_get(br, 1u) == 0u) ++mode;
    if (mode == 8u) {  // reserved: all zero
        for (bc_u t = 0u; t < 16u; ++t) out[t] = 0u;
        return;
    }
    const bc_u m = mode * 10u;
    const bc_u subsets = kBc7Mode[m], colorBits = kBc7Mode[m + 4u], alphaBits = kBc7Mode[m + 5u];
    const bc_u indexBits = kBc7Mode[m + 8u], indexBits2 = kBc7Mode[m + 9u];
    const bc_u part = bc_get(br, kBc7Mode[m + 1u]), rotation = bc_get(br, kBc7Mode[m + 2u]), indexSel = bc_get(br, kBc7Mode[m + 3u]);
    const bc_u ends = 2u * subsets;
    bc_u ep[6][4];
    bc_u pbit[6];
    for (bc_u e = 0u; e < 6u; ++e) {
        pbit[e] = 0u;
        for (bc_u c = 0u; c < 4u; ++c) ep[e][c] = 0u;
    }
    for (bc_u c = 0u; c < 3u; ++c)
        for (bc_u e = 0u; e < ends; ++e) ep[e][c] = bc_get(br, colorBits);
    if (alphaBits != 0u)
        for (bc_u e = 0u; e < ends; ++e) ep[e][3] = bc_get(br, alphaBits);
    const bool hasP = kBc7Mode[m + 6u] != 0u || kBc7Mode[m + 7u] != 0u;
    if (kBc7Mode[m + 6u] != 0u)
        for (bc_u e = 0u; e < ends; ++e) pbit[e] = bc_get(br, 1u);
    if (kBc7Mode[m + 7u] != 0u)
        for (bc_u s = 0u; s < subsets; ++s) {
            pbit[2u * s] = bc_get(br, 1u);
            pbit[2u * s + 1u] = pbit[2u * s];
        }
    for (bc_u e = 0u; e < ends; ++e) {
        for (bc_u c = 0u; c < 3u; ++c) {
            const bc_u v = hasP ? ((ep[e][c] << 1) | pbit[e]) : ep[e][c];
            ep[e][c] = bc_expand_bits(v, colorBits + (hasP ? 1u : 0u));
        }
        if (alphaBits != 0u) {
            const bc_u v = hasP ? ((ep[e][3] << 1) | pbit[e]) : ep[e][3];
            ep[e][3] = bc_expand_bits(v, alphaBits + (hasP ? 1u : 0u));
        } else {
            ep[e][3] = 255u;
        }
    }
    bc_u idx[16], idx2[16];
    for (bc_u t = 0u; t < 16u; ++t) {
        idx2[t] = 0u;
        idx[t] = bc_get(br, indexBits - (bc7_anchor(subsets, part, t, bc7_subset(subsets, part, t)) ? 1u : 0u));
    }
    if (indexBits2 != 0u)
        for (bc_u t = 0u; t < 16u; ++t) idx2[t] = bc_get(br, indexBits2 - (t == 0u ? 1u : 0u));
    for (bc_u t = 0u; t < 16u; ++t) {
        const bc_u s = bc7_subset(subsets, part, t);
        // Modes 4 and 5 hold two index sets: one for colour, one for alpha (swapped by the index selection bit in mode 4).
        const bool swap = indexBits2 != 0u && indexSel != 0u;
        const bc_u cBits = swap ? indexBits2 : indexBits;
        const bc_u aBits = indexBits2 != 0u ? (indexSel != 0u ? indexBits : indexBits2) : indexBits;
        const bc_u ci = swap ? idx2[t] : idx[t];
        const bc_u ai = indexBits2 != 0u ? (indexSel != 0u ? idx[t] : idx2[t]) : idx[t];
        const bc_u wc = bc_weight(cBits, ci), wa = bc_weight(aBits, ai);
        bc_u px[4];
        for (bc_u c = 0u; c < 4u; ++c) {
            const bc_u w = c < 3u ? wc : wa;
            px[c] = ((64u - w) * ep[2u * s][c] + w * ep[2u * s + 1u][c] + 32u) >> 6;
        }
        if (rotation != 0u) {
            const bc_u tmp = px[3];
            px[3] = px[rotation - 1u];
            px[rotation - 1u] = tmp;
        }
        out[t] = (px[0] & 255u) | ((px[1] & 255u) << 8) | ((px[2] & 255u) << 16) | ((px[3] & 255u) << 24);
    }
}

// ---- BC6H -----------------------------------------------------------------------------------------

// The fields of the header in stream order: endpoint W (first of subset 0), X (second of subset 0), Y and Z (the
// subset 1 pair) per colour, and the partition D.
#define BC6_RW 0u
#define BC6_GW 1u
#define BC6_BW 2u
#define BC6_RX 3u
#define BC6_GX 4u
#define BC6_BX 5u
#define BC6_RY 6u
#define BC6_GY 7u
#define BC6_BY 8u
#define BC6_RZ 9u
#define BC6_GZ 10u
#define BC6_BZ 11u
#define BC6_DD 12u
// A token reads `n` bits into bits lo.. of a field; a token with n = 0 ends a mode's list.
#define BC6_TOK(f, lo, n) (((n) << 16) | ((lo) << 8) | (f))
#define BC6_END 0u

BC_CONST bc_u kBc6Tok[] = {
    // 0: mode 1
    BC6_TOK(BC6_GY, 4, 1), BC6_TOK(BC6_BY, 4, 1), BC6_TOK(BC6_BZ, 4, 1), BC6_TOK(BC6_RW, 0, 10), BC6_TOK(BC6_GW, 0, 10), BC6_TOK(BC6_BW, 0, 10),
    BC6_TOK(BC6_RX, 0, 5), BC6_TOK(BC6_GZ, 4, 1), BC6_TOK(BC6_GY, 0, 4), BC6_TOK(BC6_GX, 0, 5), BC6_TOK(BC6_BZ, 0, 1), BC6_TOK(BC6_GZ, 0, 4),
    BC6_TOK(BC6_BX, 0, 5), BC6_TOK(BC6_BZ, 1, 1), BC6_TOK(BC6_BY, 0, 4), BC6_TOK(BC6_RY, 0, 5), BC6_TOK(BC6_BZ, 2, 1), BC6_TOK(BC6_RZ, 0, 5),
    BC6_TOK(BC6_BZ, 3, 1), BC6_TOK(BC6_DD, 0, 5), BC6_END,
    // 21: mode 2
    BC6_TOK(BC6_GY, 5, 1), BC6_TOK(BC6_GZ, 4, 1), BC6_TOK(BC6_GZ, 5, 1), BC6_TOK(BC6_RW, 0, 7), BC6_TOK(BC6_BZ, 0, 1), BC6_TOK(BC6_BZ, 1, 1),
    BC6_TOK(BC6_BY, 4, 1), BC6_TOK(BC6_GW, 0, 7), BC6_TOK(BC6_BY, 5, 1), BC6_TOK(BC6_BZ, 2, 1), BC6_TOK(BC6_GY, 4, 1), BC6_TOK(BC6_BW, 0, 7),
    BC6_TOK(BC6_BZ, 3, 1), BC6_TOK(BC6_BZ, 5, 1), BC6_TOK(BC6_BZ, 4, 1), BC6_TOK(BC6_RX, 0, 6), BC6_TOK(BC6_GY, 0, 4), BC6_TOK(BC6_GX, 0, 6),
    BC6_TOK(BC6_GZ, 0, 4), BC6_TOK(BC6_BX, 0, 6), BC6_TOK(BC6_BY, 0, 4), BC6_TOK(BC6_RY, 0, 6), BC6_TOK(BC6_RZ, 0, 6), BC6_TOK(BC6_DD, 0, 5),
    BC6_END,
    // 46: mode 3
    BC6_TOK(BC6_RW, 0, 10), BC6_TOK(BC6_GW, 0, 10), BC6_TOK(BC6_BW, 0, 10), BC6_TOK(BC6_RX, 0, 5), BC6_TOK(BC6_RW, 10, 1), BC6_TOK(BC6_GY, 0, 4),
    BC6_TOK(BC6_GX, 0, 4), BC6_TOK(BC6_GW, 10, 1), BC6_TOK(BC6_BZ, 0, 1), BC6_TOK(BC6_GZ, 0, 4), BC6_TOK(BC6_BX, 0, 4), BC6_TOK(BC6_BW, 10, 1),
    BC6_TOK(BC6_BZ, 1, 1), BC6_TOK(BC6_BY, 0, 4), BC6_TOK(BC6_RY, 0, 5), BC6_TOK(BC6_BZ, 2, 1), BC6_TOK(BC6_RZ, 0, 5), BC6_TOK(BC6_BZ, 3, 1),
    BC6_TOK(BC6_DD, 0, 5), BC6_END,
    // 66: mode 4
    BC6_TOK(BC6_RW, 0, 10), BC6_TOK(BC6_GW, 0, 10), BC6_TOK(BC6_BW, 0, 10), BC6_TOK(BC6_RX, 0, 4), BC6_TOK(BC6_RW, 10, 1), BC6_TOK(BC6_GZ, 4, 1),
    BC6_TOK(BC6_GY, 0, 4), BC6_TOK(BC6_GX, 0, 5), BC6_TOK(BC6_GW, 10, 1), BC6_TOK(BC6_GZ, 0, 4), BC6_TOK(BC6_BX, 0, 4), BC6_TOK(BC6_BW, 10, 1),
    BC6_TOK(BC6_BZ, 1, 1), BC6_TOK(BC6_BY, 0, 4), BC6_TOK(BC6_RY, 0, 4), BC6_TOK(BC6_BZ, 0, 1), BC6_TOK(BC6_BZ, 2, 1), BC6_TOK(BC6_RZ, 0, 4),
    BC6_TOK(BC6_GY, 4, 1), BC6_TOK(BC6_BZ, 3, 1), BC6_TOK(BC6_DD, 0, 5), BC6_END,
    // 88: mode 5
    BC6_TOK(BC6_RW, 0, 10), BC6_TOK(BC6_GW, 0, 10), BC6_TOK(BC6_BW, 0, 10), BC6_TOK(BC6_RX, 0, 4), BC6_TOK(BC6_RW, 10, 1), BC6_TOK(BC6_BY, 4, 1),
    BC6_TOK(BC6_GY, 0, 4), BC6_TOK(BC6_GX, 0, 4), BC6_TOK(BC6_GW, 10, 1), BC6_TOK(BC6_BZ, 0, 1), BC6_TOK(BC6_GZ, 0, 4), BC6_TOK(BC6_BX, 0, 5),
    BC6_TOK(BC6_BW, 10, 1), BC6_TOK(BC6_BY, 0, 4), BC6_TOK(BC6_RY, 0, 4), BC6_TOK(BC6_BZ, 1, 1), BC6_TOK(BC6_BZ, 2, 1), BC6_TOK(BC6_RZ, 0, 4),
    BC6_TOK(BC6_BZ, 4, 1), BC6_TOK(BC6_BZ, 3, 1), BC6_TOK(BC6_DD, 0, 5), BC6_END,
    // 110: mode 6
    BC6_TOK(BC6_RW, 0, 9), BC6_TOK(BC6_BY, 4, 1), BC6_TOK(BC6_GW, 0, 9), BC6_TOK(BC6_GY, 4, 1), BC6_TOK(BC6_BW, 0, 9), BC6_TOK(BC6_BZ, 4, 1),
    BC6_TOK(BC6_RX, 0, 5), BC6_TOK(BC6_GZ, 4, 1), BC6_TOK(BC6_GY, 0, 4), BC6_TOK(BC6_GX, 0, 5), BC6_TOK(BC6_BZ, 0, 1), BC6_TOK(BC6_GZ, 0, 4),
    BC6_TOK(BC6_BX, 0, 5), BC6_TOK(BC6_BZ, 1, 1), BC6_TOK(BC6_BY, 0, 4), BC6_TOK(BC6_RY, 0, 5), BC6_TOK(BC6_BZ, 2, 1), BC6_TOK(BC6_RZ, 0, 5),
    BC6_TOK(BC6_BZ, 3, 1), BC6_TOK(BC6_DD, 0, 5), BC6_END,
    // 131: mode 7
    BC6_TOK(BC6_RW, 0, 8), BC6_TOK(BC6_GZ, 4, 1), BC6_TOK(BC6_BY, 4, 1), BC6_TOK(BC6_GW, 0, 8), BC6_TOK(BC6_BZ, 2, 1), BC6_TOK(BC6_GY, 4, 1),
    BC6_TOK(BC6_BW, 0, 8), BC6_TOK(BC6_BZ, 3, 1), BC6_TOK(BC6_BZ, 4, 1), BC6_TOK(BC6_RX, 0, 6), BC6_TOK(BC6_GY, 0, 4), BC6_TOK(BC6_GX, 0, 5),
    BC6_TOK(BC6_BZ, 0, 1), BC6_TOK(BC6_GZ, 0, 4), BC6_TOK(BC6_BX, 0, 5), BC6_TOK(BC6_BZ, 1, 1), BC6_TOK(BC6_BY, 0, 4), BC6_TOK(BC6_RY, 0, 6),
    BC6_TOK(BC6_RZ, 0, 6), BC6_TOK(BC6_DD, 0, 5), BC6_END,
    // 152: mode 8
    BC6_TOK(BC6_RW, 0, 8), BC6_TOK(BC6_BZ, 0, 1), BC6_TOK(BC6_BY, 4, 1), BC6_TOK(BC6_GW, 0, 8), BC6_TOK(BC6_GY, 5, 1), BC6_TOK(BC6_GY, 4, 1),
    BC6_TOK(BC6_BW, 0, 8), BC6_TOK(BC6_GZ, 5, 1), BC6_TOK(BC6_BZ, 4, 1), BC6_TOK(BC6_RX, 0, 5), BC6_TOK(BC6_GZ, 4, 1), BC6_TOK(BC6_GY, 0, 4),
    BC6_TOK(BC6_GX, 0, 6), BC6_TOK(BC6_GZ, 0, 4), BC6_TOK(BC6_BX, 0, 5), BC6_TOK(BC6_BZ, 1, 1), BC6_TOK(BC6_BY, 0, 4), BC6_TOK(BC6_RY, 0, 5),
    BC6_TOK(BC6_BZ, 2, 1), BC6_TOK(BC6_RZ, 0, 5), BC6_TOK(BC6_BZ, 3, 1), BC6_TOK(BC6_DD, 0, 5), BC6_END,
    // 175: mode 9
    BC6_TOK(BC6_RW, 0, 8), BC6_TOK(BC6_BZ, 1, 1), BC6_TOK(BC6_BY, 4, 1), BC6_TOK(BC6_GW, 0, 8), BC6_TOK(BC6_BY, 5, 1), BC6_TOK(BC6_GY, 4, 1),
    BC6_TOK(BC6_BW, 0, 8), BC6_TOK(BC6_BZ, 5, 1), BC6_TOK(BC6_BZ, 4, 1), BC6_TOK(BC6_RX, 0, 5), BC6_TOK(BC6_GZ, 4, 1), BC6_TOK(BC6_GY, 0, 4),
    BC6_TOK(BC6_GX, 0, 5), BC6_TOK(BC6_BZ, 0, 1), BC6_TOK(BC6_GZ, 0, 4), BC6_TOK(BC6_BX, 0, 6), BC6_TOK(BC6_BY, 0, 4), BC6_TOK(BC6_RY, 0, 5),
    BC6_TOK(BC6_BZ, 2, 1), BC6_TOK(BC6_RZ, 0, 5), BC6_TOK(BC6_BZ, 3, 1), BC6_TOK(BC6_DD, 0, 5), BC6_END,
    // 198: mode 10
    BC6_TOK(BC6_RW, 0, 6), BC6_TOK(BC6_GZ, 4, 1), BC6_TOK(BC6_BZ, 0, 1), BC6_TOK(BC6_BZ, 1, 1), BC6_TOK(BC6_BY, 4, 1), BC6_TOK(BC6_GW, 0, 6),
    BC6_TOK(BC6_GY, 5, 1), BC6_TOK(BC6_BY, 5, 1), BC6_TOK(BC6_BZ, 2, 1), BC6_TOK(BC6_GY, 4, 1), BC6_TOK(BC6_BW, 0, 6), BC6_TOK(BC6_GZ, 5, 1),
    BC6_TOK(BC6_BZ, 3, 1), BC6_TOK(BC6_BZ, 5, 1), BC6_TOK(BC6_BZ, 4, 1), BC6_TOK(BC6_RX, 0, 6), BC6_TOK(BC6_GY, 0, 4), BC6_TOK(BC6_GX, 0, 6),
    BC6_TOK(BC6_GZ, 0, 4), BC6_TOK(BC6_BX, 0, 6), BC6_TOK(BC6_BY, 0, 4), BC6_TOK(BC6_RY, 0, 6), BC6_TOK(BC6_RZ, 0, 6), BC6_TOK(BC6_DD, 0, 5),
    BC6_END,
    // 223: mode 11
    BC6_TOK(BC6_RW, 0, 10), BC6_TOK(BC6_GW, 0, 10), BC6_TOK(BC6_BW, 0, 10), BC6_TOK(BC6_RX, 0, 10), BC6_TOK(BC6_GX, 0, 10), BC6_TOK(BC6_BX, 0, 10),
    BC6_END,
    // 230: mode 12
    BC6_TOK(BC6_RW, 0, 10), BC6_TOK(BC6_GW, 0, 10), BC6_TOK(BC6_BW, 0, 10), BC6_TOK(BC6_RX, 0, 9), BC6_TOK(BC6_RW, 10, 1), BC6_TOK(BC6_GX, 0, 9),
    BC6_TOK(BC6_GW, 10, 1), BC6_TOK(BC6_BX, 0, 9), BC6_TOK(BC6_BW, 10, 1), BC6_END,
    // 240: mode 13
    BC6_TOK(BC6_RW, 0, 10), BC6_TOK(BC6_GW, 0, 10), BC6_TOK(BC6_BW, 0, 10), BC6_TOK(BC6_RX, 0, 8), BC6_TOK(BC6_RW, 11, 1), BC6_TOK(BC6_RW, 10, 1),
    BC6_TOK(BC6_GX, 0, 8), BC6_TOK(BC6_GW, 11, 1), BC6_TOK(BC6_GW, 10, 1), BC6_TOK(BC6_BX, 0, 8), BC6_TOK(BC6_BW, 11, 1), BC6_TOK(BC6_BW, 10, 1),
    BC6_END,
    // 253: mode 14
    BC6_TOK(BC6_RW, 0, 10), BC6_TOK(BC6_GW, 0, 10), BC6_TOK(BC6_BW, 0, 10), BC6_TOK(BC6_RX, 0, 4), BC6_TOK(BC6_RW, 15, 1), BC6_TOK(BC6_RW, 14, 1),
    BC6_TOK(BC6_RW, 13, 1), BC6_TOK(BC6_RW, 12, 1), BC6_TOK(BC6_RW, 11, 1), BC6_TOK(BC6_RW, 10, 1), BC6_TOK(BC6_GX, 0, 4), BC6_TOK(BC6_GW, 15, 1),
    BC6_TOK(BC6_GW, 14, 1), BC6_TOK(BC6_GW, 13, 1), BC6_TOK(BC6_GW, 12, 1), BC6_TOK(BC6_GW, 11, 1), BC6_TOK(BC6_GW, 10, 1), BC6_TOK(BC6_BX, 0, 4),
    BC6_TOK(BC6_BW, 15, 1), BC6_TOK(BC6_BW, 14, 1), BC6_TOK(BC6_BW, 13, 1), BC6_TOK(BC6_BW, 12, 1), BC6_TOK(BC6_BW, 11, 1), BC6_TOK(BC6_BW, 10, 1),
    BC6_END,
};
BC_CONST bc_u kBc6Start[14] = {0, 21, 46, 66, 88, 110, 131, 152, 175, 198, 223, 230, 240, 253};
// Per mode: endpoint bits, delta bits (R, G, B), two subsets, transformed (deltas against endpoint W).
BC_CONST bc_u kBc6Mode[84] = {
    10, 5, 5, 5, 1, 1,  7, 6, 6, 6, 1, 1,  11, 5, 4, 4, 1, 1,  11, 4, 5, 4, 1, 1,  11, 4, 4, 5, 1, 1,  9, 5, 5, 5, 1, 1,  8, 6, 5, 5, 1, 1,
    8, 5, 6, 5, 1, 1,  8, 5, 5, 6, 1, 1,  6, 6, 6, 6, 1, 0,  10, 10, 10, 10, 0, 0,  11, 9, 9, 9, 0, 1,  12, 8, 8, 8, 0, 1,  16, 4, 4, 4, 0, 1,
};

// Header bits -> index into the mode tables (-1: reserved).
inline bc_i bc6_mode_of(bc_u bits5) {
    if ((bits5 & 3u) < 2u) return bc_i(bits5 & 3u);
    switch (bits5) {
        case 2u: return 2;
        case 6u: return 3;
        case 10u: return 4;
        case 14u: return 5;
        case 18u: return 6;
        case 22u: return 7;
        case 26u: return 8;
        case 30u: return 9;
        case 3u: return 10;
        case 7u: return 11;
        case 11u: return 12;
        case 15u: return 13;
        default: return -1;
    }
}

inline bc_i bc_sign_extend(bc_i v, bc_u bits) {
    return bits >= 32u ? v : bc_i(bc_u(v) << (32u - bits)) >> (32u - bits);
}

inline bc_i bc6_unquantize(bc_i v, bc_u bits, bc_u isSigned) {
    if (isSigned == 0u) {
        if (bits >= 15u) return v;
        if (v == 0) return 0;
        if (v == (1 << bits) - 1) return 0xFFFF;
        return ((v << 15) + 0x4000) >> (bits - 1u);
    }
    if (bits >= 16u) return v;
    const bool neg = v < 0;
    if (neg) v = -v;
    bc_i u;
    if (v == 0) u = 0;
    else if (v >= (1 << (bits - 1u)) - 1) u = 0x7FFF;
    else u = ((v << 15) + 0x4000) >> (bits - 1u);
    return neg ? -u : u;
}

inline bc_u bc6_finish_half(bc_i v, bc_u isSigned) {
    if (isSigned == 0u) return bc_u((v * 31) >> 6) & 0xFFFFu;
    bc_u s = 0u;
    if (v < 0) {
        s = 0x8000u;
        v = -v;
    }
    return (s | bc_u((v * 31) >> 5)) & 0xFFFFu;
}

inline void bc6h_block(BC_T const bc_u* block, bc_u isSigned, BC_T bc_u* out) {
    BcBits br;
    for (bc_u i = 0u; i < 4u; ++i) br.w[i] = block[i];
    br.pos = 0u;
    bc_u mbits = bc_get(br, 2u);
    if (mbits >= 2u) mbits |= bc_get(br, 3u) << 2;
    const bc_i mi = bc6_mode_of(mbits);
    if (mi < 0) {  // reserved: zeros, alpha one
        for (bc_u t = 0u; t < 16u; ++t) out[2u * t] = 0u, out[2u * t + 1u] = 0x3C000000u;
        return;
    }
    const bc_u mp = bc_u(mi) * 6u;
    const bc_u wbits = kBc6Mode[mp];
    const bool twoSubsets = kBc6Mode[mp + 4u] != 0u, transformed = kBc6Mode[mp + 5u] != 0u;
    bc_i f[13];
    for (bc_u i = 0u; i < 13u; ++i) f[i] = 0;
    for (bc_u k = kBc6Start[mi];; ++k) {
        const bc_u tok = kBc6Tok[k], n = tok >> 16;
        if (n == 0u) break;
        f[tok & 255u] |= bc_i(bc_get(br, n)) << ((tok >> 8) & 255u);
    }
    // f[ep * 3 + colour]: ep 0..3 = W, X, Y, Z
    bc_i ep[4][3];
    for (bc_u e = 0u; e < 4u; ++e)
        for (bc_u c = 0u; c < 3u; ++c) ep[e][c] = f[3u * e + c];
    const bc_u ends = twoSubsets ? 4u : 2u;
    const bc_i mask = wbits >= 31u ? -1 : (1 << wbits) - 1;
    for (bc_u c = 0u; c < 3u; ++c) {
        if (isSigned != 0u) ep[0][c] = bc_sign_extend(ep[0][c], wbits);
        for (bc_u e = 1u; e < ends; ++e) {
            if (transformed) {
                ep[e][c] = bc_sign_extend(ep[e][c], kBc6Mode[mp + 1u + c]);
                ep[e][c] = (ep[e][c] + ep[0][c]) & mask;
                if (isSigned != 0u) ep[e][c] = bc_sign_extend(ep[e][c], wbits);
            } else if (isSigned != 0u) {
                ep[e][c] = bc_sign_extend(ep[e][c], wbits);
            }
        }
    }
    for (bc_u e = 0u; e < ends; ++e)
        for (bc_u c = 0u; c < 3u; ++c) ep[e][c] = bc6_unquantize(ep[e][c], wbits, isSigned);
    const bc_u part = twoSubsets ? bc_u(f[BC6_DD]) : 0u;
    const bc_u ibits = twoSubsets ? 3u : 4u;
    const bc_u anchor1 = twoSubsets ? kBcAnchor2[part] : 16u;
    for (bc_u t = 0u; t < 16u; ++t) {
        const bc_u s = twoSubsets ? ((kBcPart2[part] >> t) & 1u) : 0u;
        const bc_u bits = ibits - ((t == 0u || (t == anchor1 && s == 1u)) ? 1u : 0u);
        const bc_u idx = bc_get(br, bits);
        const bc_u w = ibits == 3u ? kBcWeight3[idx] : kBcWeight4[idx];
        bc_u h[3];
        for (bc_u c = 0u; c < 3u; ++c) {
            const bc_i v = ((64 - bc_i(w)) * ep[2u * s][c] + bc_i(w) * ep[2u * s + 1u][c] + 32) >> 6;
            h[c] = bc6_finish_half(v, isSigned);
        }
        out[2u * t] = h[0] | (h[1] << 16);
        out[2u * t + 1u] = h[2] | 0x3C000000u;  // alpha = 1.0
    }
}

// ---- All formats ----------------------------------------------------------------------------------

// Bytes per decoded texel.
inline bc_u bc_texel_bytes(bc_u kind) { return kind == BC_BC4 ? 1u : kind == BC_BC5 ? 2u : kind == BC_BC6H ? 8u : 4u; }

inline void bc_decode_block(bc_u kind, bc_u isSigned, BC_T const bc_u* b, BC_T bc_u* out) {
    bc_u px[16], alpha[16];
    switch (kind) {
        case BC_BC1:
            bc1_block(b, 0u, 0u, px);
            for (bc_u t = 0u; t < 16u; ++t) out[2u * t] = px[t], out[2u * t + 1u] = 0u;
            break;
        case BC_BC2:
            bc1_block(b, 8u, 1u, px);
            for (bc_u t = 0u; t < 16u; ++t) {
                const bc_u a = ((bc_byte(b, t >> 1) >> (4u * (t & 1u))) & 15u) * 17u;
                out[2u * t] = (px[t] & 0x00FFFFFFu) | (a << 24), out[2u * t + 1u] = 0u;
            }
            break;
        case BC_BC3:
            bc1_block(b, 8u, 1u, px);
            bc4_block(b, 0u, 0u, alpha);
            for (bc_u t = 0u; t < 16u; ++t) out[2u * t] = (px[t] & 0x00FFFFFFu) | (alpha[t] << 24), out[2u * t + 1u] = 0u;
            break;
        case BC_BC4:
            bc4_block(b, 0u, isSigned, px);
            for (bc_u t = 0u; t < 16u; ++t) out[2u * t] = px[t], out[2u * t + 1u] = 0u;
            break;
        case BC_BC5:
            bc4_block(b, 0u, isSigned, px);
            bc4_block(b, 8u, isSigned, alpha);
            for (bc_u t = 0u; t < 16u; ++t) out[2u * t] = px[t] | (alpha[t] << 8), out[2u * t + 1u] = 0u;
            break;
        case BC_BC6H: bc6h_block(b, isSigned, out); break;
        default:
            bc7_block(b, px);
            for (bc_u t = 0u; t < 16u; ++t) out[2u * t] = px[t], out[2u * t + 1u] = 0u;
            break;
    }
}

// ---- The kernel: one thread per block ---------------------------------------------------------------

// What a dispatch decodes: the blocks of one mip level (all `t.slices` slices from `t.firstSlice`) of a surface in guest
// memory, addressed as tiling_addr.h does (`t` is the same parameter block as the detile kernel's, in blocks), into a
// linear buffer of texels: `rowBytes` per row of the level's `texW` x `texH` texels, `sliceBytes` per slice.
struct VpBcParams {
    VptParams t;
    bc_u kind, isSigned, texW, texH;
    bc_u texBytes, rowBytes, sliceBytes, pad;
};

// The thread (gx, gy, gz) = block (gx, gy) of slice gz. Blocks on the right and bottom edge write only the texels inside.
inline void bc_kernel_thread(const BC_DEV bc_u8* tiled, BC_DEV bc_u8* linear, VPT_CONST VpBcParams& P, bc_u gx, bc_u gy, bc_u gz) {
    if (!vpt_in_range(P.t, gx, gy, gz)) return;
    const BC_DEV bc_u8* from = tiled + vpt_tiled_offset(P.t, gx, gy, gz);
    bc_u blk[4];
    for (bc_u i = 0u; i < 4u; ++i) {
        blk[i] = 0u;
        for (bc_u j = 0u; j < 4u; ++j)
            if (4u * i + j < P.t.level.bpe) blk[i] |= bc_u(from[4u * i + j]) << (8u * j);
    }
    bc_u o[32];
    bc_decode_block(P.kind, P.isSigned, blk, o);
    for (bc_u y = 0u; y < 4u; ++y) {
        const bc_u py = gy * 4u + y;
        if (py >= P.texH) break;
        for (bc_u x = 0u; x < 4u; ++x) {
            const bc_u px = gx * 4u + x;
            if (px >= P.texW) break;
            BC_DEV bc_u8* to = linear + gz * P.sliceBytes + py * P.rowBytes + px * P.texBytes;
            const bc_u t = 4u * y + x;
            for (bc_u k = 0u; k < P.texBytes; ++k) to[k] = bc_u8((o[2u * t + (k >> 2)] >> (8u * (k & 3u))) & 255u);
        }
    }
}
