// CPU decoder of BC1-BC7 (see bc_decode.h). Written from the public format specifications.
#include "vpps4/bc_decode.h"

#include <algorithm>
#include <cstring>

namespace vpps4::bc {

namespace {

using vpgpu::Format;

// ---- Bit reading ---------------------------------------------------------------------------------

struct Bits {
    uint64_t lo = 0, hi = 0;
    unsigned pos = 0;
    explicit Bits(const uint8_t* b) {
        for (unsigned i = 0; i < 8; ++i) lo |= uint64_t(b[i]) << (8 * i), hi |= uint64_t(b[8 + i]) << (8 * i);
    }
    uint32_t get(unsigned n) {
        if (!n) return 0;
        uint64_t v;
        if (pos >= 64) {
            v = hi >> (pos - 64);
        } else {
            v = lo >> pos;
            if (pos + n > 64) v |= hi << (64 - pos);
        }
        pos += n;
        return uint32_t(v & ((1ull << n) - 1));
    }
};

// ---- BC1 / BC2 / BC3 / BC4 / BC5 -------------------------------------------------------------------

uint8_t expand5(unsigned v) { return uint8_t(v << 3 | v >> 2); }
uint8_t expand6(unsigned v) { return uint8_t(v << 2 | v >> 4); }

// The four colours of a BC1 colour block. `opaque`: always the four-colour mode (the colour half of BC2 / BC3).
void bc1Colors(const uint8_t* b, bool opaque, uint8_t col[4][4]) {
    const unsigned c0 = b[0] | b[1] << 8, c1 = b[2] | b[3] << 8;
    const unsigned rgb[2][3] = {{c0 >> 11 & 31, c0 >> 5 & 63, c0 & 31}, {c1 >> 11 & 31, c1 >> 5 & 63, c1 & 31}};
    for (unsigned e = 0; e < 2; ++e) {
        col[e][0] = expand5(rgb[e][0]);
        col[e][1] = expand6(rgb[e][1]);
        col[e][2] = expand5(rgb[e][2]);
        col[e][3] = 255;
    }
    for (unsigned c = 0; c < 3; ++c) {
        if (c0 > c1 || opaque) {
            col[2][c] = uint8_t((2 * col[0][c] + col[1][c]) / 3);
            col[3][c] = uint8_t((col[0][c] + 2 * col[1][c]) / 3);
        } else {
            col[2][c] = uint8_t((col[0][c] + col[1][c]) / 2);
            col[3][c] = 0;
        }
    }
    col[2][3] = 255;
    col[3][3] = c0 > c1 || opaque ? 255 : 0;
}

void bc1(const uint8_t* b, bool opaque, uint8_t* out) {
    uint8_t col[4][4];
    bc1Colors(b, opaque, col);
    uint32_t idx = b[4] | b[5] << 8 | b[6] << 16 | uint32_t(b[7]) << 24;
    for (unsigned t = 0; t < 16; ++t, idx >>= 2) std::memcpy(out + 4 * t, col[idx & 3], 4);
}

// The 3-bit indices of a BC4 block (16 x 3 bits after two endpoint bytes).
uint64_t bc4Indices(const uint8_t* b) {
    uint64_t v = 0;
    for (unsigned i = 0; i < 6; ++i) v |= uint64_t(b[2 + i]) << (8 * i);
    return v;
}

void bc4Unsigned(const uint8_t* b, uint8_t* out, size_t stride) {
    unsigned c[8] = {b[0], b[1]};
    if (c[0] > c[1]) {
        for (unsigned i = 1; i < 7; ++i) c[1 + i] = ((7 - i) * c[0] + i * c[1]) / 7;
    } else {
        for (unsigned i = 1; i < 5; ++i) c[1 + i] = ((5 - i) * c[0] + i * c[1]) / 5;
        c[6] = 0;
        c[7] = 255;
    }
    const uint64_t idx = bc4Indices(b);
    for (unsigned t = 0; t < 16; ++t) out[stride * t] = uint8_t(c[idx >> (3 * t) & 7]);
}

void bc4Signed(const uint8_t* b, uint8_t* out, size_t stride) {
    int c[8] = {std::max(int(int8_t(b[0])), -127), std::max(int(int8_t(b[1])), -127)};
    if (c[0] > c[1]) {
        for (int i = 1; i < 7; ++i) c[1 + i] = ((7 - i) * c[0] + i * c[1]) / 7;
    } else {
        for (int i = 1; i < 5; ++i) c[1 + i] = ((5 - i) * c[0] + i * c[1]) / 5;
        c[6] = -127;
        c[7] = 127;
    }
    const uint64_t idx = bc4Indices(b);
    for (unsigned t = 0; t < 16; ++t) out[stride * t] = uint8_t(int8_t(c[idx >> (3 * t) & 7]));
}

// ---- BC7 ------------------------------------------------------------------------------------------

// Partitions: one bit per texel (2 subsets) or two (3 subsets), texel 0 in the low bits. Anchors: the texel of
// subset 1 (2 subsets) and subsets 1 and 2 (3 subsets) whose index drops its top bit; subset 0 always has texel 0.
const uint16_t kPart2[64] = {
    0xcccc, 0x8888, 0xeeee, 0xecc8, 0xc880, 0xfeec, 0xfec8, 0xec80, 0xc800, 0xffec, 0xfe80, 0xe800, 0xffe8, 0xff00, 0xfff0, 0xf000,
    0xf710, 0x008e, 0x7100, 0x08ce, 0x008c, 0x7310, 0x3100, 0x8cce, 0x088c, 0x3110, 0x6666, 0x366c, 0x17e8, 0x0ff0, 0x718e, 0x399c,
    0xaaaa, 0xf0f0, 0x5a5a, 0x33cc, 0x3c3c, 0x55aa, 0x9696, 0xa55a, 0x73ce, 0x13c8, 0x324c, 0x3bdc, 0x6996, 0xc33c, 0x9966, 0x0660,
    0x0272, 0x04e4, 0x4e40, 0x2720, 0xc936, 0x936c, 0x39c6, 0x639c, 0x9336, 0x9cc6, 0x817e, 0xe718, 0xccf0, 0x0fcc, 0x7744, 0xee22,
};
const uint32_t kPart3[64] = {
    0xaa685050, 0x6a5a5040, 0x5a5a4200, 0x5450a0a8, 0xa5a50000, 0xa0a05050, 0x5555a0a0, 0x5a5a5050, 0xaa550000, 0xaa555500,
    0xaaaa5500, 0x90909090, 0x94949494, 0xa4a4a4a4, 0xa9a59450, 0x2a0a4250, 0xa5945040, 0x0a425054, 0xa5a5a500, 0x55a0a0a0,
    0xa8a85454, 0x6a6a4040, 0xa4a45000, 0x1a1a0500, 0x0050a4a4, 0xaaa59090, 0x14696914, 0x69691400, 0xa08585a0, 0xaa821414,
    0x50a4a450, 0x6a5a0200, 0xa9a58000, 0x5090a0a8, 0xa8a09050, 0x24242424, 0x00aa5500, 0x24924924, 0x24499224, 0x50a50a50,
    0x500aa550, 0xaaaa4444, 0x66660000, 0xa5a0a5a0, 0x50a050a0, 0x69286928, 0x44aaaa44, 0x66666600, 0xaa444444, 0x54a854a8,
    0x95809580, 0x96969600, 0xa85454a8, 0x80959580, 0xaa141414, 0x96960000, 0xaaaa1414, 0xa05050a0, 0xa0a5a5a0, 0x96000000,
    0x40804080, 0xa9a8a9a8, 0xaaaaaa44, 0x2a4a5254,
};
const uint8_t kAnchor2[64] = {
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2,  8,  2,  2,  8,  8,  15, 2,  8,  2,  2,  8,  8,  2,  2,
    15, 15, 6,  8,  2,  8,  15, 15, 2,  8,  2,  2,  2,  15, 15, 6,  6,  2,  6,  8,  15, 15, 2,  2,  15, 15, 15, 15, 15, 2,  2,  15,
};
const uint8_t kAnchor3a[64] = {
    3,  3,  15, 15, 8,  3,  15, 15, 8,  8,  6,  6,  6,  5,  3,  3,  3,  3,  8,  15, 3,  3,  6,  10, 5,  8,  8,  6,  8,  5,  15, 15,
    8,  15, 3,  5,  6,  10, 8,  15, 15, 3,  15, 5,  15, 15, 15, 15, 3,  15, 5,  5,  5,  8,  5,  10, 5,  10, 8,  13, 15, 12, 3,  3,
};
const uint8_t kAnchor3b[64] = {
    15, 8,  8,  3,  15, 15, 3,  8,  15, 15, 15, 15, 15, 15, 15, 8,  15, 8,  15, 3,  15, 8,  15, 8,  3,  15, 6,  10, 15, 15, 10, 8,
    15, 3,  15, 10, 10, 8,  9,  10, 6,  15, 8,  15, 3,  6,  6,  8,  15, 3,  15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3,  15, 15, 8,
};

const uint8_t kWeight2[4] = {0, 21, 43, 64};
const uint8_t kWeight3[8] = {0, 9, 18, 27, 37, 46, 55, 64};
const uint8_t kWeight4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};
unsigned weightOf(unsigned bits, unsigned index) { return bits == 2 ? kWeight2[index] : bits == 3 ? kWeight3[index] : kWeight4[index]; }

struct Bc7Mode {
    uint8_t subsets, partitionBits, rotationBits, indexSelBit, colorBits, alphaBits, endpointPBit, sharedPBit, indexBits, indexBits2;
};
const Bc7Mode kBc7[8] = {
    {3, 4, 0, 0, 4, 0, 1, 0, 3, 0}, {2, 6, 0, 0, 6, 0, 0, 1, 3, 0}, {3, 6, 0, 0, 5, 0, 0, 0, 2, 0}, {2, 6, 0, 0, 7, 0, 1, 0, 2, 0},
    {1, 0, 2, 1, 5, 6, 0, 0, 2, 3}, {1, 0, 2, 0, 7, 8, 0, 0, 2, 2}, {1, 0, 0, 0, 7, 7, 1, 0, 4, 0}, {2, 6, 0, 0, 5, 5, 1, 0, 2, 0},
};

unsigned expandBits(unsigned v, unsigned bits) { return v << (8 - bits) | v >> (2 * bits - 8); }

void bc7(const uint8_t* block, uint8_t* out) {
    Bits br(block);
    unsigned mode = 0;
    while (mode < 8 && !br.get(1)) ++mode;
    if (mode == 8) {  // reserved: all zero
        std::memset(out, 0, 64);
        return;
    }
    const Bc7Mode& m = kBc7[mode];
    const unsigned part = br.get(m.partitionBits), rotation = br.get(m.rotationBits), indexSel = br.get(m.indexSelBit);
    const unsigned ends = 2u * m.subsets;
    unsigned ep[6][4] = {};
    for (unsigned c = 0; c < 3; ++c)
        for (unsigned e = 0; e < ends; ++e) ep[e][c] = br.get(m.colorBits);
    for (unsigned e = 0; e < ends && m.alphaBits; ++e) ep[e][3] = br.get(m.alphaBits);
    unsigned pbit[6] = {};
    const bool hasP = m.endpointPBit || m.sharedPBit;
    if (m.endpointPBit)
        for (unsigned e = 0; e < ends; ++e) pbit[e] = br.get(1);
    if (m.sharedPBit)
        for (unsigned s = 0; s < m.subsets; ++s) pbit[2 * s] = pbit[2 * s + 1] = br.get(1);
    for (unsigned e = 0; e < ends; ++e) {
        for (unsigned c = 0; c < 3; ++c) {
            const unsigned v = hasP ? ep[e][c] << 1 | pbit[e] : ep[e][c];
            ep[e][c] = expandBits(v, m.colorBits + hasP);
        }
        if (m.alphaBits) {
            const unsigned v = hasP ? ep[e][3] << 1 | pbit[e] : ep[e][3];
            ep[e][3] = expandBits(v, m.alphaBits + hasP);
        } else {
            ep[e][3] = 255;
        }
    }
    auto subsetOf = [&](unsigned t) -> unsigned {
        return m.subsets == 1 ? 0 : m.subsets == 2 ? (kPart2[part] >> t & 1) : (kPart3[part] >> (2 * t) & 3);
    };
    auto isAnchor = [&](unsigned t, unsigned s) {
        return t == 0 || (s == 1 && t == (m.subsets == 2 ? kAnchor2[part] : kAnchor3a[part])) || (s == 2 && t == kAnchor3b[part]);
    };
    uint8_t idx[16], idx2[16] = {};
    for (unsigned t = 0; t < 16; ++t) idx[t] = uint8_t(br.get(m.indexBits - (isAnchor(t, subsetOf(t)) ? 1 : 0)));
    for (unsigned t = 0; t < 16 && m.indexBits2; ++t) idx2[t] = uint8_t(br.get(m.indexBits2 - (t == 0 ? 1 : 0)));
    for (unsigned t = 0; t < 16; ++t) {
        const unsigned s = subsetOf(t);
        // Modes 4 and 5 hold two index sets: one for colour, one for alpha (swapped by the index selection bit in mode 4).
        const unsigned colorBits = m.indexBits2 && indexSel ? m.indexBits2 : m.indexBits;
        const unsigned alphaBits = m.indexBits2 ? (indexSel ? m.indexBits : m.indexBits2) : m.indexBits;
        const unsigned ci = m.indexBits2 && indexSel ? idx2[t] : idx[t];
        const unsigned ai = m.indexBits2 ? (indexSel ? idx[t] : idx2[t]) : idx[t];
        const unsigned wc = weightOf(colorBits, ci), wa = weightOf(alphaBits, ai);
        unsigned px[4];
        for (unsigned c = 0; c < 4; ++c) {
            const unsigned w = c < 3 ? wc : wa;
            px[c] = ((64 - w) * ep[2 * s][c] + w * ep[2 * s + 1][c] + 32) >> 6;
        }
        if (rotation) std::swap(px[3], px[rotation - 1]);
        for (unsigned c = 0; c < 4; ++c) out[4 * t + c] = uint8_t(px[c]);
    }
}

// ---- BC6H -----------------------------------------------------------------------------------------

// The fields of the header in stream order: endpoint W (first of subset 0), X (second of subset 0), Y and Z (the
// subset 1 pair) per colour, and the partition D.
enum : uint8_t { RW, GW, BW, RX, GX, BX, RY, GY, BY, RZ, GZ, BZ, DD };
struct Tok {
    uint8_t field, lo, n;
};
struct Bc6Mode {
    uint8_t wbits, dbits[3];
    bool twoSubsets, transformed;
    const Tok* tokens;
};
#define T(f, lo, n) {f, lo, n}
const Tok kT1[] = {T(GY, 4, 1), T(BY, 4, 1), T(BZ, 4, 1), T(RW, 0, 10), T(GW, 0, 10), T(BW, 0, 10), T(RX, 0, 5), T(GZ, 4, 1), T(GY, 0, 4),
                   T(GX, 0, 5), T(BZ, 0, 1), T(GZ, 0, 4), T(BX, 0, 5), T(BZ, 1, 1), T(BY, 0, 4), T(RY, 0, 5), T(BZ, 2, 1), T(RZ, 0, 5),
                   T(BZ, 3, 1), T(DD, 0, 5), T(0, 0, 0)};
const Tok kT2[] = {T(GY, 5, 1), T(GZ, 4, 1), T(GZ, 5, 1), T(RW, 0, 7), T(BZ, 0, 1), T(BZ, 1, 1), T(BY, 4, 1), T(GW, 0, 7), T(BY, 5, 1),
                   T(BZ, 2, 1), T(GY, 4, 1), T(BW, 0, 7), T(BZ, 3, 1), T(BZ, 5, 1), T(BZ, 4, 1), T(RX, 0, 6), T(GY, 0, 4), T(GX, 0, 6),
                   T(GZ, 0, 4), T(BX, 0, 6), T(BY, 0, 4), T(RY, 0, 6), T(RZ, 0, 6), T(DD, 0, 5), T(0, 0, 0)};
const Tok kT3[] = {T(RW, 0, 10), T(GW, 0, 10), T(BW, 0, 10), T(RX, 0, 5), T(RW, 10, 1), T(GY, 0, 4), T(GX, 0, 4), T(GW, 10, 1), T(BZ, 0, 1),
                   T(GZ, 0, 4), T(BX, 0, 4), T(BW, 10, 1), T(BZ, 1, 1), T(BY, 0, 4), T(RY, 0, 5), T(BZ, 2, 1), T(RZ, 0, 5), T(BZ, 3, 1),
                   T(DD, 0, 5), T(0, 0, 0)};
const Tok kT4[] = {T(RW, 0, 10), T(GW, 0, 10), T(BW, 0, 10), T(RX, 0, 4), T(RW, 10, 1), T(GZ, 4, 1), T(GY, 0, 4), T(GX, 0, 5), T(GW, 10, 1),
                   T(GZ, 0, 4), T(BX, 0, 4), T(BW, 10, 1), T(BZ, 1, 1), T(BY, 0, 4), T(RY, 0, 4), T(BZ, 0, 1), T(BZ, 2, 1), T(RZ, 0, 4),
                   T(GY, 4, 1), T(BZ, 3, 1), T(DD, 0, 5), T(0, 0, 0)};
const Tok kT5[] = {T(RW, 0, 10), T(GW, 0, 10), T(BW, 0, 10), T(RX, 0, 4), T(RW, 10, 1), T(BY, 4, 1), T(GY, 0, 4), T(GX, 0, 4), T(GW, 10, 1),
                   T(BZ, 0, 1), T(GZ, 0, 4), T(BX, 0, 5), T(BW, 10, 1), T(BY, 0, 4), T(RY, 0, 4), T(BZ, 1, 1), T(BZ, 2, 1), T(RZ, 0, 4),
                   T(BZ, 4, 1), T(BZ, 3, 1), T(DD, 0, 5), T(0, 0, 0)};
const Tok kT6[] = {T(RW, 0, 9), T(BY, 4, 1), T(GW, 0, 9), T(GY, 4, 1), T(BW, 0, 9), T(BZ, 4, 1), T(RX, 0, 5), T(GZ, 4, 1), T(GY, 0, 4),
                   T(GX, 0, 5), T(BZ, 0, 1), T(GZ, 0, 4), T(BX, 0, 5), T(BZ, 1, 1), T(BY, 0, 4), T(RY, 0, 5), T(BZ, 2, 1), T(RZ, 0, 5),
                   T(BZ, 3, 1), T(DD, 0, 5), T(0, 0, 0)};
const Tok kT7[] = {T(RW, 0, 8), T(GZ, 4, 1), T(BY, 4, 1), T(GW, 0, 8), T(BZ, 2, 1), T(GY, 4, 1), T(BW, 0, 8), T(BZ, 3, 1), T(BZ, 4, 1),
                   T(RX, 0, 6), T(GY, 0, 4), T(GX, 0, 5), T(BZ, 0, 1), T(GZ, 0, 4), T(BX, 0, 5), T(BZ, 1, 1), T(BY, 0, 4), T(RY, 0, 6),
                   T(RZ, 0, 6), T(DD, 0, 5), T(0, 0, 0)};
const Tok kT8[] = {T(RW, 0, 8), T(BZ, 0, 1), T(BY, 4, 1), T(GW, 0, 8), T(GY, 5, 1), T(GY, 4, 1), T(BW, 0, 8), T(GZ, 5, 1), T(BZ, 4, 1),
                   T(RX, 0, 5), T(GZ, 4, 1), T(GY, 0, 4), T(GX, 0, 6), T(GZ, 0, 4), T(BX, 0, 5), T(BZ, 1, 1), T(BY, 0, 4), T(RY, 0, 5),
                   T(BZ, 2, 1), T(RZ, 0, 5), T(BZ, 3, 1), T(DD, 0, 5), T(0, 0, 0)};
const Tok kT9[] = {T(RW, 0, 8), T(BZ, 1, 1), T(BY, 4, 1), T(GW, 0, 8), T(BY, 5, 1), T(GY, 4, 1), T(BW, 0, 8), T(BZ, 5, 1), T(BZ, 4, 1),
                   T(RX, 0, 5), T(GZ, 4, 1), T(GY, 0, 4), T(GX, 0, 5), T(BZ, 0, 1), T(GZ, 0, 4), T(BX, 0, 6), T(BY, 0, 4), T(RY, 0, 5),
                   T(BZ, 2, 1), T(RZ, 0, 5), T(BZ, 3, 1), T(DD, 0, 5), T(0, 0, 0)};
const Tok kT10[] = {T(RW, 0, 6), T(GZ, 4, 1), T(BZ, 0, 1), T(BZ, 1, 1), T(BY, 4, 1), T(GW, 0, 6), T(GY, 5, 1), T(BY, 5, 1), T(BZ, 2, 1),
                    T(GY, 4, 1), T(BW, 0, 6), T(GZ, 5, 1), T(BZ, 3, 1), T(BZ, 5, 1), T(BZ, 4, 1), T(RX, 0, 6), T(GY, 0, 4), T(GX, 0, 6),
                    T(GZ, 0, 4), T(BX, 0, 6), T(BY, 0, 4), T(RY, 0, 6), T(RZ, 0, 6), T(DD, 0, 5), T(0, 0, 0)};
const Tok kT11[] = {T(RW, 0, 10), T(GW, 0, 10), T(BW, 0, 10), T(RX, 0, 10), T(GX, 0, 10), T(BX, 0, 10), T(0, 0, 0)};
const Tok kT12[] = {T(RW, 0, 10), T(GW, 0, 10), T(BW, 0, 10), T(RX, 0, 9), T(RW, 10, 1), T(GX, 0, 9), T(GW, 10, 1), T(BX, 0, 9), T(BW, 10, 1),
                    T(0, 0, 0)};
const Tok kT13[] = {T(RW, 0, 10), T(GW, 0, 10), T(BW, 0, 10), T(RX, 0, 8), T(RW, 11, 1), T(RW, 10, 1), T(GX, 0, 8), T(GW, 11, 1), T(GW, 10, 1),
                    T(BX, 0, 8), T(BW, 11, 1), T(BW, 10, 1), T(0, 0, 0)};
const Tok kT14[] = {T(RW, 0, 10), T(GW, 0, 10), T(BW, 0, 10), T(RX, 0, 4), T(RW, 15, 1), T(RW, 14, 1), T(RW, 13, 1), T(RW, 12, 1), T(RW, 11, 1),
                    T(RW, 10, 1), T(GX, 0, 4), T(GW, 15, 1), T(GW, 14, 1), T(GW, 13, 1), T(GW, 12, 1), T(GW, 11, 1), T(GW, 10, 1), T(BX, 0, 4),
                    T(BW, 15, 1), T(BW, 14, 1), T(BW, 13, 1), T(BW, 12, 1), T(BW, 11, 1), T(BW, 10, 1), T(0, 0, 0)};
#undef T

const Bc6Mode kBc6[14] = {
    {10, {5, 5, 5}, true, true, kT1},    {7, {6, 6, 6}, true, true, kT2},     {11, {5, 4, 4}, true, true, kT3},
    {11, {4, 5, 4}, true, true, kT4},    {11, {4, 4, 5}, true, true, kT5},    {9, {5, 5, 5}, true, true, kT6},
    {8, {6, 5, 5}, true, true, kT7},     {8, {5, 6, 5}, true, true, kT8},     {8, {5, 5, 6}, true, true, kT9},
    {6, {6, 6, 6}, true, false, kT10},   {10, {10, 10, 10}, false, false, kT11}, {11, {9, 9, 9}, false, true, kT12},
    {12, {8, 8, 8}, false, true, kT13},  {16, {4, 4, 4}, false, true, kT14},
};

// Header bits -> index into kBc6 (-1: reserved).
int bc6ModeOf(unsigned bits5) {
    if ((bits5 & 3) < 2) return int(bits5 & 3);
    switch (bits5) {
        case 2: return 2;
        case 6: return 3;
        case 10: return 4;
        case 14: return 5;
        case 18: return 6;
        case 22: return 7;
        case 26: return 8;
        case 30: return 9;
        case 3: return 10;
        case 7: return 11;
        case 11: return 12;
        case 15: return 13;
        default: return -1;
    }
}

int signExtend(int v, unsigned bits) { return bits >= 32 ? v : int(uint32_t(v) << (32 - bits)) >> (32 - bits); }

int unquantize(int v, unsigned bits, bool isSigned) {
    if (!isSigned) {
        if (bits >= 15) return v;
        if (v == 0) return 0;
        if (v == (1 << bits) - 1) return 0xFFFF;
        return ((v << 15) + 0x4000) >> (bits - 1);
    }
    if (bits >= 16) return v;
    const bool neg = v < 0;
    if (neg) v = -v;
    int u;
    if (v == 0) u = 0;
    else if (v >= (1 << (bits - 1)) - 1) u = 0x7FFF;
    else u = ((v << 15) + 0x4000) >> (bits - 1);
    return neg ? -u : u;
}

uint16_t finishHalf(int v, bool isSigned) {
    if (!isSigned) return uint16_t((v * 31) >> 6);
    unsigned s = 0;
    if (v < 0) s = 0x8000, v = -v;
    return uint16_t(s | ((v * 31) >> 5));
}

void bc6h(const uint8_t* block, bool isSigned, uint8_t* out) {
    Bits br(block);
    unsigned mbits = br.get(2);
    if (mbits >= 2) mbits |= br.get(3) << 2;
    const int mi = bc6ModeOf(mbits);
    if (mi < 0) {  // reserved: zeros, alpha one
        for (unsigned t = 0; t < 16; ++t) {
            std::memset(out + 8 * t, 0, 6);
            out[8 * t + 6] = 0x00, out[8 * t + 7] = 0x3C;
        }
        return;
    }
    const Bc6Mode& m = kBc6[mi];
    int f[13] = {};
    for (const Tok* t = m.tokens; t->n; ++t) f[t->field] |= int(br.get(t->n)) << t->lo;
    // f[ep * 3 + colour]: ep 0..3 = W, X, Y, Z
    int ep[4][3];
    for (unsigned e = 0; e < 4; ++e)
        for (unsigned c = 0; c < 3; ++c) ep[e][c] = f[3 * e + c];
    const unsigned ends = m.twoSubsets ? 4 : 2;
    for (unsigned c = 0; c < 3; ++c) {
        if (isSigned) ep[0][c] = signExtend(ep[0][c], m.wbits);
        const int mask = m.wbits >= 31 ? -1 : (1 << m.wbits) - 1;
        for (unsigned e = 1; e < ends; ++e) {
            if (m.transformed) {
                ep[e][c] = signExtend(ep[e][c], m.dbits[c]);
                ep[e][c] = (ep[e][c] + ep[0][c]) & mask;
                if (isSigned) ep[e][c] = signExtend(ep[e][c], m.wbits);
            } else if (isSigned) {
                ep[e][c] = signExtend(ep[e][c], m.wbits);
            }
        }
    }
    for (unsigned e = 0; e < ends; ++e)
        for (unsigned c = 0; c < 3; ++c) ep[e][c] = unquantize(ep[e][c], m.wbits, isSigned);
    const unsigned part = m.twoSubsets ? unsigned(f[DD]) : 0;
    const unsigned ibits = m.twoSubsets ? 3 : 4;
    const unsigned anchor1 = m.twoSubsets ? kAnchor2[part] : 16;
    for (unsigned t = 0; t < 16; ++t) {
        const unsigned s = m.twoSubsets ? (kPart2[part] >> t & 1) : 0;
        const unsigned bits = ibits - (t == 0 || (t == anchor1 && s == 1) ? 1 : 0);
        const unsigned idx = br.get(bits);
        const unsigned w = ibits == 3 ? kWeight3[idx] : kWeight4[idx];
        for (unsigned c = 0; c < 3; ++c) {
            const int v = ((64 - int(w)) * ep[2 * s][c] + int(w) * ep[2 * s + 1][c] + 32) >> 6;
            const uint16_t h = finishHalf(v, isSigned);
            out[8 * t + 2 * c] = uint8_t(h), out[8 * t + 2 * c + 1] = uint8_t(h >> 8);
        }
        out[8 * t + 6] = 0x00, out[8 * t + 7] = 0x3C;  // alpha = 1.0
    }
}

}  // namespace

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
    switch (k) {
        case Kind::BC1: bc1(b, false, out); break;
        case Kind::BC2:
            bc1(b + 8, true, out);
            for (unsigned t = 0; t < 16; ++t) out[4 * t + 3] = uint8_t((b[t / 2] >> (4 * (t & 1)) & 15) * 17);
            break;
        case Kind::BC3:
            bc1(b + 8, true, out);
            bc4Unsigned(b, out + 3, 4);
            break;
        case Kind::BC4: isSigned ? bc4Signed(b, out, 1) : bc4Unsigned(b, out, 1); break;
        case Kind::BC5:
            if (isSigned) bc4Signed(b, out, 2), bc4Signed(b + 8, out + 1, 2);
            else bc4Unsigned(b, out, 2), bc4Unsigned(b + 8, out + 1, 2);
            break;
        case Kind::BC6H: bc6h(b, isSigned, out); break;
        case Kind::BC7: bc7(b, out); break;
    }
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
