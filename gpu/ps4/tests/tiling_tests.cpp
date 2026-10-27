// Tiling: every element of a tiled surface has its own place inside the surface's span, detile()
// undoes tile(), and the production layout and addressing agree bit for bit with the literal
// reference in tiling_ref.h for every PS4 tile mode, element size, mip level, slice and sample.
// Known layouts and addresses from an independent implementation are checked as golden values.
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "tiling_ref.h"
#include "vpps4/tiling.h"

using namespace vpps4;

static int failures = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c);      \
            ++failures;                                                  \
        }                                                                \
    } while (0)

static ref::Surface toRef(const TiledSurface& s) {
    ref::Surface r;
    r.tile = s.tileIndex;
    r.bpe = int(s.bytesPerElement);
    r.samples = int(s.samples);
    r.width = s.width, r.height = s.height, r.pitch = s.pitch, r.slices = s.slices, r.levels = s.levels;
    r.block = s.blockSize, r.volume = s.volume, r.pow2 = s.pow2Pad, r.bankSwizzle = s.bankSwizzle;
    return r;
}

static std::string describe(const TiledSurface& s) {
    return "mode " + std::to_string(s.tileIndex) + " (" + tileModeInfo(s.tileIndex).name + ") bpe " + std::to_string(s.bytesPerElement) +
           " " + std::to_string(s.width) + "x" + std::to_string(s.height) + " pitch " + std::to_string(s.pitch) + " slices " +
           std::to_string(s.slices) + " levels " + std::to_string(s.levels) + " samples " + std::to_string(s.samples) +
           (s.volume ? " volume" : "") + (s.pow2Pad ? " pow2" : "") + (s.blockSize > 1 ? " block" : "");
}

// Layout, addresses, bijection and round trip of one surface. `step`: compare every step-th
// element with the (slow) reference; the bijection and the round trip always cover all of them.
static void checkSurface(const TiledSurface& s, uint32_t step) {
    const SurfaceLayout lay = computeLayout(s);
    if (!lay.ok) {
        std::printf("  not covered: %s: %s\n", describe(s).c_str(), lay.why.c_str());
        ++failures;
        return;
    }
    const ref::Surface rs = toRef(s);
    const std::vector<ref::Level> rl = ref::levels(rs);
    const int before = failures;
    if (rl.size() != lay.levels.size()) {
        CHECK(false);
        return;
    }
    for (size_t i = 0; i < rl.size(); ++i) {
        const LevelLayout& l = lay.levels[i];
        const bool oneD = std::string(rl[i].array).find("1D") != std::string::npos;
        const bool linear = std::string(rl[i].array).find("LINEAR") != std::string::npos;
        if (l.pitch != rl[i].pitch || l.paddedHeight != rl[i].rows || l.offset != rl[i].offset || l.bytes != rl[i].bytes ||
            l.width != rl[i].w || l.height != rl[i].h || l.slices != rl[i].d || (isLinear(l.array) != linear) ||
            ((l.array == ArrayMode::Tiled1DThin || l.array == ArrayMode::Tiled1DThick) != oneD)) {
            std::printf("  level %zu layout differs: %s\n    pitch %u/%llu rows %u/%llu offset %llu/%llu bytes %llu/%llu\n", i,
                        describe(s).c_str(), l.pitch, (unsigned long long)rl[i].pitch, l.paddedHeight, (unsigned long long)rl[i].rows,
                        (unsigned long long)l.offset, (unsigned long long)rl[i].offset, (unsigned long long)l.bytes,
                        (unsigned long long)rl[i].bytes);
            ++failures;
            return;
        }
    }
    CHECK(lay.bytes == rl.back().offset + rl.back().bytes);

    std::vector<uint8_t> used(lay.bytes / s.bytesPerElement, 0);
    uint64_t counter = 0;
    for (uint32_t li = 0; li < lay.levels.size(); ++li) {
        const LevelLayout& l = lay.levels[li];
        for (uint32_t sm = 0; sm < s.samples; ++sm)
            for (uint32_t z = 0; z < l.slices; ++z)
                for (uint32_t y = 0; y < l.height; ++y)
                    for (uint32_t x = 0; x < l.width; ++x) {
                        const uint64_t o = lay.offsetOf(li, x, y, z, sm);
                        if (o == UINT64_MAX || o + s.bytesPerElement > lay.bytes || o % s.bytesPerElement || o < l.offset ||
                            o + s.bytesPerElement > l.offset + l.bytes) {
                            std::printf("  %s: (%u,%u,%u,s%u) level %u at %llu outside [%llu,%llu)\n", describe(s).c_str(), x, y, z, sm, li,
                                        (unsigned long long)o, (unsigned long long)l.offset, (unsigned long long)(l.offset + l.bytes));
                            ++failures;
                            return;
                        }
                        if (used[o / s.bytesPerElement]++) {
                            std::printf("  %s: (%u,%u,%u,s%u) level %u shares offset %llu\n", describe(s).c_str(), x, y, z, sm, li,
                                        (unsigned long long)o);
                            ++failures;
                            return;
                        }
                        if (counter++ % step == 0) {
                            const uint64_t want = ref::address(rs, rl, li, x, y, z, sm);
                            if (o != want) {
                                std::printf("  %s: (%u,%u,%u,s%u) level %u: %llu, reference %llu\n", describe(s).c_str(), x, y, z, sm, li,
                                            (unsigned long long)o, (unsigned long long)want);
                                ++failures;
                                return;
                            }
                        }
                    }
    }

    // Round trip, a different payload per sample plane.
    std::mt19937 rng(s.tileIndex * 131u + s.bytesPerElement);
    std::vector<std::vector<uint8_t>> planes(s.samples, std::vector<uint8_t>(lay.linearBytes));
    std::vector<uint8_t> tiled(lay.bytes, 0);
    for (uint32_t sm = 0; sm < s.samples; ++sm) {
        for (uint8_t& b : planes[sm]) b = uint8_t(rng());
        CHECK(tileLevels(lay, 0, uint32_t(lay.levels.size()), sm, planes[sm].data(), tiled.data()));
    }
    for (uint32_t sm = 0; sm < s.samples; ++sm) {
        std::vector<uint8_t> back(lay.linearBytes, 0);
        CHECK(detileLevels(lay, 0, uint32_t(lay.levels.size()), sm, tiled.data(), back.data()));
        if (back != planes[sm]) {
            std::printf("  %s: round trip of sample %u differs\n", describe(s).c_str(), sm);
            ++failures;
            return;
        }
    }
    (void)before;
}

static TiledSurface surf(uint8_t mode, uint32_t bpe, uint32_t w, uint32_t h, uint32_t pitch, uint32_t slices, uint32_t levels,
                         uint32_t samples = 1) {
    TiledSurface s;
    s.tileIndex = mode, s.bytesPerElement = bpe, s.width = w, s.height = h, s.pitch = pitch, s.slices = slices;
    s.levels = levels, s.samples = samples;
    return s;
}

struct Golden {
    uint8_t mode, bpe, samples, bankSwizzle;
    uint32_t x, y, z, sample, pitch, height, offset;
};
static const Golden kGolden[] = {
#include "tiling_golden.inc"
};

int main() {
    const uint8_t modes[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 31};
    const uint32_t sizes[] = {1, 2, 4, 8, 16};

    // Every mode x element size x several shapes (mip chains, pitch beyond the width, volumes,
    // block-compressed sizes, power-of-two padding, cube faces) against the reference.
    for (uint8_t m : modes)
        for (uint32_t bpe : sizes) {
            checkSurface(surf(m, bpe, 70, 45, 0, 2, 4), 1);
            checkSurface(surf(m, bpe, 300, 200, 320, 1, 3), 5);
            TiledSurface vol = surf(m, bpe, 40, 24, 0, 9, 4);
            vol.volume = true;
            checkSurface(vol, 1);
            TiledSurface p2 = surf(m, bpe, 100, 40, 0, 1, 4);
            p2.pow2Pad = true;
            checkSurface(p2, 1);
            checkSurface(surf(m, bpe, 64, 64, 0, 6, 7), 5);  // cube faces down to 1x1
            if (bpe == 8 || bpe == 16) {
                TiledSurface bc = surf(m, bpe, 100, 60, 0, 1, 5);
                bc.blockSize = 4;
                checkSurface(bc, 1);
            }
            // Bank swizzle from the descriptor.
            TiledSurface sw = surf(m, bpe, 130, 70, 0, 3, 2);
            sw.bankSwizzle = 3;
            checkSurface(sw, 1);
        }

    // Samples: colour planes and depth interleave, tile split planes, every mode that can be multisampled.
    for (uint8_t m : modes) {
        if (m == 8 || m == 31) continue;
        for (uint32_t samples : {2u, 4u, 8u})
            for (uint32_t bpe : {1u, 4u, 8u}) {
                if (samples * bpe * 64 * microTileThickness(tileModeInfo(m).array) > (1u << 16)) continue;
                checkSurface(surf(m, bpe, 150, 90, 0, 2, 2, samples), 1);
            }
    }

    // Golden addresses (independent implementation, 512x512 surfaces of 16 slices).
    for (const Golden& g : kGolden) {
        TiledSurface s = surf(g.mode, g.bpe, 512, 512, 0, 16, 1, g.samples);
        s.bankSwizzle = g.bankSwizzle;
        const SurfaceLayout l = computeLayout(s);
        CHECK(l.ok);
        if (!l.ok) continue;
        if (l.levels[0].pitch != g.pitch || l.levels[0].paddedHeight != g.height ||
            vpt_offset(l.levels[0].hw, g.x, g.y, g.z, g.sample) != g.offset) {
            std::printf("  golden: mode %u bpe %u samples %u bank swizzle %u (%u,%u,%u,s%u): %u, expected %u\n", g.mode, g.bpe, g.samples,
                        g.bankSwizzle, g.x, g.y, g.z, g.sample, vpt_offset(l.levels[0].hw, g.x, g.y, g.z, g.sample), g.offset);
            ++failures;
        }
    }

    // Known layouts. Linear: rows `pitch` elements apart.
    {
        TiledSurface s{8, 4, 10, 4, 16, 1};
        CHECK(tiledOffset(s, 3, 2, 0) == (2 * 16 + 3) * 4);
    }
    // 1D thin (thin micro tiles): tile (1, 0) follows tile (0, 0); inside, Z order.
    {
        TiledSurface s{13, 4, 64, 64, 0, 1};
        CHECK(tiledOffset(s, 8, 0, 0) == 256);
        CHECK(tiledOffset(s, 1, 0, 0) == 4);
        CHECK(tiledOffset(s, 0, 1, 0) == 8);
        CHECK(tiledOffset(s, 2, 0, 0) == 16);
        CHECK(tiledOffset(s, 0, 8, 0) == 8 * 256);
    }
    // Display micro tile of 32-bit elements: x0 x1 y0 x2 y1 y2.
    {
        TiledSurface s{9, 4, 8, 8, 0, 1};
        CHECK(tiledOffset(s, 1, 0, 0) == 4);
        CHECK(tiledOffset(s, 2, 0, 0) == 8);
        CHECK(tiledOffset(s, 0, 1, 0) == 16);
        CHECK(tiledOffset(s, 4, 0, 0) == 32);
    }
    // 2D: the first 256 bytes are pipe 0 bank 0, linear inside one micro tile's first elements.
    {
        TiledSurface s{14, 4, 1024, 1024, 0, 1};
        CHECK(tiledOffset(s, 0, 0, 0) == 0);
        CHECK(tiledOffset(s, 1, 0, 0) == 4);
        CHECK(tiledOffset(s, 8, 0, 0) >= 256);  // next micro tile: another pipe
    }
    // Thick micro tile of 32-bit elements: x0 y0 x1 z0 y1 z1 x2 y2 (1D thick, slices in groups of 4).
    {
        TiledSurface s = surf(19, 4, 16, 16, 0, 8, 1);
        CHECK(tiledOffset(s, 1, 0, 0) == 4);
        CHECK(tiledOffset(s, 0, 1, 0) == 8);
        CHECK(tiledOffset(s, 2, 0, 0) == 16);
        CHECK(tiledOffset(s, 0, 0, 1) == 32);
        CHECK(tiledOffset(s, 0, 2, 0) == 64);
        CHECK(tiledOffset(s, 0, 0, 2) == 128);
        CHECK(tiledOffset(s, 0, 0, 4) == 16 * 16 * 4 * 4);  // second group of four slices
    }
    // Mip chain of a 512x512 32-bit 2D thin surface (macro tile 128x64): macro tiled down to 128x128,
    // 1D tiled from 64x64 (below one macro tile), each level after the previous one.
    {
        const SurfaceLayout l = computeLayout(surf(10, 4, 512, 512, 0, 1, 10));
        CHECK(l.ok && l.levels.size() == 10);
        const uint64_t bytes[10] = {1048576, 262144, 65536, 16384, 4096, 1024, 256, 256, 256, 256};
        uint64_t at = 0;
        for (size_t i = 0; l.ok && i < 10; ++i) {
            CHECK(l.levels[i].offset == at && l.levels[i].bytes == bytes[i]);
            CHECK(l.levels[i].degraded == (i >= 3));
            at += bytes[i];
        }
        CHECK(l.bytes == at);
        CHECK(l.levels[0].hw.macroPitch == 128 && l.levels[0].hw.macroHeight == 64);
    }
    // The pitch of a 1D level grows until a slice is whole 256-byte groups; level 0 keeps the descriptor's.
    {
        CHECK(computeLayout(surf(9, 1, 8, 8, 0, 1, 1)).levels[0].pitch == 32);
        CHECK(computeLayout(surf(9, 1, 8, 8, 8, 1, 1)).levels[0].pitch == 8);
    }
    // Macro tile modes: PRT variants (+8), tile split planes of depth, multisampled colour, thick.
    {
        SurfaceLayout l = computeLayout(surf(12, 4, 512, 512, 0, 1, 1));
        CHECK(l.ok && l.macroTileMode == 10 && l.levels[0].hw.macroPitch == 128 && l.levels[0].hw.macroHeight == 128);
        CHECK(l.levels[0].hw.flags & VPT_F_PRT);
        l = computeLayout(surf(0, 4, 512, 512, 0, 1, 1));  // depth, 64-byte tile split of 256-byte micro tiles
        CHECK(l.ok && l.levels[0].hw.splits == 4 && l.levels[0].hw.splitBytes == 64 && l.macroTileMode == 0);
        l = computeLayout(surf(4, 4, 512, 512, 0, 1, 1));  // 1K split: none
        CHECK(l.ok && l.levels[0].hw.splits == 1);
        l = computeLayout(surf(10, 4, 512, 512, 0, 1, 1, 4));
        CHECK(l.ok && l.levels[0].hw.splits == 2 && l.levels[0].hw.splitBytes == 512 && l.macroTileMode == 3);
        l = computeLayout(surf(20, 2, 512, 512, 0, 8, 1));
        CHECK(l.ok && l.thickness == 4 && l.macroTileMode == 3 && l.levels[0].hw.splits == 1);
        l = computeLayout(surf(25, 1, 512, 512, 0, 16, 1));
        CHECK(l.ok && l.thickness == 8);
    }
    // FMASK as a surface of its own: bits per pixel from samples and fragments.
    {
        CHECK(fmaskSurface(14, 64, 64, 64, 1, 4, 4).bytesPerElement == 1);
        CHECK(fmaskSurface(14, 64, 64, 64, 1, 8, 4).bytesPerElement == 2);
        CHECK(fmaskSurface(14, 64, 64, 64, 1, 8, 8).bytesPerElement == 4);
        CHECK(fmaskSurface(14, 64, 64, 64, 1, 2, 2).bytesPerElement == 1);
        checkSurface(fmaskSurface(14, 200, 120, 0, 2, 8, 8), 1);
        checkSurface(fmaskSurface(14, 200, 120, 0, 2, 4, 4), 1);
    }
    // Not covered: reported, nothing computed.
    {
        std::string why;
        CHECK(tiledBytes(surf(27, 4, 64, 64, 0, 1, 1), why) == 0 && !why.empty());
        CHECK(tiledBytes(surf(40, 4, 64, 64, 0, 1, 1), why) == 0);
        CHECK(!computeLayout(surf(10, 12, 64, 64, 0, 1, 1)).ok);
        CHECK(!computeLayout(surf(10, 4, 64, 64, 0, 1, 1, 3)).ok);
        CHECK(!computeLayout(surf(8, 4, 64, 64, 0, 1, 1, 2)).ok);
        CHECK(!computeLayout(surf(10, 4, 64, 64, 0, 1, 17)).ok);
        CHECK(computeLayout(surf(10, 4, 64, 64, 0, 1, 1)).offsetOf(0, 64, 0, 0) == UINT64_MAX);
    }
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("tiling: ok\n");
    return 0;
}
