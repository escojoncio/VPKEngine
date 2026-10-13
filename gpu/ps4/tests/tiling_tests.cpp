// Tiling: every element of a tiled surface has its own place inside the surface's span, and
// detile() undoes tile(). Known layouts: linear rows, 1D thin micro tiles in raster order, the
// display micro tile order of 32-bit elements.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

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

static void bijective(const TiledSurface& s) {
    std::string why;
    const uint64_t bytes = tiledBytes(s, why);
    CHECK(bytes > 0);
    if (!bytes) {
        std::printf("  index %u bpe %u %ux%u: %s\n", s.tileIndex, s.bytesPerElement, s.width, s.height, why.c_str());
        return;
    }
    std::vector<uint8_t> used(bytes / s.bytesPerElement, 0);
    for (uint32_t z = 0; z < s.slices; ++z)
        for (uint32_t y = 0; y < s.height; ++y)
            for (uint32_t x = 0; x < s.width; ++x) {
                const uint64_t o = tiledOffset(s, x, y, z);
                if (o + s.bytesPerElement > bytes || o % s.bytesPerElement) {
                    std::printf("  index %u bpe %u: (%u,%u,%u) at %llu of %llu\n", s.tileIndex, s.bytesPerElement, x, y, z,
                                (unsigned long long)o, (unsigned long long)bytes);
                    ++failures;
                    return;
                }
                if (used[o / s.bytesPerElement]++) {
                    std::printf("  index %u bpe %u: (%u,%u,%u) shares %llu\n", s.tileIndex, s.bytesPerElement, x, y, z,
                                (unsigned long long)o);
                    ++failures;
                    return;
                }
            }
    // Round trip.
    const uint64_t linear = uint64_t(s.width) * s.height * s.slices * s.bytesPerElement;
    std::vector<uint8_t> a(linear), t(bytes, 0), b(linear, 0);
    for (uint64_t i = 0; i < linear; ++i) a[i] = uint8_t(i * 2654435761u >> 13);
    CHECK(tile(s, a.data(), t.data(), why));
    CHECK(detile(s, t.data(), b.data(), why));
    CHECK(a == b);
}

int main() {
    // Every covered mode, element size and a few shapes (including ones smaller than a macro tile,
    // odd sizes inside a padded pitch, and arrays).
    const uint8_t indices[] = {2, 3, 4, 5, 8, 9, 10, 13, 14, 31};
    const uint32_t sizes[] = {1, 2, 4, 8, 16};
    struct Shape {
        uint32_t w, h, pitch, slices;
    } shapes[] = {{8, 8, 0, 1}, {100, 37, 128, 1}, {256, 256, 0, 1}, {512, 300, 0, 2}, {1024, 64, 0, 3}};
    for (uint8_t i : indices)
        for (uint32_t bpe : sizes)
            for (const Shape& sh : shapes) {
                TiledSurface s{i, bpe, sh.w, sh.h, sh.pitch, sh.slices};
                std::string why;
                if (!tiledBytes(s, why)) {
                    // depth modes split large micro tiles: reported, not covered
                    CHECK(!why.empty());
                    continue;
                }
                bijective(s);
            }

    // Linear: rows `pitch` elements apart.
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
    // Not covered: reported.
    {
        std::string why;
        TiledSurface s{20, 4, 64, 64, 0, 1};
        CHECK(tiledBytes(s, why) == 0 && !why.empty());
    }
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("tiling: ok\n");
    return 0;
}
