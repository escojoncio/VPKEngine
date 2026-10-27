// Format plans: every data x number format of a texture has a plan (native, converted or raw
// container) or an explicit reason; dstSel composition; colour and depth selections.
#include <cstdio>

#include "vpps4/formats.h"

using namespace vpps4;

static int failures = 0;
#define CHECK(c)                                                    \
    do {                                                            \
        if (!(c)) {                                                 \
            std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                             \
        }                                                           \
    } while (0)

int main() {
    // Every plain format x valid number format yields a plan (a host format or a container).
    const unsigned plain[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19, 34, 57, 58};
    const unsigned nf[] = {0, 1, 2, 3, 4, 5, 6, 7, 9};
    for (unsigned d : plain)
        for (unsigned n : nf) {
            TexturePlan p = texturePlan(uint8_t(d), uint8_t(n));
            bool ok = p.format != vpgpu::Format::Undefined;
            if (!ok) std::printf("  no plan dfmt %u nfmt %u\n", d, n);
            CHECK(ok);
            if (!p.exact) CHECK(p.why != nullptr);
        }
    // 32_32_32 (12-byte texels) has no host texture format: it must say why.
    CHECK(texturePlan(13, 4).format == vpgpu::Format::Undefined);
    CHECK(texturePlan(13, 4).why != nullptr);
    // dstSel: identity keeps the plan's select, a swap reorders it.
    uint8_t planSel[4] = {0, 1, 2, 3}, out[4];
    composeSelect({4, 5, 6, 7}, planSel, out);
    CHECK(out[0] == 0 && out[1] == 1 && out[2] == 2 && out[3] == 3);
    composeSelect({6, 5, 4, 7}, planSel, out);
    CHECK(out[0] == 2 && out[1] == 1 && out[2] == 0 && out[3] == 3);
    // Depth: 32-bit float is exact; Z16+S8 and Z24 are flagged approximations.
    bool exact = false;
    CHECK(depthFormat(3, 0, &exact) != vpgpu::Format::Undefined);
    CHECK(exact);
    CHECK(depthFormat(1, 1, &exact) != vpgpu::Format::Undefined);
    CHECK(!exact);
    // BC with host support is native; without, it is decoded on the CPU to a plain host format and the guest layout
    // stays the block format.
    CHECK(texturePlan(35, 0).format == vpgpu::Format::BC1 && texturePlan(35, 0).softBc == 0);
    setFormatCaps({false});
    struct Row {
        unsigned dfmt, nfmt;
        vpgpu::Format host, layout;
        bool isSigned;
    };
    const Row rows[] = {
        {35, 0, vpgpu::Format::RGBA8Unorm, vpgpu::Format::BC1, false}, {35, 9, vpgpu::Format::RGBA8Srgb, vpgpu::Format::BC1Srgb, false},
        {36, 0, vpgpu::Format::RGBA8Unorm, vpgpu::Format::BC2, false}, {37, 9, vpgpu::Format::RGBA8Srgb, vpgpu::Format::BC3Srgb, false},
        {38, 0, vpgpu::Format::R8Unorm, vpgpu::Format::BC4Unorm, false}, {38, 1, vpgpu::Format::R8Snorm, vpgpu::Format::BC4Snorm, true},
        {39, 0, vpgpu::Format::RG8Unorm, vpgpu::Format::BC5Unorm, false}, {39, 1, vpgpu::Format::RG8Snorm, vpgpu::Format::BC5Snorm, true},
        {40, 7, vpgpu::Format::RGBA16Float, vpgpu::Format::BC6HUfloat, false}, {40, 1, vpgpu::Format::RGBA16Float, vpgpu::Format::BC6HSfloat, true},
        {41, 0, vpgpu::Format::RGBA8Unorm, vpgpu::Format::BC7, false}, {41, 9, vpgpu::Format::RGBA8Srgb, vpgpu::Format::BC7Srgb, false},
    };
    for (const Row& r : rows) {
        const TexturePlan p = texturePlan(uint8_t(r.dfmt), uint8_t(r.nfmt));
        CHECK(p.format == r.host && p.layout == r.layout && p.softBc == r.dfmt && p.softSigned == r.isSigned);
        CHECK(p.conv.select[0] == 4 && p.conv.select[3] == 7 && !p.conv.manual && p.elem == TexturePlan::Float);
    }
    CHECK(texturePlan(35, 0, true).format == vpgpu::Format::Undefined && texturePlan(35, 0, true).why != nullptr);  // (as a storage texture)
    CHECK(texturePlan(1, 0).softBc == 0);
    setFormatCaps({true});
    // A V# with tag bits above bit 43 (0x7 << 12 in word 1): the base is the 44-bit address.
    const uint32_t vsharp[4] = {0x8d3fe510u, 0x00007070u, 0x40000u, 0};
    CHECK(decodeBuffer(vsharp).address == 0x708d3fe510ull);
    if (failures) return 1;
    std::printf("format plans ok\n");
    return 0;
}
