#include "vpps4/msaa.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

namespace vpps4 {

namespace {
uint8_t floorPow2(uint8_t n) {
    if (n < 1) return 1;
    while (n & (n - 1)) n &= uint8_t(n - 1);
    return n;
}
}  // namespace

SamplePlan planSamples(std::span<const uint8_t> colorFragments, uint8_t depthSamples, uint8_t cap) {
    SamplePlan p;
    uint8_t host = std::max<uint8_t>(cap, 1);
    uint8_t minColor = 255, maxColor = 0;
    for (uint8_t c : colorFragments) {
        if (!c) continue;
        host = std::min(host, c);
        minColor = std::min(minColor, c);
        maxColor = std::max(maxColor, c);
        p.guest = std::max(p.guest, c);
    }
    if (depthSamples) {
        host = std::min(host, std::max<uint8_t>(depthSamples, 1));
        p.guest = std::max(p.guest, depthSamples);
        // EQAA: the depth buffer keeps more samples than the colour stores.
        if (maxColor && depthSamples > minColor) p.eqaa = true;
    }
    p.host = (p.guest == 1 && !depthSamples && !maxColor) ? 1 : floorPow2(host);
    p.reduced = p.host < p.guest;
    return p;
}

FmaskValue fmaskIdentity(uint8_t coverage, uint8_t fragments) {
    FmaskValue v;
    const unsigned frags = std::max<unsigned>(fragments, 1);
    const unsigned n = std::min<unsigned>(coverage, 16);
    for (unsigned s = 0; s < n; ++s) {
        const uint32_t nibble = s % frags;
        if (s < 8) v.lo |= nibble << (4 * s);
        else v.hi |= nibble << (4 * (s - 8));
    }
    return v;
}

vp4gpu::ResolveMode depthResolveMode(bool hardwareCopy, uint8_t copySample, uint8_t depthFunc) {
    if (hardwareCopy && copySample == 0) return vp4gpu::ResolveMode::Sample0;
    // DB_DEPTH_CONTROL.ZFUNC: FRAG_LESS 1, LEQUAL 3 conventional; GREATER 4, GEQUAL 6 reversed.
    if (depthFunc == 4 || depthFunc == 6) return vp4gpu::ResolveMode::Max;
    if (depthFunc == 1 || depthFunc == 3) return vp4gpu::ResolveMode::Min;
    return vp4gpu::ResolveMode::Sample0;
}

std::array<float, 4> resolveColor(std::span<const std::array<float, 4>> samples) {
    std::array<float, 4> sum{};
    if (samples.empty()) return sum;
    for (const auto& s : samples)
        for (int c = 0; c < 4; ++c) sum[c] += s[c];
    for (float& c : sum) c /= float(samples.size());
    return sum;
}

namespace {
float halfToFloat(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 31, man = h & 1023;
    float v;
    if (exp == 0) v = std::ldexp(float(man), -24);
    else if (exp == 31) v = man ? std::nanf("") : INFINITY;
    else v = std::ldexp(float(man | 1024), int(exp) - 25);
    return sign ? -v : v;
}
double srgbToLinear(double c) { return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4); }
}  // namespace

std::array<double, 4> fastClearColor(uint8_t cbFormat, uint8_t numberType, uint8_t compSwap,
                                     const std::array<uint32_t, 2>& words, bool& exact) {
    enum { Unorm = 0, Snorm = 1, Uint = 4, Sint = 5, Srgb = 6, Float = 7 };  // CB NUMBER_TYPE
    enum { SwapStd = 0, SwapAlt = 1, SwapStdRev = 2, SwapAltRev = 3 };       // CB COMP_SWAP
    // Components in memory order (first one in the low bits): bits per component and count.
    unsigned bits[4] = {0, 0, 0, 0}, n = 0;
    switch (cbFormat) {
        case 1: n = 1, bits[0] = 8; break;
        case 2: n = 1, bits[0] = 16; break;
        case 3: n = 2, bits[0] = bits[1] = 8; break;
        case 4: n = 1, bits[0] = 32; break;
        case 5: n = 2, bits[0] = bits[1] = 16; break;
        case 8: n = 4, bits[0] = 2, bits[1] = bits[2] = bits[3] = 10; break;  // 10_10_10_2 (alpha in the low bits)
        case 9: n = 4, bits[0] = bits[1] = bits[2] = 10, bits[3] = 2; break;  // 2_10_10_10
        case 10: n = 4, bits[0] = bits[1] = bits[2] = bits[3] = 8; break;
        case 11: n = 2, bits[0] = bits[1] = 32; break;
        case 12: n = 4, bits[0] = bits[1] = bits[2] = bits[3] = 16; break;
        case 14: n = 3, bits[0] = 5, bits[1] = 6, bits[2] = 5; break;                   // 5_6_5
        case 15: n = 4, bits[0] = bits[1] = bits[2] = 5, bits[3] = 1; break;            // 1_5_5_5
        case 16: n = 4, bits[0] = 1, bits[1] = bits[2] = bits[3] = 5; break;            // 5_5_5_1
        case 17: n = 4, bits[0] = bits[1] = bits[2] = bits[3] = 4; break;               // 4_4_4_4
        default: exact = false; return {0, 0, 0, 0};
    }
    if (numberType == Float && !(bits[0] == 16 || bits[0] == 32)) {  // (small floats are not decoded)
        exact = false;
        return {0, 0, 0, 0};
    }
    const uint64_t all = uint64_t(words[0]) | uint64_t(words[1]) << 32;
    std::array<double, 4> mem{};
    unsigned at = 0;
    for (unsigned c = 0; c < n; ++c) {
        const uint64_t mask = bits[c] >= 32 ? 0xffffffffull : (1ull << bits[c]) - 1;
        const uint64_t raw = (all >> at) & mask;
        at += bits[c];
        double v;
        switch (numberType) {
            case Float:
                v = bits[c] == 32 ? double(std::bit_cast<float>(uint32_t(raw))) : double(halfToFloat(uint16_t(raw)));
                break;
            case Snorm: {
                const int64_t sv = int64_t(raw << (64 - bits[c])) >> (64 - bits[c]);
                v = bits[c] == 1 ? double(sv) : std::max(double(sv) / double((1ull << (bits[c] - 1)) - 1), -1.0);
                break;
            }
            case Sint: v = double(int64_t(raw << (64 - bits[c])) >> (64 - bits[c])); break;
            case Uint: v = double(raw); break;
            default: v = double(raw) / double(mask); break;  // UNORM, SRGB (encoded value)
        }
        mem[c] = v;
    }
    // COMP_SWAP: where the logical R G B A sit in memory order.
    //   4 components: STD RGBA, ALT BGRA, STD_REV ABGR, ALT_REV ARGB.
    //   3 components: STD RGB; the other swaps BGR.
    //   2 components: STD RG, ALT RA, STD_REV GR, ALT_REV AR.
    //   1 component:  R, except ALT_REV, which is the alpha-only layout.
    std::array<double, 4> rgba{0, 0, 0, 1.0};
    switch (n) {
        case 4:
            switch (compSwap) {
                case SwapAlt: rgba = {mem[2], mem[1], mem[0], mem[3]}; break;
                case SwapStdRev: rgba = {mem[3], mem[2], mem[1], mem[0]}; break;
                case SwapAltRev: rgba = {mem[1], mem[2], mem[3], mem[0]}; break;
                default: rgba = {mem[0], mem[1], mem[2], mem[3]}; break;
            }
            break;
        case 3:
            if (compSwap == SwapAlt || compSwap == SwapStdRev) rgba = {mem[2], mem[1], mem[0], 1.0};
            else rgba = {mem[0], mem[1], mem[2], 1.0};
            break;
        case 2:
            switch (compSwap) {
                case SwapAlt: rgba = {mem[0], 0, 0, mem[1]}; break;
                case SwapStdRev: rgba = {mem[1], mem[0], 0, 1.0}; break;
                case SwapAltRev: rgba = {mem[1], 0, 0, mem[0]}; break;
                default: rgba = {mem[0], mem[1], 0, 1.0}; break;
            }
            break;
        default:
            if (compSwap == SwapAltRev) rgba = {0, 0, 0, mem[0]};
            else rgba = {mem[0], 0, 0, 1.0};
            break;
    }
    if (numberType == Srgb)
        for (int c = 0; c < 3; ++c) rgba[c] = srgbToLinear(rgba[c]);
    return rgba;
}

ResolveWay planResolve(const ResolveCase& c) {
    if (c.destinationIsSource || c.destinationMultisampled) return ResolveWay::Skip;
    if (!c.formatsMatch && !c.formatsViewable) return ResolveWay::Skip;
    // A blit copies between identical pixel formats only; a resolve goes through a view of the destination.
    if (!c.sourceMultisampled) return c.formatsMatch ? ResolveWay::Copy : ResolveWay::Skip;
    return c.lastPassOpenForSource ? ResolveWay::InPass : ResolveWay::OwnPass;
}

}  // namespace vpps4
