// Liverpool data / number formats: layouts and the CPU reference conversions. See vpgcn/formats.h.
// Float arithmetic is written in the same order as the VPIR emitters (format_emit.cpp) so tests can
// compare bit for bit; build this file without FMA contraction.
#include "vpgcn/formats.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

namespace vpgcn {
namespace {

constexpr FormatLayout makePlain(std::initializer_list<int> msbFirst, int bytes = 0) {
    FormatLayout l;
    l.kind = FormatLayout::Plain;
    const int n = int(msbFirst.size());
    l.comps = uint8_t(n);
    int widths[4] = {};
    int k = n - 1;
    for (int w : msbFirst) widths[k--] = w;  // last named = x
    int at = 0;
    for (int c = 0; c < n; ++c) {
        l.bits[c] = uint8_t(widths[c]);
        l.shift[c] = uint8_t(at);
        at += widths[c];
    }
    l.bytes = uint8_t(bytes ? bytes : (at + 7) / 8);
    return l;
}

constexpr FormatLayout makeKind(FormatLayout::Kind k, int bytes, int texels = 1, bool reversed = false) {
    FormatLayout l;
    l.kind = k;
    l.bytes = uint8_t(bytes);
    l.texelsPerElement = uint8_t(texels);
    l.reversed = reversed;
    return l;
}

struct Table {
    FormatLayout f[64];
    constexpr Table() {
        f[1] = makePlain({8});
        f[2] = makePlain({16});
        f[3] = makePlain({8, 8});
        f[4] = makePlain({32});
        f[5] = makePlain({16, 16});
        f[6] = makePlain({10, 11, 11});
        f[7] = makePlain({11, 11, 10});
        f[8] = makePlain({10, 10, 10, 2});
        f[9] = makePlain({2, 10, 10, 10});
        f[10] = makePlain({8, 8, 8, 8});
        f[11] = makePlain({32, 32});
        f[12] = makePlain({16, 16, 16, 16});
        f[13] = makePlain({32, 32, 32});
        f[14] = makePlain({32, 32, 32, 32});
        f[16] = makePlain({5, 6, 5});
        f[17] = makePlain({1, 5, 5, 5});
        f[18] = makePlain({5, 5, 5, 1});
        f[19] = makePlain({4, 4, 4, 4});
        f[20] = makePlain({8, 24});
        f[21] = makePlain({24, 8});
        f[22] = makePlain({8, 32}, 8);  // X24_8_32: x 32-bit float depth, y stencil in the second dword
        f[32] = makeKind(FormatLayout::Subsampled, 4, 2);
        f[33] = makeKind(FormatLayout::Subsampled, 4, 2);
        f[34] = makeKind(FormatLayout::SharedExp, 4);
        for (int k = 35; k <= 41; ++k) f[k] = makeKind(FormatLayout::Block, k == 35 || k == 38 ? 8 : 16);
        // FMASK8_S2_F1, 8_S4_F1, 8_S8_F1 (1 byte); 16_S16_F1, 16_S8_F2 (2); 32_S16_F2, 32_S8_F4, 32_S8_F8 (4);
        // 64_S16_F4, 64_S16_F8 (8)
        for (int k = 47; k <= 49; ++k) f[k] = makeKind(FormatLayout::Fmask, 1);
        for (int k = 50; k <= 51; ++k) f[k] = makeKind(FormatLayout::Fmask, 2);
        for (int k = 52; k <= 54; ++k) f[k] = makeKind(FormatLayout::Fmask, 4);
        for (int k = 55; k <= 56; ++k) f[k] = makeKind(FormatLayout::Fmask, 8);
        f[57] = makePlain({4, 4});
        f[58] = makePlain({6, 5, 5});
        f[59] = makeKind(FormatLayout::OneBit, 1, 8, false);
        f[60] = makeKind(FormatLayout::OneBit, 1, 8, true);
    }
};
constexpr Table kTable;

uint32_t maskOf(unsigned width) { return width >= 32 ? 0xffffffffu : (1u << width) - 1; }
float maxUnsigned(unsigned width) { return float((uint64_t(1) << width) - 1); }
float maxSigned(unsigned width) { return float((uint64_t(1) << (width - 1)) - 1); }
int32_t signExtend(uint32_t v, unsigned width) {
    if (width >= 32) return int32_t(v);
    const unsigned up = 32 - width;
    return int32_t(v << up) >> up;
}
float bitsF(uint32_t b) { return std::bit_cast<float>(b); }
uint32_t fBits(float f) { return std::bit_cast<uint32_t>(f); }

// f32 -> u32 as the IR converts (truncate, saturate, NaN 0).
uint32_t toU32(float v) {
    if (std::isnan(v)) return 0;
    double t = std::trunc(double(v));
    return uint32_t(std::fmin(std::fmax(t, 0.0), 4294967295.0));
}
int32_t toI32(float v) {
    if (std::isnan(v)) return 0;
    double t = std::trunc(double(v));
    return int32_t(std::fmin(std::fmax(t, -2147483648.0), 2147483647.0));
}

}  // namespace

const FormatLayout& layoutOf(unsigned dfmt) { return kTable.f[dfmt & 63]; }

FieldKind fieldKind(unsigned width, unsigned nfmt) {
    switch (nfmt) {
        case kUnorm:
        case kSrgb: return FieldKind::Unorm;
        case kSnorm: return FieldKind::Snorm;
        case kUscaled: return FieldKind::Uscaled;
        case kSscaled: return FieldKind::Sscaled;
        case kUint: return FieldKind::Uint;
        case kSint: return FieldKind::Sint;
        case kSnormNz: return FieldKind::SnormNz;
        case kUbnorm: return FieldKind::Ubnorm;
        default: return width == 32 || width == 16 || width == 11 || width == 10 ? FieldKind::Float : FieldKind::Uint;
    }
}

bool isIntegerNumFmt(unsigned nfmt) { return nfmt == kUint || nfmt == kSint; }
bool numFmtValid(unsigned nfmt) { return nfmt <= kFloat || nfmt == kSrgb || nfmt == kUbnorm; }

float srgbToLinear(float c) {
    const float a = std::pow((c + 0.055f) / 1.055f, 2.4f);
    const float b = c / 12.92f;
    return c <= 0.04045f ? b : a;
}
float linearToSrgb(float c) {
    const float a = 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
    const float b = c * 12.92f;
    return c <= 0.0031308f ? b : a;
}

float smallFloatToF32(uint32_t field, unsigned mant) {
    const uint32_t m = field & ((1u << mant) - 1), e = field >> mant;
    if (e == 31) return bitsF(0x7f800000u | (m << (23 - mant)));
    if (e == 0) return float(m) * bitsF(uint32_t(113 - mant) << 23);
    return bitsF(((e + 112) << 23) | (m << (23 - mant)));
}

uint32_t f32ToSmallFloat(float v, unsigned mant) {
    const uint32_t maxPat = (30u << mant) | ((1u << mant) - 1);
    if (std::isnan(v)) return (31u << mant) | (1u << (mant - 1));
    if (!(v > 0.0f)) return 0;
    if (std::isinf(v)) return 31u << mant;
    const uint32_t bits = fBits(v), e = (bits >> 23) & 0xff, m = bits & 0x7fffff;
    const int32_t E = int32_t(e) - 112;
    uint32_t pat;
    if (E >= 1) {
        const unsigned drop = 23 - mant;
        const uint32_t mq = m >> drop, round = (m >> (drop - 1)) & 1, sticky = (m & ((1u << (drop - 1)) - 1)) != 0;
        const uint32_t inc = round & (sticky | (mq & 1));
        pat = (uint32_t(E) << mant) + mq + inc;
        if (pat > maxPat) pat = maxPat;
    } else {
        pat = toU32(std::nearbyint(v * bitsF(uint32_t(127 + 14 + mant) << 23)));
    }
    return pat;
}

float halfToF32(uint32_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v;
    if (e == 0) v = std::ldexp(float(m), -24);
    else if (e == 31) v = m ? bitsF(0x7fc00000u | (m << 13)) : bitsF(0x7f800000u);
    else v = std::ldexp(float(m | 1024), int(e) - 25);
    return s ? -v : v;
}

uint32_t f32ToHalf(float v) {
    // Round to nearest even through double arithmetic (exact for every f32 input).
    if (std::isnan(v)) return 0x7e00u | ((fBits(v) >> 31) << 15);
    const uint32_t s = std::signbit(v) ? 0x8000u : 0;
    const double a = std::fabs(double(v));
    if (std::isinf(v) || a >= 65520.0) return s | 0x7c00u;
    if (a < std::ldexp(1.0, -14)) return s | uint32_t(std::nearbyint(std::ldexp(a, 24)));  // may round up to 0x400
    int ex;
    std::frexp(a, &ex);  // a = f * 2^ex, f in [0.5, 1)
    const int e = ex - 1;
    const uint32_t mq = uint32_t(std::nearbyint(std::ldexp(a, -e + 10))) - 1024;  // 1024..2048 -> 0..1024
    return s | ((uint32_t(e + 15) << 10) + mq);
}

namespace {

// One field of an element: value after the number format, as a bit pattern.
uint32_t decodeField(uint32_t raw, unsigned w, FieldKind k) {
    switch (k) {
        case FieldKind::Unorm: return fBits(float(raw) / maxUnsigned(w));
        case FieldKind::Snorm: {
            const float v = float(signExtend(raw, w)) / maxSigned(w);
            return fBits(std::fmax(v, -1.0f));
        }
        case FieldKind::Uscaled: return fBits(float(raw));
        case FieldKind::Sscaled: return fBits(float(signExtend(raw, w)));
        case FieldKind::Uint: return raw;
        case FieldKind::Sint: return uint32_t(signExtend(raw, w));
        case FieldKind::SnormNz: return fBits((float(signExtend(raw, w)) * 2.0f + 1.0f) / maxUnsigned(w));
        case FieldKind::Ubnorm: return fBits(float(raw) / maxUnsigned(w) * 2.0f - 1.0f);
        case FieldKind::Float:
            if (w == 32) return raw;
            if (w == 16) return fBits(halfToF32(raw));
            return fBits(smallFloatToF32(raw, w - 5));
    }
    return 0;
}

uint32_t encodeFieldRef(uint32_t v, unsigned w, FieldKind k, bool saturate) {
    const uint32_t mask = maskOf(w);
    const float fv = bitsF(v);
    switch (k) {
        case FieldKind::Unorm: {
            const float s = std::isnan(fv) ? 0.0f : std::fmin(std::fmax(fv, 0.0f), 1.0f);
            if (w == 32) return s >= 1.0f ? 0xffffffffu : toU32(std::nearbyint(s * 4294967040.0f));
            return toU32(std::nearbyint(s * maxUnsigned(w)));
        }
        case FieldKind::Snorm: {
            const float s = std::isnan(fv) ? 0.0f : std::fmin(std::fmax(fv, -1.0f), 1.0f);
            const float m = w == 32 ? 2147483520.0f : maxSigned(w);
            return uint32_t(toI32(std::nearbyint(s * m))) & mask;
        }
        case FieldKind::Uscaled: {
            const float s = std::isnan(fv) ? 0.0f : fv;
            const float m = w == 32 ? 4294967040.0f : maxUnsigned(w);
            return toU32(std::fmin(std::fmax(s, 0.0f), m));
        }
        case FieldKind::Sscaled: {
            const float s = std::isnan(fv) ? 0.0f : fv;
            const float hi = w == 32 ? 2147483520.0f : maxSigned(w);
            const float lo = -maxSigned(w) - (w == 32 ? 128.0f : 1.0f);
            return uint32_t(toI32(std::fmin(std::fmax(s, lo), hi))) & mask;
        }
        case FieldKind::Uint:
            if (!saturate) return v & mask;
            return std::min(v, mask);
        case FieldKind::Sint:
            if (!saturate) return v & mask;
            if (w >= 32) return v;
            return uint32_t(std::clamp<int32_t>(int32_t(v), -(1 << (w - 1)), (1 << (w - 1)) - 1)) & mask;
        case FieldKind::SnormNz: {
            const float s = std::isnan(fv) ? 0.0f : std::fmin(std::fmax(fv, -1.0f), 1.0f);
            const float t = std::nearbyint((s * maxUnsigned(w) - 1.0f) / 2.0f);
            const float lo = -(float)(uint64_t(1) << (w - 1)), hi = maxSigned(w);
            return uint32_t(toI32(std::fmin(std::fmax(t, lo), hi))) & mask;
        }
        case FieldKind::Ubnorm: {
            const float s = std::isnan(fv) ? 0.0f : std::fmin(std::fmax((fv + 1.0f) / 2.0f, 0.0f), 1.0f);
            return toU32(std::nearbyint(s * maxUnsigned(w)));
        }
        case FieldKind::Float:
            if (w == 32) return v;
            if (w == 16) return f32ToHalf(fv);
            return f32ToSmallFloat(fv, w - 5);
    }
    return 0;
}

}  // namespace

bool decodeTexel(unsigned dfmt, unsigned nfmt, const uint32_t words[4], unsigned sub, uint32_t out[4]) {
    if (!numFmtValid(nfmt)) return false;
    const FormatLayout& l = layoutOf(dfmt);
    const bool integer = isIntegerNumFmt(nfmt);
    const uint32_t one = integer ? 1u : fBits(1.0f);
    out[0] = out[1] = out[2] = 0;
    out[3] = one;
    switch (l.kind) {
        case FormatLayout::Plain: {
            for (unsigned c = 0; c < l.comps; ++c) {
                const unsigned w = l.bits[c], at = l.shift[c];
                uint32_t raw = words[at / 32] >> (at % 32);
                raw &= maskOf(w);
                const FieldKind k = fieldKind(w, nfmt);
                uint32_t v = decodeField(raw, w, k);
                if (nfmt == kSrgb && c < 3) v = fBits(srgbToLinear(bitsF(v)));
                out[c] = v;
            }
            return true;
        }
        case FormatLayout::SharedExp: {
            const uint32_t w = words[0];
            const int e = int(w >> 27) - 24;
            const float scale = std::ldexp(1.0f, e);
            out[0] = fBits(float(w & 0x1ff) * scale);
            out[1] = fBits(float((w >> 9) & 0x1ff) * scale);
            out[2] = fBits(float((w >> 18) & 0x1ff) * scale);
            out[3] = fBits(1.0f);
            return true;
        }
        case FormatLayout::Fmask:
            out[0] = words[0];
            out[1] = l.bytes > 4 ? words[1] : 0;
            out[2] = 0;
            out[3] = 1;
            return true;
        case FormatLayout::Subsampled: {
            if (nfmt != kUnorm && nfmt != kSrgb) return false;
            const uint32_t w = words[0];
            const unsigned b0 = w & 255, b1 = (w >> 8) & 255, b2 = (w >> 16) & 255, b3 = w >> 24;
            // GB_GR (bytes high to low G B G R): texel 0 = (R b0, G b1, B b2), texel 1 = (b0, b3, b2).
            // BG_RG (B G R G): texel 0 = (R b1, G b0, B b3), texel 1 = (b1, b2, b3).
            unsigned r, g, b;
            if (dfmt == kFmtGB_GR) r = b0, g = (sub & 1) ? b3 : b1, b = b2;
            else r = b1, g = (sub & 1) ? b2 : b0, b = b3;
            const unsigned c3[3] = {r, g, b};
            for (unsigned c = 0; c < 3; ++c) {
                float v = float(c3[c]) / 255.0f;
                if (nfmt == kSrgb) v = srgbToLinear(v);
                out[c] = fBits(v);
            }
            out[3] = fBits(1.0f);
            return true;
        }
        case FormatLayout::OneBit: {
            if (nfmt != kUnorm && nfmt != kUint) return false;
            const unsigned idx = l.reversed ? 7 - (sub & 7) : (sub & 7);
            const uint32_t bit = (words[0] >> idx) & 1;
            out[0] = integer ? bit : fBits(float(bit));
            return true;
        }
        default: return false;
    }
}

bool encodeTexel(unsigned dfmt, unsigned nfmt, const uint32_t in[4], unsigned have, bool saturate, uint32_t words[4],
                 uint32_t masks[4]) {
    if (!numFmtValid(nfmt)) return false;
    const FormatLayout& l = layoutOf(dfmt);
    for (int k = 0; k < 4; ++k) words[k] = masks[k] = 0;
    if (l.kind != FormatLayout::Plain) return false;
    for (unsigned c = 0; c < l.comps; ++c) {
        if (!(have >> c & 1)) continue;
        const unsigned w = l.bits[c], at = l.shift[c];
        const FieldKind k = fieldKind(w, nfmt);
        uint32_t v = in[c];
        if (nfmt == kSrgb && c < 3) {
            const float s = std::isnan(bitsF(v)) ? 0.0f : std::fmin(std::fmax(bitsF(v), 0.0f), 1.0f);
            v = fBits(linearToSrgb(s));
        }
        const uint32_t field = encodeFieldRef(v, w, k, saturate) & maskOf(w);
        words[at / 32] |= field << (at % 32);
        masks[at / 32] |= maskOf(w) << (at % 32);
    }
    return true;
}

}  // namespace vpgcn
