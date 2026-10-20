// Liverpool format conversions: the CPU reference (formats.cpp) against its own invariants and
// against the VPIR emitters (format_emit.cpp) run on the interpreter, for every data format x
// number format, decode and encode, bit for bit.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "diff/vpir_eval.h"
#include "vpgcn/formats.h"
#include "vp4ir/verify.h"

using namespace vpgcn;
namespace V = vp4ir;

namespace {

unsigned failures = 0;
#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            if (++failures < 40) {                        \
                std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                std::fprintf(stderr, __VA_ARGS__);        \
                std::fprintf(stderr, "\n");               \
            }                                             \
        }                                                 \
    } while (0)

float bitsF(uint32_t b) { return std::bit_cast<float>(b); }
uint32_t fBits(float f) { return std::bit_cast<uint32_t>(f); }

// Equal bits, or both NaN (any NaN is any NaN), or both zeros.
bool sameValue(uint32_t a, uint32_t b, bool floatClass) {
    if (a == b) return true;
    if (!floatClass) return false;
    if (std::isnan(bitsF(a)) && std::isnan(bitsF(b))) return true;
    return (a | b) == 0x80000000u;
}

const unsigned kNumFmts[] = {kUnorm, kSnorm, kUscaled, kSscaled, kUint, kSint, kSnormNz, kFloat, kSrgb, kUbnorm};

// ---- reference invariants -------------------------------------------------------------------------

void testHalf() {
    for (uint32_t h = 0; h < 65536; ++h) {
        const float v = halfToF32(h);
        CHECK(fBits(v) == fBits(V::halfToFloat(uint16_t(h))) || (std::isnan(v) && ((h >> 10) & 31) == 31),
              "halfToF32(%04x)", h);
        if (std::isnan(v)) continue;
        CHECK(f32ToHalf(v) == h || (h & 0x7fff) == 0, "half round trip %04x -> %08x -> %04x", h, fBits(v), f32ToHalf(v));
    }
    std::mt19937 rng(1);
    for (int i = 0; i < 200000; ++i) {
        uint32_t b = rng();
        if (i % 3 == 0) b = (b & 0x807fffffu) | ((100 + rng() % 40) << 23);  // around the half range
        const float v = bitsF(b);
        if (std::isnan(v)) continue;
        CHECK(f32ToHalf(v) == V::floatToHalf(v), "f32ToHalf(%08x): %04x vs %04x", b, f32ToHalf(v), V::floatToHalf(v));
    }
}

void testSmallFloat() {
    for (unsigned mant : {5u, 6u}) {
        const unsigned n = 1u << (5 + mant);
        std::vector<float> table(n);
        for (uint32_t p = 0; p < n; ++p) {
            table[p] = smallFloatToF32(p, mant);
            if ((p >> mant) != 31) {
                CHECK(f32ToSmallFloat(table[p], mant) == p, "small float %u round trip %x", mant + 5, p);
            }
        }
        // monotonic over finite patterns
        for (uint32_t p = 1; p < (31u << mant); ++p) CHECK(table[p] > table[p - 1], "monotonic %x", p);
        // nearest-even against the table
        std::mt19937 rng(2);
        const uint32_t maxPat = (30u << mant) | ((1u << mant) - 1);
        for (int i = 0; i < 200000; ++i) {
            float v = bitsF((rng() & 0x7fffffffu));
            if (i % 2) v = std::ldexp(float(rng() % 100000 + 1) / 100000.0f, int(rng() % 40) - 24);
            if (!(v > 0) || std::isinf(v) || std::isnan(v)) continue;
            uint32_t got = f32ToSmallFloat(v, mant);
            if (v >= table[maxPat]) {
                CHECK(got == maxPat || got == maxPat, "clamp %g -> %x", v, got);
                continue;
            }
            uint32_t lo = 0;
            while (lo + 1 <= maxPat && table[lo + 1] <= v) ++lo;
            const uint32_t hi = lo + 1;
            const double dl = double(v) - double(table[lo]), dh = double(table[hi]) - double(v);
            const uint32_t want = dl < dh ? lo : dh < dl ? hi : ((lo & 1) ? hi : lo);
            CHECK(got == want, "nearest %.9g (%u-bit): %x want %x", v, mant + 5, got, want);
        }
        CHECK(f32ToSmallFloat(-1.0f, mant) == 0, "negative");
        CHECK(f32ToSmallFloat(0.0f, mant) == 0, "zero");
        CHECK((f32ToSmallFloat(INFINITY, mant) >> mant) == 31 && (f32ToSmallFloat(INFINITY, mant) & ((1u << mant) - 1)) == 0, "inf");
        CHECK((f32ToSmallFloat(NAN, mant) >> mant) == 31 && (f32ToSmallFloat(NAN, mant) & ((1u << mant) - 1)) != 0, "nan");
    }
}

void testSrgb() {
    for (int i = 0; i < 256; ++i) {
        const float lin = srgbToLinear(float(i) / 255.0f);
        const float back = linearToSrgb(lin) * 255.0f;
        CHECK(std::fabs(back - float(i)) < 0.01f, "srgb %d -> %g -> %g", i, lin, back);
    }
    CHECK(srgbToLinear(0.0f) == 0.0f && std::fabs(srgbToLinear(1.0f) - 1.0f) < 1e-6f, "srgb ends");
    CHECK(std::fabs(srgbToLinear(0.5f) - 0.21404114f) < 1e-6f, "srgb 0.5");
}

void testLayouts() {
    for (unsigned d = 0; d < 64; ++d) {
        const FormatLayout& l = layoutOf(d);
        if (l.kind != FormatLayout::Plain) continue;
        unsigned end = 0;
        for (unsigned c = 0; c < l.comps; ++c) {
            CHECK(l.shift[c] == end, "dfmt %u comp %u not contiguous", d, c);
            end = l.shift[c] + l.bits[c];
            CHECK(l.shift[c] / 32 == (l.shift[c] + l.bits[c] - 1) / 32, "dfmt %u comp %u straddles a dword", d, c);
        }
        CHECK(end <= unsigned(l.bytes) * 8 && l.bytes <= 16, "dfmt %u size", d);
    }
    // spot checks of the naming rule (last named = x at bit 0)
    CHECK(layoutOf(kFmt10_11_11).bits[0] == 11 && layoutOf(kFmt10_11_11).bits[2] == 10 && layoutOf(kFmt10_11_11).shift[2] == 22, "10_11_11");
    CHECK(layoutOf(kFmt11_11_10).bits[0] == 10 && layoutOf(kFmt11_11_10).shift[2] == 21, "11_11_10");
    CHECK(layoutOf(kFmt2_10_10_10).bits[3] == 2 && layoutOf(kFmt2_10_10_10).shift[3] == 30, "2_10_10_10");
    CHECK(layoutOf(kFmt10_10_10_2).bits[0] == 2 && layoutOf(kFmt10_10_10_2).shift[3] == 22, "10_10_10_2");
    CHECK(layoutOf(kFmt8_24).bits[0] == 24 && layoutOf(kFmt8_24).bits[1] == 8, "8_24");
    CHECK(layoutOf(kFmtX24_8_32).bits[0] == 32 && layoutOf(kFmtX24_8_32).shift[1] == 32 && layoutOf(kFmtX24_8_32).bytes == 8, "X24_8_32");
    CHECK(layoutOf(kFmt5_6_5).bits[1] == 6 && layoutOf(kFmt5_6_5).shift[2] == 11, "5_6_5");
    CHECK(layoutOf(kFmtFmask8_1).kind == FormatLayout::Fmask && layoutOf(kFmtFmask64_8).bytes == 8, "fmask");
}

void testKnownValues() {
    uint32_t w[4] = {0, 0, 0, 0}, o[4];
    // 8_8_8_8 unorm / snorm / srgb / uint
    w[0] = 0x80ff7f01u;
    CHECK(decodeTexel(kFmt8_8_8_8, kUnorm, w, 0, o) && bitsF(o[0]) == 1.0f / 255.0f && bitsF(o[1]) == 127.0f / 255.0f &&
              bitsF(o[2]) == 1.0f && bitsF(o[3]) == 128.0f / 255.0f, "8888 unorm");
    CHECK(decodeTexel(kFmt8_8_8_8, kSnorm, w, 0, o) && bitsF(o[2]) == -1.0f / 127.0f && bitsF(o[3]) == -1.0f && bitsF(o[1]) == 1.0f, "8888 snorm");
    CHECK(decodeTexel(kFmt8_8_8_8, kSint, w, 0, o) && int32_t(o[2]) == -1 && int32_t(o[3]) == -128, "8888 sint");
    CHECK(decodeTexel(kFmt8_8_8_8, kUscaled, w, 0, o) && bitsF(o[2]) == 255.0f, "8888 uscaled");
    CHECK(decodeTexel(kFmt8_8_8_8, kSscaled, w, 0, o) && bitsF(o[3]) == -128.0f, "8888 sscaled");
    // snorm_nz: -128 -> -255/255... (2c+1)/255
    CHECK(decodeTexel(kFmt8_8_8_8, kSnormNz, w, 0, o) && bitsF(o[3]) == -1.0f && bitsF(o[0]) == 3.0f / 255.0f, "8888 snorm_nz");
    // 10_11_11 float: 1.0 = 0x3c0 (6 mantissa bits, exponent 15), 10-bit 1.0 = 0x1e0
    w[0] = 0x3c0u | (0x3c0u << 11) | (0x1e0u << 22);
    CHECK(decodeTexel(kFmt10_11_11, kFloat, w, 0, o) && bitsF(o[0]) == 1.0f && bitsF(o[1]) == 1.0f && bitsF(o[2]) == 1.0f && bitsF(o[3]) == 1.0f, "10_11_11");
    // 5_9_9_9: exponent 15 + 9 -> m * 2^(e-24): e = 24 -> scale 1
    w[0] = (24u << 27) | (3u << 18) | (2u << 9) | 1u;
    CHECK(decodeTexel(kFmt5_9_9_9, kFloat, w, 0, o) && bitsF(o[0]) == 1.0f && bitsF(o[1]) == 2.0f && bitsF(o[2]) == 3.0f, "5_9_9_9");
    // 2_10_10_10 srgb: alpha stays linear
    w[0] = 0xc0000000u | (1023u << 20) | (1023u << 10) | 1023u;
    CHECK(decodeTexel(kFmt2_10_10_10, kSrgb, w, 0, o) && bitsF(o[0]) == 1.0f && bitsF(o[3]) == 1.0f, "2_10_10_10 srgb");
    // encode 2_10_10_10 srgb 0.5 linear
    uint32_t in[4] = {fBits(0.5f), fBits(0.5f), fBits(0.5f), fBits(1.0f)}, ew[4], em[4];
    CHECK(encodeTexel(kFmt2_10_10_10, kSrgb, in, 15, false, ew, em), "encode srgb10");
    CHECK((ew[0] & 1023) == 752 && (ew[0] >> 30) == 3, "srgb10 encode: %u", ew[0] & 1023);  // 0.7354 * 1023 = 752.3
    // X24_8_32: float depth + stencil
    w[0] = fBits(0.25f);
    w[1] = 0x7f;
    CHECK(decodeTexel(kFmtX24_8_32, kFloat, w, 0, o) && bitsF(o[0]) == 0.25f && o[1] == 0x7f, "X24_8_32 float");
    // 8_24 unorm depth
    w[0] = 0xff000000u | 0xffffffu;
    CHECK(decodeTexel(kFmt8_24, kUnorm, w, 0, o) && bitsF(o[0]) == 1.0f && bitsF(o[1]) == 1.0f, "8_24");
    // 1-bit
    w[0] = 0x05;
    CHECK(decodeTexel(kFmt1, kUnorm, w, 2, o) && bitsF(o[0]) == 1.0f && decodeTexel(kFmt1, kUnorm, w, 1, o) && bitsF(o[0]) == 0.0f, "1 bit");
    CHECK(decodeTexel(kFmt1Reversed, kUnorm, w, 5, o) && bitsF(o[0]) == 1.0f, "1 bit reversed");
}

// ---- emitters against the reference -----------------------------------------------------------------

struct Rng {
    std::mt19937 g{12345};
    uint32_t next() { return g(); }
};

V::Module decodeModule(unsigned dfmt, unsigned nfmt, bool& ok) {
    V::Module m;
    m.stage = V::Stage::Compute;
    V::Resource in;
    in.name = "in";
    in.kind = V::ResourceKind::Buffer;
    in.index = 0;
    in.access = V::Access::Read;
    V::Resource out = in;
    out.name = "out";
    out.index = 1;
    out.access = V::Access::Write;
    m.resources = {in, out};
    V::FunctionBuilder b(m, "main");
    std::vector<V::ValueId> words;
    for (unsigned k = 0; k < 4; ++k) words.push_back(b.load(0, b.constU32(4 * k), V::kU32));
    V::ValueId sub = b.load(0, b.constU32(16), V::kU32);
    FormatEmitter em(b);
    V::ValueId res[4];
    ok = em.decode(dfmt, nfmt, words, sub, res);
    if (ok)
        for (unsigned k = 0; k < 4; ++k) b.store(1, b.constU32(4 * k), res[k]);
    b.finish();
    return m;
}

V::Module encodeModule(unsigned dfmt, unsigned nfmt, unsigned have, bool saturate, FormatEmitter::Encoded& enc, bool& ok) {
    V::Module m;
    m.stage = V::Stage::Compute;
    V::Resource in;
    in.name = "in";
    in.kind = V::ResourceKind::Buffer;
    in.index = 0;
    in.access = V::Access::Read;
    V::Resource out = in;
    out.name = "out";
    out.index = 1;
    out.access = V::Access::Write;
    m.resources = {in, out};
    V::FunctionBuilder b(m, "main");
    std::vector<V::ValueId> comps;
    for (unsigned k = 0; k < 4; ++k) comps.push_back(b.load(0, b.constU32(4 * k), V::kU32));
    FormatEmitter em(b);
    ok = em.encode(dfmt, nfmt, comps, have, saturate, enc);
    if (ok)
        for (unsigned k = 0; k < enc.count; ++k) b.store(1, b.constU32(4 * k), enc.word[k]);
    b.finish();
    return m;
}

void putWords(vpireval::Invocation& inv, const uint32_t* w, unsigned n) {
    std::vector<uint8_t> bytes(n * 4);
    std::memcpy(bytes.data(), w, n * 4);
    inv.memory[0] = bytes;
    inv.memory[1] = std::vector<uint8_t>(16, 0xcd);
}

uint32_t fieldRandom(Rng& r, unsigned width, unsigned shape) {
    const uint32_t mask = width >= 32 ? ~0u : (1u << width) - 1;
    switch (shape) {
        case 0: return 0;
        case 1: return mask;
        case 2: return (1u << (width - 1)) & mask;           // most negative
        case 3: return ((1u << (width - 1)) - 1) & mask;     // largest positive
        default: return r.next() & mask;
    }
}

void testDecode() {
    Rng rng;
    unsigned combos = 0;
    for (unsigned dfmt = 1; dfmt < 64; ++dfmt) {
        const FormatLayout& l = layoutOf(dfmt);
        if (l.kind == FormatLayout::None || l.kind == FormatLayout::Block) continue;
        for (unsigned nfmt : kNumFmts) {
            bool ok = false;
            V::Module m = decodeModule(dfmt, nfmt, ok);
            uint32_t ref[4], probe[4] = {};
            const bool refOk = decodeTexel(dfmt, nfmt, probe, 0, ref);
            CHECK(ok == refOk, "decode support differs: dfmt %u nfmt %u (emit %d ref %d)", dfmt, nfmt, ok, refOk);
            if (!ok) continue;
            auto errs = V::verify(m);
            CHECK(errs.empty(), "verify dfmt %u nfmt %u: %s", dfmt, nfmt, errs.empty() ? "" : errs[0].c_str());
            if (!errs.empty()) continue;
            ++combos;
            const bool integer = isIntegerNumFmt(nfmt);
            for (int i = 0; i < 160; ++i) {
                uint32_t words[5] = {};
                if (l.kind == FormatLayout::Plain) {
                    for (unsigned c = 0; c < l.comps; ++c) {
                        const unsigned shape = i < 40 ? (i + c) % 5 : 4 + (i % 2);
                        const uint32_t v = fieldRandom(rng, l.bits[c], shape);
                        words[l.shift[c] / 32] |= v << (l.shift[c] % 32);
                    }
                    if (i >= 100)  // floats: random exponents
                        for (unsigned c = 0; c < l.comps; ++c)
                            if (l.bits[c] == 32 && c == 0) words[0] = rng.next();
                } else {
                    for (unsigned k = 0; k < 4; ++k) words[k] = rng.next();
                    if (i == 0) words[0] = words[1] = 0;
                    if (i == 1) words[0] = words[1] = ~0u;
                }
                words[4] = rng.next() & 7;
                uint32_t want[4];
                const bool wantOk = decodeTexel(dfmt, nfmt, words, words[4], want);
                if (!wantOk) continue;
                vpireval::Invocation inv;
                putWords(inv, words, 5);
                std::string err;
                if (!vpireval::run(m, inv, err)) {
                    CHECK(false, "eval dfmt %u nfmt %u: %s", dfmt, nfmt, err.c_str());
                    break;
                }
                uint32_t got[4];
                std::memcpy(got, inv.memory[1].data(), 16);
                for (unsigned c = 0; c < 4; ++c)
                    CHECK(sameValue(got[c], want[c], !integer),
                          "decode dfmt %u nfmt %u comp %u words %08x %08x %08x %08x: shader %08x ref %08x", dfmt, nfmt, c,
                          words[0], words[1], words[2], words[3], got[c], want[c]);
            }
        }
    }
    CHECK(combos > 300, "decode combos %u", combos);
    std::printf("decode: %u (dfmt,nfmt) combinations\n", combos);
}

uint32_t componentRandom(Rng& r, bool integer, unsigned i) {
    if (integer) {
        switch (i % 6) {
            case 0: return 0;
            case 1: return ~0u;
            case 2: return 0x80000000u;
            case 3: return 0x7fffffffu;
            case 4: return r.next() % 300;
            default: return r.next();
        }
    }
    static const float specials[] = {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 0.25f, 2.0f, -2.0f, 65504.0f, 65519.0f, 65520.0f, 70000.0f,
                                     1e-8f, 6.1e-5f, 5.96e-8f, 3.0e38f, -3.0e38f, 0.0031308f, 0.04045f, 255.0f, 1023.0f,
                                     INFINITY, -INFINITY, NAN, 0.99999f, 0.50001f, 0.4999f, 1.0e-5f};
    switch (i % 4) {
        case 0: return fBits(specials[r.next() % (sizeof specials / sizeof specials[0])]);
        case 1: return fBits(float(int(r.next() % 4001) - 2000) / 1000.0f);
        case 2: return fBits(std::ldexp(float(r.next() % 1000000) / 1000000.0f, int(r.next() % 30) - 20));
        default: return fBits(float(r.next() % 1025) / 1023.0f - 0.0004f);
    }
}

void testEncode() {
    Rng rng;
    unsigned combos = 0;
    for (unsigned dfmt = 1; dfmt < 64; ++dfmt) {
        const FormatLayout& l = layoutOf(dfmt);
        if (l.kind != FormatLayout::Plain) continue;
        for (unsigned nfmt : kNumFmts) {
            for (int variant = 0; variant < 3; ++variant) {
                const unsigned have = variant == 2 ? (0x5u & ((1u << l.comps) - 1)) : 15u;
                const bool saturate = variant == 1;
                if (!isIntegerNumFmt(nfmt) && saturate) continue;
                FormatEmitter::Encoded enc;
                bool ok = false;
                V::Module m = encodeModule(dfmt, nfmt, have, saturate, enc, ok);
                uint32_t rw[4], rm[4], probe[4] = {};
                const bool refOk = encodeTexel(dfmt, nfmt, probe, have, saturate, rw, rm);
                CHECK(ok == refOk, "encode support differs: dfmt %u nfmt %u", dfmt, nfmt);
                if (!ok) continue;
                auto errs = V::verify(m);
                CHECK(errs.empty(), "verify encode dfmt %u nfmt %u: %s", dfmt, nfmt, errs.empty() ? "" : errs[0].c_str());
                if (!errs.empty()) continue;
                ++combos;
                for (unsigned k = 0; k < enc.count; ++k) CHECK(enc.mask[k] == rm[k], "mask dfmt %u nfmt %u word %u", dfmt, nfmt, k);
                const bool integer = isIntegerNumFmt(nfmt);
                for (int i = 0; i < 160; ++i) {
                    uint32_t comps[4];
                    for (unsigned c = 0; c < 4; ++c) comps[c] = componentRandom(rng, integer, i + c);
                    uint32_t want[4], wm[4];
                    encodeTexel(dfmt, nfmt, comps, have, saturate, want, wm);
                    vpireval::Invocation inv;
                    putWords(inv, comps, 4);
                    std::string err;
                    if (!vpireval::run(m, inv, err)) {
                        CHECK(false, "eval encode dfmt %u nfmt %u: %s", dfmt, nfmt, err.c_str());
                        break;
                    }
                    uint32_t got[4] = {};
                    std::memcpy(got, inv.memory[1].data(), std::min<size_t>(16, 4 * enc.count));
                    for (unsigned k = 0; k < enc.count; ++k)
                        CHECK(got[k] == want[k],
                              "encode dfmt %u nfmt %u have %u sat %d in %08x %08x %08x %08x word %u: shader %08x ref %08x", dfmt, nfmt,
                              have, saturate, comps[0], comps[1], comps[2], comps[3], k, got[k], want[k]);
                }
            }
        }
    }
    std::printf("encode: %u (dfmt,nfmt,mask) combinations\n", combos);
}

// encode(decode(x)) is the identity for every representable field value of the exact formats.
void testRoundTrip() {
    Rng rng;
    unsigned checked = 0;
    for (unsigned dfmt = 1; dfmt < 64; ++dfmt) {
        const FormatLayout& l = layoutOf(dfmt);
        if (l.kind != FormatLayout::Plain) continue;
        for (unsigned nfmt : {unsigned(kUnorm), unsigned(kSnorm), unsigned(kUint), unsigned(kSint), unsigned(kFloat), unsigned(kSrgb)}) {
            for (int i = 0; i < 400; ++i) {
                uint32_t words[4] = {};
                for (unsigned c = 0; c < l.comps; ++c) {
                    const uint32_t v = fieldRandom(rng, l.bits[c], i < 20 ? i % 5 : 4);
                    words[l.shift[c] / 32] |= v << (l.shift[c] % 32);
                }
                uint32_t dec[4], enc[4], masks[4];
                if (!decodeTexel(dfmt, nfmt, words, 0, dec)) continue;
                bool skip = false;
                for (unsigned c = 0; c < l.comps; ++c) {
                    // snorm -128 and float NaN/inf patterns do not survive (clamped / canonicalised)
                    const unsigned w = l.bits[c];
                    const uint32_t raw = (words[l.shift[c] / 32] >> (l.shift[c] % 32)) & (w >= 32 ? ~0u : (1u << w) - 1);
                    if (nfmt == kSnorm && raw == (1u << (w - 1))) skip = true;
                    if (nfmt == kFloat && (w == 32 || w == 16 || w == 11 || w == 10) &&
                        (!std::isfinite(bitsF(dec[c])) || bitsF(dec[c]) == 0.0f)) skip = true;
                    if (nfmt == kSrgb && c < 3) skip = true;  // pow round trip is not exact (checked in testSrgb)
                    if ((nfmt == kUnorm || nfmt == kSnorm) && w == 32) skip = true;  // beyond f32 precision
                }
                if (skip) continue;
                if (!encodeTexel(dfmt, nfmt, dec, (1u << l.comps) - 1, false, enc, masks)) continue;
                ++checked;
                for (unsigned k = 0; k < l.words(); ++k)
                    CHECK((enc[k] & masks[k]) == (words[k] & masks[k]), "round trip dfmt %u nfmt %u word %u: %08x -> %08x", dfmt, nfmt, k, words[k], enc[k]);
            }
        }
    }
    CHECK(checked > 10000, "round trips %u", checked);
    std::printf("round trip: %u texels\n", checked);
}

}  // namespace

int main() {
    testHalf();
    testSmallFloat();
    testSrgb();
    testLayouts();
    testKnownValues();
    testRoundTrip();
    testDecode();
    testEncode();
    if (failures) {
        std::fprintf(stderr, "%u failures\n", failures);
        return 1;
    }
    std::printf("format tests passed\n");
    return 0;
}
