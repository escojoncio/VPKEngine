// Format conversions through the lifter: whole GCN programs (assembled with llvm-mc, see
// format_programs.inc) lifted with format information and run on the VPIR interpreter, against the
// CPU reference of vpgcn/formats.h. Typed buffer loads / stores for every dfmt x nfmt, colour
// export conversion (float / integer / packed containers, sRGB, channel select) and texture result
// conversion (affine, sRGB, select, containers decoded / encoded in the shader).
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "diff/vpir_eval.h"
#include "format_programs.inc"
#include "vpgcn/formats.h"
#include "vpgcn/lift.h"
#include "vpir/text.h"
#include "vpir/verify.h"

using namespace vpgcn;
namespace V = vpir;

namespace {

unsigned failures = 0;
#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            if (++failures < 40) {                                          \
                std::fprintf(stderr, "FAIL line %d: ", __LINE__);           \
                std::fprintf(stderr, __VA_ARGS__);                          \
                std::fprintf(stderr, "\n");                                 \
            }                                                               \
        }                                                                   \
    } while (0)

float bitsF(uint32_t b) { return std::bit_cast<float>(b); }
uint32_t fBits(float f) { return std::bit_cast<uint32_t>(f); }
bool sameValue(uint32_t a, uint32_t b, bool floatClass) {
    if (a == b) return true;
    if (!floatClass) return false;
    if (std::isnan(bitsF(a)) && std::isnan(bitsF(b))) return true;
    return (a | b) == 0x80000000u;
}

template <size_t N>
std::vector<uint32_t> prog(const uint32_t (&w)[N]) { return std::vector<uint32_t>(w, w + N); }

// Patch the dfmt / nfmt of every MTBUF instruction of a program.
std::vector<uint32_t> withFormat(std::vector<uint32_t> w, unsigned dfmt, unsigned nfmt) {
    for (uint32_t& x : w)
        if ((x >> 26) == 0x3a) x = (x & ~(0xfu << 19) & ~(7u << 23)) | (dfmt << 19) | (nfmt << 23);
    return w;
}

struct Rng {
    std::mt19937 g{777};
    uint32_t next() { return g(); }
};

uint32_t fieldRandom(Rng& r, unsigned width, unsigned shape) {
    const uint32_t mask = width >= 32 ? ~0u : (1u << width) - 1;
    switch (shape) {
        case 0: return 0;
        case 1: return mask;
        case 2: return (1u << (width - 1)) & mask;
        case 3: return ((1u << (width - 1)) - 1) & mask;
        default: return r.next() & mask;
    }
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
    static const float specials[] = {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 0.25f, 2.0f, -2.0f, 65504.0f, 65520.0f, 70000.0f, 1e-8f,
                                     6.1e-5f, 3.0e38f, -3.0e38f, 0.0031308f, 255.0f, 1023.0f, INFINITY, -INFINITY, NAN, 0.99999f};
    switch (i % 4) {
        case 0: return fBits(specials[r.next() % (sizeof specials / sizeof specials[0])]);
        case 1: return fBits(float(int(r.next() % 4001) - 2000) / 1000.0f);
        case 2: return fBits(std::ldexp(float(r.next() % 1000000) / 1000000.0f, int(r.next() % 30) - 20));
        default: return fBits(float(r.next() % 1025) / 1023.0f - 0.0004f);
    }
}

const unsigned kNumFmts[] = {kUnorm, kSnorm, kUscaled, kSscaled, kUint, kSint, kSnormNz, kFloat};

struct Lifted {
    LiftResult r;
    int bufferSlot(unsigned sgpr) const {
        for (const Binding& b : r.bindings)
            if (b.kind == V::ResourceKind::Buffer && b.source.kind == DescriptorSource::UserData && b.source.sgpr == sgpr) return int(b.slot);
        return -1;
    }
    int textureSlot() const {
        for (const Binding& b : r.bindings)
            if (b.kind == V::ResourceKind::Texture || b.kind == V::ResourceKind::StorageTexture) return int(b.slot);
        return -1;
    }
};

bool liftIt(Lifted& l, const std::vector<uint32_t>& code, ShaderStage stage, unsigned userSgprs, LiftInput base) {
    base.stage = stage;
    base.code = code;
    base.userSgprs = userSgprs;
    l.r = lift(base);
    for (const std::string& e : l.r.errors) std::fprintf(stderr, "lift error: %s\n", e.c_str());
    if (!l.r.errors.empty()) return false;
    auto errs = V::verify(l.r.module);
    for (const std::string& e : errs) std::fprintf(stderr, "verify: %s\n", e.c_str());
    return errs.empty();
}

bool liftIt(Lifted& l, const std::vector<uint32_t>& code, ShaderStage stage, unsigned userSgprs) {
    return liftIt(l, code, stage, userSgprs, LiftInput());
}

void userData(vpireval::Invocation& inv, const Lifted& l, const std::vector<uint32_t>& sgprs) {
    if (!l.r.userDataSlot) return;
    std::vector<uint8_t> bytes(sgprs.size() * 4);
    std::memcpy(bytes.data(), sgprs.data(), bytes.size());
    inv.memory[*l.r.userDataSlot] = bytes;
}

// ---- typed buffers ---------------------------------------------------------------------------------

void testTypedBuffers() {
    Rng rng;
    const std::vector<uint32_t> stores[4] = {prog(kStorex), prog(kStorexy), prog(kStorexyz), prog(kStorexyzw)};
    const std::vector<uint32_t> loads[4] = {prog(kLoadx), prog(kLoadxy), prog(kLoadxyz), prog(kLoadxyzw)};
    unsigned combos = 0;
    for (unsigned dfmt = 1; dfmt <= 14; ++dfmt) {
        const FormatLayout& l = layoutOf(dfmt);
        for (unsigned nfmt : kNumFmts) {
            const bool integer = isIntegerNumFmt(nfmt);
            for (unsigned n = 1; n <= 4; ++n) {
                // ---- store
                Lifted ls;
                if (!liftIt(ls, withFormat(stores[n - 1], dfmt, nfmt), ShaderStage::Compute, 8)) {
                    CHECK(false, "lift store dfmt %u nfmt %u n %u", dfmt, nfmt, n);
                    continue;
                }
                const int slot = ls.bufferSlot(4);
                CHECK(slot >= 0, "store buffer slot");
                if (slot < 0) continue;
                ++combos;
                for (int i = 0; i < 24; ++i) {
                    uint32_t comps[4];
                    for (unsigned c = 0; c < 4; ++c) comps[c] = componentRandom(rng, integer, i + c);
                    vpireval::Invocation inv;
                    // (the V# of s[4:7]: NUM_RECORDS is word 2, the range the lifted code checks)
                    userData(inv, ls, {comps[0], comps[1], comps[2], comps[3], 0, 0, 0x1000, 0});
                    inv.memory[slot] = std::vector<uint8_t>(32, 0xcd);
                    std::string err;
                    if (!vpireval::run(ls.r.module, inv, err)) {
                        CHECK(false, "eval store dfmt %u nfmt %u: %s", dfmt, nfmt, err.c_str());
                        break;
                    }
                    const unsigned cnt = std::min<unsigned>(l.comps, n);
                    uint32_t want[4], wm[4];
                    encodeTexel(dfmt, nfmt, comps, (1u << cnt) - 1, false, want, wm);
                    uint8_t expect[32];
                    std::memset(expect, 0xcd, sizeof expect);
                    for (unsigned k = 0; k < 4; ++k) {
                        uint32_t old;
                        std::memcpy(&old, expect + 4 * k, 4);
                        old = (old & ~wm[k]) | (want[k] & wm[k]);
                        std::memcpy(expect + 4 * k, &old, 4);
                    }
                    CHECK(std::memcmp(inv.memory[slot].data(), expect, 32) == 0,
                          "store dfmt %u nfmt %u n %u in %08x %08x %08x %08x: got %08x %08x %08x %08x want %08x %08x %08x %08x", dfmt, nfmt, n,
                          comps[0], comps[1], comps[2], comps[3], *(uint32_t*)&inv.memory[slot][0], *(uint32_t*)&inv.memory[slot][4],
                          *(uint32_t*)&inv.memory[slot][8], *(uint32_t*)&inv.memory[slot][12], *(uint32_t*)&expect[0],
                          *(uint32_t*)&expect[4], *(uint32_t*)&expect[8], *(uint32_t*)&expect[12]);
                }
                // ---- load
                Lifted ll;
                if (!liftIt(ll, withFormat(loads[n - 1], dfmt, nfmt), ShaderStage::Compute, 12)) {
                    CHECK(false, "lift load dfmt %u nfmt %u n %u", dfmt, nfmt, n);
                    continue;
                }
                const int inSlot = ll.bufferSlot(4), outSlot = ll.bufferSlot(8);
                CHECK(inSlot >= 0 && outSlot >= 0, "load slots");
                if (inSlot < 0 || outSlot < 0) continue;
                for (int i = 0; i < 24; ++i) {
                    uint32_t words[4] = {};
                    for (unsigned c = 0; c < l.comps; ++c) {
                        const uint32_t v = fieldRandom(rng, l.bits[c], i < 10 ? (i + c) % 5 : 4);
                        words[l.shift[c] / 32] |= v << (l.shift[c] % 32);
                    }
                    vpireval::Invocation inv;
                    std::vector<uint32_t> vsharp(12, 0);
                    vsharp[6] = vsharp[10] = 0x1000;  // NUM_RECORDS of the two V#s
                    userData(inv, ll, vsharp);
                    inv.memory[inSlot].assign(reinterpret_cast<uint8_t*>(words), reinterpret_cast<uint8_t*>(words) + 16);
                    inv.memory[outSlot] = std::vector<uint8_t>(16, 0);
                    std::string err;
                    if (!vpireval::run(ll.r.module, inv, err)) {
                        CHECK(false, "eval load dfmt %u nfmt %u: %s", dfmt, nfmt, err.c_str());
                        break;
                    }
                    uint32_t dec[4], got[4];
                    decodeTexel(dfmt, nfmt, words, 0, dec);
                    std::memcpy(got, inv.memory[outSlot].data(), 16);
                    for (unsigned c = 0; c < 4; ++c) {
                        const uint32_t want = c < n ? dec[c] : 0;
                        CHECK(sameValue(got[c], want, !integer), "load dfmt %u nfmt %u n %u comp %u words %08x %08x: got %08x want %08x", dfmt,
                              nfmt, n, c, words[0], words[1], got[c], want);
                    }
                }
            }
        }
    }
    std::printf("typed buffers: %u (dfmt,nfmt,n) store programs\n", combos);
}

// A typed load past the end of the records reads zeros, and one inside decodes the element, with the V#'s NUM_RECORDS at
// the element's size, one byte under it and zero. The load reads through a clamped address and selects the result (no
// `if` statement in the module), so the reads of the components can overlap.
void testTypedRange() {
    const std::vector<uint32_t> loads[4] = {prog(kLoadx), prog(kLoadxy), prog(kLoadxyz), prog(kLoadxyzw)};
    unsigned combos = 0;
    for (unsigned dfmt = 1; dfmt <= 14; ++dfmt) {
        const FormatLayout& l = layoutOf(dfmt);
        for (unsigned nfmt : {kUnorm, kUint, kFloat}) {
            for (unsigned n = 1; n <= 4; ++n) {
                Lifted ll;
                if (!liftIt(ll, withFormat(loads[n - 1], dfmt, nfmt), ShaderStage::Compute, 12)) {
                    CHECK(false, "lift load dfmt %u nfmt %u n %u", dfmt, nfmt, n);
                    continue;
                }
                const int inSlot = ll.bufferSlot(4), outSlot = ll.bufferSlot(8);
                if (inSlot < 0 || outSlot < 0) {
                    CHECK(false, "load slots");
                    continue;
                }
                // (the program's store of the four result dwords is four guarded stores: its `if` statements are the only ones)
                const std::string text = V::print(ll.r.module);
                unsigned ifs = 0;
                for (size_t at = text.find("if %"); at != std::string::npos; at = text.find("if %", at + 1)) ++ifs;
                CHECK(ifs <= 4, "dfmt %u nfmt %u n %u: %u `if` statements, the store's 4 alone wanted", dfmt, nfmt, n, ifs);
                ++combos;
                const unsigned bytes = std::max<unsigned>(1, l.bytes);
                for (unsigned num : {0u, bytes - 1, bytes, 0x1000u}) {
                    uint32_t words[4] = {~0u, ~0u, ~0u, ~0u};
                    vpireval::Invocation inv;
                    std::vector<uint32_t> vsharp(12, 0);
                    vsharp[6] = num, vsharp[10] = 0x1000;
                    userData(inv, ll, vsharp);
                    inv.memory[inSlot].assign(reinterpret_cast<uint8_t*>(words), reinterpret_cast<uint8_t*>(words) + 16);
                    inv.memory[outSlot] = std::vector<uint8_t>(16, 0xcd);
                    std::string err;
                    if (!vpireval::run(ll.r.module, inv, err)) {
                        CHECK(false, "eval load dfmt %u nfmt %u num %u: %s", dfmt, nfmt, num, err.c_str());
                        break;
                    }
                    uint32_t dec[4] = {}, got[4];
                    const bool inside = num >= bytes;
                    if (inside) decodeTexel(dfmt, nfmt, words, 0, dec);
                    std::memcpy(got, inv.memory[outSlot].data(), 16);
                    const bool integer = isIntegerNumFmt(nfmt);
                    for (unsigned c = 0; c < 4; ++c)
                        CHECK(sameValue(got[c], c < n ? dec[c] : 0, !integer), "load dfmt %u nfmt %u n %u num_records %u comp %u: got %08x want %08x", dfmt,
                              nfmt, n, num, c, got[c], c < n ? dec[c] : 0);
                    if (!inside) CHECK(got[0] == 0, "out of range load of dfmt %u nfmt %u gave %08x", dfmt, nfmt, got[0]);
                }
            }
        }
    }
    std::printf("typed buffer range: %u (dfmt,nfmt,n) load programs\n", combos);
}

// ---- colour exports --------------------------------------------------------------------------------

bool runExport(const Lifted& l, const uint32_t c[4], uint32_t out[4], V::Type& type) {
    vpireval::Invocation inv;
    userData(inv, l, {c[0], c[1], c[2], c[3]});
    std::string err;
    if (!vpireval::run(l.r.module, inv, err)) {
        std::fprintf(stderr, "eval export: %s\n", err.c_str());
        return false;
    }
    if (inv.outputs.empty()) return false;
    const vpireval::Value& v = inv.outputs.begin()->second;
    type = v.type;
    for (unsigned k = 0; k < 4; ++k) out[k] = v.bits[k];
    return true;
}

void testExports() {
    Rng rng;
    unsigned checked = 0;
    // identity: the float export passes through unchanged
    {
        ExportConv conv;
        Lifted l;
        LiftInput in;
        in.exportConv[0] = conv;
        CHECK(liftIt(l, prog(kExport), ShaderStage::Pixel, 4, in), "lift export identity");
        uint32_t c[4] = {fBits(0.1f), fBits(2.0f), fBits(-3.0f), fBits(0.5f)}, o[4];
        V::Type t;
        CHECK(runExport(l, c, o, t) && std::memcmp(o, c, 16) == 0 && t.scalar == V::Scalar::F32, "identity export");
    }
    // select + sRGB (float kind)
    {
        ExportConv conv;
        conv.srgb = true;
        conv.select[0] = 2, conv.select[1] = 1, conv.select[2] = 0, conv.select[3] = 5;
        Lifted l;
        LiftInput in;
        in.exportConv[0] = conv;
        CHECK(liftIt(l, prog(kExport), ShaderStage::Pixel, 4, in), "lift export srgb");
        for (int i = 0; i < 200; ++i) {
            uint32_t c[4], o[4];
            V::Type t;
            for (unsigned k = 0; k < 4; ++k) c[k] = componentRandom(rng, false, i + k);
            if (!runExport(l, c, o, t)) continue;
            auto enc = [&](uint32_t v) {
                float f = bitsF(v);
                f = std::isnan(f) ? 0.0f : std::fmin(std::fmax(f, 0.0f), 1.0f);
                return fBits(linearToSrgb(f));
            };
            const uint32_t want[4] = {enc(c[2]), enc(c[1]), enc(c[0]), fBits(1.0f)};
            for (unsigned k = 0; k < 4; ++k) CHECK(sameValue(o[k], want[k], true), "srgb export comp %u in %08x: %08x vs %08x", k, c[k], o[k], want[k]);
            ++checked;
        }
    }
    // integer kinds
    for (bool sgn : {false, true}) {
        ExportConv conv;
        conv.kind = sgn ? ExportConv::Sint : ExportConv::Uint;
        conv.bits[0] = 8, conv.bits[1] = 10, conv.bits[2] = 16, conv.bits[3] = 2;
        Lifted l;
        LiftInput in;
        in.exportConv[0] = conv;
        CHECK(liftIt(l, prog(kExport), ShaderStage::Pixel, 4, in), "lift export int");
        for (int i = 0; i < 200; ++i) {
            uint32_t c[4], o[4];
            V::Type t;
            for (unsigned k = 0; k < 4; ++k) c[k] = componentRandom(rng, true, i + k);
            if (!runExport(l, c, o, t)) continue;
            CHECK(t.scalar == (sgn ? V::Scalar::I32 : V::Scalar::U32), "int export type");
            for (unsigned k = 0; k < 4; ++k) {
                uint32_t want;
                if (sgn) {
                    const int w = conv.bits[k];
                    want = uint32_t(std::clamp<int64_t>(int32_t(c[k]), -(1ll << (w - 1)), (1ll << (w - 1)) - 1));
                } else {
                    want = std::min<uint32_t>(c[k], (1u << conv.bits[k]) - 1);
                }
                CHECK(o[k] == want, "int export comp %u in %08x: %08x vs %08x", k, c[k], o[k], want);
            }
            ++checked;
        }
    }
    // packed containers: every plain data format x number format, with a channel select
    for (unsigned dfmt = 1; dfmt < 64; ++dfmt) {
        const FormatLayout& fl = layoutOf(dfmt);
        if (fl.kind != FormatLayout::Plain) continue;
        for (unsigned nfmt : {unsigned(kUnorm), unsigned(kSnorm), unsigned(kUscaled), unsigned(kSscaled), unsigned(kUint), unsigned(kSint),
                              unsigned(kFloat), unsigned(kSrgb)}) {
            ExportConv conv;
            conv.kind = ExportConv::Packed;
            conv.dfmt = uint8_t(dfmt);
            conv.nfmt = uint8_t(nfmt);
            conv.select[0] = 3, conv.select[1] = 2, conv.select[2] = 1, conv.select[3] = 0;
            Lifted l;
            LiftInput in;
            in.exportConv[0] = conv;
            if (!liftIt(l, prog(kExport), ShaderStage::Pixel, 4, in)) {
                CHECK(false, "lift packed export %u/%u", dfmt, nfmt);
                continue;
            }
            const bool integer = isIntegerNumFmt(nfmt);
            for (int i = 0; i < 40; ++i) {
                uint32_t c[4], o[4];
                V::Type t;
                for (unsigned k = 0; k < 4; ++k) c[k] = componentRandom(rng, integer, i + k);
                if (!runExport(l, c, o, t)) continue;
                const uint32_t sel[4] = {c[3], c[2], c[1], c[0]};
                uint32_t want[4], masks[4];
                encodeTexel(dfmt, nfmt, sel, 15, true, want, masks);
                CHECK(t.scalar == V::Scalar::U32, "packed type");
                for (unsigned k = 0; k < 4; ++k) CHECK(o[k] == want[k], "packed export %u/%u word %u in %08x %08x %08x %08x: %08x vs %08x", dfmt, nfmt, k, c[0], c[1], c[2], c[3], o[k], want[k]);
                ++checked;
            }
        }
    }
    std::printf("exports: %u runs\n", checked);
}

// ---- textures ---------------------------------------------------------------------------------------

struct TexLog {
    std::vector<std::vector<uint32_t>> fetchCoords;   // each fetch's coordinate bits
    std::vector<uint32_t> fetchLods;
    uint32_t storedCoord[4] = {}, storedValue[4] = {};
    bool stored = false;
};

void testTextures() {
    Rng rng;
    unsigned checked = 0;
    // Native float texture result with affine / sRGB / select.
    {
        TextureConv conv;
        conv.affine = true;
        conv.scale[0] = 255.0f, conv.scale[1] = 2.0f * 127.0f / 255.0f, conv.scale[2] = 1.0f, conv.scale[3] = 3.0f;
        conv.bias[0] = 0.0f, conv.bias[1] = 1.0f / 255.0f, conv.bias[2] = 0.0f, conv.bias[3] = 0.0f;
        conv.srgb = true;
        conv.select[0] = 6, conv.select[1] = 5, conv.select[2] = 1, conv.select[3] = 0;
        Lifted l;
        LiftInput in;
        in.describe = [&](const DescriptorSource&) -> std::optional<DescriptorInfo> {
            DescriptorInfo d;
            d.conv = conv;
            return d;
        };
        // the pointer table of the program: descriptors come from memory at s[0:1]; the lifter needs no values
        CHECK(liftIt(l, prog(kSample), ShaderStage::Pixel, 2, in), "lift native sample");
        for (int i = 0; i < 100; ++i) {
            uint32_t texel[4];
            for (unsigned k = 0; k < 4; ++k) texel[k] = fBits(float(rng.next() % 1001) / 1000.0f);
            vpireval::Invocation inv;
            userData(inv, l, {0, 0});
            inv.image = [&](const V::Inst& ins, const std::vector<vpireval::Value>&, vpireval::Value& out) {
                if (ins.op != V::Op::ImageSample) return false;
                for (unsigned k = 0; k < 4; ++k) out.bits[k] = texel[k];
                return true;
            };
            std::string err;
            if (!vpireval::run(l.r.module, inv, err)) {
                CHECK(false, "eval native sample: %s", err.c_str());
                break;
            }
            auto after = [&](unsigned k) {
                float v = bitsF(texel[k]) * conv.scale[k] + conv.bias[k];
                if (k < 3) v = srgbToLinear(v);
                return fBits(v);
            };
            const uint32_t want[4] = {after(2), after(1), fBits(1.0f), fBits(0.0f)};
            const vpireval::Value& o = inv.outputs.begin()->second;
            for (unsigned k = 0; k < 4; ++k) CHECK(sameValue(o.bits[k], want[k], true), "native sample comp %u: %08x vs %08x", k, o.bits[k], want[k]);
            ++checked;
        }
    }
    // Native gather: channel y (dmask 2) through select
    {
        TextureConv conv;
        conv.select[1] = 7;  // dmask picks hw channel y -> source w
        conv.affine = true;
        conv.scale[3] = 10.0f;
        Lifted l;
        LiftInput in;
        in.describe = [&](const DescriptorSource&) -> std::optional<DescriptorInfo> {
            DescriptorInfo d;
            d.conv = conv;
            return d;
        };
        CHECK(liftIt(l, prog(kGather), ShaderStage::Pixel, 2, in), "lift gather");
        vpireval::Invocation inv;
        userData(inv, l, {0, 0});
        unsigned gatheredComp = 99;
        inv.image = [&](const V::Inst& ins, const std::vector<vpireval::Value>&, vpireval::Value& out) {
            if (ins.op != V::Op::ImageGather) return false;
            gatheredComp = ins.lits.empty() ? 99 : ins.lits[0];
            for (unsigned k = 0; k < 4; ++k) out.bits[k] = fBits(float(k + 1));
            return true;
        };
        std::string err;
        CHECK(vpireval::run(l.r.module, inv, err), "eval gather %s", err.c_str());
        CHECK(gatheredComp == 3, "gather reads source channel w, got %u", gatheredComp);
        const vpireval::Value& o = inv.outputs.begin()->second;
        for (unsigned k = 0; k < 4; ++k) CHECK(bitsF(o.bits[k]) == float(k + 1) * 10.0f, "gather comp %u = %g", k, bitsF(o.bits[k]));
    }
    // Containers decoded in the shader: sample, load, store.
    for (unsigned dfmt = 1; dfmt < 64; ++dfmt) {
        const FormatLayout& fl = layoutOf(dfmt);
        if (fl.kind == FormatLayout::None || fl.kind == FormatLayout::Block) continue;
        for (unsigned nfmt : {unsigned(kUnorm), unsigned(kSnorm), unsigned(kUscaled), unsigned(kSscaled), unsigned(kUint), unsigned(kSint),
                              unsigned(kSnormNz), unsigned(kFloat), unsigned(kSrgb)}) {
            uint32_t probe[4] = {}, tmp[4];
            if (!decodeTexel(dfmt, nfmt, probe, 0, tmp)) continue;
            const bool integer = isIntegerNumFmt(nfmt);
            TextureConv conv;
            conv.manual = true;
            conv.texelsPerElement = fl.texelsPerElement;
            conv.select[0] = 6, conv.select[1] = 5, conv.select[2] = 4, conv.select[3] = 1;  // bgr1
            LiftInput in;
            in.describe = [&](const DescriptorSource&) -> std::optional<DescriptorInfo> {
                DescriptorInfo d;
                d.conv = conv;
                d.dfmt = uint8_t(dfmt);
                d.nfmt = uint8_t(nfmt);
                d.elem = V::Scalar::U32;
                return d;
            };
            const unsigned tpe = fl.texelsPerElement;
            // sample
            Lifted ls;
            if (!liftIt(ls, prog(kSample), ShaderStage::Pixel, 2, in)) {
                CHECK(false, "lift manual sample %u/%u", dfmt, nfmt);
                continue;
            }
            for (int i = 0; i < 12; ++i) {
                uint32_t words[4];
                for (uint32_t& w : words) w = rng.next();
                TexLog log;
                vpireval::Invocation inv;
                userData(inv, ls, {0, 0});
                inv.image = [&](const V::Inst& ins, const std::vector<vpireval::Value>& a, vpireval::Value& out) {
                    if (ins.op == V::Op::ImageSize) {
                        out.bits[0] = 8 / tpe == 0 ? 1 : 8 / tpe;  // container width: 8 texels
                        out.bits[1] = 4;
                        return true;
                    }
                    if (ins.op == V::Op::ImageFetch) {
                        log.fetchCoords.push_back({a[0].bits[0], a[0].bits[1]});
                        for (unsigned k = 0; k < 4; ++k) out.bits[k] = words[k];
                        return true;
                    }
                    return false;
                };
                std::string err;
                if (!vpireval::run(ls.r.module, inv, err)) {
                    CHECK(false, "eval manual sample %u/%u: %s", dfmt, nfmt, err.c_str());
                    break;
                }
                // coordinates (0.25, 0.75) over 8 x 4 texels: texel (2, 3)
                const unsigned sub = 2 % tpe;
                CHECK(log.fetchCoords.size() == 1 && log.fetchCoords[0][0] == 2 / tpe && log.fetchCoords[0][1] == 3,
                      "manual sample %u/%u fetched (%u,%u)", dfmt, nfmt, log.fetchCoords.empty() ? 99 : log.fetchCoords[0][0],
                      log.fetchCoords.empty() ? 99 : log.fetchCoords[0][1]);
                uint32_t dec[4];
                decodeTexel(dfmt, nfmt, words, sub, dec);
                const uint32_t one = integer ? 1u : fBits(1.0f);
                const uint32_t want[4] = {dec[2], dec[1], dec[0], one};
                const vpireval::Value& o = inv.outputs.begin()->second;
                for (unsigned k = 0; k < 4; ++k) CHECK(sameValue(o.bits[k], want[k], !integer), "manual sample %u/%u comp %u: %08x vs %08x", dfmt, nfmt, k, o.bits[k], want[k]);
                ++checked;
            }
            // load
            Lifted ll;
            if (!liftIt(ll, prog(kLoadImg), ShaderStage::Pixel, 2, in)) {
                CHECK(false, "lift manual load %u/%u", dfmt, nfmt);
                continue;
            }
            {
                uint32_t words[4];
                for (uint32_t& w : words) w = rng.next();
                TexLog log;
                vpireval::Invocation inv;
                userData(inv, ll, {0, 0});
                inv.image = [&](const V::Inst& ins, const std::vector<vpireval::Value>& a, vpireval::Value& out) {
                    if (ins.op != V::Op::ImageFetch) return false;
                    log.fetchCoords.push_back({a[0].bits[0], a[0].bits[1]});
                    for (unsigned k = 0; k < 4; ++k) out.bits[k] = words[k];
                    return true;
                };
                std::string err;
                if (vpireval::run(ll.r.module, inv, err)) {
                    CHECK(log.fetchCoords.size() == 1 && log.fetchCoords[0][0] == 3 / tpe && log.fetchCoords[0][1] == 5, "manual load coord %u/%u", dfmt, nfmt);
                    uint32_t dec[4];
                    decodeTexel(dfmt, nfmt, words, 3 % tpe, dec);
                    const uint32_t one = integer ? 1u : fBits(1.0f);
                    const uint32_t want[4] = {dec[2], dec[1], dec[0], one};
                    const vpireval::Value& o = inv.outputs.begin()->second;
                    for (unsigned k = 0; k < 4; ++k) CHECK(sameValue(o.bits[k], want[k], !integer), "manual load %u/%u comp %u", dfmt, nfmt, k);
                    ++checked;
                } else {
                    CHECK(false, "eval manual load %u/%u: %s", dfmt, nfmt, err.c_str());
                }
            }
            // store (Plain formats; the select does not apply to stores)
            if (fl.kind == FormatLayout::Plain) {
                Lifted lst;
                if (!liftIt(lst, prog(kStoreImg), ShaderStage::Pixel, 8, in)) {
                    CHECK(false, "lift manual store %u/%u", dfmt, nfmt);
                    continue;
                }
                for (int i = 0; i < 6; ++i) {
                    uint32_t comps[4];
                    for (unsigned k = 0; k < 4; ++k) comps[k] = componentRandom(rng, integer, i + k);
                    TexLog log;
                    vpireval::Invocation inv;
                    userData(inv, lst, {0, 0, comps[0], comps[1], comps[2], comps[3], 0, 0});
                    inv.image = [&](const V::Inst& ins, const std::vector<vpireval::Value>& a, vpireval::Value& out) {
                        (void)out;
                        if (ins.op != V::Op::ImageStore) return false;
                        log.stored = true;
                        for (unsigned k = 0; k < 2; ++k) log.storedCoord[k] = a[0].bits[k];
                        for (unsigned k = 0; k < 4; ++k) log.storedValue[k] = a[1].bits[k];
                        return true;
                    };
                    std::string err;
                    if (!vpireval::run(lst.r.module, inv, err)) {
                        CHECK(false, "eval manual store %u/%u: %s", dfmt, nfmt, err.c_str());
                        break;
                    }
                    uint32_t want[4], masks[4];
                    encodeTexel(dfmt, nfmt, comps, 15, false, want, masks);
                    CHECK(log.stored && log.storedCoord[0] == 3 && log.storedCoord[1] == 5, "manual store coord");
                    for (unsigned k = 0; k < 4; ++k) CHECK(log.storedValue[k] == (k < fl.words() ? want[k] : 0), "manual store %u/%u word %u in %08x %08x %08x %08x: %08x vs %08x", dfmt, nfmt, k, comps[0], comps[1], comps[2], comps[3], log.storedValue[k], want[k]);
                    ++checked;
                }
            }
        }
    }
    std::printf("textures: %u runs\n", checked);
}

}  // namespace

int main() {
    testTypedBuffers();
    testTypedRange();
    testExports();
    testTextures();
    if (failures) {
        std::fprintf(stderr, "%u failures\n", failures);
        return 1;
    }
    std::printf("format lift tests passed\n");
    return 0;
}
