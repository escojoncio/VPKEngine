// FRAMECAP (probe diagnostics, renderer.cpp): what an image read back from the GPU holds, and a picture of it.
// imageStats: pixels whose colour channels are exactly zero (alpha left out), aligned 8x8 / 32x32 tiles that are all zero
// (the "black blocks" of a tile that nothing wrote, or a block compressed / detiled to zero), NaN / Inf texels, and the
// finite range of a one-channel image (depth). writePpm: a binary PPM, `step` texels to a pixel (nearest: a zero tile
// stays zero); floats above 1 are compressed (x / (1 + x)), linear floats get a square root (a rough gamma), one-channel
// images are stretched over their finite range, NaN / Inf are magenta. Formats it cannot decode: stats on raw bytes, no picture.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "vpgpu/core/resources.h"

namespace vpps4::framecap {

inline float half(uint16_t h) {
    const uint32_t e = (h >> 10) & 31, m = h & 1023;
    const float s = (h & 0x8000) ? -1.0f : 1.0f;
    if (e == 31) return m ? NAN : s * INFINITY;
    return s * (e ? std::ldexp(float(m | 1024), int(e) - 25) : std::ldexp(float(m), -24));
}
inline float small(uint32_t v, unsigned mbits) {  // the unsigned 11 / 10 bit floats of RG11B10 (5 exponent bits)
    const uint32_t e = v >> mbits, m = v & ((1u << mbits) - 1);
    if (e == 31) return m ? NAN : INFINITY;
    return e ? std::ldexp(float(m | (1u << mbits)), int(e) - 15 - int(mbits)) : std::ldexp(float(m), -14 - int(mbits));
}

// The channels of one texel as floats: count (0: not decoded), whether a channel is the alpha, linear float data.
struct Texel {
    float c[4] = {};
    unsigned n = 0;
    bool alpha = false, floats = false;
};
inline Texel decode(const uint8_t* p, vpgpu::Format f) {
    using F = vpgpu::Format;
    Texel t;
    auto u8 = [&](unsigned i) { return p[i] / 255.0f; };
    auto u16 = [&](unsigned i) { uint16_t v; std::memcpy(&v, p + 2 * i, 2); return v; };
    auto f32 = [&](unsigned i) { float v; std::memcpy(&v, p + 4 * i, 4); return v; };
    switch (f) {
    case F::RGBA8Unorm: case F::RGBA8Srgb: t = {{u8(0), u8(1), u8(2), u8(3)}, 4, true, false}; break;
    case F::BGRA8Unorm: case F::BGRA8Srgb: t = {{u8(2), u8(1), u8(0), u8(3)}, 4, true, false}; break;
    case F::R8Unorm: t = {{u8(0)}, 1, false, false}; break;
    case F::RG8Unorm: t = {{u8(0), u8(1)}, 2, false, false}; break;
    case F::R16Unorm: case F::D16Unorm: t = {{u16(0) / 65535.0f}, 1, false, false}; break;
    case F::R16Float: t = {{half(u16(0))}, 1, false, true}; break;
    case F::RG16Float: t = {{half(u16(0)), half(u16(1))}, 2, false, true}; break;
    case F::RGBA16Float: t = {{half(u16(0)), half(u16(1)), half(u16(2)), half(u16(3))}, 4, true, true}; break;
    case F::R32Float: case F::D32Float: t = {{f32(0)}, 1, false, true}; break;
    case F::RG32Float: t = {{f32(0), f32(1)}, 2, false, true}; break;
    case F::RGBA32Float: t = {{f32(0), f32(1), f32(2), f32(3)}, 4, true, true}; break;
    case F::RG11B10Float: {
        uint32_t v;
        std::memcpy(&v, p, 4);
        t = {{small(v & 2047, 6), small((v >> 11) & 2047, 6), small(v >> 22, 5)}, 3, false, true};
        break;
    }
    case F::RGB10A2Unorm: {
        uint32_t v;
        std::memcpy(&v, p, 4);
        t = {{(v & 1023) / 1023.0f, ((v >> 10) & 1023) / 1023.0f, ((v >> 20) & 1023) / 1023.0f, (v >> 30) / 3.0f}, 4, true, false};
        break;
    }
    default: break;
    }
    return t;
}

struct Stats {
    uint64_t pixels = 0, zero = 0, nan = 0, tiles8 = 0, zero8 = 0, tiles32 = 0, zero32 = 0;
    float lo = INFINITY, hi = -INFINITY;  // finite range of channel 0 (one-channel images: the depth range)
    bool decoded = false;
};
// `data`: `w` x `h` texels of `bpe` bytes, rows packed (as Backend::readTexture gives them).
inline Stats imageStats(const uint8_t* data, uint32_t w, uint32_t h, uint32_t bpe, vpgpu::Format f) {
    Stats s;
    if (!w || !h || !bpe) return s;
    s.decoded = decode(data, f).n != 0;
    const uint32_t tw8 = w / 8, th8 = h / 8, tw32 = w / 32, th32 = h / 32;
    std::vector<uint8_t> nz8(size_t(tw8) * th8), nz32(size_t(tw32) * th32);  // tiles with a texel that is not zero
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const uint8_t* p = data + (uint64_t(y) * w + x) * bpe;
            bool zero = true;
            if (s.decoded) {
                const Texel t = decode(p, f);
                const unsigned colour = t.alpha ? t.n - 1 : t.n;
                bool bad = false;
                for (unsigned c = 0; c < t.n; ++c) bad = bad || !std::isfinite(t.c[c]);
                s.nan += bad;
                for (unsigned c = 0; c < colour; ++c) zero = zero && t.c[c] == 0.0f;
                if (std::isfinite(t.c[0])) s.lo = std::min(s.lo, t.c[0]), s.hi = std::max(s.hi, t.c[0]);
            } else {
                for (uint32_t b = 0; b < bpe; ++b) zero = zero && !p[b];
            }
            s.zero += zero;
            if (!zero) {
                if (x / 8 < tw8 && y / 8 < th8) nz8[size_t(y / 8) * tw8 + x / 8] = 1;
                if (x / 32 < tw32 && y / 32 < th32) nz32[size_t(y / 32) * tw32 + x / 32] = 1;
            }
        }
    s.pixels = uint64_t(w) * h;
    s.tiles8 = nz8.size(), s.tiles32 = nz32.size();
    for (uint8_t v : nz8) s.zero8 += !v;
    for (uint8_t v : nz32) s.zero32 += !v;
    return s;
}
inline uint64_t permille(uint64_t part, uint64_t whole) { return whole ? part * 1000 / whole : 0; }

// The picture (see the file comment); false if the format is not decoded or the file cannot be written. `bytes`: its size.
inline bool writePpm(const std::string& path, const uint8_t* data, uint32_t w, uint32_t h, uint32_t bpe, vpgpu::Format f,
                     uint32_t step, const Stats& s, uint64_t& bytes) {
    bytes = 0;
    if (!s.decoded || !w || !h || !step) return false;
    const uint32_t ow = (w + step - 1) / step, oh = (h + step - 1) / step;
    std::string out = "P6\n" + std::to_string(ow) + " " + std::to_string(oh) + "\n255\n";
    const size_t head = out.size();
    out.resize(head + size_t(ow) * oh * 3);
    const float range = s.hi > s.lo ? s.hi - s.lo : 1.0f;
    uint8_t* o = reinterpret_cast<uint8_t*>(out.data()) + head;
    for (uint32_t y = 0; y < h; y += step)
        for (uint32_t x = 0; x < w; x += step, o += 3) {
            const Texel t = decode(data + (uint64_t(y) * w + x) * bpe, f);
            float v[3] = {};
            bool bad = false;
            for (unsigned c = 0; c < 3; ++c) {
                const float in = t.n == 1 ? t.c[0] : c < t.n ? t.c[c] : 0.0f;
                if (!std::isfinite(in)) bad = true;
                float x01 = t.n == 1 ? (in - s.lo) / range : std::max(in, 0.0f);
                if (t.n != 1 && t.floats) x01 = std::sqrt(x01 / (1.0f + x01));
                v[c] = std::min(std::max(x01, 0.0f), 1.0f);
            }
            if (bad) v[0] = 1, v[1] = 0, v[2] = 1;
            for (unsigned c = 0; c < 3; ++c) o[c] = uint8_t(v[c] * 255.0f + 0.5f);
        }
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;
    const bool ok = std::fwrite(out.data(), 1, out.size(), fp) == out.size();
    std::fclose(fp);
    if (ok) bytes = out.size();
    return ok;
}

}  // namespace vpps4::framecap
