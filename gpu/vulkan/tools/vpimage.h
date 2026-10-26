// Images of the replay tools: a minimal PNG encoder / decoder (zlib is the only dependency), the conversion of the
// Vulkan formats vpreplay dumps to RGBA samples, and the metrics vpimgdiff compares two images with.
#pragma once

#include <vulkan/vulkan_core.h>
#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace vpimg {

// RGBA, `bits` (8 or 16) per sample; px holds the sample values (0..255 or 0..65535), w * h * 4 of them.
struct Image {
  uint32_t w = 0, h = 0, bits = 8;
  std::vector<uint16_t> px;
  uint32_t maxValue() const { return bits == 16 ? 65535u : 255u; }
};

// ---- PNG ----
inline void putBE32(std::vector<uint8_t>& v, uint32_t x) {
  for (int s = 24; s >= 0; s -= 8) v.push_back(uint8_t(x >> s));
}
inline void pngChunk(std::vector<uint8_t>& out, const char type[4], const std::vector<uint8_t>& data) {
  putBE32(out, uint32_t(data.size()));
  const size_t at = out.size();
  out.insert(out.end(), type, type + 4);
  out.insert(out.end(), data.begin(), data.end());
  uLong crc = crc32(0, out.data() + at, uInt(4 + data.size()));
  putBE32(out, uint32_t(crc));
}

// 8 or 16 bit RGBA, no interlace, filter type 0 on every row.
inline std::vector<uint8_t> pngEncode(const Image& im) {
  std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  std::vector<uint8_t> ihdr;
  putBE32(ihdr, im.w), putBE32(ihdr, im.h);
  ihdr.push_back(uint8_t(im.bits)), ihdr.push_back(6), ihdr.push_back(0), ihdr.push_back(0), ihdr.push_back(0);
  pngChunk(out, "IHDR", ihdr);
  const size_t bps = im.bits == 16 ? 2 : 1, row = size_t(im.w) * 4 * bps;
  std::vector<uint8_t> raw(size_t(im.h) * (row + 1));
  for (uint32_t y = 0; y < im.h; ++y) {
    uint8_t* d = raw.data() + size_t(y) * (row + 1);
    *d++ = 0;
    const uint16_t* s = im.px.data() + size_t(y) * im.w * 4;
    for (size_t i = 0; i < size_t(im.w) * 4; ++i) {
      if (bps == 2) *d++ = uint8_t(s[i] >> 8);
      *d++ = uint8_t(s[i]);
    }
  }
  uLongf zn = compressBound(uLong(raw.size()));
  std::vector<uint8_t> z(zn);
  compress2(z.data(), &zn, raw.data(), uLong(raw.size()), 6);
  z.resize(zn);
  pngChunk(out, "IDAT", z);
  pngChunk(out, "IEND", {});
  return out;
}

inline uint32_t getBE32(const uint8_t* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }

// Gray / gray+alpha / RGB / RGBA, 8 or 16 bit, no interlace, every row filter; the result is RGBA.
inline bool pngDecode(const std::vector<uint8_t>& f, Image& im, std::string* err = nullptr) {
  auto fail = [&](const char* why) {
    if (err) *err = why;
    return false;
  };
  static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  if (f.size() < 8 || std::memcmp(f.data(), sig, 8)) return fail("not a PNG");
  uint32_t w = 0, h = 0, depth = 0, type = 0;
  std::vector<uint8_t> idat;
  for (size_t p = 8; p + 12 <= f.size();) {
    const uint32_t n = getBE32(&f[p]);
    if (p + 12 + n > f.size()) return fail("truncated chunk");
    const uint8_t* d = &f[p + 8];
    if (!std::memcmp(&f[p + 4], "IHDR", 4) && n >= 13) {
      w = getBE32(d), h = getBE32(d + 4), depth = d[8], type = d[9];
      if (d[12] != 0) return fail("interlaced PNG");
    } else if (!std::memcmp(&f[p + 4], "IDAT", 4)) {
      idat.insert(idat.end(), d, d + n);
    } else if (!std::memcmp(&f[p + 4], "IEND", 4)) {
      break;
    }
    p += 12 + n;
  }
  if (!w || !h || (depth != 8 && depth != 16)) return fail("unsupported PNG bit depth");
  const uint32_t ch = type == 0 ? 1 : type == 2 ? 3 : type == 4 ? 2 : type == 6 ? 4 : 0;
  if (!ch) return fail("unsupported PNG color type");
  const size_t bps = depth / 8, bpp = ch * bps, row = size_t(w) * bpp;
  std::vector<uint8_t> raw(size_t(h) * (row + 1));
  uLongf rn = uLongf(raw.size());
  if (uncompress(raw.data(), &rn, idat.data(), uLong(idat.size())) != Z_OK || rn != raw.size()) return fail("bad PNG data");
  std::vector<uint8_t> prev(row, 0), cur(row);
  im.w = w, im.h = h, im.bits = depth;
  im.px.assign(size_t(w) * h * 4, uint16_t(depth == 16 ? 65535 : 255));
  for (uint32_t y = 0; y < h; ++y) {
    const uint8_t* s = &raw[size_t(y) * (row + 1)];
    const uint8_t ft = *s++;
    for (size_t i = 0; i < row; ++i) {
      const int a = i >= bpp ? cur[i - bpp] : 0, b = prev[i], c = i >= bpp ? prev[i - bpp] : 0;
      int v = s[i];
      switch (ft) {
        case 0: break;
        case 1: v += a; break;
        case 2: v += b; break;
        case 3: v += (a + b) / 2; break;
        case 4: {
          const int pp = a + b - c, pa = std::abs(pp - a), pb = std::abs(pp - b), pc = std::abs(pp - c);
          v += (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
          break;
        }
        default: return fail("bad PNG filter");
      }
      cur[i] = uint8_t(v);
    }
    uint16_t* d = &im.px[size_t(y) * w * 4];
    for (uint32_t x = 0; x < w; ++x) {
      uint16_t t[4] = {0, 0, 0, 0}, v[4] = {0, 0, 0, uint16_t(depth == 16 ? 65535 : 255)};
      for (uint32_t k = 0; k < ch; ++k) {
        const uint8_t* q = &cur[size_t(x) * bpp + k * bps];
        t[k] = bps == 2 ? uint16_t(q[0] << 8 | q[1]) : q[0];
      }
      if (ch <= 2) v[0] = v[1] = v[2] = t[0];  // gray (+ alpha)
      else v[0] = t[0], v[1] = t[1], v[2] = t[2];
      if (ch == 2 || ch == 4) v[3] = t[ch - 1];
      std::memcpy(d + size_t(x) * 4, v, 8);
    }
    prev = cur;
  }
  return true;
}

// ---- Vulkan formats -> RGBA ----
inline float halfToFloat(uint16_t h) {
  const uint32_t s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
  float v;
  if (e == 0) v = std::ldexp(float(m), -24);
  else if (e == 31) v = m ? NAN : INFINITY;
  else v = std::ldexp(float(m | 1024), int(e) - 25);
  return s ? -v : v;
}
// An unsigned 5-bit exponent float with `mbits` mantissa bits (R11G11B10: 6 for R and G, 5 for B).
inline float smallFloat(uint32_t bitsIn, uint32_t mbits) {
  const uint32_t e = bitsIn >> mbits, m = bitsIn & ((1u << mbits) - 1);
  if (e == 0) return std::ldexp(float(m), -14 - int(mbits));
  if (e == 31) return m ? NAN : INFINITY;
  return std::ldexp(float(m | (1u << mbits)), int(e) - 15 - int(mbits));
}
inline uint16_t unorm16(float v) {
  if (!(v > 0)) return 0;  // also NaN
  return v >= 1 ? 65535 : uint16_t(v * 65535.0f + 0.5f);
}

inline uint32_t texelBytes(VkFormat f) {
  switch (f) {
    case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_SNORM: case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT: case VK_FORMAT_S8_UINT: return 1;
    case VK_FORMAT_D16_UNORM: case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R8G8_SNORM: case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16_SNORM: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R5G6B5_UNORM_PACK16:
    case VK_FORMAT_R4G4B4A4_UNORM_PACK16: return 2;
    case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R16G16_SNORM: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R8G8B8A8_SNORM: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32: case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: return 4;
    case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R16G16B16A16_SNORM:
    case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32_SINT: return 8;
    case VK_FORMAT_R32G32B32_SFLOAT: case VK_FORMAT_R32G32B32_UINT: return 12;
    case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_UINT: return 16;
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_A8B8G8R8_UNORM_PACK32: case VK_FORMAT_A8B8G8R8_SRGB_PACK32: case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_D24_UNORM_S8_UINT: case VK_FORMAT_D32_SFLOAT_S8_UINT: return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
    default: return 0;
  }
}

// The aspect a copy of the format's depth / color aspect to a buffer produces: bytes per texel, 0 = not supported.
inline bool isDepthFormat(VkFormat f) {
  return f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_D32_SFLOAT || f == VK_FORMAT_X8_D24_UNORM_PACK32 ||
         f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT;
}
// Bytes per texel of the DEPTH aspect copied to a buffer (the stencil of a combined format is not copied).
inline uint32_t depthCopyBytes(VkFormat f) {
  switch (f) {
    case VK_FORMAT_D16_UNORM: return 2;
    case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_X8_D24_UNORM_PACK32: case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT: return 4;
    default: return 0;
  }
}

// Color or depth data (`depth`: the depth aspect copied from `fmt`) as an RGBA image. False: format not supported (the
// caller writes the bytes raw). Depth is the value itself in 0..1 as 16-bit gray (not rescaled: two dumps stay comparable);
// float colors are clamped to 0..1 and stored as 16 bit.
inline bool convert(VkFormat fmt, bool depth, const uint8_t* d, uint32_t w, uint32_t h, Image& out) {
  out.w = w, out.h = h, out.bits = 8;
  const size_t n = size_t(w) * h;
  out.px.assign(n * 4, 0);
  auto rd32 = [&](size_t i) {
    uint32_t v;
    std::memcpy(&v, d + i * 4, 4);
    return v;
  };
  if (depth) {
    if (!depthCopyBytes(fmt)) return false;
    out.bits = 16;
    for (size_t i = 0; i < n; ++i) {
      uint16_t g;
      if (fmt == VK_FORMAT_D16_UNORM) {
        std::memcpy(&g, d + i * 2, 2);
      } else if (fmt == VK_FORMAT_D32_SFLOAT || fmt == VK_FORMAT_D32_SFLOAT_S8_UINT) {
        float f;
        std::memcpy(&f, d + i * 4, 4);
        g = unorm16(f);
      } else {
        g = uint16_t((uint64_t(rd32(i) & 0xFFFFFF) * 65535 + 8388607) / 16777215);
      }
      out.px[i * 4] = out.px[i * 4 + 1] = out.px[i * 4 + 2] = g, out.px[i * 4 + 3] = 65535;
    }
    return true;
  }
  switch (fmt) {
    case VK_FORMAT_R8_UNORM:
      for (size_t i = 0; i < n; ++i) out.px[i * 4] = out.px[i * 4 + 1] = out.px[i * 4 + 2] = d[i], out.px[i * 4 + 3] = 255;
      return true;
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
      for (size_t i = 0; i < n * 4; ++i) out.px[i] = d[i];
      return true;
    case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
      for (size_t i = 0; i < n; ++i)
        out.px[i * 4] = d[i * 4 + 2], out.px[i * 4 + 1] = d[i * 4 + 1], out.px[i * 4 + 2] = d[i * 4], out.px[i * 4 + 3] = d[i * 4 + 3];
      return true;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
      out.bits = 16;
      for (size_t i = 0; i < n; ++i) {
        const uint32_t v = rd32(i);
        out.px[i * 4] = uint16_t((v & 1023) * 65535 / 1023), out.px[i * 4 + 1] = uint16_t(((v >> 10) & 1023) * 65535 / 1023);
        out.px[i * 4 + 2] = uint16_t(((v >> 20) & 1023) * 65535 / 1023), out.px[i * 4 + 3] = uint16_t((v >> 30) * 65535 / 3);
      }
      return true;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      out.bits = 16;
      for (size_t i = 0; i < n * 4; ++i) {
        uint16_t hv;
        std::memcpy(&hv, d + i * 2, 2);
        out.px[i] = unorm16(halfToFloat(hv));
      }
      return true;
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
      out.bits = 16;
      for (size_t i = 0; i < n; ++i) {
        const uint32_t v = rd32(i);
        out.px[i * 4] = unorm16(smallFloat(v & 2047, 6)), out.px[i * 4 + 1] = unorm16(smallFloat((v >> 11) & 2047, 6));
        out.px[i * 4 + 2] = unorm16(smallFloat(v >> 22, 5)), out.px[i * 4 + 3] = 65535;
      }
      return true;
    default: return false;
  }
}

// ---- comparison ----
struct DiffResult {
  bool sizeMismatch = false;
  uint32_t maxDiff[4] = {0, 0, 0, 0};  // per channel, in sample units (0..255 or 0..65535)
  double mse[4] = {0, 0, 0, 0};
  double psnr = INFINITY;  // over all channels; infinity = identical
  double psnrChannel[4] = {INFINITY, INFINITY, INFINITY, INFINITY};
  uint32_t maxOf() const { return std::max(std::max(maxDiff[0], maxDiff[1]), std::max(maxDiff[2], maxDiff[3])); }
};

inline DiffResult diffImages(const Image& a, const Image& b) {
  DiffResult r;
  if (a.w != b.w || a.h != b.h || a.bits != b.bits || a.px.size() != b.px.size()) {
    r.sizeMismatch = true;
    r.psnr = 0;
    return r;
  }
  const double peak = a.maxValue();
  const size_t n = size_t(a.w) * a.h;
  double sum[4] = {0, 0, 0, 0};
  for (size_t i = 0; i < n; ++i)
    for (int c = 0; c < 4; ++c) {
      const int d = std::abs(int(a.px[i * 4 + c]) - int(b.px[i * 4 + c]));
      r.maxDiff[c] = std::max<uint32_t>(r.maxDiff[c], uint32_t(d));
      sum[c] += double(d) * d;
    }
  double all = 0;
  for (int c = 0; c < 4; ++c) {
    r.mse[c] = n ? sum[c] / double(n) : 0;
    all += r.mse[c];
    if (r.mse[c] > 0) r.psnrChannel[c] = 10 * std::log10(peak * peak / r.mse[c]);
  }
  all /= 4;
  if (all > 0) r.psnr = 10 * std::log10(peak * peak / all);
  return r;
}

}  // namespace vpimg
