#include "texdecode.h"

#include <algorithm>
#include <cstring>

namespace vpvk {
namespace {

// Quantisation methods in ascending order (ASTC "range"): index -> levels. Trit ranges 3*2^k, quint ranges 5*2^k, else 2^k.
constexpr int kLevels[21] = {2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256};
struct QInfo {
  int kind;  // 0 bits only, 1 trits, 2 quints
  int bits;  // low bits per value
};
QInfo qinfo(int qi) {
  int l = kLevels[qi], k = 0;
  if (l % 3 == 0) {
    l /= 3, k = 1;
  } else if (l % 5 == 0) {
    l /= 5, k = 2;
  }
  int b = 0;
  while ((1 << b) < l) ++b;
  return {k, b};
}

int iseBits(int count, int qi) {
  const QInfo q = qinfo(qi);
  return q.kind == 0 ? count * q.bits : q.kind == 1 ? (8 * count + 4) / 5 + count * q.bits : (7 * count + 2) / 3 + count * q.bits;
}

struct Bits {  // little-endian bit reader over a byte array
  const uint8_t* b;
  unsigned get(int pos, int n) const {
    unsigned r = 0;
    for (int i = 0; i < n; ++i) r |= unsigned((b[(pos + i) >> 3] >> ((pos + i) & 7)) & 1) << i;
    return r;
  }
};

// Bounded integer sequence decode: `count` values of quantisation qi from bit `pos` of `in`; out[i] = digit << bits | low bits.
void iseDecode(const Bits& in, int pos, int count, int qi, int* out) {
  const QInfo q = qinfo(qi);
  if (q.kind == 0) {
    for (int i = 0; i < count; ++i) out[i] = int(in.get(pos + i * q.bits, q.bits));
    return;
  }
  const int group = q.kind == 1 ? 5 : 3;
  static const int kTritSizes[5] = {2, 2, 1, 2, 1}, kQuintSizes[3] = {3, 2, 2};
  for (int g = 0; g * group < count; ++g) {
    int m[5] = {0, 0, 0, 0, 0};
    unsigned t = 0;  // the 8 trit bits / 7 quint bits
    int tpos = 0;
    for (int i = 0; i < group && g * group + i < count; ++i) {
      m[i] = int(in.get(pos, q.bits));
      pos += q.bits;
      const int n = q.kind == 1 ? kTritSizes[i] : kQuintSizes[i];
      t |= in.get(pos, n) << tpos;
      pos += n, tpos += n;
    }
    int d[5];
    if (q.kind == 1) astcTritDecode(t, d);
    else astcQuintDecode(t, d);
    for (int i = 0; i < group && g * group + i < count; ++i) out[g * group + i] = d[i] << q.bits | m[i];
  }
}

// Patterns of the unquantisation tables: letters a.. are the low bits of the value (a = bit 0), '0' a zero; leftmost = top bit.
int pattern(const char* s, int m) {
  int v = 0;
  for (; *s; ++s) v = v << 1 | (*s == '0' ? 0 : (m >> (*s - 'a')) & 1);
  return v;
}

struct Tables {
  uint8_t color[21][256];
  uint8_t weight[21][128];  // 0..64
  Tables() {
    std::memset(color, 0, sizeof color), std::memset(weight, 0, sizeof weight);
    // Colour endpoints (range 6..256): trit C/B, quint C/B of the specification's table.
    static const char* const tritB[7] = {"", "000000000", "b000b0bb0", "cb000cbcb", "dcb000dcb", "edcb000ed", "fedcb000f"};
    static const int tritC[7] = {0, 204, 93, 44, 22, 11, 5};
    static const char* const quintB[6] = {"", "000000000", "b0000bb00", "cb0000cbc", "dcb0000dc", "edcb0000e"};
    static const int quintC[6] = {0, 113, 54, 26, 13, 6};
    for (int qi = 0; qi < 21; ++qi) {
      const QInfo q = qinfo(qi);
      const int n = kLevels[qi];
      for (int v = 0; v < n; ++v) {
        const int m = v & ((1 << q.bits) - 1), d = v >> q.bits;
        if (q.kind == 0) {  // replicate the bits up to 8
          int r = 0;
          for (int s = 8 - q.bits; s > -q.bits; s -= q.bits) r |= s >= 0 ? m << s : m >> -s;
          color[qi][v] = uint8_t(r & 0xFF);
        } else if (q.bits >= 1) {
          const int A = (m & 1) ? 0x1FF : 0;
          int T = q.kind == 1 ? d * tritC[q.bits] + pattern(tritB[q.bits], m) : d * quintC[q.bits] + pattern(quintB[q.bits], m);
          T ^= A;
          color[qi][v] = uint8_t((A & 0x80) | (T >> 2));
        }
      }
    }
    // Weights (0..64): bit replication to 6 bits, the explicit trit/quint-only tables, and the trit/quint-with-bits formula.
    static const char* const wTritB[4] = {"", "0000000", "b000bb0", "cb000cb"};
    static const int wTritC[4] = {0, 50, 23, 11};
    static const char* const wQuintB[3] = {"", "0000000", "b000b00"};
    static const int wQuintC[3] = {0, 28, 13};
    for (int qi = 0; qi < 12; ++qi) {  // weight ranges are 2..32
      const QInfo q = qinfo(qi);
      for (int v = 0; v < kLevels[qi]; ++v) {
        const int m = v & ((1 << q.bits) - 1), d = v >> q.bits;
        int w = 0;
        if (q.kind == 0) {
          for (int s = 6 - q.bits; s > -q.bits; s -= q.bits) w |= s >= 0 ? m << s : m >> -s;
          w &= 63;
        } else if (q.bits == 0) {
          static const int t3[3] = {0, 32, 63}, q5[5] = {0, 16, 32, 47, 63};
          w = q.kind == 1 ? t3[d] : q5[d];
        } else {
          const int A = (m & 1) ? 0x7F : 0;
          int T = q.kind == 1 ? d * wTritC[q.bits] + pattern(wTritB[q.bits], m) : d * wQuintC[q.bits] + pattern(wQuintB[q.bits], m);
          T ^= A;
          w = (A & 0x20) | (T >> 2);
        }
        weight[qi][v] = uint8_t(w > 32 ? w + 1 : w);
      }
    }
  }
};
const Tables& tables() {
  static const Tables t;
  return t;
}
int qindex(int levels) {
  for (int i = 0; i < 21; ++i)
    if (kLevels[i] == levels) return i;
  return 0;
}

uint32_t hash52(uint32_t p) {
  p ^= p >> 15;
  p *= 0xEEDE0891u;
  p ^= p >> 5;
  p += p << 16;
  p ^= p >> 7;
  p ^= p >> 3;
  p ^= p << 6;
  p ^= p >> 17;
  return p;
}

int selectPartition(int seed, int x, int y, int z, int parts, bool small) {
  if (small) x <<= 1, y <<= 1, z <<= 1;
  seed += (parts - 1) * 1024;
  const uint32_t r = hash52(uint32_t(seed));
  int s[12] = {int(r & 15), int((r >> 4) & 15), int((r >> 8) & 15), int((r >> 12) & 15), int((r >> 16) & 15), int((r >> 20) & 15),
               int((r >> 24) & 15), int((r >> 28) & 15), int((r >> 18) & 15), int((r >> 22) & 15), int((r >> 26) & 15),
               int(((r >> 30) | (r << 2)) & 15)};
  for (int& v : s) v *= v;
  int sh1, sh2;
  if (seed & 1) {
    sh1 = (seed & 2) ? 4 : 5;
    sh2 = parts == 3 ? 6 : 5;
  } else {
    sh1 = parts == 3 ? 6 : 5;
    sh2 = (seed & 2) ? 4 : 5;
  }
  const int sh3 = (seed & 0x10) ? sh1 : sh2;
  for (int i = 0; i < 8; ++i) s[i] >>= (i % 2 == 0 ? sh1 : sh2);
  for (int i = 8; i < 12; ++i) s[i] >>= sh3;
  int a = s[0] * x + s[1] * y + s[10] * z + int(r >> 14);
  int b = s[2] * x + s[3] * y + s[11] * z + int(r >> 10);
  int c = s[4] * x + s[5] * y + s[8] * z + int(r >> 6);
  int d = s[6] * x + s[7] * y + s[9] * z + int(r >> 2);
  a &= 63, b &= 63, c &= 63, d &= 63;
  if (parts <= 3) d = 0;
  if (parts <= 2) c = 0;
  if (parts <= 1) b = 0;
  if (a >= b && a >= c && a >= d) return 0;
  if (b >= c && b >= d) return 1;
  return c >= d ? 2 : 3;
}

int clamp255(int v) { return v < 0 ? 0 : v > 255 ? 255 : v; }
void bitTransfer(int& a, int& b) {  // a = the offset value, b = the base
  b >>= 1;
  b |= a & 0x80;
  a >>= 1;
  a &= 0x3F;
  if (a & 0x20) a -= 0x40;
}
void blueContract(int* c) {  // r, g, b, a in place
  c[0] = (c[0] + c[2]) >> 1;
  c[1] = (c[1] + c[2]) >> 1;
}

// LDR endpoint modes -> two RGBA endpoints (0..255). False: an HDR mode.
bool endpoints(int cem, const int* v, int* e0, int* e1) {
  switch (cem) {
    case 0: e0[0] = e0[1] = e0[2] = v[0], e1[0] = e1[1] = e1[2] = v[1], e0[3] = e1[3] = 255; return true;
    case 1: {
      const int l0 = (v[0] >> 2) | (v[1] & 0xC0), l1 = std::min(l0 + (v[1] & 0x3F), 255);
      e0[0] = e0[1] = e0[2] = l0, e1[0] = e1[1] = e1[2] = l1, e0[3] = e1[3] = 255;
      return true;
    }
    case 4: e0[0] = e0[1] = e0[2] = v[0], e1[0] = e1[1] = e1[2] = v[1], e0[3] = v[2], e1[3] = v[3]; return true;
    case 5: {
      int a = v[1], b = v[0], c = v[3], d = v[2];
      bitTransfer(a, b), bitTransfer(c, d);
      e0[0] = e0[1] = e0[2] = clamp255(b), e1[0] = e1[1] = e1[2] = clamp255(b + a);
      e0[3] = clamp255(d), e1[3] = clamp255(d + c);
      return true;
    }
    case 6: case 10:
      e0[0] = v[0] * v[3] >> 8, e0[1] = v[1] * v[3] >> 8, e0[2] = v[2] * v[3] >> 8;
      e1[0] = v[0], e1[1] = v[1], e1[2] = v[2];
      e0[3] = e1[3] = 255;
      if (cem == 10) e0[3] = v[4], e1[3] = v[5];
      return true;
    case 8: case 12: {
      int a0[4] = {v[0], v[2], v[4], cem == 12 ? v[6] : 255}, a1[4] = {v[1], v[3], v[5], cem == 12 ? v[7] : 255};
      if (v[1] + v[3] + v[5] >= v[0] + v[2] + v[4]) {
        std::memcpy(e0, a0, sizeof a0), std::memcpy(e1, a1, sizeof a1);
      } else {
        blueContract(a0), blueContract(a1);
        std::memcpy(e0, a1, sizeof a1), std::memcpy(e1, a0, sizeof a0);
      }
      return true;
    }
    case 9: case 13: {
      int r1 = v[1], r0 = v[0], g1 = v[3], g0 = v[2], b1 = v[5], b0 = v[4], a1 = 0, a0 = 255;
      bitTransfer(r1, r0), bitTransfer(g1, g0), bitTransfer(b1, b0);
      if (cem == 13) a1 = v[7], a0 = v[6], bitTransfer(a1, a0);
      int lo[4] = {r0, g0, b0, a0}, hi[4] = {r0 + r1, g0 + g1, b0 + b1, cem == 13 ? a0 + a1 : 255};
      if (r1 + g1 + b1 >= 0) {
        std::memcpy(e0, lo, sizeof lo), std::memcpy(e1, hi, sizeof hi);
      } else {
        blueContract(lo), blueContract(hi);
        std::memcpy(e0, hi, sizeof hi), std::memcpy(e1, lo, sizeof lo);
      }
      for (int i = 0; i < 4; ++i) e0[i] = clamp255(e0[i]), e1[i] = clamp255(e1[i]);
      return true;
    }
    default: return false;  // 2, 3, 7, 11, 14, 15: HDR
  }
}

}  // namespace

bool astcFormat(vpgpu::Format f) { return f != vpgpu::Format::Undefined && std::strncmp(vpgpu::formatInfo(f).name, "Astc", 4) == 0; }

vpgpu::Format astcDecodedFormat(vpgpu::Format f) {
  const char* n = vpgpu::formatInfo(f).name;
  const size_t len = std::strlen(n);
  return len >= 4 && std::strcmp(n + len - 4, "Srgb") == 0 ? vpgpu::Format::RGBA8Srgb : vpgpu::Format::RGBA8Unorm;
}

void astcTritDecode(unsigned T, int out[5]) {
  auto b = [&](int hi, int lo) { return int((T >> lo) & ((1u << (hi - lo + 1)) - 1)); };
  int C, t4, t3;
  if (b(4, 2) == 7) {
    C = b(7, 5) << 2 | b(1, 0), t4 = 2, t3 = 2;
  } else {
    C = b(4, 0);
    if (b(6, 5) == 3) t4 = 2, t3 = b(7, 7);
    else t4 = b(7, 7), t3 = b(6, 5);
  }
  int t2, t1, t0;
  if ((C & 3) == 3) {
    t2 = 2, t1 = (C >> 4) & 1, t0 = ((C >> 3) & 1) << 1 | (((C >> 2) & 1) & ~((C >> 3) & 1) & 1);
  } else if (((C >> 2) & 3) == 3) {
    t2 = 2, t1 = 2, t0 = C & 3;
  } else {
    t2 = (C >> 4) & 1, t1 = (C >> 2) & 3, t0 = ((C >> 1) & 1) << 1 | ((C & 1) & ~((C >> 1) & 1) & 1);
  }
  out[0] = t0, out[1] = t1, out[2] = t2, out[3] = t3, out[4] = t4;
}

void astcQuintDecode(unsigned Q, int out[3]) {
  auto b = [&](int hi, int lo) { return int((Q >> lo) & ((1u << (hi - lo + 1)) - 1)); };
  int q0, q1, q2;
  if (b(2, 1) == 3 && b(6, 5) == 0) {
    const int q0b = b(0, 0), q4 = b(4, 4), q3 = b(3, 3);
    q2 = q0b << 2 | (q4 & ~q0b & 1) << 1 | (q3 & ~q0b & 1), q1 = 4, q0 = 4;
  } else {
    int C;
    if (b(2, 1) == 3) q2 = 4, C = b(4, 3) << 3 | ((~b(6, 5)) & 3) << 1 | b(0, 0);
    else q2 = b(6, 5), C = b(4, 0);
    if ((C & 7) == 5) q1 = 4, q0 = C >> 3;
    else q1 = C >> 3, q0 = C & 7;
  }
  out[0] = q0, out[1] = q1, out[2] = q2;
}

int astcUnquantColor(int levels, int v) { return tables().color[qindex(levels)][v & 255]; }
int astcUnquantWeight(int levels, int v) { return tables().weight[qindex(levels)][v & 127]; }

bool astcDecodeBlock(const uint8_t blk[16], int bw, int bh, bool srgb, uint8_t* out) {
  auto fail = [&] {
    for (int i = 0; i < bw * bh; ++i) out[i * 4] = 255, out[i * 4 + 1] = 0, out[i * 4 + 2] = 255, out[i * 4 + 3] = 255;
    return false;
  };
  const Bits in{blk};
  const unsigned mode = in.get(0, 11);
  if ((mode & 0x1FF) == 0x1FC) {  // void extent: one colour, 16 bits per channel
    if (mode & 0x200) return fail();  // HDR
    uint8_t px[4];
    for (int c = 0; c < 4; ++c) {
      const unsigned v = blk[8 + c * 2] | blk[9 + c * 2] << 8;
      px[c] = uint8_t(srgb && c < 3 ? v >> 8 : (v + 128) / 257);
    }
    for (int i = 0; i < bw * bh; ++i) std::memcpy(out + i * 4, px, 4);
    return true;
  }
  // Block mode -> weight grid.
  int xw = 0, yw = 0, qi = 0;
  bool dual = false;
  {
    int base = (mode >> 4) & 1;
    unsigned H = (mode >> 9) & 1, D = (mode >> 10) & 1;
    const unsigned A = (mode >> 5) & 3;
    if ((mode & 3) != 0) {
      base |= int(mode & 3) << 1;
      unsigned B = (mode >> 7) & 3;
      switch ((mode >> 2) & 3) {
        case 0: xw = int(B) + 4, yw = int(A) + 2; break;
        case 1: xw = int(B) + 8, yw = int(A) + 2; break;
        case 2: xw = int(A) + 2, yw = int(B) + 8; break;
        default:
          B &= 1;
          if (mode & 0x100) xw = int(B) + 2, yw = int(A) + 2;
          else xw = int(A) + 2, yw = int(B) + 6;
      }
    } else {
      base |= int((mode >> 2) & 3) << 1;
      if (((mode >> 2) & 3) == 0) return fail();
      const unsigned B = (mode >> 9) & 3;
      switch ((mode >> 7) & 3) {
        case 0: xw = 12, yw = int(A) + 2; break;
        case 1: xw = int(A) + 2, yw = 12; break;
        case 2: xw = int(A) + 6, yw = int(B) + 6, D = 0, H = 0; break;
        default:
          if (((mode >> 5) & 3) == 0) xw = 6, yw = 10;
          else if (((mode >> 5) & 3) == 1) xw = 10, yw = 6;
          else return fail();
      }
    }
    qi = base - 2 + 6 * int(H);
    dual = D != 0;
    if (qi < 0 || qi > 11) return fail();
  }
  const int wcount = xw * yw * (dual ? 2 : 1), wbits = iseBits(wcount, qi);
  if (wcount > 64 || wbits < 24 || wbits > 96) return fail();
  const int parts = int(in.get(11, 2)) + 1;
  if (dual && parts == 4) return fail();
  int below = 128 - wbits, ccs = 0;
  if (dual) ccs = int(in.get(below - 2, 2)), below -= 2;
  int cems[4] = {0, 0, 0, 0}, colorStart, seed = 0;
  if (parts == 1) {
    cems[0] = int(in.get(13, 4)), colorStart = 17;
  } else {
    seed = int(in.get(13, 10)), colorStart = 29;
    unsigned et = in.get(23, 6);
    if ((et & 3) == 0) {
      for (int i = 0; i < parts; ++i) cems[i] = int(et >> 2);
    } else {
      const int hp = 3 * parts - 4;
      below -= hp;
      et |= in.get(below, hp) << 6;
      const int base = int(et & 3) - 1;
      for (int i = 0; i < parts; ++i) cems[i] = (base + int((et >> (2 + i)) & 1)) << 2 | int((et >> (2 + parts + 2 * i)) & 3);
    }
  }
  int nvals = 0;
  for (int i = 0; i < parts; ++i) nvals += ((cems[i] >> 2) + 1) * 2;
  if (nvals > 18) return fail();
  const int avail = below - colorStart;
  int cqi = -1;
  for (int q = 20; q >= 4; --q)
    if (iseBits(nvals, q) <= avail) {
      cqi = q;
      break;
    }
  if (cqi < 0) return fail();
  int raw[18];
  iseDecode(in, colorStart, nvals, cqi, raw);
  const Tables& tb = tables();
  int vals[18];
  for (int i = 0; i < nvals; ++i) vals[i] = tb.color[cqi][raw[i]];
  int e0[4][4], e1[4][4], at = 0;
  for (int p = 0; p < parts; ++p) {
    if (!endpoints(cems[p], vals + at, e0[p], e1[p])) {  // HDR endpoint in an LDR decode: the error colour
      for (int c = 0; c < 4; ++c) e0[p][c] = e1[p][c] = (c == 1 ? 0 : 255);
    }
    at += ((cems[p] >> 2) + 1) * 2;
  }
  // Weights: stored reversed from the top bit.
  uint8_t rev[16];
  for (int i = 0; i < 16; ++i) {
    uint8_t v = blk[15 - i], r = 0;
    for (int k = 0; k < 8; ++k) r |= ((v >> k) & 1) << (7 - k);
    rev[i] = r;
  }
  int wraw[128];
  iseDecode(Bits{rev}, 0, wcount, qi, wraw);
  int grid[2][64];
  for (int i = 0; i < xw * yw; ++i) {
    grid[0][i] = tb.weight[qi][wraw[dual ? 2 * i : i]];
    grid[1][i] = dual ? tb.weight[qi][wraw[2 * i + 1]] : 0;
  }
  const int cs = (1024 + bw / 2) / (bw - 1), ct = (1024 + bh / 2) / (bh - 1);
  const bool small = bw * bh < 31;
  for (int y = 0; y < bh; ++y)
    for (int x = 0; x < bw; ++x) {
      const int gs = (cs * x * (xw - 1) + 32) >> 6, gt = (ct * y * (yw - 1) + 32) >> 6;
      const int js = gs >> 4, fs = gs & 15, jt = gt >> 4, ft = gt & 15;
      const int v0 = js + jt * xw;
      const int w11 = (fs * ft + 8) >> 4, w10 = ft - w11, w01 = fs - w11, w00 = 16 - fs - ft + w11;
      int w[2];
      for (int pl = 0; pl < (dual ? 2 : 1); ++pl) {
        const int* g = grid[pl];
        const int p00 = g[v0], p01 = js + 1 < xw ? g[v0 + 1] : p00, p10 = jt + 1 < yw ? g[v0 + xw] : p00,
                  p11 = js + 1 < xw && jt + 1 < yw ? g[v0 + xw + 1] : p00;
        w[pl] = (p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11 + 8) >> 4;
      }
      if (!dual) w[1] = w[0];
      const int part = parts > 1 ? selectPartition(seed, x, y, 0, parts, small) : 0;
      for (int c = 0; c < 4; ++c) {
        const int wc = dual && c == ccs ? w[1] : w[0];
        const bool lowByte = srgb && c < 3;
        const int c0 = e0[part][c] << 8 | (lowByte ? 0x80 : e0[part][c]), c1 = e1[part][c] << 8 | (lowByte ? 0x80 : e1[part][c]);
        const int C = (c0 * (64 - wc) + c1 * wc + 32) >> 6;
        out[(y * bw + x) * 4 + c] = uint8_t(lowByte ? C >> 8 : (C + 128) / 257);
      }
    }
  return true;
}

}  // namespace vpvk
