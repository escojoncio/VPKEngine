// VPKTAA nativo: matemática compartida con los kernels MSL (ver taa_native.h).
#include "taa_native.h"

#include <algorithm>
#include <cmath>

namespace vpvk::taa {

float halton(uint32_t i, uint32_t base) {
  float f = 1.0f, r = 0.0f;
  while (i > 0) {
    f /= float(base);
    r += f * float(i % base);
    i /= base;
  }
  return r;
}

std::array<float, 2> haltonJitter(uint64_t frame) {
  const uint32_t k = uint32_t(frame % kPhases) + 1;
  return {halton(k, 2) - 0.5f, halton(k, 3) - 0.5f};
}

std::array<float, 2> jitterClipOffset(float jx, float jy, float w, float h, float cw) {
  return {2.0f * jx / w * cw, -2.0f * jy / h * cw};
}

Reconstruction reconstruction(float sx, float sy) {
  Reconstruction r;
  const float s[2] = {sx, sy};
  std::array<float, 3> axis[2];
  for (int a = 0; a < 2; ++a) {
    r.base[a] = int32_t(std::floor(0.5f + s[a]));
    float best = 1e9f;
    for (int o = -1; o <= 1; ++o) {
      const float d = float(r.base[a] + o) - s[a];
      axis[a][o + 1] = std::exp(-2.29f * d * d);
      if (std::fabs(d) < best) best = std::fabs(d), r.nearest[a] = o;
    }
  }
  float sum = 0;
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 3; ++x) sum += r.weights[y * 3 + x] = axis[0][x] * axis[1][y];
  for (float& w : r.weights) w /= sum;
  return r;
}

std::array<float, 4> catmullRom(float f) {
  return {f * (-0.5f + f * (1.0f - 0.5f * f)), 1.0f + f * f * (-2.5f + 1.5f * f), f * (0.5f + f * (2.0f - 1.5f * f)),
          f * f * (-0.5f + 0.5f * f)};
}

std::array<float, 4> catmullRom5(float f) {
  const auto w = catmullRom(f);
  const float w12 = w[1] + w[2];
  return {w[0], w12, w[3], w[2] / w12};
}

bool reversedZ(bool cleared, float clearValue, uint32_t greater, uint32_t less, bool fallback) {
  if (greater != less) return greater > less;
  if (cleared) return clearValue < 0.5f;
  return fallback;
}

float viewDistance(float depth, float nearZ, float farZ, bool reversedZ) {
  const float d = reversedZ ? 1.0f - depth : depth;  // como profundidad estándar
  const float n = std::max(nearZ, 1e-4f);
  if (farZ <= 0.0f) return n / std::max(1.0f - d, 1e-7f);
  return farZ * n / std::max(farZ - d * (farZ - n), 1e-7f);
}

namespace {
Quat conj(Quat q) { return {-q.x, -q.y, -q.z, q.w}; }
Quat mul(Quat a, Quat b) {
  return {a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y, a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
          a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w, a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}
Vec3 rotate(Quat q, Vec3 v) {
  const Quat p = mul(mul(q, {v.x, v.y, v.z, 0}), conj(q));
  return {p.x, p.y, p.z};
}
Quat normalized(Quat q) {
  const float n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  return n > 0 ? Quat{q.x / n, q.y / n, q.z / n, q.w / n} : Quat{};
}
}  // namespace

Reprojection reprojection(const EyeView& cur, const EyeView& prev) {
  const Quat qc = normalized(cur.orientation), qp = normalized(prev.orientation);
  const Quat rel = mul(conj(qp), qc);
  Reprojection m;
  const Vec3 cx = rotate(rel, {1, 0, 0}), cy = rotate(rel, {0, 1, 0}), cz = rotate(rel, {0, 0, 1});
  m.r = {cx.x, cy.x, cz.x, cx.y, cy.y, cz.y, cx.z, cy.z, cz.z};
  m.t = rotate(conj(qp), {cur.position.x - prev.position.x, cur.position.y - prev.position.y,
                          cur.position.z - prev.position.z});
  return m;
}

bool reprojectUv(const Reprojection& m, const std::array<float, 4>& tc, const std::array<float, 4>& tp, float u,
                 float v, float dist, float& pu, float& pv, float& prevDist) {
  // Dirección en el ojo actual con z = -1 (u: -l..r, v: t..-b).
  const Vec3 d{-tc[0] + u * (tc[0] + tc[1]), tc[2] - v * (tc[2] + tc[3]), -1.0f};
  const auto& r = m.r;
  Vec3 p{r[0] * d.x + r[1] * d.y + r[2] * d.z, r[3] * d.x + r[4] * d.y + r[5] * d.z,
         r[6] * d.x + r[7] * d.y + r[8] * d.z};
  if (dist > 0) p = {p.x * dist + m.t.x, p.y * dist + m.t.y, p.z * dist + m.t.z};
  if (p.z >= -1e-6f) return false;
  const float z = -p.z;
  prevDist = dist > 0 ? z : 0.0f;
  pu = (p.x / z + tp[0]) / (tp[0] + tp[1]);
  pv = (tp[2] - p.y / z) / (tp[2] + tp[3]);
  return true;
}

float uiMaskFromAlpha(float alpha) { return std::clamp(alpha * kUiMaskAlphaGain, 0.0f, 1.0f); }

namespace {
float clampMask(float m) { return m <= kUiMaskEps ? 0.0f : std::min(m, 1.0f); }
}  // namespace

float maskedBase(float recon, float current, float m) {
  m = clampMask(m);
  return recon + (current - recon) * m;
}

float maskedHistoryWeight(float keep, float m) { return keep * (1.0f - clampMask(m)); }

float rcasConstant(float intensity) { return std::exp2(-2.0f * (1.0f - rcasClamp(intensity))); }

std::array<float, 3> rcasPixel(const std::array<std::array<float, 3>, 5>& c, float intensity, bool denoise) {
  const auto& b = c[0];
  const auto& d = c[1];
  const auto& e = c[2];
  const auto& f = c[3];
  const auto& h = c[4];
  if (rcasClamp(intensity) <= 0.0f) return e;
  float lobe = -1.0e9f;
  for (int k = 0; k < 3; ++k) {
    const float mn = std::min(std::min(b[k], d[k]), std::min(f[k], h[k]));
    const float mx = std::max(std::max(b[k], d[k]), std::max(f[k], h[k]));
    const float hitMin = std::min(mn, e[k]) / (4.0f * std::max(mx, 1.0e-4f));
    const float hitMax = (1.0f - std::max(mx, e[k])) / std::min(4.0f * mn - 4.0f, -1.0e-4f);
    lobe = std::max(lobe, std::max(-hitMin, hitMax));
  }
  lobe = std::max(-kRcasLimit, std::min(lobe, 0.0f)) * rcasConstant(intensity);
  if (denoise) {
    auto luma = [](const std::array<float, 3>& p) { return 0.5f * p[0] + p[1] + 0.5f * p[2]; };
    const float lb = luma(b), ld = luma(d), le = luma(e), lf = luma(f), lh = luma(h);
    const float range = std::max(std::max(std::max(lb, ld), std::max(le, lf)), lh) -
                        std::min(std::min(std::min(lb, ld), std::min(le, lf)), lh);
    const float nz = std::clamp(std::fabs(0.25f * (lb + ld + lf + lh) - le) / std::max(range, 1.0e-4f), 0.0f, 1.0f);
    lobe *= 1.0f - 0.5f * nz;
  }
  std::array<float, 3> o{};
  for (int k = 0; k < 3; ++k)
    o[k] = std::clamp((lobe * (b[k] + d[k] + f[k] + h[k]) + e[k]) / (4.0f * lobe + 1.0f), 0.0f, 1.0f);
  return o;
}

}  // namespace vpvk::taa
