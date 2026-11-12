#include "foveation.h"
#include <atomic>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace vpvk::fov {
namespace {
// std::clamp lets a NaN through (a host's 0/0 tangent, an unset float): it becomes `def`.
float clampOr(float v, float lo, float hi, float def) { return v != v ? def : std::clamp(v, lo, hi); }
float smootherstep(float t) {
  t = clampOr(t, 0.0f, 1.0f, 0.0f);
  return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}
uint64_t mix(uint64_t h, uint64_t v) { return (h ^ v) * 0x100000001b3ull; }
}  // namespace

// Presets de PTVisionPro (visionos/App/Core/PTSettings.swift, defaults(for:)): center 60/70 grados, periphery 10/15 %,
// centerUp -10 grados, centerResolution 100 %; caída de 0.15 del radio (xr_host_visionos.mm, inset_fragment pad.y).
Shape presetM2() {
  Shape s;
  s.sharp = 0.481f, s.periphery = 0.10f, s.height = -0.0735f, s.fade = 0.15f, s.off = false;
  return s;
}
Shape presetM5() {
  Shape s;
  s.sharp = 0.583f, s.periphery = 0.15f, s.height = -0.0735f, s.fade = 0.15f, s.off = false;
  return s;
}

Shape shapeOf(int mode, float sharp, float periphery, float height) {
  Shape s;
  switch (mode) {
    case 1: s = presetM2(); break;
    case 2: s = presetM5(); break;
    case 3:
      s.off = false;
      s.sharp = clampOr(sharp, 0.05f, 1.0f, s.sharp);
      s.periphery = clampOr(periphery, 0.1f, 1.0f, s.periphery);
      break;
    default: return s;  // apagado
  }
  s.height = clampOr(s.height + clampOr(height, -0.5f, 0.5f, 0.0f), -0.5f, 0.5f, 0.0f);  // en los presets, `height` es un ajuste sobre el suyo
  return s;
}

const char* modeName(int mode) {
  switch (mode) {
    case 1: return "m2";
    case 2: return "m5";
    case 3: return "custom";
    default: return "off";
  }
}

Center opticalCenter(const std::array<float, 4>& t) {
  Center c;
  const float h = t[0] + t[1], v = t[2] + t[3];
  if (h > 1e-6f && t[0] >= 0 && t[1] >= 0) c.x = clampOr(t[0] / h, 0.05f, 0.95f, c.x);
  if (v > 1e-6f && t[2] >= 0 && t[3] >= 0) c.y = clampOr(t[2] / v, 0.05f, 0.95f, c.y);
  return c;
}

void axisQuality(float* q, uint32_t zones, float center, float sharp, float periphery, float fade) {
  center = clampOr(center, 0.0f, 1.0f, 0.5f);
  sharp = clampOr(sharp, 0.0f, 0.99f, 0.35f);
  periphery = clampOr(periphery, 0.1f, 1.0f, 1.0f);
  if (fade != fade) fade = -1.0f;
  for (uint32_t i = 0; i < zones; ++i) {
    const float p = (float(i) + 0.5f) / float(zones);
    // Distancia al centro normalizada con el borde de SU lado: 1 en el borde de la imagen a cada lado.
    const float d = p < center ? (center - p) / std::max(center, 1e-4f) : (p - center) / std::max(1.0f - center, 1e-4f);
    const float t = fade < 0.0f ? (d <= sharp ? 0.0f : (d - sharp) / (1.0f - sharp))
                                : (d - sharp * (1.0f - fade)) / std::max(sharp * fade, 1e-4f);
    q[i] = 1.0f - (1.0f - periphery) * smootherstep(t);
  }
}

float keptFraction(const vpgpu::RateMap& m) {
  if (!m.layers || !m.zonesX || !m.zonesY) return 1.0f;
  double sum = 0;
  for (uint16_t l = 0; l < m.layers; ++l) {
    double mx = 0, my = 0;
    for (uint32_t i = 0; i < m.zonesX; ++i) mx += m.qx[size_t(l) * m.zonesX + i];
    for (uint32_t j = 0; j < m.zonesY; ++j) my += m.qy[size_t(l) * m.zonesY + j];
    sum += (mx / m.zonesX) * (my / m.zonesY);
  }
  return float(sum / m.layers);
}

static std::atomic<uint64_t> gBuilds{0};
uint64_t rateMapBuilds() { return gBuilds.load(std::memory_order_relaxed); }

std::shared_ptr<vpgpu::RateMap> buildRateMap(uint32_t w, uint32_t h, uint16_t layers, const Shape& shape, const Eyes& eyes,
                                             uint32_t zones) {
  if (shape.off || !w || !h || !layers) return nullptr;
  gBuilds.fetch_add(1, std::memory_order_relaxed);
  zones = std::clamp<uint32_t>(zones, 1, kMaxZones);
  auto m = std::make_shared<vpgpu::RateMap>();
  m->width = w, m->height = h, m->layers = layers;
  m->zonesX = m->zonesY = uint16_t(zones);
  m->qx.resize(size_t(layers) * zones);
  m->qy.resize(size_t(layers) * zones);
  for (uint16_t l = 0; l < layers; ++l) {
    Center c;
    if (eyes.known) c = opticalCenter(eyes.tangents[std::min<size_t>(l, 1)]);
    // `height` + = el centro de la zona nítida sube (y desde arriba baja).
    const float cy = clampOr(c.y - shape.height, 0.0f, 1.0f, 0.5f);
    axisQuality(&m->qx[size_t(l) * zones], zones, c.x, shape.sharp, shape.periphery, shape.fade);
    axisQuality(&m->qy[size_t(l) * zones], zones, cy, shape.sharp, shape.periphery, shape.fade);
  }
  m->keptFraction = keptFraction(*m);
  uint64_t k = 0xcbf29ce484222325ull;
  k = mix(k, w), k = mix(k, h), k = mix(k, layers), k = mix(k, zones);
  for (float v : m->qx) k = mix(k, uint64_t(std::lround(v * 4096.0f)));
  for (float v : m->qy) k = mix(k, uint64_t(std::lround(v * 4096.0f)));
  m->key = k ? k : 1;
  char label[96];
  std::snprintf(label, sizeof label, "vpvk.fov %ux%u x%u kept=%.2f", w, h, unsigned(layers), double(m->keptFraction));
  m->label = label;
  return m;
}

}  // namespace vpvk::fov
