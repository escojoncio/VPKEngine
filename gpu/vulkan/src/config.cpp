// vpvk/config.h: host-app settings, stored and logged ("vk.fov" / "vk.cfg"). The passes that
// consume them (rate map on multiview eye passes, VPKTAA final pass) are not built yet.
#include "vpvk/config.h"

#include <algorithm>
#include <atomic>
#include <mutex>

#include "vpvk.h"

namespace {
std::mutex gMutex;
vpvk_config gCfg = {{VPVK_FOVEATION_OFF, 0.35f, 0.5f, 0.0f}, {0, 50.0f, 0.5f}, 1.0f, 1, 1};
std::atomic<uint32_t> gGen{0};
std::atomic<int> gPrewarm{1};
std::atomic<int> gAniso{16};  // default ON (app setting "Filtrado anisotrópico x16")
const char* modeName(vpvk_foveation_mode m) {
  switch (m) {
    case VPVK_FOVEATION_M2: return "m2";
    case VPVK_FOVEATION_M5: return "m5";
    case VPVK_FOVEATION_CUSTOM: return "custom";
    default: return "off";
  }
}
}  // namespace

namespace vpvk {
bool prewarmEnabled() { return gPrewarm.load() != 0; }
int anisotropyForced() { return gAniso.load(); }
}  // namespace vpvk

extern "C" {

__attribute__((visibility("default"))) void vpvk_config_set_foveation(const vpvk_foveation* f) {
  if (!f) return;
  std::lock_guard<std::mutex> l(gMutex);
  vpvk_foveation v = *f;
  if (v.mode < VPVK_FOVEATION_OFF || v.mode > VPVK_FOVEATION_CUSTOM) v.mode = VPVK_FOVEATION_OFF;
  v.sharp_zone = std::clamp(v.sharp_zone, 0.05f, 1.0f);
  v.periphery = std::clamp(v.periphery, 0.1f, 1.0f);
  v.height = std::clamp(v.height, -0.5f, 0.5f);
  gCfg.foveation = v;
  gGen.fetch_add(1);
  vpvk::logf("vk.fov config mode=%s sharp=%.2f periphery=%.2f height=%.2f applied=no(rate_map_pending)",
             modeName(v.mode), v.sharp_zone, v.periphery, v.height);
}

__attribute__((visibility("default"))) void vpvk_config_set_vpktaa(const vpvk_vpktaa* t) {
  if (!t) return;
  std::lock_guard<std::mutex> l(gMutex);
  vpvk_vpktaa v = *t;
  v.enabled = v.enabled ? 1 : 0;
  v.percent = std::clamp(v.percent, 0.0f, 100.0f);
  v.sharpness = std::clamp(v.sharpness, 0.0f, 1.0f);
  gCfg.vpktaa = v;
  gGen.fetch_add(1);
  vpvk::logf("vk.cfg vpktaa enabled=%d percent=%.0f sharpness=%.2f applied=compositor(klepton)", v.enabled,
             v.percent, v.sharpness);
}

__attribute__((visibility("default"))) void vpvk_config_set_render_scale(float s) {
  std::lock_guard<std::mutex> l(gMutex);
  gCfg.render_scale = std::clamp(s, 0.5f, 1.5f);
  gGen.fetch_add(1);
  vpvk::logf("vk.cfg render_scale=%.2f", gCfg.render_scale);
}

__attribute__((visibility("default"))) void vpvk_config_set_asw_disabled(int d) {
  std::lock_guard<std::mutex> l(gMutex);
  gCfg.asw_disabled = d ? 1 : 0;
  gGen.fetch_add(1);
  vpvk::logf("vk.cfg asw_disabled=%d", gCfg.asw_disabled);
}

__attribute__((visibility("default"))) void vpvk_config_set_quest3_profile(int on) {
  std::lock_guard<std::mutex> l(gMutex);
  gCfg.quest3_profile = on ? 1 : 0;
  gGen.fetch_add(1);
  vpvk::logf("vk.cfg quest3_profile=%d", gCfg.quest3_profile);
}

__attribute__((visibility("default"))) void vpvk_config_get(vpvk_config* out) {
  if (!out) return;
  std::lock_guard<std::mutex> l(gMutex);
  *out = gCfg;
}

__attribute__((visibility("default"))) void vpvk_config_set_prewarm(int on) {
  gPrewarm.store(on ? 1 : 0);
  vpvk::logf("vk.cfg prewarm=%d", on ? 1 : 0);
}

__attribute__((visibility("default"))) void vpvk_config_set_anisotropy(int a) {
  a = a <= 1 ? 0 : std::min(a, 16);
  gAniso.store(a);
  gGen.fetch_add(1);
  vpvk::logf("vk.cfg anisotropy=%d", a);
}

__attribute__((visibility("default"))) uint32_t vpvk_config_generation(void) { return gGen.load(); }

}  // extern "C"
