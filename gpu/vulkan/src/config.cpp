// vpvk/config.h: host-app settings, stored and logged ("vk.fov" / "vk.cfg"). The passes that
// consume them (rate map on multiview eye passes, VPKTAA final pass) are not built yet.
#include "vpvk/config.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "vpvk.h"

namespace {
std::mutex gMutex;
vpvk_config gCfg = {{VPVK_FOVEATION_OFF, 0.35f, 0.5f, 0.0f}, {0, 50.0f, 0.5f}, 1.0f, 1, 1};
std::atomic<uint32_t> gGen{0};
std::atomic<int> gPrewarm{1};
std::atomic<int> gAniso{16};
std::atomic<int> gForce1x{-1};  // -1 = follow vpktaa.enabled  // default ON (app setting "Filtrado anisotrópico x16")
// VPKTAA inline jitter: the setting (vpvk_taa_set_jitter) and what submissions applied, read and
// reset by vpvk_taa_get_applied. One mutex: taken once per vkQueueSubmit and per setter call.
std::mutex gTaaMutex;
vpvk::TaaJitter gTaa;  // disabled until the host enables it
struct TaaApplied {
  uint64_t tag = 0;
  float jx = 0, jy = 0;
  uint32_t targetW = 0, targetH = 0;
  uint32_t jittered = 0, skipped = 0, submits = 0;
  // Submissions per distinct jitter, so mixed = submits - those with the last jitter. A frame has
  // one or two; a jitter beyond the table counts as differing from every other.
  struct Seen {
    float jx, jy;
    uint32_t n;
  } seen[8];
  uint32_t nSeen = 0;
};
TaaApplied gTaaRead;  // since the last vpvk_taa_get_applied
// Since the last "vk.taa jitter" line (every 1800 submissions).
struct TaaLog {
  uint64_t nextSubmit = 0;
  bool first = true;
  uint32_t jittered = 0, skipped = 0, submits = 0, changes = 0;
  float jx = 0, jy = 0;
} gTaaLog;
bool taaEnvOff() {
  static const bool off = [] {
    const char* e = getenv("VPVK_TAA_JITTER");
    return e && e[0] == '0' && !e[1];
  }();
  return off;
}

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
int anisotropyForced() {
  // Env backup (set by the app next to vpvk_config_set_anisotropy): 0/1 off, else 2/4/8/16.
  static const int env = [] {
    const char* e = getenv("VPVK_ANISO");
    return e && *e ? atoi(e) : -1;
  }();
  return env >= 0 ? (env <= 1 ? 0 : std::min(env, 16)) : gAniso.load();
}

// The app forwards its settings before Klepton opens klepton-boot.log, so those "vk.cfg" lines
// land on the earlier stderr; vkCreateDevice repeats the effective configuration here.
void logConfig() {
  vpvk_config c;
  {
    std::lock_guard<std::mutex> l(gMutex);
    c = gCfg;
  }
  vpvk::logf("vk.cfg effective gen=%u fov=%s sharp=%.2f periphery=%.2f vpktaa=%d %.0f%% sharp=%.2f scale=%.2f "
             "asw_off=%d quest3=%d anisotropy=%d prewarm=%d force_1x=%d set_by_app=%s",
             gGen.load(), modeName(c.foveation.mode), c.foveation.sharp_zone, c.foveation.periphery, c.vpktaa.enabled,
             c.vpktaa.percent, c.vpktaa.sharpness, c.render_scale, c.asw_disabled, c.quest3_profile,
             anisotropyForced(), gPrewarm.load(), forceSingleSample() ? 1 : 0, gGen.load() ? "yes" : "no(defaults)");
}
TaaJitter taaJitter() {
  if (taaEnvOff()) return {};
  std::lock_guard<std::mutex> l(gTaaMutex);
  return gTaa;
}

void taaRecordSubmit(uint64_t submit, const TaaJitter& j, uint32_t jittered, uint32_t skipped, uint32_t targetW,
                     uint32_t targetH) {
  std::lock_guard<std::mutex> l(gTaaMutex);
  TaaApplied& a = gTaaRead;
  TaaLog& g = gTaaLog;
  a.skipped += skipped, g.skipped += skipped;
  if (jittered) {
    if (g.submits && (g.jx != j.jx || g.jy != j.jy)) ++g.changes;
    g.jx = j.jx, g.jy = j.jy;
    uint32_t i = 0;
    while (i < a.nSeen && (a.seen[i].jx != j.jx || a.seen[i].jy != j.jy)) ++i;
    if (i < a.nSeen) ++a.seen[i].n;
    else if (a.nSeen < 8) a.seen[a.nSeen++] = {j.jx, j.jy, 1};
    a.tag = j.tag, a.jx = j.jx, a.jy = j.jy;
    a.targetW = std::max(a.targetW, targetW), a.targetH = std::max(a.targetH, targetH);
    a.jittered += jittered, g.jittered += jittered;
    ++a.submits, ++g.submits;
  }
  // First jittered submission, then every 1800 submissions while the jitter is on.
  if ((g.first && jittered) || (!g.first && submit >= g.nextSubmit)) {
    logf("vk.taa jitter submit=%llu enabled=%d eye=%ux%u last=(%.3f,%.3f) tag=%llu target=%ux%u submits=%u "
         "jittered=%u skipped=%u changes=%u",
         static_cast<unsigned long long>(submit), j.enabled ? 1 : 0, j.width, j.height, a.jx, a.jy,
         static_cast<unsigned long long>(a.tag), a.targetW, a.targetH, g.submits, g.jittered, g.skipped, g.changes);
    g = TaaLog{};
    g.first = false;
    g.nextSubmit = submit + 1800;
  }
}

bool forceSingleSample() {
  if (const char* e = getenv("VPVK_FORCE_1X"); e && (*e == '0' || *e == '1')) return *e == '1';
  const int f = gForce1x.load();
  if (f >= 0) return f != 0;
  std::lock_guard<std::mutex> l(gMutex);
  return gCfg.vpktaa.enabled != 0;
}
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

__attribute__((visibility("default"))) void vpvk_config_set_force_single_sample(int on) {
  gForce1x.store(on ? 1 : 0);
  gGen.fetch_add(1);
  vpvk::logf("vk.cfg force_single_sample=%d (game MSAA -> 1x, VPKTAA does the AA)", on ? 1 : 0);
}

__attribute__((visibility("default"))) void vpvk_config_set_anisotropy(int a) {
  a = a <= 1 ? 0 : a < 4 ? 2 : a < 8 ? 4 : a < 16 ? 8 : 16;  // Metal levels: 2/4/8/16
  gAniso.store(a);
  gGen.fetch_add(1);
  vpvk::logf("vk.cfg anisotropy=%d", a);
}

__attribute__((visibility("default"))) uint32_t vpvk_config_generation(void) { return gGen.load(); }

__attribute__((visibility("default"))) void vpvk_taa_set_jitter(const vpvk_taa_jitter* j) {
  if (!j) return;
  auto clampHalf = [](float v) { return std::isfinite(v) ? std::clamp(v, -0.5f, 0.5f) : 0.0f; };
  vpvk::TaaJitter t;
  t.enabled = j->enabled != 0;
  t.width = j->width, t.height = j->height;
  t.jx = clampHalf(j->jx), t.jy = clampHalf(j->jy);
  t.tag = j->tag;
  bool changed;
  {
    std::lock_guard<std::mutex> l(gTaaMutex);
    changed = t.enabled != gTaa.enabled || t.width != gTaa.width || t.height != gTaa.height;
    gTaa = t;
  }
  // Called once per frame: only an on/off or eye-size change is logged.
  if (changed)
    vpvk::logf("vk.taa set enabled=%d eye=%ux%u env_off=%d", t.enabled ? 1 : 0, t.width, t.height,
               taaEnvOff() ? 1 : 0);
}

__attribute__((visibility("default"))) int vpvk_taa_get_applied(vpvk_taa_applied* out) {
  std::lock_guard<std::mutex> l(gTaaMutex);
  TaaApplied& a = gTaaRead;
  if (out) {
    std::memset(out, 0, sizeof *out);
    out->tag = a.tag;
    out->jx = a.jx, out->jy = a.jy;
    out->target_w = a.targetW, out->target_h = a.targetH;
    out->jittered_draws = a.jittered, out->skipped_draws = a.skipped;
    out->submits = a.submits;
    uint32_t last = a.submits ? 1 : 0;  // the last submission itself, if its jitter fell off the table
    for (uint32_t i = 0; i < a.nSeen; ++i)
      if (a.seen[i].jx == a.jx && a.seen[i].jy == a.jy) last = a.seen[i].n;
    out->mixed = a.submits - last;
  }
  const int any = a.jittered ? 1 : 0;
  // The last jitter stays as the reference for `mixed` and for a read with nothing new.
  a.targetW = a.targetH = 0;
  a.jittered = a.skipped = a.submits = a.nSeen = 0;
  return any;
}

}  // extern "C"
