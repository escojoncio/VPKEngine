// vpvk/config.h: host-app settings, stored and logged ("vk.fov" / "vk.cfg"). The passes that
// consume them: foveation = RenderPassDesc::rateMap (commands.cpp) + gpu/metal2; VPKTAA final pass = vpvk_taa_native_encode (VPKTAA nativo).
#include "vpvk/config.h"

#include <algorithm>
#include <bit>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "vpvk.h"
#include "taa_native.h"

namespace {
std::mutex gMutex;
vpvk_config gCfg = {{VPVK_FOVEATION_OFF, 0.35f, 0.5f, 0.0f}, {0, 50.0f, 0.5f}, 1.0f, 1, 1};
std::atomic<uint32_t> gGen{0};
std::atomic<int> gPrewarm{1};
std::atomic<int> gQuestIdentity{0};  // vpvk_config_set_quest_identity
std::atomic<int> gAniso{16};
std::atomic<int> gForce1x{-1};  // -1 = follow vpktaa.enabled  // default ON (app setting "Filtrado anisotrópico x16")
std::atomic<int> gFovApply{0};  // vpvk_config_set_foveation_apply (env VPVK_FOVEATION wins)
vpvk::fov::Eyes gFovEyes;       // gMutex
std::atomic<int> gMsaaOverride{-1};  // -1 = not set by the app: env VPVK_MSAA (0 = the game's own)

// Eye-pass MSAA detection (vpvk_msaa_detected). Passes are observed when recorded; the eye pass is
// the largest colour+depth pass that matches the eye size the host gave (vpvk_taa_set_jitter, any
// enabled state), or without it: the largest multiview colour+depth pass, else after 120
// submissions the largest colour+depth pass seen. A larger qualifying pass later replaces it.
struct MsaaDetect {
  std::mutex m;
  uint64_t area = 0;          // committed eye pass (0 = none yet)
  int rank = 0;               // rule that committed it: 2 eye size, 1 multiview, 0 largest pass
  bool multiview = false;
  std::atomic<int> samples{0};  // its original sample count (0 = not seen)
  uint64_t candArea = 0;      // largest single-view candidate not committed yet
  uint8_t candSamples = 0;
  uint32_t candW = 0, candH = 0;
} gMsaa;

int roundSamples(int n) { return n >= 8 ? 8 : n >= 4 ? 4 : n >= 2 ? 2 : 0; }
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

// VPKTAA nativo: puerta (ajuste de la app / VPVK_TAA_NATIVE) y fase de jitter propia. La fase
// avanza en cada vpvk_taa_native_encode que codifica el pase; los envíos entre dos llamadas llevan la misma.
std::atomic<int> gTaaNative{0};
std::atomic<float> gRcas{-1.0f};  // vpvk_config_set_rcas; < 0 = no puesto
struct TaaNativeState {
  uint64_t phase = 0;
  uint32_t jittered = 0;            // draws movidos con esta fase
  uint32_t targetW = 0, targetH = 0;
} gTaaNat;  // gTaaMutex

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
// VPVK_PREWARM=0/1 (set by the app before the driver loads) wins: vpvk_config_set_prewarm can only
// reach the driver once it is loaded, after the first vkCreateDevice may already have prewarmed.
bool prewarmEnabled() {
  const char* e = getenv("VPVK_PREWARM");
  if (e && (*e == '0' || *e == '1')) return *e == '1';
  return gPrewarm.load() != 0;
}
// Quest 3 / Adreno 740 identity is opt-in (vpvk_config_set_quest_identity or VPVK_QUEST_IDENTITY=1). Read at every
// query, never latched: the host may call the setter after the library loaded.
bool questIdentity() {
  const char* e = getenv("VPVK_QUEST_IDENTITY");
  return (e && *e == '1') || gQuestIdentity.load() != 0;
}
bool asyncPipelinesOn() {
  const char* e = getenv("VPVK_ASYNC_PIPELINES");
  if (e && (*e == '0' || *e == '1')) return *e == '1';
  // The same for games and for vpgles: a GL draw that needs a pipeline still compiling is handled by the frame gate
  // (pipelines.cpp: soft for the eye image, hard and waited for everywhere else). The CTS run (VPVK_EXTRA_EXTENSIONS=1,
  // also the app's "full Vulkan profile" for games) must not decide this: it sets VPVK_ASYNC_PIPELINES=0 itself.
  return true;
}
// A pass of this size is (a band around) the game's eye target: the eye size from VPKTAA when known, else the
// detected MSAA eye pass. Same band as msaaObservePass. Unknown eye = false (`known` says which).
bool eyeSizedPass(uint32_t w, uint32_t h, bool* known) {
  if (known) *known = false;
  uint32_t eyeW = 0, eyeH = 0;
  {
    std::lock_guard<std::mutex> l(gTaaMutex);
    eyeW = gTaa.width, eyeH = gTaa.height;
  }
  if (eyeW && eyeH) {
    if (known) *known = true;
    return w * 2 >= eyeW && h * 2 >= eyeH && w <= eyeW * 2 && h <= eyeH * 2;
  }
  uint64_t area;
  {
    std::lock_guard<std::mutex> l(gMsaa.m);
    area = gMsaa.area;
  }
  if (known) *known = area != 0;
  return area && uint64_t(w) * h * 2 >= area && uint64_t(w) * h <= area * 4;
}
int anisotropyForced() {
  // VPVK_FORCE_ANISO (per game, set by the app): 0 = the game's own, 2/4/8/16 forced. When set it
  // wins over VPVK_ANISO / vpvk_config_set_anisotropy. Clamped to Adreno's maxSamplerAnisotropy (16).
  static const int force = [] {
    const char* e = getenv("VPVK_FORCE_ANISO");
    if (!e || !*e) return -1;
    const int v = atoi(e);
    const int lvl = v <= 1 ? 0 : std::min(v, 16);
    vpvk::logf("vk.cfg VPVK_FORCE_ANISO=%d -> anisotropy override %s", v, lvl ? "x" : "off (game's own)");
    return lvl;
  }();
  if (force >= 0) return force;
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
             "taa_native=%d taa_ui_mask=%d rcas=%.2f asw_off=%d quest3=%d anisotropy=%d prewarm=%d force_1x=%d msaa_override=%d msaa_device_max=%d fov_apply=%d fov_active=%d set_by_app=%s",
             gGen.load(), modeName(c.foveation.mode), c.foveation.sharp_zone, c.foveation.periphery, c.vpktaa.enabled,
             c.vpktaa.percent, c.vpktaa.sharpness, c.render_scale, taaNativeOn() ? 1 : 0, taaUiMaskOn() ? 1 : 0, double(rcasIntensity()), c.asw_disabled, c.quest3_profile,
             anisotropyForced(), prewarmEnabled() ? 1 : 0, forceSingleSample() ? 1 : 0, msaaOverride(), platformMaxSampleCount(),
             foveationApply() ? 1 : 0, foveationActive() ? 1 : 0, gGen.load() ? "yes" : "no(defaults)");
}
bool foveationApply() {
  static const int env = [] {
    const char* e = getenv("VPVK_FOVEATION");
    return e && (e[0] == '0' || e[0] == '1') && !e[1] ? e[0] - '0' : -1;
  }();
  return env >= 0 ? env == 1 : gFovApply.load() != 0;
}

FovConfig foveationConfig() {
  FovConfig c;
  std::lock_guard<std::mutex> l(gMutex);
  c.shape = fov::shapeOf(int(gCfg.foveation.mode), gCfg.foveation.sharp_zone, gCfg.foveation.periphery, gCfg.foveation.height);
  c.eyes = gFovEyes;
  c.apply = foveationApply();
  return c;
}

bool foveationActive() {
  const FovConfig c = foveationConfig();
  return c.apply && !c.shape.off && platformRateMapSupported();
}

FoveationState::FoveationState() {
  const char* e = getenv("VPVK_FOVEATION_UI");
  if (e && e[0] == '0' && !e[1]) uiSplit = false;
}

std::shared_ptr<const vpgpu::RateMap> FoveationState::mapFor(uint32_t w, uint32_t h, uint16_t layers) {
  if (layers != fov::kViews || !w || !h || !foveationActive()) return nullptr;
  const FovConfig c = foveationConfig();
  std::shared_ptr<vpgpu::RateMap> m = fov::buildRateMap(w, h, layers, c.shape, c.eyes);
  if (!m) return nullptr;
  if (cur_ && cur_->key == m->key) return cur_;
  const fov::Center c0 = c.eyes.known ? fov::opticalCenter(c.eyes.tangents[0]) : fov::Center{};
  const fov::Center c1 = c.eyes.known ? fov::opticalCenter(c.eyes.tangents[1]) : fov::Center{};
  logf("vk.fov plan logical=%ux%u views=%u zones=%ux%u sharp=%.2f periphery=%.2f height=%.2f kept=%.2f est_saving=%.0f%% "
       "centre0=%.3f,%.3f centre1=%.3f,%.3f eyes=%s ui_split=%d key=%016llx",
       w, h, unsigned(layers), unsigned(m->zonesX), unsigned(m->zonesY), double(c.shape.sharp), double(c.shape.periphery),
       double(c.shape.height), double(m->keptFraction), 100.0 * (1.0 - double(m->keptFraction)), double(c0.x), double(c0.y),
       double(c1.x), double(c1.y), c.eyes.known ? "known" : "centred", uiSplit ? 1 : 0, (unsigned long long)m->key);
  cur_ = m;
  return cur_;
}

bool taaNativeOn() {
  static const int env = [] {
    const char* e = getenv("VPVK_TAA_NATIVE");
    return e && (e[0] == '0' || e[0] == '1') && !e[1] ? e[0] - '0' : -1;
  }();
  return env >= 0 ? env == 1 : gTaaNative.load() != 0;
}

bool taaUiMaskOn() {
  static const int env = [] {
    const char* e = getenv("VPVK_TAA_UI_MASK");
    return e && (e[0] == '0' || e[0] == '1') && !e[1] ? e[0] - '0' : -1;
  }();
  return (taaNativeOn() || rcasOn()) && env != 0;  // por defecto 1 cuando el nativo o RCAS están activos
}

float rcasIntensity() {
  static const float env = [] {
    const char* e = getenv("VPVK_RCAS");
    return e && *e ? taa::rcasClamp(static_cast<float>(atof(e))) : -1.0f;
  }();
  if (env >= 0.0f) return env;
  const float set = gRcas.load();
  if (set >= 0.0f) return set;
  if (!taaNativeOn()) return 0.0f;
  std::lock_guard<std::mutex> l(gMutex);
  return taa::rcasClamp(gCfg.vpktaa.sharpness);
}
bool rcasOn() { return rcasIntensity() > 0.0f; }
bool msaaTileOn() {
  static const bool on = [] {
    const char* e = getenv("VPVK_MSAA_TILE");
    return !(e && e[0] == '0' && !e[1]);
  }();
  return on;
}
bool rcasDenoise() {
  static const bool on = [] {
    const char* e = getenv("VPVK_RCAS_DENOISE");
    return e && e[0] == '1' && !e[1];
  }();
  return on;
}

TaaJitter taaJitter() {
  if (taaEnvOff()) return {};
  std::lock_guard<std::mutex> l(gTaaMutex);
  TaaJitter j = gTaa;
  if (j.enabled && taaNativeOn() && !j.still) {  // el motor elige el jitter (RCAS sin TAA: sin offset)
    const auto h = taa::haltonJitter(gTaaNat.phase);
    j.jx = h[0], j.jy = h[1];
  }
  return j;
}

TaaNativeApplied taaNativeTake() {
  std::lock_guard<std::mutex> l(gTaaMutex);
  TaaNativeApplied a;
  const auto h = taa::haltonJitter(gTaaNat.phase);
  a.jx = h[0], a.jy = h[1];
  a.jittered = gTaaNat.jittered;
  a.targetW = gTaaNat.targetW, a.targetH = gTaaNat.targetH;
  a.phase = gTaaNat.phase;
  // La fase no avanza aquí: lo hace taaNativeAdvance si el pase se codificó (un frame sin escena
  // o un encode fallido no gastan fases).
  gTaaNat.jittered = 0;
  gTaaNat.targetW = gTaaNat.targetH = 0;
  return a;
}

void taaNativeAdvance(uint64_t phase) {
  std::lock_guard<std::mutex> l(gTaaMutex);
  if (gTaaNat.phase == phase) ++gTaaNat.phase;
}

void taaRecordSubmit(uint64_t submit, const TaaJitter& j, uint32_t jittered, uint32_t skipped, uint32_t targetW,
                     uint32_t targetH) {
  std::lock_guard<std::mutex> l(gTaaMutex);
  TaaApplied& a = gTaaRead;
  TaaLog& g = gTaaLog;
  a.skipped += skipped, g.skipped += skipped;
  if (jittered && taaNativeOn()) {
    gTaaNat.jittered += jittered;
    gTaaNat.targetW = std::max(gTaaNat.targetW, targetW), gTaaNat.targetH = std::max(gTaaNat.targetH, targetH);
  }
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

int msaaOverride() {
  const int v = gMsaaOverride.load();
  if (v >= 0) return v;
  static const int env = [] {
    const char* e = getenv("VPVK_MSAA");
    return e && *e ? roundSamples(atoi(e)) : 0;
  }();
  return env;
}

uint8_t msaaEffective(uint8_t game, bool force1x, int overrideSamples, int deviceMax) {
  if (game <= 1) return 1;  // the override never adds MSAA where the game asked for none
  if (force1x) return 1;
  int want = overrideSamples > 1 ? overrideSamples : game;
  const int max = deviceMax >= 1 ? deviceMax : 1;
  while (want > max && want > 1) want >>= 1;
  return static_cast<uint8_t>(want);
}

uint8_t effectiveSamples(uint8_t game) {
  if (game <= 1) return 1;
  const bool f1 = forceSingleSample();
  const int ov = msaaOverride(), max = platformMaxSampleCount();
  const uint8_t eff = msaaEffective(game, f1, ov, max);
  if ((eff != game || ov > 1) && !f1) {  // forced 1x has its own "vk.msaa forced 1x" lines
    // Once per (game, override, effective): settings may change between images.
    static std::mutex m;
    static uint32_t seen[32];
    static unsigned n = 0;
    const uint32_t k = uint32_t(game) | uint32_t(ov) << 8 | uint32_t(eff) << 16;
    std::lock_guard<std::mutex> l(m);
    bool found = false;
    for (unsigned i = 0; i < n; ++i) found |= seen[i] == k;
    if (!found && n < 32) {
      seen[n++] = k;
      logf("vk.msaa override %u (game %ux, device max %d)%s", unsigned(eff), unsigned(game), max,
           ov > 1 && eff < ov ? " clamped" : ov > 1 ? "" : " (game count unsupported: clamped)");
    }
  }
  return eff;
}

void msaaObservePass(uint8_t gameSamples, uint32_t w, uint32_t h, uint32_t viewMask, bool hasColor, bool hasDepth) {
  if (!hasColor || !hasDepth || w < 256 || h < 256) return;  // shadow maps (depth only), probes, UI bits
  if (gameSamples < 1) gameSamples = 1;
  uint32_t eyeW = 0, eyeH = 0;
  {
    std::lock_guard<std::mutex> l(gTaaMutex);
    eyeW = gTaa.width, eyeH = gTaa.height;
  }
  const bool mv = std::popcount(viewMask) >= 2;
  const uint64_t area = uint64_t(w) * h;
  MsaaDetect& g = gMsaa;
  std::lock_guard<std::mutex> l(g.m);
  int rank;  // how the pass qualifies: 2 eye size, 1 multiview, 0 only a candidate
  if (eyeW && eyeH) {
    // >= 50 % of the eye per axis (render scale), up to a double-wide target of both eyes.
    if (!(w * 2 >= eyeW && h * 2 >= eyeH && w <= eyeW * 2 && h <= eyeH * 2)) return;
    rank = 2;
  } else if (mv) {
    rank = 1;
  } else {
    if (area > g.candArea || (area == g.candArea && gameSamples > g.candSamples))
      g.candArea = area, g.candSamples = gameSamples, g.candW = w, g.candH = h;
    return;  // msaaSubmitTick commits it
  }
  const int before = g.samples.load();
  // Better: a stronger rule, a larger pass, or the same size with more samples (an eye-sized 1x
  // colour+depth pass next to the MSAA scene pass must not hide the MSAA).
  const bool better = !before || rank > g.rank || (rank == g.rank && (area > g.area || (area == g.area && gameSamples > before)));
  if (!better) return;
  g.area = area, g.rank = rank, g.multiview = mv;
  g.samples.store(gameSamples);
  if (!before)
    logf("vk.msaa detected eye=%ux size=%ux%u multiview=%d by=%s", unsigned(gameSamples), w, h, mv ? 1 : 0,
         rank == 2 ? "eye_size" : "multiview");
  else if (before != gameSamples)
    logf("vk.msaa detected eye=%ux (changed from %dx) size=%ux%u multiview=%d", unsigned(gameSamples), before, w, h,
         mv ? 1 : 0);
}

void msaaSubmitTick(uint64_t submits) {
  MsaaDetect& g = gMsaa;
  if (submits < 120) return;
  std::lock_guard<std::mutex> l(g.m);
  if (g.rank > 0 || !g.candArea) return;  // a multiview / eye-size rule already decided
  if (g.samples.load() && g.candArea <= g.area && g.candSamples <= g.samples.load()) return;
  const int before = g.samples.load();
  g.area = g.candArea;
  g.samples.store(g.candSamples);
  if (!before)
    logf("vk.msaa detected eye=%ux size=%ux%u multiview=0 by=largest_pass", unsigned(g.candSamples), g.candW, g.candH);
  else if (before != g.candSamples)
    logf("vk.msaa detected eye=%ux (changed from %dx) size=%ux%u multiview=0", unsigned(g.candSamples), before,
         g.candW, g.candH);
}

// While a VkDevice lives the answer is frozen: images (vkCreateImage) and shaders (texture2d vs
// texture2d_ms, SpirvLowerOptions::multisampleAsSingle) must agree on it, and a setting toggled
// mid-game would make Metal bind a 1x texture to a multisample slot (or the reverse).
namespace {
std::mutex gHoldMutex;
int gHoldDevices = 0, gHeld = -1;
}  // namespace

void holdForceSingleSample(bool acquire) {
  std::lock_guard<std::mutex> l(gHoldMutex);
  if (acquire) {
    if (gHoldDevices++ == 0) gHeld = forceSingleSampleLive() ? 1 : 0;
  } else if (gHoldDevices > 0 && --gHoldDevices == 0) {
    gHeld = -1;
  }
}

bool forceSingleSample() {
  {
    std::lock_guard<std::mutex> l(gHoldMutex);
    if (gHeld >= 0) return gHeld != 0;
  }
  return forceSingleSampleLive();
}

bool forceSingleSampleLive() {
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
  vpvk_foveation v = *f;
  if (v.mode < VPVK_FOVEATION_OFF || v.mode > VPVK_FOVEATION_CUSTOM) v.mode = VPVK_FOVEATION_OFF;
  v.sharp_zone = std::clamp(v.sharp_zone, 0.05f, 1.0f);
  v.periphery = std::clamp(v.periphery, 0.1f, 1.0f);
  v.height = std::clamp(v.height, -0.5f, 0.5f);
  {
    std::lock_guard<std::mutex> l(gMutex);
    gCfg.foveation = v;
    gGen.fetch_add(1);
  }
  const bool apply = vpvk::foveationApply();
  vpvk::logf("vk.fov config mode=%s sharp=%.2f periphery=%.2f height=%.2f applied=%s", modeName(v.mode), v.sharp_zone,
             v.periphery, v.height,
             v.mode == VPVK_FOVEATION_OFF ? "no(off)"
             : !apply                     ? "no(switch_off: vpvk_config_set_foveation_apply / VPVK_FOVEATION=1)"
                                          : "rate_map(eye passes: multiview, 1x, VPKTAA native)");
}

__attribute__((visibility("default"))) void vpvk_config_set_foveation_apply(int on) {
  gFovApply.store(on ? 1 : 0);
  gGen.fetch_add(1);
  vpvk::logf("vk.fov apply=%d%s", on ? 1 : 0, getenv("VPVK_FOVEATION") ? " (env VPVK_FOVEATION wins)" : "");
}

__attribute__((visibility("default"))) void vpvk_config_set_foveation_eyes(const float t[8]) {
  if (!t) return;
  vpvk::fov::Eyes e;
  e.known = true;
  for (int i = 0; i < 2; ++i)
    for (int k = 0; k < 4; ++k) {
      const float v = t[i * 4 + k];
      if (!(v > 0.0f) || !std::isfinite(v)) e.known = false;
      e.tangents[size_t(i)][size_t(k)] = v;
    }
  std::lock_guard<std::mutex> l(gMutex);
  const bool changed = e.known != gFovEyes.known || (e.known && e.tangents != gFovEyes.tangents);
  gFovEyes = e;
  if (changed) {
    vpvk::logf("vk.fov eyes %s tangents l/r/t/b eye0=%.3f/%.3f/%.3f/%.3f eye1=%.3f/%.3f/%.3f/%.3f", e.known ? "known" : "invalid(ignored)",
               double(t[0]), double(t[1]), double(t[2]), double(t[3]), double(t[4]), double(t[5]), double(t[6]), double(t[7]));
    gGen.fetch_add(1);
  }
}

__attribute__((visibility("default"))) int vpvk_foveation_active(void) { return vpvk::foveationActive() ? 1 : 0; }

__attribute__((visibility("default"))) void vpvk_config_set_vpktaa(const vpvk_vpktaa* t) {
  if (!t) return;
  std::lock_guard<std::mutex> l(gMutex);
  vpvk_vpktaa v = *t;
  v.enabled = v.enabled ? 1 : 0;
  v.percent = std::clamp(v.percent, 0.0f, 100.0f);
  v.sharpness = std::clamp(v.sharpness, 0.0f, 1.0f);
  gCfg.vpktaa = v;
  gGen.fetch_add(1);
  vpvk::logf("vk.cfg vpktaa enabled=%d percent=%.0f sharpness=%.2f applied=%s", v.enabled,
             v.percent, v.sharpness, vpvk::taaNativeOn() ? "native(vpvk_taa_native_encode)" : "off(no native pass: the host leaves the frame untouched)");
}

__attribute__((visibility("default"))) void vpvk_config_set_taa_native(int on) {
  gTaaNative.store(on ? 1 : 0);
  gGen.fetch_add(1);
  vpvk::logf("vk.cfg taa_native=%d effective=%d (VPVK_TAA_NATIVE env wins)", on ? 1 : 0, vpvk::taaNativeOn() ? 1 : 0);
}

__attribute__((visibility("default"))) void vpvk_config_set_rcas(float intensity) {
  gRcas.store(vpvk::taa::rcasClamp(intensity));
  gGen.fetch_add(1);
  vpvk::logf("vk.cfg rcas intensity=%.2f stops=%.2f effective=%.2f denoise=%d (VPVK_RCAS env wins)", double(vpvk::taa::rcasClamp(intensity)),
             double(2.0f * (1.0f - vpvk::taa::rcasClamp(intensity))), double(vpvk::rcasIntensity()), vpvk::rcasDenoise() ? 1 : 0);
}
__attribute__((visibility("default"))) float vpvk_rcas_intensity(void) { return vpvk::rcasIntensity(); }

__attribute__((visibility("default"))) int vpvk_taa_native_enabled(void) { return vpvk::taaNativeOn() ? 1 : 0; }
__attribute__((visibility("default"))) int vpvk_taa_ui_mask_enabled(void) { return vpvk::taaUiMaskOn() ? 1 : 0; }

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

__attribute__((visibility("default"))) void vpvk_config_set_quest_identity(int on) {
  gQuestIdentity.store(on ? 1 : 0);
  vpvk::logf("vk.cfg quest_identity=%d effective=%d (VPVK_QUEST_IDENTITY=1 env also turns it on)", on ? 1 : 0,
             vpvk::questIdentity() ? 1 : 0);
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

__attribute__((visibility("default"))) void vpvk_config_set_msaa_override(int samples) {
  const int v = roundSamples(samples);
  gMsaaOverride.store(v);
  gGen.fetch_add(1);
  vpvk::logf("vk.cfg msaa_override=%d (0 = game's own; device max %d; images, passes and pipelines created "
             "afterwards)", v, vpvk::platformMaxSampleCount());
}

__attribute__((visibility("default"))) int vpvk_msaa_detected(void) { return gMsaa.samples.load(); }

__attribute__((visibility("default"))) int vpvk_msaa_device_max(void) { return vpvk::platformMaxSampleCount(); }

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
  t.still = j->still != 0;
  bool changed;
  {
    std::lock_guard<std::mutex> l(gTaaMutex);
    changed = t.enabled != gTaa.enabled || t.width != gTaa.width || t.height != gTaa.height || t.still != gTaa.still;
    gTaa = t;
  }
  // Called once per frame: only an on/off or eye-size change is logged.
  if (changed)
    vpvk::logf("vk.taa set enabled=%d eye=%ux%u still=%d env_off=%d", t.enabled ? 1 : 0, t.width, t.height,
               t.still ? 1 : 0, taaEnvOff() ? 1 : 0);
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
