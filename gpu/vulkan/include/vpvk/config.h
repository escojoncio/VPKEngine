// VPVulkan configuration set by the host app (VisionQuestPro: in-app settings, never text files).
// Plain C so Klepton (C) and Swift (bridging header) can call it. Thread-safe; values take effect at
// the next frame the driver builds. The game's own foveation (VK_EXT_fragment_density_map) is never
// advertised: these settings are the only foveation and AA the driver applies. See docs/VULKAN.md.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum vpvk_foveation_mode {
  VPVK_FOVEATION_OFF = 0,
  VPVK_FOVEATION_M2 = 1,      // preset tuned for Vision Pro M2
  VPVK_FOVEATION_M5 = 2,      // preset tuned for Vision Pro M5 (app only offers it on M5)
  VPVK_FOVEATION_CUSTOM = 3,  // uses the three values below
} vpvk_foveation_mode;

typedef struct vpvk_foveation {
  vpvk_foveation_mode mode;
  float sharp_zone;  // custom: radius of the full-rate zone, fraction of the half-extent (0.05..1)
  float periphery;   // custom: rate at the edge (0.1..1, 1 = no reduction)
  float height;      // custom: vertical centre offset of the sharp zone (-0.5..0.5, + = up)
} vpvk_foveation;

typedef struct vpvk_vpktaa {
  int enabled;       // 1: game MSAA removed, VPKTAA final pass before the compositor
  float percent;     // history weight 0..100 (PTVisionPro "porcentaje")
  float sharpness;   // 0..1
} vpvk_vpktaa;

typedef struct vpvk_config {
  vpvk_foveation foveation;
  vpvk_vpktaa vpktaa;
  float render_scale;  // eye render resolution relative to the compositor drawable (0.5..1.5)
  int asw_disabled;    // 1: every frame real (no SpaceWarp / extrapolation)
  int quest3_profile;  // 1: present as Quest 3 (games pick their Quest 3 variant)
} vpvk_config;

void vpvk_config_set_foveation(const vpvk_foveation* f);
void vpvk_config_set_vpktaa(const vpvk_vpktaa* t);
void vpvk_config_set_render_scale(float scale);
void vpvk_config_set_asw_disabled(int disabled);
void vpvk_config_set_quest3_profile(int on);
// Forced sampler anisotropy for linear+mipmapped, non-comparison samplers (0 = game's own, max 16).
// Default 16. Takes effect for samplers created afterwards.
void vpvk_config_set_anisotropy(int max_anisotropy);
// Render the game's multisampled targets at 1x (VPKTAA does the AA). -1 (default) = follow
// vpktaa.enabled; 0/1 = explicit. Env VPVK_FORCE_1X=0/1 overrides both. Applies to images,
// render passes and pipelines created afterwards; the game still sees its sample counts.
void vpvk_config_set_force_single_sample(int on);
// Snapshot of the current configuration (clamped values). Never null-filled: defaults are
// foveation OFF, VPKTAA off (50 %, 0.5), scale 1, ASW disabled, Quest 3 profile on.
void vpvk_config_get(vpvk_config* out);
// Monotonic counter bumped by every setter: the driver re-reads the snapshot when it changes.
uint32_t vpvk_config_generation(void);

// Shader/pipeline precompilation (recipes stored per game in VP_PIPELINE_CACHE_DIR).
// set_prewarm(0) disables the prewarm at vkCreateDevice (recipes are still recorded).
void vpvk_config_set_prewarm(int on);
// Number of stored pipeline recipes for the current VP_PIPELINE_CACHE_DIR (0 = none).
uint32_t vpvk_prewarm_count(void);
// Compiles every stored recipe on all cores, before the game runs, and saves the Metal binary
// archive so the game's own prewarm at vkCreateDevice is mostly lookups. Blocking; progress is
// called from the calling thread. Returns the number of recipes (0 = nothing to do).
typedef void (*vpvk_prewarm_progress)(uint32_t done, uint32_t total, void* user);
uint32_t vpvk_prewarm(vpvk_prewarm_progress progress, void* user);

#ifdef __cplusplus
}
#endif
