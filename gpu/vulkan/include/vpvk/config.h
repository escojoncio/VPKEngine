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

// ---- MSAA of the game (automatic AA mode, KL_AA_MODE) ----
// Original sample count (before forcing to 1x or overriding) of the game's eye pass: the largest
// render pass / dynamic rendering with colour and depth whose colour is the eye's size (>= 50 % of
// the eye per axis; multiview counts). 0 = not seen yet, 1 = no MSAA, 2/4/8 = MSAA. Log once:
// "vk.msaa detected eye=Nx" (again with "changed from" if a larger pass says otherwise).
int vpvk_msaa_detected(void);
// Largest sample count the MTLDevice renders to (supportsTextureSampleCount probed 8, 4, 2): 1/2/4/8.
int vpvk_msaa_device_max(void);
// Sample-count override, only where the game asked for > 1 sample (images, render-pass / dynamic
// rendering attachments, pipelines' rasterizationSamples; resolve targets stay 1x): 0 = the game's
// own, 2/4/8. Clamped down to vpvk_msaa_device_max ("vk.msaa override N (game Mx, device max K)").
// Forcing to 1x (vpvk_config_set_force_single_sample / VPVK_FORCE_1X) wins over it. Env VPVK_MSAA
// is used until this is called. Applies to images and pipelines created afterwards; draws always
// use a pipeline whose sample count matches the pass they land in.
void vpvk_config_set_msaa_override(int samples);
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

// ---- VPKTAA inline: sub-pixel jitter applied by the driver, resolve on the driver's own queue ----
// The jitter moves the viewport of the eye's geometry draws when vkQueueSubmit translates them (not
// when they are recorded: command buffers may be reused), so the caller learns exactly which jitter
// each submission got. Env VPVK_TAA_JITTER=0 disables it whatever is set here (A/B tests).
typedef struct {
  int enabled;              // 0 = no jitter (the default until someone sets 1)
  uint32_t width, height;   // eye size in texels; 0,0 = only multiview passes are jittered
  float jx, jy;             // offset in pixels of the jittered target, [-0.5, 0.5];
                            // +x = right, +y = down in framebuffer coordinates (= Metal texels)
  uint64_t tag;             // caller's opaque value (Klepton: frame number)
} vpvk_taa_jitter;
// Thread-safe; applies to the submissions translated after the call.
void vpvk_taa_set_jitter(const vpvk_taa_jitter* j);

typedef struct {
  uint64_t tag;                 // tag of the jitter of the last submission that jittered >= 1 draw
  float jx, jy;                 // that jitter (pixels of the target)
  uint32_t target_w, target_h;  // size of the jittered target (largest seen)
  uint32_t jittered_draws, skipped_draws;  // since the last read (skipped: same target, excluded)
  uint32_t submits;             // submissions with >= 1 jittered draw since the last read
  uint32_t mixed;               // of those, the ones whose jitter differs from the last one
} vpvk_taa_applied;
// 1 if any draw was jittered since the last read (out filled either way); resets the counters.
int vpvk_taa_get_applied(vpvk_taa_applied* out);

// Native work on the queue that runs every guest submission: a fresh id<MTLCommandBuffer>
// (unretained) is handed to `fn`, ordered on the GPU after everything vkQueueSubmit has accepted so
// far (including frames held while their pipelines compile: the call waits for those to be
// committed) and before any later submission; then it is committed. Synchronous on the calling
// thread; `fn` must only encode, ending every encoder it opens, and must not commit the command
// buffer or call vkQueueSubmit (same lock: it would deadlock). vkQueueWaitIdle /
// vkDeviceWaitIdle also wait for these command buffers. Returns 1 if committed, 0 if there is no
// device/queue (no VPVulkan device yet, or a platform without Metal).
typedef void (*vpvk_native_encode_fn)(void* mtl_command_buffer, void* user);
int vpvk_native_submit(vpvk_native_encode_fn fn, void* user);
// id<MTLDevice> of the live VPVulkan device (unretained), or NULL.
void* vpvk_native_device(void);

// MoltenVK-compatible export (also through vkGetInstanceProcAddr / vkGetDeviceProcAddr): the
// id<MTLTexture> (unretained) behind a bound VkImage, NULL if it has none. Works for any image,
// exportable or not (Klepton's overlay and shadow textures). Signature (not declared here so this
// header never clashes with MoltenVK's mvk_deprecated_api.h):
//   void vkGetMTLTextureMVK(VkImage image, void** mtl_texture /* id<MTLTexture>* */);

#ifdef __cplusplus
}
#endif
