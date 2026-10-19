// VPVulkan configuration set by the host app (VisionQuestPro: in-app settings, never text files).
// Plain C so Klepton (C) and Swift (bridging header) can call it. Thread-safe; values take effect at
// the next frame the driver builds. The game's own foveation (VK_EXT_fragment_density_map) is never
// advertised: these settings are the only foveation and AA the driver applies. See docs/VULKAN.md.
#pragma once

#include <stddef.h>
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
// Ahead-of-time build (gate: env VPVK_AOT=1, off by default; else returns 0): translates every
// <spirv_dir>/*.spv to MSL and Metal libraries, compiles every stored recipe and every compute
// variant, and serializes them all into the MTLBinaryArchive of VP_PIPELINE_CACHE_DIR, plus
// aot.manifest. Blocking, all cores. spirv_dir may be null (recipes only).
uint32_t vpvk_aot_build(const char* spirv_dir, vpvk_prewarm_progress progress, void* user);

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
// Memory heartbeat: writes one line of space-separated key=value into out (NUL-terminated, cut at
// cap) and returns its length; 0 (out untouched) without a live device. Cheap, any thread. Keys:
// alloc (MTLDevice.currentAllocatedSize) ws_max (recommendedMaxWorkingSetSize) tex_astc tex_color
// tex_depth tex_msaa (live MTLTexture bytes by kind) buf_own (MTLBuffer bytes allocated) buf_guest
// (guest memory wrapped no-copy) pso fn lib (live pipelines / functions / libraries) lib_compiled
// ar_entries ar_files (archive descriptors held / .metalar files open) tile_rejected (cumulative)
// recipes_bytes recipes_pipes recipes_shaders (recipe store). Bytes are bytes.
size_t vpvk_mem_stats(char* out, size_t cap);
// Imágenes nuevas a cero (VPVK_ZERO_NEW_IMAGES): se limpian de forma perezosa en el siguiente submit
// que las usa. Un host que lee una textura fuera del flujo Vulkan (el compositor) pregunta antes con el
// id<MTLTexture> que obtuvo (vkGetMTLTextureMVK / vkExportMetalObjectsEXT). Barata, sin esperar submits.
// 1 = aún con memoria sin inicializar; 0 = lista o desconocida.
int vpvk_image_zero_pending(void* mtl_texture);
// La limpia ya, en la cola del motor (ordenada tras lo aceptado y antes de lo posterior); solo si sigue
// pendiente. 1 = resuelta (o ya no hacía falta), 0 = desconocida, -1 = un submit tiene el cerrojo
// (no espera: reintentar el siguiente frame).
int vpvk_image_zero_clear(void* mtl_texture);

// ---- VPKTAA nativo: el TAA como pase interno del motor (taa_native.h / vpktaa_native.metal) ----
// Es el ÚNICO VPKTAA (el host ya no tiene otro). Apagado por defecto en el motor; la app lo enciende
// siempre que el VPKTAA esté permitido: vpvk_config_set_taa_native(1) o la variable VPVK_TAA_NATIVE=1 (gana sobre la llamada: 0/1). Encendido:
//   * el jitter lo elige el motor: Halton(2,3) de 8 fases, una fase por frame resuelto; los jx/jy
//     de vpvk_taa_set_jitter se ignoran (enabled y el tamaño de ojo siguen mandando);
//   * el host llama a vpvk_taa_native_encode dentro de su callback de vpvk_native_submit (el
//     command buffer va justo después del frame del juego, antes del siguiente).
int vpvk_taa_native_enabled(void);
// 1 si el pase nativo usa la máscara reactiva de UI (R8: la UI/2D queda nítida, solo se suaviza el 3D);
// VPVK_TAA_UI_MASK=0 la apaga. 0 con el nativo apagado.
int vpvk_taa_ui_mask_enabled(void);
void vpvk_config_set_taa_native(int on);

typedef struct {
  void* color;            // id<MTLTexture> 2D o 2DArray: lo que dibujó el juego (1 muestra)
  uint32_t color_slice;
  void* depth;            // opcional: id<MTLTexture> de profundidad del juego (MSAA o no), sin copia; nulo: la del pase de ojo
                          // que el motor vio dibujar `color` (y su convención: `reversed_z` del frame se ignora)
  uint32_t depth_slice;
  void* motion;           // opcional: vectores de SpaceWarp (rg = actual - anterior, NDC)
  uint32_t motion_slice;
  float motion_scale[2];  // texel de movimiento -> uv del recorte (Unity AppSW: 0.5, 0.5)
  float orientation[4];   // mundo <- ojo, xyzw (ejes OpenXR: -z al frente)
  float position[3];      // metros
  float tangents[4];      // l r t b del frustum SIN jitter (positivas)
  float crop[4];          // rect del ojo en el color: tamaño xy, origen zw (fracciones; 1,1,0,0 = todo)
  void* color_base;       // opcional: id<MTLTexture> de la que `color` es una vista (un 2D envuelto como 2DArray); con
                          // `depth` nulo el motor busca el pase de ojo que dibujó ese color (la profundidad del Translator)
} vpvk_taa_native_eye;

typedef struct {
  vpvk_taa_native_eye eye[2];
  void* output;           // id<MTLTexture> 2DArray, 2 capas (0 izq., 1 der.), tamaño del color,
                          // escribible (shaderWrite); formato unorm/float
  int output_srgb;        // 1: el kernel codifica sRGB (output = vista unorm de una imagen sRGB)
  int reversed_z;         // profundidad invertida (clear 0)
  float near_z, far_z;    // de la proyección del juego; far 0 = infinito
  int reset;              // 1: descarta la historia (corte de cámara, carga)
} vpvk_taa_native_frame;

// Codifica el pase en `mtl_command_buffer` (id<MTLCommandBuffer>): resolve de ambos ojos (un
// dispatch si comparten textura) + RCAS a `output`. Usa el jitter que el motor aplicó a los envíos
// desde la llamada anterior y avanza la fase. Mide el tiempo de GPU del command buffer y lo escribe
// en el log ("vk.taa native gpu_ms ..."). 1 = codificado; 0 = apagado, entradas inválidas o sin Metal.
int vpvk_taa_native_encode(const vpvk_taa_native_frame* frame, void* mtl_command_buffer);

// Profundidad MSAA -> depth32Float de 1 muestra (2DArray, mismas capas), propiedad del motor y
// reutilizada por `key` (p. ej. etapa de swapchain): mode 1 = la muestra más cercana, 0 = la muestra
// 0. Se codifica en `mtl_command_buffer`. Devuelve la textura (unretained) o NULL.
void* vpvk_depth_resolve_encode(void* mtl_command_buffer, void* depth_ms, int reversed_z, int mode, int key);

// MoltenVK-compatible export (also through vkGetInstanceProcAddr / vkGetDeviceProcAddr): the
// id<MTLTexture> (unretained) behind a bound VkImage, NULL if it has none. Works for any image,
// exportable or not (Klepton's overlay and shadow textures). Signature (not declared here so this
// header never clashes with MoltenVK's mvk_deprecated_api.h):
//   void vkGetMTLTextureMVK(VkImage image, void** mtl_texture /* id<MTLTexture>* */);

#ifdef __cplusplus
}
#endif
