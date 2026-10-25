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
  float sharpness;   // 0..1: RCAS intensity while VPKTAA runs, unless vpvk_config_set_rcas was called
} vpvk_vpktaa;

typedef struct vpvk_config {
  vpvk_foveation foveation;
  vpvk_vpktaa vpktaa;
  float render_scale;  // eye render resolution relative to the compositor drawable (0.5..1.5)
  int asw_disabled;    // 1: every frame real (no SpaceWarp / extrapolation)
  int quest3_profile;  // 1: present as Quest 3 (games pick their Quest 3 variant)
} vpvk_config;

void vpvk_config_set_foveation(const vpvk_foveation* f);
// Foveation rate map: apply the settings above (1) or leave the frame as it is today (0, the default: not
// proven on a device yet). Env VPVK_FOVEATION=0/1 wins. Needs VPKTAA native (the eye pass registry) and
// single-sample eye passes; takes effect for pipelines created afterwards (their multiview vertex stages write
// [[viewport_array_index]] and their fragment stages map [[position]] back to screen space, see docs/VULKAN.md
// «Foveado propio»), so call it before the game creates its pipelines.
void vpvk_config_set_foveation_apply(int on);
// Frustum tangents of both eyes as the game renders them (left, right, top, bottom, positive; eye 0 then eye 1):
// the sharp zone is centred on each eye's optical centre l/(l+r), t/(t+b). Optional: without it, the middle.
void vpvk_config_set_foveation_eyes(const float tangents[8]);
// 1 when the rate map is really in use (apply on, mode not off, the GPU supports rate maps with 2 layers).
int vpvk_foveation_active(void);
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
  int still;                // 1: identify the eye passes and keep the UI mask, but apply NO sub-pixel offset
                            // (RCAS without TAA). Appended field: hosts built before it must not call this build.
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
// GPU backlog, cheap, any thread, never blocks on a submit: age in ms of the oldest frame command
// buffer still in flight (0 idle / no device), the number of hang-split cuts (attachments stored and
// reloaded mid-encoder) in the newest submitted frame (0 = none), and 1 if the device is lost. A host skips optional
// native work (vpvk_native_submit) while the backlog is high: that work orders itself after every
// frame in flight.
unsigned vpvk_gpu_backlog_ms(void);
unsigned vpvk_gpu_last_frame_chunks(void);
int vpvk_gpu_lost(void);
// Memory pressure: frees rebuildable GPU memory (idle PSOs / libraries, staging, arg rings). level 1 = OS
// warning, 2 = critical. Writes a short key=value report into out (NUL-terminated, cut at cap; may be
// NULL/0) and returns the bytes freed; 0 without a live device. Any thread.
size_t vpvk_trim_memory(int level, char* out, size_t cap);
// Releases what the guest destroyed that the backend keeps until a frame reuses a slot (the graveyard, and the
// leftovers of frames that completed), without waiting for the GPU to go idle; min_idle_ms no longer gates it.
// Any thread, never blocks (the submit mutex is try-locked: a busy device returns 0). Short key=value report into
// out; returns the bytes released.
size_t vpvk_idle_drain(unsigned min_idle_ms, char* out, size_t cap);
// Diagnostics: one line "dt=<ms> key=<delta> ..." of what changed since the previous call with the same `which`
// (0: guest calls into the Vulkan layer: alloc<type>/free<type> (+_b bytes), buf_create/destroy/bind, img_create
// (+_px texels)/destroy/bind, map/unmap/flush/inval (+_b bytes); 1: the backend: buf_new/nocopy/int, tex_new,
// tex_views, cb, enc_*, res_add/remove/commit, drained*, and the gauges g_held / g_held_b = retired objects still
// pinned and their bytes). Only the keys that moved. Returns the length; 0 (out untouched) if nothing moved.
size_t vpvk_diag_line(int which, char* out, size_t cap);
// Memory ownership for the host's region dump: ranges of kind 0 = the live host-visible VkDeviceMemory host allocations
// (what the guest maps), 1 = the CPU-visible buffers VPKEngine itself allocated. Fills out[0..max) sorted by address
// and returns the total (may exceed max). Any thread; short mutex only.
typedef struct vpvk_range {
  uintptr_t addr;
  uint64_t size;
} vpvk_range;
size_t vpvk_diag_ranges(int kind, vpvk_range* out, size_t max);
// Memory heartbeat: writes one line of space-separated key=value into out (NUL-terminated, cut at
// cap) and returns its length; 0 (out untouched) without a live device. Cheap, any thread. Keys:
// alloc (MTLDevice.currentAllocatedSize) ws_max (recommendedMaxWorkingSetSize) tex_astc tex_color
// tex_depth tex_msaa (live MTLTexture bytes by kind) buf_own (MTLBuffer bytes allocated) buf_guest
// (guest memory wrapped no-copy) pso fn lib (live pipelines / functions / libraries) lib_compiled
// ar_entries ar_files (archive descriptors held / .metalar files open) tile_rejected (cumulative)
// msaa_memoryless (MSAA targets without memory now: bytes they would take) recipes_bytes recipes_pipes
// recipes_shaders (recipe store), then " ledger: " and the texture memory by owner with the 8 largest
// textures (resident/msaa_mat/rt_mat/memoryless MiB/count, msaa_materialized/released counts,
// top=index:WxHxL/sN/format/kind/MiB). Bytes are bytes except in the ledger (MiB) and attr_*. 4096 bytes hold it all.
// Unattributed alloc = alloc - (tex_* + buf_own + buf_guest) is split with: held_b (retired objects waiting for their
// frames), views (live texture views, a count), args_b (argument ring + chained argument buffers), buf_int_b (internal
// staging buffers), taa_b (native VPKTAA textures), fov_b (foveation shadow textures; already inside tex_*, informational). All bytes.
// Live Metal objects by creation site (metal2/mtl_attr.h), before " ledger: ": attr_tracked (bytes of live MTLBuffer / MTLTexture
// tracked, to compare with alloc), attr_devs (distinct MTLDevices seen; attr_dev_alloc lists each one's alloc in MiB when > 1),
// attr_top=tag:MiB/live (12 largest tags), attr_obj=tag:~MiB/live (pipelines / libraries: allocation growth seen around their
// creation, cumulative and approximate), attr_cnt=tag:live (views, functions). Allocations >= 32 MiB log gpu.bigalloc at once.
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

// Marca un VkImage como imagen de la capa de ojo (las del swapchain que el juego redibuja cada frame). La puerta de
// pipelines asíncronos (vk.pipeline_skip) solo descarta draws cuyo destino de color es una de estas imágenes; cualquier
// otro destino (RT de un panel, RT persistente, pases sin color) espera a su pipeline. Llamar al crear la imagen (antes o
// después del bind). Sin ninguna marca, todos los pases esperan. 1 = marcada con textura, 0 = sin dispositivo/imagen o
// aún sin textura (la marca se aplica al bind).
int vpvk_mark_eye_image(void* vk_image);

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
// The R8 UI mask of the newest eye passes, for a host that composites (and sharpens) the eye itself: 1 = UI/2D
// coverage (menus, HUD, text), 0 = 3D. NULL where this eye's last submission drew no UI, the mask is off
// (vpvk_taa_ui_mask_enabled() == 0) or there is no device. Returns an id<MTLTexture> RETAINED for the caller
// (+1: release it, e.g. CFRelease / Unmanaged.takeRetainedValue). *w / *h (optional): its size, which is the eye
// texture's own (same pass, same orientation: sample it with the eye's uv). *slice (optional): the array slice to
// read (a multiview pass holds both eyes in one 2-layer texture: slice = eye; otherwise 0; a 1-layer mask is a plain
// 2D texture). Any thread, never blocks on a submit. The mask is not frame-locked to the eye picture the host shows:
// it is that of the newest submission (a UI one frame newer is a one-frame mismatch, not corruption).
void* vpvk_ui_mask_texture(int eye, int* w, int* h, int* slice);

// ---- RCAS: el afilado del pipeline (taa_native.h «RCAS»), siempre activo si la intensidad > 0 ----
// Un pase propio (vpktaa_native_rcas) sobre el 3D del ojo, en TODOS los modos de AA (VPKTAA, MSAA del
// juego, Off) y rutas (Vulkan, vpgles; ovrp, OpenXR): con VPKTAA es su último paso; sin él, el host lo
// llama por el mismo gancho de fin de frame con `rcas_only` = 1 (vpvk_taa_native_encode). La UI/2D queda
// intacta (máscara R8; con MSAA del juego no hay máscara y se afila todo el ojo).
// intensity 0..1 (0 = pase apagado, sin coste). Env VPVK_RCAS=0..1 gana. Sin llamada: sigue a
// vpvk_vpktaa.sharpness mientras corre VPKTAA y 0 sin él. VPVK_RCAS_DENOISE=1 activa el denoise de FSR.
void vpvk_config_set_rcas(float intensity);
// Intensidad efectiva ahora (env > llamada > nitidez de VPKTAA con VPKTAA nativo; 0 = apagado).
float vpvk_rcas_intensity(void);

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
  int rcas_only;          // 1: sin TAA este frame: solo RCAS sobre `color` (sin resolve, historia, profundidad
                          // ni movimiento). Campo añadido al final.
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
