// VPVulkan internals: the Vulkan objects of VPKEngine's own driver, written from the Vulkan
// specification on top of gpu/core. Platform-neutral; the Metal glue sits behind vpvk::Platform.
#pragma once

#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_metal.h>
#include <vulkan/vk_icd.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vpgpu/core/device.h"
#include "vpvk/spirv_frontend.h"
#include "vpvk/platform.h"
#include "recipes.h"
#include "translate_pool.h"
#include "vpvk/ycbcr.h"
#include "taa_native.h"
#include "foveation.h"
#include "perf.h"

namespace vpvk {

// Every log line of the driver starts with "vk." (one grep finds them all in the visor log).
void log(const std::string& line);
// App-forced sampler anisotropy (0/1 = off); see vpvk_config_set_anisotropy.
int anisotropyForced();
// Whether a sampler takes the forced anisotropy (linear, mipmapped, wrapping; not shadow/unnormalized).
bool samplerAnisoEligible(const VkSamplerCreateInfo& ci);
void logConfig();  // config.cpp: "vk.cfg effective ..." (called from vkCreateDevice)
void logf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
bool prewarmEnabled();  // config.cpp: vpvk_config_set_prewarm
// VPVK_ASYNC_PIPELINES: default (unset, or 1) = asynchronous: vkCreate*Pipelines returns at once, translation + compile
// run in the background and draws whose pipeline is not ready are skipped within a small budget (pipelines.cpp
// "Asynchronous creation"). =0 is the opt-in synchronous mode: vkCreate*Pipelines returns with the pipelines compiled
// (pipelines.cpp "Synchronous creation"). The internal GLES device (internalClient()) ignores it: asynchronous unless
// VPVK_GLES_SYNC_PIPELINES=1. Decided at vkCreateDevice.
bool asyncPipelinesOn();
// config.cpp: robustBufferAccess in the translated shaders (VPVK_ROBUST: unset = on, "0" = off, "app" = what the app
// enabled at vkCreateDevice, on while there is no device). robustMode() is "default", "0" or "app" as of the last
// refreshRobustMode(); the app's value is -1 until vkCreateDevice sets it (not for the internal GLES device).
const char* robustMode();
int refreshRobustMode();  // reads VPVK_ROBUST again (vkCreateDevice, start of prewarm / precompile); returns the mode
void setAppRobustBufferAccess(bool on);
int appRobustBufferAccess();
bool robustBuffersEffective();
// config.cpp: harness dump (VPVK_DUMP_MSL / VPVK_DUMP_MSL_FILTER, read at every call): true = write shader `id` into `dir`.
bool dumpDirFor(uint64_t id, std::string& dir);
// Size of the most recent eye image the host marked (Device::eyeSize, vpvk_mark_eye_image): width << 40 | height << 16 | layers
// (width and height 24 bits each); 0 = none marked.
constexpr uint64_t packEyeSize(uint32_t w, uint32_t h, uint32_t layers) {
  return uint64_t(w & 0xffffffu) << 40 | uint64_t(h & 0xffffffu) << 16 | (layers & 0xffffu);
}
constexpr uint32_t eyeSizeW(uint64_t p) { return uint32_t(p >> 40); }
constexpr uint32_t eyeSizeH(uint64_t p) { return uint32_t(p >> 16 & 0xffffffu); }
// config.cpp: the pass size is the game's eye target (band of msaaObservePass); false while the eye is unknown. The eye size is
// the VPKTAA jitter's, else `markedEye` (packEyeSize of the host-marked eye image), else the detected MSAA eye pass.
bool eyeSizedPass(uint32_t w, uint32_t h, bool* known = nullptr, uint64_t markedEye = 0);
bool forceSingleSample();  // config.cpp: vpvk_config_set_force_single_sample; frozen while a device lives
bool forceSingleSampleLive();
void holdForceSingleSample(bool acquire);  // vkCreateDevice / vkDestroyDevice (also freezes msaaForce)
// Forced MSAA of the game's 1x scene passes (VPVK_MSAA_FORCE / vpvk_config_set_msaa_force): 0 = off, else 2 / 4 (clamped to
// platformMaxSampleCount). Frozen while a VkDevice lives (holdForceSingleSample); msaaForceLive reads the setting now.
int msaaForce();
int msaaForceLive();
bool gameMsaaAuto();  // VPVK_GAME_MSAA_AUTO (default on), frozen like msaaForce
bool gameMsaaAutoLive();

// ---- MSAA: detection of the game's eye pass and the sample-count override (config.cpp) ----
// Largest sample count the Metal device renders to (8/4/2 probed with supportsTextureSampleCount;
// 1 = none). Defined by the platform source (platform_metal.mm / platform_null.cpp), cached.
int platformMaxSampleCount();
// MTLDevice.name of the system GPU (the physical device's deviceName without the Quest identity), and whether it
// samples BCn textures (supportsBCTextureCompression). Platform source (platform_metal.mm / platform_null.cpp), cached.
// The null platform (Linux tests) has no GPU: a fixed name, BC supported.
std::string platformDeviceName();
bool platformSupportsBC();
// Live Metal objects by creation site (" attr_tracked=... "), walked outside any device lock; "" without Metal.
std::string platformAttrSummary();
// Pure rule (exposed for tests): the sample count a target or pipeline the game asked `game`
// samples for is created with. game <= 1 stays 1 (the override never adds MSAA); force1x wins;
// otherwise the override (2/4/8, 0 = the game's own), clamped down to the device maximum.
uint8_t msaaEffective(uint8_t game, bool force1x, int overrideSamples, int deviceMax);
// msaaEffective with the live configuration; logs "vk.msaa override ..." / clamps once per pair.
uint8_t effectiveSamples(uint8_t game);
// Override in force: vpvk_config_set_msaa_override if called, else env VPVK_MSAA, else 0.
int msaaOverride();
// A pass the game began: its ORIGINAL sample count (before forcing/override), the size of its
// colour attachment, multiview mask, and whether it has colour and depth. The largest qualifying
// one is the eye pass (vpvk_msaa_detected, log "vk.msaa detected eye=Nx").
void msaaObservePass(uint8_t gameSamples, uint32_t w, uint32_t h, uint32_t viewMask, bool hasColor, bool hasDepth,
                     uint64_t markedEye = 0);  // markedEye (packEyeSize): the eye size while the VPKTAA jitter has none
// After each vkQueueSubmit: without eye size or multiview, the largest pass decides after 120.
void msaaSubmitTick(uint64_t submits);

// VPKTAA inline jitter (config.cpp, vpvk_taa_set_jitter): the setting one submission is translated
// with, taken once per vkQueueSubmit so every draw of it gets the same jitter.
struct TaaJitter {
  bool enabled = false;
  uint32_t width = 0, height = 0;  // eye size; 0,0 = multiview passes only
  float jx = 0, jy = 0;            // target pixels, +x right, +y down
  uint64_t tag = 0;
  bool still = false;              // RCAS sin TAA: pases de ojo identificados y máscara de UI, sin offset
};
TaaJitter taaJitter();  // disabled when env VPVK_TAA_JITTER=0; VPKTAA nativo: jx/jy Halton del motor
// VPKTAA nativo (vpvk_taa_native_*): puerta y jitter aplicado desde la última toma (avanza la fase).
bool taaNativeOn();
// Máscara de UI del VPKTAA nativo (taa_native.h): VPVK_TAA_UI_MASK=0/1, por defecto 1 con el nativo activo.
bool taaUiMaskOn();
// RCAS (taa_native.h): intensidad efectiva 0..1 (0 = apagado), según env VPVK_RCAS > vpvk_config_set_rcas >
// la nitidez de VPKTAA con VPKTAA nativo; y el denoise opcional (VPVK_RCAS_DENOISE=1).
float rcasIntensity();
bool rcasOn();
bool rcasDenoise();
// Todo attachment del juego (color o profundidad, cualquier número de muestras, con cualquier otro uso): no persistente, memoryless mientras el
// compilador de frames pueda dejarlo en tile; memoria solo en los frames que la necesitan (msaa_materialized) y devuelta
// tras VPVK_MSAA_RELEASE_FRAMES frames en tile (msaa_released). VPVK_MSAA_TILE=0 lo apaga (por defecto 1).
bool msaaTileOn();
struct TaaNativeApplied {
  float jx = 0, jy = 0;
  uint32_t jittered = 0, targetW = 0, targetH = 0;
  uint64_t phase = 0;
};
TaaNativeApplied taaNativeTake();  // no avanza la fase
void taaNativeAdvance(uint64_t phase);  // tras codificar el pase de esa fase (solo si jittered > 0)
// What submission `submit` applied (vpvk_taa_get_applied counters, "vk.taa jitter" log lines).
void taaRecordSubmit(uint64_t submit, const TaaJitter&, uint32_t jittered, uint32_t skipped, uint32_t targetW,
                     uint32_t targetH);

// Foveación propia (config.cpp; perfil en foveation.h). `apply` = vpvk_config_set_foveation_apply / VPVK_FOVEATION.
struct FovConfig {
  fov::Shape shape;
  fov::Eyes eyes;
  bool apply = false;
};
FovConfig foveationConfig();
bool foveationApply();
// apply + modo no apagado + la GPU hace rate maps de 2 capas (platformRateMapSupported). Decide, al crear un
// pipeline, si sus etapas multiview llevan viewport index / reconstrucción de FragCoord.
bool foveationActive();
// El traductor marca con esto los pases de ojo (commands.cpp): el mapa del tamaño lógico del ojo, reconstruido cuando
// cambian los ajustes, las tangentes o el tamaño (clave del mapa). submitMutex.
struct FoveationState {
  bool uiSplit = true;  // la UI dibujada en el buffer del ojo se pinta fuera del pase foveado (VPVK_FOVEATION_UI=0: no)
  FoveationState();
  // Null: foveación apagada/no disponible o el ojo no es de 2 vistas.
  std::shared_ptr<const vpgpu::RateMap> mapFor(uint32_t w, uint32_t h, uint16_t layers);

 private:
  std::shared_ptr<const vpgpu::RateMap> cur_;
};
bool platformRateMapSupported();  // platform_metal.mm (sondeo una vez); platform_null.cpp: VPVK_NULL_RATEMAP=1

// Dispatchable handles start with the loader word (vk_icd.h): harmless without a loader and
// required if one is ever put in front of the driver.
struct Dispatchable {
  VK_LOADER_DATA loader;
  Dispatchable() { set_loader_magic_value(this); }
};

struct Instance;
struct Device;

struct PhysicalDevice : Dispatchable {
  Instance* instance = nullptr;
};

struct Instance : Dispatchable {
  uint32_t apiVersion = VK_API_VERSION_1_0;
  PhysicalDevice physical;
  std::string appName, engineName;
};

// The single queue family exposes kQueueCount VkQueues (Quest: queueCount 4). Each has its own submission order; they all
// execute on the one Metal command queue, in the order their batches become ready (entry.cpp drainDeferred).
inline constexpr uint32_t kQueueCount = 4;
// Timestamps are host ticks of limits.timestampPeriod; the bits above this many are 0 (commands.cpp queryEnd). Quest: 48.
inline constexpr uint32_t kTimestampValidBits = 48;

struct Queue : Dispatchable {
  Device* device = nullptr;
  uint32_t family = 0, index = 0;
};

// VK_KHR_sampler_ycbcr_conversion: a multi-planar format is a set of single-plane textures.
struct MultiPlane {
  uint8_t planes = 0;  // 0: not a multi-planar format
  VkFormat plane[3] = {VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED};
  uint8_t subX = 0, subY = 0;  // chroma size = luma size >> sub (rounded up)
};
const MultiPlane& multiPlane(VkFormat);  // physical_device.cpp
// Size of plane k (0..planes-1) of an image of `w` x `h` luma texels.
inline VkExtent2D planeExtent(const MultiPlane& m, uint32_t k, uint32_t w, uint32_t h) {
  return k == 0 ? VkExtent2D{w, h} : VkExtent2D{(w + (1u << m.subX) - 1) >> m.subX, (h + (1u << m.subY) - 1) >> m.subY};
}
// VkSamplerYcbcrConversion (entry.cpp): the parameters shaders and image views need.
struct YcbcrConversionObj {
  VkFormat format = VK_FORMAT_UNDEFINED;
  spirv::YcbcrBinding spec;  // planeSlot is filled per pipeline layout
  VkFilter chromaFilter = VK_FILTER_NEAREST;
};

struct Image;
struct DeviceMemory {
  VkDeviceSize size = 0;
  uint32_t typeIndex = 0;
  uint8_t* host = nullptr;  // page-aligned; unified memory: GPU buffers alias it in place
  bool mapped = false;
  bool lazy = false;  // VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT type: no host memory (host stays null)
  // Images bound here (Device::submitMutex): one bound at the offset of another with the same
  // description shares its texture (memory aliasing: what one writes the other reads).
  std::vector<Image*> images;
};

struct Buffer {
  VkDeviceSize size = 0;
  VkBufferUsageFlags usage = 0;
  DeviceMemory* memory = nullptr;
  VkDeviceSize offset = 0;
  vpgpu::BufferId id;  // created on bind: aliases memory->host + offset
};

struct Image {
  VkImageCreateInfo info{};
  vpgpu::TextureDesc desc;
  DeviceMemory* memory = nullptr;  // memory accounting only: Metal owns texture storage
  VkDeviceSize offset = 0;
  vpgpu::TextureId id;  // created on bind (or at creation for memoryless use)
  // Owners of `id`: images aliasing one memory range share the texture; the last one destroys it.
  std::shared_ptr<int> idOwners;
  uint8_t forcedFrom = 0;  // >1: the game asked for this many samples, created 1x (forceSingleSample)
  uint8_t gameSamples = 1;  // what the game asked for (desc.samples is what was created)
  // The host marked it as an image of the eye layer (vpvk_mark_eye_image): the one destination redrawn every frame,
  // the only one whose draws the pipeline gate may drop. Written under submitMutex.
  bool eye = false;
  // id<MTLTexture> (unretained) cached at bind for textures allocated there: vkGetMTLTextureMVK
  // and vkExportMetalObjectsEXT answer from it without touching the backend's maps (other threads).
  void* native = nullptr;
  // VK_IMAGE_TILING_LINEAR colour image: its texels must be readable/writable through the mapped
  // VkDeviceMemory (linearLayout). The texture stays the GPU copy; `hostAlias` is a core buffer
  // aliasing memory->host + offset, synced with it around every transfer (commands.cpp).
  bool linear = false;
  vpgpu::BufferId hostAlias;
  // Multi-planar format (multiPlane(info.format).planes > 1): this image is plane 0 (desc is its
  // texture); planes 1 and 2 are images of their own, bound with it at consecutive offsets.
  std::vector<std::unique_ptr<Image>> planes;
  // VK_KHR_dynamic_rendering_local_read: the copy that draws after an in-pass barrier read instead of this attachment
  // (commands.cpp localReadSplit), made on first need and destroyed with the image (submitMutex).
  mutable vpgpu::TextureId localReadCopy;
};

// The image a copy of `aspect` touches: plane images of a multi-planar image for PLANE_1 / PLANE_2,
// otherwise the image itself.
inline Image* planeOf(Image* im, VkImageAspectFlags aspect) {
  if (im->planes.empty()) return im;
  if ((aspect & VK_IMAGE_ASPECT_PLANE_2_BIT) && im->planes.size() > 1) return im->planes[1].get();
  if (aspect & VK_IMAGE_ASPECT_PLANE_1_BIT) return im->planes[0].get();
  return im;
}
// VkSamplerYcbcrConversion handle.
struct YcbcrConversionHandle {
  std::shared_ptr<const YcbcrConversionObj> obj;
};

// Where subresource (mip, layer) of a linear image lives in its memory: tightly packed rows (in
// blocks), mips in order, each mip holding all its layers (3D: all its depth slices, depthPitch
// apart). Used by vkGetImageSubresourceLayout and the translator's linearSync copies.
struct LinearLayout {
  uint64_t offset = 0, size = 0, rowPitch = 0, depthPitch = 0, arrayPitch = 0;
  uint32_t rows = 0;  // block rows per image
};
LinearLayout linearLayout(const vpgpu::TextureDesc&, uint32_t mip, uint32_t layer);

struct ImageView {
  Image* image = nullptr;
  VkImageViewType type = VK_IMAGE_VIEW_TYPE_2D;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkImageSubresourceRange range{};
  vpgpu::TextureId view;  // backend view when type/format/range differ from the image (bindings)
  // Cube / cube-array views of storage images: the same slices as a 2D-array view. Metal has no texture atomics (and the
  // translator no cube storage access) on cubes: shaders declare these images texture2d_array (msl.cpp textureType).
  vpgpu::TextureId storageView;
  // Multi-planar image viewed through a Y'CbCr conversion: views of planes 1 and 2 (this view is
  // plane 0's). Owned.
  std::vector<ImageView*> planeViews;
};

struct BufferView {
  Buffer* buffer = nullptr;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkDeviceSize offset = 0, range = 0;
};

struct Sampler {
  vpgpu::SamplerId id;
  std::shared_ptr<const YcbcrConversionObj> ycbcr;  // sampler Y'CbCr conversion (immutable samplers only)
};

// vpvk_dump_pending (pending.cpp): what the host asked about an object. Relaxed atomics inside the object itself.
inline int64_t pendingNowNs() {  // steady clock, ns
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Polled {
  std::atomic<uint32_t> n{0};
  std::atomic<int64_t> ns{0};      // when it was last asked
  std::atomic<int32_t> result{0};  // what it answered (VkResult)
  void note(int32_t r) {
    n.fetch_add(1, std::memory_order_relaxed);
    ns.store(pendingNowNs(), std::memory_order_relaxed);
    result.store(r, std::memory_order_relaxed);
  }
};
struct Fence {
  std::atomic<bool> signaled{false};
  // Its submission waits on a timeline value nothing has signaled yet (Device::deferred): not on the
  // GPU, so not done whatever `frames` says.
  std::atomic<bool> deferred{false};
  std::atomic<bool> submitted{false};  // signaled by a submission: the GPU may still be running it
  std::atomic<uint64_t> frames{0};     // done once every vpgpu frame below this index has finished
  Polled poll;                         // vkGetFenceStatus / vkWaitForFences
  std::atomic<int64_t> submitNs{0};    // when the vkQueueSubmit that signals it was made
  std::atomic<uint32_t> waiting{0};    // threads inside vkWaitForFences on it
};
struct Semaphore {
  uint64_t value = 0;
  // VK_KHR_timeline_semaphore (ext.cpp): the counter the host sees, and the values submissions will
  // reach once the vpgpu frames below `frames` have finished on the GPU.
  bool timeline = false;
  std::mutex m;
  uint64_t completed = 0;
  std::vector<std::pair<uint64_t, uint64_t>> pending;  // (value, frames)
  Polled poll;                                         // vkGetSemaphoreCounterValue
  std::atomic<uint64_t> waitMax{0};                    // highest value vkWaitSemaphores was asked for
  std::atomic<int64_t> waitNs{0};                      // when it was last asked
  std::atomic<uint32_t> waiting{0};                    // threads inside vkWaitSemaphores on it
};
// config.cpp: the Quest 3 / Adreno 740 identity (vpvk_config_set_quest_identity, VPVK_QUEST_IDENTITY=1). Off by
// default: the device then reports what the Apple GPU supports. Not latched (read per query).
bool questIdentity();
// ext.cpp: the 1.2/1.3 extensions, features and formats the Adreno profile adds. Always on, except with the
// Quest identity, where they are advertised only with VPVK_EXTRA_EXTENSIONS=1 (the conformance run) until the CTS
// has validated them, so games keep the code paths they were tested on.
bool extraExtensions();
// ext.cpp: the Vulkan 1.3 extensions Half-Life: Alyx (Source 2) uses that an Adreno driver does not list
// (VK_EXT_host_image_copy, VK_EXT_memory_budget, ...): only with the native identity and extraExtensions().
// With the Quest identity nothing of them is announced, not even with VPVK_EXTRA_EXTENSIONS=1.
bool nativeExtras();
bool internalClient();  // ext.cpp: inside vpvk_internal_scope(1) (vpgles' own instance/device)
bool reportGeom();  // VPVK_REPORT_GEOM=1: report the Adreno 740 geometry/tessellation/interpolator caps (games only)
// vkCmdSetEvent / vkCmdResetEvent take effect when their submission finishes: `frame` is that
// submission's frame + 1 (0 = none pending) and `value` what it leaves; vkGetEventStatus settles it.
struct Event {
  std::mutex m;
  bool set = false;
  uint64_t frame = 0;
  bool value = false;
  Polled poll;                                // vkGetEventStatus
  uint32_t hostSet = 0, hostReset = 0, cmdStamps = 0;  // vkSetEvent / vkResetEvent calls, vkCmdSetEvent / vkCmdResetEvent applied by a submit (m)
  int64_t stampNs = 0;                        // the last of those (m)
};

struct RenderPassObj {
  std::vector<VkAttachmentDescription2> attachments;
  struct Subpass {
    std::vector<VkAttachmentReference2> colors, resolves, inputs;
    VkAttachmentReference2 depth{};
    bool hasDepth = false;
    uint32_t viewMask = 0;
    // VK_KHR_depth_stencil_resolve (VkSubpassDescriptionDepthStencilResolve); attachment UNUSED = none.
    uint32_t depthResolve = VK_ATTACHMENT_UNUSED;
    VkResolveModeFlagBits depthResolveMode = VK_RESOLVE_MODE_NONE, stencilResolveMode = VK_RESOLVE_MODE_NONE;
  };
  std::vector<Subpass> subpasses;
  int32_t densityMap = -1;  // VK_EXT_fragment_density_map attachment index
};

// ---- shaders, descriptors, pipelines (pipelines.cpp) ----
struct ShaderModuleObj {
  // Shared: a pipeline that may be retranslated later (texel buffer view formats, see
  // PipelineObj::texel) keeps the parsed module alive after vkDestroyShaderModule.
  std::shared_ptr<spirv::SpirvModule> module;
  std::string error;  // parse error (the pipeline reports it)
  // Kept only while recipes are recorded: the stage inputs reference the module by content hash
  // and the store saves the words once, so a later translator version can retranslate them.
  std::shared_ptr<const std::vector<uint32_t>> words;  // null unless recipes are recorded
  uint64_t hash = 0;
};

struct SetLayoutBinding {
  uint32_t binding = 0, count = 1;
  VkDescriptorType type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  VkShaderStageFlags stages = 0;
  std::vector<Sampler*> immutable;
};
struct SetLayoutData {
  std::vector<SetLayoutBinding> bindings;  // sorted by binding
  bool push = false;                       // VK_KHR_push_descriptor layout: never allocated from a pool
  const SetLayoutBinding* find(uint32_t b) const {
    for (const auto& x : bindings)
      if (x.binding == b) return &x;
    return nullptr;
  }
};
struct DescriptorSetLayoutObj {
  std::shared_ptr<const SetLayoutData> data;
};

// Where a (set, binding) lives in the shaders' argument tables.
struct LayoutSlot {
  uint32_t set = 0, binding = 0, count = 1;
  VkDescriptorType type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  vpgpu::BindKind kind = vpgpu::BindKind::Buffer;
  uint16_t slot = 0, samplerSlot = 0xFFFF;
  // Immutable sampler with a Y'CbCr conversion: planes 1 and 2 are Textures at planeSlot (+ element).
  std::shared_ptr<const YcbcrConversionObj> ycbcr;
  uint16_t planeSlot[2] = {0xFFFF, 0xFFFF};
};
struct PipelineLayoutObj {
  std::vector<std::shared_ptr<const SetLayoutData>> sets;
  uint32_t pushSize = 0;
  std::vector<LayoutSlot> slots;
  int pushSlot = -1;
  uint16_t nextBuffer = 0;  // first Buffer slot after the descriptors (vertex buffers go here)
};

struct Descriptor {
  VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
  const Buffer* buffer = nullptr;
  VkDeviceSize offset = 0, range = 0;
  const ImageView* view = nullptr;
  const Sampler* sampler = nullptr;
  const BufferView* texel = nullptr;
  bool immutable = false;  // the layout's immutable sampler: descriptor writes leave `sampler` alone
};
struct DescriptorPoolObj;
struct DescriptorSetObj {
  std::shared_ptr<const SetLayoutData> layout;
  std::vector<std::vector<Descriptor>> slots;  // parallel to layout->bindings, one per element
  // VK_EXT_inline_uniform_block: the bytes of each inline block binding (parallel to layout->bindings,
  // empty for the others). The binding's descriptorCount is its size; it has a single slot.
  std::vector<std::vector<uint8_t>> inlineData;
  DescriptorPoolObj* pool = nullptr;
};
struct DescriptorPoolObj {
  std::vector<DescriptorSetObj*> sets;
};

struct DescriptorUpdateTemplateObj {
  std::vector<VkDescriptorUpdateTemplateEntry> entries;
  bool push = false;  // VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR: bindPoint / set tell where vkCmdPushDescriptorSetWithTemplate goes
  VkPipelineBindPoint bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  uint32_t set = 0;
};

struct PipelineObj {
  bool compute = false;
  // Asynchronous creation (pipelines.cpp "Asynchronous creation"): until `resolved`, a translate-pool job is still
  // writing ok, key, inputs, used, pushStages, texel, sources, attrs, ownedKeys and the shader ids / write masks of
  // `state`; nothing that reads them runs before resolvePipeline() returned true (draws, variants, destroy).
  // Everything the create info decides is final when vkCreate*Pipelines returns.
  std::atomic<bool> resolved{true};
  TranslatePool::JobPtr job;  // set once, before the object is published; kept until destroy
  bool ok = false;  // shaders translated; draws with a failed pipeline are skipped
  // Synchronous creation with VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT: failOnCompile is the create
  // flag; compileRequired says the pipeline was not in the cache / recipe store, nothing was requested and the
  // vkCreate*Pipelines call deletes the object (VK_PIPELINE_COMPILE_REQUIRED).
  bool failOnCompile = false, compileRequired = false;
  vpgpu::PipelineState state;
  vpgpu::PipelineKey key;
  std::shared_ptr<const PipelineLayoutObj> layout;
  struct Used {
    LayoutSlot slot;
    vpgpu::StageMask stages = 0;
    bool written = false;
  };
  std::vector<Used> used;
  vpgpu::StageMask pushStages = 0;
  std::vector<spirv::VertexBufferBinding> vertexBuffers;
  uint32_t viewCount = 1;
  // Instance multiplier of multiview draws: viewCount, or 1 when the vertex shader is amplified
  // (SpirvInterface::amplified; the backend then sets the encoder's vertex amplification count). Resolved fields.
  uint32_t instanceViews = 1;
  bool amplified = false;
  // Multiview pipeline that is not amplified: why (AmpWhy; None when amplified or single view). Counted per draw (gpu.amp).
  enum AmpWhy : uint8_t { AmpNone, AmpLayer, AmpClip, AmpMask, AmpOff, AmpWhyCount };
  uint8_t ampWhy = AmpNone;
  uint32_t viewMask = 0;  // multiview mask with gaps / not from bit 0 (SpirvLowerOptions::viewMask)
  VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  bool primitiveRestart = false;
  VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  std::vector<VkDynamicState> dynamic;
  bool hasViewport = false, hasScissor = false;
  VkViewport viewport{};
  VkRect2D scissor{};
  uint32_t stencilRef[2] = {0, 0};  // front, back
  float blendConstants[4] = {0, 0, 0, 0};
  bool depthBiasEnable = false, depthClamp = false;
  bool cullBoth = false;  // VK_CULL_MODE_FRONT_AND_BACK: every triangle is culled
  // VK_EXT_extended_dynamic_state(2) / Vulkan 1.3: the pipeline's own values of the states a
  // vkCmdSet* may replace (the core state keeps depth test + write fused), and the variants made
  // for draws that replace them (pipelines.cpp pipelineFor, keyed by a hash of the replaced values).
  VkCullModeFlags cullMode = VK_CULL_MODE_NONE;
  bool rasterDiscard = false;
  bool dynStride = false;  // VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE: strides come with the draw
  struct RawDepth {
    bool test = false, write = false, stencil = false;
    VkCompareOp compare = VK_COMPARE_OP_ALWAYS;
  } rawDepth;
  mutable std::unordered_map<uint64_t, vpgpu::PipelineKey> edsVariants;  // under variantMutex
  bool wireframe = false;  // VK_POLYGON_MODE_LINE
  bool pointMode = false;  // VK_POLYGON_MODE_POINT: triangle topologies draw their vertices as points (pipelines.cpp primitiveOf)
  float depthBias[3] = {0, 0, 0};  // constant, clamp, slope
  // MSAA: the game's rasterizationSamples; state.samples is the effective count at creation. A
  // draw into a pass of another (effective) count uses a variant with the pass's count (Metal
  // requires pipeline == attachments), created on first use and kept here.
  uint8_t gameSamples = 1;
  std::array<uint64_t, 3> inputs{};  // recipe stage inputs (variants are recorded too)
  mutable std::mutex variantMutex;
  // Core pipeline keys this object retained (itself and every variant); released when it is destroyed.
  mutable std::vector<uint64_t> ownedKeys;  // under variantMutex (or before the object is shared)
  mutable std::vector<std::pair<uint8_t, vpgpu::PipelineKey>> variants;
  // Texel buffers: SPIR-V only knows the element format of a storage texel buffer with a format
  // qualifier; uniform texel buffers (and Unknown storage ones) take it from the bound
  // VkBufferView. `texel` lists the texel buffer bindings the shaders use with the format they were
  // lowered with (R32Uint words when unknown at creation). A draw/dispatch whose bound views have
  // other formats uses a variant retranslated with them (texelVariant), made on first use from
  // `sources` (kept only when `texel` is not empty).
  struct TexelBinding {
    uint32_t set = 0, binding = 0;
    vpgpu::Format lowered = vpgpu::Format::Undefined;
  };
  std::vector<TexelBinding> texel;
  struct StageSrc {
    std::shared_ptr<spirv::SpirvModule> module;
    std::shared_ptr<const std::vector<uint32_t>> words;  // recipes only
    uint64_t hash = 0;
    std::string entry;
    VkShaderStageFlagBits stage = VK_SHADER_STAGE_VERTEX_BIT;
    std::vector<std::pair<uint32_t, uint64_t>> spec;
  };
  std::vector<StageSrc> sources;
  std::vector<spirv::VertexAttribute> attrs;
  uint32_t sampleMask = ~0u;  // VkSampleMask of the pipeline (pSampleMask[0]) when it clears samples of the game's count, else ~0u
  bool a2cTest = false;  // alpha-to-coverage pipeline created at 1x (MSAA forced off): alpha test in the fragment shader
  uint16_t colorIntKinds = 0;  // SpirvLowerOptions::colorIntKinds of the fragment stage
  uint8_t fragmentOutputs = 0;  // colour outputs the fragment shader writes (a dynamic colour write mask never enables others)
  bool dualSource = false;      // the fragment shader writes [[color(0), index(1)]] (Metal: one render target)
  struct TexelVariant {
    std::vector<vpgpu::Format> formats;  // parallel to `texel`
    std::vector<uint32_t> strides;       // parallel to `vertexBuffers` (dynamic strides, else the pipeline's)
    bool ok = false;
    vpgpu::PipelineState state;
    vpgpu::PipelineKey key;
    std::array<uint64_t, 3> inputs{};
    std::vector<std::pair<uint8_t, vpgpu::PipelineKey>> samples;  // MSAA variants of this one
  };
  mutable std::deque<TexelVariant> texelVariants;  // deque: stable addresses; under variantMutex
  // VPKTAA UI mask (maskPipelineKey): variants for draws into an eye pass with the R8 mask attachment;
  // `tv` = the texel variant they derive from (null = the pipeline itself). Under variantMutex.
  struct MaskVariant {
    const TexelVariant* tv = nullptr;
    uint8_t samples = 0;
    bool ui = false, ok = false;
    vpgpu::PipelineKey key{};
    uint64_t fragment = 0;  // fragment shader id of the variant (tests)
  };
  mutable std::vector<MaskVariant> maskVariants;
  bool isDynamic(VkDynamicState s) const {
    for (auto d : dynamic)
      if (d == s) return true;
    return false;
  }
};

// Queries (pipelines.cpp). Occlusion pools own a range of the device's visibility buffer (8 bytes
// a query); `frame[q]` is 1 + the frame whose vkCmdEndQuery made q pending (available once it
// finishes), 0 after a reset. Without a range (buffer full) occlusion reads as "visible" (1), never culled.
struct QueryPoolObj {
  VkQueryType type = VK_QUERY_TYPE_OCCLUSION;
  uint32_t count = 0;
  Device* device = nullptr;
  uint32_t base = UINT32_MAX;  // first visibility slot
  std::mutex m;
  std::vector<uint64_t> frame;
  Polled poll;                                          // vkGetQueryPoolResults
  std::atomic<uint32_t> lastNotReady{UINT32_MAX};       // the last query it had no result for
};

struct Framebuffer {
  std::vector<ImageView*> views;
  uint32_t width = 0, height = 0, layers = 1;
};

struct CommandPool;
struct CommandBuffer;
struct Recorder;  // commands.cpp

struct CommandBuffer : Dispatchable {
  Device* device = nullptr;
  CommandPool* pool = nullptr;
  VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  enum class State : uint8_t { Initial, Recording, Executable, Invalid } state = State::Initial;
  std::unique_ptr<Recorder> rec;
  ~CommandBuffer();
};

// One VkSubmitInfo held back by vkQueueSubmit (entry.cpp): it waits on a timeline value that no
// submission on the GPU will reach and the host has not signaled yet (wait-before-signal). Batches
// leave Device::deferred in order once every wait can be met (submitReady): their command buffers
// are translated then (the client keeps them pending until the fence/semaphore says done).
struct DeferredBatch {
  std::vector<const CommandBuffer*> cbs;
  std::vector<std::pair<Semaphore*, uint64_t>> waits, signals;  // timeline semaphores only
  Fence* fence = nullptr;                                     // last batch of its vkQueueSubmit
  uint32_t queue = 0;                                         // index of the VkQueue it was submitted to
};

struct CommandPool {
  uint32_t family = 0;
  std::vector<CommandBuffer*> buffers;
};

// VPKTAA UI mask (taa_native.h): the classifier that outlives a command buffer (a scene is often
// recorded as several), the R8 mask textures eye passes draw into and what the next
// vpvk_taa_native_encode reads. submitMutex.
struct UiMaskState {
  Device* dev = nullptr;
  taa::DrawClassifier classifier;
  struct Target {
    vpgpu::TextureId tex;
    uint32_t w = 0, h = 0;
    uint16_t layers = 0;
    vpgpu::TextureId ms;  // transient multisample twin (MSAA eye passes): drawn into, resolved into `tex`
    uint8_t msSamples = 0;
  };
  Target target[2];  // by masked pass (single-view: one per eye; multiview: [0] has both layers)
  struct Live {
    bool valid = false;
    Target t;
  } live[2];         // written since the last take
  uint32_t maskedPasses = 0;  // masked eye passes since the last take
  // The newest masks, for hosts that composite the eye themselves (vpvk_ui_mask_texture): published at the end of
  // each submission that held eye passes (retained native MTLTextures, so they outlive a resize or the next frame's
  // reuse); cleared when such a submission drew no UI. Own mutex: read from the compositor thread, never needs
  // submitMutex. Order: liveDeviceMutex -> lastMu.
  struct Published {
    void* tex = nullptr;  // +1 (Platform::retainNative)
    uint32_t w = 0, h = 0, slice = 0;
  };
  std::mutex lastMu;
  Published last[2];  // by eye
  // Eye passes since the last take (taaPass: depth attachment + colour 0), for the depth vpvk_taa_native_encode
  // finds itself when the host gives none (vpgles, OpenXR: no host-side eye table). Ring: slot = n % kEyePasses.
  struct EyePass {
    vpgpu::TextureId color, depth;
    uint16_t colorLayer = 0, depthLayer = 0, layers = 1;
    bool cleared = false;
    float clearValue = 1.0f;
    uint32_t greater = 0, less = 0;  // depth compare of its draws (GREATER-like / LESS-like)
  };
  static constexpr uint32_t kEyePasses = 4;
  EyePass eyePass[kEyePasses];
  uint32_t eyePasses = 0;
  taa::UiMaskCounters counters;  // since the last "vk.taa uimask" line
  // Synchronous pipeline mode: mask variants requested while a submission was translated; runBatches awaits them all at
  // once (awaitFreshVariants) instead of one wait per variant under submitMutex.
  std::vector<vpgpu::PipelineKey> freshVariants;
  uint64_t nextLogSubmit = 0;
  uint32_t logLines = 0;
  // The texture for masked pass number `slot` (0/1), created or recreated for this size. samples > 1:
  // target[slot & 1].ms is also created with that sample count (the pass attaches it and resolves into
  // the returned 1x texture); {} is returned if either fails.
  vpgpu::TextureId acquire(uint32_t slot, uint32_t w, uint32_t h, uint16_t layers, uint8_t samples = 1);
  void release();  // vkDestroyDevice
};

// Forced MSAA (msaaForce, commands.cpp Translator::forceMsaa): the transient N-sample twins the game's 1x scene / eye passes
// render into (always tile memory: never stored, resolved into the game's image), cached per guest image, plus the counters
// of the "vk.msaa force" line. submitMutex.
struct MsaaTwins {
  struct Twin {
    vpgpu::TextureId guest, twin;
    uint8_t samples = 0;
    uint16_t layers = 0, mip = 0, layer = 0;
  };
  std::vector<Twin> twins;
  // Passes forced, and the passes that qualified as scene passes but were left at 1x, by reason (cumulative).
  uint32_t passes = 0, skipLoad = 0, skipInput = 0, skipDsb = 0, skipEds = 0, skipFmt = 0, skipSamples = 0, skipArea = 0, skipShape = 0;
  uint32_t gameAuto = 0;  // attachments of the game's own MSAA passes handled as autoMsaa (VPVK_GAME_MSAA_AUTO)
  uint32_t skipUnreadable = 0;  // ... and those left alone because their resolve target is not shader-readable (no sampled usage)
  uint32_t acquires = 0;  // calls of acquire (the sweep of dead guests runs every 256th)
  // forceMsaa rejections ("vk.msaa force ... rej{...}"): rejN force off (<2 samples), rejEye not an eye scene pass, rejCd no colour 0
  // / depth, rejSmp the game's own samples, rejRes the game's own resolve, rejDesc a texture description unavailable.
  uint32_t rejN = 0, rejEye = 0, rejCd = 0, rejSmp = 0, rejRes = 0, rejDesc = 0;
  // Rejected passes logged ("vk.msaa rej"): up to 12 distinct (size, layers, reason) under 1 Mpx, and 12 at or above it.
  struct RejSeen { uint32_t w = 0, h = 0; uint16_t layers = 0; const char* why = nullptr; };
  RejSeen rejSeen[2][12];
  uint8_t rejLogged[2] = {0, 0};
  void noteRej(Device* dev, const vpgpu::RenderPassDesc& d, const char* why);  // logs a rejected pass (see above); count it separately
  int64_t nextLogNs = 0;
  // The twin of `guest` (texture description `gd`) for a pass of `layers` layers over mip `mip`, layer `layer`; created on the
  // first use, and entries whose guest image is gone are freed first. {} if it cannot be created.
  vpgpu::TextureId acquire(Device* dev, vpgpu::TextureId guest, const vpgpu::TextureDesc& gd, uint8_t samples, uint16_t layers,
                           uint16_t mip, uint16_t layer);
  void release(Device* dev);  // vkDestroyDevice
};

struct Device : Dispatchable {
  PhysicalDevice* physical = nullptr;
  std::unique_ptr<Platform> platform;
  std::unique_ptr<vpgpu::Device> gpu;
  std::unique_ptr<RecipeStore> recipes;  // pipeline recipes for prewarming the next run
  // Recipes stored when this device was created (0 = no recipes / prewarm off) and how the game's own pipeline creates
  // fared against them (pipelines.cpp noteRecipeKey: gpu.pso.recipe game hits/misses). A miss is a key no earlier run
  // recorded: a new pipeline, or recipes that do not match the game's keys.
  uint32_t prewarmedRecipes = 0;
  std::atomic<uint32_t> recipeHits{0}, recipeMisses{0};
  // vpvk_recipe_stats (the GLES layer's gles.pso.recipe line): recipes read from disk at vkCreateDevice (even with the
  // prewarm off), recipes the prewarm left ready, and the game's creates no recipe covered (counted with nothing stored too).
  uint32_t recipesLoaded = 0;
  std::atomic<uint32_t> recipesPrewarmed{0}, recipesFresh{0};
  Queue queues[kQueueCount];
  bool multiview = false, densityMap = false, metalObjects = false;
  std::mutex submitMutex;
  std::mutex resourceMutex;  // gpu->create*/destroy from recording threads (upload arenas)
  // GPU indirect draws that need their arguments rewritten (multiview: instances x views; draw-count
  // buffers): a small compute kernel run before the render pass (pipelines.cpp indirectPatch).
  std::once_flag indirectPatchOnce;
  vpgpu::PipelineKey indirectPatchKey{};
  std::once_flag indexU8Once;  // the 8-bit index widening kernel (pipelines.cpp)
  vpgpu::PipelineKey indexU8Key{};
  std::atomic<int64_t> indexU8BackoffNs{0};
  // A wait for the kernel that timed out arms a backoff: until the deadline (steady clock ns) no draw waits again, they take the
  // CPU path / skip at once (100 draws must not cost 100 waits). Budgets are fields so tests can shorten them.
  std::atomic<int64_t> indirectPatchBackoffNs{0};
  uint32_t indirectPatchWaitMs = 2000, indirectPatchBackoffMs = 3000;
  uint64_t submits = 0;
  std::vector<DeferredBatch> deferred;  // submitMutex; submission order (batches of one queue keep their order, queues are independent)
  // vpvk_dump_pending: when each vpgpu frame was handed to the backend (by frame index % 64), and how many are noted.
  std::atomic<int64_t> frameNs[64]{};
  std::atomic<uint64_t> frameNsNoted{0};
  UiMaskState uiMask;                   // submitMutex
  MsaaTwins msaaTwins;                  // submitMutex
  FoveationState fovState;              // submitMutex
  HitchLine hitch;                      // vk.hitch (perf.h), submitMutex: this device's slow-frame lines
  struct {
    uint32_t passes = 0, splits = 0, lines = 0;  // since the last "vk.fov frame" line
    uint64_t next = 0;
  } fovLog;                             // submitMutex
  // Zero-initialised images (entry.cpp zeroInit*): textures created at bind that nothing has written
  // yet, keyed by texture handle (generation << 32 | index). submitMutex.
  struct ZeroPending {
    vpgpu::TextureId id;
    vpgpu::TextureDesc desc;
    uint64_t written = 0;  // subresources (mip * layers + layer) a full overwrite has covered, if trackable
    bool touched = false;
    bool viaCopy = false;  // cleared by a copy from zeroBuffer (not a render target)
    void* native = nullptr;  // id<MTLTexture> once exposed to the host (key in zeroNative)
    std::weak_ptr<int> flag;  // Image::idOwners (weak: its use_count decides who destroys the texture): nonzero while pending (record time reads it, see zeroPendingImage)
  };
  vpgpu::BufferId zeroBuffer;  // lazily created, submitMutex
  std::unordered_map<uint64_t, ZeroPending> zeroPending;
  // Host-visible view of the pending set (vpvk_image_zero_pending/_clear): MTLTexture -> zeroPending
  // key, only for textures the host asked for. Own mutex so the query never waits for a submit.
  std::mutex zeroNativeMutex;
  std::unordered_map<void*, uint64_t> zeroNative;
  // Textures of the eye layer's images (vpvk_mark_eye_image; key generation << 32 | index). Lock order:
  // submitMutex, then eyeMutex. The pipeline gate (pipelines.cpp) soft-gates a pass only when it draws into one.
  std::mutex eyeMutex;
  std::unordered_set<uint64_t> eyeTextures;
  // Size of the latest image marked by vpvk_mark_eye_image (packEyeSize; written under submitMutex, read lock-free on the hot path).
  std::atomic<uint64_t> eyeSize{0};
  // Shader ids already handed to the platform (translateStage): a recipe hit registers each once.
  std::mutex shaderMutex;
  std::unordered_set<uint64_t> shadersAdded;
  // Asynchronous pipeline creation (null = every vkCreate*Pipelines translates and waits on the calling thread):
  // the pool translates stages off the game's threads; the prewarm thread compiles the recipes of earlier runs.
  // Both are stopped (joined) before gpu / platform go (vkDestroyDevice).
  std::unique_ptr<TranslatePool> translator;
  std::thread prewarmThread;
  std::atomic<bool> prewarmCancel{false};
  // Per-boot summary line (pipelines.cpp pipelineSummary): next time one is due, steady clock ms since create.
  std::chrono::steady_clock::time_point created = std::chrono::steady_clock::now();
  // The device was created by vpgles (internalClient() at vkCreateDevice; the flag is thread-local, so it is kept here).
  // GL has no "the game redraws it" guarantee for render-to-texture: its hard passes wait longer (persistWaitMs).
  bool internalClient = false;
  // Synchronous pipeline mode (pipelines.cpp awaitPipelines): steady-clock ns until which a wait that already timed out
  // makes the next ones short (kBackoffWaitMs).
  std::atomic<int64_t> pipelineBackoffUntilNs{0};
  uint64_t nextSummaryMs = 20000;
  uint64_t zeroCleared = 0;  // images cleared so far
  uint32_t zeroLogLines = 0;
  // Occlusion query counters (Metal visibility result buffer, pipelines.cpp): one host-aliased
  // buffer per device, `used` = (first slot, count) ranges of live pools, sorted.
  struct Visibility {
    std::mutex m;
    uint8_t* host = nullptr;
    vpgpu::BufferId id;
    std::vector<std::pair<uint32_t, uint32_t>> used;
  } visibility;
};
void releaseVisibility(Device*);  // vkDestroyDevice
inline uint64_t markedEyeOf(const Device* d) { return d ? d->eyeSize.load(std::memory_order_relaxed) : 0; }  // (packEyeSize)
// VPVK_ZERO_NEW_IMAGES (default 1): a new image reads as zeros until first written, as on Android.
bool zeroNewImagesOn();
// True while the image's texture is zero-init pending (shared by images aliasing one texture). Read
// without locks at record time: a stale answer only costs a lookup (or a missed read the bind-time
// state could not have known about).
inline bool zeroPendingImage(const Image* i) {
  return i && i->idOwners && std::atomic_ref<int>(*i->idOwners).load(std::memory_order_relaxed) != 0;
}
// What a submit did to a pending image, in order (commands.cpp translate logs only these, so the
// resolve costs O(touches), not O(frame)).
struct ZeroTouch {
  enum Kind : uint8_t { Read, WriteWhole, WriteSpan } kind = Read;
  vpgpu::TextureId tex;
  uint16_t mip = 0, layer = 0, layers = 1;
  uint32_t x = 0, y = 0, z = 0, width = 0, height = 0;  // WriteSpan: the region written
};
struct ZeroLog {
  const Device* dev = nullptr;  // its zeroPending says which textures are worth logging
  std::vector<ZeroTouch> touches;
};
// entry.cpp (submitMutex held): bind marks a fresh texture pending; translate logs the touches of
// the pending ones; the submit then appends the zero clears of the images read first (or only partly
// written) to `pre`, which must run before the frame. Returns the number of images cleared.
void zeroInitMark(Device&, const Image&);
// Eye-layer images (vpvk_mark_eye_image). markEyeTexture: the texture is an eye image (submitMutex held or not: takes
// eyeMutex only). isEyeTexture: the pipeline gate's question; unknown / unmarked = false.
void markEyeTexture(Device&, vpgpu::TextureId);
bool isEyeTexture(Device&, vpgpu::TextureId);
// pipelines.cpp: how long a submit waits for the pipelines of its hard passes (VPVK_PIPELINE_PERSIST_WAIT_MS wins).
uint32_t persistWaitMs(bool internalClient);
// pipelines.cpp: the extra wait for the eye's passes (VPVK_PIPELINE_EYE_HARD_WAIT_MS wins; default 0, 1500 for vpgles).
uint32_t eyeHardWaitMs(bool internalClient);
// pipelines.cpp (tests): forget which pipelines already had their eye wait and the per-second budget spent.
void eyeWaitReset();
// submitMutex held: the image's MTLTexture became known; a pending one becomes queryable by it.
void zeroNoteNative(Device&, const Image&, void* native);
void zeroForgetNative(Device&, Device::ZeroPending&);  // drops its host-visible entry (before erasing it)
size_t zeroInitResolve(Device&, const ZeroLog& log, vpgpu::Frame& pre);
// entry.cpp: runs the deferred batches whose timeline waits can now be met (takes submitMutex);
// called after a host vkSignalSemaphore.
void submitDeferred(Device*);
// pending.cpp (vpvk_dump_pending): the objects the host polls, listed from create to destroy (Fence, Event,
// Semaphore (timeline only), QueryPoolObj); submitMutex held: a submit has just advanced the frame count to `frames`.
template <class T> void pendingTrack(T*);
template <class T> void pendingUntrack(T*);
void pendingFrames(Device&, uint64_t frames);
// vkWaitForFences / vkWaitSemaphores around the real wait: who is inside it, and what it returned.
void pendingFenceWait(uint32_t n, const VkFence* fs, bool enter, VkBool32 waitAll = VK_FALSE, VkResult r = VK_SUCCESS);
void pendingSemaphoreWait(const VkSemaphoreWaitInfo&, bool enter);
// The report vpvk_dump_pending logs, one "vk.pending ..." line each (at most ~40).
std::vector<std::string> pendingReport(Device&);

// Physical device model (physical_device.cpp).
const VkPhysicalDeviceProperties& deviceProperties();
// VkPhysicalDeviceFeatures::textureCompressionBC and the BCn formats: the GPU samples them and the Quest identity is off.
bool textureCompressionBC();
const VkPhysicalDeviceMemoryProperties& memoryProperties();
// Feature bits of a format for optimal tiling / buffers (0 = unsupported).
VkFormatFeatureFlags formatFeatures(VkFormat, bool buffer);
// A format stored as its core format with R and B swapped (B4G4R4A4 as ABGR4).
bool isSwappedRB(VkFormat f);
vpgpu::Format toCoreFormat(VkFormat);
// A 3D image of an ASTC format: created as RGBA8, its buffer->image copies decoded on the CPU (texdecode.h).
inline bool isDecodedVolume(const Image& im) { return im.info.imageType == VK_IMAGE_TYPE_3D && im.desc.format != toCoreFormat(im.info.format); }
std::vector<VkExtensionProperties> instanceExtensions();
std::vector<VkExtensionProperties> deviceExtensions();
std::vector<VkExtensionProperties> hiddenDeviceExtensions();

// Entry points (entry.cpp): null for names the driver does not implement (logged once).
PFN_vkVoidFunction lookup(const char* name, bool deviceLevel);
PFN_vkVoidFunction lookupPhase3(const char* name);  // pipelines.cpp
PFN_vkVoidFunction lookupExt(const char* name);     // ext.cpp
// capture.cpp: the VPCAP capture hook of the entry point `real` (found by address; `name` is only for the log of what is
// not hooked), `real` itself when it has none.
PFN_vkVoidFunction captureWrap(const char* name, PFN_vkVoidFunction real);

template <class T, class H>
T* from(H h) { return reinterpret_cast<T*>(h); }
template <class H, class T>
H to(T* p) { return reinterpret_cast<H>(p); }

}  // namespace vpvk

// Private hook of vpgles (pipelines.cpp; reached through vkGetDeviceProcAddr): the attachments of the open
// dynamic-rendering pass that are undefined when it ends (glInvalidateFramebuffer). The complete set, replacing an
// earlier call. See the definition for what keeps its store.
extern "C" void vpvk_cmd_pass_drop(VkCommandBuffer cb, uint32_t colorMask, int depth, int stencil);
extern "C" void vpvk_recipe_stats(VkDevice dv, uint32_t out[4]);  // pipelines.cpp: gles.pso.recipe counters
