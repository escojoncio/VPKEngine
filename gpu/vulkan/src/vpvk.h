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
#include <vector>

#include "vpgpu/core/device.h"
#include "vpvk/spirv_frontend.h"
#include "vpvk/platform.h"
#include "recipes.h"

namespace vpvk {

// Every log line of the driver starts with "vk." (one grep finds them all in the visor log).
void log(const std::string& line);
// App-forced sampler anisotropy (0/1 = off); see vpvk_config_set_anisotropy.
int anisotropyForced();
void logConfig();  // config.cpp: "vk.cfg effective ..." (called from vkCreateDevice)
void logf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
bool prewarmEnabled();  // config.cpp: vpvk_config_set_prewarm
bool forceSingleSample();  // config.cpp: vpvk_config_set_force_single_sample; frozen while a device lives
bool forceSingleSampleLive();
void holdForceSingleSample(bool acquire);  // vkCreateDevice / vkDestroyDevice

// ---- MSAA: detection of the game's eye pass and the sample-count override (config.cpp) ----
// Largest sample count the Metal device renders to (8/4/2 probed with supportsTextureSampleCount;
// 1 = none). Defined by the platform source (platform_metal.mm / platform_null.cpp), cached.
int platformMaxSampleCount();
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
void msaaObservePass(uint8_t gameSamples, uint32_t w, uint32_t h, uint32_t viewMask, bool hasColor, bool hasDepth);
// After each vkQueueSubmit: without eye size or multiview, the largest pass decides after 120.
void msaaSubmitTick(uint64_t submits);

// VPKTAA inline jitter (config.cpp, vpvk_taa_set_jitter): the setting one submission is translated
// with, taken once per vkQueueSubmit so every draw of it gets the same jitter.
struct TaaJitter {
  bool enabled = false;
  uint32_t width = 0, height = 0;  // eye size; 0,0 = multiview passes only
  float jx = 0, jy = 0;            // target pixels, +x right, +y down
  uint64_t tag = 0;
};
TaaJitter taaJitter();  // disabled when env VPVK_TAA_JITTER=0
// What submission `submit` applied (vpvk_taa_get_applied counters, "vk.taa jitter" log lines).
void taaRecordSubmit(uint64_t submit, const TaaJitter&, uint32_t jittered, uint32_t skipped, uint32_t targetW,
                     uint32_t targetH);

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

struct Queue : Dispatchable {
  Device* device = nullptr;
  uint32_t family = 0, index = 0;
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
  // id<MTLTexture> (unretained) cached at bind for textures allocated there: vkGetMTLTextureMVK
  // and vkExportMetalObjectsEXT answer from it without touching the backend's maps (other threads).
  void* native = nullptr;
  // VK_IMAGE_TILING_LINEAR colour image: its texels must be readable/writable through the mapped
  // VkDeviceMemory (linearLayout). The texture stays the GPU copy; `hostAlias` is a core buffer
  // aliasing memory->host + offset, synced with it around every transfer (commands.cpp).
  bool linear = false;
  vpgpu::BufferId hostAlias;
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
};

struct BufferView {
  Buffer* buffer = nullptr;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkDeviceSize offset = 0, range = 0;
};

struct Sampler {
  vpgpu::SamplerId id;
};

struct Fence {
  std::atomic<bool> signaled{false};
  // Its submission waits on a timeline value nothing has signaled yet (Device::deferred): not on the
  // GPU, so not done whatever `frames` says.
  std::atomic<bool> deferred{false};
  std::atomic<bool> submitted{false};  // signaled by a submission: the GPU may still be running it
  std::atomic<uint64_t> frames{0};     // done once every vpgpu frame below this index has finished
};
struct Semaphore {
  uint64_t value = 0;
  // VK_KHR_timeline_semaphore (ext.cpp): the counter the host sees, and the values submissions will
  // reach once the vpgpu frames below `frames` have finished on the GPU.
  bool timeline = false;
  std::mutex m;
  uint64_t completed = 0;
  std::vector<std::pair<uint64_t, uint64_t>> pending;  // (value, frames)
};
// ext.cpp: extensions advertised only with VPVK_EXTRA_EXTENSIONS=1 (the conformance run) until the
// CTS has validated them, so games keep the code paths they were tested on.
bool extraExtensions();
// vkCmdSetEvent / vkCmdResetEvent take effect when their submission finishes: `frame` is that
// submission's frame + 1 (0 = none pending) and `value` what it leaves; vkGetEventStatus settles it.
struct Event {
  std::mutex m;
  bool set = false;
  uint64_t frame = 0;
  bool value = false;
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
};
struct DescriptorPoolObj;
struct DescriptorSetObj {
  std::shared_ptr<const SetLayoutData> layout;
  std::vector<std::vector<Descriptor>> slots;  // parallel to layout->bindings, one per element
  DescriptorPoolObj* pool = nullptr;
};
struct DescriptorPoolObj {
  std::vector<DescriptorSetObj*> sets;
};

struct DescriptorUpdateTemplateObj {
  std::vector<VkDescriptorUpdateTemplateEntry> entries;
};

struct PipelineObj {
  bool compute = false;
  bool ok = false;  // shaders translated; draws with a failed pipeline are skipped
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
  bool wireframe = false;  // VK_POLYGON_MODE_LINE (POINT is drawn filled, logged once)
  float depthBias[3] = {0, 0, 0};  // constant, clamp, slope
  // MSAA: the game's rasterizationSamples; state.samples is the effective count at creation. A
  // draw into a pass of another (effective) count uses a variant with the pass's count (Metal
  // requires pipeline == attachments), created on first use and kept here.
  uint8_t gameSamples = 1;
  std::array<uint64_t, 3> inputs{};  // recipe stage inputs (variants are recorded too)
  mutable std::mutex variantMutex;
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
  uint16_t colorIntKinds = 0;  // SpirvLowerOptions::colorIntKinds of the fragment stage
  struct TexelVariant {
    std::vector<vpgpu::Format> formats;  // parallel to `texel`
    bool ok = false;
    vpgpu::PipelineState state;
    vpgpu::PipelineKey key;
    std::array<uint64_t, 3> inputs{};
    std::vector<std::pair<uint8_t, vpgpu::PipelineKey>> samples;  // MSAA variants of this one
  };
  mutable std::deque<TexelVariant> texelVariants;  // deque: stable addresses; under variantMutex
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
};

struct CommandPool {
  uint32_t family = 0;
  std::vector<CommandBuffer*> buffers;
};

struct Device : Dispatchable {
  PhysicalDevice* physical = nullptr;
  std::unique_ptr<Platform> platform;
  std::unique_ptr<vpgpu::Device> gpu;
  std::unique_ptr<RecipeStore> recipes;  // pipeline recipes for prewarming the next run
  Queue queue;
  bool multiview = false, densityMap = false, metalObjects = false;
  std::mutex submitMutex;
  std::mutex resourceMutex;  // gpu->create*/destroy from recording threads (upload arenas)
  // GPU indirect draws that need their arguments rewritten (multiview: instances x views; draw-count
  // buffers): a small compute kernel run before the render pass (pipelines.cpp indirectPatch).
  std::once_flag indirectPatchOnce;
  vpgpu::PipelineKey indirectPatchKey{};
  bool indirectPatchOk = false;
  uint64_t submits = 0;
  std::vector<DeferredBatch> deferred;  // submitMutex; queue order (front = next to run)
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
// entry.cpp: runs the deferred batches whose timeline waits can now be met (takes submitMutex);
// called after a host vkSignalSemaphore.
void submitDeferred(Device*);

// Physical device model (physical_device.cpp).
const VkPhysicalDeviceProperties& deviceProperties();
const VkPhysicalDeviceMemoryProperties& memoryProperties();
// Feature bits of a format for optimal tiling / buffers (0 = unsupported).
VkFormatFeatureFlags formatFeatures(VkFormat, bool buffer);
vpgpu::Format toCoreFormat(VkFormat);
std::vector<VkExtensionProperties> instanceExtensions();
std::vector<VkExtensionProperties> deviceExtensions();
std::vector<VkExtensionProperties> hiddenDeviceExtensions();

// Entry points (entry.cpp): null for names the driver does not implement (logged once).
PFN_vkVoidFunction lookup(const char* name, bool deviceLevel);
PFN_vkVoidFunction lookupPhase3(const char* name);  // pipelines.cpp
PFN_vkVoidFunction lookupExt(const char* name);     // ext.cpp

template <class T, class H>
T* from(H h) { return reinterpret_cast<T*>(h); }
template <class H, class T>
H to(T* p) { return reinterpret_cast<H>(p); }

}  // namespace vpvk
