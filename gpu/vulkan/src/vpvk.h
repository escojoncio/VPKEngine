// VPVulkan internals: the Vulkan objects of VPKEngine's own driver, written from the Vulkan
// specification on top of gpu/core. Platform-neutral; the Metal glue sits behind vpvk::Platform.
#pragma once

#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_metal.h>
#include <vulkan/vk_icd.h>

#include <array>
#include <atomic>
#include <cstdint>
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
bool forceSingleSample();  // config.cpp: vpvk_config_set_force_single_sample

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

struct DeviceMemory {
  VkDeviceSize size = 0;
  uint32_t typeIndex = 0;
  uint8_t* host = nullptr;  // page-aligned; unified memory: GPU buffers alias it in place
  bool mapped = false;
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
  uint8_t forcedFrom = 0;  // >1: the game asked for this many samples, created 1x (forceSingleSample)
  uint8_t gameSamples = 1;  // what the game asked for (desc.samples is what was created)
  // id<MTLTexture> (unretained) cached at bind for textures allocated there: vkGetMTLTextureMVK
  // and vkExportMetalObjectsEXT answer from it without touching the backend's maps (other threads).
  void* native = nullptr;
};

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
// A submission's timeline signals (VkTimelineSemaphoreSubmitInfo) once its frames finish.
void timelineSignals(const VkSubmitInfo& si, uint64_t frames);
struct Event {
  std::atomic<bool> set{false};
};

struct RenderPassObj {
  std::vector<VkAttachmentDescription2> attachments;
  struct Subpass {
    std::vector<VkAttachmentReference2> colors, resolves, inputs;
    VkAttachmentReference2 depth{};
    bool hasDepth = false;
    uint32_t viewMask = 0;
  };
  std::vector<Subpass> subpasses;
  int32_t densityMap = -1;  // VK_EXT_fragment_density_map attachment index
};

// ---- shaders, descriptors, pipelines (pipelines.cpp) ----
struct ShaderModuleObj {
  spirv::SpirvModule* module = nullptr;
  std::string error;  // parse error (the pipeline reports it)
  // Kept only while recipes are recorded: the stage inputs reference the module by content hash
  // and the store saves the words once, so a later translator version can retranslate them.
  std::vector<uint32_t> words;
  uint64_t hash = 0;
  ~ShaderModuleObj() { spirv::release(module); }
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
  VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  std::vector<VkDynamicState> dynamic;
  bool hasViewport = false, hasScissor = false;
  VkViewport viewport{};
  VkRect2D scissor{};
  uint32_t stencilRef = 0;
  float blendConstants[4] = {0, 0, 0, 0};
  bool depthBiasEnable = false, depthClamp = false;
  bool cullBoth = false;  // VK_CULL_MODE_FRONT_AND_BACK: every triangle is culled
  float depthBias[3] = {0, 0, 0};  // constant, clamp, slope
  // MSAA: the game's rasterizationSamples; state.samples is the effective count at creation. A
  // draw into a pass of another (effective) count uses a variant with the pass's count (Metal
  // requires pipeline == attachments), created on first use and kept here.
  uint8_t gameSamples = 1;
  std::array<uint64_t, 3> inputs{};  // recipe stage inputs (variants are recorded too)
  mutable std::mutex variantMutex;
  mutable std::vector<std::pair<uint8_t, vpgpu::PipelineKey>> variants;
  bool isDynamic(VkDynamicState s) const {
    for (auto d : dynamic)
      if (d == s) return true;
    return false;
  }
};

struct QueryPoolObj {
  VkQueryType type = VK_QUERY_TYPE_OCCLUSION;
  uint32_t count = 0;
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
  uint64_t submits = 0;
};

// Physical device model (physical_device.cpp).
const VkPhysicalDeviceProperties& deviceProperties();
const VkPhysicalDeviceMemoryProperties& memoryProperties();
// Feature bits of a format for optimal tiling / buffers (0 = unsupported).
VkFormatFeatureFlags formatFeatures(VkFormat, bool buffer);
vpgpu::Format toCoreFormat(VkFormat);
std::vector<VkExtensionProperties> instanceExtensions();
std::vector<VkExtensionProperties> deviceExtensions();

// Entry points (entry.cpp): null for names the driver does not implement (logged once).
PFN_vkVoidFunction lookup(const char* name, bool deviceLevel);
PFN_vkVoidFunction lookupPhase3(const char* name);  // pipelines.cpp
PFN_vkVoidFunction lookupExt(const char* name);     // ext.cpp

template <class T, class H>
T* from(H h) { return reinterpret_cast<T*>(h); }
template <class H, class T>
H to(T* p) { return reinterpret_cast<H>(p); }

}  // namespace vpvk
