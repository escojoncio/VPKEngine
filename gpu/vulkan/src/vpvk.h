// VPVulkan internals: the Vulkan objects of VPEngine's own driver, written from the Vulkan
// specification on top of gpu/core. Platform-neutral; the Metal glue sits behind vpvk::Platform.
#pragma once

#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_metal.h>
#include <vulkan/vk_icd.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "vpgpu/core/device.h"
#include "vpvk/platform.h"

namespace vpvk {

// Every log line of the driver starts with "vk." (one grep finds them all in the visor log).
void log(const std::string& line);
void logf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

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
};

struct ImageView {
  Image* image = nullptr;
  VkImageViewType type = VK_IMAGE_VIEW_TYPE_2D;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkImageSubresourceRange range{};
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
};
struct Semaphore {
  uint64_t value = 0;
};
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
  Queue queue;
  bool multiview = false, densityMap = false, metalObjects = false;
  std::mutex submitMutex;
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

template <class T, class H>
T* from(H h) { return reinterpret_cast<T*>(h); }
template <class H, class T>
H to(T* p) { return reinterpret_cast<H>(p); }

}  // namespace vpvk
