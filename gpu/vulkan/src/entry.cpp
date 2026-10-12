// VPVulkan entry points: phase 1 (instance, physical device, device, queue, memory, buffers, images,
// views, samplers, VK_EXT_metal_objects), phase 2 (command pools/buffers as deferred lists, render
// passes, framebuffers) and the synchronisation a frame loop needs (fences, semaphores, submit).
// Every function is exported under its Vulkan name so dlsym() on the dylib finds it, as a client
// that loads a driver directly (Klepton's kl_vulkan.c) does for a few vkCmd* entry points.
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <unordered_map>

#include "commands.h"
#include "vpvk.h"

#define VPVK_EXPORT extern "C" __attribute__((visibility("default")))

namespace vpvk {

void log(const std::string& line) {
  static std::mutex m;
  std::lock_guard<std::mutex> lock(m);
  std::fprintf(stderr, "%s\n", line.c_str());
}

// The game's density-map attachment is never used to shade (its foveation is off); the
// attachment stays in the list as a plain unused attachment and rp->densityMap stays -1.
static void logFdmIgnored(uint32_t att) {
  static std::atomic<unsigned> n{0};
  if (n.fetch_add(1) < 8) logf("vk.fov ignored=render_pass_density_map attachment=%u", att);
}

void logf(const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  log(buf);
}

CommandBuffer::~CommandBuffer() = default;

namespace {

uint64_t alignUp(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }
constexpr uint64_t kPage = 16384;  // Apple page size: newBufferWithBytesNoCopy needs it

// Fills a Vulkan "two-call" array query.
template <class T>
VkResult fillArray(const std::vector<T>& src, uint32_t* count, T* out) {
  if (!out) {
    *count = static_cast<uint32_t>(src.size());
    return VK_SUCCESS;
  }
  const uint32_t n = std::min<uint32_t>(*count, static_cast<uint32_t>(src.size()));
  std::copy_n(src.begin(), n, out);
  *count = n;
  return n < src.size() ? VK_INCOMPLETE : VK_SUCCESS;
}

bool hasExt(const std::vector<VkExtensionProperties>& v, const char* name) {
  for (const auto& e : v)
    if (std::strcmp(e.extensionName, name) == 0) return true;
  return false;
}

const void* findNext(const void* chain, VkStructureType t) {
  for (auto* s = static_cast<const VkBaseInStructure*>(chain); s; s = s->pNext)
    if (s->sType == t) return s;
  return nullptr;
}
void* findNextOut(void* chain, VkStructureType t) {
  for (auto* s = static_cast<VkBaseOutStructure*>(chain); s; s = s->pNext)
    if (s->sType == t) return s;
  return nullptr;
}

vpgpu::TextureDesc imageDesc(const VkImageCreateInfo& ci) {
  vpgpu::TextureDesc d;
  d.format = toCoreFormat(ci.format);
  d.width = ci.extent.width;
  d.height = ci.extent.height;
  d.depth = ci.extent.depth;
  d.mips = static_cast<uint16_t>(ci.mipLevels);
  d.layers = static_cast<uint16_t>(ci.arrayLayers);
  d.samples = static_cast<uint8_t>(ci.samples);
  switch (ci.imageType) {
    case VK_IMAGE_TYPE_1D: d.type = vpgpu::TextureType::Tex1D; break;
    case VK_IMAGE_TYPE_3D: d.type = vpgpu::TextureType::Tex3D; break;
    default:
      d.type = (ci.flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) && ci.arrayLayers % 6 == 0
                   ? vpgpu::TextureType::Cube : vpgpu::TextureType::Tex2D;
  }
  d.usage = 0;
  if (ci.usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT |
                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
    d.usage |= vpgpu::usage::sampled;
  if (ci.usage & VK_IMAGE_USAGE_STORAGE_BIT) d.usage |= vpgpu::usage::storage;
  if (ci.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) d.usage |= vpgpu::usage::colorTarget;
  if (ci.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) d.usage |= vpgpu::usage::depthTarget;
  if (!d.usage) d.usage = vpgpu::usage::sampled;
  // Transient attachments never need memory: the core keeps them on tile when it can.
  d.transient = (ci.usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) != 0;
  // Anything sampled may be read in a later frame (Vulkan gives no frame boundary).
  d.persistent = !d.transient;
  return d;
}

uint64_t imageBytes(const vpgpu::TextureDesc& d) {
  const vpgpu::FormatInfo& fi = vpgpu::formatInfo(d.format);
  uint64_t total = 0;
  for (uint32_t m = 0; m < d.mips; ++m) {
    const uint64_t w = std::max<uint32_t>(1, d.width >> m), h = std::max<uint32_t>(1, d.height >> m);
    const uint64_t z = std::max<uint32_t>(1, d.depth >> m);
    total += (w + fi.blockSize - 1) / fi.blockSize * ((h + fi.blockSize - 1) / fi.blockSize) * z * fi.bytesPerBlock;
  }
  return std::max<uint64_t>(alignUp(total * d.layers * std::max<uint8_t>(1, d.samples), 256), 256);
}

std::mutex gUnimplMutex;
std::set<std::string> gUnimpl;

}  // namespace
}  // namespace vpvk

using namespace vpvk;

// ---------------------------------------------------------------------------------------------
// Instance
// ---------------------------------------------------------------------------------------------
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceVersion(uint32_t* v) {
  *v = VK_API_VERSION_1_1;
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(
    const char* layer, uint32_t* count, VkExtensionProperties* out) {
  if (layer) return VK_ERROR_LAYER_NOT_PRESENT;
  return fillArray(instanceExtensions(), count, out);
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t* count,
                                                                             VkLayerProperties*) {
  *count = 0;
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(const VkInstanceCreateInfo* ci,
                                                            const VkAllocationCallbacks*,
                                                            VkInstance* out) {
  const auto have = instanceExtensions();
  for (uint32_t i = 0; i < ci->enabledExtensionCount; ++i)
    if (!hasExt(have, ci->ppEnabledExtensionNames[i])) {
      logf("vk.err what=instance_ext name=%s", ci->ppEnabledExtensionNames[i]);
      return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
  auto* inst = new Instance;
  inst->physical.instance = inst;
  if (const VkApplicationInfo* a = ci->pApplicationInfo) {
    inst->apiVersion = a->apiVersion ? a->apiVersion : VK_API_VERSION_1_0;
    if (a->pApplicationName) inst->appName = a->pApplicationName;
    if (a->pEngineName) inst->engineName = a->pEngineName;
  }
  logf("vk.instance app=%s engine=%s api=%u.%u exts=%u", inst->appName.c_str(),
       inst->engineName.c_str(), VK_API_VERSION_MAJOR(inst->apiVersion),
       VK_API_VERSION_MINOR(inst->apiVersion), ci->enabledExtensionCount);
  *out = to<VkInstance>(inst);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyInstance(VkInstance i, const VkAllocationCallbacks*) {
  delete from<Instance>(i);
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumeratePhysicalDevices(VkInstance i, uint32_t* count,
                                                                      VkPhysicalDevice* out) {
  std::vector<VkPhysicalDevice> v{to<VkPhysicalDevice>(&from<Instance>(i)->physical)};
  return fillArray(v, count, out);
}

// ---------------------------------------------------------------------------------------------
// Physical device
// ---------------------------------------------------------------------------------------------
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties(VkPhysicalDevice,
                                                                     VkPhysicalDeviceProperties* p) {
  *p = deviceProperties();
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties2(VkPhysicalDevice pd,
                                                                      VkPhysicalDeviceProperties2* p) {
  vkGetPhysicalDeviceProperties(pd, &p->properties);
  for (auto* s = static_cast<VkBaseOutStructure*>(p->pNext); s; s = s->pNext) {
    switch (s->sType) {
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES: {
        auto* m = reinterpret_cast<VkPhysicalDeviceMultiviewProperties*>(s);
        m->maxMultiviewViewCount = 2;  // Metal vertex amplification: 2 on Apple GPUs
        m->maxMultiviewInstanceIndex = (1u << 27) - 1;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_PROPERTIES_EXT: {
        auto* f = reinterpret_cast<VkPhysicalDeviceFragmentDensityMapPropertiesEXT*>(s);
        f->minFragmentDensityTexelSize = {32, 32};
        f->maxFragmentDensityTexelSize = {32, 32};
        f->fragmentDensityInvocations = VK_FALSE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES: {
        auto* id = reinterpret_cast<VkPhysicalDeviceIDProperties*>(s);
        std::memcpy(id->deviceUUID, deviceProperties().pipelineCacheUUID, VK_UUID_SIZE);
        std::memcpy(id->driverUUID, deviceProperties().pipelineCacheUUID, VK_UUID_SIZE);
        id->deviceLUIDValid = VK_FALSE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES: {
        auto* m = reinterpret_cast<VkPhysicalDeviceMaintenance3Properties*>(s);
        m->maxPerSetDescriptors = 1024;
        m->maxMemoryAllocationSize = 1ull << 32;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES: {
        auto* sg = reinterpret_cast<VkPhysicalDeviceSubgroupProperties*>(s);
        sg->subgroupSize = 32;  // Apple SIMD-group width
        sg->supportedStages = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
        sg->supportedOperations = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT |
                                  VK_SUBGROUP_FEATURE_BALLOT_BIT;
        sg->quadOperationsInAllStages = VK_FALSE;
        break;
      }
      default: break;
    }
  }
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures(VkPhysicalDevice,
                                                                   VkPhysicalDeviceFeatures* f) {
  *f = {};
  f->robustBufferAccess = VK_TRUE;
  f->fullDrawIndexUint32 = VK_TRUE;
  f->imageCubeArray = VK_TRUE;
  f->independentBlend = VK_TRUE;
  f->sampleRateShading = VK_TRUE;
  f->dualSrcBlend = VK_TRUE;
  f->depthClamp = VK_TRUE;
  f->depthBiasClamp = VK_TRUE;
  f->fillModeNonSolid = VK_TRUE;
  f->samplerAnisotropy = VK_TRUE;
  f->textureCompressionBC = VK_TRUE;  // Apple GPUs sample BCn, ETC2/EAC and ASTC LDR natively
  f->textureCompressionETC2 = VK_TRUE;
  f->textureCompressionASTC_LDR = VK_TRUE;  // square blocks only (4x4..12x12); others unsupported
  f->occlusionQueryPrecise = VK_TRUE;
  f->fragmentStoresAndAtomics = VK_TRUE;
  f->shaderImageGatherExtended = VK_TRUE;
  f->shaderStorageImageExtendedFormats = VK_TRUE;
  f->shaderClipDistance = VK_TRUE;
  f->shaderSampledImageArrayDynamicIndexing = VK_TRUE;
  f->shaderUniformBufferArrayDynamicIndexing = VK_TRUE;
  f->shaderStorageBufferArrayDynamicIndexing = VK_TRUE;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(VkPhysicalDevice pd,
                                                                    VkPhysicalDeviceFeatures2* f) {
  vkGetPhysicalDeviceFeatures(pd, &f->features);
  for (auto* s = static_cast<VkBaseOutStructure*>(f->pNext); s; s = s->pNext) {
    if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES) {
      auto* m = reinterpret_cast<VkPhysicalDeviceMultiviewFeatures*>(s);
      m->multiview = VK_TRUE;
      m->multiviewGeometryShader = VK_FALSE;
      m->multiviewTessellationShader = VK_FALSE;
    } else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_FEATURES_EXT) {
      auto* d = reinterpret_cast<VkPhysicalDeviceFragmentDensityMapFeaturesEXT*>(s);
      d->fragmentDensityMap = VK_FALSE;  // not advertised: foveation is VPEngine's (vpvk/config.h)
      d->fragmentDensityMapDynamic = VK_FALSE;
      d->fragmentDensityMapNonSubsampledImages = VK_FALSE;
    }
  }
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties(
    VkPhysicalDevice, uint32_t* count, VkQueueFamilyProperties* out) {
  VkQueueFamilyProperties q{};
  q.queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
  q.queueCount = 1;
  q.timestampValidBits = 64;
  q.minImageTransferGranularity = {1, 1, 1};
  fillArray(std::vector<VkQueueFamilyProperties>{q}, count, out);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties2(
    VkPhysicalDevice pd, uint32_t* count, VkQueueFamilyProperties2* out) {
  if (!out) return vkGetPhysicalDeviceQueueFamilyProperties(pd, count, nullptr);
  if (*count) vkGetPhysicalDeviceQueueFamilyProperties(pd, count, &out[0].queueFamilyProperties);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties(
    VkPhysicalDevice, VkPhysicalDeviceMemoryProperties* m) {
  *m = memoryProperties();
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties2(
    VkPhysicalDevice pd, VkPhysicalDeviceMemoryProperties2* m) {
  vkGetPhysicalDeviceMemoryProperties(pd, &m->memoryProperties);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties(VkPhysicalDevice,
                                                                           VkFormat f,
                                                                           VkFormatProperties* p) {
  p->optimalTilingFeatures = formatFeatures(f, false);
  // Linear images are only offered for transfers (staging): Metal linear textures are restrictive.
  p->linearTilingFeatures = p->optimalTilingFeatures &
                            (VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
                             VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);
  p->bufferFeatures = formatFeatures(f, true);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties2(VkPhysicalDevice pd,
                                                                            VkFormat f,
                                                                            VkFormatProperties2* p) {
  vkGetPhysicalDeviceFormatProperties(pd, f, &p->formatProperties);
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties(
    VkPhysicalDevice, VkFormat f, VkImageType type, VkImageTiling tiling, VkImageUsageFlags usage,
    VkImageCreateFlags, VkImageFormatProperties* out) {
  const VkFormatFeatureFlags ff = formatFeatures(f, false);
  if (!ff) return VK_ERROR_FORMAT_NOT_SUPPORTED;
  if ((usage & VK_IMAGE_USAGE_STORAGE_BIT) && !(ff & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
  if ((usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) && !(ff & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT))
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
  if ((usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) &&
      !(ff & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
  if (tiling == VK_IMAGE_TILING_LINEAR &&
      (usage & ~(VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)))
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
  const auto& l = deviceProperties().limits;
  *out = {};
  out->maxExtent = {l.maxImageDimension2D, type == VK_IMAGE_TYPE_1D ? 1u : l.maxImageDimension2D,
                    type == VK_IMAGE_TYPE_3D ? l.maxImageDimension3D : 1u};
  out->maxMipLevels = 15;
  out->maxArrayLayers = type == VK_IMAGE_TYPE_3D ? 1 : l.maxImageArrayLayers;
  out->sampleCounts = tiling == VK_IMAGE_TILING_OPTIMAL && type == VK_IMAGE_TYPE_2D
                          ? l.framebufferColorSampleCounts : VkSampleCountFlags{VK_SAMPLE_COUNT_1_BIT};
  out->maxResourceSize = 1ull << 32;
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties2(
    VkPhysicalDevice pd, const VkPhysicalDeviceImageFormatInfo2* in, VkImageFormatProperties2* out) {
  return vkGetPhysicalDeviceImageFormatProperties(pd, in->format, in->type, in->tiling, in->usage,
                                                  in->flags, &out->imageFormatProperties);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceSparseImageFormatProperties(
    VkPhysicalDevice, VkFormat, VkImageType, VkSampleCountFlagBits, VkImageUsageFlags, VkImageTiling,
    uint32_t* count, VkSparseImageFormatProperties*) {
  *count = 0;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(
    VkPhysicalDevice, const char* layer, uint32_t* count, VkExtensionProperties* out) {
  if (layer) return VK_ERROR_LAYER_NOT_PRESENT;
  return fillArray(deviceExtensions(), count, out);
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(VkPhysicalDevice,
                                                                           uint32_t* count,
                                                                           VkLayerProperties*) {
  *count = 0;
  return VK_SUCCESS;
}

// ---------------------------------------------------------------------------------------------
// Device and queue
// ---------------------------------------------------------------------------------------------
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice pd,
                                                          const VkDeviceCreateInfo* ci,
                                                          const VkAllocationCallbacks*,
                                                          VkDevice* out) {
  const auto have = deviceExtensions();
  auto* d = new Device;
  d->physical = from<PhysicalDevice>(pd);
  for (uint32_t i = 0; i < ci->enabledExtensionCount; ++i) {
    const char* e = ci->ppEnabledExtensionNames[i];
    // A game that enables the density map without checking (it is not advertised) is let through
    // with it ignored: refusing the device would stop the game for a feature it can live without.
    if (!std::strcmp(e, VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME) ||
        !std::strcmp(e, "VK_EXT_fragment_density_map2")) {
      logf("vk.fov ignored=device_ext name=%s", e);
      continue;
    }
    if (!hasExt(have, e)) {
      logf("vk.err what=device_ext name=%s", e);
      delete d;
      return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    if (!std::strcmp(e, VK_KHR_MULTIVIEW_EXTENSION_NAME)) d->multiview = true;
    if (!std::strcmp(e, VK_EXT_METAL_OBJECTS_EXTENSION_NAME)) d->metalObjects = true;
  }
  d->platform = createPlatform();
  d->gpu = std::make_unique<vpgpu::Device>(d->platform->backend(), 2, [](const std::string& s) { log(s); });
  d->queue.device = d;
  logf("vk.device exts=%u multiview=%d fdm=%d metal_objects=%d", ci->enabledExtensionCount,
       d->multiview, d->densityMap, d->metalObjects);
  *out = to<VkDevice>(d);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyDevice(VkDevice dv, const VkAllocationCallbacks*) {
  if (!dv) return;
  Device* d = from<Device>(dv);
  d->platform->waitIdle();
  d->gpu.reset();
  delete d;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetDeviceQueue(VkDevice dv, uint32_t family, uint32_t index,
                                                        VkQueue* out) {
  Device* d = from<Device>(dv);
  d->queue.family = family;
  d->queue.index = index;
  *out = to<VkQueue>(&d->queue);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetDeviceQueue2(VkDevice dv, const VkDeviceQueueInfo2* i,
                                                         VkQueue* out) {
  vkGetDeviceQueue(dv, i->queueFamilyIndex, i->queueIndex, out);
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkDeviceWaitIdle(VkDevice dv) {
  from<Device>(dv)->platform->waitIdle();
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkQueueWaitIdle(VkQueue q) {
  from<Queue>(q)->device->platform->waitIdle();
  return VK_SUCCESS;
}

// ---------------------------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------------------------
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkAllocateMemory(VkDevice, const VkMemoryAllocateInfo* ai,
                                                            const VkAllocationCallbacks*,
                                                            VkDeviceMemory* out) {
  if (ai->memoryTypeIndex >= memoryProperties().memoryTypeCount) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
  auto* m = new DeviceMemory;
  m->size = ai->allocationSize;
  m->typeIndex = ai->memoryTypeIndex;
  // Page-aligned host memory for every type: GPU buffers alias it in place (no staging).
  m->host = static_cast<uint8_t*>(std::aligned_alloc(kPage, alignUp(std::max<uint64_t>(m->size, 1), kPage)));
  if (!m->host) {
    delete m;
    logf("vk.err what=alloc size=%llu", static_cast<unsigned long long>(ai->allocationSize));
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
  }
  *out = to<VkDeviceMemory>(m);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice, VkDeviceMemory mem,
                                                    const VkAllocationCallbacks*) {
  if (!mem) return;
  DeviceMemory* m = from<DeviceMemory>(mem);
  std::free(m->host);
  delete m;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkMapMemory(VkDevice, VkDeviceMemory mem, VkDeviceSize off,
                                                       VkDeviceSize, VkMemoryMapFlags, void** out) {
  DeviceMemory* m = from<DeviceMemory>(mem);
  if (!(memoryProperties().memoryTypes[m->typeIndex].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
    return VK_ERROR_MEMORY_MAP_FAILED;
  m->mapped = true;
  *out = m->host + off;
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkUnmapMemory(VkDevice, VkDeviceMemory mem) {
  from<DeviceMemory>(mem)->mapped = false;
}

// Coherent unified memory: nothing to flush or invalidate.
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkFlushMappedMemoryRanges(VkDevice, uint32_t,
                                                                     const VkMappedMemoryRange*) {
  return VK_SUCCESS;
}
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkInvalidateMappedMemoryRanges(VkDevice, uint32_t,
                                                                          const VkMappedMemoryRange*) {
  return VK_SUCCESS;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetDeviceMemoryCommitment(VkDevice, VkDeviceMemory m,
                                                                   VkDeviceSize* out) {
  *out = from<DeviceMemory>(m)->size;
}

// ---------------------------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------------------------
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(VkDevice, const VkBufferCreateInfo* ci,
                                                          const VkAllocationCallbacks*, VkBuffer* out) {
  auto* b = new Buffer;
  b->size = ci->size;
  b->usage = ci->usage;
  *out = to<VkBuffer>(b);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyBuffer(VkDevice dv, VkBuffer buf,
                                                       const VkAllocationCallbacks*) {
  if (!buf) return;
  Buffer* b = from<Buffer>(buf);
  if (b->id) from<Device>(dv)->gpu->destroy(b->id);
  delete b;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements(VkDevice, VkBuffer buf,
                                                                     VkMemoryRequirements* r) {
  r->size = alignUp(std::max<VkDeviceSize>(from<Buffer>(buf)->size, 1), 256);
  r->alignment = 256;
  r->memoryTypeBits = 0x7;  // not the lazily-allocated type
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements2(
    VkDevice dv, const VkBufferMemoryRequirementsInfo2* in, VkMemoryRequirements2* out) {
  vkGetBufferMemoryRequirements(dv, in->buffer, &out->memoryRequirements);
  if (auto* ded = static_cast<VkMemoryDedicatedRequirements*>(
          findNextOut(out->pNext, VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS))) {
    ded->prefersDedicatedAllocation = VK_FALSE;
    ded->requiresDedicatedAllocation = VK_FALSE;
  }
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory(VkDevice dv, VkBuffer buf,
                                                              VkDeviceMemory mem, VkDeviceSize off) {
  Buffer* b = from<Buffer>(buf);
  DeviceMemory* m = from<DeviceMemory>(mem);
  b->memory = m;
  b->offset = off;
  vpgpu::BufferDesc d;
  d.size = b->size;
  d.guestMemory = m->host + off;
  d.label = "vk.buffer";
  b->id = from<Device>(dv)->gpu->createBuffer(std::move(d));
  return b->id ? VK_SUCCESS : VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory2(VkDevice dv, uint32_t n,
                                                               const VkBindBufferMemoryInfo* in) {
  for (uint32_t i = 0; i < n; ++i)
    if (VkResult r = vkBindBufferMemory(dv, in[i].buffer, in[i].memory, in[i].memoryOffset); r) return r;
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateBufferView(VkDevice, const VkBufferViewCreateInfo* ci,
                                                              const VkAllocationCallbacks*,
                                                              VkBufferView* out) {
  auto* v = new BufferView;
  v->buffer = from<Buffer>(ci->buffer);
  v->format = ci->format;
  v->offset = ci->offset;
  v->range = ci->range;
  *out = to<VkBufferView>(v);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyBufferView(VkDevice, VkBufferView v,
                                                           const VkAllocationCallbacks*) {
  delete from<BufferView>(v);
}

// ---------------------------------------------------------------------------------------------
// Images, views, samplers
// ---------------------------------------------------------------------------------------------
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateImage(VkDevice, const VkImageCreateInfo* ci,
                                                         const VkAllocationCallbacks*, VkImage* out) {
  vpgpu::TextureDesc desc = imageDesc(*ci);
  if (desc.format == vpgpu::Format::Undefined || !vpgpu::validate(desc)) {
    logf("vk.err what=image_desc format=%d %ux%ux%u mips=%u layers=%u", ci->format, ci->extent.width,
         ci->extent.height, ci->extent.depth, ci->mipLevels, ci->arrayLayers);
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
  }
  auto* img = new Image;
  img->info = *ci;
  img->info.pNext = nullptr;
  img->info.pQueueFamilyIndices = nullptr;
  img->desc = std::move(desc);
  *out = to<VkImage>(img);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyImage(VkDevice dv, VkImage im,
                                                      const VkAllocationCallbacks*) {
  if (!im) return;
  Image* i = from<Image>(im);
  if (i->id) from<Device>(dv)->gpu->destroy(i->id);
  delete i;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetImageMemoryRequirements(VkDevice, VkImage im,
                                                                    VkMemoryRequirements* r) {
  const Image* i = from<Image>(im);
  r->size = imageBytes(i->desc);
  r->alignment = 256;
  r->memoryTypeBits = (i->info.usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) ? 0xF : 0x7;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetImageMemoryRequirements2(
    VkDevice dv, const VkImageMemoryRequirementsInfo2* in, VkMemoryRequirements2* out) {
  vkGetImageMemoryRequirements(dv, in->image, &out->memoryRequirements);
  if (auto* ded = static_cast<VkMemoryDedicatedRequirements*>(
          findNextOut(out->pNext, VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS))) {
    ded->prefersDedicatedAllocation = VK_FALSE;
    ded->requiresDedicatedAllocation = VK_FALSE;
  }
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetImageSparseMemoryRequirements(
    VkDevice, VkImage, uint32_t* count, VkSparseImageMemoryRequirements*) {
  *count = 0;
}

// Texture storage is Metal's (private, tiled); the VkDeviceMemory only accounts for it. A texture
// that aliases another resource's memory is not supported (logged on first sight: phase 2b).
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkBindImageMemory(VkDevice dv, VkImage im, VkDeviceMemory mem,
                                                             VkDeviceSize off) {
  Image* i = from<Image>(im);
  i->memory = from<DeviceMemory>(mem);
  i->offset = off;
  i->id = from<Device>(dv)->gpu->createTexture(i->desc);
  return i->id ? VK_SUCCESS : VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkBindImageMemory2(VkDevice dv, uint32_t n,
                                                              const VkBindImageMemoryInfo* in) {
  for (uint32_t i = 0; i < n; ++i)
    if (VkResult r = vkBindImageMemory(dv, in[i].image, in[i].memory, in[i].memoryOffset); r) return r;
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetImageSubresourceLayout(VkDevice, VkImage im,
                                                                   const VkImageSubresource* sub,
                                                                   VkSubresourceLayout* out) {
  // Only meaningful for linear images; reported tightly packed, mips then layers.
  const vpgpu::TextureDesc& d = from<Image>(im)->desc;
  const vpgpu::FormatInfo& fi = vpgpu::formatInfo(d.format);
  uint64_t off = 0;
  for (uint32_t m = 0; m <= sub->mipLevel && m < d.mips; ++m) {
    const uint64_t w = std::max<uint32_t>(1, d.width >> m), h = std::max<uint32_t>(1, d.height >> m);
    const uint64_t row = (w + fi.blockSize - 1) / fi.blockSize * fi.bytesPerBlock;
    const uint64_t slice = row * ((h + fi.blockSize - 1) / fi.blockSize);
    if (m == sub->mipLevel) {
      out->offset = off + slice * sub->arrayLayer;
      out->size = slice;
      out->rowPitch = row;
      out->arrayPitch = out->depthPitch = slice;
      return;
    }
    off += slice * d.layers;
  }
  *out = {};
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateImageView(VkDevice dv, const VkImageViewCreateInfo* ci,
                                                             const VkAllocationCallbacks*,
                                                             VkImageView* out) {
  auto* v = new ImageView;
  v->image = from<Image>(ci->image);
  v->type = ci->viewType;
  v->format = ci->format;
  v->range = ci->subresourceRange;
  // A backend view when sampling it as the image itself would bind something else: another type
  // (2D of an array, cube of a layered image, 2D array of a single layer…), another format, a
  // mip/layer subset, or only the stencil aspect.
  const Image* im = v->image;
  if (im && im->id) {
    const vpgpu::TextureDesc& d = im->desc;
    Platform::ViewInfo info;
    const uint32_t mips = v->range.levelCount == VK_REMAINING_MIP_LEVELS ? d.mips - v->range.baseMipLevel : v->range.levelCount;
    const uint32_t layers = v->range.layerCount == VK_REMAINING_ARRAY_LAYERS ? d.layers - v->range.baseArrayLayer : v->range.layerCount;
    info.baseMip = uint16_t(v->range.baseMipLevel), info.mips = uint16_t(mips);
    info.baseLayer = uint16_t(v->range.baseArrayLayer), info.layers = uint16_t(layers);
    info.format = toCoreFormat(ci->format);
    switch (ci->viewType) {
      case VK_IMAGE_VIEW_TYPE_1D: info.type = vpgpu::TextureType::Tex1D; break;
      case VK_IMAGE_VIEW_TYPE_3D: info.type = vpgpu::TextureType::Tex3D; break;
      case VK_IMAGE_VIEW_TYPE_CUBE: info.type = vpgpu::TextureType::Cube; break;
      case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: info.type = vpgpu::TextureType::Cube, info.array = true; break;
      case VK_IMAGE_VIEW_TYPE_2D_ARRAY: info.type = vpgpu::TextureType::Tex2D, info.array = true; break;
      default: info.type = vpgpu::TextureType::Tex2D; break;
    }
    // What binding the image directly would give.
    const bool imgArray = d.type != vpgpu::TextureType::Tex3D && (d.type == vpgpu::TextureType::Cube ? d.layers > 6 : d.layers > 1);
    const bool sameType = info.type == d.type && info.array == imgArray;
    const bool sameFormat = info.format == d.format || info.format == vpgpu::Format::Undefined;
    const bool whole = info.baseMip == 0 && mips == d.mips && info.baseLayer == 0 && layers == d.layers;
    const bool depthStencil = vpgpu::formatInfo(d.format).depth() && vpgpu::formatInfo(d.format).stencil();
    if (depthStencil) info.format = d.format;  // depth-or-stencil aspect views keep the combined format
    if (!sameType || (!sameFormat && !depthStencil) || !whole) {
      v->view = from<Device>(dv)->platform->createTextureView(im->id, info);
    }
  }
  *out = to<VkImageView>(v);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyImageView(VkDevice dv, VkImageView v,
                                                          const VkAllocationCallbacks*) {
  if (!v) return;
  ImageView* iv = from<ImageView>(v);
  if (iv->view) from<Device>(dv)->platform->destroyTextureView(iv->view);
  delete iv;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateSampler(VkDevice dv, const VkSamplerCreateInfo* ci,
                                                           const VkAllocationCallbacks*, VkSampler* out) {
  auto addr = [](VkSamplerAddressMode m) {
    switch (m) {
      case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: return vpgpu::AddressMode::MirrorRepeat;
      case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: return vpgpu::AddressMode::ClampToEdge;
      case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER: return vpgpu::AddressMode::ClampToBorder;
      case VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE: return vpgpu::AddressMode::MirrorClampToEdge;
      default: return vpgpu::AddressMode::Repeat;
    }
  };
  vpgpu::SamplerDesc d;
  d.minFilter = ci->minFilter == VK_FILTER_NEAREST ? vpgpu::Filter::Nearest : vpgpu::Filter::Linear;
  d.magFilter = ci->magFilter == VK_FILTER_NEAREST ? vpgpu::Filter::Nearest : vpgpu::Filter::Linear;
  d.mipFilter = ci->mipmapMode == VK_SAMPLER_MIPMAP_MODE_NEAREST ? vpgpu::MipFilter::Nearest
                                                                 : vpgpu::MipFilter::Linear;
  d.u = addr(ci->addressModeU);
  d.v = addr(ci->addressModeV);
  d.w = addr(ci->addressModeW);
  d.maxAnisotropy = ci->anisotropyEnable ? static_cast<uint8_t>(std::clamp(ci->maxAnisotropy, 1.0f, 16.0f)) : 1;
  if (ci->compareEnable) d.compare = static_cast<vpgpu::CompareOp>(ci->compareOp);  // same order
  switch (ci->borderColor) {
    case VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK:
    case VK_BORDER_COLOR_INT_OPAQUE_BLACK: d.border = vpgpu::BorderColor::OpaqueBlack; break;
    case VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE:
    case VK_BORDER_COLOR_INT_OPAQUE_WHITE: d.border = vpgpu::BorderColor::OpaqueWhite; break;
    default: d.border = vpgpu::BorderColor::TransparentBlack;
  }
  d.lodMin = ci->minLod;
  d.lodMax = ci->maxLod;
  d.label = "vk.sampler";
  auto* s = new Sampler;
  s->id = from<Device>(dv)->gpu->createSampler(std::move(d));
  *out = to<VkSampler>(s);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroySampler(VkDevice dv, VkSampler sm,
                                                        const VkAllocationCallbacks*) {
  if (!sm) return;
  Sampler* s = from<Sampler>(sm);
  from<Device>(dv)->gpu->destroy(s->id);
  delete s;
}

// ---------------------------------------------------------------------------------------------
// VK_EXT_metal_objects: what Klepton's compositor path uses to reach each eye's MTLTexture.
// ---------------------------------------------------------------------------------------------
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkExportMetalObjectsEXT(VkDevice dv, VkExportMetalObjectsInfoEXT* info) {
  Device* d = from<Device>(dv);
  for (auto* s = static_cast<VkBaseOutStructure*>(const_cast<void*>(info->pNext)); s; s = s->pNext) {
    switch (s->sType) {
      case VK_STRUCTURE_TYPE_EXPORT_METAL_DEVICE_INFO_EXT:
        reinterpret_cast<VkExportMetalDeviceInfoEXT*>(s)->mtlDevice =
            static_cast<MTLDevice_id>(d->platform->nativeDevice());
        break;
      case VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT: {
        auto* t = reinterpret_cast<VkExportMetalTextureInfoEXT*>(s);
        const Image* img = t->image ? from<Image>(t->image)
                         : t->imageView ? from<ImageView>(t->imageView)->image : nullptr;
        t->mtlTexture = img && img->id ? static_cast<MTLTexture_id>(d->platform->nativeTexture(img->id)) : nullptr;
        if (!t->mtlTexture) log("vk.err what=export_texture (image not bound)");
        break;
      }
      case VK_STRUCTURE_TYPE_EXPORT_METAL_BUFFER_INFO_EXT: {
        auto* b = reinterpret_cast<VkExportMetalBufferInfoEXT*>(s);
        // A VkDeviceMemory has no single MTLBuffer here (buffers alias it per bind): unsupported.
        b->mtlBuffer = nullptr;
        log("vk.err what=export_buffer (per-memory MTLBuffer not exposed)");
        break;
      }
      default: break;
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Render passes and framebuffers
// ---------------------------------------------------------------------------------------------
namespace {
VkAttachmentReference2 ref2(const VkAttachmentReference& r) {
  VkAttachmentReference2 o{VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2};
  o.attachment = r.attachment;
  o.layout = r.layout;
  return o;
}
}  // namespace

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateRenderPass(VkDevice, const VkRenderPassCreateInfo* ci,
                                                              const VkAllocationCallbacks*,
                                                              VkRenderPass* out) {
  auto* rp = new RenderPassObj;
  for (uint32_t i = 0; i < ci->attachmentCount; ++i) {
    const VkAttachmentDescription& a = ci->pAttachments[i];
    VkAttachmentDescription2 b{VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2};
    b.flags = a.flags;
    b.format = a.format;
    b.samples = a.samples;
    b.loadOp = a.loadOp;
    b.storeOp = a.storeOp;
    b.stencilLoadOp = a.stencilLoadOp;
    b.stencilStoreOp = a.stencilStoreOp;
    b.initialLayout = a.initialLayout;
    b.finalLayout = a.finalLayout;
    rp->attachments.push_back(b);
  }
  const auto* mv = static_cast<const VkRenderPassMultiviewCreateInfo*>(
      findNext(ci->pNext, VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO));
  for (uint32_t s = 0; s < ci->subpassCount; ++s) {
    const VkSubpassDescription& sd = ci->pSubpasses[s];
    RenderPassObj::Subpass sp;
    for (uint32_t i = 0; i < sd.colorAttachmentCount; ++i) {
      sp.colors.push_back(ref2(sd.pColorAttachments[i]));
      if (sd.pResolveAttachments) sp.resolves.push_back(ref2(sd.pResolveAttachments[i]));
    }
    for (uint32_t i = 0; i < sd.inputAttachmentCount; ++i) sp.inputs.push_back(ref2(sd.pInputAttachments[i]));
    if (sd.pDepthStencilAttachment) {
      sp.depth = ref2(*sd.pDepthStencilAttachment);
      sp.hasDepth = true;
    }
    if (mv && s < mv->subpassCount) sp.viewMask = mv->pViewMasks[s];
    rp->subpasses.push_back(std::move(sp));
  }
  if (const auto* fdm = static_cast<const VkRenderPassFragmentDensityMapCreateInfoEXT*>(
          findNext(ci->pNext, VK_STRUCTURE_TYPE_RENDER_PASS_FRAGMENT_DENSITY_MAP_CREATE_INFO_EXT)))
    logFdmIgnored(fdm->fragmentDensityMapAttachment.attachment);
  *out = to<VkRenderPass>(rp);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateRenderPass2(VkDevice, const VkRenderPassCreateInfo2* ci,
                                                               const VkAllocationCallbacks*,
                                                               VkRenderPass* out) {
  auto* rp = new RenderPassObj;
  rp->attachments.assign(ci->pAttachments, ci->pAttachments + ci->attachmentCount);
  for (auto& a : rp->attachments) a.pNext = nullptr;
  for (uint32_t s = 0; s < ci->subpassCount; ++s) {
    const VkSubpassDescription2& sd = ci->pSubpasses[s];
    RenderPassObj::Subpass sp;
    sp.colors.assign(sd.pColorAttachments, sd.pColorAttachments + sd.colorAttachmentCount);
    if (sd.pResolveAttachments)
      sp.resolves.assign(sd.pResolveAttachments, sd.pResolveAttachments + sd.colorAttachmentCount);
    sp.inputs.assign(sd.pInputAttachments, sd.pInputAttachments + sd.inputAttachmentCount);
    if (sd.pDepthStencilAttachment) {
      sp.depth = *sd.pDepthStencilAttachment;
      sp.hasDepth = true;
    }
    sp.viewMask = sd.viewMask;
    rp->subpasses.push_back(std::move(sp));
  }
  if (const auto* fdm = static_cast<const VkRenderPassFragmentDensityMapCreateInfoEXT*>(
          findNext(ci->pNext, VK_STRUCTURE_TYPE_RENDER_PASS_FRAGMENT_DENSITY_MAP_CREATE_INFO_EXT)))
    logFdmIgnored(fdm->fragmentDensityMapAttachment.attachment);
  *out = to<VkRenderPass>(rp);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyRenderPass(VkDevice, VkRenderPass rp,
                                                           const VkAllocationCallbacks*) {
  delete from<RenderPassObj>(rp);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetRenderAreaGranularity(VkDevice, VkRenderPass, VkExtent2D* g) {
  *g = {1, 1};
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateFramebuffer(VkDevice, const VkFramebufferCreateInfo* ci,
                                                               const VkAllocationCallbacks*,
                                                               VkFramebuffer* out) {
  auto* fb = new Framebuffer;
  for (uint32_t i = 0; i < ci->attachmentCount && ci->pAttachments; ++i)
    fb->views.push_back(from<ImageView>(ci->pAttachments[i]));
  fb->width = ci->width;
  fb->height = ci->height;
  fb->layers = ci->layers;
  *out = to<VkFramebuffer>(fb);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyFramebuffer(VkDevice, VkFramebuffer fb,
                                                            const VkAllocationCallbacks*) {
  delete from<Framebuffer>(fb);
}

// ---------------------------------------------------------------------------------------------
// Command pools and buffers (deferred lists, translated at submit)
// ---------------------------------------------------------------------------------------------
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateCommandPool(VkDevice, const VkCommandPoolCreateInfo* ci,
                                                               const VkAllocationCallbacks*,
                                                               VkCommandPool* out) {
  auto* p = new CommandPool;
  p->family = ci->queueFamilyIndex;
  *out = to<VkCommandPool>(p);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyCommandPool(VkDevice, VkCommandPool cp,
                                                            const VkAllocationCallbacks*) {
  if (!cp) return;
  CommandPool* p = from<CommandPool>(cp);
  for (CommandBuffer* c : p->buffers) delete c;
  delete p;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetCommandPool(VkDevice, VkCommandPool cp,
                                                              VkCommandPoolResetFlags) {
  for (CommandBuffer* c : from<CommandPool>(cp)->buffers) {
    c->rec->reset();
    c->state = CommandBuffer::State::Initial;
  }
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkAllocateCommandBuffers(VkDevice dv,
                                                                    const VkCommandBufferAllocateInfo* ai,
                                                                    VkCommandBuffer* out) {
  CommandPool* p = from<CommandPool>(ai->commandPool);
  for (uint32_t i = 0; i < ai->commandBufferCount; ++i) {
    auto* c = new CommandBuffer;
    c->device = from<Device>(dv);
    c->pool = p;
    c->level = ai->level;
    c->rec = std::make_unique<Recorder>();
    c->rec->device = c->device;
    p->buffers.push_back(c);
    out[i] = to<VkCommandBuffer>(c);
  }
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkFreeCommandBuffers(VkDevice, VkCommandPool cp, uint32_t n,
                                                            const VkCommandBuffer* bufs) {
  CommandPool* p = from<CommandPool>(cp);
  for (uint32_t i = 0; i < n; ++i) {
    if (!bufs[i]) continue;
    CommandBuffer* c = from<CommandBuffer>(bufs[i]);
    p->buffers.erase(std::remove(p->buffers.begin(), p->buffers.end(), c), p->buffers.end());
    delete c;
  }
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkBeginCommandBuffer(VkCommandBuffer cb,
                                                                const VkCommandBufferBeginInfo*) {
  CommandBuffer* c = from<CommandBuffer>(cb);
  c->rec->reset();  // implicit reset (pools are always resettable here)
  c->state = CommandBuffer::State::Recording;
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEndCommandBuffer(VkCommandBuffer cb) {
  from<CommandBuffer>(cb)->state = CommandBuffer::State::Executable;
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetCommandBuffer(VkCommandBuffer cb,
                                                                VkCommandBufferResetFlags) {
  CommandBuffer* c = from<CommandBuffer>(cb);
  c->rec->reset();
  c->state = CommandBuffer::State::Initial;
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBeginRenderPass(VkCommandBuffer cb,
                                                            const VkRenderPassBeginInfo* bi,
                                                            VkSubpassContents) {
  OpBeginPass op;
  op.pass = from<RenderPassObj>(bi->renderPass);
  op.framebuffer = from<Framebuffer>(bi->framebuffer);
  op.clears.assign(bi->pClearValues, bi->pClearValues + bi->clearValueCount);
  op.area = bi->renderArea;
  Recorder& r = *from<CommandBuffer>(cb)->rec;
  r.bound.pass = op.pass;
  r.bound.subpass = 0;
  r.bound.targetWidth = uint32_t(op.area.offset.x) + op.area.extent.width;
  r.bound.targetHeight = uint32_t(op.area.offset.y) + op.area.extent.height;
  r.ops.emplace_back(std::move(op));
  r.inPass = true;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBeginRenderPass2(VkCommandBuffer cb,
                                                             const VkRenderPassBeginInfo* bi,
                                                             const VkSubpassBeginInfo* si) {
  vkCmdBeginRenderPass(cb, bi, si ? si->contents : VK_SUBPASS_CONTENTS_INLINE);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdEndRenderPass(VkCommandBuffer cb) {
  Recorder& r = *from<CommandBuffer>(cb)->rec;
  r.ops.emplace_back(OpEndPass{});
  r.inPass = false;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdEndRenderPass2(VkCommandBuffer cb, const VkSubpassEndInfo*) {
  vkCmdEndRenderPass(cb);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier(
    VkCommandBuffer cb, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags, uint32_t,
    const VkMemoryBarrier*, uint32_t, const VkBufferMemoryBarrier*, uint32_t, const VkImageMemoryBarrier*) {
  ++from<CommandBuffer>(cb)->rec->barriers;  // the core places barriers from real hazards
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer cb, VkBuffer src, VkBuffer dst,
                                                       uint32_t n, const VkBufferCopy* regions) {
  Recorder& r = *from<CommandBuffer>(cb)->rec;
  for (uint32_t i = 0; i < n; ++i) r.ops.emplace_back(OpCopyBuffer{from<Buffer>(src), from<Buffer>(dst), regions[i]});
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdFillBuffer(VkCommandBuffer cb, VkBuffer dst, VkDeviceSize off,
                                                       VkDeviceSize size, uint32_t data) {
  Recorder& r = *from<CommandBuffer>(cb)->rec;
  const Buffer* b = from<Buffer>(dst);
  const VkDeviceSize n = size == VK_WHOLE_SIZE ? (b->size - off) & ~VkDeviceSize(3) : size;
  if (data == (data & 0xFF) * 0x01010101u) {
    r.ops.emplace_back(OpFillBuffer{b, off, n, data});
    return;
  }
  // A word pattern whose bytes differ: written into upload memory and copied.
  std::vector<uint32_t> words(n / 4, data);
  OpUpload u;
  u.src = r.arena.put(r.device, words.data(), n, u.srcOffset);
  u.dst = b, u.dstOffset = off, u.size = n;
  r.ops.emplace_back(u);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdClearColorImage(VkCommandBuffer cb, VkImage im, VkImageLayout,
                                                            const VkClearColorValue* color, uint32_t n,
                                                            const VkImageSubresourceRange* ranges) {
  Recorder& r = *from<CommandBuffer>(cb)->rec;
  for (uint32_t i = 0; i < n; ++i) r.ops.emplace_back(OpClearColor{from<Image>(im), *color, ranges[i]});
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer src, VkImage dst,
                                                              VkImageLayout, uint32_t n,
                                                              const VkBufferImageCopy* regions) {
  Recorder& r = *from<CommandBuffer>(cb)->rec;
  for (uint32_t i = 0; i < n; ++i)
    r.ops.emplace_back(OpCopyBufferToImage{from<Buffer>(src), from<Image>(dst), regions[i]});
}

// ---------------------------------------------------------------------------------------------
// Synchronisation and submission
// ---------------------------------------------------------------------------------------------
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateFence(VkDevice, const VkFenceCreateInfo* ci,
                                                         const VkAllocationCallbacks*, VkFence* out) {
  auto* f = new Fence;
  f->signaled = (ci->flags & VK_FENCE_CREATE_SIGNALED_BIT) != 0;
  *out = to<VkFence>(f);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyFence(VkDevice, VkFence f, const VkAllocationCallbacks*) {
  delete from<Fence>(f);
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetFences(VkDevice, uint32_t n, const VkFence* fs) {
  for (uint32_t i = 0; i < n; ++i) from<Fence>(fs[i])->signaled = false;
  return VK_SUCCESS;
}

// A fence is signaled when its submission's GPU work is done. Until the backend reports completion
// per submission, a submitted fence is waited for with a full waitIdle (correct, coarse).
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetFenceStatus(VkDevice dv, VkFence f) {
  Fence* fe = from<Fence>(f);
  if (!fe->signaled) return VK_NOT_READY;
  // Signaled by a submission whose GPU work may still run: the client would reuse its command
  // buffers (upload memory, descriptors) as soon as this says done, so it waits first.
  if (fe->submitted) {
    from<Device>(dv)->platform->waitIdle();
    fe->submitted = false;
  }
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(VkDevice dv, uint32_t n, const VkFence* fs,
                                                           VkBool32, uint64_t) {
  from<Device>(dv)->platform->waitIdle();
  for (uint32_t i = 0; i < n; ++i)
    if (!from<Fence>(fs[i])->signaled) return VK_TIMEOUT;  // never submitted
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateSemaphore(VkDevice, const VkSemaphoreCreateInfo*,
                                                             const VkAllocationCallbacks*, VkSemaphore* out) {
  *out = to<VkSemaphore>(new Semaphore);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroySemaphore(VkDevice, VkSemaphore s,
                                                          const VkAllocationCallbacks*) {
  delete from<Semaphore>(s);
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateEvent(VkDevice, const VkEventCreateInfo*,
                                                         const VkAllocationCallbacks*, VkEvent* out) {
  *out = to<VkEvent>(new Event);
  return VK_SUCCESS;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyEvent(VkDevice, VkEvent e, const VkAllocationCallbacks*) {
  delete from<Event>(e);
}
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetEventStatus(VkDevice, VkEvent e) {
  return from<Event>(e)->set ? VK_EVENT_SET : VK_EVENT_RESET;
}
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkSetEvent(VkDevice, VkEvent e) {
  from<Event>(e)->set = true;
  return VK_SUCCESS;
}
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetEvent(VkDevice, VkEvent e) {
  from<Event>(e)->set = false;
  return VK_SUCCESS;
}

// One queue, executed in order: semaphores between submissions of this queue are already satisfied
// by submission order, so they need no GPU work. All command buffers of one vkQueueSubmit become one
// core frame, which lets the frame compiler merge passes across command-buffer boundaries.
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(VkQueue q, uint32_t n, const VkSubmitInfo* si,
                                                         VkFence fence) {
  Device* d = from<Queue>(q)->device;
  std::lock_guard<std::mutex> lock(d->submitMutex);
  vpgpu::Frame frame;
  TranslateStats st;
  uint32_t barriers = 0, cbs = 0;
  for (uint32_t i = 0; i < n; ++i)
    for (uint32_t j = 0; j < si[i].commandBufferCount; ++j) {
      const CommandBuffer* c = from<CommandBuffer>(si[i].pCommandBuffers[j]);
      translate(*c->rec, frame, st);
      barriers += c->rec->barriers;
      ++cbs;
    }
  if (!frame.commands().empty()) d->gpu->submit(frame);
  ++d->submits;
  if (st.dropped || st.skippedDraws || st.skippedDispatches || d->submits <= 3 || d->submits % 600 == 0)
    logf("vk.submit n=%llu cbs=%u passes=%u draws=%u dispatches=%u copies=%u clears=%u vk_barriers=%u "
         "draws_skipped=%u dispatch_skipped=%u dropped=%u",
         static_cast<unsigned long long>(d->submits), cbs, st.passes, st.draws, st.dispatches, st.copies, st.clears,
         barriers, st.skippedDraws, st.skippedDispatches, st.dropped);
  if (fence) {
    from<Fence>(fence)->signaled = true;  // see vkWaitForFences / vkGetFenceStatus: wait idle first
    from<Fence>(fence)->submitted = true;
  }
  return VK_SUCCESS;
}

// ---------------------------------------------------------------------------------------------
// Proc addresses
// ---------------------------------------------------------------------------------------------
namespace {

struct Entry {
  const char* name;
  PFN_vkVoidFunction fn;
  bool device;
};

#define E(f, dev) {#f, reinterpret_cast<PFN_vkVoidFunction>(&f), dev}
const Entry kEntries[] = {
    E(vkEnumerateInstanceVersion, false), E(vkEnumerateInstanceExtensionProperties, false),
    E(vkEnumerateInstanceLayerProperties, false), E(vkCreateInstance, false), E(vkDestroyInstance, false),
    E(vkEnumeratePhysicalDevices, false), E(vkGetPhysicalDeviceProperties, false),
    E(vkGetPhysicalDeviceProperties2, false), E(vkGetPhysicalDeviceFeatures, false),
    E(vkGetPhysicalDeviceFeatures2, false), E(vkGetPhysicalDeviceQueueFamilyProperties, false),
    E(vkGetPhysicalDeviceQueueFamilyProperties2, false), E(vkGetPhysicalDeviceMemoryProperties, false),
    E(vkGetPhysicalDeviceMemoryProperties2, false), E(vkGetPhysicalDeviceFormatProperties, false),
    E(vkGetPhysicalDeviceFormatProperties2, false), E(vkGetPhysicalDeviceImageFormatProperties, false),
    E(vkGetPhysicalDeviceImageFormatProperties2, false),
    E(vkGetPhysicalDeviceSparseImageFormatProperties, false),
    E(vkEnumerateDeviceExtensionProperties, false), E(vkEnumerateDeviceLayerProperties, false),
    E(vkCreateDevice, false),
    E(vkDestroyDevice, true), E(vkGetDeviceQueue, true), E(vkGetDeviceQueue2, true),
    E(vkDeviceWaitIdle, true), E(vkQueueWaitIdle, true), E(vkAllocateMemory, true), E(vkFreeMemory, true),
    E(vkMapMemory, true), E(vkUnmapMemory, true), E(vkFlushMappedMemoryRanges, true),
    E(vkInvalidateMappedMemoryRanges, true), E(vkGetDeviceMemoryCommitment, true),
    E(vkCreateBuffer, true), E(vkDestroyBuffer, true), E(vkGetBufferMemoryRequirements, true),
    E(vkGetBufferMemoryRequirements2, true), E(vkBindBufferMemory, true), E(vkBindBufferMemory2, true),
    E(vkCreateBufferView, true), E(vkDestroyBufferView, true), E(vkCreateImage, true),
    E(vkDestroyImage, true), E(vkGetImageMemoryRequirements, true), E(vkGetImageMemoryRequirements2, true),
    E(vkGetImageSparseMemoryRequirements, true), E(vkBindImageMemory, true), E(vkBindImageMemory2, true),
    E(vkGetImageSubresourceLayout, true), E(vkCreateImageView, true), E(vkDestroyImageView, true),
    E(vkCreateSampler, true), E(vkDestroySampler, true), E(vkExportMetalObjectsEXT, true),
    E(vkCreateRenderPass, true), E(vkCreateRenderPass2, true), E(vkDestroyRenderPass, true),
    E(vkGetRenderAreaGranularity, true), E(vkCreateFramebuffer, true), E(vkDestroyFramebuffer, true),
    E(vkCreateCommandPool, true), E(vkDestroyCommandPool, true), E(vkResetCommandPool, true),
    E(vkAllocateCommandBuffers, true), E(vkFreeCommandBuffers, true), E(vkBeginCommandBuffer, true),
    E(vkEndCommandBuffer, true), E(vkResetCommandBuffer, true), E(vkCmdBeginRenderPass, true),
    E(vkCmdBeginRenderPass2, true), E(vkCmdEndRenderPass, true), E(vkCmdEndRenderPass2, true),
    E(vkCmdPipelineBarrier, true), E(vkCmdCopyBuffer, true), E(vkCmdFillBuffer, true),
    E(vkCmdClearColorImage, true), E(vkCmdCopyBufferToImage, true), E(vkCreateFence, true), E(vkDestroyFence, true),
    E(vkResetFences, true), E(vkGetFenceStatus, true), E(vkWaitForFences, true),
    E(vkCreateSemaphore, true), E(vkDestroySemaphore, true), E(vkCreateEvent, true),
    E(vkDestroyEvent, true), E(vkGetEventStatus, true), E(vkSetEvent, true), E(vkResetEvent, true),
    E(vkQueueSubmit, true),
};
#undef E

// Promoted-extension aliases: same function under its KHR name.
const std::pair<const char*, const char*> kAliases[] = {
    {"vkGetPhysicalDeviceProperties2KHR", "vkGetPhysicalDeviceProperties2"},
    {"vkGetPhysicalDeviceFeatures2KHR", "vkGetPhysicalDeviceFeatures2"},
    {"vkGetPhysicalDeviceQueueFamilyProperties2KHR", "vkGetPhysicalDeviceQueueFamilyProperties2"},
    {"vkGetPhysicalDeviceMemoryProperties2KHR", "vkGetPhysicalDeviceMemoryProperties2"},
    {"vkGetPhysicalDeviceFormatProperties2KHR", "vkGetPhysicalDeviceFormatProperties2"},
    {"vkGetPhysicalDeviceImageFormatProperties2KHR", "vkGetPhysicalDeviceImageFormatProperties2"},
    {"vkGetBufferMemoryRequirements2KHR", "vkGetBufferMemoryRequirements2"},
    {"vkGetImageMemoryRequirements2KHR", "vkGetImageMemoryRequirements2"},
    {"vkBindBufferMemory2KHR", "vkBindBufferMemory2"},
    {"vkBindImageMemory2KHR", "vkBindImageMemory2"},
    {"vkCreateRenderPass2KHR", "vkCreateRenderPass2"},
    {"vkCmdBeginRenderPass2KHR", "vkCmdBeginRenderPass2"},
    {"vkCmdEndRenderPass2KHR", "vkCmdEndRenderPass2"},
};

}  // namespace

PFN_vkVoidFunction vpvk::lookup(const char* name, bool deviceLevel) {
  if (!name) return nullptr;
  for (const auto& [alias, real] : kAliases)
    if (!std::strcmp(name, alias)) name = real;
  for (const Entry& e : kEntries)
    if (!std::strcmp(e.name, name)) return deviceLevel && !e.device ? nullptr : e.fn;
  if (PFN_vkVoidFunction f = lookupPhase3(name)) return f;
  {
    std::lock_guard<std::mutex> lock(gUnimplMutex);
    if (gUnimpl.insert(name).second) logf("vk.unimpl %s", name);
  }
  return nullptr;
}

VPVK_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice, const char* name) {
  if (name && !std::strcmp(name, "vkGetDeviceProcAddr"))
    return reinterpret_cast<PFN_vkVoidFunction>(&vkGetDeviceProcAddr);
  return lookup(name, true);
}

VPVK_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance, const char* name) {
  if (name && !std::strcmp(name, "vkGetInstanceProcAddr"))
    return reinterpret_cast<PFN_vkVoidFunction>(&vkGetInstanceProcAddr);
  if (name && !std::strcmp(name, "vkGetDeviceProcAddr"))
    return reinterpret_cast<PFN_vkVoidFunction>(&vkGetDeviceProcAddr);
  return lookup(name, false);
}

// Loader-facing ICD interface (only used if a Vulkan loader is ever put in front of the driver).
VPVK_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vk_icdGetInstanceProcAddr(VkInstance i, const char* name) {
  return vkGetInstanceProcAddr(i, name);
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vk_icdNegotiateLoaderICDInterfaceVersion(uint32_t* v) {
  if (*v > 5) *v = 5;
  return VK_SUCCESS;
}
