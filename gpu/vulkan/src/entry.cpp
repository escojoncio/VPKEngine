// VPVulkan entry points: phase 1 (instance, physical device, device, queue, memory, buffers, images,
// views, samplers, VK_EXT_metal_objects), phase 2 (command pools/buffers as deferred lists, render
// passes, framebuffers) and the synchronisation a frame loop needs (fences, semaphores, submit).
// Every function is exported under its Vulkan name so dlsym() on the dylib finds it, as a client
// that loads a driver directly (Klepton's kl_vulkan.c) does for a few vkCmd* entry points.
#include <atomic>
#include <chrono>
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>

#include "commands.h"
#include "vpvk.h"
#include "vpvk/config.h"

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

// The device vpvk_native_submit / vpvk_native_device act on: the last one created (a game makes
// one). Leaked on purpose, like the platform registry: never destroyed under a running guest thread.
std::mutex& liveDeviceMutex() { static auto* m = new std::mutex; return *m; }
Device*& liveDevice() { static Device* d = nullptr; return d; }

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
    // 1D images are 2D textures one texel high (Metal 1D textures take no mips and are no
    // render targets); the SPIR-V translator samples them as 2D (spirv lower.cpp texDim).
    case VK_IMAGE_TYPE_1D: d.type = vpgpu::TextureType::Tex2D; break;
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
  // A blit destination is drawn into by the backend (sampled blits: format conversion, scaling),
  // so uncompressed colour images that can receive transfers are made renderable too.
  {
    const vpgpu::FormatInfo& fi = vpgpu::formatInfo(d.format);
    // 1D images are 2D textures one texel high: blits into them are drawn the same way.
    if ((ci.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) && ci.samples == 1 &&
        d.format != vpgpu::Format::Undefined && d.format != vpgpu::Format::RGB9E5Float && fi.blockSize == 1 &&
        !fi.depth() && !fi.stencil())
      d.usage |= vpgpu::usage::colorTarget;
  }
  // MUTABLE_FORMAT: views of other formats reinterpret the texels (attachments, sampling, storage).
  // Sampled depth/stencil images that are no attachment: their stencil aspect is read through an
  // X32_Stencil8 view (render targets keep Metal's lossless compression: no format views there).
  const vpgpu::FormatInfo& ufi = vpgpu::formatInfo(d.format);
  if ((ci.flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) ||
      (ufi.depth() && ufi.stencil() && (ci.usage & VK_IMAGE_USAGE_SAMPLED_BIT) &&
       !(ci.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)))
    d.usage |= vpgpu::usage::formatView;
  // Multisampled colour images may be resolved by a shader reading their samples (integer
  // formats, which Metal does not resolve: commands.cpp sampleZeroResolve; vkCmdResolveImage
  // regions and sizes Metal's resolve cannot express): they must be readable.
  if (ci.samples > 1 && !vpgpu::formatInfo(d.format).depth() && !vpgpu::formatInfo(d.format).stencil())
    d.usage |= vpgpu::usage::sampled;
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
        // Views are instances (view = instance % count, layer = the mask's bit index; spirv lower.cpp),
        // not Metal vertex amplification: any count up to the layered-rendering limit works.
        m->maxMultiviewViewCount = 8;
        m->maxMultiviewInstanceIndex = (1u << 27) - 1;  // x8 views still fits 32-bit instance ids
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
        static const uint8_t luid[VK_LUID_SIZE] = {'V', 'P', 'V', 'K', 0, 0, 0, 1};
        std::memcpy(id->deviceLUID, luid, VK_LUID_SIZE);
        id->deviceNodeMask = 1;
        id->deviceLUIDValid = VK_TRUE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES: {
        reinterpret_cast<VkPhysicalDeviceTimelineSemaphoreProperties*>(s)->maxTimelineSemaphoreValueDifference = UINT64_MAX;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_PROPERTIES:
        reinterpret_cast<VkPhysicalDeviceProtectedMemoryProperties*>(s)->protectedNoFault = VK_FALSE;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_POINT_CLIPPING_PROPERTIES:
        reinterpret_cast<VkPhysicalDevicePointClippingProperties*>(s)->pointClippingBehavior =
            VK_POINT_CLIPPING_BEHAVIOR_ALL_CLIP_PLANES;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES: {
        // Metal: depth sample 0 / min / max, stencil sample 0; each aspect has its own resolve.
        auto* r = reinterpret_cast<VkPhysicalDeviceDepthStencilResolveProperties*>(s);
        r->supportedDepthResolveModes = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT | VK_RESOLVE_MODE_MIN_BIT | VK_RESOLVE_MODE_MAX_BIT;
        r->supportedStencilResolveModes = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
        r->independentResolveNone = VK_TRUE;
        r->independentResolve = VK_TRUE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTI_DRAW_PROPERTIES_EXT: {
        reinterpret_cast<VkPhysicalDeviceMultiDrawPropertiesEXT*>(s)->maxMultiDrawCount = 2048;
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
  // Indirect draws: n records (GPU-read, pipelines.cpp recordIndirectDraw) with their firstInstance.
  f->multiDrawIndirect = VK_TRUE;
  f->drawIndirectFirstInstance = VK_TRUE;
  f->imageCubeArray = VK_TRUE;
  f->independentBlend = VK_TRUE;
  f->sampleRateShading = VK_TRUE;
  f->dualSrcBlend = VK_TRUE;
  f->depthClamp = VK_TRUE;
  f->depthBiasClamp = VK_TRUE;
  f->fillModeNonSolid = VK_TRUE;
  // Metal rasterizes [[point_size]] up to 511 (pointSizeRange); Adreno reports largePoints too.
  f->largePoints = VK_TRUE;
  f->samplerAnisotropy = VK_TRUE;
  f->textureCompressionBC = VK_TRUE;  // Apple GPUs sample BCn, ETC2/EAC and ASTC LDR natively
  f->textureCompressionETC2 = VK_TRUE;
  f->textureCompressionASTC_LDR = VK_TRUE;  // square blocks only (4x4..12x12); others unsupported
  f->occlusionQueryPrecise = VK_TRUE;
  f->fragmentStoresAndAtomics = VK_TRUE;
  // Metal vertex functions write device memory as fragment ones do; behind the CTS gate until
  // the conformance run has validated the translator's vertex-stage stores (ext.cpp).
  f->vertexPipelineStoresAndAtomics = extraExtensions() ? VK_TRUE : VK_FALSE;
  f->shaderImageGatherExtended = VK_TRUE;
  f->shaderStorageImageExtendedFormats = VK_TRUE;
  f->shaderClipDistance = VK_TRUE;
  f->shaderSampledImageArrayDynamicIndexing = VK_TRUE;
  f->shaderUniformBufferArrayDynamicIndexing = VK_TRUE;
  f->shaderStorageBufferArrayDynamicIndexing = VK_TRUE;
  // Adreno 740 (Quest 3) reports both, and Unity decides per SUBSHADER whether a shader is
  // usable from these bits: a subshader compiled for `#pragma target` 4.6/5.0 (or `require
  // geometry/tessellation`) is dropped when they are false, and a shader left with no usable
  // subshader draws Unity's error colour — 255,0,255. AC Nexus's whole UI (rendered to textures
  // and shown on OVROverlay quads) came out pure magenta while AoT's overlays were fine.
  // VPVK_REPORT_GEOM=1 (set by VisionQuestPro for games, never for the CTS) reports them; a
  // pipeline that really carries a geometry/tessellation stage still fails with
  // `vk.err what=stage` (pipelines.cpp), which makes the case visible in the log.
  {
    const char* g = std::getenv("VPVK_REPORT_GEOM");
    if (g && *g == '1') {
      f->geometryShader = VK_TRUE;
      f->tessellationShader = VK_TRUE;
    }
  }
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
    } else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES) {
      reinterpret_cast<VkPhysicalDeviceTimelineSemaphoreFeatures*>(s)->timelineSemaphore = extraExtensions();
    } else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES) {
      reinterpret_cast<VkPhysicalDeviceSynchronization2Features*>(s)->synchronization2 = extraExtensions();
    } else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES) {
      reinterpret_cast<VkPhysicalDeviceDynamicRenderingFeatures*>(s)->dynamicRendering = extraExtensions();
    } else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES) {
      reinterpret_cast<VkPhysicalDeviceSeparateDepthStencilLayoutsFeatures*>(s)->separateDepthStencilLayouts =
          extraExtensions();
    } else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTI_DRAW_FEATURES_EXT) {
      reinterpret_cast<VkPhysicalDeviceMultiDrawFeaturesEXT*>(s)->multiDraw = extraExtensions();
    } else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES) {
      // Buffer members are addressed by their SPIR-V Offset / ArrayStride / MatrixStride and any
      // access not provably aligned is a packed_* load (MSL emitter), so C-like layouts need nothing.
      reinterpret_cast<VkPhysicalDeviceScalarBlockLayoutFeatures*>(s)->scalarBlockLayout = extraExtensions();
    } else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES) {
      reinterpret_cast<VkPhysicalDeviceUniformBufferStandardLayoutFeatures*>(s)->uniformBufferStandardLayout =
          extraExtensions();
    } else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES) {
      reinterpret_cast<VkPhysicalDeviceProtectedMemoryFeatures*>(s)->protectedMemory = VK_FALSE;
    } else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES) {
      auto* f16 = reinterpret_cast<VkPhysicalDevice16BitStorageFeatures*>(s);
      f16->storageBuffer16BitAccess = f16->uniformAndStorageBuffer16BitAccess = VK_FALSE;
      f16->storagePushConstant16 = f16->storageInputOutput16 = VK_FALSE;
    } else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES) {
      auto* vp = reinterpret_cast<VkPhysicalDeviceVariablePointersFeatures*>(s);
      vp->variablePointersStorageBuffer = vp->variablePointers = VK_FALSE;
    } else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES) {
      // gl_BaseVertex / gl_BaseInstance / gl_DrawID are not lowered (spirv lower.cpp).
      reinterpret_cast<VkPhysicalDeviceShaderDrawParametersFeatures*>(s)->shaderDrawParameters = VK_FALSE;
    } else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES) {
      reinterpret_cast<VkPhysicalDeviceSamplerYcbcrConversionFeatures*>(s)->samplerYcbcrConversion = VK_FALSE;
    } else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_FEATURES_EXT) {
      auto* d = reinterpret_cast<VkPhysicalDeviceFragmentDensityMapFeaturesEXT*>(s);
      d->fragmentDensityMap = VK_FALSE;  // not advertised: foveation is VPKEngine's (vpvk/config.h)
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
    VkImageCreateFlags flags, VkImageFormatProperties* out) {
  *out = {};  // every field 0 when the combination is refused (spec; the CTS checks it)
  const VkFormatFeatureFlags ff = formatFeatures(f, false);
  if (!ff) return VK_ERROR_FORMAT_NOT_SUPPORTED;
  if ((usage & VK_IMAGE_USAGE_STORAGE_BIT) && !(ff & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
  if ((usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) && !(ff & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT))
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
  if ((usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) &&
      !(ff & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
  if ((usage & (VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT)) &&
      !(ff & (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)))
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
  if (tiling == VK_IMAGE_TILING_LINEAR &&
      (usage & ~(VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)))
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
  // Metal's limits (see metalInvalidTexture in the backend): no compressed or depth 1D textures
  // (1D images are 2D textures one texel high there, so they do take mips), no depth 3D textures.
  {
    const vpgpu::FormatInfo& fi = vpgpu::formatInfo(toCoreFormat(f));
    const bool depthStencil = fi.depth() || fi.stencil();
    if (type == VK_IMAGE_TYPE_1D && (fi.blockSize > 1 || depthStencil)) return VK_ERROR_FORMAT_NOT_SUPPORTED;
    if (type == VK_IMAGE_TYPE_3D && depthStencil) return VK_ERROR_FORMAT_NOT_SUPPORTED;
    // Uncompressed views of compressed texels (BLOCK_TEXEL_VIEW_COMPATIBLE): Metal makes no such
    // views (newTextureViewWithPixelFormat aborts; the backend refuses them), so reading the block
    // bits would return garbage. Not offered.
    if ((flags & VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT) && fi.blockSize > 1) return VK_ERROR_FORMAT_NOT_SUPPORTED;
    if (type == VK_IMAGE_TYPE_3D && fi.name &&
        (!strncmp(fi.name, "Etc2", 4) || !strncmp(fi.name, "Eac", 3) || !strncmp(fi.name, "Astc", 4)))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
  }
  const auto& l = deviceProperties().limits;
  *out = {};
  out->maxExtent = {l.maxImageDimension2D, type == VK_IMAGE_TYPE_1D ? 1u : l.maxImageDimension2D,
                    type == VK_IMAGE_TYPE_3D ? l.maxImageDimension3D : 1u};
  out->maxMipLevels = 15;
  out->maxArrayLayers = type == VK_IMAGE_TYPE_3D ? 1 : l.maxImageArrayLayers;
  // Multisample only where Vulkan allows it: optimal 2D, not cube-compatible, an attachment format
  // (Metal renders MSAA into it), not storage (storageImageSampleCounts = 1).
  const bool msaa = tiling == VK_IMAGE_TILING_OPTIMAL && type == VK_IMAGE_TYPE_2D &&
                    !(flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) && !(usage & VK_IMAGE_USAGE_STORAGE_BIT) &&
                    (ff & (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT));
  out->sampleCounts = msaa ? l.framebufferColorSampleCounts : VkSampleCountFlags{VK_SAMPLE_COUNT_1_BIT};
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
namespace {
// Is every feature `ci` enables one this driver reports (VK_ERROR_FEATURE_NOT_PRESENT otherwise)?
// Only for conformance runs (extraExtensions): games built for the Quest sometimes enable features
// they never use without asking, and refusing their device would stop them.
bool featuresSupported(VkPhysicalDevice pd, const VkDeviceCreateInfo* ci) {
  auto subset = [](const void* want, const void* have, size_t bytes) {
    const auto* w = static_cast<const VkBool32*>(want);
    const auto* h = static_cast<const VkBool32*>(have);
    for (size_t i = 0; i < bytes / sizeof(VkBool32); ++i)
      if (w[i] && !h[i]) return false;
    return true;
  };
  VkPhysicalDeviceFeatures base{};
  vkGetPhysicalDeviceFeatures(pd, &base);
  if (ci->pEnabledFeatures && !subset(ci->pEnabledFeatures, &base, sizeof base)) return false;
  // (sType, VkBool32 members after the header): only those are compared, never the padding.
  static const std::pair<VkStructureType, size_t> kStructs[] = {
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES, 3},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES, 1},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES, 1},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES, 1},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES, 1},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTI_DRAW_FEATURES_EXT, 1},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES, 1},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, 4},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES, 2},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES, 1},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES, 1},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES, 1},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES, 1},
  };
  const size_t head = sizeof(VkBaseOutStructure);
  for (auto* s = static_cast<const VkBaseInStructure*>(ci->pNext); s; s = s->pNext) {
    if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) {
      if (!subset(&reinterpret_cast<const VkPhysicalDeviceFeatures2*>(s)->features, &base, sizeof base)) return false;
      continue;
    }
    for (const auto& [type, bools] : kStructs) {
      if (s->sType != type) continue;
      alignas(8) uint8_t have[256] = {};
      reinterpret_cast<VkBaseOutStructure*>(have)->sType = type;
      VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
      f2.pNext = have;
      vkGetPhysicalDeviceFeatures2(pd, &f2);
      if (!subset(reinterpret_cast<const uint8_t*>(s) + head, have + head, bools * sizeof(VkBool32))) return false;
    }
  }
  return true;
}
}  // namespace

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
  if (extraExtensions() && !featuresSupported(pd, ci)) {
    logf("vk.err what=device_feature_not_present");
    delete d;
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  d->platform = createPlatform();
  // Pipeline compiles run on background workers (never the submit thread); leave the game's
  // render/main threads two cores.
  const unsigned workers = std::clamp(std::thread::hardware_concurrency(), 4u, 8u) - 2u;
  d->gpu = std::make_unique<vpgpu::Device>(d->platform->backend(), workers, [](const std::string& s) { log(s); });
  d->queue.device = d;
  // Pipeline recipes of earlier runs: compiled now, before the game's first frame, so its
  // vkCreate*Pipelines calls are cache hits. With vpvk_prewarm() run by the app beforehand this is
  // mostly binary-archive lookups.
  d->recipes = std::make_unique<RecipeStore>(RecipeStore::defaultDir());
  if (d->recipes->enabled() && prewarmEnabled()) prewarm(*d->platform, d->gpu->pipelines(), *d->recipes, nullptr, 20.0);
  logf("vk.device exts=%u multiview=%d fdm=%d metal_objects=%d", ci->enabledExtensionCount,
       d->multiview, d->densityMap, d->metalObjects);
  logConfig();
  {
    std::lock_guard<std::mutex> l(liveDeviceMutex());
    liveDevice() = d;
  }
  *out = to<VkDevice>(d);
  holdForceSingleSample(true);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyDevice(VkDevice dv, const VkAllocationCallbacks*) {
  if (!dv) return;
  holdForceSingleSample(false);
  Device* d = from<Device>(dv);
  {
    std::lock_guard<std::mutex> l(liveDeviceMutex());
    if (liveDevice() == d) liveDevice() = nullptr;
  }
  // A vpvk_native_submit that found the device before it left the registry holds this mutex
  // until its command buffer is committed.
  { std::lock_guard<std::mutex> l(d->submitMutex); }
  d->platform->waitIdle(UINT64_MAX);  // bounded by a lost device (Platform::waitIdle)
  releaseVisibility(d);
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

// Waits end in VK_ERROR_DEVICE_LOST instead of blocking forever once the GPU hung or rejected work.
static VkResult waitResult(Platform::WaitResult r) {
  return r == Platform::WaitResult::Done ? VK_SUCCESS
         : r == Platform::WaitResult::Timeout ? VK_TIMEOUT
                                              : VK_ERROR_DEVICE_LOST;
}

// Batches held back on a timeline wait (Device::deferred) are queue work too: idle means they ran.
// Another thread's vkSignalSemaphore releases them; polled in 1 ms steps.
static VkResult waitDeviceIdle(Device* d) {
  // Batches parked on a timeline value nobody will signal would hang this forever
  // (an application bug, but a hang is worse than a log line): bounded at 2 s.
  const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  for (;;) {
    {
      std::lock_guard<std::mutex> lock(d->submitMutex);
      if (d->deferred.empty()) break;
    }
    if (d->platform->lost()) return VK_ERROR_DEVICE_LOST;
    if (std::chrono::steady_clock::now() > until) {
      logf("vk.err what=wait_idle_deferred reason=timeline_wait_never_signalled");
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  const VkResult r = waitResult(d->platform->waitIdle(UINT64_MAX));
  return r == VK_TIMEOUT ? VK_ERROR_DEVICE_LOST : r;  // no timeout was asked for
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkDeviceWaitIdle(VkDevice dv) { return waitDeviceIdle(from<Device>(dv)); }

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkQueueWaitIdle(VkQueue q) { return waitDeviceIdle(from<Queue>(q)->device); }

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

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice dv, VkDeviceMemory mem,
                                                    const VkAllocationCallbacks*) {
  if (!mem) return;
  DeviceMemory* m = from<DeviceMemory>(mem);
  if (!m->images.empty() && dv) {  // images still bound (legal: they must not be used again) forget it
    std::lock_guard<std::mutex> lock(from<Device>(dv)->submitMutex);
    for (Image* i : m->images) i->memory = nullptr;
  }
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
  if (b->id) {
    Device* d = from<Device>(dv);
    std::lock_guard<std::mutex> lock(d->submitMutex);  // see vkDestroyImage
    d->gpu->destroy(b->id);
  }
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
  Device* dev = from<Device>(dv);
  std::lock_guard<std::mutex> lock(dev->submitMutex);  // see vkBindImageMemory
  if (b->id) dev->gpu->destroy(b->id);  // rebinding (invalid use): do not leak the previous one
  b->id = dev->gpu->createBuffer(std::move(d));
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
  // MSAA: the game's count, then forcing to 1x / the override / the device clamp (effectiveSamples;
  // a count of 1 never changes). A multisampled image that ends up 1x takes the forced path
  // (resolves become copies, see forcedResolveCopy).
  const uint8_t gameSamples = desc.samples ? desc.samples : 1;
  uint8_t forcedFrom = 0;
  if (gameSamples > 1) {
    const uint8_t eff = effectiveSamples(gameSamples);
    if (eff <= 1) {
      forcedFrom = gameSamples;
      desc.samples = 1;
      static std::atomic<unsigned> k{0};
      if (k.fetch_add(1) < 16)
        logf("vk.msaa forced 1x (was %ux) image=%ux%u format=%d", forcedFrom, ci->extent.width, ci->extent.height,
             ci->format);
    } else if (eff != gameSamples) {
      desc.samples = eff;
      static std::atomic<unsigned> k{0};
      if (k.fetch_add(1) < 16)
        logf("vk.msaa image %ux->%ux %ux%u format=%d", unsigned(gameSamples), unsigned(eff), ci->extent.width,
             ci->extent.height, ci->format);
    }
  }
  if (desc.format == vpgpu::Format::Undefined || !vpgpu::validate(desc)) {
    logf("vk.err what=image_desc format=%d %ux%ux%u mips=%u layers=%u", ci->format, ci->extent.width,
         ci->extent.height, ci->extent.depth, ci->mipLevels, ci->arrayLayers);
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
  }
  auto* img = new Image;
  img->info = *ci;
  img->info.pNext = nullptr;
  img->forcedFrom = forcedFrom;
  img->gameSamples = gameSamples;
  img->info.pQueueFamilyIndices = nullptr;
  img->desc = std::move(desc);
  // Linear colour images are host-visible staging (the CTS reads every result image through one):
  // their memory must hold the texels (see Image::linear). Depth/stencil linear images are refused
  // by the format features, multisampled ones by the image format properties.
  {
    const vpgpu::FormatInfo& fi = vpgpu::formatInfo(img->desc.format);
    img->linear = ci->tiling == VK_IMAGE_TILING_LINEAR && !fi.depth() && !fi.stencil() && img->desc.samples == 1;
  }
  *out = to<VkImage>(img);
  return VK_SUCCESS;
}

namespace {
// Images that may share one texture (aliasing the same memory): everything Metal fixes at creation.
bool sameTexture(const vpgpu::TextureDesc& a, const vpgpu::TextureDesc& b) {
  return a.type == b.type && a.format == b.format && a.width == b.width && a.height == b.height &&
         a.depth == b.depth && a.mips == b.mips && a.layers == b.layers && a.samples == b.samples &&
         a.usage == b.usage && a.persistent == b.persistent && a.transient == b.transient;
}
// Unbinds `i` from its memory and drops its texture; the last image sharing it destroys it
// (submitMutex held).
void releaseImageTexture(Device& d, Image& i) {
  if (i.memory) {
    auto& v = i.memory->images;
    v.erase(std::remove(v.begin(), v.end(), &i), v.end());
  }
  if (i.id && (!i.idOwners || i.idOwners.use_count() == 1)) d.gpu->destroy(i.id);
  if (i.hostAlias) d.gpu->destroy(i.hostAlias);
  i.id = {};
  i.hostAlias = {};
  i.native = nullptr;
  i.idOwners.reset();
}
}  // namespace

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyImage(VkDevice dv, VkImage im,
                                                      const VkAllocationCallbacks*) {
  if (!im) return;
  Image* i = from<Image>(im);
  {
    // The backend's texture map is read by vkQueueSubmit on another thread and its retire list is
    // moved by the next frame slot: destroying outside the submit mutex could lose the retire (the
    // MTLTexture released while in flight) or corrupt the map. Released when its slot is reused.
    Device* d = from<Device>(dv);
    std::lock_guard<std::mutex> lock(d->submitMutex);
    releaseImageTexture(*d, *i);
  }
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
  Device* d = from<Device>(dv);
  // Same lock as vkQueueSubmit: creating textures inserts into maps the submit thread walks.
  std::lock_guard<std::mutex> lock(d->submitMutex);
  // Rebinding (invalid use, seen with wild handles): release the previous texture.
  releaseImageTexture(*d, *i);
  i->memory = from<DeviceMemory>(mem);
  i->offset = off;
  // Metal owns texture storage, so memory aliasing is done by sharing: an image bound where a live
  // image of the same description already is (VK_IMAGE_CREATE_ALIAS_BIT, transient targets placed
  // over each other by an allocator) uses its texture, and writes through one are seen by the other.
  if (i->memory)
    for (Image* o : i->memory->images)
      if (o != i && o->offset == off && o->id && sameTexture(o->desc, i->desc)) {
        i->id = o->id, i->native = o->native, i->idOwners = o->idOwners;
        break;
      }
  if (!i->id) {
    i->id = d->gpu->createTexture(i->desc);
    if (!i->id) { i->memory = nullptr; return VK_ERROR_OUT_OF_DEVICE_MEMORY; }
    i->idOwners = std::make_shared<int>(0);
    // Persistent images are allocated now, so the lookup only reads; render targets that may stay
    // on tile memory are not touched (looking them up would allocate them).
    if (i->desc.persistent) i->native = d->platform->nativeTexture(i->id);
  }
  if (i->memory) i->memory->images.push_back(i);
  // Linear image: a buffer over its texels in the VkDeviceMemory (linearLayout), which the
  // translator fills from the texture after GPU writes and reads into it before GPU reads.
  if (i->linear && i->memory && i->memory->host && off < i->memory->size) {
    vpgpu::BufferDesc bd;
    bd.size = std::min<uint64_t>(imageBytes(i->desc), i->memory->size - off);
    bd.guestMemory = i->memory->host + off;
    bd.label = "vk.linear_image";
    i->hostAlias = d->gpu->createBuffer(std::move(bd));
    if (!i->hostAlias) logf("vk.err what=linear_image_alias size=%ux%u", i->desc.width, i->desc.height);
  }
  return VK_SUCCESS;
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
  // Only meaningful for linear images (the layout their memory really has, see linearLayout).
  const vpgpu::TextureDesc& d = from<Image>(im)->desc;
  if (sub->mipLevel >= d.mips || sub->arrayLayer >= d.layers) {
    *out = {};
    return;
  }
  const LinearLayout l = linearLayout(d, sub->mipLevel, sub->arrayLayer);
  out->offset = l.offset;
  out->size = l.size;
  out->rowPitch = l.rowPitch;
  out->arrayPitch = l.arrayPitch;
  out->depthPitch = l.depthPitch;
}

LinearLayout vpvk::linearLayout(const vpgpu::TextureDesc& d, uint32_t mip, uint32_t layer) {
  const vpgpu::FormatInfo& fi = vpgpu::formatInfo(d.format);
  LinearLayout l;
  uint64_t off = 0;
  for (uint32_t m = 0; m <= mip && m < d.mips; ++m) {
    const uint64_t w = std::max<uint32_t>(1, d.width >> m), h = std::max<uint32_t>(1, d.height >> m);
    const uint64_t z = d.type == vpgpu::TextureType::Tex3D ? std::max<uint32_t>(1, d.depth >> m) : 1;
    const uint64_t row = (w + fi.blockSize - 1) / fi.blockSize * fi.bytesPerBlock;
    const uint64_t rows = (h + fi.blockSize - 1) / fi.blockSize;
    const uint64_t image = row * rows * z;  // one layer of this mip (all its depth slices)
    if (m == mip) {
      l.offset = off + image * layer;
      l.size = image;
      l.rowPitch = row;
      l.depthPitch = row * rows;
      l.arrayPitch = image;
      l.rows = uint32_t(rows);
      return l;
    }
    off += image * d.layers;
  }
  return l;
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
      case VK_IMAGE_VIEW_TYPE_1D_ARRAY: info.type = vpgpu::TextureType::Tex2D, info.array = true; break;
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
    // Depth-aspect views keep the combined format; a stencil-aspect view samples the stencil
    // (S8Uint: the backend's stencil view of the combined texture).
    const bool stencilView = depthStencil && v->range.aspectMask == VK_IMAGE_ASPECT_STENCIL_BIT;
    if (depthStencil) info.format = stencilView ? vpgpu::Format::S8Uint : d.format;
    // VkComponentMapping -> MTLTextureSwizzle (zero 0, one 1, r..a 2..5). Not for depth/stencil.
    bool swizzled = false;
    const vpgpu::FormatInfo& vfi = vpgpu::formatInfo(info.format == vpgpu::Format::Undefined ? d.format : info.format);
    if (!vfi.depth() && !vfi.stencil()) {
      const VkComponentSwizzle cs[4] = {ci->components.r, ci->components.g, ci->components.b, ci->components.a};
      for (int c = 0; c < 4; ++c) {
        uint8_t s = uint8_t(2 + c);
        switch (cs[c]) {
          case VK_COMPONENT_SWIZZLE_ZERO: s = 0; break;
          case VK_COMPONENT_SWIZZLE_ONE: s = 1; break;
          case VK_COMPONENT_SWIZZLE_R: s = 2; break;
          case VK_COMPONENT_SWIZZLE_G: s = 3; break;
          case VK_COMPONENT_SWIZZLE_B: s = 4; break;
          case VK_COMPONENT_SWIZZLE_A: s = 5; break;
          default: break;  // IDENTITY
        }
        info.swizzle[c] = s;
        if (s != 2 + c) swizzled = true;
      }
      if (swizzled) {
        static std::mutex sm;
        static std::set<uint64_t> seen;
        const uint64_t key = uint64_t(ci->format) << 32 | uint32_t(info.swizzle[0]) | uint32_t(info.swizzle[1]) << 8 |
                             uint32_t(info.swizzle[2]) << 16 | uint32_t(info.swizzle[3]) << 24;
        std::lock_guard<std::mutex> lock(sm);
        if (seen.insert(key).second) {
          static const char* nm = "01rgba";
          logf("gpu.diag swizzle fmt=%d map=%c,%c,%c,%c", int(ci->format), nm[info.swizzle[0]], nm[info.swizzle[1]],
               nm[info.swizzle[2]], nm[info.swizzle[3]]);
        }
      }
    }
    // Swizzled views are only bound for sampling; attachments use the image itself.
    if (!sameType || (!sameFormat && !depthStencil) || !whole || swizzled || stencilView) {
      Device* d = from<Device>(dv);
      std::lock_guard<std::mutex> lock(d->submitMutex);  // the backend's view map (see vkBindImageMemory)
      v->view = d->platform->createTextureView(im->id, info);
    }
  }
  *out = to<VkImageView>(v);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyImageView(VkDevice dv, VkImageView v,
                                                          const VkAllocationCallbacks*) {
  if (!v) return;
  ImageView* iv = from<ImageView>(v);
  if (iv->view) {
    Device* d = from<Device>(dv);
    std::lock_guard<std::mutex> lock(d->submitMutex);
    d->platform->destroyTextureView(iv->view);
  }
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
  // App setting "Filtrado anisotrópico" (level 2/4/8/16): only material-style samplers — linear
  // min/mag/mip (trilinear) with a mip range to walk and a wrapping address mode on U or V.
  // Excluded, keeping the game's value: nearest, comparison (shadow), unnormalized, and all-clamp
  // samplers (post-processing, render-target reads, cubemaps, 3D LUTs use clamp-to-edge/border),
  // where anisotropic taps cost bandwidth on full-screen passes and give nothing.
  {
    const auto wraps = [](VkSamplerAddressMode m) {
      return m == VK_SAMPLER_ADDRESS_MODE_REPEAT || m == VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    };
    const int forced = vpvk::anisotropyForced();
    const bool eligible = ci->minFilter == VK_FILTER_LINEAR && ci->magFilter == VK_FILTER_LINEAR &&
                          ci->mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR && !ci->compareEnable &&
                          !ci->unnormalizedCoordinates && ci->maxLod > ci->minLod + 0.25f &&
                          (wraps(ci->addressModeU) || wraps(ci->addressModeV));
    if (forced > 1 && eligible) {
      const auto f = static_cast<uint8_t>(std::min(forced, 16));
      if (f > d.maxAnisotropy) d.maxAnisotropy = f;
    }
    static std::atomic<unsigned> logged{0}, total{0}, forcedN{0};
    const unsigned n = total.fetch_add(1) + 1;
    const unsigned fN = forcedN.fetch_add(forced > 1 && eligible ? 1 : 0) + (forced > 1 && eligible ? 1 : 0);
    if (logged.fetch_add(1) < 4 || (n & (n - 1)) == 0)
      vpvk::logf("vk.sampler aniso level=%d forced=%u/%u last: min=%d mag=%d mip=%d lod=%.1f..%.1f addr=%d,%d,%d "
                 "cmp=%d game_aniso=%d -> %u",
                 forced, fN, n, ci->minFilter, ci->magFilter, ci->mipmapMode, ci->minLod, ci->maxLod,
                 ci->addressModeU, ci->addressModeV, ci->addressModeW, ci->compareEnable,
                 ci->anisotropyEnable ? int(ci->maxAnisotropy) : 0, unsigned(d.maxAnisotropy));
  }
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
  d.lodBias = std::clamp(ci->mipLodBias, -16.0f, 15.99f);  // maxSamplerLodBias; Metal S4.6
  // Texel coordinates: Vulkan's restrictions (one filter, no mips, clamp modes, lod 0) match
  // Metal's for normalizedCoordinates = NO.
  d.normalized = !ci->unnormalizedCoordinates;
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
  {
    Device* d = from<Device>(dv);
    std::lock_guard<std::mutex> lock(d->submitMutex);  // see vkDestroyImage
    d->gpu->destroy(s->id);
  }
  delete s;
}

// The id<MTLTexture> behind an image (unretained): cached at bind for persistent images; otherwise
// looked up (allocating a lazy render target) under the submit mutex, since the backend's maps
// belong to the submit thread.
namespace {
// Set while a vpvk_native_submit callback runs on this thread (it holds the submit mutex): a
// lookup from inside it must not take the mutex again.
thread_local bool tInNativeSubmit = false;
void* imageNative(Device* d, const Image* img) {
  if (!d || !img || !img->id) return nullptr;
  if (img->native) return img->native;
  if (tInNativeSubmit) return d->platform->nativeTexture(img->id);
  std::lock_guard<std::mutex> lock(d->submitMutex);
  return d->platform->nativeTexture(img->id);
}
}  // namespace

// MoltenVK's vkGetMTLTextureMVK (deprecated there, a plain dylib export): Klepton reaches the
// overlay and shadow textures with it (klvk_overlay_mtl_texture), which are never marked exportable.
// The image carries no device pointer, so the live device answers (a game makes one).
VPVK_EXPORT void vkGetMTLTextureMVK(VkImage image, void** out) {
  if (!out) return;
  *out = nullptr;
  if (!image) return;
  Device* d;
  {
    std::lock_guard<std::mutex> l(liveDeviceMutex());
    d = liveDevice();
  }
  *out = imageNative(d, from<Image>(image));
  static std::atomic<unsigned> ok{0}, miss{0};
  if (*out ? ok.fetch_add(1) < 1 : miss.fetch_add(1) < 8)
    logf("vk.mvk get_mtl_texture image=%p texture=%p%s", static_cast<void*>(image), *out,
         *out ? "" : (d ? " (no texture: not bound or none on this platform)" : " (no device)"));
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
        t->mtlTexture = static_cast<MTLTexture_id>(imageNative(d, img));
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
    if (const auto* r = static_cast<const VkSubpassDescriptionDepthStencilResolve*>(
            findNext(sd.pNext, VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE));
        r && r->pDepthStencilResolveAttachment) {
      sp.depthResolve = r->pDepthStencilResolveAttachment->attachment;
      sp.depthResolveMode = r->depthResolveMode;
      sp.stencilResolveMode = r->stencilResolveMode;
    }
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
                                                                const VkCommandBufferBeginInfo* bi) {
  CommandBuffer* c = from<CommandBuffer>(cb);
  c->rec->reset();  // implicit reset (pools are always resettable here)
  c->state = CommandBuffer::State::Recording;
  // A secondary recorded inside a render pass: its draws use pipelines with the pass's sample
  // count (MSAA override), known from the inheritance info when the framebuffer is given.
  if (c->level == VK_COMMAND_BUFFER_LEVEL_SECONDARY && bi && bi->pInheritanceInfo &&
      (bi->flags & VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT)) {
    const VkCommandBufferInheritanceInfo& in = *bi->pInheritanceInfo;
    BoundState& b = c->rec->bound;
    b.pass = in.renderPass ? from<RenderPassObj>(in.renderPass) : nullptr;
    b.framebuffer = in.framebuffer ? from<Framebuffer>(in.framebuffer) : nullptr;
    b.subpass = in.subpass;
    b.passSamples = b.pass && b.framebuffer ? subpassSamples(*b.pass, *b.framebuffer, in.subpass, false) : 0;
  }
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
  r.bound.framebuffer = op.framebuffer;
  r.bound.subpass = 0;
  r.bound.passSamples = op.pass && op.framebuffer ? subpassSamples(*op.pass, *op.framebuffer, 0) : 0;
  r.bound.targetWidth = uint32_t(op.area.offset.x) + op.area.extent.width;
  r.bound.targetHeight = uint32_t(op.area.offset.y) + op.area.extent.height;
  r.passBegin = r.ops.size();
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
  r.passBegin = SIZE_MAX;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdEndRenderPass2(VkCommandBuffer cb, const VkSubpassEndInfo*) {
  vkCmdEndRenderPass(cb);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier(
    VkCommandBuffer cb, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags, uint32_t,
    const VkMemoryBarrier*, uint32_t, const VkBufferMemoryBarrier*, uint32_t, const VkImageMemoryBarrier*) {
  Recorder& r = *from<CommandBuffer>(cb)->rec;
  ++r.barriers;  // the core places barriers from real hazards
  // Inside a render pass a barrier is a subpass self-dependency: kept as a marker, the translator
  // splits a subpass that reads its own attachments there (commands.cpp).
  if (r.inPass) r.ops.emplace_back(OpPassBarrier{});
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

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyFence(VkDevice dv, VkFence f, const VkAllocationCallbacks*) {
  Fence* fe = from<Fence>(f);
  if (!fe) return;
  // A batch still parked on a timeline wait must not signal a freed fence later.
  if (Device* d = from<Device>(dv)) {
    std::lock_guard<std::mutex> lock(d->submitMutex);
    for (DeferredBatch& b : d->deferred)
      if (b.fence == fe) b.fence = nullptr;
  }
  delete fe;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetFences(VkDevice, uint32_t n, const VkFence* fs) {
  for (uint32_t i = 0; i < n; ++i) from<Fence>(fs[i])->signaled = false;
  return VK_SUCCESS;
}

// A fence is signaled when its submission's GPU work is done: every vpgpu frame up to the one its
// vkQueueSubmit made (Fence::frames) has completed on the GPU. The client reuses that submission's
// command buffers (upload memory, descriptors) as soon as this says done.
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetFenceStatus(VkDevice dv, VkFence f) {
  Fence* fe = from<Fence>(f);
  Platform& p = *from<Device>(dv)->platform;
  if (p.lost()) return VK_ERROR_DEVICE_LOST;
  if (!fe->signaled || fe->deferred) return VK_NOT_READY;
  if (fe->submitted) {
    const VkResult r = waitResult(p.waitFrames(fe->frames, 0));
    if (r != VK_SUCCESS) return r == VK_TIMEOUT ? VK_NOT_READY : r;
    fe->submitted = false;
  }
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(VkDevice dv, uint32_t n, const VkFence* fs,
                                                           VkBool32 waitAll, uint64_t timeout) {
  Platform& p = *from<Device>(dv)->platform;
  if (p.lost()) return VK_ERROR_DEVICE_LOST;
  // A fence whose batch is held back on a timeline wait (Fence::deferred) gets its frame once
  // another thread's vkSignalSemaphore runs it: poll in 1 ms steps until then (or the timeout).
  {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    for (;;) {
      bool deferred = false, readyOne = false;
      for (uint32_t i = 0; i < n; ++i) {
        const Fence* fe = from<Fence>(fs[i]);
        if (fe->signaled && fe->deferred) deferred = true;
        else if (fe->signaled) readyOne = true;
      }
      if (!deferred || (!waitAll && readyOne)) break;
      const uint64_t elapsed =
          uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
      if (elapsed >= timeout) return VK_TIMEOUT;
      if (p.lost()) return VK_ERROR_DEVICE_LOST;
      std::this_thread::sleep_for(std::chrono::nanoseconds(std::min<uint64_t>(timeout - elapsed, 1'000'000)));
    }
    const uint64_t elapsed =
        uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
    timeout = elapsed >= timeout ? 0 : timeout - elapsed;
  }
  // A fence that is not signaled was never submitted (or was reset): nothing here will signal it.
  // waitAll: wait for the latest of the submitted ones; any: the earliest, or done at once if one
  // is already signaled with no GPU work pending.
  uint64_t target = waitAll ? 0 : UINT64_MAX;
  bool pending = false;
  for (uint32_t i = 0; i < n; ++i) {
    Fence* fe = from<Fence>(fs[i]);
    if (!fe->signaled || fe->deferred) {  // deferred: only with !waitAll and another fence ready
      if (waitAll) return VK_TIMEOUT;
      continue;
    }
    if (!fe->submitted) {
      if (!waitAll) return VK_SUCCESS;
      continue;
    }
    pending = true;
    target = waitAll ? std::max<uint64_t>(target, fe->frames) : std::min<uint64_t>(target, fe->frames);
  }
  if (!pending) return waitAll ? VK_SUCCESS : VK_TIMEOUT;  // any: none signaled
  const VkResult r = waitResult(p.waitFrames(target, timeout));
  if (r != VK_SUCCESS) return r;
  for (uint32_t i = 0; i < n; ++i) {
    Fence* fe = from<Fence>(fs[i]);
    if (fe->signaled && !fe->deferred && fe->submitted && fe->frames <= target) fe->submitted = false;
  }
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateSemaphore(VkDevice, const VkSemaphoreCreateInfo* ci,
                                                             const VkAllocationCallbacks*, VkSemaphore* out) {
  auto* s = new Semaphore;
  if (const auto* t = static_cast<const VkSemaphoreTypeCreateInfo*>(
          ci ? findNext(ci->pNext, VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO) : nullptr);
      t && t->semaphoreType == VK_SEMAPHORE_TYPE_TIMELINE) {
    s->timeline = true;
    s->completed = t->initialValue;
  }
  *out = to<VkSemaphore>(s);
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
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetEventStatus(VkDevice dv, VkEvent e) {
  Event* ev = from<Event>(e);
  std::lock_guard<std::mutex> l(ev->m);
  if (ev->frame) {
    const auto w = from<Device>(dv)->platform->waitFrames(ev->frame - 1, 0);
    if (w == Platform::WaitResult::Lost) return VK_ERROR_DEVICE_LOST;
    if (w == Platform::WaitResult::Done) ev->set = ev->value, ev->frame = 0;
  }
  return ev->set ? VK_EVENT_SET : VK_EVENT_RESET;
}
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkSetEvent(VkDevice, VkEvent e) {
  Event* ev = from<Event>(e);
  std::lock_guard<std::mutex> l(ev->m);
  ev->set = true, ev->frame = 0;
  return VK_SUCCESS;
}
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetEvent(VkDevice, VkEvent e) {
  Event* ev = from<Event>(e);
  std::lock_guard<std::mutex> l(ev->m);
  ev->set = false, ev->frame = 0;
  return VK_SUCCESS;
}

// One queue, executed in order: semaphores between submissions of this queue are already satisfied
// by submission order, so they need no GPU work. Consecutive batches that can run go into one core
// frame, which lets the frame compiler merge passes across command-buffer boundaries.
//
// Timeline waits (VkTimelineSemaphoreSubmitInfo) are the exception: a batch may wait on a value
// that only the host will signal later (wait-before-signal). Such a batch and every batch after it
// are held in Device::deferred and run from vkSignalSemaphore once their waits can be met; running
// them now would let the GPU read what the host has not written yet.
namespace vpvk {
namespace {
// A timeline wait is met when the value is already reached, or reached by work that is on the GPU
// (pending) or ahead of it in the same frame (`ahead`): the queue runs in order.
bool waitMet(Semaphore* s, uint64_t value, const std::vector<std::pair<Semaphore*, uint64_t>>& ahead) {
  for (const auto& [as, av] : ahead)
    if (as == s && av >= value) return true;
  std::lock_guard<std::mutex> l(s->m);
  if (s->completed >= value) return true;
  for (const auto& [v, f] : s->pending)
    if (v >= value) return true;
  return false;
}

// Runs batches [0, count) of `list` as one frame (submitMutex held), then stamps their timeline
// signals, queries, events and fences with its frame.
void runBatches(Device* d, std::vector<DeferredBatch>& list, size_t count) {
  vpgpu::Frame frame;
  TranslateStats st;
  uint32_t barriers = 0, cbs = 0;
  // One VPKTAA jitter for the whole submission (read once: a concurrent set applies to the next).
  const TaaJitter taa = taaJitter();
  for (size_t i = 0; i < count; ++i)
    for (const CommandBuffer* c : list[i].cbs) {
      translate(*c->rec, frame, st, taa.enabled ? &taa : nullptr);
      barriers += c->rec->barriers;
      ++cbs;
    }
  if (!frame.commands().empty()) d->gpu->submit(frame);
  ++d->submits;
  msaaSubmitTick(d->submits);
  if (taa.enabled) taaRecordSubmit(d->submits, taa, st.taaJittered, st.taaSkipped, st.taaTargetW, st.taaTargetH);
  if (st.dropped || st.skippedDraws || st.skippedDispatches || d->submits <= 3 || d->submits % 600 == 0)
    logf("vk.submit n=%llu cbs=%u passes=%u draws=%u dispatches=%u copies=%u clears=%u vk_barriers=%u "
         "draws_skipped=%u dispatch_skipped=%u dropped=%u",
         static_cast<unsigned long long>(d->submits), cbs, st.passes, st.draws, st.dispatches, st.copies, st.clears,
         barriers, st.skippedDraws, st.skippedDispatches, st.dropped);
  const uint64_t frames = d->gpu->submittedFrames();
  for (size_t i = 0; i < count; ++i)
    for (const auto& [s, v] : list[i].signals) {  // reached once this frame finishes
      std::lock_guard<std::mutex> l(s->m);
      s->pending.push_back({v, frames});
    }
  if (!st.queries.empty()) applyQueryEvents(st.queries, frames);
  for (const auto& [ev, value] : st.events) {  // vkCmdSetEvent / vkCmdResetEvent, in order
    std::lock_guard<std::mutex> l(ev->m);
    ev->frame = frames + 1;  // +1: 0 means nothing pending (see Event)
    ev->value = value;
  }
  for (size_t i = 0; i < count; ++i)
    if (Fence* fe = list[i].fence) {  // done when this frame (and every earlier one) has finished
      fe->frames = frames;
      fe->submitted = true;
      fe->signaled = true;
      fe->deferred = false;
    }
  list.erase(list.begin(), list.begin() + std::ptrdiff_t(count));
}

// Runs the longest ready prefix of d->deferred (submitMutex held); stops at the first batch with a
// wait nothing can meet yet.
void drainDeferred(Device* d) {
  while (!d->deferred.empty()) {
    std::vector<std::pair<Semaphore*, uint64_t>> ahead;
    size_t ready = 0;
    for (; ready < d->deferred.size(); ++ready) {
      const DeferredBatch& b = d->deferred[ready];
      bool met = true;
      for (const auto& [s, v] : b.waits) met = met && waitMet(s, v, ahead);
      if (!met) break;
      ahead.insert(ahead.end(), b.signals.begin(), b.signals.end());
    }
    if (!ready) return;
    runBatches(d, d->deferred, ready);
  }
}
}  // namespace

void submitDeferred(Device* d) {
  std::lock_guard<std::mutex> lock(d->submitMutex);
  if (!d->deferred.empty() && !d->platform->lost()) drainDeferred(d);
}
}  // namespace vpvk

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(VkQueue q, uint32_t n, const VkSubmitInfo* si,
                                                         VkFence fence) {
  Device* d = from<Queue>(q)->device;
  std::lock_guard<std::mutex> lock(d->submitMutex);
  if (d->platform->lost()) return VK_ERROR_DEVICE_LOST;
  const size_t first = d->deferred.size();
  for (uint32_t i = 0; i < n; ++i) {
    DeferredBatch b;
    for (uint32_t j = 0; j < si[i].commandBufferCount; ++j) b.cbs.push_back(from<CommandBuffer>(si[i].pCommandBuffers[j]));
    const auto* t = static_cast<const VkTimelineSemaphoreSubmitInfo*>(
        findNext(si[i].pNext, VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO));
    if (t && t->pWaitSemaphoreValues)
      for (uint32_t j = 0; j < si[i].waitSemaphoreCount && j < t->waitSemaphoreValueCount; ++j)
        if (Semaphore* s = from<Semaphore>(si[i].pWaitSemaphores[j]); s && s->timeline)
          b.waits.push_back({s, t->pWaitSemaphoreValues[j]});
    if (t && t->pSignalSemaphoreValues)
      for (uint32_t j = 0; j < si[i].signalSemaphoreCount && j < t->signalSemaphoreValueCount; ++j)
        if (Semaphore* s = from<Semaphore>(si[i].pSignalSemaphores[j]); s && s->timeline)
          b.signals.push_back({s, t->pSignalSemaphoreValues[j]});
    d->deferred.push_back(std::move(b));
  }
  if (fence) {
    if (d->deferred.size() == first) d->deferred.emplace_back();  // n == 0: the fence alone
    Fence* fe = from<Fence>(fence);
    fe->deferred = true;  // until its batch runs (cleared by runBatches)
    fe->signaled = true;
    d->deferred.back().fence = fe;
  }
  const size_t before = d->deferred.size();
  drainDeferred(d);
  if (!d->deferred.empty() && d->deferred.size() == before) {
    static std::atomic<unsigned> k{0};
    if (k.fetch_add(1) < 16)
      logf("vk.submit deferred batches=%zu (timeline wait before signal)", d->deferred.size());
  }
  return VK_SUCCESS;
}

// VPKTAA inline resolve (vpvk/config.h). The submit mutex orders the native command buffer with
// guest submissions: a vkQueueSubmit holding a frame while its pipelines compile (gpu.hold) keeps
// the mutex until that frame is committed, so taking it here means every submission accepted so far
// is already on the Metal queue; the backend then orders the native work after it on the GPU.
VPVK_EXPORT int vpvk_native_submit(vpvk_native_encode_fn fn, void* user) {
  if (!fn) return 0;
  // Lock order registry -> submit (as vkDestroyDevice): the registry is released before `fn` runs,
  // so the callback may call vpvk_native_device; it must not call vkQueueSubmit (same mutex).
  std::unique_lock<std::mutex> live(liveDeviceMutex());
  Device* d = liveDevice();
  if (!d || !d->platform) return 0;
  std::lock_guard<std::mutex> lock(d->submitMutex);
  live.unlock();
  tInNativeSubmit = true;
  const bool ok = d->platform->encodeNative(fn, user);
  tInNativeSubmit = false;
  // The first success and the first few failures (a failure would repeat every frame).
  static std::atomic<unsigned> okLogged{0}, failLogged{0};
  if ((ok ? okLogged : failLogged).fetch_add(1) < (ok ? 1u : 8u))
    logf("vk.native submit ok=%d after_submit=%llu", ok ? 1 : 0, static_cast<unsigned long long>(d->submits));
  return ok ? 1 : 0;
}

VPVK_EXPORT void* vpvk_native_device(void) {
  std::lock_guard<std::mutex> live(liveDeviceMutex());
  Device* d = liveDevice();
  return d && d->platform ? d->platform->nativeDevice() : nullptr;
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
    E(vkQueueSubmit, true), E(vkGetMTLTextureMVK, true),
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
  if (PFN_vkVoidFunction f = lookupExt(name)) return f;
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
