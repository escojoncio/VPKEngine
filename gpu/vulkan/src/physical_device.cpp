// The one physical device VPVulkan exposes. Limits and features are what a Quest 3 (Adreno 740)
// driver reports where Apple GPUs can honour them, so engines that pick their Vulkan path from these
// numbers (Unity) take the same one they take on the headset; where the Vision Pro GPU is stricter,
// the stricter value is reported.
#include <cstring>

#include "vpvk.h"

namespace vpvk {
namespace {

VkPhysicalDeviceProperties makeProps() {
  VkPhysicalDeviceProperties p{};
  p.apiVersion = VK_API_VERSION_1_1;
  p.driverVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
  p.vendorID = 0x5143;      // Qualcomm: engines key their mobile paths off it
  p.deviceID = 0x43050a01;  // Adreno 740
  p.deviceType = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
  std::strncpy(p.deviceName, "Adreno (TM) 740 (VPVulkan)", sizeof p.deviceName - 1);
  const char uuid[] = "VPVulkan-AVP-01";
  std::memcpy(p.pipelineCacheUUID, uuid, sizeof uuid - 1);

  VkPhysicalDeviceLimits& l = p.limits;
  l.maxImageDimension1D = 16384;
  l.maxImageDimension2D = 16384;
  l.maxImageDimension3D = 2048;
  l.maxImageDimensionCube = 16384;
  l.maxImageArrayLayers = 2048;
  l.maxTexelBufferElements = 1u << 26;
  l.maxUniformBufferRange = 65536;
  l.maxStorageBufferRange = 1u << 30;
  l.maxPushConstantsSize = 128;
  l.maxMemoryAllocationCount = 4096;
  l.maxSamplerAllocationCount = 4000;
  l.bufferImageGranularity = 1;  // unified memory, no linear/optimal aliasing hazard
  l.sparseAddressSpaceSize = 0;
  l.maxBoundDescriptorSets = 4;
  l.maxPerStageDescriptorSamplers = 16;
  l.maxPerStageDescriptorUniformBuffers = 15;
  l.maxPerStageDescriptorStorageBuffers = 24;
  l.maxPerStageDescriptorSampledImages = 128;
  l.maxPerStageDescriptorStorageImages = 8;
  l.maxPerStageDescriptorInputAttachments = 8;
  l.maxPerStageResources = 184;
  l.maxDescriptorSetSamplers = 96;
  l.maxDescriptorSetUniformBuffers = 90;
  l.maxDescriptorSetUniformBuffersDynamic = 32;
  l.maxDescriptorSetStorageBuffers = 144;
  l.maxDescriptorSetStorageBuffersDynamic = 16;
  l.maxDescriptorSetSampledImages = 768;
  l.maxDescriptorSetStorageImages = 48;
  l.maxDescriptorSetInputAttachments = 8;
  l.maxVertexInputAttributes = 31;  // Metal's vertex buffer table keeps slot 30 for arguments
  l.maxVertexInputBindings = 31;
  l.maxVertexInputAttributeOffset = 2047;
  l.maxVertexInputBindingStride = 2048;
  l.maxVertexOutputComponents = 124;
  l.maxTessellationGenerationLevel = 0;
  l.maxGeometryShaderInvocations = 0;
  l.maxFragmentInputComponents = 124;
  l.maxFragmentOutputAttachments = 8;
  l.maxFragmentDualSrcAttachments = 1;
  l.maxFragmentCombinedOutputResources = 72;
  l.maxComputeSharedMemorySize = 32768;
  l.maxComputeWorkGroupCount[0] = l.maxComputeWorkGroupCount[1] = l.maxComputeWorkGroupCount[2] = 65535;
  l.maxComputeWorkGroupInvocations = 1024;
  l.maxComputeWorkGroupSize[0] = 1024;
  l.maxComputeWorkGroupSize[1] = 1024;
  l.maxComputeWorkGroupSize[2] = 64;
  l.subPixelPrecisionBits = 8;
  l.subTexelPrecisionBits = 8;
  l.mipmapPrecisionBits = 8;
  l.maxDrawIndexedIndexValue = 0xFFFFFFFFu;
  l.maxDrawIndirectCount = 1;
  l.maxSamplerLodBias = 15.99f;
  l.maxSamplerAnisotropy = 16;
  l.maxViewports = 1;
  l.maxViewportDimensions[0] = l.maxViewportDimensions[1] = 16384;
  l.viewportBoundsRange[0] = -32768;
  l.viewportBoundsRange[1] = 32767;
  l.viewportSubPixelBits = 8;
  l.minMemoryMapAlignment = 64;
  l.minTexelBufferOffsetAlignment = 64;
  l.minUniformBufferOffsetAlignment = 64;
  l.minStorageBufferOffsetAlignment = 64;
  l.minTexelOffset = -8;
  l.maxTexelOffset = 7;
  l.minTexelGatherOffset = -8;
  l.maxTexelGatherOffset = 7;
  l.minInterpolationOffset = -0.5f;
  l.maxInterpolationOffset = 0.4375f;
  l.subPixelInterpolationOffsetBits = 4;
  l.maxFramebufferWidth = l.maxFramebufferHeight = 16384;
  l.maxFramebufferLayers = 2048;
  const VkSampleCountFlags ms = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_2_BIT | VK_SAMPLE_COUNT_4_BIT;
  l.framebufferColorSampleCounts = l.framebufferDepthSampleCounts = ms;
  l.framebufferStencilSampleCounts = l.framebufferNoAttachmentsSampleCounts = ms;
  l.maxColorAttachments = 8;
  l.sampledImageColorSampleCounts = l.sampledImageIntegerSampleCounts = ms;
  l.sampledImageDepthSampleCounts = l.sampledImageStencilSampleCounts = ms;
  l.storageImageSampleCounts = VK_SAMPLE_COUNT_1_BIT;
  l.maxSampleMaskWords = 1;
  l.timestampComputeAndGraphics = VK_TRUE;
  l.timestampPeriod = 1.0f;  // nanoseconds
  l.maxClipDistances = 8;
  l.maxCullDistances = 0;
  l.maxCombinedClipAndCullDistances = 8;
  l.discreteQueuePriorities = 2;
  l.pointSizeRange[0] = 1;
  l.pointSizeRange[1] = 511;
  l.lineWidthRange[0] = l.lineWidthRange[1] = 1;
  l.pointSizeGranularity = 1;
  l.lineWidthGranularity = 0;
  l.strictLines = VK_FALSE;
  l.standardSampleLocations = VK_TRUE;
  l.optimalBufferCopyOffsetAlignment = 64;
  l.optimalBufferCopyRowPitchAlignment = 64;
  l.nonCoherentAtomSize = 64;
  return p;
}

VkPhysicalDeviceMemoryProperties makeMem() {
  // One heap: Apple silicon memory is unified. Types mirror a mobile driver so allocators that
  // look for DEVICE_LOCAL|HOST_VISIBLE (and LAZILY_ALLOCATED for transient targets) find them.
  VkPhysicalDeviceMemoryProperties m{};
  m.memoryHeapCount = 1;
  m.memoryHeaps[0].size = 6ull << 30;  // what a Quest 3 title may budget; the visor has more
  m.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
  const VkMemoryPropertyFlags dl = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  const VkMemoryPropertyFlags types[] = {dl, dl | hv, dl | hv | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                                         dl | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT};
  m.memoryTypeCount = 4;
  for (uint32_t i = 0; i < 4; ++i) m.memoryTypes[i] = {types[i], 0};
  return m;
}

struct FormatRow {
  VkFormat vk;
  vpgpu::Format core;
};
// Formats the core can create. Anything else reports no features, so the engine falls back.
constexpr FormatRow kFormats[] = {
    {VK_FORMAT_R8_UNORM, vpgpu::Format::R8Unorm}, {VK_FORMAT_R8_SNORM, vpgpu::Format::R8Snorm},
    {VK_FORMAT_R8_UINT, vpgpu::Format::R8Uint}, {VK_FORMAT_R8_SINT, vpgpu::Format::R8Sint},
    {VK_FORMAT_R8G8_UNORM, vpgpu::Format::RG8Unorm}, {VK_FORMAT_R8G8_SNORM, vpgpu::Format::RG8Snorm},
    {VK_FORMAT_R8G8_UINT, vpgpu::Format::RG8Uint}, {VK_FORMAT_R8G8_SINT, vpgpu::Format::RG8Sint},
    {VK_FORMAT_R8G8B8A8_UNORM, vpgpu::Format::RGBA8Unorm}, {VK_FORMAT_R8G8B8A8_SRGB, vpgpu::Format::RGBA8Srgb},
    {VK_FORMAT_R8G8B8A8_SNORM, vpgpu::Format::RGBA8Snorm}, {VK_FORMAT_R8G8B8A8_UINT, vpgpu::Format::RGBA8Uint},
    {VK_FORMAT_R8G8B8A8_SINT, vpgpu::Format::RGBA8Sint}, {VK_FORMAT_B8G8R8A8_UNORM, vpgpu::Format::BGRA8Unorm},
    {VK_FORMAT_B8G8R8A8_SRGB, vpgpu::Format::BGRA8Srgb},
    {VK_FORMAT_A8B8G8R8_UNORM_PACK32, vpgpu::Format::RGBA8Unorm},
    {VK_FORMAT_A8B8G8R8_SRGB_PACK32, vpgpu::Format::RGBA8Srgb},
    {VK_FORMAT_R16_UNORM, vpgpu::Format::R16Unorm}, {VK_FORMAT_R16_SNORM, vpgpu::Format::R16Snorm},
    {VK_FORMAT_R16_UINT, vpgpu::Format::R16Uint}, {VK_FORMAT_R16_SINT, vpgpu::Format::R16Sint},
    {VK_FORMAT_R16_SFLOAT, vpgpu::Format::R16Float}, {VK_FORMAT_R16G16_UNORM, vpgpu::Format::RG16Unorm},
    {VK_FORMAT_R16G16_SNORM, vpgpu::Format::RG16Snorm}, {VK_FORMAT_R16G16_UINT, vpgpu::Format::RG16Uint},
    {VK_FORMAT_R16G16_SINT, vpgpu::Format::RG16Sint}, {VK_FORMAT_R16G16_SFLOAT, vpgpu::Format::RG16Float},
    {VK_FORMAT_R16G16B16A16_UNORM, vpgpu::Format::RGBA16Unorm},
    {VK_FORMAT_R16G16B16A16_SNORM, vpgpu::Format::RGBA16Snorm},
    {VK_FORMAT_R16G16B16A16_UINT, vpgpu::Format::RGBA16Uint},
    {VK_FORMAT_R16G16B16A16_SINT, vpgpu::Format::RGBA16Sint},
    {VK_FORMAT_R16G16B16A16_SFLOAT, vpgpu::Format::RGBA16Float},
    {VK_FORMAT_R32_UINT, vpgpu::Format::R32Uint}, {VK_FORMAT_R32_SINT, vpgpu::Format::R32Sint},
    {VK_FORMAT_R32_SFLOAT, vpgpu::Format::R32Float}, {VK_FORMAT_R32G32_UINT, vpgpu::Format::RG32Uint},
    {VK_FORMAT_R32G32_SINT, vpgpu::Format::RG32Sint}, {VK_FORMAT_R32G32_SFLOAT, vpgpu::Format::RG32Float},
    {VK_FORMAT_R32G32B32A32_UINT, vpgpu::Format::RGBA32Uint},
    {VK_FORMAT_R32G32B32A32_SINT, vpgpu::Format::RGBA32Sint},
    {VK_FORMAT_R32G32B32A32_SFLOAT, vpgpu::Format::RGBA32Float},
    {VK_FORMAT_A2B10G10R10_UNORM_PACK32, vpgpu::Format::RGB10A2Unorm},
    {VK_FORMAT_A2B10G10R10_UINT_PACK32, vpgpu::Format::RGB10A2Uint},
    {VK_FORMAT_B10G11R11_UFLOAT_PACK32, vpgpu::Format::RG11B10Float},
    {VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, vpgpu::Format::RGB9E5Float},
    {VK_FORMAT_R5G6B5_UNORM_PACK16, vpgpu::Format::B5G6R5Unorm},
    {VK_FORMAT_D16_UNORM, vpgpu::Format::D16Unorm}, {VK_FORMAT_D32_SFLOAT, vpgpu::Format::D32Float},
    {VK_FORMAT_D24_UNORM_S8_UINT, vpgpu::Format::D24UnormS8},
    {VK_FORMAT_D32_SFLOAT_S8_UINT, vpgpu::Format::D32FloatS8}, {VK_FORMAT_S8_UINT, vpgpu::Format::S8Uint},
    {VK_FORMAT_BC1_RGBA_UNORM_BLOCK, vpgpu::Format::BC1}, {VK_FORMAT_BC1_RGBA_SRGB_BLOCK, vpgpu::Format::BC1Srgb},
    {VK_FORMAT_BC3_UNORM_BLOCK, vpgpu::Format::BC3}, {VK_FORMAT_BC3_SRGB_BLOCK, vpgpu::Format::BC3Srgb},
    {VK_FORMAT_BC7_UNORM_BLOCK, vpgpu::Format::BC7}, {VK_FORMAT_BC7_SRGB_BLOCK, vpgpu::Format::BC7Srgb},
};

}  // namespace

const VkPhysicalDeviceProperties& deviceProperties() {
  static const VkPhysicalDeviceProperties p = makeProps();
  return p;
}

const VkPhysicalDeviceMemoryProperties& memoryProperties() {
  static const VkPhysicalDeviceMemoryProperties m = makeMem();
  return m;
}

vpgpu::Format toCoreFormat(VkFormat f) {
  for (const FormatRow& r : kFormats)
    if (r.vk == f) return r.core;
  return vpgpu::Format::Undefined;
}

VkFormatFeatureFlags formatFeatures(VkFormat f, bool buffer) {
  const vpgpu::Format core = toCoreFormat(f);
  if (core == vpgpu::Format::Undefined) return 0;
  const vpgpu::FormatInfo& info = vpgpu::formatInfo(core);
  const bool compressed = info.blockSize > 1;
  if (buffer) {
    if (compressed || info.depth() || info.stencil()) return 0;
    VkFormatFeatureFlags b = VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT | VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT;
    if (!info.srgb()) b |= VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT;
    return b;
  }
  VkFormatFeatureFlags t = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                           VK_FORMAT_FEATURE_TRANSFER_DST_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT;
  if (compressed) return t | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
  if (info.depth() || info.stencil()) {
    t |= VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (info.depth()) t |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    return t;
  }
  t |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
  if (!info.integer()) {
    t |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT;
  }
  if (!info.srgb()) t |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
  return t;
}

std::vector<VkExtensionProperties> instanceExtensions() {
  return {{VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME, 2},
          {VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME, 1},
          {VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME, 1},
          {VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME, 1}};
}

std::vector<VkExtensionProperties> deviceExtensions() {
  return {{VK_KHR_MULTIVIEW_EXTENSION_NAME, 1},
          {VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME, 1},
          {VK_KHR_MAINTENANCE1_EXTENSION_NAME, 2},
          {VK_KHR_BIND_MEMORY_2_EXTENSION_NAME, 1},
          {VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME, 1},
          {VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME, 3},
          {VK_KHR_DESCRIPTOR_UPDATE_TEMPLATE_EXTENSION_NAME, 1},
          {VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME, 2},
          {VK_EXT_METAL_OBJECTS_EXTENSION_NAME, 2}};
}

}  // namespace vpvk
