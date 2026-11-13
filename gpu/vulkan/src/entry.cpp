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
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>

#if defined(__APPLE__) || defined(__linux__)
#include <sys/mman.h>
#define VPVK_HOST_MMAP 1
#endif

#include "commands.h"
#include "perf.h"
#include "vpgpu/core/background.h"
#include "vpgpu/core/scoped_pool.h"
#include "recipes.h"
#include "texdecode.h"
#include "vpvk.h"
#include "vpvk/config.h"

#define VPVK_EXPORT extern "C" __attribute__((visibility("default")))

#ifndef VPK_GIT_SHA  // set by gpu/vulkan/CMakeLists.txt; builds outside it have no identity
#define VPK_GIT_SHA "unknown"
#endif

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

// Bytes of VkDeviceMemory currently allocated (VK_EXT_memory_budget heapUsage); lazily allocated memory takes none.
std::atomic<uint64_t>& allocatedBytes() {
  static std::atomic<uint64_t> n{0};
  return n;
}

namespace {

// Diagnostics (vpvk_diag_line): what the guest asked of the Vulkan layer, as relaxed counters. No log line per
// call: the host reads deltas every 250 ms.
namespace diag {
constexpr unsigned kTypes = 8;  // memory types counted separately (the device has 4)
struct Counters {
  std::atomic<uint64_t> allocN[kTypes], allocB[kTypes], freeN[kTypes], freeB[kTypes];
  std::atomic<uint64_t> bufCreateN, bufCreateB, bufDestroyN, bufBindN, bufBindB;
  std::atomic<uint64_t> imgCreateN, imgCreatePx, imgDestroyN, imgBindN;
  // vkCreateImage by requested tiling (img_create_linear / img_create_optimal in [vkagg]; the sum of the deltas is the total).
  std::atomic<uint64_t> imgLinearN, imgOptimalN;
  // Image/texture lifetime: textures really destroyed / skipped because an aliasing image still shares them (releaseImageTexture
  // of a bound image), and vkDestroyImage calls on an image that never got a texture (never bound).
  std::atomic<uint64_t> imgRelDestroyN, imgRelSharedN, imgRelNoTexN, viewCreateN, viewDestroyN;
  std::atomic<uint64_t> mapN, mapB, unmapN, flushN, flushB, invN, invB;
};
Counters& c() {
  static Counters* k = new Counters();  // zero-initialized atomics; leaked like the other registries
  return *k;
}
inline void add(std::atomic<uint64_t>& a, uint64_t n = 1) { a.fetch_add(n, std::memory_order_relaxed); }
// Host allocations of the host-visible VkDeviceMemory that are alive (address -> size): vpvk_diag_ranges tells
// the host which regions of the process the guest's mapped memory occupies.
struct HostRanges {
  std::mutex mu;
  std::map<uintptr_t, uint64_t> r;
};
HostRanges& ranges() {
  static HostRanges* h = new HostRanges();
  return *h;
}
void rangeAdd(const void* p, uint64_t n) {
  if (!p) return;
  std::lock_guard<std::mutex> l(ranges().mu);
  ranges().r[reinterpret_cast<uintptr_t>(p)] = n;
}
void rangeDel(const void* p) {
  if (!p) return;
  std::lock_guard<std::mutex> l(ranges().mu);
  ranges().r.erase(reinterpret_cast<uintptr_t>(p));
}
}  // namespace diag

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

// "vk.imgzero fmt=<core format> vkfmt=<n> w= h= mips= layers= usage=0x.. why=..": an image that ends with zeroed contents
// (zeroInitResolve) or whose format was converted / has no core format (imageDesc). Once per (why, format), 16 lines at most.
// vkfmt is -1 where the Vulkan format is not at hand; usage is the Vulkan flags (imageDesc) or the core ones (zeroInitResolve).
void noteImgZero(const char* why, int vkFmt, vpgpu::Format fmt, uint32_t w, uint32_t h, uint32_t mips, uint32_t layers, uint32_t usage) {
  static std::atomic<size_t> count{0};
  if (count.load(std::memory_order_relaxed) >= 16) return;  // the limit is reached: no string, no mutex
  static std::mutex m;
  static std::set<std::string> seen;
  const std::string key = std::string(why) + "/" + std::to_string(vkFmt >= 0 ? vkFmt : -1 - int(fmt));
  std::lock_guard<std::mutex> lock(m);
  if (seen.size() >= 16 || !seen.insert(key).second) return;
  count.store(seen.size(), std::memory_order_relaxed);
  logf("vk.imgzero fmt=%s vkfmt=%d w=%u h=%u mips=%u layers=%u usage=0x%x why=%s", vpgpu::formatInfo(fmt).name, vkFmt, w, h, mips,
       layers, usage, why);
}

vpgpu::TextureDesc imageDesc(const VkImageCreateInfo& ci) {
  vpgpu::TextureDesc d;
  d.format = toCoreFormat(ci.format);
  if (d.format == vpgpu::Format::Undefined && ci.format != VK_FORMAT_UNDEFINED)
    noteImgZero("unsupported", int(ci.format), d.format, ci.extent.width, ci.extent.height, ci.mipLevels, ci.arrayLayers, ci.usage);
  // Metal has no 3D ASTC: the volume is RGBA8 and its uploads are decoded on the CPU (texdecode.h, commands.cpp).
  if (ci.imageType == VK_IMAGE_TYPE_3D && astcFormat(d.format)) {
    d.format = astcDecodedFormat(d.format);
    noteImgZero("astc3d_decoded", int(ci.format), d.format, ci.extent.width, ci.extent.height, ci.mipLevels, ci.arrayLayers, ci.usage);
  }
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
  d.samplesReadable = (ci.usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) != 0;
  // (frame compiler: msaaDeadStore.) The stencil usage counts whether or not vkCreateImage merged it (that needs nativeExtras()).
  VkImageUsageFlags stencilUsage = 0;
  if (const auto* su = static_cast<const VkImageStencilUsageCreateInfo*>(findNext(ci.pNext, VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO)))
    stencilUsage = su->stencilUsage;
  d.shaderReadable = ((ci.usage | stencilUsage) & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) != 0;
  // The 1x images forced MSAA resolves into (and seeds samples from, Attachment::needsSeed): a shader reads them. The game's own
  // MSAA (gameMsaaAuto) adds nothing: commands.cpp gameAuto only serves resolve targets that are already readable.
  if (d.samples <= 1 && (ci.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) &&
      !(ci.usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) && msaaForce() > 0) {
    d.usage |= vpgpu::usage::sampled;
    // A depth/stencil one also gets format views: a forced twin that loads its stencil is seeded from it through a stencil view
    // (commands.cpp forceMsaa; its depth plane is read as is).
    if (ufi.depth() && ufi.stencil()) d.usage |= vpgpu::usage::formatView;
  }
  // Transient attachments never need memory: the core keeps them on tile when it can.
  d.transient = (ci.usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) != 0;
  // Anything sampled may be read in a later frame (Vulkan gives no frame boundary).
  d.persistent = !d.transient;
  return d;
}

}  // namespace

// ---- texture budget ---------------------------------------------------------------------------------------------------------
// A Quest keeps a title alive through its texture peaks with what the title's engine streams and with swap; here every byte of a
// texture counts against the process limit. The host turns the process' memory state into a size cap (vpvk_texture_budget) and
// an image created above it loses its largest levels: the same picture, less sharp up close, instead of a kill.
namespace {
std::atomic<uint32_t> g_textureBudget{0};
std::atomic<uint64_t> g_budgetImages{0}, g_budgetSavedBytes{0};
}  // namespace
uint32_t textureBudget() { return g_textureBudget.load(std::memory_order_relaxed); }
void setTextureBudget(uint32_t maxDim) { g_textureBudget.store(maxDim, std::memory_order_relaxed); }

bool budgetCanShrink(vpgpu::Format f) {
  switch (f) {
    case vpgpu::Format::R8Unorm: case vpgpu::Format::R8Srgb: case vpgpu::Format::RG8Unorm: case vpgpu::Format::RG8Srgb:
    case vpgpu::Format::RGBA8Unorm: case vpgpu::Format::RGBA8Srgb: case vpgpu::Format::BGRA8Unorm: case vpgpu::Format::BGRA8Srgb:
      return true;
    default: return false;
  }
}

uint32_t textureBudgetDrop(const VkImageCreateInfo& ci, vpgpu::Format core, uint32_t cap) {
  if (!cap || core == vpgpu::Format::Undefined) return 0;
  const VkImageUsageFlags plain = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if (ci.imageType != VK_IMAGE_TYPE_2D || ci.samples != VK_SAMPLE_COUNT_1_BIT || ci.tiling != VK_IMAGE_TILING_OPTIMAL ||
      !(ci.usage & VK_IMAGE_USAGE_SAMPLED_BIT) || (ci.usage & ~plain) || multiPlane(ci.format).planes ||
      (ci.flags & (VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT | VK_IMAGE_CREATE_SPARSE_ALIASED_BIT)))
    return 0;
  const bool compressed = vpgpu::formatInfo(core).blockSize > 1;
  if (!compressed && !budgetCanShrink(core)) return 0;
  uint32_t k = 0;
  while ((std::max(ci.extent.width, ci.extent.height) >> k) > cap) ++k;
  // A mip chain keeps at least its smallest level asked for; a single level only shrinks where its upload can be scaled.
  if (ci.mipLevels > 1 || compressed) k = std::min(k, ci.mipLevels - 1);
  return k;
}

BudgetShrink budgetShrink(const Image& im, const VkBufferImageCopy& c) {
  BudgetShrink o;
  if (!im.mipDrop || c.imageSubresource.mipLevel != 0 || !budgetCanShrink(im.desc.format)) return o;
  const uint32_t s = im.mipDrop, ox = uint32_t(c.imageOffset.x), oy = uint32_t(c.imageOffset.y);
  const uint32_t x1 = std::min(im.desc.width, (ox + c.imageExtent.width + (1u << s) - 1) >> s);
  const uint32_t y1 = std::min(im.desc.height, (oy + c.imageExtent.height + (1u << s) - 1) >> s);
  if (x1 <= (ox >> s) || y1 <= (oy >> s) || c.imageSubresource.baseArrayLayer >= im.desc.layers) return o;
  o.x = ox >> s, o.y = oy >> s, o.width = x1 - o.x, o.height = y1 - o.y;
  o.layers = c.imageSubresource.layerCount == VK_REMAINING_ARRAY_LAYERS ? im.desc.layers - c.imageSubresource.baseArrayLayer
                                                                          : c.imageSubresource.layerCount;
  o.texelBytes = vpgpu::formatInfo(im.desc.format).bytesPerBlock;
  o.bytes = VkDeviceSize(o.width) * o.height * o.texelBytes * o.layers;
  return o;
}

bool budgetRange(const Image* im, VkImageSubresourceRange& r) {
  const uint32_t k = im->mipDrop;
  const uint32_t count = r.levelCount == VK_REMAINING_MIP_LEVELS ? im->info.mipLevels - r.baseMipLevel : r.levelCount;
  const uint32_t end = r.baseMipLevel + count;
  const uint32_t b = r.baseMipLevel > k ? r.baseMipLevel - k : 0, e = std::min<uint32_t>(end > k ? end - k : 0, im->desc.mips);
  if (e <= b) return false;
  r.baseMipLevel = b, r.levelCount = e - b;
  return true;
}

namespace {

uint64_t imageBytes(const vpgpu::TextureDesc& d) {
  const vpgpu::FormatInfo& fi = vpgpu::formatInfo(d.format);
  uint64_t total = 0;
  for (uint32_t m = 0; m < d.mips; ++m) {
    const uint64_t w = std::max<uint32_t>(1, d.width >> m), h = std::max<uint32_t>(1, d.height >> m);
    const uint64_t z = std::max<uint32_t>(1, d.depth >> m);
    total += (w + fi.blockSize - 1) / fi.blockSize * ((h + fi.blockHeight - 1) / fi.blockHeight) * z * fi.bytesPerBlock;
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
  *v = deviceProperties().apiVersion;  // 1.3 (1.3.295 with the Quest identity): physical_device.cpp makeProps
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
  static std::once_flag versionOnce;
  std::call_once(versionOnce, [] { logf("vpk.version sha=%s", VPK_GIT_SHA); });
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
  static std::once_flag once;  // one line with what engines decide on, to compare with the real Adreno 740
  std::call_once(once, [&] {
    const VkPhysicalDeviceLimits& l = p->limits;
    logf("vk.caps identity=%s vendor=0x%x device=\"%s\"", questIdentity() ? "quest" : "native", p->vendorID, p->deviceName);
    logf("vk.caps limits apiVersion=%u.%u.%u maxVertexOutputComponents=%u maxFragmentInputComponents=%u "
         "maxColorAttachments=%u maxComputeWorkGroupInvocations=%u maxComputeSharedMemorySize=%u "
         "maxImageDimension2D=%u maxImageArrayLayers=%u maxPerStageDescriptorStorageBuffers=%u "
         "maxPerStageDescriptorStorageImages=%u maxBoundDescriptorSets=%u maxSamplerAnisotropy=%g "
         "framebufferColorSampleCounts=0x%x sampledImageColorSampleCounts=0x%x",
         VK_API_VERSION_MAJOR(p->apiVersion), VK_API_VERSION_MINOR(p->apiVersion), VK_API_VERSION_PATCH(p->apiVersion),
         l.maxVertexOutputComponents, l.maxFragmentInputComponents, l.maxColorAttachments,
         l.maxComputeWorkGroupInvocations, l.maxComputeSharedMemorySize, l.maxImageDimension2D,
         l.maxImageArrayLayers, l.maxPerStageDescriptorStorageBuffers, l.maxPerStageDescriptorStorageImages,
         l.maxBoundDescriptorSets, double(l.maxSamplerAnisotropy), unsigned(l.framebufferColorSampleCounts),
         unsigned(l.sampledImageColorSampleCounts));
  });
}

namespace {
constexpr uint32_t kSubgroupSize = 32;  // Apple SIMD-group width (the Adreno's 64 cannot be honoured)
constexpr VkShaderStageFlags kSubgroupStages = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
constexpr VkSubgroupFeatureFlags kSubgroupOps =
    VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT |
    VK_SUBGROUP_FEATURE_SHUFFLE_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT | VK_SUBGROUP_FEATURE_QUAD_BIT |
    VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_CLUSTERED_BIT;  // arithmetic / clustered: vpir subgroup_arith
constexpr uint32_t kMaxInlineUniformBlockSize = 256, kMaxInlineUniformBlocks = 4;  // the minimums
// What the Adreno driver reports with the Quest identity (VK_KHR_driver_properties): engines key Qualcomm
// workarounds off driverID, never off the name.
void fillDriver(VkPhysicalDeviceDriverProperties& d) {
  std::memset(d.driverName, 0, sizeof d.driverName);
  std::memset(d.driverInfo, 0, sizeof d.driverInfo);
  if (questIdentity()) {
    d.driverID = VK_DRIVER_ID_QUALCOMM_PROPRIETARY;
    std::strncpy(d.driverName, "Qualcomm Technologies Inc. Adreno Vulkan Driver", sizeof d.driverName - 1);
    std::strncpy(d.driverInfo, "Driver Build: 512.837.7", sizeof d.driverInfo - 1);
  } else {
    d.driverID = VK_DRIVER_ID_QUALCOMM_PROPRIETARY;  // no Apple id exists; the one AVP validated, so no MoltenVK paths turn on
    std::strncpy(d.driverName, "VPVulkan", sizeof d.driverName - 1);
    std::strncpy(d.driverInfo, "VPKEngine Vulkan on Metal", sizeof d.driverInfo - 1);
  }
  // The API version reported (deviceProperties().apiVersion) is 1.3; the Quest identity reports the headset's
  // conformance version too (1.3.0.6: major, minor, subminor, patch).
  const uint32_t api = deviceProperties().apiVersion;
  d.conformanceVersion = {uint8_t(VK_API_VERSION_MAJOR(api)), uint8_t(VK_API_VERSION_MINOR(api)), 0,
                          uint8_t(questIdentity() ? 6 : 0)};
}
}  // namespace

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties2(VkPhysicalDevice pd,
                                                                      VkPhysicalDeviceProperties2* p) {
  vkGetPhysicalDeviceProperties(pd, &p->properties);
  for (auto* s = static_cast<VkBaseOutStructure*>(p->pNext); s; s = s->pNext) {
    switch (s->sType) {
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES: {
        auto* m = reinterpret_cast<VkPhysicalDeviceMultiviewProperties*>(s);
        // Views are instances (view = instance % count, layer = the mask's bit index; spirv lower.cpp),
        // not Metal vertex amplification: any count up to the layered-rendering limit works.
        m->maxMultiviewViewCount = questIdentity() ? 6 : 8;  // Quest 3 real: 6
        m->maxMultiviewInstanceIndex = (1u << 27) - 1;  // x8 views (native) still fits 32-bit instance ids
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
        r->independentResolveNone = r->independentResolve = questIdentity() ? VK_FALSE : VK_TRUE;  // Quest 3 real: false
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
        sg->subgroupSize = kSubgroupSize;
        sg->supportedStages = kSubgroupStages;
        sg->supportedOperations = kSubgroupOps;
        sg->quadOperationsInAllStages = VK_FALSE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES:
        fillDriver(*reinterpret_cast<VkPhysicalDeviceDriverProperties*>(s));
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES: {
        auto* v = reinterpret_cast<VkPhysicalDeviceVulkan11Properties*>(s);
        std::memcpy(v->deviceUUID, deviceProperties().pipelineCacheUUID, VK_UUID_SIZE);
        std::memcpy(v->driverUUID, deviceProperties().pipelineCacheUUID, VK_UUID_SIZE);
        static const uint8_t luid[VK_LUID_SIZE] = {'V', 'P', 'V', 'K', 0, 0, 0, 1};
        std::memcpy(v->deviceLUID, luid, VK_LUID_SIZE);
        v->deviceNodeMask = 1;
        v->deviceLUIDValid = VK_TRUE;
        v->subgroupSize = kSubgroupSize;
        v->subgroupSupportedStages = kSubgroupStages;
        v->subgroupSupportedOperations = kSubgroupOps;
        v->subgroupQuadOperationsInAllStages = VK_FALSE;
        v->pointClippingBehavior = VK_POINT_CLIPPING_BEHAVIOR_ALL_CLIP_PLANES;
        v->maxMultiviewViewCount = questIdentity() ? 6 : 8;  // Quest 3 real: 6
        v->maxMultiviewInstanceIndex = (1u << 27) - 1;
        v->protectedNoFault = VK_FALSE;
        v->maxPerSetDescriptors = 1024;
        v->maxMemoryAllocationSize = 1ull << 32;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES: {
        auto* v = reinterpret_cast<VkPhysicalDeviceVulkan12Properties*>(s);
        void* next = v->pNext;
        *v = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES, next};
        VkPhysicalDeviceDriverProperties d{};
        fillDriver(d);
        v->driverID = d.driverID;
        std::memcpy(v->driverName, d.driverName, sizeof v->driverName);
        std::memcpy(v->driverInfo, d.driverInfo, sizeof v->driverInfo);
        v->conformanceVersion = d.conformanceVersion;
        v->denormBehaviorIndependence = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE;
        v->roundingModeIndependence = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE;
        v->shaderRoundingModeRTEFloat32 = VK_TRUE;
        v->supportedDepthResolveModes = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT | VK_RESOLVE_MODE_MIN_BIT | VK_RESOLVE_MODE_MAX_BIT;
        v->supportedStencilResolveModes = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
        v->independentResolveNone = v->independentResolve = questIdentity() ? VK_FALSE : VK_TRUE;  // Quest 3 real: false
        v->maxTimelineSemaphoreValueDifference = UINT64_MAX;
        v->framebufferIntegerColorSampleCounts = VK_SAMPLE_COUNT_1_BIT;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES: {
        auto* v = reinterpret_cast<VkPhysicalDeviceVulkan13Properties*>(s);
        void* next = v->pNext;
        *v = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES, next};
        v->minSubgroupSize = v->maxSubgroupSize = kSubgroupSize;
        v->maxComputeWorkgroupSubgroups = deviceProperties().limits.maxComputeWorkGroupInvocations / kSubgroupSize;
        const auto& l = deviceProperties().limits;
        v->storageTexelBufferOffsetAlignmentBytes = l.minTexelBufferOffsetAlignment;
        v->uniformTexelBufferOffsetAlignmentBytes = l.minTexelBufferOffsetAlignment;
        v->maxBufferSize = 1ull << 32;
        v->requiredSubgroupSizeStages = VK_SHADER_STAGE_COMPUTE_BIT;
        v->maxInlineUniformBlockSize = kMaxInlineUniformBlockSize;
        v->maxPerStageDescriptorInlineUniformBlocks = kMaxInlineUniformBlocks;
        v->maxDescriptorSetInlineUniformBlocks = kMaxInlineUniformBlocks;
        v->maxPerStageDescriptorUpdateAfterBindInlineUniformBlocks = kMaxInlineUniformBlocks;
        v->maxDescriptorSetUpdateAfterBindInlineUniformBlocks = kMaxInlineUniformBlocks;
        v->maxInlineUniformTotalSize = kMaxInlineUniformBlockSize * kMaxInlineUniformBlocks;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES: {
        auto* sc = reinterpret_cast<VkPhysicalDeviceSubgroupSizeControlProperties*>(s);
        sc->minSubgroupSize = sc->maxSubgroupSize = kSubgroupSize;
        sc->maxComputeWorkgroupSubgroups = deviceProperties().limits.maxComputeWorkGroupInvocations / kSubgroupSize;
        sc->requiredSubgroupSizeStages = VK_SHADER_STAGE_COMPUTE_BIT;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_PROPERTIES: {
        auto* iu = reinterpret_cast<VkPhysicalDeviceInlineUniformBlockProperties*>(s);
        iu->maxInlineUniformBlockSize = kMaxInlineUniformBlockSize;
        iu->maxPerStageDescriptorInlineUniformBlocks = kMaxInlineUniformBlocks;
        iu->maxPerStageDescriptorUpdateAfterBindInlineUniformBlocks = kMaxInlineUniformBlocks;  // (limit only: the feature is false)
        iu->maxDescriptorSetInlineUniformBlocks = kMaxInlineUniformBlocks;
        iu->maxDescriptorSetUpdateAfterBindInlineUniformBlocks = kMaxInlineUniformBlocks;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES: {
        // Same values as the float-controls part of Vulkan12Properties.
        auto* x = reinterpret_cast<VkPhysicalDeviceFloatControlsProperties*>(s);
        void* next = x->pNext;
        *x = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES, next};
        x->denormBehaviorIndependence = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE;
        x->roundingModeIndependence = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE;
        x->shaderRoundingModeRTEFloat32 = VK_TRUE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_FILTER_MINMAX_PROPERTIES: {  // samplerFilterMinmax is false
        auto* x = reinterpret_cast<VkPhysicalDeviceSamplerFilterMinmaxProperties*>(s);
        void* next = x->pNext;
        *x = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_FILTER_MINMAX_PROPERTIES, next};
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES: {  // no dot-product acceleration
        auto* x = reinterpret_cast<VkPhysicalDeviceShaderIntegerDotProductProperties*>(s);
        void* next = x->pNext;
        *x = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES, next};
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_PROPERTIES: {
        auto* x = reinterpret_cast<VkPhysicalDeviceTexelBufferAlignmentProperties*>(s);
        void* next = x->pNext;
        *x = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_PROPERTIES, next};
        x->storageTexelBufferOffsetAlignmentBytes = x->uniformTexelBufferOffsetAlignmentBytes =
            deviceProperties().limits.minTexelBufferOffsetAlignment;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR:
        reinterpret_cast<VkPhysicalDevicePushDescriptorPropertiesKHR*>(s)->maxPushDescriptors = 32;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_PROPERTIES_EXT:
        reinterpret_cast<VkPhysicalDeviceVertexAttributeDivisorPropertiesEXT*>(s)->maxVertexAttribDivisor = 65535;  // Adreno's value
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES:
        reinterpret_cast<VkPhysicalDeviceMaintenance4Properties*>(s)->maxBufferSize = 1ull << 32;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES: {
        // descriptorIndexing is false, but an engine that chains the struct reads its limits: none of it is left unset.
        auto* x = reinterpret_cast<VkPhysicalDeviceDescriptorIndexingProperties*>(s);
        void* next = x->pNext;
        *x = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES, next};
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_PROPERTIES_EXT: {
        auto* h = reinterpret_cast<VkPhysicalDeviceHostImageCopyPropertiesEXT*>(s);
        if (!nativeExtras()) break;
        // Every layout works: layouts are not tracked (copies go through the GPU).
        static const VkImageLayout layouts[] = {
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR};
        const uint32_t n = sizeof layouts / sizeof *layouts;
        auto fill = [&](uint32_t& count, VkImageLayout* out) {
          if (!out) {
            count = n;
            return;
          }
          count = std::min(count, n);
          std::memcpy(out, layouts, count * sizeof *out);
        };
        fill(h->copySrcLayoutCount, h->pCopySrcLayouts);
        fill(h->copyDstLayoutCount, h->pCopyDstLayouts);
        std::memcpy(h->optimalTilingLayoutUUID, deviceProperties().pipelineCacheUUID, VK_UUID_SIZE);
        h->identicalMemoryTypeRequirements = VK_TRUE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_PROPERTIES_EXT:
        reinterpret_cast<VkPhysicalDeviceExtendedDynamicState3PropertiesEXT*>(s)->dynamicPrimitiveTopologyUnrestricted = VK_FALSE;
        break;
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
  // Per-sample shading as a shader asks for it: SampleId, the Sample decoration and interpolateAtSample
  // are lowered (Metal runs the fragment function per sample when it reads sample_id). The
  // minSampleShading hint of a pipeline is not forced: Metal cannot be told to shade a fraction.
  f->sampleRateShading = VK_TRUE;
  f->dualSrcBlend = VK_TRUE;
  f->depthClamp = VK_TRUE;
  f->depthBiasClamp = VK_TRUE;
  f->fillModeNonSolid = VK_TRUE;  // LINE: Metal fill mode lines; POINT: triangles drawn as points (pipelines.cpp primitiveOf)
  // Metal rasterizes [[point_size]] up to 511 (pointSizeRange); Adreno reports largePoints too.
  f->largePoints = VK_TRUE;
  f->samplerAnisotropy = VK_TRUE;
  // BCn when the GPU samples it (the Adreno 740 reports it as well).
  f->textureCompressionBC = textureCompressionBC() ? VK_TRUE : VK_FALSE;
  f->textureCompressionETC2 = VK_TRUE;
  f->textureCompressionASTC_LDR = VK_TRUE;  // every LDR block size, 4x4..12x12 (Metal ASTC_*_LDR)
  f->occlusionQueryPrecise = VK_TRUE;
  f->fragmentStoresAndAtomics = VK_TRUE;
  // Adreno 740 reports it; behind the CTS gate until dEQP-VK robustness/binding_model pass on the
  // headset (exposed to games it hung AC Nexus's shader warmup, 2026-10-16).
  f->vertexPipelineStoresAndAtomics = extraExtensions() ? VK_TRUE : VK_FALSE;
  f->shaderImageGatherExtended = VK_TRUE;
  f->shaderStorageImageExtendedFormats = VK_TRUE;
  f->shaderClipDistance = VK_TRUE;
  f->shaderSampledImageArrayDynamicIndexing = VK_TRUE;
  f->shaderUniformBufferArrayDynamicIndexing = VK_TRUE;
  f->shaderStorageBufferArrayDynamicIndexing = VK_TRUE;
  // Run-time index into a storage image array: the same per-element chain as sampled images
  // (spirv lower.cpp dynamicHandleOperand -> unrollDynamic covers OpImageRead/Write/atomics).
  f->shaderStorageImageArrayDynamicIndexing = VK_TRUE;
  // 16-bit integers live in 32-bit registers, re-normalised after every op (spirv lower.cpp
  // narrow16); memory keeps 32-bit slots, so no 16-bit storage feature comes with it.
  // Behind the CTS gate until dEQP-VK.spirv_assembly.*int16* pass on the headset.
  f->shaderInt16 = extraExtensions() ? VK_TRUE : VK_FALSE;
  // COMPATIBILITY MASK, not an implementation: there is no geometry shader on Metal and no tessellation
  // stage in the translator (pipelines.cpp rejects a pipeline that carries either with `vk.err what=stage`).
  // Off by default: the answer is then the truth (false, with zero geometry / tessellation limits).
  // The Adreno 740 (Quest 3) reports both bits and Unity decides per SUBSHADER whether a shader is
  // usable from them: a subshader compiled for `#pragma target` 4.6/5.0 (or `require
  // geometry/tessellation`) is dropped when they are false, and a shader left with no usable
  // subshader draws Unity's error colour (255,0,255; AC Nexus's whole UI came out magenta). Those
  // subshaders carry no geometry stage in practice, so VisionQuestPro sets VPVK_REPORT_GEOM=1 for games
  // (never for the CTS) and the rare pipeline that really has one fails visibly in the log.
  if (reportGeom()) {
    f->geometryShader = VK_TRUE;
    f->tessellationShader = VK_TRUE;
  }
  static std::once_flag once;  // one line with what engines decide on, to compare with the real Adreno 740
  std::call_once(once, [&] {
    const VkPhysicalDeviceFeatures& x = *f;
    logf("vk.caps features geometryShader=%u tessellationShader=%u imageCubeArray=%u sampleRateShading=%u "
         "fragmentStoresAndAtomics=%u vertexPipelineStoresAndAtomics=%u shaderInt16=%u shaderFloat64=%u "
         "shaderStorageImageMultisample=%u shaderClipDistance=%u multiViewport=%u astc=%u etc2=%u bc=%u "
         "depthClamp=%u independentBlend=%u dualSrcBlend=%u logicOp=%u multiDrawIndirect=%u",
         x.geometryShader, x.tessellationShader, x.imageCubeArray, x.sampleRateShading, x.fragmentStoresAndAtomics,
         x.vertexPipelineStoresAndAtomics, x.shaderInt16, x.shaderFloat64, x.shaderStorageImageMultisample,
         x.shaderClipDistance, x.multiViewport, x.textureCompressionASTC_LDR, x.textureCompressionETC2,
         x.textureCompressionBC, x.depthClamp, x.independentBlend, x.dualSrcBlend, x.logicOp, x.multiDrawIndirect);
  });
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(VkPhysicalDevice pd,
                                                                    VkPhysicalDeviceFeatures2* f) {
  vkGetPhysicalDeviceFeatures(pd, &f->features);
  // What the driver really does, in one place; every extension / VulkanNN struct reads from it.
  // Not offered (the translator or the driver does not do it): 16-bit shader inputs/outputs,
  // variablePointers (workgroup pointers), protected memory, descriptor indexing, shader Int64
  // atomics, availability / visibility chains, update-after-bind (see bitacora.md, "No viable"). Games see
  // apiVersion 1.3 (1.3.295 with the Quest identity) like the Adreno 740.
  const VkBool32 multiDraw = extraExtensions();  // EXT_multi_draw is no Adreno extension
  // Everything added for the Adreno profile that the Vulkan CTS has not validated yet answers only
  // in the conformance run (VPVK_EXTRA_EXTENSIONS=1); games see the pre-profile answers.
  const VkBool32 gate = extraExtensions();
  const VkBool32 native = nativeExtras();  // what Alyx uses and an Adreno does not list (native identity only)
  for (auto* s = static_cast<VkBaseOutStructure*>(f->pNext); s; s = s->pNext) {
    switch (s->sType) {
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES: {
        auto* m = reinterpret_cast<VkPhysicalDeviceMultiviewFeatures*>(s);
        m->multiview = VK_TRUE;
        m->multiviewGeometryShader = VK_FALSE;
        m->multiviewTessellationShader = VK_FALSE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES:
        reinterpret_cast<VkPhysicalDeviceTimelineSemaphoreFeatures*>(s)->timelineSemaphore = gate;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES:
        reinterpret_cast<VkPhysicalDeviceSynchronization2Features*>(s)->synchronization2 = gate;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES:
        reinterpret_cast<VkPhysicalDeviceDynamicRenderingFeatures*>(s)->dynamicRendering = gate;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES:
        reinterpret_cast<VkPhysicalDeviceSeparateDepthStencilLayoutsFeatures*>(s)->separateDepthStencilLayouts = gate;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTI_DRAW_FEATURES_EXT:
        reinterpret_cast<VkPhysicalDeviceMultiDrawFeaturesEXT*>(s)->multiDraw = multiDraw;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES:
        // Buffer members are addressed by their SPIR-V Offset / ArrayStride / MatrixStride and any
        // access not provably aligned is a packed_* load (MSL emitter), so C-like layouts need nothing.
        reinterpret_cast<VkPhysicalDeviceScalarBlockLayoutFeatures*>(s)->scalarBlockLayout = gate;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES:
        reinterpret_cast<VkPhysicalDeviceUniformBufferStandardLayoutFeatures*>(s)->uniformBufferStandardLayout = gate;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES:
        reinterpret_cast<VkPhysicalDeviceProtectedMemoryFeatures*>(s)->protectedMemory = VK_FALSE;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES:
        reinterpret_cast<VkPhysicalDeviceHostQueryResetFeatures*>(s)->hostQueryReset = gate;
        break;
      // Host-side Vulkan 1.3 pieces (ext.cpp, entry.cpp), behind the CTS gate.
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES:
        reinterpret_cast<VkPhysicalDevicePrivateDataFeatures*>(s)->privateData = gate;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES:
        reinterpret_cast<VkPhysicalDeviceMaintenance4Features*>(s)->maintenance4 = gate;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES:
        reinterpret_cast<VkPhysicalDeviceImagelessFramebufferFeatures*>(s)->imagelessFramebuffer = gate;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES: {
        // One subgroup size (32): any required size is that one; threadgroups of a multiple of 32
        // run full SIMD-groups.
        auto* sc = reinterpret_cast<VkPhysicalDeviceSubgroupSizeControlFeatures*>(s);
        sc->subgroupSizeControl = sc->computeFullSubgroups = gate;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES: {
        // Buffers / push constants: 2-byte members (spirv lower.cpp load16 / store16). Not the
        // shader interface (storageInputOutput16).
        auto* f16 = reinterpret_cast<VkPhysicalDevice16BitStorageFeatures*>(s);
        f16->storageBuffer16BitAccess = f16->uniformAndStorageBuffer16BitAccess = gate;
        f16->storagePushConstant16 = gate;
        f16->storageInputOutput16 = VK_FALSE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT: {
        reinterpret_cast<VkPhysicalDeviceIndexTypeUint8FeaturesEXT*>(s)->indexTypeUint8 = gate;  // widened to 16 bits per draw
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES: {
        // Bytes 1 apart in buffers / push constants (spirv lower.cpp load16 / store16 with 1-byte leaves).
        auto* f8 = reinterpret_cast<VkPhysicalDevice8BitStorageFeatures*>(s);
        f8->storageBuffer8BitAccess = f8->uniformAndStorageBuffer8BitAccess = f8->storagePushConstant8 = gate;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES: {
        // Half arithmetic is MSL half; 8-bit integers live in 32-bit registers, extended after each op.
        auto* fi = reinterpret_cast<VkPhysicalDeviceShaderFloat16Int8Features*>(s);
        fi->shaderFloat16 = fi->shaderInt8 = gate;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES: {
        // Acquire / release, MakePointerAvailable / Visible and memory barriers become sequentially
        // consistent atomic_thread_fence (spirv lower.cpp semanticsFence). No availability / visibility chains.
        auto* mm = reinterpret_cast<VkPhysicalDeviceVulkanMemoryModelFeatures*>(s);
        mm->vulkanMemoryModel = mm->vulkanMemoryModelDeviceScope = gate;
        mm->vulkanMemoryModelAvailabilityVisibilityChains = VK_FALSE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES: {
        // PhysicalStorageBuffer pointers (spirv lower.cpp, vpir kTargetDevice). Not capture / replay.
        auto* ba = reinterpret_cast<VkPhysicalDeviceBufferDeviceAddressFeatures*>(s);
        ba->bufferDeviceAddress = gate;
        ba->bufferDeviceAddressCaptureReplay = ba->bufferDeviceAddressMultiDevice = VK_FALSE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES: {
        // OpSelect / OpPhi / OpPtrAccessChain of storage-buffer pointers (spirv lower.cpp: a {tag, address}
        // value, accesses branch on the tag). Not variablePointers: Workgroup pointers and
        // OpPtrEqual / OpPtrDiff are not lowered.
        auto* vp = reinterpret_cast<VkPhysicalDeviceVariablePointersFeatures*>(s);
        vp->variablePointersStorageBuffer = gate;
        vp->variablePointers = VK_FALSE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES:
        // OpDemoteToHelperInvocation -> discard_fragment() (a demote in MSL 2.3+), IsHelperInvocation ->
        // simd_is_helper_thread().
        reinterpret_cast<VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures*>(s)->shaderDemoteToHelperInvocation = gate;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES: {
        // The block's bytes live in the descriptor set and reach the draw through the upload arena as a
        // constant buffer (pipelines.cpp resolveBindings). No update after bind.
        auto* iu = reinterpret_cast<VkPhysicalDeviceInlineUniformBlockFeatures*>(s);
        iu->inlineUniformBlock = gate;
        iu->descriptorBindingInlineUniformBlockUpdateAfterBind = VK_FALSE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT:
        // Cull mode, front face, topology, depth/stencil state, strides and viewport/scissor counts
        // (pipelines.cpp pipelineFor / makeDraw): pipeline variants over the replaced values.
        reinterpret_cast<VkPhysicalDeviceExtendedDynamicStateFeaturesEXT*>(s)->extendedDynamicState = gate;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT: {
        // Rasterizer discard, depth bias enable, primitive restart enable. Not the logic op (Metal has
        // none) nor patch control points (no tessellation).
        auto* e2 = reinterpret_cast<VkPhysicalDeviceExtendedDynamicState2FeaturesEXT*>(s);
        e2->extendedDynamicState2 = gate;
        e2->extendedDynamicState2LogicOp = VK_FALSE;
        e2->extendedDynamicState2PatchControlPoints = VK_FALSE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT: {
        // Only the colour write mask (Alyx declares it on its pipelines): pipelines.cpp applyEds, a pipeline
        // variant over the masks. Everything else of the extension is false.
        auto* e3 = reinterpret_cast<VkPhysicalDeviceExtendedDynamicState3FeaturesEXT*>(s);
        void* next = e3->pNext;
        *e3 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT, next};
        e3->extendedDynamicState3ColorWriteMask = native;
        break;
      }
      // Features this driver does not have, written as "not supported" instead of the caller's bytes (an engine that
      // does not zero its structs reads garbage). Native identity only.
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT:
        if (native) reinterpret_cast<VkPhysicalDeviceMemoryPriorityFeaturesEXT*>(s)->memoryPriority = VK_FALSE;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES: {
        auto* x = reinterpret_cast<VkPhysicalDeviceDescriptorIndexingFeatures*>(s);
        void* next = x->pNext;
        *x = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES, next};
        break;
      }
      // Core 1.2 / 1.3 feature structs of things this driver does not do: written as false for every identity (an
      // unwritten struct reads as the caller's bytes) and in step with the VulkanNN structs below.
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES: {
        auto* x = reinterpret_cast<VkPhysicalDeviceShaderAtomicInt64Features*>(s);
        x->shaderBufferInt64Atomics = x->shaderSharedInt64Atomics = VK_FALSE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES:
        reinterpret_cast<VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures*>(s)->shaderSubgroupExtendedTypes = gate;  // laneOp on any scalar/vector type (lower.cpp)
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT:
        reinterpret_cast<VkPhysicalDeviceDepthClipEnableFeaturesEXT*>(s)->depthClipEnable = gate;  // pipelines.cpp: Metal clip mode
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT: {
        auto* x = reinterpret_cast<VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT*>(s);  // lower.cpp dividedInstance
        x->vertexAttributeInstanceRateDivisor = x->vertexAttributeInstanceRateZeroDivisor = gate;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ASTC_DECODE_FEATURES_EXT:
        reinterpret_cast<VkPhysicalDeviceASTCDecodeFeaturesEXT*>(s)->decodeModeSharedExponent = VK_FALSE;  // the mode itself is accepted and ignored
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_FEATURES_EXT:
        reinterpret_cast<VkPhysicalDeviceTexelBufferAlignmentFeaturesEXT*>(s)->texelBufferAlignment = gate;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES:
        // OpTerminateInvocation -> discard_fragment (same answer as Vulkan13Features).
        reinterpret_cast<VkPhysicalDeviceShaderTerminateInvocationFeatures*>(s)->shaderTerminateInvocation = gate;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES:
        reinterpret_cast<VkPhysicalDeviceImageRobustnessFeatures*>(s)->robustImageAccess = gate;  // lower.cpp imageInBounds
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES:
        reinterpret_cast<VkPhysicalDevicePipelineCreationCacheControlFeatures*>(s)->pipelineCreationCacheControl = gate;  // pipelines.cpp
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES:
        reinterpret_cast<VkPhysicalDeviceShaderIntegerDotProductFeatures*>(s)->shaderIntegerDotProduct = gate;  // lower.cpp dotProduct
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES:
        reinterpret_cast<VkPhysicalDeviceTextureCompressionASTCHDRFeatures*>(s)->textureCompressionASTC_HDR = VK_FALSE;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES:
        reinterpret_cast<VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeatures*>(s)->shaderZeroInitializeWorkgroupMemory = gate;  // lower.cpp zeroWorkgroup
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT:
        // vkCopyMemoryToImageEXT & co. (ext.cpp): staged GPU copies, waited for.
        reinterpret_cast<VkPhysicalDeviceHostImageCopyFeaturesEXT*>(s)->hostImageCopy = native;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES_KHR:
        // An in-pass barrier splits the dynamic rendering pass where a later draw reads an attachment, which is
        // read through a copy (commands.cpp localReadSplit).
        reinterpret_cast<VkPhysicalDeviceDynamicRenderingLocalReadFeaturesKHR*>(s)->dynamicRenderingLocalRead = native;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES:
        // gl_BaseVertex / gl_BaseInstance: Metal [[base_vertex]] / [[base_instance]]; gl_DrawID: the
        // record index of indirect multi-draws, 0 otherwise (spirv lower.cpp, metal2 draw params).
        reinterpret_cast<VkPhysicalDeviceShaderDrawParametersFeatures*>(s)->shaderDrawParameters = gate;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES:
        // 8-bit 2- and 3-plane 420 / 422 / 444 formats: one texture per plane, chroma reconstructed
        // and converted in the shader (spirv lower.cpp ycbcrSample). No external formats.
        reinterpret_cast<VkPhysicalDeviceSamplerYcbcrConversionFeatures*>(s)->samplerYcbcrConversion = gate;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_FEATURES_EXT: {
        auto* d = reinterpret_cast<VkPhysicalDeviceFragmentDensityMapFeaturesEXT*>(s);
        d->fragmentDensityMap = VK_FALSE;  // not advertised: foveation is VPKEngine's (vpvk/config.h)
        d->fragmentDensityMapDynamic = VK_FALSE;
        d->fragmentDensityMapNonSubsampledImages = VK_FALSE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES: {
        auto* v = reinterpret_cast<VkPhysicalDeviceVulkan11Features*>(s);
        void* next = v->pNext;
        *v = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, next};
        v->multiview = VK_TRUE;
        v->shaderDrawParameters = gate;
        v->variablePointersStorageBuffer = gate;
        v->samplerYcbcrConversion = gate;
        v->storageBuffer16BitAccess = v->uniformAndStorageBuffer16BitAccess = v->storagePushConstant16 = gate;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: {
        auto* v = reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(s);
        void* next = v->pNext;
        *v = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, next};
        if (!gate) break;  // games: the struct stays zero until the CTS validates it
        v->samplerMirrorClampToEdge = VK_TRUE;
        v->drawIndirectCount = VK_TRUE;
        v->scalarBlockLayout = VK_TRUE;
        v->uniformBufferStandardLayout = VK_TRUE;
        v->separateDepthStencilLayouts = VK_TRUE;
        v->imagelessFramebuffer = VK_TRUE;
        v->timelineSemaphore = VK_TRUE;
        v->hostQueryReset = VK_TRUE;
        // gl_Layer from the vertex stage: Metal [[render_target_array_index]] (multiview uses it).
        v->shaderOutputLayer = VK_TRUE;
        v->subgroupBroadcastDynamicId = VK_TRUE;  // simd_broadcast takes a run-time lane
        v->shaderFloat16 = v->shaderInt8 = VK_TRUE;
        v->bufferDeviceAddress = VK_TRUE;
        v->vulkanMemoryModel = v->vulkanMemoryModelDeviceScope = VK_TRUE;
        v->storageBuffer8BitAccess = v->uniformAndStorageBuffer8BitAccess = v->storagePushConstant8 = VK_TRUE;
        v->shaderSubgroupExtendedTypes = VK_TRUE;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: {
        auto* v = reinterpret_cast<VkPhysicalDeviceVulkan13Features*>(s);
        void* next = v->pNext;
        *v = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, next};
        if (!gate) break;
        v->synchronization2 = VK_TRUE;
        v->dynamicRendering = VK_TRUE;
        v->privateData = VK_TRUE;
        v->maintenance4 = VK_TRUE;
        v->subgroupSizeControl = VK_TRUE;
        v->computeFullSubgroups = VK_TRUE;
        v->shaderTerminateInvocation = VK_TRUE;  // OpTerminateInvocation -> discard_fragment
        v->inlineUniformBlock = VK_TRUE;
        v->shaderDemoteToHelperInvocation = VK_TRUE;
        v->robustImageAccess = VK_TRUE;
        v->pipelineCreationCacheControl = VK_TRUE;
        v->shaderZeroInitializeWorkgroupMemory = VK_TRUE;
        v->shaderIntegerDotProduct = VK_TRUE;
        break;
      }
      default: break;
    }
  }
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties(
    VkPhysicalDevice, uint32_t* count, VkQueueFamilyProperties* out) {
  VkQueueFamilyProperties q{};
  q.queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
  q.queueCount = kQueueCount;
  q.timestampValidBits = kTimestampValidBits;
  q.minImageTransferGranularity = {1, 1, 1};
  fillArray(std::vector<VkQueueFamilyProperties>{q}, count, out);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties2(
    VkPhysicalDevice pd, uint32_t* count, VkQueueFamilyProperties2* out) {
  if (!out) return vkGetPhysicalDeviceQueueFamilyProperties(pd, count, nullptr);
  if (*count) vkGetPhysicalDeviceQueueFamilyProperties(pd, count, &out[0].queueFamilyProperties);
}

// vk.mem_seen: what a game was told about memory, once per question (the heaps; the budget when it chains
// VkPhysicalDeviceMemoryBudgetPropertiesEXT: filled=0 = not answered, the extension is not announced with this identity).
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties(
    VkPhysicalDevice, VkPhysicalDeviceMemoryProperties* m) {
  *m = memoryProperties();
  static std::atomic<bool> seen{false};
  if (!internalClient() && !seen.exchange(true))
    logf("vk.mem_seen heaps=%u heap0=%llu heap0_flags=%#x types=%u", m->memoryHeapCount, (unsigned long long)m->memoryHeaps[0].size,
         unsigned(m->memoryHeaps[0].flags), m->memoryTypeCount);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties2(
    VkPhysicalDevice pd, VkPhysicalDeviceMemoryProperties2* m) {
  vkGetPhysicalDeviceMemoryProperties(pd, &m->memoryProperties);
  // VK_EXT_memory_budget (native identity): usage = VkDeviceMemory allocated; budget = usage + what the OS still lets
  // the process allocate (visionOS jetsam limit, Platform::availableMemory), capped by the heap. Without a device or
  // on hosts that cannot tell (Linux tests): 80 % of the heap. Source 2 sizes its texture streaming pool from it.
  if (auto* b = static_cast<VkPhysicalDeviceMemoryBudgetPropertiesEXT*>(
          findNextOut(m->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT));
      b && nativeExtras()) {
    std::memset(b->heapBudget, 0, sizeof b->heapBudget);
    std::memset(b->heapUsage, 0, sizeof b->heapUsage);
    const uint64_t heap = memoryProperties().memoryHeaps[0].size, used = allocatedBytes().load();
    uint64_t avail = 0;
    {
      std::lock_guard<std::mutex> l(liveDeviceMutex());
      const Device* d = liveDevice();
      if (d && d->platform) avail = d->platform->availableMemory();
    }
    b->heapUsage[0] = used;
    b->heapBudget[0] = std::max(used, avail ? std::min(heap, used + avail) : heap / 10 * 8);  // never under the usage
  }
  static std::atomic<bool> seen{false};
  if (const auto* b = static_cast<const VkPhysicalDeviceMemoryBudgetPropertiesEXT*>(
          findNextOut(m->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT));
      b && !internalClient() && !seen.exchange(true))
    logf("vk.mem_seen budget_asked=1 filled=%d budget0=%llu usage0=%llu heap0=%llu", int(nativeExtras()),
         (unsigned long long)(nativeExtras() ? b->heapBudget[0] : 0), (unsigned long long)(nativeExtras() ? b->heapUsage[0] : 0),
         (unsigned long long)m->memoryProperties.memoryHeaps[0].size);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties(VkPhysicalDevice,
                                                                           VkFormat f,
                                                                           VkFormatProperties* p) {
  p->optimalTilingFeatures = formatFeatures(f, false);
  // Linear images are only offered for transfers (staging): Metal linear textures are restrictive.
  // Multi-planar linear images: transfers only (each plane is a linear image of its own, see vkCreateImage).
  // A linear colour image is the same texture as an optimal one (the host memory is its staging), so what a sampler or a blit
  // source does with it holds for both: Quest reports linear = optimal for these.
  const vpgpu::FormatInfo& lin = vpgpu::formatInfo(toCoreFormat(f));
  const VkFormatFeatureFlags linearReads = lin.depth() || lin.stencil() ? 0
                                           : VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT;
  p->linearTilingFeatures = p->optimalTilingFeatures &
                            (multiPlane(f).planes ? (VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT)
                                                  : (VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
                                                     VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | linearReads));
  p->bufferFeatures = formatFeatures(f, true);
  // A format the game asks about and gets nothing for: the first 64 are named (an engine that refuses to start over a
  // missing format then says which one in the log).
  if (nativeExtras() && !p->optimalTilingFeatures && !p->bufferFeatures) {
    static std::mutex m;
    static std::set<int> seen;
    std::lock_guard<std::mutex> lock(m);
    if (seen.size() < 64 && seen.insert(int(f)).second) logf("vk.format unsupported=%d", int(f));
  }
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties2(VkPhysicalDevice pd,
                                                                            VkFormat f,
                                                                            VkFormatProperties2* p) {
  vkGetPhysicalDeviceFormatProperties(pd, f, &p->formatProperties);
  // VkFormatProperties3 (core 1.3, VK_KHR_format_feature_flags2) is answered for every identity: it carries at least
  // the 32-bit flags of VkFormatProperties (the CTS reads usage support from it).
  if (auto* p3 = static_cast<VkFormatProperties3*>(findNextOut(p->pNext, VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3))) {
    const VkFormatProperties& fp = p->formatProperties;
    p3->linearTilingFeatures = fp.linearTilingFeatures;
    p3->optimalTilingFeatures = fp.optimalTilingFeatures;
    p3->bufferFeatures = fp.bufferFeatures;
    // Depth formats can be sampled with comparison, linear or optimal (a sampler compare mode on the depth texture).
    if (vpgpu::formatInfo(toCoreFormat(f)).depth()) {
      if (fp.linearTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
        p3->linearTilingFeatures |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT;
      if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
        p3->optimalTilingFeatures |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT;
    }
    if (!nativeExtras()) return;
    // Storage images are read and written without a declared format (the MSL type comes from the SPIR-V
    // sampled type); every image can be a host transfer target.
    if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)
      p3->optimalTilingFeatures |= VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT |
                                   VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
    if (fp.optimalTilingFeatures && !multiPlane(f).planes)
      p3->optimalTilingFeatures |= VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT;
  }
}

static VkResult imageFormatProperties(VkFormat f, VkImageType type, VkImageTiling tiling, VkImageUsageFlags usage,
                                      VkImageCreateFlags flags, VkImageFormatProperties* out) {
  *out = {};  // every field 0 when the combination is refused (spec; the CTS checks it)
  // VK_EXT_host_image_copy: any image can be a vkCopyMemoryToImageEXT target (host transfers are staged copies).
  if (nativeExtras()) usage &= ~VkImageUsageFlags(VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT);
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
  // Multi-planar (Y'CbCr) formats: optimal-tiling 2D images only.
  // Linear ones: transfers only (one mip level, one layer: see maxMipLevels / maxArrayLayers below).
  if (multiPlane(f).planes && (type != VK_IMAGE_TYPE_2D || (flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) ||
                               (tiling == VK_IMAGE_TILING_LINEAR &&
                                (usage & ~(VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)))))
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
    if (type == VK_IMAGE_TYPE_3D && fi.name && (!strncmp(fi.name, "Etc2", 4) || !strncmp(fi.name, "Eac", 3)))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    // 3D ASTC is decoded to RGBA8 (texdecode.h): sampling works, but the compressed blocks are gone, so block-exact
    // copies (the CTS copy_and_blit / compressed_3D cases) cannot. Not offered in the conformance run; vkCreateImage
    // still takes it for a game that creates one without asking.
    if (extraExtensions() && type == VK_IMAGE_TYPE_3D && fi.name && !strncmp(fi.name, "Astc", 4))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
  }
  const auto& l = deviceProperties().limits;
  *out = {};
  out->maxExtent = {l.maxImageDimension2D, type == VK_IMAGE_TYPE_1D ? 1u : l.maxImageDimension2D,
                    type == VK_IMAGE_TYPE_3D ? l.maxImageDimension3D : 1u};
  // Native identity: a 3D image is limited by maxImageDimension3D on every axis (the limit the spec ties it to).
  if (nativeExtras() && type == VK_IMAGE_TYPE_3D)
    out->maxExtent = {l.maxImageDimension3D, l.maxImageDimension3D, l.maxImageDimension3D};
  out->maxMipLevels = 15;
  out->maxArrayLayers = type == VK_IMAGE_TYPE_3D ? 1 : l.maxImageArrayLayers;
  if (multiPlane(f).planes && tiling == VK_IMAGE_TILING_LINEAR) out->maxMipLevels = out->maxArrayLayers = 1;
  // Multisample only where Vulkan allows it: optimal 2D, not cube-compatible, an attachment format
  // (Metal renders MSAA into it), not storage (storageImageSampleCounts = 1), and not transfer-only.
  const bool msaa = tiling == VK_IMAGE_TILING_OPTIMAL && type == VK_IMAGE_TYPE_2D &&
                    !(flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) && !(usage & VK_IMAGE_USAGE_STORAGE_BIT) &&
                    (usage & ~VkImageUsageFlags(VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)) &&
                    (ff & (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT));
  out->sampleCounts = msaa ? l.framebufferColorSampleCounts : VkSampleCountFlags{VK_SAMPLE_COUNT_1_BIT};
  out->maxResourceSize = 1ull << 32;
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties(
    VkPhysicalDevice, VkFormat f, VkImageType type, VkImageTiling tiling, VkImageUsageFlags usage,
    VkImageCreateFlags flags, VkImageFormatProperties* out) {
  VkResult r = imageFormatProperties(f, type, tiling, usage, flags, out);
  // VK_KHR_maintenance2 EXTENDED_USAGE on a mutable image: the usage is valid when some compatible uncompressed colour
  // format (same texel size) supports it without the flags (an sRGB image used as storage through its UNORM view).
  if (r == VK_ERROR_FORMAT_NOT_SUPPORTED && (flags & VK_IMAGE_CREATE_EXTENDED_USAGE_BIT) &&
      (flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT)) {
    const vpgpu::FormatInfo& fi = vpgpu::formatInfo(toCoreFormat(f));
    const VkImageCreateFlags plain = flags & ~VkImageCreateFlags(VK_IMAGE_CREATE_EXTENDED_USAGE_BIT);
    if (fi.blockSize == 1 && !fi.depth() && !fi.stencil() && !multiPlane(f).planes)
      for (int g = 1; g <= int(VK_FORMAT_ASTC_12x12_SRGB_BLOCK) && r != VK_SUCCESS; ++g) {
        const VkFormat vf = VkFormat(g);
        if (vf == f || !formatFeatures(vf, false)) continue;
        const vpgpu::FormatInfo& gi = vpgpu::formatInfo(toCoreFormat(vf));
        if (gi.blockSize != 1 || gi.depth() || gi.stencil() || multiPlane(vf).planes || gi.bytesPerBlock != fi.bytesPerBlock)
          continue;
        r = imageFormatProperties(vf, type, tiling, usage, plain, out);
      }
  }
  return r;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties2(
    VkPhysicalDevice pd, const VkPhysicalDeviceImageFormatInfo2* in, VkImageFormatProperties2* out) {
  VkImageUsageFlags usage = in->usage;
  // VK_EXT_separate_stencil_usage: one Metal texture holds both aspects, so the union must be possible.
  if (nativeExtras())
    if (const auto* su = static_cast<const VkImageStencilUsageCreateInfo*>(
            findNext(in->pNext, VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO)))
      usage |= su->stencilUsage;
  VkResult r = vkGetPhysicalDeviceImageFormatProperties(pd, in->format, in->type, in->tiling, usage, in->flags,
                                                        &out->imageFormatProperties);
  // VK_KHR_maintenance2 EXTENDED_USAGE: the usage is valid when a view format of the list supports it (an sRGB image
  // used as storage through its UNORM view).
  if (r == VK_ERROR_FORMAT_NOT_SUPPORTED && (in->flags & VK_IMAGE_CREATE_EXTENDED_USAGE_BIT) &&
      (in->flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT))
    if (const auto* fl = static_cast<const VkImageFormatListCreateInfo*>(
            findNext(in->pNext, VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO)))
      for (uint32_t i = 0; i < fl->viewFormatCount && r != VK_SUCCESS; ++i)
        if (fl->pViewFormats[i] != in->format)
          r = vkGetPhysicalDeviceImageFormatProperties(pd, fl->pViewFormats[i], in->type, in->tiling, usage, in->flags,
                                                       &out->imageFormatProperties);
  if (auto* q = static_cast<VkHostImageCopyDevicePerformanceQueryEXT*>(
          findNextOut(out->pNext, VK_STRUCTURE_TYPE_HOST_IMAGE_COPY_DEVICE_PERFORMANCE_QUERY_EXT))) {
    if (nativeExtras()) {
      q->optimalDeviceAccess = VK_TRUE;  // Metal textures keep their layout; nothing is degraded
      q->identicalMemoryLayout = VK_FALSE;
    }
  }
  if (auto* yp = static_cast<VkSamplerYcbcrConversionImageFormatProperties*>(
          findNextOut(out->pNext, VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES)))
    yp->combinedImageSamplerDescriptorCount = 1;  // one descriptor, whatever the number of planes
  return r;
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
// Diagnostics: vk.device_req (what vkCreateDevice asked) and vk.tex_hist (texture residency by mips written)
// ---------------------------------------------------------------------------------------------
namespace vpvk {
namespace {
// Feature structs a vkCreateDevice chain is decoded from: VkBool32 count after the header and the member names (space separated;
// "" = print the index). A struct not listed shows up as `other_sType=`: its size is unknown, so it is not read.
struct FeatTable {
  VkStructureType type;
  const char* label;
  uint32_t bools;
  const char* names;
};
const FeatTable kFeatTable[] = {
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, "vulkan11", 12,
     "storageBuffer16BitAccess uniformAndStorageBuffer16BitAccess storagePushConstant16 storageInputOutput16 multiview "
     "multiviewGeometryShader multiviewTessellationShader variablePointersStorageBuffer variablePointers protectedMemory "
     "samplerYcbcrConversion shaderDrawParameters"},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, "vulkan12", 47,
     "samplerMirrorClampToEdge drawIndirectCount storageBuffer8BitAccess uniformAndStorageBuffer8BitAccess storagePushConstant8 "
     "shaderBufferInt64Atomics shaderSharedInt64Atomics shaderFloat16 shaderInt8 descriptorIndexing "
     "shaderInputAttachmentArrayDynamicIndexing shaderUniformTexelBufferArrayDynamicIndexing "
     "shaderStorageTexelBufferArrayDynamicIndexing shaderUniformBufferArrayNonUniformIndexing "
     "shaderSampledImageArrayNonUniformIndexing shaderStorageBufferArrayNonUniformIndexing "
     "shaderStorageImageArrayNonUniformIndexing shaderInputAttachmentArrayNonUniformIndexing "
     "shaderUniformTexelBufferArrayNonUniformIndexing shaderStorageTexelBufferArrayNonUniformIndexing "
     "descriptorBindingUniformBufferUpdateAfterBind descriptorBindingSampledImageUpdateAfterBind "
     "descriptorBindingStorageImageUpdateAfterBind descriptorBindingStorageBufferUpdateAfterBind "
     "descriptorBindingUniformTexelBufferUpdateAfterBind descriptorBindingStorageTexelBufferUpdateAfterBind "
     "descriptorBindingUpdateUnusedWhilePending descriptorBindingPartiallyBound descriptorBindingVariableDescriptorCount "
     "runtimeDescriptorArray samplerFilterMinmax scalarBlockLayout imagelessFramebuffer uniformBufferStandardLayout "
     "shaderSubgroupExtendedTypes separateDepthStencilLayouts hostQueryReset timelineSemaphore bufferDeviceAddress "
     "bufferDeviceAddressCaptureReplay bufferDeviceAddressMultiDevice vulkanMemoryModel vulkanMemoryModelDeviceScope "
     "vulkanMemoryModelAvailabilityVisibilityChains shaderOutputViewportIndex shaderOutputLayer subgroupBroadcastDynamicId"},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, "vulkan13", 15,
     "robustImageAccess inlineUniformBlock descriptorBindingInlineUniformBlockUpdateAfterBind pipelineCreationCacheControl "
     "privateData shaderDemoteToHelperInvocation shaderTerminateInvocation subgroupSizeControl computeFullSubgroups "
     "synchronization2 textureCompressionASTC_HDR shaderZeroInitializeWorkgroupMemory dynamicRendering "
     "shaderIntegerDotProduct maintenance4"},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES, "multiview", 3,
     "multiview multiviewGeometryShader multiviewTessellationShader"},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES, "descriptor_indexing", 20,
     "shaderInputAttachmentArrayDynamicIndexing shaderUniformTexelBufferArrayDynamicIndexing "
     "shaderStorageTexelBufferArrayDynamicIndexing shaderUniformBufferArrayNonUniformIndexing "
     "shaderSampledImageArrayNonUniformIndexing shaderStorageBufferArrayNonUniformIndexing "
     "shaderStorageImageArrayNonUniformIndexing shaderInputAttachmentArrayNonUniformIndexing "
     "shaderUniformTexelBufferArrayNonUniformIndexing shaderStorageTexelBufferArrayNonUniformIndexing "
     "descriptorBindingUniformBufferUpdateAfterBind descriptorBindingSampledImageUpdateAfterBind "
     "descriptorBindingStorageImageUpdateAfterBind descriptorBindingStorageBufferUpdateAfterBind "
     "descriptorBindingUniformTexelBufferUpdateAfterBind descriptorBindingStorageTexelBufferUpdateAfterBind "
     "descriptorBindingUpdateUnusedWhilePending descriptorBindingPartiallyBound descriptorBindingVariableDescriptorCount "
     "runtimeDescriptorArray"},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES, "atomic_int64", 2,
     "shaderBufferInt64Atomics shaderSharedInt64Atomics"},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT, "image_atomic_int64", 2,
     "shaderImageInt64Atomics sparseImageInt64Atomics"},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES, "buffer_device_address", 3,
     "bufferDeviceAddress bufferDeviceAddressCaptureReplay bufferDeviceAddressMultiDevice"},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT, "robustness2", 3,
     "robustBufferAccess2 robustImageAccess2 nullDescriptor"},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES, "float16_int8", 2, "shaderFloat16 shaderInt8"},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, "16bit_storage", 4, ""},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES, "8bit_storage", 3, ""},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES, "timeline_semaphore", 1, "timelineSemaphore"},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES, "sync2", 1, "synchronization2"},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES, "dynamic_rendering", 1, "dynamicRendering"},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES, "scalar_block_layout", 1, "scalarBlockLayout"},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES, "host_query_reset", 1, "hostQueryReset"},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES, "draw_parameters", 1, "shaderDrawParameters"},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT, "index_uint8", 1, "indexTypeUint8"},
};
const char* kCoreFeatureNames =
    "robustBufferAccess fullDrawIndexUint32 imageCubeArray independentBlend geometryShader tessellationShader sampleRateShading "
    "dualSrcBlend logicOp multiDrawIndirect drawIndirectFirstInstance depthClamp depthBiasClamp fillModeNonSolid depthBounds "
    "wideLines largePoints alphaToOne multiViewport samplerAnisotropy textureCompressionETC2 textureCompressionASTC_LDR "
    "textureCompressionBC occlusionQueryPrecise pipelineStatisticsQuery vertexPipelineStoresAndAtomics fragmentStoresAndAtomics "
    "shaderTessellationAndGeometryPointSize shaderImageGatherExtended shaderStorageImageExtendedFormats "
    "shaderStorageImageMultisample shaderStorageImageReadWithoutFormat shaderStorageImageWriteWithoutFormat "
    "shaderUniformBufferArrayDynamicIndexing shaderSampledImageArrayDynamicIndexing shaderStorageBufferArrayDynamicIndexing "
    "shaderStorageImageArrayDynamicIndexing shaderClipDistance shaderCullDistance shaderFloat64 shaderInt64 shaderInt16 "
    "shaderResourceResidency shaderResourceMinLod sparseBinding sparseResidencyBuffer sparseResidencyImage2D "
    "sparseResidencyImage3D sparseResidency2Samples sparseResidency4Samples sparseResidency8Samples sparseResidency16Samples "
    "sparseResidencyAliased variableMultisampleRate inheritedQueries";

std::string nameAt(const char* names, size_t index) {
  size_t i = 0;
  const char* p = names;
  while (*p) {
    const char* e = std::strchr(p, ' ');
    const size_t len = e ? size_t(e - p) : std::strlen(p);
    if (i++ == index) return std::string(p, len);
    if (!e) break;
    p = e + 1;
  }
  return "#" + std::to_string(index);
}

// "label on=a,b unsupported=c": the features set in `want` (bools of them) and, of those, the ones `have` does not report.
std::string featureLine(const char* label, const VkBool32* want, const VkBool32* have, size_t bools, const char* names) {
  std::string on, bad;
  for (size_t i = 0; i < bools; ++i) {
    if (!want[i]) continue;
    const std::string n = nameAt(names, i);
    on += (on.empty() ? "" : ",") + n;
    if (!have[i]) bad += (bad.empty() ? "" : ",") + n;
  }
  return std::string("vk.device_req features struct=") + label + " on=" + (on.empty() ? "none" : on) +
         " unsupported=" + (bad.empty() ? "none" : bad);
}
}  // namespace

std::vector<std::string> deviceRequestLines(VkPhysicalDevice pd, const VkDeviceCreateInfo* ci) {
  std::vector<std::string> out;
  std::string exts;
  for (uint32_t i = 0; i < ci->enabledExtensionCount; ++i) exts += (i ? "," : "") + std::string(ci->ppEnabledExtensionNames[i]);
  out.push_back("vk.device_req exts=" + std::to_string(ci->enabledExtensionCount) + " names=" + exts);
  VkPhysicalDeviceFeatures base{};
  vkGetPhysicalDeviceFeatures(pd, &base);
  if (ci->pEnabledFeatures)
    out.push_back(featureLine("features", reinterpret_cast<const VkBool32*>(ci->pEnabledFeatures),
                              reinterpret_cast<const VkBool32*>(&base), sizeof base / sizeof(VkBool32), kCoreFeatureNames));
  std::string other;
  const size_t head = sizeof(VkBaseOutStructure);
  for (auto* s = static_cast<const VkBaseInStructure*>(ci->pNext); s; s = s->pNext) {
    if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) {
      out.push_back(featureLine("features2", reinterpret_cast<const VkBool32*>(&reinterpret_cast<const VkPhysicalDeviceFeatures2*>(s)->features),
                                reinterpret_cast<const VkBool32*>(&base), sizeof base / sizeof(VkBool32), kCoreFeatureNames));
      continue;
    }
    const FeatTable* t = nullptr;
    for (const FeatTable& f : kFeatTable)
      if (f.type == s->sType) t = &f;
    if (!t) {
      other += (other.empty() ? "" : ",") + std::to_string(unsigned(s->sType));
      continue;
    }
    alignas(8) uint8_t have[256] = {};  // what the driver reports for this struct (all zero when it does not know it)
    reinterpret_cast<VkBaseOutStructure*>(have)->sType = t->type;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = have;
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    out.push_back(featureLine(t->label, reinterpret_cast<const VkBool32*>(reinterpret_cast<const uint8_t*>(s) + head),
                              reinterpret_cast<const VkBool32*>(have + head), t->bools, t->names));
  }
  if (!other.empty()) out.push_back("vk.device_req features other_sType_not_decoded=" + other);
  return out;
}

// ---- vk.tex_hist ----
namespace {
std::mutex& imgRegMutex() { static auto* m = new std::mutex; return *m; }  // leaked on purpose, like liveDeviceMutex
std::unordered_set<const Image*>& imgReg() { static auto* s = new std::unordered_set<const Image*>; return *s; }
// Sampled images that are not attachments (the population of vk.tex_hist): created / destroyed so far, under imgRegMutex.
uint64_t g_texNew = 0, g_texGone = 0;
bool texHistCounts(const Image* im) {
  return (im->info.usage & VK_IMAGE_USAGE_SAMPLED_BIT) &&
         !(im->info.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT));
}
void imgRegAdd(const Image* im) { std::lock_guard<std::mutex> l(imgRegMutex()); imgReg().insert(im); g_texNew += texHistCounts(im); }
void imgRegRemove(const Image* im) { std::lock_guard<std::mutex> l(imgRegMutex()); imgReg().erase(im); g_texGone += texHistCounts(im); }
bool isAstcFormat(VkFormat f) {
  return (f >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK) ||
         (f >= VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK_EXT && f <= VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK_EXT);
}
}  // namespace

void noteMipsWritten(const Image* im, uint32_t mip) {
  if (im && mip < 32) im->mipsWritten.fetch_or(1u << mip, std::memory_order_relaxed);
}

void texHistAdd(TexHistAcc& a, const Image& im) {
  if (!(im.info.usage & VK_IMAGE_USAGE_SAMPLED_BIT)) return;
  const uint64_t bytes = imageBytes(im.desc);
  if (im.info.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
    ++a.rtN, a.rtBytes += bytes;
    return;
  }
  const uint32_t mask = im.mipsWritten.load(std::memory_order_relaxed);
  const int written = __builtin_popcount(mask);
  auto& b = a.b[written == 0 ? 0 : written <= 2 ? 1 : written <= 5 ? 2 : 3];
  ++a.sampled, ++b.n, b.bytes += bytes;
  const uint32_t side = std::max(im.info.extent.width, im.info.extent.height);
  ++a.top[side <= 64 ? 0 : side <= 128 ? 1 : side <= 256 ? 2 : side <= 512 ? 3 : side <= 1024 ? 4 : 5];
  if (isAstcFormat(im.info.format)) ++b.astcN, b.astcBytes += bytes;
  if (written > 0 && uint32_t(written) < im.info.mipLevels) ++a.partial;
  if (written > 0 && !(mask & 1u)) ++a.top0Missing;  // some mips came in but the base one never did: the streamer has not delivered the top
}

std::string texHistFormat(const TexHistAcc& a) {
  static const char* kName[4] = {"m0", "m1_2", "m3_5", "m6p"};
  char buf[640];
  std::string s = "sampled=" + std::to_string(a.sampled);
  for (int i = 0; i < 4; ++i) {
    std::snprintf(buf, sizeof buf, " | %s n=%llu mib=%.1f astc_n=%llu astc_mib=%.1f", kName[i], (unsigned long long)a.b[i].n,
                  double(a.b[i].bytes) / 1048576.0, (unsigned long long)a.b[i].astcN, double(a.b[i].astcBytes) / 1048576.0);
    s += buf;
  }
  std::snprintf(buf, sizeof buf, " | partial=%llu top0_unwritten=%llu | rt n=%llu mib=%.1f", (unsigned long long)a.partial,
                (unsigned long long)a.top0Missing, (unsigned long long)a.rtN, double(a.rtBytes) / 1048576.0);
  s += buf;
  // The same images by the larger side of their level 0: a streamer that never brings the large levels keeps them in 64.
  std::snprintf(buf, sizeof buf, " | top{64=%llu,128=%llu,256=%llu,512=%llu,1024=%llu,2048p=%llu}", (unsigned long long)a.top[0],
                (unsigned long long)a.top[1], (unsigned long long)a.top[2], (unsigned long long)a.top[3], (unsigned long long)a.top[4],
                (unsigned long long)a.top[5]);
  return s + buf;
}

// Every 5 s from the submit tick (psoCensus's place): the residency of sampled images by how many mips got a copy recorded (m0 =
// none: render-target-like or not streamed yet), ASTC apart, so a streamer that keeps lacking mips shows as a flat line. Printed
// when the numbers changed, at most 240 lines. "Written" = a copy/blit into that mip was RECORDED (not necessarily executed yet).
// new= / gone= : the sampled non-attachment images created / destroyed since the previous line (a streamer swapping textures
// in and out moves them even when the histogram stays the same; the image vkGetDeviceImageMemoryRequirements makes to answer
// counts once in both).
void texHistTick(bool force) {
  static std::atomic<int64_t> nextNs{0};
  const int64_t nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
  if (!force && nowNs < nextNs.load(std::memory_order_relaxed)) return;
  nextNs.store(nowNs + 5'000'000'000LL, std::memory_order_relaxed);
  // Every device calls this from its own submit thread (its own submitMutex): the statics below need a lock of their own.
  static std::mutex tickMutex;
  std::lock_guard<std::mutex> tick(tickMutex);
  static std::string last;
  static unsigned lines = 0;
  static uint64_t lastNew = 0, lastGone = 0;
  static const int64_t startNs = nowNs;
  TexHistAcc a;
  uint64_t made, gone;
  {
    std::lock_guard<std::mutex> l(imgRegMutex());
    for (const Image* im : imgReg()) texHistAdd(a, *im);
    made = g_texNew, gone = g_texGone;
  }
  const std::string body = texHistFormat(a);
  if ((body == last && made == lastNew && gone == lastGone) || lines >= 240) return;
  logf("vk.tex_hist t=%llus %s | new=%llu gone=%llu", (unsigned long long)((nowNs - startNs) / 1'000'000'000LL), body.c_str(),
       (unsigned long long)(made - lastNew), (unsigned long long)(gone - lastGone));
  last = body, ++lines, lastNew = made, lastGone = gone;
}
}  // namespace vpvk

// ---------------------------------------------------------------------------------------------
// Device and queue
// ---------------------------------------------------------------------------------------------
namespace {
// Is every feature `ci` enables one this driver reports? `bad` lists the ones that are not, as
// "<struct>[<VkBool32 index>]". Only VPVK_STRICT_FEATURES=1 (conformance) turns that into
// VK_ERROR_FEATURE_NOT_PRESENT: games built for the Quest sometimes enable features they never use
// without asking, and refusing their device would stop them.
bool featuresSupported(VkPhysicalDevice pd, const VkDeviceCreateInfo* ci, std::string& bad) {
  bool ok = true;
  auto subset = [&](const std::string& what, const void* want, const void* have, size_t bytes) {
    const auto* w = static_cast<const VkBool32*>(want);
    const auto* h = static_cast<const VkBool32*>(have);
    for (size_t i = 0; i < bytes / sizeof(VkBool32); ++i)
      if (w[i] && !h[i]) bad += " " + what + "[" + std::to_string(i) + "]", ok = false;
  };
  VkPhysicalDeviceFeatures base{};
  vkGetPhysicalDeviceFeatures(pd, &base);
  if (ci->pEnabledFeatures) subset("features", ci->pEnabledFeatures, &base, sizeof base);
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
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES, 1},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES, 1},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES, 1},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES, 1},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES, 2},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT, 1},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES, 3},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES, 2},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES, 3},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES, 3},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES, 1},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES, 2},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT, 1},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT, 3},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT, 31},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES_KHR, 1},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT, 1},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES, 1},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES, 1},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES, 1},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES, 1},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES, 1},
      // The VulkanNN structs hold exactly this many VkBool32 (sizeof also counts tail padding).
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, 12},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, 47},
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, 15},
  };
  const size_t head = sizeof(VkBaseOutStructure);
  for (auto* s = static_cast<const VkBaseInStructure*>(ci->pNext); s; s = s->pNext) {
    if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) {
      subset("features2", &reinterpret_cast<const VkPhysicalDeviceFeatures2*>(s)->features, &base, sizeof base);
      continue;
    }
    for (const auto& [type, bools] : kStructs) {
      if (s->sType != type) continue;
      alignas(8) uint8_t have[256] = {};
      reinterpret_cast<VkBaseOutStructure*>(have)->sType = type;
      VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
      f2.pNext = have;
      vkGetPhysicalDeviceFeatures2(pd, &f2);
      subset("sType" + std::to_string(type), reinterpret_cast<const uint8_t*>(s) + head, have + head, bools * sizeof(VkBool32));
    }
  }
  return ok;
}

// robustBufferAccess as the app asked for it: pEnabledFeatures, or the features of a VkPhysicalDeviceFeatures2 in pNext.
bool requestedRobustBufferAccess(const VkDeviceCreateInfo* ci) {
  if (ci->pEnabledFeatures && ci->pEnabledFeatures->robustBufferAccess) return true;
  for (auto* s = static_cast<const VkBaseInStructure*>(ci->pNext); s; s = s->pNext)
    if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 &&
        reinterpret_cast<const VkPhysicalDeviceFeatures2*>(s)->features.robustBufferAccess)
      return true;
  return false;
}
}  // namespace

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice pd,
                                                          const VkDeviceCreateInfo* ci,
                                                          const VkAllocationCallbacks*,
                                                          VkDevice* out) {
  const auto have = deviceExtensions();
  if (!internalClient()) {  // vk.device_req: what the game asked, once per distinct request (a retry with the same one is not repeated)
    static std::mutex reqMutex;
    static std::set<std::string> reqSeen;
    std::string all;
    const std::vector<std::string> lines = deviceRequestLines(pd, ci);
    for (const std::string& l : lines) all += l + "\n";
    bool first;
    {
      std::lock_guard<std::mutex> lock(reqMutex);
      first = reqSeen.insert(all).second;
    }
    if (first)
      for (const std::string& l : lines) log(l);
  }
  auto* d = new Device;
  d->physical = from<PhysicalDevice>(pd);
  bool wantIndexU8 = false;
  for (uint32_t i = 0; i < ci->enabledExtensionCount; ++i) {
    const char* e = ci->ppEnabledExtensionNames[i];
    // A game that enables the density map without checking (it is not advertised) is let through
    // with it ignored: refusing the device would stop the game for a feature it can live without.
    if (!std::strcmp(e, VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME) ||
        !std::strcmp(e, "VK_EXT_fragment_density_map2")) {
      logf("vk.fov ignored=device_ext name=%s", e);
      continue;
    }
    if (!hasExt(have, e) && !hasExt(hiddenDeviceExtensions(), e)) {
      logf("vk.err what=device_ext name=%s", e);
      delete d;
      return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    if (!std::strcmp(e, VK_KHR_MULTIVIEW_EXTENSION_NAME)) d->multiview = true;
    if (!std::strcmp(e, VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME)) wantIndexU8 = true;
    if (!std::strcmp(e, VK_EXT_METAL_OBJECTS_EXTENSION_NAME)) d->metalObjects = true;
  }
  if (extraExtensions()) {
    std::string bad;
    if (!featuresSupported(pd, ci, bad)) {
      const char* strict = std::getenv("VPVK_STRICT_FEATURES");
      if (strict && *strict == '1') {
        logf("vk.err what=device_feature_not_present features=%s", bad.c_str());
        delete d;
        return VK_ERROR_FEATURE_NOT_PRESENT;
      }
      static std::mutex warnedMutex;
      static std::set<std::string> warned;  // once per distinct feature string
      bool first;
      {
        std::lock_guard<std::mutex> lock(warnedMutex);
        first = warned.insert(bad).second;
      }
      if (first) logf("vk.warn what=device_feature_not_present features=%s (continuing)", bad.c_str());
    }
  }
  refreshRobustMode();  // VPVK_ROBUST as the host set it for this run (the engine may have been used before)
  if (!internalClient()) {  // the game's request decides VPVK_ROBUST=app; the internal GLES device is not the game
    setAppRobustBufferAccess(requestedRobustBufferAccess(ci));
    logf("vk.robust app=%d mode=%s effective=%d", appRobustBufferAccess(), robustMode(), robustBuffersEffective() ? 1 : 0);
  }
  if (aotEnabled()) {
    // AOT (VPVK_AOT=1): the binary archive built in preparation is loaded by createPlatform(),
    // before the first frame. A pipeline missing from it (a variant never seen) compiles on the
    // workers without holding the frame (VP_HOLD_MS=0 unless the app set it): its draws are skipped
    // for the few frames it takes, and the backend archives it for the next run. No ubershader:
    // Vulkan pipelines are arbitrary game SPIR-V, so there is no fixed-function state to interpret
    // generically as in Dolphin; an "uber" variant would be a second full translation per shader.
    if (!std::getenv("VP_HOLD_MS")) setenv("VP_HOLD_MS", "0", 1);
    std::string manifest;
    if (FILE* f = std::fopen((RecipeStore::defaultDir() + "/aot.manifest").c_str(), "rb")) {
      char buf[256] = {0};
      manifest.assign(buf, std::fread(buf, 1, sizeof buf - 1, f));
      std::fclose(f);
      for (char& c : manifest)
        if (c == '\n') c = ' ';
    }
    logf("gpu.aot device manifest=\"%s\" hold_ms=%s", manifest.empty() ? "none" : manifest.c_str(),
         std::getenv("VP_HOLD_MS"));
  }
  d->platform = createPlatform();
  // Pipeline compiles run on background workers (never the submit thread); leave the game's
  // render/main threads two cores.
  const unsigned workers = std::clamp(std::thread::hardware_concurrency(), 4u, 8u) - 2u;
  d->gpu = std::make_unique<vpgpu::Device>(d->platform->backend(), workers, [](const std::string& s) { log(s); });
  for (uint32_t i = 0; i < kQueueCount; ++i) d->queues[i].device = d, d->queues[i].index = i;
  requestIndirectPatch(d);  // engine-internal kernel of GPU indirect draws: compiling (Urgent) before the game's first frame
  if (wantIndexU8) requestIndexU8(d);  // the 8-bit index widening kernel too: no draw waits for its first compile
  // Pipeline recipes of earlier runs: compiled now, before the game's first frame, so its
  // vkCreate*Pipelines calls are cache hits. With vpvk_prewarm() run by the app beforehand this is
  // mostly binary-archive lookups.
  refreshTranslatorMode();  // VPVK_AMPLIFY / VPVK_FETCH_NATIVE as the host set them for this device
  d->recipes = RecipeStore::forDir(RecipeStore::defaultDir());  // one store per directory: every device appends to the one file
  d->created = std::chrono::steady_clock::now();
  d->internalClient = internalClient();
  const bool doPrewarm = d->recipes->enabled() && prewarmEnabled();
  d->recipesLoaded = uint32_t(d->recipes->pipelineCount());
  if (doPrewarm) d->prewarmedRecipes = d->recipesLoaded;
  // vk.pso.census prewarmed_unused / requested_not_prewarmed: the recipe keys once the prewarm's refresh() has rekeyed them
  // (a changed translator gives other MSL ids, so other keys); read earlier they would not match anything the game asks for.
  const auto noteRecipeKeys = [dev = d] { dev->recipeUse.setPrewarmed(dev->recipes->keys()); };
  if (asyncPipelinesOn()) {
    // vkCreate*Pipelines returns before translating (pipelines.cpp "compile policy"); the frame gate replaces the
    // core's frame hold, and the prewarm runs in the background (it used to block this call for up to 20 s).
    // Translation threads: the compiles already use `workers`; these are the SPIR-V -> MSL jobs of the game's burst.
    d->translator = std::make_unique<TranslatePool>(std::clamp(std::thread::hardware_concurrency(), 3u, 6u) - 1u);
    if (!std::getenv("VP_HOLD_MS")) d->gpu->setHoldMs(0);
    if (doPrewarm) {
      Device* dev = d;
      d->prewarmThread = std::thread([dev, noteRecipeKeys] {
#if defined(__APPLE__)
        pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
#endif
        PrewarmHooks hooks;
        hooks.cancel = &dev->prewarmCancel;
        hooks.refreshThreads = 1;
        hooks.afterRefresh = noteRecipeKeys;
        hooks.pollMs = 250;  // the game runs meanwhile: one status() lock per recipe per poll is not free
        hooks.addShader = [dev](uint64_t id, ShaderBlob blob) {
          std::lock_guard<std::mutex> lock(dev->shaderMutex);
          if (!dev->shadersAdded.insert(id).second) return;
          dev->platform->addShader(id, std::move(blob));
        };
        const PrewarmResult r = prewarm(
            *dev->platform, dev->gpu->pipelines(), *dev->recipes,
            [dev](uint32_t done, uint32_t) { dev->recipesPrewarmed = done; }, 0.0, hooks);  // live count, then the exact one
        dev->recipesPrewarmed = r.ready;
        logf("vk.prewarm background done");
      });
    }
  } else if (doPrewarm) {
    // Bounded (8 s from the prewarm's start, recipes and forced-MSAA variants alike): vkCreateDevice must not sit through a whole
    // cold prewarm; what is left (variants included) keeps compiling on the workers.
    PrewarmHooks hooks;
    hooks.afterRefresh = noteRecipeKeys;
    d->recipesPrewarmed = prewarm(*d->platform, d->gpu->pipelines(), *d->recipes, nullptr, 8.0, hooks).ready;
  }
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
  // Contract: the host calls that find the device through the registry (vpvk_native_submit, liveDeviceSubmitLocked) may race this call:
  // they are pinned in hostUsers, get the submit mutex first, and whoever holds it (a vpvk_native_submit until its command buffer is
  // committed) is waited for here. Calls that take the device from their own handle (vkQueueSubmit through the queue, submitDeferred
  // through the device handle it is given) are NOT pinned and nothing here waits for them: parked for `d` (parkWhileBackground(d)) they are
  // woken by the release below and go on to take d->submitMutex, so one that is still running when `d` is deleted uses freed memory.
  // Vulkan requires the app to stop using the device's queues and semaphores before vkDestroyDevice (external synchronisation):
  // calling them concurrently with it is a misuse, not something this function defends against.
  vpgpu::backgroundGate().release(d);  // host threads parked for this device (app in the background) go on and find no device (registry
                                       // calls) / only wake instead of hanging (a misusing handle call, see above)
  // In the background a thread can be parked INSIDE the backend (a commit under the submit mutex, owner = the backend, not `d`), and
  // the host calls pinned in hostUsers go on to park there once they have the mutex: this thread would wait for the app to come
  // back, which it may be the one to signal. So whoever is still to get the mutex is released again while it is waited for.
  auto wakeParked = [&] { if (vpgpu::backgroundGate().active()) d->platform->releaseBackground(); };
  while (d->hostUsers.load() > 0) {
    wakeParked();
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  auto lockSubmit = [&] {
    // Never a blocking lock: whoever holds the mutex can park in the backend if the app goes to the background while this waits (the gate
    // changes on the host's thread, whatever thread is destroying), and a blocked lock is not woken by that. So poll, waking whatever
    // parks in the backend (wakeParked looks at the gate each turn). Cold path; uncontended it is one try_lock.
    std::unique_lock<std::mutex> l(d->submitMutex, std::try_to_lock);
    while (!l.owns_lock()) {
      wakeParked();
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      static_cast<void>(l.try_lock());
    }
    return l;
  };
  {
    std::unique_lock<std::mutex> l = lockSubmit();
    drainRetiredChunks(d);  // (arena chunks a command buffer reset gave up)
  }
  d->prewarmCancel = true;
  if (d->prewarmThread.joinable()) d->prewarmThread.join();
  joinCacheFlush(d);
  pipelineSummary(d, true);
  psoCensus(d, true);
  d->platform->waitIdle(UINT64_MAX);  // bounded by a lost device (Platform::waitIdle)
  releaseVisibility(d);
  {
    std::unique_lock<std::mutex> l = lockSubmit();  // uiMask lo usa vpvk_taa_native_encode bajo este mutex
    d->uiMask.release();
    d->msaaTwins.release(d);
    if (d->zeroBuffer) d->gpu->destroy(d->zeroBuffer);
    if (d->zeroAstcBuffer) d->gpu->destroy(d->zeroAstcBuffer);
  }
  d->translator.reset();  // after every pipeline job it could still hold ran or was dropped (jobs touch d->gpu)
  d->gpu.reset();
  delete d;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetDeviceQueue(VkDevice dv, uint32_t family, uint32_t index,
                                                        VkQueue* out) {
  Device* d = from<Device>(dv);
  d->queues[std::min(index, kQueueCount - 1)].family = family;
  *out = to<VkQueue>(&d->queues[std::min(index, kQueueCount - 1)]);
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
// `queueMask`: bit i = VkQueue i (vkQueueWaitIdle waits for its own queue's held batches, vkDeviceWaitIdle for all).
static VkResult waitDeviceIdle(Device* d, uint32_t queueMask = ~0u) {
  PerfWait perfWait(perf().idleWaits, perf().idleUs, perf().idleUsMax);  // vk.perf idle_wait
  // Batches parked on a timeline value nobody will signal would hang this forever
  // (an application bug, but a hang is worse than a log line): bounded at 2 s.
  const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  for (;;) {
    {
      std::lock_guard<std::mutex> lock(d->submitMutex);
      bool held = false;
      for (const DeferredBatch& b : d->deferred) held = held || (queueMask >> b.queue & 1);
      if (!held) break;
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

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkQueueWaitIdle(VkQueue q) {
  return waitDeviceIdle(from<Queue>(q)->device, 1u << from<Queue>(q)->index);
}

// ---------------------------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------------------------
namespace {
// Host memory of a VkDeviceMemory. From kHostMapBytes up it is a mapping of its own rather than a malloc block: pages nobody
// writes (the memory of an image, whose texels live in the texture) are never resident whatever the allocator does with a
// reused block, they go back to the system at vkFreeMemory, and on Apple the region map names them (VM tag 241, "[hb] vm
// tag241/...": resident = staging and buffer contents).
constexpr uint64_t kHostMapBytes = 1u << 20;
uint8_t* hostAlloc(uint64_t bytes) {
#ifdef VPVK_HOST_MMAP
  if (bytes >= kHostMapBytes) {
#ifdef __APPLE__
    const int tag = int(241u << 24);  // VM_MAKE_TAG: Darwin reads an anonymous mapping's tag from the fd argument
#else
    const int tag = -1;
#endif
    // kPage-aligned whatever the system's page size (4 KiB hosts): map one kPage more and give the ends back.
    void* p = mmap(nullptr, bytes + kPage, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, tag, 0);
    if (p == MAP_FAILED) return nullptr;
    uint8_t* raw = static_cast<uint8_t*>(p);
    uint8_t* out = raw + (kPage - reinterpret_cast<uintptr_t>(raw) % kPage) % kPage;
    if (out > raw) munmap(raw, size_t(out - raw));
    if (const size_t tail = size_t((raw + bytes + kPage) - (out + bytes))) munmap(out + bytes, tail);
    return out;
  }
#endif
  return static_cast<uint8_t*>(std::aligned_alloc(kPage, bytes));
}
void hostFree(uint8_t* p, uint64_t bytes) {
  if (!p) return;
#ifdef VPVK_HOST_MMAP
  if (bytes >= kHostMapBytes) {
    munmap(p, bytes);
    return;
  }
#endif
  (void)bytes;
  std::free(p);
}
}  // namespace

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkAllocateMemory(VkDevice, const VkMemoryAllocateInfo* ai,
                                                            const VkAllocationCallbacks*,
                                                            VkDeviceMemory* out) {
  vpgpu::HitchSpan hitch(vpgpu::hitchPerf().allocUs);  // vk.hitch alloc_ms
  if (ai->memoryTypeIndex >= memoryProperties().memoryTypeCount) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
  auto* m = new DeviceMemory;
  m->size = ai->allocationSize;
  m->typeIndex = ai->memoryTypeIndex;
  // Lazily-allocated memory (transient attachments only, vkGetImageMemoryRequirements): like the
  // Adreno's on-chip GMEM it takes no RAM. The texture bound to it lives in memoryless tile
  // storage (frame compiler TileLocal) or, when it must be loaded/stored, gets its own Metal memory.
  m->lazy = (memoryProperties().memoryTypes[m->typeIndex].propertyFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) != 0;
  {
    const unsigned t = std::min<unsigned>(m->typeIndex, diag::kTypes - 1);
    diag::add(diag::c().allocN[t]);
    diag::add(diag::c().allocB[t], m->size);
  }
  if (m->lazy) {
    *out = to<VkDeviceMemory>(m);
    return VK_SUCCESS;
  }
  // Page-aligned host memory for every other type: GPU buffers alias it in place (no staging).
  m->host = hostAlloc(alignUp(std::max<uint64_t>(m->size, 1), kPage));
  if (!m->host) {
    delete m;
    logf("vk.err what=alloc size=%llu", static_cast<unsigned long long>(ai->allocationSize));
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
  }
  if (memoryProperties().memoryTypes[m->typeIndex].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
    diag::rangeAdd(m->host, alignUp(std::max<uint64_t>(m->size, 1), kPage));
  allocatedBytes() += m->size;
  *out = to<VkDeviceMemory>(m);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice dv, VkDeviceMemory mem,
                                                    const VkAllocationCallbacks*) {
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  if (!mem) return;
  DeviceMemory* m = from<DeviceMemory>(mem);
  {
    const unsigned t = std::min<unsigned>(m->typeIndex, diag::kTypes - 1);
    diag::add(diag::c().freeN[t]);
    diag::add(diag::c().freeB[t], m->size);
  }
  if (!m->images.empty() && dv) {  // images still bound (legal: they must not be used again) forget it
    std::lock_guard<std::mutex> lock(from<Device>(dv)->submitMutex);
    for (Image* i : m->images) i->memory = nullptr;
  }
  diag::rangeDel(m->host);
  if (m->host) allocatedBytes() -= m->size;  // counted only for what vkAllocateMemory allocated (not lazy memory)
  hostFree(m->host, alignUp(std::max<uint64_t>(m->size, 1), kPage));
  delete m;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkMapMemory(VkDevice, VkDeviceMemory mem, VkDeviceSize off,
                                                       VkDeviceSize size, VkMemoryMapFlags, void** out) {
  DeviceMemory* m = from<DeviceMemory>(mem);
  diag::add(diag::c().mapN);
  diag::add(diag::c().mapB, size == VK_WHOLE_SIZE ? (m->size > off ? m->size - off : 0) : size);
  if (!(memoryProperties().memoryTypes[m->typeIndex].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
    return VK_ERROR_MEMORY_MAP_FAILED;
  m->mapped = true;
  *out = m->host + off;
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkUnmapMemory(VkDevice, VkDeviceMemory mem) {
  diag::add(diag::c().unmapN);
  from<DeviceMemory>(mem)->mapped = false;
}

// Coherent unified memory: nothing to flush or invalidate.
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkFlushMappedMemoryRanges(VkDevice, uint32_t n,
                                                                     const VkMappedMemoryRange* r) {
  diag::add(diag::c().flushN, n);
  for (uint32_t i = 0; r && i < n; ++i)
    if (r[i].size != VK_WHOLE_SIZE) diag::add(diag::c().flushB, r[i].size);
  return VK_SUCCESS;
}
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkInvalidateMappedMemoryRanges(VkDevice, uint32_t n,
                                                                          const VkMappedMemoryRange* r) {
  diag::add(diag::c().invN, n);
  for (uint32_t i = 0; r && i < n; ++i)
    if (r[i].size != VK_WHOLE_SIZE) diag::add(diag::c().invB, r[i].size);
  return VK_SUCCESS;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetDeviceMemoryCommitment(VkDevice, VkDeviceMemory m,
                                                                   VkDeviceSize* out) {
  const DeviceMemory* dm = from<DeviceMemory>(m);
  *out = dm->lazy ? 0 : dm->size;  // lazily-allocated: no RAM committed (tile memory)
}

// ---------------------------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------------------------
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(VkDevice, const VkBufferCreateInfo* ci,
                                                          const VkAllocationCallbacks*, VkBuffer* out) {
  vpgpu::HitchSpan hitch(vpgpu::hitchPerf().allocUs);  // vk.hitch alloc_ms
  auto* b = new Buffer;
  b->size = ci->size;
  b->usage = ci->usage;
  diag::add(diag::c().bufCreateN);
  diag::add(diag::c().bufCreateB, ci->size);
  *out = to<VkBuffer>(b);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyBuffer(VkDevice dv, VkBuffer buf,
                                                       const VkAllocationCallbacks*) {
  if (!buf) return;
  Buffer* b = from<Buffer>(buf);
  diag::add(diag::c().bufDestroyN);
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
  if (!m || !m->host) return VK_ERROR_OUT_OF_DEVICE_MEMORY;  // lazily-allocated memory holds no buffers
  b->memory = m;
  b->offset = off;
  diag::add(diag::c().bufBindN);
  diag::add(diag::c().bufBindB, b->size);
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
  vpgpu::HitchSpan hitch(vpgpu::hitchPerf().allocUs);  // vk.hitch alloc_ms
  // VK_EXT_separate_stencil_usage: one Metal texture holds both aspects, so it gets both usages.
  VkImageCreateInfo merged = *ci;
  if (nativeExtras())
    if (const auto* su = static_cast<const VkImageStencilUsageCreateInfo*>(
            findNext(ci->pNext, VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO))) {
      merged.usage |= su->stencilUsage;
      ci = &merged;
    }
  // Multi-planar format: plane 0 is this image (as its single-plane format), the others images of
  // their own (Image::planes). Only 2D, single-sample, optimal-tiling images exist.
  diag::add(diag::c().imgCreateN);
  diag::add(diag::c().imgCreatePx, uint64_t{ci->extent.width} * ci->extent.height * std::max(1u, ci->extent.depth) *
                                       std::max(1u, ci->arrayLayers));
  const MultiPlane& mp = multiPlane(ci->format);
  VkImageCreateInfo plane0 = *ci;
  if (mp.planes) {
    if (ci->imageType != VK_IMAGE_TYPE_2D || ci->samples != VK_SAMPLE_COUNT_1_BIT ||
        (ci->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) || ci->extent.width < 2 || ci->extent.height < 2 ||
        (ci->tiling == VK_IMAGE_TILING_LINEAR &&
         (ci->mipLevels != 1 || ci->arrayLayers != 1 ||
          (ci->usage & ~(VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)))))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    plane0.format = mp.plane[0];
  }
  vpgpu::TextureDesc desc = imageDesc(plane0);
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
  // Every ATTACHMENT (colour or depth/stencil, any sample count, whatever else its usage says) is memoryless first, as
  // on the Adreno's GMEM: not persistent, so the backend gives it no memory until a frame cannot keep it on tile (a
  // load, a sample, a copy, a stored contents: msaa_materialized names the frame compiler's verdict) and takes that
  // memory back after VPVK_MSAA_RELEASE_FRAMES tile-local frames (msaa_released). A client that keeps the MTLTexture
  // (vkGetMTLTextureMVK, VPKTAA's depth, a host-marked eye image) pins it in memory. Contents stay correct across
  // frames: a tile-local frame neither loads nor stores them (Vulkan leaves them undefined there), and a frame that
  // stores them materialises the memory first. A "Too much geometry for memoryless" tiler flush bans the memoryless
  // stand-in for the textures of that command buffer only (backend tileBanned), not for the whole process.
  // Multi-planar images keep their persistent planes. VPVK_MSAA_TILE=0 restores fully resident attachments.
  const bool attachment =
      (ci->usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) != 0 && !mp.planes;
  // Storage writes may go through view ids the frame compiler does not track as writes of this texture
  // (ResourceTable::memValid), so such images keep their memory. Attachments (also of mutable-format
  // images) always name the base texture, so formatView alone keeps the image memoryless.
  if (attachment && desc.persistent && msaaTileOn() && ci->tiling == VK_IMAGE_TILING_OPTIMAL &&
      !(desc.usage & vpgpu::usage::storage)) {
    desc.persistent = false;
    static std::atomic<unsigned> k{0};
    if (k.fetch_add(1) < 64)
      logf("vk.msaa_tile memoryless image=%ux%ux%u samples=%u format=%d usage=%#x (%.1f MiB if resident)",
           ci->extent.width, ci->extent.height, ci->arrayLayers, unsigned(desc.samples), ci->format,
           unsigned(ci->usage), double(imageBytes(desc)) / 1048576.0);
  }
  // Texture budget: a plain sampled texture above the cap is created without its largest levels (Image::mipDrop).
  const uint32_t mipDrop = textureBudgetDrop(*ci, desc.format, textureBudget());
  if (mipDrop) {
    const uint64_t full = imageBytes(desc);
    desc.width = std::max(1u, desc.width >> mipDrop), desc.height = std::max(1u, desc.height >> mipDrop);
    desc.mips = static_cast<uint16_t>(std::max<uint32_t>(1, ci->mipLevels > mipDrop ? ci->mipLevels - mipDrop : 1));
    const uint64_t saved = full - std::min(full, imageBytes(desc));
    const uint64_t n = g_budgetImages.fetch_add(1, std::memory_order_relaxed) + 1;
    const uint64_t total = g_budgetSavedBytes.fetch_add(saved, std::memory_order_relaxed) + saved;
    if (n <= 16 || n % 64 == 0)
      logf("vk.tex_budget cap=%u image=%ux%u mips=%u layers=%u format=%d -> %ux%u mips=%u (%.1f MiB less; %llu images, %.0f MiB so far)",
           textureBudget(), ci->extent.width, ci->extent.height, ci->mipLevels, ci->arrayLayers, ci->format, desc.width, desc.height,
           unsigned(desc.mips), double(saved) / 1048576.0, static_cast<unsigned long long>(n), double(total) / 1048576.0);
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
  img->mipDrop = static_cast<uint8_t>(mipDrop);
  img->info.pQueueFamilyIndices = nullptr;
  img->desc = std::move(desc);
  for (uint32_t k = 1; k < mp.planes; ++k) {
    auto p = std::make_unique<Image>();
    p->info = *ci;
    p->info.pNext = nullptr;
    p->info.pQueueFamilyIndices = nullptr;
    p->info.format = mp.plane[k];
    const VkExtent2D e = planeExtent(mp, k, ci->extent.width, ci->extent.height);
    p->info.extent = {e.width, e.height, 1};
    p->desc = imageDesc(p->info);
    img->planes.push_back(std::move(p));
  }
  // Linear colour images are host-visible staging (the CTS reads every result image through one):
  // their memory must hold the texels (see Image::linear). Depth/stencil linear images are refused
  // by the format features, multisampled ones by the image format properties.
  {
    const vpgpu::FormatInfo& fi = vpgpu::formatInfo(img->desc.format);
    img->linear = ci->tiling == VK_IMAGE_TILING_LINEAR && !fi.depth() && !fi.stencil() && img->desc.samples == 1;
    for (auto& pl : img->planes) pl->linear = img->linear;  // planes of a linear multi-planar image are linear as well
  }
  *out = to<VkImage>(img);
  imgRegAdd(img);
  const bool linearTiling = ci->tiling == VK_IMAGE_TILING_LINEAR;
  if (linearTiling) img->desc.label = "vk.linear";  // gpu.bigframe tiling (core knows no tiling); labels don't affect sharing
  diag::add(linearTiling ? diag::c().imgLinearN : diag::c().imgOptimalN);
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
  if (i.id && i.idOwners && i.idOwners.use_count() != 1) diag::add(diag::c().imgRelSharedN);
  if (i.id && (!i.idOwners || i.idOwners.use_count() == 1)) {
    diag::add(diag::c().imgRelDestroyN);
    const auto zit = d.zeroPending.find(uint64_t{i.id.generation} << 32 | i.id.index);
    if (zit != d.zeroPending.end()) {
      zeroForgetNative(d, zit->second);
      d.zeroPending.erase(zit);
    }
    {
      std::lock_guard<std::mutex> l(d.eyeMutex);
      d.eyeTextures.erase(uint64_t{i.id.generation} << 32 | i.id.index);
    }
    d.gpu->destroy(i.id);
  }
  if (i.hostAlias) d.gpu->destroy(i.hostAlias);
  i.id = {};
  i.hostAlias = {};
  i.native = nullptr;
  i.idOwners.reset();
}
}  // namespace

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyImage(VkDevice dv, VkImage im,
                                                      const VkAllocationCallbacks*) {
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  if (!im) return;
  Image* i = from<Image>(im);
  diag::add(diag::c().imgDestroyN);
  {
    // The backend's texture map is read by vkQueueSubmit on another thread and its retire list is
    // moved by the next frame slot: destroying outside the submit mutex could lose the retire (the
    // MTLTexture released while in flight) or corrupt the map. Released when its slot is reused.
    Device* d = from<Device>(dv);
    std::lock_guard<std::mutex> lock(d->submitMutex);
    if (!i->id) diag::add(diag::c().imgRelNoTexN);
    for (auto& p : i->planes) releaseImageTexture(*d, *p);
    releaseImageTexture(*d, *i);
    if (i->localReadCopy) d->gpu->destroy(i->localReadCopy);
    if (i->depthTileShadow) d->gpu->destroy(i->depthTileShadow);
  }
  imgRegRemove(i);
  delete i;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetImageMemoryRequirements(VkDevice, VkImage im,
                                                                    VkMemoryRequirements* r) {
  const Image* i = from<Image>(im);
  r->size = imageBytes(i->desc);
  for (const auto& p : i->planes) r->size += imageBytes(p->desc);
  r->alignment = 256;
  r->memoryTypeBits = (i->info.usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) ? 0xF : 0x7;
  // vk.memreq: the size and alignment a game is told for an image (what a streaming pool budgets with), for the first 32
  // distinct shapes of block-compressed formats and the first 32 of the others (render targets and the 1x1 defaults of an
  // engine come first and must not use up the textures' lines). vkGetImageMemoryRequirements2 and
  // vkGetDeviceImageMemoryRequirements answer through here too.
  static std::atomic<unsigned> lines[2];
  const int blocks = vpgpu::formatInfo(toCoreFormat(i->info.format)).blockSize > 1;
  if (lines[blocks].load(std::memory_order_relaxed) < 32 && !internalClient()) {
    char b[200];
    std::snprintf(b, sizeof b, "fmt=%d %ux%ux%u mips=%u layers=%u samples=%u usage=%#x size=%llu align=%llu", int(i->info.format),
                  i->info.extent.width, i->info.extent.height, i->info.extent.depth, i->info.mipLevels, i->info.arrayLayers,
                  unsigned(i->info.samples), unsigned(i->info.usage), (unsigned long long)r->size, (unsigned long long)r->alignment);
    static std::mutex mu;
    static std::set<std::string> seen[2];
    std::lock_guard<std::mutex> lock(mu);
    if (seen[blocks].size() < 32 && seen[blocks].insert(b).second) {
      lines[blocks].store(unsigned(seen[blocks].size()), std::memory_order_relaxed);
      logf("vk.memreq %s", b);
    }
  }
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetImageMemoryRequirements2(
    VkDevice dv, const VkImageMemoryRequirementsInfo2* in, VkMemoryRequirements2* out) {
  vkGetImageMemoryRequirements(dv, in->image, &out->memoryRequirements);
  // Disjoint multi-planar image: the requirements of the plane asked for.
  if (const auto* pi = static_cast<const VkImagePlaneMemoryRequirementsInfo*>(
          findNext(in->pNext, VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO))) {
    Image* whole = from<Image>(in->image);
    if (!whole->planes.empty())
      out->memoryRequirements.size = imageBytes(planeOf(whole, pi->planeAspect)->desc);
  }
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

namespace {
// Binds one image (a plane of a multi-planar image is an image of its own); submitMutex held.
VkResult bindOne(Device* d, Image* i, VkDeviceMemory mem, VkDeviceSize off) {
  diag::add(diag::c().imgBindN);
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
    zeroInitMark(*d, *i);
    // Persistent images are allocated now, so the lookup only reads; render targets that may stay
    // on tile memory are not touched (looking them up would allocate them).
    if (i->desc.persistent) {
      i->native = d->platform->nativeTexture(i->id);
      zeroNoteNative(*d, *i, i->native);
    }
  }
  if (i->eye) markEyeTexture(*d, i->id);
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
// Offset of plane `k` (0 = the image) of a multi-planar image bound at `off`: the planes follow each other.
VkDeviceSize planeOffset(const Image& im, size_t k, VkDeviceSize off) {
  if (k > 0) off += imageBytes(im.desc);
  for (size_t j = 1; j < k; ++j) off += imageBytes(im.planes[j - 1]->desc);
  return off;
}
}  // namespace

// Texture storage is Metal's (private, tiled); the VkDeviceMemory only accounts for it. A texture
// that aliases another resource's memory is not supported (logged on first sight: phase 2b).
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkBindImageMemory(VkDevice dv, VkImage im, VkDeviceMemory mem,
                                                             VkDeviceSize off) {
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  Image* i = from<Image>(im);
  Device* d = from<Device>(dv);
  // Same lock as vkQueueSubmit: creating textures inserts into maps the submit thread walks.
  std::lock_guard<std::mutex> lock(d->submitMutex);
  if (VkResult r = bindOne(d, i, mem, off); r) return r;
  for (size_t k = 0; k < i->planes.size(); ++k)
    if (VkResult r = bindOne(d, i->planes[k].get(), mem, planeOffset(*i, k + 1, off)); r) return r;
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkBindImageMemory2(VkDevice dv, uint32_t n,
                                                              const VkBindImageMemoryInfo* in) {
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  for (uint32_t i = 0; i < n; ++i) {
    // Disjoint multi-planar image: one call per plane, each bound with its own memory.
    if (const auto* pi = static_cast<const VkBindImagePlaneMemoryInfo*>(
            findNext(in[i].pNext, VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO))) {
      Image* whole = from<Image>(in[i].image);
      Device* d = from<Device>(dv);
      std::lock_guard<std::mutex> lock(d->submitMutex);
      if (VkResult r = bindOne(d, planeOf(whole, pi->planeAspect), in[i].memory, in[i].memoryOffset); r) return r;
      continue;
    }
    if (VkResult r = vkBindImageMemory(dv, in[i].image, in[i].memory, in[i].memoryOffset); r) return r;
  }
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetImageSubresourceLayout(VkDevice, VkImage im,
                                                                   const VkImageSubresource* sub,
                                                                   VkSubresourceLayout* out) {
  // Only meaningful for linear images (the layout their memory really has, see linearLayout). Multi-planar: the plane
  // asked for, its offset after the earlier planes (they follow each other, see planeOffset).
  const Image* whole = from<Image>(im);
  const Image* pl = whole;
  VkDeviceSize planeBase = 0;
  if (!whole->planes.empty()) {
    const uint32_t k = (sub->aspectMask & VK_IMAGE_ASPECT_PLANE_2_BIT) ? 2 : (sub->aspectMask & VK_IMAGE_ASPECT_PLANE_1_BIT) ? 1 : 0;
    if (k > 0 && k <= whole->planes.size()) pl = whole->planes[k - 1].get();
    for (uint32_t j = 0; j < k; ++j) planeBase += imageBytes(j == 0 ? whole->desc : whole->planes[j - 1]->desc);
  }
  const vpgpu::TextureDesc& d = pl->desc;
  if (sub->mipLevel >= d.mips || sub->arrayLayer >= d.layers) {
    *out = {};
    return;
  }
  const LinearLayout l = linearLayout(d, sub->mipLevel, sub->arrayLayer);
  out->offset = planeBase + l.offset;
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
    const uint64_t rows = (h + fi.blockHeight - 1) / fi.blockHeight;
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

namespace {
VkResult createViewOne(VkDevice dv, const VkImageViewCreateInfo* ci, VkImageView* out);
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateImageView(VkDevice dv, const VkImageViewCreateInfo* ci,
                                                             const VkAllocationCallbacks*,
                                                             VkImageView* out) {
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  // Multi-planar image: the view of the whole format (used with a sampler Y'CbCr conversion) is the
  // view of plane 0 plus one view per other plane; a view of a plane's own format and aspect is a
  // plain view of that plane's image.
  if (Image* whole = from<Image>(ci->image); whole && !whole->planes.empty()) {
    const MultiPlane& mp = multiPlane(whole->info.format);
    VkImageViewCreateInfo c = *ci;
    c.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    if (ci->format != whole->info.format) {
      c.image = to<VkImage>(planeOf(whole, ci->subresourceRange.aspectMask));
      return createViewOne(dv, &c, out);
    }
    c.format = mp.plane[0];
    c.components = {};
    VkImageView first = VK_NULL_HANDLE;
    if (VkResult r = createViewOne(dv, &c, &first); r) return r;
    ImageView* v0 = from<ImageView>(first);
    v0->format = ci->format;
    for (uint32_t k = 1; k < mp.planes; ++k) {
      c.image = to<VkImage>(whole->planes[k - 1].get());
      c.format = mp.plane[k];
      VkImageView pv = VK_NULL_HANDLE;
      if (VkResult r = createViewOne(dv, &c, &pv); r) {
        vkDestroyImageView(dv, first, nullptr);
        return r;
      }
      v0->planeViews.push_back(from<ImageView>(pv));
    }
    *out = first;
    return VK_SUCCESS;
  }
  return createViewOne(dv, ci, out);
}

namespace {
VkResult createViewOne(VkDevice dv, const VkImageViewCreateInfo* ci, VkImageView* out) {
  auto* v = new ImageView;
  v->image = from<Image>(ci->image);
  v->type = ci->viewType;
  v->format = ci->format;
  v->range = ci->subresourceRange;
  // Texture budget: the levels of the range the texture has; a view of dropped levels only shows the largest one left.
  if (v->image && v->image->mipDrop && !budgetRange(v->image, v->range)) v->range.baseMipLevel = 0, v->range.levelCount = 1;
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
    info.format = isDecodedVolume(*im) ? vpgpu::Format::Undefined : toCoreFormat(ci->format);  // the volume's own RGBA8
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
        // BC1_RGB lives in Metal's BC1_RGBA: alpha reads one (a 3-colour block's index 3 is black, not transparent black).
        if (s == 5 && (im->info.format == VK_FORMAT_BC1_RGB_UNORM_BLOCK || im->info.format == VK_FORMAT_BC1_RGB_SRGB_BLOCK))
          s = 1;
        info.swizzle[c] = s;
        if (s != 2 + c) swizzled = true;
      }
      // R/B-swapped storage (B4G4R4A4 as ABGR4): every component read through the swap.
      if (isSwappedRB(im->info.format)) {
        static const uint8_t kSwap[4] = {4, 3, 2, 5};
        for (int c = 0; c < 4; ++c)
          if (info.swizzle[c] >= 2) info.swizzle[c] = kSwap[info.swizzle[c] - 2];
        swizzled = true;
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
      if (v->view) diag::add(diag::c().viewCreateN);
    }
  }
  // Storage cube images are accessed as 2D arrays of faces (see ImageView::storageView).
  if (im && im->id && (ci->viewType == VK_IMAGE_VIEW_TYPE_CUBE || ci->viewType == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) &&
      (im->info.usage & VK_IMAGE_USAGE_STORAGE_BIT)) {
    const vpgpu::TextureDesc& d = im->desc;
    Platform::ViewInfo info;
    const uint32_t mips = v->range.levelCount == VK_REMAINING_MIP_LEVELS ? d.mips - v->range.baseMipLevel : v->range.levelCount;
    const uint32_t layers = v->range.layerCount == VK_REMAINING_ARRAY_LAYERS ? d.layers - v->range.baseArrayLayer : v->range.layerCount;
    info.baseMip = uint16_t(v->range.baseMipLevel), info.mips = uint16_t(mips);
    info.baseLayer = uint16_t(v->range.baseArrayLayer), info.layers = uint16_t(layers);
    info.format = toCoreFormat(ci->format);
    info.type = vpgpu::TextureType::Tex2D, info.array = true;
    Device* dd = from<Device>(dv);
    std::lock_guard<std::mutex> lock(dd->submitMutex);
    v->storageView = dd->platform->createTextureView(im->id, info);
  }
  *out = to<VkImageView>(v);
  return VK_SUCCESS;
}

}  // namespace

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyImageView(VkDevice dv, VkImageView v,
                                                          const VkAllocationCallbacks*) {
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  if (!v) return;
  ImageView* iv = from<ImageView>(v);
  for (ImageView* pv : iv->planeViews) vkDestroyImageView(dv, to<VkImageView>(pv), nullptr);
  if (iv->view || iv->storageView) {
    Device* d = from<Device>(dv);
    std::lock_guard<std::mutex> lock(d->submitMutex);
    if (iv->storageView) d->platform->destroyTextureView(iv->storageView);
    if (iv->view) {
      d->platform->destroyTextureView(iv->view);
      diag::add(diag::c().viewDestroyN);
    }
  }
  delete iv;
}

namespace vpvk {
// Only material-style samplers take the forced anisotropy — linear min/mag/mip (trilinear) with a mip
// range to walk and a wrapping address mode on U or V. Excluded, keeping the game's value: nearest,
// comparison (shadow), unnormalized, and all-clamp samplers (post-processing, render-target reads,
// cubemaps, 3D LUTs use clamp-to-edge/border), where anisotropic taps cost bandwidth on full-screen
// passes and give nothing.
bool samplerAnisoEligible(const VkSamplerCreateInfo& ci) {
  const auto wraps = [](VkSamplerAddressMode m) {
    return m == VK_SAMPLER_ADDRESS_MODE_REPEAT || m == VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
  };
  return ci.minFilter == VK_FILTER_LINEAR && ci.magFilter == VK_FILTER_LINEAR &&
         ci.mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR && !ci.compareEnable && !ci.unnormalizedCoordinates &&
         ci.maxLod > ci.minLod + 0.25f && (wraps(ci.addressModeU) || wraps(ci.addressModeV));
}
}  // namespace vpvk

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
  // App setting "Filtrado anisotrópico" (global or per game: VPVK_FORCE_ANISO), see samplerAnisotropy.
  {
    const int forced = vpvk::anisotropyForced();
    const bool eligible = vpvk::samplerAnisoEligible(*ci);
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
  d.shaderLodBias = std::clamp(ci->mipLodBias, -16.0f, 15.99f);  // maxSamplerLodBias; applied by the shader (SamplerBias)
  // Texel coordinates: Vulkan's restrictions (one filter, no mips, clamp modes, lod 0) match
  // Metal's for normalizedCoordinates = NO.
  d.normalized = !ci->unnormalizedCoordinates;
  d.label = "vk.sampler";
  auto* s = new Sampler;
  if (const auto* yi = static_cast<const VkSamplerYcbcrConversionInfo*>(
          findNext(ci->pNext, VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO));
      yi && yi->conversion)
    s->ycbcr = from<YcbcrConversionHandle>(yi->conversion)->obj;
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
// The device of that callback (vpvk_taa_native_encode / vpvk_depth_resolve_encode run inside it and
// must not take the registry mutex: lock order is registry -> submit).
thread_local Device* tNativeDevice = nullptr;
void* imageNative(Device* d, const Image* img, bool submitHeld = false) {
  vpgpu::ScopedPool pool;  // guest thread: nativeTexture hands back autoreleased Metal objects
  if (!d || !img || !img->id) return nullptr;
  if (img->native) return img->native;
  if (tInNativeSubmit || submitHeld) {
    void* n = d->platform->nativeTexture(img->id);
    zeroNoteNative(*d, *img, n);
    return n;
  }
  std::lock_guard<std::mutex> lock(d->submitMutex);
  void* n = d->platform->nativeTexture(img->id);
  zeroNoteNative(*d, *img, n);
  return n;
}
// Parks the calling host thread while the app is in the background, as the device it will use (the owner vkDestroyDevice releases:
// a thread parked for a device that is being destroyed is woken, not left for ever). Before any lock is taken.
// `d`: the device the call is about (vkQueueSubmit: the queue's, which need not be the live one); null: the live device
// (vpvk_native_submit), read under the registry, which stays held until the gate has the wait registered (BackgroundGate::
// waitForeground with a registry): a vkDestroyDevice that takes the device out of the registry and releases it can no longer slip
// in between the read and the park. No live device: nothing to park for, the call that follows finds none and answers at once.
void parkWhileBackground(const Device* d = nullptr) {
  vpgpu::BackgroundGate& gate = vpgpu::backgroundGate();
  if (!gate.active()) return;  // (lock free: the foreground does not touch the registry)
  if (d) return gate.waitForeground(d);
  std::unique_lock<std::mutex> live(liveDeviceMutex());
  const Device* owner = liveDevice();
  if (!owner) return;
  gate.waitForeground(owner, live);
}
// The live device with its submit mutex taken (or tried: `lock` stays unowned when it is busy). Null: no device.
// The submit mutex is never waited for WITH the registry: a frame held while its pipelines compile (gpu.hold) keeps it for as long as the
// compile takes, and a thread waiting for it holding the registry would stop every per-frame call that takes the registry (and deadlock a
// vpvk_native_submit callback, which holds the submit mutex and may call back into the driver). The device is pinned instead
// (Device::hostUsers, counted under the registry; vkDestroyDevice waits for it after taking the device out of the registry), the registry
// is released and the wait is an ordinary blocking lock: no polling, the mutex's own wake-up order.
// Inside a vpvk_native_submit callback this thread already holds the submit mutex: a host call that needs it (not the try-only one, which
// answers "busy") cannot be served and gets null, a clean failure instead of a self-deadlock.
Device* liveDeviceSubmitLocked(std::unique_lock<std::mutex>& lock, bool tryOnly = false) {
  if (tInNativeSubmit) {
    lock = std::unique_lock<std::mutex>();
    if (!tryOnly) {
      static std::atomic<unsigned> n{0};
      if (n.fetch_add(1) < 4) logf("vk.err what=reentrant_host_call (vpvk_native_submit / vpvk_mark_eye_image from inside a native-submit callback: refused)");
    }
    return tryOnly ? tNativeDevice : nullptr;
  }
  Device* d;
  {
    std::lock_guard<std::mutex> live(liveDeviceMutex());
    d = liveDevice();
    if (!d) return nullptr;
    if (tryOnly) {
      lock = std::unique_lock<std::mutex>(d->submitMutex, std::try_to_lock);
      return d;
    }
    d->hostUsers.fetch_add(1);
  }
  lock = std::unique_lock<std::mutex>(d->submitMutex);
  d->hostUsers.fetch_sub(1);  // (holding the submit mutex now: vkDestroyDevice drains it before it deletes the device)
  return d;
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
  std::unique_lock<std::mutex> submit;
  if (tInNativeSubmit) {  // inside a vpvk_native_submit callback: this thread holds the submit mutex, the device cannot go
    d = tNativeDevice;
  } else if (from<Image>(image)->native) {  // known already: the device is only tested for null, never used
    std::lock_guard<std::mutex> l(liveDeviceMutex());
    d = liveDevice();
  } else {
    d = liveDeviceSubmitLocked(submit);
  }
  *out = imageNative(d, from<Image>(image), true);
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
      case VK_STRUCTURE_TYPE_EXPORT_METAL_COMMAND_QUEUE_INFO_EXT:
        // The queue the game's frames are committed to: a host that watches it (GPU time, pace) needs the real one.
        reinterpret_cast<VkExportMetalCommandQueueInfoEXT*>(s)->mtlCommandQueue =
            static_cast<MTLCommandQueue_id>(d->platform->nativeQueue());
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
    b.loadOp = a.loadOp == VK_ATTACHMENT_LOAD_OP_NONE_KHR ? VK_ATTACHMENT_LOAD_OP_LOAD : a.loadOp;  // NONE keeps the contents
    b.storeOp = a.storeOp;
    b.stencilLoadOp = a.stencilLoadOp == VK_ATTACHMENT_LOAD_OP_NONE_KHR ? VK_ATTACHMENT_LOAD_OP_LOAD : a.stencilLoadOp;
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
  for (VkAttachmentDescription2& a : rp->attachments) {  // LOAD_OP_NONE (VK_EXT_load_store_op_none) keeps the contents: Load
    if (a.loadOp == VK_ATTACHMENT_LOAD_OP_NONE_KHR) a.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    if (a.stencilLoadOp == VK_ATTACHMENT_LOAD_OP_NONE_KHR) a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
  }
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
  // VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT: no views now (pAttachments ignored); they come with each
  // vkCmdBeginRenderPass (VkRenderPassAttachmentBeginInfo).
  if (!(ci->flags & VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT))
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
                                                              VkCommandPoolResetFlags flags) {
  for (CommandBuffer* c : from<CommandPool>(cp)->buffers) {
    c->rec->reset();
    if (flags & VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT) c->rec->releaseResources();
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
    c->poolIndex = p->buffers.size();
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
    p->removeBuffer(c);
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
  from<CommandBuffer>(cb)->rec->flushHoisted();  // (a pass left open to be resumed)
  from<CommandBuffer>(cb)->state = CommandBuffer::State::Executable;
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetCommandBuffer(VkCommandBuffer cb,
                                                                VkCommandBufferResetFlags flags) {
  CommandBuffer* c = from<CommandBuffer>(cb);
  c->rec->reset();
  if (flags & VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT) c->rec->releaseResources();
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
  // Imageless framebuffer: this pass's copy with the views given now.
  if (const auto* ab = static_cast<const VkRenderPassAttachmentBeginInfo*>(
          findNext(bi->pNext, VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO));
      ab && op.framebuffer) {
    auto fb = std::make_unique<Framebuffer>(*op.framebuffer);
    fb->views.clear();
    for (uint32_t i = 0; i < ab->attachmentCount; ++i) fb->views.push_back(from<ImageView>(ab->pAttachments[i]));
    op.framebuffer = fb.get();
    r.imageless.push_back(std::move(fb));
  }
  r.bound.passImages.clear();  // dynamic rendering only (local read): render passes split at self-dependencies
  r.bound.localRead = false;
  r.bound.pass = op.pass;
  r.bound.framebuffer = op.framebuffer;
  r.bound.subpass = 0;
  r.bound.passSamples = op.pass && op.framebuffer ? subpassSamples(*op.pass, *op.framebuffer, 0, true, vpvk::markedEyeOf(r.device)) : 0;
  r.bound.targetWidth = uint32_t(op.area.offset.x) + op.area.extent.width;
  r.bound.targetHeight = uint32_t(op.area.offset.y) + op.area.extent.height;
  r.bound.renderArea = op.area;
  r.bound.fbWidth = op.framebuffer ? op.framebuffer->width : 0;
  r.bound.fbHeight = op.framebuffer ? op.framebuffer->height : 0;
  r.bound.fbLayers = op.framebuffer ? op.framebuffer->layers : 0;
  r.flushHoisted();  // (a pass that was never ended)
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
  r.flushHoisted();
  r.ops.emplace_back(OpEndPass{});
  r.inPass = false;
  r.passBegin = SIZE_MAX;
  r.bound.localRead = false;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdEndRenderPass2(VkCommandBuffer cb, const VkSubpassEndInfo*) {
  vkCmdEndRenderPass(cb);
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier(
    VkCommandBuffer cb, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags, uint32_t,
    const VkMemoryBarrier*, uint32_t, const VkBufferMemoryBarrier*, uint32_t, const VkImageMemoryBarrier*) {
  Recorder& r = *from<CommandBuffer>(cb)->rec;
  ++r.barriers;  // the core places barriers from real hazards
  recordInPassBarrier(r);
}

// Inside a render pass a barrier is a subpass self-dependency (or, in dynamic rendering, a local read): kept as a
// marker, the translator splits a subpass that reads its own attachments there (commands.cpp).
void vpvk::recordInPassBarrier(Recorder& r) {
  if (!r.inPass) return;
  if (!r.bound.passImages.empty()) r.hasLocalRead = true;
  r.bound.localRead = true;
  r.ops.emplace_back(OpPassBarrier{});
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
  for (uint32_t i = 0; i < n; ++i) {
    VkImageSubresourceRange range = ranges[i];
    if (from<Image>(im)->mipDrop && !budgetRange(from<Image>(im), range)) continue;  // only levels the budget dropped
    r.ops.emplace_back(OpClearColor{from<Image>(im), *color, range});
  }
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer src, VkImage dst,
                                                              VkImageLayout, uint32_t n,
                                                              const VkBufferImageCopy* regions) {
  Recorder& r = *from<CommandBuffer>(cb)->rec;
  for (uint32_t i = 0; i < n; ++i) {
    VkBufferImageCopy c = regions[i];
    Image* im = planeOf(from<Image>(dst), c.imageSubresource.aspectMask);  // a plane of a multi-planar image
    if (im != from<Image>(dst)) c.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    noteMipsWritten(im, c.imageSubresource.mipLevel);
    OpCopyBufferToImage op{from<Buffer>(src), im, c};
    if (im->mipDrop) {
      uint32_t level = c.imageSubresource.mipLevel;
      if (budgetLevel(im, level)) {
        op.region.imageSubresource.mipLevel = level;
      } else {
        // A level the budget dropped. Its texels are not kept, except the top level of an uncompressed image, scaled down into
        // the largest level left (commands.cpp shrunkCopy): what a single-level image shows, and what a chain built from its top
        // level (blits) starts from. A chain uploaded level by level overwrites it with its own level.
        const BudgetShrink bs = budgetShrink(*im, c);
        if (!bs.bytes) continue;
        op.shrink = im->mipDrop;
        op.decoded = r.arena.put(r.device, nullptr, bs.bytes, op.decodedOffset);
        op.decodedHost = r.arena.chunks[r.arena.current].host + op.decodedOffset;
      }
    }
    if (isDecodedVolume(*im)) {  // room for the RGBA8 texels; the blocks are decoded at submit (commands.cpp)
      const VkDeviceSize bytes = VkDeviceSize(c.imageExtent.width) * c.imageExtent.height * c.imageExtent.depth * 4;
      op.decoded = r.arena.put(r.device, nullptr, bytes, op.decodedOffset);
      op.decodedHost = r.arena.chunks[r.arena.current].host + op.decodedOffset;
    }
    r.ops.emplace_back(std::move(op));
  }
}

// ---------------------------------------------------------------------------------------------
// Synchronisation and submission
// ---------------------------------------------------------------------------------------------
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateFence(VkDevice, const VkFenceCreateInfo* ci,
                                                         const VkAllocationCallbacks*, VkFence* out) {
  auto* f = new Fence;
  f->signaled = (ci->flags & VK_FENCE_CREATE_SIGNALED_BIT) != 0;
  pendingTrack(f);
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
  pendingUntrack(fe);
  delete fe;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetFences(VkDevice, uint32_t n, const VkFence* fs) {
  for (uint32_t i = 0; i < n; ++i) from<Fence>(fs[i])->signaled = false;
  return VK_SUCCESS;
}

// A fence is signaled when its submission's GPU work is done: every vpgpu frame up to the one its
// vkQueueSubmit made (Fence::frames) has completed on the GPU. The client reuses that submission's
// command buffers (upload memory, descriptors) as soon as this says done.
static VkResult fenceStatus(VkDevice dv, VkFence f) {
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
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetFenceStatus(VkDevice dv, VkFence f) {
  const VkResult r = fenceStatus(dv, f);
  from<Fence>(f)->poll.note(r);  // vk.pending (pending.cpp)
  return r;
}

static VkResult waitForFences(VkDevice dv, uint32_t n, const VkFence* fs, VkBool32 waitAll, uint64_t timeout) {
  PerfWait perfWait(perf().fenceWaits, perf().fenceUs, perf().fenceUsMax);  // vk.perf fence_wait
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
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(VkDevice dv, uint32_t n, const VkFence* fs,
                                                           VkBool32 waitAll, uint64_t timeout) {
  pendingFenceWait(n, fs, true);  // vk.pending (pending.cpp)
  const VkResult r = waitForFences(dv, n, fs, waitAll, timeout);
  pendingFenceWait(n, fs, false, waitAll, r);
  return r;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateSemaphore(VkDevice, const VkSemaphoreCreateInfo* ci,
                                                             const VkAllocationCallbacks*, VkSemaphore* out) {
  auto* s = new Semaphore;
  if (const auto* t = static_cast<const VkSemaphoreTypeCreateInfo*>(
          ci ? findNext(ci->pNext, VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO) : nullptr);
      t && t->semaphoreType == VK_SEMAPHORE_TYPE_TIMELINE) {
    s->timeline = true;
    s->completed = t->initialValue;
    pendingTrack(s);
  }
  *out = to<VkSemaphore>(s);
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroySemaphore(VkDevice, VkSemaphore s,
                                                          const VkAllocationCallbacks*) {
  if (Semaphore* sem = from<Semaphore>(s); sem && sem->timeline) pendingUntrack(sem);
  delete from<Semaphore>(s);
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateEvent(VkDevice, const VkEventCreateInfo*,
                                                         const VkAllocationCallbacks*, VkEvent* out) {
  auto* e = new Event;
  pendingTrack(e);
  *out = to<VkEvent>(e);
  return VK_SUCCESS;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyEvent(VkDevice, VkEvent e, const VkAllocationCallbacks*) {
  pendingUntrack(from<Event>(e));
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
  const VkResult r = ev->set ? VK_EVENT_SET : VK_EVENT_RESET;
  ev->poll.note(r);  // vk.pending (pending.cpp)
  return r;
}
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkSetEvent(VkDevice, VkEvent e) {
  Event* ev = from<Event>(e);
  std::lock_guard<std::mutex> l(ev->m);
  ev->set = true, ev->frame = 0, ++ev->hostSet;
  return VK_SUCCESS;
}
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetEvent(VkDevice, VkEvent e) {
  Event* ev = from<Event>(e);
  std::lock_guard<std::mutex> l(ev->m);
  ev->set = false, ev->frame = 0, ++ev->hostReset;
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

}  // namespace

// ---- zero-initialised images ----
// A Quest driver hands out zeroed memory, so an image a game reads before writing it (Unity copies
// OVROverlay layers it never rendered) is transparent there; Metal's private textures hold whatever
// the heap held, which showed up as magenta. Chosen design: lazy, at the submit that first uses the
// image, not at bind. Bind only records the texture (cheap: thousands of textures are bound while a
// game loads and nearly all are fully overwritten by their upload). The submit already holds its
// translated frame, so one scan of it tells which pending images are read first (or only partly
// written) and need a clear; the others are dropped for free. The clears go in a frame of their own
// submitted just before, so the main frame is not copied. Once nothing is pending the scan is skipped.
bool zeroNewImagesOn() {
  static const bool on = [] {
    const char* e = getenv("VPVK_ZERO_NEW_IMAGES");
    return !(e && e[0] == '0' && !e[1]);
  }();
  return on;
}

namespace {
uint64_t zeroKey(uint32_t index, uint32_t generation) { return uint64_t{generation} << 32 | index; }
// Subresources of the image the bitmask tracks (0: not trackable, always cleared once touched).
uint64_t zeroFullMask(const vpgpu::TextureDesc& d) {
  const uint64_t n = uint64_t{d.mips} * d.layers;
  if (d.type == vpgpu::TextureType::Tex3D || n == 0 || n > 64) return 0;
  return n == 64 ? ~uint64_t{0} : (uint64_t{1} << n) - 1;
}
}  // namespace

void markEyeTexture(Device& d, vpgpu::TextureId id) {
  if (!id) return;
  std::lock_guard<std::mutex> l(d.eyeMutex);
  d.eyeTextures.insert(uint64_t{id.generation} << 32 | id.index);
}
bool isEyeTexture(Device& d, vpgpu::TextureId id) {
  if (!id) return false;
  std::lock_guard<std::mutex> l(d.eyeMutex);
  return d.eyeTextures.count(uint64_t{id.generation} << 32 | id.index) != 0;
}

void zeroInitMark(Device& d, const Image& i) {
  if (!zeroNewImagesOn() || !i.id || i.linear || i.desc.transient) return;
  const vpgpu::FormatInfo& fi = vpgpu::formatInfo(i.desc.format);
  const bool depth = fi.depth() || fi.stencil();
  // Colour and depth/stencil targets are cleared with a pass. Everything else (sampled or storage
  // images, compressed ones included) is zeroed by copying from a buffer of zeros; depth formats
  // that cannot be rendered into have no such copy and stay as they are.
  const bool target = depth ? (i.desc.usage & vpgpu::usage::depthTarget) : (i.desc.usage & vpgpu::usage::colorTarget);
  if (!target && depth) return;
  Device::ZeroPending p;
  p.id = i.id;
  p.desc = i.desc;
  p.viaCopy = !target;
  p.flag = i.idOwners;
  if (i.idOwners) std::atomic_ref<int>(*i.idOwners).store(1, std::memory_order_relaxed);
  d.zeroPending[zeroKey(i.id.index, i.id.generation)] = std::move(p);
}

void zeroForgetNative(Device& d, Device::ZeroPending& p) {
  if (!p.native) return;
  std::lock_guard<std::mutex> l(d.zeroNativeMutex);
  d.zeroNative.erase(p.native);
  p.native = nullptr;
}

void zeroNoteNative(Device& d, const Image& i, void* native) {
  if (!native || !zeroPendingImage(&i)) return;
  auto it = d.zeroPending.find(zeroKey(i.id.index, i.id.generation));
  if (it == d.zeroPending.end()) return;
  it->second.native = native;
  std::lock_guard<std::mutex> l(d.zeroNativeMutex);
  d.zeroNative[native] = it->first;
}

size_t zeroInitResolve(Device& d, const ZeroLog& log, vpgpu::Frame& pre) {
  if (d.zeroPending.empty() || log.touches.empty()) return 0;
  std::vector<Device::ZeroPending> clear;
  std::vector<uint64_t> touchedKeys;
  // Settled: leaves the pending set (record time stops logging it).
  auto settle = [&](std::unordered_map<uint64_t, Device::ZeroPending>::iterator it, bool needClear) {
    if (auto f = it->second.flag.lock()) std::atomic_ref<int>(*f).store(0, std::memory_order_relaxed);
    zeroForgetNative(d, it->second);
    if (needClear) clear.push_back(std::move(it->second));
    d.zeroPending.erase(it);
  };
  // Touches come in submit order; an image is decided by its first read, or after the last touch.
  for (const ZeroTouch& t : log.touches) {
    auto it = d.zeroPending.find(zeroKey(t.tex.index, t.tex.generation));
    if (it == d.zeroPending.end()) continue;
    Device::ZeroPending& p = it->second;
    if (t.kind == ZeroTouch::Read) {
      // First access is a read (or the image is only partly written): clear it, it is settled.
      const uint64_t full = zeroFullMask(p.desc);
      settle(it, !full || p.written != full);
      continue;
    }
    if (!p.touched) touchedKeys.push_back(it->first);
    p.touched = true;
    if (t.kind == ZeroTouch::WriteSpan &&
        !(t.x == 0 && t.y == 0 && t.z == 0 && t.width >= std::max(1u, p.desc.width >> t.mip) &&
          t.height >= std::max(1u, p.desc.height >> t.mip)))
      continue;  // a partial write: touches the image, covers nothing
    for (uint32_t l = t.layer; l < uint32_t(t.layer) + t.layers && l < p.desc.layers; ++l)
      if (const uint64_t b = uint64_t{t.mip} * p.desc.layers + l; b < 64 && t.mip < p.desc.mips) p.written |= uint64_t{1} << b;
  }
  // Touched but never read: a full overwrite settles it for free, anything less is cleared.
  for (uint64_t key : touchedKeys) {
    auto it = d.zeroPending.find(key);
    if (it == d.zeroPending.end()) continue;
    const uint64_t full = zeroFullMask(it->second.desc);
    settle(it, !full || it->second.written != full);
  }
  for (const Device::ZeroPending& p : clear) {
    const vpgpu::FormatInfo& fi = vpgpu::formatInfo(p.desc.format);
    const bool depth = fi.depth() || fi.stencil();
    const bool is3d = p.desc.type == vpgpu::TextureType::Tex3D;
    noteImgZero(!p.viaCopy ? "zero_clear" : astcFormat(p.desc.format) ? "zero_astc_void" : "zero_copy", -1, p.desc.format,
                p.desc.width, p.desc.height, p.desc.mips, p.desc.layers, p.desc.usage);
    if (p.viaCopy) {
      // Row bands of at most kZeroBufferBytes per copy, one layer (3D: one slice) at a time.
      constexpr uint64_t kZeroBufferBytes = 8u << 20;
      if (!d.zeroBuffer) {
        vpgpu::BufferDesc bd;
        bd.size = kZeroBufferBytes;
        bd.label = "vk.zeros";
        d.zeroBuffer = d.gpu->createBuffer(std::move(bd));
        if (!d.zeroBuffer) continue;
        pre.fillBuffer({d.zeroBuffer, 0, kZeroBufferBytes}, 0);  // new buffers are not guaranteed zero
      }
      // An all-zero ASTC block has a reserved block mode: it decodes to the error colour (magenta). The void-extent block of
      // colour (0, 0, 0, 0) is what zeroed memory should read as; ETC2 / EAC / BC blocks of zeros are valid already.
      const bool astc = astcFormat(p.desc.format);
      if (astc && !d.zeroAstcBuffer) {
        vpgpu::BufferDesc bd;
        bd.size = kZeroBufferBytes;
        bd.label = "vk.zeros_astc";
        d.zeroAstcBuffer = d.gpu->createBuffer(std::move(bd));
        if (!d.zeroAstcBuffer) continue;
        // Block: FC FD, six FF (void extent, LDR, no extent given), eight 0 (R, G, B, A = 0). Then doubled up to the whole buffer.
        const vpgpu::BufferId z = d.zeroAstcBuffer;
        pre.fillBuffer({z, 0, 1}, 0xFC);
        pre.fillBuffer({z, 1, 1}, 0xFD);
        pre.fillBuffer({z, 2, 6}, 0xFF);
        pre.fillBuffer({z, 8, 8}, 0);
        for (uint64_t n = 16; n < kZeroBufferBytes; n *= 2) pre.copyBuffer({z, 0, n}, {z, n, n});
      }
      const vpgpu::BufferId zeros = astc ? d.zeroAstcBuffer : d.zeroBuffer;
      const uint32_t bw = std::max<uint32_t>(1, fi.blockSize), bh = std::max<uint32_t>(1, fi.blockHeight);
      for (uint32_t m = 0; m < p.desc.mips; ++m) {
        const uint32_t w = std::max(1u, p.desc.width >> m), h = std::max(1u, p.desc.height >> m);
        const uint64_t rowBytes = uint64_t{(w + bw - 1) / bw} * fi.bytesPerBlock;
        const uint32_t bandBlocks = uint32_t(std::max<uint64_t>(1, kZeroBufferBytes / std::max<uint64_t>(1, rowBytes)));
        const uint32_t slices = is3d ? std::max(1u, p.desc.depth >> m) : p.desc.layers;
        for (uint32_t l = 0; l < slices; ++l)
          for (uint32_t y = 0; y < h; y += bandBlocks * bh) {
            vpgpu::Copy c;
            c.kind = vpgpu::CopyKind::BufferToTexture;
            c.srcBuffer = {zeros, 0, kZeroBufferBytes, uint32_t(rowBytes), 0};
            vpgpu::TextureSpan& t = c.dstTexture;
            t.texture = p.id, t.mip = uint16_t(m), t.layer = is3d ? 0 : uint16_t(l), t.layerCount = 1;
            t.y = y, t.z = is3d ? l : 0;
            t.width = w, t.height = std::min(h - y, bandBlocks * bh), t.depth = 1;
            c.srcBuffer.rowsPerImage = (t.height + bh - 1) / bh;
            c.label = "vk.zeroinit";
            pre.copy(std::move(c));
          }
      }
      continue;
    }
    for (uint32_t m = 0; m < p.desc.mips; ++m)
      for (uint32_t l = 0, n = is3d ? std::max(1u, p.desc.depth >> m) : p.desc.layers; l < n; ++l) {
        vpgpu::RenderPassDesc rd;  // Clear load with zero values (ClearValue is zero-initialised, depth aside)
        vpgpu::Attachment& a = depth ? rd.depth : rd.colors[0];
        a.texture = p.id, a.mip = static_cast<uint16_t>(m), a.layer = static_cast<uint16_t>(l);
        a.load = vpgpu::LoadAction::Clear, a.store = vpgpu::StoreAction::Store;
        a.clear.depth = 0.0f, a.clear.stencil = 0;
        rd.label = "vk.zeroinit";
        pre.clear(std::move(rd));
      }
  }
  d.zeroCleared += clear.size();
  return clear.size();
}

namespace {
// vk.perf (perf.h): at most one line per second, written by whichever submit crosses the second (submitMutex held).
void perfTick(Device* d) {
  static PerfLine line;
  static std::mutex lineMutex;  // one PerfLine for every device: a submit that finds it busy leaves the line to the other
  std::unique_lock<std::mutex> busy(lineMutex, std::try_to_lock);
  if (!busy.owns_lock()) return;
  PipelineCounters& pc = pipelineCounters();
  PerfSample ext;
  ext.psoCreateUs = pc.blockedCreateUs.load(std::memory_order_relaxed);
  ext.psoRecordUs = pc.blockedRecordUs.load(std::memory_order_relaxed);
  ext.psoGateUs = pc.blockedGateUs.load(std::memory_order_relaxed);
  ext.psoHardWaits = pc.hardWaits.load(std::memory_order_relaxed);
  ext.psoHoldUs = uint64_t(d->gpu->holdTotalMs() * 1000.0);
  const std::string text = line.tick(PerfLine::Clock::now(), ext);
  if (!text.empty()) logf("%s", text.c_str());
}

// Runs batches [0, count) of `list` as one frame (submitMutex held), then stamps their timeline
// signals, queries, events and fences with its frame.
void runBatches(Device* d, std::vector<DeferredBatch>& list, size_t count) {
  const auto perfT0 = std::chrono::steady_clock::now();  // vk.perf
  drainRetiredChunks(d);  // (submit mutex held: the arena chunks command buffer resets gave up since the last submission)
  vpgpu::Frame frame;
  TranslateStats st;
  uint32_t barriers = 0, cbs = 0;
  // One VPKTAA jitter for the whole submission (read once: a concurrent set applies to the next).
  const TaaJitter taa = taaJitter();
  d->uiMask.dev = d;
  // Nobody takes the masks when the native pass is off (RCAS fused in the host's compositor): a frame's two masks
  // would pile up and the slot parity (first masked pass = eye 0) drift. A full frame's worth already here is
  // stale: start the new one clean.
  if (!taaNativeOn() && d->uiMask.maskedPasses >= 2) {
    d->uiMask.maskedPasses = 0;
    d->uiMask.live[0].valid = d->uiMask.live[1].valid = false;
  }
  ZeroLog zl;
  zl.dev = d;
  for (size_t i = 0; i < count; ++i)
    for (const CommandBuffer* c : list[i].cbs) {
      translate(*c->rec, frame, st, taa.enabled ? &taa : nullptr, &d->uiMask, &zl, &d->fovState, nativeExtras() ? d : nullptr);
      barriers += c->rec->barriers;
      ++cbs;
    }
  const uint64_t perfTranslateUs = perfUsSince(perfT0);
  PassDump& pdump = passDump();  // VPVK_PASS_DUMP (passdump.h): a few scene frames pass by pass
  bool dumpFrame = false;
  if (pdump.on) {
    dumpFrame = passDumpScene(frame, *d->gpu) && passDumpNext(pdump);
    frame.setPassDump(dumpFrame);
  }
  if (!zl.touches.empty()) {
    vpgpu::Frame zeros;
    if (const size_t n = zeroInitResolve(*d, zl, zeros)) {
      d->gpu->submit(zeros);  // before the frame: what it reads starts as zeros
      if (d->zeroLogLines++ < 64) logf("vk.zeroinit images=%zu pending=%zu", n, d->zeroPending.size());
    }
  }
  awaitFreshVariants(d);  // synchronous mode: the UI-mask variants this submission requested, awaited together
  const std::vector<uint32_t> drawsBeforeGate = dumpFrame ? passDrawCounts(frame) : std::vector<uint32_t>{};
  const GateResult gate = gateFramePipelines(d, frame, d->submits + 1);  // pipelines still compiling: a short wait, else their draws wait a frame
  if (pdump.on) {
    if (dumpFrame) {
      std::vector<uint32_t> gateDropped = passDrawCounts(frame);
      for (size_t i = 0; i < gateDropped.size() && i < drawsBeforeGate.size(); ++i) gateDropped[i] = drawsBeforeGate[i] - gateDropped[i];
      for (std::string& l : passDumpLines(d->gpu->submittedFrames(), st.passNotes, gateDropped)) frame.dumpLine(std::move(l));  // (logged by the core, under its line cap)
    }
    passDumpAccumulate(st.passNotes, gate.skipped);
  }
  vpgpu::FrameReport perfReport;
  uint64_t perfBackendUs = 0;
  const bool perfSent = !frame.commands().empty();
  if (perfSent) {
    const auto b0 = std::chrono::steady_clock::now();
    perfReport = d->gpu->submit(frame);
    perfBackendUs = perfUsSince(b0);
  }
  ++d->submits;
  pipelineSummary(d, false);
  psoCensus(d, false);
  texHistTick();
  cacheStateTick(d);
  ampCensus();
  indirectStatsLog();
  msaaSubmitTick(d->submits);
  if (taa.enabled) taaRecordSubmit(d->submits, taa, st.taaJittered, st.taaSkipped, st.taaTargetW, st.taaTargetH);
  if (taa.enabled && st.ui.passes) {  // masks for hosts that composite the eye themselves (vpvk_ui_mask_texture)
    UiMaskState& u = d->uiMask;
    std::lock_guard<std::mutex> l(u.lastMu);
    for (UiMaskState::Published& p : u.last) {
      if (p.tex) d->platform->releaseNative(p.tex);
      p = UiMaskState::Published{};
    }
    if (st.ui.maskedPasses) {
      const bool shared = u.live[0].valid && u.live[0].t.layers >= 2;
      for (int e = 0; e < 2; ++e) {
        const UiMaskState::Live& lv = shared ? u.live[0] : u.live[e];
        if (!lv.valid || !lv.t.tex) continue;
        void* p = d->platform->retainNative(d->platform->nativeTexture(lv.t.tex));
        if (!p) continue;
        u.last[e] = {p, lv.t.w, lv.t.h, shared ? uint32_t(e) : 0u};
      }
    }
  }
  if (taa.enabled && st.ui.passes) {  // VPKTAA UI mask: draws by class (first submits, then every 600)
    UiMaskState& u = d->uiMask;
    u.counters.scene += st.ui.scene, u.counters.ui += st.ui.ui, u.counters.other += st.ui.other;
    u.counters.passes += st.ui.passes, u.counters.maskedPasses += st.ui.maskedPasses;
    if (u.logLines < 4 || d->submits >= u.nextLogSubmit) {
      logf("vk.taa uimask submit=%llu draws_3d=%u draws_ui=%u draws_other=%u eye_passes=%u masked_passes=%u "
           "(acumulado desde la línea anterior; clasificación: UI = sin escritura de profundidad, tras lo opaco, con mezcla o sin test)",
           static_cast<unsigned long long>(d->submits), u.counters.scene, u.counters.ui, u.counters.other, u.counters.passes,
           u.counters.maskedPasses);
      u.counters = {};
      ++u.logLines;
      u.nextLogSubmit = d->submits + 600;
    }
  }
  if (st.fovPasses) {  // foveation: eye passes given a rate map (the backend logs physical vs logical sizes)
    d->fovLog.passes += st.fovPasses, d->fovLog.splits += st.fovUiSplits;
    if (d->fovLog.lines < 4 || d->submits >= d->fovLog.next) {
      logf("vk.fov frame submit=%llu eye_passes=%u ui_splits=%u logical=%ux%u (acumulado desde la línea anterior)",
           static_cast<unsigned long long>(d->submits), d->fovLog.passes, d->fovLog.splits, st.fovLogicalW, st.fovLogicalH);
      d->fovLog.passes = d->fovLog.splits = 0;
      ++d->fovLog.lines;
      d->fovLog.next = d->submits + 600;
    }
  }
  // Skips and drops: the first 64 submits that have any, then one in 600 of them (each line is I/O on the submit thread).
  const bool skipLine = (st.dropped || st.skippedDraws || st.skippedDispatches) &&
                        (d->skipLogCount < 64 || d->skipLogCount % 600 == 0);
  if (st.dropped || st.skippedDraws || st.skippedDispatches) ++d->skipLogCount;
  if (skipLine || d->submits <= 3 || d->submits % 600 == 0) {
    std::string skipBy;  // skipped draws by reason
    if (st.skippedDraws) {
      skipBy = " skip_by{";
      for (size_t i = 0; i < size_t(SkipReason::Count); ++i)
        skipBy += (i ? " " : "") + std::string(kSkipReasonNames[i]) + "=" + std::to_string(st.skippedBy[i]);
      skipBy += "}";
    }
    logf("vk.submit n=%llu cbs=%u passes=%u draws=%u dispatches=%u copies=%u clears=%u vk_barriers=%u "
         "draws_skipped=%u dispatch_skipped=%u dropped=%u%s",
         static_cast<unsigned long long>(d->submits), cbs, st.passes, st.draws, st.dispatches, st.copies, st.clears,
         barriers, st.skippedDraws, st.skippedDispatches, st.dropped, skipBy.c_str());
  }
  if (perfSent) {  // vk.perf (perf.h): relaxed atomics, one clock read per submit
    PerfCounters& pc = perf();
    const uint64_t submitUs = perfUsSince(perfT0);  // all but the signal/fence bookkeeping below
    perfAdd(pc.frames, 1);
    perfAdd(pc.submitUs, submitUs), vpgpu::raiseMax(pc.submitUsMax, submitUs);
    perfAdd(pc.translateUs, perfTranslateUs), vpgpu::raiseMax(pc.translateUsMax, perfTranslateUs);
    perfAdd(pc.backendUs, perfBackendUs), vpgpu::raiseMax(pc.backendUsMax, perfBackendUs);
    const uint64_t compileUs = uint64_t(perfReport.cpuCompileMs * 1000.0);
    perfAdd(pc.compileUs, compileUs), vpgpu::raiseMax(pc.compileUsMax, compileUs);
    perfAdd(pc.dispatches, st.dispatches), perfAdd(pc.draws, st.draws), perfAdd(pc.drawsInstanced, st.instancedDraws);
    perfAdd(pc.indirectGpu, st.indirectGpu), perfAdd(pc.indirectCpu, st.indirectCpu);
    perfAdd(pc.indirectCpuSecondary, st.indirectCpuSecondary);
    perfAdd(pc.barriers, perfReport.barriers), perfAdd(pc.barriersComputeDraw, perfReport.compile.barriersComputeDraw);
    // vk.hitch (perf.h): this frame's dt is the time since the previous frame sent
    const std::string hl = d->hitch.frame(HitchLine::Clock::now(), {d->submits, submitUs, perfTranslateUs, perfBackendUs, st.uploadBytes});
    if (!hl.empty()) logf("%s", hl.c_str());
  }
  perfTick(d);
  if (pdump.on) {
    const std::string sum = passDumpSummary();
    if (!sum.empty()) log(sum);
  }
  const uint64_t frames = d->gpu->submittedFrames();
  pendingFrames(*d, frames);
  for (size_t i = 0; i < count; ++i)
    for (const auto& [s, v] : list[i].signals) {  // reached once this frame finishes
      std::lock_guard<std::mutex> l(s->m);
      advanceTimeline(*d->platform, *s);  // a counter nobody polls would keep every signal ever made
      s->pending.push_back({v, frames});
    }
  if (!st.queries.empty()) applyQueryEvents(st.queries, frames);
  for (const auto& [ev, value] : st.events) {  // vkCmdSetEvent / vkCmdResetEvent, in order
    std::lock_guard<std::mutex> l(ev->m);
    ev->frame = frames + 1;  // +1: 0 means nothing pending (see Event)
    ev->value = value;
    ++ev->cmdStamps, ev->stampNs = pendingNowNs();
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

// Runs the ready batches of d->deferred (submitMutex held). A batch is ready when its waits can be met and every batch
// submitted before it to the same VkQueue is ready too: each queue keeps its own order, and a queue held back on a wait
// nothing can meet yet does not hold back the others. The ready batches keep their submission order among themselves
// (they go to the one Metal queue in that order, which is what makes cross-queue timeline waits satisfiable).
void drainDeferred(Device* d) {
  while (!d->deferred.empty()) {
    std::vector<std::pair<Semaphore*, uint64_t>> ahead;
    uint32_t blocked = 0;  // queues with a batch that is not ready
    std::vector<bool> run(d->deferred.size(), false);
    size_t ready = 0;
    for (size_t i = 0; i < d->deferred.size(); ++i) {
      const DeferredBatch& b = d->deferred[i];
      if (blocked >> b.queue & 1) continue;
      bool met = true;
      for (const auto& [s, v] : b.waits) met = met && waitMet(s, v, ahead);
      if (!met) {
        blocked |= 1u << b.queue;
        continue;
      }
      ahead.insert(ahead.end(), b.signals.begin(), b.signals.end());
      run[i] = true;
      ++ready;
    }
    if (!ready) return;
    if (ready != d->deferred.size()) {  // ready batches first, both groups in submission order
      std::vector<DeferredBatch> ordered;
      ordered.reserve(d->deferred.size());
      for (size_t i = 0; i < d->deferred.size(); ++i)
        if (run[i]) ordered.push_back(std::move(d->deferred[i]));
      for (size_t i = 0; i < d->deferred.size(); ++i)
        if (!run[i]) ordered.push_back(std::move(d->deferred[i]));
      d->deferred = std::move(ordered);
    }
    runBatches(d, d->deferred, ready);
  }
}
}  // namespace

void submitDeferred(Device* d) {
  parkWhileBackground(d);  // park before the lock: a parked thread must not hold submitMutex
  std::lock_guard<std::mutex> lock(d->submitMutex);
  if (!d->deferred.empty() && !d->platform->lost()) drainDeferred(d);
}

// submitMutex held, and NOT inside an encodeNative callback: settles the pending image `key` now, with a frame of its own that goes in
// the engine queue in order (before anything submitted later, and so before the head of a native submit). False: not pending (any more).
bool zeroClearPending(Device* d, uint64_t key) {
  auto it = d->zeroPending.find(key);
  if (it == d->zeroPending.end()) return false;
  ZeroLog zl;
  zl.dev = d;
  ZeroTouch t;  // a read first: settles it, with a clear unless a full overwrite already covered it
  t.tex = it->second.id;
  zl.touches.push_back(t);
  vpgpu::Frame zeros;
  const size_t n = zeroInitResolve(*d, zl, zeros);
  if (n) d->gpu->submit(zeros);
  if (d->zeroLogLines++ < 64) logf("vk.zeroinit host images=%zu pending=%zu", n, d->zeroPending.size());
  return true;
}
}  // namespace vpvk

// Host queries (vpvk/config.h): the compositor reads game textures outside the Vulkan flow, so it asks
// whether one still holds uninitialised Metal memory. Only textures the host already obtained
// (vkGetMTLTextureMVK / vkExportMetalObjectsEXT) are known; any other answer is 0.
VPVK_EXPORT int vpvk_image_zero_pending(void* mtl) {
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  if (!mtl) return 0;
  // Registry mutex held throughout: vkDestroyDevice takes the device out of the registry first, so it waits for this. Inside a
  // vpvk_native_submit callback this thread holds the submit mutex (registry -> submit order): the device cannot go, no registry.
  std::unique_lock<std::mutex> live;
  vpvk::Device* d;
  if (tInNativeSubmit) {
    d = tNativeDevice;
  } else {
    live = std::unique_lock<std::mutex>(liveDeviceMutex());
    d = liveDevice();
  }
  if (!d) return 0;
  std::lock_guard<std::mutex> l(d->zeroNativeMutex);
  return d->zeroNative.count(mtl) ? 1 : 0;
}

// The host (Klepton) says which VkImages are the eye layer's swapchain images: the only destination the pipeline gate
// (pipelines.cpp) may drop draws from while a pipeline compiles. Called when the image is created, bound or not yet.
// Takes submitMutex (the Image is only safe to touch under it): a creation-time call, never per frame.
// 1 = marked, 0 = no device / null / the image has no texture to mark yet (the mark is kept and applied at bind).
VPVK_EXPORT int vpvk_mark_eye_image(void* image) {
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  if (!image) return 0;
  std::unique_lock<std::mutex> lock;
  vpvk::Device* d = liveDeviceSubmitLocked(lock);
  if (!d) return 0;
  vpvk::Image* i = vpvk::from<vpvk::Image>(reinterpret_cast<VkImage>(image));
  i->eye = true;
  for (auto& p : i->planes) p->eye = true;
  // The host reads its MTLTexture itself: never a tile-only target (vk.msaa_tile defers 1x attachments). Not bound yet:
  // persistent before the texture exists; bound and deferred: obtained (and so pinned) now.
  if (!i->desc.persistent && !i->desc.transient && i->desc.samples == 1) {
    if (!i->id) {
      i->desc.persistent = true;
    } else {
      i->native = d->platform->nativeTexture(i->id);
      vpvk::zeroNoteNative(*d, *i, i->native);
    }
  }
  vpvk::markEyeTexture(*d, i->id);
  d->eyeSize.store(vpvk::packEyeSize(i->desc.width, i->desc.height, i->desc.layers), std::memory_order_relaxed);
  static std::atomic<unsigned> n{0};
  if (n.fetch_add(1) < 16)
    vpvk::logf("vk.eye_image marked=%dx%d layers=%u samples=%u bound=%d", i->desc.width, i->desc.height, i->desc.layers,
               i->desc.samples, i->id ? 1 : 0);
  return i->id ? 1 : 0;
}

// Never blocks: if a submit (or a native submit) holds the mutex it returns -1 and the caller retries
// next frame. Under submitMutex a game write is either already translated (then the image is no longer
// pending and nothing is cleared) or still to come, behind the clear in the queue (so it overwrites it).
// Inside a vpvk_native_submit callback (this thread holds the mutex) the clear cannot be ordered: the callback is encoding the native
// command buffer that follows a head already fenced on the frames, so a frame submitted from there would run unordered against what the
// callback encodes (and would use up the "native work since the last frame" mark of the backend). Nothing is submitted there: the image
// is queued, -1 is answered (not cleared: do not read it in this callback) and the next vpvk_native_submit clears it before it makes its
// own command buffers; it then reads as not pending. 0 for a texture that is unknown or no longer pending, as everywhere.
VPVK_EXPORT int vpvk_image_zero_clear(void* mtl) {
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  if (!mtl) return 0;
  std::unique_lock<std::mutex> lock;
  vpvk::Device* d = liveDeviceSubmitLocked(lock, true);
  if (!d) return 0;
  const bool inNative = tInNativeSubmit;
  if (!lock.owns_lock() && !inNative) return -1;  // (inside a vpvk_native_submit callback this thread holds the submit mutex)
  uint64_t key;
  {
    std::lock_guard<std::mutex> l(d->zeroNativeMutex);
    auto it = d->zeroNative.find(mtl);
    if (it == d->zeroNative.end()) return 0;
    key = it->second;
  }
  if (inNative) {
    if (!d->zeroPending.count(key)) return 0;
    if (std::find(d->zeroClearDeferred.begin(), d->zeroClearDeferred.end(), key) == d->zeroClearDeferred.end())
      d->zeroClearDeferred.push_back(key);
    return -1;
  }
  return vpvk::zeroClearPending(d, key) ? 1 : 0;
}


VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(VkQueue q, uint32_t n, const VkSubmitInfo* si,
                                                         VkFence fence) {
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  Device* d = from<Queue>(q)->device;
  const auto hitchLock0 = std::chrono::steady_clock::now();  // vk.hitch lock_ms: the gate park + the submit lock
  parkWhileBackground(d);  // park before the lock (a parked thread must not hold submitMutex), for the queue's device, not the live one
  std::lock_guard<std::mutex> lock(d->submitMutex);
  {
    const uint64_t us = perfUsSince(hitchLock0);
    vpgpu::hitchPerf().lockUs.fetch_add(us, std::memory_order_relaxed);
    vpgpu::hitchPerf().engineUs.fetch_add(us, std::memory_order_relaxed);
  }
  if (d->platform->lost()) return VK_ERROR_DEVICE_LOST;
  const size_t first = d->deferred.size();
  for (uint32_t i = 0; i < n; ++i) {
    DeferredBatch b;
    b.queue = from<Queue>(q)->index;
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
    if (d->deferred.size() == first) d->deferred.emplace_back().queue = from<Queue>(q)->index;  // n == 0: the fence alone
    Fence* fe = from<Fence>(fence);
    fe->deferred = true;  // until its batch runs (cleared by runBatches)
    fe->submitNs = pendingNowNs();
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
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  if (!fn) return 0;
  parkWhileBackground();  // park before liveDeviceMutex/submitMutex: a parked thread must not hold them
  // The submit mutex is waited for without the registry (liveDeviceSubmitLocked): a frame held for its pipelines (gpu.hold) must not
  // stop the per-frame calls that take the registry. The callback may call vpvk_native_device (which then does not take it); it must not
  // call vkQueueSubmit (same mutex), nor vpvk_native_submit / vpvk_mark_eye_image again (refused: 0).
  std::unique_lock<std::mutex> lock;
  Device* d = liveDeviceSubmitLocked(lock);
  if (!d || !d->platform) return 0;
  // Clears the callbacks of earlier native submits asked for (vpvk_image_zero_clear): frames of the queue, so before this one's head.
  if (!d->zeroClearDeferred.empty()) {
    std::vector<uint64_t> keys;
    keys.swap(d->zeroClearDeferred);
    for (uint64_t key : keys) vpvk::zeroClearPending(d, key);
  }
  bool ok;
  {
    // Restored whichever way the callback leaves (an exception through the platform's encode, a longjmp-free unwind): a flag left set
    // would make every later host call on this thread believe it still holds the submit mutex.
    struct NativeScope {
      NativeScope(Device* dev) { tInNativeSubmit = true, tNativeDevice = dev; }
      ~NativeScope() { tNativeDevice = nullptr, tInNativeSubmit = false; }
    } scope(d);
    ok = d->platform->encodeNative(fn, user);
  }
  // The first success and the first few failures (a failure would repeat every frame).
  static std::atomic<unsigned> okLogged{0}, failLogged{0};
  if ((ok ? okLogged : failLogged).fetch_add(1) < (ok ? 1u : 8u))
    logf("vk.native submit ok=%d after_submit=%llu", ok ? 1 : 0, static_cast<unsigned long long>(d->submits));
  return ok ? 1 : 0;
}

// VPKTAA nativo: solo dentro del callback de vpvk_native_submit (mismo hilo, submitMutex tomado).
VPVK_EXPORT int vpvk_taa_native_encode(const vpvk_taa_native_frame* f, void* cb) {
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  Device* d = tNativeDevice;
  const bool rcasOnly = f && f->rcas_only != 0;  // sin TAA este frame: solo RCAS
  if (!f || !cb || !d || !d->platform || !(rcasOnly ? rcasOn() : taaNativeOn())) {
    static std::atomic<unsigned> k{0};
    if (k.fetch_add(1) < 4)
      logf("vk.taa native refused frame=%d cb=%d in_native_submit=%d enabled=%d rcas_only=%d rcas=%.2f", f ? 1 : 0, cb ? 1 : 0,
           d ? 1 : 0, taaNativeOn() ? 1 : 0, rcasOnly ? 1 : 0, double(rcasIntensity()));
    return 0;
  }
  const TaaNativeApplied a = taaNativeTake();
  vpvk_config c;
  vpvk_config_get(&c);
  Platform::TaaNativeArgs args;
  // Depth the host did not give (vpgles, OpenXR: no host-side eye table): that of the eye pass the Translator
  // saw drawing the eye's colour (taa::DrawClassifier's pass record), with the depth convention it observed.
  vpvk_taa_native_frame fr = *f;
  {
    UiMaskState& em = d->uiMask;
    const uint32_t n = std::min(em.eyePasses, UiMaskState::kEyePasses);
    bool filled = false, eye0Own = false, rev = fr.reversed_z != 0;
    for (int e = 0; e < 2 && !rcasOnly; ++e) {  // RCAS sin TAA no usa profundidad
      if (fr.eye[e].depth) continue;
      for (uint32_t k = 0; k < n; ++k) {  // newest first
        const UiMaskState::EyePass& r = em.eyePass[(em.eyePasses - 1 - k) % UiMaskState::kEyePasses];
        void* key = fr.eye[e].color_base ? fr.eye[e].color_base : fr.eye[e].color;
        // A multisampled eye colour can never be the resolved texture `key`, and looking it up would give a
        // tile-only MSAA target its memory (vk.msaa_tile): skip it.
        if (vpgpu::TextureDesc cd; d->gpu->textureDesc(r.color, cd) && cd.samples > 1) continue;
        if (!key || d->platform->nativeTexture(r.color) != key) continue;
        if (r.layers < 2 && r.colorLayer != fr.eye[e].color_slice) continue;
        void* depth = d->platform->nativeTexture(r.depth);
        if (!depth) continue;
        fr.eye[e].depth = depth;
        fr.eye[e].depth_slice = r.layers >= 2 ? r.depthLayer + uint32_t(e) : r.depthLayer;
        if (e == 0) rev = taa::reversedZ(r.cleared, r.clearValue, r.greater, r.less, rev), eye0Own = true;
        filled = true;
        break;
      }
    }
    if (filled) {
      if (eye0Own) fr.reversed_z = rev ? 1 : 0;
      static std::atomic<unsigned> k{0};
      if (k.fetch_add(1) < 4)
        logf("vk.taa native depth from the eye pass (host gave none): eye0=%d eye1=%d reversed_z=%d", fr.eye[0].depth ? 1 : 0,
             fr.eye[1].depth ? 1 : 0, fr.reversed_z);
    }
    em.eyePasses = 0;
  }
  args.frame = &fr;
  args.jx = a.jx, args.jy = a.jy;
  args.targetW = a.targetW, args.targetH = a.targetH;
  args.jittered = a.jittered > 0 && !rcasOnly;
  args.percent = c.vpktaa.percent, args.sharpness = rcasIntensity();
  args.rcasOnly = rcasOnly, args.rcasDenoise = rcasDenoise();
  args.phase = a.phase;
  // VPKTAA UI mask: the R8 textures the eye passes of this frame drew UI into (none: no UI this
  // frame, or the mask is off). One pass with both layers (multiview) serves both eyes.
  UiMaskState& um = d->uiMask;
  if (taaUiMaskOn()) {
    const bool shared = um.live[0].valid && um.live[0].t.layers >= 2;
    for (int e = 0; e < 2; ++e) {
      const UiMaskState::Live& l = shared ? um.live[0] : um.live[e];
      if (!l.valid || !l.t.tex) continue;
      args.mask[e].texture = d->platform->nativeTexture(l.t.tex);
      args.mask[e].slice = shared ? uint32_t(e) : 0u;
    }
  }
  um.live[0].valid = um.live[1].valid = false;
  um.maskedPasses = 0;
  {
    // «RCAS activo»: una vez, y otra vez cuando cambia el modo (con TAA / solo RCAS), la intensidad o la máscara.
    static std::atomic<uint64_t> last{~0ull};
    const uint64_t key = (rcasOnly ? 1u : 0u) | (args.mask[0].texture || args.mask[1].texture ? 2u : 0u) |
                         (uint64_t(args.sharpness * 100.0f + 0.5f) << 8);
    if (last.exchange(key) != key)
      logf("vk.rcas active intensity=%.2f stops=%.2f denoise=%d mode=%s ui_mask=%s", double(args.sharpness),
           double(2.0f * (1.0f - args.sharpness)), args.rcasDenoise ? 1 : 0, rcasOnly ? "rcas_only(no TAA)" : "host_sharpens(VPKTAA output unsharpened)",
           args.mask[0].texture || args.mask[1].texture ? "yes" : "no(no UI draws this frame, MSAA pass or mask off)");
  }
  const bool encoded = d->platform->taaNativeEncode(args, cb);
  if (encoded && a.jittered && !rcasOnly) taaNativeAdvance(a.phase);  // la fase se gasta solo si el pase se codificó
  return encoded ? 1 : 0;
}

VPVK_EXPORT void* vpvk_depth_resolve_encode(void* cb, void* depth, int reversedZ, int mode, int key) {
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  Device* d = tNativeDevice;
  if (!cb || !depth || !d || !d->platform) return nullptr;
  return d->platform->depthResolveEncode(cb, depth, reversedZ != 0, mode, key);
}

VPVK_EXPORT size_t vpvk_mem_stats(char* out, size_t cap) {
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  if (!out || !cap) return 0;
  std::string s, ledger;
  {
    std::lock_guard<std::mutex> live(liveDeviceMutex());
    Device* d = liveDevice();
    if (!d || !d->platform) return 0;
    s = d->platform->memStats();
    // The ledger (" ledger: ...", the longest part) stays last, so a short buffer cuts it rather than the counters.
    if (const size_t at = s.find(" ledger: "); at != std::string::npos) ledger = s.substr(at), s.resize(at);
    if (d->recipes && d->recipes->enabled()) {
      char b[96];
      std::snprintf(b, sizeof b, "%srecipes_bytes=%llu recipes_pipes=%zu recipes_shaders=%zu", s.empty() ? "" : " ",
                    (unsigned long long)d->recipes->memoryBytes(), d->recipes->pipelineCount(),
                    d->recipes->shaderCount());
      s += b;
    }
  }
  s += platformAttrSummary();  // walks the Metal object tables: outside liveDeviceMutex (taken by per-frame calls)
  s += ledger;
  const size_t n = std::min(s.size(), cap - 1);
  std::memcpy(out, s.data(), n);
  out[n] = 0;
  return n;
}

VPVK_EXPORT size_t vpvk_mem_brief(char* out, size_t cap) {
  if (!out || !cap) return 0;
  std::unique_lock<std::mutex> live(liveDeviceMutex(), std::try_to_lock);
  if (!live.owns_lock()) return 0;  // a trim or a per-frame call holds it: skip this sample
  Device* d = liveDevice();
  if (!d || !d->platform) return 0;
  vpgpu::ScopedPool pool;  // host thread: no autorelease pool of its own (see scoped_pool.h)
  std::string s = d->platform->memStats();
  if (const size_t at = s.find(" ledger: "); at != std::string::npos) s.resize(at);
  const size_t n = std::min(s.size(), cap - 1);
  std::memcpy(out, s.data(), n);
  out[n] = 0;
  return n;
}

VPVK_EXPORT unsigned vpvk_gpu_backlog_ms(void) {
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  std::lock_guard<std::mutex> live(liveDeviceMutex());
  Device* d = liveDevice();
  return d && d->platform ? d->platform->gpuBacklogMs() : 0;
}

VPVK_EXPORT unsigned vpvk_gpu_last_frame_chunks(void) {
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  std::lock_guard<std::mutex> live(liveDeviceMutex());
  Device* d = liveDevice();
  return d && d->platform ? d->platform->lastFrameChunks() : 0;
}

VPVK_EXPORT void vpvk_dump_pending(void) {
  vpgpu::ScopedPool pool;  // host thread: no autorelease pool of its own (see scoped_pool.h)
  std::unique_lock<std::mutex> live(liveDeviceMutex(), std::try_to_lock);
  if (!live.owns_lock()) {  // a stalled thread may hold it: the report must not wait
    logf("vk.pending device_mutex=BUSY");
    return;
  }
  Device* d = liveDevice();
  if (!d || !d->platform) return;
  for (const std::string& l : pendingReport(*d)) log(l);
  d->platform->dumpInFlight("vk_pending");
}

// Host (dlsym): whether the game's shader cache looks complete (see VpvkCacheState in vpvk.h). Poll it; every change of
// complete / reason is also logged ("vk.cache_state"). 1 = filled from the live device; 0 = no device or null `out`.
VPVK_EXPORT int vpvk_cache_state(VpvkCacheState* out) {
  if (!out) return 0;
  *out = VpvkCacheState{};
  out->version = kCacheStateVersion;
  std::lock_guard<std::mutex> live(liveDeviceMutex());
  Device* d = liveDevice();
  if (!d || !d->gpu) return 0;
  cacheState(d, *out);
  return 1;
}

VPVK_EXPORT int vpvk_gpu_lost(void) {
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  std::lock_guard<std::mutex> live(liveDeviceMutex());
  Device* d = liveDevice();
  return d && d->platform && d->platform->lost() ? 1 : 0;
}

// For a host's prewarm device (no game running): blocks until every stage its vkCreate*Pipelines calls queued was
// translated, every shader library and pipeline compiled, then writes the binary archive, so retiring the device drops
// nothing. Returns the milliseconds waited. Found with dlsym (kl_fossilize.cpp). The recipe prewarm thread is not waited
// for: it compiles the earlier runs' recipes, which the host's own replay makes again.
VPVK_EXPORT uint32_t vpvk_device_wait_compiles(VkDevice dv) {
  Device* d = dv ? from<Device>(dv) : nullptr;
  if (!d || !d->gpu || !d->platform) return 0;
  vpgpu::ScopedPool pool;  // host thread: no autorelease pool of its own (see scoped_pool.h)
  const auto t0 = std::chrono::steady_clock::now();
  while (d->translator && d->translator->pending()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  d->platform->drainCompiles();
  d->gpu->pipelines().waitIdle();
  d->platform->drainCompiles();  // libraries the last pipelines asked for
  d->platform->flushCaches(true);  // the host's replay drain: nobody waits, the archive must be complete
  psoCensus(d, true);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  const vpgpu::PipelineStats t = d->gpu->pipelines().totals();
  logf("vk.prewarm drained ms=%.0f pipelines compiled=%u failed=%u hits=%u", ms, t.compiled, t.failed, t.hits);
  return uint32_t(ms);
}

VPVK_EXPORT void vpvk_hide_index_count(unsigned n) {
  static std::mutex mu;     // writers only (the draw path just reads the slots)
  static unsigned next = 0;
  std::lock_guard<std::mutex> lock(mu);
  if (!n) {
    for (auto& h : gHideIndexCount) h.store(0);
    logf("vk.hide_mesh: off");
    return;
  }
  for (const auto& h : gHideIndexCount) if (h.load() == n) return;   // already there: no log
  gHideIndexCount[next].store(n);
  next = (next + 1) % kHideSlots;
  logf("vk.hide_mesh: icount=%u registered (direct indexed draws that write colour with this index count are dropped; up to %d counts, 0 = off)", n, kHideSlots);
}

VPVK_EXPORT void vpvk_texture_budget(unsigned max_dim) {
  if (max_dim == textureBudget()) return;
  setTextureBudget(max_dim);
  logf("vk.tex_budget cap=%u (0 = none): plain sampled textures created from here keep levels of at most this size", max_dim);
}

VPVK_EXPORT size_t vpvk_trim_memory(int level, char* out, size_t cap) {
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  if (out && cap) out[0] = 0;
  std::unique_lock<std::mutex> live(liveDeviceMutex());
  Device* d = liveDevice();
  if (!d || !d->platform) return 0;
  std::string report;
  size_t freed = d->platform->trimMemory(level, report);
  // Also what the guest destroyed and no frame in flight reads, and the MSAA backings / argument buffers trimMemory could
  // only queue. Nothing is ever blocked on: the submit mutex is try-locked (order live -> submit, as vpvk_native_submit) and
  // retried for ~50 ms, with liveDeviceMutex RELEASED while waiting between attempts (a sleeping holder would stall every
  // per-frame call that takes it) and the device fetched again each time. With frames running the submit mutex is free for
  // gaps of a few ms, and one attempt per second (the caller's rate) almost never caught one, so the trim freed 0.
  std::unique_lock<std::mutex> lock;
  for (int i = 0;; ++i) {
    if (live.owns_lock()) {
      Device* cur = liveDevice();
      if (cur != d || !cur || !cur->platform) {  // the device went away (or was replaced) while we waited
        report += " device_gone=1";
        d = nullptr;
        break;
      }
      lock = std::unique_lock<std::mutex>(d->submitMutex, std::try_to_lock);
      if (lock.owns_lock()) break;
      live.unlock();
    }
    if (i >= 50) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));  // no lock held here
    (void)live.try_lock();  // busy: counts as a failed attempt
  }
  if (d && lock.owns_lock()) {
    drainRetiredChunks(d);  // (arena chunks command buffer resets gave up: freed here as at the next submission)
    d->platform->trimSerialized(report);  // moves to the graveyard; the drain below returns what no frame reads
    std::string drained;
    freed += d->platform->drainRetired(0, drained);
    if (!drained.empty()) report += " " + drained;
  } else if (d) {
    report += " submit_busy=1";
  }
  if (out && cap) {
    const size_t n = std::min(report.size(), cap - 1);
    std::memcpy(out, report.data(), n);
    out[n] = 0;
  }
  return freed;
}

// Frees the objects the guest destroyed that the backend keeps until a frame reuses a slot, those no frame in flight
// can still read (min_idle_ms no longer gates it: a completed frame's leftovers go while others run). Never blocks:
// the submit mutex is only try-locked (a busy device means "not now"). Any thread. Writes a short report; returns
// the bytes released.
VPVK_EXPORT size_t vpvk_idle_drain(unsigned min_idle_ms, char* out, size_t cap) {
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  if (out && cap) out[0] = 0;
  std::lock_guard<std::mutex> live(liveDeviceMutex());
  Device* d = liveDevice();
  if (!d || !d->platform) return 0;
  std::unique_lock<std::mutex> lock(d->submitMutex, std::try_to_lock);
  if (!lock.owns_lock()) return 0;
  drainRetiredChunks(d);  // (arena chunks command buffer resets gave up)
  std::string report;
  const size_t freed = d->platform->drainRetired(min_idle_ms, report);
  lock.unlock();
  if (out && cap) {
    const size_t n = std::min(report.size(), cap - 1);
    std::memcpy(out, report.data(), n);
    out[n] = 0;
  }
  return freed;
}

// One line of deltas since the previous call with the same `which` (0: the Vulkan layer, 1: the backend), as
// "dt=<ms> key=<delta> ...", only the keys that moved (gauges, "g_*", when non-zero). 0 (out untouched) when
// nothing moved or there is no device. Cheap and lock-free apart from a tiny mutex for the previous values.
VPVK_EXPORT size_t vpvk_diag_line(int which, char* out, size_t cap) {
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  if (!out || !cap || which < 0 || which > 1) return 0;
  std::vector<std::pair<std::string, uint64_t>> now;
  if (which == 0) {
    auto& c = diag::c();
    auto put = [&now](const std::string& k, const std::atomic<uint64_t>& v) {
      now.emplace_back(k, v.load(std::memory_order_relaxed));
    };
    for (unsigned t = 0; t < diag::kTypes; ++t) {
      const std::string ts = std::to_string(t);
      put("alloc" + ts, c.allocN[t]);
      put("alloc" + ts + "_b", c.allocB[t]);
      put("free" + ts, c.freeN[t]);
      put("free" + ts + "_b", c.freeB[t]);
    }
    put("buf_create", c.bufCreateN);
    put("buf_create_b", c.bufCreateB);
    put("buf_destroy", c.bufDestroyN);
    put("buf_bind", c.bufBindN);
    put("buf_bind_b", c.bufBindB);
    put("img_create", c.imgCreateN);
    put("img_create_px", c.imgCreatePx);
    put("img_destroy", c.imgDestroyN);
    put("img_bind", c.imgBindN);
    put("img_create_linear", c.imgLinearN);
    put("img_create_optimal", c.imgOptimalN);
    put("img_rel_destroy", c.imgRelDestroyN);
    put("img_rel_shared", c.imgRelSharedN);
    put("img_rel_notex", c.imgRelNoTexN);
    put("view_create", c.viewCreateN);
    put("view_destroy", c.viewDestroyN);
    put("map", c.mapN);
    put("map_b", c.mapB);
    put("unmap", c.unmapN);
    put("flush", c.flushN);
    put("flush_b", c.flushB);
    put("inval", c.invN);
    put("inval_b", c.invB);
  } else {
    std::lock_guard<std::mutex> live(liveDeviceMutex());
    Device* d = liveDevice();
    if (!d || !d->platform) return 0;
    std::vector<std::pair<const char*, uint64_t>> v;
    d->platform->diagCounters(v);
    for (auto& kv : v) now.emplace_back(kv.first, kv.second);
  }
  static std::mutex mu;
  static std::unordered_map<std::string, uint64_t> prev[2];
  static std::chrono::steady_clock::time_point last[2];
  std::lock_guard<std::mutex> lock(mu);
  const auto t = std::chrono::steady_clock::now();
  const long long dt = last[which].time_since_epoch().count()
                           ? std::chrono::duration_cast<std::chrono::milliseconds>(t - last[which]).count()
                           : 0;
  last[which] = t;
  std::string line;
  for (auto& [k, v] : now) {
    const bool gauge = k.compare(0, 2, "g_") == 0;
    uint64_t& p = prev[which][k];
    const uint64_t delta = gauge ? v : (v >= p ? v - p : 0);
    p = v;
    if (!delta) continue;
    char b[64];
    std::snprintf(b, sizeof b, " %s=%llu", k.c_str(), static_cast<unsigned long long>(delta));
    line += b;
  }
  if (line.empty()) return 0;
  char head[32];
  std::snprintf(head, sizeof head, "dt=%lld", dt);
  line.insert(0, head);
  const size_t n = std::min(line.size(), cap - 1);
  std::memcpy(out, line.data(), n);
  out[n] = 0;
  return n;
}

// Memory ownership: the (address, size) ranges of kind 0 = the live host-visible VkDeviceMemory host allocations (what
// the guest maps), 1 = the CPU-visible buffers VPKEngine's backend allocated itself. Fills out[0..max) sorted by
// address and returns the total (may exceed max). Any thread; takes a short mutex, never a submit lock.
VPVK_EXPORT size_t vpvk_diag_ranges(int kind, vpvk_range* out, size_t max) {
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  std::vector<std::pair<uintptr_t, uint64_t>> v;
  if (kind == 0) {
    std::lock_guard<std::mutex> l(diag::ranges().mu);
    v.assign(diag::ranges().r.begin(), diag::ranges().r.end());
  } else if (kind == 1) {
    std::lock_guard<std::mutex> live(liveDeviceMutex());
    Device* d = liveDevice();
    if (d && d->platform) d->platform->diagRanges(v);
    std::sort(v.begin(), v.end());
  } else {
    return 0;
  }
  if (out)
    for (size_t i = 0; i < v.size() && i < max; ++i) out[i] = vpvk_range{v[i].first, v[i].second};
  return v.size();
}

VPVK_EXPORT void* vpvk_ui_mask_texture(int eye, int* w, int* h, int* slice) {
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  if (w) *w = 0;
  if (h) *h = 0;
  if (slice) *slice = 0;
  if (eye < 0 || eye > 1) return nullptr;
  // Inside a vpvk_native_submit callback this thread holds the submit mutex (registry -> submit order): the device cannot go and the
  // registry must not be taken (a thread holding it may be waiting for that mutex).
  std::unique_lock<std::mutex> live;
  Device* d;
  if (tInNativeSubmit) {
    d = tNativeDevice;
  } else {
    live = std::unique_lock<std::mutex>(liveDeviceMutex());
    d = liveDevice();
  }
  if (!d || !d->platform) return nullptr;
  UiMaskState& u = d->uiMask;
  std::lock_guard<std::mutex> l(u.lastMu);
  const UiMaskState::Published& p = u.last[eye];
  if (!p.tex) return nullptr;
  void* r = d->platform->retainNative(p.tex);
  if (!r) return nullptr;
  if (w) *w = int(p.w);
  if (h) *h = int(p.h);
  if (slice) *slice = int(p.slice);
  return r;
}

VPVK_EXPORT void* vpvk_native_device(void) {
  vpgpu::ScopedPool pool;  // guest thread: no autorelease pool of its own (see scoped_pool.h)
  std::unique_lock<std::mutex> live;
  Device* d;
  if (tInNativeSubmit) {  // in the callback (submit mutex held by this thread): no registry, see vpvk_ui_mask_texture
    d = tNativeDevice;
  } else {
    live = std::unique_lock<std::mutex>(liveDeviceMutex());
    d = liveDevice();
  }
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

static PFN_vkVoidFunction lookupRaw(const char* name, bool deviceLevel) {
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

// What the game (and vpgles) calls is the capture hook of the entry point (capture.cpp): one atomic load while idle.
PFN_vkVoidFunction vpvk::lookup(const char* name, bool deviceLevel) {
  return captureWrap(name, lookupRaw(name, deviceLevel));
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
