// VPVulkan tests on the null backend. Usage: vpvk_tests <group> <path to libVPVulkan>.
#include "vpvk/config.h"
#include <dlfcn.h>
#include <unistd.h>  // mkdtemp on macOS (libc++ <cstdlib> does not declare it)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include "commands.h"
#include "foveation.h"
#include "taa_native.h"
#include <cmath>
#include "vpvk.h"

using namespace vpvk;

#include "vpgpu/metal2/archive_index.h"

// MoltenVK-compatible export (entry.cpp); not in any header on purpose (see vpvk/config.h).
extern "C" void vkGetMTLTextureMVK(VkImage, void**);

namespace {

int failures = 0;
const char* gLib = nullptr;
#define CHECK(cond)                                                 \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                   \
    }                                                               \
  } while (0)

// The entry points kl_vulkan.c resolves (through vkGetInstanceProcAddr / vkGetDeviceProcAddr, or
// dlsym on the driver) that this driver implements today. Must all resolve.
const char* const kPhase1[] = {
    "vkCreateInstance", "vkEnumerateInstanceVersion", "vkEnumerateInstanceExtensionProperties",
    "vkEnumeratePhysicalDevices", "vkGetPhysicalDeviceProperties", "vkGetPhysicalDeviceProperties2",
    "vkGetPhysicalDeviceProperties2KHR", "vkGetPhysicalDeviceFeatures2KHR",
    "vkGetPhysicalDeviceMemoryProperties", "vkGetPhysicalDeviceFormatProperties2KHR",
    "vkGetPhysicalDeviceImageFormatProperties2KHR", "vkGetPhysicalDeviceQueueFamilyProperties",
    "vkEnumerateDeviceExtensionProperties", "vkEnumerateDeviceLayerProperties", "vkCreateDevice",
};
const char* const kDevice[] = {
    "vkGetDeviceQueue", "vkCreateImage", "vkDestroyImage", "vkGetImageMemoryRequirements",
    "vkAllocateMemory", "vkFreeMemory", "vkBindImageMemory", "vkCreateBuffer", "vkDestroyBuffer",
    "vkGetBufferMemoryRequirements", "vkBindBufferMemory", "vkMapMemory", "vkUnmapMemory",
    "vkInvalidateMappedMemoryRanges", "vkCreateCommandPool", "vkDestroyCommandPool",
    "vkAllocateCommandBuffers", "vkBeginCommandBuffer", "vkEndCommandBuffer", "vkResetCommandBuffer",
    "vkCmdPipelineBarrier", "vkCmdClearColorImage", "vkQueueSubmit", "vkCreateFence",
    "vkDestroyFence", "vkWaitForFences", "vkResetFences", "vkDeviceWaitIdle", "vkQueueWaitIdle",
    "vkExportMetalObjectsEXT", "vkCreateImageView", "vkCreateRenderPass", "vkCreateRenderPass2KHR",
    "vkCmdBeginRenderPass", "vkCmdBeginRenderPass2KHR", "vkCmdCopyBufferToImage", "vkCmdCopyBuffer",
};

void testExports() {
  void* h = dlopen(gLib, RTLD_NOW | RTLD_LOCAL);
  CHECK(h != nullptr);
  if (!h) {
    std::printf("  dlopen: %s\n", dlerror());
    return;
  }
  auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(h, "vkGetInstanceProcAddr"));
  CHECK(gipa && dlsym(h, "vkGetDeviceProcAddr") && dlsym(h, "vk_icdGetInstanceProcAddr"));
  CHECK(dlsym(h, "vkCmdCopyBufferToImage") != nullptr);  // kl_vulkan.c dlsyms some vkCmd* directly
  CHECK(dlsym(h, "vkGetMTLTextureMVK") != nullptr);      // klvk_overlay_mtl_texture (MoltenVK API)
  CHECK(dlsym(h, "vpvk_msaa_detected") && dlsym(h, "vpvk_msaa_device_max") && dlsym(h, "vpvk_config_set_msaa_override"));
  CHECK(dlsym(h, "_ZN4vpvk6lookupEPKcb") == nullptr);   // internals stay hidden
  {  // memory trim / UI mask exports: without a live device they answer 0 / NULL and leave out clean
    auto trim = reinterpret_cast<size_t (*)(int, char*, size_t)>(dlsym(h, "vpvk_trim_memory"));
    auto mask = reinterpret_cast<void* (*)(int, int*, int*, int*)>(dlsym(h, "vpvk_ui_mask_texture"));
    CHECK(trim && mask);
    char out[16] = "x";
    CHECK(trim(2, out, sizeof out) == 0 && out[0] == 0 && trim(1, nullptr, 0) == 0);
    int w = 7, hh = 7, sl = 7;
    CHECK(mask(0, &w, &hh, &sl) == nullptr && w == 0 && hh == 0 && sl == 0);
    CHECK(mask(2, nullptr, nullptr, nullptr) == nullptr);
    auto drain = reinterpret_cast<size_t (*)(unsigned, char*, size_t)>(dlsym(h, "vpvk_idle_drain"));
    auto diagLine = reinterpret_cast<size_t (*)(int, char*, size_t)>(dlsym(h, "vpvk_diag_line"));
    CHECK(drain && diagLine);
    char rep[16] = "x";
    CHECK(drain(0, rep, sizeof rep) == 0 && rep[0] == 0 && drain(5, nullptr, 0) == 0);
    char line[16] = "x";
    CHECK(diagLine(0, line, sizeof line) == 0 && diagLine(1, line, sizeof line) == 0 && diagLine(7, line, sizeof line) == 0);
    CHECK(diagLine(0, nullptr, 0) == 0 && line[0] == 'x');
  }
  if (!gipa) return;
  for (const char* n : kPhase1) {
    if (!gipa(nullptr, n)) std::printf("  missing %s\n", n);
    CHECK(gipa(nullptr, n) != nullptr);
  }
  VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  VkInstance inst = nullptr;
  auto create = reinterpret_cast<PFN_vkCreateInstance>(gipa(nullptr, "vkCreateInstance"));
  CHECK(create(&ci, nullptr, &inst) == VK_SUCCESS);
  auto gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(gipa(inst, "vkGetDeviceProcAddr"));
  CHECK(gdpa != nullptr);
  for (const char* n : kDevice) {
    if (!gdpa(nullptr, n)) std::printf("  missing %s\n", n);
    CHECK(gdpa(nullptr, n) != nullptr);
  }
  CHECK(gdpa(nullptr, "vkCreateInstance") == nullptr);  // instance-level: not via device
  CHECK(gipa(inst, "vkNotAFunction") == nullptr);
  // Dispatchable handles carry the ICD loader word.
  CHECK(valid_loader_magic_value(inst));
  reinterpret_cast<PFN_vkDestroyInstance>(gipa(inst, "vkDestroyInstance"))(inst, nullptr);
  dlclose(h);
}

struct Ctx {
  VkInstance inst = nullptr;
  VkPhysicalDevice pd = nullptr;
  VkDevice dev = nullptr;
  VkQueue queue = nullptr;
  explicit Ctx(std::vector<const char*> exts = {}) {
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    vkCreateInstance(&ci, nullptr, &inst);
    uint32_t n = 1;
    vkEnumeratePhysicalDevices(inst, &n, &pd);
    float prio = 1;
    VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    q.queueCount = 1;
    q.pQueuePriorities = &prio;
    VkDeviceCreateInfo d{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    d.queueCreateInfoCount = 1;
    d.pQueueCreateInfos = &q;
    d.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    d.ppEnabledExtensionNames = exts.data();
    vkCreateDevice(pd, &d, nullptr, &dev);
    vkGetDeviceQueue(dev, 0, 0, &queue);
  }
  ~Ctx() {
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
  }
  vpgpu::Device& gpu() { return *from<Device>(dev)->gpu; }
};

void testPhysical() {
  vpvk_config_set_quest_identity(1);  // asserts the Quest 3 / Adreno 740 profile: opt-in
  Ctx c;
  CHECK(c.dev != nullptr);
  VkPhysicalDeviceProperties p;
  vkGetPhysicalDeviceProperties(c.pd, &p);
  CHECK(p.vendorID == 0x5143);
  CHECK(p.limits.maxColorAttachments == 8 && p.limits.maxPushConstantsSize >= 128);
  VkPhysicalDeviceMultiviewFeatures mv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES};
  VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &mv};
  vkGetPhysicalDeviceFeatures2(c.pd, &f);
  CHECK(mv.multiview == VK_TRUE);
  VkPhysicalDeviceMultiviewProperties mvp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES};
  VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &mvp};
  vkGetPhysicalDeviceProperties2(c.pd, &p2);
  CHECK(mvp.maxMultiviewViewCount == 8);
  // Formats: RGBA8 renderable + blendable, sRGB not storage, D24S8 depth, unknown format none.
  VkFormatProperties fp;
  vkGetPhysicalDeviceFormatProperties(c.pd, VK_FORMAT_R8G8B8A8_UNORM, &fp);
  CHECK(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT);
  vkGetPhysicalDeviceFormatProperties(c.pd, VK_FORMAT_R8G8B8A8_SRGB, &fp);
  CHECK(!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT));
  vkGetPhysicalDeviceFormatProperties(c.pd, VK_FORMAT_D24_UNORM_S8_UINT, &fp);
  CHECK(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
  vkGetPhysicalDeviceFormatProperties(c.pd, VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, &fp);
  CHECK(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);
  CHECK(!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT));
  vkGetPhysicalDeviceFormatProperties(c.pd, VK_FORMAT_ASTC_6x6_SRGB_BLOCK, &fp);
  CHECK(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);
  vkGetPhysicalDeviceFormatProperties(c.pd, VK_FORMAT_ASTC_8x6_SRGB_BLOCK, &fp);
  CHECK(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT);  // every ASTC block size
  // No BCn at all (Adreno 740): the format and the feature are both absent.
  vkGetPhysicalDeviceFormatProperties(c.pd, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, &fp);
  CHECK(fp.optimalTilingFeatures == 0 && fp.bufferFeatures == 0);
  vkGetPhysicalDeviceFormatProperties(c.pd, VK_FORMAT_BC7_SRGB_BLOCK, &fp);
  CHECK(fp.optimalTilingFeatures == 0);
  // Mandatory atomics on R32 integer storage images and texel buffers.
  for (VkFormat af : {VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT}) {
    vkGetPhysicalDeviceFormatProperties(c.pd, af, &fp);
    CHECK(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT);
    CHECK(fp.bufferFeatures & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT);
  }
  vkGetPhysicalDeviceFormatProperties(c.pd, VK_FORMAT_R32_SFLOAT, &fp);
  CHECK(!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT));
  // R4G4B4A4 = Metal ABGR4 (same bits): sampled, rendered, blended.
  vkGetPhysicalDeviceFormatProperties(c.pd, VK_FORMAT_R4G4B4A4_UNORM_PACK16, &fp);
  CHECK(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT);
  VkImageFormatProperties ip;
  CHECK(vkGetPhysicalDeviceImageFormatProperties(c.pd, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TYPE_2D,
                                                 VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT, 0,
                                                 &ip) == VK_ERROR_FORMAT_NOT_SUPPORTED);
  // Extension enumeration and device creation with the VR set.
  uint32_t n = 0;
  vkEnumerateDeviceExtensionProperties(c.pd, nullptr, &n, nullptr);
  std::vector<VkExtensionProperties> e(n);
  CHECK(vkEnumerateDeviceExtensionProperties(c.pd, nullptr, &n, e.data()) == VK_SUCCESS);
  uint32_t two = 2;
  CHECK(vkEnumerateDeviceExtensionProperties(c.pd, nullptr, &two, e.data()) == VK_INCOMPLETE);
  // The density map is never advertised (VPKEngine foveation only); enabling it anyway is ignored.
  for (const auto& x : e) CHECK(std::strcmp(x.extensionName, VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME) != 0);
  // Nothing an Adreno driver lacks: VK_EXT_metal_objects is not listed, yet enabling it works (below).
  for (const auto& x : e) CHECK(std::strcmp(x.extensionName, VK_EXT_METAL_OBJECTS_EXTENSION_NAME) != 0);
  Ctx vr({VK_KHR_MULTIVIEW_EXTENSION_NAME, VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME,
          VK_EXT_METAL_OBJECTS_EXTENSION_NAME});
  CHECK(vr.dev && from<Device>(vr.dev)->multiview && !from<Device>(vr.dev)->densityMap && from<Device>(vr.dev)->metalObjects);
  // Host configuration API: clamped, generation bumped.
  uint32_t g0 = vpvk_config_generation();
  vpvk_foveation fv{VPVK_FOVEATION_CUSTOM, 2.0f, 0.0f, 0.1f};
  vpvk_config_set_foveation(&fv);
  vpvk_vpktaa ta{1, 150.0f, 0.4f};
  vpvk_config_set_vpktaa(&ta);
  vpvk_config cfg;
  vpvk_config_get(&cfg);
  CHECK(cfg.foveation.mode == VPVK_FOVEATION_CUSTOM && cfg.foveation.sharp_zone == 1.0f &&
        cfg.foveation.periphery == 0.1f && cfg.vpktaa.enabled == 1 && cfg.vpktaa.percent == 100.0f);
  CHECK(vpvk_config_generation() == g0 + 2);
  VkDeviceCreateInfo bad{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  const char* nope = "VK_KHR_swapchain";
  bad.enabledExtensionCount = 1;
  bad.ppEnabledExtensionNames = &nope;
  VkDevice d2 = nullptr;
  CHECK(vkCreateDevice(c.pd, &bad, nullptr, &d2) == VK_ERROR_EXTENSION_NOT_PRESENT);
}

VkDeviceMemory alloc(Ctx& c, VkDeviceSize size, uint32_t type) {
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = size;
  ai.memoryTypeIndex = type;
  VkDeviceMemory m = VK_NULL_HANDLE;
  vkAllocateMemory(c.dev, &ai, nullptr, &m);
  return m;
}

VkBuffer buffer(Ctx& c, VkDeviceSize size, VkDeviceMemory m, VkDeviceSize off) {
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size;
  bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VkBuffer b;
  vkCreateBuffer(c.dev, &bi, nullptr, &b);
  vkBindBufferMemory(c.dev, b, m, off);
  return b;
}

VkImage image(Ctx& c, VkFormat f, uint32_t w, uint32_t layers, VkImageUsageFlags usage, VkDeviceMemory m) {
  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = VK_IMAGE_TYPE_2D;
  ii.format = f;
  ii.extent = {w, w, 1};
  ii.mipLevels = 1;
  ii.arrayLayers = layers;
  ii.samples = VK_SAMPLE_COUNT_1_BIT;
  ii.tiling = VK_IMAGE_TILING_OPTIMAL;
  ii.usage = usage;
  VkImage i = VK_NULL_HANDLE;
  if (vkCreateImage(c.dev, &ii, nullptr, &i) != VK_SUCCESS) return VK_NULL_HANDLE;
  if (m) vkBindImageMemory(c.dev, i, m, 0);
  return i;
}

void testMemory() {
  vpvk_config_set_quest_identity(1);  // asserts the Quest 3 / Adreno 740 profile: opt-in
  // An earlier group may leave force-1x on (vpktaa enabled): the MSAA block below needs real multisampled images, so
  // the env switch (wins over the config API) is off for this group and put back at its end.
  const char* oldForce = getenv("VPVK_FORCE_1X");
  const std::string savedForce = oldForce ? oldForce : "";
  setenv("VPVK_FORCE_1X", "0", 1);
  Ctx c;
  const auto& mp = memoryProperties();
  CHECK(mp.memoryHeapCount == 1);
  CHECK(mp.memoryTypes[1].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  VkDeviceMemory m = alloc(c, 1 << 20, 1);
  {
    // Ownership ranges (vpvk_diag_ranges): the host-visible allocation is listed while it lives; the null platform
    // has no buffers of its own; an unknown kind is empty.
    auto listed = [&](VkDeviceMemory mem) {
      std::vector<vpvk_range> rg(4096);
      const size_t n = vpvk_diag_ranges(0, rg.data(), rg.size());
      for (size_t i = 0; i < n && i < rg.size(); ++i)
        if (rg[i].addr == reinterpret_cast<uintptr_t>(from<DeviceMemory>(mem)->host)) return rg[i].size;
      return uint64_t{0};
    };
    CHECK(listed(m) == (1u << 20));
    CHECK(vpvk_diag_ranges(1, nullptr, 0) == 0 && vpvk_diag_ranges(9, nullptr, 0) == 0);
  }
  void* p = nullptr;
  CHECK(vkMapMemory(c.dev, m, 4096, VK_WHOLE_SIZE, 0, &p) == VK_SUCCESS);
  CHECK(p == from<DeviceMemory>(m)->host + 4096);
  CHECK(reinterpret_cast<uintptr_t>(from<DeviceMemory>(m)->host) % 16384 == 0);
  VkDeviceMemory devOnly = alloc(c, 4096, 0);
  CHECK(vkMapMemory(c.dev, devOnly, 0, VK_WHOLE_SIZE, 0, &p) == VK_ERROR_MEMORY_MAP_FAILED);
  // A bound buffer is a core buffer aliasing memory + offset (unified memory, no copy).
  VkBuffer b = buffer(c, 1000, m, 8192);
  const Buffer* bb = from<Buffer>(b);
  CHECK(bb->id);
  const vpgpu::BufferDesc* d = c.gpu().resources().get(bb->id);
  CHECK(d && d->guestMemory == from<DeviceMemory>(m)->host + 8192 && d->size == 1000);
  VkMemoryRequirements r;
  vkGetBufferMemoryRequirements(c.dev, b, &r);
  CHECK(r.size == 1024 && r.alignment == 256 && !(r.memoryTypeBits & 8));
  // Images: transient ones may use lazily allocated memory and are not persistent.
  VkImage t = image(c, VK_FORMAT_D24_UNORM_S8_UINT, 64, 1,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT, 0);
  vkGetImageMemoryRequirements(c.dev, t, &r);
  CHECK(r.memoryTypeBits & 8);
  CHECK(r.size >= 64 * 64 * 4);
  CHECK(from<Image>(t)->desc.transient && !from<Image>(t)->desc.persistent);
  {
    // Every attachment (colour or depth, any sample count, whatever else its usage says) starts memoryless: not
    // persistent, so it may stay on tile and only gets memory for frames that need it. Not an attachment, or LINEAR
    // tiling: persistent.
    auto make = [&](VkFormat f, VkImageUsageFlags u, VkSampleCountFlagBits n) {
      VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      ii.imageType = VK_IMAGE_TYPE_2D, ii.format = f, ii.extent = {64, 64, 1}, ii.mipLevels = 1, ii.arrayLayers = 1;
      ii.samples = n, ii.tiling = VK_IMAGE_TILING_OPTIMAL, ii.usage = u;
      VkImage i = VK_NULL_HANDLE;
      return vkCreateImage(c.dev, &ii, nullptr, &i) == VK_SUCCESS ? i : VK_NULL_HANDLE;
    };
    auto msaa = [&](VkFormat f, VkImageUsageFlags u) { return make(f, u, VK_SAMPLE_COUNT_4_BIT); };
    const VkImageUsageFlags colorUse = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, depthUse = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    VkImage ma = msaa(VK_FORMAT_R8G8B8A8_UNORM, colorUse);
    VkImage ms = msaa(VK_FORMAT_R8G8B8A8_UNORM, colorUse | VK_IMAGE_USAGE_SAMPLED_BIT);
    VkImage mi = msaa(VK_FORMAT_R8G8B8A8_UNORM, colorUse | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    VkImage md = msaa(VK_FORMAT_D32_SFLOAT, depthUse);
    VkImage mds = msaa(VK_FORMAT_D24_UNORM_S8_UINT, depthUse | VK_IMAGE_USAGE_SAMPLED_BIT);
    VkImage mtc = msaa(VK_FORMAT_R8G8B8A8_UNORM, colorUse | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT);
    VkImage mt = msaa(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    VkImage d1 = make(VK_FORMAT_D32_SFLOAT, depthUse, VK_SAMPLE_COUNT_1_BIT);
    VkImage d1s = make(VK_FORMAT_D24_UNORM_S8_UINT, depthUse | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT);
    VkImage c1 = make(VK_FORMAT_R8G8B8A8_UNORM, colorUse, VK_SAMPLE_COUNT_1_BIT);
    VkImage c1s = make(VK_FORMAT_R8G8B8A8_UNORM, colorUse | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT);
    VkImage c1t = make(VK_FORMAT_R8G8B8A8_UNORM, colorUse | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT, VK_SAMPLE_COUNT_1_BIT);
    VkImage t1 = make(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SAMPLE_COUNT_1_BIT);
    VkImage c1e = make(VK_FORMAT_R8G8B8A8_UNORM, colorUse | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT);
    VkImage c1l = VK_NULL_HANDLE;
    {
      VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      ii.imageType = VK_IMAGE_TYPE_2D, ii.format = VK_FORMAT_R8G8B8A8_UNORM, ii.extent = {64, 64, 1}, ii.mipLevels = 1;
      ii.arrayLayers = 1, ii.samples = VK_SAMPLE_COUNT_1_BIT, ii.tiling = VK_IMAGE_TILING_LINEAR;
      ii.usage = colorUse | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
      if (vkCreateImage(c.dev, &ii, nullptr, &c1l) != VK_SUCCESS) c1l = VK_NULL_HANDLE;
    }
    CHECK(ma && ms && mi && md && mds && mtc && mt && d1 && d1s && c1 && c1s && c1t && t1 && c1e && c1l);
    CHECK(from<Image>(ma)->desc.samples > 1);  // the device is not clamping to 1x here: these checks always run
    auto deferred = [&](VkImage i) { return !from<Image>(i)->desc.persistent && !from<Image>(i)->desc.transient; };
    CHECK(deferred(ma));
    CHECK(deferred(ms));   // MSAA colour that is also SAMPLED
    CHECK(deferred(mi));   // INPUT_ATTACHMENT | TRANSFER_SRC
    CHECK(deferred(md));   // MSAA depth
    CHECK(deferred(mds));  // MSAA depth+stencil, SAMPLED
    // The TRANSIENT path is untouched: transient, not persistent, as before.
    CHECK(from<Image>(mtc)->desc.samples > 1 && from<Image>(mtc)->desc.transient && !from<Image>(mtc)->desc.persistent);
    // Not an attachment at all (sampled only): nothing to keep on tile, it stays persistent.
    CHECK(from<Image>(mt)->desc.samples > 1 && from<Image>(mt)->desc.persistent);
    // Single sample: depth/stencil and colour attachments are deferred too.
    CHECK(from<Image>(d1)->desc.samples == 1 && deferred(d1));
    CHECK(from<Image>(d1s)->desc.samples == 1 && deferred(d1s));  // D24S8 | SAMPLED
    CHECK(from<Image>(c1)->desc.samples == 1 && deferred(c1));
    CHECK(from<Image>(c1s)->desc.samples == 1 && deferred(c1s));  // COLOR | SAMPLED
    // 1x TRANSIENT: unchanged (transient, not persistent).
    CHECK(from<Image>(c1t)->desc.samples == 1 && from<Image>(c1t)->desc.transient && !from<Image>(c1t)->desc.persistent);
    // 1x, no attachment usage: persistent. LINEAR tiling: persistent (host-visible staging).
    CHECK(from<Image>(t1)->desc.samples == 1 && from<Image>(t1)->desc.persistent && !from<Image>(t1)->desc.transient);
    CHECK(from<Image>(c1l)->desc.samples == 1 && from<Image>(c1l)->desc.persistent);
    // A host-marked eye image (vpvk_mark_eye_image) is never a tile-only target: persistent once marked before bind.
    CHECK(deferred(c1e));
    CHECK(vpvk_mark_eye_image(c1e) == 0 && from<Image>(c1e)->desc.persistent);
  }
  {
    // Lazily-allocated memory (type 3, like the Adreno's GMEM): no host RAM, nothing committed, no
    // buffers; a transient image binds to it and gets a (tile-deferred) texture.
    VkDeviceMemory lazy = alloc(c, 64 << 20, 3);
    CHECK(lazy && !from<DeviceMemory>(lazy)->host && from<DeviceMemory>(lazy)->lazy);
    VkDeviceSize committed = 1;
    vkGetDeviceMemoryCommitment(c.dev, lazy, &committed);
    CHECK(committed == 0);
    VkImage li = image(c, VK_FORMAT_R8G8B8A8_UNORM, 64, 1,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT, lazy);
    CHECK(li && from<Image>(li)->id);
    // Audit: bound to lazy memory, the transient image is deferred (not persistent: no native lookup at bind, so no
    // MTLTexture is allocated until a frame materialises it) and the memory holds no host pages.
    CHECK(from<Image>(li)->desc.transient && !from<Image>(li)->desc.persistent && !from<Image>(li)->native &&
          !from<DeviceMemory>(lazy)->host);
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = 256;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkBuffer lb = VK_NULL_HANDLE;
    vkCreateBuffer(c.dev, &bi, nullptr, &lb);
    CHECK(vkBindBufferMemory(c.dev, lb, lazy, 0) != VK_SUCCESS);
    void* p = nullptr;
    CHECK(vkMapMemory(c.dev, lazy, 0, VK_WHOLE_SIZE, 0, &p) == VK_ERROR_MEMORY_MAP_FAILED);
    vkDestroyBuffer(c.dev, lb, nullptr);
    vkDestroyImage(c.dev, li, nullptr);
    vkFreeMemory(c.dev, lazy, nullptr);
  }
  CHECK(image(c, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 64, 1, VK_IMAGE_USAGE_SAMPLED_BIT, 0) == VK_NULL_HANDLE);
  {
    // Non-square ASTC: rows are counted in block heights (8x6 over 64x64 -> 8 x 11 blocks of 16 bytes).
    VkImage a = image(c, VK_FORMAT_ASTC_8x6_UNORM_BLOCK, 64, 1, VK_IMAGE_USAGE_SAMPLED_BIT, 0);
    CHECK(a != VK_NULL_HANDLE);
    vkGetImageMemoryRequirements(c.dev, a, &r);
    CHECK(r.size >= 8 * 11 * 16);
    vkDestroyImage(c.dev, a, nullptr);
  }
  {
    VkImage e = image(c, VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 64, 1, VK_IMAGE_USAGE_SAMPLED_BIT, 0);
    CHECK(e != VK_NULL_HANDLE);
    vkGetImageMemoryRequirements(c.dev, e, &r);
    CHECK(r.size >= 16 * 16 * 8);  // 4x4 blocks of 8 bytes
    vkDestroyImage(c.dev, e, nullptr);
  }
  vkDestroyImage(c.dev, t, nullptr);
  vkDestroyBuffer(c.dev, b, nullptr);
  vkFreeMemory(c.dev, m, nullptr);
  vkFreeMemory(c.dev, devOnly, nullptr);
  {
    // Diagnostics: the layer's counters report deltas once, only the keys that moved; the backend half is empty
    // on the null platform; with no frame ever submitted there is nothing to drain.
    char line[2048];
    CHECK(vpvk_diag_line(0, line, sizeof line) > 0);
    CHECK(std::strstr(line, "dt=") == line && std::strstr(line, " alloc1=") && std::strstr(line, " alloc1_b=") &&
          std::strstr(line, " free1=") && std::strstr(line, " map=") && std::strstr(line, " buf_create=") &&
          std::strstr(line, " buf_bind=") && std::strstr(line, " img_create="));
    CHECK(vpvk_diag_line(0, line, sizeof line) == 0);
    CHECK(vpvk_diag_line(1, line, sizeof line) == 0);
    char rep[64] = "x";
    CHECK(vpvk_idle_drain(0, rep, sizeof rep) == 0 && rep[0] == 0);
  }
  if (oldForce) setenv("VPVK_FORCE_1X", savedForce.c_str(), 1);
  else unsetenv("VPVK_FORCE_1X");
}

// VPVK_MSAA_TILE=0 (set by ctest for this group only: msaaTileOn() reads it once per process): attachments (MSAA and 1x) stay
// fully resident.
void testMsaaTileOff() {
  if (!getenv("VPVK_MSAA_TILE") || std::strcmp(getenv("VPVK_MSAA_TILE"), "0") != 0) {
    std::printf("skipped (run with VPVK_MSAA_TILE=0: ctest gpu_vulkan.msaa_tile_off)\n");
    return;
  }
  setenv("VPVK_FORCE_1X", "0", 1);
  Ctx c;
  auto msaa = [&](VkFormat f, VkImageUsageFlags u) {
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D, ii.format = f, ii.extent = {64, 64, 1}, ii.mipLevels = 1, ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_4_BIT, ii.tiling = VK_IMAGE_TILING_OPTIMAL, ii.usage = u;
    VkImage i = VK_NULL_HANDLE;
    return vkCreateImage(c.dev, &ii, nullptr, &i) == VK_SUCCESS ? i : VK_NULL_HANDLE;
  };
  VkImage ma = msaa(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
  VkImage ms = msaa(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
  VkImage md = msaa(VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
  CHECK(ma && ms && md);
  CHECK(from<Image>(ma)->desc.samples > 1 && from<Image>(ma)->desc.persistent);
  CHECK(from<Image>(ms)->desc.samples > 1 && from<Image>(ms)->desc.persistent);
  CHECK(from<Image>(md)->desc.samples > 1 && from<Image>(md)->desc.persistent);
  // 1x attachments too (the switch covers every deferred attachment).
  auto one = [&](VkFormat f, VkImageUsageFlags u) {
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D, ii.format = f, ii.extent = {64, 64, 1}, ii.mipLevels = 1, ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT, ii.tiling = VK_IMAGE_TILING_OPTIMAL, ii.usage = u;
    VkImage i = VK_NULL_HANDLE;
    return vkCreateImage(c.dev, &ii, nullptr, &i) == VK_SUCCESS ? i : VK_NULL_HANDLE;
  };
  VkImage oc = one(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
  VkImage od = one(VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
  CHECK(oc && od);
  CHECK(from<Image>(oc)->desc.samples == 1 && from<Image>(oc)->desc.persistent);
  CHECK(from<Image>(od)->desc.samples == 1 && from<Image>(od)->desc.persistent);
  unsetenv("VPVK_FORCE_1X");
}

// A two-eye multiview pass with DontCare depth: one core pass, two layers, depth never stored.
void testPasses() {
  Ctx c({VK_KHR_MULTIVIEW_EXTENSION_NAME});
  VkDeviceMemory m = alloc(c, 64 << 20, 0);
  VkImage color = image(c, VK_FORMAT_R8G8B8A8_SRGB, 256, 2,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, m);
  VkImage depth = image(c, VK_FORMAT_D24_UNORM_S8_UINT, 256, 2,
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT, m);
  VkImageView views[2];
  VkImage imgs[2] = {color, depth};
  for (int i = 0; i < 2; ++i) {
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = imgs[i];
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    vi.subresourceRange = {i ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 2};
    vkCreateImageView(c.dev, &vi, nullptr, &views[i]);
  }
  VkAttachmentDescription att[2]{};
  att[0].format = VK_FORMAT_R8G8B8A8_SRGB;
  att[0].samples = VK_SAMPLE_COUNT_1_BIT;
  att[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  att[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  att[1] = att[0];
  att[1].format = VK_FORMAT_D24_UNORM_S8_UINT;
  att[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  // Both aspects of the D24S8 image: a stored (or loaded) stencil would keep the image.
  att[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  att[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  VkAttachmentReference cref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkAttachmentReference dref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
  VkSubpassDescription sp{};
  sp.colorAttachmentCount = 1;
  sp.pColorAttachments = &cref;
  sp.pDepthStencilAttachment = &dref;
  const uint32_t mask = 0b11;
  VkRenderPassMultiviewCreateInfo mv{VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO};
  mv.subpassCount = 1;
  mv.pViewMasks = &mask;
  VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, &mv};
  rpi.attachmentCount = 2;
  rpi.pAttachments = att;
  rpi.subpassCount = 1;
  rpi.pSubpasses = &sp;
  VkRenderPass rp;
  vkCreateRenderPass(c.dev, &rpi, nullptr, &rp);
  CHECK(from<RenderPassObj>(rp)->subpasses[0].viewMask == 3);
  VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  fi.renderPass = rp;
  fi.attachmentCount = 2;
  fi.pAttachments = views;
  fi.width = fi.height = 256;
  fi.layers = 1;
  VkFramebuffer fb;
  vkCreateFramebuffer(c.dev, &fi, nullptr, &fb);

  VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  VkCommandPool pool;
  vkCreateCommandPool(c.dev, &pi, nullptr, &pool);
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = pool;
  ai.commandBufferCount = 1;
  VkCommandBuffer cb;
  vkAllocateCommandBuffers(c.dev, &ai, &cb);
  CHECK(valid_loader_magic_value(cb));
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(cb, &bi);
  VkClearValue clears[2]{};
  clears[0].color.float32[0] = 0.25f;
  clears[1].depthStencil = {0.0f, 7};
  VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  rbi.renderPass = rp;
  rbi.framebuffer = fb;
  rbi.renderArea = {{0, 0}, {256, 256}};
  rbi.clearValueCount = 2;
  rbi.pClearValues = clears;
  vkCmdPipelineBarrier(cb, 0, 0, 0, 0, nullptr, 0, nullptr, 0, nullptr);
  vkCmdBeginRenderPass(cb, &rbi, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdDraw(cb, 3, 1, 0, 0);
  vkCmdEndRenderPass(cb);
  vkEndCommandBuffer(cb);

  const Recorder& rec = *from<CommandBuffer>(cb)->rec;
  CHECK(rec.ops.size() == 3 && rec.barriers == 1);
  vpgpu::Frame frame;
  TranslateStats st;
  translate(rec, frame, st);
  CHECK(st.passes == 1 && st.skippedDraws == 1 && st.dropped == 0);
  CHECK(frame.commands().size() == 1);
  if (const auto* p = std::get_if<vpgpu::RenderPass>(&frame.commands()[0])) {
    CHECK(p->desc.layerCount == 2);
    CHECK(p->desc.colors[0].texture == from<Image>(color)->id);
    CHECK(p->desc.colors[0].load == vpgpu::LoadAction::Clear && p->desc.colors[0].store == vpgpu::StoreAction::Store);
    CHECK(p->desc.colors[0].clear.color[0] == 0.25f);
    CHECK(p->desc.depth.texture == from<Image>(depth)->id);
    CHECK(p->desc.depth.store == vpgpu::StoreAction::DontCare);
    CHECK(p->desc.depth.clear.depth == 0.0f && p->desc.depth.clear.stencil == 7);
  } else {
    CHECK(false);
  }
  vkDestroyCommandPool(c.dev, pool, nullptr);
  vkDestroyFramebuffer(c.dev, fb, nullptr);
  vkDestroyRenderPass(c.dev, rp, nullptr);
  for (VkImageView v : views) vkDestroyImageView(c.dev, v, nullptr);
  vkDestroyImage(c.dev, color, nullptr);
  vkDestroyImage(c.dev, depth, nullptr);
  vkFreeMemory(c.dev, m, nullptr);

  CHECK(toStore(VK_ATTACHMENT_STORE_OP_NONE, false) == vpgpu::StoreAction::Store);  // NONE keeps contents
  CHECK(toStore(VK_ATTACHMENT_STORE_OP_DONT_CARE, true) == vpgpu::StoreAction::Resolve);
  CHECK(toLoad(VK_ATTACHMENT_LOAD_OP_NONE_KHR) == vpgpu::LoadAction::DontCare);
}

// Upload + clear + copies through vkQueueSubmit, fences, and the deferred list reaching the core.
void testSubmit() {
  Ctx c;
  VkDeviceMemory m = alloc(c, 1 << 20, 1);
  VkBuffer staging = buffer(c, 64 * 64 * 4, m, 0);
  VkBuffer other = buffer(c, 4096, m, 65536);
  VkImage tex = image(c, VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 64, 1,  // 8-byte 4x4 blocks (no BCn, as Adreno)
                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, m);
  VkImage rt = image(c, VK_FORMAT_R8G8B8A8_UNORM, 64, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, m);
  VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  VkCommandPool pool;
  vkCreateCommandPool(c.dev, &pi, nullptr, &pool);
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = pool;
  ai.commandBufferCount = 1;
  VkCommandBuffer cb;
  vkAllocateCommandBuffers(c.dev, &ai, &cb);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(cb, &bi);
  VkBufferImageCopy region{};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = {64, 64, 1};
  vkCmdCopyBufferToImage(cb, staging, tex, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  VkClearColorValue col{};
  VkImageSubresourceRange all{VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
  vkCmdClearColorImage(cb, rt, VK_IMAGE_LAYOUT_GENERAL, &col, 1, &all);
  VkBufferCopy bc{0, 0, 256};
  vkCmdCopyBuffer(cb, staging, other, 1, &bc);
  vkCmdFillBuffer(cb, other, 0, VK_WHOLE_SIZE, 0);
  vkCmdFillBuffer(cb, other, 0, 16, 0x01020304);  // word pattern: upload memory + copy
  vkEndCommandBuffer(cb);

  vpgpu::Frame frame;
  TranslateStats st;
  translate(*from<CommandBuffer>(cb)->rec, frame, st);
  CHECK(st.copies == 4 && st.clears == 1 && st.dropped == 0);
  const auto* up = std::get_if<vpgpu::Copy>(&frame.commands()[0]);
  CHECK(up && up->kind == vpgpu::CopyKind::BufferToTexture);
  if (up) CHECK(up->srcBuffer.bytesPerRow == 16 * 8 && up->srcBuffer.rowsPerImage == 16 && up->srcBuffer.size == 2048);
  const auto* fill = std::get_if<vpgpu::Copy>(&frame.commands()[3]);
  CHECK(fill && fill->kind == vpgpu::CopyKind::FillBuffer && fill->dstBuffer.size == 4096);

  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  vkCreateFence(c.dev, &fci, nullptr, &fence);
  CHECK(vkGetFenceStatus(c.dev, fence) == VK_NOT_READY);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb;
  CHECK(vkQueueSubmit(c.queue, 1, &si, fence) == VK_SUCCESS);
  CHECK(vkWaitForFences(c.dev, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS);
  vkResetFences(c.dev, 1, &fence);
  CHECK(vkGetFenceStatus(c.dev, fence) == VK_NOT_READY);
  CHECK(vkQueueWaitIdle(c.queue) == VK_SUCCESS);

  // Blits: same format and size = plain copy; compressed/sRGB source into another format, or
  // scaling/mirroring, = sampled blit (TextureBlit); the blit destination is made renderable.
  {
    VkImage etc = image(c, VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK, 64, 1,
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, m);
    VkImage rgba = image(c, VK_FORMAT_R8G8B8A8_SRGB, 64, 2,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, m);
    CHECK(from<Image>(rgba)->desc.usage & vpgpu::usage::colorTarget);
    CHECK(!(from<Image>(etc)->desc.usage & vpgpu::usage::colorTarget));
    VkCommandBuffer bc2;
    vkAllocateCommandBuffers(c.dev, &ai, &bc2);
    vkBeginCommandBuffer(bc2, &bi);
    VkImageBlit bl{};
    bl.srcSubresource = bl.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    bl.srcOffsets[1] = bl.dstOffsets[1] = {64, 64, 1};
    vkCmdBlitImage(bc2, etc, VK_IMAGE_LAYOUT_GENERAL, rgba, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_LINEAR);
    VkImageBlit same = bl;
    same.dstSubresource.baseArrayLayer = 1;
    vkCmdBlitImage(bc2, rgba, VK_IMAGE_LAYOUT_GENERAL, rgba, VK_IMAGE_LAYOUT_GENERAL, 1, &same, VK_FILTER_LINEAR);
    VkImageBlit scaled = same;
    scaled.dstOffsets[0] = {32, 32, 0};
    scaled.dstOffsets[1] = {0, 0, 1};  // half size, mirrored on both axes
    vkCmdBlitImage(bc2, rgba, VK_IMAGE_LAYOUT_GENERAL, rgba, VK_IMAGE_LAYOUT_GENERAL, 1, &scaled, VK_FILTER_NEAREST);
    vkEndCommandBuffer(bc2);
    vpgpu::Frame bf;
    TranslateStats bs;
    translate(*from<CommandBuffer>(bc2)->rec, bf, bs);
    CHECK(bf.commands().size() == 3 && bs.copies == 3 && bs.dropped == 0);
    if (bf.commands().size() == 3) {
      const auto* b0 = std::get_if<vpgpu::Copy>(&bf.commands()[0]);
      CHECK(b0 && b0->kind == vpgpu::CopyKind::TextureBlit && b0->srcTexture.texture == from<Image>(etc)->id &&
            b0->dstTexture.width == 64 && !b0->blitFlipX && b0->blitFilter == vpgpu::Filter::Linear);
      const auto* b1 = std::get_if<vpgpu::Copy>(&bf.commands()[1]);
      CHECK(b1 && b1->kind == vpgpu::CopyKind::TextureToTexture && b1->dstTexture.layer == 1);
      const auto* b2 = std::get_if<vpgpu::Copy>(&bf.commands()[2]);
      CHECK(b2 && b2->kind == vpgpu::CopyKind::TextureBlit && b2->dstTexture.x == 0 && b2->dstTexture.width == 32 &&
            b2->blitFlipX && b2->blitFlipY && b2->blitFilter == vpgpu::Filter::Nearest);
      if (b2) {
        const auto acc = vpgpu::copyAccesses(*b2);
        CHECK(acc.size() == 2 && !acc[0].write && acc[1].write);
      }
    }
    vkDestroyImage(c.dev, etc, nullptr);
    vkDestroyImage(c.dev, rgba, nullptr);
  }
  // 5000 small buffer->image regions into one subresource (a texture atlas / virtual-texture upload): one core copy each.
  // Without rects every one is ordered after the previous (a blit encoder each on Metal); with VPVK_COPY_RECT_HAZARD
  // (CompileOptions::copyRects) only the repeats of an identical 4x4 cell keep a barrier, overlapping ones too.
  {
    VkImage img = image(c, VK_FORMAT_R8G8B8A8_UNORM, 64, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, m);
    VkCommandBuffer rc;
    vkAllocateCommandBuffers(c.dev, &ai, &rc);
    vkBeginCommandBuffer(rc, &bi);
    std::vector<VkBufferImageCopy> regions(5000);
    for (uint32_t i = 0; i < regions.size(); ++i) {
      regions[i] = {};
      regions[i].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      regions[i].imageOffset = {int32_t(i % 16 * 4), int32_t(i / 16 % 16 * 4), 0};  // 256 distinct 4x4 cells
      regions[i].imageExtent = {4, 4, 1};
    }
    vkCmdCopyBufferToImage(rc, staging, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, uint32_t(regions.size()), regions.data());
    vkEndCommandBuffer(rc);
    vpgpu::Frame rf;
    TranslateStats rs;
    translate(*from<CommandBuffer>(rc)->rec, rf, rs);
    CHECK(rf.commands().size() == 5000 && rs.copies == 5000 && rs.dropped == 0);
    const auto* r17 = rf.commands().size() > 17 ? std::get_if<vpgpu::Copy>(&rf.commands()[17]) : nullptr;
    CHECK(r17 && r17->dstTexture.x == 4 && r17->dstTexture.y == 4 && r17->dstTexture.width == 4 && r17->dstTexture.height == 4);
    const vpgpu::CompiledFrame off = vpgpu::compileFrame(rf);
    CHECK(off.encoders.size() == 1 && off.stats.intraBarriers == 4999);
    vpgpu::CompileOptions on;
    on.copyRects = true;
    const vpgpu::CompiledFrame rects = vpgpu::compileFrame(rf, nullptr, on);
    CHECK(rects.encoders.size() == 1 && rects.stats.intraBarriers == 19);  // repeats of cell 0 at 256, 512, ... 4864
    vkDestroyImage(c.dev, img, nullptr);
  }
  vkDestroyFence(c.dev, fence, nullptr);
  vkDestroyCommandPool(c.dev, pool, nullptr);
  vkDestroyImage(c.dev, tex, nullptr);
  vkDestroyImage(c.dev, rt, nullptr);
  vkDestroyBuffer(c.dev, staging, nullptr);
  vkDestroyBuffer(c.dev, other, nullptr);
  vkFreeMemory(c.dev, m, nullptr);
}

std::vector<uint32_t> spirvFile(const std::string& name) {
  std::string path = std::string(VPVK_SPIRV_TESTS) + "/" + name;
  FILE* f = std::fopen(path.c_str(), "rb");
  std::vector<uint32_t> w;
  if (!f) return w;
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  w.resize(size_t(n) / 4);
  if (std::fread(w.data(), 4, w.size(), f) != w.size()) w.clear();
  std::fclose(f);
  return w;
}

VkShaderModule module(Ctx& c, const std::string& name) {
  std::vector<uint32_t> w = spirvFile(name);
  VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  ci.codeSize = w.size() * 4;
  ci.pCode = w.data();
  VkShaderModule m;
  vkCreateShaderModule(c.dev, &ci, nullptr, &m);
  return m;
}

// A multiview draw through the whole driver: our own vertex + fragment shaders (gpu/shader/spirv
// tests), descriptors of every kind they use, push constants, two vertex buffers, a dynamic
// uniform offset. Checks the pipeline compiled and the core draw carries what the shaders read.
void testPipeline() {
  Ctx c({VK_KHR_MULTIVIEW_EXTENSION_NAME});
  VkDeviceMemory m = alloc(c, 64 << 20, 0);
  VkBuffer ubo = buffer(c, 4096, m, 0), lights = buffer(c, 4096, m, 4096), vb0 = buffer(c, 4096, m, 8192),
           vb1 = buffer(c, 4096, m, 12288), ib = buffer(c, 4096, m, 16384);
  VkImage tex = image(c, VK_FORMAT_ASTC_6x6_SRGB_BLOCK, 64, 1, VK_IMAGE_USAGE_SAMPLED_BIT, m);
  VkImage shadow = image(c, VK_FORMAT_D32_SFLOAT, 64, 2, VK_IMAGE_USAGE_SAMPLED_BIT, m);
  VkImage color = image(c, VK_FORMAT_R8G8B8A8_SRGB, 128, 2, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, m);
  auto view = [&](VkImage i, VkImageViewType t, uint32_t layers, VkImageAspectFlags a) {
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = i, vi.viewType = t, vi.subresourceRange = {a, 0, 1, 0, layers};
    VkImageView v;
    vkCreateImageView(c.dev, &vi, nullptr, &v);
    return v;
  };
  VkImageView texView = view(tex, VK_IMAGE_VIEW_TYPE_2D, 1, VK_IMAGE_ASPECT_COLOR_BIT);
  VkImageView shadowView = view(shadow, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 2, VK_IMAGE_ASPECT_DEPTH_BIT);
  VkImageView colorView = view(color, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 2, VK_IMAGE_ASPECT_COLOR_BIT);
  VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  VkSampler smp;
  vkCreateSampler(c.dev, &si, nullptr, &smp);

  // set 0: the fragment shader's resources + the camera; set 1: the object (dynamic offset).
  VkDescriptorSetLayoutBinding b0[7] = {
      {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {3, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
  VkDescriptorSetLayoutBinding b1 = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  VkDescriptorSetLayout sl[2];
  lci.bindingCount = 7, lci.pBindings = b0;
  vkCreateDescriptorSetLayout(c.dev, &lci, nullptr, &sl[0]);
  lci.bindingCount = 1, lci.pBindings = &b1;
  vkCreateDescriptorSetLayout(c.dev, &lci, nullptr, &sl[1]);
  VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, 20};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 2, plci.pSetLayouts = sl, plci.pushConstantRangeCount = 1, plci.pPushConstantRanges = &pcr;
  VkPipelineLayout pl;
  vkCreatePipelineLayout(c.dev, &plci, nullptr, &pl);

  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  VkDescriptorPool pool;
  vkCreateDescriptorPool(c.dev, &dpci, nullptr, &pool);
  VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dai.descriptorPool = pool, dai.descriptorSetCount = 2, dai.pSetLayouts = sl;
  VkDescriptorSet sets[2];
  vkAllocateDescriptorSets(c.dev, &dai, sets);
  VkDescriptorBufferInfo uboInfo{ubo, 0, 512}, lightInfo{lights, 0, VK_WHOLE_SIZE}, objInfo{ubo, 0, 256};
  VkDescriptorImageInfo texInfo{smp, texView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkDescriptorImageInfo shadowInfo{smp, shadowView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkDescriptorImageInfo smpInfo{smp, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
  VkWriteDescriptorSet w[8]{};
  auto wr = [&](int i, VkDescriptorSet s, uint32_t b, VkDescriptorType t) {
    w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[i].dstSet = s, w[i].dstBinding = b, w[i].descriptorCount = 1, w[i].descriptorType = t;
  };
  wr(0, sets[0], 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER), w[0].pBufferInfo = &uboInfo;
  wr(1, sets[0], 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER), w[1].pImageInfo = &texInfo;
  wr(2, sets[0], 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE), w[2].pImageInfo = &texInfo;
  wr(3, sets[0], 3, VK_DESCRIPTOR_TYPE_SAMPLER), w[3].pImageInfo = &smpInfo;
  wr(4, sets[0], 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER), w[4].pImageInfo = &shadowInfo;
  wr(5, sets[0], 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER), w[5].pBufferInfo = &lightInfo;
  wr(6, sets[0], 6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER), w[6].pBufferInfo = &uboInfo;
  wr(7, sets[1], 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC), w[7].pBufferInfo = &objInfo;
  vkUpdateDescriptorSets(c.dev, 8, w, 0, nullptr);

  // Multiview pass.
  VkAttachmentDescription att{};
  att.format = VK_FORMAT_R8G8B8A8_SRGB, att.samples = VK_SAMPLE_COUNT_1_BIT;
  att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  VkAttachmentReference cref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkSubpassDescription sp{};
  sp.colorAttachmentCount = 1, sp.pColorAttachments = &cref;
  const uint32_t mask = 0b11;
  VkRenderPassMultiviewCreateInfo mv{VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO};
  mv.subpassCount = 1, mv.pViewMasks = &mask;
  VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, &mv};
  rpi.attachmentCount = 1, rpi.pAttachments = &att, rpi.subpassCount = 1, rpi.pSubpasses = &sp;
  VkRenderPass rp;
  vkCreateRenderPass(c.dev, &rpi, nullptr, &rp);
  VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  fi.renderPass = rp, fi.attachmentCount = 1, fi.pAttachments = &colorView, fi.width = 128, fi.height = 128, fi.layers = 1;
  VkFramebuffer fb;
  vkCreateFramebuffer(c.dev, &fi, nullptr, &fb);

  VkShaderModule vs = module(c, "mesh_multiview.vert.opt.spv"), fs = module(c, "lit.frag.opt.spv");
  VkPipelineShaderStageCreateInfo stages[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                               {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT, stages[0].module = vs, stages[0].pName = "main";
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT, stages[1].module = fs, stages[1].pName = "main";
  VkVertexInputBindingDescription vbd[2] = {{0, 32, VK_VERTEX_INPUT_RATE_VERTEX}, {1, 16, VK_VERTEX_INPUT_RATE_INSTANCE}};
  VkVertexInputAttributeDescription vad[4] = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
                                              {1, 0, VK_FORMAT_R8G8B8A8_SNORM, 12},
                                              {2, 0, VK_FORMAT_R16G16_SFLOAT, 16},
                                              {3, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0}};
  VkPipelineVertexInputStateCreateInfo vis{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vis.vertexBindingDescriptionCount = 2, vis.pVertexBindingDescriptions = vbd;
  vis.vertexAttributeDescriptionCount = 4, vis.pVertexAttributeDescriptions = vad;
  VkPipelineInputAssemblyStateCreateInfo ias{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ias.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.cullMode = VK_CULL_MODE_BACK_BIT, rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineColorBlendAttachmentState cba{};
  cba.blendEnable = VK_TRUE, cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, cba.colorWriteMask = 0xF;
  VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cbs.attachmentCount = 1, cbs.pAttachments = &cba;
  VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dys{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dys.dynamicStateCount = 2, dys.pDynamicStates = dyn;
  VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  gp.stageCount = 2, gp.pStages = stages, gp.pVertexInputState = &vis, gp.pInputAssemblyState = &ias;
  gp.pRasterizationState = &rs, gp.pMultisampleState = &ms, gp.pColorBlendState = &cbs, gp.pDynamicState = &dys;
  gp.layout = pl, gp.renderPass = rp, gp.subpass = 0;
  VkPipeline pipe;
  vkCreateGraphicsPipelines(c.dev, VK_NULL_HANDLE, 1, &gp, nullptr, &pipe);
  const PipelineObj* po = from<PipelineObj>(pipe);
  CHECK(po->ok && po->viewCount == 2 && po->state.vertexShader && po->state.fragmentShader);
  CHECK(po->state.colorFormats[0] == vpgpu::Format::RGBA8Srgb && po->state.cull == vpgpu::CullMode::Back);
  c.gpu().pipelines().waitIdle();
  CHECK(c.gpu().pipelines().status(po->key) == vpgpu::PipelineCache::Status::Ready);

  VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  VkCommandPool cp;
  vkCreateCommandPool(c.dev, &cpi, nullptr, &cp);
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = cp, ai.commandBufferCount = 1;
  VkCommandBuffer cb;
  vkAllocateCommandBuffers(c.dev, &ai, &cb);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(cb, &bi);
  VkClearValue clear{};
  VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  rbi.renderPass = rp, rbi.framebuffer = fb, rbi.renderArea = {{0, 0}, {128, 128}}, rbi.clearValueCount = 1, rbi.pClearValues = &clear;
  vkCmdBeginRenderPass(cb, &rbi, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
  uint32_t dynOff = 256;
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 2, sets, 1, &dynOff);
  float push[5] = {1, 0.5f, 0.25f, 1, 0};
  vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT, 0, 20, push);
  VkBuffer vbs[2] = {vb0, vb1};
  VkDeviceSize offs[2] = {0, 64};
  vkCmdBindVertexBuffers(cb, 0, 2, vbs, offs);
  vkCmdBindIndexBuffer(cb, ib, 0, VK_INDEX_TYPE_UINT16);
  VkViewport vp{0, 0, 128, 128, 0, 1};
  VkRect2D sc{{0, 0}, {128, 128}};
  vkCmdSetViewport(cb, 0, 1, &vp);
  vkCmdSetScissor(cb, 0, 1, &sc);
  vkCmdDrawIndexed(cb, 36, 3, 6, 2, 1);
  vkCmdEndRenderPass(cb);
  vkEndCommandBuffer(cb);

  vpgpu::Frame frame;
  TranslateStats st;
  translate(*from<CommandBuffer>(cb)->rec, frame, st);
  CHECK(st.passes == 1 && st.draws == 1 && st.skippedDraws == 0);
  const auto* pass = frame.commands().empty() ? nullptr : std::get_if<vpgpu::RenderPass>(&frame.commands()[0]);
  CHECK(pass && pass->draws.size() == 1 && pass->desc.layerCount == 2);
  if (pass && pass->draws.size() == 1) {
    const vpgpu::Draw& d = pass->draws[0];
    CHECK(d.pipeline == po->key);
    CHECK(d.instances == 6 && d.baseInstance == 2 && d.count == 36 && d.first == 6 && d.baseVertex == 2);
    CHECK(d.index && !d.index->wide);
    CHECK(d.frontFace == vpgpu::Winding::Clockwise && d.viewport.width == 128);
    int cbufs = 0, bufs = 0, texs = 0, smps = 0;
    bool dynamicApplied = false;
    for (const vpgpu::Access& a : d.bindings) {
      if (a.kind == vpgpu::BindKind::ConstantBuffer) ++cbufs;
      if (a.kind == vpgpu::BindKind::Buffer) ++bufs;
      if (a.kind == vpgpu::BindKind::Texture) ++texs;
      if (a.kind == vpgpu::BindKind::Sampler) ++smps;
      if (a.kind == vpgpu::BindKind::ConstantBuffer && a.range.begin == 256) dynamicApplied = true;
    }
    // camera, material, object, push constants | lights + 2 vertex buffers | 3 textures | 3 samplers
    CHECK(cbufs == 4 && bufs == 3 && texs == 3 && smps == 3);
    CHECK(dynamicApplied);
  }
  vkDestroyPipeline(c.dev, pipe, nullptr);
  vkDestroyShaderModule(c.dev, vs, nullptr);
  vkDestroyShaderModule(c.dev, fs, nullptr);
  vkDestroyCommandPool(c.dev, cp, nullptr);
  vkDestroyDescriptorPool(c.dev, pool, nullptr);
  vkDestroyPipelineLayout(c.dev, pl, nullptr);
  vkDestroyDescriptorSetLayout(c.dev, sl[0], nullptr);
  vkDestroyDescriptorSetLayout(c.dev, sl[1], nullptr);
}

// vertexPipelineStoresAndAtomics: a vertex shader that stores to and atomically adds into a storage
// buffer (vertex_store.vert, gen_int16_drawparams.py) builds into a ready pipeline, its buffer is
// reported written, and the draw binds it.
void testVertexStores() {
  setenv("VPVK_EXTRA_EXTENSIONS", "1", 1);  // vertex stores sit behind the CTS gate
  Ctx c;
  VkPhysicalDeviceFeatures f;
  vkGetPhysicalDeviceFeatures(c.pd, &f);
  CHECK(f.vertexPipelineStoresAndAtomics && f.fragmentStoresAndAtomics);
  VkDeviceMemory m = alloc(c, 1 << 20, 0);
  VkBuffer sb = buffer(c, 4096, m, 0);
  VkImage color = image(c, VK_FORMAT_R8G8B8A8_UNORM, 64, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, m);
  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.image = color, vi.viewType = VK_IMAGE_VIEW_TYPE_2D, vi.format = VK_FORMAT_R8G8B8A8_UNORM;
  vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkImageView cv;
  vkCreateImageView(c.dev, &vi, nullptr, &cv);
  VkDescriptorSetLayoutBinding b{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  lci.bindingCount = 1, lci.pBindings = &b;
  VkDescriptorSetLayout sl;
  vkCreateDescriptorSetLayout(c.dev, &lci, nullptr, &sl);
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1, plci.pSetLayouts = &sl;
  VkPipelineLayout pl;
  vkCreatePipelineLayout(c.dev, &plci, nullptr, &pl);
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  VkDescriptorPool pool;
  vkCreateDescriptorPool(c.dev, &dpci, nullptr, &pool);
  VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dai.descriptorPool = pool, dai.descriptorSetCount = 1, dai.pSetLayouts = &sl;
  VkDescriptorSet set;
  vkAllocateDescriptorSets(c.dev, &dai, &set);
  VkDescriptorBufferInfo bi{sb, 0, VK_WHOLE_SIZE};
  VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  w.dstSet = set, w.descriptorCount = 1, w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, w.pBufferInfo = &bi;
  vkUpdateDescriptorSets(c.dev, 1, &w, 0, nullptr);

  VkAttachmentDescription att{};
  att.format = VK_FORMAT_R8G8B8A8_UNORM, att.samples = VK_SAMPLE_COUNT_1_BIT;
  att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  VkAttachmentReference cref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkSubpassDescription sp{};
  sp.colorAttachmentCount = 1, sp.pColorAttachments = &cref;
  VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  rpi.attachmentCount = 1, rpi.pAttachments = &att, rpi.subpassCount = 1, rpi.pSubpasses = &sp;
  VkRenderPass rp;
  vkCreateRenderPass(c.dev, &rpi, nullptr, &rp);
  VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  fi.renderPass = rp, fi.attachmentCount = 1, fi.pAttachments = &cv, fi.width = 64, fi.height = 64, fi.layers = 1;
  VkFramebuffer fb;
  vkCreateFramebuffer(c.dev, &fi, nullptr, &fb);

  VkShaderModule vs = module(c, "vertex_store.vert.spv");
  VkPipelineShaderStageCreateInfo st{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  st.stage = VK_SHADER_STAGE_VERTEX_BIT, st.module = vs, st.pName = "main";
  VkPipelineVertexInputStateCreateInfo vis{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo ias{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ias.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.rasterizerDiscardEnable = VK_TRUE;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineColorBlendAttachmentState cba{};
  cba.colorWriteMask = 0xF;
  VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cbs.attachmentCount = 1, cbs.pAttachments = &cba;
  VkViewport vp{0, 0, 64, 64, 0, 1};
  VkRect2D sc{{0, 0}, {64, 64}};
  VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vps.viewportCount = 1, vps.pViewports = &vp, vps.scissorCount = 1, vps.pScissors = &sc;
  VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  gp.stageCount = 1, gp.pStages = &st, gp.pVertexInputState = &vis, gp.pInputAssemblyState = &ias;
  gp.pViewportState = &vps, gp.pRasterizationState = &rs, gp.pMultisampleState = &ms, gp.pColorBlendState = &cbs;
  gp.layout = pl, gp.renderPass = rp;
  VkPipeline pipe = VK_NULL_HANDLE;
  CHECK(vkCreateGraphicsPipelines(c.dev, VK_NULL_HANDLE, 1, &gp, nullptr, &pipe) == VK_SUCCESS);
  const PipelineObj* po = pipe ? from<PipelineObj>(pipe) : nullptr;
  CHECK(po && po->ok && po->state.vertexShader);
  if (po && po->ok) {
    c.gpu().pipelines().waitIdle();
    CHECK(c.gpu().pipelines().status(po->key) == vpgpu::PipelineCache::Status::Ready);
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    VkCommandPool cp;
    vkCreateCommandPool(c.dev, &cpi, nullptr, &cp);
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = cp, ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    vkAllocateCommandBuffers(c.dev, &ai, &cb);
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cb, &cbi);
    VkClearValue clear{};
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = rp, rbi.framebuffer = fb, rbi.renderArea = {{0, 0}, {64, 64}}, rbi.clearValueCount = 1;
    rbi.pClearValues = &clear;
    vkCmdBeginRenderPass(cb, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &set, 0, nullptr);
    vkCmdDraw(cb, 16, 1, 0, 0);
    // gl_BaseVertex of a non-indexed draw comes from Draw::first (VpDrawParams.baseVertex), gl_DrawID
    // of a multi-draw from Draw::drawIndex (an empty entry records nothing and shifts no index).
    vkCmdDraw(cb, 8, 1, 5, 0);
    auto multi = reinterpret_cast<PFN_vkCmdDrawMultiEXT>(vkGetDeviceProcAddr(c.dev, "vkCmdDrawMultiEXT"));
    CHECK(multi != nullptr);
    const VkMultiDrawInfoEXT md[3] = {{3, 4}, {9, 0}, {2, 6}};  // {firstVertex, vertexCount}
    if (multi) multi(cb, 3, md, 1, 0, sizeof(VkMultiDrawInfoEXT));
    vkCmdEndRenderPass(cb);
    vkEndCommandBuffer(cb);
    vpgpu::Frame frame;
    TranslateStats ts;
    translate(*from<CommandBuffer>(cb)->rec, frame, ts);
    CHECK(ts.draws == 4 && ts.skippedDraws == 0);
    const auto* pass = frame.commands().empty() ? nullptr : std::get_if<vpgpu::RenderPass>(&frame.commands()[0]);
    CHECK(pass && pass->draws.size() == 4);
    if (pass && pass->draws.size() == 4) {
      const auto& ds = pass->draws;
      CHECK(!ds[1].index && ds[1].first == 5 && ds[1].drawIndex == 0);
      CHECK(ds[2].first == 3 && ds[2].drawIndex == 0 && ds[3].first == 2 && ds[3].drawIndex == 2);
    }
    if (pass && !pass->draws.empty()) {
      bool written = false;
      for (const vpgpu::Access& a : pass->draws[0].bindings)
        if (a.kind == vpgpu::BindKind::Buffer && (a.stages & vpgpu::stage::vertex) && a.write) written = true;
      CHECK(written);  // the vertex stage writes the storage buffer (hazard tracking sees it)
    }
    vkDestroyCommandPool(c.dev, cp, nullptr);
  }
  if (pipe) vkDestroyPipeline(c.dev, pipe, nullptr);
  vkDestroyShaderModule(c.dev, vs, nullptr);
  vkDestroyFramebuffer(c.dev, fb, nullptr);
  vkDestroyRenderPass(c.dev, rp, nullptr);
  vkDestroyDescriptorPool(c.dev, pool, nullptr);
  vkDestroyPipelineLayout(c.dev, pl, nullptr);
  vkDestroyDescriptorSetLayout(c.dev, sl, nullptr);
  vkDestroyImageView(c.dev, cv, nullptr);
  vkDestroyImage(c.dev, color, nullptr);
  vkDestroyBuffer(c.dev, sb, nullptr);
  vkFreeMemory(c.dev, m, nullptr);
  unsetenv("VPVK_EXTRA_EXTENSIONS");
}

// Default identity (no Quest identity): what the Apple GPU supports, honestly. The Adreno 740 profile is opt-in
// (vpvk_config_set_quest_identity) and read per query, so a host may choose it after the device exists.
void testNativeIdentity() {
  unsetenv("VPVK_QUEST_IDENTITY");
  unsetenv("VPVK_EXTRA_EXTENSIONS");
  vpvk_config_set_quest_identity(0);
  Ctx c;
  CHECK(c.dev != nullptr);
  VkPhysicalDeviceProperties p;
  vkGetPhysicalDeviceProperties(c.pd, &p);
  CHECK(p.vendorID == 0x106B && p.vendorID != 0x5143);
  CHECK(!std::strstr(p.deviceName, "Adreno") && p.deviceName[0]);
  CHECK(p.apiVersion == VK_API_VERSION_1_3);
  uint32_t iv = 0;
  vkEnumerateInstanceVersion(&iv);
  CHECK(iv == VK_API_VERSION_1_3);
  CHECK(p.limits.maxPerStageDescriptorSamplers == 64);
  VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
  VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &drv};
  vkGetPhysicalDeviceProperties2(c.pd, &p2);
  CHECK(!std::strstr(drv.driverName, "Adreno"));
  VkPhysicalDeviceFeatures f;
  vkGetPhysicalDeviceFeatures(c.pd, &f);
  CHECK(f.textureCompressionBC && f.textureCompressionETC2 && f.textureCompressionASTC_LDR);
  for (VkFormat bc : {VK_FORMAT_BC1_RGB_UNORM_BLOCK, VK_FORMAT_BC1_RGBA_SRGB_BLOCK, VK_FORMAT_BC2_UNORM_BLOCK,
                      VK_FORMAT_BC3_SRGB_BLOCK, VK_FORMAT_BC4_SNORM_BLOCK, VK_FORMAT_BC5_UNORM_BLOCK,
                      VK_FORMAT_BC6H_UFLOAT_BLOCK, VK_FORMAT_BC7_SRGB_BLOCK}) {
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(c.pd, bc, &fp);
    CHECK((fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) &&
          (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT));
  }
  VkImage bcImage = image(c, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 64, 1, VK_IMAGE_USAGE_SAMPLED_BIT, 0);
  CHECK(bcImage != VK_NULL_HANDLE);
  vkDestroyImage(c.dev, bcImage, nullptr);
  uint32_t n = 0;
  vkEnumerateDeviceExtensionProperties(c.pd, nullptr, &n, nullptr);
  std::vector<VkExtensionProperties> e(n);
  vkEnumerateDeviceExtensionProperties(c.pd, nullptr, &n, e.data());
  auto has = [&](const char* name) {
    for (const auto& x : e)
      if (!std::strcmp(x.extensionName, name)) return true;
    return false;
  };
  // Hidden with the Quest identity; listed here, with what EXTRA_EXTENSIONS adds there.
  CHECK(has(VK_EXT_METAL_OBJECTS_EXTENSION_NAME) && has(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME) &&
        has(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME) && has(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME));
  // Not implemented: not advertised (phase 1b).
  CHECK(!has("VK_KHR_dynamic_rendering_local_read") && !has("VK_EXT_host_image_copy") && !has("VK_EXT_memory_budget") &&
        !has("VK_EXT_separate_stencil_usage") && !has("VK_EXT_swapchain_mutable_format"));
  VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
  vkGetPhysicalDeviceFeatures2(c.pd, &f2);
  CHECK(f13.dynamicRendering && f13.synchronization2);
  // Explicit VPVK_EXTRA_EXTENSIONS=0 turns the extras off even here; the API version stays 1.3.
  setenv("VPVK_EXTRA_EXTENSIONS", "0", 1);
  vkGetPhysicalDeviceProperties(c.pd, &p);
  vkGetPhysicalDeviceFeatures2(c.pd, &f2);
  CHECK(p.apiVersion == VK_API_VERSION_1_3 && !f13.dynamicRendering);
  unsetenv("VPVK_EXTRA_EXTENSIONS");
  // Chosen late (the device exists): the next query answers as the Quest 3.
  vpvk_config_set_quest_identity(1);
  vkGetPhysicalDeviceProperties(c.pd, &p);
  CHECK(p.vendorID == 0x5143 && std::strcmp(p.deviceName, "Adreno (TM) 740") == 0 && p.apiVersion == VK_API_VERSION_1_1);
  CHECK(p.limits.maxPerStageDescriptorSamplers == 16);
  vkGetPhysicalDeviceFeatures(c.pd, &f);
  CHECK(!f.textureCompressionBC);
  vpvk_config_set_quest_identity(0);
}

// The Adreno 740 identity seen through Properties2 / Features2 (Vulkan 1.1/1.2/1.3 structs,
// driver properties) and the VP_ANDROID_baseline_2022 extensions. What the CTS has not validated
// (shaderInt16, draw parameters, the 1.2/1.3 extensions and structs) answers only with
// VPVK_EXTRA_EXTENSIONS=1; games see the pre-profile answers.
void testProfile() {
  vpvk_config_set_quest_identity(1);  // asserts the Quest 3 / Adreno 740 profile: opt-in
  unsetenv("VPVK_EXTRA_EXTENSIONS");
  {
    Ctx g;
    VkPhysicalDeviceShaderDrawParametersFeatures dp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &dp};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &f12};
    VkPhysicalDeviceTimelineSemaphoreFeatures tl{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES, &f13};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &tl};
    vkGetPhysicalDeviceFeatures2(g.pd, &f2);
    CHECK(!f2.features.shaderInt16 && !f2.features.vertexPipelineStoresAndAtomics);
    CHECK(!dp.shaderDrawParameters && !tl.timelineSemaphore && !f12.drawIndirectCount && !f13.dynamicRendering);
    CHECK(f2.features.robustBufferAccess && !f2.features.textureCompressionBC);
    // Games see Vulkan 1.1; the conformance run sees 1.3 (below).
    VkPhysicalDeviceProperties gp;
    vkGetPhysicalDeviceProperties(g.pd, &gp);
    uint32_t iv = 0;
    vkEnumerateInstanceVersion(&iv);
    VkPhysicalDeviceDriverProperties gd{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 gp2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &gd};
    vkGetPhysicalDeviceProperties2(g.pd, &gp2);
    CHECK(gp.apiVersion == VK_API_VERSION_1_1 && iv == VK_API_VERSION_1_1);
    CHECK(gd.conformanceVersion.major == 1 && gd.conformanceVersion.minor == 1);
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(g.pd, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> e(n);
    vkEnumerateDeviceExtensionProperties(g.pd, nullptr, &n, e.data());
    bool hidden = true, base = false;
    for (const auto& x : e) {
      for (const char* h : {VK_KHR_SHADER_DRAW_PARAMETERS_EXTENSION_NAME, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
                            VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
                            VK_EXT_MULTI_DRAW_EXTENSION_NAME})
        if (!std::strcmp(x.extensionName, h)) hidden = false;
      if (!std::strcmp(x.extensionName, VK_KHR_MAINTENANCE_2_EXTENSION_NAME)) base = true;
    }
    CHECK(hidden && base);
  }
  setenv("VPVK_EXTRA_EXTENSIONS", "1", 1);
  Ctx c;
  VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
  VkPhysicalDeviceVulkan12Properties p12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES, &drv};
  VkPhysicalDeviceVulkan11Properties p11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES, &p12};
  VkPhysicalDeviceVulkan13Properties p13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES, &p11};
  VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &p13};
  vkGetPhysicalDeviceProperties2(c.pd, &p2);
  CHECK(drv.driverID == VK_DRIVER_ID_QUALCOMM_PROPRIETARY && p12.driverID == VK_DRIVER_ID_QUALCOMM_PROPRIETARY);
  CHECK(std::strstr(drv.driverName, "Qualcomm") && std::strstr(p12.driverName, "Adreno"));
  CHECK(p11.maxMultiviewViewCount == 8 && p11.subgroupSize == 32 && p11.deviceLUIDValid);
  CHECK(p13.minSubgroupSize == 32 && p13.maxSubgroupSize == 32);
  CHECK(p2.pNext == &p13 && p13.pNext == &p11 && p11.pNext == &p12 && p12.pNext == &drv);  // chain kept
  CHECK(std::strcmp(p2.properties.deviceName, "Adreno (TM) 740") == 0);
  CHECK((p2.properties.driverVersion >> 22) == 512);
  CHECK(p2.properties.apiVersion == VK_API_VERSION_1_3);
  CHECK(drv.conformanceVersion.major == 1 && drv.conformanceVersion.minor == 3);
  CHECK(p12.conformanceVersion.major == 1 && p12.conformanceVersion.minor == 3);
  {
    uint32_t iv = 0;
    vkEnumerateInstanceVersion(&iv);
    CHECK(iv == VK_API_VERSION_1_3);
    // The 1.1 / 1.3 core entry points a 1.3 loader asks for resolve and answer for one device.
    uint32_t ng = 0, nt = 7;
    CHECK(vkEnumeratePhysicalDeviceGroups(c.inst, &ng, nullptr) == VK_SUCCESS && ng == 1);
    VkPhysicalDeviceGroupProperties grp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES};
    CHECK(vkEnumeratePhysicalDeviceGroups(c.inst, &ng, &grp) == VK_SUCCESS && grp.physicalDeviceCount == 1 &&
          grp.physicalDevices[0] == c.pd);
    CHECK(vkGetPhysicalDeviceToolProperties(c.pd, &nt, nullptr) == VK_SUCCESS && nt == 0);
    VkPhysicalDeviceExternalBufferInfo ebi{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
    VkExternalBufferProperties ebp{VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
    ebp.externalMemoryProperties.compatibleHandleTypes = 1;
    vkGetPhysicalDeviceExternalBufferProperties(c.pd, &ebi, &ebp);
    CHECK(ebp.externalMemoryProperties.compatibleHandleTypes == 0 && ebp.externalMemoryProperties.externalMemoryFeatures == 0);
    for (const char* name : {"vkGetPhysicalDeviceToolProperties", "vkEnumeratePhysicalDeviceGroupsKHR", "vkCmdDispatchBase",
                             "vkGetPhysicalDeviceExternalSemaphoreProperties", "vkGetImageSparseMemoryRequirements2",
                             "vkCreateSamplerYcbcrConversion", "vkDestroySamplerYcbcrConversionKHR"})
      CHECK(vkGetInstanceProcAddr(c.inst, name) != nullptr);
    // Device creation refuses features that are advertised false (featuresSupported): the structs
    // added with the 1.2 / 1.3 work are checked too.
    VkPhysicalDeviceVariablePointersFeatures vp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES};
    vp.variablePointers = VK_TRUE;
    VkPhysicalDeviceBufferDeviceAddressFeatures ba{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    ba.bufferDeviceAddressCaptureReplay = VK_TRUE;
    VkPhysicalDeviceExtendedDynamicState2FeaturesEXT e2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT};
    e2.extendedDynamicState2LogicOp = VK_TRUE;
    VkPhysicalDeviceVulkanMemoryModelFeatures mm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES};
    mm.vulkanMemoryModelAvailabilityVisibilityChains = VK_TRUE;
    VkPhysicalDeviceInlineUniformBlockFeatures iu{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES};
    iu.descriptorBindingInlineUniformBlockUpdateAfterBind = VK_TRUE;
    void* bads[] = {&vp, &ba, &e2, &mm, &iu};
    // Only VPVK_STRICT_FEATURES=1 (conformance) refuses; otherwise the device is created and the features are logged once.
    for (const char* strict : {"1", (const char*)nullptr}) {
      if (strict) setenv("VPVK_STRICT_FEATURES", strict, 1);
      else unsetenv("VPVK_STRICT_FEATURES");
      for (void* bad : bads) {
        float prio = 1;
        VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        q.queueCount = 1, q.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, bad};
        dci.queueCreateInfoCount = 1, dci.pQueueCreateInfos = &q;
        VkDevice dev = nullptr;
        const VkResult r = vkCreateDevice(c.pd, &dci, nullptr, &dev);
        CHECK(r == (strict ? VK_ERROR_FEATURE_NOT_PRESENT : VK_SUCCESS));
        if (dev) vkDestroyDevice(dev, nullptr);
      }
    }
    // ... and what is supported is accepted.
    VkPhysicalDeviceBufferDeviceAddressFeatures okba{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    okba.bufferDeviceAddress = VK_TRUE;
    float prio = 1;
    VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    q.queueCount = 1, q.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &okba};
    dci.queueCreateInfoCount = 1, dci.pQueueCreateInfos = &q;
    VkDevice dev = nullptr;
    CHECK(vkCreateDevice(c.pd, &dci, nullptr, &dev) == VK_SUCCESS);
    if (dev) vkDestroyDevice(dev, nullptr);
  }

  VkPhysicalDeviceShaderDrawParametersFeatures dp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES};
  VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &dp};
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f11};
  VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &f12};
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
  vkGetPhysicalDeviceFeatures2(c.pd, &f2);
  CHECK(f2.features.shaderInt16 && f2.features.shaderStorageImageArrayDynamicIndexing);
  CHECK(!f2.features.textureCompressionBC && f2.features.textureCompressionASTC_LDR);
  CHECK(dp.shaderDrawParameters && f11.shaderDrawParameters && f11.multiview);
  CHECK(f11.samplerYcbcrConversion && f11.variablePointersStorageBuffer && !f11.variablePointers && f11.storageBuffer16BitAccess && !f11.storageInputOutput16);
  CHECK(f12.timelineSemaphore && f12.scalarBlockLayout && f12.drawIndirectCount && f12.bufferDeviceAddress && !f12.bufferDeviceAddressCaptureReplay);
  CHECK(f12.vulkanMemoryModel && f12.vulkanMemoryModelDeviceScope && !f12.vulkanMemoryModelAvailabilityVisibilityChains);
  CHECK(f13.dynamicRendering && f13.synchronization2 && f13.maintenance4 && f13.inlineUniformBlock && f13.shaderDemoteToHelperInvocation);
  CHECK(f12.shaderFloat16 && f12.shaderInt8 && f12.storageBuffer8BitAccess && f12.uniformAndStorageBuffer8BitAccess && f12.storagePushConstant8);
  CHECK(f2.pNext == &f13 && f13.pNext == &f12 && f12.pNext == &f11 && f11.pNext == &dp);

  uint32_t n = 0;
  vkEnumerateDeviceExtensionProperties(c.pd, nullptr, &n, nullptr);
  std::vector<VkExtensionProperties> e(n);
  vkEnumerateDeviceExtensionProperties(c.pd, nullptr, &n, e.data());
  auto has = [&](const char* name) {
    for (const auto& x : e)
      if (!std::strcmp(x.extensionName, name)) return true;
    return false;
  };
  for (const char* x : {VK_KHR_MAINTENANCE_2_EXTENSION_NAME, VK_KHR_MAINTENANCE_3_EXTENSION_NAME,
                        VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME, VK_KHR_SHADER_DRAW_PARAMETERS_EXTENSION_NAME,
                        VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME, VK_KHR_MULTIVIEW_EXTENSION_NAME,
                        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME})
    CHECK(has(x));
  CHECK(!has(VK_EXT_METAL_OBJECTS_EXTENSION_NAME) && has(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME));
  // A device asking for every advertised extension is created.
  std::vector<const char*> all;
  for (const auto& x : e) all.push_back(x.extensionName);
  Ctx every(all);
  CHECK(every.dev != nullptr);
  unsetenv("VPVK_EXTRA_EXTENSIONS");
}

// VPVK_REPORT_GEOM=1 (games only) reports the Adreno 740's 128 interpolator components so Unity keeps its
// `target 5.0` subshaders; the CTS (variable unset) keeps seeing 124.
void testReportGeomInterpolators() {
  unsetenv("VPVK_REPORT_GEOM");
  Ctx c;
  VkPhysicalDeviceProperties p;
  vkGetPhysicalDeviceProperties(c.pd, &p);
  CHECK(p.limits.maxVertexOutputComponents == 124 && p.limits.maxFragmentInputComponents == 124);
  setenv("VPVK_REPORT_GEOM", "1", 1);
  vkGetPhysicalDeviceProperties(c.pd, &p);
  CHECK(p.limits.maxVertexOutputComponents == 128 && p.limits.maxFragmentInputComponents == 128);
  VkPhysicalDeviceFeatures f;
  vkGetPhysicalDeviceFeatures(c.pd, &f);
  CHECK(f.geometryShader && f.tessellationShader);
  unsetenv("VPVK_REPORT_GEOM");
  vkGetPhysicalDeviceProperties(c.pd, &p);
  CHECK(p.limits.maxVertexOutputComponents == 124 && p.limits.maxFragmentInputComponents == 124);
}

// Announced = implemented, and the compile policy: vkCreateGraphicsPipelines returns with the
// pipelines compiled (no draw skipped while they build); polygon mode POINT draws triangle
// topologies as points; no multi-viewport and no per-pipeline features that are only hints.
void testPipelinePolicy() {
  unsetenv("VPVK_REPORT_GEOM");
  Ctx c;
  VkPhysicalDeviceFeatures f;
  vkGetPhysicalDeviceFeatures(c.pd, &f);
  VkPhysicalDeviceProperties p;
  vkGetPhysicalDeviceProperties(c.pd, &p);
  // Without the Unity compatibility mask the answers are the truth: no geometry / tessellation stage
  // (and no limits that suggest one), 124 interpolator components (Metal's), one viewport.
  CHECK(!f.geometryShader && !f.tessellationShader && !f.multiViewport && !f.shaderFloat64 && !f.shaderInt64);
  CHECK(p.limits.maxGeometryOutputVertices == 0 && p.limits.maxTessellationGenerationLevel == 0);
  CHECK(p.limits.maxVertexOutputComponents == 124 && p.limits.maxViewports == 1);
  CHECK(f.fillModeNonSolid && f.sampleRateShading);

  VkAttachmentDescription att{};
  att.format = VK_FORMAT_R8G8B8A8_UNORM, att.samples = VK_SAMPLE_COUNT_1_BIT;
  att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  VkAttachmentReference cref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkSubpassDescription sp{};
  sp.colorAttachmentCount = 1, sp.pColorAttachments = &cref;
  VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  rpi.attachmentCount = 1, rpi.pAttachments = &att, rpi.subpassCount = 1, rpi.pSubpasses = &sp;
  VkRenderPass rp;
  vkCreateRenderPass(c.dev, &rpi, nullptr, &rp);
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  VkPipelineLayout pl;
  vkCreatePipelineLayout(c.dev, &plci, nullptr, &pl);
  VkShaderModule vs = module(c, "uimask.vert.spv"), fs = module(c, "uimask.frag.spv");
  VkPipelineShaderStageCreateInfo stg[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                         {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
  stg[0].stage = VK_SHADER_STAGE_VERTEX_BIT, stg[0].module = vs, stg[0].pName = "main";
  stg[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT, stg[1].module = fs, stg[1].pName = "main";
  VkPipelineVertexInputStateCreateInfo vis{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo ias{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ias.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.lineWidth = 1;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineColorBlendAttachmentState cba{};
  cba.colorWriteMask = 0xF;
  VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cbs.attachmentCount = 1, cbs.pAttachments = &cba;
  VkViewport vp{0, 0, 64, 64, 0, 1};
  VkRect2D sc{{0, 0}, {64, 64}};
  VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vps.viewportCount = 1, vps.pViewports = &vp, vps.scissorCount = 1, vps.pScissors = &sc;
  VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  gp.stageCount = 2, gp.pStages = stg, gp.pVertexInputState = &vis, gp.pInputAssemblyState = &ias;
  gp.pViewportState = &vps, gp.pRasterizationState = &rs, gp.pMultisampleState = &ms, gp.pColorBlendState = &cbs;
  gp.layout = pl, gp.renderPass = rp;

  // A batch (distinct states: cull mode x front face x polygon mode) comes back compiled, no waitIdle().
  constexpr uint32_t kN = 12;
  VkGraphicsPipelineCreateInfo infos[kN];
  VkPipelineRasterizationStateCreateInfo rss[kN];
  const VkCullModeFlags cull[3] = {VK_CULL_MODE_NONE, VK_CULL_MODE_BACK_BIT, VK_CULL_MODE_FRONT_BIT};
  for (uint32_t i = 0; i < kN; ++i) {
    rss[i] = rs;
    rss[i].cullMode = cull[i % 3];
    rss[i].frontFace = i / 3 % 2 ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rss[i].polygonMode = i < 6 ? VK_POLYGON_MODE_FILL : VK_POLYGON_MODE_POINT;
    infos[i] = gp;
    infos[i].pRasterizationState = &rss[i];
  }
  VkPipeline pipes[kN] = {};
  CHECK(vkCreateGraphicsPipelines(c.dev, VK_NULL_HANDLE, kN, infos, nullptr, pipes) == VK_SUCCESS);
  for (uint32_t i = 0; i < kN; ++i) {
    const PipelineObj* po = pipes[i] ? from<PipelineObj>(pipes[i]) : nullptr;
    CHECK(po && po->ok);
    if (!po || !po->ok) continue;
    CHECK(c.gpu().pipelines().status(po->key) == vpgpu::PipelineCache::Status::Ready);
    CHECK(po->pointMode == (i >= 6));
    CHECK(po->state.primitive == (i >= 6 ? vpgpu::Primitive::Points : vpgpu::Primitive::Triangles));
  }
  // POINT mode leaves lines and points alone (it is a polygon mode).
  ias.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
  rss[0] = rs;
  rss[0].polygonMode = VK_POLYGON_MODE_POINT;
  gp.pRasterizationState = &rss[0];
  VkPipeline lines = VK_NULL_HANDLE;
  CHECK(vkCreateGraphicsPipelines(c.dev, VK_NULL_HANDLE, 1, &gp, nullptr, &lines) == VK_SUCCESS);
  if (lines) CHECK(from<PipelineObj>(lines)->state.primitive == vpgpu::Primitive::Lines);
}

// Host-side Vulkan 1.3 pieces behind the CTS gate: private data, maintenance4 requirements,
// imageless framebuffers (views from VkRenderPassAttachmentBeginInfo), subgroup size control and
// B4G4R4A4 (ABGR4 with R/B swapped; only its mandatory features). Hidden from games.
void testProfile13() {
  vpvk_config_set_quest_identity(1);  // asserts the Quest 3 / Adreno 740 profile: opt-in
  unsetenv("VPVK_EXTRA_EXTENSIONS");
  {
    Ctx g;
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(g.pd, VK_FORMAT_B4G4R4A4_UNORM_PACK16, &fp);
    CHECK(fp.optimalTilingFeatures == 0);
    VkPhysicalDevicePrivateDataFeatures pdf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &pdf};
    vkGetPhysicalDeviceFeatures2(g.pd, &f2);
    CHECK(!pdf.privateData);
  }
  setenv("VPVK_EXTRA_EXTENSIONS", "1", 1);
  Ctx c({VK_EXT_PRIVATE_DATA_EXTENSION_NAME, VK_KHR_MAINTENANCE_4_EXTENSION_NAME,
         VK_KHR_IMAGELESS_FRAMEBUFFER_EXTENSION_NAME, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
         VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME});
  CHECK(c.dev != nullptr);
  {
    // Multisample counts need a use beyond transfers; formats of the CTS format_properties group.
    VkImageFormatProperties ip;
    CHECK(vkGetPhysicalDeviceImageFormatProperties(c.pd, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0, &ip) == VK_SUCCESS);
    CHECK(ip.sampleCounts == VK_SAMPLE_COUNT_1_BIT);
    CHECK(vkGetPhysicalDeviceImageFormatProperties(c.pd, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                                                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 0, &ip) == VK_SUCCESS);
    CHECK(ip.sampleCounts > VK_SAMPLE_COUNT_1_BIT);
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(c.pd, VK_FORMAT_A1R5G5B5_UNORM_PACK16, &fp);
    CHECK(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT);
    vkGetPhysicalDeviceFormatProperties(c.pd, VK_FORMAT_A8B8G8R8_SNORM_PACK32, &fp);
    CHECK(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);
    vkGetPhysicalDeviceFormatProperties(c.pd, VK_FORMAT_R32G32B32_SFLOAT, &fp);
    CHECK(fp.bufferFeatures == VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT);
  }
  VkPhysicalDeviceSubgroupSizeControlFeatures sc{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
  VkPhysicalDeviceImagelessFramebufferFeatures il{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES, &sc};
  VkPhysicalDeviceMaintenance4Features m4{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES, &il};
  VkPhysicalDevicePrivateDataFeatures pdf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES, &m4};
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &pdf};
  vkGetPhysicalDeviceFeatures2(c.pd, &f2);
  CHECK(pdf.privateData && m4.maintenance4 && il.imagelessFramebuffer && sc.subgroupSizeControl);
  VkPhysicalDeviceSubgroupSizeControlProperties scp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
  VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &scp};
  vkGetPhysicalDeviceProperties2(c.pd, &p2);
  CHECK(scp.minSubgroupSize == 32 && scp.maxSubgroupSize == 32);

  // Private data: per (object, slot), 0 until set.
  auto create = reinterpret_cast<PFN_vkCreatePrivateDataSlot>(vkGetDeviceProcAddr(c.dev, "vkCreatePrivateDataSlotEXT"));
  auto set = reinterpret_cast<PFN_vkSetPrivateData>(vkGetDeviceProcAddr(c.dev, "vkSetPrivateData"));
  auto get = reinterpret_cast<PFN_vkGetPrivateData>(vkGetDeviceProcAddr(c.dev, "vkGetPrivateDataEXT"));
  auto destroy = reinterpret_cast<PFN_vkDestroyPrivateDataSlot>(vkGetDeviceProcAddr(c.dev, "vkDestroyPrivateDataSlot"));
  CHECK(create && set && get && destroy);
  if (create && set && get && destroy) {
    VkPrivateDataSlotCreateInfo pci{VK_STRUCTURE_TYPE_PRIVATE_DATA_SLOT_CREATE_INFO};
    VkPrivateDataSlot a = VK_NULL_HANDLE, b = VK_NULL_HANDLE;
    CHECK(create(c.dev, &pci, nullptr, &a) == VK_SUCCESS && create(c.dev, &pci, nullptr, &b) == VK_SUCCESS);
    const uint64_t h = uint64_t(reinterpret_cast<uintptr_t>(c.dev));
    uint64_t v = 7;
    get(c.dev, VK_OBJECT_TYPE_DEVICE, h, a, &v);
    CHECK(v == 0);
    CHECK(set(c.dev, VK_OBJECT_TYPE_DEVICE, h, a, 42) == VK_SUCCESS);
    get(c.dev, VK_OBJECT_TYPE_DEVICE, h, a, &v);
    CHECK(v == 42);
    get(c.dev, VK_OBJECT_TYPE_DEVICE, h, b, &v);
    CHECK(v == 0);
    destroy(c.dev, a, nullptr);
    destroy(c.dev, b, nullptr);
  }

  // maintenance4: requirements without an object match those of a real one.
  auto bufReq = reinterpret_cast<PFN_vkGetDeviceBufferMemoryRequirements>(
      vkGetDeviceProcAddr(c.dev, "vkGetDeviceBufferMemoryRequirementsKHR"));
  CHECK(bufReq != nullptr);
  if (bufReq) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = 1000;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkDeviceBufferMemoryRequirements q{VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS, nullptr, &bci};
    VkMemoryRequirements2 r{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    bufReq(c.dev, &q, &r);
    VkBuffer b = VK_NULL_HANDLE;
    vkCreateBuffer(c.dev, &bci, nullptr, &b);
    VkMemoryRequirements real;
    vkGetBufferMemoryRequirements(c.dev, b, &real);
    CHECK(r.memoryRequirements.size == real.size && r.memoryRequirements.memoryTypeBits == real.memoryTypeBits);
    vkDestroyBuffer(c.dev, b, nullptr);
  }
  // Creación temporal fallida (multiplanar de 1x1): requisitos a cero, sin desreferenciar un handle nulo.
  auto imgReq = reinterpret_cast<PFN_vkGetDeviceImageMemoryRequirements>(
      vkGetDeviceProcAddr(c.dev, "vkGetDeviceImageMemoryRequirementsKHR"));
  CHECK(imgReq != nullptr);
  if (imgReq) {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
    ici.extent = {1, 1, 1};
    ici.mipLevels = ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    VkImage probe = VK_NULL_HANDLE;
    const bool fails = vkCreateImage(c.dev, &ici, nullptr, &probe) != VK_SUCCESS;
    if (!fails) vkDestroyImage(c.dev, probe, nullptr);
    CHECK(fails);
    VkDeviceImageMemoryRequirements q{VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS, nullptr, &ici};
    VkMemoryRequirements2 r{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    r.memoryRequirements = {123, 456, 0xFF};
    imgReq(c.dev, &q, &r);
    CHECK(r.memoryRequirements.size == 0 && r.memoryRequirements.alignment == 0 && r.memoryRequirements.memoryTypeBits == 0);
  }

  // B4G4R4A4: the mandatory features only.
  VkFormatProperties fp;
  vkGetPhysicalDeviceFormatProperties(c.pd, VK_FORMAT_B4G4R4A4_UNORM_PACK16, &fp);
  const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                                    VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
  CHECK((fp.optimalTilingFeatures & need) == need && !(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT));
  CHECK(fp.bufferFeatures == 0);
  CHECK(toCoreFormat(VK_FORMAT_B4G4R4A4_UNORM_PACK16) == vpgpu::Format::ABGR4Unorm);

  // Imageless framebuffer: created without views; a pass begun with one draws into it.
  VkDeviceMemory mem = alloc(c, 1 << 20, 0);
  VkImage img = image(c, VK_FORMAT_R8G8B8A8_UNORM, 16, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, mem);
  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.image = img;
  vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vi.format = VK_FORMAT_R8G8B8A8_UNORM;
  vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkImageView view = VK_NULL_HANDLE;
  vkCreateImageView(c.dev, &vi, nullptr, &view);
  VkAttachmentDescription ad{};
  ad.format = VK_FORMAT_R8G8B8A8_UNORM;
  ad.samples = VK_SAMPLE_COUNT_1_BIT;
  ad.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  ad.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  ad.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
  VkAttachmentReference ar{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkSubpassDescription sd{};
  sd.colorAttachmentCount = 1;
  sd.pColorAttachments = &ar;
  VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  rpi.attachmentCount = 1, rpi.pAttachments = &ad, rpi.subpassCount = 1, rpi.pSubpasses = &sd;
  VkRenderPass rp = VK_NULL_HANDLE;
  vkCreateRenderPass(c.dev, &rpi, nullptr, &rp);
  VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  fci.flags = VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT;
  fci.renderPass = rp, fci.attachmentCount = 1, fci.width = 16, fci.height = 16, fci.layers = 1;
  VkFramebuffer fb = VK_NULL_HANDLE;
  vkCreateFramebuffer(c.dev, &fci, nullptr, &fb);
  CHECK(from<Framebuffer>(fb)->views.empty());
  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  VkCommandPool pool = VK_NULL_HANDLE;
  vkCreateCommandPool(c.dev, &pci, nullptr, &pool);
  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = pool, cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, cai.commandBufferCount = 1;
  VkCommandBuffer cb = VK_NULL_HANDLE;
  vkAllocateCommandBuffers(c.dev, &cai, &cb);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(cb, &bi);
  VkRenderPassAttachmentBeginInfo abi{VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO};
  abi.attachmentCount = 1, abi.pAttachments = &view;
  VkClearValue cv{};
  VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, &abi};
  rbi.renderPass = rp, rbi.framebuffer = fb, rbi.renderArea = {{0, 0}, {16, 16}};
  rbi.clearValueCount = 1, rbi.pClearValues = &cv;
  vkCmdBeginRenderPass(cb, &rbi, VK_SUBPASS_CONTENTS_INLINE);
  const Framebuffer* used = from<CommandBuffer>(cb)->rec->bound.framebuffer;
  CHECK(used && used != from<Framebuffer>(fb) && used->views.size() == 1 && used->views[0] == from<ImageView>(view));
  vkCmdEndRenderPass(cb);
  vkEndCommandBuffer(cb);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1, si.pCommandBuffers = &cb;
  CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS);
  vkQueueWaitIdle(c.queue);
  vkDestroyCommandPool(c.dev, pool, nullptr);
  vkDestroyFramebuffer(c.dev, fb, nullptr);
  vkDestroyRenderPass(c.dev, rp, nullptr);
  vkDestroyImageView(c.dev, view, nullptr);
  vkDestroyImage(c.dev, img, nullptr);
  vkFreeMemory(c.dev, mem, nullptr);
  unsetenv("VPVK_EXTRA_EXTENSIONS");
}

// Pipeline recipes survive translator changes: a compute pipeline recorded through the driver is
// rebuilt from its stored SPIR-V + lowering options under another translator version (same MSL id
// and key); a recipe whose stored MSL id is stale is remapped to the real one by the prewarm;
// pre-v3 files are discarded; recording the same pipeline twice adds nothing to the file.
void testRecipes() {
  char tmpl[] = "/tmp/vpvk_recipes_XXXXXX";
  const char* dir = mkdtemp(tmpl);
  CHECK(dir != nullptr);
  if (!dir) return;
  setenv("VP_PIPELINE_CACHE_DIR", dir, 1);
  const std::string file = std::string(dir) + "/recipes.bin";
  uint64_t key = 0, msl = 0;
  {
    Ctx c;
    VkDescriptorSetLayoutBinding b[4] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = 4, lci.pBindings = b;
    VkDescriptorSetLayout sl;
    vkCreateDescriptorSetLayout(c.dev, &lci, nullptr, &sl);
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 100};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1, plci.pSetLayouts = &sl, plci.pushConstantRangeCount = 1, plci.pPushConstantRanges = &pcr;
    VkPipelineLayout pl;
    vkCreatePipelineLayout(c.dev, &plci, nullptr, &pl);
    VkShaderModule cs = module(c, "cull.comp.opt.spv");
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT, ci.stage.module = cs, ci.stage.pName = "main";
    ci.layout = pl;
    VkPipeline pipe[2];
    vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe[0]);
    const PipelineObj* po = from<PipelineObj>(pipe[0]);
    CHECK(po->ok && po->state.computeShader != 0);
    key = po->key.value, msl = po->state.computeShader;
    std::error_code ec;
    const auto size1 = std::filesystem::file_size(file, ec);
    const uint64_t tr0 = stageTranslations();
    vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe[1]);  // already recorded
    CHECK(std::filesystem::file_size(file, ec) == size1);
    CHECK(stageTranslations() == tr0);  // a recorded stage is not translated again
    CHECK(from<PipelineObj>(pipe[1])->ok && from<PipelineObj>(pipe[1])->key.value == key);
    CHECK(from<PipelineObj>(pipe[1])->used.size() == po->used.size() && from<PipelineObj>(pipe[1])->pushStages == po->pushStages);
    c.gpu().pipelines().waitIdle();
    for (VkPipeline p : pipe) vkDestroyPipeline(c.dev, p, nullptr);
    vkDestroyShaderModule(c.dev, cs, nullptr);
    vkDestroyPipelineLayout(c.dev, pl, nullptr);
    vkDestroyDescriptorSetLayout(c.dev, sl, nullptr);
  }
  {  // a later run (new device, recipes read from disk): the same pipeline without translating
    Ctx c;
    VkDescriptorSetLayoutBinding b[4] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = 4, lci.pBindings = b;
    VkDescriptorSetLayout sl;
    vkCreateDescriptorSetLayout(c.dev, &lci, nullptr, &sl);
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 100};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1, plci.pSetLayouts = &sl, plci.pushConstantRangeCount = 1, plci.pPushConstantRanges = &pcr;
    VkPipelineLayout pl;
    vkCreatePipelineLayout(c.dev, &plci, nullptr, &pl);
    VkShaderModule cs = module(c, "cull.comp.opt.spv");
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT, ci.stage.module = cs, ci.stage.pName = "main";
    ci.layout = pl;
    const uint64_t tr0 = stageTranslations();
    VkPipeline pipe;
    vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe);
    const PipelineObj* po = from<PipelineObj>(pipe);
    CHECK(po->ok && po->key.value == key && po->state.computeShader == msl && !po->used.empty());
    CHECK(stageTranslations() == tr0);
    c.gpu().pipelines().waitIdle();
    vkDestroyPipeline(c.dev, pipe, nullptr);
    vkDestroyShaderModule(c.dev, cs, nullptr);
    vkDestroyPipelineLayout(c.dev, pl, nullptr);
    vkDestroyDescriptorSetLayout(c.dev, sl, nullptr);
  }
  vpgpu::PipelineState st;
  {
    RecipeStore s(dir);  // same translator: used as stored
    CHECK(s.pipelineCount() == 1 && s.stageInputs().size() == 1 && !s.needsRefresh());
    CHECK(s.stateFor(key, st) && st.computeShader == msl);
  }
  {
    RecipeStore s(dir, kTranslatorVersion + 1);  // translator bumped: retranslated from SPIR-V
    CHECK(s.needsRefresh());
    const RecipeStore::RefreshResult r = s.refresh(4);
    CHECK(r.retranslated == 1 && r.failed == 0 && r.droppedPipelines == 0 && r.rewritten);
    vpgpu::PipelineState again;
    CHECK(s.stateFor(key, again) && again.computeShader == msl);
  }
  StageInput in;
  {
    RecipeStore s(dir, kTranslatorVersion + 1);  // the rewritten file is a straight hit
    CHECK(!s.needsRefresh() && s.pipelineCount() == 1);
    const auto inputs = s.stageInputs();
    CHECK(inputs.size() == 1);
    if (inputs.size() == 1) in = inputs[0];
  }
  // A recipe from an "older translator" whose MSL had another id: the prewarm retranslates it and
  // compiles the pipeline under the key the driver computes today.
  const std::string dir2 = std::string(dir) + "/old";
  {
    RecipeStore s(dir2, 7);
    ShaderBlob old;
    old.source = "// old translator output";
    s.addStage(in, in.hash(), spirvFile("cull.comp.opt.spv"), 0xBAD, old);
    vpgpu::PipelineState stale = st;
    stale.computeShader = 0xBAD;
    s.addPipeline(stale, {0, 0, in.hash()});
    CHECK(s.pipelineCount() == 1);
  }
  {
    RecipeStore s(dir2, 8);
    std::unique_ptr<Platform> platform = createPlatform();
    PrewarmResult r;
    {
      vpgpu::Device gpu(platform->backend(), 2, [](const std::string&) {});
      r = prewarm(*platform, gpu.pipelines(), s, nullptr);
      CHECK(gpu.pipelines().status({key}) == vpgpu::PipelineCache::Status::Ready);
    }
    CHECK(r.retranslated == 1 && r.recipes == 1 && r.failed == 0);
    vpgpu::PipelineState now;
    CHECK(s.keys().size() == 1 && s.stateFor(key, now) && now.computeShader == msl);
    CHECK(s.shaderCount() == 1);  // the stale MSL blob was dropped
    // Same input, same translator version, other MSL (translator changed without a bump): a
    // drift record makes the next load retranslate everything.
    ShaderBlob other;
    other.source = "// unbumped translator output";
    s.addStage(in, in.hash(), spirvFile("cull.comp.opt.spv"), 0xBEEF, other);
  }
  {
    RecipeStore s(dir2, 8);
    CHECK(s.needsRefresh());
    const RecipeStore::RefreshResult r = s.refresh(2);
    vpgpu::PipelineState now;
    CHECK(r.retranslated == 1 && s.stateFor(key, now) && now.computeShader == msl);
  }
  {
    RecipeStore s(dir2, 8);
    CHECK(!s.needsRefresh());
  }
  // Pre-v3 file (MSL only, no translator inputs): discarded, the store starts over.
  const std::string dir3 = std::string(dir) + "/v2";
  std::filesystem::create_directories(dir3);
  if (FILE* f = std::fopen((dir3 + "/recipes.bin").c_str(), "wb")) {
    const uint32_t hdr[2] = {2, uint32_t(vpgpu::PipelineCache::encodedSize())};
    std::fwrite("VQRC", 1, 4, f);
    std::fwrite(hdr, sizeof hdr, 1, f);
    std::fwrite("S\0\0\0\0", 1, 5, f);
    std::fclose(f);
  }
  {
    RecipeStore s(dir3);
    CHECK(s.enabled() && s.pipelineCount() == 0 && s.shaderCount() == 0);
  }
  unsetenv("VP_PIPELINE_CACHE_DIR");
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

// VPKTAA inline jitter: which draws of a submission get their viewport moved, and the counters.
void testTaa() {
  // Hand-built ops: a 1x eye pass (depth, eye size) with a geometry draw, a fullscreen draw and a
  // draw without depth test; a same-size pass without depth; a multiview pass of another size.
  auto rendering = [](uint32_t w, uint32_t h, bool depth, uint32_t viewMask) {
    OpBeginRendering b;
    b.area = {{0, 0}, {w, h}};
    b.viewMask = viewMask;
    if (depth) b.depthWidth = w, b.depthHeight = h;
    return b;
  };
  auto draw = [](bool depthTested, bool vb, bool indexed, uint32_t count, float vpH) {
    OpDraw d;
    d.draw.count = count;
    d.draw.viewport = {10, 20, 100, vpH, 0, 1};
    if (indexed) d.draw.index = vpgpu::IndexBuffer{};
    d.depthTested = depthTested, d.vertexBuffers = vb;
    return d;
  };
  Recorder rec;
  rec.ops.emplace_back(rendering(1920, 1824, true, 0));
  rec.ops.emplace_back(draw(true, true, true, 300, 100));   // jittered
  rec.ops.emplace_back(draw(true, false, false, 3, 100));   // fullscreen: skipped
  rec.ops.emplace_back(draw(false, true, true, 300, 100));  // no depth test: skipped
  rec.ops.emplace_back(draw(true, false, false, 900, -100));  // vertex-pulled geometry, flipped: jittered
  rec.ops.emplace_back(OpEndPass{});
  rec.ops.emplace_back(rendering(1920, 1824, false, 0));
  rec.ops.emplace_back(draw(true, true, true, 300, 100));   // no depth attachment: untouched, not counted
  rec.ops.emplace_back(OpEndPass{});
  rec.ops.emplace_back(rendering(1536, 1459, true, 3));
  rec.ops.emplace_back(draw(true, true, false, 36, 100));   // multiview at 80 % render scale: jittered
  rec.ops.emplace_back(OpEndPass{});
  rec.ops.emplace_back(rendering(512, 512, true, 3));
  rec.ops.emplace_back(draw(true, true, false, 36, 100));   // multiview probe/shadow size: untouched
  rec.ops.emplace_back(OpEndPass{});
  rec.ops.emplace_back(rendering(1024, 1024, true, 0));
  rec.ops.emplace_back(draw(true, true, true, 300, 100));   // shadow map size: untouched
  rec.ops.emplace_back(OpEndPass{});

  auto viewports = [](const vpgpu::Frame& f) {
    std::vector<vpgpu::Viewport> v;
    for (const auto& c : f.commands())
      if (const auto* rp = std::get_if<vpgpu::RenderPass>(&c))
        for (const auto& d : rp->draws) v.push_back(d.viewport);
    return v;
  };
  {
    vpgpu::Frame f;
    TranslateStats st;
    translate(rec, f, st);  // disabled: nothing moves
    const auto v = viewports(f);
    CHECK(v.size() == 8 && st.taaJittered == 0 && st.taaSkipped == 0);
    for (const auto& x : v) CHECK(x.x == 10 && x.y == 20);
  }
  TaaJitter j;
  j.enabled = true, j.width = 1920, j.height = 1824, j.jx = 0.25f, j.jy = -0.375f, j.tag = 42;
  vpgpu::Frame f;
  TranslateStats st;
  translate(rec, f, st, &j);
  const auto v = viewports(f);
  CHECK(v.size() == 8);
  if (v.size() == 8) {
    CHECK(v[0].x == 10.25f && v[0].y == 19.625f && v[0].width == 100);
    CHECK(v[1].x == 10 && v[1].y == 20);
    CHECK(v[2].x == 10 && v[2].y == 20);
    CHECK(v[3].x == 10.25f && v[3].y == 19.625f && v[3].height == -100);
    CHECK(v[4].x == 10 && v[4].y == 20);
    CHECK(v[5].x == 10.25f && v[5].y == 19.625f);
    CHECK(v[6].x == 10 && v[6].y == 20);
    CHECK(v[7].x == 10 && v[7].y == 20);
  }
  CHECK(st.taaJittered == 3 && st.taaSkipped == 2 && st.taaTargetW == 1920 && st.taaTargetH == 1824);

  // Counters through the C API: two submissions with one jitter, one with another.
  vpvk_taa_applied a;
  vpvk_taa_get_applied(&a);  // reset
  taaRecordSubmit(1, j, 3, 2, 1920, 1824);
  taaRecordSubmit(2, j, 1, 0, 512, 512);
  TaaJitter k = j;
  k.jx = -0.125f, k.tag = 43;
  taaRecordSubmit(3, k, 2, 1, 1920, 1824);
  taaRecordSubmit(4, k, 0, 4, 0, 0);  // nothing jittered: only skipped counts
  CHECK(vpvk_taa_get_applied(&a) == 1);
  CHECK(a.tag == 43 && a.jx == -0.125f && a.jy == -0.375f && a.target_w == 1920 && a.target_h == 1824);
  CHECK(a.jittered_draws == 6 && a.skipped_draws == 7 && a.submits == 3 && a.mixed == 2);
  CHECK(vpvk_taa_get_applied(&a) == 0 && a.submits == 0 && a.jittered_draws == 0 && a.tag == 43);

  // The setter clamps to half a pixel and the jitter is off until enabled.
  CHECK(!taaJitter().enabled);
  vpvk_taa_jitter in{1, 1920, 1824, 0.9f, -2.0f, 7};
  vpvk_taa_set_jitter(&in);
  const TaaJitter t = taaJitter();
  CHECK(t.enabled && t.jx == 0.5f && t.jy == -0.5f && t.tag == 7 && t.width == 1920);
  in.enabled = 0;
  vpvk_taa_set_jitter(&in);
  CHECK(!taaJitter().enabled);
  CHECK(vpvk_native_device() == nullptr);  // no device in this process
}

// VPKTAA nativo: jitter Halton propio solo en pases de escena, puerta, Catmull-Rom, reconstrucción
// y reproyección de cámara (la misma matemática que vpktaa_native.metal).
void testTaaNative() {
  auto near = [](float a, float b, float eps = 1e-5f) { return std::fabs(a - b) <= eps; };
  // Halton(2,3): 8 fases distintas en [-0.5, 0.5), media ~0, ciclo de 8.
  float mx = 0, my = 0;
  for (uint64_t i = 0; i < taa::kPhases; ++i) {
    const auto h = taa::haltonJitter(i);
    CHECK(h[0] >= -0.5f && h[0] < 0.5f && h[1] >= -0.5f && h[1] < 0.5f);
    mx += h[0], my += h[1];
    for (uint64_t k = 0; k < i; ++k) CHECK(taa::haltonJitter(k) != h);
    CHECK(taa::haltonJitter(i + taa::kPhases) == h);
  }
  CHECK(std::fabs(mx / 8) < 0.07f && std::fabs(my / 8) < 0.07f);
  CHECK(near(taa::haltonJitter(0)[0], 0.0f) && near(taa::haltonJitter(0)[1], 1.0f / 3 - 0.5f));

  // Mover el viewport jx px = sumar 2·jx/W·w a la x de clip: la posición en píxel coincide.
  {
    const float W = 1920, H = 1824, jx = 0.25f, jy = -0.375f, cx = 0.3f, cy = -0.2f, cw = 2.5f;
    const float px = (cx / cw * 0.5f + 0.5f) * W + jx, py = (cy / cw * 0.5f + 0.5f) * H + jy;  // Vulkan: +y abajo
    const auto o = taa::jitterClipOffset(jx, jy, W, H, cw);
    const float qx = ((cx + o[0]) / cw * 0.5f + 0.5f) * W, qy = ((cy - o[1]) / cw * 0.5f + 0.5f) * H;
    CHECK(near(px, qx, 1e-3f) && near(py, qy, 1e-3f));
  }

  // Puerta: apagado por defecto; encendido el traductor usa la fase Halton, no el jx/jy del host,
  // y solo en el pase de escena (profundidad, tamaño de ojo), nunca en UI / shadow / postproceso.
  CHECK(!taaNativeOn() && vpvk_taa_native_enabled() == 0);
  vpvk_taa_jitter in{1, 1920, 1824, 0.1f, 0.1f, 9};
  vpvk_taa_set_jitter(&in);
  CHECK(taaJitter().jx == 0.1f);
  vpvk_config_set_taa_native(1);
  CHECK(taaNativeOn() && vpvk_taa_native_enabled() == 1);
  CHECK(vpvk_taa_ui_mask_enabled() == (taaUiMaskOn() ? 1 : 0));
  const TaaJitter j = taaJitter();
  const auto h0 = taa::haltonJitter(0);
  CHECK(j.enabled && j.jx == h0[0] && j.jy == h0[1]);
  {
    Recorder rec;
    OpBeginRendering scene;
    scene.area = {{0, 0}, {1920, 1824}};
    scene.depthWidth = 1920, scene.depthHeight = 1824;
    OpBeginRendering ui = scene;  // overlay del mismo tamaño sin profundidad
    ui.depthWidth = ui.depthHeight = 0;
    OpBeginRendering shadow;
    shadow.area = {{0, 0}, {2048, 2048}};
    shadow.depthWidth = shadow.depthHeight = 2048;
    OpDraw geo;
    geo.draw.count = 300, geo.draw.index = vpgpu::IndexBuffer{};
    geo.draw.viewport = {0, 0, 1920, 1824, 0, 1};
    geo.depthTested = geo.vertexBuffers = true;
    OpDraw post = geo;  // postproceso: triángulo de pantalla completa
    post.draw.count = 3, post.draw.index.reset(), post.vertexBuffers = false;
    rec.ops.emplace_back(scene), rec.ops.emplace_back(geo), rec.ops.emplace_back(post), rec.ops.emplace_back(OpEndPass{});
    rec.ops.emplace_back(ui), rec.ops.emplace_back(geo), rec.ops.emplace_back(OpEndPass{});
    rec.ops.emplace_back(shadow), rec.ops.emplace_back(geo), rec.ops.emplace_back(OpEndPass{});
    vpgpu::Frame f;
    TranslateStats st;
    translate(rec, f, st, &j);
    std::vector<vpgpu::Viewport> v;
    for (const auto& c : f.commands())
      if (const auto* rp = std::get_if<vpgpu::RenderPass>(&c))
        for (const auto& d : rp->draws) v.push_back(d.viewport);
    CHECK(v.size() == 4);
    if (v.size() == 4) {
      CHECK(v[0].x == h0[0] && v[0].y == h0[1]);  // escena
      CHECK(v[1].x == 0 && v[1].y == 0);          // pantalla completa
      CHECK(v[2].x == 0 && v[2].y == 0);          // UI sin profundidad
      CHECK(v[3].x == 0 && v[3].y == 0);          // shadow map
    }
    CHECK(st.taaJittered == 1);
    // La fase avanza solo tras un frame con escena movida.
    taaRecordSubmit(100, j, st.taaJittered, st.taaSkipped, st.taaTargetW, st.taaTargetH);
    const TaaNativeApplied a = taaNativeTake();
    CHECK(a.jittered == 1 && a.jx == h0[0] && a.targetW == 1920 && a.phase == 0);
    const auto h1 = taa::haltonJitter(1);
    CHECK(taaJitter().jx == h0[0]);  // encode fallido (sin taaNativeAdvance): la fase no se consume
    taaNativeAdvance(a.phase + 5);   // fase ajena: ignorada
    CHECK(taaJitter().jx == h0[0]);
    taaNativeAdvance(a.phase);       // pase codificado
    CHECK(taaJitter().jx == h1[0]);
    CHECK(taaNativeTake().jittered == 0 && taaJitter().jx == h1[0]);  // sin escena no gasta fase
  }
  vpvk_taa_applied ap;
  vpvk_taa_get_applied(&ap);
  CHECK(vpvk_taa_native_encode(nullptr, nullptr) == 0);  // fuera de vpvk_native_submit: rechazado
  CHECK(vpvk_depth_resolve_encode(nullptr, nullptr, 0, 1, 0) == nullptr);
  vpvk_config_set_taa_native(0);
  CHECK(!taaNativeOn() && vpvk_taa_ui_mask_enabled() == 0);
  in.enabled = 0;
  vpvk_taa_set_jitter(&in);

  // Catmull-Rom: suma 1, interpola en los nodos, forma de 5 lecturas equivalente a 4 pesos.
  for (float fr : {0.0f, 0.25f, 0.5f, 0.8f}) {
    const auto w = taa::catmullRom(fr);
    CHECK(near(w[0] + w[1] + w[2] + w[3], 1.0f));
    // Reproduce una rampa lineal exactamente: sum w_i·(i-1) = f.
    CHECK(near(-w[0] + w[2] + 2 * w[3], fr));
    const auto c5 = taa::catmullRom5(fr);
    CHECK(near(c5[1], w[1] + w[2]) && near(c5[3] * c5[1], w[2]));
  }
  CHECK(taa::catmullRom(0)[1] == 1.0f && taa::catmullRom(0)[0] == 0.0f);

  // Reconstrucción: sin jitter, pico en el centro y simétrica; con jitter, el texel más cercano.
  {
    const auto r = taa::reconstruction(0, 0);
    float sum = 0;
    for (float w : r.weights) sum += w;
    CHECK(near(sum, 1.0f) && r.base[0] == 0 && r.base[1] == 0 && r.nearest[0] == 0);
    CHECK(r.weights[4] > r.weights[3] && near(r.weights[3], r.weights[5]) && near(r.weights[0], r.weights[8]));
    const auto q = taa::reconstruction(0.4f, -0.45f);  // contenido 0.4 px a la derecha
    CHECK(q.base[0] == 0 && q.base[1] == 0 && q.weights[5] > q.weights[3] && q.weights[1] > q.weights[7]);
    const auto z = taa::reconstruction(0.6f, 0);
    CHECK(z.base[0] == 1 && z.nearest[0] == 0);
  }

  // Profundidad: estándar e invertida, finita e infinita.
  CHECK(near(taa::viewDistance(0, 0.1f, 100, false), 0.1f));
  CHECK(near(taa::viewDistance(1, 0.1f, 100, false), 100, 1e-2f));
  CHECK(near(taa::viewDistance(1, 0.1f, 0, true), 0.1f));
  CHECK(near(taa::viewDistance(0.01f, 0.1f, 0, true), 10.0f, 1e-3f));
  CHECK(taa::emptyDepth(0, true) && taa::emptyDepth(1, false) && !taa::emptyDepth(0.5f, true));

  // Reproyección: cámara quieta = identidad; giro de yaw desplaza en x; traslación con paralaje.
  {
    taa::EyeView a;
    a.tangents = {1, 1, 1, 1};
    float pu, pv, pd;
    CHECK(taa::reprojectUv(taa::reprojection(a, a), a.tangents, a.tangents, 0.3f, 0.7f, 5, pu, pv, pd));
    CHECK(near(pu, 0.3f) && near(pv, 0.7f) && near(pd, 5));
    // La cabeza gira 10° a la izquierda (yaw +, eje y): un punto al frente queda a la derecha.
    taa::EyeView b = a;
    const float h = 0.5f * 10.0f * 3.14159265f / 180;
    b.orientation = {0, std::sin(h), 0, std::cos(h)};
    // Del ojo actual (girado) al anterior (a): el centro del actual estaba a la izquierda en a.
    CHECK(taa::reprojectUv(taa::reprojection(b, a), b.tangents, a.tangents, 0.5f, 0.5f, 0, pu, pv, pd));
    CHECK(near(pu, 0.5f * (1 - std::tan(10 * 3.14159265f / 180)), 1e-4f) && near(pv, 0.5f) && pd == 0);
    // Traslación de 0.1 m a la derecha: el punto a 1 m en el centro estaba 0.1 m a la derecha antes
    // (u anterior = 0.5 + 0.1/2 con tangentes 1); a infinito no se mueve.
    taa::EyeView c = a;
    c.position = {0.1f, 0, 0};
    CHECK(taa::reprojectUv(taa::reprojection(c, a), c.tangents, a.tangents, 0.5f, 0.5f, 1, pu, pv, pd));
    CHECK(near(pu, 0.55f) && near(pv, 0.5f) && near(pd, 1));
    CHECK(taa::reprojectUv(taa::reprojection(c, a), c.tangents, a.tangents, 0.5f, 0.5f, 0, pu, pv, pd) && near(pu, 0.5f));
    // Detrás de la cámara anterior: rechazado.
    taa::EyeView d = a;
    d.position = {0, 0, 2};  // 2 m detrás: el punto a 1 m queda detrás de la cámara anterior
    CHECK(!taa::reprojectUv(taa::reprojection(d, a), d.tangents, a.tangents, 0.5f, 0.5f, 1, pu, pv, pd));
  }
}


// MSAA override / clamp rule, effective counts of images, pipeline variants, eye-pass detection and
// the vkGetMTLTextureMVK export. The null platform reports a device maximum of 4.
void testMsaa() {
  // Pure rule: game 1x never gains MSAA; forced 1x wins; override clamps to the device maximum.
  CHECK(msaaEffective(1, false, 8, 8) == 1);
  CHECK(msaaEffective(4, false, 0, 8) == 4);
  CHECK(msaaEffective(4, true, 8, 8) == 1);
  CHECK(msaaEffective(2, false, 8, 4) == 4);
  CHECK(msaaEffective(4, false, 2, 8) == 2);
  CHECK(msaaEffective(8, false, 0, 4) == 4);  // the game's own count, unsupported: clamped
  CHECK(msaaEffective(8, false, 0, 1) == 1);
  CHECK(msaaEffective(4, false, 8, 2) == 2);
  CHECK(vpvk_msaa_device_max() == 4);

  Ctx c({VK_KHR_MULTIVIEW_EXTENSION_NAME});
  VkDeviceMemory m = alloc(c, 64 << 20, 0);
  auto msImage = [&](VkFormat f, VkSampleCountFlagBits n, VkImageUsageFlags usage) {
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = f;
    ii.extent = {2048, 2048, 1};
    ii.mipLevels = 1;
    ii.arrayLayers = 2;
    ii.samples = n;
    ii.usage = usage;
    VkImage i = VK_NULL_HANDLE;
    vkCreateImage(c.dev, &ii, nullptr, &i);
    if (i) vkBindImageMemory(c.dev, i, m, 0);
    return i;
  };
  const VkImageUsageFlags colorUse = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
  const VkImageUsageFlags depthUse = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  vpvk_config_set_msaa_override(8);  // more than the device: clamped to 4
  VkImage col = msImage(VK_FORMAT_R8G8B8A8_SRGB, VK_SAMPLE_COUNT_2_BIT, colorUse);
  VkImage dep = msImage(VK_FORMAT_D24_UNORM_S8_UINT, VK_SAMPLE_COUNT_2_BIT, depthUse);
  VkImage one = msImage(VK_FORMAT_R8G8B8A8_SRGB, VK_SAMPLE_COUNT_1_BIT, colorUse | VK_IMAGE_USAGE_SAMPLED_BIT);
  CHECK(from<Image>(col)->desc.samples == 4 && from<Image>(col)->gameSamples == 2);
  CHECK(from<Image>(dep)->desc.samples == 4);
  CHECK(from<Image>(one)->desc.samples == 1);  // the resolve target stays 1x
  vpvk_config_set_msaa_override(0);
  VkImage own = msImage(VK_FORMAT_R8G8B8A8_SRGB, VK_SAMPLE_COUNT_2_BIT, colorUse);
  CHECK(from<Image>(own)->desc.samples == 2);

  // A pipeline made with one count drawn into a pass of another: a variant with the pass's count.
  PipelineObj p;
  p.gameSamples = 2;
  p.state.samples = 2;
  p.state.label = "test.pipe";
  p.key = p.state.key();
  CHECK(pipelineFor(nullptr, p, 2).value == p.key.value);
  CHECK(pipelineFor(nullptr, p, 0).value == p.key.value);
  const vpgpu::PipelineKey v4 = pipelineFor(nullptr, p, 4);
  vpgpu::PipelineState s4 = p.state;
  s4.samples = 4;
  CHECK(v4.value != p.key.value && v4.value == s4.key().value);
  CHECK(pipelineFor(nullptr, p, 4).value == v4.value && p.variants.size() == 1);

  // Dynamic rendering into the overridden 4x attachments: the recorder tracks the pass's
  // effective count, and the detection reports the game's original 2x (multiview eye pass).
  VkImageView cv, dv;
  for (auto [img, view, aspect] : {std::tuple{col, &cv, VK_IMAGE_ASPECT_COLOR_BIT}, std::tuple{dep, &dv, VK_IMAGE_ASPECT_DEPTH_BIT}}) {
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = img;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    vi.subresourceRange = {VkImageAspectFlags(aspect), 0, 1, 0, 2};
    vkCreateImageView(c.dev, &vi, nullptr, view);
  }
  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  VkCommandPool pool;
  vkCreateCommandPool(c.dev, &pci, nullptr, &pool);
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = pool;
  ai.commandBufferCount = 1;
  VkCommandBuffer cb;
  vkAllocateCommandBuffers(c.dev, &ai, &cb);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(cb, &bi);
  VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  ca.imageView = cv;
  ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  VkRenderingAttachmentInfo da = ca;
  da.imageView = dv;
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, {2048, 2048}};
  ri.viewMask = 0b11;
  ri.colorAttachmentCount = 1;
  ri.pColorAttachments = &ca;
  ri.pDepthAttachment = &da;
  vkCmdBeginRendering(cb, &ri);
  CHECK(from<CommandBuffer>(cb)->rec->bound.passSamples == 4);
  vkCmdEndRendering(cb);
  vkEndCommandBuffer(cb);
  CHECK(vpvk_msaa_detected() == 2);

  // MoltenVK's export answers for any bound image (null backend: no texture, no crash).
  void* t = reinterpret_cast<void*>(1);
  vkGetMTLTextureMVK(one, &t);
  CHECK(t == nullptr);
  CHECK(vkGetDeviceProcAddr(c.dev, "vkGetMTLTextureMVK") != nullptr);
  CHECK(vkGetInstanceProcAddr(c.inst, "vkGetMTLTextureMVK") != nullptr);

  vkDestroyCommandPool(c.dev, pool, nullptr);
  vkDestroyImageView(c.dev, cv, nullptr);
  vkDestroyImageView(c.dev, dv, nullptr);
  for (VkImage i : {col, dep, one, own}) vkDestroyImage(c.dev, i, nullptr);
  vkFreeMemory(c.dev, m, nullptr);
}

}  // namespace

// Occlusion queries: a pool takes visibility slots; reset / end / copy translate into a fill, the
// pending mark and per-query copies; results read the counter once the submission finished.
void testQueries() {
  Ctx c;
  VkDeviceMemory m = alloc(c, 1 << 16, 1);
  VkBuffer out = buffer(c, 4096, m, 0);
  VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  qi.queryType = VK_QUERY_TYPE_OCCLUSION;
  qi.queryCount = 4;
  VkQueryPool qp;
  CHECK(vkCreateQueryPool(c.dev, &qi, nullptr, &qp) == VK_SUCCESS);
  QueryPoolObj* p = from<QueryPoolObj>(qp);
  CHECK(p->base != UINT32_MAX && from<Device>(c.dev)->visibility.id);
  VkQueryPool qp2;
  CHECK(vkCreateQueryPool(c.dev, &qi, nullptr, &qp2) == VK_SUCCESS);
  CHECK(from<QueryPoolObj>(qp2)->base == p->base + 4);

  uint64_t r[8] = {};
  // Never ended: not ready, availability 0.
  CHECK(vkGetQueryPoolResults(c.dev, qp, 0, 1, sizeof r, r, 16,
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) == VK_NOT_READY);
  CHECK(r[1] == 0);

  VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  VkCommandPool pool;
  vkCreateCommandPool(c.dev, &pi, nullptr, &pool);
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = pool;
  ai.commandBufferCount = 1;
  VkCommandBuffer cb;
  vkAllocateCommandBuffers(c.dev, &ai, &cb);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(cb, &bi);
  vkCmdResetQueryPool(cb, qp, 0, 4);
  vkCmdBeginQuery(cb, qp, 1, 0);
  vkCmdEndQuery(cb, qp, 1);
  vkCmdCopyQueryPoolResults(cb, qp, 0, 2, out, 8, 16, VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
  vkEndCommandBuffer(cb);
  {
    vpgpu::Frame frame;
    TranslateStats st;
    translate(*from<CommandBuffer>(cb)->rec, frame, st);
    // fill, then query 0: availability only (not ended); query 1: counter + availability.
    CHECK(frame.commands().size() == 4 && st.queries.size() == 2);
    const auto* f = frame.commands().empty() ? nullptr : std::get_if<vpgpu::Copy>(&frame.commands()[0]);
    CHECK(f && f->kind == vpgpu::CopyKind::FillBuffer && f->dstBuffer.offset == uint64_t(p->base) * 8 && f->dstBuffer.size == 32);
    if (frame.commands().size() == 4) {
      const auto* a0 = std::get_if<vpgpu::Copy>(&frame.commands()[1]);
      CHECK(a0 && a0->dstBuffer.offset == 8 + 4 && a0->srcBuffer.size == 4);
      const auto* v1 = std::get_if<vpgpu::Copy>(&frame.commands()[2]);
      CHECK(v1 && v1->srcBuffer.offset == uint64_t(p->base + 1) * 8 && v1->dstBuffer.offset == 8 + 16);
    }
  }
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb;
  CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS);
  CHECK(p->frame[1] != 0 && p->frame[0] == 0);
  // The null platform finishes every frame at once; the counter is host memory.
  const uint64_t samples = 77;
  std::memcpy(from<Device>(c.dev)->visibility.host + size_t(p->base + 1) * 8, &samples, 8);
  uint32_t r32[2] = {};
  CHECK(vkGetQueryPoolResults(c.dev, qp, 1, 1, sizeof r32, r32, 8,
                              VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) == VK_SUCCESS);
  CHECK(r32[0] == 77 && r32[1] == 1);
  vkResetQueryPool(c.dev, qp, 0, 4);
  CHECK(vkGetQueryPoolResults(c.dev, qp, 1, 1, sizeof r32, r32, 8, 0) == VK_NOT_READY);

  // Timestamps: host time written at submission, increasing, available once the frame finished.
  {
    VkQueryPoolCreateInfo ti{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    ti.queryType = VK_QUERY_TYPE_TIMESTAMP;
    ti.queryCount = 2;
    VkQueryPool tp;
    CHECK(vkCreateQueryPool(c.dev, &ti, nullptr, &tp) == VK_SUCCESS);
    VkCommandBuffer tc;
    vkAllocateCommandBuffers(c.dev, &ai, &tc);
    vkBeginCommandBuffer(tc, &bi);
    vkCmdResetQueryPool(tc, tp, 0, 2);
    vkCmdWriteTimestamp(tc, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, tp, 0);
    vkCmdWriteTimestamp(tc, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, tp, 1);
    vkEndCommandBuffer(tc);
    VkSubmitInfo ts{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    ts.commandBufferCount = 1;
    ts.pCommandBuffers = &tc;
    CHECK(vkQueueSubmit(c.queue, 1, &ts, VK_NULL_HANDLE) == VK_SUCCESS);
    uint64_t t[4] = {};
    CHECK(vkGetQueryPoolResults(c.dev, tp, 0, 2, sizeof t, t, 16,
                                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT | VK_QUERY_RESULT_WAIT_BIT) ==
          VK_SUCCESS);
    CHECK(t[0] > 0 && t[2] > t[0] && t[1] == 1 && t[3] == 1);
    vkDestroyQueryPool(c.dev, tp, nullptr);
  }
  vkDestroyQueryPool(c.dev, qp, nullptr);
  CHECK(from<Device>(c.dev)->visibility.used.size() == 1);
  vkDestroyQueryPool(c.dev, qp2, nullptr);
  vkDestroyCommandPool(c.dev, pool, nullptr);
  vkDestroyBuffer(c.dev, out, nullptr);
  vkFreeMemory(c.dev, m, nullptr);
}

// Timeline wait-before-signal: a batch waiting on a value only the host signals is held back
// (not on the GPU, its fence not done) and runs, with the batches queued behind it, at
// vkSignalSemaphore. Images bound to the same memory and offset with the same description share
// one texture (memory aliasing); the last one destroyed releases it.
void testTimeline() {
  Ctx c;
  VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  VkCommandPool pool;
  vkCreateCommandPool(c.dev, &pi, nullptr, &pool);
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = pool;
  ai.commandBufferCount = 1;
  VkDeviceMemory m = alloc(c, 1 << 16, 1);
  VkBuffer b = buffer(c, 4096, m, 0);
  VkCommandBuffer cb[2];
  for (VkCommandBuffer& x : cb) {
    vkAllocateCommandBuffers(c.dev, &ai, &x);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(x, &bi);
    vkCmdFillBuffer(x, b, 0, 256, 7);
    vkEndCommandBuffer(x);
  }
  VkSemaphoreTypeCreateInfo ti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  sci.pNext = &ti;
  VkSemaphore sem;
  vkCreateSemaphore(c.dev, &sci, nullptr, &sem);
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  vkCreateFence(c.dev, &fci, nullptr, &fence);

  const uint64_t frames0 = c.gpu().submittedFrames();
  // Batch 1 waits on 5 (host), signals 6; batch 2 waits on 6 (reached by batch 1), signals 7.
  const uint64_t w1 = 5, s1 = 6, w2 = 6, s2 = 7;
  VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  VkTimelineSemaphoreSubmitInfo t1{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  t1.waitSemaphoreValueCount = 1, t1.pWaitSemaphoreValues = &w1, t1.signalSemaphoreValueCount = 1, t1.pSignalSemaphoreValues = &s1;
  VkSubmitInfo si1{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si1.pNext = &t1, si1.waitSemaphoreCount = 1, si1.pWaitSemaphores = &sem, si1.pWaitDstStageMask = &stage;
  si1.commandBufferCount = 1, si1.pCommandBuffers = &cb[0], si1.signalSemaphoreCount = 1, si1.pSignalSemaphores = &sem;
  CHECK(vkQueueSubmit(c.queue, 1, &si1, VK_NULL_HANDLE) == VK_SUCCESS);
  VkTimelineSemaphoreSubmitInfo t2 = t1;
  t2.pWaitSemaphoreValues = &w2, t2.pSignalSemaphoreValues = &s2;
  VkSubmitInfo si2 = si1;
  si2.pNext = &t2, si2.pCommandBuffers = &cb[1];
  CHECK(vkQueueSubmit(c.queue, 1, &si2, fence) == VK_SUCCESS);
  CHECK(c.gpu().submittedFrames() == frames0);  // both held back
  CHECK(from<Device>(c.dev)->deferred.size() == 2);
  CHECK(vkGetFenceStatus(c.dev, fence) == VK_NOT_READY);
  CHECK(vkWaitForFences(c.dev, 1, &fence, VK_TRUE, 1000000) == VK_TIMEOUT);
  uint64_t v = 0;
  vkGetSemaphoreCounterValue(c.dev, sem, &v);
  CHECK(v == 0);
  VkSemaphoreSignalInfo sig{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
  sig.semaphore = sem, sig.value = 5;
  CHECK(vkSignalSemaphore(c.dev, &sig) == VK_SUCCESS);
  CHECK(from<Device>(c.dev)->deferred.empty());
  CHECK(c.gpu().submittedFrames() == frames0 + 1);  // one frame for both batches
  CHECK(vkWaitForFences(c.dev, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS);
  vkGetSemaphoreCounterValue(c.dev, sem, &v);
  CHECK(v == 7);
  // A wait already met runs at once.
  VkTimelineSemaphoreSubmitInfo t3 = t1;
  const uint64_t w3 = 7, s3 = 8;
  t3.pWaitSemaphoreValues = &w3, t3.pSignalSemaphoreValues = &s3;
  VkSubmitInfo si3 = si1;
  si3.pNext = &t3;
  CHECK(vkQueueSubmit(c.queue, 1, &si3, VK_NULL_HANDLE) == VK_SUCCESS);
  CHECK(from<Device>(c.dev)->deferred.empty() && c.gpu().submittedFrames() == frames0 + 2);
  CHECK(vkDeviceWaitIdle(c.dev) == VK_SUCCESS);

  // Aliased images share one texture.
  VkDeviceMemory im = alloc(c, 1 << 20, 0);
  VkImage a = image(c, VK_FORMAT_R8G8B8A8_UINT, 33, 1, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, im);
  VkImage a2 = image(c, VK_FORMAT_R8G8B8A8_UINT, 33, 1, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, im);
  VkImage other = image(c, VK_FORMAT_R8G8B8A8_UNORM, 33, 1, VK_IMAGE_USAGE_TRANSFER_SRC_BIT, im);
  CHECK(from<Image>(a)->id && from<Image>(a)->id == from<Image>(a2)->id);
  CHECK(from<Image>(other)->id && from<Image>(other)->id != from<Image>(a)->id);
  const vpgpu::TextureId shared = from<Image>(a)->id;
  vkDestroyImage(c.dev, a, nullptr);
  CHECK(c.gpu().resources().get(shared) != nullptr);  // a2 still uses it
  vkDestroyImage(c.dev, a2, nullptr);
  CHECK(c.gpu().resources().get(shared) == nullptr);  // the last one released it
  vkDestroyImage(c.dev, other, nullptr);
  vkFreeMemory(c.dev, im, nullptr);

  vkDestroyFence(c.dev, fence, nullptr);
  vkDestroySemaphore(c.dev, sem, nullptr);
  vkDestroyCommandPool(c.dev, pool, nullptr);
  vkDestroyBuffer(c.dev, b, nullptr);
  vkFreeMemory(c.dev, m, nullptr);
}

// Render-pass load/store rules Metal's whole-attachment actions cannot express directly (CTS
// renderpasses.attachment.*, multiview.*clear*): partial render areas keep the outside, multiview
// load ops apply per view at its first subpass, one store for both depth/stencil aspects, and a
// subpass reading its own attachment is split at its self-dependency barrier.
void testRenderPassOps() {
  Ctx c({VK_KHR_MULTIVIEW_EXTENSION_NAME});
  VkDeviceMemory m = alloc(c, 64 << 20, 0);
  VkImage color = image(c, VK_FORMAT_R8G8B8A8_UNORM, 64, 4,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, m);
  VkImage ds = image(c, VK_FORMAT_D32_SFLOAT_S8_UINT, 64, 1, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, m);
  auto makeView = [&](VkImage im, VkImageAspectFlags aspect, uint32_t layers) {
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = im;
    vi.viewType = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    vi.subresourceRange = {aspect, 0, 1, 0, layers};
    VkImageView v = VK_NULL_HANDLE;
    vkCreateImageView(c.dev, &vi, nullptr, &v);
    return v;
  };
  VkImageView cv1 = makeView(color, VK_IMAGE_ASPECT_COLOR_BIT, 1), cv4 = makeView(color, VK_IMAGE_ASPECT_COLOR_BIT, 4);
  VkImageView dv = makeView(ds, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 1);
  VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  VkCommandPool pool;
  vkCreateCommandPool(c.dev, &pi, nullptr, &pool);
  struct Pass {
    VkRenderPass rp = VK_NULL_HANDLE;
    VkFramebuffer fb = VK_NULL_HANDLE;
  };
  std::vector<Pass> made;
  auto pass = [&](std::vector<VkAttachmentDescription> att, std::vector<VkSubpassDescription> subs,
                  std::vector<uint32_t> masks, std::vector<VkImageView> views) {
    VkRenderPassMultiviewCreateInfo mv{VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO};
    mv.subpassCount = uint32_t(masks.size());
    mv.pViewMasks = masks.data();
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, masks.empty() ? nullptr : &mv};
    rpi.attachmentCount = uint32_t(att.size());
    rpi.pAttachments = att.data();
    rpi.subpassCount = uint32_t(subs.size());
    rpi.pSubpasses = subs.data();
    Pass p;
    vkCreateRenderPass(c.dev, &rpi, nullptr, &p.rp);
    VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fi.renderPass = p.rp;
    fi.attachmentCount = uint32_t(views.size());
    fi.pAttachments = views.data();
    fi.width = fi.height = 64;
    fi.layers = 1;
    vkCreateFramebuffer(c.dev, &fi, nullptr, &p.fb);
    made.push_back(p);
    return p;
  };
  // Records begin / (next subpass | barrier)... / end and translates it.
  auto run = [&](const Pass& p, VkRect2D area, const std::function<void(VkCommandBuffer)>& body, vpgpu::Frame& frame,
                 TranslateStats& st) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    vkAllocateCommandBuffers(c.dev, &ai, &cb);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cb, &bi);
    VkClearValue clears[2]{};
    clears[0].color.float32[0] = 1.0f;
    clears[1].depthStencil = {1.0f, 5};
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = p.rp;
    rbi.framebuffer = p.fb;
    rbi.renderArea = area;
    rbi.clearValueCount = 2;
    rbi.pClearValues = clears;
    vkCmdBeginRenderPass(cb, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    if (body) body(cb);
    vkCmdEndRenderPass(cb);
    vkEndCommandBuffer(cb);
    translate(*from<CommandBuffer>(cb)->rec, frame, st);
  };
  auto passes = [](const vpgpu::Frame& f) {
    std::vector<const vpgpu::RenderPass*> out;
    for (const auto& cmd : f.commands())
      if (const auto* p = std::get_if<vpgpu::RenderPass>(&cmd)) out.push_back(p);
    return out;
  };
  VkAttachmentDescription col{};
  col.format = VK_FORMAT_R8G8B8A8_UNORM;
  col.samples = VK_SAMPLE_COUNT_1_BIT;
  col.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  col.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  VkAttachmentReference cref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkSubpassDescription one{};
  one.colorAttachmentCount = 1;
  one.pColorAttachments = &cref;

  {  // DONT_CARE load and store with a render area smaller than the image: load and store.
    VkAttachmentDescription a = col;
    a.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, a.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    Pass p = pass({a}, {one}, {}, {cv1});
    vpgpu::Frame f;
    TranslateStats st;
    run(p, {{3, 17}, {32, 32}}, nullptr, f, st);
    auto ps = passes(f);
    CHECK(ps.size() == 1 && ps[0]->desc.colors[0].load == vpgpu::LoadAction::Load &&
          ps[0]->desc.colors[0].store == vpgpu::StoreAction::Store);
    vpgpu::Frame g;
    run(p, {{0, 0}, {64, 64}}, nullptr, g, st);  // the whole image: DontCare stays
    ps = passes(g);
    CHECK(ps.size() == 1 && ps[0]->desc.colors[0].load == vpgpu::LoadAction::DontCare &&
          ps[0]->desc.colors[0].store == vpgpu::StoreAction::DontCare);
  }
  {  // D32S8: depth DONT_CARE store, stencil STORE -> the image is stored.
    VkAttachmentDescription a{};
    a.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
    a.samples = VK_SAMPLE_COUNT_1_BIT;
    a.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, a.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    a.initialLayout = a.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference dref{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sd{};
    sd.pDepthStencilAttachment = &dref;
    Pass p = pass({a}, {sd}, {}, {dv});
    vpgpu::Frame f;
    TranslateStats st;
    run(p, {{0, 0}, {64, 64}}, nullptr, f, st);
    auto ps = passes(f);
    CHECK(ps.size() == 1 && ps[0]->desc.depth.load == vpgpu::LoadAction::Clear &&
          ps[0]->desc.depth.store == vpgpu::StoreAction::Store);
  }
  {  // Multiview 0b1000 then 0b0001: clears only the subpass's own new view, never the others.
    VkAttachmentDescription a = col;
    a.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    Pass p = pass({a}, {one, one}, {0b1000, 0b0001}, {cv4});
    vpgpu::Frame f;
    TranslateStats st;
    run(p, {{0, 0}, {64, 64}}, [](VkCommandBuffer cb) { vkCmdNextSubpass(cb, VK_SUBPASS_CONTENTS_INLINE); }, f, st);
    auto ps = passes(f);
    CHECK(ps.size() == 2);
    for (size_t i = 0; i < ps.size() && i < 2; ++i) {
      const vpgpu::RenderPass& rp = *ps[i];
      CHECK(rp.desc.colors[0].load == vpgpu::LoadAction::Load && rp.desc.colors[0].store == vpgpu::StoreAction::Store);
      CHECK(rp.draws.size() == 1 && rp.draws[0].clear && rp.draws[0].clear->rects.size() == 1);
      if (rp.draws.size() == 1 && rp.draws[0].clear && !rp.draws[0].clear->rects.empty()) {
        const auto& r = rp.draws[0].clear->rects[0];
        CHECK(r.layer == (i == 0 ? 3 : 0) && r.layerCount == 1);
      }
    }
    // vkCmdClearAttachments in a 0b101 subpass: the mask's layers only (0 and 2, not 1).
    VkAttachmentDescription b = col;
    b.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, b.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    Pass q = pass({b}, {one}, {0b101}, {cv4});
    vpgpu::Frame g;
    run(q, {{0, 0}, {64, 64}},
        [](VkCommandBuffer cb) {
          VkClearAttachment ca{VK_IMAGE_ASPECT_COLOR_BIT, 0, {}};
          VkClearRect cr{{{0, 0}, {64, 64}}, 0, 1};
          vkCmdClearAttachments(cb, 1, &ca, 1, &cr);
        },
        g, st);
    ps = passes(g);
    CHECK(ps.size() == 1 && ps[0]->draws.size() == 1 && ps[0]->draws[0].clear &&
          ps[0]->draws[0].clear->rects.size() == 2);
    if (ps.size() == 1 && ps[0]->draws.size() == 1 && ps[0]->draws[0].clear && ps[0]->draws[0].clear->rects.size() == 2)
      CHECK(ps[0]->draws[0].clear->rects[0].layer == 0 && ps[0]->draws[0].clear->rects[1].layer == 2);
  }
  {  // Subpass reading its own colour attachment: split at the barrier, the second part loads.
    VkAttachmentDescription a = col;
    a.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, a.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    VkAttachmentReference in{0, VK_IMAGE_LAYOUT_GENERAL}, out{0, VK_IMAGE_LAYOUT_GENERAL};
    VkSubpassDescription self{};
    self.inputAttachmentCount = 1;
    self.pInputAttachments = &in;
    self.colorAttachmentCount = 1;
    self.pColorAttachments = &out;
    Pass p = pass({a}, {self}, {}, {cv1});
    vpgpu::Frame f;
    TranslateStats st;
    run(p, {{0, 0}, {64, 64}},
        [](VkCommandBuffer cb) { vkCmdPipelineBarrier(cb, 0, 0, 0, 0, nullptr, 0, nullptr, 0, nullptr); }, f, st);
    auto ps = passes(f);
    CHECK(st.selfDependencySplits == 1 && ps.size() == 2);
    if (ps.size() == 2) {
      CHECK(ps[0]->desc.colors[0].load == vpgpu::LoadAction::Clear && ps[0]->desc.colors[0].store == vpgpu::StoreAction::Store);
      CHECK(ps[1]->desc.colors[0].load == vpgpu::LoadAction::Load);
    }
  }
  {  // Integer MSAA colour with a resolve: samples stored, sample 0 blitted into the target after.
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = VK_FORMAT_R8G8B8A8_UINT;
    ii.extent = {64, 64, 1};
    ii.mipLevels = ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_4_BIT;
    ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    VkImage ms = VK_NULL_HANDLE, single = VK_NULL_HANDLE;
    vkCreateImage(c.dev, &ii, nullptr, &ms);
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    vkCreateImage(c.dev, &ii, nullptr, &single);
    vkBindImageMemory(c.dev, ms, m, 0);
    vkBindImageMemory(c.dev, single, m, 0);
    if (from<Image>(ms)->desc.samples > 1) {  // (not when the configuration forces MSAA off)
      VkImageView mv = makeView(ms, VK_IMAGE_ASPECT_COLOR_BIT, 1), ov = makeView(single, VK_IMAGE_ASPECT_COLOR_BIT, 1);
      VkAttachmentDescription a = col, r = col;
      a.format = r.format = VK_FORMAT_R8G8B8A8_UINT;
      a.samples = VK_SAMPLE_COUNT_4_BIT;
      a.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, a.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      r.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, r.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      VkAttachmentReference rref{1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
      VkSubpassDescription sd = one;
      sd.pResolveAttachments = &rref;
      Pass p = pass({a, r}, {sd}, {}, {mv, ov});
      vpgpu::Frame f;
      TranslateStats st;
      run(p, {{0, 0}, {64, 64}}, nullptr, f, st);
      auto ps = passes(f);
      CHECK(ps.size() == 1 && ps[0]->desc.colors[0].store == vpgpu::StoreAction::Store && !ps[0]->desc.colors[0].resolve);
      bool blit = false;
      for (const auto& cmd : f.commands())
        if (const auto* cp = std::get_if<vpgpu::Copy>(&cmd))
          blit = blit || (cp->kind == vpgpu::CopyKind::TextureBlit && cp->dstTexture.texture == from<Image>(single)->id &&
                          cp->blitFilter == vpgpu::Filter::Nearest);
      CHECK(blit);
      vkDestroyImageView(c.dev, mv, nullptr);
      vkDestroyImageView(c.dev, ov, nullptr);
    }
    vkDestroyImage(c.dev, ms, nullptr);
    vkDestroyImage(c.dev, single, nullptr);
  }
  {  // MSAA colour with a resolve: a client STORE keeps its samples (keepSamples), DONT_CARE stays a memoryless Resolve.
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent = {64, 64, 1};
    ii.mipLevels = ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_4_BIT;
    ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    VkImage ms = VK_NULL_HANDLE, single = VK_NULL_HANDLE;
    vkCreateImage(c.dev, &ii, nullptr, &ms);
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    vkCreateImage(c.dev, &ii, nullptr, &single);
    vkBindImageMemory(c.dev, ms, m, 0);
    vkBindImageMemory(c.dev, single, m, 0);
    if (from<Image>(ms)->desc.samples > 1 && from<Image>(ms)->forcedFrom <= 1) {  // (not when MSAA is forced off)
      VkImageView mv = makeView(ms, VK_IMAGE_ASPECT_COLOR_BIT, 1), ov = makeView(single, VK_IMAGE_ASPECT_COLOR_BIT, 1);
      VkAttachmentDescription a = col, r = col;
      a.samples = VK_SAMPLE_COUNT_4_BIT;
      a.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      r.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, r.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      VkAttachmentReference rref{1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
      VkSubpassDescription sd = one;
      sd.pResolveAttachments = &rref;
      for (const VkAttachmentStoreOp op : {VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_STORE_OP_DONT_CARE}) {
        a.storeOp = op;
        Pass p = pass({a, r}, {sd}, {}, {mv, ov});
        vpgpu::Frame f;
        TranslateStats st;
        run(p, {{0, 0}, {64, 64}}, nullptr, f, st);
        auto ps = passes(f);
        const bool store = op == VK_ATTACHMENT_STORE_OP_STORE;
        CHECK(ps.size() == 1 && ps[0]->desc.colors[0].resolve);
        if (ps.size() == 1)
          CHECK(ps[0]->desc.colors[0].store == (store ? vpgpu::StoreAction::StoreAndResolve : vpgpu::StoreAction::Resolve) &&
                ps[0]->desc.colors[0].keepSamples == store);
      }
      vkDestroyImageView(c.dev, mv, nullptr);
      vkDestroyImageView(c.dev, ov, nullptr);
    }
    vkDestroyImage(c.dev, ms, nullptr);
    vkDestroyImage(c.dev, single, nullptr);
  }
  vkDestroyCommandPool(c.dev, pool, nullptr);
  for (const Pass& p : made) {
    vkDestroyFramebuffer(c.dev, p.fb, nullptr);
    vkDestroyRenderPass(c.dev, p.rp, nullptr);
  }
  for (VkImageView v : {cv1, cv4, dv}) vkDestroyImageView(c.dev, v, nullptr);
  vkDestroyImage(c.dev, color, nullptr);
  vkDestroyImage(c.dev, ds, nullptr);
  vkFreeMemory(c.dev, m, nullptr);
}

// Linear images: memory holds the texels (copies to/from the host-aliasing buffer around transfers)
// and multi-subpass multiview: views a later subpass adds get the attachment's load-op clear.
void testLinear() {
  Ctx c({VK_KHR_MULTIVIEW_EXTENSION_NAME});
  VkDeviceMemory m = alloc(c, 4 << 20, 1);
  VkImage opt = image(c, VK_FORMAT_R8G8B8A8_UNORM, 64, 1,
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, m);
  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = VK_IMAGE_TYPE_2D;
  ii.format = VK_FORMAT_R8G8B8A8_UNORM;
  ii.extent = {64, 64, 1};
  ii.mipLevels = 1;
  ii.arrayLayers = 1;
  ii.samples = VK_SAMPLE_COUNT_1_BIT;
  ii.tiling = VK_IMAGE_TILING_LINEAR;
  ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  VkImage lin = VK_NULL_HANDLE;
  CHECK(vkCreateImage(c.dev, &ii, nullptr, &lin) == VK_SUCCESS);
  CHECK(vkBindImageMemory(c.dev, lin, m, 1 << 20) == VK_SUCCESS);
  CHECK(from<Image>(lin)->linear && from<Image>(lin)->hostAlias);
  CHECK(!from<Image>(opt)->linear && !from<Image>(opt)->hostAlias);
  VkImageSubresource sub{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
  VkSubresourceLayout lay{};
  vkGetImageSubresourceLayout(c.dev, lin, &sub, &lay);
  CHECK(lay.offset == 0 && lay.rowPitch == 256 && lay.size == 256 * 64);

  VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  VkCommandPool pool;
  vkCreateCommandPool(c.dev, &pi, nullptr, &pool);
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = pool;
  ai.commandBufferCount = 1;
  VkCommandBuffer cb;
  vkAllocateCommandBuffers(c.dev, &ai, &cb);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(cb, &bi);
  VkImageCopy r{};
  r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  r.extent = {64, 64, 1};
  vkCmdCopyImage(cb, opt, VK_IMAGE_LAYOUT_GENERAL, lin, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
  vkEndCommandBuffer(cb);
  {
    vpgpu::Frame frame;
    TranslateStats st;
    translate(*from<CommandBuffer>(cb)->rec, frame, st);
    // memory -> linear texture, the copy, linear texture -> memory
    CHECK(frame.commands().size() == 3);
    const auto* in = frame.commands().size() == 3 ? std::get_if<vpgpu::Copy>(&frame.commands()[0]) : nullptr;
    const auto* out = frame.commands().size() == 3 ? std::get_if<vpgpu::Copy>(&frame.commands()[2]) : nullptr;
    CHECK(in && in->kind == vpgpu::CopyKind::BufferToTexture && in->srcBuffer.buffer == from<Image>(lin)->hostAlias &&
          in->srcBuffer.bytesPerRow == 256 && in->dstTexture.texture == from<Image>(lin)->id);
    CHECK(out && out->kind == vpgpu::CopyKind::TextureToBuffer && out->dstBuffer.buffer == from<Image>(lin)->hostAlias &&
          out->dstBuffer.size == 256 * 64 && out->srcTexture.width == 64);
  }

  // Two subpasses, view masks 0b01 and 0b10, both clearing on load: subpass 1 clears layer 1.
  VkImage color = image(c, VK_FORMAT_R8G8B8A8_UNORM, 32, 2, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, m);
  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.image = color;
  vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  vi.format = VK_FORMAT_R8G8B8A8_UNORM;
  vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 2};
  VkImageView view;
  vkCreateImageView(c.dev, &vi, nullptr, &view);
  VkAttachmentDescription att{};
  att.format = VK_FORMAT_R8G8B8A8_UNORM;
  att.samples = VK_SAMPLE_COUNT_1_BIT;
  att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  VkAttachmentReference cref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkSubpassDescription sps[2]{};
  for (auto& s : sps) s.colorAttachmentCount = 1, s.pColorAttachments = &cref;
  const uint32_t masks[2] = {0b01, 0b10};
  VkRenderPassMultiviewCreateInfo mv{VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO};
  mv.subpassCount = 2;
  mv.pViewMasks = masks;
  VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, &mv};
  rpi.attachmentCount = 1;
  rpi.pAttachments = &att;
  rpi.subpassCount = 2;
  rpi.pSubpasses = sps;
  VkRenderPass rp;
  vkCreateRenderPass(c.dev, &rpi, nullptr, &rp);
  VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  fi.renderPass = rp;
  fi.attachmentCount = 1;
  fi.pAttachments = &view;
  fi.width = fi.height = 32;
  fi.layers = 1;
  VkFramebuffer fb;
  vkCreateFramebuffer(c.dev, &fi, nullptr, &fb);
  VkCommandBuffer cb2;
  vkAllocateCommandBuffers(c.dev, &ai, &cb2);
  vkBeginCommandBuffer(cb2, &bi);
  VkClearValue cv{};
  cv.color.float32[1] = 1.0f;
  VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  rbi.renderPass = rp;
  rbi.framebuffer = fb;
  rbi.renderArea = {{0, 0}, {32, 32}};
  rbi.clearValueCount = 1;
  rbi.pClearValues = &cv;
  vkCmdBeginRenderPass(cb2, &rbi, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdNextSubpass(cb2, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdEndRenderPass(cb2);
  vkEndCommandBuffer(cb2);
  {
    vpgpu::Frame frame;
    TranslateStats st;
    translate(*from<CommandBuffer>(cb2)->rec, frame, st);
    CHECK(st.passes == 2 && frame.commands().size() == 2);
    const auto* p0 = std::get_if<vpgpu::RenderPass>(&frame.commands()[0]);
    const auto* p1 = frame.commands().size() > 1 ? std::get_if<vpgpu::RenderPass>(&frame.commands()[1]) : nullptr;
    CHECK(p0 && p0->desc.layerCount == 1 && p0->desc.colors[0].load == vpgpu::LoadAction::Clear && p0->draws.empty());
    CHECK(p1 && p1->desc.layerCount == 2 && p1->desc.colors[0].load == vpgpu::LoadAction::Load);
    CHECK(p1 && p1->draws.size() == 1 && p1->draws[0].clear && p1->draws[0].clear->colorMask == 1 &&
          p1->draws[0].clear->rects.size() == 1 && p1->draws[0].clear->rects[0].layer == 1 &&
          p1->draws[0].clear->rects[0].layerCount == 1 && p1->draws[0].clear->colorBits[0][1] == 0x3f800000u);
  }
  vkDestroyCommandPool(c.dev, pool, nullptr);
  vkDestroyFramebuffer(c.dev, fb, nullptr);
  vkDestroyRenderPass(c.dev, rp, nullptr);
  vkDestroyImageView(c.dev, view, nullptr);
  vkDestroyImage(c.dev, color, nullptr);
  vkDestroyImage(c.dev, lin, nullptr);
  vkDestroyImage(c.dev, opt, nullptr);
  vkFreeMemory(c.dev, m, nullptr);
}

// Image work the Metal backend needs spelled out: 3D clears per depth plane, integer MSAA
// resolves as a sample-0 blit, 3D linear images synced with their memory, render passes into a
// view of another format (mutable-format image), compressed texel views refused.
// Image/texture lifetime counters of the [vkagg] line (img_rel_*): guest images created and destroyed over and over, some
// aliasing one texture (same memory, offset and description), must release exactly one texture per distinct texture.
uint64_t diagValue(const char* line, const char* key) {
  const std::string k = std::string(" ") + key + "=";
  const char* at = std::strstr(line, k.c_str());
  return at ? std::strtoull(at + k.size(), nullptr, 10) : 0;
}

void testImageLifetime() {
  Ctx c;
  VkDeviceMemory m = alloc(c, 16 << 20, 1);
  char line[4096];
  vpvk_diag_line(0, line, sizeof line);  // baseline: later lines carry only what moved
  constexpr int kCycles = 12;
  const VkImageUsageFlags use = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  for (int n = 0; n < kCycles; ++n) {
    VkImage a = image(c, VK_FORMAT_R8G8B8A8_UNORM, 64, 1, use, m);
    VkImage b = image(c, VK_FORMAT_R8G8B8A8_UNORM, 64, 1, use, m);  // aliases a
    VkImage u = image(c, VK_FORMAT_R8G8B8A8_UNORM, 64, 1, use, 0);  // never bound
    CHECK(a && b && u && from<Image>(a)->id && from<Image>(a)->id == from<Image>(b)->id && !from<Image>(u)->id);
    vkDestroyImage(c.dev, a, nullptr);  // the texture survives: b still uses it
    vkDestroyImage(c.dev, b, nullptr);
    vkDestroyImage(c.dev, u, nullptr);
  }
  CHECK(from<DeviceMemory>(m)->images.empty());
  CHECK(vpvk_diag_line(0, line, sizeof line) > 0);
  CHECK(diagValue(line, "img_destroy") == 3 * kCycles);
  CHECK(diagValue(line, "img_rel_destroy") == kCycles);
  CHECK(diagValue(line, "img_rel_shared") == kCycles);
  CHECK(diagValue(line, "img_rel_notex") == kCycles);
  vkFreeMemory(c.dev, m, nullptr);
}

void testImages() {
  Ctx c;
  VkDeviceMemory m = alloc(c, 16 << 20, 1);
  VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  VkCommandPool pool;
  vkCreateCommandPool(c.dev, &pi, nullptr, &pool);
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = pool;
  ai.commandBufferCount = 1;
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  auto make = [&](VkImageType type, VkFormat f, VkExtent3D e, VkSampleCountFlagBits n, VkImageTiling tiling,
                  VkImageUsageFlags usage, VkImageCreateFlags flags = 0) {
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = type, ii.format = f, ii.extent = e, ii.mipLevels = 1, ii.arrayLayers = 1, ii.samples = n;
    ii.tiling = tiling, ii.usage = usage, ii.flags = flags;
    VkImage i = VK_NULL_HANDLE;
    vkCreateImage(c.dev, &ii, nullptr, &i);
    if (i) vkBindImageMemory(c.dev, i, m, 0);
    return i;
  };
  // 3D clear: one clear pass per depth slice, between the linear image's memory -> texture and
  // texture -> memory copies (linearSync); the copies of the whole volume order against them.
  VkImage vol = make(VK_IMAGE_TYPE_3D, VK_FORMAT_R8G8B8A8_UNORM, {16, 16, 4}, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_LINEAR,
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
  CHECK(vol && from<Image>(vol)->linear && from<Image>(vol)->hostAlias);
  VkSubresourceLayout l{};
  VkImageSubresource sub{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
  vkGetImageSubresourceLayout(c.dev, vol, &sub, &l);
  CHECK(l.rowPitch == 64 && l.depthPitch == 64 * 16 && l.size == 64 * 16 * 4);
  VkCommandBuffer cb;
  vkAllocateCommandBuffers(c.dev, &ai, &cb);
  vkBeginCommandBuffer(cb, &bi);
  VkClearColorValue col{};
  VkImageSubresourceRange all{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdClearColorImage(cb, vol, VK_IMAGE_LAYOUT_GENERAL, &col, 1, &all);
  vkEndCommandBuffer(cb);
  {
    vpgpu::Frame f;
    TranslateStats st;
    translate(*from<CommandBuffer>(cb)->rec, f, st);
    CHECK(st.clears == 4);
    // memory -> volume, 4 clears (planes 0..3), volume -> memory.
    CHECK(f.commands().size() == 6);
    if (f.commands().size() == 6) {
      const auto* in = std::get_if<vpgpu::Copy>(&f.commands()[0]);
      CHECK(in && in->kind == vpgpu::CopyKind::BufferToTexture && in->dstTexture.depth == 4 && in->dstTexture.layerCount == 1);
      const auto* p3 = std::get_if<vpgpu::RenderPass>(&f.commands()[4]);
      CHECK(p3 && p3->desc.colors[0].layer == 3);
      const auto* mc = std::get_if<vpgpu::Copy>(&f.commands()[5]);
      CHECK(mc && mc->kind == vpgpu::CopyKind::TextureToBuffer && mc->srcTexture.depth == 4 &&
            mc->dstBuffer.bytesPerRow == 64 && mc->dstBuffer.rowsPerImage == 16 && mc->dstBuffer.size == 64 * 16 * 4 &&
            mc->dstBuffer.buffer == from<Image>(vol)->hostAlias);
      if (mc) {
        const auto acc = vpgpu::copyAccesses(*mc);
        CHECK(!acc.empty() && acc[0].range.layerBegin == 0 && acc[0].range.layerEnd == 4);
      }
    }
  }
  // Integer MSAA resolve (vkCmdResolveImage): a sampled blit of sample 0.
  VkImage msi = make(VK_IMAGE_TYPE_2D, VK_FORMAT_R16G16_UINT, {32, 32, 1}, VK_SAMPLE_COUNT_4_BIT, VK_IMAGE_TILING_OPTIMAL,
                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
  VkImage one = make(VK_IMAGE_TYPE_2D, VK_FORMAT_R16G16_UINT, {32, 32, 1}, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
  CHECK(msi && (from<Image>(msi)->desc.usage & vpgpu::usage::sampled));
  VkCommandBuffer cb2;
  vkAllocateCommandBuffers(c.dev, &ai, &cb2);
  vkBeginCommandBuffer(cb2, &bi);
  VkImageResolve rr{};
  rr.srcSubresource = rr.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  rr.extent = {32, 32, 1};
  vkCmdResolveImage(cb2, msi, VK_IMAGE_LAYOUT_GENERAL, one, VK_IMAGE_LAYOUT_GENERAL, 1, &rr);
  vkEndCommandBuffer(cb2);
  {
    vpgpu::Frame f;
    TranslateStats st;
    translate(*from<CommandBuffer>(cb2)->rec, f, st);
    CHECK(f.commands().size() == 1 && st.dropped == 0);
    const auto* bl = f.commands().empty() ? nullptr : std::get_if<vpgpu::Copy>(&f.commands()[0]);
    CHECK(bl && bl->kind == vpgpu::CopyKind::TextureBlit && bl->blitFilter == vpgpu::Filter::Nearest &&
          bl->dstTexture.width == 32);
  }
  // Mutable-format image rendered through an R32_UINT view: the attachment carries the format.
  VkImage mut = make(VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {32, 32, 1}, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT);
  CHECK(mut && (from<Image>(mut)->desc.usage & vpgpu::usage::formatView));
  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.image = mut, vi.viewType = VK_IMAGE_VIEW_TYPE_2D, vi.format = VK_FORMAT_R32_UINT;
  vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkImageView mv;
  vkCreateImageView(c.dev, &vi, nullptr, &mv);
  VkAttachmentDescription ad{};
  ad.format = VK_FORMAT_R32_UINT, ad.samples = VK_SAMPLE_COUNT_1_BIT;
  ad.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, ad.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  VkAttachmentReference ar{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkSubpassDescription sd{};
  sd.colorAttachmentCount = 1, sd.pColorAttachments = &ar;
  VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  rpi.attachmentCount = 1, rpi.pAttachments = &ad, rpi.subpassCount = 1, rpi.pSubpasses = &sd;
  VkRenderPass rp;
  vkCreateRenderPass(c.dev, &rpi, nullptr, &rp);
  VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  fi.renderPass = rp, fi.attachmentCount = 1, fi.pAttachments = &mv, fi.width = fi.height = 32, fi.layers = 1;
  VkFramebuffer fb;
  vkCreateFramebuffer(c.dev, &fi, nullptr, &fb);
  VkCommandBuffer cb3;
  vkAllocateCommandBuffers(c.dev, &ai, &cb3);
  vkBeginCommandBuffer(cb3, &bi);
  VkClearValue cv{};
  cv.color.uint32[0] = 7;
  VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  rbi.renderPass = rp, rbi.framebuffer = fb, rbi.renderArea = {{0, 0}, {32, 32}}, rbi.clearValueCount = 1, rbi.pClearValues = &cv;
  vkCmdBeginRenderPass(cb3, &rbi, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdEndRenderPass(cb3);
  vkEndCommandBuffer(cb3);
  {
    vpgpu::Frame f;
    TranslateStats st;
    translate(*from<CommandBuffer>(cb3)->rec, f, st);
    const auto* p = f.commands().empty() ? nullptr : std::get_if<vpgpu::RenderPass>(&f.commands()[0]);
    CHECK(p && p->desc.colors[0].format == vpgpu::Format::R32Uint && p->desc.colors[0].clear.color[0] == 7.0);
  }
  // BLOCK_TEXEL_VIEW_COMPATIBLE of a compressed format: refused (Metal has no such views).
  VkImageFormatProperties ifp;
  CHECK(vkGetPhysicalDeviceImageFormatProperties(c.pd, VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, VK_IMAGE_TYPE_2D,
                                                 VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT,
                                                 VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT |
                                                     VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
                                                 &ifp) == VK_ERROR_FORMAT_NOT_SUPPORTED);
  // Texel buffer formats: what the translator unpacks; no stores of packed floats / BGRA.
  VkFormatProperties fp;
  vkGetPhysicalDeviceFormatProperties(c.pd, VK_FORMAT_B10G11R11_UFLOAT_PACK32, &fp);
  CHECK((fp.bufferFeatures & VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT) && !(fp.bufferFeatures & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT));
  vkGetPhysicalDeviceFormatProperties(c.pd, VK_FORMAT_R8G8B8A8_SRGB, &fp);
  CHECK(!(fp.bufferFeatures & VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT));
  vkDestroyFramebuffer(c.dev, fb, nullptr);
  vkDestroyRenderPass(c.dev, rp, nullptr);
  vkDestroyImageView(c.dev, mv, nullptr);
  vkDestroyCommandPool(c.dev, pool, nullptr);
  for (VkImage i : {vol, msi, one, mut}) vkDestroyImage(c.dev, i, nullptr);
  vkFreeMemory(c.dev, m, nullptr);
}

// ---------------------------------------------------------------------------------------------
// Fixture for tests that record draws: a 64x64 colour + D32S8 pass, our trivial position/solid
// shaders (shader/spirv/tests/*.spvasm), and a command buffer inside the pass.
// ---------------------------------------------------------------------------------------------
struct DrawRig {
  Ctx& c;
  VkDeviceMemory m = VK_NULL_HANDLE;
  VkImage color = VK_NULL_HANDLE, depth = VK_NULL_HANDLE;
  VkImageView views[2]{};
  VkRenderPass rp = VK_NULL_HANDLE;
  VkFramebuffer fb = VK_NULL_HANDLE;
  VkCommandPool cp = VK_NULL_HANDLE;
  VkCommandBuffer cb = VK_NULL_HANDLE;
  explicit DrawRig(Ctx& ctx) : c(ctx) {
    m = alloc(c, 8 << 20, 0);
    color = image(c, VK_FORMAT_R8G8B8A8_UNORM, 64, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, m);
    depth = image(c, VK_FORMAT_D32_SFLOAT_S8_UINT, 64, 1, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, m);
    const VkFormat fm[2] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT_S8_UINT};
    const VkImage im[2] = {color, depth};
    for (int i = 0; i < 2; ++i) {
      VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      vi.image = im[i], vi.viewType = VK_IMAGE_VIEW_TYPE_2D, vi.format = fm[i];
      vi.subresourceRange = {VkImageAspectFlags(i ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
                                                  : VK_IMAGE_ASPECT_COLOR_BIT), 0, 1, 0, 1};
      vkCreateImageView(c.dev, &vi, nullptr, &views[i]);
    }
    VkAttachmentDescription att[2]{};
    for (int i = 0; i < 2; ++i) {
      att[i].format = fm[i], att[i].samples = VK_SAMPLE_COUNT_1_BIT;
      att[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, att[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      att[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, att[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    }
    VkAttachmentReference cref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference dref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sp{};
    sp.colorAttachmentCount = 1, sp.pColorAttachments = &cref, sp.pDepthStencilAttachment = &dref;
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = 2, rpi.pAttachments = att, rpi.subpassCount = 1, rpi.pSubpasses = &sp;
    vkCreateRenderPass(c.dev, &rpi, nullptr, &rp);
    VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fi.renderPass = rp, fi.attachmentCount = 2, fi.pAttachments = views, fi.width = 64, fi.height = 64, fi.layers = 1;
    vkCreateFramebuffer(c.dev, &fi, nullptr, &fb);
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    vkCreateCommandPool(c.dev, &cpi, nullptr, &cp);
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = cp, ai.commandBufferCount = 1;
    vkAllocateCommandBuffers(c.dev, &ai, &cb);
  }
  void begin() {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cb, &bi);
    VkClearValue clears[2]{};
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = rp, rbi.framebuffer = fb, rbi.renderArea = {{0, 0}, {64, 64}}, rbi.clearValueCount = 2;
    rbi.pClearValues = clears;
    vkCmdBeginRenderPass(cb, &rbi, VK_SUBPASS_CONTENTS_INLINE);
  }
  // Ends the recording and translates it: the draws of the one pass.
  std::vector<vpgpu::Draw> finish(vpgpu::Frame& frame, TranslateStats& ts) {
    vkCmdEndRenderPass(cb);
    vkEndCommandBuffer(cb);
    translate(*from<CommandBuffer>(cb)->rec, frame, ts);
    const auto* pass = frame.commands().empty() ? nullptr : std::get_if<vpgpu::RenderPass>(&frame.commands()[0]);
    return pass ? pass->draws : std::vector<vpgpu::Draw>{};
  }
  ~DrawRig() {
    vkDestroyCommandPool(c.dev, cp, nullptr);
    vkDestroyFramebuffer(c.dev, fb, nullptr);
    vkDestroyRenderPass(c.dev, rp, nullptr);
    for (VkImageView v : views) vkDestroyImageView(c.dev, v, nullptr);
    vkDestroyImage(c.dev, color, nullptr);
    vkDestroyImage(c.dev, depth, nullptr);
    vkFreeMemory(c.dev, m, nullptr);
  }
};

struct PipeOpts {
  std::vector<VkDynamicState> dynamic;
  VkCullModeFlags cull = VK_CULL_MODE_NONE;
  VkFrontFace front = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  VkPrimitiveTopology topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  bool depthTest = false, depthWrite = false, stencilTest = false, discard = false, bias = false;
  VkCompareOp depthOp = VK_COMPARE_OP_ALWAYS;
  uint32_t stride = 16;
  bool noFragment = false;
};

VkPipeline makePosPipeline(Ctx& c, DrawRig& rig, VkPipelineLayout pl, const PipeOpts& o, VkShaderModule vs, VkShaderModule fs) {
  VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                           {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
  st[0].stage = VK_SHADER_STAGE_VERTEX_BIT, st[0].module = vs, st[0].pName = "main";
  st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT, st[1].module = fs, st[1].pName = "main";
  VkVertexInputBindingDescription vbd{0, o.stride, VK_VERTEX_INPUT_RATE_VERTEX};
  VkVertexInputAttributeDescription vad{0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0};
  VkPipelineVertexInputStateCreateInfo vis{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vis.vertexBindingDescriptionCount = 1, vis.pVertexBindingDescriptions = &vbd;
  vis.vertexAttributeDescriptionCount = 1, vis.pVertexAttributeDescriptions = &vad;
  VkPipelineInputAssemblyStateCreateInfo ias{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ias.topology = o.topo;
  VkViewport vp{0, 0, 64, 64, 0, 1};
  VkRect2D sc{{0, 0}, {64, 64}};
  VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vps.viewportCount = 1, vps.pViewports = &vp, vps.scissorCount = 1, vps.pScissors = &sc;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.cullMode = o.cull, rs.frontFace = o.front, rs.rasterizerDiscardEnable = o.discard, rs.depthBiasEnable = o.bias;
  rs.lineWidth = 1;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  ds.depthTestEnable = o.depthTest, ds.depthWriteEnable = o.depthWrite, ds.depthCompareOp = o.depthOp;
  ds.stencilTestEnable = o.stencilTest;
  ds.front = ds.back = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS, 0xFF, 0xFF, 0};
  VkPipelineColorBlendAttachmentState cba{};
  cba.colorWriteMask = 0xF;
  VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cbs.attachmentCount = 1, cbs.pAttachments = &cba;
  VkPipelineDynamicStateCreateInfo dys{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dys.dynamicStateCount = uint32_t(o.dynamic.size()), dys.pDynamicStates = o.dynamic.data();
  VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  gp.stageCount = o.noFragment ? 1 : 2, gp.pStages = st, gp.pVertexInputState = &vis, gp.pInputAssemblyState = &ias;
  gp.pViewportState = &vps, gp.pRasterizationState = &rs, gp.pMultisampleState = &ms, gp.pDepthStencilState = &ds;
  gp.pColorBlendState = &cbs, gp.pDynamicState = o.dynamic.empty() ? nullptr : &dys;
  gp.layout = pl, gp.renderPass = rig.rp;
  VkPipeline pipe = VK_NULL_HANDLE;
  vkCreateGraphicsPipelines(c.dev, VK_NULL_HANDLE, 1, &gp, nullptr, &pipe);
  return pipe;
}

// VK_EXT_extended_dynamic_state(2) / Vulkan 1.3: vkCmdSetCullMode & co. replace the pipeline's
// cull mode, front face, topology, depth/stencil state, discard and bias enables, and the strides
// of vkCmdBindVertexBuffers2; each distinct combination is one core pipeline variant, a state the
// pipeline does not list as dynamic is never replaced.
void testExtendedDynamicState() {
  vpvk_config_set_quest_identity(1);  // asserts the Quest 3 / Adreno 740 profile: opt-in
  unsetenv("VPVK_EXTRA_EXTENSIONS");
  {
    Ctx g;
    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT e1{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT};
    VkPhysicalDeviceExtendedDynamicState2FeaturesEXT e2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT, &e1};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &e2};
    vkGetPhysicalDeviceFeatures2(g.pd, &f2);
    CHECK(!e1.extendedDynamicState && !e2.extendedDynamicState2);  // games: behind the gate
  }
  setenv("VPVK_EXTRA_EXTENSIONS", "1", 1);
  Ctx c({VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME, VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME});
  CHECK(c.dev != nullptr);
  VkPhysicalDeviceExtendedDynamicStateFeaturesEXT e1{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT};
  VkPhysicalDeviceExtendedDynamicState2FeaturesEXT e2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT, &e1};
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &e2};
  vkGetPhysicalDeviceFeatures2(c.pd, &f2);
  CHECK(e1.extendedDynamicState && e2.extendedDynamicState2);
  CHECK(!e2.extendedDynamicState2LogicOp && !e2.extendedDynamicState2PatchControlPoints);

  DrawRig rig(c);
  VkShaderModule vs = module(c, "pos.vert.spv"), fs = module(c, "solid.frag.spv");
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  VkPipelineLayout pl;
  vkCreatePipelineLayout(c.dev, &plci, nullptr, &pl);
  VkDeviceMemory bm = alloc(c, 1 << 20, 0);
  VkBuffer vb = buffer(c, 4096, bm, 0);

  PipeOpts all;
  all.dynamic = {VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE, VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
                 VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT, VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT,
                 VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE, VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
                 VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
                 VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
                 VK_DYNAMIC_STATE_STENCIL_OP, VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE,
                 VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE, VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE};
  all.cull = VK_CULL_MODE_BACK_BIT;
  all.depthTest = true, all.depthWrite = true, all.depthOp = VK_COMPARE_OP_LESS;
  VkPipeline dynPipe = makePosPipeline(c, rig, pl, all, vs, fs);
  PipeOpts fixed = all;
  fixed.dynamic.clear();
  VkPipeline fixedPipe = makePosPipeline(c, rig, pl, fixed, vs, fs);
  const PipelineObj* po = from<PipelineObj>(dynPipe);
  const PipelineObj* fo = from<PipelineObj>(fixedPipe);
  CHECK(po->ok && fo->ok && po->dynStride && !fo->dynStride && !po->sources.empty() && fo->sources.empty());
  c.gpu().pipelines().waitIdle();

  rig.begin();
  VkBuffer bufs[1] = {vb};
  VkDeviceSize offs[1] = {0}, sizes[1] = {2048}, strides[1] = {32};
  auto setViewport = reinterpret_cast<PFN_vkCmdSetViewportWithCount>(vkGetDeviceProcAddr(c.dev, "vkCmdSetViewportWithCountEXT"));
  auto setScissor = reinterpret_cast<PFN_vkCmdSetScissorWithCount>(vkGetDeviceProcAddr(c.dev, "vkCmdSetScissorWithCount"));
  auto bind2 = reinterpret_cast<PFN_vkCmdBindVertexBuffers2>(vkGetDeviceProcAddr(c.dev, "vkCmdBindVertexBuffers2EXT"));
  CHECK(setViewport && setScissor && bind2);
  VkViewport vp{4, 8, 32, 16, 0, 1};
  VkRect2D sc{{2, 2}, {40, 40}};
  // 0: dynamic pipeline, nothing set: its own state (same pipeline).
  vkCmdBindPipeline(rig.cb, VK_PIPELINE_BIND_POINT_GRAPHICS, dynPipe);
  vkCmdBindVertexBuffers(rig.cb, 0, 1, bufs, offs);
  // El contador se respeta: con maxViewports = 1 solo cuenta el primero (los demás no se leen).
  const VkViewport vps[3] = {vp, {99, 99, 1, 1, 0, 1}, {77, 77, 1, 1, 0, 1}};
  const VkRect2D scs[3] = {sc, {{9, 9}, {1, 1}}, {{8, 8}, {1, 1}}};
  setViewport(rig.cb, 3, vps), setScissor(rig.cb, 3, scs);
  vkCmdDraw(rig.cb, 3, 1, 0, 0);
  // 1: everything replaced.
  vkCmdSetCullMode(rig.cb, VK_CULL_MODE_FRONT_BIT);
  vkCmdSetFrontFace(rig.cb, VK_FRONT_FACE_CLOCKWISE);
  vkCmdSetPrimitiveTopology(rig.cb, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP);
  vkCmdSetDepthTestEnable(rig.cb, VK_TRUE);
  vkCmdSetDepthWriteEnable(rig.cb, VK_FALSE);
  vkCmdSetDepthCompareOp(rig.cb, VK_COMPARE_OP_GREATER);
  vkCmdSetDepthBoundsTestEnable(rig.cb, VK_FALSE);
  vkCmdSetStencilTestEnable(rig.cb, VK_TRUE);
  vkCmdSetStencilOp(rig.cb, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                    VK_STENCIL_OP_ZERO, VK_COMPARE_OP_EQUAL);
  vkCmdSetDepthBiasEnable(rig.cb, VK_TRUE);
  vkCmdSetPrimitiveRestartEnable(rig.cb, VK_FALSE);
  vkCmdSetRasterizerDiscardEnable(rig.cb, VK_FALSE);
  bind2(rig.cb, 0, 1, bufs, offs, sizes, strides);
  vkCmdDraw(rig.cb, 3, 1, 0, 0);
  // 2: the same values again: the same variant. 3: rasterizer discard on.
  vkCmdDraw(rig.cb, 6, 1, 0, 0);
  vkCmdSetRasterizerDiscardEnable(rig.cb, VK_TRUE);
  vkCmdDraw(rig.cb, 3, 1, 0, 0);
  vkCmdSetRasterizerDiscardEnable(rig.cb, VK_FALSE);
  // 4: front and back culling removes triangles, not points.
  vkCmdSetCullMode(rig.cb, VK_CULL_MODE_FRONT_AND_BACK);
  vkCmdDraw(rig.cb, 3, 1, 0, 0);
  vkCmdSetPrimitiveTopology(rig.cb, VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
  vkCmdDraw(rig.cb, 3, 1, 0, 0);
  // 6: a pipeline with the same states fixed ignores every vkCmdSet*.
  vkCmdBindPipeline(rig.cb, VK_PIPELINE_BIND_POINT_GRAPHICS, fixedPipe);
  vkCmdDraw(rig.cb, 3, 1, 0, 0);
  vpgpu::Frame frame;
  TranslateStats ts;
  const std::vector<vpgpu::Draw> d = rig.finish(frame, ts);
  CHECK(d.size() == 6 && ts.skippedDraws == 1);  // the culled triangle draw is skipped
  if (d.size() == 6) {
    CHECK(d[0].pipeline == po->key && d[0].viewport.x == 4 && d[0].viewport.height == 16 && d[0].scissor.x == 2);
    // Draw 1: expected core state = the pipeline's with the replaced values.
    vpgpu::PipelineState want = po->state;
    want.cull = vpgpu::CullMode::Front;
    want.primitive = vpgpu::Primitive::TriangleStrip;
    want.depthTest = true, want.depthWrite = false, want.depthCompare = vpgpu::CompareOp::Greater;
    want.stencilTest = true;
    for (vpgpu::StencilFace* f : {&want.stencilFront, &want.stencilBack}) {
      f->fail = vpgpu::StencilOp::Replace, f->pass = vpgpu::StencilOp::IncrementClamp;
      f->depthFail = vpgpu::StencilOp::Zero, f->compare = vpgpu::CompareOp::Equal;
    }
    CHECK(want.key() != po->key);
    // Strides: 32 from vkCmdBindVertexBuffers2 -> a retranslated shader pair (the vertex fetch is in
    // the shader), so the variant also carries the variant's shader ids.
    CHECK(d[1].pipeline != po->key && !po->texelVariants.empty());
    const PipelineObj::TexelVariant& tv = po->texelVariants.front();
    CHECK(tv.ok && tv.strides.size() == 1 && tv.strides[0] == 32 && tv.state.vertexShader != po->state.vertexShader);
    want.vertexShader = tv.state.vertexShader, want.fragmentShader = tv.state.fragmentShader;
    CHECK(d[1].pipeline == want.key());
    CHECK(d[1].frontFace == vpgpu::Winding::Clockwise && d[1].primitive == vpgpu::Primitive::TriangleStrip);
    CHECK(d[2].pipeline == d[1].pipeline);
    // Draw 3: discard = no fragment function, nothing written.
    vpgpu::PipelineState disc = want;
    disc.fragmentShader = 0;
    for (auto& b : disc.blend) b.writeMask = 0;
    disc.depthTest = disc.depthWrite = disc.stencilTest = false;
    CHECK(d[3].pipeline == disc.key());
    // Draw 4 (cull both + points): drawn; the triangle-strip draw before it was skipped.
    CHECK(d[4].primitive == vpgpu::Primitive::Points);
    // Draw 5: the fixed pipeline, whatever was set.
    CHECK(d[5].pipeline == fo->key && d[5].frontFace == vpgpu::Winding::CounterClockwise);
    // vkCmdBindVertexBuffers2 sizes bound the vertex buffer range.
    bool sized = false;
    for (const vpgpu::Access& a : d[1].bindings)
      if (a.kind == vpgpu::BindKind::Buffer && a.range.end - a.range.begin == 2048) sized = true;
    CHECK(sized);
  }
  vkDestroyPipeline(c.dev, dynPipe, nullptr);
  vkDestroyPipeline(c.dev, fixedPipe, nullptr);
  vkDestroyPipelineLayout(c.dev, pl, nullptr);
  vkDestroyBuffer(c.dev, vb, nullptr);
  vkFreeMemory(c.dev, bm, nullptr);
  vkDestroyShaderModule(c.dev, vs, nullptr);
  vkDestroyShaderModule(c.dev, fs, nullptr);
  unsetenv("VPVK_EXTRA_EXTENSIONS");
}

// VK_EXT_inline_uniform_block / Vulkan 1.3: a binding whose descriptorCount is a byte size, written
// through VkWriteDescriptorSetInlineUniformBlock / copied / updated from a template; a draw copies
// the bytes into its upload memory as a constant buffer at the binding's slot.
void testInlineUniformBlock() {
  vpvk_config_set_quest_identity(1);  // asserts the Quest 3 / Adreno 740 profile: opt-in
  unsetenv("VPVK_EXTRA_EXTENSIONS");
  {
    Ctx g;
    VkPhysicalDeviceInlineUniformBlockFeatures iu{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &iu};
    vkGetPhysicalDeviceFeatures2(g.pd, &f2);
    CHECK(!iu.inlineUniformBlock);
  }
  setenv("VPVK_EXTRA_EXTENSIONS", "1", 1);
  Ctx c({VK_EXT_INLINE_UNIFORM_BLOCK_EXTENSION_NAME});
  CHECK(c.dev != nullptr);
  VkPhysicalDeviceInlineUniformBlockFeatures iu{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES};
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &iu};
  vkGetPhysicalDeviceFeatures2(c.pd, &f2);
  CHECK(iu.inlineUniformBlock && !iu.descriptorBindingInlineUniformBlockUpdateAfterBind);
  VkPhysicalDeviceInlineUniformBlockProperties ip{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_PROPERTIES};
  VkPhysicalDeviceVulkan13Properties p13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES, &ip};
  VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &p13};
  vkGetPhysicalDeviceProperties2(c.pd, &p2);
  CHECK(ip.maxInlineUniformBlockSize >= 256 && ip.maxDescriptorSetInlineUniformBlocks >= 4 &&
        p13.maxInlineUniformBlockSize == ip.maxInlineUniformBlockSize && p13.maxInlineUniformTotalSize >= 256);

  DrawRig rig(c);
  VkShaderModule vs = module(c, "pos.vert.spv"), fs = module(c, "inline_ubo.frag.spv");
  VkDescriptorSetLayoutBinding b{3, VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK, 16, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  lci.bindingCount = 1, lci.pBindings = &b;
  VkDescriptorSetLayout sl;
  vkCreateDescriptorSetLayout(c.dev, &lci, nullptr, &sl);
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1, plci.pSetLayouts = &sl;
  VkPipelineLayout pl;
  vkCreatePipelineLayout(c.dev, &plci, nullptr, &pl);
  VkDescriptorPoolInlineUniformBlockCreateInfo pin{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_INLINE_UNIFORM_BLOCK_CREATE_INFO};
  pin.maxInlineUniformBlockBindings = 3;
  VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK, 48};
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, &pin};
  dpci.maxSets = 3, dpci.poolSizeCount = 1, dpci.pPoolSizes = &psz;
  VkDescriptorPool pool;
  vkCreateDescriptorPool(c.dev, &dpci, nullptr, &pool);
  VkDescriptorSetLayout sls[3] = {sl, sl, sl};
  VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dai.descriptorPool = pool, dai.descriptorSetCount = 3, dai.pSetLayouts = sls;
  VkDescriptorSet sets[3];
  vkAllocateDescriptorSets(c.dev, &dai, sets);

  // Set 0: bytes 4..11 written (offset in dstArrayElement, size in descriptorCount).
  const float data[2] = {0.5f, 0.75f};
  VkWriteDescriptorSetInlineUniformBlock wib{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK};
  wib.dataSize = 8, wib.pData = data;
  VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &wib};
  w.dstSet = sets[0], w.dstBinding = 3, w.dstArrayElement = 4, w.descriptorCount = 8;
  w.descriptorType = VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK;
  vkUpdateDescriptorSets(c.dev, 1, &w, 0, nullptr);
  // Set 1: a copy of set 0 (all 16 bytes); set 2: from a template (bytes 8..15).
  VkCopyDescriptorSet cp{VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET};
  cp.srcSet = sets[0], cp.srcBinding = 3, cp.dstSet = sets[1], cp.dstBinding = 3, cp.descriptorCount = 16;
  vkUpdateDescriptorSets(c.dev, 0, nullptr, 1, &cp);
  VkDescriptorUpdateTemplateEntry te{3, 8, 8, VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK, 4, 0};  // dstArrayElement 8, 8 bytes at offset 4
  VkDescriptorUpdateTemplateCreateInfo tci{VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO};
  tci.descriptorUpdateEntryCount = 1, tci.pDescriptorUpdateEntries = &te;
  tci.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET, tci.descriptorSetLayout = sl;
  VkDescriptorUpdateTemplate tpl;
  vkCreateDescriptorUpdateTemplate(c.dev, &tci, nullptr, &tpl);
  const float tdata[4] = {-1.0f, 9.0f, 2.0f, 3.0f};  // bytes 4..11 of this are {9, 2}
  vkUpdateDescriptorSetWithTemplate(c.dev, sets[2], tpl, tdata);

  PipeOpts o;
  VkPipeline pipe = makePosPipeline(c, rig, pl, o, vs, fs);
  const PipelineObj* po = from<PipelineObj>(pipe);
  CHECK(po && po->ok);
  rig.begin();
  vkCmdBindPipeline(rig.cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
  VkDeviceMemory bm = alloc(c, 1 << 20, 0);
  VkBuffer vb = buffer(c, 4096, bm, 0);
  VkDeviceSize zero = 0;
  vkCmdBindVertexBuffers(rig.cb, 0, 1, &vb, &zero);
  for (int i = 0; i < 3; ++i) {
    vkCmdBindDescriptorSets(rig.cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &sets[i], 0, nullptr);
    vkCmdDraw(rig.cb, 3, 1, 0, 0);
  }
  // A later update does not reach the draws already recorded.
  const float late = 123.0f;
  wib.dataSize = 4, wib.pData = &late;
  w.dstArrayElement = 4, w.descriptorCount = 4;
  vkUpdateDescriptorSets(c.dev, 1, &w, 0, nullptr);
  vpgpu::Frame frame;
  TranslateStats ts;
  Recorder& rec = *from<CommandBuffer>(rig.cb)->rec;
  const std::vector<vpgpu::Draw> d = rig.finish(frame, ts);
  CHECK(d.size() == 3 && ts.skippedDraws == 0 && !rec.arena.chunks.empty());
  const float want[3][4] = {{0, 0.5f, 0.75f, 0}, {0, 0.5f, 0.75f, 0}, {0, 0, 9.0f, 2.0f}};
  for (size_t i = 0; i < d.size() && !rec.arena.chunks.empty(); ++i) {
    int found = 0;
    for (const vpgpu::Access& a : d[i].bindings) {
      if (a.kind != vpgpu::BindKind::ConstantBuffer || a.slot != 0) continue;
      ++found;
      float got[4];
      std::memcpy(got, rec.arena.chunks[0].host + a.range.begin, sizeof got);
      CHECK(a.range.end - a.range.begin >= 16);
      CHECK(!std::memcmp(got, want[i], sizeof got));
    }
    CHECK(found == 1);
  }
  vkDestroyPipeline(c.dev, pipe, nullptr);
  vkDestroyDescriptorUpdateTemplate(c.dev, tpl, nullptr);
  vkDestroyDescriptorPool(c.dev, pool, nullptr);
  vkDestroyPipelineLayout(c.dev, pl, nullptr);
  vkDestroyDescriptorSetLayout(c.dev, sl, nullptr);
  vkDestroyBuffer(c.dev, vb, nullptr);
  vkFreeMemory(c.dev, bm, nullptr);
  vkDestroyShaderModule(c.dev, vs, nullptr);
  vkDestroyShaderModule(c.dev, fs, nullptr);
  unsetenv("VPVK_EXTRA_EXTENSIONS");
}

// VK_KHR_buffer_device_address / Vulkan 1.2: addresses of bound buffers (the backend's, here the null
// backend's distinct fake ones), the aliases, and a compute pipeline that dereferences a pointer.
void testBufferDeviceAddress() {
  vpvk_config_set_quest_identity(1);  // asserts the Quest 3 / Adreno 740 profile: opt-in
  unsetenv("VPVK_EXTRA_EXTENSIONS");
  {
    Ctx g;
    VkPhysicalDeviceBufferDeviceAddressFeatures ba{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &ba};
    vkGetPhysicalDeviceFeatures2(g.pd, &f2);
    CHECK(!ba.bufferDeviceAddress);
  }
  setenv("VPVK_EXTRA_EXTENSIONS", "1", 1);
  Ctx c({VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME});
  CHECK(c.dev != nullptr);
  VkPhysicalDeviceBufferDeviceAddressFeatures ba{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &ba};
  vkGetPhysicalDeviceFeatures2(c.pd, &f2);
  CHECK(ba.bufferDeviceAddress && !ba.bufferDeviceAddressCaptureReplay && !ba.bufferDeviceAddressMultiDevice);
  CHECK(vkGetDeviceProcAddr(c.dev, "vkGetBufferDeviceAddress") && vkGetDeviceProcAddr(c.dev, "vkGetBufferDeviceAddressKHR") &&
        vkGetDeviceProcAddr(c.dev, "vkGetBufferDeviceAddressEXT"));

  VkMemoryAllocateFlagsInfo fl{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  fl.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fl};
  ai.allocationSize = 1 << 16;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  CHECK(vkAllocateMemory(c.dev, &ai, nullptr, &mem) == VK_SUCCESS);
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = 4096;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  VkBuffer b1, b2;
  vkCreateBuffer(c.dev, &bi, nullptr, &b1);
  vkCreateBuffer(c.dev, &bi, nullptr, &b2);
  VkBufferDeviceAddressInfo ni{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  ni.buffer = b1;
  CHECK(vkGetBufferDeviceAddress(c.dev, &ni) == 0);  // not bound yet
  vkBindBufferMemory(c.dev, b1, mem, 0);
  vkBindBufferMemory(c.dev, b2, mem, 4096);
  ni.buffer = b1;
  const VkDeviceAddress a1 = vkGetBufferDeviceAddress(c.dev, &ni);
  ni.buffer = b2;
  const VkDeviceAddress a2 = vkGetBufferDeviceAddress(c.dev, &ni);
  ni.buffer = b1;
  CHECK(a1 != 0 && a2 != 0 && a1 != a2 && vkGetBufferDeviceAddress(c.dev, &ni) == a1);
  CHECK(vkGetBufferOpaqueCaptureAddress(c.dev, &ni) == 0);

  // A compute pipeline whose shader reads a pointer from a storage buffer and dereferences it.
  VkDescriptorSetLayoutBinding b[2] = {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                       {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
  VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  lci.bindingCount = 2, lci.pBindings = b;
  VkDescriptorSetLayout sl;
  vkCreateDescriptorSetLayout(c.dev, &lci, nullptr, &sl);
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1, plci.pSetLayouts = &sl;
  VkPipelineLayout pl;
  vkCreatePipelineLayout(c.dev, &plci, nullptr, &pl);
  VkShaderModule cs = module(c, "bda.comp.spv");
  VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT, ci.stage.module = cs, ci.stage.pName = "main";
  ci.layout = pl;
  VkPipeline pipe;
  vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe);
  const PipelineObj* po = from<PipelineObj>(pipe);
  CHECK(po && po->ok && po->state.computeShader != 0);
  c.gpu().pipelines().waitIdle();
  vkDestroyPipeline(c.dev, pipe, nullptr);
  vkDestroyShaderModule(c.dev, cs, nullptr);
  vkDestroyPipelineLayout(c.dev, pl, nullptr);
  vkDestroyDescriptorSetLayout(c.dev, sl, nullptr);
  vkDestroyBuffer(c.dev, b1, nullptr);
  vkDestroyBuffer(c.dev, b2, nullptr);
  vkFreeMemory(c.dev, mem, nullptr);
  unsetenv("VPVK_EXTRA_EXTENSIONS");
}

// VK_KHR_sampler_ycbcr_conversion behind the CTS gate: multi-planar formats as one texture per plane,
// conversion objects, views over all planes, immutable samplers with a conversion in the pipeline
// layout (plane textures at their own slots), copies by plane aspect.
void testYcbcr() {
  vpvk_config_set_quest_identity(1);  // asserts the Quest 3 / Adreno 740 profile: opt-in
  unsetenv("VPVK_EXTRA_EXTENSIONS");
  {
    Ctx g;
    VkPhysicalDeviceSamplerYcbcrConversionFeatures yf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &yf};
    vkGetPhysicalDeviceFeatures2(g.pd, &f2);
    CHECK(!yf.samplerYcbcrConversion);
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(g.pd, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, &fp);
    CHECK(fp.optimalTilingFeatures == 0);
  }
  setenv("VPVK_EXTRA_EXTENSIONS", "1", 1);
  Ctx c({VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME});
  CHECK(c.dev != nullptr);
  VkPhysicalDeviceSamplerYcbcrConversionFeatures yf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES};
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &yf};
  vkGetPhysicalDeviceFeatures2(c.pd, &f2);
  CHECK(yf.samplerYcbcrConversion);

  // Format and image format properties.
  const VkFormat formats[] = {VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM,
                              VK_FORMAT_G8_B8R8_2PLANE_422_UNORM, VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM,
                              VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM};
  for (VkFormat f : formats) {
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(c.pd, f, &fp);
    const VkFormatFeatureFlags want = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                      VK_FORMAT_FEATURE_TRANSFER_DST_BIT | VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT |
                                      VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT |
                                      VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT;
    CHECK((fp.optimalTilingFeatures & want) == want);
    CHECK(!(fp.optimalTilingFeatures & (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
                                        VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT)));
    CHECK(fp.linearTilingFeatures == 0 && fp.bufferFeatures == 0);
    VkSamplerYcbcrConversionImageFormatProperties yp{VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES};
    VkImageFormatProperties2 ip{VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2, &yp};
    VkPhysicalDeviceImageFormatInfo2 info{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2};
    info.format = f, info.type = VK_IMAGE_TYPE_2D, info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    CHECK(vkGetPhysicalDeviceImageFormatProperties2(c.pd, &info, &ip) == VK_SUCCESS);
    CHECK(yp.combinedImageSamplerDescriptorCount == 1 && ip.imageFormatProperties.sampleCounts == VK_SAMPLE_COUNT_1_BIT);
    info.tiling = VK_IMAGE_TILING_LINEAR;
    CHECK(vkGetPhysicalDeviceImageFormatProperties2(c.pd, &info, &ip) == VK_ERROR_FORMAT_NOT_SUPPORTED);
    info.tiling = VK_IMAGE_TILING_OPTIMAL, info.type = VK_IMAGE_TYPE_3D;
    CHECK(vkGetPhysicalDeviceImageFormatProperties2(c.pd, &info, &ip) == VK_ERROR_FORMAT_NOT_SUPPORTED);
    info.type = VK_IMAGE_TYPE_2D, info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    CHECK(vkGetPhysicalDeviceImageFormatProperties2(c.pd, &info, &ip) == VK_ERROR_FORMAT_NOT_SUPPORTED);
  }

  // Conversion objects: single-plane formats are refused, the parameters reach the shader description.
  VkSamplerYcbcrConversionCreateInfo cci{VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO};
  cci.format = VK_FORMAT_R8G8B8A8_UNORM;
  VkSamplerYcbcrConversion bad = VK_NULL_HANDLE;
  CHECK(vkCreateSamplerYcbcrConversion(c.dev, &cci, nullptr, &bad) != VK_SUCCESS);
  cci.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
  cci.ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;
  cci.ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_NARROW;
  cci.xChromaOffset = VK_CHROMA_LOCATION_COSITED_EVEN, cci.yChromaOffset = VK_CHROMA_LOCATION_MIDPOINT;
  cci.chromaFilter = VK_FILTER_LINEAR;
  cci.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_ONE};
  VkSamplerYcbcrConversion conv = VK_NULL_HANDLE;
  CHECK(vkCreateSamplerYcbcrConversion(c.dev, &cci, nullptr, &conv) == VK_SUCCESS && conv);
  const YcbcrConversionObj& co = *from<YcbcrConversionHandle>(conv)->obj;
  CHECK(co.spec.planes == 2 && co.spec.model == spirv::YcbcrModel::Ycbcr709 && co.spec.range == spirv::YcbcrRange::Narrow);
  CHECK(co.spec.cositedX && !co.spec.cositedY && co.spec.subX && co.spec.subY && co.spec.swizzle[3] == 2);

  // A 64 x 32 2-plane 4:2:0 image: plane 0 R8 64 x 32, plane 1 RG8 32 x 16; memory covers both.
  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = VK_IMAGE_TYPE_2D, ii.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
  ii.extent = {64, 32, 1}, ii.mipLevels = 1, ii.arrayLayers = 1, ii.samples = VK_SAMPLE_COUNT_1_BIT;
  ii.tiling = VK_IMAGE_TILING_OPTIMAL, ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  VkImage im = VK_NULL_HANDLE;
  CHECK(vkCreateImage(c.dev, &ii, nullptr, &im) == VK_SUCCESS);
  const Image* img = from<Image>(im);
  CHECK(img->planes.size() == 1 && img->desc.format == vpgpu::Format::R8Unorm && img->desc.width == 64 && img->desc.height == 32);
  CHECK(img->planes[0]->desc.format == vpgpu::Format::RG8Unorm && img->planes[0]->desc.width == 32 && img->planes[0]->desc.height == 16);
  VkMemoryRequirements mr;
  vkGetImageMemoryRequirements(c.dev, im, &mr);
  CHECK(mr.size == 64 * 32 + 32 * 16 * 2 + 0 || mr.size >= 64 * 32 + 32 * 16 * 2);
  VkImagePlaneMemoryRequirementsInfo pri{VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO};
  pri.planeAspect = VK_IMAGE_ASPECT_PLANE_1_BIT;
  VkImageMemoryRequirementsInfo2 mri{VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, &pri, im};
  VkMemoryRequirements2 mr2{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
  vkGetImageMemoryRequirements2(c.dev, &mri, &mr2);
  CHECK(mr2.memoryRequirements.size >= 32 * 16 * 2 && mr2.memoryRequirements.size < mr.size);
  VkDeviceMemory mem = alloc(c, 1 << 20, 0);
  CHECK(vkBindImageMemory(c.dev, im, mem, 0) == VK_SUCCESS);
  CHECK(img->id && img->planes[0]->id && img->id != img->planes[0]->id);
  CHECK(img->planes[0]->offset == img->offset + 64 * 32 || img->planes[0]->offset > img->offset);
  // Odd sizes and unsupported shapes are refused.
  VkImageCreateInfo bi = ii;
  bi.imageType = VK_IMAGE_TYPE_3D;
  VkImage none = VK_NULL_HANDLE;
  CHECK(vkCreateImage(c.dev, &bi, nullptr, &none) == VK_ERROR_FORMAT_NOT_SUPPORTED);
  bi = ii, bi.samples = VK_SAMPLE_COUNT_4_BIT;
  CHECK(vkCreateImage(c.dev, &bi, nullptr, &none) == VK_ERROR_FORMAT_NOT_SUPPORTED);

  // Views: the multi-planar format gives views of all planes; a plane format and aspect, that plane.
  VkSamplerYcbcrConversionInfo yci{VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO};
  yci.conversion = conv;
  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, &yci};
  vi.image = im, vi.viewType = VK_IMAGE_VIEW_TYPE_2D, vi.format = ii.format;
  vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkImageView view = VK_NULL_HANDLE;
  CHECK(vkCreateImageView(c.dev, &vi, nullptr, &view) == VK_SUCCESS);
  const ImageView* iv = from<ImageView>(view);
  CHECK(iv->image == img && iv->planeViews.size() == 1 && iv->planeViews[0]->image == img->planes[0].get());
  VkImageViewCreateInfo pv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  pv.image = im, pv.viewType = VK_IMAGE_VIEW_TYPE_2D, pv.format = VK_FORMAT_R8G8_UNORM;
  pv.subresourceRange = {VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 1, 0, 1};
  VkImageView chroma = VK_NULL_HANDLE;
  CHECK(vkCreateImageView(c.dev, &pv, nullptr, &chroma) == VK_SUCCESS);
  CHECK(from<ImageView>(chroma)->image == img->planes[0].get());

  // Pipeline layout with the immutable sampler: planes 1.. get Texture slots after the binding's own.
  VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, &yci};
  sci.magFilter = sci.minFilter = VK_FILTER_LINEAR, sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  VkSampler smp = VK_NULL_HANDLE;
  CHECK(vkCreateSampler(c.dev, &sci, nullptr, &smp) == VK_SUCCESS);
  CHECK(from<Sampler>(smp)->ycbcr == from<YcbcrConversionHandle>(conv)->obj);
  VkDescriptorSetLayoutBinding lb{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, &smp};
  VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  lci.bindingCount = 1, lci.pBindings = &lb;
  VkDescriptorSetLayout sl;
  vkCreateDescriptorSetLayout(c.dev, &lci, nullptr, &sl);
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1, plci.pSetLayouts = &sl;
  VkPipelineLayout pl;
  vkCreatePipelineLayout(c.dev, &plci, nullptr, &pl);
  const PipelineLayoutObj* plo = reinterpret_cast<std::shared_ptr<const PipelineLayoutObj>*>(pl)->get();
  CHECK(plo->slots.size() == 1);
  if (plo->slots.size() == 1) {
    CHECK(plo->slots[0].ycbcr && plo->slots[0].slot == 0 && plo->slots[0].planeSlot[0] == 1);
  }
  VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1, dpci.poolSizeCount = 1, dpci.pPoolSizes = &psz;
  VkDescriptorPool pool;
  vkCreateDescriptorPool(c.dev, &dpci, nullptr, &pool);
  VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dai.descriptorPool = pool, dai.descriptorSetCount = 1, dai.pSetLayouts = &sl;
  VkDescriptorSet set;
  vkAllocateDescriptorSets(c.dev, &dai, &set);
  // The write names no sampler (immutable) or a different one: the immutable one stays.
  VkSampler other = VK_NULL_HANDLE;
  VkSamplerCreateInfo plain{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  vkCreateSampler(c.dev, &plain, nullptr, &other);
  VkDescriptorImageInfo dii{other, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  w.dstSet = set, w.dstBinding = 0, w.descriptorCount = 1, w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  w.pImageInfo = &dii;
  vkUpdateDescriptorSets(c.dev, 1, &w, 0, nullptr);
  CHECK(from<DescriptorSetObj>(set)->slots[0][0].sampler == from<Sampler>(smp));

  DrawRig rig(c);
  VkShaderModule vs = module(c, "pos.vert.spv"), fs = module(c, "ycbcr.frag.spv");
  PipeOpts po;
  VkPipeline pipe = makePosPipeline(c, rig, pl, po, vs, fs);
  const PipelineObj* pobj = from<PipelineObj>(pipe);
  CHECK(pobj && pobj->ok);
  rig.begin();
  vkCmdBindPipeline(rig.cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
  VkDeviceMemory bm = alloc(c, 1 << 20, 0);
  VkBuffer vb = buffer(c, 4096, bm, 0);
  VkDeviceSize zero = 0;
  vkCmdBindVertexBuffers(rig.cb, 0, 1, &vb, &zero);
  vkCmdBindDescriptorSets(rig.cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &set, 0, nullptr);
  vkCmdDraw(rig.cb, 3, 1, 0, 0);
  vpgpu::Frame frame;
  TranslateStats ts;
  const std::vector<vpgpu::Draw> d = rig.finish(frame, ts);
  CHECK(d.size() == 1 && ts.skippedDraws == 0);
  if (!d.empty()) {
    int plane0 = 0, plane1 = 0, samplers = 0;
    for (const vpgpu::Access& a : d[0].bindings) {
      if (a.kind == vpgpu::BindKind::Texture && a.slot == 0 && (a.range.index == img->id.index && a.range.generation == img->id.generation)) ++plane0;
      if (a.kind == vpgpu::BindKind::Texture && a.slot == 1 && (a.range.index == img->planes[0]->id.index && a.range.generation == img->planes[0]->id.generation)) ++plane1;
      if (a.kind == vpgpu::BindKind::Sampler && a.slot == 0) ++samplers;
    }
    CHECK(plane0 == 1 && plane1 == 1 && samplers == 1);
  }

  // Copies by plane aspect reach that plane's texture, with the plane's own extent.
  {
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    VkCommandPool cp2;
    vkCreateCommandPool(c.dev, &cpi, nullptr, &cp2);
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = cp2, ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    vkAllocateCommandBuffers(c.dev, &ai, &cb);
    VkCommandBufferBeginInfo bgi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cb, &bgi);
    VkBufferImageCopy r0{}, r1{};
    r0.imageSubresource = {VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1}, r0.imageExtent = {64, 32, 1};
    r1.bufferOffset = 4096, r1.imageSubresource = {VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 0, 1}, r1.imageExtent = {32, 16, 1};
    vkCmdCopyBufferToImage(cb, vb, im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r0);
    vkCmdCopyBufferToImage(cb, vb, im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r1);
    vkEndCommandBuffer(cb);
    vpgpu::Frame f2;
    TranslateStats ts2;
    translate(*from<CommandBuffer>(cb)->rec, f2, ts2);
    int to0 = 0, to1 = 0;
    for (const auto& cmd : f2.commands())
      if (const auto* cp = std::get_if<vpgpu::Copy>(&cmd)) {
        if (cp->dstTexture.texture == img->id && cp->dstTexture.width == 64) ++to0;
        if (cp->dstTexture.texture == img->planes[0]->id && cp->dstTexture.width == 32 && cp->dstTexture.height == 16 &&
            cp->srcBuffer.bytesPerRow == 64)
          ++to1;
      }
    CHECK(to0 == 1 && to1 == 1);
    vkDestroyCommandPool(c.dev, cp2, nullptr);
  }

  // Disjoint binding: each plane bound on its own, in any order.
  {
    VkImageCreateInfo di = ii;
    di.flags = VK_IMAGE_CREATE_DISJOINT_BIT;
    VkImage dj = VK_NULL_HANDLE;
    CHECK(vkCreateImage(c.dev, &di, nullptr, &dj) == VK_SUCCESS);
    VkBindImagePlaneMemoryInfo p1{VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO, nullptr, VK_IMAGE_ASPECT_PLANE_1_BIT};
    VkBindImagePlaneMemoryInfo p0{VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO, nullptr, VK_IMAGE_ASPECT_PLANE_0_BIT};
    VkBindImageMemoryInfo bind[2] = {{VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, &p1, dj, mem, 65536},
                                     {VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, &p0, dj, mem, 0}};
    CHECK(vkBindImageMemory2(c.dev, 2, bind) == VK_SUCCESS);
    const Image* dimg = from<Image>(dj);
    CHECK(dimg->id && dimg->planes[0]->id && dimg->planes[0]->offset == 65536 && dimg->offset == 0);
    vkDestroyImage(c.dev, dj, nullptr);
  }

  vkDestroyPipeline(c.dev, pipe, nullptr);
  vkDestroyDescriptorPool(c.dev, pool, nullptr);
  vkDestroyPipelineLayout(c.dev, pl, nullptr);
  vkDestroyDescriptorSetLayout(c.dev, sl, nullptr);
  vkDestroySampler(c.dev, smp, nullptr);
  vkDestroySampler(c.dev, other, nullptr);
  vkDestroyImageView(c.dev, chroma, nullptr);
  vkDestroyImageView(c.dev, view, nullptr);
  vkDestroyImage(c.dev, im, nullptr);
  vkDestroySamplerYcbcrConversion(c.dev, conv, nullptr);
  vkDestroyBuffer(c.dev, vb, nullptr);
  unsetenv("VPVK_EXTRA_EXTENSIONS");
}


// VPKTAA nativo, máscara de UI: clasificación de draws (taa::DrawClassifier), mezcla con máscara,
// UI sin jitter en el traductor y, con un dispositivo (backend nulo), el attachment R8 del pase de
// ojo y las variantes de pipeline (3D con máscara a write mask 0, UI escribiendo la máscara).
// VPKTAA nativo sobre vpgles/OpenXR: convención de profundidad del pase de ojo, registro del pase para la
// profundidad que el host no da, y el borrado de profundidad dentro del pase (vkCmdClearAttachments).
void testTaaEyeDepth() {
  // 1. reversedZ: el recuento de comparaciones manda; empate o nada -> valor de limpieza; si no, fallback.
  CHECK(taa::reversedZ(false, 1, 10, 2, false) && !taa::reversedZ(false, 0, 2, 10, true));
  CHECK(taa::reversedZ(true, 0.0f, 3, 3, false) && !taa::reversedZ(true, 1.0f, 0, 0, true));
  CHECK(taa::reversedZ(false, 1, 0, 0, true) && !taa::reversedZ(false, 0, 0, 0, false));

  vpvk_taa_jitter in{1, 1920, 1824, 0, 0, 5};
  vpvk_taa_set_jitter(&in);
  vpvk_config_set_taa_native(1);
  const TaaJitter j = taaJitter();
  vpgpu::TextureId color, depth;
  color.index = 1, color.generation = 1, depth.index = 2, depth.generation = 1;
  OpDraw geo;
  geo.draw.count = 300, geo.draw.index = vpgpu::IndexBuffer{};
  geo.draw.viewport = {0, 0, 1920, 1824, 0, 1};
  geo.depthTested = geo.vertexBuffers = true;
  OpDraw opaque = geo, glass = geo;
  opaque.depthWrite = true, opaque.depthCmp = 1;  // GREATER: profundidad invertida
  glass.blended = true, glass.depthCmp = 1;
  // 2. El traductor anota el pase de ojo (color 0, profundidad, limpieza, recuento).
  {
    UiMaskState um;
    Recorder rec;
    OpBeginRendering scene;
    scene.area = {{0, 0}, {1920, 1824}};
    scene.depthWidth = 1920, scene.depthHeight = 1824;
    scene.desc.colors[0].texture = color, scene.desc.colors[0].layer = 1;
    scene.desc.depth.texture = depth, scene.desc.depth.load = vpgpu::LoadAction::Clear, scene.desc.depth.clear.depth = 0.0f;
    rec.ops.emplace_back(scene), rec.ops.emplace_back(opaque), rec.ops.emplace_back(opaque), rec.ops.emplace_back(glass),
        rec.ops.emplace_back(OpEndPass{});
    vpgpu::Frame f;
    TranslateStats st;
    translate(rec, f, st, &j, &um);
    CHECK(um.eyePasses == 1);
    const UiMaskState::EyePass& r = um.eyePass[0];
    CHECK(r.color == color && r.depth == depth && r.colorLayer == 1 && r.layers == 1);
    CHECK(r.cleared && r.clearValue == 0.0f && r.greater == 3 && r.less == 0);
    CHECK(taa::reversedZ(r.cleared, r.clearValue, r.greater, r.less, false));
  }
  // 3. Un pase sin profundidad o sin color 0 (sombras, post) no se anota.
  {
    UiMaskState um;
    Recorder rec;
    OpBeginRendering noColor;
    noColor.area = {{0, 0}, {1920, 1824}};
    noColor.depthWidth = 1920, noColor.depthHeight = 1824;
    noColor.desc.depth.texture = depth;
    rec.ops.emplace_back(noColor), rec.ops.emplace_back(opaque), rec.ops.emplace_back(OpEndPass{});
    vpgpu::Frame f;
    TranslateStats st;
    translate(rec, f, st, &j, &um);
    CHECK(um.eyePasses == 0);
  }
  // 4. vpgles carga siempre la profundidad y limpia con quads: el borrado de profundidad dentro del pase
  // empieza una escena nueva para el clasificador (sin él, «ya hubo opaco» pasaría al frame siguiente).
  {
    UiMaskState um;
    um.classifier.beginPass(false);
    CHECK(um.classifier.classify({true, true, false, false}) == taa::DrawClass::Scene && um.classifier.sawOpaque());
    Recorder rec;
    OpBeginRendering scene;
    scene.area = {{0, 0}, {1920, 1824}};
    scene.depthWidth = 1920, scene.depthHeight = 1824;
    scene.desc.colors[0].texture = color;
    scene.desc.depth.texture = depth, scene.desc.depth.load = vpgpu::LoadAction::Load;  // hereda «ya hubo opaco»
    OpClearAttachments clear;
    VkClearAttachment ca{};
    ca.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT, ca.clearValue.depthStencil.depth = 1.0f;
    clear.attachments.push_back(ca);
    VkClearRect cr{{{0, 0}, {1920, 1824}}, 0, 1};
    clear.rects.push_back(cr);
    OpDraw overlay = glass;  // sin escritura de profundidad
    overlay.depthCmp = -1;
    rec.ops.emplace_back(scene), rec.ops.emplace_back(overlay), rec.ops.emplace_back(clear), rec.ops.emplace_back(overlay),
        rec.ops.emplace_back(OpEndPass{});
    vpgpu::Frame f;
    TranslateStats st;
    translate(rec, f, st, &j, &um);
    CHECK(st.ui.ui == 1 && st.ui.scene == 1);  // antes del borrado: UI; después: escena nueva
    CHECK(um.eyePass[0].cleared && um.eyePass[0].clearValue == 1.0f);
  }
  vpvk_config_set_taa_native(0);
}

void testTaaUiMask() {
  auto near = [](float a, float b, float eps = 1e-6f) { return std::fabs(a - b) <= eps; };
  using taa::DrawClass;
  using taa::DrawFacts;
  // 1. Clasificador: lo que se dibuja sin escribir profundidad DESPUÉS de lo opaco y con mezcla o sin
  // test es UI; antes de lo opaco, o con test y sin mezcla (cielo), es escena; pantalla completa, nunca.
  {
    const DrawFacts opaque{true, true, false, false}, sky{true, false, false, false}, glass{true, false, true, false},
        noTest{false, false, false, false}, noTestBlend{false, false, true, false}, full{false, false, true, true},
        blendWrite{true, true, true, false};
    taa::DrawClassifier cl;
    CHECK(cl.classify(glass) == DrawClass::Scene);   // transparente antes de lo opaco
    CHECK(cl.classify(noTest) == DrawClass::Other);  // fondo antes de lo opaco
    CHECK(!cl.sawOpaque());
    CHECK(cl.classify(opaque) == DrawClass::Scene && cl.sawOpaque());
    CHECK(cl.classify(sky) == DrawClass::Scene);     // test sin mezcla ni escritura
    CHECK(cl.classify(glass) == DrawClass::Ui);      // mezcla + test, sin escritura, tras lo opaco
    CHECK(cl.classify(noTest) == DrawClass::Ui);
    CHECK(cl.classify(noTestBlend) == DrawClass::Ui);
    CHECK(cl.classify(blendWrite) == DrawClass::Scene);  // escribe profundidad: geometría
    CHECK(cl.classify(full) == DrawClass::Other);        // postproceso / fundido de pantalla
    cl.beginPass(true);                                  // overlay que carga la profundidad hereda
    CHECK(cl.classify(glass) == DrawClass::Ui);
    cl.beginPass(false);                                 // escena nueva: olvida
    CHECK(!cl.sawOpaque() && cl.classify(glass) == DrawClass::Scene);
  }
  // 2. Máscara: alfa -> cobertura y mezcla (réplica del MSL).
  CHECK(near(taa::uiMaskFromAlpha(0), 0) && near(taa::uiMaskFromAlpha(0.25f), 0.5f) && near(taa::uiMaskFromAlpha(0.5f), 1) &&
        near(taa::uiMaskFromAlpha(1), 1) && near(taa::uiMaskFromAlpha(7), 1) && near(taa::uiMaskFromAlpha(-1), 0));
  CHECK(near(taa::maskedHistoryWeight(0.9f, 0), 0.9f) && near(taa::maskedHistoryWeight(0.9f, 1), 0) &&
        near(taa::maskedHistoryWeight(0.9f, 0.5f), 0.45f));
  CHECK(near(taa::maskedHistoryWeight(0.9f, 0.5f / 255), 0.9f));  // ruido de cuantización = 0
  CHECK(near(taa::maskedBase(0.2f, 0.8f, 0), 0.2f) && near(taa::maskedBase(0.2f, 0.8f, 1), 0.8f) &&
        near(taa::maskedBase(0.2f, 0.8f, 0.5f), 0.5f));
  // Con m = 1 el resultado es el texel actual: ni historia ni reconstrucción (texto nítido).
  {
    const float rec = 0.3f, cur = 0.9f, hist = 0.1f, keep = 0.9f, m = 1;
    const float base = taa::maskedBase(rec, cur, m), w = taa::maskedHistoryWeight(keep, m);
    CHECK(near(base + (hist - base) * w, cur));
  }

  // 2b. RCAS (réplica del kernel vpktaa_native_rcas): identidad en plano y con intensidad 0, sin rebasar
  // 0..1 ni el rango local (sin ringing), más contraste con más intensidad, stops como en FSR.
  {
    using P = std::array<float, 3>;
    auto cross = [](P b, P d, P e, P f, P h) { return std::array<P, 5>{b, d, e, f, h}; };
    const P g{0.4f, 0.4f, 0.4f};
    const auto flat = taa::rcasPixel(cross(g, g, g, g, g), 1.0f, false);
    CHECK(near(flat[0], 0.4f) && near(flat[1], 0.4f) && near(flat[2], 0.4f));
    // Stops: exp2(-2·(1 - intensidad)).
    CHECK(near(taa::rcasConstant(1.0f), 1.0f) && near(taa::rcasConstant(0.5f), 0.5f) && near(taa::rcasConstant(0.0f), 0.25f));
    CHECK(near(taa::rcasConstant(5.0f), 1.0f) && near(taa::rcasConstant(std::nanf("")), 0.25f));
    // Centro algo más claro que su cruz: sube; más intensidad, más; 0: identidad exacta; nunca > 1.
    const P lo{0.3f, 0.3f, 0.3f}, mid{0.45f, 0.45f, 0.45f};
    const auto full = taa::rcasPixel(cross(lo, lo, mid, lo, lo), 1.0f, false);
    const auto half = taa::rcasPixel(cross(lo, lo, mid, lo, lo), 0.5f, false);
    const auto off = taa::rcasPixel(cross(lo, lo, mid, lo, lo), 0.0f, false);
    CHECK(full[0] > half[0] && half[0] > 0.45f && near(off[0], 0.45f));
    CHECK(full[0] <= 1.0f && full[1] == full[0]);
    // Cruz negra con centro medio: lóbulo -0.125, el resultado sube sin rebasar 1.
    const P z{0.0f, 0.0f, 0.0f}, c5{0.5f, 0.5f, 0.5f};
    const auto lim = taa::rcasPixel(cross(z, z, c5, z, z), 1.0f, false);
    CHECK(lim[0] >= 0.5f && lim[0] <= 1.0f);
    // Lóbulo acotado a -0.1875 (centro .45 sobre cruz .3: -0.196 sin acotar): .9 exacto.
    CHECK(near(full[0], 0.9f, 1e-4f));
    // Sin ringing: nunca por debajo de 0 ni por encima de 1 en picos extremos.
    const P wh{1.0f, 1.0f, 1.0f};
    for (const auto& px : {taa::rcasPixel(cross(wh, wh, z, wh, wh), 1.0f, false), taa::rcasPixel(cross(z, z, wh, z, z), 1.0f, false)})
      CHECK(px[0] >= 0.0f && px[0] <= 1.0f);
    // Negro y blanco puros (e = mn o mx): sin lóbulo.
    CHECK(near(taa::rcasPixel(cross(z, z, z, z, z), 1.0f, false)[0], 0.0f) && near(taa::rcasPixel(cross(wh, wh, wh, wh, wh), 1.0f, false)[0], 1.0f));
    // Denoise: atenúa el lóbulo (queda entre el original y el afilado sin denoise).
    const auto dn = taa::rcasPixel(cross(lo, lo, mid, lo, lo), 1.0f, true);
    CHECK(dn[0] < full[0] && dn[0] > 0.45f);
  }

  // 3. Traductor sin dispositivo: UI sin jitter, contadores por clase.
  vpvk_taa_jitter in{1, 1920, 1824, 0, 0, 3};
  vpvk_taa_set_jitter(&in);
  vpvk_config_set_taa_native(1);
  CHECK(taaUiMaskOn());
  const TaaJitter j = taaJitter();
  const auto h0 = taa::haltonJitter(taaNativeTake().phase);  // la fase que vale ahora
  {
    Recorder rec;
    OpBeginRendering scene;
    scene.area = {{0, 0}, {1920, 1824}};
    scene.depthWidth = 1920, scene.depthHeight = 1824;
    scene.desc.depth.load = vpgpu::LoadAction::Clear;
    OpBeginRendering overlay = scene;  // UI en un pase aparte que carga la profundidad
    overlay.desc.depth.load = vpgpu::LoadAction::Load;
    OpDraw geo;
    geo.draw.count = 300, geo.draw.index = vpgpu::IndexBuffer{};
    geo.draw.viewport = {0, 0, 1920, 1824, 0, 1};
    geo.depthTested = geo.vertexBuffers = true;
    OpDraw opaque = geo, sky = geo, glass = geo, hud = geo, fade = geo;
    opaque.depthWrite = true;
    glass.blended = true;                   // texto / panel de world-space
    hud.depthTested = false;                // 2D sin test de profundidad
    fade.draw.count = 3, fade.draw.index.reset(), fade.vertexBuffers = false, fade.depthTested = false, fade.blended = true;
    rec.ops.emplace_back(scene);
    for (const OpDraw* d : {&opaque, &sky, &glass, &hud, &fade}) rec.ops.emplace_back(*d);
    rec.ops.emplace_back(OpEndPass{});
    rec.ops.emplace_back(overlay), rec.ops.emplace_back(glass), rec.ops.emplace_back(OpEndPass{});
    vpgpu::Frame f;
    TranslateStats st;
    translate(rec, f, st, &j);
    std::vector<vpgpu::Viewport> v;
    for (const auto& c : f.commands())
      if (const auto* rp = std::get_if<vpgpu::RenderPass>(&c))
        for (const auto& d : rp->draws) v.push_back(d.viewport);
    CHECK(v.size() == 6);
    if (v.size() == 6) {
      CHECK(v[0].x == h0[0] && v[0].y == h0[1]);  // opaco: jitter
      CHECK(v[1].x == h0[0]);                     // cielo: jitter
      CHECK(v[2].x == 0 && v[2].y == 0);          // cristal / texto: UI, sin jitter
      CHECK(v[3].x == 0 && v[3].y == 0);          // 2D sin test: sin jitter
      CHECK(v[4].x == 0 && v[4].y == 0);          // pantalla completa: sin jitter
      CHECK(v[5].x == 0 && v[5].y == 0);          // overlay heredó «ya hubo opaco»: UI
    }
    CHECK(st.ui.scene == 2 && st.ui.ui == 3 && st.ui.other == 1 && st.ui.passes == 2 && st.ui.maskedPasses == 0);
    CHECK(st.taaJittered == 2);
  }
  // Puerta: con el nativo apagado nada cambia (el cristal sigue moviéndose como antes).
  vpvk_config_set_taa_native(0);
  CHECK(!taaUiMaskOn());
  {
    Recorder rec;
    OpBeginRendering scene;
    scene.area = {{0, 0}, {1920, 1824}};
    scene.depthWidth = 1920, scene.depthHeight = 1824;
    OpDraw geo;
    geo.draw.count = 300, geo.draw.index = vpgpu::IndexBuffer{};
    geo.draw.viewport = {0, 0, 1920, 1824, 0, 1};
    geo.depthTested = geo.vertexBuffers = true;
    OpDraw opaque = geo, glass = geo;
    opaque.depthWrite = true, glass.blended = true;
    rec.ops.emplace_back(scene), rec.ops.emplace_back(opaque), rec.ops.emplace_back(glass), rec.ops.emplace_back(OpEndPass{});
    vpgpu::Frame f;
    TranslateStats st;
    TaaJitter old = j;
    old.jx = 0.25f, old.jy = 0.125f;
    translate(rec, f, st, &old);
    CHECK(st.taaJittered == 2 && st.ui.passes == 0 && st.ui.ui == 0);
  }

  // 4. Con un dispositivo (backend nulo): attachment R8 y variantes de pipeline.
  vpvk_config_set_taa_native(1);
  in = {1, 64, 64, 0, 0, 4};
  vpvk_taa_set_jitter(&in);
  const TaaJitter j64 = taaJitter();
  char rtmpl[] = "/tmp/vpvk_uimask_XXXXXX";
  const char* rdir = mkdtemp(rtmpl);
  CHECK(rdir != nullptr);
  if (rdir) setenv("VP_PIPELINE_CACHE_DIR", rdir, 1);
  {
    Ctx c;
    Device* dev = from<Device>(c.dev);
    dev->uiMask.dev = dev;
    VkDeviceMemory m = alloc(c, 64 << 20, 0);
    VkImage color = image(c, VK_FORMAT_R8G8B8A8_UNORM, 64, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, m);
    VkImage depth = image(c, VK_FORMAT_D32_SFLOAT, 64, 1, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, m);
    VkImageView views[2];
    for (int i = 0; i < 2; ++i) {
      VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      vi.image = i ? depth : color, vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
      vi.format = i ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
      vi.subresourceRange = {VkImageAspectFlags(i ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 1, 0, 1};
      vkCreateImageView(c.dev, &vi, nullptr, &views[i]);
    }
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout pl;
    vkCreatePipelineLayout(c.dev, &plci, nullptr, &pl);
    VkAttachmentDescription att[2]{};
    att[0].format = VK_FORMAT_R8G8B8A8_UNORM, att[0].samples = VK_SAMPLE_COUNT_1_BIT;
    att[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, att[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[1].format = VK_FORMAT_D32_SFLOAT, att[1].samples = VK_SAMPLE_COUNT_1_BIT;
    att[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, att[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, att[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    VkAttachmentReference cref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
        dref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sp{};
    sp.colorAttachmentCount = 1, sp.pColorAttachments = &cref, sp.pDepthStencilAttachment = &dref;
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = 2, rpi.pAttachments = att, rpi.subpassCount = 1, rpi.pSubpasses = &sp;
    VkRenderPass rp;
    vkCreateRenderPass(c.dev, &rpi, nullptr, &rp);
    VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fi.renderPass = rp, fi.attachmentCount = 2, fi.pAttachments = views, fi.width = 64, fi.height = 64, fi.layers = 1;
    VkFramebuffer fb;
    vkCreateFramebuffer(c.dev, &fi, nullptr, &fb);

    VkShaderModule vs = module(c, "uimask.vert.spv"), fs = module(c, "uimask.frag.spv");
    VkPipelineShaderStageCreateInfo stg[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                           {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    stg[0].stage = VK_SHADER_STAGE_VERTEX_BIT, stg[0].module = vs, stg[0].pName = "main";
    stg[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT, stg[1].module = fs, stg[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vis{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ias{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ias.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkViewport vp{0, 0, 64, 64, 0, 1};
    VkRect2D sc{{0, 0}, {64, 64}};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1, vps.pViewports = &vp, vps.scissorCount = 1, vps.pScissors = &sc;
    auto pipeline = [&](bool depthWrite, bool blend) {
      VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
      ds.depthTestEnable = VK_TRUE, ds.depthWriteEnable = depthWrite, ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
      VkPipelineColorBlendAttachmentState cba{};
      cba.colorWriteMask = 0xF, cba.blendEnable = blend;
      cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA, cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE, cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
      cbs.attachmentCount = 1, cbs.pAttachments = &cba;
      VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
      gp.stageCount = 2, gp.pStages = stg, gp.pVertexInputState = &vis, gp.pInputAssemblyState = &ias;
      gp.pViewportState = &vps, gp.pRasterizationState = &rs, gp.pMultisampleState = &ms, gp.pColorBlendState = &cbs;
      gp.pDepthStencilState = &ds, gp.layout = pl, gp.renderPass = rp;
      VkPipeline p = VK_NULL_HANDLE;
      CHECK(vkCreateGraphicsPipelines(c.dev, VK_NULL_HANDLE, 1, &gp, nullptr, &p) == VK_SUCCESS);
      return p;
    };
    VkPipeline opaquePipe = pipeline(true, false), glassPipe = pipeline(false, true);
    const PipelineObj* po = opaquePipe ? from<PipelineObj>(opaquePipe) : nullptr;
    const PipelineObj* pg = glassPipe ? from<PipelineObj>(glassPipe) : nullptr;
    CHECK(po && po->ok && pg && pg->ok && !pg->sources.empty());  // los fuentes se conservan para la variante de UI
    if (po && po->ok && pg && pg->ok) {
      VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
      VkCommandPool cp;
      vkCreateCommandPool(c.dev, &cpi, nullptr, &cp);
      VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      ai.commandPool = cp, ai.commandBufferCount = 2;
      VkCommandBuffer cbs[2];
      vkAllocateCommandBuffers(c.dev, &ai, cbs);
      for (int k = 0; k < 2; ++k) {  // 0: escena + cristal; 1: solo escena
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cbs[k], &cbi);
        VkClearValue clears[2]{};
        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass = rp, rbi.framebuffer = fb, rbi.renderArea = {{0, 0}, {64, 64}}, rbi.clearValueCount = 2;
        rbi.pClearValues = clears;
        vkCmdBeginRenderPass(cbs[k], &rbi, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cbs[k], VK_PIPELINE_BIND_POINT_GRAPHICS, opaquePipe);
        vkCmdDraw(cbs[k], 300, 1, 0, 0);
        if (k == 0) {
          vkCmdBindPipeline(cbs[k], VK_PIPELINE_BIND_POINT_GRAPHICS, glassPipe);
          vkCmdDraw(cbs[k], 300, 1, 0, 0);
        }
        vkCmdEndRenderPass(cbs[k]);
        vkEndCommandBuffer(cbs[k]);
      }
      const Recorder& r0 = *from<CommandBuffer>(cbs[0])->rec;
      const OpDraw* d0 = nullptr;
      const OpDraw* d1 = nullptr;
      for (const Op& op : r0.ops)
        if (const auto* d = std::get_if<OpDraw>(&op)) (d0 ? d1 : d0) = d;
      CHECK(d0 && d1 && d0->depthWrite && !d0->blended && !d1->depthWrite && d1->blended && d1->depthTested);

      vpgpu::Frame f;
      TranslateStats st;
      translate(r0, f, st, &j64, &dev->uiMask);
      const auto* pass = f.commands().empty() ? nullptr : std::get_if<vpgpu::RenderPass>(&f.commands()[0]);
      CHECK(pass && pass->draws.size() == 2);
      CHECK(st.ui.scene == 1 && st.ui.ui == 1 && st.ui.passes == 1 && st.ui.maskedPasses == 1);
      if (pass && pass->draws.size() == 2) {
        const vpgpu::Attachment& mk = pass->desc.colors[taa::kUiMaskSlot];
        CHECK(mk.texture && mk.load == vpgpu::LoadAction::Clear && mk.store == vpgpu::StoreAction::Store);
        CHECK(pass->desc.colors[0].texture && !pass->desc.colors[1].texture);
        CHECK(pass->draws[0].pipeline != po->key && pass->draws[1].pipeline != pg->key);
        CHECK(pass->draws[0].viewport.x == j64.jx && pass->draws[1].viewport.x == 0);  // el UI no se mueve
        CHECK(po->maskVariants.size() == 1 && !po->maskVariants[0].ui && po->maskVariants[0].key == pass->draws[0].pipeline);
        CHECK(pg->maskVariants.size() == 1 && pg->maskVariants[0].ui && pg->maskVariants[0].key == pass->draws[1].pipeline);
        CHECK(pg->maskVariants[0].fragment && pg->maskVariants[0].fragment != pg->state.fragmentShader);  // otro MSL
        CHECK(po->maskVariants[0].fragment == po->state.fragmentShader);  // el 3D no cambia de shader
        const UiMaskState::Live& l = dev->uiMask.live[0];
        CHECK(l.valid && l.t.tex == mk.texture && l.t.w == 64 && l.t.h == 64 && l.t.layers == 1);
        c.gpu().pipelines().waitIdle();
        CHECK(c.gpu().pipelines().status(pass->draws[1].pipeline) == vpgpu::PipelineCache::Status::Ready);
      }
      // Segundo envío: las mismas variantes y la misma textura, sin recrear nada.
      {
        vpgpu::Frame f2;
        TranslateStats st2;
        const vpgpu::TextureId first = dev->uiMask.target[0].tex;
        dev->uiMask.maskedPasses = 0;  // lo que hace vpvk_taa_native_encode al tomar la máscara
        translate(r0, f2, st2, &j64, &dev->uiMask);
        CHECK(po->maskVariants.size() == 1 && pg->maskVariants.size() == 1 && dev->uiMask.target[0].tex == first);
      }
      // Un frame sin UI: ni attachment ni variantes (cero coste).
      {
        vpgpu::Frame f3;
        TranslateStats st3;
        translate(*from<CommandBuffer>(cbs[1])->rec, f3, st3, &j64, &dev->uiMask);
        const auto* p3 = f3.commands().empty() ? nullptr : std::get_if<vpgpu::RenderPass>(&f3.commands()[0]);
        CHECK(p3 && p3->draws.size() == 1 && !p3->desc.colors[taa::kUiMaskSlot].texture);
        CHECK(p3 && p3->draws[0].pipeline == po->key && st3.ui.passes == 1 && st3.ui.maskedPasses == 0);
      }
      // Pase de ojo con MSAA del juego (4x): la máscara es un attachment multisample transitorio que se
      // resuelve (Resolve) en la R8 de 1 muestra; el 1x de arriba no cambia.
      {
        vpgpu::TextureId col, dep;
        col.index = 11, col.generation = 1, dep.index = 12, dep.generation = 1;
        OpDraw geo;
        geo.draw.count = 300, geo.draw.index = vpgpu::IndexBuffer{};
        geo.draw.viewport = {0, 0, 64, 64, 0, 1};
        geo.depthTested = geo.vertexBuffers = true;
        geo.passSamples = 4;
        OpDraw opq = geo, gls = geo;
        opq.depthWrite = true, gls.blended = true;
        Recorder rec;
        OpBeginRendering scene;
        scene.area = {{0, 0}, {64, 64}};
        scene.depthWidth = 64, scene.depthHeight = 64;
        scene.desc.colors[0].texture = col;
        scene.desc.depth.texture = dep, scene.desc.depth.load = vpgpu::LoadAction::Clear;
        rec.ops.emplace_back(scene), rec.ops.emplace_back(opq), rec.ops.emplace_back(gls), rec.ops.emplace_back(OpEndPass{});
        vpgpu::Frame f4;
        TranslateStats st4;
        dev->uiMask.maskedPasses = 0;
        translate(rec, f4, st4, &j64, &dev->uiMask);
        const auto* p4 = f4.commands().empty() ? nullptr : std::get_if<vpgpu::RenderPass>(&f4.commands()[0]);
        CHECK(p4 && st4.ui.ui == 1 && st4.ui.maskedPasses == 1);
        if (p4) {
          const vpgpu::Attachment& mk = p4->desc.colors[taa::kUiMaskSlot];
          const UiMaskState::Target& t = dev->uiMask.target[0];
          CHECK(mk.texture && mk.texture == t.ms && t.ms != t.tex && t.msSamples == 4);
          CHECK(mk.load == vpgpu::LoadAction::Clear && mk.store == vpgpu::StoreAction::Resolve && mk.resolve == t.tex);
          CHECK(dev->uiMask.live[0].valid && dev->uiMask.live[0].t.tex == t.tex);
        }
      }
      vkDestroyCommandPool(c.dev, cp, nullptr);
    }
    if (opaquePipe) vkDestroyPipeline(c.dev, opaquePipe, nullptr);
    if (glassPipe) vkDestroyPipeline(c.dev, glassPipe, nullptr);
    vkDestroyShaderModule(c.dev, vs, nullptr);
    vkDestroyShaderModule(c.dev, fs, nullptr);
    vkDestroyFramebuffer(c.dev, fb, nullptr);
    vkDestroyRenderPass(c.dev, rp, nullptr);
    vkDestroyPipelineLayout(c.dev, pl, nullptr);
    for (VkImageView v : views) vkDestroyImageView(c.dev, v, nullptr);
    vkDestroyImage(c.dev, color, nullptr);
    vkDestroyImage(c.dev, depth, nullptr);
    vkFreeMemory(c.dev, m, nullptr);
  }
  // Las variantes quedan en las recetas con su opción de máscara: se cargan igual y, con otro
  // traductor, se retraducen a partir del SPIR-V (la opción viaja en la entrada).
  if (rdir) {
    RecipeStore s(rdir);
    int masked = 0, alpha = 0;
    for (const StageInput& si : s.stageInputs()) {
      masked += si.opt.uiMaskSlot == taa::kUiMaskSlot;
      alpha += si.opt.uiMaskSlot == taa::kUiMaskSlot && si.opt.uiMaskAlpha;
    }
    CHECK(masked == 1 && alpha == 1);  // el fragment de UI con alfa (mezcla SrcAlpha)
    StageInput plain = s.stageInputs()[0];
    plain.opt.uiMaskSlot = -1, plain.opt.uiMaskAlpha = false;  // independent of stageInputs() order
    StageInput withMask = plain;
    withMask.opt.uiMaskSlot = 7;
    StageInput withAlpha = withMask;
    withAlpha.opt.uiMaskAlpha = true;
    CHECK(plain.hash() != withMask.hash() && withMask.hash() != withAlpha.hash());
    RecipeStore t(rdir, kTranslatorVersion + 1);
    CHECK(t.needsRefresh());
    const RecipeStore::RefreshResult r = t.refresh(2);
    CHECK(r.failed == 0 && r.droppedPipelines == 0 && r.retranslated >= 1);
    std::error_code ec;
    std::filesystem::remove_all(rdir, ec);
  }
  unsetenv("VP_PIPELINE_CACHE_DIR");
  vpvk_config_set_taa_native(0);
  in.enabled = 0;
  vpvk_taa_set_jitter(&in);
}

// AOT build (VPVK_AOT): gate off does nothing; gate on translates a dir of SPIR-V, builds one
// compute pipeline per compute module and writes aot.manifest; the gpu.shader.lib index is written.
void testAot() {
  char tmpl[] = "/tmp/vpvk_aot_XXXXXX";
  const char* dir = mkdtemp(tmpl);
  CHECK(dir != nullptr);
  if (!dir) return;
  const std::string spv = std::string(dir) + "/spirv";
  std::filesystem::create_directories(spv);
  for (const char* n : {"array_length.comp.spv", "dyn_buffer.comp.spv", "clip.vert.spv"})
    std::filesystem::copy_file(std::string(VPVK_SPIRV_TESTS) + "/" + n, spv + "/" + n);
  setenv("VP_PIPELINE_CACHE_DIR", dir, 1);
  unsetenv("VPVK_AOT");
  CHECK(vpvk_aot_build(spv.c_str(), nullptr, nullptr) == 0);
  CHECK(!std::filesystem::exists(std::string(dir) + "/aot.manifest"));
  setenv("VPVK_AOT", "1", 1);
  uint32_t calls = 0;
  const uint32_t n = vpvk_aot_build(spv.c_str(), [](uint32_t, uint32_t, void* u) { ++*static_cast<uint32_t*>(u); }, &calls);
  unsetenv("VPVK_AOT");
  CHECK(n >= 3 + 2);  // >= 3 libraries + 2 compute pipelines
  CHECK(calls > 0);
  std::string m;
  if (FILE* f = std::fopen((std::string(dir) + "/aot.manifest").c_str(), "rb")) {
    char buf[512] = {0};
    m.assign(buf, std::fread(buf, 1, sizeof buf - 1, f));
    std::fclose(f);
  }
  CHECK(m.find("compute=2") != std::string::npos);
  CHECK(m.find("failed=0") != std::string::npos);
  CHECK(std::filesystem::file_size(std::string(dir) + "/msl_index.bin") >= 3 * 8);
  unsetenv("VP_PIPELINE_CACHE_DIR");
  std::filesystem::remove_all(dir);
}

void testAniso() {
  // VPVK_FORCE_ANISO (per game) wins over VPVK_ANISO and the API setter; read once, so set first.
  setenv("VPVK_ANISO", "2", 1);
  setenv("VPVK_FORCE_ANISO", "32", 1);  // above Adreno's 16: clamped
  CHECK(vpvk::anisotropyForced() == 16);
  vpvk_config_set_anisotropy(4);
  CHECK(vpvk::anisotropyForced() == 16);

  VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  ci.minFilter = ci.magFilter = VK_FILTER_LINEAR;
  ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  ci.addressModeU = ci.addressModeV = ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  ci.maxLod = 12.0f;
  CHECK(vpvk::samplerAnisoEligible(ci));
  auto with = [&](auto mod) {
    VkSamplerCreateInfo c = ci;
    mod(c);
    return vpvk::samplerAnisoEligible(c);
  };
  CHECK(!with([](auto& c) { c.minFilter = VK_FILTER_NEAREST; }));
  CHECK(!with([](auto& c) { c.magFilter = VK_FILTER_NEAREST; }));
  CHECK(!with([](auto& c) { c.compareEnable = VK_TRUE; }));
  CHECK(!with([](auto& c) { c.unnormalizedCoordinates = VK_TRUE; }));
  CHECK(!with([](auto& c) { c.maxLod = 0.0f; }));
  CHECK(!with([](auto& c) { c.addressModeU = c.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; }));
  CHECK(with([](auto& c) { c.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; }));  // V still wraps
}

// Archive key index (gpu/metal2 archive_index.h): round trip, atomic file, rejection of foreign files.
void testArchiveIndex() {
  namespace ai = vpgpu::metal2::archive_index;
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "vpvk_archive_index_test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const fs::path ar = dir / "pipelines-7.metalar";
  CHECK(ai::keysPath(ar) == dir / "pipelines-7.keys");
  std::vector<uint64_t> in = {1, 0xdeadbeefcafef00dull, ~0ull}, out;
  CHECK(ai::write(ai::keysPath(ar), in));
  CHECK(ai::read(ai::keysPath(ar), out) && out == in);
  CHECK(!fs::exists(dir / "pipelines-7.keys.tmp"));
  CHECK(ai::write(ai::keysPath(ar), {}) && ai::read(ai::keysPath(ar), out) && out.empty());
  CHECK(!ai::read(dir / "missing.keys", out));
  {  // truncated, wrong magic, trailing bytes -> treated as a legacy file
    CHECK(ai::write(ai::keysPath(ar), in));
    fs::resize_file(ai::keysPath(ar), fs::file_size(ai::keysPath(ar)) - 3);
    CHECK(!ai::read(ai::keysPath(ar), out) && out.empty());
    CHECK(ai::write(ai::keysPath(ar), in));
    if (FILE* f = std::fopen(ai::keysPath(ar).c_str(), "ab")) { std::fputc(0, f); std::fclose(f); }
    CHECK(!ai::read(ai::keysPath(ar), out));
    if (FILE* f = std::fopen(ai::keysPath(ar).c_str(), "wb")) { uint64_t bad[2] = {1, 0}; std::fwrite(bad, 8, 2, f); std::fclose(f); }
    CHECK(!ai::read(ai::keysPath(ar), out));
  }
  fs::remove_all(dir);
}

// New images read as zeros until first written (Android's zeroed memory): bind marks the texture
// pending; the submit's frame decides which pending images need a clear pass (null backend: the
// frame the clears go in is inspected).
void testZeroInit() {
  CHECK(vpvk::zeroNewImagesOn());  // default on
  Ctx c;
  Device* d = from<Device>(c.dev);
  VkDeviceMemory m = alloc(c, 64 << 20, 1);  // images get memory of their own: same range + description would alias
  VkBuffer buf = buffer(c, 1 << 20, m, 32 << 20);
  const VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  auto key = [&](VkImage i) {
    const vpgpu::TextureId t = from<Image>(i)->id;
    return uint64_t{t.generation} << 32 | t.index;
  };
  VkImage a = image(c, VK_FORMAT_R8G8B8A8_UNORM, 16, 1, usage, alloc(c, 1 << 20, 1));  // read first
  VkImage b = image(c, VK_FORMAT_R8G8B8A8_UNORM, 16, 1, usage, alloc(c, 1 << 20, 1));  // fully uploaded, then read
  VkImage p = image(c, VK_FORMAT_R8G8B8A8_UNORM, 16, 1, usage, alloc(c, 1 << 20, 1));  // partly uploaded
  VkImage u = image(c, VK_FORMAT_R8G8B8A8_UNORM, 16, 1, usage, alloc(c, 1 << 20, 1));  // untouched
  VkImage z = image(c, VK_FORMAT_D32_SFLOAT, 16, 1, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, alloc(c, 1 << 20, 1));
  VkImage tr = image(c, VK_FORMAT_R8G8B8A8_UNORM, 16, 1,
                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT, alloc(c, 1 << 20, 1));
  // Not renderable (no transfer destination / compressed): zeroed by a copy from a buffer of zeros.
  VkImage so = image(c, VK_FORMAT_R8G8B8A8_UNORM, 16, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, alloc(c, 1 << 20, 1));
  VkImage bc = image(c, VK_FORMAT_ASTC_4x4_UNORM_BLOCK, 16, 1, usage, alloc(c, 1 << 20, 1));
  CHECK(a && b && p && u && z && tr && so && bc);
  CHECK(d->zeroPending.size() == 7 && d->zeroPending.at(key(so)).viaCopy && d->zeroPending.at(key(bc)).viaCopy &&
        !d->zeroPending.at(key(a)).viaCopy);
  CHECK( d->zeroPending.count(key(a)) && d->zeroPending.count(key(z)));
  CHECK(!d->zeroPending.count(key(tr)));  // transient: memoryless, nothing to clear
  // vpvk_mark_eye_image: bound image -> its texture is an eye texture; the mark dies with the texture.
  CHECK(vpvk_mark_eye_image(nullptr) == 0);
  CHECK(!isEyeTexture(*d, from<Image>(a)->id) && vpvk_mark_eye_image(a) == 1);
  CHECK(isEyeTexture(*d, from<Image>(a)->id) && !isEyeTexture(*d, from<Image>(u)->id));
  {
    const vpgpu::TextureId ta = from<Image>(a)->id;
    VkImage e2 = image(c, VK_FORMAT_R8G8B8A8_UNORM, 16, 1, usage, alloc(c, 1 << 20, 1));
    CHECK(vpvk_mark_eye_image(e2) == 1);
    const vpgpu::TextureId te = from<Image>(e2)->id;
    CHECK(isEyeTexture(*d, te));
    vkDestroyImage(c.dev, e2, nullptr);
    CHECK(!isEyeTexture(*d, te) && isEyeTexture(*d, ta));
  }

  VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  VkCommandPool pool;
  vkCreateCommandPool(c.dev, &pi, nullptr, &pool);
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = pool;
  ai.commandBufferCount = 1;
  VkCommandBuffer cb;
  vkAllocateCommandBuffers(c.dev, &ai, &cb);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(cb, &bi);
  VkBufferImageCopy full{};
  full.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  full.imageExtent = {16, 16, 1};
  VkBufferImageCopy part = full;
  part.imageExtent = {8, 8, 1};
  vkCmdCopyImageToBuffer(cb, a, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &full);
  vkCmdCopyBufferToImage(cb, buf, b, VK_IMAGE_LAYOUT_GENERAL, 1, &full);
  vkCmdCopyImageToBuffer(cb, b, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &full);
  vkCmdCopyBufferToImage(cb, buf, p, VK_IMAGE_LAYOUT_GENERAL, 1, &part);
  vkCmdCopyImageToBuffer(cb, so, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &full);
  vkCmdCopyImageToBuffer(cb, bc, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &full);
  vkEndCommandBuffer(cb);

  vpgpu::Frame frame, pre;
  TranslateStats st;
  ZeroLog zl;
  zl.dev = d;
  translate(*from<CommandBuffer>(cb)->rec, frame, st, nullptr, nullptr, &zl);
  CHECK(zeroInitResolve(*d, zl, pre) == 4);  // a, so, bc (read first) and p (partial); b was overwritten whole
  bool clearsA = false, clearsP = false, copiesSo = false, copiesBc = false, fillsZeros = false;
  uint32_t passes = 0;
  for (const vpgpu::Command& cmd : pre.commands())
    if (const auto* rp = std::get_if<vpgpu::RenderPass>(&cmd)) {
      ++passes;
      const vpgpu::Attachment& at = rp->desc.colors[0];
      CHECK(at.load == vpgpu::LoadAction::Clear && at.clear.color == (std::array<double, 4>{}) && rp->draws.empty());
      clearsA = clearsA || at.texture == from<Image>(a)->id;
      clearsP = clearsP || at.texture == from<Image>(p)->id;
    }
    else if (const auto* cp = std::get_if<vpgpu::Copy>(&cmd)) {
      if (cp->kind == vpgpu::CopyKind::FillBuffer) fillsZeros = cp->fillByte == 0 && cp->dstBuffer.buffer == d->zeroBuffer;
      if (cp->kind != vpgpu::CopyKind::BufferToTexture) continue;
      CHECK(cp->srcBuffer.buffer == d->zeroBuffer && cp->dstTexture.width == 16 && cp->dstTexture.height == 16);
      if (cp->dstTexture.texture == from<Image>(so)->id) {  // RGBA8: 16 texels x 4 bytes, 16 rows
        copiesSo = true;
        CHECK(cp->srcBuffer.bytesPerRow == 64 && cp->srcBuffer.rowsPerImage == 16);
      }
      if (cp->dstTexture.texture == from<Image>(bc)->id) {  // ASTC 4x4: 4 blocks x 16 bytes, 4 block rows
        copiesBc = true;
        CHECK(cp->srcBuffer.bytesPerRow == 64 && cp->srcBuffer.rowsPerImage == 4);
      }
    }
  CHECK(clearsA && clearsP && copiesSo && copiesBc && fillsZeros && passes == 2);
  CHECK(d->zeroPending.size() == 2 && d->zeroPending.count(key(u)) && d->zeroPending.count(key(z)));  // untouched ones wait

  // The same through vkQueueSubmit: reading the untouched colour image queues its clear.
  vkResetCommandBuffer(cb, 0);
  vkBeginCommandBuffer(cb, &bi);
  vkCmdCopyImageToBuffer(cb, u, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &full);
  vkEndCommandBuffer(cb);
  const uint64_t before = d->zeroCleared;
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb;
  CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS);
  CHECK(d->zeroCleared == before + 1 && d->zeroPending.size() == 1 && !d->zeroPending.count(key(u)));

  vkDestroyImage(c.dev, z, nullptr);  // destroying a pending image forgets it
  CHECK(d->zeroPending.empty());
  vkDestroyCommandPool(c.dev, pool, nullptr);
}

// Foveado propio: matemática del perfil, interruptor, pases de ojo con mapa en el traductor, separación de la UI.
void testFoveation() {
  auto near = [](float a, float b, float eps = 1e-5f) { return std::fabs(a - b) <= eps; };
  // 1. Perfil: 1 dentro de la zona nítida, periferia en el borde, monótono, asimétrico con tangentes desiguales.
  {
    float q[fov::kZones];
    fov::axisQuality(q, fov::kZones, 0.5f, 0.4f, 0.3f);
    CHECK(near(q[fov::kZones / 2], 1.0f) && near(q[fov::kZones / 2 - 1], 1.0f));
    CHECK(q[0] < 0.35f && q[fov::kZones - 1] < 0.35f && near(q[0], q[fov::kZones - 1], 1e-4f));
    for (uint32_t i = 1; i <= fov::kZones / 2; ++i) CHECK(q[i - 1] <= q[i] + 1e-6f);  // sube hasta el centro
    fov::axisQuality(q, fov::kZones, 0.25f, 0.4f, 0.3f);  // centro desplazado: cada lado normalizado a su borde
    CHECK(q[fov::kZones / 4] > 0.99f && q[fov::kZones - 1] < 0.35f);
    CHECK(near(fov::opticalCenter({1, 1, 1, 1}).x, 0.5f));
    CHECK(near(fov::opticalCenter({1.2f, 0.8f, 1, 1}).x, 0.6f) && near(fov::opticalCenter({1, 1, 0.9f, 1.1f}).y, 0.45f));
    CHECK(near(fov::opticalCenter({0, 0, 0, 0}).x, 0.5f));  // sin datos válidos: el centro
  }
  {
    CHECK(fov::shapeOf(0, 0.5f, 0.5f, 0).off && !fov::shapeOf(1, 0, 0, 0).off);
    const fov::Shape m2 = fov::shapeOf(1, 0.9f, 0.9f, 0), m5 = fov::shapeOf(2, 0.1f, 0.1f, 0);  // los presets ignoran los valores
    CHECK(near(m2.sharp, fov::presetM2().sharp) && near(m5.periphery, fov::presetM5().periphery));
    CHECK(m2.periphery < m5.periphery);  // M2 reduce más que M5
    // Valores de PTVisionPro: M2 60 grados / 10 %, M5 70 grados / 15 %, ambos 10 grados bajo el eje.
    CHECK(near(m2.periphery, 0.10f) && near(m5.periphery, 0.15f) && m2.sharp < m5.sharp && m2.height < 0 && near(m2.height, m5.height));
    CHECK(near(fov::shapeOf(1, 0, 0, 0.02f).height, m2.height + 0.02f));  // `height` ajusta sobre el del preset
    {
      float q[fov::kZones];  // caída de PTVisionPro: 1 hasta sharp*(1-fade), periferia más allá de sharp
      fov::axisQuality(q, fov::kZones, 0.5f, 0.6f, 0.2f, 0.15f);
      CHECK(near(q[fov::kZones / 2], 1.0f) && near(q[0], 0.2f) && near(q[fov::kZones - 1], 0.2f));
    }
    const fov::Shape cu = fov::shapeOf(3, 0.6f, 0.7f, 0.1f);
    CHECK(near(cu.sharp, 0.6f) && near(cu.periphery, 0.7f) && near(cu.height, 0.1f));
    CHECK(near(fov::shapeOf(3, 9, 9, 9).sharp, 1.0f) && near(fov::shapeOf(3, -1, -1, 0).periphery, 0.1f));  // acotado
    fov::Eyes eyes;
    eyes.known = true, eyes.tangents = {{{1.3f, 0.7f, 1, 1}, {0.7f, 1.3f, 1, 1}}};
    const auto m = fov::buildRateMap(1920, 1824, 2, fov::presetM2(), eyes);
    CHECK(m && m->layers == 2 && m->zonesX == fov::kZones && m->key != 0 && m->keptFraction > 0.2f && m->keptFraction < 1.0f);
    if (m) {
      CHECK(near(m->keptFraction, fov::keptFraction(*m)));
      // Ojos con centro distinto: la zona nítida no cae en la misma zona.
      const float* l0 = &m->qx[0];
      const float* l1 = &m->qx[fov::kZones];
      uint32_t a0 = 0, a1 = 0;
      for (uint32_t i = 0; i < fov::kZones; ++i) a0 += l0[i] > 0.999f ? 1 : 0, a1 += l1[i] > 0.999f ? 1 : 0;
      CHECK(a0 > 0 && a1 > 0 && !std::equal(l0, l0 + fov::kZones, l1));
      const auto same = fov::buildRateMap(1920, 1824, 2, fov::presetM2(), eyes);
      const auto other = fov::buildRateMap(1920, 1824, 2, fov::presetM5(), eyes);
      const auto size = fov::buildRateMap(1536, 1459, 2, fov::presetM2(), eyes);
      CHECK(same->key == m->key && other->key != m->key && size->key != m->key);
    }
    CHECK(!fov::buildRateMap(1920, 1824, 2, fov::shapeOf(0, 0, 0, 0), eyes) && !fov::buildRateMap(0, 10, 2, fov::presetM2(), eyes));
  }
  // 2. Core: el mapa forma parte de la identidad del pase (no se funden pases de distinto mapa).
  {
    vpgpu::RenderPassDesc a, b;
    a.colors[0].texture = {1, 1};
    b = a;
    CHECK(a.sameTargets(b));
    a.rateMap = fov::buildRateMap(64, 64, 2, fov::presetM2(), {});
    CHECK(!a.sameTargets(b));
    b.rateMap = fov::buildRateMap(64, 64, 2, fov::presetM2(), {});
    CHECK(a.sameTargets(b));  // mismo mapa (clave igual), objetos distintos
    b.rateMap = fov::buildRateMap(64, 64, 2, fov::presetM5(), {});
    CHECK(!a.sameTargets(b));
  }
  // 3. Interruptor: apagado por defecto; activo con el interruptor, un modo y un dispositivo con rate maps.
  CHECK(!foveationActive() && vpvk_foveation_active() == 0);
  vpvk_foveation fcfg{VPVK_FOVEATION_M2, 0.4f, 0.3f, 0.0f};
  vpvk_config_set_foveation(&fcfg);
  CHECK(!foveationActive());  // modo sin interruptor
  vpvk_config_set_foveation_apply(1);
  unsetenv("VPVK_NULL_RATEMAP");
  CHECK(!foveationActive());  // sin soporte de rate maps en el dispositivo
  setenv("VPVK_NULL_RATEMAP", "1", 1);
  CHECK(foveationActive() && vpvk_foveation_active() == 1);

  // 4. Traductor: pase de ojo estéreo con mapa; sin FoveationState / sin jitter / con vistas distintas, nada cambia.
  vpvk_taa_jitter in{1, 1920, 1824, 0, 0, 7};
  vpvk_taa_set_jitter(&in);
  vpvk_config_set_taa_native(1);
  const TaaJitter j = taaJitter();
  auto eye = [](bool areaClear) {
    OpBeginRendering b;
    b.area = {{0, 0}, {1920, 1824}};
    b.viewMask = 3, b.depthWidth = 1920, b.depthHeight = 1824;
    b.desc.layerCount = 2;
    b.desc.colors[0].texture = {1, 1};
    b.desc.colors[0].store = vpgpu::StoreAction::DontCare;
    b.desc.depth.texture = {2, 1};
    b.desc.depth.load = vpgpu::LoadAction::Clear;
    if (areaClear) {
      auto ac = std::make_shared<vpgpu::ClearRects>();
      ac->rects.push_back({0, 0, 100, 100, 0, 2});
      ac->colorMask = 1;
      b.areaClear = ac;
    }
    return b;
  };
  OpDraw geo;
  geo.draw.count = 300, geo.draw.index = vpgpu::IndexBuffer{};
  geo.draw.viewport = {0, 0, 1920, 1824, 0, 1};
  geo.depthTested = geo.vertexBuffers = geo.depthWrite = true;
  OpDraw glass = geo;
  glass.blended = true, glass.depthWrite = false;  // texto / panel: clase UI
  auto passes = [](const vpgpu::Frame& f) {
    std::vector<const vpgpu::RenderPass*> v;
    for (const auto& c : f.commands())
      if (const auto* rp = std::get_if<vpgpu::RenderPass>(&c)) v.push_back(rp);
    return v;
  };
  {
    Recorder rec;
    rec.ops.emplace_back(eye(false)), rec.ops.emplace_back(geo), rec.ops.emplace_back(OpEndPass{});
    FoveationState fs;
    vpgpu::Frame f;
    TranslateStats st;
    translate(rec, f, st, &j, nullptr, nullptr, &fs);
    const auto p = passes(f);
    CHECK(p.size() == 1 && p[0]->desc.rateMap && p[0]->desc.rateMap->width == 1920 && p[0]->desc.rateMap->height == 1824);
    CHECK(p.size() == 1 && p[0]->desc.rateMap->layers == 2 && st.fovPasses == 1 && st.fovLogicalW == 1920);
    // El mapa se conserva: la salida de color se guarda (el desdoblado la lee) aunque el juego no la quisiera.
    CHECK(p.size() == 1 && p[0]->desc.colors[0].store == vpgpu::StoreAction::Store);
    vpgpu::Frame f2;
    TranslateStats st2;
    translate(rec, f2, st2, &j);  // sin estado: como hoy
    const auto p2 = passes(f2);
    CHECK(p2.size() == 1 && !p2[0]->desc.rateMap && st2.fovPasses == 0 && p2[0]->desc.colors[0].store == vpgpu::StoreAction::DontCare);
    vpgpu::Frame f3;
    TranslateStats st3;
    translate(rec, f3, st3, nullptr, nullptr, nullptr, &fs);  // sin jitter: no hay pase de ojo identificado
    CHECK(passes(f3).size() == 1 && !passes(f3)[0]->desc.rateMap);
  }
  {  // la UI se pinta fuera del pase foveado: el pase sigue sin mapa, cargando todo
    Recorder rec;
    rec.ops.emplace_back(eye(false)), rec.ops.emplace_back(geo), rec.ops.emplace_back(glass), rec.ops.emplace_back(OpEndPass{});
    FoveationState fs;
    vpgpu::Frame f;
    TranslateStats st;
    translate(rec, f, st, &j, nullptr, nullptr, &fs);
    const auto p = passes(f);
    CHECK(p.size() == 2 && st.fovUiSplits == 1);
    if (p.size() == 2) {
      CHECK(p[0]->desc.rateMap && p[0]->draws.size() == 1);
      CHECK(!p[1]->desc.rateMap && p[1]->draws.size() == 1 && p[1]->desc.colors[0].load == vpgpu::LoadAction::Load &&
            p[1]->desc.depth.load == vpgpu::LoadAction::Load);
    }
    FoveationState nosplit;
    nosplit.uiSplit = false;
    vpgpu::Frame g;
    TranslateStats sg;
    translate(rec, g, sg, &j, nullptr, nullptr, &nosplit);
    CHECK(passes(g).size() == 1 && sg.fovUiSplits == 0 && passes(g)[0]->desc.rateMap);
  }
  {  // los clears de área no llevan índice de viewport: ese pase no se foveará
    Recorder rec;
    rec.ops.emplace_back(eye(true)), rec.ops.emplace_back(geo), rec.ops.emplace_back(OpEndPass{});
    FoveationState fs;
    vpgpu::Frame f;
    TranslateStats st;
    translate(rec, f, st, &j, nullptr, nullptr, &fs);
    const auto p = passes(f);
    CHECK(!p.empty() && !p[0]->desc.rateMap && st.fovPasses == 0);
  }
  // Apagado otra vez: ningún pase lleva mapa.
  vpvk_config_set_foveation_apply(0);
  {
    Recorder rec;
    rec.ops.emplace_back(eye(false)), rec.ops.emplace_back(geo), rec.ops.emplace_back(OpEndPass{});
    FoveationState fs;
    vpgpu::Frame f;
    TranslateStats st;
    translate(rec, f, st, &j, nullptr, nullptr, &fs);
    CHECK(passes(f).size() == 1 && !passes(f)[0]->desc.rateMap && st.fovPasses == 0);
  }
  vpvk_foveation off{VPVK_FOVEATION_OFF, 0.4f, 0.3f, 0.0f};
  vpvk_config_set_foveation(&off);
  vpvk_config_set_taa_native(0);
  unsetenv("VPVK_NULL_RATEMAP");
}


// ---- stable caches: what the stage keys and the version components do (recipes.h "when to bump") -------------------
// Digest of what the translator produces for the SPIR-V corpus of gpu/shader/spirv/tests (every module as the stage its
// file name says, default options; only the modules that need no descriptor layout translate, the rest count as
// rejected, which is also an output): the shader id (hash of the MSL text + argument slots) of each. It replaces the
// CMake source fingerprint: a refactor that leaves the output alone keeps the digest (and every cache), a change of
// output moves it. When this test fails the translator's OUTPUT changed: bump kLowerVersion (SPIR-V -> VPIR) or
// kEmitVersion (VPIR -> MSL) in recipes.h, then update kCorpusDigest to the printed value in the same change.
constexpr uint64_t kCorpusDigest = 0x80f07f6e026dc9bfull;  // the "corpus digest" line the test prints

uint64_t corpusDigest(unsigned& modules, unsigned& failed) {
  std::vector<std::string> names;
  for (const auto& e : std::filesystem::directory_iterator(VPVK_SPIRV_TESTS))
    if (e.path().extension() == ".spv") names.push_back(e.path().filename().string());
  std::sort(names.begin(), names.end());
  uint64_t h = 0xcbf29ce484222325ull;
  auto mix = [&](uint64_t v) { h = (h ^ v) * 0x100000001b3ull; };
  modules = failed = 0;
  for (const std::string& n : names) {
    vpir::Stage stage;
    if (n.find(".vert") != std::string::npos) stage = vpir::Stage::Vertex;
    else if (n.find(".frag") != std::string::npos) stage = vpir::Stage::Fragment;
    else if (n.find(".comp") != std::string::npos) stage = vpir::Stage::Compute;
    else continue;
    const std::vector<uint32_t> w = spirvFile(n);
    std::string error;
    std::shared_ptr<spirv::SpirvModule> m(spirv::parse(w.data(), w.size(), error), spirv::release);
    ++modules;
    for (char ch : n) mix(uint8_t(ch));
    if (!m) {
      ++failed, mix(1);
      continue;
    }
    spirv::SpirvLowerOptions opt;
    opt.entry = "main", opt.stage = stage;
    uint64_t id = 0;
    ShaderBlob blob;
    if (!translateRecipeStage(*m, opt, id, blob, error, nullptr)) {  // a rejected module counts too
      ++failed, id = 0;
      if (std::getenv("VPVK_CORPUS_VERBOSE")) std::printf("  corpus reject %s: %s\n", n.c_str(), error.c_str());
    }
    mix(id);
  }
  return h;
}

void testStageKeys() {
  // Version components: one bit per component that moved.
  CHECK(translatorDiff(kTranslatorCode) == 0);
  CHECK(translatorDiff(kTranslatorCode + 1) == kDiffBaseline);
  CHECK(translatorDiff(kTranslatorCode + (1ull << 16)) == kDiffLower);
  CHECK(translatorDiff(kTranslatorCode + (1ull << 32)) == kDiffEmit);
  CHECK(translatorDiff(kTranslatorCode + (1ull << 48)) == kDiffDriver);
  CHECK(translatorDiff(kTranslatorCode + (1ull << 16) + (1ull << 32)) == (kDiffLower | kDiffEmit));
  CHECK(translatorTag(kTranslatorCode + (1ull << 16)) != translatorTag());
  CHECK(kTranslatorVersion == kTranslatorBaseline + kLowerVersion + kEmitVersion + kDriverVersion);

  // The translator's output for the corpus: bump discipline (see kCorpusDigest).
  unsigned modules = 0, failed = 0;
  const uint64_t digest = corpusDigest(modules, failed);
  std::printf("corpus digest modules=%u failed=%u digest=0x%016llx translator=%s\n", modules, failed,
              (unsigned long long)digest, translatorTag().c_str());
  CHECK(modules > 20);
  CHECK(digest == kCorpusDigest);  // false: output changed, bump kLowerVersion / kEmitVersion and update kCorpusDigest

  // spirvUsesMultisample: only a module that declares a multisampled image has a key that depends on MSAA.
  auto imageModule = [](uint32_t ms) {
    std::vector<uint32_t> w = {0x07230203, 0x00010000, 0, 10, 0};
    w.insert(w.end(), {(9u << 16) | 25u, 1, 2, 1, 0, 0, ms, 1, 0});  // %1 = OpTypeImage %2 2D depth=0 arrayed=0 MS=ms
    return w;
  };
  std::vector<uint32_t> ms1 = imageModule(1), ms0 = imageModule(0);
  CHECK(spirvUsesMultisample(ms1.data(), ms1.size()));
  CHECK(!spirvUsesMultisample(ms0.data(), ms0.size()));
  std::vector<uint32_t> bad = ms0;
  bad.back() = 0, bad[5] = 0;  // zero-length instruction: assume the worst
  CHECK(spirvUsesMultisample(bad.data(), bad.size()));
  const std::vector<uint32_t> cull = spirvFile("cull.comp.opt.spv");
  CHECK(!cull.empty() && !spirvUsesMultisample(cull.data(), cull.size()));

  // Cache levels: why a stage misses (recipes.h StageMiss), and that unrelated state never changes the key.
  char tmpl[] = "/tmp/vpvk_keys_XXXXXX";
  const char* dir = mkdtemp(tmpl);
  CHECK(dir != nullptr);
  if (!dir) return;
  StageInput in;
  in.spirv = spirvHash(cull.data(), cull.size());
  in.opt.entry = "main", in.opt.stage = vpir::Stage::Compute;
  const uint64_t h = in.hash();
  StageInput same = in;  // an equal input (another object, e.g. another run) has the same key
  CHECK(same.hash() == h);
  StageInput other = in;
  other.opt.viewCount = 2;  // an option that changes the output changes the key
  CHECK(other.hash() != h);
  StageInput single = in;
  single.opt.multisampleAsSingle = true;
  CHECK(single.hash() != h);  // ... which is why translateStage clears it for modules without multisample images
  {
    RecipeStore s(dir);
    ShaderBlob b;
    b.source = "// msl";
    StageIface f;
    s.addStage(in, h, cull, 0x1234, b, &f);
  }
  {
    RecipeStore s(dir);
    RecipeStore::StageHit hit;
    RecipeStore::MissInfo miss;
    CHECK(s.lookupStage(h, hit, &miss, in.spirv) && hit.mslId == 0x1234);
    CHECK(!s.lookupStage(other.hash(), hit, &miss, in.spirv) && miss.why == RecipeStore::StageMiss::NewOptions);
    CHECK(!s.lookupStage(h ^ 1, hit, &miss, in.spirv ^ 1) && miss.why == RecipeStore::StageMiss::NewContent);
  }
  for (uint8_t bit : {kDiffLower, kDiffEmit, kDiffDriver, kDiffBaseline}) {
    const uint64_t shift = bit == kDiffLower ? 16 : bit == kDiffEmit ? 32 : bit == kDiffDriver ? 48 : 0;
    RecipeStore s(dir, kTranslatorVersion + 1, kTranslatorCode + (1ull << shift));
    RecipeStore::StageHit hit;
    RecipeStore::MissInfo miss;
    CHECK(!s.lookupStage(h, hit, &miss, in.spirv));
    CHECK(miss.why == RecipeStore::StageMiss::Version && miss.components == bit);
    // The record is retranslated by the refresh; its MSL did not change (the fake translator below cannot tell, so the
    // count only has to be consistent): what matters is that nothing but the moved component is blamed.
  }
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);

  // A module without multisample images keeps its key when MSAA forcing is toggled (cache entry reused).
  uint64_t translations[2] = {0, 0};
  char t2[] = "/tmp/vpvk_keys2_XXXXXX";
  const char* d2 = mkdtemp(t2);
  CHECK(d2 != nullptr);
  if (!d2) return;
  setenv("VP_PIPELINE_CACHE_DIR", d2, 1);
  for (int force = 0; force < 2; ++force) {
    vpvk_config_set_force_single_sample(force);
    const uint64_t tr0 = stageTranslations();
    {
      Ctx c;
      VkDescriptorSetLayoutBinding b[4] = {
          {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
          {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
          {3, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
      VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
      lci.bindingCount = 4, lci.pBindings = b;
      VkDescriptorSetLayout sl;
      vkCreateDescriptorSetLayout(c.dev, &lci, nullptr, &sl);
      VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 100};
      VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      plci.setLayoutCount = 1, plci.pSetLayouts = &sl, plci.pushConstantRangeCount = 1, plci.pPushConstantRanges = &pcr;
      VkPipelineLayout pl;
      vkCreatePipelineLayout(c.dev, &plci, nullptr, &pl);
      VkShaderModule cs = module(c, "cull.comp.opt.spv");
      VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
      ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
      ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT, ci.stage.module = cs, ci.stage.pName = "main";
      ci.layout = pl;
      VkPipeline pipe;
      vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe);
      const PipelineObj* po = from<PipelineObj>(pipe);
      CHECK(po->ok);
      c.gpu().pipelines().waitIdle();
      vkDestroyPipeline(c.dev, pipe, nullptr);
      vkDestroyShaderModule(c.dev, cs, nullptr);
      vkDestroyPipelineLayout(c.dev, pl, nullptr);
      vkDestroyDescriptorSetLayout(c.dev, sl, nullptr);
    }
    vpvk_config_set_force_single_sample(0);
    translations[force] = stageTranslations() - tr0;
  }
  unsetenv("VP_PIPELINE_CACHE_DIR");
  std::filesystem::remove_all(d2, ec);
  CHECK(translations[0] == 1 && translations[1] == 0);  // the second run (MSAA forced off) hit the stage cache
}

// ---- asynchronous creation, the frame gate (pipelines.cpp "compile policy") ------------------------------------------
void testAsyncPipelines() {
  setenv("VPVK_ASYNC_PIPELINES", "1", 1);
  setenv("VPVK_PIPELINE_PERSIST_WAIT_MS", "300", 1);  // read once by the first gate call
  unsetenv("VPVK_REPORT_GEOM");
  Ctx c;
  Device* d = from<Device>(c.dev);
  CHECK(d->translator != nullptr);
  if (!d->translator) return;
  CHECK(!c.gpu().holdMs());  // the gate replaces the core's frame hold

  VkAttachmentDescription att{};
  att.format = VK_FORMAT_R8G8B8A8_UNORM, att.samples = VK_SAMPLE_COUNT_1_BIT;
  att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  VkAttachmentReference cref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkSubpassDescription sp{};
  sp.colorAttachmentCount = 1, sp.pColorAttachments = &cref;
  VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  rpi.attachmentCount = 1, rpi.pAttachments = &att, rpi.subpassCount = 1, rpi.pSubpasses = &sp;
  VkRenderPass rp;
  vkCreateRenderPass(c.dev, &rpi, nullptr, &rp);
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  VkPipelineLayout pl;
  vkCreatePipelineLayout(c.dev, &plci, nullptr, &pl);
  VkShaderModule vs = module(c, "uimask.vert.spv"), fs = module(c, "uimask.frag.spv");
  VkPipelineShaderStageCreateInfo stg[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                         {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
  stg[0].stage = VK_SHADER_STAGE_VERTEX_BIT, stg[0].module = vs, stg[0].pName = "main";
  stg[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT, stg[1].module = fs, stg[1].pName = "main";
  VkPipelineVertexInputStateCreateInfo vis{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo ias{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ias.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.lineWidth = 1;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineColorBlendAttachmentState cba{};
  cba.colorWriteMask = 0xF;
  VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cbs.attachmentCount = 1, cbs.pAttachments = &cba;
  VkViewport vp{0, 0, 64, 64, 0, 1};
  VkRect2D sc{{0, 0}, {64, 64}};
  VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vps.viewportCount = 1, vps.pViewports = &vp, vps.scissorCount = 1, vps.pScissors = &sc;
  VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  gp.stageCount = 2, gp.pStages = stg, gp.pVertexInputState = &vis, gp.pInputAssemblyState = &ias;
  gp.pViewportState = &vps, gp.pRasterizationState = &rs, gp.pMultisampleState = &ms, gp.pColorBlendState = &cbs;
  gp.layout = pl, gp.renderPass = rp;

  // 1. The pool is busy with other work: vkCreateGraphicsPipelines still returns at once, nothing translated yet.
  const unsigned nThreads = d->translator->threads();
  std::atomic<unsigned> started{0};
  std::atomic<bool> release{false};
  for (unsigned i = 0; i < nThreads; ++i)
    d->translator->submit(
        [&] {
          ++started;
          while (!release) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        },
        0);
  for (int i = 0; i < 2000 && started < nThreads; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  CHECK(started == nThreads);

  constexpr uint32_t kN = 12;
  VkGraphicsPipelineCreateInfo infos[kN];
  VkPipelineRasterizationStateCreateInfo rss[kN];
  const VkCullModeFlags cull[3] = {VK_CULL_MODE_NONE, VK_CULL_MODE_BACK_BIT, VK_CULL_MODE_FRONT_BIT};
  for (uint32_t i = 0; i < kN; ++i) {
    rss[i] = rs;
    rss[i].cullMode = cull[i % 3];
    rss[i].frontFace = i / 3 % 2 ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rss[i].polygonMode = i < 6 ? VK_POLYGON_MODE_FILL : VK_POLYGON_MODE_POINT;
    infos[i] = gp;
    infos[i].pRasterizationState = &rss[i];
  }
  const uint64_t async0 = pipelineCounters().pipelinesAsync.load();
  VkPipeline pipes[kN] = {};
  CHECK(vkCreateGraphicsPipelines(c.dev, VK_NULL_HANDLE, kN, infos, nullptr, pipes) == VK_SUCCESS);
  CHECK(pipelineCounters().pipelinesAsync.load() - async0 == kN);
  for (uint32_t i = 0; i < kN; ++i) {
    const PipelineObj* po = from<PipelineObj>(pipes[i]);
    CHECK(po && !po->resolved.load() && po->job && !po->job->done());
    CHECK(po && po->pointMode == (i >= 6) && po->state.primitive == (i >= 6 ? vpgpu::Primitive::Points : vpgpu::Primitive::Triangles));
  }

  // 2. A draw needing the translation waits for it (boosted past everything queued), then has the pipeline.
  vpgpu::NullBackend& nb = static_cast<vpgpu::NullBackend&>(d->platform->backend());
  std::atomic<bool> compileGate{false};
  nb.onCompile = [&] {
    while (!compileGate) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  };
  std::thread releaser([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    release = true;
  });
  const auto t0 = std::chrono::steady_clock::now();
  const PipelineObj* p0 = from<PipelineObj>(pipes[0]);
  CHECK(resolvePipeline(d, *p0) && p0->ok && p0->key.value != 0);
  CHECK(std::chrono::steady_clock::now() - t0 >= std::chrono::milliseconds(30));  // it did wait for the busy pool
  releaser.join();
  for (uint32_t i = 0; i < kN; ++i) {
    const PipelineObj* po = from<PipelineObj>(pipes[i]);
    CHECK(resolvePipeline(d, *po) && po->ok && po->state.vertexShader && po->state.fragmentShader);
  }

  // 3. Compiles are held: the frame gate. Soft pass (a texture nobody reads later): the draw is dropped after the
  // small wait, never drawn with anything else.
  vpgpu::TextureDesc td;
  td.format = vpgpu::Format::RGBA8Unorm, td.width = td.height = 64, td.usage = vpgpu::usage::colorTarget | vpgpu::usage::sampled;
  const vpgpu::TextureId soft = c.gpu().createTexture(td);
  td.persistent = true;
  const vpgpu::TextureId hard = c.gpu().createTexture(td);
  CHECK(soft && hard);
  markEyeTexture(*d, soft);  // the eye layer's image (vpvk_mark_eye_image): the only soft destination
  CHECK(isEyeTexture(*d, soft) && !isEyeTexture(*d, hard));
  td.persistent = false;
  const vpgpu::TextureId transientRt = c.gpu().createTexture(td);  // not marked, not persistent: hard now
  CHECK(transientRt);
  auto frameOf = [&](vpgpu::TextureId tex, vpgpu::PipelineKey k, int draws) {
    vpgpu::Frame f;
    vpgpu::RenderPassDesc rd;
    rd.colors[0].texture = tex;
    rd.colors[0].store = vpgpu::StoreAction::Store;
    f.beginRenderPass(rd);
    for (int i = 0; i < draws; ++i) {
      vpgpu::Draw dr;
      dr.pipeline = k, dr.count = 3;
      f.draw(dr);
    }
    f.endRenderPass();
    return f;
  };
  auto drawsIn = [](const vpgpu::Frame& f) {
    size_t n = 0;
    for (const vpgpu::Command& cmd : f.commands())
      if (const auto* rp2 = std::get_if<vpgpu::RenderPass>(&cmd)) n += rp2->draws.size();
    return n;
  };
  vpgpu::PipelineCache& cache = c.gpu().pipelines();
  for (int i = 0; i < 400 && cache.idle(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  CHECK(!cache.idle());  // the compiles are held
  {
    vpgpu::Frame f = frameOf(soft, p0->key, 3);
    const uint64_t skipped0 = pipelineCounters().gateSkippedDraws.load();
    const GateResult r = gateFramePipelines(d, f, 1);
    CHECK(r.draws == 3 && r.skipped == 3 && r.softPending == 1 && r.hardPending == 0);
    CHECK(drawsIn(f) == 0 && r.waitedMs < 250);
    CHECK(pipelineCounters().gateSkippedDraws.load() - skipped0 == 3);
  }
  // 4. Hard pass (a persistent, non-eye render target): the frame waits for the compile; here it never comes within the
  // budget (300 ms), so the draw is dropped and the timeout counted.
  {
    vpgpu::Frame f = frameOf(hard, p0->key, 1);
    const uint64_t timeouts0 = pipelineCounters().hardTimeouts.load();
    const GateResult r = gateFramePipelines(d, f, 2);
    CHECK(r.hardPending == 1 && r.hardWaitedMs >= 250 && r.skipped == 1 && drawsIn(f) == 0);
    CHECK(pipelineCounters().hardTimeouts.load() - timeouts0 == 1);
  }
  // 4a. Nothing but a marked eye image is soft: an unmarked non-persistent RT, and a pass with no colour, wait too.
  {
    vpgpu::Frame f = frameOf(transientRt, p0->key, 2);
    const GateResult r = gateFramePipelines(d, f, 2);
    CHECK(r.hardPending == 1 && r.softPending == 0 && r.pendingByClass[2] == 2 && r.skippedByClass[2] == 2);
  }
  {
    vpgpu::Frame f;
    vpgpu::RenderPassDesc rd;  // depth/no-colour pass
    f.beginRenderPass(rd);
    vpgpu::Draw dr;
    dr.pipeline = p0->key, dr.count = 3;
    f.draw(dr);
    f.endRenderPass();
    const GateResult r = gateFramePipelines(d, f, 2);
    CHECK(r.hardPending == 1 && r.softPending == 0 && r.pendingByClass[1] == 1);
  }
  {  // the eye image through its resolve target (MSAA colour resolved into it) is soft; the class is logged
    vpgpu::Frame f;
    vpgpu::RenderPassDesc rd;
    rd.colors[0].texture = transientRt;
    rd.colors[0].resolve = soft;
    rd.colors[0].store = vpgpu::StoreAction::Resolve;
    f.beginRenderPass(rd);
    vpgpu::Draw dr;
    dr.pipeline = p0->key, dr.count = 3;
    f.draw(dr);
    f.endRenderPass();
    const GateResult r = gateFramePipelines(d, f, 2);
    CHECK(r.softPending == 1 && r.hardPending == 0 && r.skippedByClass[0] == 1);
  }
  // 4b. A layered pass (shadow cascades, cube probes: layerCount is the framebuffer's layers) is no eye: hard as well.
  {
    vpgpu::Frame f;
    vpgpu::RenderPassDesc rd;
    rd.colors[0].texture = hard;
    rd.colors[0].store = vpgpu::StoreAction::Store;
    rd.layerCount = 4;
    f.beginRenderPass(rd);
    vpgpu::Draw dr;
    dr.pipeline = p0->key, dr.count = 3;
    f.draw(dr);
    f.endRenderPass();
    const GateResult r = gateFramePipelines(d, f, 2);
    CHECK(r.hardPending == 1 && r.softPending == 0 && r.skipped == 1);
  }
  // 5. The compile finishes while the hard wait runs: the draw stays, nothing skipped.
  {
    std::thread opener([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(60));
      compileGate = true;
    });
    vpgpu::Frame f = frameOf(hard, from<PipelineObj>(pipes[1])->key, 2);
    const GateResult r = gateFramePipelines(d, f, 3);
    opener.join();
    CHECK(r.hardPending == 1 && r.skipped == 0 && drawsIn(f) == 2 && r.hardWaitedMs < 290);
  }
  // 6. Everything compiled: the gate is free (no wait, no drop).
  cache.waitIdle();
  nb.onCompile = nullptr;
  for (uint32_t i = 0; i < kN; ++i) CHECK(cache.status(from<PipelineObj>(pipes[i])->key) == vpgpu::PipelineCache::Status::Ready);
  {
    vpgpu::Frame f = frameOf(soft, p0->key, 2);
    const GateResult r = gateFramePipelines(d, f, 4);
    CHECK(r.skipped == 0 && r.softPending == 0 && r.hardPending == 0 && drawsIn(f) == 2);
  }
  // 7. vkDestroyPipeline of a pipeline whose job still waits in a busy pool does not deadlock.
  {
    std::atomic<unsigned> st2{0};
    std::atomic<bool> rel2{false};
    for (unsigned i = 0; i < nThreads; ++i)
      d->translator->submit(
          [&] {
            ++st2;
            while (!rel2) std::this_thread::sleep_for(std::chrono::milliseconds(1));
          },
          0);
    for (int i = 0; i < 2000 && st2 < nThreads; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    VkPipeline extra = VK_NULL_HANDLE;
    vkCreateGraphicsPipelines(c.dev, VK_NULL_HANDLE, 1, &gp, nullptr, &extra);
    std::thread opener([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
      rel2 = true;
    });
    vkDestroyPipeline(c.dev, extra, nullptr);
    opener.join();
  }
  c.gpu().destroy(soft);
  c.gpu().destroy(hard);
  c.gpu().destroy(transientRt);
  for (VkPipeline p : pipes) vkDestroyPipeline(c.dev, p, nullptr);
  // vpgles (internal client) takes the asynchronous mode by default too. No second device here: creating one while `c`
  // is alive hung the Metal CI run; the flag is decided by persistWaitMs(bool), checked on its own.
  unsetenv("VPVK_ASYNC_PIPELINES");
  CHECK(asyncPipelinesOn());
  CHECK(persistWaitMs(false) == 300 && persistWaitMs(true) == 300);  // the env var wins for both
  setenv("VPVK_ASYNC_PIPELINES", "0", 1);
  CHECK(!asyncPipelinesOn());
}

// vpvk_cmd_pass_drop: the host (vpgles' glInvalidateFramebuffer) learns after vkCmdBeginRendering that an attachment is
// undefined at the end of the pass. Plain stores become DontCare (and back, the call replaces the earlier one); resolve
// targets, areas smaller than the attachment, aspects not attached and eye images keep their store.
void testPassDrop() {
  Ctx c;
  VkDeviceMemory m = alloc(c, 64 << 20, 0);
  const VkImageUsageFlags cu = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  const VkImageUsageFlags du = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  VkImage col = image(c, VK_FORMAT_R8G8B8A8_UNORM, 64, 1, cu, m);
  VkImage col2 = image(c, VK_FORMAT_R8G8B8A8_UNORM, 64, 1, cu, m);
  VkImage eye = image(c, VK_FORMAT_R8G8B8A8_UNORM, 64, 1, cu, m);
  VkImage dep = image(c, VK_FORMAT_D32_SFLOAT, 64, 1, du, m);
  VkImage ds = image(c, VK_FORMAT_D24_UNORM_S8_UINT, 64, 1, du, m);
  CHECK(col && col2 && eye && dep && ds);
  auto view = [&](VkImage i, VkFormat f, VkImageAspectFlags a) {
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = i, vi.viewType = VK_IMAGE_VIEW_TYPE_2D, vi.format = f;
    vi.subresourceRange = {a, 0, 1, 0, 1};
    VkImageView v = VK_NULL_HANDLE;
    vkCreateImageView(c.dev, &vi, nullptr, &v);
    return v;
  };
  VkImageView cv = view(col, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
  VkImageView rv = view(col2, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
  VkImageView ev = view(eye, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
  VkImageView dv = view(dep, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);
  VkImageView sv = view(ds, VK_FORMAT_D24_UNORM_S8_UINT, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
  CHECK(cv && rv && ev && dv && sv);
  CHECK(vpvk_mark_eye_image(eye) == 1);
  VkImage eyeDep = image(c, VK_FORMAT_D32_SFLOAT, 64, 1, du, m);
  VkImageView edv = view(eyeDep, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);
  CHECK(eyeDep && edv);
  CHECK(vpvk_mark_eye_image(eyeDep) == 1);

  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  VkCommandPool pool;
  vkCreateCommandPool(c.dev, &pci, nullptr, &pool);
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = pool;
  ai.commandBufferCount = 1;
  VkCommandBuffer cb;
  vkAllocateCommandBuffers(c.dev, &ai, &cb);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(cb, &bi);
  auto beginOp = [&]() -> OpBeginRendering& {
    Recorder& r = *from<CommandBuffer>(cb)->rec;
    return std::get<OpBeginRendering>(r.ops[r.passBegin]);
  };
  const auto STORE = vpgpu::StoreAction::Store, DONT = vpgpu::StoreAction::DontCare;
  auto att = [](VkImageView v) {
    VkRenderingAttachmentInfo a{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    a.imageView = v, a.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    a.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    return a;
  };
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, {64, 64}};
  ri.layerCount = 1;
  ri.colorAttachmentCount = 1;

  // Colour + depth: dropped, then restored by a later call (the attachment was written again).
  VkRenderingAttachmentInfo ca = att(cv), da = att(dv);
  ri.pColorAttachments = &ca, ri.pDepthAttachment = &da, ri.pStencilAttachment = nullptr;
  vkCmdBeginRendering(cb, &ri);
  CHECK(beginOp().desc.colors[0].store == STORE && beginOp().desc.depth.store == STORE);
  vpvk_cmd_pass_drop(cb, 1, 0, 0);
  CHECK(beginOp().desc.colors[0].store == DONT && beginOp().desc.depth.store == STORE);
  vpvk_cmd_pass_drop(cb, 1, 1, 0);
  CHECK(beginOp().desc.colors[0].store == DONT && beginOp().desc.depth.store == DONT);
  vpvk_cmd_pass_drop(cb, 0, 1, 0);
  CHECK(beginOp().desc.colors[0].store == STORE && beginOp().desc.depth.store == DONT);
  vpvk_cmd_pass_drop(cb, 0, 0, 0);
  CHECK(beginOp().desc.colors[0].store == STORE && beginOp().desc.depth.store == STORE);
  vkCmdEndRendering(cb);
  vpvk_cmd_pass_drop(cb, 1, 1, 1);  // no open pass: nothing to change, nothing to crash

  // Depth and stencil of one image: dropped only when both aspects are.
  VkRenderingAttachmentInfo dd = att(sv), sd = att(sv);
  ri.colorAttachmentCount = 0, ri.pColorAttachments = nullptr, ri.pDepthAttachment = &dd, ri.pStencilAttachment = &sd;
  vkCmdBeginRendering(cb, &ri);
  vpvk_cmd_pass_drop(cb, 0, 1, 0);
  CHECK(beginOp().desc.depth.store == STORE);
  vpvk_cmd_pass_drop(cb, 0, 0, 1);
  CHECK(beginOp().desc.depth.store == STORE);
  vpvk_cmd_pass_drop(cb, 0, 1, 1);
  CHECK(beginOp().desc.depth.store == DONT);
  vkCmdEndRendering(cb);
  // Only the depth aspect attached: the stencil aspect of the image must survive, never dropped.
  ri.pStencilAttachment = nullptr;
  vkCmdBeginRendering(cb, &ri);
  vpvk_cmd_pass_drop(cb, 0, 1, 1);
  CHECK(beginOp().desc.depth.store == STORE);
  vkCmdEndRendering(cb);

  // A resolve target keeps its store; so does an area smaller than the attachment; so does an eye image.
  ri.colorAttachmentCount = 1, ri.pColorAttachments = &ca, ri.pDepthAttachment = nullptr;
  VkRenderingAttachmentInfo cr = att(cv);
  cr.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT, cr.resolveImageView = rv, cr.resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL;
  ri.pColorAttachments = &cr;
  vkCmdBeginRendering(cb, &ri);
  const vpgpu::StoreAction resolved = beginOp().desc.colors[0].store;
  CHECK(resolved == vpgpu::StoreAction::StoreAndResolve);
  vpvk_cmd_pass_drop(cb, 1, 0, 0);
  CHECK(beginOp().desc.colors[0].store == resolved);
  vkCmdEndRendering(cb);
  ri.pColorAttachments = &ca;
  ri.renderArea = {{0, 0}, {32, 32}};
  vkCmdBeginRendering(cb, &ri);
  vpvk_cmd_pass_drop(cb, 1, 0, 0);
  CHECK(beginOp().desc.colors[0].store == STORE);
  vkCmdEndRendering(cb);
  ri.renderArea = {{0, 0}, {64, 64}};
  VkRenderingAttachmentInfo ce = att(ev);
  ri.pColorAttachments = &ce;
  vkCmdBeginRendering(cb, &ri);
  vpvk_cmd_pass_drop(cb, 1, 0, 0);
  CHECK(beginOp().desc.colors[0].store == STORE);
  vkCmdEndRendering(cb);
  // An eye depth image is never dropped either.
  VkRenderingAttachmentInfo de = att(edv);
  ri.colorAttachmentCount = 0, ri.pColorAttachments = nullptr, ri.pDepthAttachment = &de, ri.pStencilAttachment = nullptr;
  vkCmdBeginRendering(cb, &ri);
  vpvk_cmd_pass_drop(cb, 0, 1, 0);
  CHECK(beginOp().desc.depth.store == STORE);
  vkCmdEndRendering(cb);
  vkEndCommandBuffer(cb);

  vkDestroyCommandPool(c.dev, pool, nullptr);
  for (VkImageView v : {cv, rv, ev, dv, sv, edv}) vkDestroyImageView(c.dev, v, nullptr);
  for (VkImage i : {col, col2, eye, dep, ds, eyeDep}) vkDestroyImage(c.dev, i, nullptr);
  vkFreeMemory(c.dev, m, nullptr);
}

// Draws dropped at record time say why (OpDraw::reason) and the replay counts them per reason (TranslateStats::skippedBy):
// an empty scissor (alone, and outside the render area) and a static FRONT_AND_BACK cull; the draw that is fine is drawn.
void testSkipReasons() {
  Ctx c;
  DrawRig rig(c);
  VkShaderModule vs = module(c, "pos.vert.spv"), fs = module(c, "solid.frag.spv");
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  VkPipelineLayout pl;
  vkCreatePipelineLayout(c.dev, &plci, nullptr, &pl);
  PipeOpts dyn;
  dyn.dynamic = {VK_DYNAMIC_STATE_SCISSOR};
  PipeOpts culled;
  culled.cull = VK_CULL_MODE_FRONT_AND_BACK;
  VkPipeline scissored = makePosPipeline(c, rig, pl, dyn, vs, fs);
  VkPipeline both = makePosPipeline(c, rig, pl, culled, vs, fs);
  CHECK(from<PipelineObj>(scissored)->ok && from<PipelineObj>(both)->ok && from<PipelineObj>(both)->cullBoth);
  c.gpu().pipelines().waitIdle();

  rig.begin();
  vkCmdBindPipeline(rig.cb, VK_PIPELINE_BIND_POINT_GRAPHICS, scissored);
  VkRect2D empty{{4, 4}, {0, 0}}, outside{{100, 100}, {10, 10}}, fine{{0, 0}, {32, 32}};
  vkCmdSetScissor(rig.cb, 0, 1, &empty);
  vkCmdDraw(rig.cb, 3, 1, 0, 0);  // 0: empty scissor
  vkCmdSetScissor(rig.cb, 0, 1, &outside);
  vkCmdDraw(rig.cb, 3, 1, 0, 0);  // 1: scissor outside the 64x64 target: empty once clamped
  vkCmdSetScissor(rig.cb, 0, 1, &fine);
  vkCmdDraw(rig.cb, 3, 1, 0, 0);  // 2: drawn
  vkCmdBindPipeline(rig.cb, VK_PIPELINE_BIND_POINT_GRAPHICS, both);
  vkCmdDraw(rig.cb, 3, 1, 0, 0);  // 3: every triangle culled
  vpgpu::Frame frame;
  TranslateStats ts;
  rig.finish(frame, ts);
  std::vector<SkipReason> got;
  for (const Op& op : from<CommandBuffer>(rig.cb)->rec->ops)
    if (const auto* d = std::get_if<OpDraw>(&op)) got.push_back(d->ok ? SkipReason::None : d->reason);
  CHECK(got.size() == 4 && got[0] == SkipReason::Scissor && got[1] == SkipReason::Scissor && got[2] == SkipReason::None &&
        got[3] == SkipReason::Cull);
  CHECK(ts.draws == 1 && ts.skippedDraws == 3);
  CHECK(ts.skippedBy[size_t(SkipReason::Scissor)] == 2 && ts.skippedBy[size_t(SkipReason::Cull)] == 1);
  uint32_t total = 0;
  for (uint32_t n : ts.skippedBy) total += n;
  CHECK(total == ts.skippedDraws);  // every skipped draw has exactly one reason

  vkDestroyPipeline(c.dev, scissored, nullptr), vkDestroyPipeline(c.dev, both, nullptr);
  vkDestroyPipelineLayout(c.dev, pl, nullptr);
}

int main(int argc, char** argv) {
  gLib = argc > 2 ? argv[2] : "libVPVulkan.so";
  // The suites read the pipelines they create right away (the legacy, synchronous mode); the async group turns the
  // asynchronous mode on for its own devices.
  setenv("VPVK_ASYNC_PIPELINES", "0", 1);
  const std::vector<std::pair<const char*, std::function<void()>>> groups = {
      {"exports", testExports}, {"physical", testPhysical}, {"memory", testMemory}, {"msaa_tile_off", testMsaaTileOff},
      {"passes", testPasses},   {"submit", testSubmit}, {"pipeline", testPipeline},
      {"recipes", testRecipes}, {"taa", testTaa}, {"msaa", testMsaa}, {"queries", testQueries},
      {"timeline", testTimeline}, {"renderpass_ops", testRenderPassOps}, {"linear", testLinear}, {"images", testImages}, {"image_lifetime", testImageLifetime},
      {"vertex_stores", testVertexStores}, {"profile", testProfile}, {"profile13", testProfile13}, {"report_geom", testReportGeomInterpolators}, {"pipeline_policy", testPipelinePolicy}, {"eds", testExtendedDynamicState},
      {"native_identity", testNativeIdentity}, {"inline_uniform", testInlineUniformBlock}, {"bda", testBufferDeviceAddress}, {"ycbcr", testYcbcr},
      {"taa_native", testTaaNative}, {"taa_eye_depth", testTaaEyeDepth}, {"taa_uimask", testTaaUiMask}, {"aot", testAot}, {"aniso", testAniso}, {"zeroinit", testZeroInit}, {"archive_index", testArchiveIndex}, {"foveation", testFoveation},
      {"stage_keys", testStageKeys}, {"async_pipelines", testAsyncPipelines}, {"pass_drop", testPassDrop}, {"skip_reasons", testSkipReasons}};
  bool ran = false;
  for (const auto& [name, fn] : groups) {
    if (argc > 1 && std::strcmp(argv[1], name) != 0) continue;
    std::printf("[%s]\n", name);
    vpvk_config_set_quest_identity(0);  // each group opts in to the Quest identity itself
    fn();
    ran = true;
  }
  if (!ran) std::printf("unknown group\n");
  std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
  return failures || !ran ? 1 : 0;
}
