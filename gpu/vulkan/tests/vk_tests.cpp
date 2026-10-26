// VPVulkan tests on the null backend. Usage: vpvk_tests <group> <path to libVPVulkan>.
#include "vpvk/config.h"
#include <dlfcn.h>
#include <unistd.h>  // mkdtemp on macOS (libc++ <cstdlib> does not declare it)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <functional>
#include <string>
#include <chrono>
#include <thread>
#include <tuple>
#include <vector>

#include "commands.h"
#include "foveation.h"
#include "perf.h"
#include "taa_native.h"
#include <cmath>
#include "vpvk.h"

using namespace vpvk;

#include "vpgpu/metal2/archive_index.h"

// MoltenVK-compatible export (entry.cpp); not in any header on purpose (see vpvk/config.h).
extern "C" void vkGetMTLTextureMVK(VkImage, void**);
extern "C" uint32_t vpvk_device_wait_compiles(VkDevice);  // entry.cpp (AVP 0017, dlsym'd by the host)
extern "C" void vpvk_internal_scope(int on);  // ext.cpp

namespace {

int failures = 0;
const char* gLib = nullptr;
const char* gReplay = nullptr;  // vpreplay executable (groups replay, replay_lvp)
std::string gKeepImages;        // testCaptureImages keeps its file here when set (group replay)
const char* gIcd = nullptr;     // ICD manifest of lavapipe (group replay_lvp)
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

VkImage image(Ctx& c, VkFormat f, uint32_t w, uint32_t layers, VkImageUsageFlags usage, VkDeviceMemory m,
              VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT) {
  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = VK_IMAGE_TYPE_2D;
  ii.format = f;
  ii.extent = {w, w, 1};
  ii.mipLevels = 1;
  ii.arrayLayers = layers;
  ii.samples = samples;
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
  // VPVK_AMPLIFY=1 (gpu_vulkan.pipeline_amplify): vertex amplification, one instance per Vulkan instance.
  const bool amplify = (translatorMode() & 2) != 0;
  CHECK(po->amplified == amplify && po->instanceViews == (amplify ? 1u : 2u));
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
    CHECK(d.instances == 3 * po->instanceViews && d.baseInstance == 1 * po->instanceViews && d.count == 36 && d.first == 6 &&
          d.baseVertex == 2);
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
  // The prewarm host's wait: nothing left queued afterwards, and a null device is a no-op.
  CHECK(vpvk_device_wait_compiles(c.dev) < 60000);
  CHECK(c.gpu().pipelines().idle());
  CHECK(vpvk_device_wait_compiles(VK_NULL_HANDLE) == 0);
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
  // Phase 1b (what Half-Life: Alyx asks of a Vulkan 1.3 device): listed here, never with the Quest identity.
  const char* const kAlyx1b[] = {VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME, VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME,
                                 VK_EXT_MEMORY_BUDGET_EXTENSION_NAME, VK_EXT_SEPARATE_STENCIL_USAGE_EXTENSION_NAME,
                                 VK_KHR_SWAPCHAIN_MUTABLE_FORMAT_EXTENSION_NAME,
                                 VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME};
  for (const char* x : kAlyx1b) CHECK(has(x));
  VkPhysicalDeviceExtendedDynamicState3FeaturesEXT e3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
  VkPhysicalDeviceHostImageCopyFeaturesEXT hf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT, &e3};
  VkPhysicalDeviceDynamicRenderingLocalReadFeaturesKHR lr{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES_KHR, &hf};
  VkPhysicalDeviceFeatures2 f1b{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &lr};
  vkGetPhysicalDeviceFeatures2(c.pd, &f1b);
  CHECK(e3.extendedDynamicState3ColorWriteMask && !e3.extendedDynamicState3AlphaToCoverageEnable && hf.hostImageCopy &&
        lr.dynamicRenderingLocalRead);
  {  // 3D images: every axis is limited by maxImageDimension3D
    VkImageFormatProperties ifp;
    CHECK(vkGetPhysicalDeviceImageFormatProperties(c.pd, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TYPE_3D, VK_IMAGE_TILING_OPTIMAL,
                                                   VK_IMAGE_USAGE_SAMPLED_BIT, 0, &ifp) == VK_SUCCESS);
    CHECK(ifp.maxExtent.width == p.limits.maxImageDimension3D && ifp.maxExtent.height == p.limits.maxImageDimension3D);
    VkFormatProperties vfp;
    vkGetPhysicalDeviceFormatProperties(c.pd, VK_FORMAT_R16G16B16_SNORM, &vfp);
    CHECK(vfp.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT);
  }
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
  // Quest identity: none of the phase 1b extensions, not even with VPVK_EXTRA_EXTENSIONS=1.
  for (const char* v : {"0", "1"}) {
    setenv("VPVK_EXTRA_EXTENSIONS", v, 1);
    uint32_t qn = 0;
    vkEnumerateDeviceExtensionProperties(c.pd, nullptr, &qn, nullptr);
    std::vector<VkExtensionProperties> qe(qn);
    vkEnumerateDeviceExtensionProperties(c.pd, nullptr, &qn, qe.data());
    for (const auto& x : qe)
      for (const char* y : kAlyx1b) CHECK(std::strcmp(x.extensionName, y) != 0);
    vkGetPhysicalDeviceFeatures2(c.pd, &f1b);
    CHECK(!e3.extendedDynamicState3ColorWriteMask && !hf.hostImageCopy && !lr.dynamicRenderingLocalRead);
  }
  unsetenv("VPVK_EXTRA_EXTENSIONS");
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
    CHECK(from<Device>(c.dev)->prewarmedRecipes == 0 && from<Device>(c.dev)->recipeMisses.load() == 0);  // nothing stored yet
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
    // gpu.pso.recipe game: the game's create found its recipe stored by the earlier run (a hit, no miss).
    const Device* dv = from<Device>(c.dev);
    CHECK(dv->prewarmedRecipes == 1 && dv->recipeHits.load() == 1 && dv->recipeMisses.load() == 0);
    CHECK(vpvk_device_wait_compiles(c.dev) < 60000);
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

// Translator modes (VPVK_AMPLIFY, VPVK_FETCH_NATIVE): read from the environment, part of the recipe store's translator id
// (a mode change retranslates the stored stages), and the stage interface keeps whether a vertex stage is amplified.
void testTranslatorModes() {
  unsetenv("VPVK_AMPLIFY");
  unsetenv("VPVK_FETCH_NATIVE");
  CHECK(refreshTranslatorMode() == 0 && translatorMode() == 0 && translatorId() == kTranslatorVersion);
  spirv::SpirvLowerOptions o;
  applyTranslatorMode(o);
  CHECK(!o.amplify && !o.nativeFetch && !o.unpackFetch && o.killExit);  // kill exit is on by default (mode bit = off)
  setenv("VPVK_AMPLIFY", "1", 1);
  CHECK(refreshTranslatorMode() == 2 && translatorId() == (kTranslatorVersion | 2u << 16));
  applyTranslatorMode(o);
  CHECK(o.amplify && !o.nativeFetch);
  setenv("VPVK_FETCH_NATIVE", "1", 1);
  setenv("VPVK_AMPLIFY", "0", 1);
  CHECK(refreshTranslatorMode() == 1 && translatorId() == (kTranslatorVersion | 1u << 16));
  applyTranslatorMode(o);
  CHECK(!o.amplify && o.nativeFetch);
  setenv("VPVK_AMPLIFY", "1", 1);
  CHECK(refreshTranslatorMode() == 3);
  // VPVK_FETCH_NATIVE is a mask (1 half, 2 unpack); VPSPV_KILL_EXIT=0 sets bit 3.
  setenv("VPVK_AMPLIFY", "0", 1);
  setenv("VPVK_FETCH_NATIVE", "2", 1);
  CHECK(refreshTranslatorMode() == 4 && translatorId() == (kTranslatorVersion | 4u << 16));
  applyTranslatorMode(o);
  CHECK(!o.nativeFetch && o.unpackFetch && o.killExit);
  setenv("VPVK_FETCH_NATIVE", "3", 1);
  CHECK(refreshTranslatorMode() == 5);
  setenv("VPSPV_KILL_EXIT", "0", 1);
  CHECK(refreshTranslatorMode() == 13);
  applyTranslatorMode(o);
  CHECK(o.nativeFetch && o.unpackFetch && !o.killExit);
  setenv("VPSPV_KILL_EXIT", "1", 1);
  CHECK(refreshTranslatorMode() == 5);
  unsetenv("VPSPV_KILL_EXIT");
  setenv("VPVK_AMPLIFY", "1", 1);
  setenv("VPVK_FETCH_NATIVE", "1", 1);

  char tmpl[] = "/tmp/vpvk_modes_XXXXXX";
  const char* dir = mkdtemp(tmpl);
  CHECK(dir != nullptr);
  if (!dir) return;
  const std::vector<uint32_t> words = spirvFile("cull.comp.opt.spv");
  StageInput in;
  in.spirv = spirvHash(words.data(), words.size());
  in.opt.entry = "main", in.opt.stage = vpir::Stage::Compute;
  {
    RecipeStore s(dir);  // the current mode's id
    ShaderBlob blob;
    blob.source = "// amplified vertex stage";
    StageIface iface;
    iface.amplified = true;
    s.addStage(in, in.hash(), words, 0xA11CE, blob, &iface);
    RecipeStore::StageHit hit;
    CHECK(s.lookupStage(in.hash(), hit) && hit.iface.amplified);
  }
  {
    RecipeStore s(dir);  // reopened: the flag came back from disk
    RecipeStore::StageHit hit;
    CHECK(!s.needsRefresh() && s.lookupStage(in.hash(), hit) && hit.iface.amplified && hit.mslId == 0xA11CE);
  }
  {
    RecipeStore s(dir, kTranslatorVersion);  // modes off: another translator, the stage is stale
    RecipeStore::StageHit hit;
    RecipeStore::MissInfo miss;
    CHECK(s.needsRefresh() && !s.lookupStage(in.hash(), hit, &miss) && miss.why == RecipeStore::StageMiss::Version);
  }
  {  // the stage interface keeps dualSource too (trailing flag byte, bit 1), alone and next to amplified
    char t2[] = "/tmp/vpvk_dual_XXXXXX";
    const char* dir2 = mkdtemp(t2);
    CHECK(dir2 != nullptr);
    if (dir2) {
      StageInput in2 = in;
      in2.opt.entry = "dual";
      {
        RecipeStore s(dir2);
        ShaderBlob blob;
        blob.source = "// dual-source fragment stage";
        StageIface iface, none;
        iface.dualSource = true;
        s.addStage(in2, in2.hash(), words, 0xD0A1, blob, &iface);
        StageIface both;
        both.amplified = both.dualSource = true;
        s.addStage(in, in.hash(), words, 0xA11CE, blob, &both);
        // Pipelines blending with the second source: the recipe keeps the factors only for the stage whose interface
        // says it writes index 1 (the key createGraphics computes), the other falls back to the first source.
        StageInput in3 = in;
        in3.opt.entry = "plain";
        s.addStage(in3, in3.hash(), words, 0xB1A1, blob, &none);
        vpgpu::PipelineState ps;
        ps.blend[0].enabled = true;
        ps.blend[0].srcColor = vpgpu::BlendFactor::Src1Color, ps.blend[0].dstColor = vpgpu::BlendFactor::OneMinusSrc1Alpha;
        ps.fragmentShader = 0xD0A1;
        s.addPipeline(ps, {0, in2.hash(), 0});
        ps.fragmentShader = 0xB1A1;
        s.addPipeline(ps, {0, in3.hash(), 0});
      }
      RecipeStore s(dir2);  // reopened: both flags came back from disk
      for (const vpgpu::PipelineState& ps : s.snapshot().pipelines) {
        const bool dual = ps.fragmentShader == 0xD0A1;
        CHECK(ps.blend[0].srcColor == (dual ? vpgpu::BlendFactor::Src1Color : vpgpu::BlendFactor::SrcColor));
        CHECK(ps.blend[0].dstColor == (dual ? vpgpu::BlendFactor::OneMinusSrc1Alpha : vpgpu::BlendFactor::OneMinusSrcAlpha));
      }
      CHECK(s.snapshot().pipelines.size() == 2);
      RecipeStore::StageHit hit;
      CHECK(s.lookupStage(in2.hash(), hit) && hit.iface.dualSource && !hit.iface.amplified && hit.mslId == 0xD0A1);
      CHECK(s.lookupStage(in.hash(), hit) && hit.iface.dualSource && hit.iface.amplified);
      std::error_code ec2;
      std::filesystem::remove_all(dir2, ec2);
    }
  }
  unsetenv("VPVK_AMPLIFY");
  unsetenv("VPVK_FETCH_NATIVE");
  unsetenv("VPSPV_KILL_EXIT");
  refreshTranslatorMode();
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

  // VPVK_OCCLUSION=0: occlusion queries are not measured (the pool stays unbacked: every query reads 1, available);
  // timestamp pools are unaffected. Unset: measured, as above.
  {
    setenv("VPVK_OCCLUSION", "0", 1);
    VkQueryPool off, ts;
    CHECK(vkCreateQueryPool(c.dev, &qi, nullptr, &off) == VK_SUCCESS);
    VkQueryPoolCreateInfo ti{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    ti.queryType = VK_QUERY_TYPE_TIMESTAMP, ti.queryCount = 2;
    CHECK(vkCreateQueryPool(c.dev, &ti, nullptr, &ts) == VK_SUCCESS);
    CHECK(from<QueryPoolObj>(off)->base == UINT32_MAX && from<QueryPoolObj>(ts)->base != UINT32_MAX);
    uint64_t v[2] = {7, 7};
    CHECK(vkGetQueryPoolResults(c.dev, off, 1, 1, sizeof v, v, 16,
                                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) == VK_SUCCESS);
    CHECK(v[0] == 1 && v[1] == 1);
    vkDestroyQueryPool(c.dev, off, nullptr);
    vkDestroyQueryPool(c.dev, ts, nullptr);
    unsetenv("VPVK_OCCLUSION");
    VkQueryPool on;
    CHECK(vkCreateQueryPool(c.dev, &qi, nullptr, &on) == VK_SUCCESS);
    CHECK(from<QueryPoolObj>(on)->base != UINT32_MAX);
    vkDestroyQueryPool(c.dev, on, nullptr);
  }
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
  {  // What the frame compiler makes of MSAA colour + resolve: a TRANSIENT image wins over the translator's keepSamples (its
     // STORE is dropped with the tile: memoryless), a non-transient one with a client STORE keeps its samples (memory),
     // DONT_CARE stays a memoryless Resolve either way.
    for (const bool transient : {false, true})
      for (const VkAttachmentStoreOp op : {VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_STORE_OP_DONT_CARE}) {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.format = VK_FORMAT_R8G8B8A8_UNORM;
        ii.extent = {64, 64, 1};
        ii.mipLevels = ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_4_BIT;
        ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | (transient ? VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT : 0);
        VkImage ms = VK_NULL_HANDLE, single = VK_NULL_HANDLE;
        vkCreateImage(c.dev, &ii, nullptr, &ms);
        ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        vkCreateImage(c.dev, &ii, nullptr, &single);
        vkBindImageMemory(c.dev, ms, m, 0);
        vkBindImageMemory(c.dev, single, m, 0);
        if (from<Image>(ms)->desc.samples > 1 && from<Image>(ms)->forcedFrom <= 1) {
          VkImageView mv = makeView(ms, VK_IMAGE_ASPECT_COLOR_BIT, 1), ov = makeView(single, VK_IMAGE_ASPECT_COLOR_BIT, 1);
          VkAttachmentDescription a = col, r = col;
          a.samples = VK_SAMPLE_COUNT_4_BIT;
          a.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
          a.storeOp = op;
          r.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, r.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
          VkAttachmentReference rref{1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
          VkSubpassDescription sd = one;
          sd.pResolveAttachments = &rref;
          Pass p = pass({a, r}, {sd}, {}, {mv, ov});
          vpgpu::Frame f;
          TranslateStats st;
          run(p, {{0, 0}, {64, 64}}, nullptr, f, st);
          const vpgpu::CompiledFrame cf = vpgpu::compileFrame(f, &c.gpu().resources());
          CHECK(cf.encoders.size() == 1 && cf.stats.transientViolations == 0);
          if (cf.encoders.size() == 1) {
            const bool store = op == VK_ATTACHMENT_STORE_OP_STORE;
            const bool tile = transient || !store;
            CHECK(cf.encoders[0].tileLocal(0) == tile);
            CHECK(cf.encoders[0].pass.colors[0].store ==
                  (tile ? vpgpu::StoreAction::Resolve : vpgpu::StoreAction::StoreAndResolve));
          }
          vkDestroyImageView(c.dev, mv, nullptr);
          vkDestroyImageView(c.dev, ov, nullptr);
        }
        vkDestroyImage(c.dev, ms, nullptr);
        vkDestroyImage(c.dev, single, nullptr);
      }
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
  void begin(VkRect2D area = {{0, 0}, {64, 64}}) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cb, &bi);
    VkClearValue clears[2]{};
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = rp, rbi.framebuffer = fb, rbi.renderArea = area, rbi.clearValueCount = 2;
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
  VkRect2D scissor{{0, 0}, {64, 64}};  // the pipeline's static scissor
  bool a2c = false;
  bool src1 = false;  // blend with the second-source factors (dual-source blending)
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
  VkRect2D sc = o.scissor;
  VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vps.viewportCount = 1, vps.pViewports = &vp, vps.scissorCount = 1, vps.pScissors = &sc;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.cullMode = o.cull, rs.frontFace = o.front, rs.rasterizerDiscardEnable = o.discard, rs.depthBiasEnable = o.bias;
  rs.lineWidth = 1;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  ms.alphaToCoverageEnable = o.a2c;
  VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  ds.depthTestEnable = o.depthTest, ds.depthWriteEnable = o.depthWrite, ds.depthCompareOp = o.depthOp;
  ds.stencilTestEnable = o.stencilTest;
  ds.front = ds.back = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS, 0xFF, 0xFF, 0};
  VkPipelineColorBlendAttachmentState cba{};
  cba.colorWriteMask = 0xF;
  if (o.src1) {
    cba.blendEnable = VK_TRUE;
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC1_COLOR, cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC1_ALPHA, cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
  }
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
  // 5. Profundidad de ojo TRANSIENT: VPKTAA la lee tras el frame, fuera de lo que ve el compilador; el pase de ojo la marca
  // (transientViolated) para que el Store forzado no se pierda con el tile. La de un pase que no es de ojo, no.
  {
    Ctx c;
    Device* d = from<Device>(c.dev);
    vpgpu::TextureDesc td;
    td.format = vpgpu::Format::D32Float;
    td.width = 1920, td.height = 1824;
    td.usage = vpgpu::usage::depthTarget;
    td.transient = true;
    const vpgpu::TextureId eyeDepth = c.gpu().createTexture(td), otherDepth = c.gpu().createTexture(td);
    CHECK(eyeDepth && otherDepth && !c.gpu().resources().transientViolated(eyeDepth));
    auto pass = [&](vpgpu::TextureId depthTex, uint32_t w) {
      UiMaskState um;
      Recorder rec;
      OpBeginRendering scene;
      scene.area = {{0, 0}, {w, w == 1920 ? 1824u : 64u}};
      scene.depthWidth = w, scene.depthHeight = w == 1920 ? 1824u : 64u;
      scene.desc.colors[0].texture = color;
      scene.desc.depth.texture = depthTex, scene.desc.depth.load = vpgpu::LoadAction::Clear;
      scene.desc.depth.store = vpgpu::StoreAction::DontCare;
      rec.ops.emplace_back(scene), rec.ops.emplace_back(opaque), rec.ops.emplace_back(OpEndPass{});
      vpgpu::Frame f;
      TranslateStats st;
      translate(rec, f, st, &j, &um, nullptr, nullptr, d);
      return st.passes;
    };
    CHECK(pass(otherDepth, 64) == 1);
    CHECK(!c.gpu().resources().transientViolated(otherDepth));
    CHECK(pass(eyeDepth, 1920) == 1);
    CHECK(c.gpu().resources().transientViolated(eyeDepth));
    c.gpu().destroy(eyeDepth);
    c.gpu().destroy(otherDepth);
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
constexpr uint64_t kCorpusDigest = 0x9293a3ef024461d7ull;  // the "corpus digest" line the test prints

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

  // robustBuffers: the checks against the buffer sizes (vp_args.bs<N>) are in the MSL by default and gone with it off.
  {
    const std::vector<uint32_t> w = spirvFile("store_f.comp.spv");
    std::string error;
    std::shared_ptr<spirv::SpirvModule> m(spirv::parse(w.data(), w.size(), error), spirv::release);
    CHECK(m != nullptr);
    if (m) {
      spirv::SpirvLowerOptions opt;
      opt.entry = "main", opt.stage = vpir::Stage::Compute;
      for (uint32_t i = 0; i < 4; ++i) {
        spirv::SpirvBinding b;
        b.binding = i, b.slot = uint16_t(i + 1);
        opt.layout.push_back(b);
      }
      uint64_t idOn = 0, idOff = 0;
      ShaderBlob on, off;
      CHECK(translateRecipeStage(*m, opt, idOn, on, error, nullptr));
      opt.robustBuffers = false;
      CHECK(translateRecipeStage(*m, opt, idOff, off, error, nullptr));
      CHECK(on.source.find("vp_args.bs") != std::string::npos);
      CHECK(off.source.find("vp_args.bs") == std::string::npos && idOn != idOff);
    }
  }

  // VPVK_ROBUST is read again at refreshRobustMode(), not frozen at the first use (the host sets it after the engine ran).
  {
    const char* old = std::getenv("VPVK_ROBUST");
    const std::string saved = old ? old : "";
    const int app = appRobustBufferAccess();
    unsetenv("VPVK_ROBUST");
    refreshRobustMode();
    CHECK(!std::strcmp(robustMode(), "default") && robustBuffersEffective());
    setenv("VPVK_ROBUST", "0", 1);
    CHECK(!std::strcmp(robustMode(), "default"));  // the host's setenv is picked up at the next refresh
    refreshRobustMode();
    CHECK(!std::strcmp(robustMode(), "0") && !robustBuffersEffective());
    setenv("VPVK_ROBUST", "app", 1);
    refreshRobustMode();
    setAppRobustBufferAccess(false);
    CHECK(!std::strcmp(robustMode(), "app") && !robustBuffersEffective());
    setAppRobustBufferAccess(true);
    CHECK(robustBuffersEffective());
    if (old) setenv("VPVK_ROBUST", saved.c_str(), 1);
    else unsetenv("VPVK_ROBUST");
    refreshRobustMode();
    setAppRobustBufferAccess(app > 0);
  }

  // Harness dump filter: comma and/or space separated hex ids, invalid tokens ignored, empty = not defined.
  {
    auto env = [](const char* name) { const char* e = std::getenv(name); return std::pair<bool, std::string>(e != nullptr, e ? e : ""); };
    const auto oldDir = env("VPVK_DUMP_MSL"), oldFilter = env("VPVK_DUMP_MSL_FILTER");
    auto hit = [](uint64_t id) { std::string d; return dumpDirFor(id, d); };
    unsetenv("VPVK_DUMP_MSL");
    unsetenv("VPVK_DUMP_MSL_FILTER");
    CHECK(!hit(0xabc));  // nothing defined
    setenv("VPVK_DUMP_MSL_FILTER", "", 1);
    CHECK(!hit(0xabc));  // empty filter and no dir: as not defined
    setenv("VPVK_DUMP_MSL", "/tmp/vpvk_dump_test", 1);
    CHECK(hit(0xabc) && hit(0x1));  // empty filter with a dir: everything
    setenv("VPVK_DUMP_MSL_FILTER", " , ", 1);
    CHECK(hit(0x1));  // only separators: still empty
    setenv("VPVK_DUMP_MSL_FILTER", "ABC,0x123 zz 12zz,,deadbeefcafef00d", 1);
    CHECK(hit(0xabc) && hit(0x123) && hit(0xdeadbeefcafef00dull));
    CHECK(!hit(0x12) && !hit(0x1) && !hit(0));  // "zz" / "12zz" are ignored, not read as 0 or 0x12
    setenv("VPVK_DUMP_MSL_FILTER", "11111111111111111", 1);  // 17 digits: invalid
    CHECK(!hit(0x1111111111111111ull));
    std::string d;
    CHECK(dumpDirFor(0xdeadbeefcafef00dull, d) == false && d.empty());
    setenv("VPVK_DUMP_MSL_FILTER", "7", 1);
    CHECK(dumpDirFor(7, d) && d == "/tmp/vpvk_dump_test");
    auto restore = [](const char* name, const std::pair<bool, std::string>& v) {
      if (v.first) setenv(name, v.second.c_str(), 1);
      else unsetenv(name);
    };
    restore("VPVK_DUMP_MSL", oldDir);
    restore("VPVK_DUMP_MSL_FILTER", oldFilter);
  }

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
  CHECK(h == 0x1480060cfddfc187ull);  // default options (robust, same loop guard): the key earlier builds wrote to recipes.bin
  StageInput same = in;  // an equal input (another object, e.g. another run) has the same key
  CHECK(same.hash() == h);
  StageInput other = in;
  other.opt.viewCount = 2;  // an option that changes the output changes the key
  CHECK(other.hash() != h);
  StageInput single = in;
  single.opt.multisampleAsSingle = true;
  CHECK(single.hash() != h);  // ... which is why translateStage clears it for modules without multisample images
  StageInput noRobust = in;
  noRobust.opt.robustBuffers = false;  // VPVK_ROBUST=0 / app: other MSL, so other key (the default one is unchanged, above)
  CHECK(noRobust.hash() != h && noRobust.hash() != other.hash());
  {
    RecipeStore s(dir);
    ShaderBlob b;
    b.source = "// msl";
    StageIface f;
    s.addStage(in, h, cull, 0x1234, b, &f);
    s.addStage(noRobust, noRobust.hash(), cull, 0x5678, b, &f);
  }
  {
    RecipeStore s(dir);
    RecipeStore::StageHit hit;
    RecipeStore::MissInfo miss;
    CHECK(s.lookupStage(h, hit, &miss, in.spirv) && hit.mslId == 0x1234);
    CHECK(s.lookupStage(noRobust.hash(), hit, &miss, in.spirv) && hit.mslId == 0x5678);
    int off = 0;  // the option survives the file: a retranslation keeps it
    for (const StageInput& si : s.stageInputs()) off += !si.opt.robustBuffers;
    CHECK(s.stageInputs().size() == 2 && off == 1);
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
  // Asynchronous is the default for every client; VPVK_ASYNC_PIPELINES=0 is the synchronous opt-in. No second device
  // here: creating one while `c` is alive hung the Metal CI run; the flag is decided by persistWaitMs(bool) and
  // asyncPipelinesOn(), checked on their own.
  unsetenv("VPVK_ASYNC_PIPELINES");
  unsetenv("VPVK_GLES_SYNC_PIPELINES");
  CHECK(asyncPipelinesOn());
  setenv("VPVK_ASYNC_PIPELINES", "1", 1);
  CHECK(asyncPipelinesOn());
  CHECK(persistWaitMs(false) == 300 && persistWaitMs(true) == 300);  // the env var wins for both
  setenv("VPVK_ASYNC_PIPELINES", "0", 1);
  CHECK(!asyncPipelinesOn());
  // The internal client (vpgles) ignores =0: asynchronous unless VPVK_GLES_SYNC_PIPELINES=1.
  vpvk_internal_scope(1);
  CHECK(asyncPipelinesOn());
  setenv("VPVK_GLES_SYNC_PIPELINES", "1", 1);
  CHECK(!asyncPipelinesOn());
  unsetenv("VPVK_GLES_SYNC_PIPELINES");
  vpvk_internal_scope(0);
  CHECK(!asyncPipelinesOn());  // a game thread is not affected by the internal scope
}

// ---- synchronous creation, opt-in with VPVK_ASYNC_PIPELINES=0 (pipelines.cpp "compile policy") ---------------------------------------------
// vkCreateGraphicsPipelines does not return while a pipeline of the batch is still compiling, and with
// VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT a pipeline nobody compiled is not requested at all.
void testSyncPipelines() {
  setenv("VPVK_ASYNC_PIPELINES", "0", 1);  // main() pins it for every group already; explicit here
  unsetenv("VPVK_REPORT_GEOM");
  unsetenv("VP_PIPELINE_CACHE_DIR");  // no recipe store: no pipeline is "known" beforehand
  CHECK(!asyncPipelinesOn());
  Ctx c;
  Device* d = from<Device>(c.dev);
  CHECK(d->translator == nullptr);
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
  VkPipelineRasterizationStateCreateInfo rs[3];  // cull none / back / front: three distinct core keys
  const VkCullModeFlags cull[3] = {VK_CULL_MODE_NONE, VK_CULL_MODE_BACK_BIT, VK_CULL_MODE_FRONT_BIT};
  for (int i = 0; i < 3; ++i) {
    rs[i] = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs[i].lineWidth = 1, rs[i].cullMode = cull[i];
  }
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
  VkGraphicsPipelineCreateInfo gp[3];
  for (int i = 0; i < 3; ++i) {
    gp[i] = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp[i].stageCount = 2, gp[i].pStages = stg, gp[i].pVertexInputState = &vis, gp[i].pInputAssemblyState = &ias;
    gp[i].pViewportState = &vps, gp[i].pRasterizationState = &rs[i], gp[i].pMultisampleState = &ms;
    gp[i].pColorBlendState = &cbs, gp[i].layout = pl, gp[i].renderPass = rp;
  }
  vpgpu::NullBackend& nb = static_cast<vpgpu::NullBackend&>(d->platform->backend());
  vpgpu::PipelineCache& cache = c.gpu().pipelines();
  PipelineCounters& pc = pipelineCounters();
  const uint64_t timed0 = pc.createTimedOut.load();

  // 1. Every compile of the batch is held for 80 ms: the call returns after it, with each pipeline Ready.
  std::atomic<bool> gate{false};
  nb.onCompile = [&] {
    while (!gate) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  };
  const auto t0 = std::chrono::steady_clock::now();
  std::thread opener([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    gate = true;
  });
  VkPipeline batch[2] = {};
  CHECK(vkCreateGraphicsPipelines(c.dev, VK_NULL_HANDLE, 2, gp, nullptr, batch) == VK_SUCCESS);
  CHECK(std::chrono::steady_clock::now() - t0 >= std::chrono::milliseconds(75));
  opener.join();
  for (VkPipeline p : batch) {
    const PipelineObj* po = p ? from<PipelineObj>(p) : nullptr;
    CHECK(po && po->ok);
    if (po && po->ok) CHECK(cache.status(po->key) == vpgpu::PipelineCache::Status::Ready);
  }
  CHECK(pc.createTimedOut.load() == timed0);
  nb.onCompile = nullptr;

  // 2. FAIL_ON_PIPELINE_COMPILE_REQUIRED: cull front was never created, so it is refused and nothing is requested.
  VkGraphicsPipelineCreateInfo fail = gp[2];
  fail.flags = VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT;
  const uint32_t compiled0 = cache.totals().compiled;
  VkPipeline refused = batch[0];  // overwritten with VK_NULL_HANDLE
  CHECK(vkCreateGraphicsPipelines(c.dev, VK_NULL_HANDLE, 1, &fail, nullptr, &refused) == VK_PIPELINE_COMPILE_REQUIRED);
  CHECK(refused == VK_NULL_HANDLE);
  cache.waitIdle();
  CHECK(cache.totals().compiled == compiled0);
  // A batch: the pipeline the cache holds is created, the other one is refused (the call reports the error).
  VkGraphicsPipelineCreateInfo mixed[2] = {gp[1], fail};
  mixed[0].flags = VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT;
  VkPipeline mix[2] = {};
  CHECK(vkCreateGraphicsPipelines(c.dev, VK_NULL_HANDLE, 2, mixed, nullptr, mix) == VK_PIPELINE_COMPILE_REQUIRED);
  CHECK(mix[0] != VK_NULL_HANDLE && mix[1] == VK_NULL_HANDLE);
  cache.waitIdle();
  CHECK(cache.totals().compiled == compiled0);
  // Without the flag it is compiled; asked again with the flag it is known now.
  VkPipeline front = VK_NULL_HANDLE, again = VK_NULL_HANDLE;
  CHECK(vkCreateGraphicsPipelines(c.dev, VK_NULL_HANDLE, 1, &gp[2], nullptr, &front) == VK_SUCCESS);
  CHECK(front && cache.totals().compiled == compiled0 + 1);
  CHECK(vkCreateGraphicsPipelines(c.dev, VK_NULL_HANDLE, 1, &fail, nullptr, &again) == VK_SUCCESS);
  CHECK(again != VK_NULL_HANDLE);

  for (VkPipeline p : {batch[0], batch[1], mix[0], front, again}) vkDestroyPipeline(c.dev, p, nullptr);
  vkDestroyShaderModule(c.dev, vs, nullptr);
  vkDestroyShaderModule(c.dev, fs, nullptr);
  vkDestroyPipelineLayout(c.dev, pl, nullptr);
  vkDestroyRenderPass(c.dev, rp, nullptr);
  setenv("VPVK_ASYNC_PIPELINES", "0", 1);
}

// vk.pso.census: the counters the creation calls and awaitPipelines move (created, translation failures, timeouts).
namespace vpvk {
unsigned awaitPipelines(Device*, const std::vector<vpgpu::PipelineKey>&, uint32_t, const char*);  // pipelines.cpp
}
void testPsoCensus() {
  Ctx c;
  Device* d = from<Device>(c.dev);
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
  VkPipelineShaderStageCreateInfo stg[3]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                         {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                         {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
  stg[0].stage = VK_SHADER_STAGE_VERTEX_BIT, stg[0].module = module(c, "uimask.vert.spv"), stg[0].pName = "main";
  stg[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT, stg[1].module = module(c, "uimask.frag.spv"), stg[1].pName = "main";
  stg[2] = stg[1];
  stg[2].stage = VK_SHADER_STAGE_GEOMETRY_BIT;  // unsupported: the pipeline comes back not ok
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

  PipelineCounters& pc = pipelineCounters();
  const uint64_t created0 = pc.pipelinesCreated.load(), failed0 = pc.translateFailed.load(), timed0 = pc.createTimedOut.load();
  VkPipeline good = VK_NULL_HANDLE, bad = VK_NULL_HANDLE;
  CHECK(vkCreateGraphicsPipelines(c.dev, VK_NULL_HANDLE, 1, &gp, nullptr, &good) == VK_SUCCESS);
  CHECK(good && from<PipelineObj>(good)->ok);
  CHECK(pc.pipelinesCreated.load() - created0 == 1 && pc.translateFailed.load() == failed0 && pc.createTimedOut.load() == timed0);
  gp.stageCount = 3;
  CHECK(vkCreateGraphicsPipelines(c.dev, VK_NULL_HANDLE, 1, &gp, nullptr, &bad) == VK_SUCCESS);  // always VK_SUCCESS
  CHECK(bad && !from<PipelineObj>(bad)->ok);
  CHECK(pc.pipelinesCreated.load() - created0 == 2 && pc.translateFailed.load() - failed0 == 1 && pc.createTimedOut.load() == timed0);

  // A compile that stays held past the budget: awaitPipelines reports the pipeline still pending, then none once it is done.
  vpgpu::NullBackend& nb = static_cast<vpgpu::NullBackend&>(d->platform->backend());
  std::atomic<bool> gate{false};
  nb.onCompile = [&] {
    while (!gate) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  };
  vpgpu::PipelineState held = from<PipelineObj>(good)->state;
  held.cull = vpgpu::CullMode::Front;  // a state nobody compiled yet
  const vpgpu::PipelineKey k = held.key();
  c.gpu().pipelines().request(held, k, vpgpu::Priority::Visible);
  CHECK(awaitPipelines(d, {k}, 40, "test") == 1);
  gate = true;
  c.gpu().pipelines().waitIdle();
  nb.onCompile = nullptr;
  CHECK(awaitPipelines(d, {k}, 40, "test") == 0);

  Platform::ArchiveCensus a;
  CHECK(!d->platform->archiveCensus(a));  // the null platform has no archive
  psoCensus(d, true);                     // prints one vk.pso.census line
  vkDestroyPipeline(c.dev, good, nullptr);
  vkDestroyPipeline(c.dev, bad, nullptr);
}

// vpvk_cmd_pass_drop: the host (vpgles' glInvalidateFramebuffer) learns after vkCmdBeginRendering that an attachment is
// undefined at the end of the pass. Plain stores become DontCare (and back, the call replaces the earlier one); resolve
// targets, areas smaller than the attachment, aspects not attached and eye images keep their store.
void testPassDrop() {
  Ctx c;
  VkDeviceMemory m = alloc(c, 64 << 20, 0);
  const VkImageUsageFlags cu = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  const VkImageUsageFlags du = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  VkImage col = image(c, VK_FORMAT_R8G8B8A8_UNORM, 64, 1, cu, m, VK_SAMPLE_COUNT_2_BIT);  // MSAA: a 1x resolve is a copy
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

// Suspend/resume (Source 2's eye pass): the chunks are ONE render pass instance. Every chunk repeats loadOp CLEAR
// and storeOp DONT_CARE, as Vulkan requires; only the first chunk may clear, every chunk but the last must store
// (and not resolve), and the resolve happens at the last chunk.
void testSuspendResume() {
  Ctx c;
  VkDeviceMemory m = alloc(c, 64 << 20, 0);
  const VkImageUsageFlags cu = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  const VkImageUsageFlags du = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  VkImage col = image(c, VK_FORMAT_R8G8B8A8_UNORM, 64, 2, cu, m, VK_SAMPLE_COUNT_2_BIT);  // MSAA: a 1x resolve is a copy
  VkImage col2 = image(c, VK_FORMAT_R8G8B8A8_UNORM, 64, 2, cu, m);
  VkImage dep = image(c, VK_FORMAT_D32_SFLOAT, 64, 2, du, m);
  CHECK(col && col2 && dep);
  CHECK(from<Image>(col)->desc.samples == 2);
  auto view = [&](VkImage i, VkFormat f, VkImageAspectFlags a) {
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = i, vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY, vi.format = f;
    vi.subresourceRange = {a, 0, 1, 0, 2};
    VkImageView v = VK_NULL_HANDLE;
    vkCreateImageView(c.dev, &vi, nullptr, &v);
    return v;
  };
  VkImageView cv = view(col, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
  VkImageView rv = view(col2, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
  VkImageView dv = view(dep, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);
  CHECK(cv && rv && dv);
  VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  VkCommandPool pool;
  vkCreateCommandPool(c.dev, &pi, nullptr, &pool);
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = pool, ai.commandBufferCount = 1;
  VkCommandBuffer cb;
  vkAllocateCommandBuffers(c.dev, &ai, &cb);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(cb, &bi);
  auto beginOp = [&]() -> OpBeginRendering& {
    Recorder& r = *from<CommandBuffer>(cb)->rec;
    return std::get<OpBeginRendering>(r.ops[r.passBegin]);
  };
  VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  ca.imageView = cv, ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  da.imageView = dv, da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, da.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, {64, 64}}, ri.layerCount = 1, ri.viewMask = 0x3;
  ri.colorAttachmentCount = 1, ri.pColorAttachments = &ca, ri.pDepthAttachment = &da;
  const VkRenderingFlags chunks[3] = {VK_RENDERING_SUSPENDING_BIT,
                                      VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT, VK_RENDERING_RESUMING_BIT};
  const auto LOAD = vpgpu::LoadAction::Load, CLEAR = vpgpu::LoadAction::Clear;
  const auto STORE = vpgpu::StoreAction::Store;
  for (int i = 0; i < 3; ++i) {
    ri.flags = chunks[i];
    vkCmdBeginRendering(cb, &ri);
    const vpgpu::RenderPassDesc& d = beginOp().desc;
    CHECK(d.colors[0].load == (i == 0 ? CLEAR : LOAD) && d.depth.load == (i == 0 ? CLEAR : LOAD));
    if (i < 2) CHECK(d.colors[0].store == STORE && d.depth.store == STORE);
    else CHECK(d.colors[0].store == STORE && d.depth.store == vpgpu::StoreAction::DontCare);  // the instance's own storeOp
    if (i == 1) CHECK(beginOp().areaClear == nullptr);  // a resumed chunk never clears, not even with quads
    vkCmdEndRendering(cb);
  }
  // A resolve happens when the instance ends: not at a suspension, whatever the storeOp.
  ca.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT, ca.resolveImageView = rv, ca.resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL;
  ca.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  for (int i = 0; i < 3; ++i) {
    ri.flags = chunks[i];
    vkCmdBeginRendering(cb, &ri);
    const vpgpu::RenderPassDesc& d = beginOp().desc;
    CHECK(bool(d.colors[0].resolve) == (i == 2));
    CHECK(d.colors[0].store == (i < 2 ? STORE : vpgpu::StoreAction::Resolve));
    vkCmdEndRendering(cb);
  }
  // Without the flags nothing changes: the pass clears and follows its storeOp.
  ri.flags = 0;
  vkCmdBeginRendering(cb, &ri);
  CHECK(beginOp().desc.colors[0].load == CLEAR && beginOp().desc.depth.store == vpgpu::StoreAction::DontCare);
  vkCmdEndRendering(cb);
  vkEndCommandBuffer(cb);

  // Source 2 with vr_msaa 0 "resolves" its 1x eye depth into a second image (multiview, suspended and resumed chunks). Metal
  // has no 1x resolve (a resolve store of a 1x texture is undefined: the eye passes hung the GPU): the attachment is stored and
  // copied into the target once, after the chunk that ends the render pass instance. vkCmdResolveImage of a 1x image is a copy too.
  {
    VkImage dst = image(c, VK_FORMAT_D32_SFLOAT, 64, 2, du, m);
    CHECK(dst);
    VkImageView dstv = view(dst, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);
    VkCommandBuffer cb2;
    vkAllocateCommandBuffers(c.dev, &ai, &cb2);
    vkBeginCommandBuffer(cb2, &bi);
    VkRenderingAttachmentInfo dr{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    dr.imageView = dv, dr.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, dr.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    dr.resolveMode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT, dr.resolveImageView = dstv;
    VkRenderingInfo rd{VK_STRUCTURE_TYPE_RENDERING_INFO};
    rd.renderArea = {{0, 0}, {64, 64}}, rd.layerCount = 1, rd.viewMask = 0x3, rd.pDepthAttachment = &dr;
    rd.flags = VK_RENDERING_SUSPENDING_BIT;
    vkCmdBeginRendering(cb2, &rd);
    vkCmdEndRendering(cb2);
    rd.flags = VK_RENDERING_RESUMING_BIT;
    vkCmdBeginRendering(cb2, &rd);
    vkCmdEndRendering(cb2);
    vkEndCommandBuffer(cb2);
    vpgpu::Frame frame;
    TranslateStats st;
    translate(*from<CommandBuffer>(cb2)->rec, frame, st);
    uint32_t passes = 0, copies = 0;
    bool resolvesInMetal = false, copyOk = false;
    for (const auto& cmd : frame.commands()) {
      if (const auto* p = std::get_if<vpgpu::RenderPass>(&cmd)) {
        ++passes;
        resolvesInMetal = resolvesInMetal || bool(p->desc.depth.resolve) || p->desc.depth.store == vpgpu::StoreAction::Resolve ||
                          p->desc.depth.store == vpgpu::StoreAction::StoreAndResolve;
      } else if (const auto* cp = std::get_if<vpgpu::Copy>(&cmd)) {
        ++copies;
        copyOk = cp->kind == vpgpu::CopyKind::TextureToTexture && cp->srcTexture.texture == from<Image>(dep)->id &&
                 cp->dstTexture.texture == from<Image>(dst)->id && cp->srcTexture.layerCount == 2;
      }
    }
    CHECK(passes == 2 && !resolvesInMetal);
    CHECK(copies == 1 && copyOk);
    VkCommandBuffer cb3;
    vkAllocateCommandBuffers(c.dev, &ai, &cb3);
    vkBeginCommandBuffer(cb3, &bi);
    VkImageResolve res{};
    res.srcSubresource = res.dstSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 2};
    res.extent = {64, 64, 1};
    vkCmdResolveImage(cb3, dep, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &res);
    vkEndCommandBuffer(cb3);
    vpgpu::Frame f2;
    TranslateStats s2;
    translate(*from<CommandBuffer>(cb3)->rec, f2, s2);
    bool anyPass = false;
    uint32_t copies2 = 0;
    for (const auto& cmd : f2.commands()) {
      anyPass = anyPass || std::holds_alternative<vpgpu::RenderPass>(cmd);
      copies2 += std::holds_alternative<vpgpu::Copy>(cmd) ? 1 : 0;
    }
    CHECK(!anyPass && copies2 == 1);
    // sRGB / UNORM variants of one format (mutable-format images) copy as before: still one copy, not a dropped resolve.
    VkImage srgb = image(c, VK_FORMAT_R8G8B8A8_SRGB, 64, 2, cu, m), unorm = image(c, VK_FORMAT_R8G8B8A8_UNORM, 64, 2, cu, m);
    CHECK(srgb && unorm);
    VkImageView sv = view(srgb, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
    VkImageView uv = view(unorm, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
    VkCommandBuffer cb4;
    vkAllocateCommandBuffers(c.dev, &ai, &cb4);
    vkBeginCommandBuffer(cb4, &bi);
    VkRenderingAttachmentInfo cx{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    cx.imageView = sv, cx.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, cx.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    cx.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT, cx.resolveImageView = uv;
    VkRenderingInfo rx{VK_STRUCTURE_TYPE_RENDERING_INFO};
    rx.renderArea = {{0, 0}, {64, 64}}, rx.layerCount = 1, rx.viewMask = 0x3, rx.colorAttachmentCount = 1, rx.pColorAttachments = &cx;
    vkCmdBeginRendering(cb4, &rx);
    vkCmdEndRendering(cb4);
    vkEndCommandBuffer(cb4);
    vpgpu::Frame f3;
    TranslateStats s3;
    translate(*from<CommandBuffer>(cb4)->rec, f3, s3);
    uint32_t copies3 = 0;
    for (const auto& cmd : f3.commands()) copies3 += std::holds_alternative<vpgpu::Copy>(cmd) ? 1 : 0;
    CHECK(copies3 == 1);
    vkDestroyImageView(c.dev, sv, nullptr);
    vkDestroyImageView(c.dev, uv, nullptr);
    vkDestroyImage(c.dev, srgb, nullptr);
    vkDestroyImage(c.dev, unorm, nullptr);
    vkDestroyImageView(c.dev, dstv, nullptr);
    vkDestroyImage(c.dev, dst, nullptr);
  }
  vkDestroyCommandPool(c.dev, pool, nullptr);
  for (VkImageView v : {cv, rv, dv}) vkDestroyImageView(c.dev, v, nullptr);
  for (VkImage i : {col, col2, dep}) vkDestroyImage(c.dev, i, nullptr);
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

  // Indirect arguments read at submit (VPVK_GPU_INDIRECT=0, read once per process: no earlier group here records an indirect
  // command): a record with a garbage count or instance count is not drawn (it would trip the GPU watchdog), the others are.
  setenv("VPVK_GPU_INDIRECT", "0", 1);
  {
    VkDeviceMemory am = alloc(c, 1 << 16, 1);  // host-visible
    VkBuffer ab = buffer(c, 4096, am, 0);
    void* ap = nullptr;
    CHECK(vkMapMemory(c.dev, am, 0, VK_WHOLE_SIZE, 0, &ap) == VK_SUCCESS);
    const uint32_t rec[4][4] = {{3, 1, 0, 0}, {0x7fffffffu, 1, 0, 0}, {3, 0x7fffffffu, 0, 0}, {6, 2, 0, 0}};
    std::memcpy(ap, rec, sizeof rec);
    vkUnmapMemory(c.dev, am);
    rig.begin();
    vkCmdBindPipeline(rig.cb, VK_PIPELINE_BIND_POINT_GRAPHICS, scissored);
    vkCmdSetScissor(rig.cb, 0, 1, &fine);
    vkCmdDrawIndirect(rig.cb, ab, 0, 4, 16);
    vpgpu::Frame f2;
    TranslateStats s2;
    const std::vector<vpgpu::Draw> drawn = rig.finish(f2, s2);
    CHECK(drawn.size() == 2 && drawn[0].count == 3 && drawn[1].count == 6 && drawn[1].instances == 2);
    CHECK(s2.draws == 2 && s2.skippedBy[size_t(SkipReason::Indirect)] == 2);
    vkDestroyBuffer(c.dev, ab, nullptr);
    vkFreeMemory(c.dev, am, nullptr);
  }
  unsetenv("VPVK_GPU_INDIRECT");

  // Dual-source blending: the second-source factors stay only when the fragment shader writes index 1 (attachment 0);
  // without it they fall back to the first source (Metal rejects the pipeline otherwise).
  {
    VkShaderModule dfs = module(c, "dual_source.frag.spv");
    PipeOpts o;
    o.src1 = true;
    VkPipeline dual = makePosPipeline(c, rig, pl, o, vs, dfs);
    VkPipeline plain = makePosPipeline(c, rig, pl, o, vs, fs);
    const PipelineObj* dp = from<PipelineObj>(dual);
    const PipelineObj* pp = from<PipelineObj>(plain);
    CHECK(dp->ok && pp->ok && dp->dualSource && !pp->dualSource);
    using B = vpgpu::BlendFactor;
    const vpgpu::BlendState& db = dp->state.blend[0];
    CHECK(db.srcColor == B::Src1Color && db.dstColor == B::OneMinusSrc1Color && db.srcAlpha == B::Src1Alpha &&
          db.dstAlpha == B::OneMinusSrc1Alpha);
    const vpgpu::BlendState& pb = pp->state.blend[0];
    CHECK(pb.srcColor == B::SrcColor && pb.dstColor == B::OneMinusSrcColor && pb.srcAlpha == B::SrcAlpha &&
          pb.dstAlpha == B::OneMinusSrcAlpha);
    vkDestroyPipeline(c.dev, dual, nullptr), vkDestroyPipeline(c.dev, plain, nullptr);
    vkDestroyShaderModule(c.dev, dfs, nullptr);
  }

  vkDestroyPipeline(c.dev, scissored, nullptr), vkDestroyPipeline(c.dev, both, nullptr);

  // Dynamic viewport/scissor "with count" (extended dynamic state, core 1.3): the pipeline's static rectangle (here
  // empty, ignored by the spec) must not replace a scissor set before the bind, and a pass whose render area is 0x0
  // (as seen from some guests) does not empty a valid scissor.
  PipeOpts wc;
  wc.dynamic = {VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT, VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT};
  wc.scissor = {{0, 0}, {0, 0}};
  VkPipeline withCount = makePosPipeline(c, rig, pl, wc, vs, fs);
  CHECK(from<PipelineObj>(withCount)->ok);
  c.gpu().pipelines().waitIdle();
  const VkRect2D area0{{0, 0}, {0, 0}};
  for (const VkRect2D& area : {VkRect2D{{0, 0}, {64, 64}}, area0}) {
    rig.begin(area);
    vkCmdSetScissorWithCount(rig.cb, 1, &fine);  // before the bind
    vkCmdBindPipeline(rig.cb, VK_PIPELINE_BIND_POINT_GRAPHICS, withCount);
    vkCmdDraw(rig.cb, 3, 1, 0, 0);
    vpgpu::Frame f2;
    TranslateStats ts2;
    const auto draws = rig.finish(f2, ts2);
    CHECK(ts2.skippedDraws == 0 && draws.size() == 1 && draws[0].scissor.width == 32 && draws[0].scissor.height == 32);
  }
  vkDestroyPipeline(c.dev, withCount, nullptr);

  // The empty_scissor log names the pass the draw belongs to: its render area and the framebuffer (render pass) or
  // first colour attachment (dynamic rendering) extent, kept in the recorder's bound state.
  {
    rig.begin({{1, 2}, {30, 40}});
    const BoundState& b = from<CommandBuffer>(rig.cb)->rec->bound;
    CHECK(b.renderArea.offset.x == 1 && b.renderArea.offset.y == 2 && b.renderArea.extent.width == 30 &&
          b.renderArea.extent.height == 40);
    CHECK(b.fbWidth == 64 && b.fbHeight == 64 && b.fbLayers == 1);
    vkCmdEndRenderPass(rig.cb);
    VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    ca.imageView = rig.views[0], ca.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{3, 4}, {20, 10}};
    ri.layerCount = 1, ri.colorAttachmentCount = 1, ri.pColorAttachments = &ca;
    vkCmdBeginRendering(rig.cb, &ri);
    CHECK(b.renderArea.offset.x == 3 && b.renderArea.offset.y == 4 && b.renderArea.extent.width == 20 &&
          b.renderArea.extent.height == 10);
    CHECK(b.fbWidth == 64 && b.fbHeight == 64 && b.fbLayers == 1);
    vkCmdEndRendering(rig.cb);
    vkEndCommandBuffer(rig.cb);
  }
  vkDestroyPipelineLayout(c.dev, pl, nullptr);
}

// vk.perf (perf.h): the draw counters of the replay (instanced, indirect by where the arguments are read, secondary), the
// once-per-second line and the wait timer.
void testPerf() {
  auto rendering = [] {
    OpBeginRendering b;
    b.area = {{0, 0}, {64, 64}};
    return b;
  };
  auto direct = [](uint32_t instances, uint32_t views) {
    OpDraw d;
    d.draw.count = 3, d.draw.instances = instances * views, d.viewCount = views;  // makeDraw multiplies by the views
    return d;
  };
  DeviceMemory mem;
  uint8_t host[64] = {};
  mem.size = sizeof host, mem.host = host;
  Buffer args;
  args.size = sizeof host, args.memory = &mem;
  const uint32_t records[8] = {3, 1, 0, 0, 3, 9, 0, 0};  // two non-indexed records: 1 instance, 9 instances
  std::memcpy(host, records, sizeof records);
  auto cpuIndirect = [&](uint32_t n) {
    OpDraw d;
    d.indirect = &args, d.indirectCount = n, d.indirectStride = 16;
    return d;
  };
  Recorder secondary;
  secondary.ops.emplace_back(cpuIndirect(2));
  Recorder rec;
  rec.ops.emplace_back(rendering());
  rec.ops.emplace_back(direct(1, 1));  // not instanced
  rec.ops.emplace_back(direct(5, 1));  // instanced
  rec.ops.emplace_back(direct(1, 2));  // multiview, one app instance: not instanced
  rec.ops.emplace_back(direct(3, 2));  // multiview, three app instances: instanced
  rec.ops.emplace_back(cpuIndirect(2));  // arguments read at submit: 2 records, the second instanced
  OpDraw gpu = cpuIndirect(0);
  gpu.gpuIndirect = true, gpu.indirectCount = 3;
  rec.ops.emplace_back(gpu);  // arguments read by the GPU: counted, instances unknown
  rec.ops.emplace_back(OpExecute{&secondary});  // a secondary command buffer's CPU-read indirect draw
  rec.ops.emplace_back(OpEndPass{});
  vpgpu::Frame frame;
  TranslateStats st;
  translate(rec, frame, st);
  CHECK(st.instancedDraws == 2 + 1 + 1);  // direct x2, CPU-read records x1 (first pass), CPU-read in the secondary x1
  CHECK(st.indirectGpu == 3);
  CHECK(st.indirectCpu == 4 && st.indirectCpuSecondary == 2);  // 2 in the primary, 2 more through the secondary
  CHECK(st.draws == 4 + 2 + 3 + 2);  // direct, CPU-read, GPU-read, CPU-read in the secondary

  // The line: baseline, then nothing before a second, then the interval's counters and the delta of the cumulative ones.
  {
    PerfCounters pc;
    vpgpu::GpuPerf g;
    PerfLine line;
    PerfSample ext;
    const auto t0 = PerfLine::Clock::now();
    CHECK(line.tick(t0, ext, pc, g).empty());
    pc.frames = 72, pc.submitUs = 72 * 3000, pc.submitUsMax = 9000, pc.dispatches = 5, pc.draws = 900, pc.drawsInstanced = 40;
    pc.indirectGpu = 7, pc.indirectCpu = 2, pc.indirectCpuSecondary = 1, pc.barriers = 30, pc.barriersComputeDraw = 4;
    pc.fenceWaits = 3, pc.fenceUs = 12000, pc.fenceUsMax = 6000;
    g.addCb(2000), g.addCb(4000);
    CHECK(line.tick(t0 + std::chrono::milliseconds(500), ext, pc, g).empty());  // too early: nothing consumed
    ext.psoGateUs = 25000, ext.psoHoldUs = 1500, ext.psoHardWaits = 2;
    const std::string s = line.tick(t0 + std::chrono::milliseconds(1000), ext, pc, g);
    CHECK(s.starts_with("vk.perf dt_ms=1000 frames=72 submit_ms{avg=3.00 max=9.00} "));
    CHECK(s.find(" gpu_cb{n=2 avg_ms=3.00 max_ms=4.00 sum_ms=6.0} ") != std::string::npos);
    CHECK(s.find(" dispatches=5 draws=900 draws_inst=40 ind{gpu=7 cpu=2 cpu_sec=1} barriers=30 barr_c2d=4 ") != std::string::npos);
    CHECK(s.find(" pso_wait_ms{create=0.0 record=0.0 gate=25.0 hold=1.5} hard_waits=2 ") != std::string::npos);
    CHECK(s.find(" fence_wait{n=3 ms=12.0 max=6.0} idle_wait{n=0 ms=0.0 max=0.0}") != std::string::npos);
    const std::string next = line.tick(t0 + std::chrono::milliseconds(2000), ext, pc, g);  // a clean interval
    CHECK(next.find(" frames=0 ") != std::string::npos && next.find(" gpu_cb{n=0 ") != std::string::npos &&
          next.find("gate=0.0 hold=0.0} hard_waits=0 ") != std::string::npos);
  }
  // Wait timer: a poll that returns at once is not a wait; a blocked one is counted with its longest.
  {
    PerfCounters pc;
    { PerfWait w(pc.fenceWaits, pc.fenceUs, pc.fenceUsMax); }
    CHECK(pc.fenceWaits == 0 && pc.fenceUs == 0);
    {
      PerfWait w(pc.fenceWaits, pc.fenceUs, pc.fenceUsMax);
      std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    CHECK(pc.fenceWaits == 1 && pc.fenceUs >= 3000 && pc.fenceUsMax == pc.fenceUs);
  }
  // vk.hitch: nothing under 40 ms, the per-frame delta of the cumulative counters over it, max 1 line / 250 ms, 400 per session.
  {
    vpgpu::HitchPerf hp;
    HitchLine h;
    const auto t0 = HitchLine::Clock::now();
    auto ms = [&](int n) { return t0 + std::chrono::milliseconds(n); };
    CHECK(h.frame(ms(0), {}, hp).empty());                        // baseline
    hp.psoCreateUs = 30000, hp.waitUs = 7000, hp.engineUs = 40000;  // before the slow frame: not part of an earlier interval
    CHECK(h.frame(ms(10), {}, hp).empty());                       // dt 10 ms: no line (its counters are consumed)
    hp.psoCreateUs += 12000, hp.libCompileUs += 3000, hp.waitUs += 5000, hp.allocUs += 2000, hp.glLinkUs += 1000;
    hp.lockUs += 4000, hp.engineUs += 25000;
    const std::string s = h.frame(ms(110), {77, 8000, 5000, 2500, 4096}, hp);  // dt 100 ms; engine 25 + 8 ms; gap 67
    CHECK(s == "vk.hitch submit=77 dt_ms=100.0 submit_ms=8.00 translate_ms=5.00 backend_ms=2.50 pso_create_ms=12.00 "
               "lib_compile_ms=3.00 fence_wait_ms=5.00 alloc_ms=2.00 upload_bytes=4096 gles_link_ms=1.00 lock_ms=4.00 "
               "guest_gap_ms=67.0");
    CHECK(h.frame(ms(200), {}, hp).empty());                      // dt 90 ms but 90 ms after the line: under the 250 ms gap
    CHECK(!h.frame(ms(400), {}, hp).empty());                     // dt 200 ms, 290 ms after the line
    CHECK(h.frame(ms(430), {}, hp).empty());                      // dt 30 ms
    // Every frame slow and 300 ms apart: the first 50 lines keep the 250 ms gap, then at most one per second and no total cap
    // (lines at frames 1..50, then every 4th frame from 54: 1200 ms apart is the first multiple of 300 over 1000).
    HitchLine cap;
    int lines = 0, early = 0;
    for (int i = 0; i < 600; ++i) {
      const bool line = !cap.frame(ms(i * 300), {}, hp).empty();
      lines += line;
      if (i <= 53) early += line;
    }
    CHECK(early == int(HitchLine::kFullLines));
    CHECK(lines == 50 + (599 - 54) / 4 + 1);
  }
  // A real submit goes through the counters.
  {
    Ctx c;
    VkDeviceMemory m = alloc(c, 1 << 20, 1);
    VkBuffer a = buffer(c, 4096, m, 0), b = buffer(c, 4096, m, 65536);
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    VkCommandPool pool;
    vkCreateCommandPool(c.dev, &pi, nullptr, &pool);
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool, ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    vkAllocateCommandBuffers(c.dev, &ai, &cb);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cb, &bi);
    VkBufferCopy bc{0, 0, 256};
    vkCmdCopyBuffer(cb, a, b, 1, &bc);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1, si.pCommandBuffers = &cb;
    // The line's own tick may take the counters at a second boundary (then they read 0): two submits, one of them shows.
    uint64_t seen = 0;
    for (int i = 0; i < 2; ++i) {
      CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS);
      seen = std::max(seen, perf().frames.load());
    }
    CHECK(seen >= 1);
    vkDestroyCommandPool(c.dev, pool, nullptr);
    vkFreeMemory(c.dev, m, nullptr);
  }
}

// Phase 1b, native identity: EDS3 colour write mask (a pipeline variant per mask set), alpha-to-coverage at 1x as a
// shader test, VK_EXT_memory_budget, VK_EXT_host_image_copy (staged copies waited for) and VK_KHR_dynamic_rendering_local_read
// (an in-pass barrier followed by a draw reading an attachment: split, copy, rebind).
void testPhase1b() {
  vpvk_config_set_quest_identity(0);
  unsetenv("VPVK_EXTRA_EXTENSIONS");
  {  // colour write mask + A2C
    Ctx c({VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME});
    CHECK(c.dev != nullptr);
    DrawRig rig(c);
    VkShaderModule vs = module(c, "pos.vert.spv"), fs = module(c, "solid.frag.spv");
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout pl;
    vkCreatePipelineLayout(c.dev, &plci, nullptr, &pl);
    VkDeviceMemory bm = alloc(c, 1 << 20, 0);
    VkBuffer vb = buffer(c, 4096, bm, 0);
    PipeOpts o;
    o.dynamic = {VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT};
    VkPipeline dyn = makePosPipeline(c, rig, pl, o, vs, fs);
    PipeOpts fixed;
    VkPipeline fix = makePosPipeline(c, rig, pl, fixed, vs, fs);
    const PipelineObj* po = from<PipelineObj>(dyn);
    CHECK(po->ok && (po->fragmentOutputs & 1));
    c.gpu().pipelines().waitIdle();
    auto setMask = reinterpret_cast<PFN_vkCmdSetColorWriteMaskEXT>(vkGetDeviceProcAddr(c.dev, "vkCmdSetColorWriteMaskEXT"));
    CHECK(setMask != nullptr);
    rig.begin();
    VkBuffer bufs[1] = {vb};
    VkDeviceSize offs[1] = {0};
    vkCmdBindPipeline(rig.cb, VK_PIPELINE_BIND_POINT_GRAPHICS, dyn);
    vkCmdBindVertexBuffers(rig.cb, 0, 1, bufs, offs);
    vkCmdDraw(rig.cb, 3, 1, 0, 0);  // 0: nothing set: the pipeline's own
    VkColorComponentFlags rgb = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT;
    setMask(rig.cb, 0, 1, &rgb);
    vkCmdDraw(rig.cb, 3, 1, 0, 0);  // 1: variant, mask 7
    vkCmdDraw(rig.cb, 3, 1, 0, 0);  // 2: the same variant
    VkColorComponentFlags all = rgb | VK_COLOR_COMPONENT_A_BIT;
    setMask(rig.cb, 0, 1, &all);
    vkCmdDraw(rig.cb, 3, 1, 0, 0);  // 3: the pipeline's own mask again: no variant
    vkCmdBindPipeline(rig.cb, VK_PIPELINE_BIND_POINT_GRAPHICS, fix);
    setMask(rig.cb, 0, 1, &rgb);
    vkCmdDraw(rig.cb, 3, 1, 0, 0);  // 4: a pipeline that does not list it as dynamic ignores it
    vpgpu::Frame frame;
    TranslateStats ts;
    const std::vector<vpgpu::Draw> d = rig.finish(frame, ts);
    CHECK(d.size() == 5);
    if (d.size() == 5) {
      CHECK(d[0].pipeline == po->key && d[3].pipeline == po->key);
      CHECK(d[1].pipeline != po->key && d[2].pipeline == d[1].pipeline);
      vpgpu::PipelineState want = po->state;
      want.blend[0].writeMask = 7;
      CHECK(d[1].pipeline == want.key());
      CHECK(d[4].pipeline == from<PipelineObj>(fix)->key);
    }
    vkDestroyPipeline(c.dev, dyn, nullptr);
    vkDestroyPipeline(c.dev, fix, nullptr);
    // Alpha-to-coverage on a single-sample pipeline: the shader tests alpha instead (hardware A2C would dither).
    PipeOpts ao;
    ao.a2c = true;
    VkPipeline a2c = makePosPipeline(c, rig, pl, ao, vs, fs);
    CHECK(from<PipelineObj>(a2c)->a2cTest && !from<PipelineObj>(a2c)->state.alphaToCoverage);
    vkDestroyPipeline(c.dev, a2c, nullptr);
    vkDestroyPipelineLayout(c.dev, pl, nullptr);
    vkDestroyBuffer(c.dev, vb, nullptr);
    vkFreeMemory(c.dev, bm, nullptr);
    vkDestroyShaderModule(c.dev, vs, nullptr);
    vkDestroyShaderModule(c.dev, fs, nullptr);
  }
  {  // With the Quest identity (conformance run) a 1x A2C pipeline keeps the hardware path
    vpvk_config_set_quest_identity(1);
    setenv("VPVK_EXTRA_EXTENSIONS", "1", 1);
    Ctx c;
    DrawRig rig(c);
    VkShaderModule vs = module(c, "pos.vert.spv"), fs = module(c, "solid.frag.spv");
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout pl;
    vkCreatePipelineLayout(c.dev, &plci, nullptr, &pl);
    PipeOpts ao;
    ao.a2c = true;
    VkPipeline a2c = makePosPipeline(c, rig, pl, ao, vs, fs);
    CHECK(!from<PipelineObj>(a2c)->a2cTest && from<PipelineObj>(a2c)->state.alphaToCoverage);
    vkDestroyPipeline(c.dev, a2c, nullptr);
    vkDestroyPipelineLayout(c.dev, pl, nullptr);
    vkDestroyShaderModule(c.dev, vs, nullptr);
    vkDestroyShaderModule(c.dev, fs, nullptr);
    unsetenv("VPVK_EXTRA_EXTENSIONS");
    vpvk_config_set_quest_identity(0);
  }
  {  // memory budget, host image copy
    Ctx c({VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME});
    CHECK(c.dev != nullptr);
    VkPhysicalDeviceMemoryBudgetPropertiesEXT mb{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 mp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &mb};
    vkGetPhysicalDeviceMemoryProperties2(c.pd, &mp);
    const uint64_t used0 = mb.heapUsage[0];
    VkDeviceMemory big = alloc(c, 64 << 20, 0);
    vkGetPhysicalDeviceMemoryProperties2(c.pd, &mp);
    CHECK(mb.heapUsage[0] == used0 + (64 << 20) && mb.heapBudget[0] >= mb.heapUsage[0] && mb.heapBudget[1] == 0);
    vkFreeMemory(c.dev, big, nullptr);
    vkGetPhysicalDeviceMemoryProperties2(c.pd, &mp);
    CHECK(mb.heapUsage[0] == used0);

    auto& nb = dynamic_cast<vpgpu::NullBackend&>(from<Device>(c.dev)->platform->backend());
    VkDeviceMemory m = alloc(c, 1 << 20, 0);
    VkImage im = image(c, VK_FORMAT_R8G8B8A8_UNORM, 64, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT, m);
    CHECK(im != VK_NULL_HANDLE);
    std::vector<uint8_t> data(64 * 64 * 4, 7);
    VkMemoryToImageCopyEXT r{VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT};
    r.pHostPointer = data.data();
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {64, 64, 1};
    VkCopyMemoryToImageInfoEXT ci{VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO_EXT};
    ci.dstImage = im, ci.dstImageLayout = VK_IMAGE_LAYOUT_GENERAL, ci.regionCount = 1, ci.pRegions = &r;
    const size_t before = nb.calls.size();
    auto copyToImage = reinterpret_cast<PFN_vkCopyMemoryToImageEXT>(vkGetDeviceProcAddr(c.dev, "vkCopyMemoryToImageEXT"));
    CHECK(copyToImage && copyToImage(c.dev, &ci) == VK_SUCCESS);
    bool submitted = false, staged = false;
    for (size_t i = before; i < nb.calls.size(); ++i) {
      submitted |= nb.calls[i].rfind("submit", 0) == 0;
      staged |= nb.calls[i].rfind("createBuffer", 0) == 0 && nb.calls[i].find("size=16384") != std::string::npos;
    }
    CHECK(submitted && staged);
    std::vector<uint8_t> back(data.size(), 0);
    VkImageToMemoryCopyEXT rb{VK_STRUCTURE_TYPE_IMAGE_TO_MEMORY_COPY_EXT};
    rb.pHostPointer = back.data(), rb.imageSubresource = r.imageSubresource, rb.imageExtent = r.imageExtent;
    VkCopyImageToMemoryInfoEXT bi{VK_STRUCTURE_TYPE_COPY_IMAGE_TO_MEMORY_INFO_EXT};
    bi.srcImage = im, bi.regionCount = 1, bi.pRegions = &rb;
    auto copyToMem = reinterpret_cast<PFN_vkCopyImageToMemoryEXT>(vkGetDeviceProcAddr(c.dev, "vkCopyImageToMemoryEXT"));
    CHECK(copyToMem && copyToMem(c.dev, &bi) == VK_SUCCESS);
    VkHostImageLayoutTransitionInfoEXT tr{VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO_EXT};
    tr.image = im;
    auto transition = reinterpret_cast<PFN_vkTransitionImageLayoutEXT>(vkGetDeviceProcAddr(c.dev, "vkTransitionImageLayoutEXT"));
    CHECK(transition && transition(c.dev, 1, &tr) == VK_SUCCESS);
    VkPhysicalDeviceHostImageCopyPropertiesEXT hp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &hp};
    vkGetPhysicalDeviceProperties2(c.pd, &p2);
    CHECK(hp.copySrcLayoutCount > 0 && hp.copyDstLayoutCount > 0);
    VkFormatProperties3 fp3{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
    VkFormatProperties2 fp2{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &fp3};
    vkGetPhysicalDeviceFormatProperties2(c.pd, VK_FORMAT_R8G8B8A8_UNORM, &fp2);
    CHECK(fp3.optimalTilingFeatures & VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT);
    vkDestroyImage(c.dev, im, nullptr);
    vkFreeMemory(c.dev, m, nullptr);
  }
  {  // dynamic rendering local read
    Ctx c({VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME, VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME});
    CHECK(c.dev != nullptr);
    VkDeviceMemory m = alloc(c, 32 << 20, 0);
    VkImage col = image(c, VK_FORMAT_B10G11R11_UFLOAT_PACK32, 64, 2, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, m);
    VkImage dep = image(c, VK_FORMAT_D32_SFLOAT_S8_UINT, 64, 2,
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, m);
    VkImageView cv, dv;
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = col, vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY, vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 2};
    vkCreateImageView(c.dev, &vi, nullptr, &cv);
    vi.image = dep, vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    vkCreateImageView(c.dev, &vi, nullptr, &dv);
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    VkCommandPool pool;
    vkCreateCommandPool(c.dev, &pi, nullptr, &pool);
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool, ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    vkAllocateCommandBuffers(c.dev, &ai, &cb);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}, da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    ca.imageView = cv, ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    da.imageView = dv, da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, da.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {64, 64}}, ri.viewMask = 3, ri.layerCount = 1, ri.colorAttachmentCount = 1, ri.pColorAttachments = &ca;
    ri.pDepthAttachment = &da;
    const vpgpu::TextureId depthId = from<Image>(dep)->id;
    CHECK(depthId);
    Recorder& r = *from<CommandBuffer>(cb)->rec;
    auto draw = [&](bool readsDepth, bool viewed) {
      OpDraw d;
      d.draw.count = 3;
      if (readsDepth) {
        vpgpu::Access a{vpgpu::ResourceRange::wholeTexture(depthId), vpgpu::stage::fragment, false};
        a.kind = vpgpu::BindKind::Texture, a.slot = 2;
        d.draw.bindings.push_back(a);
        if (viewed) d.draw.views.push_back({vpgpu::BindKind::Texture, 2, depthId});  // a non-trivial view of the attachment
        if (r.bound.localRead && std::find(r.bound.passImages.begin(), r.bound.passImages.end(), from<Image>(dep)) !=
                                     r.bound.passImages.end())
          d.images.push_back(from<Image>(dep));
      }
      r.ops.emplace_back(std::move(d));
    };
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    auto record = [&](bool soft, bool viewed = true) {
      vkBeginCommandBuffer(cb, &bi);
      vkCmdBeginRendering(cb, &ri);
      draw(false, false);
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           VK_DEPENDENCY_BY_REGION_BIT, 1, &mb, 0, nullptr, 0, nullptr);
      draw(soft, viewed);
      vkCmdEndRendering(cb);
      vkEndCommandBuffer(cb);
    };
    record(true);
    vpgpu::Frame frame;
    TranslateStats st;
    translate(r, frame, st, nullptr, nullptr, nullptr, nullptr, from<Device>(c.dev));
    const auto& cmds = frame.commands();
    CHECK(st.localReads == 1 && cmds.size() == 3);
    if (cmds.size() == 3) {
      const auto* p0 = std::get_if<vpgpu::RenderPass>(&cmds[0]);
      const auto* cp = std::get_if<vpgpu::Copy>(&cmds[1]);
      const auto* p1 = std::get_if<vpgpu::RenderPass>(&cmds[2]);
      CHECK(p0 && p0->desc.depth.store == vpgpu::StoreAction::Store);  // kept for the second half
      CHECK(cp && cp->srcTexture.texture == depthId && cp->dstTexture.texture == from<Image>(dep)->localReadCopy &&
            cp->srcTexture.layerCount == 2);
      CHECK(p1 && p1->desc.depth.load == vpgpu::LoadAction::Load && p1->draws.size() == 1);
      if (p1 && p1->draws.size() == 1) {
        CHECK(p1->draws[0].bindings[0].range.index == from<Image>(dep)->localReadCopy.index);
        CHECK(p1->draws[0].views.empty());  // a non-trivial view cannot follow: the copy's default view (logged once)
      }
    }
    {  // A trivial view (no ViewBinding) just follows the copy
      record(true, false);
      vpgpu::Frame ft;
      TranslateStats stt;
      translate(r, ft, stt, nullptr, nullptr, nullptr, nullptr, from<Device>(c.dev));
      const auto* pt = ft.commands().size() == 3 ? std::get_if<vpgpu::RenderPass>(&ft.commands()[2]) : nullptr;
      CHECK(pt && pt->draws.size() == 1 && pt->draws[0].views.empty() &&
            pt->draws[0].bindings[0].range.index == from<Image>(dep)->localReadCopy.index);
    }
    // Without a draw reading an attachment the barrier costs nothing: one pass.
    record(false);
    vpgpu::Frame f2;
    TranslateStats s2;
    translate(r, f2, s2, nullptr, nullptr, nullptr, nullptr, from<Device>(c.dev));
    CHECK(f2.commands().size() == 1 && s2.localReads == 0);
    {  // Quest identity (even in the conformance run): the pass is neither split nor copied, as before the extension
      auto& nb = dynamic_cast<vpgpu::NullBackend&>(from<Device>(c.dev)->platform->backend());
      auto encoders = [&](bool quest) {
        vpvk_config_set_quest_identity(quest);
        setenv("VPVK_EXTRA_EXTENSIONS", "1", 1);
        record(true);
        const size_t before = nb.calls.size();
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1, si.pCommandBuffers = &cb;
        CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS);
        vkQueueWaitIdle(c.queue);
        unsetenv("VPVK_EXTRA_EXTENSIONS");
        vpvk_config_set_quest_identity(0);
        int n = -1;
        for (size_t i = before; i < nb.calls.size(); ++i)
          if (nb.calls[i].rfind("submit", 0) == 0) std::sscanf(nb.calls[i].c_str(), "submit %*u encoders=%d", &n);
        return n;
      };
      const int native = encoders(false), quest = encoders(true);
      CHECK(native > quest && quest == 1);
    }
    vkDestroyCommandPool(c.dev, pool, nullptr);
    vkDestroyImageView(c.dev, cv, nullptr);
    vkDestroyImageView(c.dev, dv, nullptr);
    vkDestroyImage(c.dev, col, nullptr);
    vkDestroyImage(c.dev, dep, nullptr);
    vkFreeMemory(c.dev, m, nullptr);
  }
}

// ---- VPCAP trimmed capture (capture.cpp) ----
// Everything goes through the hooked entry points (vkGetInstanceProcAddr), as a game or vpgles reaches them.
#include "vpvk/vpcap_calls.h"

#define CAP_FNS(X)                                                                                                    \
  X(vkCreateInstance) X(vkDestroyInstance) X(vkCreateDevice) X(vkDestroyDevice) X(vkGetDeviceQueue)                   \
  X(vkAllocateMemory) X(vkFreeMemory) X(vkMapMemory) X(vkCreateBuffer) X(vkDestroyBuffer) X(vkBindBufferMemory)       \
  X(vkCreateImage) X(vkBindImageMemory) X(vkCreateImageView) X(vkCreateSampler) X(vkCreateDescriptorSetLayout)        \
  X(vkDestroyDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkCreateDescriptorPool) X(vkAllocateDescriptorSets)     \
  X(vkUpdateDescriptorSets) X(vkCreateRenderPass) X(vkCreateFramebuffer) X(vkCreateShaderModule)                      \
  X(vkDestroyShaderModule) X(vkCreateGraphicsPipelines) X(vkCreateCommandPool) X(vkAllocateCommandBuffers)            \
  X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkFreeCommandBuffers) X(vkCmdBeginRenderPass) X(vkCmdEndRenderPass) X(vkCmdBindPipeline)    \
  X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) X(vkCmdBindVertexBuffers) X(vkCmdBindIndexBuffer)                  \
  X(vkCmdSetViewport) X(vkCmdSetScissor) X(vkCmdDraw) X(vkCmdDrawIndexed) X(vkQueueSubmit) X(vkQueueWaitIdle)                \
  X(vkQueueSubmit2) X(vkCmdBeginRendering) X(vkCmdEndRendering) X(vkCmdClearColorImage) X(vkCmdCopyImage)             \
  X(vkCmdCopyImageToBuffer) X(vkCreateEvent) X(vkCreateQueryPool) X(vkCmdSetEvent) X(vkCmdResetEvent)                \
  X(vkCmdWaitEvents) X(vkCmdSetEvent2) X(vkCmdResetEvent2) X(vkCmdWaitEvents2) X(vkCmdPipelineBarrier2)               \
  X(vkCmdBeginQuery) X(vkCmdEndQuery) X(vkCmdResetQueryPool) X(vkCmdWriteTimestamp) X(vkCmdWriteTimestamp2)           \
  X(vkCmdCopyBuffer2) X(vkCmdCopyImage2) X(vkCmdCopyBufferToImage2) X(vkCmdCopyImageToBuffer2) X(vkCmdBlitImage2)     \
  X(vkCmdBindVertexBuffers2) X(vkCmdDrawIndirectCount) X(vkCmdDrawIndexedIndirectCount)

struct CapApi {
#define X(n) PFN_##n n = nullptr;
  CAP_FNS(X)
#undef X
  void load(VkInstance inst) {
#define X(n) n = reinterpret_cast<PFN_##n>(vkGetInstanceProcAddr(inst, #n)); CHECK(n != nullptr);
    CAP_FNS(X)
#undef X
  }
};

// A device with a vertex+index buffer, a uniform buffer behind a descriptor set, a render target and a pipeline, all made
// through the hooks. Memory type 1 (host visible) holds the buffers.
struct CapRig {
  CapApi v;
  VkInstance inst = nullptr;
  VkDevice dev = nullptr;
  VkQueue queue = nullptr;
  VkDeviceMemory host = nullptr, devMem = nullptr;
  VkImage color = nullptr;  // the 64x64 render target (cleared by the pass of record())
  VkBuffer vbib = nullptr, ub = nullptr;
  VkShaderModule vs = nullptr, fs = nullptr;
  VkDescriptorSetLayout dsl = nullptr;
  VkPipelineLayout pl = nullptr;
  VkDescriptorSet set = nullptr;
  VkRenderPass rp = nullptr;
  VkFramebuffer fb = nullptr;
  VkPipeline pipe = nullptr;
  VkCommandPool cp = nullptr;
  VkCommandBuffer cb[3] = {};
  uint8_t* map = nullptr;
  std::vector<uint32_t> vsWords, fsWords;
  uint64_t deadBytes = 0;

  CapRig() {
    v.vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(vkGetInstanceProcAddr(nullptr, "vkCreateInstance"));
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vpcap-test", app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    CHECK(v.vkCreateInstance(&ici, nullptr, &inst) == VK_SUCCESS);
    v.load(inst);
    VkPhysicalDevice pd = nullptr;
    uint32_t n = 1;
    vkEnumeratePhysicalDevices(inst, &n, &pd);
    float prio = 1;
    VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    q.queueCount = 1, q.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.timelineSemaphore = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
    VkDeviceCreateInfo d{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
    d.queueCreateInfoCount = 1, d.pQueueCreateInfos = &q;
    CHECK(v.vkCreateDevice(pd, &d, nullptr, &dev) == VK_SUCCESS);
    v.vkGetDeviceQueue(dev, 0, 0, &queue);

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = 1 << 20, ai.memoryTypeIndex = 1;
    v.vkAllocateMemory(dev, &ai, nullptr, &host);
    ai.allocationSize = 8 << 20, ai.memoryTypeIndex = 0;
    v.vkAllocateMemory(dev, &ai, nullptr, &devMem);
    void* p = nullptr;
    v.vkMapMemory(dev, host, 0, VK_WHOLE_SIZE, 0, &p);
    map = static_cast<uint8_t*>(p);
    auto buf = [&](VkDeviceSize size, VkBufferUsageFlags usage, VkDeviceSize off) {
      VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      bi.size = size, bi.usage = usage;
      VkBuffer b = nullptr;
      v.vkCreateBuffer(dev, &bi, nullptr, &b);
      v.vkBindBufferMemory(dev, b, host, off);
      return b;
    };
    VkBuffer dead = buf(512, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 65536);  // destroyed before any capture: not in the state
    v.vkDestroyBuffer(dev, dead, nullptr);
    vbib = buf(8192, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, 0);
    ub = buf(256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 16384);

    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D, ii.format = VK_FORMAT_R8G8B8A8_UNORM, ii.extent = {64, 64, 1};
    ii.mipLevels = 1, ii.arrayLayers = 1, ii.samples = VK_SAMPLE_COUNT_1_BIT, ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    v.vkCreateImage(dev, &ii, nullptr, &color);
    v.vkBindImageMemory(dev, color, devMem, 0);
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = color, vi.viewType = VK_IMAGE_VIEW_TYPE_2D, vi.format = VK_FORMAT_R8G8B8A8_UNORM;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView view = nullptr;
    v.vkCreateImageView(dev, &vi, nullptr, &view);
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    VkSampler smp = nullptr;
    v.vkCreateSampler(dev, &si, nullptr, &smp);

    VkAttachmentDescription att{};
    att.format = VK_FORMAT_R8G8B8A8_UNORM, att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkAttachmentReference cref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sp{};
    sp.colorAttachmentCount = 1, sp.pColorAttachments = &cref;
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = 1, rpi.pAttachments = &att, rpi.subpassCount = 1, rpi.pSubpasses = &sp;
    v.vkCreateRenderPass(dev, &rpi, nullptr, &rp);
    VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fi.renderPass = rp, fi.attachmentCount = 1, fi.pAttachments = &view, fi.width = 64, fi.height = 64, fi.layers = 1;
    v.vkCreateFramebuffer(dev, &fi, nullptr, &fb);

    VkDescriptorSetLayoutBinding b0{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = 1, lci.pBindings = &b0;
    v.vkCreateDescriptorSetLayout(dev, &lci, nullptr, &dsl);
    VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, 16};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1, plci.pSetLayouts = &dsl, plci.pushConstantRangeCount = 1, plci.pPushConstantRanges = &pcr;
    v.vkCreatePipelineLayout(dev, &plci, nullptr, &pl);
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1, dpi.poolSizeCount = 1, dpi.pPoolSizes = &ps;
    VkDescriptorPool pool = nullptr;
    v.vkCreateDescriptorPool(dev, &dpi, nullptr, &pool);
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = pool, dai.descriptorSetCount = 1, dai.pSetLayouts = &dsl;
    v.vkAllocateDescriptorSets(dev, &dai, &set);
    VkDescriptorBufferInfo bi{ub, 0, 256};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = set, w.descriptorCount = 1, w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, w.pBufferInfo = &bi;
    v.vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);

    auto module = [&](const char* name, std::vector<uint32_t>& words) {
      words = spirvFile(name);
      VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      ci.codeSize = words.size() * 4, ci.pCode = words.data();
      VkShaderModule m = nullptr;
      v.vkCreateShaderModule(dev, &ci, nullptr, &m);
      return m;
    };
    vs = module("pos.vert.spv", vsWords), fs = module("solid.frag.spv", fsWords);
    VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                             {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT, st[0].module = vs, st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT, st[1].module = fs, st[1].pName = "main";
    VkVertexInputBindingDescription vbd{0, 16, VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription vad{0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0};
    VkPipelineVertexInputStateCreateInfo vis{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vis.vertexBindingDescriptionCount = 1, vis.pVertexBindingDescriptions = &vbd;
    vis.vertexAttributeDescriptionCount = 1, vis.pVertexAttributeDescriptions = &vad;
    VkPipelineInputAssemblyStateCreateInfo ias{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ias.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1, vps.scissorCount = 1;  // dynamic
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.cullMode = VK_CULL_MODE_BACK_BIT, rs.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cbs.attachmentCount = 1, cbs.pAttachments = &cba;
    const VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dys{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dys.dynamicStateCount = 2, dys.pDynamicStates = dyn;
    VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.stageCount = 2, gp.pStages = st, gp.pVertexInputState = &vis, gp.pInputAssemblyState = &ias, gp.pViewportState = &vps;
    gp.pRasterizationState = &rs, gp.pMultisampleState = &ms, gp.pColorBlendState = &cbs, gp.pDynamicState = &dys;
    gp.layout = pl, gp.renderPass = rp;
    CHECK(v.vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &pipe) == VK_SUCCESS);
    // Legal and common: the modules and the set layout go once the pipeline and the layout exist.
    v.vkDestroyShaderModule(dev, vs, nullptr);
    v.vkDestroyShaderModule(dev, fs, nullptr);
    v.vkDestroyDescriptorSetLayout(dev, dsl, nullptr);

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    v.vkCreateCommandPool(dev, &cpi, nullptr, &cp);
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = cp, cai.commandBufferCount = 3;
    v.vkAllocateCommandBuffers(dev, &cai, cb);
  }
  // begin, pass, bind everything, one draw, end.
  void record(VkCommandBuffer c, bool indexed) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    v.vkBeginCommandBuffer(c, &bi);
    VkClearValue clear{};
    clear.color.float32[1] = 0.5f;
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = rp, rbi.framebuffer = fb, rbi.renderArea = {{0, 0}, {64, 64}}, rbi.clearValueCount = 1, rbi.pClearValues = &clear;
    v.vkCmdBeginRenderPass(c, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    v.vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    v.vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &set, 0, nullptr);
    const float push[4] = {1, 2, 3, 4};
    v.vkCmdPushConstants(c, pl, VK_SHADER_STAGE_VERTEX_BIT, 0, 16, push);
    VkDeviceSize off = 32;
    v.vkCmdBindVertexBuffers(c, 0, 1, &vbib, &off);
    v.vkCmdBindIndexBuffer(c, vbib, 4096, VK_INDEX_TYPE_UINT16);
    const VkViewport vp{0, 0, 64, 64, 0, 1};
    const VkRect2D sc{{0, 0}, {64, 64}};
    v.vkCmdSetViewport(c, 0, 1, &vp);
    v.vkCmdSetScissor(c, 0, 1, &sc);
    if (indexed) v.vkCmdDrawIndexed(c, 36, 3, 6, 2, 1);
    else v.vkCmdDraw(c, 12, 2, 3, 4);
    v.vkCmdEndRenderPass(c);
    v.vkEndCommandBuffer(c);
  }
  void submit(std::initializer_list<VkCommandBuffer> cbs) {
    std::vector<VkCommandBuffer> l(cbs);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = uint32_t(l.size()), si.pCommandBuffers = l.data();
    CHECK(v.vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS);
    v.vkQueueWaitIdle(queue);
  }
  void fillBuffers(uint8_t seed) {
    for (int i = 0; i < 8192; ++i) map[i] = uint8_t(i * 7 + seed);
    for (int i = 0; i < 256; ++i) map[16384 + i] = uint8_t(i * 3 + seed);
  }
  ~CapRig() {
    v.vkDestroyDevice(dev, nullptr);
    v.vkDestroyInstance(inst, nullptr);
  }
};

void testCapture() {
  const std::string path = (std::filesystem::temp_directory_path() / ("vpcap_test_" + std::to_string(getpid()) + ".vpcap")).string();
  const std::string path2 = path + "2";
  std::remove(path.c_str());
  vpvk_capture_track(0);
  CHECK(vpvk_capture_request(1, path.c_str()) == -1);  // tracking is off
  CHECK(vpvk_capture_track(1) == 0);
  {
    CapRig r;
    r.fillBuffers(1);
    r.record(r.cb[0], true);
    r.submit({r.cb[0]});  // not captured: only tracked
    CHECK(vpvk_capture_state() == 0 && !std::filesystem::exists(path));
    CHECK(vpvk_capture_request(2, path.c_str()) == 1 && vpvk_capture_state() == 1);
    CHECK(vpvk_capture_request(2, path.c_str()) == 0);  // already pending

    r.fillBuffers(5);
    r.submit({r.cb[0]});  // the frame boundary: the capture starts when it returns
    CHECK(vpvk_capture_state() == 2);
    r.record(r.cb[1], true);
    r.submit({r.cb[0], r.cb[1]});  // frame 0; cb[0] was recorded before the capture: dropped from the recorded submit
    CHECK(vpvk_capture_state() == 2);
    r.map[4096 + 17] ^= 0xFF;      // frame 1: the index page changed, nothing else
    r.record(r.cb[2], false);
    r.submit({r.cb[2]});
    CHECK(vpvk_capture_state() == 0);
    r.submit({r.cb[2]});           // after the capture: nothing more is written

    // A second capture of the same device: a freed command buffer is gone from the state, the file starts over (every
    // page again), and cb[2], recorded during the first capture, is not part of this one.
    r.v.vkFreeCommandBuffers(r.dev, r.cp, 1, &r.cb[0]);
    CHECK(vpvk_capture_request(1, path2.c_str()) == 1);
    r.submit({r.cb[2]});
    r.record(r.cb[1], true);
    r.submit({r.cb[1], r.cb[2]});
    CHECK(vpvk_capture_state() == 0);
  }
  vpvk_capture_track(0);

  vpcap::Reader rd;
  std::string err;
  CHECK(rd.open(path, &err));
  CHECK(rd.version() == vpcap::kVersion);
  struct Rec {
    uint16_t id;
    vpcap::Section section;
    uint32_t frame;
    std::vector<uint8_t> data;
  };
  std::vector<Rec> all;
  vpcap::Call c;
  while (rd.next(c)) all.push_back({c.id, c.section, c.frame, std::vector<uint8_t>(c.data, c.data + c.size)});
  CHECK(rd.error().empty());
  CHECK(rd.meta().find("format=VPCAP") != std::string::npos && rd.meta().find("frames=2\n") != std::string::npos);
  CHECK(rd.meta().find("frame_unit=submit\n") != std::string::npos && rd.meta().find("invalid=0\n") != std::string::npos);  // a frame is a submit
  auto asCall = [&](const Rec& x) {
    vpcap::Call k;
    k.id = x.id, k.data = x.data.data(), k.size = uint32_t(x.data.size());
    return k;
  };
  using vpcap::Section;
  auto count = [&](Section s, uint16_t id) {
    int n = 0;
    for (const Rec& x : all) n += x.section == s && x.id == id;
    return n;
  };
  // State: one object of each kind; the destroyed buffer is gone; the modules and the set layout stay because the
  // pipeline and the pipeline layout still reference them.
  CHECK(count(Section::State, vpcap::id::vkCreateInstance) == 1 && count(Section::State, vpcap::id::vkCreateDevice) == 1);
  CHECK(count(Section::State, vpcap::id::vkGetDeviceQueue) == 1 && count(Section::State, vpcap::id::vkAllocateMemory) == 2);
  CHECK(count(Section::State, vpcap::id::vkCreateBuffer) == 2 && count(Section::State, vpcap::id::vkBindBufferMemory) == 2);
  CHECK(count(Section::State, vpcap::id::vkCreateImage) == 1 && count(Section::State, vpcap::id::vkCreateImageView) == 1);
  CHECK(count(Section::State, vpcap::id::vkCreateSampler) == 1 && count(Section::State, vpcap::id::vkCreateFramebuffer) == 1);
  CHECK(count(Section::State, vpcap::id::vkCreateShaderModule) == 2 && count(Section::State, vpcap::id::vkCreateDescriptorSetLayout) == 1);
  CHECK(count(Section::State, vpcap::id::vkCreateGraphicsPipelines) == 1 && count(Section::State, vpcap::id::vkCreateRenderPass) == 1);
  CHECK(count(Section::State, vpcap::id::vkAllocateCommandBuffers) == 3 && count(Section::State, vpcap::id::vkAllocateDescriptorSets) == 1);
  CHECK(count(Section::State, vpcap::id::vkCreateDescriptorPool) == 1 && count(Section::State, vpcap::id::vkUpdateDescriptorSets) == 1);
  CHECK(count(Section::State, vpcap::id::vkDestroyShaderModule) == 0 && count(Section::State, vpcap::id::vkDestroyBuffer) == 0);

  const std::vector<uint32_t> vsWords = spirvFile("pos.vert.spv"), fsWords = spirvFile("solid.frag.spv");
  uint64_t memHostId = 0, ubId = 0, vbibId = 0, vsId = 0, fsId = 0;
  int mem = 0, modules = 0;
  for (const Rec& x : all) {
    if (x.section != Section::State) continue;
    if (x.id == vpcap::id::vkCreateDevice) {  // queues and the feature chain survive
      auto d = vpcap::decode<vpcap::id::vkCreateDevice>(asCall(x));
      CHECK(d && d->ok && std::get<1>(d->args)->queueCreateInfoCount == 1 && std::get<1>(d->args)->pQueueCreateInfos[0].queueCount == 1);
      const auto* f2 = static_cast<const VkPhysicalDeviceFeatures2*>(std::get<1>(d->args)->pNext);
      CHECK(f2 && f2->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
      const auto* f12 = f2 ? static_cast<const VkPhysicalDeviceVulkan12Features*>(f2->pNext) : nullptr;
      CHECK(f12 && f12->timelineSemaphore == VK_TRUE && !f12->pNext);
    }
    if (x.id == vpcap::id::vkAllocateMemory && mem++ == 0) {
      auto d = vpcap::decode<vpcap::id::vkAllocateMemory>(asCall(x));
      CHECK(d && d->ok && std::get<1>(d->args)->allocationSize == (1u << 20) && std::get<1>(d->args)->memoryTypeIndex == 1);
      memHostId = vpcap::idOfHandle(*std::get<3>(d->args));
    }
    if (x.id == vpcap::id::vkCreateBuffer) {
      auto d = vpcap::decode<vpcap::id::vkCreateBuffer>(asCall(x));
      CHECK(d && d->ok);
      (std::get<1>(d->args)->size == 8192 ? vbibId : ubId) = vpcap::idOfHandle(*std::get<3>(d->args));
    }
    if (x.id == vpcap::id::vkCreateShaderModule) {
      auto d = vpcap::decode<vpcap::id::vkCreateShaderModule>(asCall(x));
      CHECK(d && d->ok);
      const std::vector<uint32_t>& want = modules == 0 ? vsWords : fsWords;
      CHECK(std::get<1>(d->args)->codeSize == want.size() * 4 &&
            std::memcmp(std::get<1>(d->args)->pCode, want.data(), want.size() * 4) == 0);
      (modules++ == 0 ? vsId : fsId) = vpcap::idOfHandle(*std::get<3>(d->args));
    }
  }
  CHECK(memHostId && ubId && vbibId && vsId && fsId && ubId != vbibId && vsId != fsId);
  for (const Rec& x : all) {
    if (x.section != Section::State) continue;
    if (x.id == vpcap::id::vkCreateGraphicsPipelines) {  // the whole create info, with stable ids for the modules
      auto d = vpcap::decode<vpcap::id::vkCreateGraphicsPipelines>(asCall(x));
      CHECK(d && d->ok && std::get<2>(d->args) == 1);
      const VkGraphicsPipelineCreateInfo& gp = std::get<3>(d->args)[0];
      CHECK(gp.stageCount == 2 && gp.pStages[0].stage == VK_SHADER_STAGE_VERTEX_BIT);
      CHECK(vpcap::idOfHandle(gp.pStages[0].module) == vsId && vpcap::idOfHandle(gp.pStages[1].module) == fsId);
      CHECK(std::strcmp(gp.pStages[0].pName, "main") == 0);
      CHECK(gp.pVertexInputState && gp.pVertexInputState->pVertexBindingDescriptions[0].stride == 16);
      CHECK(gp.pRasterizationState && gp.pRasterizationState->cullMode == VK_CULL_MODE_BACK_BIT);
      CHECK(gp.pDynamicState && gp.pDynamicState->dynamicStateCount == 2 && gp.pDynamicState->pDynamicStates[1] == VK_DYNAMIC_STATE_SCISSOR);
      CHECK(gp.pViewportState && gp.pViewportState->pViewports == nullptr && !gp.pNext);
    }
    if (x.id == vpcap::id::vkUpdateDescriptorSets) {  // read from the driver's descriptor state
      auto d = vpcap::decode<vpcap::id::vkUpdateDescriptorSets>(asCall(x));
      CHECK(d && d->ok && std::get<1>(d->args) == 1);
      const VkWriteDescriptorSet& w = std::get<2>(d->args)[0];
      CHECK(w.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER && w.pBufferInfo && w.pBufferInfo[0].range == 256);
      CHECK(vpcap::idOfHandle(w.pBufferInfo[0].buffer) == ubId);
    }
  }

  // Frames: the calls of each, in the order they were made (memory data and the submit last).
  auto frameCalls = [&](uint32_t f) {
    std::vector<uint16_t> v;
    for (const Rec& x : all)
      if (x.section == Section::Frame && x.frame == f && x.id != vpcap::id::vpcapMemoryData) v.push_back(x.id);
    return v;
  };
  namespace I = vpcap::id;
  // (each frame opens with the vkQueueWaitIdle the test made after the previous submit)
  const std::vector<uint16_t> common1 = {I::vkQueueWaitIdle, I::vkBeginCommandBuffer, I::vkCmdBeginRenderPass, I::vkCmdBindPipeline,
                                         I::vkCmdBindDescriptorSets, I::vkCmdPushConstants, I::vkCmdBindVertexBuffers,
                                         I::vkCmdBindIndexBuffer, I::vkCmdSetViewport, I::vkCmdSetScissor};
  auto expect = [&](uint16_t draw) {
    std::vector<uint16_t> v = common1;
    v.insert(v.end(), {draw, I::vkCmdEndRenderPass, I::vkEndCommandBuffer, I::vkQueueSubmit, I::vpcapFrameEnd});
    return v;
  };
  CHECK(frameCalls(0) == expect(I::vkCmdDrawIndexed));
  CHECK(frameCalls(1) == expect(I::vkCmdDraw));
  CHECK(count(Section::Frame, I::vpcapFrameEnd) == 2);
  uint32_t last = 0;
  for (const Rec& x : all) last = x.section == Section::Frame ? x.frame : last;
  CHECK(last == 1);  // the submit after the capture left no third frame

  // Arguments round-trip.
  int submits = 0;
  for (const Rec& x : all) {
    if (x.section != Section::Frame) continue;
    if (x.id == I::vkCmdDrawIndexed) {
      auto d = vpcap::decode<I::vkCmdDrawIndexed>(asCall(x));
      CHECK(d && d->ok && std::get<1>(d->args) == 36 && std::get<2>(d->args) == 3 && std::get<3>(d->args) == 6 &&
            std::get<4>(d->args) == 2 && std::get<5>(d->args) == 1);
    }
    if (x.id == I::vkCmdDraw) {
      auto d = vpcap::decode<I::vkCmdDraw>(asCall(x));
      CHECK(d && d->ok && std::get<1>(d->args) == 12 && std::get<2>(d->args) == 2 && std::get<3>(d->args) == 3 && std::get<4>(d->args) == 4);
    }
    if (x.id == I::vkCmdPushConstants) {
      auto d = vpcap::decode<I::vkCmdPushConstants>(asCall(x));
      const float push[4] = {1, 2, 3, 4};
      CHECK(d && d->ok && std::get<4>(d->args) == 16 && std::memcmp(std::get<5>(d->args), push, 16) == 0);
    }
    if (x.id == I::vkCmdBindVertexBuffers) {
      auto d = vpcap::decode<I::vkCmdBindVertexBuffers>(asCall(x));
      CHECK(d && d->ok && std::get<2>(d->args) == 1 && std::get<4>(d->args)[0] == 32 &&
            vpcap::idOfHandle(std::get<3>(d->args)[0]) == vbibId);
    }
    if (x.id == I::vkCmdBeginRenderPass) {
      auto d = vpcap::decode<I::vkCmdBeginRenderPass>(asCall(x));
      CHECK(d && d->ok && std::get<1>(d->args)->clearValueCount == 1 && std::get<1>(d->args)->pClearValues[0].color.float32[1] == 0.5f);
    }
    if (x.id == I::vkQueueSubmit && submits++ == 0) {  // frame 0 submitted {cb0, cb1}: cb0 is not in the capture
      auto d = vpcap::decode<I::vkQueueSubmit>(asCall(x));
      CHECK(d && d->ok && std::get<1>(d->args) == 1 && std::get<2>(d->args)[0].commandBufferCount == 1);
    }
  }

  // Buffer contents: frame 0 carries the vertex/index pages and the uniform page; frame 1 only the page that changed.
  std::vector<uint8_t> image(1 << 20, 0);
  std::vector<std::pair<uint64_t, uint64_t>> frame1;
  for (const Rec& x : all) {
    vpcap::MemoryData md;
    if (x.section != Section::Frame || x.id != I::vpcapMemoryData) continue;
    CHECK(vpcap::decodeMemoryData(asCall(x), md) && md.memory == memHostId);
    if (x.frame == 1) frame1.push_back({md.offset, md.size});
    std::memcpy(image.data() + md.offset, md.data, md.size);
  }
  CHECK(frame1.size() == 1 && frame1[0].first == 4096 && frame1[0].second == 4096);
  bool same = true;
  for (int i = 0; i < 8192; ++i) {
    uint8_t want = uint8_t(i * 7 + 5);
    if (i == 4096 + 17) want ^= 0xFF;
    same = same && image[i] == want;
  }
  for (int i = 0; i < 256; ++i) same = same && image[16384 + i] == uint8_t(i * 3 + 5);
  CHECK(same);
  {
    vpcap::Reader r2;
    CHECK(r2.open(path2, &err));
    int allocs = 0, frameEnds = 0, submitsSeen = 0;
    uint64_t bytes = 0;
    std::vector<uint8_t> image2(1 << 20, 0);
    while (r2.next(c)) {
      allocs += c.section == Section::State && c.id == I::vkAllocateCommandBuffers;
      frameEnds += c.id == I::vpcapFrameEnd;
      vpcap::MemoryData md;
      if (vpcap::decodeMemoryData(c, md)) bytes += md.size, std::memcpy(image2.data() + md.offset, md.data, md.size);
      if (c.id == I::vkQueueSubmit) {
        auto d = vpcap::decode<I::vkQueueSubmit>(c);
        CHECK(d && d->ok && std::get<2>(d->args)[0].commandBufferCount == 1);
        ++submitsSeen;
      }
    }
    CHECK(r2.error().empty() && allocs == 2 && frameEnds == 1 && submitsSeen == 1 && bytes >= 8192 + 4096);
    CHECK(std::memcmp(image2.data(), image.data(), 8192) == 0 && std::memcmp(image2.data() + 16384, image.data() + 16384, 256) == 0);
    CHECK(r2.meta().find("instance_in_state=1") != std::string::npos);
  }
  if (!std::getenv("VPCAP_TEST_KEEP")) std::remove(path.c_str()), std::remove(path2.c_str());  // kept for a look with vpcap_dump -l
}

// ---- VPCAP image contents ----
std::vector<uint8_t> capPattern(uint8_t seed, size_t n) {
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 31 + seed + (i >> 8));
  return v;
}

// One capture of several frames that read images in every way the capture knows (copy source, LOAD attachment of a render
// pass and of dynamic rendering, a descriptor set), plus the ones it must leave out. The null backend returns what the test
// stored for a texture (NullBackend::textureData), standing for what the GPU wrote before the capture.
void testCaptureImages() {
  namespace I = vpcap::id;
  using vpcap::Section;
  const std::string path = (std::filesystem::temp_directory_path() / ("vpcap_img_" + std::to_string(getpid()) + ".vpcap")).string();
  std::remove(path.c_str());
  vpvk_capture_track(0);
  CHECK(vpvk_capture_track(1) == 0);
  const uint32_t kBigW = 4100, kBigH = 1024;  // 16 MiB + 192 KiB: two records
  {
    CapRig r;
    CapApi& v = r.v;
    auto& nb = static_cast<vpgpu::NullBackend&>(from<Device>(r.dev)->platform->backend());
    VkDeviceMemory bigMem = nullptr;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = 20u << 20, mai.memoryTypeIndex = 0;
    CHECK(v.vkAllocateMemory(r.dev, &mai, nullptr, &bigMem) == VK_SUCCESS);
    struct Img {
      VkImage image = nullptr;
      VkImageView view = nullptr;
    };
    auto make = [&](uint32_t w, uint32_t h, VkFormat fmt, uint32_t mips, uint32_t layers, VkImageUsageFlags usage, VkDeviceMemory mem,
                    VkDeviceSize off) {
      VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      ii.imageType = VK_IMAGE_TYPE_2D, ii.format = fmt, ii.extent = {w, h, 1};
      ii.mipLevels = mips, ii.arrayLayers = layers, ii.samples = VK_SAMPLE_COUNT_1_BIT, ii.tiling = VK_IMAGE_TILING_OPTIMAL;
      ii.usage = usage;
      Img m;
      CHECK(v.vkCreateImage(r.dev, &ii, nullptr, &m.image) == VK_SUCCESS);
      CHECK(v.vkBindImageMemory(r.dev, m.image, mem, off) == VK_SUCCESS);
      VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      vi.image = m.image, vi.viewType = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, vi.format = fmt;
      vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, layers};
      CHECK(v.vkCreateImageView(r.dev, &vi, nullptr, &m.view) == VK_SUCCESS);
      return m;
    };
    const VkImageUsageFlags kXfer = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    const VkFormat rgba = VK_FORMAT_R8G8B8A8_UNORM, r8 = VK_FORMAT_R8_UNORM;
    const Img tex = make(8, 8, rgba, 2, 2, VK_IMAGE_USAGE_SAMPLED_BIT | kXfer, r.devMem, 65536);
    const Img texB = make(12, 12, r8, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | kXfer, r.devMem, 131072);
    const Img loadImg = make(16, 16, rgba, 1, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | kXfer, r.devMem, 196608);
    const Img zeroImg = make(20, 20, r8, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | kXfer, r.devMem, 262144);  // never written
    const Img late = make(24, 24, r8, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | kXfer, r.devMem, 327680);
    const Img loadDyn = make(18, 18, rgba, 1, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | kXfer, r.devMem, 393216);
    const Img trans = make(32, 32, rgba, 1, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT, r.devMem, 458752);
    const Img sink = make(4, 4, rgba, 1, 1, VK_IMAGE_USAGE_TRANSFER_DST_BIT, r.devMem, 524288);
    const Img big = make(kBigW, kBigH, rgba, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | kXfer, bigMem, 0);

    // A render pass that loads its colour attachment, and a set that samples texB.
    VkAttachmentDescription att{};
    att.format = rgba, att.samples = VK_SAMPLE_COUNT_1_BIT, att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = att.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentReference cref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sp{};
    sp.colorAttachmentCount = 1, sp.pColorAttachments = &cref;
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = 1, rpi.pAttachments = &att, rpi.subpassCount = 1, rpi.pSubpasses = &sp;
    VkRenderPass rpLoad = nullptr;
    CHECK(v.vkCreateRenderPass(r.dev, &rpi, nullptr, &rpLoad) == VK_SUCCESS);
    VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fci.renderPass = rpLoad, fci.attachmentCount = 1, fci.pAttachments = &loadImg.view, fci.width = 16, fci.height = 16, fci.layers = 1;
    VkFramebuffer fbLoad = nullptr;
    CHECK(v.vkCreateFramebuffer(r.dev, &fci, nullptr, &fbLoad) == VK_SUCCESS);

    VkSampler smp = nullptr;
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    v.vkCreateSampler(r.dev, &sci, nullptr, &smp);
    VkDescriptorSetLayoutBinding b0{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = 1, lci.pBindings = &b0;
    VkDescriptorSetLayout dsl = nullptr;
    v.vkCreateDescriptorSetLayout(r.dev, &lci, nullptr, &dsl);
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1, plci.pSetLayouts = &dsl;
    VkPipelineLayout pl = nullptr;
    v.vkCreatePipelineLayout(r.dev, &plci, nullptr, &pl);
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1, dpi.poolSizeCount = 1, dpi.pPoolSizes = &ps;
    VkDescriptorPool pool = nullptr;
    v.vkCreateDescriptorPool(r.dev, &dpi, nullptr, &pool);
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = pool, dai.descriptorSetCount = 1, dai.pSetLayouts = &dsl;
    VkDescriptorSet set = nullptr;
    CHECK(v.vkAllocateDescriptorSets(r.dev, &dai, &set) == VK_SUCCESS);
    VkDescriptorImageInfo dii{smp, texB.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = set, w.descriptorCount = 1, w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, w.pImageInfo = &dii;
    v.vkUpdateDescriptorSets(r.dev, 1, &w, 0, nullptr);

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = 4096, bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer readback = nullptr;
    v.vkCreateBuffer(r.dev, &bci, nullptr, &readback);
    v.vkBindBufferMemory(r.dev, readback, r.host, 32768);

    VkCommandBuffer cbs[9] = {};
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = r.cp, cai.commandBufferCount = 9;
    CHECK(v.vkAllocateCommandBuffers(r.dev, &cai, cbs) == VK_SUCCESS);
    auto begin = [&](VkCommandBuffer c) {
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      v.vkBeginCommandBuffer(c, &bi);
    };
    auto copyOne = [&](VkCommandBuffer c, VkImage src) {
      const VkImageCopy reg{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {1, 1, 1}};
      v.vkCmdCopyImage(c, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sink.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
    };

    // Written by the GPU before the capture: full clears settle the zero-init of the images the frames then read.
    VkCommandBuffer up = cbs[0];
    begin(up);
    const VkClearColorValue zero{};
    const struct { const Img* im; uint32_t mips, layers; } written[] = {{&tex, 2, 2},   {&texB, 1, 1},    {&loadImg, 1, 1},
                                                                      {&late, 1, 1},  {&loadDyn, 1, 1},  {&big, 1, 1}};
    for (const auto& x : written) {
      const VkImageSubresourceRange rg{VK_IMAGE_ASPECT_COLOR_BIT, 0, x.mips, 0, x.layers};
      v.vkCmdClearColorImage(up, x.im->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &rg);
    }
    v.vkEndCommandBuffer(up);
    r.submit({up});
    for (const auto& x : written) CHECK(!zeroPendingImage(from<Image>(x.im->image)));
    CHECK(zeroPendingImage(from<Image>(zeroImg.image)));
    auto store = [&](const Img& im, uint32_t mip, uint32_t layer, std::vector<uint8_t> bytes) {
      nb.textureData[vpgpu::NullBackend::dataKey(from<Image>(im.image)->id, uint16_t(mip), uint16_t(layer))] = std::move(bytes);
    };
    for (uint32_t l = 0; l < 2; ++l) {
      store(tex, 0, l, capPattern(uint8_t(10 + l), 8 * 8 * 4));
      store(tex, 1, l, capPattern(uint8_t(20 + l), 4 * 4 * 4));
    }
    store(texB, 0, 0, capPattern(40, 12 * 12));
    store(loadImg, 0, 0, capPattern(50, 16 * 16 * 4));
    store(loadDyn, 0, 0, capPattern(60, 18 * 18 * 4));
    store(late, 0, 0, capPattern(70, 24 * 24));
    store(big, 0, 0, capPattern(80, size_t(kBigW) * kBigH * 4));
    store(zeroImg, 0, 0, capPattern(90, 20 * 20));          // stored, but the image was never written: reads as zero
    store(trans, 0, 0, capPattern(100, 32 * 32 * 4));       // transient: no memory behind it
    nb.textureData[vpgpu::NullBackend::dataKey(from<Image>(r.color)->id, 0, 0)] = capPattern(110, 64 * 64 * 4);  // only cleared

    CHECK(vpvk_capture_request(2, path.c_str()) == 1);
    r.submit({up});  // the boundary
    CHECK(vpvk_capture_state() == 2);
    // frame 0: copy source, LOAD render pass, descriptor set, a never-written copy source, a CLEAR pass, dynamic rendering
    begin(cbs[1]);
    copyOne(cbs[1], tex.image);
    v.vkEndCommandBuffer(cbs[1]);
    begin(cbs[2]);
    {
      VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      rbi.renderPass = rpLoad, rbi.framebuffer = fbLoad, rbi.renderArea = {{0, 0}, {16, 16}};
      v.vkCmdBeginRenderPass(cbs[2], &rbi, VK_SUBPASS_CONTENTS_INLINE);
      v.vkCmdEndRenderPass(cbs[2]);
    }
    v.vkEndCommandBuffer(cbs[2]);
    begin(cbs[3]);
    v.vkCmdBindDescriptorSets(cbs[3], VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &set, 0, nullptr);
    v.vkEndCommandBuffer(cbs[3]);
    begin(cbs[4]);
    copyOne(cbs[4], zeroImg.image);
    v.vkEndCommandBuffer(cbs[4]);
    r.record(cbs[5], true);
    begin(cbs[6]);
    {
      VkRenderingAttachmentInfo ra[2] = {{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}, {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}};
      ra[0].imageView = loadDyn.view, ra[1].imageView = trans.view;
      for (auto& a : ra) a.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, a.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
      ri.renderArea = {{0, 0}, {18, 18}}, ri.layerCount = 1, ri.colorAttachmentCount = 2, ri.pColorAttachments = ra;
      v.vkCmdBeginRendering(cbs[6], &ri);
      v.vkCmdEndRendering(cbs[6]);
    }
    v.vkEndCommandBuffer(cbs[6]);
    r.submit({cbs[1], cbs[2], cbs[3], cbs[4], cbs[5], cbs[6]});
    CHECK(vpvk_capture_state() == 2);
    // frame 1: tex again (already in the file), a first read of late (to a buffer) and of the big image
    begin(cbs[7]);
    copyOne(cbs[7], tex.image);
    {
      const VkBufferImageCopy reg{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {24, 24, 1}};
      v.vkCmdCopyImageToBuffer(cbs[7], late.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, &reg);
    }
    copyOne(cbs[7], big.image);
    v.vkEndCommandBuffer(cbs[7]);
    r.submit({cbs[7]});
    CHECK(vpvk_capture_state() == 0);
  }
  vpvk_capture_track(0);

  vpcap::Reader rd;
  std::string err;
  CHECK(rd.open(path, &err));
  struct Rec {
    uint16_t id;
    Section section;
    uint32_t frame;
    uint64_t index;
    std::vector<uint8_t> data;
  };
  std::vector<Rec> all;
  vpcap::Call c;
  while (rd.next(c)) all.push_back({c.id, c.section, c.frame, c.index, std::vector<uint8_t>(c.data, c.data + c.size)});
  CHECK(rd.error().empty());
  CHECK(rd.meta().find("image_contents=first_read") != std::string::npos);
  auto asCall = [&](const Rec& x) {
    vpcap::Call k;
    k.id = x.id, k.data = x.data.data(), k.size = uint32_t(x.data.size());
    return k;
  };
  std::map<uint32_t, uint64_t> idByWidth;  // image id by the width the state created it with
  for (const Rec& x : all)
    if (x.section == Section::State && x.id == I::vkCreateImage) {
      auto d = vpcap::decode<I::vkCreateImage>(asCall(x));
      CHECK(d && d->ok);
      idByWidth[std::get<1>(d->args)->extent.width] = vpcap::idOfHandle(*std::get<3>(d->args));
    }
  struct Sub {
    std::vector<uint8_t> bytes;
    uint32_t frame = 0, w = 0, h = 0, depth = 0;
    uint64_t firstIndex = 0, total = 0;
    int records = 0;
  };
  std::map<std::tuple<uint64_t, uint32_t, uint32_t, uint32_t>, Sub> subs;  // image, mip, layer, aspect
  std::map<uint32_t, uint64_t> firstSubmit;                              // frame -> index of its vkQueueSubmit
  std::map<uint64_t, int> recordsOf;                                     // image -> vpcapImageData records
  for (const Rec& x : all) {
    if (x.section == Section::Frame && x.id == I::vkQueueSubmit && !firstSubmit.count(x.frame)) firstSubmit[x.frame] = x.index;
    vpcap::ImageData im;
    if (x.id != I::vpcapImageData) continue;
    CHECK(x.section == Section::Frame && vpcap::decodeImageData(asCall(x), im));
    Sub& sb = subs[{im.image, im.mip, im.layer, im.aspect}];
    if (!sb.records++) sb.bytes.assign(im.total, 0), sb.frame = x.frame, sb.firstIndex = x.index, sb.total = im.total;
    sb.w = im.width, sb.h = im.height, sb.depth = im.depth;
    CHECK(im.total == sb.total && im.offset + im.size <= im.total);
    std::memcpy(sb.bytes.data() + im.offset, im.data, im.size);
    ++recordsOf[im.image];
  }
  auto sub = [&](uint32_t width, uint32_t mip, uint32_t layer) -> const Sub* {
    const auto it = subs.find({idByWidth[width], mip, layer, VK_IMAGE_ASPECT_COLOR_BIT});
    return it == subs.end() ? nullptr : &it->second;
  };
  auto has = [&](uint32_t width, uint32_t mip, uint32_t layer, uint32_t frame, const std::vector<uint8_t>& want, uint32_t w, uint32_t h) {
    const Sub* s = sub(width, mip, layer);
    CHECK(s != nullptr);
    if (!s) return;
    CHECK(s->frame == frame && s->bytes == want && s->w == w && s->h == h && s->depth == 1);
    CHECK(s->firstIndex < firstSubmit[frame]);  // written before the submit that first reads it
  };
  for (uint32_t l = 0; l < 2; ++l) {
    has(8, 0, l, 0, capPattern(uint8_t(10 + l), 8 * 8 * 4), 8, 8);
    has(8, 1, l, 0, capPattern(uint8_t(20 + l), 4 * 4 * 4), 4, 4);
  }
  has(12, 0, 0, 0, capPattern(40, 12 * 12), 12, 12);          // through the descriptor set
  has(16, 0, 0, 0, capPattern(50, 16 * 16 * 4), 16, 16);      // LOAD of a render pass
  has(18, 0, 0, 0, capPattern(60, 18 * 18 * 4), 18, 18);      // LOAD of dynamic rendering
  has(24, 0, 0, 1, capPattern(70, 24 * 24), 24, 24);          // first read in the second frame
  has(kBigW, 0, 0, 1, capPattern(80, size_t(kBigW) * kBigH * 4), kBigW, kBigH);
  CHECK(sub(kBigW, 0, 0) && sub(kBigW, 0, 0)->records == 2);  // split in 16 MiB records
  CHECK(recordsOf[idByWidth[8]] == 4);                         // read again in frame 1: not written twice
  CHECK(!sub(64, 0, 0) && !sub(20, 0, 0) && !sub(32, 0, 0) && !sub(4, 0, 0));  // CLEAR only / never written / transient / destination
  const size_t want = 4 + 1 + 1 + 1 + 1 + 1;
  CHECK(subs.size() == want);
  const uint64_t bytes = 2 * (256 + 64) + 144 + 1024 + 1296 + 576 + uint64_t(kBigW) * kBigH * 4;
  const std::string m = rd.meta();
  CHECK(m.find("image_subresources=" + std::to_string(want) + "\n") != std::string::npos);
  CHECK(m.find("image_bytes=" + std::to_string(bytes) + "\n") != std::string::npos);
  CHECK(m.find("image_zero=1\n") != std::string::npos && m.find("image_memoryless=1\n") != std::string::npos);
  CHECK(m.find("image_unsupported=0\n") != std::string::npos && m.find("image_failed=0\n") != std::string::npos);
  CHECK(m.find("img_deferred=0\n") != std::string::npos && m.find("invalid=0\n") != std::string::npos);  // the default budget holds it all
  if (!gKeepImages.empty()) std::filesystem::copy_file(path, gKeepImages, std::filesystem::copy_options::overwrite_existing);  // testReplay replays it
  std::remove(path.c_str());
}

// vkQueueSubmit2 as the frame boundary and delimiter, and the Vulkan 1.3 / synchronisation / query commands the first phase
// left unhooked, all through vkGetInstanceProcAddr like a game.
void testCaptureHooks() {
  namespace I = vpcap::id;
  using vpcap::Section;
  const std::string path = (std::filesystem::temp_directory_path() / ("vpcap_hooks_" + std::to_string(getpid()) + ".vpcap")).string();
  std::remove(path.c_str());
  vpvk_capture_track(0);
  CHECK(vpvk_capture_track(1) == 0);
  {
    CapRig r;
    CapApi& v = r.v;
    auto submit2 = [&](std::initializer_list<std::vector<VkCommandBuffer>> batches) {
      std::vector<std::vector<VkCommandBufferSubmitInfo>> cbi;
      for (const auto& b : batches) {
        cbi.emplace_back();
        for (VkCommandBuffer c : b) cbi.back().push_back({VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, nullptr, c, 0});
      }
      std::vector<VkSubmitInfo2> si;
      for (const auto& b : cbi) {
        VkSubmitInfo2 s2{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        s2.commandBufferInfoCount = uint32_t(b.size()), s2.pCommandBufferInfos = b.data();
        si.push_back(s2);
      }
      CHECK(v.vkQueueSubmit2(r.queue, uint32_t(si.size()), si.data(), VK_NULL_HANDLE) == VK_SUCCESS);
      v.vkQueueWaitIdle(r.queue);
    };
    auto image = [&](VkDeviceSize off) {
      VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      ii.imageType = VK_IMAGE_TYPE_2D, ii.format = VK_FORMAT_R8G8B8A8_UNORM, ii.extent = {8, 8, 1}, ii.mipLevels = 1, ii.arrayLayers = 1;
      ii.samples = VK_SAMPLE_COUNT_1_BIT, ii.tiling = VK_IMAGE_TILING_OPTIMAL;
      ii.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      VkImage im = nullptr;
      CHECK(v.vkCreateImage(r.dev, &ii, nullptr, &im) == VK_SUCCESS);
      if (off != ~VkDeviceSize(0)) CHECK(v.vkBindImageMemory(r.dev, im, r.devMem, off) == VK_SUCCESS);
      return im;
    };
    const VkImage imgA = image(65536), imgB = image(131072), ded = image(~VkDeviceSize(0));
    // A dedicated allocation: the image comes through the memory's pNext.
    VkMemoryDedicatedAllocateInfo dedi{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedi.image = ded;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &dedi};
    mai.allocationSize = 1 << 16, mai.memoryTypeIndex = 0;
    VkDeviceMemory dedMem = nullptr;
    CHECK(v.vkAllocateMemory(r.dev, &mai, nullptr, &dedMem) == VK_SUCCESS);
    CHECK(v.vkBindImageMemory(r.dev, ded, dedMem, 0) == VK_SUCCESS);
    VkEvent ev = nullptr;
    VkEventCreateInfo eci{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
    CHECK(v.vkCreateEvent(r.dev, &eci, nullptr, &ev) == VK_SUCCESS);
    VkQueryPool occ = nullptr, ts = nullptr;
    VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qci.queryType = VK_QUERY_TYPE_OCCLUSION, qci.queryCount = 4;
    CHECK(v.vkCreateQueryPool(r.dev, &qci, nullptr, &occ) == VK_SUCCESS);
    qci.queryType = VK_QUERY_TYPE_TIMESTAMP, qci.queryCount = 2;
    CHECK(v.vkCreateQueryPool(r.dev, &qci, nullptr, &ts) == VK_SUCCESS);
    VkCommandBuffer cbs[3] = {};
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = r.cp, cai.commandBufferCount = 3;
    CHECK(v.vkAllocateCommandBuffers(r.dev, &cai, cbs) == VK_SUCCESS);
    auto begin = [&](VkCommandBuffer c) {
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      v.vkBeginCommandBuffer(c, &bi);
    };
    // The KHR / EXT names of the same entry points resolve to the very hook of the core name.
    const std::pair<const char*, const char*> aliases[] = {
        {"vkQueueSubmit2KHR", "vkQueueSubmit2"},       {"vkCmdPipelineBarrier2KHR", "vkCmdPipelineBarrier2"},
        {"vkCmdSetEvent2KHR", "vkCmdSetEvent2"},       {"vkCmdWaitEvents2KHR", "vkCmdWaitEvents2"},
        {"vkCmdCopyBuffer2KHR", "vkCmdCopyBuffer2"},   {"vkCmdWriteTimestamp2KHR", "vkCmdWriteTimestamp2"},
        {"vkCmdBindVertexBuffers2EXT", "vkCmdBindVertexBuffers2"}};
    for (const auto& [alias, core] : aliases) {
      CHECK(vkGetInstanceProcAddr(r.inst, core) != nullptr && vkGetInstanceProcAddr(r.inst, alias) == vkGetInstanceProcAddr(r.inst, core));
      CHECK(vkGetDeviceProcAddr(r.dev, alias) == vkGetInstanceProcAddr(r.inst, core));
    }
    r.fillBuffers(3);
    r.record(r.cb[0], true);  // recorded before the capture: dropped from the recorded vkQueueSubmit2

    CHECK(vpvk_capture_request(2, path.c_str()) == 1);
    submit2({{r.cb[0]}});  // the boundary is a vkQueueSubmit2
    CHECK(vpvk_capture_state() == 2);

    // frame 0: queries, events (v1 and sync2), barriers, the copy2 family
    VkCommandBuffer c = cbs[0];
    begin(c);
    v.vkCmdResetQueryPool(c, occ, 0, 4);
    v.vkCmdBeginQuery(c, occ, 1, 0);
    v.vkCmdEndQuery(c, occ, 1);
    v.vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ts, 0);
    v.vkCmdWriteTimestamp2(c, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, ts, 1);
    v.vkCmdSetEvent(c, ev, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    v.vkCmdResetEvent(c, ev, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    v.vkCmdWaitEvents(c, 1, &ev, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, nullptr, 0, nullptr, 0, nullptr);
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT, mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    VkBufferMemoryBarrier2 bb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
    bb.buffer = r.vbib, bb.offset = 16, bb.size = 64;
    VkImageMemoryBarrier2 ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    ib.image = imgA, ib.newLayout = VK_IMAGE_LAYOUT_GENERAL, ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1, dep.pMemoryBarriers = &mb, dep.bufferMemoryBarrierCount = 1, dep.pBufferMemoryBarriers = &bb;
    dep.imageMemoryBarrierCount = 1, dep.pImageMemoryBarriers = &ib;
    v.vkCmdSetEvent2(c, ev, &dep);
    v.vkCmdResetEvent2(c, ev, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    v.vkCmdWaitEvents2(c, 1, &ev, &dep);
    v.vkCmdPipelineBarrier2(c, &dep);
    VkBufferCopy2 bc{VK_STRUCTURE_TYPE_BUFFER_COPY_2};
    bc.srcOffset = 64, bc.dstOffset = 128, bc.size = 32;
    VkCopyBufferInfo2 cbi{VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2};
    cbi.srcBuffer = r.vbib, cbi.dstBuffer = r.ub, cbi.regionCount = 1, cbi.pRegions = &bc;
    v.vkCmdCopyBuffer2(c, &cbi);
    VkImageCopy2 ic{VK_STRUCTURE_TYPE_IMAGE_COPY_2};
    ic.srcSubresource = ic.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, ic.extent = {8, 8, 1};
    VkCopyImageInfo2 cii{VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2};
    cii.srcImage = imgA, cii.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, cii.dstImage = imgB;
    cii.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, cii.regionCount = 1, cii.pRegions = &ic;
    v.vkCmdCopyImage2(c, &cii);
    VkBufferImageCopy2 bic{VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2};
    bic.bufferOffset = 256, bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, bic.imageExtent = {8, 8, 1};
    VkCopyBufferToImageInfo2 b2i{VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2};
    b2i.srcBuffer = r.vbib, b2i.dstImage = imgB, b2i.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b2i.regionCount = 1, b2i.pRegions = &bic;
    v.vkCmdCopyBufferToImage2(c, &b2i);
    VkCopyImageToBufferInfo2 i2b{VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2};
    i2b.srcImage = imgA, i2b.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, i2b.dstBuffer = r.ub;
    i2b.regionCount = 1, i2b.pRegions = &bic;
    v.vkCmdCopyImageToBuffer2(c, &i2b);
    VkImageBlit2 bl{VK_STRUCTURE_TYPE_IMAGE_BLIT_2};
    bl.srcSubresource = bl.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    bl.srcOffsets[1] = bl.dstOffsets[1] = {8, 8, 1};
    VkBlitImageInfo2 bli{VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2};
    bli.srcImage = imgA, bli.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, bli.dstImage = imgB;
    bli.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, bli.regionCount = 1, bli.pRegions = &bl, bli.filter = VK_FILTER_LINEAR;
    v.vkCmdBlitImage2(c, &bli);
    v.vkEndCommandBuffer(c);
    submit2({{r.cb[0], c}});  // r.cb[0] is not in the capture

    // frame 1: a pass with vkCmdBindVertexBuffers2 and the indirect-count draws, in two batches (one empty)
    c = cbs[1];
    begin(c);
    VkClearValue clear{};
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = r.rp, rbi.framebuffer = r.fb, rbi.renderArea = {{0, 0}, {64, 64}}, rbi.clearValueCount = 1, rbi.pClearValues = &clear;
    v.vkCmdBeginRenderPass(c, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    v.vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, r.pipe);
    const VkDeviceSize off = 32, size = 1024, stride = 16;
    v.vkCmdBindVertexBuffers2(c, 0, 1, &r.vbib, &off, &size, &stride);
    v.vkCmdBindIndexBuffer(c, r.vbib, 4096, VK_INDEX_TYPE_UINT16);
    v.vkCmdDrawIndirectCount(c, r.vbib, 0, r.ub, 0, 4, 16);
    v.vkCmdDrawIndexedIndirectCount(c, r.vbib, 64, r.ub, 4, 2, 20);
    v.vkCmdEndRenderPass(c);
    v.vkEndCommandBuffer(c);
    submit2({{c}, {}});
    CHECK(vpvk_capture_state() == 0);
  }
  vpvk_capture_track(0);

  vpcap::Reader rd;
  std::string err;
  CHECK(rd.open(path, &err));
  struct Rec {
    uint16_t id;
    Section section;
    uint32_t frame;
    std::vector<uint8_t> data;
  };
  std::vector<Rec> all;
  vpcap::Call k;
  while (rd.next(k)) all.push_back({k.id, k.section, k.frame, std::vector<uint8_t>(k.data, k.data + k.size)});
  CHECK(rd.error().empty());
  auto asCall = [&](const Rec& x) {
    vpcap::Call q;
    q.id = x.id, q.data = x.data.data(), q.size = uint32_t(x.data.size());
    return q;
  };
  auto count = [&](Section s, uint16_t id) {
    int n = 0;
    for (const Rec& x : all) n += x.section == s && x.id == id;
    return n;
  };
  CHECK(count(Section::State, I::vkCreateEvent) == 1 && count(Section::State, I::vkCreateQueryPool) == 2);
  CHECK(rd.meta().find("unhooked_count=0\n") != std::string::npos && rd.meta().find("frames=2\n") != std::string::npos);
  uint64_t dedId = 0;
  for (const Rec& x : all) {  // the dedicated image comes through the pNext of the memory
    if (x.section != Section::State) continue;
    if (x.id == I::vkAllocateMemory) {
      auto d = vpcap::decode<I::vkAllocateMemory>(asCall(x));
      CHECK(d && d->ok);
      const auto* dd = std::get<1>(d->args)->pNext ? static_cast<const VkMemoryDedicatedAllocateInfo*>(std::get<1>(d->args)->pNext) : nullptr;
      if (dd) {
        CHECK(dd->sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO && dd->buffer == nullptr && dd->image != nullptr);
        dedId = vpcap::idOfHandle(dd->image);
      }
    }
  }
  int dedImages = 0;
  for (const Rec& x : all)
    if (x.section == Section::State && x.id == I::vkCreateImage) dedImages += vpcap::idOfHandle(*std::get<3>(vpcap::decode<I::vkCreateImage>(asCall(x))->args)) == dedId;
  CHECK(dedId != 0 && dedImages == 1);

  auto frameCalls = [&](uint32_t f) {
    std::vector<uint16_t> v;
    for (const Rec& x : all)
      if (x.section == Section::Frame && x.frame == f && x.id != I::vpcapMemoryData && x.id != I::vpcapImageData) v.push_back(x.id);
    return v;
  };
  const std::vector<uint16_t> f0 = {I::vkQueueWaitIdle, I::vkBeginCommandBuffer, I::vkCmdResetQueryPool, I::vkCmdBeginQuery, I::vkCmdEndQuery,
                                    I::vkCmdWriteTimestamp, I::vkCmdWriteTimestamp2, I::vkCmdSetEvent, I::vkCmdResetEvent, I::vkCmdWaitEvents,
                                    I::vkCmdSetEvent2, I::vkCmdResetEvent2, I::vkCmdWaitEvents2, I::vkCmdPipelineBarrier2,
                                    I::vkCmdCopyBuffer2, I::vkCmdCopyImage2, I::vkCmdCopyBufferToImage2, I::vkCmdCopyImageToBuffer2,
                                    I::vkCmdBlitImage2, I::vkEndCommandBuffer, I::vkQueueSubmit2, I::vpcapFrameEnd};
  const std::vector<uint16_t> f1 = {I::vkQueueWaitIdle, I::vkBeginCommandBuffer, I::vkCmdBeginRenderPass, I::vkCmdBindPipeline,
                                    I::vkCmdBindVertexBuffers2, I::vkCmdBindIndexBuffer, I::vkCmdDrawIndirectCount,
                                    I::vkCmdDrawIndexedIndirectCount, I::vkCmdEndRenderPass, I::vkEndCommandBuffer, I::vkQueueSubmit2,
                                    I::vpcapFrameEnd};
  CHECK(frameCalls(0) == f0);
  CHECK(frameCalls(1) == f1);
  CHECK(count(Section::Frame, I::vkQueueSubmit) == 0 && count(Section::Frame, I::vpcapFrameEnd) == 2);

  int submits = 0;
  for (const Rec& x : all) {
    if (x.section != Section::Frame) continue;
    if (x.id == I::vkQueueSubmit2) {
      auto d = vpcap::decode<I::vkQueueSubmit2>(asCall(x));
      CHECK(d && d->ok);
      if (submits++ == 0) {  // {cb[0], c}: cb[0] was recorded before the capture
        CHECK(std::get<1>(d->args) == 1 && std::get<2>(d->args)[0].commandBufferInfoCount == 1);
        CHECK(std::get<2>(d->args)[0].sType == VK_STRUCTURE_TYPE_SUBMIT_INFO_2 && std::get<2>(d->args)[0].pCommandBufferInfos[0].commandBuffer != nullptr);
      } else {  // {{c}, {}}
        CHECK(std::get<1>(d->args) == 2 && std::get<2>(d->args)[0].commandBufferInfoCount == 1 && std::get<2>(d->args)[1].commandBufferInfoCount == 0);
      }
    }
    if (x.id == I::vkCmdWaitEvents2) {
      auto d = vpcap::decode<I::vkCmdWaitEvents2>(asCall(x));
      CHECK(d && d->ok && std::get<1>(d->args) == 1 && std::get<2>(d->args)[0] != nullptr);
      const VkDependencyInfo& di = std::get<3>(d->args)[0];
      CHECK(di.memoryBarrierCount == 1 && di.pMemoryBarriers[0].srcStageMask == VK_PIPELINE_STAGE_2_TRANSFER_BIT);
      CHECK(di.bufferMemoryBarrierCount == 1 && di.pBufferMemoryBarriers[0].size == 64 && di.pBufferMemoryBarriers[0].buffer != nullptr);
      CHECK(di.imageMemoryBarrierCount == 1 && di.pImageMemoryBarriers[0].newLayout == VK_IMAGE_LAYOUT_GENERAL && di.pImageMemoryBarriers[0].image != nullptr);
    }
    if (x.id == I::vkCmdBlitImage2) {
      auto d = vpcap::decode<I::vkCmdBlitImage2>(asCall(x));
      CHECK(d && d->ok && std::get<1>(d->args)->filter == VK_FILTER_LINEAR && std::get<1>(d->args)->regionCount == 1);
      CHECK(std::get<1>(d->args)->pRegions[0].dstOffsets[1].x == 8);
    }
    if (x.id == I::vkCmdWriteTimestamp2) {
      auto d = vpcap::decode<I::vkCmdWriteTimestamp2>(asCall(x));
      CHECK(d && d->ok && std::get<1>(d->args) == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT && std::get<3>(d->args) == 1);
    }
    if (x.id == I::vkCmdBindVertexBuffers2) {
      auto d = vpcap::decode<I::vkCmdBindVertexBuffers2>(asCall(x));
      CHECK(d && d->ok && std::get<4>(d->args)[0] == 32 && std::get<5>(d->args)[0] == 1024 && std::get<6>(d->args)[0] == 16);
    }
    if (x.id == I::vkCmdDrawIndexedIndirectCount) {
      auto d = vpcap::decode<I::vkCmdDrawIndexedIndirectCount>(asCall(x));
      CHECK(d && d->ok && std::get<2>(d->args) == 64 && std::get<4>(d->args) == 4 && std::get<5>(d->args) == 2 && std::get<6>(d->args) == 20);
    }
  }
  std::remove(path.c_str());
}

// VPVK_CAPTURE_FRAME=2 VPVK_CAPTURE_DIR=<dir> (set by the test registration, before the first entry point is looked up).
void testCaptureEnv() {
  const char* dir = std::getenv("VPVK_CAPTURE_DIR");
  if (!dir || !std::getenv("VPVK_CAPTURE_FRAME")) {
    std::puts("  skipped: run by ctest gpu_vulkan.capture_env");
    return;
  }
  for (auto& e : std::filesystem::directory_iterator(dir)) std::filesystem::remove(e.path());
  {
    CapRig r;  // tracking came from the environment: no vpvk_capture_track call
    r.fillBuffers(2);
    for (int i = 0; i < 4; ++i) {
      r.record(r.cb[i % 3], true);
      r.submit({r.cb[i % 3]});
      CHECK(vpvk_capture_state() == (i == 1 ? 2 : 0));  // started when frame 2 returned, one frame, done
    }
  }
  int files = 0;
  for (auto& e : std::filesystem::directory_iterator(dir)) {
    ++files;
    vpcap::Reader rd;
    CHECK(rd.open(e.path().string()));
    vpcap::Call c;
    int frameEnds = 0;
    while (rd.next(c)) frameEnds += c.id == vpcap::id::vpcapFrameEnd;
    CHECK(rd.error().empty() && frameEnds == 1);
  }
  CHECK(files == 1);
}

// A damaged or cut .vpcap is refused cleanly (nothing is allocated from a header that lies), and complete() tells a cut file.
void testCaptureCorrupt() {
  const std::string path = (std::filesystem::temp_directory_path() / ("vpcap_corrupt_" + std::to_string(getpid()) + ".vpcap")).string();
  const std::string end = "end=test\n";
  // 3 FrameEnd records in one FRAME block (84 bytes: stored raw) + the END block.
  auto write = [&](size_t junk) {
    vpcap::Writer w;
    CHECK(w.open(path));
    w.begin(vpcap::Section::Frame);
    for (uint64_t i = 0; i < 3; ++i) {
      const uint64_t payload[2] = {i, i};
      vpcap::CallHeader h{vpcap::id::vpcapFrameEnd, 0, 1, 16};
      uint8_t rec[28];
      std::memcpy(rec, &h, 12), std::memcpy(rec + 12, payload, 16);
      w.append(rec, sizeof rec);
    }
    if (junk) w.append("\1\2\3\4\5\6\7\x08", junk);  // bytes that are not a call header
    CHECK(w.close(end));
  };
  struct Out {
    int calls = 0;
    bool opened = false, complete = false;
    std::string error;
  };
  auto read = [&]() {
    Out o;
    vpcap::Reader rd;
    o.opened = rd.open(path);
    vpcap::Call c;
    while (o.opened && rd.next(c)) ++o.calls;
    o.complete = rd.complete(), o.error = rd.error();
    return o;
  };
  auto patch = [&](long at, uint32_t v) {
    std::FILE* f = std::fopen(path.c_str(), "r+b");
    CHECK(f != nullptr);
    std::fseek(f, at, SEEK_SET);
    std::fwrite(&v, 4, 1, f);
    std::fclose(f);
  };
  const uintmax_t endBlock = 16 + end.size();
  write(0);
  const uintmax_t full = std::filesystem::file_size(path);
  {
    Out o = read();  // intact
    CHECK(o.opened && o.calls == 3 && o.complete && o.error.empty());
    vpcap::Reader rd;  // opening again resets the state (and closes the first file)
    std::string err;
    CHECK(rd.open(path, &err));
    vpcap::Call c;
    int n = 0;
    while (rd.next(c)) ++n;
    CHECK(rd.open(path, &err));
    while (rd.next(c)) ++n;
    CHECK(n == 6 && rd.complete());
  }
  std::filesystem::resize_file(path, full - endBlock);  // cut at a block boundary: readable, not complete
  {
    Out o = read();
    CHECK(o.calls == 3 && !o.complete && o.error.empty());
  }
  write(0);
  std::filesystem::resize_file(path, full - 3);  // cut inside the END block's payload
  {
    Out o = read();
    CHECK(o.calls == 3 && !o.complete && o.error == "truncated block");
  }
  write(0);
  std::filesystem::resize_file(path, full - endBlock + 5);  // 5 bytes of a block header left over
  {
    Out o = read();
    CHECK(o.calls == 3 && !o.complete && o.error == "truncated block header");
  }
  write(0);
  patch(32 + 8, 0xFFFFFFF0u);  // the FRAME block claims 4 GiB stored: more than the file holds
  {
    Out o = read();
    CHECK(o.calls == 0 && !o.complete && o.error == "block size out of range");  // raw codec: rawBytes (84) < storedBytes
  }
  write(0);
  patch(32 + 4, vpcap::kCodecZlib), patch(32 + 8, 0xFFFFFFF0u), patch(32 + 12, 84);
  {
    Out o = read();  // zlib: stored larger than the rest of the file
    CHECK(o.calls == 0 && !o.complete && o.error == "truncated block");
  }
  write(0);
  patch(32 + 4, vpcap::kCodecZlib), patch(32 + 12, 0xFFFFFFFFu);  // claims 4 GiB once inflated
  {
    Out o = read();
    CHECK(o.calls == 0 && !o.complete && o.error == "block size out of range");
  }
  write(5);  // a block that ends with 5 bytes that are not a call header
  {
    Out o = read();
    CHECK(o.calls == 3 && !o.complete && o.error == "truncated call record");
  }
  // The codec reads a count that its payload cannot hold as an error, without making room for it first.
  {
    std::vector<uint8_t> p(8, 0);
    const uint32_t n = 50;  // vkQueueSubmit(queue, n, infos, fence): 50 VkSubmitInfo announced, ~100 bytes follow
    p.insert(p.end(), reinterpret_cast<const uint8_t*>(&n), reinterpret_cast<const uint8_t*>(&n) + 4);
    p.push_back(1);
    p.insert(p.end(), 100, 0);
    vpcap::Call call;
    call.id = vpcap::id::vkQueueSubmit, call.data = p.data(), call.size = uint32_t(p.size());
    auto d = vpcap::decode<vpcap::id::vkQueueSubmit>(call);
    CHECK(d && !d->ok);
  }
  std::remove(path.c_str());
}

// Pools keep the list of their sets / command buffers: a reset or a free takes exactly those out of the capture's state.
void testCapturePools() {
  const std::string path = (std::filesystem::temp_directory_path() / ("vpcap_pools_" + std::to_string(getpid()) + ".vpcap")).string();
  std::remove(path.c_str());
  vpvk_capture_track(0);
  CHECK(vpvk_capture_track(1) == 0);
  {
    CapRig r;
    auto fn = [&](const char* name) { return vkGetInstanceProcAddr(r.inst, name); };
    auto resetDescPool = reinterpret_cast<PFN_vkResetDescriptorPool>(fn("vkResetDescriptorPool"));
    auto resetCmdPool = reinterpret_cast<PFN_vkResetCommandPool>(fn("vkResetCommandPool"));
    auto freeSets = reinterpret_cast<PFN_vkFreeDescriptorSets>(fn("vkFreeDescriptorSets"));
    CHECK(resetDescPool && resetCmdPool && freeSets);
    VkDescriptorSetLayoutBinding b0{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = 1, lci.pBindings = &b0;
    VkDescriptorSetLayout dsl2 = nullptr;
    CHECK(r.v.vkCreateDescriptorSetLayout(r.dev, &lci, nullptr, &dsl2) == VK_SUCCESS);
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, dpi.maxSets = 4, dpi.poolSizeCount = 1, dpi.pPoolSizes = &ps;
    VkDescriptorPool pool = nullptr;
    CHECK(r.v.vkCreateDescriptorPool(r.dev, &dpi, nullptr, &pool) == VK_SUCCESS);
    const VkDescriptorSetLayout layouts[3] = {dsl2, dsl2, dsl2};
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = pool, dai.descriptorSetCount = 3, dai.pSetLayouts = layouts;
    VkDescriptorSet sets[3] = {};
    CHECK(r.v.vkAllocateDescriptorSets(r.dev, &dai, sets) == VK_SUCCESS);
    CHECK(freeSets(r.dev, pool, 1, &sets[2]) == VK_SUCCESS);  // one freed on its own, the two others go with the reset
    CHECK(resetDescPool(r.dev, pool, 0) == VK_SUCCESS);
    dai.descriptorSetCount = 1;
    VkDescriptorSet again = nullptr;
    CHECK(r.v.vkAllocateDescriptorSets(r.dev, &dai, &again) == VK_SUCCESS);

    r.fillBuffers(1);
    r.record(r.cb[0], true);
    r.submit({r.cb[0]});
    CHECK(vpvk_capture_request(1, path.c_str()) == 1);
    r.submit({r.cb[0]});  // the boundary
    r.record(r.cb[1], true);
    r.record(r.cb[2], false);
    CHECK(resetCmdPool(r.dev, r.cp, 0) == VK_SUCCESS);  // cb[1] and cb[2] are no longer recorded
    r.record(r.cb[2], false);
    r.submit({r.cb[1], r.cb[2]});  // only cb[2], recorded again, is part of the file's submit
    CHECK(vpvk_capture_state() == 0);
  }
  vpvk_capture_track(0);
  vpcap::Reader rd;
  CHECK(rd.open(path));
  vpcap::Call c;
  int stateSets = 0, stateCbs = 0, resets = 0, submitCbs = -1;
  while (rd.next(c)) {
    stateSets += c.section == vpcap::Section::State && c.id == vpcap::id::vkAllocateDescriptorSets;
    stateCbs += c.section == vpcap::Section::State && c.id == vpcap::id::vkAllocateCommandBuffers;
    resets += c.section == vpcap::Section::Frame && c.id == vpcap::id::vkResetCommandPool;
    if (c.section == vpcap::Section::Frame && c.id == vpcap::id::vkQueueSubmit) {
      auto d = vpcap::decode<vpcap::id::vkQueueSubmit>(c);
      CHECK(d && d->ok);
      if (d && d->ok) submitCbs = int(std::get<2>(d->args)[0].commandBufferCount);
    }
  }
  CHECK(rd.error().empty() && rd.complete());
  CHECK(stateSets == 2);  // the rig's own set and the one allocated after the reset
  CHECK(stateCbs == 3 && resets == 1 && submitCbs == 1);
  std::remove(path.c_str());
}

// The host calls vpvk_frame_boundary: its calls are the frames (frame_unit=host), the submits are not.
void testCaptureHostFrames() {
  const std::string path = (std::filesystem::temp_directory_path() / ("vpcap_host_" + std::to_string(getpid()) + ".vpcap")).string();
  std::remove(path.c_str());
  vpvk_capture_track(0);
  CHECK(vpvk_capture_track(1) == 0);
  {
    CapRig r;
    r.fillBuffers(1);
    r.record(r.cb[0], true);
    r.submit({r.cb[0]});
    vpvk_frame_boundary();  // the first call: from here on the host's calls are the boundary
    CHECK(vpvk_capture_request(2, path.c_str()) == 1);
    r.submit({r.cb[0]});  // a submit no longer starts the capture
    CHECK(vpvk_capture_state() == 1);
    vpvk_frame_boundary();  // starts here
    CHECK(vpvk_capture_state() == 2);
    r.record(r.cb[1], true);
    r.submit({r.cb[1]});  // two submits, one frame
    r.submit({r.cb[1]});
    CHECK(vpvk_capture_state() == 2);
    vpvk_frame_boundary();  // frame 0 ends
    CHECK(vpvk_capture_state() == 2);
    r.submit({r.cb[1]});
    CHECK(vpvk_capture_state() == 2);
    vpvk_frame_boundary();  // frame 1 ends: done
    CHECK(vpvk_capture_state() == 0);
  }
  vpvk_capture_track(0);
  vpcap::Reader rd;
  CHECK(rd.open(path));
  vpcap::Call c;
  int frameEnds = 0, submits = 0;
  uint64_t frame1Submits = 0;
  while (rd.next(c)) {
    if (c.id == vpcap::id::vpcapFrameEnd) {
      uint64_t f = 0, n = 0;
      CHECK(vpcap::decodeFrameEnd(c, f, n) && f == uint64_t(frameEnds));
      if (frameEnds++ == 1) frame1Submits = n;
    }
    submits += c.id == vpcap::id::vkQueueSubmit;
  }
  CHECK(rd.error().empty() && rd.meta().find("frame_unit=host\n") != std::string::npos && rd.meta().find("frames=2\n") != std::string::npos);
  CHECK(frameEnds == 2 && submits == 3 && frame1Submits == 3);
  std::remove(path.c_str());
}

// VPVK_CAPTURE_FRAME=2 (set by the test registration) with a host that calls vpvk_frame_boundary: frames are the host's.
void testCaptureEnvHost() {
  const char* dir = std::getenv("VPVK_CAPTURE_DIR");
  if (!dir || !std::getenv("VPVK_CAPTURE_FRAME")) {
    std::puts("  skipped: run by ctest gpu_vulkan.capture_env_host");
    return;
  }
  for (auto& e : std::filesystem::directory_iterator(dir)) std::filesystem::remove(e.path());
  {
    CapRig r;
    r.fillBuffers(2);
    r.record(r.cb[0], true);
    r.submit({r.cb[0]});  // one submit before the first call (it counted as a submit frame): the call resets that
    vpvk_frame_boundary();
    CHECK(vpvk_capture_state() == 0);
    r.submit({r.cb[0]});  // two submits per frame from here: they would have started the capture as submit frames
    r.submit({r.cb[0]});
    CHECK(vpvk_capture_state() == 0);
    vpvk_frame_boundary();  // host frame 2: the capture starts
    CHECK(vpvk_capture_state() == 2);
    r.record(r.cb[2], true);
    r.submit({r.cb[2]});
    r.submit({r.cb[2]});
    vpvk_frame_boundary();  // the one frame ends
    CHECK(vpvk_capture_state() == 0);
  }
  int files = 0;
  for (auto& e : std::filesystem::directory_iterator(dir)) {
    ++files;
    vpcap::Reader rd;
    CHECK(rd.open(e.path().string()));
    vpcap::Call c;
    int frameEnds = 0, submits = 0;
    while (rd.next(c)) frameEnds += c.id == vpcap::id::vpcapFrameEnd, submits += c.id == vpcap::id::vkQueueSubmit;
    CHECK(rd.error().empty() && frameEnds == 1 && submits == 2);
    CHECK(rd.meta().find("frame_unit=host\n") != std::string::npos);
  }
  CHECK(files == 1);
}

// ---- capture: the device filter, state calls without a recording hook, the image budget, the locks ----
struct CapFile {
  struct Rec {
    uint16_t id;
    vpcap::Section section;
    uint32_t frame;
    std::vector<uint8_t> data;
  };
  std::vector<Rec> all;
  std::string meta;  // META + END text
  bool ok = false;
  explicit CapFile(const std::string& path) {
    vpcap::Reader rd;
    vpcap::Call c;
    if (!rd.open(path)) return;
    while (rd.next(c)) all.push_back({c.id, c.section, c.frame, std::vector<uint8_t>(c.data, c.data + c.size)});
    ok = rd.error().empty();
    meta = rd.meta();
  }
  int count(vpcap::Section s, uint16_t id) const {
    int n = 0;
    for (const Rec& x : all) n += x.section == s && x.id == id;
    return n;
  }
  std::vector<uint16_t> frame(uint32_t f) const {  // the calls of one frame, the memory / image data left out
    std::vector<uint16_t> v;
    for (const Rec& x : all)
      if (x.section == vpcap::Section::Frame && x.frame == f && x.id != vpcap::id::vpcapMemoryData && x.id != vpcap::id::vpcapImageData)
        v.push_back(x.id);
    return v;
  }
  static vpcap::Call call(const Rec& x) {
    vpcap::Call k;
    k.id = x.id, k.data = x.data.data(), k.size = uint32_t(x.data.size());
    return k;
  }
  bool has(const char* kv) const { return meta.find(kv) != std::string::npos; }
};

// Two devices in the process (a game's and an internal one, as with vpgles): the capture is of the device that submits the
// boundary; the other one's objects, calls and submits stay out of the file, and destroying it leaves the capture running.
void testCaptureDevices() {
  namespace I = vpcap::id;
  using vpcap::Section;
  const std::string path = (std::filesystem::temp_directory_path() / ("vpcap_dev_" + std::to_string(getpid()) + ".vpcap")).string();
  std::remove(path.c_str());
  vpvk_capture_track(0);
  CHECK(vpvk_capture_track(1) == 0);
  {
    CapRig a;
    auto b = std::make_unique<CapRig>();
    a.fillBuffers(1), b->fillBuffers(9);
    a.record(a.cb[0], true), b->record(b->cb[0], true);
    CHECK(vpvk_capture_request(1, path.c_str()) == 1);
    a.submit({a.cb[0]});  // the boundary: the capture is of A
    CHECK(vpvk_capture_state() == 2);
    a.record(a.cb[1], true);
    b->record(b->cb[1], true);  // begun during the capture, on the other device
    b->submit({b->cb[1]});      // not a frame of the capture
    CHECK(vpvk_capture_state() == 2);
    b.reset();  // destroying the other device does not close it
    CHECK(vpvk_capture_state() == 2);
    a.submit({a.cb[1]});
    CHECK(vpvk_capture_state() == 0);
  }
  vpvk_capture_track(0);
  const CapFile f(path);
  CHECK(f.ok);
  CHECK(f.count(Section::State, I::vkCreateInstance) == 1 && f.count(Section::State, I::vkCreateDevice) == 1);
  CHECK(f.count(Section::State, I::vkAllocateMemory) == 2 && f.count(Section::State, I::vkAllocateCommandBuffers) == 3);
  const std::vector<uint16_t> want = {I::vkQueueWaitIdle, I::vkBeginCommandBuffer, I::vkCmdBeginRenderPass, I::vkCmdBindPipeline,
                                      I::vkCmdBindDescriptorSets, I::vkCmdPushConstants, I::vkCmdBindVertexBuffers, I::vkCmdBindIndexBuffer,
                                      I::vkCmdSetViewport, I::vkCmdSetScissor, I::vkCmdDrawIndexed, I::vkCmdEndRenderPass,
                                      I::vkEndCommandBuffer, I::vkQueueSubmit, I::vpcapFrameEnd};
  CHECK(f.frame(0) == want);
  CHECK(f.has("frames=1\n") && f.has("other_device_submits=1\n") && !f.has("other_device_calls=0\n"));
  std::remove(path.c_str());
}

// State-changing calls that have no VPCAP record of their own: the ones with an equivalent are recorded as it, the others are
// counted (unhooked_used) and, when the replay cannot reproduce them, make the capture invalid.
void testCaptureUnhooked() {
  namespace I = vpcap::id;
  using vpcap::Section;
  const std::string path = (std::filesystem::temp_directory_path() / ("vpcap_unh_" + std::to_string(getpid()) + ".vpcap")).string();
  std::remove(path.c_str());
  vpvk_capture_track(0);
  CHECK(vpvk_capture_track(1) == 0);
  {
    CapRig r;
    CapApi& v = r.v;
    auto fn = [&](const char* n) {
      PFN_vkVoidFunction f = vkGetInstanceProcAddr(r.inst, n);
      CHECK(f != nullptr);
      return f;
    };
    const auto createTpl = reinterpret_cast<PFN_vkCreateDescriptorUpdateTemplate>(fn("vkCreateDescriptorUpdateTemplate"));
    const auto updateTpl = reinterpret_cast<PFN_vkUpdateDescriptorSetWithTemplate>(fn("vkUpdateDescriptorSetWithTemplate"));
    const auto bind2 = reinterpret_cast<PFN_vkBindBufferMemory2>(fn("vkBindBufferMemory2"));
    const auto next2 = reinterpret_cast<PFN_vkCmdNextSubpass2>(fn("vkCmdNextSubpass2"));
    const auto dispatchBase = reinterpret_cast<PFN_vkCmdDispatchBase>(fn("vkCmdDispatchBase"));
    const auto drawMulti = reinterpret_cast<PFN_vkCmdDrawMultiEXT>(fn("vkCmdDrawMultiEXT"));
    const auto setEvent = reinterpret_cast<PFN_vkSetEvent>(fn("vkSetEvent"));
    // The driver does not offer these: no game can call them, so there is nothing to record.
    CHECK(vkGetInstanceProcAddr(r.inst, "vkCmdPushDescriptorSetKHR") == nullptr);
    CHECK(vkGetInstanceProcAddr(r.inst, "vkCmdPushDescriptorSetWithTemplateKHR") == nullptr);
    CHECK(vkGetInstanceProcAddr(r.inst, "vkCmdSetFragmentShadingRateKHR") == nullptr);

    VkDescriptorUpdateTemplateEntry te{0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, sizeof(VkDescriptorBufferInfo)};
    VkDescriptorUpdateTemplateCreateInfo tci{VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO};
    tci.descriptorUpdateEntryCount = 1, tci.pDescriptorUpdateEntries = &te, tci.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;
    VkDescriptorUpdateTemplate tpl = nullptr;
    CHECK(createTpl(r.dev, &tci, nullptr, &tpl) == VK_SUCCESS);
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = 512, bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    VkBuffer late = nullptr;  // bound inside the capture
    CHECK(v.vkCreateBuffer(r.dev, &bci, nullptr, &late) == VK_SUCCESS);
    VkEvent ev = nullptr;
    VkEventCreateInfo eci{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
    CHECK(v.vkCreateEvent(r.dev, &eci, nullptr, &ev) == VK_SUCCESS);
    VkCommandBuffer cbs[2] = {};
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = r.cp, cai.commandBufferCount = 2;
    CHECK(v.vkAllocateCommandBuffers(r.dev, &cai, cbs) == VK_SUCCESS);
    auto begin = [&](VkCommandBuffer c) {
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      v.vkBeginCommandBuffer(c, &bi);
    };
    r.fillBuffers(2);
    r.record(r.cb[0], true);
    CHECK(vpvk_capture_request(1, path.c_str()) == 1);
    r.submit({r.cb[0]});  // the boundary
    CHECK(vpvk_capture_state() == 2);

    const VkDescriptorBufferInfo bi{r.vbib, 64, 128};
    updateTpl(r.dev, r.set, tpl, &bi);  // the set now points at vbib
    VkBindBufferMemoryInfo bmi{VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO};
    bmi.buffer = late, bmi.memory = r.host, bmi.memoryOffset = 20480;
    CHECK(bind2(r.dev, 1, &bmi) == VK_SUCCESS);
    VkSubpassBeginInfo sbi{VK_STRUCTURE_TYPE_SUBPASS_BEGIN_INFO};
    sbi.contents = VK_SUBPASS_CONTENTS_INLINE;
    VkSubpassEndInfo sei{VK_STRUCTURE_TYPE_SUBPASS_END_INFO};
    begin(cbs[0]);
    next2(cbs[0], &sbi, &sei);
    dispatchBase(cbs[0], 0, 0, 0, 1, 2, 3);
    v.vkEndCommandBuffer(cbs[0]);
    const VkMultiDrawInfoEXT md[2] = {{0, 3}, {3, 3}};
    begin(cbs[1]);
    dispatchBase(cbs[1], 1, 0, 0, 1, 1, 1);  // a workgroup-id offset: the driver drops it, the file cannot hold it
    drawMulti(cbs[1], 2, md, 1, 0, sizeof(VkMultiDrawInfoEXT));
    v.vkEndCommandBuffer(cbs[1]);
    setEvent(r.dev, ev);
    r.record(r.cb[1], true);
    r.submit({r.cb[1]});
    CHECK(vpvk_capture_state() == 0);
  }
  vpvk_capture_track(0);
  const CapFile f(path);
  CHECK(f.ok);
  uint64_t vbibId = 0, lateId = 0, hostId = 0, otherId = 0;
  int mem = 0;
  for (const auto& x : f.all) {
    if (x.section != Section::State) continue;
    if (x.id == I::vkCreateBuffer) {
      auto d = vpcap::decode<I::vkCreateBuffer>(CapFile::call(x));
      CHECK(d && d->ok);
      (std::get<1>(d->args)->size == 8192 ? vbibId : std::get<1>(d->args)->size == 512 ? lateId : otherId) = vpcap::idOfHandle(*std::get<3>(d->args));
    }
    if (x.id == I::vkAllocateMemory && mem++ == 0) hostId = vpcap::idOfHandle(*std::get<3>(vpcap::decode<I::vkAllocateMemory>(CapFile::call(x))->args));
  }
  CHECK(vbibId && lateId && hostId);
  const std::vector<uint16_t> want = {I::vkQueueWaitIdle, I::vkUpdateDescriptorSets, I::vkBindBufferMemory, I::vkBeginCommandBuffer,
                                      I::vkCmdNextSubpass, I::vkCmdDispatch, I::vkEndCommandBuffer, I::vkBeginCommandBuffer,
                                      I::vkEndCommandBuffer, I::vkBeginCommandBuffer, I::vkCmdBeginRenderPass, I::vkCmdBindPipeline,
                                      I::vkCmdBindDescriptorSets, I::vkCmdPushConstants, I::vkCmdBindVertexBuffers, I::vkCmdBindIndexBuffer,
                                      I::vkCmdSetViewport, I::vkCmdSetScissor, I::vkCmdDrawIndexed, I::vkCmdEndRenderPass,
                                      I::vkEndCommandBuffer, I::vkQueueSubmit, I::vpcapFrameEnd};
  CHECK(f.frame(0) == want);
  for (const auto& x : f.all) {
    if (x.section != Section::Frame) continue;
    if (x.id == I::vkUpdateDescriptorSets) {  // the template update, as the set's current contents
      auto d = vpcap::decode<I::vkUpdateDescriptorSets>(CapFile::call(x));
      CHECK(d && d->ok && std::get<1>(d->args) == 1);
      const VkWriteDescriptorSet& w = std::get<2>(d->args)[0];
      CHECK(w.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER && w.pBufferInfo && w.pBufferInfo[0].offset == 64 && w.pBufferInfo[0].range == 128);
      CHECK(w.pBufferInfo && vpcap::idOfHandle(w.pBufferInfo[0].buffer) == vbibId);
    }
    if (x.id == I::vkBindBufferMemory) {  // vkBindBufferMemory2 as a plain bind
      auto d = vpcap::decode<I::vkBindBufferMemory>(CapFile::call(x));
      CHECK(d && d->ok && vpcap::idOfHandle(std::get<1>(d->args)) == lateId && vpcap::idOfHandle(std::get<2>(d->args)) == hostId &&
            std::get<3>(d->args) == 20480);
    }
    if (x.id == I::vkCmdDispatch) {  // vkCmdDispatchBase with a zero base
      auto d = vpcap::decode<I::vkCmdDispatch>(CapFile::call(x));
      CHECK(d && d->ok && std::get<1>(d->args) == 1 && std::get<2>(d->args) == 2 && std::get<3>(d->args) == 3);
    }
  }
  CHECK(f.has("unhooked_used=2\n") && f.has("unhooked_used_calls=vkCmdDrawMultiEXT:1,vkSetEvent:1\n"));
  CHECK(f.has("invalid=1\n") && f.has("dispatch_base_offset") && f.has("unhooked:vkCmdDrawMultiEXT") && !f.has("unhooked:vkSetEvent"));
  std::remove(path.c_str());
}

// VPVK_CAPTURE_IMG_BUDGET_MB: the bytes of image contents read at one submit. Images over it wait for the next submit
// (img_deferred) and the capture says it is not exact (invalid).
void testCaptureBudget() {
  namespace I = vpcap::id;
  using vpcap::Section;
  const std::string path = (std::filesystem::temp_directory_path() / ("vpcap_bud_" + std::to_string(getpid()) + ".vpcap")).string();
  std::remove(path.c_str());
  setenv("VPVK_CAPTURE_IMG_BUDGET_MB", "0.0015", 1);  // 1572 bytes: two of the four 1 KiB images fit
  vpvk_capture_track(0);
  CHECK(vpvk_capture_track(1) == 0);
  {
    CapRig r;
    CapApi& v = r.v;
    auto& nb = static_cast<vpgpu::NullBackend&>(from<Device>(r.dev)->platform->backend());
    VkImage im[5] = {};  // four 16x16 RGBA (1 KiB) and the 4x4 destination
    for (int i = 0; i < 5; ++i) {
      VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      ii.imageType = VK_IMAGE_TYPE_2D, ii.format = VK_FORMAT_R8G8B8A8_UNORM, ii.extent = {i < 4 ? 16u : 4u, i < 4 ? 16u : 4u, 1};
      ii.mipLevels = 1, ii.arrayLayers = 1, ii.samples = VK_SAMPLE_COUNT_1_BIT, ii.tiling = VK_IMAGE_TILING_OPTIMAL;
      ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      CHECK(v.vkCreateImage(r.dev, &ii, nullptr, &im[i]) == VK_SUCCESS);
      CHECK(v.vkBindImageMemory(r.dev, im[i], r.devMem, 65536 * (i + 1)) == VK_SUCCESS);
    }
    VkCommandBuffer cbs[3] = {};
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = r.cp, cai.commandBufferCount = 3;
    CHECK(v.vkAllocateCommandBuffers(r.dev, &cai, cbs) == VK_SUCCESS);
    auto begin = [&](VkCommandBuffer c) {
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      v.vkBeginCommandBuffer(c, &bi);
    };
    auto copyOne = [&](VkCommandBuffer c, VkImage src) {
      const VkImageCopy reg{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {1, 1, 1}};
      v.vkCmdCopyImage(c, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, im[4], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
    };
    begin(cbs[0]);  // written by the GPU before the capture: full clears settle the zero-init
    const VkClearColorValue zero{};
    const VkImageSubresourceRange rg{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    for (int i = 0; i < 4; ++i) v.vkCmdClearColorImage(cbs[0], im[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &rg);
    v.vkEndCommandBuffer(cbs[0]);
    r.submit({cbs[0]});
    for (int i = 0; i < 4; ++i)
      nb.textureData[vpgpu::NullBackend::dataKey(from<Image>(im[i])->id, 0, 0)] = capPattern(uint8_t(10 + i), 16 * 16 * 4);
    CHECK(vpvk_capture_request(2, path.c_str()) == 1);
    r.submit({cbs[0]});  // the boundary
    CHECK(vpvk_capture_state() == 2);
    begin(cbs[1]);  // frame 0 reads all four for the first time
    for (int i = 0; i < 4; ++i) copyOne(cbs[1], im[i]);
    v.vkEndCommandBuffer(cbs[1]);
    r.submit({cbs[1]});
    begin(cbs[2]);  // frame 1 reads the first again
    copyOne(cbs[2], im[0]);
    v.vkEndCommandBuffer(cbs[2]);
    r.submit({cbs[2]});
    CHECK(vpvk_capture_state() == 0);
  }
  vpvk_capture_track(0);
  unsetenv("VPVK_CAPTURE_IMG_BUDGET_MB");
  const CapFile f(path);
  CHECK(f.ok);
  std::vector<uint64_t> ids;  // the four 16x16 images in creation order
  for (const auto& x : f.all)
    if (x.section == Section::State && x.id == I::vkCreateImage) {
      auto d = vpcap::decode<I::vkCreateImage>(CapFile::call(x));
      if (d && d->ok && std::get<1>(d->args)->extent.width == 16) ids.push_back(vpcap::idOfHandle(*std::get<3>(d->args)));
    }
  CHECK(ids.size() == 4);
  std::vector<std::set<uint64_t>> got(2);
  for (const auto& x : f.all) {
    vpcap::ImageData idt;
    if (x.id == I::vpcapImageData && vpcap::decodeImageData(CapFile::call(x), idt) && x.frame < 2) got[x.frame].insert(idt.image);
  }
  if (ids.size() == 4) {
    CHECK((got[0] == std::set<uint64_t>{ids[0], ids[1]}));  // what fits in the budget
    CHECK((got[1] == std::set<uint64_t>{ids[2], ids[3]}));  // the two that waited
  }
  CHECK(f.has("img_deferred=2\n") && f.has("img_deferred_pending=0\n"));
  CHECK(f.has("invalid=1\n") && f.has("img_budget_deferred"));
  std::remove(path.c_str());
}

// A submit that waits for the GPU to go idle (images read for the first time) must not hold the capture's lock: another thread
// that has to record a call before it can signal the timeline value a parked batch waits for would be stuck for the 2 s the
// idle wait gives up after.
void testCaptureLocks() {
  const std::string path = (std::filesystem::temp_directory_path() / ("vpcap_lock_" + std::to_string(getpid()) + ".vpcap")).string();
  std::remove(path.c_str());
  vpvk_capture_track(0);
  CHECK(vpvk_capture_track(1) == 0);
  long long ms = 0;
  {
    CapRig r;
    CapApi& v = r.v;
    auto& nb = static_cast<vpgpu::NullBackend&>(from<Device>(r.dev)->platform->backend());
    VkImage im[2] = {};  // the source (8x8) and the destination (4x4)
    for (int i = 0; i < 2; ++i) {
      VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      ii.imageType = VK_IMAGE_TYPE_2D, ii.format = VK_FORMAT_R8G8B8A8_UNORM, ii.extent = {i ? 4u : 8u, i ? 4u : 8u, 1};
      ii.mipLevels = 1, ii.arrayLayers = 1, ii.samples = VK_SAMPLE_COUNT_1_BIT, ii.tiling = VK_IMAGE_TILING_OPTIMAL;
      ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      CHECK(v.vkCreateImage(r.dev, &ii, nullptr, &im[i]) == VK_SUCCESS);
      CHECK(v.vkBindImageMemory(r.dev, im[i], r.devMem, 65536 * (i + 1)) == VK_SUCCESS);
    }
    const auto createSem = reinterpret_cast<PFN_vkCreateSemaphore>(vkGetInstanceProcAddr(r.inst, "vkCreateSemaphore"));
    const auto signal = reinterpret_cast<PFN_vkSignalSemaphore>(vkGetInstanceProcAddr(r.inst, "vkSignalSemaphore"));
    CHECK(createSem && signal);
    VkSemaphoreTypeCreateInfo sti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    sti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &sti};
    VkSemaphore sem = nullptr;
    CHECK(createSem(r.dev, &sci, nullptr, &sem) == VK_SUCCESS);
    VkCommandBuffer cbs[4] = {};  // up, parked, mid (stays open), copy
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = r.cp, cai.commandBufferCount = 4;
    CHECK(v.vkAllocateCommandBuffers(r.dev, &cai, cbs) == VK_SUCCESS);
    auto begin = [&](VkCommandBuffer c) {
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      v.vkBeginCommandBuffer(c, &bi);
    };
    begin(cbs[0]);
    const VkClearColorValue zero{};
    const VkImageSubresourceRange rg{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    v.vkCmdClearColorImage(cbs[0], im[0], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &rg);
    v.vkEndCommandBuffer(cbs[0]);
    r.submit({cbs[0]});
    nb.textureData[vpgpu::NullBackend::dataKey(from<Image>(im[0])->id, 0, 0)] = capPattern(7, 8 * 8 * 4);
    begin(cbs[1]);
    v.vkEndCommandBuffer(cbs[1]);
    CHECK(vpvk_capture_request(3, path.c_str()) == 1);
    r.submit({cbs[0]});  // the boundary
    CHECK(vpvk_capture_state() == 2);
    begin(cbs[2]);
    begin(cbs[3]);
    const VkImageCopy reg{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {1, 1, 1}};
    v.vkCmdCopyImage(cbs[3], im[0], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, im[1], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
    v.vkEndCommandBuffer(cbs[3]);

    // Thread B parks a batch on timeline value 1; thread C records a call (the capture's lock) and then signals it.
    const uint64_t wait = 1;
    VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    ts.waitSemaphoreValueCount = 1, ts.pWaitSemaphoreValues = &wait;
    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo parked{VK_STRUCTURE_TYPE_SUBMIT_INFO, &ts};
    parked.waitSemaphoreCount = 1, parked.pWaitSemaphores = &sem, parked.pWaitDstStageMask = &stage;
    parked.commandBufferCount = 1, parked.pCommandBuffers = &cbs[1];
    std::thread tb([&] { v.vkQueueSubmit(r.queue, 1, &parked, VK_NULL_HANDLE); });
    Device* d = from<Device>(r.dev);
    bool isParked = false;
    for (int i = 0; i < 2000 && !isParked; ++i) {
      {
        std::lock_guard<std::mutex> l(d->submitMutex);
        isParked = !d->deferred.empty();
      }
      if (!isParked) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(isParked);
    std::thread tc([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
      const VkViewport vp{0, 0, 8, 8, 0, 1};
      v.vkCmdSetViewport(cbs[2], 0, 1, &vp);
      VkSemaphoreSignalInfo sig{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
      sig.semaphore = sem, sig.value = 1;
      signal(r.dev, &sig);
    });
    VkSubmitInfo copy{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    copy.commandBufferCount = 1, copy.pCommandBuffers = &cbs[3];
    const auto t0 = std::chrono::steady_clock::now();
    CHECK(v.vkQueueSubmit(r.queue, 1, &copy, VK_NULL_HANDLE) == VK_SUCCESS);  // first read of im[0]: waits for the GPU
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    tb.join(), tc.join();
    v.vkEndCommandBuffer(cbs[2]);
    r.submit({cbs[2]});  // the third submit with commands: the capture is done
    CHECK(vpvk_capture_state() == 0);
  }
  vpvk_capture_track(0);
  CHECK(ms < 1500);
  const CapFile f(path);
  CHECK(f.ok && f.has("frames=3\n"));
  std::remove(path.c_str());
}

// The VPCAP call codec on structures the capture test above does not reach: encode, decode, compare. Handles are fake
// addresses; the stored ids are those addresses.
struct FakeIds final : vpcap::HandleIds {
  uint64_t idOf(const void* h) override { return reinterpret_cast<uintptr_t>(h); }
};
template <uint16_t ID, class... A>
std::unique_ptr<vpcap::Decoded<ID>> codecRoundTrip(uint32_t* dropped, size_t cut, A... a) {
  FakeIds ids;
  vpcap::Cod c(&ids);
  vpcap::ArgsOf<ID> t{a...};
  vpcap::CallDef<ID>::cod(c, t);
  if (dropped) *dropped = c.droppedNext;
  vpcap::Call call;
  call.id = ID, call.data = c.out.data(), call.size = uint32_t(c.out.size() - cut);
  return vpcap::decode<ID>(call);
}
template <class H>
H fake(uintptr_t v) {
  return reinterpret_cast<H>(v);
}

void testCaptureCodec() {
  namespace I = vpcap::id;
  using vpcap::idOfHandle;
  // vkCreateRenderPass: multiview kept, an unknown pNext struct in the chain dropped and counted.
  {
    VkAttachmentDescription att{};
    att.format = VK_FORMAT_R8G8B8A8_UNORM, att.samples = VK_SAMPLE_COUNT_4_BIT;
    VkAttachmentReference cref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}, dref{1, VK_IMAGE_LAYOUT_GENERAL};
    VkSubpassDescription sp{};
    sp.colorAttachmentCount = 1, sp.pColorAttachments = &cref, sp.pDepthStencilAttachment = &dref;
    const uint32_t mask = 0b11, corr = 0b11;
    VkRenderPassFragmentDensityMapCreateInfoEXT fdm{VK_STRUCTURE_TYPE_RENDER_PASS_FRAGMENT_DENSITY_MAP_CREATE_INFO_EXT};
    VkRenderPassMultiviewCreateInfo mv{VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO, &fdm};
    mv.subpassCount = 1, mv.pViewMasks = &mask, mv.correlationMaskCount = 1, mv.pCorrelationMasks = &corr;
    VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, &mv};
    ci.attachmentCount = 1, ci.pAttachments = &att, ci.subpassCount = 1, ci.pSubpasses = &sp;
    VkRenderPass out = fake<VkRenderPass>(0x77);
    uint32_t dropped = 0;
    auto d = codecRoundTrip<I::vkCreateRenderPass>(&dropped, 0, fake<VkDevice>(0x10), &ci, nullptr, &out);
    CHECK(d && d->ok && dropped == 1 && idOfHandle(std::get<0>(d->args)) == 0x10 && idOfHandle(*std::get<3>(d->args)) == 0x77);
    const VkRenderPassCreateInfo* r = std::get<1>(d->args);
    CHECK(r->pAttachments[0].samples == VK_SAMPLE_COUNT_4_BIT && r->pSubpasses[0].pColorAttachments[0].layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    CHECK(r->pSubpasses[0].pDepthStencilAttachment->layout == VK_IMAGE_LAYOUT_GENERAL && r->pSubpasses[0].pResolveAttachments == nullptr);
    const auto* m = static_cast<const VkRenderPassMultiviewCreateInfo*>(r->pNext);
    CHECK(m && m->sType == VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO && m->pViewMasks[0] == 3 && m->pCorrelationMasks[0] == 3 && !m->pNext);
    CHECK(codecRoundTrip<I::vkCreateRenderPass>(nullptr, 5, fake<VkDevice>(0x10), &ci, nullptr, &out)->ok == false);  // truncated
  }
  // vkCreateRenderPass2 with depth/stencil resolve in the subpass pNext.
  {
    VkAttachmentDescription2 at[2] = {{VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2}, {VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2}};
    at[1].format = VK_FORMAT_D32_SFLOAT;
    VkAttachmentReference2 colorRef{VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2}, depthRef{VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2},
        resolveRef{VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2};
    depthRef.attachment = 1, resolveRef.attachment = 1;
    VkSubpassDescriptionDepthStencilResolve dsr{VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE};
    dsr.pDepthStencilResolveAttachment = &resolveRef, dsr.depthResolveMode = VK_RESOLVE_MODE_MIN_BIT;
    VkSubpassDescription2 sp{VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2, &dsr};
    sp.colorAttachmentCount = 1, sp.pColorAttachments = &colorRef, sp.pDepthStencilAttachment = &depthRef, sp.viewMask = 3;
    VkSubpassDependency2 dep{VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    const uint32_t corr = 3;
    VkRenderPassCreateInfo2 ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2};
    ci.attachmentCount = 2, ci.pAttachments = at, ci.subpassCount = 1, ci.pSubpasses = &sp, ci.dependencyCount = 1;
    ci.pDependencies = &dep, ci.correlatedViewMaskCount = 1, ci.pCorrelatedViewMasks = &corr;
    VkRenderPass out = fake<VkRenderPass>(0x78);
    uint32_t dropped = 9;
    auto d = codecRoundTrip<I::vkCreateRenderPass2>(&dropped, 0, fake<VkDevice>(0x10), &ci, nullptr, &out);
    CHECK(d && d->ok && dropped == 0);
    const VkRenderPassCreateInfo2* r = std::get<1>(d->args);
    CHECK(r->pAttachments[1].format == VK_FORMAT_D32_SFLOAT && r->pDependencies[0].srcSubpass == VK_SUBPASS_EXTERNAL);
    CHECK(r->pSubpasses[0].viewMask == 3 && r->pSubpasses[0].pDepthStencilAttachment->attachment == 1 && r->pCorrelatedViewMasks[0] == 3);
    const auto* x = static_cast<const VkSubpassDescriptionDepthStencilResolve*>(r->pSubpasses[0].pNext);
    CHECK(x && x->depthResolveMode == VK_RESOLVE_MODE_MIN_BIT && x->pDepthStencilResolveAttachment->attachment == 1);
  }
  // vkCmdBeginRendering + vkCmdPipelineBarrier: handles inside nested arrays.
  {
    VkRenderingAttachmentInfo col{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}, dep{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    col.imageView = fake<VkImageView>(0x31), col.resolveImageView = fake<VkImageView>(0x32), col.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    col.clearValue.color.float32[2] = 0.25f;
    dep.imageView = fake<VkImageView>(0x33);
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.colorAttachmentCount = 1, ri.pColorAttachments = &col, ri.pDepthAttachment = &dep, ri.layerCount = 2, ri.viewMask = 3;
    auto d = codecRoundTrip<I::vkCmdBeginRendering>(nullptr, 0, fake<VkCommandBuffer>(0x20), &ri);
    CHECK(d && d->ok);
    const VkRenderingInfo* r = std::get<1>(d->args);
    CHECK(r->viewMask == 3 && idOfHandle(r->pColorAttachments[0].imageView) == 0x31 && idOfHandle(r->pColorAttachments[0].resolveImageView) == 0x32);
    CHECK(r->pColorAttachments[0].clearValue.color.float32[2] == 0.25f && idOfHandle(r->pDepthAttachment->imageView) == 0x33 && !r->pStencilAttachment);

    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = 1, mb.dstAccessMask = 2;
    VkBufferMemoryBarrier bb[2] = {{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}, {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}};
    bb[0].buffer = fake<VkBuffer>(0x41), bb[1].buffer = fake<VkBuffer>(0x42), bb[1].size = 99;
    VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    ib.image = fake<VkImage>(0x43), ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    auto b = codecRoundTrip<I::vkCmdPipelineBarrier>(nullptr, 0, fake<VkCommandBuffer>(0x20), VkPipelineStageFlags(1), VkPipelineStageFlags(2),
                                                     VkDependencyFlags(0), 1u, &mb, 2u, bb, 1u, &ib);
    CHECK(b && b->ok && std::get<5>(b->args)[0].dstAccessMask == 2 && idOfHandle(std::get<7>(b->args)[1].buffer) == 0x42);
    CHECK(std::get<7>(b->args)[1].size == 99 && idOfHandle(std::get<9>(b->args)[0].image) == 0x43 && std::get<9>(b->args)[0].newLayout == VK_IMAGE_LAYOUT_GENERAL);
  }
  // vkCreateComputePipelines: the embedded stage, name, specialization constants.
  {
    const VkSpecializationMapEntry me{7, 0, 4};
    const uint32_t value = 0xCAFEF00D;
    VkSpecializationInfo spec{1, &me, 4, &value};
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT, ci.stage.module = fake<VkShaderModule>(0x51), ci.stage.pName = "cs_main";
    ci.stage.pSpecializationInfo = &spec, ci.layout = fake<VkPipelineLayout>(0x52);
    VkPipeline out[2] = {fake<VkPipeline>(0x53), fake<VkPipeline>(0x54)};
    auto d = codecRoundTrip<I::vkCreateComputePipelines>(nullptr, 0, fake<VkDevice>(0x10), fake<VkPipelineCache>(0x55), 1u, &ci, nullptr, out);
    CHECK(d && d->ok && std::get<1>(d->args) == nullptr);  // the cache is not stored
    const VkComputePipelineCreateInfo& r = std::get<3>(d->args)[0];
    CHECK(std::strcmp(r.stage.pName, "cs_main") == 0 && idOfHandle(r.stage.module) == 0x51 && idOfHandle(r.layout) == 0x52);
    CHECK(r.stage.pSpecializationInfo->mapEntryCount == 1 && r.stage.pSpecializationInfo->pMapEntries[0].constantID == 7);
    uint32_t got = 0;
    std::memcpy(&got, r.stage.pSpecializationInfo->pData, 4);
    CHECK(got == 0xCAFEF00D && idOfHandle(std::get<5>(d->args)[0]) == 0x53);
  }
  // vkCreateDescriptorSetLayout: immutable samplers kept for a sampler binding, the junk pointer of a buffer binding ignored.
  {
    VkSampler smp[2] = {fake<VkSampler>(0x61), fake<VkSampler>(0x62)};
    VkDescriptorSetLayoutBinding b[2] = {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, VK_SHADER_STAGE_FRAGMENT_BIT, smp},
                                         {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, smp}};
    VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ci.bindingCount = 2, ci.pBindings = b;
    VkDescriptorSetLayout out = fake<VkDescriptorSetLayout>(0x63);
    auto d = codecRoundTrip<I::vkCreateDescriptorSetLayout>(nullptr, 0, fake<VkDevice>(0x10), &ci, nullptr, &out);
    CHECK(d && d->ok);
    const VkDescriptorSetLayoutCreateInfo* r = std::get<1>(d->args);
    CHECK(r->pBindings[0].pImmutableSamplers && idOfHandle(r->pBindings[0].pImmutableSamplers[1]) == 0x62 && !r->pBindings[1].pImmutableSamplers);
  }
  // vkUpdateDescriptorSets: image, texel, inline block and a copy.
  {
    const VkDescriptorImageInfo ii{fake<VkSampler>(0x71), fake<VkImageView>(0x72), VK_IMAGE_LAYOUT_GENERAL};
    VkBufferView tv = fake<VkBufferView>(0x73);
    const uint8_t blob[6] = {1, 2, 3, 4, 5, 6};
    VkWriteDescriptorSetInlineUniformBlock inl{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK};
    inl.dataSize = 6, inl.pData = blob;
    VkWriteDescriptorSet w[3] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET},
                                 {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &inl}};
    w[0].dstSet = fake<VkDescriptorSet>(0x74), w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, w[0].descriptorCount = 1;
    w[0].pImageInfo = &ii;
    w[1].dstSet = fake<VkDescriptorSet>(0x74), w[1].dstBinding = 2, w[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    w[1].descriptorCount = 1, w[1].pTexelBufferView = &tv;
    w[2].dstSet = fake<VkDescriptorSet>(0x74), w[2].dstBinding = 3, w[2].descriptorType = VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK;
    w[2].descriptorCount = 6;
    VkCopyDescriptorSet cp{VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET};
    cp.srcSet = fake<VkDescriptorSet>(0x75), cp.dstSet = fake<VkDescriptorSet>(0x74), cp.descriptorCount = 3;
    auto d = codecRoundTrip<I::vkUpdateDescriptorSets>(nullptr, 0, fake<VkDevice>(0x10), 3u, w, 1u, &cp);
    CHECK(d && d->ok);
    const VkWriteDescriptorSet* r = std::get<2>(d->args);
    CHECK(idOfHandle(r[0].dstSet) == 0x74 && idOfHandle(r[0].pImageInfo[0].imageView) == 0x72 && idOfHandle(r[0].pImageInfo[0].sampler) == 0x71);
    CHECK(r[0].pBufferInfo == nullptr && idOfHandle(r[1].pTexelBufferView[0]) == 0x73 && r[1].pImageInfo == nullptr);
    const auto* ib = static_cast<const VkWriteDescriptorSetInlineUniformBlock*>(r[2].pNext);
    CHECK(ib && ib->dataSize == 6 && std::memcmp(ib->pData, blob, 6) == 0);
    CHECK(idOfHandle(std::get<4>(d->args)[0].srcSet) == 0x75 && std::get<4>(d->args)[0].descriptorCount == 3);
  }
  // Commands with counts: dynamic offsets, copy regions, clears.
  {
    VkDescriptorSet sets[2] = {fake<VkDescriptorSet>(0x81), fake<VkDescriptorSet>(0x82)};
    const uint32_t dyn[3] = {16, 32, 48};
    auto d = codecRoundTrip<I::vkCmdBindDescriptorSets>(nullptr, 0, fake<VkCommandBuffer>(0x20), VK_PIPELINE_BIND_POINT_COMPUTE,
                                                        fake<VkPipelineLayout>(0x83), 1u, 2u, sets, 3u, dyn);
    CHECK(d && d->ok && std::get<3>(d->args) == 1 && idOfHandle(std::get<5>(d->args)[1]) == 0x82 && std::get<7>(d->args)[2] == 48);
    const VkBufferImageCopy reg[2] = {{0, 16, 16, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {16, 16, 1}},
                                      {512, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, 1}, {1, 2, 3}, {8, 8, 1}}};
    auto c = codecRoundTrip<I::vkCmdCopyBufferToImage>(nullptr, 0, fake<VkCommandBuffer>(0x20), fake<VkBuffer>(0x84), fake<VkImage>(0x85),
                                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 2u, reg);
    CHECK(c && c->ok && std::get<5>(c->args)[1].bufferOffset == 512 && std::get<5>(c->args)[1].imageOffset.z == 3);
    VkClearColorValue color{};
    color.uint32[3] = 9;
    const VkImageSubresourceRange rg{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 2};
    auto k = codecRoundTrip<I::vkCmdClearColorImage>(nullptr, 0, fake<VkCommandBuffer>(0x20), fake<VkImage>(0x86), VK_IMAGE_LAYOUT_GENERAL, &color, 1u, &rg);
    CHECK(k && k->ok && std::get<3>(k->args)->uint32[3] == 9 && std::get<5>(k->args)[0].layerCount == 2);
    // An absurd count in a corrupt payload must not allocate or read out of bounds.
    std::vector<uint8_t> bad(40, 0xFF);
    vpcap::Call call;
    call.id = I::vkCmdBindVertexBuffers, call.data = bad.data(), call.size = uint32_t(bad.size());
    auto b = vpcap::decode<I::vkCmdBindVertexBuffers>(call);
    CHECK(b && !b->ok);
  }
  // vkQueueSubmit2: waits, command buffers (with device mask) and signals per batch.
  {
    VkSemaphoreSubmitInfo wait[2] = {{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr, fake<VkSemaphore>(0x91), 5, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0},
                                     {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr, fake<VkSemaphore>(0x92), 6, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 1}};
    VkCommandBufferSubmitInfo cbi[2] = {{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, nullptr, fake<VkCommandBuffer>(0x93), 0},
                                        {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, nullptr, fake<VkCommandBuffer>(0x94), 3}};
    VkSemaphoreSubmitInfo sig{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr, fake<VkSemaphore>(0x95), 7, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0};
    VkSubmitInfo2 si[2] = {{VK_STRUCTURE_TYPE_SUBMIT_INFO_2}, {VK_STRUCTURE_TYPE_SUBMIT_INFO_2}};
    si[0].waitSemaphoreInfoCount = 2, si[0].pWaitSemaphoreInfos = wait, si[0].commandBufferInfoCount = 2, si[0].pCommandBufferInfos = cbi;
    si[0].signalSemaphoreInfoCount = 1, si[0].pSignalSemaphoreInfos = &sig;
    auto d = codecRoundTrip<I::vkQueueSubmit2>(nullptr, 0, fake<VkQueue>(0x90), 2u, si, fake<VkFence>(0x96));
    CHECK(d && d->ok && idOfHandle(std::get<0>(d->args)) == 0x90 && std::get<1>(d->args) == 2 && idOfHandle(std::get<3>(d->args)) == 0x96);
    const VkSubmitInfo2* r = std::get<2>(d->args);
    CHECK(r[0].waitSemaphoreInfoCount == 2 && idOfHandle(r[0].pWaitSemaphoreInfos[1].semaphore) == 0x92 && r[0].pWaitSemaphoreInfos[1].value == 6);
    CHECK(r[0].pWaitSemaphoreInfos[1].stageMask == VK_PIPELINE_STAGE_2_TRANSFER_BIT && r[0].pWaitSemaphoreInfos[1].deviceIndex == 1);
    CHECK(r[0].commandBufferInfoCount == 2 && idOfHandle(r[0].pCommandBufferInfos[1].commandBuffer) == 0x94 && r[0].pCommandBufferInfos[1].deviceMask == 3);
    CHECK(r[0].signalSemaphoreInfoCount == 1 && idOfHandle(r[0].pSignalSemaphoreInfos[0].semaphore) == 0x95 && r[0].pSignalSemaphoreInfos[0].value == 7);
    CHECK(r[1].commandBufferInfoCount == 0 && r[1].pCommandBufferInfos == nullptr && !r[0].pNext);
    CHECK(codecRoundTrip<I::vkQueueSubmit2>(nullptr, 3, fake<VkQueue>(0x90), 2u, si, fake<VkFence>(0x96))->ok == false);
  }
  // The copy2 family: a region with a pNext this build does not store is dropped and counted.
  {
    VkRenderPassFragmentDensityMapCreateInfoEXT unknown{VK_STRUCTURE_TYPE_RENDER_PASS_FRAGMENT_DENSITY_MAP_CREATE_INFO_EXT};
    VkBufferCopy2 bc[2] = {{VK_STRUCTURE_TYPE_BUFFER_COPY_2, nullptr, 1, 2, 3}, {VK_STRUCTURE_TYPE_BUFFER_COPY_2, &unknown, 4, 5, 6}};
    VkCopyBufferInfo2 cb2{VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2};
    cb2.srcBuffer = fake<VkBuffer>(0xA1), cb2.dstBuffer = fake<VkBuffer>(0xA2), cb2.regionCount = 2, cb2.pRegions = bc;
    uint32_t dropped = 0;
    auto a = codecRoundTrip<I::vkCmdCopyBuffer2>(&dropped, 0, fake<VkCommandBuffer>(0x20), &cb2);
    CHECK(a && a->ok && dropped == 1);
    const VkCopyBufferInfo2* r = std::get<1>(a->args);
    CHECK(idOfHandle(r->srcBuffer) == 0xA1 && idOfHandle(r->dstBuffer) == 0xA2 && r->regionCount == 2 && r->pRegions[1].size == 6 && r->pRegions[0].srcOffset == 1);
    CHECK(r->pRegions[1].sType == VK_STRUCTURE_TYPE_BUFFER_COPY_2 && r->pRegions[1].pNext == nullptr);

    VkImageCopy2 ic{VK_STRUCTURE_TYPE_IMAGE_COPY_2};
    ic.srcOffset = {1, 2, 3}, ic.extent = {3, 4, 5}, ic.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 2, 1, 1};
    VkCopyImageInfo2 ci{VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2};
    ci.srcImage = fake<VkImage>(0xA3), ci.srcImageLayout = VK_IMAGE_LAYOUT_GENERAL, ci.dstImage = fake<VkImage>(0xA4);
    ci.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, ci.regionCount = 1, ci.pRegions = &ic;
    auto b = codecRoundTrip<I::vkCmdCopyImage2>(nullptr, 0, fake<VkCommandBuffer>(0x20), &ci);
    CHECK(b && b->ok && idOfHandle(std::get<1>(b->args)->srcImage) == 0xA3 && idOfHandle(std::get<1>(b->args)->dstImage) == 0xA4);
    CHECK(std::get<1>(b->args)->dstImageLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && std::get<1>(b->args)->pRegions[0].srcOffset.z == 3);
    CHECK(std::get<1>(b->args)->pRegions[0].extent.height == 4 && std::get<1>(b->args)->pRegions[0].dstSubresource.mipLevel == 2);

    VkBufferImageCopy2 bi{VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2};
    bi.bufferOffset = 77, bi.imageExtent = {9, 8, 1};
    VkCopyBufferToImageInfo2 b2i{VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2};
    b2i.srcBuffer = fake<VkBuffer>(0xA5), b2i.dstImage = fake<VkImage>(0xA6), b2i.regionCount = 1, b2i.pRegions = &bi;
    auto c1 = codecRoundTrip<I::vkCmdCopyBufferToImage2>(nullptr, 0, fake<VkCommandBuffer>(0x20), &b2i);
    CHECK(c1 && c1->ok && idOfHandle(std::get<1>(c1->args)->srcBuffer) == 0xA5 && idOfHandle(std::get<1>(c1->args)->dstImage) == 0xA6);
    CHECK(std::get<1>(c1->args)->pRegions[0].bufferOffset == 77 && std::get<1>(c1->args)->pRegions[0].imageExtent.width == 9);
    VkCopyImageToBufferInfo2 i2b{VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2};
    i2b.srcImage = fake<VkImage>(0xA7), i2b.dstBuffer = fake<VkBuffer>(0xA8), i2b.regionCount = 1, i2b.pRegions = &bi;
    auto c2 = codecRoundTrip<I::vkCmdCopyImageToBuffer2>(nullptr, 0, fake<VkCommandBuffer>(0x20), &i2b);
    CHECK(c2 && c2->ok && idOfHandle(std::get<1>(c2->args)->srcImage) == 0xA7 && idOfHandle(std::get<1>(c2->args)->dstBuffer) == 0xA8);

    VkImageBlit2 bl{VK_STRUCTURE_TYPE_IMAGE_BLIT_2};
    bl.srcOffsets[1] = {8, 8, 1}, bl.dstOffsets[0] = {1, 1, 0}, bl.dstOffsets[1] = {4, 4, 1};
    VkBlitImageInfo2 bli{VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2};
    bli.srcImage = fake<VkImage>(0xA9), bli.dstImage = fake<VkImage>(0xAA), bli.regionCount = 1, bli.pRegions = &bl, bli.filter = VK_FILTER_LINEAR;
    auto d = codecRoundTrip<I::vkCmdBlitImage2>(nullptr, 0, fake<VkCommandBuffer>(0x20), &bli);
    CHECK(d && d->ok && std::get<1>(d->args)->filter == VK_FILTER_LINEAR && std::get<1>(d->args)->pRegions[0].dstOffsets[0].y == 1);
    CHECK(std::get<1>(d->args)->pRegions[0].srcOffsets[1].x == 8 && idOfHandle(std::get<1>(d->args)->dstImage) == 0xAA);

    VkImageResolve2 rs{VK_STRUCTURE_TYPE_IMAGE_RESOLVE_2};
    rs.extent = {16, 16, 1};
    VkResolveImageInfo2 rsi{VK_STRUCTURE_TYPE_RESOLVE_IMAGE_INFO_2};
    rsi.srcImage = fake<VkImage>(0xAB), rsi.dstImage = fake<VkImage>(0xAC), rsi.regionCount = 1, rsi.pRegions = &rs;
    auto e = codecRoundTrip<I::vkCmdResolveImage2>(nullptr, 0, fake<VkCommandBuffer>(0x20), &rsi);
    CHECK(e && e->ok && idOfHandle(std::get<1>(e->args)->srcImage) == 0xAB && std::get<1>(e->args)->pRegions[0].extent.width == 16);
  }
  // Barriers and events of both generations, with handles inside the nested arrays.
  {
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = 11, mb.dstAccessMask = 12;
    VkBufferMemoryBarrier2 bb[2] = {{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2}, {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2}};
    bb[0].buffer = fake<VkBuffer>(0xB1), bb[1].buffer = fake<VkBuffer>(0xB2), bb[1].size = 99;
    VkImageMemoryBarrier2 ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    ib.image = fake<VkImage>(0xB3), ib.newLayout = VK_IMAGE_LAYOUT_GENERAL, ib.subresourceRange.layerCount = 4;
    VkDependencyInfo dep[2] = {{VK_STRUCTURE_TYPE_DEPENDENCY_INFO}, {VK_STRUCTURE_TYPE_DEPENDENCY_INFO}};
    dep[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT, dep[0].memoryBarrierCount = 1, dep[0].pMemoryBarriers = &mb;
    dep[0].bufferMemoryBarrierCount = 2, dep[0].pBufferMemoryBarriers = bb, dep[0].imageMemoryBarrierCount = 1, dep[0].pImageMemoryBarriers = &ib;
    auto check = [&](const VkDependencyInfo& r) {
      CHECK(r.dependencyFlags == VK_DEPENDENCY_BY_REGION_BIT && r.memoryBarrierCount == 1 && r.pMemoryBarriers[0].srcStageMask == 11 && r.pMemoryBarriers[0].dstAccessMask == 12);
      CHECK(r.bufferMemoryBarrierCount == 2 && idOfHandle(r.pBufferMemoryBarriers[1].buffer) == 0xB2 && r.pBufferMemoryBarriers[1].size == 99);
      CHECK(r.imageMemoryBarrierCount == 1 && idOfHandle(r.pImageMemoryBarriers[0].image) == 0xB3 && r.pImageMemoryBarriers[0].subresourceRange.layerCount == 4);
    };
    auto a = codecRoundTrip<I::vkCmdPipelineBarrier2>(nullptr, 0, fake<VkCommandBuffer>(0x20), &dep[0]);
    CHECK(a && a->ok);
    check(*std::get<1>(a->args));
    auto b = codecRoundTrip<I::vkCmdSetEvent2>(nullptr, 0, fake<VkCommandBuffer>(0x20), fake<VkEvent>(0xB4), &dep[0]);
    CHECK(b && b->ok && idOfHandle(std::get<1>(b->args)) == 0xB4);
    check(*std::get<2>(b->args));
    VkEvent evs[2] = {fake<VkEvent>(0xB5), fake<VkEvent>(0xB6)};
    auto c = codecRoundTrip<I::vkCmdWaitEvents2>(nullptr, 0, fake<VkCommandBuffer>(0x20), 2u, evs, dep);
    CHECK(c && c->ok && std::get<1>(c->args) == 2 && idOfHandle(std::get<2>(c->args)[1]) == 0xB6);
    check(std::get<3>(c->args)[0]);
    CHECK(std::get<3>(c->args)[1].memoryBarrierCount == 0 && std::get<3>(c->args)[1].pMemoryBarriers == nullptr);
    auto d = codecRoundTrip<I::vkCmdResetEvent2>(nullptr, 0, fake<VkCommandBuffer>(0x20), fake<VkEvent>(0xB4), VkPipelineStageFlags2(0x1'0000'0001ull));
    CHECK(d && d->ok && std::get<2>(d->args) == 0x1'0000'0001ull);

    VkMemoryBarrier m1{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    m1.srcAccessMask = 3;
    VkBufferMemoryBarrier b1{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    b1.buffer = fake<VkBuffer>(0xB7), b1.size = 5;
    VkImageMemoryBarrier i1{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    i1.image = fake<VkImage>(0xB8), i1.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    auto e = codecRoundTrip<I::vkCmdWaitEvents>(nullptr, 0, fake<VkCommandBuffer>(0x20), 2u, evs, VkPipelineStageFlags(0x10), VkPipelineStageFlags(0x20),
                                               1u, &m1, 1u, &b1, 1u, &i1);
    CHECK(e && e->ok && std::get<1>(e->args) == 2 && idOfHandle(std::get<2>(e->args)[0]) == 0xB5 && std::get<3>(e->args) == 0x10 && std::get<4>(e->args) == 0x20);
    CHECK(std::get<6>(e->args)[0].srcAccessMask == 3 && idOfHandle(std::get<8>(e->args)[0].buffer) == 0xB7 && idOfHandle(std::get<10>(e->args)[0].image) == 0xB8);
    auto f = codecRoundTrip<I::vkCmdSetEvent>(nullptr, 0, fake<VkCommandBuffer>(0x20), fake<VkEvent>(0xB9), VkPipelineStageFlags(0x40));
    CHECK(f && f->ok && idOfHandle(std::get<1>(f->args)) == 0xB9 && std::get<2>(f->args) == 0x40);
    auto g = codecRoundTrip<I::vkCmdResetEvent>(nullptr, 0, fake<VkCommandBuffer>(0x20), fake<VkEvent>(0xBA), VkPipelineStageFlags(0x80));
    CHECK(g && g->ok && idOfHandle(std::get<1>(g->args)) == 0xBA && std::get<2>(g->args) == 0x80);
  }
  // Queries, timestamps, indirect-count draws, dynamic vertex strides, the query pool and a dedicated allocation.
  {
    auto a = codecRoundTrip<I::vkCmdBeginQuery>(nullptr, 0, fake<VkCommandBuffer>(0x20), fake<VkQueryPool>(0xC1), 3u, VkQueryControlFlags(VK_QUERY_CONTROL_PRECISE_BIT));
    CHECK(a && a->ok && idOfHandle(std::get<1>(a->args)) == 0xC1 && std::get<2>(a->args) == 3 && std::get<3>(a->args) == VK_QUERY_CONTROL_PRECISE_BIT);
    auto b = codecRoundTrip<I::vkCmdEndQuery>(nullptr, 0, fake<VkCommandBuffer>(0x20), fake<VkQueryPool>(0xC1), 3u);
    CHECK(b && b->ok && std::get<2>(b->args) == 3);
    auto c = codecRoundTrip<I::vkCmdResetQueryPool>(nullptr, 0, fake<VkCommandBuffer>(0x20), fake<VkQueryPool>(0xC1), 1u, 7u);
    CHECK(c && c->ok && std::get<2>(c->args) == 1 && std::get<3>(c->args) == 7);
    auto d = codecRoundTrip<I::vkCmdWriteTimestamp>(nullptr, 0, fake<VkCommandBuffer>(0x20), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, fake<VkQueryPool>(0xC2), 5u);
    CHECK(d && d->ok && std::get<1>(d->args) == VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT && idOfHandle(std::get<2>(d->args)) == 0xC2 && std::get<3>(d->args) == 5);
    auto e = codecRoundTrip<I::vkCmdWriteTimestamp2>(nullptr, 0, fake<VkCommandBuffer>(0x20), VkPipelineStageFlags2(0x1'0000'0000ull), fake<VkQueryPool>(0xC2), 6u);
    CHECK(e && e->ok && std::get<1>(e->args) == 0x1'0000'0000ull && std::get<3>(e->args) == 6);
    auto f = codecRoundTrip<I::vkCmdDrawIndirectCount>(nullptr, 0, fake<VkCommandBuffer>(0x20), fake<VkBuffer>(0xC3), VkDeviceSize(16), fake<VkBuffer>(0xC4),
                                                      VkDeviceSize(32), 100u, 20u);
    CHECK(f && f->ok && idOfHandle(std::get<1>(f->args)) == 0xC3 && std::get<2>(f->args) == 16 && idOfHandle(std::get<3>(f->args)) == 0xC4 &&
          std::get<4>(f->args) == 32 && std::get<5>(f->args) == 100 && std::get<6>(f->args) == 20);
    auto g = codecRoundTrip<I::vkCmdDrawIndexedIndirectCount>(nullptr, 0, fake<VkCommandBuffer>(0x20), fake<VkBuffer>(0xC3), VkDeviceSize(0), fake<VkBuffer>(0xC4),
                                                             VkDeviceSize(8), 1u, 24u);
    CHECK(g && g->ok && std::get<4>(g->args) == 8 && std::get<5>(g->args) == 1 && std::get<6>(g->args) == 24);
    VkBuffer bufs[2] = {fake<VkBuffer>(0xC5), fake<VkBuffer>(0xC6)};
    const VkDeviceSize offs[2] = {0, 64}, sizes[2] = {128, VK_WHOLE_SIZE}, strides[2] = {16, 32};
    auto h = codecRoundTrip<I::vkCmdBindVertexBuffers2>(nullptr, 0, fake<VkCommandBuffer>(0x20), 1u, 2u, bufs, offs, sizes, strides);
    CHECK(h && h->ok && std::get<1>(h->args) == 1 && idOfHandle(std::get<3>(h->args)[1]) == 0xC6 && std::get<4>(h->args)[1] == 64 &&
          std::get<5>(h->args)[1] == VK_WHOLE_SIZE && std::get<6>(h->args)[1] == 32);
    auto h2 = codecRoundTrip<I::vkCmdBindVertexBuffers2>(nullptr, 0, fake<VkCommandBuffer>(0x20), 0u, 2u, bufs, offs, static_cast<const VkDeviceSize*>(nullptr),
                                                        static_cast<const VkDeviceSize*>(nullptr));
    CHECK(h2 && h2->ok && std::get<5>(h2->args) == nullptr && std::get<6>(h2->args) == nullptr && std::get<4>(h2->args)[1] == 64);

    VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qci.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS, qci.queryCount = 12, qci.pipelineStatistics = 0x3FF;
    VkQueryPool out = fake<VkQueryPool>(0xC7);
    auto i = codecRoundTrip<I::vkCreateQueryPool>(nullptr, 0, fake<VkDevice>(0x10), &qci, nullptr, &out);
    CHECK(i && i->ok && std::get<1>(i->args)->queryCount == 12 && std::get<1>(i->args)->pipelineStatistics == 0x3FF && idOfHandle(*std::get<3>(i->args)) == 0xC7);
    auto j = codecRoundTrip<I::vkDestroyQueryPool>(nullptr, 0, fake<VkDevice>(0x10), fake<VkQueryPool>(0xC7), nullptr);
    CHECK(j && j->ok && idOfHandle(std::get<1>(j->args)) == 0xC7);

    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, &flags};
    ded.image = fake<VkImage>(0xC8);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &ded};
    ai.allocationSize = 4096;
    VkDeviceMemory mem = fake<VkDeviceMemory>(0xC9);
    uint32_t dropped = 9;
    auto k = codecRoundTrip<I::vkAllocateMemory>(&dropped, 0, fake<VkDevice>(0x10), &ai, nullptr, &mem);
    CHECK(k && k->ok && dropped == 0);
    const auto* dd = static_cast<const VkMemoryDedicatedAllocateInfo*>(std::get<1>(k->args)->pNext);
    CHECK(dd && dd->sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO && idOfHandle(dd->image) == 0xC8 && dd->buffer == nullptr);
    const auto* ff = dd ? static_cast<const VkMemoryAllocateFlagsInfo*>(dd->pNext) : nullptr;
    CHECK(ff && ff->sType == VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO && ff->flags == VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT && !ff->pNext);
  }
  // An image data record: header fields and the byte range.
  {
    std::vector<uint8_t> rec(vpcap::kImageDataHeader + 5, 0);
    const uint64_t image = 0x1234;
    const uint32_t f[6] = {2, 3, 4, 5, 6, 7};
    const uint64_t t[3] = {100, 40, 5};
    std::memcpy(rec.data(), &image, 8);
    std::memcpy(rec.data() + 8, f, sizeof f);
    std::memcpy(rec.data() + 40, t, sizeof t);
    for (int i = 0; i < 5; ++i) rec[vpcap::kImageDataHeader + i] = uint8_t(i + 1);
    vpcap::Call call;
    call.id = I::vpcapImageData, call.data = rec.data(), call.size = uint32_t(rec.size());
    vpcap::ImageData im;
    CHECK(vpcap::decodeImageData(call, im) && im.image == 0x1234 && im.mip == 2 && im.layer == 3 && im.aspect == 4 && im.width == 5 && im.height == 6 && im.depth == 7);
    CHECK(im.total == 100 && im.offset == 40 && im.size == 5 && im.data[4] == 5);
    call.size -= 1;  // the payload ends before the bytes it announces
    CHECK(!vpcap::decodeImageData(call, im));
  }
}

// ---- vpreplay and its image tools (tools/vpimage.h, tools/vpimgdiff.h, tools/vpreplay.cpp) ----
#include "vpimgdiff.h"

namespace {

vpimg::Image rgba(uint32_t w, uint32_t h, uint32_t bits, const std::function<void(uint32_t, uint32_t, uint16_t*)>& f) {
  vpimg::Image im;
  im.w = w, im.h = h, im.bits = bits;
  im.px.resize(size_t(w) * h * 4);
  for (uint32_t y = 0; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x) f(x, y, &im.px[(size_t(y) * w + x) * 4]);
  return im;
}
vpimg::Image gradient(uint32_t w, uint32_t h, uint32_t bits) {
  const uint32_t m = bits == 16 ? 65535 : 255;
  return rgba(w, h, bits, [&](uint32_t x, uint32_t y, uint16_t* p) {
    p[0] = uint16_t(x * m / w), p[1] = uint16_t(y * m / h), p[2] = uint16_t((x * 7 + y * 3) % (m + 1)), p[3] = uint16_t(m - x);
  });
}
// A PNG whose rows all use filter `ft` (the filters the decoder must undo), 8-bit RGBA.
std::vector<uint8_t> filteredPng(const vpimg::Image& im, int ft) {
  const size_t row = size_t(im.w) * 4;
  std::vector<uint8_t> raw, prev(row, 0), cur(row);
  for (uint32_t y = 0; y < im.h; ++y) {
    for (size_t i = 0; i < row; ++i) cur[i] = uint8_t(im.px[y * row + i]);
    raw.push_back(uint8_t(ft));
    for (size_t i = 0; i < row; ++i) {
      const int a = i >= 4 ? cur[i - 4] : 0, b = prev[i], c = i >= 4 ? prev[i - 4] : 0;
      int pr = 0;
      if (ft == 1) pr = a;
      else if (ft == 2) pr = b;
      else if (ft == 3) pr = (a + b) / 2;
      else if (ft == 4) {
        const int p = a + b - c, pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
        pr = (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
      }
      raw.push_back(uint8_t(cur[i] - pr));
    }
    prev = cur;
  }
  std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A}, ihdr;
  vpimg::putBE32(ihdr, im.w), vpimg::putBE32(ihdr, im.h);
  for (uint8_t b : {8, 6, 0, 0, 0}) ihdr.push_back(b);
  vpimg::pngChunk(out, "IHDR", ihdr);
  uLongf zn = compressBound(uLong(raw.size()));
  std::vector<uint8_t> z(zn);
  compress2(z.data(), &zn, raw.data(), uLong(raw.size()), 6);
  z.resize(zn);
  vpimg::pngChunk(out, "IDAT", z);
  vpimg::pngChunk(out, "IEND", {});
  return out;
}

}  // namespace

void testPng() {
  for (const uint32_t bits : {8u, 16u}) {
    const vpimg::Image a = gradient(13, 7, bits);
    const std::vector<uint8_t> png = vpimg::pngEncode(a);
    CHECK(png.size() > 8 && !std::memcmp(png.data(), "\x89PNG\r\n\x1a\n", 8));
    vpimg::Image b;
    std::string err;
    CHECK(vpimg::pngDecode(png, b, &err) && b.w == 13 && b.h == 7 && b.bits == bits && b.px == a.px);
  }
  const vpimg::Image g = gradient(9, 5, 8);
  for (int ft = 0; ft <= 4; ++ft) {
    vpimg::Image b;
    CHECK(vpimg::pngDecode(filteredPng(g, ft), b) && b.px == g.px);
  }
  // A damaged file is refused.
  {
    std::vector<uint8_t> png = vpimg::pngEncode(g);
    vpimg::Image b;
    std::string err;
    png[png.size() / 2] ^= 0x55;
    CHECK(!vpimg::pngDecode(png, b, &err) || b.px != g.px);
    CHECK(!vpimg::pngDecode({1, 2, 3}, b, &err) && !err.empty());
  }
  // Formats -> RGBA.
  vpimg::Image o;
  const uint8_t rgba8[8] = {10, 20, 30, 40, 50, 60, 70, 80};
  CHECK(vpimg::convert(VK_FORMAT_R8G8B8A8_UNORM, false, rgba8, 2, 1, o) && o.bits == 8 && o.px[0] == 10 && o.px[3] == 40 && o.px[4] == 50);
  CHECK(vpimg::convert(VK_FORMAT_B8G8R8A8_UNORM, false, rgba8, 2, 1, o) && o.px[0] == 30 && o.px[1] == 20 && o.px[2] == 10 && o.px[3] == 40);
  const uint16_t half[4] = {0x3C00, 0x3800, 0x0000, 0xBC00};  // 1.0, 0.5, 0, -1 (clamped)
  CHECK(vpimg::convert(VK_FORMAT_R16G16B16A16_SFLOAT, false, reinterpret_cast<const uint8_t*>(half), 1, 1, o) && o.bits == 16);
  CHECK(o.px[0] == 65535 && o.px[1] == 32768 && o.px[2] == 0 && o.px[3] == 0);
  const uint32_t packed = 960u | (960u << 11) | (480u << 22);  // 1.0 in all three (6/6/5 bit mantissas, exponent 15)
  CHECK(vpimg::convert(VK_FORMAT_B10G11R11_UFLOAT_PACK32, false, reinterpret_cast<const uint8_t*>(&packed), 1, 1, o));
  CHECK(o.px[0] == 65535 && o.px[1] == 65535 && o.px[2] == 65535 && o.px[3] == 65535);
  const float d32[2] = {0.5f, 1.0f};
  CHECK(vpimg::convert(VK_FORMAT_D32_SFLOAT, true, reinterpret_cast<const uint8_t*>(d32), 2, 1, o) && o.px[0] == 32768 && o.px[4] == 65535);
  const uint16_t d16 = 65535;
  CHECK(vpimg::convert(VK_FORMAT_D16_UNORM, true, reinterpret_cast<const uint8_t*>(&d16), 1, 1, o) && o.px[0] == 65535);
  const uint32_t d24 = 0x00FFFFFF;
  CHECK(vpimg::convert(VK_FORMAT_D24_UNORM_S8_UINT, true, reinterpret_cast<const uint8_t*>(&d24), 1, 1, o) && o.px[0] == 65535);
  const uint8_t any[16] = {};
  CHECK(!vpimg::convert(VK_FORMAT_R32G32B32A32_SFLOAT, false, any, 1, 1, o));  // not converted: the caller writes it raw
}

namespace {

struct DumpDir {
  std::string dir;
  std::vector<std::string> entries;
  explicit DumpDir(const std::string& d) : dir(d) {
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
  }
  void png(const std::string& name, const vpimg::Image& im, const char* call, int callIndex, int pipeline) {
    const auto bytes = vpimg::pngEncode(im);
    std::ofstream(dir + "/" + name, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    entry(name, "png", call, callIndex, pipeline);
  }
  void raw(const std::string& name, const std::string& bytes, const char* call, int callIndex, int pipeline) {
    std::ofstream(dir + "/" + name, std::ios::binary) << bytes;
    entry(name, "raw", call, callIndex, pipeline);
  }
  void entry(const std::string& name, const char* kind, const char* call, int callIndex, int pipeline) {
    entries.push_back("{\"file\":\"" + name + "\",\"kind\":\"" + kind + "\",\"call\":\"" + call + "\",\"call_index\":" + std::to_string(callIndex) +
                      ",\"pipeline\":" + std::to_string(pipeline) + "}");
  }
  void index() {
    std::ofstream f(dir + "/index.json");
    f << "{\"version\":1,\"dumps\":[\n";
    for (size_t i = 0; i < entries.size(); ++i) f << entries[i] << (i + 1 < entries.size() ? ",\n" : "\n");
    f << "]}\n";
  }
};

std::string diffOutput(const std::string& a, const std::string& b, const vpimgdiff::Options& opt, int* rc, vpimgdiff::Counts* counts = nullptr) {
  std::FILE* f = std::tmpfile();
  *rc = vpimgdiff::compareDirs(a, b, opt, f, counts);
  std::rewind(f);
  std::string out;
  char buf[512];
  for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) out.append(buf, n);
  std::fclose(f);
  return out;
}

}  // namespace

void testImgDiff() {
  // Metrics.
  {
    const vpimg::Image a = gradient(4, 4, 8);
    vpimg::Image b = a;
    vpimg::DiffResult r = vpimg::diffImages(a, b);
    CHECK(!r.sizeMismatch && r.maxOf() == 0 && std::isinf(r.psnr));
    b.px[0] = uint16_t(b.px[0] + 10);  // one red sample differs by 10 of 16 pixels
    r = vpimg::diffImages(a, b);
    CHECK(r.maxDiff[0] == 10 && r.maxDiff[1] == 0 && std::fabs(r.mse[0] - 100.0 / 16) < 1e-9);
    CHECK(std::fabs(r.psnr - 10 * std::log10(255.0 * 255.0 / (100.0 / 16 / 4))) < 1e-9);
    CHECK(std::fabs(r.psnrChannel[0] - 10 * std::log10(255.0 * 255.0 / (100.0 / 16))) < 1e-9 && std::isinf(r.psnrChannel[1]));
    CHECK(vpimg::diffImages(a, gradient(4, 5, 8)).sizeMismatch && vpimg::diffImages(a, gradient(4, 4, 16)).sizeMismatch);
  }
  // Two dump directories: the first divergence, in the order of A's index.json, names its call and pipeline.
  const std::string root = (std::filesystem::temp_directory_path() / ("vpimgdiff_test_" + std::to_string(getpid()))).string();
  DumpDir a(root + "/a"), b(root + "/b");
  const vpimg::Image base = gradient(8, 8, 8);
  vpimg::Image small = base, big = base;
  small.px[5] = uint16_t(small.px[5] + 3);
  for (size_t i = 0; i < 64; ++i) big.px[i * 4 + 1] = uint16_t(255 - big.px[i * 4 + 1]);
  a.png("f0_s0_p0_d2_c0.png", base, "vkCmdEndRenderPass", 10, 5);
  a.png("f0_s0_p1_d4_c0.png", base, "vkCmdEndRenderPass", 42, 7);
  a.png("f0_s0_p2_d1_c0.png", base, "vkCmdEndRenderPass", 77, 9);
  b.png("f0_s0_p0_d2_c0.png", base, "vkCmdEndRenderPass", 10, 5);
  b.png("f0_s0_p1_d4_c0.png", small, "vkCmdEndRenderPass", 42, 7);
  b.png("f0_s0_p2_d1_c0.png", big, "vkCmdEndRenderPass", 77, 9);
  a.index(), b.index();
  int rc = 0;
  vpimgdiff::Counts cnt;
  std::string out = diffOutput(a.dir, b.dir, {}, &rc, &cnt);
  CHECK(rc == 1 && cnt.compared == 3 && cnt.divergent == 2);
  CHECK(out.find("FIRST DIVERGENCE f0_s0_p1_d4_c0.png") != std::string::npos);
  CHECK(out.find("call=vkCmdEndRenderPass call_index=42 pipeline=7") != std::string::npos);
  CHECK(out.find("max_diff rgba=3,0,0,0") != std::string::npos || out.find("max_diff rgba=0,3,0,0") != std::string::npos ||
        out.find("max_diff rgba=0,0,3,0") != std::string::npos || out.find("max_diff rgba=0,0,0,3") != std::string::npos);
  CHECK(out.find("p2_d1") == std::string::npos);  // only the first one is listed
  // A threshold that forgives the small difference moves the first divergence on; --all lists both.
  vpimgdiff::Options opt;
  opt.maxDiff = 3;
  out = diffOutput(a.dir, b.dir, opt, &rc);
  CHECK(rc == 1 && out.find("FIRST DIVERGENCE f0_s0_p2_d1_c0.png") != std::string::npos && out.find("pipeline=9") != std::string::npos);
  opt = {};
  opt.all = true;
  out = diffOutput(a.dir, b.dir, opt, &rc);
  CHECK(out.find("f0_s0_p1_d4_c0.png") != std::string::npos && out.find("f0_s0_p2_d1_c0.png") != std::string::npos);
  opt = {};
  opt.minPsnr = 60;  // the +3 difference is about 70 dB: passes; the inverted green channel does not
  out = diffOutput(a.dir, b.dir, opt, &rc);
  CHECK(rc == 1);
  opt.maxDiff = 255;
  out = diffOutput(a.dir, b.dir, opt, &rc, &cnt);
  CHECK(rc == 1 && cnt.divergent == 1 && out.find("FIRST DIVERGENCE f0_s0_p2_d1_c0.png") != std::string::npos);
  // Identical directories, a dump missing in B, an extra one, raw dumps.
  DumpDir c(root + "/c"), d(root + "/d");
  c.png("x.png", base, "vkCmdDraw", 1, 2), d.png("x.png", base, "vkCmdDraw", 1, 2);
  c.raw("y.raw", "abcd", "vkCmdDraw", 3, 2), d.raw("y.raw", "abcd", "vkCmdDraw", 3, 2);
  c.index(), d.index();
  out = diffOutput(c.dir, d.dir, {}, &rc, &cnt);
  CHECK(rc == 0 && cnt.compared == 2 && cnt.divergent == 0);
  d.raw("y.raw", "abce", "vkCmdDraw", 3, 2);
  d.entries.erase(d.entries.begin() + 1);
  d.index();
  out = diffOutput(c.dir, d.dir, {}, &rc, &cnt);
  CHECK(rc == 1 && out.find("raw bytes differ") != std::string::npos && cnt.divergent == 1);
  d.entries.pop_back();
  d.index();
  out = diffOutput(c.dir, d.dir, {}, &rc, &cnt);
  CHECK(rc == 1 && cnt.missingInB == 1 && out.find("missing in B") != std::string::npos);
  d.png("z.png", base, "vkCmdDraw", 9, 9);
  d.index();
  out = diffOutput(d.dir, c.dir, {}, &rc, &cnt);
  CHECK(rc == 1 && cnt.extraInB >= 1);
  out = diffOutput(root + "/nope", c.dir, {}, &rc);
  CHECK(rc == 2);
  // Nothing in common proves nothing: two empty replays, or two with different names, fail.
  DumpDir e(root + "/e"), f(root + "/f");
  e.index(), f.index();
  out = diffOutput(e.dir, f.dir, {}, &rc, &cnt);
  CHECK(rc == 1 && cnt.compared == 0 && out.find("NOTHING COMPARED") != std::string::npos);
  e.png("only_a.png", base, "vkCmdDraw", 1, 2), f.png("only_b.png", base, "vkCmdDraw", 1, 2);
  e.index(), f.index();
  out = diffOutput(e.dir, f.dir, {}, &rc, &cnt);
  CHECK(rc == 1 && cnt.compared == 0 && cnt.missingInB == 1 && cnt.extraInB == 1 && out.find("NOTHING COMPARED") != std::string::npos);
  // A name in an index.json that leaves the directory is refused.
  for (const char* bad : {"../x.png", "sub/x.png", ".."}) {
    DumpDir g(root + "/g");
    g.entry(bad, "png", "vkCmdDraw", 1, 2);
    g.index();
    out = diffOutput(g.dir, c.dir, {}, &rc);
    CHECK(rc == 2 && out.find("not a plain file name") != std::string::npos);
    out = diffOutput(c.dir, g.dir, {}, &rc);
    CHECK(rc == 2);
  }
  std::filesystem::remove_all(root);
}

namespace {

struct CapSummary {
  std::vector<uint16_t> frameIds;  // FRAME calls without the data records, the frame ends and the idle waits (the replay waits after every submit)
  std::map<std::string, int> state;
  uint64_t memBytes = 0;
  int frames = 0;
};
CapSummary summarize(const std::string& path) {
  CapSummary s;
  vpcap::Reader rd;
  CHECK(rd.open(path));
  vpcap::Call c;
  while (rd.next(c)) {
    vpcap::MemoryData md;
    if (vpcap::decodeMemoryData(c, md)) s.memBytes += md.size;
    if (c.id == vpcap::id::vpcapFrameEnd) ++s.frames;
    if (c.section == vpcap::Section::State) ++s.state[vpcap::callName(c.id)];
    else if (c.id != vpcap::id::vpcapFrameEnd && c.id != vpcap::id::vpcapMemoryData && c.id != vpcap::id::vpcapImageData &&
             c.id != vpcap::id::vkQueueWaitIdle && c.id != vpcap::id::vkDeviceWaitIdle)
      s.frameIds.push_back(c.id);
  }
  CHECK(rd.error().empty());
  return s;
}

}  // namespace

namespace {

// One triangle of solid colour (0.25, 0.25, 0.25) on a (0, 0.5, 0, 0) clear, in both windings so exactly one is not culled.
void recordTriangle(CapRig& r, VkCommandBuffer c) {
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  r.v.vkBeginCommandBuffer(c, &bi);
  VkClearValue clear{};
  clear.color.float32[1] = 0.5f;
  VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  rbi.renderPass = r.rp, rbi.framebuffer = r.fb, rbi.renderArea = {{0, 0}, {64, 64}}, rbi.clearValueCount = 1, rbi.pClearValues = &clear;
  r.v.vkCmdBeginRenderPass(c, &rbi, VK_SUBPASS_CONTENTS_INLINE);
  r.v.vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, r.pipe);
  r.v.vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, r.pl, 0, 1, &r.set, 0, nullptr);
  VkDeviceSize off = 0;
  r.v.vkCmdBindVertexBuffers(c, 0, 1, &r.vbib, &off);
  const VkViewport vp{0, 0, 64, 64, 0, 1};
  const VkRect2D sc{{0, 0}, {64, 64}};
  r.v.vkCmdSetViewport(c, 0, 1, &vp);
  r.v.vkCmdSetScissor(c, 0, 1, &sc);
  r.v.vkCmdDraw(c, 6, 1, 0, 0);
  r.v.vkCmdEndRenderPass(c);
  r.v.vkEndCommandBuffer(c);
}

// Captures the triangle frame into `cap`.
void captureTriangle(const std::string& cap) {
  vpvk_capture_track(0);
  CHECK(vpvk_capture_track(1) == 0);
  {
    CapRig r;
    const float tri[6][4] = {{0, 0.5f, 0, 1}, {0.5f, -0.5f, 0, 1}, {-0.5f, -0.5f, 0, 1}, {-0.5f, -0.5f, 0, 1}, {0.5f, -0.5f, 0, 1}, {0, 0.5f, 0, 1}};
    std::memcpy(r.map, tri, sizeof tri);
    recordTriangle(r, r.cb[0]);
    CHECK(vpvk_capture_request(1, cap.c_str()) == 1);
    r.submit({r.cb[0]});  // the boundary
    recordTriangle(r, r.cb[1]);
    r.submit({r.cb[1]});  // frame 0
    CHECK(vpvk_capture_state() == 0);
  }
  vpvk_capture_track(0);
}

// Captures the triangle frame, replays it through the Vulkan loader with the ICD in `icdJson`, and (checkPixels) looks at the
// pixels: the clear colour in a corner, the triangle's colour inside it.
void triangleThroughLoader(const std::filesystem::path& tmp, const std::string& icdJson, bool checkPixels) {
  const std::string cap = (tmp / "tri.vpcap").string();
  captureTriangle(cap);
  for (const char* at : {"pass", "draw"}) {
    const std::string dir = (tmp / (std::string("tri_") + at)).string();
    const std::string cmd = "\"" + std::string(gReplay) + "\" \"" + cap + "\" --icd system --icd-file \"" + icdJson + "\" --dump \"" + dir +
                            "\" --dump-at " + at + " --strict > \"" + (tmp / "tri.log").string() + "\" 2>&1";
    CHECK(std::system(cmd.c_str()) == 0);
    const std::string file = dir + (std::string(at) == "pass" ? "/f0_s0_p0_d1_c0.png" : "/f0_s0_p0_d0_c0.png");
    std::vector<uint8_t> bytes;
    vpimg::Image im;
    CHECK(vpimgdiff::readFile(file, bytes) && vpimg::pngDecode(bytes, im) && im.w == 64 && im.h == 64);
    if (!checkPixels || im.px.size() != 64 * 64 * 4) continue;
    auto near = [&](int x, int y, int r, int g, int b, int a) {
      const uint16_t* p = &im.px[(size_t(y) * 64 + x) * 4];
      return std::abs(p[0] - r) <= 1 && std::abs(p[1] - g) <= 1 && std::abs(p[2] - b) <= 1 && std::abs(p[3] - a) <= 1;
    };
    CHECK(near(2, 2, 0, 128, 0, 0));       // outside the triangle: the clear colour
    CHECK(near(61, 61, 0, 128, 0, 0));
    CHECK(near(32, 24, 64, 64, 64, 255));  // inside: 0.25 grey, opaque
    CHECK(near(32, 20, 64, 64, 64, 255));
    CHECK(near(4, 40, 0, 128, 0, 0));      // left of the bottom apex
  }
}

// Captures one frame of a small scene: a textured quad (8x8 texture sampled with nearest filtering, no depth bias), a triangle
// behind it that the depth test cuts, and an alpha-blended triangle in front, all in one 4x MSAA colour + depth pass that
// resolves into a 64x64 RGBA8 image (the only thing a replay dumps). tex_tint.frag samples at gl_FragCoord / 64 and multiplies
// by a push-constant tint, so the same pixels come out of any driver that follows the spec.
void captureScene(const std::string& cap) {
  vpvk_capture_track(0);
  CHECK(vpvk_capture_track(1) == 0);
  {
    CapRig r;
    CapApi& v = r.v;
    auto& nb = static_cast<vpgpu::NullBackend&>(from<Device>(r.dev)->platform->backend());
    auto image = [&](VkFormat fmt, uint32_t size, VkSampleCountFlagBits samples, VkImageUsageFlags usage, VkDeviceSize off, VkImageAspectFlags aspect,
                     VkImageView* view) {
      VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      ii.imageType = VK_IMAGE_TYPE_2D, ii.format = fmt, ii.extent = {size, size, 1};
      ii.mipLevels = 1, ii.arrayLayers = 1, ii.samples = samples, ii.tiling = VK_IMAGE_TILING_OPTIMAL, ii.usage = usage;
      VkImage im = nullptr;
      CHECK(v.vkCreateImage(r.dev, &ii, nullptr, &im) == VK_SUCCESS);
      CHECK(v.vkBindImageMemory(r.dev, im, r.devMem, off) == VK_SUCCESS);
      VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      vi.image = im, vi.viewType = VK_IMAGE_VIEW_TYPE_2D, vi.format = fmt, vi.subresourceRange = {aspect, 0, 1, 0, 1};
      CHECK(v.vkCreateImageView(r.dev, &vi, nullptr, view) == VK_SUCCESS);
      return im;
    };
    const VkImageUsageFlags kXfer = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkImageView texView = nullptr, colView = nullptr, depView = nullptr, resView = nullptr;
    const VkImage tex = image(VK_FORMAT_R8G8B8A8_UNORM, 8, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_SAMPLED_BIT | kXfer, 65536, VK_IMAGE_ASPECT_COLOR_BIT, &texView);
    image(VK_FORMAT_R8G8B8A8_UNORM, 64, VK_SAMPLE_COUNT_4_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 1 << 20, VK_IMAGE_ASPECT_COLOR_BIT, &colView);
    image(VK_FORMAT_D32_SFLOAT, 64, VK_SAMPLE_COUNT_4_BIT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, 2 << 20, VK_IMAGE_ASPECT_DEPTH_BIT, &depView);
    image(VK_FORMAT_R8G8B8A8_UNORM, 64, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 3 << 20,
          VK_IMAGE_ASPECT_COLOR_BIT, &resView);

    // The texture is written by the GPU before the capture (a full clear settles its zero-init); the capture then reads its
    // contents from the first frame that binds it.
    VkCommandBuffer up = r.cb[0];
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    v.vkBeginCommandBuffer(up, &bi);
    const VkClearColorValue zero{};
    const VkImageSubresourceRange all{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    v.vkCmdClearColorImage(up, tex, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &all);
    v.vkEndCommandBuffer(up);
    r.submit({up});
    std::vector<uint8_t> texels(8 * 8 * 4);
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) {
        uint8_t* p = &texels[size_t(y * 8 + x) * 4];
        p[0] = uint8_t(x * 36), p[1] = uint8_t(y * 36), p[2] = ((x ^ y) & 1) ? 220 : 40, p[3] = x < 4 ? 255 : 170;
      }
    nb.textureData[vpgpu::NullBackend::dataKey(from<Image>(tex)->id, 0, 0)] = texels;

    // Clip-space positions (x, y, z, 1): the quad at z 0.5, the triangle behind it at z 0.75, the blended one in front at z 0.25.
    const float verts[12][4] = {{-0.8f, -0.8f, 0.5f, 1}, {0.4f, -0.8f, 0.5f, 1}, {0.4f, 0.8f, 0.5f, 1},  {-0.8f, -0.8f, 0.5f, 1},
                                {0.4f, 0.8f, 0.5f, 1},   {-0.8f, 0.8f, 0.5f, 1}, {-0.3f, -0.9f, 0.75f, 1}, {0.9f, -0.9f, 0.75f, 1},
                                {0.9f, 0.5f, 0.75f, 1},  {-0.5f, 0.6f, 0.25f, 1}, {0.7f, 0.2f, 0.25f, 1},  {-0.1f, -0.7f, 0.25f, 1}};
    std::memcpy(r.map, verts, sizeof verts);

    VkAttachmentDescription att[3] = {};
    att[0].format = VK_FORMAT_R8G8B8A8_UNORM, att[0].samples = VK_SAMPLE_COUNT_4_BIT;
    att[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, att[0].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att[1] = att[0];
    att[1].format = VK_FORMAT_D32_SFLOAT, att[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    att[2] = att[0];
    att[2].samples = VK_SAMPLE_COUNT_1_BIT, att[2].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, att[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    const VkAttachmentReference cref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}, dref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL},
        rref{2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sp{};
    sp.colorAttachmentCount = 1, sp.pColorAttachments = &cref, sp.pResolveAttachments = &rref, sp.pDepthStencilAttachment = &dref;
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = 3, rpi.pAttachments = att, rpi.subpassCount = 1, rpi.pSubpasses = &sp;
    VkRenderPass rp = nullptr;
    CHECK(v.vkCreateRenderPass(r.dev, &rpi, nullptr, &rp) == VK_SUCCESS);
    const VkImageView views[3] = {colView, depView, resView};
    VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fi.renderPass = rp, fi.attachmentCount = 3, fi.pAttachments = views, fi.width = 64, fi.height = 64, fi.layers = 1;
    VkFramebuffer fb = nullptr;
    CHECK(v.vkCreateFramebuffer(r.dev, &fi, nullptr, &fb) == VK_SUCCESS);

    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_NEAREST, sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VkSampler smp = nullptr;
    CHECK(v.vkCreateSampler(r.dev, &sci, nullptr, &smp) == VK_SUCCESS);
    VkDescriptorSetLayoutBinding b0{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = 1, lci.pBindings = &b0;
    VkDescriptorSetLayout dsl = nullptr;
    CHECK(v.vkCreateDescriptorSetLayout(r.dev, &lci, nullptr, &dsl) == VK_SUCCESS);
    const VkPushConstantRange pcr{VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1, plci.pSetLayouts = &dsl, plci.pushConstantRangeCount = 1, plci.pPushConstantRanges = &pcr;
    VkPipelineLayout pl = nullptr;
    CHECK(v.vkCreatePipelineLayout(r.dev, &plci, nullptr, &pl) == VK_SUCCESS);
    const VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1, dpi.poolSizeCount = 1, dpi.pPoolSizes = &ps;
    VkDescriptorPool pool = nullptr;
    CHECK(v.vkCreateDescriptorPool(r.dev, &dpi, nullptr, &pool) == VK_SUCCESS);
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = pool, dai.descriptorSetCount = 1, dai.pSetLayouts = &dsl;
    VkDescriptorSet set = nullptr;
    CHECK(v.vkAllocateDescriptorSets(r.dev, &dai, &set) == VK_SUCCESS);
    const VkDescriptorImageInfo dii{smp, texView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = set, w.descriptorCount = 1, w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, w.pImageInfo = &dii;
    v.vkUpdateDescriptorSets(r.dev, 1, &w, 0, nullptr);

    auto module = [&](const char* name) {
      const std::vector<uint32_t> words = spirvFile(name);
      VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      ci.codeSize = words.size() * 4, ci.pCode = words.data();
      VkShaderModule m = nullptr;
      CHECK(v.vkCreateShaderModule(r.dev, &ci, nullptr, &m) == VK_SUCCESS);
      return m;
    };
    const VkShaderModule vs = module("pos.vert.spv"), fs = module("tex_tint.frag.spv");
    VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT, st[0].module = vs, st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT, st[1].module = fs, st[1].pName = "main";
    const VkVertexInputBindingDescription vbd{0, 16, VK_VERTEX_INPUT_RATE_VERTEX};
    const VkVertexInputAttributeDescription vad{0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0};
    VkPipelineVertexInputStateCreateInfo vis{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vis.vertexBindingDescriptionCount = 1, vis.pVertexBindingDescriptions = &vbd;
    vis.vertexAttributeDescriptionCount = 1, vis.pVertexAttributeDescriptions = &vad;
    VkPipelineInputAssemblyStateCreateInfo ias{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ias.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1, vps.scissorCount = 1;  // dynamic
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.cullMode = VK_CULL_MODE_NONE, rs.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_4_BIT;
    VkPipelineDepthStencilStateCreateInfo ds[2] = {{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO}, {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO}};
    for (auto& d : ds) d.depthTestEnable = VK_TRUE, d.depthCompareOp = VK_COMPARE_OP_LESS;
    ds[0].depthWriteEnable = VK_TRUE;  // [0] opaque, [1] blended: tests the depth but does not write it
    VkPipelineColorBlendAttachmentState cba[2] = {};
    cba[0].colorWriteMask = cba[1].colorWriteMask = 0xF;
    cba[1].blendEnable = VK_TRUE;
    cba[1].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA, cba[1].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba[1].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE, cba[1].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    VkPipelineColorBlendStateCreateInfo cbs[2] = {{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}, {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}};
    for (int k = 0; k < 2; ++k) cbs[k].attachmentCount = 1, cbs[k].pAttachments = &cba[k];
    const VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dys{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dys.dynamicStateCount = 2, dys.pDynamicStates = dyn;
    VkGraphicsPipelineCreateInfo gp[2] = {{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO}, {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO}};
    for (int k = 0; k < 2; ++k) {
      gp[k].stageCount = 2, gp[k].pStages = st, gp[k].pVertexInputState = &vis, gp[k].pInputAssemblyState = &ias, gp[k].pViewportState = &vps;
      gp[k].pRasterizationState = &rs, gp[k].pMultisampleState = &ms, gp[k].pDepthStencilState = &ds[k], gp[k].pColorBlendState = &cbs[k];
      gp[k].pDynamicState = &dys, gp[k].layout = pl, gp[k].renderPass = rp;
    }
    VkPipeline pipe[2] = {};
    CHECK(v.vkCreateGraphicsPipelines(r.dev, VK_NULL_HANDLE, 2, gp, nullptr, pipe) == VK_SUCCESS);
    v.vkDestroyShaderModule(r.dev, vs, nullptr);
    v.vkDestroyShaderModule(r.dev, fs, nullptr);

    auto record = [&](VkCommandBuffer c) {
      v.vkBeginCommandBuffer(c, &bi);
      VkClearValue clear[3] = {};
      clear[0].color.float32[0] = 0.1f, clear[0].color.float32[1] = 0.2f, clear[0].color.float32[2] = 0.4f, clear[0].color.float32[3] = 1;
      clear[1].depthStencil.depth = 1;
      VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      rbi.renderPass = rp, rbi.framebuffer = fb, rbi.renderArea = {{0, 0}, {64, 64}}, rbi.clearValueCount = 3, rbi.pClearValues = clear;
      v.vkCmdBeginRenderPass(c, &rbi, VK_SUBPASS_CONTENTS_INLINE);
      v.vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &set, 0, nullptr);
      const VkDeviceSize off = 0;
      v.vkCmdBindVertexBuffers(c, 0, 1, &r.vbib, &off);
      const VkViewport vp{0, 0, 64, 64, 0, 1};
      const VkRect2D sc{{0, 0}, {64, 64}};
      v.vkCmdSetViewport(c, 0, 1, &vp);
      v.vkCmdSetScissor(c, 0, 1, &sc);
      const struct { int pipeline; uint32_t count, first; float tint[4]; } draws[3] = {
          {0, 6, 0, {1, 1, 1, 1}}, {0, 3, 6, {0.2f, 0.9f, 0.4f, 1}}, {1, 3, 9, {1, 0.5f, 0.25f, 0.6f}}};
      for (const auto& d : draws) {
        v.vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe[d.pipeline]);
        v.vkCmdPushConstants(c, pl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, d.tint);
        v.vkCmdDraw(c, d.count, 1, d.first, 0);
      }
      v.vkCmdEndRenderPass(c);
      v.vkEndCommandBuffer(c);
    };
    CHECK(vpvk_capture_request(1, cap.c_str()) == 1);
    r.submit({up});  // the boundary
    record(r.cb[1]);
    r.submit({r.cb[1]});  // frame 0
    CHECK(vpvk_capture_state() == 0);
  }
  vpvk_capture_track(0);
}

}  // namespace

// A capture of two frames made by the driver, replayed by vpreplay in libVPVulkan (the null backend executes nothing, so
// what is compared is the call stream: the replay is captured again and must hold the same state and the same frame calls),
// then the dumps: one file per render target per pass, an index.json that names call and pipeline, vpimgdiff over two runs.
void testReplay() {
  if (!gReplay) {
    std::puts("  skipped: run by ctest gpu_vulkan.replay (needs the vpreplay path)");
    return;
  }
  const auto tmp = std::filesystem::temp_directory_path() / ("vpreplay_test_" + std::to_string(getpid()));
  std::filesystem::remove_all(tmp);
  std::filesystem::create_directories(tmp);
  const std::string capA = (tmp / "a.vpcap").string(), capB = (tmp / "b.vpcap").string();
  vpvk_capture_track(0);
  CHECK(vpvk_capture_track(1) == 0);
  {
    CapRig r;
    r.fillBuffers(1);
    r.record(r.cb[0], true);
    r.submit({r.cb[0]});
    CHECK(vpvk_capture_request(2, capA.c_str()) == 1);
    r.fillBuffers(5);
    r.submit({r.cb[0]});  // the boundary
    r.record(r.cb[1], true);
    r.submit({r.cb[0], r.cb[1]});  // frame 0
    r.map[4096 + 17] ^= 0xFF;
    r.record(r.cb[2], false);
    r.submit({r.cb[2]});  // frame 1
    CHECK(vpvk_capture_state() == 0);
  }
  vpvk_capture_track(0);
  const std::string base = std::string("\"") + gReplay + "\" \"" + capA + "\" --icd vpvk --lib \"" + gLib + "\" --strict";
  auto sh = [&](const std::string& args, const std::string& log) {
    return std::system((base + " " + args + " > \"" + (tmp / log).string() + "\" 2>&1").c_str());
  };
  auto slurp = [&](const std::string& f) {
    std::ifstream in(tmp / f);
    return std::string(std::istreambuf_iterator<char>(in), {});
  };

  // 1. The replay, captured again.
  CHECK(sh("--recapture \"" + capB + "\"", "rec.log") == 0);
  CHECK(slurp("rec.log").find("errors=0") != std::string::npos);
  CHECK(slurp("rec.log").find("vpreplay: capture frame_unit=submit frames=2 ") != std::string::npos);  // the tool says what a frame is
  const CapSummary a = summarize(capA), b = summarize(capB);
  CHECK(a.frames == 2 && b.frames == 2);
  CHECK(a.frameIds == b.frameIds);
  if (a.frameIds != b.frameIds) {
    for (size_t i = 0; i < std::max(a.frameIds.size(), b.frameIds.size()); ++i)
      std::printf("    %3zu  %-28s %-28s\n", i, i < a.frameIds.size() ? vpcap::callName(a.frameIds[i]) : "-", i < b.frameIds.size() ? vpcap::callName(b.frameIds[i]) : "-");
  }
  CHECK(a.memBytes == b.memBytes && a.memBytes > 0);
  auto state = b.state;
  --state["vkCreateCommandPool"], --state["vkAllocateCommandBuffers"];  // the empty submit the capture of the replay starts at
  for (auto it = state.begin(); it != state.end();) it = it->second == 0 ? state.erase(it) : std::next(it);
  CHECK(a.state == state);
  CHECK(a.state.count("vkCreateGraphicsPipelines") && a.state.at("vkCreateGraphicsPipelines") == 1);

  // 2. Dumps at pass level: frame 0 ran one pass (cb[1]; cb[0] was recorded before the capture), frame 1 one.
  CHECK(sh("--dump \"" + (tmp / "p1").string() + "\" --dump-at pass", "p1.log") == 0);
  CHECK(sh("--dump \"" + (tmp / "p2").string() + "\" --dump-at pass", "p2.log") == 0);
  vpjson::Value idx;
  {
    const std::string text = slurp("p1/index.json");
    vpjson::Parser ps(text);
    CHECK(ps.parse(idx) && idx.get("dumps") && idx.get("dumps")->arr.size() == 2);
  }
  if (idx.get("dumps") && idx.get("dumps")->arr.size() == 2) {
    const vpjson::Value& d0 = idx.get("dumps")->arr[0];
    const vpjson::Value& d1 = idx.get("dumps")->arr[1];
    CHECK(d0.text("file") == "f0_s0_p0_d1_c0.png" && d1.text("file") == "f1_s0_p0_d1_c0.png");
    CHECK(d0.text("call") == "vkCmdEndRenderPass" && d0.number("pipeline") > 0 && d0.number("call_index") > 0);
    CHECK(d0.number("width") == 64 && d0.number("height") == 64 && d0.text("format_name") == "R8G8B8A8_UNORM");
    CHECK(d0.get("descriptor_sets") && d0.get("descriptor_sets")->arr[0].num > 0 && d0.get("vertex_buffers")->arr[0].num > 0);
    CHECK(d0.number("index_buffer") > 0 && d0.number("render_pass") > 0 && d0.number("image") > 0);
    for (const auto& d : idx.get("dumps")->arr) {
      std::vector<uint8_t> f;
      vpimg::Image im;
      CHECK(vpimgdiff::readFile((tmp / "p1" / d.text("file")).string(), f) && vpimg::pngDecode(f, im) && im.w == 64 && im.h == 64);
    }
  }
  int rc = 0;
  vpimgdiff::Counts cnt;
  diffOutput((tmp / "p1").string(), (tmp / "p2").string(), {}, &rc, &cnt);
  CHECK(rc == 0 && cnt.compared == 2);
  {  // a pixel changed in the second run: the first divergence is that dump, with the pipeline of its draw
    std::vector<uint8_t> f;
    vpimg::Image im;
    const std::string victim = (tmp / "p2" / "f1_s0_p0_d1_c0.png").string();
    CHECK(vpimgdiff::readFile(victim, f) && vpimg::pngDecode(f, im));
    im.px[0] = 200;
    const auto bytes = vpimg::pngEncode(im);
    std::ofstream(victim, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    const std::string out = diffOutput((tmp / "p1").string(), (tmp / "p2").string(), {}, &rc);
    CHECK(rc == 1 && out.find("FIRST DIVERGENCE f1_s0_p0_d1_c0.png") != std::string::npos && out.find("call=vkCmdEndRenderPass") != std::string::npos);
  }

  // 3. Dumps at draw level: the pass is cut after each draw and resumed (frame 0 draws once with 3 instances, frame 1 once).
  CHECK(sh("--dump \"" + (tmp / "d1").string() + "\" --dump-at draw", "d1.log") == 0);
  {
    const std::string text = slurp("d1/index.json");
    vpjson::Value di;
    vpjson::Parser ps(text);
    CHECK(ps.parse(di) && di.get("dumps") && di.get("dumps")->arr.size() == 2);
    if (di.get("dumps") && di.get("dumps")->arr.size() == 2) {
      const vpjson::Value& d0 = di.get("dumps")->arr[0];
      CHECK(d0.text("file") == "f0_s0_p0_d0_c0.png" && (d0.text("call") == "vkCmdDrawIndexed") && d0.number("pipeline") > 0);
      CHECK(di.get("dumps")->arr[1].text("call") == "vkCmdDraw");
    }
  }
  // 4. The same through the Vulkan loader, with libVPVulkan as the ICD (it exports the loader interface): the path a
  // lavapipe run takes, minus the pixels.
  if (void* loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL)) {
    dlclose(loader);
    const std::string icd = (tmp / "vpvk_icd.json").string();
    std::ofstream(icd) << "{\"file_format_version\":\"1.0.0\",\"ICD\":{\"library_path\":\"" << std::filesystem::absolute(gLib).string()
                       << "\",\"api_version\":\"1.3.0\"}}\n";
    triangleThroughLoader(tmp, icd, false);
    // The loader run and the direct run dump the same files for the same capture.
    CHECK(sh("--dump \"" + (tmp / "v1").string() + "\" --dump-at pass", "v1.log") == 0);
    diffOutput((tmp / "p1").string(), (tmp / "v1").string(), {}, &rc, &cnt);
    CHECK(rc == 0 && cnt.compared == 2);
  } else {
    std::puts("  loader test skipped: libvulkan.so.1 not found");
  }
  // 5. A capture with image contents (vpcapImageData: 2D / array / mip / big images in two records) and several kinds of pass
  // (LOAD, dynamic rendering, a transient attachment): every record is uploaded, nothing fails; --raw writes bit-exact files.
  {
    const std::string img = (tmp / "img.vpcap").string();
    gKeepImages = img;
    testCaptureImages();
    gKeepImages.clear();
    CHECK(std::filesystem::exists(img));
    const std::string cmd = "\"" + std::string(gReplay) + "\" \"" + img + "\" --icd vpvk --lib \"" + gLib + "\" --strict --raw --dump \"" +
                            (tmp / "i1").string() + "\" > \"" + (tmp / "img.log").string() + "\" 2>&1";
    CHECK(std::system(cmd.c_str()) == 0);
    const std::string log = slurp("img.log");
    CHECK(log.find("image uploads=9 failed=0") != std::string::npos);  // 9 subresources (the 16 MiB one in two records)
    CHECK(log.find("errors=0") != std::string::npos && log.find("png=0 raw=3") != std::string::npos);  // LOAD pass, r.record pass, dynamic pass
    const std::string cmd2 = "\"" + std::string(gReplay) + "\" \"" + img + "\" --icd vpvk --lib \"" + gLib + "\" --raw --dump \"" + (tmp / "i2").string() +
                             "\" > /dev/null 2>&1";
    CHECK(std::system(cmd2.c_str()) == 0);
    vpimgdiff::Counts c2;
    const std::string out = diffOutput((tmp / "i1").string(), (tmp / "i2").string(), {}, &rc, &c2);
    CHECK(rc == 0 && c2.compared > 0);
    bool sawJson = false;
    for (auto& e : std::filesystem::directory_iterator(tmp / "i1")) sawJson = sawJson || (e.path().extension() == ".json" && e.path().filename() != "index.json");
    CHECK(sawJson);
  }
  // 6. A capture that began after the instance existed (META instance_in_state=0): the replay makes an instance of its own.
  {
    const std::string cap = (tmp / "late.vpcap").string();
    vpvk_capture_track(0);
    CapApi v;
    v.vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(vkGetInstanceProcAddr(nullptr, "vkCreateInstance"));
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    VkInstance inst = nullptr;
    CHECK(v.vkCreateInstance(&ici, nullptr, &inst) == VK_SUCCESS);
    v.load(inst);
    CHECK(vpvk_capture_track(1) == 0);
    VkPhysicalDevice pd = nullptr;
    uint32_t n = 1;
    vkEnumeratePhysicalDevices(inst, &n, &pd);
    float prio = 1;
    VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    q.queueCount = 1, q.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1, dci.pQueueCreateInfos = &q;
    VkDevice dev = nullptr;
    CHECK(v.vkCreateDevice(pd, &dci, nullptr, &dev) == VK_SUCCESS);
    VkQueue queue = nullptr;
    v.vkGetDeviceQueue(dev, 0, 0, &queue);
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    VkCommandPool pool = nullptr;
    v.vkCreateCommandPool(dev, &cpi, nullptr, &pool);
    VkCommandBuffer cb[2] = {};
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool, cai.commandBufferCount = 2;
    v.vkAllocateCommandBuffers(dev, &cai, cb);
    auto empty = [&](VkCommandBuffer c) {
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      v.vkBeginCommandBuffer(c, &bi);
      v.vkEndCommandBuffer(c);
      VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      si.commandBufferCount = 1, si.pCommandBuffers = &c;
      CHECK(v.vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS);
      v.vkQueueWaitIdle(queue);
    };
    empty(cb[0]);
    CHECK(vpvk_capture_request(1, cap.c_str()) == 1);
    empty(cb[0]);  // the boundary
    empty(cb[1]);  // frame 0 (cb[0] was begun before the capture: it is not in the file)
    CHECK(vpvk_capture_state() == 0);
    v.vkDestroyDevice(dev, nullptr);
    v.vkDestroyInstance(inst, nullptr);
    vpvk_capture_track(0);
    vpcap::Reader rd;
    CHECK(rd.open(cap));
    vpcap::Call c;
    while (rd.next(c)) {
    }
    CHECK(rd.meta().find("instance_in_state=0") != std::string::npos);
    const std::string cmd = "\"" + std::string(gReplay) + "\" \"" + cap + "\" --icd vpvk --lib \"" + gLib + "\" --strict > \"" + (tmp / "late2.log").string() + "\" 2>&1";
    CHECK(std::system(cmd.c_str()) == 0);
    CHECK(slurp("late2.log").find("errors=0") != std::string::npos);
  }
  // 8. What the host adds to the driver's picture (VPKTAA, RCAS, foveation, forced 1x / MSAA / anisotropy) is not in the capture:
  // the replay turns it off whatever the environment says.
  {
    const std::string cmd = "VPVK_TAA_NATIVE=1 VPVK_RCAS=0.8 VPVK_FOVEATION=1 VPVK_FORCE_1X=1 VPVK_FORCE_ANISO=4 VPVK_MSAA=4 VPVK_TAA_JITTER=1 " +
                            base + " > \"" + (tmp / "cfg.log").string() + "\" 2>&1";
    CHECK(std::system(cmd.c_str()) == 0);
    const std::string log = slurp("cfg.log");
    CHECK(log.find("vk.cfg effective") != std::string::npos);
    CHECK(log.find("taa_native=0") != std::string::npos && log.find("rcas=0.00") != std::string::npos && log.find("force_1x=0") != std::string::npos);
    CHECK(log.find("anisotropy=0") != std::string::npos && log.find("msaa_override=0") != std::string::npos && log.find("fov_apply=0") != std::string::npos);
  }
  // 9. Damaged captures: a cut one is an error unless --allow-truncated; an image upload that does not fit its image is
  // skipped and counted; a command buffer that lost a command is not submitted.
  {
    struct Rec {
      uint16_t id;
      uint32_t thread;
      vpcap::Section section;
      std::vector<uint8_t> data;
    };
    auto load = [&](const std::string& path, std::string& meta) {
      std::vector<Rec> all;
      vpcap::Reader rd;
      CHECK(rd.open(path));
      vpcap::Call c;
      while (rd.next(c)) all.push_back({c.id, c.thread, c.section, std::vector<uint8_t>(c.data, c.data + c.size)});
      meta = rd.meta();
      return all;
    };
    auto store = [&](const std::string& path, const std::string& meta, const std::vector<Rec>& all, bool close) {
      vpcap::Writer w;
      CHECK(w.open(path));
      w.meta(meta);
      for (const Rec& x : all) {
        std::vector<uint8_t> rec(sizeof(vpcap::CallHeader) + x.data.size());
        const vpcap::CallHeader h{x.id, 0, x.thread, uint32_t(x.data.size())};
        std::memcpy(rec.data(), &h, sizeof h);
        if (!x.data.empty()) std::memcpy(rec.data() + sizeof h, x.data.data(), x.data.size());
        w.begin(x.section);
        w.append(rec.data(), rec.size());
      }
      if (close) CHECK(w.close("end=test\n"));
      else w.sync();  // no END block: what a capture cut short leaves
    };
    auto replayRc = [&](const std::string& cap, const std::string& args, const std::string& log) {
      return std::system(("\"" + std::string(gReplay) + "\" \"" + cap + "\" --icd vpvk --lib \"" + gLib + "\" " + args + " > \"" + (tmp / log).string() +
                          "\" 2>&1").c_str());
    };
    std::string meta;
    const std::vector<Rec> good = load(capA, meta);
    const std::string cut = (tmp / "cut.vpcap").string();
    store(cut, meta, good, false);
    CHECK(replayRc(cut, "", "cut1.log") != 0 && slurp("cut1.log").find("capture is incomplete") != std::string::npos);
    CHECK(replayRc(cut, "--allow-truncated --strict", "cut2.log") == 0);
    const std::string whole = (tmp / "whole.vpcap").string();
    store(whole, meta, good, true);
    CHECK(replayRc(whole, "--strict", "whole.log") == 0);

    // an image upload whose size is not the image's: the image file of section 5, the first image record cut in half
    std::string imgMeta;
    std::vector<Rec> img = load((tmp / "img.vpcap").string(), imgMeta);
    for (Rec& x : img)
      if (x.id == vpcap::id::vpcapImageData) {
        uint64_t sz = 0;
        std::memcpy(&sz, x.data.data() + 56, 8);
        const uint64_t half = sz / 2;  // total = size = half: consistent in itself, but not the image's size
        std::memcpy(x.data.data() + 40, &half, 8), std::memcpy(x.data.data() + 48, std::vector<uint8_t>(8, 0).data(), 8), std::memcpy(x.data.data() + 56, &half, 8);
        break;
      }
    const std::string badImg = (tmp / "badimg.vpcap").string();
    store(badImg, imgMeta, img, true);
    CHECK(replayRc(badImg, "--strict", "badimg1.log") != 0);
    CHECK(slurp("badimg1.log").find("image upload skipped") != std::string::npos && slurp("badimg1.log").find("skipped=1 ") != std::string::npos);
    CHECK(replayRc(badImg, "", "badimg2.log") == 0);

    // a command buffer that lost a command (its pipeline is not in the file) is left out of the submit
    std::vector<Rec> noPipe;
    for (const Rec& x : good)
      if (x.id != vpcap::id::vkCreateGraphicsPipelines) noPipe.push_back(x);
    const std::string np = (tmp / "nopipe.vpcap").string();
    store(np, meta, noPipe, true);
    CHECK(replayRc(np, "--strict", "np1.log") != 0);
    CHECK(slurp("np1.log").find("cbs skipped=0") == std::string::npos && slurp("np1.log").find("cbs skipped=") != std::string::npos);
    CHECK(replayRc(np, "", "np2.log") == 0);
  }
  // 7. A file the tool cannot use.
  CHECK(std::system(("\"" + std::string(gReplay) + "\" \"" + (tmp / "nope.vpcap").string() + "\" --lib \"" + gLib + "\" > /dev/null 2>&1").c_str()) != 0);
  if (const char* keep = std::getenv("VPREPLAY_TEST_KEEP"))  // for looking at the captures, logs and dumps of a failing run
    std::filesystem::copy(tmp, keep, std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing);
  std::filesystem::remove_all(tmp);
}

// vpreplay on lavapipe through the Vulkan loader (registered by CMake only when lavapipe's ICD manifest exists): the triangle
// capture must come out as a PNG with the triangle in it.
void testReplayLvp() {
  if (!gReplay || !gIcd) {
    std::puts("  skipped: run by ctest gpu_vulkan.replay_lvp (needs vpreplay and an ICD manifest)");
    return;
  }
  const auto tmp = std::filesystem::temp_directory_path() / ("vpreplay_lvp_" + std::to_string(getpid()));
  std::filesystem::remove_all(tmp);
  std::filesystem::create_directories(tmp);
  triangleThroughLoader(tmp, gIcd, true);
  std::filesystem::remove_all(tmp);
}

// Writes the reference captures of the replay comparison (tools/replay/compare.sh; workflows gpu.yml and replay.yml) into the
// directory in argv[3] (the slot gReplay reads): tri.vpcap, the flat triangle, and scene.vpcap, texture + depth + MSAA resolve + blend.
void testRefCaps() {
  if (!gReplay) {
    std::puts("  skipped: run by ctest gpu_vulkan.refcaps (needs the output directory)");
    return;
  }
  const std::filesystem::path dir = gReplay;
  std::filesystem::create_directories(dir);
  captureTriangle((dir / "tri.vpcap").string());
  captureScene((dir / "scene.vpcap").string());
  for (const char* name : {"tri.vpcap", "scene.vpcap"}) {
    vpcap::Reader rd;
    std::string err;
    CHECK(rd.open((dir / name).string(), &err));
    vpcap::Call c;
    int frames = 0, draws = 0, imageData = 0, submits = 0;
    while (rd.next(c)) {
      frames += c.id == vpcap::id::vpcapFrameEnd, draws += c.id == vpcap::id::vkCmdDraw, imageData += c.id == vpcap::id::vpcapImageData;
      submits += c.section == vpcap::Section::Frame && c.id == vpcap::id::vkQueueSubmit;
    }
    CHECK(rd.error().empty());
    const bool scene = std::strcmp(name, "scene.vpcap") == 0;
    CHECK(frames == 1 && submits == 1 && draws == (scene ? 3 : 1) && imageData == (scene ? 1 : 0));
  }
}

int main(int argc, char** argv) {
  gLib = argc > 2 ? argv[2] : "libVPVulkan.so";
  gReplay = argc > 3 ? argv[3] : nullptr;
  gIcd = argc > 4 ? argv[4] : nullptr;
  // The suites read the pipelines they create right away (the synchronous mode, opt-in); the async group turns the
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
      {"stage_keys", testStageKeys}, {"async_pipelines", testAsyncPipelines}, {"sync_pipelines", testSyncPipelines}, {"pso_census", testPsoCensus}, {"pass_drop", testPassDrop}, {"skip_reasons", testSkipReasons},
      {"perf", testPerf}, {"phase1b", testPhase1b}, {"suspend_resume", testSuspendResume}, {"translator_modes", testTranslatorModes}, {"capture", testCapture}, {"capture_images", testCaptureImages}, {"capture_hooks", testCaptureHooks}, {"capture_codec", testCaptureCodec}, {"capture_env", testCaptureEnv}, {"capture_pools", testCapturePools}, {"capture_host_frames", testCaptureHostFrames}, {"capture_corrupt", testCaptureCorrupt}, {"capture_env_host", testCaptureEnvHost}, {"capture_devices", testCaptureDevices}, {"capture_unhooked", testCaptureUnhooked}, {"capture_budget", testCaptureBudget}, {"capture_locks", testCaptureLocks}, {"png", testPng}, {"imgdiff", testImgDiff}, {"replay", testReplay}, {"replay_lvp", testReplayLvp}, {"refcaps", testRefCaps}};
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
