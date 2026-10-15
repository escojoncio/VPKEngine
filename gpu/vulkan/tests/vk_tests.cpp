// VPVulkan tests on the null backend. Usage: vpvk_tests <group> <path to libVPVulkan>.
#include "vpvk/config.h"
#include <dlfcn.h>

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
#include "vpvk.h"

using namespace vpvk;

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
  Ctx c;
  const auto& mp = memoryProperties();
  CHECK(mp.memoryHeapCount == 1);
  CHECK(mp.memoryTypes[1].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  VkDeviceMemory m = alloc(c, 1 << 20, 1);
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
    vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe[1]);  // already recorded
    CHECK(std::filesystem::file_size(file, ec) == size1);
    c.gpu().pipelines().waitIdle();
    for (VkPipeline p : pipe) vkDestroyPipeline(c.dev, p, nullptr);
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

int main(int argc, char** argv) {
  gLib = argc > 2 ? argv[2] : "libVPVulkan.so";
  const std::vector<std::pair<const char*, std::function<void()>>> groups = {
      {"exports", testExports}, {"physical", testPhysical}, {"memory", testMemory},
      {"passes", testPasses},   {"submit", testSubmit}, {"pipeline", testPipeline},
      {"recipes", testRecipes}, {"taa", testTaa}, {"msaa", testMsaa}, {"queries", testQueries},
      {"timeline", testTimeline}, {"renderpass_ops", testRenderPassOps}, {"linear", testLinear}, {"images", testImages}};
  bool ran = false;
  for (const auto& [name, fn] : groups) {
    if (argc > 1 && std::strcmp(argv[1], name) != 0) continue;
    std::printf("[%s]\n", name);
    fn();
    ran = true;
  }
  if (!ran) std::printf("unknown group\n");
  std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
  return failures || !ran ? 1 : 0;
}
