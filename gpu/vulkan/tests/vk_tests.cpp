// VPVulkan tests on the null backend. Usage: vpvk_tests <group> <path to libVPVulkan>.
#include "vpvk/config.h"
#include <dlfcn.h>

#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "commands.h"
#include "vpvk.h"

using namespace vpvk;

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
  CHECK(mvp.maxMultiviewViewCount == 2);
  // Formats: RGBA8 renderable + blendable, sRGB not storage, D24S8 depth, unknown format none.
  VkFormatProperties fp;
  vkGetPhysicalDeviceFormatProperties(c.pd, VK_FORMAT_R8G8B8A8_UNORM, &fp);
  CHECK(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT);
  vkGetPhysicalDeviceFormatProperties(c.pd, VK_FORMAT_R8G8B8A8_SRGB, &fp);
  CHECK(!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT));
  vkGetPhysicalDeviceFormatProperties(c.pd, VK_FORMAT_D24_UNORM_S8_UINT, &fp);
  CHECK(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
  vkGetPhysicalDeviceFormatProperties(c.pd, VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, &fp);
  CHECK(fp.optimalTilingFeatures == 0);
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
  // The density map is never advertised (VPEngine foveation only); enabling it anyway is ignored.
  for (const auto& x : e) CHECK(std::strcmp(x.extensionName, VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME) != 0);
  Ctx vr({VK_KHR_MULTIVIEW_EXTENSION_NAME, VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME,
          VK_EXT_METAL_OBJECTS_EXTENSION_NAME});
  CHECK(vr.dev && from<Device>(vr.dev)->multiview && !from<Device>(vr.dev)->densityMap);
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
  CHECK(image(c, VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 64, 1, VK_IMAGE_USAGE_SAMPLED_BIT, 0) == VK_NULL_HANDLE);
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

  CHECK(toStore(VK_ATTACHMENT_STORE_OP_NONE, false) == vpgpu::StoreAction::DontCare);
  CHECK(toStore(VK_ATTACHMENT_STORE_OP_DONT_CARE, true) == vpgpu::StoreAction::Resolve);
  CHECK(toLoad(VK_ATTACHMENT_LOAD_OP_NONE_KHR) == vpgpu::LoadAction::DontCare);
}

// Upload + clear + copies through vkQueueSubmit, fences, and the deferred list reaching the core.
void testSubmit() {
  Ctx c;
  VkDeviceMemory m = alloc(c, 1 << 20, 1);
  VkBuffer staging = buffer(c, 64 * 64 * 4, m, 0);
  VkBuffer other = buffer(c, 4096, m, 65536);
  VkImage tex = image(c, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 64, 1,
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
  vkCmdFillBuffer(cb, other, 0, 16, 0x01020304);  // not a byte pattern: dropped for now
  vkEndCommandBuffer(cb);

  vpgpu::Frame frame;
  TranslateStats st;
  translate(*from<CommandBuffer>(cb)->rec, frame, st);
  CHECK(st.copies == 3 && st.clears == 1 && st.dropped == 1);
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
  vkDestroyFence(c.dev, fence, nullptr);
  vkDestroyCommandPool(c.dev, pool, nullptr);
  vkDestroyImage(c.dev, tex, nullptr);
  vkDestroyImage(c.dev, rt, nullptr);
  vkDestroyBuffer(c.dev, staging, nullptr);
  vkDestroyBuffer(c.dev, other, nullptr);
  vkFreeMemory(c.dev, m, nullptr);
}

}  // namespace

int main(int argc, char** argv) {
  gLib = argc > 2 ? argv[2] : "libVPVulkan.so";
  const std::vector<std::pair<const char*, std::function<void()>>> groups = {
      {"exports", testExports}, {"physical", testPhysical}, {"memory", testMemory},
      {"passes", testPasses},   {"submit", testSubmit}};
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
