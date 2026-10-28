// Device, submission, streaming memory, images and samplers.
#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "context.h"

namespace vpgles {

namespace {
Device* gDevice = nullptr;
thread_local Context* tCurrent = nullptr;
Context* gLastCurrent = nullptr;  // last context made current on any thread (guarded by Device::lock)
}  // namespace

void logf(const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "%s\n", buf);
}

Device* device() { return gDevice; }
void setDevice(Device* d) { gDevice = d; }
Context* current() { return tCurrent; }
void setCurrent(Context* c) {
    tCurrent = c;
    if (c) gLastCurrent = c;
}
// For host calls (vpgles_wrap_image / vpgles_flush) from a thread that never made a context current
// (the OVRPlugin entry points are not always called from the render thread). Call with Device::lock held.
Context* currentOrLast() { return tCurrent ? tCurrent : gLastCurrent; }
void forgetContext(Context* c) {
    if (gLastCurrent == c) gLastCurrent = nullptr;
}

// ---- diagnostics: pipeline recipes ----------------------------------------------------------------------
namespace {
struct RecipeStats {
    uint32_t last[4] = {~0u, ~0u, ~0u, ~0u};  // counters of the last line (none printed yet)
    std::chrono::steady_clock::time_point at = std::chrono::steady_clock::now();
} gRecipeStats;
}  // namespace

void recipeStatsTick(bool force) {
    Device* dv = device();
    if (!dv || !dv->device || !dv->vk.vpvk_recipe_stats) return;
    RecipeStats& st = gRecipeStats;
    const auto now = std::chrono::steady_clock::now();
    if (!force && now - st.at < std::chrono::seconds(10)) return;
    st.at = now;
    uint32_t v[4];  // loaded, saved, prewarmed, miss_in_game
    dv->vk.vpvk_recipe_stats(dv->device, v);
    if (!force && std::equal(v, v + 4, st.last)) return;
    std::copy(v, v + 4, st.last);
    logf("gles.pso.recipe saved=%u loaded=%u prewarmed=%u miss_in_game=%u", v[1], v[0], v[2], v[3]);
}

// ---- Device -----------------------------------------------------------------------------------------
extern "C" void vpvk_internal_scope(int on);  // VPVulkan (ext.cpp): full extension list for this internal client

bool Device::init(PFN_vkGetInstanceProcAddr gipa) {
    if (!vk.loadInstance(gipa, nullptr)) return false;
    struct Scope {
        Scope() { vpvk_internal_scope(1); }
        ~Scope() { vpvk_internal_scope(0); }
    } scope;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vpgles";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (vk.vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) return false;
    vk.loadInstance(gipa, instance);
    uint32_t n = 1;
    if (vk.vkEnumeratePhysicalDevices(instance, &n, &phys) < 0 || !phys) return false;
    vk.vkGetPhysicalDeviceProperties(phys, &props);
    vk.vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
    float prio = 1.0f;
    VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    q.queueFamilyIndex = 0, q.queueCount = 1, q.pQueuePriorities = &prio;
    const char* exts[] = {VK_KHR_MULTIVIEW_EXTENSION_NAME, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
                          VK_KHR_MAINTENANCE1_EXTENSION_NAME};
    VkPhysicalDeviceDynamicRenderingFeatures dr{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES};
    dr.dynamicRendering = VK_TRUE;
    VkPhysicalDeviceMultiviewFeatures mv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES, &dr};
    mv.multiview = VK_TRUE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &mv};
    dci.queueCreateInfoCount = 1, dci.pQueueCreateInfos = &q;
    dci.enabledExtensionCount = 3, dci.ppEnabledExtensionNames = exts;
    if (vk.vkCreateDevice(phys, &dci, nullptr, &device) != VK_SUCCESS) return false;
    if (!vk.loadDevice(device)) {
        logf("gles.err what=vk_device_functions");
        return false;
    }
    vk.vkGetDeviceQueue(device, 0, 0, &queue);
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vk.vkCreateCommandPool(device, &pci, nullptr, &pool);
    logf("gles.init device=%s vendor=0x%x", props.deviceName, props.vendorID);
    return true;
}

void Device::shutdown() {
    if (!device) return;
    texStatsTick(true);
    recipeStatsTick(true);
    vk.vkDeviceWaitIdle(device);
    retire(true);
    for (VkDescriptorPool p : freePools) vk.vkDestroyDescriptorPool(device, p, nullptr);
    freePools.clear();
    vk.vkDestroyCommandPool(device, pool, nullptr);
    vk.vkDestroyDevice(device, nullptr);
    vk.vkDestroyInstance(instance, nullptr);
    device = VK_NULL_HANDLE;
}

int Device::memoryType(uint32_t bits, VkMemoryPropertyFlags want) const {
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & want) == want) return int(i);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
        if (bits & (1u << i)) return int(i);
    return -1;
}

StorageP Device::newStorage(VkDeviceSize size, VkBufferUsageFlags usage) {
    auto s = std::make_shared<BufferStorage>();
    s->dev = this;
    s->size = std::max<VkDeviceSize>(size, 16);
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = s->size;
    bci.usage = usage;
    if (vk.vkCreateBuffer(device, &bci, nullptr, &s->buffer) != VK_SUCCESS) return nullptr;
    VkMemoryRequirements req;
    vk.vkGetBufferMemoryRequirements(device, s->buffer, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = uint32_t(memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    if (vk.vkAllocateMemory(device, &mai, nullptr, &s->memory) != VK_SUCCESS) return nullptr;
    vk.vkBindBufferMemory(device, s->buffer, s->memory, 0);
    void* p = nullptr;
    vk.vkMapMemory(device, s->memory, 0, VK_WHOLE_SIZE, 0, &p);
    s->host = static_cast<uint8_t*>(p);
    return s;
}

BufferStorage::~BufferStorage() {
    if (!dev || !dev->device) return;
    if (host) dev->vk.vkUnmapMemory(dev->device, memory);
    dev->vk.vkDestroyBuffer(dev->device, buffer, nullptr);
    dev->vk.vkFreeMemory(dev->device, memory, nullptr);
}

void Device::retire(bool wait) {
    for (auto it = inFlight.begin(); it != inFlight.end();) {
        VkResult r = wait ? vk.vkWaitForFences(device, 1, &it->fence, VK_TRUE, UINT64_MAX) : vk.vkGetFenceStatus(device, it->fence);
        if (r != VK_SUCCESS) {
            ++it;
            continue;
        }
        completed = std::max(completed, it->serial);
        vk.vkDestroyFence(device, it->fence, nullptr);
        vk.vkFreeCommandBuffers(device, pool, 1, &it->cb);
        if (it->descPool) {
            vk.vkResetDescriptorPool(device, it->descPool, 0);
            freePools.push_back(it->descPool);
        }
        it = inFlight.erase(it);
    }
}

void Device::waitSerial(uint64_t serial) {
    while (completed < serial && !inFlight.empty()) {
        InFlight& f = inFlight.front();
        vk.vkWaitForFences(device, 1, &f.fence, VK_TRUE, UINT64_MAX);
        retire(false);
    }
}

VkDescriptorPool Device::takePool() {
    if (!freePools.empty()) {
        VkDescriptorPool p = freePools.back();
        freePools.pop_back();
        return p;
    }
    VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1024},
                                    {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4096},
                                    {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8192},
                                    {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 2048},
                                    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096},
                                    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2048},
                                    {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1024}};
    VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ci.maxSets = 1024, ci.poolSizeCount = 7, ci.pPoolSizes = sizes;
    VkDescriptorPool p = VK_NULL_HANDLE;
    vk.vkCreateDescriptorPool(device, &ci, nullptr, &p);
    return p;
}

TexelBufferView::~TexelBufferView() {
    if (dev && dev->device && view) dev->vk.vkDestroyBufferView(dev->device, view, nullptr);
}

// ---- Images ----------------------------------------------------------------------------------------
Image::~Image() {
    if (!dev || !dev->device) return;
    for (auto& [k, v] : views) dev->vk.vkDestroyImageView(dev->device, v, nullptr);
    if (!external) {
        dev->vk.vkDestroyImage(dev->device, image, nullptr);
        dev->vk.vkFreeMemory(dev->device, memory, nullptr);
    }
}

VkImageAspectFlags Image::aspect() const {
    VkImageAspectFlags a = 0;
    if (format->depth) a |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if (format->stencil) a |= VK_IMAGE_ASPECT_STENCIL_BIT;
    return a ? a : VkImageAspectFlags(VK_IMAGE_ASPECT_COLOR_BIT);
}

VkImageView Image::attachmentView(uint32_t level, uint32_t layer, uint32_t layerCount) {
    const uint64_t key = 1ull << 63 | uint64_t(level) << 40 | uint64_t(layer) << 20 | layerCount;
    auto it = views.find(key);
    if (it != views.end()) return it->second;
    VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ci.image = image;
    ci.viewType = layerCount > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    ci.format = format->vk;
    ci.subresourceRange = {aspect(), level, 1, layer, layerCount};
    VkImageView v = VK_NULL_HANDLE;
    dev->vk.vkCreateImageView(dev->device, &ci, nullptr, &v);
    views[key] = v;
    return v;
}

VkImageView Image::storageView(uint32_t level, uint32_t layer, uint32_t layerCount, VkImageViewType type) {
    const uint64_t key = 1ull << 62 | uint64_t(type) << 56 | uint64_t(level) << 40 | uint64_t(layer) << 20 | layerCount;
    auto it = views.find(key);
    if (it != views.end()) return it->second;
    VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ci.image = image;
    ci.viewType = type;
    ci.format = format->vk;
    ci.subresourceRange = {aspect(), level, 1, layer, layerCount};
    VkImageView v = VK_NULL_HANDLE;
    dev->vk.vkCreateImageView(dev->device, &ci, nullptr, &v);
    views[key] = v;
    return v;
}

bool SamplerParams::operator==(const SamplerParams& o) const {
    return minFilter == o.minFilter && magFilter == o.magFilter && wrapS == o.wrapS && wrapT == o.wrapT &&
           wrapR == o.wrapR && minLod == o.minLod && maxLod == o.maxLod && maxAnisotropy == o.maxAnisotropy &&
           compareMode == o.compareMode && compareFunc == o.compareFunc && std::memcmp(border, o.border, sizeof border) == 0;
}

uint64_t Texture::viewKey() const {
    uint64_t k = uint64_t(baseLevel & 0xFF) | uint64_t(std::min(maxLevel, 255) & 0xFF) << 8;
    for (int i = 0; i < 4; ++i) k |= uint64_t(swizzle[i] & 0xF) << (16 + 4 * i);
    k |= uint64_t(depthStencilMode == gl::STENCIL_INDEX) << 32;
    k |= uint64_t(!srgbDecode) << 33;
    return k;
}

static VkComponentSwizzle glSwizzle(GLenum s, VkComponentSwizzle base[4]) {
    switch (s) {
        case gl::RED: return base[0] == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_R : base[0];
        case gl::GREEN: return base[1] == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_G : base[1];
        case gl::BLUE: return base[2] == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_B : base[2];
        case gl::ALPHA: return base[3] == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_A : base[3];
        case gl::ZERO: return VK_COMPONENT_SWIZZLE_ZERO;
        case gl::ONE: return VK_COMPONENT_SWIZZLE_ONE;
        default: return VK_COMPONENT_SWIZZLE_IDENTITY;
    }
}

VkImageView Texture::sampledView() {
    if (!image) return VK_NULL_HANDLE;
    const uint64_t key = viewKey();
    auto it = image->views.find(key);
    if (it != image->views.end()) return it->second;
    VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ci.image = image->image;
    switch (target) {
        case gl::TEXTURE_CUBE_MAP: ci.viewType = VK_IMAGE_VIEW_TYPE_CUBE; break;
        case gl::TEXTURE_3D: ci.viewType = VK_IMAGE_VIEW_TYPE_3D; break;
        case gl::TEXTURE_2D_ARRAY: ci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY; break;
        default: ci.viewType = image->layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    }
    ci.format = image->format->vk;
    if (!srgbDecode && image->format->vk == VK_FORMAT_R8G8B8A8_SRGB) ci.format = VK_FORMAT_R8G8B8A8_UNORM;
    VkComponentSwizzle base[4];
    for (int i = 0; i < 4; ++i) base[i] = image->format->swz[i];
    ci.components = {glSwizzle(swizzle[0], base), glSwizzle(swizzle[1], base), glSwizzle(swizzle[2], base),
                     glSwizzle(swizzle[3], base)};
    VkImageAspectFlags aspect = image->aspect();
    if (aspect == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))
        aspect = depthStencilMode == gl::STENCIL_INDEX ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
    const uint32_t base0 = uint32_t(std::min<GLint>(baseLevel, GLint(image->levels) - 1));
    const uint32_t top = uint32_t(std::min<GLint>(maxLevel, GLint(image->levels) - 1));
    ci.subresourceRange = {aspect, base0, std::max<uint32_t>(1, top - base0 + 1), 0, image->layers};
    VkImageView v = VK_NULL_HANDLE;
    image->dev->vk.vkCreateImageView(image->dev->device, &ci, nullptr, &v);
    image->views[key] = v;
    return v;
}

// ---- Context: recording -----------------------------------------------------------------------------
Context::Context(Device* d, ShareGroup* s, int maj, int min) : dev(d), share(s), major(maj), minor(min) {
    ++share->refs;
    vaos[0] = std::make_unique<VertexArray>();
    fbos[0] = std::make_unique<Framebuffer>();
    fbos[0]->drawBuffers[0] = gl::BACK;
    fbos[0]->readBuffer = gl::BACK;
    for (auto& a : currentAttrib) a = {0, 0, 0, 1};
}

Context::~Context() {
    flush(true);
    if (occPool_) dev->vk.vkDestroyQueryPool(dev->device, occPool_, nullptr);
    for (VkSampler s : samplerCacheList_) dev->vk.vkDestroySampler(dev->device, s, nullptr);
    if (--share->refs == 0) delete share;
}

VkCommandBuffer Context::cmd() {
    if (cb_) return cb_;
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = dev->pool, ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, ai.commandBufferCount = 1;
    dev->vk.vkAllocateCommandBuffers(dev->device, &ai, &cb_);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    dev->vk.vkBeginCommandBuffer(cb_, &bi);
    if (occActive_ != UINT32_MAX) dev->vk.vkCmdBeginQuery(cb_, occPool_, occActive_, 0);  // the counting goes on in the new buffer
    return cb_;
}

// ---- Occlusion queries: VPVulkan counts the samples of the draws recorded between vkCmdBeginQuery and vkCmdEndQuery ----
uint32_t Context::occSlot() {
    constexpr uint32_t kSlots = 256;
    if (!occPool_) {
        VkQueryPoolCreateInfo ci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        ci.queryType = VK_QUERY_TYPE_OCCLUSION, ci.queryCount = kSlots;
        if (dev->vk.vkCreateQueryPool(dev->device, &ci, nullptr, &occPool_) != VK_SUCCESS) {
            occPool_ = VK_NULL_HANDLE;
            return UINT32_MAX;
        }
        occBusy_.assign(kSlots, false);
    }
    for (uint32_t i = 0; i < kSlots; ++i)
        if (!occBusy_[i]) return occBusy_[i] = true, i;
    return UINT32_MAX;
}

void Context::occBegin(Query& q) {
    if (q.slot == UINT32_MAX) q.slot = occSlot();
    if (q.slot == UINT32_MAX || occActive_ != UINT32_MAX) return;  // no slot / another query counting: this one stays coherent
    endPass();
    dev->vk.vkCmdResetQueryPool(cmd(), occPool_, q.slot, 1);
    dev->vk.vkCmdBeginQuery(cmd(), occPool_, q.slot, 0);
    occActive_ = q.slot;
}

void Context::occEnd(Query& q) {
    if (q.slot == UINT32_MAX || occActive_ != q.slot) return;
    dev->vk.vkCmdEndQuery(cmd(), occPool_, q.slot);
    occActive_ = UINT32_MAX;
    q.endSerial = dev->submitted + 1;  // the serial the open command buffer will get
}

void Context::occPause() {
    if (occActive_ != UINT32_MAX) dev->vk.vkCmdEndQuery(cmd(), occPool_, occActive_);
}

void Context::occResume() {
    if (occActive_ != UINT32_MAX) dev->vk.vkCmdBeginQuery(cmd(), occPool_, occActive_, 0);
}

bool Context::occResult(Query& q, bool wait, uint64_t& value) {
    if (dev->submitted < q.endSerial) flush(false);
    uint64_t data[2] = {0, 0};
    const VkResult r = dev->vk.vkGetQueryPoolResults(dev->device, occPool_, q.slot, 1, sizeof data, data, sizeof data,
                                                     VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT |
                                                         (wait ? VK_QUERY_RESULT_WAIT_BIT : 0));
    if (r < 0) {  // device lost: the coherent answer stands
        value = 1;
        return true;
    }
    value = data[0];
    return data[1] != 0;
}

void Context::endPass() {
    if (!passOpen_) return;
    dev->vk.vkCmdEndRendering(cb_);
    passOpen_ = false;
    pass_ = PassKey{};
    passDrop_ = 0;
}

void Context::barrier() {
    endPass();
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    dev->vk.vkCmdPipelineBarrier(cmd(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1,
                                 &mb, 0, nullptr, 0, nullptr);
}

void Context::useStorage(const StorageP& s) {
    if (!s) return;
    if (s->lastUse != dev->submitted + 1) {
        s->lastUse = dev->submitted + 1;  // the serial the open command buffer will get
        used_.push_back(s);
    }
}

uint64_t Context::flush(bool wait) {
    if (!cb_) {
        if (wait) dev->waitSerial(dev->submitted);
        return 0;
    }
    endPass();
    dev->vk.vkEndCommandBuffer(cb_);
    Device::InFlight f;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    dev->vk.vkCreateFence(dev->device, &fci, nullptr, &f.fence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1, si.pCommandBuffers = &cb_;
    dev->vk.vkQueueSubmit(dev->queue, 1, &si, f.fence);
    f.serial = ++dev->submitted;
    f.cb = cb_;
    f.descPool = descPool_;
    for (auto& s : used_) f.keep.push_back(s);
    for (auto& c : chunks_) f.keep.push_back(c.storage);
    for (auto& k : keep_) f.keep.push_back(std::move(k));
    dev->inFlight.push_back(std::move(f));
    used_.clear();
    keep_.clear();
    // Chunks go back to the spare list once their submission retires (kept alive by keep above;
    // reused only through spareChunks_ after the wait below or a later retire).
    chunks_.clear();
    cb_ = VK_NULL_HANDLE;
    descPool_ = VK_NULL_HANDLE;
    const uint64_t serial = dev->submitted;
    if (wait) dev->waitSerial(serial);
    else dev->retire(false);
    return serial;
}

Context::Alloc Context::stream(VkDeviceSize size, VkDeviceSize align) {
    constexpr VkDeviceSize kChunk = 4 << 20;
    if (chunks_.empty() || (chunks_.back().used + align - 1) / align * align + size > chunks_.back().storage->size) {
        Chunk c;
        c.storage = dev->newStorage(std::max(kChunk, size), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                              VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        chunks_.push_back(c);
    }
    Chunk& c = chunks_.back();
    VkDeviceSize off = (c.used + align - 1) / align * align;
    c.used = off + size;
    return {c.storage->buffer, off, c.storage->host + off};
}

VkDescriptorSet Context::allocSet(VkDescriptorSetLayout layout) {
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (!descPool_) descPool_ = dev->takePool();
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = descPool_, ai.descriptorSetCount = 1, ai.pSetLayouts = &layout;
        VkDescriptorSet s = VK_NULL_HANDLE;
        if (dev->vk.vkAllocateDescriptorSets(dev->device, &ai, &s) == VK_SUCCESS) return s;
        // Pool exhausted: submit what we have (the pool travels with it) and start a new one.
        flush(false);
    }
    return VK_NULL_HANDLE;
}

void Context::prepareWrite(Buffer& b, bool discard) {
    if (!b.storage) return;
    const uint64_t pending = b.storage->lastUse;
    const bool gpuMayRead = pending && pending > dev->completed;
    if (!gpuMayRead) return;
    // New storage (copy on write); the old one stays alive with the work that reads it.
    StorageP fresh = dev->newStorage(b.storage->size, b.storage->dev ? VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                                      : 0);
    if (!fresh) return;
    if (!discard) std::memcpy(fresh->host, b.storage->host, size_t(b.storage->size));
    b.storage = fresh;
}

void Context::imageInit(Image& img) {
    endPass();
    VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    ib.srcAccessMask = 0;
    ib.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = img.image;
    ib.subresourceRange = {img.aspect(), 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    dev->vk.vkCmdPipelineBarrier(cmd(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                                 0, nullptr, 1, &ib);
}

static VkFilter filterOf(GLenum f) { return f == gl::NEAREST || f == gl::NEAREST_MIPMAP_NEAREST || f == gl::NEAREST_MIPMAP_LINEAR ? VK_FILTER_NEAREST : VK_FILTER_LINEAR; }
static VkSamplerAddressMode wrapOf(GLenum w) {
    switch (w) {
        case gl::CLAMP_TO_EDGE: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case gl::MIRRORED_REPEAT: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case gl::CLAMP_TO_BORDER: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        default: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }
}

VkSampler Context::samplerFor(const SamplerParams& p, const Texture& t) {
    // Integer textures and external images: nearest; textures without mips: no mip filtering.
    SamplerParams q = p;
    const bool mipmapped = p.minFilter != gl::NEAREST && p.minFilter != gl::LINEAR;
    const bool integer = t.image && t.image->format->integer;
    if (integer) q.minFilter = q.magFilter = gl::NEAREST;
    for (size_t i = 0; i < samplerCache_.size(); ++i)
        if (samplerCache_[i].first == q && samplerCacheMip_[i] == mipmapped) return samplerCacheList_[i];
    VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    ci.magFilter = filterOf(q.magFilter);
    ci.minFilter = filterOf(q.minFilter);
    ci.mipmapMode = q.minFilter == gl::NEAREST_MIPMAP_LINEAR || q.minFilter == gl::LINEAR_MIPMAP_LINEAR ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                                                                                         : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    ci.addressModeU = wrapOf(q.wrapS), ci.addressModeV = wrapOf(q.wrapT), ci.addressModeW = wrapOf(q.wrapR);
    ci.minLod = mipmapped ? q.minLod : 0.0f;
    ci.maxLod = mipmapped ? q.maxLod : 0.25f;  // GL: no mipmapping = level base only
    if (ci.minLod < 0) ci.minLod = 0;
    ci.anisotropyEnable = q.maxAnisotropy > 1.0f;
    ci.maxAnisotropy = q.maxAnisotropy;
    ci.compareEnable = q.compareMode == gl::COMPARE_REF_TO_TEXTURE;
    ci.compareOp = VkCompareOp(q.compareFunc - gl::NEVER);
    ci.borderColor = q.border[3] > 0.5f ? (q.border[0] > 0.5f ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE : VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK)
                                        : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    VkSampler s = VK_NULL_HANDLE;
    dev->vk.vkCreateSampler(dev->device, &ci, nullptr, &s);
    samplerCache_.push_back({q, s});
    samplerCacheMip_.push_back(mipmapped);
    samplerCacheList_.push_back(s);
    return s;
}

}  // namespace vpgles
