// Extensions layered on the core driver: VK_KHR_copy_commands2, VK_KHR_synchronization2,
// VK_KHR_timeline_semaphore and VK_EXT_multi_draw. Each entry point maps onto what the driver
// already does (one in-order queue, per-frame completion tracking), so none adds GPU work of its own.
//
// They are advertised only with VPVK_EXTRA_EXTENSIONS=1 (physical_device.cpp), which the conformance
// run sets: a game that sees them may switch to code paths the Vulkan CTS has to validate first.
// The functions resolve either way (vkGetDeviceProcAddr), as for any unadvertised extension.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "commands.h"
#include "vpvk.h"

#define VPVK_EXPORT extern "C" __attribute__((visibility("default")))

namespace vpvk {

namespace {
// VPKEngine's own GLES layer (gpu/gles) is an internal client: it creates its instance/device inside
// vpvk_internal_scope(1)...(0) and sees the full extension list (dynamic rendering, …) regardless of the
// CTS gate, which only governs what GAMES see.
thread_local int tInternal = 0;
}  // namespace

bool reportGeom() {
  const char* g = std::getenv("VPVK_REPORT_GEOM");
  return g && *g == '1';
}

bool extraExtensions() {
  if (tInternal) return true;
  const char* e = std::getenv("VPVK_EXTRA_EXTENSIONS");
  return e && *e == '1';
}

namespace {
// Moves the semaphore's finished submissions into `completed` (non-blocking).
void advance(Platform& p, Semaphore& s) {
  auto it = s.pending.begin();
  while (it != s.pending.end()) {
    // Submissions append in order, so their frames only grow: the first unfinished one ends it.
    if (p.waitFrames(it->second, 0) != Platform::WaitResult::Done) break;
    s.completed = std::max(s.completed, it->first);
    it = s.pending.erase(it);
  }
}
}  // namespace

}  // namespace vpvk

using namespace vpvk;

// ---- VK_KHR_timeline_semaphore -------------------------------------------------------------------
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetSemaphoreCounterValue(VkDevice dv, VkSemaphore sem, uint64_t* value) {
  Device* d = from<Device>(dv);
  if (d->platform->lost()) return VK_ERROR_DEVICE_LOST;
  Semaphore* s = from<Semaphore>(sem);
  std::lock_guard<std::mutex> l(s->m);
  advance(*d->platform, *s);
  *value = s->completed;
  return VK_SUCCESS;
}

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkSignalSemaphore(VkDevice dv, const VkSemaphoreSignalInfo* info) {
  Semaphore* s = from<Semaphore>(info->semaphore);
  {
    std::lock_guard<std::mutex> l(s->m);
    s->completed = std::max(s->completed, info->value);
  }
  submitDeferred(from<Device>(dv));  // batches waiting on this value (wait-before-signal) can run now
  return VK_SUCCESS;
}

// Waits by the frames the awaited values depend on; a value only the host will signal (no pending
// submission reaches it) is polled until the timeout.
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkWaitSemaphores(VkDevice dv, const VkSemaphoreWaitInfo* info, uint64_t timeout) {
  Device* d = from<Device>(dv);
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  const bool any = info->flags & VK_SEMAPHORE_WAIT_ANY_BIT;
  for (;;) {
    if (d->platform->lost()) return VK_ERROR_DEVICE_LOST;
    uint32_t done = 0;
    uint64_t needFrames = 0;
    bool frameWait = false, hostOnly = false;  // hostOnly: a value no pending submission reaches
    for (uint32_t i = 0; i < info->semaphoreCount; ++i) {
      Semaphore* s = from<Semaphore>(info->pSemaphores[i]);
      std::lock_guard<std::mutex> l(s->m);
      advance(*d->platform, *s);
      if (s->completed >= info->pValues[i]) {
        ++done;
        continue;
      }
      // The earliest pending submission that reaches the value.
      uint64_t best = UINT64_MAX;
      for (const auto& [v, f] : s->pending)
        if (v >= info->pValues[i]) best = std::min(best, f);
      if (best != UINT64_MAX) {
        needFrames = frameWait ? (any ? std::min(needFrames, best) : std::max(needFrames, best)) : best;
        frameWait = true;
      } else {
        hostOnly = true;
      }
    }
    if (any ? done > 0 : done == info->semaphoreCount) return VK_SUCCESS;
    const uint64_t elapsed =
        uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
    if (elapsed >= timeout) return VK_TIMEOUT;
    const uint64_t left = timeout - elapsed;
    // A host signal (vkSignalSemaphore from another thread) wakes nothing here: when one could end
    // the wait (ANY), the GPU wait is cut into 1 ms slices so it is noticed.
    if (frameWait) {
      const uint64_t slice = any && hostOnly ? std::min<uint64_t>(left, 1'000'000) : left;
      const Platform::WaitResult r = d->platform->waitFrames(needFrames, slice);
      if (r == Platform::WaitResult::Lost) return VK_ERROR_DEVICE_LOST;
    } else {
      std::this_thread::sleep_for(std::chrono::nanoseconds(std::min<uint64_t>(left, 1'000'000)));
    }
  }
}

// ---- VK_KHR_copy_commands2 -----------------------------------------------------------------------
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer2(VkCommandBuffer cb, const VkCopyBufferInfo2* i) {
  std::vector<VkBufferCopy> r(i->regionCount);
  for (uint32_t k = 0; k < i->regionCount; ++k) r[k] = {i->pRegions[k].srcOffset, i->pRegions[k].dstOffset, i->pRegions[k].size};
  vkCmdCopyBuffer(cb, i->srcBuffer, i->dstBuffer, i->regionCount, r.data());
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyImage2(VkCommandBuffer cb, const VkCopyImageInfo2* i) {
  std::vector<VkImageCopy> r(i->regionCount);
  for (uint32_t k = 0; k < i->regionCount; ++k) {
    const VkImageCopy2& s = i->pRegions[k];
    r[k] = {s.srcSubresource, s.srcOffset, s.dstSubresource, s.dstOffset, s.extent};
  }
  vkCmdCopyImage(cb, i->srcImage, i->srcImageLayout, i->dstImage, i->dstImageLayout, i->regionCount, r.data());
}
static std::vector<VkBufferImageCopy> bufferImageRegions(uint32_t n, const VkBufferImageCopy2* p) {
  std::vector<VkBufferImageCopy> r(n);
  for (uint32_t k = 0; k < n; ++k)
    r[k] = {p[k].bufferOffset, p[k].bufferRowLength, p[k].bufferImageHeight, p[k].imageSubresource, p[k].imageOffset,
            p[k].imageExtent};
  return r;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyBufferToImage2(VkCommandBuffer cb, const VkCopyBufferToImageInfo2* i) {
  auto r = bufferImageRegions(i->regionCount, i->pRegions);
  vkCmdCopyBufferToImage(cb, i->srcBuffer, i->dstImage, i->dstImageLayout, i->regionCount, r.data());
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyImageToBuffer2(VkCommandBuffer cb, const VkCopyImageToBufferInfo2* i) {
  auto r = bufferImageRegions(i->regionCount, i->pRegions);
  vkCmdCopyImageToBuffer(cb, i->srcImage, i->srcImageLayout, i->dstBuffer, i->regionCount, r.data());
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBlitImage2(VkCommandBuffer cb, const VkBlitImageInfo2* i) {
  std::vector<VkImageBlit> r(i->regionCount);
  for (uint32_t k = 0; k < i->regionCount; ++k) {
    const VkImageBlit2& s = i->pRegions[k];
    r[k].srcSubresource = s.srcSubresource, r[k].dstSubresource = s.dstSubresource;
    r[k].srcOffsets[0] = s.srcOffsets[0], r[k].srcOffsets[1] = s.srcOffsets[1];
    r[k].dstOffsets[0] = s.dstOffsets[0], r[k].dstOffsets[1] = s.dstOffsets[1];
  }
  vkCmdBlitImage(cb, i->srcImage, i->srcImageLayout, i->dstImage, i->dstImageLayout, i->regionCount, r.data(), i->filter);
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdResolveImage2(VkCommandBuffer cb, const VkResolveImageInfo2* i) {
  std::vector<VkImageResolve> r(i->regionCount);
  for (uint32_t k = 0; k < i->regionCount; ++k) {
    const VkImageResolve2& s = i->pRegions[k];
    r[k] = {s.srcSubresource, s.srcOffset, s.dstSubresource, s.dstOffset, s.extent};
  }
  vkCmdResolveImage(cb, i->srcImage, i->srcImageLayout, i->dstImage, i->dstImageLayout, i->regionCount, r.data());
}

// ---- VK_KHR_synchronization2 ---------------------------------------------------------------------
// One in-order queue whose hazards the frame compiler derives from the commands themselves: barriers
// and event waits are counted, not executed (as their v1 forms).
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo*) {
  Recorder& r = *from<CommandBuffer>(cb)->rec;
  ++r.barriers;
  if (r.inPass) r.ops.emplace_back(OpPassBarrier{});  // subpass self-dependency (see vkCmdPipelineBarrier)
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetEvent2(VkCommandBuffer cb, VkEvent e, const VkDependencyInfo*) {
  vkCmdSetEvent(cb, e, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdResetEvent2(VkCommandBuffer cb, VkEvent e, VkPipelineStageFlags2) {
  vkCmdResetEvent(cb, e, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdWaitEvents2(VkCommandBuffer cb, uint32_t, const VkEvent*, const VkDependencyInfo*) {
  ++from<CommandBuffer>(cb)->rec->barriers;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdWriteTimestamp2(VkCommandBuffer cb, VkPipelineStageFlags2, VkQueryPool qp, uint32_t q) {
  vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, qp, q);
}

// vkQueueSubmit2 as vkQueueSubmit: command buffers in order, timeline wait / signal values carried
// over (waits matter: a wait on a value the host signals later holds the batch back).
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit2(VkQueue q, uint32_t n, const VkSubmitInfo2* si, VkFence fence) {
  struct One {
    std::vector<VkCommandBuffer> cbs;
    std::vector<VkSemaphore> waits, signals;
    std::vector<VkPipelineStageFlags> waitStages;
    std::vector<uint64_t> waitValues, values;
    VkTimelineSemaphoreSubmitInfo tl{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  };
  std::vector<One> ones(n);
  std::vector<VkSubmitInfo> v1(n);
  for (uint32_t i = 0; i < n; ++i) {
    One& o = ones[i];
    for (uint32_t k = 0; k < si[i].commandBufferInfoCount; ++k) o.cbs.push_back(si[i].pCommandBufferInfos[k].commandBuffer);
    for (uint32_t k = 0; k < si[i].waitSemaphoreInfoCount; ++k) {
      o.waits.push_back(si[i].pWaitSemaphoreInfos[k].semaphore);
      o.waitValues.push_back(si[i].pWaitSemaphoreInfos[k].value);
      o.waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    }
    for (uint32_t k = 0; k < si[i].signalSemaphoreInfoCount; ++k) {
      o.signals.push_back(si[i].pSignalSemaphoreInfos[k].semaphore);
      o.values.push_back(si[i].pSignalSemaphoreInfos[k].value);
    }
    o.tl.waitSemaphoreValueCount = uint32_t(o.waitValues.size());
    o.tl.pWaitSemaphoreValues = o.waitValues.data();
    o.tl.signalSemaphoreValueCount = uint32_t(o.values.size());
    o.tl.pSignalSemaphoreValues = o.values.data();
    VkSubmitInfo& s = v1[i];
    s = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    s.pNext = &o.tl;
    s.waitSemaphoreCount = uint32_t(o.waits.size());
    s.pWaitSemaphores = o.waits.data();
    s.pWaitDstStageMask = o.waitStages.data();
    s.commandBufferCount = uint32_t(o.cbs.size());
    s.pCommandBuffers = o.cbs.data();
    s.signalSemaphoreCount = uint32_t(o.signals.size());
    s.pSignalSemaphores = o.signals.data();
  }
  return vkQueueSubmit(q, n, v1.data(), fence);
}

// ---- VK_EXT_multi_draw ---------------------------------------------------------------------------
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDrawMultiEXT(VkCommandBuffer cb, uint32_t count, const VkMultiDrawInfoEXT* info,
                                                         uint32_t instances, uint32_t firstInstance, uint32_t stride) {
  const auto* p = reinterpret_cast<const uint8_t*>(info);
  if (!stride) stride = sizeof(VkMultiDrawInfoEXT);
  for (uint32_t i = 0; i < count; ++i) {
    const auto& d = *reinterpret_cast<const VkMultiDrawInfoEXT*>(p + size_t(i) * stride);
    const size_t before = recordedOps(cb);
    vkCmdDraw(cb, d.vertexCount, instances, d.firstVertex, firstInstance);
    setDrawIndexSince(cb, before, i);  // gl_DrawID = the draw's index in the multi-draw
  }
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDrawMultiIndexedEXT(VkCommandBuffer cb, uint32_t count,
                                                                const VkMultiDrawIndexedInfoEXT* info, uint32_t instances,
                                                                uint32_t firstInstance, uint32_t stride,
                                                                const int32_t* vertexOffset) {
  const auto* p = reinterpret_cast<const uint8_t*>(info);
  if (!stride) stride = sizeof(VkMultiDrawIndexedInfoEXT);
  for (uint32_t i = 0; i < count; ++i) {
    const auto& d = *reinterpret_cast<const VkMultiDrawIndexedInfoEXT*>(p + size_t(i) * stride);
    const size_t before = recordedOps(cb);
    vkCmdDrawIndexed(cb, d.indexCount, instances, d.firstIndex, vertexOffset ? *vertexOffset : d.vertexOffset,
                     firstInstance);
    setDrawIndexSince(cb, before, i);
  }
}

// ---- VK_EXT_private_data (core 1.3) -------------------------------------------------------------
// A slot is a map from object handle to value. Behind the CTS gate with the rest of the 1.3 profile.
namespace {
struct PrivateSlot {
  std::mutex m;
  std::unordered_map<uint64_t, uint64_t> values;  // (objectType << 56) ^ handle -> value
};
uint64_t privateKey(VkObjectType t, uint64_t h) { return (uint64_t(t) << 56) ^ h; }
}  // namespace
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreatePrivateDataSlot(VkDevice, const VkPrivateDataSlotCreateInfo*,
                                                                   const VkAllocationCallbacks*, VkPrivateDataSlot* out) {
  *out = reinterpret_cast<VkPrivateDataSlot>(new PrivateSlot);
  return VK_SUCCESS;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyPrivateDataSlot(VkDevice, VkPrivateDataSlot s, const VkAllocationCallbacks*) {
  delete reinterpret_cast<PrivateSlot*>(s);
}
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkSetPrivateData(VkDevice, VkObjectType t, uint64_t h, VkPrivateDataSlot s,
                                                            uint64_t v) {
  auto* p = reinterpret_cast<PrivateSlot*>(s);
  std::lock_guard<std::mutex> l(p->m);
  p->values[privateKey(t, h)] = v;
  return VK_SUCCESS;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPrivateData(VkDevice, VkObjectType t, uint64_t h, VkPrivateDataSlot s,
                                                        uint64_t* v) {
  auto* p = reinterpret_cast<PrivateSlot*>(s);
  std::lock_guard<std::mutex> l(p->m);
  auto it = p->values.find(privateKey(t, h));
  *v = it == p->values.end() ? 0 : it->second;
}

// ---- VK_KHR_buffer_device_address (core 1.2) ---------------------------------------------------------
// The address of a buffer is its backend's GPU address (Metal: MTLBuffer.gpuAddress), known as soon as the
// buffer is bound; shaders reach it through PhysicalStorageBuffer pointers (spirv lower.cpp, kTargetDevice).
// No capture / replay: opaque addresses are 0.
VPVK_EXPORT VKAPI_ATTR VkDeviceAddress VKAPI_CALL vkGetBufferDeviceAddress(VkDevice dv, const VkBufferDeviceAddressInfo* i) {
  Buffer* b = from<Buffer>(i->buffer);
  Device* d = from<Device>(dv);
  if (!b || !b->id) return 0;
  std::lock_guard<std::mutex> lock(d->submitMutex);  // the backend's buffer table is written under it (vkBindBufferMemory)
  return d->gpu->bufferAddress(b->id);
}
VPVK_EXPORT VKAPI_ATTR uint64_t VKAPI_CALL vkGetBufferOpaqueCaptureAddress(VkDevice, const VkBufferDeviceAddressInfo*) { return 0; }
VPVK_EXPORT VKAPI_ATTR uint64_t VKAPI_CALL vkGetDeviceMemoryOpaqueCaptureAddress(VkDevice,
                                                                              const VkDeviceMemoryOpaqueCaptureAddressInfo*) {
  return 0;
}

// ---- VK_KHR_maintenance4 (core 1.3) ----------------------------------------------------------------
// Requirements without an object: a temporary one asked the usual way.
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetDeviceBufferMemoryRequirements(VkDevice dv, const VkDeviceBufferMemoryRequirements* i,
                                                                           VkMemoryRequirements2* r) {
  VkBuffer b = VK_NULL_HANDLE;
  if (vkCreateBuffer(dv, i->pCreateInfo, nullptr, &b) != VK_SUCCESS || !b) {
    r->memoryRequirements = {};  // creación temporal fallida: requisitos a cero, sin tocar el handle
    return;
  }
  VkBufferMemoryRequirementsInfo2 q{VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2, nullptr, b};
  vkGetBufferMemoryRequirements2(dv, &q, r);
  vkDestroyBuffer(dv, b, nullptr);
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetDeviceImageMemoryRequirements(VkDevice dv, const VkDeviceImageMemoryRequirements* i,
                                                                          VkMemoryRequirements2* r) {
  VkImage im = VK_NULL_HANDLE;
  if (vkCreateImage(dv, i->pCreateInfo, nullptr, &im) != VK_SUCCESS || !im) {
    r->memoryRequirements = {};
    return;
  }
  VkImageMemoryRequirementsInfo2 q{VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, nullptr, im};
  vkGetImageMemoryRequirements2(dv, &q, r);
  vkDestroyImage(dv, im, nullptr);
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetDeviceImageSparseMemoryRequirements(VkDevice, const VkDeviceImageMemoryRequirements*,
                                                                                uint32_t* n, VkSparseImageMemoryRequirements2*) {
  *n = 0;
}

// ---- VK_KHR_sampler_ycbcr_conversion (core 1.1) ---------------------------------------------------
// The conversion is only parameters: image views of a multi-planar image keep one texture per plane
// (entry.cpp), the immutable sampler carries the conversion into the pipeline layout (pipelines.cpp
// assignSlots) and the shader reader samples the planes and converts (spirv lower.cpp ycbcrSample).
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateSamplerYcbcrConversion(VkDevice, const VkSamplerYcbcrConversionCreateInfo* ci,
                                                                          const VkAllocationCallbacks*,
                                                                          VkSamplerYcbcrConversion* out) {
  const MultiPlane& mp = multiPlane(ci->format);
  if (!mp.planes) {  // single-plane or external formats (VkExternalFormatANDROID) are not supported
    logf("vk.err what=ycbcr_format value=%d", int(ci->format));
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
  }
  auto o = std::make_shared<YcbcrConversionObj>();
  o->format = ci->format;
  spirv::YcbcrBinding& y = o->spec;
  y.planes = mp.planes;
  y.model = static_cast<spirv::YcbcrModel>(ci->ycbcrModel);  // same numbering as VkSamplerYcbcrModelConversion
  y.range = static_cast<spirv::YcbcrRange>(ci->ycbcrRange);
  y.cositedX = ci->xChromaOffset == VK_CHROMA_LOCATION_COSITED_EVEN;
  y.cositedY = ci->yChromaOffset == VK_CHROMA_LOCATION_COSITED_EVEN;
  y.subX = mp.subX, y.subY = mp.subY;
  const VkComponentSwizzle c[4] = {ci->components.r, ci->components.g, ci->components.b, ci->components.a};
  for (int i = 0; i < 4; ++i) y.swizzle[i] = uint8_t(c[i]);  // same numbering as VkComponentSwizzle
  o->chromaFilter = ci->chromaFilter;
  auto* h = new YcbcrConversionHandle;
  h->obj = std::move(o);
  *out = to<VkSamplerYcbcrConversion>(h);
  return VK_SUCCESS;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroySamplerYcbcrConversion(VkDevice, VkSamplerYcbcrConversion c,
                                                                       const VkAllocationCallbacks*) {
  delete from<YcbcrConversionHandle>(c);
}

// ---- Vulkan 1.1 / 1.3 core entry points a single-device, no-external-memory driver answers trivially ----
// Named by the conformance run (apiVersion 1.3, physical_device.cpp makeProps) and by loaders.
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceToolProperties(VkPhysicalDevice, uint32_t* n,
                                                                             VkPhysicalDeviceToolProperties*) {
  *n = 0;  // no layers or tools
  return VK_SUCCESS;
}
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumeratePhysicalDeviceGroups(VkInstance i, uint32_t* n,
                                                                           VkPhysicalDeviceGroupProperties* p) {
  if (!p) {
    *n = 1;
    return VK_SUCCESS;
  }
  if (*n == 0) return VK_INCOMPLETE;
  p[0].physicalDeviceCount = 1;
  p[0].physicalDevices[0] = to<VkPhysicalDevice>(&from<Instance>(i)->physical);
  p[0].subsetAllocation = VK_FALSE;
  *n = 1;
  return VK_SUCCESS;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetDeviceGroupPeerMemoryFeatures(VkDevice, uint32_t, uint32_t, uint32_t,
                                                                          VkPeerMemoryFeatureFlags* f) {
  *f = 0;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetDeviceMask(VkCommandBuffer, uint32_t) {}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDispatchBase(VkCommandBuffer cb, uint32_t bx, uint32_t by, uint32_t bz,
                                                         uint32_t gx, uint32_t gy, uint32_t gz) {
  if (bx || by || bz) {  // a workgroup-id offset the shader would have to add: not translated
    logf("vk.err what=dispatch_base base=%u,%u,%u", bx, by, bz);
    return;
  }
  vkCmdDispatch(cb, gx, gy, gz);
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceExternalBufferProperties(
    VkPhysicalDevice, const VkPhysicalDeviceExternalBufferInfo*, VkExternalBufferProperties* p) {
  p->externalMemoryProperties = {};  // no external handle types
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceExternalFenceProperties(
    VkPhysicalDevice, const VkPhysicalDeviceExternalFenceInfo*, VkExternalFenceProperties* p) {
  p->exportFromImportedHandleTypes = p->compatibleHandleTypes = 0, p->externalFenceFeatures = 0;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceExternalSemaphoreProperties(
    VkPhysicalDevice, const VkPhysicalDeviceExternalSemaphoreInfo*, VkExternalSemaphoreProperties* p) {
  p->exportFromImportedHandleTypes = p->compatibleHandleTypes = 0, p->externalSemaphoreFeatures = 0;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetImageSparseMemoryRequirements2(VkDevice, const VkImageSparseMemoryRequirementsInfo2*,
                                                                           uint32_t* n, VkSparseImageMemoryRequirements2*) {
  *n = 0;
}
VPVK_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceSparseImageFormatProperties2(
    VkPhysicalDevice, const VkPhysicalDeviceSparseImageFormatInfo2*, uint32_t* n, VkSparseImageFormatProperties2*) {
  *n = 0;
}

namespace vpvk {
PFN_vkVoidFunction lookupExt(const char* name) {
  struct E {
    const char* name;
    PFN_vkVoidFunction fn;
  };
#define F(f) {#f, reinterpret_cast<PFN_vkVoidFunction>(&f)}
#define K(f) {#f "KHR", reinterpret_cast<PFN_vkVoidFunction>(&f)}
#define E(f) {#f "EXT", reinterpret_cast<PFN_vkVoidFunction>(&f)}
  static const E table[] = {
      F(vkGetSemaphoreCounterValue), K(vkGetSemaphoreCounterValue), F(vkSignalSemaphore), K(vkSignalSemaphore),
      F(vkWaitSemaphores), K(vkWaitSemaphores),
      F(vkCmdCopyBuffer2), K(vkCmdCopyBuffer2), F(vkCmdCopyImage2), K(vkCmdCopyImage2),
      F(vkCmdCopyBufferToImage2), K(vkCmdCopyBufferToImage2), F(vkCmdCopyImageToBuffer2), K(vkCmdCopyImageToBuffer2),
      F(vkCmdBlitImage2), K(vkCmdBlitImage2), F(vkCmdResolveImage2), K(vkCmdResolveImage2),
      F(vkCmdPipelineBarrier2), K(vkCmdPipelineBarrier2), F(vkCmdSetEvent2), K(vkCmdSetEvent2),
      F(vkCmdResetEvent2), K(vkCmdResetEvent2), F(vkCmdWaitEvents2), K(vkCmdWaitEvents2),
      F(vkCmdWriteTimestamp2), K(vkCmdWriteTimestamp2), F(vkQueueSubmit2), K(vkQueueSubmit2),
      F(vkCmdDrawMultiEXT), F(vkCmdDrawMultiIndexedEXT),
      F(vkCreatePrivateDataSlot), E(vkCreatePrivateDataSlot), F(vkDestroyPrivateDataSlot), E(vkDestroyPrivateDataSlot),
      F(vkSetPrivateData), E(vkSetPrivateData), F(vkGetPrivateData), E(vkGetPrivateData),
      F(vkGetBufferDeviceAddress), K(vkGetBufferDeviceAddress), E(vkGetBufferDeviceAddress),
      F(vkGetBufferOpaqueCaptureAddress), K(vkGetBufferOpaqueCaptureAddress),
      F(vkGetDeviceMemoryOpaqueCaptureAddress), K(vkGetDeviceMemoryOpaqueCaptureAddress),
      F(vkGetDeviceBufferMemoryRequirements), K(vkGetDeviceBufferMemoryRequirements),
      F(vkGetDeviceImageMemoryRequirements), K(vkGetDeviceImageMemoryRequirements),
      F(vkGetDeviceImageSparseMemoryRequirements), K(vkGetDeviceImageSparseMemoryRequirements),
      F(vkGetPhysicalDeviceToolProperties), E(vkGetPhysicalDeviceToolProperties),
      F(vkEnumeratePhysicalDeviceGroups), K(vkEnumeratePhysicalDeviceGroups),
      F(vkGetDeviceGroupPeerMemoryFeatures), K(vkGetDeviceGroupPeerMemoryFeatures),
      F(vkCmdSetDeviceMask), K(vkCmdSetDeviceMask), F(vkCmdDispatchBase), K(vkCmdDispatchBase),
      F(vkGetPhysicalDeviceExternalBufferProperties), K(vkGetPhysicalDeviceExternalBufferProperties),
      F(vkGetPhysicalDeviceExternalFenceProperties), K(vkGetPhysicalDeviceExternalFenceProperties),
      F(vkGetPhysicalDeviceExternalSemaphoreProperties), K(vkGetPhysicalDeviceExternalSemaphoreProperties),
      F(vkGetImageSparseMemoryRequirements2), K(vkGetImageSparseMemoryRequirements2),
      F(vkGetPhysicalDeviceSparseImageFormatProperties2), K(vkGetPhysicalDeviceSparseImageFormatProperties2),
      F(vkCreateSamplerYcbcrConversion), K(vkCreateSamplerYcbcrConversion),
      F(vkDestroySamplerYcbcrConversion), K(vkDestroySamplerYcbcrConversion),
  };
#undef K
#undef E
#undef F
  for (const E& e : table)
    if (!std::strcmp(e.name, name)) return e.fn;
  return nullptr;
}
}  // namespace vpvk

extern "C" __attribute__((visibility("default"))) void vpvk_internal_scope(int on) { vpvk::tInternal = on ? 1 : 0; }
