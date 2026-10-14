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
#include <thread>
#include <vector>

#include "commands.h"
#include "vpvk.h"

#define VPVK_EXPORT extern "C" __attribute__((visibility("default")))

namespace vpvk {

bool extraExtensions() {
  const char* e = std::getenv("VPVK_EXTRA_EXTENSIONS");
  return e && *e == '1';
}

namespace {
const void* nextOf(const void* chain, VkStructureType t) {
  for (auto* s = static_cast<const VkBaseInStructure*>(chain); s; s = s->pNext)
    if (s->sType == t) return s;
  return nullptr;
}

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

void timelineSignals(const VkSubmitInfo& si, uint64_t frames) {
  const auto* t = static_cast<const VkTimelineSemaphoreSubmitInfo*>(
      nextOf(si.pNext, VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO));
  if (!t || !t->pSignalSemaphoreValues) return;
  for (uint32_t j = 0; j < si.signalSemaphoreCount && j < t->signalSemaphoreValueCount; ++j) {
    Semaphore* s = from<Semaphore>(si.pSignalSemaphores[j]);
    if (!s || !s->timeline) continue;
    std::lock_guard<std::mutex> l(s->m);
    s->pending.push_back({t->pSignalSemaphoreValues[j], frames});
  }
}

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

VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkSignalSemaphore(VkDevice, const VkSemaphoreSignalInfo* info) {
  Semaphore* s = from<Semaphore>(info->semaphore);
  std::lock_guard<std::mutex> l(s->m);
  s->completed = std::max(s->completed, info->value);
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
  ++from<CommandBuffer>(cb)->rec->barriers;
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

// vkQueueSubmit2 as vkQueueSubmit: command buffers in order, timeline signal values carried over.
VPVK_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit2(VkQueue q, uint32_t n, const VkSubmitInfo2* si, VkFence fence) {
  struct One {
    std::vector<VkCommandBuffer> cbs;
    std::vector<VkSemaphore> signals;
    std::vector<uint64_t> values;
    VkTimelineSemaphoreSubmitInfo tl{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  };
  std::vector<One> ones(n);
  std::vector<VkSubmitInfo> v1(n);
  for (uint32_t i = 0; i < n; ++i) {
    One& o = ones[i];
    for (uint32_t k = 0; k < si[i].commandBufferInfoCount; ++k) o.cbs.push_back(si[i].pCommandBufferInfos[k].commandBuffer);
    for (uint32_t k = 0; k < si[i].signalSemaphoreInfoCount; ++k) {
      o.signals.push_back(si[i].pSignalSemaphoreInfos[k].semaphore);
      o.values.push_back(si[i].pSignalSemaphoreInfos[k].value);
    }
    o.tl.signalSemaphoreValueCount = uint32_t(o.values.size());
    o.tl.pSignalSemaphoreValues = o.values.data();
    VkSubmitInfo& s = v1[i];
    s = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    s.pNext = &o.tl;
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
    vkCmdDraw(cb, d.vertexCount, instances, d.firstVertex, firstInstance);
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
    vkCmdDrawIndexed(cb, d.indexCount, instances, d.firstIndex, vertexOffset ? *vertexOffset : d.vertexOffset,
                     firstInstance);
  }
}

namespace vpvk {
PFN_vkVoidFunction lookupExt(const char* name) {
  struct E {
    const char* name;
    PFN_vkVoidFunction fn;
  };
#define F(f) {#f, reinterpret_cast<PFN_vkVoidFunction>(&f)}
#define K(f) {#f "KHR", reinterpret_cast<PFN_vkVoidFunction>(&f)}
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
  };
#undef K
#undef F
  for (const E& e : table)
    if (!std::strcmp(e.name, name)) return e.fn;
  return nullptr;
}
}  // namespace vpvk
