// Command buffer recording as a deferred list, and its translation into a gpu/core Frame.
// vkCmd* only append; nothing touches the GPU until vkQueueSubmit translates the list, so the core's
// frame compiler sees the whole submission and places barriers only where a real hazard exists.
// Vulkan pipeline barriers are counted (diagnostics) but never turned into GPU barriers.
#pragma once

#include <array>
#include <variant>
#include <vector>

#include "vpvk.h"

namespace vpvk {

struct OpBeginPass {
  const RenderPassObj* pass = nullptr;
  const Framebuffer* framebuffer = nullptr;  // null: dynamic rendering (views below)
  std::vector<VkClearValue> clears;
  VkRect2D area{};
};
struct OpEndPass {};
struct OpCopyBuffer {
  const Buffer* src = nullptr;
  const Buffer* dst = nullptr;
  VkBufferCopy region{};
};
struct OpFillBuffer {
  const Buffer* dst = nullptr;
  VkDeviceSize offset = 0, size = 0;
  uint32_t data = 0;
};
struct OpClearColor {
  const Image* image = nullptr;
  VkClearColorValue color{};
  VkImageSubresourceRange range{};
};
struct OpCopyBufferToImage {
  const Buffer* src = nullptr;
  const Image* dst = nullptr;
  VkBufferImageCopy region{};
};
// Recorded for counting until pipelines exist (phase 3).
struct OpDraw {
  uint32_t count = 0, instances = 1;
  bool indexed = false;
};
struct OpDispatch {
  uint32_t x = 1, y = 1, z = 1;
};

using Op = std::variant<OpBeginPass, OpEndPass, OpCopyBuffer, OpFillBuffer, OpClearColor,
                        OpCopyBufferToImage, OpDraw, OpDispatch>;

struct Recorder {
  std::vector<Op> ops;
  uint32_t barriers = 0;  // vkCmdPipelineBarrier calls dropped (the core derives its own)
  bool inPass = false;
  void reset() {
    ops.clear();
    barriers = 0;
    inPass = false;
  }
};

struct TranslateStats {
  uint32_t passes = 0, copies = 0, clears = 0, skippedDraws = 0, skippedDispatches = 0, dropped = 0;
};

// Appends the recorded work to `frame`. Ops whose resources are not bound yet are dropped
// (counted in `dropped`, a vk.err line is the caller's job).
void translate(const Recorder&, vpgpu::Frame& frame, TranslateStats& stats);

// Exposed for tests.
vpgpu::LoadAction toLoad(VkAttachmentLoadOp);
vpgpu::StoreAction toStore(VkAttachmentStoreOp, bool resolve);

}  // namespace vpvk
