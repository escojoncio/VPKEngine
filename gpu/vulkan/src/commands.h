// Command buffer recording as a deferred list, and its translation into a gpu/core Frame.
// vkCmd* only append; nothing touches the GPU until vkQueueSubmit translates the list, so the core's
// frame compiler sees the whole submission and places barriers only where a real hazard exists.
// Vulkan pipeline barriers are counted (diagnostics) but never turned into GPU barriers.
//
// Draws and dispatches are resolved when they are recorded: the bound pipeline, descriptor sets,
// push constants and vertex/index buffers become a ready vpgpu::Draw / Dispatch (Vulkan forbids
// changing what a recorded command buffer references until it has executed).
#pragma once

#include <array>
#include <variant>
#include <vector>

#include "vpvk.h"

namespace vpvk {

struct OpBeginPass {
  const RenderPassObj* pass = nullptr;
  const Framebuffer* framebuffer = nullptr;
  std::vector<VkClearValue> clears;
  VkRect2D area{};
};
// Dynamic rendering: the core pass is built when recorded.
struct OpBeginRendering {
  vpgpu::RenderPassDesc desc;
};
struct OpNextSubpass {};
struct OpEndPass {};
struct OpCopyBuffer {
  const Buffer* src = nullptr;
  const Buffer* dst = nullptr;
  VkBufferCopy region{};
};
// A copy from the command buffer's own upload memory (vkCmdUpdateBuffer, pattern fills).
struct OpUpload {
  vpgpu::BufferId src;
  VkDeviceSize srcOffset = 0;
  const Buffer* dst = nullptr;
  VkDeviceSize dstOffset = 0, size = 0;
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
struct OpClearDepth {
  const Image* image = nullptr;
  VkClearDepthStencilValue value{};
  VkImageSubresourceRange range{};
};
// vkCmdClearAttachments inside a pass: the pass is split around clear passes.
struct OpClearAttachments {
  std::vector<VkClearAttachment> attachments;
};
struct OpCopyBufferToImage {
  const Buffer* src = nullptr;
  const Image* dst = nullptr;
  VkBufferImageCopy region{};
};
struct OpCopyImageToBuffer {
  const Image* src = nullptr;
  const Buffer* dst = nullptr;
  VkBufferImageCopy region{};
};
struct OpCopyImage {
  const Image* src = nullptr;
  const Image* dst = nullptr;
  VkImageCopy region{};
  bool genMips = false;  // a scaled blit chain over one image's mips: the backend builds the chain
};
struct OpDraw {
  vpgpu::Draw draw;
  // Indirect draws: arguments read from the buffer at submit (unified memory).
  const Buffer* indirect = nullptr;
  VkDeviceSize indirectOffset = 0;
  uint32_t indirectCount = 0, indirectStride = 0;
  uint32_t viewCount = 1;
  bool ok = true;  // false: the pipeline or a binding was missing (counted as skipped)
};
struct OpDispatch {
  vpgpu::Dispatch dispatch;
  const Buffer* indirect = nullptr;
  VkDeviceSize indirectOffset = 0;
  bool ok = true;
};
struct Recorder;
struct OpExecute {
  const Recorder* secondary = nullptr;
};

using Op = std::variant<OpBeginPass, OpBeginRendering, OpNextSubpass, OpEndPass, OpCopyBuffer, OpUpload,
                        OpFillBuffer, OpClearColor, OpClearDepth, OpClearAttachments, OpCopyBufferToImage,
                        OpCopyImageToBuffer, OpCopyImage, OpDraw, OpDispatch, OpExecute>;

// Host memory owned by a command buffer for the data it carries (push constants, vkCmdUpdateBuffer):
// 64 KiB chunks aliased by core buffers, reused after a reset.
struct UploadArena {
  struct Chunk {
    uint8_t* host = nullptr;
    vpgpu::BufferId id;
    VkDeviceSize size = 0, used = 0;
  };
  std::vector<Chunk> chunks;
  size_t current = 0;
  // Copies `size` bytes, 256-aligned (constant buffers need 16). Returns the chunk's buffer.
  vpgpu::BufferId put(Device* d, const void* data, VkDeviceSize size, VkDeviceSize& offset);
  void reset() {
    for (auto& c : chunks) c.used = 0;
    current = 0;
  }
  void release(Device* d);
};

// Draw state at the time of a vkCmd*.
struct BoundState {
  const PipelineObj* graphics = nullptr;
  const PipelineObj* compute = nullptr;
  struct Set {
    const DescriptorSetObj* set = nullptr;
    std::vector<uint32_t> dynamicOffsets;
  };
  std::array<Set, 8> graphicsSets{}, computeSets{};
  std::array<uint8_t, 256> push{};
  struct VB {
    const Buffer* buffer = nullptr;
    VkDeviceSize offset = 0;
  };
  std::array<VB, 16> vertex{};
  const Buffer* index = nullptr;
  VkDeviceSize indexOffset = 0;
  bool indexWide = false;
  bool hasViewport = false, hasScissor = false;
  VkViewport viewport{};
  VkRect2D scissor{};
  uint32_t stencilRef = 0;
  float blend[4] = {0, 0, 0, 0};
  float depthBias[3] = {0, 0, 0};  // constant, clamp, slope (vkCmdSetDepthBias order)
  uint32_t targetWidth = 0, targetHeight = 0;  // render area of the open pass (scissor clamp)
  uint32_t subpass = 0;
  const RenderPassObj* pass = nullptr;
};

struct Recorder {
  Device* device = nullptr;
  std::vector<Op> ops;
  uint32_t barriers = 0;  // vkCmdPipelineBarrier calls dropped (the core derives its own)
  bool inPass = false;
  BoundState bound;
  UploadArena arena;
  void reset() {
    ops.clear();
    barriers = 0;
    inPass = false;
    bound = BoundState{};
    arena.reset();
  }
  ~Recorder();
};

struct TranslateStats {
  uint32_t passes = 0, copies = 0, clears = 0, draws = 0, dispatches = 0, skippedDraws = 0,
           skippedDispatches = 0, dropped = 0;
};

// Appends the recorded work to `frame`. Ops whose resources are not bound yet are dropped
// (counted in `dropped`, a vk.err line is the caller's job).
void translate(const Recorder&, vpgpu::Frame& frame, TranslateStats& stats);

// Records a draw/dispatch with the current state (pipelines.cpp).
OpDraw makeDraw(Recorder&, uint32_t first, uint32_t count, uint32_t instances, uint32_t firstInstance,
                int32_t baseVertex, bool indexed);
OpDispatch makeDispatch(Recorder&, uint32_t x, uint32_t y, uint32_t z);

// Exposed for tests.
vpgpu::LoadAction toLoad(VkAttachmentLoadOp);
vpgpu::StoreAction toStore(VkAttachmentStoreOp, bool resolve);

}  // namespace vpvk
