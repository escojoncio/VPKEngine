// What the core asks of a GPU API. gpu/metal implements this with Metal 4; NullBackend records
// the calls so tests can assert on them.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include <utility>

#include "vpgpu/core/frame_compiler.h"
#include "vpgpu/core/pipeline_cache.h"  // PipelineState complete: NullBackend keeps a vector of them

namespace vpgpu {

// Per-frame GPU timings reported back by the backend (from command buffer / encoder timestamps).
struct GpuTimings {
  double frameMs = 0;
  std::vector<double> encoderMs;  // parallel to CompiledFrame::encoders when available
};

// Backend::readTexture outcome. Memoryless: the texture has no memory (tile only, or never allocated): nothing is read and
// nothing is allocated for it.
enum class ReadResult : uint8_t { Ok, Unsupported, Memoryless, Failed };

class Backend {
 public:
  virtual ~Backend() = default;

  virtual void createBuffer(BufferId, const BufferDesc&) = 0;
  virtual void createTexture(TextureId, const TextureDesc&) = 0;
  virtual void createSampler(SamplerId, const SamplerDesc&) = 0;
  virtual void destroyBuffer(BufferId) = 0;
  virtual void destroyTexture(TextureId) = 0;
  virtual void destroySampler(SamplerId) = 0;
  // Whether the createBuffer just made for this id left a usable buffer (a backend that can refuse an allocation, e.g.
  // a no-copy wrapper of guest memory, says false; Device::createBuffer then returns an invalid id).
  virtual bool bufferCreated(BufferId) const { return true; }

  // GPU virtual address of the first byte of a buffer's memory (VK_KHR_buffer_device_address): stable
  // for the buffer's lifetime and readable by shaders through a raw pointer. 0 when the backend has
  // none (or the buffer is unknown).
  virtual uint64_t bufferAddress(BufferId) const { return 0; }

  // Builds the native pipeline object; called from pipeline-cache worker threads.
  // Returns false on failure (the error goes to `error`).
  virtual bool compilePipeline(const PipelineState&, std::string& error) = 0;
  // The native pipeline of `key` is no longer wanted (PipelineCache::release); frames already
  // submitted may still run it. A later compilePipeline of the same key rebuilds it. Any thread.
  virtual void destroyPipeline(PipelineKey) {}

  // Executes one compiled frame. The frame's commands are passed for the payloads (draws,
  // dispatch sizes, copy regions); `compiled` says how to group and order them.
  virtual void submit(uint64_t frameIndex, const Frame&, const CompiledFrame& compiled) = 0;

  // Timings of an earlier frame once the GPU finished it; false if none are ready yet.
  virtual bool takeTimings(uint64_t& frameIndex, GpuTimings& out) = 0;

  // Every frame with an index below this one has finished on the GPU (completion handlers ran).
  // Backends that execute nothing report everything finished.
  virtual uint64_t completedFrames() const { return ~uint64_t(0); }

  // Host copy of one subresource of a texture: span.texture / mip / layer (layerCount 1) / aspect, with width x height x
  // depth the texels of that mip level (the caller knows them). `out` is the subresource tightly packed: slices of rows
  // of blocks (a compressed format is copied block for block, as stored), one aspect of a depth/stencil texture
  // (Aspect::All = colour or the only aspect). The GPU must be idle (the caller waited): this is a debugging read (VPCAP
  // image contents), not a frame operation. Unsupported by default.
  virtual ReadResult readTexture(const TextureSpan&, std::vector<uint8_t>& /*out*/) { return ReadResult::Unsupported; }

  // Blocks until every frame with an index below `frames` has finished on the GPU (a partial
  // fence: later frames keep running), or `timeoutNs` passes. False on timeout or a lost device.
  // Backends that execute nothing are always done.
  virtual bool waitFramesBelow(uint64_t frames, uint64_t timeoutNs) {
    (void)frames, (void)timeoutNs;
    return true;
  }
  // Time submit() spent waiting for a free frame slot (ms) since the last call; resets. Any thread.
  // Backends without frame slots report 0.
  virtual double takeSlotWaitMs() { return 0; }

  // Blocks until every submitted frame has finished (the full drain).
  virtual bool waitAll(uint64_t timeoutNs) {
    (void)timeoutNs;
    return true;
  }

  // The device was lost: no more GPU work runs, completedFrames() never advances again and waits
  // fail. Device then releases every GPU-ordered label (nothing else would ever write them).
  // Any thread.
  virtual bool lost() const { return false; }

  // Called (on the backend's completion thread, any thread) each time a frame finished and
  // completedFrames() may have advanced. Device uses it to release GPU-ordered label writes.
  // Set it before the first submit; clearing waits for a callback that is running.
  void setFrameDoneNotify(std::function<void()> f) {
    std::lock_guard<std::mutex> lock(notifyMutex_);
    frameDone_ = std::move(f);
  }

 protected:
  // For backends: a frame finished (completedFrames() already counts it).
  void notifyFrameDone() {
    std::lock_guard<std::mutex> lock(notifyMutex_);
    if (frameDone_) frameDone_();
  }

 private:
  std::mutex notifyMutex_;
  std::function<void()> frameDone_;
};

// Records every call as a text line; executes nothing.
class NullBackend final : public Backend {
 public:
  void createBuffer(BufferId, const BufferDesc&) override;
  void createTexture(TextureId, const TextureDesc&) override;
  void createSampler(SamplerId, const SamplerDesc&) override;
  void destroyBuffer(BufferId) override;
  void destroyTexture(TextureId) override;
  void destroySampler(SamplerId) override;
  // Fake but stable and distinct: buffer index + 1 in the high word (tests of the address plumbing).
  uint64_t bufferAddress(BufferId id) const override { return id ? (uint64_t(id.index) + 1) << 32 : 0; }
  bool bufferCreated(BufferId id) const override { return !failedBuffers.count({id.index, id.generation}); }
  bool compilePipeline(const PipelineState&, std::string& error) override;
  void destroyPipeline(PipelineKey) override;
  void submit(uint64_t frameIndex, const Frame&, const CompiledFrame&) override;
  bool takeTimings(uint64_t& frameIndex, GpuTimings& out) override;
  uint64_t completedFrames() const override { return completedBelow.load(); }
  // Returns what a test stored in `textureData` for the subresource (nothing stored: empty, i.e. never written).
  ReadResult readTexture(const TextureSpan&, std::vector<uint8_t>& out) override;
  using DataKey = std::tuple<uint32_t, uint32_t, uint16_t, uint16_t, uint8_t>;  // texture index, generation, mip, layer, aspect
  static DataKey dataKey(TextureId t, uint16_t mip, uint16_t layer, Aspect a = Aspect::All) {
    return {t.index, t.generation, mip, layer, static_cast<uint8_t>(a)};
  }
  std::map<DataKey, std::vector<uint8_t>> textureData;
  // A partial wait: records the target and lets "the GPU" finish up to it (tests then see which
  // wait was asked for). waitAll finishes everything submitted so far.
  bool waitFramesBelow(uint64_t frames, uint64_t timeoutNs) override;
  bool waitAll(uint64_t timeoutNs) override;
  // Tests: the device is lost (waits fail at once, nothing finishes any more).
  bool lost() const override { return isLost.load(); }
  std::atomic<bool> isLost{false};
  // Tests: a frame finished by itself (moves completedBelow, runs the Device's notification).
  void finishFramesBelow(uint64_t frames);
  std::vector<uint64_t> waits;  // targets of waitFramesBelow; waitAll records ~0
  uint64_t submittedFrames = 0;

  std::vector<std::string> calls;
  std::vector<PipelineState> pipelines;  // every state compilePipeline saw (tests)
  std::mutex pipelinesMutex;             // compilePipeline runs on several compile workers
  std::vector<Draw> draws;               // every draw submitted, in render passes (tests)
  std::vector<RenderPassDesc> passes;    // the render passes submitted (tests)
  std::atomic<uint64_t> completedBelow{0};  // what completedFrames reports (tests move it: nothing finishes by itself)
  uint64_t failPipeline = 0;   // compilePipeline fails for this key value (0 = never)
  std::function<bool(const BufferDesc&)> failBuffer;  // createBuffer refuses the buffers it says true for (tests)
  std::set<std::pair<uint32_t, uint32_t>> failedBuffers;  // (index, generation) of the refused ones
  double fakeEncoderMs = 0.1;  // reported per encoder by takeTimings
  std::function<void()> onCreate;   // runs first in createBuffer, before it records anything (tests make a create slow)
  std::function<void()> onCompile;  // runs inside compilePipeline, on the compile worker (tests hold compiles)

 private:
  struct Pending {
    uint64_t frame;
    size_t encoders;
  };
  std::vector<Pending> pending_;
};

}  // namespace vpgpu
