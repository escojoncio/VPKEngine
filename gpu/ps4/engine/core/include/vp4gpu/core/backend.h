// What the core asks of a GPU API. gpu/metal implements this with Metal 4; NullBackend records
// the calls so tests can assert on them.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "vp4gpu/core/frame_compiler.h"

namespace vp4gpu {

struct PipelineState;

// Per-frame GPU timings reported back by the backend (from command buffer / encoder timestamps).
struct GpuTimings {
  double frameMs = 0;
  std::vector<double> encoderMs;  // parallel to CompiledFrame::encoders when available
};

class Backend {
 public:
  virtual ~Backend() = default;

  virtual void createBuffer(BufferId, const BufferDesc&) = 0;
  virtual void createTexture(TextureId, const TextureDesc&) = 0;
  virtual void createSampler(SamplerId, const SamplerDesc&) = 0;
  virtual void destroyBuffer(BufferId) = 0;
  virtual void destroyTexture(TextureId) = 0;
  virtual void destroySampler(SamplerId) = 0;

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
  bool compilePipeline(const PipelineState&, std::string& error) override;
  void destroyPipeline(PipelineKey) override;
  void submit(uint64_t frameIndex, const Frame&, const CompiledFrame&) override;
  bool takeTimings(uint64_t& frameIndex, GpuTimings& out) override;
  uint64_t completedFrames() const override { return completedBelow; }

  std::vector<std::string> calls;
  uint64_t completedBelow = 0;  // what completedFrames reports (tests move it: nothing finishes by itself)
  uint64_t failPipeline = 0;   // compilePipeline fails for this key value (0 = never)
  double fakeEncoderMs = 0.1;  // reported per encoder by takeTimings
  std::function<void()> onCompile;  // runs inside compilePipeline, on the compile worker (tests hold compiles)

 private:
  struct Pending {
    uint64_t frame;
    size_t encoders;
  };
  std::vector<Pending> pending_;
};

}  // namespace vp4gpu
