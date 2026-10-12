// The client-facing entry point: owns the resource table, forwards to the backend, compiles and
// submits frames, and produces the per-frame log lines.
#pragma once

#include <functional>
#include <string>

#include "vpgpu/core/backend.h"
#include "vpgpu/core/diagnostics.h"
#include "vpgpu/core/pipeline_cache.h"

namespace vpgpu {

class Device {
 public:
  using LogSink = std::function<void(const std::string&)>;

  Device(Backend&, unsigned pipelineWorkers, LogSink log);

  BufferId createBuffer(BufferDesc);
  TextureId createTexture(TextureDesc);
  SamplerId createSampler(SamplerDesc);
  void destroy(BufferId);
  void destroy(TextureId);
  void destroy(SamplerId);
  const ResourceTable& resources() const { return resources_; }
  PipelineCache& pipelines() { return pipelines_; }

  // Compiles and submits a frame and logs its summary; every `detailEvery`-th frame (0 = never)
  // also logs one line per encoder. GPU timings of finished frames are logged as they arrive.
  FrameReport submit(const Frame&);
  void setDetailEvery(uint64_t n) { detailEvery_ = n; }
  // Frames with an index below this one have finished on the GPU (see Backend::completedFrames).
  uint64_t completedFrames() const { return backend_.completedFrames(); }

 private:
  Backend& backend_;
  ResourceTable resources_;
  PipelineCache pipelines_;
  LogSink log_;
  uint64_t frame_ = 0, detailEvery_ = 0;
};

}  // namespace vpgpu
