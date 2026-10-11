#include "vpgpu/core/device.h"

#include <chrono>
#include <cstdio>

namespace vpgpu {

Device::Device(Backend& backend, unsigned pipelineWorkers, LogSink log)
    : backend_(backend), pipelines_(backend, pipelineWorkers), log_(std::move(log)) {}

BufferId Device::createBuffer(BufferDesc d) {
  const BufferId id = resources_.createBuffer(d);
  if (id) backend_.createBuffer(id, *resources_.get(id));
  else log_("gpu.err what=buffer_desc label=" + d.label);
  return id;
}

TextureId Device::createTexture(TextureDesc d) {
  const TextureId id = resources_.createTexture(d);
  if (id) backend_.createTexture(id, *resources_.get(id));
  else log_("gpu.err what=texture_desc label=" + d.label);
  return id;
}

SamplerId Device::createSampler(SamplerDesc d) {
  const SamplerId id = resources_.createSampler(std::move(d));
  backend_.createSampler(id, *resources_.get(id));
  return id;
}

void Device::destroy(BufferId id) { if (resources_.destroy(id)) backend_.destroyBuffer(id); }
void Device::destroy(TextureId id) { if (resources_.destroy(id)) backend_.destroyTexture(id); }
void Device::destroy(SamplerId id) { if (resources_.destroy(id)) backend_.destroySampler(id); }

FrameReport Device::submit(const Frame& frame) {
  const uint64_t index = frame_++;
  const auto start = std::chrono::steady_clock::now();
  const CompiledFrame compiled = compileFrame(frame, &resources_);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  backend_.submit(index, frame, compiled);

  FrameReport report = makeReport(index, frame, compiled);
  report.cpuCompileMs = ms;
  report.pipelines = pipelines_.takeFrameStats();
  for (const std::string& line : formatReport(report)) log_(line);
  if (detailEvery_ && index % detailEvery_ == 0)
    for (const std::string& line : formatEncoders(index, frame, compiled)) log_(line);

  uint64_t done;
  GpuTimings t;
  while (backend_.takeTimings(done, t)) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "gpu.time f=%llu gpu_ms=%.3f", (unsigned long long)done, t.frameMs);
    log_(buf);
  }
  return report;
}

}  // namespace vpgpu
