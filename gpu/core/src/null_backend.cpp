#include <format>

#include "vpgpu/core/backend.h"
#include "vpgpu/core/pipeline_cache.h"

namespace vpgpu {

void NullBackend::createBuffer(BufferId id, const BufferDesc& d) {
  calls.push_back(std::format("createBuffer {} size={} guest={}", id.index, d.size, d.guestMemory != nullptr));
}
void NullBackend::createTexture(TextureId id, const TextureDesc& d) {
  calls.push_back(std::format("createTexture {} {} {}x{}", id.index, formatInfo(d.format).name, d.width, d.height));
}
void NullBackend::createSampler(SamplerId id, const SamplerDesc&) { calls.push_back(std::format("createSampler {}", id.index)); }
void NullBackend::destroyBuffer(BufferId id) { calls.push_back(std::format("destroyBuffer {}", id.index)); }
void NullBackend::destroyTexture(TextureId id) { calls.push_back(std::format("destroyTexture {}", id.index)); }
void NullBackend::destroySampler(SamplerId id) { calls.push_back(std::format("destroySampler {}", id.index)); }

bool NullBackend::compilePipeline(const PipelineState& s, std::string& error) {
  if (s.key().value == failPipeline) {
    error = "forced failure";
    return false;
  }
  return true;
}

void NullBackend::submit(uint64_t frameIndex, const Frame&, const CompiledFrame& c) {
  calls.push_back(std::format("submit {} encoders={} barriers={}", frameIndex, c.encoders.size(), c.barriers.size()));
  pending_.push_back({frameIndex, c.encoders.size()});
}

bool NullBackend::takeTimings(uint64_t& frameIndex, GpuTimings& out) {
  if (pending_.empty()) return false;
  const Pending p = pending_.front();
  pending_.erase(pending_.begin());
  frameIndex = p.frame;
  out.encoderMs.assign(p.encoders, fakeEncoderMs);
  out.frameMs = fakeEncoderMs * static_cast<double>(p.encoders);
  return true;
}

}  // namespace vpgpu
