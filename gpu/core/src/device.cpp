#include "vpgpu/core/device.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <variant>

namespace vpgpu {

Device::Device(Backend& backend, unsigned pipelineWorkers, LogSink log)
    : backend_(backend), pipelines_(backend, pipelineWorkers), log_(std::move(log)) {
  if (const char* e = std::getenv("VP_PASS_TIMING"); e && *e && *e != '0') {
    passEvery_ = std::strtoull(e, nullptr, 10) > 1 ? std::strtoull(e, nullptr, 10) : 90;
    log_("gpu.passtime armed every=" + std::to_string(passEvery_));
  }
}

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
  // Per-frame summaries: every frame for the first 300, then 1 in 90 (or when pipelines were
  // compiled) — three log lines per frame on the render thread was itself a stutter source.
  const bool verbose = index < 300 || index % 90 == 0 || report.pipelines.compiled > 0;
  if (verbose)
    for (const std::string& line : formatReport(report)) log_(line);
  if (!compiled.badDraws.empty()) {  // client bug: those draws do not run (see CompiledFrame::badDraws)
    const DrawRef& d = compiled.badDraws.front();
    char buf[128];
    std::snprintf(buf, sizeof buf, "gpu.err what=bad_fetch f=%llu draws=%zu reads=%u first=c%u.d%u",
                  (unsigned long long)index, compiled.badDraws.size(), compiled.stats.badFetch, d.command, d.draw);
    log_(buf);
  }
  if (detailEvery_ && index % detailEvery_ == 0)
    for (const std::string& line : formatEncoders(index, frame, compiled)) log_(line);

  // Per-pass timing (diagnostic): describe the encoders of the first frame with render work after
  // each `passEvery_` boundary (VPVulkan alternates render frames with trivial compute submits).
  if (passEvery_ && index >= passNext_ && passPending_.size() < 8) {
    bool render = false;
    for (const Encoder& e : compiled.encoders) render |= e.kind == Encoder::Kind::Render;
    if (render) {
      passNext_ = index + passEvery_;
      std::vector<std::string> desc;
      for (size_t i = 0; i < compiled.encoders.size(); ++i) {
        const Encoder& e = compiled.encoders[i];
        char buf[256];
        if (e.kind == Encoder::Kind::Render) {
          const Attachment& a = e.pass.colors[0].texture ? e.pass.colors[0] : e.pass.depth;
          const TextureDesc* td = a.texture ? resources_.get(a.texture) : nullptr;
          uint32_t draws = 0;
          std::string label;
          for (uint32_t c : e.commands)
            if (const auto* rp = std::get_if<RenderPass>(&frame.commands()[c])) {
              draws += uint32_t(rp->draws.size());
              if (label.empty()) label = rp->desc.label;
            }
          for (char& ch : label)
            if (ch == ' ') ch = '_';
          std::snprintf(buf, sizeof buf, "pass=%zu kind=render label=%s fmt=%s %ux%ux%u samples=%u draws=%u merged=%u",
                        i, label.empty() ? "-" : label.substr(0, 40).c_str(),
                        td ? formatInfo(td->format).name : "-", td ? td->width >> a.mip : 0u,
                        td ? td->height >> a.mip : 0u, unsigned(e.pass.layerCount), td ? unsigned(td->samples) : 0u,
                        draws, e.passes);
        } else {
          uint32_t dispatches = 0, copies = 0;
          for (uint32_t c : e.commands)
            (std::holds_alternative<Dispatch>(frame.commands()[c]) ? dispatches : copies)++;
          std::snprintf(buf, sizeof buf, "pass=%zu kind=compute label=- fmt=- 0x0x0 samples=0 draws=%u copies=%u", i,
                        dispatches, copies);
        }
        desc.emplace_back(buf);
      }
      passPending_.emplace_back(index, std::move(desc));
    }
  }

  uint64_t done;
  GpuTimings t;
  while (backend_.takeTimings(done, t)) {
    while (!passPending_.empty() && passPending_.front().first < done) passPending_.pop_front();
    if (!passPending_.empty() && passPending_.front().first == done) {
      const auto& desc = passPending_.front().second;
      double sum = 0;
      for (size_t i = 0; i < desc.size(); ++i) {
        const double ms = i < t.encoderMs.size() ? t.encoderMs[i] : -1.0;
        if (ms > 0) sum += ms;
        char buf[48];
        std::snprintf(buf, sizeof buf, " ms=%.3f", ms);
        log_("gpu.passtime f=" + std::to_string(done) + " " + desc[i] + buf);
      }
      char buf[128];
      std::snprintf(buf, sizeof buf, "gpu.passtime f=%llu total gpu_ms=%.3f passes_ms=%.3f stamps=%zu",
                    (unsigned long long)done, t.frameMs, sum, t.encoderMs.size());
      log_(buf);
      passPending_.pop_front();
    }
    if (!(done < 300 || done % 90 == 0 || t.frameMs > 12.0)) continue;  // slow frames always logged
    char buf[64];
    std::snprintf(buf, sizeof buf, "gpu.time f=%llu gpu_ms=%.3f", (unsigned long long)done, t.frameMs);
    log_(buf);
  }
  return report;
}

}  // namespace vpgpu
