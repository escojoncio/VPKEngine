#include "vpgpu/core/device.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <variant>
#include <vector>

namespace vpgpu {

Device::Device(Backend& backend, unsigned pipelineWorkers, LogSink log)
    : backend_(backend), pipelines_(backend, pipelineWorkers), log_(std::move(log)) {
  if (const char* e = std::getenv("VP_PASS_TIMING"); e && *e && *e != '0') {
    passEvery_ = std::strtoull(e, nullptr, 10) > 1 ? std::strtoull(e, nullptr, 10) : 90;
    log_("gpu.passtime armed every=" + std::to_string(passEvery_));
  }
  // gpu.enc detail (load/store/tile/reason per encoder): one frame in 600 (a few seconds) by default,
  // a handful of lines per sample. VP_ENC_DETAIL=N changes the period, 0 turns it off.
  detailEvery_ = 600;
  if (const char* e = std::getenv("VP_ENC_DETAIL"); e && *e) detailEvery_ = std::strtoull(e, nullptr, 10);
  // Copies to disjoint rectangles of one texture subresource without a barrier between them (off by default: see
  // CompileOptions::copyRects and the backend note about ordering inside a blit encoder).
  if (const char* e = std::getenv("VPVK_COPY_RECT_HAZARD"); e && *e && *e != '0') compile_.copyRects = true;
  // VPVK_MSAA_DEAD_STORE=1: drop client Stores of MSAA attachments nothing reads (opt-in, speculative: see
  // CompileOptions::msaaDeadStore). Anything else or unset = off.
  if (const char* e = std::getenv("VPVK_MSAA_DEAD_STORE"); e && e[0] == '1' && !e[1]) compile_.msaaDeadStore = true;
  if (const char* e = std::getenv("VP_HOLD_MS"); e && *e) holdMs_ = uint32_t(std::strtoul(e, nullptr, 10));
  log_("gpu.hold armed max_ms=" + std::to_string(holdMs_));
  // MSAA attachments that are stored and resolved drop the samples' store (resolve from tile) unless Attachment::keepSamples.
  // Colour is on by default (CompileOptions::msaaColorResolveOnTile). VP_MSAA_TILE: 0 / off / false = off; 1 / depth =
  // colour + depth (CompileOptions::msaaResolveOnTile); anything else = the default. Not VPVK_MSAA_TILE, which is the
  // Vulkan driver's memoryless knob.
  compile_.msaaColorResolveOnTile = true;
  if (const char* e = std::getenv("VP_MSAA_TILE"); e && *e) {
    const std::string v = e;
    if (v == "0" || v == "off" || v == "false") compile_.msaaColorResolveOnTile = false;
    else if (v == "1" || v == "depth") compile_.msaaResolveOnTile = true;
  }
  log_(std::string("gpu.msaa resolve_on_tile=") +
       (compile_.msaaResolveOnTile ? "1 (VP_MSAA_TILE=1: colour+depth)" : compile_.msaaColorResolveOnTile ? "colour" : "0"));
  // VPVK_TILE_MAX_DRAWS / VPVK_TILE_MAX_VERTICES: an encoder past either keeps its attachments in memory (HeavyGeometry).
  if (const char* e = std::getenv("VPVK_TILE_MAX_DRAWS"); e && *e) compile_.tileMaxDraws = uint32_t(std::strtoul(e, nullptr, 10));
  if (const char* e = std::getenv("VPVK_TILE_MAX_VERTICES"); e && *e)
    compile_.tileMaxVertices = std::strtoull(e, nullptr, 10);
  log_("gpu.tile heavy_max_draws=" + std::to_string(compile_.tileMaxDraws) +
       " heavy_max_vertices=" + std::to_string(compile_.tileMaxVertices));
  // VP_FRAME_DETAIL=N: same as VP_ENC_DETAIL (kept for AVP's scripts); wins when both are set.
  if (const char* e = std::getenv("VP_FRAME_DETAIL"); e && *e) {
    detailEvery_ = std::strtoull(e, nullptr, 10);
    if (detailEvery_) log_("gpu.detail armed every=" + std::to_string(detailEvery_));
  }
}

// The table is changed under resMutex_; the backend call (allocation, can be slow) and the log run outside it. A backend
// destroy that follows a reuse of the index is harmless: backends compare the generation.
BufferId Device::createBuffer(BufferDesc d) {
  BufferId id;
  BufferDesc copy;
  {
    std::lock_guard lock(resMutex_);
    id = resources_.createBuffer(d);
    if (id) copy = *resources_.get(id);
  }
  if (id) backend_.createBuffer(id, copy);
  else log_("gpu.err what=buffer_desc label=" + d.label);
  return id;
}

TextureId Device::createTexture(TextureDesc d) {
  TextureId id;
  TextureDesc copy;
  {
    std::lock_guard lock(resMutex_);
    id = resources_.createTexture(d);
    if (id) copy = *resources_.get(id);
  }
  if (id) backend_.createTexture(id, copy);
  else log_("gpu.err what=texture_desc label=" + d.label);
  return id;
}

SamplerId Device::createSampler(SamplerDesc d) {
  SamplerId id;
  SamplerDesc copy;
  {
    std::lock_guard lock(resMutex_);
    id = resources_.createSampler(std::move(d));
    copy = *resources_.get(id);
  }
  backend_.createSampler(id, copy);
  return id;
}

void Device::destroy(BufferId id) {
  bool gone;
  {
    std::lock_guard lock(resMutex_);
    gone = resources_.destroy(id);
  }
  if (gone) backend_.destroyBuffer(id);
}
void Device::destroy(TextureId id) {
  bool gone;
  {
    std::lock_guard lock(resMutex_);
    gone = resources_.destroy(id);
  }
  if (gone) backend_.destroyTexture(id);
}
void Device::destroy(SamplerId id) {
  bool gone;
  {
    std::lock_guard lock(resMutex_);
    gone = resources_.destroy(id);
  }
  if (gone) backend_.destroySampler(id);
}

FrameReport Device::submit(const Frame& frame) {
  const uint64_t index = frame_++;
  const auto start = std::chrono::steady_clock::now();
  std::unique_lock resLock(resMutex_);
  const CompiledFrame compiled = compileFrame(frame, &resources_, compile_);
  resLock.unlock();
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  if (holdMs_) {
    std::vector<PipelineKey> pending;
    auto note = [&](PipelineKey k) {
      if (!k.value) return;
      const auto st = pipelines_.status(k);
      if (st != PipelineCache::Status::Queued && st != PipelineCache::Status::Compiling) return;
      for (PipelineKey p : pending)
        if (p.value == k.value) return;
      pending.push_back(k);
    };
    for (const Command& c : frame.commands()) {
      if (const auto* rp = std::get_if<RenderPass>(&c))
        for (const Draw& d : rp->draws) note(d.pipeline);
      else if (const auto* dp = std::get_if<Dispatch>(&c))
        note(dp->pipeline);
    }
    if (!pending.empty()) {
      const auto h0 = std::chrono::steady_clock::now();
      const bool backoff = h0 < holdBackoffUntil_;
      const uint32_t budget = backoff ? std::min(holdMs_, kHoldBackoffMs) : holdMs_;
      const bool complete = pipelines_.waitReady(pending, std::chrono::milliseconds(budget));
      const double held = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h0).count();
      ++holdFrames_, holdTotalMs_ += held;
      if (!complete) {
        ++holdTimeouts_;
        holdBackoffUntil_ = std::chrono::steady_clock::now() + std::chrono::seconds(3);
      }
      if (holdFrames_ <= 16 || holdFrames_ % 64 == 0 || !complete) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "gpu.hold f=%llu pipelines=%zu ms=%.0f complete=%d total_frames=%llu total_ms=%.0f timeouts=%llu",
                      (unsigned long long)index, pending.size(), held, complete ? 1 : 0,
                      (unsigned long long)holdFrames_, holdTotalMs_, (unsigned long long)holdTimeouts_);
        log_(buf);
      }
    }
  }
  backend_.submit(index, frame, compiled);
  {
    std::lock_guard lock(resMutex_);  // memValid_ is part of the resource table: createTexture/destroy run under resMutex_
    commitMemory(compiled, resources_);
  }

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
  // One line (24 per run, then every 256th big frame) for a frame that costs hundreds of Metal encoders.
  if (bigFrame(compiled) && (bigLogged_ < 24 || index % 256 == 0)) {
    ++bigLogged_;
    std::string bigLine;
    {
      std::lock_guard lock(resMutex_);
      bigLine = formatBigFrame(index, frame, compiled, &resources_);
    }
    log_(bigLine);
  }
  for (const TextureId t : compiled.elideMisses) {  // banned right after: once per texture
    char buf[128];
    std::snprintf(buf, sizeof buf, "gpu.msaa elide_miss f=%llu tex=%u (elided store was needed; elision banned)",
                  (unsigned long long)index, t.index);
    log_(buf);
  }
  if (detailEvery_ && index % detailEvery_ == 0)
    for (const std::string& line : formatEncoders(index, frame, compiled)) log_(line);
  // MSAA attachments the resolve-on-tile option could not keep on tile: the first few of the run and then
  // one frame in every detailEvery_ (the first rejects fall in the menu; the scene's matter), with
  // everything the verdict was decided on (so a log says why, not just tile_rejected=N).
  const bool detailFrame = detailEvery_ && index % detailEvery_ == 0;
  for (const MsaaReject& r : compiled.msaaRejects) {
    if (msaaRejectLogs_ >= 120 || (msaaRejectLogs_ >= 12 && !detailFrame)) break;
    ++msaaRejectLogs_;
    const std::string slot = r.slot == kDepthSlot ? "d" : "c" + std::to_string(r.slot);
    char buf[320];
    std::snprintf(buf, sizeof buf,
                  "gpu.msaa rejected f=%llu enc=%u slot=%s why=%s tex=%u %ux%ux%u s%u mips=%u type=%u layer_count=%u layer=%u "
                  "persistent=%d transient=%d load=%d attached_in=%d first_read=%d last_use=%d multi=%d sampled=%d "
                  "elsewhere=%d bad_fetch=%d",
                  (unsigned long long)index, r.encoder, slot.c_str(),
                  toString(r.why), r.texture.index, r.width, r.height, r.layers, r.samples, r.mips, r.type, r.layerCount,
                  r.layer, r.persistent ? 1 : 0, r.transient ? 1 : 0, int(r.load), r.attachedIn, r.firstRead, r.lastUse,
                  r.multi ? 1 : 0, r.sampled ? 1 : 0, r.elsewhere ? 1 : 0, r.badFetch ? 1 : 0);
    log_(buf);
  }

  // Per-pass timing (diagnostic): describe the encoders of the first frame with render work after
  // each `passEvery_` boundary (VPVulkan alternates render frames with trivial compute submits).
  if (passEvery_ && index >= passNext_ && passPending_.size() < 8) {
    bool render = false;
    for (const Encoder& e : compiled.encoders) render |= e.kind == Encoder::Kind::Render;
    if (render) {
      passNext_ = index + passEvery_;
      std::lock_guard resGuard(resMutex_);  // resources_.get below
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
    {  // One cheap line per second: p50/p95/max of gpu_ms and frames over 10 ms (gpu.time is only >12 ms).
      const auto now = std::chrono::steady_clock::now();
      if (secMs_.empty()) secStart_ = now;
      secMs_.push_back(float(t.frameMs));
      if (now - secStart_ >= std::chrono::seconds(1)) {
        std::vector<float>& v = secMs_;
        const size_t n = v.size();
        unsigned over = 0;
        for (float x : v) over += x > 10.f;
        const float mx = *std::max_element(v.begin(), v.end());
        std::nth_element(v.begin(), v.begin() + n / 2, v.end());
        const float p50 = v[n / 2];
        const size_t i95 = std::min(n - 1, size_t(double(n) * 0.95));
        std::nth_element(v.begin(), v.begin() + i95, v.end());
        const float p95 = v[i95];
        char sb[128];
        std::snprintf(sb, sizeof sb, "gpu.sec f=%llu n=%zu p50=%.2f p95=%.2f max=%.2f over10=%u", (unsigned long long)done,
                      n, p50, p95, mx, over);
        log_(sb);
        v.clear();
      }
    }
    if (!(done < 300 || done % 90 == 0 || t.frameMs > 12.0)) continue;  // slow frames always logged
    char buf[64];
    std::snprintf(buf, sizeof buf, "gpu.time f=%llu gpu_ms=%.3f", (unsigned long long)done, t.frameMs);
    std::string line = buf;
    // Slow frames carry each encoder's GPU time (submission order) so the costly pass shows
    // without the "Diagnóstico de rendimiento" switch. Every 8th slow frame, to keep the log small.
    static uint64_t slowSeen = 0;
    if (t.frameMs > 12.0 && !t.encoderMs.empty() && (slowSeen++ % 8) == 0) {
      line += " enc_ms=";
      for (size_t i = 0; i < t.encoderMs.size() && i < 24; ++i) {
        char e[16];
        std::snprintf(e, sizeof e, i ? ",%.1f" : "%.1f", t.encoderMs[i]);
        line += e;
      }
    }
    log_(line);
  }
  return report;
}

}  // namespace vpgpu
