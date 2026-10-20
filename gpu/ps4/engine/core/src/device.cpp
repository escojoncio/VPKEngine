#include "vp4gpu/core/device.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <variant>
#include <vector>

namespace vp4gpu {

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
  if (const char* e = std::getenv("VP_HOLD_MS"); e && *e) holdMs_ = uint32_t(std::strtoul(e, nullptr, 10));
  log_("gpu.hold armed max_ms=" + std::to_string(holdMs_));
  backend_.setFrameDoneNotify([this] { pumpLabels(true); });
}

Device::~Device() {
  {
    std::unique_lock lock(qMutex_);
    qIdle_.wait(lock, [&] { return tasks_.empty() && !busy_; });
    stop_ = true;
  }
  qWake_.notify_all();
  if (worker_.joinable()) worker_.join();
  backend_.waitAll(UINT64_MAX);          // completion handlers may still call pumpLabels
  backend_.setFrameDoneNotify({});       // (waits for a callback that is running)
  labels_.release(~uint64_t(0));         // the GPU is idle: whatever was queued has happened
}

void Device::settleCreate(char kind, uint32_t index) {
  if (!maxQueued_) return;
  bool wait;
  {
    std::lock_guard lock(qMutex_);
    wait = pendingDestroy_.count(destroyKey(kind, index)) != 0;
  }
  if (wait) drain();  // the backend still holds the previous resource of this slot
}

BufferId Device::createBuffer(BufferDesc d) {
  const BufferId id = resources_.createBuffer(d);
  if (id) {
    settleCreate('b', id.index);
    std::lock_guard lock(backendMutex_);
    backend_.createBuffer(id, *resources_.get(id));
  } else {
    log_("gpu.err what=buffer_desc label=" + d.label);
  }
  return id;
}

TextureId Device::createTexture(TextureDesc d) {
  const TextureId id = resources_.createTexture(d);
  if (id) {
    settleCreate('t', id.index);
    std::lock_guard lock(backendMutex_);
    backend_.createTexture(id, *resources_.get(id));
  } else {
    log_("gpu.err what=texture_desc label=" + d.label);
  }
  return id;
}

SamplerId Device::createSampler(SamplerDesc d) {
  const SamplerId id = resources_.createSampler(std::move(d));
  settleCreate('s', id.index);
  std::lock_guard lock(backendMutex_);
  backend_.createSampler(id, *resources_.get(id));
  return id;
}

void Device::destroy(BufferId id) {
  if (resources_.destroy(id)) runOrdered([this, id] { backend_.destroyBuffer(id); }, destroyKey('b', id.index));
}
void Device::destroy(TextureId id) {
  if (resources_.destroy(id)) runOrdered([this, id] { backend_.destroyTexture(id); }, destroyKey('t', id.index));
}
void Device::destroy(SamplerId id) {
  if (resources_.destroy(id)) runOrdered([this, id] { backend_.destroySampler(id); }, destroyKey('s', id.index));
}

bool Device::waitFrames(uint64_t frames, uint64_t timeoutNs) {
  if (frames > frame_) frames = frame_;
  if (lostNoted_.load()) return false;  // nothing finishes any more: waiting would only spin
  if (maxQueued_) {  // the frames below `frames` must have reached the backend before it can fence on them
    // (not the whole queue: frames queued after them keep the submit thread busy)
    std::unique_lock lock(qMutex_);
    const auto ready = [&] { return stop_ || handed_ >= frames || (tasks_.empty() && !busy_); };
    if (timeoutNs == UINT64_MAX) qIdle_.wait(lock, ready);
    else if (!qIdle_.wait_for(lock, std::chrono::nanoseconds(timeoutNs), ready)) return false;
  }
  const bool done = backend_.waitFramesBelow(frames, timeoutNs);
  if (!done && backend_.lost()) noteLost();
  return done;
}

bool Device::waitIdle(uint64_t timeoutNs) { return waitFrames(frame_, timeoutNs); }

// A lost device never finishes another frame: the labels (data + interrupt) queued behind its work
// would stay unwritten for ever and the title's WAIT_REG_MEM would spin on them. They are released
// (the work is gone, the title is told what it waits for), once, with one log line.
void Device::noteLost() {
  if (lostNoted_.exchange(true)) return;
  log_("gpu.err what=labels_released reason=device_lost pending=" + std::to_string(labels_.size()));
  pumpLabels();
}

uint64_t Device::releasable() const {
  return lostNoted_.load() ? ~uint64_t(0) : backend_.completedFrames();
}

void Device::pumpLabels(bool wait) {
  if (!lostNoted_.load() && backend_.lost()) noteLost();  // (releases through pumpLabels again)
  const auto completed = [this] { return releasable(); };
  if (wait)
    labels_.release(completed);
  else
    labels_.tryRelease(completed);
}

bool Device::waitLabels(uint64_t frames, unsigned timeoutMs) {
  if (!lostNoted_.load() && backend_.lost()) noteLost();
  return labels_.waitBelow(frames, [this] { return releasable(); }, timeoutMs);
}

void Device::afterFrame(uint64_t frame, uint64_t address, uint32_t size, std::function<void()> action) {
  labels_.add(frame, address, size, std::move(action));
  pumpLabels();
}

void Device::startSubmitThread(unsigned maxQueued) {
  if (!maxQueued || maxQueued_) return;
  maxQueued_ = maxQueued;
  worker_ = std::thread([this] { workerLoop(); });
}

void Device::drain() {
  if (!maxQueued_) return;
  std::unique_lock lock(qMutex_);
  qIdle_.wait(lock, [&] { return stop_ || (tasks_.empty() && !busy_); });
}

void Device::withBackend(const std::function<void()>& fn) {
  std::lock_guard lock(backendMutex_);
  fn();
}

void Device::enqueue(Task&& t) {
  std::unique_lock lock(qMutex_);
  qSpace_.wait(lock, [&] { return stop_ || tasks_.size() < maxQueued_; });
  tasks_.push_back(std::move(t));
  lock.unlock();
  qWake_.notify_one();
}

void Device::runOrdered(std::function<void()> call, uint64_t key) {
  if (!maxQueued_) {
    std::lock_guard lock(backendMutex_);
    call();
    return;
  }
  Task t;
  t.call = std::move(call);
  if (key) {
    std::lock_guard lock(qMutex_);
    ++pendingDestroy_[key];
  }
  // (a destroy never blocks the caller for long: it waits for queue space like a frame, no more)
  std::unique_lock lock(qMutex_);
  qSpace_.wait(lock, [&] { return stop_ || tasks_.size() < maxQueued_ + 8; });
  t.index = key;
  tasks_.push_back(std::move(t));
  lock.unlock();
  qWake_.notify_one();
}

void Device::workerLoop() {
  for (;;) {
    Task t;
    {
      std::unique_lock lock(qMutex_);
      qWake_.wait(lock, [&] { return stop_ || !tasks_.empty(); });
      if (tasks_.empty()) return;  // stop_ and nothing left
      t = std::move(tasks_.front());
      tasks_.pop_front();
      busy_ = true;
    }
    qSpace_.notify_all();
    if (t.call) {
      {
        std::lock_guard lock(backendMutex_);
        t.call();
      }
      if (t.index) {
        std::lock_guard lock(qMutex_);
        if (--pendingDestroy_[t.index] <= 0) pendingDestroy_.erase(t.index);
      }
    } else {
      execute(t.index, t.frame, t.compiled, t.lines, t.passDesc);
      std::lock_guard lock(qMutex_);
      handed_ = t.index + 1;
    }
    {
      std::lock_guard lock(qMutex_);
      busy_ = false;
    }
    qIdle_.notify_all();
  }
}

FrameReport Device::submit(const Frame& frame) {
  if (maxQueued_) return submit(Frame(frame));
  CompiledFrame compiled;
  std::vector<std::string> lines, passDesc;
  uint64_t index;
  FrameReport report = takeFrame(frame, compiled, lines, passDesc, index);
  execute(index, frame, compiled, lines, passDesc);
  return report;
}

FrameReport Device::submit(Frame&& frame) {
  if (!maxQueued_) return submit(static_cast<const Frame&>(frame));
  Task t;
  FrameReport report = takeFrame(frame, t.compiled, t.lines, t.passDesc, t.index);
  t.frame = std::move(frame);
  enqueue(std::move(t));
  return report;
}

// On the caller's thread: compile, hold for pipelines, report and the log lines of the frame (they
// read the resource table, which only the caller may touch).
FrameReport Device::takeFrame(const Frame& frame, CompiledFrame& compiled, std::vector<std::string>& lines,
                              std::vector<std::string>& passDesc, uint64_t& indexOut) {
  const uint64_t index = frame_++;
  indexOut = index;
  const auto start = std::chrono::steady_clock::now();
  compiled = compileFrame(frame, &resources_);
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
      const bool complete = pipelines_.waitReady(pending, std::chrono::milliseconds(holdMs_));
      const double held = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h0).count();
      ++holdFrames_, holdTotalMs_ += held;
      if (!complete) ++holdTimeouts_;
      if (holdFrames_ <= 16 || holdFrames_ % 64 == 0 || !complete) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "gpu.hold f=%llu pipelines=%zu ms=%.0f complete=%d total_frames=%llu total_ms=%.0f timeouts=%llu",
                      (unsigned long long)index, pending.size(), held, complete ? 1 : 0,
                      (unsigned long long)holdFrames_, holdTotalMs_, (unsigned long long)holdTimeouts_);
        log_(buf);
      }
    }
  }
  FrameReport report = makeReport(index, frame, compiled);
  report.cpuCompileMs = ms;
  report.pipelines = pipelines_.takeFrameStats();
  // Per-frame summaries: every frame for the first 300, then 1 in 90 (or when pipelines were
  // compiled) — three log lines per frame on the render thread was itself a stutter source.
  const bool verbose = index < 300 || index % 90 == 0 || report.pipelines.compiled > 0;
  if (verbose)
    for (std::string& line : formatReport(report)) lines.push_back(std::move(line));
  if (!compiled.badDraws.empty()) {  // client bug: those draws do not run (see CompiledFrame::badDraws)
    const DrawRef& d = compiled.badDraws.front();
    char buf[128];
    std::snprintf(buf, sizeof buf, "gpu.err what=bad_fetch f=%llu draws=%zu reads=%u first=c%u.d%u",
                  (unsigned long long)index, compiled.badDraws.size(), compiled.stats.badFetch, d.command, d.draw);
    lines.push_back(buf);
  }
  if (detailEvery_ && index % detailEvery_ == 0)
    for (std::string& line : formatEncoders(index, frame, compiled)) lines.push_back(std::move(line));

  // Per-pass timing (diagnostic): describe the encoders of the first frame with render work after
  // each `passEvery_` boundary (VPVulkan alternates render frames with trivial compute submits).
  if (passEvery_ && index >= passNext_ && passCount_.load() < 8) {
    bool render = false;
    for (const Encoder& e : compiled.encoders) render |= e.kind == Encoder::Kind::Render;
    if (render) {
      passNext_ = index + passEvery_;
      std::vector<std::string>& desc = passDesc;
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
    }
  }
  return report;
}

// Where the backend runs the frame (the submit thread, or the caller without one).
void Device::execute(uint64_t index, const Frame& frame, const CompiledFrame& compiled,
                     const std::vector<std::string>& lines, std::vector<std::string>& passDesc) {
  (void)frame;
  {
    std::lock_guard lock(backendMutex_);
    backend_.submit(index, frame, compiled);
  }
  for (const std::string& line : lines) log_(line);
  if (!passDesc.empty()) passPending_.emplace_back(index, std::move(passDesc)), ++passCount_;
  reportTimings();
  pumpLabels();
}

void Device::reportTimings() {
  uint64_t done;
  GpuTimings t;
  while (backend_.takeTimings(done, t)) {
    while (!passPending_.empty() && passPending_.front().first < done) (passPending_.pop_front(), --passCount_);
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
      (passPending_.pop_front(), --passCount_);
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
}

}  // namespace vp4gpu
