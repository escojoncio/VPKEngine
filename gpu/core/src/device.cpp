#include "vpgpu/core/device.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <variant>
#include <vector>

namespace vpgpu {

namespace {
const auto kProcessStart = std::chrono::steady_clock::now();  // gpu.time t= counts from here
}

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
  // VPVK_FRAME_OPT=0: no merge / hoist / sink in the frame compiler (CompileOptions::frameOpt). Anything else or unset = on.
  if (const char* e = std::getenv("VPVK_FRAME_OPT"); e && e[0] == '0' && !e[1]) compile_.frameOpt = false;
  log_(std::string("gpu.frame_opt enabled=") + (compile_.frameOpt ? "1" : "0"));
  if (const char* e = std::getenv("VP_HOLD_MS"); e && *e) holdMs_ = uint32_t(std::strtoul(e, nullptr, 10));
  // The two-regime hold is opt-in (see setHoldSteadyMs): these env vars are aliases of its setters.
  if (const char* e = std::getenv("VPKENGINE_HOLD_MS"); e && *e) setHoldSteadyMs(uint32_t(std::strtoul(e, nullptr, 10)));
  if (const char* e = std::getenv("VPKENGINE_HOLD_WARMUP_FRAMES"); e && *e)
    setHoldWarmupFrames(uint32_t(std::strtoul(e, nullptr, 10)));
  log_("gpu.hold armed max_ms=" + std::to_string(holdMs_) +
       (holdTwoRegime_ ? " steady_ms=" + std::to_string(holdSteadyMs_) + " warmup_frames=" + std::to_string(holdWarmupFrames_)
                       : std::string()));
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
  // VPVK_TILE_HEAVY=on|off|observe (CompileOptions::heavyMode): off / observe keep the attachments of heavy encoders memoryless
  // (a diagnostic: does the tiler survive partial renders without memory?); observe also logs them (gpu.tile observe).
  if (const char* e = std::getenv("VPVK_TILE_HEAVY"); e && *e) {
    const std::string v = e;
    if (v == "off") compile_.heavyMode = CompileOptions::HeavyMode::Off;
    else if (v == "observe") compile_.heavyMode = CompileOptions::HeavyMode::Observe;
  }
  tileStart_ = tileLogged_ = compile_.tileMaxVertices;
  tileMaxVerticesStart().store(tileStart_, std::memory_order_relaxed);
  rtMemNext_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(rtMemMs_);  // gpu.mem.rt: first line one period in
  log_("gpu.tile heavy_max_draws=" + std::to_string(compile_.tileMaxDraws) +
       " heavy_max_vertices=" + std::to_string(compile_.tileMaxVertices) + " heavy_mode=" +
       (compile_.heavyMode == CompileOptions::HeavyMode::On ? "on"
        : compile_.heavyMode == CompileOptions::HeavyMode::Off ? "off" : "observe"));
  // VP_FRAME_DETAIL=N: same as VP_ENC_DETAIL (kept for AVP's scripts); wins when both are set.
  if (const char* e = std::getenv("VP_FRAME_DETAIL"); e && *e) {
    detailEvery_ = std::strtoull(e, nullptr, 10);
    if (detailEvery_) log_("gpu.detail armed every=" + std::to_string(detailEvery_));
  }
  // GPU-ordered labels (afterFrame) are released as frames finish; a backend with no notification (the default) leaves
  // it to the next Device call.
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
  backend_.waitAll(UINT64_MAX);     // completion handlers may still call pumpLabels
  backend_.setFrameDoneNotify({});  // (waits for a callback that is running)
  labels_.release(~uint64_t(0));    // the GPU is idle: whatever was queued has happened
}

// Waits for the backend to drop the previous resource of this slot before a create reuses it (a destroy still queued
// behind frames on the submit thread).
void Device::settleCreate(char kind, uint32_t index) {
  if (!maxQueued_) return;
  bool wait;
  {
    std::lock_guard lock(qMutex_);
    wait = pendingDestroy_.count(destroyKey(kind, index)) != 0;
  }
  if (wait) drain();  // the backend still holds the previous resource of this slot
}

// The table is changed under resMutex_; the backend call (allocation, can be slow) and the log run outside it, under
// backendMutex_ (lock order: resMutex_, then backendMutex_; here they are never held together). A backend destroy that
// follows a reuse of the index is harmless: backends compare the generation.
BufferId Device::createBuffer(BufferDesc d) {
  BufferId id;
  BufferDesc copy;
  {
    std::lock_guard lock(resMutex_);
    id = resources_.createBuffer(d);
    if (id) copy = *resources_.get(id);
  }
  if (id) {
    if (copy.guestMemory) live_.guestBuffers.fetch_add(1), live_.guestBytes.fetch_add(copy.size);
    else live_.buffers.fetch_add(1), live_.bufferBytes.fetch_add(copy.size);
    settleCreate('b', id.index);
    bool made;
    {
      auto lock = lockBackend();
      backend_.createBuffer(id, copy);
      made = !refuseFailedBuffers_ || backend_.bufferCreated(id);
    }
    if (!made && refuseFailedBuffers_) {  // the backend refused it: no id for the caller to bind (it would read as address 0)
      bool tell;
      {
        std::lock_guard lock(refusedLogMutex_);
        tell = refusedLogged_.size() < 64 && refusedLogged_.insert({copy.size, copy.label}).second;
      }
      if (tell) log_("gpu.err what=buffer_backend size=" + std::to_string(copy.size) + " label=" + copy.label);
      destroy(id);
      return BufferId{};
    }
  } else {
    log_("gpu.err what=buffer_desc label=" + d.label);
  }
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
  if (id) {
    live_.texCount[texKind(copy)].fetch_add(1);
    live_.texBytes[texKind(copy)].fetch_add(texLogicalBytes(copy));
    settleCreate('t', id.index);
    auto lock = lockBackend();
    backend_.createTexture(id, copy);
  } else {
    log_("gpu.err what=texture_desc label=" + d.label);
  }
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
  live_.samplers.fetch_add(1);
  settleCreate('s', id.index);
  auto lock = lockBackend();
  backend_.createSampler(id, copy);
  return id;
}

void Device::destroy(BufferId id) {
  bool gone;
  {
    std::lock_guard lock(resMutex_);
    if (const BufferDesc* d = resources_.get(id)) {
      if (d->guestMemory) live_.guestBuffers.fetch_sub(1), live_.guestBytes.fetch_sub(d->size);
      else live_.buffers.fetch_sub(1), live_.bufferBytes.fetch_sub(d->size);
    }
    gone = resources_.destroy(id);
  }
  if (gone) runOrdered([this, id] { backend_.destroyBuffer(id); }, destroyKey('b', id.index));
}
void Device::destroy(TextureId id) {
  bool gone;
  {
    std::lock_guard lock(resMutex_);
    if (const TextureDesc* d = resources_.get(id)) {
      live_.texCount[texKind(*d)].fetch_sub(1);
      live_.texBytes[texKind(*d)].fetch_sub(texLogicalBytes(*d));
    }
    gone = resources_.destroy(id);
  }
  if (gone) runOrdered([this, id] { backend_.destroyTexture(id); }, destroyKey('t', id.index));
}
void Device::destroy(SamplerId id) {
  bool gone;
  {
    std::lock_guard lock(resMutex_);
    if (resources_.get(id)) live_.samplers.fetch_sub(1);
    gone = resources_.destroy(id);
  }
  if (gone) runOrdered([this, id] { backend_.destroySampler(id); }, destroyKey('s', id.index));
}

void Device::trackRenderTargets(uint64_t index, const CompiledFrame& compiled, std::vector<std::string>& lines) {
  for (const TransientViolation& v : compiled.transientViolations) {
    ++transientViolations_;
    const TextureDesc* d = resources_.get(v.texture);
    if (!d || transientLogs_ >= 32) continue;
    ++transientLogs_;
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "gpu.mem.transient_violation f=%llu tex=%u %ux%ux%u s%u fmt=%s why=%s load=%d (a TRANSIENT image whose contents are "
                  "needed: it gets memory, see gpu.mem.rt)",
                  (unsigned long long)index, v.texture.index, d->width, d->height, unsigned(d->layers), unsigned(d->samples),
                  formatInfo(d->format).name, toString(v.why), int(v.load));
    lines.emplace_back(buf);
  }
  for (const Encoder& e : compiled.encoders) {
    if (e.kind != Encoder::Kind::Render) continue;
    for (int slot = 0; slot <= kMaxColorAttachments; ++slot) {
      const Attachment& a = slot == kDepthSlot ? e.pass.depth : e.pass.colors[slot];
      const TextureDesc* d = a.texture ? resources_.get(a.texture) : nullptr;
      if (!d) continue;
      RtMem& r = rtMem_[uint64_t{a.texture.generation} << 32 | a.texture.index];
      if (r.frame != index) {  // first encoder of this frame that attaches it
        r = {uint64_t{d->width} * d->height * d->samples * d->layers * formatInfo(d->format).bytesPerBlock, index,
             e.tileLocal(slot), e.tile[slot]};
      } else if (r.memoryless && !e.tileLocal(slot)) {  // one encoder that needs memory is enough
        r.memoryless = false, r.why = e.tile[slot];
      }
    }
  }
  const auto now = std::chrono::steady_clock::now();
  if (now < rtMemNext_) return;
  rtMemNext_ = now + std::chrono::milliseconds(rtMemMs_);
  constexpr size_t kV = static_cast<size_t>(TileVerdict::Count);
  uint32_t memoryless = 0, mat = 0, whyN[kV] = {};
  uint64_t memorylessB = 0, matB = 0, whyB[kV] = {};
  for (auto it = rtMem_.begin(); it != rtMem_.end();) {
    if (!resources_.get(TextureId{uint32_t(it->first & 0xffffffffu), uint32_t(it->first >> 32)})) {
      it = rtMem_.erase(it);  // destroyed
      continue;
    }
    const RtMem& r = it->second;
    if (r.memoryless) {
      ++memoryless, memorylessB += r.bytes;
    } else {
      ++mat, matB += r.bytes;
      ++whyN[static_cast<size_t>(r.why)], whyB[static_cast<size_t>(r.why)] += r.bytes;
    }
    ++it;
  }
  if (rtMem_.empty() && !transientViolations_) return;
  char buf[256];
  std::snprintf(buf, sizeof buf, "gpu.mem.rt memoryless_n=%u memoryless_mib_saved=%.1f mat_n=%u mat_mib=%.1f transient_violation=%llu mat_why=",
                memoryless, double(memorylessB) / 1048576.0, mat, double(matB) / 1048576.0,
                (unsigned long long)transientViolations_);
  std::string line = buf;
  bool any = false;
  for (size_t v = 0; v < kV; ++v) {
    if (!whyN[v]) continue;
    std::snprintf(buf, sizeof buf, "%s%s:%u/%.1f", any ? "," : "", toString(static_cast<TileVerdict>(v)), whyN[v],
                  double(whyB[v]) / 1048576.0);
    line += buf, any = true;
  }
  if (!any) line += "-";
  if (line == rtMemLast_) return;
  rtMemLast_ = line;
  lines.push_back(std::move(line));
}

unsigned Device::texKind(const TextureDesc& d) {
  if (d.usage & usage::depthTarget) return 1;
  if (d.samples > 1) return 2;
  if (d.usage & usage::colorTarget) return 0;
  return 3;
}

uint64_t Device::texLogicalBytes(const TextureDesc& d) {
  const FormatInfo& fi = formatInfo(d.format);
  const uint64_t bw = std::max<uint32_t>(fi.blockSize, 1), bh = std::max<uint32_t>(fi.blockHeight, 1), bpb = std::max<uint32_t>(fi.bytesPerBlock, 1);
  uint64_t total = 0;
  for (uint32_t level = 0; level < std::min<uint32_t>(std::max<uint16_t>(d.mips, 1), 32); ++level) {
    const uint64_t w = std::max<uint32_t>(d.width >> level, 1), h = std::max<uint32_t>(d.height >> level, 1);
    const uint64_t slices = d.type == TextureType::Tex3D ? std::max<uint32_t>(d.depth >> level, 1) : std::max<uint16_t>(d.layers, 1);
    total += (w + bw - 1) / bw * ((h + bh - 1) / bh) * bpb * slices;
  }
  return total * std::max<uint8_t>(d.samples, 1);
}

void Device::publishProbe() {
  if (!probe::enabled()) return;
  using namespace probe;
  static const char* kTex[4] = {"tex_color", "tex_depth", "tex_ms", "tex_other"};
  constexpr double kMb = 1.0 / (1024.0 * 1024.0);
  uint64_t texN = 0, texB = 0;
  for (unsigned k = 0; k < 4; ++k) {
    const uint64_t n = live_.texCount[k].load(), b = live_.texBytes[k].load();
    gauge("RES", std::string(kTex[k]) + "_live").set(double(n));
    gauge("RES", std::string(kTex[k]) + "_mb").set(double(b) * kMb);
    texN += n, texB += b;
  }
  gauge("RES", "tex_live").set(double(texN));
  gauge("RES", "tex_logical_mb").set(double(texB) * kMb);
  gauge("RES", "buf_own_live").set(double(live_.buffers.load()));
  gauge("RES", "buf_own_mb").set(double(live_.bufferBytes.load()) * kMb);
  gauge("RES", "buf_guest_live").set(double(live_.guestBuffers.load()));
  gauge("RES", "buf_guest_mb").set(double(live_.guestBytes.load()) * kMb);
  gauge("RES", "samplers_live").set(double(live_.samplers.load()));
  gauge("RES", "frames_submitted").set(double(frame_));
  gauge("RES", "frames_completed").set(double(backend_.completedFrames() > frame_ ? frame_ : backend_.completedFrames()));
  gauge("RES", "labels_queued").set(labels_.empty() ? 0 : 1);
  gauge("RES", "submit_queue").set(double(passCount_.load()));
  const PipelineCensus c = pipelines_.census();
  const PipelineStats t = pipelines_.totals();
  gauge("PSO", "entries").set(c.entries);
  gauge("PSO", "ready").set(c.ready);
  gauge("PSO", "queued").set(c.queued);
  gauge("PSO", "compiling").set(c.compiling);
  gauge("PSO", "failed_entries").set(c.failed);
  gauge("PSO", "orphan").set(c.orphan);
  gauge("PSO", "queue_length").set(c.queueLength);
  gauge("PSO", "workers").set(c.workers);
  gauge("PSO", "busy_workers").set(c.busyWorkers);
  gauge("PSO", "owners").set(c.owners);
  counter("PSO", "compiled").set(t.compiled);
  counter("PSO", "failed").set(t.failed);
  counter("PSO", "request_hits").set(t.hits);
  counter("PSO", "request_misses").set(t.misses);
  gauge("PSO", "compile_ms_total").set(t.totalMs);
  gauge("PSO", "compile_ms_max").set(t.maxMs);
  counter("PSO", "hold_frames").set(holdFrames_);
  counter("PSO", "hold_timeouts").set(holdTimeouts_);
  gauge("PSO", "hold_ms_total").set(holdTotalMs_);
  gauge("PSO", "hold_steady_ms").set(holdSteadyMs_);
  gauge("PSO", "hold_budget_ms").set(holdMs());
}

bool Device::waitFrames(uint64_t frames, uint64_t timeoutNs) {
  static probe::Timer& waits = probe::timer("WAIT", "device_wait_frames");
  const probe::Scoped scope(waits);
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
  if (labels_.empty()) return;  // (nothing queued: a host that never uses labels pays one atomic load)
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
  if (!maxQueued_) {  // no submit thread: straight to the backend, unserialized like the creates
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
// read the resource table, which the loading threads also change: under resMutex_).
FrameReport Device::takeFrame(const Frame& frame, CompiledFrame& compiled, std::vector<std::string>& lines,
                              std::vector<std::string>& passDesc, uint64_t& indexOut) {
  const uint64_t index = frame_++;
  indexOut = index;
  const auto start = std::chrono::steady_clock::now();
  compile_.splitViews = splitViewsSwitch().load(std::memory_order_relaxed) != 0;  // run-time switch: read once per frame
  // Run-time heavy-geometry limit (0 = the value this Device started with). One above it applies to frames only while it is
  // waiting for its canary or accepted (tileProbeState()); a canary frame is the first with an encoder only that limit keeps light.
  const uint64_t hotTile = tileMaxVerticesSwitch().load(std::memory_order_relaxed);
  const int probe = tileProbeState().load();
  const bool raised = hotTile > tileStart_ && (probe == kProbeWaiting || probe == kProbeAccepted);
  compile_.tileMaxVertices = raised ? hotTile : hotTile && hotTile <= tileStart_ ? hotTile : tileStart_;
  compile_.tileBaseVertices = raised ? tileStart_ : 0;
  std::unique_lock resLock(resMutex_);
  compiled = compileFrame(frame, &resources_, compile_);
  resLock.unlock();
  if (raised && probe == kProbeWaiting && compiled.stats.tileRaised) {
    int waiting = kProbeWaiting;
    tileProbeFrame().store(index);  // before the backend can see the frame
    if (tileProbeState().compare_exchange_strong(waiting, kProbeProbing))
      log_("gpu.tile probe start max_vertices=" + std::to_string(hotTile) + " raised_enc=" + std::to_string(compiled.stats.tileRaised) +
           " f=" + std::to_string(index));
  }
  if (compile_.tileMaxVertices != tileLogged_) {
    tileLogged_ = compile_.tileMaxVertices;
    log_("gpu.tile max_vertices=" + std::to_string(tileLogged_) + " f=" + std::to_string(index));
  }
  if (int(compile_.splitViews) != splitMode_) {
    splitMode_ = compile_.splitViews;
    log_("gpu.split mode=" + std::to_string(splitMode_) + " f=" + std::to_string(index));
  }
  // One line per kind of two-layer pass: split into two encoders (eligible=1) or why not.
  for (const Encoder& e : compiled.encoders) {
    if (!e.splitWhy || e.view == 1 || !splitLogged_.insert(e.splitSig).second) continue;
    std::string label = e.pass.label.empty() ? "-" : e.pass.label.substr(0, 40);
    for (char& c : label)
      if (c == ' ' || c == '=') c = '_';
    uint64_t draws = 0;
    for (uint32_t c : e.commands)
      if (const auto* rp = std::get_if<RenderPass>(&frame.commands()[c])) draws += rp->draws.size();
    char buf[256];
    std::snprintf(buf, sizeof buf, "gpu.split pass=%016llx eligible=%d reason=%s label=%s view_verts=%llu draws=%llu heavy_after=%d",
                  (unsigned long long)e.splitSig, e.view == 0 ? 1 : 0, e.splitWhy, label.c_str(),
                  (unsigned long long)e.splitViewVerts, (unsigned long long)draws, e.splitHeavyAfter ? 1 : 0);
    log_(buf);
  }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  compileNs_ += uint64_t(ms * 1e6);
  if (probe::enabled()) {
    static probe::Hist& compileHist = probe::hist("TIME", "submit_compile_us");
    compileHist.addMs(ms);
  }
  {  // Engine-internal kernels (Dispatch::required) are never skipped for a pipeline still compiling: waited for, bounded, with the hold off too.
    std::vector<PipelineKey> req;
    for (const Command& c : frame.commands()) {
      const auto* dp = std::get_if<Dispatch>(&c);
      if (!dp || !dp->required || !dp->pipeline.value) continue;
      const auto st = pipelines_.status(dp->pipeline);
      if (st != PipelineCache::Status::Queued && st != PipelineCache::Status::Compiling) continue;
      bool known = false;
      for (PipelineKey p : req) known = known || p.value == dp->pipeline.value;
      if (!known) req.push_back(dp->pipeline);
    }
    if (!req.empty()) {
      for (PipelineKey k : req) pipelines_.boost(k);
      const auto w0 = std::chrono::steady_clock::now();
      const bool complete = pipelines_.waitReady(req, std::chrono::milliseconds(2000));
      char buf[128];
      std::snprintf(buf, sizeof buf, "gpu.required_wait f=%llu pipelines=%zu ms=%.0f complete=%d", (unsigned long long)index, req.size(),
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count(), complete ? 1 : 0);
      log_(buf);
    }
  }
  // A draw whose geometry expansion (ES/GS Dispatch, Draw::expandedBy) is not built is not drawn: it would read an
  // expansion nobody wrote (garbage geometry).
  const auto expansionPending = [&](const Draw& d) {
    for (PipelineKey k : d.expandedBy)
      if (k.value && pipelines_.status(k) != PipelineCache::Status::Ready) return true;
    return false;
  };
  // Hold the frame while pipelines it uses are still compiling (see holdMs_). Held time, whether it ran out, how many
  // pipelines were waited for, and (two-regime hold) how many of them were compute work.
  double held = 0;
  bool complete = true;
  size_t waited = 0, waitedCompute = 0;
  const bool warm = holdTwoRegime_ && holdWarm(index);
  const uint32_t budgetMs = !holdTwoRegime_ || warm ? holdMs_ : std::min(holdMs_, holdSteadyMs_);  // (holdMs() for this frame)
  if (holdMs_) {
    std::vector<PipelineKey> pendingDraws, pendingCompute;
    auto note = [&](std::vector<PipelineKey>& pending, PipelineKey k) {
      if (!k.value) return;
      const auto st = pipelines_.status(k);
      if (st != PipelineCache::Status::Queued && st != PipelineCache::Status::Compiling) return;
      for (PipelineKey p : pending)
        if (p.value == k.value) return;
      pending.push_back(k);
    };
    if (!holdTwoRegime_) {
      // One wait for every pipeline of the frame (draws and compute alike), up to holdMs_; after one that timed out,
      // kHoldBackoffMs for a while.
      for (const Command& c : frame.commands()) {
        if (const auto* rp = std::get_if<RenderPass>(&c))
          for (const Draw& d : rp->draws) note(pendingDraws, d.pipeline);
        else if (const auto* dp = std::get_if<Dispatch>(&c))
          note(pendingDraws, dp->pipeline);
      }
      if (!pendingDraws.empty()) {
        const auto h0 = std::chrono::steady_clock::now();
        const bool backoff = h0 < holdBackoffUntil_;
        const uint32_t budget = backoff ? std::min(holdMs_, kHoldBackoffMs) : holdMs_;
        complete = pipelines_.waitReady(pendingDraws, std::chrono::milliseconds(budget));
        held = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h0).count();
        waited = pendingDraws.size();
        if (!complete) holdBackoffUntil_ = std::chrono::steady_clock::now() + std::chrono::seconds(3);
      }
    } else {
      // Compute work (Dispatch: compute, ES/GS expansion, the work a draw reads its input from) waits for the whole
      // hold (holdMs_) at any time: a Dispatch left out would leave its outputs unwritten (a draw reading an expansion
      // nobody filled, a buffer or guest memory never updated, and for good if nothing writes it again). Only the draws
      // of render passes use the budget of the moment: the whole hold in the warm-up, the short steady one after it. A
      // draw that is not built by then is left out of this frame only (the backend counts it as a missing pipeline):
      // the frame is incomplete for one display frame, like the "async shaders" of other emulators, and the caller
      // does not stall for the whole hold on every frame that meets a new pipeline.
      // (a draw whose geometry expansion is not built will be left out, see below: no reason to wait for its pipeline)
      for (const Command& c : frame.commands())
        if (const auto* dp = std::get_if<Dispatch>(&c)) note(pendingCompute, dp->pipeline);
      auto wait = [&](const std::vector<PipelineKey>& keys, uint32_t ms) {
        if (keys.empty()) return;
        const auto h0 = std::chrono::steady_clock::now();
        complete = pipelines_.waitReady(keys, std::chrono::milliseconds(ms)) && complete;
        held += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h0).count();
      };
      wait(pendingCompute, holdMs_);
      // The draws wait after the compute work (so an expansion that gave up is known), within one deadline: in the
      // warm-up the whole hold covers both (never 2 x holdMs_); in steady state the compute work may take holdMs_
      // and the draws then get the short budget on top.
      if (budgetMs)
        for (const Command& c : frame.commands())
          if (const auto* rp = std::get_if<RenderPass>(&c))
            for (const Draw& d : rp->draws)
              if (!expansionPending(d)) note(pendingDraws, d.pipeline);
      wait(pendingDraws, warm ? uint32_t(std::max(0.0, double(budgetMs) - held)) : budgetMs);
      waited = pendingCompute.size() + pendingDraws.size(), waitedCompute = pendingCompute.size();
      if (warm) holdWarmMs_ += held;
    }
    if (waited) {
      ++holdFrames_, holdTotalMs_ += held;
      if (!complete) ++holdTimeouts_;
      holdNs_ += uint64_t(held * 1e6);
      if (probe::enabled()) {
        static probe::Hist& holdHist = probe::hist("TIME", "submit_hold_us");
        holdHist.addMs(held);
      }
      ++holdFramesT_;
      if (!complete) ++holdTimeoutsT_;
      if (holdFrames_ <= 16 || holdFrames_ % 64 == 0 || !complete) {
        char buf[224];
        if (!holdTwoRegime_)
          std::snprintf(buf, sizeof buf, "gpu.hold f=%llu pipelines=%zu ms=%.0f complete=%d total_frames=%llu total_ms=%.0f timeouts=%llu",
                        (unsigned long long)index, waited, held, complete ? 1 : 0, (unsigned long long)holdFrames_, holdTotalMs_,
                        (unsigned long long)holdTimeouts_);
        else
          std::snprintf(buf, sizeof buf,
                        "gpu.hold f=%llu pipelines=%zu compute=%zu ms=%.0f complete=%d total_frames=%llu total_ms=%.0f timeouts=%llu budget_ms=%u warm=%d",
                        (unsigned long long)index, waited, waitedCompute, held, complete ? 1 : 0, (unsigned long long)holdFrames_,
                        holdTotalMs_, (unsigned long long)holdTimeouts_, budgetMs, warm ? 1 : 0);
        log_(buf);
      }
    }
  }
  // Memory validity after this frame (the next compileFrame reads it, so it is settled here, not on the submit thread)
  // and the render-target ledger. These lines come first: they are logged right after the backend submit.
  {
    std::lock_guard lock(resMutex_);  // memValid_ is part of the resource table: createTexture/destroy run under resMutex_
    commitMemory(compiled, resources_);
    trackRenderTargets(index, compiled, lines);
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
  if (frame.hasExpansion()) {  // Draws whose expansion is not built (see expansionPending): left out like the Dispatch is.
    size_t chained = 0;
    DrawRef first;
    for (uint32_t ci = 0; ci < frame.commands().size(); ++ci) {
      const auto* rp = std::get_if<RenderPass>(&frame.commands()[ci]);
      if (!rp) continue;
      for (uint32_t j = 0; j < rp->draws.size(); ++j)
        if (expansionPending(rp->draws[j])) {
          if (!chained++) first = {ci, j};
          compiled.badDraws.push_back({ci, j});
        }
    }
    if (chained) {
      std::sort(compiled.badDraws.begin(), compiled.badDraws.end());
      compiled.badDraws.erase(std::unique(compiled.badDraws.begin(), compiled.badDraws.end()), compiled.badDraws.end());
      char buf[128];
      std::snprintf(buf, sizeof buf, "gpu.err what=expansion_pending f=%llu draws=%zu first=c%u.d%u",
                    (unsigned long long)index, chained, first.command, first.draw);
      lines.push_back(buf);
    }
  }
  // One line (24 per run, then every 256th big frame) for a frame that costs hundreds of Metal encoders.
  if (bigFrame(compiled) && (bigLogged_ < 24 || index % 256 == 0)) {
    ++bigLogged_;
    std::string bigLine;
    {
      std::lock_guard lock(resMutex_);
      bigLine = formatBigFrame(index, frame, compiled, &resources_);
    }
    lines.push_back(std::move(bigLine));
  }
  for (const TextureId t : compiled.elideMisses) {  // banned right after: once per texture
    char buf[128];
    std::snprintf(buf, sizeof buf, "gpu.msaa elide_miss f=%llu tex=%u (elided store was needed; elision banned)",
                  (unsigned long long)index, t.index);
    lines.push_back(buf);
  }
  {
    static std::atomic<uint32_t> observed{0};  // VPVK_TILE_HEAVY=observe: the first 40 heavy encoders, then one frame in 600
    for (const CompiledFrame::HeavyObs& h : compiled.heavyObs)
      if (observed.fetch_add(1) < 40 || index % 600 == 0) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "gpu.tile observe f=%llu enc=%u draws=%llu verts=%llu (memoryless kept; limit draws=%u verts=%llu)",
                      (unsigned long long)index, h.encoder, (unsigned long long)h.draws, (unsigned long long)h.vertices,
                      compile_.tileMaxDraws, (unsigned long long)compile_.tileMaxVertices);
        log_(buf);
      }
  }
  {  // heavy all-autoMsaa encoders splitHeavyAuto left whole (they keep tile-memory samples and may trip the tiler)
    const CompileStats& st = compiled.stats;
    const struct {
      const char* why;
      uint32_t n;
    } skips[] = {{"stencil", st.tileSplitSkipStencil}, {"single_draw", st.tileSplitSkipSingle}, {"cap", st.tileSplitSkipCap}};
    for (const auto& s : skips)
      if (s.n && (splitSkipLogs_ < 8 || index % 600 == 0)) {
        ++splitSkipLogs_;
        char buf[96];
        std::snprintf(buf, sizeof buf, "gpu.tile split_skip why=%s enc=%u f=%llu", s.why, s.n, (unsigned long long)index);
        log_(buf);
      }
  }
  if (detailEvery_ && index % detailEvery_ == 0)
    for (std::string& line : formatEncoders(index, frame, compiled)) lines.push_back(std::move(line));
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
    lines.push_back(buf);
  }


  // Per-pass timing (diagnostic): describe the encoders of the first frame with render work after
  // each `passEvery_` boundary (VPVulkan alternates render frames with trivial compute submits).
  if (passEvery_ && index >= passNext_ && passCount_.load() < 8) {
    bool render = false;
    for (const Encoder& e : compiled.encoders) render |= e.kind == Encoder::Kind::Render;
    if (render) {
      passNext_ = index + passEvery_;
      std::lock_guard resGuard(resMutex_);  // resources_.get below
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
  const auto b0 = std::chrono::steady_clock::now();
  {
    auto lock = lockBackend();
    backend_.submit(index, frame, compiled);
  }
  const uint64_t backendNs = uint64_t(std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - b0).count());
  backendNs_ += backendNs;
  if (probe::enabled()) {
    static probe::Hist& backendHist = probe::hist("TIME", "submit_backend_us");
    backendHist.add(backendNs / 1000);
  }
  for (const std::string& line : lines) log_(line);
  if (!passDesc.empty()) passPending_.emplace_back(index, std::move(passDesc)), ++passCount_;
  reportTimings();
  pumpLabels();
}

Device::SubmitTimes Device::takeSubmitTimes() {
  const auto ms = [](std::atomic<uint64_t>& ns) { return double(ns.exchange(0)) / 1e6; };
  SubmitTimes t;
  t.holdMs = ms(holdNs_);
  t.compileMs = ms(compileNs_);
  t.backendMs = ms(backendNs_);
  t.slotWaitMs = backend_.takeSlotWaitMs();
  t.holdFrames = holdFramesT_.exchange(0);
  t.holdTimeouts = holdTimeoutsT_.exchange(0);
  return t;
}

void Device::reportTimings() {
  uint64_t done;
  GpuTimings t;
  while (backend_.takeTimings(done, t)) {
    if (probe::enabled()) {  // the GPU's own time for the frame and for each pass (timestamps from the backend)
      static probe::Hist& frameGpu = probe::hist("GPU", "frame_gpu_us");
      static probe::Hist& passGpu = probe::hist("GPU", "pass_gpu_us");
      frameGpu.addMs(t.frameMs);
      for (double ms : t.encoderMs)
        if (ms > 0) passGpu.addMs(ms);
    }
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
    std::snprintf(buf, sizeof buf, " t=%.1f",
                  std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - kProcessStart).count());
    line += buf;
    log_(line);
  }
}

}  // namespace vpgpu
