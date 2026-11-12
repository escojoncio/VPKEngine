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
// Elided MSAA stores that a later frame needed (CompileOptions::msaaDeadStore), over the whole run, before dead-store elision is switched off.
constexpr uint32_t kDeadStoreMissLimit = 8;
// Times an MSAA store must be seen kept (same texture, slot, origin and verdict), before gpu.msaa.kept_store reports it.
constexpr uint32_t kKeptStoreSeen = 8;
}  // namespace

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
  // Drop client Stores of MSAA attachments nothing reads (CompileOptions::msaaDeadStore, speculative but banned per texture on a
  // miss, and off for good after kDeadStoreMissLimit misses). On by default; VPVK_MSAA_DEAD_STORE=0 turns it off.
  compile_.msaaDeadStore = true;
  if (const char* e = std::getenv("VPVK_MSAA_DEAD_STORE"); e && e[0] == '0' && !e[1]) compile_.msaaDeadStore = false;
  log_(std::string("gpu.msaa dead_store=") + (compile_.msaaDeadStore ? "1" : "0 (VPVK_MSAA_DEAD_STORE=0)"));
  // VPVK_FRAME_OPT=0: no merge / hoist / sink in the frame compiler (CompileOptions::frameOpt). Anything else or unset = on.
  if (const char* e = std::getenv("VPVK_FRAME_OPT"); e && e[0] == '0' && !e[1]) compile_.frameOpt = false;
  log_(std::string("gpu.frame_opt enabled=") + (compile_.frameOpt ? "1" : "0"));
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

// The backend destroy runs under resMutex_: once the slot is free another thread can create into the same index, and its
// backend create must not reach the backend before this destroy (resMutex_ comes before every backend table lock).
// Checked for every backend and platform in the tree (NullBackend, metal, metal2, vulkan platform_null / platform_metal):
// destroy* takes at most the backend's own table locks (metal2: tableMutex, then the leaf addressMutex / rangesMutex /
// doneMutex), logs nothing and calls nothing outside the backend, and no backend or platform holds a Device reference or
// calls back into Device, so no thread takes one of those locks and then resMutex_ (the reverse order never happens).
// A new backend must keep it that way: a destroy that calls out (log sink, callback) while resMutex_ is held can deadlock.
void Device::destroy(BufferId id) {
  std::lock_guard lock(resMutex_);
  if (resources_.destroy(id)) backend_.destroyBuffer(id);
}
void Device::destroy(TextureId id) {
  std::lock_guard lock(resMutex_);
  if (!resources_.destroy(id)) return;
  keptStoreSeen_.erase(uint64_t{id.generation} << 32 | id.index);
  backend_.destroyTexture(id);
}
void Device::destroy(SamplerId id) {
  std::lock_guard lock(resMutex_);
  if (resources_.destroy(id)) backend_.destroySampler(id);
}

void Device::trackRenderTargets(uint64_t index, const CompiledFrame& compiled, std::vector<std::string>& lines) {
  // One line per cause (TileVerdict) the first time an MSAA texture could not stay memoryless: what the game does that needs the
  // samples in memory. The verdicts are the compiler's; "game_needs" says what in the recorded work produces each.
  auto nonMemoryless = [&](TextureId t, TileVerdict why) {
    const TextureDesc* d = resources_.get(t);
    if (!d || d->samples < 2 || why == TileVerdict::None || why == TileVerdict::TileLocal || (nonMemorylessSeen_ >> int(why) & 1)) return;
    nonMemorylessSeen_ |= 1u << int(why);
    const char* need = "?";
    switch (why) {
      case TileVerdict::SampledRead: need = "a pass samples or reads the multisample image as a texture (texture2d_ms, input attachment, compute)"; break;
      case TileVerdict::OtherEncoder: need = "a later encoder Loads the samples (the pass was cut: feedback, attachment change, compute or copy between)"; break;
      case TileVerdict::LoadsContents: need = "the first pass Loads samples written by an earlier submit"; break;
      case TileVerdict::UsedElsewhere: need = "the image is copied or written outside its render encoder"; break;
      case TileVerdict::Persistent: need = "the image is persistent (read after the frame: eye image, history)"; break;
      case TileVerdict::Partial: need = "the attachment covers only some mips or layers"; break;
      case TileVerdict::SizeMismatch: need = "size or sample count differs from the encoder's other attachments"; break;
      case TileVerdict::BadFetch: need = "a pixel-local read of something that is not a colour attachment"; break;
      case TileVerdict::HeavyGeometry: need = "the encoder has so much geometry the tiler may flush (tileMaxDraws / tileMaxVertices)"; break;
      default: break;
    }
    std::string label = d->label.empty() ? "-" : d->label.substr(0, 40);
    for (char& c : label)
      if (c == ' ') c = '_';
    char buf[320];
    std::snprintf(buf, sizeof buf, "gpu.msaa.nonmemoryless rp=%s tex=%u %ux%ux%u s%u fmt=%s why=%s game_needs=\"%s\"", label.c_str(),
                  t.index, d->width, d->height, unsigned(d->layers), unsigned(d->samples), formatInfo(d->format).name, toString(why), need);
    lines.emplace_back(buf);
  };
  for (const TransientViolation& v : compiled.transientViolations) nonMemoryless(v.texture, v.why);
  for (const MsaaReject& r : compiled.msaaRejects) nonMemoryless(r.texture, r.why);
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
  // MSAA textures this frame read with contents from an earlier submit (stats.msaaStaleReads): what the last frame that attached
  // them did with the samples (before this frame's entry below replaces it). store D / R (or memoryless) there = the read sees
  // undefined samples (mem_valid: the resource table after this frame). The first 32 lines, then those of one frame in 600.
  for (const TextureId t : compiled.msaaStale) {
    if (msaaStaleLogs_ >= 32 && index % 600) break;
    ++msaaStaleLogs_;
    const TextureDesc* d = resources_.get(t);
    if (!d) continue;
    const auto it = rtMem_.find(uint64_t{t.generation} << 32 | t.index);
    static const char* const kStore[] = {"D", "S", "R", "SR"};
    char buf[320];
    std::snprintf(buf, sizeof buf,
                  "gpu.msaa_stale f=%llu tex=%u %ux%ux%u s%u fmt=%s transient=%d violated=%d persistent=%d mem_valid=%d prev_f=%lld "
                  "prev_store=%s prev_keep=%d prev_memoryless=%d prev_why=%s label=%s",
                  (unsigned long long)index, t.index, d->width, d->height, unsigned(d->layers), unsigned(d->samples),
                  formatInfo(d->format).name, d->transient ? 1 : 0, resources_.transientViolated(t) ? 1 : 0, d->persistent ? 1 : 0,
                  resources_.memValid(t) ? 1 : 0, it == rtMem_.end() ? -1LL : (long long)it->second.frame,
                  it == rtMem_.end() ? "-" : kStore[int(it->second.store) & 3], it == rtMem_.end() ? 0 : int(it->second.keepSamples),
                  it == rtMem_.end() ? 0 : int(it->second.memoryless), it == rtMem_.end() ? "-" : toString(it->second.why),
                  d->label.empty() ? "-" : d->label.substr(0, 40).c_str());
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
      r.store = a.store, r.keepSamples = a.keepSamples;
      // A store of MSAA samples that stays in memory (not elided, not tile-local): who asked for it, and what kept it, reported
      // once it has been seen kKeptStoreSeen times (gpu.msaa.nonmemoryless skips why=None and has no origin).
      if (d->samples > 1 && !a.autoMsaa && (a.store == StoreAction::Store || a.store == StoreAction::StoreAndResolve) &&
          !e.tileLocal(slot)) {
        const uint64_t tex = uint64_t{a.texture.generation} << 32 | a.texture.index;
        const uint32_t key = uint32_t(slot) << 16 | uint32_t(a.keepOrigin) << 8 | uint32_t(e.tile[slot]);
        uint32_t* seen = nullptr;
        if (const auto it = keptStoreSeen_.find(tex); it != keptStoreSeen_.end() || keptStoreSeen_.size() < 1024) seen = &keptStoreSeen_[tex][key];
        if (seen && ++*seen == kKeptStoreSeen) {
          static const char* const kOrigin[] = {"unknown", "client", "upgrade", "required"};
          const ResourceTable::StoreHistory h = resources_.storeHistory(a.texture);
          bool cand = false;
          for (const StoreObs& o : compiled.storeObs) cand = cand || (o.texture == a.texture && o.candidate);
          std::string label = d->label.empty() ? "-" : d->label.substr(0, 40);
          for (char& c : label)
            if (c == ' ') c = '_';
          char buf[400];
          std::snprintf(buf, sizeof buf,
                        "gpu.msaa.kept_store f=%llu rp=%s tex=%u %ux%ux%u s%u fmt=%s slot=%s origin=%s store=%s resolve=%d load=%d why=%s "
                        "dead_cand=%d dead=%u banned=%d shader_readable=%d persistent=%d transient=%d heavy=%d dead_store=%d",
                        (unsigned long long)index, label.c_str(), a.texture.index, d->width, d->height, unsigned(d->layers),
                        unsigned(d->samples), formatInfo(d->format).name, slot == kDepthSlot ? "depth" : "colour",
                        kOrigin[unsigned(a.keepOrigin) & 3], a.store == StoreAction::Store ? "S" : "SR", a.resolve ? 1 : 0, int(a.load),
                        toString(e.tile[slot]), cand ? 1 : 0, unsigned(h.dead), h.banned ? 1 : 0, d->shaderReadable ? 1 : 0,
                        d->persistent ? 1 : 0, d->transient ? 1 : 0, e.heavy >= 1.0f ? 1 : 0, compile_.msaaDeadStore && !compile_.msaaDeadStoreHalt ? 1 : 0);
          lines.emplace_back(buf);
        }
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

FrameReport Device::submit(const Frame& frame) {
  const uint64_t index = frame_.fetch_add(1, std::memory_order_relaxed);
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
  const CompiledFrame compiled = compileFrame(frame, &resources_, compile_);
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
  std::vector<std::string> memLines;
  {
    std::lock_guard lock(resMutex_);  // memValid_ is part of the resource table: createTexture/destroy run under resMutex_
    commitMemory(compiled, resources_);
    trackRenderTargets(index, compiled, memLines);
  }
  for (const std::string& l : memLines) log_(l);

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
  deadStoreMisses_ += uint32_t(compiled.elideMisses.size());
  if (compile_.msaaDeadStore && !compile_.msaaDeadStoreHalt && deadStoreMisses_ >= kDeadStoreMissLimit) {  // the guess keeps failing: stop guessing
    compile_.msaaDeadStoreHalt = true;  // (already elided stores keep being watched)
    log_("gpu.msaa dead_store=0 (" + std::to_string(deadStoreMisses_) + " elided stores were needed: switched off)");
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
  if (detailEvery_ && index % detailEvery_ == 0) {
    for (const std::string& line : formatEncoders(index, frame, compiled)) log_(line);
    std::vector<std::string> fuse;
    {
      std::lock_guard lock(resMutex_);
      fuse = formatPassFuse(index, frame, compiled, &resources_);
      for (std::string& line : formatResolveUse(index, frame, compiled, &resources_)) fuse.push_back(std::move(line));
    }
    for (const std::string& line : fuse) log_(line);
  }
  if (frame.passDump()) {  // VPVK_PASS_DUMP (set by the Vulkan layer for a few frames): this frame pass by pass; its GPU times follow
    const bool detailed = detailEvery_ && index % detailEvery_ == 0;  // gpu.enc / gpu.pass.fuse / gpu.resolve.use were just logged
    std::vector<std::string> dump;
    {
      std::lock_guard lock(resMutex_);
      dump = formatPassDump(index, frame, compiled, &resources_, &pipelines_);
      for (const std::string& l : frame.dumpLines()) dump.push_back(l);
      if (!detailed) {
        for (std::string& l : formatEncoders(index, frame, compiled)) dump.push_back(std::move(l));
        for (std::string& l : formatPassFuse(index, frame, compiled, &resources_)) dump.push_back(std::move(l));
        for (std::string& l : formatResolveUse(index, frame, compiled, &resources_)) dump.push_back(std::move(l));
      }
    }
    capPassDump(index, dump);
    for (const std::string& l : dump) log_(l);
    if (pdumpPending_.size() < 8) pdumpPending_.emplace_back(index, passDumpEncoders(frame, compiled));
  }
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
      std::vector<uint32_t> rpOf(frame.commands().size());  // ordinal of each render pass among the frame's render passes
      for (uint32_t c = 0, n = 0; c < rpOf.size(); ++c)
        if (std::holds_alternative<RenderPass>(frame.commands()[c])) rpOf[c] = n++;
      for (size_t i = 0; i < compiled.encoders.size(); ++i) {
        const Encoder& e = compiled.encoders[i];
        char buf[256];
        if (e.kind == Encoder::Kind::Render) {
          const Attachment& a = e.pass.colors[0].texture ? e.pass.colors[0] : e.pass.depth;
          const TextureDesc* td = a.texture ? resources_.get(a.texture) : nullptr;
          uint32_t draws = 0;
          std::string label;
          std::vector<uint32_t> ords;  // the render passes the encoder runs
          for (uint32_t c : e.commands)
            if (const auto* rp = std::get_if<RenderPass>(&frame.commands()[c])) {
              draws += uint32_t(rp->draws.size());
              if (label.empty()) label = rp->desc.label;
              ords.push_back(rpOf[c]);
            }
          for (char& ch : label)
            if (ch == ' ') ch = '_';
          std::sort(ords.begin(), ords.end());
          std::string rps;  // "3" / "3-5,8": runs of consecutive ordinals (the first 8 runs)
          for (size_t k = 0, runs = 0; k < ords.size(); ++runs) {
            size_t end = k;
            while (end + 1 < ords.size() && ords[end + 1] <= ords[end] + 1) ++end;
            if (runs == 8) {
              rps += ",..";
              break;
            }
            rps += (rps.empty() ? "" : ",") + std::to_string(ords[k]) + (ords[end] != ords[k] ? "-" + std::to_string(ords[end]) : "");
            k = end + 1;
          }
          std::snprintf(buf, sizeof buf, "pass=%zu kind=render label=%s fmt=%s %ux%ux%u samples=%u draws=%u merged=%u rp=%s",
                        i, label.empty() ? "-" : label.substr(0, 40).c_str(),
                        td ? formatInfo(td->format).name : "-", td ? mipShift(td->width, a.mip) : 0u,
                        td ? mipShift(td->height, a.mip) : 0u, unsigned(e.pass.layerCount), td ? unsigned(td->samples) : 0u,
                        draws, e.passes, rps.empty() ? "-" : rps.c_str());
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
    while (!pdumpPending_.empty() && pdumpPending_.front().first < done) pdumpPending_.pop_front();
    if (!pdumpPending_.empty() && pdumpPending_.front().first == done) {
      log_(formatPassDumpTime(done, pdumpPending_.front().second, t));
      pdumpPending_.pop_front();
    }
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
    // Slow frames: all of the first 200, then one in four (a game that runs slow throughout, ITR2, made
    // this the loudest tag; gpu.sec above keeps p50/p95/max/over10 of every frame, so nothing is lost).
    if (!(done < 300 || done % 90 == 0)) {
      if (!(t.frameMs > 12.0)) continue;
      if (slowLogged_++ >= 200 && (slowLogged_ & 3)) continue;
    }
    char buf[64];
    std::snprintf(buf, sizeof buf, "gpu.time f=%llu gpu_ms=%.3f", (unsigned long long)done, t.frameMs);
    std::string line = buf;
    // Slow frames carry each encoder's GPU time (submission order) so the costly pass shows
    // without the "Diagnóstico de rendimiento" switch. Every 2nd logged slow frame (1 in 8 of the slow ones), to keep the log small.
    if (t.frameMs > 12.0 && !t.encoderMs.empty() && (slowSeen_++ % 2) == 0) {
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
  return report;
}

}  // namespace vpgpu
