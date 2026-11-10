// The client-facing entry point: owns the resource table, forwards to the backend, compiles and
// submits frames, and produces the per-frame log lines.
#pragma once

#include <chrono>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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
  const ResourceTable& resources() const { return resources_; }  // submit-thread reads only: prefer textureDesc()
  // A copy of a texture's description, taken under the table lock (safe from any thread).
  bool textureDesc(TextureId id, TextureDesc& out) {
    std::lock_guard lock(resMutex_);
    const TextureDesc* d = resources_.get(id);
    if (!d) return false;
    out = *d;
    return true;
  }
  // A TextureDesc::transient texture the client needs the contents of outside what a Frame shows the compiler (VPKTAA reads
  // the eye depth after the frame): from now on an ordinary texture, as if a frame had caught it (ResourceTable::transientViolated).
  void markTransientViolated(TextureId id) {
    std::lock_guard lock(resMutex_);
    resources_.setTransientViolated(id);
  }
  uint64_t bufferAddress(BufferId id) const { return backend_.bufferAddress(id); }
  PipelineCache& pipelines() { return pipelines_; }

  // Compiles and submits a frame and logs its summary; every `detailEvery`-th frame (0 = never)
  // also logs one line per encoder. GPU timings of finished frames are logged as they arrive.
  FrameReport submit(const Frame&);
  void setDetailEvery(uint64_t n) { detailEvery_ = n; }
  // Diagnostic: one "gpu.passtime" line per encoder (GPU ms from stage-boundary timestamps) for
  // one rendering frame every `every` frames (0 = off). Also enabled by env VP_PASS_TIMING=1.
  void setPassTiming(uint64_t every) { passEvery_ = every; }
  // Frame hold (see holdMs_): 0 = never hold a frame for compiling pipelines (the client decides what to do with
  // draws whose pipeline is not ready). Set before the first submit.
  // CompileOptions::tileMaxDraws as this Device was started (env VPVK_TILE_MAX_DRAWS): the heavy-encoder draw limit.
  uint32_t tileMaxDraws() const { return compile_.tileMaxDraws; }
  void setHoldMs(uint32_t ms) { holdMs_ = ms; }
  uint32_t holdMs() const { return holdMs_; }
  // Total ms submit() has held frames for compiling pipelines (written by submit(): read it from the submit thread).
  double holdTotalMs() const { return holdTotalMs_; }
  // Frames with an index below this one have finished on the GPU (see Backend::completedFrames).
  uint64_t completedFrames() const { return backend_.completedFrames(); }
  // How many frames submit() has handed to the backend (the next frame's index).
  uint64_t submittedFrames() const { return frame_; }
  // gpu.mem.rt (render-target memory by the frame compiler's verdicts): every `ms` (default 10000, 0 = every frame), logged
  // only when its numbers changed; the first line comes one period after the Device was made, or at once after this call.
  void setRtMemLogMs(uint32_t ms) { rtMemMs_ = ms, rtMemNext_ = {}; }

 private:
  Backend& backend_;
  // resources_ (including ResourceTable::memValid_, written by commitMemory) is written by create/destroy on the client's
  // loading threads and read by compileFrame on the submit thread (a vector-backed pool: a create that grows it moves
  // what compileFrame is reading). Taken before any backend table lock, never the other way round.
  std::mutex resMutex_;
  ResourceTable resources_;
  PipelineCache pipelines_;
  LogSink log_;
  uint64_t frame_ = 0, detailEvery_ = 0;
  uint32_t nonMemorylessSeen_ = 0;  // bit per TileVerdict already reported by gpu.msaa.nonmemoryless
  uint32_t deadStoreMisses_ = 0;    // elided MSAA stores a later frame needed (CompileOptions::msaaDeadStore is switched off at a limit)
  // times each texture kept its MSAA store per (slot, origin, verdict) key (gpu.msaa.kept_store); a texture's entry goes with it
  std::unordered_map<uint64_t, std::unordered_map<uint32_t, uint32_t>> keptStoreSeen_;
  uint32_t msaaRejectLogs_ = 0;  // gpu.msaa rejected lines written (capped)
  uint32_t splitSkipLogs_ = 0;   // gpu.tile split_skip lines written (the first 8, then one frame in 600)
  CompileOptions compile_;  // copyRects: env VPVK_COPY_RECT_HAZARD=1
  uint32_t bigLogged_ = 0;  // gpu.bigframe lines written (capped)
  int splitMode_ = 0;       // CompileOptions::splitViews as the last frame saw it (off at the start); a change is logged
  uint64_t tileStart_ = 0, tileLogged_ = 0;  // CompileOptions::tileMaxVertices at the start (env), and as the last frame saw it (a change is logged)
  std::unordered_set<uint64_t> splitLogged_;  // Encoder::splitSig of the gpu.split lines written (one per kind of pass)
  // gpu.mem.rt: every attachment texture with what the last frame that attached it decided (tile-local = memoryless, else it
  // needs memory: `why` is the verdict). The compiler's view, not the backend's: Metal frees a materialised target only after
  // VPVK_MSAA_RELEASE_FRAMES tile-local frames (gpu.mem ledger: resident/msaa_mat/rt_mat/memoryless is the allocator's truth).
  struct RtMem {
    uint64_t bytes = 0, frame = ~uint64_t{0};
    bool memoryless = false;
    TileVerdict why = TileVerdict::None;
    StoreAction store = StoreAction::Store;  // the last encoder of that frame that attaches it: its compiled store (gpu.msaa_stale)
    bool keepSamples = false;
  };
  uint32_t msaaStaleLogs_ = 0;
  std::unordered_map<uint64_t, RtMem> rtMem_;
  uint32_t rtMemMs_ = 10000, transientLogs_ = 0;
  uint64_t transientViolations_ = 0;  // TextureDesc::transient textures caught needing memory (TransientViolation), cumulative
  std::chrono::steady_clock::time_point rtMemNext_{};
  std::string rtMemLast_;
  // Under resMutex_ after commitMemory: updates rtMem_ from `compiled`, appends the gpu.mem.* lines to log.
  void trackRenderTargets(uint64_t frame, const CompiledFrame& compiled, std::vector<std::string>& lines);
  uint64_t passEvery_ = 0, passNext_ = 0;
  // Hold a frame (block the submit) while pipelines it draws with are still compiling, up to this
  // many ms (env VP_HOLD_MS, default 1500; 0 = draw what is ready, the old behaviour). A held
  // frame is one the host compositor never sees half-drawn: it keeps reprojecting the last
  // complete one, which in VR reads as a brief pause instead of corrupted geometry.
  uint32_t holdMs_ = 1500;
  // After a hold that timed out (a cold shader cache: every frame brings pipelines still compiling, build 18
  // held 1.5 s per frame = 0.7 fps), holds are cut to kHoldBackoffMs for a while: draws that are not ready
  // are skipped instead of stalling the game thread frame after frame.
  static constexpr uint32_t kHoldBackoffMs = 15;
  std::chrono::steady_clock::time_point holdBackoffUntil_{};
  uint64_t holdFrames_ = 0, holdTimeouts_ = 0;
  double holdTotalMs_ = 0;
  // Frames waiting for their timings: index + one descriptor per encoder ("pass=i label=.. fmt=.. WxH draws=..").
  std::deque<std::pair<uint64_t, std::vector<std::string>>> passPending_;
  // GPU frame times of the current second (gpu.sec summary line).
  std::vector<float> secMs_;
  std::chrono::steady_clock::time_point secStart_{};
};

}  // namespace vpgpu
