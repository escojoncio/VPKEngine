// The client-facing entry point: owns the resource table, forwards to the backend, compiles and
// submits frames, and produces the per-frame log lines.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "vpgpu/core/backend.h"
#include "vpgpu/core/diagnostics.h"
#include "vpgpu/core/label_queue.h"
#include "vpgpu/core/pipeline_cache.h"

namespace vpgpu {

class Device {
 public:
  using LogSink = std::function<void(const std::string&)>;

  Device(Backend&, unsigned pipelineWorkers, LogSink log);
  ~Device();

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
  // The same, taking the frame over (no copy when a submit thread is running).
  FrameReport submit(Frame&&);
  void setDetailEvery(uint64_t n) { detailEvery_ = n; }
  // Diagnostic: one "gpu.passtime" line per encoder (GPU ms from stage-boundary timestamps) for
  // one rendering frame every `every` frames (0 = off). Also enabled by env VP_PASS_TIMING=1.
  void setPassTiming(uint64_t every) { passEvery_ = every; }
  // Frame hold (see holdMs_): 0 = never hold a frame for compiling pipelines (the client decides what to do with
  // draws whose pipeline is not ready). Set before the first submit.
  void setHoldMs(uint32_t ms) { holdMs_ = ms; }
  // Two-regime hold (opt-in; by default the hold is the single budget above, with its back-off after a timeout):
  // the first `setHoldWarmupFrames` frames (or until the warm-up spent kHoldWarmupMaxMs held) wait up to holdMs_ for
  // every pipeline; afterwards draws wait at most `setHoldSteadyMs` (Dispatch work keeps the whole holdMs_) and a draw
  // whose pipeline is not ready is left out of that frame only. Calling either setter, or setting env
  // VPKENGINE_HOLD_MS / VPKENGINE_HOLD_WARMUP_FRAMES, turns the regime on; the one not given keeps 12 ms / 120 frames.
  // Set before the first submit.
  void setHoldSteadyMs(uint32_t ms) { holdSteadyMs_ = ms, holdTwoRegime_ = true; }
  void setHoldWarmupFrames(uint32_t frames) { holdWarmupFrames_ = frames, holdTwoRegime_ = true; }
  // Opt-in (default off: createBuffer returns the id even if the backend left no buffer behind it, as it always did):
  // a buffer the backend refused (Backend::bufferCreated false) is destroyed and createBuffer returns an invalid id, so
  // the client can tell it has nothing to bind. Logged once per (size, label), 64 lines at most. Set before the first
  // createBuffer.
  void setRefuseFailedBuffers(bool on) { refuseFailedBuffers_ = on; }
  // The hold in force for the next frame's draws: holdMs_ (default regime), or in the two-regime hold the warm-up's,
  // then the steady budget. 0 = frames are not held (a draw whose pipeline is not ready is not even recorded).
  uint32_t holdMs() const {
    return !holdTwoRegime_ || holdWarm(frame_) ? holdMs_ : std::min(holdMs_, holdSteadyMs_);
  }
  // The hold for compute work (Dispatch, ES/GS expansion): always the whole holdMs_, warm-up or not, because a
  // Dispatch left out leaves its outputs unwritten (see Device::takeFrame). 0 = not recorded either.
  uint32_t holdComputeMs() const { return holdMs_; }
  // Total ms submit() has held frames for compiling pipelines (written by submit(): read it from the submit thread).
  double holdTotalMs() const { return holdTotalMs_; }

  // Where the caller's time in submit() went since the last call (the "submit phases" of the CPU report).
  // `backendMs` is the backend's submit (encode + commit, and the wait for a frame slot): on the caller's
  // thread, or on the submit thread when there is one. `slotWaitMs` is the part of it spent waiting for a frame slot.
  struct SubmitTimes {
    double holdMs = 0, compileMs = 0, backendMs = 0, slotWaitMs = 0;
    uint64_t holdFrames = 0, holdTimeouts = 0;  // frames that waited for pipelines / of them, those that gave up
  };
  SubmitTimes takeSubmitTimes();
  // Frames with an index below this one have finished on the GPU (see Backend::completedFrames).
  uint64_t completedFrames() const { return backend_.completedFrames(); }
  // How many frames submit() has taken (the next frame's index). With a submit thread they may not
  // have reached the backend yet; completedFrames() still only counts finished ones.
  uint64_t submittedFrames() const { return frame_; }
  // gpu.mem.rt (render-target memory by the frame compiler's verdicts): every `ms` (default 10000, 0 = every frame), logged
  // only when its numbers changed; the first line comes one period after the Device was made, or at once after this call.
  void setRtMemLogMs(uint32_t ms) { rtMemMs_ = ms, rtMemNext_ = {}; }

  // Partial fence: blocks until every frame with an index below `frames` has finished on the GPU
  // (frames after it keep running). Waits for the submit thread to hand them over first. False on
  // timeout / lost device. waitIdle() = all frames submitted so far.
  bool waitFrames(uint64_t frames, uint64_t timeoutNs = UINT64_MAX);
  bool waitIdle(uint64_t timeoutNs = UINT64_MAX);

  // GPU-ordered side effects: `action` runs, on a completion thread or inside a later Device call,
  // once frames 0..`frame` have finished (immediately if they have and nothing older is queued).
  // [address, +size) is the memory it writes (see LabelQueue). Use the index the work will get
  // (submittedFrames() for work not submitted yet).
  void afterFrame(uint64_t frame, uint64_t address, uint32_t size, std::function<void()> action);
  bool labelsQueued() const { return !labels_.empty(); }
  bool labelPending(uint64_t address, uint32_t size) const { return labels_.pending(address, size); }
  // Frames that must finish before every queued write overlapping the range has run (0 = none).
  uint64_t labelWaitTarget(uint64_t address, uint32_t size) const { return labels_.waitTarget(address, size); }
  // Runs the label writes whose frames finished (also called from the backend's notification).
  // `wait` false (the default): never blocks on a release running on another thread (that thread
  // looks again before it finishes): for the thread that holds the host's engine lock, see
  // LabelQueue::runMutex_. `wait` true: returns when the finished frames' labels have run (the
  // backend's completion thread, and a caller that needs a particular label).
  void pumpLabels(bool wait = false);
  // For the thread that holds the engine lock: runs the label writes of frames below `frames` and waits
  // (polling, never blocking on another thread's release) until they have all run; false on timeout.
  bool waitLabels(uint64_t frames, unsigned timeoutMs = 2000);
  // The backend reported a lost device (Backend::lost): GPU-ordered labels were released for good.
  bool deviceLost() const { return lostNoted_.load(); }

  // Submit thread (opt-in): compile stays on the caller; encoding + committing run on a thread so
  // the recording thread goes on with the next picture. At most `maxQueued` frames wait for it (the
  // caller blocks beyond that: this also bounds the latency). Every backend call made by the Device is serialized
  // by one mutex (withBackend); destroys are ordered after the frames queued before them. Call before
  // the first submit; 0 = none (the default: everything inline).
  void startSubmitThread(unsigned maxQueued);
  bool asyncSubmit() const { return maxQueued_ != 0; }
  // Returns once every frame taken so far was handed to the backend (no-op without the thread).
  void drain();
  // Runs `fn` with the backend call mutex held (native handles, interop encodes): after drain() it
  // is ordered after every frame submitted before. Without a submit thread the Device's own backend calls do not
  // take that mutex (they run in parallel with each other), so it only excludes other withBackend calls. `fn` must not call Device::create*/destroy (they
  // take resMutex_, which goes before backendMutex_).
  void withBackend(const std::function<void()>& fn);

 private:
  struct Task {
    uint64_t index = 0;
    Frame frame;
    CompiledFrame compiled;
    std::vector<std::string> lines, passDesc;
    std::function<void()> call;  // not a frame: runs in order
  };
  FrameReport takeFrame(const Frame&, CompiledFrame&, std::vector<std::string>& lines, std::vector<std::string>& passDesc,
                        uint64_t& index);
  void execute(uint64_t index, const Frame&, const CompiledFrame&, const std::vector<std::string>& lines,
               std::vector<std::string>& passDesc);
  void reportTimings();
  void enqueue(Task&&);
  void runOrdered(std::function<void()> call, uint64_t destroyKey = 0);
  void workerLoop();
  uint64_t destroyKey(char kind, uint32_t index) const { return (uint64_t(kind) << 32 | index) + 1; }
  void settleCreate(char kind, uint32_t index);
  void noteLost();
  uint64_t releasable() const;  // frames below this have finished (everything once the device is lost)
  std::atomic<bool> lostNoted_{false};

  Backend& backend_;
  // resources_ (including ResourceTable::memValid_, written by commitMemory) is written by create/destroy on the client's
  // loading threads and read by compileFrame on the submit thread (a vector-backed pool: a create that grows it moves
  // what compileFrame is reading). Lock order: resMutex_ first, then backendMutex_ (or any backend table lock), never
  // the other way round: nothing that holds backendMutex_ (execute, withBackend callbacks) may take resMutex_.
  std::mutex resMutex_;
  ResourceTable resources_;
  PipelineCache pipelines_;
  LabelQueue labels_;
  // Serializes every backend call the Device makes (create/destroy, submit): the submit thread and the creating threads.
  std::mutex backendMutex_;
  // Taken only while a submit thread exists (maxQueued_ != 0, fixed before the first submit): without one, create/destroy
  // on the loading threads and submit run in parallel, as they always did, and the backend's own tables serialize them.
  std::unique_lock<std::mutex> lockBackend() {
    std::unique_lock<std::mutex> l(backendMutex_, std::defer_lock);
    if (maxQueued_) l.lock();
    return l;
  }
  // Submit thread.
  unsigned maxQueued_ = 0;
  std::thread worker_;
  std::mutex qMutex_;
  std::condition_variable qWake_, qSpace_, qIdle_;
  std::deque<Task> tasks_;
  bool busy_ = false, stop_ = false;
  uint64_t handed_ = 0;  // frames below this index have reached the backend (under qMutex_)
  std::map<uint64_t, int> pendingDestroy_;  // destroyKey -> count, under qMutex_
  LogSink log_;
  bool refuseFailedBuffers_ = false;
  std::mutex refusedLogMutex_;
  std::set<std::pair<uint64_t, std::string>> refusedLogged_;  // (size, label) of the refusals already logged
  uint64_t frame_ = 0, detailEvery_ = 0;
  uint32_t msaaRejectLogs_ = 0;  // gpu.msaa rejected lines written (capped)
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
  };
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
  // Two-regime hold (setHoldSteadyMs / setHoldWarmupFrames / env VPKENGINE_HOLD_*; off by default): the warm-up is the
  // first holdWarmupFrames_ frames (or until it has spent kHoldWarmupMaxMs held, whichever comes first), when a frame
  // waits up to holdMs_ for every pipeline it draws with. Afterwards the draws wait at most holdSteadyMs_ (about one
  // display frame): a pipeline that is not ready by then leaves its draws out of that frame only (the backend counts
  // them as missing_pipelines) and the caller goes on, instead of stalling for the whole hold on every frame that
  // meets a new pipeline. Compute work (Dispatch) always gets holdMs_. No back-off in this regime.
  bool holdTwoRegime_ = false;
  uint32_t holdSteadyMs_ = 12, holdWarmupFrames_ = 120;
  static constexpr double kHoldWarmupMaxMs = 10000;
  bool holdWarm(uint64_t index) const { return index < holdWarmupFrames_ && holdWarmMs_ < kHoldWarmupMaxMs; }
  // Default regime: after a hold that timed out (a cold shader cache: every frame brings pipelines still compiling, build 18
  // held 1.5 s per frame = 0.7 fps), holds are cut to kHoldBackoffMs for a while: draws that are not ready
  // are skipped instead of stalling the game thread frame after frame.
  static constexpr uint32_t kHoldBackoffMs = 15;
  std::chrono::steady_clock::time_point holdBackoffUntil_{};
  uint64_t holdFrames_ = 0, holdTimeouts_ = 0;
  double holdTotalMs_ = 0, holdWarmMs_ = 0;  // (caller's thread only)
  // takeSubmitTimes: ns / counts since the last call (the backend part is written by the submit thread).
  std::atomic<uint64_t> holdNs_{0}, compileNs_{0}, backendNs_{0}, holdFramesT_{0}, holdTimeoutsT_{0};
  // Frames waiting for their timings: index + one descriptor per encoder ("pass=i label=.. fmt=.. WxH draws=..").
  std::deque<std::pair<uint64_t, std::vector<std::string>>> passPending_;
  std::atomic<size_t> passCount_{0};  // passPending_.size(), readable from the caller's thread
  // GPU frame times of the current second (gpu.sec summary line).
  std::vector<float> secMs_;
  std::chrono::steady_clock::time_point secStart_{};
};

}  // namespace vpgpu
