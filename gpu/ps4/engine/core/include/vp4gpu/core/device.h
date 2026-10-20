// The client-facing entry point: owns the resource table, forwards to the backend, compiles and
// submits frames, and produces the per-frame log lines.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "vp4gpu/core/backend.h"
#include "vp4gpu/core/diagnostics.h"
#include "vp4gpu/core/label_queue.h"
#include "vp4gpu/core/pipeline_cache.h"

namespace vp4gpu {

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
  const ResourceTable& resources() const { return resources_; }
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
  uint32_t holdMs() const { return holdMs_; }
  // Frames with an index below this one have finished on the GPU (see Backend::completedFrames).
  uint64_t completedFrames() const { return backend_.completedFrames(); }
  // How many frames submit() has taken (the next frame's index). With a submit thread they may not
  // have reached the backend yet; completedFrames() still only counts finished ones.
  uint64_t submittedFrames() const { return frame_; }

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
  // The backend reported a lost device (Backend::lost): GPU-ordered labels were released for good.
  bool deviceLost() const { return lostNoted_.load(); }

  // Submit thread (opt-in): compile stays on the caller; encoding + committing run on a thread so
  // the recording thread goes on with the next picture. At most `maxQueued` frames wait for it (the
  // caller blocks beyond that: this also bounds the latency). Every backend call is serialized by
  // one mutex (withBackend); destroys are ordered after the frames queued before them. Call before
  // the first submit; 0 = none (the default: everything inline).
  void startSubmitThread(unsigned maxQueued);
  bool asyncSubmit() const { return maxQueued_ != 0; }
  // Returns once every frame taken so far was handed to the backend (no-op without the thread).
  void drain();
  // Runs `fn` with the backend call mutex held (native handles, interop encodes): after drain() it
  // is ordered after every frame submitted before.
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
  LabelQueue labels_;
  std::mutex backendMutex_;
  // Submit thread.
  unsigned maxQueued_ = 0;
  std::thread worker_;
  std::mutex qMutex_;
  std::condition_variable qWake_, qSpace_, qIdle_;
  std::deque<Task> tasks_;
  bool busy_ = false, stop_ = false;
  uint64_t handed_ = 0;  // frames below this index have reached the backend (under qMutex_)
  std::map<uint64_t, int> pendingDestroy_;  // destroyKey -> count, under qMutex_
  ResourceTable resources_;
  PipelineCache pipelines_;
  LogSink log_;
  uint64_t frame_ = 0, detailEvery_ = 0;
  uint64_t passEvery_ = 0, passNext_ = 0;
  // Hold a frame (block the submit) while pipelines it draws with are still compiling, up to this
  // many ms (env VP_HOLD_MS, default 1500; 0 = draw what is ready, the old behaviour). A held
  // frame is one the host compositor never sees half-drawn: it keeps reprojecting the last
  // complete one, which in VR reads as a brief pause instead of corrupted geometry.
  uint32_t holdMs_ = 1500;
  uint64_t holdFrames_ = 0, holdTimeouts_ = 0;
  double holdTotalMs_ = 0;
  // Frames waiting for their timings: index + one descriptor per encoder ("pass=i label=.. fmt=.. WxH draws=..").
  std::deque<std::pair<uint64_t, std::vector<std::string>>> passPending_;
  std::atomic<size_t> passCount_{0};
  // GPU frame times of the current second (gpu.sec summary line).
  std::vector<float> secMs_;
  std::chrono::steady_clock::time_point secStart_{};
};

}  // namespace vp4gpu
