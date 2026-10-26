// "vk.perf": one log line per second saying where a frame's time goes, so ONE log tells whether the FPS limit is the GPU,
// the CPU translating, pipeline compilation or waits. Counters are relaxed atomics (no locks on the hot path); the
// line is built by the submit thread (entry.cpp perfTick) from the interval since the previous one.
//
// Time nesting per frame sent (vkQueueSubmit):
//   submit_ms  = translate_ms + pipeline gate wait + backend_ms + bookkeeping
//   backend_ms = vpgpu::Device::submit = compile_ms (frame compiler) + hold (waiting for pipelines) + Metal encoding,
//                commit and the wait for a free frame slot (a full ring = the GPU is behind)
// gpu_cb is the GPU's own time (MTLCommandBuffer GPUEndTime - GPUStartTime) per command buffer, summed over the
// interval: sum_ms near dt_ms = GPU bound; submit_ms near the frame period with a small gpu sum = CPU bound.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>

#include "vpgpu/core/perf.h"

namespace vpvk {

// Fields owned by PerfCounters (summed over the interval; *Max: the largest single value).
#define VPVK_PERF_COUNTED(X)                                                                                         \
  X(frames) X(submitUs) X(submitUsMax) X(translateUs) X(translateUsMax) X(backendUs) X(backendUsMax) X(compileUs)    \
  X(compileUsMax) X(dispatches) X(draws) X(drawsInstanced) X(indirectGpu) X(indirectCpu) X(indirectCpuSecondary)     \
  X(barriers) X(barriersComputeDraw) X(fenceWaits) X(fenceUs) X(fenceUsMax) X(idleWaits) X(idleUs) X(idleUsMax)

struct PerfSample {
#define VPVK_X(n) uint64_t n = 0;
  VPVK_PERF_COUNTED(VPVK_X)
#undef VPVK_X
  // Filled by the reader from other counters (vpgpu::gpuPerf, vpvk::pipelineCounters, vpgpu::Device::holdTotalMs).
  uint64_t gpuCbs = 0, gpuCbUs = 0, gpuCbUsMax = 0;
  uint64_t psoCreateUs = 0, psoRecordUs = 0, psoGateUs = 0, psoHoldUs = 0, psoHardWaits = 0;
};

struct PerfCounters {
#define VPVK_X(n) std::atomic<uint64_t> n{0};
  VPVK_PERF_COUNTED(VPVK_X)
#undef VPVK_X
  // Moves the interval into `s` and starts a new one.
  void take(PerfSample& s) {
#define VPVK_X(n) s.n = n.exchange(0, std::memory_order_relaxed);
    VPVK_PERF_COUNTED(VPVK_X)
#undef VPVK_X
  }
};

inline PerfCounters& perf() {
  static PerfCounters c;
  return c;
}

inline void perfAdd(std::atomic<uint64_t>& c, uint64_t v) { c.fetch_add(v, std::memory_order_relaxed); }

inline uint64_t perfUsSince(std::chrono::steady_clock::time_point t0) {
  return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
}

// Times a wait the game's thread blocks in (vkWaitForFences, vkQueueWaitIdle): only waits of at least kBlockedUs count
// (a poll with timeout 0 that returns at once is not a wait).
class PerfWait {
 public:
  static constexpr uint64_t kBlockedUs = 100;
  PerfWait(std::atomic<uint64_t>& n, std::atomic<uint64_t>& us, std::atomic<uint64_t>& max)
      : n_(n), us_(us), max_(max), t0_(std::chrono::steady_clock::now()) {}
  ~PerfWait() {
    const uint64_t e = perfUsSince(t0_);
    if (e < kBlockedUs) return;
    perfAdd(n_, 1);
    perfAdd(us_, e);
    vpgpu::raiseMax(max_, e);
  }
  PerfWait(const PerfWait&) = delete;
  PerfWait& operator=(const PerfWait&) = delete;

 private:
  std::atomic<uint64_t>& n_;
  std::atomic<uint64_t>& us_;
  std::atomic<uint64_t>& max_;
  std::chrono::steady_clock::time_point t0_;
};

// Cuts the intervals: tick() gives the line once at least a second passed since the previous one (else ""). `ext` carries
// the CUMULATIVE pipeline counters (pso*Us = vpvk::pipelineCounters blockedCreate/Record/Gate, psoHoldUs = hold total,
// psoHardWaits); their delta since the previous line is what the line shows. The first call only starts the clock.
// Single reader (the submit thread, under submitMutex).
class PerfLine {
 public:
  using Clock = std::chrono::steady_clock;
  std::string tick(Clock::time_point now, const PerfSample& ext, PerfCounters& c = perf(),
                   vpgpu::GpuPerf& g = vpgpu::gpuPerf());

 private:
  bool started_ = false;
  Clock::time_point last_{};
  PerfSample prev_;
};

// The "vk.perf" line for an interval of `dtMs` (keys never change meaning; groups are key{k=v k=v}, like vk.submit).
inline std::string formatPerf(const PerfSample& s, double dtMs) {
  auto ms = [](uint64_t us) { return double(us) / 1000.0; };
  auto avg = [&](uint64_t us, uint64_t n) { return n ? ms(us) / double(n) : 0.0; };
  char b[1024];
  std::snprintf(b, sizeof b,
                "vk.perf dt_ms=%.0f frames=%llu submit_ms{avg=%.2f max=%.2f} translate_ms{avg=%.2f max=%.2f} "
                "backend_ms{avg=%.2f max=%.2f} compile_ms{avg=%.2f max=%.2f} gpu_cb{n=%llu avg_ms=%.2f max_ms=%.2f sum_ms=%.1f} "
                "dispatches=%llu draws=%llu draws_inst=%llu ind{gpu=%llu cpu=%llu cpu_sec=%llu} barriers=%llu barr_c2d=%llu "
                "pso_wait_ms{create=%.1f record=%.1f gate=%.1f hold=%.1f} hard_waits=%llu "
                "fence_wait{n=%llu ms=%.1f max=%.1f} idle_wait{n=%llu ms=%.1f max=%.1f}",
                dtMs, (unsigned long long)s.frames, avg(s.submitUs, s.frames), ms(s.submitUsMax),
                avg(s.translateUs, s.frames), ms(s.translateUsMax), avg(s.backendUs, s.frames), ms(s.backendUsMax),
                avg(s.compileUs, s.frames), ms(s.compileUsMax), (unsigned long long)s.gpuCbs, avg(s.gpuCbUs, s.gpuCbs),
                ms(s.gpuCbUsMax), ms(s.gpuCbUs), (unsigned long long)s.dispatches, (unsigned long long)s.draws,
                (unsigned long long)s.drawsInstanced, (unsigned long long)s.indirectGpu, (unsigned long long)s.indirectCpu,
                (unsigned long long)s.indirectCpuSecondary, (unsigned long long)s.barriers,
                (unsigned long long)s.barriersComputeDraw, ms(s.psoCreateUs), ms(s.psoRecordUs), ms(s.psoGateUs),
                ms(s.psoHoldUs), (unsigned long long)s.psoHardWaits, (unsigned long long)s.fenceWaits, ms(s.fenceUs),
                ms(s.fenceUsMax), (unsigned long long)s.idleWaits, ms(s.idleUs), ms(s.idleUsMax));
  return b;
}

inline std::string PerfLine::tick(Clock::time_point now, const PerfSample& ext, PerfCounters& c, vpgpu::GpuPerf& g) {
  PerfSample s;
  if (!started_) {  // the first interval starts here (counters from before it ride along: a few loading-time waits)
    started_ = true, last_ = now, prev_ = ext;
    return {};
  }
  const double dtMs = std::chrono::duration<double, std::milli>(now - last_).count();
  if (dtMs < 1000.0) return {};
  c.take(s);
  s.gpuCbs = g.cbs.exchange(0, std::memory_order_relaxed);
  s.gpuCbUs = g.cbUs.exchange(0, std::memory_order_relaxed);
  s.gpuCbUsMax = g.cbUsMax.exchange(0, std::memory_order_relaxed);
  s.psoCreateUs = ext.psoCreateUs >= prev_.psoCreateUs ? ext.psoCreateUs - prev_.psoCreateUs : 0;  // a new device or a reset counter: no wrap
  s.psoRecordUs = ext.psoRecordUs >= prev_.psoRecordUs ? ext.psoRecordUs - prev_.psoRecordUs : 0;  // a new device or a reset counter: no wrap
  s.psoGateUs = ext.psoGateUs >= prev_.psoGateUs ? ext.psoGateUs - prev_.psoGateUs : 0;  // a new device or a reset counter: no wrap
  s.psoHoldUs = ext.psoHoldUs >= prev_.psoHoldUs ? ext.psoHoldUs - prev_.psoHoldUs : 0;  // a new device or a reset counter: no wrap
  s.psoHardWaits = ext.psoHardWaits >= prev_.psoHardWaits ? ext.psoHardWaits - prev_.psoHardWaits : 0;  // a new device or a reset counter: no wrap
  prev_ = ext;
  last_ = now;
  return formatPerf(s, dtMs);
}

}  // namespace vpvk
