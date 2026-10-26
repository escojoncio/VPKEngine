// Process-wide counters for the "vk.perf" line (gpu/vulkan): relaxed atomics, no locks, safe from any thread (Metal
// completion handlers included). The reader takes the interval with exchange(0).
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

namespace vpgpu {

inline void raiseMax(std::atomic<uint64_t>& m, uint64_t v) {
  uint64_t cur = m.load(std::memory_order_relaxed);
  while (v > cur && !m.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
  }
}

// GPU time of the command buffers that completed (MTLCommandBuffer GPUEndTime - GPUStartTime in gpu/metal2), in
// microseconds: how many, their sum and the longest one since the last take.
struct GpuPerf {
  std::atomic<uint64_t> cbs{0}, cbUs{0}, cbUsMax{0};
  void addCb(uint64_t us) {
    cbs.fetch_add(1, std::memory_order_relaxed);
    cbUs.fetch_add(us, std::memory_order_relaxed);
    raiseMax(cbUsMax, us);
  }
};
inline GpuPerf& gpuPerf() {
  static GpuPerf p;
  return p;
}

// "vk.hitch" (gpu/vulkan perf.h HitchLine): CUMULATIVE microsecond counters, never reset; the reader takes the difference
// between two frames. Written from any thread (guest, gles, Metal compile threads). The categories overlap (a gles link
// that creates pipelines counts in both) and mean "time inside the call", not "time blocked".
struct HitchPerf {
  std::atomic<uint64_t> psoCreateUs{0}, libCompileUs{0}, waitUs{0}, allocUs{0}, glLinkUs{0}, lockUs{0}, engineUs{0};
  struct Snap {
    uint64_t psoCreateUs, libCompileUs, waitUs, allocUs, glLinkUs, lockUs, engineUs;
  };
  Snap snap() const {
    return {psoCreateUs.load(std::memory_order_relaxed), libCompileUs.load(std::memory_order_relaxed),
            waitUs.load(std::memory_order_relaxed),      allocUs.load(std::memory_order_relaxed),
            glLinkUs.load(std::memory_order_relaxed),    lockUs.load(std::memory_order_relaxed),
            engineUs.load(std::memory_order_relaxed)};
  }
};
inline HitchPerf& hitchPerf() {
  static HitchPerf h;
  return h;
}

// Times its scope into `us`. `engine`: the scope is a call into the engine made by the game's thread, so its outermost span
// per thread also adds to engineUs (nested spans are not summed twice); false for work on the engine's own threads
// (Metal compiles), which is not time the game's thread loses.
class HitchSpan {
 public:
  explicit HitchSpan(std::atomic<uint64_t>& us, bool engine = true)
      : us_(us), engine_(engine), t0_(std::chrono::steady_clock::now()) {
    if (engine_) ++depth();
  }
  ~HitchSpan() {
    const uint64_t e = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0_).count());
    us_.fetch_add(e, std::memory_order_relaxed);
    if (engine_ && --depth() == 0) hitchPerf().engineUs.fetch_add(e, std::memory_order_relaxed);
  }
  HitchSpan(const HitchSpan&) = delete;
  HitchSpan& operator=(const HitchSpan&) = delete;

 private:
  static int& depth() {
    static thread_local int d = 0;
    return d;
  }
  std::atomic<uint64_t>& us_;
  bool engine_;
  std::chrono::steady_clock::time_point t0_;
};

}  // namespace vpgpu
