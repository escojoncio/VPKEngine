// Process-wide counters for the "vk.perf" line (gpu/vulkan): relaxed atomics, no locks, safe from any thread (Metal
// completion handlers included). The reader takes the interval with exchange(0).
#pragma once

#include <atomic>
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

}  // namespace vpgpu
