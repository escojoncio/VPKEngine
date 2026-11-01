// Binary-archive harvest policy. The harvest (adding the session's pipelines to the Metal binary archive, which
// recompiles each one in Metal's compiler service) runs continuously on a utility queue and steps aside only while a
// thread of the game is blocked waiting for a pipeline (demandWaiters), unless the queue is starving. Lock free,
// platform neutral (the null platform tests it).
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace vpgpu {

// Threads blocked right now on a pipeline compile a draw needs (awaitPipelines, the frame gate's waits).
inline std::atomic<uint32_t>& demandWaiters() {
  static std::atomic<uint32_t> n{0};
  return n;
}

// Counts the calling thread as a demand waiter for the scope.
class DemandWait {
 public:
  DemandWait() { demandWaiters().fetch_add(1, std::memory_order_acq_rel); }
  ~DemandWait() { demandWaiters().fetch_sub(1, std::memory_order_acq_rel); }
  DemandWait(const DemandWait&) = delete;
  DemandWait& operator=(const DemandWait&) = delete;
};

// Pipeline compiles queued or running in the PipelineCache workers (all caches). The harvest steps aside while it is
// non-zero: both recompile in the same compiler service and hold descriptors in memory (startup burst).
inline std::atomic<uint32_t>& compileBacklog() {
  static std::atomic<uint32_t> n{0};
  return n;
}

inline constexpr size_t kHarvestBatch = 16;         // entries added per step
inline constexpr size_t kHarvestStarvePending = 512;  // above this many queued entries the harvest never yields
inline constexpr std::chrono::seconds kHarvestStarveAge{20};  // nor when the oldest queued entry is older than this

// True when the harvest must step aside now: a demand wait is in progress and the queue is not starving.
inline bool archiveHarvestYields(uint32_t waiters, size_t pending, std::chrono::nanoseconds oldestAge) {
  if (!waiters) return false;
  return pending <= kHarvestStarvePending && oldestAge <= kHarvestStarveAge;
}

// The full decision: yield while compiles are queued (compileBacklog), while free memory is under minFreeMB (0 disables
// the check), or for a demand wait (archiveHarvestYields). Low memory always yields; the backlog and the demand wait step
// aside only while the queue is not starving (pending <= kHarvestStarvePending and oldestAge <= kHarvestStarveAge).
inline bool archiveHarvestShouldYield(uint32_t waiters, uint32_t backlog, size_t pending, std::chrono::nanoseconds oldestAge,
                                      uint64_t freeMB, uint64_t minFreeMB) {
  if (minFreeMB && freeMB < minFreeMB) return true;
  return archiveHarvestYields(waiters + backlog, pending, oldestAge);
}

}  // namespace vpgpu
