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

// ---- flush and serialize policy (the Metal backend's saveArchive / archiveSaveDue use these) ---------------------------------
// A flush adds the queue step by step (each add recompiles the pipeline in Metal's compiler service, ~80 ms) and serializes as it
// goes. At the end of the process or in the background it is bounded (the app may be killed, and what is left stays queued: the
// recipes bring it back next run) but ~100 entries fit in that budget, so a flush that is the only thing archiving (the harvest
// steps aside the whole session) leaves thousands of misses to compile again at every launch. The explicit "prepare the game"
// step (vpvk_prewarm, the host's replay drain) has nobody waiting and must leave the archive complete: a generous bound.
inline constexpr std::chrono::milliseconds kFlushBudgetExit{8000};
inline constexpr std::chrono::milliseconds kFlushBudgetComplete{600000};
inline constexpr std::chrono::milliseconds archiveFlushBudget(bool complete) {
  return complete ? kFlushBudgetComplete : kFlushBudgetExit;
}
// Another flush step: the last one added something and the budget is not spent.
inline bool archiveFlushContinues(size_t lastStepAdded, std::chrono::milliseconds elapsed, std::chrono::milliseconds budget) {
  return lastStepAdded != 0 && elapsed < budget;
}

// A complete flush under memory pressure (free memory under the harvest's threshold; 0 disables) waits this long before its next
// step instead of skipping entries or running flat out; 0 = no pause.
inline constexpr std::chrono::milliseconds kFlushLowMemoryPause{200};
inline constexpr std::chrono::milliseconds archiveFlushPause(bool complete, uint64_t freeMB, uint64_t minFreeMB) {
  return complete && minFreeMB && freeMB < minFreeMB ? kFlushLowMemoryPause : std::chrono::milliseconds{0};
}

inline constexpr uint32_t kSerializeEvery = 32;                    // serialize after this many additions
inline constexpr std::chrono::seconds kSerializeMaxAge{30};        // or when the oldest unsaved addition is this old
// A serialize is due: entries were added since the last one and 32 of them are waiting, the session file is full
// (archivePackCap: a sealed pack is written at once), or 30 s passed.
inline bool archiveSerializeDue(uint32_t unsaved, size_t fileEntries, size_t packCap, std::chrono::nanoseconds sinceSave) {
  return unsaved != 0 && (unsaved >= kSerializeEvery || fileEntries >= packCap || sinceSave >= kSerializeMaxAge);
}

}  // namespace vpgpu
