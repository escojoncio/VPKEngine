// Background GPU pause. visionOS fails every MTLCommandBuffer committed while the app is in the background
// (kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted) and writes GPU hang dumps. The host sets the
// flag from its scene-phase callback (main thread). A backend calls waitForeground(owner) before it takes the lock that
// fixes the order of its commits, so a submit / interop thread parks (holding no lock and no frame slot) until the app is
// foreground again; a thread that would park under that lock hands its command buffer over instead (CommitQueue /
// DeferredCommits in submit_order.h, wouldPark() decides) and the resume hook sends the batch. The older metal2 backend
// still parks in waitForeground(owner) right before its commits. `owner` is the same pointer the backend later passes to
// release(owner) (its Impl*); a caller without an owner (nullptr) is woken only by release() / shutdown().
//
// The main thread never parks: the thread that last called set(true) is "main", and a commit it makes while in the
// background goes through and is counted (mainThreadPassed()). Skipping the commit instead would leave the command
// buffer's completion handler (frame slot, fences) pending for ever. The gate is process wide, shared by every backend:
// release(owner) wakes only the threads that parked for that owner (a backend being destroyed must not make another live
// backend's threads commit in the background). Platform neutral (no Metal): the null platform tests it.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>

#ifdef __cplusplus
extern "C" {
#endif
// Host API (call from the main thread). `on` != 0: the app went to the background, commits park from now on;
// 0: it is foreground again, parked threads resume. Logs "[vpgpu] bg paused" / "[vpgpu] bg resumed blocked=N" once
// per transition (N = commits that parked during the pause).
__attribute__((visibility("default"))) void vpgpu_set_background(int on);
// 1 while commits park (background and not shut down), else 0.
__attribute__((visibility("default"))) int vpgpu_is_background(void);
// Releases every parked thread and turns the gate off for good (process teardown).
__attribute__((visibility("default"))) void vpgpu_shutdown_background(void);
#ifdef __cplusplus
}

namespace vpgpu {

class BackgroundGate {
 public:
  using LogFn = std::function<void(const std::string&)>;  // default: one line to stderr

  void set(bool background);
  // True while commits park: background and not shut down. Lock free (stall detectors poll it).
  bool active() const { return active_.load(std::memory_order_acquire); }
  // Called before committing a command buffer: returns at once in the foreground and on the main thread, else parks
  // until foreground, release() (all owners), release(owner) for the same `owner`, or shutdown(). `owner`: whoever will call
  // release(owner) for it (a backend passes itself); nullptr = only the global release() and shutdown() wake it.
  void waitForeground(const void* owner = nullptr);
  // True when waitForeground() on this thread would park right now (it does not park).
  bool wouldPark() const;
  // Wakes the threads parked now (a backend being destroyed waits for their command buffers); a later pause parks again.
  // release(owner), owner != nullptr: only the threads that parked with that owner; release() / nullptr: every parked thread.
  void release(const void* owner = nullptr);
  // release() and never park again.
  void shutdown();
  void setLog(LogFn);
  // `fn` runs on the thread that calls set(true), at each transition to the background (after the state changed, outside
  // the gate's own lock). It must not block nor call add/removePauseHook. Returns an id for removePauseHook, which waits for
  // a running call of that hook to end (the owner can then be destroyed).
  using PauseHook = std::function<void()>;
  uint64_t addPauseHook(PauseHook fn);
  void removePauseHook(uint64_t id);
  // Same, for each transition back to the foreground (set(false); not release() nor shutdown()): `fn` runs on the thread that
  // calls set(false), after parked threads were woken. A backend uses it to send command buffers it had to hold back.
  // removeResumeHook waits for a running call of that hook to end.
  uint64_t addResumeHook(PauseHook fn);
  void removeResumeHook(uint64_t id);

  uint64_t blockedTotal() const;      // commits that ever parked
  uint64_t mainThreadPassed() const;  // commits the main thread made in the background (not parked)

 private:
  void log(const std::string&) const;

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<bool> active_{false};
  bool background_ = false, shutdown_ = false;
  uint64_t epoch_ = 0;  // bumped by release() (all owners)
  struct OwnerState { uint64_t epoch = 0; uint32_t waiters = 0; };
  std::map<const void*, OwnerState> owners_;  // owners with threads parked now (the entry goes with the last one); release(owner) bumps epoch
  uint64_t episodeBlocked_ = 0, blockedTotal_ = 0, mainPassed_ = 0;
  std::thread::id main_;
  LogFn log_;
  std::mutex hookMutex_;  // guards hooks_ and is held while they run
  std::map<uint64_t, PauseHook> hooks_, resumeHooks_;
  uint64_t nextHook_ = 1;
};

// The process-wide gate the C API drives and the Metal backends consult.
BackgroundGate& backgroundGate();

}  // namespace vpgpu
#endif
