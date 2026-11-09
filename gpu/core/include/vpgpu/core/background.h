// Background GPU pause. visionOS fails every MTLCommandBuffer committed while the app is in the background
// (kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted) and writes GPU hang dumps. The host sets the
// flag from its scene-phase callback (main thread); each backend calls gateCommit() right before it commits a
// command buffer, so the committing thread (renderer / submit threads) parks until the app is foreground again.
//
// The main thread never parks: the thread that last called set(true) is "main", and a commit it makes while in the
// background goes through and is counted (mainThreadPassed()). Skipping the commit instead would leave the command
// buffer's completion handler (frame slot, fences) pending for ever. The engine itself commits only from its submit
// threads. Platform neutral (no Metal): the null platform tests it.
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
  // until foreground, release() or shutdown().
  void waitForeground();
  // Wakes the threads parked now (a backend being destroyed waits for their command buffers); a later pause parks again.
  void release();
  // release() and never park again.
  void shutdown();
  void setLog(LogFn);
  // `fn` runs on the thread that calls set(true), at each transition to the background (after the state changed, outside
  // the gate's own lock). It must not block nor call add/removePauseHook. Returns an id for removePauseHook, which waits for
  // a running call of that hook to end (the owner can then be destroyed).
  using PauseHook = std::function<void()>;
  uint64_t addPauseHook(PauseHook fn);
  void removePauseHook(uint64_t id);

  uint64_t blockedTotal() const;      // commits that ever parked
  uint64_t mainThreadPassed() const;  // commits the main thread made in the background (not parked)

 private:
  void log(const std::string&) const;

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<bool> active_{false};
  bool background_ = false, shutdown_ = false;
  uint64_t epoch_ = 0;  // bumped by release()
  uint64_t episodeBlocked_ = 0, blockedTotal_ = 0, mainPassed_ = 0;
  std::thread::id main_;
  LogFn log_;
  std::mutex hookMutex_;  // guards hooks_ and is held while they run
  std::map<uint64_t, PauseHook> hooks_;
  uint64_t nextHook_ = 1;
};

// The process-wide gate the C API drives and the Metal backends consult.
BackgroundGate& backgroundGate();

}  // namespace vpgpu
#endif
