// Work-in-progress gauge: how many pieces of work are running and when the last one ended. Background work that should
// not compete with the foreground (speculative library compiles, binary-archive harvesting) asks it whether the
// foreground is quiet. Lock free for the work that counts (enter / leave take a lock only to wake a waiter); platform
// neutral (the null platform tests it).
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace vpgpu {

class ActivityGauge {
 public:
  using Clock = std::chrono::steady_clock;

  void enter() { active_.fetch_add(1, std::memory_order_acq_rel); }
  // The end time goes first: a reader that sees the count drop also sees it.
  void leave(Clock::time_point at = Clock::now()) {
    endNs_.store(ns(at), std::memory_order_release);
    // seq_cst on this and on waitInactive's side: a leave that drops the count to 0 either sees the waiter (and wakes it) or the
    // waiter sees the 0.
    if (active_.fetch_sub(1, std::memory_order_seq_cst) == 1 && waiters_.load(std::memory_order_seq_cst)) {
      std::lock_guard lock(mutex_);
      cv_.notify_all();
    }
  }
  // Blocks until nothing is running or `timeout` passes (a leave() wakes it at once); true = nothing is running. A timeout is kept
  // within [0, a year]: now + the largest duration would overflow the clock and end the wait at once, a negative one is a poll.
  bool waitInactive(std::chrono::nanoseconds timeout) {
    timeout = std::clamp(timeout, std::chrono::nanoseconds::zero(), std::chrono::nanoseconds(std::chrono::hours(24 * 365)));
    std::unique_lock lock(mutex_);
    waiters_.fetch_add(1, std::memory_order_seq_cst);
    cv_.wait_for(lock, timeout, [this] { return active_.load(std::memory_order_seq_cst) == 0; });
    waiters_.fetch_sub(1, std::memory_order_seq_cst);
    return active_.load(std::memory_order_acquire) == 0;
  }
  uint32_t active() const { return active_.load(std::memory_order_acquire); }
  // Nothing is running and nothing ended within the last `quiet` (a gauge never used is quiet).
  bool idleFor(std::chrono::nanoseconds quiet, Clock::time_point now = Clock::now()) const {
    if (active()) return false;
    const int64_t end = endNs_.load(std::memory_order_acquire);
    return end == 0 || ns(now) - end >= quiet.count();
  }

  // enter() now, leave() at the end of the scope; `on` false does nothing (one code path for counted and uncounted callers).
  class Scope {
   public:
    explicit Scope(ActivityGauge& g, bool on = true) : g_(on ? &g : nullptr) {
      if (g_) g_->enter();
    }
    ~Scope() {
      if (g_) g_->leave();
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

   private:
    ActivityGauge* g_;
  };

 private:
  static int64_t ns(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
  }

  std::atomic<uint32_t> active_{0};
  std::atomic<uint32_t> waiters_{0};  // threads inside waitInactive
  std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<int64_t> endNs_{0};  // steady-clock ns of the last leave(); 0 = none yet
};

}  // namespace vpgpu
