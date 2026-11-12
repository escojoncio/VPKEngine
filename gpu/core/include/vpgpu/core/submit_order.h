// Bookkeeping the Metal backends share for ordering and completion of command buffers. Platform neutral (no Metal):
// the null platform tests it.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <utility>
#include <vector>

#include "vpgpu/core/background.h"

namespace vpgpu {

// Command buffers that have taken their place in the queue's order (their event values are fixed) but are not committed yet
// because the app went to the background while they were being encoded. A thread must never park in the background gate
// while it holds the lock that fixes the order: the thread that ends the pause may need that lock (an interop commit made by
// the main thread, which the gate lets through), and nothing would wake the parked one. So a thread that would park hands its
// command buffer over instead, and the next thread that may commit sends the whole batch, oldest first (the event values
// require commit order = order of the values). Not thread safe: the caller's order lock guards it.
template <class T>
class DeferredCommits {
 public:
  void push(T item) { items_.push_back(std::move(item)); }
  // Everything pushed so far, oldest first, if this thread may commit now (it would not park in `gate`); else nothing.
  std::vector<T> take(const BackgroundGate& gate) {
    std::vector<T> out;
    if (!gate.wouldPark()) out.swap(items_);
    return out;
  }
  // Everything, whatever the gate says (teardown: the gate is released and the commits must happen to free their slots).
  std::vector<T> takeAll() { return std::exchange(items_, {}); }
  size_t size() const { return items_.size(); }

 private:
  std::vector<T> items_;
};

// "Frames below completed() have finished on the GPU" for a backend whose frames are command buffers on one in-order queue,
// where a frame without encoders has no command buffer. Such an empty frame is complete once every earlier frame is, so it
// advances the frontier only when none of the frames before it is still in flight (never past an earlier frame's work).
class CompletionFrontier {
 public:
  // Frame `index` was handed to the backend; `work`: it has a command buffer, and finished(index) will follow.
  // Call before that command buffer is committed.
  void submitted(uint64_t index, bool work) {
    std::lock_guard lock(mutex_);
    end_ = std::max(end_, index + 1);
    if (work) inFlight_.push_back(index);
    publish();
  }
  // The command buffer of frame `index` finished; one in-order queue, so every earlier one did too.
  void finished(uint64_t index) {
    std::lock_guard lock(mutex_);
    while (!inFlight_.empty() && inFlight_.front() <= index) inFlight_.pop_front();
    publish();
  }
  uint64_t completed() const { return completed_.load(std::memory_order_acquire); }

 private:
  // Below the oldest frame in flight everything is done; with none in flight, everything submitted is.
  void publish() {
    const uint64_t now = inFlight_.empty() ? end_ : inFlight_.front();
    if (now > completed_.load(std::memory_order_relaxed)) completed_.store(now, std::memory_order_release);
  }

  std::mutex mutex_;
  std::deque<uint64_t> inFlight_;  // frames with a command buffer not yet finished, oldest first
  uint64_t end_ = 0;               // newest frame handed over + 1
  std::atomic<uint64_t> completed_{0};
};

}  // namespace vpgpu
