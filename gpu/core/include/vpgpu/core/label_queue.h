// GPU-ordered side effects (PM4 labels: EVENT_WRITE_EOP / EOS / RELEASE_MEM data + interrupt).
//
// The command processor parses a label long before the GPU runs the draws recorded ahead of it.
// Writing the label at parse time tells the title "that work is done" too early. Each effect is
// instead queued against the frame that holds the work before it and runs, in queue order, once that
// frame (and every one before it) has finished on the GPU.
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>

namespace vpgpu {

class LabelQueue {
 public:
  // `action` runs once every frame with an index <= `frame` has finished. [address, +size) is the
  // memory it writes (0/0 for an interrupt only): pending() and waitTarget() look at it.
  // `pureWrite`: the action only writes [address, +size) (no interrupt, nothing else): supersede() may drop it.
  void add(uint64_t frame, uint64_t address, uint32_t size, std::function<void()> action, bool pureWrite = false);
  // A write the caller makes later in GPU order covers [address, +size) (a DMA fill / copy of the command processor, after
  // the label in the command stream): the pure writes queued entirely inside it, not yet running, would only be overwritten
  // by it, so they are dropped (their entries stay, empty, in order). Returns how many were dropped.
  size_t supersede(uint64_t address, uint64_t size);
  // Runs, on the calling thread and in the order they were added, the entries whose frame is below
  // `completedBelow`; stops at the first that is not ready (labels never overtake each other).
  // Actions must not call release(). Returns how many ran. Any thread.
  size_t release(uint64_t completedBelow);
  // The same, reading the number again at each entry (frames finish while it runs).
  size_t release(const std::function<uint64_t()>& completed);
  // The same for a caller that must never wait for another thread's release (the thread that holds
  // the engine lock): when a release is running on another thread, that thread is asked to look
  // again once it is done (`completed` is read again at each entry) and 0 is returned at once.
  // `completed` returns the same number release() takes, now.
  size_t tryRelease(const std::function<uint64_t()>& completed);
  // Entries of frames below `frames` that have not finished running (an action in progress counts).
  size_t pendingBelow(uint64_t frames) const;
  // Waits, without ever blocking on runMutex_ (the caller holds the engine lock: see below), until no entry
  // of a frame below `frames` is left: it releases what is ready with tryRelease and polls while another
  // thread runs the rest. False when `timeoutMs` passed first.
  bool waitBelow(uint64_t frames, const std::function<uint64_t()>& completed, unsigned timeoutMs);
  // An entry that writes memory overlapping [address, +size) has not run yet.
  bool pending(uint64_t address, uint32_t size) const;
  // The number of frames that must have finished before those entries have run (frame + 1 of the
  // newest overlapping entry), 0 if none overlaps.
  uint64_t waitTarget(uint64_t address, uint32_t size) const;
  size_t size() const;
  // No entry is queued. Lock-free (an atomic mirror of the queue length): the hot paths of a host that queues rarely
  // ask it on every call without touching mutex_.
  bool empty() const { return count_.load(std::memory_order_acquire) == 0; }

 private:
  struct Entry {
    uint64_t frame, address;
    uint32_t size;
    std::function<void()> action;
    bool pureWrite = false, running = false;
  };
  size_t runOnce(const std::function<uint64_t()>& completed);    // one pass, runMutex_ held
  size_t runLocked(const std::function<uint64_t()>& completed);  // runMutex_ held
  mutable std::mutex mutex_;
  // One release at a time: entries must run in order. Lock order: engine lock -> (never blocks on)
  // runMutex_; a thread that holds the engine lock only try-locks it (tryRelease), because the
  // action of an entry takes the memory manager's lock, which a thread waiting for the engine lock
  // (a fault on a page the renderer owns) may hold.
  std::mutex runMutex_;
  std::atomic<bool> again_{false};  // a tryRelease found runMutex_ taken: the holder looks again
  std::deque<Entry> queue_;
  std::atomic<size_t> count_{0};  // queue_.size(), written under mutex_
};

}  // namespace vpgpu
