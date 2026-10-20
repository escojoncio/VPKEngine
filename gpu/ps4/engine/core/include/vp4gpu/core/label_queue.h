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

namespace vp4gpu {

class LabelQueue {
 public:
  // `action` runs once every frame with an index <= `frame` has finished. [address, +size) is the
  // memory it writes (0/0 for an interrupt only): pending() and waitTarget() look at it.
  void add(uint64_t frame, uint64_t address, uint32_t size, std::function<void()> action);
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
  // An entry that writes memory overlapping [address, +size) has not run yet.
  bool pending(uint64_t address, uint32_t size) const;
  // The number of frames that must have finished before those entries have run (frame + 1 of the
  // newest overlapping entry), 0 if none overlaps.
  uint64_t waitTarget(uint64_t address, uint32_t size) const;
  size_t size() const;
  bool empty() const { return size() == 0; }

 private:
  struct Entry {
    uint64_t frame, address;
    uint32_t size;
    std::function<void()> action;
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
};

}  // namespace vp4gpu
