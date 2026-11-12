// Bookkeeping the Metal backends share for ordering and completion of command buffers. Platform neutral (no Metal):
// the null platform tests it.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "vpgpu/core/background.h"

namespace vpgpu {

// Command buffers that have taken their place in the queue's order (their event values are fixed) but are not committed yet
// because the app went to the background while they were being encoded. A thread must never park in the background gate
// while it holds the lock that fixes the order: the thread that ends the pause may need that lock (an interop commit made by
// the main thread, which the gate lets through), and nothing would wake the parked one. So a thread that would park hands its
// command buffer over instead, and the next thread that may commit sends the whole batch, oldest first (the event values
// require commit order = order of the values). The caller's order lock guards push/take; pending() alone is lock free, so a
// thread without the lock can tell whether anyone still has to send something (a pause that ended with no later commit).
template <class T>
class DeferredCommits {
 public:
  void push(T item) {
    items_.push_back(std::move(item));
    pending_.store(items_.size(), std::memory_order_release);
  }
  // Everything pushed so far, oldest first, if this thread may commit now (it would not park in `gate`); else nothing.
  std::vector<T> take(const BackgroundGate& gate) {
    std::vector<T> out;
    if (!gate.wouldPark()) {
      out.swap(items_);
      pending_.store(0, std::memory_order_release);
    }
    return out;
  }
  // Everything, whatever the gate says (teardown: the gate is released and the commits must happen to free their slots).
  std::vector<T> takeAll() {
    pending_.store(0, std::memory_order_release);
    return std::exchange(items_, {});
  }
  // take(gate), or everything when the owner is `closing`: a thread parked in the gate that release() woke at teardown still
  // sees the gate in the background and would hand its command buffer over to a flush that never comes.
  std::vector<T> takeReady(const BackgroundGate& gate, bool closing) { return closing ? takeAll() : take(gate); }
  size_t size() const { return items_.size(); }
  // Lock free; how many are waiting to be committed (may be stale by the time the caller acts).
  size_t pending() const { return pending_.load(std::memory_order_acquire); }

 private:
  std::vector<T> items_;
  std::atomic<size_t> pending_{0};
};

// The lock that fixes the order of the commits, with its owner recorded: a callback that runs under it (the encode callback of
// an interop commit) may call something that wants to flush, and std::mutex::try_lock by the thread that owns the mutex is
// undefined. try_lock() here answers false for the owner.
class OrderLock {
 public:
  void lock() {
    m_.lock();
    owner_.store(std::this_thread::get_id(), std::memory_order_relaxed);
  }
  bool try_lock() {
    if (owner_.load(std::memory_order_relaxed) == std::this_thread::get_id()) return false;
    if (!m_.try_lock()) return false;
    owner_.store(std::this_thread::get_id(), std::memory_order_relaxed);
    return true;
  }
  void unlock() {
    owner_.store(std::thread::id{}, std::memory_order_relaxed);
    m_.unlock();
  }

 private:
  std::mutex m_;
  std::atomic<std::thread::id> owner_{};
};

// Everything a Metal backend needs to commit its command buffers in the order of their event values while the app may go to
// the background (see DeferredCommits): the order lock, the held-back batch, the teardown flag and a count of the threads that
// are inside a submit / interop call (so teardown can wait for the ones the gate releases). `commit` below is a callable that
// commits one T.
template <class T>
class CommitQueue {
 public:
  explicit CommitQueue(const BackgroundGate& gate) : gate_(gate) {}
  CommitQueue(const CommitQueue&) = delete;
  CommitQueue& operator=(const CommitQueue&) = delete;

  OrderLock lock;  // fixes the order; guards `deferred`
  DeferredCommits<T> deferred;
  std::atomic<bool> closing{false};  // teardown started: nothing is held back for the background any more

  // `lock` held: commits what may be committed now (everything when closing: a thread released from the gate at teardown still
  // sees the background, and no flush would come for what it handed over).
  template <class Commit>
  void commitReady(Commit&& commit) {
    for (T p : deferred.takeReady(gate_, closing.load(std::memory_order_acquire))) commit(p);
  }
  // `lock` held: `item` takes its place in the order, then whatever may be committed is.
  template <class Commit>
  void push(T item, Commit&& commit) {
    deferred.push(std::move(item));
    commitReady(commit);
  }
  // Without `lock`: sends the held-back batch if this thread may commit now. Never waits for the lock (its holder can be
  // encoding for a long time, or be the caller of this very call): false = busy, try again; the holder ends with push(), which
  // sends the batch too. True = nothing left for this thread to do. A thread that would park skips even the lock.
  template <class Commit>
  bool flush(Commit&& commit) {
    if (!deferred.pending()) return true;
    if (!closing.load(std::memory_order_acquire) && gate_.wouldPark()) return true;
    std::unique_lock l(lock, std::try_to_lock);
    if (!l.owns_lock()) return false;
    commitReady(commit);
    return true;
  }

  // A thread inside a submit / interop call: make one BEFORE the call can park in the gate and keep it until the call is over
  // (the destructor of the object ends the use of the queue's owner).
  class Inside {
   public:
    explicit Inside(CommitQueue& q) : q_(q.inside_) { q_.fetch_add(1, std::memory_order_acq_rel); }
    ~Inside() { q_.fetch_sub(1, std::memory_order_acq_rel); }  // the last use of the owner
    Inside(const Inside&) = delete;
    Inside& operator=(const Inside&) = delete;

   private:
    std::atomic<uint32_t>& q_;
  };
  uint32_t insideCount() const { return inside_.load(std::memory_order_acquire); }
  // Teardown: waits until no thread is inside any more, calling `poke` between polls (it must release the gate and flush: a
  // thread that reached the gate after the first release has to be woken, and one waiting for a slot needs the batch sent).
  template <class Poke>
  void waitNoneInside(Poke&& poke) {
    while (inside_.load(std::memory_order_acquire)) {
      poke();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

 private:
  const BackgroundGate& gate_;
  std::atomic<uint32_t> inside_{0};
};

// "Frames below completed() have finished on the GPU" for a backend whose frames are command buffers on one in-order queue,
// where a frame without encoders has no command buffer. Such an empty frame is complete once every earlier frame is, so it
// advances the frontier only when none of the frames before it is still in flight (never past an earlier frame's work).
// The calls may come out of index order (the indices are handed out by a counter, the frames are encoded on several threads):
// a frame that has not been handed over yet is not finished, whatever arrived after it, and finished() follows the order
// in which the command buffers were handed over (the queue's), not the indices. Every index below the highest one must
// reach submitted() eventually; one that never does holds the frontier back.
class CompletionFrontier {
 public:
  // Frame `index` was handed to the backend; `work`: it has a command buffer, and finished(index) will follow.
  // Call before that command buffer is committed. Contract: for a frame with work, this call and the commit (or the hand-over
  // to DeferredCommits) happen under the same lock (the backend's commits.lock), so that the order of the calls is the
  // order of the queue that finished() relies on. A frame whose index never reaches this call freezes completed() below it
  // (nothing can tell it was skipped) until kMaxGaps later ranges are pending: every index handed out must be submitted, even if its
  // frame is dropped.
  // An index that was already submitted (repeated call) is ignored, work or not: a second entry in the queue could never finish
  // and would hold the frontier for good.
  void submitted(uint64_t index, bool work) {
    std::lock_guard lock(mutex_);
    if (index >= end_) {
      if (index > end_) {
        gaps_[end_] = index;  // [end_, index) is still to come
        // Indices that never come would make this grow without bound: past kMaxGaps pending ranges the oldest is given up (its
        // indices count as handed over, so the frontier moves on instead of freezing for good). A frame of a given-up range that
        // does arrive later is taken for a repeat and ignored, including its work: the bound is for a caller that breaks the
        // "every index is submitted" contract, which is not supposed to happen.
        while (gaps_.size() > kMaxGaps) gaps_.erase(gaps_.begin());
      }
      end_ = index + 1;
    } else if (!closeGap(index)) {
      return;  // a repeated index (or one of a given-up range): nothing to record
    }
    if (work) inFlight_.push_back(index);
    publish();
  }
  // The command buffer of frame `index` finished; one in-order queue, so every one handed over before it did too.
  void finished(uint64_t index) {
    std::lock_guard lock(mutex_);
    const auto it = std::find(inFlight_.begin(), inFlight_.end(), index);
    if (it != inFlight_.end()) inFlight_.erase(inFlight_.begin(), it + 1);
    publish();
  }
  uint64_t completed() const { return completed_.load(std::memory_order_acquire); }

 private:
  // Below the lowest frame not handed over yet or still in flight everything is done; with none, everything submitted is.
  void publish() {
    uint64_t now = gaps_.empty() ? end_ : gaps_.begin()->first;
    for (const uint64_t i : inFlight_) now = std::min(now, i);
    if (now > completed_.load(std::memory_order_relaxed)) completed_.store(now, std::memory_order_release);
  }
  // True if `index` was in a gap (a late frame), which is now closed at it.
  bool closeGap(uint64_t index) {
    auto it = gaps_.upper_bound(index);
    if (it == gaps_.begin()) return false;
    --it;
    if (index >= it->second) return false;
    const uint64_t from = it->first, to = it->second;
    gaps_.erase(it);
    if (from < index) gaps_[from] = index;
    if (index + 1 < to) gaps_[index + 1] = to;
    return true;
  }

  static constexpr size_t kMaxGaps = 1024;  // ranges of indices not handed over yet that are tracked at once

  std::mutex mutex_;
  std::deque<uint64_t> inFlight_;           // frames with a command buffer not yet finished, in queue order
  std::map<uint64_t, uint64_t> gaps_;       // [from, to): indices below end_ not handed over yet
  uint64_t end_ = 0;                        // highest frame handed over + 1
  std::atomic<uint64_t> completed_{0};
};

}  // namespace vpgpu
