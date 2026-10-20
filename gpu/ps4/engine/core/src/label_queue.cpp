#include "vp4gpu/core/label_queue.h"

#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>
#include <utility>

namespace vp4gpu {

namespace {
bool overlap(uint64_t a, uint64_t n, uint64_t b, uint64_t m) {
  if (!n || !m) return false;
  return a < b + m && b < a + n;
}
}  // namespace

void LabelQueue::add(uint64_t frame, uint64_t address, uint32_t size, std::function<void()> action) {
  std::lock_guard lock(mutex_);
  // Frames only grow: an entry never needs to run before one added earlier.
  if (!queue_.empty()) frame = std::max(frame, queue_.back().frame);
  queue_.push_back({frame, address, size, std::move(action)});
}

size_t LabelQueue::release(uint64_t completedBelow) {
  return release([completedBelow] { return completedBelow; });
}

size_t LabelQueue::release(const std::function<uint64_t()>& completed) {
  size_t ran = 0;
  do {
    {
      std::lock_guard run(runMutex_);
      ran += runLocked(completed);
    }
    // A tryRelease that found runMutex_ taken after runLocked's last look left a request: serve it
    // (otherwise its entries would wait for the next unrelated release).
  } while (again_.load());
  return ran;
}

size_t LabelQueue::pendingBelow(uint64_t frames) const {
  std::lock_guard lock(mutex_);
  size_t n = 0;
  for (const Entry& e : queue_)
    if (e.frame < frames) ++n;
  return n;
}

bool LabelQueue::waitBelow(uint64_t frames, const std::function<uint64_t()>& completed, unsigned timeoutMs) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  for (;;) {
    tryRelease(completed);
    if (pendingBelow(frames) == 0) return true;
    if (std::chrono::steady_clock::now() >= end) return false;
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
}

size_t LabelQueue::tryRelease(const std::function<uint64_t()>& completed) {
  size_t ran = 0;
  for (;;) {
    if (!runMutex_.try_lock()) {
      again_.store(true);
      // (the holder may have just passed its last look: if it let go, do the work here)
      if (!runMutex_.try_lock()) return ran;
    }
    ran += runLocked(completed);
    runMutex_.unlock();
    // A tryRelease that failed after the holder's last look asked for another round.
    if (!again_.load()) return ran;
  }
}

size_t LabelQueue::runLocked(const std::function<uint64_t()>& completed) {
  size_t ran = 0;
  do {
    again_.store(false);
    ran += runOnce(completed);
  } while (again_.load());
  return ran;
}

size_t LabelQueue::runOnce(const std::function<uint64_t()>& completed) {
  size_t ran = 0;
  for (;;) {
    std::function<void()> action;
    {
      std::lock_guard lock(mutex_);
      if (queue_.empty() || queue_.front().frame >= completed()) break;
      action = std::move(queue_.front().action);
      // (kept in the queue while it runs, so pending() still says "not written yet")
    }
    if (action) action();
    {
      std::lock_guard lock(mutex_);
      queue_.pop_front();
    }
    ++ran;
  }
  return ran;
}

bool LabelQueue::pending(uint64_t address, uint32_t size) const {
  std::lock_guard lock(mutex_);
  for (const Entry& e : queue_)
    if (overlap(address, size, e.address, e.size)) return true;
  return false;
}

uint64_t LabelQueue::waitTarget(uint64_t address, uint32_t size) const {
  std::lock_guard lock(mutex_);
  uint64_t target = 0;
  for (const Entry& e : queue_)
    if (overlap(address, size, e.address, e.size)) target = std::max(target, e.frame + 1);
  return target;
}

size_t LabelQueue::size() const {
  std::lock_guard lock(mutex_);
  return queue_.size();
}

}  // namespace vp4gpu
