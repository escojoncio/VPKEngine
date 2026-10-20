#include "vp4gpu/core/label_queue.h"

#include <algorithm>
#include <vector>

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
  std::lock_guard run(runMutex_);
  size_t ran = 0;
  for (;;) {
    std::function<void()> action;
    {
      std::lock_guard lock(mutex_);
      if (queue_.empty() || queue_.front().frame >= completedBelow) break;
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
