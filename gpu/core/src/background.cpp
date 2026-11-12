#include "vpgpu/core/background.h"

#include <cstdio>

namespace vpgpu {

void BackgroundGate::log(const std::string& line) const {
  LogFn fn;
  {
    std::lock_guard lock(mutex_);
    fn = log_;
  }
  if (fn) fn(line);
  else std::fprintf(stderr, "%s\n", line.c_str());
}

void BackgroundGate::setLog(LogFn fn) {
  std::lock_guard lock(mutex_);
  log_ = std::move(fn);
}

void BackgroundGate::set(bool background) {
  std::string line;
  {
    std::lock_guard lock(mutex_);
    if (background == background_) return;
    background_ = background;
    if (background) {
      main_ = std::this_thread::get_id();
      episodeBlocked_ = 0;
      line = "[vpgpu] bg paused";
    } else {
      line = "[vpgpu] bg resumed blocked=" + std::to_string(episodeBlocked_);
    }
    active_.store(background_ && !shutdown_, std::memory_order_release);
  }
  if (!background) cv_.notify_all();
  log(line);
  if (background) {
    std::lock_guard hooks(hookMutex_);
    for (auto& [id, fn] : hooks_) fn();
  }
}

uint64_t BackgroundGate::addPauseHook(PauseHook fn) {
  std::lock_guard lock(hookMutex_);
  const uint64_t id = nextHook_++;
  hooks_.emplace(id, std::move(fn));
  return id;
}

void BackgroundGate::removePauseHook(uint64_t id) {
  std::lock_guard lock(hookMutex_);
  hooks_.erase(id);
}

void BackgroundGate::waitForeground() {
  if (!active()) return;
  std::unique_lock lock(mutex_);
  if (!background_ || shutdown_) return;
  if (std::this_thread::get_id() == main_) {
    ++mainPassed_;
    return;
  }
  ++episodeBlocked_, ++blockedTotal_;
  const uint64_t epoch = epoch_;
  cv_.wait(lock, [&] { return !background_ || shutdown_ || epoch_ != epoch; });
}

bool BackgroundGate::wouldPark() const {
  if (!active()) return false;
  std::lock_guard lock(mutex_);
  return background_ && !shutdown_ && std::this_thread::get_id() != main_;
}

void BackgroundGate::release() {
  {
    std::lock_guard lock(mutex_);
    ++epoch_;
  }
  cv_.notify_all();
}

void BackgroundGate::shutdown() {
  bool wasActive;
  uint64_t blocked;
  {
    std::lock_guard lock(mutex_);
    wasActive = background_ && !shutdown_;
    blocked = episodeBlocked_;
    shutdown_ = true;
    active_.store(false, std::memory_order_release);
  }
  cv_.notify_all();
  if (wasActive) log("[vpgpu] bg shutdown blocked=" + std::to_string(blocked));
}

uint64_t BackgroundGate::blockedTotal() const {
  std::lock_guard lock(mutex_);
  return blockedTotal_;
}

uint64_t BackgroundGate::mainThreadPassed() const {
  std::lock_guard lock(mutex_);
  return mainPassed_;
}

BackgroundGate& backgroundGate() {
  static BackgroundGate* gate = new BackgroundGate;  // never destroyed: threads may still be parked at exit
  return *gate;
}

}  // namespace vpgpu

extern "C" {
void vpgpu_set_background(int on) { vpgpu::backgroundGate().set(on != 0); }
int vpgpu_is_background(void) { return vpgpu::backgroundGate().active() ? 1 : 0; }
void vpgpu_shutdown_background(void) { vpgpu::backgroundGate().shutdown(); }
}
