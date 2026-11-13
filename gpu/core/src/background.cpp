#include "vpgpu/core/background.h"

#include <cstdio>

#ifdef VPGPU_BACKGROUND_TESTING  // (defined only by the vpgpu_core_bgtest object library the test executables link: the shipped library has no hook)
namespace {
std::atomic<void (*)(const void*)> gParkHook{nullptr};
}
extern "C" void vpgpu_test_park_hook(void (*f)(const void*)) { gParkHook = f; }
#endif

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
  std::lock_guard hooks(hookMutex_);
  for (auto& [id, fn] : background ? hooks_ : resumeHooks_) fn();
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

uint64_t BackgroundGate::addResumeHook(PauseHook fn) {
  std::lock_guard lock(hookMutex_);
  const uint64_t id = nextHook_++;
  resumeHooks_.emplace(id, std::move(fn));
  return id;
}

void BackgroundGate::removeResumeHook(uint64_t id) {
  std::lock_guard lock(hookMutex_);
  resumeHooks_.erase(id);
}

void BackgroundGate::waitForeground(const void* owner) { park(owner, nullptr); }

void BackgroundGate::waitForeground(const void* owner, std::unique_lock<std::mutex>& registry) { park(owner, &registry); }

void BackgroundGate::park(const void* owner, std::unique_lock<std::mutex>* registry) {
  // Lock order: the caller's registry, then the gate's lock (nothing takes them the other way round). The registry goes before the
  // wait starts, but only once the wait is registered under the gate's lock: a release(owner) issued after the registry was
  // dropped (a device being destroyed) then finds this thread among the waiters instead of passing before it.
  auto dropRegistry = [&] { if (registry && registry->owns_lock()) registry->unlock(); };
  if (!active()) return dropRegistry();
#ifdef VPGPU_BACKGROUND_TESTING
  // Test hook: called with the caller's registry still held, as part of taking the gate's lock (the window the registry closes), so a test
  // can run the worst interleaving (another thread releasing `owner` the moment the registry is free) there on purpose, not by timing.
  // It is in the same statement as the lock, not before it: a registry dropped by any statement before the lock is then dropped before
  // the hook too, there is no gap between the hook and the lock where a drop would go unseen.
  std::unique_lock lock = ([&] { if (auto* h = gParkHook.load()) h(owner); }(), std::unique_lock(mutex_));
#else
  std::unique_lock lock(mutex_);
#endif
  if (!background_ || shutdown_) return dropRegistry();
  if (std::this_thread::get_id() == main_) {
    ++mainPassed_;
    return dropRegistry();
  }
  ++episodeBlocked_, ++blockedTotal_;
  const uint64_t epoch = epoch_;
  OwnerState* st = owner ? &owners_[owner] : nullptr;  // (std::map nodes are stable while waiters > 0 keeps the entry)
  const uint64_t ownerEpoch = st ? st->epoch : 0;
  if (st) ++st->waiters;
  dropRegistry();
  cv_.wait(lock, [&] { return !background_ || shutdown_ || epoch_ != epoch || (st && st->epoch != ownerEpoch); });
  if (st && --st->waiters == 0) owners_.erase(owner);
}

bool BackgroundGate::wouldPark() const {
  if (!active()) return false;
  std::lock_guard lock(mutex_);
  return background_ && !shutdown_ && std::this_thread::get_id() != main_;
}

void BackgroundGate::release(const void* owner) {
  {
    std::lock_guard lock(mutex_);
    if (owner) {
      const auto it = owners_.find(owner);  // nobody parked for it: nothing to wake, nothing to remember
      if (it != owners_.end()) ++it->second.epoch;
    } else ++epoch_;
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
