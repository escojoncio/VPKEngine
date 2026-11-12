// A map whose values are built on demand by whichever thread asks first, and never by two threads at once for the same key.
// The Metal backend builds its internal pipelines (clear, blit) this way: a background job warms them at start, and the submit
// thread asks for the same ones synchronously if it gets there first (it then waits for the build in progress, or builds
// another key at the same time). Platform neutral (no Metal): the null platform tests it.
#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace vpgpu {

template <class K, class V, class Hash = std::hash<K>>
class OnceMap {
 public:
  struct Built {
    V value;
    bool keep = true;  // false: this caller gets `value`, nothing is stored and a later get() builds again (a transient failure)
  };

  // The stored value of `key`; else `build()` (returning Built) runs on this thread, unless another thread is building the same
  // key: then this one waits and takes what that one stored (or builds it itself if that one stored nothing).
  // `build` runs outside the map's lock and must not call get() for the same key.
  template <class F>
  V get(const K& key, F&& build) {
    std::unique_lock lock(mutex_);
    for (;;) {
      if (auto it = done_.find(key); it != done_.end()) return it->second;
      if (building_.insert(key).second) break;
      cv_.wait(lock);
    }
    lock.unlock();
    struct Release {  // also when build() throws: the key must not stay "building" for ever
      OnceMap& m;
      const K& k;
      bool armed = true;
      ~Release() {
        if (!armed) return;
        {
          std::lock_guard l(m.mutex_);
          m.building_.erase(k);
        }
        m.cv_.notify_all();
      }
    } release{*this, key};
    Built b = build();
    {
      std::lock_guard l(mutex_);
      if (b.keep) done_.emplace(key, b.value);
      building_.erase(key);
      release.armed = false;
    }
    cv_.notify_all();
    return std::move(b.value);
  }

  // True and `out` set when `key` is stored. Never builds, never waits.
  bool peek(const K& key, V& out) const {
    std::lock_guard l(mutex_);
    const auto it = done_.find(key);
    if (it == done_.end()) return false;
    out = it->second;
    return true;
  }
  size_t size() const {
    std::lock_guard l(mutex_);
    return done_.size();
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::unordered_map<K, V, Hash> done_;
  std::unordered_set<K, Hash> building_;
};

}  // namespace vpgpu
