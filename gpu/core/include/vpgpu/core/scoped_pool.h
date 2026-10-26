// RAII Objective-C autorelease pool for threads that have none. Guest threads are long-lived pthreads: whatever
// ARC autoreleases on them (getter returns, temporary descriptors) is never freed, so a Metal object handed out
// that way outlives its destruction. Open one at the top of any entry point such a thread calls into the backend.
// A no-op everywhere but Apple platforms (Linux tests, the null platform).
#pragma once

#if defined(__APPLE__)
extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void*);
#endif

namespace vpgpu {

class ScopedPool {
 public:
#if defined(__APPLE__)
  ScopedPool() : pool_(objc_autoreleasePoolPush()) {}
  ~ScopedPool() { objc_autoreleasePoolPop(pool_); }
#else
  ScopedPool() {}
  ~ScopedPool() {}  // user-provided so a named instance is not an unused variable
#endif
  ScopedPool(const ScopedPool&) = delete;
  ScopedPool& operator=(const ScopedPool&) = delete;

 private:
#if defined(__APPLE__)
  void* pool_;
#endif
};

}  // namespace vpgpu
