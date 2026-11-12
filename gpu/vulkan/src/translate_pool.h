// Background threads for the work that must not run on the game's threads: SPIR-V -> MSL translation of the
// stages of a pipeline (vkCreate*Pipelines returns before it; see pipelines.cpp "Asynchronous creation").
//
// Jobs have a priority: 0 = a draw is waiting for it (boost), 1 = created by the game, 2 = background. A job is run
// at most once: boosting pushes a second queue entry for the same job and whichever entry is popped first claims it.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

namespace vpvk {

class TranslatePool {
 public:
  struct Job {
    std::function<void()> fn;
    std::atomic<int> state{0};  // 0 queued, 1 running, 2 done, 3 dropped (the pool went away before it started; fn never ran)
    std::mutex m;
    std::condition_variable cv;
    bool finished() const { return state.load(std::memory_order_acquire) >= 2; }  // done or dropped: nothing will change
    bool done() const { return state.load(std::memory_order_acquire) == 2; }
  };
  using JobPtr = std::shared_ptr<Job>;

  explicit TranslatePool(unsigned threads) {
    for (unsigned i = 0; i < std::max(1u, threads); ++i) workers_.emplace_back([this] { loop(); });
  }
  // Jobs that never started are dropped (their owners are gone with the device).
  ~TranslatePool() {
    {
      std::lock_guard<std::mutex> l(m_);
      stop_ = true;
    }
    wake_.notify_all();
    for (std::thread& t : workers_) t.join();
    // A thread blocked in wait(job, UINT32_MAX) on a job that never started would wait for ever: release them (state "dropped",
    // wait() returns false; the function never ran and is dropped with what it captured).
    while (!queue_.empty()) {
      const JobPtr job = queue_.top().job;
      queue_.pop();
      int expect = 0;
      if (!job->state.compare_exchange_strong(expect, 1)) continue;  // claimed through its other entry
      job->fn = nullptr;
      {
        std::lock_guard<std::mutex> jl(job->m);
        job->state.store(3, std::memory_order_release);
      }
      job->cv.notify_all();
    }
  }
  TranslatePool(const TranslatePool&) = delete;
  TranslatePool& operator=(const TranslatePool&) = delete;

  JobPtr submit(std::function<void()> fn, int priority = 1) {
    auto job = std::make_shared<Job>();
    job->fn = std::move(fn);
    push(job, priority);
    return job;
  }
  // A draw needs this job now. No-op once it started.
  void boost(const JobPtr& job) {
    if (job && job->state.load(std::memory_order_acquire) == 0) push(job, 0);
  }
  // True when the job is done (it ran); false on timeout or when it was dropped at the pool's destruction. `ms` UINT32_MAX = no limit.
  bool wait(const JobPtr& job, uint32_t ms) {
    if (!job) return true;
    if (job->finished()) return job->done();
    std::unique_lock<std::mutex> l(job->m);
    if (ms == UINT32_MAX)
      job->cv.wait(l, [&] { return job->finished(); });
    else if (!job->cv.wait_for(l, std::chrono::milliseconds(ms), [&] { return job->finished(); }))
      return false;
    return job->done();
  }
  // Jobs queued or running (diagnostics).
  size_t pending() const {
    std::lock_guard<std::mutex> l(m_);
    return queue_.size() + running_;
  }
  unsigned threads() const { return unsigned(workers_.size()); }

 private:
  struct Entry {
    int priority;
    uint64_t order;
    JobPtr job;
    bool operator<(const Entry& o) const {  // std::priority_queue pops the largest
      return priority != o.priority ? priority > o.priority : order > o.order;
    }
  };

  void push(const JobPtr& job, int priority) {
    {
      std::lock_guard<std::mutex> l(m_);
      queue_.push({priority, next_++, job});
    }
    wake_.notify_one();
  }

  void loop() {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);  // never ahead of the game's render / submit threads
#endif
    std::unique_lock<std::mutex> l(m_);
    for (;;) {
      wake_.wait(l, [&] { return stop_ || !queue_.empty(); });
      if (stop_) return;
      Entry e = queue_.top();
      queue_.pop();
      int expect = 0;
      if (!e.job->state.compare_exchange_strong(expect, 1)) continue;  // claimed through its other entry
      ++running_;
      l.unlock();
      try {
        e.job->fn();
      } catch (...) {
      }
      e.job->fn = nullptr;  // drops what the job captured
      {
        std::lock_guard<std::mutex> jl(e.job->m);
        e.job->state.store(2, std::memory_order_release);
      }
      e.job->cv.notify_all();
      l.lock();
      --running_;
    }
  }

  mutable std::mutex m_;
  std::condition_variable wake_;
  std::priority_queue<Entry> queue_;
  uint64_t next_ = 0;
  size_t running_ = 0;
  bool stop_ = false;
  std::vector<std::thread> workers_;
};

}  // namespace vpvk
