// Tests for vpgpu/core/submit_order.h and BackgroundGate::wouldPark (the Metal backends' commit order and completion frontier).
// Usage: vpgpu_submit_order_tests [group]  (no group = all).
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "vpgpu/core/activity_gauge.h"
#include "vpgpu/core/once_map.h"
#include "vpgpu/core/submit_order.h"

using namespace vpgpu;

namespace {

int failures = 0;
#define CHECK(cond)                                                 \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                   \
    }                                                               \
  } while (0)

// Gate with its log silenced.
struct QuietGate : BackgroundGate {
  QuietGate() { setLog([](const std::string&) {}); }
};

void testWouldPark() {
  QuietGate g;
  CHECK(!g.wouldPark());  // foreground
  g.set(true);            // this thread is "main"
  CHECK(!g.wouldPark());  // main never parks
  bool other = false;
  std::thread([&] { other = g.wouldPark(); }).join();
  CHECK(other);  // any other thread would
  g.set(false);
  other = true;
  std::thread([&] { other = g.wouldPark(); }).join();
  CHECK(!other);
  g.set(true);
  g.shutdown();
  other = true;
  std::thread([&] { other = g.wouldPark(); }).join();
  CHECK(!other);  // shut down: nothing parks
}

void testDeferredCommits() {
  QuietGate g;
  DeferredCommits<int> d;
  // Foreground: a pushed item comes straight back, and the queue is empty after.
  d.push(1);
  CHECK(d.take(g) == std::vector<int>{1});
  CHECK(d.size() == 0 && d.take(g).empty());
  // Background, committing thread that would park: items are kept, in order, across takes.
  g.set(true);
  std::vector<int> a, b;
  std::thread([&] {
    d.push(2);
    a = d.take(g);
    d.push(3);
    b = d.take(g);
  }).join();
  CHECK(a.empty() && b.empty() && d.size() == 2 && d.pending() == 2);
  // Main thread in the background may commit (the gate lets it through): it gets the others' items first, oldest first.
  d.push(4);
  CHECK((d.take(g) == std::vector<int>{2, 3, 4}));
  // After the pause, whoever comes next sends the batch.
  g.set(true);
  std::thread([&] { d.push(5); }).join();
  std::thread([&] { d.push(6); }).join();
  g.set(false);
  std::vector<int> c;
  std::thread([&] { c = d.take(g); }).join();
  CHECK((c == std::vector<int>{5, 6}) && d.pending() == 0);
  // takeAll ignores the gate.
  g.set(true);
  d.push(7);
  std::vector<int> e;
  std::thread([&] {
    d.push(8);
    e = d.takeAll();
  }).join();
  CHECK((e == std::vector<int>{7, 8}) && d.size() == 0 && d.pending() == 0);
}

// The Metal backend's commit machinery (CommitQueue) as the backend drives it. `alive` stands for the backend object: checks on
// it in the threads that call into the model are the "use after free" of a destroyed backend.
struct Model {
  QuietGate g;
  CommitQueue<int> q{g};
  std::atomic<bool> alive{true};
  std::vector<int> committed;  // written with q.lock held
  std::function<void(int)> rec = [this](int v) { committed.push_back(v); };
  // Impl::submit / encodeInterop: counted before the gate can park the thread.
  void submit(int v) {
    CommitQueue<int>::Inside in(q);
    g.waitForeground();
    std::lock_guard l(q.lock);
    CHECK(alive);
    q.push(v, rec);
    CHECK(alive);
  }
  bool flush() { return q.flush(rec); }
  // ~MetalBackend.
  void destroy() {
    q.closing = true;
    g.release();
    {
      std::lock_guard l(q.lock);
      q.commitReady(rec);
    }
    q.waitNoneInside([&] {
      g.release();
      flush();
    });
    alive = false;
  }
};

// A commit handed over in the background must go out when the app is foreground again even if no later commit comes. The
// backend's resume hook runs on the main thread and must not block, not even while another thread holds the order lock: it only
// queues the flush to a worker (a serial queue in the backend), which retries until the lock is free. Here: the hook returns at
// once with the order lock held elsewhere, and the item is committed once the lock is free.
void testResumeFlush() {
  Model m;
  std::mutex qm;
  std::condition_variable qcv;
  int queued = 0;
  bool stop = false;
  std::thread worker([&] {  // the serial flush queue
    std::unique_lock ql(qm);
    for (;;) {
      qcv.wait(ql, [&] { return queued || stop; });
      if (!queued) return;
      --queued;
      ql.unlock();
      while (!m.flush()) std::this_thread::yield();  // the lock is busy: try again
      ql.lock();
    }
  });
  int hookRuns = 0;
  const uint64_t id = m.g.addResumeHook([&] {
    ++hookRuns;
    if (!m.q.deferred.pending()) return;
    std::lock_guard ql(qm);
    ++queued;
    qcv.notify_one();
  });
  m.g.set(true);
  CHECK(hookRuns == 0);  // not on the way to the background
  std::unique_lock held(m.q.lock);  // another thread is encoding: it holds the order lock
  std::thread([&] {
    m.q.deferred.push(1);  // (the thread that handed its commit over, then the lock moved on)
    CHECK(m.q.deferred.take(m.g).empty());  // would park: handed over
  }).join();
  CHECK(m.q.deferred.pending() == 1);
  m.g.set(false);  // returns although the lock is held: the hook did not wait for it
  CHECK(hookRuns == 1);
  held.unlock();
  while (m.q.deferred.pending()) std::this_thread::yield();
  m.g.removeResumeHook(id);
  {
    std::lock_guard ql(qm);
    stop = true;
  }
  qcv.notify_all();
  worker.join();  // the worker is done with the lock after this
  CHECK((m.committed == std::vector<int>{1}));
  m.g.set(true);
  m.g.set(false);
  CHECK(hookRuns == 1);  // removed
}

// flush(): a thread that would park skips even the lock; one that may commit never waits for it; the owner of the lock gets
// "busy" instead of undefined behaviour.
void testFlushRules() {
  Model m;
  m.g.set(true);  // this thread is main
  std::thread([&] {
    std::lock_guard l(m.q.lock);
    m.q.push(1, m.rec);  // would park: held back
  }).join();
  CHECK(m.q.deferred.pending() == 1 && m.committed.empty());
  {
    std::lock_guard l(m.q.lock);
    bool r = false;
    std::thread([&] { r = m.flush(); }).join();
    CHECK(r);  // would park: nothing to do, and the busy lock was not even tried
    CHECK(m.committed.empty());
  }
  m.g.set(false);
  {
    std::lock_guard l(m.q.lock);
    bool r = true;
    std::thread([&] { r = m.flush(); }).join();
    CHECK(!r);                 // may commit but the lock is busy
    CHECK(!m.flush());         // the owner itself: busy, no deadlock
    CHECK(m.committed.empty());
  }
  CHECK(m.flush());
  CHECK((m.committed == std::vector<int>{1}) && m.q.deferred.pending() == 0);
  CHECK(m.flush());  // nothing pending
}

// Teardown while threads are parked in the gate (or have just been let through): the destructor sets `closing`, releases the
// gate and must not return before they left the call, and what they handed over is committed (with take(gate) it would stay
// held back: the gate still says "background" for them). `alive` fails the check if a thread still runs in the model afterwards.
void testClosingTeardown() {
  for (int round = 0; round < 20; ++round) {
    Model m;
    m.g.set(true);  // this thread is main; the others park
    const int n = 3;
    std::vector<std::thread> ts;
    for (int i = 0; i < n; ++i) ts.emplace_back([&m, i] { m.submit(i); });
    while (m.g.blockedTotal() < uint64_t(n)) std::this_thread::yield();  // all parked (their Inside is counted)
    CHECK(m.q.insideCount() == uint32_t(n));
    m.destroy();
    CHECK(m.q.insideCount() == 0);
    CHECK(m.committed.size() == size_t(n) && m.q.deferred.pending() == 0);
    for (auto& t : ts) t.join();
  }
}

// The scenario of the audit (C6): a thread that cannot commit must not sit on the order lock. Model: an order lock, a "submit"
// thread that finishes encoding in the background, and the main thread that ends the pause but first needs the order lock
// to commit interop work. With parking under the lock this never ends; with the hand-over both finish.
void testNoParkUnderOrderLock() {
  QuietGate g;
  DeferredCommits<int> d;
  std::mutex order;
  std::atomic<bool> submitDone{false}, mainDone{false};
  std::vector<int> committed;
  std::mutex committedMutex;
  auto commit = [&](std::vector<int> v) {
    std::lock_guard l(committedMutex);
    committed.insert(committed.end(), v.begin(), v.end());
  };
  g.set(true);  // this (test) thread is main
  std::thread submit([&] {
    std::lock_guard l(order);
    d.push(1);
    commit(d.take(g));  // would park: hands over, releases the lock, returns
    submitDone = true;
  });
  const auto t0 = std::chrono::steady_clock::now();
  while (!submitDone && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5)) std::this_thread::yield();
  CHECK(submitDone);
  {  // main: interop commit in the background, needs the order lock
    std::lock_guard l(order);
    d.push(2);
    commit(d.take(g));
    mainDone = true;
  }
  submit.join();
  CHECK(mainDone);
  CHECK((committed == std::vector<int>{1, 2}));
}

// The hazard itself (C6), as the old code did it: parking in the gate while holding the order lock. The thread that must end the
// pause cannot take the lock for as long as the holder is parked, so it can never end it: shown here by a timed lock that fails
// (the test then resumes the gate by hand so it can finish).
void testParkUnderLockIsADeadlock() {
  QuietGate g;
  std::timed_mutex order;
  std::atomic<bool> holding{false};
  g.set(true);  // this thread is main
  std::thread submit([&] {
    std::lock_guard l(order);
    holding = true;
    g.waitForeground();  // parks: nobody has resumed the gate
  });
  const auto t0 = std::chrono::steady_clock::now();
  while (!holding && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5)) std::this_thread::yield();
  CHECK(holding);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));  // let it reach the park
  const bool got = order.try_lock_for(std::chrono::milliseconds(300));
  CHECK(!got);  // main cannot take the lock the parked thread holds, so it could not have resumed the gate either
  if (got) order.unlock();
  g.set(false);  // by hand
  submit.join();
}

void testCompletionFrontier() {
  {  // nothing submitted
    CompletionFrontier f;
    CHECK(f.completed() == 0);
  }
  {  // work frames in order
    CompletionFrontier f;
    f.submitted(0, true);
    f.submitted(1, true);
    CHECK(f.completed() == 0);
    f.finished(0);
    CHECK(f.completed() == 1);
    f.finished(1);
    CHECK(f.completed() == 2);
  }
  {  // an empty frame with nothing in flight completes at once
    CompletionFrontier f;
    f.submitted(0, false);
    CHECK(f.completed() == 1);
    f.submitted(1, false);
    CHECK(f.completed() == 2);
  }
  {  // an empty frame behind an in-flight one must not declare it finished
    CompletionFrontier f;
    f.submitted(0, true);
    f.submitted(1, false);
    f.submitted(2, false);
    CHECK(f.completed() == 0);
    f.finished(0);
    CHECK(f.completed() == 3);  // frame 0 done, 1 and 2 had nothing
  }
  {  // empty frames between two in-flight ones
    CompletionFrontier f;
    f.submitted(0, true);
    f.submitted(1, false);
    f.submitted(2, true);
    CHECK(f.completed() == 0);
    f.finished(0);
    CHECK(f.completed() == 2);  // frame 2 still runs
    f.submitted(3, false);
    CHECK(f.completed() == 2);
    f.finished(2);
    CHECK(f.completed() == 4);
  }
  {  // never goes backwards, even with a repeated or stale finish
    CompletionFrontier f;
    f.submitted(0, true);
    f.submitted(1, true);
    f.finished(1);
    CHECK(f.completed() == 2);
    f.finished(0);
    CHECK(f.completed() == 2);
  }
  {  // calls out of index order (the indices come from an atomic counter, the calls from different threads): the frontier follows
     // the queue order of the command buffers for finished() and the indices for what is still unknown
    CompletionFrontier f;
    f.submitted(1, true);
    CHECK(f.completed() == 0);  // frame 0 was not handed over yet
    f.submitted(0, true);       // queue order is now 1, 0
    CHECK(f.completed() == 0);
    f.finished(1);
    CHECK(f.completed() == 0);  // frame 0 is still in flight although a higher index finished
    f.finished(0);
    CHECK(f.completed() == 2);
  }
  {  // an empty frame that arrives before an earlier one is not complete: the earlier frame is not even known yet
    CompletionFrontier f;
    f.submitted(1, false);
    CHECK(f.completed() == 0);
    f.submitted(0, true);
    CHECK(f.completed() == 0);
    f.finished(0);
    CHECK(f.completed() == 2);
  }
  {  // gaps closed in any order; a repeated index changes nothing
    CompletionFrontier f;
    f.submitted(3, false);
    f.submitted(0, false);
    CHECK(f.completed() == 1);
    f.submitted(2, false);
    CHECK(f.completed() == 1);
    f.submitted(2, false);
    CHECK(f.completed() == 1);
    f.submitted(1, false);
    CHECK(f.completed() == 4);
  }
  {  // a repeated index with work is ignored: a second entry in the queue could never finish
    CompletionFrontier f;
    f.submitted(0, true);
    f.submitted(0, true);
    f.finished(0);
    CHECK(f.completed() == 1);
    f.submitted(1, true);
    f.finished(1);
    f.submitted(1, true);  // repeat of a frame that already finished
    CHECK(f.completed() == 2);
    f.submitted(2, false);
    CHECK(f.completed() == 3);
  }
  {  // a repeat of a late frame (gap closed once) is ignored too
    CompletionFrontier f;
    f.submitted(2, true);
    f.submitted(0, true);
    f.submitted(0, true);
    f.submitted(1, false);
    f.finished(2);
    f.finished(0);
    CHECK(f.completed() == 3);
  }
  {  // indices that never arrive: the pending ranges are bounded, the oldest is given up and the frontier moves on
    CompletionFrontier f;
    f.submitted(0, false);
    CHECK(f.completed() == 1);
    // every other index is missing: 1, 3, 5, ... never come
    for (uint64_t i = 2; i < 2 * 5000; i += 2) f.submitted(i, false);
    CHECK(f.pendingGaps() == CompletionFrontier::kMaxGaps);  // the bound is really held
    CHECK(f.completed() > 1);       // not frozen at the first missing index
    CHECK(f.completed() < 2 * 5000);  // but not past the newest ranges still pending
    const uint64_t c = f.completed();
    f.submitted(2 * 5000 - 1, true);  // the newest missing index still counts: it holds the frontier... and finishes
    f.finished(2 * 5000 - 1);
    CHECK(f.completed() >= c);
    f.submitted(1, true);  // a frame of a given-up range: ignored (taken for a repeat), no stale entry in flight
    CHECK(f.completed() >= c);
  }
  {  // the partition path (a late frame closing a gap in its middle) is bounded too
    CompletionFrontier f;
    f.submitted(10000000, false);
    CHECK(f.pendingGaps() == 1 && f.completed() == 0);
    size_t most = 0;
    for (uint64_t i = 2; i <= 200000; i += 2) {
      f.submitted(i, false);
      most = std::max(most, f.pendingGaps());
    }
    CHECK(most <= CompletionFrontier::kMaxGaps);
    CHECK(most == CompletionFrontier::kMaxGaps);  // it did grow to the bound
    CHECK(f.completed() > 0);  // and the frontier moved on instead of freezing
  }
  {  // against a model, random arrival order: completed() is the lowest frame not yet handed over or not yet finished
    uint64_t seed = 88172645463325252ull;
    auto rnd = [&] { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
    for (int round = 0; round < 300; ++round) {
      const uint64_t n = 1 + rnd() % 12;
      std::vector<uint64_t> order(n);
      for (uint64_t i = 0; i < n; ++i) order[i] = i;
      for (uint64_t i = n; i > 1; --i) std::swap(order[i - 1], order[rnd() % i]);
      std::vector<bool> work(n);
      for (uint64_t i = 0; i < n; ++i) work[i] = rnd() & 1;
      CompletionFrontier f;
      std::vector<bool> handed(n, false), done(n, true);
      std::deque<uint64_t> queue;  // work frames in queue order, not finished
      uint64_t next = 0;
      auto model = [&] {
        uint64_t m = n;
        for (uint64_t i = 0; i < n; ++i)
          if (!handed[i] || !done[i]) { m = i; break; }
        return m;
      };
      while (next < n || !queue.empty()) {
        if (next < n && (queue.empty() || rnd() % 3)) {
          const uint64_t i = order[next++];
          handed[i] = true;
          if (work[i]) { done[i] = false; queue.push_back(i); }
          f.submitted(i, work[i]);
        } else {
          const size_t k = rnd() % queue.size();  // some command buffer finishes: it and every earlier one in the queue
          for (size_t j = 0; j <= k; ++j) done[queue[j]] = true;
          const uint64_t i = queue[k];
          queue.erase(queue.begin(), queue.begin() + long(k) + 1);
          f.finished(i);
        }
        CHECK(f.completed() == model());
        if (f.completed() != model()) break;
      }
    }
  }
  {  // concurrent: a reader never sees a frame done before its finish, writers on different threads
    CompletionFrontier f;
    std::atomic<uint64_t> finishedUpTo{0};  // frames [0, n) whose finished() call has been made
    std::atomic<bool> stop{false}, bad{false};
    std::thread reader([&] {
      while (!stop) {
        const uint64_t before = finishedUpTo.load();
        const uint64_t c = f.completed();
        // completed() may run ahead of `finishedUpTo` only by the call in flight: finished(n) publishes before the counter.
        if (c < before) bad = true;
      }
    });
    constexpr uint64_t kFrames = 20000;
    std::atomic<uint64_t> submittedWork{0};
    std::thread gpu([&] {
      for (uint64_t n = 0; n < kFrames;) {
        if (n < submittedWork.load()) {
          f.finished(n);
          finishedUpTo = n + 1;
          ++n;
        }
      }
    });
    for (uint64_t i = 0; i < kFrames; ++i) {
      f.submitted(i, true);
      submittedWork = i + 1;
    }
    gpu.join();
    stop = true;
    reader.join();
    CHECK(!bad);
    CHECK(f.completed() == kFrames);
  }
}

// ActivityGauge::waitInactive replaces the 2 ms sleep loop of the prewarm: returns at once when idle, times out while busy, and is
// woken by the leave() that ends the last piece of work.
void testGaugeWait() {
  using namespace std::chrono;
  ActivityGauge g;
  CHECK(g.waitInactive(milliseconds(1)));  // idle: immediate
  g.enter();
  auto t0 = steady_clock::now();
  CHECK(!g.waitInactive(milliseconds(20)));  // busy: times out
  CHECK(steady_clock::now() - t0 >= milliseconds(19));
  // Woken by leave(): far earlier than the timeout. Repeated to catch a lost wake-up.
  for (int i = 0; i < 200; ++i) {
    g.enter();  // (the count is 2 now; the helper leaves one, this loop leaves the other)
    std::thread t([&] {
      std::this_thread::sleep_for(microseconds(200));
      g.leave();
      g.leave();
    });
    const auto s = steady_clock::now();
    const bool idle = g.waitInactive(seconds(5));
    CHECK(idle);
    CHECK(steady_clock::now() - s < seconds(2));
    t.join();
    g.enter();  // back to one busy piece for the next round
  }
  g.leave();
  CHECK(g.active() == 0);
  {  // an unbounded wait (the largest duration) must not overflow the clock arithmetic into "already over"
    g.enter();
    std::thread t([&] {
      std::this_thread::sleep_for(milliseconds(100));
      g.leave();
    });
    const auto s = steady_clock::now();
    const bool idle = g.waitInactive(nanoseconds::max());
    CHECK(idle);
    CHECK(steady_clock::now() - s >= milliseconds(90));
    t.join();
    g.enter();
    CHECK(!g.waitInactive(nanoseconds(-5)));  // a negative timeout is a poll
    CHECK(!g.waitInactive(nanoseconds::min()));  // so is the most negative one (it must not overflow either)
    g.leave();
  }
}

// OnceMap: each key is built once even when the warm job and the submit thread ask together; other keys build in parallel; a
// failure that is not kept is built again by the next caller; a waiter takes the builder's result.
void testOnceMap() {
  using Map = OnceMap<std::string, int>;
  {  // basic: built once, stored
    Map m;
    int builds = 0;
    CHECK(m.get("a", [&] { ++builds; return Map::Built{7, true}; }) == 7);
    CHECK(m.get("a", [&] { ++builds; return Map::Built{9, true}; }) == 7);
    CHECK(builds == 1 && m.size() == 1);
    int v = 0;
    CHECK(m.peek("a", v) && v == 7 && !m.peek("b", v));
  }
  {  // not kept: asked again builds again
    Map m;
    int builds = 0;
    CHECK(m.get("a", [&] { ++builds; return Map::Built{0, false}; }) == 0);
    CHECK(m.get("a", [&] { ++builds; return Map::Built{5, true}; }) == 5);
    CHECK(builds == 2 && m.size() == 1);
  }
  {  // a throwing build releases the key
    Map m;
    bool threw = false;
    try {
      m.get("a", []() -> Map::Built { throw 1; });
    } catch (int) {
      threw = true;
    }
    CHECK(threw);
    CHECK(m.get("a", [] { return Map::Built{3, true}; }) == 3);
  }
  {  // same key from two threads: one build, both get its value; a different key is not blocked by it
    Map m;
    std::atomic<int> builds{0};
    std::atomic<bool> inBuild{false}, release{false};
    int r1 = 0, r2 = 0, other = 0;
    std::thread t1([&] {
      r1 = m.get("k", [&] {
        ++builds;
        inBuild = true;
        while (!release) std::this_thread::yield();
        return Map::Built{11, true};
      });
    });
    while (!inBuild) std::this_thread::yield();
    std::thread t2([&] { r2 = m.get("k", [&] { ++builds; return Map::Built{99, true}; }); });
    other = m.get("j", [&] { ++builds; return Map::Built{4, true}; });  // while "k" is still building
    CHECK(other == 4);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    release = true;
    t1.join();
    t2.join();
    CHECK(r1 == 11 && r2 == 11 && builds == 2);
  }
  {  // a waiter whose builder kept nothing builds the key itself
    Map m;
    std::atomic<bool> inBuild{false}, release{false};
    int r1 = 0, r2 = 0;
    std::thread t1([&] {
      r1 = m.get("k", [&] {
        inBuild = true;
        while (!release) std::this_thread::yield();
        return Map::Built{-1, false};
      });
    });
    while (!inBuild) std::this_thread::yield();
    std::thread t2([&] { r2 = m.get("k", [] { return Map::Built{8, true}; }); });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    release = true;
    t1.join();
    t2.join();
    CHECK(r1 == -1 && r2 == 8);
  }
  {  // stress: many threads, few keys, every key built exactly once
    Map m;
    std::atomic<int> builds{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < 8; ++t)
      ts.emplace_back([&, t] {
        for (int i = 0; i < 2000; ++i) {
          const std::string k = std::to_string((i + t) % 16);
          const int v = m.get(k, [&] { ++builds; return Map::Built{(i + t) % 16, true}; });
          if (v != (i + t) % 16) ++failures;
        }
      });
    for (auto& t : ts) t.join();
    CHECK(builds == 16 && m.size() == 16);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::pair<const char*, std::function<void()>>> groups = {
      {"wouldpark", testWouldPark},
      {"deferred", testDeferredCommits},
      {"resumeflush", testResumeFlush},
      {"flushrules", testFlushRules},
      {"closing", testClosingTeardown},
      {"noparkunderlock", testNoParkUnderOrderLock},
      {"parkunderlock", testParkUnderLockIsADeadlock},
      {"frontier", testCompletionFrontier},
      {"gaugewait", testGaugeWait},
      {"oncemap", testOnceMap}};
  bool ran = false;
  for (const auto& [name, fn] : groups) {
    if (argc > 1 && std::strcmp(argv[1], name) != 0) continue;
    std::printf("[%s]\n", name);
    fn();
    ran = true;
  }
  if (!ran) std::printf("unknown group\n");
  std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
  return failures || !ran ? 1 : 0;
}
