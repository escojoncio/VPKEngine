// Tests for vpgpu/core/submit_order.h and BackgroundGate::wouldPark (the Metal backends' commit order and completion frontier).
// Usage: vpgpu_submit_order_tests [group]  (no group = all).
#include <atomic>
#include <chrono>
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
  CHECK(a.empty() && b.empty() && d.size() == 2);
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
  CHECK((c == std::vector<int>{5, 6}));
  // takeAll ignores the gate.
  g.set(true);
  d.push(7);
  std::vector<int> e;
  std::thread([&] {
    d.push(8);
    e = d.takeAll();
  }).join();
  CHECK((e == std::vector<int>{7, 8}) && d.size() == 0);
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
