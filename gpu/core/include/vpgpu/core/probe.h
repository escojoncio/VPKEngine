// Diagnostic probe: one place where the engine (and its host) publish counters, gauges, timers and
// distributions, and one sampler that turns them, once per interval, into
//   - one log line per family:  "VPKENGINE_PROBE <FAM> i=<n> t=<s> dt=<s> key=value ..." with every counter
//     both CUMULATIVE (key) and as the DELTA of the interval (key_d); gauges as their current value; timers as
//     calls, cumulative ms, ms per second of the interval and the longest call; distributions (microseconds by
//     convention) as p50/p90/p99/max of the interval and ms per second;
//   - one CSV row (one column per metric, header rewritten when metrics appear later) in a file the host opens;
//   - a final "VPKENGINE_PROBE SUMMARY" with peaks and means.
// Lines carry the interval index "i=" so a logger that drops repeated messages never sees two equal ones.
//
// Cost: producers use relaxed atomics, and timers/scopes do nothing at all (one relaxed load) while the probe is
// off. Off by default; the host turns it on (setEnabled) with its diagnostics switch, or VPKENGINE_PROBE=1 does.
// Nothing here changes what the engine does.
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vpgpu::probe {

// ---- Switch and clock -------------------------------------------------------------------------------------------

bool enabled();
void setEnabled(bool on);
uint64_t nowNs();  // steady clock

// ---- Metrics ----------------------------------------------------------------------------------------------------

// A cumulative count. Producers add; a publisher of a total that is kept elsewhere stores it with set().
class Counter {
 public:
  void add(uint64_t n = 1) { v_.fetch_add(n, std::memory_order_relaxed); }
  void set(uint64_t v) { v_.store(v, std::memory_order_relaxed); }
  uint64_t get() const { return v_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint64_t> v_{0};
};

// A value at the moment of the sample.
class Gauge {
 public:
  void set(double v) { v_.store(v, std::memory_order_relaxed); }
  // For a running total of signed amounts (the footprint change of big allocations).
  void add(double d) {
    double cur = v_.load(std::memory_order_relaxed);
    while (!v_.compare_exchange_weak(cur, cur + d, std::memory_order_relaxed)) {
    }
  }
  double get() const { return v_.load(std::memory_order_relaxed); }

 private:
  std::atomic<double> v_{0};
};

// Time spent in something: calls, total, and the longest call of the interval.
class Timer {
 public:
  void add(uint64_t ns) {
    calls_.fetch_add(1, std::memory_order_relaxed);
    ns_.fetch_add(ns, std::memory_order_relaxed);
    noteMax(ns);
  }
  // For a publisher of time summed elsewhere (milliseconds, `calls` calls). `maxMs` is the longest single call, when the
  // publisher knows it; without it the maximum is unknown (not 0, not the whole interval) and stays out of the lines.
  void addMs(double ms, uint64_t calls = 1, double maxMs = 0) {
    calls_.fetch_add(calls, std::memory_order_relaxed);
    ns_.fetch_add(ms > 0 ? uint64_t(ms * 1e6) : 0, std::memory_order_relaxed);
    if (maxMs > 0) noteMax(uint64_t(maxMs * 1e6));
  }
  // Sampled timing (ScopedSampled): every call is counted, one in `weight` is timed and stands for `weight` of them.
  void addCall() { calls_.fetch_add(1, std::memory_order_relaxed); }
  void addEstimated(uint64_t ns, unsigned weight) {
    ns_.fetch_add(ns * weight, std::memory_order_relaxed);
    noteMax(ns);
  }
  uint64_t calls() const { return calls_.load(std::memory_order_relaxed); }
  uint64_t ns() const { return ns_.load(std::memory_order_relaxed); }
  // The longest call of the interval in ns (cleared); `known` is false while no call has reported one.
  uint64_t takeMax(bool* known = nullptr) {
    if (known) *known = known_.load(std::memory_order_relaxed);
    return max_.exchange(0, std::memory_order_relaxed);
  }

 private:
  void noteMax(uint64_t ns) {
    known_.store(true, std::memory_order_relaxed);
    uint64_t cur = max_.load(std::memory_order_relaxed);
    while (ns > cur && !max_.compare_exchange_weak(cur, ns, std::memory_order_relaxed)) {
    }
  }
  std::atomic<uint64_t> calls_{0}, ns_{0}, max_{0};
  std::atomic<bool> known_{false};
};

// The distribution of positive integers (microseconds by convention) over an interval: exact below 16, then
// four buckets per power of two (about 12 % wide). The sampler takes and clears it.
class Hist {
 public:
  static constexpr unsigned kBuckets = 192;
  struct Snapshot {
    uint64_t n = 0, sum = 0, max = 0;
    uint64_t p50 = 0, p90 = 0, p99 = 0;
  };
  void add(uint64_t v) {
    b_[bucketOf(v)].fetch_add(1, std::memory_order_relaxed);
    n_.fetch_add(1, std::memory_order_relaxed);
    sum_.fetch_add(v, std::memory_order_relaxed);
    uint64_t cur = max_.load(std::memory_order_relaxed);
    while (v > cur && !max_.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
    }
  }
  void addMs(double ms) { add(ms > 0 ? uint64_t(ms * 1000.0 + 0.5) : 0); }
  Snapshot take();
  uint64_t total() const { return total_.load(std::memory_order_relaxed); }  // observations since the start
  static unsigned bucketOf(uint64_t v);
  static uint64_t bucketMid(unsigned bucket);

 private:
  std::atomic<uint64_t> b_[kBuckets]{};
  std::atomic<uint64_t> n_{0}, sum_{0}, max_{0}, total_{0};
};

// Times its scope into a Timer / Hist (microseconds). Nothing is read while the probe is off.
class Scoped {
 public:
  explicit Scoped(Timer& t) : t_(enabled() ? &t : nullptr), t0_(t_ ? nowNs() : 0) {}
  ~Scoped() {
    if (t_) t_->add(nowNs() - t0_);
  }
  Scoped(const Scoped&) = delete;
  Scoped& operator=(const Scoped&) = delete;

 private:
  Timer* t_;
  uint64_t t0_;
};
// For paths so hot that two clock reads per call would be a measurable cost of their own: every call is counted, one in
// kEvery is timed, and its time stands for kEvery calls. The ms/s of such a timer is an estimate (and its max is the
// longest of the sampled calls); the call count is exact.
class ScopedSampled {
 public:
  static constexpr unsigned kEvery = 16;
  explicit ScopedSampled(Timer& t) {
    if (!enabled()) return;
    t.addCall();
    static thread_local unsigned tick = 0;
    if (tick++ % kEvery == 0) {
      t_ = &t;
      t0_ = nowNs();
    }
  }
  ~ScopedSampled() {
    if (t_) t_->addEstimated(nowNs() - t0_, kEvery);
  }
  ScopedSampled(const ScopedSampled&) = delete;
  ScopedSampled& operator=(const ScopedSampled&) = delete;

 private:
  Timer* t_ = nullptr;
  uint64_t t0_ = 0;
};

class ScopedHist {
 public:
  explicit ScopedHist(Hist& h) : h_(enabled() ? &h : nullptr), t0_(h_ ? nowNs() : 0) {}
  ~ScopedHist() {
    if (h_) h_->add((nowNs() - t0_) / 1000);
  }
  ScopedHist(const ScopedHist&) = delete;
  ScopedHist& operator=(const ScopedHist&) = delete;

 private:
  Hist* h_;
  uint64_t t0_;
};

// GPU busy time as the UNION of the intervals of work, not their sum (command buffers overlap in flight: the sum of their GPU
// times exceeds the wall time). Intervals are reported in completion order (a queue: each ends no earlier than the one before);
// the time not already covered by an earlier one is added to `timer` (calls = intervals, ms/s of the line = busy fraction).
class BusyUnion {
 public:
  explicit BusyUnion(Timer& timer) : timer_(timer) {}
  void add(uint64_t startNs, uint64_t endNs) {
    uint64_t cur = covered_.load(std::memory_order_relaxed);
    uint64_t from;
    do {
      if (endNs <= cur) {
        timer_.add(0);
        return;
      }
      from = startNs > cur ? startNs : cur;
    } while (!covered_.compare_exchange_weak(cur, endNs, std::memory_order_relaxed));
    timer_.add(endNs - from);
  }

 private:
  Timer& timer_;
  std::atomic<uint64_t> covered_{0};
};

// ---- Registry and sampler ---------------------------------------------------------------------------------------

class Registry {
 public:
  Registry();
  // Metrics are found (or made) by family and name; the reference stays valid for the registry's life.
  Counter& counter(std::string_view family, std::string_view name);
  Gauge& gauge(std::string_view family, std::string_view name);
  Timer& timer(std::string_view family, std::string_view name);
  Hist& hist(std::string_view family, std::string_view name);
  // A text value shown in the log line (spaces become '_'), not in the CSV. Replaced at each call.
  void info(std::string_view family, std::string_view name, std::string_view text);

  // Opens the CSV (truncating it). False if it cannot be written. "" closes it.
  bool openCsv(const std::string& path);
  const std::string& csvPath() const { return csvPath_; }

  // Closes the interval that ends at `nowSeconds` (any monotonic clock; the first interval starts at construction
  // or reset). Returns the log lines (one per family that has something) and appends the CSV row.
  // The lines are complete (every key of every metric) and cut into pieces of at most ~2 KB. With `compact`, it is filled
  // with the lines for a log that must stay small: only the keys that moved in the interval (counters with a delta,
  // timers and histograms with calls, gauges and info that changed), and every kFullEvery-th interval (the first
  // included) all of them. The CSV always gets everything.
  std::vector<std::string> sample(double nowSeconds, std::vector<std::string>* compact = nullptr);
  std::vector<std::string> sample(std::vector<std::string>* compact = nullptr);  // with the registry's own clock
  static constexpr uint64_t kFullEvery = 15;
  uint64_t samples() const;
  // The summary lines (peaks and means over every interval so far), numbered so no two are equal.
  std::vector<std::string> summary();

  // Forget everything (tests).
  void reset();

  // A name made of [a-z0-9_] from any text (runs of other characters become one '_'), at most 48 characters.
  static std::string sanitize(std::string_view text);
  // A number as the lines print it: integral values without decimals, others with up to 3.
  static std::string number(double v);

 private:
  enum class Kind : uint8_t { Counter, Gauge, Timer, Hist, Info };
  struct Agg {
    double peak = 0, sum = 0, last = 0;
    uint64_t n = 0;
    void put(double v) {
      peak = n == 0 || v > peak ? v : peak;
      sum += v;
      last = v;
      ++n;
    }
    double mean() const { return n ? sum / double(n) : 0; }
  };
  struct Entry {
    Kind kind;
    size_t index;  // in its deque
    std::string family, name;
    uint64_t lastCum = 0, lastCalls = 0, lastNs = 0;
    Agg a, b, c;
    std::string text;
    size_t fam = 0;  // in families_
    // What the compact log line last showed of it (gauges and info are shown when they change).
    double shownGauge = 0;
    bool gaugeShown = false;
    std::string shownText;
  };
  // A line per family is cut into pieces of at most this many bytes: "<head> part=i/n <keys>".
  static constexpr size_t kLineBytes = 2000;
  static void chunk(const std::string& head, const std::vector<std::string>& tokens, std::vector<std::string>& out);
  Entry& entry(Kind kind, std::string_view family, std::string_view name);
  void writeCsv(const std::vector<double>& row, bool grew, bool allKept);
  static void putColumns(Kind kind, const std::string& family, const std::string& name, std::vector<std::string>& cols);

  mutable std::mutex mutex_;
  std::deque<Counter> counters_;
  std::deque<Gauge> gauges_;
  std::deque<Timer> timers_;
  std::deque<Hist> hists_;
  std::deque<Entry> entries_;
  std::unordered_map<std::string, size_t> byName_;
  std::vector<std::string> families_;  // in order of first use
  std::vector<std::string> columns_;
  std::vector<std::vector<double>> rows_;  // kept for a rewrite when columns appear (bounded)
  std::string csvPath_;
  void* csv_ = nullptr;  // FILE*
  uint64_t index_ = 0, summaries_ = 0;
  double start_ = 0, prev_ = 0, last_ = 0;
  bool started_ = false;
  uint64_t startNs_ = 0;
  size_t columned_ = 0;  // entries whose columns are in columns_
  static constexpr size_t kMaxRows = 4096;
};

Registry& registry();

// ---- Shorthand for producers ------------------------------------------------------------------------------------

inline Counter& counter(std::string_view family, std::string_view name) { return registry().counter(family, name); }
inline Gauge& gauge(std::string_view family, std::string_view name) { return registry().gauge(family, name); }
inline Timer& timer(std::string_view family, std::string_view name) { return registry().timer(family, name); }
inline Hist& hist(std::string_view family, std::string_view name) { return registry().hist(family, name); }

// A mutex that measures the time threads wait for it (LOCK family: <name>_acq acquisitions, <name>_wait the
// contended ones with their wait time) while the probe is on; a plain mutex otherwise. Works with the standard
// lock guards and condition_variable_any.
template <class M>
class TimedMutex {
 public:
  explicit TimedMutex(const char* name) : name_(name) {}
  void lock() {
    if (!enabled()) {
      m_.lock();
      return;
    }
    Slots& s = slots();
    s.acquired->add();
    if (m_.try_lock()) return;
    const uint64_t t0 = nowNs();
    m_.lock();
    s.waited->add(nowNs() - t0);
  }
  bool try_lock() { return m_.try_lock(); }
  void unlock() { m_.unlock(); }

 private:
  struct Slots {
    Counter* acquired;
    Timer* waited;
  };
  Slots& slots() {
    Slots* s = slots_.load(std::memory_order_acquire);
    if (!s) {
      auto* made = new Slots{&counter("LOCK", std::string(name_) + "_acq"), &timer("LOCK", std::string(name_) + "_wait")};
      Slots* none = nullptr;
      if (slots_.compare_exchange_strong(none, made, std::memory_order_acq_rel)) {
        s = made;
      } else {
        delete made;
        s = none;
      }
    }
    return *s;
  }
  M m_;
  const char* name_;
  std::atomic<Slots*> slots_{nullptr};
};

// ---- Memory around big allocations ------------------------------------------------------------------------------

// The process's physical footprint in bytes (task_vm_info on Apple; 0 elsewhere).
uint64_t processFootprint();

// A resource of at least kBigAlloc bytes is about to be created: the footprint (and the device's allocated size,
// which the caller reads) before and after it are taken, and the difference accumulated by category in the ALLOC
// family. begin() returns an inactive token (no syscall) while the probe is off or the resource is small.
inline constexpr uint64_t kBigAlloc = 8ull << 20;
struct AllocToken {
  const char* category = nullptr;
  uint64_t bytes = 0, footprint = 0, device = 0;
  bool active = false;
};
AllocToken bigAllocBegin(const char* category, uint64_t bytes, uint64_t deviceAllocated);
void bigAllocEnd(const AllocToken& token, uint64_t deviceAllocated);

// Records a creation of any size in the ALLOC family (count and bytes of the category).
void allocCreated(const char* category, uint64_t bytes);

}  // namespace vpgpu::probe
