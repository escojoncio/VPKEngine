#include "vpgpu/core/probe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace vpgpu::probe {

namespace {
std::atomic<bool>& flag() {
  static std::atomic<bool> on{[] {
    const char* v = std::getenv("VPKENGINE_PROBE");
    return v && v[0] == '1' && v[1] == '\0';
  }()};
  return on;
}
}  // namespace

bool enabled() { return flag().load(std::memory_order_relaxed); }
void setEnabled(bool on) { flag().store(on, std::memory_order_relaxed); }
uint64_t nowNs() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

// ---- Hist -------------------------------------------------------------------------------------------------------

unsigned Hist::bucketOf(uint64_t v) {
  if (v < 16) return unsigned(v);
  if (v >= (1ull << 48)) v = (1ull << 48) - 1;
  const unsigned b = 63u - unsigned(__builtin_clzll(v));
  return 16 + (b - 4) * 4 + unsigned((v >> (b - 2)) & 3);
}

uint64_t Hist::bucketMid(unsigned bucket) {
  if (bucket < 16) return bucket;
  const unsigned k = bucket - 16, b = k / 4 + 4, sub = k % 4;
  return (uint64_t(4 + sub) << (b - 2)) + (1ull << (b - 2)) / 2;
}

Hist::Snapshot Hist::take() {
  uint64_t counts[kBuckets];
  Snapshot s;
  for (unsigned i = 0; i < kBuckets; ++i) {
    counts[i] = b_[i].exchange(0, std::memory_order_relaxed);
    s.n += counts[i];
  }
  n_.store(0, std::memory_order_relaxed);
  s.sum = sum_.exchange(0, std::memory_order_relaxed);
  s.max = max_.exchange(0, std::memory_order_relaxed);
  total_.fetch_add(s.n, std::memory_order_relaxed);
  if (!s.n) return s;
  const auto at = [&](double p) {
    const uint64_t rank = std::max<uint64_t>(1, uint64_t(std::ceil(p * double(s.n))));
    uint64_t seen = 0;
    for (unsigned i = 0; i < kBuckets; ++i) {
      seen += counts[i];
      if (seen >= rank) return std::min(bucketMid(i), s.max);
    }
    return s.max;
  };
  s.p50 = at(0.50);
  s.p90 = at(0.90);
  s.p99 = at(0.99);
  return s;
}

// ---- Registry ---------------------------------------------------------------------------------------------------

Registry::Registry() : startNs_(nowNs()) {}

std::string Registry::sanitize(std::string_view text) {
  std::string out;
  bool gap = false;
  for (char c : text) {
    if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    const bool ok = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
    if (ok) {
      if (gap && !out.empty()) out += '_';
      gap = false;
      out += c;
    } else {
      gap = true;
    }
  }
  if (out.empty()) return "_";
  if (out.size() <= 48) return out;
  // Too long: the first 41 characters, then a short hash of the whole name, so that names sharing a long prefix stay apart.
  uint32_t h = 2166136261u;
  for (char c : out) h = (h ^ uint8_t(c)) * 16777619u;
  char suffix[16];
  std::snprintf(suffix, sizeof suffix, "_%06x", h & 0xFFFFFFu);
  return out.substr(0, 41) + suffix;
}

std::string Registry::number(double v) {
  if (!std::isfinite(v)) return "0";
  char b[48];
  if (std::fabs(v) < 1e15 && v == std::floor(v)) {
    std::snprintf(b, sizeof b, "%.0f", v);
    return b;
  }
  std::snprintf(b, sizeof b, "%.3f", v);
  std::string s = b;
  while (!s.empty() && s.back() == '0') s.pop_back();
  if (!s.empty() && s.back() == '.') s.pop_back();
  return s;
}

Registry::Entry& Registry::entry(Kind kind, std::string_view family, std::string_view name) {
  std::string key;
  key.reserve(family.size() + name.size() + 2);
  key.append(family).push_back('\x1f');
  key.append(name);
  key.push_back(char('0' + int(kind)));
  std::lock_guard lock(mutex_);
  const auto found = byName_.find(key);
  if (found != byName_.end()) return entries_[found->second];
  Entry e;
  e.kind = kind;
  e.family = family;
  e.name = name;
  switch (kind) {
    case Kind::Counter: counters_.emplace_back(), e.index = counters_.size() - 1; break;
    case Kind::Gauge: gauges_.emplace_back(), e.index = gauges_.size() - 1; break;
    case Kind::Timer: timers_.emplace_back(), e.index = timers_.size() - 1; break;
    case Kind::Hist: hists_.emplace_back(), e.index = hists_.size() - 1; break;
    case Kind::Info: e.index = 0; break;
  }
  const auto f = std::find(families_.begin(), families_.end(), e.family);
  e.fam = size_t(f - families_.begin());
  if (f == families_.end()) families_.push_back(e.family);
  entries_.push_back(std::move(e));
  byName_.emplace(std::move(key), entries_.size() - 1);
  return entries_.back();
}

Counter& Registry::counter(std::string_view family, std::string_view name) {
  Entry& e = entry(Kind::Counter, family, name);
  std::lock_guard lock(mutex_);
  return counters_[e.index];
}
Gauge& Registry::gauge(std::string_view family, std::string_view name) {
  Entry& e = entry(Kind::Gauge, family, name);
  std::lock_guard lock(mutex_);
  return gauges_[e.index];
}
Timer& Registry::timer(std::string_view family, std::string_view name) {
  Entry& e = entry(Kind::Timer, family, name);
  std::lock_guard lock(mutex_);
  return timers_[e.index];
}
Hist& Registry::hist(std::string_view family, std::string_view name) {
  Entry& e = entry(Kind::Hist, family, name);
  std::lock_guard lock(mutex_);
  return hists_[e.index];
}
void Registry::info(std::string_view family, std::string_view name, std::string_view text) {
  Entry& e = entry(Kind::Info, family, name);
  std::string t(text);
  for (char& c : t)
    if (c == ' ' || c == '\n' || c == '\t') c = '_';
  std::lock_guard lock(mutex_);
  e.text = std::move(t);
}

void Registry::putColumns(Kind kind, const std::string& family, const std::string& name, std::vector<std::string>& cols) {
  const std::string p = family + "." + name;
  switch (kind) {
    case Kind::Counter: cols.push_back(p), cols.push_back(p + "_d"); break;
    case Kind::Gauge: cols.push_back(p); break;
    case Kind::Timer: cols.push_back(p + "_n_d"), cols.push_back(p + "_ms_s"), cols.push_back(p + "_max_us"); break;
    case Kind::Hist:
      cols.push_back(p + "_n_d"), cols.push_back(p + "_p50"), cols.push_back(p + "_p90"), cols.push_back(p + "_p99"),
          cols.push_back(p + "_max"), cols.push_back(p + "_ms_s");
      break;
    case Kind::Info: break;
  }
}

bool Registry::openCsv(const std::string& path) {
  std::lock_guard lock(mutex_);
  if (csv_) std::fclose(static_cast<FILE*>(csv_));
  csv_ = nullptr;
  csvPath_.clear();
  if (path.empty()) return true;
  csv_ = std::fopen(path.c_str(), "w");
  if (!csv_) return false;
  csvPath_ = path;
  return true;
}

void Registry::writeCsv(const std::vector<double>& row, bool grew, bool allKept) {
  FILE* f = static_cast<FILE*>(csv_);
  if (!f) return;
  const auto header = [&](FILE* out) {
    std::fputs("i,t,dt", out);
    for (const std::string& c : columns_) std::fprintf(out, ",%s", c.c_str());
    std::fputc('\n', out);
  };
  const auto put = [&](FILE* out, const std::vector<double>& r) {
    for (size_t k = 0; k < 3 + columns_.size(); ++k) {
      if (k) std::fputc(',', out);
      if (k < r.size() && !std::isnan(r[k])) std::fputs(number(r[k]).c_str(), out);  // (NaN: unknown, an empty cell)
    }
    std::fputc('\n', out);
  };
  if (index_ == 1) {
    header(f);
  } else if (grew && allKept && rows_.size() + 1 == index_) {
    // Every earlier row is still in memory: the file is written again with the wider header.
    std::fclose(f);
    f = std::fopen(csvPath_.c_str(), "w");
    csv_ = f;
    if (!f) return;
    header(f);
    for (const auto& r : rows_) put(f, r);
  } else if (grew) {
    header(f);  // (earlier rows were dropped to bound memory: a new header starts a new block)
  }
  put(f, row);
  std::fflush(f);
}

std::vector<std::string> Registry::sample(std::vector<std::string>* compact) {
  return sample(double(nowNs() - startNs_) / 1e9, compact);
}

void Registry::chunk(const std::string& head, const std::vector<std::string>& tokens, std::vector<std::string>& out) {
  if (tokens.empty()) return;
  std::vector<std::string> bodies(1);
  for (const std::string& t : tokens) {
    if (!bodies.back().empty() && head.size() + 16 + bodies.back().size() + t.size() > kLineBytes) bodies.emplace_back();
    bodies.back() += t;
  }
  for (size_t i = 0; i < bodies.size(); ++i) {
    char part[40];
    std::snprintf(part, sizeof part, " part=%zu/%zu", i + 1, bodies.size());
    out.push_back(head + part + bodies[i]);
  }
}

uint64_t Registry::samples() const {
  std::lock_guard lock(mutex_);
  return index_;
}

std::vector<std::string> Registry::sample(double now, std::vector<std::string>* compactOut) {
  std::lock_guard lock(mutex_);
  if (!started_) {
    started_ = true;
    start_ = prev_ = 0;
    last_ = 0;
  }
  const double t = now;
  const double dt = std::max(t - prev_, 0.0);
  const double per = dt > 0 ? dt : 1.0;  // (rates over a zero-length interval read as the delta itself)
  prev_ = t;
  last_ = t;
  ++index_;

  const bool fullDump = (index_ - 1) % kFullEvery == 0;
  std::vector<std::vector<std::string>> body(families_.size()), shown(families_.size());
  std::vector<double> row;
  row.push_back(double(index_));
  row.push_back(t);
  row.push_back(dt);
  for (Entry& e : entries_) {
    bool show = true;  // whether the compact log line carries this metric
    std::vector<std::string>& out = body[e.fam];
    std::vector<std::string>& mark = shown[e.fam];
    size_t first = out.size();
    const auto put = [&](const std::string& key, const std::string& value) { out.push_back(' ' + key + '=' + value); };
    const std::string& n = e.name;
    switch (e.kind) {
      case Kind::Counter: {
        const uint64_t cum = counters_[e.index].get();
        const uint64_t d = cum >= e.lastCum ? cum - e.lastCum : 0;
        e.lastCum = cum;
        show = d != 0;
        put(n, std::to_string(cum));
        put(n + "_d", std::to_string(d));
        row.push_back(double(cum));
        row.push_back(double(d));
        e.a.put(double(d) / per);
        break;
      }
      case Kind::Gauge: {
        const double v = gauges_[e.index].get();
        show = !e.gaugeShown || v != e.shownGauge;
        if (show) e.shownGauge = v, e.gaugeShown = true;
        put(n, number(v));
        row.push_back(v);
        e.a.put(v);
        break;
      }
      case Kind::Timer: {
        Timer& tm = timers_[e.index];
        const uint64_t calls = tm.calls(), ns = tm.ns();
        const uint64_t dCalls = calls >= e.lastCalls ? calls - e.lastCalls : 0, dNs = ns >= e.lastNs ? ns - e.lastNs : 0;
        e.lastCalls = calls, e.lastNs = ns;
        bool maxKnown = false;
        const uint64_t maxNs = tm.takeMax(&maxKnown);
        const double msS = double(dNs) / 1e6 / per, maxUs = double(maxNs) / 1e3;
        show = dCalls != 0;
        put(n + "_n", std::to_string(calls));
        put(n + "_n_d", std::to_string(dCalls));
        put(n + "_ms", number(double(ns) / 1e6));
        put(n + "_ms_s", number(msS));
        put(n + "_max_us", maxKnown ? number(maxUs) : std::string("-"));  // "-": the publisher does not know the longest call
        row.push_back(double(dCalls));
        row.push_back(msS);
        row.push_back(maxKnown ? maxUs : std::nan(""));
        e.a.put(msS);
        if (maxKnown) e.b.put(maxUs);
        break;
      }
      case Kind::Hist: {
        Hist& h = hists_[e.index];
        const Hist::Snapshot s = h.take();
        const double msS = double(s.sum) / 1e3 / per;
        show = s.n != 0;
        put(n + "_n", std::to_string(h.total()));
        put(n + "_n_d", std::to_string(s.n));
        put(n + "_p50", std::to_string(s.p50));
        put(n + "_p90", std::to_string(s.p90));
        put(n + "_p99", std::to_string(s.p99));
        put(n + "_max", std::to_string(s.max));
        put(n + "_ms_s", number(msS));
        row.push_back(double(s.n));
        row.push_back(double(s.p50));
        row.push_back(double(s.p90));
        row.push_back(double(s.p99));
        row.push_back(double(s.max));
        row.push_back(msS);
        e.a.put(double(s.p50));
        e.b.put(double(s.p99));
        e.c.put(double(s.max));
        break;
      }
      case Kind::Info:
        show = e.text != e.shownText;
        e.shownText = e.text;
        put(n, e.text.empty() ? "-" : e.text);
        break;
    }
    if (show || fullDump)
      for (size_t k = first; k < out.size(); ++k) mark.push_back(out[k]);
  }
  bool grew = false;
  for (; columned_ < entries_.size(); ++columned_) {
    putColumns(entries_[columned_].kind, entries_[columned_].family, entries_[columned_].name, columns_);
    grew = true;
  }
  const bool kept = rows_.size() + 1 == index_;
  if (csv_) writeCsv(row, grew, kept);
  if (kept && index_ <= kMaxRows) rows_.push_back(row);

  std::vector<std::string> lines;
  if (compactOut) compactOut->clear();
  for (size_t f = 0; f < families_.size(); ++f) {
    char head[96];
    std::snprintf(head, sizeof head, "VPKENGINE_PROBE %s i=%llu t=%.1f dt=%.2f", families_[f].c_str(),
                  (unsigned long long)index_, t, dt);
    chunk(head, body[f], lines);
    if (compactOut) chunk(head, shown[f], *compactOut);
  }
  return lines;
}

std::vector<std::string> Registry::summary() {
  std::lock_guard lock(mutex_);
  ++summaries_;
  std::vector<std::string> lines;
  char head[128];
  if (index_ == 0) {
    std::snprintf(head, sizeof head, "VPKENGINE_PROBE SUMMARY empty seq=%llu samples=0", (unsigned long long)summaries_);
    lines.push_back(head);
    return lines;
  }
  std::vector<std::vector<std::string>> body(families_.size());
  for (const Entry& e : entries_) {
    std::vector<std::string>& out = body[e.fam];
    const auto put = [&](const std::string& key, const std::string& value) { out.push_back(' ' + key + '=' + value); };
    const std::string& n = e.name;
    switch (e.kind) {
      case Kind::Counter:
        put(n + "_total", std::to_string(e.lastCum));
        put(n + "_rate_peak", number(e.a.peak));
        put(n + "_rate_avg", number(e.a.mean()));
        break;
      case Kind::Gauge:
        put(n + "_peak", number(e.a.peak));
        put(n + "_avg", number(e.a.mean()));
        put(n + "_last", number(e.a.last));
        break;
      case Kind::Timer:
        put(n + "_ms_s_peak", number(e.a.peak));
        put(n + "_ms_s_avg", number(e.a.mean()));
        if (e.b.n) put(n + "_max_us_peak", number(e.b.peak));
        break;
      case Kind::Hist:
        put(n + "_p50_avg", number(e.a.mean()));
        put(n + "_p99_peak", number(e.b.peak));
        put(n + "_p99_avg", number(e.b.mean()));
        put(n + "_max_peak", number(e.c.peak));
        break;
      case Kind::Info: break;
    }
  }
  for (size_t f = 0; f < families_.size(); ++f) {
    std::snprintf(head, sizeof head, "VPKENGINE_PROBE SUMMARY %s seq=%llu samples=%llu elapsed=%.1f", families_[f].c_str(),
                  (unsigned long long)summaries_, (unsigned long long)index_, last_);
    chunk(head, body[f], lines);
  }
  return lines;
}

void Registry::reset() {
  std::lock_guard lock(mutex_);
  if (csv_) std::fclose(static_cast<FILE*>(csv_));
  csv_ = nullptr;
  csvPath_.clear();
  counters_.clear();
  gauges_.clear();
  timers_.clear();
  hists_.clear();
  entries_.clear();
  byName_.clear();
  families_.clear();
  columns_.clear();
  rows_.clear();
  columned_ = 0;
  index_ = summaries_ = 0;
  start_ = prev_ = last_ = 0;
  started_ = false;
  startNs_ = nowNs();
}

Registry& registry() {
  static Registry* r = new Registry;  // never destroyed: threads may still publish at exit
  return *r;
}

// ---- Memory around big allocations -----------------------------------------------------------------------------

uint64_t processFootprint() {
#if defined(__APPLE__)
  task_vm_info_data_t info;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) return 0;
  return info.phys_footprint;
#else
  return 0;
#endif
}

void allocCreated(const char* category, uint64_t bytes) {
  if (!enabled()) return;
  const std::string c = category;
  counter("ALLOC", c + "_n").add();
  counter("ALLOC", c + "_bytes").add(bytes);
}

AllocToken bigAllocBegin(const char* category, uint64_t bytes, uint64_t deviceAllocated) {
  if (!enabled()) return {};
  allocCreated(category, bytes);
  if (bytes < kBigAlloc) return {};
  return {category, bytes, processFootprint(), deviceAllocated, true};
}

void bigAllocEnd(const AllocToken& t, uint64_t deviceAllocated) {
  if (!t.active || !enabled()) return;
  const std::string c = t.category;
  const double mb = 1.0 / (1024.0 * 1024.0);
  counter("ALLOC", c + "_big_n").add();
  counter("ALLOC", c + "_big_bytes").add(t.bytes);
  gauge("ALLOC", c + "_big_footprint_mb").add((double(processFootprint()) - double(t.footprint)) * mb);
  gauge("ALLOC", c + "_big_device_mb").add((double(deviceAllocated) - double(t.device)) * mb);
}

}  // namespace vpgpu::probe
