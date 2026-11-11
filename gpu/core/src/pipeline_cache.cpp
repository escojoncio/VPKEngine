#include "vpgpu/core/pipeline_cache.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <utility>
#if defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#endif

#include "vpgpu/core/archive_harvest.h"
#include "vpgpu/core/backend.h"

namespace vpgpu {

namespace {

// Every field of PipelineState except the label, in a fixed order. Both the key hash and the
// disk format walk this list, so they cannot drift apart.
template <class State, class F>
void forEachField(State& s, F&& f) {
  f(s.vertexShader), f(s.fragmentShader), f(s.computeShader);
  for (auto& c : s.colorFormats) f(c);
  f(s.depthFormat), f(s.samples), f(s.primitive);
  for (auto& b : s.blend) {
    f(b.enabled), f(b.srcColor), f(b.dstColor), f(b.srcAlpha), f(b.dstAlpha), f(b.colorOp), f(b.alphaOp);
    f(b.writeMask);
  }
  f(s.depthTest), f(s.depthWrite), f(s.depthCompare), f(s.cull), f(s.alphaToCoverage);
  f(s.stencilTest);
  for (auto* st : {&s.stencilFront, &s.stencilBack})
    f(st->fail), f(st->depthFail), f(st->pass), f(st->compare), f(st->readMask), f(st->writeMask);
  f(s.fetch.mask);
  for (auto& t : s.fetch.textureSlot) f(t);
}

template <class T>
uint64_t asInt(T v) { return static_cast<uint64_t>(v); }

using Clock = std::chrono::steady_clock;

}  // namespace

bool fetchInputs(const Draw& d, const RenderPassDesc& writer, FetchInputs& out) {
  out = {};
  for (const Access& a : d.bindings) {
    if (!a.pixelLocal) continue;
    const int i = fetchIndex(writer, a.range);
    if (i < 0) return false;
    // One [[color(i)]] input per attachment: two texture slots reading it would need two names.
    if ((out.mask >> i & 1) && out.textureSlot[i] != a.slot) return false;
    out.mask |= uint8_t(1u << i);
    out.textureSlot[i] = a.slot;
  }
  return true;
}

PipelineKey PipelineState::key() const {
  uint64_t h = 1469598103934665603ull;  // FNV-1a over 64-bit field values
  forEachField(*this, [&](const auto& v) {
    uint64_t x = asInt(v);
    for (int i = 0; i < 8; ++i, x >>= 8) h = (h ^ (x & 0xFF)) * 1099511628211ull;
  });
  return {h ? h : 1};  // 0 means "no pipeline"
}

std::string describePipelineState(const PipelineState& s, const std::string& more) {
  static const char* const kPrimitive[] = {"points", "lines", "line_strip", "triangles", "triangle_strip"};
  char b[160];
  std::string label = s.label;
  for (char& c : label)
    if (c == ' ') c = '_';
  std::string o;
  std::snprintf(b, sizeof b, "key=%016llx ", (unsigned long long)s.key().value);
  o += b;
  o += "label=" + label + " samples=" + std::to_string(int(s.samples)) + " colorFormats=";
  bool any = false;
  for (const Format f : s.colorFormats)
    if (f != Format::Undefined) o += (any ? "/" : "") + std::string(formatInfo(f).name), any = true;
  if (!any) o += "none";
  o += std::string(" depthFormat=") + formatInfo(s.depthFormat).name;
  o += std::string(" primitive=") + kPrimitive[std::min<size_t>(size_t(s.primitive), 4)];
  std::snprintf(b, sizeof b, " vs=%016llx fs=%016llx extra{cs=%016llx", (unsigned long long)s.vertexShader,
                (unsigned long long)s.fragmentShader, (unsigned long long)s.computeShader);
  o += b;
  for (size_t i = 0; i < s.blend.size(); ++i) {
    const BlendState& x = s.blend[i];
    if (s.colorFormats[i] == Format::Undefined && !x.enabled) continue;
    std::snprintf(b, sizeof b, " blend%zu=%d:%d.%d.%d/%d.%d.%d:m%x", i, int(x.enabled), int(x.srcColor), int(x.dstColor),
                  int(x.colorOp), int(x.srcAlpha), int(x.dstAlpha), int(x.alphaOp), unsigned(x.writeMask));
    o += b;
  }
  std::snprintf(b, sizeof b, " depth=%d%d:%d cull=%d a2c=%d fetch=%x", int(s.depthTest), int(s.depthWrite), int(s.depthCompare),
                int(s.cull), int(s.alphaToCoverage), unsigned(s.fetch.mask));
  o += b;
  if (s.stencilTest) {
    for (const StencilFace* f : {&s.stencilFront, &s.stencilBack}) {
      std::snprintf(b, sizeof b, " stencil%s=%d.%d.%d:%d:%x.%x", f == &s.stencilFront ? "F" : "B", int(f->fail), int(f->depthFail),
                    int(f->pass), int(f->compare), unsigned(f->readMask), unsigned(f->writeMask));
      o += b;
    }
  }
  if (!more.empty()) o += " " + more;
  return o + "}";
}

PipelineCache::PipelineCache(Backend& backend, unsigned workers) : backend_(backend) {
  for (unsigned i = 0; i < workers; ++i) workers_.emplace_back([this] { workerLoop(); });
}

PipelineCache::~PipelineCache() {
  {
    std::lock_guard lock(mutex_);
    stop_ = true;
  }
  wake_.notify_all();
  for (std::thread& t : workers_) t.join();
  compileBacklog().fetch_sub(backlogReported_, std::memory_order_relaxed);
}

// Adds the change of queued + running jobs since the last call to the global backlog the archive harvest reads.
void PipelineCache::syncBacklogLocked() {
  const uint32_t n = uint32_t(queue_.size()) + busy_;
  compileBacklog().fetch_add(n - backlogReported_, std::memory_order_relaxed);  // unsigned wrap-around adds the delta
  backlogReported_ = n;
}

PipelineCache::Status PipelineCache::request(const PipelineState& state, Priority priority) {
  return request(state, state.key(), priority);
}

PipelineCache::Status PipelineCache::request(const PipelineState& state, PipelineKey stateKey, Priority priority) {
  const uint64_t key = stateKey.value;
  {
    std::lock_guard lock(mutex_);
    Entry& e = entries_[key];
    e.orphan = false;  // somebody wants it again
    if (e.status == Status::Ready) {
      ++frame_.hits, ++total_.hits;
      return Status::Ready;
    }
    if (e.status == Status::Failed && e.speculativeFailure && priority != Priority::Prefetch) {
      e.speculativeFailure = false;  // retry once for the real request
      e.status = Status::Missing;
    }
    if (e.status == Status::Failed || e.status == Status::Compiling) return e.status;
    ++frame_.misses, ++total_.misses;
    if (!workers_.empty()) {
      // Re-queuing an already queued state is how its priority gets raised; the stale job is
      // skipped by the worker once the state is no longer Queued.
      e.state = state;
      e.status = Status::Queued;
      e.queuedAt = Clock::now();
      queue_.push({priority, nextOrder_++, state});
      syncBacklogLocked();
      wake_.notify_one();
      return Status::Queued;
    }
  }
  compile(state, priority);
  return status(stateKey);
}

PipelineCache::Status PipelineCache::status(PipelineKey key) const {
  std::lock_guard lock(mutex_);
  auto it = entries_.find(key.value);
  return it == entries_.end() ? Status::Missing : it->second.status;
}

bool PipelineCache::stateOf(PipelineKey key, PipelineState& out) const {
  std::lock_guard lock(mutex_);
  auto it = entries_.find(key.value);
  if (it == entries_.end()) return false;
  out = it->second.state;
  return true;
}

void PipelineCache::retain(PipelineKey key) {
  std::lock_guard lock(mutex_);
  ++owners_[key.value];
  auto it = entries_.find(key.value);
  if (it != entries_.end()) it->second.orphan = false;  // a new owner takes over a queued/compiling pipeline
}

void PipelineCache::release(PipelineKey key) {
  std::lock_guard lock(mutex_);
  auto o = owners_.find(key.value);
  if (o == owners_.end() || --o->second) return;
  owners_.erase(o);
  auto it = entries_.find(key.value);
  if (it == entries_.end()) return;
  if (it->second.status == Status::Queued) {  // nothing built yet: forget it; its stale job is skipped by the workers
    entries_.erase(it);
    done_.notify_all();  // waitReady() of this key must not sleep out its whole budget
    return;
  }
  if (it->second.status == Status::Compiling) {  // no owner left: compile() drops the result when it finishes
    it->second.orphan = true;
    return;
  }
  if (it->second.status != Status::Ready && it->second.status != Status::Failed) return;
  entries_.erase(it);
  backend_.destroyPipeline(key);  // under the lock: a request() of the key waits, then recompiles
}

void PipelineCache::compile(const PipelineState& state, Priority priority) {
  std::string error;
  const auto start = Clock::now();
  const bool ok = backend_.compilePipeline(state, error);
  const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  std::lock_guard lock(mutex_);
  Entry& e = entries_[state.key().value];
  e.state = state;
  e.status = ok ? Status::Ready : Status::Failed;
  e.speculativeFailure = !ok && priority == Priority::Prefetch;
  for (PipelineStats* s : {&frame_, &total_}) {
    ++(ok ? s->compiled : s->failed);
    s->totalMs += ms;
    s->maxMs = std::max(s->maxMs, ms);
  }
  if (e.orphan) {  // released while compiling and nobody retained it since: discard instead of leaking
    entries_.erase(state.key().value);
    backend_.destroyPipeline(state.key());  // under the lock, like release()
  }
  done_.notify_all();
}

bool PipelineCache::idle() const {
  std::lock_guard lock(mutex_);
  return queue_.empty() && !busy_;
}

PipelineCache::Pending PipelineCache::pending() const {
  Pending p;
  const auto now = Clock::now();
  std::lock_guard lock(mutex_);
  for (const auto& [key, e] : entries_) {
    if (e.status != Status::Queued && e.status != Status::Compiling) continue;
    ++(e.status == Status::Queued ? p.queued : p.compiling);
    const double ms = std::chrono::duration<double, std::milli>(now - e.queuedAt).count();
    if (ms >= p.oldestMs) p.oldestMs = ms, p.oldestLabel = e.state.label;
  }
  return p;
}

void PipelineCache::boost(PipelineKey key) {
  std::lock_guard lock(mutex_);
  auto it = entries_.find(key.value);
  if (it == entries_.end() || it->second.status != Status::Queued || it->second.boosted || workers_.empty()) return;
  it->second.boosted = true;
  queue_.push({Priority::Urgent, nextOrder_++, it->second.state});
  syncBacklogLocked();
  wake_.notify_one();
}

bool PipelineCache::waitReady(const std::vector<PipelineKey>& keys, std::chrono::milliseconds budget) {
  auto pending = [&] {
    for (PipelineKey k : keys) {
      auto it = entries_.find(k.value);
      if (it != entries_.end() && (it->second.status == Status::Queued || it->second.status == Status::Compiling))
        return true;
    }
    return false;
  };
  // One absolute deadline for the whole call: every wake-up (notify_all of an unrelated pipeline, a
  // spurious one) is measured against it, so the total never exceeds `budget` plus the time to get
  // the mutex back.
  const auto deadline = Clock::now() + budget;
  std::unique_lock lock(mutex_);
  if (workers_.empty()) return !pending();
  if (done_.wait_until(lock, deadline, [&] { return stop_ || !pending(); })) return true;
  return false;
}

void PipelineCache::workerLoop() {
#if defined(__APPLE__)
  // These threads compile what a draw is waiting for (Urgent / Blocking) as well as prefetch: above the speculative
  // library compiles of the Metal backend (UTILITY), so under load the scheduler favours the former.
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
  std::unique_lock lock(mutex_);
  for (;;) {
    wake_.wait(lock, [&] { return stop_ || !queue_.empty(); });
    if (stop_) return;
    Job job = queue_.top();
    queue_.pop();
    auto entry = entries_.find(job.state.key().value);  // absent: released while queued
    if (entry == entries_.end() || entry->second.status != Status::Queued) {  // a duplicate job already claimed by another worker
      syncBacklogLocked();
      if (queue_.empty() && !busy_) idle_.notify_all();  // it may have been the last job (waitIdle)
      continue;
    }
    entry->second.status = Status::Compiling;
    entry->second.boosted = false;
    ++busy_;
    syncBacklogLocked();
    lock.unlock();
    compile(job.state, job.priority);
    lock.lock();
    --busy_;
    syncBacklogLocked();
    if (queue_.empty() && !busy_) idle_.notify_all();
  }
}

void PipelineCache::waitIdle() {
  std::unique_lock lock(mutex_);
  idle_.wait(lock, [&] { return workers_.empty() || (queue_.empty() && !busy_); });
}

PipelineStats PipelineCache::takeFrameStats() {
  std::lock_guard lock(mutex_);
  return std::exchange(frame_, {});
}

PipelineStats PipelineCache::totals() const {
  std::lock_guard lock(mutex_);
  return total_;
}

bool PipelineCache::save(const DiskCachePath& path) const {
  std::error_code ec;
  std::filesystem::create_directories(path.root, ec);
  std::ofstream out(path.keyList(), std::ios::trunc);
  if (!out) return false;
  std::lock_guard lock(mutex_);
  for (const auto& [key, e] : entries_) {
    if (e.status != Status::Ready) continue;
    forEachField(e.state, [&](const auto& v) { out << asInt(v) << ' '; });
    out << '\n';
  }
  return static_cast<bool>(out);
}

size_t PipelineCache::prefetch(const DiskCachePath& path) {
  std::ifstream in(path.keyList());
  size_t queued = 0;
  for (std::string line; std::getline(in, line);) {
    std::istringstream fields(line);
    PipelineState s;
    bool ok = true;
    forEachField(s, [&](auto& v) {
      uint64_t x = 0;
      ok = ok && static_cast<bool>(fields >> x);
      v = static_cast<std::remove_reference_t<decltype(v)>>(x);
    });
    if (ok && request(s, Priority::Prefetch) != Status::Failed) ++queued;
  }
  return queued;
}

void PipelineCache::encode(const PipelineState& s, std::vector<uint64_t>& out) {
  forEachField(s, [&](const auto& v) { out.push_back(asInt(v)); });
}

size_t PipelineCache::encodedSize() {
  static const size_t n = [] {
    std::vector<uint64_t> v;
    encode(PipelineState{}, v);
    return v.size();
  }();
  return n;
}

bool PipelineCache::decode(const uint64_t* in, size_t count, PipelineState& s) {
  if (count != encodedSize()) return false;
  size_t i = 0;
  forEachField(s, [&](auto& v) { v = static_cast<std::remove_reference_t<decltype(v)>>(in[i++]); });
  return true;
}

}  // namespace vpgpu
