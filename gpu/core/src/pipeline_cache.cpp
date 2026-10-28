#include "vpgpu/core/pipeline_cache.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <utility>

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

constexpr uint64_t kTomb = ~uint64_t(0);
// Keys 0 (empty slot) and ~0 (tombstone) are stored as other values; the same map on every use.
uint64_t slotKey(uint64_t key) { return key == 0 || key == kTomb ? key ^ 0x5bd1e995ull : key; }

}  // namespace

bool PipelineCache::readyFast(uint64_t key) const {
  const uint64_t k = slotKey(key);
  for (size_t i = 0, at = (k * 0x9E3779B97F4A7C15ull >> 40) & (kReadySlots - 1); i < 16; ++i, at = (at + 1) & (kReadySlots - 1)) {
    const uint64_t v = ready_[at].load(std::memory_order_acquire);
    if (v == k) return true;
    if (v == 0) return false;
  }
  return false;
}

void PipelineCache::markReady(uint64_t key) {  // under mutex_
  const uint64_t k = slotKey(key);
  size_t free = kReadySlots;
  for (size_t i = 0, at = (k * 0x9E3779B97F4A7C15ull >> 40) & (kReadySlots - 1); i < 16; ++i, at = (at + 1) & (kReadySlots - 1)) {
    const uint64_t v = ready_[at].load(std::memory_order_relaxed);
    if (v == k) return;
    if (v == kTomb && free == kReadySlots) free = at;
    if (v == 0) {
      if (free == kReadySlots) free = at;
      break;
    }
  }
  if (free != kReadySlots) ready_[free].store(k, std::memory_order_release);  // (full run of 16: the locked path serves it)
}

void PipelineCache::unmarkReady(uint64_t key) {  // under mutex_
  const uint64_t k = slotKey(key);
  for (size_t i = 0, at = (k * 0x9E3779B97F4A7C15ull >> 40) & (kReadySlots - 1); i < 16; ++i, at = (at + 1) & (kReadySlots - 1)) {
    const uint64_t v = ready_[at].load(std::memory_order_relaxed);
    if (v == k) return ready_[at].store(kTomb, std::memory_order_release);
    if (v == 0) return;
  }
}

bool PipelineCache::ready(PipelineKey key) {
  if (!readyFast(key.value)) return false;
  hitsFrame_.fetch_add(1, std::memory_order_relaxed);
  hitsTotal_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

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
}

PipelineCache::Status PipelineCache::request(const PipelineState& state, Priority priority) {
  return request(state, state.key(), priority);
}

PipelineCache::Status PipelineCache::request(const PipelineState& state, PipelineKey stateKey, Priority priority) {
  const uint64_t key = stateKey.value;
  if (ready(stateKey)) return Status::Ready;
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
      queue_.push({priority, nextOrder_++, state});
      wake_.notify_one();
      return Status::Queued;
    }
  }
  compile(state, priority);
  return status(stateKey);
}

PipelineCache::Status PipelineCache::status(PipelineKey key) const {
  if (readyFast(key.value)) return Status::Ready;
  std::lock_guard lock(mutex_);
  auto it = entries_.find(key.value);
  return it == entries_.end() ? Status::Missing : it->second.status;
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
  unmarkReady(key.value);
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
  if (ok && !e.orphan) markReady(state.key().value);  // (an orphan is discarded below: never visible to ready())
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

void PipelineCache::boost(PipelineKey key) {
  std::lock_guard lock(mutex_);
  auto it = entries_.find(key.value);
  if (it == entries_.end() || it->second.status != Status::Queued || it->second.boosted || workers_.empty()) return;
  it->second.boosted = true;
  queue_.push({Priority::Urgent, nextOrder_++, it->second.state});
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
  std::unique_lock lock(mutex_);
  for (;;) {
    wake_.wait(lock, [&] { return stop_ || !queue_.empty(); });
    if (stop_) return;
    Job job = queue_.top();
    queue_.pop();
    auto entry = entries_.find(job.state.key().value);  // absent: released while queued
    if (entry == entries_.end() || entry->second.status != Status::Queued) {  // a duplicate job already claimed by another worker
      if (queue_.empty() && !busy_) idle_.notify_all();  // it may have been the last job (waitIdle)
      continue;
    }
    entry->second.status = Status::Compiling;
    entry->second.boosted = false;
    ++busy_;
    lock.unlock();
    compile(job.state, job.priority);
    lock.lock();
    --busy_;
    if (queue_.empty() && !busy_) idle_.notify_all();
  }
}

void PipelineCache::waitIdle() {
  std::unique_lock lock(mutex_);
  idle_.wait(lock, [&] { return workers_.empty() || (queue_.empty() && !busy_); });
}

PipelineStats PipelineCache::takeFrameStats() {
  std::lock_guard lock(mutex_);
  PipelineStats s = std::exchange(frame_, {});
  s.hits += hitsFrame_.exchange(0, std::memory_order_relaxed);
  return s;
}

PipelineStats PipelineCache::totals() const {
  std::lock_guard lock(mutex_);
  PipelineStats s = total_;
  s.hits += hitsTotal_.load(std::memory_order_relaxed);
  return s;
}

PipelineCensus PipelineCache::census() const {
  std::lock_guard lock(mutex_);
  PipelineCensus c;
  c.entries = uint32_t(entries_.size());
  for (const auto& [key, e] : entries_) {
    switch (e.status) {
      case Status::Ready: ++c.ready; break;
      case Status::Queued: ++c.queued; break;
      case Status::Compiling: ++c.compiling; break;
      case Status::Failed: ++c.failed; break;
      case Status::Missing: break;
    }
    c.orphan += e.orphan;
  }
  c.queueLength = uint32_t(queue_.size());
  c.busyWorkers = busy_;
  c.workers = uint32_t(workers_.size());
  c.owners = uint32_t(owners_.size());
  return c;
}

bool PipelineCache::save(const DiskCachePath& path) const {
  std::error_code ec;
  std::filesystem::create_directories(path.root, ec);
  // Snapshot under the lock, written without it: file IO must not stall the threads that ask for pipelines.
  std::vector<PipelineState> ready;
  {
    std::lock_guard lock(mutex_);
    ready.reserve(entries_.size());
    for (const auto& [key, e] : entries_)
      if (e.status == Status::Ready) ready.push_back(e.state);
  }
  // Written beside and renamed over: a crash mid-write leaves the previous list.
  const std::filesystem::path tmp = path.keyList().string() + ".tmp";
  bool written = false;
  {
    std::ofstream out(tmp, std::ios::trunc);
    if (out) {
      for (const PipelineState& st : ready) {
        forEachField(st, [&](const auto& v) { out << asInt(v) << ' '; });
        out << '\n';
      }
      written = static_cast<bool>(out);
    }
  }
  if (written) std::filesystem::rename(tmp, path.keyList(), ec);
  if (!written || ec) {  // leave no .tmp behind (the previous list stays)
    std::error_code ignored;
    std::filesystem::remove(tmp, ignored);
    return false;
  }
  return true;
}

std::vector<PipelineState> PipelineCache::readKeyList(const DiskCachePath& path) {
  std::ifstream in(path.keyList());
  std::vector<PipelineState> states;
  for (std::string line; std::getline(in, line);) {
    std::istringstream fields(line);
    PipelineState s;
    bool ok = true;
    forEachField(s, [&](auto& v) {
      uint64_t x = 0;
      ok = ok && static_cast<bool>(fields >> x);
      v = static_cast<std::remove_reference_t<decltype(v)>>(x);
    });
    if (ok) states.push_back(std::move(s));
  }
  return states;
}

size_t PipelineCache::prefetch(const DiskCachePath& path) {
  size_t queued = 0;
  for (const PipelineState& s : readKeyList(path))
    if (request(s, Priority::Prefetch) != Status::Failed) ++queued;
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
