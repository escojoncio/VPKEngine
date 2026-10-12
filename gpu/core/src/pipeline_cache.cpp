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
  {
    std::lock_guard lock(mutex_);
    Entry& e = entries_[key];
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
  std::lock_guard lock(mutex_);
  auto it = entries_.find(key.value);
  return it == entries_.end() ? Status::Missing : it->second.status;
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
  done_.notify_all();
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
  std::unique_lock lock(mutex_);
  if (workers_.empty()) return !pending();
  return done_.wait_for(lock, budget, [&] { return stop_ || !pending(); });
}

void PipelineCache::workerLoop() {
  std::unique_lock lock(mutex_);
  for (;;) {
    wake_.wait(lock, [&] { return stop_ || !queue_.empty(); });
    if (stop_) return;
    Job job = queue_.top();
    queue_.pop();
    Status& st = entries_[job.state.key().value].status;
    if (st != Status::Queued) continue;  // a duplicate job already claimed by another worker
    st = Status::Compiling;
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
