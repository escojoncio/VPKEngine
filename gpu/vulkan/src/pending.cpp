// vpvk_dump_pending: "what is the game waiting for". The host (Klepton's stall detector) calls it when the guest sends no
// frames: it lists what the game polled and was told "not yet", and what in the driver could still make it so.
// Nothing here runs in the normal path but the relaxed counters inside the objects themselves (Polled, vpvk.h) and a
// registry that changes at create / destroy. The report is built only when asked.
#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>

#include "vpvk.h"

namespace vpvk {
namespace {
constexpr int64_t kWindowNs = 2'000'000'000;  // an object polled within this long ago counts as "being waited for"
constexpr size_t kMaxFences = 12, kMaxEvents = 6, kMaxSemaphores = 6, kMaxQueries = 6, kMaxDeferred = 4;

// Leaked on purpose: guest threads may outlive static destructors (like liveDeviceMutex).
template <class T>
struct Live {
  std::mutex m;
  std::unordered_set<T*> set;
};
template <class T>
Live<T>& live() {
  static auto* l = new Live<T>;
  return *l;
}

int64_t agoMs(int64_t now, int64_t ns) { return ns ? (now - ns) / 1'000'000 : -1; }
const char* resultName(int32_t r) {
  switch (r) {
    case VK_SUCCESS: return "SUCCESS";
    case VK_NOT_READY: return "NOT_READY";
    case VK_TIMEOUT: return "TIMEOUT";
    case VK_EVENT_SET: return "SET";
    case VK_EVENT_RESET: return "RESET";
    case VK_ERROR_DEVICE_LOST: return "DEVICE_LOST";
    default: return "OTHER";
  }
}

struct Row {
  int64_t key;  // the largest first
  std::string text;
};
// The `max` rows with the largest key as lines, then "<what>_more=N" if some did not fit.
void emit(std::vector<Row>& rows, size_t max, const char* what, std::vector<std::string>& out) {
  std::stable_sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.key > b.key; });
  for (size_t i = 0; i < rows.size() && i < max; ++i) out.push_back(std::move(rows[i].text));
  if (rows.size() > max) out.push_back("vk.pending " + std::string(what) + "_more=" + std::to_string(rows.size() - max));
}
std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
  char b[512];
  va_list ap;
  va_start(ap, f);
  std::vsnprintf(b, sizeof b, f, ap);
  va_end(ap);
  return b;
}
}  // namespace

template <class T>
void pendingTrack(T* p) {
  Live<T>& l = live<T>();
  std::lock_guard<std::mutex> g(l.m);
  l.set.insert(p);
}
template <class T>
void pendingUntrack(T* p) {
  Live<T>& l = live<T>();
  std::lock_guard<std::mutex> g(l.m);
  l.set.erase(p);
}
template void pendingTrack(Fence*);
template void pendingTrack(Event*);
template void pendingTrack(Semaphore*);
template void pendingTrack(QueryPoolObj*);
template void pendingUntrack(Fence*);
template void pendingUntrack(Event*);
template void pendingUntrack(Semaphore*);
template void pendingUntrack(QueryPoolObj*);

void pendingFrames(Device& d, uint64_t frames) {
  const uint64_t noted = d.frameNsNoted.load(std::memory_order_relaxed);
  if (frames <= noted) return;
  const int64_t now = pendingNowNs();
  for (uint64_t f = std::max<uint64_t>(noted, frames > 64 ? frames - 64 : 0); f < frames; ++f)
    d.frameNs[f % 64].store(now, std::memory_order_relaxed);
  d.frameNsNoted.store(frames, std::memory_order_release);
}

void pendingFenceWait(uint32_t n, const VkFence* fs, bool enter, VkBool32 waitAll, VkResult r) {
  for (uint32_t i = 0; i < n; ++i) {
    Fence* f = from<Fence>(fs[i]);
    if (enter) {
      f->waiting.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    f->waiting.fetch_sub(1, std::memory_order_relaxed);
    if (r != VK_SUCCESS || waitAll || n == 1) f->poll.note(r);  // "any" that succeeded says nothing about the others
  }
}

void pendingSemaphoreWait(const VkSemaphoreWaitInfo& info, bool enter) {
  const int64_t now = pendingNowNs();
  for (uint32_t i = 0; i < info.semaphoreCount; ++i) {
    Semaphore* s = from<Semaphore>(info.pSemaphores[i]);
    if (!enter) {
      s->waiting.fetch_sub(1, std::memory_order_relaxed);
      continue;
    }
    s->waiting.fetch_add(1, std::memory_order_relaxed);
    s->waitNs.store(now, std::memory_order_relaxed);
    uint64_t seen = s->waitMax.load(std::memory_order_relaxed);
    while (seen < info.pValues[i] && !s->waitMax.compare_exchange_weak(seen, info.pValues[i], std::memory_order_relaxed)) {
    }
  }
}

// Lines for the objects the host asked about and was told "not yet". The registries are locked while their objects are
// read (destroy removes the object first), the submit mutex only with try_lock: the thread that holds it may be the
// stuck one, and the report must not wait for it.
std::vector<std::string> pendingReport(Device& d) {
  std::vector<std::string> out;
  Platform& p = *d.platform;
  const int64_t now = pendingNowNs();
  auto gpuDone = [&](uint64_t framesBelow) { return p.waitFrames(framesBelow, 0) == Platform::WaitResult::Done; };

  // Frames handed to the backend and not finished: the first unfinished one by bisection (finished frames form a prefix).
  const uint64_t noted = d.frameNsNoted.load(std::memory_order_acquire);
  uint64_t lo = noted > 64 ? noted - 64 : 0, a = lo, b = noted;
  while (a < b) {
    const uint64_t m = a + (b - a) / 2;
    if (gpuDone(m + 1)) a = m + 1;
    else b = m;
  }
  const uint64_t inflight = noted - a;
  const bool more = inflight == 64 && lo > 0;  // the ring is full: there may be older ones
  std::unique_lock<std::mutex> submit(d.submitMutex, std::try_to_lock);
  out.push_back(fmt("vk.pending gpu frames=%" PRIu64 " inflight=%s%" PRIu64 " oldest_ms=%" PRId64 " lost=%d submit_mutex=%s deferred=%zu",
                    noted, more ? ">=" : "", inflight, inflight ? agoMs(now, d.frameNs[a % 64].load(std::memory_order_relaxed)) : 0,
                    p.lost() ? 1 : 0, submit.owns_lock() ? "free" : "BUSY", submit.owns_lock() ? d.deferred.size() : size_t(0)));
  if (d.gpu) {
    const vpgpu::PipelineCache::Pending pso = d.gpu->pipelines().pending();
    out.push_back(fmt("vk.pending pso queued=%u compiling=%u oldest_ms=%.0f oldest=%s translate_jobs=%zu", pso.queued, pso.compiling,
                      pso.oldestMs, pso.oldestLabel.empty() ? "-" : pso.oldestLabel.c_str(), d.translator ? d.translator->pending() : size_t(0)));
  }
  // Submits held back on a timeline wait nothing has signaled (wait-before-signal): their fence stays NOT_READY.
  if (submit.owns_lock()) {
    for (size_t i = 0; i < d.deferred.size() && i < kMaxDeferred; ++i) {
      const DeferredBatch& bt = d.deferred[i];
      std::string waits, signals;
      for (const auto& [s, v] : bt.waits) {
        std::lock_guard<std::mutex> l(s->m);
        waits += fmt(" %p>=%" PRIu64 "(has %" PRIu64 ")", static_cast<const void*>(s), v, s->completed);
      }
      for (const auto& [s, v] : bt.signals) signals += fmt(" %p=%" PRIu64, static_cast<const void*>(s), v);
      out.push_back(fmt("vk.pending deferred #%zu cbs=%zu fence=%p waits:%s signals:%s", i, bt.cbs.size(), static_cast<const void*>(bt.fence),
                        waits.c_str(), signals.c_str()));
    }
    if (d.deferred.size() > kMaxDeferred) out.push_back(fmt("vk.pending deferred_more=%zu", d.deferred.size() - kMaxDeferred));
  }
  if (submit.owns_lock()) submit.unlock();  // (try_to_lock: unlocking one that failed throws)

  std::vector<Row> rows;
  {
    Live<Fence>& l = live<Fence>();
    std::lock_guard<std::mutex> g(l.m);
    for (const Fence* f : l.set) {
      const uint32_t waiting = f->waiting.load(std::memory_order_relaxed);
      const int32_t res = f->poll.result.load(std::memory_order_relaxed);
      const int64_t polled = f->poll.ns.load(std::memory_order_relaxed);
      if (!waiting && !(f->poll.n.load(std::memory_order_relaxed) && res != VK_SUCCESS && now - polled <= kWindowNs)) continue;
      const bool signaled = f->signaled, deferred = f->deferred, submitted = f->submitted;
      const uint64_t frames = f->frames;
      const int64_t sub = f->submitNs.load(std::memory_order_relaxed);
      std::string st;  // the submit that will signal it
      if (!signaled) st = "none(unsubmitted_or_reset)";
      else if (deferred) st = "held(timeline_wait_unsignaled)";
      else if (submitted) st = fmt("gpu(frames=%" PRIu64 " gpu_done=%d)", frames, gpuDone(frames) ? 1 : 0);
      else st = "done";
      rows.push_back({signaled && sub ? now - sub : 0,
                      fmt("vk.pending fence=%p polls=%u last=%s last_poll_ms=%" PRId64 " waiting=%u submit=%s submit_age_ms=%" PRId64,
                          static_cast<const void*>(f), f->poll.n.load(std::memory_order_relaxed), resultName(res), agoMs(now, polled), waiting,
                          st.c_str(), signaled ? agoMs(now, sub) : -1)});
    }
  }
  emit(rows, kMaxFences, "fence", out);

  rows.clear();
  {
    Live<Event>& l = live<Event>();
    std::lock_guard<std::mutex> g(l.m);
    for (Event* e : l.set) {
      std::lock_guard<std::mutex> el(e->m);
      const int64_t polled = e->poll.ns.load(std::memory_order_relaxed);
      if (!e->poll.n.load(std::memory_order_relaxed) || e->poll.result.load(std::memory_order_relaxed) != VK_EVENT_RESET || now - polled > kWindowNs)
        continue;
      const std::string cmd = e->frame ? fmt("%" PRIu64 "(gpu_done=%d)", e->frame - 1, gpuDone(e->frame - 1) ? 1 : 0) : "none";
      rows.push_back({now - polled, fmt("vk.pending event=%p polls=%u last_poll_ms=%" PRId64 " set=%d pending_frames=%s host_set=%u host_reset=%u "
                                        "cmd_stamps=%u last_stamp_ms=%" PRId64,
                                        static_cast<const void*>(e), e->poll.n.load(std::memory_order_relaxed), agoMs(now, polled), e->set ? 1 : 0,
                                        cmd.c_str(), e->hostSet, e->hostReset, e->cmdStamps, agoMs(now, e->stampNs))});
    }
  }
  emit(rows, kMaxEvents, "event", out);

  rows.clear();
  {
    Live<Semaphore>& l = live<Semaphore>();
    std::lock_guard<std::mutex> g(l.m);
    for (Semaphore* s : l.set) {
      const uint32_t waiting = s->waiting.load(std::memory_order_relaxed);
      const int64_t polled = s->poll.ns.load(std::memory_order_relaxed), waited = s->waitNs.load(std::memory_order_relaxed);
      if (!waiting && now - polled > kWindowNs && now - waited > kWindowNs) continue;
      uint64_t completed, pendingMax = 0;
      size_t pendingN;
      {
        std::lock_guard<std::mutex> sl(s->m);
        completed = s->completed, pendingN = s->pending.size();
        for (const auto& [v, f] : s->pending) pendingMax = std::max(pendingMax, v);
      }
      const uint64_t waitMax = s->waitMax.load(std::memory_order_relaxed);
      rows.push_back({waitMax > completed ? (int64_t(1) << 40) | (now - std::min(polled, waited)) : 0,
                      fmt("vk.pending timeline=%p completed=%" PRIu64 " wait_max=%" PRIu64 " in_flight_max=%" PRIu64 "(%zu) waiting=%u polls=%u "
                          "last_poll_ms=%" PRId64 " last_wait_ms=%" PRId64,
                          static_cast<const void*>(s), completed, waitMax, pendingMax, pendingN, waiting, s->poll.n.load(std::memory_order_relaxed),
                          agoMs(now, polled), agoMs(now, waited))});
    }
  }
  emit(rows, kMaxSemaphores, "timeline", out);

  rows.clear();
  {
    Live<QueryPoolObj>& l = live<QueryPoolObj>();
    std::lock_guard<std::mutex> g(l.m);
    for (QueryPoolObj* q : l.set) {
      const int64_t polled = q->poll.ns.load(std::memory_order_relaxed);
      if (!q->poll.n.load(std::memory_order_relaxed) || q->poll.result.load(std::memory_order_relaxed) == VK_SUCCESS || now - polled > kWindowNs) continue;
      const uint32_t miss = q->lastNotReady.load(std::memory_order_relaxed);
      uint64_t f = 0;
      {
        std::lock_guard<std::mutex> ql(q->m);
        f = miss < q->frame.size() ? q->frame[miss] : 0;
      }
      const std::string st = f ? fmt("ended_frames=%" PRIu64 " gpu_done=%d", f - 1, gpuDone(f - 1) ? 1 : 0) : "never_ended";
      rows.push_back({now - polled, fmt("vk.pending query_pool=%p type=%d count=%u polls=%u last_poll_ms=%" PRId64 " missing=%u %s",
                                        static_cast<const void*>(q), int(q->type), q->count, q->poll.n.load(std::memory_order_relaxed),
                                        agoMs(now, polled), miss, st.c_str())});
    }
  }
  emit(rows, kMaxQueries, "query", out);
  return out;
}

}  // namespace vpvk
