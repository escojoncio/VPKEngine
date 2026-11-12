// Regression tests of the misc audit (pipeline cache, diagnostics, command ranges). One case per process: audit_misc_tests <case>.
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <thread>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#include "vpgpu/core/archive_harvest.h"
#include "vpgpu/core/background.h"
#include "vpgpu/core/backend.h"
#include "vpgpu/core/commands.h"
#include "vpgpu/core/device.h"
#include "vpgpu/core/diagnostics.h"
#include "vpgpu/core/frame_compiler.h"
#include "vpgpu/core/pipeline_cache.h"
#include "vpgpu/core/resources.h"

using namespace vpgpu;

static int failures = 0;
#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);            \
      ++failures;                                                            \
    }                                                                        \
  } while (0)

// Re-requesting a state that is already Queued at the same or a lower priority must not queue another job: the queue grew by
// one state copy per draw per frame while the workers were busy, and queuedAt (the age pending() reports) restarted at every
// request. A more urgent request still queues one job (that is how a priority is raised), once.
static void testRequeue() {
  NullBackend nb;
  std::mutex m;
  std::condition_variable cv;
  bool open = false, entered = false;
  nb.onCompile = [&] {
    std::unique_lock lock(m);
    entered = true;
    cv.notify_all();
    cv.wait(lock, [&] { return open; });
  };
  PipelineState a, q;
  a.vertexShader = 1, a.fragmentShader = 2, a.label = "a";
  q.vertexShader = 3, q.fragmentShader = 4, q.label = "q";
  {
    PipelineCache cache(nb, 1);
    cache.request(a, Priority::Visible);  // the single worker takes it and blocks inside compile
    {
      std::unique_lock lock(m);
      cv.wait(lock, [&] { return entered; });
    }
    CHECK(cache.request(q, Priority::Prefetch) == PipelineCache::Status::Queued);
    const uint32_t queued = compileBacklog().load();
    for (int i = 0; i < 50; ++i) {
      cache.request(q, Priority::Prefetch);
      cache.request(q, Priority::Prefetch);
    }
    CHECK(compileBacklog().load() == queued);  // same priority: nothing more is queued
    CHECK(cache.request(q, Priority::Visible) == PipelineCache::Status::Queued);
    CHECK(compileBacklog().load() == queued + 1);  // more urgent: one job that raises it
    for (int i = 0; i < 50; ++i) {
      cache.request(q, Priority::Visible);
      cache.request(q, Priority::Prefetch);
    }
    CHECK(compileBacklog().load() == queued + 1);  // not again at that priority or below
    CHECK(cache.request(q, Priority::Blocking) == PipelineCache::Status::Queued);
    CHECK(compileBacklog().load() == queued + 2);
    cache.boost(q.key());  // boost raises it to Urgent once
    cache.boost(q.key());
    CHECK(compileBacklog().load() == queued + 3);
    cache.request(q, Priority::Blocking);
    CHECK(compileBacklog().load() == queued + 3);
    {
      std::lock_guard lock(m);
      open = true;
    }
    cv.notify_all();
    cache.waitIdle();
    CHECK(cache.status(q.key()) == PipelineCache::Status::Ready);
    CHECK(cache.totals().compiled == 2);  // each state compiled once whatever the duplicates
  }
}

// A saved key list from another build, or damaged, must not hand the backend a format outside the enum.
static void testDecodeRange() {
  PipelineState s;
  std::vector<uint64_t> v;
  PipelineCache::encode(s, v);
  PipelineState out;
  CHECK(PipelineCache::decode(v.data(), v.size(), out));
  // the colour formats come right after the three shader ids
  v[3] = uint64_t(Format::Count);
  CHECK(!PipelineCache::decode(v.data(), v.size(), out));
  v[3] = 0xFFFFFFFFull;
  CHECK(!PipelineCache::decode(v.data(), v.size(), out));
}

// An attachment mip of 32 or more used to shift a 32-bit width by that many bits in the pass dump (undefined behaviour).
static void testDumpDeepMip() {
  ResourceTable res;
  TextureDesc d;
  d.format = Format::RGBA8Unorm;
  d.width = d.height = 64;
  d.usage = usage::sampled | usage::colorTarget;
  const TextureId t = res.createTexture(d);
  RenderPassDesc pd;
  pd.colors[0].texture = t;
  pd.colors[0].mip = 40;
  Frame f;
  f.beginRenderPass(pd);
  Draw dr;
  dr.pipeline = {1};
  dr.count = 3;
  f.draw(dr);
  f.endRenderPass();
  CompiledFrame cf;  // by hand: the frame compiler has the same shift (extent()) and is not what this case is about
  Encoder e;
  e.kind = Encoder::Kind::Render;
  e.pass = pd;
  e.commands = {0};
  e.passes = 1;
  cf.encoders.push_back(e);
  const auto lines = formatPassDump(1, f, cf, &res, nullptr);
  bool seen = false;
  for (const std::string& l : lines) seen = seen || l.find("att{c0=") != std::string::npos;
  CHECK(seen);
}

// The frame compiler sizes attachments by `width >> mip`: a client mip of 32 or more was a shift past the width (undefined).
static void testExtentDeepMip() {
  ResourceTable res;
  TextureDesc d;
  d.format = Format::RGBA8Unorm;
  d.width = d.height = 64;
  d.usage = usage::sampled | usage::colorTarget;
  const TextureId t = res.createTexture(d);
  RenderPassDesc pd;
  pd.colors[0].texture = t;
  pd.colors[0].mip = 40;
  pd.colors[0].store = StoreAction::Store;
  Frame f;
  f.beginRenderPass(pd);
  Draw dr;
  dr.pipeline = {1};
  dr.count = 3;
  f.draw(dr);
  f.endRenderPass();
  const CompiledFrame cf = compileFrame(f, &res);
  CHECK(cf.encoders.size() == 1);
  NullBackend be;
  Device dev(be, 0, [](const std::string&) {});
  const TextureId t2 = dev.createTexture(d);
  pd.colors[0].texture = t2;
  Frame f2;
  f2.beginRenderPass(pd);
  f2.draw(dr);
  f2.endRenderPass();
  dev.submit(f2);
  CHECK(dev.pipelines().totals().failed == 0);
}

// save() replaces the key list as a whole (a temporary file renamed over it): a stop in the middle of a write leaves the previous
// list instead of a truncated one. The replacement is a new file, so its inode changes, and no temporary is left behind.
static void testSaveAtomic() {
  NullBackend be;
  PipelineCache cache(be, 0);
  PipelineState a;
  a.vertexShader = 1, a.fragmentShader = 2;
  CHECK(cache.request(a, Priority::Visible) == PipelineCache::Status::Ready);
  const DiskCachePath path{std::filesystem::temp_directory_path() / ("vpgpu_audit_save_" + std::to_string(getpid()))};
  std::filesystem::remove_all(path.root);
  CHECK(cache.save(path));
  struct stat s1 {}, s2 {};
  CHECK(stat(path.keyList().c_str(), &s1) == 0);
  CHECK(cache.save(path));
  CHECK(stat(path.keyList().c_str(), &s2) == 0);
  CHECK(s1.st_ino != s2.st_ino);
  size_t files = 0;
  for (const auto& e : std::filesystem::directory_iterator(path.root)) files += e.is_regular_file();
  CHECK(files == 1);
  PipelineCache fresh(be, 0);
  CHECK(fresh.prefetch(path) == 1);
  CHECK(fresh.status(a.key()) == PipelineCache::Status::Ready);
  // a directory that cannot hold the file: false, and nothing half written
  const DiskCachePath bad{path.root / "pipelines.keys" / "x"};
  CHECK(!cache.save(bad));
  std::filesystem::remove_all(path.root);
}

// Pause hooks run while the gate's hook lock is held and removePauseHook waits for a running call: with hooks that obey the
// contract (no blocking, no add/remove) set(), add and remove from several threads must neither deadlock nor race, and a hook
// never runs once its removePauseHook returned. Meant to run under ThreadSanitizer.
static void testBackgroundHooks() {
  BackgroundGate g;
  g.setLog([](const std::string&) {});
  struct Hook {
    std::atomic<bool> removed{false};
    std::atomic<int> late{0}, runs{0};
  };
  std::atomic<bool> stop{false};
  std::thread toggler([&] {  // this thread is the gate's "main"
    for (int i = 0; i < 3000 && !stop; ++i) {
      g.set(true);
      g.set(false);
    }
  });
  std::vector<std::shared_ptr<Hook>> all;
  std::mutex allMutex;
  std::vector<std::thread> workers;
  for (int w = 0; w < 3; ++w)
    workers.emplace_back([&] {
      for (int i = 0; i < 400; ++i) {
        auto h = std::make_shared<Hook>();
        const uint64_t id = g.addPauseHook([h] {
          if (h->removed) ++h->late;
          ++h->runs;
        });
        std::this_thread::yield();
        g.removePauseHook(id);
        h->removed = true;
        std::lock_guard lock(allMutex);
        all.push_back(h);
      }
    });
  for (auto& t : workers) t.join();
  stop = true;
  toggler.join();
  int late = 0;
  for (auto& h : all) late += h->late;
  CHECK(late == 0);
  CHECK(all.size() == 1200);
}

// The index range of a draw must not wrap round to the start of the buffer, nor the rect of a copy to the origin.
static void testRangeSaturation() {
  Draw d;
  IndexBuffer ib;
  ib.buffer = BufferId{1, 1};
  ib.offset = UINT64_MAX - 15;
  d.index = ib;
  d.first = 100;
  d.count = 6;
  const std::vector<Access> acc = drawAccesses(d);
  const Access& idx = acc.back();
  CHECK(idx.range.begin >= UINT64_MAX - 15);  // not 184 after the wrap
  CHECK(idx.range.begin <= idx.range.end);

  Copy c;
  c.kind = CopyKind::TextureToTexture;
  c.srcTexture.texture = TextureId{1, 1};
  c.dstTexture.texture = TextureId{2, 1};
  c.srcTexture.x = c.dstTexture.x = UINT32_MAX - 3;
  c.srcTexture.width = c.dstTexture.width = 16;
  c.srcTexture.height = c.dstTexture.height = 1;
  const std::vector<Access> ca = copyAccesses(c, true);
  CHECK(ca.size() == 2 && ca[1].range.rect);
  CHECK(ca[1].range.x1 >= ca[1].range.x0);  // x + width saturates instead of wrapping to 12
}

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  const std::string t = argv[1];
  if (t == "requeue") testRequeue();
  else if (t == "decoderange") testDecodeRange();
  else if (t == "dumpdeepmip") testDumpDeepMip();
  else if (t == "rangesat") testRangeSaturation();
  else if (t == "extentmip") testExtentDeepMip();
  else if (t == "save") testSaveAtomic();
  else if (t == "background") testBackgroundHooks();
  else return 2;
  std::printf(failures ? "%s: %d FAILED\n" : "%s: ok\n", t.c_str(), failures);
  return failures != 0;
}
