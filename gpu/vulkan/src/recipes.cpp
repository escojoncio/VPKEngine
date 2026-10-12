#include "recipes.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <thread>
#include <algorithm>
#include <memory>

#include "vpvk.h"
#include "vpvk/config.h"

namespace vpvk {
namespace {

constexpr char kMagic[4] = {'V', 'Q', 'R', 'C'};
constexpr uint32_t kRecipeVersion = 1;  // bump when the translator or the state encoding changes
constexpr uint8_t kTagShader = 'S', kTagPipeline = 'P';

struct Writer {
  std::vector<uint8_t> b;
  template <class T>
  void pod(T v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    b.insert(b.end(), p, p + sizeof v);
  }
  void str(const std::string& s) {
    pod(uint32_t(s.size()));
    b.insert(b.end(), s.begin(), s.end());
  }
};

struct Reader {
  const uint8_t* p;
  const uint8_t* end;
  bool ok = true;
  template <class T>
  T pod() {
    T v{};
    if (end - p < ptrdiff_t(sizeof v)) return ok = false, v;
    std::memcpy(&v, p, sizeof v);
    p += sizeof v;
    return v;
  }
  std::string str() {
    const uint32_t n = pod<uint32_t>();
    if (!ok || end - p < ptrdiff_t(n)) return ok = false, std::string();
    std::string s(reinterpret_cast<const char*>(p), n);
    p += n;
    return s;
  }
};

uint32_t headerFields() { return uint32_t(vpgpu::PipelineCache::encodedSize()); }

}  // namespace

std::string RecipeStore::defaultDir() {
  const char* d = std::getenv("VP_PIPELINE_CACHE_DIR");
  return d && *d ? d : "";
}

RecipeStore::RecipeStore(std::string dir) {
  if (dir.empty()) return;
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  const std::string path = dir + "/recipes.bin";
  load(path);
  const bool fresh = shaders_.empty() && pipelines_.empty();
  file_ = std::fopen(path.c_str(), fresh ? "wb" : "ab");
  if (!file_) {
    logf("gpu.pso.recipe what=open_failed path=%s", path.c_str());
    return;
  }
  if (fresh) {
    std::fwrite(kMagic, 1, 4, file_);
    const uint32_t hdr[2] = {kRecipeVersion, headerFields()};
    std::fwrite(hdr, sizeof hdr, 1, file_);
    std::fflush(file_);
  }
  logf("gpu.pso.recipe loaded shaders=%zu pipelines=%zu", shaders_.size(), pipelines_.size());
}

RecipeStore::~RecipeStore() {
  if (!file_) return;
  logf("gpu.pso.recipe stored=%zu total=%zu", storedThisRun_, pipelines_.size());
  std::fclose(file_);
}

void RecipeStore::load(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return;
  std::vector<uint8_t> data;
  uint8_t buf[65536];
  for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) data.insert(data.end(), buf, buf + n);
  std::fclose(f);
  Reader r{data.data(), data.data() + data.size()};
  char magic[4] = {};
  for (char& c : magic) c = char(r.pod<uint8_t>());
  const uint32_t ver = r.pod<uint32_t>(), fields = r.pod<uint32_t>();
  if (!r.ok || std::memcmp(magic, kMagic, 4) || ver != kRecipeVersion || fields != headerFields()) {
    logf("gpu.pso.recipe discarded=stale version=%u fields=%u", ver, fields);
    return;  // the constructor rewrites the file from scratch
  }
  size_t bad = 0;
  size_t goodEnd = size_t(r.p - data.data());
  while (r.ok && r.p < r.end) {
    const uint8_t tag = r.pod<uint8_t>();
    const uint32_t len = r.pod<uint32_t>();
    if (!r.ok || r.end - r.p < ptrdiff_t(len)) break;  // truncated tail (crash while appending)
    Reader rec{r.p, r.p + len};
    r.p += len;
    goodEnd = size_t(r.p - data.data());
    if (tag == kTagShader) {
      const uint64_t id = rec.pod<uint64_t>();
      ShaderBlob b;
      b.entry = rec.str();
      b.source = rec.str();
      for (auto& t : b.threadsPerGroup) t = rec.pod<uint32_t>();
      b.colorOutputs = rec.pod<uint8_t>();
      const uint32_t n = rec.pod<uint32_t>();
      for (uint32_t i = 0; rec.ok && i < n && i < 4096; ++i) {
        ShaderBlob::Arg a;
        a.kind = static_cast<vpgpu::BindKind>(rec.pod<uint8_t>());
        a.slot = rec.pod<uint16_t>();
        a.offset = rec.pod<uint32_t>();
        b.arguments.push_back(a);
      }
      if (rec.ok) shaders_[id] = std::move(b);
      else ++bad;
    } else if (tag == kTagPipeline) {
      std::vector<uint64_t> v(headerFields());
      for (auto& x : v) x = rec.pod<uint64_t>();
      vpgpu::PipelineState s;
      if (rec.ok && vpgpu::PipelineCache::decode(v.data(), v.size(), s)) {
        s.label = "vk.prewarm";
        pipelines_[s.key().value] = s;
      } else {
        ++bad;
      }
    } else {
      ++bad;
    }
  }
  if (bad) logf("gpu.pso.recipe bad_records=%zu", bad);
  if (goodEnd < data.size()) {  // drop a torn tail so later appends stay framed
    std::error_code ec;
    std::filesystem::resize_file(path, goodEnd, ec);
    logf("gpu.pso.recipe truncated_tail=%zu", data.size() - goodEnd);
  }
}

void RecipeStore::write(uint8_t tag, const std::vector<uint8_t>& payload) {
  const uint32_t len = uint32_t(payload.size());
  std::fwrite(&tag, 1, 1, file_);
  std::fwrite(&len, sizeof len, 1, file_);
  std::fwrite(payload.data(), 1, payload.size(), file_);
  std::fflush(file_);
}

void RecipeStore::addShader(uint64_t id, const ShaderBlob& b) {
  std::lock_guard lock(mutex_);
  if (!file_ || shaders_.count(id)) return;
  Writer w;
  w.pod(id);
  w.str(b.entry);
  w.str(b.source);
  for (auto t : b.threadsPerGroup) w.pod(uint32_t(t));
  w.pod(uint8_t(b.colorOutputs));
  w.pod(uint32_t(b.arguments.size()));
  for (const auto& a : b.arguments) w.pod(uint8_t(a.kind)), w.pod(uint16_t(a.slot)), w.pod(uint32_t(a.offset));
  write(kTagShader, w.b);
  shaders_[id] = b;
}

void RecipeStore::addPipeline(const vpgpu::PipelineState& s) {
  const uint64_t key = s.key().value;
  std::lock_guard lock(mutex_);
  if (!file_ || pipelines_.count(key)) return;
  std::vector<uint64_t> v;
  vpgpu::PipelineCache::encode(s, v);
  Writer w;
  for (uint64_t x : v) w.pod(x);
  write(kTagPipeline, w.b);
  pipelines_[key] = s;
  ++storedThisRun_;
}

RecipeStore::Snapshot RecipeStore::snapshot() const {
  std::lock_guard lock(mutex_);
  Snapshot out;
  for (const auto& [id, b] : shaders_) out.shaders.emplace_back(id, b);
  auto has = [&](uint64_t id) { return !id || shaders_.count(id); };
  for (const auto& [k, s] : pipelines_)
    if (has(s.vertexShader) && has(s.fragmentShader) && has(s.computeShader)) out.pipelines.push_back(s);
  return out;
}

std::vector<uint64_t> RecipeStore::keys() const {
  std::lock_guard lock(mutex_);
  std::vector<uint64_t> k;
  for (const auto& [key, s] : pipelines_) k.push_back(key);
  return k;
}

bool RecipeStore::stateFor(uint64_t key, vpgpu::PipelineState& out) const {
  std::lock_guard lock(mutex_);
  auto it = pipelines_.find(key);
  if (it == pipelines_.end()) return false;
  out = it->second;
  return true;
}

size_t RecipeStore::pipelineCount() const {
  std::lock_guard lock(mutex_);
  return pipelines_.size();
}

PrewarmResult prewarm(Platform& platform, vpgpu::PipelineCache& cache, const RecipeStore& store,
                      const std::function<void(uint32_t, uint32_t)>& progress, double maxSeconds) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  PrewarmResult r;
  RecipeStore::Snapshot snap = store.snapshot();
  r.recipes = uint32_t(snap.pipelines.size());
  if (snap.pipelines.empty()) return r;
  for (auto& [id, b] : snap.shaders) platform.addShader(id, std::move(b));
  std::vector<uint64_t> keys;
  for (const auto& s : snap.pipelines) {
    const vpgpu::PipelineKey k = s.key();
    keys.push_back(k.value);
    if (cache.request(s, k, vpgpu::Priority::Prefetch) == vpgpu::PipelineCache::Status::Ready) ++r.hits;
  }
  for (;;) {
    uint32_t done = 0, failed = 0;
    for (uint64_t k : keys) {
      const auto st = cache.status({k});
      if (st == vpgpu::PipelineCache::Status::Ready) ++done;
      else if (st == vpgpu::PipelineCache::Status::Failed) ++done, ++failed;
    }
    if (progress) progress(done, r.recipes);
    if (done == r.recipes) {
      r.failed = failed;
      break;
    }
    // Bounded wait (game boot / watchdog): the rest keeps compiling on the workers.
    if (maxSeconds > 0 && std::chrono::duration<double>(Clock::now() - start).count() > maxSeconds) {
      logf("gpu.prewarm timeout done=%u of %u after %.0f s (rest continues in background)", done, r.recipes, maxSeconds);
      r.failed = failed;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
  }
  r.compiled = r.recipes - r.failed - r.hits;
  r.ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  logf("gpu.prewarm recipes=%u compiled=%u hits=%u failed=%u ms=%.0f", r.recipes, r.compiled, r.hits, r.failed, r.ms);
  return r;
}

}  // namespace vpvk

extern "C" {

__attribute__((visibility("default"))) uint32_t vpvk_prewarm_count(void) {
  const std::string dir = vpvk::RecipeStore::defaultDir();
  if (dir.empty()) return 0;
  vpvk::RecipeStore store(dir);
  return uint32_t(store.snapshot().pipelines.size());
}

// Standalone platform + device with every core as a pipeline worker (the game is not running).
// Destroying the platform saves the Metal binary archive; the game's device then loads it.
__attribute__((visibility("default"))) uint32_t vpvk_prewarm(vpvk_prewarm_progress progress, void* user) {
  const std::string dir = vpvk::RecipeStore::defaultDir();
  if (dir.empty()) {
    vpvk::log("gpu.prewarm skipped=no_cache_dir");
    return 0;
  }
  vpvk::RecipeStore store(dir);
  if (!store.pipelineCount()) {
    vpvk::log("gpu.prewarm skipped=no_recipes");
    return 0;
  }
  std::unique_ptr<vpvk::Platform> platform = vpvk::createPlatform();
  const unsigned workers = std::max(2u, std::thread::hardware_concurrency());
  vpvk::PrewarmResult r;
  {
    vpgpu::Device gpu(platform->backend(), workers, [](const std::string& s) { vpvk::log(s); });
    r = vpvk::prewarm(*platform, gpu.pipelines(), store, [&](uint32_t done, uint32_t total) {
      if (progress) progress(done, total, user);
    });
  }
  platform.reset();
  vpvk::logf("gpu.prewarm app done recipes=%u workers=%u", r.recipes, workers);
  return r.recipes;
}

}  // extern "C"
