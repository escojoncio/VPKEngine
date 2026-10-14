#include "recipes.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <thread>

#include "vpvk.h"
#include "vpvk/config.h"

namespace vpvk {
namespace {

constexpr char kMagic[4] = {'V', 'Q', 'R', 'C'};
// File format. v3: stage inputs ('W' SPIR-V, 'I' input), pipelines with their inputs ('Q'), drift
// marker ('D'); MSL blobs ('S') as before. Translator changes bump kTranslatorVersion, not this.
constexpr uint32_t kRecipeVersion = 3;
constexpr uint8_t kTagShader = 'S', kTagSpirv = 'W', kTagInput = 'I', kTagPipeline = 'Q', kTagDrift = 'D';
constexpr size_t kHeaderBytes = 12;
constexpr size_t kRecordHeader = 5;  // tag + length

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
  void raw(const void* p, size_t n) {
    const auto* c = static_cast<const uint8_t*>(p);
    b.insert(b.end(), c, c + n);
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
  // Element count with a sanity bound (a corrupt count fails the record instead of allocating).
  uint32_t count(uint32_t max) {
    const uint32_t n = pod<uint32_t>();
    if (n > max) ok = false;
    return ok ? n : 0;
  }
};

uint32_t headerFields() { return uint32_t(vpgpu::PipelineCache::encodedSize()); }

uint64_t fnvBytes(const uint8_t* p, size_t n) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 0x100000001b3ull;
  return h ? h : 1;
}

void encodeShader(Writer& w, uint64_t id, const ShaderBlob& b) {
  w.pod(id);
  w.str(b.entry);
  w.str(b.source);
  for (auto t : b.threadsPerGroup) w.pod(uint32_t(t));
  w.pod(uint8_t(b.colorOutputs));
  w.pod(uint32_t(b.arguments.size()));
  for (const auto& a : b.arguments) w.pod(uint8_t(a.kind)), w.pod(uint16_t(a.slot)), w.pod(uint32_t(a.offset));
}

bool decodeShader(Reader& r, uint64_t& id, ShaderBlob& b) {
  id = r.pod<uint64_t>();
  b.entry = r.str();
  b.source = r.str();
  for (auto& t : b.threadsPerGroup) t = r.pod<uint32_t>();
  b.colorOutputs = r.pod<uint8_t>();
  const uint32_t n = r.count(4096);
  for (uint32_t i = 0; r.ok && i < n; ++i) {
    ShaderBlob::Arg a;
    a.kind = static_cast<vpgpu::BindKind>(r.pod<uint8_t>());
    a.slot = r.pod<uint16_t>();
    a.offset = r.pod<uint32_t>();
    b.arguments.push_back(a);
  }
  return r.ok;
}

// Every field of SpirvLowerOptions, fixed widths (the hash of these bytes identifies the input).
void encodeInput(Writer& w, const StageInput& in) {
  const spirv::SpirvLowerOptions& o = in.opt;
  w.pod(in.spirv);
  w.str(o.entry);
  w.pod(uint8_t(o.stage));
  w.pod(uint32_t(o.specConstants.size()));
  for (const auto& [id, v] : o.specConstants) w.pod(uint32_t(id)), w.pod(uint64_t(v));
  w.pod(uint32_t(o.viewCount));
  w.pod(uint32_t(o.layout.size()));
  for (const spirv::SpirvBinding& b : o.layout) {
    w.pod(uint32_t(b.set)), w.pod(uint32_t(b.binding)), w.pod(uint32_t(b.arraySize));
    w.pod(uint8_t(b.kind)), w.pod(uint16_t(b.slot)), w.pod(uint16_t(b.samplerSlot));
    w.pod(uint8_t(b.written)), w.pod(uint32_t(b.texelFormat));
  }
  w.pod(int32_t(o.pushConstantSlot));
  w.pod(uint32_t(o.vertexAttributes.size()));
  for (const spirv::VertexAttribute& a : o.vertexAttributes)
    w.pod(uint32_t(a.location)), w.pod(uint32_t(a.binding)), w.pod(uint32_t(a.offset)), w.pod(uint32_t(a.format)),
        w.pod(uint8_t(a.components));
  w.pod(uint32_t(o.vertexBuffers.size()));
  for (const spirv::VertexBufferBinding& v : o.vertexBuffers)
    w.pod(uint32_t(v.binding)), w.pod(uint32_t(v.stride)), w.pod(uint8_t(v.perInstance)), w.pod(uint16_t(v.slot));
  // Trailing and only when set: inputs recorded before the field keep their bytes (and hashes).
  if (o.multisampleAsSingle) w.pod(uint8_t(1));
}

bool decodeInput(Reader& r, StageInput& in) {
  spirv::SpirvLowerOptions& o = in.opt;
  in.spirv = r.pod<uint64_t>();
  o.entry = r.str();
  o.stage = static_cast<vpir::Stage>(r.pod<uint8_t>());
  for (uint32_t i = 0, n = r.count(4096); r.ok && i < n; ++i) {
    const uint32_t id = r.pod<uint32_t>();
    o.specConstants.push_back({id, r.pod<uint64_t>()});
  }
  o.viewCount = r.pod<uint32_t>();
  for (uint32_t i = 0, n = r.count(65536); r.ok && i < n; ++i) {
    spirv::SpirvBinding b;
    b.set = r.pod<uint32_t>(), b.binding = r.pod<uint32_t>(), b.arraySize = r.pod<uint32_t>();
    b.kind = static_cast<vpgpu::BindKind>(r.pod<uint8_t>());
    b.slot = r.pod<uint16_t>(), b.samplerSlot = r.pod<uint16_t>();
    b.written = r.pod<uint8_t>() != 0;
    b.texelFormat = static_cast<vpgpu::Format>(r.pod<uint32_t>());
    o.layout.push_back(b);
  }
  o.pushConstantSlot = r.pod<int32_t>();
  for (uint32_t i = 0, n = r.count(4096); r.ok && i < n; ++i) {
    spirv::VertexAttribute a;
    a.location = r.pod<uint32_t>(), a.binding = r.pod<uint32_t>(), a.offset = r.pod<uint32_t>();
    a.format = static_cast<vpgpu::Format>(r.pod<uint32_t>());
    a.components = r.pod<uint8_t>();
    o.vertexAttributes.push_back(a);
  }
  for (uint32_t i = 0, n = r.count(4096); r.ok && i < n; ++i) {
    spirv::VertexBufferBinding v;
    v.binding = r.pod<uint32_t>(), v.stride = r.pod<uint32_t>();
    v.perInstance = r.pod<uint8_t>() != 0;
    v.slot = r.pod<uint16_t>();
    o.vertexBuffers.push_back(v);
  }
  if (r.ok && r.p < r.end) o.multisampleAsSingle = r.pod<uint8_t>() != 0;
  return r.ok && in.spirv != 0;
}

void encodePipeline(Writer& w, const vpgpu::PipelineState& s, const std::array<uint64_t, 3>& inputs) {
  for (uint64_t x : inputs) w.pod(x);
  std::vector<uint64_t> v;
  vpgpu::PipelineCache::encode(s, v);
  for (uint64_t x : v) w.pod(x);
}

uint64_t* shaderField(vpgpu::PipelineState& s, int stage) {
  return stage == 0 ? &s.vertexShader : stage == 1 ? &s.fragmentShader : &s.computeShader;
}

}  // namespace

uint64_t spirvHash(const uint32_t* words, size_t count) {
  uint64_t h = 0xcbf29ce484222325ull ^ (uint64_t(count) << 1);
  for (size_t i = 0; i < count; ++i) h = (h ^ words[i]) * 0x100000001b3ull;
  h ^= h >> 29;
  return h ? h : 1;
}

uint64_t StageInput::hash() const {
  Writer w;
  encodeInput(w, *this);
  return fnvBytes(w.b.data(), w.b.size());
}

std::string RecipeStore::defaultDir() {
  const char* d = std::getenv("VP_PIPELINE_CACHE_DIR");
  return d && *d ? d : "";
}

RecipeStore::RecipeStore(std::string dir, uint32_t translatorVersion) : translator_(translatorVersion) {
  if (dir.empty()) return;
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  path_ = dir + "/recipes.bin";
  load();
  if (!openForAppend(size_ == 0)) return;
  size_t stale = 0;
  for (const auto& [h, rec] : inputs_) stale += rec.translator != translator_;
  logf("gpu.pso.recipe read shaders=%zu pipelines=%zu inputs=%zu spirv=%zu stale_inputs=%zu drift=%d dead=%zu "
       "translator=%u bytes=%llu",
       shaders_.size(), pipelines_.size(), inputs_.size(), spirv_.size(), stale, drift_ ? 1 : 0, deadRecords_,
       translator_, (unsigned long long)size_);
}

RecipeStore::~RecipeStore() {
  if (!file_) return;
  logf("gpu.pso.recipe stored=%zu total=%zu", storedThisRun_, pipelines_.size());
  std::fclose(file_);
}

bool RecipeStore::openForAppend(bool fresh) {
  file_ = std::fopen(path_.c_str(), fresh ? "wb" : "ab");
  if (!file_) {
    logf("gpu.pso.recipe what=open_failed path=%s", path_.c_str());
    return false;
  }
  if (fresh) {
    std::fwrite(kMagic, 1, 4, file_);
    const uint32_t hdr[2] = {kRecipeVersion, headerFields()};
    std::fwrite(hdr, sizeof hdr, 1, file_);
    std::fflush(file_);
    size_ = kHeaderBytes;
    records_ = deadRecords_ = 0;
  }
  return true;
}

void RecipeStore::load() {
  FILE* f = std::fopen(path_.c_str(), "rb");
  if (!f) return;
  std::vector<uint8_t> data;
  uint8_t buf[65536];
  for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) data.insert(data.end(), buf, buf + n);
  std::fclose(f);
  Reader r{data.data(), data.data() + data.size()};
  const uint32_t magic = r.pod<uint32_t>();
  const uint32_t ver = r.pod<uint32_t>(), fields = r.pod<uint32_t>();
  if (!r.ok || std::memcmp(&magic, kMagic, 4) || ver != kRecipeVersion || fields != headerFields()) {
    // Older files (v2 and before) hold MSL only, without the translator inputs: nothing to rebuild from.
    logf("gpu.pso.recipe discarded=stale version=%u fields=%u", ver, fields);
    return;  // size_ stays 0: the constructor rewrites the file from scratch
  }
  size_t bad = 0;
  size_t goodEnd = size_t(r.p - data.data());
  std::vector<PipeRec> pending;
  while (r.ok && r.p < r.end) {
    const uint8_t tag = r.pod<uint8_t>();
    const uint32_t len = r.pod<uint32_t>();
    if (!r.ok || r.end - r.p < ptrdiff_t(len)) break;  // truncated tail (crash while appending)
    const uint64_t payload = uint64_t(r.p - data.data());
    Reader rec{r.p, r.p + len};
    r.p += len;
    goodEnd = size_t(r.p - data.data());
    ++records_;
    if (tag == kTagSpirv) {
      const uint64_t h = rec.pod<uint64_t>();
      const uint32_t n = rec.pod<uint32_t>();
      if (!rec.ok || uint64_t(rec.end - rec.p) != uint64_t(n) * 4 || !n) {
        ++bad;
        continue;
      }
      if (!spirv_.emplace(h, SpirvRec{payload + 12, n}).second) ++deadRecords_;
    } else if (tag == kTagShader) {
      uint64_t id;
      ShaderBlob b;
      if (!decodeShader(rec, id, b)) {
        ++bad;
        continue;
      }
      if (!shaders_.emplace(id, std::move(b)).second) ++deadRecords_;
    } else if (tag == kTagInput) {
      InputRec in;
      const uint64_t h = rec.pod<uint64_t>();
      in.translator = rec.pod<uint32_t>();
      in.msl = rec.pod<uint64_t>();
      if (!decodeInput(rec, in.input) || !in.msl) {
        ++bad;
        continue;
      }
      auto [it, added] = inputs_.try_emplace(h);
      if (!added) ++deadRecords_;
      it->second = std::move(in);  // the latest record of an input wins
    } else if (tag == kTagPipeline) {
      PipeRec p;
      for (auto& x : p.inputs) x = rec.pod<uint64_t>();
      std::vector<uint64_t> v(headerFields());
      for (auto& x : v) x = rec.pod<uint64_t>();
      if (!rec.ok || !vpgpu::PipelineCache::decode(v.data(), v.size(), p.state)) {
        ++bad;
        continue;
      }
      p.state.label = "vk.prewarm";
      pending.push_back(std::move(p));
    } else if (tag == kTagDrift) {
      drift_ = true;
    } else {
      ++bad;
    }
  }
  size_ = goodEnd;
  // Pipelines name their shaders through the stage inputs: a later 'I' record (same input, other
  // MSL) moves them along.
  size_t orphans = 0;
  for (PipeRec& p : pending) {
    if (!remapLocked(p)) {
      ++orphans, ++deadRecords_;
      continue;
    }
    insertPipelineLocked(std::move(p));
  }
  if (bad || orphans) logf("gpu.pso.recipe bad_records=%zu orphan_pipelines=%zu", bad, orphans);
  if (goodEnd < data.size()) {  // drop a torn tail so later appends stay framed
    std::error_code ec;
    std::filesystem::resize_file(path_, goodEnd, ec);
    logf("gpu.pso.recipe truncated_tail=%zu", data.size() - goodEnd);
  }
}

bool RecipeStore::remapLocked(PipeRec& p) const {
  for (int s = 0; s < 3; ++s) {
    uint64_t* id = shaderField(p.state, s);
    if (!*id && !p.inputs[s]) continue;
    if (!p.inputs[s]) return false;
    auto it = inputs_.find(p.inputs[s]);
    if (it == inputs_.end()) return false;
    *id = it->second.msl;
    if (s == 1) {  // outputs the fragment shader does not write keep their contents (as createGraphics)
      auto sh = shaders_.find(*id);
      if (sh != shaders_.end())
        for (int c = 0; c < vpgpu::kMaxColorAttachments; ++c)
          if (!(sh->second.colorOutputs & (1u << c))) p.state.blend[c].writeMask = 0;
    }
  }
  return true;
}

void RecipeStore::insertPipelineLocked(PipeRec p) {
  const uint64_t key = p.state.key().value;
  if (!pipelines_.emplace(key, std::move(p)).second) ++deadRecords_;
}

void RecipeStore::write(uint8_t tag, const std::vector<uint8_t>& payload, uint64_t* payloadOffset) {
  const uint32_t len = uint32_t(payload.size());
  if (payloadOffset) *payloadOffset = size_ + kRecordHeader;
  std::fwrite(&tag, 1, 1, file_);
  std::fwrite(&len, sizeof len, 1, file_);
  if (len) std::fwrite(payload.data(), 1, payload.size(), file_);
  std::fflush(file_);
  size_ += kRecordHeader + len;
  ++records_;
}

bool RecipeStore::readWords(const SpirvRec& s, std::vector<uint32_t>& out) const {
  out.clear();
  FILE* f = std::fopen(path_.c_str(), "rb");
  if (!f) return false;
  bool ok = std::fseek(f, long(s.offset), SEEK_SET) == 0;
  if (ok) {
    out.resize(s.count);
    ok = std::fread(out.data(), 4, out.size(), f) == out.size();
  }
  std::fclose(f);
  if (!ok) out.clear();
  return ok;
}

void RecipeStore::addStage(const StageInput& in, uint64_t h, const std::vector<uint32_t>& words, uint64_t msl,
                           const ShaderBlob& blob) {
  if (words.empty() || !msl) return;
  std::lock_guard lock(mutex_);
  if (!file_) return;
  auto it = inputs_.find(h);
  const bool known = it != inputs_.end();
  if (known && it->second.msl == msl && it->second.translator == translator_ && shaders_.count(msl) &&
      spirv_.count(in.spirv))
    return;  // the common case after the first run: nothing to record
  if (!spirv_.count(in.spirv)) {
    Writer w;
    w.pod(in.spirv);
    w.pod(uint32_t(words.size()));
    w.raw(words.data(), words.size() * 4);
    uint64_t off = 0;
    write(kTagSpirv, w.b, &off);
    spirv_[in.spirv] = {off + 12, uint32_t(words.size())};
  }
  if (!shaders_.count(msl)) {
    Writer w;
    encodeShader(w, msl, blob);
    write(kTagShader, w.b);
    shaders_[msl] = blob;
  }
  if (known && it->second.translator == translator_ && it->second.msl != msl) {
    // Same input, same translator version, other MSL: the translator changed without a version
    // bump (or is not deterministic). The next load retranslates everything.
    static std::atomic<uint32_t> lines{0};
    if (lines++ < 8)
      logf("gpu.pso.recipe drift input=%016llx old=%016llx new=%016llx (translator changed without kTranslatorVersion bump?)",
           (unsigned long long)h, (unsigned long long)it->second.msl, (unsigned long long)msl);
    if (!driftWritten_) write(kTagDrift, {}), driftWritten_ = true;
  }
  if (known) ++deadRecords_;
  Writer w;
  w.pod(h);
  w.pod(translator_);
  w.pod(msl);
  encodeInput(w, in);
  write(kTagInput, w.b);
  inputs_[h] = InputRec{in, msl, translator_};
}

void RecipeStore::addPipeline(const vpgpu::PipelineState& s, const std::array<uint64_t, 3>& inputs) {
  const uint64_t key = s.key().value;
  std::lock_guard lock(mutex_);
  if (!file_ || pipelines_.count(key)) return;
  for (int k = 0; k < 3; ++k)  // a stage without a recorded input cannot be rebuilt later
    if ((k == 0 ? s.vertexShader : k == 1 ? s.fragmentShader : s.computeShader) && !inputs[k]) return;
  Writer w;
  encodePipeline(w, s, inputs);
  write(kTagPipeline, w.b);
  pipelines_[key] = PipeRec{s, inputs};
  ++storedThisRun_;
}

bool RecipeStore::needsRefresh() const {
  std::lock_guard lock(mutex_);
  if (drift_ || compactionDueLocked()) return true;
  for (const auto& [h, rec] : inputs_)
    if (rec.translator != translator_) return true;
  return false;
}

RecipeStore::RefreshResult RecipeStore::refresh(unsigned threads) {
  using Clock = std::chrono::steady_clock;
  const auto t0 = Clock::now();
  RefreshResult r;
  struct Job {
    uint64_t spirv = 0;
    SpirvRec rec;
    std::vector<std::pair<uint64_t, spirv::SpirvLowerOptions>> inputs;
  };
  struct Out {
    uint64_t input = 0, msl = 0;
    ShaderBlob blob;
    bool ok = false;
  };
  std::vector<Job> jobs;
  std::vector<uint64_t> unbuildable;
  {
    std::lock_guard lock(mutex_);
    if (!file_) return r;
    std::unordered_map<uint64_t, size_t> jobOf;
    for (const auto& [h, rec] : inputs_) {
      if (!drift_ && rec.translator == translator_) continue;
      auto s = spirv_.find(rec.input.spirv);
      if (s == spirv_.end()) {
        unbuildable.push_back(h);
        continue;
      }
      auto [j, added] = jobOf.emplace(rec.input.spirv, jobs.size());
      if (added) jobs.push_back(Job{rec.input.spirv, s->second, {}});
      jobs[j->second].inputs.push_back({h, rec.input.opt});
    }
    if (jobs.empty() && unbuildable.empty() && !drift_ && !compactionDueLocked()) return r;
  }

  // Translation runs unlocked, one SPIR-V module per job (parsed once for all its inputs).
  std::vector<std::vector<Out>> outs(jobs.size());
  std::atomic<size_t> next{0};
  std::atomic<uint32_t> failLines{0};
  auto work = [&] {
    for (size_t j; (j = next++) < jobs.size();) {
      const Job& job = jobs[j];
      std::vector<uint32_t> words;
      std::string error;
      spirv::SpirvModule* m = nullptr;
      if (!readWords(job.rec, words)) error = "read";
      else if (spirvHash(words.data(), words.size()) != job.spirv) error = "spirv hash mismatch";
      else m = spirv::parse(words.data(), words.size(), error);
      for (const auto& [h, opt] : job.inputs) {
        Out o;
        o.input = h;
        std::string e = error;
        o.ok = m && translateRecipeStage(*m, opt, o.msl, o.blob, e);
        if (!o.ok && failLines++ < 16)
          logf("gpu.pso.recipe retranslate_fail input=%016llx stage=%s error=\"%s\"", (unsigned long long)h,
               vpir::toString(opt.stage), e.c_str());
        outs[j].push_back(std::move(o));
      }
      spirv::release(m);
    }
  };
  const unsigned n = unsigned(std::min<size_t>(std::clamp(threads, 1u, 16u), jobs.size()));
  if (n <= 1) {
    work();
  } else {
    std::vector<std::thread> pool;
    for (unsigned k = 0; k < n; ++k) pool.emplace_back(work);
    for (auto& t : pool) t.join();
  }

  std::lock_guard lock(mutex_);
  if (!file_) return r;
  for (uint64_t h : unbuildable) inputs_.erase(h), ++r.failed;
  for (auto& job : outs)
    for (Out& o : job) {
      auto it = inputs_.find(o.input);
      if (it == inputs_.end()) continue;
      if (!o.ok) {
        inputs_.erase(it), ++r.failed;
        continue;
      }
      it->second.msl = o.msl;
      it->second.translator = translator_;
      shaders_.try_emplace(o.msl, std::move(o.blob));
      ++r.retranslated;
    }
  // Pipelines follow their inputs to the new MSL ids (new keys); those that lost an input go.
  std::unordered_map<uint64_t, PipeRec> old = std::move(pipelines_);
  pipelines_.clear();
  for (auto& [key, p] : old) {
    if (!remapLocked(p)) {
      ++r.droppedPipelines;
      continue;
    }
    insertPipelineLocked(std::move(p));
  }
  // Keep only what some pipeline still needs.
  std::unordered_set<uint64_t> usedInputs, usedShaders, usedSpirv;
  for (const auto& [key, p] : pipelines_) {
    for (uint64_t h : p.inputs)
      if (h) usedInputs.insert(h);
    for (uint64_t id : {p.state.vertexShader, p.state.fragmentShader, p.state.computeShader})
      if (id) usedShaders.insert(id);
  }
  for (auto it = inputs_.begin(); it != inputs_.end();) {
    if (usedInputs.count(it->first)) {
      usedSpirv.insert(it->second.input.spirv);
      ++it;
    } else {
      it = inputs_.erase(it);
    }
  }
  for (auto it = shaders_.begin(); it != shaders_.end();) it = usedShaders.count(it->first) ? std::next(it) : shaders_.erase(it);
  for (auto it = spirv_.begin(); it != spirv_.end();) it = usedSpirv.count(it->first) ? std::next(it) : spirv_.erase(it);
  drift_ = false;
  r.rewritten = rewriteLocked();
  r.ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  logf("gpu.pso.recipe refresh retranslated=%u failed=%u dropped_pipelines=%u pipelines=%zu shaders=%zu spirv=%zu "
       "rewritten=%d translator=%u workers=%u ms=%.0f",
       r.retranslated, r.failed, r.droppedPipelines, pipelines_.size(), shaders_.size(), spirv_.size(),
       r.rewritten ? 1 : 0, translator_, n, r.ms);
  return r;
}

// Writes every live record to recipes.bin.tmp and renames it over the file (a crash mid-way leaves
// the old file intact). SPIR-V words are copied from the old file.
bool RecipeStore::rewriteLocked() {
  const std::string tmp = path_ + ".tmp";
  FILE* out = std::fopen(tmp.c_str(), "wb");
  if (!out) {
    logf("gpu.pso.recipe what=rewrite_open_failed path=%s", tmp.c_str());
    return false;
  }
  uint64_t size = 0;
  size_t records = 0;
  auto emit = [&](uint8_t tag, const std::vector<uint8_t>& payload) {
    const uint32_t len = uint32_t(payload.size());
    const uint64_t off = size + kRecordHeader;
    std::fwrite(&tag, 1, 1, out);
    std::fwrite(&len, sizeof len, 1, out);
    if (len) std::fwrite(payload.data(), 1, len, out);
    size += kRecordHeader + len;
    ++records;
    return off;
  };
  std::fwrite(kMagic, 1, 4, out);
  const uint32_t hdr[2] = {kRecipeVersion, headerFields()};
  std::fwrite(hdr, sizeof hdr, 1, out);
  size = kHeaderBytes;
  std::unordered_map<uint64_t, SpirvRec> spirv;
  std::vector<uint32_t> words;
  for (const auto& [h, s] : spirv_) {
    if (!readWords(s, words)) continue;  // unreadable: its inputs stay usable while the version matches
    Writer w;
    w.pod(h);
    w.pod(uint32_t(words.size()));
    w.raw(words.data(), words.size() * 4);
    spirv[h] = {emit(kTagSpirv, w.b) + 12, uint32_t(words.size())};
  }
  for (const auto& [id, b] : shaders_) {
    Writer w;
    encodeShader(w, id, b);
    emit(kTagShader, w.b);
  }
  for (const auto& [h, rec] : inputs_) {
    Writer w;
    w.pod(h);
    w.pod(rec.translator);
    w.pod(rec.msl);
    encodeInput(w, rec.input);
    emit(kTagInput, w.b);
  }
  for (const auto& [key, p] : pipelines_) {
    Writer w;
    encodePipeline(w, p.state, p.inputs);
    emit(kTagPipeline, w.b);
  }
  const bool ok = std::fflush(out) == 0 && !std::ferror(out);
  std::fclose(out);
  std::error_code ec;
  if (!ok) {
    std::filesystem::remove(tmp, ec);
    logf("gpu.pso.recipe what=rewrite_failed");
    return false;
  }
  std::fclose(file_);
  file_ = nullptr;
  std::filesystem::rename(tmp, path_, ec);
  if (ec) {
    std::filesystem::remove(tmp, ec);
    logf("gpu.pso.recipe what=rewrite_rename_failed");
    openForAppend(false);  // the old file (and the old offsets) stay valid
    return false;
  }
  spirv_ = std::move(spirv);
  size_ = size;
  records_ = records;
  deadRecords_ = 0;
  driftWritten_ = false;
  openForAppend(false);
  return file_ != nullptr;
}

RecipeStore::Snapshot RecipeStore::snapshot() const {
  std::lock_guard lock(mutex_);
  Snapshot out;
  for (const auto& [id, b] : shaders_) out.shaders.emplace_back(id, b);
  auto has = [&](uint64_t id) { return !id || shaders_.count(id); };
  for (const auto& [k, p] : pipelines_)
    if (has(p.state.vertexShader) && has(p.state.fragmentShader) && has(p.state.computeShader))
      out.pipelines.push_back(p.state);
  return out;
}

std::vector<uint64_t> RecipeStore::keys() const {
  std::lock_guard lock(mutex_);
  std::vector<uint64_t> k;
  for (const auto& [key, p] : pipelines_) k.push_back(key);
  return k;
}

bool RecipeStore::stateFor(uint64_t key, vpgpu::PipelineState& out,
                           std::vector<std::pair<uint64_t, ShaderBlob>>* shaders) const {
  std::lock_guard lock(mutex_);
  auto it = pipelines_.find(key);
  if (it == pipelines_.end()) return false;
  out = it->second.state;
  if (shaders) {
    shaders->clear();
    for (int st = 0; st < 3; ++st) {
      const uint64_t id = *shaderField(out, st);
      if (!id) continue;
      auto sh = shaders_.find(id);
      if (sh == shaders_.end()) return false;
      shaders->emplace_back(id, sh->second);
    }
  }
  return true;
}

size_t RecipeStore::pipelineCount() const {
  std::lock_guard lock(mutex_);
  return pipelines_.size();
}

size_t RecipeStore::shaderCount() const {
  std::lock_guard lock(mutex_);
  return shaders_.size();
}

std::vector<StageInput> RecipeStore::stageInputs() const {
  std::lock_guard lock(mutex_);
  std::vector<StageInput> v;
  for (const auto& [h, rec] : inputs_) v.push_back(rec.input);
  return v;
}

PrewarmResult prewarm(Platform& platform, vpgpu::PipelineCache& cache, RecipeStore& store,
                      const std::function<void(uint32_t, uint32_t)>& progress, double maxSeconds) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  PrewarmResult r;
  // Recipes recorded by another translator version: retranslated from their SPIR-V first.
  const RecipeStore::RefreshResult rf = store.refresh(std::max(2u, std::thread::hardware_concurrency()));
  r.retranslated = rf.retranslated;
  RecipeStore::Snapshot snap = store.snapshot();
  logf("gpu.pso.recipe loaded shaders=%zu pipelines=%zu retranslated=%u failed=%u ms=%.0f", snap.shaders.size(),
       snap.pipelines.size(), rf.retranslated, rf.failed, rf.ms);
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
  logf("gpu.prewarm recipes=%u compiled=%u hits=%u failed=%u retranslated=%u ms=%.0f", r.recipes, r.compiled, r.hits,
       r.failed, r.retranslated, r.ms);
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
