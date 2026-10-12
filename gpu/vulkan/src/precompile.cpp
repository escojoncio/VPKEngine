// First-run shader precompile: SPIR-V modules harvested from a game's data (VisionQuestPro
// vq_harvest.c) are translated (SPIR-V -> VPIR -> MSL) and their Metal libraries compiled before
// the game runs, on every core. The OS Metal compiler cache keeps the compiled libraries keyed by
// MSL source, so the game's own compile of the same MSL is a cache hit; the MSL ids go to
// <VP_PIPELINE_CACHE_DIR>/msl_index.bin so the runtime can count hits/misses (gpu.shader.lib).
//
// MSL depends on pipeline state the module alone does not carry. The variants compiled here are the
// most likely ones for a Quest Unity game: layout = the module's own bindings in (set, binding)
// order with the driver's slot assignment, push constants after the descriptors, default spec
// constants, viewCount 1 and 2 when the shader reads ViewIndex. Vertex shaders depend on the
// game's vertex layout (fetch is generated into the MSL) and are compiled with a guessed tightly
// packed float layout, so they hit less often.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_set>

#include "vpmsl/msl.h"
#include "vpvk/config.h"
#include "vpvk.h"

namespace vpvk {
namespace {

uint64_t fnvStr(const std::string& s) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (unsigned char c : s) h = (h ^ c) * 0x100000001b3ull;
  return h ? h : 1;
}

// ---- minimal reflection over the raw words -------------------------------------------------
struct Entry {
  vpir::Stage stage;
  std::string name;
};
struct Reflect {
  std::vector<Entry> entries;
  std::vector<spirv::SpirvBinding> bindings;
  bool push = false, viewIndex = false;
  std::vector<std::pair<uint32_t, uint32_t>> inputs;  // vertex: (location, components)
};

bool reflect(const std::vector<uint32_t>& w, Reflect& r) {
  if (w.size() < 6) return false;
  std::map<uint32_t, uint32_t> set, binding, location, builtin, constant, vecSize;
  std::unordered_set<uint32_t> block, bufferBlock;
  struct Ty {
    uint32_t op = 0, a = 0, b = 0, c = 0;
  };
  std::map<uint32_t, Ty> types;
  struct Var {
    uint32_t id, ptr, storage;
  };
  std::vector<Var> vars;
  for (size_t i = 5; i < w.size();) {
    const uint32_t wc = w[i] >> 16, op = w[i] & 0xFFFF;
    if (!wc || i + wc > w.size()) return false;
    const uint32_t* o = &w[i + 1];
    switch (op) {
      case 15: {  // OpEntryPoint
        if (wc < 4) break;
        Entry e;
        e.stage = o[0] == 4 ? vpir::Stage::Fragment : o[0] == 5 ? vpir::Stage::Compute : vpir::Stage::Vertex;
        if (o[0] != 0 && o[0] != 4 && o[0] != 5) break;  // tess / geometry: not supported by the driver
        e.name.assign(reinterpret_cast<const char*>(o + 2), strnlen(reinterpret_cast<const char*>(o + 2), (wc - 3) * 4));
        r.entries.push_back(e);
        break;
      }
      case 71:  // OpDecorate
        if (wc >= 3) {
          if (o[1] == 2) block.insert(o[0]);
          if (o[1] == 3) bufferBlock.insert(o[0]);
          if (wc >= 4) {
            if (o[1] == 34) set[o[0]] = o[2];
            if (o[1] == 33) binding[o[0]] = o[2];
            if (o[1] == 30) location[o[0]] = o[2];
            if (o[1] == 11) builtin[o[0]] = o[2];
          }
        }
        break;
      case 72:  // OpMemberDecorate BuiltIn (ViewIndex is never a member, ignore)
        break;
      case 43:  // OpConstant
        if (wc >= 4) constant[o[1]] = o[2];
        break;
      case 23:  // OpTypeVector
        if (wc >= 4) vecSize[o[0]] = o[2];
        [[fallthrough]];
      case 21: case 22: case 25: case 26: case 27: case 28: case 29: case 30: case 32:
        if (wc >= 2) types[o[0]] = {op, wc > 2 ? o[1] : 0, wc > 3 ? o[2] : 0, wc > 7 ? o[6] : 0};
        break;
      case 59:  // OpVariable
        if (wc >= 4) vars.push_back({o[1], o[0], o[2]});
        break;
      default: break;
    }
    i += wc;
  }
  for (const Var& v : vars) {
    if (builtin.count(v.id) && builtin[v.id] == 4440) r.viewIndex = true;
    if (v.storage == 9) r.push = true;
    const Ty pt = types[v.ptr];  // OpTypePointer: a = storage, b = pointee
    uint32_t t = pt.b, count = 1;
    if (v.storage == 1 && location.count(v.id)) {  // vertex inputs
      const Ty& ty = types[t];
      r.inputs.push_back({location[v.id], ty.op == 23 ? vecSize[t] : 1});
    }
    if (!set.count(v.id) && !binding.count(v.id)) continue;
    while (types[t].op == 28 || types[t].op == 29) {  // arrays
      if (types[t].op == 28) count *= std::max<uint32_t>(1, constant[types[t].b]);
      t = types[t].a;
    }
    const Ty& ty = types[t];
    spirv::SpirvBinding b;
    b.set = set[v.id], b.binding = binding[v.id], b.arraySize = count;
    if (v.storage == 12 || (v.storage == 2 && bufferBlock.count(t))) b.kind = vpgpu::BindKind::Buffer;
    else if (v.storage == 2) b.kind = vpgpu::BindKind::ConstantBuffer;
    else if (ty.op == 26) b.kind = vpgpu::BindKind::Sampler;
    else if (ty.op == 27) b.kind = vpgpu::BindKind::Texture, b.samplerSlot = 0;  // combined: sampler slot below
    else if (ty.op == 25) {
      const uint32_t dim = ty.b;
      // OpTypeImage words: result, sampledType, dim, depth, arrayed, ms, sampled, format; c = sampled
      if (dim == 5) b.kind = vpgpu::BindKind::Buffer;  // texel buffer
      else b.kind = ty.c == 2 ? vpgpu::BindKind::StorageTexture : vpgpu::BindKind::Texture;
    } else continue;
    r.bindings.push_back(b);
  }
  std::sort(r.bindings.begin(), r.bindings.end(),
            [](const auto& a, const auto& b) { return a.set != b.set ? a.set < b.set : a.binding < b.binding; });
  return !r.entries.empty();
}

// Same assignment as assignSlots() in pipelines.cpp (per-kind counters in (set, binding) order).
int assign(std::vector<spirv::SpirvBinding>& bs, bool push) {
  uint16_t next[6] = {0, 0, 0, 0, 0, 0};
  for (auto& b : bs) {
    const bool combined = b.samplerSlot != 0xFFFF;
    b.slot = next[int(b.kind)];
    next[int(b.kind)] = uint16_t(next[int(b.kind)] + std::max<uint32_t>(1, b.arraySize));
    if (combined) {
      b.samplerSlot = next[int(vpgpu::BindKind::Sampler)];
      next[int(vpgpu::BindKind::Sampler)] = uint16_t(next[int(vpgpu::BindKind::Sampler)] + std::max<uint32_t>(1, b.arraySize));
    }
  }
  return push ? next[int(vpgpu::BindKind::ConstantBuffer)] : -1;
}

std::mutex gIndexMutex;
std::unordered_set<uint64_t>* gIndex = nullptr;  // precompiled MSL ids (loaded lazily at runtime)
std::atomic<uint32_t> gHit{0}, gMiss{0};

}  // namespace

// Runtime: called by translateStage with the MSL id. Logs gpu.shader.lib hit/miss periodically.
void noteShaderLib(uint64_t mslId) {
  {
    std::lock_guard<std::mutex> l(gIndexMutex);
    if (!gIndex) {
      gIndex = new std::unordered_set<uint64_t>;
      const std::string dir = RecipeStore::defaultDir();
      if (!dir.empty())
        if (FILE* f = std::fopen((dir + "/msl_index.bin").c_str(), "rb")) {
          uint64_t v;
          while (std::fread(&v, 8, 1, f) == 1) gIndex->insert(v);
          std::fclose(f);
        }
      logf("gpu.shader.lib index=%zu", gIndex->size());
    }
  }
  const bool hit = gIndex->count(mslId) != 0;
  const uint32_t h = hit ? ++gHit : gHit.load(), m = hit ? gMiss.load() : ++gMiss;
  if (!hit && m <= 16) logf("gpu.shader.lib miss id=%016llx", (unsigned long long)mslId);
  if ((h + m) <= 8 || (h + m) % 32 == 0) logf("gpu.shader.lib hit=%u miss=%u", h, m);
}

struct PrecompileResult {
  uint32_t modules = 0, libs = 0, failed = 0;
  double ms = 0;
};

PrecompileResult precompileDir(Platform& platform, const std::string& dir, const std::string& cacheDir,
                               const std::function<void(uint32_t, uint32_t)>& progress) {
  const auto t0 = std::chrono::steady_clock::now();
  PrecompileResult r;
  std::vector<std::string> files;
  std::error_code ec;
  for (auto& e : std::filesystem::directory_iterator(dir, ec))
    if (e.path().extension() == ".spv") files.push_back(e.path().string());
  std::sort(files.begin(), files.end());
  r.modules = uint32_t(files.size());
  std::atomic<uint32_t> next{0}, done{0}, libs{0}, failed{0};
  std::mutex idsMutex;
  std::unordered_set<uint64_t> ids;
  auto work = [&] {
    for (;;) {
      const uint32_t i = next++;
      if (i >= files.size()) return;
      std::vector<uint32_t> words;
      if (FILE* f = std::fopen(files[i].c_str(), "rb")) {
        std::fseek(f, 0, SEEK_END);
        const long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (n > 0 && n % 4 == 0) {
          words.resize(size_t(n) / 4);
          if (std::fread(words.data(), 4, words.size(), f) != words.size()) words.clear();
        }
        std::fclose(f);
      }
      Reflect rf;
      std::string error;
      spirv::SpirvModule* mod = words.empty() ? nullptr : spirv::parse(words.data(), words.size(), error);
      if (!mod || !reflect(words, rf)) {
        ++failed;
        logf("gpu.precompile fail file=%s error=\"%s\"", files[i].c_str(), mod ? "reflect" : error.c_str());
        spirv::release(mod);
        ++done;
        continue;
      }
      for (const Entry& e : rf.entries) {
        for (uint32_t views = 1; views <= (rf.viewIndex ? 2u : 1u); ++views) {
          spirv::SpirvLowerOptions opt;
          opt.entry = e.name, opt.stage = e.stage, opt.viewCount = views;
          opt.layout = rf.bindings;
          opt.pushConstantSlot = assign(opt.layout, rf.push);
          if (e.stage == vpir::Stage::Vertex && !rf.inputs.empty()) {
            // Guess: one interleaved float buffer (binding 0) in location order.
            auto in = rf.inputs;
            std::sort(in.begin(), in.end());
            uint32_t off = 0;
            uint16_t bufSlot = 0;
            for (auto& b : opt.layout)
              if (b.kind == vpgpu::BindKind::Buffer) bufSlot = uint16_t(std::max<uint32_t>(bufSlot, b.slot + b.arraySize));
            for (auto& [loc, comps] : in) {
              spirv::VertexAttribute a;
              a.location = loc, a.binding = 0, a.offset = off;
              a.format = comps == 1 ? vpgpu::Format::R32Float : comps == 2 ? vpgpu::Format::RG32Float : vpgpu::Format::RGBA32Float;
              a.components = uint8_t(comps == 3 ? 3 : 0);
              off += 4 * std::max<uint32_t>(1, comps);
              opt.vertexAttributes.push_back(a);
            }
            spirv::VertexBufferBinding vb;
            vb.binding = 0, vb.stride = off, vb.slot = bufSlot;
            opt.vertexBuffers.push_back(vb);
          }
          vpir::Module ir;
          spirv::SpirvInterface iface;
          if (!spirv::lower(*mod, opt, ir, iface, error)) {
            ++failed;
            logf("gpu.precompile fail file=%s entry=%s error=\"%s\"", files[i].c_str(), e.name.c_str(), error.c_str());
            continue;
          }
          vpmsl::EmitResult em = vpmsl::emitMsl(ir);
          if (!em.errors.empty()) {
            ++failed;
            logf("gpu.precompile fail file=%s entry=%s error=\"msl %s\"", files[i].c_str(), e.name.c_str(), em.errors[0].c_str());
            continue;
          }
          const uint64_t id = fnvStr(em.shader.source);
          {
            std::lock_guard<std::mutex> l(idsMutex);
            if (!ids.insert(id).second) continue;
          }
          ShaderBlob blob;
          blob.source = std::move(em.shader.source);
          blob.threadsPerGroup = em.shader.workgroupSize;
          blob.colorOutputs = em.shader.colorOutputs;
          for (const auto& a : em.shader.arguments)
            blob.arguments.push_back({static_cast<vpgpu::BindKind>(int(a.kind) + 1), uint16_t(a.index), a.offset});
          std::string cerr;
          if (platform.compileShaderNow(id, std::move(blob), cerr)) ++libs;
          else {
            ++failed;
            logf("gpu.precompile fail file=%s entry=%s error=\"%s\"", files[i].c_str(), e.name.c_str(), cerr.c_str());
          }
        }
      }
      spirv::release(mod);
      ++done;
    }
  };
  const unsigned n = std::max(2u, std::thread::hardware_concurrency());
  std::vector<std::thread> pool;
  for (unsigned k = 0; k < n; ++k) pool.emplace_back(work);
  while (done < files.size()) {
    if (progress) progress(done, r.modules);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
  }
  for (auto& t : pool) t.join();
  if (progress) progress(r.modules, r.modules);
  if (!cacheDir.empty()) {
    std::filesystem::create_directories(cacheDir, ec);
    if (FILE* f = std::fopen((cacheDir + "/msl_index.bin").c_str(), "wb")) {
      for (uint64_t id : ids) std::fwrite(&id, 8, 1, f);
      std::fclose(f);
    }
  }
  r.libs = libs, r.failed = failed;
  r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  logf("gpu.precompile modules=%u libs=%u failed=%u workers=%u ms=%.0f", r.modules, r.libs, r.failed, n, r.ms);
  return r;
}

}  // namespace vpvk

extern "C" {
// Translates and compiles every <dir>/*.spv (harvested by the app) into the Metal compiler cache
// and writes the MSL index into VP_PIPELINE_CACHE_DIR. Returns the libraries compiled.
__attribute__((visibility("default"))) uint32_t vpvk_precompile_spirv(const char* dir, vpvk_prewarm_progress progress,
                                                                      void* user) {
  if (!dir || !*dir) return 0;
  std::unique_ptr<vpvk::Platform> platform = vpvk::createPlatform();
  const auto r = vpvk::precompileDir(*platform, dir, vpvk::RecipeStore::defaultDir(), [&](uint32_t d, uint32_t t) {
    if (progress) progress(d, t, user);
  });
  return r.libs;
}
}
