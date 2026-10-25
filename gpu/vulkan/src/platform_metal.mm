// visionOS platform: gpu/metal2 (VPVulkan copy of gpu/metal).
// Shaders reach the backend through `shaders_` (pipelines.cpp: SPIR-V -> VPIR -> MSL).
#import <Metal/Metal.h>

#include "vpgpu/metal2/metal_backend.h"
#include "vpgpu/metal2/mtl_attr.h"
#include "vpvk.h"
#include "taa_native_metal.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_set>

namespace vpvk {
namespace {
class MetalPlatform;
// Live platforms, for vpvk_flush_caches() on a fast-exit path. Leaked on purpose: never destroyed
// by static destructors while guest threads may still be running.
std::mutex& livePlatformsMutex() { static auto* m = new std::mutex; return *m; }
std::unordered_set<MetalPlatform*>& livePlatforms() { static auto* s = new std::unordered_set<MetalPlatform*>; return *s; }
class MetalPlatform final : public Platform {
 public:
  MetalPlatform() : backend_(shaders_, [](const std::string& s) { log(s); }, options()) {
    if (!backend_.ok()) log("vk.err what=metal_device");
    std::lock_guard<std::mutex> l(livePlatformsMutex());
    livePlatforms().insert(this);
  }
  ~MetalPlatform() override {
    { std::lock_guard<std::mutex> l(livePlatformsMutex()); livePlatforms().erase(this); }
    backend_.saveArchive();
  }
  void flushCaches() { backend_.saveArchive(); }  // pipeline binary archive, if any is pending
  vpgpu::Backend& backend() override { return backend_; }
  WaitResult waitIdle(uint64_t timeoutNs) override { return map(backend_.waitIdle(timeoutNs)); }
  WaitResult waitFrames(uint64_t frames, uint64_t timeoutNs) override {
    return map(backend_.waitFrames(frames, timeoutNs));
  }
  bool lost() const override { return backend_.lost(); }
  uint32_t gpuBacklogMs() const override { return backend_.gpuBacklogMs(); }
  uint32_t lastFrameChunks() const override { return backend_.lastFrameHangSplits(); }
  size_t trimMemory(int level, std::string& report) override { return backend_.trimMemory(level, report); }
  size_t drainRetired(uint32_t minIdleMs, std::string& report) override { return backend_.drainRetired(minIdleMs, report); }
  void diagCounters(std::vector<std::pair<const char*, uint64_t>>& out) const override { backend_.diagCounters(out); }
  void diagRanges(std::vector<std::pair<uintptr_t, uint64_t>>& out) const override { backend_.diagRanges(out); }
  std::string archiveSummary() override {
    const auto s = backend_.memStats();
    const char* reason = !s.archiveFiles && !s.archiveMisses ? "no_cache"
                         : !s.archiveMisses                  ? "all_cached"
                         : s.archiveEvicted                  ? "files_evicted"
                         : s.archiveKeyed < s.archiveFiles   ? "unindexed_files"
                                                             : "new_pipelines";
    char b[192];
    std::snprintf(b, sizeof b, "ar_hits=%u ar_misses=%u ar_files=%u ar_keyed=%u ar_evicted=%u reason=%s", s.archiveHits,
                  s.archiveMisses, s.archiveFiles, s.archiveKeyed, s.archiveEvicted, reason);
    return b;
  }
  std::string memStats() override {
    const auto s = backend_.memStats();
    char b[896];
    std::snprintf(b, sizeof b,
                  "alloc=%llu ws_max=%llu tex_astc=%llu tex_color=%llu tex_depth=%llu tex_msaa=%llu buf_own=%llu "
                  "buf_guest=%llu pso=%u fn=%u lib=%u lib_compiled=%llu ar_entries=%u ar_files=%u tile_rejected=%llu msaa_memoryless=%llu "
                  "held_b=%llu views=%llu args_b=%llu buf_int_b=%llu taa_b=%llu fov_b=%llu",
                  (unsigned long long)s.deviceAllocated, (unsigned long long)s.workingSetMax,
                  (unsigned long long)s.texAstc, (unsigned long long)s.texColor, (unsigned long long)s.texDepth,
                  (unsigned long long)s.texMsaa, (unsigned long long)s.bufOwn, (unsigned long long)s.bufGuest,
                  s.pipelines, s.functions, s.libraries, (unsigned long long)s.libCompiled, s.archiveEntries,
                  s.archiveFiles, (unsigned long long)s.tileRejected, (unsigned long long)s.msaaMemoryless,
                  (unsigned long long)s.heldBytes, (unsigned long long)s.views, (unsigned long long)s.argsBytes,
                  (unsigned long long)s.bufIntBytes, (unsigned long long)taaNativeBytes(), (unsigned long long)s.fovBytes);
    std::string out = b;
    if (!s.ledger.empty()) out += " ledger: " + s.ledger;  // VPKEngine memory by owner (metal_backend.mm memLedger)
    return out;
  }
  bool encodeNative(void (*fn)(void*, void*), void* user) override { return backend_.encodeNative(fn, user); }
  // VPKTAA nativo: creado al primer uso (dentro de vpvk_native_submit: un solo hilo).
  bool taaNativeEncode(const TaaNativeArgs& a, void* cb) override { return taa()->encode(a, cb); }
  void* depthResolveEncode(void* cb, void* depth, bool rev, int mode, int key) override {
    return taa()->resolveDepth(cb, depth, rev, mode, key);
  }
  void addShader(uint64_t id, ShaderBlob b) override {
    vpgpu::metal2::MslShader m;
    m.source = std::move(b.source);
    m.entry = b.entry;
    m.argumentBufferIndex = 0;  // the MSL emitter's [[buffer(0)]] argument table
    m.threadsPerGroup = b.threadsPerGroup;
    m.colorOutputs = b.colorOutputs;
    for (const auto& a : b.arguments) m.arguments.push_back({a.kind, a.slot, a.offset});
    shaders_.add(id, std::move(m));
    backend_.prewarmShader(id);  // compile the MTLLibrary now, off the pipeline worker's critical path
  }
  bool compileShaderNow(uint64_t id, ShaderBlob b, std::string& error) override {
    vpgpu::metal2::MslShader m;
    m.source = std::move(b.source);
    m.entry = b.entry;
    m.argumentBufferIndex = 0;
    m.threadsPerGroup = b.threadsPerGroup;
    m.colorOutputs = b.colorOutputs;
    for (const auto& a : b.arguments) m.arguments.push_back({a.kind, a.slot, a.offset});
    shaders_.add(id, std::move(m));
    return backend_.compileShader(id, error);
  }
  // Unretained pointers (VK_EXT_metal_objects): the image's texture is allocated here if it was lazy.
  void* nativeDevice() override { return backend_.nativeDevice(); }
  vpgpu::TextureId createTextureView(vpgpu::TextureId base, const ViewInfo& i) override {
    vpgpu::metal2::MetalBackend::ViewDesc d;
    d.type = i.type, d.array = i.array, d.format = i.format;
    d.baseMip = i.baseMip, d.mips = i.mips, d.baseLayer = i.baseLayer, d.layers = i.layers;
    for (int c = 0; c < 4; ++c) d.swizzle[c] = i.swizzle[c];
    return backend_.createView(base, d);
  }
  void destroyTextureView(vpgpu::TextureId v) override { backend_.destroyView(v); }
  void* nativeTexture(vpgpu::TextureId id) override { return backend_.nativeTexture(id); }
  void* retainNative(void* p) override { return p ? (void*)CFBridgingRetain((__bridge id)p) : nullptr; }
  void releaseNative(void* p) override {
    if (p) (void)CFBridgingRelease(p);
  }

 private:
  // VP_PIPELINE_CACHE_DIR (set by the app per game): persistent MTLBinaryArchive across runs.
  static vpgpu::metal2::MetalOptions options() {
    vpgpu::metal2::MetalOptions o;
    if (const char* dir = std::getenv("VP_PIPELINE_CACHE_DIR"); dir && *dir) o.cacheRoot = dir;
    return o;
  }
  static WaitResult map(vpgpu::metal2::MetalBackend::WaitResult r) {
    using R = vpgpu::metal2::MetalBackend::WaitResult;
    return r == R::Done ? WaitResult::Done : r == R::Timeout ? WaitResult::Timeout : WaitResult::Lost;
  }
  TaaNativeMetal* taa() {
    if (!taa_) taa_ = std::make_unique<TaaNativeMetal>(backend_.nativeDevice());
    return taa_.get();
  }
  vpgpu::metal2::ShaderTable shaders_;
  vpgpu::metal2::MetalBackend backend_;
  std::unique_ptr<TaaNativeMetal> taa_;
};
}  // namespace

std::unique_ptr<Platform> createPlatform() { return std::make_unique<MetalPlatform>(); }

// Largest sample count the system GPU renders to, probed once (8, 4, 2). Apple GPUs support the
// same counts for colour and depth formats (Metal feature set tables), so one probe covers both.
int platformMaxSampleCount() {
  static const int n = [] {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    int max = 1;
    if (dev)
      for (int c : {8, 4, 2})
        if ([dev supportsTextureSampleCount:c]) {
          max = c;
          break;
        }
    logf("vk.msaa device_max=%d (supportsTextureSampleCount 8=%d 4=%d 2=%d)", max,
         dev && [dev supportsTextureSampleCount:8] ? 1 : 0, dev && [dev supportsTextureSampleCount:4] ? 1 : 0,
         dev && [dev supportsTextureSampleCount:2] ? 1 : 0);
    return max;
  }();
  return n;
}

std::string platformAttrSummary() { return vpgpu::metal2::attr::summary(); }  // mtl_attr.h

// Rate maps with the two layers (one per eye) the foveation uses; probed once.
bool platformRateMapSupported() {
  static const bool ok = [] {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    const bool s = dev && [dev supportsRasterizationRateMapWithLayerCount:2];
    logf("vk.fov device rate_map_2_layers=%d", s ? 1 : 0);
    return s;
  }();
  return ok;
}
}  // namespace vpvk

// Saves pending on-disk caches (pipeline binary archive) of every live device. For hosts that
// leave with _exit() (no destructors) — found with dlsym(RTLD_DEFAULT, "vpvk_flush_caches").
extern "C" __attribute__((visibility("default"))) void vpvk_flush_caches(void) {
  std::lock_guard<std::mutex> l(vpvk::livePlatformsMutex());
  for (auto* p : vpvk::livePlatforms()) p->flushCaches();
}
namespace vpvk {

}  // namespace vpvk
