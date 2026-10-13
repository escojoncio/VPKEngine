// visionOS platform: gpu/metal2 (VPVulkan copy of gpu/metal).
// Shaders reach the backend through `shaders_` (pipelines.cpp: SPIR-V -> VPIR -> MSL).
#import <Metal/Metal.h>

#include "vpgpu/metal2/metal_backend.h"
#include "vpvk.h"

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
  bool encodeNative(void (*fn)(void*, void*), void* user) override { return backend_.encodeNative(fn, user); }
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
  vpgpu::metal2::ShaderTable shaders_;
  vpgpu::metal2::MetalBackend backend_;
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
}  // namespace vpvk

// Saves pending on-disk caches (pipeline binary archive) of every live device. For hosts that
// leave with _exit() (no destructors) — found with dlsym(RTLD_DEFAULT, "vpvk_flush_caches").
extern "C" __attribute__((visibility("default"))) void vpvk_flush_caches(void) {
  std::lock_guard<std::mutex> l(vpvk::livePlatformsMutex());
  for (auto* p : vpvk::livePlatforms()) p->flushCaches();
}
namespace vpvk {

}  // namespace vpvk
