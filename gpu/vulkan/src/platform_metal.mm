// visionOS platform: gpu/metal2 (VPVulkan copy of gpu/metal).
// Shaders reach the backend through `shaders_` (pipelines.cpp: SPIR-V -> VPIR -> MSL).
#include "vpgpu/metal2/metal_backend.h"
#include "vpvk.h"

#include <cstdlib>

namespace vpvk {
namespace {
class MetalPlatform final : public Platform {
 public:
  MetalPlatform() : backend_(shaders_, [](const std::string& s) { log(s); }, options()) {
    if (!backend_.ok()) log("vk.err what=metal_device");
  }
  ~MetalPlatform() override { backend_.saveArchive(); }  // pipeline binary archive, if any is pending
  vpgpu::Backend& backend() override { return backend_; }
  void waitIdle() override { backend_.waitIdle(); }
  void addShader(uint64_t id, ShaderBlob b) override {
    vpgpu::metal2::MslShader m;
    m.source = std::move(b.source);
    m.entry = b.entry;
    m.argumentBufferIndex = 0;  // the MSL emitter's [[buffer(0)]] argument table
    m.threadsPerGroup = b.threadsPerGroup;
    m.colorOutputs = b.colorOutputs;
    for (const auto& a : b.arguments) m.arguments.push_back({a.kind, a.slot, a.offset});
    shaders_.add(id, std::move(m));
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
  vpgpu::metal2::ShaderTable shaders_;
  vpgpu::metal2::MetalBackend backend_;
};
}  // namespace

std::unique_ptr<Platform> createPlatform() { return std::make_unique<MetalPlatform>(); }

}  // namespace vpvk
