// visionOS platform: gpu/metal2 (VPVulkan copy of gpu/metal).
// Shaders reach the backend through `shaders_` (pipelines.cpp: SPIR-V -> VPIR -> MSL).
#include "vpgpu/metal2/metal_backend.h"
#include "vpvk.h"

namespace vpvk {
namespace {
class MetalPlatform final : public Platform {
 public:
  MetalPlatform() : backend_(shaders_, [](const std::string& s) { log(s); }) {
    if (!backend_.ok()) log("vk.err what=metal_device");
  }
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
  void* nativeTexture(vpgpu::TextureId id) override { return backend_.nativeTexture(id); }

 private:
  vpgpu::metal2::ShaderTable shaders_;
  vpgpu::metal2::MetalBackend backend_;
};
}  // namespace

std::unique_ptr<Platform> createPlatform() { return std::make_unique<MetalPlatform>(); }

}  // namespace vpvk
