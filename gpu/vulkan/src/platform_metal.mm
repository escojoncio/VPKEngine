// visionOS platform: gpu/metal's Metal 4 backend.
// Shaders reach the backend through `shaders_` once pipelines exist (phase 3: SPIR-V -> VPIR -> MSL).
#include "vpgpu/metal/metal_backend.h"
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
  // Unretained pointers (VK_EXT_metal_objects): the image's texture is allocated here if it was lazy.
  void* nativeDevice() override { return backend_.nativeDevice(); }
  void* nativeTexture(vpgpu::TextureId id) override { return backend_.nativeTexture(id); }

 private:
  vpgpu::metal::ShaderTable shaders_;
  vpgpu::metal::MetalBackend backend_;
};
}  // namespace

std::unique_ptr<Platform> createPlatform() { return std::make_unique<MetalPlatform>(); }

}  // namespace vpvk
