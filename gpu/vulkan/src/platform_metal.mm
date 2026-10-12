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
  // PENDIENTE: MetalBackend aún no expone su MTLDevice/MTLTexture (gpu/metal es de otra sesión).
  // Hace falta `void* nativeTexture(TextureId)` + `void* nativeDevice()` en metal_backend.h
  // (devolviendo (__bridge void*) del id<MTLTexture>, forzando la asignación perezosa).
  void* nativeTexture(vpgpu::TextureId) override { return nullptr; }

 private:
  vpgpu::metal::ShaderTable shaders_;
  vpgpu::metal::MetalBackend backend_;
};
}  // namespace

std::unique_ptr<Platform> createPlatform() { return std::make_unique<MetalPlatform>(); }

}  // namespace vpvk
