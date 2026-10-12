// What VPVulkan needs from the machine: a gpu/core backend and the native objects behind it.
// visionOS: platform_metal.mm (gpu/metal). Linux tests: the null backend (platform_null.cpp).
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "vpgpu/core/backend.h"

namespace vpvk {

// A translated shader as the backend takes it (gpu/metal2's MslShader, platform-neutral).
struct ShaderBlob {
  std::string source, entry = "vp_main";
  struct Arg {
    vpgpu::BindKind kind = vpgpu::BindKind::Buffer;
    uint16_t slot = 0;
    uint32_t offset = 0;
  };
  std::vector<Arg> arguments;
  std::array<uint32_t, 3> threadsPerGroup{1, 1, 1};
  uint8_t colorOutputs = 0xFF;
};

class Platform {
 public:
  // Registers shader `id` for pipeline compilation (thread-safe).
  virtual void addShader(uint64_t, ShaderBlob) {}
  virtual ~Platform() = default;
  virtual vpgpu::Backend& backend() = 0;
  // VK_EXT_metal_objects: id<MTLDevice>, id<MTLTexture>, id<MTLBuffer> as untyped pointers
  // (unretained); null where there is none (null backend).
  virtual void* nativeDevice() { return nullptr; }
  virtual void* nativeTexture(vpgpu::TextureId) { return nullptr; }
  virtual void* nativeBuffer(vpgpu::BufferId) { return nullptr; }
  // Texture views (VkImageView with another type, format or subresource range). Null: unsupported,
  // the image itself is bound.
  struct ViewInfo {
    vpgpu::TextureType type = vpgpu::TextureType::Tex2D;
    bool array = false;
    vpgpu::Format format = vpgpu::Format::Undefined;
    uint16_t baseMip = 0, mips = 1, baseLayer = 0, layers = 1;
  };
  virtual vpgpu::TextureId createTextureView(vpgpu::TextureId, const ViewInfo&) { return {}; }
  virtual void destroyTextureView(vpgpu::TextureId) {}
  // Blocks until all submitted GPU work finished (vkQueueWaitIdle / vkDeviceWaitIdle / fences).
  virtual void waitIdle() {}
};

// Defined by exactly one platform source linked into the dylib.
std::unique_ptr<Platform> createPlatform();

}  // namespace vpvk
