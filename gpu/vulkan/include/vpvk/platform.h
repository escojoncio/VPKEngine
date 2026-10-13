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
  // Registers and compiles shader `id`'s library now on the calling thread (first-run precompile,
  // precompile.cpp). True when compiled (or nothing to compile on this platform).
  virtual bool compileShaderNow(uint64_t id, ShaderBlob b, std::string&) {
    addShader(id, std::move(b));
    return true;
  }
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
    // Sampling swizzle per r,g,b,a as MTLTextureSwizzle values (0 zero, 1 one, 2..5 r,g,b,a).
    uint8_t swizzle[4] = {2, 3, 4, 5};
  };
  virtual vpgpu::TextureId createTextureView(vpgpu::TextureId, const ViewInfo&) { return {}; }
  virtual void destroyTextureView(vpgpu::TextureId) {}
  // GPU waits. Lost: the device is lost (a hung or rejected command buffer, see gpu/metal2's
  // MetalBackend): waits return it, nothing new runs, and Vulkan reports VK_ERROR_DEVICE_LOST.
  enum class WaitResult { Done, Timeout, Lost };
  // All submitted GPU work finished (vkQueueWaitIdle / vkDeviceWaitIdle). UINT64_MAX: no timeout.
  virtual WaitResult waitIdle(uint64_t /*timeoutNs*/) { return WaitResult::Done; }
  // Every vpgpu::Device frame with an index below `frames` finished (fences). 0 ns polls.
  virtual WaitResult waitFrames(uint64_t /*frames*/, uint64_t /*timeoutNs*/) { return WaitResult::Done; }
  virtual bool lost() const { return false; }
  // vpvk_native_submit: a native command buffer (id<MTLCommandBuffer>, unretained) on the queue of
  // the guest's work, ordered after every frame submitted so far and before the next; `fn` encodes
  // into it, then it is committed. Called with the device's submit mutex held. False: no native
  // queue (null backend).
  virtual bool encodeNative(void (*)(void* commandBuffer, void* user), void*) { return false; }
};

// Defined by exactly one platform source linked into the dylib.
std::unique_ptr<Platform> createPlatform();

}  // namespace vpvk
