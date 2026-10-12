// What VPVulkan needs from the machine: a gpu/core backend and the native objects behind it.
// visionOS: platform_metal.mm (gpu/metal). Linux tests: the null backend (platform_null.cpp).
#pragma once

#include <memory>

#include "vpgpu/core/backend.h"

namespace vpvk {

class Platform {
 public:
  virtual ~Platform() = default;
  virtual vpgpu::Backend& backend() = 0;
  // VK_EXT_metal_objects: id<MTLDevice>, id<MTLTexture>, id<MTLBuffer> as untyped pointers
  // (unretained); null where there is none (null backend).
  virtual void* nativeDevice() { return nullptr; }
  virtual void* nativeTexture(vpgpu::TextureId) { return nullptr; }
  virtual void* nativeBuffer(vpgpu::BufferId) { return nullptr; }
  // Blocks until all submitted GPU work finished (vkQueueWaitIdle / vkDeviceWaitIdle / fences).
  virtual void waitIdle() {}
};

// Defined by exactly one platform source linked into the dylib.
std::unique_ptr<Platform> createPlatform();

}  // namespace vpvk
