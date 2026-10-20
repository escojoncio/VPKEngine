// What VPVulkan needs from the machine: a gpu/core backend and the native objects behind it.
// visionOS: platform_metal.mm (gpu/metal). Linux tests: the null backend (platform_null.cpp).
#pragma once

#include <array>
#include <memory>
#include <string>
#include <utility>
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
  // +1 on a pointer nativeTexture returned (a CFRetain) / its balancing release; null / no-op where there is none.
  virtual void* retainNative(void*) { return nullptr; }
  virtual void releaseNative(void*) {}
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
  // GPU backlog for vpvk_gpu_backlog_ms / vpvk_gpu_last_frame_chunks (0 where there is no queue).
  virtual uint32_t gpuBacklogMs() const { return 0; }
  virtual uint32_t lastFrameChunks() const { return 0; }
  // vpvk_trim_memory: drop rebuildable GPU-side memory (level 1 warning, 2 critical); bytes freed, `report` a
  // "key=value ..." line. Any thread. 0 where there is nothing to trim.
  virtual size_t trimMemory(int /*level*/, std::string& /*report*/) { return 0; }
  // vpvk_idle_drain: release what the guest destroyed and the backend keeps until a frame reuses a slot, when no
  // frame is in flight and none was submitted for `minIdleMs`. Called with the device's submit mutex held (try-lock).
  // Bytes released; 0 where there is nothing / it is not safe now.
  virtual size_t drainRetired(uint32_t /*minIdleMs*/, std::string& /*report*/) { return 0; }
  // vpvk_diag_line: cumulative backend counters (key, value); keys starting "g_" are gauges. Any thread, lock-free.
  virtual void diagCounters(std::vector<std::pair<const char*, uint64_t>>& /*out*/) const {}
  // vpvk_diag_ranges (kind 1): CPU-visible (address, length) of the buffers the backend allocated itself.
  virtual void diagRanges(std::vector<std::pair<uintptr_t, uint64_t>>& /*out*/) const {}
  // "key=value ..." memory / cache sizes for the heartbeat (vpvk_mem_stats); empty where there are none.
  virtual std::string memStats() { return {}; }
  // "ar_hits=N ar_misses=N ... reason=..." pipeline binary archive lookups of this process (gpu.prewarm); empty without one.
  virtual std::string archiveSummary() { return {}; }
  // vpvk_native_submit: a native command buffer (id<MTLCommandBuffer>, unretained) on the queue of
  // the guest's work, ordered after every frame submitted so far and before the next; `fn` encodes
  // into it, then it is committed. Called with the device's submit mutex held. False: no native
  // queue (null backend).
  virtual bool encodeNative(void (*)(void* commandBuffer, void* user), void*) { return false; }
  // VPKTAA nativo (taa_native_metal.mm): codifica el pase en un command buffer nativo. `jitter` en
  // píxeles del target `targetW x targetH` (0: el color); percent/sharp del ajuste de la app.
  struct TaaNativeArgs {
    const void* frame = nullptr;  // const vpvk_taa_native_frame*
    float jx = 0, jy = 0;
    uint32_t targetW = 0, targetH = 0;
    bool jittered = false;
    float percent = 80;
    // RCAS (taa_native.h): intensidad 0..1 (0 = copia sin afilar; con TAA el pase siempre corre).
    float sharpness = 0.5f;
    // Sin TAA: solo RCAS sobre el color del juego, sin resolve, historia ni jitter (el frame no lleva jitter).
    bool rcasOnly = false;
    bool rcasDenoise = false;
    uint64_t phase = 0;
    // Máscara de UI de este frame por ojo (id<MTLTexture> R8 2D/2DArray del tamaño del pase de ojo y
    // la capa); null = sin UI. Ver gpu/vulkan/src/taa_native.h.
    struct Mask {
      void* texture = nullptr;
      uint32_t slice = 0;
    } mask[2];
  };
  virtual bool taaNativeEncode(const TaaNativeArgs&, void* /*commandBuffer*/) { return false; }
  // Profundidad MSAA -> depth32Float de 1 muestra (vpvk_depth_resolve_encode). Null: sin Metal.
  virtual void* depthResolveEncode(void* /*commandBuffer*/, void* /*depthMS*/, bool /*reversedZ*/, int /*mode*/,
                                   int /*key*/) {
    return nullptr;
  }
};

// Defined by exactly one platform source linked into the dylib.
std::unique_ptr<Platform> createPlatform();

}  // namespace vpvk
