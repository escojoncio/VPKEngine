// Metal backend for gpu/core. Plain C++ header (the Objective-C++ lives in the .mm), so clients and
// tests stay ordinary C++.
//
// Binding model (what MSL shaders see), the contract of gpu/shader/msl (vpmsl/msl.h):
//   Each stage of a draw/dispatch gets its own argument table at [[buffer(argumentBufferIndex)]],
//   laid out by that stage's MslShader::arguments: each declared (kind, slot) is looked up among the
//   draw's bindings (bindKind(), Access::slot); bindings the stage does not declare are ignored, and
//   a declared argument with no binding skips the draw (gpu.err). Buffers are their GPU address plus
//   the range's begin; textures and samplers their MTLResourceID. Pixel-local bindings are never
//   looked up: the fragment shader reads them as [[color(n)]], n = fetchIndex() of the binding (see
//   PipelineState::fetch), and the draw's fetches must equal the pipeline's or it is skipped. No
//   vertex descriptors: vertex shaders fetch by [[vertex_id]], as guest fetch shaders do.
//
// Tile memory: attachments the frame compiler marked TileLocal are rendered into memoryless
// textures (cached per format/size/samples) and never stored. Render-target textures that are not
// persistent get their memory lazily, on first use that needs it, so a G-buffer that only ever
// lives on tile never allocates RAM.
//
// Pipelines of every pass in one render encoder must declare the formats of all the encoder's
// attachments, including those a fetch merge brought in (Metal requires it per draw). Attachments a
// fragment shader does not write (MslShader::colorOutputs) get write mask None and no blending, so
// declaring a fetched G-buffer format never clobbers it.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "vpgpu/core/backend.h"
#include "vpgpu/core/pipeline_cache.h"

namespace vpgpu::metal {

// One entry of a stage's argument table: binding (kind, slot) at byte `offset`. From
// vpmsl::Argument: kind = BindKind(ResourceKind + 1) (same order, BindKind starts with Auto).
struct ShaderArgument {
  BindKind kind = BindKind::Buffer;
  uint16_t slot = 0;
  uint32_t offset = 0;
};

// One shader as the MSL emitter hands it over.
struct MslShader {
  std::string source;                         // complete MSL translation unit
  std::string entry;                          // function name inside it
  uint32_t argumentBufferIndex = 30;          // [[buffer(n)]] of the argument table
  std::array<uint32_t, 3> threadsPerGroup{1, 1, 1};  // compute only: fixed in the kernel
  std::vector<ShaderArgument> arguments;      // argument table layout; empty = no table
  uint8_t colorOutputs = 0xFF;                // fragment: bit n = writes [[color(n)]]
};

// Resolves PipelineState shader ids to MSL. Called from pipeline worker threads: must be thread-safe.
class ShaderProvider {
 public:
  virtual ~ShaderProvider() = default;
  virtual bool find(uint64_t shaderId, MslShader& out) const = 0;
  // Fragment shader for a pipeline with framebuffer-fetch inputs: a translator emits reads of
  // VPIR texture slot fetch.textureSlot[i] as [[color(i)]] (vpir::toFramebufferFetch); its
  // arguments then no longer list what the rewrite consumed.
  virtual bool findFetch(uint64_t shaderId, const FetchInputs&, MslShader& out) const = 0;
};

// Mutex-guarded id -> shader map; enough for tests and for a client that translates up front.
// Its shaders are final MSL: one that reads [[color(n)]] already declares it, so findFetch returns
// the shader as stored.
class ShaderTable final : public ShaderProvider {
 public:
  void add(uint64_t shaderId, MslShader shader);
  bool find(uint64_t shaderId, MslShader& out) const override;
  bool findFetch(uint64_t shaderId, const FetchInputs&, MslShader& out) const override { return find(shaderId, out); }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, MslShader> shaders_;
};

// What the device offers; logged once on the `gpu.metal` startup line.
struct MetalCaps {
  bool simulator = false;        // visionOS simulator build: Metal runs on the host Mac's GPU
  bool metal3 = false;           // MTLGPUFamilyMetal3
  bool apple4 = false;           // MTLGPUFamilyApple4: memoryless + [[color(n)]] on device
  bool apple7 = false;           // MTLGPUFamilyApple7
  int argumentBuffersTier = 0;   // 1 or 2 (0 = no device)
  bool bindless = false;         // Metal 3 + tier 2: gpuAddress / gpuResourceID argument tables
  bool residencySet = false;     // false: bound resources made resident per encoder (useResource)
};

struct MetalOptions {
  std::filesystem::path cacheRoot;  // binary archive lives here; empty = no disk cache
  unsigned framesInFlight = 3;
};

class MetalBackend final : public Backend {
 public:
  using LogSink = std::function<void(const std::string&)>;

  // Uses the system default device; no window or layer is needed (headless works).
  MetalBackend(const ShaderProvider&, LogSink, MetalOptions = {});
  ~MetalBackend() override;
  MetalBackend(const MetalBackend&) = delete;
  MetalBackend& operator=(const MetalBackend&) = delete;

  bool ok() const;  // false if no Metal device could be created (already logged)
  // Apple-family GPU: memoryless textures and [[color(n)]] framebuffer fetch are available.
  bool tileMemory() const;
  const MetalCaps& caps() const;

  void createBuffer(BufferId, const BufferDesc&) override;
  void createTexture(TextureId, const TextureDesc&) override;
  void createSampler(SamplerId, const SamplerDesc&) override;
  void destroyBuffer(BufferId) override;
  void destroyTexture(TextureId) override;
  void destroySampler(SamplerId) override;
  bool compilePipeline(const PipelineState&, std::string& error) override;
  void submit(uint64_t frameIndex, const Frame&, const CompiledFrame&) override;
  bool takeTimings(uint64_t& frameIndex, GpuTimings& out) override;
  uint64_t completedFrames() const override;

  // Blocks until every submitted frame has finished on the GPU.
  void waitIdle();
  // CPU pointer to a buffer's first byte (shared storage), or null.
  void* contents(BufferId) const;
  // Native objects for interop (VK_EXT_metal_objects, the visionOS compositor), as unretained
  // (__bridge) pointers: id<MTLDevice> and id<MTLTexture>. nativeTexture forces the lazy allocation
  // of a render target; null for an unknown/stale id or a failed allocation. The texture lives until
  // destroyTexture. Call from the thread that creates resources and submits (not thread-safe).
  void* nativeDevice() const;
  void* nativeTexture(TextureId);
  // Encodes interop work (the visionOS compositor's frame, say) in a command buffer of VPEngine's
  // queue, ordered after every frame submitted before it and before every frame after it.
  // `encode` gets the id<MTLCommandBuffer> (unretained); it is committed when `encode` returns.
  // `completed` (optional) runs on Metal's completion thread once the GPU has finished it (also on
  // failure, which is logged as gpu.err what=command_buffer f=interop).
  bool encodeInterop(const std::function<void(void* commandBuffer)>& encode, const char* label = nullptr,
                     std::function<void()> completed = {});
  // Writes the pipeline binary archive under cacheRoot; false (and a gpu.err line) on failure.
  bool saveArchive();

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace vpgpu::metal
