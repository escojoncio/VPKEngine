// Metal backend for gpu/core. Plain C++ header (the Objective-C++ lives in the .mm), so clients and
// tests stay ordinary C++.
//
// Binding model (what MSL shaders see), the contract of gpu/shader/msl (vpmsl/msl.h):
//   Each draw/dispatch gets an argument table at [[buffer(argumentBufferIndex)]]: one 8-byte entry
//   per binding of Draw::bindings / Dispatch::bindings, sorted by (bindKind(), Access::slot) in
//   BindKind order Buffer, ConstantBuffer, Texture, Sampler, StorageTexture (stable for equal keys).
//   Buffers are their GPU address plus the range's begin; textures and samplers their
//   MTLResourceID. Pixel-local bindings get no entry: the fragment shader reads them as
//   [[color(n)]], n = fetchIndex() of the binding (see PipelineState::fetch). No vertex
//   descriptors: vertex shaders fetch by [[vertex_id]], as guest fetch shaders do.
//
// Tile memory: attachments the frame compiler marked TileLocal are rendered into memoryless
// textures (cached per format/size/samples) and never stored. Render-target textures that are not
// persistent get their memory lazily, on first use that needs it, so a G-buffer that only ever
// lives on tile never allocates RAM.
//
// Pipelines of every pass in one render encoder must declare the formats of all the encoder's
// attachments, including those a fetch merge brought in (Metal requires it per draw).
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "vpgpu/core/backend.h"
#include "vpgpu/core/pipeline_cache.h"

namespace vpgpu::metal {

// One shader as the MSL emitter hands it over.
struct MslShader {
  std::string source;                         // complete MSL translation unit
  std::string entry;                          // function name inside it
  uint32_t argumentBufferIndex = 30;          // [[buffer(n)]] of the argument table
  std::array<uint32_t, 3> threadsPerGroup{1, 1, 1};  // compute only: fixed in the kernel
};

// Resolves PipelineState shader ids to MSL. Called from pipeline worker threads: must be thread-safe.
class ShaderProvider {
 public:
  virtual ~ShaderProvider() = default;
  virtual bool find(uint64_t id, MslShader& out) const = 0;
  // Fragment shader for a pipeline with framebuffer-fetch inputs: a translator emits reads of
  // VPIR texture slot fetch.textureSlot[i] as [[color(i)]]. Default: the shader as found.
  virtual bool findFetch(uint64_t id, const FetchInputs&, MslShader& out) const { return find(id, out); }
};

// Mutex-guarded id -> shader map; enough for tests and for a client that translates up front.
class ShaderTable final : public ShaderProvider {
 public:
  void add(uint64_t id, MslShader shader);
  bool find(uint64_t id, MslShader& out) const override;

 private:
  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, MslShader> shaders_;
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

  void createBuffer(BufferId, const BufferDesc&) override;
  void createTexture(TextureId, const TextureDesc&) override;
  void createSampler(SamplerId, const SamplerDesc&) override;
  void destroyBuffer(BufferId) override;
  void destroyTexture(TextureId) override;
  void destroySampler(SamplerId) override;
  bool compilePipeline(const PipelineState&, std::string& error) override;
  void submit(uint64_t frameIndex, const Frame&, const CompiledFrame&) override;
  bool takeTimings(uint64_t& frameIndex, GpuTimings& out) override;

  // Blocks until every submitted frame has finished on the GPU.
  void waitIdle();
  // CPU pointer to a buffer's first byte (shared storage), or null.
  void* contents(BufferId) const;
  // Writes the pipeline binary archive under cacheRoot; false (and a gpu.err line) on failure.
  bool saveArchive();

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace vpgpu::metal
