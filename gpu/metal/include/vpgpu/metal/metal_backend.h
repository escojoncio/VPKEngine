// Metal backend for gpu/core. Plain C++ header (the Objective-C++ lives in the .mm), so clients and
// tests stay ordinary C++.
//
// Binding model (what MSL shaders see):
//   Each draw/dispatch gets an argument table at [[buffer(argumentBufferIndex)]]: one 8-byte slot per
//   entry of Draw::bindings / Dispatch::bindings, in order. A buffer slot is its GPU address plus the
//   range's begin; a texture slot is its MTLResourceID. In MSL that is a plain struct of
//   `device T*` and `texture*<T>` members read through `constant Args&`. No vertex descriptors:
//   vertex shaders fetch by [[vertex_id]], as guest fetch shaders do.
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
