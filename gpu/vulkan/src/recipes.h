// Pipeline recipe store: everything needed to rebuild a game's pipelines without the game, so
// they can be compiled before gameplay (vkCreateDevice prewarm, vpvk_prewarm from the app).
//
// A recipe is the translated shader (MSL blob, addressed by its content hash = the shader id the
// pipeline state names) plus the keyed PipelineState. Storing the translator's output rather than
// SPIR-V means the prewarm skips SPIR-V lowering too, and the pipeline key it produces is exactly
// the key vkCreateGraphicsPipelines computes later (same MSL -> same id -> same key).
// File: <VP_PIPELINE_CACHE_DIR>/recipes.bin, append-only, versioned header (kRecipeVersion must be
// bumped whenever the SPIR-V->MSL translator or PipelineState encoding changes meaning).
#pragma once

#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vpgpu/core/pipeline_cache.h"
#include "vpvk/platform.h"

namespace vpvk {

class RecipeStore {
 public:
  // dir empty: disabled (every call is a no-op). Loads existing records; a stale/corrupt file is
  // discarded and started over.
  explicit RecipeStore(std::string dir);
  ~RecipeStore();

  bool enabled() const { return file_ != nullptr; }
  void addShader(uint64_t id, const ShaderBlob&);
  void addPipeline(const vpgpu::PipelineState&);

  // Snapshot of what was loaded from disk + added since.
  struct Snapshot {
    std::vector<std::pair<uint64_t, ShaderBlob>> shaders;
    std::vector<vpgpu::PipelineState> pipelines;  // only those whose shaders are all present
  };
  Snapshot snapshot() const;
  std::vector<uint64_t> keys() const;
  bool stateFor(uint64_t key, vpgpu::PipelineState& out) const;
  size_t pipelineCount() const;

  static std::string defaultDir();  // VP_PIPELINE_CACHE_DIR or ""

 private:
  void load(const std::string& path);
  void write(uint8_t tag, const std::vector<uint8_t>& payload);

  mutable std::mutex mutex_;
  FILE* file_ = nullptr;
  std::unordered_map<uint64_t, ShaderBlob> shaders_;
  std::unordered_map<uint64_t, vpgpu::PipelineState> pipelines_;
  size_t storedThisRun_ = 0;
};

// Compiles every stored recipe on `device`'s pipeline workers and waits for them.
// progress(done, total) is called from the calling thread (may be null). Logs gpu.prewarm.
struct PrewarmResult {
  uint32_t recipes = 0, compiled = 0, hits = 0, failed = 0;
  double ms = 0;
};
PrewarmResult prewarm(Platform&, vpgpu::PipelineCache&, const RecipeStore&,
                      const std::function<void(uint32_t, uint32_t)>& progress, double maxSeconds = 0);

}  // namespace vpvk
