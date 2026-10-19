// Pipeline recipe store: everything needed to rebuild a game's pipelines without the game, so
// they can be compiled before gameplay (vkCreateDevice prewarm, vpvk_prewarm from the app).
//
// Two layers per shader stage:
//  * the translator INPUT (StageInput: SPIR-V module by content hash + the lowering options the
//    driver derived from the pipeline layout / vertex input / multiview / spec constants), and
//  * the translator OUTPUT (MSL blob, addressed by its content hash = the shader id the pipeline
//    state names), tagged with the translator version that produced it.
// A pipeline recipe is the keyed PipelineState plus the stage inputs of its shaders. While the
// translator version matches, the prewarm uses the stored MSL directly (no SPIR-V lowering, and the
// key it produces is exactly the key vkCreateGraphicsPipelines computes later). When it does not
// match (kTranslatorVersion bumped), refresh() re-runs the translation from the stored inputs,
// remaps the pipeline states to the new MSL ids and rewrites the file, so translator changes no
// longer throw away the recorded history.
//
// File: <VP_PIPELINE_CACHE_DIR>/recipes.bin, append-only between rewrites, versioned header.
// kRecipeVersion changes only with the FILE / PipelineState encoding; translator changes bump
// kTranslatorVersion instead.
#pragma once

#include <array>
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
#include "vpvk/spirv_frontend.h"

namespace vpvk {

// Bump whenever SPIR-V -> VPIR -> MSL output changes for the same input (stored recipes are then
// retranslated at the next prewarm instead of being discarded).
// 1: per-variable private slots (was recipe file version 2).
// 2: positional MSL names + shaderId.
// 3: 1D images as 2D, texelFetch offsets applied, texture2d_ms (sample index, textureSamples).
// 4: gl_ClipDistance as [[clip_distance]] (was dropped).
// 5: fragment colour outputs padded to 4 components, dual-source index(1).
// 6: dynamic descriptor / interface indexing, gather offsets, packed varyings, texel buffer stores.
// 7: Vulkan CTS round: OpCompositeInsert operand order, packed vertex inputs / colour outputs,
//    depth-compare gradients/bias, exact frexp/ldexp, carry/borrow/mulExtended, clip distance
//    varyings, early_fragment_tests, bool subgroup ops / AllEqual, ballot bit count over word 0,
//    odd-stride and robust (out-of-range = 0) vertex fetch, cube-array storage coords split
//    (face, layer), loop guard kVulkanLoopGuard, subpassInputMS as texture2d_ms, texel buffers in
//    the format qualifier's / bound view's format (sub-word texels shifted), integer colour
//    outputs of the attachment's signedness.
// 8: Adreno baseline: 16-bit integers (shaderInt16), BaseVertex / BaseInstance / DrawIndex
//    (shaderDrawParameters; VpDrawParams {drawIndex, baseVertex, metalBase} at [[buffer(29)]]).
// 9: robustBufferAccess for storage buffers (EmitOptions::robustBuffers: loads/stores/atomics
//    checked against the bound range, BufferSize argument per slot).
// 10: loop nests bounded as a whole (vp_nestN: kNestFactor * loopGuard inner trips), not only per loop.
// 11: 16-bit storage (int16 / half members 2 bytes apart in buffers and push constants).
// 12: Y'CbCr lowering (ycbcr bindings, only behind the CTS gate).
// 13: VPKTAA nativo: fragment shaders de UI con salida extra de máscara (SpirvLowerOptions::uiMaskSlot);
//     el MSL sin máscara no cambia. Recetas: bit 4 de ext = uiMaskSlot/uiMaskAlpha.
inline constexpr uint32_t kTranslatorVersion = 13;
// Loop iterations per invocation before the emitted hang guard breaks out (vpmsl::EmitOptions::loopGuard).
// Vulkan shaders loop legitimately past the GCN default of 8192 (a 16384-element copy in one
// invocation, CTS synchronization.*.buffer_262144); 65536 covers a 1 MiB uvec4 loop and still cuts a
// runaway loop of a small dispatch short of the GPU watchdog.
inline constexpr uint32_t kVulkanLoopGuard = 65536;

// Content hash of a SPIR-V module (never 0).
uint64_t spirvHash(const uint32_t* words, size_t count);

// Everything translateStage feeds the SPIR-V reader for one stage.
struct StageInput {
  uint64_t spirv = 0;  // spirvHash of the module
  spirv::SpirvLowerOptions opt;
  uint64_t hash() const;  // content hash of the whole input (never 0)
};

// What translateStage needs from a translation besides the MSL (pipelines.cpp collectUsed, colour
// write masks, varying check). Stored next to the stage input so a pipeline whose input is already
// recorded skips SPIR-V -> VPIR -> MSL altogether. Bindings carry only set, binding, written and
// texelFormat (all that collectUsed reads).
struct StageIface {
  std::vector<spirv::SpirvBinding> bindings;
  bool usesPushConstants = false;
  uint8_t colorOutputs = 0;     // SpirvInterface::colorOutputs
  uint8_t mslColorOutputs = 0;  // MslShader::colorOutputs
  std::array<uint32_t, 3> localSize{1, 1, 1};
  uint32_t varyingComponents = 0;  // vertex stage, see kMetalMaxVaryingComponents
};

// Translation used by refresh() (pipelines.cpp: same code path as vkCreate*Pipelines, quiet).
bool translateRecipeStage(const spirv::SpirvModule&, const spirv::SpirvLowerOptions&, uint64_t& mslId, ShaderBlob& blob,
                          std::string& error, StageIface* iface = nullptr);
// Number of SPIR-V -> MSL translations this process has run (tests: a recipe hit must not add one).
uint64_t stageTranslations();

class RecipeStore {
 public:
  // dir empty: disabled (every call is a no-op). Loads existing records; a stale/corrupt file is
  // discarded and started over. translatorVersion: tests only.
  explicit RecipeStore(std::string dir, uint32_t translatorVersion = kTranslatorVersion);
  ~RecipeStore();

  bool enabled() const { return file_ != nullptr; }
  // One translated stage: the input (hash = input.hash()), its module words (stored once per
  // module) and the MSL it produced. Cheap when already known.
  // `iface` (optional) is stored too, which is what lets lookupStage answer without translating.
  void addStage(const StageInput& input, uint64_t inputHash, const std::vector<uint32_t>& words, uint64_t mslId,
                const ShaderBlob& blob, const StageIface* iface = nullptr);
  // A recorded translation of this exact input (current translator, MSL and interface both stored):
  // the shader id, the MSL source hash and the interface, with no translation. False: translate.
  struct StageHit {
    uint64_t mslId = 0, sourceHash = 0;
    StageIface iface;
  };
  bool lookupStage(uint64_t inputHash, StageHit& out) const;
  bool shaderBlob(uint64_t mslId, ShaderBlob& out) const;
  // inputs: stage input hashes for vertex, fragment, compute (0 = stage absent).
  void addPipeline(const vpgpu::PipelineState&, const std::array<uint64_t, 3>& inputs);

  // Retranslates every stage input recorded by another translator version (or flagged by a drift
  // record), remaps pipeline states to the new MSL ids and rewrites the file compacted. Also
  // compacts a file with many superseded records. No-op when nothing is stale. Thread-safe.
  struct RefreshResult {
    uint32_t retranslated = 0, failed = 0, droppedPipelines = 0;
    bool rewritten = false;
    double ms = 0;
  };
  RefreshResult refresh(unsigned threads);
  bool needsRefresh() const;

  // Snapshot of what was loaded from disk + added since.
  struct Snapshot {
    std::vector<std::pair<uint64_t, ShaderBlob>> shaders;
    std::vector<vpgpu::PipelineState> pipelines;  // only those whose shaders are all present
  };
  Snapshot snapshot() const;
  std::vector<uint64_t> keys() const;
  // The recipe's state. With `shaders`, also the MSL of each of its stages, and false when one of
  // them is missing (the pipeline could not be built: its shader was never registered).
  bool stateFor(uint64_t key, vpgpu::PipelineState& out,
                std::vector<std::pair<uint64_t, ShaderBlob>>* shaders = nullptr) const;
  size_t pipelineCount() const;
  size_t shaderCount() const;
  // Approximate heap held by the loaded records (MSL sources, stage inputs, pipeline states) plus
  // the file size, for the memory heartbeat. Walks the shader table under the lock.
  uint64_t memoryBytes() const;
  std::vector<StageInput> stageInputs() const;  // diagnostics / tests

  static std::string defaultDir();  // VP_PIPELINE_CACHE_DIR or ""

 private:
  struct InputRec {
    StageInput input;
    uint64_t msl = 0;
    uint32_t translator = 0;
  };
  struct IfaceRec {
    uint32_t translator = 0;
    uint64_t code = 0;  // translator code id (translatorId() in recipes.cpp)
    StageIface iface;
  };
  struct PipeRec {
    vpgpu::PipelineState state;
    std::array<uint64_t, 3> inputs{};
  };
  struct SpirvRec {
    uint64_t offset = 0;  // file offset of the first word
    uint32_t count = 0;   // words
  };

  void load();
  bool openForAppend(bool fresh);
  void write(uint8_t tag, const std::vector<uint8_t>& payload, uint64_t* payloadOffset = nullptr);
  bool readWords(const SpirvRec&, std::vector<uint32_t>& out) const;
  bool remapLocked(PipeRec& p) const;  // shader ids from the current stage inputs
  bool rewriteLocked();
  void insertPipelineLocked(PipeRec p);

  bool compactionDueLocked() const { return deadRecords_ > 256 && deadRecords_ * 4 > records_; }

  std::string path_;
  uint64_t size_ = 0;  // bytes of path_ (0 = no valid file yet)
  uint32_t translator_ = kTranslatorVersion;
  mutable std::mutex mutex_;
  FILE* file_ = nullptr;
  std::unordered_map<uint64_t, ShaderBlob> shaders_;
  std::unordered_map<uint64_t, InputRec> inputs_;
  std::unordered_map<uint64_t, IfaceRec> ifaces_;  // by input hash ('F' records)
  std::unordered_map<uint64_t, SpirvRec> spirv_;
  std::unordered_map<uint64_t, PipeRec> pipelines_;
  size_t storedThisRun_ = 0, records_ = 0, deadRecords_ = 0;
  bool drift_ = false;        // a run saw the same input give another MSL under the same version
  bool driftWritten_ = false; // drift record appended this run
};

// Compiles every stored recipe on `device`'s pipeline workers and waits for them. Retranslates
// stale stage inputs first (RecipeStore::refresh, on all cores).
// progress(done, total) is called from the calling thread (may be null). Logs gpu.prewarm.
struct PrewarmResult {
  uint32_t recipes = 0, compiled = 0, hits = 0, failed = 0, retranslated = 0;
  double ms = 0;
};
// precompile.cpp: ahead-of-time build into the Metal binary archive (VPVK_AOT=1, see aotBuild).
bool aotEnabled();
struct AotResult {
  uint32_t modules = 0, libs = 0, recipes = 0, computePipelines = 0, failed = 0;
  double ms = 0;
};
AotResult aotBuild(const std::string& spirvDir, const std::function<void(uint32_t, uint32_t)>& progress);

// precompile.cpp: MSL source hashes compiled from recipes (gpu.shader.lib counts them as hits).
void addShaderLibRecipes(const std::vector<uint64_t>& sourceHashes);

PrewarmResult prewarm(Platform&, vpgpu::PipelineCache&, RecipeStore&,
                      const std::function<void(uint32_t, uint32_t)>& progress, double maxSeconds = 0);

}  // namespace vpvk
