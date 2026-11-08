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
#include <atomic>
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

// ---- Translator versions (what makes a cached stage translation stale) ----------------------------------
// A cached stage ('F' record + recipe input) answers "this SPIR-V with these lowering options gives this MSL". It is
// valid while the code that produced it still produces the same text. That is decided by explicit, per-component
// versions, NOT by hashing the translator's sources: a hash of the sources changes with every comment, refactor or
// fix that leaves the MSL alone, and each change threw away thousands of valid translations (AC Nexus: 2432 stages
// translated again on the game thread, 4.1 s, after a change that altered no output).
//
// Bump a component ONLY when it changes the MSL text (or the interface the driver reads back with it) of some input
// that already exists. Never for: comments, renames, refactors, new diagnostics, speed-ups, support for input that
// used to fail (nothing cached for it), new options that default to the old output.
//  * kLowerVersion  (gpu/shader/spirv, gpu/shader/ir): SPIR-V -> VPIR lowering and the VPIR passes.
//  * kEmitVersion   (gpu/shader/msl): VPIR -> MSL text.
//  * kDriverVersion (gpu/vulkan/src/pipelines.cpp lowerAndEmit / translateStage): the EmitOptions it passes
//    (positionalNames, loopGuard, robustBuffers = VPVK_ROBUST, hashed in StageInput), kVulkanLoopGuard, how the lowering options are derived from the
//    pipeline, the StageIface it keeps. Also the encoding of StageInput that changes which options are hashed.
// The test `translator_corpus` (vk_tests.cpp) hashes the MSL of every SPIR-V module of gpu/shader/spirv/tests and
// every golden of gpu/shader/msl/tests; when it fails, the output changed: bump what applies and copy the new
// digest into kExpectedCorpusDigest in the same commit. Metal-side compile options are a separate version
// (kMslCompileVersion in gpu/metal2/src/metal_backend.mm).
// An MSL text that comes out identical after a bump is still the same shader: ids and pipeline keys are content hashes
// of the MSL, so the Metal binary archive keeps serving it (RecipeStore::refresh counts those: `same_msl`).
//
// History before the split (kTranslatorBaseline): 1 per-variable private slots; 2 positional MSL names + shaderId;
// 3 1D images as 2D, texelFetch offsets, texture2d_ms; 4 gl_ClipDistance; 5 fragment colour outputs padded to 4,
// dual-source; 6 dynamic descriptor / interface indexing, gather offsets, packed varyings, texel buffer stores;
// 7 Vulkan CTS round (composite insert order, packed vertex inputs, depth-compare gradients, frexp/ldexp, carry/borrow,
// clip distance varyings, early fragment tests, subgroup bool ops, odd-stride + robust vertex fetch, cube-array
// storage, loop guard, subpassInputMS, texel buffer formats, integer colour outputs); 8 Adreno baseline (shaderInt16,
// BaseVertex/BaseInstance/DrawIndex); 9 robustBufferAccess; 10 loop nests bounded as a whole; 11 16-bit storage;
// 12 Y'CbCr lowering; 13 VPKTAA nativo: UI fragment shaders with an extra mask output (uiMaskSlot).
inline constexpr uint32_t kTranslatorBaseline = 13;
inline constexpr uint32_t kLowerVersion = 13;  // 13: every vertex position is Invariant (invariantPos, VPVK_INVARIANT_POS); 12: float -> integer conversions skip the NaN/overflow selects when the operand range is bounded; 11: OpGroupNonUniform{I,F}{Add,Mul} / {S,U,F}{Min,Max} / Bitwise / Logical (subgroup_arith); 10: OpGroupNonUniformShuffleXor / Up / Down; 9: ldexp exponents beyond the float range (three-way split), Output variable initializers, memory pointers returned by functions, image memory barriers fence the device; 8: gl_Subgroup{Eq,Ge,Gt,Le,Lt}Mask; 7: narrow-int vector <-> word bitcasts, OpSpecConstantOp ShiftRightArithmetic; 6: 16-bit float math (constants in binary16, hyperbolics / exp / log / atan / frexp / ldexp / cross / reflect / determinant in 32 bits); 5: float -> integer conversions saturate (NaN = 0); 4: integer dot product, zero-initialised workgroup memory; 1: Invariant position, OpKill leaves loops, storage image access from use; 2: killExit on by default; 3: row-major matrices load one vector per memory row, one killExit check for a chain of kills
inline constexpr uint32_t kEmitVersion = 12;  // 12: Mix as fma(y, a, fma(x, -a, x)) (exact at a = 0 and 1), NMin / NMax / NClamp as precise::fmin / fmax; 11: sampler mipLodBias added in the shader (SamplerBias argument per sampled sampler); 10: subgroup ballot / active mask / read-first as simd_or of one bit per lane (simd_ballot and simd_active_threads_mask are merged across divergent branches; simd_broadcast_first hung a fragment shader); 9: subgroup_arith (simd_sum / prefix builtins, lane loop for the other scans and clustered reductions); 8: subgroup read-first of vectors lane by lane; 7: integer varyings under 4 wide cross VS/mesh -> FS as 4-wide; 6: subgroup read-first from the active mask (simd_shuffle of the lowest active lane); 5: storage cubes are texture2d_array (slice = layer * 6 + face); 4: robust storage-buffer bounds count the range rounded up to a word; 1: constant-only private arrays as program-scope `constant` tables; 2: no loop guard on loops that cannot repeat (switches, inlined calls); 3: vertex amplification view = amplification_id + vp_view_base (buffer 27)
inline constexpr uint32_t kDriverVersion = 1;  // 1: StageIface keeps dualSource (Src1 blend factors need it)
// What recipes.bin stores per record (`translator`): equal to 13 while every component is at 0, so files written
// before the split stay valid. Any component bump changes it.
inline constexpr uint32_t kTranslatorVersion = kTranslatorBaseline + kLowerVersion + kEmitVersion + kDriverVersion;
// The components themselves, packed (baseline | lower << 16 | emit << 32 | driver << 48): stored in 'F' records so a
// miss can say WHICH component moved. Bits of the mask translatorDiff() returns.
inline constexpr uint64_t kTranslatorCode = uint64_t(kTranslatorBaseline) | uint64_t(kLowerVersion) << 16 |
                                            uint64_t(kEmitVersion) << 32 | uint64_t(kDriverVersion) << 48;
enum : uint8_t { kDiffBaseline = 1, kDiffLower = 2, kDiffEmit = 4, kDiffDriver = 8 };
uint8_t translatorDiff(uint64_t storedCode, uint64_t currentCode = kTranslatorCode);
// "b13.l1.e1.d1": the versions in logs and in the manifests of the AOT index.
std::string translatorTag(uint64_t code = kTranslatorCode);
// Loop iterations per invocation before the emitted hang guard breaks out (vpmsl::EmitOptions::loopGuard).
// Vulkan shaders loop legitimately past the GCN default of 8192 (a 16384-element copy in one
// invocation, CTS synchronization.*.buffer_262144); 65536 covers a 1 MiB uvec4 loop and still cuts a
// runaway loop of a small dispatch short of the GPU watchdog.
inline constexpr uint32_t kVulkanLoopGuard = 65536;

// Translator modes, read from the environment once per process (bits; every lowering applies them):
//  bit 0 nativeFetch  VPVK_FETCH_NATIVE (default 0): half vertex attributes through one half2 conversion.
//  bit 1 amplify      VPVK_AMPLIFY (default 0): two-view multiview through Metal vertex amplification.
//  bit 2 unpackFetch  VPVK_FETCH_NATIVE & 2 (VPVK_FETCH_NATIVE is a mask, 3 = both): unorm/snorm vertex attributes of whole
//                     words through MSL's unpack_*_to_float.
//  bit 3 killExitOff  VPSPV_KILL_EXIT=0 (default 1 = on, so the default mode stays 0): without it a fragment shader keeps
//                     running after every lane of its SIMD-group is killed.
//  bit 4 invariantPosOff  VPVK_INVARIANT_POS=0 / off / false (default on; any other value leaves it on): without it only the
//                     positions the shader decorates Invariant (GLSL `invariant gl_Position`, #pragma STDGL invariant(all))
//                     are [[position, invariant]].
// They change the MSL of the stored inputs without being part of them (SpirvLowerOptions::amplify / nativeFetch are not in
// the input's hash or encoding), so they travel with the translator id instead.
uint32_t translatorMode();
// Re-reads the environment (the host may set it after an earlier prewarm in the same process): called where a
// device or a prewarm starts, so the mode is fixed for its lifetime. Returns the mode.
uint32_t refreshTranslatorMode();
// What the recipe store records with every translated input: kTranslatorVersion | modes << 16. A mode
// change is then a translator change: the stored recipes are retranslated (refresh) instead of being
// compiled in the other mode's shape (all modes off = the plain kTranslatorVersion of earlier builds).
uint32_t translatorId();
void applyTranslatorMode(spirv::SpirvLowerOptions&);

// Content hash of a SPIR-V module (never 0).
uint64_t spirvHash(const uint32_t* words, size_t count);
// Does the module declare a multisampled image type (OpTypeImage with MS = 1)? Only then can
// SpirvLowerOptions::multisampleAsSingle change its translation, so only then does it belong to the stage key
// (toggling MSAA in the app must not retranslate the shaders that never see a multisample image).
bool spirvUsesMultisample(const uint32_t* words, size_t count);

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
  bool amplified = false;          // vertex stage: multiview through vertex amplification (SpirvInterface::amplified)
  bool dualSource = false;         // fragment stage: writes [[color(0), index(1)]] (MslShader::dualSource)
};

// Second-source blend factors need [[color(0), index(1)]] in the fragment shader and only apply to attachment 0
// (Metal rejects the pipeline otherwise): without them they fall back to the first source. Applied by
// createGraphics and by the recipe store, so a prewarmed pipeline has the key the runtime computes.
void degradeSrc1Blend(vpgpu::PipelineState& st, bool dualSource);

// The variants of a pipeline the runtime asks for under forced MSAA, derived in one place so the prewarm and the runtime
// (pipelines.cpp pipelineFor / maskPipelineKey) get the same state and therefore the same key. msaaVariantState: the
// pipeline with the pass's sample count ('.msN'). applyUiMaskTarget: the R8 mask target at taa::kUiMaskSlot on a mask
// variant ('.ums': the draw leaves it untouched; writes: '.umu', the fragment shader also writes it).
vpgpu::PipelineState msaaVariantState(const vpgpu::PipelineState& base, uint8_t samples);
void applyUiMaskTarget(vpgpu::PipelineState& st, bool writes);
// What the runtime would add to `s` (a recipe) with forced MSAA of `n` (>= 2) samples, appended to `out`: its '.msN' variant,
// and with `masks` (VPKTAA UI mask on) its '.ums' variant; a recipe that already is a mask variant ('.ums' / '.umu') comes
// again with n samples. Only pipelines of a pass forceMsaa can serve (colour 0 + depth, nothing else) and below n samples.
void forcedMsaaVariants(const vpgpu::PipelineState& s, int n, bool masks, std::vector<vpgpu::PipelineState>& out);

// Translation used by refresh() (pipelines.cpp: same code path as vkCreate*Pipelines, quiet).
bool translateRecipeStage(const spirv::SpirvModule&, const spirv::SpirvLowerOptions&, uint64_t& mslId, ShaderBlob& blob,
                          std::string& error, StageIface* iface = nullptr);
// Number of SPIR-V -> MSL translations this process has run (tests: a recipe hit must not add one).
uint64_t stageTranslations();

class RecipeStore {
 public:
  // dir empty: disabled (every call is a no-op). Loads existing records; a stale/corrupt file is
  // discarded and started over. translatorVersion: tests only.
  // translatorVersion / translatorCode: tests only (simulate a bump of the translator).
  explicit RecipeStore(std::string dir, uint32_t translatorVersion = translatorId(),
                       uint64_t translatorCode = kTranslatorCode);
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
  // Why lookupStage said no (aggregated by the driver into the per-boot summary): the store is off / drifted,
  // the SPIR-V module is new, the module is known but not with these options, the record was made by another
  // translator (`components`: kDiff* bits of what moved), or its MSL is gone.
  enum class StageMiss : uint8_t { None, Disabled, NewContent, NewOptions, Version, NoShader };
  struct MissInfo {
    StageMiss why = StageMiss::None;
    uint8_t components = 0;
  };
  // `spirv`: spirvHash of the module (0 = unknown), only to tell NewOptions from NewContent.
  bool lookupStage(uint64_t inputHash, StageHit& out, MissInfo* miss = nullptr, uint64_t spirv = 0) const;
  bool shaderBlob(uint64_t mslId, ShaderBlob& out) const;
  // inputs: stage input hashes for vertex, fragment, compute (0 = stage absent).
  void addPipeline(const vpgpu::PipelineState&, const std::array<uint64_t, 3>& inputs);

  // Retranslates every stage input recorded by another translator version (or flagged by a drift
  // record), remaps pipeline states to the new MSL ids and rewrites the file compacted. Also
  // compacts a file with many superseded records. No-op when nothing is stale. Thread-safe.
  struct RefreshResult {
    uint32_t retranslated = 0, failed = 0, droppedPipelines = 0;
    uint32_t sameMsl = 0;  // retranslated inputs whose MSL (so shader id and pipeline keys) did not change
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
  // Identity of a recipe that survives a translator change. The pipeline key hashes the shader ids (MSL content hashes), so it
  // changes whenever refresh() retranslates to other MSL; the identity hashes the pipeline state without its shader ids plus
  // the stage inputs (SPIR-V + lowering options) the shaders come from, which no translator change moves. It is what
  // vkGetPipelineCacheData hands the game: the blob of an earlier launch still names the recipes afterwards.
  static uint64_t recipeId(const vpgpu::PipelineState&, const std::array<uint64_t, 3>& inputs);
  std::vector<uint64_t> ids() const;  // recipeId of every stored recipe
  // The pipeline key of the recipe `idOrKey` names: a recipeId, or (blobs written before the identity existed) a pipeline key.
  bool resolve(uint64_t idOrKey, uint64_t& key) const;
  // The recipe's state. With `shaders`, also the MSL of each of its stages, and false when one of
  // them is missing (the pipeline could not be built: its shader was never registered).
  bool stateFor(uint64_t key, vpgpu::PipelineState& out,
                std::vector<std::pair<uint64_t, ShaderBlob>>* shaders = nullptr) const;
  bool has(uint64_t key) const;  // a recipe with this pipeline key is stored
  size_t pipelineCount() const;
  size_t storedThisRun() const;  // recipes this process added (the `saved` of gles.pso.recipe)
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
  uint64_t code_ = kTranslatorCode;
  mutable std::mutex mutex_;
  FILE* file_ = nullptr;
  std::unordered_map<uint64_t, ShaderBlob> shaders_;
  // Shaders whose MSL text is not in shaders_[id].source but only in recipes.bin (file offset, length, source hash):
  // read back on demand (fillSource), so the store holds no second copy of the MSL.
  struct SrcRef {
    uint64_t off = 0;
    uint32_t len = 0;
    uint64_t hash = 0;
  };
  std::unordered_map<uint64_t, SrcRef> srcRef_;
  bool fillSource(uint64_t id, ShaderBlob&, FILE* f = nullptr) const;
  std::unordered_map<uint64_t, InputRec> inputs_;
  std::unordered_map<uint64_t, IfaceRec> ifaces_;  // by input hash ('F' records)
  std::unordered_map<uint64_t, SpirvRec> spirv_;
  std::unordered_map<uint64_t, PipeRec> pipelines_;
  std::unordered_map<uint64_t, uint64_t> byId_;  // recipeId -> key in pipelines_
  size_t storedThisRun_ = 0, records_ = 0, deadRecords_ = 0;
  bool drift_ = false;        // a run saw the same input give another MSL under the same version
  bool driftWritten_ = false; // drift record appended this run
};

// Compiles every stored recipe on `device`'s pipeline workers and waits for them. Retranslates
// stale stage inputs first (RecipeStore::refresh, on all cores).
// progress(done, total) is called from the calling thread (may be null). Logs gpu.prewarm.
struct PrewarmResult {
  uint32_t recipes = 0, compiled = 0, hits = 0, failed = 0, retranslated = 0;
  uint32_t ready = 0;  // recipes whose pipeline was ready (or already released by the game) when the prewarm ended
  uint32_t variants = 0, variantsFailed = 0;  // forced MSAA: variants added to the recipes (forcedMsaaVariants), and those that failed
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

// Background prewarm (vkCreateDevice with async pipelines): `cancel` ends it early, `addShader` replaces
// Platform::addShader (the device registers every shader once, under its shader mutex, shared with the translate
// jobs), `refreshThreads` bounds the retranslation of stale recipes (0 = every core), `pollMs` is how often it looks
// at the compiles. The pipelines are requested at Prefetch priority: whatever the game asks for goes first.
constexpr uint32_t kPrewarmMaxVariants = 3072;  // forced-MSAA variants a prewarm compiles at most (they stay resident, ownerless)
struct PrewarmHooks {
  const std::atomic<bool>* cancel = nullptr;
  std::function<void(uint64_t, ShaderBlob)> addShader;
  unsigned refreshThreads = 0;
  uint32_t pollMs = 15;
  uint32_t maxVariants = 0;  // cap of forced-MSAA variants (0 = kPrewarmMaxVariants)
  std::function<void()> afterRefresh;  // called once the retranslation is done (the store's keys are final for this run)
};
PrewarmResult prewarm(Platform&, vpgpu::PipelineCache&, RecipeStore&,
                      const std::function<void(uint32_t, uint32_t)>& progress, double maxSeconds = 0,
                      const PrewarmHooks& hooks = {});

}  // namespace vpvk
