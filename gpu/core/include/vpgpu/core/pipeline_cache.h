// Pipeline cache: key -> compiled state, background compilation by priority, on-disk key list.
#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "vpgpu/core/commands.h"

namespace vpgpu {

class Backend;

// (New factors go last: pipeline recipes store the numbers.)
enum class BlendFactor : uint8_t { Zero, One, SrcColor, OneMinusSrcColor, SrcAlpha, OneMinusSrcAlpha,
                                   DstColor, OneMinusDstColor, DstAlpha, OneMinusDstAlpha, Constant,
                                   OneMinusConstant, ConstantAlpha, OneMinusConstantAlpha, SrcAlphaSaturate,
                                   Src1Color, OneMinusSrc1Color, Src1Alpha, OneMinusSrc1Alpha };
enum class BlendOp : uint8_t { Add, Subtract, ReverseSubtract, Min, Max };
enum class CullMode : uint8_t { None, Front, Back };

struct BlendState {
  bool enabled = false;
  BlendFactor srcColor = BlendFactor::One, dstColor = BlendFactor::Zero;
  BlendFactor srcAlpha = BlendFactor::One, dstAlpha = BlendFactor::Zero;
  BlendOp colorOp = BlendOp::Add, alphaOp = BlendOp::Add;
  uint8_t writeMask = 0xF;
};

enum class StencilOp : uint8_t { Keep, Zero, Replace, IncrementClamp, DecrementClamp, Invert, IncrementWrap,
                                 DecrementWrap };

struct StencilFace {
  StencilOp fail = StencilOp::Keep, depthFail = StencilOp::Keep, pass = StencilOp::Keep;
  CompareOp compare = CompareOp::Always;
  uint8_t readMask = 0xFF, writeMask = 0xFF;
};

// Framebuffer-fetch inputs: colour attachment `i` is read by the fragment shader as VPIR texture
// slot `textureSlot[i]` when bit i of `mask` is set. The shader translator turns reads of that slot
// into [[color(i)]] inputs; `i` is fetchIndex() of the draw's pixel-local binding (the slot the
// texture was rendered through). fetchInputs() below computes it from a draw.
struct FetchInputs {
  uint8_t mask = 0;
  std::array<uint16_t, kMaxColorAttachments> textureSlot{};
};
// Fills from the draw's pixelLocal bindings (Access::slot = VPIR texture slot) against the pass
// that wrote the textures; false if one of them is not a colour attachment there, or two texture
// slots read the same attachment.
bool fetchInputs(const Draw&, const RenderPassDesc& writer, FetchInputs& out);

// Everything a native pipeline object bakes in. Shader ids are content hashes from the shader
// translator; the core does not look inside them.
struct PipelineState {
  uint64_t vertexShader = 0, fragmentShader = 0, computeShader = 0;
  std::array<Format, kMaxColorAttachments> colorFormats{};
  Format depthFormat = Format::Undefined;
  uint8_t samples = 1;
  Primitive primitive = Primitive::Triangles;
  std::array<BlendState, kMaxColorAttachments> blend{};
  bool depthTest = false, depthWrite = false;
  CompareOp depthCompare = CompareOp::Always;
  bool stencilTest = false;  // reference value: Draw::stencilRef
  StencilFace stencilFront, stencilBack;
  FetchInputs fetch;
  CullMode cull = CullMode::None;
  bool alphaToCoverage = false;
  std::string label;

  PipelineKey key() const;  // stable across runs: used as the disk cache key
};

// Lower value runs first: a pipeline needed for the current frame jumps ahead of prefetching.
enum class Priority : uint8_t { Blocking, Visible, Prefetch };

// Where cached pipeline keys/blobs live. The client points it under the VPS4 folder.
struct DiskCachePath {
  std::filesystem::path root;
  std::filesystem::path keyList() const { return root / "pipelines.keys"; }
};

struct PipelineStats {
  uint32_t compiled = 0, failed = 0, hits = 0, misses = 0;
  double totalMs = 0, maxMs = 0;
};

class PipelineCache {
 public:
  enum class Status : uint8_t { Missing, Queued, Compiling, Ready, Failed };

  // workers = 0: compile synchronously inside request() (tests, or a backend without threads).
  PipelineCache(Backend&, unsigned workers);
  ~PipelineCache();
  PipelineCache(const PipelineCache&) = delete;
  PipelineCache& operator=(const PipelineCache&) = delete;

  // Ensures the pipeline is or will be compiled; returns its current status (a hit if Ready).
  Status request(const PipelineState&, Priority);
  // The same with the state's key already computed (key() hashes the whole state).
  Status request(const PipelineState&, PipelineKey, Priority);
  Status status(PipelineKey) const;
  void waitIdle();
  // Blocks until every key is Ready or Failed (or was never requested), or `budget` passes.
  // Returns true when none is still Queued/Compiling. Used to hold a frame rather than draw it
  // with holes where pipelines are still compiling.
  bool waitReady(const std::vector<PipelineKey>& keys, std::chrono::milliseconds budget);

  // Pipeline stats since the last call (per-frame counters for the diagnostics report).
  PipelineStats takeFrameStats();
  PipelineStats totals() const;

  // Persist / restore the states seen, so a next run can prefetch them before they are needed.
  bool save(const DiskCachePath&) const;
  size_t prefetch(const DiskCachePath&);  // queues saved states at Prefetch priority

  // Flat encoding of every keyed field (no label): the recipe store and prefetch share it.
  static void encode(const PipelineState&, std::vector<uint64_t>& out);
  static bool decode(const uint64_t* in, size_t count, PipelineState& out);
  static size_t encodedSize();

 private:
  struct Job {
    Priority priority;
    uint64_t order;
    PipelineState state;
    bool operator<(const Job& o) const {  // std::priority_queue pops the largest
      return priority != o.priority ? priority > o.priority : order > o.order;
    }
  };
  struct Entry {
    PipelineState state;
    Status status = Status::Missing;
    // Failed while prefetching/prewarming: the next real (non-Prefetch) request retries it once,
    // so a speculative compile never poisons the game's own request.
    bool speculativeFailure = false;
  };

  void compile(const PipelineState&, Priority);
  void workerLoop();

  Backend& backend_;
  mutable std::mutex mutex_;
  std::condition_variable wake_, idle_, done_;
  std::priority_queue<Job> queue_;
  std::unordered_map<uint64_t, Entry> entries_;
  uint64_t nextOrder_ = 0;
  unsigned busy_ = 0;
  bool stop_ = false;
  PipelineStats frame_, total_;
  std::vector<std::thread> workers_;
};

}  // namespace vpgpu
