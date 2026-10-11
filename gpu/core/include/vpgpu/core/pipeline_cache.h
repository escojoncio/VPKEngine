// Pipeline cache: key -> compiled state, background compilation by priority, on-disk key list.
#pragma once

#include <array>
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

enum class BlendFactor : uint8_t { Zero, One, SrcColor, OneMinusSrcColor, SrcAlpha, OneMinusSrcAlpha,
                                   DstColor, OneMinusDstColor, DstAlpha, OneMinusDstAlpha, Constant };
enum class BlendOp : uint8_t { Add, Subtract, ReverseSubtract, Min, Max };
enum class CullMode : uint8_t { None, Front, Back };

struct BlendState {
  bool enabled = false;
  BlendFactor srcColor = BlendFactor::One, dstColor = BlendFactor::Zero;
  BlendFactor srcAlpha = BlendFactor::One, dstAlpha = BlendFactor::Zero;
  BlendOp colorOp = BlendOp::Add, alphaOp = BlendOp::Add;
  uint8_t writeMask = 0xF;
};

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
  Status status(PipelineKey) const;
  void waitIdle();

  // Pipeline stats since the last call (per-frame counters for the diagnostics report).
  PipelineStats takeFrameStats();
  PipelineStats totals() const;

  // Persist / restore the states seen, so a next run can prefetch them before they are needed.
  bool save(const DiskCachePath&) const;
  size_t prefetch(const DiskCachePath&);  // queues saved states at Prefetch priority

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
  };

  void compile(const PipelineState&);
  void workerLoop();

  Backend& backend_;
  mutable std::mutex mutex_;
  std::condition_variable wake_, idle_;
  std::priority_queue<Job> queue_;
  std::unordered_map<uint64_t, Entry> entries_;
  uint64_t nextOrder_ = 0;
  unsigned busy_ = 0;
  bool stop_ = false;
  PipelineStats frame_, total_;
  std::vector<std::thread> workers_;
};

}  // namespace vpgpu
