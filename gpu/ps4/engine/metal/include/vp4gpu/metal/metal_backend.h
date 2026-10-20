// Metal backend for gpu/core. Plain C++ header (the Objective-C++ lives in the .mm), so clients and
// tests stay ordinary C++.
//
// Binding model (what MSL shaders see), the contract of gpu/shader/msl (vp4msl/msl.h):
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
#include <utility>
#include <vector>

#include "vp4gpu/core/backend.h"
#include "vp4gpu/core/pipeline_cache.h"

namespace vp4gpu::metal {

// One entry of a stage's argument table: binding (kind, slot) at byte `offset`. From
// vp4msl::Argument: kind = BindKind(ResourceKind + 1) (same order, BindKind starts with Auto).
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
  // VPIR texture slot fetch.textureSlot[i] as [[color(i)]] (vp4ir::toFramebufferFetch); its
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
  bool bc = false;               // supportsBCTextureCompression (macOS yes, visionOS/iOS no)
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
  // Frees the native pipeline on the next submit (see Backend::destroyPipeline); the functions only
  // it used go with it. Compiling the key again before that cancels the drop.
  void destroyPipeline(PipelineKey) override;
  // MTLBuffer.gpuAddress + the guest pointer's offset in its page window; 0 without a residency set
  // (bindless): shaders reach such memory only through raw pointers, which need it resident.
  uint64_t bufferAddress(BufferId) const override;
  bool compilePipeline(const PipelineState&, std::string& error) override;
  void submit(uint64_t frameIndex, const Frame&, const CompiledFrame&) override;
  bool takeTimings(uint64_t& frameIndex, GpuTimings& out) override;
  // Frames below this index have finished (their completion handlers ran). Any thread.
  uint64_t completedFrames() const override;
  // Backend's partial / full fences over waitFrames / waitIdle (true = Done).
  bool waitFramesBelow(uint64_t frames, uint64_t timeoutNs) override;
  bool waitAll(uint64_t timeoutNs) override;

  // GPU waits never block forever. A command buffer that fails with a revoked/ignored submission
  // or a removed device (any non-memoryless failure — hang, page fault — with
  // VPVK_DEVICE_LOST_STRICT=1), or 20 s with work in flight and no command buffer completing,
  // marks the backend LOST
  // (`gpu.err what=device_lost`): new frames are not run on the GPU, waits return Lost, and the
  // stuck command buffers are given up on 2 s later (Vulkan's VK_ERROR_DEVICE_LOST).
  enum class WaitResult { Done, Timeout, Lost };
  // Blocks until every submitted frame (and native command buffer) has finished on the GPU, or
  // `timeoutNs` passes (UINT64_MAX: no timeout, but never past a lost device).
  WaitResult waitIdle(uint64_t timeoutNs = UINT64_MAX);
  // Blocks until every frame with an index below `frames` has finished (completion handler run).
  // timeoutNs 0 polls.
  WaitResult waitFrames(uint64_t frames, uint64_t timeoutNs);
  bool lost() const override;
  // First loss reason, empty while healthy (no in-process recovery: see lostReason in metal_backend.mm).
  std::string lostReason() const;
  // Age in ms of the oldest in-flight frame command buffer (0 when idle): the GPU's backlog. Any thread.
  uint32_t gpuBacklogMs() const;
  // Hang-split cuts (attachments stored and reloaded mid-encoder) in the newest submitted frame. Any thread.
  uint32_t lastFrameHangSplits() const;
  // Client work outside the frame model (VPVulkan's vpvk_native_submit: the VPKTAA resolve). Makes
  // a command buffer on this backend's queue, hands it to `fn` as an unretained id<MTLCommandBuffer>
  // and commits it. GPU order: after every frame submitted before the call (resources are
  // hazard-untracked, so an MTLEvent orders it, not the queue alone) and before every frame
  // submitted after it. Counts as a frame in flight (waitIdle covers it). Same thread rules as
  // submit (callers serialize the two). False if there is no device/queue.
  using NativeEncodeFn = void (*)(void* commandBuffer, void* user);
  bool encodeNative(NativeEncodeFn fn, void* user);
  // PS4 copy only (not in gpu/metal2): encodeNative with a std::function and a completion callback, for the
  // visionOS compositor's frame (AstroVisionPro). `encode` gets the id<MTLCommandBuffer> (unretained) before it is
  // committed; `label` (optional) is appended to the command buffer's label; `completed` (optional) runs on Metal's
  // completion thread once the GPU has finished it (also on failure, which encodeNative already logs). False, with
  // neither callback run, when encodeNative refuses (no device, device lost, no frame slot).
  bool encodeInterop(const std::function<void(void* commandBuffer)>& encode, const char* label = nullptr,
                     std::function<void()> completed = {});
  // CPU pointer to a buffer's first byte (shared storage), or null.
  void* contents(BufferId) const;
  // Native objects for interop (VK_EXT_metal_objects, the visionOS compositor), as unretained
  // (__bridge) pointers: id<MTLDevice> and id<MTLTexture>. nativeTexture forces the lazy allocation
  // of a render target; null for an unknown/stale id or a failed allocation. The texture lives until
  // destroyTexture. Call from the thread that creates resources and submits (not thread-safe).
  // Texture views (VkImageView): another type, format or mip/layer range of `base`, made when first
  // bound (the base may be allocated lazily). Ids live in their own space (index high bit set) and
  // go in Access::view; destroyView releases one.
  struct ViewDesc {
    TextureType type = TextureType::Tex2D;
    bool array = false;   // 2D_ARRAY / CUBE_ARRAY
    Format format = Format::Undefined;
    uint16_t baseMip = 0, mips = 1, baseLayer = 0, layers = 1;
    uint8_t swizzle[4] = {2, 3, 4, 5};  // MTLTextureSwizzle per r,g,b,a; identity by default
  };
  TextureId createView(TextureId base, const ViewDesc&);
  void destroyView(TextureId);
  void* nativeDevice() const;
  void* nativeTexture(TextureId);
  // Writes the pipeline binary archive under cacheRoot; false (and a gpu.err line) on failure.
  bool saveArchive();
  // Compiles the MTLLibrary/MTLFunction of a shader already in the ShaderProvider on a background
  // queue, so pipeline creation later finds it compiled. Deduplicated by shader id (= MSL hash).
  void prewarmShader(uint64_t shaderId);
  // Compiles shader `shaderId`'s library synchronously (same variant as prewarmShader).
  bool compileShader(uint64_t shaderId, std::string& error);

  // Memory and cache sizes for the runtime heartbeat. Cheap and callable from any thread: byte
  // counters are kept as resources come and go. Texture bytes are MTLTexture.allocatedSize by kind
  // (MSAA of any format counts as msaa; memoryless attachments are 0). Buffers: `bufOwn` is what
  // Metal allocated, `bufGuest` the guest memory wrapped no-copy (counted by the guest, not Metal).
  struct MemStats {
    uint64_t deviceAllocated = 0, workingSetMax = 0;  // MTLDevice.currentAllocatedSize / recommendedMaxWorkingSetSize
    uint64_t texAstc = 0, texColor = 0, texDepth = 0, texMsaa = 0;
    uint64_t bufOwn = 0, bufGuest = 0;
    uint32_t pipelines = 0, functions = 0, libraries = 0;  // one MTLLibrary per function
    uint32_t archiveEntries = 0;    // session archive descriptors still held (last value if busy)
    uint32_t archiveFiles = 0;      // .metalar files opened for lookups
    uint32_t archiveKeyed = 0;      // of those, files with a key index (looked up directly)
    uint32_t archiveEvicted = 0;    // files deleted at start for exceeding the file cap
    uint32_t archiveHits = 0, archiveMisses = 0;  // pipelines found in / missing from the archives (cumulative)
    uint64_t libCompiled = 0;       // libraries compiled this process (cumulative)
    uint64_t tileRejected = 0;      // attachments that could not stay tile-local (cumulative)
    uint64_t msaaMemoryless = 0;    // MSAA colour targets with no memory (tile only): size had they been resident
  };
  MemStats memStats() const;
  // Memory warning (level 1) / critical (2); any thread. Frees what is rebuildable (see metal_backend.mm), appends
  // "key=MiB ..." to `report`, returns bytes freed immediately (submit-thread work is queued, reported as *_queued).
  size_t trimMemory(int level, std::string& report);
  // Releases the objects the guest destroyed that otherwise wait for the next frame to take a slot (they are only
  // drained there, so a long load without frames pins them). Safe only when nothing is in flight: every frame slot
  // idle, and no frame took one for `minIdleMs`; otherwise 0 and nothing happens. NOT any-thread: same serialization
  // as submit/destroyTexture (VPVulkan calls it under its submit mutex, try-lock). Never waits. Bytes released.
  size_t drainRetired(uint32_t minIdleMs, std::string& report);
  // Cumulative counters (key, value) of what the backend created and encoded, for the per-second diagnostics; keys
  // starting "g_" are gauges. Any thread, lock-free.
  void diagCounters(std::vector<std::pair<const char*, uint64_t>>& out) const;
  // CPU-visible (address, length) of the buffers the backend allocated itself (shared buffers, argument rings), to tell
  // whose a region of the process is. Any thread.
  void diagRanges(std::vector<std::pair<uintptr_t, uint64_t>>& out) const;

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace vp4gpu::metal
