// Command buffer recording as a deferred list, and its translation into a gpu/core Frame.
// vkCmd* only append; nothing touches the GPU until vkQueueSubmit translates the list, so the core's
// frame compiler sees the whole submission and places barriers only where a real hazard exists.
// Vulkan pipeline barriers are counted (diagnostics) but never turned into GPU barriers.
//
// Draws and dispatches are resolved when they are recorded: the bound pipeline, descriptor sets,
// push constants and vertex/index buffers become a ready vpgpu::Draw / Dispatch (Vulkan forbids
// changing what a recorded command buffer references until it has executed).
#pragma once

#include <array>
#include <memory>
#include <variant>
#include <vector>

#include "vpvk.h"

namespace vpvk {

struct OpBeginPass {
  const RenderPassObj* pass = nullptr;
  const Framebuffer* framebuffer = nullptr;
  std::vector<VkClearValue> clears;
  VkRect2D area{};
};
// Dynamic rendering: the core pass is built when recorded.
struct ImageView;
// Copy standing in for the resolve of a forced-1x MSAA attachment (commands.cpp).
vpgpu::Copy forcedResolveCopy(const ImageView* src, const ImageView* dst, uint16_t layers);
// A resolve done as that copy instead of a Metal resolve: the game's MSAA was forced to 1x, or the game itself "resolves" a
// single-sample attachment (Source 2 with vr_msaa 0 resolves its 1x depth into a second image). Metal has no 1x resolve:
// a resolve store of a 1x texture is undefined behaviour (the eye passes with that depth hung the GPU).
bool resolveAsCopy(const ImageView* src);
// Whether the stand-in copy can be made: different image, same format and sample count. Logged (once per reason) when not;
// the resolve is then dropped (the source is still stored).
bool resolveCopyUsable(const ImageView* src, const ImageView* dst);
// Integer MSAA attachments: Metal resolves only float formats, so their resolve is a blit of
// sample 0 after the (sub)pass, which stores the samples instead.
bool integerResolve(const ImageView* src);
vpgpu::Copy sampleZeroResolve(const ImageView* src, const ImageView* dst, uint16_t layers);
vpgpu::Format attachmentFormat(const ImageView&);  // colour view format a pass renders as

struct OpBeginRendering {
  vpgpu::RenderPassDesc desc;
  // Game MSAA forced to 1x: copies standing in for the resolves, emitted after the pass.
  std::vector<vpgpu::Copy> resolveCopies;
  // What the VPKTAA jitter needs at translate time (the core desc keeps neither): render area,
  // multiview mask and the depth attachment's size at its mip (0 = no depth attachment).
  VkRect2D area{};
  uint32_t viewMask = 0;
  uint32_t depthWidth = 0, depthHeight = 0;
  // Clears the load ops could not express (render area smaller than an attachment, depth and
  // stencil with different load ops): drawn as quads right after the pass opens.
  std::shared_ptr<const vpgpu::ClearRects> areaClear;
  // vpvk_cmd_pass_drop (a host that learns after the begin which attachments are undefined at the end of the pass):
  // the colour slots (bits) and depth/stencil aspects (bit 0 depth, bit 1 stencil; 0 = none) whose store is a plain
  // Store the host may turn into DontCare. Resolved, forced-Store (forced 1x, integer resolve, areas smaller than the
  // attachment, aspects not attached, eye images) attachments are not in these masks.
  uint32_t dropColor = 0;
  uint8_t dropDepth = 0;
};
struct OpNextSubpass {};
struct OpEndPass {};
// vkCmdPipelineBarrier inside a render pass (a subpass self-dependency): a subpass that reads one
// of its own attachments as an input attachment is split there (commands.cpp), so the reads after
// the barrier see what was rendered (cleared) before it.
struct OpPassBarrier {};
struct OpCopyBuffer {
  const Buffer* src = nullptr;
  const Buffer* dst = nullptr;
  VkBufferCopy region{};
};
// A copy from the command buffer's own upload memory (vkCmdUpdateBuffer, pattern fills).
struct OpUpload {
  vpgpu::BufferId src;
  VkDeviceSize srcOffset = 0;
  const Buffer* dst = nullptr;
  VkDeviceSize dstOffset = 0, size = 0;
};
struct OpFillBuffer {
  const Buffer* dst = nullptr;
  VkDeviceSize offset = 0, size = 0;
  uint32_t data = 0;
};
struct OpClearColor {
  const Image* image = nullptr;
  VkClearColorValue color{};
  VkImageSubresourceRange range{};
};
struct OpClearDepth {
  const Image* image = nullptr;
  VkClearDepthStencilValue value{};
  VkImageSubresourceRange range{};
};
// vkCmdClearAttachments inside a pass: drawn as quads over the rects (core Draw::clear).
struct OpClearAttachments {
  std::vector<VkClearAttachment> attachments;
  std::vector<VkClearRect> rects;
};
struct OpCopyBufferToImage {
  const Buffer* src = nullptr;
  const Image* dst = nullptr;
  VkBufferImageCopy region{};
};
struct OpCopyImageToBuffer {
  const Image* src = nullptr;
  const Buffer* dst = nullptr;
  VkBufferImageCopy region{};
};
struct OpCopyImage {
  const Image* src = nullptr;
  const Image* dst = nullptr;
  VkImageCopy region{};
  bool genMips = false;  // a scaled blit chain over one image's mips: the backend builds the chain
};
// vkCmdResolveImage region: an MSAA colour image averaged into a single-sample one.
struct OpResolveImage {
  const Image* src = nullptr;
  const Image* dst = nullptr;
  VkImageResolve region{};
};
// vkCmdBlitImage region that is not a plain copy (formats differ or it scales): a sampled blit.
struct OpBlitImage {
  const Image* src = nullptr;
  const Image* dst = nullptr;
  VkImageBlit region{};
  VkFilter filter = VK_FILTER_LINEAR;
};
// Why a draw is not drawn: set where OpDraw::ok turns false (pipelines.cpp makeDraw) or at the replay skip (commands.cpp),
// counted per reason in TranslateStats::skippedBy and printed in vk.submit. None in the counters = a skip with no reason set.
enum class SkipReason : uint8_t { None, TranslateTimeout, NoPipeline, PipelineFailed, Unbound, NoIndex, Fan, Scissor, Cull,
                                  Indirect, Pass, Count };
constexpr const char* kSkipReasonNames[] = {"other", "translate_timeout", "no_pipeline", "pipeline_failed", "unbound",
                                            "no_index", "fan", "scissor", "cull", "indirect", "pass"};
static_assert(sizeof kSkipReasonNames / sizeof *kSkipReasonNames == size_t(SkipReason::Count));
struct OpDraw {
  vpgpu::Draw draw;
  // Indirect draws: arguments read from the buffer at submit (unified memory) — or, with gpuIndirect,
  // by the GPU when it gets to the draw (arguments written by compute in the same submission):
  // indirectCount draws from gpuBuffer at gpuOffset + i * indirectStride (the app's buffer, or the
  // patched copy in the command buffer's arena).
  const Buffer* indirect = nullptr;
  VkDeviceSize indirectOffset = 0;
  uint32_t indirectCount = 0, indirectStride = 0;
  bool gpuIndirect = false;
  vpgpu::BufferId gpuBuffer;
  VkDeviceSize gpuOffset = 0;
  uint32_t viewCount = 1;  // instance multiplier of multiview (PipelineObj::instanceViews; 1 when amplified)
  bool ok = true;  // false: the pipeline or a binding was missing (counted as skipped)
  SkipReason reason = SkipReason::None;  // why, when !ok
  std::vector<vpgpu::TextureId> zeroReads;  // bound images that were zero-init pending when recorded
  // VPKTAA jitter eligibility: the pipeline tests or writes depth; a vertex buffer is bound (a
  // fullscreen pass draws a few vertices generated from the vertex id, with none).
  bool depthTested = false, vertexBuffers = false;
  bool fanConverted = false;  // a non-indexed fan drawn as an uploaded index list (still "no index buffer")
  // VPKTAA UI mask classification (taa::DrawClassifier) and what the mask variant is made from.
  bool depthWrite = false, blended = false;
  int8_t depthCmp = 0;  // depth test of the draw: +1 GREATER-like (reversed Z), -1 LESS-like, 0 none / other
  const PipelineObj* pipe = nullptr;
  PipelineObj::TexelVariant* texelVariant = nullptr;
  bool edsVariant = false;  // the pipeline key carries extended-dynamic-state overrides (no UI-mask variant)
  uint8_t passSamples = 0;
  // After an in-pass barrier of a dynamic rendering pass only (VK_KHR_dynamic_rendering_local_read): the attachments of
  // the open pass this draw reads as textures (commands.cpp localReadSplit).
  std::vector<const Image*> images;
};
struct OpDispatch {
  vpgpu::Dispatch dispatch;
  std::vector<vpgpu::TextureId> zeroReads;  // as OpDraw
  const Buffer* indirect = nullptr;  // CPU-read indirect (no GPU path); dispatch.indirect: GPU-read
  VkDeviceSize indirectOffset = 0;
  bool ok = true;
};
// Occlusion queries (pipelines.cpp / commands.cpp): reset fills the counters with 0; end marks the queries pending
// for the submission; copy moves counters (and availability) into a buffer.
struct OpQueryReset {
  QueryPoolObj* pool = nullptr;
  uint32_t first = 0, count = 0;
};
struct OpQueryEnd {
  QueryPoolObj* pool = nullptr;
  uint32_t query = 0;
  bool timestamp = false;  // vkCmdWriteTimestamp: the host writes the time at submission
};
struct OpQueryCopy {
  QueryPoolObj* pool = nullptr;
  uint32_t first = 0, count = 0;
  const Buffer* dst = nullptr;
  VkDeviceSize offset = 0, stride = 0;
  VkQueryResultFlags flags = 0;
  vpgpu::BufferId constants;  // 16 bytes in the command buffer's arena: uint64 0, uint64 1
  VkDeviceSize constantsOffset = 0;
};
struct OpEvent {
  Event* event = nullptr;
  bool set = false;
};
struct Recorder;
struct OpExecute {
  const Recorder* secondary = nullptr;
};

using Op = std::variant<OpBeginPass, OpBeginRendering, OpNextSubpass, OpEndPass, OpCopyBuffer, OpUpload,
                        OpFillBuffer, OpClearColor, OpClearDepth, OpClearAttachments, OpCopyBufferToImage,
                        OpCopyImageToBuffer, OpCopyImage, OpBlitImage, OpResolveImage, OpDraw, OpDispatch, OpExecute,
                        OpQueryReset, OpQueryEnd, OpQueryCopy, OpEvent, OpPassBarrier>;

// Host memory owned by a command buffer for the data it carries (push constants, vkCmdUpdateBuffer):
// 64 KiB chunks aliased by core buffers, reused after a reset.
struct UploadArena {
  struct Chunk {
    uint8_t* host = nullptr;
    vpgpu::BufferId id;
    VkDeviceSize size = 0, used = 0;
  };
  std::vector<Chunk> chunks;
  size_t current = 0;
  // Copies `size` bytes, 256-aligned (constant buffers need 16). Returns the chunk's buffer.
  vpgpu::BufferId put(Device* d, const void* data, VkDeviceSize size, VkDeviceSize& offset);
  void reset() {
    for (auto& c : chunks) c.used = 0;
    current = 0;
  }
  void release(Device* d);
};

// Draw state at the time of a vkCmd*.
struct BoundState {
  const PipelineObj* graphics = nullptr;
  const PipelineObj* compute = nullptr;
  struct Set {
    const DescriptorSetObj* set = nullptr;
    std::vector<uint32_t> dynamicOffsets;
  };
  std::array<Set, 8> graphicsSets{}, computeSets{};
  std::array<uint8_t, 256> push{};
  struct VB {
    const Buffer* buffer = nullptr;
    VkDeviceSize offset = 0;
    VkDeviceSize size = VK_WHOLE_SIZE;  // vkCmdBindVertexBuffers2
    int64_t stride = -1;                 // vkCmdBindVertexBuffers2 (dynamic strides); -1 = the pipeline's
  };
  std::array<VB, 16> vertex{};
  const Buffer* index = nullptr;
  VkDeviceSize indexOffset = 0;
  bool indexWide = false;
  bool hasViewport = false, hasScissor = false;
  VkViewport viewport{};
  VkRect2D scissor{};
  uint32_t stencilRef[2] = {0, 0};  // front, back
  // Active occlusion query (vkCmdBeginQuery): draws count into its visibility slot.
  QueryPoolObj* queryPool = nullptr;
  uint32_t query = 0;
  // Dynamic stencil compare / write masks (front, back); -1 = the pipeline's own.
  int32_t stencilCompareMask[2] = {-1, -1}, stencilWriteMask[2] = {-1, -1};
  float blend[4] = {0, 0, 0, 0};
  float depthBias[3] = {0, 0, 0};  // constant, clamp, slope (vkCmdSetDepthBias order)
  // VK_EXT_extended_dynamic_state(2) (core in 1.3): what the vkCmdSet* calls left, -1 = never set.
  // A draw takes a value only when the pipeline lists that state as dynamic.
  struct Eds {
    int32_t cullMode = -1, frontFace = -1, topology = -1, depthCompare = -1;
    int8_t depthTest = -1, depthWrite = -1, stencilTest = -1, rasterDiscard = -1, depthBias = -1, restart = -1;
    bool stencilOp[2] = {false, false};  // front, back
    VkStencilOp fail[2] = {}, pass[2] = {}, depthFail[2] = {};
    VkCompareOp compare[2] = {};
    // VK_EXT_extended_dynamic_state3 (colour write mask only): per attachment, -1 = never set.
    int8_t colorWrite[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  } eds;
  uint32_t targetWidth = 0, targetHeight = 0;  // render area of the open pass (scissor clamp)
  // Which pass this is, for the skip log: its render area and the framebuffer (render pass) or first colour
  // attachment (dynamic rendering) size.
  VkRect2D renderArea{};
  uint32_t fbWidth = 0, fbHeight = 0, fbLayers = 0;
  uint32_t subpass = 0;
  const RenderPassObj* pass = nullptr;
  const Framebuffer* framebuffer = nullptr;
  uint8_t passSamples = 0;  // effective sample count of the open pass's attachments (0 = unknown)
  // Dynamic rendering: the images attached to the open pass, and whether an in-pass barrier was recorded in it.
  std::vector<const Image*> passImages;
  bool localRead = false;
};

// MSAA of a pass being recorded (pipelines.cpp): the effective count of its attachments (what the
// pipelines drawn into it must use), the game's original count and the first colour attachment's
// size (eye-pass detection, msaaObservePass).
struct PassSamples {
  uint8_t effective = 0, game = 1;
  bool mismatch = false, hasColor = false, hasDepth = false;
  uint32_t w = 0, h = 0;
  void add(const ImageView* v, bool depth);
  void finish(uint32_t viewMask, const char* what);  // logs a mismatch, feeds the detection
};
// Effective sample count of subpass `sub` (0 = no attachments); `observe` feeds the detection.
uint8_t subpassSamples(const RenderPassObj&, const Framebuffer&, uint32_t sub, bool observe = true);
// Pipeline key for a draw into a pass of `passSamples` (variant with that count if it differs).
vpgpu::PipelineKey pipelineFor(Device*, const PipelineObj&, uint8_t passSamples, PipelineObj::TexelVariant* tv = nullptr,
                               const BoundState::Eds* eds = nullptr);

// Asynchronous pipeline creation (pipelines.cpp "compile policy"). resolvePipeline waits (boosted) for the pipeline's
// translation: true when ok / key / bindings are valid. gateFramePipelines runs at vkQueueSubmit on the finished
// Frame: waits a little for pipelines still compiling, then drops the draws that remain without one (never drawn
// with another shader). pipelineSummary: the one-line "vk.boot_summary".
bool resolvePipeline(Device*, const PipelineObj&);
// Per-boot counters of the pipeline machinery (all threads).
struct PipelineCounters {
  std::atomic<uint64_t> stagesCached{0}, stagesTranslated{0}, translateUs{0};
  // Why a stage was translated although a cache exists: the SPIR-V is new, only its options are (another layout,
  // vertex format, view mask), a translator component moved (lower / emit / driver / baseline), MSL gone, or no cache.
  std::atomic<uint64_t> missNewContent{0}, missNewOptions{0}, missLower{0}, missEmit{0}, missDriver{0}, missBaseline{0},
      missNoShader{0}, missUncached{0};
  // Microseconds the threads that call the Vulkan API were blocked by the pipeline machinery.
  std::atomic<uint64_t> blockedCreateUs{0}, blockedRecordUs{0}, blockedGateUs{0};
  std::atomic<uint64_t> pipelinesCreated{0}, pipelinesAsync{0};
  // vk.pso.census: creations whose SPIR-V translation failed (the Metal compile failures are the cache's: totals().failed),
  // and pipelines still not ready when vkCreate*Pipelines gave up waiting (legacy mode, VPVK_PIPELINE_CREATE_WAIT_MS).
  std::atomic<uint64_t> translateFailed{0}, createTimedOut{0};
  std::atomic<uint64_t> recordWaits{0}, recordTimeouts{0};
  std::atomic<uint64_t> gateFrames{0}, gateSkipFrames{0}, gateSkippedDraws{0}, hardWaits{0}, hardTimeouts{0};
  // gpu.amp (ampCensus): draws of multiview pipelines since the last line, amplified or instanced (by PipelineObj::AmpWhy for the
  // latter), and the view-vertices (count x instances, views included) of the instanced ones.
  std::atomic<uint64_t> ampDraws{0}, ampInstDraws{0}, ampInstVerts{0}, ampBy[PipelineObj::AmpWhyCount]{};
};
PipelineCounters& pipelineCounters();
struct GateResult {
  uint32_t draws = 0, skipped = 0, softPending = 0, hardPending = 0;
  double waitedMs = 0, hardWaitedMs = 0;
  // Draws with a pending pipeline / dropped, by destination class (pipelines.cpp DestClass: eye, no_color,
  // transient_rt, persistent_rt, eye_sized_not_marked). Only "eye" is ever soft.
  uint32_t pendingByClass[5] = {}, skippedByClass[5] = {};
};
GateResult gateFramePipelines(Device*, vpgpu::Frame&, uint64_t submitNo);
void pipelineSummary(Device*, bool final);
// "vk.pso.census": one line with how many pipelines the game created and what became of each (pipelines.cpp). Without
// `force`, at most every 10 s and only when a counter moved; `force` (cache flush, device destroy) always prints.
void psoCensus(Device*, bool force);
// The "key=value ..." body of that line.
std::string psoCensusLine(const PipelineCounters&, const vpgpu::PipelineStats&, const Platform::ArchiveCensus&);
// "gpu.amp": multiview draws per 10 s, vertex amplified vs by instancing (and why not amplified); called from the submit tick.
void ampCensus();

struct Recorder {
  Device* device = nullptr;
  std::vector<Op> ops;
  uint32_t barriers = 0;  // vkCmdPipelineBarrier calls dropped (the core derives its own)
  bool inPass = false;
  size_t passBegin = SIZE_MAX;  // index of the open pass's begin op (indirect argument patches go before it)
  bool hasLocalRead = false;    // an in-pass barrier was recorded in a dynamic rendering pass (translator: local read)
  BoundState bound;
  UploadArena arena;
  // VK_KHR_imageless_framebuffer: the framebuffer of each pass begun with its views given at
  // vkCmdBeginRenderPass (VkRenderPassAttachmentBeginInfo); ops point at them until reset.
  std::vector<std::unique_ptr<Framebuffer>> imageless;
  void reset() {
    imageless.clear();
    ops.clear();
    barriers = 0;
    inPass = false;
    passBegin = SIZE_MAX;
    hasLocalRead = false;
    bound = BoundState{};
    arena.reset();
  }
  ~Recorder();
};

struct TranslateStats {
  uint32_t passes = 0, copies = 0, clears = 0, draws = 0, dispatches = 0, skippedDraws = 0,
           skippedDispatches = 0, dropped = 0;
  uint32_t skippedBy[size_t(SkipReason::Count)] = {};  // skippedDraws by SkipReason
  uint32_t selfDependencySplits = 0;  // subpasses split at a self-dependency barrier (OpPassBarrier)
  // vk.perf: draws whose app-side instanceCount is > 1 (indirect ones only when their arguments are read on the CPU), and
  // indirect draws by where the arguments are read: by the GPU, on the CPU at submit (may be stale if the GPU writes
  // them), and of those the ones recorded in a secondary command buffer (multiview/count patches need the primary).
  uint32_t instancedDraws = 0, indirectGpu = 0, indirectCpu = 0, indirectCpuSecondary = 0;
  uint64_t uploadBytes = 0;           // vk.hitch: bytes of the buffer -> image copies of this frame
  uint32_t localReads = 0;            // in-pass barriers of dynamic rendering followed by a draw reading an attachment
  // Foveation: eye passes opened with a rate map, passes split at the first UI draw, logical size of the last one.
  uint32_t fovPasses = 0, fovUiSplits = 0, fovLogicalW = 0, fovLogicalH = 0;
  // VPKTAA jitter: draws moved, draws in an eye pass left alone, largest jittered target (texels).
  uint32_t taaJittered = 0, taaSkipped = 0, taaTargetW = 0, taaTargetH = 0;
  // VPKTAA UI mask: eye-pass draws by class (taa::DrawClassifier), eye passes seen / given the R8 mask.
  taa::UiMaskCounters ui;
  // Query resets / ends in submission order: vkQueueSubmit stamps them with the frame
  // (applyQueryEvents, pipelines.cpp).
  struct QueryEvent {
    QueryPoolObj* pool;
    uint32_t first, count;
    bool end;
  };
  std::vector<QueryEvent> queries;
  std::vector<std::pair<Event*, bool>> events;  // vkCmdSetEvent (true) / vkCmdResetEvent (false)
};
void applyQueryEvents(const std::vector<TranslateStats::QueryEvent>&, uint64_t frame);

// Appends the recorded work to `frame`. Ops whose resources are not bound yet are dropped
// (counted in `dropped`, a vk.err line is the caller's job). With `taa` enabled, the viewport of
// the eye's geometry draws is moved by its sub-pixel jitter (see vpvk_taa_set_jitter).
// With `um` (the device's UiMaskState; null in tests) the UI of an eye pass also marks the R8 mask
// (taa_native.h) and the classifier outlives the call; without it only the classification runs.
// With `zl` the touches of the zero-init pending images are logged in order (see ZeroLog).
// With `fov` (and `taa`: the eye passes are the ones the jitter identifies) the stereo eye passes carry a rate
// map (RenderPassDesc::rateMap, foveation.h) when FoveationState::mapFor gives one.
// With `device` the attachments a dynamic rendering pass reads after an in-pass barrier are copied (VK_KHR_dynamic_rendering_local_read);
// null (tests) leaves those reads on the attachments.
void translate(const Recorder&, vpgpu::Frame& frame, TranslateStats& stats, const TaaJitter* taa = nullptr,
               UiMaskState* um = nullptr, ZeroLog* zl = nullptr, FoveationState* fov = nullptr, Device* device = nullptr);

// vkCmdPipelineBarrier(2) inside a render pass / dynamic rendering (entry.cpp): a marker for the translator.
void recordInPassBarrier(Recorder&);

// VPKTAA UI mask (pipelines.cpp): the key of pipeline `p` (or its texel variant `tv`) for a draw into an eye
// pass whose R8 mask is colour attachment taa::kUiMaskSlot. `ui`: the fragment shader also writes the
// mask (retranslated from the SPIR-V kept by the pipeline); otherwise the slot is bound with write mask
// 0. A UI variant that cannot be translated falls back to the plain one. False: no variant (failed
// pipeline); the draw is then left out of the mask pass logic by the caller.
bool maskPipelineKey(Device*, const PipelineObj& p, PipelineObj::TexelVariant* tv, uint8_t passSamples, bool ui,
                     vpgpu::PipelineKey& out);
// Synchronous pipeline mode: waits once (VPVK_PIPELINE_DRAW_WAIT_MS) for the mask variants maskPipelineKey requested since
// the last call (UiMaskState::freshVariants). submitMutex held; no-op in the asynchronous mode.
void awaitFreshVariants(Device*);

// Records a draw/dispatch with the current state (pipelines.cpp).
size_t recordedOps(VkCommandBuffer);
void setDrawIndexSince(VkCommandBuffer, size_t before, uint32_t index);
OpDraw makeDraw(Recorder&, uint32_t first, uint32_t count, uint32_t instances, uint32_t firstInstance,
                int32_t baseVertex, bool indexed);
OpDispatch makeDispatch(Recorder&, uint32_t x, uint32_t y, uint32_t z);
// One "vk.skip" line (the first 32 of the process) for a draw / dispatch that is not drawn; `fmt` adds details.
void logSkip(const char* why, const PipelineObj* p, const char* fmt = nullptr, ...) __attribute__((format(printf, 3, 4)));

// Exposed for tests.
vpgpu::LoadAction toLoad(VkAttachmentLoadOp);
// A clear colour as Metal takes it (doubles), read from the union as the format says: float32 for
// float/normalized formats, int32 / uint32 for integer ones.
std::array<double, 4> clearColor(const VkClearColorValue&, vpgpu::Format);
// Does `area` (non-empty) leave part of the view's mip uncovered? Load-op clears cover a whole
// attachment in Metal; Vulkan clears only the render area.
bool partialArea(const VkRect2D& area, const ImageView* v);
vpgpu::StoreAction toStore(VkAttachmentStoreOp, bool resolve);
vpgpu::ResolveMode resolveMode(VkResolveModeFlagBits);

}  // namespace vpvk
