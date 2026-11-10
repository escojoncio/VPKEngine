// Turns a recorded Frame into the minimum number of encoders and barriers.
//
// Encoder shape follows Metal 4: render passes become render encoders; dispatches, copies and
// fills share compute encoders. Encoders are expensive (each one is a tile load/store and a
// driver round trip), so the compiler:
//   - merges consecutive render passes with the same attachments into one encoder;
//   - moves copies out of the way of render encoders: hoisted into the earliest compute encoder
//     no hazard forbids, or else sunk to right after the render encoder;
//   - places a barrier only where a real RAW/WAR/WAW hazard exists, per byte range / mip-layer box;
//   - records why every encoder was started, so the log explains the encoder count;
//   - keeps G-buffers on tile (Apple TBDR): a pass that reads attachments of the open encoder only
//     pixel-locally (Access::pixelLocal) merges into it, and an attachment the client released
//     (store DontCare/Resolve, or TextureDesc::transient) whose whole life is that one encoder (or that
//     each encoder that attaches it starts clean: Clear/DontCare, no Load of an earlier one's contents)
//     becomes tile-local: store DontCare, memoryless allocation in the backend.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <utility>
#include <vector>

#include "vpgpu/core/commands.h"

namespace vpgpu {

#define VPGPU_CUT_REASONS(X)                                                            \
  X(FrameStart)       /* first render encoder of the frame */                           \
  X(AttachmentChange) /* render targets differ from the previous render encoder */      \
  X(ClearLoad)        /* same targets, but the pass clears: needs a fresh load action */ \
  X(Feedback)         /* pass reads/writes what the open render encoder touches (not  \
                         counting pixel-local reads of its attachments) */              \
  X(TileBudget)       /* fetch merge would exceed CompileOptions::tileBytesPerPixel */ \
  X(PinnedCopy)       /* a sunk copy must run before this pass and could not move */    \
  X(ComputeBetween)   /* a dispatch separates passes that could otherwise merge */      \
  X(ComputeWork)      /* compute encoder opened for dispatches */                       \
  X(CopyWork)         /* compute encoder opened for copies/fills */                     \
  X(ViewSplit)        /* second encoder of a pass split per view (Encoder::view) */      \
  X(TileSplit)        /* chunk of a heavy all-autoMsaa encoder split at draw boundaries (Encoder::drawBegin) */

enum class CutReason : uint8_t {
#define VPGPU_X(name) name,
  VPGPU_CUT_REASONS(VPGPU_X)
#undef VPGPU_X
      Count
};
const char* toString(CutReason);

enum class HazardKind : uint8_t { RAW, WAR, WAW };
inline constexpr int kHazardKinds = 3;

// Orders all earlier work in `after` stages against later work in `before` stages. It sits
// before encoder `encoder`, item `item` (item 0 = encoder boundary; >0 = inside a compute encoder).
struct Barrier {
  uint32_t encoder = 0, item = 0;
  StageMask after = 0, before = 0;
  uint8_t hazards = 0;  // bit per HazardKind
};

// Why an attachment did or did not stay on tile. Candidates are only attachments the client lets
// go of: store DontCare/Resolve, or TextureDesc::transient. Tile-local then requires every access of
// the texture in the frame to be inside one render encoder (as attachment or pixel-local read; a texture attached by
// several encoders needs each later one to Clear or DontCare it: only a later Load makes it OtherEncoder),
// contents not loaded from memory, not persistent, whole texture covered, and the size/sample count
// of the encoder's other attachments.
#define VPGPU_TILE_VERDICTS(X)                                                           \
  X(None)           /* empty slot, stored by the client, or no resource table */         \
  X(TileLocal)      /* stays on tile: never stored, memoryless */                        \
  X(Persistent)     /* TextureDesc::persistent (history, next-frame reads) */            \
  X(LoadsContents)  /* first pass loads it: contents come from memory */                 \
  X(SampledRead)    /* read without pixelLocal somewhere (offsets, filtering, compute) */ \
  X(OtherEncoder)   /* attachment of more than one render encoder, a later one Loads it */ \
  X(UsedElsewhere)  /* copied, or written outside its encoder */                         \
  X(Partial)        /* attachment covers only some mips/layers of the texture */         \
  X(SizeMismatch)   /* size or sample count differs from the encoder's first attachment */ \
  X(BadFetch)       /* pixel-local read of something that is not a colour attachment */ \
  X(HeavyGeometry)  /* encoder with so much geometry the tiler may flush: memoryless would fail */

enum class TileVerdict : uint8_t {
#define VPGPU_X(name) name,
  VPGPU_TILE_VERDICTS(VPGPU_X)
#undef VPGPU_X
      Count
};
const char* toString(TileVerdict);
inline constexpr int kDepthSlot = kMaxColorAttachments;  // index of depth in Encoder::tile

struct Encoder {
  enum class Kind : uint8_t { Render, Compute } kind = Kind::Render;
  CutReason reason = CutReason::FrameStart;
  RenderPassDesc pass;                // render: load actions of the first pass, stores of the last
  std::vector<uint32_t> commands;     // indices into Frame::commands(), execution order
  uint32_t passes = 0;                // render: client passes merged in
  // render: per attachment slot (colours, then depth). TileLocal slots have store DontCare (or
  // Resolve) in `pass` and may be allocated memoryless by the backend.
  std::array<TileVerdict, kMaxColorAttachments + 1> tile{};
  // render: slots that would stay on tile but the encoder is HeavyGeometry: the backend gives them real memory (never a
  // memoryless stand-in). Their MSAA samples are still not stored when the client only wants the resolve.
  std::array<bool, kMaxColorAttachments + 1> memBacked{};
  // render: draws/vertices of the encoder relative to CompileOptions::tileMaxDraws/tileMaxVertices (1 = at the limit).
  float heavy = 0;
  // render: the draws of the encoder's commands (counted in order across them) this encoder runs: [drawBegin, drawEnd). A heavy
  // encoder whose attachments are all Attachment::autoMsaa is cut into chunks under the tile budget (splitHeavyAuto): every
  // chunk resolves at its end and the next one is seeded from that resolve. Whole encoder = 0 .. UINT32_MAX.
  uint32_t drawBegin = 0, drawEnd = UINT32_MAX;
  // render, CompileOptions::splitViews: -1 = the encoder renders every layer of the pass. 0 / 1 = a two-layer pass of vertex
  // amplified draws rendered as two encoders (same commands in both), this one only layer `view` of every attachment (and of
  // its resolve target), with an amplification count of 1; the backend draws view `view` (vpmsl kViewBaseDecl).
  int8_t view = -1;
  // render, layerCount 2, only while splitViews is on: why the pass was split ("ok") or not (a short reason), and a signature
  // of its attachments (formats, size, samples) for the one-line log per pass kind. splitWhy null: not evaluated.
  const char* splitWhy = nullptr;
  uint64_t splitSig = 0;
  // ... and what one view of it would hold: its vertices, and whether the full heavy test (draws, vertices, indirect draws) still
  // calls that HeavyGeometry (the tile verdict of a split pass; "still heavy" by draws alone does not stop the split).
  uint64_t splitViewVerts = 0;
  bool splitHeavyAfter = false;
  // render, reason Feedback (diagnostic only): the first thing the pass read or wrote that the render encoder it could not join
  // touches. texture null = not found (the conflict was a resolve write). prevDepth: it is that encoder's depth attachment;
  // write: the new pass writes it (else it samples / copies it, a read of what the earlier encoder wrote).
  struct FeedbackInfo {
    TextureId texture;
    bool prevDepth = false, write = false;
  } feedback;
  bool tileLocal(int slot) const { return tile[slot] == TileVerdict::TileLocal; }
};

struct CompileStats {
  uint32_t commands = 0;
  uint32_t passesMerged = 0;   // passes that joined an already open render encoder
  uint32_t copiesHoisted = 0;  // moved before the render encoder they were recorded in
  uint32_t copiesSunk = 0;     // moved after an open render encoder
  uint32_t dropped = 0;        // commands referencing destroyed resources
  uint32_t cuts[static_cast<int>(CutReason::Count)] = {};
  uint32_t hazards[kHazardKinds] = {};
  uint32_t intraBarriers = 0;  // barriers inside a compute encoder
  uint32_t barriersComputeDraw = 0;  // barriers after compute work before vertex/fragment reads (an indirect draw's arguments included)
  uint32_t tileLocal = 0;      // textures kept on tile
  uint32_t tileRejected = 0;   // attachment textures that had to stay in memory
  uint32_t badFetch = 0;       // pixel-local reads that cannot be served: their draws are skipped
  uint64_t tileBytes = 0;      // memory the tile-local textures would have occupied
  uint32_t storesKept = 0;     // DontCare stores turned into Store: a later encoder uses the texture
  // MSAA textures read (sampled, copied, loaded) in this frame before any encoder rendered them in it:
  // their contents come from an earlier submit, which msaaResolveOnTile may have left on tile.
  uint32_t msaaStaleReads = 0;
  uint32_t heavyResolve = 0;   // StoreAndResolve MSAA attachments of HeavyGeometry encoders reduced to Resolve with real memory
  uint32_t heavyIndirect = 0;  // render encoders HeavyGeometry because they hold an indirect draw (GPU-written counts)
  uint64_t maxEncoderVertices = 0;  // the largest vertex count (as CompileOptions::tileMaxVertices judges it) of one render encoder
  uint64_t tileMaxVertices = 0;     // CompileOptions::tileMaxVertices this frame was compiled with
  uint32_t tileRaised = 0;          // encoders not heavy only because CompileOptions::tileMaxVertices is above tileBaseVertices
  uint32_t heavyEncoders = 0;       // render encoders past the heavy limits (draws, vertices or an indirect draw): their attachments get memory
  uint32_t loadsDropped = 0;  // Loads of undefined memory (ResourceTable::memValid) turned into DontCare so the texture stays tile-local
  uint32_t autoMsaaSlots = 0;  // Attachment::autoMsaa slots kept tile-local (never memBacked, never stored)
  uint32_t seedDraws = 0;      // Attachment::needsSeed slots: the backend fills their samples from the 1x at the encoder start
  uint32_t seedMissing = 0;    // autoMsaa Loads with no defined 1x to seed from: they start undefined (DontCare)
  uint32_t tileSplits = 0;     // chunks made by splitHeavyAuto (encoders beyond the first of each split encoder)
  uint32_t tileSplitSkipped = 0;  // heavy all-autoMsaa encoders left whole (the three below: stencil, one chunk, too many chunks)
  uint32_t tileSplitSkipStencil = 0, tileSplitSkipSingle = 0, tileSplitSkipCap = 0;  // ... by reason (their sum is tileSplitSkipped)
  uint32_t heavyObserved = 0;  // heavy encoders left memoryless by CompileOptions::heavyMode off / observe
  uint32_t storesElided = 0;   // client Stores of MSAA attachments dropped as dead (CompileOptions::msaaDeadStore)
  uint32_t elideMiss = 0;      // frames' first access needed contents whose store had been elided (the texture is banned from it)
  uint32_t transientViolations = 0;  // TextureDesc::transient textures first caught needing memory (CompiledFrame::transientViolations)
  // Estimated bytes of multisampled attachments the render encoders load from / store to memory this frame (final decisions,
  // after placeTiles: a tile-local attachment counts nothing; samples x texel size of the format, one layer per rendered view).
  uint64_t msaaLoadBytes = 0, msaaStoreBytes = 0;
  bool splitViews = false;     // CompileOptions::splitViews was on for this frame
  uint32_t splitEncoders = 0;  // render encoders split per view (each is now two: Encoder::view 0 and 1)
};

// A draw by position: Frame::commands()[command] is a RenderPass, draws[draw] the draw.
struct DrawRef {
  uint32_t command = 0, draw = 0;
  auto operator<=>(const DrawRef&) const = default;
};

// An MSAA attachment stored and resolved that CompileOptions::msaaResolveOnTile could not keep on
// tile: why, and what the frame compiler saw of it (Device logs the first few as gpu.msaa).
struct MsaaReject {
  uint32_t encoder = 0;
  uint8_t slot = 0;             // colour index, or kMaxColorAttachments for depth
  TileVerdict why = TileVerdict::None;
  TextureId texture;
  uint32_t width = 0, height = 0;
  uint16_t layers = 0, mips = 0, samples = 0, layerCount = 0, layer = 0;
  uint8_t type = 0;             // TextureType
  bool persistent = false, transient = false;
  int firstRead = -1, lastUse = -1, attachedIn = -1;
  bool multi = false, sampled = false, elsewhere = false, badFetch = false;
  LoadAction load = LoadAction::DontCare;
};

// What one frame showed about an MSAA texture it touched, for ResourceTable::StoreHistory (applied by commitMemory).
struct StoreObs {
  TextureId texture;
  bool readFirst = false;   // the frame's first access needs the contents from before it (Load, sample, copy source, partial)
  bool candidate = false;   // the client Stores it and nothing else in the frame uses it: the store could be dropped
  bool elided = false;      // ... and it was
};

// A TextureDesc::transient texture the frame used the way the contract forbids: its contents are loaded by the first pass,
// or sampled / copied / written outside its encoder (why = LoadsContents, SampledRead or UsedElsewhere; load = the first
// attachment's load action; a first-attach Load of memory no frame defined is undefined content, not a violation). commitMemory marks it in the ResourceTable (ResourceTable::transientViolated): from the next
// frame an explicit Store of it is kept, so it gets memory lazily instead of losing its contents with the tile. A texture
// is reported once. The frame it is reported in already keeps whatever the verdict above rejects (verdict != TileLocal).
struct TransientViolation {
  TextureId texture;
  TileVerdict why = TileVerdict::None;
  LoadAction load = LoadAction::DontCare;
};

struct CompiledFrame {
  std::vector<Encoder> encoders;
  std::vector<TransientViolation> transientViolations;
  // CompileOptions::HeavyMode::Observe: heavy encoders that kept memoryless attachments (Device logs "gpu.tile observe").
  struct HeavyObs {
    uint32_t encoder = 0;
    uint64_t draws = 0, vertices = 0;
  };
  std::vector<HeavyObs> heavyObs;
  // Textures whose last writer in this frame was (valid) / was not (!valid) an autoMsaa slot, and the 1x image holding the
  // picture (ResourceTable::autoSource). Applied by commitMemory.
  struct AutoAfter {
    TextureId texture, src;
    uint16_t mip = 0, layer = 0;
    bool valid = false;
  };
  std::vector<AutoAfter> autoAfter;
  std::vector<StoreObs> storeObs;  // MSAA textures touched (only with CompileOptions::msaaDeadStore and a resource table)
  std::vector<TextureId> elideMisses;  // textures whose elided contents were needed this frame (Device logs them once)
  std::vector<Barrier> barriers;  // sorted by (encoder, item)
  std::vector<MsaaReject> msaaRejects;  // MSAA attachments the resolve-on-tile option could not keep on tile
  std::vector<TextureId> msaaStale;     // the first 8 textures counted in stats.msaaStaleReads (Device logs gpu.msaa_stale)
  // Draws the backend must not execute: a pixel-local read that no colour attachment of their
  // encoder serves (the shader would read garbage from [[color(n)]]). Sorted, unique.
  std::vector<DrawRef> badDraws;
  // Textures whose memory holds defined contents (true) or not (false) once this frame has run;
  // applied to the ResourceTable by commitMemory. Needs a resource table.
  std::vector<std::pair<TextureId, bool>> memAfter;
  CompileStats stats;
  bool skipped(uint32_t command, uint32_t draw) const;
};

struct CompileOptions {
  // Colour-attachment bytes per pixel and sample that one render encoder may hold. A fetch merge
  // that would gather more is refused (cut TileBudget, or a bad fetch for a brought-in slot): the
  // merged pipelines would not fit the tile. Whether an attachment is memoryless does not change
  // its tile footprint, so tile-local decisions do not consult it. Default 64: Metal feature set
  // tables, "maximum implicit image block size, per pixel, per sample, when using multiple color
  // render targets", Apple4-Apple6 (Apple7+ allow 128; the visionOS M2 is Apple8).
  uint32_t tileBytesPerPixel = 64;
  // A render encoder with more draws or vertices (count x instances; indirect counted as 3072), or with ANY indirect
  // draw (its counts are GPU-written, unbounded), keeps its attachments in memory instead of memoryless
  // (TileVerdict::HeavyGeometry): the tiler's parameter buffer may overflow and flush, which memoryless attachments
  // cannot survive. Cost: real memory for those attachments (e.g. MSAA colour+depth of both eyes) and no dead-store
  // elision (msaaDeadStore) in such encoders; samples of MSAA StoreAndResolve are still not written.
  // Device: VPVK_TILE_MAX_DRAWS / VPVK_TILE_MAX_VERTICES. Vertices count once per layer of the pass.
  uint32_t tileMaxDraws = 2000;
  uint64_t tileMaxVertices = 3000000;
  // An MSAA attachment stored AND resolved that nothing in the frame reads keeps only its resolve: the samples are not
  // written to memory every frame (only the store bandwidth is saved: a persistent texture is still allocated by the
  // backend). The persistent flag is ignored for it. Attachment::keepSamples opts an attachment out: the Vulkan translator
  // sets it on EVERY StoreAndResolve it emits (a client STORE + resolve, and the upgrades for a later subpass, a partial
  // area, self-dependencies, a resolve recorded as a pass, VPKTAA depth; Attachment::keepOrigin tells which), so for Vulkan
  // clients this flag changes nothing: a client STORE + resolve is only dropped by msaaDeadStore (below), and only a
  // DONT_CARE + resolve (already a Resolve) is memoryless otherwise. A read of the samples in a LATER submit would see undefined samples
  // (CompiledFrame::stats.msaaStaleReads counts such reads: msaa_stale in the frame report). Not to be confused with
  // VPVK_MSAA_TILE (gpu/vulkan): that one decides whether game attachments are created deferred / memoryless.
  // This flag covers colour AND depth; it is OFF by default (Device: VP_MSAA_TILE=1 / "depth").
  bool msaaResolveOnTile = false;
  // The same for colour attachments only (depth is left alone: VPKTAA reads the MSAA depth after the frame). Off in a bare
  // CompileOptions; Device turns it on by default (VP_MSAA_TILE=0/off/false turns it off).
  bool msaaColorResolveOnTile = false;
  // Copies to disjoint x/y rectangles of one texture subresource are not a hazard (see copyAccesses). Relies on
  // the backend keeping such copies independent inside one blit encoder; off by default (Device: VPVK_COPY_RECT_HAZARD=1).
  bool copyRects = false;
  // A client Store of an MSAA attachment that nothing in the frame reads afterwards is dropped (DontCare / Resolve, so the
  // texture can stay on tile) once ResourceTable::StoreHistory shows 3 consecutive frames whose next first access did not need
  // those contents. A frame that then needs them (miss) bans the texture from it. Only for textures no shader can read
  // (TextureDesc::shaderReadable false) and never for a store the translator needs (Attachment::keepOrigin Required, e.g. the
  // depth VPKTAA reads after the frame). Off in a bare CompileOptions; Device turns it on by default (VPVK_MSAA_DEAD_STORE=0
  // turns it off, and 8 misses switch it off for the run). Speculative: "nothing read it for 3 frames" does not prove the next
  // frame will not, so a miss costs one frame of undefined samples before the per-texture ban.
  bool msaaDeadStore = false;
  // msaaDeadStore stops guessing (no new store is elided; Device sets it after too many misses) but keeps watching: a texture whose
  // store was already elided still gets its miss counted and logged, and is banned, when a later frame needs its contents.
  bool msaaDeadStoreHalt = false;
  // A two-layer render pass that would be HeavyGeometry as one encoder, whose draws are all vertex amplified and that only draws
  // (no writes to resources, no framebuffer fetch, no occlusion queries; in-pass clears are fine, each encoder clears the rects of
  // its view; a rate map only with multisampled attachments, which gpu/metal2 never foveates) becomes two encoders, one per view
  // (Encoder::view): each draws one view, so the geometry the tiler must hold is halved and the encoder can be tile-local
  // (memoryless) again instead of HeavyGeometry. The heavy test counts one view's vertices for such a pass. Passes that are
  // eligible but not heavy keep one encoder (Encoder::splitWhy "not_heavy"), and so do those whose single view is still heavy by
  // geometry (an indirect draw, or more vertices than tileMaxVertices: "still_heavy"; the draw count alone does not stop it). Needs a backend that honours Encoder::view
  // (gpu/metal2). Switched at run time (splitViewsSwitch(); Device copies it here every frame); the pipelines do not change.
  bool splitViews = false;
  // false (Device: VPVK_FRAME_OPT=0): no pass is merged into an open render encoder and no copy is hoisted or sunk: every command
  // keeps its recorded position (an A/B switch against the optimiser; the hazards still get their barriers). Default true.
  bool frameOpt = true;
  // The limit Device started with while tileMaxVertices is a raised one (0 = not raised): CompileStats::tileRaised counts the
  // encoders that only the raised limit keeps off HeavyGeometry.
  uint64_t tileBaseVertices = 0;
  // What a HeavyGeometry encoder does (Device: VPVK_TILE_HEAVY=on|off|observe, default on). On: its attachments get real
  // memory (above). Off / Observe: they stay memoryless as for a light encoder, to find out whether the tiler survives
  // partial renders with them (the backend's memorylessFailed / "Too much geometry" tell); Observe also lists the heavy
  // encoders in CompiledFrame::heavyObs. Attachment::autoMsaa slots always behave as Off.
  enum class HeavyMode : uint8_t { On, Off, Observe } heavyMode = HeavyMode::On;
};

// Process-wide run-time switch of CompileOptions::splitViews, read by Device at the start of every frame. Starts from the
// environment (VPVK_SPLIT_VIEWS=1; default off). gpu/metal2 exports it as vpgpu_metal2_set_split_views.
std::atomic<int>& splitViewsSwitch();

// Process-wide run-time override of CompileOptions::tileMaxVertices, read by Device at the start of every frame: 0 = the value
// Device started with (tileMaxVerticesStart(), set by Device from its environment). gpu/metal2 exports it as
// vpgpu_metal2_set_tile_max_vertices (setTileMaxVertices). A value above the start value is tried first on ONE frame (the
// canary: the first one with an encoder the start value would call heavy, CompileStats::tileRaised) and the start value rules
// meanwhile; the backend reports that frame's command buffers (tileProbeCompleted / tileRaisedFailed): completed without a
// memoryless error = the value rules from then on; failed = back to the start value until the next set.
enum { kProbeIdle, kProbeWaiting, kProbeProbing, kProbeAccepted, kProbeFailed };
enum { kRaisedNone, kRaisedProbeFail, kRaisedRevert };
std::atomic<uint64_t>& tileMaxVerticesSwitch();
std::atomic<uint64_t>& tileMaxVerticesStart();
std::atomic<int>& tileMaxVerticesReverted();  // 1 once a raised value failed; a new set clears it
std::atomic<int>& tileProbeState();           // kProbe*
std::atomic<uint64_t>& tileProbeFrame();      // the canary's frame index (Device writes it before the backend gets the frame)
void setTileMaxVertices(uint64_t n);
// Backend, a command buffer of a frame with CompileStats::tileRaised > 0 failed on memoryless / too much geometry:
// kRaisedProbeFail (the canary), kRaisedRevert (a frame under the accepted value), kRaisedNone (nothing was in effect).
int tileRaisedFailed();
// Backend, every command buffer of `frame` completed without error: true when it was the canary (now accepted).
bool tileProbeCompleted(uint64_t frame);

// Vertices `d` runs per frame over `layers` views, as the heavy-geometry limit counts them: count x instances, once per layer
// unless the instances already hold the views (Draw::instanceViews > 1); an indirect draw counts as 3072.
uint64_t drawViewVertices(const Draw& d, uint32_t layers);

class ResourceTable;
// `resources` (optional) drops commands that touch destroyed resources instead of handing the
// backend dangling objects; it is also needed to decide tile-local attachments and fetch merges
// (persistent flag, sizes), which are all left in memory / refused without it.
CompiledFrame compileFrame(const Frame&, const ResourceTable* resources = nullptr, const CompileOptions& = {});
// Records in `resources` what `compiled` leaves in texture memory. Call once per frame that runs.
void commitMemory(const CompiledFrame& compiled, ResourceTable& resources);

}  // namespace vpgpu
