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
//     (store DontCare/Resolve, or TextureDesc::transient) whose whole life is that one encoder
//     becomes tile-local: store DontCare, memoryless allocation in the backend.
#pragma once

#include <array>
#include <cstdint>
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
  X(CopyWork)         /* compute encoder opened for copies/fills */

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
// the texture in the frame to be inside one render encoder (as attachment or pixel-local read),
// contents not loaded from memory, not persistent, whole texture covered, and the size/sample count
// of the encoder's other attachments.
#define VPGPU_TILE_VERDICTS(X)                                                           \
  X(None)           /* empty slot, stored by the client, or no resource table */         \
  X(TileLocal)      /* stays on tile: never stored, memoryless */                        \
  X(Persistent)     /* TextureDesc::persistent (history, next-frame reads) */            \
  X(LoadsContents)  /* first pass loads it: contents come from memory */                 \
  X(SampledRead)    /* read without pixelLocal somewhere (offsets, filtering, compute) */ \
  X(OtherEncoder)   /* attachment of more than one render encoder */                     \
  X(UsedElsewhere)  /* copied, or written outside its encoder */                         \
  X(Partial)        /* attachment covers only some mips/layers of the texture */         \
  X(SizeMismatch)   /* size or sample count differs from the encoder's first attachment */ \
  X(BadFetch)       /* pixel-local read of something that is not a colour attachment */

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
  uint32_t tileLocal = 0;      // textures kept on tile
  uint32_t tileRejected = 0;   // attachment textures that had to stay in memory
  uint32_t badFetch = 0;       // pixel-local reads that cannot be served: their draws are skipped
  uint64_t tileBytes = 0;      // memory the tile-local textures would have occupied
  uint32_t storesKept = 0;     // DontCare stores turned into Store: a later encoder uses the texture
};

// A draw by position: Frame::commands()[command] is a RenderPass, draws[draw] the draw.
struct DrawRef {
  uint32_t command = 0, draw = 0;
  auto operator<=>(const DrawRef&) const = default;
};

struct CompiledFrame {
  std::vector<Encoder> encoders;
  std::vector<Barrier> barriers;  // sorted by (encoder, item)
  // Draws the backend must not execute: a pixel-local read that no colour attachment of their
  // encoder serves (the shader would read garbage from [[color(n)]]). Sorted, unique.
  std::vector<DrawRef> badDraws;
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
};

class ResourceTable;
// `resources` (optional) drops commands that touch destroyed resources instead of handing the
// backend dangling objects; it is also needed to decide tile-local attachments and fetch merges
// (persistent flag, sizes), which are all left in memory / refused without it.
CompiledFrame compileFrame(const Frame&, const ResourceTable* resources = nullptr, const CompileOptions& = {});

}  // namespace vpgpu
