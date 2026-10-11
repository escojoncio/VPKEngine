// Turns a recorded Frame into the minimum number of encoders and barriers.
//
// Encoder shape follows Metal 4: render passes become render encoders; dispatches, copies and
// fills share compute encoders. Encoders are expensive (each one is a tile load/store and a
// driver round trip), so the compiler:
//   - merges consecutive render passes with the same attachments into one encoder;
//   - moves copies out of the way of render encoders: hoisted into the earliest compute encoder
//     no hazard forbids, or else sunk to right after the render encoder;
//   - places a barrier only where a real RAW/WAR/WAW hazard exists, per byte range / mip-layer box;
//   - records why every encoder was started, so the log explains the encoder count.
#pragma once

#include <cstdint>
#include <vector>

#include "vpgpu/core/commands.h"

namespace vpgpu {

#define VPGPU_CUT_REASONS(X)                                                            \
  X(FrameStart)       /* first render encoder of the frame */                           \
  X(AttachmentChange) /* render targets differ from the previous render encoder */      \
  X(ClearLoad)        /* same targets, but the pass clears: needs a fresh load action */ \
  X(Feedback)         /* pass reads/writes what the open render encoder touches */      \
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

struct Encoder {
  enum class Kind : uint8_t { Render, Compute } kind = Kind::Render;
  CutReason reason = CutReason::FrameStart;
  RenderPassDesc pass;                // render: load actions of the first pass, stores of the last
  std::vector<uint32_t> commands;     // indices into Frame::commands(), execution order
  uint32_t passes = 0;                // render: client passes merged in
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
};

struct CompiledFrame {
  std::vector<Encoder> encoders;
  std::vector<Barrier> barriers;  // sorted by (encoder, item)
  CompileStats stats;
};

class ResourceTable;
// `resources` (optional) drops commands that touch destroyed resources instead of handing the
// backend dangling objects.
CompiledFrame compileFrame(const Frame&, const ResourceTable* resources = nullptr);

}  // namespace vpgpu
