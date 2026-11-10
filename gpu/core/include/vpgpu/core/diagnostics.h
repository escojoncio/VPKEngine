// Per-frame report written to the log. With no Mac and no GPU debugger, these lines are how
// encoder counts, barriers and costs are inspected, so the format is fixed and grep-friendly:
//   every line starts with a tag ("gpu.frame", "gpu.cut", "gpu.enc", "gpu.pso") followed by
//   space-separated key=value pairs; keys never change meaning, values never contain spaces.
//   grep 'gpu.frame' log | awk  → encoders per frame;  grep 'gpu.enc.*reason=Feedback'  → culprits.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vpgpu/core/backend.h"
#include "vpgpu/core/frame_compiler.h"
#include "vpgpu/core/pipeline_cache.h"
#include "vpgpu/core/resources.h"

namespace vpgpu {

struct FrameReport {
  uint64_t frame = 0;
  uint32_t encoders = 0, renderEncoders = 0, computeEncoders = 0;
  uint32_t naiveEncoders = 0;  // one encoder per command: what an unmerged translation would do
  uint32_t passes = 0, barriers = 0;
  CompileStats compile;
  PipelineStats pipelines;
  double cpuCompileMs = 0;  // time spent in compileFrame
  GpuTimings gpu;           // empty until the backend reports this frame
};

FrameReport makeReport(uint64_t frame, const Frame&, const CompiledFrame&);

// Summary lines: "gpu.frame", "gpu.cut" and (if any pipeline activity) "gpu.pso".
std::vector<std::string> formatReport(const FrameReport&);
// One "gpu.enc" line per encoder, for the frames chosen for a detailed dump.
std::vector<std::string> formatEncoders(uint64_t frame, const Frame&, const CompiledFrame&,
                                        const GpuTimings* = nullptr);

// "gpu.pass.fuse": one line per render encoder that merged several client passes (fused=1), was cut by Feedback, or has draws that
// read its own depth attachment: how (sampled = the samples must be in memory, so the pass could not join the encoder that wrote
// the depth; tile = pixel-local, depth_input_in_tile=1) and, for a Feedback cut, the first texture that caused it. At most 8 lines.
std::vector<std::string> formatPassFuse(uint64_t frame, const Frame&, const CompiledFrame&, const ResourceTable* res);

// "gpu.resolve.use": one line per Resolve / StoreAndResolve attachment of a render encoder: what the encoders after it do with the
// resolve target in this frame (sampled draws, Loads and other attachments, copies, dispatches). dead_this_frame=1: nothing reads
// it later in the frame (a read in a later frame is not visible). At most 8 lines.
std::vector<std::string> formatResolveUse(uint64_t frame, const Frame&, const CompiledFrame&, const ResourceTable* res);

// One "gpu.bigframe" line for a frame that will cost many Metal encoders (encoders + barriers inside compute
// encoders, see bigFrame): encoders by kind, cuts, hazards, copies per label, and the destination textures with
// the most copies (format, size, mips/layers, label, copies, distinct rects). `res` may be null.
bool bigFrame(const CompiledFrame&, uint32_t threshold = 256);
std::string formatBigFrame(uint64_t frame, const Frame&, const CompiledFrame&, const ResourceTable* res);

}  // namespace vpgpu
