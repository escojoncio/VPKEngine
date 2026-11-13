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

// "gpu.pdump.*" (a client sets Frame::setPassDump; the Vulkan layer does for VPVK_PASS_DUMP): one frame explained pass by pass.
// `pass` is the index of the command in Frame::commands() (p<i> render pass, d<i> dispatch, x<i> copy), `rp` the ordinal of the
// pass among the frame's render passes (also in gpu.passtime), `enc` the encoder that runs it (the `pass=` of gpu.passtime).
//   gpu.pdump.frame f= cmds= render= dispatch= copy= encoders=
//   gpu.pdump.pass  f= pass= rp= enc= [enc2=] label= layers= area= att{c0=t<id>/<fmt>/s<samples>/<w>x<h>x<layers>/m<mip>/l<layer>/<load><store>/u<usage>[->t<resolve>]/<label>,d=..}
//                   draws= clears= indirect= idx= verts= inst= inst_max= work= pipes= pipe_state{ready= pending= failed= missing=} vp{x,y,wxh:n;..} sc{..}
//   gpu.pdump.pipe  f= pass= draws= count= inst= status= + describePipelineState (shader hashes, blend, depth, cull); the 3 busiest of a pass
//                   plus, for the pipelines whose label holds '.umu' (UI-mask variants), up to 12 more per pass and 48 per frame:
//                   gpu.pdump.pipe f= pass= umu=<n> draws= idx_draws= idx= count= inst= tex=[t<id>:<fmt>:<w>x<h>,..] + describePipelineState
//   gpu.pdump.res   f= pass= dir=r|w n= tex= buf= shown= list=[t<id>:<fmt>:<w>x<h>x<layers>:m<mips>:s<samples>:<draws>,b<id>:<bytes>:<draws>]
//   gpu.pdump.link  f= tex= kind=color|depth fmt= s= size= label= w=[..] wdc=[..] r=[..] stale_read=[..] stale_load=[..] self=[..]
//                   (w: passes that store it, wdc: render into it without storing, stale_*: read / loaded before any stored write of the
//                   frame = the previous frame's contents, self: sampled by the pass that writes it)
// `res` and `pipes` may be null. Not capped: Device::submit caps the whole dump (capPassDump).
std::vector<std::string> formatPassDump(uint64_t frame, const Frame&, const CompiledFrame&, const ResourceTable* res,
                                        const PipelineCache* pipes);
// Keeps the first `max` lines of a frame's dump and appends "gpu.pdump.cut f= dropped=N" (N = lines dropped) when there were more.
void capPassDump(uint64_t frame, std::vector<std::string>& lines, size_t max = 300);
// The commands each encoder runs ("e3[p5+p6]"), kept until the frame's GPU timings arrive, then "gpu.pdump.time f= gpu_ms= stamps= e3[p5+p6]=<ms>".
std::vector<std::string> passDumpEncoders(const Frame&, const CompiledFrame&);
std::string formatPassDumpTime(uint64_t frame, const std::vector<std::string>& encoders, const GpuTimings&);

}  // namespace vpgpu
