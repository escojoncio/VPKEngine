#include "vpgpu/core/diagnostics.h"

#include <cstdio>

namespace vpgpu {

namespace {

// Appends " key=value" to a line; printf-style keeps the formatting in one place.
template <class... Args>
void kv(std::string& line, const char* fmt, Args... args) {
  char buf[96];
  std::snprintf(buf, sizeof buf, fmt, args...);
  line += buf;
}

std::string hazardList(uint8_t bits) {
  static constexpr const char* names[kHazardKinds] = {"RAW", "WAR", "WAW"};
  std::string s;
  for (int k = 0; k < kHazardKinds; ++k)
    if (bits >> k & 1) s += (s.empty() ? "" : ",") + std::string(names[k]);
  return s;
}

std::string label(const Command& c) {
  std::string s;
  if (const auto* rp = std::get_if<RenderPass>(&c)) s = rp->desc.label;
  else if (const auto* d = std::get_if<Dispatch>(&c)) s = d->label;
  else s = std::get<Copy>(c).label;
  for (char& ch : s)
    if (ch == ' ' || ch == '=') ch = '_';  // keep the line splittable on spaces
  return s;
}

}  // namespace

FrameReport makeReport(uint64_t frame, const Frame& f, const CompiledFrame& c) {
  FrameReport r;
  r.frame = frame;
  r.encoders = static_cast<uint32_t>(c.encoders.size());
  for (const Encoder& e : c.encoders) {
    if (e.kind == Encoder::Kind::Render) ++r.renderEncoders, r.passes += e.passes;
    else ++r.computeEncoders;
  }
  r.naiveEncoders = static_cast<uint32_t>(f.commands().size());
  r.barriers = static_cast<uint32_t>(c.barriers.size());
  r.compile = c.stats;
  return r;
}

std::vector<std::string> formatReport(const FrameReport& r) {
  std::vector<std::string> lines;
  std::string l = "gpu.frame";
  kv(l, " f=%llu enc=%u render=%u compute=%u naive=%u", (unsigned long long)r.frame, r.encoders,
     r.renderEncoders, r.computeEncoders, r.naiveEncoders);
  kv(l, " passes=%u merged=%u hoisted=%u sunk=%u dropped=%u", r.passes, r.compile.passesMerged,
     r.compile.copiesHoisted, r.compile.copiesSunk, r.compile.dropped);
  kv(l, " barriers=%u intra=%u raw=%u war=%u waw=%u", r.barriers, r.compile.intraBarriers,
     r.compile.hazards[0], r.compile.hazards[1], r.compile.hazards[2]);
  kv(l, " tile_local=%u tile_rejected=%u saved_mb=%.1f", r.compile.tileLocal, r.compile.tileRejected,
     double(r.compile.tileBytes) / (1024.0 * 1024.0));
  if (r.compile.storesKept) kv(l, " stores_kept=%u", r.compile.storesKept);
  if (r.compile.badFetch) kv(l, " bad_fetch=%u", r.compile.badFetch);
  kv(l, " cpu_ms=%.3f", r.cpuCompileMs);
  if (r.gpu.frameMs > 0) kv(l, " gpu_ms=%.3f", r.gpu.frameMs);
  lines.push_back(std::move(l));

  l = "gpu.cut";
  kv(l, " f=%llu", (unsigned long long)r.frame);
  for (int i = 0; i < static_cast<int>(CutReason::Count); ++i)
    kv(l, " %s=%u", toString(static_cast<CutReason>(i)), r.compile.cuts[i]);
  lines.push_back(std::move(l));

  const PipelineStats& p = r.pipelines;
  if (p.compiled || p.failed || p.misses) {
    l = "gpu.pso";
    kv(l, " f=%llu compiled=%u failed=%u hits=%u misses=%u ms=%.2f max_ms=%.2f", (unsigned long long)r.frame,
       p.compiled, p.failed, p.hits, p.misses, p.totalMs, p.maxMs);
    lines.push_back(std::move(l));
  }
  return lines;
}

std::vector<std::string> formatEncoders(uint64_t frame, const Frame& f, const CompiledFrame& c, const GpuTimings* t) {
  std::vector<std::string> lines;
  size_t b = 0;
  for (size_t i = 0; i < c.encoders.size(); ++i) {
    const Encoder& e = c.encoders[i];
    std::string l = "gpu.enc";
    kv(l, " f=%llu i=%zu kind=%s reason=%s cmds=%zu", (unsigned long long)frame, i,
       e.kind == Encoder::Kind::Render ? "render" : "compute", toString(e.reason), e.commands.size());
    if (e.kind == Encoder::Kind::Render) {
      kv(l, " passes=%u", e.passes);
      // "tile=c0,c1,d" lists attachments kept on tile; "mem=c2:SampledRead" why others were not.
      std::string onTile, inMemory;
      for (int s = 0; s <= kMaxColorAttachments; ++s) {
        const TileVerdict v = e.tile[s];
        if (v == TileVerdict::None) continue;
        const std::string name = s == kDepthSlot ? "d" : "c" + std::to_string(s);
        std::string& to = v == TileVerdict::TileLocal ? onTile : inMemory;
        to += (to.empty() ? "" : ",") + name + (v == TileVerdict::TileLocal ? "" : std::string(":") + toString(v));
      }
      if (!onTile.empty()) l += " tile=" + onTile;
      if (!inMemory.empty()) l += " mem=" + inMemory;
    }
    uint32_t inside = 0;
    for (; b < c.barriers.size() && c.barriers[b].encoder == i; ++b) {
      const Barrier& br = c.barriers[b];
      if (br.item) ++inside;
      else kv(l, " barrier=%s:%x>%x", hazardList(br.hazards).c_str(), br.after, br.before);
    }
    if (inside) kv(l, " intra_barriers=%u", inside);
    if (t && i < t->encoderMs.size()) kv(l, " ms=%.3f", t->encoderMs[i]);
    if (!e.commands.empty()) {
      const std::string first = label(f.commands()[e.commands.front()]);
      if (!first.empty()) l += " label=" + first.substr(0, 48);
    }
    lines.push_back(std::move(l));
  }
  return lines;
}

}  // namespace vpgpu
