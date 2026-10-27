#include "vpgpu/core/diagnostics.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <tuple>

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
  if (r.compile.loadsDropped) kv(l, " loads_dropped=%u", r.compile.loadsDropped);
  if (r.compile.storesElided) kv(l, " stores_elided=%u", r.compile.storesElided);
  if (r.compile.elideMiss) kv(l, " elide_miss=%u", r.compile.elideMiss);
  if (r.compile.heavyResolve) kv(l, " heavy_resolve=%u", r.compile.heavyResolve);
  if (r.compile.barriersComputeDraw) kv(l, " barr_c2d=%u", r.compile.barriersComputeDraw);
  if (r.compile.heavyIndirect) kv(l, " heavy_indirect=%u", r.compile.heavyIndirect);
  kv(l, " split=%d", r.compile.splitViews ? 1 : 0);  // CompileOptions::splitViews; split_enc = passes it split into two encoders
  if (r.compile.splitEncoders) kv(l, " split_enc=%u", r.compile.splitEncoders);
  kv(l, " verts=%llu", (unsigned long long)r.compile.maxEncoderVertices);  // largest render encoder, as the heavy limit counts
  if (r.compile.msaaStaleReads) kv(l, " msaa_stale=%u", r.compile.msaaStaleReads);
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
  constexpr size_t kMaxLines = 64;  // production sampling (VP_ENC_DETAIL): bounded per frame
  for (size_t i = 0; i < c.encoders.size(); ++i) {
    if (i == kMaxLines) {
      lines.push_back("gpu.enc f=" + std::to_string(frame) + " truncated=" + std::to_string(c.encoders.size() - i));
      break;
    }
    const Encoder& e = c.encoders[i];
    std::string l = "gpu.enc";
    kv(l, " f=%llu i=%zu kind=%s reason=%s cmds=%zu", (unsigned long long)frame, i,
       e.kind == Encoder::Kind::Render ? "render" : "compute", toString(e.reason), e.commands.size());
    if (e.kind == Encoder::Kind::Render) {
      kv(l, " passes=%u", e.passes);
      if (e.view >= 0) kv(l, " view=%d", int(e.view));
      // "ls=c0:LS,d:CD": load (L=load, C=clear, D=dont care) and store (S=store, D=dont care,
      // R=resolve, SR=store+resolve) of every attachment.
      std::string ls;
      for (int s = 0; s <= kMaxColorAttachments; ++s) {
        const Attachment& a = s == kDepthSlot ? e.pass.depth : e.pass.colors[s];
        if (!a.texture) continue;
        static const char* const kStore[] = {"D", "S", "R", "SR"};
        const char ld = a.load == LoadAction::Load ? 'L' : a.load == LoadAction::Clear ? 'C' : 'D';
        ls += (ls.empty() ? "" : ",") + (s == kDepthSlot ? std::string("d") : "c" + std::to_string(s)) + ":" + ld +
              kStore[int(a.store)];
      }
      if (!ls.empty()) l += " ls=" + ls;
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

bool bigFrame(const CompiledFrame& c, uint32_t threshold) {
  // A compute encoder is one Metal encoder per barrier-separated run of items, so the barriers inside compute encoders
  // count: a handful of core encoders can still be tens of thousands of Metal ones.
  return c.encoders.size() + size_t(c.stats.intraBarriers) > threshold;
}

std::string formatBigFrame(uint64_t frame, const Frame& f, const CompiledFrame& c, const ResourceTable* res) {
  uint32_t render = 0, compute = 0;
  for (const Encoder& e : c.encoders) (e.kind == Encoder::Kind::Render ? render : compute)++;
  std::string l = "gpu.bigframe";
  kv(l, " f=%llu encoders=%zu render=%u compute=%u intra_barriers=%u est_mtl_encoders=%zu", (unsigned long long)frame,
     c.encoders.size(), render, compute, c.stats.intraBarriers, c.encoders.size() + size_t(c.stats.intraBarriers));
  std::string cuts;
  for (int i = 0; i < static_cast<int>(CutReason::Count); ++i)
    if (c.stats.cuts[i]) cuts += (cuts.empty() ? "" : ",") + std::string(toString(static_cast<CutReason>(i))) + ":" + std::to_string(c.stats.cuts[i]);
  l += " cuts=" + (cuts.empty() ? std::string("-") : cuts);
  kv(l, " hazards=RAW:%u,WAR:%u,WAW:%u barriers=%zu", c.stats.hazards[0], c.stats.hazards[1], c.stats.hazards[2],
     c.barriers.size());

  struct Dst {
    uint32_t copies = 0;
    std::set<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>> rects;  // mip, layer, x, y, w, h
    TextureId id;
  };
  std::map<std::string, uint32_t> byLabel;
  std::map<uint64_t, Dst> dsts;
  uint32_t copies = 0;
  constexpr size_t kMaxRects = 4096;  // per texture: the count saturates, the line stays cheap
  for (const Command& cmd : f.commands()) {
    const auto* cp = std::get_if<Copy>(&cmd);
    if (!cp) continue;
    ++copies;
    ++byLabel[cp->label.empty() ? "-" : label(cmd)];
    if (cp->kind != CopyKind::BufferToTexture && cp->kind != CopyKind::TextureToTexture && cp->kind != CopyKind::TextureBlit) continue;
    const TextureSpan& d = cp->dstTexture;
    Dst& dst = dsts[uint64_t{d.texture.generation} << 32 | d.texture.index];
    dst.id = d.texture;
    ++dst.copies;
    if (dst.rects.size() < kMaxRects) dst.rects.insert({d.mip, d.layer, d.x, d.y, d.width, d.height});
  }
  kv(l, " copies=%u", copies);
  std::vector<std::pair<std::string, uint32_t>> hist(byLabel.begin(), byLabel.end());
  std::sort(hist.begin(), hist.end(), [](auto& a, auto& b) { return a.second > b.second; });
  std::string h;
  for (size_t i = 0; i < hist.size() && i < 12; ++i) h += (h.empty() ? "" : ",") + hist[i].first.substr(0, 24) + ":" + std::to_string(hist[i].second);
  l += " by_label=" + (h.empty() ? std::string("-") : h);
  std::vector<const Dst*> top;
  for (const auto& [k, d] : dsts) top.push_back(&d);
  std::sort(top.begin(), top.end(), [](const Dst* a, const Dst* b) { return a->copies > b->copies; });
  for (size_t i = 0; i < top.size() && i < 5; ++i) {
    const Dst& d = *top[i];
    const TextureDesc* td = res ? res->get(d.id) : nullptr;
    std::string name = td && !td->label.empty() ? td->label : "-";
    for (char& ch : name)
      if (ch == ' ' || ch == '=') ch = '_';
    char b[200];
    // tiling: the Vulkan layer labels its linear images "vk.linear"; anything else is the driver's own layout.
    std::snprintf(b, sizeof b, " top%zu=%s:%ux%ux%u:mips%u:layers%u:label=%s:tiling=%s:copies=%u:rects=%zu%s", i + 1,
                  td ? formatInfo(td->format).name : "?", td ? td->width : 0u, td ? td->height : 0u, td ? td->depth : 0u,
                  td ? unsigned(td->mips) : 0u, td ? unsigned(td->layers) : 0u, name.substr(0, 32).c_str(),
                  name.find("linear") != std::string::npos ? "linear" : "optimal", d.copies, d.rects.size(),
                  d.rects.size() >= kMaxRects ? "+" : "");
    l += b;
  }
  return l;
}

}  // namespace vpgpu
