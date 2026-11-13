#include "vpgpu/core/diagnostics.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <map>
#include <set>
#include <tuple>

namespace vpgpu {

namespace {

// Appends " key=value" to a line; printf-style keeps the formatting in one place.
template <class... Args>
void kv(std::string& line, const char* fmt, Args... args) {
  char buf[256];
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
  if (r.compile.autoMsaaSlots) kv(l, " auto_msaa=%u seed_draws=%u seed_missing=%u", r.compile.autoMsaaSlots, r.compile.seedDraws, r.compile.seedMissing);
  if (r.compile.heavyObserved) kv(l, " heavy_observed=%u", r.compile.heavyObserved);
  if (r.compile.barriersComputeDraw) kv(l, " barr_c2d=%u", r.compile.barriersComputeDraw);
  if (r.compile.heavyIndirect) kv(l, " heavy_indirect=%u", r.compile.heavyIndirect);
  kv(l, " split=%d", r.compile.splitViews ? 1 : 0);  // CompileOptions::splitViews; split_enc = passes it split into two encoders
  if (r.compile.splitEncoders) kv(l, " split_enc=%u", r.compile.splitEncoders);
  kv(l, " verts=%llu", (unsigned long long)r.compile.maxEncoderVertices);  // largest render encoder, as the heavy limit counts
  kv(l, " tile_max=%llu heavy_enc=%u", (unsigned long long)r.compile.tileMaxVertices, r.compile.heavyEncoders);
  if (r.compile.tileRaised) kv(l, " tile_raised=%u", r.compile.tileRaised);  // encoders light only because of the raised limit
  if (r.compile.msaaStaleReads) kv(l, " msaa_stale=%u", r.compile.msaaStaleReads);
  if (r.compile.msaaLoadBytes || r.compile.msaaStoreBytes)  // estimated: samples moved between tile and memory (final load / store actions)
    kv(l, " msaa_load_mb=%.1f msaa_store_mb=%.1f", double(r.compile.msaaLoadBytes) / (1024.0 * 1024.0),
       double(r.compile.msaaStoreBytes) / (1024.0 * 1024.0));
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

namespace {

bool isTexture(const ResourceRange& r, TextureId t) {
  return t && r.kind == ResourceRange::Kind::Texture && r.index == t.index && r.generation == t.generation;
}

}  // namespace

std::vector<std::string> formatPassFuse(uint64_t frame, const Frame& f, const CompiledFrame& c, const ResourceTable* res) {
  std::vector<std::string> lines;
  constexpr size_t kMaxLines = 8;
  for (size_t i = 0; i < c.encoders.size() && lines.size() < kMaxLines; ++i) {
    const Encoder& e = c.encoders[i];
    if (e.kind != Encoder::Kind::Render) continue;
    // Draws that read the encoder's own depth attachment: sampled (a texture binding: needs the samples in memory) or pixel-local.
    uint32_t draws = 0, sampled = 0, tile = 0;
    for (uint32_t ci : e.commands) {
      const auto* rp = std::get_if<RenderPass>(&f.commands()[ci]);
      if (!rp) continue;
      for (const Draw& d : rp->draws) {
        if (d.clear) continue;
        ++draws;
        bool s = false, t = false;
        for (const Access& b : d.bindings)
          if (isTexture(b.range, e.pass.depth.texture)) (b.pixelLocal ? t : s) = true;
        sampled += s, tile += t;
      }
    }
    if (e.passes < 2 && !e.feedback.texture && !sampled && !tile) continue;
    std::string l = "gpu.pass.fuse";
    kv(l, " f=%llu enc=%zu passes=%u fused=%d reason=%s draws=%u", (unsigned long long)frame, i, e.passes, e.passes > 1 ? 1 : 0,
       toString(e.reason), draws);
    if (sampled || tile) kv(l, " own_depth_reads=sampled:%u,tile:%u depth_input_in_tile=%d", sampled, tile, sampled == 0 ? 1 : 0);
    if (e.feedback.texture) {
      kv(l, " cut_tex=%u cut_is_prev_depth=%d cut_write=%d", e.feedback.texture.index, e.feedback.prevDepth ? 1 : 0, e.feedback.write ? 1 : 0);
      if (const TextureDesc* d = res ? res->get(e.feedback.texture) : nullptr)
        kv(l, " cut_fmt=%s cut_s=%u", formatInfo(d->format).name, unsigned(d->samples));
    }
    if (const TextureDesc* d = res && e.pass.depth.texture ? res->get(e.pass.depth.texture) : nullptr)
      kv(l, " depth=%s,s%u,%ux%u", formatInfo(d->format).name, unsigned(d->samples), d->width, d->height);
    lines.push_back(std::move(l));
  }
  return lines;
}

std::vector<std::string> formatResolveUse(uint64_t frame, const Frame& f, const CompiledFrame& c, const ResourceTable* res) {
  std::vector<std::string> lines;
  constexpr size_t kMaxLines = 8;
  for (size_t i = 0; i < c.encoders.size() && lines.size() < kMaxLines; ++i) {
    const Encoder& e = c.encoders[i];
    if (e.kind != Encoder::Kind::Render) continue;
    for (int s = 0; s <= kMaxColorAttachments && lines.size() < kMaxLines; ++s) {
      const Attachment& a = s == kDepthSlot ? e.pass.depth : e.pass.colors[s];
      if (!a.texture || !a.resolve || (a.store != StoreAction::Resolve && a.store != StoreAction::StoreAndResolve)) continue;
      // What the encoders after this one do with the resolve target in this frame (a read next frame is not visible here).
      uint32_t sampled = 0, loaded = 0, attached = 0, copied = 0, dispatched = 0;
      for (size_t j = i + 1; j < c.encoders.size(); ++j) {
        for (uint32_t ci : c.encoders[j].commands) {
          const Command& cmd = f.commands()[ci];
          if (const auto* rp = std::get_if<RenderPass>(&cmd)) {
            const RenderPassDesc& d = rp->desc;
            auto look = [&](const Attachment& x) {
              if (x.texture == a.resolve) ++(x.load == LoadAction::Load ? loaded : attached);
            };
            for (const Attachment& x : d.colors) look(x);
            look(d.depth);
            for (const Draw& dr : rp->draws) {
              bool hit = false;
              for (const Access& b : dr.bindings) hit = hit || (!b.pixelLocal && isTexture(b.range, a.resolve));
              sampled += hit;
            }
          } else if (const auto* cp = std::get_if<Copy>(&cmd)) {
            copied += cp->srcTexture.texture == a.resolve;
          } else if (const auto* dp = std::get_if<Dispatch>(&cmd)) {
            bool hit = false;
            for (const Access& b : dp->bindings) hit = hit || isTexture(b.range, a.resolve);
            dispatched += hit;
          }
        }
      }
      std::string l = "gpu.resolve.use";
      kv(l, " f=%llu enc=%zu slot=%s src=%u dst=%u store=%s", (unsigned long long)frame, i,
         s == kDepthSlot ? "d" : ("c" + std::to_string(s)).c_str(), a.texture.index, a.resolve.index,
         a.store == StoreAction::Resolve ? "R" : "SR");
      if (const TextureDesc* d = res ? res->get(a.texture) : nullptr)
        kv(l, " fmt=%s s=%u %ux%ux%u", formatInfo(d->format).name, unsigned(d->samples), d->width, d->height, unsigned(e.pass.layerCount));
      kv(l, " later_sampled_draws=%u later_loaded=%u later_attached=%u", sampled, loaded, attached);
      kv(l, " later_copy_src=%u later_dispatch=%u dead_this_frame=%d", copied, dispatched,
         sampled + loaded + attached + copied + dispatched == 0 ? 1 : 0);
      lines.push_back(std::move(l));
    }
  }
  return lines;
}

// ---- gpu.pdump.* ----
namespace {

std::string cmdId(const std::vector<Command>& cmds, size_t i) {
  return std::string(std::holds_alternative<RenderPass>(cmds[i]) ? "p" : std::holds_alternative<Dispatch>(cmds[i]) ? "d" : "x") +
         std::to_string(i);
}

// A label that cannot break the key=value and list syntax of the dump lines.
std::string clean(std::string s, size_t max) {
  s = s.substr(0, max);
  for (char& c : s)
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '_' || c == '-')) c = '_';
  return s.empty() ? "-" : s;
}

std::string usageStr(uint8_t u) {
  std::string s;
  if (u & usage::sampled) s += 'S';
  if (u & usage::storage) s += 'G';
  if (u & usage::colorTarget) s += 'C';
  if (u & usage::depthTarget) s += 'D';
  if (u & usage::formatView) s += 'F';
  return s.empty() ? "-" : s;
}

std::string attStr(const char* slot, const Attachment& a, uint16_t layers, const ResourceTable* res) {
  static const char* const kStore[] = {"D", "S", "R", "SR"};
  const TextureDesc* d = res ? res->get(a.texture) : nullptr;
  const char ld = a.load == LoadAction::Load ? 'L' : a.load == LoadAction::Clear ? 'C' : 'D';
  const unsigned mip = std::min<unsigned>(a.mip, 31);  // a shift by 32 or more is undefined; a mip that deep is 1 texel anyway
  char b[160];
  std::snprintf(b, sizeof b, "%s=t%u/%s/s%u/%ux%ux%u/m%u/l%u/%c%s/u%s", slot, a.texture.index,
                d ? formatInfo(a.format != Format::Undefined ? a.format : d->format).name : "?", d ? unsigned(d->samples) : 0u,
                d ? std::max(1u, d->width >> mip) : 0u, d ? std::max(1u, d->height >> mip) : 0u, unsigned(layers),
                unsigned(a.mip), unsigned(a.layer), ld, kStore[int(a.store)], d ? usageStr(d->usage).c_str() : "?");
  std::string o = b;
  if (a.resolve) o += "->t" + std::to_string(a.resolve.index);
  if (d && !d->label.empty()) o += "/" + clean(d->label, 24);
  return o;
}

std::string listOf(const std::vector<uint32_t>& v, const std::vector<Command>& cmds, size_t cap = 8) {
  std::string s = "[";
  for (size_t i = 0; i < v.size() && i < cap; ++i) s += (i ? "," : "") + cmdId(cmds, v[i]);
  if (v.size() > cap) s += "+" + std::to_string(v.size() - cap);
  return s + "]";
}

// What happens to one texture over the frame, by command index.
struct TexEv {
  std::vector<uint32_t> w, wdc, r, load;
  bool target = false;
};

}  // namespace

std::vector<std::string> formatPassDump(uint64_t frame, const Frame& f, const CompiledFrame& c, const ResourceTable* res,
                                        const PipelineCache* pipes) {
  constexpr size_t kMaxLinks = 40, kMaxList = 24, kMaxPipes = 3, kMaxUmuPipes = 12, kMaxUmuFrame = 48;  // (umu lines: their own quota, the dump's 300-line cap must still hold the drop / tex / enc lines after them)
  const std::vector<Command>& cmds = f.commands();
  const unsigned long long fr = frame;
  std::vector<int> encOf(cmds.size(), -1), enc2Of(cmds.size(), -1);
  for (size_t e = 0; e < c.encoders.size(); ++e)
    for (uint32_t ci : c.encoders[e].commands)
      if (ci < cmds.size()) (encOf[ci] < 0 ? encOf : enc2Of)[ci] = int(e);

  std::string head = "gpu.pdump.frame";
  size_t nRender = 0, nDispatch = 0, nCopy = 0;
  for (const Command& cmd : cmds) (std::holds_alternative<RenderPass>(cmd) ? nRender : std::holds_alternative<Dispatch>(cmd) ? nDispatch : nCopy)++;
  kv(head, " f=%llu cmds=%zu render=%zu dispatch=%zu copy=%zu encoders=%zu", fr, cmds.size(), nRender, nDispatch, nCopy, c.encoders.size());

  std::map<uint64_t, TexEv> ev;  // by generation << 32 | index
  auto add = [&](std::vector<uint32_t> TexEv::*list, TextureId t, uint32_t cmd, bool target = false) {
    TexEv& e = ev[uint64_t{t.generation} << 32 | t.index];
    std::vector<uint32_t>& v = e.*list;
    if (v.empty() || v.back() != cmd) v.push_back(cmd);
    e.target = e.target || target;
  };
  auto accessEv = [&](const Access& a, uint32_t cmd) {
    if (a.range.kind != ResourceRange::Kind::Texture || a.pixelLocal) return;
    add(a.write ? &TexEv::w : &TexEv::r, TextureId{a.range.index, a.range.generation}, cmd);
  };

  std::vector<std::string> passLines;
  uint32_t renderPasses = 0;  // render passes before command i: the `rp=` of its line
  size_t umuFrame = 0;        // '.umu' pipeline lines of the frame so far
  for (uint32_t i = 0; i < cmds.size(); ++i) {
    if (const auto* dp = std::get_if<Dispatch>(&cmds[i])) {
      for (const Access& a : dp->bindings) accessEv(a, i);
      continue;
    }
    if (const auto* cp = std::get_if<Copy>(&cmds[i])) {
      for (const Access& a : copyAccesses(*cp)) accessEv(a, i);
      continue;
    }
    const RenderPass& rp = std::get<RenderPass>(cmds[i]);
    const RenderPassDesc& pd = rp.desc;
    std::string att;
    for (int s = 0; s <= kMaxColorAttachments; ++s) {
      const Attachment& a = s == kDepthSlot ? pd.depth : pd.colors[s];
      if (!a.texture) continue;
      att += (att.empty() ? "" : ",") + attStr(s == kDepthSlot ? "d" : ("c" + std::to_string(s)).c_str(), a, pd.layerCount, res);
      const bool stored = a.store == StoreAction::Store || a.store == StoreAction::StoreAndResolve;
      const bool resolved = a.store == StoreAction::Resolve || a.store == StoreAction::StoreAndResolve;
      if (a.load == LoadAction::Load) add(&TexEv::load, a.texture, i, true);
      add(stored ? &TexEv::w : &TexEv::wdc, a.texture, i, true);
      if (resolved && a.resolve) add(&TexEv::w, a.resolve, i, true);
    }
    struct PipeUse {
      uint32_t draws = 0, inst = 0, idxDraws = 0;
      uint64_t count = 0, idxCount = 0;
      std::vector<TextureId> tex;  // distinct sampled textures (at most 16 kept)
    };
    struct ResUse {
      uint32_t reads = 0, writes = 0;
    };
    std::map<uint64_t, PipeUse> pipeUse;
    std::map<std::array<float, 4>, uint32_t> vps;
    std::map<std::array<uint32_t, 4>, uint32_t> scs;
    std::map<std::tuple<int, uint32_t, uint32_t>, ResUse> resUse;  // 0 texture / 1 buffer, index, generation
    uint32_t draws = 0, clears = 0, indirect = 0, instMax = 0, status[5] = {};  // ready, pending, failed, missing, unknown (no cache)
    uint64_t idx = 0, verts = 0, inst = 0, work = 0;
    std::map<uint64_t, uint8_t> statusOf;
    for (const Draw& d : rp.draws) {
      if (d.clear) {
        ++clears;
        continue;
      }
      ++draws;
      PipeUse& pu = pipeUse[d.pipeline.value];
      ++pu.draws;
      if (d.indirect) {
        ++indirect;  // count is the upper bound only
      } else {
        (d.index ? idx : verts) += d.count;
        inst += d.instances, instMax = std::max(instMax, d.instances);
        work += uint64_t{d.count} * d.instances;
        pu.count += d.count, pu.inst += d.instances;
        if (d.index) ++pu.idxDraws, pu.idxCount += d.count;
      }
      ++vps[{d.viewport.x, d.viewport.y, d.viewport.width, d.viewport.height}];
      ++scs[{d.scissor.x, d.scissor.y, d.scissor.width, d.scissor.height}];
      auto [it, fresh] = statusOf.emplace(d.pipeline.value, pipes ? 3 : 4);
      if (fresh && pipes) {
        const PipelineCache::Status st = pipes->status(d.pipeline);
        it->second = st == PipelineCache::Status::Ready ? 0 : st == PipelineCache::Status::Failed ? 2
                     : st == PipelineCache::Status::Missing ? 3 : 1;
      }
      ++status[it->second];
      for (const Access& b : d.bindings) {
        accessEv(b, i);
        if (b.range.kind == ResourceRange::Kind::Sampler || b.pixelLocal) continue;
        if (b.range.kind == ResourceRange::Kind::Texture && pu.tex.size() < 16) {
          bool seen = false;
          for (const TextureId& t : pu.tex) seen = seen || (t.index == b.range.index && t.generation == b.range.generation);
          if (!seen) pu.tex.push_back(TextureId{b.range.index, b.range.generation});
        }
        ResUse& u = resUse[{b.range.kind == ResourceRange::Kind::Texture ? 0 : 1, b.range.index, b.range.generation}];
        ++(b.write ? u.writes : u.reads);
      }
      if (d.index) ++resUse[{1, d.index->buffer.index, d.index->buffer.generation}].reads;
    }
    std::string l = "gpu.pdump.pass";
    kv(l, " f=%llu pass=%u rp=%u enc=%d", fr, i, renderPasses++, encOf[i]);
    if (enc2Of[i] >= 0) kv(l, " enc2=%d", enc2Of[i]);
    kv(l, " label=%s layers=%u area=%ux%u", clean(pd.label, 40).c_str(), unsigned(pd.layerCount), pd.renderWidth, pd.renderHeight);
    l += " att{" + att + "}";
    kv(l, " draws=%u clears=%u indirect=%u idx=%llu verts=%llu inst=%llu inst_max=%u work=%llu pipes=%zu", draws, clears, indirect,
       (unsigned long long)idx, (unsigned long long)verts, (unsigned long long)inst, instMax, (unsigned long long)work, pipeUse.size());
    if (pipes) kv(l, " pipe_state{ready=%u pending=%u failed=%u missing=%u}", status[0], status[1], status[2], status[3]);
    auto top = [&](const auto& m, const char* name, auto fmt) {  // the 3 most used rectangles
      std::vector<std::pair<uint32_t, std::decay_t<decltype(m.begin()->first)>>> v;
      for (auto& [k, n] : m) v.push_back({n, k});
      std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.first > b.first; });
      l += std::string(" ") + name + "{";
      for (size_t k = 0; k < v.size() && k < 3; ++k) l += (k ? ";" : "") + fmt(v[k].second) + ":" + std::to_string(v[k].first);
      if (v.size() > 3) l += ";+" + std::to_string(v.size() - 3);
      l += "}";
    };
    char b[96];
    top(vps, "vp", [&](const std::array<float, 4>& v) {
      std::snprintf(b, sizeof b, "%g,%g,%gx%g", v[0], v[1], v[2], v[3]);
      return std::string(b);
    });
    top(scs, "sc", [&](const std::array<uint32_t, 4>& v) {
      std::snprintf(b, sizeof b, "%u,%u,%ux%u", v[0], v[1], v[2], v[3]);
      return std::string(b);
    });
    passLines.push_back(std::move(l));

    std::vector<std::pair<uint64_t, uint64_t>> busiest;  // draws, key
    for (auto& [k, u] : pipeUse) busiest.push_back({u.draws, k});
    std::sort(busiest.begin(), busiest.end(), [](auto& a, auto& b2) { return a.first > b2.first; });
    for (size_t k = 0; k < busiest.size() && k < kMaxPipes; ++k) {
      const PipeUse& u = pipeUse[busiest[k].second];
      static const char* const kStatus[] = {"ready", "pending", "failed", "missing", "unknown"};
      std::string pl = "gpu.pdump.pipe";
      kv(pl, " f=%llu pass=%u draws=%u count=%llu inst=%u status=%s", fr, i, u.draws, (unsigned long long)u.count, u.inst,
         kStatus[statusOf[busiest[k].second]]);
      PipelineState ps;
      if (pipes && pipes->stateOf(PipelineKey{busiest[k].second}, ps)) {
        ps.label = clean(ps.label, 40);  // (the game's own name: no spaces, '=' or line breaks)
        pl += " " + describePipelineState(ps);
      }
      else kv(pl, " key=%016llx state=unknown", (unsigned long long)busiest[k].second);
      passLines.push_back(std::move(pl));
    }
    // The UI-mask variants ('.umu' in the label) of the pass, beyond the 3 busiest: draws, indexed draws / indices and the textures they sample.
    if (pipes) {
      size_t shown = 0;
      for (size_t k = 0; k < busiest.size() && shown < kMaxUmuPipes && umuFrame < kMaxUmuFrame; ++k) {
        PipelineState ps;
        if (!pipes->stateOf(PipelineKey{busiest[k].second}, ps) || ps.label.find(".umu") == std::string::npos) continue;
        const PipeUse& u = pipeUse[busiest[k].second];
        ps.label = clean(ps.label, 40);
        std::string pl = "gpu.pdump.pipe";
        kv(pl, " f=%llu pass=%u umu=%zu draws=%u idx_draws=%u idx=%llu count=%llu inst=%u tex=[", fr, i, shown, u.draws, u.idxDraws,
           (unsigned long long)u.idxCount, (unsigned long long)u.count, u.inst);
        for (size_t t = 0; t < u.tex.size() && t < 6; ++t) {
          const TextureDesc* d = res ? res->get(u.tex[t]) : nullptr;
          kv(pl, "%st%u:%s:%ux%u", t ? "," : "", u.tex[t].index, d ? formatInfo(d->format).name : "?", d ? d->width : 0u, d ? d->height : 0u);
        }
        if (u.tex.size() > 6) kv(pl, ",+%zu", u.tex.size() - 6);
        pl += "] " + describePipelineState(ps);
        passLines.push_back(std::move(pl));
        ++shown, ++umuFrame;
      }
    }

    for (int dir = 0; dir < 2; ++dir) {  // 0 read, 1 written
      struct Item {
        uint32_t uses;
        std::string text;
      };
      std::vector<Item> items;
      uint32_t nTex = 0, nBuf = 0;
      for (auto& [k, u] : resUse) {
        const uint32_t uses = dir ? u.writes : u.reads;
        if (!uses) continue;
        const auto [kind, index, gen] = k;
        char t[160];
        if (kind == 0) {
          ++nTex;
          const TextureDesc* d = res ? res->get(TextureId{index, gen}) : nullptr;
          std::snprintf(t, sizeof t, "t%u:%s:%ux%ux%u:m%u:s%u:%u", index, d ? formatInfo(d->format).name : "?", d ? d->width : 0u,
                        d ? d->height : 0u, d ? unsigned(d->layers) : 0u, d ? unsigned(d->mips) : 0u, d ? unsigned(d->samples) : 0u, uses);
        } else {
          ++nBuf;
          const BufferDesc* d = res ? res->get(BufferId{index, gen}) : nullptr;
          std::snprintf(t, sizeof t, "b%u:%llu:%u", index, d ? (unsigned long long)d->size : 0ull, uses);
        }
        items.push_back({uses, t});
      }
      if (items.empty()) continue;
      std::stable_sort(items.begin(), items.end(), [](const Item& a, const Item& b2) { return a.uses > b2.uses; });
      std::string rl = "gpu.pdump.res";
      kv(rl, " f=%llu pass=%u dir=%s n=%zu tex=%u buf=%u shown=%zu list=[", fr, i, dir ? "w" : "r", items.size(), nTex, nBuf,
         std::min(items.size(), kMaxList));
      for (size_t k = 0; k < items.size() && k < kMaxList; ++k) rl += (k ? "," : "") + items[k].text;
      passLines.push_back(rl + "]");
    }
  }

  // Producers and consumers of every render target (colour and depth) of the frame.
  struct Link {
    bool stale;
    uint64_t key;
    std::string text;
  };
  std::vector<Link> links;
  for (const auto& [key, e] : ev) {
    if (!e.target) continue;
    const auto firstW = [&](uint32_t before, bool inclusive) {  // is there a stored write of the frame earlier than `before`?
      for (uint32_t w : e.w)
        if (w < before || (inclusive && w == before)) return true;
      return false;
    };
    std::vector<uint32_t> staleRead, staleLoad, self;
    for (uint32_t r : e.r) {
      if (firstW(r, false)) continue;
      (firstW(r, true) ? self : staleRead).push_back(r);
    }
    for (uint32_t r : e.load)
      if (!firstW(r, false)) staleLoad.push_back(r);
    const TextureId t{uint32_t(key & 0xffffffffu), uint32_t(key >> 32)};
    const TextureDesc* d = res ? res->get(t) : nullptr;
    std::string l = "gpu.pdump.link";
    kv(l, " f=%llu tex=%u kind=%s fmt=%s s=%u size=%ux%ux%u label=%s", fr, t.index, d && formatInfo(d->format).depth() ? "depth" : "color",
       d ? formatInfo(d->format).name : "?", d ? unsigned(d->samples) : 0u, d ? d->width : 0u, d ? d->height : 0u, d ? unsigned(d->layers) : 0u,
       d ? clean(d->label, 24).c_str() : "-");
    l += " w=" + listOf(e.w, cmds) + " wdc=" + listOf(e.wdc, cmds) + " r=" + listOf(e.r, cmds) + " stale_read=" + listOf(staleRead, cmds) +
         " stale_load=" + listOf(staleLoad, cmds) + " self=" + listOf(self, cmds);
    links.push_back({!staleRead.empty() || !staleLoad.empty(), key, std::move(l)});
  }
  std::stable_sort(links.begin(), links.end(), [](const Link& a, const Link& b) { return a.stale > b.stale; });

  std::vector<std::string> lines{std::move(head)};
  for (size_t k = 0; k < links.size() && k < kMaxLinks; ++k) lines.push_back(std::move(links[k].text));
  for (std::string& l : passLines) lines.push_back(std::move(l));
  return lines;
}

void capPassDump(uint64_t frame, std::vector<std::string>& lines, size_t max) {
  if (lines.size() <= max) return;
  const size_t dropped = lines.size() - max;
  lines.resize(max);
  lines.push_back("gpu.pdump.cut f=" + std::to_string(frame) + " dropped=" + std::to_string(dropped));
}

std::vector<std::string> passDumpEncoders(const Frame& f, const CompiledFrame& c) {
  std::vector<std::string> out;
  for (size_t e = 0; e < c.encoders.size(); ++e) {
    std::string s = "e" + std::to_string(e) + "[";
    const std::vector<uint32_t>& cs = c.encoders[e].commands;
    for (size_t k = 0; k < cs.size() && k < 6; ++k) s += (k ? "+" : "") + cmdId(f.commands(), cs[k]);
    if (cs.size() > 6) s += "+" + std::to_string(cs.size() - 6);
    out.push_back(s + "]");
  }
  return out;
}

std::string formatPassDumpTime(uint64_t frame, const std::vector<std::string>& encoders, const GpuTimings& t) {
  std::string l = "gpu.pdump.time";
  kv(l, " f=%llu gpu_ms=%.3f stamps=%zu", (unsigned long long)frame, t.frameMs, t.encoderMs.size());
  for (size_t e = 0; e < encoders.size() && e < t.encoderMs.size() && e < 64; ++e) kv(l, " %s=%.2f", encoders[e].c_str(), t.encoderMs[e]);
  return l;
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
