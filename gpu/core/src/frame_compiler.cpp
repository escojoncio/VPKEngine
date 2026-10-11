#include "vpgpu/core/frame_compiler.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "vpgpu/core/pipeline_cache.h"
#include "vpgpu/core/resources.h"

namespace vpgpu {

const char* toString(CutReason r) {
  static constexpr const char* names[] = {
#define VPGPU_X(name) #name,
      VPGPU_CUT_REASONS(VPGPU_X)
#undef VPGPU_X
  };
  const auto i = static_cast<size_t>(r);
  return i < std::size(names) ? names[i] : "?";
}

const char* toString(TileVerdict v) {
  static constexpr const char* names[] = {
#define VPGPU_X(name) #name,
      VPGPU_TILE_VERDICTS(VPGPU_X)
#undef VPGPU_X
  };
  const auto i = static_cast<size_t>(v);
  return i < std::size(names) ? names[i] : "?";
}

namespace {

// An access plus whether it comes from a render attachment: attachment-vs-attachment overlaps
// inside one render encoder are ordered by the rasterizer and never force a cut. Pixel-local reads
// count as attachment accesses: they are served from the tile, in rasterization order.
struct Tagged {
  Access access;
  bool attachment = false;
};
using AccessList = std::vector<Tagged>;

bool conflicts(const AccessList& a, const AccessList& b) {
  for (const Tagged& x : a)
    for (const Tagged& y : b) {
      if (x.attachment && y.attachment) continue;
      if ((x.access.write || y.access.write) && x.access.range.overlaps(y.access.range)) return true;
    }
  return false;
}

void append(AccessList& to, const AccessList& from) { to.insert(to.end(), from.begin(), from.end()); }

bool clears(const RenderPassDesc& d) {
  if (d.depth.texture && d.depth.load == LoadAction::Clear) return true;
  for (const Attachment& a : d.colors)
    if (a.texture && a.load == LoadAction::Clear) return true;
  return false;
}

// Some attachment of `d` clears, among the slots `existing` already has: a clear of a slot the
// encoder already holds needs a fresh load action, a clear of a new slot runs at encoder start.
bool clearsExisting(const RenderPassDesc& existing, const RenderPassDesc& d) {
  if (d.depth.texture && d.depth.load == LoadAction::Clear && existing.depth.texture) return true;
  for (int c = 0; c < kMaxColorAttachments; ++c)
    if (d.colors[c].texture && d.colors[c].load == LoadAction::Clear && existing.colors[c].texture) return true;
  return false;
}

bool alive(const ResourceRange& r, const ResourceTable& t) {
  switch (r.kind) {
    case ResourceRange::Kind::Buffer: return t.get(BufferId{r.index, r.generation}) != nullptr;
    case ResourceRange::Kind::Texture: return t.get(TextureId{r.index, r.generation}) != nullptr;
    case ResourceRange::Kind::Sampler: return t.get(SamplerId{r.index, r.generation}) != nullptr;
  }
  return false;
}

// Render size and sample count of an attachment, for union-merge and tile checks.
struct Extent {
  uint32_t w = 0, h = 0;
  uint8_t samples = 0;
  bool operator==(const Extent&) const = default;
};
Extent extent(const Attachment& a, const ResourceTable& t) {
  const TextureDesc* d = t.get(a.texture);
  if (!d) return {};
  return {std::max(1u, d->width >> a.mip), std::max(1u, d->height >> a.mip), d->samples};
}

template <class F>
void forEachSlot(RenderPassDesc& d, F&& f) {
  for (int c = 0; c < kMaxColorAttachments; ++c) f(d.colors[c], c);
  f(d.depth, kDepthSlot);
}
template <class F>
void forEachSlot(const RenderPassDesc& d, F&& f) {
  for (int c = 0; c < kMaxColorAttachments; ++c) f(d.colors[c], c);
  f(d.depth, kDepthSlot);
}

// Slot of `pass` holding texture `t` (any mip/layer), or -1.
int slotOf(const RenderPassDesc& pass, TextureId t) {
  int found = -1;
  forEachSlot(pass, [&](const Attachment& a, int s) {
    if (found < 0 && a.texture == t) found = s;
  });
  return found;
}

// Two attachment sets that can share one encoder: no slot holds different targets, no texture sits
// in two slots, same layer count, and every attachment has the same size and sample count (Metal
// requires it).
bool compatible(const RenderPassDesc& a, const RenderPassDesc& b, const ResourceTable& t) {
  if (a.layerCount != b.layerCount) return false;
  bool ok = true;
  Extent ref;
  auto check = [&](const Attachment& x) {
    if (!x.texture) return;
    const Extent e = extent(x, t);
    if (!e.samples || (ref.samples && !(e == ref))) ok = false;
    ref = e;
  };
  forEachSlot(a, [&](const Attachment& x, int) { check(x); });
  forEachSlot(b, [&](const Attachment& x, int) { check(x); });
  forEachSlot(b, [&](const Attachment& x, int s) {
    if (!x.texture) return;
    const Attachment& y = s == kDepthSlot ? a.depth : a.colors[s];
    if (y.texture && !y.sameTarget(x)) ok = false;
    if (const int at = slotOf(a, x.texture); at >= 0 && at != s) ok = false;
  });
  return ok;
}

// Tile bytes per pixel and sample of the colour attachments of `a` and `b` together.
uint64_t colorBytes(const RenderPassDesc& a, const RenderPassDesc& b, const ResourceTable& t) {
  uint64_t n = 0;
  for (int c = 0; c < kMaxColorAttachments; ++c) {
    const Attachment& x = a.colors[c].texture ? a.colors[c] : b.colors[c];
    if (const TextureDesc* d = x.texture ? t.get(x.texture) : nullptr) n += formatInfo(d->format).bytesPerBlock;
  }
  return n;
}

AccessList accessesOf(const Command& cmd) {
  AccessList out;
  auto add = [&](const std::vector<Access>& list, bool attachment) {
    for (const Access& a : list) out.push_back({a, attachment});
  };
  if (const auto* rp = std::get_if<RenderPass>(&cmd)) {
    add(attachmentAccesses(rp->desc), true);
    for (const Draw& d : rp->draws)
      for (const Access& a : drawAccesses(d)) out.push_back({a, a.pixelLocal});
  } else if (const auto* dp = std::get_if<Dispatch>(&cmd)) {
    add(dp->bindings, false);
  } else {
    add(copyAccesses(std::get<Copy>(cmd)), false);
  }
  return out;
}

// Phase 1: decide encoder grouping and command order.
class Scheduler {
 public:
  Scheduler(CompiledFrame& out, const ResourceTable* resources, const CompileOptions& options)
      : out_(out), res_(resources), options_(options) {}

  // `acc` is updated in place: pixel-local reads that cannot be served from the tile lose their
  // attachment tag; they are reported as bad fetches and their draws listed in badDraws.
  void renderPass(uint32_t i, const RenderPass& rp, AccessList& acc) {
    const RenderPassDesc& desc = rp.desc;
    // Pixel-local reads are always served as colour attachments of the encoder, at the slot the
    // texture was last rendered through (see fetchIndex). If the texture is not one of this pass's
    // own attachments it is added: merged into the encoder that holds it, or else loaded.
    RenderPassDesc need = desc;
    bool fetchesOpen = false;
    AccessList added;  // slots brought in only for a fetch are stored back: they count as writes
    std::vector<ResourceRange> bad;
    for (uint32_t j = 0; j < rp.draws.size(); ++j) {
      bool drawBad = false;
      for (const Access& b : rp.draws[j].bindings) {
        if (!b.pixelLocal) continue;
        const ResourceRange& r = b.range;
        int slot = fetchIndex(need, r);
        if (slot < 0 && (slot = lastSlot(r)) >= 0) {
          if (!addFetchSlot(need, r, slot)) slot = -1;
          else added.push_back({write(r, stage::fragment), true});
        }
        if (slot < 0) {
          drawBad = true;
          ++out_.stats.badFetch;
          bad.push_back(r);
          continue;
        }
        fetchesOpen = fetchesOpen || (open_ >= 0 && fetchIndex(out_.encoders[open_].pass, r) == slot);
      }
      // Every read served, but two texture slots on one attachment: one [[color(n)]] cannot be both.
      FetchInputs fi;
      if (!drawBad && !fetchInputs(rp.draws[j], need, fi)) drawBad = true, ++out_.stats.badFetch;
      if (drawBad) out_.badDraws.push_back({i, j});
    }
    // A range that failed once fails for every draw of the pass (`need` only grows on success).
    for (Tagged& t : acc)
      if (t.access.pixelLocal)
        for (const ResourceRange& r : bad)
          if (t.access.range.contains(r) && r.contains(t.access.range))
            t.attachment = false;
    append(acc, added);

    CutReason reason;
    if (open_ >= 0) {
      Encoder& r = out_.encoders[open_];
      // A pass with other targets may still join when it reads the encoder's attachments on tile:
      // that is the G-buffer -> lighting case, the one that matters most on a TBDR GPU.
      // Its pipelines must then declare the formats of the merged encoder (Metal checks every
      // attachment), which a framebuffer-fetch pipeline does anyway.
      const bool same = r.pass.sameTargets(need);
      const bool joinable = same || (fetchesOpen && res_ && compatible(r.pass, need, *res_));
      if (!joinable) reason = CutReason::AttachmentChange;
      else if (!same && colorBytes(r.pass, need, *res_) > options_.tileBytesPerPixel) reason = CutReason::TileBudget;
      else if (clearsExisting(r.pass, desc)) reason = CutReason::ClearLoad;
      else if (conflicts(acc, acc_[open_])) reason = CutReason::Feedback;
      else if (conflicts(acc, sunkAcc_)) reason = CutReason::PinnedCopy;
      else return merge(i, desc, need, acc);
      closeRender();
    } else if (!haveLastRender_) {
      reason = CutReason::FrameStart;
    } else if (!lastRender_.sameTargets(need)) {
      reason = CutReason::AttachmentChange;
    } else {
      reason = clears(desc) ? CutReason::ClearLoad : CutReason::ComputeBetween;
    }
    open_ = push(Encoder::Kind::Render, reason, i, acc);
    out_.encoders[open_].pass = need;
    out_.encoders[open_].passes = 1;
  }

  void copy(uint32_t i, AccessList acc) {
    if (open_ < 0) return computeWork(i, CutReason::CopyWork, std::move(acc));
    if (conflicts(acc, acc_[open_]) || conflicts(acc, sunkAcc_)) {
      sunk_.push_back(i);
      append(sunkAcc_, acc);
      ++out_.stats.copiesSunk;
      return;
    }
    // Hoist as far up as no earlier encoder conflicts: copies of a whole frame then gather in
    // a few compute encoders instead of one in front of every render encoder.
    int target = -1;
    for (int e = open_ - 1; e >= 0; --e) {
      if (out_.encoders[e].kind == Encoder::Kind::Compute) target = e;
      if (conflicts(acc, acc_[e])) break;
    }
    if (target < 0) target = insertComputeBeforeOpen();
    out_.encoders[target].commands.push_back(i);
    append(acc_[target], acc);
    ++out_.stats.copiesHoisted;
  }

  void dispatch(uint32_t i, AccessList acc) {
    closeRender();
    computeWork(i, CutReason::ComputeWork, std::move(acc));
  }

  void finish() { closeRender(); }

 private:
  // Brings texture range `r` in at colour `slot` of `need` for a fetch. Refused (the read is then a
  // bad fetch) when compatible() rejects it against what `need` already holds (slot taken, texture
  // in another slot, other size or sample count), it would exceed the tile budget, or there is no
  // resource table to check that with.
  bool addFetchSlot(RenderPassDesc& need, const ResourceRange& r, int slot) const {
    if (!res_) return false;
    RenderPassDesc one;
    one.layerCount = need.layerCount;
    Attachment& a = one.colors[slot];
    a.texture = TextureId{r.index, r.generation};
    a.mip = r.mipBegin;
    a.layer = r.layerBegin;
    a.load = LoadAction::Load;
    a.store = StoreAction::Store;
    if (!compatible(need, one, *res_) || colorBytes(need, one, *res_) > options_.tileBytesPerPixel) return false;
    need.colors[slot] = a;
    return true;
  }

  // Slot of the most recent render encoder attachment that is exactly `r`, or -1.
  int lastSlot(const ResourceRange& r) const {
    for (auto e = out_.encoders.rbegin(); e != out_.encoders.rend(); ++e)
      if (e->kind == Encoder::Kind::Render)
        if (const int s = fetchIndex(e->pass, r); s >= 0) return s;
    return -1;
  }

  void merge(uint32_t i, const RenderPassDesc& desc, const RenderPassDesc& need, const AccessList& acc) {
    Encoder& r = out_.encoders[open_];
    // Loads come from the pass that brought the slot in (new slots load at encoder start: nothing
    // in the encoder touched them before); stores from the last pass that declared the slot.
    auto join = [&](Attachment& have, const Attachment& want, const Attachment& own) {
      if (!have.texture) {
        have = want;
      } else if (own.texture) {
        have.store = own.store;
        have.resolve = own.resolve;
        have.resolveMip = own.resolveMip;
        have.resolveLayer = own.resolveLayer;
      }
    };
    for (int c = 0; c < kMaxColorAttachments; ++c) join(r.pass.colors[c], need.colors[c], desc.colors[c]);
    join(r.pass.depth, need.depth, desc.depth);
    r.commands.push_back(i);
    ++r.passes;
    append(acc_[open_], acc);
    ++out_.stats.passesMerged;
  }

  int push(Encoder::Kind kind, CutReason reason, uint32_t i, AccessList acc) {
    Encoder e;
    e.kind = kind;
    e.reason = reason;
    e.commands = {i};
    out_.encoders.push_back(std::move(e));
    acc_.push_back(std::move(acc));
    return static_cast<int>(out_.encoders.size()) - 1;
  }

  int insertComputeBeforeOpen() {
    Encoder e;
    e.kind = Encoder::Kind::Compute;
    e.reason = CutReason::CopyWork;
    out_.encoders.insert(out_.encoders.begin() + open_, std::move(e));
    acc_.insert(acc_.begin() + open_, AccessList{});
    return open_++;
  }

  void computeWork(uint32_t i, CutReason reason, AccessList acc) {
    if (!out_.encoders.empty() && out_.encoders.back().kind == Encoder::Kind::Compute) {
      out_.encoders.back().commands.push_back(i);
      append(acc_.back(), acc);
    } else {
      push(Encoder::Kind::Compute, reason, i, std::move(acc));
    }
  }

  void closeRender() {
    if (open_ < 0) return;
    lastRender_ = out_.encoders[open_].pass;
    haveLastRender_ = true;
    open_ = -1;
    if (sunk_.empty()) return;
    const int c = push(Encoder::Kind::Compute, CutReason::CopyWork, sunk_.front(), std::move(sunkAcc_));
    out_.encoders[c].commands = std::move(sunk_);
    sunk_.clear();
    sunkAcc_.clear();
  }

  CompiledFrame& out_;
  const ResourceTable* res_;
  const CompileOptions& options_;
  std::vector<AccessList> acc_;  // parallel to out_.encoders
  int open_ = -1;                // open render encoder
  std::vector<uint32_t> sunk_;   // copies deferred to right after the open render encoder
  AccessList sunkAcc_;
  RenderPassDesc lastRender_;
  bool haveLastRender_ = false;
};

// Phase 2: walk the final order and place barriers only for real hazards.
class BarrierPlacer {
 public:
  explicit BarrierPlacer(CompiledFrame& out) : out_(out) {}

  void access(const Access& a, uint32_t encoder, uint32_t item) {
    if (a.range.kind == ResourceRange::Kind::Sampler) return;  // immutable: never a hazard
    const Pos p{encoder, item};
    auto& records = records_[key(a.range)];
    uint8_t kinds = 0;
    StageMask after = 0;
    for (const Record& r : records) {
      if (r.pos == p || !r.access.range.overlaps(a.range)) continue;
      if (!r.access.write && !a.write) continue;
      if (covered(r.pos, r.access.stages, a.stages)) continue;
      const HazardKind k = !r.access.write ? HazardKind::WAR : a.write ? HazardKind::WAW : HazardKind::RAW;
      kinds |= 1 << static_cast<int>(k);
      after |= r.access.stages;
    }
    if (kinds) addBarrier(p, after, a.stages, kinds);
    // A write orders everything it fully covers (through the barrier just placed), so those
    // records can no longer cause a hazard on their own.
    if (a.write)
      std::erase_if(records, [&](const Record& r) { return r.pos < p && a.range.contains(r.access.range); });
    records.push_back({a, p});
  }

 private:
  struct Pos {
    uint32_t encoder = 0, item = 0;
    auto operator<=>(const Pos&) const = default;
  };
  struct Record {
    Access access;
    Pos pos;
  };

  static uint64_t key(const ResourceRange& r) {
    return (uint64_t{r.index} << 1 | (r.kind == ResourceRange::Kind::Texture)) ^ (uint64_t{r.generation} << 33);
  }

  bool covered(Pos from, StageMask src, StageMask dst) const {
    for (auto b = out_.barriers.rbegin(); b != out_.barriers.rend(); ++b) {
      if (Pos{b->encoder, b->item} <= from) break;
      if ((b->after & src) == src && (b->before & dst) == dst) return true;
    }
    return false;
  }

  void addBarrier(Pos p, StageMask after, StageMask before, uint8_t kinds) {
    auto& list = out_.barriers;
    if (list.empty() || Pos{list.back().encoder, list.back().item} != p) list.push_back({p.encoder, p.item, 0, 0, 0});
    Barrier& b = list.back();
    b.after |= after;
    b.before |= before;
    for (int k = 0; k < kHazardKinds; ++k)
      if ((kinds >> k & 1) && !(b.hazards >> k & 1)) ++out_.stats.hazards[k];
    b.hazards |= kinds;
  }

  CompiledFrame& out_;
  std::unordered_map<uint64_t, std::vector<Record>> records_;
};

// Phase 3: decide which attachments never need memory (see TileVerdict).
void placeTiles(CompiledFrame& out, const std::vector<Command>& cmds, const std::vector<AccessList>& accesses,
                const ResourceTable& res) {
  struct Use {
    int encoder = -1, last = -1;  // render encoder holding it as attachment; last encoder using it
    bool multi = false, pixelLocal = false, sampled = false, elsewhere = false, badFetch = false;
  };
  std::unordered_map<uint64_t, Use> uses;
  auto key = [](uint32_t index, uint32_t generation) { return uint64_t{generation} << 32 | index; };
  for (uint32_t e = 0; e < out.encoders.size(); ++e) {
    const Encoder& enc = out.encoders[e];
    for (uint32_t ci : enc.commands) {
      const bool isCopy = std::holds_alternative<Copy>(cmds[ci]);
      for (const Tagged& t : accesses[ci]) {
        const ResourceRange& r = t.access.range;
        if (r.kind != ResourceRange::Kind::Texture) continue;
        Use& u = uses[key(r.index, r.generation)];
        u.last = static_cast<int>(e);
        if (t.access.pixelLocal) (t.attachment ? u.pixelLocal : u.badFetch) = true;
        else if (t.attachment) continue;  // membership comes from the encoder's pass below
        else if (isCopy || t.access.write) u.elsewhere = true;
        else u.sampled = true;
      }
    }
    if (enc.kind != Encoder::Kind::Render) continue;
    forEachSlot(enc.pass, [&](const Attachment& a, int) {
      if (!a.texture) return;
      Use& u = uses[key(a.texture.index, a.texture.generation)];
      u.multi = u.multi || (u.encoder >= 0 && u.encoder != static_cast<int>(e));
      u.encoder = static_cast<int>(e);
      if (a.resolve) uses[key(a.resolve.index, a.resolve.generation)].elsewhere = true;  // resolved into memory
    });
  }

  std::unordered_set<uint64_t> rejected;
  for (uint32_t e = 0; e < out.encoders.size(); ++e) {
    Encoder& enc = out.encoders[e];
    if (enc.kind != Encoder::Kind::Render) continue;
    Extent ref;
    forEachSlot(enc.pass, [&](const Attachment& a, int) {
      if (a.texture && !ref.samples) ref = extent(a, res);
    });
    forEachSlot(enc.pass, [&](Attachment& a, int slot) {
      TileVerdict& v = enc.tile[slot];
      const TextureDesc* d = a.texture ? res.get(a.texture) : nullptr;
      if (!d) return;
      const uint64_t k = key(a.texture.index, a.texture.generation);
      const Use& u = uses[k];
      // Only textures the client lets go of are candidates: not stored, or marked transient. An
      // explicit Store may be presented, read back by the CPU or read next frame.
      const bool resolves = a.store == StoreAction::Resolve || a.store == StoreAction::StoreAndResolve;
      if (a.store != StoreAction::DontCare && a.store != StoreAction::Resolve && !d->transient) v = TileVerdict::None;
      else if (d->persistent) v = TileVerdict::Persistent;
      else if (u.elsewhere) v = TileVerdict::UsedElsewhere;
      else if (u.badFetch) v = TileVerdict::BadFetch;
      else if (u.sampled) v = TileVerdict::SampledRead;
      else if (u.multi) v = TileVerdict::OtherEncoder;
      else if (a.load == LoadAction::Load) v = TileVerdict::LoadsContents;
      else if (d->type != TextureType::Tex2D || d->mips != 1 || d->layers != enc.pass.layerCount || a.layer) v = TileVerdict::Partial;
      else if (!(extent(a, res) == ref)) v = TileVerdict::SizeMismatch;
      else v = TileVerdict::TileLocal;
      if (v == TileVerdict::TileLocal) {
        a.store = resolves ? StoreAction::Resolve : StoreAction::DontCare;
        ++out.stats.tileLocal;
        out.stats.tileBytes += uint64_t{d->width} * d->height * d->samples * d->layers * formatInfo(d->format).bytesPerBlock;
        return;
      }
      if (v != TileVerdict::None && rejected.insert(k).second) ++out.stats.tileRejected;
      // Discarding contents something later in the frame reads would lose them (the classic broken
      // G-buffer): store them anyway.
      if (u.last > static_cast<int>(e) && (a.store == StoreAction::DontCare || a.store == StoreAction::Resolve)) {
        a.store = a.store == StoreAction::Resolve ? StoreAction::StoreAndResolve : StoreAction::Store;
        ++out.stats.storesKept;
      }
    });
  }
}

}  // namespace

bool CompiledFrame::skipped(uint32_t command, uint32_t draw) const {
  return std::binary_search(badDraws.begin(), badDraws.end(), DrawRef{command, draw});
}

CompiledFrame compileFrame(const Frame& frame, const ResourceTable* resources, const CompileOptions& options) {
  CompiledFrame out;
  const auto& cmds = frame.commands();
  out.stats.commands = static_cast<uint32_t>(cmds.size());
  std::vector<AccessList> accesses(cmds.size());
  Scheduler scheduler(out, resources, options);
  for (uint32_t i = 0; i < cmds.size(); ++i) {
    accesses[i] = accessesOf(cmds[i]);
    if (resources) {
      bool ok = true;
      for (const Tagged& t : accesses[i]) ok = ok && alive(t.access.range, *resources);
      if (!ok) {
        ++out.stats.dropped;
        continue;
      }
    }
    if (const auto* rp = std::get_if<RenderPass>(&cmds[i])) scheduler.renderPass(i, *rp, accesses[i]);
    else if (std::holds_alternative<Dispatch>(cmds[i])) scheduler.dispatch(i, accesses[i]);
    else scheduler.copy(i, accesses[i]);
  }
  scheduler.finish();
  if (resources) placeTiles(out, cmds, accesses, *resources);

  BarrierPlacer placer(out);
  for (uint32_t e = 0; e < out.encoders.size(); ++e) {
    const Encoder& enc = out.encoders[e];
    ++out.stats.cuts[static_cast<int>(enc.reason)];
    for (uint32_t k = 0; k < enc.commands.size(); ++k) {
      // Render encoders are one sync scope; compute items can each take a barrier.
      const uint32_t item = enc.kind == Encoder::Kind::Render ? 0 : k;
      for (const Tagged& t : accesses[enc.commands[k]]) placer.access(t.access, e, item);
    }
  }
  for (const Barrier& b : out.barriers) out.stats.intraBarriers += b.item > 0;
  return out;
}

}  // namespace vpgpu
