#include "vpgpu/core/frame_compiler.h"

#include <unordered_map>

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

namespace {

// An access plus whether it comes from a render attachment: attachment-vs-attachment overlaps
// inside one render encoder are ordered by the rasterizer and never force a cut.
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

bool alive(const ResourceRange& r, const ResourceTable& t) {
  if (r.kind == ResourceRange::Kind::Buffer) return t.get(BufferId{r.index, r.generation}) != nullptr;
  return t.get(TextureId{r.index, r.generation}) != nullptr;
}

AccessList accessesOf(const Command& cmd) {
  AccessList out;
  auto add = [&](const std::vector<Access>& list, bool attachment) {
    for (const Access& a : list) out.push_back({a, attachment});
  };
  if (const auto* rp = std::get_if<RenderPass>(&cmd)) {
    add(attachmentAccesses(rp->desc), true);
    for (const Draw& d : rp->draws) add(drawAccesses(d), false);
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
  explicit Scheduler(CompiledFrame& out) : out_(out) {}

  void renderPass(uint32_t i, const RenderPassDesc& desc, AccessList acc) {
    CutReason reason;
    if (open_ >= 0) {
      Encoder& r = out_.encoders[open_];
      if (!r.pass.sameTargets(desc)) reason = CutReason::AttachmentChange;
      else if (clears(desc)) reason = CutReason::ClearLoad;
      else if (conflicts(acc, acc_[open_])) reason = CutReason::Feedback;
      else if (conflicts(acc, sunkAcc_)) reason = CutReason::PinnedCopy;
      else return merge(i, desc, std::move(acc));
      closeRender();
    } else if (!haveLastRender_) {
      reason = CutReason::FrameStart;
    } else if (!lastRender_.sameTargets(desc)) {
      reason = CutReason::AttachmentChange;
    } else {
      reason = clears(desc) ? CutReason::ClearLoad : CutReason::ComputeBetween;
    }
    open_ = push(Encoder::Kind::Render, reason, i, std::move(acc));
    out_.encoders[open_].pass = desc;
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
  void merge(uint32_t i, const RenderPassDesc& desc, AccessList acc) {
    Encoder& r = out_.encoders[open_];
    // Loads come from the first pass; stores from the last one, which owns the final contents.
    for (int c = 0; c < kMaxColorAttachments; ++c) r.pass.colors[c].store = desc.colors[c].store;
    r.pass.depth.store = desc.depth.store;
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

}  // namespace

CompiledFrame compileFrame(const Frame& frame, const ResourceTable* resources) {
  CompiledFrame out;
  const auto& cmds = frame.commands();
  out.stats.commands = static_cast<uint32_t>(cmds.size());
  std::vector<AccessList> accesses(cmds.size());
  Scheduler scheduler(out);
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
    if (const auto* rp = std::get_if<RenderPass>(&cmds[i])) scheduler.renderPass(i, rp->desc, accesses[i]);
    else if (std::holds_alternative<Dispatch>(cmds[i])) scheduler.dispatch(i, accesses[i]);
    else scheduler.copy(i, accesses[i]);
  }
  scheduler.finish();

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
