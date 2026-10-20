#include "vpps4/lifetime.h"

#include <algorithm>
#include <iterator>
#include <variant>
#include <vector>
#include <tuple>

namespace vpps4 {

using namespace vp4gpu;

namespace {

uint64_t keyOf(TextureId t) { return uint64_t(t.generation) << 32 | t.index; }
uint64_t keyOf(const ResourceRange& r) { return uint64_t(r.generation) << 32 | r.index; }

// The first element of `v` greater than `i`, or -1.
int64_t after(const std::vector<uint32_t>& v, uint32_t i) {
    auto it = std::upper_bound(v.begin(), v.end(), i);
    return it == v.end() ? -1 : int64_t(*it);
}

bool sameSub(const Attachment& b, const Attachment& a) {
    return b.texture == a.texture && b.mip == a.mip && b.layer == a.layer;
}

// The attachment of `p` that is subresource (mip, layer) of a.texture, or null.
const Attachment* findSub(const RenderPass& p, const Attachment& a) {
    for (const Attachment& b : p.desc.colors)
        if (sameSub(b, a)) return &b;
    return sameSub(p.desc.depth, a) ? &p.desc.depth : nullptr;
}

// A pass that only attaches OTHER subresources of a.texture (another face of a shadow cube, the other
// eye): it neither overwrites nor needs the contents of this one.
bool onlyOtherSubs(const RenderPass& p, const Attachment& a) {
    bool other = false;
    for (const Attachment& b : p.desc.colors) {
        if (b.resolve == a.texture) return false;
        if (b.texture == a.texture) {
            if (sameSub(b, a)) return false;
            other = true;
        }
    }
    if (p.desc.depth.resolve == a.texture) return false;
    if (p.desc.depth.texture == a.texture) {
        if (sameSub(p.desc.depth, a)) return false;
        other = true;
    }
    return other;
}

bool isMs(const ResourceTable& res, TextureId t) {
    const TextureDesc* d = res.get(t);
    return d && d->samples > 1;
}

}  // namespace

void LifetimeTracker::noteRead(TextureId t, uint64_t frame) {
    if (!t) return;
    Life& l = life_[key(t)];
    if (l.born == ~uint64_t(0)) l.born = frame;
    l.lastRead = int64_t(frame);
}

void LifetimeTracker::noteSticky(TextureId t, uint64_t frame) {
    if (!t) return;
    noteRead(t, frame);
    life_[key(t)].sticky = true;
}

void LifetimeTracker::noteStale(TextureId t, uint64_t frame) {
    if (!t) return;
    Life& l = life_[key(t)];
    if (l.born == ~uint64_t(0)) l.born = frame;
    l.lost = true;
}

void LifetimeTracker::noteSamplesRead(TextureId ms, uint64_t frame) {
    if (!ms) return;
    Life& l = life_[key(ms)];
    if (l.born == ~uint64_t(0)) l.born = frame;
    l.lastSamplesRead = int64_t(frame);
}

bool LifetimeTracker::wants(TextureId t, uint64_t frame) const {
    auto it = life_.find(key(t));
    if (it == life_.end() || it->second.born == ~uint64_t(0)) return true;
    const Life& l = it->second;
    return l.sticky || frame - l.born < opt_.warmupFrames || int64_t(frame) - l.lastRead <= int64_t(opt_.holdFrames);
}

bool LifetimeTracker::sticky(TextureId t) const {
    auto it = life_.find(key(t));
    return it != life_.end() && it->second.sticky;
}

bool LifetimeTracker::contentsLost(TextureId t) const {
    auto it = life_.find(key(t));
    return it != life_.end() && it->second.lost;
}

LifetimeStats LifetimeTracker::plan(Frame& frame, const ResourceTable& res, uint64_t frameIndex) {
    LifetimeStats st;
    std::vector<Command>& cmds = frame.mutableCommands();
    const uint64_t serial = ++planSerial_;

    // Textures destroyed by the host leave their entries behind: sweep now and then.
    if (serial % 128 == 0) {
        for (auto it = life_.begin(); it != life_.end();) {
            const TextureId id{uint32_t(it->first & 0xffffffffu), uint32_t(it->first >> 32)};
            it = res.get(id) ? std::next(it) : life_.erase(it);
        }
        for (auto it = uses_.begin(); it != uses_.end();) {
            const TextureId id{uint32_t(it->first & 0xffffffffu), uint32_t(it->first >> 32)};
            it = res.get(id) ? std::next(it) : uses_.erase(it);
        }
    }

    // The entries of uses_ are reused between frames: one is valid when its stamp is this plan's.
    auto useOf = [&](uint64_t k) -> Use& {
        Use& u = uses_[k];
        if (u.stamp != serial) {
            u.touch.clear();
            u.attach.clear();
            u.reads.clear();
            u.stamp = serial;
        }
        return u;
    };
    auto usedNow = [&](uint64_t k) -> const Use* {
        auto it = uses_.find(k);
        return it != uses_.end() && it->second.stamp == serial ? &it->second : nullptr;
    };

    // Pass 1: who uses what, where.
    auto textureAccess = [&](const Access& a, uint32_t i) {
        if (a.range.kind != ResourceRange::Kind::Texture) return;
        Use& u = useOf(keyOf(a.range));
        u.add(u.touch, i);
        if (!a.write) u.add(u.reads, i);
    };
    for (uint32_t i = 0; i < cmds.size(); ++i) {
        if (const auto* rp = std::get_if<RenderPass>(&cmds[i])) {
            auto slot = [&](const Attachment& a) {
                if (!a.texture) return;
                Use& u = useOf(keyOf(a.texture));
                u.add(u.touch, i);
                u.add(u.attach, i);
                if (a.resolve) {
                    Use& r = useOf(keyOf(a.resolve));
                    r.add(r.touch, i);  // written by the pass
                }
            };
            for (const Attachment& a : rp->desc.colors) slot(a);
            slot(rp->desc.depth);
            // (only the bindings hold textures: no copy of the draw's access list)
            for (const Draw& d : rp->draws)
                for (const Access& a : d.bindings) textureAccess(a, i);
        } else if (const auto* dp = std::get_if<Dispatch>(&cmds[i])) {
            for (const Access& a : dp->bindings) textureAccess(a, i);
        } else {
            for (const Access& a : copyAccesses(std::get<Copy>(cmds[i]))) textureAccess(a, i);
        }
    }

    auto lifeOf = [&](TextureId t) -> Life& {
        Life& l = life_[key(t)];
        if (l.born == ~uint64_t(0)) l.born = frameIndex;
        return l;
    };
    auto warm = [&](const Life& l) { return frameIndex - l.born < opt_.warmupFrames; };
    auto recent = [&](int64_t when) { return int64_t(frameIndex) - when <= int64_t(opt_.holdFrames); };

    // Predictions that failed in the previous frames: contents dropped, read now before being written.
    for (auto& [k, u] : uses_) {
        if (u.stamp != serial) continue;
        auto it = life_.find(k);
        if (it == life_.end() || !it->second.lost || u.touch.empty()) continue;
        const uint32_t first = u.touch.front();
        bool readFirst = !u.reads.empty() && u.reads.front() <= first;
        bool loadsFirst = false;
        if (!u.attach.empty() && u.attach.front() == first) {
            const auto* rp = std::get_if<RenderPass>(&cmds[first]);
            auto check = [&](const Attachment& a) {
                if (a.texture && keyOf(a.texture) == k && a.load == LoadAction::Load) loadsFirst = true;
            };
            for (const Attachment& a : rp->desc.colors) check(a);
            check(rp->desc.depth);
        }
        if (readFirst || loadsFirst) {
            it->second.sticky = true;
            ++st.misses;
        }
    }

    // Pass 2: the actions.
    for (uint32_t i = 0; i < cmds.size(); ++i) {
        auto* rp = std::get_if<RenderPass>(&cmds[i]);
        if (!rp) continue;
        auto decide = [&](Attachment& a) {
            if (!a.texture) return;
            const Use& u = *usedNow(keyOf(a.texture));
            Life& life = lifeOf(a.texture);
            const bool ms = isMs(res, a.texture);
            // A foveated pass folds and expands its attachments through shadow textures: they must live in
            // memory (memoryless ones decline), so its stores stay.
            if (rp->desc.rateMap) {
                if (a.store == StoreAction::DontCare) a.store = StoreAction::Store;
                return;
            }

            // The next command that matters to THIS subresource: passes that only attach other layers or
            // mips of the texture (per-face shadow passes, one pass per eye) are skipped.
            int64_t next = -1;
            for (auto it = std::upper_bound(u.touch.begin(), u.touch.end(), i); it != u.touch.end(); ++it) {
                const auto* np = std::get_if<RenderPass>(&cmds[*it]);
                if (np && onlyOtherSubs(*np, a) && !std::binary_search(u.reads.begin(), u.reads.end(), *it)) continue;
                next = *it;
                break;
            }
            // The next pass attaching this very subresource.
            int64_t nextAttach = -1;
            for (auto it = std::upper_bound(u.attach.begin(), u.attach.end(), i); it != u.attach.end(); ++it)
                if (findSub(std::get<RenderPass>(cmds[*it]), a)) {
                    nextAttach = *it;
                    break;
                }

            // What the rest of the frame needs of the contents after this pass.
            bool neededInFrame = false;
            if (next >= 0) {
                neededInFrame = true;
                if (auto* np = std::get_if<RenderPass>(&cmds[size_t(next)])) {
                    // The next user clears it (or does not load it) and nobody reads it in that pass.
                    const bool readsIt = std::binary_search(u.reads.begin(), u.reads.end(), uint32_t(next));
                    const Attachment* b = findSub(*np, a);
                    const bool overwrites = b && b->load != LoadAction::Load;
                    neededInFrame = !overwrites || readsIt;
                }
            }

            // Across frames: the next frame depends on these contents unless the texture is
            // cleared at the start of every frame (this one and the previous one) and nobody
            // outside reads it.
            bool clearedNow = false;
            for (uint32_t j : u.attach) {
                if (const Attachment* b = findSub(std::get<RenderPass>(cmds[j]), a)) {
                    clearedNow = b->load == LoadAction::Clear;
                    break;
                }
            }
            const bool stable = clearedNow && life.clearedLast;
            const bool crossFrame = warm(life) || life.sticky || !stable;

            if (!ms) {
                const bool read = recent(life.lastRead);
                const bool keep = neededInFrame || crossFrame || read;
                if (!keep && a.store != StoreAction::DontCare) ++st.storesDropped;
                if (keep && next < 0) ++st.storesKept;
                a.store = keep ? StoreAction::Store : StoreAction::DontCare;
                a.resolve = {};
                return;
            }

            // Multisampled: samples and the single-sample companion are decided apart.
            const bool samplesRead = recent(life.lastSamplesRead);
            const bool keepSamples = neededInFrame || samplesRead || crossFrame;
            bool resolve = false;
            bool resolveDeferred = false;
            const TextureId twin = a.resolve;
            if (twin) {
                Life& rl = lifeOf(twin);
                const Use* ru = usedNow(keyOf(twin));
                bool readLater = false;
                if (ru && !ru->reads.empty()) {
                    // A read of the companion before this subresource's next pass. A read INSIDE that pass
                    // comes before its resolve (the resolve happens when the pass ends), hence `<=`.
                    const int64_t firstRead = after(ru->reads, i);
                    readLater = firstRead >= 0 && (nextAttach < 0 || firstRead <= nextAttach);
                }
                const bool wanted = recent(rl.lastRead) || rl.sticky || warm(rl) || readLater;
                const bool lastPass = nextAttach < 0;
                resolve = wanted && (lastPass || readLater);
                resolveDeferred = wanted && !resolve;
            }
            const bool wasResolve = a.store == StoreAction::Resolve || a.store == StoreAction::StoreAndResolve;
            if (resolve && keepSamples) a.store = StoreAction::StoreAndResolve;
            else if (resolve) a.store = StoreAction::Resolve;
            else if (keepSamples) a.store = StoreAction::Store;
            else a.store = StoreAction::DontCare;
            if (!resolve) a.resolve = {};
            if (twin && !resolveDeferred) {
                // The companion is never an attachment, so no end-of-frame bookkeeping sees it: a dropped
                // resolve leaves it stale (a read before its next resolve is a miss), a done one fresh.
                lifeOf(twin).lost = !resolve;
            }
            if (!keepSamples && next < 0) ++st.samplesDropped;
            if (wasResolve && !resolve) {
                if (resolveDeferred) ++st.resolvesDeferred;
                else ++st.resolvesDropped;
            }
        };
        for (Attachment& a : rp->desc.colors) decide(a);
        decide(rp->desc.depth);
    }

    // End of the frame: what the next one will find, per subresource and then per texture.
    subs_.clear();
    for (const Command& c : cmds) {
        const auto* p = std::get_if<RenderPass>(&c);
        if (!p) continue;
        auto note = [&](const Attachment& b) {
            if (!b.texture) return;
            const uint64_t k = keyOf(b.texture);
            for (SubUse& s : subs_)
                if (s.k == k && s.mip == b.mip && s.layer == b.layer) {
                    s.lastStore = b.store;
                    return;
                }
            subs_.push_back({k, b.mip, b.layer, b.load, b.store});
        };
        for (const Attachment& b : p->desc.colors) note(b);
        note(p->desc.depth);
    }
    for (size_t n = 0; n < subs_.size(); ++n) {
        const uint64_t k = subs_[n].k;
        bool seen = false;
        for (size_t m = 0; m < n; ++m) seen |= subs_[m].k == k;
        if (seen) continue;
        auto it = life_.find(k);
        if (it == life_.end()) continue;
        bool cleared = true, lost = false;
        for (const SubUse& s : subs_) {
            if (s.k != k) continue;
            cleared &= s.firstLoad == LoadAction::Clear;
            lost |= s.lastStore == StoreAction::DontCare || s.lastStore == StoreAction::Resolve;
        }
        it->second.clearedLast = cleared;
        it->second.lost = lost;
    }
    return st;
}

}  // namespace vpps4
