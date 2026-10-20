#include "vpps4/lifetime.h"

#include <algorithm>
#include <variant>
#include <vector>

namespace vpps4 {

using namespace vp4gpu;

namespace {

uint64_t keyOf(TextureId t) { return uint64_t(t.generation) << 32 | t.index; }
uint64_t keyOf(const ResourceRange& r) { return uint64_t(r.generation) << 32 | r.index; }

struct Use {
    std::vector<uint32_t> touch;   // commands that use the texture in any way, ascending
    std::vector<uint32_t> attach;  // render passes with it as an attachment, ascending
    std::vector<uint32_t> reads;   // commands that read it (bindings, copies), ascending
    void add(std::vector<uint32_t>& v, uint32_t i) {
        if (v.empty() || v.back() != i) v.push_back(i);
    }
};

// The first element of `v` greater than `i`, or -1.
int64_t after(const std::vector<uint32_t>& v, uint32_t i) {
    auto it = std::upper_bound(v.begin(), v.end(), i);
    return it == v.end() ? -1 : int64_t(*it);
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
    std::unordered_map<uint64_t, Use> uses;

    // Pass 1: who uses what, where.
    auto textureAccess = [&](const Access& a, uint32_t i) {
        if (a.range.kind != ResourceRange::Kind::Texture) return;
        Use& u = uses[keyOf(a.range)];
        u.add(u.touch, i);
        if (!a.write) u.add(u.reads, i);
    };
    for (uint32_t i = 0; i < cmds.size(); ++i) {
        if (const auto* rp = std::get_if<RenderPass>(&cmds[i])) {
            auto slot = [&](const Attachment& a) {
                if (!a.texture) return;
                Use& u = uses[keyOf(a.texture)];
                u.add(u.touch, i);
                u.add(u.attach, i);
                if (a.resolve) {
                    Use& r = uses[keyOf(a.resolve)];
                    r.add(r.touch, i);  // written by the pass
                }
            };
            for (const Attachment& a : rp->desc.colors) slot(a);
            slot(rp->desc.depth);
            for (const Draw& d : rp->draws)
                for (const Access& a : drawAccesses(d)) textureAccess(a, i);
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
    for (auto& [k, u] : uses) {
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
            Use& u = uses[keyOf(a.texture)];
            Life& life = lifeOf(a.texture);
            const bool ms = isMs(res, a.texture);
            // A foveated pass folds and expands its attachments through shadow textures: they must live in
            // memory (memoryless ones decline), so its stores stay.
            if (rp->desc.rateMap) {
                if (a.store == StoreAction::DontCare) a.store = StoreAction::Store;
                return;
            }

            // What the rest of the frame needs of the contents after this pass.
            const int64_t next = after(u.touch, i);
            bool neededInFrame = false;
            if (next >= 0) {
                neededInFrame = true;
                if (auto* np = std::get_if<RenderPass>(&cmds[size_t(next)])) {
                    // The next user clears it (or does not load it) and nobody reads it in that pass.
                    bool overwrites = false, readsIt = std::binary_search(u.reads.begin(), u.reads.end(), uint32_t(next));
                    auto check = [&](const Attachment& b) {
                        if (b.texture == a.texture && b.load != LoadAction::Load) overwrites = true;
                    };
                    for (const Attachment& b : np->desc.colors) check(b);
                    check(np->desc.depth);
                    neededInFrame = !overwrites || readsIt;
                }
            }

            // Across frames: the next frame depends on these contents unless the texture is
            // cleared at the start of every frame (this one and the previous one) and nobody
            // outside reads it.
            bool clearedNow = false;
            if (!u.attach.empty()) {
                const auto* fp = std::get_if<RenderPass>(&cmds[u.attach.front()]);
                auto check = [&](const Attachment& b) {
                    if (b.texture == a.texture && b.load == LoadAction::Clear) clearedNow = true;
                };
                for (const Attachment& b : fp->desc.colors) check(b);
                check(fp->desc.depth);
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
            if (a.resolve) {
                Life& rl = lifeOf(a.resolve);
                const Use& ru = uses[keyOf(a.resolve)];
                bool readLater = false;
                if (!ru.reads.empty()) {
                    // A read of the companion before this texture's next pass.
                    const int64_t nextAttach = after(u.attach, i);
                    const int64_t firstRead = after(ru.reads, i);
                    readLater = firstRead >= 0 && (nextAttach < 0 || firstRead < nextAttach);
                }
                const bool wanted = recent(rl.lastRead) || rl.sticky || warm(rl) || readLater;
                const bool lastPass = after(u.attach, i) < 0;
                resolve = wanted && (lastPass || readLater);
                resolveDeferred = wanted && !resolve;
            }
            const bool wasResolve = a.store == StoreAction::Resolve || a.store == StoreAction::StoreAndResolve;
            if (resolve && keepSamples) a.store = StoreAction::StoreAndResolve;
            else if (resolve) a.store = StoreAction::Resolve;
            else if (keepSamples) a.store = StoreAction::Store;
            else a.store = StoreAction::DontCare;
            if (!resolve) a.resolve = {};
            if (!keepSamples && next < 0) ++st.samplesDropped;
            if (wasResolve && !resolve) {
                if (resolveDeferred) ++st.resolvesDeferred;
                else ++st.resolvesDropped;
            }
        };
        for (Attachment& a : rp->desc.colors) decide(a);
        decide(rp->desc.depth);
    }

    // End of the frame: what the next one will find.
    for (auto& [k, u] : uses) {
        if (u.attach.empty()) continue;
        auto it = life_.find(k);
        if (it == life_.end()) continue;
        Life& life = it->second;
        const auto* fp = std::get_if<RenderPass>(&cmds[u.attach.front()]);
        const auto* lp = std::get_if<RenderPass>(&cmds[u.attach.back()]);
        auto find = [&](const RenderPass* p, bool last) -> const Attachment* {
            const Attachment* found = nullptr;
            auto check = [&](const Attachment& b) {
                if (b.texture && keyOf(b.texture) == k) found = &b;
            };
            for (const Attachment& b : p->desc.colors) check(b);
            check(p->desc.depth);
            (void)last;
            return found;
        };
        const Attachment* first = find(fp, false);
        const Attachment* final = find(lp, true);
        life.clearedLast = first && first->load == LoadAction::Clear;
        life.lost = final && (final->store == StoreAction::DontCare || final->store == StoreAction::Resolve);
    }
    return st;
}

}  // namespace vpps4
