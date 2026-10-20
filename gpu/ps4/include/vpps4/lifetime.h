// Load/store planning of a recorded frame for a tile-based GPU (Apple TBDR).
//
// The renderer records every render pass with the conservative actions (Load, StoreAndResolve): it
// cannot know while recording what the rest of the frame, or the next frames, will read. At submit
// the whole frame is known, and this planner rewrites the actions of its render passes:
//
//   * a store is dropped (DontCare; MSAA samples: Resolve only) when nothing reads the contents
//     later in the frame (draws, copies, dispatches, a later pass that loads them) and nothing has
//     read them across frames: the texture was not sampled or presented in the last `holdFrames`
//     frames, is not "sticky", and its first pass cleared it this frame and the previous one (so the
//     next frame does not depend on this frame's contents);
//   * a multisampled colour target resolves into its single-sample companion only if somebody
//     looks at the companion (sampled/presented lately, or read later in the frame) or nobody reads
//     the samples themselves; it keeps its samples only if a shader reads them (texture2d_ms) or a
//     later pass loads them;
//   * only the last pass of a frame that draws into a texture resolves it, unless the companion is
//     read in between;
//   * a prediction that fails (a texture whose contents were dropped is read before it is written)
//     makes the texture "sticky": stored from then on, counted as a miss.
//
// With the dropped stores and clear loads the frame compiler (gpu/core) can keep a texture on tile
// (memoryless): that needs TextureDesc::persistent == false.
#pragma once

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include "vp4gpu/core/commands.h"

namespace vpps4 {

struct LifetimeOptions {
    uint32_t holdFrames = 2;    // a texture read in the last N frames keeps being stored
    uint32_t warmupFrames = 3;  // a texture younger than this is always stored
    // A multisampled target resolves into its single-sample twin when that twin was read within this many
    // frames (a much longer memory than holdFrames: a resolve costs a little bandwidth, a picture that
    // goes black costs the frame).
    uint32_t resolveHoldFrames = 60;
};

struct LifetimeStats {
    uint32_t storesDropped = 0;     // attachments whose final store became DontCare
    uint32_t samplesDropped = 0;    // multisampled attachments whose samples are not stored (Resolve)
    uint32_t resolvesDropped = 0;   // resolves not done (nobody looks at the single-sample copy)
    uint32_t resolvesDeferred = 0;  // resolves left to a later pass of the frame
    uint32_t storesKept = 0;        // final stores kept for the next frames
    uint32_t misses = 0;            // contents read after they were dropped (the texture turns sticky)
};

class LifetimeTracker {
public:
    explicit LifetimeTracker(LifetimeOptions o = {}) : opt_(o) {}

    // A shader, a copy or the present reads `t` (the host keeps calling it as long as it does).
    void noteRead(vp4gpu::TextureId t, uint64_t frame);
    // The same, and `t` must always be stored from now on (presented surfaces, stereo eye textures).
    void noteSticky(vp4gpu::TextureId t, uint64_t frame);
    // `t` (the single-sample picture of a multisampled target) was left without this frame's resolve: its
    // contents are stale. Reading it before the next resolve is a miss (it turns sticky).
    void noteStale(vp4gpu::TextureId t, uint64_t frame);
    // A shader reads the samples of multisampled `ms` (texture2d_ms).
    void noteSamplesRead(vp4gpu::TextureId ms, uint64_t frame);

    // Rewrites load/store/resolve of the render passes of `frame` (see the file comment). `frameIndex`
    // is the renderer's frame counter.
    LifetimeStats plan(vp4gpu::Frame& frame, const vp4gpu::ResourceTable& res, uint64_t frameIndex);

    // Whether `t` has to be kept up to date now: young, read lately, or sticky.
    bool wants(vp4gpu::TextureId t, uint64_t frame) const;

    // Told (once per pass) when a multisampled attachment loses its resolve: `ms` is the attachment, `twin`
    // the single-sample companion it would have resolved into (null if it has none), `why` a short reason.
    using ResolveDropSink = std::function<void(vp4gpu::TextureId ms, vp4gpu::TextureId twin, const char* why)>;
    void setResolveDropSink(ResolveDropSink s) { dropSink_ = std::move(s); }

    // For tests and logs.
    bool sticky(vp4gpu::TextureId t) const;
    bool contentsLost(vp4gpu::TextureId t) const;

private:
    struct Life {
        uint64_t born = ~uint64_t(0);
        int64_t lastRead = -1000, lastSamplesRead = -1000;
        bool sticky = false;
        bool clearedLast = false;   // the first pass of the previous frame cleared it
        bool lost = false;          // the previous frame ended with its contents dropped
    };
    // Who uses a texture in the frame being planned (the vectors keep their capacity between frames).
    struct Use {
        std::vector<uint32_t> touch;   // commands that use the texture in any way, ascending
        std::vector<uint32_t> attach;  // render passes with it as an attachment, ascending
        std::vector<uint32_t> reads;   // commands that read it (bindings, copies), ascending
        uint64_t stamp = ~uint64_t(0); // plan() that filled it
        void add(std::vector<uint32_t>& v, uint32_t i) {
            if (v.empty() || v.back() != i) v.push_back(i);
        }
    };
    // First load and last store of one subresource of an attachment over the frame.
    struct SubUse {
        uint64_t k;
        uint16_t mip, layer;
        vp4gpu::LoadAction firstLoad;
        vp4gpu::StoreAction lastStore;
    };
    static uint64_t key(vp4gpu::TextureId t) { return uint64_t(t.generation) << 32 | t.index; }
    LifetimeOptions opt_;
    std::unordered_map<uint64_t, Life> life_;
    std::unordered_map<uint64_t, Use> uses_;
    std::vector<SubUse> subs_;
    uint64_t planSerial_ = 0;
    ResolveDropSink dropSink_;
};

}  // namespace vpps4
