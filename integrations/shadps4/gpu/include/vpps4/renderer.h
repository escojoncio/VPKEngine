// The PS4 renderer of VPEngine: Liverpool draws and dispatches (vpps4::DrawState) -> gpu/core work.
//
// Per draw: colour/depth targets come from a target cache keyed by guest address and shape; each
// shader stage is lifted (gpu/shader/gcn) with the draw's real descriptors and translated to MSL
// once per program + state that changes its code; its bindings are resolved from user data and
// guest memory: buffers alias guest memory in place (one gpu/core buffer per guest mapping),
// textures come from a texture cache (linear ones uploaded from guest memory, re-uploaded after a
// CPU write), samplers from S#; per-draw constants (user data) go through an upload ring. Draws
// with the same targets share one render pass; gpu/core merges and orders the rest.
//
// Not yet: tiled texture detiling (tiled textures stay unfilled, counted), RectList (skipped,
// counted), fast-clear metadata, writing render targets back to guest memory.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "vpgpu/core/device.h"
#include "vpir/ir.h"
#include "vpmsl/msl.h"
#include "vpps4/state.h"

namespace vpps4 {

// A guest memory mapping: [base, base + size) at host pointer `host` (page aligned).
struct GuestRegion {
    uint64_t base = 0, size = 0;
    void* host = nullptr;
};

struct Hooks {
    // The guest mapping that contains [address, address + size); false if there is none.
    std::function<bool(uint64_t address, uint64_t size, GuestRegion&)> region;
    // A translated shader, under the id pipelines refer to it by (register it with the backend's
    // shader provider before the pipeline compiles).
    std::function<void(uint64_t id, vpir::Stage, const vpmsl::MslShader&)> shader;
    std::function<void(const std::string&)> log;
};

struct Stats {
    uint64_t draws = 0, drawn = 0, dispatches = 0, dispatched = 0, passes = 0, copies = 0, resolves = 0;
    uint64_t shaders = 0, shaderErrors = 0, textures = 0, uploads = 0, targets = 0;
    std::map<std::string, uint64_t> skipped;  // reason -> draws/dispatches left out
    std::map<std::string, uint64_t> notes;    // approximations taken
};

class Renderer {
public:
    Renderer(vpgpu::Device&, Hooks);
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void draw(const DrawState&);
    void dispatch(const DispatchState&);
    // CB_COLOR_CONTROL.MODE = RESOLVE: colour target `src` (CB0) resolved into `dst` (CB1). With
    // the host at 1 sample (VPKTAA instead of MSAA) it is a texture copy; formats must be identical (else skipped).
    void resolve(const ColorTarget& src, const ColorTarget& dst);
    // The CPU wrote guest memory: textures read from there are uploaded again on their next use.
    void invalidate(uint64_t address, uint64_t size);
    // Submits the frame recorded so far (and logs a summary line every `summaryEvery` frames).
    vpgpu::FrameReport submit();

    // The colour target most recently rendered at `address` (null if none): what a flip presents.
    vpgpu::TextureId colorTargetAt(uint64_t address) const;
    const Stats& stats() const { return stats_; }
    void setSummaryEvery(uint64_t frames) { summaryEvery_ = frames; }
    // Upper bound for target sample counts (power of two, >= 1; the host's MSAA setting). Every
    // attachment of a draw gets the same count: the smallest of its targets' and this (Metal needs
    // one count per pass; EQAA depth with more samples than colour drops to the colour's).
    void setMaxSamples(uint8_t samples);
    uint8_t maxSamples() const { return maxSamples_; }

    struct Impl;

private:
    vpgpu::Device& device_;
    Hooks hooks_;
    Stats stats_;
    uint64_t summaryEvery_ = 300, frame_ = 0;
    uint8_t maxSamples_ = 8;
    // Draws/dispatches recorded since the last submit. A host that never ends its frames (a present
    // path without the submit call) would grow one frame forever: past kRunawayCommands it is
    // submitted here and logged once (gpu.err what=frame_not_ended).
    static constexpr uint64_t kRunawayCommands = 50000;
    uint64_t sinceSubmit_ = 0;
    bool runawayTold_ = false;
    void guardRunaway();
    std::unique_ptr<Impl> impl_;
};

}  // namespace vpps4
