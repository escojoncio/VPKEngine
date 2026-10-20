// The PS4 renderer of VPKEngine: Liverpool draws and dispatches (vpps4::DrawState) -> gpu/core work.
//
// Per draw: colour/depth targets come from a target cache keyed by guest address and shape; each
// shader stage is lifted (gpu/shader/gcn) with the draw's real descriptors and translated to MSL
// once per program + state that changes its code; its bindings are resolved from user data and
// guest memory: buffers alias guest memory in place (one gpu/core buffer per guest mapping),
// textures come from a texture cache (linear ones uploaded from guest memory, re-uploaded after a
// CPU write), samplers from S#; per-draw constants (user data) go through an upload ring. Draws
// with the same targets share one render pass; gpu/core merges and orders the rest.
//
// Memory coherence (memory_tracker.h): every image is a copy of a guest range; writes by passes,
// dispatches, DMA and the CPU are versioned when recorded, an image out of date is refilled before it
// is used and one ahead of memory is written back before anything reads the bytes raw. Tiled
// textures are detiled with every mip level and layer, and tiled render targets / depth written back,
// on the GPU timeline by the GpuTiler kernels (CPU detile while those compile, gpu_tiler.h).
// Not yet: RectList (skipped, counted), fast-clear metadata.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "vp4gpu/core/device.h"
#include "vp4ir/ir.h"
#include "vp4msl/msl.h"
#include "vpps4/formats.h"
#include "vpps4/memory_tracker.h"
#include "vpps4/state.h"

namespace vpps4 {

// A guest memory mapping: [base, base + size) at host pointer `host` (page aligned).
struct GuestRegion {
    uint64_t base = 0, size = 0;
    void* host = nullptr;
    // `host` is a second mapping of the same pages (not the guest address itself): the guest
    // addresses may then be page-protected to catch CPU writes (Hooks::watch) without breaking the
    // no-copy buffers, which alias the mirror.
    bool mirrored = false;
};

// A view of a texture the renderer made: another format of the same texel size, a layer range,
// or another dimensionality (a T# naming a render target as an array slice, a cube, ...).
struct ViewRequest {
    vp4gpu::TextureId base;
    vp4gpu::Format format = vp4gpu::Format::Undefined;  // Undefined: the base's
    vp4gpu::TextureType type = vp4gpu::TextureType::Tex2D;
    bool array = false;
    uint16_t baseLayer = 0, layers = 1;
};

struct Hooks {
    // The guest mapping that contains [address, address + size); false if there is none.
    std::function<bool(uint64_t address, uint64_t size, GuestRegion&)> region;
    // A translated shader, under the id pipelines refer to it by (register it with the backend's
    // shader provider before the pipeline compiles).
    std::function<void(uint64_t id, vp4ir::Stage, const vp4msl::MslShader&)> shader;
    std::function<void(const std::string&)> log;
    // A texture was filled from guest memory [address, address + bytes) (optional; informational:
    // CPU writes are found by the renderer itself, comparing the source's content).
    std::function<void(uint64_t address, uint64_t bytes)> uploaded;
    // Optional views (the renderer samples other formats / layers of an image through them; without
    // the hook it falls back to the image itself and counts a note). destroyView is called before
    // the base texture is destroyed.
    std::function<vp4gpu::TextureId(const ViewRequest&)> view;
    std::function<void(vp4gpu::TextureId)> destroyView;
    // Optional CPU access detection, used for ranges of mirrored regions only (GuestRegion::mirrored):
    //  watchWrites(address, bytes): an image was filled from this guest range; the host calls
    //    Renderer::invalidate when the CPU writes to it and drops the watch (the renderer asks
    //    again at the next fill). With it the renderer stops hashing the source.
    //  watchOwned(address, bytes, on): an image holds newer bytes than this range of memory; the
    //    host must call Renderer::syncToMemory (then submit and wait) before the CPU reads or
    //    writes it, and may stop protecting it once that is done.
    std::function<void(uint64_t address, uint64_t bytes)> watchWrites;
    std::function<void(uint64_t address, uint64_t bytes, bool on)> watchOwned;
};

struct Stats {
    uint64_t draws = 0, drawn = 0, dispatches = 0, dispatched = 0, passes = 0, copies = 0, resolves = 0;
    uint64_t shaders = 0, shaderErrors = 0, textures = 0, uploads = 0, targets = 0, detiled = 0, indirect = 0;
    uint64_t gpuDetiled = 0;  // of `detiled`, those done by the GPU kernels (GpuTiler)
    uint64_t gpuRetiled = 0;  // render targets written back to guest memory (writeBack)
    // Memory coherence: images refilled from memory after another writer wrote their range, images
    // written back to memory before a raw read / partial raw write / CPU read, images evicted.
    uint64_t refilled = 0, retiled = 0, evicted = 0, viewsMade = 0, syncs = 0;
    std::map<std::string, uint64_t> skipped;  // reason -> draws/dispatches left out
    std::map<std::string, uint64_t> notes;    // approximations taken
};

// CPU time spent inside the renderer (steady_clock), accumulated until takeCpuTimes().
struct CpuTimes {
    double drawMs = 0;      // Renderer::draw + dispatch, whole call
    double shaderMs = 0;    // shader() lookups (program hash, translation cache)
    double bindMs = 0;      // bind(): descriptors, buffers, textures, samplers, user data
    double pipelineMs = 0;  // pipeline cache request
    double submitMs = 0;    // Device::submit: frame compile + backend encode/commit
    uint64_t calls = 0;     // draws + dispatches
    uint64_t submits = 0, emptySubmits = 0;
    uint64_t userDataReused = 0;  // user data uploads served from an identical earlier upload
};

class Renderer {
public:
    Renderer(vp4gpu::Device&, Hooks);
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void draw(const DrawState&);
    void dispatch(const DispatchState&);
    // CB_COLOR_CONTROL.MODE = RESOLVE: colour target `src` (CB0) resolved into `dst` (CB1). With
    // the host at 1 sample (VPKTAA instead of MSAA) it is a texture copy; formats must be identical (else skipped).
    void resolve(const ColorTarget& src, const ColorTarget& dst);
    // Writes the render target last drawn at `c.address` back to guest memory in the layout the CB names
    // (any tile mode; one sample plane), as a GPU kernel on the timeline. False if there is no such
    // target, it is multisampled on the host, or the kernel is still compiling (call again later).
    bool writeBack(const ColorTarget& c);
    // DMA on the GPU timeline (recorded in order with the draws): fill guest memory with a 32-bit
    // value, copy guest memory to guest memory. False (logged) if a range is not guest memory.
    bool fill(uint64_t address, uint64_t bytes, uint32_t value);
    bool copy(uint64_t dst, uint64_t src, uint64_t bytes);
    // The CPU wrote guest memory: images filled from there are filled again on their next use.
    void invalidate(uint64_t address, uint64_t size);
    // The CPU (or a host-side DMA, a label read, a readback) is about to read or write guest
    // [address, +size): images that hold newer bytes are written back to memory (re-tiled) on the GPU
    // timeline. True if work was recorded: submit and wait for the GPU before touching the bytes.
    bool syncToMemory(uint64_t address, uint64_t size);
    // Work the renderer did not record itself wrote [address, +size) (a host-side GPU write):
    // images over it are out of date.
    void noteGpuWrite(uint64_t address, uint64_t size);
    // Residency: images unused for `idleSubmits` submits, or the least recently used ones past
    // `byteBudget` of GPU memory, are written back (when they own newer bytes) and destroyed.
    void setResidency(uint64_t idleSubmits, uint64_t byteBudget);
    const MemoryTracker& tracker() const;
    // Whether work recorded since the last gpuIdle() may read or write guest [address, +size).
    // Ranges of submitted frames retire as the GPU finishes them (Device::completedFrames).
    bool gpuMayAccess(uint64_t address, uint64_t size) const;
    // Only the work recorded since the last submit (not yet submitted) may touch the range: a DMA
    // that overlaps nothing else needs a submit before waiting; otherwise waiting is enough.
    bool recordedMayAccess(uint64_t address, uint64_t size) const;
    // The caller waited for every submitted frame (not for work recorded since): their ranges retire.
    void submittedIdle();
    // The caller waited for the GPU: nothing recorded so far is still pending.
    void gpuIdle();
    // Submits the frame recorded so far (and logs a summary line every `summaryEvery` frames).
    vp4gpu::FrameReport submit();

    // The GDS (64 KiB, 16384 dwords) shaders' ds_append/ds_consume count in: the CPU may read or
    // write it once the GPU is idle (EVENT_WRITE_EOS GdsStore, DMA to/from GDS). Null if no memory.
    uint32_t* gds();
    // The texture a T# names, as a draw would sample it: the render target at its address, or
    // the texture filled from guest memory (recorded into the current frame: submit before use).
    vp4gpu::TextureId textureFor(const ImageDesc&);
    // The colour target most recently rendered at `address` (null if none): what a flip presents.
    vp4gpu::TextureId colorTargetAt(uint64_t address) const;
    // The array colour target one of whose slices past 0 starts at `address` (null if none), with
    // that slice in `layer`: a T# naming one eye of a two-slice stereo target.
    vp4gpu::TextureId colorTargetSlice(uint64_t address, uint32_t width, uint32_t height, uint16_t& layer) const;
    // Diagnostics: the GPU-side surfaces (colour/depth targets, shader-written textures) that
    // overlap guest range [address, address + bytes), one line ("nothing ..." if none).
    std::string describeAt(uint64_t address, uint64_t bytes) const;
    const Stats& stats() const { return stats_; }
    // The CPU time accumulated since the last call (then reset).
    CpuTimes takeCpuTimes();
    void setSummaryEvery(uint64_t frames) { summaryEvery_ = frames; }
    // Upper bound for target sample counts (power of two, >= 1; the host's MSAA setting). Every
    // attachment of a draw gets the same count: the smallest of its targets' and this (Metal needs
    // one count per pass; EQAA depth with more samples than colour drops to the colour's).
    void setMaxSamples(uint8_t samples);
    uint8_t maxSamples() const { return maxSamples_; }
    // Detile textures with the GpuTiler kernels (default), or always on the CPU (tests, diagnosis).
    void setGpuTiling(bool on);

    struct Impl;

private:
    vp4gpu::Device& device_;
    Hooks hooks_;
    Stats stats_;
    CpuTimes cpu_;
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
