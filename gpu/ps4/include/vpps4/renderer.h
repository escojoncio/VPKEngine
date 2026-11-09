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
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "vpgpu/core/device.h"
#include "vpir/ir.h"
#include "vpmsl/msl.h"
#include "vpps4/formats.h"
#include "vpps4/memory_tracker.h"
#include "vpps4/resolved.h"
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
    vpgpu::TextureId base;
    vpgpu::Format format = vpgpu::Format::Undefined;  // Undefined: the base's
    vpgpu::TextureType type = vpgpu::TextureType::Tex2D;
    bool array = false;
    uint16_t baseLayer = 0, layers = 1;
    uint16_t baseMip = 0, mips = 0;  // mip range of the view (mips 0: every level from baseMip); a T#'s base_level / last_level
};

// A buffer a compute program accesses: the base address of its V# (read from the dispatch's user data).
struct ComputeBuffer {
    uint64_t address = 0;
    bool written = false;  // the program writes it (it may read it too)
};

// A surface the renderer is about to use (Hooks::seed): a render target, a depth target or a texture, with the
// guest address it stands for and the format the renderer holds it in.
struct SeedRequest {
    enum class Kind : uint8_t { Color, Depth, Texture };
    Kind kind = Kind::Color;
    uint64_t address = 0;
    uint32_t width = 0, height = 0;
    vpgpu::Format format = vpgpu::Format::Undefined;
};
// Texels the host holds for it: `height` rows of `width` texels of the request's format, `rowBytes` apart (0: tight).
// A D32FloatS8 surface takes the Z plane (32-bit floats) from `data` and, if given, the stencil plane (bytes, `height`
// rows of `width`, `stencilRowBytes` apart: 0 tight) from `stencil`.
struct SeedImage {
    const uint8_t* data = nullptr;
    uint32_t rowBytes = 0;
    const uint8_t* stencil = nullptr;
    uint32_t stencilRowBytes = 0;
};

struct Hooks {
    // The guest mapping that contains [address, address + size); false if there is none.
    std::function<bool(uint64_t address, uint64_t size, GuestRegion&)> region;
    // Optional: a counter that changes whenever the host's answers of `region` may have changed (a
    // mapping made, removed or replaced; never otherwise). With it the renderer keeps the mirrored
    // regions it was given (a few of them, across draws) for as long as the value stays the same,
    // instead of asking `region` again in every draw. It is called often: keep it cheap and
    // thread-safe. Without it a region is kept within one draw / dispatch only.
    std::function<uint64_t()> regionGeneration;
    // A translated shader, under the id pipelines refer to it by (register it with the backend's
    // shader provider before the pipeline compiles).
    std::function<void(uint64_t id, vpir::Stage, const vpmsl::MslShader&)> shader;
    std::function<void(const std::string&)> log;
    // A texture was filled from guest memory [address, address + bytes) (optional; informational:
    // CPU writes are found by the renderer itself, comparing the source's content).
    std::function<void(uint64_t address, uint64_t bytes)> uploaded;
    // Optional views (the renderer samples other formats / layers of an image through them; without
    // the hook it falls back to the image itself and counts a note). destroyView is called before
    // the base texture is destroyed.
    std::function<vpgpu::TextureId(const ViewRequest&)> view;
    std::function<void(vpgpu::TextureId)> destroyView;
    // Optional CPU access detection, used for ranges of mirrored regions only (GuestRegion::mirrored):
    //  watchWrites(address, bytes): an image was filled from this guest range; the host calls
    //    Renderer::invalidate when the CPU writes to it and drops the watch (the renderer asks
    //    again at the next fill). With it the renderer stops hashing the source.
    //  watchOwned(address, bytes, on): an image holds newer bytes than this range of memory; the
    //    host must call Renderer::syncToMemory (then submit and wait) before the CPU reads or
    //    writes it, and may stop protecting it once that is done.
    std::function<void(uint64_t address, uint64_t bytes)> watchWrites;
    std::function<void(uint64_t address, uint64_t bytes, bool on)> watchOwned;
    // Optional: asked before a dispatch is recorded, with the buffers its program accesses through V#s in user
    // data and whether the program uses a bitwise xor (the address arithmetic of tiled surfaces has one; a
    // clear loop does not). `complete`: every buffer of the program is in the span (false: some V# could not be
    // read, or comes from somewhere other than user data / a pointer, and the program may touch memory the span
    // does not say). true: the host took the dispatch over (e.g. it only clears surface metadata the host
    // tracks): it is left out.
    std::function<bool(std::span<const ComputeBuffer>, bool usesXor, bool complete)> computeBuffers;
    // Optional: asked first; false (the host has nothing to look for) skips computeBuffers and the work to prepare it.
    std::function<bool()> computeBuffersWanted;
    // Optional diagnostics: why `region` has no span for `address` (unmapped, and the nearest mapped
    // region; mapped but protected without a mirror; relocated by the host...), one short line.
    std::function<std::string(uint64_t address)> explainAddress;
    // Optional, off without it: called at the end of every draw, dispatch, fill and copy (also when it was left
    // out) with what the renderer resolved for it (resolved.h). On the calling thread, inside the call.
    std::function<void(const Resolved&)> resolved;
    // Optional, off without it: asked each time a single-sample 2D surface of one level and one layer (not block
    // compressed; depth only D32Float / D16Unorm / D32FloatS8) is about to be used. True with `image` filled: the surface takes
    // those texels (copied before the call returns) instead of the ones from guest memory, and any clear still waiting
    // for it is dropped. False: the surface is filled from memory as usual. For hosts that compare this renderer with
    // another one draw by draw: both start from the same pixels.
    std::function<bool(const SeedRequest&, SeedImage& image)> seed;
};

struct Stats {
    uint64_t draws = 0, drawn = 0, dispatches = 0, dispatched = 0, passes = 0, copies = 0, resolves = 0;
    uint64_t gsHoisted = 0, gsSplit = 0;  // geometry expansions run above their pass / splitting it
    uint64_t shaders = 0, shadersCached = 0, shaderErrors = 0, textures = 0, uploads = 0, targets = 0, detiled = 0, indirect = 0, ringChains = 0;
    uint64_t gpuDetiled = 0;  // of `detiled`, those done by the GPU kernels (GpuTiler)
    uint64_t gpuRetiled = 0;  // render targets written back to guest memory (writeBack)
    // Memory coherence: images refilled from memory after another writer wrote their range, images
    // written back to memory before a raw read / partial raw write / CPU read, images evicted.
    uint64_t refilled = 0, retiled = 0, evicted = 0, viewsMade = 0, syncs = 0;
    uint64_t seeded = 0;  // surfaces filled from Hooks::seed
    std::map<std::string, uint64_t> skipped;  // reason -> draws/dispatches left out
    std::map<std::string, uint64_t> handled;  // reason -> draws that became something else than a pass draw (depth copy, clear)
    std::map<std::string, uint64_t> notes;    // approximations taken
};

// CPU time spent inside the renderer (steady_clock), accumulated until takeCpuTimes().
struct CpuTimes {
    double drawMs = 0;      // Renderer::draw + dispatch, whole call
    double shaderMs = 0;    // shader() lookups (program hash, translation cache)
    double bindMs = 0;      // bind(): descriptors, buffers, textures, samplers, user data
    double bindDescMs = 0, bindTexMs = 0, bindBufMs = 0, bindCbMs = 0, bindUploadMs = 0;  // phases of bind() (the rest of bindMs is the argument table)
    double pipelineMs = 0;  // pipeline cache request
    double submitMs = 0;    // the caller's time in Renderer::submit: lifetime plan + Device::submit (hold, compile, backend when inline)
    // Phases of it (Renderer::takeCpuTimes fills the Device's from Device::takeSubmitTimes, whose backend part runs on
    // the submit thread when there is one: then it is not part of submitMs).
    double lifeMs = 0;      // lifetime plan (load/store actions of the frame)
    double holdMs = 0;      // waiting for pipelines that were still compiling (Device hold)
    double compileMs = 0;   // frame compile
    double backendMs = 0;   // backend submit: encode + commit, including the wait for a frame slot
    double slotWaitMs = 0;  // of backendMs, waiting for a free frame slot
    uint64_t holdFrames = 0, holdTimeouts = 0;  // frames that waited for pipelines / of them, those that gave up at the budget
    uint64_t calls = 0;     // draws + dispatches
    uint64_t submits = 0, emptySubmits = 0;
    uint64_t userDataReused = 0;  // user data uploads served from an identical earlier upload
    uint64_t alignedCopyReused = 0;  // unaligned constant buffers served from a copy made earlier in the frame
    uint64_t regionHits = 0, regionMisses = 0;  // regions served by the cache across draws / asked of Hooks::region
    uint64_t resolveDestClearDropped = 0, resolveDestClearFlushed = 0;  // fast clears of a resolve destination: dropped (the resolve overwrites it all) / emitted before the resolve
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
    // Writes the render target last drawn at `c.address` back to guest memory in the layout the CB names
    // (any tile mode; one sample plane), as a GPU kernel on the timeline. False if there is no such
    // target, it is multisampled on the host, or the kernel is still compiling (call again later).
    bool writeBack(const ColorTarget& c);
    // DMA on the GPU timeline (recorded in order with the draws): fill guest memory with a 32-bit
    // value, copy guest memory to guest memory. False (logged) if a range is not guest memory.
    bool fill(uint64_t address, uint64_t bytes, uint32_t value);
    bool copy(uint64_t dst, uint64_t src, uint64_t bytes);
    // The HTILE of the depth buffer whose Z plane is at `zAddress` was cleared (a fill or a copy of cleared entries, a
    // kernel that wrote it): every slice holds `value` until a draw binds it. The hardware reads that value through a
    // texture of the surface (HTILE the texture unit understands); guest memory keeps whatever was there. A texture
    // filled from memory at that base takes the value for the slices not drawn into since.
    void depthCleared(uint64_t zAddress, float value);
    // The CPU wrote guest memory: images filled from there are filled again on their next use.
    void invalidate(uint64_t address, uint64_t size);
    // The title synchronised its caches over [address, +size) (ACQUIRE_MEM / SURFACE_SYNC): what the
    // host cannot watch there (shader code, textures without write watches) is looked at again.
    void coherencePoint(uint64_t address, uint64_t size);
    // The CPU (or a host-side DMA, a label read, a readback) is about to read or write guest
    // [address, +size): images that hold newer bytes are written back to memory (re-tiled) on the GPU
    // timeline. True if work was recorded: submit and wait for the GPU before touching the bytes.
    bool syncToMemory(uint64_t address, uint64_t size);
    // The images ahead of memory in [address, +size) are given up (their write back never happens): they stop
    // owning the range and release its pages. For a host that waited long enough; their bytes are lost to memory.
    void abandonOwners(uint64_t address, uint64_t size);
    // The host is about to unmap (or replace) guest memory it gave through Hooks::region: the buffers made over it (the
    // no-copy wraps of its windows) are given up, so none stays alive over unmapped pages and a later map at the same
    // address is wrapped afresh. Submit first; waits (up to 5 s) for the GPU and destroys them. False if the GPU did not
    // go idle in time: the buffers are retired as any replaced one, and the memory must not be unmapped.
    bool dropGuestBuffers();
    // Work the renderer did not record itself wrote [address, +size) (a host-side GPU write):
    // images over it are out of date.
    void noteGpuWrite(uint64_t address, uint64_t size);
    // Residency: images unused for `idleSubmits` submits, or the least recently used ones past
    // `byteBudget` of GPU memory, are written back (when they own newer bytes) and destroyed.
    void setResidency(uint64_t idleSubmits, uint64_t byteBudget);
    // Host memory the renderer may hold: dedicated staging (decoded block-compressed textures, big uploads; default 512 MiB,
    // 128 MiB on a host without BC), the geometry pipeline's scratch (default 768 MiB), and, on a host without BC, the bytes
    // of block-compressed textures decoded per frame (one submit; default 64 MiB; the first decode of a frame is always
    // admitted). A texture past a limit is not filled and tried again later; its draw is left out meanwhile.
    void setMemoryLimits(uint64_t stagingBytes, uint64_t geometryScratchBytes, uint64_t bcDecodedBytesPerFrame);
    const MemoryTracker& tracker() const;
    // Whether work recorded since the last gpuIdle() may read or write guest [address, +size).
    // Ranges of submitted frames retire as the GPU finishes them (Device::completedFrames).
    bool gpuMayAccess(uint64_t address, uint64_t size) const;
    // Only the work recorded since the last submit (not yet submitted) may touch the range: a DMA
    // that overlaps nothing else needs a submit before waiting; otherwise waiting is enough.
    bool recordedMayAccess(uint64_t address, uint64_t size) const;
    // The caller waited for every submitted frame (not for work recorded since): their ranges retire.
    void submittedIdle();
    // Returned by waitTarget when work not submitted yet may touch the range.
    static constexpr uint64_t kUnsubmitted = ~uint64_t(0);
    // What a CPU access to guest [a, +n) (and [b, +m)) must wait for: the number of frames that have to
    // finish (Device::waitFrames), 0 if no queued GPU work touches it, kUnsubmitted if the recorded
    // frame does (submit it first: then the target is Device::submittedFrames()).
    uint64_t waitTarget(uint64_t a, uint64_t n, uint64_t b = 0, uint64_t m = 0) const;
    // The caller waited for the frames below `frames`: their ranges retire.
    void waitedBelow(uint64_t frames);
    // Commands recorded and not submitted yet.
    bool hasRecordedWork() const;
    uint64_t recordedCommands() const;
    // Whether the renderer has ever read guest [a, +n) on the CPU (descriptors, indices, program code, texture sources, constant
    // copies). A DMA recorded on the GPU timeline lands later than a CPU read would see, so such a range is DMA'd by the CPU.
    bool cpuReadSeen(uint64_t a, uint64_t n) const;
    // The CPU side of the command processor (a DMA done on the CPU, DUMP_CONST_RAM, WRITE_DATA) had to wait for recorded
    // work to finish before touching [a, +n): the range counts as read by the CPU from now on, so a small read-only buffer
    // bound from it is copied when its draw is recorded (see cpuReadSeen) and the next such access finds nothing to wait for.
    void cpuAccessed(uint64_t a, uint64_t n);
    // Probe (first 50): a DMA over [a, +n) stays on the CPU because of cpuReadSeen: one DMA_CPU_READ line naming the first
    // readers of the ranges it overlaps (cbuf_copy, read_guest, code, ...).
    void probeDmaCpuRead(uint64_t a, uint64_t n, uint64_t src);
    // Probe (DMASYNC, needs vpgpu::probe enabled): the busy range a CPU access to [a, +n) (and [b, +m)) overlaps with who made
    // it ("" if none), and one access that waited `frames` frames (0: did not wait) for `ms` (counters, timer and a line per caller).
    std::string busyText(uint64_t a, uint64_t n, uint64_t b = 0, uint64_t m = 0) const;
    void noteCpuSync(const char* caller, uint64_t a, uint64_t n, uint64_t b, uint64_t m, const std::string& busy, uint64_t frames,
                     bool submitted, double ms);
    // Host address of GDS (as the busy ranges name it; 64 KiB): waitTarget(gdsAddress(), 0x10000) says
    // whether queued work uses it.
    uint64_t gdsAddress() const;
    // Runs `action` when the GPU has finished all the work recorded or submitted so far (a PM4 label
    // write: EOP / EOS / RELEASE_MEM). In order with every other afterGpu action. [address, +size) is
    // what it writes (labelPending / labelWaitTarget). Immediately if nothing is outstanding. `pureWrite`: the action only
    // writes that memory (vpgpu::LabelQueue::add).
    void afterGpu(uint64_t address, uint32_t size, std::function<void()> action, bool pureWrite = false);
    bool labelPending(uint64_t address, uint32_t size) const;
    uint64_t labelWaitTarget(uint64_t address, uint32_t size) const;
    // The caller waited for the GPU: nothing recorded so far is still pending.
    void gpuIdle();
    // Submits the frame recorded so far (and logs a summary line every `summaryEvery` frames).
    vpgpu::FrameReport submit();

    // EVENT_WRITE ZPASS_DONE: samples the cumulative count of samples that passed (all draws since
    // the first call) into guest memory, `pairs` slots of 16 bytes, the u64 at `address` of each
    // (bit 63 = written). The values land when the frame holding the draws has finished on the GPU:
    // call drainOcclusion after submits (gpuIdle = the caller submitted and waited for the GPU).
    // False: not counted (the caller writes a stand-in).
    bool zpassDone(uint64_t address, uint32_t pairs);
    void drainOcclusion(bool gpuIdle);
    // Whether a ZPASS_DONE result overlapping guest [address, +size) is still to be written.
    bool occlusionPending(uint64_t address, uint64_t size) const;

    // The GDS (64 KiB, 16384 dwords) shaders' ds_append/ds_consume count in: the CPU may read or
    // write it once the GPU is idle (EVENT_WRITE_EOS GdsStore, DMA to/from GDS). Null if no memory.
    uint32_t* gds();
    // The texture a T# names, as a draw would sample it: the render target at its address, or
    // the texture filled from guest memory (recorded into the current frame: submit before use).
    // `integerOnly`: the shader reads it with integer coordinates only (a T# smaller than its render target then keeps
    // naming the target; a normalised read gets a texture of the T#'s size, see renderer.cpp mirrorTarget).
    vpgpu::TextureId textureFor(const ImageDesc&, bool integerOnly = false);
    // The colour target most recently rendered at `address` (null if none): what a flip presents.
    vpgpu::TextureId colorTargetAt(uint64_t address) const;
    // The array colour target one of whose slices past 0 starts at `address` (null if none), with
    // that slice in `layer`: a T# naming one eye of a two-slice stereo target.
    vpgpu::TextureId colorTargetSlice(uint64_t address, uint32_t width, uint32_t height, uint16_t& layer) const;
    // Diagnostics: the GPU-side surfaces (colour/depth targets, shader-written textures) that
    // overlap guest range [address, address + bytes), one line ("nothing ..." if none).
    std::string describeAt(uint64_t address, uint64_t bytes) const;
    // Diagnostics: the GPU-side surface that backs a T# (the colour target at its address or a slice of an array
    // one, else a texture the renderer holds there), without touching its lifetime; null if there is none. `kind`
    // says what it is, `layer` the slice of an array colour target.
    vpgpu::TextureId backingAt(const ImageDesc&, uint16_t& layer, std::string& kind) const;
    // Probes (diagnostics, off by default): while on, every drawn call leaves a record per colour target address of
    // what it sampled and expanded. takeProbes() hands over and clears the records of the draws since the last call.
    // The geometry ring pointers are host memory of scratch slabs, valid until those rewind (kFrames submits).
    struct DrawProbe {
        uint64_t colorAddress = 0;
        bool hasPsTexture = false;
        ImageDesc psTexture;                // T# of the pixel shader's first texture binding
        bool geom = false;                  // a geometry-shader draw wrote this target
        const uint32_t* gsIndex = nullptr;  // GS output index list (0xffffffff: padding) ...
        uint64_t gsIndexCount = 0;
        const uint32_t* gsVs = nullptr;     // ... and the GS->VS ring dwords the copy shader reads
        uint64_t gsVsDwords = 0;
        uint32_t maxVertOut = 0;
    };
    void setProbes(bool on);
    std::vector<DrawProbe> takeProbes();
    const Stats& stats() const { return stats_; }
    // The CPU time accumulated since the last call (then reset).
    CpuTimes takeCpuTimes();
    // Probe (vpgpu/core/probe.h): sets the renderer's families (ENG, RATES, SKIP, HANDLED, NOTE, QUAL, APPROX) from its own
    // state; `pictures` is the number of pictures presented so far (the per-picture figures divide by the difference between
    // calls). Cheap but not free (walks the texture and target tables): call it once per sampling interval, from the thread
    // that draws.
    void publishProbe(uint64_t pictures);
    // Draws recorded into the colour target at `address` (width x height, 0: any) since the last call for it.
    uint64_t takeTargetDraws(uint64_t address, uint32_t width, uint32_t height);
    void setSummaryEvery(uint64_t frames) { summaryEvery_ = frames; }
    // Persistent cache under `root` (created): the translations of every program seen (so a later
    // run translates none of them again) and the pipeline states built (compiled ahead at the next
    // start). Loaded and written on a thread of its own; what it loaded reaches the renderer
    // (and Hooks::shader) over the following draws. Stale after a change of the translator: the
    // file is then replaced. Call once, before the first draw.
    void setCacheRoot(std::filesystem::path root);
    // Blocks until the persistent cache is loaded and merged (tests; a host may just let it load).
    void waitCache();
    // Writes the records queued so far (tests).
    void flushCache();
    // Hands the sink the per-draw reasons counted since the last submit (a submit does it too).
    void flushReports();
    // Upper bound for target sample counts (power of two, >= 1; the host's MSAA setting). Every
    // attachment of a draw gets the same count: the smallest of its targets' and this (Metal needs
    // one count per pass; EQAA depth with more samples than colour drops to the colour's).
    void setMaxSamples(uint8_t samples);
    uint8_t maxSamples() const { return maxSamples_; }
    // Detile textures with the GpuTiler kernels (default), or always on the CPU (tests, diagnosis).
    void setGpuTiling(bool on);
    // Off by default. On, the renderer never writes guest memory through its own work: no write back (re-tile), no
    // fill / copy / ZPASS_DONE, and a draw or dispatch that binds a V# buffer for writing is left out (counted). It
    // cannot stop a shader storing through a raw guest pointer (FLAT address): the host must hand out copies of the
    // memory from Hooks::region for that.
    void setGuestReadOnly(bool on);
    // On by default (the environment variable VPKENGINE_NO_IMAGE_ALIAS turns it off at start). A sampled T# whose memory
    // an image of the renderer wrote last, with the same layout, is read through that image instead of through a
    // round trip to memory (write back + fill); off, it is always filled from memory.
    void setImageAlias(bool on);

    // Foveated rendering of two-layer eye passes (foveation.h), experimental: the pass is drawn through a
    // rate map and expanded back when it ends. Passes it cannot serve are drawn at full rate (the
    // backend logs why once). Off by default; needs vertex shaders that export the render target array
    // index. Set before the first draw.
    void setFoveation(bool on) { foveation_ = on; }
    bool foveation() const { return foveation_; }

    struct Impl;
    struct ResolvedScope;

private:
    vpgpu::Device& device_;
    Hooks hooks_;
    Stats stats_;
    CpuTimes cpu_;
    uint64_t summaryEvery_ = 300, frame_ = 0;
    uint8_t maxSamples_ = 8;
    bool foveation_ = false;
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
