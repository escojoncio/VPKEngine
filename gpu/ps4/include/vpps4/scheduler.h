// When the PS4 renderer hands its recorded work to the GPU, and how the CPU side stays ordered with it.
//
// Liverpool's command processor runs on the CPU, ahead of the GPU. Three things need care:
//  - Frames: work from many command buffers accumulates into one Frame per picture (see
//    vp4gpu/core/submit_policy.h for what forces a submit); a submit is a frame start for the compiler.
//  - Waits: a CPU access to memory GPU work touches (DMA, GDS, WRITE_DATA) waits only for the frames
//    that overlap it (partial fence), never for the whole queue.
//  - Labels: EVENT_WRITE_EOP / EOS / RELEASE_MEM data and interrupts are written when the GPU has
//    finished the work recorded before them, not when the packet is parsed.
// All calls come from the thread that owns the renderer (the host's engine lock); `tick` too (a
// watchdog thread takes the same lock). The clock is injected for tests.
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <string>

#include "vp4gpu/core/device.h"
#include "vp4gpu/core/submit_policy.h"
#include "vpps4/renderer.h"

namespace vpps4 {

class Scheduler {
public:
    using Clock = std::function<double()>;  // milliseconds, monotonic
    Scheduler(Renderer&, vp4gpu::Device&, vp4gpu::SubmitPolicy, Clock);

    // Work was recorded (a draw, dispatch or queued DMA). True when it is the first of a frame: the
    // watchdog should be woken to compute its deadline.
    bool noteWork();
    // Submits the recorded work now. No-op without any. Returns whether a frame was submitted.
    bool submit(vp4gpu::SubmitReason why);
    // Runs just before a frame is handed over (the host's per-submit housekeeping).
    void setSubmitHook(std::function<void()> hook) { hook_ = std::move(hook); }
    // The picture is presented: everything recorded must run.
    void present() { submit(vp4gpu::SubmitReason::Flip); }
    // A Liverpool command buffer batch ended (SubmitDone): work stays recorded unless the policy says so.
    void flush() { tick(); }
    // Applies the policy to what is recorded (also the watchdog's step). Returns whether it submitted.
    bool tick();
    // Milliseconds until tick() may act (0 = now), negative when there is nothing recorded.
    double nextCheckMs() const;

    // The CPU is about to read or write guest [a, +n) (and [b, +m)): waits, with a partial fence, for the
    // GPU work that touches it and for label writes into it, submitting the recorded frame first only if
    // it is part of that. False = nothing to wait for.
    bool syncCpu(uint64_t a, uint64_t n, vp4gpu::SubmitReason why, uint64_t b = 0, uint64_t m = 0);
    // The same for GDS.
    bool syncGds() { return syncCpu(renderer_.gdsAddress(), 0x10000, vp4gpu::SubmitReason::Gds); }
    // Everything submitted and recorded has finished.
    void finish();
    // `action` (a label write + its interrupt) runs once the GPU has finished the work recorded so far.
    void scheduleLabel(uint64_t address, uint32_t size, bool irq, std::function<void()> action);
    // A WAIT_REG_MEM is about to poll [a, +n): makes sure the work that produces it is on the GPU
    // (submits the recorded frame if a pending label or recorded work writes there). Never waits.
    void prepareWait(uint64_t a, uint64_t n);

    struct Counters {
        std::array<uint64_t, size_t(vp4gpu::SubmitReason::Count)> submits{};
        uint64_t frames = 0, cpuSyncs = 0, cpuSyncFree = 0, partialWaits = 0, fullWaits = 0, labels = 0;
        double waitMs = 0;
    };
    const Counters& counters() const { return counters_; }
    // "submits flip=.. dma=.. ..." since the last call (counters are reset).
    std::string takeSummary();

private:
    void resetOpen();
    Renderer& renderer_;
    vp4gpu::Device& device_;
    vp4gpu::SubmitPolicy policy_;
    Clock now_;
    std::function<void()> hook_;
    double openSince_ = -1, labelSince_ = -1;
    bool labelIrq_ = false;
    Counters counters_;
};

}  // namespace vpps4
