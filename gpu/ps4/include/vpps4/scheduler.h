// When the PS4 renderer hands its recorded work to the GPU, and how the CPU side stays ordered with it.
//
// Liverpool's command processor runs on the CPU, ahead of the GPU. Three things need care:
//  - Frames: work from many command buffers accumulates into one Frame per picture (see
//    vp4gpu/core/submit_policy.h for what forces a submit); a submit is a frame start for the compiler.
//  - Waits: a CPU access to memory GPU work touches (DMA, GDS, WRITE_DATA) waits only for the frames
//    that overlap it (partial fence), never for the whole queue.
//  - Labels: EVENT_WRITE_EOP / EOS / RELEASE_MEM data and interrupts are written when the GPU has
//    finished the work recorded before them, not when the packet is parsed.
// Lock order (host): engine lock -> LabelQueue::runMutex_ -> memory manager lock, never the reverse.
// A label's action (the write into guest memory) runs on the backend's completion thread holding
// runMutex_ and takes the memory manager's lock; a guest thread inside the memory manager that
// faults on a page the renderer owns waits for the engine lock. So the thread that holds the
// engine lock never blocks on runMutex_ for a label it does not need: pumps only try
// (Device::pumpLabels(false), the completion thread drains what became ready), and syncCpu waits
// for the label writes only when the range it syncs is named by one.
// All calls come from the thread that owns the renderer (the host's engine lock); `tick` too (a
// watchdog thread takes the same lock). The clock is injected for tests.
#pragma once

#include <array>
#include <cstdint>
#include <deque>
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
    // `value` (`size` bytes at `address`) is what the action writes: lets prepareWait recognise a wait the
    // GPU's own order satisfies. `hasValue` false = unknown (the label is never taken as satisfying).
    void scheduleLabel(uint64_t address, uint32_t size, bool irq, std::function<void()> action, uint64_t value = 0,
                       bool hasValue = false);
    // A WAIT_REG_MEM is about to poll [a, +n). `wanted(word)` says whether a value in memory satisfies it.
    //  Ordered: a label write already queued behind work recorded on this (in-order) queue writes a
    //           satisfying value. Whatever the command processor records from now on runs after that
    //           work, so the wait is met in GPU order: nothing is submitted, nothing is spun on (the
    //           write itself still happens when the GPU completes the work, for CPU observers).
    //  Poll:    the value comes from elsewhere (guest CPU, recorded shader writes, unknown label): the
    //           work that produces it is put on the GPU (a recorded frame is submitted if a pending label
    //           or recorded work writes there) and the caller polls, see idleWait. Never waits.
    enum class WaitKind { Poll, Ordered };
    WaitKind prepareWait(uint64_t a, uint64_t n, const std::function<bool(uint32_t)>& wanted = {});
    // A poll of [a, +n) found nothing yet: blocks (condition wait on the frame the label belongs to, at most
    // `timeoutMs`) instead of spinning. Returns whether it blocked on a GPU frame (false: no GPU frame
    // is involved, the caller should sleep a little itself).
    bool idleWait(uint64_t a, uint64_t n, unsigned timeoutMs);

    struct Counters {
        std::array<uint64_t, size_t(vp4gpu::SubmitReason::Count)> submits{};
        uint64_t orderedWaits = 0, polledWaits = 0, idleWaits = 0;
        uint64_t frames = 0, cpuSyncs = 0, cpuSyncFree = 0, partialWaits = 0, fullWaits = 0, labels = 0, labelTimeouts = 0;
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
    struct PendingValue {
        uint64_t address, value, frame;
        uint32_t size;
    };
    std::deque<PendingValue> values_;  // label writes queued with a known value, oldest first
};

}  // namespace vpps4
