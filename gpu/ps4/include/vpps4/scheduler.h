// When the PS4 renderer hands its recorded work to the GPU, and how the CPU side stays ordered with it.
//
// Liverpool's command processor runs on the CPU, ahead of the GPU. Three things need care:
//  - Frames: work from many command buffers accumulates into one Frame per picture (see
//    vpgpu/core/submit_policy.h for what forces a submit); a submit is a frame start for the compiler.
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

#include "vpgpu/core/device.h"
#include "vpgpu/core/submit_policy.h"
#include "vpps4/renderer.h"

namespace vpps4 {

class Scheduler {
public:
    using Clock = std::function<double()>;  // milliseconds, monotonic
    Scheduler(Renderer&, vpgpu::Device&, vpgpu::SubmitPolicy, Clock);

    // Work was recorded (a draw, dispatch or queued DMA). True when it is the first of a frame: the
    // watchdog should be woken to compute its deadline.
    bool noteWork();
    // Submits the recorded work now. No-op without any. Returns whether a frame was submitted.
    bool submit(vpgpu::SubmitReason why);
    // Runs just before a frame is handed over (the host's per-submit housekeeping).
    void setSubmitHook(std::function<void()> hook) { hook_ = std::move(hook); }
    // The picture is presented: everything recorded must run.
    void present() { submit(vpgpu::SubmitReason::Flip); }
    // A Liverpool command buffer batch ended (SubmitDone): work stays recorded unless the policy says so.
    void flush() { tick(); }
    // Applies the policy to what is recorded (also the watchdog's step). Returns whether it submitted.
    bool tick();
    // Milliseconds until tick() may act (0 = now), negative when there is nothing recorded.
    double nextCheckMs() const;

    // The CPU is about to read or write guest [a, +n) (and [b, +m)): waits, with a partial fence, for the
    // GPU work that touches it and for label writes into it, submitting the recorded frame first only if
    // it is part of that. False = nothing to wait for.
    // `caller` names who asks (probe DMASYNC only: "fill", "copy", "owned_fault", "cp_read", ...).
    bool syncCpu(uint64_t a, uint64_t n, vpgpu::SubmitReason why, uint64_t b = 0, uint64_t m = 0, const char* caller = "");
    // A host DMA (fill / copy) over [a, +n) (and [b, +m)) is about to run: true = queued GPU work touches the range, so
    // syncCpu would wait; the caller records it on the GPU timeline (Renderer::fill / copy) instead, in order with that work,
    // with no wait and no submit. False = the CPU may do it in place after syncCpu (nothing GPU touches the range, a label
    // write into it is pending: that one must land first, or VPKENGINE_DMA_CPU is set). Whoever reads the result on the CPU
    // (a CP read: IB, label, WAIT_REG_MEM) goes through syncCpu, which finds the recorded range busy.
    // `dst` is what the DMA writes ([dst, +n)), `src` (0: none) what a copy reads. A destination the renderer has read on the CPU
    // (Renderer::cpuReadSeen) stays on the CPU path: it would read old bytes until the GPU runs the DMA.
    bool dmaOnGpu(uint64_t dst, uint64_t n, uint64_t src = 0);
    // The caller recorded the DMA that dmaOnGpu allowed (Renderer::fill / copy succeeded): counted.
    void noteDmaRecorded() { ++counters_.dmaRecorded; }
    // The same for GDS.
    bool syncGds() { return syncCpu(renderer_.gdsAddress(), 0x10000, vpgpu::SubmitReason::Gds); }
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
    // Memory [a, +n) was written by someone else (CPU, WRITE_DATA, a wider label): queued label values
    // there no longer say what a wait will find.
    void forgetLabelValues(uint64_t a, uint64_t n);
    enum class WaitKind { Poll, Ordered };
    WaitKind prepareWait(uint64_t a, uint64_t n, const std::function<bool(uint32_t)>& wanted = {});
    // A poll of [a, +n) found nothing yet: blocks (condition wait on the frame the label belongs to, at most
    // `timeoutMs`) instead of spinning. Returns whether it blocked on a GPU frame (false: no GPU frame
    // is involved, the caller should sleep a little itself).
    bool idleWait(uint64_t a, uint64_t n, unsigned timeoutMs);
    // idleWait in two halves, so a host can wait without holding its lock: idleTarget (under the lock) returns
    // the number of frames to wait for (0 = no GPU frame involved); the host calls waitGpu(target, ms) outside
    // the lock, then pumpLabels() under it.
    uint64_t idleTarget(uint64_t a, uint64_t n);
    bool waitGpu(uint64_t target, unsigned timeoutMs) { return device_.waitFrames(target, uint64_t(timeoutMs) * 1000000ull); }
    void pumpLabels() { device_.pumpLabels(false); }

    struct Counters {
        std::array<uint64_t, size_t(vpgpu::SubmitReason::Count)> submits{};
        uint64_t orderedWaits = 0, polledWaits = 0, idleWaits = 0;
        uint64_t dmaRecorded = 0;  // host DMAs recorded on the GPU timeline instead of waiting (dmaOnGpu)
        uint64_t dmaCpuRead = 0, dmaLabel = 0, dmaIdle = 0;  // dmaOnGpu said no: range read by the CPU, label pending, nothing queued
        uint64_t frames = 0, cpuSyncs = 0, cpuSyncFree = 0, partialWaits = 0, fullWaits = 0, labels = 0, labelTimeouts = 0;
        double waitMs = 0;
    };
    const Counters& counters() const { return counters_; }
    // "submits flip=.. dma=.. ..." since the last call (counters are reset).
    std::string takeSummary();

private:
    void resetOpen();
    Renderer& renderer_;
    vpgpu::Device& device_;
    vpgpu::SubmitPolicy policy_;
    Clock now_;
    std::function<void()> hook_;
    bool dmaGpu_;  // false (VPKENGINE_DMA_CPU set): a DMA always waits and runs on the CPU
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
