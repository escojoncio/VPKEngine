#include "vpps4/scheduler.h"

#include <algorithm>

namespace vpps4 {

using vp4gpu::SubmitReason;

Scheduler::Scheduler(Renderer& r, vp4gpu::Device& d, vp4gpu::SubmitPolicy p, Clock c)
    : renderer_(r), device_(d), policy_(p), now_(std::move(c)) {}

void Scheduler::resetOpen() {
    openSince_ = labelSince_ = -1;
    labelIrq_ = false;
}

bool Scheduler::noteWork() {
    if (openSince_ >= 0) return false;
    openSince_ = now_();
    return true;
}

bool Scheduler::submit(SubmitReason why) {
    if (!renderer_.hasRecordedWork()) {
        renderer_.submit();  // (nothing to hand over; lets the renderer forget ranges of draws that were skipped)
        resetOpen();
        return false;
    }
    ++counters_.submits[size_t(why)];
    ++counters_.frames;
    if (hook_) hook_();
    renderer_.submit();
    resetOpen();
    return true;
}

namespace {
vp4gpu::SubmitState stateOf(const Renderer& r, double now, double openSince, double labelSince, bool irq) {
    vp4gpu::SubmitState s;
    s.commands = r.recordedCommands();
    s.openMs = openSince >= 0 ? now - openSince : 0;
    s.labelWaiting = labelSince >= 0;
    s.labelMs = s.labelWaiting ? now - labelSince : 0;
    s.labelIrq = irq;
    return s;
}
}  // namespace

bool Scheduler::tick() {
    const vp4gpu::SubmitState st = stateOf(renderer_, now_(), openSince_, labelSince_, labelIrq_);
    const SubmitReason why = vp4gpu::decide(policy_, st);
    if (why == SubmitReason::Count) return false;
    return submit(why);
}

double Scheduler::nextCheckMs() const {
    return vp4gpu::nextCheckMs(policy_, stateOf(renderer_, now_(), openSince_, labelSince_, labelIrq_));
}

bool Scheduler::syncCpu(uint64_t a, uint64_t n, SubmitReason why, uint64_t b, uint64_t m) {
    uint64_t target = renderer_.waitTarget(a, n, b, m);
    uint64_t labels = std::max(device_.labelWaitTarget(a, uint32_t(std::min<uint64_t>(n, ~0u))),
                               m ? device_.labelWaitTarget(b, uint32_t(std::min<uint64_t>(m, ~0u))) : 0);
    if (target == 0 && labels == 0) {
        ++counters_.cpuSyncFree;
        renderer_.waitedBelow(device_.completedFrames());  // (write backs the GPU has run give their pages back)
        return false;
    }
    ++counters_.cpuSyncs;
    // Recorded work (or a label waiting for it) is part of what must finish: it goes first. Frames
    // are numbered in submit order, so the frame it becomes is the last one.
    if (target == Renderer::kUnsubmitted || labels > device_.submittedFrames()) {
        submit(why);
        // (the submit moved the recorded ranges to the frame it made, or dropped those of skipped
        // draws: either way the range names its own frame now, never "everything")
        target = renderer_.waitTarget(a, n, b, m);
        if (target == Renderer::kUnsubmitted) target = device_.submittedFrames();
        labels = std::max(device_.labelWaitTarget(a, uint32_t(std::min<uint64_t>(n, ~0u))),
                          m ? device_.labelWaitTarget(b, uint32_t(std::min<uint64_t>(m, ~0u))) : 0);
    }
    const uint64_t frames = std::max(target, labels);
    const double t0 = now_();
    if (frames) {
        (frames >= device_.submittedFrames() ? counters_.fullWaits : counters_.partialWaits)++;
        device_.waitFrames(frames);
        renderer_.waitedBelow(frames);
        // (a label of those frames being written on the completion thread has finished: waited for
        // only when this call needs a label; otherwise that thread drains it. See LabelQueue.)
        device_.pumpLabels(labels != 0);
    }
    counters_.waitMs += now_() - t0;
    return frames != 0;
}

void Scheduler::finish() {
    submit(SubmitReason::Finish);
    ++counters_.fullWaits;
    const double t0 = now_();
    device_.waitIdle();
    renderer_.gpuIdle();
    device_.pumpLabels(device_.labelsQueued());
    counters_.waitMs += now_() - t0;
}

void Scheduler::scheduleLabel(uint64_t address, uint32_t size, bool irq, std::function<void()> action) {
    ++counters_.labels;
    if (renderer_.hasRecordedWork()) {
        if (labelSince_ < 0) labelSince_ = now_();
        labelIrq_ = labelIrq_ || irq;
        noteWork();
    }
    renderer_.afterGpu(address, size, std::move(action));
}

void Scheduler::prepareWait(uint64_t a, uint64_t n) {
    const uint32_t size = uint32_t(std::min<uint64_t>(n, ~0u));
    const bool labelOnRecorded = device_.labelWaitTarget(a, size) > device_.submittedFrames();
    if (labelOnRecorded || renderer_.waitTarget(a, n) == Renderer::kUnsubmitted) submit(SubmitReason::WaitLabel);
}

std::string Scheduler::takeSummary() {
    std::string s = "submits";
    for (size_t i = 0; i < counters_.submits.size(); ++i)
        s += std::string(" ") + vp4gpu::toString(SubmitReason(i)) + "=" + std::to_string(counters_.submits[i]);
    s += " frames=" + std::to_string(counters_.frames) + " cpu_sync=" + std::to_string(counters_.cpuSyncs) + "/" +
         std::to_string(counters_.cpuSyncs + counters_.cpuSyncFree) + " partial_waits=" +
         std::to_string(counters_.partialWaits) + " full_waits=" + std::to_string(counters_.fullWaits) +
         " labels=" + std::to_string(counters_.labels) + " wait_ms=" + std::to_string(int(counters_.waitMs));
    counters_ = {};
    return s;
}

}  // namespace vpps4
