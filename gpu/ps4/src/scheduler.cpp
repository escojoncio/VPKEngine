#include "vpps4/scheduler.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "vpgpu/core/probe.h"

namespace vpps4 {

using vpgpu::SubmitReason;

Scheduler::Scheduler(Renderer& r, vpgpu::Device& d, vpgpu::SubmitPolicy p, Clock c)
    : renderer_(r), device_(d), policy_(p), now_(std::move(c)), dmaGpu_(!std::getenv("VPKENGINE_DMA_CPU")),
      supersede_(!std::getenv("VPKENGINE_LABEL_KEEP")) {
    summarySince_ = now_ ? now_() : 0;
}

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
vpgpu::SubmitState stateOf(const Renderer& r, double now, double openSince, double labelSince, bool irq) {
    vpgpu::SubmitState s;
    s.commands = r.recordedCommands();
    s.openMs = openSince >= 0 ? now - openSince : 0;
    s.labelWaiting = labelSince >= 0;
    s.labelMs = s.labelWaiting ? now - labelSince : 0;
    s.labelIrq = irq;
    s.costlyCut = r.multisampledPassOpen();
    return s;
}
}  // namespace

bool Scheduler::tick() {
    const vpgpu::SubmitState st = stateOf(renderer_, now_(), openSince_, labelSince_, labelIrq_);
    const SubmitReason why = vpgpu::decide(policy_, st);
    if (why == SubmitReason::Count) return false;
    return submit(why);
}

double Scheduler::nextCheckMs() const {
    return vpgpu::nextCheckMs(policy_, stateOf(renderer_, now_(), openSince_, labelSince_, labelIrq_));
}

bool Scheduler::syncCpu(uint64_t a, uint64_t n, SubmitReason why, uint64_t b, uint64_t m, const char* caller) {
    uint64_t target = renderer_.waitTarget(a, n, b, m);
    uint64_t labels = std::max(device_.labelWaitTarget(a, uint32_t(std::min<uint64_t>(n, ~0u))),
                               m ? device_.labelWaitTarget(b, uint32_t(std::min<uint64_t>(m, ~0u))) : 0);
    if (target == 0 && labels == 0) {
        ++counters_.cpuSyncFree;
        renderer_.waitedBelow(device_.completedFrames());  // (write backs the GPU has run give their pages back)
        renderer_.noteCpuSync(caller, a, n, b, m, "", 0, false, 0);
        return false;
    }
    ++counters_.cpuSyncs;
    const double tSync = now_();
    const bool probing = vpgpu::probe::enabled();
    const std::string busyRange = probing ? renderer_.busyText(a, n, b, m) : std::string();
    const bool unsubmitted = target == Renderer::kUnsubmitted || labels > device_.submittedFrames();
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
        if (labels != 0) {
            // (never blocks on the release lock: this thread holds the engine lock, see the header)
            if (!device_.waitLabels(labels)) ++counters_.labelTimeouts;
        } else {
            device_.pumpLabels(false);
        }
    }
    counters_.waitMs += now_() - t0;
    counters_.syncMs += now_() - tSync;
    if (probing) renderer_.noteCpuSync(caller, a, n, b, m, busyRange, frames, unsubmitted, now_() - t0);
    return frames != 0;
}

bool Scheduler::dmaOnGpu(uint64_t dst, uint64_t n, uint64_t src) {
    // A label still to be written whose bytes this DMA all overwrites and that the command processor's own WAIT_REG_MEM already
    // met in order (prepareWait: Ordered): in GPU order the label lands, then the DMA's bytes. Dropped, the DMA needs no wait for
    // it on either path. (One with an interrupt, one nobody met in order, whose value a CPU poll or another ring may wait for,
    // or one overlapping what a copy reads stays.)
    if (supersede_ && device_.labelWaitTarget(dst, uint32_t(std::min<uint64_t>(n, ~0u))))
        if (const size_t k = device_.supersedeLabels(dst, n, src, src ? n : 0)) counters_.labelsSuperseded += k, forgetLabelValues(dst, n);
    if (!dmaGpu_) return false;
    if (renderer_.cpuReadSeen(dst, n)) return renderer_.probeDmaCpuRead(dst, n, src), ++counters_.dmaCpuRead, false;
    const uint64_t m = src ? n : 0;
    // (a label write into the range lands when its frame completes: it must come before the DMA's bytes, which only the
    // CPU waiting for it guarantees)
    if (device_.labelWaitTarget(dst, uint32_t(std::min<uint64_t>(n, ~0u))) ||
        (m && device_.labelWaitTarget(src, uint32_t(std::min<uint64_t>(m, ~0u)))))
        return ++counters_.dmaLabel, false;
    if (renderer_.waitTarget(dst, n, src, m) == 0) return ++counters_.dmaIdle, false;
    return true;
}

void Scheduler::finish() {
    submit(SubmitReason::Finish);
    ++counters_.fullWaits;
    const double t0 = now_();
    device_.waitIdle();
    renderer_.gpuIdle();
    if (device_.labelsQueued() && !device_.waitLabels(device_.submittedFrames())) ++counters_.labelTimeouts;
    counters_.waitMs += now_() - t0;
}

void Scheduler::scheduleLabel(uint64_t address, uint32_t size, bool irq, std::function<void()> action, uint64_t value,
                              bool hasValue) {
    ++counters_.labels;
    // A write whose value is not known (or too wide to track) hides what older entries promised there.
    if (address && size && !(hasValue && size <= 8)) forgetLabelValues(address, size);
    if (hasValue && address && size && size <= 8) {
        // (the frame the label waits for, as Renderer::afterGpu numbers it; none = written at once)
        const uint64_t submitted = device_.submittedFrames();
        if (renderer_.hasRecordedWork() || submitted != 0) {
            const uint64_t frame = renderer_.hasRecordedWork() ? submitted : submitted - 1;
            values_.push_back({address, value, frame, size});
            while (values_.size() > 512) values_.pop_front();
        }
    }
    if (renderer_.hasRecordedWork()) {
        if (labelSince_ < 0) labelSince_ = now_();
        labelIrq_ = labelIrq_ || irq;
        noteWork();
    }
    renderer_.afterGpu(address, size, std::move(action), !irq && address && size);
}

void Scheduler::forgetLabelValues(uint64_t a, uint64_t n) {
    std::erase_if(values_, [&](const PendingValue& v) { return v.address < a + n && a < v.address + v.size; });
}

Scheduler::WaitKind Scheduler::prepareWait(uint64_t a, uint64_t n, const std::function<bool(uint32_t)>& wanted) {
    const uint32_t size = uint32_t(std::min<uint64_t>(n, ~0u));
    if (wanted && size <= 4 && !values_.empty()) {
        // Entries whose frame has finished have run (or are about to): the memory is final.
        const uint64_t done = device_.completedFrames();
        while (!values_.empty() && values_.front().frame < done) values_.pop_front();
        // Only the newest entry that touches the word counts: it is what memory will hold last.
        for (auto it = values_.rbegin(); it != values_.rend(); ++it) {
            if (it->frame < done) continue;
            if (it->address >= a + 4 || a >= it->address + it->size) continue;  // no overlap
            if (a < it->address || a + 4 > it->address + it->size) break;       // partial: unknown
            const unsigned shift = unsigned(a - it->address) * 8;
            if (wanted(uint32_t(it->value >> shift))) {
                device_.markLabelOrdered(it->address, it->size);
                ++counters_.orderedWaits;
                return WaitKind::Ordered;
            }
            break;
        }
    }
    ++counters_.polledWaits;
    const bool labelOnRecorded = device_.labelWaitTarget(a, size) > device_.submittedFrames();
    if (labelOnRecorded || renderer_.waitTarget(a, n) == Renderer::kUnsubmitted) submit(SubmitReason::WaitLabel);
    return WaitKind::Poll;
}

uint64_t Scheduler::idleTarget(uint64_t a, uint64_t n) {
    const uint32_t size = uint32_t(std::min<uint64_t>(n, ~0u));
    uint64_t target = std::max(device_.labelWaitTarget(a, size), renderer_.waitTarget(a, n));
    if (target == 0) {
        device_.pumpLabels(false);
        return 0;
    }
    // (only work still recorded is submitted here; once it is a frame, later looks find it submitted)
    if (target == Renderer::kUnsubmitted || target > device_.submittedFrames()) {
        submit(SubmitReason::WaitLabel);
        target = std::max(device_.labelWaitTarget(a, size), renderer_.waitTarget(a, n));
        if (target == Renderer::kUnsubmitted) target = device_.submittedFrames();
    }
    if (target) ++counters_.idleWaits;
    return std::min(target, device_.submittedFrames());
}

bool Scheduler::idleWait(uint64_t a, uint64_t n, unsigned timeoutMs) {
    const uint64_t target = idleTarget(a, n);
    if (!target) return false;
    device_.waitFrames(target, uint64_t(timeoutMs) * 1000000ull);
    device_.pumpLabels(false);
    return true;
}

std::string Scheduler::takeSummary() {
    std::string s = "submits";
    for (size_t i = 0; i < counters_.submits.size(); ++i)
        s += std::string(" ") + vpgpu::toString(SubmitReason(i)) + "=" + std::to_string(counters_.submits[i]);
    s += " frames=" + std::to_string(counters_.frames) + " cpu_sync=" + std::to_string(counters_.cpuSyncs) + "/" +
         std::to_string(counters_.cpuSyncs + counters_.cpuSyncFree) + " partial_waits=" +
         std::to_string(counters_.partialWaits) + " full_waits=" + std::to_string(counters_.fullWaits) +
         " dma_gpu=" + std::to_string(counters_.dmaRecorded) + " labels=" + std::to_string(counters_.labels) + " ordered_waits=" + std::to_string(counters_.orderedWaits) +
         " polled_waits=" + std::to_string(counters_.polledWaits) + " idle_waits=" + std::to_string(counters_.idleWaits) + " wait_ms=" + std::to_string(int(counters_.waitMs));
    s += " dma_cpu_read=" + std::to_string(counters_.dmaCpuRead) + " dma_label=" + std::to_string(counters_.dmaLabel) +
         " dma_idle=" + std::to_string(counters_.dmaIdle);
    const double now = now_(), span = now - summarySince_;
    const uint64_t flips = counters_.submits[size_t(SubmitReason::Flip)];
    char extra[160];
    std::snprintf(extra, sizeof extra, " label_superseded=%llu sync_ms=%d cp_wait_pct=%.1f frames_per_flip=%.1f",
                  (unsigned long long)counters_.labelsSuperseded, int(counters_.syncMs), span > 0 ? counters_.syncMs * 100.0 / span : 0.0,
                  flips ? double(counters_.frames) / double(flips) : 0.0);
    s += extra;
    if (counters_.labelTimeouts) s += " label_timeouts=" + std::to_string(counters_.labelTimeouts);
    counters_ = {};
    summarySince_ = now;
    return s;
}

}  // namespace vpps4
