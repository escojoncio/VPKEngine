#include "vpps4/memory_tracker.h"

#include <algorithm>
#include <iterator>

namespace vpps4 {

namespace {
uint64_t endOf(uint64_t address, uint64_t size) { return size > ~uint64_t(0) - address ? ~uint64_t(0) : address + size; }
}  // namespace

std::map<uint64_t, MemoryTracker::Span>::const_iterator MemoryTracker::firstOverlap(uint64_t begin) const {
    auto it = spans_.upper_bound(begin);
    if (it != spans_.begin() && std::prev(it)->second.end > begin) --it;
    return it;
}

void MemoryTracker::assign(uint64_t begin, uint64_t end, const Span& value) {
    if (begin >= end) return;
    auto it = spans_.upper_bound(begin);
    if (it != spans_.begin()) {
        auto p = std::prev(it);
        if (p->second.end > begin) {
            Span tail = p->second;
            const uint64_t pend = p->second.end;
            p->second.end = begin;  // keep the head
            if (pend > end) {       // the new range sits inside it: keep the tail too
                tail.end = pend;
                spans_[end] = tail;
            }
        }
    }
    it = spans_.lower_bound(begin);
    while (it != spans_.end() && it->first < end) {
        if (it->second.end > end) {  // starts inside, ends past: keep the tail
            Span tail = it->second;
            it = spans_.erase(it);
            spans_[end] = tail;
            break;
        }
        it = spans_.erase(it);
    }
    Span s = value;
    s.end = end;
    spans_[begin] = s;
    peak_ = std::max(peak_, spans_.size());
    if (spans_.size() > kMaxIntervals) forgetHistory();
}

uint64_t MemoryTracker::markGpuWrite(uint64_t address, uint64_t size, WriterId writer) {
    ++counter_;
    if (!size) return counter_;
    Span s;
    s.version = counter_;
    s.writer = writer;
    s.stale = isImage(writer);
    assign(address, endOf(address, size), s);
    return counter_;
}

uint64_t MemoryTracker::markCpuWrite(uint64_t address, uint64_t size) {
    ++counter_;
    if (!size) return counter_;
    Span s;
    s.version = counter_;
    s.writer = kCpu;
    assign(address, endOf(address, size), s);
    return counter_;
}

uint64_t MemoryTracker::version(uint64_t address, uint64_t size) const {
    const uint64_t end = endOf(address, std::max<uint64_t>(size, 1));
    uint64_t v = floor_;
    for (auto it = firstOverlap(address); it != spans_.end() && it->first < end; ++it) v = std::max(v, it->second.version);
    return v;
}

MemoryTracker::WriterId MemoryTracker::ownerImage(uint64_t address, uint64_t size) const {
    const uint64_t end = endOf(address, std::max<uint64_t>(size, 1));
    WriterId best = kNone;
    uint64_t bestVersion = 0;
    for (auto it = firstOverlap(address); it != spans_.end() && it->first < end; ++it)
        if (it->second.stale && it->second.version >= bestVersion) best = it->second.writer, bestVersion = it->second.version;
    return best;
}

bool MemoryTracker::writtenByOthers(uint64_t address, uint64_t size, WriterId self, uint64_t since) const {
    if (floor_ > since) return true;
    const uint64_t end = endOf(address, std::max<uint64_t>(size, 1));
    for (auto it = firstOverlap(address); it != spans_.end() && it->first < end; ++it)
        if (it->second.version > since && it->second.writer != self) return true;
    return false;
}

void MemoryTracker::writersSince(uint64_t address, uint64_t size, WriterId self, uint64_t since, std::vector<WriterId>& out) const {
    out.clear();
    if (floor_ > since) out.push_back(kNone);
    const uint64_t end = endOf(address, std::max<uint64_t>(size, 1));
    for (auto it = firstOverlap(address); it != spans_.end() && it->first < end; ++it)
        if (it->second.version > since && it->second.writer != self &&
            std::find(out.begin(), out.end(), it->second.writer) == out.end())
            out.push_back(it->second.writer);
}

void MemoryTracker::staleOwners(uint64_t address, uint64_t size, std::vector<Owner>& out, WriterId except) const {
    out.clear();
    const uint64_t end = endOf(address, std::max<uint64_t>(size, 1));
    for (auto it = firstOverlap(address); it != spans_.end() && it->first < end; ++it) {
        const Span& s = it->second;
        if (!s.stale || s.writer == except) continue;
        auto o = std::find_if(out.begin(), out.end(), [&](const Owner& x) { return x.writer == s.writer; });
        if (o == out.end()) {
            out.push_back({s.writer, s.version, it->first, s.end});
        } else {
            o->version = std::max(o->version, s.version);
            o->begin = std::min(o->begin, it->first);
            o->end = std::max(o->end, s.end);
        }
    }
    std::sort(out.begin(), out.end(), [](const Owner& a, const Owner& b) { return a.version < b.version; });
}

void MemoryTracker::markSynced(WriterId writer) {
    ++sweeps_, swept_ += spans_.size();
    for (auto& [b, s] : spans_)
        if (s.writer == writer) s.stale = false;
}

void MemoryTracker::release(WriterId writer) {
    ++sweeps_, swept_ += spans_.size();
    for (auto& [b, s] : spans_)
        if (s.writer == writer) s.writer = kNone, s.stale = false;
}

void MemoryTracker::forgetHistory() {
    ++forgetCalls_;
    floor_ = counter_;
    for (auto it = spans_.begin(); it != spans_.end();) {
        if (it->second.stale) ++it;  // an image's unsynced claim is not history
        else it = spans_.erase(it);
    }
}

MemoryTracker::WriterId MemoryTracker::writerAt(uint64_t address) const {
    auto it = firstOverlap(address);
    return it != spans_.end() && it->first <= address ? it->second.writer : kNone;
}

bool MemoryTracker::staleAt(uint64_t address) const {
    auto it = firstOverlap(address);
    return it != spans_.end() && it->first <= address && it->second.stale;
}

}  // namespace vpps4
