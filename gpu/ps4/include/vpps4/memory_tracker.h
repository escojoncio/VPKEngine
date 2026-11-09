// Who holds the newest bytes of guest memory.
//
// On the PS4 every resource is guest memory read through a descriptor: the same bytes are written
// as a render target, a depth buffer, a storage image, a raw buffer or by DMA, and read as any of
// them. The host keeps images as GPU objects that never write back by themselves, so the renderer
// needs one place that knows, per range of guest memory, which writer holds the newest copy:
//
//  * every write (recorded draw / dispatch / DMA, or the CPU) gets a version from one counter;
//  * a range is owned by its last writer. An image that rendered into the range owns it with
//    memoryStale set: its GPU copy is newer than the bytes in memory, which must be re-tiled
//    (image -> buffer) before anything reads them raw;
//  * a resource that was filled from memory (or wrote it itself) remembers the version it holds:
//    it is out of date when some other writer has written its range since.
//
// All marks are made when the work is RECORDED. Recorded order is the GPU timeline's order, so
// "later version" means "executes later", which is what makes lazily recorded syncs correct.
//
// Ranges are kept at byte granularity (an interval map), so a buffer and an image that share a
// 16 KB page do not falsely share state; kPageBytes is the granularity CPU write detection works at
// (page protection on the host) and what pages()/pageOf() round to.
#pragma once

#include <cstdint>
#include <map>
#include <vector>

namespace vpps4 {

class MemoryTracker {
public:
    using WriterId = uint32_t;
    // Writers that are not images. Image writer ids come from newImage() (>= kFirstImage).
    static constexpr WriterId kNone = 0, kCpu = 1, kGpuRaw = 2, kDma = 3, kFirstImage = 16;
    static constexpr uint64_t kPageBytes = 16384;

    static constexpr bool isImage(WriterId w) { return w >= kFirstImage; }
    static constexpr uint64_t pageOf(uint64_t address) { return address / kPageBytes; }

    WriterId newImage() { return nextImage_++; }

    // A recorded GPU write of [address, address + size) by `writer` (an image id, kGpuRaw or
    // kDma). Returns the new version. An image writer's range becomes memory-stale: its copy is
    // newer than memory until markSynced().
    uint64_t markGpuWrite(uint64_t address, uint64_t size, WriterId writer);
    // The CPU wrote the range (found by page protection or by content): memory is the truth.
    uint64_t markCpuWrite(uint64_t address, uint64_t size);

    // The newest version written to any byte of the range (0: never written).
    uint64_t version(uint64_t address, uint64_t size) const;
    // The counter's current value: what a resource that was just filled from memory holds.
    uint64_t current() const { return counter_; }

    // The image whose GPU copy is newer than memory somewhere in the range (the most recently
    // written one); 0 if memory is current there.
    WriterId ownerImage(uint64_t address, uint64_t size) const;

    // Is the last writer of some byte of the range the GPU (kGpuRaw, kDma or an image)? Recorded writes count before they run.
    bool gpuWritten(uint64_t address, uint64_t size) const;
    // Has a writer other than `self` written part of the range after version `since`?
    bool writtenByOthers(uint64_t address, uint64_t size, WriterId self, uint64_t since) const;

    // The writers other than `self` that wrote part of the range after version `since`, each once (kNone: a released
    // image, or the history was forgotten since: who wrote is unknown).
    void writersSince(uint64_t address, uint64_t size, WriterId self, uint64_t since, std::vector<WriterId>& out) const;

    struct Owner {
        WriterId writer = 0;
        uint64_t version = 0;       // newest version of that writer's stale intervals in the range
        uint64_t begin = 0, end = 0;  // their extent
    };
    // The images with memory-stale intervals in the range (other than `except`), oldest write
    // first: re-tiling them in this order leaves the newest data last.
    void staleOwners(uint64_t address, uint64_t size, std::vector<Owner>& out, WriterId except = kNone) const;
    // The image's bytes were written back to memory (re-tile recorded): memory is current again
    // where it owned. It stays the owner (its GPU copy is still valid), no longer stale.
    void markSynced(WriterId writer);
    // The image is gone: its intervals have no owner (versions are kept).
    void release(WriterId writer);

    // Drops the history of non-image writes older than the current version (bounded memory):
    // every resource then counts as out of date once. Called automatically past kMaxIntervals.
    void forgetHistory();
    size_t intervals() const { return spans_.size(); }
    // Diagnostics: the most intervals held at once, and the calls to forgetHistory() so far.
    size_t peakIntervals() const { return peak_; }
    uint64_t forgetHistoryCalls() const { return forgetCalls_; }
    // Probe: the version below which every resource counts as out of date (forgetHistory), the O(n) sweeps of markSynced /
    // release and the intervals they walked, and the newest version.
    uint64_t floor() const { return floor_; }
    uint64_t sweeps() const { return sweeps_; }
    uint64_t swept() const { return swept_; }
    // Writer / stale state at one byte (tests, diagnostics).
    WriterId writerAt(uint64_t address) const;
    bool staleAt(uint64_t address) const;

    static constexpr size_t kMaxIntervals = 1u << 17;

private:
    struct Span {
        uint64_t end = 0, version = 0;
        WriterId writer = kNone;
        bool stale = false;
    };
    std::map<uint64_t, Span> spans_;  // by begin; disjoint
    uint64_t counter_ = 0;
    uint64_t floor_ = 0;  // forgetHistory(): versions at or below this are all "unknown"
    WriterId nextImage_ = kFirstImage;
    size_t peak_ = 0;
    uint64_t forgetCalls_ = 0;
    uint64_t sweeps_ = 0, swept_ = 0;

    void assign(uint64_t begin, uint64_t end, const Span&);
    std::map<uint64_t, Span>::const_iterator firstOverlap(uint64_t begin) const;
};

}  // namespace vpps4
