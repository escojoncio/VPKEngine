// vpps4::MemoryTracker: versions, ownership, staleness, release (no renderer involved).
#include <cstdio>
#include <string_view>
#include <vector>

#include "vpps4/memory_tracker.h"

namespace {
int failures = 0;
#define CHECK(c)                                                                       \
    do {                                                                               \
        if (!(c)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #c); \
            ++failures;                                                                \
        }                                                                              \
    } while (0)

using vpps4::MemoryTracker;

void versions() {
    MemoryTracker t;
    CHECK(t.version(0x1000, 0x100) == 0);
    const uint64_t a = t.markCpuWrite(0x1000, 0x100);
    const uint64_t b = t.markGpuWrite(0x2000, 0x100, MemoryTracker::kDma);
    CHECK(b > a);
    CHECK(t.version(0x1000, 0x100) == a);
    CHECK(t.version(0x10ff, 1) == a);
    CHECK(t.version(0x1100, 1) == 0);  // just past
    CHECK(t.version(0x0f00, 0x200) == a);
    CHECK(t.version(0x1000, 0x2000) == b);  // spans both
    // A rewrite replaces part of an older range: the old remainder keeps its version.
    const uint64_t c = t.markGpuWrite(0x1040, 0x20, MemoryTracker::kGpuRaw);
    CHECK(t.version(0x1000, 0x40) == a);
    CHECK(t.version(0x1040, 0x20) == c);
    CHECK(t.version(0x1060, 0xa0) == a);
    CHECK(t.writerAt(0x1050) == MemoryTracker::kGpuRaw);
    CHECK(t.writerAt(0x1000) == MemoryTracker::kCpu);
    CHECK(t.writerAt(0x1060) == MemoryTracker::kCpu);
}

void writtenByOthers() {
    MemoryTracker t;
    const auto img = t.newImage();
    CHECK(MemoryTracker::isImage(img));
    const uint64_t v = t.markGpuWrite(0x10000, 0x4000, img);
    CHECK(!t.writtenByOthers(0x10000, 0x4000, img, v));  // its own write
    const uint64_t raw = t.markGpuWrite(0x11000, 0x10, MemoryTracker::kGpuRaw);
    CHECK(t.writtenByOthers(0x10000, 0x4000, img, v));   // a raw write inside
    CHECK(!t.writtenByOthers(0x10000, 0x1000, img, v));  // outside the part asked about
    CHECK(!t.writtenByOthers(0x10000, 0x4000, img, raw));
    t.markCpuWrite(0x13fff, 1);
    CHECK(t.writtenByOthers(0x10000, 0x4000, img, raw));
}

void ownership() {
    MemoryTracker t;
    const auto a = t.newImage(), b = t.newImage();
    CHECK(a != b);
    t.markGpuWrite(0x20000, 0x8000, a);
    CHECK(t.ownerImage(0x20000, 0x10) == a);
    CHECK(t.ownerImage(0x27fff, 1) == a);
    CHECK(t.ownerImage(0x28000, 1) == 0);
    CHECK(t.staleAt(0x20000));
    // A second image over the second half: both are owners, oldest first; the newest wins a query
    // spanning both.
    t.markGpuWrite(0x24000, 0x4000, b);
    CHECK(t.ownerImage(0x20000, 0x8000) == b);
    CHECK(t.ownerImage(0x20000, 0x1000) == a);
    std::vector<MemoryTracker::Owner> o;
    t.staleOwners(0x20000, 0x8000, o);
    CHECK(o.size() == 2 && o[0].writer == a && o[1].writer == b);
    CHECK(o[0].begin == 0x20000 && o[0].end == 0x24000);  // a keeps only its first half
    t.staleOwners(0x20000, 0x8000, o, a);
    CHECK(o.size() == 1 && o[0].writer == b);
    // Re-tiled: no longer stale, still the writer (its copy is valid).
    t.markSynced(a);
    CHECK(t.ownerImage(0x20000, 0x4000) == 0);
    CHECK(t.writerAt(0x20000) == a);
    CHECK(t.ownerImage(0x24000, 1) == b);
    // A raw write over image b ends its claim on those bytes.
    t.markGpuWrite(0x24000, 0x4000, MemoryTracker::kGpuRaw);
    CHECK(t.ownerImage(0x24000, 0x4000) == 0);
    // A CPU write too.
    t.markGpuWrite(0x30000, 0x100, a);
    t.markCpuWrite(0x30000, 0x100);
    CHECK(t.ownerImage(0x30000, 0x100) == 0);
    // Released: ownerless, versions stay.
    const uint64_t v = t.markGpuWrite(0x40000, 0x100, b);
    t.release(b);
    CHECK(t.ownerImage(0x40000, 0x100) == 0);
    CHECK(t.version(0x40000, 0x100) == v);
}

void forgetting() {
    MemoryTracker t;
    const auto img = t.newImage();
    const uint64_t v = t.markGpuWrite(0x1000, 0x100, img);
    t.markCpuWrite(0x5000, 0x10);
    t.forgetHistory();
    CHECK(t.intervals() == 1);                   // the unsynced image claim survives
    CHECK(t.ownerImage(0x1000, 1) == img);
    CHECK(t.writtenByOthers(0x9000, 1, img, v));  // everything older than the floor is "unknown"
    CHECK(!t.writtenByOthers(0x9000, 1, img, t.current()));
}

void bounded() {
    MemoryTracker t;
    for (uint64_t i = 0; i < MemoryTracker::kMaxIntervals + 10; ++i) t.markCpuWrite(i * 64, 16);
    CHECK(t.intervals() < MemoryTracker::kMaxIntervals);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string_view n = argc > 1 ? argv[1] : "";
    if (n == "versions") versions();
    else if (n == "written_by_others") writtenByOthers();
    else if (n == "ownership") ownership();
    else if (n == "forgetting") forgetting();
    else if (n == "bounded") bounded();
    else return std::fprintf(stderr, "unknown test\n"), 2;
    return failures ? 1 : 0;
}
