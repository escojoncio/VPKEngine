#include "vpps4/renderer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <set>
#include <type_traits>
#include <unordered_map>

#include "vpgcn/lift.h"
#include "vpps4/fixed_function.h"
#include "vpps4/formats.h"
#include "vpps4/gpu_tiler.h"
#include "vpps4/foveation.h"
#include "vpps4/lifetime.h"
#include "vpps4/msaa.h"
#include "translation_cache.h"
#include "vpps4/report.h"
#include "vpps4/tiling.h"

namespace vpps4 {

using vp4gpu::Access;
using vp4gpu::BufferId;
using vp4gpu::ResourceRange;
using vp4gpu::SamplerId;
using vp4gpu::TextureId;

namespace {

constexpr uint64_t kFnvBasis = 0xcbf29ce484222325ull, kFnvPrime = 0x100000001b3ull;

struct Hash {
    uint64_t h = kFnvBasis;
    static constexpr uint64_t kFnvBasisValue() { return kFnvBasis; }
    static uint64_t mix(uint64_t h, uint64_t w) {
        h = (h ^ w) * 0x9E3779B97F4A7C15ull;
        return h ^ (h >> 32);
    }
    // Eight bytes per step (the programs hashed are kilobytes long, byte-wise FNV was a hot spot).
    void bytes(const void* p, size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        uint64_t w;
        for (; n >= 8; n -= 8, b += 8) std::memcpy(&w, b, 8), h = mix(h, w);
        if (n) {
            w = 0;
            std::memcpy(&w, b, n);
            h = mix(h, w ^ (uint64_t(n) << 56));
        }
    }
    template <class T>
    void value(const T& v) {
        if constexpr (std::is_scalar_v<T> && sizeof(T) <= 8) {
            uint64_t w = 0;
            std::memcpy(&w, &v, sizeof v);
            h = mix(h, w ^ (uint64_t(sizeof v) << 56));
        } else {
            bytes(&v, sizeof v);
        }
    }
    void words(std::span<const uint32_t> w) {
        value(uint64_t(w.size()));
        bytes(w.data(), w.size_bytes());
    }
};

constexpr uint64_t kAddressMask = (uint64_t(1) << 48) - 1;  // guest virtual addresses are 48-bit

// Upload ring: memory of our own, page aligned, aliased as one gpu/core buffer per frame in flight.
// A frame's uploads stay intact until the ring comes round again kFrames submits later.
// Upload rings, one per submit in turn. A ring is reused kFrames submits later: kFrames must exceed
// the backend's frames in flight + 1 (the host's MetalOptions::framesInFlight, 6).
using Clock = std::chrono::steady_clock;
double msSince(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr unsigned kFrames = 8;
constexpr uint64_t kRingBytes = 16ull << 20;
constexpr uint64_t kPage = 16384;
constexpr uint64_t kGdsBytes = 64 * 1024;
constexpr uint64_t kDummyBytes = 64 * 1024;

vp4gpu::Primitive toPrimitive(PrimitiveType p) {
    switch (p) {
        case PrimitiveType::PointList: return vp4gpu::Primitive::Points;
        case PrimitiveType::LineList:
        case PrimitiveType::LineListAdj:
        case PrimitiveType::LineStripAdj:
        case PrimitiveType::LineLoop: return vp4gpu::Primitive::Lines;
        case PrimitiveType::LineStrip: return vp4gpu::Primitive::LineStrip;
        case PrimitiveType::TriangleStrip: return vp4gpu::Primitive::TriangleStrip;
        default: return vp4gpu::Primitive::Triangles;  // lists, and fans/quads/polygons once converted
    }
}

bool isAdjacency(PrimitiveType p) {
    return p == PrimitiveType::LineListAdj || p == PrimitiveType::LineStripAdj || p == PrimitiveType::TriangleListAdj ||
           p == PrimitiveType::TriangleStripAdj;
}

// Primitives Metal lacks: converted to an index list in the upload ring.
bool needsConversion(PrimitiveType p) {
    return p == PrimitiveType::TriangleFan || p == PrimitiveType::Polygon || p == PrimitiveType::QuadList ||
           p == PrimitiveType::QuadStrip || p == PrimitiveType::LineLoop || isAdjacency(p);
}

// Indices of the converted list for `n` input vertices (positions in the input order).
std::vector<uint32_t> convertedOrder(PrimitiveType p, uint32_t n) {
    std::vector<uint32_t> o;
    switch (p) {
        case PrimitiveType::TriangleFan:
        case PrimitiveType::Polygon:
            for (uint32_t i = 2; i < n; ++i) o.insert(o.end(), {0, i - 1, i});
            break;
        case PrimitiveType::QuadList:
            for (uint32_t i = 0; i + 3 < n; i += 4) o.insert(o.end(), {i, i + 1, i + 2, i, i + 2, i + 3});
            break;
        case PrimitiveType::QuadStrip:
            for (uint32_t i = 0; i + 3 < n; i += 2) o.insert(o.end(), {i, i + 1, i + 3, i, i + 3, i + 2});
            break;
        case PrimitiveType::LineLoop:
            for (uint32_t i = 0; i + 1 < n; ++i) o.insert(o.end(), {i, i + 1});
            if (n > 1) o.insert(o.end(), {n - 1, 0});
            break;
        // Adjacency without a geometry shader: the hardware rasterises the main vertices only.
        case PrimitiveType::LineListAdj:
            for (uint32_t i = 0; i + 3 < n; i += 4) o.insert(o.end(), {i + 1, i + 2});
            break;
        case PrimitiveType::LineStripAdj:
            for (uint32_t i = 0; i + 3 < n; ++i) o.insert(o.end(), {i + 1, i + 2});
            break;
        case PrimitiveType::TriangleListAdj:
            for (uint32_t i = 0; i + 5 < n; i += 6) o.insert(o.end(), {i, i + 2, i + 4});
            break;
        case PrimitiveType::TriangleStripAdj:
            // Main vertices are the even entries; they form an ordinary strip (odd triangles flipped).
            for (uint32_t k = 0; 2 * k + 6 <= n; ++k) {
                const uint32_t a = 2 * k, b = 2 * k + 2, c = 2 * k + 4;
                if (k & 1) o.insert(o.end(), {b, a, c});
                else o.insert(o.end(), {a, b, c});
            }
            break;
        default: break;
    }
    return o;
}

vp4ir::TexDim texDim(const ImageDesc& d) {
    switch (d.type) {
        case 8: return vp4ir::TexDim::D1;
        case 10: return vp4ir::TexDim::D3;
        case 11: return d.lastArray - d.baseArray + 1 > 6 ? vp4ir::TexDim::CubeArray : vp4ir::TexDim::Cube;
        case 12:
        case 13:
        case 15: return vp4ir::TexDim::D2Array;
        default: return vp4ir::TexDim::D2;
    }
}

vp4ir::Scalar texElem(const TexturePlan& p) {
    return p.elem == TexturePlan::Uint ? vp4ir::Scalar::U32 : p.elem == TexturePlan::Sint ? vp4ir::Scalar::I32 : vp4ir::Scalar::F32;
}

vp4gpu::BindKind bindKind(vp4ir::ResourceKind k) {
    switch (k) {
        case vp4ir::ResourceKind::Buffer: return vp4gpu::BindKind::Buffer;
        case vp4ir::ResourceKind::ConstantBuffer: return vp4gpu::BindKind::ConstantBuffer;
        case vp4ir::ResourceKind::Texture: return vp4gpu::BindKind::Texture;
        case vp4ir::ResourceKind::Sampler: return vp4gpu::BindKind::Sampler;
        case vp4ir::ResourceKind::StorageTexture: return vp4gpu::BindKind::StorageTexture;
        case vp4ir::ResourceKind::BufferSize: return vp4gpu::BindKind::BufferSize;
    }
    return vp4gpu::BindKind::Auto;
}

}  // namespace

struct Renderer::Impl {
    explicit Impl(Renderer& owner) : r(owner) {}
    Renderer& r;
    vp4gpu::Frame frame;

    // ---- Guest memory ---------------------------------------------------------------------------

    struct GuestBuffer {
        uint64_t size = 0;
        void* host = nullptr;
        BufferId id;
    };
    std::map<uint64_t, GuestBuffer> guestBuffers;  // by region base
    // Buffers replaced while frames in flight may still use them: destroyed kFrames submits later.
    struct Retired {
        BufferId id;
        uint64_t after;
    };
    std::vector<Retired> retired;

    // Inside draw()/dispatch() the host is asked about a mapping once per mapping: the last answer
    // serves every later address it covers (the hook may take a lock). Cleared on entering and
    // leaving, so a mapping change between calls is always seen.
    GuestRegion lastRegion;
    bool haveRegion = false, regionScope = false;
    struct RegionScope {
        Impl& i;
        explicit RegionScope(Impl& impl) : i(impl) { i.regionScope = true, i.haveRegion = false; }
        ~RegionScope() { i.regionScope = false, i.haveRegion = false; }
    };
    bool region(uint64_t address, uint64_t size, GuestRegion& out) {
        if (haveRegion && address >= lastRegion.base && address + size <= lastRegion.base + lastRegion.size) {
            out = lastRegion;
            return true;
        }
        if (!r.hooks_.region || !r.hooks_.region(address, size ? size : 1, out)) return false;
        if (!(address >= out.base && address + size <= out.base + out.size)) return false;
        if (regionScope) lastRegion = out, haveRegion = true;
        return true;
    }

    bool readGuest(uint64_t address, void* dst, size_t bytes) {
        GuestRegion g;
        if (!region(address, bytes, g)) return false;
        std::memcpy(dst, static_cast<const uint8_t*>(g.host) + (address - g.base), bytes);
        return true;
    }

    // The gpu/core buffer aliasing the guest mapping that holds [address, address + size).
    bool guestBuffer(uint64_t address, uint64_t size, BufferId& id, uint64_t& offset, uint64_t& available) {
        GuestRegion g;
        if (!region(address, 1, g)) return false;
        (void)size;
        GuestBuffer& b = guestBuffers[g.base];
        if (!b.id || b.size != g.size || b.host != g.host) {
            if (b.id) retired.push_back({b.id, r.frame_ + kFrames});
            b = {g.size, g.host, r.device_.createBuffer({g.size, g.host, "guest " + hex(g.base)})};
        }
        id = b.id;
        offset = address - g.base;
        available = g.size - offset;
        if (id) touch(address, size && size < available ? size : available);
        return bool(id);
    }

    // Guest ranges the GPU may read or write through work recorded since it was last idle
    // (start -> end, merged). DMA done on the CPU must wait for the GPU only when it overlaps.
    std::map<uint64_t, uint64_t> busy;
    bool busyAll = false;  // too many ranges: everything counts as busy
    static constexpr size_t kBusyRanges = 8192;
    // The ranges of submitted frames still running on the GPU, by device frame index (oldest first).
    struct InFlight {
        uint64_t frame;
        std::map<uint64_t, uint64_t> ranges;
        bool all;
    };
    std::deque<InFlight> inFlight;
    void retireBusy() {
        const uint64_t done = r.device_.completedFrames();
        while (!inFlight.empty() && inFlight.front().frame < done) inFlight.pop_front();
    }
    static bool overlaps(const std::map<uint64_t, uint64_t>& m, bool all, uint64_t a, uint64_t n) {
        if (all) return true;
        const uint64_t end = a + std::max<uint64_t>(n, 1);
        auto it = m.upper_bound(a);
        if (it != m.begin() && std::prev(it)->second > a) return true;
        return it != m.end() && it->first < end;
    }
    bool mayAccess(uint64_t a, uint64_t n) {
        if (overlaps(busy, busyAll, a, n)) return true;
        retireBusy();
        for (const InFlight& f : inFlight)
            if (overlaps(f.ranges, f.all, a, n)) return true;
        return false;
    }
    // The last range touch() left in `busy` (merged): most buffers of a draw lie in it already.
    // Whoever empties `busy` forgets it.
    uint64_t touchedLo = 0, touchedHi = 0;
    void touch(uint64_t a, uint64_t n) {
        if (busyAll || !n) return;
        uint64_t lo = a, hi = a + n < a ? ~uint64_t(0) : a + n;
        if (lo >= touchedLo && hi <= touchedHi) return;
        auto it = busy.upper_bound(lo);
        if (it != busy.begin()) {
            auto prev = std::prev(it);
            if (prev->second >= lo) {
                if (prev->second >= hi) {  // already covered
                    touchedLo = prev->first, touchedHi = prev->second;
                    return;
                }
                lo = prev->first;
                it = busy.erase(prev);
            }
        }
        while (it != busy.end() && it->first <= hi) {
            hi = std::max(hi, it->second);
            it = busy.erase(it);
        }
        busy[lo] = hi;
        touchedLo = lo, touchedHi = hi;
        if (busy.size() > kBusyRanges) {
            busy.clear();
            busyAll = true;
            touchedLo = touchedHi = 0;
        }
    }
    bool busyOverlaps(uint64_t a, uint64_t n) const {
        if (busyAll) return true;
        const uint64_t end = a + std::max<uint64_t>(n, 1);
        auto it = busy.upper_bound(a);
        if (it != busy.begin() && std::prev(it)->second > a) return true;
        return it != busy.end() && it->first < end;
    }

    // ---- GDS ------------------------------------------------------------------------------------
    // The global data share: 64 KiB of our own memory (shared with the CPU), one for the session.
    struct Gds {
        void* memory = nullptr;
        BufferId id;
    } gds;
    bool gdsBuffer() {
        if (gds.id) return true;
        if (!gds.memory) {
            gds.memory = std::aligned_alloc(kPage, kGdsBytes);
            if (!gds.memory) return false;
            std::memset(gds.memory, 0, kGdsBytes);
        }
        gds.id = r.device_.createBuffer({kGdsBytes, gds.memory, "vpps4 gds"});
        return bool(gds.id);
    }

    // A buffer of zeros for V#s that point nowhere (writes to it are dropped: cleared each use).
    struct Dummy {
        void* memory = nullptr;
        BufferId id;
    } dummy;
    bool dummyBuffer() {
        if (dummy.id) return true;
        if (!dummy.memory) dummy.memory = std::aligned_alloc(kPage, kDummyBytes);
        if (!dummy.memory) return false;
        std::memset(dummy.memory, 0, kDummyBytes);
        dummy.id = r.device_.createBuffer({kDummyBytes, dummy.memory, "vpps4 zeros"});
        return bool(dummy.id);
    }

    // ---- Occlusion queries (EVENT_WRITE ZPASS_DONE) -----------------------------------------------
    // Draws count the samples that pass into 64-bit slots of a host-aliased visibility buffer (Metal
    // counting mode, accumulating). A ZPASS_DONE closes the current slot: it is the interval of draws
    // since the previous one. The counter the guest reads is cumulative, so a snapshot's value is the
    // sum of the slots up to it; it is written to guest memory once the frame holding the draws has
    // finished on the GPU (drainOcclusion).
    static constexpr uint32_t kVisSlots = 4096;
    struct Vis {
        void* memory = nullptr;
        BufferId id;
        uint64_t seq = 0;   // slot the next draws count into (modulo kVisSlots)
        uint64_t total = 0; // cumulative count of the drained slots
        bool active = false;
    } vis;
    struct Snapshot {
        uint64_t frame, seq, address;
        uint32_t pairs;
    };
    std::deque<Snapshot> snapshots;

    bool ensureVis() {
        if (vis.memory) return true;
        vis.memory = std::aligned_alloc(kPage, (kVisSlots * 8 + kPage - 1) / kPage * kPage);
        if (!vis.memory) return false;
        std::memset(vis.memory, 0, (kVisSlots * 8 + kPage - 1) / kPage * kPage);
        vis.id = r.device_.createBuffer({(kVisSlots * 8 + kPage - 1) / kPage * kPage, vis.memory, "vpps4 visibility"});
        return bool(vis.id);
    }

    bool zpassDone(uint64_t address, uint32_t pairs) {
        if (!pairs || !address || snapshots.size() + 2 >= kVisSlots || !ensureVis()) return false;
        GuestRegion reg;
        const uint64_t bytes = (pairs - 1) * 16ull + 8;
        if (!r.hooks_.region || !r.hooks_.region(address, bytes, reg) || !reg.host) return false;
        vis.active = true;
        snapshots.push_back({r.device_.submittedFrames(), vis.seq, address, pairs});
        ++vis.seq;
        return true;
    }

    // Writes the snapshots whose frame finished (all of them when the caller waited for the GPU).
    void drainOcclusion(bool gpuIdle) {
        const uint64_t done = r.device_.completedFrames();
        while (!snapshots.empty() && (gpuIdle || snapshots.front().frame < done)) {
            const Snapshot s = snapshots.front();
            snapshots.pop_front();
            volatile uint64_t* slot = static_cast<volatile uint64_t*>(vis.memory) + s.seq % kVisSlots;
            vis.total += *slot;
            *slot = 0;
            GuestRegion reg;
            const uint64_t bytes = (s.pairs - 1) * 16ull + 8;
            if (!r.hooks_.region || !r.hooks_.region(s.address, bytes, reg) || !reg.host) continue;
            uint8_t* at = static_cast<uint8_t*>(reg.host) + (s.address - reg.base);
            constexpr uint64_t kValid = 1ull << 63;
            for (uint32_t p = 0; p < s.pairs; ++p) {
                const uint64_t v = (p == 0 ? (vis.total & ~kValid) : 0) | kValid;
                std::memcpy(at + 16ull * p, &v, 8);
            }
            // The results are memory the host wrote: images over that range are stale now.
            tracker.markCpuWrite(s.address, bytes);
        }
    }

    bool occlusionPending(uint64_t address, uint64_t size) const {
        for (const Snapshot& s : snapshots)
            if (address < s.address + (s.pairs - 1) * 16ull + 8 && s.address < address + size) return true;
        return false;
    }

    // ---- Upload ring ----------------------------------------------------------------------------

    struct Ring {
        void* memory = nullptr;
        BufferId id;
        uint64_t head = 0;
    };
    std::array<Ring, kFrames> rings{};
    unsigned ringIndex = 0;

    bool ensureRing() {
        Ring& ring = rings[ringIndex];
        if (!ring.memory) {
            ring.memory = std::aligned_alloc(kPage, kRingBytes);
            if (!ring.memory) return false;
            ring.id = r.device_.createBuffer({kRingBytes, ring.memory, "vpps4 upload " + std::to_string(ringIndex)});
        }
        return true;
    }

    uint64_t ringGeneration = 0;  // bumped when the current ring changes (uploads before it are gone)

    // User data (16 dwords per stage) repeats across draws: an identical copy uploaded earlier into
    // the current ring is bound again instead of a new one.
    struct UserDataCopy {
        std::array<uint32_t, 16> data;
        uint64_t at = 0, generation = ~uint64_t(0);
    };
    std::array<UserDataCopy, 8> userDataCopies;
    unsigned userDataNext = 0;
    std::optional<uint64_t> uploadUserData(const std::array<uint32_t, 16>& d) {
        for (const UserDataCopy& c : userDataCopies)
            if (c.generation == ringGeneration && c.data == d) {
                ++r.cpu_.userDataReused;
                return c.at;
            }
        auto at = upload(d.data(), sizeof d);
        if (!at) return at;
        UserDataCopy& c = userDataCopies[userDataNext++ % userDataCopies.size()];
        c.data = d;
        c.at = *at;
        c.generation = ringGeneration;
        return at;
    }

    std::optional<uint64_t> upload(const void* data, uint64_t bytes, uint64_t align = 256) {
        Ring& ring = rings[ringIndex];
        if (!ensureRing()) return std::nullopt;
        const uint64_t at = (ring.head + align - 1) / align * align;
        if (at + bytes > kRingBytes) return std::nullopt;
        std::memcpy(static_cast<uint8_t*>(ring.memory) + at, data, bytes);
        ring.head = at + bytes;
        ++r.stats_.uploads;
        return at;
    }

    // Room in the upload ring the GPU fills (not copied from the CPU).
    std::optional<uint64_t> reserve(uint64_t bytes, uint64_t align) {
        Ring& ring = rings[ringIndex];
        if (!ensureRing()) return std::nullopt;
        const uint64_t at = (ring.head + align - 1) / align * align;
        if (at + bytes > kRingBytes) return std::nullopt;
        ring.head = at + bytes;
        return at;
    }

    // ---- Indirect draws on the GPU timeline ---------------------------------------------------------
    // A compute kernel (hand-written MSL, run on the GPU between the passes) turns the PS4's draw
    // arguments into Metal's: it zeroes the draw when the count buffer says so (multi-draw with a
    // count in memory) and, for fan / quad / loop / polygon primitives, builds the triangle / line
    // list index buffer from the arguments (counts, first vertex, index buffer gather). Nothing is
    // read on the CPU. Fans, quad lists and quad strips are prefixes of one order, so the number of
    // indices is all that depends on the count; the rest of the capacity is simply not drawn.
    static constexpr uint32_t kConvCapNonIndexed = 16384, kConvCapMax = 65536;
    uint64_t convId = 0;

    static const char* convSource() {
        return R"MSL(
#include <metal_stdlib>
using namespace metal;
struct VpArgs {
    device const uint* src;
    device uint* dstArgs;
    device const uint* srcIdx;
    device uint* dstIdx;
    device const uint* cnt;
    constant uint* prm;
};
static uint vp_pos(uint prim, uint k, uint n) {
    const uint quad[6] = {0, 1, 2, 0, 2, 3};
    const uint strip[6] = {0, 1, 3, 0, 3, 2};
    switch (prim) {
        case 1: { uint t = k / 3, j = k % 3; return j == 0 ? 0u : t + j; }
        case 2: return 4 * (k / 6) + quad[k % 6];
        case 3: return 2 * (k / 6) + strip[k % 6];
        default: { uint s = k / 2, j = k % 2; if (s + 1 < n) return s + j; return j == 0 ? n - 1 : 0u; }
    }
}
kernel void vp_main(constant VpArgs& a [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    const uint prim = a.prm[0], indexed = a.prm[1], wide = a.prm[2], cap = a.prm[3];
    const uint drawIndex = a.prm[4], hasCount = a.prm[5], avail = a.prm[6];
    const bool live = hasCount == 0 || drawIndex < a.cnt[0];
    const uint n = live ? a.src[0] : 0u;
    const uint inst = live ? a.src[1] : 0u;
    const uint start = a.src[2];
    const uint baseV = indexed != 0 ? a.src[3] : 0u;
    const uint baseI = indexed != 0 ? a.src[4] : a.src[3];
    if (prim == 0) {
        if (tid == 0) {
            a.dstArgs[0] = n; a.dstArgs[1] = inst; a.dstArgs[2] = start;
            if (indexed != 0) { a.dstArgs[3] = baseV; a.dstArgs[4] = baseI; } else { a.dstArgs[3] = baseI; }
        }
        return;
    }
    uint outN = 0;
    switch (prim) {
        case 1: outN = n >= 3 ? (n - 2) * 3 : 0u; break;
        case 2: outN = (n / 4) * 6; break;
        case 3: outN = n >= 4 ? ((n - 2) / 2) * 6 : 0u; break;
        default: outN = n >= 2 ? 2 * n : 0u; break;
    }
    const uint total = min(outN, cap);
    if (tid == 0) { a.dstArgs[0] = total; a.dstArgs[1] = inst; a.dstArgs[2] = 0; a.dstArgs[3] = baseV; a.dstArgs[4] = baseI; }
    if (tid < total) {
        const uint p = vp_pos(prim, tid, n);
        uint v;
        if (indexed != 0) {
            const uint at = start + p;
            v = at < avail ? (wide != 0 ? a.srcIdx[at] : uint(((device const ushort*)a.srcIdx)[at])) : 0u;
        } else {
            v = start + p;
        }
        a.dstIdx[tid] = v;
    }
}
)MSL";
    }

    bool ensureConv() {
        if (convId) return true;
        vp4msl::MslShader m;
        m.source = convSource();
        m.workgroupSize = {64, 1, 1};
        using K = vp4ir::ResourceKind;
        m.arguments = {{"src", K::Buffer, 0, 0},    {"dstArgs", K::Buffer, 1, 8}, {"srcIdx", K::Buffer, 2, 16},
                       {"dstIdx", K::Buffer, 3, 24}, {"cnt", K::Buffer, 4, 32},    {"prm", K::ConstantBuffer, 0, 40}};
        convId = vp4msl::shaderId(m);
        if (r.hooks_.shader) r.hooks_.shader(convId, vp4ir::Stage::Compute, m);
        return true;
    }

    // Records the converter for draw `d` (arguments at guest buffer `argBuf`+`argOff`) and points
    // `draw` at its output: `indirect` is always set; with `conv` the index buffer and count too.
    bool convertIndirect(const DrawState& d, bool conv, BufferId argBuf, uint64_t argOff, vp4gpu::Indirect& indirect,
                         vp4gpu::Draw& draw, std::string& why) {
        ensureConv();
        vp4gpu::PipelineState ps;
        ps.computeShader = convId;
        ps.label = "indirect draw converter";
        const vp4gpu::PipelineCache::Status st = r.device_.pipelines().request(ps, vp4gpu::Priority::Visible);
        if (st == vp4gpu::PipelineCache::Status::Failed) return why = "indirect converter failed", false;
        if (st != vp4gpu::PipelineCache::Status::Ready) return why = "indirect converter compiling", false;

        const bool indexed = d.indexAddress != 0;
        uint32_t cap = 16;
        uint32_t avail = 0;
        BufferId ib;
        uint64_t ioff = 0;
        if (indexed) {
            uint64_t iavail = 0;
            if (!guestBuffer(d.indexAddress, 4, ib, ioff, iavail)) return why = "index buffer outside guest memory", false;
            if (ioff % (d.index32 ? 4 : 2)) return why = "misaligned index buffer", false;
            avail = uint32_t(std::min<uint64_t>(iavail / (d.index32 ? 4 : 2), UINT32_MAX));
        }
        if (conv) cap = indexed ? std::clamp<uint32_t>((d.count * 2 + 8 + 63) / 64 * 64, 64, kConvCapMax) : kConvCapNonIndexed;
        BufferId cb;
        uint64_t coff = 0;
        if (d.indirectCountAddress) {
            uint64_t cavail = 0;
            if (!guestBuffer(d.indirectCountAddress, 4, cb, coff, cavail) || cavail < 4 || coff % 4)
                return why = "indirect count outside guest memory", false;
        }
        auto outArgs = reserve(32, 16);
        auto outIdx = reserve(conv ? uint64_t(cap) * 4 : 16, 16);
        const uint32_t prim = !conv ? 0 : (d.primitive == PrimitiveType::TriangleFan || d.primitive == PrimitiveType::Polygon) ? 1
                              : d.primitive == PrimitiveType::QuadList ? 2
                              : d.primitive == PrimitiveType::QuadStrip ? 3 : 4;
        const uint32_t prm[8] = {prim, indexed ? 1u : 0u, d.index32 ? 1u : 0u, cap, d.drawIndex, d.indirectCountAddress ? 1u : 0u, avail, 0};
        auto prmAt = upload(prm, sizeof prm, 16);
        if (!outArgs || !outIdx || !prmAt) return why = "upload ring full", false;
        const BufferId ring = rings[ringIndex].id;

        vp4gpu::Dispatch x;
        auto buf = [&](uint16_t slot, BufferId b, uint64_t off, uint64_t bytes, bool write) {
            Access a;
            a.range = ResourceRange::bytes(b, off, bytes);
            a.stages = vp4gpu::stage::compute;
            a.kind = vp4gpu::BindKind::Buffer;
            a.slot = slot;
            a.write = write;
            x.bindings.push_back(a);
        };
        buf(0, argBuf, argOff, indexed ? 20 : 16, false);
        buf(1, ring, *outArgs, 20, true);
        if (indexed) buf(2, ib, ioff, uint64_t(avail) * (d.index32 ? 4 : 2), false);
        else buf(2, ring, *outArgs + 20, 4, false);  // (unused)
        buf(3, ring, *outIdx, conv ? uint64_t(cap) * 4 : 4, conv);
        if (d.indirectCountAddress) buf(4, cb, coff, 4, false);
        else buf(4, ring, *outArgs + 24, 4, false);  // (unused)
        Access c;
        c.range = ResourceRange::bytes(ring, *prmAt, sizeof prm);
        c.stages = vp4gpu::stage::compute;
        c.kind = vp4gpu::BindKind::ConstantBuffer;
        c.slot = 0;
        x.bindings.push_back(c);
        x.pipeline = ps.key();
        x.groups = {conv ? cap / 64 : 1u, 1, 1};
        x.label = ps.label;
        closePass();
        frame.dispatch(std::move(x));
        ++r.stats_.dispatched;

        indirect = vp4gpu::Indirect{ring, *outArgs};
        if (conv) {
            draw.index = vp4gpu::IndexBuffer{ring, *outIdx, true};
            draw.count = cap;
            if (!indexed) note("indirect fan/quad/loop draw: up to 16384 indices drawn");
            note("fan/quad/loop primitives converted to lists");
        }
        return true;
    }

    // ---- Diagnostics ----------------------------------------------------------------------------

    static std::string hex(uint64_t v) {
        char b[24];
        std::snprintf(b, sizeof b, "0x%llx", static_cast<unsigned long long>(v));
        return b;
    }
    void log(const std::string& s) {
        if (r.hooks_.log) r.hooks_.log(s);
    }
    // Every skip / note is also reported to the host's sink (report.h): the census counts them all.
    // Notes about something the renderer cannot do ("... unsupported", "... not detiled") are
    // "unsupported", the rest "approx".
    static const char* noteCategory(const char* n) {
        return std::strstr(n, "unsupported") || std::strstr(n, "not detiled") || std::strstr(n, "left unfilled")
                   ? "unsupported"
                   : "approx";
    }
    // The reports of one frame are counted here and handed to the sink once at the next submit (flushReports):
    // the per-event cost of a literal reason is one pointer-hash probe and two increments. (The renderer is
    // driven from one thread, as the stats it keeps are.)
    struct Lit {
        const char* key = nullptr;  // the literal's address
        const char* category = nullptr;
        uint64_t* counter = nullptr;  // in the stats map (nodes do not move)
        uint64_t pending = 0;         // not yet reported
    };
    struct LitTable {
        static constexpr size_t kSlots = 1024, kMax = 768;
        std::array<Lit, kSlots> slot{};
        size_t used = 0;
        std::vector<uint16_t> dirty;  // slots with pending > 0
        Lit* find(const char* k, bool insert) {
            size_t i = size_t((reinterpret_cast<uintptr_t>(k) >> 2) * 0x9E3779B97F4A7C15ull >> 54) & (kSlots - 1);
            for (size_t n = 0; n < kSlots; ++n, i = (i + 1) & (kSlots - 1)) {
                if (slot[i].key == k) return &slot[i];
                if (!slot[i].key) {
                    if (!insert || used >= kMax) return nullptr;
                    ++used;
                    slot[i].key = k;
                    return &slot[i];
                }
            }
            return nullptr;
        }
        uint16_t index(const Lit* l) const { return uint16_t(l - slot.data()); }
    };
    LitTable skipLits, noteLits;
    void countLit(LitTable& t, std::map<std::string, uint64_t>& m, const char* k, const char* category, const char* kind) {
        Lit* l = t.find(k, true);
        if (!l) {  // (table full: the slow way)
            uint64_t& c = m[k];
            if (c++ == 0) log(std::string("vpps4.first ") + kind + "=\"" + k + "\"");
            vpps4_report(category, k, "");
            return;
        }
        if (!l->counter) {
            l->category = category;
            l->counter = &m[k];
            if ((*l->counter)++ == 0) log(std::string("vpps4.first ") + kind + "=\"" + k + "\"");
        } else {
            ++*l->counter;
        }
        if (l->pending++ == 0) t.dirty.push_back(t.index(l));
    }
    void flushLits(LitTable& t) {
        for (uint16_t i : t.dirty) {
            Lit& l = t.slot[i];
            vpps4_report_count(l.category, l.key, "", l.pending);
            l.pending = 0;
        }
        t.dirty.clear();
    }
    // The counted reports to the sink (once per submit and when the renderer goes away).
    void flushReports() {
        flushLits(skipLits);
        flushLits(noteLits);
    }
    void skip(const std::string& reason) {
        vpps4_report("draw-skip", reason.c_str(), "");
        if (r.stats_.skipped[reason]++ == 0) log("vpps4.first skip=\"" + reason + "\"");
    }
    void note(const std::string& n) {
        vpps4_report(noteCategory(n.c_str()), n.c_str(), "");
        if (r.stats_.notes[n]++ == 0) log("vpps4.first note=\"" + n + "\"");
    }
    // Literal reasons (the per-draw ones): counted through the literal's address, no string built.
    void skip(const char* reason) { countLit(skipLits, r.stats_.skipped, reason, "draw-skip", "skip"); }
    void note(const char* n) {
        // (the category is worked out once per literal, on its first sighting)
        Lit* l = noteLits.find(n, false);
        countLit(noteLits, r.stats_.notes, n, l && l->category ? l->category : noteCategory(n), "note");
    }

    // ---- Shaders --------------------------------------------------------------------------------

    // A translation error to the host's sink: key = stage and the reason, detail = the program key.
    void reportShaderError(vpgcn::ShaderStage stage, uint64_t programKey, const std::string& error) {
        const char* names[] = {"vs", "ps", "cs", "es", "gs", "copy-vs"};
        const unsigned st = unsigned(stage);
        vpps4_report("shader", ((st < 6 ? names[st] : "stage") + std::string(": ") + error).c_str(), hex(programKey).c_str());
    }

    struct Translated {
        bool ok = false;
        std::string error;
        uint64_t id = 0;
        std::vector<vpgcn::Binding> bindings;
        std::vector<vp4ir::Resource> resources;  // Module::resources (kind index, access)
        std::optional<uint32_t> userDataSlot;
        std::vector<int16_t> snapIndex;  // per binding: its place in the draw's descriptor snapshot, or -1
        bool fromCache = false;          // loaded from the persistent cache
        bool touched = false;            // ... and the cache was told this run used it
    };
    // The descriptors shader() read for the draw's probe list, kept for bind() (per stage).
    struct Snap {
        std::array<uint32_t, 8> w;
        bool ok;
    };
    std::array<std::vector<Snap>, 6> snaps;
    // Level 1: what the program reads (bindings), and which of them the descriptors must be probed
    // for (first binding of each source that is a texture or a typed buffer).
    struct ProgramInfo {
        std::vector<vpgcn::Binding> bindings;
        std::vector<uint16_t> probe;
    };
    std::unordered_map<uint64_t, ProgramInfo> sourcesByProgram;

    // The bindings whose descriptors are probed: the first of each source that is a texture or a
    // typed buffer.
    static std::vector<uint16_t> probeList(const std::vector<vpgcn::Binding>& bindings) {
        std::vector<uint16_t> probe;
        for (size_t i = 0; i < bindings.size(); ++i) {
            const vpgcn::Binding& b = bindings[i];
            const bool tex = b.kind == vp4ir::ResourceKind::Texture || b.kind == vp4ir::ResourceKind::StorageTexture;
            if (!b.dwords || !(tex || b.kind == vp4ir::ResourceKind::Buffer)) continue;
            bool seen = false;
            for (uint16_t j : probe) seen = seen || bindings[j].source == b.source;
            if (!seen) probe.push_back(uint16_t(i));
        }
        return probe;
    }
    // Per binding of a translation: its place in the program's probe list, or -1.
    static std::vector<int16_t> snapIndexFor(const ProgramInfo& pi, const std::vector<vpgcn::Binding>& bindings) {
        std::vector<int16_t> out;
        for (const vpgcn::Binding& lb : bindings) {
            int16_t at = -1;
            for (size_t j = 0; j < pi.probe.size() && at < 0; ++j) {
                const vpgcn::Binding& pb = pi.bindings[pi.probe[j]];
                if (pb.source == lb.source && pb.kind == lb.kind && pb.dwords == lb.dwords) at = int16_t(j);
            }
            out.push_back(at);
        }
        return out;
    }

    // ---- Persistent translations ------------------------------------------------------------------
    std::unique_ptr<tcache::Worker> cache;
    bool cacheLoading = false;
    tcache::Loaded pendingLoad;
    size_t mergedPrograms = 0, mergedTranslations = 0, mergedPipelines = 0;

    static vp4ir::Stage irStage(vpgcn::ShaderStage stage) {
        return stage == vpgcn::ShaderStage::Vertex || stage == vpgcn::ShaderStage::CopyVertex ? vp4ir::Stage::Vertex
             : stage == vpgcn::ShaderStage::Pixel ? vp4ir::Stage::Fragment
                                                   : vp4ir::Stage::Compute;
    }
    // Takes what the cache thread loaded, a slice per call (the shaders are handed to the host's
    // provider here, on the render thread, without a long pause).
    void pollCache() {
        if (!cacheLoading) return;
        if (pendingLoad.programs.empty() && pendingLoad.translations.empty() && pendingLoad.pipelines.empty() &&
            pendingLoad.failures.empty() && mergedPrograms == 0 && mergedTranslations == 0 && mergedPipelines == 0) {
            if (!cache->take(pendingLoad)) return;
            pendingByKey2.clear();
            for (size_t i = 0; i < pendingLoad.translations.size(); ++i) pendingByKey2.emplace(pendingLoad.translations[i].key2, i);
            // Programs the lifter could not translate in an earlier run: failed again without trying.
            for (tcache::FailureRec& f : pendingLoad.failures) {
                Translated t;
                t.error = std::move(f.error);
                t.fromCache = true;
                if (f.level == 1) sourcesByProgram.emplace(f.key, ProgramInfo{});
                if (translated.emplace(f.key, t).second) {
                    ++r.stats_.shaderErrors;
                    reportShaderError(vpgcn::ShaderStage(f.stage), f.key, translated[f.key].error);
                }
            }
            pendingLoad.failures.clear();
            pendingLoad.failures.shrink_to_fit();
        }
        for (; mergedPrograms < pendingLoad.programs.size(); ++mergedPrograms) {
            tcache::ProgramRec& p = pendingLoad.programs[mergedPrograms];
            ProgramInfo pi;
            pi.bindings = std::move(p.bindings);
            pi.probe = probeList(pi.bindings);
            sourcesByProgram.emplace(p.key1, std::move(pi));
        }
        for (unsigned n = 0; n < 128 && mergedTranslations < pendingLoad.translations.size(); ++mergedTranslations, ++n)
            mergeTranslation(mergedTranslations);
        if (mergedTranslations < pendingLoad.translations.size()) return;
        // The pipelines earlier runs built, compiled ahead (their shaders are registered now).
        for (unsigned n = 0; n < 256 && mergedPipelines < pendingLoad.pipelines.size(); ++mergedPipelines, ++n)
            r.device_.pipelines().request(pendingLoad.pipelines[mergedPipelines], vp4gpu::Priority::Prefetch);
        if (mergedPipelines < pendingLoad.pipelines.size()) return;
        pendingLoad = {};
        pendingByKey2 = {};
        cacheLoading = false;
    }
    // One loaded translation into the level-2 table (id 0 marks the ones merged already).
    void mergeTranslation(size_t i) {
        tcache::TranslationRec& t = pendingLoad.translations[i];
        if (!t.id) return;
        auto pi = sourcesByProgram.find(t.key1);
        if (pi == sourcesByProgram.end() || translated.count(t.key2)) {
            t.id = 0;
            return;
        }
        Translated x;
        x.ok = true;
        x.fromCache = true;
        x.id = t.id;
        x.bindings = std::move(t.bindings);
        x.resources = std::move(t.resources);
        x.userDataSlot = t.userDataSlot;
        x.snapIndex = snapIndexFor(pi->second, x.bindings);
        if (registered.insert(x.id).second && r.hooks_.shader)
            r.hooks_.shader(x.id, irStage(vpgcn::ShaderStage(t.stage)), t.shader);
        translated.emplace(t.key2, std::move(x));
        t.id = 0;
        t.shader = {};
        ++r.stats_.shadersCached;
    }
    // A draw needs the translation of `key2` before the slice merge reached it: merged now, so it is
    // not translated again (and appended a second time).
    const Translated* mergeNow(uint64_t key2) {
        if (!cacheLoading) return nullptr;
        auto at = pendingByKey2.find(key2);
        if (at == pendingByKey2.end()) return nullptr;
        mergeTranslation(at->second);
        auto it = translated.find(key2);
        return it == translated.end() ? nullptr : &it->second;
    }
    std::unordered_map<uint64_t, size_t> pendingByKey2;  // of pendingLoad.translations
    std::unordered_map<uint64_t, Translated> translated;                          // level 2
    std::set<uint64_t> registered;

    // The colour export conversions of the draw being recorded (set by draw() before it asks for the
    // pixel shader; part of the shader's identity).
    std::array<vpgcn::ExportConv, 8> curExport{};

    static vpgcn::LiftInput liftInput(const Stage& s, vpgcn::ShaderStage stage, const std::array<vpgcn::ExportConv, 8>* conv = nullptr) {
        vpgcn::LiftInput in;
        if (conv && stage == vpgcn::ShaderStage::Pixel)
            for (unsigned k = 0; k < 8; ++k) in.exportConv[k] = (*conv)[k];
        in.stage = stage;
        in.code = s.code;
        in.fetch = s.fetch;
        in.userSgprs = s.userSgprs;
        in.vgprCompCnt = s.vgprCompCnt;
        in.vsOutCntl = s.vsOutCntl;
        in.rectList = s.rectList;
        in.geom.esgsItemDwords = s.geom.esgsItemDwords;
        in.geom.gsvsVertDwords = s.geom.gsvsVertDwords;
        in.geom.maxVertOut = s.geom.maxVertOut;
        in.geom.inputPrim = s.geomInputPrim;
        in.geom.outPrim = s.geom.outPrim;
        in.geom.gsInstances = std::max(s.geom.gsInstances, 1u);
        in.psInputAddr = s.psInputAddr;
        in.instanceStepRate[0] = s.instanceStepRate[0];
        in.instanceStepRate[1] = s.instanceStepRate[1];
        for (unsigned k = 0; k < 3; ++k) {
            in.tgidEnable[k] = s.tgidEnable[k];
            in.workgroupSize[k] = s.workgroupSize[k];
        }
        in.tgSizeEnable = s.tgSizeEnable;
        in.dualSource = s.dualSource;
        in.clipSpaceGL = s.clipSpaceGL;
        in.tidigCompCnt = s.tidigCompCnt;
        in.ldsBytes = s.ldsBytes;
        // SPI_SHADER_COL_FORMAT: 4 FP16_ABGR, 5 UNORM16, 6 SNORM16, 7 UINT16, 8 SINT16 are the packed ones.
        for (unsigned k = 0; k < 8; ++k) {
            const uint8_t f = s.exportFormat[k];
            in.compressedExport[k] = f >= 4 && f <= 8 ? vpgcn::CompressedExport(f - 4) : vpgcn::CompressedExport::Fp16;
        }
        if (stage == vpgcn::ShaderStage::Pixel)
            in.psInputCntl.assign(s.psInputCntl.begin(), s.psInputCntl.begin() + std::min<size_t>(s.psNumInterp, 32));
        return in;
    }

    static uint64_t programKey(const Stage& s, vpgcn::ShaderStage stage, bool withCode = true) {
        Hash h;
        h.value(stage);
        if (withCode) {
            h.words(s.code);
            h.words(s.fetch);
        }
        h.value(s.userSgprs);
        h.value(s.vgprCompCnt);
        if (stage == vpgcn::ShaderStage::Pixel) {
            h.value(s.psInputAddr);
            h.value(s.exportFormat);
            h.value(s.psNumInterp);
            h.value(s.dualSource);
            h.bytes(s.psInputCntl.data(), 4 * std::min<size_t>(s.psNumInterp, 32));
        }
        if (stage == vpgcn::ShaderStage::Vertex || stage == vpgcn::ShaderStage::Export) h.value(s.instanceStepRate);
        if (stage == vpgcn::ShaderStage::Vertex) h.value(s.clipSpaceGL);
        if (stage == vpgcn::ShaderStage::Vertex || stage == vpgcn::ShaderStage::CopyVertex) {
            h.value(s.vsOutCntl);
            h.value(s.rectList);
        }
        if (stage == vpgcn::ShaderStage::Export || stage == vpgcn::ShaderStage::Geometry ||
            stage == vpgcn::ShaderStage::CopyVertex) {
            h.value(s.geom.esgsItemDwords);
            h.value(s.geom.gsvsVertDwords);
            h.value(s.geom.maxVertOut);
            h.value(s.geom.outPrim);
            h.value(s.geom.gsInstances);
            h.value(s.geomInputPrim);
        }
        if (stage == vpgcn::ShaderStage::Compute) {
            h.value(s.workgroupSize);
            h.value(s.tgidEnable);
            h.value(s.tgSizeEnable);
            h.value(s.tidigCompCnt);
            h.value(s.ldsBytes);
        }
        return h.h;
    }

    // The dwords of a descriptor (dwords > 0, at most 8) at its source, at draw time.
    bool readDescriptor(const Stage& s, const vpgcn::DescriptorSource& src, unsigned dwords, uint32_t* out) {
        if (dwords > 8) return false;
        uint64_t at;
        switch (src.kind) {
            case vpgcn::DescriptorSource::UserData:
                if (src.sgpr + dwords > 16) return false;
                std::copy_n(s.userData.begin() + src.sgpr, dwords, out);
                return true;
            case vpgcn::DescriptorSource::Code:
                if (src.path.empty()) return false;
                at = s.address + src.path[0];
                break;
            case vpgcn::DescriptorSource::Pointer:
                if (!pointerTarget(s, src, at)) return false;
                break;
            default: return false;
        }
        return readGuest(at, out, 4 * dwords);
    }

    // Pointer source: where the path ends (the descriptor's, or the stored address's, location).
    bool pointerTarget(const Stage& s, const vpgcn::DescriptorSource& src, uint64_t& at) {
        if (src.sgpr + 2 > 16 || src.path.empty()) return false;
        at = (uint64_t(s.userData[src.sgpr]) | uint64_t(s.userData[src.sgpr + 1]) << 32) & kAddressMask;
        for (size_t k = 0; k + 1 < src.path.size(); ++k) {
            uint64_t next = 0;
            if (!readGuest(at + src.path[k], &next, 8)) return false;
            at = next & kAddressMask;
        }
        at += src.path.back();
        return true;
    }

    // A binding with dwords 0: the source holds the address of the memory to bind.
    bool memoryAddress(const Stage& s, const vpgcn::DescriptorSource& src, uint64_t& address) {
        switch (src.kind) {
            case vpgcn::DescriptorSource::UserData:
                if (src.sgpr + 2 > 16) return false;
                address = (uint64_t(s.userData[src.sgpr]) | uint64_t(s.userData[src.sgpr + 1]) << 32) & kAddressMask;
                return true;
            case vpgcn::DescriptorSource::Code:
                if (src.path.empty()) return false;
                address = s.address + src.path[0];
                return true;
            case vpgcn::DescriptorSource::Pointer: {
                uint64_t at;
                if (!pointerTarget(s, src, at) || !readGuest(at, &address, 8)) return false;
                address &= kAddressMask;
                return true;
            }
            default: return false;
        }
    }

    // The words of a program the lifter refused, in the log (no device debugger: a later session
    // turns them into a lifter test). The first few programs only, at most 2048 words each.
    unsigned dumped = 0;
    void dumpProgram(const Stage& s, uint64_t key) {
        if (dumped >= 8) return;
        ++dumped;
        auto lines = [&](const char* what, std::span<const uint32_t> w) {
            const size_t n = std::min<size_t>(w.size(), 2048);
            for (size_t at = 0; at < n; at += 128) {
                std::string l = "vpps4.shader dump key=" + hex(key) + " " + what + "@" + std::to_string(at) + ":";
                char buf[12];
                for (size_t k = at; k < std::min(n, at + 128); ++k) {
                    std::snprintf(buf, sizeof buf, " %08x", w[k]);
                    l += buf;
                }
                log(l);
            }
        };
        log("vpps4.shader dump key=" + hex(key) + " userSgprs=" + std::to_string(s.userSgprs) +
            " code=" + std::to_string(s.code.size()) + " fetch=" + std::to_string(s.fetch.size()) + " words");
        lines("code", s.code);
        lines("fetch", s.fetch);
    }

    // programKey hashes the whole program (kilobytes) byte by byte: within one submit, and while
    // the CPU reports no write (invalidate), the same code/fetch spans with the same other state
    // reuse the key found before. Direct mapped by the code address.
    struct ProgramMemo {
        const uint32_t* code = nullptr;
        const uint32_t* fetch = nullptr;
        size_t codeWords = 0, fetchWords = 0;
        uint64_t rest = 0, key = 0, generation = ~uint64_t(0);
    };
    std::array<ProgramMemo, 64> programMemo;
    uint64_t programGeneration = 0;
    uint64_t programKeyCached(const Stage& s, vpgcn::ShaderStage stage) {
        const uint64_t rest = programKey(s, stage, false);
        ProgramMemo& m = programMemo[(reinterpret_cast<uintptr_t>(s.code.data()) >> 4 ^ uintptr_t(stage)) % programMemo.size()];
        if (m.generation == programGeneration && m.code == s.code.data() && m.codeWords == s.code.size() &&
            m.fetch == s.fetch.data() && m.fetchWords == s.fetch.size() && m.rest == rest)
            return m.key;
        m = {s.code.data(), s.fetch.data(), s.code.size(), s.fetch.size(), rest, programKey(s, stage), programGeneration};
        return m.key;
    }

    // What a descriptor's dwords say about the slot (the lifter's DescriptorInfo). The answer depends on
    // the descriptor, on the render targets that exist (planFor) and on the pass's sample counts, so it
    // is memoised on those: direct mapped, valid for one texGen.
    struct InfoMemo {
        std::array<uint32_t, 8> w{};
        uint8_t kind = 0, dwords = 0, cov = 0, frag = 0;
        uint64_t gen = ~0ull;
        bool ok = false;
        vpgcn::DescriptorInfo info;
    };
    std::array<InfoMemo, 128> infoMemo;
    bool info_of(const vpgcn::Binding& b, const uint32_t* w, vpgcn::DescriptorInfo& info) {
        const bool image = b.kind == vp4ir::ResourceKind::Texture || b.kind == vp4ir::ResourceKind::StorageTexture;
        if (!image && b.kind != vp4ir::ResourceKind::Buffer) return false;
        if (b.dwords > 8) return false;
        InfoMemo m;
        m.w.fill(0);
        std::copy(w, w + b.dwords, m.w.begin());
        m.kind = uint8_t(b.kind), m.dwords = uint8_t(b.dwords), m.cov = msCoverage, m.frag = msFragments;
        uint64_t hk = texGen * 0x9e3779b97f4a7c15ull ^ (uint64_t(m.kind) << 8 | m.dwords) ^ (uint64_t(m.cov) << 20) ^ (uint64_t(m.frag) << 28);
        for (uint32_t x : m.w) hk = (hk ^ x) * 0x100000001b3ull;
        InfoMemo& slot = infoMemo[(hk ^ hk >> 29) % infoMemo.size()];
        if (slot.gen == texGen && slot.w == m.w && slot.kind == m.kind && slot.dwords == m.dwords && slot.cov == m.cov &&
            slot.frag == m.frag) {
            info = slot.info;
            return slot.ok;
        }
        vpgcn::DescriptorInfo out;
        bool ok = true;
        if (image) {
            const ImageDesc d = decodeImage({w, b.dwords});
            bool depthTarget = false;
            const TexturePlan plan = planFor(d, b.kind == vp4ir::ResourceKind::StorageTexture, depthTarget);
            out.dim = texDim(d);
            out.elem = texElem(plan);
            out.depth = depthTarget;
            out.conv = plan.conv;
            composeSelect(d.dstSel, plan.conv.select, out.conv.select);
            if (plan.conv.manual) out.dfmt = d.dataFormat, out.nfmt = d.numFormat;
            if (isFmaskFormat(d.dataFormat)) {
                // FMASK: the host keeps its samples uncompressed; reads answer with the identity map.
                const FmaskValue v = fmaskIdentity(msCoverage, msFragments);
                out.fmask = true;
                out.fmaskLo = v.lo;
                out.fmaskHi = v.hi;
            } else {
                out.multisample = readsSamples(d);
            }
        } else {
            const BufferDesc d = decodeBuffer({w, b.dwords});
            out.dfmt = d.dataFormat;
            out.nfmt = d.numFormat;
        }
        m.gen = texGen, m.ok = ok, m.info = out;
        slot = m;
        info = out;
        return ok;
    }

    const Translated* shader(const Stage& s, vpgcn::ShaderStage stage) {
        const uint64_t key1 = programKeyCached(s, stage);
        auto it1 = sourcesByProgram.find(key1);
        if (it1 == sourcesByProgram.end()) {
            vpgcn::LiftResult first = vpgcn::lift(liftInput(s, stage));
            ProgramInfo pi;
            if (first.errors.empty()) {
                pi.bindings = first.bindings;
                pi.probe = probeList(pi.bindings);
                if (cache) cache->append(tcache::ProgramRec{key1, pi.bindings});
            }
            it1 = sourcesByProgram.emplace(key1, std::move(pi)).first;
            if (!first.errors.empty()) {
                Translated t;
                t.error = first.errors.front();
                translated.emplace(key1, t);  // level 2 keyed by key1 alone: the program fails whatever the state
                if (cache) cache->append(tcache::FailureRec{1, key1, uint8_t(stage), t.error});
                ++r.stats_.shaderErrors;
                log("vpps4.shader error stage=" + std::to_string(int(stage)) + " key=" + hex(key1) + " \"" + t.error + "\"");
                reportShaderError(stage, key1, t.error);
                dumpProgram(s, key1);
                return &translated[key1];
            }
        }
        if (auto bad = translated.find(key1); bad != translated.end() && !bad->second.ok) return &bad->second;

        // What the real descriptors say about the texture and typed-buffer slots.
        Hash h;
        h.value(key1);
        std::vector<Snap>& snap = snaps[size_t(stage)];
        snap.clear();
        for (uint16_t i : it1->second.probe) {
            const vpgcn::Binding& b = it1->second.bindings[i];
            Snap& sn = snap.emplace_back();
            sn.ok = b.dwords <= 8 && readDescriptor(s, b.source, b.dwords, sn.w.data());
            vpgcn::DescriptorInfo info;
            if (!sn.ok || !info_of(b, sn.w.data(), info)) continue;
            h.value(b.slot);
            h.value(info.dim);
            h.value(info.elem);
            h.value(info.depth);
            h.value(info.multisample);
            h.value(info.fmask);
            h.value(info.fmaskLo);
            h.value(info.fmaskHi);
            h.value(info.dfmt);
            h.value(info.nfmt);
            const vpgcn::TextureConv& c = info.conv;
            h.bytes(c.select, 4);
            h.value(c.affine);
            h.bytes(c.scale, sizeof c.scale);
            h.bytes(c.bias, sizeof c.bias);
            h.value(c.srgb);
            h.value(c.manual);
            h.value(c.texelsPerElement);
        }
        if (stage == vpgcn::ShaderStage::Pixel)
            for (const vpgcn::ExportConv& e : curExport) {
                h.value(e.kind);
                h.bytes(e.select, 4);
                h.value(e.srgb);
                h.bytes(e.bits, 4);
                h.value(e.dfmt);
                h.value(e.nfmt);
            }
        h.value(viewIndexedNow && stage != vpgcn::ShaderStage::Compute);
        h.value(alphaTestNow && stage == vpgcn::ShaderStage::Pixel);
        const uint64_t key2 = h.h;
        auto it2 = translated.find(key2);
        if (it2 != translated.end()) {
            if (it2->second.fromCache && !it2->second.touched && cache) {  // (its place in the cache's order of use)
                it2->second.touched = true;
                cache->touch(key2);
            }
            return &it2->second;
        }
        if (const Translated* merged = mergeNow(key2)) {
            if (cache) cache->touch(key2);
            translated[key2].touched = true;
            return merged;
        }

        vpgcn::LiftInput in = liftInput(s, stage, &curExport);
        in.viewIndexed = viewIndexedNow && stage != vpgcn::ShaderStage::Compute;
        in.alphaTest = alphaTestNow && stage == vpgcn::ShaderStage::Pixel;
        std::map<vpgcn::DescriptorSource, vpgcn::DescriptorInfo> infos;
        for (size_t pi = 0; pi < it1->second.probe.size(); ++pi) {
            const vpgcn::Binding& b = it1->second.bindings[it1->second.probe[pi]];
            vpgcn::DescriptorInfo info;
            const Snap& sn = snap[pi];
            if (sn.ok && info_of(b, sn.w.data(), info)) infos[b.source] = info;
        }
        in.describe = [&](const vpgcn::DescriptorSource& src) -> std::optional<vpgcn::DescriptorInfo> {
            auto f = infos.find(src);
            return f == infos.end() ? std::nullopt : std::optional(f->second);
        };
        vpgcn::LiftResult lifted = vpgcn::lift(in);
        Translated t;
        for (const std::string& n : lifted.notes) note("shader: " + n);
        if (!lifted.errors.empty()) {
            t.error = lifted.errors.front();
        } else {
            vp4msl::EmitResult msl = vp4msl::emitMsl(lifted.module);
            if (!msl.errors.empty()) {
                t.error = "msl: " + msl.errors.front();
            } else {
                Hash idh;
                idh.value(stage);
                idh.bytes(msl.shader.source.data(), msl.shader.source.size());
                t.id = idh.h ? idh.h : 1;
                t.ok = true;
                t.bindings = lifted.bindings;
                t.resources = lifted.module.resources;
                t.userDataSlot = lifted.userDataSlot;
                t.snapIndex = snapIndexFor(it1->second, lifted.bindings);
                if (registered.insert(t.id).second && r.hooks_.shader) {
                    r.hooks_.shader(t.id, irStage(stage), msl.shader);
                }
                if (cache)
                    cache->append(tcache::TranslationRec{key1, key2, t.id, uint8_t(stage), msl.shader, lifted.bindings,
                                                         lifted.module.resources, lifted.userDataSlot});
                ++r.stats_.shaders;
            }
        }
        if (!t.ok) {
            if (cache) cache->append(tcache::FailureRec{2, key2, uint8_t(stage), t.error});
            ++r.stats_.shaderErrors;
            log("vpps4.shader error stage=" + std::to_string(int(stage)) + " key=" + hex(key2) + " \"" + t.error + "\"");
            reportShaderError(stage, key2, t.error);
        }
        return &translated.emplace(key2, std::move(t)).first->second;
    }

    // ---- Images: render targets, depth targets and textures ---------------------------------------
    //
    // On the PS4 an image is guest memory read through a descriptor. Each GPU image here is a host
    // copy of guest range [address, address + bytes); the MemoryTracker says who holds the newest
    // bytes of every range (memory_tracker.h):
    //  * before an image is used it is brought up to date: when another writer (the CPU, a raw
    //    buffer write, DMA, another image) wrote its range since it last held it, it is refilled
    //    from memory on the GPU timeline, after images that own newer bytes were written back;
    //  * every pass or dispatch that writes an image marks its range when it is recorded. The bytes
    //    in guest memory are then stale until the image is re-tiled, which happens lazily: before a
    //    raw buffer access, a DMA, an eviction or a CPU read (syncToMemory).
    // The tiling seam (detileToImage / retileToMemory) is the only place that moves bytes between
    // memory and an image; the GpuTiler of the tiling area takes over its bodies.

    MemoryTracker tracker;

    struct Surface {
        uint32_t wid = 0;  // tracker writer id
        TextureId id;
        TextureId ms;
        vp4gpu::Format format = vp4gpu::Format::Undefined;
        bool writable = false;  // `id` has storage usage (a compute pass wrote it)
        vp4gpu::TextureType type = vp4gpu::TextureType::Tex2D;
        uint32_t width = 0, height = 0, depth = 1;
        uint16_t layers = 1;
        uint8_t samples = 1;
        uint8_t tile = 8;    // TILE_MODE_INDEX
        uint32_t pitch = 0;  // texels per row as allocated (0: width)
        bool isDepth = false;
        uint16_t mips = 1;       // levels in the guest layout (textures)
        bool pow2Pad = false;    // T# POW2_PAD: arrays padded to a power of two slices in memory
        uint8_t memSamples = 1;  // samples per texel as laid out in memory (MSAA textures)
        uint64_t address = 0, bytes = 0;  // guest range it is a copy of (mip 0, all slices)
        uint64_t version = 0;             // tracker version of the memory contents it holds
        uint64_t lastUse = 0;
        bool filled = false;         // filled from memory at least once (targets: always)
        bool sourceWatched = false;  // the host catches CPU writes to the source: no hashing
        bool ownedWatched = false;   // the host catches CPU access to memory this image is ahead of
    };
    struct Target : Surface {};

    // Counts the creations of targets and textures: what a T# resolves to (and so what its descriptor
    // info says, see info_of) can only change with them.
    uint64_t texGen = 1;
    std::map<uint64_t, std::vector<Target>> colorTargets, depthTargets;  // by guest address
    std::map<uint64_t, TextureId> latestColor;

    struct Texture : Surface {
        // CPU writes to the source are found by content when the host cannot watch it: a few
        // sampled lines every frame it is used, the whole source every kRecheck frames.
        uint64_t sample = 0, full = 0;
        uint64_t checkedFrame = ~0ull, fullFrame = 0;
        bool storage = false;  // created writable
    };
    struct RetiredTexture {
        TextureId id;
        uint64_t after;
    };
    std::vector<RetiredTexture> retiredTextures;  // replaced or evicted, destroyed kFrames later
    std::vector<RetiredTexture> retiredViews;
    static constexpr uint64_t kRecheck = 30;
    std::map<uint64_t, std::map<uint64_t, Texture>> textures;  // address -> shape hash -> texture

    // wid -> where the image lives (kind 0 colour target, 1 depth target, 2 texture).
    struct SurfaceRef {
        uint8_t kind;
        uint64_t address, shape;
    };
    std::unordered_map<uint32_t, SurfaceRef> surfaceRefs;
    Surface* surfaceOf(uint32_t wid) {
        auto it = surfaceRefs.find(wid);
        if (it == surfaceRefs.end()) return nullptr;
        const SurfaceRef& ref = it->second;
        if (ref.kind == 2) {
            auto a = textures.find(ref.address);
            if (a == textures.end()) return nullptr;
            auto t = a->second.find(ref.shape);
            return t == a->second.end() ? nullptr : &t->second;
        }
        auto& cache = ref.kind == 0 ? colorTargets : depthTargets;
        auto a = cache.find(ref.address);
        if (a == cache.end()) return nullptr;
        for (Target& t : a->second)
            if (t.wid == wid) return &t;
        return nullptr;
    }

    // Residency (see setResidency).
    uint64_t idleSubmits = 20000, byteBudget = 2ull << 30;  // (~20 s of play at ~900 submits/s)

    // Bytes of mip 0, all slices, of a surface in guest memory.
    static uint64_t surfaceBytes(uint8_t tile, vp4gpu::Format f, uint32_t w, uint32_t h, uint32_t pitch, uint32_t slices) {
        const vp4gpu::FormatInfo& fi = vp4gpu::formatInfo(f);
        const uint32_t bs = std::max<uint32_t>(fi.blockSize, 1), bpb = std::max<uint32_t>(fi.bytesPerBlock, 1);
        slices = std::max<uint32_t>(slices, 1);
        const uint32_t bw = (std::max(pitch, w) + bs - 1) / bs, bh = (h + bs - 1) / bs;
        if (!linearTiling(tile)) {
            TiledSurface s;
            s.tileIndex = tile;
            s.bytesPerElement = bpb;
            s.width = (w + bs - 1) / bs;
            s.height = bh;
            s.pitch = bw;
            s.slices = slices;
            std::string why;
            if (const uint64_t b = tiledBytes(s, why)) return b;
        }
        return uint64_t(bw) * bpb * bh * slices;  // (a layout not covered: the linear size as an estimate)
    }
    static uint64_t slicesOf(const Surface& s) { return s.type == vp4gpu::TextureType::Tex3D ? s.depth : s.layers; }
    static uint64_t gpuBytesOf(const Surface& s) {
        const vp4gpu::FormatInfo& fi = vp4gpu::formatInfo(s.format);
        const uint64_t bs = std::max<uint32_t>(fi.blockSize, 1);
        return (uint64_t(s.width) + bs - 1) / bs * ((s.height + bs - 1) / bs) * std::max<uint32_t>(fi.bytesPerBlock, 1) *
               slicesOf(s) * std::max<uint8_t>(s.samples, 1);
    }

    // Formats a texture might be seen through as another one of the same texel size.
    static bool sameTexel(vp4gpu::Format a, vp4gpu::Format b) {
        const vp4gpu::FormatInfo &x = vp4gpu::formatInfo(a), &y = vp4gpu::formatInfo(b);
        return x.bytesPerBlock == y.bytesPerBlock && x.blockSize == y.blockSize && x.integer() == y.integer() &&
               x.depth() == y.depth() && !x.depth();
    }

    // RGBA8 / BGRA8 differ only in channel order. A T# over an image of the other order is read as
    // the image's own format (the game's dst_sel swizzle, which is not applied, restores the order).
    static vp4gpu::Format swapOrder(vp4gpu::Format f) {
        using F = vp4gpu::Format;
        switch (f) {
            case F::RGBA8Unorm: return F::BGRA8Unorm;
            case F::BGRA8Unorm: return F::RGBA8Unorm;
            case F::RGBA8Srgb: return F::BGRA8Srgb;
            case F::BGRA8Srgb: return F::RGBA8Srgb;
            default: return f;
        }
    }
    static bool bgra(vp4gpu::Format f) { return f == vp4gpu::Format::BGRA8Unorm || f == vp4gpu::Format::BGRA8Srgb; }
    static bool rgba(vp4gpu::Format f) { return f == vp4gpu::Format::RGBA8Unorm || f == vp4gpu::Format::RGBA8Srgb; }
    static vp4gpu::Format inOrderOf(vp4gpu::Format f, vp4gpu::Format ref) {
        return (bgra(f) && rgba(ref)) || (rgba(f) && bgra(ref)) ? swapOrder(f) : f;
    }

    // ---- Tracker bookkeeping

    bool mirrored(uint64_t address, uint64_t bytes) {
        GuestRegion g;
        return bytes && region(address, bytes, g) && g.mirrored;
    }
    // Inside the range only: a page that also holds other data must not be protected for it.
    static bool inwardPages(uint64_t address, uint64_t bytes, uint64_t& a, uint64_t& n) {
        const uint64_t lo = (address + MemoryTracker::kPageBytes - 1) / MemoryTracker::kPageBytes * MemoryTracker::kPageBytes;
        const uint64_t hi = (address + bytes) / MemoryTracker::kPageBytes * MemoryTracker::kPageBytes;
        if (hi <= lo) return false;
        a = lo, n = hi - lo;
        return true;
    }
    void watchOwned(Surface& s, bool on) {
        // (an image that cannot be written back is not watched: the CPU would only be stopped for nothing)
        if (s.ownedWatched == on || !r.hooks_.watchOwned || (on && !canRetile(s))) return;
        uint64_t a, n;
        if (!inwardPages(s.address, s.bytes, a, n) || !mirrored(a, n)) return;
        s.ownedWatched = on;
        r.hooks_.watchOwned(a, n, on);
    }
    // The image's range after a recorded write: it holds the newest bytes, memory is behind it.
    void markWrite(Surface& s) {
        if (!s.bytes) return;
        s.version = tracker.markGpuWrite(s.address, s.bytes, s.wid);
        watchOwned(s, true);
    }
    void markWriteWid(uint32_t wid) {
        if (Surface* s = surfaceOf(wid)) markWrite(*s);
    }
    bool ownsStale(const Surface& s) {
        std::vector<MemoryTracker::Owner> o;
        tracker.staleOwners(s.address, s.bytes, o);
        for (const auto& x : o)
            if (x.writer == s.wid) return true;
        return false;
    }

    // ---- Tiling seam
    //
    // The only place that moves bytes between guest memory and an image: linear surfaces by buffer
    // <-> texture copies, tiled ones through the GpuTiler kernels (CPU detile while they compile).

    bool gpuTiling = true;
    std::unique_ptr<GpuTiler> tilerPtr;
    GpuTiler& tiler() {
        if (!tilerPtr)
            tilerPtr = std::make_unique<GpuTiler>(
                r.device_,
                [this](uint64_t id, const vp4msl::MslShader& shader) {
                    if (r.hooks_.shader) r.hooks_.shader(id, vp4ir::Stage::Compute, shader);
                },
                [this](uint64_t bytes, Scratch& out) {
                    BufferId id;
                    uint64_t at = 0;
                    uint8_t* memory = staging(bytes, id, at);
                    if (!memory) return false;
                    out = {id, at, memory};
                    return true;
                });
        return *tilerPtr;
    }

    // The guest layout of an image: mip chain (textures), arrays, samples as laid out in memory.
    static TiledSurface tiledOf(const Surface& s) {
        const vp4gpu::FormatInfo& fi = vp4gpu::formatInfo(s.format);
        TiledSurface t;
        t.tileIndex = s.tile;
        t.bytesPerElement = fi.bytesPerBlock;
        t.blockSize = std::max<uint32_t>(fi.blockSize, 1);
        t.width = s.width;
        t.height = s.height;
        t.pitch = std::max(s.pitch, s.width);
        t.volume = s.type == vp4gpu::TextureType::Tex3D;
        t.slices = t.volume ? s.depth : s.layers;
        t.pow2Pad = s.pow2Pad;
        if (t.pow2Pad && !t.volume) t.slices = std::bit_ceil(t.slices);
        t.levels = std::max<uint16_t>(s.mips, 1);
        t.samples = std::max<uint8_t>(s.memSamples, 1);
        return t;
    }
    static bool supportedFormat(const Surface& s) {
        if (s.isDepth) return s.format == vp4gpu::Format::D32Float || s.format == vp4gpu::Format::D16Unorm;
        return true;
    }
    static vp4gpu::Aspect aspectOf(const Surface& s) { return s.isDepth ? vp4gpu::Aspect::Depth : vp4gpu::Aspect::All; }

    // Whether the image can be written back to its guest range (re-tiled).
    bool canRetile(const Surface& s) const {
        if (!s.bytes || !s.id || s.memSamples != 1 || !supportedFormat(s)) return false;
        if (vp4gpu::formatInfo(s.format).blockSize > 1) return false;  // (block-compressed images are never written)
        if (linearTiling(s.tile)) return true;
        return gpuTiling && computeLayout(tiledOf(s)).ok;
    }

    // image -> guest memory, on the GPU timeline (GpuTiler::retile for tiled surfaces).
    bool retileToMemory(Surface& s) {
        if (!canRetile(s)) return false;
        const vp4gpu::FormatInfo& fi = vp4gpu::formatInfo(s.format);
        if (!linearTiling(s.tile)) {
            const SurfaceLayout lay = computeLayout(tiledOf(s));
            BufferId gb;
            uint64_t at = 0, available = 0;
            if (!guestBuffer(s.address, lay.bytes, gb, at, available) || available < lay.bytes || at % 4) {
                note("image newer than memory lies outside guest memory: not written back");
                return false;
            }
            Subresource sub;
            sub.levelCount = uint32_t(lay.levels.size());
            sub.sliceCount = s.type == vp4gpu::TextureType::Tex3D ? 0 : s.layers;
            sub.aspect = aspectOf(s);
            closePass();
            const GpuTiler::Result res = tiler().retile(frame, s.id, {gb, at, available}, lay, sub);
            if (res == GpuTiler::Result::NotReady) return false;  // (kernel compiling: the image stays ahead, tried again)
            if (res != GpuTiler::Result::Recorded) return note("GPU retile refused the request"), false;
            ++r.stats_.retiled;
            ++r.stats_.gpuRetiled;
            ++r.stats_.copies;
            return true;
        }
        const uint32_t blocksWide = (std::max(s.pitch, s.width) + fi.blockSize - 1) / fi.blockSize;
        const uint32_t blocksHigh = (s.height + fi.blockSize - 1) / fi.blockSize;
        const uint64_t rowBytes = uint64_t(blocksWide) * fi.bytesPerBlock;
        const uint64_t total = rowBytes * blocksHigh * slicesOf(s);
        BufferId buffer;
        uint64_t offset = 0, available = 0;
        if (!guestBuffer(s.address, total, buffer, offset, available) || total > available || offset % fi.bytesPerBlock) {
            note("image newer than memory lies outside guest memory: not written back");
            return false;
        }
        closePass();
        vp4gpu::Copy c;
        c.kind = vp4gpu::CopyKind::TextureToBuffer;
        c.srcTexture.texture = s.id;
        c.srcTexture.layerCount = uint16_t(s.type == vp4gpu::TextureType::Tex3D ? 1 : slicesOf(s));
        c.srcTexture.width = s.width;
        c.srcTexture.height = s.height;
        c.srcTexture.depth = s.type == vp4gpu::TextureType::Tex3D ? s.depth : 1;
        c.srcTexture.aspect = aspectOf(s);
        c.dstBuffer = {buffer, offset, total, uint32_t(rowBytes), blocksHigh};
        c.label = "retile " + hex(s.address);
        frame.copy(c);
        ++r.stats_.copies;
        ++r.stats_.retiled;
        return true;
    }

    // guest memory -> image, on the GPU timeline (every mip level of every layer). False (noted) if
    // the layout or format is not covered; the image is then left as it was.
    bool detileToImage(Surface& s) {
        if (s.samples > 1) return note("multisampled image not filled from memory"), false;
        if (!supportedFormat(s)) return note("depth image of this format not filled from memory"), false;
        const SurfaceLayout lay = computeLayout(tiledOf(s));
        if (!lay.ok) return note((linearTiling(s.tile) ? "linear texture not uploaded: " : "tiled texture not detiled: ") + lay.why), false;
        const vp4gpu::FormatInfo& fi = vp4gpu::formatInfo(s.format);
        const bool volume = s.type == vp4gpu::TextureType::Tex3D;
        const uint32_t layers = volume ? 1 : s.layers;
        const uint32_t levels = uint32_t(lay.levels.size());
        if (linearTiling(s.tile)) {
            // Every level straight from guest memory, rows `pitch` texels apart.
            BufferId buffer;
            uint64_t offset = 0, available = 0;
            if (!guestBuffer(s.address, lay.bytes, buffer, offset, available) || lay.bytes > available || offset % fi.bytesPerBlock)
                return note("linear texture outside guest memory or misaligned: left unfilled"), false;
            closePass();
            for (uint32_t li = 0; li < levels; ++li) {
                const LevelLayout& l = lay.levels[li];
                vp4gpu::Copy c;
                c.kind = vp4gpu::CopyKind::BufferToTexture;
                c.srcBuffer = {buffer, offset + l.offset, l.bytes, uint32_t(uint64_t(l.pitch) * fi.bytesPerBlock), l.paddedHeight};
                c.dstTexture.texture = s.id;
                c.dstTexture.mip = uint16_t(li);
                c.dstTexture.layerCount = uint16_t(layers);
                c.dstTexture.width = std::max(s.width >> li, 1u);
                c.dstTexture.height = std::max(s.height >> li, 1u);
                c.dstTexture.depth = volume ? std::max(s.depth >> li, 1u) : 1;
                c.dstTexture.aspect = aspectOf(s);
                c.label = "upload " + hex(s.address) + (li ? " mip " + std::to_string(li) : "");
                frame.copy(c);
                ++r.stats_.copies;
            }
            return true;
        }
        GuestRegion g;
        if (!region(s.address, lay.bytes, g)) return note("tiled texture outside guest memory: left unfilled"), false;
        if (gpuTiling) {
            BufferId gb;
            uint64_t at = 0, available = 0;
            if (guestBuffer(s.address, lay.bytes, gb, at, available) && available >= lay.bytes && at % 4 == 0) {
                Subresource sub;
                sub.levelCount = levels;
                sub.sliceCount = volume ? 0 : s.layers;  // (pow2 padded arrays hold more slices than the texture)
                sub.aspect = aspectOf(s);
                closePass();
                const GpuTiler::Result res = tiler().detile(frame, {gb, at, available}, s.id, lay, sub);
                if (res == GpuTiler::Result::Recorded) {
                    r.stats_.copies += levels;
                    ++r.stats_.gpuDetiled;
                    ++r.stats_.detiled;
                    return true;
                }
                if (res == GpuTiler::Result::Unsupported) note("GPU detile refused the request: detiled on the CPU");
            }
        }
        // CPU: the kernels are still compiling (or refused): detile into a staging buffer.
        const uint8_t* src = static_cast<const uint8_t*>(g.host) + (s.address - g.base);
        BufferId buffer;
        uint64_t offset = 0;
        uint8_t* dst = staging(lay.linearBytes, buffer, offset);
        if (!dst) return note("no staging memory for a tiled texture: left unfilled"), false;
        if (!detileLevels(lay, 0, levels, 0, src, dst)) return note("tiled texture not detiled: " + lay.why), false;
        if (recordedMayWrite(s.address, lay.bytes)) note("CPU detile of memory that recorded GPU work may still write");
        closePass();
        for (uint32_t li = 0; li < levels; ++li) {
            const LevelLayout& l = lay.levels[li];
            vp4gpu::Copy c;
            c.kind = vp4gpu::CopyKind::BufferToTexture;
            const uint32_t slices = volume ? l.slices : s.layers;
            c.srcBuffer = {buffer, offset + l.linearOffset, uint64_t(l.width) * l.height * slices * fi.bytesPerBlock,
                           uint32_t(uint64_t(l.width) * fi.bytesPerBlock), l.height};
            c.dstTexture.texture = s.id;
            c.dstTexture.mip = uint16_t(li);
            c.dstTexture.layerCount = uint16_t(layers);
            c.dstTexture.width = std::max(s.width >> li, 1u);
            c.dstTexture.height = std::max(s.height >> li, 1u);
            c.dstTexture.depth = volume ? std::max(s.depth >> li, 1u) : 1;
            c.dstTexture.aspect = aspectOf(s);
            c.label = "detile " + hex(s.address) + (li ? " mip " + std::to_string(li) : "");
            frame.copy(c);
            ++r.stats_.copies;
        }
        ++r.stats_.detiled;
        return true;
    }
    // (the busy ranges also count reads: only a hint)
    bool recordedMayWrite(uint64_t a, uint64_t n) const { return busyOverlaps(a, n); }

    // ---- Coherence operations

    // Writes back (re-tiles) the images that hold newer bytes than memory in [address, +size),
    // oldest write first. `write`: the caller overwrites the range, so an image it covers completely
    // is not written back (the write takes its bytes over). True if work was recorded.
    bool syncOwners(uint64_t address, uint64_t size, bool write, uint32_t except = 0) {
        if (!size || tracker.ownerImage(address, size) == 0) return false;
        std::vector<MemoryTracker::Owner> owners;
        tracker.staleOwners(address, size, owners, except);
        bool any = false;
        for (const MemoryTracker::Owner& o : owners) {
            Surface* s = surfaceOf(o.writer);
            if (!s) {
                tracker.release(o.writer);
                continue;
            }
            if (write && address <= s->address && address + size >= s->address + s->bytes) continue;
            if (!retileToMemory(*s)) {
                note("image newer than memory cannot be written back (layout not covered): raw readers see stale bytes");
                continue;
            }
            tracker.markSynced(s->wid);
            watchOwned(*s, false);
            any = true;
        }
        return any;
    }
    // A raw (buffer / DMA) access of guest memory: images ahead of memory there are written back
    // first; a write then takes the range over.
    void rawAccess(uint64_t address, uint64_t size, bool write) {
        if (!size) return;
        if (size > kMaxRawExtent) size = kMaxRawExtent;  // (a descriptor claiming the whole address space)
        syncOwners(address, size, write);
        if (write) tracker.markGpuWrite(address, size, MemoryTracker::kGpuRaw);
    }
    // A binding that is only an address has no size: a small window is tracked (syncing a whole
    // render target for every table pointer that happens to lie below it would cost far more than the
    // stale read it prevents).
    static constexpr uint64_t kUnknownExtent = 4096;
    static constexpr uint64_t kMaxRawExtent = 256ull << 20;
    // The buffer of a raw access (indirect arguments, indices): the range read is tracked.
    bool rawBuffer(uint64_t address, uint64_t size, bool write, BufferId& id, uint64_t& offset, uint64_t& available) {
        if (!guestBuffer(address, size, id, offset, available)) return false;
        rawAccess(address, size && size < available ? size : std::min<uint64_t>(available, kUnknownExtent), write);
        return true;
    }

    // Fills the image from guest memory on the GPU timeline, after images that hold newer bytes in
    // its range were written back. Always leaves it marked up to date (a layout that cannot be
    // filled is noted once, not retried every use).
    void fillFromMemory(Surface& s, Texture* tex) {
        syncOwners(s.address, s.bytes, false, s.wid);
        const bool ok = detileToImage(s);
        s.version = tracker.current();
        s.filled = true;
        if (!ok) return;
        ++r.stats_.refilled;
        if (tex) stamp(*tex);
        if (r.hooks_.uploaded) r.hooks_.uploaded(s.address, s.bytes);
        if (r.hooks_.watchWrites && mirrored(s.address, s.bytes)) {
            r.hooks_.watchWrites(s.address, s.bytes);
            s.sourceWatched = true;
        }
    }
    // Before a use: has another writer changed the image's range since it last held it?
    void refresh(Surface& s, Texture* tex) {
        if (!s.bytes) return;
        const bool first = !s.filled;
        if (!first && tex) check(*tex);
        if (!first && !tracker.writtenByOthers(s.address, s.bytes, s.wid, s.version)) return;
        if (first && !tex) {  // a target: starts from memory only if somebody wrote there
            s.filled = true;
            if (!tracker.writtenByOthers(s.address, s.bytes, s.wid, 0)) return;
        }
        // Newer bytes of this image that cannot be written back would be lost by a refill (memory
        // behind them is stale): keep the GPU copy.
        if (!first && ownsStale(s) && !canRetile(s)) {
            note("image newer than memory written over by another writer: GPU copy kept");
            s.version = tracker.current();
            return;
        }
        fillFromMemory(s, tex);
    }

    static uint64_t mix(uint64_t h, uint64_t w) {
        h ^= w * 0x9E3779B97F4A7C15ull;
        h = (h << 27) | (h >> 37);
        return h * 0xBF58476D1CE4E5B9ull;
    }
    static uint64_t hashRange(const uint8_t* p, uint64_t n) {
        uint64_t h = n, w;
        uint64_t i = 0;
        for (; i + 32 <= n; i += 32) {
            uint64_t a, b, c, d;
            std::memcpy(&a, p + i, 8), std::memcpy(&b, p + i + 8, 8), std::memcpy(&c, p + i + 16, 8),
                std::memcpy(&d, p + i + 24, 8);
            h = mix(h, a ^ (b << 1) ^ (c << 2) ^ (d << 3) ^ (b >> 63) ^ (c >> 62) ^ (d >> 61));
        }
        for (; i + 8 <= n; i += 8) std::memcpy(&w, p + i, 8), h = mix(h, w);
        for (; i < n; ++i) h = mix(h, p[i]);
        return h;
    }
    // 16 lines of 64 bytes spread over the source.
    static uint64_t sampleRange(const uint8_t* p, uint64_t n) {
        if (n <= 1024) return hashRange(p, n);
        uint64_t h = n;
        for (uint64_t k = 0; k < 16; ++k) h = mix(h, hashRange(p + (n - 64) * k / 15, 64));
        return h;
    }
    const uint8_t* guestBytes(uint64_t address, uint64_t bytes) {
        GuestRegion g;
        if (!bytes || !region(address, bytes, g)) return nullptr;
        return static_cast<const uint8_t*>(g.host) + (address - g.base);
    }
    // After a fill: remember what the source held.
    void stamp(Texture& t) {
        if (const uint8_t* p = guestBytes(t.address, t.bytes)) {
            t.sample = sampleRange(p, t.bytes);
            t.full = hashRange(p, t.bytes);
        }
        t.checkedFrame = t.fullFrame = r.frame_;
    }
    // CPU writes the host cannot watch are found by content (a mirrored source is watched by page
    // protection instead): a change is recorded as a CPU write, which every image over the range
    // then sees as a newer version. Not for images ahead of memory (their source is stale by design).
    void check(Texture& t) {
        if (!t.filled || !t.bytes || t.sourceWatched || t.checkedFrame == r.frame_) return;
        t.checkedFrame = r.frame_;
        if (tracker.writerAt(t.address) == t.wid) return;
        const uint8_t* p = guestBytes(t.address, t.bytes);
        if (!p) return;
        if (sampleRange(p, t.bytes) != t.sample) {
            tracker.markCpuWrite(t.address, t.bytes);
            return;
        }
        if (r.frame_ - t.fullFrame < kRecheck) return;
        // Full hashes are spread out: at most kFullHashBytes per frame (one frame is one Liverpool
        // submit, many per presented picture; hundreds of textures hashed whole at once cost tens of
        // milliseconds). A texture over the budget waits for a later frame, the first is always done.
        if (fullHashFrame != r.frame_) fullHashFrame = r.frame_, fullHashBytes = 0;
        if (fullHashBytes && fullHashBytes + t.bytes > kFullHashBytes) return;
        fullHashBytes += t.bytes;
        t.fullFrame = r.frame_;
        if (hashRange(p, t.bytes) != t.full) tracker.markCpuWrite(t.address, t.bytes);
    }
    static constexpr uint64_t kFullHashBytes = 4ull << 20;
    uint64_t fullHashFrame = ~uint64_t(0), fullHashBytes = 0;

    LifetimeTracker life;
    LifetimeStats lifeTotal;
    // The sample layout of the most recent multisampled pass (what an FMASK read describes): coverage
    // samples the title asked for and fragments the host stores. Before any: the app's limit.
    uint8_t msCoverage = 4, msFragments = 4;
    // The draw being recorded renders both eyes through a rate map: its shaders export/read the view.
    bool viewIndexedNow = false;
    bool alphaTestNow = false;  // DB_ALPHA_TO_MASK on a one-sample pass: the pixel shader tests alpha (lift)
    std::map<uint64_t, std::shared_ptr<const vp4gpu::RateMap>> rateMaps;
    const std::shared_ptr<const vp4gpu::RateMap>& rateMap(uint32_t w, uint32_t h) {
        auto& slot = rateMaps[uint64_t(w) << 32 | h];
        if (!slot) slot = makeFoveationMap(w, h, 2);
        return slot;
    }

    // The texture a pass draws into and the one it resolves into (null: no resolve).
    struct TargetIds {
        TextureId render, resolved;
        bool multisampled = false;
        uint32_t wid = 0;  // tracker writer id of the surface
    };

    vp4gpu::TextureDesc targetDesc(vp4gpu::Format f, uint32_t w, uint32_t h, uint8_t samples, uint16_t layers, bool depth,
                                   bool storage, uint64_t address, const char* what) {
        vp4gpu::TextureDesc d;
        d.format = f;
        d.width = w;
        d.height = h;
        d.samples = samples;
        d.layers = layers;
        d.usage = vp4gpu::usage::sampled | (depth ? vp4gpu::usage::depthTarget : vp4gpu::usage::colorTarget);
        // No storage usage unless a compute pass writes the target: ShaderWrite turns off Apple's lossless
        // framebuffer compression for the texture. A target written by compute is recreated writable on its
        // first such use (makeWritable).
        if (storage && !depth && samples == 1) d.usage |= vp4gpu::usage::storage;
        if (!depth && samples == 1 && r.hooks_.view && !vp4gpu::formatInfo(f).depth() && vp4gpu::formatInfo(f).blockSize == 1)
            d.usage |= vp4gpu::usage::formatView;  // other formats of its texel size may be drawn / sampled through views
        // Colour surfaces that are shown or sampled are kept in memory by the lifetime planner (it never
        // drops what was read); the rest may stay on tile.
        d.persistent = false;
        d.label = std::string(depth ? "depth " : "color ") + hex(address) + what;
        return d;
    }

    TargetIds idsOf(const Target& t) const {
        return {t.ms ? t.ms : t.id, t.ms ? t.id : TextureId{}, bool(t.ms), t.wid};
    }

    // The single-sample picture of a multisampled depth target, made on first use (a shader samples the
    // depth as a 1x texture). The passes of this frame that already drew into it resolve into it too.
    TextureId depthCompanion(Target& t) {
        if (t.id || !t.ms) return t.id;
        t.id = r.device_.createTexture(targetDesc(t.format, t.width, t.height, 1, t.layers, true, false, t.address, " resolved"));
        if (!t.id) return {};
        life.noteSticky(t.id, r.frame_);
        closePass();
        for (auto c = frame.mutableCommands().rbegin(); c != frame.mutableCommands().rend(); ++c)
            if (auto* rp = std::get_if<vp4gpu::RenderPass>(&*c))
                if (rp->desc.depth.texture == t.ms) {
                    rp->desc.depth.resolve = t.id;
                    rp->desc.depth.store = vp4gpu::StoreAction::StoreAndResolve;
                    break;
                }
        return t.id;
    }

    // ---- Sampling render targets

    // Whether a shader reading this T# sees the multisampled surface (declared texture2d_ms, loads with a
    // sample index) rather than the single-sample picture.
    bool readsSamples(const ImageDesc& d) {
        if (!msaaImage(d)) return false;
        const Bound b = boundTarget(d);
        return b.target && b.target->ms;
    }

    static bool msaaImage(const ImageDesc& d) { return d.type == 14 || d.type == 15; }

    // A compute pass writes this colour target: recreate its single-sample surface with storage usage (the
    // old one stays alive for the passes of this frame already recorded; its contents are copied over).
    void makeWritable(Target& c) {
        const TextureId old = c.id;
        const TextureId rw = r.device_.createTexture(targetDesc(c.format, c.width, c.height, 1, c.layers, false, true, c.address, " rw"));
        if (!rw) {
            note("render target could not be made writable");
            return;
        }
        closePass();
        vp4gpu::Copy cp;
        cp.kind = vp4gpu::CopyKind::TextureToTexture;
        cp.srcTexture.texture = old;
        cp.dstTexture.texture = rw;
        cp.srcTexture.layerCount = cp.dstTexture.layerCount = c.layers;
        cp.srcTexture.width = cp.dstTexture.width = c.width;
        cp.srcTexture.height = cp.dstTexture.height = c.height;
        cp.label = "writable " + hex(c.address);
        frame.copy(cp);
        ++r.stats_.copies;
        retireViews(old);
        retiredTextures.push_back({old, r.frame_ + kFrames});
        for (auto& [address, id] : latestColor)
            if (id == old) id = rw;
        life.noteSticky(rw, r.frame_);
        c.id = rw;
        c.writable = true;
        note("colour target recreated with storage usage (written by a compute pass)");
    }

    // ---- Clears that wait for the pass that draws ---------------------------------------------------
    // A fast clear (CMASK) or a depth/stencil clear rectangle does not get a render pass of its own: that
    // would store the surface and load it again in the next pass. The clear waits and becomes the load
    // action of the next pass that draws into the target; anything else that touches the target first
    // (a sample, a copy, a resolve) or the end of the frame flushes it as a pass of its own.
    struct PendingClear {
        TextureId texture;            // the attachment texture (ms when multisampled)
        TextureId resolved;           // its single-sample companion (null for depth or 1x)
        bool depth = false;
        vp4gpu::ClearValue value;
    };
    std::vector<PendingClear> pendingClears;

    void setPendingClear(PendingClear c) {
        for (PendingClear& p : pendingClears)
            if (p.texture == c.texture) return void(p = c);
        pendingClears.push_back(c);
    }
    const PendingClear* takePendingClear(TextureId t, PendingClear& out) {
        for (size_t i = 0; i < pendingClears.size(); ++i)
            if (pendingClears[i].texture == t) {
                out = pendingClears[i];
                pendingClears.erase(pendingClears.begin() + long(i));
                return &out;
            }
        return nullptr;
    }
    void emitClear(const PendingClear& p) {
        closePass();
        vp4gpu::RenderPassDesc desc;
        vp4gpu::Attachment& a = p.depth ? desc.depth : desc.colors[0];
        a.texture = p.texture;
        a.load = vp4gpu::LoadAction::Clear;
        a.clear = p.value;
        a.store = vp4gpu::StoreAction::Store;
        if (p.resolved) {
            a.resolve = p.resolved;
            a.store = vp4gpu::StoreAction::StoreAndResolve;
        }
        desc.label = "clear";
        frame.clear(desc);
        ++r.stats_.passes;
    }
    void flushClear(const Target& c) {
        if (pendingClears.empty()) return;
        for (size_t i = 0; i < pendingClears.size();)
            if (pendingClears[i].texture == c.ms || pendingClears[i].texture == c.id) {
                const PendingClear p = pendingClears[i];
                pendingClears.erase(pendingClears.begin() + long(i));
                emitClear(p);
            } else {
                ++i;
            }
    }
    void flushAllClears() {
        std::vector<PendingClear> all;
        all.swap(pendingClears);
        for (const PendingClear& p : all) emitClear(p);
    }

    // ---- Views

    struct ViewKey {
        uint32_t index, generation;
        uint8_t format, type;
        bool array;
        uint16_t baseLayer, layers;
        bool operator<(const ViewKey& o) const {
            return std::tie(index, generation, format, type, array, baseLayer, layers) <
                   std::tie(o.index, o.generation, o.format, o.type, o.array, o.baseLayer, o.layers);
        }
    };
    std::map<ViewKey, TextureId> views;
    // The view of `s` that `want` describes (null if the host has no views).
    TextureId viewOf(const Surface& s, ViewRequest want) {
        if (!r.hooks_.view) return {};
        want.base = s.id;
        const ViewKey key{s.id.index, s.id.generation, uint8_t(want.format), uint8_t(want.type), want.array, want.baseLayer, want.layers};
        auto it = views.find(key);
        if (it != views.end()) return it->second;
        const TextureId v = r.hooks_.view(want);
        if (v) ++r.stats_.viewsMade;
        views[key] = v;
        return v;
    }
    void retireViews(TextureId base) {
        for (auto it = views.begin(); it != views.end();) {
            if (it->first.index == base.index && it->first.generation == base.generation) {
                if (it->second) retiredViews.push_back({it->second, r.frame_ + kFrames});
                it = views.erase(it);
            } else {
                ++it;
            }
        }
    }
    // Retires `s`'s GPU texture (and its views): destroyed kFrames submits later.
    void retireTexture(const Surface& s) {
        if (s.ms) retiredTextures.push_back({s.ms, r.frame_ + kFrames});
        if (!s.id) return;
        retireViews(s.id);
        retiredTextures.push_back({s.id, r.frame_ + kFrames});
    }

    // What a draw needs to sample `s` through a T#: its own id, or a view of it (another format,
    // a layer range, a cube or array of an image made another way).
    struct TexRef {
        TextureId id, view;
    };
    TexRef sampleThrough(const Surface& s, const ImageDesc& d, vp4gpu::Format f) {
        f = inOrderOf(f, s.format);
        ViewRequest want;
        want.format = f;
        want.type = d.type == 8 ? vp4gpu::TextureType::Tex1D : d.type == 10 ? vp4gpu::TextureType::Tex3D
                  : d.type == 11 ? vp4gpu::TextureType::Cube : vp4gpu::TextureType::Tex2D;
        const bool arrayed = d.type == 12 || d.type == 13 || d.type == 15;
        if (want.type == vp4gpu::TextureType::Cube) {
            want.baseLayer = 0;
            want.layers = uint16_t(6 * (uint32_t(d.lastArray) / 6 + 1));
            want.array = want.layers > 6;
        } else {
            want.baseLayer = uint16_t(d.baseArray);
            want.layers = arrayed ? uint16_t(d.lastArray - d.baseArray + 1) : 1;
            want.array = arrayed;
        }
        if (want.type == vp4gpu::TextureType::Tex3D) want.baseLayer = 0, want.layers = 1, want.array = false;
        const bool sameLayers = want.type == vp4gpu::TextureType::Tex3D ? s.type == want.type
                              : want.baseLayer == 0 && want.layers == s.layers && s.type == want.type &&
                                want.array == (s.layers > 1 && want.type != vp4gpu::TextureType::Cube);
        if (f == s.format && sameLayers) return {s.id, {}};
        if (const TextureId v = viewOf(s, want)) return {s.id, v};
        if (f != s.format) note("image sampled with another format (no view available: its own format)");
        if (!sameLayers) note("image sampled as another slice range or dimension (no view available: whole image)");
        return {s.id, {}};
    }

    // ---- Targets

    // The surface rendered to at `address`. A draw to slice 1 alone and one to slices 0-1 must meet
    // in the same array (CB address is slice 0): the first with at least the layers asked for.
    // Another format of the same texel size at the same address is the same memory: it is drawn
    // through a view format (`viewAs`) of the existing target when the host has views.
    Target* target(std::map<uint64_t, std::vector<Target>>& cache, uint64_t address, vp4gpu::Format f, uint32_t w,
                   uint32_t h, uint8_t samples, uint16_t layers, bool depth, uint8_t tile, uint32_t pitch,
                   vp4gpu::Format* viewAs = nullptr) {
        layers = std::max<uint16_t>(layers, 1);
        if (depth)  // (a depth surface written at another base than it is read from answers for both)
            if (auto a = depthAliases.find(address); a != depthAliases.end())
                for (Target& t : cache[a->second])
                    if (t.format == f && t.width == w && t.height == h && t.samples == samples && t.tile == tile && t.layers >= layers) {
                        t.lastUse = r.frame_;
                        return &t;
                    }
        auto& list = cache[address];
        for (Target& t : list)
            if (t.format == f && t.width == w && t.height == h && t.samples == samples && t.tile == tile && t.layers >= layers) {
                t.lastUse = r.frame_;
                return &t;
            }
        if (!depth && viewAs && r.hooks_.view && samples == 1)
            for (Target& t : list)
                if (t.format != f && sameTexel(t.format, f) && t.width == w && t.height == h && t.samples == samples &&
                    t.tile == tile && t.layers >= layers) {
                    t.lastUse = r.frame_;
                    *viewAs = f;
                    return &t;
                }
        Target t;
        if (samples > 1) {
            t.ms = r.device_.createTexture(targetDesc(f, w, h, samples, layers, depth, false, address, " ms"));
            if (!t.ms) return nullptr;
            if (!depth) {
                t.id = r.device_.createTexture(targetDesc(f, w, h, 1, layers, false, false, address, ""));
                if (!t.id) return nullptr;
            }
        } else {
            t.id = r.device_.createTexture(targetDesc(f, w, h, 1, layers, depth, false, address, ""));
            if (!t.id) return nullptr;
        }
        t.wid = tracker.newImage();
        t.format = f;
        t.width = w;
        t.height = h;
        t.samples = samples;
        t.layers = layers;
        t.tile = tile;
        t.pitch = pitch;
        t.isDepth = depth;
        t.address = address;
        t.bytes = surfaceBytes(tile, f, w, h, pitch, layers);
        t.lastUse = r.frame_;
        ++r.stats_.targets;
        ++texGen;
        life.noteRead(t.id ? t.id : t.ms, r.frame_);  // (born now: stored while it warms up)
        surfaceRefs[t.wid] = {uint8_t(depth ? 1 : 0), address, 0};
        auto at = list.begin();
        while (at != list.end() && at->layers > layers) ++at;
        return &*list.insert(at, t);
    }

    // The depth surface read at `from` also answers for `to` (written at another address than it is
    // read from): later lookups at `to` find it.
    std::map<uint64_t, uint64_t> depthAliases;
    void aliasTarget(uint64_t to, uint64_t from) { depthAliases[to] = from; }

    // A draw with colour off that forces depth (or stencil) dirty and valid copies the surface it
    // reads into the one it writes (a depth copy / decompression done with a draw).
    static bool isDepthCopy(const DrawState& d) {
        for (const ColorTarget& c : d.color)
            if (c.format && c.writeMask) return false;
        const DepthTarget& z = d.depth;
        const bool depth = z.zFormat && z.forceZDirty && z.forceZValid && z.zWriteAddress && z.zWriteAddress != z.zAddress;
        const bool stencil = z.stencilFormat && z.forceStencilDirty && z.forceStencilValid && z.stencilWriteAddress &&
                             z.stencilWriteAddress != z.stencilAddress;
        return depth || stencil;
    }

    void depthCopy(const DrawState& d) {
        const DepthTarget& z = d.depth;
        const vp4gpu::Format zf = depthFormat(z.zFormat, z.stencilFormat);
        if (zf == vp4gpu::Format::Undefined) return skip("depth copy: no depth format");
        const bool depth = z.zFormat && z.forceZDirty && z.forceZValid && z.zWriteAddress && z.zWriteAddress != z.zAddress;
        const uint64_t from = depth ? z.zAddress : z.stencilAddress;
        const uint64_t to = depth ? z.zWriteAddress : z.stencilWriteAddress;
        const uint16_t layers = uint16_t(z.sliceStart + z.sliceCount);
        Target* src = target(depthTargets, from, zf, z.width, z.height, 1, layers, true, z.tileMode, 0);
        Target* dst = target(depthTargets, to, zf, z.width, z.height, 1, layers, true, z.tileMode, 0);
        if (!src || !dst) return skip("depth copy: target creation failed");
        if (src->wid == dst->wid) return;  // one texture already answers for both addresses
        if (!src->id || !dst->id) return skip("depth copy: multisampled depth without a single-sample picture");
        refresh(*src, nullptr);
        if (z.zFormat && z.stencilFormat && (depth != (z.stencilFormat && z.forceStencilDirty && z.forceStencilValid)))
            note("depth copy of one plane copies both");
        closePass();
        vp4gpu::Copy c;
        c.kind = vp4gpu::CopyKind::TextureToTexture;
        c.srcTexture.texture = src->id;
        c.srcTexture.layer = z.sliceStart;
        c.dstTexture.texture = dst->id;
        c.dstTexture.layer = z.sliceStart;
        c.srcTexture.layerCount = c.dstTexture.layerCount = std::max<uint16_t>(z.sliceCount, 1);
        c.srcTexture.width = c.dstTexture.width = z.width;
        c.srcTexture.height = c.dstTexture.height = z.height;
        c.label = "depth copy " + hex(from) + " -> " + hex(to);
        frame.copy(c);
        markWrite(*dst);
        ++r.stats_.copies;
        note("depth copy draw done as a texture copy");
    }

    // The colour target with slice `layer` (> 0) at `address`: an array is one surface, slice k
    // at base + k * slice bytes (stereo eyes are often the two slices of one target, and the
    // present reads each through its own T#). The most recently used match.
    const Target* targetSlice(uint64_t address, uint32_t w, uint32_t h, uint16_t& layer) const {
        const Target* best = nullptr;
        auto it = colorTargets.lower_bound(address > (256ull << 20) ? address - (256ull << 20) : 0);
        for (; it != colorTargets.end() && it->first < address; ++it)
            for (const Target& t : it->second) {
                if (t.layers < 2 || t.width < w || t.height < h) continue;
                const vp4gpu::FormatInfo& fi = vp4gpu::formatInfo(t.format);
                if (fi.blockSize > 1) continue;
                const uint64_t slice = uint64_t(t.width) * t.height * std::max<uint32_t>(fi.bytesPerBlock, 1);
                const uint64_t off = address - it->first;
                if (!slice || off % slice || off / slice >= t.layers) continue;
                if (!best || t.lastUse > best->lastUse) {
                    best = &t;
                    layer = uint16_t(off / slice);
                }
            }
        return best;
    }

    // The render target a T# names: same address, base level, a slice range inside the target's, a
    // format of the same texel size (sampled through a view) and a size the target covers (CB width
    // is the pitch). Anything else is another picture of the same memory: it goes through memory.
    Target* findTarget(std::map<uint64_t, std::vector<Target>>& cache, const ImageDesc& d, vp4gpu::Format f, bool depth,
                       bool anyFormat = false) {
        auto t = cache.find(d.address);
        if (&cache == &depthTargets)
            if (auto a = depthAliases.find(d.address); a != depthAliases.end() && (t == cache.end() || t->second.empty()))
                t = cache.find(a->second);
        if (t == cache.end() || d.baseLevel != 0) return nullptr;
        const bool arrayed = d.type == 12 || d.type == 13 || d.type == 15;
        const uint32_t lastSlice = d.type == 11 ? 6 * (uint32_t(d.lastArray) / 6 + 1) - 1 : arrayed ? d.lastArray : d.baseArray;
        Target* best = nullptr;
        for (Target& c : t->second) {
            if (c.width < d.width || c.height < d.height || lastSlice >= c.layers) continue;
            if (!depth && !anyFormat && c.format != f && !sameTexel(c.format, f)) continue;
            if (!best || (c.format == f && best->format != f)) best = &c;
        }
        return best;
    }

    // ---- Textures

    // The render target a T# names (any format of the same address), colour first.
    struct Bound {
        const Target* target = nullptr;
        bool depth = false;
    };
    Bound boundTarget(const ImageDesc& d) {
        const vp4gpu::Format f = textureFormat(d.dataFormat, d.numFormat);
        if (Target* c = findTarget(colorTargets, d, f, false, true); c && sameTexel(c->format, f)) return {c, false};
        if (Target* c = findTarget(depthTargets, d, f, true, true)) return {c, true};
        return {};
    }

    // How a T# is read: its format plan, adapted to the render target it names when there is one (a
    // target is read as it is stored: the plan's conversion is adjusted, no view is made).
    TexturePlan planFor(const ImageDesc& d, bool storage, bool& depthTarget) {
        TexturePlan p = texturePlan(d.dataFormat, d.numFormat, storage);
        depthTarget = false;
        const Bound b = boundTarget(d);
        if (!b.target) return p;
        const vp4gpu::Format tf = b.target->format;
        if (b.depth) {
            depthTarget = true;
            p = TexturePlan{};
            p.format = tf;
            return p;
        }
        if (tf == p.format) return p;
        const vp4gpu::FormatInfo& ft = vp4gpu::formatInfo(tf);
        if ((bgra(tf) && rgba(p.format)) || (rgba(tf) && bgra(p.format))) {
            // A BGRA target holds the same bytes as the RGBA texture format (and the reverse): red and
            // blue swap in the read; the sRGB-ness of the T# stays (a view of the target when it differs).
            p.format = swapOrder(p.format);
            if (!p.conv.manual)
                for (uint8_t& s : p.conv.select) s = s == 4 ? uint8_t(6) : s == 6 ? uint8_t(4) : s;
            return p;
        }
        if (ft.integer() && !p.conv.manual) {
            TexturePlan c = containerTexturePlan(d.dataFormat, d.numFormat);
            if (c.format == tf) return c;
        }
        return p;  // another format of the texel size: a view of the target (or its own format, noted); another size: through memory
    }

    TextureId texture(const ImageDesc& d, bool storage) {
        const TexRef t = textureRef(d, storage, storage);
        return t.view ? t.view : t.id;
    }

    // The texture a T# names: a render target at that address (matching format, slices and level),
    // else the texture of that shape, filled from guest memory (after images ahead of memory there
    // were written back) whenever another writer wrote its range. `write`: a shader writes it.
    TexRef textureRef(const ImageDesc& desc, bool storage, bool write) {
        bool depthTarget = false;
        const TexturePlan plan = planFor(desc, storage, depthTarget);
        const vp4gpu::Format f = plan.format;
        if (f == vp4gpu::Format::Undefined) {
            note("texture format " + std::to_string(desc.dataFormat) + "/" + std::to_string(desc.numFormat) + " unsupported" +
                 (plan.why ? std::string(" (") + plan.why + ")" : std::string()));
            return {};
        }
        if (Target* c = findTarget(colorTargets, desc, f, false)) {
            c->lastUse = r.frame_;
            flushClear(*c);
            if (!storage && msaaImage(desc) && c->ms) {  // the shader reads the samples themselves
                life.noteSamplesRead(c->ms, r.frame_);
                return {c->ms, {}};
            }
            if (storage && !c->writable) makeWritable(*c);
            refresh(*c, nullptr);
            if (write) markWrite(*c);
            life.noteRead(c->id, r.frame_);
            return sampleThrough(*c, desc, f);
        }
        if (Target* c = findTarget(depthTargets, desc, f, true)) {
            c->lastUse = r.frame_;
            flushClear(*c);
            if (!storage && msaaImage(desc) && c->ms) {
                life.noteSamplesRead(c->ms, r.frame_);
                return {c->ms, {}};
            }
            const TextureId id = c->ms ? depthCompanion(*c) : c->id;
            if (!id) return {};
            refresh(*c, nullptr);
            if (write) markWrite(*c);
            life.noteRead(id, r.frame_);
            return {id, {}};
        }
        if (!plan.exact && plan.why) note(plan.why);
        ImageDesc d = desc;  // (the host texture of a subsampled / 1-bit format is narrower)
        if (plan.widthDivisor > 1) {
            d.width = (d.width + plan.widthDivisor - 1) / plan.widthDivisor;
            d.pitch = (d.pitch + plan.widthDivisor - 1) / plan.widthDivisor;
        }
        if (d.type == 14 || d.type == 15) note("MSAA texture with no render target behind it read as single-sample");
        if (d.lastLevel > d.baseLevel) note("texture mip chain: level 0 only");

        vp4gpu::TextureDesc td;
        td.format = f;
        td.width = d.width;
        td.height = d.height;
        if (d.type == 12) note("1D array texture made a 2D array of height 1");
        td.type = d.type == 8                 ? vp4gpu::TextureType::Tex1D
                : d.type == 10               ? vp4gpu::TextureType::Tex3D
                : d.type == 11               ? vp4gpu::TextureType::Cube
                                             : vp4gpu::TextureType::Tex2D;
        if (td.type == vp4gpu::TextureType::Tex1D || d.type == 12) td.height = 1;
        if (td.type == vp4gpu::TextureType::Tex3D) td.depth = d.depth;
        const bool arrayed = d.type == 12 || d.type == 13 || d.type == 15;
        const uint32_t layers = arrayed ? uint32_t(d.lastArray) + 1 : 1;
        td.layers = uint16_t(td.type == vp4gpu::TextureType::Cube ? 6 * (uint32_t(d.lastArray) / 6 + 1) : layers);
        td.mips = textureMips(d, td);
        if (d.baseLevel) note("T# base level ignored: the texture holds the whole mip chain from its address");
        td.usage = vp4gpu::usage::sampled | (storage ? vp4gpu::usage::storage : 0);
        if (r.hooks_.view && vp4gpu::formatInfo(f).blockSize == 1 && !vp4gpu::formatInfo(f).depth())
            td.usage |= vp4gpu::usage::formatView;
        td.label = "tex " + hex(d.address);

        Hash h;
        h.value(f);
        h.value(td.type);
        h.value(td.width);
        h.value(td.height);
        h.value(td.depth);
        h.value(td.layers);
        // The subresource: a chain of N levels, or a view starting past the first, is not the same
        // object as one with other levels (mips used to collide in one level-0 texture).
        h.value(td.mips);
        h.value(d.baseLevel);
        h.value(d.lastLevel);
        h.value(d.baseArray);
        h.value(d.pow2Pad);
        // Usage is not part of the key: a storage write and a later sample of the same T# (e.g. a
        // compute pass producing the eye pictures that the present reads) must meet in one object.
        h.value(d.tilingIndex);
        h.value(d.pitch);
        auto& atAddress = textures[d.address];
        const uint64_t uncovered = surfaceBytes(d.tilingIndex, f, td.width, td.height, d.pitch,
                                                td.type == vp4gpu::TextureType::Tex3D ? td.depth : td.layers);
        const SurfaceLayout fullLayout = computeLayout(tiledSurfaceOf(d, td, vp4gpu::formatInfo(f)));
        const uint64_t layoutBytes = fullLayout.ok ? fullLayout.bytes : uncovered;
        // A shader wrote an image at this address through a T# of another format with the same
        // texel size (typically a compute pass writing UNORM storage that is then read as sRGB:
        // Metal has no sRGB storage). Memory never held that picture, and re-tiling it into another
        // layout would round-trip through a format conversion: this T# sees the writer's image
        // through a view of its format (the writer's own format without views).
        if (!storage) {
            auto own = atAddress.find(h.h);
            const uint64_t ownVersion = own == atAddress.end() || !own->second.filled ? 0 : own->second.version;
            for (auto& [key, other] : atAddress)
                if (key != h.h && other.id && other.format != f && other.filled &&
                    other.version > ownVersion &&
                    tracker.writerAt(d.address) == other.wid && other.type == td.type && other.width == td.width &&
                    other.height == td.height && other.depth == td.depth && other.layers == td.layers && sameTexel(other.format, f)) {
                    other.lastUse = r.frame_;
                    if (inOrderOf(f, other.format) == other.format) return {other.id, {}};  // (channel order only)
                    ViewRequest want;
                    want.format = inOrderOf(f, other.format);
                    want.type = td.type;
                    want.layers = td.layers;
                    want.array = td.layers > 1 && td.type != vp4gpu::TextureType::Cube;
                    if (const TextureId v = viewOf(other, want)) return {other.id, v};
                    note("GPU-written image sampled with another format of the same size (its own format)");
                    return {other.id, {}};
                }
        }
        auto found = atAddress.find(h.h);
        Texture& tex = atAddress[h.h];
        if (found == atAddress.end()) surfaceRefs[tex.wid = tracker.newImage()] = {2, d.address, h.h};
        if (tex.id && storage && !tex.storage) {
            // First write to a texture so far only sampled: recreate it writable (the old object
            // stays alive, earlier passes of this frame may still use it); it starts from memory.
            retireTexture(tex);
            tex.id = {};
            tex.filled = false;
            ++texGen;
        }
        if (!tex.id) {
            tex.id = r.device_.createTexture(td);
            if (!tex.id) return {};
            tex.storage = storage;
            tex.format = f;
            tex.type = td.type;
            tex.width = td.width;
            tex.height = td.height;
            tex.depth = td.depth;
            tex.layers = td.layers;
            tex.mips = td.mips;
            tex.pow2Pad = d.pow2Pad;
            tex.memSamples = d.type == 14 || d.type == 15 ? uint8_t(1u << std::min<uint32_t>(d.lastLevel, 3)) : 1;
            tex.tile = d.tilingIndex;
            tex.pitch = d.pitch;
            tex.address = d.address;
            tex.bytes = layoutBytes;
            tex.version = 0;
            tex.filled = false;
            ++r.stats_.textures;
            ++texGen;
        }
        tex.lastUse = r.frame_;
        refresh(tex, &tex);
        if (write) markWrite(tex);
        return {tex.id, {}};
    }

    // Mip levels a T# names (its chain starts at its address; MSAA and 1D textures have one), as
    // many as the sizes allow.
    static uint16_t textureMips(const ImageDesc& d, const vp4gpu::TextureDesc& td) {
        if (td.type == vp4gpu::TextureType::Tex1D || d.type == 12 || d.type == 14 || d.type == 15) return 1;
        const uint32_t largest = std::max({td.width, td.height, td.type == vp4gpu::TextureType::Tex3D ? td.depth : 1u});
        return uint16_t(std::min<uint32_t>(uint32_t(d.lastLevel) + 1, std::bit_width(largest)));
    }

    // The surface in guest memory that a T# and the texture made of it describe.
    static TiledSurface tiledSurfaceOf(const ImageDesc& d, const vp4gpu::TextureDesc& td, const vp4gpu::FormatInfo& fi) {
        TiledSurface s;
        s.tileIndex = d.tilingIndex;
        s.bytesPerElement = fi.bytesPerBlock;
        s.blockSize = std::max<uint32_t>(fi.blockSize, 1);
        s.width = td.width;
        s.height = td.height;
        s.pitch = std::max(d.pitch, td.width);
        s.volume = td.type == vp4gpu::TextureType::Tex3D;
        s.slices = s.volume ? td.depth : td.layers;
        s.pow2Pad = d.pow2Pad;
        if (s.pow2Pad && !s.volume) s.slices = std::bit_ceil(s.slices);
        s.levels = td.mips;
        s.samples = d.type == 14 || d.type == 15 ? 1u << std::min<uint32_t>(d.lastLevel, 3) : 1u;
        return s;
    }

    // ---- Residency

    // Destroys the image: written back first if it holds newer bytes than memory.
    void destroySurface(uint32_t wid) {
        Surface* s = surfaceOf(wid);
        if (!s) {
            surfaceRefs.erase(wid);
            return;
        }
        if (ownsStale(*s) && !(retileToMemory(*s) && (tracker.markSynced(wid), true)))
            note("image newer than memory evicted without a write back");
        watchOwned(*s, false);
        tracker.release(wid);
        retireTexture(*s);
        for (auto it = latestColor.begin(); it != latestColor.end();)
            it = it->second == s->id ? latestColor.erase(it) : std::next(it);
        const SurfaceRef ref = surfaceRefs[wid];
        surfaceRefs.erase(wid);
        ++r.stats_.evicted;
        if (ref.kind == 2) {
            auto a = textures.find(ref.address);
            a->second.erase(ref.shape);
            if (a->second.empty()) textures.erase(a);
        } else {
            auto& cache = ref.kind == 0 ? colorTargets : depthTargets;
            auto a = cache.find(ref.address);
            std::erase_if(a->second, [&](const Target& t) { return t.wid == wid; });
            if (a->second.empty()) cache.erase(a);
        }
    }
    // Idle images, then the least recently used ones past the byte budget.
    void evict() {
        struct Cand {
            uint32_t wid;
            uint64_t lastUse, bytes;
        };
        std::vector<Cand> all;
        uint64_t total = 0;
        auto add = [&](const Surface& s) {
            const uint64_t b = gpuBytesOf(s);
            total += b;
            all.push_back({s.wid, s.lastUse, b});
        };
        for (auto& [a, v] : colorTargets) for (Target& t : v) add(t);
        for (auto& [a, v] : depthTargets) for (Target& t : v) add(t);
        for (auto& [a, m] : textures) for (auto& [k, t] : m) add(t);
        std::vector<uint32_t> victims;
        std::vector<Cand> keep;
        for (const Cand& c : all) {
            if (c.lastUse >= r.frame_) keep.push_back(c);  // used since the last submit
            else if (r.frame_ - c.lastUse > idleSubmits) victims.push_back(c.wid), total -= c.bytes;
            else keep.push_back(c);
        }
        if (total > byteBudget) {
            std::sort(keep.begin(), keep.end(), [](const Cand& a, const Cand& b) { return a.lastUse < b.lastUse; });
            for (const Cand& c : keep) {
                if (total <= byteBudget || c.lastUse + 1 >= r.frame_) break;
                victims.push_back(c.wid);
                total -= c.bytes;
            }
        }
        for (uint32_t w : victims) destroySurface(w);
    }

    // Memory for a copy source: from the upload ring when it fits, else a buffer of its own that
    // lives until the GPU is done with this frame.
    struct Staging {
        void* memory;
        BufferId id;
        uint64_t after;
    };
    std::vector<Staging> stagings;
    uint8_t* staging(uint64_t bytes, BufferId& id, uint64_t& offset) {
        Ring& ring = rings[ringIndex];
        if (bytes <= kRingBytes / 2) {
            ensureRing();
            const uint64_t at = (ring.head + 255) / 256 * 256;
            if (ring.memory && at + bytes <= kRingBytes) {
                ring.head = at + bytes;
                id = ring.id;
                offset = at;
                return static_cast<uint8_t*>(ring.memory) + at;
            }
        }
        const uint64_t size = (bytes + kPage - 1) / kPage * kPage;
        void* memory = std::aligned_alloc(kPage, size);
        if (!memory) return nullptr;
        Staging st{memory, r.device_.createBuffer({size, memory, "vpps4 staging"}), r.frame_ + kFrames + 2};
        if (!st.id) {
            std::free(memory);
            return nullptr;
        }
        stagings.push_back(st);
        id = st.id;
        offset = 0;
        return static_cast<uint8_t*>(memory);
    }

    std::unordered_map<uint64_t, SamplerId> samplers;

    SamplerId sampler(std::span<const uint32_t> words) {
        Hash h;
        h.bytes(words.data(), words.size_bytes());
        SamplerId& id = samplers[h.h];
        if (!id) id = r.device_.createSampler(decodeSampler(words));
        return id;
    }

    // ---- Geometry pipeline buffers ----------------------------------------------------------------

    // What DescriptorSource::Ring ids (vpgcn::GeomRing) are bound to for the draw being built.
    struct RingBinding {
        BufferId id;
        uint64_t offset = 0, size = 0;
    };
    std::array<RingBinding, 8> ringBinds{};

    // GeomScratch buffers of the ES/GS rings: reused once the frames that used them are over.
    struct GeomScratch {
        void* memory;
        BufferId id;
        uint64_t size;
        uint64_t freeAt;  // reusable when the frame counter passes this
    };
    std::vector<GeomScratch> scratchPool;
    uint64_t scratchBytes = 0;
    static constexpr uint64_t kScratchLimit = 768ull << 20;
    bool scratch(uint64_t bytes, RingBinding& out) {
        bytes = std::max<uint64_t>((bytes + 255) / 256 * 256, 256);
        GeomScratch* best = nullptr;
        for (GeomScratch& x : scratchPool)
            if (x.freeAt < r.frame_ && x.size >= bytes && x.size <= 2 * bytes + (1 << 20) && (!best || x.size < best->size)) best = &x;
        if (!best) {
            if (scratchBytes + bytes > kScratchLimit) {
                // Free the unused ones and try again before giving up.
                for (size_t k = 0; k < scratchPool.size();)
                    if (scratchPool[k].freeAt < r.frame_) {
                        r.device_.destroy(scratchPool[k].id);
                        std::free(scratchPool[k].memory);
                        scratchBytes -= scratchPool[k].size;
                        scratchPool.erase(scratchPool.begin() + long(k));
                    } else {
                        ++k;
                    }
                if (scratchBytes + bytes > kScratchLimit) return false;
            }
            const uint64_t size = (bytes + kPage - 1) / kPage * kPage;
            void* memory = std::aligned_alloc(kPage, size);
            if (!memory) return false;
            GeomScratch x{memory, r.device_.createBuffer({size, memory, "vpps4 geometry scratch"}), size, 0};
            if (!x.id) {
                std::free(memory);
                return false;
            }
            scratchBytes += size;
            best = &scratchPool.emplace_back(x);
        }
        best->freeAt = r.frame_ + kFrames + 2;
        out = {best->id, 0, bytes};
        return true;
    }
    // Words to the upload ring as a ring binding.
    bool uploadRing(const void* data, uint64_t bytes, RingBinding& out) {
        auto at = upload(data, std::max<uint64_t>(bytes, 4), 256);
        if (!at) return false;
        out = {rings[ringIndex].id, *at, std::max<uint64_t>(bytes, 4)};
        return true;
    }

    // The GuestMap table of a stage whose shader uses FLAT instructions (vpgcn::DescriptorSource::GuestMap, layout at
    // vpgcn::kGuestMapEntries): the guest regions the GPU can reach with a raw pointer, each with the device
    // address of the gpu/core buffer that aliases it. A FLAT pointer is a guest GPU virtual address; the shader
    // finds its region in this table and addresses `device + offset` (kTargetDevice). Regions are those already
    // aliased this run (V#s and indirect data bind them) plus the ones the stage's user data point at
    // (64-bit values in a region): a pointer that leads elsewhere (a pointer stored in guest memory) is not found
    // until some other binding of the run maps its region, and reads zeros / drops writes until then.
    // Needs real device addresses (Device::bufferAddress): the null backend has fake ones, Vulkan needs
    // bufferDeviceAddress and Metal the residency set, otherwise the draw is skipped.
    bool guestMap(const Stage& s, vp4gpu::StageMask mask, std::vector<Access>& out, uint64_t& at, std::string& why) {
        for (unsigned k = 0; k + 1 < s.userData.size(); ++k) {
            const uint64_t va = s.userData[k] | uint64_t(s.userData[k + 1]) << 32;
            GuestRegion g;
            if (va < 0x10000 || !region(va, 1, g)) continue;
            BufferId id;
            uint64_t offset = 0, available = 0;
            guestBuffer(va, 1, id, offset, available);
        }
        struct Entry {
            uint64_t base, size, dev;
        };
        std::vector<Entry> entries;
        for (auto& [base, b] : guestBuffers) {
            if (!b.id) continue;
            const uint64_t dev = r.device_.bufferAddress(b.id);
            if (!dev) return why = "FLAT access needs buffer device addresses", false;
            entries.push_back({base, b.size, dev});
        }
        if (entries.size() > vpgcn::kGuestMapEntries) {
            note("more guest regions than the FLAT map holds: the rest read as zeros");
            entries.resize(vpgcn::kGuestMapEntries);
        }
        std::vector<uint32_t> table(vpgcn::kGuestMapBytes / 4, 0);
        table[0] = uint32_t(entries.size());
        for (size_t k = 0; k < entries.size(); ++k) {
            uint32_t* e = &table[4 + 8 * k];
            e[0] = uint32_t(entries[k].base);
            e[1] = uint32_t(entries[k].base >> 32);
            e[2] = uint32_t(entries[k].size);
            e[3] = uint32_t(entries[k].size >> 32);
            e[4] = uint32_t(entries[k].dev);
            e[5] = uint32_t(entries[k].dev >> 32);
        }
        auto where = upload(table.data(), vpgcn::kGuestMapBytes);
        if (!where) return why = "upload ring full", false;
        at = *where;
        // (the shader may write anywhere in them: ordered like any other GPU write to guest memory)
        for (auto& [base, b] : guestBuffers) {
            if (!b.id) continue;
            Access a;
            a.stages = mask;
            a.kind = vp4gpu::BindKind::Buffer;
            a.slot = 0xffff;  // (reachable by address only: no shader slot)
            a.write = true;
            a.range = ResourceRange::bytes(b.id, 0, b.size);
            touch(base, b.size);
            out.push_back(a);
        }
        return true;
    }

    // ---- Bindings ---------------------------------------------------------------------------------

    // Texture bindings that bind a view of their texture (collected by bind(), taken by the draw).
    std::vector<vp4gpu::ViewBinding> pendingViews;

    // The accesses of one translated stage, with the draw's real resources. False (with `why`) if a
    // resource cannot be resolved: the draw is skipped.
    bool bind(const Stage& s, const Translated& t, vpgcn::ShaderStage stage, vp4gpu::StageMask mask, std::vector<Access>& out,
              std::string& why) {
        const std::vector<Snap>& snap = snaps[size_t(stage)];
        uint32_t own[8];
        out.reserve(out.size() + t.bindings.size() + 1);
        for (size_t bi = 0; bi < t.bindings.size(); ++bi) {
            const vpgcn::Binding& b = t.bindings[bi];
            // The words shader() already read for this binding, or read now.
            const uint32_t* words = own;
            auto load = [&]() {
                const int at = bi < t.snapIndex.size() ? t.snapIndex[bi] : -1;
                if (at >= 0 && size_t(at) < snap.size()) {
                    words = snap[at].w.data();
                    return snap[at].ok;
                }
                words = own;
                return b.dwords <= 8 && readDescriptor(s, b.source, b.dwords, own);
            };
            const vp4ir::Resource& res = t.resources.at(b.slot);
            Access a;
            a.stages = mask;
            a.kind = bindKind(res.kind);
            a.slot = uint16_t(res.index);
            a.write = res.access != vp4ir::Access::Read &&
                      (res.kind == vp4ir::ResourceKind::Buffer || res.kind == vp4ir::ResourceKind::StorageTexture);
            switch (res.kind) {
                case vp4ir::ResourceKind::Buffer:
                case vp4ir::ResourceKind::ConstantBuffer: {
                    if (b.source.kind == vpgcn::DescriptorSource::Ring) {
                        const RingBinding& rb = ringBinds[b.source.sgpr % ringBinds.size()];
                        if (!rb.id) return why = "geometry ring unavailable", false;
                        a.range = ResourceRange::bytes(rb.id, rb.offset, rb.size);
                        break;
                    }
                    if (b.source.kind == vpgcn::DescriptorSource::GuestMap) {
                        uint64_t at = 0;
                        if (!guestMap(s, mask, out, at, why)) return false;
                        a.range = ResourceRange::bytes(rings[ringIndex].id, at, vpgcn::kGuestMapBytes);
                        break;
                    }
                    if (b.source.kind == vpgcn::DescriptorSource::Gds) {
                        if (!gdsBuffer()) return why = "no GDS buffer", false;
                        a.range = ResourceRange::bytes(gds.id, 0, kGdsBytes);
                        touch(reinterpret_cast<uint64_t>(gds.memory), kGdsBytes);  // the CPU reads GDS: see gdsAddress()
                        break;
                    }
                    uint64_t address = 0, size = 0;
                    if (b.dwords == 0 || b.source.kind == vpgcn::DescriptorSource::Code) {
                        // (Code: the base is an address in the program image; its size/format words are
                        // set by the program, so there is no descriptor to read.)
                        if (!memoryAddress(s, b.source, address)) return why = "buffer address unreadable", false;
                    } else {
                        if (!load()) return why = "V# unreadable", false;
                        const BufferDesc d = decodeBuffer({words, b.dwords});
                        address = d.address;
                        size = d.bytes();
                    }
                    BufferId id;
                    uint64_t offset = 0, available = 0;
                    if (!address || !guestBuffer(address, size, id, offset, available)) {
                        // The hardware reads zeros past a buffer's records and drops writes: an unset or
                        // unmapped V# gets a zero buffer of our own instead of losing the whole draw.
                        if (!dummyBuffer()) return why = "buffer outside guest memory", false;
                        note(address ? "buffer outside guest memory bound as zeros" : "null buffer bound as zeros");
                        std::memset(dummy.memory, 0, kDummyBytes);
                        a.range = ResourceRange::bytes(dummy.id, 0, kDummyBytes);
                        break;
                    }
                    // (a buffer of unknown extent, e.g. a bare address: only a window of it is tracked)
                    rawAccess(address, size && size <= available ? size : std::min<uint64_t>(available, kUnknownExtent), a.write);
                    if (!size || size > available) size = available;
                    if (res.kind == vp4ir::ResourceKind::Buffer && address % 4) return why = "buffer not 4-byte aligned", false;
                    if (res.kind == vp4ir::ResourceKind::ConstantBuffer && address % 16) {
                        // Constant buffers are read with aligned vector loads: an unaligned one is copied.
                        const uint64_t bytes = std::min<uint64_t>(size, 64 * 1024);
                        GuestRegion g;
                        region(address, bytes, g);
                        auto at = upload(static_cast<const uint8_t*>(g.host) + (address - g.base), bytes);
                        if (!at) return why = "upload ring full", false;
                        note("unaligned constant buffer copied (GPU writes to it in this frame are not seen)");
                        id = rings[ringIndex].id;
                        offset = *at;
                        size = bytes;
                    }
                    a.range = ResourceRange::bytes(id, offset, size);
                    break;
                }
                case vp4ir::ResourceKind::Texture:
                case vp4ir::ResourceKind::StorageTexture: {
                    if (!load()) return why = "T# unreadable", false;
                    const TexRef ref = textureRef(decodeImage({words, b.dwords}), res.kind == vp4ir::ResourceKind::StorageTexture, a.write);
                    if (!ref.id) return why = "texture unavailable", false;
                    a.range = ResourceRange::wholeTexture(ref.id);
                    if (ref.view) pendingViews.push_back({a.kind, a.slot, ref.view});
                    break;
                }
                case vp4ir::ResourceKind::Sampler: {
                    if (!load()) return why = "S# unreadable", false;
                    a.range = ResourceRange::sampler(sampler({words, b.dwords}));
                    break;
                }
                case vp4ir::ResourceKind::BufferSize:  // SPIR-V only (the GCN lifter never makes one)
                    return why = "buffer_size resource", false;
            }
            out.push_back(a);
        }
        if (t.userDataSlot) {
            auto at = uploadUserData(s.userData);
            if (!at) return why = "upload ring full", false;
            Access a;
            a.range = ResourceRange::bytes(rings[ringIndex].id, *at, sizeof s.userData);
            a.stages = mask;
            a.kind = vp4gpu::BindKind::ConstantBuffer;
            a.slot = uint16_t(t.resources.at(*t.userDataSlot).index);
            out.push_back(a);
        }
        return true;
    }

    // ---- Render passes ----------------------------------------------------------------------------

    vp4gpu::RenderPassDesc pass;
    bool passOpen = false;

    void closePass() {
        if (passOpen) frame.endRenderPass();
        passOpen = false;
    }

    void openPass(const vp4gpu::RenderPassDesc& want, bool clears) {
        if (passOpen && !clears && pass.sameTargets(want)) return;
        closePass();
        pass = want;
        frame.beginRenderPass(pass);
        passOpen = true;
        ++r.stats_.passes;
        // Later draws on the same targets continue it: no clear on reopen.
        for (vp4gpu::Attachment& c : pass.colors) c.load = vp4gpu::LoadAction::Load;
        pass.depth.load = vp4gpu::LoadAction::Load;
    }

    // ---- Draws ------------------------------------------------------------------------------------

    void draw(const DrawState& d) {
        pollCache();
        const RegionScope regions(*this);
        ++r.stats_.draws;
        if (d.count == 0 || d.instances == 0) return skip("empty draw");
        if (isDepthCopy(d)) return depthCopy(d);
        if (d.raster.cullFront && d.raster.cullBack) return skip("both faces culled");

        // Targets.
        vp4gpu::RenderPassDesc desc;
        vp4gpu::PipelineState ps;
        struct Drawn {
            uint64_t address;
            TextureId id;
            uint32_t wid;
        };
        std::vector<Drawn> drawnColors;
        uint32_t depthWid = 0;
        pendingViews.clear();
        uint32_t width = 0, height = 0;
        // One sample count for the whole pass (msaa.h): the smallest of the bound surfaces' counts and the
        // app's limit; EQAA depth with more samples than the colour stores keeps the colour's.
        std::array<uint8_t, 8> fragments{};
        bool dualSource = false;
        for (unsigned i = 0; i < 8; ++i)
            if (d.color[i].format && d.color[i].writeMask) fragments[i] = std::max<uint8_t>(d.color[i].samples, 1);
        bool depthExact = true;
        const vp4gpu::Format zf = depthFormat(d.depth.zFormat, d.depth.stencilFormat, &depthExact);
        const SamplePlan plan = planSamples(fragments, zf != vp4gpu::Format::Undefined ? std::max<uint8_t>(d.depth.samples, 1) : 0,
                                            r.maxSamples_);
        const uint8_t samples = plan.host;
        if (plan.reduced)
            note(samples >= 4 ? "MSAA reduced to 4 samples on the host"
                 : samples == 2 ? "MSAA reduced to 2 samples on the host"
                                : "MSAA reduced to 1 samples on the host");
        curExport = {};
        if (samples > 1) {
            uint8_t cov = samples;
            for (unsigned i = 0; i < 8; ++i)
                if (fragments[i]) cov = std::max<uint8_t>(cov, d.color[i].coverage ? d.color[i].coverage : fragments[i]);
            if (zf != vp4gpu::Format::Undefined) cov = std::max<uint8_t>(cov, d.depth.samples);
            msCoverage = cov;
            msFragments = samples;
        }
        if (plan.eqaa && samples > 1) note("EQAA: the depth buffer has more samples than the colour stores; the host draws with the colour's");
        std::vector<std::pair<unsigned, TargetIds>> colorIds;
        for (unsigned i = 0; i < 8; ++i) {
            const ColorTarget& c = d.color[i];
            if (!c.format || !c.writeMask) continue;
            const ColorPlan plan = colorPlan(c.format, c.numberType, c.compSwap);
            const vp4gpu::Format f = plan.format;
            if (f == vp4gpu::Format::Undefined) {
                note("colour target format " + std::to_string(c.format) + "/" + std::to_string(c.numberType) +
                     " unsupported: slot left out" + (plan.why ? std::string(" (") + plan.why + ")" : std::string()));
                continue;
            }
            if (!plan.exact && plan.why) note(plan.why);
            curExport[i] = plan.conv;
            if (!linearTiling(c.tileMode)) note("render targets kept in GPU memory (guest copy not written)");
            vp4gpu::Format viewAs = vp4gpu::Format::Undefined;
            Target* tg = target(colorTargets, c.address, f, c.width, c.height, samples,
                                uint16_t(c.sliceStart + c.sliceCount), false, c.tileMode, c.pitch, &viewAs);
            if (!tg) return skip("colour target creation failed");
            refresh(*tg, nullptr);  // another writer changed its memory since it last held it?
            const TargetIds t = idsOf(*tg);
            colorIds.push_back({i, t});
            drawnColors.push_back({c.address, t.resolved ? t.resolved : t.render, t.wid});
            desc.colors[i].texture = t.render;
            desc.colors[i].layer = c.sliceStart;
            desc.colors[i].format = viewAs;  // (drawn as another format of the same texel size)
            if (t.resolved) {
                // Resolved at the end of the pass; the lifetime planner decides what is kept (lifetime.h).
                desc.colors[i].resolve = t.resolved;
                desc.colors[i].resolveLayer = c.sliceStart;
                desc.colors[i].store = vp4gpu::StoreAction::StoreAndResolve;
            }
            ps.colorFormats[i] = f;
            const BlendResolved br = resolveBlend(d.blend[i], c, vp4gpu::formatInfo(f).integer());
            ps.blend[i] = br.state;
            if (br.state.enabled && !br.exact) note("blend factor with an undefined encoding: zero");
            if (i == 0 && br.dualSource) dualSource = true;
            if (i != 0 && br.dualSource) note("dual-source blend on a target other than 0 ignored");
            width = width ? std::min(width, c.width) : c.width;
            height = height ? std::min(height, c.height) : c.height;
        }
        bool clears = false;
        const DepthTarget& z = d.depth;
        std::shared_ptr<vp4gpu::ClearRects> partialClear;  // a clear of one plane, or of part of the target
        StencilResolved stencil;
        TargetIds depthIds;
        if (zf != vp4gpu::Format::Undefined) {
            if (!depthExact) note(z.zFormat == 2 ? "24-bit depth held as 32-bit float" : "16-bit depth with stencil held as 32-bit float");
            const uint64_t readAddress = z.zFormat ? z.zAddress : z.stencilAddress;
            const uint64_t writeAddress = z.zFormat ? z.zWriteAddress : z.stencilWriteAddress;
            Target* dt = target(depthTargets, readAddress, zf, z.width, z.height,
                                samples, uint16_t(z.sliceStart + z.sliceCount), true, z.tileMode, 0);
            if (!dt) return skip("depth target creation failed");
            if (!(z.depthClear || z.stencilClear)) refresh(*dt, nullptr);  // (a clear replaces the contents)
            depthWid = dt->wid;
            depthIds = idsOf(*dt);
            desc.depth.texture = depthIds.render;
            desc.depth.layer = z.sliceStart;
            if (depthIds.resolved) {
                desc.depth.resolve = depthIds.resolved;
                desc.depth.resolveLayer = z.sliceStart;
                desc.depth.store = vp4gpu::StoreAction::StoreAndResolve;
                desc.depth.depthResolve = depthResolveMode(z.depthCopy, z.copySample,
                                                          d.depthStencil.depthTest ? d.depthStencil.depthFunc : uint8_t(7));
            }
            ps.depthFormat = zf;
            width = width ? std::min(width, z.width) : z.width;
            height = height ? std::min(height, z.height) : z.height;
            const DepthStencil& ds = d.depthStencil;
            // The surface a draw writes may be another address than the one it reads: later reads
            // of the write address find this texture.
            if (writeAddress && writeAddress != readAddress && ds.depthWrite && !z.zReadOnly)
                aliasTarget(writeAddress, readAddress);
            ps.depthTest = ds.depthTest || ds.depthWrite;
            ps.depthWrite = ds.depthWrite && z.zFormat && !z.zReadOnly && !z.depthClear;
            ps.depthCompare = ds.depthTest ? compareOp(ds.depthFunc) : vp4gpu::CompareOp::Always;
            ps.stencilTest = ds.stencilTest && z.stencilFormat;
            if (ps.stencilTest) {
                stencil = resolveStencil(ds, z.stencilReadOnly || z.stencilClear);
                ps.stencilFront = stencil.front;
                ps.stencilBack = stencil.back;
                if (!stencil.exact) note("stencil operation approximated (step value or reference conflict)");
            }
            if (ds.depthBounds) note("depth bounds test ignored (no depth bounds on the host)");
            if (ds.colorWritesOnDepthFail || ds.noColorWritesOnDepthPass)
                note("colour writes conditioned on the depth result ignored");
        }
        if (!width || !height) return skip("no render targets");

        // Scissor: the registers' combination, clipped to the targets.
        Scissor sc = d.scissor;
        if (d.scissorRegs.present) sc = combineScissor(d.scissorRegs);
        const bool scissorSet = d.scissorRegs.present || sc.right > sc.left || sc.bottom > sc.top;
        int32_t sl = 0, stp = 0, sr = int32_t(width), sb = int32_t(height);
        if (scissorSet) {
            sl = std::max(0, sc.left), stp = std::max(0, sc.top);
            sr = std::min<int32_t>(sc.right, int32_t(width)), sb = std::min<int32_t>(sc.bottom, int32_t(height));
            if (sr <= sl || sb <= stp) return skip("empty scissor");
        }
        const bool scissorFull = sl == 0 && stp == 0 && sr == int32_t(width) && sb == int32_t(height);

        // A depth/stencil clear: the whole target in the pass's load action when every plane of the
        // format is cleared; otherwise (one plane, or less than the target) a clear inside the pass.
        if (zf != vp4gpu::Format::Undefined && (z.depthClear || z.stencilClear)) {
            const bool allPlanes = (z.depthClear || !z.zFormat) && (z.stencilClear || !z.stencilFormat);
            if (allPlanes && scissorFull) {
                desc.depth.load = vp4gpu::LoadAction::Clear;
                desc.depth.clear.depth = z.clearDepth;
                desc.depth.clear.stencil = z.clearStencil;
                clears = true;
            } else {
                partialClear = std::make_shared<vp4gpu::ClearRects>();
                vp4gpu::ClearRects::Rect rc;
                rc.x = uint32_t(sl), rc.y = uint32_t(stp), rc.width = uint32_t(sr - sl), rc.height = uint32_t(sb - stp);
                rc.layer = z.sliceStart, rc.layerCount = z.sliceCount;
                partialClear->rects.push_back(rc);
                partialClear->depth = z.depthClear && z.zFormat;
                partialClear->stencil = z.stencilClear && z.stencilFormat;
                partialClear->depthValue = z.clearDepth;
                partialClear->stencilValue = z.clearStencil;
            }
        }
        if (d.primitive == PrimitiveType::RectList && !d.geom.enabled) {
            // Mostly clears. A fast clear (CB_COLORn_INFO.FAST_CLEAR: CMASK only, the colour is in the clear
            // words) and a whole-target depth/stencil clear wait for the next pass that draws into the target
            // and become its load action (setPendingClear); a clear of one plane or of part of the target is a
            // clear inside a pass; other rectangles are not drawn yet.
            bool any = false;
            for (const auto& [i, t] : colorIds) {
                const ColorTarget& c = d.color[i];
                if (!c.fastClear) continue;
                bool exact = true;
                PendingClear pc;
                pc.texture = t.render;
                pc.resolved = t.resolved;
                pc.value.color = fastClearColor(c.format, c.numberType, c.compSwap, c.clearWord, exact);
                if (!exact) note("fast clear colour approximated (format not decoded)");
                closePass();
                setPendingClear(pc);
                markWriteWid(t.wid);
                any = true;
            }
            if (clears) {
                note("RectList depth/stencil clear done as a whole-target clear");
                PendingClear pc;
                pc.texture = depthIds.render;
                pc.resolved = depthIds.resolved;
                pc.depth = true;
                pc.value = desc.depth.clear;
                closePass();
                setPendingClear(pc);
                markWriteWid(depthWid);
                any = true;
            } else if (partialClear) {
                note("RectList depth/stencil clear done as a clear inside the pass");
                desc.colors = {};
                openPass(desc, false);
                vp4gpu::Draw cd;
                cd.clear = std::move(partialClear);
                frame.draw(std::move(cd));
                markWriteWid(depthWid);
                any = true;
            }
            // (nothing cleared: the rectangle is drawn by the vertex stage, below)
            if (any) return;
        }
        if (d.primitive == PrimitiveType::RectList && d.geom.enabled) {
            return skip("RectList with a geometry shader");
            return;
        }
        if (d.primitive == PrimitiveType::Patch) {
            note("tessellation patch draw (HS/DS stages) left out");
            return skip("tessellation patches");
        }
        // Layered rendering: the vertex stage that rasterises writes the render target slice
        // (PA_CL_VS_OUT_CNTL.USE_VTX_RENDER_TARGET_INDX); the pass renders to all the slices bound.
        if (d.vs.vsOutCntl & (1u << 18)) {
            uint16_t layers = 1;
            for (unsigned i = 0; i < 8; ++i)
                if (d.color[i].format && d.color[i].writeMask) layers = std::max<uint16_t>(layers, uint16_t(d.color[i].sliceCount));
            if (depthFormat(d.depth.zFormat, d.depth.stencilFormat) != vp4gpu::Format::Undefined)
                layers = std::max<uint16_t>(layers, uint16_t(d.depth.sliceCount));
            desc.layerCount = layers;
        }
        ps.samples = samples;
        // DB_ALPHA_TO_MASK: Metal's alpha-to-coverage with several samples; with one, the pixel shader does an
        // alpha test at 0.5 (LiftInput::alphaTest), which is what the coverage mask comes to.
        ps.alphaToCoverage = d.alphaToMask && d.color[0].format != 0 && samples > 1;
        alphaTestNow = d.alphaToMask && d.color[0].format != 0 && samples <= 1;

        // Foveation: a pass over two-slice targets renders both eyes through a rate map.
        viewIndexedNow = false;
        if (r.foveation_ && samples == 1 && !colorIds.empty()) {
            bool eligible = true;
            for (const auto& entry : colorIds) {
                const ColorTarget& c = d.color[entry.first];
                eligible = eligible && c.sliceStart == 0 && c.sliceCount == 2;
            }
            if (zf != vp4gpu::Format::Undefined) eligible = eligible && z.sliceStart == 0 && z.sliceCount == 2;
            if (eligible) {
                if (const auto& rm = rateMap(width, height)) {
                    desc.layerCount = 2;
                    desc.rateMap = rm;
                    viewIndexedNow = true;
                }
            }
        }
        const bool geom = d.geom.enabled;
        const bool rect = d.primitive == PrimitiveType::RectList && !geom;
        if (geom && (d.es.code.empty() || d.gs.code.empty())) return skip("geometry shader without ES or GS program");

        // Shaders.
        if (d.vs.code.empty()) return skip(geom ? "no copy shader" : "no vertex shader");
        auto tShader = Clock::now();
        Stage vsStage = d.vs, psStage = d.ps;
        vsStage.clipSpaceGL = !d.raster.clipSpaceZeroToW;
        psStage.dualSource = dualSource;
        const uint32_t inPrim = uint32_t(d.primitive);
        Stage vsCopy, esStage, gsStage;
        const Stage* vsSrc = &vsStage;
        if (geom || rect) {
            vsCopy = vsStage;
            vsCopy.rectList = rect;
            vsCopy.geom = d.geom;
            vsCopy.geomInputPrim = inPrim;
            vsSrc = &vsCopy;
        }
        if (geom) {
            const bool known = (inPrim >= 1 && inPrim <= 6) || (inPrim >= 10 && inPrim <= 13) || inPrim == 21;
            if (!known) {
                note("geometry shader with an unsupported input primitive type: draw left out");
                return skip("geometry shader input primitive");
            }
            if (d.indirectAddress) return skip("indirect draw with a geometry shader");
            esStage = d.es;
            gsStage = d.gs;
            esStage.geom = gsStage.geom = d.geom;
            esStage.geomInputPrim = gsStage.geomInputPrim = inPrim;
        }
        if (rect && d.indirectAddress) return skip("indirect RectList draw");
        const Translated* vs = shader(*vsSrc, geom ? vpgcn::ShaderStage::CopyVertex : vpgcn::ShaderStage::Vertex);
        if (!vs->ok) return skip(geom ? "copy shader not translated" : "vertex shader not translated");
        const Translated* esT = nullptr;
        const Translated* gsT = nullptr;
        if (geom) {
            esT = shader(esStage, vpgcn::ShaderStage::Export);
            if (!esT->ok) return skip("export shader not translated");
            gsT = shader(gsStage, vpgcn::ShaderStage::Geometry);
            if (!gsT->ok) return skip("geometry shader not translated");
        }
        const Translated* fs = nullptr;
        if (!d.ps.code.empty()) {
            fs = shader(psStage, vpgcn::ShaderStage::Pixel);
            if (!fs->ok) return skip("pixel shader not translated");
        }
        r.cpu_.shaderMs += msSince(tShader);
        ps.vertexShader = vs->id;
        ps.fragmentShader = fs ? fs->id : 0;

        // Geometry pipeline / RectList: the buffers its kernels work on.
        vp4gpu::Draw draw;
        std::string why;
        ringBinds = {};
        vp4gpu::Dispatch esJob, gsJob;
        uint64_t gsDrawIndices = 0;
        uint32_t rectCount = 0;
        if (geom) {
            const GeometryState& g = d.geom;
            const uint32_t entries = d.count;
            const uint32_t prims = vpgcn::inputPrimitives(inPrim, entries);
            const uint32_t gsInst = std::max(g.gsInstances, 1u);
            const uint32_t cap = vpgcn::outputIndexCapacity(g.outPrim, g.maxVertOut);
            if (!prims || !cap || !g.esgsItemDwords || !g.gsvsVertDwords) return skip("empty geometry shader draw");
            const uint64_t esCount = uint64_t(d.instances) * entries;
            const uint64_t gsCount = uint64_t(d.instances) * prims * gsInst;
            const uint64_t esWaves = (esCount + 63) / 64, gsWaves = (gsCount + 63) / 64;
            const uint64_t esBytes = esWaves * g.esgsItemDwords * 256;
            const uint64_t gsvsBytes = gsWaves * g.gsvsVertDwords * g.maxVertOut * 256;
            const uint64_t idxBytes = 4ull * cap * gsCount;
            if (esBytes > (256ull << 20) || gsvsBytes > (256ull << 20) || idxBytes > (256ull << 20) || cap * gsCount > 0x7fffffffull)
                return skip("geometry shader draw too large");
            using vpgcn::GeomRing;
            auto ring = [&](GeomRing id) -> RingBinding& { return ringBinds[size_t(id)]; };
            if (!scratch(esBytes, ring(GeomRing::EsGs)) || !scratch(gsvsBytes, ring(GeomRing::GsVs)) ||
                !scratch(idxBytes, ring(GeomRing::GsIndex)))
                return skip("geometry scratch memory exhausted");
            std::array<uint32_t, vpgcn::geomparam::Words> params{};
            params[vpgcn::geomparam::EsCount] = uint32_t(esCount);
            params[vpgcn::geomparam::Entries] = entries;
            params[vpgcn::geomparam::FirstVertex] = d.firstVertex;
            params[vpgcn::geomparam::Indexed] = d.indexAddress ? (d.index32 ? 2 : 1) : 0;
            params[vpgcn::geomparam::Prims] = prims;
            params[vpgcn::geomparam::GsCount] = uint32_t(gsCount);
            params[vpgcn::geomparam::GsInstances] = gsInst;
            params[vpgcn::geomparam::EsWaveBytes] = g.esgsItemDwords * 256;
            params[vpgcn::geomparam::GsWaveBytes] = g.gsvsVertDwords * g.maxVertOut * 256;
            params[vpgcn::geomparam::Instances] = d.instances;
            if (!uploadRing(params.data(), sizeof params, ring(GeomRing::Params))) return skip("upload ring full");
            if (d.indexAddress) {
                const uint64_t stride = d.index32 ? 4 : 2;
                std::vector<uint8_t> ib(uint64_t(entries) * stride + 4, 0);
                if (!readGuest(d.indexAddress, ib.data(), uint64_t(entries) * stride)) return skip("index buffer outside guest memory");
                if (!uploadRing(ib.data(), ib.size(), ring(GeomRing::InputIndex))) return skip("upload ring full");
            } else {
                const uint32_t zero = 0;
                if (!uploadRing(&zero, 4, ring(GeomRing::InputIndex))) return skip("upload ring full");
            }
            if (!vpgcn::slotsInKernel(inPrim)) {
                const unsigned nv = vpgcn::inputVertices(inPrim);
                std::vector<uint32_t> table(size_t(prims) * nv);
                for (uint32_t p = 0; p < prims; ++p)
                    for (unsigned j = 0; j < nv; ++j) table[size_t(p) * nv + j] = vpgcn::primitiveSlot(inPrim, entries, p, j);
                if (!uploadRing(table.data(), 4 * table.size(), ring(GeomRing::SlotTable))) return skip("upload ring full");
            }
            esJob.groups = {uint32_t(esWaves), 1, 1};
            gsJob.groups = {uint32_t(gsWaves), 1, 1};
            gsDrawIndices = cap * gsCount;
            if (g.maxVertOut > 1024) note("geometry shader with a very large max vertex count");
        } else if (rect) {
            // RectList: three vertices per rectangle; the lifted VS builds the rectangle (two triangles).
            rectCount = d.count / 3;
            if (!rectCount) return skip("empty RectList");
            std::vector<uint32_t> ids(size_t(rectCount) * 3);
            if (d.indexAddress) {
                const uint64_t stride = d.index32 ? 4 : 2;
                std::vector<uint8_t> ib(uint64_t(ids.size()) * stride);
                if (!readGuest(d.indexAddress, ib.data(), ib.size())) return skip("index buffer outside guest memory");
                for (size_t k = 0; k < ids.size(); ++k) {
                    uint32_t v = 0;
                    std::memcpy(&v, &ib[k * stride], stride);
                    ids[k] = v + d.firstVertex;
                }
            } else {
                for (size_t k = 0; k < ids.size(); ++k) ids[k] = d.firstVertex + uint32_t(k);
            }
            if (!uploadRing(ids.data(), 4 * ids.size(), ringBinds[size_t(vpgcn::GeomRing::RectTable)])) return skip("upload ring full");
            note("RectList drawn as two triangles per rectangle (fourth vertex derived)");
        }

        // Bindings (texture uploads close the pass: done before it is opened).
        const auto tBind = Clock::now();
        if (geom) {
            if (!bind(esStage, *esT, vpgcn::ShaderStage::Export, vp4gpu::stage::compute, esJob.bindings, why)) return skip("es: " + why);
            if (!bind(gsStage, *gsT, vpgcn::ShaderStage::Geometry, vp4gpu::stage::compute, gsJob.bindings, why)) return skip("gs: " + why);
        }
        if (!bind(*vsSrc, *vs, geom ? vpgcn::ShaderStage::CopyVertex : vpgcn::ShaderStage::Vertex, vp4gpu::stage::vertex, draw.bindings, why))
            return skip("vs: " + why);
        if (fs && !bind(psStage, *fs, vpgcn::ShaderStage::Pixel, vp4gpu::stage::fragment, draw.bindings, why)) return skip("ps: " + why);
        draw.views = std::move(pendingViews);
        pendingViews.clear();
        r.cpu_.bindMs += msSince(tBind);

        // Geometry.
        ps.primitive = toPrimitive(d.primitive);
        if (geom) ps.primitive = d.geom.outPrim == 0 ? vp4gpu::Primitive::Points
                                 : d.geom.outPrim == 1 ? vp4gpu::Primitive::Lines : vp4gpu::Primitive::Triangles;
        draw.primitive = ps.primitive;
        draw.instances = d.instances;
        if (geom) {
            // The GS output is an index list into the copy shader's vertices; every instance is in it.
            const RingBinding& gi = ringBinds[size_t(vpgcn::GeomRing::GsIndex)];
            draw.instances = 1;
            draw.index = vp4gpu::IndexBuffer{gi.id, gi.offset, true};
            draw.count = uint32_t(gsDrawIndices);
        } else if (rect) {
            draw.first = 0;
            draw.count = 6 * rectCount;
        } else if (d.indirectAddress) {
            const bool conv = needsConversion(d.primitive);
            if (!conv && indicesNeedPrep(d)) note("index swap / restart index ignored on an indirect draw");
            BufferId id;
            uint64_t offset = 0, available = 0;
            const uint64_t argBytes = d.indexAddress ? 20 : 16;
            if (!rawBuffer(d.indirectAddress, argBytes, false, id, offset, available) || argBytes > available || offset % 4)
                return skip("indirect arguments outside guest memory");
            vp4gpu::Indirect indirect{id, offset};
            if (conv || d.indirectCountAddress) {
                if (!convertIndirect(d, conv, id, offset, indirect, draw, why)) return skip(why);
            }
            draw.indirect = indirect;
            if (conv) {
                // (index buffer and count come from the converter)
            } else if (d.indexAddress) {
                BufferId ib;
                uint64_t ioff = 0, iavail = 0;
                if (!rawBuffer(d.indexAddress, uint64_t(std::max(d.count, 1u)) * (d.index32 ? 4 : 2), false, ib, ioff, iavail)) return skip("index buffer outside guest memory");
                if (ioff % (d.index32 ? 4 : 2)) return skip("misaligned index buffer");
                draw.index = vp4gpu::IndexBuffer{ib, ioff, d.index32};
                draw.count = uint32_t(std::min<uint64_t>(d.count, iavail / (d.index32 ? 4 : 2)));
            } else {
                draw.count = std::max(d.count, 1u);
            }
            ++r.stats_.indirect;
        } else if (needsConversion(d.primitive)) {
            std::vector<uint32_t> order = convertedOrder(d.primitive, d.count);
            if (order.empty()) return skip("degenerate primitive");
            std::vector<uint32_t> indices(order.size());
            if (d.indexAddress) {
                const uint64_t bytes = uint64_t(d.count) * (d.index32 ? 4 : 2);
                std::vector<uint8_t> src(bytes);
                if (!readGuest(d.indexAddress, src.data(), bytes)) return skip("index buffer outside guest memory");
                if (indicesNeedPrep(d)) {
                    std::vector<uint8_t> swapped;
                    prepareIndices(d, src, swapped);
                    src.swap(swapped);
                }
                for (size_t k = 0; k < order.size(); ++k) {
                    if (d.index32) std::memcpy(&indices[k], &src[4 * order[k]], 4);
                    else {
                        uint16_t v;
                        std::memcpy(&v, &src[2 * order[k]], 2);
                        indices[k] = v;
                    }
                }
                draw.baseVertex = int32_t(d.firstVertex);
            } else {
                for (size_t k = 0; k < order.size(); ++k) indices[k] = d.firstVertex + order[k];
            }
            auto at = upload(indices.data(), 4 * indices.size(), 4);
            if (!at) return skip("upload ring full");
            draw.index = vp4gpu::IndexBuffer{rings[ringIndex].id, *at, true};
            draw.count = uint32_t(indices.size());
            note(isAdjacency(d.primitive) ? "adjacency vertices dropped (no geometry shader)"
                                          : "fan/quad/loop primitives converted to lists");
        } else if (d.indexAddress && indicesNeedPrep(d)) {
            // Byte-swapped indices, or a primitive restart index other than the host's (all ones).
            const uint64_t bytes = uint64_t(d.count) * (d.index32 ? 4 : 2);
            std::vector<uint8_t> src(bytes), prepared;
            if (!readGuest(d.indexAddress, src.data(), bytes)) return skip("index buffer outside guest memory");
            prepareIndices(d, src, prepared);
            auto at = upload(prepared.data(), prepared.size(), 4);
            if (!at) return skip("upload ring full");
            draw.index = vp4gpu::IndexBuffer{rings[ringIndex].id, *at, d.index32};
            draw.count = d.count;
            draw.baseVertex = int32_t(d.firstVertex);
            note(d.indexSwap ? "index buffer byte-swapped on the CPU" : "primitive restart index rewritten on the CPU");
        } else if (d.indexAddress) {
            BufferId id;
            uint64_t offset = 0, available = 0;
            const uint64_t bytes = uint64_t(d.count) * (d.index32 ? 4 : 2);
            if (!rawBuffer(d.indexAddress, bytes, false, id, offset, available) || bytes > available)
                return skip("index buffer outside guest memory");
            if (offset % (d.index32 ? 4 : 2)) return skip("misaligned index buffer");
            draw.index = vp4gpu::IndexBuffer{id, offset, d.index32};
            draw.count = d.count;
            draw.baseVertex = int32_t(d.firstVertex);
        } else {
            draw.first = d.firstVertex;
            draw.count = d.count;
        }

        // Fixed function.
        ps.cull = d.raster.cullBack ? vp4gpu::CullMode::Back : d.raster.cullFront ? vp4gpu::CullMode::Front : vp4gpu::CullMode::None;
        if (rect) ps.cull = vp4gpu::CullMode::None;  // (rectangles have no facing)
        draw.frontFace = d.raster.frontFaceClockwise ? vp4gpu::Winding::Clockwise : vp4gpu::Winding::CounterClockwise;
        draw.stencilRef = stencil.refFront;
        draw.stencilRefBack = stencil.refBack;
        draw.blendColor = d.blendColor;
        const Viewport& v = d.viewport;
        if (v.xScale != 0 && v.yScale != 0) {
            // GCN: screen = ndc * scale + offset (y down when yScale < 0). Metal: y down from the top.
            if (v.yScale > 0) {
                // Metal cannot flip y in the viewport: the image comes out upside down; at least keep
                // the facing right (mirroring swaps the winding).
                note("viewport with y up (positive y scale) drawn flipped");
                draw.frontFace = draw.frontFace == vp4gpu::Winding::Clockwise ? vp4gpu::Winding::CounterClockwise
                                                                             : vp4gpu::Winding::Clockwise;
            }
            const float h = 2 * std::abs(v.yScale);
            const ViewportZ vz = viewportDepthRange(v, d.raster.clipSpaceZeroToW);
            draw.viewport = {v.xOffset - std::abs(v.xScale), v.yOffset - std::abs(v.yScale), 2 * std::abs(v.xScale), h,
                             vz.znear, vz.zfar};
        }
        if (scissorSet) draw.scissor = {uint32_t(sl), uint32_t(stp), uint32_t(sr - sl), uint32_t(sb - stp)};
        // Polygon offset, depth clamp, wireframe.
        if (zf != vp4gpu::Format::Undefined && z.zFormat) {
            const DepthBias bias = resolveDepthBias(d.raster, d.primitive, z.zFormat, z.stencilFormat != 0);
            if (bias.enabled) {
                draw.depthBias = bias.constant;
                draw.depthBiasSlope = bias.slope;
                draw.depthBiasClamp = bias.clamp;
                if (bias.approximated) note("front and back polygon offsets differ: one used for both faces");
            }
        }
        bool clampApprox = false;
        draw.depthClamp = depthClamp(d.raster, clampApprox);
        if (clampApprox) note("depth clip disabled on one plane only: both clamped");
        const PolygonMode pm = resolvePolygonMode(d.raster, d.primitive);
        draw.wireframe = pm.wireframe;
        if (pm.pointMode) note("polygon mode point drawn filled");
        if (d.raster.userClipPlanes) note("user clip planes ignored");
        if (d.raster.provokingLast) note("provoking vertex last ignored (flat attributes take the first vertex)");
        if (d.raster.vsOutputsViewportIndex) note("viewport index output: only viewport 0 is set up");
        if (d.rop3 != 0xCC) note("logic op (ROP3) ignored");
        if (d.degamma) note("CB degamma ignored");
        const auto tPipe = Clock::now();
        // The label only matters for a compile: built only when the state is not ready yet.
        const vp4gpu::PipelineKey key = ps.key();  // (the label is not part of it)
        vp4gpu::PipelineCache::Status st = vp4gpu::PipelineCache::Status::Ready;
        if (!r.device_.pipelines().ready(key)) {
            ps.label = "vs " + hex(vs->id) + (fs ? " ps " + hex(fs->id) : "");
            st = r.device_.pipelines().request(ps, key, vp4gpu::Priority::Visible);
        }
        r.cpu_.pipelineMs += msSince(tPipe);
        if (st == vp4gpu::PipelineCache::Status::Failed) return skip("pipeline failed");
        if (st != vp4gpu::PipelineCache::Status::Ready) {
            // Still compiling: the draw is recorded all the same and Device::submit holds the frame (bounded,
            // Device::holdMs) until its pipelines are built, instead of dropping it (holes in the picture).
            if (r.device_.holdMs() == 0) return skip("pipeline compiling");
            note("draw recorded before its pipeline was ready (frame held)");
        }
        draw.pipeline = key;

        // Clears that were waiting for this target become the load action of the pass.
        if (!pendingClears.empty()) {
            PendingClear pc;
            for (vp4gpu::Attachment& a : desc.colors)
                if (a.texture && takePendingClear(a.texture, pc)) {
                    a.load = vp4gpu::LoadAction::Clear;
                    a.clear = pc.value;
                    clears = true;
                }
            if (desc.depth.texture && takePendingClear(desc.depth.texture, pc)) {
                desc.depth.load = vp4gpu::LoadAction::Clear;
                desc.depth.clear = pc.value;
                clears = true;
            }
        }
        if (geom) {
            auto computeStage = [&](const Translated& t, const char* what, vp4gpu::Dispatch& job) {
                vp4gpu::PipelineState cs;
                cs.computeShader = t.id;
                cs.label = std::string(what) + " " + hex(t.id);
                const vp4gpu::PipelineCache::Status cst = r.device_.pipelines().request(cs, vp4gpu::Priority::Visible);
                if (cst == vp4gpu::PipelineCache::Status::Failed) {
                    skip("pipeline failed");
                    return false;
                }
                if (cst != vp4gpu::PipelineCache::Status::Ready) {
                    if (r.device_.holdMs() == 0) {
                        skip("pipeline compiling");
                        return false;
                    }
                    note("geometry stage recorded before its pipeline was ready (frame held)");
                }
                job.pipeline = cs.key();
                job.label = cs.label;
                return true;
            };
            if (!computeStage(*esT, "es", esJob) || !computeStage(*gsT, "gs", gsJob)) return;
            // The expansion runs before the render pass the draw joins (hoisted above it, the pass stays
            // whole) unless it touches something the pass has written so far; then it runs in between and
            // the pass is split (the compute work reads a finished picture).
            const vp4gpu::RenderPass* open = frame.openRenderPass();
            const bool joins = passOpen && !clears && open && pass.sameTargets(desc);
            if (joins && !vp4gpu::dispatchHazard(esJob, *open) && !vp4gpu::dispatchHazard(gsJob, *open)) {
                frame.dispatchBeforeOpenPass(std::move(esJob));
                frame.dispatchBeforeOpenPass(std::move(gsJob));
                ++r.stats_.gsHoisted;
                note("geometry shader expansion hoisted above its render pass");
            } else {
                if (joins) {
                    ++r.stats_.gsSplit;
                    note("geometry shader expansion reads what its render pass wrote: pass split");
                }
                closePass();
                frame.dispatch(std::move(esJob));
                frame.dispatch(std::move(gsJob));
            }
            r.stats_.dispatches += 2;
            r.stats_.dispatched += 2;
            note("geometry shader expanded by compute (ES, GS) before the draw");
        }

        if (vis.active) draw.visibility = vp4gpu::BufferSpan{vis.id, (vis.seq % kVisSlots) * 8, 8};
        openPass(desc, clears);
        if (partialClear) {
            vp4gpu::Draw cd;
            cd.clear = std::move(partialClear);
            frame.draw(std::move(cd));
        }
        if (!frame.draw(std::move(draw))) return skip("no open pass");
        ++r.stats_.drawn;
        // What the pass wrote: those images now hold the newest bytes of their ranges.
        for (const Drawn& dc : drawnColors) {
            latestColor[dc.address] = dc.id;
            markWriteWid(dc.wid);
        }
        if (depthWid && (clears || (d.depthStencil.depthWrite && z.zFormat) || (d.depthStencil.stencilTest && z.stencilFormat)))
            markWriteWid(depthWid);
    }

    void dispatch(const DispatchState& d) {
        pollCache();
        const RegionScope regions(*this);
        ++r.stats_.dispatches;
        if (d.cs.code.empty()) return skip("no compute shader");
        if (!d.indirectAddress && (!d.groups[0] || !d.groups[1] || !d.groups[2])) return skip("empty dispatch");
        const Translated* cs = shader(d.cs, vpgcn::ShaderStage::Compute);
        if (!cs->ok) return skip("compute shader not translated");
        vp4gpu::Dispatch x;
        std::string why;
        pendingViews.clear();
        if (!bind(d.cs, *cs, vpgcn::ShaderStage::Compute, vp4gpu::stage::compute, x.bindings, why)) return skip("cs: " + why);
        x.views = std::move(pendingViews);
        pendingViews.clear();
        vp4gpu::PipelineState ps;
        ps.computeShader = cs->id;
        ps.label = "cs " + hex(cs->id);
        const vp4gpu::PipelineCache::Status st = r.device_.pipelines().request(ps, vp4gpu::Priority::Visible);
        if (st == vp4gpu::PipelineCache::Status::Failed) return skip("pipeline failed");
        if (st != vp4gpu::PipelineCache::Status::Ready) {
            if (r.device_.holdMs() == 0) return skip("pipeline compiling");
            note("dispatch recorded before its pipeline was ready (frame held)");
        }
        x.pipeline = ps.key();
        x.groups = d.groups;
        x.label = ps.label;
        if (d.indirectAddress) {
            BufferId id;
            uint64_t offset = 0, available = 0;
            if (!rawBuffer(d.indirectAddress, 12, false, id, offset, available) || available < 12 || offset % 4)
                return skip("indirect arguments outside guest memory");
            x.indirect = vp4gpu::Indirect{id, offset};
            ++r.stats_.indirect;
        }
        closePass();
        frame.dispatch(std::move(x));
        ++r.stats_.dispatched;
    }

    // Every GPU-side surface that overlaps [address, address + bytes): what a T# there could be
    // reading instead of guest memory (diagnostics for pictures that come out empty).
    std::string describe(uint64_t address, uint64_t bytes) const {
        std::string out;
        auto span = [](const Target& t) {
            const vp4gpu::FormatInfo& fi = vp4gpu::formatInfo(t.format);
            return uint64_t(t.width) * t.height * std::max<uint32_t>(fi.bytesPerBlock, 1) * t.layers /
                   (uint64_t(std::max<uint8_t>(fi.blockSize, 1)) * std::max<uint8_t>(fi.blockSize, 1));
        };
        auto targets = [&](const std::map<uint64_t, std::vector<Target>>& cache, const char* kind) {
            // Surfaces start at most 256 MiB below: enough for any render target.
            auto it = cache.lower_bound(address > (256ull << 20) ? address - (256ull << 20) : 0);
            for (; it != cache.end() && it->first < address + bytes; ++it)
                for (const Target& t : it->second) {
                    if (it->first + span(t) <= address) continue;
                    out += std::string(" ") + kind + "@" + hex(it->first) + "+" + hex(address - std::min(address, it->first)) +
                           (it->first > address ? "(after)" : "") + " " + vp4gpu::formatInfo(t.format).name + " " +
                           std::to_string(t.width) + "x" + std::to_string(t.height) + "x" + std::to_string(t.layers) +
                           " s" + std::to_string(t.samples) + " used" + std::to_string(r.frame_ - t.lastUse);
                }
        };
        targets(colorTargets, "colour");
        targets(depthTargets, "depth");
        if (auto it = textures.find(address); it != textures.end())
            for (const auto& [key, t] : it->second)
                out += std::string(" tex ") + (t.format == vp4gpu::Format::Undefined ? "?" : vp4gpu::formatInfo(t.format).name) +
                       " " + std::to_string(t.width) + "x" + std::to_string(t.height) + (t.storage ? " storage" : "") +
                       (tracker.writerAt(it->first) == t.wid ? " gpu_written" : "") + (tracker.staleAt(it->first) && tracker.writerAt(it->first) == t.wid ? " ahead_of_memory" : "");
        return out.empty() ? " nothing (only guest memory)" : out;
    }

    // Writes the colour target the GPU rendered at c.address back to guest memory in the layout the
    // CB names (tile mode, pitch), as a retile kernel on the GPU timeline: for a CPU that reads a
    // render target, or a later T# that names it as tiled memory. One sample plane; false if the
    // target is not there, is multisampled on the host, or the kernel is not ready (try again).
    bool writeBack(const ColorTarget& c) {
        const vp4gpu::Format f = colorFormat(c.format, c.numberType, c.compSwap);
        auto found = colorTargets.find(c.address);
        if (f == vp4gpu::Format::Undefined || found == colorTargets.end()) return false;
        Target* t = nullptr;
        for (Target& cand : found->second)
            if (cand.format == f && cand.width == c.width && cand.height == c.height) t = &cand;
        if (!t) return false;
        const vp4gpu::FormatInfo& fi = vp4gpu::formatInfo(f);
        TiledSurface s;
        s.tileIndex = c.tileMode;
        s.bytesPerElement = fi.bytesPerBlock;
        s.width = c.width;
        s.height = c.height;
        s.pitch = std::max(c.pitch, c.width);
        s.slices = std::max<uint32_t>(c.sliceStart + c.sliceCount, 1);
        s.samples = std::max<uint8_t>(c.samples, 1);
        const SurfaceLayout lay = computeLayout(s);
        if (!lay.ok) {
            note("render target not written back: " + lay.why);
            return false;
        }
        BufferId gb;
        uint64_t at = 0, available = 0;
        if (!guestBuffer(c.address, lay.bytes, gb, at, available) || available < lay.bytes || at % 4) {
            note("render target write-back outside guest memory");
            return false;
        }
        Subresource sub;
        sub.firstSlice = c.sliceStart;
        sub.sliceCount = std::max<uint16_t>(c.sliceCount, 1);
        closePass();
        if (tiler().retile(frame, t->id, {gb, at, available}, lay, sub) != GpuTiler::Result::Recorded) return false;
        ++r.stats_.gpuRetiled;
        ++r.stats_.retiled;
        ++r.stats_.copies;
        tracker.markSynced(t->wid);  // memory now holds what the image holds
        watchOwned(*t, false);
        return true;
    }

    // The last render pass of the frame that draws into multisampled `ms`, if no command recorded after it
    // touches `dst` or `ms` (so the resolve can move into it without reordering anything dst's readers see).
    vp4gpu::RenderPass* lastPassWriting(TextureId ms, TextureId dst) {
        auto& cmds = frame.mutableCommands();
        size_t looked = 0;
        for (auto c = cmds.rbegin(); c != cmds.rend() && looked < 16; ++c, ++looked) {
            std::vector<Access> acc;
            if (auto* rp = std::get_if<vp4gpu::RenderPass>(&*c)) {
                for (const vp4gpu::Attachment& a : rp->desc.colors)
                    if (a.texture == ms) return rp;
                acc = vp4gpu::attachmentAccesses(rp->desc);
                for (const vp4gpu::Draw& d : rp->draws) {
                    auto da = vp4gpu::drawAccesses(d);
                    acc.insert(acc.end(), da.begin(), da.end());
                }
            } else if (auto* dp = std::get_if<vp4gpu::Dispatch>(&*c)) {
                acc = dp->bindings;
            } else {
                acc = vp4gpu::copyAccesses(std::get<vp4gpu::Copy>(*c));
            }
            for (const Access& a : acc)
                if (a.range.kind == ResourceRange::Kind::Texture &&
                    ((a.range.index == dst.index && a.range.generation == dst.generation) ||
                     (a.range.index == ms.index && a.range.generation == ms.generation)))
                    return nullptr;
        }
        return nullptr;
    }

    // CB_COLOR_CONTROL.MODE = RESOLVE: colour target `src` (CB0) into `dst` (CB1). The way is chosen by
    // planResolve (msaa.h): by the pass that drew into the multisampled source when it still can, else a
    // pass of its own that loads the samples; a 1x source (or one whose single-sample picture is read
    // anyway) is a texture copy.
    void resolve(const ColorTarget& src, const ColorTarget& dst) {
        ++r.stats_.resolves;
        const vp4gpu::Format f = colorFormat(src.format, src.numberType, src.compSwap);
        const vp4gpu::Format fd = colorFormat(dst.format, dst.numberType, dst.compSwap);
        if (f == vp4gpu::Format::Undefined || fd == vp4gpu::Format::Undefined) return skip("resolve: format unsupported");
        // A Metal blit copies between identical pixel formats only.
        if (f != fd) return skip("resolve: source and destination formats differ");
        const uint16_t layers = uint16_t(src.sliceStart + src.sliceCount);
        // The source exists at the count its draws used (the most recent one of this shape).
        Target* from = nullptr;
        for (Target& t : colorTargets[src.address])
            if (t.format == f && t.width == src.width && t.height == src.height && t.layers >= layers &&
                (!from || t.lastUse >= from->lastUse))
                from = &t;
        if (!from) {
            const std::array<uint8_t, 1> frag{std::max<uint8_t>(src.samples, 1)};
            from = target(colorTargets, src.address, f, src.width, src.height, planSamples(frag, 0, r.maxSamples_).host, layers,
                          false, src.tileMode, src.pitch);
        }
        if (!from) return skip("resolve: target creation failed");
        const uint32_t fromWid = from->wid;
        Target* to = target(colorTargets, dst.address, fd, dst.width, dst.height, 1, uint16_t(dst.sliceStart + dst.sliceCount),
                            false, dst.tileMode, dst.pitch);
        if (!to) return skip("resolve: target creation failed");
        const uint32_t toWid = to->wid;
        // (the second target() may have moved the first when both live at one address)
        from = static_cast<Target*>(surfaceOf(fromWid));
        to = static_cast<Target*>(surfaceOf(toWid));
        if (!from->ms) refresh(*from, nullptr);
        refresh(*to, nullptr);
        flushClear(*from);
        ResolveCase rc;
        rc.sourceMultisampled = from->ms && !life.wants(from->id, r.frame_);
        rc.destinationIsSource = from->id == to->id;
        closePass();
        vp4gpu::RenderPass* last = rc.sourceMultisampled ? lastPassWriting(from->ms, to->id) : nullptr;
        rc.lastPassOpenForSource = last != nullptr;
        const uint16_t n = uint16_t(std::max<uint16_t>(std::min(src.sliceCount, dst.sliceCount), 1));
        switch (planResolve(rc)) {
            case ResolveWay::Skip: return skip("resolve: destination is the source");
            case ResolveWay::InPass:
                for (vp4gpu::Attachment& a : last->desc.colors)
                    if (a.texture == from->ms) {
                        a.resolve = to->id;
                        a.resolveLayer = dst.sliceStart;
                        a.store = vp4gpu::StoreAction::StoreAndResolve;
                    }
                note("MSAA resolve done by the pass that drew the samples");
                break;
            case ResolveWay::OwnPass: {
                vp4gpu::RenderPassDesc desc;
                vp4gpu::Attachment& a = desc.colors[0];
                a.texture = from->ms;
                a.layer = src.sliceStart;
                a.load = vp4gpu::LoadAction::Load;
                a.store = vp4gpu::StoreAction::Resolve;
                a.resolve = to->id;
                a.resolveLayer = dst.sliceStart;
                desc.label = "resolve " + hex(src.address) + " -> " + hex(dst.address);
                frame.beginRenderPass(desc);
                frame.endRenderPass();
                ++r.stats_.passes;
                note("MSAA resolve in a pass of its own");
                break;
            }
            case ResolveWay::Copy: {
                // (a multisampled source only gets here while its single-sample picture is wanted anyway:
                // young or read by others, so the passes that drew into it resolve into it)
                vp4gpu::Copy c;
                c.kind = vp4gpu::CopyKind::TextureToTexture;
                c.srcTexture.texture = from->id;
                c.srcTexture.layer = src.sliceStart;
                c.dstTexture.texture = to->id;
                c.dstTexture.layer = dst.sliceStart;
                c.srcTexture.layerCount = c.dstTexture.layerCount = n;
                c.srcTexture.width = c.dstTexture.width = std::min(src.width, dst.width);
                c.srcTexture.height = c.dstTexture.height = std::min(src.height, dst.height);
                c.label = "resolve " + hex(src.address) + " -> " + hex(dst.address);
                frame.copy(c);
                ++r.stats_.copies;
                break;
            }
        }
        latestColor[dst.address] = to->id;
        life.noteRead(from->ms ? from->ms : from->id, r.frame_);  // (the source is read by the resolve)
        markWriteWid(toWid);
    }

    bool fill(uint64_t address, uint64_t bytes, uint32_t value) {
        BufferId id;
        uint64_t offset = 0, available = 0;
        if (!bytes) return true;
        if (!guestBuffer(address, bytes, id, offset, available) || bytes > available) {
            note("DMA fill outside guest memory: dropped");
            return false;
        }
        syncOwners(address, bytes, true);  // (images it covers completely are simply overwritten)
        closePass();
        vp4gpu::Copy c;
        c.dstBuffer = {id, offset, bytes, 0, 0};
        c.label = "fill " + hex(address);
        const uint8_t b0 = uint8_t(value);
        if (value == uint32_t(b0) * 0x01010101u) {
            c.kind = vp4gpu::CopyKind::FillBuffer;
            c.fillByte = b0;
        } else {
            BufferId sid;
            uint64_t so = 0;
            uint8_t* p = staging(bytes, sid, so);
            if (!p) {
                note("no staging memory for a DMA fill: dropped");
                return false;
            }
            for (uint64_t k = 0; k + 4 <= bytes; k += 4) std::memcpy(p + k, &value, 4);
            for (uint64_t k = bytes & ~uint64_t(3); k < bytes; ++k) p[k] = uint8_t(value >> (8 * (k & 3)));
            c.kind = vp4gpu::CopyKind::BufferToBuffer;
            c.srcBuffer = {sid, so, bytes, 0, 0};
        }
        frame.copy(c);
        ++r.stats_.copies;
        ++programGeneration;  // (code may be what was written)
        tracker.markGpuWrite(address, bytes, MemoryTracker::kDma);
        return true;
    }

    bool copy(uint64_t dst, uint64_t src, uint64_t bytes) {
        if (!bytes) return true;
        BufferId di, si;
        uint64_t doff = 0, dav = 0, soff = 0, sav = 0;
        if (!guestBuffer(dst, bytes, di, doff, dav) || bytes > dav || !guestBuffer(src, bytes, si, soff, sav) || bytes > sav) {
            note("DMA copy outside guest memory: dropped");
            return false;
        }
        syncOwners(src, bytes, false);
        syncOwners(dst, bytes, true);
        closePass();
        vp4gpu::Copy c;
        c.kind = vp4gpu::CopyKind::BufferToBuffer;
        c.srcBuffer = {si, soff, bytes, 0, 0};
        c.dstBuffer = {di, doff, bytes, 0, 0};
        c.label = "dma " + hex(src) + " -> " + hex(dst);
        frame.copy(c);
        ++r.stats_.copies;
        ++programGeneration;
        tracker.markGpuWrite(dst, bytes, MemoryTracker::kDma);
        return true;
    }

    // The program keys that cover [address, +size) are hashed again (the others stay; if the range
    // cannot be placed, all of them).
    void forgetPrograms(uint64_t address, uint64_t size) {
        GuestRegion w;
        if (size && region(address, size, w)) {
            const uint8_t* lo = static_cast<const uint8_t*>(w.host) + (address - w.base);
            const uint8_t* hi = lo + size;
            auto hits = [&](const uint32_t* p, size_t words) {
                const uint8_t* a = reinterpret_cast<const uint8_t*>(p);
                return words && a < hi && a + 4 * words > lo;
            };
            for (ProgramMemo& m : programMemo)
                if (hits(m.code, m.codeWords) || hits(m.fetch, m.fetchWords)) m.generation = ~uint64_t(0);
        } else {
            ++programGeneration;
        }
    }

    // ACQUIRE_MEM / SURFACE_SYNC with a bounded range and cache actions (texture cache, scalar cache,
    // instruction cache): the title says memory there was written and the GPU must read it again.
    // Nothing is written back or marked written (a whole-memory acquire says nothing specific, and a
    // render target flush needs nothing: the tracker already orders images); what the host cannot
    // watch is looked at again: shader code hashes, and the content of textures whose source pages
    // are not write-watched (checked in full the next time they are used).
    void coherencePoint(uint64_t address, uint64_t size) {
        if (!size || size >= kMaxRawExtent) return;
        forgetPrograms(address, size);
        auto it = textures.lower_bound(address > (1ull << 30) ? address - (1ull << 30) : 0);
        for (; it != textures.end() && it->first < address + size; ++it)
            for (auto& [shape, t] : it->second)
                if (!t.sourceWatched && it->first + std::max<uint64_t>(t.bytes, 1) > address) {
                    t.checkedFrame = ~uint64_t(0);
                    t.fullFrame = 0;
                }
    }

    // The CPU wrote guest memory: every image over the range sees a newer version of it.
    void invalidate(uint64_t address, uint64_t size) {
        forgetPrograms(address, size);
        tracker.markCpuWrite(address, std::max<uint64_t>(size, 1));
        // The host dropped its write watches on the pages that were hit.
        auto it = textures.lower_bound(address > (1ull << 30) ? address - (1ull << 30) : 0);
        for (; it != textures.end() && it->first < address + size; ++it)
            for (auto& [shape, t] : it->second)
                if (it->first + std::max<uint64_t>(t.bytes, 1) > address) t.sourceWatched = false;
        for (auto* cache : {&colorTargets, &depthTargets}) {
            auto c = cache->lower_bound(address > (1ull << 30) ? address - (1ull << 30) : 0);
            for (; c != cache->end() && c->first < address + size; ++c)
                for (Target& t : c->second)
                    if (c->first + std::max<uint64_t>(t.bytes, 1) > address) t.sourceWatched = false;
        }
    }

    // The CPU is about to touch [address, +size): images ahead of memory there are written back.
    bool syncToMemory(uint64_t address, uint64_t size) {
        const uint64_t n = std::max<uint64_t>(size, 1);
        const bool any = syncOwners(address, n, false);
        if (any) ++r.stats_.syncs;
        // The host releases the pages it protected for the access: images that were watched and are
        // still around (their claim on the range was taken over meanwhile) are not any more.
        auto overlaps = [&](const Surface& s) { return s.address < address + n && s.address + s.bytes > address; };
        auto unwatch = [&](Surface& s) {
            if (s.ownedWatched && overlaps(s)) watchOwned(s, false);
        };
        for (auto& [a, v] : colorTargets) for (Target& t : v) unwatch(t);
        for (auto& [a, v] : depthTargets) for (Target& t : v) unwatch(t);
        for (auto& [a, m] : textures) for (auto& [k, t] : m) unwatch(t);
        return any;
    }

    vp4gpu::FrameReport submit() {
        flushReports();
        flushAllClears();  // (a clear nobody drew after still clears)
        closePass();
        // (the write backs of evicted images are recorded into this frame)
        if (r.frame_ % 16 == 15) {
            evict();
            closePass();
        }
        ++r.cpu_.submits;
        if (frame.commands().empty()) {  // nothing recorded (a Flush after a Flush): no GPU frame
            ++r.cpu_.emptySubmits;
            busy.clear();  // (ranges bound by draws that were skipped afterwards)
            busyAll = false;
            touchedLo = touchedHi = 0;  // (touch() skips ranges inside the last one it added: that set is gone)
            return {};
        }
        const auto t0 = Clock::now();
        // Load/store/resolve actions for a tile-based GPU, now that the whole frame is known.
        {
            const LifetimeStats ls = life.plan(frame, r.device_.resources(), r.frame_);
            lifeTotal.storesDropped += ls.storesDropped;
            lifeTotal.samplesDropped += ls.samplesDropped;
            lifeTotal.resolvesDropped += ls.resolvesDropped;
            lifeTotal.resolvesDeferred += ls.resolvesDeferred;
            lifeTotal.storesKept += ls.storesKept;
            if (ls.misses) {
                lifeTotal.misses += ls.misses;
                note("lifetime: contents dropped by an earlier frame were read (texture kept stored from now on)");
            }
        }
        vp4gpu::FrameReport report = r.device_.submit(std::move(frame));
        r.cpu_.submitMs += msSince(t0);
        frame = vp4gpu::Frame{};
        if (!busy.empty() || busyAll) {
            inFlight.push_back({report.frame, std::move(busy), busyAll});
            busy.clear();
            busyAll = false;
            touchedLo = touchedHi = 0;
            if (inFlight.size() > 64) {  // a GPU far behind: fold the oldest into one "everything"
                inFlight.pop_front();
                inFlight.front().all = true;
            }
        }
        ringIndex = (ringIndex + 1) % kFrames;
        rings[ringIndex].head = 0;
        ++ringGeneration;
        ++programGeneration;
        std::erase_if(retired, [&](const Retired& x) {
            if (x.after > r.frame_ + 1) return false;
            r.device_.destroy(x.id);
            return true;
        });
        std::erase_if(retiredViews, [&](const RetiredTexture& x) {
            if (x.after > r.frame_ + 1) return false;
            if (r.hooks_.destroyView) r.hooks_.destroyView(x.id);
            return true;
        });
        std::erase_if(retiredTextures, [&](const RetiredTexture& x) {
            if (x.after > r.frame_ + 1) return false;
            r.device_.destroy(x.id);
            return true;
        });
        std::erase_if(stagings, [&](const Staging& x) {
            if (x.after > r.frame_ + 1) return false;
            r.device_.destroy(x.id);
            std::free(x.memory);
            return true;
        });
        return report;
    }

    ~Impl() {
        if (gds.id) r.device_.destroy(gds.id);
        std::free(gds.memory);
        if (dummy.id) r.device_.destroy(dummy.id);
        std::free(dummy.memory);
        for (Staging& st : stagings) {
            r.device_.destroy(st.id);
            std::free(st.memory);
        }
        for (GeomScratch& x : scratchPool) {
            r.device_.destroy(x.id);
            std::free(x.memory);
        }
        for (Ring& ring : rings)
            if (ring.memory) {
                r.device_.destroy(ring.id);
                std::free(ring.memory);
            }
    }
};

Renderer::Renderer(vp4gpu::Device& device, Hooks hooks)
    : device_(device), hooks_(std::move(hooks)), impl_(std::make_unique<Impl>(*this)) {}

Renderer::~Renderer() {
    if (impl_) impl_->flushReports();
}

void Renderer::flushReports() { impl_->flushReports(); }

void Renderer::setCacheRoot(std::filesystem::path root) {
    if (root.empty() || impl_->cache) return;
    impl_->cache = std::make_unique<tcache::Worker>(std::move(root), tcache::buildFingerprint(), device_.pipelines());
    impl_->cacheLoading = true;
}

void Renderer::waitCache() {
    if (!impl_->cache) return;
    impl_->cache->waitLoaded();
    while (impl_->cacheLoading) impl_->pollCache();
}

void Renderer::flushCache() {
    if (impl_->cache) impl_->cache->flush();
}

void Renderer::setMaxSamples(uint8_t samples) {
    samples = std::clamp<uint8_t>(samples, 1, 8);
    while (samples & (samples - 1)) samples &= uint8_t(samples - 1);
    maxSamples_ = samples;
}

void Renderer::setGpuTiling(bool on) { impl_->gpuTiling = on; }

void Renderer::guardRunaway() {
    if (++sinceSubmit_ < kRunawayCommands) return;
    if (!runawayTold_ && hooks_.log)
        hooks_.log("gpu.err what=frame_not_ended commands=" + std::to_string(sinceSubmit_) +
                   " (the host never called submit: submitted here)");
    runawayTold_ = true;
    submit();
}

void Renderer::draw(const DrawState& d) {
    const auto t0 = Clock::now();
    impl_->draw(d);
    cpu_.drawMs += msSince(t0);
    ++cpu_.calls;
    guardRunaway();
}
void Renderer::dispatch(const DispatchState& d) {
    const auto t0 = Clock::now();
    impl_->dispatch(d);
    cpu_.drawMs += msSince(t0);
    ++cpu_.calls;
    guardRunaway();
}
void Renderer::resolve(const ColorTarget& src, const ColorTarget& dst) { impl_->resolve(src, dst); }
bool Renderer::writeBack(const ColorTarget& c) { return impl_->writeBack(c); }
void Renderer::invalidate(uint64_t address, uint64_t size) { impl_->invalidate(address, size); }
bool Renderer::syncToMemory(uint64_t address, uint64_t size) { return impl_->syncToMemory(address, size); }
void Renderer::coherencePoint(uint64_t address, uint64_t size) { impl_->coherencePoint(address, size); }
void Renderer::noteGpuWrite(uint64_t address, uint64_t size) {
    impl_->syncOwners(address, size, true);
    impl_->tracker.markGpuWrite(address, size, MemoryTracker::kGpuRaw);
}
void Renderer::setResidency(uint64_t idleSubmits, uint64_t byteBudget) {
    impl_->idleSubmits = idleSubmits;
    impl_->byteBudget = byteBudget;
}
const MemoryTracker& Renderer::tracker() const { return impl_->tracker; }
bool Renderer::gpuMayAccess(uint64_t address, uint64_t size) const { return impl_->mayAccess(address, size); }
bool Renderer::recordedMayAccess(uint64_t address, uint64_t size) const {
    return Impl::overlaps(impl_->busy, impl_->busyAll, address, size);
}
bool Renderer::zpassDone(uint64_t address, uint32_t pairs) { return impl_->zpassDone(address, pairs); }
void Renderer::drainOcclusion(bool gpuIdle) { impl_->drainOcclusion(gpuIdle); }
bool Renderer::occlusionPending(uint64_t address, uint64_t size) const { return impl_->occlusionPending(address, size); }
void Renderer::submittedIdle() { impl_->inFlight.clear(); }
bool Renderer::hasRecordedWork() const { return !impl_->frame.commands().empty(); }
uint64_t Renderer::recordedCommands() const { return impl_->frame.commands().size(); }
uint64_t Renderer::waitTarget(uint64_t a, uint64_t n, uint64_t b, uint64_t m) const {
    Impl& i = *impl_;
    if (Impl::overlaps(i.busy, i.busyAll, a, n) || (m && Impl::overlaps(i.busy, i.busyAll, b, m))) return kUnsubmitted;
    i.retireBusy();
    uint64_t target = 0;
    for (const Impl::InFlight& f : i.inFlight)
        if (Impl::overlaps(f.ranges, f.all, a, n) || (m && Impl::overlaps(f.ranges, f.all, b, m)))
            target = std::max(target, f.frame + 1);
    return target;
}
void Renderer::waitedBelow(uint64_t frames) {
    auto& q = impl_->inFlight;
    while (!q.empty() && q.front().frame < frames) q.pop_front();
}
uint64_t Renderer::gdsAddress() const { return reinterpret_cast<uint64_t>(impl_->gds.memory); }
void Renderer::afterGpu(uint64_t address, uint32_t size, std::function<void()> action) {
    const uint64_t submitted = device_.submittedFrames();
    if (hasRecordedWork()) {
        device_.afterFrame(submitted, address, size, std::move(action));  // the frame the next submit() makes
    } else if (submitted == 0) {
        action();  // nothing ever ran
    } else {
        device_.afterFrame(submitted - 1, address, size, std::move(action));
    }
}
bool Renderer::labelPending(uint64_t address, uint32_t size) const { return device_.labelPending(address, size); }
uint64_t Renderer::labelWaitTarget(uint64_t address, uint32_t size) const { return device_.labelWaitTarget(address, size); }
void Renderer::gpuIdle() {
    impl_->busy.clear();
    impl_->busyAll = false;
    impl_->touchedLo = impl_->touchedHi = 0;
    impl_->inFlight.clear();
}

CpuTimes Renderer::takeCpuTimes() {
    CpuTimes t = cpu_;
    cpu_ = {};
    return t;
}

vp4gpu::FrameReport Renderer::submit() {
    sinceSubmit_ = 0;
    vp4gpu::FrameReport report = impl_->submit();
    ++frame_;
    if (summaryEvery_ && frame_ % summaryEvery_ == 0) {
        std::string line = "vpps4.frame n=" + std::to_string(frame_) + " draws=" + std::to_string(stats_.drawn) + "/" +
                           std::to_string(stats_.draws) + " dispatches=" + std::to_string(stats_.dispatched) + "/" +
                           std::to_string(stats_.dispatches) + " passes=" + std::to_string(stats_.passes) +
                           " shaders=" + std::to_string(stats_.shaders) + " shader_errors=" + std::to_string(stats_.shaderErrors) +
                           " textures=" + std::to_string(stats_.textures) + " detiled=" + std::to_string(stats_.detiled) + " gpu_detiled=" + std::to_string(stats_.gpuDetiled) + " indirect=" + std::to_string(stats_.indirect) + " targets=" + std::to_string(stats_.targets) +
                           " uploads=" + std::to_string(stats_.uploads) + " resolves=" + std::to_string(stats_.resolves) +
                           " stores_dropped=" + std::to_string(impl_->lifeTotal.storesDropped) + " samples_dropped=" +
                           std::to_string(impl_->lifeTotal.samplesDropped) + " resolves_dropped=" +
                           std::to_string(impl_->lifeTotal.resolvesDropped) + " life_misses=" +
                           std::to_string(impl_->lifeTotal.misses);
        for (const auto& [why, n] : stats_.skipped) line += " skip[" + why + "]=" + std::to_string(n);
        if (hooks_.log) hooks_.log(line);
    }
    return report;
}

vp4gpu::TextureId Renderer::textureFor(const ImageDesc& d) {
    const vp4gpu::TextureId t = impl_->texture(d, false);
    impl_->life.noteSticky(t, frame_);  // shown by the host: always stored
    return t;
}
bool Renderer::fill(uint64_t address, uint64_t bytes, uint32_t value) { return impl_->fill(address, bytes, value); }
bool Renderer::copy(uint64_t dst, uint64_t src, uint64_t bytes) { return impl_->copy(dst, src, bytes); }

uint32_t* Renderer::gds() {
    return impl_->gdsBuffer() ? static_cast<uint32_t*>(impl_->gds.memory) : nullptr;
}

std::string Renderer::describeAt(uint64_t address, uint64_t bytes) const { return impl_->describe(address, bytes); }

vp4gpu::TextureId Renderer::colorTargetSlice(uint64_t address, uint32_t width, uint32_t height, uint16_t& layer) const {
    layer = 0;
    const Impl::Target* t = impl_->targetSlice(address, width, height, layer);
    if (t) impl_->life.noteSticky(t->id, frame_);
    return t ? t->id : vp4gpu::TextureId{};
}

vp4gpu::TextureId Renderer::colorTargetAt(uint64_t address) const {
    auto it = impl_->latestColor.find(address);
    if (it == impl_->latestColor.end()) return {};
    impl_->life.noteSticky(it->second, frame_);  // what a flip presents is always stored
    return it->second;
}

}  // namespace vpps4
