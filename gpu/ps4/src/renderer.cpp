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
#include <unordered_set>
#include <bit>
#include <iterator>

#include "vpgpu/core/probe.h"
#include "vpgcn/gcn.h"
#include "vpgcn/lift.h"
#include "vpps4/bc_decode.h"
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

using vpgpu::Access;
using vpgpu::BufferId;
using vpgpu::ResourceRange;
using vpgpu::SamplerId;
using vpgpu::TextureId;

namespace {

constexpr uint64_t kFnvBasis = 0xcbf29ce484222325ull;

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
// The host exposes guest memory in windows (the region hook); a range that crosses the end of its window is asked for in a
// window widened to hold it, unless it is longer than this (an inflated or unbounded V#): that one keeps the window around
// its first byte, and `available` says where it ends.
constexpr uint64_t kMaxGuestSpan = 256ull << 20;
// A window the backend refuses is retried as the part of it that holds the range, cut at multiples of this.
constexpr uint64_t kShrinkUnit = 16ull << 20;

// The part of region `g` that holds [address, address + size) (cut at kShrinkUnit boundaries, so the same pieces are asked
// for again); false if that is all of `g`. `address` lies in `g`.
bool shrinkRegion(const GuestRegion& g, uint64_t address, uint64_t size, GuestRegion& out) {
    const uint64_t end = g.base + g.size, want = std::max<uint64_t>(size, 1);
    const uint64_t reach = want >= end - address ? end : address + want;
    const uint64_t lo = std::max(g.base, address & ~(kShrinkUnit - 1));
    const uint64_t hi = std::min(end, (reach + kShrinkUnit - 1) & ~(kShrinkUnit - 1));
    if (hi - lo >= g.size) return false;
    out = {lo, hi - lo, static_cast<uint8_t*>(g.host) + (lo - g.base), g.mirrored};
    return true;
}

vpgpu::Primitive toPrimitive(PrimitiveType p) {
    switch (p) {
        case PrimitiveType::PointList: return vpgpu::Primitive::Points;
        case PrimitiveType::LineList:
        case PrimitiveType::LineListAdj:
        case PrimitiveType::LineStripAdj:
        case PrimitiveType::LineLoop: return vpgpu::Primitive::Lines;
        case PrimitiveType::LineStrip: return vpgpu::Primitive::LineStrip;
        case PrimitiveType::TriangleStrip: return vpgpu::Primitive::TriangleStrip;
        default: return vpgpu::Primitive::Triangles;  // lists, and fans/quads/polygons once converted
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

vpir::TexDim texDim(const ImageDesc& d) {
    switch (d.type) {
        case 8: return vpir::TexDim::D1;
        case 10: return vpir::TexDim::D3;
        case 11: return d.lastArray - d.baseArray + 1 > 6 ? vpir::TexDim::CubeArray : vpir::TexDim::Cube;
        case 12:
        case 13:
        case 15: return vpir::TexDim::D2Array;
        default: return vpir::TexDim::D2;
    }
}

vpir::Scalar texElem(const TexturePlan& p) {
    return p.elem == TexturePlan::Uint ? vpir::Scalar::U32 : p.elem == TexturePlan::Sint ? vpir::Scalar::I32 : vpir::Scalar::F32;
}

vpgpu::BindKind bindKind(vpir::ResourceKind k) {
    switch (k) {
        case vpir::ResourceKind::Buffer: return vpgpu::BindKind::Buffer;
        case vpir::ResourceKind::ConstantBuffer: return vpgpu::BindKind::ConstantBuffer;
        case vpir::ResourceKind::Texture: return vpgpu::BindKind::Texture;
        case vpir::ResourceKind::Sampler: return vpgpu::BindKind::Sampler;
        case vpir::ResourceKind::StorageTexture: return vpgpu::BindKind::StorageTexture;
        case vpir::ResourceKind::BufferSize: return vpgpu::BindKind::BufferSize;
        case vpir::ResourceKind::SamplerBias: return vpgpu::BindKind::SamplerBias;
    }
    return vpgpu::BindKind::Auto;
}

}  // namespace

struct Renderer::Impl {
    explicit Impl(Renderer& owner) : r(owner) {}
    Renderer& r;
    vpgpu::Frame frame;

    // Why a T# whose memory an image of the renderer wrote last is not read through that image (see aliasWriter; QUAL alias_rejected_*).
    enum AliasReject : unsigned { kRejDepth, kRejAddress, kRejShape, kRejLayout, kRejMips, kRejSamples, kRejFormat, kRejNoView, kRejRange, kRejFeedback, kAliasReasons };

    // ---- Probe counters (diagnostics; plain integers, the renderer is driven from one thread; published by Renderer::publishProbe)
    struct ProbeStats {
        uint64_t aliasHits = 0, aliasHitsLegacy = 0, aliasRejected[kAliasReasons] = {};  // T#s read through the image that wrote their memory / those that could not
        uint64_t guestBufAdds = 0, guestBufRemoves = 0, guestBufRefused = 0;
        uint64_t uploadBytes = 0, alignedCopies = 0;
        uint64_t refills[3] = {}, refillBytes[3] = {}, refillFailed = 0;  // first fill, written by others (version), history forgotten
        uint64_t refillSkipped = 0, refillSkippedBytes = 0, refillLines = 0;  // refills of a target left out because only images wrote its range (R1)
        uint64_t fillPath[5] = {}, fillPathBytes[5] = {};                  // none, GPU detile, CPU detile, linear upload, not filled
        uint64_t mirrorCalls = 0, mirrorCopies = 0, mirrorBytes = 0;
        uint64_t msDraws[4][4] = {};  // samples the title asked for (1,2,4,8) x samples the host draws with
        std::unordered_map<const char*, uint64_t> resolveDrops;
        struct Plan {
            uint64_t count = 0;
            const char* host = "";
            uint8_t dfmt = 0, nfmt = 0, flags = 0;  // flags: 1 sRGB through unorm, 2 inexact, 4 soft BC, 8 storage
        };
        std::unordered_map<uint64_t, Plan> texPlans;
        // Textures made and refilled from memory by tile mode, T# format and whether level 0 is stored 1D (QUAL tile_plan_*,
        // tile_refill_*; the TILE lines).
        struct TileCell {
            uint64_t textures = 0, refills = 0;
            uint8_t tile = 0, dfmt = 0, nfmt = 0, degraded = 0;
        };
        std::unordered_map<uint64_t, TileCell> tilePlans;
        uint64_t tileLines = 0;
        uint64_t texSrgbAsUnorm = 0, texInexact = 0, texSoftBc = 0;
        // T#s the program samples with compare (the lifted resource is a depth texture): how each was bound and what the host
        // resource it got is (QUAL cmp_*, compare_mismatch). A mismatch is a depth texture expected over a host resource that is not.
        enum CmpRoute : uint8_t { kCmpDepthTarget, kCmpColorTarget, kCmpMemory, kCmpAlias, kCmpOther, kCmpRoutes };
        uint64_t cmpBinds = 0, cmpDraws = 0, cmpUnresolved = 0, cmpHostDepth = 0, cmpHostNotDepth = 0, cmpMismatchBinds = 0, cmpMismatchDraws = 0;
        uint64_t cmpRoute[kCmpRoutes] = {}, cmpFill[5] = {}, cmpLines = 0;
        uint64_t cmpLastDraw[2] = {~uint64_t(0), ~uint64_t(0)}, cmpLastMismatch[2] = {~uint64_t(0), ~uint64_t(0)};  // the call last counted (draw, dispatch)
        struct CmpCell {
            uint64_t count = 0;
            const char* host = "";
            uint8_t dfmt = 0, route = 0;
        };
        std::unordered_map<uint64_t, CmpCell> cmpHist;  // (dfmt, host format, route)
        std::unordered_map<std::string, uint16_t> approxIds;
        std::vector<std::string> approxNames;
        std::vector<uint64_t> approxHits;  // draws / dispatches that use a program translated with that approximation
        uint64_t ringHighWater = 0;
        // Publishing: the totals at the last call, for the per-picture figures.
        std::unordered_map<std::string, uint64_t> lastReason;  // family/name -> its total at the previous publishProbe
        std::set<std::string> published;                         // names of the open-ended families already given a column
        uint64_t lastPictures = 0;
    } pstat;
    // The probe's tables are fed only while it is on, and hold at most kProbeMaxEntries entries each.
    static constexpr size_t kProbeMaxEntries = 512;
    static constexpr uint16_t kNoApprox = 0xFFFF;
    uint16_t approxId(const std::string& n) {
        if (!vpgpu::probe::enabled()) return kNoApprox;
        const auto found = pstat.approxIds.find(n);
        if (found != pstat.approxIds.end()) return found->second;
        if (pstat.approxNames.size() >= kProbeMaxEntries) return kNoApprox;
        const uint16_t id = uint16_t(pstat.approxNames.size());
        pstat.approxIds.emplace(n, id);
        pstat.approxNames.push_back(n), pstat.approxHits.push_back(0);
        return id;
    }

    // ---- Guest memory ---------------------------------------------------------------------------

    struct GuestBuffer {
        uint64_t base = 0, size = 0;
        void* host = nullptr;
        BufferId id;
        // The region the host gave when it was made (the window itself, or the one a smaller piece was cut from).
        uint64_t spanBase = 0, spanSize = 0;
        bool mirrored = false;
        // The backend refused to make it (e.g. the no-copy wrap of a window): not tried again while the host's map is at
        // the generation of the refusal, and for kRefusedFrames frames at most (a refusal may not last).
        bool failed = false, failedLogged = false;
        uint64_t failedGeneration = 0, failedFrame = 0;
    };
    static constexpr uint64_t kRefusedFrames = 120;
    // By (base, size) of the region: windows with the same base and another end (one widened over a boundary) coexist.
    std::map<std::pair<uint64_t, uint64_t>, GuestBuffer> guestBuffers;
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
    // Across draws: the mirrored regions the host gave (a mirror stays the same until the host's map
    // changes), each valid while Hooks::regionGeneration() is the value it had before the host
    // answered. Regions that are not mirrored depend on what is watched at the moment, so the host
    // is asked for them again in every draw.
    struct CachedRegion {
        GuestRegion g;
        uint64_t generation = 0;
        bool valid = false;
    };
    std::array<CachedRegion, 8> cachedRegions;
    unsigned cachedRegionNext = 0;
    bool region(uint64_t address, uint64_t size, GuestRegion& out) {
        if (haveRegion && address >= lastRegion.base && address + size <= lastRegion.base + lastRegion.size) {
            out = lastRegion;
            return true;
        }
        const bool kept = bool(r.hooks_.regionGeneration);
        const uint64_t generation = kept ? r.hooks_.regionGeneration() : 0;
        if (kept) {
            const uint64_t want = size ? size : 1;
            for (const CachedRegion& c : cachedRegions)
                if (c.valid && c.generation == generation && address >= c.g.base && address + want <= c.g.base + c.g.size) {
                    out = c.g;
                    ++r.cpu_.regionHits;
                    if (regionScope) lastRegion = out, haveRegion = true;
                    return true;
                }
        }
        ++r.cpu_.regionMisses;
        if (!r.hooks_.region || !r.hooks_.region(address, size ? size : 1, out)) return false;
        if (!(address >= out.base && address + size <= out.base + out.size)) return false;
        if (regionScope) lastRegion = out, haveRegion = true;
        if (kept && out.mirrored) {
            CachedRegion* slot = nullptr;
            for (CachedRegion& c : cachedRegions)
                if (c.valid && c.g.base == out.base && c.g.size == out.size) slot = &c;
            if (!slot) slot = &cachedRegions[cachedRegionNext++ % cachedRegions.size()];
            *slot = {out, generation, true};
        }
        return true;
    }

    bool readGuest(uint64_t address, void* dst, size_t bytes) {
        GuestRegion g;
        if (!region(address, bytes, g)) return false;
        std::memcpy(dst, static_cast<const uint8_t*>(g.host) + (address - g.base), bytes);
        return true;
    }

    // Whether the host has since given the memory of a mirrored window another mirror (its pages changed: the old mirror
    // retires and is unmapped some seconds later, which would leave the buffer over unmapped pages). Only an answer
    // that covers the window's region and puts it elsewhere says so; no answer (not mapped now) does not.
    bool windowSuperseded(const GuestBuffer& b) {
        GuestRegion now;
        if (!b.mirrored || !r.hooks_.region || !r.hooks_.region(b.spanBase, b.spanSize, now)) return false;
        if (b.spanBase < now.base || b.spanBase + b.spanSize > now.base + now.size) return false;
        return uintptr_t(now.host) + (b.spanBase - now.base) != uintptr_t(b.host) - (b.base - b.spanBase);
    }
    // Every window buffer is given up (Renderer::dropGuestBuffers): destroyed once the GPU is idle, or retired like a
    // replaced one if it does not get there in time. The next use makes new ones from what the host offers then.
    bool dropGuestBuffers() {
        const bool idle = r.device_.waitIdle(kDropWaitNs);
        for (auto& [key, b] : guestBuffers) {
            if (!b.id) continue;
            ++pstat.guestBufRemoves;
            if (idle) r.device_.destroy(b.id);
            else retired.push_back({b.id, r.frame_ + kFrames});
        }
        guestBuffers.clear();
        for (CachedRegion& c : cachedRegions) c.valid = false;
        haveRegion = false;
        return idle;
    }
    static constexpr uint64_t kDropWaitNs = 5'000'000'000ull;
    // Windows that overlap the new one `g` (made at `host`) and were made of another mirror of that memory: gone from the
    // table, their buffers retired like any replaced one.
    void retireSuperseded(const GuestBuffer& fresh) {
        for (auto it = guestBuffers.begin(); it != guestBuffers.end();) {
            const GuestBuffer& o = it->second;
            const bool overlaps = &o != &fresh && o.base < fresh.base + fresh.size && fresh.base < o.base + o.size;
            const bool coherent = uintptr_t(o.host) - uintptr_t(fresh.host) == o.base - fresh.base;  // (the same bytes of host memory)
            if (overlaps && !coherent && windowSuperseded(o)) {
                if (o.id) retired.push_back({o.id, r.frame_ + kFrames}), ++pstat.guestBufRemoves;
                it = guestBuffers.erase(it);
            } else {
                ++it;
            }
        }
    }

    // The gpu/core buffer aliasing region `g` (made once per region and host pointer; `span` is the region of the host it
    // was cut from). False if the backend refused it: that is kept while Hooks::regionGeneration stays the same (without
    // the hook it is tried again each time) and for kRefusedFrames frames, so a window that cannot be wrapped is not made
    // again at every use. Logged once per window.
    bool windowBuffer(const GuestRegion& g, const GuestRegion& span, BufferId& id) {
        GuestBuffer& b = guestBuffers[{g.base, g.size}];
        if (b.id && b.host == g.host) {
            id = b.id;
            return true;
        }
        const bool kept = bool(r.hooks_.regionGeneration);
        const uint64_t generation = kept ? r.hooks_.regionGeneration() : 0;
        if (kept && b.failed && b.host == g.host && b.failedGeneration == generation && r.frame_ < b.failedFrame + kRefusedFrames)
            return false;
        if (b.id) retired.push_back({b.id, r.frame_ + kFrames}), ++pstat.guestBufRemoves;
        if (b.host != g.host) b.failedLogged = false;
        b.base = g.base, b.size = g.size, b.host = g.host;
        b.spanBase = span.base, b.spanSize = span.size, b.mirrored = g.mirrored;
        b.id = r.device_.createBuffer({g.size, g.host, "guest " + hex(g.base)});
        b.failed = !b.id;
        if (b.id) ++pstat.guestBufAdds;
        else ++pstat.guestBufRefused;
        b.failedGeneration = generation;
        b.failedFrame = r.frame_;
        if (b.failed && !b.failedLogged) {
            b.failedLogged = true;
            log("gpu.err what=guest_buffer_refused base=" + hex(g.base) + " size=" + std::to_string(g.size) + " host=" +
                hex(reinterpret_cast<uintptr_t>(g.host)) + " mirrored=" + (g.mirrored ? "1" : "0") + " page_aligned=" +
                (((reinterpret_cast<uintptr_t>(g.host) | g.size) % kPage) == 0 ? "1" : "0"));
        }
        retireSuperseded(b);
        id = b.id;
        return !b.failed;
    }

    // The gpu/core buffer aliasing guest memory from `address` on, and how many bytes of it are there (`available`). It
    // holds [address, address + size) whole whenever the host can map that range: the window around `address` is cut at
    // the host's window boundary, so a range that crosses it asks for a wider one. When a buffer is refused by the backend
    // a smaller window that still holds the range is tried, then the window around `address` alone (`available` tells
    // the caller how much of the range it holds).
    bool guestBuffer(uint64_t address, uint64_t size, BufferId& id, uint64_t& offset, uint64_t& available) {
        GuestRegion g;
        if (!region(address, 1, g)) return false;
        GuestRegion use, wide, part;
        const auto take = [&](const GuestRegion& w, const GuestRegion& span) {
            if (!windowBuffer(w, span, id)) return false;
            use = w;
            return true;
        };
        bool ok = false;
        if (size > 1 && size <= kMaxGuestSpan && address + size > g.base + g.size && region(address, size, wide)) {
            ok = take(wide, wide) || (shrinkRegion(wide, address, size, part) && take(part, wide));
        }
        if (!ok) ok = take(g, g) || (shrinkRegion(g, address, size, part) && take(part, g));
        if (!ok) return false;
        offset = address - use.base;
        available = use.size - offset;
        // (a bare address, no V#: its extent is unknown, so a window is tracked as for a raw access, not the whole mapping)
        touch(address, size && size < available ? size : !size && !touchWindow ? std::min<uint64_t>(available, kUnknownExtent) : available);
        return true;
    }
    // VPKENGINE_TOUCH_WINDOW=1: a buffer of unknown extent counts the whole window around it as busy, as before.
    bool touchWindow = std::getenv("VPKENGINE_TOUCH_WINDOW") != nullptr;

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
        std::map<uint64_t, const char*> who = {};  // probe only: see busyWho
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
    // Probe only: what made each range of `busy` (by its start; a merged range keeps the first one's), `touchWho` being
    // what is recording now (see Who). DMASYNC names it.
    std::map<uint64_t, const char*> busyWho;
    const char* touchWho = "draw";
    struct Who {
        const char*& slot;
        const char* prev;
        Who(const char*& s, const char* w) : slot(s), prev(s) { slot = w; }
        ~Who() { slot = prev; }
    };
    void touch(uint64_t a, uint64_t n) {
        if (busyAll || !n) return;
        static vpgpu::probe::Timer& touchTimer = vpgpu::probe::timer("TIME", "hazard_touch");
        const vpgpu::probe::ScopedSampled touchScope(touchTimer);  // (hot: calls exact, time estimated from 1 in 16)
        uint64_t lo = a, hi = a + n < a ? ~uint64_t(0) : a + n;
        if (lo >= touchedLo && hi <= touchedHi) return;
        const bool probing = vpgpu::probe::enabled();
        const char* who = touchWho;
        auto it = busy.upper_bound(lo);
        if (it != busy.begin()) {
            auto prev = std::prev(it);
            if (prev->second >= lo) {
                if (prev->second >= hi) {  // already covered
                    touchedLo = prev->first, touchedHi = prev->second;
                    return;
                }
                lo = prev->first;
                if (probing)
                    if (auto w = busyWho.find(lo); w != busyWho.end()) who = w->second;
                it = busy.erase(prev);
            }
        }
        while (it != busy.end() && it->first <= hi) {
            hi = std::max(hi, it->second);
            if (probing) busyWho.erase(it->first);
            it = busy.erase(it);
        }
        busy[lo] = hi;
        if (probing) busyWho[lo] = who;
        touchedLo = lo, touchedHi = hi;
        if (busy.size() > kBusyRanges) coalesceBusy();
    }
    // Too many ranges: the closest neighbours merge (a quarter of them) instead of everything becoming busy, so
    // a CPU copy elsewhere in memory still does not wait for the GPU.
    void coalesceBusy() {
        std::vector<uint64_t> gaps;
        gaps.reserve(busy.size());
        for (auto it = busy.begin(), next = std::next(it); next != busy.end(); ++it, ++next) gaps.push_back(next->first - it->second);
        const size_t drop = busy.size() / 4;
        std::nth_element(gaps.begin(), gaps.begin() + std::ptrdiff_t(drop), gaps.end());
        const uint64_t limit = gaps[drop];
        for (auto it = busy.begin(); it != busy.end();) {
            auto next = std::next(it);
            if (next != busy.end() && next->first - it->second <= limit) {
                it->second = std::max(it->second, next->second);
                busyWho.erase(next->first);
                busy.erase(next);
            } else {
                ++it;
            }
        }
        touchedLo = touchedHi = 0;
    }
    bool busyOverlaps(uint64_t a, uint64_t n) const {
        if (busyAll) return true;
        const uint64_t end = a + std::max<uint64_t>(n, 1);
        auto it = busy.upper_bound(a);
        if (it != busy.begin() && std::prev(it)->second > a) return true;
        return it != busy.end() && it->first < end;
    }

    // Probe DMASYNC: the first busy range a CPU access to [a, +n) (and [b, +m)) overlaps, "lo-hi by=<who> in=<recorded|frame N>"
    // (who made it: see busyWho); "" if none.
    static bool firstOverlap(const std::map<uint64_t, uint64_t>& m, uint64_t a, uint64_t n, uint64_t& lo, uint64_t& hi) {
        const uint64_t end = a + std::max<uint64_t>(n, 1);
        auto it = m.upper_bound(a);
        if (it != m.begin() && std::prev(it)->second > a) --it;
        if (it == m.end() || it->first >= end || it->second <= a) return false;
        lo = it->first, hi = it->second;
        return true;
    }
    std::string busyText(uint64_t a, uint64_t n, uint64_t b, uint64_t m) const {
        for (const auto& [x, y] : {std::pair{a, n}, std::pair{b, m}}) {
            if (!y) continue;
            uint64_t lo = 0, hi = 0;
            const auto text = [&](const std::map<uint64_t, const char*>& who, const std::string& in) {
                const auto w = who.find(lo);
                return hex(lo) + "-" + hex(hi) + " by=" + (w == who.end() ? "?" : w->second) + " in=" + in;
            };
            if (busyAll) return "all";
            if (firstOverlap(busy, x, y, lo, hi)) return text(busyWho, "recorded");
            for (const InFlight& f : inFlight) {
                if (f.all) return "all in=frame " + std::to_string(f.frame);
                if (firstOverlap(f.ranges, x, y, lo, hi)) return text(f.who, "frame " + std::to_string(f.frame));
            }
        }
        return "";
    }
    // Probe DMASYNC: one CPU access (`caller`) to [a, +n) (and [b, +m)) that waited for `frames` frames (0: it did not wait)
    // for `ms`. Counters and a timer per caller (family DMASYNC); the first 200 waits, and then every 50th or one over 2 ms,
    // leave a line with the busy range that made it wait and who made that.
    uint64_t dmaSyncLines = 0;
    void noteCpuSync(const char* caller, uint64_t a, uint64_t n, uint64_t b, uint64_t m, const std::string& busyRange, uint64_t frames,
                     bool submitted, double ms) {
        namespace pr = vpgpu::probe;
        if (!pr::enabled()) return;
        const std::string c = pr::Registry::sanitize(caller[0] ? caller : "other");
        if (!frames) {
            pr::counter("DMASYNC", c + "_free").add();
            return;
        }
        pr::counter("DMASYNC", c + "_waits").add();
        pr::counter("DMASYNC", c + "_bytes").add(n + m);
        pr::timer("DMASYNC", c + "_wait").addMs(ms);
        const uint64_t k = ++dmaSyncLines;
        if (k > 200 && k % 50 && ms < 2.0) return;
        log("VPKENGINE_PROBE DMASYNC_WAIT n=" + std::to_string(k) + " f=" + std::to_string(r.frame_) + " why=" + c + " a=" + hex(a) +
            " bytes=" + std::to_string(n) + (m ? " b=" + hex(b) + " bytes_b=" + std::to_string(m) : std::string()) + " frames=" +
            std::to_string(frames) + " submitted_first=" + (submitted ? "1" : "0") + " busy=[" + busyRange + "] ms=" + pr::Registry::number(ms));
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
        uint64_t lastZpassFrame = 0;  // submitted-frame count at the latest ZPASS_DONE
        static constexpr uint64_t kIdleFrames = 8;  // frames without one before draws stop counting
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
        if (guestReadOnly || !pairs || !address || snapshots.size() + 2 >= kVisSlots || !ensureVis()) return false;
        GuestRegion reg;
        const uint64_t bytes = (pairs - 1) * 16ull + 8;
        if (!r.hooks_.region || !r.hooks_.region(address, bytes, reg) || !reg.host) return false;
        vis.active = true;
        vis.lastZpassFrame = r.device_.submittedFrames();
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
            // Memory an image is ahead of is written back before the results land in it (an image
            // that the results cover completely is simply overwritten); the copy is recorded, so a
            // partly covered image's older bytes can still reach the range when the GPU runs it.
            if (tracker.ownerImage(s.address, bytes) != 0 && syncOwners(s.address, bytes, true))
                note("occlusion results written over memory an image was ahead of");
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
        uint64_t head = 0, cap = kRingBytes;
    };
    std::array<Ring, kFrames> rings{};
    unsigned ringIndex = 0;
    // The rings are sized by what the previous frame used (steady state, at most kRingSteadyMax); a frame that
    // runs out chains a further segment (chainRing) instead of skipping draws, up to kRingMax in all. A ring is
    // resized when it comes round again.
    static constexpr uint64_t kRingMax = 512ull << 20;
    static constexpr uint64_t kRingSteadyMax = 128ull << 20;
    static constexpr uint64_t kRingHeadroom = 2ull << 20;  // a draw starts with this much room or a new segment
    uint64_t ringWant = kRingBytes;
    uint64_t ringUsed = 0;  // bytes taken from this frame's earlier segments
    uint64_t ringNeed = 0;  // what the frame wanted from the ring when it ran out (a refused request counts)
    bool ringBudgetTold = false;
    void sizeRing(Ring& ring) {
        if (ring.cap >= ringWant && ring.cap <= 2 * ringWant) return;
        if (ring.memory) {  // (as a staging buffer: kept until the GPU is done with it)
            stagings.push_back({ring.memory, ring.id, r.frame_ + kFrames + 2});
            ring.memory = nullptr;
            ring.id = {};
        }
        ring.cap = ringWant;
    }
    // The current ring is full: the frame goes on in a new segment of at least `need` bytes. The old segment
    // stays alive as a staging buffer (the draws already recorded read it). False when the frame's budget is spent.
    bool chainRing(uint64_t need) {
        Ring& ring = rings[ringIndex];
        const uint64_t spent = ringUsed + ring.head;
        ringNeed = std::max(ringNeed, spent + need);
        const uint64_t want = (need + (1ull << 20) - 1) / (1ull << 20) * (1ull << 20);
        uint64_t cap = std::max<uint64_t>(ring.cap, want);
        if (spent + cap > kRingMax) cap = (kRingMax > spent ? kRingMax - spent : 0) / kPage * kPage;
        if (cap < need) {
            if (!ringBudgetTold) {
                ringBudgetTold = true;
                log("vpps4.ring_budget_hit used=" + std::to_string(spent >> 20) + "MB need=" + std::to_string(need) + "B");
            }
            return false;
        }
        void* memory = std::aligned_alloc(kPage, cap);
        if (!memory) return false;
        const BufferId id = r.device_.createBuffer({cap, memory, "vpps4 upload " + std::to_string(ringIndex) + "+"});
        if (!id) {
            std::free(memory);
            return false;
        }
        if (ring.memory) stagings.push_back({ring.memory, ring.id, r.frame_ + kFrames + 2});
        ringUsed += ring.head;
        ring = {memory, id, 0, cap};
        ++ringGeneration;
        ++r.stats_.ringChains;
        return true;
    }
    void ringRoom() {
        const Ring& ring = rings[ringIndex];
        if (ring.memory && ring.cap - ring.head < kRingHeadroom) chainRing(kRingHeadroom);
    }

    bool ensureRing() {
        Ring& ring = rings[ringIndex];
        if (!ring.memory) {
            ring.memory = std::aligned_alloc(kPage, ring.cap);
            if (!ring.memory) return false;
            ring.id = r.device_.createBuffer({ring.cap, ring.memory, "vpps4 upload " + std::to_string(ringIndex)});
        }
        return true;
    }

    uint64_t ringGeneration = 0;  // bumped when the current ring changes (uploads before it are gone)

    // User data (16 dwords per stage) repeats across draws: an identical copy uploaded earlier into
    // the current ring is bound again instead of a new one. A copy is 64 bytes, packed at 16 (vector
    // loads): 128 recent ones are remembered, compared by hash first.
    struct UserDataCopy {
        std::array<uint32_t, 16> data;
        uint64_t hash = 0, at = 0, generation = ~uint64_t(0);
    };
    std::array<UserDataCopy, 128> userDataCopies;
    unsigned userDataNext = 0;
    std::optional<uint64_t> uploadUserData(const std::array<uint32_t, 16>& d) {
        uint64_t h = 0xcbf29ce484222325ull;
        for (uint32_t w : d) h = (h ^ w) * 0x100000001b3ull;
        for (const UserDataCopy& c : userDataCopies)
            if (c.hash == h && c.generation == ringGeneration && c.data == d) {
                ++r.cpu_.userDataReused;
                return c.at;
            }
        auto at = upload(d.data(), sizeof d, 16);
        if (!at) return at;
        UserDataCopy& c = userDataCopies[userDataNext++ % userDataCopies.size()];
        c.data = d;
        c.hash = h;
        c.at = *at;
        c.generation = ringGeneration;
        return at;
    }

    // Constant buffers at an address that is not 16-byte aligned are copied into the ring (vector loads
    // need alignment). The copy is kept with the tracker's version of the source: the same buffer bound
    // by the next draws of the submit (no write to it in between, by the GPU or the CPU) reuses it. Copies are found by
    // source address, size and version (a write to the source gives a new key); they are forgotten when the ring
    // changes (their bytes are in another segment) and past kAlignedCopies of them in one ring.
    struct AlignedKey {
        uint64_t address, bytes, version;
        bool operator==(const AlignedKey&) const = default;
    };
    struct AlignedKeyHash {
        size_t operator()(const AlignedKey& k) const {
            uint64_t h = k.address * 0x9e3779b97f4a7c15ull;
            h = (h ^ (h >> 32) ^ k.bytes) * 0x100000001b3ull;
            h = (h ^ (h >> 32) ^ k.version) * 0x9e3779b97f4a7c15ull;
            return size_t(h ^ (h >> 32));
        }
    };
    static constexpr size_t kAlignedCopies = 4096;
    std::unordered_map<AlignedKey, uint64_t, AlignedKeyHash> alignedCopies;  // -> offset of the copy in the current ring
    uint64_t alignedGeneration = ~uint64_t(0);                               // the ringGeneration they are valid for
    uint64_t alignedCopyBytes = 0, alignedCopyBytesLogged = 0;                // bytes copied (probe P7: since its last line)
    void forgetAlignedCopies() {
        alignedCopies.clear();
        alignedGeneration = ringGeneration;
    }
    std::optional<uint64_t> alignedConstantCopy(uint64_t address, uint64_t bytes, const void* host) {
        if (alignedGeneration != ringGeneration || alignedCopies.size() >= kAlignedCopies) forgetAlignedCopies();
        const AlignedKey key{address, bytes, tracker.version(address, bytes)};
        const auto it = alignedCopies.find(key);
        // (the CPU may have rewritten the source without the tracker hearing of it: the bytes decide)
        if (it != alignedCopies.end() && std::memcmp(static_cast<const uint8_t*>(rings[ringIndex].memory) + it->second, host, bytes) == 0) {
            ++r.cpu_.alignedCopyReused;
            return it->second;
        }
        auto at = upload(host, bytes, 16);
        if (!at) return at;
        alignedCopyBytes += bytes;
        ++pstat.alignedCopies;
        if (alignedGeneration != ringGeneration) forgetAlignedCopies();  // (the upload chained the ring: the copy is in the new segment)
        alignedCopies[key] = *at;
        return at;
    }

    // Room for `bytes` in the current segment; when it does not fit, a new segment of at least that size is chained
    // (a request bigger than a segment gets a segment of its own) and the request goes there. Whoever takes more than
    // one piece for one draw and binds them through a single buffer id must check ringGeneration did not change.
    std::optional<uint64_t> take(uint64_t bytes, uint64_t align) {
        if (!ensureRing()) return std::nullopt;
        for (int attempt = 0; attempt < 2; ++attempt) {
            Ring& ring = rings[ringIndex];
            const uint64_t at = (ring.head + align - 1) / align * align;
            if (at + bytes <= ring.cap) {
                ring.head = at + bytes;
                return at;
            }
            if (!chainRing(bytes + align)) break;
        }
        return std::nullopt;
    }

    std::optional<uint64_t> upload(const void* data, uint64_t bytes, uint64_t align = 256) {
        const auto at = take(bytes, align);
        if (!at) return at;
        std::memcpy(static_cast<uint8_t*>(rings[ringIndex].memory) + *at, data, bytes);
        ++r.stats_.uploads;
        pstat.uploadBytes += bytes;
        return at;
    }

    // Room in the upload ring the GPU fills (not copied from the CPU).
    std::optional<uint64_t> reserve(uint64_t bytes, uint64_t align) { return take(bytes, align); }

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
        vpmsl::MslShader m;
        m.source = convSource();
        m.workgroupSize = {64, 1, 1};
        using K = vpir::ResourceKind;
        m.arguments = {{"src", K::Buffer, 0, 0},    {"dstArgs", K::Buffer, 1, 8}, {"srcIdx", K::Buffer, 2, 16},
                       {"dstIdx", K::Buffer, 3, 24}, {"cnt", K::Buffer, 4, 32},    {"prm", K::ConstantBuffer, 0, 40}};
        convId = vpmsl::shaderId(m);
        if (r.hooks_.shader) r.hooks_.shader(convId, vpir::Stage::Compute, m);
        return true;
    }

    // Records the converter for draw `d` (arguments at guest buffer `argBuf`+`argOff`) and points
    // `draw` at its output: `indirect` is always set; with `conv` the index buffer and count too.
    bool convertIndirect(const DrawState& d, bool conv, BufferId argBuf, uint64_t argOff, vpgpu::Indirect& indirect,
                         vpgpu::Draw& draw, std::string& why) {
        ensureConv();
        vpgpu::PipelineState ps;
        ps.computeShader = convId;
        ps.label = "indirect draw converter";
        const vpgpu::PipelineCache::Status st = r.device_.pipelines().request(ps, vpgpu::Priority::Visible);
        if (st == vpgpu::PipelineCache::Status::Failed) return why = "indirect converter failed", false;
        if (st != vpgpu::PipelineCache::Status::Ready) return why = "indirect converter compiling", false;

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
        const uint32_t prim = !conv ? 0 : (d.primitive == PrimitiveType::TriangleFan || d.primitive == PrimitiveType::Polygon) ? 1
                              : d.primitive == PrimitiveType::QuadList ? 2
                              : d.primitive == PrimitiveType::QuadStrip ? 3 : 4;
        const uint32_t prm[8] = {prim, indexed ? 1u : 0u, d.index32 ? 1u : 0u, cap, d.drawIndex, d.indirectCountAddress ? 1u : 0u, avail, 0};
        // (one buffer holds the three: if the ring chained between them, they are taken again in the new segment, once)
        decltype(reserve(0, 0)) outArgs, outIdx, prmAt;
        bool together = false;
        for (int attempt = 0; attempt < 2 && !together; ++attempt) {
            const uint64_t ringGen = ringGeneration;
            outArgs = reserve(32, 16);
            outIdx = reserve(conv ? uint64_t(cap) * 4 : 16, 16);
            prmAt = upload(prm, sizeof prm, 16);
            together = outArgs && outIdx && prmAt && ringGeneration == ringGen;
        }
        if (!together) return why = "upload ring full", false;
        const BufferId ring = rings[ringIndex].id;

        vpgpu::Dispatch x;
        auto buf = [&](uint16_t slot, BufferId b, uint64_t off, uint64_t bytes, bool write) {
            Access a;
            a.range = ResourceRange::bytes(b, off, bytes);
            a.stages = vpgpu::stage::compute;
            a.kind = vpgpu::BindKind::Buffer;
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
        c.stages = vpgpu::stage::compute;
        c.kind = vpgpu::BindKind::ConstantBuffer;
        c.slot = 0;
        x.bindings.push_back(c);
        x.pipeline = ps.key();
        x.groups = {conv ? cap / 64 : 1u, 1, 1};
        x.label = ps.label;
        closePass();
        frame.dispatch(std::move(x));
        ++r.stats_.dispatched;

        indirect = vpgpu::Indirect{ring, *outArgs};
        if (conv) {
            draw.index = vpgpu::IndexBuffer{ring, *outIdx, true};
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
    // The record of the call in progress when the host asked for them (Hooks::resolved); null otherwise.
    Resolved* resolved_ = nullptr;
    void resolvedStatus(Resolved::Status status, const char* why) {
        if (resolved_) {
            resolved_->status = status;
            resolved_->reason = why;
        }
    }
    void skip(const std::string& reason) {
        resolvedStatus(Resolved::Status::Skipped, reason.c_str());
        vpps4_report("draw-skip", reason.c_str(), "");
        if (r.stats_.skipped[reason]++ == 0) log("vpps4.first skip=\"" + reason + "\"");
    }
    void note(const std::string& n) {
        vpps4_report(noteCategory(n.c_str()), n.c_str(), "");
        if (r.stats_.notes[n]++ == 0) log("vpps4.first note=\"" + n + "\"");
    }
    // Literal reasons (the per-draw ones): counted through the literal's address, no string built.
    // A draw that was done as something other than a draw of a pass (a copy, a clear): not skipped, not "drawn".
    void handled(const char* what) {
        resolvedStatus(Resolved::Status::Handled, what);
        if (r.stats_.handled[what]++ == 0) log(std::string("vpps4.first handled=\"") + what + "\"");
        vpps4_report("draw-handled", what, "");
    }
    void skip(const char* reason) {
        resolvedStatus(Resolved::Status::Skipped, reason);
        countLit(skipLits, r.stats_.skipped, reason, "draw-skip", "skip");
    }
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

    // Marks the textures the module reads only with integer coordinates (see vpir::Resource::integerOnly).
    static void markIntegerOnly(vpir::Module& m) {
        std::vector<uint8_t> fetched(m.resources.size(), 0), sampled(m.resources.size(), 0);
        auto scan = [&](auto&& self, const vpir::Block& body) -> void {
            for (const vpir::Node& n : body) {
                if (n.kind == vpir::NodeKind::Inst) {
                    const vpir::Inst& i = n.inst;
                    if (i.sym >= m.resources.size()) continue;
                    switch (i.op) {
                        case vpir::Op::ImageFetch:
                        case vpir::Op::ImageSize:
                        case vpir::Op::ImageLevels:
                        case vpir::Op::ImageSamples: fetched[i.sym] = 1; break;
                        case vpir::Op::ImageSample:
                        case vpir::Op::ImageGather:
                        case vpir::Op::ImageQueryLod: sampled[i.sym] = 1; break;
                        default: break;
                    }
                }
                self(self, n.body);
                self(self, n.elseBody);
            }
        };
        for (const vpir::Function& f : m.functions) scan(scan, f.body);
        for (size_t i = 0; i < m.resources.size(); ++i)
            m.resources[i].integerOnly = m.resources[i].kind == vpir::ResourceKind::Texture && fetched[i] && !sampled[i];
    }
    struct Translated {
        bool ok = false;
        std::string error;
        uint64_t id = 0;
        std::vector<vpgcn::Binding> bindings;
        std::vector<vpir::Resource> resources;  // Module::resources (kind index, access)
        std::optional<uint32_t> userDataSlot;
        std::vector<int16_t> snapIndex;  // per binding: its place in the draw's descriptor snapshot, or -1
        bool fromCache = false;          // loaded from the persistent cache
        bool touched = false;            // ... and the cache was told this run used it
        std::vector<uint16_t> approx;    // the approximations its translation took (ProbeStats::approxNames); unknown for one loaded from the cache
    };
    void countApprox(const Translated* t) {
        if (t && !t->approx.empty())
            for (uint16_t a : t->approx) ++pstat.approxHits[a];
    }
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
            const bool tex = b.kind == vpir::ResourceKind::Texture || b.kind == vpir::ResourceKind::StorageTexture;
            if (!b.dwords || !(tex || b.kind == vpir::ResourceKind::Buffer)) continue;
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

    static vpir::Stage irStage(vpgcn::ShaderStage stage) {
        return stage == vpgcn::ShaderStage::Vertex || stage == vpgcn::ShaderStage::CopyVertex ? vpir::Stage::Vertex
             : stage == vpgcn::ShaderStage::Pixel ? vpir::Stage::Fragment
                                                   : vpir::Stage::Compute;
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
        // (Only those whose shaders this run has: shader ids are content hashes, and a shader the translation
        // cache lost leaves its pipelines stale - built on demand if the title draws them, never counted as failed.)
        for (unsigned n = 0; n < 256 && mergedPipelines < pendingLoad.pipelines.size(); ++mergedPipelines, ++n) {
            const vpgpu::PipelineState& ps = pendingLoad.pipelines[mergedPipelines];
            const auto has = [&](uint64_t id) { return !id || registered.count(id) != 0; };
            if (has(ps.vertexShader) && has(ps.fragmentShader) && has(ps.computeShader))
                r.device_.pipelines().request(ps, vpgpu::Priority::Prefetch);
        }
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
        uint64_t content = 0;  // sampleRange of the code and the fetch words when the key was made
    };
    std::array<ProgramMemo, 64> programMemo;
    uint64_t programGeneration = 0;
    uint64_t programKeyCached(const Stage& s, vpgcn::ShaderStage stage) {
        const uint64_t rest = programKey(s, stage, false);
        // A program the title rewrites in place without the CPU write being reported (invalidate) must not keep an old
        // key: the content is looked at too, as sampleRange does (the whole of a program up to 4 KiB, else its ends and
        // 16 lines of 64 bytes spread over it; a few hundred ns, against the kilobytes programKey hashes).
        const uint64_t content =
            mix(sampleRange(reinterpret_cast<const uint8_t*>(s.code.data()), s.code.size() * 4),
                sampleRange(reinterpret_cast<const uint8_t*>(s.fetch.data()), s.fetch.size() * 4));
        ProgramMemo& m = programMemo[(reinterpret_cast<uintptr_t>(s.code.data()) >> 4 ^ uintptr_t(stage)) % programMemo.size()];
        if (m.generation == programGeneration && m.code == s.code.data() && m.codeWords == s.code.size() &&
            m.fetch == s.fetch.data() && m.fetchWords == s.fetch.size() && m.rest == rest && m.content == content)
            return m.key;
        m = {s.code.data(), s.fetch.data(), s.code.size(), s.fetch.size(), rest, programKey(s, stage), programGeneration, content};
        return m.key;
    }

    // The sample layout an FMASK T# describes: the multisampled colour surface it belongs to. Its base
    // address is the FMASK's own, so the surface is the multisampled colour target at that address if
    // there is one, else the most recently drawn one of the same size; before any pass: the global layout
    // of the latest multisampled pass.
    void fmaskLayout(const ImageDesc& d, uint8_t& cov, uint8_t& frag) const {
        const Target* best = nullptr;
        for (const auto& [address, list] : colorTargets)
            for (const Target& t : list) {
                if (!t.ms || !t.msCov) continue;
                const bool here = address == d.address;
                const bool sameSize = t.width == d.width && t.height == d.height;
                if (here || sameSize) {
                    if (!best || (here && best->address != d.address) || (here == (best->address == d.address) && t.lastUse > best->lastUse))
                        best = &t;
                }
            }
        cov = best ? best->msCov : msCoverage;
        frag = best ? best->msFrag : msFragments;
    }

    // Strips draw with restart off (or a restart index other than all ones) are scanned for data the host
    // would misread; when the scan found nothing to change, that is remembered per (address, size, restart
    // setting) and the tracker version of the range, so a static index buffer is scanned once.
    struct IndexScan {
        uint64_t address = 0, bytes = 0, version = 0;
        uint32_t restart = 0;
        bool index32 = false, restartOn = false, valid = false;
    };
    std::array<IndexScan, 64> indexScans;
    IndexScan& indexScanSlot(const DrawState& d, uint64_t bytes) {
        return indexScans[((d.indexAddress >> 1) ^ (bytes * 0x9e3779b1ull)) % indexScans.size()];
    }
    bool indexScanMatches(const IndexScan& e, const DrawState& d, uint64_t bytes) const {
        return e.valid && e.address == d.indexAddress && e.bytes == bytes && e.index32 == d.index32 &&
               e.restartOn == d.raster.primitiveRestart && e.restart == d.raster.restartIndex;
    }
    bool indicesKnownClean(const DrawState& d) {
        if (d.indexSwap) return false;
        const uint64_t bytes = uint64_t(d.count) * (d.index32 ? 4 : 2);
        const IndexScan& e = indexScanSlot(d, bytes);
        return indexScanMatches(e, d, bytes) && e.version == tracker.version(d.indexAddress, bytes);
    }
    void noteIndicesClean(const DrawState& d, bool clean) {
        const uint64_t bytes = uint64_t(d.count) * (d.index32 ? 4 : 2);
        IndexScan& e = indexScanSlot(d, bytes);
        if (!clean) {
            if (indexScanMatches(e, d, bytes)) e.valid = false;
            return;
        }
        e = {d.indexAddress, bytes, tracker.version(d.indexAddress, bytes), d.raster.restartIndex, d.index32,
             d.raster.primitiveRestart, true};
    }

    // What a descriptor's dwords say about the slot (the lifter's DescriptorInfo). The answer depends on
    // the descriptor, on the render targets that exist (planFor) and on the pass's sample counts, so it
    // is memoised on those: direct mapped, valid for one texGen.
    struct InfoMemo {
        std::array<uint32_t, 8> w{};
        uint8_t kind = 0, dwords = 0, cov = 0, frag = 0;
        bool needsFormat = false;
        uint64_t gen = ~0ull;
        bool ok = false;
        vpgcn::DescriptorInfo info;
    };
    std::array<InfoMemo, 128> infoMemo;
    bool info_of(const vpgcn::Binding& b, const uint32_t* w, vpgcn::DescriptorInfo& info) {
        const bool image = b.kind == vpir::ResourceKind::Texture || b.kind == vpir::ResourceKind::StorageTexture;
        if (!image && b.kind != vpir::ResourceKind::Buffer) return false;
        if (b.dwords > 8) return false;
        InfoMemo m;
        m.w.fill(0);
        std::copy(w, w + b.dwords, m.w.begin());
        m.kind = uint8_t(b.kind), m.dwords = uint8_t(b.dwords), m.needsFormat = b.needsFormat;
        ImageDesc imageDesc;
        if (image) {
            imageDesc = decodeImage({w, b.dwords});
            // (only an FMASK read depends on the sample layout: that of the surface it describes)
            if (isFmaskFormat(imageDesc.dataFormat)) fmaskLayout(imageDesc, m.cov, m.frag);
        }
        uint64_t hk = texGen * 0x9e3779b97f4a7c15ull ^ (uint64_t(m.kind) << 8 | m.dwords) ^ (uint64_t(m.cov) << 20) ^ (uint64_t(m.frag) << 28) ^ (uint64_t(m.needsFormat) << 36);
        for (uint32_t x : m.w) hk = (hk ^ x) * 0x100000001b3ull;
        InfoMemo& slot = infoMemo[(hk ^ hk >> 29) % infoMemo.size()];
        if (slot.gen == texGen && slot.w == m.w && slot.kind == m.kind && slot.dwords == m.dwords && slot.cov == m.cov &&
            slot.frag == m.frag && slot.needsFormat == m.needsFormat) {
            info = slot.info;
            return slot.ok;
        }
        vpgcn::DescriptorInfo out;
        bool ok = true;
        if (image) {
            const ImageDesc& d = imageDesc;
            bool depthTarget = false;
            const TexturePlan plan = planFor(d, b.kind == vpir::ResourceKind::StorageTexture, depthTarget);
            out.dim = texDim(d);
            out.elem = texElem(plan);
            out.depth = depthTarget;
            out.conv = plan.conv;
            composeSelect(d.dstSel, plan.conv.select, out.conv.select);
            if (plan.conv.manual) out.dfmt = d.dataFormat, out.nfmt = d.numFormat;
            if (isFmaskFormat(d.dataFormat)) {
                // FMASK: the host keeps its samples uncompressed; reads answer with the identity map.
                const FmaskValue v = fmaskIdentity(m.cov, m.frag);
                out.fmask = true;
                out.fmaskLo = v.lo;
                out.fmaskHi = v.hi;
            } else {
                out.multisample = readsSamples(d);
            }
            if (b.needsFormat) {
                // (channels of equal width only; the packed and compressed formats leave it 0)
                static const uint8_t kChannelBits[16] = {0, 8, 16, 8, 32, 16, 0, 0, 0, 0, 8, 32, 16, 32, 32, 0};
                out.texBits = d.dataFormat < 16 ? kChannelBits[d.dataFormat] : 0;
                out.texNumFormat = d.numFormat;
            }
        } else {
            const BufferDesc d = decodeBuffer({w, b.dwords});
            out.dfmt = d.dataFormat;
            out.nfmt = d.numFormat;
            out.layoutKnown = true;
            out.swizzled = d.swizzle && d.stride != 0;
            out.addTid = d.addTid;
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
            static vpgpu::probe::Timer& firstLiftTimer = vpgpu::probe::timer("TIME", "translate_first_lift");
            const vpgpu::probe::Scoped firstLiftScope(firstLiftTimer);
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
            h.value(info.texBits);
            h.value(info.texNumFormat);
            h.value(info.layoutKnown);
            h.value(info.swizzled);
            h.value(info.addTid);
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

        static vpgpu::probe::Timer& translateTimer = vpgpu::probe::timer("TIME", "translate");
        const vpgpu::probe::Scoped translateScope(translateTimer);
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
        markIntegerOnly(lifted.module);
        Translated t;
        for (const std::string& n : lifted.notes) {
            note("shader: " + n);
            if (const uint16_t id = approxId(n); id != kNoApprox) t.approx.push_back(id);
        }
        if (!lifted.errors.empty()) {
            t.error = lifted.errors.front();
        } else {
            vpmsl::EmitOptions emit;
            emit.constTables = false;  // as c7737aa: constant tables have no GCN coverage
            vpmsl::EmitResult msl = vpmsl::emitMsl(lifted.module, emit);
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
        vpgpu::Format format = vpgpu::Format::Undefined;
        // Block-compressed texture on a host without BC (TexturePlan::softBc): `format` is the decoded host format,
        // `layout` the block format the guest memory holds (what the guest range and the detile follow).
        vpgpu::Format layout = vpgpu::Format::Undefined;
        uint8_t softBc = 0;
        bool softSigned = false;
        uint32_t bcLevels = 0;  // soft BC: the levels (bit per level) whose decoded texels are in the image
        bool writable = false;  // `id` has storage usage (a compute pass wrote it)
        vpgpu::TextureType type = vpgpu::TextureType::Tex2D;
        uint32_t width = 0, height = 0, depth = 1;
        uint16_t layers = 1;
        uint8_t samples = 1;
        uint8_t tile = 8;    // TILE_MODE_INDEX
        uint8_t dfmt = 0, nfmt = 0;  // T# DATA_FORMAT / NUM_FORMAT (textures; diagnostics)
        uint32_t pitch = 0;  // texels per row as allocated (0: width)
        bool isDepth = false;
        uint16_t mips = 1;       // levels in the guest layout (textures)
        bool pow2Pad = false;    // T# POW2_PAD: arrays padded to a power of two slices in memory
        uint8_t memSamples = 1;  // samples per texel as laid out in memory (MSAA textures)
        uint64_t address = 0, bytes = 0;  // guest range it is a copy of (mip 0, all slices)
        uint64_t version = 0;             // tracker version of the memory contents it holds
        uint64_t writeSeq = 0;            // counts the writes and fills recorded into the image (any plane: copies of it compare this)
        uint64_t lastUse = 0;
        bool filled = false;         // filled from memory at least once (targets: always)
        // How the last fill from memory went (diagnostics, see probeSampled): 0 none, 1 GPU detile, 2 CPU detile,
        // 3 linear upload, 4 not filled (fillWhy says why).
        uint8_t fillPath = 0;
        const char* fillWhy = "";
        uint32_t fills = 0;  // fills from memory attempted
        bool sourceWatched = false;  // the host catches CPU writes to the source: no hashing
        bool ownedWatched = false;   // the host catches CPU access to memory this image is ahead of
        // Multisampled targets: the sample layout of the last pass that drew into it (what an FMASK read
        // of this surface describes): coverage samples the title asked for and fragments the host stores.
        uint8_t msCov = 0, msFrag = 0;
        // Multisampled colour targets: the frame (r.frame_) in which a resolve into another target replaced the
        // resolve into the single-sample picture (layers staleFirst..+staleCount); ~0 = up to date.
        uint64_t twinStale = ~uint64_t(0);
        uint16_t staleFirst = 0, staleCount = 1;
        // Depth targets: the resolve mode of the last pass that drew into it (applied when the
        // single-sample companion is made after the fact).
        vpgpu::ResolveMode depthMode = vpgpu::ResolveMode::Sample0;
        // Depth targets: DB_Z_INFO.FORMAT of the draw that last wrote it (0: unknown, or stencil only) and the base and
        // size of its stencil plane (kept apart from Z in guest memory; 0: none known). They say how the guest holds it.
        // `followed`: the plane at `address` (Z; the stencil for a stencil-only image) is at one base for the draw's reads
        // and writes (DB_Z_READ_BASE = DB_Z_WRITE_BASE or no write base): a draw that reads one surface and writes another
        // is not followed. A plane is written back only if it is followed and authoritative (see planesOf).
        uint8_t zFormat = 0;
        bool followed = false;
        uint64_t stencilAddress = 0, stencilBytes = 0;
        // Depth targets: the image holds the whole plane (a clear of all of it, or a fill from memory), so writing it over
        // memory loses nothing. False when made: an image that was only drawn into holds texels nothing defined.
        bool zAuthoritative = false, stencilAuthoritative = false;
        uint64_t lastDraw = ~uint64_t(0);  // colour targets: the frame (r.frame_) a pass last drew into it (~0: never; diagnostics)
        uint64_t draws = 0, drawsTaken = 0;  // draws into it, and those already handed to Renderer::takeTargetDraws
        int32_t lastScissor[4] = {};         // x, y, width, height / viewport of the last colour pass that drew into it (probe REFILL only)
        float lastViewport[4] = {};
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
        bool seeded = false;   // holds a seed of the host, not the source: nothing is looked at in memory until a fill of it
        bool storage = false;  // created writable
        // A copy of the corner of a render target (see mirrorTarget): the texture it was copied from and the target's
        // write version at the copy.
        TextureId mirrorSrc;
        uint64_t mirrorVersion = ~uint64_t(0);
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
    static uint64_t surfaceBytes(uint8_t tile, vpgpu::Format f, uint32_t w, uint32_t h, uint32_t pitch, uint32_t slices) {
        const vpgpu::FormatInfo& fi = vpgpu::formatInfo(f);
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
    static uint64_t slicesOf(const Surface& s) { return s.type == vpgpu::TextureType::Tex3D ? s.depth : s.layers; }
    // GPU memory behind an image (what the byte budget counts): the whole mip chain of each host texture it holds.
    // `ms` has the samples; `id` is one sample (alone, the image itself), the resolved picture next to `ms` (colour
    // targets always, depth targets once a shader sampled them).
    static uint64_t gpuBytesOf(const Surface& s) {
        const vpgpu::FormatInfo& fi = vpgpu::formatInfo(s.format);
        const uint64_t bs = std::max<uint32_t>(fi.blockSize, 1), bpb = std::max<uint32_t>(fi.bytesPerBlock, 1);
        uint64_t chain = 0;
        for (uint32_t level = 0; level < std::min<uint32_t>(std::max<uint16_t>(s.mips, 1), 32); ++level) {
            const uint64_t w = std::max<uint32_t>(s.width >> level, 1), h = std::max<uint32_t>(s.height >> level, 1);
            const uint64_t slices = s.type == vpgpu::TextureType::Tex3D ? std::max<uint32_t>(s.depth >> level, 1) : s.layers;
            chain += (w + bs - 1) / bs * ((h + bs - 1) / bs) * bpb * slices;
        }
        const uint64_t samples = std::max<uint8_t>(s.samples, 1);
        if (s.ms) return chain * samples + (s.id ? chain : 0);
        return s.id ? chain * samples : 0;
    }

    // Formats a texture might be seen through as another one of the same texel size.
    static bool sameTexel(vpgpu::Format a, vpgpu::Format b) {
        const vpgpu::FormatInfo &x = vpgpu::formatInfo(a), &y = vpgpu::formatInfo(b);
        return x.bytesPerBlock == y.bytesPerBlock && x.blockSize == y.blockSize && x.integer() == y.integer() &&
               x.depth() == y.depth() && !x.depth();
    }

    // RGBA8 / BGRA8 differ only in channel order. A T# over an image of the other order is read as
    // the image's own format (the game's dst_sel swizzle, which is not applied, restores the order).
    static vpgpu::Format swapOrder(vpgpu::Format f) {
        using F = vpgpu::Format;
        switch (f) {
            case F::RGBA8Unorm: return F::BGRA8Unorm;
            case F::BGRA8Unorm: return F::RGBA8Unorm;
            case F::RGBA8Srgb: return F::BGRA8Srgb;
            case F::BGRA8Srgb: return F::RGBA8Srgb;
            default: return f;
        }
    }
    static bool bgra(vpgpu::Format f) { return f == vpgpu::Format::BGRA8Unorm || f == vpgpu::Format::BGRA8Srgb; }
    static bool rgba(vpgpu::Format f) { return f == vpgpu::Format::RGBA8Unorm || f == vpgpu::Format::RGBA8Srgb; }
    static vpgpu::Format inOrderOf(vpgpu::Format f, vpgpu::Format ref) {
        return (bgra(f) && rgba(ref)) || (rgba(f) && bgra(ref)) ? swapOrder(f) : f;
    }

    // ---- Tracker bookkeeping

    bool mirrored(uint64_t address, uint64_t bytes) {
        GuestRegion g;
        return bytes && region(address, bytes, g) && g.mirrored;
    }
    // Inside the range only: a page that also holds other data must not be protected for it. The
    // partial pages at both ends of an image therefore stay unprotected: a CPU access that touches
    // only them is not caught (the same limit as every page-granular watch; the tracker's content
    // checks cover textures, a render target's edge rows may read stale bytes).
    static bool inwardPages(uint64_t address, uint64_t bytes, uint64_t& a, uint64_t& n) {
        const uint64_t lo = (address + MemoryTracker::kPageBytes - 1) / MemoryTracker::kPageBytes * MemoryTracker::kPageBytes;
        const uint64_t hi = (address + bytes) / MemoryTracker::kPageBytes * MemoryTracker::kPageBytes;
        if (hi <= lo) return false;
        a = lo, n = hi - lo;
        return true;
    }
    // Pages given back to the CPU by a write back that is only recorded: they stay protected until the
    // GPU has run it (frame = the one that holds the write back), or the CPU would write memory that
    // the re-tile then overwrites with the image's older bytes (or read bytes it has not written yet).
    struct DeferredOwned {
        uint64_t frame, address, bytes;
    };
    std::deque<DeferredOwned> deferredOwned;
    // `afterGpu`: the release belongs to a write back recorded into the open frame; the host hears
    // of it when that frame has finished (retireOwned). Otherwise it is immediate (the caller has waited
    // for the GPU, or nothing was recorded).
    void watchOwned(Surface& s, bool on, bool afterGpu = false) {
        // (an image that cannot be written back is not watched: the CPU would only be stopped for nothing)
        if (s.ownedWatched == on || !r.hooks_.watchOwned || (on && (!canRetile(s) || !ownsBase(s)))) return;
        uint64_t a, n;
        if (!inwardPages(s.address, s.bytes, a, n) || !mirrored(a, n)) return;
        if (on) {
            // A release still waiting for the GPU is taken back: the host keeps its reference.
            for (auto it = deferredOwned.begin(); it != deferredOwned.end(); ++it) {
                if (it->address == a && it->bytes == n) {
                    deferredOwned.erase(it);
                    s.ownedWatched = true;
                    return;
                }
            }
            s.ownedWatched = true;
            r.hooks_.watchOwned(a, n, true);
            return;
        }
        s.ownedWatched = false;
        if (afterGpu)
            deferredOwned.push_back({r.device_.submittedFrames(), a, n});
        else
            r.hooks_.watchOwned(a, n, false);
    }
    // Hands the host the releases whose write backs the GPU has run (`all`: it is idle, or lost).
    void retireOwned(bool all = false) {
        if (deferredOwned.empty()) return;
        const uint64_t done = all || r.device_.deviceLost() ? ~uint64_t(0) : r.device_.completedFrames();
        while (!deferredOwned.empty() && deferredOwned.front().frame < done) {
            const DeferredOwned d = deferredOwned.front();
            deferredOwned.pop_front();
            if (r.hooks_.watchOwned) r.hooks_.watchOwned(d.address, d.bytes, false);
        }
    }
    // The image's range after a recorded write: it holds the newest bytes, memory is behind it. A depth image owns only
    // the planes it can write back (planesOf): memory stays the truth for the others.
    void markWrite(Surface& s) {
        ++s.writeSeq;
        if (!s.bytes) return;
        Plane planes[2];
        const unsigned n = planesOf(s, planes);
        for (unsigned i = 0; i < n; ++i)
            s.version = tracker.markGpuWrite(planes[i].address, planes[i].bytes, s.wid);
        watchOwned(s, true);
    }
    void markWriteWid(uint32_t wid) {
        if (Surface* s = surfaceOf(wid)) markWrite(*s);
    }
    bool ownsStale(const Surface& s) {
        std::vector<MemoryTracker::Owner> o;
        Plane planes[2];
        const unsigned n = planesOf(s, planes);
        for (unsigned i = 0; i < n; ++i) tracker.staleOwners(planes[i].address, planes[i].bytes, o);
        for (const auto& x : o)
            if (x.writer == s.wid) return true;
        return false;
    }

    // ---- Tiling seam
    //
    // The only place that moves bytes between guest memory and an image: linear surfaces by buffer
    // <-> texture copies, tiled ones through the GpuTiler kernels (CPU detile while they compile).

    bool gpuTiling = true;
    bool guestReadOnly = false;  // Renderer::setGuestReadOnly
    bool imageAlias = !std::getenv("VPKENGINE_NO_IMAGE_ALIAS");  // Renderer::setImageAlias (the environment variable turns it off)
    std::unique_ptr<GpuTiler> tilerPtr;
    GpuTiler& tiler() {
        if (!tilerPtr)
            tilerPtr = std::make_unique<GpuTiler>(
                r.device_,
                [this](uint64_t id, const vpmsl::MslShader& shader) {
                    if (r.hooks_.shader) r.hooks_.shader(id, vpir::Stage::Compute, shader);
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
    // `plane`: the format of the plane being moved (a depth image holds Z and stencil apart, see planesOf).
    static TiledSurface tiledOf(const Surface& s) {
        return tiledOf(s, s.isDepth ? zPlaneFormat(s.format) : s.layout != vpgpu::Format::Undefined ? s.layout : s.format);
    }
    static TiledSurface tiledOf(const Surface& s, vpgpu::Format plane) {
        const vpgpu::FormatInfo& fi = vpgpu::formatInfo(plane);
        TiledSurface t;
        t.tileIndex = s.tile;
        t.bytesPerElement = fi.bytesPerBlock;
        t.blockSize = std::max<uint32_t>(fi.blockSize, 1);
        t.width = s.width;
        t.height = s.height;
        t.pitch = std::max(s.pitch, s.width);
        t.volume = s.type == vpgpu::TextureType::Tex3D;
        t.slices = t.volume ? s.depth : s.layers;
        t.pow2Pad = s.pow2Pad;
        if (t.pow2Pad && !t.volume) t.slices = std::bit_ceil(t.slices);
        t.levels = std::max<uint16_t>(s.mips, 1);
        t.samples = std::max<uint8_t>(s.memSamples, 1);
        return t;
    }
    static bool supportedFormat(const Surface& s) {
        if (s.isDepth) return s.format == vpgpu::Format::D32Float || s.format == vpgpu::Format::D16Unorm;
        return true;
    }
    static vpgpu::Aspect aspectOf(const Surface& s) { return s.isDepth ? vpgpu::Aspect::Depth : vpgpu::Aspect::All; }

    // D32FloatS8 is one host image; in guest memory the 32-bit float Z and the 8-bit stencil are two planes at two bases.
    static vpgpu::Format zPlaneFormat(vpgpu::Format f) { return f == vpgpu::Format::D32FloatS8 ? vpgpu::Format::D32Float : f; }
    // A part of the image that is written back on its own: format as stored in guest memory, aspect of the host
    // image it comes from, guest base and size.
    struct Plane {
        vpgpu::Format format;
        vpgpu::Aspect aspect;
        uint64_t address, bytes;
    };
    // The planes to write back (0: none can be). Colour: the image. Depth: a plane only if the image holds all of it
    // (Surface::zAuthoritative / stencilAuthoritative) and, as shadPS4 does for the planes of a depth buffer (the write
    // base when there is one), the draw's read and write base of the plane agree (Surface::followed for Z, a stencil base
    // is kept only then): a draw that reads one surface and writes another is not followed. D16 and D32Float are Z alone;
    // a D32FloatS8 only when Z is Z_32_FLOAT (Z_16 / Z_24 with stencil are held as 32-bit float: the guest words differ in
    // size, nothing is written) with the stencil plane if its base is known; S8Uint is the stencil plane.
    static unsigned planesOf(const Surface& s, Plane (&out)[2]) {
        using F = vpgpu::Format;
        if (!s.isDepth) return out[0] = {s.format, vpgpu::Aspect::All, s.address, s.bytes}, 1;
        unsigned n = 0;
        switch (s.format) {
            case F::D16Unorm:
            case F::D32Float:
                if (s.followed && s.zAuthoritative) out[n++] = {s.format, vpgpu::Aspect::Depth, s.address, s.bytes};
                break;
            case F::D32FloatS8:
                if (s.zFormat != 3) break;
                if (s.followed && s.zAuthoritative) out[n++] = {F::D32Float, vpgpu::Aspect::Depth, s.address, s.bytes};
                if (s.stencilAddress && s.stencilAuthoritative)
                    out[n++] = {F::S8Uint, vpgpu::Aspect::Stencil, s.stencilAddress, s.stencilBytes};
                break;
            case F::S8Uint:
                if (s.followed && s.stencilAuthoritative) out[n++] = {F::S8Uint, vpgpu::Aspect::Stencil, s.address, s.bytes};
                break;
            default: break;
        }
        return n;
    }
    // Whether the range at the image's base (the one the host watches for it) is one of the planes written back.
    static bool ownsBase(const Surface& s) {
        Plane planes[2];
        const unsigned n = planesOf(s, planes);
        for (unsigned i = 0; i < n; ++i)
            if (planes[i].address == s.address) return true;
        return false;
    }

    // ---- Texture probe (diagnostics; only the first kProbeFrames submits, a few hundred lines at most)
    //
    // vpps4.texprobe: one line per texture that a pass writing a large colour target (an eye) samples: the whole T#,
    // the surface that backs it, the guest layout of that surface and how it was filled from memory.
    // vpps4.texfail: a fill or a write back that went outside guest memory or was refused: the address, the bytes the
    // layout needs, the bytes the host maps there, the tile mode.
    static constexpr uint64_t kProbeFrames = 300;
    static constexpr unsigned kProbeLines = 400, kFailLines = 160;
    std::string probePass;  // the pass being bound (empty: none worth a line)
    unsigned probeLines = 0, failLines = 0;
    std::unordered_set<uint64_t> probeSeen, failSeen;
    bool probeOpen() const { return r.frame_ < kProbeFrames; }

    // Bytes of guest memory the host maps from `address` on (0: none; `why` then says what the host knows of it).
    uint64_t mappedFrom(uint64_t address, std::string& why) {
        GuestRegion g;
        if (region(address, 1, g)) return g.base + g.size - address;
        if (r.hooks_.explainAddress) why = r.hooks_.explainAddress(address);
        return 0;
    }
    std::string describeSurface(const Surface& s) {
        const SurfaceLayout lay = computeLayout(tiledOf(s));
        std::string t;
        if (const auto it = surfaceRefs.find(s.wid); it != surfaceRefs.end())
            t += it->second.kind == 0 ? "kind=color " : it->second.kind == 1 ? "kind=depth " : "kind=texture ";
        t += "addr=" + hex(s.address) + " tile=" + std::to_string(s.tile) + " fmt=" + std::to_string(int(s.format)) + " " +
             std::to_string(s.width) + "x" + std::to_string(s.height) + "x" + std::to_string(slicesOf(s)) +
             " pitch=" + std::to_string(s.pitch) + " mips=" + std::to_string(s.mips) + " samples=" + std::to_string(s.samples) + "/" +
             std::to_string(s.memSamples) + " pow2=" + std::to_string(s.pow2Pad) + " bytes=" + std::to_string(s.bytes);
        if (lay.ok) {
            const LevelLayout& l0 = lay.levels[0];
            t += " tiledOf.bytes=" + std::to_string(lay.bytes) + " linear=" + std::to_string(lay.linearBytes) + " levels=" +
                 std::to_string(lay.levels.size()) + " l0=" + std::to_string(l0.pitch) + "x" + std::to_string(l0.paddedHeight) +
                 (l0.degraded ? "/1d" : "") + " macro_mode=" + std::to_string(lay.macroTileMode);
        } else {
            t += " tiledOf=none(" + lay.why + ")";
        }
        return t;
    }
    // A texture made, or refilled from memory: counted by tile mode, T# format and whether its level 0 is stored 1D (QUAL tile_plan_*,
    // tile_refill_*); the first 40 and then one in 600 also leave a TILE line with the layout.
    void probeTile(const Surface& s, const SurfaceLayout& lay, bool refill) {
        if (!vpgpu::probe::enabled() || !lay.ok) return;
        const LevelLayout& l0 = lay.levels[0];
        unsigned degradedLevels = 0;
        for (const LevelLayout& l : lay.levels) degradedLevels += l.degraded;
        const uint64_t key = uint64_t(s.tile) | uint64_t(s.dfmt) << 8 | uint64_t(s.nfmt) << 16 | uint64_t(l0.degraded) << 24;
        if (pstat.tilePlans.size() < kProbeMaxEntries || pstat.tilePlans.count(key)) {
            ProbeStats::TileCell& c = pstat.tilePlans[key];
            ++(refill ? c.refills : c.textures);
            c.tile = s.tile, c.dfmt = s.dfmt, c.nfmt = s.nfmt, c.degraded = l0.degraded ? 1 : 0;
        }
        const uint64_t n = ++pstat.tileLines;
        if (n > 40 && n % 600) return;  // (the first 40, then one in 600)
        log("VPKENGINE_PROBE TILE n=" + std::to_string(n) + " f=" + std::to_string(r.frame_) + (refill ? " refill" : " texture") +
            " T#=" + hex(s.address) + " tile=" + std::to_string(s.tile) + " dfmt=" + std::to_string(s.dfmt) + " nfmt=" + std::to_string(s.nfmt) +
            " pitch=" + std::to_string(s.pitch) + " w=" + std::to_string(s.width) + " h=" + std::to_string(s.height) + " mips=" +
            std::to_string(s.mips) + " l0=" + std::to_string(l0.pitch) + "x" + std::to_string(l0.paddedHeight) + (l0.degraded ? "/1d" : "") +
            " degraded=" + (l0.degraded ? "1" : "0") + " degraded_levels=" + std::to_string(degradedLevels));
    }
    void probeFailure(const char* kind, const Surface& s, uint64_t expected) {
        if (!probeOpen() || failLines >= kFailLines) return;
        Hash h;
        h.bytes(kind, std::strlen(kind));
        h.value(s.address);
        h.value(expected);
        if (!failSeen.insert(h.h).second) return;
        ++failLines;
        std::string why;
        const uint64_t mapped = mappedFrom(s.address, why);
        log("vpps4.texfail f=" + std::to_string(r.frame_) + " " + kind + " " + describeSurface(s) + " expected=" + std::to_string(expected) +
            " mapped=" + std::to_string(mapped) + (mapped ? std::string() : " unmapped: " + why) + " fill_why=" + s.fillWhy +
            (failLines == kFailLines ? " (last texfail line: cap)" : ""));
    }
    // The route textureRef took for the T# bind() armed (cmpArm): which image it ended on. Nothing is written while the probe is off.
    bool cmpArm = false;
    uint8_t cmpRoute = ProbeStats::kCmpOther;
    const Surface* cmpSurface = nullptr;
    void routed(uint8_t route, const Surface& s) {
        if (cmpArm) cmpRoute = route, cmpSurface = &s;
    }
    void probeSampled(const ImageDesc& d, const Surface& s, const char* backing, bool storage) {
        if (probePass.empty() || !probeOpen() || probeLines >= kProbeLines) return;
        Hash h;
        h.bytes(probePass.data(), probePass.size());
        h.value(d.address);
        h.value(s.wid);
        h.value(s.fillPath);
        h.value(uint64_t(d.width) << 32 | d.height);
        h.value(uint64_t(d.pitch) << 32 | d.depth);
        h.value(uint64_t(d.tilingIndex) | uint64_t(d.dataFormat) << 8 | uint64_t(d.numFormat) << 16 | uint64_t(d.baseLevel) << 24 |
                uint64_t(d.lastLevel) << 32 | uint64_t(d.type) << 40);
        h.value(uint64_t(d.baseArray) << 16 | d.lastArray);
        h.bytes(d.dstSel.data(), d.dstSel.size());
        if (!probeSeen.insert(h.h).second) return;
        ++probeLines;
        static const char* const kFill[] = {"none", "gpu", "cpu", "linear", "unfilled"};
        std::string sel;
        for (uint8_t v : d.dstSel) sel += char('0' + (v & 7));
        log("vpps4.texprobe f=" + std::to_string(r.frame_) + " pass=[" + probePass + "] " + (storage ? "storage " : "") + "T# base=" +
            hex(d.address) + " " + std::to_string(d.width) + "x" + std::to_string(d.height) + " depth=" + std::to_string(d.depth) +
            " pitch=" + std::to_string(d.pitch) + " tile=" + std::to_string(d.tilingIndex) + " fmt=" + std::to_string(d.dataFormat) + "/" +
            std::to_string(d.numFormat) + " type=" + std::to_string(d.type) + " level=" + std::to_string(d.baseLevel) + ".." +
            std::to_string(d.lastLevel) + " array=" + std::to_string(d.baseArray) + ".." + std::to_string(d.lastArray) +
            " pow2=" + std::to_string(d.pow2Pad) + " dst_sel=" + sel + " | backing=" + backing + " " + describeSurface(s) +
            " fill=" + kFill[std::min<unsigned>(s.fillPath, 4)] + (s.fillPath == 4 ? std::string("(") + s.fillWhy + ")" : std::string()) +
            " fills=" + std::to_string(s.fills) + (probeLines == kProbeLines ? " (last texprobe line: cap)" : ""));
    }

    // Whether the image can be written back to its guest range (re-tiled).
    bool canRetile(const Surface& s) const {
        if (guestReadOnly) return false;
        if (!s.bytes || !s.id || s.memSamples != 1 || s.softBc) return false;  // (multisampled depth has no single-sample image: not written)
        Plane planes[2];
        const unsigned n = planesOf(s, planes);
        if (!n) return false;
        for (unsigned i = 0; i < n; ++i) {
            if (vpgpu::formatInfo(planes[i].format).blockSize > 1) return false;  // (block-compressed images are never written)
            if (linearTiling(s.tile)) continue;
            if (!gpuTiling || !computeLayout(tiledOf(s, planes[i].format)).ok) return false;
        }
        return true;
    }

    // image -> guest memory, on the GPU timeline (GpuTiler::retile for tiled surfaces), every plane of the image. When
    // it returns false, retileWhy says why (diagnostics only).
    const char* retileWhy = "";
    bool retileToMemory(Surface& s) {
        retileWhy = "image cannot be retiled (samples, format or layout)";
        if (!canRetile(s)) return false;
        // A clear that waits for the next pass must reach the image first; the single-sample picture of a
        // multisampled target is what is written, so it is up to date and the lifetime planner keeps
        // resolving it from now on (CPU, DMA and retile readers are reads like a shader's).
        flushClear(s);
        catchUpTwin(s);
        life.noteRead(s.id, r.frame_);
        Plane planes[2];
        const unsigned n = planesOf(s, planes);
        for (unsigned i = 0; i < n; ++i)
            if (!retilePlane(s, planes[i])) return false;
        ++r.stats_.retiled;
        return true;
    }
    bool retilePlane(Surface& s, const Plane& p) {
        const vpgpu::FormatInfo& fi = vpgpu::formatInfo(p.format);
        if (!linearTiling(s.tile)) {
            const SurfaceLayout lay = computeLayout(tiledOf(s, p.format));
            BufferId gb;
            uint64_t at = 0, available = 0;
            if (!guestBuffer(p.address, lay.bytes, gb, at, available) || available < lay.bytes || at % 4) {
                retileWhy = "tiled outside guest memory";
                note("image newer than memory lies outside guest memory: not written back");
                return false;
            }
            Subresource sub;
            sub.levelCount = uint32_t(lay.levels.size());
            sub.sliceCount = s.type == vpgpu::TextureType::Tex3D ? 0 : s.layers;
            sub.aspect = p.aspect;
            closePass();
            const GpuTiler::Result res = tiler().retile(frame, s.id, {gb, at, available}, lay, sub);
            if (res == GpuTiler::Result::NotReady) {  // (kernel compiling: the image stays ahead, tried again)
                retileWhy = "retile kernel compiling";
                return false;
            }
            retileWhy = "GPU retile refused the request";
            if (res != GpuTiler::Result::Recorded) return note("GPU retile refused the request"), false;
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
        if (!guestBuffer(p.address, total, buffer, offset, available) || total > available || offset % fi.bytesPerBlock) {
            retileWhy = "linear outside guest memory or misaligned";
            note("image newer than memory lies outside guest memory: not written back");
            return false;
        }
        closePass();
        vpgpu::Copy c;
        c.kind = vpgpu::CopyKind::TextureToBuffer;
        c.srcTexture.texture = s.id;
        c.srcTexture.layerCount = uint16_t(s.type == vpgpu::TextureType::Tex3D ? 1 : slicesOf(s));
        c.srcTexture.width = s.width;
        c.srcTexture.height = s.height;
        c.srcTexture.depth = s.type == vpgpu::TextureType::Tex3D ? s.depth : 1;
        c.srcTexture.aspect = p.aspect;
        c.dstBuffer = {buffer, offset, total, uint32_t(rowBytes), blocksHigh};
        c.label = "retile " + hex(p.address);
        frame.copy(c);
        ++r.stats_.copies;
        return true;
    }

    // guest memory -> image, on the GPU timeline (every mip level of every layer). False (noted) if
    // the layout or format is not covered; the image is then left as it was.
    bool detileToImage(Surface& s) {
        s.fillPath = 4;
        s.fillWhy = "multisampled image";
        if (s.samples > 1) return note("multisampled image not filled from memory"), false;
        s.fillWhy = "depth format";
        if (!supportedFormat(s)) return note("depth image of this format not filled from memory"), false;
        const SurfaceLayout lay = computeLayout(tiledOf(s));
        s.fillWhy = "layout not covered";
        if (!lay.ok) return note((linearTiling(s.tile) ? "linear texture not uploaded: " : "tiled texture not detiled: ") + lay.why), false;
        const vpgpu::FormatInfo& fi = vpgpu::formatInfo(s.format);
        const bool volume = s.type == vpgpu::TextureType::Tex3D;
        const uint32_t layers = volume ? 1 : s.layers;
        const uint32_t levels = uint32_t(lay.levels.size());
        if (s.softBc) return fillSoftBlock(s, lay);
        if (linearTiling(s.tile)) {
            // Every level straight from guest memory, rows `pitch` texels apart.
            BufferId buffer;
            uint64_t offset = 0, available = 0;
            if (!guestBuffer(s.address, lay.bytes, buffer, offset, available) || lay.bytes > available || offset % fi.bytesPerBlock) {
                s.fillWhy = "linear outside guest memory or misaligned";
                probeFailure("linear outside/misaligned", s, lay.bytes);
                return note("linear texture outside guest memory or misaligned: left unfilled"), false;
            }
            closePass();
            for (uint32_t li = 0; li < levels; ++li) {
                const LevelLayout& l = lay.levels[li];
                vpgpu::Copy c;
                c.kind = vpgpu::CopyKind::BufferToTexture;
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
            s.fillPath = 3, s.fillWhy = "";
            return true;
        }
        GuestRegion g;
        if (!region(s.address, lay.bytes, g)) {
            s.fillWhy = "tiled outside guest memory";
            probeFailure("tiled outside", s, lay.bytes);
            return note("tiled texture outside guest memory: left unfilled"), false;
        }
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
                    s.fillPath = 1, s.fillWhy = "";
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
        s.fillWhy = "no staging memory";
        if (!dst) return fillRetry = true, note("no staging memory for a tiled texture: left unfilled"), false;
        s.fillWhy = "detileLevels refused";
        if (!detileLevels(lay, 0, levels, 0, src, dst)) return note("tiled texture not detiled: " + lay.why), false;
        if (recordedMayWrite(s.address, lay.bytes)) note("CPU detile of memory that recorded GPU work may still write");
        closePass();
        for (uint32_t li = 0; li < levels; ++li) {
            const LevelLayout& l = lay.levels[li];
            vpgpu::Copy c;
            c.kind = vpgpu::CopyKind::BufferToTexture;
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
        s.fillPath = 2, s.fillWhy = "";
        return true;
    }
    // Block-compressed textures on a host without BC: the decoded bytes (the memory they take, and the CPU time when the CPU
    // decodes) admitted per frame (r.frame_, one submit). A texture past it stays unfilled and its draw is left out, until
    // a later frame; the first decode of a frame is always admitted, so a texture bigger than the budget still gets through.
    uint64_t bcFrameBytes = 64ull << 20, bcSpent = 0, bcSpentFrame = ~uint64_t(0);
    bool bcAdmits(uint64_t bytes) {
        if (bcSpentFrame != r.frame_) bcSpentFrame = r.frame_, bcSpent = 0;
        return !bcSpent || bcSpent + bytes <= bcFrameBytes;
    }
    // A fill that stays undone for a reason that passes: the image is not filled, tried again at its next use, and
    // textureRef leaves the draw out (the reason is the skip's).
    const char* fillDeferred = nullptr;
    bool deferFill(Surface& s, const char* why) {
        s.fillPath = 4, s.fillWhy = why, fillDeferred = why;
        return fillRetry = true, false;
    }
    // A block-compressed texture on a host without BC. First choice: the BC kernel of the GPU tiler reads the blocks from
    // guest memory (detiling as it goes) and decodes them on the GPU, only the levels whose memory changed since they were
    // last decoded. While that kernel is still compiling the texture waits for it (the draw is left out: decoding on the
    // render thread would hold the frame for seconds). When it refuses (or cannot compile): every level is read from guest
    // memory (detiled on the CPU when tiled), decoded on the CPU to the host format and copied in as plain texels, at the
    // texel size of each level. Either way within the frame's budget of decoded bytes (bcAdmits), all of the levels or none.
    bool fillSoftBlock(Surface& s, const SurfaceLayout& lay) {
        bc::Kind kind{};
        bc::kindOf(s.softBc, kind);  // (the plan sets softBc for the data formats 35..41 only)
        const uint32_t levelCount = uint32_t(lay.levels.size());
        const uint32_t allLevels = levelCount >= 32 ? ~0u : (1u << levelCount) - 1;
        const bool volume = s.type == vpgpu::TextureType::Tex3D;
        const uint32_t texel = bc::texelBytes(kind);
        // (the texels of a level as the host holds them)
        auto texels = [&](uint32_t li, uint32_t& w, uint32_t& h, uint32_t& slices) {
            w = std::max(s.width >> li, 1u), h = std::max(s.height >> li, 1u);
            slices = volume ? std::max(s.depth >> li, 1u) : s.layers;
        };
        if (gpuTiling) {
            BufferId gb;
            uint64_t at = 0, available = 0;
            if (guestBuffer(s.address, lay.bytes, gb, at, available) && available >= lay.bytes && at % 4 == 0) {
                bool recorded = false, refused = false;
                uint32_t todo = 0;  // (the levels whose memory changed since they were last decoded)
                uint64_t need = 0;
                for (uint32_t li = 0; li < levelCount; ++li) {
                    const LevelLayout& l = lay.levels[li];
                    if ((s.bcLevels >> li & 1u) && !tracker.writtenByOthers(s.address + l.offset, l.bytes, s.wid, s.version)) continue;
                    uint32_t w, h, n;
                    texels(li, w, h, n);
                    todo |= 1u << li, need += uint64_t(w) * h * n * texel;
                }
                if (need && !bcAdmits(need)) return deferFill(s, "bc decode budget");
                closePass();
                for (uint32_t li = 0; li < levelCount && !refused; ++li) {
                    if (!(todo >> li & 1u)) continue;
                    Subresource sub;
                    sub.firstLevel = li;
                    sub.levelCount = 1;
                    sub.sliceCount = s.type == vpgpu::TextureType::Tex3D ? 0 : s.layers;
                    const GpuTiler::Result res = tiler().decodeBc(frame, {gb, at, available}, s.id, lay, sub, {uint32_t(kind), s.softSigned});
                    if (res == GpuTiler::Result::NotReady) return deferFill(s, "bc kernel compiling");
                    if (res != GpuTiler::Result::Recorded) {
                        refused = true;
                        note("GPU block decode refused the request: decoded on the CPU");
                        break;
                    }
                    s.bcLevels |= 1u << li;
                    recorded = true;
                    ++r.stats_.copies;
                    ++r.stats_.gpuDetiled;
                }
                if (!refused) {
                    if (recorded) ++r.stats_.detiled;
                    bcSpent += need;
                    s.fillPath = 1, s.fillWhy = "";
                    return true;
                }
                if (recorded) {  // (levels before the refused one are in: the rest is tried again at the next use)
                    s.fillWhy = "GPU block decode stopped";
                    return fillRetry = true, false;
                }
            }
        }
        s.bcLevels = 0;
        const uint32_t blockBytes = vpgpu::formatInfo(s.layout).bytesPerBlock;
        const uint32_t layers = volume ? 1 : s.layers;
        const uint32_t levels = uint32_t(lay.levels.size());
        const bool linear = linearTiling(s.tile);
        GuestRegion g;
        if (!region(s.address, lay.bytes, g)) {
            s.fillWhy = linear ? "linear outside guest memory" : "tiled outside guest memory";
            probeFailure(linear ? "linear outside/misaligned" : "tiled outside", s, lay.bytes);
            return note(linear ? "linear texture outside guest memory: left unfilled" : "tiled texture outside guest memory: left unfilled"), false;
        }
        const uint8_t* src = static_cast<const uint8_t*>(g.host) + (s.address - g.base);
        uint64_t total = 0;
        for (uint32_t li = 0; li < levels; ++li) {
            uint32_t w, h, n;
            texels(li, w, h, n);
            total += uint64_t(w) * h * n * texel;
        }
        if (!bcAdmits(total)) return deferFill(s, "bc decode budget");
        BufferId buffer;
        uint64_t offset = 0;
        uint8_t* dst = staging(total, buffer, offset);
        s.fillWhy = "no staging memory";
        if (!dst) return fillRetry = true, note("no staging memory for a block-compressed texture: left unfilled"), false;
        std::vector<uint8_t> detiled;
        if (!linear) {
            detiled.resize(lay.linearBytes);
            s.fillWhy = "detileLevels refused";
            if (!detileLevels(lay, 0, levels, 0, src, detiled.data())) return note("tiled texture not detiled: " + lay.why), false;
            if (recordedMayWrite(s.address, lay.bytes)) note("CPU detile of memory that recorded GPU work may still write");
        }
        closePass();
        uint64_t at = 0;
        for (uint32_t li = 0; li < levels; ++li) {
            const LevelLayout& l = lay.levels[li];
            uint32_t w, h, n;
            texels(li, w, h, n);
            // (slices of the guest level are `slice` bytes apart: tightly packed once detiled, `pitch` x `paddedHeight` in a linear one)
            const uint64_t rowBytes = linear ? uint64_t(l.pitch) * blockBytes : uint64_t(l.width) * blockBytes;
            const uint64_t slice = rowBytes * (linear ? l.paddedHeight : l.height);
            const uint8_t* level = linear ? src + l.offset : detiled.data() + l.linearOffset;
            for (uint32_t k = 0; k < n; ++k)
                bc::decodeImage(kind, s.softSigned, level + k * slice, rowBytes, w, h, dst + at + uint64_t(k) * w * h * texel, uint64_t(w) * texel);
            vpgpu::Copy c;
            c.kind = vpgpu::CopyKind::BufferToTexture;
            c.srcBuffer = {buffer, offset + at, uint64_t(w) * h * n * texel, uint32_t(uint64_t(w) * texel), h};
            c.dstTexture.texture = s.id;
            c.dstTexture.mip = uint16_t(li);
            c.dstTexture.layerCount = uint16_t(layers);
            c.dstTexture.width = w;
            c.dstTexture.height = h;
            c.dstTexture.depth = volume ? n : 1;
            c.dstTexture.aspect = vpgpu::Aspect::All;
            c.label = "decode " + hex(s.address) + (li ? " mip " + std::to_string(li) : "");
            frame.copy(c);
            ++r.stats_.copies;
            at += uint64_t(w) * h * n * texel;
        }
        s.bcLevels = allLevels;
        bcSpent += total;
        ++r.stats_.detiled;
        s.fillPath = linear ? 3 : 2, s.fillWhy = "";
        return true;
    }
    // (the busy ranges also count reads: only a hint)
    bool recordedMayWrite(uint64_t a, uint64_t n) const { return busyOverlaps(a, n); }

    // ---- Coherence operations

    // Writes back (re-tiles) the images that hold newer bytes than memory in [address, +size),
    // oldest write first. `write`: the caller overwrites the range, so an image it covers completely
    // is not written back (the write takes its bytes over). True if work was recorded.
    bool syncRefused = false;  // set by syncOwners when an image ahead of memory could not be written back
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
                probeFailure(retileWhy, *s, s->bytes);
                note("image newer than memory cannot be written back (layout not covered): raw readers see stale bytes");
                syncRefused = true;
                continue;
            }
            tracker.markSynced(s->wid);
            watchOwned(*s, false, true);  // (the re-tile is only recorded: the pages stay protected until it ran)
            any = true;
        }
        return any;
    }
    // The host gave up waiting for a write back that never comes (the re-tiler refuses the layout): the images
    // ahead of memory in the range stop being owners and let go of their pages; their bytes are not in memory.
    void abandonOwners(uint64_t address, uint64_t size) {
        std::vector<MemoryTracker::Owner> owners;
        tracker.staleOwners(address, size, owners);
        for (const MemoryTracker::Owner& o : owners) {
            Surface* s = surfaceOf(o.writer);
            if (!s) {
                tracker.release(o.writer);
                continue;
            }
            tracker.markSynced(s->wid);
            watchOwned(*s, false);
            note("image newer than memory abandoned (could not be written back)");
        }
    }
    // A raw (buffer / DMA) access of guest memory: images ahead of memory there are written back
    // first; a write then takes the range over.
    void rawAccess(uint64_t address, uint64_t size, bool write) {
        if (!size) return;
        if (size > kMaxRawExtent) size = kMaxRawExtent;  // (a descriptor claiming the whole address space)
        static vpgpu::probe::Timer& rawTimer = vpgpu::probe::timer("TIME", "hazard_raw_access");
        const vpgpu::probe::ScopedSampled rawScope(rawTimer);  // (hot: calls exact, time estimated from 1 in 16)
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
    // its range were written back. When memory cannot be trusted (an image ahead of it could not be written back) or the
    // fill itself fails, the image stays not filled and is tried again at its next use (never marked up to date with
    // stale or missing bytes).
    bool fillRetry = false;  // set by a fill that failed for a reason that passes (staging memory, a decode that stopped)
    void fillFromMemory(Surface& s, Texture* tex) {
        syncRefused = false;
        syncOwners(s.address, s.bytes, false, s.wid);
        if (syncRefused) {
            s.fillPath = 4;
            s.fillWhy = "image ahead of memory not written back";
            s.filled = false;
            return;
        }
        fillRetry = false;
        const bool ok = detileToImage(s);
        ++pstat.fillPath[std::min<unsigned>(s.fillPath, 4)];
        pstat.fillPathBytes[std::min<unsigned>(s.fillPath, 4)] += s.bytes;
        ++s.fills;
        if (!ok && fillRetry) {  // (no staging memory right now: the image stays not filled and is tried again at its next use)
            s.filled = false;
            return;
        }
        s.version = tracker.current();
        s.filled = true;  // (a layout that cannot be filled is noted once, not retried every use)
        if (!ok) return;
        ++s.writeSeq;  // (the fill wrote the image: a copy of it, see mirrorTarget, is out of date)
        if (s.isDepth) {  // (the fill moves Z only: it makes that plane the image's own, and a failed one leaves it behind memory)
            if (s.format != vpgpu::Format::S8Uint) s.zAuthoritative = true;
            else s.stencilAuthoritative = true;
        }
        ++r.stats_.refilled;
        if (tex) stamp(*tex);
        if (r.hooks_.uploaded) r.hooks_.uploaded(s.address, s.bytes);
        if (r.hooks_.watchWrites && mirrored(s.address, s.bytes)) {
            r.hooks_.watchWrites(s.address, s.bytes);
            s.sourceWatched = true;
        }
    }
    // Hooks::seed: the host holds the texels of this surface (the image another renderer drew): they replace the
    // surface's contents. A clear that waited for the surface is dropped, not emitted (the texels are what counts).
    bool seedFromHost(Surface& s, Texture* tex) {
        const vpgpu::FormatInfo& fi = vpgpu::formatInfo(s.format);
        // (depth: Z as 32-bit float or 16-bit, or the Z and stencil planes of a D32FloatS8)
        const bool combined = s.format == vpgpu::Format::D32FloatS8;
        const bool depthOk = !s.isDepth || combined || s.format == vpgpu::Format::D32Float || s.format == vpgpu::Format::D16Unorm;
        if (!s.id || s.samples > 1 || s.layers != 1 || s.mips > 1 || s.softBc || s.type != vpgpu::TextureType::Tex2D ||
            fi.blockSize > 1 || !depthOk || !s.width || !s.height)
            return false;
        SeedRequest q;
        q.kind = tex ? SeedRequest::Kind::Texture : s.isDepth ? SeedRequest::Kind::Depth : SeedRequest::Kind::Color;
        q.address = s.address;
        q.width = s.width;
        q.height = s.height;
        q.format = s.format;
        SeedImage img;
        if (!r.hooks_.seed(q, img) || !img.data) return false;
        if (combined && !img.stencil) return false;  // (a stencil plane left as it was would be undefined: filled from memory instead)
        const uint32_t texel = combined ? 4 : fi.bytesPerBlock;
        const uint64_t row = uint64_t(s.width) * texel;
        const uint64_t stencilRow = combined ? (img.stencilRowBytes ? img.stencilRowBytes : s.width) : 0;
        const uint64_t stencilBytes = stencilRow ? uint64_t(s.width) * s.height : 0;
        const uint64_t stride = img.rowBytes ? img.rowBytes : row;
        BufferId buffer;
        uint64_t offset = 0;
        uint8_t* dst = staging(row * s.height + stencilBytes, buffer, offset);
        if (!dst) return note("no staging memory for a seed: the surface is filled from memory"), false;
        for (uint32_t y = 0; y < s.height; ++y) std::memcpy(dst + y * row, img.data + y * stride, row);
        for (uint32_t y = 0; stencilBytes && y < s.height; ++y)
            std::memcpy(dst + row * s.height + uint64_t(y) * s.width, img.stencil + y * stencilRow, s.width);
        // (a texture takes the host's texels as they are; a target's own waiting clear still applies on top of them, as the
        // load action of its pass: the host took the texels before that pass, which is where the other renderer clears)
        if (tex) std::erase_if(pendingClears, [&](const PendingClear& p) { return p.texture == s.id || p.texture == s.ms; });
        closePass();
        vpgpu::Copy c;
        c.kind = vpgpu::CopyKind::BufferToTexture;
        c.srcBuffer = {buffer, offset, row * s.height, uint32_t(row), s.height};
        c.dstTexture.texture = s.id;
        c.dstTexture.mip = 0;
        c.dstTexture.layerCount = 1;
        c.dstTexture.width = s.width;
        c.dstTexture.height = s.height;
        c.dstTexture.depth = 1;
        c.dstTexture.aspect = aspectOf(s);
        c.label = "seed " + hex(s.address);
        frame.copy(c);
        ++r.stats_.copies;
        if (stencilBytes) {
            c.srcBuffer = {buffer, offset + row * s.height, stencilBytes, s.width, s.height};
            c.dstTexture.aspect = vpgpu::Aspect::Stencil;
            c.label = "seed stencil " + hex(s.address);
            frame.copy(c);
            ++r.stats_.copies;
        }
        ++r.stats_.seeded;
        ++s.writeSeq;
        s.version = tracker.current();
        s.filled = true;
        if (s.isDepth) {
            s.zAuthoritative = true;
            if (stencilBytes) s.stencilAuthoritative = true;
        }
        if (tex) tex->seeded = true;  // (the source in memory is not what the texture holds: a check must not read that as a CPU write, nor read the memory at all)
        return true;
    }
    // R1: the writers newer than the target (`tex` null) in its range are all images of the renderer, none of them at its own
    // base: another target, or a texture, that shares part of its memory (the free half of a dynamic resolution target holds
    // smaller targets). The hardware writes only what each of them covers, but the range is marked whole for each write, so
    // refilling from memory would write back the other image (all of it) and detile it over this one, giving it stale
    // texels of the other. The target keeps its own; CPU, DMA and raw writers, depth, and an image of the same base (the
    // same picture seen another way) still refill. VPKENGINE_R1_ROUNDTRIP=1 turns this off. Fills `refillWriters`
    // (the probe line reads them) when it can tell.
    bool imageWritersSkip = !std::getenv("VPKENGINE_R1_ROUNDTRIP");
    std::vector<MemoryTracker::WriterId> refillWriters;
    bool onlyImagesWrote(const Surface& s, const Texture* tex, uint64_t since) {
        refillWriters.clear();
        if (tex && !vpgpu::probe::enabled()) return false;
        tracker.writersSince(s.address, s.bytes, s.wid, since, refillWriters);
        if (tex || !imageWritersSkip || s.isDepth || refillWriters.empty()) return false;
        for (const MemoryTracker::WriterId w : refillWriters) {
            const Surface* o = MemoryTracker::isImage(w) ? surfaceOf(w) : nullptr;
            if (!o || o->isDepth || o->address == s.address) return false;
        }
        return true;
    }
    // A refill of a target / texture from memory, or the one left out (QUAL refill_*): the first 200 and then one in 600 leave
    // a REFILL line with the writers that caused it (an image's base, size and the last scissor / viewport it drew with).
    void probeRefill(const Surface& s, const Texture* tex, unsigned why, bool skipped) {
        if (!vpgpu::probe::enabled()) return;
        const uint64_t n = ++pstat.refillLines;
        if (n > 200 && n % 600) return;
        static const char* const reason[3] = {"first", "written_by_others", "forget_history"};
        std::string line = "VPKENGINE_PROBE REFILL n=" + std::to_string(n) + " f=" + std::to_string(r.frame_) + " wid=" + std::to_string(s.wid) +
                           (tex ? " kind=texture addr=" : " kind=target addr=") + hex(s.address) + " bytes=" + std::to_string(s.bytes) +
                           " shape=" + std::to_string(s.width) + "x" + std::to_string(s.height) + "x" + std::to_string(slicesOf(s)) +
                           " fmt=" + std::to_string(int(s.format)) + " pitch=" + std::to_string(s.pitch) + " tile=" + std::to_string(s.tile) +
                           " reason=" + reason[why] + " skipped=" + (skipped ? "1" : "0") + " writers=" + std::to_string(refillWriters.size());
        for (size_t i = 0; i < refillWriters.size() && i < 4; ++i) {
            const MemoryTracker::WriterId w = refillWriters[i];
            const Surface* o = MemoryTracker::isImage(w) ? surfaceOf(w) : nullptr;
            if (!o) {
                line += w == MemoryTracker::kCpu ? " [cpu]" : w == MemoryTracker::kDma ? " [dma]" : w == MemoryTracker::kGpuRaw ? " [gpu_raw]" : " [unknown]";
                continue;
            }
            line += " [wid=" + std::to_string(w) + " addr=" + hex(o->address) + " bytes=" + std::to_string(o->bytes) + " scissor=" +
                    std::to_string(o->lastScissor[0]) + "," + std::to_string(o->lastScissor[1]) + "," + std::to_string(o->lastScissor[2]) + "," +
                    std::to_string(o->lastScissor[3]) + " viewport=" + std::to_string(int(o->lastViewport[0])) + "," +
                    std::to_string(int(o->lastViewport[1])) + "," + std::to_string(int(o->lastViewport[2])) + "," +
                    std::to_string(int(o->lastViewport[3])) + "]";
        }
        log(line);
    }
    // Before a use: has another writer changed the image's range since it last held it?
    void refresh(Surface& s, Texture* tex) {
        if (!s.bytes) return;
        if (r.hooks_.seed && seedFromHost(s, tex)) return;
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
        const unsigned why = first ? 0u : tracker.floor() > s.version ? 2u : 1u;
        const bool skip = onlyImagesWrote(s, tex, first ? 0 : s.version);
        probeRefill(s, tex, why, skip);
        if (skip) {  // the target keeps its texels (see onlyImagesWrote)
            s.version = tracker.current();
            ++pstat.refillSkipped;
            pstat.refillSkippedBytes += s.bytes;
            return;
        }
        ++pstat.refills[why];
        pstat.refillBytes[why] += s.bytes;
        if (tex && vpgpu::probe::enabled()) probeTile(s, computeLayout(tiledOf(s)), true);
        fillFromMemory(s, tex);
        if (!s.filled) ++pstat.refillFailed;
        // P3: a colour target refilled from memory (what a refill that fails or repeats costs).
        if (!tex && !s.isDepth && s.samples <= 1 && refillProbe.due(r.frame_))
            log("vpps4.probe P3 refill f=" + std::to_string(r.frame_) + " wid=" + std::to_string(s.wid) + " " + hex(s.address) +
                " reason=" + (first ? "first" : "written_by_others") + " filled=" + (s.filled ? "1" : "0") + " staging_live=" +
                std::to_string(stagingLive() >> 20) + "MB/" + std::to_string(stagingLimit >> 20) + "MB stagings=" +
                std::to_string(stagings.size()));
    }
    ProbeGate refillProbe, resolveProbe, mirrorProbe;
    uint64_t memProbeFrame = ~uint64_t(0);

    static uint64_t mix(uint64_t h, uint64_t w) {
        h ^= w * 0x9E3779B97F4A7C15ull;
        h = (h << 27) | (h >> 37);
        return h * 0xBF58476D1CE4E5B9ull;
    }
    static uint64_t hashRange(const uint8_t* p, uint64_t n) {
        uint64_t h = n, w;
        uint64_t i = 0;
        if (n >= 32) {  // four lanes, each mixed word by word: independent chains (fast), no pair of changes cancels
            uint64_t l0 = h, l1 = h + 1, l2 = h + 2, l3 = h + 3;
            for (; i + 32 <= n; i += 32) {
                uint64_t a, b, c, d;
                std::memcpy(&a, p + i, 8), std::memcpy(&b, p + i + 8, 8), std::memcpy(&c, p + i + 16, 8),
                    std::memcpy(&d, p + i + 24, 8);
                l0 = mix(l0, a), l1 = mix(l1, b), l2 = mix(l2, c), l3 = mix(l3, d);
            }
            h = mix(mix(mix(l0, l1), l2), l3);
        }
        for (; i + 8 <= n; i += 8) std::memcpy(&w, p + i, 8), h = mix(h, w);
        for (; i < n; ++i) h = mix(h, p[i]);
        return h;
    }
    // The whole of a source up to 4 KiB; a larger one: its first and last 256 bytes and 16 lines of 64 bytes spread over it.
    static uint64_t sampleRange(const uint8_t* p, uint64_t n) {
        if (n <= kSampleWhole) return hashRange(p, n);
        uint64_t h = mix(hashRange(p, 256), hashRange(p + n - 256, 256));
        for (uint64_t k = 0; k < 16; ++k) h = mix(h, hashRange(p + (n - 64) * k / 15, 64));
        return h;
    }
    static constexpr uint64_t kSampleWhole = 4096;
    const uint8_t* guestBytes(uint64_t address, uint64_t bytes) {
        GuestRegion g;
        if (!bytes || !region(address, bytes, g)) return nullptr;
        return static_cast<const uint8_t*>(g.host) + (address - g.base);
    }
    // After a fill: remember what the source held.
    void stamp(Texture& t) {
        t.seeded = false;
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
        if (!t.filled || !t.bytes || t.sourceWatched || t.seeded || t.checkedFrame == r.frame_) return;
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
    // A buffer binding whose V# base is not in memory the host offers: base, size and what the host knows
    // about the address (census "buffer-unmapped", key = 16 MB bucket; the log: first sightings per bucket
    // and then every 1024th).
    std::unordered_map<uint64_t, uint64_t> unmappedBuffers;
    void noteUnmappedBuffer(uint64_t address, uint64_t size, bool constant) {
        const uint64_t bucket = address >> 24;
        const uint64_t n = ++unmappedBuffers[bucket];
        const bool tell = n <= 4 || (n & 1023) == 0;
        const std::string why = tell && r.hooks_.explainAddress ? r.hooks_.explainAddress(address) : std::string();
        const std::string detail = tell ? "V# base " + hex(address) + " size " + hex(size) + (constant ? " (constant)" : "") + ": " + why : std::string();
        vpps4_report("buffer-unmapped", ("bucket " + hex(bucket << 24)).c_str(), detail.c_str());
        if (tell) log("vpps4.buffer_unmapped " + detail + " times=" + std::to_string(n));
    }
    bool dropSinkSet = false;
    // A multisampled target lost its resolve: which one (address, size, format) and why, to the census
    // (category "resolve-dropped") and, rate-limited, to the log.
    void noteResolveDropped(TextureId ms, TextureId twin, const char* why) {
        // (counted per texture first; the strings are only built for the first sighting and the powers of two)
        if (vpgpu::probe::enabled() && (pstat.resolveDrops.size() < kProbeMaxEntries || pstat.resolveDrops.count(why))) ++pstat.resolveDrops[why];
        const uint64_t n = ++resolveDropCounts[uint64_t(ms.generation) << 32 | ms.index];
        const bool tell = n == 1 || (n >= 256 && (n & (n - 1)) == 0);
        if (!tell) return;
        const vpgpu::TextureDesc* d = r.device_.resources().get(ms);
        if (!d) return;
        const std::string key = d->label + " " + std::to_string(d->width) + "x" + std::to_string(d->height) + "x" +
                                std::to_string(d->layers) + " " + std::to_string(d->samples) + "xMS fmt" +
                                std::to_string(int(d->format));
        vpps4_report("resolve-dropped", key.c_str(), why);
        log("vpps4.resolve_dropped " + key + " why=" + why + " twin=" + std::to_string(twin ? 1 : 0) + " times=" + std::to_string(n));
    }
    std::unordered_map<uint64_t, uint64_t> resolveDropCounts;
    // The sample layout of the most recent multisampled pass (what an FMASK read describes): coverage
    // samples the title asked for and fragments the host stores. Before any: the app's limit.
    uint8_t msCoverage = 4, msFragments = 4;
    // The draw being recorded renders both eyes through a rate map: its shaders export/read the view.
    bool viewIndexedNow = false;
    bool alphaTestNow = false;  // DB_ALPHA_TO_MASK on a one-sample pass: the pixel shader tests alpha (lift)
    std::map<uint64_t, std::shared_ptr<const vpgpu::RateMap>> rateMaps;
    const std::shared_ptr<const vpgpu::RateMap>& rateMap(uint32_t w, uint32_t h) {
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

    vpgpu::TextureDesc targetDesc(vpgpu::Format f, uint32_t w, uint32_t h, uint8_t samples, uint16_t layers, bool depth,
                                   bool storage, uint64_t address, const char* what) {
        vpgpu::TextureDesc d;
        d.format = f;
        d.width = w;
        d.height = h;
        d.samples = samples;
        d.layers = layers;
        d.usage = vpgpu::usage::sampled | (depth ? vpgpu::usage::depthTarget : vpgpu::usage::colorTarget);
        // No storage usage unless a compute pass writes the target: ShaderWrite turns off Apple's lossless
        // framebuffer compression for the texture. A target written by compute is recreated writable on its
        // first such use (makeWritable).
        if (storage && !depth && samples == 1) d.usage |= vpgpu::usage::storage;
        if (!depth && samples == 1 && r.hooks_.view && !vpgpu::formatInfo(f).depth() && vpgpu::formatInfo(f).blockSize == 1)
            d.usage |= vpgpu::usage::formatView;  // other formats of its texel size may be drawn / sampled through views
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
            if (auto* rp = std::get_if<vpgpu::RenderPass>(&*c))
                if (rp->desc.depth.texture == t.ms) {
                    rp->desc.depth.resolve = t.id;
                    rp->desc.depth.resolveLayer = rp->desc.depth.layer;
                    rp->desc.depth.depthResolve = t.depthMode;  // as the pass would have set it with the companion at hand
                    rp->desc.depth.store = vpgpu::StoreAction::StoreAndResolve;
                    rp->desc.depth.keepSamples = true;  // main drops the samples of StoreAndResolve on tile unless asked; the planner decides
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
        vpgpu::Copy cp;
        cp.kind = vpgpu::CopyKind::TextureToTexture;
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

    // The single-sample picture of a multisampled target lost this frame's resolve to a resolve into another
    // target (see resolve()) and somebody reads it now: resolve the samples into it in a pass of its own.
    void catchUpTwin(Surface& c) {
        if (c.twinStale != r.frame_ || !c.ms || !c.id) return;
        c.twinStale = ~uint64_t(0);
        closePass();
        for (uint16_t l = 0; l < c.staleCount; ++l) {
            vpgpu::RenderPassDesc desc;
            vpgpu::Attachment& a = desc.colors[0];
            a.texture = c.ms;
            a.layer = uint16_t(c.staleFirst + l);
            a.load = vpgpu::LoadAction::Load;
            a.store = vpgpu::StoreAction::Resolve;
            a.resolve = c.id;
            a.resolveLayer = a.layer;
            desc.label = "resolve (catch-up) " + hex(c.address);
            frame.beginRenderPass(desc);
            frame.endRenderPass();
            ++r.stats_.passes;
        }
        life.noteSamplesRead(c.ms, r.frame_);  // (the samples are loaded here: the pass that drew them keeps them)
        note("single-sample picture resolved late: a reader appeared after its resolve was given to another target");
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
        vpgpu::ClearValue value;
        // A clear of the whole target is `whole` (the load action of the next pass). Clears of one plane or
        // of part of a depth target wait too, as in-pass clear draws that open that pass (`rects`, in order,
        // after the whole clear if there is one): a pass of their own would store the target and reload it.
        bool whole = true;
        std::vector<std::shared_ptr<const vpgpu::ClearRects>> rects;
    };
    std::vector<PendingClear> pendingClears;

    // A whole-target clear replaces what waited for the texture (its partial clears included).
    void setPendingClear(PendingClear c) {
        for (PendingClear& p : pendingClears)
            if (p.texture == c.texture) return void(p = c);
        pendingClears.push_back(c);
    }
    void addPendingRects(TextureId texture, TextureId resolved, std::shared_ptr<const vpgpu::ClearRects> rects) {
        for (PendingClear& p : pendingClears)
            if (p.texture == texture) return p.rects.push_back(std::move(rects));
        PendingClear pc;
        pc.texture = texture;
        pc.resolved = resolved;
        pc.depth = true;
        pc.whole = false;
        pc.rects.push_back(std::move(rects));
        pendingClears.push_back(std::move(pc));
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
        vpgpu::RenderPassDesc desc;
        vpgpu::Attachment& a = p.depth ? desc.depth : desc.colors[0];
        a.texture = p.texture;
        a.load = p.whole ? vpgpu::LoadAction::Clear : vpgpu::LoadAction::Load;
        a.clear = p.value;
        a.store = vpgpu::StoreAction::Store;
        if (p.resolved) {
            a.resolve = p.resolved;
            a.store = vpgpu::StoreAction::StoreAndResolve;
            a.keepSamples = true;
        }
        desc.label = "clear";
        if (p.whole && p.rects.empty()) {
            frame.clear(desc);
        } else {
            frame.beginRenderPass(desc);
            for (const auto& rc : p.rects) {
                vpgpu::Draw cd;
                cd.clear = rc;
                frame.draw(std::move(cd));
            }
            frame.endRenderPass();
        }
        ++r.stats_.passes;
    }
    void flushClear(const Surface& c) {
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
        uint16_t baseLayer, layers, baseMip, mips;
        bool operator<(const ViewKey& o) const {
            return std::tie(index, generation, format, type, array, baseLayer, layers, baseMip, mips) <
                   std::tie(o.index, o.generation, o.format, o.type, o.array, o.baseLayer, o.layers, o.baseMip, o.mips);
        }
    };
    std::map<ViewKey, TextureId> views;
    // The view of `s` that `want` describes (null if the host has no views).
    TextureId viewOf(const Surface& s, ViewRequest want) {
        if (!r.hooks_.view) return {};
        want.base = s.id;
        const ViewKey key{s.id.index, s.id.generation, uint8_t(want.format), uint8_t(want.type), want.array, want.baseLayer, want.layers, want.baseMip, want.mips};
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
    // A T# the program samples with compare was bound (to `ref`, which may be empty): the route, the host resource, the fills.
    void probeCompare(const ImageDesc& d, vpgcn::ShaderStage stage, const TexRef& ref) {
        ProbeStats& p = pstat;
        static const char* const routeName[ProbeStats::kCmpRoutes] = {"depth_target_found", "color_target_found", "texture_from_memory", "alias", "other"};
        static const char* const fillName[5] = {"none", "gpu", "cpu", "linear", "unfilled"};
        const uint8_t route = cmpRoute;
        const Surface* s = cmpSurface;
        vpgpu::TextureDesc td;
        const bool resolved = ref.id && r.device_.textureDesc(ref.id, td);
        const bool hostDepth = resolved && vpgpu::formatInfo(td.format).depth();
        const char* host = resolved ? vpgpu::formatInfo(td.format).name : "none";
        const bool mismatch = resolved && !hostDepth;
        const unsigned kind = stage == vpgcn::ShaderStage::Compute ? 1 : 0;
        const uint64_t call = kind ? r.stats_.dispatches : r.stats_.draws;  // (one count per call, not per T#)
        ++p.cmpBinds, ++p.cmpRoute[route];
        if (p.cmpLastDraw[kind] != call) p.cmpLastDraw[kind] = call, ++p.cmpDraws;
        if (!resolved) ++p.cmpUnresolved;
        else if (hostDepth) ++p.cmpHostDepth;
        else ++p.cmpHostNotDepth;
        if (mismatch) {
            ++p.cmpMismatchBinds;
            if (p.cmpLastMismatch[kind] != call) p.cmpLastMismatch[kind] = call, ++p.cmpMismatchDraws;
        }
        const unsigned fill = s ? std::min<unsigned>(s->fillPath, 4) : 0;
        if (s) ++p.cmpFill[fill];
        const uint64_t key = uint64_t(d.dataFormat) | uint64_t(resolved ? uint64_t(td.format) : 0xFFFF) << 8 | uint64_t(route) << 32;
        if (p.cmpHist.size() < kProbeMaxEntries || p.cmpHist.count(key)) {
            ProbeStats::CmpCell& c = p.cmpHist[key];
            ++c.count, c.host = host, c.dfmt = uint8_t(d.dataFormat), c.route = route;
        }
        const uint64_t n = ++p.cmpLines;
        if (n > 40 && n % 600) return;  // (the first 40, then one in 600)
        std::string sel;
        for (uint8_t v : d.dstSel) sel += char('0' + (v & 7));
        std::string line = "VPKENGINE_PROBE CMP n=" + std::to_string(n) + " f=" + std::to_string(r.frame_) + " stage=" + std::to_string(int(stage)) +
                           " route=" + routeName[route] + " expects=depth host=" + host + " host_depth=" + (hostDepth ? "1" : "0") +
                           " mismatch=" + (mismatch ? "1" : "0") + " T#=" + hex(d.address) + " dfmt=" + std::to_string(d.dataFormat) + " nfmt=" +
                           std::to_string(d.numFormat) + " " + std::to_string(d.width) + "x" + std::to_string(d.height) + " type=" +
                           std::to_string(d.type) + " array=" + std::to_string(d.baseArray) + ".." + std::to_string(d.lastArray) + " level=" +
                           std::to_string(d.baseLevel) + ".." + std::to_string(d.lastLevel) + " dst_sel=" + sel;
        if (resolved)
            line += " host_size=" + std::to_string(td.width) + "x" + std::to_string(td.height) + " layers=" + std::to_string(td.layers) + " mips=" +
                    std::to_string(td.mips) + " samples=" + std::to_string(td.samples) + " view=" + (ref.view ? "1" : "0");
        if (s)
            line += std::string(" fill=") + fillName[fill] + (fill == 4 ? std::string("(") + s->fillWhy + ")" : std::string()) + " fills=" +
                    std::to_string(s->fills) + " filled=" + (s->filled ? "1" : "0") + " surface_depth=" + (s->isDepth ? "1" : "0");
        log(line);
    }
    // A storage cube (image load / store / atomics) is declared in the shader as a 2D array of faces
    // (layer = slice * 6 + face): it is bound through a 2D-array view of the cube.
    TexRef storageCubeAsArray(const Surface& s, const ImageDesc& d, vpgpu::Format f, TexRef plain) {
        if (d.type != 11) return plain;
        ViewRequest want;
        want.format = f;
        want.type = vpgpu::TextureType::Tex2D;
        want.baseLayer = 0;
        want.layers = uint16_t(6 * (uint32_t(d.lastArray) / 6 + 1));
        want.array = true;
        // (a surface that already is a 2D array of exactly these faces is bound as it is)
        if (s.type == vpgpu::TextureType::Tex2D && s.layers == want.layers && s.layers > 1 && f == s.format) return plain;
        if (const TextureId v = viewOf(s, want)) return {s.id, v};
        // The shader declares a texture2d_array: a cube (or a single 2D layer) bound there would be wrong, so the
        // draw / dispatch is skipped with the reason instead.
        note("storage cube texture without a 2D-array view: not bound");
        texWhy = "storage cube without a 2D-array view";
        return {};
    }
    // `exact` (optional): cleared when no view could be made and the image itself is returned instead (another format / slices).
    TexRef sampleThrough(const Surface& s, const ImageDesc& d, vpgpu::Format f, bool storage = false, bool* exact = nullptr) {
        f = inOrderOf(f, s.format);
        if (storage && d.type == 11) return storageCubeAsArray(s, d, f, {s.id, {}});
        ViewRequest want;
        want.format = f;
        want.type = d.type == 8 ? vpgpu::TextureType::Tex1D : d.type == 10 ? vpgpu::TextureType::Tex3D
                  : d.type == 11 ? vpgpu::TextureType::Cube : vpgpu::TextureType::Tex2D;
        const bool arrayed = d.type == 12 || d.type == 13 || d.type == 15;
        if (want.type == vpgpu::TextureType::Cube) {
            want.baseLayer = 0;
            want.layers = uint16_t(6 * (uint32_t(d.lastArray) / 6 + 1));
            want.array = want.layers > 6;
        } else {
            want.baseLayer = uint16_t(d.baseArray);
            want.layers = arrayed ? uint16_t(d.lastArray - d.baseArray + 1) : 1;
            want.array = arrayed;
        }
        if (want.type == vpgpu::TextureType::Tex3D) want.baseLayer = 0, want.layers = 1, want.array = false;
        const bool sameLayers = want.type == vpgpu::TextureType::Tex3D ? s.type == want.type
                              : want.baseLayer == 0 && want.layers == s.layers && s.type == want.type &&
                                want.array == (s.layers > 1 && want.type != vpgpu::TextureType::Cube);
        if (f == s.format && sameLayers) return {s.id, {}};
        if (const TextureId v = viewOf(s, want)) return {s.id, v};
        if (exact) *exact = false;
        if (f != s.format) note("image sampled with another format (no view available: its own format)");
        if (!sameLayers) note("image sampled as another slice range or dimension (no view available: whole image)");
        return {s.id, {}};
    }

    // ---- Targets

    // The surface rendered to at `address`. A draw to slice 1 alone and one to slices 0-1 must meet
    // in the same array (CB address is slice 0): the first with at least the layers asked for.
    // Another format of the same texel size at the same address is the same memory: it is drawn
    // through a view format (`viewAs`) of the existing target when the host has views.
    Target* target(std::map<uint64_t, std::vector<Target>>& cache, uint64_t address, vpgpu::Format f, uint32_t w,
                   uint32_t h, uint8_t samples, uint16_t layers, bool depth, uint8_t tile, uint32_t pitch,
                   vpgpu::Format* viewAs = nullptr) {
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
        t.bytes = surfaceBytes(tile, depth ? zPlaneFormat(f) : f, w, h, pitch, layers);
        t.lastUse = r.frame_;
        ++r.stats_.targets;
        ++texGen;
        life.noteRead(t.id ? t.id : t.ms, r.frame_);  // (born now: stored while it warms up)
        surfaceRefs[t.wid] = {uint8_t(depth ? 1 : 0), address, 0};
        auto at = list.begin();
        while (at != list.end() && at->layers > layers) ++at;
        return &*list.insert(at, t);
    }

    // How the guest holds depth target `t` (see planesOf), from the DB state of a draw that writes it: the Z format, whether
    // the plane at its base is followed, and the base of the stencil plane (0: the image has none, or it is not followed).
    // The stencil is a plane of its own in guest memory; another base than before says nothing of the new one.
    void noteDepthPlanes(Target& t, uint8_t zFormat, bool followed, uint64_t stencilBase) {
        t.zFormat = zFormat;
        t.followed = followed;
        if (stencilBase != t.stencilAddress) t.stencilAuthoritative = false;
        t.stencilAddress = stencilBase;
        t.stencilBytes = stencilBase ? surfaceBytes(t.tile, vpgpu::Format::S8Uint, t.width, t.height, 0, t.layers) : 0;
    }
    // The base a draw's depth (or stencil) plane is held at when the read and write bases agree or only one is set (the
    // write base if there is one, as shadPS4 does); 0 if they differ: that draw reads one surface and writes another.
    static uint64_t agreedBase(uint64_t read, uint64_t write) {
        return !write || !read || write == read ? (write ? write : read) : 0;
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

    // The depth surface at `address` the passes drew into last, of any sample count (null: none yet).
    Target* existingDepth(uint64_t address, vpgpu::Format f, uint32_t w, uint32_t h, uint16_t layers, uint8_t tile) {
        Target* best = nullptr;
        auto scan = [&](uint64_t at) {
            auto it = depthTargets.find(at);
            if (it == depthTargets.end()) return;
            for (Target& t : it->second)
                if (t.format == f && t.width == w && t.height == h && t.tile == tile && t.layers >= layers && (!best || t.lastUse >= best->lastUse))
                    best = &t;
        };
        if (auto a = depthAliases.find(address); a != depthAliases.end()) scan(a->second);
        scan(address);
        if (best) best->lastUse = r.frame_;
        return best;
    }

    void depthCopy(const DrawState& d) {
        const DepthTarget& z = d.depth;
        const vpgpu::Format zf = depthFormat(z.zFormat, z.stencilFormat);
        if (zf == vpgpu::Format::Undefined) return skip("depth copy: no depth format");
        const bool depth = z.zFormat && z.forceZDirty && z.forceZValid && z.zWriteAddress && z.zWriteAddress != z.zAddress;
        const uint64_t from = depth ? z.zAddress : z.stencilAddress;
        const uint64_t to = depth ? z.zWriteAddress : z.stencilWriteAddress;
        const uint16_t layers = uint16_t(z.sliceStart + z.sliceCount);
        // The surfaces the passes drew into, at whatever sample count they have (a multisampled depth is copied
        // through its single-sample picture); a new empty 1x one only if nothing was drawn there yet.
        Target* src = existingDepth(from, zf, z.width, z.height, layers, z.tileMode);
        if (!src) src = target(depthTargets, from, zf, z.width, z.height, 1, layers, true, z.tileMode, 0);
        if (!src) return skip("depth copy: target creation failed");
        const uint32_t srcWid = src->wid;
        Target* dst = existingDepth(to, zf, z.width, z.height, layers, z.tileMode);
        if (!dst) dst = target(depthTargets, to, zf, z.width, z.height, 1, layers, true, z.tileMode, 0);
        if (!dst) return skip("depth copy: target creation failed");
        src = static_cast<Target*>(surfaceOf(srcWid));  // (a creation may have moved it)
        if (src->wid == dst->wid) return;  // one texture already answers for both addresses
        refresh(*src, nullptr);
        flushClear(*src);  // (clears still waiting for a pass land before the copy)
        const bool srcZ = src->zAuthoritative, srcStencil = src->stencilAuthoritative;
        flushClear(*dst);
        const uint32_t dstWid = dst->wid;
        const TextureId srcId = src->ms ? depthCompanion(*src) : src->id;
        dst = static_cast<Target*>(surfaceOf(dstWid));
        const TextureId dstId = dst->ms ? depthCompanion(*dst) : dst->id;
        if (!srcId || !dstId) return skip("depth copy: multisampled depth without a single-sample picture");
        if (z.zFormat && z.stencilFormat && (depth != (z.stencilFormat && z.forceStencilDirty && z.forceStencilValid)))
            note("depth copy of one plane copies both");
        closePass();
        vpgpu::Copy c;
        c.kind = vpgpu::CopyKind::TextureToTexture;
        c.srcTexture.texture = srcId;
        c.srcTexture.layer = z.sliceStart;
        c.dstTexture.texture = dstId;
        c.dstTexture.layer = z.sliceStart;
        c.srcTexture.layerCount = c.dstTexture.layerCount = std::max<uint16_t>(z.sliceCount, 1);
        c.srcTexture.width = c.dstTexture.width = z.width;
        c.srcTexture.height = c.dstTexture.height = z.height;
        c.label = "depth copy " + hex(from) + " -> " + hex(to);
        frame.copy(c);
        // (the copy is written back to the write bases like a drawn surface, holding what the source held; the stencil plane
        // only if its copy is forced too)
        if (depth) {
            const bool stencilCopied = z.stencilFormat && z.forceStencilDirty && z.forceStencilValid && z.stencilWriteAddress &&
                                       z.stencilWriteAddress != z.stencilAddress;
            noteDepthPlanes(*dst, z.zFormat, true, stencilCopied ? z.stencilWriteAddress : dst->stencilAddress);
            dst->zAuthoritative = srcZ;
            dst->stencilAuthoritative = stencilCopied && srcStencil;
        } else if (dst->format == vpgpu::Format::S8Uint) {
            dst->followed = true;
            dst->stencilAuthoritative = srcStencil;
        }
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
                const vpgpu::FormatInfo& fi = vpgpu::formatInfo(t.format);
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
    // format of the same texel size (sampled through a view) and a size the target covers. A target is allocated
    // at the CB's pitch and height, so a T# of the picture drawn in it (816x870 in a 896x896 allocation) is
    // smaller. `exact`: only a target of the T#'s own size (a shader sampling with normalised coordinates sees the
    // picture, not the allocation; textureRef copies the corner into a texture of the T#'s size otherwise).
    Target* findTarget(std::map<uint64_t, std::vector<Target>>& cache, const ImageDesc& d, vpgpu::Format f, bool depth,
                       bool anyFormat = false, bool exact = false) {
        auto t = cache.find(d.address);
        if (&cache == &depthTargets)
            if (auto a = depthAliases.find(d.address); a != depthAliases.end() && (t == cache.end() || t->second.empty()))
                t = cache.find(a->second);
        if (t == cache.end() || d.baseLevel != 0) return nullptr;
        const bool arrayed = d.type == 12 || d.type == 13 || d.type == 15;
        const uint32_t lastSlice = d.type == 11 ? 6 * (uint32_t(d.lastArray) / 6 + 1) - 1 : arrayed ? d.lastArray : d.baseArray;
        Target* best = nullptr;
        for (Target& c : t->second) {
            if (exact ? c.width != d.width || c.height != d.height : c.width < d.width || c.height < d.height) continue;
            if (lastSlice >= c.layers) continue;
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
        const vpgpu::Format f = textureFormat(d.dataFormat, d.numFormat);
        if (Target* c = findTarget(colorTargets, d, f, false, true); c && sameTexel(c->format, f)) return {c, false};
        if (Target* c = findTarget(depthTargets, d, f, true, true)) return {c, true};
        return {};
    }

    // How a T# is read: its format plan, adapted to the render target it names when there is one (a
    // target is read as it is stored: the plan's conversion is adjusted, no view is made).
    TexturePlan planFor(const ImageDesc& d, bool storage, bool& depthTarget) {
        TexturePlan p = texturePlan(d.dataFormat, d.numFormat, storage);
        depthTarget = false;
        const Bound b = p.softBc ? Bound{} : boundTarget(d);
        if (!b.target) return p;
        const vpgpu::Format tf = b.target->format;
        if (b.depth) {
            depthTarget = true;
            p = TexturePlan{};
            p.format = tf;
            return p;
        }
        if (tf == p.format) return p;
        const vpgpu::FormatInfo& ft = vpgpu::formatInfo(tf);
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

    // The texture a T# names: a render target at that address (matching format, slices and level),
    // else the texture of that shape, filled from guest memory (after images ahead of memory there
    // were written back) whenever another writer wrote its range. `write`: a shader writes it.
    // Why the last textureRef gave nothing (the skip reason of the draw says it).
    std::string texWhy;
    static const char* invalidTexture(const vpgpu::TextureDesc& td) {
        if (!td.width || !td.height) return "zero size";
        if (td.type == vpgpu::TextureType::Cube && (td.layers % 6 || td.width != td.height)) return "cube not square or without 6 faces";
        if (td.mips > std::bit_width(std::max({td.width, td.height, td.type == vpgpu::TextureType::Tex3D ? td.depth : 1u})))
            return "more mip levels than the size allows";
        return "refused by the device";
    }
    // A sampled T# smaller than the render target it names (the picture drawn in a larger allocation): a texture of the
    // T#'s own size holding the corner of the target (a GPU copy, redone when the target is written again), so the
    // shader's normalised coordinates span the picture. Empty when it cannot be done (another format of the target,
    // no single-sample picture, ...): the caller then samples the target itself.
    TexRef mirrorTarget(const ImageDesc& desc, uint32_t wid, bool depth, bool msaa, vpgpu::Format f) {
        Target* c = static_cast<Target*>(surfaceOf(wid));
        if (!c || !c->width || inOrderOf(f, c->format) != c->format) return {};
        // (a multisampled T# copies samples: no host copy between multisampled textures yet, the target itself is sampled)
        if (msaa && c->ms) return {};
        TextureId src = c->id;
        if (depth && c->ms) src = depthCompanion(*c);
        if (!src) return {};
        // (a pending clear, the twin's resolve and a refill from memory write it: before the question whether it holds anything)
        flushClear(*c);
        if (!depth) catchUpTwin(*c);
        refresh(*c, nullptr);
        // A target nothing wrote (no pass drew into it, no resolve, compute or fill wrote it) holds nothing to copy: a
        // made-up multisampled one is empty. The caller samples the target itself.
        if (mirrorProbe.due(r.frame_))
            log("vpps4.probe P5 mirror f=" + std::to_string(r.frame_) + " desc=" + hex(desc.address) + " " + std::to_string(desc.width) + "x" +
                std::to_string(desc.height) + " target=" + std::to_string(c->width) + "x" + std::to_string(c->height) + " wid=" +
                std::to_string(c->wid) + " last_draw=" + (c->lastDraw == ~uint64_t(0) ? std::string("never") : std::to_string(c->lastDraw)) +
                " write_seq=" + std::to_string(c->writeSeq) + " src=" + std::to_string(src.index) + (depth ? " depth" : ""));
        if (!depth && c->lastDraw == ~uint64_t(0) && !c->writeSeq) return {};
        ++pstat.mirrorCalls;
        const uint8_t samples = 1;
        Hash h;
        h.value(uint32_t(0x6d697272));
        h.value(c->wid);
        h.value(desc.width);
        h.value(desc.height);
        h.value(samples);
        auto& atAddress = textures[desc.address];
        auto found = atAddress.find(h.h);
        if (found == atAddress.end()) {
            const TextureId id = r.device_.createTexture(
                targetDesc(c->format, desc.width, desc.height, samples, c->layers, depth, false, desc.address, " sampled corner"));
            if (!id) {
                if (atAddress.empty()) textures.erase(desc.address);
                return {};
            }
            Texture& m = atAddress[h.h];
            m.wid = tracker.newImage();
            surfaceRefs[m.wid] = {2, desc.address, h.h};
            m.id = id;
            m.format = c->format;
            m.isDepth = depth;
            m.width = desc.width;
            m.height = desc.height;
            m.layers = c->layers;
            m.samples = samples;
            m.address = desc.address;
            m.bytes = 0;  // (never filled from memory: the target is its source)
            m.filled = true;
            ++texGen;
        }
        Texture& m = atAddress[h.h];
        m.lastUse = c->lastUse = r.frame_;
        if (m.mirrorSrc != src || m.mirrorVersion != c->writeSeq) {
            closePass();
            vpgpu::Copy cp;
            cp.kind = vpgpu::CopyKind::TextureToTexture;
            cp.srcTexture.texture = src;
            cp.dstTexture.texture = m.id;
            cp.srcTexture.layerCount = cp.dstTexture.layerCount = c->layers;
            cp.srcTexture.width = cp.dstTexture.width = std::min(desc.width, c->width);
            cp.srcTexture.height = cp.dstTexture.height = std::min(desc.height, c->height);
            cp.label = "sampled corner " + hex(desc.address);
            frame.copy(cp);
            ++r.stats_.copies;
            ++pstat.mirrorCopies;
            pstat.mirrorBytes += uint64_t(cp.srcTexture.width) * cp.srcTexture.height * c->layers *
                                 std::max<uint32_t>(vpgpu::formatInfo(c->format).bytesPerBlock, 1);
            m.mirrorSrc = src;
            m.mirrorVersion = c->writeSeq;
        }
        life.noteRead(src, r.frame_);
        life.noteRead(m.id, r.frame_);
        if (depth) return {m.id, {}};
        return sampleThrough(m, desc, f, false);
    }
    // A sampled T# whose memory an image of the renderer wrote last, with exactly the layout of the T# (same address, size,
    // slices, tiling, pitch, levels and texel size), is read through that image: it holds the newest bytes, and writing it
    // back to memory only to fill a texture from it is a round trip (a retile, a detile and the GPU waits of raw reads) at
    // every use. Another numeric class of the same texel size (a float target read as integers) goes through a view. Where
    // anything differs, or another writer touched the range after the image, or the image is not whole (depth, samples), the
    // texture is filled from memory as always (`aliasRejected` says why). Nothing is cached: the checks run at every use, so a
    // write of the CPU / a DMA / another image, or the image's eviction (written back first), ends the alias at its next use.
    bool aliasWriter(const ImageDesc& desc, const vpgpu::TextureDesc& td, const TexturePlan& plan, const Texture* own, vpgpu::Format f,
                     TexRef& out) {
        if (!imageAlias || r.hooks_.seed) return false;  // (a host with seeds holds the texels of its surfaces: refresh() takes them)
        const uint32_t wid = tracker.writerAt(desc.address);
        if (!MemoryTracker::isImage(wid) || (own && wid == own->wid)) return false;
        Surface* w = surfaceOf(wid);
        // (no image of ours behind the memory, or the texture already holds what it wrote)
        if (!w || !w->id || w->version <= (own && own->filled ? own->version : 0)) return false;
        const auto reject = [&](AliasReject why) { return ++pstat.aliasRejected[why], false; };
        if (w->isDepth) return reject(kRejDepth);
        if (!w->filled) return false;
        if (w->address != desc.address) return reject(kRejAddress);
        if (w->samples > 1 || w->ms || w->memSamples != 1 || desc.type == 14 || desc.type == 15) return reject(kRejSamples);
        if (w->type != td.type || w->width != td.width || w->height != td.height || w->depth != td.depth || w->layers != td.layers ||
            desc.baseArray)
            return reject(kRejShape);
        if (w->tile != desc.tilingIndex || std::max(w->pitch, w->width) != std::max(desc.pitch, td.width) ||
            (td.layers > 1 && w->pow2Pad != desc.pow2Pad))
            return reject(kRejLayout);
        if (w->mips != td.mips || desc.baseLevel) return reject(kRejMips);
        const vpgpu::FormatInfo &fw = vpgpu::formatInfo(w->format), &ff = vpgpu::formatInfo(f);
        if (w->softBc || plan.softBc || plan.widthDivisor > 1 || fw.depth() || ff.depth() || fw.blockSize != 1 || ff.blockSize != 1 ||
            fw.bytesPerBlock != ff.bytesPerBlock || bgra(f) != bgra(w->format))  // (the channel order of a view is not compensated here, as planFor does for a bound target)
            return reject(kRejFormat);
        // (the image is attached to the pass of the draw that reads it: the way through memory, as before)
        if (std::find(attachedWids.begin(), attachedWids.end(), w->wid) != attachedWids.end()) return reject(kRejFeedback);
        if (tracker.writtenByOthers(w->address, w->bytes, w->wid, w->version)) return reject(kRejRange);
        if (!r.hooks_.view && inOrderOf(f, w->format) != w->format) return reject(kRejNoView);
        flushClear(*w);  // (a clear that waits for the next pass lands first, as for a target a T# names)
        bool exact = true;
        const TexRef ref = sampleThrough(*w, desc, f, false, &exact);
        if (!exact) return reject(kRejNoView);
        w->lastUse = r.frame_;
        life.noteRead(w->id, r.frame_);
        ++pstat.aliasHits;
        routed(ProbeStats::kCmpAlias, *w);
        probeSampled(desc, *w, "image written by the GPU (T# of the same memory, aliased)", false);
        out = ref;
        return true;
    }
    std::vector<uint32_t> attachedWids;  // the images the draw being recorded renders to (colour and depth)
    TexRef textureRef(const ImageDesc& desc, bool storage, bool write, bool integerOnly = false) {
        texWhy.clear();
        bool depthTarget = false;
        const TexturePlan plan = planFor(desc, storage, depthTarget);
        const vpgpu::Format f = plan.format;
        if (f == vpgpu::Format::Undefined) {
            note("texture format " + std::to_string(desc.dataFormat) + "/" + std::to_string(desc.numFormat) + " unsupported" +
                 (plan.why ? std::string(" (") + plan.why + ")" : std::string()));
            texWhy = "format " + std::to_string(desc.dataFormat) + "/" + std::to_string(desc.numFormat) + " unsupported";
            return {};
        }
        // A sampled T# that reads normalised coordinates sees a picture of its own size (see mirrorTarget); an integer
        // fetch or a storage access addresses the target's texels, which the target covers.
        const bool ownSize = !storage && !integerOnly;
        Target* ct = nullptr;
        if (!plan.softBc) {
            if (ownSize) ct = findTarget(colorTargets, desc, f, false, false, true);
            if (!ct) ct = findTarget(colorTargets, desc, f, false, false);
        }
        if (Target* c = ct) {
            routed(ProbeStats::kCmpColorTarget, *c);
            if (ownSize && (c->width != desc.width || c->height != desc.height)) {
                const uint32_t wid = c->wid;
                if (const TexRef m = mirrorTarget(desc, wid, false, msaaImage(desc), f); m.id) return m;
                c = static_cast<Target*>(surfaceOf(wid));
                note("T# smaller than its render target sampled through the whole target");
            }
            c->lastUse = r.frame_;
            flushClear(*c);
            if (!storage && msaaImage(desc) && c->ms) {  // the shader reads the samples themselves
                life.noteSamplesRead(c->ms, r.frame_);
                return {c->ms, {}};
            }
            catchUpTwin(*c);
            if (storage && !c->writable) makeWritable(*c);
            refresh(*c, nullptr);
            probeSampled(desc, *c, "colour target", storage);
            if (write) markWrite(*c);
            life.noteRead(c->id, r.frame_);
            return sampleThrough(*c, desc, f, storage);
        }
        Target* dt = nullptr;
        if (!plan.softBc) {
            if (ownSize) dt = findTarget(depthTargets, desc, f, true, false, true);
            if (!dt) dt = findTarget(depthTargets, desc, f, true, false);
        }
        if (Target* c = dt) {
            routed(ProbeStats::kCmpDepthTarget, *c);
            if (ownSize && (c->width != desc.width || c->height != desc.height)) {
                const uint32_t wid = c->wid;
                if (const TexRef m = mirrorTarget(desc, wid, true, msaaImage(desc), f); m.id) return m;
                c = static_cast<Target*>(surfaceOf(wid));
                note("T# smaller than its depth target sampled through the whole target");
            }
            c->lastUse = r.frame_;
            flushClear(*c);
            if (!storage && msaaImage(desc) && c->ms) {
                life.noteSamplesRead(c->ms, r.frame_);
                return {c->ms, {}};
            }
            const TextureId id = c->ms ? depthCompanion(*c) : c->id;
            if (!id) return texWhy = "depth target without a single-sample picture", TexRef{};
            refresh(*c, nullptr);
            probeSampled(desc, *c, "depth target", storage);
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

        vpgpu::TextureDesc td;
        td.format = f;
        td.width = d.width;
        td.height = d.height;
        if (d.type == 12) note("1D array texture made a 2D array of height 1");
        td.type = d.type == 8                 ? vpgpu::TextureType::Tex1D
                : d.type == 10               ? vpgpu::TextureType::Tex3D
                : d.type == 11               ? vpgpu::TextureType::Cube
                                             : vpgpu::TextureType::Tex2D;
        if (td.type == vpgpu::TextureType::Tex1D || d.type == 12) td.height = 1;
        if (td.type == vpgpu::TextureType::Tex3D) td.depth = d.depth;
        const bool arrayed = d.type == 12 || d.type == 13 || d.type == 15;
        const uint32_t layers = arrayed ? uint32_t(d.lastArray) + 1 : 1;
        td.layers = uint16_t(td.type == vpgpu::TextureType::Cube ? 6 * (uint32_t(d.lastArray) / 6 + 1) : layers);
        td.mips = textureMips(d, td);
        td.usage = vpgpu::usage::sampled | (storage ? vpgpu::usage::storage : 0);
        if (r.hooks_.view && vpgpu::formatInfo(f).blockSize == 1 && !vpgpu::formatInfo(f).depth())
            td.usage |= vpgpu::usage::formatView;
        td.label = "tex " + hex(d.address);

        Hash h;
        h.value(f);
        if (plan.softBc) h.value(uint32_t(plan.softBc) << 1 | plan.softSigned);  // (BC1 / BC2 / BC3 / BC7 decode to the same host format)
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
        // A shader wrote an image at this address through a T# of another format with the same
        // texel size (typically a compute pass writing UNORM storage that is then read as sRGB:
        // Metal has no sRGB storage). Memory never held that picture, and re-tiling it into another
        // layout would round-trip through a format conversion: this T# sees the writer's image
        // through a view of its format (the writer's own format without views).
        if (!storage && !plan.softBc) {
            auto own = atAddress.find(h.h);
            const uint64_t ownVersion = own == atAddress.end() || !own->second.filled ? 0 : own->second.version;
            for (auto& [key, other] : atAddress)
                if (key != h.h && other.id && other.format != f && other.filled &&
                    other.version > ownVersion &&
                    tracker.writerAt(d.address) == other.wid && other.type == td.type && other.width == td.width &&
                    other.height == td.height && other.depth == td.depth && other.layers == td.layers && sameTexel(other.format, f)) {
                    other.lastUse = r.frame_;
                    ++pstat.aliasHitsLegacy;
                    routed(ProbeStats::kCmpAlias, other);
                    probeSampled(desc, other, "texture written by the GPU (T# of another format)", storage);
                    if (inOrderOf(f, other.format) == other.format) return {other.id, {}};  // (channel order only)
                    ViewRequest want;
                    want.format = inOrderOf(f, other.format);
                    want.type = td.type;
                    want.layers = td.layers;
                    want.array = td.layers > 1 && td.type != vpgpu::TextureType::Cube;
                    if (const TextureId v = viewOf(other, want)) return {other.id, v};
                    note("GPU-written image sampled with another format of the same size (its own format)");
                    return {other.id, {}};
                }
            TexRef alias;
            if (aliasWriter(desc, td, plan, own == atAddress.end() ? nullptr : &own->second, f, alias)) return alias;
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
            if (!tex.id) return texWhy = invalidTexture(td), TexRef{};
            tex.storage = storage;
            tex.format = f;
            tex.layout = plan.softBc ? plan.layout : vpgpu::Format::Undefined;
            tex.softBc = plan.softBc;
            tex.softSigned = plan.softSigned;
            tex.type = td.type;
            tex.width = td.width;
            tex.height = td.height;
            tex.depth = td.depth;
            tex.layers = td.layers;
            tex.mips = td.mips;
            tex.pow2Pad = d.pow2Pad;
            tex.memSamples = d.type == 14 || d.type == 15 ? uint8_t(1u << std::min<uint32_t>(d.lastLevel, 3)) : 1;
            tex.tile = d.tilingIndex;
            tex.dfmt = uint8_t(d.dataFormat), tex.nfmt = uint8_t(d.numFormat);
            tex.pitch = d.pitch;
            tex.address = d.address;
            // (the guest range follows from the fields the key hashes: worked out when the texture is made, not at every draw)
            const vpgpu::Format guestFormat = plan.softBc ? plan.layout : f;  // (what the guest memory holds)
            const SurfaceLayout fullLayout = computeLayout(tiledSurfaceOf(d, td, vpgpu::formatInfo(guestFormat)));
            tex.bytes = fullLayout.ok ? fullLayout.bytes
                                      : surfaceBytes(d.tilingIndex, guestFormat, td.width, td.height, d.pitch,
                                                     td.type == vpgpu::TextureType::Tex3D ? td.depth : td.layers);
            tex.version = 0;
            tex.filled = false;
            ++r.stats_.textures;
            ++texGen;
            if (vpgpu::probe::enabled()) {
                const bool srgbAsUnorm = plan.conv.srgb && !vpgpu::formatInfo(f).srgb();
                const uint8_t flags = uint8_t((srgbAsUnorm ? 1 : 0) | (plan.exact ? 0 : 2) | (plan.softBc ? 4 : 0) | (storage ? 8 : 0));
                const uint64_t planKey = uint64_t(d.dataFormat) | uint64_t(d.numFormat) << 8 | uint64_t(f) << 16 | uint64_t(flags) << 24;
                if (pstat.texPlans.size() < kProbeMaxEntries || pstat.texPlans.count(planKey)) {
                    ProbeStats::Plan& pl = pstat.texPlans[planKey];
                    ++pl.count;
                    pl.host = vpgpu::formatInfo(f).name, pl.dfmt = uint8_t(d.dataFormat), pl.nfmt = uint8_t(d.numFormat), pl.flags = flags;
                }
                pstat.texSrgbAsUnorm += srgbAsUnorm, pstat.texInexact += !plan.exact, pstat.texSoftBc += plan.softBc != 0;
                probeTile(tex, fullLayout, false);
            }
        }
        tex.lastUse = r.frame_;
        fillDeferred = nullptr;
        refresh(tex, &tex);
        if (fillDeferred) return texWhy = fillDeferred, TexRef{};
        routed(ProbeStats::kCmpMemory, tex);
        if (probeOpen())
            probeSampled(desc, tex, tex.filled && tracker.writerAt(tex.address) == tex.wid ? "texture written by the GPU" : "texture from memory", storage);
        if (write) markWrite(tex);
        if (storage) {
            if (d.baseLevel) note("T# base level ignored: the texture holds the whole mip chain from its address");
            return storageCubeAsArray(tex, d, f, {tex.id, {}});
        }
        // Sampling starts at the T#'s base level (the chain starts at its address): a view of the levels base..last, so
        // the shader's level 0 and its level of detail count from the base, as on the hardware.
        if (d.baseLevel) {
            if (d.baseLevel < td.mips) {
                ViewRequest want;
                want.format = f;
                want.type = td.type;
                want.layers = td.layers;
                want.array = td.layers > 1 && td.type != vpgpu::TextureType::Cube;
                want.baseMip = d.baseLevel;
                want.mips = uint16_t(td.mips - d.baseLevel);
                if (const TextureId v = viewOf(tex, want)) return {tex.id, v};
            }
            note("T# base level ignored: the texture holds the whole mip chain from its address");
        }
        return {tex.id, {}};
    }

    // Mip levels a T# names (its chain starts at its address; MSAA and 1D textures have one), as
    // many as the sizes allow.
    static uint16_t textureMips(const ImageDesc& d, const vpgpu::TextureDesc& td) {
        if (td.type == vpgpu::TextureType::Tex1D || d.type == 12 || d.type == 14 || d.type == 15) return 1;
        const uint32_t largest = std::max({td.width, td.height, td.type == vpgpu::TextureType::Tex3D ? td.depth : 1u});
        return uint16_t(std::min<uint32_t>(uint32_t(d.lastLevel) + 1, std::bit_width(largest)));
    }

    // The surface in guest memory that a T# and the texture made of it describe.
    static TiledSurface tiledSurfaceOf(const ImageDesc& d, const vpgpu::TextureDesc& td, const vpgpu::FormatInfo& fi) {
        TiledSurface s;
        s.tileIndex = d.tilingIndex;
        s.bytesPerElement = fi.bytesPerBlock;
        s.blockSize = std::max<uint32_t>(fi.blockSize, 1);
        s.width = td.width;
        s.height = td.height;
        s.pitch = std::max(d.pitch, td.width);
        s.volume = td.type == vpgpu::TextureType::Tex3D;
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
        bool writtenBack = false;
        if (ownsStale(*s)) {
            writtenBack = retileToMemory(*s);
            if (writtenBack)
                tracker.markSynced(wid);
            else {
                probeFailure(retileWhy, *s, s->bytes);
                note("image newer than memory evicted without a write back");
            }
        }
        watchOwned(*s, false, writtenBack);
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
                if (!c.bytes) continue;  // (holds no GPU memory: nothing to gain)
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
        uint64_t size = 0;   // dedicated buffers only (counted against stagingLimit)
        uint64_t pooled = 0; // from the pool of this size class (power of two): back to it when the GPU is done, not destroyed
    };
    std::vector<Staging> stagings;
    // Dedicated staging up to kPoolMaxClass is served from buffers kept by size class (powers of two) instead of an aligned_alloc
    // + createBuffer per request (a no-copy MTLBuffer each time: 260 of 21.7 MB in 2 s). A buffer returns to its class when its
    // frame is done (where a dedicated one was destroyed) and is destroyed after kPoolIdle submits unused or past kPoolMaxBytes
    // kept. VPKENGINE_NO_STAGING_POOL=1: one buffer per request, as before.
    struct Pooled {
        void* memory;
        BufferId id;
        uint64_t cls, since;
    };
    std::vector<Pooled> stagingPool;
    uint64_t stagingPoolHits = 0, stagingPoolMisses = 0;
    bool stagingPoolOn = !std::getenv("VPKENGINE_NO_STAGING_POOL");
    static constexpr uint64_t kPoolMaxClass = 64ull << 20, kPoolMaxBytes = 192ull << 20;
    static constexpr uint64_t kPoolIdle = 120;
    static uint64_t poolClass(uint64_t size) {
        uint64_t c = kPage;
        while (c < size) c <<= 1;
        return c;
    }
    void poolTrim() {
        uint64_t kept = 0;
        for (const Pooled& x : stagingPool) kept += x.cls;
        std::erase_if(stagingPool, [&](const Pooled& x) {
            if (x.since + kPoolIdle > r.frame_ && kept <= kPoolMaxBytes) return false;
            kept -= x.cls;
            r.device_.destroy(x.id);
            std::free(x.memory);
            return true;
        });
    }
    uint64_t stagingLive() const {
        uint64_t live = 0;
        for (const Staging& x : stagings) live += x.size;
        return live;
    }
    // Dedicated staging (a copy source larger than half the ring: decoded block-compressed textures, big uploads) is kept
    // until the GPU is done with the frame; past this much alive a request is refused (the image stays unfilled and is
    // tried again at its next use, when earlier frames released theirs). A host without BC decodes every block-compressed
    // texture into it (up to 8x the guest bytes): less of it (Renderer::setMemoryLimits).
    uint64_t stagingLimit = 512ull << 20;
    uint8_t* staging(uint64_t bytes, BufferId& id, uint64_t& offset) {
        Ring& ring = rings[ringIndex];
        if (bytes <= ring.cap / 2) {
            ensureRing();
            const uint64_t at = (ring.head + 255) / 256 * 256;
            if (ring.memory && at + bytes <= ring.cap) {
                ring.head = at + bytes;
                id = ring.id;
                offset = at;
                return static_cast<uint8_t*>(ring.memory) + at;
            }
        }
        const uint64_t size = (bytes + kPage - 1) / kPage * kPage;
        uint64_t live = 0;
        for (const Staging& x : stagings) live += x.size;
        if (live + size > stagingLimit) {
            note("staging memory limit reached: copy source refused");
            return nullptr;
        }
        const uint64_t cls = stagingPoolOn && size <= kPoolMaxClass ? poolClass(size) : 0;
        if (cls) {
            auto hit = std::find_if(stagingPool.begin(), stagingPool.end(), [&](const Pooled& x) { return x.cls == cls; });
            if (hit != stagingPool.end()) {
                ++stagingPoolHits;
                stagings.push_back({hit->memory, hit->id, r.frame_ + kFrames + 2, size, cls});
                stagingPool.erase(hit);
                id = stagings.back().id;
                offset = 0;
                return static_cast<uint8_t*>(stagings.back().memory);
            }
            ++stagingPoolMisses;
        }
        void* memory = std::aligned_alloc(kPage, cls ? cls : size);
        if (!memory) return nullptr;
        Staging st{memory, r.device_.createBuffer({cls ? cls : size, memory, "vpps4 staging"}), r.frame_ + kFrames + 2, size, cls};
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

    // Scratch of the ES/GS rings: a bump allocator over large slabs, one set of slabs per submit slot like the
    // upload ring. A draw's three buffers are three slices of the current slab (no buffer, page rounding or pool
    // search per buffer); the slabs of a slot are rewound when the slot comes round again kFrames submits later,
    // when the frames that used it are over. A request bigger than a slab gets a slab of its own that is released
    // at the rewind.
    struct ScratchSlab {
        void* memory;
        BufferId id;
        uint64_t size, head;
    };
    std::array<std::vector<ScratchSlab>, kFrames> scratchSlots;
    std::array<uint64_t, kFrames> scratchDone{};  // slot -> 1 + the device frame its slabs were last used by (0: none)
    uint64_t scratchBytes = 0;
    static constexpr uint64_t kScratchSlab = 32ull << 20;
    uint64_t scratchLimit = 768ull << 20;  // (Renderer::setMemoryLimits)
    bool scratch(uint64_t bytes, RingBinding& out) {
        bytes = std::max<uint64_t>((bytes + 255) / 256 * 256, 256);
        std::vector<ScratchSlab>& slabs = scratchSlots[ringIndex];
        for (ScratchSlab& s : slabs)
            if (s.size - s.head >= bytes) {
                out = {s.id, s.head, bytes};
                s.head += bytes;
                return true;
            }
        const uint64_t size = bytes > kScratchSlab ? (bytes + kPage - 1) / kPage * kPage : kScratchSlab;
        if (scratchBytes + size > scratchLimit) reclaimScratch();
        if (scratchBytes + size > scratchLimit) return false;
        void* memory = std::aligned_alloc(kPage, size);
        if (!memory) return false;
        const BufferId id = r.device_.createBuffer({size, memory, "vpps4 geometry scratch"});
        if (!id) {
            std::free(memory);
            return false;
        }
        scratchBytes += size;
        slabs.push_back({memory, id, size, bytes});
        out = {id, 0, bytes};
        return true;
    }
    // The slabs of the submitted frames the GPU has finished with are released now, not when their slot comes round: slabs
    // kept by an earlier heavy frame would otherwise eat the limit for every later frame.
    void reclaimScratch() {
        const uint64_t done = r.device_.completedFrames();
        for (unsigned slot = 0; slot < kFrames; ++slot) {
            if (slot == ringIndex || !scratchDone[slot] || scratchDone[slot] > done) continue;
            for (ScratchSlab& slab : scratchSlots[slot]) {
                r.device_.destroy(slab.id);
                std::free(slab.memory);
                scratchBytes -= slab.size;
            }
            scratchSlots[slot].clear();
            scratchDone[slot] = 0;
        }
    }
    // The slot's slabs are free again: rewind them, release the oversized ones.
    void rewindScratch(unsigned slot) {
        std::vector<ScratchSlab>& slabs = scratchSlots[slot];
        for (size_t k = 0; k < slabs.size();) {
            if (slabs[k].size > kScratchSlab) {
                r.device_.destroy(slabs[k].id);
                std::free(slabs[k].memory);
                scratchBytes -= slabs[k].size;
                slabs.erase(slabs.begin() + long(k));
            } else {
                slabs[k++].head = 0;
            }
        }
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
    bool guestMap(const Stage& s, vpgpu::StageMask mask, std::vector<Access>& out, uint64_t& at, std::string& why) {
        for (unsigned k = 0; k + 1 < s.userData.size(); ++k) {
            const uint64_t va = s.userData[k] | uint64_t(s.userData[k + 1]) << 32;
            GuestRegion g;
            if (va < 0x10000 || !region(va, 1, g)) continue;
            BufferId id;
            uint64_t offset = 0, available = 0;
            guestBuffer(va, 1, id, offset, available);
        }
        // A window over a mirror the host has replaced is left out (and retired). The table must be sorted by base and
        // disjoint for the shader's search (the last base <= the pointer): by base, the larger first, and a window inside
        // the ones before it is dropped (windows made over a boundary overlap those cut at it).
        std::vector<const GuestBuffer*> live;
        for (auto it = guestBuffers.begin(); it != guestBuffers.end();) {
            GuestBuffer& b = it->second;
            if (b.id && windowSuperseded(b)) {
                retired.push_back({b.id, r.frame_ + kFrames});
                it = guestBuffers.erase(it);
                continue;
            }
            if (b.id) live.push_back(&b);
            ++it;
        }
        std::sort(live.begin(), live.end(), [](const GuestBuffer* a, const GuestBuffer* b) {
            return a->base != b->base ? a->base < b->base : a->size > b->size;
        });
        struct Entry {
            const GuestBuffer* b;
            uint64_t base, size, dev;
        };
        std::vector<Entry> entries;
        uint64_t reach = 0;
        for (const GuestBuffer* b : live) {
            if (b->base + b->size <= reach) continue;
            reach = b->base + b->size;
            const uint64_t dev = r.device_.bufferAddress(b->id);
            if (!dev) return why = "FLAT access needs buffer device addresses", false;
            entries.push_back({b, b->base, b->size, dev});
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
        for (const Entry& e : entries) {
            Access a;
            a.stages = mask;
            a.kind = vpgpu::BindKind::Buffer;
            a.slot = 0xffff;  // (reachable by address only: no shader slot)
            a.write = true;
            a.range = ResourceRange::bytes(e.b->id, 0, e.size);
            touch(e.base, e.size);
            out.push_back(a);
        }
        return true;
    }

    // ---- Bindings ---------------------------------------------------------------------------------

    // Adds the time of its scope to one of the bind() phase counters (CpuTimes).
    struct Phase {
        double& acc;
        Clock::time_point t0 = Clock::now();
        explicit Phase(double& a) : acc(a) {}
        ~Phase() { acc += msSince(t0); }
    };

    // Texture bindings that bind a view of their texture (collected by bind(), taken by the draw).
    std::vector<vpgpu::ViewBinding> pendingViews;

    // Probes (Renderer::setProbes): off, none of this runs.
    bool probeOn = false;
    std::vector<Renderer::DrawProbe> probes;
    struct ProbePs {
        bool valid = false;
        ImageDesc image;
    } probePs;  // the first texture of the pixel shader of the draw being built

    // Host memory of a ring binding living in a normal-size scratch slab of this slot (null otherwise: the
    // oversized slabs are freed at the rewind).
    const uint32_t* ringHost(const RingBinding& rb) const {
        for (const ScratchSlab& s : scratchSlots[ringIndex])
            if (s.id == rb.id && s.size <= kScratchSlab)
                return reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(s.memory) + rb.offset);
        return nullptr;
    }

    // A drawn call leaves its probe record (per colour target address; a later draw replaces what it sets).
    void noteProbe(uint64_t colorAddress, bool geom, uint32_t maxVertOut, uint64_t gsIndices) {
        Renderer::DrawProbe* p = nullptr;
        for (Renderer::DrawProbe& x : probes)
            if (x.colorAddress == colorAddress) {
                p = &x;
                break;
            }
        if (!p) {
            if (probes.size() >= 64) return;
            p = &probes.emplace_back();
            p->colorAddress = colorAddress;
        }
        if (probePs.valid) {
            p->hasPsTexture = true;
            p->psTexture = probePs.image;
        }
        if (geom) {
            const RingBinding& gi = ringBinds[size_t(vpgcn::GeomRing::GsIndex)];
            const RingBinding& gv = ringBinds[size_t(vpgcn::GeomRing::GsVs)];
            p->geom = true;
            p->gsIndex = ringHost(gi);
            p->gsIndexCount = gsIndices;
            p->gsVs = ringHost(gv);
            p->gsVsDwords = gv.size / 4;
            p->maxVertOut = maxVertOut;
        }
    }

    // The accesses of one translated stage, with the draw's real resources. False (with `why`) if a
    // resource cannot be resolved: the draw is skipped.
    bool bind(const Stage& s, const Translated& t, vpgcn::ShaderStage stage, vpgpu::StageMask mask, std::vector<Access>& out,
              std::string& why) {
        const std::vector<Snap>& snap = snaps[size_t(stage)];
        uint32_t own[8];
        out.reserve(out.size() + t.bindings.size() + 1);
        for (size_t bi = 0; bi < t.bindings.size(); ++bi) {
            const vpgcn::Binding& b = t.bindings[bi];
            // The words shader() already read for this binding, or read now.
            const uint32_t* words = own;
            auto load = [&]() {
                const Phase ph(r.cpu_.bindDescMs);
                const int at = bi < t.snapIndex.size() ? t.snapIndex[bi] : -1;
                if (at >= 0 && size_t(at) < snap.size()) {
                    words = snap[at].w.data();
                    return snap[at].ok;
                }
                words = own;
                return b.dwords <= 8 && readDescriptor(s, b.source, b.dwords, own);
            };
            const vpir::Resource& res = t.resources.at(b.slot);
            Access a;
            a.stages = mask;
            a.kind = bindKind(res.kind);
            a.slot = uint16_t(res.index);
            a.write = res.access != vpir::Access::Read &&
                      (res.kind == vpir::ResourceKind::Buffer || res.kind == vpir::ResourceKind::StorageTexture);
            switch (res.kind) {
                case vpir::ResourceKind::Buffer:
                case vpir::ResourceKind::ConstantBuffer: {
                    if (b.source.kind == vpgcn::DescriptorSource::Ring) {
                        const RingBinding& rb = ringBinds[b.source.sgpr % ringBinds.size()];
                        if (!rb.id) return why = "geometry ring unavailable", false;
                        a.range = ResourceRange::bytes(rb.id, rb.offset, rb.size);
                        break;
                    }
                    if (b.source.kind == vpgcn::DescriptorSource::GuestMap) {
                        uint64_t at = 0;
                        const Phase ph(r.cpu_.bindUploadMs);
                        if (!guestMap(s, mask, out, at, why)) return false;
                        a.range = ResourceRange::bytes(rings[ringIndex].id, at, vpgcn::kGuestMapBytes);
                        break;
                    }
                    if (b.source.kind == vpgcn::DescriptorSource::Gds) {
                        if (!gdsBuffer()) return why = "no GDS buffer", false;
                        a.range = ResourceRange::bytes(gds.id, 0, kGdsBytes);
                        const Who gdsWho(touchWho, "gds");
                        touch(reinterpret_cast<uint64_t>(gds.memory), kGdsBytes);  // the CPU reads GDS: see gdsAddress()
                        break;
                    }
                    uint64_t address = 0, size = 0;
                    uint32_t stride = 0;
                    const bool pointer = b.dwords == 0 || b.source.kind == vpgcn::DescriptorSource::Code;
                    if (pointer) {
                        // (Code: the base is an address in the program image; its size/format words are
                        // set by the program, so there is no descriptor to read.)
                        if (!memoryAddress(s, b.source, address)) return why = "buffer address unreadable", false;
                    } else {
                        if (!load()) return why = "V# unreadable", false;
                        const BufferDesc d = decodeBuffer({words, b.dwords});
                        address = d.address;
                        size = d.bytes();
                        stride = d.stride;
                    }
                    BufferId id;
                    uint64_t offset = 0, available = 0;
                    bool found = false;
                    {
                        const Phase ph(r.cpu_.bindBufMs);
                        found = address && guestBuffer(address, size, id, offset, available);
                    }
                    if (found && a.write && guestReadOnly) return why = "guest memory read-only", false;
                    if (!found) {
                        // The hardware reads zeros past a buffer's records and drops writes: an unset or
                        // unmapped V# gets a zero buffer of our own instead of losing the whole draw.
                        if (!dummyBuffer()) return why = "buffer outside guest memory", false;
                        note(address ? "buffer outside guest memory bound as zeros" : "null buffer bound as zeros");
                        if (address) noteUnmappedBuffer(address, size, res.kind == vpir::ResourceKind::ConstantBuffer);
                        std::memset(dummy.memory, 0, kDummyBytes);
                        a.range = ResourceRange::bytes(dummy.id, 0, kDummyBytes);
                        if (resolved_ && address) resolved_->buffers.push_back({uint8_t(stage), address, 0, stride, a.write, pointer});
                        break;
                    }
                    // (a buffer of unknown extent, e.g. a bare address: only a window of it is tracked)
                    rawAccess(address, size && size <= available ? size : std::min<uint64_t>(available, kUnknownExtent), a.write);
                    if (!size || size > available) size = available;
                    if (resolved_) resolved_->buffers.push_back({uint8_t(stage), address, size, stride, a.write, pointer});
                    if (res.kind == vpir::ResourceKind::Buffer && address % 4) return why = "buffer not 4-byte aligned", false;
                    if (res.kind == vpir::ResourceKind::ConstantBuffer && address % 16) {
                        // Constant buffers are read with aligned vector loads: an unaligned one is copied.
                        const uint64_t bytes = std::min<uint64_t>(size, 64 * 1024);
                        GuestRegion g;
                        region(address, bytes, g);
                        const Phase cbPhase(r.cpu_.bindCbMs);
                        auto at = alignedConstantCopy(address, bytes, static_cast<const uint8_t*>(g.host) + (address - g.base));
                        if (!at) return why = "upload ring full", false;
                        note("unaligned constant buffer copied (GPU writes to it in this frame are not seen)");
                        id = rings[ringIndex].id;
                        offset = *at;
                        size = bytes;
                    }
                    a.range = ResourceRange::bytes(id, offset, size);
                    break;
                }
                case vpir::ResourceKind::Texture:
                case vpir::ResourceKind::StorageTexture: {
                    if (!load()) return why = "T# unreadable", false;
                    const Phase texPhase(r.cpu_.bindTexMs);
                    const ImageDesc image = decodeImage({words, b.dwords});
                    if (probeOn && stage == vpgcn::ShaderStage::Pixel && !probePs.valid) probePs = {true, image};
                    if ((cmpArm = res.depth && vpgpu::probe::enabled())) cmpRoute = ProbeStats::kCmpOther, cmpSurface = nullptr;
                    const TexRef ref = textureRef(image, res.kind == vpir::ResourceKind::StorageTexture, a.write, res.integerOnly);
                    if (cmpArm) probeCompare(image, stage, ref), cmpArm = false;
                    if (resolved_) {
                        ResolvedTexture rt;
                        rt.stage = uint8_t(stage);
                        rt.image = image;
                        rt.storage = res.kind == vpir::ResourceKind::StorageTexture;
                        rt.found = bool(ref.id);
                        vpgpu::TextureDesc td;
                        if (ref.id && r.device_.textureDesc(ref.id, td)) {
                            rt.hostFormat = vpgpu::formatInfo(td.format).name;
                            rt.hostWidth = td.width, rt.hostHeight = td.height;
                        }
                        resolved_->textures.push_back(std::move(rt));
                    }
                    if (!ref.id) return why = texWhy.empty() ? "texture unavailable" : "texture unavailable: " + texWhy, false;
                    a.range = ResourceRange::wholeTexture(ref.id);
                    if (ref.view) pendingViews.push_back({a.kind, a.slot, ref.view});
                    break;
                }
                case vpir::ResourceKind::Sampler: {
                    if (!load()) return why = "S# unreadable", false;
                    if (resolved_) {
                        const vpgpu::SamplerDesc sd = decodeSampler({words, b.dwords});
                        const auto mode = [](vpgpu::AddressMode m) -> uint8_t {
                            switch (m) {
                                case vpgpu::AddressMode::Repeat: return 0;
                                case vpgpu::AddressMode::MirrorRepeat: return 1;
                                case vpgpu::AddressMode::ClampToEdge: return 2;
                                case vpgpu::AddressMode::MirrorClampToEdge: return 3;
                                default: return 4;
                            }
                        };
                        resolved_->samplers.push_back({uint8_t(stage), mode(sd.u), mode(sd.v), mode(sd.w),
                                                       uint8_t(sd.magFilter == vpgpu::Filter::Linear), uint8_t(sd.minFilter == vpgpu::Filter::Linear),
                                                       uint8_t(sd.mipFilter == vpgpu::MipFilter::Linear), sd.maxAnisotropy > 1, sd.compare.has_value()});
                    }
                    a.range = ResourceRange::sampler(sampler({words, b.dwords}));
                    break;
                }
                case vpir::ResourceKind::BufferSize:  // SPIR-V only (the GCN lifter never makes one)
                case vpir::ResourceKind::SamplerBias:  // opt-in (EmitOptions::samplerBias), Vulkan only
                    return why = "buffer_size resource", false;
            }
            out.push_back(a);
        }
        if (t.userDataSlot) {
            std::optional<uint64_t> at;
            {
                const Phase ph(r.cpu_.bindUploadMs);
                at = uploadUserData(s.userData);
            }
            if (!at) return why = "upload ring full", false;
            Access a;
            a.range = ResourceRange::bytes(rings[ringIndex].id, *at, sizeof s.userData);
            a.stages = mask;
            a.kind = vpgpu::BindKind::ConstantBuffer;
            a.slot = uint16_t(t.resources.at(*t.userDataSlot).index);
            out.push_back(a);
        }
        return true;
    }

    // ---- Render passes ----------------------------------------------------------------------------

    vpgpu::RenderPassDesc pass;
    bool passOpen = false;
    vpgpu::PassAccessSet passAccess;  // incremental read/write set of the open pass (ES/GS hoisting)

    void closePass() {
        if (passOpen) frame.endRenderPass();
        passOpen = false;
        passAccess.clear();
    }

    void openPass(const vpgpu::RenderPassDesc& want, bool clears) {
        if (passOpen && !clears && pass.sameTargets(want)) return;
        closePass();
        pass = want;
        frame.beginRenderPass(pass);
        passOpen = true;
        ++r.stats_.passes;
        // Later draws on the same targets continue it: no clear on reopen.
        for (vpgpu::Attachment& c : pass.colors) c.load = vpgpu::LoadAction::Load;
        pass.depth.load = vpgpu::LoadAction::Load;
    }

    // ---- Draws ------------------------------------------------------------------------------------

    void draw(const DrawState& d) {
        pollCache();
        ringRoom();
        const RegionScope regions(*this);
        ++r.stats_.draws;
        if (d.count == 0 || d.instances == 0) return skip("empty draw");
        if (isDepthCopy(d)) return handled("depth copy"), depthCopy(d);
        if (d.raster.cullFront && d.raster.cullBack) return skip("both faces culled");

        // Targets.
        vpgpu::RenderPassDesc desc;
        vpgpu::PipelineState ps;
        struct Drawn {
            uint64_t address;
            TextureId id;
            uint32_t wid;
        };
        std::vector<Drawn> drawnColors;
        uint32_t depthWid = 0;
        pendingViews.clear();
        attachedWids.clear();
        if (probeOn) probePs = {};
        uint32_t width = 0, height = 0;
        // One sample count for the whole pass (msaa.h): the smallest of the bound surfaces' counts and the
        // app's limit; EQAA depth with more samples than the colour stores keeps the colour's.
        std::array<uint8_t, 8> fragments{};
        bool dualSource = false;
        for (unsigned i = 0; i < 8; ++i)
            if (d.color[i].format && d.color[i].writeMask) fragments[i] = std::max<uint8_t>(d.color[i].samples, 1);
        bool depthExact = true;
        const vpgpu::Format zf = depthFormat(d.depth.zFormat, d.depth.stencilFormat, &depthExact);
        const SamplePlan plan = planSamples(fragments, zf != vpgpu::Format::Undefined ? std::max<uint8_t>(d.depth.samples, 1) : 0,
                                            r.maxSamples_);
        const uint8_t samples = plan.host;
        {
            const auto slot = [](unsigned n) { return n >= 8 ? 3u : n >= 4 ? 2u : n >= 2 ? 1u : 0u; };
            unsigned asked = zf != vpgpu::Format::Undefined ? std::max<uint8_t>(d.depth.samples, 1) : 1u;
            for (unsigned i = 0; i < 8; ++i)
                if (fragments[i]) asked = std::max<unsigned>(asked, std::max<unsigned>(fragments[i], d.color[i].coverage));
            ++pstat.msDraws[slot(asked)][slot(samples)];
        }
        if (plan.reduced)
            note(samples >= 4 ? "MSAA reduced to 4 samples on the host"
                 : samples == 2 ? "MSAA reduced to 2 samples on the host"
                                : "MSAA reduced to 1 samples on the host");
        curExport = {};
        if (samples > 1) {
            uint8_t cov = samples;
            for (unsigned i = 0; i < 8; ++i)
                if (fragments[i]) cov = std::max<uint8_t>(cov, d.color[i].coverage ? d.color[i].coverage : fragments[i]);
            if (zf != vpgpu::Format::Undefined) cov = std::max<uint8_t>(cov, d.depth.samples);
            msCoverage = cov;
            msFragments = samples;
        }
        if (plan.eqaa && samples > 1) note("EQAA: the depth buffer has more samples than the colour stores; the host draws with the colour's");
        std::vector<std::pair<unsigned, TargetIds>> colorIds;
        for (unsigned i = 0; i < 8; ++i) {
            const ColorTarget& c = d.color[i];
            if (!c.format || !c.writeMask) continue;
            const ColorPlan plan = colorPlan(c.format, c.numberType, c.compSwap);
            const vpgpu::Format f = plan.format;
            if (f == vpgpu::Format::Undefined) {
                note("colour target format " + std::to_string(c.format) + "/" + std::to_string(c.numberType) +
                     " unsupported: slot left out" + (plan.why ? std::string(" (") + plan.why + ")" : std::string()));
                continue;
            }
            if (!plan.exact && plan.why) note(plan.why);
            curExport[i] = plan.conv;
            if (!linearTiling(c.tileMode)) note("render targets kept in GPU memory (guest copy not written)");
            vpgpu::Format viewAs = vpgpu::Format::Undefined;
            Target* tg = target(colorTargets, c.address, f, c.width, c.height, samples,
                                uint16_t(c.sliceStart + c.sliceCount), false, c.tileMode, c.pitch, &viewAs);
            if (!tg) return skip("colour target creation failed");
            if (samples > 1) tg->msCov = std::max<uint8_t>(samples, c.coverage), tg->msFrag = samples;
            tg->lastDraw = r.frame_;
            ++tg->draws;
            refresh(*tg, nullptr);  // another writer changed its memory since it last held it?
            if (resolved_) {
                ResolvedTarget& rt = resolved_->color[i];
                rt = {true, c.address, tg->width, tg->height, c.pitch, c.format, c.numberType, c.compSwap, c.tileMode,
                      c.sliceStart, c.sliceCount, c.samples, vpgpu::formatInfo(f).name, false};  // (`cleared`: set below, when a clear that waited becomes the pass's load action)
            }
            const TargetIds t = idsOf(*tg);
            colorIds.push_back({i, t});
            drawnColors.push_back({c.address, t.resolved ? t.resolved : t.render, t.wid});
            attachedWids.push_back(t.wid);
            desc.colors[i].texture = t.render;
            desc.colors[i].layer = c.sliceStart;
            desc.colors[i].format = viewAs;  // (drawn as another format of the same texel size)
            if (t.resolved) {
                // Resolved at the end of the pass; the lifetime planner decides what is kept (lifetime.h).
                desc.colors[i].resolve = t.resolved;
                desc.colors[i].resolveLayer = c.sliceStart;
                desc.colors[i].store = vpgpu::StoreAction::StoreAndResolve;
                desc.colors[i].keepSamples = true;
            }
            ps.colorFormats[i] = f;
            const BlendResolved br = resolveBlend(d.blend[i], c, vpgpu::formatInfo(f).integer());
            ps.blend[i] = br.state;
            if (plan.conv.kind != vpgcn::ExportConv::Packed)
                ps.blend[i].writeMask = permuteWriteMask(ps.blend[i].writeMask, plan.conv.select);
            if (br.state.enabled && !br.exact) note("blend factor with an undefined encoding: zero");
            if (i == 0 && br.dualSource) dualSource = true;
            if (i != 0 && br.dualSource) note("dual-source blend on a target other than 0 ignored");
            width = width ? std::min(width, c.width) : c.width;
            height = height ? std::min(height, c.height) : c.height;
        }
        bool clears = false;
        const DepthTarget& z = d.depth;
        uint32_t depthWidth = 0, depthHeight = 0;  // the depth image (a clear that leaves part of it alone does not make it the guest's)
        uint16_t depthLayers = 1;
        std::shared_ptr<vpgpu::ClearRects> partialClear;  // a clear of one plane, or of part of the target
        StencilResolved stencil;
        TargetIds depthIds;
        if (zf != vpgpu::Format::Undefined) {
            if (!depthExact) note(z.zFormat == 2 ? "24-bit depth held as 32-bit float" : "16-bit depth with stencil held as 32-bit float");
            const uint64_t readAddress = z.zFormat ? z.zAddress : z.stencilAddress;
            const uint64_t writeAddress = z.zFormat ? z.zWriteAddress : z.stencilWriteAddress;
            Target* dt = target(depthTargets, readAddress, zf, z.width, z.height,
                                samples, uint16_t(z.sliceStart + z.sliceCount), true, z.tileMode, 0);
            if (!dt) return skip("depth target creation failed");
            {
                const uint64_t zBase = agreedBase(z.zAddress, z.zWriteAddress), stencilBase = agreedBase(z.stencilAddress, z.stencilWriteAddress);
                const uint64_t primary = z.zFormat ? zBase : stencilBase;
                const bool single = dt->samples == 1;  // (a multisampled depth image has no single-sample picture to write back)
                noteDepthPlanes(*dt, z.zFormat, single && primary && primary == dt->address, single && z.zFormat && z.stencilFormat ? stencilBase : 0);
            }
            depthWidth = dt->width, depthHeight = dt->height, depthLayers = dt->layers;
            if (resolved_)
                resolved_->depth = {true, z.zAddress, z.stencilAddress, dt->width, dt->height, z.zFormat, z.stencilFormat,
                              z.tileMode, z.samples, vpgpu::formatInfo(zf).name, z.depthClear, z.stencilClear};
            // (a clear replaces the contents; a host that seeds the surface still gets it first: the clear then applies to
            // the seeded pixels, as it does to the ones the other renderer started from)
            if (!(z.depthClear || z.stencilClear)) refresh(*dt, nullptr);
            else if (r.hooks_.seed) seedFromHost(*dt, nullptr);
            depthWid = dt->wid;
            attachedWids.push_back(dt->wid);
            depthIds = idsOf(*dt);
            if (samples > 1) dt->msCov = std::max<uint8_t>(samples, z.samples), dt->msFrag = samples;
            dt->depthMode = depthResolveMode(z.depthCopy, z.copySample, d.depthStencil.depthTest ? d.depthStencil.depthFunc : uint8_t(7));
            desc.depth.texture = depthIds.render;
            desc.depth.layer = z.sliceStart;
            if (depthIds.resolved) {
                desc.depth.resolve = depthIds.resolved;
                desc.depth.resolveLayer = z.sliceStart;
                desc.depth.store = vpgpu::StoreAction::StoreAndResolve;
                desc.depth.keepSamples = true;
                desc.depth.depthResolve = dt->depthMode;
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
            ps.depthCompare = ds.depthTest ? compareOp(ds.depthFunc) : vpgpu::CompareOp::Always;
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
        probePass.clear();
        if (probeOpen() && probeLines < kProbeLines)
            for (unsigned i = 0; i < 8; ++i) {
                const ColorTarget& c = d.color[i];
                if (!c.format || !c.writeMask || c.width < 1024 || c.height < 1024) continue;
                probePass = "cb" + std::to_string(i) + " " + hex(c.address) + " " + std::to_string(c.width) + "x" + std::to_string(c.height) +
                            " pitch=" + std::to_string(c.pitch) + " tile=" + std::to_string(c.tileMode) + " slices=" +
                            std::to_string(c.sliceStart) + "+" + std::to_string(c.sliceCount);
                break;
            }

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

        // The planes a depth/stencil clear leaves whole in the image (so that the image is what the guest holds there, see
        // Surface::zAuthoritative), once the clear is recorded: the scissor takes all of the image, and so do the slices
        // cleared (`layers`: those of the pass's load action, else those of the clear rectangle).
        const bool clearCoversImage = zf != vpgpu::Format::Undefined && scissorFull && width == depthWidth && height == depthHeight;
        const bool clearsZ = clearCoversImage && z.depthClear && z.zFormat, clearsStencil = clearCoversImage && z.stencilClear && z.stencilFormat;
        const auto clearDone = [&](uint16_t layers) {
            Surface* t = clearsZ || clearsStencil ? surfaceOf(depthWid) : nullptr;
            if (!t || z.sliceStart != 0 || layers < depthLayers) return;
            if (clearsZ) t->zAuthoritative = true;
            if (clearsStencil) t->stencilAuthoritative = true;
        };
        // A depth/stencil clear: the whole target in the pass's load action when every plane of the
        // format is cleared; otherwise (one plane, or less than the target) a clear inside the pass.
        if (zf != vpgpu::Format::Undefined && (z.depthClear || z.stencilClear)) {
            const bool allPlanes = (z.depthClear || !z.zFormat) && (z.stencilClear || !z.stencilFormat);
            if (allPlanes && scissorFull) {
                desc.depth.load = vpgpu::LoadAction::Clear;
                desc.depth.clear.depth = z.clearDepth;
                desc.depth.clear.stencil = z.clearStencil;
                clears = true;
            } else {
                partialClear = std::make_shared<vpgpu::ClearRects>();
                vpgpu::ClearRects::Rect rc;
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
                clearDone(depthLayers);
                markWriteWid(depthWid);
                any = true;
            } else if (partialClear) {
                note("RectList depth/stencil clear done as a clear inside the next pass that draws into the target");
                closePass();
                addPendingRects(depthIds.render, depthIds.resolved, std::move(partialClear));
                clearDone(z.sliceCount);
                markWriteWid(depthWid);
                any = true;
            }
            // (nothing cleared: the rectangle is drawn by the vertex stage, below)
            if (any) return handled("rect list clear");
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
            if (depthFormat(d.depth.zFormat, d.depth.stencilFormat) != vpgpu::Format::Undefined)
                layers = std::max<uint16_t>(layers, uint16_t(d.depth.sliceCount));
            desc.layerCount = layers;
        }
        ps.samples = samples;
        // DB_ALPHA_TO_MASK: Metal's alpha-to-coverage with several samples; with one, the pixel shader does an
        // alpha test at 0.5 (LiftInput::alphaTest), which is what the coverage mask comes to.
        // (colour 0 must be bound for real: a slot left out, or with its write mask empty, has no alpha to test)
        const bool color0 = ps.colorFormats[0] != vpgpu::Format::Undefined;
        ps.alphaToCoverage = d.alphaToMask && color0 && samples > 1;
        alphaTestNow = d.alphaToMask && color0 && samples <= 1;

        // Foveation: a pass over two-slice targets renders both eyes through a rate map.
        viewIndexedNow = false;
        if (r.foveation_ && samples == 1 && !colorIds.empty()) {
            bool eligible = true;
            for (const auto& entry : colorIds) {
                const ColorTarget& c = d.color[entry.first];
                eligible = eligible && c.sliceStart == 0 && c.sliceCount == 2;
            }
            if (zf != vpgpu::Format::Undefined) eligible = eligible && z.sliceStart == 0 && z.sliceCount == 2;
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
        countApprox(vs), countApprox(esT), countApprox(gsT), countApprox(fs);
        ps.vertexShader = vs->id;
        ps.fragmentShader = fs ? fs->id : 0;

        // Geometry pipeline / RectList: the buffers its kernels work on.
        vpgpu::Draw draw;
        std::string why;
        ringBinds = {};
        vpgpu::Dispatch esJob, gsJob;
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
            // (each stage takes the format views its own bindings asked for: they are not the draw's)
            if (!bind(esStage, *esT, vpgcn::ShaderStage::Export, vpgpu::stage::compute, esJob.bindings, why)) return skip("es: " + why);
            esJob.views = std::move(pendingViews);
            pendingViews.clear();
            if (!bind(gsStage, *gsT, vpgcn::ShaderStage::Geometry, vpgpu::stage::compute, gsJob.bindings, why)) return skip("gs: " + why);
            gsJob.views = std::move(pendingViews);
            pendingViews.clear();
        }
        if (!bind(*vsSrc, *vs, geom ? vpgcn::ShaderStage::CopyVertex : vpgcn::ShaderStage::Vertex, vpgpu::stage::vertex, draw.bindings, why))
            return skip("vs: " + why);
        if (fs && !bind(psStage, *fs, vpgcn::ShaderStage::Pixel, vpgpu::stage::fragment, draw.bindings, why)) return skip("ps: " + why);
        draw.views = std::move(pendingViews);
        pendingViews.clear();
        r.cpu_.bindMs += msSince(tBind);

        // Geometry.
        ps.primitive = toPrimitive(d.primitive);
        if (geom) ps.primitive = d.geom.outPrim == 0 ? vpgpu::Primitive::Points
                                 : d.geom.outPrim == 1 ? vpgpu::Primitive::Lines : vpgpu::Primitive::Triangles;
        draw.primitive = ps.primitive;
        draw.instances = d.instances;
        if (geom) {
            // The GS output is an index list into the copy shader's vertices; every instance is in it.
            const RingBinding& gi = ringBinds[size_t(vpgcn::GeomRing::GsIndex)];
            draw.instances = 1;
            draw.index = vpgpu::IndexBuffer{gi.id, gi.offset, true};
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
            vpgpu::Indirect indirect{id, offset};
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
                draw.index = vpgpu::IndexBuffer{ib, ioff, d.index32};
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
            draw.index = vpgpu::IndexBuffer{rings[ringIndex].id, *at, true};
            draw.count = uint32_t(indices.size());
            note(isAdjacency(d.primitive) ? "adjacency vertices dropped (no geometry shader)"
                                          : "fan/quad/loop primitives converted to lists");
        } else if (d.indexAddress && indicesNeedPrep(d) && !indicesKnownClean(d)) {
            // Byte-swapped indices, or a primitive restart index other than the host's (all ones).
            const uint64_t bytes = uint64_t(d.count) * (d.index32 ? 4 : 2);
            std::vector<uint8_t> src(bytes), prepared;
            if (!readGuest(d.indexAddress, src.data(), bytes)) return skip("index buffer outside guest memory");
            bool widened = false;
            prepareIndices(d, src, prepared, &widened);
            if (!d.indexSwap) noteIndicesClean(d, !widened && prepared == src);
            auto at = upload(prepared.data(), prepared.size(), 4);
            if (!at) return skip("upload ring full");
            draw.index = vpgpu::IndexBuffer{rings[ringIndex].id, *at, d.index32 || widened};
            draw.count = d.count;
            draw.baseVertex = int32_t(d.firstVertex);
            note(widened ? "strip indices widened to 32 bits (the all-ones index is a vertex, not a restart)"
                 : d.indexSwap ? "index buffer byte-swapped on the CPU" : "primitive restart index rewritten on the CPU");
        } else if (d.indexAddress) {
            BufferId id;
            uint64_t offset = 0, available = 0;
            const uint64_t bytes = uint64_t(d.count) * (d.index32 ? 4 : 2);
            if (!rawBuffer(d.indexAddress, bytes, false, id, offset, available) || bytes > available)
                return skip("index buffer outside guest memory");
            if (offset % (d.index32 ? 4 : 2)) return skip("misaligned index buffer");
            draw.index = vpgpu::IndexBuffer{id, offset, d.index32};
            draw.count = d.count;
            draw.baseVertex = int32_t(d.firstVertex);
        } else {
            draw.first = d.firstVertex;
            draw.count = d.count;
        }

        // Fixed function.
        ps.cull = d.raster.cullBack ? vpgpu::CullMode::Back : d.raster.cullFront ? vpgpu::CullMode::Front : vpgpu::CullMode::None;
        if (rect) ps.cull = vpgpu::CullMode::None;  // (rectangles have no facing)
        draw.frontFace = d.raster.frontFaceClockwise ? vpgpu::Winding::Clockwise : vpgpu::Winding::CounterClockwise;
        draw.stencilRef = stencil.refFront;
        draw.stencilRefBack = stencil.refBack;
        draw.blendColor = d.blendColor;
        const Viewport& v = d.viewport;
        const ViewportRect vr = hostViewportRect(v, width, height);
        if (vr.wholeTarget) note("viewport transform off: positions taken as clip space over the whole target");
        if (vr.valid) {
            // GCN: screen = ndc * scale + offset (y down when yScale < 0). Metal: y down from the top.
            if (vr.flipped) {
                // Metal cannot flip y in the viewport: the image comes out upside down; at least keep
                // the facing right (mirroring swaps the winding).
                note("viewport with y up (positive y scale) drawn flipped");
                draw.frontFace = draw.frontFace == vpgpu::Winding::Clockwise ? vpgpu::Winding::CounterClockwise
                                                                             : vpgpu::Winding::Clockwise;
            }
            const ViewportZ vz = viewportDepthRange(v, d.raster.clipSpaceZeroToW);
            draw.viewport = {vr.x, vr.y, vr.width, vr.height, vz.znear, vz.zfar};
        }
        if (scissorSet) draw.scissor = {uint32_t(sl), uint32_t(stp), uint32_t(sr - sl), uint32_t(sb - stp)};
        // Polygon offset, depth clamp, wireframe.
        if (zf != vpgpu::Format::Undefined && z.zFormat) {
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
        if (resolved_) {
            resolved_->topology = toPrimitive(d.primitive) == vpgpu::Primitive::Points  ? 0
                            : toPrimitive(d.primitive) == vpgpu::Primitive::Lines || toPrimitive(d.primitive) == vpgpu::Primitive::LineStrip ? 1
                                                                                  : 2;
            resolved_->viewportValid = vr.valid;
            if (vr.valid) {
                const ViewportZ vz = viewportDepthRange(v, d.raster.clipSpaceZeroToW);
                resolved_->viewport[0] = vr.x, resolved_->viewport[1] = vr.y, resolved_->viewport[2] = vr.width, resolved_->viewport[3] = vr.height;
                resolved_->viewport[4] = vz.znear, resolved_->viewport[5] = vz.zfar;
            }
            resolved_->scissorValid = true;
            resolved_->scissor[0] = sl, resolved_->scissor[1] = stp, resolved_->scissor[2] = sr - sl, resolved_->scissor[3] = sb - stp;
            resolved_->stateValid = true;
            resolved_->cull = ps.cull == vpgpu::CullMode::Front ? 1 : ps.cull == vpgpu::CullMode::Back ? 2 : 0;
            resolved_->frontClockwise = draw.frontFace == vpgpu::Winding::Clockwise;
            resolved_->depthTest = zf != vpgpu::Format::Undefined && z.zFormat && d.depthStencil.depthTest;
            resolved_->depthWrite = ps.depthWrite;
            resolved_->depthFunc = uint8_t(ps.depthCompare);
            resolved_->stencilTest = ps.stencilTest;
            for (unsigned i = 0; i < 8; ++i) resolved_->blendEnable[i] = ps.colorFormats[i] != vpgpu::Format::Undefined && ps.blend[i].enabled;
        }
        const auto tPipe = Clock::now();
        // The label only matters for a compile: built only when the state is not ready yet.
        const vpgpu::PipelineKey key = ps.key();  // (the label is not part of it)
        vpgpu::PipelineCache::Status st = vpgpu::PipelineCache::Status::Ready;
        if (!r.device_.pipelines().ready(key)) {
            ps.label = "vs " + hex(vs->id) + (fs ? " ps " + hex(fs->id) : "");
            st = r.device_.pipelines().request(ps, key, vpgpu::Priority::Visible);
        }
        r.cpu_.pipelineMs += msSince(tPipe);
        if (st == vpgpu::PipelineCache::Status::Failed) return skip("pipeline failed");
        if (st != vpgpu::PipelineCache::Status::Ready) {
            // Still compiling: the draw is recorded all the same and Device::submit holds the frame (bounded,
            // Device::holdMs) until its pipelines are built, instead of dropping it (holes in the picture).
            if (r.device_.holdMs() == 0) return skip("pipeline compiling");
            note("draw recorded before its pipeline was ready (frame held)");
        }
        draw.pipeline = key;

        // Clears that were waiting for this target become the load action of the pass (or clear draws in it).
        std::vector<std::shared_ptr<const vpgpu::ClearRects>> inPassClears;
        if (!pendingClears.empty()) {
            PendingClear pc;
            for (unsigned i = 0; i < desc.colors.size(); ++i) {
                vpgpu::Attachment& a = desc.colors[i];
                if (a.texture && takePendingClear(a.texture, pc)) {
                    a.load = vpgpu::LoadAction::Clear;
                    a.clear = pc.value;
                    clears = true;
                    if (resolved_) resolved_->color[i].cleared = true;
                }
            }
            if (desc.depth.texture && takePendingClear(desc.depth.texture, pc)) {
                if (pc.whole) {
                    desc.depth.load = vpgpu::LoadAction::Clear;
                    desc.depth.clear = pc.value;
                    clears = true;
                }
                inPassClears = std::move(pc.rects);
            }
        }
        if (geom) {
            auto computeStage = [&](const Translated& t, const char* what, vpgpu::Dispatch& job) {
                vpgpu::PipelineState cs;
                cs.computeShader = t.id;
                cs.label = std::string(what) + " " + hex(t.id);
                const vpgpu::PipelineCache::Status cst = r.device_.pipelines().request(cs, vpgpu::Priority::Visible);
                if (cst == vpgpu::PipelineCache::Status::Failed) {
                    skip("pipeline failed");
                    return false;
                }
                if (cst != vpgpu::PipelineCache::Status::Ready) {
                    if (r.device_.holdComputeMs() == 0) {
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
            draw.expandedBy = {esJob.pipeline, gsJob.pipeline};  // (left out with them if they are not built, see Device::takeFrame)
            // The expansion runs before the render pass the draw joins (hoisted above it, the pass stays
            // whole) unless it touches something the pass has written so far; then it runs in between and
            // the pass is split (the compute work reads a finished picture).
            const vpgpu::RenderPass* open = frame.openRenderPass();
            const bool joins = passOpen && !clears && open && pass.sameTargets(desc);
            if (joins) passAccess.sync(*open, open);
            if (joins && !passAccess.hazard(esJob) && !passAccess.hazard(gsJob)) {
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

        // A title that stopped asking for occlusion results stops paying for the counting.
        if (vis.active && r.device_.submittedFrames() - vis.lastZpassFrame > Vis::kIdleFrames) vis.active = false;
        if (vis.active) draw.visibility = vpgpu::BufferSpan{vis.id, (vis.seq % kVisSlots) * 8, 8};
        openPass(desc, clears);
        for (auto& rc : inPassClears) {
            vpgpu::Draw cd;
            cd.clear = std::move(rc);
            frame.draw(std::move(cd));
        }
        if (partialClear) {
            vpgpu::Draw cd;
            cd.clear = std::move(partialClear);
            frame.draw(std::move(cd));
        }
        if (!frame.draw(std::move(draw))) return skip("no open pass");
        ++r.stats_.drawn;
        // What the pass wrote: those images now hold the newest bytes of their ranges.
        for (const Drawn& dc : drawnColors) {
            latestColor[dc.address] = dc.id;
            markWriteWid(dc.wid);
            if (vpgpu::probe::enabled())
                if (Surface* ds = surfaceOf(dc.wid)) {
                    ds->lastScissor[0] = sl, ds->lastScissor[1] = stp, ds->lastScissor[2] = sr - sl, ds->lastScissor[3] = sb - stp;
                    ds->lastViewport[0] = vr.x, ds->lastViewport[1] = vr.y, ds->lastViewport[2] = vr.width, ds->lastViewport[3] = vr.height;
                }
        }
        if (probeOn && !drawnColors.empty()) noteProbe(drawnColors[0].address, geom, d.geom.maxVertOut, gsDrawIndices);
        if (depthWid) clearDone(clears ? desc.layerCount : z.sliceCount);  // (a clear of one plane or of part of the target is a rectangle inside the pass)
        if (depthWid && (clears || clearsZ || clearsStencil || (d.depthStencil.depthWrite && z.zFormat) || (d.depthStencil.stencilTest && z.stencilFormat)))
            markWriteWid(depthWid);
    }

    // Whether a compute program has what the reference translation emits as a bitwise xor 32: v_xor_b32, s_xor_b32,
    // ds_swizzle_b32 (its lane index), and a negate modifier (it flips the sign bit of operands it does not treat as
    // float; the float ones are counted too: a program wrongly taken for a clear loop is dropped, the reverse is
    // harmless). Decoded up to s_endpgm. By program key.
    std::unordered_map<uint64_t, bool> xorPrograms;
    bool usesXor(const Stage& s, uint64_t key) {
        auto [it, fresh] = xorPrograms.try_emplace(key, false);
        if (!fresh) return it->second;
        for (size_t at = 0; at < s.code.size();) {
            vpgcn::Instruction in;
            if (!vpgcn::decode(s.code.subspan(at), in) || in.size < 4) {
                ++at;
                continue;
            }
            if (in.op == vpgcn::Op::s_endpgm) break;
            if (in.op == vpgcn::Op::v_xor_b32 || in.op == vpgcn::Op::s_xor_b32 || in.op == vpgcn::Op::ds_swizzle_b32 || in.neg)
                return it->second = true;
            at += in.size / 4;
        }
        return false;
    }

    void dispatch(const DispatchState& d) {
        const Who who(touchWho, "dispatch");
        pollCache();
        ringRoom();
        const RegionScope regions(*this);
        ++r.stats_.dispatches;
        probePass.clear();  // (the probe follows the passes that write a large colour target)
        if (d.cs.code.empty()) return skip("no compute shader");
        if (!d.indirectAddress && (!d.groups[0] || !d.groups[1] || !d.groups[2])) return skip("empty dispatch");
        const Translated* cs = shader(d.cs, vpgcn::ShaderStage::Compute);
        if (!cs->ok) return skip("compute shader not translated");
        countApprox(cs);
        if (r.hooks_.computeBuffers && (!r.hooks_.computeBuffersWanted || r.hooks_.computeBuffersWanted())) {
            std::vector<ComputeBuffer> buffers;
            bool complete = true;  // every buffer the program accesses is in `buffers`
            for (const vpgcn::Binding& b : cs->bindings) {
                const vpir::Resource& res = cs->resources.at(b.slot);
                if (res.kind != vpir::ResourceKind::Buffer && res.kind != vpir::ResourceKind::ConstantBuffer) continue;
                uint32_t words[8];
                if (b.dwords == 0 || b.dwords > 8 ||
                    (b.source.kind != vpgcn::DescriptorSource::UserData && b.source.kind != vpgcn::DescriptorSource::Pointer) ||
                    !readDescriptor(d.cs, b.source, b.dwords, words)) {
                    complete = false;
                    continue;
                }
                buffers.push_back({decodeBuffer({words, b.dwords}).address,
                                   res.kind == vpir::ResourceKind::Buffer && res.access != vpir::Access::Read});
            }
            if (!buffers.empty() &&
                r.hooks_.computeBuffers(buffers, usesXor(d.cs, programKeyCached(d.cs, vpgcn::ShaderStage::Compute)), complete))
                return handled("compute handled by the host");
        }
        vpgpu::Dispatch x;
        std::string why;
        pendingViews.clear();
        attachedWids.clear();
        if (!bind(d.cs, *cs, vpgcn::ShaderStage::Compute, vpgpu::stage::compute, x.bindings, why)) return skip("cs: " + why);
        x.views = std::move(pendingViews);
        pendingViews.clear();
        vpgpu::PipelineState ps;
        ps.computeShader = cs->id;
        ps.label = "cs " + hex(cs->id);
        const vpgpu::PipelineCache::Status st = r.device_.pipelines().request(ps, vpgpu::Priority::Visible);
        if (st == vpgpu::PipelineCache::Status::Failed) return skip("pipeline failed");
        if (st != vpgpu::PipelineCache::Status::Ready) {
            if (r.device_.holdComputeMs() == 0) return skip("pipeline compiling");
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
            x.indirect = vpgpu::Indirect{id, offset};
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
            const vpgpu::FormatInfo& fi = vpgpu::formatInfo(t.format);
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
                           (it->first > address ? "(after)" : "") + " " + vpgpu::formatInfo(t.format).name + " " +
                           std::to_string(t.width) + "x" + std::to_string(t.height) + "x" + std::to_string(t.layers) +
                           " s" + std::to_string(t.samples) + " used" + std::to_string(r.frame_ - t.lastUse) + " id=" + std::to_string(t.wid) +
                           " draw=" + (t.lastDraw == ~uint64_t(0) ? std::string("never") : std::to_string(t.lastDraw));
                }
        };
        targets(colorTargets, "colour");
        targets(depthTargets, "depth");
        if (auto it = textures.find(address); it != textures.end())
            for (const auto& [key, t] : it->second)
                out += std::string(" tex ") + (t.format == vpgpu::Format::Undefined ? "?" : vpgpu::formatInfo(t.format).name) +
                       " " + std::to_string(t.width) + "x" + std::to_string(t.height) + (t.storage ? " storage" : "") +
                       (tracker.writerAt(it->first) == t.wid ? " gpu_written" : "") + (tracker.staleAt(it->first) && tracker.writerAt(it->first) == t.wid ? " ahead_of_memory" : "");
        return out.empty() ? " nothing (only guest memory)" : out;
    }

    // Writes the colour target the GPU rendered at c.address back to guest memory in the layout the
    // CB names (tile mode, pitch), as a retile kernel on the GPU timeline: for a CPU that reads a
    // render target, or a later T# that names it as tiled memory. One sample plane; false if the
    // target is not there, is multisampled on the host, or the kernel is not ready (try again).
    bool writeBack(const ColorTarget& c) {
        if (guestReadOnly) return false;
        const vpgpu::Format f = colorFormat(c.format, c.numberType, c.compSwap);
        auto found = colorTargets.find(c.address);
        if (f == vpgpu::Format::Undefined || found == colorTargets.end()) return false;
        Target* t = nullptr;
        for (Target& cand : found->second)
            if (cand.format == f && cand.width == c.width && cand.height == c.height) t = &cand;
        if (!t) return false;
        flushClear(*t);  // (a fast clear / depth clear still waiting for a pass)
        catchUpTwin(*t);
        life.noteRead(t->id, r.frame_);
        const vpgpu::FormatInfo& fi = vpgpu::formatInfo(f);
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
        watchOwned(*t, false, true);
        return true;
    }

    // The last render pass of the frame that draws into multisampled `ms`, if no command recorded after it
    // touches `dst` or `ms` (so the resolve can move into it without reordering anything dst's readers see).
    vpgpu::RenderPass* lastPassWriting(TextureId ms, TextureId dst) {
        auto& cmds = frame.mutableCommands();
        size_t looked = 0;
        for (auto c = cmds.rbegin(); c != cmds.rend() && looked < 16; ++c, ++looked) {
            std::vector<Access> acc;
            if (auto* rp = std::get_if<vpgpu::RenderPass>(&*c)) {
                for (const vpgpu::Attachment& a : rp->desc.colors)
                    if (a.texture == ms) return rp;
                acc = vpgpu::attachmentAccesses(rp->desc);
                for (const vpgpu::Draw& d : rp->draws) vpgpu::appendDrawAccesses(d, acc);
            } else if (auto* dp = std::get_if<vpgpu::Dispatch>(&*c)) {
                acc = dp->bindings;
            } else {
                acc = vpgpu::copyAccesses(std::get<vpgpu::Copy>(*c));
            }
            for (const Access& a : acc)
                if (a.range.kind == ResourceRange::Kind::Texture &&
                    ((a.range.index == dst.index && a.range.generation == dst.generation) ||
                     (a.range.index == ms.index && a.range.generation == ms.generation)))
                    return nullptr;
        }
        return nullptr;
    }

    // vpps4.resolve_src (diagnostics only): where a resolve takes its samples from. One line for the first 32 resolves and
    // then one every 600 frames: the destination, the source chosen (new = target() made it just now, last_draw = the
    // frame a pass last drew into it) and the multisampled targets at the source address. `empty_source` (own counter,
    // same limits): the source is a multisampled target created by this very resolve, so it holds nothing.
    uint64_t resolveSrcCalls = 0, resolveSrcFrame = 0, emptySrcCalls = 0, emptySrcFrame = 0;
    static std::string describeResolveTarget(const Target& t) {
        return "id=" + std::to_string(t.wid) + " " + std::to_string(t.width) + "x" + std::to_string(t.height) + "x" + std::to_string(t.layers) +
               " s" + std::to_string(t.samples) + " fmt=" + vpgpu::formatInfo(t.format).name + " tile=" + std::to_string(t.tile) +
               " last_draw=" + (t.lastDraw == ~uint64_t(0) ? std::string("never") : std::to_string(t.lastDraw));
    }
    void noteResolveSource(const ColorTarget& src, const ColorTarget& dst, const Target& from, bool fresh, const char* way) {
        const auto due = [&](uint64_t& calls, uint64_t& frame) {
            if (++calls > 32 && r.frame_ < frame + 600) return false;
            frame = r.frame_;
            return true;
        };
        const bool empty = fresh && from.samples > 1;
        const bool tellEmpty = empty && due(emptySrcCalls, emptySrcFrame);
        if (!due(resolveSrcCalls, resolveSrcFrame) && !tellEmpty) return;
        std::string candidates;
        if (const auto it = colorTargets.find(src.address); it != colorTargets.end())
            for (const Target& t : it->second)
                if (t.samples > 1) candidates += (candidates.empty() ? "" : "; ") + describeResolveTarget(t);
        log(std::string("vpps4.resolve_src ") + (tellEmpty ? "empty_source " : "") + "f=" + std::to_string(r.frame_) + " way=" + way +
            " dst=" + hex(dst.address) + " " + std::to_string(dst.width) + "x" + std::to_string(dst.height) + "x" +
            std::to_string(dst.sliceStart + dst.sliceCount) + " s" + std::to_string(dst.samples) + " src=" + hex(src.address) + " " +
            std::to_string(src.width) + "x" + std::to_string(src.height) + "x" + std::to_string(src.sliceStart + src.sliceCount) + " s" +
            std::to_string(src.samples) + " | from " + describeResolveTarget(from) + " ms_tex=" + (from.ms ? "1" : "0") + " new=" +
            (fresh ? "1" : "0") + " | ms_candidates=[" + candidates + "]");
    }

    // CB_COLOR_CONTROL.MODE = RESOLVE: colour target `src` (CB0) into `dst` (CB1). The way is chosen by
    // planResolve (msaa.h): by the pass that drew into the multisampled source when it still can, else a
    // pass of its own that loads the samples; a 1x source (or one whose single-sample picture is read
    // anyway) is a texture copy.
    void resolve(const ColorTarget& src, const ColorTarget& dst) {
        ++r.stats_.resolves;
        const vpgpu::Format f = colorFormat(src.format, src.numberType, src.compSwap);
        const vpgpu::Format fd = colorFormat(dst.format, dst.numberType, dst.compSwap);
        if (f == vpgpu::Format::Undefined || fd == vpgpu::Format::Undefined) return skip("resolve: format unsupported");
        // Another format of the same texel size (RGBA vs sRGB, RGBA vs BGRA): the backend resolves through a view
        // of the destination in the source's pixel format (only for multisampled sources: a Metal blit copies
        // between identical pixel formats only).
        const bool differ = f != fd;
        if (differ && !(sameTexel(f, fd) && r.hooks_.view)) return skip("resolve: source and destination formats differ");
        const uint16_t layers = uint16_t(src.sliceStart + src.sliceCount);
        // The source exists at the count its draws used (the most recent one of this shape).
        Target* from = nullptr;
        // (the most recent wins; on a tie of lastUse, the one with samples when the resolve is of a multisampled source)
        const bool wantMs = src.samples > 1;
        for (Target& t : colorTargets[src.address]) {
            if (t.format != f || t.width != src.width || t.height != src.height || t.layers < layers) continue;
            if (!from || t.lastUse > from->lastUse ||
                (t.lastUse == from->lastUse && ((t.samples > 1) == wantMs || (from->samples > 1) != wantMs)))
                from = &t;
        }
        // No target of exactly that size. A title sets its targets up smaller than the CB's tile-aligned pitch and height (the
        // extent of SetRenderTarget), so a resolve that names the allocation still means the picture the passes drew: the
        // most recently drawn target at the address that fits inside it (the largest of those drawn together), not a new one
        // that nothing drew into; also when a target of the exact size is there but no pass ever drew into it.
        if (!from || from->lastDraw == ~uint64_t(0)) {
            Target* drawn = nullptr;
            const auto area = [](const Target& t) { return uint64_t(t.width) * t.height; };
            for (Target& t : colorTargets[src.address]) {
                if (t.format != f || t.width > src.width || t.height > src.height || t.layers < layers || t.lastDraw == ~uint64_t(0)) continue;
                if (!drawn || t.lastDraw > drawn->lastDraw ||
                    (t.lastDraw == drawn->lastDraw &&
                     (area(t) > area(*drawn) ||
                      (area(t) == area(*drawn) && (t.samples > 1) == wantMs && (drawn->samples > 1) != wantMs))))
                    drawn = &t;
            }
            if (drawn) from = drawn;
        }
        const bool freshSource = !from;
        if (!from) {
            const std::array<uint8_t, 1> frag{std::max<uint8_t>(src.samples, 1)};
            from = target(colorTargets, src.address, f, src.width, src.height, planSamples(frag, 0, r.maxSamples_).host, layers,
                          false, src.tileMode, src.pitch);
        }
        if (!from) return skip("resolve: target creation failed");
        const uint32_t fromWid = from->wid;
        // A source smaller than the request (the picture drawn in the allocation): the destination is the same picture, as
        // large as the source (an attachment resolves into a texture of its own size). Otherwise the request, as ever.
        const bool fromSmaller = from->width != src.width || from->height != src.height;
        const uint32_t dw = fromSmaller ? std::min(dst.width, from->width) : dst.width;
        const uint32_t dh = fromSmaller ? std::min(dst.height, from->height) : dst.height;
        Target* to = target(colorTargets, dst.address, fd, dw, dh, 1, uint16_t(dst.sliceStart + dst.sliceCount),
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
        const bool twinWanted = from->ms && life.wants(from->id, r.frame_);
        // (a wanted single-sample picture stays resolved by the passes; with another destination format the
        // resolve is a pass of its own, so nothing replaces the picture's resolve either)
        rc.sourceMultisampled = from->ms && (!twinWanted || differ);
        rc.formatsMatch = !differ;
        rc.formatsViewable = differ;
        rc.destinationIsSource = from->id == to->id;
        const uint16_t n = uint16_t(std::max<uint16_t>(std::min(src.sliceCount, dst.sliceCount), 1));
        // A fast clear of the destination still waiting for a pass would be emitted after the resolve and wipe the
        // resolved picture. A whole clear that the resolve overwrites completely (every texel, every layer) is simply
        // dropped; any other is flushed now, before the resolve (and before `last` is chosen: the clear pass touches
        // the destination, so the pass that drew the samples cannot take the resolve over).
        if (planResolve(rc) != ResolveWay::Skip) {
            bool pending = false, whole = false;
            TextureId pendingTex;
            for (const PendingClear& p : pendingClears)
                if (p.texture == to->id || (to->ms && p.texture == to->ms)) pending = true, pendingTex = p.texture, whole = p.whole && p.rects.empty();
            if (pending) {
                if (whole && from->width == to->width && from->height == to->height && dst.sliceStart == 0 && n >= to->layers) {
                    PendingClear dropped;
                    takePendingClear(pendingTex, dropped);
                    ++r.cpu_.resolveDestClearDropped;
                } else {
                    flushClear(*to);
                    ++r.cpu_.resolveDestClearFlushed;
                }
            }
        }
        closePass();
        vpgpu::RenderPass* last = rc.sourceMultisampled && !twinWanted ? lastPassWriting(from->ms, to->id) : nullptr;
        // The pass can take the resolve over only when it is an attachment of exactly this slice (mip 0, the
        // source's first layer), when it resolves nowhere else (nothing, the source's own picture, or already this
        // destination slice), and when one layer is all there is: an attachment resolves a single slice.
        if (last) {
            bool fits = n == 1;
            bool found = false;
            for (const vpgpu::Attachment& a : last->desc.colors) {
                if (a.texture != from->ms) continue;
                found = true;
                const bool ownSlice = a.mip == 0 && a.layer == src.sliceStart;
                const bool freeOrOurs = !a.resolve || a.resolve == from->id ||
                                        (a.resolve == to->id && a.resolveMip == 0 && a.resolveLayer == dst.sliceStart);
                if (!ownSlice || !freeOrOurs) fits = false;
            }
            if (!found || !fits) last = nullptr;
        }
        rc.lastPassOpenForSource = last != nullptr;
        const ResolveWay way = planResolve(rc);
        noteResolveSource(src, dst, *from, freshSource,
                          way == ResolveWay::InPass ? "in_pass" : way == ResolveWay::OwnPass ? "own_pass" : way == ResolveWay::Copy ? "copy" : "skip");
        switch (way) {
            case ResolveWay::Skip: return skip(differ && !from->ms ? "resolve: formats differ and the source is single-sample"
                                                            : "resolve: destination is the source");
            case ResolveWay::InPass:
                for (vpgpu::Attachment& a : last->desc.colors)
                    if (a.texture == from->ms) {
                        // The picture nobody was reading gives up this pass's resolve: should it be read after all
                        // in this frame, textureRef catches it up (catchUpTwin).
                        if (a.resolve == from->id) {
                            from->twinStale = r.frame_;
                            from->staleFirst = a.resolveLayer;
                            from->staleCount = std::max<uint16_t>(src.sliceCount, 1);
                            life.noteStale(from->id, r.frame_);
                        }
                        a.resolve = to->id;
                        a.resolveRequired = true;
                        a.resolveLayer = dst.sliceStart;
                        a.store = vpgpu::StoreAction::StoreAndResolve;
                        a.keepSamples = true;
                    }
                note("MSAA resolve done by the pass that drew the samples");
                break;
            case ResolveWay::OwnPass: {
                // One pass per layer: an attachment resolves a single slice (PSVR eye arrays have two).
                for (uint16_t l = 0; l < n; ++l) {
                    vpgpu::RenderPassDesc desc;
                    vpgpu::Attachment& a = desc.colors[0];
                    a.texture = from->ms;
                    a.layer = uint16_t(src.sliceStart + l);
                    a.load = vpgpu::LoadAction::Load;
                    a.store = vpgpu::StoreAction::Resolve;
                    a.resolve = to->id;
                    a.resolveRequired = true;
                    a.resolveLayer = uint16_t(dst.sliceStart + l);
                    desc.label = "resolve " + hex(src.address) + " -> " + hex(dst.address);
                    frame.beginRenderPass(desc);
                    frame.endRenderPass();
                    ++r.stats_.passes;
                }
                note("MSAA resolve in a pass of its own");
                break;
            }
            case ResolveWay::Copy: {
                // (a multisampled source only gets here while its single-sample picture is wanted anyway:
                // young or read by others, so the passes that drew into it resolve into it)
                vpgpu::Copy c;
                c.kind = vpgpu::CopyKind::TextureToTexture;
                c.srcTexture.texture = from->id;
                c.srcTexture.layer = src.sliceStart;
                c.dstTexture.texture = to->id;
                c.dstTexture.layer = dst.sliceStart;
                c.srcTexture.layerCount = c.dstTexture.layerCount = n;
                c.srcTexture.width = c.dstTexture.width = std::min(from->width, to->width);
                c.srcTexture.height = c.dstTexture.height = std::min(from->height, to->height);
                c.label = "resolve " + hex(src.address) + " -> " + hex(dst.address);
                frame.copy(c);
                ++r.stats_.copies;
                break;
            }
        }
        latestColor[dst.address] = to->id;
        life.noteRead(from->ms ? from->ms : from->id, r.frame_);  // (the source is read by the resolve)
        markWriteWid(toWid);
        if (resolveProbe.due(r.frame_)) {  // P2: what the resolve took and what it wrote, and the other targets over the destination
            std::string others;
            for (const Target& t : colorTargets[dst.address])
                if (t.wid != toWid)
                    others += (others.empty() ? "" : "; ") + describeResolveTarget(t);
            log("vpps4.probe P2 resolve f=" + std::to_string(r.frame_) + " src_req=" + std::to_string(src.width) + "x" + std::to_string(src.height) +
                " from id=" + std::to_string(fromWid) + " last_draw=" + (from->lastDraw == ~uint64_t(0) ? std::string("never") : std::to_string(from->lastDraw)) +
                " " + std::to_string(from->width) + "x" + std::to_string(from->height) + " relaxed=" + (fromSmaller ? "1" : "0") +
                " | to id=" + std::to_string(toWid) + " " + std::to_string(to->width) + "x" + std::to_string(to->height) +
                " dst_req=" + std::to_string(dst.width) + "x" + std::to_string(dst.height) + " | others_at_dst=[" + others + "]");
        }
    }

    bool fill(uint64_t address, uint64_t bytes, uint32_t value) {
        if (guestReadOnly) return false;
        const Who who(touchWho, "fill");
        BufferId id;
        uint64_t offset = 0, available = 0;
        if (!bytes) return true;
        if (!guestBuffer(address, bytes, id, offset, available) || bytes > available) {
            note("DMA fill outside guest memory: dropped");
            return false;
        }
        syncOwners(address, bytes, true);  // (images it covers completely are simply overwritten)
        closePass();
        vpgpu::Copy c;
        c.dstBuffer = {id, offset, bytes, 0, 0};
        c.label = "fill " + hex(address);
        const uint8_t b0 = uint8_t(value);
        if (value == uint32_t(b0) * 0x01010101u) {
            c.kind = vpgpu::CopyKind::FillBuffer;
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
            c.kind = vpgpu::CopyKind::BufferToBuffer;
            c.srcBuffer = {sid, so, bytes, 0, 0};
        }
        frame.copy(c);
        ++r.stats_.copies;
        ++programGeneration;  // (code may be what was written)
        tracker.markGpuWrite(address, bytes, MemoryTracker::kDma);
        return true;
    }

    bool copy(uint64_t dst, uint64_t src, uint64_t bytes) {
        if (guestReadOnly) return false;
        if (!bytes) return true;
        const Who who(touchWho, "copy");
        BufferId di, si;
        uint64_t doff = 0, dav = 0, soff = 0, sav = 0;
        if (!guestBuffer(dst, bytes, di, doff, dav) || bytes > dav || !guestBuffer(src, bytes, si, soff, sav) || bytes > sav) {
            note("DMA copy outside guest memory: dropped");
            return false;
        }
        syncOwners(src, bytes, false);
        syncOwners(dst, bytes, true);
        closePass();
        vpgpu::Copy c;
        c.kind = vpgpu::CopyKind::BufferToBuffer;
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
        if (!size) return;
        // The whole address space (size field all ones, or a range beyond what can be mapped at once):
        // every program is looked at again and every texture the host cannot watch is checked in full.
        const bool everything = size >= kMaxRawExtent;
        if (everything) {
            address = 0;
            size = ~uint64_t(0);
            ++programGeneration;
        } else {
            forgetPrograms(address, size);
        }
        auto it = everything ? textures.begin() : textures.lower_bound(address > (1ull << 30) ? address - (1ull << 30) : 0);
        for (; it != textures.end() && (everything || it->first < address + size); ++it)
            for (auto& [shape, t] : it->second)
                if (!t.sourceWatched && (everything || it->first + std::max<uint64_t>(t.bytes, 1) > address)) {
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

    vpgpu::FrameReport submit() {
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
            busy.clear(), busyWho.clear();  // (ranges bound by draws that were skipped afterwards)
            busyAll = false;
            touchedLo = touchedHi = 0;  // (touch() skips ranges inside the last one it added: that set is gone)
            return {};
        }
        const auto t0 = Clock::now();
        // Load/store/resolve actions for a tile-based GPU, now that the whole frame is known.
        {
            if (!dropSinkSet) {
                dropSinkSet = true;
                life.setResolveDropSink([this](TextureId ms, TextureId twin, const char* why) { noteResolveDropped(ms, twin, why); });
                life.setProbeSink([this](const std::string& l) { log(l); });
            }
            const auto tLife = Clock::now();
            const LifetimeStats ls = life.plan(frame, r.device_.resources(), r.frame_);
            r.cpu_.lifeMs += msSince(tLife);
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
        vpgpu::FrameReport report = r.device_.submit(std::move(frame));
        r.cpu_.submitMs += msSince(t0);
        scratchDone[ringIndex] = report.frame + 1;
        frame = vpgpu::Frame{};
        retireOwned();
        // P3 (memory): once per 600 frames, what the engine holds (the memory growth of a long run).
        if (memProbeFrame == ~uint64_t(0) || r.frame_ >= memProbeFrame + 600) {
            memProbeFrame = r.frame_;
            uint64_t targets = 0, msBytes = 0, gpuBytes = 0, liveTextures = 0;
            for (const auto* cache : {&colorTargets, &depthTargets})
                for (const auto& [a, list] : *cache)
                    for (const Target& t : list) {
                        ++targets;
                        gpuBytes += gpuBytesOf(t);
                        if (t.samples > 1) msBytes += uint64_t(t.width) * t.height * std::max<uint32_t>(vpgpu::formatInfo(t.format).bytesPerBlock, 1) * t.layers * t.samples;
                    }
            log("vpps4.probe P3 mem f=" + std::to_string(r.frame_) + " staging_live=" + std::to_string(stagingLive() >> 20) + "MB stagings=" +
                std::to_string(stagings.size()) + " targets=" + std::to_string(targets) + " ms_bytes=" + std::to_string(msBytes >> 20) + "MB");
            // P7 (what the engine keeps track of): the tracker's intervals, forgetHistory calls, ring chains, the unaligned
            // constant buffer bytes copied since the last line, the GPU bytes of live images (gpuBytesOf) and how many.
            for (const auto& [a, m] : textures)
                for (const auto& [k, t] : m) ++liveTextures, gpuBytes += gpuBytesOf(t);
            log("VPKENGINE_PROBE P7 f=" + std::to_string(r.frame_) + " tracker_spans=" + std::to_string(tracker.intervals()) + " tracker_spans_max=" +
                std::to_string(tracker.peakIntervals()) + " forget_history=" + std::to_string(tracker.forgetHistoryCalls()) + " ring_chains=" +
                std::to_string(r.stats_.ringChains) + " aligned_copy_bytes=" + std::to_string(alignedCopyBytes - alignedCopyBytesLogged) +
                " gpu_bytes=" + std::to_string(gpuBytes) + " targets=" + std::to_string(targets) + " textures=" + std::to_string(liveTextures) +
                " staging_live=" + std::to_string(stagingLive() >> 20) + "MB");
            alignedCopyBytesLogged = alignedCopyBytes;
        }
        if (!busy.empty() || busyAll) {
            inFlight.push_back({report.frame, std::move(busy), busyAll, std::move(busyWho)});
            busy.clear();
            busyWho.clear();
            busyAll = false;
            touchedLo = touchedHi = 0;
            if (inFlight.size() > 64) {  // a GPU far behind: fold the oldest into one "everything"
                inFlight.pop_front();
                inFlight.front().all = true;
            }
        }
        {
            const uint64_t used = std::max(ringUsed + rings[ringIndex].head, ringNeed);
            pstat.ringHighWater = std::max(pstat.ringHighWater, used);
            uint64_t want = kRingBytes;
            while (want < used + used / 4 && want < kRingSteadyMax) want *= 2;
            ringWant = want;
            ringUsed = 0;
            ringNeed = 0;
        }
        ringIndex = (ringIndex + 1) % kFrames;
        rings[ringIndex].head = 0;
        sizeRing(rings[ringIndex]);
        rewindScratch(ringIndex);
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
            if (x.pooled) {
                stagingPool.push_back({x.memory, x.id, x.pooled, r.frame_});
                return true;
            }
            r.device_.destroy(x.id);
            std::free(x.memory);
            return true;
        });
        poolTrim();
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
        for (Pooled& x : stagingPool) {
            r.device_.destroy(x.id);
            std::free(x.memory);
        }
        for (auto& slabs : scratchSlots)
            for (ScratchSlab& x : slabs) {
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

Renderer::Renderer(vpgpu::Device& device, Hooks hooks)
    : device_(device), hooks_(std::move(hooks)), impl_(std::make_unique<Impl>(*this)) {
    if (!formatCaps().bc) {
        impl_->stagingLimit = 128ull << 20;  // (every block-compressed texture is decoded into it)
        impl_->tiler().prefetchBc();         // (block-compressed textures are decoded by that kernel)
    }
}

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
void Renderer::setGuestReadOnly(bool on) { impl_->guestReadOnly = on; }
void Renderer::setImageAlias(bool on) { impl_->imageAlias = on; }

void Renderer::guardRunaway() {
    if (++sinceSubmit_ < kRunawayCommands) return;
    if (!runawayTold_ && hooks_.log)
        hooks_.log("gpu.err what=frame_not_ended commands=" + std::to_string(sinceSubmit_) +
                   " (the host never called submit: submitted here)");
    runawayTold_ = true;
    submit();
}

// Collects what a call resolved for Hooks::resolved (nothing without the hook).
struct Renderer::ResolvedScope {
    Renderer& r;
    std::optional<Resolved> out;
    ResolvedScope(Renderer& r_, Resolved::Kind kind) : r{r_} {
        if (!r.hooks_.resolved) return;
        out.emplace();
        out->kind = kind;
        r.impl_->resolved_ = &*out;
    }
    ~ResolvedScope() {
        if (!out) return;
        r.impl_->resolved_ = nullptr;
        try {
            r.hooks_.resolved(*out);
        } catch (...) {
            r.impl_->resolved_ = nullptr;
        }
    }
};

void Renderer::draw(const DrawState& d) {
    const ResolvedScope scope(*this, Resolved::Kind::Draw);
    const auto t0 = Clock::now();
    impl_->draw(d);
    cpu_.drawMs += msSince(t0);
    ++cpu_.calls;
    guardRunaway();
}
void Renderer::dispatch(const DispatchState& d) {
    const ResolvedScope scope(*this, Resolved::Kind::Dispatch);
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
void Renderer::abandonOwners(uint64_t address, uint64_t size) { impl_->abandonOwners(address, size); }
bool Renderer::dropGuestBuffers() { return impl_->dropGuestBuffers(); }
void Renderer::coherencePoint(uint64_t address, uint64_t size) { impl_->coherencePoint(address, size); }
void Renderer::noteGpuWrite(uint64_t address, uint64_t size) {
    impl_->syncOwners(address, size, true);
    impl_->tracker.markGpuWrite(address, size, MemoryTracker::kGpuRaw);
}
void Renderer::setMemoryLimits(uint64_t stagingBytes, uint64_t geometryScratchBytes, uint64_t bcDecodedBytesPerFrame) {
    impl_->stagingLimit = stagingBytes;
    impl_->scratchLimit = geometryScratchBytes;
    impl_->bcFrameBytes = bcDecodedBytesPerFrame;
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
void Renderer::submittedIdle() {
    impl_->inFlight.clear();
    impl_->retireOwned(true);
}
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
    impl_->retireOwned();
}
std::string Renderer::busyText(uint64_t a, uint64_t n, uint64_t b, uint64_t m) const { return impl_->busyText(a, n, b, m); }
void Renderer::noteCpuSync(const char* caller, uint64_t a, uint64_t n, uint64_t b, uint64_t m, const std::string& busy, uint64_t frames,
                           bool submitted, double ms) {
    impl_->noteCpuSync(caller, a, n, b, m, busy, frames, submitted, ms);
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
    impl_->busyWho.clear();
    impl_->busyAll = false;
    impl_->touchedLo = impl_->touchedHi = 0;
    impl_->inFlight.clear();
    impl_->retireOwned(true);
}

CpuTimes Renderer::takeCpuTimes() {
    CpuTimes t = cpu_;
    cpu_ = {};
    const vpgpu::Device::SubmitTimes d = device_.takeSubmitTimes();
    t.holdMs = d.holdMs;
    t.compileMs = d.compileMs;
    t.backendMs = d.backendMs;
    t.slotWaitMs = d.slotWaitMs;
    t.holdFrames = d.holdFrames;
    t.holdTimeouts = d.holdTimeouts;
    return t;
}

namespace {

uint64_t textureBytes(const vpgpu::TextureDesc& d) {
    const vpgpu::FormatInfo& fi = vpgpu::formatInfo(d.format);
    const uint64_t bw = (d.width + std::max<uint32_t>(fi.blockSize, 1) - 1) / std::max<uint32_t>(fi.blockSize, 1);
    const uint64_t bh = (d.height + std::max<uint32_t>(fi.blockHeight, 1) - 1) / std::max<uint32_t>(fi.blockHeight, 1);
    uint64_t b = bw * bh * std::max<uint32_t>(d.depth, 1) * std::max<uint32_t>(d.layers, 1) * std::max<uint32_t>(fi.bytesPerBlock, 1) *
                 std::max<uint32_t>(d.samples, 1);
    if (d.mips > 1) b += b / 3;
    return b;
}

}  // namespace

void Renderer::publishProbe(uint64_t pictures) {
    namespace pr = vpgpu::probe;
    if (!pr::enabled()) return;
    Impl& im = *impl_;
    Impl::ProbeStats& ps = im.pstat;
    const double dPictures = double(std::max<uint64_t>(pictures - ps.lastPictures, 1));
    ps.lastPictures = pictures;
    const auto C = [](const char* fam, const std::string& name, uint64_t v) { pr::counter(fam, pr::Registry::sanitize(name)).set(v); };
    const auto G = [](const char* fam, const std::string& name, double v) { pr::gauge(fam, pr::Registry::sanitize(name)).set(v); };
    const auto MB = [](uint64_t b) { return double(b) / double(1 << 20); };
    // (the description is copied under the device's resource lock: this runs on the recording thread, the backend's submit thread may be creating or destroying)
    const auto bytesOf = [&](vpgpu::TextureId id) -> uint64_t {
        vpgpu::TextureDesc d;
        return id && device_.textureDesc(id, d) ? textureBytes(d) : 0;
    };
    // Open-ended families (reasons, formats): a name gets a column when first seen, at most kMaxNames of them; the rest add to "other".
    constexpr size_t kMaxNames = 300;
    const auto reasons = [&](const char* fam, const char* prefix, const std::map<std::string, uint64_t>& m) {
        uint64_t total = 0, other = 0;
        for (const auto& [why, n] : m) {
            total += n;
            const std::string key = std::string(fam) + "/" + pr::Registry::sanitize(why);
            if (!ps.published.count(key) && ps.published.size() >= kMaxNames) {
                other += n;
                continue;
            }
            ps.published.insert(key);
            const std::string name = std::string(prefix) + why;
            C(fam, name, n);
            uint64_t& last = ps.lastReason[key];
            G(fam, name + "_pf", double(n - std::min(n, last)) / dPictures);
            last = n;
        }
        C(fam, std::string(prefix) + "total", total);
        C(fam, std::string(prefix) + "other", other);
        uint64_t& last = ps.lastReason[std::string(fam) + "/total"];
        G(fam, std::string(prefix) + "total_pf", double(total - std::min(total, last)) / dPictures);
        last = total;
    };
    reasons("SKIP", "skip_", stats_.skipped);
    reasons("HANDLED", "handled_", stats_.handled);
    reasons("NOTE", "note_", stats_.notes);

    // ---- RATES: cumulative counts (the registry derives the interval deltas)
    {
        const Stats& s = stats_;
        const char* f = "RATES";
        C(f, "draws", s.draws), C(f, "drawn", s.drawn), C(f, "dispatches", s.dispatches), C(f, "dispatched", s.dispatched);
        C(f, "passes", s.passes), C(f, "copies", s.copies), C(f, "resolves", s.resolves);
        C(f, "gs_hoisted", s.gsHoisted), C(f, "gs_split", s.gsSplit);
        C(f, "shaders", s.shaders), C(f, "shaders_cached", s.shadersCached), C(f, "shader_errors", s.shaderErrors);
        C(f, "textures_made", s.textures), C(f, "targets_made", s.targets), C(f, "uploads", s.uploads), C(f, "upload_bytes", ps.uploadBytes);
        C(f, "detiled", s.detiled), C(f, "gpu_detiled", s.gpuDetiled), C(f, "gpu_retiled", s.gpuRetiled), C(f, "indirect", s.indirect);
        C(f, "ring_chains", s.ringChains), C(f, "refilled", s.refilled), C(f, "retiled", s.retiled), C(f, "evicted", s.evicted);
        C(f, "views_made", s.viewsMade), C(f, "syncs", s.syncs), C(f, "seeded", s.seeded);
        C(f, "pictures", pictures);
        C(f, "aligned_copies", ps.alignedCopies), C(f, "aligned_copy_bytes", im.alignedCopyBytes);
        C(f, "life_stores_dropped", im.lifeTotal.storesDropped);
        C(f, "life_samples_dropped", im.lifeTotal.samplesDropped), C(f, "life_resolves_dropped", im.lifeTotal.resolvesDropped);
        C(f, "life_resolves_deferred", im.lifeTotal.resolvesDeferred), C(f, "life_misses", im.lifeTotal.misses);
        if (im.tilerPtr) {
            const GpuTiler::Counts& tc = im.tilerPtr->counts();
            static const char* const kernels[3] = {"detile", "retile", "bcdecode"};
            for (int k = 0; k < 3; ++k)
                C(f, std::string("tiler_") + kernels[k], tc.dispatches[k]), C(f, std::string("tiler_") + kernels[k] + "_bytes", tc.bytes[k]);
            C(f, "tiler_not_ready", tc.notReady);
        }
    }

    // ---- ENG: what the renderer holds
    {
        const char* f = "ENG";
        // Guest no-copy window buffers
        uint64_t gb = 0, gbBytes = 0, failed = 0;
        uint64_t bucket[5] = {}, bucketBytes[5] = {};
        for (const auto& [key, b] : im.guestBuffers) {
            if (b.failed) ++failed;
            if (!b.id) continue;
            ++gb, gbBytes += b.size;
            const unsigned k = b.size < (64u << 10) ? 0 : b.size < (1u << 20) ? 1 : b.size < (16u << 20) ? 2 : b.size < (256u << 20) ? 3 : 4;
            ++bucket[k], bucketBytes[k] += b.size;
        }
        G(f, "guest_bufs", double(gb)), G(f, "guest_buf_mb", MB(gbBytes)), G(f, "guest_buf_entries", double(im.guestBuffers.size()));
        G(f, "guest_buf_failed", double(failed)), G(f, "guest_buf_retired", double(im.retired.size()));
        static const char* const bn[5] = {"lt64k", "lt1m", "lt16m", "lt256m", "ge256m"};
        for (int k = 0; k < 5; ++k)
            G(f, std::string("guest_buf_") + bn[k], double(bucket[k])), G(f, std::string("guest_buf_") + bn[k] + "_mb", MB(bucketBytes[k]));
        C(f, "guest_buf_adds", ps.guestBufAdds), C(f, "guest_buf_removes", ps.guestBufRemoves), C(f, "guest_buf_refused", ps.guestBufRefused);
        // Staging, scratch, upload ring
        G(f, "staging_live", double(im.stagings.size())), G(f, "staging_mb", MB(im.stagingLive())), G(f, "staging_limit_mb", MB(im.stagingLimit));
        uint64_t poolBytes = 0;
        for (const auto& x : im.stagingPool) poolBytes += x.cls;
        G(f, "staging_pool", double(im.stagingPool.size())), G(f, "staging_pool_mb", MB(poolBytes));
        C(f, "staging_pool_hits", im.stagingPoolHits), C(f, "staging_pool_misses", im.stagingPoolMisses);
        uint64_t slabs = 0;
        for (const auto& v : im.scratchSlots) slabs += v.size();
        G(f, "scratch_slabs", double(slabs)), G(f, "scratch_mb", MB(im.scratchBytes)), G(f, "scratch_limit_mb", MB(im.scratchLimit));
        uint64_t ringBytes = 0, ringHead = 0;
        for (const auto& rg : im.rings) ringBytes += rg.memory ? rg.cap : 0, ringHead += rg.head;
        G(f, "ring_mb", MB(ringBytes)), G(f, "ring_used_mb", MB(ringHead)), G(f, "ring_chain_mb", MB(im.ringUsed));
        G(f, "ring_high_water_mb", MB(ps.ringHighWater)), G(f, "ring_want_mb", MB(im.ringWant));
        // Targets
        const auto targets = [&](const char* tag, const std::map<uint64_t, std::vector<Impl::Target>>& m) {
            uint64_t n = 0, ms = 0, single = 0, msBytes = 0, singleBytes = 0, twinPending = 0;
            for (const auto& [addr, v] : m)
                for (const Impl::Target& t : v) {
                    ++n;
                    if (t.twinStale != ~uint64_t(0)) ++twinPending;
                    if (const uint64_t b = bytesOf(t.id)) ++single, singleBytes += b;
                    if (const uint64_t b = bytesOf(t.ms)) ++ms, msBytes += b;
                }
            const std::string t = tag;
            G(f, t + "_n", double(n)), G(f, t + "_single", double(single)), G(f, t + "_single_mb", MB(singleBytes));
            G(f, t + "_ms", double(ms)), G(f, t + "_ms_mb", MB(msBytes)), G(f, t + "_twin_stale", double(twinPending));
        };
        targets("rt", im.colorTargets);
        targets("depth", im.depthTargets);
        // Textures: by type, mip count, samples and host format
        static const char* const typeNames[4] = {"1d", "2d", "3d", "cube"};
        uint64_t texN = 0, texBytes = 0, byType[4] = {}, byTypeBytes[4] = {}, byMips[3] = {}, bySamples[4] = {}, unfilled = 0, seeded = 0;
        uint64_t storage = 0, softBc = 0, mirrored = 0;
        std::map<std::string, std::pair<uint64_t, uint64_t>> byFormat;
        for (const auto& [addr, shapes] : im.textures)
            for (const auto& [shape, t] : shapes) {
                if (!t.id) continue;
                ++texN;
                const uint64_t b = bytesOf(t.id);
                texBytes += b;
                const unsigned ty = std::min<unsigned>(unsigned(t.type), 3);
                ++byType[ty], byTypeBytes[ty] += b;
                ++byMips[t.mips <= 1 ? 0 : t.mips <= 4 ? 1 : 2];
                ++bySamples[t.memSamples >= 8 ? 3 : t.memSamples >= 4 ? 2 : t.memSamples >= 2 ? 1 : 0];
                auto& fm = byFormat[vpgpu::formatInfo(t.format).name];
                ++fm.first, fm.second += b;
                if (!t.filled) ++unfilled;
                if (t.seeded) ++seeded;
                if (t.storage) ++storage;
                if (t.softBc) ++softBc;
                if (t.mirrorSrc) ++mirrored;
            }
        G(f, "tex_n", double(texN)), G(f, "tex_mb", MB(texBytes)), G(f, "tex_unfilled", double(unfilled)), G(f, "tex_seeded", double(seeded));
        G(f, "tex_storage", double(storage)), G(f, "tex_soft_bc", double(softBc)), G(f, "tex_mirror_copies", double(mirrored));
        for (int k = 0; k < 4; ++k)
            G(f, std::string("tex_") + typeNames[k], double(byType[k])), G(f, std::string("tex_") + typeNames[k] + "_mb", MB(byTypeBytes[k]));
        G(f, "tex_mips1", double(byMips[0])), G(f, "tex_mips2_4", double(byMips[1])), G(f, "tex_mips5p", double(byMips[2]));
        G(f, "tex_samples1", double(bySamples[0])), G(f, "tex_samples2", double(bySamples[1]));
        G(f, "tex_samples4", double(bySamples[2])), G(f, "tex_samples8", double(bySamples[3]));
        for (const auto& [name, v] : byFormat) {
            const std::string key = "ENG/fmt_" + pr::Registry::sanitize(name);
            if (!ps.published.count(key) && ps.published.size() >= kMaxNames) continue;
            ps.published.insert(key);
            G(f, "fmt_" + name, double(v.first)), G(f, "fmt_" + name + "_mb", MB(v.second));
        }
        for (const auto& key : ps.published)  // a format that has no texture now reads 0 (and not its last value)
            if (key.rfind("ENG/fmt_", 0) == 0) {
                const std::string name = key.substr(4);
                bool live = false;
                for (const auto& [n, v] : byFormat) live = live || "fmt_" + pr::Registry::sanitize(n) == name;
                if (!live) G(f, name, 0), G(f, name + "_mb", 0);
            }
        G(f, "tex_retired", double(im.retiredTextures.size())), G(f, "view_retired", double(im.retiredViews.size()));
        G(f, "views", double(im.views.size())), G(f, "samplers", double(im.samplers.size())), G(f, "surface_refs", double(im.surfaceRefs.size()));
        // Hazard tracking and caches
        G(f, "busy_ranges", double(im.busy.size())), G(f, "busy_all", im.busyAll ? 1 : 0), G(f, "in_flight", double(im.inFlight.size()));
        G(f, "snapshots", double(im.snapshots.size())), G(f, "deferred_owned", double(im.deferredOwned.size()));
        G(f, "pending_clears", double(im.pendingClears.size())), G(f, "translated", double(im.translated.size()));
        G(f, "programs", double(im.sourcesByProgram.size())), G(f, "registered_programs", double(im.registered.size()));
        G(f, "resolve_drop_textures", double(im.resolveDropCounts.size())), G(f, "unmapped_buffers", double(im.unmappedBuffers.size()));
        G(f, "rate_maps", double(im.rateMaps.size())), G(f, "depth_aliases", double(im.depthAliases.size()));
        G(f, "tracker_spans", double(im.tracker.intervals())), G(f, "tracker_peak", double(im.tracker.peakIntervals()));
        G(f, "tracker_floor", double(im.tracker.floor())), G(f, "tracker_version", double(im.tracker.current()));
        C(f, "tracker_forget_calls", im.tracker.forgetHistoryCalls()), C(f, "tracker_sweeps", im.tracker.sweeps());
        C(f, "tracker_swept", im.tracker.swept());
        G(f, "frame", double(frame_));
    }

    // ---- QUAL: what the picture loses
    {
        const char* f = "QUAL";
        static const char* const why[3] = {"first", "written_by_others", "forget_history"};
        for (int k = 0; k < 3; ++k)
            C(f, std::string("refill_") + why[k], ps.refills[k]), C(f, std::string("refill_") + why[k] + "_bytes", ps.refillBytes[k]);
        C(f, "refill_unfilled", ps.refillFailed);
        C(f, "refill_skipped", ps.refillSkipped), C(f, "refill_skipped_bytes", ps.refillSkippedBytes);
        static const char* const aliasWhy[Impl::kAliasReasons] = {"depth", "address", "shape", "layout", "mips", "samples", "format", "no_view", "range", "feedback"};
        C(f, "alias_hits", ps.aliasHits), C(f, "alias_hits_legacy", ps.aliasHitsLegacy);
        for (unsigned k = 0; k < Impl::kAliasReasons; ++k) C(f, std::string("alias_rejected_") + aliasWhy[k], ps.aliasRejected[k]);
        static const char* const paths[5] = {"none", "gpu_detile", "cpu_detile", "linear_upload", "not_filled"};
        for (int k = 0; k < 5; ++k)
            C(f, std::string("fill_") + paths[k], ps.fillPath[k]), C(f, std::string("fill_") + paths[k] + "_bytes", ps.fillPathBytes[k]);
        C(f, "mirror_calls", ps.mirrorCalls), C(f, "mirror_copies", ps.mirrorCopies), C(f, "mirror_bytes", ps.mirrorBytes);
        static const unsigned counts[4] = {1, 2, 4, 8};
        for (int a = 0; a < 4; ++a)
            for (int h = 0; h < 4; ++h)
                C(f, "ms_req" + std::to_string(counts[a]) + "_host" + std::to_string(counts[h]), ps.msDraws[a][h]);
        std::map<std::string, uint64_t> drops;
        for (const auto& [text, n] : ps.resolveDrops) drops[text ? text : "unknown"] += n;
        for (const auto& [text, n] : drops) {
            const std::string key = "QUAL/rd_" + pr::Registry::sanitize(text);
            if (!ps.published.count(key) && ps.published.size() >= kMaxNames) continue;
            ps.published.insert(key);
            C(f, "resolve_drop_" + text, n);
        }
        C(f, "tex_srgb_as_unorm", ps.texSrgbAsUnorm), C(f, "tex_inexact", ps.texInexact), C(f, "tex_soft_bc_made", ps.texSoftBc);
        for (const auto& [k, cell] : ps.tilePlans) {
            const std::string tail = std::to_string(cell.tile) + "_" + std::to_string(cell.dfmt) + "_" + std::to_string(cell.nfmt) + "_deg" +
                                     std::to_string(cell.degraded);
            for (int refill = 0; refill < 2; ++refill) {
                const std::string name = std::string(refill ? "tile_refill_" : "tile_plan_") + tail;
                const std::string key = "QUAL/" + pr::Registry::sanitize(name);
                if (!ps.published.count(key) && ps.published.size() >= kMaxNames) continue;
                ps.published.insert(key);
                C(f, name, refill ? cell.refills : cell.textures);
            }
        }
        for (const auto& [k, plan] : ps.texPlans) {
            const std::string name = "plan_" + std::to_string(plan.dfmt) + "_" + std::to_string(plan.nfmt) + "_" + plan.host +
                                     "_f" + std::to_string(plan.flags);
            const std::string key = "QUAL/" + pr::Registry::sanitize(name);
            if (!ps.published.count(key) && ps.published.size() >= kMaxNames) continue;
            ps.published.insert(key);
            C(f, name, plan.count);
        }
        // Compare-sampled T#s (the program expects a depth texture): the route that bound them, the host resource and the draws whose
        // resource is not depth (compare_mismatch). Histogram names are summed by their sanitized form, so two formats that sanitize alike
        // add up in one column instead of overwriting each other.
        static const char* const cmpRoutes[Impl::ProbeStats::kCmpRoutes] = {"depth_target_found", "color_target_found", "texture_from_memory", "alias", "other"};
        C(f, "cmp_binds", ps.cmpBinds), C(f, "cmp_draws", ps.cmpDraws), C(f, "cmp_unresolved", ps.cmpUnresolved);
        C(f, "cmp_host_depth", ps.cmpHostDepth), C(f, "cmp_host_not_depth", ps.cmpHostNotDepth);
        C(f, "compare_mismatch", ps.cmpMismatchDraws), C(f, "compare_mismatch_binds", ps.cmpMismatchBinds);
        for (int k = 0; k < Impl::ProbeStats::kCmpRoutes; ++k) C(f, std::string("cmp_route_") + cmpRoutes[k], ps.cmpRoute[k]);
        for (int k = 0; k < 5; ++k) C(f, std::string("cmp_fill_") + paths[k], ps.cmpFill[k]);
        std::map<std::string, uint64_t> cmpHist;
        for (const auto& [k, cell] : ps.cmpHist)
            cmpHist[pr::Registry::sanitize("cmp_h_" + std::to_string(cell.dfmt) + "_" + cell.host + "_" + cmpRoutes[cell.route])] += cell.count;
        for (const auto& [name, n] : cmpHist) {
            const std::string key = "QUAL/" + name;
            if (!ps.published.count(key) && ps.published.size() >= kMaxNames) continue;
            ps.published.insert(key);
            C(f, name, n);
        }
    }

    // ---- APPROX: draws and dispatches that used a program translated with an approximation
    for (size_t i = 0; i < ps.approxNames.size(); ++i) {
        const std::string key = "APPROX/" + pr::Registry::sanitize(ps.approxNames[i]);
        if (!ps.published.count(key) && ps.published.size() >= kMaxNames) continue;
        ps.published.insert(key);
        C("APPROX", "approx_" + ps.approxNames[i], ps.approxHits[i]);
    }
    C("APPROX", "approx_kinds", ps.approxNames.size());
}

uint64_t Renderer::takeTargetDraws(uint64_t address, uint32_t width, uint32_t height) {
    auto it = impl_->colorTargets.find(address);
    if (it == impl_->colorTargets.end()) return 0;
    uint64_t n = 0;
    for (Impl::Target& t : it->second)
        if ((!width || t.width == width) && (!height || t.height == height)) n += t.draws - t.drawsTaken, t.drawsTaken = t.draws;
    return n;
}

vpgpu::FrameReport Renderer::submit() {
    static vpgpu::probe::Hist& submitCallHist = vpgpu::probe::hist("TIME", "submit_call_us");
    const vpgpu::probe::ScopedHist submitCallScope(submitCallHist);
    sinceSubmit_ = 0;
    vpgpu::FrameReport report = impl_->submit();
    ++frame_;
    if (summaryEvery_ && frame_ % summaryEvery_ == 0) {
        std::string line = "vpps4.frame n=" + std::to_string(frame_) + " draws=" + std::to_string(stats_.drawn) + "/" +
                           std::to_string(stats_.draws) + " dispatches=" + std::to_string(stats_.dispatched) + "/" +
                           std::to_string(stats_.dispatches) + " passes=" + std::to_string(stats_.passes) +
                           " shaders=" + std::to_string(stats_.shaders) + " shader_errors=" + std::to_string(stats_.shaderErrors) +
                           " textures=" + std::to_string(stats_.textures) + " detiled=" + std::to_string(stats_.detiled) + " gpu_detiled=" + std::to_string(stats_.gpuDetiled) + " indirect=" + std::to_string(stats_.indirect) + " targets=" + std::to_string(stats_.targets) +
                           " uploads=" + std::to_string(stats_.uploads) + " ring_chains=" + std::to_string(stats_.ringChains) + " resolves=" + std::to_string(stats_.resolves) +
                           " stores_dropped=" + std::to_string(impl_->lifeTotal.storesDropped) + " samples_dropped=" +
                           std::to_string(impl_->lifeTotal.samplesDropped) + " resolves_dropped=" +
                           std::to_string(impl_->lifeTotal.resolvesDropped) + " life_misses=" +
                           std::to_string(impl_->lifeTotal.misses);
        uint64_t skippedTotal = 0, handledTotal = 0;
        for (const auto& [why, n] : stats_.skipped) line += " skip[" + why + "]=" + std::to_string(n), skippedTotal += n;
        for (const auto& [why, n] : stats_.handled) line += " handled[" + why + "]=" + std::to_string(n), handledTotal += n;
        // draws + dispatches that left no trace at all: neither drawn, skipped nor handled (should stay 0)
        const int64_t unexplained = int64_t(stats_.draws + stats_.dispatches) - int64_t(stats_.drawn + stats_.dispatched) -
                                    int64_t(skippedTotal) - int64_t(handledTotal);
        const vpgpu::PipelineStats ps = device_.pipelines().totals();
        line += " skipped=" + std::to_string(skippedTotal) + " handled=" + std::to_string(handledTotal) +
                " unexplained=" + std::to_string(unexplained) + " psos_compiled=" + std::to_string(ps.compiled) +
                " psos_failed=" + std::to_string(ps.failed);
        if (hooks_.log) hooks_.log(line);
    }
    return report;
}

vpgpu::TextureId Renderer::textureFor(const ImageDesc& d, bool integerOnly) {
    impl_->probePass.clear();
    impl_->attachedWids.clear();  // (asked outside a draw)
    const auto ref = impl_->textureRef(d, false, false, integerOnly);
    const vpgpu::TextureId t = ref.view ? ref.view : ref.id;
    impl_->life.noteSticky(t, frame_);  // shown by the host: always stored
    return t;
}
bool Renderer::fill(uint64_t address, uint64_t bytes, uint32_t value) {
    const ResolvedScope scope(*this, Resolved::Kind::Fill);
    const bool ok = impl_->fill(address, bytes, value);
    if (!ok) impl_->resolvedStatus(Resolved::Status::Skipped, "range is not guest memory");
    return ok;
}
bool Renderer::copy(uint64_t dst, uint64_t src, uint64_t bytes) {
    const ResolvedScope scope(*this, Resolved::Kind::Copy);
    const bool ok = impl_->copy(dst, src, bytes);
    if (!ok) impl_->resolvedStatus(Resolved::Status::Skipped, "range is not guest memory");
    return ok;
}

uint32_t* Renderer::gds() {
    return impl_->gdsBuffer() ? static_cast<uint32_t*>(impl_->gds.memory) : nullptr;
}

std::string Renderer::describeAt(uint64_t address, uint64_t bytes) const { return impl_->describe(address, bytes); }

void Renderer::setProbes(bool on) {
    impl_->probeOn = on;
    impl_->probes.clear();
}

std::vector<Renderer::DrawProbe> Renderer::takeProbes() {
    std::vector<DrawProbe> out;
    out.swap(impl_->probes);
    return out;
}

vpgpu::TextureId Renderer::backingAt(const ImageDesc& d, uint16_t& layer, std::string& kind) const {
    layer = 0;
    if (auto it = impl_->latestColor.find(d.address); it != impl_->latestColor.end()) {
        kind = "colour target";
        return it->second;
    }
    if (const Impl::Target* t = impl_->targetSlice(d.address, d.width, d.height, layer)) {
        kind = "colour target slice " + std::to_string(layer);
        return t->id;
    }
    layer = 0;
    if (auto a = impl_->textures.find(d.address); a != impl_->textures.end()) {
        const Impl::Texture* pick = nullptr;
        for (const auto& [key, t] : a->second)
            if (!pick || (t.width == d.width && t.height == d.height)) pick = &t;
        if (pick) {
            kind = pick->storage ? "storage texture" : "texture";
            kind += impl_->tracker.writerAt(a->first) == pick->wid ? " gpu_written" : " from memory";
            return pick->id;
        }
    }
    return {};
}

vpgpu::TextureId Renderer::colorTargetSlice(uint64_t address, uint32_t width, uint32_t height, uint16_t& layer) const {
    layer = 0;
    const Impl::Target* t = impl_->targetSlice(address, width, height, layer);
    if (t) impl_->life.noteSticky(t->id, frame_);
    return t ? t->id : vpgpu::TextureId{};
}

vpgpu::TextureId Renderer::colorTargetAt(uint64_t address) const {
    auto it = impl_->latestColor.find(address);
    if (it == impl_->latestColor.end()) return {};
    impl_->life.noteSticky(it->second, frame_);  // what a flip presents is always stored
    return it->second;
}

}  // namespace vpps4
