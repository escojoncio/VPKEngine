#include "vpps4/renderer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <set>

#include "vpgcn/lift.h"
#include "vpps4/formats.h"
#include "vpps4/tiling.h"

namespace vpps4 {

using vpgpu::Access;
using vpgpu::BufferId;
using vpgpu::ResourceRange;
using vpgpu::SamplerId;
using vpgpu::TextureId;

namespace {

constexpr uint64_t kFnvBasis = 0xcbf29ce484222325ull, kFnvPrime = 0x100000001b3ull;

struct Hash {
    uint64_t h = kFnvBasis;
    void bytes(const void* p, size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * kFnvPrime;
    }
    template <class T>
    void value(const T& v) { bytes(&v, sizeof v); }
    void words(std::span<const uint32_t> w) {
        value(uint64_t(w.size()));
        bytes(w.data(), w.size_bytes());
    }
};

constexpr uint64_t kAddressMask = (uint64_t(1) << 48) - 1;  // guest virtual addresses are 48-bit

// Upload ring: memory of our own, page aligned, aliased as one gpu/core buffer per frame in flight.
// A frame's uploads stay intact until the ring comes round again kFrames submits later.
constexpr unsigned kFrames = 3;
constexpr uint64_t kRingBytes = 32ull << 20;
constexpr uint64_t kPage = 16384;
constexpr uint64_t kGdsBytes = 64 * 1024;
constexpr uint64_t kDummyBytes = 64 * 1024;

vpgpu::Primitive toPrimitive(PrimitiveType p) {
    switch (p) {
        case PrimitiveType::PointList: return vpgpu::Primitive::Points;
        case PrimitiveType::LineList:
        case PrimitiveType::LineLoop: return vpgpu::Primitive::Lines;
        case PrimitiveType::LineStrip: return vpgpu::Primitive::LineStrip;
        case PrimitiveType::TriangleStrip: return vpgpu::Primitive::TriangleStrip;
        default: return vpgpu::Primitive::Triangles;  // lists, and fans/quads/polygons once converted
    }
}

// Primitives Metal lacks: converted to an index list in the upload ring.
bool needsConversion(PrimitiveType p) {
    return p == PrimitiveType::TriangleFan || p == PrimitiveType::Polygon || p == PrimitiveType::QuadList ||
           p == PrimitiveType::QuadStrip || p == PrimitiveType::LineLoop;
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

vpir::Scalar texElem(const ImageDesc& d) {
    return d.numFormat == 4 ? vpir::Scalar::U32 : d.numFormat == 5 ? vpir::Scalar::I32 : vpir::Scalar::F32;
}

vpgpu::BindKind bindKind(vpir::ResourceKind k) {
    switch (k) {
        case vpir::ResourceKind::Buffer: return vpgpu::BindKind::Buffer;
        case vpir::ResourceKind::ConstantBuffer: return vpgpu::BindKind::ConstantBuffer;
        case vpir::ResourceKind::Texture: return vpgpu::BindKind::Texture;
        case vpir::ResourceKind::Sampler: return vpgpu::BindKind::Sampler;
        case vpir::ResourceKind::StorageTexture: return vpgpu::BindKind::StorageTexture;
    }
    return vpgpu::BindKind::Auto;
}

}  // namespace

struct Renderer::Impl {
    explicit Impl(Renderer& owner) : r(owner) {}
    Renderer& r;
    vpgpu::Frame frame;

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

    bool region(uint64_t address, uint64_t size, GuestRegion& out) {
        if (!r.hooks_.region || !r.hooks_.region(address, size ? size : 1, out)) return false;
        return address >= out.base && address + size <= out.base + out.size;
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
    void touch(uint64_t a, uint64_t n) {
        if (busyAll || !n) return;
        uint64_t lo = a, hi = a + n < a ? ~uint64_t(0) : a + n;
        auto it = busy.upper_bound(lo);
        if (it != busy.begin()) {
            auto prev = std::prev(it);
            if (prev->second >= lo) {
                if (prev->second >= hi) return;  // already covered
                lo = prev->first;
                it = busy.erase(prev);
            }
        }
        while (it != busy.end() && it->first <= hi) {
            hi = std::max(hi, it->second);
            it = busy.erase(it);
        }
        busy[lo] = hi;
        if (busy.size() > kBusyRanges) {
            busy.clear();
            busyAll = true;
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

    // ---- Diagnostics ----------------------------------------------------------------------------

    static std::string hex(uint64_t v) {
        char b[24];
        std::snprintf(b, sizeof b, "0x%llx", static_cast<unsigned long long>(v));
        return b;
    }
    void log(const std::string& s) {
        if (r.hooks_.log) r.hooks_.log(s);
    }
    void skip(const std::string& reason) {
        if (r.stats_.skipped[reason]++ == 0) log("vpps4.first skip=\"" + reason + "\"");
    }
    void note(const std::string& n) {
        if (r.stats_.notes[n]++ == 0) log("vpps4.first note=\"" + n + "\"");
    }

    // ---- Shaders --------------------------------------------------------------------------------

    struct Translated {
        bool ok = false;
        std::string error;
        uint64_t id = 0;
        std::vector<vpgcn::Binding> bindings;
        std::vector<vpir::Resource> resources;  // Module::resources (kind index, access)
        std::optional<uint32_t> userDataSlot;
    };
    std::unordered_map<uint64_t, std::vector<vpgcn::Binding>> sourcesByProgram;  // level 1
    std::unordered_map<uint64_t, Translated> translated;                          // level 2
    std::set<uint64_t> registered;

    static vpgcn::LiftInput liftInput(const Stage& s, vpgcn::ShaderStage stage) {
        vpgcn::LiftInput in;
        in.stage = stage;
        in.code = s.code;
        in.fetch = s.fetch;
        in.userSgprs = s.userSgprs;
        in.vgprCompCnt = s.vgprCompCnt;
        in.psInputAddr = s.psInputAddr;
        in.instanceStepRate[0] = s.instanceStepRate[0];
        in.instanceStepRate[1] = s.instanceStepRate[1];
        for (unsigned k = 0; k < 3; ++k) {
            in.tgidEnable[k] = s.tgidEnable[k];
            in.workgroupSize[k] = s.workgroupSize[k];
        }
        in.tgSizeEnable = s.tgSizeEnable;
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

    static uint64_t programKey(const Stage& s, vpgcn::ShaderStage stage) {
        Hash h;
        h.value(stage);
        h.words(s.code);
        h.words(s.fetch);
        h.value(s.userSgprs);
        h.value(s.vgprCompCnt);
        if (stage == vpgcn::ShaderStage::Pixel) {
            h.value(s.psInputAddr);
            h.value(s.exportFormat);
            h.value(s.psNumInterp);
            h.bytes(s.psInputCntl.data(), 4 * std::min<size_t>(s.psNumInterp, 32));
        }
        if (stage == vpgcn::ShaderStage::Vertex) h.value(s.instanceStepRate);
        if (stage == vpgcn::ShaderStage::Compute) {
            h.value(s.workgroupSize);
            h.value(s.tgidEnable);
            h.value(s.tgSizeEnable);
            h.value(s.tidigCompCnt);
            h.value(s.ldsBytes);
        }
        return h.h;
    }

    // The dwords of a descriptor (dwords > 0) at its source, at draw time.
    bool readDescriptor(const Stage& s, const vpgcn::DescriptorSource& src, unsigned dwords, std::vector<uint32_t>& out) {
        out.assign(dwords, 0);
        uint64_t at;
        switch (src.kind) {
            case vpgcn::DescriptorSource::UserData:
                if (src.sgpr + dwords > 16) return false;
                std::copy_n(s.userData.begin() + src.sgpr, dwords, out.begin());
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
        return readGuest(at, out.data(), 4 * dwords);
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

    const Translated* shader(const Stage& s, vpgcn::ShaderStage stage) {
        const uint64_t key1 = programKey(s, stage);
        auto it1 = sourcesByProgram.find(key1);
        if (it1 == sourcesByProgram.end()) {
            vpgcn::LiftResult first = vpgcn::lift(liftInput(s, stage));
            it1 = sourcesByProgram.emplace(key1, first.errors.empty() ? first.bindings : std::vector<vpgcn::Binding>{}).first;
            if (!first.errors.empty()) {
                Translated t;
                t.error = first.errors.front();
                translated.emplace(key1, t);  // level 2 keyed by key1 alone: the program fails whatever the state
                ++r.stats_.shaderErrors;
                log("vpps4.shader error stage=" + std::to_string(int(stage)) + " key=" + hex(key1) + " \"" + t.error + "\"");
                dumpProgram(s, key1);
                return &translated[key1];
            }
        }
        if (auto bad = translated.find(key1); bad != translated.end() && !bad->second.ok) return &bad->second;

        // What the real descriptors say about the texture and typed-buffer slots.
        std::map<vpgcn::DescriptorSource, vpgcn::DescriptorInfo> infos;
        Hash h;
        h.value(key1);
        std::vector<uint32_t> words;
        for (const vpgcn::Binding& b : it1->second) {
            if (!b.dwords || infos.count(b.source)) continue;
            vpgcn::DescriptorInfo info;
            bool known = false;
            if ((b.kind == vpir::ResourceKind::Texture || b.kind == vpir::ResourceKind::StorageTexture) &&
                readDescriptor(s, b.source, b.dwords, words)) {
                const ImageDesc d = decodeImage(words);
                info.dim = texDim(d);
                info.elem = texElem(d);
                known = true;
            } else if (b.kind == vpir::ResourceKind::Buffer && readDescriptor(s, b.source, b.dwords, words)) {
                const BufferDesc d = decodeBuffer(words);
                info.dfmt = d.dataFormat;
                info.nfmt = d.numFormat;
                known = true;
            }
            if (!known) continue;
            infos[b.source] = info;
            h.value(b.slot);
            h.value(info.dim);
            h.value(info.elem);
            h.value(info.dfmt);
            h.value(info.nfmt);
        }
        const uint64_t key2 = h.h;
        auto it2 = translated.find(key2);
        if (it2 != translated.end()) return &it2->second;

        vpgcn::LiftInput in = liftInput(s, stage);
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
            vpmsl::EmitResult msl = vpmsl::emitMsl(lifted.module);
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
                if (registered.insert(t.id).second && r.hooks_.shader) {
                    const vpir::Stage st = stage == vpgcn::ShaderStage::Vertex  ? vpir::Stage::Vertex
                                         : stage == vpgcn::ShaderStage::Pixel ? vpir::Stage::Fragment
                                                                              : vpir::Stage::Compute;
                    r.hooks_.shader(t.id, st, msl.shader);
                }
                ++r.stats_.shaders;
            }
        }
        if (!t.ok) {
            ++r.stats_.shaderErrors;
            log("vpps4.shader error stage=" + std::to_string(int(stage)) + " key=" + hex(key2) + " \"" + t.error + "\"");
        }
        return &translated.emplace(key2, std::move(t)).first->second;
    }

    // ---- Textures, targets, samplers --------------------------------------------------------------

    struct Target {
        TextureId id;
        vpgpu::Format format = vpgpu::Format::Undefined;
        uint32_t width = 0, height = 0;
        uint8_t samples = 1;
        uint16_t layers = 1;
        uint64_t lastUse = 0;
    };
    std::map<uint64_t, std::vector<Target>> colorTargets, depthTargets;  // by guest address
    std::map<uint64_t, TextureId> latestColor;

    TextureId target(std::map<uint64_t, std::vector<Target>>& cache, uint64_t address, vpgpu::Format f, uint32_t w,
                     uint32_t h, uint8_t samples, uint16_t layers, bool depth) {
        layers = std::max<uint16_t>(layers, 1);
        // One surface per address whatever slice range a draw renders to (CB address is slice 0;
        // a draw to slice 1 alone and one to slices 0-1 must meet in the same array): the first
        // with at least the layers asked for (bigger arrays are kept in front).
        for (Target& t : cache[address])
            if (t.format == f && t.width == w && t.height == h && t.samples == samples && t.layers >= layers) {
                t.lastUse = r.frame_;
                return t.id;
            }
        vpgpu::TextureDesc d;
        d.format = f;
        d.width = w;
        d.height = h;
        d.samples = samples;
        d.layers = layers;
        d.usage = vpgpu::usage::sampled | (depth ? vpgpu::usage::depthTarget : vpgpu::usage::colorTarget);
        // Compute passes may write a render target as a storage image (post-processing, eye
        // resolves): the same object must be writable, not a second texture at that address.
        if (!depth && samples == 1) d.usage |= vpgpu::usage::storage;
        d.persistent = true;  // a later frame may sample it or present it
        d.label = (depth ? "depth " : "color ") + hex(address);
        Target t{r.device_.createTexture(d), f, w, h, samples, layers, r.frame_};
        if (!t.id) return {};
        ++r.stats_.targets;
        auto& list = cache[address];
        auto at = list.begin();
        while (at != list.end() && at->layers > layers) ++at;
        list.insert(at, t);
        return t.id;
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

    struct Texture {
        TextureId id;
        bool dirty = true;
        uint64_t bytes = 0;  // guest bytes it was uploaded from (for invalidation)
        // CPU writes to the source are found by content, not by page protection (guest memory is
        // aliased by no-copy GPU buffers, which cannot be made over read-only pages): a few
        // sampled lines every frame it is used, the whole source every kRecheck frames.
        uint64_t sample = 0, full = 0;
        uint64_t checkedFrame = ~0ull, fullFrame = 0;
        bool storage = false;     // created writable
        bool gpuWritten = false;  // a shader wrote it: the GPU copy is the truth, never re-upload
        // Shape, to find it again through a T# of another format (same texel size).
        vpgpu::Format format = vpgpu::Format::Undefined;
        vpgpu::TextureType type = vpgpu::TextureType::Tex2D;
        uint32_t width = 0, height = 0, depth = 1;
        uint16_t layers = 1;
    };
    struct RetiredTexture {
        TextureId id;
        uint64_t after;
    };
    std::vector<RetiredTexture> retiredTextures;  // replaced by a writable twin, destroyed kFrames later
    static constexpr uint64_t kRecheck = 30;

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
    // After an upload: remember what the source held.
    void stamp(Texture& t, uint64_t address) {
        if (const uint8_t* p = guestBytes(address, t.bytes)) {
            t.sample = sampleRange(p, t.bytes);
            t.full = hashRange(p, t.bytes);
        }
        t.checkedFrame = t.fullFrame = r.frame_;
    }
    // Before a use: has the CPU changed the source?
    void check(Texture& t, uint64_t address) {
        if (t.dirty || !t.bytes || t.checkedFrame == r.frame_) return;
        t.checkedFrame = r.frame_;
        const uint8_t* p = guestBytes(address, t.bytes);
        if (!p) return;
        if (sampleRange(p, t.bytes) != t.sample) {
            t.dirty = true;
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
        if (hashRange(p, t.bytes) != t.full) t.dirty = true;
    }
    static constexpr uint64_t kFullHashBytes = 4ull << 20;
    uint64_t fullHashFrame = ~uint64_t(0), fullHashBytes = 0;
    std::map<uint64_t, std::map<uint64_t, Texture>> textures;  // address -> shape hash -> texture

    // The texture a T# names: a render target rendered at that address, or one from the cache
    // (linear textures uploaded from guest memory; tiled ones created but left unfilled for now).
    TextureId texture(const ImageDesc& d, bool storage) {
        const vpgpu::Format f = textureFormat(d.dataFormat, d.numFormat);
        if (f == vpgpu::Format::Undefined) {
            note("texture format " + std::to_string(d.dataFormat) + "/" + std::to_string(d.numFormat) + " unsupported");
            return {};
        }
        if (auto t = colorTargets.find(d.address); t != colorTargets.end())
            for (const Target& c : t->second)
                if (c.width >= d.width && c.height >= d.height) {  // (CB width is the pitch)
                    if (c.format != f) note("render target sampled with another format (no view yet: its own format)");
                    return c.id;
                }
        if (auto t = depthTargets.find(d.address); t != depthTargets.end())
            for (const Target& c : t->second)
                if (c.width >= d.width && c.height >= d.height) return c.id;
        if (d.dstSel != std::array<uint8_t, 4>{4, 5, 6, 7}) note("texture component swizzle ignored");
        if (d.type == 14 || d.type == 15) note("MSAA texture read as single-sample");

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
        td.usage = vpgpu::usage::sampled | (storage ? vpgpu::usage::storage : 0);
        td.label = "tex " + hex(d.address);

        Hash h;
        h.value(f);
        h.value(td.type);
        h.value(td.width);
        h.value(td.height);
        h.value(td.depth);
        h.value(td.layers);
        // Usage is not part of the key: a storage write and a later sample of the same T# (e.g. a
        // compute pass producing the eye pictures that the present reads) must meet in one object.
        h.value(d.tilingIndex);
        h.value(d.pitch);
        auto& atAddress = textures[d.address];
        // A shader wrote an image at this address through a T# of another format with the same
        // texel size (typically a compute pass writing UNORM storage that is then read as sRGB:
        // Metal has no sRGB storage). Guest memory never held that picture, so this T# must see
        // the GPU copy (no texture views yet: the writer's own format; the present converts).
        if (!storage) {
            auto own = atAddress.find(h.h);
            if (own == atAddress.end() || !own->second.gpuWritten)
                for (auto& [key, other] : atAddress)
                    if (key != h.h && other.gpuWritten && other.id && other.type == td.type &&
                        other.width == td.width && other.height == td.height && other.depth == td.depth &&
                        other.layers == td.layers &&
                        vpgpu::formatInfo(other.format).bytesPerBlock == vpgpu::formatInfo(f).bytesPerBlock &&
                        vpgpu::formatInfo(other.format).blockSize == vpgpu::formatInfo(f).blockSize &&
                        // same class: the shader declares float / integer / depth textures apart
                        vpgpu::formatInfo(other.format).integer() == vpgpu::formatInfo(f).integer() &&
                        vpgpu::formatInfo(other.format).depth() == vpgpu::formatInfo(f).depth()) {
                        note("GPU-written image sampled with another format of the same size (its own format)");
                        return other.id;
                    }
        }
        Texture& tex = atAddress[h.h];
        if (tex.id && storage && !tex.storage) {
            // First write to a texture so far only sampled: recreate it writable (the old object
            // stays alive, earlier passes of this frame may still use it).
            retiredTextures.push_back({tex.id, r.frame_ + kFrames});
            tex = Texture{};
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
            ++r.stats_.textures;
        }
        // Once a shader wrote it, the GPU copy is the truth: guest memory never holds what it
        // wrote, so neither a content check nor an upload may replace it. The first write still
        // starts from the guest contents (uploaded below while dirty).
        struct MarkWritten {
            Texture& t;
            bool on;
            ~MarkWritten() {
                if (on) t.gpuWritten = true;
            }
        } markWritten{tex, storage};
        if (tex.gpuWritten) return tex.id;
        check(tex, d.address);
        if (!tex.dirty) return tex.id;
        const vpgpu::FormatInfo& fi = vpgpu::formatInfo(f);
        if (!linearTiling(d.tilingIndex)) {
            uploadTiled(d, td, fi, tex);
            return tex.id;
        }
        // Linear: mip 0 of every layer, rows `pitch` texels apart.
        const uint32_t blocksWide = (d.pitch + fi.blockSize - 1) / fi.blockSize;
        const uint32_t blocksHigh = (td.height + fi.blockSize - 1) / fi.blockSize;
        const uint64_t rowBytes = uint64_t(blocksWide) * fi.bytesPerBlock;
        const uint64_t imageBytes = rowBytes * blocksHigh;
        const uint32_t slices = td.type == vpgpu::TextureType::Tex3D ? td.depth : td.layers;
        const uint64_t total = imageBytes * slices;
        BufferId buffer;
        uint64_t offset = 0, available = 0;
        if (!guestBuffer(d.address, total, buffer, offset, available) || total > available || offset % fi.bytesPerBlock) {
            note("linear texture outside guest memory or misaligned: left unfilled");
            tex.dirty = false;
            return tex.id;
        }
        closePass();
        vpgpu::Copy c;
        c.kind = vpgpu::CopyKind::BufferToTexture;
        c.srcBuffer = {buffer, offset, total, uint32_t(rowBytes), blocksHigh};
        c.dstTexture.texture = tex.id;
        c.dstTexture.layerCount = uint16_t(td.type == vpgpu::TextureType::Tex3D ? 1 : slices);
        c.dstTexture.width = td.width;
        c.dstTexture.height = td.height;
        c.dstTexture.depth = td.type == vpgpu::TextureType::Tex3D ? td.depth : 1;
        c.label = "upload " + hex(d.address);
        frame.copy(c);
        ++r.stats_.copies;
        tex.dirty = false;
        tex.bytes = total;
        stamp(tex, d.address);
        if (r.hooks_.uploaded) r.hooks_.uploaded(d.address, total);
        return tex.id;
    }

    // Tiled: mip 0 of every layer detiled on the CPU into a staging buffer, then copied in.
    void uploadTiled(const ImageDesc& d, const vpgpu::TextureDesc& td, const vpgpu::FormatInfo& fi, Texture& tex) {
        tex.dirty = false;
        TiledSurface s;
        s.tileIndex = d.tilingIndex;
        s.bytesPerElement = fi.bytesPerBlock;
        s.width = (td.width + fi.blockSize - 1) / fi.blockSize;
        s.height = (td.height + fi.blockSize - 1) / fi.blockSize;
        s.pitch = (std::max(d.pitch, td.width) + fi.blockSize - 1) / fi.blockSize;
        s.slices = td.type == vpgpu::TextureType::Tex3D ? td.depth : td.layers;
        std::string why;
        const uint64_t bytes = tiledBytes(s, why);
        if (!bytes) {
            note("tiled texture not detiled: " + why);
            return;
        }
        GuestRegion g;
        if (!region(d.address, bytes, g)) {
            note("tiled texture outside guest memory: left unfilled");
            return;
        }
        const uint64_t rowBytes = uint64_t(s.width) * s.bytesPerElement;
        const uint64_t linearBytes = rowBytes * s.height * s.slices;
        BufferId buffer;
        uint64_t offset = 0;
        uint8_t* dst = staging(linearBytes, buffer, offset);
        if (!dst) {
            note("no staging memory for a tiled texture: left unfilled");
            return;
        }
        if (!detile(s, static_cast<const uint8_t*>(g.host) + (d.address - g.base), dst, why)) {
            note("tiled texture not detiled: " + why);
            return;
        }
        closePass();
        vpgpu::Copy c;
        c.kind = vpgpu::CopyKind::BufferToTexture;
        c.srcBuffer = {buffer, offset, linearBytes, uint32_t(rowBytes), s.height};
        c.dstTexture.texture = tex.id;
        c.dstTexture.layerCount = uint16_t(td.type == vpgpu::TextureType::Tex3D ? 1 : s.slices);
        c.dstTexture.width = td.width;
        c.dstTexture.height = td.height;
        c.dstTexture.depth = td.type == vpgpu::TextureType::Tex3D ? td.depth : 1;
        c.label = "detile " + hex(d.address);
        frame.copy(c);
        ++r.stats_.copies;
        ++r.stats_.detiled;
        tex.bytes = bytes;
        stamp(tex, d.address);
        if (r.hooks_.uploaded) r.hooks_.uploaded(d.address, bytes);
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

    // ---- Bindings ---------------------------------------------------------------------------------

    // The accesses of one translated stage, with the draw's real resources. False (with `why`) if a
    // resource cannot be resolved: the draw is skipped.
    bool bind(const Stage& s, const Translated& t, vpgpu::StageMask mask, std::vector<Access>& out, std::string& why) {
        std::vector<uint32_t> words;
        for (const vpgcn::Binding& b : t.bindings) {
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
                    if (b.source.kind == vpgcn::DescriptorSource::Gds) {
                        if (!gdsBuffer()) return why = "no GDS buffer", false;
                        a.range = ResourceRange::bytes(gds.id, 0, kGdsBytes);
                        break;
                    }
                    uint64_t address = 0, size = 0;
                    if (b.dwords == 0 || b.source.kind == vpgcn::DescriptorSource::Code) {
                        // (Code: the base is an address in the program image; its size/format words are
                        // set by the program, so there is no descriptor to read.)
                        if (!memoryAddress(s, b.source, address)) return why = "buffer address unreadable", false;
                    } else {
                        if (!readDescriptor(s, b.source, b.dwords, words)) return why = "V# unreadable", false;
                        const BufferDesc d = decodeBuffer(words);
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
                    if (!size || size > available) size = available;
                    if (res.kind == vpir::ResourceKind::Buffer && address % 4) return why = "buffer not 4-byte aligned", false;
                    if (res.kind == vpir::ResourceKind::ConstantBuffer && address % 16) {
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
                case vpir::ResourceKind::Texture:
                case vpir::ResourceKind::StorageTexture: {
                    if (!readDescriptor(s, b.source, b.dwords, words)) return why = "T# unreadable", false;
                    const TextureId id = texture(decodeImage(words), res.kind == vpir::ResourceKind::StorageTexture);
                    if (!id) return why = "texture unavailable", false;
                    a.range = ResourceRange::wholeTexture(id);
                    break;
                }
                case vpir::ResourceKind::Sampler: {
                    if (!readDescriptor(s, b.source, b.dwords, words)) return why = "S# unreadable", false;
                    a.range = ResourceRange::sampler(sampler(words));
                    break;
                }
            }
            out.push_back(a);
        }
        if (t.userDataSlot) {
            auto at = upload(s.userData.data(), sizeof s.userData);
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

    void closePass() {
        if (passOpen) frame.endRenderPass();
        passOpen = false;
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
        ++r.stats_.draws;
        if (d.count == 0 || d.instances == 0) return skip("empty draw");
        if (d.raster.cullFront && d.raster.cullBack) return skip("both faces culled");

        // Targets.
        vpgpu::RenderPassDesc desc;
        vpgpu::PipelineState ps;
        std::vector<std::pair<uint64_t, TextureId>> drawnColors;
        uint32_t width = 0, height = 0;
        // One sample count for the whole pass: the smallest of the bound targets and the host limit.
        uint8_t samples = r.maxSamples_;
        uint8_t guestSamples = 1;
        for (unsigned i = 0; i < 8; ++i)
            if (d.color[i].format && d.color[i].writeMask) {
                samples = std::min(samples, std::max<uint8_t>(d.color[i].samples, 1));
                guestSamples = std::max(guestSamples, d.color[i].samples);
            }
        if (depthFormat(d.depth.zFormat, d.depth.stencilFormat) != vpgpu::Format::Undefined) {
            samples = std::min(samples, std::max<uint8_t>(d.depth.samples, 1));
            guestSamples = std::max(guestSamples, d.depth.samples);
        }
        while (samples & (samples - 1)) samples &= uint8_t(samples - 1);  // power of two
        if (samples < guestSamples) note("MSAA reduced to " + std::to_string(samples) + " samples on the host");
        for (unsigned i = 0; i < 8; ++i) {
            const ColorTarget& c = d.color[i];
            if (!c.format || !c.writeMask) continue;
            const vpgpu::Format f = colorFormat(c.format, c.numberType, c.compSwap);
            if (f == vpgpu::Format::Undefined) {
                note("colour target format " + std::to_string(c.format) + "/" + std::to_string(c.numberType) +
                     " unsupported: slot left out");
                continue;
            }
            if (!linearTiling(c.tileMode)) note("render targets kept in GPU memory (guest copy not written)");
            const TextureId id = target(colorTargets, c.address, f, c.width, c.height, samples,
                                        uint16_t(c.sliceStart + c.sliceCount), false);
            if (!id) return skip("colour target creation failed");
            drawnColors.push_back({c.address, id});
            desc.colors[i].texture = id;
            desc.colors[i].layer = c.sliceStart;
            ps.colorFormats[i] = f;
            const Blend& b = d.blend[i];
            vpgpu::BlendState& bs = ps.blend[i];
            bs.writeMask = c.writeMask;
            bs.enabled = b.enable && !vpgpu::formatInfo(f).integer();
            if (bs.enabled) {
                bool exact = true;
                bs.srcColor = blendFactor(b.colorSrc, exact);
                bs.dstColor = blendFactor(b.colorDst, exact);
                bs.colorOp = blendOp(b.colorFunc);
                bs.srcAlpha = b.separateAlpha ? blendFactor(b.alphaSrc, exact) : bs.srcColor;
                bs.dstAlpha = b.separateAlpha ? blendFactor(b.alphaDst, exact) : bs.dstColor;
                bs.alphaOp = b.separateAlpha ? blendOp(b.alphaFunc) : bs.colorOp;
                if (!exact) note("blend factor approximated (saturate / dual source)");
            }
            width = width ? std::min(width, c.width) : c.width;
            height = height ? std::min(height, c.height) : c.height;
        }
        bool clears = false;
        const DepthTarget& z = d.depth;
        const vpgpu::Format zf = depthFormat(z.zFormat, z.stencilFormat);
        if (zf != vpgpu::Format::Undefined) {
            const TextureId id = target(depthTargets, z.zFormat ? z.zAddress : z.stencilAddress, zf, z.width, z.height,
                                        samples, uint16_t(z.sliceStart + z.sliceCount), true);
            if (!id) return skip("depth target creation failed");
            desc.depth.texture = id;
            desc.depth.layer = z.sliceStart;
            if (z.depthClear || z.stencilClear) {
                desc.depth.load = vpgpu::LoadAction::Clear;
                desc.depth.clear.depth = z.clearDepth;
                desc.depth.clear.stencil = z.clearStencil;
                clears = true;
                if (z.depthClear != z.stencilClear && z.stencilFormat && z.zFormat)
                    note("depth or stencil clear alone clears both");
            }
            ps.depthFormat = zf;
            width = width ? std::min(width, z.width) : z.width;
            height = height ? std::min(height, z.height) : z.height;
            const DepthStencil& ds = d.depthStencil;
            ps.depthTest = ds.depthTest || ds.depthWrite;
            ps.depthWrite = ds.depthWrite && z.zFormat;
            ps.depthCompare = ds.depthTest ? compareOp(ds.depthFunc) : vpgpu::CompareOp::Always;
            ps.stencilTest = ds.stencilTest && z.stencilFormat;
            if (ps.stencilTest) {
                bool exact = true;
                auto face = [&](const DepthStencil::Face& f) {
                    vpgpu::StencilFace o;
                    o.compare = compareOp(f.func);
                    o.fail = stencilOp(f.fail, exact);
                    o.pass = stencilOp(f.zPass, exact);
                    o.depthFail = stencilOp(f.zFail, exact);
                    o.readMask = f.readMask;
                    o.writeMask = f.writeMask;
                    return o;
                };
                ps.stencilFront = face(ds.front);
                ps.stencilBack = ds.backfaceEnable ? face(ds.back) : ps.stencilFront;
                if (!exact) note("stencil op approximated");
                if (ds.backfaceEnable && ds.back.ref != ds.front.ref) note("back-face stencil reference uses the front one");
            }
        }
        if (!width || !height) return skip("no render targets");
        if (d.primitive == PrimitiveType::RectList) {
            // Mostly clears: a depth/stencil clear becomes a clearing pass (the whole attachment).
            if (!clears) return skip("RectList");
            note("RectList depth/stencil clear done as a whole-target clear");
            closePass();
            desc.colors = {};
            frame.clear(desc);
            ++r.stats_.passes;
            return;
        }
        ps.samples = samples;

        // Shaders.
        if (d.vs.code.empty()) return skip("no vertex shader");
        const Translated* vs = shader(d.vs, vpgcn::ShaderStage::Vertex);
        if (!vs->ok) return skip("vertex shader not translated");
        const Translated* fs = nullptr;
        if (!d.ps.code.empty()) {
            fs = shader(d.ps, vpgcn::ShaderStage::Pixel);
            if (!fs->ok) return skip("pixel shader not translated");
        }
        ps.vertexShader = vs->id;
        ps.fragmentShader = fs ? fs->id : 0;

        // Bindings (texture uploads close the pass: done before it is opened).
        vpgpu::Draw draw;
        std::string why;
        if (!bind(d.vs, *vs, vpgpu::stage::vertex, draw.bindings, why)) return skip("vs: " + why);
        if (fs && !bind(d.ps, *fs, vpgpu::stage::fragment, draw.bindings, why)) return skip("ps: " + why);

        // Geometry.
        ps.primitive = toPrimitive(d.primitive);
        draw.primitive = ps.primitive;
        draw.instances = d.instances;
        if (d.indirectAddress) {
            if (needsConversion(d.primitive)) return skip("indirect draw of fan/quad/loop primitives");
            BufferId id;
            uint64_t offset = 0, available = 0;
            const uint64_t argBytes = d.indexAddress ? 20 : 16;
            if (!guestBuffer(d.indirectAddress, argBytes, id, offset, available) || argBytes > available || offset % 4)
                return skip("indirect arguments outside guest memory");
            draw.indirect = vpgpu::Indirect{id, offset};
            if (d.indexAddress) {
                BufferId ib;
                uint64_t ioff = 0, iavail = 0;
                if (!guestBuffer(d.indexAddress, uint64_t(std::max(d.count, 1u)) * (d.index32 ? 4 : 2), ib, ioff, iavail)) return skip("index buffer outside guest memory");
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
            note("fan/quad/loop primitives converted to lists");
        } else if (d.indexAddress) {
            BufferId id;
            uint64_t offset = 0, available = 0;
            const uint64_t bytes = uint64_t(d.count) * (d.index32 ? 4 : 2);
            if (!guestBuffer(d.indexAddress, bytes, id, offset, available) || bytes > available)
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
        draw.frontFace = d.raster.frontFaceClockwise ? vpgpu::Winding::Clockwise : vpgpu::Winding::CounterClockwise;
        draw.stencilRef = d.depthStencil.front.ref;
        draw.blendColor = d.blendColor;
        const Viewport& v = d.viewport;
        if (v.xScale != 0 && v.yScale != 0) {
            // GCN: screen = ndc * scale + offset (y down when yScale < 0). Metal: y down from the top.
            if (v.yScale > 0) {
                // Metal cannot flip y in the viewport: the image comes out upside down; at least keep
                // the facing right (mirroring swaps the winding).
                note("viewport with y up (positive y scale) drawn flipped");
                draw.frontFace = draw.frontFace == vpgpu::Winding::Clockwise ? vpgpu::Winding::CounterClockwise
                                                                             : vpgpu::Winding::Clockwise;
            }
            const float h = 2 * std::abs(v.yScale);
            draw.viewport = {v.xOffset - std::abs(v.xScale), v.yOffset - std::abs(v.yScale), 2 * std::abs(v.xScale), h,
                             std::clamp(v.zOffset, 0.0f, 1.0f), std::clamp(v.zOffset + v.zScale, 0.0f, 1.0f)};
        }
        const Scissor& sc = d.scissor;
        if (sc.right > sc.left && sc.bottom > sc.top) {
            const int32_t l = std::max(0, sc.left), t = std::max(0, sc.top);
            const int32_t rr = std::min<int32_t>(sc.right, int32_t(width)), b = std::min<int32_t>(sc.bottom, int32_t(height));
            if (rr <= l || b <= t) return skip("empty scissor");
            draw.scissor = {uint32_t(l), uint32_t(t), uint32_t(rr - l), uint32_t(b - t)};
        }
        ps.label = "vs " + hex(vs->id) + (fs ? " ps " + hex(fs->id) : "");

        const vpgpu::PipelineCache::Status st = r.device_.pipelines().request(ps, vpgpu::Priority::Visible);
        if (st == vpgpu::PipelineCache::Status::Failed) return skip("pipeline failed");
        if (st != vpgpu::PipelineCache::Status::Ready) return skip("pipeline compiling");
        draw.pipeline = ps.key();

        openPass(desc, clears);
        if (!frame.draw(std::move(draw))) return skip("no open pass");
        ++r.stats_.drawn;
        for (auto& [address, id] : drawnColors) latestColor[address] = id;
    }

    void dispatch(const DispatchState& d) {
        ++r.stats_.dispatches;
        if (d.cs.code.empty()) return skip("no compute shader");
        if (!d.indirectAddress && (!d.groups[0] || !d.groups[1] || !d.groups[2])) return skip("empty dispatch");
        const Translated* cs = shader(d.cs, vpgcn::ShaderStage::Compute);
        if (!cs->ok) return skip("compute shader not translated");
        vpgpu::Dispatch x;
        std::string why;
        if (!bind(d.cs, *cs, vpgpu::stage::compute, x.bindings, why)) return skip("cs: " + why);
        vpgpu::PipelineState ps;
        ps.computeShader = cs->id;
        ps.label = "cs " + hex(cs->id);
        const vpgpu::PipelineCache::Status st = r.device_.pipelines().request(ps, vpgpu::Priority::Visible);
        if (st == vpgpu::PipelineCache::Status::Failed) return skip("pipeline failed");
        if (st != vpgpu::PipelineCache::Status::Ready) return skip("pipeline compiling");
        x.pipeline = ps.key();
        x.groups = d.groups;
        x.label = ps.label;
        if (d.indirectAddress) {
            BufferId id;
            uint64_t offset = 0, available = 0;
            if (!guestBuffer(d.indirectAddress, 12, id, offset, available) || available < 12 || offset % 4)
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
                           " s" + std::to_string(t.samples) + " used" + std::to_string(r.frame_ - t.lastUse);
                }
        };
        targets(colorTargets, "colour");
        targets(depthTargets, "depth");
        if (auto it = textures.find(address); it != textures.end())
            for (const auto& [key, t] : it->second)
                out += std::string(" tex ") + (t.format == vpgpu::Format::Undefined ? "?" : vpgpu::formatInfo(t.format).name) +
                       " " + std::to_string(t.width) + "x" + std::to_string(t.height) + (t.storage ? " storage" : "") +
                       (t.gpuWritten ? " gpu_written" : "");
        return out.empty() ? " nothing (only guest memory)" : out;
    }

    void resolve(const ColorTarget& src, const ColorTarget& dst) {
        ++r.stats_.resolves;
        const vpgpu::Format f = colorFormat(src.format, src.numberType, src.compSwap);
        const vpgpu::Format fd = colorFormat(dst.format, dst.numberType, dst.compSwap);
        if (f == vpgpu::Format::Undefined || fd == vpgpu::Format::Undefined) return skip("resolve: format unsupported");
        // A Metal blit copies between identical pixel formats only.
        if (f != fd) return skip("resolve: source and destination formats differ");
        // The source exists at the count its draws used (same rule as draw()).
        uint8_t samples = std::min<uint8_t>(r.maxSamples_, std::max<uint8_t>(src.samples, 1));
        while (samples & (samples - 1)) samples &= uint8_t(samples - 1);
        if (samples > 1) return skip("resolve: multisampled source (host MSAA on)");
        const uint16_t layers = uint16_t(src.sliceStart + src.sliceCount);
        const TextureId from = target(colorTargets, src.address, f, src.width, src.height, 1, layers, false);
        const TextureId to = target(colorTargets, dst.address, fd, dst.width, dst.height, 1,
                                    uint16_t(dst.sliceStart + dst.sliceCount), false);
        if (!from || !to) return skip("resolve: target creation failed");
        closePass();
        vpgpu::Copy c;
        c.kind = vpgpu::CopyKind::TextureToTexture;
        c.srcTexture.texture = from;
        c.srcTexture.layer = src.sliceStart;
        c.dstTexture.texture = to;
        c.dstTexture.layer = dst.sliceStart;
        c.srcTexture.layerCount = c.dstTexture.layerCount =
            uint16_t(std::max<uint16_t>(std::min(src.sliceCount, dst.sliceCount), 1));
        c.srcTexture.width = c.dstTexture.width = std::min(src.width, dst.width);
        c.srcTexture.height = c.dstTexture.height = std::min(src.height, dst.height);
        c.label = "resolve " + hex(src.address) + " -> " + hex(dst.address);
        frame.copy(c);
        ++r.stats_.copies;
        latestColor[dst.address] = to;
    }

    bool fill(uint64_t address, uint64_t bytes, uint32_t value) {
        BufferId id;
        uint64_t offset = 0, available = 0;
        if (!bytes) return true;
        if (!guestBuffer(address, bytes, id, offset, available) || bytes > available) {
            note("DMA fill outside guest memory: dropped");
            return false;
        }
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
        invalidate(address, bytes);
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
        closePass();
        vpgpu::Copy c;
        c.kind = vpgpu::CopyKind::BufferToBuffer;
        c.srcBuffer = {si, soff, bytes, 0, 0};
        c.dstBuffer = {di, doff, bytes, 0, 0};
        c.label = "dma " + hex(src) + " -> " + hex(dst);
        frame.copy(c);
        ++r.stats_.copies;
        invalidate(dst, bytes);
        return true;
    }

    void invalidate(uint64_t address, uint64_t size) {
        // Textures starting up to 1 GiB below the range may reach into it.
        auto it = textures.lower_bound(address > (1ull << 30) ? address - (1ull << 30) : 0);
        for (; it != textures.end() && it->first < address + size; ++it)
            for (auto& [shape, t] : it->second)
                if (it->first + std::max<uint64_t>(t.bytes, 1) > address) t.dirty = true;
    }

    vpgpu::FrameReport submit() {
        closePass();
        vpgpu::FrameReport report = r.device_.submit(frame);
        frame = vpgpu::Frame{};
        ringIndex = (ringIndex + 1) % kFrames;
        rings[ringIndex].head = 0;
        std::erase_if(retired, [&](const Retired& x) {
            if (x.after > r.frame_ + 1) return false;
            r.device_.destroy(x.id);
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
        for (Ring& ring : rings)
            if (ring.memory) {
                r.device_.destroy(ring.id);
                std::free(ring.memory);
            }
    }
};

Renderer::Renderer(vpgpu::Device& device, Hooks hooks)
    : device_(device), hooks_(std::move(hooks)), impl_(std::make_unique<Impl>(*this)) {}

Renderer::~Renderer() = default;

void Renderer::setMaxSamples(uint8_t samples) {
    samples = std::clamp<uint8_t>(samples, 1, 8);
    while (samples & (samples - 1)) samples &= uint8_t(samples - 1);
    maxSamples_ = samples;
}

void Renderer::guardRunaway() {
    if (++sinceSubmit_ < kRunawayCommands) return;
    if (!runawayTold_ && hooks_.log)
        hooks_.log("gpu.err what=frame_not_ended commands=" + std::to_string(sinceSubmit_) +
                   " (the host never called submit: submitted here)");
    runawayTold_ = true;
    submit();
}

void Renderer::draw(const DrawState& d) {
    impl_->draw(d);
    guardRunaway();
}
void Renderer::dispatch(const DispatchState& d) {
    impl_->dispatch(d);
    guardRunaway();
}
void Renderer::resolve(const ColorTarget& src, const ColorTarget& dst) { impl_->resolve(src, dst); }
void Renderer::invalidate(uint64_t address, uint64_t size) { impl_->invalidate(address, size); }
bool Renderer::gpuMayAccess(uint64_t address, uint64_t size) const { return impl_->busyOverlaps(address, size); }
void Renderer::gpuIdle() {
    impl_->busy.clear();
    impl_->busyAll = false;
}

vpgpu::FrameReport Renderer::submit() {
    sinceSubmit_ = 0;
    vpgpu::FrameReport report = impl_->submit();
    ++frame_;
    if (summaryEvery_ && frame_ % summaryEvery_ == 0) {
        std::string line = "vpps4.frame n=" + std::to_string(frame_) + " draws=" + std::to_string(stats_.drawn) + "/" +
                           std::to_string(stats_.draws) + " dispatches=" + std::to_string(stats_.dispatched) + "/" +
                           std::to_string(stats_.dispatches) + " passes=" + std::to_string(stats_.passes) +
                           " shaders=" + std::to_string(stats_.shaders) + " shader_errors=" + std::to_string(stats_.shaderErrors) +
                           " textures=" + std::to_string(stats_.textures) + " detiled=" + std::to_string(stats_.detiled) + " indirect=" + std::to_string(stats_.indirect) + " targets=" + std::to_string(stats_.targets) +
                           " uploads=" + std::to_string(stats_.uploads) + " resolves=" + std::to_string(stats_.resolves);
        for (const auto& [why, n] : stats_.skipped) line += " skip[" + why + "]=" + std::to_string(n);
        if (hooks_.log) hooks_.log(line);
    }
    return report;
}

vpgpu::TextureId Renderer::textureFor(const ImageDesc& d) { return impl_->texture(d, false); }
bool Renderer::fill(uint64_t address, uint64_t bytes, uint32_t value) { return impl_->fill(address, bytes, value); }
bool Renderer::copy(uint64_t dst, uint64_t src, uint64_t bytes) { return impl_->copy(dst, src, bytes); }

uint32_t* Renderer::gds() {
    return impl_->gdsBuffer() ? static_cast<uint32_t*>(impl_->gds.memory) : nullptr;
}

std::string Renderer::describeAt(uint64_t address, uint64_t bytes) const { return impl_->describe(address, bytes); }

vpgpu::TextureId Renderer::colorTargetSlice(uint64_t address, uint32_t width, uint32_t height, uint16_t& layer) const {
    layer = 0;
    const Impl::Target* t = impl_->targetSlice(address, width, height, layer);
    return t ? t->id : vpgpu::TextureId{};
}

vpgpu::TextureId Renderer::colorTargetAt(uint64_t address) const {
    auto it = impl_->latestColor.find(address);
    return it == impl_->latestColor.end() ? vpgpu::TextureId{} : it->second;
}

}  // namespace vpps4
