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
constexpr uint64_t kRingBytes = 8ull << 20;
constexpr uint64_t kPage = 16384;

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
        return bool(id);
    }

    // ---- Upload ring ----------------------------------------------------------------------------

    struct Ring {
        void* memory = nullptr;
        BufferId id;
        uint64_t head = 0;
    };
    std::array<Ring, kFrames> rings{};
    unsigned ringIndex = 0;

    std::optional<uint64_t> upload(const void* data, uint64_t bytes, uint64_t align = 256) {
        Ring& ring = rings[ringIndex];
        if (!ring.memory) {
            ring.memory = std::aligned_alloc(kPage, kRingBytes);
            if (!ring.memory) return std::nullopt;
            ring.id = r.device_.createBuffer({kRingBytes, ring.memory, "vpps4 upload " + std::to_string(ringIndex)});
        }
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
        for (Target& t : cache[address])
            if (t.format == f && t.width == w && t.height == h && t.samples == samples && t.layers == layers) {
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
        d.persistent = true;  // a later frame may sample it or present it
        d.label = (depth ? "depth " : "color ") + hex(address);
        Target t{r.device_.createTexture(d), f, w, h, samples, layers, r.frame_};
        if (!t.id) return {};
        ++r.stats_.targets;
        cache[address].push_back(t);
        return t.id;
    }

    struct Texture {
        TextureId id;
        bool dirty = true;
        uint64_t bytes = 0;  // guest bytes it was uploaded from (for invalidation)
    };
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
        h.value(td.usage);
        h.value(d.tilingIndex);
        h.value(d.pitch);
        Texture& tex = textures[d.address][h.h];
        if (!tex.id) {
            tex.id = r.device_.createTexture(td);
            if (!tex.id) return {};
            ++r.stats_.textures;
        }
        if (!tex.dirty) return tex.id;
        if (!linearTiling(d.tilingIndex)) {
            note("tiled texture left unfilled (detiling not done yet)");
            tex.dirty = false;  // nothing to refresh until detiling exists
            return tex.id;
        }
        // Linear: mip 0 of every layer, rows `pitch` texels apart.
        const vpgpu::FormatInfo& fi = vpgpu::formatInfo(f);
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
        return tex.id;
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
                    if (!address) return why = "null buffer", false;
                    BufferId id;
                    uint64_t offset = 0, available = 0;
                    if (!guestBuffer(address, size, id, offset, available)) return why = "buffer outside guest memory", false;
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
        if (needsConversion(d.primitive)) {
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
        if (!d.groups[0] || !d.groups[1] || !d.groups[2]) return skip("empty dispatch");
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
        closePass();
        frame.dispatch(std::move(x));
        ++r.stats_.dispatched;
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
        return report;
    }

    ~Impl() {
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

void Renderer::draw(const DrawState& d) { impl_->draw(d); }
void Renderer::dispatch(const DispatchState& d) { impl_->dispatch(d); }
void Renderer::resolve(const ColorTarget& src, const ColorTarget& dst) { impl_->resolve(src, dst); }
void Renderer::invalidate(uint64_t address, uint64_t size) { impl_->invalidate(address, size); }

vpgpu::FrameReport Renderer::submit() {
    vpgpu::FrameReport report = impl_->submit();
    ++frame_;
    if (summaryEvery_ && frame_ % summaryEvery_ == 0) {
        std::string line = "vpps4.frame n=" + std::to_string(frame_) + " draws=" + std::to_string(stats_.drawn) + "/" +
                           std::to_string(stats_.draws) + " dispatches=" + std::to_string(stats_.dispatched) + "/" +
                           std::to_string(stats_.dispatches) + " passes=" + std::to_string(stats_.passes) +
                           " shaders=" + std::to_string(stats_.shaders) + " shader_errors=" + std::to_string(stats_.shaderErrors) +
                           " textures=" + std::to_string(stats_.textures) + " targets=" + std::to_string(stats_.targets) +
                           " uploads=" + std::to_string(stats_.uploads) + " resolves=" + std::to_string(stats_.resolves);
        for (const auto& [why, n] : stats_.skipped) line += " skip[" + why + "]=" + std::to_string(n);
        if (hooks_.log) hooks_.log(line);
    }
    return report;
}

vpgpu::TextureId Renderer::colorTargetAt(uint64_t address) const {
    auto it = impl_->latestColor.find(address);
    return it == impl_->latestColor.end() ? vpgpu::TextureId{} : it->second;
}

}  // namespace vpps4
