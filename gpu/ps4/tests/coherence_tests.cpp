// Memory coherence on the NullBackend: the same guest bytes written as a render target, depth
// buffer, storage image, raw buffer or by DMA, and read as any of them. Frames are captured so the
// ORDER of dispatches, draws and copies on the GPU timeline can be asserted.
// Usage: vpps4_coherence_tests TEST
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "vp4gpu/core/device.h"
#include "vpps4/renderer.h"

#ifndef VPPS4_LIFT_DIR
#error "VPPS4_LIFT_DIR must point at gpu/shader/gcn/tests/lift"
#endif

namespace {

int failures = 0;
#define CHECK(c)                                                                       \
    do {                                                                               \
        if (!(c)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #c); \
            ++failures;                                                                \
        }                                                                              \
    } while (0)

struct Program {
    std::vector<uint32_t> code, fetch;
};
Program load(const char* name) {
    std::ifstream f(std::string(VPPS4_LIFT_DIR) + "/" + name + ".lift");
    Program p;
    for (std::string line; std::getline(f, line);) {
        std::istringstream ls(line);
        std::string key;
        ls >> key;
        std::vector<uint32_t>* dst = key == "code" ? &p.code : key == "fetch" ? &p.fetch : nullptr;
        if (!dst) continue;
        for (std::string t; ls >> t;) dst->push_back(uint32_t(std::stoul(t, nullptr, 16)));
    }
    if (p.code.empty()) std::fprintf(stderr, "missing program %s\n", name), std::exit(2);
    return p;
}

// A backend that records the frames it was given (the commands, in order) next to the NullBackend's
// call log.
class Recording final : public vp4gpu::Backend {
public:
    vp4gpu::NullBackend inner;
    std::vector<vp4gpu::Command> commands;

    void createBuffer(vp4gpu::BufferId i, const vp4gpu::BufferDesc& d) override { inner.createBuffer(i, d); }
    void createTexture(vp4gpu::TextureId i, const vp4gpu::TextureDesc& d) override { inner.createTexture(i, d); }
    void createSampler(vp4gpu::SamplerId i, const vp4gpu::SamplerDesc& d) override { inner.createSampler(i, d); }
    void destroyBuffer(vp4gpu::BufferId i) override { inner.destroyBuffer(i); }
    void destroyTexture(vp4gpu::TextureId i) override { inner.destroyTexture(i); }
    void destroySampler(vp4gpu::SamplerId i) override { inner.destroySampler(i); }
    uint64_t bufferAddress(vp4gpu::BufferId i) const override { return inner.bufferAddress(i); }
    bool compilePipeline(const vp4gpu::PipelineState& s, std::string& e) override { return inner.compilePipeline(s, e); }
    void destroyPipeline(vp4gpu::PipelineKey k) override { inner.destroyPipeline(k); }
    void submit(uint64_t n, const vp4gpu::Frame& f, const vp4gpu::CompiledFrame& c) override {
        commands.insert(commands.end(), f.commands().begin(), f.commands().end());
        inner.submit(n, f, c);
    }
    bool takeTimings(uint64_t& n, vp4gpu::GpuTimings& t) override { return inner.takeTimings(n, t); }
    uint64_t completedFrames() const override { return inner.completedFrames(); }
};

constexpr uint64_t kBase = 0x10000000, kSize = 1 << 20;
constexpr uint64_t kRt = kBase + 0x40000;     // 128x128 RGBA8, 64 KiB
constexpr uint64_t kTex = kBase + 0x10000;    // 64x64 RGBA8, 16 KiB
constexpr uint64_t kImg = kBase + 0x20000;    // storage image 64x64 RGBA8
constexpr uint64_t kDepth = kBase + 0x60000;  // 128x128 D32, 64 KiB

struct Fixture {
    void* host = std::aligned_alloc(16384, kSize);
    Recording backend;
    std::vector<std::string> log;
    vp4gpu::Device device{backend, 0, [this](const std::string& s) { log.push_back(s); }};
    bool mirrored = false;
    std::vector<std::pair<uint64_t, uint64_t>> watchedWrites, ownedOn, ownedOff;
    std::vector<vpps4::ViewRequest> viewRequests;
    vpps4::Renderer renderer{device, hooks()};
    Program vs = load("vs_fetch"), ps = load("ps_textured"), store = load("cs_partial_store"), atomic = load("cs_image_atomic");

    explicit Fixture(bool mirror = false, bool views = false) : mirrored(mirror) {
        std::memset(host, 0, kSize);
        (void)views;
    }
    ~Fixture() { std::free(host); }

    bool withViews = false;
    vpps4::Hooks hooks() {
        vpps4::Hooks h;
        h.region = [this](uint64_t a, uint64_t s, vpps4::GuestRegion& g) {
            if (a < kBase || a + s > kBase + kSize) return false;
            g = {kBase, kSize, host, mirrored};
            return true;
        };
        h.shader = [](uint64_t, vp4ir::Stage, const vp4msl::MslShader&) {};
        h.log = [this](const std::string& s) { log.push_back(s); };
        h.watchWrites = [this](uint64_t a, uint64_t n) { watchedWrites.push_back({a, n}); };
        h.watchOwned = [this](uint64_t a, uint64_t n, bool on) { (on ? ownedOn : ownedOff).push_back({a, n}); };
        h.view = [this](const vpps4::ViewRequest& r) -> vp4gpu::TextureId {
            if (!withViews) return {};
            viewRequests.push_back(r);
            vp4gpu::TextureDesc d;
            d.format = r.format == vp4gpu::Format::Undefined ? vp4gpu::Format::RGBA8Unorm : r.format;
            d.width = d.height = 4;
            return device.createTexture(d);
        };
        return h;
    }
    uint32_t* at(uint64_t guest) { return reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(host) + (guest - kBase)); }

    static void tsharp(uint32_t* t, uint64_t a, uint32_t w, uint32_t h, uint32_t dfmt, uint32_t nfmt, uint32_t tile) {
        t[0] = uint32_t(a >> 8);
        t[1] = uint32_t(a >> 40) | (dfmt << 20) | (nfmt << 26);
        t[2] = (w - 1) | ((h - 1) << 14);
        t[3] = 4 | (5u << 3) | (6u << 6) | (7u << 9) | (tile << 20) | (9u << 28);
        t[4] = (w - 1) << 13;
    }

    // The textured draw of the renderer tests; the sampled T# is rewritten by sample().
    vpps4::DrawState state() {
        uint32_t* vtable = at(kBase + 0x100);
        const uint64_t vb = kBase + 0x1000;
        vtable[0] = uint32_t(vb);
        vtable[1] = uint32_t(vb >> 32) | (16u << 16);
        vtable[2] = 64;
        vtable[3] = (7u << 12) | (14u << 15);
        sample(kTex, 64, 64, 10, 0, 8);
        at(kBase + 0x200)[8] = 0;
        vpps4::DrawState d;
        d.color[0] = {kRt, 128, 128, 128, 10, 0, 0, 8};
        d.viewport = {64, 64, -64, 64, 1, 0};
        d.count = 3;
        d.vs.code = vs.code;
        d.vs.fetch = vs.fetch;
        d.vs.address = kBase + 0x80000;
        d.vs.userSgprs = 6;
        d.vs.userData[4] = uint32_t(kBase + 0x100);
        d.vs.vgprCompCnt = 3;
        d.ps.code = ps.code;
        d.ps.address = kBase + 0x81000;
        d.ps.userSgprs = 2;
        d.ps.userData[0] = uint32_t(kBase + 0x200);
        d.ps.psInputAddr = 2;
        d.ps.exportFormat[0] = 4;
        return d;
    }
    void sample(uint64_t addr, uint32_t w, uint32_t h, uint32_t dfmt, uint32_t nfmt, uint32_t tile) {
        tsharp(at(kBase + 0x200), addr, w, h, dfmt, nfmt, tile);
    }
    // A draw into a depth buffer (D32, linear) instead of the colour target's neighbours.
    vpps4::DrawState depthState(bool write) {
        vpps4::DrawState d = state();
        d.depth = {kDepth, 0, 128, 128, 3, 0, 8, 0, 1, 1, false, false, 1.0f, 0};
        d.depthStencil.depthTest = true;
        d.depthStencil.depthWrite = write;
        d.depthStencil.depthFunc = 1;
        return d;
    }
    // cs_partial_store over [addr, addr + bytes): a raw buffer write.
    void rawWrite(uint64_t addr, uint32_t bytes) {
        vpps4::DispatchState x;
        x.cs.code = store.code;
        x.cs.fetch = store.fetch;
        x.cs.address = kBase + 0x82000;
        x.cs.userSgprs = 4;
        x.cs.userData[0] = uint32_t(addr);
        x.cs.userData[1] = uint32_t(addr >> 32) | (4u << 16);
        x.cs.userData[2] = bytes / 4;
        x.cs.userData[3] = (4u << 12) | (4u << 15);
        renderer.dispatch(x);
        renderer.dispatch(x);  // (the pipeline compiles on the first one on the null backend)
    }
    // cs_image_atomic: writes the storage images at kImg and kImg + 0x10000.
    void imageWrite() {
        std::array<uint32_t, 8> a{}, b{};
        tsharp(a.data(), kImg, 64, 64, 10, 0, 8);
        tsharp(b.data(), kImg + 0x10000, 64, 64, 10, 0, 8);
        vpps4::DispatchState x;
        x.cs.code = atomic.code;
        x.cs.fetch = atomic.fetch;
        x.cs.address = kBase + 0x83000;
        x.cs.userSgprs = 16;
        for (int i = 0; i < 8; ++i) x.cs.userData[i] = a[i], x.cs.userData[8 + i] = b[i];
        renderer.dispatch(x);
        renderer.dispatch(x);
    }

    // The captured frames as one event list: "draw", "dispatch", "copy <label>" in timeline order.
    std::vector<std::string> events() {
        renderer.submit();
        std::vector<std::string> out;
        for (const vp4gpu::Command& c : backend.commands) {
            if (const auto* p = std::get_if<vp4gpu::RenderPass>(&c)) {
                for (size_t i = 0; i < p->draws.size(); ++i) out.push_back("draw");
            } else if (std::holds_alternative<vp4gpu::Dispatch>(c)) {
                out.push_back("dispatch");
            } else if (const auto* cp = std::get_if<vp4gpu::Copy>(&c)) {
                out.push_back("copy " + cp->label);
            }
        }
        return out;
    }
    static std::string hex(uint64_t v) {
        char b[24];
        std::snprintf(b, sizeof b, "0x%llx", static_cast<unsigned long long>(v));
        return b;
    }
    static long indexOf(const std::vector<std::string>& e, const std::string& s, size_t from = 0) {
        for (size_t i = from; i < e.size(); ++i)
            if (e[i] == s) return long(i);
        return -1;
    }
    static size_t count(const std::vector<std::string>& e, const std::string& prefix) {
        size_t n = 0;
        for (const std::string& s : e) n += s.rfind(prefix, 0) == 0;
        return n;
    }
    void dump() {
        for (const std::string& l : log) std::fprintf(stderr, "  %s\n", l.c_str());
    }
};

// A T# naming the render target: its own image, nothing copied. A smaller one too. Another
// format of the same texel size without views falls back to it (noted); a texel size that
// differs goes through memory: the target is written back, then the texture filled from it.
void rtToTexture() {
    Fixture f;
    f.renderer.draw(f.state());
    const uint64_t copies = f.renderer.stats().copies, textures = f.renderer.stats().textures;
    const auto rt = f.renderer.colorTargetAt(kRt);
    CHECK(bool(rt));
    CHECK(f.renderer.tracker().ownerImage(kRt, 128 * 128 * 4) != 0);  // memory behind the target is stale
    f.sample(kRt, 128, 128, 10, 0, 8);
    CHECK(f.renderer.textureFor(vpps4::decodeImage(std::span<const uint32_t>(f.at(kBase + 0x200), 8))) == rt);
    f.sample(kRt, 64, 64, 10, 0, 8);
    CHECK(f.renderer.textureFor(vpps4::decodeImage(std::span<const uint32_t>(f.at(kBase + 0x200), 8))) == rt);
    CHECK(f.renderer.stats().copies == copies && f.renderer.stats().textures == textures);
    f.sample(kRt, 128, 128, 5, 0, 8);  // 16_16 unorm: same texel size, no views in this host
    CHECK(f.renderer.textureFor(vpps4::decodeImage(std::span<const uint32_t>(f.at(kBase + 0x200), 8))) == rt);
    CHECK(f.renderer.stats().notes.count("image sampled with another format (no view available: its own format)") == 1);
    f.sample(kRt, 128, 128, 3, 0, 8);  // 8_8: another texel size
    const auto other = f.renderer.textureFor(vpps4::decodeImage(std::span<const uint32_t>(f.at(kBase + 0x200), 8)));
    CHECK(bool(other) && other != rt);
    CHECK(f.renderer.stats().retiled == 1);
    const auto e = f.events();
    const long retile = Fixture::indexOf(e, "copy retile " + Fixture::hex(kRt));
    const long upload = Fixture::indexOf(e, "copy upload " + Fixture::hex(kRt));
    CHECK(retile >= 0 && upload > retile);  // written back before the new texture reads it
    CHECK(f.renderer.tracker().ownerImage(kRt, 128 * 128 * 4) == 0);
    if (failures) f.dump();
}

// A render target written in BGRA order and read through an RGBA T# (or the reverse) is one image
// as before: no view, no note (the T#'s own swizzle restores the order).
void channelOrder() {
    Fixture f;
    f.withViews = true;
    vpps4::DrawState d = f.state();
    d.color[0].compSwap = 1;  // alt: BGRA
    f.renderer.draw(d);
    const auto rt = f.renderer.colorTargetAt(kRt);
    f.sample(kRt, 128, 128, 10, 0, 8);
    CHECK(f.renderer.textureFor(vpps4::decodeImage(std::span<const uint32_t>(f.at(kBase + 0x200), 8))) == rt);
    f.sample(kRt, 128, 128, 10, 9, 8);  // sRGB: a real format change, in the target's channel order
    CHECK(f.renderer.textureFor(vpps4::decodeImage(std::span<const uint32_t>(f.at(kBase + 0x200), 8))) != rt);
    CHECK(f.viewRequests.size() == 1 && f.viewRequests[0].format == vp4gpu::Format::BGRA8Srgb);
    CHECK(f.renderer.stats().notes.count("image sampled with another format (no view available: its own format)") == 0);
}

// A raw buffer write over a texture's source: the next use fills it again, after the write on the
// timeline; a write elsewhere does not.
void rawWriteToTexture() {
    Fixture f;
    f.renderer.draw(f.state());
    f.renderer.draw(f.state());
    CHECK(f.renderer.stats().refilled == 1);
    f.rawWrite(kTex + 0x1000, 256);
    f.renderer.draw(f.state());
    CHECK(f.renderer.stats().refilled == 2);
    f.rawWrite(kBase + 0x30000, 256);  // elsewhere
    f.renderer.draw(f.state());
    CHECK(f.renderer.stats().refilled == 2);
    const auto e = f.events();
    const long firstUpload = Fixture::indexOf(e, "copy upload " + Fixture::hex(kTex));
    const long write = Fixture::indexOf(e, "dispatch", size_t(firstUpload) + 1);
    const long second = Fixture::indexOf(e, "copy upload " + Fixture::hex(kTex), size_t(firstUpload) + 1);
    CHECK(firstUpload >= 0 && write > firstUpload && second > write);
    if (failures) f.dump();
}

// A raw write over one layer of an array texture refills the texture (every layer is part of its
// range); a write just past the last layer does not.
void rawWriteToLayer() {
    Fixture f;
    vpps4::DrawState d = f.state();
    uint32_t* t = f.at(kBase + 0x200);
    Fixture::tsharp(t, kTex, 64, 64, 10, 0, 8);
    t[3] = (t[3] & ~(15u << 28)) | (13u << 28);  // 2D array
    t[5] = 1u << 13;                              // layers 0..1: 32 KiB
    f.renderer.draw(d);
    CHECK(f.renderer.stats().refilled == 1);
    f.rawWrite(kTex + 0x4100, 256);  // inside layer 1
    f.renderer.draw(d);
    CHECK(f.renderer.stats().refilled == 2);
    f.rawWrite(kTex + 0x8000, 256);  // just past the array
    f.renderer.draw(d);
    CHECK(f.renderer.stats().refilled == 2);
    if (failures) f.dump();
}

// Storage image written by a dispatch, then a raw write over part of it: the image is written back
// first (so the raw write lands on its bytes), and its claim on memory ends. A DMA that covers it
// completely does not bother.
void textureToRaw() {
    Fixture f;
    f.imageWrite();
    CHECK(f.renderer.tracker().ownerImage(kImg, 64 * 64 * 4) != 0);
    f.rawWrite(kImg + 0x100, 256);
    CHECK(f.renderer.stats().retiled == 1);
    CHECK(f.renderer.tracker().ownerImage(kImg, 64 * 64 * 4) == 0);
    f.imageWrite();
    CHECK(f.renderer.tracker().ownerImage(kImg, 64 * 64 * 4) != 0);
    CHECK(f.renderer.fill(kImg, 64 * 64 * 4, 0));  // covers the whole image
    CHECK(f.renderer.stats().retiled == 1);
    CHECK(f.renderer.tracker().ownerImage(kImg, 64 * 64 * 4) == 0);
    f.imageWrite();
    CHECK(f.renderer.copy(kBase + 0x5000, kImg + 0x40, 128));  // a DMA reads the image's bytes
    CHECK(f.renderer.stats().retiled == 2);
    const auto e = f.events();
    const long retile = Fixture::indexOf(e, "copy retile " + Fixture::hex(kImg));
    CHECK(retile > 0 && e[size_t(retile) - 1] == "dispatch");
    CHECK(Fixture::count(e, "copy retile") == 2);
    if (failures) f.dump();
}

// DMA writes to a texture's source: refilled on the next use, after the DMA.
void dmaToTexture() {
    Fixture f;
    f.renderer.draw(f.state());
    CHECK(f.renderer.stats().refilled == 1);
    CHECK(f.renderer.copy(kTex + 0x100, kBase + 0x5000, 64));
    f.renderer.draw(f.state());
    CHECK(f.renderer.stats().refilled == 2);
    CHECK(f.renderer.fill(kTex, 16, 0x01020304u));
    f.renderer.draw(f.state());
    CHECK(f.renderer.stats().refilled == 3);
    CHECK(f.renderer.fill(kTex + 0x8000, 16, 0));  // past the 16 KiB texture
    f.renderer.draw(f.state());
    CHECK(f.renderer.stats().refilled == 3);
    const auto e = f.events();
    const long dma = Fixture::indexOf(e, "copy dma " + Fixture::hex(kBase + 0x5000) + " -> " + Fixture::hex(kTex + 0x100));
    const long up = Fixture::indexOf(e, "copy upload " + Fixture::hex(kTex), size_t(Fixture::indexOf(e, "copy upload " + Fixture::hex(kTex))) + 1);
    CHECK(dma >= 0 && up > dma);
}

// The CPU's writes: invalidate() and (without a host watch) the content check both end up as
// versions, so every image over the range is refilled once.
void cpuWrite() {
    Fixture f;
    f.renderer.draw(f.state());
    f.renderer.invalidate(kTex + 8, 4);
    f.renderer.draw(f.state());
    f.renderer.draw(f.state());
    CHECK(f.renderer.stats().refilled == 2);
    f.renderer.submit();
    *f.at(kTex) ^= 0xff;  // found by content
    f.renderer.draw(f.state());
    CHECK(f.renderer.stats().refilled == 3);
}

// A compute pass writes UNORM storage; the same memory sampled as sRGB is the writer's image seen
// through a view of that format, requested once, bound as a view of the draw. Without the hook the
// image itself, noted.
void aliasFormats() {
    Fixture f;
    f.withViews = true;
    f.imageWrite();
    const uint64_t textures = f.renderer.stats().textures;
    vpps4::DrawState d = f.state();
    f.sample(kImg, 64, 64, 10, 9, 8);  // sRGB
    f.renderer.draw(d);
    f.renderer.draw(d);
    CHECK(f.renderer.stats().drawn == 2);
    CHECK(f.renderer.stats().textures == textures);  // no second image at that address
    CHECK(f.viewRequests.size() == 1);
    CHECK(!f.viewRequests.empty() && f.viewRequests[0].format == vp4gpu::Format::RGBA8Srgb);
    CHECK(f.renderer.stats().viewsMade == 1);
    f.renderer.submit();
    size_t withViews = 0;
    for (const vp4gpu::Command& c : f.backend.commands)
        if (const auto* p = std::get_if<vp4gpu::RenderPass>(&c))
            for (const vp4gpu::Draw& dr : p->draws) withViews += dr.views.size();
    CHECK(withViews == 2);
    // Without views.
    Fixture g;
    g.imageWrite();
    vpps4::DrawState gd = g.state();
    g.sample(kImg, 64, 64, 10, 9, 8);
    g.renderer.draw(gd);
    CHECK(g.renderer.stats().notes.count("GPU-written image sampled with another format of the same size (its own format)") == 1);
    CHECK(g.renderer.stats().viewsMade == 0);
}

// The same render target drawn as another format of its texel size (sRGB over UNORM): one image,
// the pass renders through a view format.
void aliasTargets() {
    Fixture f;
    f.withViews = true;
    vpps4::DrawState d = f.state();
    f.renderer.draw(d);
    d.color[0].numberType = 6;  // sRGB
    f.renderer.draw(d);
    CHECK(f.renderer.stats().targets == 1);
    f.renderer.submit();
    size_t viewFormats = 0;
    for (const vp4gpu::Command& c : f.backend.commands)
        if (const auto* p = std::get_if<vp4gpu::RenderPass>(&c))
            viewFormats += p->desc.colors[0].format != vp4gpu::Format::Undefined;
    CHECK(viewFormats == 1);
}

// The KITCHEN depth clear: a depth buffer rendered by a pass, a compute dispatch writes raw bytes
// over its start, the next depth-tested draw sees them: the depth image is written back before the
// dispatch and filled from memory after it.
void rawDepthClear() {
    Fixture f;
    f.renderer.draw(f.depthState(true));
    CHECK(f.renderer.tracker().ownerImage(kDepth, 128 * 128 * 4) != 0);
    f.rawWrite(kDepth, 2304 * 4);
    CHECK(f.renderer.stats().retiled == 1);
    f.renderer.draw(f.depthState(true));
    const auto e = f.events();
    const long retile = Fixture::indexOf(e, "copy retile " + Fixture::hex(kDepth));
    const long dispatch = Fixture::indexOf(e, "dispatch");
    const long refill = Fixture::indexOf(e, "copy upload " + Fixture::hex(kDepth));
    CHECK(retile >= 0 && dispatch > retile && refill > dispatch);
    CHECK(f.renderer.stats().refilled >= 2);  // the texture (first use) and the depth buffer
    if (failures) f.dump();
}

// A tiled render target is written back by the GPU tiler (a retile dispatch on the GPU timeline)
// before a raw reader sees its range, and is not refilled over its own newer bytes.
void tiledRetile() {
    Fixture f;
    vpps4::DrawState d = f.state();
    d.color[0].tileMode = 14;
    f.renderer.draw(d);
    f.rawWrite(kRt + 0x100, 256);
    CHECK(f.renderer.stats().retiled == 1);
    CHECK(f.renderer.stats().gpuRetiled == 1);
    CHECK(f.renderer.stats().notes.count("image newer than memory cannot be written back (layout not covered): raw readers see stale bytes") == 0);
    f.renderer.draw(d);
    if (failures) f.dump();
}

// Idle images are written back (when ahead of memory) and destroyed, a later use recreates them.
void eviction() {
    Fixture f;
    f.renderer.setResidency(3, 1ull << 40);
    f.renderer.draw(f.state());
    for (int i = 0; i < 24; ++i) f.renderer.submit();
    CHECK(f.renderer.stats().evicted == 2);  // the render target and the texture
    CHECK(f.renderer.stats().retiled == 1);
    CHECK(!bool(f.renderer.colorTargetAt(kRt)));
    CHECK(f.renderer.tracker().ownerImage(kRt, 128 * 128 * 4) == 0);
    const auto e = f.events();
    CHECK(Fixture::indexOf(e, "copy retile " + Fixture::hex(kRt)) >= 0);
    for (int i = 0; i < 12; ++i) {
        f.renderer.fill(kBase + 0x5000, 16, 0);  // (retired objects are destroyed by submits that carry work)
        f.renderer.submit();
    }
    size_t destroyed = 0;
    for (const std::string& c : f.backend.inner.calls) destroyed += c.rfind("destroyTexture", 0) == 0;
    CHECK(destroyed == 2);
    f.renderer.draw(f.state());  // used again
    CHECK(f.renderer.stats().textures == 2 && f.renderer.stats().targets == 2);
    // A byte budget evicts the least recently used first, never what the last submit used.
    Fixture g;
    g.renderer.setResidency(1ull << 40, 20000);
    g.renderer.draw(g.state());
    g.renderer.submit();
    g.renderer.draw(g.state());
    for (int i = 0; i < 20; ++i) g.renderer.submit();
    CHECK(g.renderer.stats().evicted >= 1);
}

// On a mirrored region the host watches pages: sources are watched after a fill (no hashing from
// then on), GPU-owned memory is watched until it is written back; syncToMemory answers a CPU read.
void hostWatches() {
    Fixture f(true);
    f.renderer.draw(f.state());
    CHECK(f.watchedWrites.size() == 1 && f.watchedWrites[0] == std::make_pair(kTex, uint64_t(64 * 64 * 4)));
    CHECK(f.ownedOn.size() == 1 && f.ownedOn[0] == std::make_pair(kRt, uint64_t(128 * 128 * 4)));
    *f.at(kTex) ^= 0xff;  // a CPU write the host did not report (it would have): no hashing
    f.renderer.submit();
    f.renderer.draw(f.state());
    CHECK(f.renderer.stats().refilled == 1);
    f.renderer.invalidate(kTex, 4);  // the host caught it
    f.renderer.draw(f.state());
    CHECK(f.renderer.stats().refilled == 2);
    CHECK(f.watchedWrites.size() == 2);  // watched again by the new fill
    CHECK(f.renderer.syncToMemory(kRt + 4, 16));
    CHECK(f.ownedOff.size() == 1);
    CHECK(!f.renderer.syncToMemory(kRt + 4, 16));  // memory is current now
    CHECK(f.renderer.stats().syncs == 1);
    const auto e = f.events();
    CHECK(Fixture::indexOf(e, "copy retile " + Fixture::hex(kRt)) >= 0);
    // A region that is not mirrored is never watched.
    Fixture g(false);
    g.renderer.draw(g.state());
    CHECK(g.watchedWrites.empty() && g.ownedOn.empty());
}

}  // namespace

int main(int argc, char** argv) {
    const std::string_view t = argc > 1 ? argv[1] : "";
    if (t == "rt_to_texture") rtToTexture();
    else if (t == "channel_order") channelOrder();
    else if (t == "raw_write_to_texture") rawWriteToTexture();
    else if (t == "raw_write_to_layer") rawWriteToLayer();
    else if (t == "texture_to_raw") textureToRaw();
    else if (t == "dma_to_texture") dmaToTexture();
    else if (t == "cpu_write") cpuWrite();
    else if (t == "alias_formats") aliasFormats();
    else if (t == "alias_targets") aliasTargets();
    else if (t == "raw_depth_clear") rawDepthClear();
    else if (t == "tiled_retile") tiledRetile();
    else if (t == "eviction") eviction();
    else if (t == "host_watches") hostWatches();
    else return std::fprintf(stderr, "unknown test\n"), 2;
    return failures ? 1 : 0;
}
