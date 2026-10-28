// Memory coherence on the NullBackend: the same guest bytes written as a render target, depth
// buffer, storage image, raw buffer or by DMA, and read as any of them. Frames are captured so the
// ORDER of dispatches, draws and copies on the GPU timeline can be asserted.
// Usage: vpps4_coherence_tests TEST
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "vpgpu/core/device.h"
#include "vpps4/renderer.h"
#include "vpps4/tiling.h"

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
class Recording final : public vpgpu::Backend {
public:
    vpgpu::NullBackend inner;
    std::vector<vpgpu::Command> commands;
    std::map<uint32_t, uint8_t*> memory;  // host memory of the buffers that alias some (guest windows, upload rings), by index

    void createBuffer(vpgpu::BufferId i, const vpgpu::BufferDesc& d) override {
        if (d.guestMemory) memory[i.index] = static_cast<uint8_t*>(d.guestMemory);
        inner.createBuffer(i, d);
    }
    void createTexture(vpgpu::TextureId i, const vpgpu::TextureDesc& d) override { inner.createTexture(i, d); }
    void createSampler(vpgpu::SamplerId i, const vpgpu::SamplerDesc& d) override { inner.createSampler(i, d); }
    void destroyBuffer(vpgpu::BufferId i) override { inner.destroyBuffer(i); }
    void destroyTexture(vpgpu::TextureId i) override { inner.destroyTexture(i); }
    void destroySampler(vpgpu::SamplerId i) override { inner.destroySampler(i); }
    uint64_t bufferAddress(vpgpu::BufferId i) const override { return inner.bufferAddress(i); }
    static inline bool refuseBcKernel = false;  // the block decoding kernel does not compile (the CPU decoder takes over)
    static inline std::atomic<bool> holdBcKernel{false};  // ... is still compiling (on a compile worker, see Fixture::workers)
    bool compilePipeline(const vpgpu::PipelineState& s, std::string& e) override {
        if (refuseBcKernel && s.label == "bcdecode") return e = "refused", false;
        while (holdBcKernel && s.label == "bcdecode") std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return inner.compilePipeline(s, e);
    }
    void destroyPipeline(vpgpu::PipelineKey k) override { inner.destroyPipeline(k); }
    void submit(uint64_t n, const vpgpu::Frame& f, const vpgpu::CompiledFrame& c) override {
        commands.insert(commands.end(), f.commands().begin(), f.commands().end());
        inner.submit(n, f, c);
    }
    bool takeTimings(uint64_t& n, vpgpu::GpuTimings& t) override { return inner.takeTimings(n, t); }
    uint64_t completedFrames() const override { return inner.completedFrames(); }
    bool lost() const override { return inner.lost(); }
    bool waitFramesBelow(uint64_t f, uint64_t t) override { return inner.waitFramesBelow(f, t); }
    bool waitAll(uint64_t t) override { return inner.waitAll(t); }
};

constexpr uint64_t kBase = 0x10000000, kSize = 1 << 20;
constexpr uint64_t kRt = kBase + 0x40000;     // 128x128 RGBA8, 64 KiB
constexpr uint64_t kTex = kBase + 0x10000;    // 64x64 RGBA8, 16 KiB
constexpr uint64_t kImg = kBase + 0x20000;    // storage image 64x64 RGBA8
constexpr uint64_t kDepth = kBase + 0x60000;  // 128x128 D32, 64 KiB

struct Fixture {
    static inline uint64_t memSize = kSize;   // guest memory (a test of a big texture makes more)
    static inline unsigned workers = 0;       // pipeline compile workers (0: compiled inside the request)
    void* host = std::aligned_alloc(16384, memSize);
    Recording backend;
    std::vector<std::string> log;
    vpgpu::Device device{backend, workers, [this](const std::string& s) { log.push_back(s); }};
    bool mirrored = false;
    std::vector<std::pair<uint64_t, uint64_t>> watchedWrites, ownedOn, ownedOff;
    std::vector<vpps4::ViewRequest> viewRequests;
    vpps4::Renderer renderer{device, hooks()};
    Program vs = load("vs_fetch"), ps = load("ps_textured"), store = load("cs_partial_store"), atomic = load("cs_image_atomic");

    explicit Fixture(bool mirror = false, bool views = false) : mirrored(mirror) {
        std::memset(host, 0, memSize);
        (void)views;
    }
    ~Fixture() { std::free(host); }

    bool withViews = false;
    vpps4::Hooks hooks() {
        vpps4::Hooks h;
        h.region = [this](uint64_t a, uint64_t s, vpps4::GuestRegion& g) {
            if (a < kBase || a + s > kBase + memSize) return false;
            g = {kBase, memSize, host, mirrored};
            return true;
        };
        h.shader = [](uint64_t, vpir::Stage, const vpmsl::MslShader&) {};
        h.log = [this](const std::string& s) { log.push_back(s); };
        h.watchWrites = [this](uint64_t a, uint64_t n) { watchedWrites.push_back({a, n}); };
        h.watchOwned = [this](uint64_t a, uint64_t n, bool on) { (on ? ownedOn : ownedOff).push_back({a, n}); };
        h.view = [this](const vpps4::ViewRequest& r) -> vpgpu::TextureId {
            if (!withViews) return {};
            viewRequests.push_back(r);
            vpgpu::TextureDesc d;
            d.format = r.format == vpgpu::Format::Undefined ? vpgpu::Format::RGBA8Unorm : r.format;
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
        for (const vpgpu::Command& c : backend.commands) {
            if (const auto* p = std::get_if<vpgpu::RenderPass>(&c)) {
                for (size_t i = 0; i < p->draws.size(); ++i) out.push_back("draw");
            } else if (std::holds_alternative<vpgpu::Dispatch>(c)) {
                out.push_back("dispatch");
            } else if (const auto* cp = std::get_if<vpgpu::Copy>(&c)) {
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

// A T# naming the render target: its own image, nothing copied. Another
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

const vpgpu::Copy* decodeCopy(Fixture& f, const std::string& label, size_t nth = 0) {
    for (const vpgpu::Command& c : f.backend.commands)
        if (const auto* cp = std::get_if<vpgpu::Copy>(&c))
            if (cp->label == label && nth-- == 0) return cp;
    return nullptr;
}
const uint8_t* copyBytes(Fixture& f, const vpgpu::Copy& c) {
    const auto it = f.backend.memory.find(c.srcBuffer.buffer.index);
    return it == f.backend.memory.end() ? nullptr : it->second + c.srcBuffer.offset;
}
// A sampled texture has the T#'s width and height: a T# smaller than the target at its address (the picture drawn in an
// allocation at the CB's pitch) is not read through the target, whose normalised coordinates would span the allocation.
// It is a texture of the T#'s size holding the corner of the target (a GPU copy redone when the target is written again,
// no round trip through memory). A read with integer coordinates only, or a target that cannot be copied, keeps the target.
void sampleExtent() {
    Fixture f;
    f.renderer.draw(f.state());
    const auto rt = f.renderer.colorTargetAt(kRt);
    auto extent = [&](vpgpu::TextureId id) {
        vpgpu::TextureDesc td;
        CHECK(f.device.textureDesc(id, td));
        return std::make_pair(td.width, td.height);
    };
    auto image = [&] { return vpps4::decodeImage(std::span<const uint32_t>(f.at(kBase + 0x200), 8)); };
    auto sampled = [&] { return f.renderer.textureFor(image()); };
    auto corners = [&] {
        unsigned n = 0;
        for (const std::string& e : f.events()) n += e == "copy sampled corner " + Fixture::hex(kRt);
        return n;
    };
    f.sample(kRt, 128, 128, 10, 0, 8);
    CHECK(sampled() == rt);
    CHECK(extent(sampled()) == std::make_pair(128u, 128u));
    CHECK(corners() == 0);
    f.sample(kRt, 64, 64, 10, 0, 8);
    const auto smaller = sampled();
    CHECK(smaller && smaller != rt);
    CHECK(extent(smaller) == std::make_pair(64u, 64u));
    CHECK(f.renderer.stats().retiled == 0);  // (no write back to memory and fill)
    CHECK(corners() == 1);
    // The copy is the corner of the target, from its origin, into the texture of the T#'s size.
    const vpgpu::Copy* cp = decodeCopy(f, "sampled corner " + Fixture::hex(kRt));
    CHECK(cp && cp->kind == vpgpu::CopyKind::TextureToTexture && cp->srcTexture.texture == rt && cp->dstTexture.texture == smaller);
    CHECK(cp && cp->srcTexture.x == 0 && cp->srcTexture.y == 0 && cp->dstTexture.x == 0 && cp->dstTexture.y == 0);
    CHECK(cp && cp->srcTexture.width == 64 && cp->srcTexture.height == 64 && cp->dstTexture.width == 64 && cp->dstTexture.height == 64);
    // Sampled again with the target unchanged: the same texture, no new copy.
    CHECK(sampled() == smaller && corners() == 1);
    // The target written again: the next use copies again.
    f.renderer.draw(f.state());
    f.sample(kRt, 64, 64, 10, 0, 8);  // (the draw's state names its own T#)
    CHECK(sampled() == smaller && corners() == 2);
    CHECK(sampled() == smaller && corners() == 2);
    f.sample(kRt, 128, 100, 10, 0, 8);  // (one side only)
    CHECK(sampled() != rt && extent(sampled()) == std::make_pair(128u, 100u));
    // Integer coordinates only: the target itself (texels addressed as in the allocation).
    f.sample(kRt, 64, 64, 10, 0, 8);
    CHECK(f.renderer.textureFor(image(), true) == rt);
    // A format of the same texel size without views: no copy can be sampled as that format, the target is kept.
    f.sample(kRt, 64, 64, 5, 0, 8);
    CHECK(sampled() == rt);
    // A T# larger than the target never was the target.
    f.sample(kRt, 256, 128, 10, 0, 8);
    CHECK(sampled() != rt && extent(sampled()) == std::make_pair(256u, 128u));
    if (failures) f.dump();
}

// A target nothing wrote (a multisampled source a resolve had to make up) holds nothing to copy: a T# smaller than it is
// not given a corner of it (an empty picture), the target itself is sampled. Drawn into, it is copied as usual.
void mirrorUnwritten() {
    Fixture f;
    f.renderer.setMaxSamples(4);
    const uint64_t a = kBase + 0x70000;
    vpps4::ColorTarget src = f.state().color[0], dst = src;
    src.address = a;
    src.samples = 4;
    dst.address = kBase + 0xA0000;
    f.renderer.resolve(src, dst);  // (nothing was drawn at `a`)
    auto corners = [&] {
        unsigned n = 0;
        for (const std::string& e : f.events()) n += e == "copy sampled corner " + Fixture::hex(a);
        return n;
    };
    f.sample(a, 64, 64, 10, 0, 8);
    const auto t = f.renderer.textureFor(vpps4::decodeImage(std::span<const uint32_t>(f.at(kBase + 0x200), 8)));
    vpgpu::TextureDesc td;
    CHECK(t && f.device.textureDesc(t, td) && td.width == 128 && td.height == 128);
    CHECK(corners() == 0);
    // The destination of that resolve was written by it: its corner is copied.
    f.sample(dst.address, 64, 64, 10, 0, 8);
    const auto m = f.renderer.textureFor(vpps4::decodeImage(std::span<const uint32_t>(f.at(kBase + 0x200), 8)));
    CHECK(m && f.device.textureDesc(m, td) && td.width == 64 && td.height == 64);
    // Drawn into, it is copied like any target.
    vpps4::DrawState d = f.state();
    d.color[0].address = a;
    d.color[0].samples = 4;
    f.renderer.draw(d);
    f.sample(a, 64, 64, 10, 0, 8);
    const auto u = f.renderer.textureFor(vpps4::decodeImage(std::span<const uint32_t>(f.at(kBase + 0x200), 8)));
    CHECK(u && f.device.textureDesc(u, td) && td.samples == 1 && td.width == 64 && td.height == 64);
    CHECK(corners() == 1);
    if (failures) f.dump();
}

// A target no pass drew into whose memory the CPU wrote since is filled from it before the question whether it holds
// anything: its corner is copied for a smaller T#.
void mirrorRefilled() {
    Fixture f;
    const uint64_t a = kBase + 0x70000;
    vpps4::ColorTarget src = f.state().color[0], dst = src;
    src.address = a;
    dst.address = kBase + 0xA0000;
    f.renderer.resolve(src, dst);  // makes the 1x source target at `a`, never drawn and never written
    auto corners = [&] {
        unsigned n = 0;
        for (const std::string& e : f.events()) n += e == "copy sampled corner " + Fixture::hex(a);
        return n;
    };
    f.sample(a, 64, 64, 10, 0, 8);
    f.renderer.textureFor(vpps4::decodeImage(std::span<const uint32_t>(f.at(kBase + 0x200), 8)));
    CHECK(corners() == 0);
    f.renderer.invalidate(a, 8);  // the CPU wrote the target's memory
    f.renderer.textureFor(vpps4::decodeImage(std::span<const uint32_t>(f.at(kBase + 0x200), 8)));
    CHECK(corners() == 1);
    // The copy never reaches past the source target.
    const vpgpu::Copy* cp = decodeCopy(f, "sampled corner " + Fixture::hex(a));
    CHECK(cp && cp->srcTexture.width <= 128 && cp->srcTexture.height <= 128 && cp->srcTexture.width == 64);
    if (failures) f.dump();
}

// A render target refreshed from memory (the CPU wrote its range, the image is filled again) is written too: the copy of its
// corner is made again, though no pass drew into it.
void colorMirrorRefill() {
    Fixture f;
    f.renderer.draw(f.state());
    auto corners = [&] {
        unsigned n = 0;
        for (const std::string& e : f.events()) n += e == "copy sampled corner " + Fixture::hex(kRt);
        return n;
    };
    auto uploads = [&] { return Fixture::count(f.events(), "copy upload " + Fixture::hex(kRt)); };
    auto sampled = [&] {
        f.sample(kRt, 64, 64, 10, 0, 8);
        return f.renderer.textureFor(vpps4::decodeImage(std::span<const uint32_t>(f.at(kBase + 0x200), 8)));
    };
    const auto mirror = sampled();
    CHECK(mirror && mirror != f.renderer.colorTargetAt(kRt));
    CHECK(corners() == 1 && uploads() == 0);
    CHECK(sampled() == mirror && corners() == 1);
    f.renderer.invalidate(kRt, 8);  // the CPU wrote the target's memory
    CHECK(sampled() == mirror);
    CHECK(uploads() == 1);
    CHECK(corners() == 2);
    CHECK(sampled() == mirror && corners() == 2);
    if (failures) f.dump();
}

// An image ahead of memory that cannot be written back (the guest memory is read only here) leaves the texture over its range
// unfilled, not marked as filled with stale bytes: once the write back can be done, the next use does it and fills.
void syncRefusedFill() {
    Fixture f;
    f.renderer.setGuestReadOnly(true);
    f.renderer.draw(f.state());
    f.sample(kRt, 128, 128, 3, 0, 8);  // 8_8: another texel size, through memory
    auto image = [&] { return vpps4::decodeImage(std::span<const uint32_t>(f.at(kBase + 0x200), 8)); };
    CHECK(f.renderer.textureFor(image()));
    CHECK(f.renderer.stats().notes.count("image newer than memory cannot be written back (layout not covered): raw readers see stale bytes") == 1);
    auto e = f.events();
    CHECK(Fixture::indexOf(e, "copy retile " + Fixture::hex(kRt)) < 0 && Fixture::indexOf(e, "copy upload " + Fixture::hex(kRt)) < 0);
    f.renderer.setGuestReadOnly(false);
    CHECK(f.renderer.textureFor(image()));
    e = f.events();
    const long retile = Fixture::indexOf(e, "copy retile " + Fixture::hex(kRt));
    CHECK(retile >= 0 && Fixture::indexOf(e, "copy upload " + Fixture::hex(kRt)) > retile);
    if (failures) f.dump();
}

// A depth target sampled through a smaller T#: the copy of its corner is made again after every write of the target, a draw
// or a clear.
void depthMirror() {
    Fixture f;
    auto corners = [&] {
        unsigned n = 0;
        for (const std::string& e : f.events()) n += e == "copy sampled corner " + Fixture::hex(kDepth);
        return n;
    };
    auto sampled = [&] {
        f.sample(kDepth, 64, 64, 4, 7, 8);
        return f.renderer.textureFor(vpps4::decodeImage(std::span<const uint32_t>(f.at(kBase + 0x200), 8)));
    };
    vpps4::DrawState cleared = f.depthState(true);
    cleared.depth.depthClear = true;
    f.renderer.draw(cleared);
    const auto mirror = sampled();
    vpgpu::TextureDesc td;
    CHECK(mirror && f.device.textureDesc(mirror, td) && td.width == 64 && td.height == 64);
    CHECK(corners() == 1);
    CHECK(sampled() == mirror && corners() == 1);  // (nothing written since)
    f.renderer.draw(f.depthState(true));
    CHECK(sampled() == mirror && corners() == 2);
    f.renderer.draw(cleared);
    CHECK(sampled() == mirror && corners() == 3);
    f.renderer.invalidate(kDepth, 8);  // the CPU wrote the target's memory: the target is filled again, no pass drew into it
    CHECK(sampled() == mirror && corners() == 4);
    if (failures) f.dump();
}

// A multisampled T# smaller than its target: no host copy between multisampled textures, so the target's samples are
// sampled as they are.
void msaaMirror() {
    Fixture f;
    vpps4::DrawState d = f.state();
    d.color[0].samples = 4;
    f.renderer.draw(d);
    f.sample(kRt, 64, 64, 10, 0, 8);
    f.at(kBase + 0x200)[3] = (f.at(kBase + 0x200)[3] & ~(15u << 28)) | (14u << 28) | (2u << 16);
    const auto t = f.renderer.textureFor(vpps4::decodeImage(std::span<const uint32_t>(f.at(kBase + 0x200), 8)));
    vpgpu::TextureDesc td;
    CHECK(t && f.device.textureDesc(t, td) && td.samples == 4 && td.width == 128 && td.height == 128);
    unsigned corners = 0;
    for (const std::string& e : f.events()) corners += e.rfind("copy sampled corner", 0) == 0;
    CHECK(corners == 0);
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
    CHECK(f.viewRequests.size() == 1 && f.viewRequests[0].format == vpgpu::Format::BGRA8Srgb);
    CHECK(f.renderer.stats().notes.count("image sampled with another format (no view available: its own format)") == 0);
}

// A BC1 texture on a host without BC (the visionOS case): a host texture of plain RGBA8 at the T#'s texel size (also when
// it is not a multiple of 4), every mip level decoded from the guest's blocks, decoded again when memory changes.
void softBcUpload() {
    vpps4::setFormatCaps({false});
    Fixture f;
    f.renderer.setGpuTiling(false);  // (the CPU decoder: the fallback, and what the GPU kernel is compared with)
    // 6x5 texels, three levels (6x5, 3x2, 1x1), linear, pitch 6: level 0 is 2x2 blocks (red, green / blue, white), levels 1
    // and 2 one block each (flat magenta, cyan). A BC1 block is 8 bytes.
    const uint8_t red[8] = {0x00, 0xF8, 0x00, 0xF8, 0, 0, 0, 0}, green[8] = {0xE0, 0x07, 0xE0, 0x07, 0, 0, 0, 0};
    const uint8_t blue[8] = {0x1F, 0x00, 0x1F, 0x00, 0, 0, 0, 0}, white[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0};
    const uint8_t magenta[8] = {0x1F, 0xF8, 0x1F, 0xF8, 0, 0, 0, 0}, cyan[8] = {0xFF, 0x07, 0xFF, 0x07, 0, 0, 0, 0};
    vpps4::TiledSurface surface;
    surface.tileIndex = 8;
    surface.bytesPerElement = 8;
    surface.blockSize = 4;
    surface.width = 6;
    surface.height = 5;
    surface.pitch = 6;
    surface.levels = 3;
    const vpps4::SurfaceLayout lay = vpps4::computeLayout(surface);
    CHECK(lay.ok && lay.levels.size() == 3);
    uint8_t* mem = reinterpret_cast<uint8_t*>(f.at(kTex));
    const uint64_t pitchBytes = uint64_t(lay.levels[0].pitch) * 8;
    std::memcpy(mem + 0, red, 8), std::memcpy(mem + 8, green, 8);
    std::memcpy(mem + pitchBytes, blue, 8), std::memcpy(mem + pitchBytes + 8, white, 8);
    std::memcpy(mem + lay.levels[1].offset, magenta, 8), std::memcpy(mem + lay.levels[2].offset, cyan, 8);
    vpps4::DrawState d = f.state();
    f.sample(kTex, 6, 5, 35, 0, 8);
    f.at(kBase + 0x200)[3] |= 2u << 16;  // last level 2
    f.renderer.draw(d);
    CHECK(f.renderer.stats().drawn == 1 && f.renderer.stats().copies == 3 && f.renderer.stats().textures == 1);
    const vpps4::ImageDesc image = vpps4::decodeImage(std::span<const uint32_t>(f.at(kBase + 0x200), 8));
    vpgpu::TextureDesc td;
    CHECK(f.device.textureDesc(f.renderer.textureFor(image), td));
    CHECK(td.format == vpgpu::Format::RGBA8Unorm && td.width == 6 && td.height == 5 && td.mips == 3);
    f.renderer.submit();
    const std::string base = "decode " + Fixture::hex(kTex);
    const vpgpu::Copy* l0 = decodeCopy(f, base);
    const vpgpu::Copy* l1 = decodeCopy(f, base + " mip 1");
    const vpgpu::Copy* l2 = decodeCopy(f, base + " mip 2");
    CHECK(l0 && l1 && l2);
    if (!l0 || !l1 || !l2) return f.dump();
    CHECK(l0->dstTexture.width == 6 && l0->dstTexture.height == 5 && l0->srcBuffer.bytesPerRow == 24);
    CHECK(l1->dstTexture.width == 3 && l1->dstTexture.height == 2 && l2->dstTexture.width == 1 && l2->dstTexture.height == 1);
    const uint8_t* p0 = copyBytes(f, *l0);
    CHECK(p0 != nullptr);
    if (!p0) return;
    auto px = [&](const uint8_t* base0, unsigned pitch, unsigned x, unsigned y) { return base0 + y * pitch + 4 * x; };
    const uint8_t rgbaRed[4] = {255, 0, 0, 255}, rgbaGreen[4] = {0, 255, 0, 255}, rgbaBlue[4] = {0, 0, 255, 255};
    CHECK(std::memcmp(px(p0, 24, 0, 0), rgbaRed, 4) == 0 && std::memcmp(px(p0, 24, 5, 0), rgbaGreen, 4) == 0);
    CHECK(std::memcmp(px(p0, 24, 3, 4), rgbaBlue, 4) == 0 && px(p0, 24, 5, 4)[0] == 255 && px(p0, 24, 5, 4)[1] == 255);
    const uint8_t* p1 = copyBytes(f, *l1);
    const uint8_t* p2 = copyBytes(f, *l2);
    CHECK(p1 && p2 && p1[0] == 255 && p1[1] == 0 && p1[2] == 255 && px(p1, 12, 2, 1)[2] == 255 && p2[0] == 0 && p2[1] == 255 && p2[2] == 255);
    // Memory written by the CPU: decoded again, the new texels in a second set of copies.
    std::memcpy(mem, blue, 8);
    f.renderer.invalidate(kTex, 8);
    f.renderer.draw(d);
    CHECK(f.renderer.stats().refilled == 2);
    f.renderer.submit();
    const vpgpu::Copy* again = decodeCopy(f, base, 1);
    CHECK(again && copyBytes(f, *again) && std::memcmp(copyBytes(f, *again), rgbaBlue, 4) == 0);
    if (failures) f.dump();
    vpps4::setFormatCaps({true});
}

// The same texture through the GPU kernel: chosen when it is ready, only the levels whose memory changed decoded again, the
// CPU decoder when the kernel cannot run.
unsigned countLabel(Fixture& f, const std::string& label, bool dispatches) {
    f.renderer.submit();
    unsigned n = 0;
    for (const vpgpu::Command& c : f.backend.commands) {
        if (const auto* cp = std::get_if<vpgpu::Copy>(&c)) n += !dispatches && cp->label == label;
        else if (const auto* dp = std::get_if<vpgpu::Dispatch>(&c)) n += dispatches && dp->label == label;
    }
    return n;
}
void softBcGpu() {
    vpps4::setFormatCaps({false});
    const uint8_t blue[8] = {0x1F, 0x00, 0x1F, 0x00, 0, 0, 0, 0}, cyan[8] = {0xFF, 0x07, 0xFF, 0x07, 0, 0, 0, 0};
    vpps4::TiledSurface surface;
    surface.tileIndex = 8;
    surface.bytesPerElement = 8;
    surface.blockSize = 4;
    surface.width = 6;
    surface.height = 5;
    surface.pitch = 6;
    surface.levels = 3;
    const vpps4::SurfaceLayout lay = vpps4::computeLayout(surface);
    CHECK(lay.ok && lay.levels.size() == 3);
    auto prepare = [&](Fixture& f) {
        std::memcpy(f.at(kTex), blue, 8);
        f.sample(kTex, 6, 5, 35, 0, 8);
        f.at(kBase + 0x200)[3] |= 2u << 16;  // last level 2
    };
    const std::string m0 = "gpu bcdecode mip 0", m1 = "gpu bcdecode mip 1", m2 = "gpu bcdecode mip 2";
    {
        Fixture f(true);  // (guest memory the host watches: a CPU write is told, not looked for in the content)
        vpps4::DrawState d = f.state();
        prepare(f);  // (after the state: it names its own T#)
        f.renderer.draw(d);
        CHECK(f.renderer.stats().drawn == 1 && f.renderer.stats().textures == 1);
        // One dispatch and one copy per level, and no CPU decoded copy.
        CHECK(countLabel(f, m0, true) == 1 && countLabel(f, m1, true) == 1 && countLabel(f, m2, true) == 1);
        CHECK(countLabel(f, m0, false) == 1 && countLabel(f, m1, false) == 1 && countLabel(f, m2, false) == 1);
        CHECK(countLabel(f, "decode " + Fixture::hex(kTex), false) == 0);
        // Nothing written: the next use decodes nothing.
        f.renderer.draw(d);
        CHECK(countLabel(f, m0, true) == 1);
        // Memory of the last level written: only that level is decoded again.
        std::memcpy(f.at(kTex) + lay.levels[2].offset, cyan, 8);
        f.renderer.invalidate(kTex + lay.levels[2].offset, 8);
        f.renderer.draw(d);
        CHECK(countLabel(f, m0, true) == 1 && countLabel(f, m1, true) == 1 && countLabel(f, m2, true) == 2);
        // ... and the first one.
        std::memcpy(f.at(kTex), cyan, 8);
        f.renderer.invalidate(kTex, 8);
        f.renderer.draw(d);
        // (the content check of a source whose watch was dropped may find the change in its sampled lines and call it a write of
        // the whole texture: the other levels may be decoded too)
        CHECK(countLabel(f, m0, true) == 2 && countLabel(f, m2, true) >= 2);
        // The host tells of a write to the last level (its watch is dropped); a second write, to the first level, is not told
        // (the page is open now): both are decoded again.
        const unsigned before0 = countLabel(f, m0, true), before2 = countLabel(f, m2, true);
        std::memcpy(f.at(kTex) + lay.levels[2].offset, blue, 8);
        f.renderer.invalidate(kTex + lay.levels[2].offset, 8);
        std::memcpy(f.at(kTex), blue, 8);
        f.at(kTex)[1] ^= 0x40;
        f.renderer.draw(d);
        CHECK(countLabel(f, m0, true) == before0 + 1 && countLabel(f, m2, true) == before2 + 1);
        if (failures) f.dump();
    }
    {  // The kernel does not compile: decoded on the CPU, the texture is filled all the same.
        Recording::refuseBcKernel = true;  // (before the renderer starts compiling it)
        Fixture f;
        vpps4::DrawState d = f.state();
        prepare(f);
        f.renderer.draw(d);
        Recording::refuseBcKernel = false;
        CHECK(f.renderer.stats().drawn == 1);
        CHECK(countLabel(f, m0, true) == 0 && countLabel(f, "decode " + Fixture::hex(kTex), false) == 1);
        if (failures) f.dump();
    }
    vpps4::setFormatCaps({true});
}

// Block-compressed textures on a host without BC are decoded within a budget per frame: a texture past it stays unfilled, its
// draw is left out (it says why) and it is decoded at a later frame; the first decode of a frame is always admitted. Through
// the GPU kernel, and through the CPU decoder (GPU tiling off, or the kernel not compiling).
void softBcBudget() {
    vpps4::setFormatCaps({false});
    const uint64_t second = kBase + 0x30000;
    const uint8_t blue[8] = {0x1F, 0x00, 0x1F, 0x00, 0, 0, 0, 0};
    for (int mode = 0; mode < 3; ++mode) {  // GPU kernel, GPU tiling off, kernel refused
        Recording::refuseBcKernel = mode == 2;
        Fixture f(true);
        if (mode == 1) f.renderer.setGpuTiling(false);
        vpps4::DrawState d = f.state();
        std::memcpy(f.at(kTex), blue, 8), std::memcpy(f.at(second), blue, 8);
        f.renderer.setMemoryLimits(512ull << 20, 768ull << 20, 100);  // (a 6x5 texture with 3 levels decodes 148 bytes)
        auto drawAt = [&](uint64_t address) {
            f.sample(address, 6, 5, 35, 0, 8);
            f.at(kBase + 0x200)[3] |= 2u << 16;  // last level 2
            f.renderer.draw(d);
        };
        auto skipped = [&] {
            const auto& m = f.renderer.stats().skipped;
            return m.count("ps: texture unavailable: bc decode budget") ? m.at("ps: texture unavailable: bc decode budget") : 0;
        };
        auto decodes = [&] {  // (submits: a new frame)
            const auto e = f.events();
            unsigned n = unsigned(Fixture::count(e, "copy gpu bcdecode mip 0"));  // (the level 0 of each texture)
            for (const std::string& x : e) n += x == "copy decode " + Fixture::hex(kTex) || x == "copy decode " + Fixture::hex(second);
            return n;
        };
        drawAt(kTex);  // (the first decode of the frame: admitted, though over the budget)
        CHECK(f.renderer.stats().drawn == 1 && skipped() == 0);
        drawAt(second);  // the same frame: over the budget
        CHECK(f.renderer.stats().drawn == 1 && skipped() == 1);
        drawAt(second);
        CHECK(f.renderer.stats().drawn == 1 && skipped() == 2);
        CHECK(decodes() == 1);  // (only the first texture's)
        drawAt(second);         // the next frame
        CHECK(f.renderer.stats().drawn == 2 && skipped() == 2);
        CHECK(decodes() == 2);
        drawAt(second);  // filled: nothing decoded again
        CHECK(f.renderer.stats().drawn == 3);
        CHECK(decodes() == 2);
        if (failures) f.dump();
    }
    Recording::refuseBcKernel = false;
    vpps4::setFormatCaps({true});
}

// While the BC kernel is still compiling nothing is decoded on the render thread: the draw is left out and the texture is
// decoded by the kernel once it is ready.
void softBcKernelWait() {
    vpps4::setFormatCaps({false});
    Recording::holdBcKernel = true;
    Fixture::workers = 2;  // (one worker waits for the kernel, the other compiles the draw's pipelines)
    {
        Fixture f;
        vpps4::DrawState d = f.state();
        for (int i = 0; i < 5000 && f.renderer.stats().drawn == 0; ++i) {  // (a plain texture: no kernel)
            f.renderer.draw(d);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        CHECK(f.renderer.stats().drawn == 1);
        const uint8_t blue[8] = {0x1F, 0x00, 0x1F, 0x00, 0, 0, 0, 0};
        std::memcpy(f.at(kTex), blue, 8);
        f.sample(kTex, 6, 5, 35, 0, 8);
        f.at(kBase + 0x200)[3] |= 2u << 16;
        f.renderer.draw(d);
        f.renderer.draw(d);
        const auto& skipped = f.renderer.stats().skipped;
        CHECK(skipped.count("ps: texture unavailable: bc kernel compiling") && skipped.at("ps: texture unavailable: bc kernel compiling") == 2);
        CHECK(f.renderer.stats().drawn == 1);
        CHECK(countLabel(f, "decode " + Fixture::hex(kTex), false) == 0 && countLabel(f, "gpu bcdecode mip 0", false) == 0);
        Recording::holdBcKernel = false;
        f.device.pipelines().waitIdle();
        f.renderer.draw(d);
        CHECK(f.renderer.stats().drawn == 2);
        CHECK(countLabel(f, "gpu bcdecode mip 0", true) == 1 && countLabel(f, "decode " + Fixture::hex(kTex), false) == 0);
        if (failures) f.dump();
    }
    Recording::holdBcKernel = false;
    Fixture::workers = 0;
    vpps4::setFormatCaps({true});
}

// A host without BC holds less staging memory (every block-compressed texture is decoded into it): a texture whose decoded
// texels do not fit is not decoded (noted), and is when the limit is raised.
void softBcStagingLimit() {
    vpps4::setFormatCaps({false});
    Fixture::memSize = 32ull << 20;
    {
        Fixture f;
        f.renderer.setGpuTiling(false);
        // BC6H 4096x4224 (17.3 MiB of blocks, linear): decoded to 16-bit float RGBA, 138 MiB.
        const uint64_t big = kBase + 0x100000;
        vpps4::DrawState d = f.state();
        f.sample(big, 4096, 4224, 40, 0, 8);
        f.renderer.draw(d);
        CHECK(f.renderer.stats().notes.count("staging memory limit reached: copy source refused") == 1);
        CHECK(countLabel(f, "decode " + Fixture::hex(big), false) == 0);
        f.renderer.setMemoryLimits(512ull << 20, 768ull << 20, 1ull << 30);
        f.renderer.draw(d);
        CHECK(countLabel(f, "decode " + Fixture::hex(big), false) == 1);
        if (failures) f.dump();
    }
    Fixture::memSize = kSize;
    vpps4::setFormatCaps({true});
}

// The same through a tiled layout (thin 2D), 16x16 texels as 4x4 blocks, each block flat with its own colour, and BC4 as
// an 8-bit single channel texture.
void softBcTiled() {
    vpps4::setFormatCaps({false});
    Fixture f;
    f.renderer.setGpuTiling(false);
    vpps4::TiledSurface surface;
    surface.tileIndex = 14;
    surface.bytesPerElement = 8;
    surface.blockSize = 4;
    surface.width = surface.height = surface.pitch = 16;
    const vpps4::SurfaceLayout lay = vpps4::computeLayout(surface);
    CHECK(lay.ok && lay.levels.size() == 1);
    if (!lay.ok) return f.dump();
    std::vector<uint8_t> blocks(16 * 8, 0), tiled(lay.bytes, 0);
    auto color565 = [](unsigned bx, unsigned by) { return uint16_t(((bx * 8 + 1) << 11) | ((by * 16 + 2) << 5) | (bx + by)); };
    for (unsigned by = 0; by < 4; ++by)
        for (unsigned bx = 0; bx < 4; ++bx) {
            const uint16_t c = color565(bx, by);
            uint8_t* b = &blocks[(by * 4 + bx) * 8];
            b[0] = b[2] = uint8_t(c), b[1] = b[3] = uint8_t(c >> 8);
        }
    CHECK(vpps4::tileLevels(lay, 0, 1, 0, blocks.data(), tiled.data()));
    const uint64_t tex = kBase + 0x30000;
    std::memcpy(f.at(tex), tiled.data(), tiled.size());
    vpps4::DrawState d = f.state();
    f.sample(tex, 16, 16, 35, 0, 14);
    f.renderer.draw(d);
    CHECK(f.renderer.stats().drawn == 1 && f.renderer.stats().detiled == 1 && f.renderer.stats().gpuDetiled == 0);
    f.renderer.submit();
    const vpgpu::Copy* c = decodeCopy(f, "decode " + Fixture::hex(tex));
    const uint8_t* p = c ? copyBytes(f, *c) : nullptr;
    CHECK(p != nullptr);
    if (p)
        for (unsigned by = 0; by < 4; ++by)
            for (unsigned bx = 0; bx < 4; ++bx) {
                const uint16_t col = color565(bx, by);
                const uint8_t* t = p + (by * 4) * 64 + bx * 16;  // (rows of 16 texels x 4 bytes)
                CHECK(t[0] == uint8_t((col >> 11) << 3 | (col >> 11) >> 2) && t[1] == uint8_t(((col >> 5) & 63) << 2 | ((col >> 5) & 63) >> 4));
                CHECK(t[2] == uint8_t((col & 31) << 3 | (col & 31) >> 2) && t[3] == 255);
            }
    // BC4 is a single 8-bit channel.
    const uint64_t tex4 = kBase + 0x34000;
    const uint8_t bc4[8] = {77, 77, 0, 0, 0, 0, 0, 0};  // (six-value mode, index 0 everywhere: 77)
    std::memcpy(f.at(tex4), bc4, 8);
    f.sample(tex4, 4, 4, 38, 0, 8);
    f.renderer.draw(d);
    f.renderer.submit();
    const vpps4::ImageDesc image = vpps4::decodeImage(std::span<const uint32_t>(f.at(kBase + 0x200), 8));
    vpgpu::TextureDesc td;
    CHECK(f.device.textureDesc(f.renderer.textureFor(image), td));
    CHECK(td.format == vpgpu::Format::R8Unorm && td.width == 4 && td.height == 4);
    const vpgpu::Copy* c4 = decodeCopy(f, "decode " + Fixture::hex(tex4));
    const uint8_t* p4 = c4 ? copyBytes(f, *c4) : nullptr;
    CHECK(p4 && p4[0] == 77 && p4[15] == 77);
    if (failures) f.dump();
    vpps4::setFormatCaps({true});
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
    CHECK(!f.viewRequests.empty() && f.viewRequests[0].format == vpgpu::Format::RGBA8Srgb);
    CHECK(f.renderer.stats().viewsMade == 1);
    f.renderer.submit();
    size_t withViews = 0;
    for (const vpgpu::Command& c : f.backend.commands)
        if (const auto* p = std::get_if<vpgpu::RenderPass>(&c))
            for (const vpgpu::Draw& dr : p->draws) withViews += dr.views.size();
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

// A T# with base_level 1 (levels 1..3) is sampled through a view of those levels, so the shader's level 0 is the
// base; without views the old note remains.
void baseLevelView() {
    Fixture f;
    f.withViews = true;
    const vpps4::DrawState d = f.state();
    f.at(kBase + 0x200)[3] |= (1u << 12) | (3u << 16);  // base level 1, last level 3
    f.renderer.draw(d);
    f.renderer.draw(d);
    CHECK(f.renderer.stats().drawn == 2);
    CHECK(f.viewRequests.size() == 1);
    CHECK(!f.viewRequests.empty() && f.viewRequests[0].baseMip == 1 && f.viewRequests[0].mips == 3);
    CHECK(f.renderer.stats().notes.count("T# base level ignored: the texture holds the whole mip chain from its address") == 0);
    Fixture g;
    const vpps4::DrawState gd = g.state();
    g.at(kBase + 0x200)[3] |= (1u << 12) | (3u << 16);
    g.renderer.draw(gd);
    CHECK(g.renderer.stats().notes.count("T# base level ignored: the texture holds the whole mip chain from its address") == 1);
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
    for (const vpgpu::Command& c : f.backend.commands)
        if (const auto* p = std::get_if<vpgpu::RenderPass>(&c))
            viewFormats += p->desc.colors[0].format != vpgpu::Format::Undefined;
    CHECK(viewFormats == 1);
}

// The KITCHEN depth clear: a depth buffer rendered by a pass, a compute dispatch writes raw bytes
// over its start, the next depth-tested draw sees them: the depth image is written back before the
// dispatch and filled from memory after it.
void rawDepthClear() {
    Fixture f;
    vpps4::DrawState first = f.depthState(true);
    first.depth.depthClear = true;  // (the image holds the whole plane only after a clear: only then it is written back)
    f.renderer.draw(first);
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

// ---- Depth write back ----------------------------------------------------------------------------------------

constexpr uint64_t kStencil = kBase + 0x70000;  // 128x128 S8, 16 KiB (the stencil plane is apart from Z in guest memory)

// A depth draw with Z format `z` (1 Z_16, 3 Z_32_FLOAT), a stencil plane at kStencil if `stencil`, tile mode `tile`; `clear`
// clears every plane the draw has (the image then holds the whole of them).
vpps4::DrawState zState(Fixture& f, uint8_t z, bool stencil, uint8_t tile, bool clear = true) {
    vpps4::DrawState d = f.depthState(true);
    d.depth.zFormat = z;
    d.depth.stencilAddress = stencil ? kStencil : 0;
    d.depth.stencilFormat = stencil ? 1 : 0;
    d.depth.tileMode = tile;
    d.depth.depthClear = clear;
    d.depth.stencilClear = clear && stencil;
    d.depthStencil.stencilTest = stencil;
    return d;
}

// Plays the recorded retile of one plane of a 128x128 depth image as the GPU would: the copy that fills the scratch with
// the image's texels (pseudo-random bytes here), then the kernel that tiles them into the guest buffer. Guest memory at
// `address` must then be what the CPU tiler makes of the same texels, and detile back to them.
void checkTiledPlane(Fixture& f, vpgpu::Aspect aspect, uint64_t address, uint32_t bpe, uint8_t tile) {
    vpps4::TiledSurface t;
    t.tileIndex = tile, t.bytesPerElement = bpe, t.width = t.height = 128;
    const vpps4::SurfaceLayout lay = vpps4::computeLayout(t);
    CHECK(lay.ok);
    if (!lay.ok) return;
    std::vector<uint8_t> linear(lay.linearBytes);
    uint32_t x = 0x9E3779B9u * (bpe + tile);
    for (uint8_t& b : linear) x = x * 1664525u + 1013904223u, b = uint8_t(x >> 24);
    uint8_t* guest = static_cast<uint8_t*>(f.host) + (address - kBase);
    std::memset(guest, 0xCD, lay.bytes);
    bool armed = false;
    unsigned ran = 0;
    for (const vpgpu::Command& c : f.backend.commands) {
        if (const auto* cp = std::get_if<vpgpu::Copy>(&c)) {
            armed = cp->kind == vpgpu::CopyKind::TextureToBuffer && cp->srcTexture.aspect == aspect && cp->label.rfind("gpu retile", 0) == 0;
            if (armed) std::memcpy(f.backend.memory.at(cp->dstBuffer.buffer.index) + cp->dstBuffer.offset, linear.data(), linear.size());
        } else if (const auto* dp = std::get_if<vpgpu::Dispatch>(&c)) {
            if (!armed || dp->label.rfind("gpu retile", 0) != 0) continue;
            armed = false;
            const uint8_t *lin = nullptr, *params = nullptr;
            uint8_t* tiled = nullptr;
            for (const vpgpu::Access& a : dp->bindings) {
                uint8_t* m = f.backend.memory.at(a.range.index) + a.range.begin;
                if (a.kind == vpgpu::BindKind::Buffer && a.slot == 0) lin = m;
                if (a.kind == vpgpu::BindKind::Buffer && a.slot == 1) tiled = m;
                if (a.kind == vpgpu::BindKind::ConstantBuffer && a.slot == 0) params = m;
            }
            CHECK(lin && tiled && params);
            if (!lin || !tiled || !params) return;
            CHECK(tiled == guest);  // the plane lands at its own base
            VptParams P;
            std::memcpy(&P, params, sizeof P);
            CHECK(P.level.bpe == bpe);
            for (uint32_t gy = 0; gy < dp->groups[1] * 8; ++gy)
                for (uint32_t gx = 0; gx < dp->groups[0] * 8; ++gx)
                    if (vpt_in_range(P, gx, gy, 0)) std::memcpy(tiled + vpt_tiled_offset(P, gx, gy, 0), lin + vpt_linear_offset(P, gx, gy, 0), bpe);
            ++ran;
        }
    }
    CHECK(ran == 1);
    std::vector<uint8_t> want(lay.bytes, 0xCD), back(lay.linearBytes);
    CHECK(vpps4::tileLevels(lay, 0, 1, 0, linear.data(), want.data()));
    CHECK(std::memcmp(guest, want.data(), lay.bytes) == 0);
    CHECK(vpps4::detileLevels(lay, 0, 1, 0, guest, back.data()));
    CHECK(back == linear);
}

// Depth the GPU rendered is written back to the guest memory where compute or the CPU read it, in the tile mode of the
// depth buffer: Z_32_FLOAT with stencil (host D32FloatS8) as a 4-byte Z plane at the Z base and a 1-byte stencil plane at
// its own base; Z_16 and plain Z_32_FLOAT as one plane. Z_16 with stencil (held as 32-bit float on the host) is not. A plane
// is written only if the image holds all of it: after a clear of all of the plane (or a fill from memory); a plane that was
// only drawn into, cleared in part, or read at another base than it is written is not.
void depthWriteBack() {
    for (uint8_t tile : {uint8_t(3), uint8_t(5), uint8_t(0)}) {  // depth 2D thin (two tile splits), depth 1D thin
        Fixture f;
        f.renderer.draw(zState(f, 3, true, tile));
        CHECK(f.renderer.tracker().ownerImage(kDepth, 128 * 128 * 4) != 0);
        CHECK(f.renderer.tracker().ownerImage(kStencil, 128 * 128) != 0);
        f.rawWrite(kDepth + 0x100, 256);  // a raw reader of the Z plane: both planes are written back
        CHECK(f.renderer.stats().retiled == 1 && f.renderer.stats().gpuRetiled == 2);
        CHECK(f.renderer.stats().notes.count("image newer than memory cannot be written back (layout not covered): raw readers see stale bytes") == 0);
        CHECK(!f.renderer.tracker().staleAt(kDepth) && !f.renderer.tracker().staleAt(kStencil));
        f.renderer.submit();
        checkTiledPlane(f, vpgpu::Aspect::Depth, kDepth, 4, tile);
        checkTiledPlane(f, vpgpu::Aspect::Stencil, kStencil, 1, tile);
        if (failures) f.dump();
    }
    {  // drawn into without a clear: the image is not the guest's, nothing is owned or written (memory stays the truth)
        Fixture f;
        f.renderer.draw(zState(f, 3, true, 3, false));
        CHECK(f.renderer.tracker().ownerImage(kDepth, 128 * 128 * 4) == 0);
        CHECK(f.renderer.tracker().ownerImage(kStencil, 128 * 128) == 0);
        f.rawWrite(kDepth + 0x100, 256);
        CHECK(f.renderer.stats().retiled == 0 && f.renderer.stats().gpuRetiled == 0);
        CHECK(f.renderer.stats().notes.count("image newer than memory cannot be written back (layout not covered): raw readers see stale bytes") == 0);
        CHECK(f.ownedOn.empty());
    }
    {  // only Z cleared (the stencil plane was only drawn into): Z is written back alone
        Fixture f;
        vpps4::DrawState d = zState(f, 3, true, 3);
        d.depth.stencilClear = false;
        f.renderer.draw(d);
        CHECK(f.renderer.tracker().ownerImage(kDepth, 128 * 128 * 4) != 0);
        CHECK(f.renderer.tracker().ownerImage(kStencil, 128 * 128) == 0);
        f.rawWrite(kDepth, 256);
        CHECK(f.renderer.stats().retiled == 1 && f.renderer.stats().gpuRetiled == 1);
        f.renderer.submit();
        checkTiledPlane(f, vpgpu::Aspect::Depth, kDepth, 4, 3);
    }
    {  // a clear inside a scissor smaller than the image leaves the rest undefined: not written
        Fixture f;
        vpps4::DrawState d = zState(f, 3, true, 3);
        d.scissor = {0, 0, 64, 64};
        f.renderer.draw(d);
        CHECK(f.renderer.tracker().ownerImage(kDepth, 128 * 128 * 4) == 0);
        CHECK(f.renderer.tracker().ownerImage(kStencil, 128 * 128) == 0);
    }
    {  // read at one base, written at another: not followed (neither plane is written back)
        Fixture f;
        vpps4::DrawState d = zState(f, 3, true, 3);
        d.depth.zWriteAddress = kBase + 0x90000;
        d.depth.stencilWriteAddress = kBase + 0xA0000;
        f.renderer.draw(d);
        CHECK(f.renderer.tracker().ownerImage(kDepth, 128 * 128 * 4) == 0);
        CHECK(f.renderer.tracker().ownerImage(kBase + 0x90000, 128 * 128 * 4) == 0);
        CHECK(f.renderer.tracker().ownerImage(kStencil, 128 * 128) == 0);
        CHECK(f.renderer.tracker().ownerImage(kBase + 0xA0000, 128 * 128) == 0);
        f.rawWrite(kDepth, 256);
        CHECK(f.renderer.stats().retiled == 0);
    }
    {  // a write base that is the read base (or none) is the one base: written back there
        Fixture f;
        vpps4::DrawState d = zState(f, 3, true, 3);
        d.depth.zWriteAddress = kDepth;
        d.depth.stencilWriteAddress = kStencil;
        f.renderer.draw(d);
        CHECK(f.renderer.tracker().ownerImage(kDepth, 128 * 128 * 4) != 0);
        CHECK(f.renderer.tracker().ownerImage(kStencil, 128 * 128) != 0);
    }
    {  // Z_16: one 2-byte plane
        Fixture f;
        f.renderer.draw(zState(f, 1, false, 3));
        CHECK(f.renderer.tracker().ownerImage(kDepth, 128 * 128 * 2) != 0);
        f.rawWrite(kDepth, 256);
        CHECK(f.renderer.stats().retiled == 1 && f.renderer.stats().gpuRetiled == 1);
        f.renderer.submit();
        checkTiledPlane(f, vpgpu::Aspect::Depth, kDepth, 2, 3);
    }
    {  // Linear depth buffer with stencil: a texture to buffer copy per plane, rows of the plane's own texel size
        Fixture f;
        f.renderer.draw(zState(f, 3, true, 8));
        f.rawWrite(kDepth, 256);
        CHECK(f.renderer.stats().retiled == 1);
        f.renderer.submit();
        unsigned z = 0, st = 0;
        for (const vpgpu::Command& c : f.backend.commands)
            if (const auto* cp = std::get_if<vpgpu::Copy>(&c); cp && cp->label.rfind("retile ", 0) == 0) {
                const bool zp = cp->srcTexture.aspect == vpgpu::Aspect::Depth;
                CHECK(cp->label == "retile " + Fixture::hex(zp ? kDepth : kStencil));
                CHECK(cp->dstBuffer.bytesPerRow == (zp ? 128u * 4 : 128u) && cp->dstBuffer.rowsPerImage == 128);
                z += zp, st += !zp;
            }
        CHECK(z == 1 && st == 1);
    }
    {  // The destination of a depth copy draw (both planes forced) is written back at the write bases, holding what the source
       // held: here a cleared source
        Fixture f;
        f.renderer.draw(zState(f, 3, true, 3));
        vpps4::DrawState d = zState(f, 3, true, 3, false);
        d.color = {};
        d.depth.zWriteAddress = kBase + 0x90000;
        d.depth.stencilWriteAddress = kBase + 0xA0000;
        d.depth.forceZDirty = d.depth.forceZValid = d.depth.forceStencilDirty = d.depth.forceStencilValid = true;
        f.renderer.draw(d);
        CHECK(f.renderer.stats().notes.count("depth copy draw done as a texture copy") == 1);
        f.rawWrite(kBase + 0x90000, 256);
        CHECK(f.renderer.stats().retiled == 1 && f.renderer.stats().gpuRetiled == 2);
        f.renderer.submit();
        checkTiledPlane(f, vpgpu::Aspect::Depth, kBase + 0x90000, 4, 3);
        checkTiledPlane(f, vpgpu::Aspect::Stencil, kBase + 0xA0000, 1, 3);
    }
    {  // The same copy of a source that was only drawn into holds nothing the guest has: not written back
        Fixture f;
        f.renderer.draw(zState(f, 3, true, 3, false));
        vpps4::DrawState d = zState(f, 3, true, 3, false);
        d.color = {};
        d.depth.zWriteAddress = kBase + 0x90000;
        d.depth.stencilWriteAddress = kBase + 0xA0000;
        d.depth.forceZDirty = d.depth.forceZValid = d.depth.forceStencilDirty = d.depth.forceStencilValid = true;
        f.renderer.draw(d);
        CHECK(f.renderer.stats().notes.count("depth copy draw done as a texture copy") == 1);
        CHECK(f.renderer.tracker().ownerImage(kBase + 0x90000, 128 * 128 * 4) == 0);
        f.rawWrite(kBase + 0x90000, 256);
        CHECK(f.renderer.stats().retiled == 0);
    }
    {  // a depth image of two layers (stereo) cleared whole by a RectList: written (the pass's own layer count is still 1 then)
        Fixture f;
        vpps4::DrawState d = zState(f, 3, false, 3);
        d.primitive = vpps4::PrimitiveType::RectList;
        d.depth.sliceCount = 2;
        f.renderer.draw(d);
        CHECK(f.renderer.stats().handled.count("rect list clear") == 1);
        CHECK(f.renderer.tracker().ownerImage(kDepth, 128 * 128 * 4) != 0);
    }
    {  // Z_16 with stencil: guest words of 2 bytes, host texels of 4: nothing can be written, so nothing is owned
        Fixture f;
        f.renderer.draw(zState(f, 1, true, 3));
        CHECK(f.renderer.tracker().ownerImage(kDepth, 128 * 128 * 2) == 0);
        f.rawWrite(kDepth, 256);
        CHECK(f.renderer.stats().retiled == 0);
    }
}

// ---- Guest windows -----------------------------------------------------------------------------------------------

constexpr uint64_t kWinBase = 0x40000000, kMiB = 1ull << 20;

// Guest memory offered in windows the test picks (`offered`, the smallest that holds the range answers): windows that overlap,
// as the host makes them when a range crosses the end of its 64 MiB window. `epoch` says which copy of the memory the
// windows alias (a mirror the host replaced is another copy of the same pages).
struct Windows {
    static constexpr uint64_t kSpan = 192 * kMiB;
    void* mem[2] = {std::aligned_alloc(16384, kSpan), std::aligned_alloc(16384, kSpan)};
    Recording backend;
    std::vector<std::string> log;
    vpgpu::Device device{backend, 0, [this](const std::string& s) { log.push_back(s); }};
    std::vector<std::pair<uint64_t, uint64_t>> offered;  // (base, size)
    unsigned epoch = 0;
    uint64_t generation = 1;
    bool mirrored = true;
    vpps4::Renderer renderer{device, hooks()};
    Program flat = {{0x7e040200u, 0x7e060201u, 0x7e080202u, 0x7e0a0203u, 0xdc300000u, 0x0a000002u, 0xdc700000u, 0x00000a04u, 0xbf810000u}, {}};

    Windows() { device.setRefuseFailedBuffers(true); }
    ~Windows() {
        std::free(mem[0]);
        std::free(mem[1]);
    }
    vpps4::Hooks hooks() {
        vpps4::Hooks h;
        h.region = [this](uint64_t a, uint64_t s, vpps4::GuestRegion& g) {
            const std::pair<uint64_t, uint64_t>* best = nullptr;  // (the smallest window that holds the range answers)
            for (const auto& w : offered)
                if (a >= w.first && a + s <= w.first + w.second && (!best || w.second < best->second)) best = &w;
            if (!best) return false;
            g = {best->first, best->second, static_cast<uint8_t*>(mem[epoch]) + (best->first - kWinBase), mirrored};
            return true;
        };
        h.regionGeneration = [this] { return generation; };
        h.shader = [](uint64_t, vpir::Stage, const vpmsl::MslShader&) {};
        h.log = [this](const std::string& s) { log.push_back(s); };
        return h;
    }
    // The window with the memory of [base, base + size) is made: DMA fill of its first page.
    void use(uint64_t base, uint64_t size) {
        offered = {{base, size}};
        CHECK(renderer.fill(base + 4096, 4096, 0));
    }
    size_t destroyedBuffers() const {
        size_t n = 0;
        for (const std::string& c : backend.inner.calls) n += c.rfind("destroyBuffer", 0) == 0;
        return n;
    }
    // Runs the FLAT program (pointers outside every window, so that it makes none) and returns the guest map the dispatch
    // got: (base, size) of each entry.
    std::vector<std::pair<uint64_t, uint64_t>> flatTable() {
        vpps4::DispatchState d;
        d.cs.code = flat.code;
        d.cs.address = kWinBase + 190 * kMiB;
        d.cs.userSgprs = 4;
        d.cs.workgroupSize = {64, 1, 1};
        d.cs.userData[0] = uint32_t(kWinBase + 191 * kMiB);
        d.cs.userData[1] = uint32_t((kWinBase + 191 * kMiB) >> 32);
        d.cs.userData[2] = uint32_t(kWinBase + 191 * kMiB + 4096);
        d.cs.userData[3] = uint32_t((kWinBase + 191 * kMiB + 4096) >> 32);
        const size_t before = backend.commands.size();
        renderer.dispatch(d);
        renderer.dispatch(d);  // (the pipeline compiles on the first one on the null backend)
        renderer.submit();
        std::vector<std::pair<uint64_t, uint64_t>> out;
        for (size_t i = before; i < backend.commands.size(); ++i) {
            const auto* dp = std::get_if<vpgpu::Dispatch>(&backend.commands[i]);
            if (!dp) continue;
            for (const vpgpu::Access& a : dp->bindings) {
                if (a.kind != vpgpu::BindKind::ConstantBuffer || a.range.end - a.range.begin != 16 + 32 * 256) continue;
                const uint32_t* t = reinterpret_cast<const uint32_t*>(backend.memory.at(a.range.index) + a.range.begin);
                out.clear();
                for (uint32_t k = 0; k < t[0]; ++k) {
                    const uint32_t* e = t + 4 + 8 * k;
                    out.push_back({e[0] | uint64_t(e[1]) << 32, e[2] | uint64_t(e[3]) << 32});
                }
            }
        }
        return out;
    }
};

// The FLAT shader looks the pointer up with a binary search for the last base <= the pointer: the table it gets must be
// sorted by base and must not hold a window inside another (it would hide the end of the one around it).
void flatMap() {
    Windows w;
    w.mirrored = false;
    w.use(kWinBase, 128 * kMiB);              // a window over a boundary
    w.use(kWinBase + 16 * kMiB, 16 * kMiB);   // a piece inside it
    w.use(kWinBase + 100 * kMiB, 40 * kMiB);  // one that overlaps its end
    w.offered = {{kWinBase, 128 * kMiB}};
    const auto table = w.flatTable();
    CHECK(table.size() == 2);
    if (table.size() == 2) {
        CHECK(table[0] == std::make_pair(kWinBase, 128 * kMiB));
        CHECK(table[1] == std::make_pair(kWinBase + 100 * kMiB, 40 * kMiB));
    }
    for (size_t k = 1; k < table.size(); ++k) CHECK(table[k - 1].first < table[k].first);
}

// A window over a mirror the host has replaced is not left behind: made again over the new one it retires the overlapping
// windows made of the old copy of the pages (the buffer is destroyed once the frames in flight are done), and one that
// does not overlap it is left out of the FLAT map and retired there. Windows of the same copy stay.
void windowSuperseded() {
    Windows w;
    w.use(kWinBase, 128 * kMiB);
    ++w.generation;  // (the host's answers are asked for again)
    w.use(kWinBase, 64 * kMiB);  // the same pages, another extent: not a different copy
    CHECK(w.destroyedBuffers() == 0);
    w.epoch = 1;  // the host mapped the pages again
    ++w.generation;
    w.offered = {{kWinBase, 128 * kMiB}, {kWinBase, 64 * kMiB}, {kWinBase + 64 * kMiB, 64 * kMiB}};
    CHECK(w.renderer.fill(kWinBase + 64 * kMiB + 4096, 4096, 0));
    // (retired buffers are destroyed by submits that carry work)
    const auto churn = [&] {
        for (int i = 0; i < 12; ++i) {
            w.renderer.fill(kWinBase + 64 * kMiB + 8192, 16, 0);
            w.renderer.submit();
        }
    };
    churn();
    CHECK(w.destroyedBuffers() == 1);  // the 128 MiB window made of the old copy overlaps the new one
    const auto table = w.flatTable();  // (the 64 MiB one, which does not overlap it, is found out of date here)
    CHECK(table.size() == 1 && table[0] == std::make_pair(kWinBase + 64 * kMiB, 64 * kMiB));
    churn();
    CHECK(w.destroyedBuffers() == 2);
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
    // The write back is only recorded: the pages stay protected until the GPU has run it.
    CHECK(f.ownedOff.empty());
    CHECK(!f.renderer.syncToMemory(kRt + 4, 16));  // memory is current now
    f.renderer.submit();
    f.renderer.waitedBelow(f.device.completedFrames());  // (submitted, not finished)
    CHECK(f.ownedOff.empty());
    f.backend.inner.finishFramesBelow(f.device.submittedFrames());
    f.renderer.waitedBelow(f.device.completedFrames());
    CHECK(f.ownedOff.size() == 1);
    CHECK(f.renderer.stats().syncs == 1);
    const auto e = f.events();
    CHECK(Fixture::indexOf(e, "copy retile " + Fixture::hex(kRt)) >= 0);
    // A region that is not mirrored is never watched.
    Fixture g(false);
    g.renderer.draw(g.state());
    CHECK(g.watchedWrites.empty() && g.ownedOn.empty());
}

// The pages an image's write back gives to the CPU are released when the GPU has run the re-tile,
// not when it is recorded; an image written again meanwhile takes its release back; a lost device
// lets go of them.
void ownedRelease() {
    {
        Fixture f(true);
        f.renderer.draw(f.state());
        CHECK(f.ownedOn.size() == 1);
        // A raw read of the render target's memory: the image is written back, recorded only.
        f.renderer.noteGpuWrite(kRt, 16);
        CHECK(f.ownedOff.empty());
        f.renderer.submit();
        CHECK(f.ownedOff.empty());  // submitted is not finished
        f.backend.inner.finishFramesBelow(f.device.submittedFrames());
        f.renderer.waitedBelow(f.device.completedFrames());
        CHECK(f.ownedOff.size() == 1 && f.ownedOff[0] == f.ownedOn[0]);
    }
    {
        Fixture f(true);
        f.renderer.draw(f.state());
        f.renderer.noteGpuWrite(kRt, 16);  // released later
        f.renderer.submit();
        f.renderer.draw(f.state());        // the image is written again: owns the range again
        CHECK(f.ownedOn.size() == 1);      // (the host's reference was never given back: no second request)
        f.backend.inner.finishFramesBelow(f.device.submittedFrames());
        f.renderer.waitedBelow(f.device.completedFrames());
        CHECK(f.ownedOff.empty());         // and the retired release is gone
    }
    {
        Fixture f(true);
        f.renderer.draw(f.state());
        f.renderer.noteGpuWrite(kRt, 16);
        f.renderer.submit();
        f.backend.inner.isLost = true;
        f.device.waitFrames(f.device.submittedFrames());  // Lost: no frame will ever finish
        f.renderer.waitedBelow(0);
        CHECK(f.ownedOff.size() == 1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string_view t = argc > 1 ? argv[1] : "";
    if (t == "rt_to_texture") rtToTexture();
    else if (t == "sample_extent") sampleExtent();
    else if (t == "soft_bc_upload") softBcUpload();
    else if (t == "soft_bc_tiled") softBcTiled();
    else if (t == "soft_bc_gpu") softBcGpu();
    else if (t == "soft_bc_budget") softBcBudget();
    else if (t == "soft_bc_kernel_wait") softBcKernelWait();
    else if (t == "soft_bc_staging_limit") softBcStagingLimit();
    else if (t == "color_mirror_refill") colorMirrorRefill();
    else if (t == "mirror_unwritten") mirrorUnwritten();
    else if (t == "mirror_refilled") mirrorRefilled();
    else if (t == "sync_refused_fill") syncRefusedFill();
    else if (t == "depth_mirror") depthMirror();
    else if (t == "msaa_mirror") msaaMirror();
    else if (t == "channel_order") channelOrder();
    else if (t == "raw_write_to_texture") rawWriteToTexture();
    else if (t == "raw_write_to_layer") rawWriteToLayer();
    else if (t == "texture_to_raw") textureToRaw();
    else if (t == "dma_to_texture") dmaToTexture();
    else if (t == "cpu_write") cpuWrite();
    else if (t == "alias_formats") aliasFormats();
    else if (t == "base_level_view") baseLevelView();
    else if (t == "alias_targets") aliasTargets();
    else if (t == "raw_depth_clear") rawDepthClear();
    else if (t == "tiled_retile") tiledRetile();
    else if (t == "depth_write_back") depthWriteBack();
    else if (t == "flat_map") flatMap();
    else if (t == "window_superseded") windowSuperseded();
    else if (t == "eviction") eviction();
    else if (t == "host_watches") hostWatches();
    else if (t == "owned_release") ownedRelease();
    else return std::fprintf(stderr, "unknown test\n"), 2;
    return failures ? 1 : 0;
}
