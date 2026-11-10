// Memory coherence on the NullBackend: the same guest bytes written as a render target, depth
// buffer, storage image, raw buffer or by DMA, and read as any of them. Frames are captured so the
// ORDER of dispatches, draws and copies on the GPU timeline can be asserted.
// Usage: vpps4_coherence_tests TEST
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "vpgpu/core/device.h"
#include "vpgpu/core/probe.h"
#include "vpps4/renderer.h"
#include "vpps4/tiling.h"
#include "../src/frame_capture.h"

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
    std::vector<vpgpu::CompiledFrame> compiled;  // what the frame compiler made of each frame given
    std::map<uint32_t, uint8_t*> memory;  // host memory of the buffers that alias some (guest windows, upload rings), by index

    std::map<uint32_t, vpgpu::BufferDesc> bufferDescs;
    void createBuffer(vpgpu::BufferId i, const vpgpu::BufferDesc& d) override {
        bufferDescs[i.index] = d;
        if (d.guestMemory) memory[i.index] = static_cast<uint8_t*>(d.guestMemory);
        inner.createBuffer(i, d);
    }
    void createTexture(vpgpu::TextureId i, const vpgpu::TextureDesc& d) override {
        textureDescs[i.index] = d;
        inner.createTexture(i, d);
    }
    // readTexture plays the last upload into the subresource from a buffer with host memory (TEXVERIFY tests); `corrupt`
    // flips a byte of what it reads.
    std::map<uint32_t, vpgpu::TextureDesc> textureDescs;
    bool corrupt = false;
    uint32_t readDivisor = 1;  // the readback comes back this many times shorter (a size the format does not explain)
    vpgpu::ReadResult readTexture(const vpgpu::TextureSpan& s, std::vector<uint8_t>& out) override {
        for (auto it = commands.rbegin(); it != commands.rend(); ++it) {
            const auto* c = std::get_if<vpgpu::Copy>(&*it);
            if (!c || c->kind != vpgpu::CopyKind::BufferToTexture || !(c->dstTexture.texture == s.texture) || c->dstTexture.mip != s.mip)
                continue;
            const auto m = memory.find(c->srcBuffer.buffer.index);
            if (m == memory.end()) return vpgpu::ReadResult::Unsupported;
            const uint32_t bpe = vpgpu::formatInfo(textureDescs[s.texture.index].format).bytesPerBlock;
            const uint8_t* src = m->second + c->srcBuffer.offset + uint64_t(s.layer) * c->srcBuffer.rowsPerImage * c->srcBuffer.bytesPerRow;
            out.clear();
            for (uint32_t y = 0; y < s.height; ++y)
                out.insert(out.end(), src + uint64_t(y) * c->srcBuffer.bytesPerRow, src + uint64_t(y) * c->srcBuffer.bytesPerRow + uint64_t(s.width) * bpe);
            if (corrupt && !out.empty()) out[out.size() / 2] ^= 0xFF;
            if (readDivisor > 1) out.resize(out.size() / readDivisor);
            return vpgpu::ReadResult::Ok;
        }
        return vpgpu::ReadResult::Failed;
    }
    void createSampler(vpgpu::SamplerId i, const vpgpu::SamplerDesc& d) override { inner.createSampler(i, d); }
    void destroyBuffer(vpgpu::BufferId i) override { inner.destroyBuffer(i); }
    void destroyTexture(vpgpu::TextureId i) override { inner.destroyTexture(i); }
    void destroySampler(vpgpu::SamplerId i) override { inner.destroySampler(i); }
    uint64_t bufferAddress(vpgpu::BufferId i) const override { return inner.bufferAddress(i); }
    static inline bool refuseBcKernel = false;  // the block decoding kernel does not compile (the CPU decoder takes over)
    static inline std::atomic<bool> holdBcKernel{false};  // ... is still compiling (on a compile worker, see Fixture::workers)
    std::vector<vpgpu::PipelineState> pipelines;  // every state compiled
    bool compilePipeline(const vpgpu::PipelineState& s, std::string& e) override {
        pipelines.push_back(s);
        if (refuseBcKernel && s.label == "bcdecode") return e = "refused", false;
        while (holdBcKernel && s.label == "bcdecode") std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return inner.compilePipeline(s, e);
    }
    void destroyPipeline(vpgpu::PipelineKey k) override { inner.destroyPipeline(k); }
    void submit(uint64_t n, const vpgpu::Frame& f, const vpgpu::CompiledFrame& c) override {
        commands.insert(commands.end(), f.commands().begin(), f.commands().end());
        compiled.push_back(c);
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
    static inline std::function<void(Fixture&)> hostSubmit;  // Hooks::submit of the fixtures made while it is set
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
        if (hostSubmit) h.submit = [this, host = hostSubmit] { host(*this); };
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

// A depth surface drawn at the title's smaller size (the SetDepthRenderTarget hint of a dynamic resolution: 64x64) inside its
// allocation (DB_DEPTH_SIZE pitch and height: 256x256): guest memory is laid out at the allocation, as a T# of the whole surface
// reads it (shadow cascades sampled as a 1024x1024 array, drawn 512x512). The image owns the whole allocation and the retile
// tiles the 64x64 picture with the allocation's pitch and rows; at the hinted size the rows land at other offsets and the
// slices inside the first one. VPKENGINE_DEPTH_LAYOUT_HINT=1: the hinted size (before).
void depthHintLayout() {
    for (const bool old : {false, true}) {
        if (old) setenv("VPKENGINE_DEPTH_LAYOUT_HINT", "1", 1);
        Fixture f;
        vpps4::DrawState d = zState(f, 3, false, 3);  // Z_32_FLOAT, depth 2D thin
        d.depth.width = d.depth.height = 64;
        d.depth.pitch = d.depth.rows = 256;
        f.renderer.draw(d);
        vpps4::TiledSurface full;
        full.tileIndex = 3, full.bytesPerElement = 4, full.width = full.height = 64, full.pitch = 256, full.rows = 256;
        vpps4::TiledSurface hinted = full;
        hinted.pitch = hinted.rows = 0;
        const vpps4::SurfaceLayout want = vpps4::computeLayout(old ? hinted : full);
        CHECK(want.ok && vpps4::computeLayout(full).levels[0].pitch == 256 && vpps4::computeLayout(full).levels[0].paddedHeight == 256);
        CHECK(vpps4::computeLayout(hinted).levels[0].pitch < 256);  // (the two layouts differ)
        const uint64_t bytes = 256 * 256 * 4;
        CHECK((f.renderer.tracker().ownerImage(kDepth + bytes / 2, bytes / 2) != 0) == !old);
        f.rawWrite(kDepth + 0x100, 256);  // a raw reader of the Z plane: written back
        CHECK(f.renderer.stats().gpuRetiled == 1);
        f.renderer.submit();
        unsigned ran = 0;
        for (const vpgpu::Command& c : f.backend.commands) {
            const auto* dp = std::get_if<vpgpu::Dispatch>(&c);
            if (!dp || dp->label.rfind("gpu retile", 0) != 0) continue;
            for (const vpgpu::Access& a : dp->bindings) {
                if (a.kind != vpgpu::BindKind::ConstantBuffer || a.slot != 0) continue;
                VptParams P;
                std::memcpy(&P, f.backend.memory.at(a.range.index) + a.range.begin, sizeof P);
                CHECK(P.level.pitch == want.levels[0].pitch && P.level.height == want.levels[0].paddedHeight);
                CHECK(P.width == 64 && P.height == 64);
                ++ran;
            }
        }
        CHECK(ran == 1);
        if (old) unsetenv("VPKENGINE_DEPTH_LAYOUT_HINT");
        if (failures) f.dump();
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

// Guest windows are made resident by the work that names them (BufferDesc::residentByUse: gpu/metal2 keeps them out of the
// residency set): a FLAT dispatch, which may follow a pointer into any window of its GuestMap, names every one of them in
// its bindings (address-only accesses), so each is declared to its encoder.
void flatWindowsNamed() {
    Windows w;
    w.mirrored = false;
    w.use(kWinBase, 64 * kMiB);
    w.use(kWinBase + 64 * kMiB, 64 * kMiB);
    w.offered = {{kWinBase, 64 * kMiB}, {kWinBase + 64 * kMiB, 64 * kMiB}};
    const size_t before = w.backend.commands.size();
    CHECK(w.flatTable().size() == 2);
    std::set<uint32_t> windows, named;
    for (const auto& [index, d] : w.backend.bufferDescs)
        if (d.label.rfind("guest ", 0) == 0) {
            CHECK(d.residentByUse);
            windows.insert(index);
        }
    for (size_t i = before; i < w.backend.commands.size(); ++i)
        if (const auto* dp = std::get_if<vpgpu::Dispatch>(&w.backend.commands[i]))
            for (const vpgpu::Access& a : dp->bindings)
                if (a.range.kind == vpgpu::ResourceRange::Kind::Buffer && windows.count(a.range.index)) named.insert(a.range.index);
    CHECK(windows.size() == 2 && named == windows);
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

// The byte budget counts what the GPU holds for an image: its whole mip chain, and for a multisampled colour target the
// single-sample picture next to the samples. Whatever the last submit used is never evicted, whatever the budget.
void evictionBytes() {
    // A 128x128 target (64 KiB) and a 64x64 texture of 7 levels (21844 B with its chain, 16 KiB without): 87380 B in all.
    Fixture f;
    f.renderer.setResidency(1ull << 40, 85000);
    vpps4::DrawState d = f.state();
    f.at(kBase + 0x200)[3] |= 6u << 16;  // T# last level 6
    f.renderer.draw(d);
    for (int i = 0; i < 20; ++i) f.renderer.submit();
    CHECK(f.renderer.stats().evicted == 1);
    // The same target with 4 samples: 256 KiB of samples and 64 KiB of resolved picture, with the 16 KiB texture.
    Fixture g;
    g.renderer.setMaxSamples(4);
    g.renderer.setResidency(1ull << 40, 340000);
    vpps4::DrawState m = g.state();
    m.color[0].samples = 4;
    g.renderer.draw(m);
    for (int i = 0; i < 20; ++i) g.renderer.submit();
    CHECK(g.renderer.stats().evicted == 1);
    // Used in the submit that evicts: kept (the eviction runs in the submit of frame 15).
    Fixture h;
    h.renderer.setResidency(0, 0);
    for (int i = 0; i < 15; ++i) h.renderer.submit();
    h.renderer.draw(h.state());
    h.renderer.submit();
    CHECK(h.renderer.stats().evicted == 0);
    CHECK(bool(h.renderer.colorTargetAt(kRt)));
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

// ---- Aliases of images the GPU wrote ---------------------------------------------------------------------------

// A T# of the same memory a render target / storage image wrote last, with the same layout and texel size but another numeric
// class (integers over unorm), is read through a view of that image: the target is not written back to memory and the
// texture filled from it at every use (a retile, a detile and a GPU wait per use). Whatever differs, or anything else writes
// the range, keeps the way through memory.

constexpr uint64_t kOut = kBase + 0xB0000;  // the target the reading draws write (128x128 RGBA8)

struct ProbeOn {  // the QUAL counters are published only while the probe is on (the registry is not reset: the renderers keep references into it)
    ProbeOn() { vpgpu::probe::setEnabled(true); }
    ~ProbeOn() { vpgpu::probe::setEnabled(false); }
};
uint64_t qual(Fixture& f, const std::string& name) {
    f.renderer.publishProbe(1);
    return vpgpu::probe::counter("QUAL", name).get();
}
uint64_t rejected(Fixture& f) {
    uint64_t n = 0;
    for (const char* why : {"depth", "address", "shape", "layout", "mips", "samples", "format", "no_view", "range", "feedback"})
        n += qual(f, std::string("alias_rejected_") + why);
    return n;
}

// One frame: a draw into kRt (tile mode `tile`), then a draw that reads memory `at` through a T# of `dfmt`/`nfmt`, `w` x `h`,
// tiling `ttile` (0xFF: `tile`), word 3 or-ed with `w3`, and writes kOut.
void writeThenRead(Fixture& f, uint32_t dfmt, uint32_t nfmt, uint8_t tile = 8, uint32_t w = 128, uint32_t h = 128, uint8_t ttile = 0xFF,
                   uint32_t w3 = 0, uint64_t at = kRt) {
    vpps4::DrawState a = f.state();
    a.color[0].tileMode = tile;
    f.renderer.draw(a);
    vpps4::DrawState b = f.state();
    b.color[0].address = kOut;
    b.color[0].tileMode = tile;
    f.sample(at, w, h, dfmt, nfmt, ttile == 0xFF ? tile : ttile);
    f.at(kBase + 0x200)[3] |= w3;
    f.renderer.draw(b);
}
vpps4::ImageDesc sampled(Fixture& f) { return vpps4::decodeImage(std::span<const uint32_t>(f.at(kBase + 0x200), 8)); }

// The views the recorded draws bind.
std::vector<vpgpu::TextureId> boundViews(Fixture& f) {
    f.renderer.submit();
    std::vector<vpgpu::TextureId> out;
    for (const vpgpu::Command& c : f.backend.commands)
        if (const auto* p = std::get_if<vpgpu::RenderPass>(&c))
            for (const vpgpu::Draw& d : p->draws)
                for (const vpgpu::ViewBinding& v : d.views) out.push_back(v.view);
    return out;
}

// A target drawn and then read as integers over several frames: the image is read through a view, with nothing copied
// after the first frame; the fill of the target never happens (no retile, no detile of it).
void aliasClassChange(uint8_t tile) {
    ProbeOn probe;
    Fixture f;
    f.withViews = true;
    writeThenRead(f, 10, 4, tile);  // RGBA8 uint over the RGBA8 unorm target
    const auto rt = f.renderer.colorTargetAt(kRt);
    const auto& st = f.renderer.stats();
    CHECK(bool(rt) && st.drawn == 2);
    CHECK(st.retiled == 0 && st.refilled == 1);  // (the refill is the first fill of the draw's own texture at kTex)
    const uint64_t copies = st.copies;
    for (int i = 0; i < 5; ++i) {
        writeThenRead(f, 10, 4, tile);
        CHECK(st.retiled == 0 && st.refilled == 1 && st.copies == copies);
    }
    CHECK(st.drawn == 12);
    CHECK(f.viewRequests.size() == 1 && f.viewRequests[0].base == rt && f.viewRequests[0].format == vpgpu::Format::RGBA8Uint);
    const vpgpu::TextureId view = f.renderer.textureFor(sampled(f));
    CHECK(view && view != rt);
    CHECK(f.renderer.tracker().ownerImage(kRt, 128 * 128 * 4) != 0);  // (memory is still behind the target: nothing wrote it back)
    CHECK(qual(f, "alias_hits") == 7);
    CHECK(rejected(f) == 0);
    const auto views = boundViews(f);
    CHECK(views.size() == 6);
    for (const vpgpu::TextureId v : views) CHECK(v == view);
    CHECK(Fixture::count(f.events(), "copy retile") == 0);
    if (failures) f.dump();
}
void aliasClassChangeLinear() { aliasClassChange(8); }
void aliasClassChangeTiled() { aliasClassChange(14); }

// The same through a storage image a dispatch wrote (instead of a target).
void aliasTextureWriter() {
    ProbeOn probe;
    Fixture f;
    f.withViews = true;
    f.imageWrite();
    f.sample(kImg, 64, 64, 10, 0, 8);
    const auto image = f.renderer.textureFor(sampled(f));  // the storage image itself
    const uint64_t textures = f.renderer.stats().textures;
    for (int i = 0; i < 4; ++i) {
        f.imageWrite();
        vpps4::DrawState d = f.state();
        f.sample(kImg, 64, 64, 10, 4, 8);  // uint over the unorm storage image
        f.renderer.draw(d);
    }
    CHECK(f.renderer.stats().drawn == 4);
    CHECK(f.renderer.stats().retiled == 0);
    CHECK(f.renderer.stats().textures == textures);  // (no texture at kImg but the storage image: the sampling draws read it through a view)
    CHECK(f.viewRequests.size() == 1 && f.viewRequests[0].base == image && f.viewRequests[0].format == vpgpu::Format::RGBA8Uint);
    CHECK(qual(f, "alias_hits") == 4 && rejected(f) == 0);
    if (failures) f.dump();
}

// A render target read in the pass that writes it is not aliased (it is attached to the pass): the way through memory as
// before, counted as `feedback`. Once no pass of the draw has it attached, the alias is back with nothing written back.
void aliasSamePass() {
    ProbeOn probe;
    Fixture f;
    f.withViews = true;
    f.renderer.draw(f.state());
    for (int i = 0; i < 3; ++i) {
        vpps4::DrawState d = f.state();
        f.sample(kRt, 128, 128, 10, 4, 8);
        f.renderer.draw(d);  // writes kRt and reads it as integers
    }
    const auto& st = f.renderer.stats();
    CHECK(st.drawn == 4 && st.retiled >= 1);
    CHECK(qual(f, "alias_hits") == 0 && qual(f, "alias_rejected_feedback") == 3 && rejected(f) == 3);
    const uint64_t retiles = st.retiled, refills = st.refilled;
    vpps4::DrawState d = f.state();
    d.color[0].address = kOut;
    f.sample(kRt, 128, 128, 10, 4, 8);
    f.renderer.draw(d);  // reads kRt, writes another target
    CHECK(qual(f, "alias_hits") == 1 && rejected(f) == 3);
    CHECK(st.retiled == retiles && st.refilled == refills);
    if (failures) f.dump();
}

// A BGRA target read through an RGBA format (or the reverse): not aliased, the channel order is not compensated.
void aliasChannelOrder() {
    ProbeOn probe;
    Fixture f;
    f.withViews = true;
    vpps4::DrawState a = f.state();
    a.color[0].compSwap = 1;  // BGRA
    f.renderer.draw(a);
    vpps4::DrawState b = f.state();
    b.color[0].address = kOut;
    f.sample(kRt, 128, 128, 10, 4, 8);
    f.renderer.draw(b);
    CHECK(qual(f, "alias_hits") == 0 && qual(f, "alias_rejected_format") == 1);
    CHECK(f.renderer.stats().retiled == 1);
}

// The CPU writes the target's memory (after the host had it written back), or a DMA does: no alias from then on; the texture is
// filled from memory, with the new bytes, once, and read from then on; where the target is drawn again the alias is back.
void aliasWritten(bool dma, uint64_t offset) {
    ProbeOn probe;
    Fixture f;
    f.withViews = true;
    for (int i = 0; i < 3; ++i) writeThenRead(f, 10, 4, 8);
    const auto& st = f.renderer.stats();
    CHECK(st.retiled == 0 && st.refilled == 1);
    const vpgpu::TextureId aliased = f.renderer.textureFor(sampled(f));
    if (dma) {
        CHECK(f.renderer.fill(kRt + offset, 16, 0xCAFEF00Du));
    } else {
        CHECK(f.renderer.syncToMemory(kRt, 128 * 128 * 4));
        f.at(kRt + offset)[0] = 0xCAFEF00Du;
        f.renderer.invalidate(kRt + offset, 4);
    }
    const vpgpu::TextureId after = f.renderer.textureFor(sampled(f));
    CHECK(after && after != aliased);             // a texture of its own now
    CHECK(st.refilled == 2 && st.retiled == 1);  // the target written back (once), the texture filled
    CHECK(offset == 0 ? qual(f, "alias_rejected_range") == 0 : qual(f, "alias_rejected_range") == 1);
    const auto e = f.events();
    const long retile = Fixture::indexOf(e, "copy retile " + Fixture::hex(kRt));
    const long upload = Fixture::indexOf(e, "copy upload " + Fixture::hex(kRt));
    CHECK(retile >= 0 && upload > retile);
    const vpgpu::Copy* cp = decodeCopy(f, "upload " + Fixture::hex(kRt));
    const uint8_t* bytes = cp ? copyBytes(f, *cp) : nullptr;
    if (!dma) CHECK(bytes && *reinterpret_cast<const uint32_t*>(bytes + offset) == 0xCAFEF00Du);  // (the new bytes are what it reads)
    CHECK(f.renderer.textureFor(sampled(f)) == after && st.refilled == 2 && st.retiled == 1);  // read again: the texture, nothing more
    writeThenRead(f, 10, 4, 8);  // the target drawn again (filled from the CPU's bytes first, as any target): the alias is back
    CHECK(f.renderer.textureFor(sampled(f)) == aliased);
    const uint64_t refills = st.refilled, retiles = st.retiled;
    writeThenRead(f, 10, 4, 8);
    CHECK(st.refilled == refills && st.retiled == retiles);
    if (failures) f.dump();
}
void aliasCpuWriteFirstByte() { aliasWritten(false, 0); }
void aliasCpuWriteInside() { aliasWritten(false, 0x8000); }
void aliasDmaWriteFirstByte() { aliasWritten(true, 0); }
void aliasDmaWriteInside() { aliasWritten(true, 0x8000); }

// What differs keeps the way through memory (the target written back, the texture filled), and says why.
void aliasRejects() {
    struct Case {
        const char* name;
        const char* why;
        std::function<void(Fixture&)> run;
    };
    const Case cases[] = {
        {"texel size", "format", [](Fixture& f) { writeThenRead(f, 3, 0); }},                                  // 8_8 over RGBA8
        {"size", "shape", [](Fixture& f) { writeThenRead(f, 10, 4, 8, 256, 128); }},                           // wider than the target
        {"tiling", "layout", [](Fixture& f) { writeThenRead(f, 10, 4, 8, 128, 128, 13); }},                    // another tile mode
        {"levels", "mips", [](Fixture& f) { writeThenRead(f, 10, 4, 8, 128, 128, 0xFF, 3u << 16); }},          // 4 levels of a 1 level image
        {"address", "address", [](Fixture& f) { writeThenRead(f, 10, 4, 8, 64, 64, 0xFF, 0, kRt + 0x4000); }},  // inside the target
        {"no views", "no_view", [](Fixture& f) { f.withViews = false, writeThenRead(f, 10, 4); }},
    };
    for (const Case& c : cases) {
        ProbeOn probe;
        Fixture f;
        f.withViews = true;
        c.run(f);
        const auto& st = f.renderer.stats();
        if (std::string(c.why) != "address") CHECK(st.retiled == 1);  // (written back, as before the alias)
        CHECK(qual(f, "alias_hits") == 0);
        if (qual(f, std::string("alias_rejected_") + c.why) < 1) std::fprintf(stderr, "case %s: no alias_rejected_%s\n", c.name, c.why);
        CHECK(qual(f, std::string("alias_rejected_") + c.why) >= 1);
        if (failures) f.dump();
    }
    {  // a depth target (sampled larger than itself): depth targets are not aliased from here
        ProbeOn probe;
        Fixture f;
        vpps4::DrawState z = f.depthState(true);
        z.depth.depthClear = true;
        f.renderer.draw(z);
        vpps4::DrawState d = f.state();
        f.sample(kDepth, 256, 128, 4, 7, 8);
        f.renderer.draw(d);
        CHECK(qual(f, "alias_rejected_depth") == 1 && qual(f, "alias_hits") == 0);
    }
    {  // a multisampled target (sampled larger than itself)
        ProbeOn probe;
        Fixture f;
        f.withViews = true;
        f.renderer.setMaxSamples(4);
        vpps4::DrawState m = f.state();
        m.color[0].samples = 4;
        f.renderer.draw(m);
        vpps4::DrawState d = f.state();
        d.color[0].address = kOut;
        f.sample(kRt, 256, 128, 10, 4, 8);
        f.renderer.draw(d);
        CHECK(qual(f, "alias_rejected_samples") == 1 && qual(f, "alias_hits") == 0);
    }
}

// Turned off, the way through memory as before: the target written back at the first read, the texture filled from it.
void aliasOff() {
    ProbeOn probe;
    Fixture f;
    f.withViews = true;
    f.renderer.setImageAlias(false);
    for (int i = 0; i < 3; ++i) writeThenRead(f, 10, 4, 8);
    CHECK(f.renderer.stats().retiled == 3 && f.renderer.stats().refilled == 4);
    CHECK(qual(f, "alias_hits") == 0 && rejected(f) == 0);
    CHECK(f.viewRequests.empty());
}

// The target is evicted while its memory is read through the alias: written back first, then the texture comes from memory.
void aliasEvicted() {
    ProbeOn probe;
    Fixture f;
    f.withViews = true;
    f.renderer.setResidency(3, 1ull << 40);
    writeThenRead(f, 10, 4, 8);
    CHECK(f.renderer.stats().retiled == 0 && qual(f, "alias_hits") == 1);
    for (int i = 0; i < 24; ++i) f.renderer.submit();
    CHECK(!bool(f.renderer.colorTargetAt(kRt)));
    CHECK(f.renderer.stats().retiled >= 1);  // (the target, written back when evicted)
    const uint64_t textures = f.renderer.stats().textures;
    vpps4::DrawState d = f.state();
    f.sample(kRt, 128, 128, 10, 4, 8);
    f.renderer.draw(d);  // reads the memory of the target that is gone
    CHECK(f.renderer.stats().textures > textures);
    CHECK(qual(f, "alias_hits") == 1);
    const auto e = f.events();
    const long retile = Fixture::indexOf(e, "copy retile " + Fixture::hex(kRt));
    const long upload = Fixture::indexOf(e, "copy upload " + Fixture::hex(kRt));
    CHECK(retile >= 0 && upload > retile);
    if (failures) f.dump();
}

// ---- A small target in the memory of a bigger one (the free half of a dynamic resolution target)

constexpr uint64_t kSub = kRt + 0x8000;  // 64x64 RGBA8 in the second half of the 128x128 target at kRt

// One frame: the big target drawn, the small one drawn, then each sampled by a draw into kOut.
void bigAndSmallFrame(Fixture& f) {
    f.renderer.draw(f.state());
    vpps4::DrawState s = f.state();
    s.color[0] = {kSub, 64, 64, 64, 10, 0, 0, 8};
    f.renderer.draw(s);
    vpps4::DrawState d = f.state();
    d.color[0].address = kOut;
    f.sample(kRt, 128, 128, 10, 0, 8);
    f.renderer.draw(d);
    f.sample(kSub, 64, 64, 10, 0, 8);
    f.renderer.draw(d);
}

// Images of the renderer wrote each other's ranges: none is written back or filled from memory because of it (the small
// one keeps its texels, the big one keeps the rest of its own). A CPU / raw writer of the range still refills as always, and
// VPKENGINE_R1_ROUNDTRIP=1 brings the round trip back.
void bigAndSmallTargets() {
    {
        ProbeOn probe;
        Fixture f;
        bigAndSmallFrame(f);
        const auto& st = f.renderer.stats();
        const uint64_t retiled = st.retiled, refilled = st.refilled, copies = st.copies;
        for (int i = 0; i < 3; ++i) bigAndSmallFrame(f);
        CHECK(st.retiled == retiled && st.refilled == refilled && st.copies == copies);
        CHECK(st.retiled == 0);
        const auto e = f.events();
        CHECK(Fixture::count(e, "copy retile ") == 0);
        CHECK(Fixture::count(e, "copy upload " + Fixture::hex(kRt)) == 0 && Fixture::count(e, "copy upload " + Fixture::hex(kSub)) == 0);
        CHECK(f.renderer.tracker().ownerImage(kSub, 64 * 64 * 4) != 0);  // the images still hold the newest bytes
        CHECK(qual(f, "refill_skipped") >= 1 && qual(f, "refill_skipped_bytes") >= 64 * 64 * 4);
        bool line = false;  // (the REFILL line names the writer: the big target, its base and its scissor)
        for (const std::string& l : f.log)
            if (l.rfind("VPKENGINE_PROBE REFILL ", 0) == 0 && l.size() <= 2048 && l.find("skipped=1") != std::string::npos &&
                l.find("addr=" + Fixture::hex(kRt) + " bytes=65536") != std::string::npos && l.find("scissor=0,0,128,128") != std::string::npos)
                line = true;
        CHECK(line);
        if (failures) f.dump();
    }
    {  // a raw write over the small target: memory is the truth there
        Fixture f;
        bigAndSmallFrame(f);
        f.rawWrite(kSub, 16);
        const uint64_t refilled = f.renderer.stats().refilled;
        vpps4::DrawState s = f.state();
        s.color[0] = {kSub, 64, 64, 64, 10, 0, 0, 8};
        f.renderer.draw(s);
        CHECK(f.renderer.stats().refilled == refilled + 1);
        if (failures) f.dump();
    }
    {  // turned off: the round trip as before
        setenv("VPKENGINE_R1_ROUNDTRIP", "1", 1);
        Fixture f;
        unsetenv("VPKENGINE_R1_ROUNDTRIP");
        bigAndSmallFrame(f);
        const auto& st = f.renderer.stats();
        const uint64_t retiled = st.retiled, refilled = st.refilled;
        bigAndSmallFrame(f);
        CHECK(st.retiled > retiled && st.refilled > refilled);
        if (failures) f.dump();
    }
}

// The small target drawn first, then the big one over its range (twice), then the small one sampled: nothing is written back or filled
// from memory (the small one keeps its texels, the big one's bytes are not overwritten with the small one's), and the big one stays the
// owner of the shared range.
void smallThenBigTargets() {
    Fixture f;
    vpps4::DrawState s = f.state();
    s.color[0] = {kSub, 64, 64, 64, 10, 0, 0, 8};
    f.renderer.draw(s);
    f.renderer.draw(f.state());
    f.renderer.draw(f.state());
    vpps4::DrawState d = f.state();
    d.color[0].address = kOut;
    f.sample(kSub, 64, 64, 10, 0, 8);
    f.renderer.draw(d);
    f.sample(kRt, 128, 128, 10, 0, 8);
    f.renderer.draw(d);
    const auto& st = f.renderer.stats();
    CHECK(st.retiled == 0);
    const auto e = f.events();
    CHECK(Fixture::count(e, "copy retile ") == 0);
    CHECK(Fixture::count(e, "copy upload " + Fixture::hex(kRt)) == 0 && Fixture::count(e, "copy upload " + Fixture::hex(kSub)) == 0);
    CHECK(f.renderer.tracker().ownerImage(kSub, 64 * 64 * 4) == f.renderer.tracker().ownerImage(kRt, 128 * 128 * 4));
    if (failures) f.dump();
}

// ---- Compare-sampled T#s (QUAL cmp_*, compare_mismatch)

// The draw of ps_sample_compare (a depth texture in the lifted program) reading the T# `at` (dfmt/nfmt, w x h, tile 8) and writing kOut.
void compareDraw(Fixture& f, const Program& cmp, uint64_t at, uint32_t w, uint32_t h, uint32_t dfmt, uint32_t nfmt) {
    vpps4::DrawState d = f.state();
    d.color[0].address = kOut;
    d.ps.code = cmp.code;
    d.ps.userSgprs = 12;
    d.ps.userData.fill(0);
    Fixture::tsharp(d.ps.userData.data(), at, w, h, dfmt, nfmt, 8);
    f.renderer.draw(d);
}
bool compareLine(Fixture& f) {
    for (const std::string& l : f.log)
        if (l.rfind("VPKENGINE_PROBE CMP ", 0) == 0) return true;
    return false;
}

// A T# the program samples with compare over memory and over a colour target (no depth target anywhere): the host resource is not a
// depth texture, so the draws count as compare_mismatch.
void compareMismatch() {
    ProbeOn probe;
    Fixture f;
    const Program cmp = load("ps_sample_compare");
    CHECK(qual(f, "compare_mismatch") == 0);
    const uint64_t before = qual(f, "compare_mismatch");
    compareDraw(f, cmp, kTex, 64, 64, 4, 7);  // memory: a texture filled from guest memory, made a depth texture (the program
                                              // declares depth2d: a colour texture bound there is undefined on Metal)
    CHECK(qual(f, "cmp_binds") == 1 && qual(f, "cmp_draws") == 1);
    CHECK(qual(f, "cmp_route_texture_from_memory") == 1);
    CHECK(qual(f, "cmp_host_depth") == 1 && qual(f, "cmp_host_not_depth") == 0);
    CHECK(qual(f, "compare_mismatch") == before && qual(f, "compare_mismatch_binds") == 0);
    CHECK(qual(f, "cmp_h_4_d32float_texture_from_memory") == 1);
    CHECK(qual(f, "cmp_fill_none") + qual(f, "cmp_fill_gpu_detile") + qual(f, "cmp_fill_cpu_detile") + qual(f, "cmp_fill_linear_upload") + qual(f, "cmp_fill_not_filled") == 1);
    CHECK(qual(f, "cmp_fill_not_filled") == 0);
    f.renderer.draw(f.state());  // a colour target at kRt
    compareDraw(f, cmp, kRt, 128, 128, 4, 7);
    CHECK(qual(f, "cmp_route_color_target_found") == 1);
    CHECK(qual(f, "compare_mismatch") == before + 1 && qual(f, "cmp_host_depth") == 1);
    CHECK(compareLine(f));
    if (failures) f.dump();
    {  // VPKENGINE_CMP_COLOR=1: the old colour texture
        setenv("VPKENGINE_CMP_COLOR", "1", 1);
        Fixture g;
        compareDraw(g, cmp, kTex, 64, 64, 4, 7);
        CHECK(qual(g, "cmp_host_not_depth") == 1 && qual(g, "compare_mismatch") == 1);
        unsetenv("VPKENGINE_CMP_COLOR");
        if (failures) g.dump();
    }
    // The same fill from memory for a 16-bit depth, a cube (6 slices) and a 2D array (4 slices): a depth texture, filled, no mismatch.
    struct Case {
        const char* name;
        uint32_t dfmt, nfmt, type, lastSlice;
    };
    for (const Case c : {Case{"d16unorm", 2, 0, 9, 0}, Case{"cube", 4, 7, 11, 5}, Case{"array", 4, 7, 13, 3}}) {
        Fixture g;
        vpps4::DrawState d = g.state();
        d.color[0].address = kOut;
        d.ps.code = cmp.code;
        d.ps.userSgprs = 12;
        d.ps.userData.fill(0);
        Fixture::tsharp(d.ps.userData.data(), kTex, 64, 64, c.dfmt, c.nfmt, 8);
        d.ps.userData[3] = (d.ps.userData[3] & 0x0fffffffu) | (c.type << 28);
        d.ps.userData[5] = c.lastSlice << 13;
        g.renderer.draw(d);
        CHECK(qual(g, "cmp_binds") == 1 && qual(g, "cmp_route_texture_from_memory") == 1);
        CHECK(qual(g, "cmp_host_depth") == 1 && qual(g, "cmp_host_not_depth") == 0);
        CHECK(qual(g, "compare_mismatch") == 0 && qual(g, "cmp_fill_not_filled") == 0);
        if (failures) g.dump();
    }
}

// Shadow cascades drawn at a dynamic resolution (64x64, one slice per pass) in a 256x256 allocation and sampled with compare through
// a T# of the allocation (a 256x256 array of 2): the depth target is read (its picture copied into the corner of a texture of the
// T#'s size, again whenever it is drawn), not a texture made from memory that the slice passes never write back (a slice cleared
// alone does not make the image the guest's: the texture stayed as first filled). VPKENGINE_DEPTH_LAYOUT_HINT=1: from memory (before).
void depthHintSample() {
    ProbeOn probe;
    const Program cmp = load("ps_sample_compare");
    for (const bool old : {false, true}) {
        if (old) setenv("VPKENGINE_DEPTH_LAYOUT_HINT", "1", 1);
        Fixture f;
        auto drawSlices = [&] {
            for (uint16_t k = 0; k < 2; ++k) {
                vpps4::DrawState d = f.depthState(true);
                d.color[0] = {};
                d.depth.width = d.depth.height = 64;
                d.depth.pitch = d.depth.rows = 256;
                d.depth.sliceStart = k;
                d.depth.depthClear = true;
                f.renderer.draw(d);
            }
        };
        auto sampleArray = [&] {
            vpps4::DrawState d = f.state();
            d.ps.code = cmp.code;
            d.ps.userSgprs = 12;
            d.ps.userData.fill(0);
            Fixture::tsharp(d.ps.userData.data(), kDepth, 256, 256, 4, 7, 8);
            d.ps.userData[3] = (d.ps.userData[3] & 0x0fffffffu) | (13u << 28);  // 2D array
            d.ps.userData[5] = 1u << 13;                                        // slices 0..1
            f.renderer.draw(d);
        };
        drawSlices();
        f.renderer.submit();
        drawSlices();
        sampleArray();
        CHECK(qual(f, "cmp_route_depth_target_found") == (old ? 0u : 1u) && qual(f, "cmp_route_texture_from_memory") == (old ? 1u : 0u));
        drawSlices();
        sampleArray();
        const auto e = f.events();
        CHECK(Fixture::count(e, "copy sampled corner " + Fixture::hex(kDepth)) == (old ? 0u : 2u));  // (copied again after the draws)
        if (old) unsetenv("VPKENGINE_DEPTH_LAYOUT_HINT");
        if (failures) f.dump();
    }
}

uint64_t rate(Fixture& f, const std::string& name) {  // (RATES counters are published only while the probe is on)
    f.renderer.publishProbe(1);
    return vpgpu::probe::counter("RATES", name).get();
}
// The frames' render passes recorded from command `from` on: how many, and the last one.
size_t renderPasses(Fixture& f, size_t from, const vpgpu::RenderPass** last = nullptr) {
    size_t n = 0;
    for (size_t i = from; i < f.backend.commands.size(); ++i)
        if (const auto* rp = std::get_if<vpgpu::RenderPass>(&f.backend.commands[i])) ++n, last && (*last = rp);
    return n;
}
// The draws of the prepass / colour pass of an eye: 4x depth, depth-only, then 4x colour + depth.
vpps4::DrawState depthOnlyDraw(Fixture& f, uint32_t size = 128) {
    vpps4::DrawState d = f.depthState(true);
    d.color[0] = {};
    d.ps = {};
    d.depth.width = d.depth.height = size;
    d.depth.samples = 4;
    return d;
}
vpps4::DrawState colourDraw(Fixture& f) {
    vpps4::DrawState d = f.depthState(true);
    d.color[0].samples = 4;
    d.depth.samples = 4;
    return d;
}

// A depth-only draw on a multisampled depth target keeps the colour target that the draws reading that depth wrote attached
// (unwritten): the prepass, the colour pass and a later depth-only draw are one pass, not a store and a reload of every sample
// each. Learned from the first colour draw; a depth of another size has no partner.
// VPKENGINE_DEPTH_ONLY_SPLIT=1: split as before.
void depthOnlyKeepsPartner() {
    ProbeOn probe;
    for (const bool off : {false, true}) {
        if (off) setenv("VPKENGINE_DEPTH_ONLY_SPLIT", "1", 1);
        Fixture f;
        f.renderer.setMaxSamples(4);
        auto frame = [&] {
            f.renderer.draw(depthOnlyDraw(f));
            f.renderer.draw(depthOnlyDraw(f));
            f.renderer.draw(colourDraw(f));
            f.renderer.draw(colourDraw(f));
            f.renderer.submit();
        };
        const uint64_t p0 = f.renderer.stats().passes;
        frame();  // (learning: depth-only pass, then colour pass)
        CHECK(f.renderer.stats().passes - p0 == 2);
        const uint64_t p1 = f.renderer.stats().passes;
        const size_t from = f.backend.commands.size();
        frame();
        const vpgpu::RenderPass* rp = nullptr;
        CHECK(f.renderer.stats().passes - p1 == (off ? 2u : 1u) && renderPasses(f, from, &rp) == (off ? 2u : 1u));
        if (!off) CHECK(rp && rp->draws.size() == 4 && rp->desc.colors[0].texture && rp->desc.colors[0].load == vpgpu::LoadAction::Load);
        size_t withColour = 0;  // the depth-only states that have the colour format
        for (const vpgpu::PipelineState& s : f.backend.pipelines)
            if (!s.fragmentShader && s.depthFormat != vpgpu::Format::Undefined && s.colorFormats[0] != vpgpu::Format::Undefined) {
                ++withColour;
                CHECK(s.blend[0].writeMask == 0 && !s.blend[0].enabled && s.samples == 4);
            }
        CHECK(withColour == (off ? 0u : 1u));
        CHECK(rate(f, "depth_partner_draws") == (off ? 0u : 2u));
        // Another size at the same address: not the depth the colour target was drawn with.
        const uint64_t p2 = f.renderer.stats().passes;
        const size_t from2 = f.backend.commands.size();
        f.renderer.draw(depthOnlyDraw(f, 64));
        f.renderer.submit();
        rp = nullptr;
        CHECK(f.renderer.stats().passes - p2 == 1 && renderPasses(f, from2, &rp) == 1 && rp && !rp->desc.colors[0].texture);
        CHECK(rate(f, "depth_partner_draws") == (off ? 0u : 2u));
        if (failures) f.dump();
        if (off) unsetenv("VPKENGINE_DEPTH_ONLY_SPLIT");
    }
}

// What one depth-only draw with a pixel shader got: the pipeline state recorded for it.
struct A2mDraw {
    bool found = false, coverage = false, colour0 = false;
    uint32_t writeMask = 0;
    uint64_t fs = 0;
    uint64_t unwritten = 0;  // a2m_unwritten_draws
};
// variant 0: the colour partner learned (4x), 1: a depth of another size, so no partner (4x), 2: one sample.
A2mDraw a2mDepthDraw(bool alphaToMask, unsigned variant) {
    Fixture f;
    f.renderer.setMaxSamples(4);
    if (variant != 2) {
        vpps4::DrawState learn = colourDraw(f);
        learn.color[0].numberType = 4;  // (uint: the export conversion of the partner is not the plain one, so it shows in the pixel shader)
        f.renderer.draw(learn);  // (learns the partner)
        f.renderer.submit();
    }
    const size_t before = f.backend.pipelines.size();
    vpps4::DrawState d = depthOnlyDraw(f, variant == 1 ? 64 : 128);
    if (variant == 2) d.depth.samples = 1;
    d.ps = colourDraw(f).ps;
    d.alphaToMask = alphaToMask;
    f.renderer.draw(d);
    f.renderer.submit();
    A2mDraw out;
    for (size_t i = before; i < f.backend.pipelines.size(); ++i) {
        const vpgpu::PipelineState& s = f.backend.pipelines[i];
        if (!s.fragmentShader || s.depthFormat == vpgpu::Format::Undefined) continue;
        out.found = true;
        out.coverage = s.alphaToCoverage;
        out.colour0 = s.colorFormats[0] != vpgpu::Format::Undefined;
        out.writeMask = s.blend[0].writeMask;
        out.fs = s.fragmentShader;
    }
    out.unwritten = rate(f, "a2m_unwritten_draws");
    if (failures) f.dump();
    return out;
}

// DB_ALPHA_TO_MASK masks by the alpha the pixel shader exports to colour 0 whether or not that target is written: a depth-only draw
// that keeps its colour partner attached (unwritten, write mask 0) gets the host's alpha-to-coverage; one with no colour 0 (no partner,
// or one sample) gets the alpha test in its pixel shader (another translation than without DB_ALPHA_TO_MASK).
// VPKENGINE_A2M_WRITTEN_ONLY=1: neither, as before.
void a2mUnwrittenColour() {
    ProbeOn probe;
    for (const bool off : {false, true}) {
        if (off) setenv("VPKENGINE_A2M_WRITTEN_ONLY", "1", 1);
        for (unsigned v = 0; v < 3; ++v) {
            const A2mDraw on = a2mDepthDraw(true, v), plain = a2mDepthDraw(false, v);
            CHECK(on.found && plain.found);
            CHECK(!plain.coverage && plain.unwritten == 0);
            CHECK(on.colour0 == (v == 0) && plain.colour0 == (v == 0));
            if (v == 0) CHECK(on.writeMask == 0 && plain.writeMask == 0);
            CHECK(on.coverage == (!off && v == 0));
            CHECK((on.fs != plain.fs) == !off);
            CHECK(on.unwritten == 1);
        }
        if (off) unsetenv("VPKENGINE_A2M_WRITTEN_ONLY");
    }
}

// A colour fast clear that comes while the open pass has that colour attached and no draw of the pass has written it (a depth-only
// draw's partner) is the pass's load action: no cut. One that comes after a draw wrote it still waits for the next pass.
void depthPartnerRetroClear() {
    ProbeOn probe;
    Fixture f;
    f.renderer.setMaxSamples(4);
    f.renderer.draw(colourDraw(f));  // (learns the partner)
    f.renderer.submit();
    auto clear = [&] {
        vpps4::DrawState c = f.state();
        c.primitive = vpps4::PrimitiveType::RectList;
        c.color[0].samples = 4;
        c.color[0].fastClear = true;
        c.color[0].clearWord = {0xff204080u, 0};
        f.renderer.draw(c);
    };
    const uint64_t p0 = f.renderer.stats().passes;
    const size_t from = f.backend.commands.size();
    f.renderer.draw(depthOnlyDraw(f));
    f.renderer.draw(depthOnlyDraw(f));
    clear();
    f.renderer.draw(colourDraw(f));
    f.renderer.draw(colourDraw(f));
    f.renderer.submit();
    const vpgpu::RenderPass* rp = nullptr;
    CHECK(f.renderer.stats().passes - p0 == 1 && renderPasses(f, from, &rp) == 1);
    CHECK(rp && rp->draws.size() == 4 && rp->desc.colors[0].load == vpgpu::LoadAction::Clear);
    CHECK(rp && std::abs(rp->desc.colors[0].clear.color[0] - 128.0 / 255) < 1e-9 && std::abs(rp->desc.colors[0].clear.color[2] - 32.0 / 255) < 1e-9);
    CHECK(rate(f, "depth_partner_draws") == 2 && rate(f, "depth_partner_retro_clears") == 1);
    // Written by a draw of the pass first: the clear cuts it, as before.
    const uint64_t p1 = f.renderer.stats().passes;
    f.renderer.draw(depthOnlyDraw(f));
    f.renderer.draw(colourDraw(f));
    clear();
    f.renderer.draw(colourDraw(f));
    f.renderer.submit();
    CHECK(f.renderer.stats().passes - p1 == 2);
    CHECK(rate(f, "depth_partner_retro_clears") == 1);
    if (failures) f.dump();
}

// The same at one sample (no multisampled depth target): a depth-only draw of the depth that an earlier colour pass was drawn with
// keeps that colour attached, unwritten, so the prepass and the opaque pass are one render pass. The colour is loaded and stored as
// any attachment (no samples to keep: no autoMsaa, no kept samples). Learned from the first colour draw; a depth of another size has
// no partner; a clear of the colour that comes before any draw wrote it is the pass's load action; a guest resolve, a depth clear
// of the whole target and a colour written first still cut. VPKENGINE_DEPTH_ONLY_SPLIT=1: split as before (any other value: not).
vpps4::DrawState depthOnly1x(Fixture& f, uint32_t size = 128) {
    vpps4::DrawState d = f.depthState(true);
    d.color[0] = {};
    d.ps = {};
    d.depth.width = d.depth.height = size;
    return d;
}
std::vector<const vpgpu::RenderPass*> passesFrom(Fixture& f, size_t from) {
    std::vector<const vpgpu::RenderPass*> out;
    for (size_t i = from; i < f.backend.commands.size(); ++i)
        if (const auto* rp = std::get_if<vpgpu::RenderPass>(&f.backend.commands[i])) out.push_back(rp);
    return out;
}
void depthPartner1x() {
    ProbeOn probe;
    for (const char* mode : {"", "0", "1"}) {
        const bool off = mode[0] == '1';
        if (mode[0]) setenv("VPKENGINE_DEPTH_ONLY_SPLIT", mode, 1);
        // Prepass, opaque pass and a later depth-only draw of the frame.
        Fixture f;
        f.renderer.setMaxSamples(1);
        auto frame = [&] {
            f.renderer.draw(depthOnly1x(f));
            f.renderer.draw(depthOnly1x(f));
            f.renderer.draw(f.depthState(true));
            f.renderer.draw(f.depthState(true));
            f.renderer.draw(depthOnly1x(f));
            f.renderer.submit();
        };
        size_t from = f.backend.commands.size();
        frame();  // (learning: the depth-only pass, the colour pass; the late depth-only draw joins it, or cuts again)
        const auto learn = passesFrom(f, from);
        CHECK(learn.size() == (off ? 3u : 2u));
        const vpgpu::TextureId learnt = learn.size() > 1 ? learn[1]->desc.colors[0].texture : vpgpu::TextureId{};  // (the commands move as more are recorded)
        from = f.backend.commands.size();
        frame();
        const auto merged = passesFrom(f, from);
        CHECK(merged.size() == (off ? 3u : 1u));
        if (!off && merged.size() == 1) {
            const vpgpu::Attachment& c = merged[0]->desc.colors[0];
            CHECK(merged[0]->draws.size() == 5);
            CHECK(c.texture && c.texture == learnt);  // (the colour the opaque draws write)
            CHECK(c.load == vpgpu::LoadAction::Load && c.store == vpgpu::StoreAction::Store);
            CHECK(!c.autoMsaa && !c.keepSamples && !c.resolve && merged[0]->desc.samples == 1);
            CHECK(merged[0]->desc.depth.texture && merged[0]->desc.depth.load == vpgpu::LoadAction::Load);
        }
        size_t withColour = 0, written = 0;  // the depth-only states that have the colour format / the states that write it
        vpgpu::Format colourFormat = vpgpu::Format::Undefined;
        for (const vpgpu::PipelineState& st : f.backend.pipelines) {
            if (!st.fragmentShader && st.depthFormat != vpgpu::Format::Undefined && st.colorFormats[0] != vpgpu::Format::Undefined) {
                ++withColour;
                colourFormat = st.colorFormats[0];
                CHECK(st.blend[0].writeMask == 0 && !st.blend[0].enabled && st.samples == 1);
            } else if (st.fragmentShader && st.blend[0].writeMask != 0) {
                ++written;
            }
        }
        CHECK(withColour == (off ? 0u : 1u) && written >= 1);
        CHECK(rate(f, "depth_partner_draws") == (off ? 0u : 4u));  // (the late one of the learning frame, and three of this one)
        if (!off) {  // (the format of the partner is the one the opaque draws write)
            for (const vpgpu::PipelineState& st : f.backend.pipelines)
                if (st.fragmentShader && st.depthFormat != vpgpu::Format::Undefined) CHECK(st.colorFormats[0] == colourFormat);
        }
        // A depth of another size at the same address: not the depth the colour was drawn with. Neither is a pass that never
        // gets its opaque draws: the colour stays attached, loaded and stored unchanged.
        from = f.backend.commands.size();
        f.renderer.draw(depthOnly1x(f, 64));
        f.renderer.submit();
        auto other = passesFrom(f, from);
        CHECK(other.size() == 1 && !other[0]->desc.colors[0].texture);
        from = f.backend.commands.size();
        f.renderer.draw(depthOnly1x(f));
        f.renderer.submit();
        auto alone = passesFrom(f, from);
        CHECK(alone.size() == 1 && alone[0]->draws.size() == 1);
        if (!off && alone.size() == 1)
            CHECK(alone[0]->desc.colors[0].texture && alone[0]->desc.colors[0].load == vpgpu::LoadAction::Load &&
                  alone[0]->desc.colors[0].store == vpgpu::StoreAction::Store);
        if (failures) f.dump();
        if (mode[0]) unsetenv("VPKENGINE_DEPTH_ONLY_SPLIT");
    }
    // A fast clear of the colour that comes while the open pass holds it unwritten (the partner) is the pass's load action; one that
    // comes after a draw wrote it cuts.
    {
        Fixture f;
        f.renderer.setMaxSamples(1);
        f.renderer.draw(f.depthState(true));  // (learns the partner)
        f.renderer.submit();
        auto clear = [&] {
            vpps4::DrawState c = f.state();
            c.primitive = vpps4::PrimitiveType::RectList;
            c.color[0].fastClear = true;
            c.color[0].clearWord = {0xff204080u, 0};
            f.renderer.draw(c);
        };
        size_t from = f.backend.commands.size();
        f.renderer.draw(depthOnly1x(f));
        clear();
        f.renderer.draw(f.depthState(true));
        f.renderer.submit();
        auto passes = passesFrom(f, from);
        CHECK(passes.size() == 1 && passes[0]->draws.size() == 2);
        if (passes.size() == 1) {
            CHECK(passes[0]->desc.colors[0].load == vpgpu::LoadAction::Clear);
            CHECK(std::abs(passes[0]->desc.colors[0].clear.color[0] - 128.0 / 255) < 1e-9);
        }
        CHECK(rate(f, "depth_partner_retro_clears") == 1);
        from = f.backend.commands.size();
        f.renderer.draw(depthOnly1x(f));
        f.renderer.draw(f.depthState(true));
        clear();
        f.renderer.draw(f.depthState(true));
        f.renderer.submit();
        CHECK(passesFrom(f, from).size() == 2);
        CHECK(rate(f, "depth_partner_retro_clears") == 1);
        if (failures) f.dump();
    }
    // A depth clear: by the depth-only draw itself it is the load action of the one pass; a whole-target clear (rectangle) cuts
    // the pass and is the load action of the next one, which keeps the colour all the same.
    {
        Fixture f;
        f.renderer.setMaxSamples(1);
        f.renderer.draw(f.depthState(true));
        f.renderer.submit();
        size_t from = f.backend.commands.size();
        vpps4::DrawState c = depthOnly1x(f);
        c.depth.depthClear = true;
        f.renderer.draw(c);
        f.renderer.draw(f.depthState(true));
        f.renderer.submit();
        auto passes = passesFrom(f, from);
        CHECK(passes.size() == 1 && passes[0]->draws.size() == 2);
        if (passes.size() == 1)
            CHECK(passes[0]->desc.depth.load == vpgpu::LoadAction::Clear && passes[0]->desc.colors[0].texture &&
                  passes[0]->desc.colors[0].load == vpgpu::LoadAction::Load);
        from = f.backend.commands.size();
        f.renderer.draw(depthOnly1x(f));
        vpps4::DrawState r = depthOnly1x(f);
        r.primitive = vpps4::PrimitiveType::RectList;
        r.depth.depthClear = true;
        f.renderer.draw(r);
        f.renderer.draw(depthOnly1x(f));
        f.renderer.draw(f.depthState(true));
        f.renderer.submit();
        passes = passesFrom(f, from);
        CHECK(passes.size() == 2);
        if (passes.size() == 2) {
            CHECK(passes[1]->draws.size() == 2 && passes[1]->desc.depth.load == vpgpu::LoadAction::Clear);
            CHECK(passes[1]->desc.colors[0].texture && passes[1]->desc.colors[0].load == vpgpu::LoadAction::Load);
        }
        if (failures) f.dump();
    }
    // The colour of the last opaque draw is the partner (another format at the same depth), and a guest resolve of the colour
    // between the prepass and the opaque pass cuts them.
    {
        Fixture f;
        f.renderer.setMaxSamples(1);
        f.renderer.draw(f.depthState(true));
        vpps4::DrawState b = f.depthState(true);
        b.color[0].format = 5;  // 16_16 unorm
        f.renderer.draw(b);
        f.renderer.submit();
        size_t from = f.backend.commands.size();
        f.renderer.draw(depthOnly1x(f));
        f.renderer.draw(b);
        f.renderer.submit();
        CHECK(passesFrom(f, from).size() == 1);
        vpgpu::Format partnerFormat = vpgpu::Format::Undefined, wrote = vpgpu::Format::Undefined, first = vpgpu::Format::Undefined;
        for (const vpgpu::PipelineState& st : f.backend.pipelines) {
            if (!st.fragmentShader) partnerFormat = st.colorFormats[0];
            else if (first == vpgpu::Format::Undefined) first = st.colorFormats[0];
            else wrote = st.colorFormats[0];
        }
        CHECK(partnerFormat != vpgpu::Format::Undefined && partnerFormat == wrote && wrote != first);
        from = f.backend.commands.size();
        f.renderer.draw(depthOnly1x(f));
        vpps4::ColorTarget src = b.color[0], dst = src;
        dst.address = kBase + 0x70000;
        f.renderer.resolve(src, dst);
        f.renderer.draw(b);
        f.renderer.submit();
        CHECK(passesFrom(f, from).size() == 2);
        if (failures) f.dump();
    }
    // Both eyes as layers of one pair of two-slice targets, foveated: the depth-only draw gets the rate map and both layers of the
    // pass of the opaque draws (else it could not join it). Another eye (other targets) is another pass, not mixed in.
    for (const bool fov : {false, true}) {
        Fixture f;
        f.renderer.setMaxSamples(1);
        f.renderer.setFoveation(fov);
        const uint64_t colour2 = kBase + 0xb0000, depth2 = kBase + 0x90000;  // (not adjacent: the tracker would see one span)
        auto eye = [&](bool second, bool depthOnly) {
            vpps4::DrawState d = f.depthState(true);
            d.color[0].sliceCount = 2;
            d.depth.sliceCount = 2;
            if (second) d.color[0].address = colour2, d.depth.zAddress = depth2;
            if (depthOnly) d.color[0] = {}, d.ps = {};
            return d;
        };
        for (int n = 0; n < 2; ++n) {  // (the first frame learns)
            const size_t from = f.backend.commands.size();
            for (const bool second : {false, true}) {
                f.renderer.draw(eye(second, true));
                f.renderer.draw(eye(second, false));
            }
            f.renderer.submit();
            const auto passes = passesFrom(f, from);
            CHECK(passes.size() == (n == 0 ? 4u : 2u));
            if (n == 1 && passes.size() == 2)
                for (const vpgpu::RenderPass* rp : passes) {
                    CHECK(rp->draws.size() == 2 && rp->desc.colors[0].texture);
                    CHECK((rp->desc.rateMap != nullptr) == fov && rp->desc.layerCount == (fov ? 2u : 1u));
                }
        }
        if (failures) f.dump();
    }
}

// What learns a partner and what it is not used for.
// (1) A one-sample pass over the multisampled depth (colour 1x) does not replace the partner the multisampled draws learned: the
// next multisampled prepass is still one pass with its colour.
void depthPartnerKeepsMs() {
    ProbeOn probe;
    Fixture f;
    f.renderer.setMaxSamples(4);
    auto frame = [&] {
        f.renderer.draw(depthOnlyDraw(f));
        f.renderer.draw(colourDraw(f));
        f.renderer.submit();
    };
    frame();
    const size_t from0 = f.backend.commands.size();
    frame();
    CHECK(passesFrom(f, from0).size() == 1);
    vpps4::DrawState one = f.depthState(true);  // (colour 1x over the multisampled depth)
    one.depth.samples = 4;
    one.color[0].address = kBase + 0xb0000;
    f.renderer.draw(one);
    f.renderer.submit();
    const size_t from = f.backend.commands.size();
    frame();
    CHECK(passesFrom(f, from).size() == 1);
    if (failures) f.dump();
}

// (2) A partner a pass holds without writing it (Load + Store of the whole target for nothing) is dropped after two such passes in a
// row, for 60 frames; a pass that writes it starts the count again. RATES depth_partner_idle_passes counts them.
void depthPartnerIdle() {
    ProbeOn probe;
    const uint64_t colour2 = kBase + 0xb0000;
    {  // the colour of the frame is written in two targets of one depth: the prepass attaches the last one, the opaque pass starts with the other
        Fixture f;
        f.renderer.setMaxSamples(1);
        auto c2 = [&] {
            vpps4::DrawState d = f.depthState(true);
            d.color[0].address = colour2;
            return d;
        };
        f.renderer.draw(f.depthState(true));
        f.renderer.draw(c2());
        f.renderer.submit();
        for (int n = 0; n < 5; ++n) {
            const size_t from = f.backend.commands.size();
            f.renderer.draw(depthOnly1x(f));
            f.renderer.draw(f.depthState(true));
            f.renderer.draw(c2());
            f.renderer.submit();
            const auto passes = passesFrom(f, from);
            CHECK(passes.size() == 3);
            if (!passes.empty()) CHECK(bool(passes[0]->desc.colors[0].texture) == (n < 2));
        }
        CHECK(rate(f, "depth_partner_idle_passes") == 2);
        for (int n = 0; n < 61; ++n) f.renderer.submit();  // (60 frames later it is tried again)
        const size_t from = f.backend.commands.size();
        f.renderer.draw(depthOnly1x(f));
        f.renderer.draw(f.depthState(true));
        f.renderer.draw(c2());
        f.renderer.submit();
        const auto passes = passesFrom(f, from);
        CHECK(!passes.empty() && passes[0]->desc.colors[0].texture);
        if (failures) f.dump();
    }
    {  // depth-only draws that no opaque draw follows; a pass that writes the colour in between starts the count again
        Fixture f;
        f.renderer.setMaxSamples(1);
        f.renderer.draw(f.depthState(true));
        f.renderer.submit();
        auto lone = [&] {
            const size_t from = f.backend.commands.size();
            f.renderer.draw(depthOnly1x(f));
            f.renderer.submit();
            const auto passes = passesFrom(f, from);
            return passes.size() == 1 && passes[0]->desc.colors[0].texture;
        };
        auto used = [&] {
            const size_t from = f.backend.commands.size();
            f.renderer.draw(depthOnly1x(f));
            f.renderer.draw(f.depthState(true));
            f.renderer.submit();
            const auto passes = passesFrom(f, from);
            return passes.size() == 1 && passes[0]->desc.colors[0].texture;
        };
        for (int n = 0; n < 3; ++n) CHECK(lone() && used());
        CHECK(rate(f, "depth_partner_idle_passes") == 3);
        CHECK(lone() && lone());  // (two in a row)
        CHECK(!lone() && !lone());
        CHECK(rate(f, "depth_partner_idle_passes") == 5);
        if (failures) f.dump();
    }
}

// (3) A depth-only draw whose pixel shader samples the colour of its partner is recorded without it (it would read what it is
// attached to); one that samples another texture keeps it.
void depthPartnerSamplesItself() {
    ProbeOn probe;
    Fixture f;
    f.renderer.setMaxSamples(1);
    f.renderer.draw(f.depthState(true));
    f.renderer.submit();
    for (const bool itself : {true, false}) {
        vpps4::DrawState d = depthOnly1x(f);
        d.ps = f.depthState(true).ps;
        if (itself) f.sample(kRt, 128, 128, 10, 0, 8);
        const uint64_t draws = f.renderer.stats().draws, drawn = f.renderer.stats().drawn;
        const size_t from = f.backend.commands.size();
        f.renderer.draw(d);
        f.renderer.submit();
        const auto passes = passesFrom(f, from);
        CHECK(f.renderer.stats().draws == draws + 1 && f.renderer.stats().drawn == drawn + 1);
        CHECK(passes.size() == 1 && passes[0]->draws.size() == 1);
        if (passes.size() == 1) CHECK(bool(passes[0]->desc.colors[0].texture) == !itself);
        f.sample(kTex, 64, 64, 10, 0, 8);
    }
    if (failures) f.dump();
}

// (4) The partner has as many slices as the depth of the draw that would take it (else the pass would have more layers than the colour).
void depthPartnerSlices() {
    ProbeOn probe;
    Fixture f;
    f.renderer.setMaxSamples(1);
    vpps4::DrawState learn = f.depthState(true);
    learn.depth.sliceCount = 2;  // (one slice of colour, two of depth)
    f.renderer.draw(learn);
    f.renderer.submit();
    vpps4::DrawState d = depthOnly1x(f);
    d.depth.sliceCount = 2;
    d.vs.vsOutCntl |= 1u << 18;  // (layered: the pass takes the layers of the depth)
    const size_t from = f.backend.commands.size();
    f.renderer.draw(d);
    f.renderer.submit();
    const auto passes = passesFrom(f, from);
    CHECK(passes.size() == 1 && !passes[0]->desc.colors[0].texture);
    if (failures) f.dump();
}

// (5) A partner is the colour target of its memory: when another target was drawn at that address since (other format), it is not
// attached.
void depthPartnerStale() {
    ProbeOn probe;
    Fixture f;
    f.renderer.setMaxSamples(1);
    f.renderer.draw(f.depthState(true));  // (learns it: format 10 at kRt)
    f.renderer.submit();
    size_t from = f.backend.commands.size();
    f.renderer.draw(depthOnly1x(f));
    f.renderer.submit();
    auto passes = passesFrom(f, from);
    CHECK(passes.size() == 1 && passes[0]->desc.colors[0].texture);
    vpps4::DrawState two = f.depthState(true);  // (two colour targets: learns nothing, writes kRt as 16)
    two.color[0].format = 2;  // (16: another texel size, so another image)
    two.color[1] = {kBase + 0xb0000, 128, 128, 128, 10, 0, 0, 8};
    const uint64_t drawn = f.renderer.stats().drawn;
    f.renderer.draw(two);
    f.renderer.submit();
    CHECK(f.renderer.stats().drawn == drawn + 1);
    from = f.backend.commands.size();
    f.renderer.draw(depthOnly1x(f));
    f.renderer.submit();
    passes = passesFrom(f, from);
    CHECK(passes.size() == 1 && !passes[0]->desc.colors[0].texture);
    if (failures) f.dump();
}

// A one-sample pass (one-fragment colour target) that binds a multisampled depth surface the scene was drawn into tests against that
// surface's single-sample picture: the multisampled pass resolves into it and the one-sample pass attaches it, no depth of its own
// is made at that address. VPKENGINE_DEPTH_1X_SEPARATE=1: a depth of its own (before).
void depth1xSharesMs() {
    ProbeOn probe;
    for (const bool separate : {false, true}) {
        if (separate) setenv("VPKENGINE_DEPTH_1X_SEPARATE", "1", 1);
        Fixture f;
        f.renderer.setMaxSamples(4);
        f.renderer.draw(colourDraw(f));  // the scene: 4x colour + 4x depth
        vpps4::DrawState d = f.depthState(false);
        d.color[0].address = kBase + 0xA0000;  // a one-fragment colour target of its own
        d.depth.samples = 4;
        const uint64_t targets = f.renderer.stats().targets;
        f.renderer.draw(d);
        CHECK(f.renderer.stats().targets - targets == (separate ? 2u : 1u));  // (colour; + a depth of its own)
        f.renderer.submit();
        std::vector<const vpgpu::RenderPass*> passes;
        for (const vpgpu::Command& c : f.backend.commands)
            if (const auto* rp = std::get_if<vpgpu::RenderPass>(&c)) passes.push_back(rp);
        CHECK(passes.size() == 2);
        if (passes.size() == 2) {
            const vpgpu::Attachment &scene = passes[0]->desc.depth, &one = passes[1]->desc.depth;
            CHECK(scene.texture && one.texture && scene.texture != one.texture);
            if (separate) {
                CHECK(!scene.resolve || scene.resolve != one.texture);
            } else {
                CHECK(scene.resolve == one.texture && scene.store == vpgpu::StoreAction::StoreAndResolve && scene.resolveRequired);
                vpgpu::TextureDesc td;
                CHECK(f.device.textureDesc(one.texture, td) && td.samples == 1);
            }
        }
        CHECK(rate(f, "depth_1x_shared") == (separate ? 0u : 1u));
        if (!separate) {
            // Only the last multisampled pass before a one-sample read, and the last of a frame, resolve the picture.
            f.backend.commands.clear();
            vpps4::DrawState other = d;  // a one-sample colour draw with no depth: it ends the 4x pass and reads no picture
            other.depth = decltype(other.depth){};
            f.renderer.draw(colourDraw(f));
            f.renderer.draw(other);
            f.renderer.draw(colourDraw(f));
            f.renderer.submit();
            std::vector<const vpgpu::RenderPass*> later;
            for (const vpgpu::Command& c : f.backend.commands)
                if (const auto* rp = std::get_if<vpgpu::RenderPass>(&c)) later.push_back(rp);
            CHECK(later.size() == 3);
            if (later.size() == 3) CHECK(!later[0]->desc.depth.resolveRequired && later[2]->desc.depth.resolveRequired);
        }
        if (failures) f.dump();
        if (separate) unsetenv("VPKENGINE_DEPTH_1X_SEPARATE");
    }
}

// The render passes recorded so far, and the encoders of the last compiled frame that attach `texture` as colour 0.
std::vector<const vpgpu::RenderPass*> recordedPasses(Fixture& f) {
    std::vector<const vpgpu::RenderPass*> out;
    for (const vpgpu::Command& c : f.backend.commands)
        if (const auto* rp = std::get_if<vpgpu::RenderPass>(&c)) out.push_back(rp);
    return out;
}
std::vector<const vpgpu::Encoder*> encodersOf(Fixture& f, vpgpu::TextureId texture) {
    std::vector<const vpgpu::Encoder*> out;
    if (!f.backend.compiled.empty())
        for (const vpgpu::Encoder& e : f.backend.compiled.back().encoders)
            if (e.kind == vpgpu::Encoder::Kind::Render && e.pass.colors[0].texture == texture) out.push_back(&e);
    return out;
}
uint8_t samplesOf(Fixture& f, vpgpu::TextureId t) {
    vpgpu::TextureDesc d;
    return f.device.textureDesc(t, d) ? d.samples : 0;
}

// A multisampled colour target keeps its samples on tile (Attachment::autoMsaa): the single-sample picture is its resolve target and
// the seed of the next pass that continues it, no pass stores the samples, the guest's resolve of it is a copy of the picture, and a
// shader that reads the samples makes the target store them from then on. The depth attachment of the same pass is unchanged (its
// samples are tested EQUAL by the colour pass). VPKENGINE_MSAA_COLOUR_STORED=1: stored at every pass boundary, as before.
void msaaColourOnTile() {
    ProbeOn probe;
    const uint64_t other = kBase + 0xA0000, copyDst = kBase + 0xC0000;
    {  // (a) the attachment, (b) a pass that continues the picture, (c) the guest's resolve
        // (a draw to another target is a cut of the engine, which keeps the samples: msaa_colour_cut_keeps_samples; here the picture is seeded)
        setenv("VPKENGINE_MSAA_CUT_SAMPLES_OFF", "1", 1);
        Fixture f;
        f.renderer.setMaxSamples(4);
        f.renderer.draw(colourDraw(f));
        vpps4::DrawState o = f.state();  // a draw to another target cuts the pass
        o.color[0].address = other;
        f.renderer.draw(o);
        f.renderer.draw(colourDraw(f));  // continues the picture of the first
        vpps4::ColorTarget src = f.state().color[0], dst = src;
        src.samples = 4;
        dst.address = copyDst;
        dst.samples = 1;
        const uint64_t copies = f.renderer.stats().copies;
        f.renderer.resolve(src, dst);
        CHECK(f.renderer.stats().copies == copies + 1);
        f.renderer.submit();
        const auto passes = recordedPasses(f);
        CHECK(passes.size() == 3);
        if (passes.size() == 3) {
            const vpgpu::Attachment& c = passes[0]->desc.colors[0];
            CHECK(c.autoMsaa && !c.keepSamples && c.resolve && c.resolve != c.texture);
            CHECK(samplesOf(f, c.texture) == 4 && samplesOf(f, c.resolve) == 1);
            const vpgpu::Attachment& z = passes[0]->desc.depth;
            CHECK(z.texture && !z.autoMsaa && samplesOf(f, z.texture) == 4);
            const vpgpu::Attachment& again = passes[2]->desc.colors[0];
            CHECK(again.autoMsaa && again.texture == c.texture && again.resolve == c.resolve && again.load == vpgpu::LoadAction::Load);
            CHECK(!passes[1]->desc.colors[0].autoMsaa);  // (a one-sample target)
            // The compiler: the second encoder is seeded from the picture, no encoder stores the samples.
            const auto enc = encodersOf(f, c.texture);
            CHECK(enc.size() == 2);
            for (const vpgpu::Encoder* e : enc) {
                const vpgpu::Attachment& a = e->pass.colors[0];
                CHECK(e->tileLocal(0) && a.autoMsaa && a.store != vpgpu::StoreAction::Store && a.store != vpgpu::StoreAction::StoreAndResolve);
            }
            if (enc.size() == 2) {
                CHECK(enc[0]->pass.colors[0].store == vpgpu::StoreAction::Resolve && !enc[0]->pass.colors[0].needsSeed);
                const vpgpu::Attachment& second = enc[1]->pass.colors[0];
                CHECK(second.needsSeed && second.seedSrc == c.resolve && second.load == vpgpu::LoadAction::DontCare);
            }
            CHECK(f.backend.compiled.back().stats.seedDraws >= 1);
            // The guest's resolve: a copy of the single-sample picture, no pass of its own, the pass that drew keeps its resolve.
            const vpgpu::Copy* cp = decodeCopy(f, "resolve " + Fixture::hex(kRt) + " -> " + Fixture::hex(copyDst));
            CHECK(cp && cp->kind == vpgpu::CopyKind::TextureToTexture && cp->srcTexture.texture == c.resolve);
            CHECK(f.renderer.stats().notes.count("MSAA resolve done by the pass that drew the samples") == 0);
            CHECK(f.renderer.stats().notes.count("MSAA resolve in a pass of its own") == 0);
        }
        CHECK(rate(f, "msaa_colour_on_tile") == 2 && rate(f, "msaa_colour_stored") == 0);
        unsetenv("VPKENGINE_MSAA_CUT_SAMPLES_OFF");
        if (failures) f.dump();
    }
    {  // (d) a shader reads the samples: they are made (a pass of its own, seeded from the picture) and kept from then on
        Fixture f;
        f.renderer.setMaxSamples(4);
        f.renderer.draw(colourDraw(f));
        f.sample(kRt, 128, 128, 10, 0, 8);
        f.at(kBase + 0x200)[3] = (f.at(kBase + 0x200)[3] & ~(15u << 28)) | (14u << 28) | (2u << 16);
        const auto ms = f.renderer.textureFor(vpps4::decodeImage(std::span<const uint32_t>(f.at(kBase + 0x200), 8)));
        CHECK(ms && samplesOf(f, ms) == 4);
        f.renderer.draw(colourDraw(f));
        f.renderer.submit();
        const auto passes = recordedPasses(f);
        CHECK(passes.size() == 3);
        if (passes.size() == 3) {
            const vpgpu::Attachment &before = passes[0]->desc.colors[0], &made = passes[1]->desc.colors[0], &after = passes[2]->desc.colors[0];
            CHECK(before.autoMsaa && before.texture == ms && made.texture == ms && after.texture == ms);
            CHECK(!made.autoMsaa && made.load == vpgpu::LoadAction::Load && made.store == vpgpu::StoreAction::Store);
            CHECK(!after.autoMsaa && after.keepSamples && after.store == vpgpu::StoreAction::StoreAndResolve && after.resolve == before.resolve);
            const auto enc = encodersOf(f, ms);
            CHECK(enc.size() == 3);
            if (enc.size() == 3) {
                CHECK(enc[0]->pass.colors[0].store == vpgpu::StoreAction::Resolve);  // (the picture the samples are made from)
                const vpgpu::Attachment& seed = enc[1]->pass.colors[0];
                CHECK(seed.needsSeed && seed.seedSrc == before.resolve && seed.store == vpgpu::StoreAction::Store);
                CHECK(!enc[2]->pass.colors[0].needsSeed && enc[2]->pass.colors[0].load == vpgpu::LoadAction::Load);
            }
        }
        CHECK(f.renderer.stats().notes.count("colour target samples made real (a shader reads them): stored at every pass boundary from now on") == 1);
        CHECK(rate(f, "msaa_colour_on_tile") == 1 && rate(f, "msaa_colour_stored") == 1);
        if (failures) f.dump();
    }
    {  // a target in colour slot 1 (the backend seeds colour 0 only) stores its samples: a layered one is anchored layer by layer
        Fixture f;
        f.renderer.setMaxSamples(4);
        vpps4::DrawState a = f.state();
        a.color[0].samples = 4;
        a.color[0].sliceCount = 2;
        f.renderer.draw(a);
        vpps4::DrawState b = f.state();
        b.color[0].address = other;
        b.color[0].samples = 4;
        b.color[1] = a.color[0];
        f.renderer.draw(b);
        f.renderer.submit();
        const auto passes = recordedPasses(f);
        CHECK(!passes.empty() && passes[0]->desc.colors[0].autoMsaa);
        const vpgpu::TextureId ms = passes.empty() ? vpgpu::TextureId{} : passes[0]->desc.colors[0].texture;
        std::set<uint16_t> seeded;
        for (const vpgpu::Encoder* e : encodersOf(f, ms)) {
            const vpgpu::Attachment& s = e->pass.colors[0];
            if (!s.autoMsaa && s.needsSeed && s.seedSrc == passes[0]->desc.colors[0].resolve) seeded.insert(s.seedLayer);
        }
        CHECK(seeded == (std::set<uint16_t>{0, 1}));
        CHECK(f.renderer.stats().notes.count("colour target samples made real (a shader reads them): stored at every pass boundary from now on") == 1);
        if (failures) f.dump();
    }
    {  // (e) VPKENGINE_MSAA_COLOUR_STORED=1: the attachment as before
        setenv("VPKENGINE_MSAA_COLOUR_STORED", "1", 1);
        Fixture f;
        f.renderer.setMaxSamples(4);
        f.renderer.draw(colourDraw(f));
        f.renderer.submit();
        const auto passes = recordedPasses(f);
        CHECK(passes.size() == 1);
        if (passes.size() == 1) {
            const vpgpu::Attachment& c = passes[0]->desc.colors[0];
            CHECK(!c.autoMsaa && c.keepSamples && c.store == vpgpu::StoreAction::StoreAndResolve && samplesOf(f, c.texture) == 4 && c.resolve);
        }
        CHECK(rate(f, "msaa_colour_on_tile") == 0 && rate(f, "msaa_colour_stored") == 1);
        unsetenv("VPKENGINE_MSAA_COLOUR_STORED");
        if (failures) f.dump();
    }
}

// A cut the engine makes in the middle of a multisampled colour pass (here a DMA between two draws) keeps the real samples: the pass that
// ends stores them and the pass that continues loads them, instead of a resolve and a reseed of every sample from the picture (the
// reseed mixes the clear colour into the samples of the edge texels: a dark line on polygon edges). The guest's own resolve is not such
// a cut: the picture is what it asked for and the next pass is seeded from it. VPKENGINE_MSAA_CUT_SAMPLES_OFF=1: seeded as before.
void msaaColourCutKeepsSamples() {
    ProbeOn probe;
    const uint64_t copyDst = kBase + 0xC0000;
    // (a fast clear of the target waits and becomes the load action of its next pass: only a pass that begins with a clear is kept)
    const auto clearRt = [](Fixture& f, uint64_t address = kRt, uint8_t samples = 4, uint16_t sliceStart = 0, uint16_t sliceCount = 1) {
        vpps4::DrawState c = f.state();
        c.primitive = vpps4::PrimitiveType::RectList;
        c.color[0].address = address;
        c.color[0].samples = samples;
        c.color[0].sliceStart = sliceStart;
        c.color[0].sliceCount = sliceCount;
        c.color[0].fastClear = true;
        c.color[0].clearWord = {0xff204080u, 0};
        f.renderer.draw(c);
    };
    const auto guestResolve = [&](Fixture& f) {
        vpps4::ColorTarget src = f.state().color[0], dst = src;
        src.samples = 4;
        dst.address = copyDst;
        dst.samples = 1;
        f.renderer.resolve(src, dst);
    };
    {  // (a) a DMA between two draws into one target
        Fixture f;
        f.renderer.setMaxSamples(4);
        clearRt(f);
        f.renderer.draw(colourDraw(f));
        CHECK(f.renderer.copy(kBase + 0x6000, kBase + 0x5000, 64));
        f.renderer.draw(colourDraw(f));
        f.renderer.submit();
        const auto passes = recordedPasses(f);
        CHECK(passes.size() == 2);
        if (passes.size() == 2) {
            const vpgpu::Attachment &first = passes[0]->desc.colors[0], &second = passes[1]->desc.colors[0];
            CHECK(!first.autoMsaa && !second.autoMsaa && first.texture == second.texture && first.resolve == second.resolve);
            CHECK(first.load == vpgpu::LoadAction::Clear && second.load == vpgpu::LoadAction::Load);
            const auto enc = encodersOf(f, first.texture);
            CHECK(enc.size() == 2);
            if (enc.size() == 2) {
                const vpgpu::Attachment &a = enc[0]->pass.colors[0], &b = enc[1]->pass.colors[0];
                CHECK(a.store == vpgpu::StoreAction::StoreAndResolve && a.resolve == first.resolve);  // (the samples and the picture)
                CHECK(!b.needsSeed && b.load == vpgpu::LoadAction::Load);                              // (the samples, not a seed)
            }
            CHECK(f.backend.compiled.back().stats.seedDraws == 0);
        }
        CHECK(rate(f, "msaa_cut_kept") == 1);  // (the DMA; the end of the submit is not a cut)
        if (failures) f.dump();
    }
    {  // (b) two cuts in a row: the pass between them stores what it loaded
        Fixture f;
        f.renderer.setMaxSamples(4);
        clearRt(f);
        f.renderer.draw(colourDraw(f));
        CHECK(f.renderer.copy(kBase + 0x6000, kBase + 0x5000, 64));
        f.renderer.draw(colourDraw(f));
        CHECK(f.renderer.copy(kBase + 0x6000, kBase + 0x5000, 64));
        f.renderer.draw(colourDraw(f));
        f.renderer.submit();
        const auto passes = recordedPasses(f);
        CHECK(passes.size() == 3);
        if (passes.size() == 3) {
            CHECK(!passes[0]->desc.colors[0].autoMsaa && !passes[1]->desc.colors[0].autoMsaa && !passes[2]->desc.colors[0].autoMsaa);
            const auto enc = encodersOf(f, passes[0]->desc.colors[0].texture);
            CHECK(enc.size() == 3);
            if (enc.size() == 3) {
                CHECK(enc[0]->pass.colors[0].store == vpgpu::StoreAction::StoreAndResolve);
                CHECK(enc[1]->pass.colors[0].store == vpgpu::StoreAction::StoreAndResolve && enc[1]->pass.colors[0].load == vpgpu::LoadAction::Load);
                CHECK(!enc[2]->pass.colors[0].needsSeed && enc[2]->pass.colors[0].load == vpgpu::LoadAction::Load);
            }
        }
        CHECK(rate(f, "msaa_cut_kept") == 2);
        if (failures) f.dump();
    }
    {  // (c) the guest's resolve between the draws is not a cut of the engine: the pass is on tile and the next is seeded
        Fixture f;
        f.renderer.setMaxSamples(4);
        clearRt(f);
        f.renderer.draw(colourDraw(f));
        guestResolve(f);
        f.renderer.draw(colourDraw(f));
        guestResolve(f);
        f.renderer.submit();
        const auto passes = recordedPasses(f);
        CHECK(passes.size() == 2);
        if (passes.size() == 2) CHECK(passes[0]->desc.colors[0].autoMsaa && passes[1]->desc.colors[0].autoMsaa);
        CHECK(rate(f, "msaa_cut_kept") == 0);
        if (failures) f.dump();
    }
    {  // (d) a pass that continues the picture (a Load: it is a seed) is not kept, and nothing is kept before a draw
        Fixture f;
        f.renderer.setMaxSamples(4);
        CHECK(f.renderer.copy(kBase + 0x6000, kBase + 0x5000, 64));
        f.renderer.draw(colourDraw(f));
        f.renderer.submit();
        const auto passes = recordedPasses(f);
        CHECK(passes.size() == 1);
        if (passes.size() == 1) CHECK(passes[0]->desc.colors[0].autoMsaa);
        CHECK(rate(f, "msaa_cut_kept") == 0);
        if (failures) f.dump();
    }
    {  // (f) the cut is made by the draw itself (another depth target: the pass cannot continue): it loads the samples the cut kept
        Fixture f;
        f.renderer.setMaxSamples(4);
        clearRt(f);
        f.renderer.draw(colourDraw(f));
        vpps4::DrawState other = colourDraw(f);
        other.depth.zAddress = other.depth.zWriteAddress = kDepth + 0x40000;
        f.renderer.draw(other);
        f.renderer.submit();
        const auto passes = recordedPasses(f);
        CHECK(passes.size() == 2);
        if (passes.size() == 2) {
            CHECK(!passes[0]->desc.colors[0].autoMsaa && !passes[1]->desc.colors[0].autoMsaa);
            CHECK(passes[0]->desc.colors[0].texture == passes[1]->desc.colors[0].texture);
            CHECK(passes[1]->desc.colors[0].load == vpgpu::LoadAction::Load);
        }
        CHECK(f.backend.compiled.back().stats.seedDraws == 0);
        if (failures) f.dump();
    }
    {  // (g) the samples a cut kept are those of one slice: a pass of another slice of the target is seeded from the picture
        Fixture f;
        f.renderer.setMaxSamples(4);
        clearRt(f, kRt, 4, 1, 1);
        vpps4::DrawState one = colourDraw(f);
        one.color[0].sliceStart = 1;
        one.color[0].sliceCount = 1;
        f.renderer.draw(one);
        CHECK(f.renderer.copy(kBase + 0x6000, kBase + 0x5000, 64));  // (kept: slice 1)
        vpps4::DrawState zero = colourDraw(f);
        zero.color[0].sliceStart = 0;
        zero.color[0].sliceCount = 2;  // (the same target: two slices, the first)
        f.renderer.draw(zero);
        guestResolve(f);  // (ends the pass without keeping it)
        f.renderer.submit();
        const auto passes = recordedPasses(f);
        CHECK(passes.size() == 2);
        if (passes.size() == 2) {
            CHECK(passes[0]->desc.colors[0].texture == passes[1]->desc.colors[0].texture);
            CHECK(!passes[0]->desc.colors[0].autoMsaa);
            CHECK(passes[1]->desc.colors[0].autoMsaa && passes[1]->desc.colors[0].load == vpgpu::LoadAction::Load);
        }
        if (failures) f.dump();
    }
    {  // (h) the guest's resolve, whatever it flushes first (a clear waiting for its destination), ends the pass without keeping
        Fixture f;
        f.renderer.setMaxSamples(4);
        clearRt(f, copyDst, 1);
        clearRt(f);
        f.renderer.draw(colourDraw(f));
        vpps4::ColorTarget src = f.state().color[0], dst = src;
        src.samples = 4;
        dst.address = copyDst;
        dst.samples = 1;
        dst.width = dst.height = 64;  // (not the whole destination: its clear is flushed before the resolve, not dropped)
        f.renderer.resolve(src, dst);
        f.renderer.submit();
        const auto passes = recordedPasses(f);
        bool autoDrawn = false;
        for (const vpgpu::RenderPass* rp : passes) autoDrawn = autoDrawn || (rp->desc.colors[0].autoMsaa && !rp->draws.empty());
        CHECK(autoDrawn);
        CHECK(rate(f, "msaa_cut_kept") == 0);
        if (failures) f.dump();
    }
    {  // (i) a fast clear of the target ends the pass that drew into it without keeping, and what a cut kept before is gone
        Fixture f;
        f.renderer.setMaxSamples(4);
        clearRt(f);
        f.renderer.draw(colourDraw(f));
        CHECK(f.renderer.copy(kBase + 0x6000, kBase + 0x5000, 64));  // (kept)
        f.renderer.draw(colourDraw(f));
        clearRt(f);
        f.renderer.submit();  // (the clear lands as a pass of its own)
        CHECK(rate(f, "msaa_cut_kept") == 1);  // (the DMA; not the clear, which discards what the open pass drew)
        f.renderer.draw(colourDraw(f));
        guestResolve(f);
        f.renderer.submit();
        const auto passes = recordedPasses(f);
        CHECK(!passes.empty() && passes.back()->desc.colors[0].autoMsaa && passes.back()->desc.colors[0].load == vpgpu::LoadAction::Load);
        if (failures) f.dump();
    }
    {  // (j) the picture of the target is written by somebody else (a refill from memory): the next pass is seeded from it
        Fixture f;
        f.renderer.setMaxSamples(4);
        clearRt(f);
        f.renderer.draw(colourDraw(f));
        CHECK(f.renderer.copy(kBase + 0x6000, kBase + 0x5000, 64));  // (kept)
        CHECK(f.renderer.fill(kRt, 64 * 1024, 0x01020304u));          // (a DMA over its memory)
        f.renderer.draw(colourDraw(f));
        guestResolve(f);
        f.renderer.submit();
        const auto passes = recordedPasses(f);
        CHECK(!passes.empty() && passes.back()->desc.colors[0].autoMsaa);
        if (failures) f.dump();
    }
    {  // (k) a pass that does not continue the kept slice and overwrites it (layered, an autoMsaa Load) forgets what was kept
        Fixture f;
        f.renderer.setMaxSamples(4);
        clearRt(f, kRt, 4, 1, 1);
        vpps4::DrawState one = colourDraw(f);
        one.color[0].sliceStart = 1;
        one.color[0].sliceCount = 1;
        f.renderer.draw(one);
        CHECK(f.renderer.copy(kBase + 0x6000, kBase + 0x5000, 64));  // (kept: slice 1)
        vpps4::DrawState layered = colourDraw(f);
        layered.color[0].sliceCount = 2;
        layered.vs.vsOutCntl = 1u << 18;
        f.renderer.draw(layered);
        CHECK(f.renderer.copy(kBase + 0x6000, kBase + 0x5000, 64));  // (the layered pass is a Load: not kept)
        f.renderer.draw(one);
        guestResolve(f);
        f.renderer.submit();
        const auto passes = recordedPasses(f);
        CHECK(passes.size() == 3);
        if (passes.size() == 3) {
            CHECK(passes[1]->desc.layerCount == 2 && passes[1]->desc.colors[0].autoMsaa);
            CHECK(passes[2]->desc.colors[0].autoMsaa && passes[2]->desc.colors[0].load == vpgpu::LoadAction::Load);
        }
        if (failures) f.dump();
    }
    {  // (l) a compute that writes the picture of the target while a pass of it is open does not keep that pass
        Fixture f;
        f.renderer.setMaxSamples(4);
        const auto rtWrite = [&](int times) {
            std::array<uint32_t, 8> a{};
            Fixture::tsharp(a.data(), kRt, 128, 128, 10, 0, 8);
            vpps4::DispatchState x;
            x.cs.code = f.atomic.code;
            x.cs.fetch = f.atomic.fetch;
            x.cs.address = kBase + 0x83000;
            x.cs.userSgprs = 16;
            for (int i = 0; i < 8; ++i) x.cs.userData[i] = a[i], x.cs.userData[8 + i] = a[i];
            for (int i = 0; i < times; ++i) f.renderer.dispatch(x);
        };
        clearRt(f);
        f.renderer.draw(colourDraw(f));
        rtWrite(2);  // (the target is made writable: its pass ends)
        clearRt(f);
        f.renderer.draw(colourDraw(f));
        rtWrite(1);  // (a pass that begins with a clear is open)
        f.renderer.draw(colourDraw(f));
        guestResolve(f);
        f.renderer.submit();
        const auto passes = recordedPasses(f);
        CHECK(!passes.empty() && passes.back()->desc.colors[0].autoMsaa && passes.back()->desc.colors[0].load == vpgpu::LoadAction::Load);
        if (failures) f.dump();
    }
    {  // (m) the end of a submit is not a cut: nothing is kept, and the next submit starts from the pictures
        Fixture f;
        f.renderer.setMaxSamples(4);
        clearRt(f);
        f.renderer.draw(colourDraw(f));
        f.renderer.submit();
        CHECK(rate(f, "msaa_cut_kept") == 0);
        clearRt(f);
        f.renderer.draw(colourDraw(f));
        CHECK(f.renderer.copy(kBase + 0x6000, kBase + 0x5000, 64));  // (kept)
        f.renderer.submit();
        f.renderer.draw(colourDraw(f));
        guestResolve(f);
        f.renderer.submit();
        const auto passes = recordedPasses(f);
        CHECK(!passes.empty() && passes.back()->desc.colors[0].autoMsaa && passes.back()->desc.colors[0].load == vpgpu::LoadAction::Load);
        if (failures) f.dump();
    }
    {  // (n) a fast clear of the target forgets what a cut kept, also when no pass is open and the clear lands on its own
        Fixture f;
        f.renderer.setMaxSamples(4);
        clearRt(f);
        f.renderer.draw(colourDraw(f));
        CHECK(f.renderer.copy(kBase + 0x6000, kBase + 0x5000, 64));  // (kept)
        clearRt(f);
        vpps4::DrawState reader = f.state();  // (a draw to another target that samples the cleared one: the clear lands first)
        reader.color[0].address = kBase + 0xA0000;
        f.sample(kRt, 128, 128, 10, 4, 8);
        f.renderer.draw(reader);
        f.renderer.draw(colourDraw(f));
        guestResolve(f);
        f.renderer.submit();
        const auto passes = recordedPasses(f);
        CHECK(!passes.empty() && passes.back()->desc.colors[0].autoMsaa && passes.back()->desc.colors[0].load == vpgpu::LoadAction::Load);
        if (failures) f.dump();
    }
    {  // (o) the guest's resolve forgets what a cut kept, also when no pass is open
        Fixture f;
        f.renderer.setMaxSamples(4);
        clearRt(f);
        f.renderer.draw(colourDraw(f));
        CHECK(f.renderer.copy(kBase + 0x6000, kBase + 0x5000, 64));  // (kept)
        guestResolve(f);
        f.renderer.draw(colourDraw(f));
        guestResolve(f);
        f.renderer.submit();
        const auto passes = recordedPasses(f);
        CHECK(!passes.empty() && passes.back()->desc.colors[0].autoMsaa && passes.back()->desc.colors[0].load == vpgpu::LoadAction::Load);
        if (failures) f.dump();
    }
    {  // (p) a shader that reads the samples ends the pass without keeping it: the samples are made from the picture, as before
        Fixture f;
        f.renderer.setMaxSamples(4);
        clearRt(f);
        f.renderer.draw(colourDraw(f));
        f.sample(kRt, 128, 128, 10, 0, 8);
        f.at(kBase + 0x200)[3] = (f.at(kBase + 0x200)[3] & ~(15u << 28)) | (14u << 28) | (2u << 16);
        const auto ms = f.renderer.textureFor(vpps4::decodeImage(std::span<const uint32_t>(f.at(kBase + 0x200), 8)));
        CHECK(ms && samplesOf(f, ms) == 4);
        f.renderer.submit();
        const auto passes = recordedPasses(f);
        CHECK(!passes.empty() && passes[0]->desc.colors[0].autoMsaa);
        CHECK(rate(f, "msaa_cut_kept") == 0);
        if (failures) f.dump();
    }
    {  // (e) VPKENGINE_MSAA_CUT_SAMPLES_OFF=1: reseeded from the picture as before
        setenv("VPKENGINE_MSAA_CUT_SAMPLES_OFF", "1", 1);
        Fixture f;
        f.renderer.setMaxSamples(4);
        clearRt(f);
        f.renderer.draw(colourDraw(f));
        CHECK(f.renderer.copy(kBase + 0x6000, kBase + 0x5000, 64));
        f.renderer.draw(colourDraw(f));
        f.renderer.submit();
        const auto passes = recordedPasses(f);
        CHECK(passes.size() == 2);
        if (passes.size() == 2) CHECK(passes[0]->desc.colors[0].autoMsaa && passes[1]->desc.colors[0].autoMsaa);
        CHECK(f.backend.compiled.back().stats.seedDraws >= 1 && rate(f, "msaa_cut_kept") == 0);
        unsetenv("VPKENGINE_MSAA_CUT_SAMPLES_OFF");
        if (failures) f.dump();
    }
}

// A pixel shader that reads the sample barycentrics (SPI_PS_INPUT_ENA bit 0 / 4) makes the host run it once per sample; on the PS4 that
// is the rasterizer state PS_ITER_SAMPLE (Stage::psIterSample). Without it the draw gets another translation of the pixel shader (the
// barycentrics at the centroid: a pixel shader of its own, so the host runs it once per pixel). Counters: ps_sample_bary_draws
// (multisampled draws that enable a sample barycentric), ps_sample_iter_draws (of those, the ones that still run per sample).
// VPKENGINE_PS_SAMPLE_RATE=1: always per sample, as before.
void psSampleRate() {
    ProbeOn probe;
    auto pixelPipelines = [](Fixture& f, std::vector<uint64_t>& keys) {  // the pipeline of each recorded draw
        for (const vpgpu::Command& c : f.backend.commands)
            if (const auto* rp = std::get_if<vpgpu::RenderPass>(&c))
                for (const vpgpu::Draw& d : rp->draws)
                    if (!d.clear) keys.push_back(d.pipeline.value);
    };
    for (const bool off : {false, true}) {
        if (off) setenv("VPKENGINE_PS_SAMPLE_RATE", "1", 1);
        Fixture f;
        f.renderer.setMaxSamples(4);
        vpps4::DrawState a = colourDraw(f);
        a.ps.psInputAddr = 1;  // PERSP_SAMPLE (psIterSample defaults to true)
        vpps4::DrawState b = a;
        b.ps.psIterSample = false;
        f.renderer.draw(a);
        const uint64_t shadersA = f.renderer.stats().shaders;
        f.renderer.draw(b);
        const uint64_t shadersB = f.renderer.stats().shaders;
        CHECK(shadersB == shadersA + (off ? 0u : 1u));  // (the vertex shader is shared, the pixel shader is not)
        f.renderer.draw(a);
        CHECK(f.renderer.stats().shaders == shadersB);
        vpps4::DrawState lin = b;  // LINEAR_SAMPLE
        lin.ps.psInputAddr = 0x10;
        f.renderer.draw(lin);
        vpps4::DrawState centre = a;  // no sample barycentric: not counted
        centre.ps.psInputAddr = 2;
        f.renderer.draw(centre);
        vpps4::DrawState single = f.state();  // one sample on the host: not counted
        single.ps.psInputAddr = 1;
        single.ps.psIterSample = false;
        f.renderer.draw(single);
        f.renderer.submit();
        std::vector<uint64_t> keys;
        pixelPipelines(f, keys);
        CHECK(keys.size() >= 5);
        if (keys.size() >= 5) {
            CHECK((keys[0] != keys[1]) == !off);
            CHECK(keys[0] == keys[2]);
        }
        CHECK(rate(f, "ps_sample_bary_draws") == 4);
        CHECK(rate(f, "ps_sample_iter_draws") == (off ? 4u : 2u));
        if (off) unsetenv("VPKENGINE_PS_SAMPLE_RATE");
        if (failures) f.dump();
    }
}

// A T# larger than every depth target drawn at its base (a 128x128 array of 4 over a depth array drawn 64x64, 2 slices), sampled
// with compare: no depth target serves it and memory holds the bytes of the smaller layout (or of nothing). As the reference
// renderer, which grows its image and does not read memory for it, the texture is not filled from memory: every layer takes the
// depth clear value (a copy each), once, whatever is drawn into the smaller target afterwards. Without a depth target at the base,
// the same holds for a surface whose HTILE was cleared (Renderer::depthCleared) until something else writes its memory.
// VPKENGINE_HTILE_CLEAR_MEMORY=1: filled from memory (before).
void depthClearedTexture() {
    ProbeOn probe;
    const Program cmp = load("ps_sample_compare");
    for (const bool old : {false, true}) {
        if (old) setenv("VPKENGINE_HTILE_CLEAR_MEMORY", "1", 1);
        const std::string label = "copy depth clear value " + Fixture::hex(kDepth);
        auto sampleArray = [&](Fixture& f) {
            vpps4::DrawState d = f.state();
            d.ps.code = cmp.code;
            d.ps.userSgprs = 12;
            d.ps.userData.fill(0);
            Fixture::tsharp(d.ps.userData.data(), kDepth, 128, 128, 4, 7, 8);
            d.ps.userData[3] = (d.ps.userData[3] & 0x0fffffffu) | (13u << 28);  // 2D array
            d.ps.userData[5] = 3u << 13;                                        // slices 0..3
            f.renderer.draw(d);
        };
        {
            Fixture f;
            auto drawSlice = [&](uint16_t k, bool clear) {
                vpps4::DrawState d = f.depthState(true);
                d.color[0] = {};
                d.depth.width = d.depth.height = 64;
                d.depth.sliceStart = k;
                d.depth.depthClear = clear;
                f.renderer.draw(d);
            };
            drawSlice(0, true), drawSlice(1, true);
            f.renderer.submit();
            const uint64_t base = f.renderer.stats().refilled;
            sampleArray(f);
            CHECK(qual(f, "cmp_route_texture_from_memory") == 1);
            CHECK(Fixture::count(f.events(), label) == (old ? 0u : 4u) && qual(f, "depth_clear_fills") == (old ? 0u : 4u));
            CHECK(f.renderer.stats().refilled - base == (old ? 1u : 0u));  // (memory was not read)
            drawSlice(1, false);  // the smaller target is drawn into: the texture is not its picture, nothing changes
            sampleArray(f);
            f.renderer.submit();
            sampleArray(f);
            CHECK(Fixture::count(f.events(), label) == (old ? 0u : 4u));
            if (failures) f.dump();
        }
        {
            Fixture f;  // no depth target at the base: the HTILE clear alone
            f.renderer.depthCleared(kDepth, 1.0f);
            const uint64_t base = f.renderer.stats().refilled;
            sampleArray(f);
            CHECK(Fixture::count(f.events(), label) == (old ? 0u : 4u));
            CHECK(f.renderer.stats().refilled - base == (old ? 1u : 0u));
            f.renderer.depthCleared(kDepth, 1.0f);
            sampleArray(f);
            CHECK(Fixture::count(f.events(), label) == (old ? 0u : 4u));
            if (!old) {
                f.renderer.invalidate(kDepth, 4096);  // the CPU writes the surface: the clear no longer stands, memory is read
                const uint64_t before = f.renderer.stats().refilled;
                sampleArray(f);
                CHECK(f.renderer.stats().refilled == before + 1 && Fixture::count(f.events(), label) == 4u);
            }
            if (failures) f.dump();
        }
        if (old) unsetenv("VPKENGINE_HTILE_CLEAR_MEMORY");
    }
}

// The same depth array drawn 64x64 first in 64x64 slices, then in 256x256 slices (the allocation changes, the drawn size
// stays): the second is another image laid out at its allocation, so a T# of 256x256 reads its corner, not memory. With
// VPKENGINE_DEPTH_REUSE_ANY_PITCH=1 the first image takes the second's draws and the T# goes through memory.
void depthAllocPitchChange() {
    ProbeOn probe;
    const Program cmp = load("ps_sample_compare");
    for (const bool old : {false, true}) {
        if (old) setenv("VPKENGINE_DEPTH_REUSE_ANY_PITCH", "1", 1);
        Fixture f;
        auto drawSlices = [&](uint32_t alloc) {
            for (uint16_t k = 0; k < 2; ++k) {
                vpps4::DrawState d = f.depthState(true);
                d.color[0] = {};
                d.depth.width = d.depth.height = 64;
                d.depth.pitch = d.depth.rows = alloc;
                d.depth.sliceStart = k;
                d.depth.depthClear = true;
                f.renderer.draw(d);
            }
        };
        drawSlices(64);
        f.renderer.submit();
        drawSlices(256);
        vpps4::DrawState d = f.state();
        d.ps.code = cmp.code;
        d.ps.userSgprs = 12;
        d.ps.userData.fill(0);
        Fixture::tsharp(d.ps.userData.data(), kDepth, 256, 256, 4, 7, 8);
        d.ps.userData[3] = (d.ps.userData[3] & 0x0fffffffu) | (13u << 28);  // 2D array
        d.ps.userData[5] = 1u << 13;                                        // slices 0..1
        f.renderer.draw(d);
        CHECK(qual(f, "cmp_route_depth_target_found") == (old ? 0u : 1u) && qual(f, "cmp_route_texture_from_memory") == (old ? 1u : 0u));
        if (old) unsetenv("VPKENGINE_DEPTH_REUSE_ANY_PITCH");
        if (failures) f.dump();
    }
}

// A depth target drawn at 64x64 in a 256x256 allocation, and a T# of `size` over it (dfmt / nfmt).
void drawDepthAlloc(Fixture& f, bool stencil) {
    vpps4::DrawState d = zState(f, 3, stencil, 8);
    d.color[0] = {};
    d.depth.width = d.depth.height = 64;
    d.depth.pitch = d.depth.rows = 256;
    f.renderer.draw(d);
}
vpps4::ImageDesc depthAllocImage(uint32_t dfmt, uint32_t nfmt, uint32_t size) {
    uint32_t w[8]{};
    Fixture::tsharp(w, kDepth, size, size, dfmt, nfmt, 8);
    return vpps4::decodeImage(std::span<const uint32_t>(w, 8));
}

// A T# of the allocation read with integer coordinates addresses the target's texels, which a corner copy does not cover: it
// is not the depth target nor its corner, but the texture made from memory; sampled, it is the corner.
void depthAllocInteger() {
    Fixture f;
    drawDepthAlloc(f, false);
    const vpgpu::TextureId target = f.renderer.textureFor(depthAllocImage(4, 7, 64));
    CHECK(target);
    const std::string corner = "copy sampled corner " + Fixture::hex(kDepth);
    CHECK(f.renderer.textureFor(depthAllocImage(4, 7, 256), true) != target);
    CHECK(Fixture::count(f.events(), corner) == 0);
    CHECK(f.renderer.textureFor(depthAllocImage(4, 7, 256)) != target);
    CHECK(Fixture::count(f.events(), corner) == 1);
    if (failures) f.dump();
}

// A depth target whose corner cannot be copied (a depth and stencil one: the mirror is of D32 / D16 only) is not sampled through
// the whole target scaled to the T#: the T# reads memory.
void depthAllocNoMirror() {
    Fixture f;
    drawDepthAlloc(f, true);
    const vpgpu::TextureId target = f.renderer.textureFor(depthAllocImage(4, 7, 64));
    CHECK(target);
    CHECK(f.renderer.textureFor(depthAllocImage(4, 7, 256)) != target);
    CHECK(Fixture::count(f.events(), "copy sampled corner " + Fixture::hex(kDepth)) == 0);
    if (failures) f.dump();
}

// A T# of the allocation's address in a format that is not the depth's (16 bits, 8888) is another image: memory.
void depthAllocFormat() {
    Fixture f;
    drawDepthAlloc(f, false);
    const vpgpu::TextureId target = f.renderer.textureFor(depthAllocImage(4, 7, 64));
    CHECK(target);
    const std::string corner = "copy sampled corner " + Fixture::hex(kDepth);
    CHECK(f.renderer.textureFor(depthAllocImage(2, 0, 256)) != target);
    CHECK(f.renderer.textureFor(depthAllocImage(10, 0, 256)) != target);
    CHECK(Fixture::count(f.events(), corner) == 0);
    CHECK(f.renderer.textureFor(depthAllocImage(4, 7, 256)) != target);  // (the depth's own format)
    CHECK(Fixture::count(f.events(), corner) == 1);
    if (failures) f.dump();
}

// The same T# over a depth target: the host resource is a depth texture, nothing is counted as a mismatch.
void compareDepth() {
    ProbeOn probe;
    Fixture f;
    const Program cmp = load("ps_sample_compare");
    f.renderer.draw(f.depthState(true));
    compareDraw(f, cmp, kDepth, 128, 128, 4, 7);
    CHECK(qual(f, "cmp_binds") == 1 && qual(f, "cmp_draws") == 1);
    CHECK(qual(f, "cmp_route_depth_target_found") == 1);
    CHECK(qual(f, "cmp_host_depth") == 1 && qual(f, "cmp_host_not_depth") == 0);
    CHECK(qual(f, "compare_mismatch") == 0 && qual(f, "compare_mismatch_binds") == 0);
    CHECK(qual(f, "cmp_h_4_d32float_depth_target_found") == 1);
    CHECK(compareLine(f));
    if (failures) f.dump();
}

// A T# array of 8 slices over a depth target drawn in slices 0-5 only (shadow cascades: the T# names more slices than the frame
// draws): when ONE target holds every drawn slice (drawn from the top slice down, or a later frame reusing the biggest target)
// the depth target itself is sampled and no texture is refilled from memory. Drawn in ascending order for the first time, each slice
// has a target of its own (layers 1..6) and the biggest holds only the last slice: the T# is then filled from memory, which has them
// all. One layer is not an array, an MSAA array (type 15) is never taken loosely. VPKENGINE_DEPTH_ARRAY_STRICT=1 keeps the old rule.
void compareDepthArray() {
    ProbeOn probe;
    const Program cmp = load("ps_sample_compare");
    auto arrayDraw = [&](Fixture& f, uint32_t type = 13) {
        vpps4::DrawState d = f.state();
        d.color[0].address = kOut;
        d.ps.code = cmp.code;
        d.ps.userSgprs = 12;
        d.ps.userData.fill(0);
        Fixture::tsharp(d.ps.userData.data(), kDepth, 128, 128, 4, 7, 8);
        d.ps.userData[3] = (d.ps.userData[3] & 0x0fffffffu) | (type << 28);  // 2D array
        d.ps.userData[5] = 7u << 13;                                        // slices 0..7
        f.renderer.draw(d);
    };
    auto drawSlices = [](Fixture& f, uint16_t last, bool descending = false) {
        for (uint16_t k = 0; k <= last; ++k) {
            vpps4::DrawState d = f.depthState(true);
            d.depth.sliceStart = descending ? uint16_t(last - k) : k;
            f.renderer.draw(d);
        }
    };
    auto fromMemory = [&](Fixture& f) { return qual(f, "cmp_route_texture_from_memory") == 1 && qual(f, "cmp_route_depth_target_found") == 0; };
    auto fromTarget = [&](Fixture& f) { return qual(f, "cmp_route_depth_target_found") == 1 && qual(f, "cmp_route_texture_from_memory") == 0; };
    {  // top slice first: one target of 6 layers holds them all
        Fixture f;
        drawSlices(f, 5, true);
        arrayDraw(f);
        CHECK(fromTarget(f));
        CHECK(qual(f, "compare_mismatch") == 0);
        CHECK(Fixture::count(f.events(), "copy retile ") == 0);
        if (failures) f.dump();
    }
    {  // ascending, first frame: six targets of 1..6 layers, the biggest has only slice 5 -> memory
        Fixture f;
        drawSlices(f, 5);
        arrayDraw(f);
        CHECK(fromMemory(f));
        if (failures) f.dump();
    }
    {  // ... the next frame's draws all go to the biggest target (it holds every slice now) -> the target
        Fixture f;
        drawSlices(f, 5);
        f.renderer.submit();
        drawSlices(f, 5);
        arrayDraw(f);
        CHECK(fromTarget(f));
        if (failures) f.dump();
    }
    {
        Fixture f;
        drawSlices(f, 0);
        arrayDraw(f);
        CHECK(fromMemory(f));
    }
    {  // an MSAA array T# is not taken loosely
        Fixture f;
        drawSlices(f, 5, true);
        arrayDraw(f, 15);
        CHECK(fromMemory(f));
    }
    {
        setenv("VPKENGINE_DEPTH_ARRAY_STRICT", "1", 1);
        Fixture f;
        drawSlices(f, 5, true);
        arrayDraw(f);
        unsetenv("VPKENGINE_DEPTH_ARRAY_STRICT");
        CHECK(fromMemory(f));
    }
}

}  // namespace

// TEXVERIFY (probe on; VPKENGINE_TEXVERIFY=1 here, default 50): a texture filled from memory is read back once its frame is submitted and compared
// with the guest memory it came from; a read that differs says so (Recording plays the upload; `corrupt` breaks it).
void texVerify() {
    ProbeOn probe;
    setenv("VPKENGINE_TEXVERIFY", "1", 1);
    auto find = [](Fixture& f, const char* what) {
        for (const std::string& l : f.log)
            if (l.rfind("VPKENGINE_PROBE TEXVERIFY ", 0) == 0 && l.find(what) != std::string::npos) return true;
        return false;
    };
    for (bool corrupt : {false, true}) {
        Fixture f;
        f.backend.corrupt = corrupt;
        for (uint32_t i = 0; i < 64 * 64; ++i) f.at(kTex)[i] = 0x01020304u * (i + 1);
        f.renderer.draw(f.state());
        f.renderer.submit();
        CHECK(find(f, " L0S0 "));
        CHECK(find(f, corrupt ? "result=differs" : "result=ok"));
        CHECK(find(f, "diff_mem=") && find(f, "mem_changed=0"));
        if (failures) f.dump();
    }
    // Not set: one fill in 50 is checked (the probe is on); 0 turns it off.
    for (const char* value : {static_cast<const char*>(nullptr), "0"}) {
        if (value) setenv("VPKENGINE_TEXVERIFY", value, 1);
        else unsetenv("VPKENGINE_TEXVERIFY");
        Fixture f;
        for (int i = 0; i < 49; ++i) {
            f.renderer.invalidate(kTex, 4);
            f.renderer.draw(f.state());
            f.renderer.submit();
        }
        CHECK(!find(f, "TEXVERIFY"));
        f.renderer.invalidate(kTex, 4);
        f.renderer.draw(f.state());
        f.renderer.submit();
        CHECK(find(f, " L0S0 ") == (value == nullptr));
    }
    {  // probe off: nothing, whatever the default
        unsetenv("VPKENGINE_TEXVERIFY");
        vpgpu::probe::setEnabled(false);
        Fixture f;
        for (int i = 0; i < 60; ++i) {
            f.renderer.invalidate(kTex, 4);
            f.renderer.draw(f.state());
            f.renderer.submit();
        }
        CHECK(!find(f, "TEXVERIFY"));
        vpgpu::probe::setEnabled(true);
    }
}

// FRAMECAP (probe on, SHADPS4_PROBE_DIR set; VPKENGINE_FRAMECAP=1f here: every frame): the frame after the arming submit
// lists its passes and reads back what it sampled (Recording plays the upload of the texture): zero texels / tiles counted,
// a picture written for the first capture. "0" or the probe off: nothing. Then the stats themselves on known texels.
void frameCapture() {
    namespace fc = vpps4::framecap;
    {  // stats: RGBA8 64x64 with its first 8x8 tile black (alpha ignored), RGBA16F with a NaN, D32 range
        std::vector<uint8_t> rgba(64 * 64 * 4, 0x40);
        for (uint32_t y = 0; y < 8; ++y)
            for (uint32_t x = 0; x < 8; ++x) std::memcpy(&rgba[(y * 64 + x) * 4], "\0\0\0\xff", 4);
        fc::Stats s = fc::imageStats(rgba.data(), 64, 64, 4, vpgpu::Format::RGBA8Unorm);
        CHECK(s.zero == 64 && s.zero8 == 1 && s.tiles8 == 64 && s.zero32 == 0 && s.tiles32 == 4 && s.nan == 0);
        std::vector<uint16_t> h(16 * 16 * 4, 0x3c00);  // 1.0
        h[5] = 0x7e00;                                 // NaN
        s = fc::imageStats(reinterpret_cast<const uint8_t*>(h.data()), 16, 16, 8, vpgpu::Format::RGBA16Float);
        CHECK(s.nan == 1 && s.zero == 0);
        std::vector<float> z(8 * 8, 0.5f);
        z[0] = 0.25f, z[1] = 0.0f;
        s = fc::imageStats(reinterpret_cast<const uint8_t*>(z.data()), 8, 8, 4, vpgpu::Format::D32Float);
        CHECK(s.lo == 0.0f && s.hi == 0.5f && s.zero == 1 && s.zero8 == 0);
        s = fc::imageStats(rgba.data(), 64, 64, 4, vpgpu::Format::R32Uint);  // not decoded: zero bytes
        CHECK(!s.decoded && s.zero == 0);
        CHECK(fc::imageStats(rgba.data(), 64, 64, 4, vpgpu::Format::RGBA8Unorm).channels == 4 && fc::imageStats(reinterpret_cast<const uint8_t*>(z.data()), 8, 8, 4, vpgpu::Format::D32Float).channels == 1);
        s = fc::imageStats(rgba.data(), 64, 64, 4, vpgpu::Format::RGBA8Unorm, 2);  // every other texel of every other row
        CHECK(s.pixels == 32 * 32 && s.zero == 16 && s.zero8 == 1);
        // A readback whose texels are not the format's size is never decoded (its buffer is exactly as long as it says): raw counts.
        const std::vector<uint8_t> shortRead(16 * 16 * 2, 0);
        s = fc::imageStats(shortRead.data(), 16, 16, 2, vpgpu::Format::RGBA32Float);
        CHECK(!s.decoded && s.pixels == 256 && s.zero == 256 && s.nan == 0);
        s = fc::imageStats(shortRead.data(), 16, 16, 2, vpgpu::Format::RGBA8Unorm);
        CHECK(!s.decoded && s.zero == 256);
        std::uint64_t written = 0;
        CHECK(!fc::writePpm("/nonexistent/x.ppm", shortRead.data(), 16, 16, 2, vpgpu::Format::RGBA32Float, 1, fc::Stats{.decoded = true}, written));
        // The budget: no slice of an image over 64 MB (8k x 8k RGBA32F), the slices of one image within it, 64 M texels in all
        // (a coarser step fits an image in), 256 MB read.
        CHECK(fc::Budget::slices(8192ull * 8192, 16, 8) == 0 && fc::Budget::slices(2048ull * 2048, 16, 8) == 1 &&
              fc::Budget::slices(1024ull * 1024, 4, 8) == 8 && fc::Budget::slices(2048ull * 2048, 4, 8) == 4);
        fc::Budget b;
        uint32_t step = 0;
        for (unsigned i = 0; i < 3; ++i) CHECK(b.take(4096ull * 4096, 1, step) && step == 1);
        CHECK(b.take(8192ull * 4096, 1, step) && step == 2);  // (16.7 M texels of room left: 33.5 M at every other texel and row)
        CHECK(b.take(64ull * 64, 1, step) && step == 1);
        fc::Budget c;
        c.texels = fc::kTotalTexels - 1;
        CHECK(!c.take(4096ull * 4096, 4, step));            // (even at step 8, 262 144 texels do not fit in 1)
        fc::Budget d;
        d.bytes = fc::kTotalBytes - 15;
        CHECK(!d.take(2, 8, step) && d.take(1, 8, step));   // (16 bytes do not fit in 15, 8 do)
    }
    char dir[] = "/tmp/vpps4_framecap_XXXXXX";
    CHECK(mkdtemp(dir) != nullptr);
    setenv("SHADPS4_PROBE_DIR", dir, 1);
    // (the pictures of an earlier run are replaced by the first capture of this one; nothing else in the directory is touched)
    const auto put = [&](const char* name) { std::ofstream(std::string(dir) + "/" + name) << "x"; };
    const auto exists = [&](const char* name) { return std::filesystem::exists(std::string(dir) + "/" + name); };
    put("framecap-9-0-old.ppm"), put("notes.txt"), put("framecap-keep.txt"), put("other.ppm"), put("framecap-9-1-old.bin");
    auto find = [](Fixture& f, const std::string& what) {
        for (const std::string& l : f.log)
            if (l.find(what) != std::string::npos) return &l;
        return static_cast<const std::string*>(nullptr);
    };
    {
        ProbeOn probe;
        setenv("VPKENGINE_FRAMECAP", "1f", 1);
        Fixture f;
        for (uint32_t i = 0; i < 64 * 64; ++i) f.at(kTex)[i] = (i % 64) < 8 && (i / 64) < 8 ? 0xff000000u : 0xff102030u;
        f.renderer.draw(f.state());
        f.renderer.submit();  // arms
        CHECK(!find(f, "FRAMECAP"));
        f.renderer.invalidate(kTex, 4);
        f.renderer.draw(f.state());
        f.renderer.submit();
        CHECK(find(f, "VPKENGINE_PROBE FRAMECAP n=1 ") && find(f, " files=1"));
        CHECK(find(f, "VPKENGINE_PROBE FRAMECAP_PASS n=1 pass=0 ") && find(f, " draws=1 sampled=["));
        const std::string* img = find(f, "kind=texture addr=" + Fixture::hex(kTex) + " fmt=");
        CHECK(img && img->find("read=ok zero_px=15 zero_t8=15 ") != std::string::npos && img->find(" file=framecap-1-") != std::string::npos);
        CHECK(img && img->find(" lo=") == std::string::npos && img->find(" raw_bpe=") == std::string::npos);  // (RGBA: no range of channel 0)
        CHECK(!exists("framecap-9-0-old.ppm") && exists("notes.txt") && exists("framecap-keep.txt") && exists("other.ppm") && exists("framecap-9-1-old.bin"));
        CHECK(find(f, "kind=color addr=" + Fixture::hex(kRt)));  // the target: listed (this backend cannot read it back)
        if (img) {
            const size_t at = img->find(" file=") + 6;
            std::FILE* fp = std::fopen((std::string(dir) + "/" + img->substr(at, img->find(' ', at) - at)).c_str(), "rb");
            CHECK(fp != nullptr);
            if (fp) std::fclose(fp);
        }
        if (failures) f.dump();
    }
    {  // a readback of a size the format does not explain: hash and raw counts, no picture, nothing read out of its bounds
        ProbeOn probe;
        setenv("VPKENGINE_FRAMECAP", "1f", 1);
        Fixture f;
        f.backend.readDivisor = 2;
        for (uint32_t i = 0; i < 64 * 64; ++i) f.at(kTex)[i] = 0xff102030u;
        f.renderer.draw(f.state());
        f.renderer.submit();
        f.renderer.invalidate(kTex, 4);
        f.renderer.draw(f.state());
        f.renderer.submit();
        const std::string* img = find(f, "kind=texture addr=" + Fixture::hex(kTex) + " fmt=");
        CHECK(img && img->find("read=ok ") != std::string::npos && img->find(" raw_bpe=2") != std::string::npos &&
              img->find(" file=") == std::string::npos && img->find(" hash=") != std::string::npos);
        if (failures) f.dump();
    }
    {  // the disk limit: the first picture is written, the next ones of the capture are not
        ProbeOn probe;
        setenv("VPKENGINE_FRAMECAP", "1f", 1);
        setenv("VPKENGINE_FRAMECAP_DISK", "1", 1);
        Fixture f;
        constexpr uint64_t kTex2 = kBase + 0x14000;
        for (uint32_t i = 0; i < 64 * 64; ++i) f.at(kTex)[i] = f.at(kTex2)[i] = 0xff102030u;
        const auto twoDraws = [&] {
            vpps4::DrawState d = f.state();
            f.renderer.draw(d);
            f.sample(kTex2, 64, 64, 10, 0, 8);
            f.renderer.draw(d);
            f.renderer.submit();
        };
        twoDraws();  // arms
        f.renderer.invalidate(kTex, 4);
        twoDraws();
        unsigned images = 0, files = 0;
        for (const std::string& l : f.log)
            if (l.find("FRAMECAP_IMG") != std::string::npos && l.find("kind=texture") != std::string::npos)
                images += l.find("read=ok") != std::string::npos, files += l.find(" file=") != std::string::npos;
        CHECK(images == 2 && files == 1);
        unsetenv("VPKENGINE_FRAMECAP_DISK");
        if (failures) f.dump();
    }
    {  // The user's capture: every frame until it ends, however the periodic one is set. A depth target that drew with colour and
       // is cleared again (the next view) cuts the frame first, and a clear that waits stays with what is drawn next; one
       // SCENEDRAW line per draw on such a depth target, none for a depth target that never drew with colour.
        ProbeOn probe;
        setenv("VPKENGINE_FRAMECAP", "0", 1);
        Fixture f;
        CHECK(f.renderer.endSceneCapture() == 0);
        CHECK(f.renderer.beginSceneCapture() == 1 && f.renderer.beginSceneCapture() == 0);
        const vpps4::DrawState d = f.depthState(true);
        vpps4::DrawState newView = d;
        newView.depth.depthClear = true;
        vpps4::DrawState shadow = d;  // (depth only, another depth target)
        shadow.color[0] = {};
        shadow.depth.zAddress = kDepth + 0x10000;
        vpps4::DrawState clear = f.state();
        clear.primitive = vpps4::PrimitiveType::RectList;
        clear.color[0].fastClear = true;
        clear.color[0].clearWord = {0xff204080u, 0};
        f.renderer.draw(shadow);
        f.renderer.draw(newView);
        f.renderer.draw(d);
        f.renderer.draw(clear);  // (waits for the next draw into the colour target)
        const size_t frames = f.backend.compiled.size(), commands = f.backend.commands.size();
        f.renderer.draw(newView);
        CHECK(f.backend.compiled.size() == frames + 1);
        for (size_t i = commands; i < f.backend.commands.size(); ++i)  // (the first view's frame: the clear is not in it)
            if (const auto* rp = std::get_if<vpgpu::RenderPass>(&f.backend.commands[i])) CHECK(rp->desc.colors[0].load != vpgpu::LoadAction::Clear);
        f.renderer.draw(d);
        f.renderer.submit();
        {
            bool cleared = false;
            for (size_t i = f.backend.commands.size(); i-- > commands;)
                if (const auto* rp = std::get_if<vpgpu::RenderPass>(&f.backend.commands[i]); rp && rp->desc.colors[0].texture) {
                    cleared = rp->desc.colors[0].load == vpgpu::LoadAction::Clear;  // (the first pass of the second view's frame)
                    if (cleared) break;
                }
            CHECK(cleared);
        }
        CHECK(!find(f, "VPKENGINE_PROBE SCENEDRAW"));  // (written when the capture ends)
        CHECK(f.renderer.endSceneCapture() == 1);
        CHECK(find(f, "VPKENGINE_PROBE FRAMECAP n=u1.0 ") && find(f, "VPKENGINE_PROBE FRAMECAP n=u1.1 "));
        CHECK(find(f, "VPKENGINE_PROBE CLEARPASS n=1 f=1 tex=") && find(f, " whole=1 rects=0 depth=0 stencil=0 value=colour why=in_pass"));
        unsigned lines = 0;
        for (const std::string& l : f.log) lines += l.find("VPKENGINE_PROBE SCENEDRAW cap=1 ") != std::string::npos;
        CHECK(lines == 4);
        CHECK(find(f, " frame=0 pass=1 view=1 depth=" + Fixture::hex(kDepth) + " samples=1 d1x=0 color=" + Fixture::hex(kRt)));
        CHECK(find(f, " frame=1 pass=0 view=2 depth=" + Fixture::hex(kDepth)) && !find(f, " depth=" + Fixture::hex(kDepth + 0x10000) + " samples="));
        CHECK(find(f, " test=1 func=1 write=1 zro=0 zclear=10 stencil=0 ") && find(f, " count=3 instances=1"));
        CHECK(find(f, "VPKENGINE_PROBE SCENECAP n=1 end frames=2 draws=4/5 depths=[" + Fixture::hex(kDepth) + ":views=2] "));
        f.renderer.draw(d);
        f.renderer.submit();
        CHECK(!find(f, "FRAMECAP n=u1.2 "));  // (over: nothing more is captured)
        CHECK(f.renderer.beginSceneCapture() == 2);
        f.renderer.draw(d);
        f.renderer.submit();
        CHECK(f.renderer.endSceneCapture() == 2 && find(f, "VPKENGINE_PROBE FRAMECAP n=u2.0 "));
        if (failures) f.dump();
    }
    {  // The user's capture cuts the frame through the host's submit when it has one; one the host never ends is over after 32
       // frames; its pictures have a budget of their own, for each capture, and the lines say what they took.
        ProbeOn probe;
        setenv("VPKENGINE_FRAMECAP", "0", 1);
        setenv("VPKENGINE_SCENECAP_DISK", "1", 1);
        unsigned hostCuts = 0;
        Fixture::hostSubmit = [&](Fixture& x) { ++hostCuts, x.renderer.submit(); };
        Fixture f;
        Fixture::hostSubmit = nullptr;
        unsetenv("VPKENGINE_SCENECAP_DISK");
        constexpr uint64_t kBig = kBase + 0x90000;  // a texture as big as the depth target: its picture is kept
        for (uint32_t i = 0; i < 128 * 128; ++i) f.at(kBig)[i] = 0xff102030u;
        vpps4::DrawState d = f.depthState(true);
        vpps4::DrawState newView = d;
        newView.depth.depthClear = true;
        const auto files = [&](const std::string& n) {
            unsigned k = 0;
            for (const std::string& l : f.log) k += l.find("FRAMECAP_IMG n=" + n) != std::string::npos && l.find(" file=framecap-" + n) != std::string::npos;
            return k;
        };
        for (const char* n : {"u1.", "u2."}) {
            CHECK(f.renderer.beginSceneCapture() != 0);
            f.sample(kBig, 128, 128, 10, 0, 8);
            f.renderer.draw(d);
            f.renderer.draw(newView);  // (the cut: the host's submit)
            f.renderer.draw(d);
            f.renderer.submit();
            CHECK(f.renderer.endSceneCapture() != 0);
            CHECK(files(n) == 1);  // (two frames read the texture; the budget is one picture, for each capture)
        }
        CHECK(hostCuts == 2);
        CHECK(find(f, "FRAMECAP n=u2.1 ") && find(f, "SCENECAP n=2 end frames=2 ") && find(f, " disk_mb=0 disk_full=1 "));
        CHECK(f.renderer.beginSceneCapture() == 3);
        for (int i = 0; i < 40; ++i) {
            f.renderer.draw(d);
            f.renderer.submit();
        }
        CHECK(find(f, "FRAMECAP n=u3.31 ") && !find(f, "FRAMECAP n=u3.32 ") && find(f, "SCENECAP n=3 end frames=32 ") && find(f, " stopped=frames"));
        CHECK(f.renderer.endSceneCapture() == 0);
        if (failures) f.dump();
    }
    for (bool probeOn : {true, false}) {  // "0", and the probe off: nothing
        setenv("VPKENGINE_FRAMECAP", probeOn ? "0" : "1f", 1);
        vpgpu::probe::setEnabled(probeOn);
        Fixture f;
        for (int i = 0; i < 3; ++i) {
            f.renderer.draw(f.state());
            f.renderer.submit();
        }
        CHECK(!find(f, "FRAMECAP"));
        vpgpu::probe::setEnabled(false);
    }
    unsetenv("VPKENGINE_FRAMECAP");
    unsetenv("SHADPS4_PROBE_DIR");
    std::filesystem::remove_all(dir);
}

int main(int argc, char** argv) {
    const std::string_view t = argc > 1 ? argv[1] : "";
    if (t == "rt_to_texture") rtToTexture();
    else if (t == "frame_capture") frameCapture();
    else if (t == "flat_windows_named") flatWindowsNamed();
    else if (t == "depth_alloc_pitch_change") depthAllocPitchChange();
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
    else if (t == "depth_hint_layout") depthHintLayout();
    else if (t == "depth_hint_sample") depthHintSample();
    else if (t == "depth_alloc_integer") depthAllocInteger();
    else if (t == "depth_alloc_no_mirror") depthAllocNoMirror();
    else if (t == "depth_alloc_format") depthAllocFormat();
    else if (t == "flat_map") flatMap();
    else if (t == "window_superseded") windowSuperseded();
    else if (t == "eviction") eviction();
    else if (t == "eviction_bytes") evictionBytes();
    else if (t == "host_watches") hostWatches();
    else if (t == "owned_release") ownedRelease();
    else if (t == "alias_class_change") aliasClassChangeLinear();
    else if (t == "alias_class_change_tiled") aliasClassChangeTiled();
    else if (t == "alias_texture_writer") aliasTextureWriter();
    else if (t == "alias_same_pass") aliasSamePass();
    else if (t == "alias_cpu_write") aliasCpuWriteFirstByte(), aliasCpuWriteInside();
    else if (t == "alias_dma_write") aliasDmaWriteFirstByte(), aliasDmaWriteInside();
    else if (t == "alias_rejects") aliasRejects();
    else if (t == "alias_evicted") aliasEvicted();
    else if (t == "alias_off") aliasOff();
    else if (t == "big_and_small_targets") bigAndSmallTargets();
    else if (t == "small_then_big_targets") smallThenBigTargets();
    else if (t == "compare_mismatch") compareMismatch();
    else if (t == "depth_cleared_texture") depthClearedTexture();
    else if (t == "depth_only_keeps_partner") depthOnlyKeepsPartner();
    else if (t == "a2m_unwritten_colour") a2mUnwrittenColour();
    else if (t == "depth_partner_retro_clear") depthPartnerRetroClear();
    else if (t == "depth_partner_1x") depthPartner1x();
    else if (t == "depth_partner_keeps_ms") depthPartnerKeepsMs();
    else if (t == "depth_partner_idle") depthPartnerIdle();
    else if (t == "depth_partner_samples_itself") depthPartnerSamplesItself();
    else if (t == "depth_partner_slices") depthPartnerSlices();
    else if (t == "depth_partner_stale") depthPartnerStale();
    else if (t == "depth_1x_shares_ms") depth1xSharesMs();
    else if (t == "msaa_colour_on_tile") msaaColourOnTile();
    else if (t == "msaa_colour_cut_keeps_samples") msaaColourCutKeepsSamples();
    else if (t == "ps_sample_rate") psSampleRate();
    else if (t == "compare_depth") compareDepth();
    else if (t == "compare_depth_array") compareDepthArray();
    else if (t == "tex_verify") texVerify();
    else if (t == "alias_channel_order") aliasChannelOrder();
    else return std::fprintf(stderr, "unknown test\n"), 2;
    return failures ? 1 : 0;
}
