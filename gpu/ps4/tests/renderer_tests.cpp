// vpps4::Renderer on the NullBackend: guest memory is a local page-aligned block, shaders are the
// lifter's test programs (gpu/shader/gcn/tests/lift/*.lift). Usage: vpps4_tests TEST
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "vpgpu/core/device.h"
#include "vpps4/fetch.h"
#include "vpps4/formats.h"
#include "vpps4/renderer.h"
#include "vpps4/tiling.h"

#ifndef VPPS4_LIFT_DIR
#error "VPPS4_LIFT_DIR must point at gpu/shader/gcn/tests/lift"
#endif

namespace {

int failures = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #c); \
            ++failures;                                                       \
        }                                                                     \
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

// Guest memory: 1 MiB at guest address 0x10000000.
constexpr uint64_t kBase = 0x10000000, kSize = 1 << 20;

// NullBackend plus the host memory of the renderer's zero buffer (label "vpps4 zeros"), which tests write like the GPU would.
class SpyBackend final : public vpgpu::Backend {
 public:
    vpgpu::NullBackend null;
    std::vector<std::string>& calls = null.calls;
    uint64_t& completedBelow = null.completedBelow;
    void* zerosMemory = nullptr;
    void* sinkMemory = nullptr;  // (label "vpps4 sink": what writable null V#s are bound to)
    void createBuffer(vpgpu::BufferId id, const vpgpu::BufferDesc& d) override {
        if (d.label == "vpps4 zeros") zerosMemory = d.guestMemory;
        if (d.label == "vpps4 sink") sinkMemory = d.guestMemory;
        null.createBuffer(id, d);
    }
    void createTexture(vpgpu::TextureId id, const vpgpu::TextureDesc& d) override { null.createTexture(id, d); }
    void createSampler(vpgpu::SamplerId id, const vpgpu::SamplerDesc& d) override { null.createSampler(id, d); }
    void destroyBuffer(vpgpu::BufferId id) override { null.destroyBuffer(id); }
    void destroyTexture(vpgpu::TextureId id) override { null.destroyTexture(id); }
    void destroySampler(vpgpu::SamplerId id) override { null.destroySampler(id); }
    uint64_t bufferAddress(vpgpu::BufferId id) const override { return null.bufferAddress(id); }
    bool compilePipeline(const vpgpu::PipelineState& p, std::string& error) override { return null.compilePipeline(p, error); }
    void destroyPipeline(vpgpu::PipelineKey k) override { null.destroyPipeline(k); }
    void submit(uint64_t frame, const vpgpu::Frame& f, const vpgpu::CompiledFrame& c) override { null.submit(frame, f, c); }
    bool takeTimings(uint64_t& frame, vpgpu::GpuTimings& out) override { return null.takeTimings(frame, out); }
    uint64_t completedFrames() const override { return null.completedFrames(); }
    vpgpu::ReadResult readTexture(const vpgpu::TextureSpan& s, std::vector<uint8_t>& out) override { return null.readTexture(s, out); }
};

struct Fixture {
    uint64_t budget;  // Hooks::targetBudgetBytes (0 = default)
    void* host = std::aligned_alloc(16384, kSize);
    SpyBackend backend;
    std::vector<std::string> log;
    vpgpu::Device device{backend, 0, [this](const std::string& s) { log.push_back(s); }};
    std::vector<std::pair<uint64_t, vpir::Stage>> shaders;
    std::unique_ptr<vpps4::Renderer> rendererOwner{new vpps4::Renderer(device, hooks())};  // (a test may drop it to see its teardown)
    vpps4::Renderer& renderer = *rendererOwner;
    Program vs = load("vs_fetch"), ps = load("ps_textured");

    explicit Fixture(uint64_t targetBudget = 0) : budget(targetBudget) { std::memset(host, 0, kSize); }
    ~Fixture() { std::free(host); }

    vpps4::Hooks hooks() {
        vpps4::Hooks h;
        h.targetBudgetBytes = budget;
        h.region = [this](uint64_t a, uint64_t s, vpps4::GuestRegion& g) {
            if (a < kBase || a + s > kBase + kSize) return false;
            g = {kBase, kSize, host};
            return true;
        };
        h.shader = [this](uint64_t id, vpir::Stage st, const vpmsl::MslShader&) { shaders.push_back({id, st}); };
        h.log = [this](const std::string& s) { log.push_back(s); };
        return h;
    }
    uint32_t* at(uint64_t guest) { return reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(host) + (guest - kBase)); }

    // A draw of the test programs: VS fetches from a V# table at +0x100 (vertex buffer +0x1000),
    // PS samples a linear 64x64 RGBA8 texture at +0x10000 through a T#/S# table at +0x200.
    vpps4::DrawState state() {
        uint32_t* vtable = at(kBase + 0x100);
        const uint64_t vb = kBase + 0x1000;
        vtable[0] = uint32_t(vb);
        vtable[1] = uint32_t(vb >> 32) | (16u << 16);  // stride 16
        vtable[2] = 64;                                 // records
        vtable[3] = (7u << 12) | (14u << 15);           // float 32_32_32_32
        uint32_t* ttable = at(kBase + 0x200);
        const uint64_t tex = kBase + 0x10000;
        ttable[0] = uint32_t(tex >> 8);
        ttable[1] = uint32_t(tex >> 40) | (10u << 20) | (0u << 26);  // 8_8_8_8 unorm
        ttable[2] = 63 | (63u << 14);                                 // 64x64
        ttable[3] = 4 | (5u << 3) | (6u << 6) | (7u << 9) | (8u << 20) | (9u << 28);  // xyzw, linear, 2D
        ttable[4] = 63u << 13;                                        // pitch 64
        ttable[8] = 0;                                                // S#: wrap, point
        vpps4::DrawState d;
        d.color[0] = {kBase + 0x40000, 128, 128, 128, 10, 0, 0, 8};
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
        d.ps.exportFormat[0] = 4;  // FP16_ABGR
        return d;
    }
    size_t count(const char* prefix) const {
        size_t n = 0;
        for (const std::string& c : backend.calls) n += c.rfind(prefix, 0) == 0;
        return n;
    }
};

void clearAndDraw() {
    Fixture f;
    vpps4::DrawState clear = f.state();
    clear.primitive = vpps4::PrimitiveType::RectList;
    clear.depth = {kBase + 0x60000, 0, 128, 128, 3, 0, 0, 0, 1, 1, true, false, 0.5f, 0};
    f.renderer.draw(clear);
    vpps4::DrawState d = f.state();
    d.depth = clear.depth;
    d.depth.depthClear = false;
    d.depthStencil.depthTest = d.depthStencil.depthWrite = true;
    d.depthStencil.depthFunc = 1;
    f.renderer.draw(d);
    f.renderer.draw(d);
    f.renderer.submit();
    const vpps4::Stats& s = f.renderer.stats();
    CHECK(s.draws == 3);
    CHECK(s.drawn == 2);
    CHECK(s.shaderErrors == 0);
    CHECK(f.shaders.size() == 2);
    CHECK(f.count("submit") == 1);
    CHECK(bool(f.renderer.colorTargetAt(kBase + 0x40000)));
    if (failures)
        for (const std::string& l : f.log) std::fprintf(stderr, "  %s\n", l.c_str());
}

void textured() {
    Fixture f;
    f.renderer.draw(f.state());
    f.renderer.draw(f.state());
    CHECK(f.renderer.stats().drawn == 2);
    CHECK(f.renderer.stats().copies == 1);  // uploaded once
    CHECK(f.renderer.stats().textures == 1);
    f.renderer.submit();
}

// A 2D tiled texture is detiled into a staging buffer and copied in: one copy, counted.
void tiledUpload() {
    Fixture f;
    vpps4::DrawState d = f.state();
    uint32_t* ttable = f.at(kBase + 0x200);
    ttable[3] = (ttable[3] & ~(31u << 20)) | (14u << 20);  // thin_2d_thin
    f.renderer.draw(d);
    CHECK(f.renderer.stats().detiled == 1);
    CHECK(f.renderer.stats().copies == 1);
    f.renderer.submit();
    // A layout that is not covered is reported and left unfilled.
    ttable[3] = (ttable[3] & ~(31u << 20)) | (20u << 20);  // thick_2d_thick
    f.renderer.draw(d);
    CHECK(f.renderer.stats().detiled == 1);
    CHECK(f.renderer.stats().notes.count("tiled texture not detiled: tile mode thick_2d_thick (thick, 3D or PRT) not covered") == 1);
    f.renderer.submit();
}

// Indirect draws and dispatches: arguments stay in guest memory, read by the GPU.
void indirect() {
    Fixture f;
    vpps4::DrawState d = f.state();
    d.indirectAddress = kBase + 0x3000;
    d.count = 1;
    f.renderer.draw(d);
    CHECK(f.renderer.stats().drawn == 1);
    CHECK(f.renderer.stats().indirect == 1);
    d.indirectAddress = kBase + kSize - 8;  // arguments cut off by the end of memory
    f.renderer.draw(d);
    CHECK(f.renderer.stats().skipped.count("indirect arguments outside guest memory") == 1);
    f.renderer.submit();
}

// A compute shader with ds_append on GDS: lifted to an atomic on the renderer's GDS buffer, run
// directly and through indirect arguments.
void gdsAppend() {
    Fixture f;
    static const uint32_t code[] = {0xD8FA0000u, 0x01000000u,  // ds_append v1 gds
                                    0xBF810000u};               // s_endpgm
    vpps4::DispatchState d;
    d.cs.code = code;
    d.cs.address = kBase + 0x82000;
    d.groups = {4, 1, 1};
    f.renderer.dispatch(d);
    f.renderer.dispatch(d);  // (the pipeline compiles on the first one on the null backend)
    CHECK(f.renderer.stats().shaderErrors == 0);
    CHECK(f.renderer.stats().dispatched >= 1);
    CHECK(f.renderer.gds() != nullptr);
    d.indirectAddress = kBase + 0x3000;
    f.renderer.dispatch(d);
    CHECK(f.renderer.stats().indirect == 1);
    f.renderer.submit();
    if (failures)
        for (const std::string& l : f.log) std::fprintf(stderr, "  %s\n", l.c_str());
}

// DMA on the GPU timeline: byte-pattern fills are GPU fills, others go through staging; copies
// between guest ranges; ranges outside guest memory are dropped.
void dma() {
    Fixture f;
    CHECK(f.renderer.fill(kBase + 0x5000, 256, 0));
    CHECK(f.renderer.fill(kBase + 0x5000, 256, 0x3f800000u));
    CHECK(f.renderer.copy(kBase + 0x6000, kBase + 0x5000, 256));
    CHECK(!f.renderer.copy(kBase + 0x6000, kBase + kSize - 16, 256));
    CHECK(f.renderer.stats().copies == 3);
    f.renderer.submit();
}

// A CPU write to a texture's source is found without invalidate(): by content, next frame.
void contentCheck() {
    Fixture f;
    f.renderer.draw(f.state());
    f.renderer.submit();
    f.renderer.draw(f.state());
    CHECK(f.renderer.stats().copies == 1);  // unchanged: not uploaded again
    f.renderer.submit();
    *f.at(kBase + 0x10000) ^= 0xffu;  // first texel (sampled)
    f.renderer.draw(f.state());
    CHECK(f.renderer.stats().copies == 2);
    f.renderer.submit();
}

// The fetch subroutine of the test VS is found by its end, and its call through the user SGPRs.
void fetchSubroutine() {
    Program vs = load("vs_fetch");
    CHECK(vpps4::fetchLength(vs.fetch) == vs.fetch.size());
    std::array<uint32_t, 16> ud{};
    for (unsigned i = 0; i < 16; ++i) ud[i] = 0x1000u * (i + 1);
    const uint64_t a = vpps4::fetchAddress(vs.code, ud, 16);
    CHECK(a != 0);
    CHECK(vpps4::fetchAddress(vs.code, ud, 0) == 0);  // not from user data
}

void linearUpload() {
    Fixture f;
    f.renderer.draw(f.state());
    f.renderer.submit();
    f.renderer.invalidate(kBase + 0x10100, 4);  // inside the texture
    f.renderer.draw(f.state());
    CHECK(f.renderer.stats().copies == 2);
    f.renderer.invalidate(kBase + 0x30000, 4);  // past it
    f.renderer.draw(f.state());
    CHECK(f.renderer.stats().copies == 2);
    f.renderer.submit();
}

void fan() {
    Fixture f;
    vpps4::DrawState d = f.state();
    d.primitive = vpps4::PrimitiveType::TriangleFan;
    d.count = 5;
    f.renderer.draw(d);
    CHECK(f.renderer.stats().drawn == 1);
    CHECK(f.renderer.stats().notes.count("fan/quad/loop primitives converted to lists") == 1);
    f.renderer.submit();
}

void shaderError() {
    Fixture f;
    vpps4::DrawState d = f.state();
    static const uint32_t junk[] = {0xffffffff, 0xffffffff};
    d.vs.code = junk;
    f.renderer.draw(d);
    f.renderer.draw(d);
    CHECK(f.renderer.stats().drawn == 0);
    CHECK(f.renderer.stats().skipped.at("vertex shader not translated") == 2);
    CHECK(f.renderer.stats().shaderErrors == 1);  // translated once, remembered
}

}  // namespace

// EQAA (colour 4, depth 8) under a host limit of 2: every attachment and the pipeline get 2;
// without a limit, colour and depth still share one count (the colour's 4).
void msaaLimit() {
    for (uint8_t limit : {uint8_t(2), uint8_t(8)}) {
        Fixture f;
        f.renderer.setMaxSamples(limit);
        vpps4::DrawState d = f.state();
        d.color[0].samples = 4;
        d.depth = {kBase + 0x60000, 0, 128, 128, 3, 0, 0, 0, 1, 8, false, false, 1.0f, 0};
        d.depthStencil.depthTest = d.depthStencil.depthWrite = true;
        f.renderer.draw(d);
        f.renderer.submit();
        const std::string want = limit == 2 ? " samples=2" : " samples=4";
        size_t targets = 0;
        for (const std::string& c : f.backend.calls)
            if (c.rfind("createTexture", 0) == 0 && c.find("128x128") != std::string::npos) {
                ++targets;
                CHECK(c.size() >= want.size() && c.compare(c.size() - want.size(), want.size(), want) == 0);
            }
        CHECK(targets == 2);
        CHECK(f.renderer.stats().drawn == 1);
    }
    Fixture f;
    f.renderer.setMaxSamples(3);
    CHECK(f.renderer.maxSamples() == 2);
    f.renderer.setMaxSamples(0);
    CHECK(f.renderer.maxSamples() == 1);
}

// Resolve at 1 sample: a texture copy into CB1, which becomes what a flip at its address presents;
// mismatched formats and a multisampled source (host MSAA on) are skipped, not approximated.
void resolveCopy() {
    Fixture f;
    f.renderer.setMaxSamples(1);
    vpps4::ColorTarget src = f.state().color[0], dst = src;
    src.samples = 4;
    dst.address = kBase + 0x70000;
    f.renderer.resolve(src, dst);
    vpps4::ColorTarget other = dst;
    other.format = 12;  // 16_16_16_16: not the source's format
    f.renderer.resolve(src, other);
    f.renderer.submit();
    CHECK(f.renderer.stats().resolves == 2);
    CHECK(f.renderer.stats().copies == 1);
    CHECK(bool(f.renderer.colorTargetAt(kBase + 0x70000)));
    CHECK(f.renderer.stats().skipped.size() == 1);
    Fixture g;  // limit 8: the source stays multisampled
    g.renderer.resolve(src, dst);
    CHECK(g.renderer.stats().copies == 0);
    CHECK(g.renderer.stats().skipped.count("resolve: multisampled source (host MSAA on)") == 1);
}

// A compute pass writes a UNORM storage image; the same memory read through an sRGB T# of the same
// shape is the GPU-written image (not a fresh, empty upload from guest memory), and the overlap
// diagnostics name it.
void storageAlias() {
    Fixture f;
    const Program cs = load("cs_image_atomic");
    std::array<uint32_t, 8> unorm{}, other{};
    auto tsharp = [](std::array<uint32_t, 8>& t, uint64_t a, uint32_t num) {
        t[0] = uint32_t(a >> 8);
        t[1] = uint32_t(a >> 40) | (10u << 20) | (num << 26);  // 8_8_8_8
        t[2] = 63 | (63u << 14);                                // 64x64
        t[3] = 4 | (5u << 3) | (6u << 6) | (7u << 9) | (8u << 20) | (9u << 28);
        t[4] = 63u << 13;
    };
    tsharp(unorm, kBase + 0x20000, 0);
    tsharp(other, kBase + 0x30000, 0);
    vpps4::DispatchState x;
    x.cs.code = cs.code;
    x.cs.fetch = cs.fetch;
    x.cs.address = kBase + 0x82000;
    x.cs.userSgprs = 16;
    for (int i = 0; i < 8; ++i) x.cs.userData[i] = unorm[i], x.cs.userData[8 + i] = other[i];
    f.renderer.dispatch(x);
    CHECK(f.renderer.stats().dispatched == 1);
    const uint64_t made = f.renderer.stats().textures;
    std::array<uint32_t, 8> srgb = unorm;
    tsharp(srgb, kBase + 0x20000, 9);
    CHECK(bool(f.renderer.textureFor(vpps4::decodeImage(srgb))));
    CHECK(f.renderer.stats().textures == made);  // no second texture at that address
    CHECK(f.renderer.stats().notes.count("GPU-written image sampled with another format of the same size (its own format)") == 1);
    CHECK(f.renderer.describeAt(kBase + 0x20000, 64 * 64 * 4).find("gpu_written") != std::string::npos);
    f.renderer.draw(f.state());
    CHECK(f.renderer.describeAt(kBase + 0x41000, 4).find("colour@") != std::string::npos);
    CHECK(f.renderer.describeAt(kBase + 0x90000, 4).find("nothing") != std::string::npos);
    f.renderer.submit();
    if (failures)
        for (const std::string& l : f.log) std::fprintf(stderr, "  %s\n", l.c_str());
}

// A host that never submits: the renderer submits by itself after kRunawayCommands and says so.
void runaway() {
    Fixture f;
    vpps4::DrawState d = f.state();
    for (int i = 0; i < 50000; ++i) f.renderer.draw(d);
    CHECK(f.count("submit") == 1);
    size_t told = 0;
    for (const std::string& l : f.log) told += l.rfind("gpu.err what=frame_not_ended", 0) == 0;
    CHECK(told == 1);
}

// GPU-pending guest ranges: what recorded work touches is busy until gpuIdle(), nothing else is.
void busyRanges() {
    Fixture f;
    CHECK(!f.renderer.gpuMayAccess(kBase + 0x5000, 256));
    CHECK(f.renderer.fill(kBase + 0x5000, 256, 0));
    CHECK(f.renderer.copy(kBase + 0x8000, kBase + 0x7000, 64));
    CHECK(f.renderer.gpuMayAccess(kBase + 0x5000, 4));
    CHECK(f.renderer.gpuMayAccess(kBase + 0x50fc, 16));     // overlaps the tail
    CHECK(f.renderer.gpuMayAccess(kBase + 0x4ff0, 0x20));   // overlaps the head
    CHECK(!f.renderer.gpuMayAccess(kBase + 0x5100, 16));    // just past it
    CHECK(f.renderer.gpuMayAccess(kBase + 0x7020, 4));      // copy source
    CHECK(f.renderer.gpuMayAccess(kBase + 0x8000, 4));      // copy destination
    CHECK(!f.renderer.gpuMayAccess(kBase + 0x6000, 0x1000));
    f.renderer.submit();
    CHECK(f.renderer.gpuMayAccess(kBase + 0x5000, 4));  // submitted is not finished
    f.backend.completedBelow = ~uint64_t(0);             // the GPU finished that frame
    CHECK(!f.renderer.gpuMayAccess(kBase + 0x5000, 4));  // its ranges retired without gpuIdle
    f.backend.completedBelow = 0;
    f.renderer.fill(kBase + 0x5000, 0x100, 0);
    CHECK(f.renderer.recordedMayAccess(kBase + 0x5000, 4));
    f.renderer.submit();
    CHECK(!f.renderer.recordedMayAccess(kBase + 0x5000, 4));  // submitted, no longer recorded
    CHECK(f.renderer.gpuMayAccess(kBase + 0x5000, 4));
    f.renderer.fill(kBase + 0x9000, 0x100, 0);
    f.renderer.submittedIdle();
    CHECK(!f.renderer.gpuMayAccess(kBase + 0x5000, 4));      // the submitted frame retired
    CHECK(f.renderer.gpuMayAccess(kBase + 0x9000, 4));       // the recorded fill did not
    f.renderer.gpuIdle();
    CHECK(!f.renderer.gpuMayAccess(kBase + 0x5000, 4));
}

// What the renderer caches on the device does not grow without end: a render target the game moved on from (another surface at its
// address, unused for a long time in flips) is released, one a draw uses every submit is not, and the renderer's teardown releases
// everything it made.
void evictIdle() {
    Fixture f;
    vpps4::DrawState keep = f.state();  // renders to kBase + 0x40000 and samples the texture at +0x10000
    f.renderer.draw(keep);
    f.renderer.submit();
    constexpr int kSubmits = 3000;
    for (int i = 0; i < kSubmits; ++i) {
        vpps4::DrawState d = f.state();
        d.color[0].address = kBase + 0x200000 + uint64_t(i % 4) * 0x1000;  // four addresses, a new size each time (a resolution change)
        d.color[0].width = d.color[0].pitch = 64 + uint32_t(i);
        f.renderer.draw(d);
        if (i % 50 == 0) f.renderer.draw(keep);  // the first one stays in use
        f.renderer.submit();
        f.renderer.colorTargetAt(kBase + 0x40000);  // the flip
    }
    const vpps4::Stats& s = f.renderer.stats();
    CHECK(s.targets == 1 + kSubmits);  // the kept target and the texture were made once: never released and made again
    CHECK(s.textures == 1);
    const size_t made = f.count("createTexture"), gone = f.count("destroyTexture");
    CHECK(gone > 1000);
    CHECK(made - gone < 1600);  // bounded by what was used in the last ~1200 flips (it was 3002 live before)
    CHECK(bool(f.renderer.colorTargetAt(kBase + 0x40000)));
    // Teardown: every object the renderer made on the device is released.
    f.rendererOwner.reset();
    CHECK(f.count("createTexture") == f.count("destroyTexture"));
    CHECK(f.count("createBuffer") == f.count("destroyBuffer"));
    CHECK(f.count("createSampler") == f.count("destroySampler"));
    CHECK(f.count("createSampler") > 0);
}

// A target drawn once holds contents that exist only on the GPU: it is not released however long it stays unused while nothing else
// takes its address (it used to vanish after ~2200 submits).
void evictKeepsWritten() {
    Fixture f;
    vpps4::DrawState once = f.state();
    once.color[0].address = kBase + 0x50000;
    f.renderer.draw(once);
    f.renderer.submit();
    const vpgpu::TextureId first = f.renderer.colorTargetAt(kBase + 0x50000);
    CHECK(bool(first));
    vpps4::DrawState keep = f.state();
    for (int i = 0; i < 5000; ++i) {
        f.renderer.draw(keep);
        f.renderer.submit();
        f.renderer.colorTargetAt(kBase + 0x40000);
    }
    CHECK(f.renderer.colorTargetAt(kBase + 0x50000) == first);
    CHECK(f.count("destroyTexture") == 0);
}

// The target the last flip presented is not released (the host may hold its id), even when its address was reused and it is idle;
// one that no flip presents last is released then.
void evictKeepsPresented() {
    Fixture f;
    auto destroyed = [&](const vpgpu::TextureId& id) {
        for (const std::string& c : f.backend.calls)
            if (c == "destroyTexture " + std::to_string(id.index)) return true;
        return false;
    };
    vpps4::DrawState stereo = f.state();  // a two-slice target: the flips below go through its second slice
    stereo.color[0].address = kBase + 0x60000;
    stereo.color[0].sliceCount = 2;
    f.renderer.draw(stereo);
    f.renderer.submit();
    uint16_t layer = 0;
    CHECK(bool(f.renderer.colorTargetSlice(kBase + 0x70000, 128, 128, layer)) && layer == 1);
    vpps4::DrawState a = f.state();
    a.color[0].address = kBase + 0x50000;
    f.renderer.draw(a);
    f.renderer.submit();
    const vpgpu::TextureId first = f.renderer.colorTargetAt(kBase + 0x50000);  // presented by the last colorTargetAt
    CHECK(bool(first));
    auto flips = [&](int n) {
        for (int i = 0; i < n; ++i) {
            f.renderer.draw(stereo);
            f.renderer.submit();
            f.renderer.colorTargetSlice(kBase + 0x70000, 128, 128, layer);
        }
    };
    flips(1);  // (a later flip: the new surface is used more recently than the old one)
    a.color[0].width = a.color[0].pitch = 96;  // the address now holds another surface
    f.renderer.draw(a);
    f.renderer.submit();
    flips(3000);
    CHECK(!destroyed(first));
    // Another flip presents the new surface: the old one is only idle and superseded now.
    const vpgpu::TextureId second = f.renderer.colorTargetAt(kBase + 0x50000);
    CHECK(bool(second) && !(second == first));
    flips(3000);
    CHECK(!destroyed(first));  // (still among the last 8 distinct targets presented: the host may hold its id)
    for (int i = 0; i < 8; ++i) {  // 8 other targets presented since: `first` is only idle and superseded now
        vpps4::DrawState o = f.state();
        o.color[0].address = kBase + 0x80000 + uint64_t(i) * 0x1000;
        f.renderer.draw(o);
        f.renderer.submit();
        CHECK(bool(f.renderer.colorTargetAt(kBase + 0x80000 + uint64_t(i) * 0x1000)));
    }
    flips(3000);
    CHECK(destroyed(first));
    CHECK(!destroyed(second));
}

// V#s that point nowhere: read-only bindings get a buffer of zeros nothing ever writes, write-only ones a separate sink, read-write ones (the
// test dispatch loads and stores) a fresh block of zeros from the upload ring each time. Whatever the GPU writes there, whenever it runs the
// dispatch, never shows in what a later draw reads: the zero buffer is not touched by the renderer, nor bound writable.
void dummyCleared() {
    const Program cs = load("cs_bytes");
    auto writeIt = [&](Fixture& f) {  // a dispatch whose buffer access goes to a null V#: bound writable
        vpps4::DispatchState x;
        x.cs.code = cs.code;
        x.cs.fetch = cs.fetch;
        x.cs.address = kBase + 0x82000;
        x.cs.userSgprs = 4;
        x.cs.ldsBytes = 64;
        f.renderer.dispatch(x);
    };
    auto readIt = [&](Fixture& f) {  // a draw whose vertex fetch reads a null V#: bound read-only
        vpps4::DrawState d = f.state();
        uint32_t* vtable = f.at(kBase + 0x100);
        vtable[0] = vtable[1] = 0;
        f.renderer.draw(d);
    };
    auto zeros = [&](Fixture& f) {
        const uint8_t* m = static_cast<const uint8_t*>(f.backend.zerosMemory);
        if (!m) return false;
        for (size_t i = 0; i < 64 * 1024; ++i)
            if (m[i]) return false;
        return true;
    };
    for (int gap : {0, 1, 7, 8, 20}) {  // submits between the writable binding and the read-only one
        Fixture f;
        writeIt(f);
        const uint64_t uploads = f.renderer.stats().uploads;
        f.renderer.submit();
        for (int i = 0; i < gap; ++i) f.renderer.submit();
        readIt(f);
        CHECK(f.backend.zerosMemory != nullptr);
        if (!f.backend.zerosMemory) {
            for (const auto& [why, n] : f.renderer.stats().skipped) std::fprintf(stderr, "  skip %s x%d\n", why.c_str(), int(n));
            return;
        }
        CHECK(f.backend.zerosMemory != f.backend.sinkMemory);
        CHECK(zeros(f));
        CHECK(uploads > 0);
        if (gap == 0) {  // a second writable binding gets zeros again (its own block of the ring), the zero buffer stays as it is
            writeIt(f);
            CHECK(zeros(f));
        }
    }
}

// The host sets the budget (Hooks::targetBudgetBytes), and the last 8 presented targets stay even over it, not only the latest one.
void evictBudgetConfigured() {
    Fixture f(256ull << 20);  // 4 targets of 64 MiB
    auto at = [](int i) { return kBase + 0x200000 + uint64_t(i) * 0x1000; };
    for (int i = 0; i < 12; ++i) {
        vpps4::DrawState d = f.state();
        d.color[0].address = at(i);
        d.color[0].width = d.color[0].height = d.color[0].pitch = 4096;
        f.renderer.draw(d);
        f.renderer.submit();
        if (i < 6) CHECK(bool(f.renderer.colorTargetAt(at(i))));  // presented: 6 of them, over the budget
    }
    for (int i = 0; i < 10; ++i) {
        f.renderer.draw(f.state());
        f.renderer.submit();
    }
    const size_t made = f.count("createTexture"), gone = f.count("destroyTexture");
    CHECK(gone >= 4);
    CHECK(made - gone <= 12);  // 6 presented + the not presented that fit + the small ones: far from the 12 big targets made
    for (int i = 0; i < 6; ++i) CHECK(bool(f.renderer.colorTargetAt(at(i))));
    CHECK(!bool(f.renderer.colorTargetAt(at(6))));
}

// Over the memory budget the oldest targets the last submit did not use go; the presented one and the latest stay.
void evictBudget() {
    Fixture f;
    constexpr int kTargets = 40;  // 64 MiB each (4096x4096 RGBA8): 2.5 GiB
    auto at = [](int i) { return kBase + 0x200000 + uint64_t(i) * 0x1000; };
    for (int i = 0; i < kTargets; ++i) {
        vpps4::DrawState d = f.state();
        d.color[0].address = at(i);
        d.color[0].width = d.color[0].height = d.color[0].pitch = 4096;
        f.renderer.draw(d);
        f.renderer.submit();
        if (i == 0) CHECK(bool(f.renderer.colorTargetAt(at(0))));  // presented by the last flip
    }
    for (int i = 0; i < 10; ++i) {  // (what was released is destroyed kFrames submits later)
        f.renderer.draw(f.state());
        f.renderer.submit();
    }
    const size_t made = f.count("createTexture"), gone = f.count("destroyTexture");
    CHECK(gone >= 8);
    CHECK(made - gone <= 35);  // 32 targets fit in 2 GiB, plus the sampled texture, the small target of the extra draws and the last big one
    CHECK(bool(f.renderer.colorTargetAt(at(0))));
    CHECK(bool(f.renderer.colorTargetAt(at(kTargets - 1))));
    CHECK(!bool(f.renderer.colorTargetAt(at(1))));
}

// A big texture source (a 1080p texture is 8 MiB) is staged in a buffer of its own, not in the draws' upload ring.
void bigStaging() {
    Fixture f;
    const uint64_t bytes = 700000;
    CHECK(f.renderer.fill(kBase + 0x1000, bytes, 0x12345678u));  // a pattern that is not one byte: staged
    const uint64_t size = (bytes + 16383) / 16384 * 16384;
    bool own = false, ring = false;
    for (const std::string& c : f.backend.calls) {
        if (c.rfind("createBuffer", 0) != 0) continue;
        own = own || c.find("size=" + std::to_string(size) + " ") != std::string::npos;
        ring = ring || c.find("size=16777216 ") != std::string::npos;
    }
    CHECK(own);
    CHECK(!ring);
}

// Microbenchmark: 16 frames of 1600 textured draws, a submit every 100 draws (the shape of the
// world scene). Prints the CPU cost per draw of the renderer + frame compiler on the NullBackend.
void drawBench() {
    Fixture f;
    vpps4::DrawState d = f.state();
    for (int k = 0; k < 4; ++k) f.renderer.draw(d);  // warm the shader / pipeline caches
    f.renderer.submit();
    f.renderer.gpuIdle();
    constexpr int kFrames = 16, kDraws = 1600;
    const auto t0 = std::chrono::steady_clock::now();
    for (int fr = 0; fr < kFrames; ++fr) {
        for (int i = 0; i < kDraws; ++i) {
            d.vs.userData[0] = uint32_t(i & 7);  // a few distinct user data sets per frame
            f.renderer.draw(d);
            if (i % 100 == 99) f.renderer.submit();
        }
        f.renderer.gpuIdle();
        f.backend.calls.clear();
    }
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    std::printf("draw_bench: %.2f us/draw (%d draws)\n", us / (kFrames * kDraws), kFrames * kDraws);
    CHECK(f.renderer.stats().drawn >= uint64_t(kFrames * kDraws));
}

// CB_BLEND factors with a gpu/core equivalent are exact (constant alpha and source-alpha saturate used to be approximated
// with other factors and reported as exact); only dual-source and reserved codes are approximations.
void blendFactors() {
    using F = vpgpu::BlendFactor;
    struct Case { uint8_t gcn; F want; bool exact; };
    const Case cases[] = {{0, F::Zero, true}, {1, F::One, true}, {10, F::SrcAlphaSaturate, true}, {13, F::Constant, true},
                          {14, F::OneMinusConstant, true}, {19, F::ConstantAlpha, true}, {20, F::OneMinusConstantAlpha, true},
                          {15, F::Zero, false}, {18, F::Zero, false}, {11, F::Zero, false}};
    for (const Case& c : cases) {
        bool exact = true;
        const F got = vpps4::blendFactor(c.gcn, exact);
        if (got != c.want || exact != c.exact) std::fprintf(stderr, "blendFactor(%u): got %d exact=%d\n", c.gcn, int(got), int(exact));
        CHECK(got == c.want && exact == c.exact);
    }
}

int main(int argc, char** argv) {
    const std::string t = argc > 1 ? argv[1] : "";
    if (t == "clear_and_draw") clearAndDraw();
    else if (t == "textured") textured();
    else if (t == "linear_upload") linearUpload();
    else if (t == "tiled_upload") tiledUpload();
    else if (t == "indirect") indirect();
    else if (t == "gds_append") gdsAppend();
    else if (t == "dma") dma();
    else if (t == "content_check") contentCheck();
    else if (t == "fetch_subroutine") fetchSubroutine();
    else if (t == "fan") fan();
    else if (t == "shader_error") shaderError();
    else if (t == "msaa_limit") msaaLimit();
    else if (t == "resolve") resolveCopy();
    else if (t == "runaway") runaway();
    else if (t == "storage_alias") storageAlias();
    else if (t == "busy_ranges") busyRanges();
    else if (t == "draw_bench") drawBench();
    else if (t == "evict_idle") evictIdle();
    else if (t == "evict_keeps_written") evictKeepsWritten();
    else if (t == "evict_keeps_presented") evictKeepsPresented();
    else if (t == "evict_budget_configured") evictBudgetConfigured();
    else if (t == "evict_budget") evictBudget();
    else if (t == "dummy_cleared") dummyCleared();
    else if (t == "big_staging") bigStaging();
    else if (t == "blend_factors") blendFactors();
    else return std::fprintf(stderr, "unknown test %s\n", t.c_str()), 2;
    return failures ? 1 : 0;
}
