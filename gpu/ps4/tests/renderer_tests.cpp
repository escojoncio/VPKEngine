// vpps4::Renderer on the NullBackend: guest memory is a local page-aligned block, shaders are the
// lifter's test programs (gpu/shader/gcn/tests/lift/*.lift). Usage: vpps4_tests TEST
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "vp4gpu/core/device.h"
#include "vp4msl/msl.h"
#include "vpgcn/lift.h"
#include "vpps4/fetch.h"
#include "vpps4/fixed_function.h"
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

struct Fixture {
    void* host = std::aligned_alloc(16384, kSize);
    vp4gpu::NullBackend backend;
    std::vector<std::string> log;
    vp4gpu::Device device{backend, 0, [this](const std::string& s) { log.push_back(s); }};
    std::vector<std::pair<uint64_t, vp4ir::Stage>> shaders;
    vpps4::Renderer renderer{device, hooks()};
    Program vs = load("vs_fetch"), ps = load("ps_textured");

    Fixture() { std::memset(host, 0, kSize); }
    ~Fixture() { std::free(host); }

    vpps4::Hooks hooks() {
        vpps4::Hooks h;
        h.region = [this](uint64_t a, uint64_t s, vpps4::GuestRegion& g) {
            if (a < kBase || a + s > kBase + kSize) return false;
            g = {kBase, kSize, host};
            return true;
        };
        h.shader = [this](uint64_t id, vp4ir::Stage st, const vp4msl::MslShader&) { shaders.push_back({id, st}); };
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

// A 2D tiled texture is detiled on the GPU (a kernel dispatch and a copy per level): one copy, counted;
// with the GPU path off, on the CPU into a staging buffer.
void tiledUpload() {
    Fixture f;
    vpps4::DrawState d = f.state();
    uint32_t* ttable = f.at(kBase + 0x200);
    ttable[3] = (ttable[3] & ~(31u << 20)) | (14u << 20);  // thin_2d_thin
    f.renderer.draw(d);
    CHECK(f.renderer.stats().detiled == 1);
    CHECK(f.renderer.stats().gpuDetiled == 1);
    CHECK(f.renderer.stats().copies == 1);
    f.renderer.submit();
    // Same on the CPU.
    f.renderer.setGpuTiling(false);
    ttable[3] = (ttable[3] & ~(31u << 20)) | (10u << 20);  // display_2d_thin
    f.renderer.draw(d);
    CHECK(f.renderer.stats().detiled == 2);
    CHECK(f.renderer.stats().gpuDetiled == 1);
    CHECK(f.renderer.stats().copies == 2);
    f.renderer.submit();
    // A layout that is not covered is reported and left unfilled.
    ttable[3] = (ttable[3] & ~(31u << 20)) | (27u << 20);  // unused slot
    f.renderer.draw(d);
    CHECK(f.renderer.stats().detiled == 2);
    CHECK(f.renderer.stats().notes.count("tiled texture not detiled: tile mode index 27 (unused_27) has no layout") == 1);
    f.renderer.submit();
}

// The texture has the T#'s real mip count, uploaded level by level (linear and tiled, GPU and CPU),
// and a T# with other levels at the same address is another texture.
void mipUpload() {
    Fixture f;
    vpps4::DrawState d = f.state();
    uint32_t* ttable = f.at(kBase + 0x200);
    ttable[3] |= 3u << 16;  // last level 3: 64, 32, 16, 8
    f.renderer.draw(d);
    CHECK(f.renderer.stats().copies == 4);
    CHECK(f.renderer.stats().textures == 1);
    f.renderer.draw(d);
    CHECK(f.renderer.stats().copies == 4);  // cached
    ttable[3] &= ~(15u << 16);
    f.renderer.draw(d);
    CHECK(f.renderer.stats().textures == 2);
    CHECK(f.renderer.stats().copies == 5);
    f.renderer.submit();
    // Tiled, on the GPU.
    ttable[3] = (ttable[3] & ~(31u << 20)) | (14u << 20) | (3u << 16);
    f.renderer.draw(d);
    CHECK(f.renderer.stats().detiled == 1 && f.renderer.stats().gpuDetiled == 1);
    CHECK(f.renderer.stats().copies == 9);
    // Tiled, on the CPU.
    f.renderer.setGpuTiling(false);
    ttable[3] = (ttable[3] & ~(31u << 20)) | (10u << 20);
    f.renderer.draw(d);
    CHECK(f.renderer.stats().detiled == 2 && f.renderer.stats().gpuDetiled == 1);
    CHECK(f.renderer.stats().copies == 13);
    f.renderer.submit();
    if (failures)
        for (const std::string& l : f.log) std::fprintf(stderr, "  %s\n", l.c_str());
}

// A render target is written back to guest memory in the CB's layout by a retile kernel.
void writeBack() {
    Fixture f;
    vpps4::DrawState d = f.state();
    d.color[0].tileMode = 14;  // thin_2d_thin
    f.renderer.draw(d);
    CHECK(f.renderer.stats().drawn == 1);
    const size_t copies = f.renderer.stats().copies;
    CHECK(f.renderer.writeBack(d.color[0]));
    CHECK(f.renderer.stats().gpuRetiled == 1 && f.renderer.stats().copies == copies + 1);
    vpps4::ColorTarget other = d.color[0];
    other.address += 0x100000;  // never rendered
    CHECK(!f.renderer.writeBack(other));
    other = d.color[0];
    other.tileMode = 27;
    CHECK(!f.renderer.writeBack(other));
    CHECK(f.renderer.stats().notes.count("render target not written back: tile mode index 27 (unused_27) has no layout") == 1);
    d.color[0].tileMode = 8;  // linear works the same way
    f.renderer.submit();
    f.renderer.draw(d);
    CHECK(f.renderer.writeBack(d.color[0]));
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


// ---- Fixed-function state ---------------------------------------------------------------------

using vp4gpu::BlendFactor;

void polyOffset() {
    vpps4::Raster r;
    r.offsetFront = true;
    r.cullBack = true;
    r.frontScale = 32, r.frontOffset = 3, r.offsetClamp = 0.5f;
    auto b = vpps4::resolveDepthBias(r, vpps4::PrimitiveType::TriangleList, 3, false);
    CHECK(b.enabled && b.constant == 3 && b.slope == 2 && b.clamp == 0.5f && !b.approximated);
    // Z_16 beside stencil lives in a 32-bit plane here: the unit is 256 times finer.
    b = vpps4::resolveDepthBias(r, vpps4::PrimitiveType::TriangleList, 1, true);
    CHECK(b.constant == 3 * 256);
    b = vpps4::resolveDepthBias(r, vpps4::PrimitiveType::TriangleList, 1, false);
    CHECK(b.constant == 3);
    // Front faces culled and no back offset: nothing. Back offset only: the back parameters.
    r.cullFront = true;
    CHECK(!vpps4::resolveDepthBias(r, vpps4::PrimitiveType::TriangleList, 3, false).enabled);
    r.cullBack = false;
    r.offsetBack = true, r.backScale = 16, r.backOffset = -1;
    b = vpps4::resolveDepthBias(r, vpps4::PrimitiveType::TriangleList, 3, false);
    CHECK(b.enabled && b.constant == -1 && b.slope == 1);
    // Both faces drawn with different offsets: approximated. Lines need the para enable.
    r.cullFront = false;
    CHECK(vpps4::resolveDepthBias(r, vpps4::PrimitiveType::TriangleList, 3, false).approximated);
    CHECK(!vpps4::resolveDepthBias(r, vpps4::PrimitiveType::LineList, 3, false).enabled);
    r.offsetPara = true;
    CHECK(vpps4::resolveDepthBias(r, vpps4::PrimitiveType::LineList, 3, false).enabled);
    vpps4::Raster none;
    CHECK(!vpps4::resolveDepthBias(none, vpps4::PrimitiveType::TriangleList, 3, false).enabled);
}

void blendFactors() {
    vpps4::Blend b;
    b.enable = true;
    b.colorSrc = 10;  // SRC_ALPHA_SATURATE
    b.colorDst = 1;
    vpps4::ColorTarget c;
    c.writeMask = 0xF;
    auto r = vpps4::resolveBlend(b, c, false);
    CHECK(r.state.enabled && r.exact && !r.dualSource);
    CHECK(r.state.srcColor == BlendFactor::SrcAlphaSaturate);
    // Dual source: SRC1_COLOR / INV_SRC1_ALPHA.
    b.colorSrc = 15, b.colorDst = 18;
    r = vpps4::resolveBlend(b, c, false);
    CHECK(r.dualSource && r.exact);
    CHECK(r.state.srcColor == BlendFactor::Src1Color && r.state.dstColor == BlendFactor::OneMinusSrc1Alpha);
    // Constant alpha is not the constant colour.
    b.colorSrc = 19, b.colorDst = 20;
    r = vpps4::resolveBlend(b, c, false);
    CHECK(r.state.srcColor == BlendFactor::ConstantAlpha && r.state.dstColor == BlendFactor::OneMinusConstantAlpha);
    // FORCE_DST_ALPHA_1: the destination alpha is one.
    b.colorSrc = 6, b.colorDst = 7;
    c.forceDstAlpha1 = true;
    r = vpps4::resolveBlend(b, c, false);
    CHECK(r.state.srcColor == BlendFactor::One && r.state.dstColor == BlendFactor::Zero);
    // Integer targets and BLEND_BYPASS do not blend; an undefined factor is inexact.
    CHECK(!vpps4::resolveBlend(b, c, true).state.enabled);
    c.blendBypass = true;
    CHECK(!vpps4::resolveBlend(b, c, false).state.enabled);
    c.blendBypass = false;
    b.colorSrc = 11;
    CHECK(!vpps4::resolveBlend(b, c, false).exact);
}

void stencilOps() {
    vpps4::DepthStencil ds;
    ds.stencilTest = true;
    ds.front.func = 7;  // always: the reference is free
    ds.front.zPass = 2;  // ONES
    ds.front.ref = 1;
    auto r = vpps4::resolveStencil(ds, false);
    CHECK(r.front.pass == vp4gpu::StencilOp::Replace && r.refFront == 0xFF && r.exact);
    // REPLACE_OP writes STENCILOPVAL.
    ds.front.zPass = 4, ds.front.opVal = 0x42;
    r = vpps4::resolveStencil(ds, false);
    CHECK(r.front.pass == vp4gpu::StencilOp::Replace && r.refFront == 0x42 && r.exact);
    // A test that uses the reference cannot also carry another replace value.
    ds.front.func = 2;  // equal
    r = vpps4::resolveStencil(ds, false);
    CHECK(!r.exact && r.refFront == 1);
    ds.front.zPass = 3;  // REPLACE_TEST: the test value itself
    r = vpps4::resolveStencil(ds, false);
    CHECK(r.exact && r.refFront == 1);
    // Add / subtract step by STENCILOPVAL: exact for 1, a no-op for 0.
    ds.front.zPass = 5, ds.front.opVal = 1;
    CHECK(vpps4::resolveStencil(ds, false).exact);
    ds.front.opVal = 3;
    CHECK(!vpps4::resolveStencil(ds, false).exact);
    ds.front.opVal = 0;
    r = vpps4::resolveStencil(ds, false);
    CHECK(r.exact && r.front.pass == vp4gpu::StencilOp::Keep);
    // Logic operations with an all-ones / zero value.
    ds.front.zPass = 12, ds.front.opVal = 0xFF;  // XOR with ones = invert
    CHECK(vpps4::resolveStencil(ds, false).front.pass == vp4gpu::StencilOp::Invert);
    ds.front.zPass = 10, ds.front.opVal = 0;     // AND with zero = zero
    CHECK(vpps4::resolveStencil(ds, false).front.pass == vp4gpu::StencilOp::Zero);
    // Separate back face reference; read-only stencil writes nothing.
    ds.backfaceEnable = true;
    ds.back = ds.front;
    ds.back.ref = 9;
    ds.back.func = 2;
    ds.front.zPass = 0;
    ds.back.zPass = 0;
    r = vpps4::resolveStencil(ds, true);
    CHECK(r.refBack == 9 && r.refFront == 1 && r.front.writeMask == 0 && r.back.writeMask == 0);
}

void scissors() {
    vpps4::ScissorRegs s;
    s.present = true;
    s.screen = {0, 0, 1000, 800};
    s.window = {10, 20, 900, 700};
    s.generic = {0, 0, 2000, 2000};
    auto c = vpps4::combineScissor(s);
    CHECK(c.left == 10 && c.top == 20 && c.right == 900 && c.bottom == 700);
    // The window offset moves the window and generic scissors.
    s.windowOffsetDisable = false;
    s.windowOffsetX = 5, s.windowOffsetY = -5;
    c = vpps4::combineScissor(s);
    CHECK(c.left == 15 && c.top == 15 && c.right == 905 && c.bottom == 695);
    // The viewport scissor only when enabled.
    s.viewportScissor = {100, 100, 200, 300};
    CHECK(vpps4::combineScissor(s).left == 15);
    s.viewportScissorEnable = true;
    c = vpps4::combineScissor(s);
    CHECK(c.left == 100 && c.top == 100 && c.right == 200 && c.bottom == 300);
    // Corners the wrong way round: empty, not negative.
    s.viewportScissor = {500, 500, 100, 100};
    c = vpps4::combineScissor(s);
    CHECK(c.right <= c.left && c.bottom <= c.top);
}

void clipAndViewport() {
    vpps4::Viewport v{64, 64, -64, 64, 0.5f, 0.5f};
    auto z = vpps4::viewportDepthRange(v, true);
    CHECK(z.znear == 0.5f && z.zfar == 1.0f);
    z = vpps4::viewportDepthRange(v, false);  // -w..w: ndc z -1..1 -> 0..1
    CHECK(z.znear == 0.0f && z.zfar == 1.0f);
    vpps4::Raster r;
    bool approx = false;
    CHECK(!vpps4::depthClamp(r, approx));
    r.zclipNearDisable = r.zclipFarDisable = true;
    CHECK(vpps4::depthClamp(r, approx) && !approx);
    r.zclipFarDisable = false;
    CHECK(vpps4::depthClamp(r, approx) && approx);
    r = {};
    r.clipDisable = true;
    CHECK(vpps4::depthClamp(r, approx));
    vpps4::Raster m;
    m.polygonModeEnable = true;
    m.polygonModeFront = m.polygonModeBack = 1;
    CHECK(vpps4::resolvePolygonMode(m, vpps4::PrimitiveType::TriangleList).wireframe);
    CHECK(!vpps4::resolvePolygonMode(m, vpps4::PrimitiveType::LineList).wireframe);
    m.polygonModeFront = m.polygonModeBack = 0;
    CHECK(vpps4::resolvePolygonMode(m, vpps4::PrimitiveType::TriangleList).pointMode);
    m.polygonModeEnable = false;
    CHECK(!vpps4::resolvePolygonMode(m, vpps4::PrimitiveType::TriangleList).pointMode);
}

void indexPrep() {
    vpps4::DrawState d;
    d.indexAddress = 0x1000;
    d.primitive = vpps4::PrimitiveType::TriangleStrip;
    CHECK(!vpps4::indicesNeedPrep(d));
    d.raster.primitiveRestart = true;
    d.raster.restartIndex = 0xFFFF;
    CHECK(!vpps4::indicesNeedPrep(d));  // the host's own
    d.raster.restartIndex = 0x7FFF;
    CHECK(vpps4::indicesNeedPrep(d));
    const uint16_t in[] = {1, 2, 0x7FFF, 3, 4};
    std::vector<uint8_t> out;
    vpps4::prepareIndices(d, std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(in), sizeof in), out);
    uint16_t got[5];
    std::memcpy(got, out.data(), sizeof got);
    CHECK(got[0] == 1 && got[2] == 0xFFFF && got[4] == 4);
    // Lists have no restart; swap modes apply to everything.
    d.primitive = vpps4::PrimitiveType::TriangleList;
    CHECK(!vpps4::indicesNeedPrep(d));
    d.raster.primitiveRestart = false;
    d.indexSwap = 1;
    CHECK(vpps4::indicesNeedPrep(d));
    vpps4::prepareIndices(d, std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(in), sizeof in), out);
    std::memcpy(got, out.data(), sizeof got);
    CHECK(got[0] == 0x0100 && got[1] == 0x0200);
    d.indexSwap = 2;
    const uint8_t w[] = {1, 2, 3, 4};
    vpps4::prepareIndices(d, w, out);
    CHECK(out[0] == 4 && out[3] == 1);
    d.indexSwap = 3;
    vpps4::prepareIndices(d, w, out);
    CHECK(out[0] == 3 && out[1] == 4 && out[2] == 1);
}

// A pixel shader program with exports: the words of `exp` instructions, s_endpgm.
std::string sourceOf(const vpgcn::LiftInput& in) {
    vpgcn::LiftResult l = vpgcn::lift(in);
    for (const std::string& e : l.errors) std::fprintf(stderr, "lift error: %s\n", e.c_str());
    CHECK(l.errors.empty());
    return vp4msl::emitMsl(l.module).shader.source;
}

void liftExports() {
    // exp mrt0 v0 x4; exp mrt1 v1 x4 done; s_endpgm
    static const uint32_t two[] = {0xF800000F, 0x00000000, 0xF800081F, 0x01010101, 0xBF810000};
    vpgcn::LiftInput in;
    in.stage = vpgcn::ShaderStage::Pixel;
    in.code = two;
    std::string plain = sourceOf(in);
    CHECK(plain.find("index(1)") == std::string::npos);
    CHECK(plain.find("[[color(1)]]") != std::string::npos);
    in.dualSource = true;
    std::string dual = sourceOf(in);
    CHECK(dual.find("index(1)") != std::string::npos);
    CHECK(dual.find("[[color(1)]]") == std::string::npos);
    // exp mrtz (Z and the sample mask) v0 / v1 done
    static const uint32_t zmask[] = {0xF8000885, 0x00010000, 0xBF810000};
    vpgcn::LiftInput zm;
    zm.stage = vpgcn::ShaderStage::Pixel;
    zm.code = zmask;
    const std::string zs = sourceOf(zm);
    CHECK(zs.find("[[sample_mask]]") != std::string::npos);
    // exp pos0 v0 x4 done: with a -w..w clip space the z is remapped.
    static const uint32_t pos[] = {0xF80008CF, 0x00000000, 0xBF810000};
    vpgcn::LiftInput vs;
    vs.stage = vpgcn::ShaderStage::Vertex;
    vs.code = pos;
    const std::string plainVs = sourceOf(vs);
    vs.clipSpaceGL = true;
    const std::string glVs = sourceOf(vs);
    CHECK(plainVs != glVs);
    CHECK(glVs.find("0.5") != std::string::npos);
}

// Through the renderer: the pipeline and draw state the registers end up as.
void stateReachesBackend() {
    Fixture f;
    vpps4::DrawState d = f.state();
    d.depth = {kBase + 0x60000, 0, 128, 128, 3, 1, 0, 0, 1, 1, false, false, 1.0f, 0};
    d.depth.stencilAddress = kBase + 0x70000;
    d.depthStencil.depthTest = d.depthStencil.depthWrite = true;
    d.depthStencil.depthFunc = 1;
    d.raster.offsetFront = true;
    d.raster.frontScale = 16, d.raster.frontOffset = 2;
    d.raster.zclipNearDisable = d.raster.zclipFarDisable = true;
    d.raster.polygonModeEnable = true;
    d.raster.polygonModeFront = d.raster.polygonModeBack = 1;
    d.alphaToMask = true;
    d.depthStencil.stencilTest = true;
    d.depthStencil.backfaceEnable = true;
    d.depthStencil.front.ref = 5;
    d.depthStencil.back.ref = 6;
    d.scissorRegs.present = true;
    d.scissorRegs.screen = {0, 0, 128, 128};
    d.scissorRegs.window = {8, 8, 100, 90};
    d.scissorRegs.generic = {0, 0, 128, 128};
    d.blend[0].enable = true;
    d.blend[0].colorSrc = 10;
    d.blend[0].colorDst = 1;
    f.renderer.draw(d);
    f.renderer.submit();
    CHECK(f.renderer.stats().drawn == 1);
    CHECK(!f.backend.draws.empty());
    if (f.backend.draws.empty()) return;
    const vp4gpu::Draw& w = f.backend.draws.back();
    CHECK(w.depthBias == 2 && w.depthBiasSlope == 1);
    CHECK(w.depthClamp && w.wireframe);
    CHECK(w.stencilRef == 5 && w.stencilRefBack == 6);
    CHECK(w.scissor.x == 8 && w.scissor.y == 8 && w.scissor.width == 92 && w.scissor.height == 82);
    CHECK(!f.backend.pipelines.empty());
    if (f.backend.pipelines.empty()) return;
    const vp4gpu::PipelineState& p = f.backend.pipelines.back();
    CHECK(p.alphaToCoverage);
    CHECK(p.blend[0].srcColor == BlendFactor::SrcAlphaSaturate);
    CHECK(p.stencilTest);
}

// A depth-only clear leaves the stencil alone: a clear inside the pass, not a load action; both
// planes cleared is the cheap load-action clear.
void planeClears() {
    for (int mode = 0; mode < 3; ++mode) {
        Fixture f;
        vpps4::DrawState c = f.state();
        c.primitive = vpps4::PrimitiveType::RectList;
        c.depth = {kBase + 0x60000, kBase + 0x70000, 128, 128, 3, 1, 0, 0, 1, 1, mode != 1, mode != 0, 0.25f, 7};
        f.renderer.draw(c);
        f.renderer.submit();
        CHECK(f.renderer.stats().skipped.empty() || f.renderer.stats().drawn == 0);
        size_t inPass = 0;
        for (const vp4gpu::Draw& d : f.backend.draws)
            if (d.clear) {
                ++inPass;
                CHECK(d.clear->depth == (mode != 1));
                CHECK(d.clear->stencil == (mode != 0));
                CHECK(d.clear->depthValue == 0.25f && d.clear->stencilValue == 7);
            }
        CHECK(inPass == (mode == 2 ? 0u : 1u));
    }
}

// A depth copy draw (colour off, force dirty+valid, other write base) copies the surface; a normal
// draw writing the other base makes the write address answer with the same texture.
void depthCopyDraw() {
    Fixture f;
    vpps4::DrawState d = f.state();
    d.color = {};
    d.depth = {kBase + 0x60000, 0, 128, 128, 3, 0, 0, 0, 1, 1, false, false, 1.0f, 0};
    d.depth.zWriteAddress = kBase + 0x90000;
    d.depth.forceZDirty = d.depth.forceZValid = true;
    f.renderer.draw(d);
    f.renderer.submit();
    CHECK(f.renderer.stats().notes.count("depth copy draw done as a texture copy") == 1);
    const auto copies = f.renderer.stats().copies;
    CHECK(copies == 1);
    // A writing draw to the same pair: the write address is the same texture, so a later copy is a no-op.
    vpps4::DrawState w = f.state();
    w.depth = d.depth;
    w.depth.forceZDirty = w.depth.forceZValid = false;
    w.depthStencil.depthTest = w.depthStencil.depthWrite = true;
    w.depth.zWriteAddress = kBase + 0xA0000;
    f.renderer.draw(w);
    const auto afterWrite = f.renderer.stats().copies;
    vpps4::DrawState c2 = d;
    c2.depth.zWriteAddress = kBase + 0xA0000;
    f.renderer.draw(c2);
    f.renderer.submit();
    CHECK(f.renderer.stats().copies == afterWrite);
}

void restartIndexDraw() {
    Fixture f;
    vpps4::DrawState d = f.state();
    d.primitive = vpps4::PrimitiveType::TriangleStrip;
    d.indexAddress = kBase + 0x3000;
    uint16_t* idx = reinterpret_cast<uint16_t*>(f.at(kBase + 0x3000));
    for (int k = 0; k < 8; ++k) idx[k] = uint16_t(k);
    idx[3] = 0x1234;
    d.count = 8;
    d.raster.primitiveRestart = true;
    d.raster.restartIndex = 0x1234;
    f.renderer.draw(d);
    f.renderer.submit();
    CHECK(f.renderer.stats().drawn == 1);
    CHECK(f.renderer.stats().notes.count("primitive restart index rewritten on the CPU") == 1);
}

int main(int argc, char** argv) {
    const std::string t = argc > 1 ? argv[1] : "";
    if (t == "clear_and_draw") clearAndDraw();
    else if (t == "textured") textured();
    else if (t == "linear_upload") linearUpload();
    else if (t == "tiled_upload") tiledUpload();
    else if (t == "mip_upload") mipUpload();
    else if (t == "write_back") writeBack();
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
    else if (t == "poly_offset") polyOffset();
    else if (t == "blend_factors") blendFactors();
    else if (t == "stencil_ops") stencilOps();
    else if (t == "scissors") scissors();
    else if (t == "clip_viewport") clipAndViewport();
    else if (t == "index_prep") indexPrep();
    else if (t == "lift_exports") liftExports();
    else if (t == "state_reaches_backend") stateReachesBackend();
    else if (t == "plane_clears") planeClears();
    else if (t == "depth_copy") depthCopyDraw();
    else if (t == "restart_index") restartIndexDraw();
    else return std::fprintf(stderr, "unknown test %s\n", t.c_str()), 2;
    return failures ? 1 : 0;
}
