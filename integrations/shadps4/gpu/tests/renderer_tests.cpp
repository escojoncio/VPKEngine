// vpps4::Renderer on the NullBackend: guest memory is a local page-aligned block, shaders are the
// lifter's test programs (gpu/shader/gcn/tests/lift/*.lift). Usage: vpps4_tests TEST
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "vpgpu/core/device.h"
#include "vpps4/fetch.h"
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
    vpgpu::NullBackend backend;
    std::vector<std::string> log;
    vpgpu::Device device{backend, 0, [this](const std::string& s) { log.push_back(s); }};
    std::vector<std::pair<uint64_t, vpir::Stage>> shaders;
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
    else return std::fprintf(stderr, "unknown test %s\n", t.c_str()), 2;
    return failures ? 1 : 0;
}
