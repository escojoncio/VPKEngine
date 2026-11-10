// vpps4::Renderer on the NullBackend: guest memory is a local page-aligned block, shaders are the
// lifter's test programs (gpu/shader/gcn/tests/lift/*.lift). Usage: vpps4_tests TEST
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "vpgpu/core/device.h"
#include "vpgpu/core/probe.h"
#include "vpmsl/msl.h"
#include "vpgcn/lift.h"
#include "vpps4/fetch.h"
#include "vpps4/fixed_function.h"
#include "vpps4/renderer.h"
#include "vpps4/scheduler.h"
#include <atomic>
#include <thread>
#include "vpps4/report.h"
#include "vpps4/tiling.h"
#include "../src/translation_cache.h"
#include <set>
#include <csignal>
#include <sys/mman.h>
#include <unistd.h>
namespace fs = std::filesystem;

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
    uint8_t userSgprs = 0, vgprCompCnt = 0;
    uint32_t vsOutCntl = 0;
};

Program load(const char* name) {
    std::ifstream f(std::string(VPPS4_LIFT_DIR) + "/" + name + ".lift");
    Program p;
    for (std::string line; std::getline(f, line);) {
        std::istringstream ls(line);
        std::string key;
        ls >> key;
        if (key == "user_sgprs") {
            unsigned v = 0;
            ls >> v;
            p.userSgprs = uint8_t(v);
        } else if (key == "vgpr_comp_cnt") {
            unsigned v = 0;
            ls >> v;
            p.vgprCompCnt = uint8_t(v);
        } else if (key == "vs_out_cntl") {
            std::string v;
            ls >> v;
            p.vsOutCntl = uint32_t(std::stoul(v, nullptr, 0));
        }
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
    std::function<bool(std::span<const vpps4::ComputeBuffer>, bool, bool)> computeBuffers;  // (set by the test that wants the hook)
    bool wantComputeBuffers = true;
    std::function<bool(const vpps4::SeedRequest&, vpps4::SeedImage&)> seed;  // (set by the test that wants the hook)
    std::vector<vpps4::Resolved> resolved;  // (what Hooks::resolved was given)
    vpps4::Renderer renderer{device, hooks()};
    Program vs = load("vs_fetch"), ps = load("ps_textured"), psHeavy = load("ps_heavy");

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
        h.computeBuffers = [this](std::span<const vpps4::ComputeBuffer> b, bool usesXor, bool complete) {
            return computeBuffers && computeBuffers(b, usesXor, complete);
        };
        h.computeBuffersWanted = [this] { return wantComputeBuffers; };
        h.resolved = [this](const vpps4::Resolved& r) { resolved.push_back(r); };
        h.seed = [this](const vpps4::SeedRequest& q, vpps4::SeedImage& i) { return seed && seed(q, i); };
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

// Hooks::resolved: what the draw resolved (targets with host formats, the size each buffer was bound with,
// the texture and sampler found, viewport and state), also for a draw that was left out, a dispatch and DMA.
void resolvedHook() {
    Fixture f;
    vpps4::DrawState d = f.state();
    f.renderer.draw(d);
    CHECK(f.resolved.size() == 1);
    const vpps4::Resolved& r = f.resolved.back();
    CHECK(r.kind == vpps4::Resolved::Kind::Draw && r.status == vpps4::Resolved::Status::Drawn);
    CHECK(r.color[0].used && r.color[0].address == kBase + 0x40000 && !r.color[0].hostFormat.empty());
    CHECK(!r.color[1].used && !r.depth.used);
    bool vertex = false;
    for (const vpps4::ResolvedBuffer& b : r.buffers)
        vertex |= b.address == kBase + 0x1000 && b.size == 16 * 64 && b.stride == 16 && b.stage == 0;
    CHECK(vertex);
    CHECK(r.textures.size() == 1 && r.textures[0].found && r.textures[0].stage == 1);
    CHECK(r.textures[0].image.address == kBase + 0x10000 && !r.textures[0].hostFormat.empty());
    CHECK(r.textures[0].hostWidth == 64 && r.textures[0].hostHeight == 64);
    CHECK(r.samplers.size() == 1 && r.samplers[0].addressU == 0 && r.samplers[0].magFilter == 0);
    CHECK(r.viewportValid && r.scissorValid && r.stateValid);
    // A V# whose records end before the buffer does: the size bound is the V#'s.
    uint32_t* vtable = f.at(kBase + 0x100);
    vtable[2] = 8;
    f.renderer.draw(d);
    bool shorter = false;
    for (const vpps4::ResolvedBuffer& b : f.resolved.back().buffers) shorter |= b.address == kBase + 0x1000 && b.size == 16 * 8;
    CHECK(shorter);
    // Left out: the reason.
    d.count = 0;
    f.renderer.draw(d);
    CHECK(f.resolved.back().status == vpps4::Resolved::Status::Skipped && f.resolved.back().reason == "empty draw");
    vpps4::DispatchState cs;
    f.renderer.dispatch(cs);
    CHECK(f.resolved.back().kind == vpps4::Resolved::Kind::Dispatch && f.resolved.back().status == vpps4::Resolved::Status::Skipped);
    f.renderer.fill(kBase + 0x2000, 64, 1);
    CHECK(f.resolved.back().kind == vpps4::Resolved::Kind::Fill && f.resolved.back().status == vpps4::Resolved::Status::Drawn);
    f.renderer.fill(0x10, 64, 1);
    CHECK(f.resolved.back().status == vpps4::Resolved::Status::Skipped);
    f.renderer.submit();
}

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
    // every draw leaves a trace: drawn, skipped or handled (the clear became a pass load action)
    CHECK(s.handled.count("rect list clear") == 1 && s.handled.at("rect list clear") == 1);
    CHECK(s.skipped.empty());
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

// Indirect draws of primitives Metal lacks, and multi-draws with a count in memory: a converter
// dispatch on the GPU timeline (nothing is read by the CPU), then the draw from its output.
void indirectConvert() {
    Fixture f;
    vpps4::DrawState d = f.state();
    d.indirectAddress = kBase + 0x3000;
    d.count = 1;
    for (vpps4::PrimitiveType p : {vpps4::PrimitiveType::QuadList, vpps4::PrimitiveType::TriangleFan,
                                   vpps4::PrimitiveType::LineLoop, vpps4::PrimitiveType::QuadStrip}) {
        d.primitive = p;
        f.renderer.draw(d);  // (the converter compiles on the first one on the null backend)
    }
    CHECK(f.renderer.stats().skipped.count("indirect draw of fan/quad/loop primitives") == 0);
    const uint64_t drawn = f.renderer.stats().drawn, dispatched = f.renderer.stats().dispatched;
    for (vpps4::PrimitiveType p : {vpps4::PrimitiveType::QuadList, vpps4::PrimitiveType::TriangleFan}) {
        d.primitive = p;
        f.renderer.draw(d);
    }
    CHECK(f.renderer.stats().drawn == drawn + 2);
    CHECK(f.renderer.stats().dispatched == dispatched + 2);  // one converter run per draw
    bool compute = false;
    for (auto& [id, st] : f.shaders) compute |= st == vpir::Stage::Compute;
    CHECK(compute);
    // Indexed, with a count buffer.
    d.primitive = vpps4::PrimitiveType::QuadList;
    d.indexAddress = kBase + 0x6000;
    d.count = 64;
    d.indirectCountAddress = kBase + 0x3800;
    d.drawIndex = 1;
    f.renderer.draw(d);
    CHECK(f.renderer.stats().drawn == drawn + 3);
    d.primitive = vpps4::PrimitiveType::TriangleList;  // a plain primitive with a count buffer: converter too
    f.renderer.draw(d);
    CHECK(f.renderer.stats().drawn == drawn + 4);
    d.indirectCountAddress = kBase + kSize - 2;  // count cut off by the end of memory
    f.renderer.draw(d);
    CHECK(f.renderer.stats().skipped.count("indirect count outside guest memory") == 1);
    f.renderer.submit();
    if (failures)
        for (const std::string& l : f.log) std::fprintf(stderr, "  %s\n", l.c_str());
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

// A host that tracks surface metadata is asked about the buffers a dispatch accesses, and whether the program has a
// bitwise xor (tiled address arithmetic); when it takes the dispatch over, nothing is recorded.
void computeBufferHook() {
    Fixture f;
    const Program cs = load("cs_bytes");
    const uint64_t base = kBase + 0x4000;
    auto dispatch = [&](const std::vector<uint32_t>& code) {
        vpps4::DispatchState d;
        d.cs.code = code;
        d.cs.address = kBase + 0x82000;
        d.cs.userSgprs = 4;
        d.cs.ldsBytes = 64;
        d.cs.workgroupSize = {64, 1, 1};
        d.cs.userData[0] = uint32_t(base);
        d.cs.userData[1] = uint32_t(base >> 32);  // V#: records of 1 byte
        d.cs.userData[2] = 0x1000;
        f.renderer.dispatch(d);
    };
    std::vector<vpps4::ComputeBuffer> seen;
    bool xorSeen = true, completeSeen = false, take = false;
    f.computeBuffers = [&](std::span<const vpps4::ComputeBuffer> b, bool usesXor, bool complete) {
        seen.assign(b.begin(), b.end());
        xorSeen = usesXor;
        completeSeen = complete;
        return take;
    };
    dispatch(cs.code);
    CHECK(f.renderer.stats().shaderErrors == 0);
    CHECK(seen.size() == 1 && seen[0].address == base && seen[0].written);
    CHECK(!xorSeen && completeSeen);
    CHECK(f.renderer.stats().handled.empty());
    // what the reference translation turns into a bitwise xor, in front of the same program (assembled with llvm-mc)
    const struct { const char* what; std::vector<uint32_t> words; } xors[] = {
        {"v_xor_b32", {0x3A000900u}},                  // v_xor_b32 v0, v0, v4
        {"v_add_f32 with a negated source", {0xD2060004u, 0x20020300u}},  // v_add_f32_e64 v4, -v0, v1
        {"ds_swizzle_b32", {0xD8D40000u, 0x04000000u}},                  // ds_swizzle_b32 v4, v0
    };
    std::vector<std::vector<uint32_t>> kept;  // (the programs live on: a freed vector's address is reused by the next, and the renderer
                                              // remembers programs by address until the CPU reports a write)
    kept.reserve(8);
    for (const auto& x : xors) {
        std::vector<uint32_t>& withXor = kept.emplace_back(x.words);
        withXor.insert(withXor.end(), cs.code.begin(), cs.code.end());
        seen.clear();
        dispatch(withXor);
        CHECK(f.renderer.stats().shaderErrors == 0);
        if (!(seen.size() == 1 && xorSeen)) std::fprintf(stderr, "  xor not seen: %s\n", x.what);
        CHECK(seen.size() == 1 && xorSeen);
    }
    // a float add without modifiers is no xor; one after s_endpgm is not part of the program
    std::vector<uint32_t> plain{0x06080300u};  // v_add_f32 v4, v0, v1
    plain.insert(plain.end(), cs.code.begin(), cs.code.end());
    plain.insert(plain.end(), {0x3A000900u, 0xD2060004u, 0x20020300u});  // (after the s_endpgm that ends cs.code)
    seen.clear();
    xorSeen = true;
    dispatch(plain);
    CHECK(f.renderer.stats().shaderErrors == 0);
    CHECK(seen.size() == 1 && !xorSeen);
    // a FLAT access reaches any guest memory (the guest map is a binding that is not a V#): the span is not complete
    std::vector<uint32_t> withFlat{0xdc300000u, 0x0a000002u};  // flat_load_dword v10, v[2:3]
    withFlat.insert(withFlat.end(), cs.code.begin(), cs.code.end());
    seen.clear();
    completeSeen = true;
    dispatch(withFlat);
    CHECK(f.renderer.stats().shaderErrors == 0);
    CHECK(seen.size() == 1 && !completeSeen);
    // the host takes it: no dispatch is recorded
    take = true;
    const uint64_t recorded = f.renderer.stats().dispatched;
    seen.clear();
    dispatch(cs.code);
    CHECK(seen.size() == 1 && !xorSeen);
    CHECK(f.renderer.stats().dispatched == recorded);
    CHECK(f.renderer.stats().handled.count("compute handled by the host") == 1);
    // the host has nothing to look for: it is not asked
    f.wantComputeBuffers = false;
    seen.clear();
    dispatch(cs.code);
    CHECK(seen.empty());
    CHECK(f.renderer.stats().handled.count("compute handled by the host") == 1);
    f.renderer.submit();
    if (failures)
        for (const std::string& l : f.log) std::fprintf(stderr, "  %s\n", l.c_str());
}

// FLAT access: the pointers in user data are guest GPU addresses, translated in the shader through the guest
// region map the renderer builds (GuestMap binding). On the null backend the buffers have fake device addresses.
void flatAccess() {
    Fixture f;
    // v2:3 <- s0:1, v4:5 <- s2:3, v10 <- [v2:3], [v4:5] <- v10, atomic add [v2:3] += v10 (returns), end
    static const uint32_t code[] = {0x7e040200u, 0x7e060201u, 0x7e080202u, 0x7e0a0203u,
                                    0xdc300000u, 0x0a000002u,  // flat_load_dword v10, v[2:3]
                                    0xdc700000u, 0x00000a04u,  // flat_store_dword v[4:5], v10
                                    0xdcc90000u, 0x0a000a02u,  // flat_atomic_add v10, v[2:3], v10 glc
                                    0xBF810000u};
    vpps4::DispatchState d;
    d.cs.code = code;
    d.cs.address = kBase + 0x82000;
    d.cs.userSgprs = 4;
    d.cs.workgroupSize = {64, 1, 1};
    const uint64_t src = kBase + 0x4000, dst = kBase + 0x4100;
    d.cs.userData[0] = uint32_t(src);
    d.cs.userData[1] = uint32_t(src >> 32);
    d.cs.userData[2] = uint32_t(dst);
    d.cs.userData[3] = uint32_t(dst >> 32);
    *f.at(src) = 42;
    f.renderer.dispatch(d);
    f.renderer.dispatch(d);  // (the pipeline compiles on the first one on the null backend)
    CHECK(f.renderer.stats().shaderErrors == 0);
    CHECK(f.renderer.stats().dispatched >= 1);
    CHECK(f.renderer.stats().skipped.empty());  // (no "needs buffer device addresses", no unbound map)
    // a pointer outside every guest region still runs (reads zeros, drops writes)
    d.cs.userData[1] = 0x7;
    f.renderer.dispatch(d);
    CHECK(f.renderer.stats().shaderErrors == 0);
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

// Dedicated staging (a copy source the upload ring has no room for) is kept by size class and reused once its frame is done: no
// buffer is created for it again. VPKENGINE_NO_STAGING_POOL=1: a buffer per request, as before.
void stagingPool() {
    Fixture f;
    const auto dedicated = [&] {
        size_t n = 0;
        for (const std::string& c : f.backend.calls) n += c.find("size=524288 ") != std::string::npos || c.find("size=311296 ") != std::string::npos;  // (class / page-rounded size of 300000 bytes)
        return n;
    };
    uint32_t v = 0x01020304u;
    const auto round = [&] {
        for (int i = 0; i < 60; ++i) CHECK(f.renderer.fill(kBase, 300000, v += 0x01010100u));  // (not a byte pattern: through staging)
        f.renderer.submit();
    };
    round();
    const size_t made = dedicated();
    CHECK(made >= 1);  // (the 16 MiB ring holds the first 53 fills; the rest need buffers of their own)
    for (int i = 0; i < 12; ++i) {  // their frames are done
        CHECK(f.renderer.fill(kBase + 0x2000, 256, 0));
        f.renderer.submit();
    }
    round();
    CHECK(dedicated() == made);
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
    // The memo gives the same answers (first as a miss, then as a hit, then for other user data).
    vpps4::FetchAddressMemo memo;
    for (const char* name : {"vs_fetch", "vs_transform", "vs_cbuf_dword", "vs_cbuf_x16"}) {
        Program p = load(name);
        for (unsigned round = 0; round < 3; ++round)
            for (uint8_t sgprs : {uint8_t(16), uint8_t(6), uint8_t(0)}) {
                ud[4] = 0x2000u * (round + 1);
                CHECK(memo.address(p.code, ud, sgprs) == vpps4::fetchAddress(p.code, ud, sgprs));
            }
    }
    // Other code at the same place is not answered from the old walk.
    std::vector<uint32_t> code = vs.code;
    CHECK(memo.address(code, ud, 16) == vpps4::fetchAddress(code, ud, 16));
    code[0] ^= 0x1;
    CHECK(memo.address(code, ud, 16) == vpps4::fetchAddress(code, ud, 16));
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

// A frame that needs more than the upload ring skips what does not fit once; the next ring is bigger.
void uploadRingGrows() {
    Fixture f;
    vpps4::DrawState d = f.state();
    d.primitive = vpps4::PrimitiveType::TriangleFan;
    d.count = 1500000;  // 12 bytes of indices per triangle: over 16 MB, served by a dedicated chunk in the same draw
    f.renderer.draw(d);
    CHECK(f.renderer.stats().skipped.count("upload ring full") == 0);
    CHECK(f.renderer.stats().drawn == 1);
    f.renderer.submit();
    const uint64_t drawn = f.renderer.stats().drawn;
    f.renderer.draw(d);
    CHECK(f.renderer.stats().drawn == drawn + 1);
    f.renderer.submit();
}

// Live ring memory stays bounded over many frames of large uploads once the frames complete.
void uploadRingBounded() {
    Fixture f;
    f.backend.autoComplete = true;
    vpps4::DrawState d = f.state();
    d.primitive = vpps4::PrimitiveType::TriangleFan;
    d.count = 1500000;
    uint64_t peak = 0;
    for (int frame = 0; frame < 40; ++frame) {
        f.renderer.draw(d);
        f.renderer.submit();
        std::map<unsigned, uint64_t> live;
        for (const std::string& c : f.backend.calls) {
            unsigned id = 0, n = 0;
            unsigned long long size = 0;
            if (std::sscanf(c.c_str(), "createBuffer %u size=%llu%n", &id, &size, &n) == 2) live[id] = size;
            else if (std::sscanf(c.c_str(), "destroyBuffer %u", &id) == 1) live.erase(id);
        }
        uint64_t total = 0;
        for (const auto& [id, size] : live) total += size;
        peak = std::max(peak, total);
    }
    CHECK(f.renderer.stats().drawn == 40);
    CHECK(f.renderer.stats().skipped.count("upload ring full") == 0);
    CHECK(peak < (uint64_t(1) << 30));
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

// ZPASS_DONE: results are written to guest memory only once the frames holding the draws finished.
void occlusion() {
    Fixture f;
    const uint64_t addr = kBase + 0x4000;
    uint64_t* g = reinterpret_cast<uint64_t*>(f.at(addr));
    CHECK(!f.renderer.zpassDone(kBase + kSize - 8, 4));  // a range outside guest memory is not counted
    CHECK(f.renderer.zpassDone(addr, 3));
    CHECK(f.renderer.occlusionPending(addr + 16, 8));
    CHECK(!f.renderer.occlusionPending(addr + 0x100, 8));
    f.renderer.drainOcclusion(false);  // NullBackend: no frame finished yet
    CHECK(f.renderer.occlusionPending(addr, 8) && g[0] == 0);
    f.renderer.drainOcclusion(true);   // the caller waited for the GPU
    CHECK(!f.renderer.occlusionPending(addr, 48));
    CHECK(g[0] == (1ull << 63) && g[2] == (1ull << 63) && g[4] == (1ull << 63));
    // Many unanswered snapshots do not overrun the slot ring: the caller is told to fake instead.
    size_t taken = 0;
    for (int i = 0; i < 5000; ++i) taken += f.renderer.zpassDone(addr, 1);
    CHECK(taken > 4000 && taken < 5000);
}

// Skips, approximations and translation errors reach the host's sink, one event each time.
void reportSink() {
    struct Seen {
        std::vector<std::string> events;
    } seen;
    vpps4::setReportSink(
        [](void* user, const char* category, const char* key, const char* detail, uint64_t n) {
            for (uint64_t i = 0; i < n; ++i)
                static_cast<Seen*>(user)->events.push_back(std::string(category) + "|" + key + "|" + detail);
        },
        &seen);
    {
        Fixture f;
        vpps4::DrawState d = f.state();
        static const uint32_t junk[] = {0xffffffff, 0xffffffff};
        d.vs.code = junk;
        f.renderer.draw(d);
        f.renderer.draw(d);
        vpps4::DrawState e = f.state();
        e.indirectAddress = kBase + kSize - 8;
        f.renderer.draw(e);
        vpps4::DrawState c = f.state();
        c.color[0].format = 63;  // a colour format the renderer does not know
        f.renderer.draw(c);
        f.renderer.submit();  // (the per-draw reasons are counted and reach the sink at the submit)
        auto count = [&](const std::string& prefix) {
            size_t n = 0;
            for (const std::string& e : seen.events) n += e.rfind(prefix, 0) == 0;
            return n;
        };
        CHECK(count("draw-skip|vertex shader not translated|") == 2);  // every time, not once
        CHECK(count("shader|vs: ") == 1);                               // translated once
        CHECK(count("draw-skip|indirect arguments outside guest memory|") == 1);
        CHECK(count("unsupported|colour target format 63/") == 1);
    }
    vpps4::setReportSink(nullptr);
    const size_t before = seen.events.size();
    Fixture g;
    vpps4::DrawState d = g.state();
    d.count = 0;
    g.renderer.draw(d);
    g.renderer.submit();
    CHECK(seen.events.size() == before);  // sink removed
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
        // Multisampled colour and depth, plus the colour's single-sample picture (what a sampler or the
        // present sees; the pass resolves into it).
        size_t ms = 0, all = 0;
        for (const std::string& c : f.backend.calls)
            if (c.rfind("createTexture", 0) == 0 && c.find("128x128") != std::string::npos) {
                ++all;
                if (c.size() >= want.size() && c.compare(c.size() - want.size(), want.size(), want) == 0) ++ms;
            }
        CHECK(ms == 2);
        CHECK(all == 3);
        CHECK(f.renderer.stats().drawn == 1);
    }
    Fixture f;
    f.renderer.setMaxSamples(3);
    CHECK(f.renderer.maxSamples() == 2);
    f.renderer.setMaxSamples(0);
    CHECK(f.renderer.maxSamples() == 1);
}

// Resolve at 1 sample: a texture copy into CB1, which becomes what a flip at its address presents;
// mismatched formats are skipped, not approximated. With host MSAA on it is a resolve pass.
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
    // limit 8 (host MSAA on): the source is multisampled and young (its single-sample picture is still
    // kept up to date), so the resolve is a copy of that picture; later ones are done by passes (msaa_tests).
    Fixture g;
    g.renderer.resolve(src, dst);
    CHECK(g.renderer.stats().copies == 1);
    CHECK(g.renderer.stats().skipped.empty());
    CHECK(bool(g.renderer.colorTargetAt(kBase + 0x70000)));
}

// The title's resolve of a multisampled draw into another target (the eye texture) is never dropped by the
// lifetime planner, however long nobody reads the destination, and the destination is the target a flip finds.
void resolveRequired() {
    Fixture f;
    vpps4::ColorTarget src = f.state().color[0], dst = src;
    src.samples = 4;
    dst.address = kBase + 0x70000;
    for (int n = 0; n < 80; ++n) {
        vpps4::DrawState d = f.state();
        d.color[0].samples = 4;
        f.renderer.draw(d);
        f.renderer.resolve(src, dst);
        f.renderer.submit();
    }
    for (const std::string& l : f.log) CHECK(l.find("resolve_dropped") == std::string::npos);
    CHECK(bool(f.renderer.colorTargetAt(dst.address)));
    CHECK(f.renderer.describeAt(dst.address, 4).find("colour@") != std::string::npos);
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

using vpgpu::BlendFactor;

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
    // DB_FMT_CNTL names the unit: 24-bit fixed on a float host plane is 1:1, 16-bit is 256 coarser.
    r.dbFmtValid = true, r.dbBits = 24;
    CHECK(vpps4::resolveDepthBias(r, vpps4::PrimitiveType::TriangleList, 3, false).constant == 3);
    r.dbBits = 16;
    CHECK(vpps4::resolveDepthBias(r, vpps4::PrimitiveType::TriangleList, 3, false).constant == 3 * 256);
    r.dbIsFloat = true;
    CHECK(vpps4::resolveDepthBias(r, vpps4::PrimitiveType::TriangleList, 3, false).constant == 3);
    r.dbFmtValid = false, r.dbIsFloat = false;
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

// A viewport transform that is switched off maps the whole target, not a 2x2 viewport.
void viewportRects() {
    vpps4::Viewport v{64, 100, -32, 50, 1, 0};
    auto r = vpps4::hostViewportRect(v, 256, 128);
    CHECK(r.valid && !r.flipped && !r.wholeTarget && r.x == 36 && r.y == 18 && r.width == 128 && r.height == 64);
    v.yScale = 32;
    CHECK(vpps4::hostViewportRect(v, 256, 128).flipped);
    v.yScale = -32;
    v.xScreen = v.yScreen = true;
    r = vpps4::hostViewportRect(v, 256, 128);
    CHECK(r.valid && r.wholeTarget && !r.flipped && r.x == 0 && r.y == 0 && r.width == 256 && r.height == 128);
    v.yScreen = false;  // only x off
    r = vpps4::hostViewportRect(v, 256, 128);
    CHECK(r.x == 0 && r.width == 256 && r.y == 18 && r.height == 64);
    CHECK(!vpps4::hostViewportRect(vpps4::Viewport{}, 256, 128).valid);
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
    CHECK(r.front.pass == vpgpu::StencilOp::Replace && r.refFront == 0xFF && r.exact);
    // REPLACE_OP writes STENCILOPVAL.
    ds.front.zPass = 4, ds.front.opVal = 0x42;
    r = vpps4::resolveStencil(ds, false);
    CHECK(r.front.pass == vpgpu::StencilOp::Replace && r.refFront == 0x42 && r.exact);
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
    CHECK(r.exact && r.front.pass == vpgpu::StencilOp::Keep);
    // Logic operations with an all-ones / zero value.
    ds.front.zPass = 12, ds.front.opVal = 0xFF;  // XOR with ones = invert
    CHECK(vpps4::resolveStencil(ds, false).front.pass == vpgpu::StencilOp::Invert);
    ds.front.zPass = 10, ds.front.opVal = 0;     // AND with zero = zero
    CHECK(vpps4::resolveStencil(ds, false).front.pass == vpgpu::StencilOp::Zero);
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

void writeMaskSwap() {
    const uint8_t id[4] = {0, 1, 2, 3}, bgra[4] = {2, 1, 0, 3}, abgr[4] = {3, 2, 1, 0}, rg1[4] = {0, 1, 5, 5};
    CHECK(vpps4::permuteWriteMask(0xF, id) == 0xF && vpps4::permuteWriteMask(0x5, id) == 0x5);
    CHECK(vpps4::permuteWriteMask(0x1, bgra) == 0x4);  // red written into storage channel 2
    CHECK(vpps4::permuteWriteMask(0x8, abgr) == 0x1);
    CHECK(vpps4::permuteWriteMask(0x3, rg1) == 0xF);   // constant channels follow "anything written"
    CHECK(vpps4::permuteWriteMask(0x0, rg1) == 0x0);
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
    // The generic scissor has its own disable bit: with it set the generic one stays unshifted.
    s.generic = {12, 12, 2000, 2000};
    CHECK(vpps4::combineScissor(s).left == 15);  // (window 10+5 beats generic 12)
    s.generic = {20, 20, 2000, 2000};
    CHECK(vpps4::combineScissor(s).left == 20);
    s.genericOffsetDisable = false;
    CHECK(vpps4::combineScissor(s).left == 25);
    s.generic = {0, 0, 2000, 2000};
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
    CHECK(vpps4::indicesNeedPrep(d));  // restart off: the host's implicit restart must not trigger
    {
        const uint16_t lit[] = {1, 2, 0xFFFF, 3};
        std::vector<uint8_t> o;
        bool wide = false;
        vpps4::prepareIndices(d, std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(lit), sizeof lit), o, &wide);
        uint32_t w[4];
        CHECK(wide && o.size() == 16);
        std::memcpy(w, o.data(), 16);
        CHECK(w[2] == 0xFFFF && w[3] == 3);
        const uint16_t none[] = {1, 2, 3};
        vpps4::prepareIndices(d, std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(none), sizeof none), o, &wide);
        CHECK(!wide && o.size() == 6);
    }
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
    return vpmsl::emitMsl(l.module).shader.source;
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
    // Dual-source blending with a shader that exports only MRT0: the second source is still written (zero).
    static const uint32_t one[] = {0xF800080F, 0x00000000, 0xBF810000};
    vpgcn::LiftInput single;
    single.stage = vpgcn::ShaderStage::Pixel;
    single.code = one;
    single.dualSource = true;
    CHECK(sourceOf(single).find("index(1)") != std::string::npos);
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
    const vpgpu::Draw& w = f.backend.draws.back();
    CHECK(w.depthBias == 2 && w.depthBiasSlope == 1);
    CHECK(w.depthClamp && w.wireframe);
    CHECK(w.stencilRef == 5 && w.stencilRefBack == 6);
    CHECK(w.scissor.x == 8 && w.scissor.y == 8 && w.scissor.width == 92 && w.scissor.height == 82);
    CHECK(!f.backend.pipelines.empty());
    if (f.backend.pipelines.empty()) return;
    const vpgpu::PipelineState& p = f.backend.pipelines.back();
    // One sample: no host alpha-to-coverage; the pixel shader does the alpha test.
    CHECK(!p.alphaToCoverage);
    CHECK(f.renderer.stats().notes.count("shader: alpha to coverage with one sample done as an alpha test at 0.5") == 1);
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
        for (const vpgpu::Draw& d : f.backend.draws)
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

// A strip with restart off over indices with no all-ones value is scanned once: the scan result is kept per
// (address, size, tracker version) and a CPU write to the buffer makes it scan again.
void indexScanMemo() {
    Fixture f;
    vpps4::DrawState d = f.state();
    d.primitive = vpps4::PrimitiveType::TriangleStrip;
    d.indexAddress = kBase + 0x3000;
    uint16_t* idx = reinterpret_cast<uint16_t*>(f.at(kBase + 0x3000));
    for (int k = 0; k < 8; ++k) idx[k] = uint16_t(k);
    d.count = 8;
    const char* scanned = "primitive restart index rewritten on the CPU";
    f.renderer.draw(d);
    CHECK(f.renderer.stats().notes.count(scanned) == 1);
    f.renderer.draw(d);
    f.renderer.draw(d);
    CHECK(f.renderer.stats().notes.count(scanned) == 1);  // the later draws used the buffer as it is
    idx[3] = 0xFFFF;                                       // the CPU writes an all-ones literal
    f.renderer.invalidate(kBase + 0x3000, 16);
    f.renderer.draw(d);
    CHECK(f.renderer.stats().notes.count(scanned) + f.renderer.stats().notes.count("strip indices widened to 32 bits (the all-ones index is a vertex, not a restart)") == 2);
    f.renderer.submit();
    CHECK(f.renderer.stats().drawn == 4);
}

// The host gave up on a write back that never comes: the image stops owning the range.
void abandonOwners() {
    Fixture f;
    vpps4::DrawState d = f.state();
    f.renderer.draw(d);
    const uint64_t a = d.color[0].address;
    const uint64_t bytes = 128 * 128 * 4;
    CHECK(f.renderer.tracker().ownerImage(a, bytes) != 0);
    f.renderer.abandonOwners(a, bytes);
    CHECK(f.renderer.tracker().ownerImage(a, bytes) == 0);
    f.renderer.submit();
}

// ---- Geometry pipeline, RectList, adjacency, patches ---------------------------------------------

// Triangles of vertices (position, colour) at kBase + 0x1000, read by the ES through a V# in user data.
void geometryState(Fixture& f, vpps4::DrawState& d, const char* es, const char* gs, const char* copy, uint32_t verts) {
    float* vb = reinterpret_cast<float*>(f.at(kBase + 0x1000));
    for (uint32_t i = 0; i < verts; ++i) {
        const float v[8] = {0.01f * float(i), -0.5f, 0.25f, 1.0f, float(i), 0.5f, 0.25f, 1.0f};
        std::memcpy(vb + 8 * i, v, sizeof v);
    }
    Program pe = load(es), pg = load(gs), pc = load(copy);
    static std::vector<Program> keep;  // the spans in DrawState point at the code
    keep = {pe, pg, pc};
    d.es.code = keep[0].code;
    d.es.address = kBase + 0x82000;
    d.es.userSgprs = pe.userSgprs;
    d.es.userData[0] = uint32_t(kBase + 0x1000);
    d.es.userData[1] = uint32_t((kBase + 0x1000) >> 32) | (32u << 16);
    d.es.userData[2] = verts;
    d.es.userData[3] = (7u << 12) | (14u << 15);
    d.es.vgprCompCnt = pe.vgprCompCnt;
    d.gs.code = keep[1].code;
    d.gs.address = kBase + 0x83000;
    d.gs.userSgprs = pg.userSgprs;
    d.vs.code = keep[2].code;
    d.vs.fetch = {};
    d.vs.userSgprs = pc.userSgprs;
    d.vs.vgprCompCnt = pc.vgprCompCnt;
    d.vs.vsOutCntl = pc.vsOutCntl;
    d.geom = {true, 8, 9, 6, 2, 1};
}

void geometryShader() {
    Fixture f;
    vpps4::DrawState d = f.state();
    geometryState(f, d, "es_basic", "gs_stereo", "copy_stereo", 6);
    d.ps.code = f.ps.code;
    d.color[0].sliceCount = 2;
    d.count = 6;
    f.renderer.draw(d);
    f.renderer.submit();
    const vpps4::Stats& s = f.renderer.stats();
    if (s.drawn != 1)
        for (const auto& [k, n] : s.skipped) std::fprintf(stderr, "  skip %s x%llu\n", k.c_str(), (unsigned long long)n);
    CHECK(s.drawn == 1);
    CHECK(s.dispatched == 2);
    CHECK(s.shaderErrors == 0);
    CHECK(s.notes.count("geometry shader expanded by compute (ES, GS) before the draw") == 1);
    // ES + GS as compute shaders, the copy shader as the vertex stage.
    size_t compute = 0, vertex = 0;
    for (const auto& sh : f.shaders) {
        compute += sh.second == vpir::Stage::Compute;
        vertex += sh.second == vpir::Stage::Vertex;
    }
    CHECK(compute == 2);
    CHECK(vertex == 1);
    if (failures)
        for (const std::string& l : f.log) std::fprintf(stderr, "  %s\n", l.c_str());
}

// Geometry expansions of several draws of one pass run before it: the pass is not split.
void geometryHoist() {
    Fixture f;
    vpps4::DrawState plain = f.state();
    vpps4::DrawState g = f.state();
    geometryState(f, g, "es_basic", "gs_stereo", "copy_stereo", 6);
    g.ps.code = f.ps.code;
    g.color[0].sliceCount = 2;
    g.count = 6;
    // The same targets as the plain draws (a two-slice view of the same surface is another pass).
    plain.color[0].sliceCount = 2;
    plain.vs.vsOutCntl = 1u << 18;  // (layered, as the geometry draws are)
    f.renderer.draw(plain);
    f.renderer.draw(g);
    f.renderer.draw(plain);
    f.renderer.draw(g);
    f.renderer.submit();
    const vpps4::Stats& s = f.renderer.stats();
    if (s.drawn != 4)
        for (const auto& [k, n] : s.skipped) std::fprintf(stderr, "  skip %s x%llu\n", k.c_str(), (unsigned long long)n);
    CHECK(s.drawn == 4);
    CHECK(s.dispatched == 4);
    CHECK(s.gsHoisted == 2 && s.gsSplit == 0);
    CHECK(s.passes == 1);
    CHECK(f.backend.passes.size() == 1);
    if (!f.backend.passes.empty()) CHECK(f.backend.draws.size() == 4);
    // The dispatches come first in the recorded order: one compute encoder, then one render encoder.
    bool oneFrame = false;
    for (const std::string& c : f.backend.calls)
        if (c.rfind("submit", 0) == 0) oneFrame = oneFrame || c.find("encoders=2") != std::string::npos;
    CHECK(oneFrame);
    if (failures)
        for (const std::string& l : f.log) std::fprintf(stderr, "  %s\n", l.c_str());
}

// The same measurement with a realistic pixel stage (ps_heavy: 12 textures, one sampler, 6 buffers)
// and state that changes between draws: user data, the first texture (16 different T#s), depth
// function, culling, blend and viewport. Prints the CPU cost per draw.
void drawBenchHeavy() {
    Fixture f;
    vpps4::DrawState d = f.state();
    constexpr uint64_t kTable = kBase + 0x400, kTex = kBase + 0x10000, kBufs = kBase + 0x70000;
    uint32_t* t = f.at(kTable);
    auto tsharp = [&](uint32_t* w, uint64_t a) {
        w[0] = uint32_t(a >> 8);
        w[1] = uint32_t(a >> 40) | (10u << 20);
        w[2] = 63 | (63u << 14);
        w[3] = 4 | (5u << 3) | (6u << 6) | (7u << 9) | (8u << 20) | (9u << 28);
        w[4] = 63u << 13;
    };
    for (int i = 0; i < 12; ++i) tsharp(t + 8 * i, kTex + 0x4000 * i);
    t[96] = 0;  // S#
    for (int j = 0; j < 6; ++j) {
        const uint64_t a = kBufs + 0x1000 * j;
        uint32_t* v = t + 100 + 4 * j;
        v[0] = uint32_t(a);
        v[1] = uint32_t(a >> 32) | (4u << 16);
        v[2] = 64;
        v[3] = (4u << 12) | (4u << 15);  // 32 float
    }
    d.color[0].address = kBase + 0xA0000;
    d.ps.code = f.psHeavy.code;
    d.ps.userData[0] = uint32_t(kTable);
    d.depthStencil.depthTest = true;
    auto vary = [&](int i) {
        tsharp(t, kTex + 0x4000 * (i % 12) + (i % 16 >= 12 ? 0x1000 * (i % 16 - 11) : 0));
        d.vs.userData[0] = uint32_t(i & 7);
        d.depthStencil.depthFunc = uint8_t(1 + (i & 3));
        d.raster.cullBack = i & 1;
        d.blend[0].enable = (i >> 1) & 1;
        d.viewport.xOffset = float(64 + (i & 3));
    };
    d.depth = {kBase + 0x60000, 0, 128, 128, 3, 0, 0, 0, 1, 1, false, false, 1.0f, 0};
    for (int k = 0; k < 32; ++k) vary(k), f.renderer.draw(d);  // warm the caches
    f.renderer.submit();
    f.renderer.gpuIdle();
    CHECK(f.renderer.stats().shaderErrors == 0);
    CHECK(f.renderer.stats().drawn >= 32);
    const int kFrames = std::getenv("VPPS4_BENCH_FRAMES") ? std::atoi(std::getenv("VPPS4_BENCH_FRAMES")) : 16;
    constexpr int kDraws = 1600;
    const auto t0 = std::chrono::steady_clock::now();
    for (int fr = 0; fr < kFrames; ++fr) {
        for (int i = 0; i < kDraws; ++i) {
            vary(i);
            f.renderer.draw(d);
            if (i % 100 == 99) f.renderer.submit();
        }
        f.renderer.gpuIdle();
        f.backend.calls.clear();
    }
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    std::printf("draw_bench_heavy: %.2f us/draw (%d draws)\n", us / (kFrames * kDraws), kFrames * kDraws);
    CHECK(f.renderer.stats().drawn >= uint64_t(kFrames * kDraws));
    if (std::getenv("VPPS4_BENCH_STATS"))
        std::printf("  passes=%llu copies=%llu uploads=%llu textures=%llu\n", (unsigned long long)f.renderer.stats().passes,
                    (unsigned long long)f.renderer.stats().copies, (unsigned long long)f.renderer.stats().uploads,
                    (unsigned long long)f.renderer.stats().textures);
    if (failures)
        for (const std::string& l : f.log) std::fprintf(stderr, "  %s\n", l.c_str());
}

void geometryIndexed() {
    Fixture f;
    vpps4::DrawState d = f.state();
    geometryState(f, d, "es_basic", "gs_stereo", "copy_stereo", 6);
    d.ps.code = f.ps.code;
    uint16_t* ib = reinterpret_cast<uint16_t*>(f.at(kBase + 0x4000));
    for (unsigned i = 0; i < 6; ++i) ib[i] = uint16_t(5 - i);
    d.indexAddress = kBase + 0x4000;
    d.count = 6;
    d.instances = 2;
    f.renderer.draw(d);
    f.renderer.submit();
    CHECK(f.renderer.stats().drawn == 1);
    CHECK(f.renderer.stats().dispatched == 2);
}

// Geometry scratch slabs of finished frames are handed back, so a long run of heavy draws does not exhaust the limit.
void geometryScratchReclaim() {
    for (int pass = 0; pass < 2; ++pass) {
        Fixture f;
        f.backend.autoComplete = pass == 0;
        vpps4::DrawState d = f.state();
        geometryState(f, d, "es_basic", "gs_stereo", "copy_stereo", 6);
        d.ps.code = f.ps.code;
        d.count = 250000;
        for (int frame = 0; frame < 2; ++frame) {
            for (int i = 0; i < (frame == 0 ? 22 : 5); ++i) f.renderer.draw(d);
            f.renderer.submit();
        }
        const uint64_t exhausted = f.renderer.stats().skipped.count("geometry scratch memory exhausted")
                                       ? f.renderer.stats().skipped.at("geometry scratch memory exhausted") : 0;
        if (pass == 0) CHECK(exhausted == 0 && f.renderer.stats().drawn == 27);
        else CHECK(exhausted > 0);  // frames still in flight keep their memory: the limit stays
        if (failures)
            for (const auto& [k, n] : f.renderer.stats().skipped) std::fprintf(stderr, "  pass %d skip %s x%llu\n", pass, k.c_str(), (unsigned long long)n);
    }
}

void geometryBadPrimitive() {
    Fixture f;
    vpps4::DrawState d = f.state();
    geometryState(f, d, "es_basic", "gs_stereo", "copy_stereo", 6);
    d.primitive = vpps4::PrimitiveType::QuadList;
    f.renderer.draw(d);
    CHECK(f.renderer.stats().drawn == 0);
    CHECK(f.renderer.stats().notes.count("geometry shader with an unsupported input primitive type: draw left out") == 1);
}

void rectList() {
    Fixture f;
    vpps4::DrawState d = f.state();
    d.primitive = vpps4::PrimitiveType::RectList;
    d.count = 6;  // two rectangles
    f.renderer.draw(d);
    f.renderer.submit();
    if (f.renderer.stats().drawn != 1)
        for (const auto& [k, n] : f.renderer.stats().skipped) std::fprintf(stderr, "  skip %s x%llu\n", k.c_str(), (unsigned long long)n);
    CHECK(f.renderer.stats().drawn == 1);
    CHECK(f.renderer.stats().notes.count("RectList drawn as two triangles per rectangle (fourth vertex derived)") == 1);
}

void adjacency() {
    const vpps4::PrimitiveType kinds[4] = {vpps4::PrimitiveType::LineListAdj, vpps4::PrimitiveType::LineStripAdj,
                                           vpps4::PrimitiveType::TriangleListAdj, vpps4::PrimitiveType::TriangleStripAdj};
    const uint32_t counts[4] = {8, 6, 12, 10};
    for (int k = 0; k < 4; ++k) {
        Fixture f;
        vpps4::DrawState d = f.state();
        d.primitive = kinds[k];
        d.count = counts[k];
        f.renderer.draw(d);
        f.renderer.submit();
        CHECK(f.renderer.stats().drawn == 1);
        CHECK(f.renderer.stats().notes.count("adjacency vertices dropped (no geometry shader)") == 1);
    }
}

void patch() {
    Fixture f;
    vpps4::DrawState d = f.state();
    d.primitive = vpps4::PrimitiveType::Patch;
    d.count = 3;
    f.renderer.draw(d);
    CHECK(f.renderer.stats().drawn == 0);
    CHECK(f.renderer.stats().notes.count("tessellation patch draw (HS/DS stages) left out") == 1);
}

// The persistent cache: a second run with the same cache folder translates nothing (the shaders come
// from the file and reach the host's hook), a torn tail is repaired, and a file from another build
// (other fingerprint) is replaced.
void translationCache() {
    const fs::path dir = fs::temp_directory_path() / ("vpps4_tc_" + std::to_string(std::rand()) + "_" + std::to_string(std::time(nullptr)));
    fs::remove_all(dir);
    {
        Fixture f;
        f.renderer.setCacheRoot(dir);
        f.renderer.waitCache();
        CHECK(f.renderer.stats().shadersCached == 0);
        f.renderer.draw(f.state());
        f.renderer.submit();
        CHECK(f.renderer.stats().shaders == 2);
        CHECK(f.renderer.stats().drawn == 1);
        f.renderer.flushCache();
    }
    const fs::path file = dir / "translations.vpsc";
    CHECK(fs::exists(file));
    const auto size = fs::file_size(file);
    {  // second run: nothing is translated, the hook still gets both shaders
        Fixture f;
        f.renderer.setCacheRoot(dir);
        f.renderer.waitCache();
        CHECK(f.renderer.stats().shadersCached == 2);
        CHECK(f.shaders.size() == 2);
        f.renderer.draw(f.state());
        f.renderer.submit();
        CHECK(f.renderer.stats().shaders == 0);
        CHECK(f.renderer.stats().drawn == 1);
        f.renderer.flushCache();
    }
    // (nothing new to append but the touches: the two translations the run used, 13 bytes each)
    const auto touched = fs::file_size(file);
    CHECK(touched == size + 2 * 13);
    {  // a crash in the middle of a write leaves a torn tail: dropped, the rest kept
        std::ofstream o(file, std::ios::binary | std::ios::app);
        const char junk[] = {2, 100, 0, 0, 0, 1, 2, 3};
        o.write(junk, sizeof junk);
    }
    {
        Fixture f;
        f.renderer.setCacheRoot(dir);
        f.renderer.waitCache();
        CHECK(f.renderer.stats().shadersCached == 2);
    }
    CHECK(fs::file_size(file) == touched);
    {  // another build: the fingerprint in the header differs
        std::fstream o(file, std::ios::binary | std::ios::in | std::ios::out);
        o.seekp(14);
        o.put('#');
    }
    {
        Fixture f;
        f.renderer.setCacheRoot(dir);
        f.renderer.waitCache();
        CHECK(f.renderer.stats().shadersCached == 0);
        f.renderer.draw(f.state());
        f.renderer.submit();
        CHECK(f.renderer.stats().shaders == 2);
        f.renderer.flushCache();
    }
    CHECK(fs::file_size(file) == size);
    fs::remove_all(dir);
}

fs::path tcDir(const char* what) {
    return fs::temp_directory_path() / (std::string("vpps4_tc_") + what + std::to_string(std::rand()) + "_" + std::to_string(std::time(nullptr)));
}

vpps4::tcache::TranslationRec tcRec(uint64_t key2, size_t bytes) {
    vpps4::tcache::TranslationRec r;
    r.key1 = 1, r.key2 = key2, r.id = key2, r.stage = 1;
    r.shader.source = std::string(bytes, 'x');
    return r;
}

// Records that repeat a key are kept once; keys already in the file are not appended again; the start
// rewrites the file only when the waste passes the slack.
void translationCacheDedupe() {
    namespace tc = vpps4::tcache;
    const fs::path dir = tcDir("dedupe");
    const fs::path file = dir / "translations.vpsc";
    Fixture f;
    auto open = [&](tc::Limits l = {}) { return std::make_unique<tc::Worker>(dir, tc::buildFingerprint(), f.device.pipelines(), l); };
    auto fill = [&](tc::Worker& w) {
        w.append(tc::ProgramRec{1, {}});
        for (uint64_t k = 1; k <= 5; ++k) w.append(tcRec(k, 1000));
        w.append(tc::FailureRec{1, 99, 1, "bad"});
        w.append(tc::FailureRec{2, 98, 2, "worse"});
    };
    {
        auto w = open();
        fill(*w);
        fill(*w);  // (twice before it is written: once)
        w->flush();
    }
    const auto size = fs::file_size(file);
    {
        auto w = open();
        w->waitLoaded();
        fill(*w);  // (all in the file already)
        w->flush();
        const tc::LoadStats st = w->loadStats();
        CHECK(st.programs == 1 && st.translations == 5 && st.failures == 2 && st.duplicates == 0 && !st.compacted);
    }
    CHECK(fs::file_size(file) == size);
    {  // a file with every record three times over
        std::ofstream o(file, std::ios::binary | std::ios::app);
        for (int i = 0; i < 2; ++i) {
            const std::string p = tc::encode(tc::ProgramRec{1, {}});
            o.write(p.data(), std::streamsize(p.size()));
            for (uint64_t k = 1; k <= 5; ++k) {
                const std::string t = tc::encode(tcRec(k, 1000));
                o.write(t.data(), std::streamsize(t.size()));
            }
            const std::string a = tc::encode(tc::FailureRec{1, 99, 1, "bad"}), b = tc::encode(tc::FailureRec{2, 98, 2, "worse"});
            o.write(a.data(), std::streamsize(a.size()));
            o.write(b.data(), std::streamsize(b.size()));
        }
    }
    const auto fat = fs::file_size(file);
    CHECK(fat > 2 * size);
    {  // below the slack: left alone, the duplicates skipped at load
        tc::Limits l;
        l.slackBytes = 1 << 20;
        l.slackFraction = 0.99;
        auto w = open(l);
        const tc::LoadStats st = w->loadStats();
        CHECK(!st.compacted && st.duplicates == 2 * 8);
        CHECK(st.programs == 1 && st.translations == 5 && st.failures == 2);
        tc::Loaded got;
        CHECK(w->take(got));
        CHECK(got.translations.size() == 5 && got.programs.size() == 1 && got.failures.size() == 2);
    }
    CHECK(fs::file_size(file) == fat);
    {  // over it: rewritten, each key once
        tc::Limits l;
        l.slackBytes = 0;
        l.slackFraction = 0;
        auto w = open(l);
        const tc::LoadStats st = w->loadStats();
        CHECK(st.compacted && st.fileBytes == fat && st.afterBytes == size);
        CHECK(st.translations == 5);
    }
    CHECK(fs::file_size(file) == size);
    fs::remove_all(dir);
}

// Over the size cap the least recently used translations are dropped, not the whole file.
void translationCacheEvict() {
    namespace tc = vpps4::tcache;
    const fs::path dir = tcDir("evict");
    const fs::path file = dir / "translations.vpsc";
    Fixture f;
    {
        tc::Worker w(dir, tc::buildFingerprint(), f.device.pipelines());
        w.append(tc::ProgramRec{1, {}});
        for (uint64_t k = 1; k <= 10; ++k) w.append(tcRec(k, 2000));
        w.append(tc::FailureRec{1, 99, 1, "bad"});
        w.flush();
        w.touch(3);  // used by a later run: newer than the rest
        w.touch(7);
        w.touch(7);  // (once)
        w.touch(500);  // (not in the file: nothing)
        w.flush();
    }
    {
        tc::Limits l;
        l.maxBytes = 12000;
        l.keepFraction = 0.5;
        tc::Worker w(dir, tc::buildFingerprint(), f.device.pipelines(), l);
        const tc::LoadStats st = w.loadStats();
        CHECK(st.compacted && st.evicted == 8 && st.translations == 2 && st.programs == 1 && st.failures == 1);
        tc::Loaded got;
        CHECK(w.take(got));
        std::set<uint64_t> keys;
        for (const auto& t : got.translations) keys.insert(t.key2);
        CHECK((keys == std::set<uint64_t>{3, 7}));
        CHECK(fs::file_size(file) < 12000);
        // an evicted key is written again if a run translates it again
        w.append(tcRec(1, 2000));
        w.flush();
    }
    {
        tc::Worker w(dir, tc::buildFingerprint(), f.device.pipelines());
        CHECK(w.loadStats().translations == 3);
    }
    fs::remove_all(dir);
}

// Saved pipelines whose shaders this run does not have are not prefetched (they would only fail).
void translationCacheStalePipelines() {
    const fs::path dir = tcDir("stale");
    {
        Fixture f;
        f.renderer.setCacheRoot(dir);
        f.renderer.waitCache();
        f.renderer.draw(f.state());
        f.renderer.submit();
        f.renderer.flushCache();
    }
    const vpgpu::DiskCachePath path{dir};
    std::vector<vpgpu::PipelineState> saved = vpgpu::PipelineCache::readKeyList(path);
    CHECK(!saved.empty());
    vpgpu::PipelineState stale = saved.front();
    stale.vertexShader ^= 0x5555;
    {
        std::vector<uint64_t> v;
        vpgpu::PipelineCache::encode(stale, v);
        std::ofstream out(path.keyList(), std::ios::app);
        for (uint64_t x : v) out << x << ' ';
        out << '\n';
    }
    Fixture f;
    f.renderer.setCacheRoot(dir);
    f.renderer.waitCache();
    CHECK(f.device.pipelines().status(stale.key()) == vpgpu::PipelineCache::Status::Missing);
    CHECK(f.device.pipelines().totals().failed == 0);
    fs::remove_all(dir);
}

// A program that does not translate is remembered: the next run does not try it again.
void translationCacheFailures() {
    const fs::path dir = tcDir("fail");
    auto errors = [](const Fixture& f) {
        size_t n = 0;
        for (const std::string& l : f.log) n += l.find("vpps4.shader error") != std::string::npos;
        return n;
    };
    for (int run = 0; run < 3; ++run) {
        Fixture f;
        f.renderer.setCacheRoot(dir);
        f.renderer.waitCache();
        vpps4::DrawState d = f.state();
        static const uint32_t junk[] = {0xffffffff, 0xffffffff};
        d.vs.code = junk;
        f.renderer.draw(d);
        f.renderer.submit();
        CHECK(f.renderer.stats().drawn == 0);
        CHECK(f.renderer.stats().shaderErrors == 1);
        CHECK(errors(f) == (run == 0 ? 1u : 0u));  // (later runs know it fails: no lifting)
        f.renderer.flushCache();
    }
    fs::remove_all(dir);
}

// A scheduler on a fixture with a hand-driven clock (ms).
struct Sched {
    Fixture f;
    double t = 0;
    vpps4::Scheduler s{f.renderer, f.device, vpgpu::SubmitPolicy{}, [this] { return t; }};
    size_t submits() const { return f.count("submit"); }
    // Records a GPU-side DMA touching [addr, +256): work in the current frame.
    void work(uint64_t addr) {
        f.renderer.fill(addr, 256, 0);
        s.noteWork();
    }
};

// ~15 Liverpool submits per picture: draws accumulate, flushes do not cut the frame, the flip submits once.
void oneFramePerImage() {
    Sched x;
    vpps4::DrawState d = x.f.state();
    for (int cmdbuf = 0; cmdbuf < 15; ++cmdbuf) {
        for (int i = 0; i < 20; ++i) {
            x.f.renderer.draw(d);
            x.s.noteWork();
        }
        x.t += 1.0;
        x.s.flush();  // SubmitDone of every command buffer
    }
    CHECK(x.submits() == 0);
    x.s.present();
    CHECK(x.submits() == 1);
    CHECK(x.s.counters().submits[size_t(vpgpu::SubmitReason::Flip)] == 1);
    x.s.present();  // nothing recorded: no empty frame
    CHECK(x.submits() == 1);
    // Safety nets: a frame that stays open too long, or grows too big, goes by itself.
    x.work(kBase + 0x5000);
    x.t += 39;
    CHECK(!x.s.tick());
    x.t += 2;
    CHECK(x.s.tick() && x.submits() == 2);
    CHECK(x.s.counters().submits[size_t(vpgpu::SubmitReason::Age)] == 1);
}

// CPU access waits for the frames that overlap it, not for the whole queue.
void partialWaits() {
    Sched x;
    x.work(kBase + 0x1000);
    x.s.present();  // frame 0
    x.work(kBase + 0x3000);
    x.s.present();  // frame 1
    x.work(kBase + 0x5000);
    x.s.present();  // frame 2
    x.work(kBase + 0x7000);  // recorded, not submitted
    CHECK(!x.s.syncCpu(kBase + 0x9000, 64, vpgpu::SubmitReason::Dma));  // untouched: free
    CHECK(x.f.backend.waits.empty());
    CHECK(x.s.syncCpu(kBase + 0x3000, 16, vpgpu::SubmitReason::Dma));  // only frames 0..1
    CHECK(x.f.backend.waits.size() == 1 && x.f.backend.waits[0] == 2);
    CHECK(x.submits() == 3);                                              // nothing was submitted for it
    CHECK(x.f.device.completedFrames() == 2);                             // frame 2 still "runs"
    CHECK(!x.f.renderer.gpuMayAccess(kBase + 0x1000, 4));                 // retired
    CHECK(x.f.renderer.gpuMayAccess(kBase + 0x5000, 4));
    // The overlap is in recorded work: it is submitted, then only that frame (and the earlier) waited for.
    CHECK(x.s.syncCpu(kBase + 0x7000, 16, vpgpu::SubmitReason::Dma));
    CHECK(x.submits() == 4 && x.f.backend.waits.back() == 4);
    CHECK(x.s.counters().submits[size_t(vpgpu::SubmitReason::Dma)] == 1);
    // Source and destination of a copy: the later frame decides.
    x.work(kBase + 0xa000);
    x.s.present();  // frame 4
    CHECK(x.s.syncCpu(kBase + 0x5000, 4, vpgpu::SubmitReason::Dma, kBase + 0xa000, 4));
    CHECK(x.f.backend.waits.back() == 5);
}

// Host DMA (Rasterizer::FillBuffer / CopyBuffer of the app) with a byte pattern: on the GPU timeline when queued GPU work
// touches the range (Scheduler::dmaOnGpu: no wait, no submit), on the CPU in place when it is free. The NullBackend does not run
// copies: gpuRuns() applies the ones submitted so far, in order, to the guest block.
struct HostDma {
    Sched x;
    size_t ran = 0;
    bool fill(uint64_t addr, uint64_t bytes, uint8_t byte) {  // true = on the CPU
        if (x.s.dmaOnGpu(addr, bytes)) {
            CHECK(x.f.renderer.fill(addr, bytes, byte * 0x01010101u));
            x.s.noteDmaRecorded();
            x.s.noteWork();
            return false;
        }
        x.s.syncCpu(addr, bytes, vpgpu::SubmitReason::Dma, 0, 0, "fill");
        std::memset(x.f.at(addr), byte, bytes);
        x.f.renderer.invalidate(addr, bytes);
        return true;
    }
    bool copy(uint64_t dst, uint64_t src, uint64_t bytes) {
        if (x.s.dmaOnGpu(dst, bytes, src)) {
            CHECK(x.f.renderer.copy(dst, src, bytes));
            x.s.noteDmaRecorded();
            x.s.noteWork();
            return false;
        }
        x.s.syncCpu(src, bytes, vpgpu::SubmitReason::Dma, dst, bytes, "copy");
        std::memmove(x.f.at(dst), x.f.at(src), bytes);
        x.f.renderer.invalidate(dst, bytes);
        return true;
    }
    void gpuRuns() {
        for (; ran < x.f.backend.copies.size(); ++ran) {
            const vpgpu::Copy& c = x.f.backend.copies[ran];
            uint8_t* dst = static_cast<uint8_t*>(x.f.host) + c.dstBuffer.offset;
            if (c.kind == vpgpu::CopyKind::FillBuffer) std::memset(dst, c.fillByte, c.dstBuffer.size);
            else std::memmove(dst, static_cast<uint8_t*>(x.f.host) + c.srcBuffer.offset, c.dstBuffer.size);
        }
    }
};

// A DMA over a range the running GPU work touches does not wait for it (nor submit): it is recorded in order with that work.
// A free range stays on the CPU, in place. The bytes are what the in-order timeline would leave.
void dmaOnGpu() {
    HostDma h;
    Sched& x = h.x;
    x.work(kBase + 0x1000);  // GPU work over [0x1000, 0x1100) ...
    x.s.present();           // ... frame 0, submitted and still running
    CHECK(!h.fill(kBase + 0x1080, 64, 0xaa));                // overlaps it: recorded
    CHECK(!h.copy(kBase + 0x3000, kBase + 0x1080, 64));      // the source is that fill's destination
    CHECK(x.f.backend.waits.empty() && x.submits() == 1);    // no wait, no submit
    CHECK(x.s.counters().cpuSyncs == 0 && x.s.counters().dmaRecorded == 2);
    CHECK(*x.f.at(kBase + 0x1080) == 0 && *x.f.at(kBase + 0x3000) == 0);  // (the GPU has not run them)
    CHECK(h.fill(kBase + 0x9000, 64, 0x55));                 // untouched by the GPU: in place, now
    CHECK(*x.f.at(kBase + 0x9000) == 0x55555555u && x.f.backend.waits.empty());
    CHECK(x.s.counters().dmaRecorded == 2 && x.s.counters().cpuSyncFree == 1);
    // A DMA after a recorded one on the same bytes keeps the order (the later fill wins).
    CHECK(!h.fill(kBase + 0x3000, 16, 0x11));
    x.s.present();
    h.gpuRuns();
    CHECK(*x.f.at(kBase + 0x1080) == 0xaaaaaaaau && *x.f.at(kBase + 0x3000) == 0x11111111u);
    CHECK(*x.f.at(kBase + 0x3010) == 0xaaaaaaaau);
    CHECK(x.f.backend.waits.empty());
}

// The CP reads (IB, label, WAIT_REG_MEM) go through syncCpu: a range with a recorded DMA waits for it, and a label queued
// after it lands after it. A label queued before it and covered by it never lands after it.
void dmaThenLabel() {
    HostDma h;
    Sched& x = h.x;
    const uint64_t label = kBase + 0x5040;
    x.work(kBase + 0x5000);
    x.s.present();
    CHECK(!h.fill(label, 4, 0x77));  // recorded, the open frame
    CHECK(*x.f.at(label) == 0);
    // The command processor reads that word: the frame is submitted and waited for, then the value is there.
    CHECK(x.s.syncCpu(label, 4, vpgpu::SubmitReason::Dma, 0, 0, "cp_read"));
    CHECK(x.submits() == 2 && !x.f.backend.waits.empty() && x.f.backend.waits.back() == 2);
    h.gpuRuns();
    CHECK(*x.f.at(label) == 0x77777777u);
    // A label written after a recorded DMA: the GPU's order (fill, then the label's own write).
    const uint64_t l2 = kBase + 0x5080;
    x.work(kBase + 0x5000);
    CHECK(!h.fill(l2, 4, 0x33));
    x.s.scheduleLabel(l2, 4, false, [&] { *x.f.at(l2) = 7; });
    x.s.present();
    h.gpuRuns();
    x.f.backend.finishFramesBelow(x.f.device.submittedFrames());
    CHECK(*x.f.at(l2) == 7);
    // A label queued before the DMA, nothing met it in GPU order (labelThenDma drops those): the DMA waits for it (CPU, in place),
    // the DMA's bytes are the last.
    const uint64_t l3 = kBase + 0x6040;
    x.work(kBase + 0x6000);
    x.s.scheduleLabel(l3, 4, false, [&] { *x.f.at(l3) = 1; });
    CHECK(h.fill(l3, 4, 0x44));
    CHECK(*x.f.at(l3) == 0x44444444u && !x.f.renderer.labelPending(l3, 4) && x.s.counters().labelsSuperseded == 0);
}

// Only a label the command processor's own WAIT_REG_MEM already met in GPU order (prepareWait: Ordered) is dropped by a DMA
// over it: a CPU poll or another ring waiting for that value would otherwise hang. A copy keeps the labels that overlap what it
// reads (its source overlaps its destination: the copy reads them written).
void labelSupersedeConsumed() {
    HostDma h;
    Sched& x = h.x;
    const uint64_t block = kBase + 0x7400, stamp = block + 0x10, unmet = block + 0x20;
    int stamps = 0, unmets = 0;
    x.work(kBase + 0x1000);
    x.s.scheduleLabel(stamp, 4, false, [&] { *x.f.at(stamp) = 0x1234, ++stamps; }, 0x1234, true);
    x.s.scheduleLabel(unmet, 4, false, [&] { *x.f.at(unmet) = 0x55, ++unmets; }, 0x55, true);
    CHECK(x.s.prepareWait(stamp, 4, [](uint32_t v) { return v == 0x1234; }) == vpps4::Scheduler::WaitKind::Ordered);
    CHECK(x.s.prepareWait(unmet, 4, [](uint32_t v) { return v == 0x66; }) == vpps4::Scheduler::WaitKind::Poll);  // (not the value it waits for)
    x.s.present();
    // Over the block: the label met in order is dropped, the other is written first (the fill waits for it).
    CHECK(h.fill(block, 0x40, 0xee));
    CHECK(stamps == 0 && unmets == 1 && x.s.counters().labelsSuperseded == 1);
    CHECK(*x.f.at(stamp) == 0xeeeeeeeeu && *x.f.at(unmet) == 0xeeeeeeeeu);
    // A copy whose source overlaps its destination: a label met in order inside both is kept, one only in the destination dropped.
    const uint64_t dst = kBase + 0x7800, src = dst + 0x20, inDst = dst + 0x4, inBoth = dst + 0x30;
    int dropped = 0, kept = 0;
    x.work(kBase + 0x1000);
    x.s.scheduleLabel(inDst, 4, false, [&] { ++dropped; }, 0x11, true);
    x.s.scheduleLabel(inBoth, 4, false, [&] { *x.f.at(inBoth) = 0x22, ++kept; }, 0x22, true);
    CHECK(x.s.prepareWait(inDst, 4, [](uint32_t v) { return v == 0x11; }) == vpps4::Scheduler::WaitKind::Ordered);
    CHECK(x.s.prepareWait(inBoth, 4, [](uint32_t v) { return v == 0x22; }) == vpps4::Scheduler::WaitKind::Ordered);
    const uint64_t superseded = x.s.counters().labelsSuperseded;
    CHECK(h.copy(dst, src, 0x40));
    CHECK(x.s.counters().labelsSuperseded == superseded + 1 && dropped == 0 && kept == 1);
}

// The scheduler takes the clock only to call it: an empty one does not throw when it is built.
void schedulerNoClock() {
    Fixture f;
    bool threw = false;
    try {
        vpps4::Scheduler s(f.renderer, f.device, vpgpu::SubmitPolicy{}, vpps4::Scheduler::Clock{});
    } catch (...) {
        threw = true;
    }
    CHECK(!threw);
}

// A title stamps labels (EOP/EOS/RELEASE_MEM, no interrupt) into a block, waits for one on its own ring (WAIT_REG_MEM, met in
// GPU order) and resets the block with a DMA fill. In GPU order the fill's bytes are the last ones: the labels it covers are
// dropped and the fill neither submits nor waits. A label it does not cover (or one with an interrupt) still lands only when
// the GPU is done with the work before it; a CP read after a fill recorded on the GPU waits for that fill.
void labelThenDma() {
    HostDma h;
    Sched& x = h.x;
    const uint64_t block = kBase + 0x7000, stamp = block + 0x10, other = kBase + 0x7900, irqLabel = kBase + 0x7a00;
    uint64_t stampWrites = 0;
    x.work(kBase + 0x1000);  // the work the labels follow (recorded)
    x.s.scheduleLabel(stamp, 8, false, [&] { *reinterpret_cast<uint64_t*>(x.f.at(stamp)) = 0x1234, ++stampWrites; }, 0x1234, true);
    x.s.scheduleLabel(other, 4, false, [&] { *x.f.at(other) = 5; }, 5, true);
    CHECK(x.s.prepareWait(stamp, 4, [](uint32_t v) { return v == 0x1234; }) == vpps4::Scheduler::WaitKind::Ordered);
    CHECK(h.fill(block, 2048, 0xee));  // nothing GPU touches the block: in place, now
    CHECK(x.submits() == 0 && x.f.backend.waits.empty() && x.s.counters().cpuSyncs == 0);
    CHECK(x.s.counters().labelsSuperseded == 1 && !x.f.renderer.labelPending(stamp, 8));
    CHECK(*x.f.at(stamp) == 0xeeeeeeeeu && *x.f.at(other) == 0);  // (the other label: not before its work is done)
    x.s.present();
    CHECK(*x.f.at(other) == 0);
    x.f.backend.finishFramesBelow(x.f.device.submittedFrames());
    CHECK(*x.f.at(other) == 5 && stampWrites == 0 && *x.f.at(stamp) == 0xeeeeeeeeu);
    // The block busy with recorded GPU work: the fill goes on the GPU timeline, the label covered is dropped, and the command
    // processor reading the label word afterwards waits for the fill (label -> fill -> CP read).
    x.work(block + 0x100);
    x.s.scheduleLabel(stamp, 8, false, [&] { *reinterpret_cast<uint64_t*>(x.f.at(stamp)) = 0x99, ++stampWrites; }, 0x99, true);
    CHECK(x.s.prepareWait(stamp, 4, [](uint32_t v) { return v == 0x99; }) == vpps4::Scheduler::WaitKind::Ordered);
    const size_t submitted = x.submits();
    CHECK(!h.fill(block, 2048, 0x22));
    CHECK(x.submits() == submitted && x.f.backend.waits.empty() && x.s.counters().labelsSuperseded == 2);
    CHECK(x.s.syncCpu(stamp, 8, vpgpu::SubmitReason::Dma, 0, 0, "cp_read"));
    CHECK(x.submits() == submitted + 1);
    h.gpuRuns();
    x.f.backend.finishFramesBelow(x.f.device.submittedFrames());
    CHECK(*x.f.at(stamp) == 0x22222222u && stampWrites == 0);
    // A label with an interrupt, or one the fill covers only in part, is waited for as before.
    int irqs = 0;
    x.work(kBase + 0x1000);
    x.s.scheduleLabel(irqLabel, 4, true, [&] { *x.f.at(irqLabel) = 7, ++irqs; }, 7, true);
    x.s.scheduleLabel(irqLabel + 0x3c, 8, false, [&] { *x.f.at(irqLabel + 0x3c) = 8; }, 8, true);
    CHECK(h.fill(irqLabel, 0x40, 0x33));
    CHECK(irqs == 1 && *x.f.at(irqLabel) == 0x33333333u && x.s.counters().labelsSuperseded == 2);
    CHECK(x.s.counters().submits[size_t(vpgpu::SubmitReason::Dma)] == 2);
    const std::string sum = x.s.takeSummary();
    CHECK(sum.find(" label_superseded=2 ") != std::string::npos && sum.find(" cp_wait_pct=") != std::string::npos);
    CHECK(sum.find(" frames_per_flip=3.0") != std::string::npos);
}

// VPKENGINE_LABEL_KEEP=1: a DMA over a pending label waits for it (submit + wait), the label is written, then the DMA.
void labelKeepOptOut() {
    setenv("VPKENGINE_LABEL_KEEP", "1", 1);
    HostDma h;
    unsetenv("VPKENGINE_LABEL_KEEP");
    Sched& x = h.x;
    const uint64_t stamp = kBase + 0x7010;
    int writes = 0;
    x.work(kBase + 0x1000);
    x.s.scheduleLabel(stamp, 4, false, [&] { *x.f.at(stamp) = 0x1234, ++writes; }, 0x1234, true);
    CHECK(h.fill(kBase + 0x7000, 2048, 0xee));
    CHECK(x.submits() == 1 && !x.f.backend.waits.empty() && writes == 1 && x.s.counters().labelsSuperseded == 0);
    CHECK(*x.f.at(stamp) == 0xeeeeeeeeu);
}

// A range the renderer reads on the CPU (texture source, descriptors, ...) is not DMA'd on the GPU timeline even when the GPU is
// busy over it: the CPU would read the old bytes until the GPU ran the DMA. A busy range it never read is.
void dmaAfterCpuRead() {
    Sched x;
    x.f.renderer.draw(x.f.state());  // reads the texture at +0x10000 and the V# table at +0x100 on the CPU
    x.s.noteWork();
    x.work(kBase + 0x10000);
    x.work(kBase + 0x100);
    x.work(kBase + 0x20000);
    x.s.present();
    CHECK(!x.s.dmaOnGpu(kBase + 0x10040, 64));
    CHECK(!x.s.dmaOnGpu(kBase + 0x108, 4));
    CHECK(x.s.dmaOnGpu(kBase + 0x20040, 64, kBase + 0x10080));  // (a copy's destination decides, not its source)
    CHECK(x.s.dmaOnGpu(kBase + 0x20040, 64));
    CHECK(x.s.counters().dmaRecorded == 0);  // (counted when the caller records it)
}

// The fetch subroutine the renderer reads on the CPU counts as read where it lives; with the probe on, a first CPU read of a range
// that a recorded fill still has to write leaves a DMA_STALE_READ line.
void dmaStaleRead() {
    vpgpu::probe::setEnabled(true);
    {
        Sched x;
        x.work(kBase + 0x10000);  // a fill recorded over the texture the draw reads
        x.f.renderer.draw(x.f.state());
        std::string line;
        for (const std::string& l : x.f.log)
            if (l.rfind("VPKENGINE_PROBE DMA_STALE_READ", 0) == 0) line = l;
        CHECK(line.find("a=0x10010000") != std::string::npos && line.find("by=") != std::string::npos);
        CHECK(line.find("busy=[0x10010000-") != std::string::npos && line.find(" by=fill in=recorded") != std::string::npos);
        vpps4::DrawState d = x.f.state();
        d.vs.userSgprs = 16;
        for (unsigned i = 0; i < 16; ++i) d.vs.userData[i] = 0x1000u * (i + 1);  // (a pair of them is the fetch subroutine's address)
        d.vs.userData[4] = uint32_t(kBase + 0x100);
        const uint64_t fetchAt = vpps4::fetchAddress(d.vs.code, d.vs.userData, d.vs.userSgprs);
        CHECK(fetchAt != 0 && !x.f.renderer.cpuReadSeen(fetchAt, 4));
        x.f.renderer.draw(d);
        CHECK(x.f.renderer.cpuReadSeen(fetchAt, 4));
        // A DMA kept on the CPU by such a read names the reader (DMA_CPU_READ).
        CHECK(!x.s.dmaOnGpu(kBase + 0x10040, 64));
        std::string cpuRead;
        for (const std::string& l : x.f.log)
            if (l.rfind("VPKENGINE_PROBE DMA_CPU_READ", 0) == 0) cpuRead = l;
        CHECK(cpuRead.find("dst=0x10010040 bytes=64") != std::string::npos && cpuRead.find("by=[0x10010000+") != std::string::npos);
    }
    vpgpu::probe::setEnabled(false);
}

// Probe DMASYNC: a CPU access that waits leaves a line with the caller, the busy range that made it wait and who made that.
void dmaSyncProbe() {
    vpgpu::probe::setEnabled(true);
    {
        Sched x;
        x.work(kBase + 0x1000);
        x.s.present();
        CHECK(x.s.syncCpu(kBase + 0x1080, 64, vpgpu::SubmitReason::Dma, 0, 0, "owned_fault"));
        std::string line;
        for (const std::string& l : x.f.log)
            if (l.rfind("VPKENGINE_PROBE DMASYNC_WAIT", 0) == 0) line = l;
        CHECK(line.find("why=owned_fault") != std::string::npos && line.find("by=fill") != std::string::npos);
        CHECK(line.find("busy=[0x10001000-0x10001100") != std::string::npos && line.find("in=frame 0") != std::string::npos);
        CHECK(line.size() < 2048);
        CHECK(vpgpu::probe::counter("DMASYNC", "owned_fault_waits").get() == 1);
    }
    vpgpu::probe::setEnabled(false);
}

// VPKENGINE_DMA_CPU: every DMA waits and runs on the CPU, as before.
void dmaOnCpuOptOut() {
    setenv("VPKENGINE_DMA_CPU", "1", 1);
    HostDma h;
    unsetenv("VPKENGINE_DMA_CPU");
    Sched& x = h.x;
    x.work(kBase + 0x1000);
    x.s.present();
    CHECK(h.fill(kBase + 0x1080, 64, 0xaa));
    CHECK(!x.f.backend.waits.empty() && x.s.counters().dmaRecorded == 0);
    CHECK(*x.f.at(kBase + 0x1080) == 0xaaaaaaaau);
}

// A buffer bound by address alone (a V# without records: no extent) marks a window of it busy, not the whole mapping it lies
// in (a CPU DMA elsewhere in the mapping would wait for the GPU). VPKENGINE_TOUCH_WINDOW=1 brings the whole mapping back.
void unknownExtentBusy() {
    const auto busyFar = [](bool env) {
        if (env) setenv("VPKENGINE_TOUCH_WINDOW", "1", 1);
        Fixture f;
        if (env) unsetenv("VPKENGINE_TOUCH_WINDOW");
        f.at(kBase + 0x100)[2] = 0;  // the vertex V# says 0 records
        f.renderer.draw(f.state());
        const bool near = f.renderer.gpuMayAccess(kBase + 0x1000, 4);
        const bool far = f.renderer.gpuMayAccess(kBase + 0x70000, 4);
        f.renderer.submit();
        return std::pair{near, far};
    };
    const auto normal = busyFar(false);
    CHECK(normal.first && !normal.second);
    const auto optOut = busyFar(true);
    CHECK(optOut.first && optOut.second);
}

// A label that waits on the recorded work is not submitted at its grace while a multisampled pass is open (a cut there stores and
// reloads every sample): SubmitPolicy::passSlackMs (passSlackIrqMs with an interrupt) more. With none open, the grace as before.
void labelWaitsForMsaaPass() {
    for (const bool ms : {true, false}) {
        {
            Sched x;
            x.f.renderer.setMaxSamples(4);
            vpps4::DrawState d = x.f.state();
            d.color[0].samples = ms ? 4 : 1;
            x.f.renderer.draw(d);
            x.s.noteWork();
            x.s.scheduleLabel(kBase + 0x100, 4, false, [] {});
            CHECK(x.f.renderer.multisampledPassOpen() == ms);
            x.t += 5;
            CHECK(!x.s.tick());
            x.t += 2;  // 7 ms: past the grace (6)
            if (ms) CHECK(x.s.nextCheckMs() > 6.9 && x.s.nextCheckMs() < 7.1);  // (14 - 7)
            CHECK(x.s.tick() == !ms);
            if (ms) {
                x.t += 6;  // 13
                CHECK(!x.s.tick() && x.submits() == 0);
                x.t += 1.5;  // 14.5: past grace + slack
                CHECK(x.s.tick() && x.submits() == 1);
            }
            CHECK(x.submits() == 1 && x.s.counters().submits[size_t(vpgpu::SubmitReason::LabelGrace)] == 1);
        }
        {  // with an interrupt: 1 ms, and 2 more
            Sched x;
            x.f.renderer.setMaxSamples(4);
            vpps4::DrawState d = x.f.state();
            d.color[0].samples = ms ? 4 : 1;
            x.f.renderer.draw(d);
            x.s.noteWork();
            x.s.scheduleLabel(kBase + 0x100, 4, true, [] {});
            x.t += 1.5;
            CHECK(x.s.tick() == !ms);
            if (ms) {
                x.t += 1;  // 2.5
                CHECK(!x.s.tick());
                x.t += 1;  // 3.5
                CHECK(x.s.tick() && x.submits() == 1);
            }
        }
    }
}

// Label writes happen when the GPU is done with the work before them, in order, with their interrupt.
void labelOrdering() {
    Sched x;
    std::vector<int> order;
    int mem = 0;
    x.work(kBase + 0x1000);
    x.s.scheduleLabel(kBase + 0x100, 4, false, [&] { mem = 1, order.push_back(1); });
    x.s.scheduleLabel(kBase + 0x100, 4, true, [&] { mem = 2, order.push_back(2); });  // EOP data + IRQ
    CHECK(mem == 0 && x.f.renderer.labelPending(kBase + 0x100, 4));
    // A command buffer ended, nothing needs the label yet: still recorded.
    x.t += 0.5;
    x.s.flush();
    CHECK(x.submits() == 0);
    // An interrupt label is not kept waiting more than a moment.
    x.t += 1.0;
    x.s.flush();
    CHECK(x.submits() == 1 && x.s.counters().submits[size_t(vpgpu::SubmitReason::LabelGrace)] == 1);
    CHECK(mem == 0);  // submitted is not finished
    x.f.backend.finishFramesBelow(1);
    CHECK(mem == 2 && order == (std::vector<int>{1, 2}));
    // A label after finished work (nothing recorded) is written at once.
    x.s.scheduleLabel(kBase + 0x104, 4, false, [&] { mem = 3; });
    CHECK(mem == 3);
    // WAIT_REG_MEM on a label the recorded work will write forces the submit, without waiting.
    x.work(kBase + 0x2000);
    x.s.scheduleLabel(kBase + 0x200, 4, false, [&] { mem = 4; });
    x.s.prepareWait(kBase + 0x900, 4);
    CHECK(x.submits() == 1);
    x.s.prepareWait(kBase + 0x200, 4);
    CHECK(x.submits() == 2 && mem == 3);
    // ... and recorded GPU work that writes the polled memory too.
    x.work(kBase + 0x3000);
    x.s.prepareWait(kBase + 0x3000, 4);
    CHECK(x.submits() == 3);
    x.f.backend.finishFramesBelow(3);
    CHECK(mem == 4);
    // A CPU write (WRITE_DATA / DMA) into a pending label comes after the label's own write.
    x.work(kBase + 0x4000);
    x.s.scheduleLabel(kBase + 0x300, 4, false, [&] { mem = 5; });
    CHECK(x.s.syncCpu(kBase + 0x300, 4, vpgpu::SubmitReason::WriteData));
    CHECK(mem == 5 && !x.f.renderer.labelPending(kBase + 0x300, 4));
}

// WAIT_REG_MEM on a label queued behind work of the same in-order queue is met in GPU order: no submit,
// no wait. Labels the CPU/guest produces, or with another value, are polled.
void gpuOrderedLabelWait() {
    using WK = vpps4::Scheduler::WaitKind;
    Sched x;
    int mem = 0;
    const auto eq = [](uint32_t v) { return [v](uint32_t w) { return w == v; }; };
    x.work(kBase + 0x1000);
    x.s.scheduleLabel(kBase + 0x200, 4, false, [&] { mem = 1; }, 1, true);
    CHECK(x.s.prepareWait(kBase + 0x200, 4, eq(1)) == WK::Ordered);
    CHECK(x.submits() == 0 && mem == 0 && x.f.backend.waits.empty());
    CHECK(x.s.counters().orderedWaits == 1);
    // A different value (or a mask that cannot match) is not produced by that label.
    CHECK(x.s.prepareWait(kBase + 0x200, 4, eq(2)) == WK::Poll);
    CHECK(x.submits() == 1);  // the label's work goes to the GPU
    // Submitted but not finished: still met in GPU order.
    CHECK(x.s.prepareWait(kBase + 0x200, 4, eq(1)) == WK::Ordered);
    CHECK(x.submits() == 1 && mem == 0);
    // Produced by the guest CPU / nobody: polled, not submitted, and the idle wait has no frame to block on.
    CHECK(x.s.prepareWait(kBase + 0x700, 4, eq(1)) == WK::Poll);
    CHECK(!x.s.idleWait(kBase + 0x700, 4, 1));
    // A label with unknown value never counts.
    x.work(kBase + 0x2000);
    x.s.scheduleLabel(kBase + 0x300, 4, false, [&] { mem = 3; });
    CHECK(x.s.prepareWait(kBase + 0x300, 4, eq(1)) == WK::Poll);
    CHECK(x.submits() == 2);
    // Without a predicate (old callers) the wait is never taken as met.
    CHECK(x.s.prepareWait(kBase + 0x200, 4) == WK::Poll);
    // Once the frame finished the label is final: polled (the value is in memory, Test passes).
    x.f.backend.finishFramesBelow(2);
    CHECK(mem == 3);
    CHECK(x.s.prepareWait(kBase + 0x200, 4, eq(1)) == WK::Poll);
    // A poll that has a frame to wait for blocks on it (condition wait), not on the CPU.
    x.work(kBase + 0x4000);
    x.s.scheduleLabel(kBase + 0x400, 4, false, [&] { mem = 4; }, 1, true);
    CHECK(x.s.idleWait(kBase + 0x400, 4, 1));
    CHECK(x.submits() == 3 && mem == 4);
    // Rewritten by someone else (WRITE_DATA, the CPU): the queued value no longer counts.
    x.work(kBase + 0x5000);
    x.s.scheduleLabel(kBase + 0x500, 4, false, [&] {}, 1, true);
    CHECK(x.s.prepareWait(kBase + 0x500, 4, eq(1)) == WK::Ordered);
    x.s.forgetLabelValues(kBase + 0x500, 4);
    CHECK(x.s.prepareWait(kBase + 0x500, 4, eq(1)) == WK::Poll);
    // A newer label to the same word hides the older value; a wider valueless one hides it too.
    x.work(kBase + 0x6000);
    x.s.scheduleLabel(kBase + 0x600, 4, false, [&] {}, 1, true);
    x.s.scheduleLabel(kBase + 0x600, 4, false, [&] {}, 0, true);
    CHECK(x.s.prepareWait(kBase + 0x600, 4, eq(1)) == WK::Poll);
    CHECK(x.s.prepareWait(kBase + 0x600, 4, eq(0)) == WK::Ordered);
    x.s.scheduleLabel(kBase + 0x600, 16, false, [&] {});
    CHECK(x.s.prepareWait(kBase + 0x600, 4, eq(0)) == WK::Poll);
}

// finish waits for everything; labels written.
void finishAll() {
    Sched x;
    int mem = 0;
    x.work(kBase + 0x1000);
    x.s.scheduleLabel(kBase + 0x100, 4, false, [&] { mem = 1; });
    x.s.finish();
    CHECK(mem == 1 && x.f.device.completedFrames() == 1 && !x.f.renderer.gpuMayAccess(kBase + 0x1000, 4));
}

// A draw whose pipeline is still compiling is recorded and the frame held (bounded), not dropped.
void holdPipeline() {
    void* host = std::aligned_alloc(16384, kSize);
    std::memset(host, 0, kSize);
    for (int holdMs : {2000, 0}) {
        vpgpu::NullBackend backend;
        std::atomic<bool> gate{false};
        backend.onCompile = [&] {
            while (!gate.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        };
        std::vector<std::string> log;
        vpgpu::Device device(backend, 1, [&](const std::string& s) { log.push_back(s); });
        device.setHoldMs(holdMs);
        std::vector<std::pair<uint64_t, vpir::Stage>> shaders;
        vpps4::Hooks h;
        h.region = [&](uint64_t a, uint64_t s, vpps4::GuestRegion& g) {
            if (a < kBase || a + s > kBase + kSize) return false;
            g = {kBase, kSize, host};
            return true;
        };
        h.shader = [&](uint64_t id, vpir::Stage st, const vpmsl::MslShader&) { shaders.push_back({id, st}); };
        vpps4::Renderer renderer(device, h);
        Fixture shapes;  // (only for the draw state: it refers to its own memory)
        const vpps4::DrawState ds = shapes.state();  // (writes its descriptor tables into the fixture's memory)
        std::memcpy(host, shapes.host, kSize);
        renderer.draw(ds);
        if (holdMs) {
            CHECK(renderer.stats().drawn == 1);  // recorded although the pipeline is not built
            std::thread release([&] {
                std::this_thread::sleep_for(std::chrono::milliseconds(60));
                gate = true;
            });
            const auto t0 = std::chrono::steady_clock::now();
            renderer.submit();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            release.join();
            CHECK(ms > 40);
            bool held = false;
            for (const std::string& l : log) held |= l.find("gpu.hold") == 0 && l.find("complete=1") != std::string::npos;
            CHECK(held);
        } else {
            CHECK(renderer.stats().drawn == 0);
            CHECK(renderer.stats().skipped.at("pipeline compiling") == 1);
            gate = true;
        }
    }
    std::free(host);
}

// Past the warm-up a frame is held for a short budget only (VPKENGINE_HOLD_MS), not for the whole
// hold of the warm-up: the caller does not wait for a pipeline that is still compiling. The draw is
// recorded all the same; the backend leaves it out of that frame.
void holdBudget() {
    setenv("VPKENGINE_HOLD_MS", "20", 1);
    setenv("VPKENGINE_HOLD_WARMUP_FRAMES", "0", 1);
    void* host = std::aligned_alloc(16384, kSize);
    std::memset(host, 0, kSize);
    vpgpu::NullBackend backend;
    std::atomic<bool> gate{false};
    backend.onCompile = [&] {
        while (!gate.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    };
    std::vector<std::string> log;
    vpgpu::Device device(backend, 1, [&](const std::string& s) { log.push_back(s); });
    unsetenv("VPKENGINE_HOLD_MS");
    unsetenv("VPKENGINE_HOLD_WARMUP_FRAMES");
    device.setHoldMs(2000);  // (the warm-up's hold: not in force any more)
    CHECK(device.holdMs() == 20);
    vpps4::Hooks h;
    h.region = [&](uint64_t a, uint64_t s, vpps4::GuestRegion& g) {
        if (a < kBase || a + s > kBase + kSize) return false;
        g = {kBase, kSize, host};
        return true;
    };
    h.shader = [](uint64_t, vpir::Stage, const vpmsl::MslShader&) {};
    vpps4::Renderer renderer(device, h);
    Fixture shapes;  // (only for the draw state: it refers to its own memory)
    const vpps4::DrawState ds = shapes.state();
    std::memcpy(host, shapes.host, kSize);
    renderer.draw(ds);
    CHECK(renderer.stats().drawn == 1);  // recorded although the pipeline is not built
    const auto t0 = std::chrono::steady_clock::now();
    renderer.submit();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    CHECK(ms < 500);  // (the pipeline is still blocked: the old hold would have waited 2000 ms)
    bool timedOut = false;
    for (const std::string& l : log) timedOut |= l.find("gpu.hold") == 0 && l.find("complete=0") != std::string::npos;
    CHECK(timedOut);
    gate = true;
    std::free(host);
}

// The submit phases of the report (VPKENGINE_GPU cpu: submit phases): the hold is counted apart from
// the compile and the backend call, and a timeout is counted.
void submitTimes() {
    void* host = std::aligned_alloc(16384, kSize);
    std::memset(host, 0, kSize);
    vpgpu::NullBackend backend;
    std::atomic<bool> gate{false};
    backend.onCompile = [&] {
        while (!gate.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    };
    vpgpu::Device device(backend, 1, [](const std::string&) {});
    device.setHoldMs(2000);
    vpps4::Hooks h;
    h.region = [&](uint64_t a, uint64_t s, vpps4::GuestRegion& g) {
        if (a < kBase || a + s > kBase + kSize) return false;
        g = {kBase, kSize, host};
        return true;
    };
    h.shader = [](uint64_t, vpir::Stage, const vpmsl::MslShader&) {};
    vpps4::Renderer renderer(device, h);
    Fixture shapes;
    const vpps4::DrawState ds = shapes.state();
    std::memcpy(host, shapes.host, kSize);
    renderer.draw(ds);
    std::thread release([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        gate = true;
    });
    renderer.submit();
    release.join();
    const vpps4::CpuTimes t = renderer.takeCpuTimes();
    CHECK(t.holdFrames == 1 && t.holdTimeouts == 0);
    CHECK(t.holdMs > 100);                      // (the hold is its own phase)
    CHECK(t.submitMs >= t.holdMs);              // (and part of the submit)
    CHECK(t.compileMs >= 0 && t.compileMs < t.holdMs);
    CHECK(t.backendMs >= 0 && t.slotWaitMs == 0);
    CHECK(t.lifeMs >= 0);
    const vpps4::CpuTimes again = renderer.takeCpuTimes();  // (taken: counted once)
    CHECK(again.holdFrames == 0 && again.holdMs == 0);
    std::free(host);
}

// Regions the host reports as mirrored are kept across draws while Hooks::regionGeneration stays the
// same, and asked again once it changes (a mapping unmapped and mapped again at the same address
// must not be served from the old answer). Regions that are not mirrored are asked again per draw.
void regionCache() {
    for (bool mirrored : {true, false}) {
        void* host = std::aligned_alloc(16384, kSize);
        std::memset(host, 0, kSize);
        vpgpu::NullBackend backend;
        vpgpu::Device device(backend, 0, [](const std::string&) {});
        uint64_t generation = 1;
        unsigned calls = 0;
        vpps4::Hooks h;
        h.region = [&](uint64_t a, uint64_t s, vpps4::GuestRegion& g) {
            if (a < kBase || a + s > kBase + kSize) return false;
            ++calls;
            g = {kBase, kSize, host, mirrored};
            return true;
        };
        h.regionGeneration = [&] { return generation; };
        h.shader = [](uint64_t, vpir::Stage, const vpmsl::MslShader&) {};
        vpps4::Renderer renderer(device, h);
        Fixture shapes;  // (only for the draw state: it refers to its own memory)
        const vpps4::DrawState ds = shapes.state();
        std::memcpy(host, shapes.host, kSize);
        renderer.draw(ds);
        CHECK(renderer.stats().drawn == 1);
        const unsigned first = calls;
        CHECK(first > 0);
        renderer.draw(ds);
        CHECK(renderer.stats().drawn == 2);
        if (mirrored) {
            CHECK(calls == first);  // served from the cache
            const vpps4::CpuTimes t = renderer.takeCpuTimes();
            CHECK(t.regionHits > 0 && t.regionMisses == first);
        } else {
            CHECK(calls > first);
        }
        const unsigned before = calls;
        ++generation;  // the host's map changed
        renderer.draw(ds);
        CHECK(calls > before);  // asked again
        const unsigned after = calls;
        renderer.draw(ds);
        if (mirrored) CHECK(calls == after);  // and kept again
        renderer.submit();
        CHECK(renderer.stats().drawn == 4);
        std::free(host);
    }
}

// Guest memory as the host exposes it: the low block as one region, and two 64 MiB windows at kHigh (window aligned) that are
// one mapping; the region hook cuts the mapping at the window around the address and widens the window to hold [a, a + s).
constexpr uint64_t kWin = 64ull << 20, kHigh = 0x40000000;
struct Windowed {
    void* low = std::aligned_alloc(16384, kSize);
    void* high = std::aligned_alloc(16384, 2 * kWin);  // (untouched pages cost nothing)
    vpgpu::NullBackend backend;
    std::vector<std::string> log;
    vpgpu::Device device{backend, 0, [this](const std::string& s) { log.push_back(s); }};
    uint64_t generation = 1;
    vpps4::Renderer renderer{device, hooks()};
    Windowed() {
        std::memset(low, 0, kSize);
        device.setRefuseFailedBuffers(true);
    }
    ~Windowed() {
        std::free(low);
        std::free(high);
    }
    vpps4::Hooks hooks() {
        vpps4::Hooks h;
        h.region = [this](uint64_t a, uint64_t s, vpps4::GuestRegion& g) {
            if (a >= kBase && a + s <= kBase + kSize) {
                g = {kBase, kSize, low};
                return true;
            }
            if (a < kHigh || a >= kHigh + 2 * kWin) return false;
            const uint64_t from = a & ~(kWin - 1);
            const uint64_t end = std::min(std::max(from + kWin, (a + s + kWin - 1) & ~(kWin - 1)), kHigh + 2 * kWin);
            if (a + s > end) return false;
            g = {from, end - from, static_cast<uint8_t*>(high) + (from - kHigh)};
            return true;
        };
        h.regionGeneration = [this] { return generation; };
        h.shader = [](uint64_t, vpir::Stage, const vpmsl::MslShader&) {};
        h.log = [this](const std::string& s) { log.push_back(s); };
        return h;
    }
    size_t buffersOf(uint64_t size) const {
        size_t n = 0;
        for (const std::string& c : backend.calls)
            n += c.rfind("createBuffer", 0) == 0 && c.find(" size=" + std::to_string(size) + " guest=true") != std::string::npos;
        return n;
    }
};

// A range that crosses the boundary of the host's windows is held whole by one buffer: a tiled render target written back
// to memory (retile), a DMA fill and a DMA copy across it work, where the buffer used to end at the window.
void windowCrossing() {
    Windowed w;
    Fixture shapes;  // (only for the draw state: it refers to its own memory)
    vpps4::DrawState d = shapes.state();
    std::memcpy(w.low, shapes.host, kSize);
    d.color[0] = {kHigh + kWin - (1 << 20), 1024, 1024, 1024, 10, 0, 0, 14};  // 4 MiB tiled: 3 MiB of it past the boundary
    w.renderer.draw(d);
    CHECK(w.renderer.stats().drawn == 1);
    CHECK(w.renderer.writeBack(d.color[0]));
    CHECK(w.renderer.stats().gpuRetiled == 1);
    CHECK(w.renderer.stats().notes.count("image newer than memory lies outside guest memory: not written back") == 0);
    CHECK(w.buffersOf(2 * kWin) == 1);  // the buffer that holds the whole range
    CHECK(w.renderer.fill(kHigh + kWin - 4096, 8192, 0));
    CHECK(w.renderer.copy(kHigh + 16, kHigh + kWin - 4096, 8192));
    CHECK(w.renderer.copy(kHigh + kWin - 4096, kBase, 8192));
    CHECK(w.renderer.stats().notes.count("DMA fill outside guest memory: dropped") == 0);
    CHECK(w.renderer.stats().notes.count("DMA copy outside guest memory: dropped") == 0);
    CHECK(w.buffersOf(2 * kWin) == 1);  // (the same buffer served them)
    // Inside one window nothing is widened.
    CHECK(w.renderer.fill(kHigh + 0x1000, 8192, 0));
    // Past the mapping there is nothing to widen to: the old answer, and the DMA is dropped.
    CHECK(!w.renderer.fill(kHigh + 2 * kWin - 4096, 8192, 0));
    CHECK(w.renderer.stats().notes.count("DMA fill outside guest memory: dropped") == 1);
    w.renderer.submit();
    if (failures)
        for (const std::string& l : w.log) std::fprintf(stderr, "  %s\n", l.c_str());
}

// The backend refusing a window (the no-copy wrap of guest memory can fail): the range is served by a smaller window that
// holds it, the refusal is not retried until the host's map changes, and it is logged once; a draw whose buffer cannot be
// made at all is not bound to a missing buffer (zeros, noted).
void guestBufferRefused() {
    {
        Windowed w;
        unsigned attempts = 0;
        w.backend.failBuffer = [&](const vpgpu::BufferDesc& bd) {
            attempts += bd.size == 2 * kWin;
            return bd.size == 2 * kWin;
        };
        for (int i = 0; i < 3; ++i) CHECK(w.renderer.fill(kHigh + kWin - 4096, 8192, 0));
        CHECK(attempts == 1);  // not made again at every use
        CHECK(w.buffersOf(2 * kWin) == 0 && w.buffersOf(32ull << 20) == 1);  // (the part of it that holds the range)
        ++w.generation;  // the host's map changed: tried again
        CHECK(w.renderer.fill(kHigh + kWin - 4096, 8192, 0));
        CHECK(attempts == 2);
        for (int i = 0; i < 125; ++i) w.renderer.submit();  // a refusal is not kept for ever: tried again after a while
        CHECK(w.renderer.fill(kHigh + kWin - 4096, 8192, 0));
        CHECK(attempts == 3);
        size_t lines = 0;
        for (const std::string& l : w.log) lines += l.find("gpu.err what=guest_buffer_refused") == 0;
        CHECK(lines == 1);
        w.renderer.submit();
    }
    {
        Fixture f;  // every buffer over the fixture's memory is refused
        f.device.setRefuseFailedBuffers(true);
        f.backend.failBuffer = [&](const vpgpu::BufferDesc& bd) { return bd.guestMemory == f.host; };
        f.renderer.draw(f.state());
        CHECK(f.renderer.stats().drawn == 1);
        CHECK(f.renderer.stats().notes.count("buffer outside guest memory bound as zeros") > 0);
        CHECK(!f.backend.failedBuffers.empty());
        f.renderer.submit();
    }
}

// The texture probe (first 300 submits): a pass that writes a large colour target logs the T# and the backing of every
// texture it samples, once; a small pass logs nothing; a texture that does not fit the mapping logs a texfail line
// with the address, the bytes the layout needs, the bytes mapped and the tile mode.
size_t countLines(const Fixture& f, const char* needle) {
    size_t n = 0;
    for (const std::string& l : f.log) n += l.find(needle) != std::string::npos;
    return n;
}
void textureProbe() {
    {
        Fixture f;
        vpps4::DrawState d = f.state();
        d.color[0] = {kBase + 0x40000, 1024, 1024, 1024, 10, 0, 0, 14};  // an eye-sized target
        uint32_t* ttable = f.at(kBase + 0x200);
        ttable[3] = (ttable[3] & ~(31u << 20)) | (14u << 20) | (1u << 16);  // thin_2d_thin, levels 0..1
        f.renderer.draw(d);
        f.renderer.draw(d);
        CHECK(f.renderer.stats().drawn == 2);
        CHECK(countLines(f, "vpps4.texprobe") == 1);
        bool ok = false;
        for (const std::string& l : f.log)
            if (l.find("vpps4.texprobe") != std::string::npos)
                ok = l.find("pass=[cb0 0x10040000 1024x1024 pitch=1024 tile=14") != std::string::npos &&
                     l.find("T# base=0x10010000 64x64 depth=1 pitch=64 tile=14 fmt=10/0 type=9 level=0..1 array=0..0") != std::string::npos &&
                     l.find("dst_sel=4567") != std::string::npos && l.find("backing=texture from memory") != std::string::npos &&
                     l.find("tile=14 fmt=") != std::string::npos && l.find("tiledOf.bytes=") != std::string::npos &&
                     l.find("levels=2") != std::string::npos && l.find("fill=gpu") != std::string::npos && l.find("fills=1") != std::string::npos;
        CHECK(ok);
        if (!ok)
            for (const std::string& l : f.log) std::fprintf(stderr, "  %s\n", l.c_str());
        f.renderer.submit();
    }
    {
        Fixture f;  // a small pass: nothing to say
        f.renderer.draw(f.state());
        CHECK(f.renderer.stats().drawn == 1);
        CHECK(countLines(f, "vpps4.texprobe") == 0);
    }
    {
        Fixture f;  // past the first 300 submits: nothing either
        for (int i = 0; i < 300; ++i) f.renderer.submit();
        vpps4::DrawState d = f.state();
        d.color[0] = {kBase + 0x40000, 1024, 1024, 1024, 10, 0, 0, 8};
        f.renderer.draw(d);
        CHECK(countLines(f, "vpps4.texprobe") == 0);
    }
}

void textureFailDetail() {
    Fixture f;
    vpps4::DrawState d = f.state();
    d.color[0] = {kBase + 0x40000, 1024, 1024, 1024, 10, 0, 0, 14};
    uint32_t* ttable = f.at(kBase + 0x200);
    const uint64_t tex = kBase + kSize - 0x1000;  // 4096 bytes mapped, a 64x64 2D tiled texture needs 32768 (level 0 is padded to one 128x64 macro tile)
    ttable[0] = uint32_t(tex >> 8);
    ttable[1] = uint32_t(tex >> 40) | (10u << 20);
    ttable[3] = (ttable[3] & ~(31u << 20)) | (14u << 20);
    f.renderer.draw(d);
    f.renderer.draw(d);
    CHECK(f.renderer.stats().notes.count("tiled texture outside guest memory: left unfilled") == 1);
    CHECK(countLines(f, "vpps4.texfail f=0 tiled outside kind=texture") == 1);  // once per address and size
    bool ok = false;
    for (const std::string& l : f.log)
        if (l.find("vpps4.texfail") != std::string::npos)
            ok = ok || (l.find("tiled outside kind=texture addr=0x100ff000 tile=14") != std::string::npos &&
                        l.find("expected=32768 mapped=4096") != std::string::npos);
    CHECK(ok);
    CHECK(countLines(f, "fill=unfilled(tiled outside guest memory)") == 1);
    if (failures)
        for (const std::string& l : f.log) std::fprintf(stderr, "  %s\n", l.c_str());
}

// The app's shadow renderer follows the real one on the NullBackend and must never write guest memory (the region the
// hook gives is the guest's own, not a mirror): draws, a dispatch, DMA, a resolve, write backs, syncs, CPU-write
// notices and the scheduler's submits and waits all run on memory the process cannot write (a write is a fault).
// Not in the list: zpassDone / drainOcclusion, which write results into guest memory by design; the shadow's host
// never calls them (it leaves ZPASS_DONE to the real renderer), and labels, which the host writes itself.
void onShadowFault(int) {
    static const char msg[] = "  FAIL: the renderer wrote guest memory it must only read\n";
    (void)!::write(2, msg, sizeof msg - 1);
    ::_exit(1);
}

void shadowNeverWritesGuestMemory() {
    Sched x;
    x.f.backend.record = false;       // what the shadow host sets: no history kept, frames finish as they are submitted
    x.f.backend.autoComplete = true;
    x.f.renderer.setMaxSamples(1);
    const vpps4::DrawState d = x.f.state();  // (writes the descriptor tables: before the memory is locked)
    static const uint32_t code[] = {0xD8FA0000u, 0x01000000u, 0xBF810000u};  // ds_append v1 gds, s_endpgm
    std::vector<uint8_t> before(kSize);
    std::memcpy(before.data(), x.f.host, kSize);
    struct sigaction sa {}, old {};
    sa.sa_handler = onShadowFault;
    sigaction(SIGSEGV, &sa, &old);
    sigaction(SIGBUS, &sa, nullptr);
    CHECK(mprotect(x.f.host, kSize, PROT_READ) == 0);

    x.f.renderer.draw(d);
    x.f.renderer.draw(d);
    x.s.noteWork();
    x.s.flush();
    vpps4::DispatchState cs;
    cs.cs.code = code;
    cs.cs.address = kBase + 0x82000;
    cs.groups = {4, 1, 1};
    x.f.renderer.dispatch(cs);
    cs.indirectAddress = kBase + 0x3000;
    x.f.renderer.dispatch(cs);
    CHECK(x.f.renderer.fill(kBase + 0x5000, 256, 0x3f800000u));
    CHECK(x.f.renderer.fill(kBase + 0x5000, 256, 0));
    CHECK(x.f.renderer.copy(kBase + 0x6000, kBase + 0x5000, 256));
    CHECK(x.f.renderer.fill(d.color[0].address + 0x100, 256, 0x12345678u));  // memory the target on the GPU is ahead of
    CHECK(x.f.renderer.copy(kBase + 0x7000, d.color[0].address, 256));
    CHECK(x.f.renderer.writeBack(d.color[0]));
    x.f.renderer.syncToMemory(d.color[0].address, 0x1000);
    vpps4::ColorTarget src = d.color[0], dst = src;
    src.samples = 4;
    dst.address = kBase + 0x70000;
    x.f.renderer.resolve(src, dst);
    x.f.renderer.invalidate(kBase + 0x10100, 4);  // the CPU wrote the texture: refilled from memory
    x.f.renderer.coherencePoint(kBase + 0x10000, 0x1000);
    x.f.renderer.noteGpuWrite(kBase + 0x20000, 0x100);
    x.f.renderer.draw(d);
    x.s.syncCpu(kBase + 0x5000, 256, vpgpu::SubmitReason::Dma);
    x.s.syncGds();
    x.s.present();
    x.f.renderer.drainOcclusion(false);  // (nothing was counted: no zpassDone)
    x.s.finish();
    x.f.renderer.drainOcclusion(true);

    mprotect(x.f.host, kSize, PROT_READ | PROT_WRITE);
    sigaction(SIGSEGV, &old, nullptr);
    CHECK(std::memcmp(before.data(), x.f.host, kSize) == 0);
    CHECK(x.f.renderer.stats().drawn >= 3);
    CHECK(x.f.renderer.stats().dispatched >= 1);
    CHECK(x.f.backend.calls.empty() && x.f.backend.draws.empty() && x.f.backend.passes.empty());  // record = false
    CHECK(x.f.device.submittedFrames() > 0 && x.f.device.completedFrames() == x.f.device.submittedFrames());
    if (failures)
        for (const std::string& l : x.f.log) std::fprintf(stderr, "  %s\n", l.c_str());
}

// Hooks::seed: the host gives the texels of a surface the renderer is about to use (the colour target here); a surface
// it gives none for is filled from memory as before. Asked at every use, and the texture of the draw is asked too.
void seedHook() {
    Fixture f;
    std::vector<uint8_t> pixels;
    std::vector<vpps4::SeedRequest> asked;
    f.seed = [&](const vpps4::SeedRequest& q, vpps4::SeedImage& img) {
        asked.push_back(q);
        if (q.kind != vpps4::SeedRequest::Kind::Color) return false;
        pixels.assign(size_t(q.width) * q.height * vpgpu::formatInfo(q.format).bytesPerBlock, 0x7f);
        img.data = pixels.data();
        return true;
    };
    f.renderer.draw(f.state());
    CHECK(f.renderer.stats().drawn == 1);
    CHECK(f.renderer.stats().seeded == 1);
    bool color = false, texture = false;
    for (const vpps4::SeedRequest& q : asked) {
        color |= q.kind == vpps4::SeedRequest::Kind::Color && q.address == kBase + 0x40000 && q.width == 128 && q.height == 128;
        texture |= q.kind == vpps4::SeedRequest::Kind::Texture && q.address == kBase + 0x10000;
    }
    CHECK(color && texture);
    CHECK(f.renderer.stats().copies == 2);  // the seed of the target and the upload of the texture (memory, not seeded)
    f.renderer.submit();
    f.renderer.draw(f.state());  // asked again at the next use
    CHECK(f.renderer.stats().seeded == 2);
    f.seed = nullptr;            // no seed: from memory as usual
    f.renderer.draw(f.state());
    CHECK(f.renderer.stats().seeded == 2 && f.renderer.stats().drawn == 3);
    f.renderer.submit();
}

// Renderer::setGuestReadOnly: the draws run, everything that would write guest memory by the renderer's own work is
// refused (write back, DMA, ZPASS_DONE, dispatch) or never happens (re-tile), on memory the process cannot write.
void guestReadOnly() {
    Sched x;
    x.f.backend.record = false;
    x.f.backend.autoComplete = true;
    x.f.renderer.setMaxSamples(1);
    x.f.renderer.setGuestReadOnly(true);
    const vpps4::DrawState d = x.f.state();  // (writes the descriptor tables: before the memory is locked)
    static const uint32_t code[] = {0xD8FA0000u, 0x01000000u, 0xBF810000u};  // ds_append v1 gds, s_endpgm
    std::vector<uint8_t> before(kSize);
    std::memcpy(before.data(), x.f.host, kSize);
    struct sigaction sa {}, old {};
    sa.sa_handler = onShadowFault;
    sigaction(SIGSEGV, &sa, &old);
    sigaction(SIGBUS, &sa, nullptr);
    CHECK(mprotect(x.f.host, kSize, PROT_READ) == 0);

    x.f.renderer.draw(d);
    x.f.renderer.draw(d);
    x.s.noteWork();
    x.s.flush();
    vpps4::DispatchState cs;
    cs.cs.code = code;
    cs.cs.address = kBase + 0x82000;
    cs.groups = {4, 1, 1};
    x.f.renderer.dispatch(cs);
    CHECK(!x.f.renderer.fill(kBase + 0x5000, 256, 0x3f800000u));
    CHECK(!x.f.renderer.copy(kBase + 0x6000, kBase + 0x5000, 256));
    CHECK(!x.f.renderer.writeBack(d.color[0]));
    CHECK(!x.f.renderer.zpassDone(kBase + 0x8000, 2));
    CHECK(!x.f.renderer.syncToMemory(d.color[0].address, 0x1000));  // (nothing to re-tile: nothing recorded)
    x.f.renderer.invalidate(kBase + 0x10100, 4);
    x.f.renderer.draw(d);
    x.s.syncCpu(kBase + 0x5000, 256, vpgpu::SubmitReason::Dma);
    x.s.present();
    x.s.finish();

    mprotect(x.f.host, kSize, PROT_READ | PROT_WRITE);
    sigaction(SIGSEGV, &old, nullptr);
    CHECK(std::memcmp(before.data(), x.f.host, kSize) == 0);
    CHECK(x.f.renderer.stats().drawn >= 3);
    CHECK(x.f.renderer.stats().dispatched == 1);  // (it writes the GDS, not guest memory)
    CHECK(x.f.renderer.stats().retiled == 0 && x.f.renderer.stats().gpuRetiled == 0);
    if (failures)
        for (const std::string& l : x.f.log) std::fprintf(stderr, "  %s\n", l.c_str());
}

// setGuestReadOnly: a dispatch that stores through a V# is left out (counted); the same dispatch runs without it.
void guestReadOnlyBufferWrite() {
    for (int readOnly = 0; readOnly < 2; ++readOnly) {
        Fixture f;
        f.renderer.setGuestReadOnly(readOnly != 0);
        const Program cs = load("cs_bytes");
        vpps4::DispatchState d;
        d.cs.code = cs.code;
        d.cs.address = kBase + 0x82000;
        d.cs.userSgprs = 4;
        d.cs.ldsBytes = 64;
        d.cs.workgroupSize = {64, 1, 1};
        d.cs.userData[0] = uint32_t(kBase + 0x4000);
        d.cs.userData[1] = uint32_t((kBase + 0x4000) >> 32);
        d.cs.userData[2] = 0x1000;
        f.renderer.dispatch(d);
        f.renderer.submit();
        CHECK(f.renderer.stats().dispatched == (readOnly ? 0u : 1u));
        CHECK(f.renderer.stats().skipped.count("cs: guest memory read-only") == (readOnly ? 1u : 0u));
    }
}

// Hooks::seed with a clear: the clear applies to the seeded pixels (the seed is taken before it, not after).
void seedThenClear() {
    Fixture f;
    uint32_t asked = 0;
    std::vector<uint8_t> pixels;
    f.seed = [&](const vpps4::SeedRequest& q, vpps4::SeedImage& img) {
        if (q.kind != vpps4::SeedRequest::Kind::Depth || asked) return false;  // (given once, as a host does)
        ++asked;
        pixels.assign(size_t(q.width) * q.height * 4, 0);
        img.data = pixels.data();
        return true;
    };
    vpps4::DrawState c = f.state();
    c.primitive = vpps4::PrimitiveType::RectList;
    c.depth = {kBase + 0x60000, 0, 128, 128, 3, 0, 0, 0, 1, 1, true, false, 0.25f, 0};  // a Z clear
    f.renderer.draw(c);
    vpps4::DrawState n = f.state();
    n.depth = c.depth;
    n.depth.depthClear = false;
    n.depthStencil.depthTest = n.depthStencil.depthWrite = true;
    f.renderer.draw(n);
    f.renderer.submit();
    CHECK(asked == 1 && f.renderer.stats().seeded == 1);  // seeded by the clear draw, once
    bool cleared = false;
    for (const vpgpu::RenderPassDesc& p : f.backend.passes) cleared |= p.depth.load == vpgpu::LoadAction::Clear;
    CHECK(cleared);  // the clear was not dropped by a later seed
}

// A D32FloatS8 target takes both planes of a seed.
void seedDepthStencil() {
    Fixture f;
    std::vector<uint8_t> z, st;
    f.seed = [&](const vpps4::SeedRequest& q, vpps4::SeedImage& img) {
        if (q.kind != vpps4::SeedRequest::Kind::Depth) return false;
        CHECK(q.format == vpgpu::Format::D32FloatS8);
        z.assign(size_t(q.width) * q.height * 4, 0);
        st.assign(size_t(q.width) * q.height, 3);
        img.data = z.data();
        img.stencil = st.data();
        return true;
    };
    const auto depthDraw = [](Fixture& x) {
        vpps4::DrawState n = x.state();
        n.depth = {kBase + 0x60000, kBase + 0x70000, 128, 128, 3, 1, 0, 0, 1, 1, false, false, 1.0f, 0};
        n.depthStencil.depthTest = n.depthStencil.depthWrite = true;
        return n;
    };
    f.renderer.draw(depthDraw(f));
    f.renderer.submit();
    CHECK(f.renderer.stats().seeded == 1);
    CHECK(f.renderer.stats().copies == 3);  // the colour target's seed is not asked for (hook says no), so: Z, stencil, texture
    const vpgpu::Copy *zc = nullptr, *sc = nullptr;
    for (const vpgpu::Copy& c : f.backend.copies) {
        if (c.label.rfind("seed stencil", 0) == 0) sc = &c;
        else if (c.label.rfind("seed ", 0) == 0) zc = &c;
    }
    CHECK(zc && sc);
    if (zc && sc) {
        CHECK(zc->dstTexture.aspect != vpgpu::Aspect::Stencil && sc->dstTexture.aspect == vpgpu::Aspect::Stencil);
        CHECK(zc->srcBuffer.size == 128u * 128 * 4 && sc->srcBuffer.size == 128u * 128);
        CHECK(sc->srcBuffer.offset == zc->srcBuffer.offset + zc->srcBuffer.size && sc->srcBuffer.bytesPerRow == 128);  // (after the Z plane, tight)
        CHECK(sc->dstTexture.width == 128 && sc->dstTexture.height == 128);
    }
    // A host that gives only Z for a combined surface is not taken: the stencil plane would be left undefined.
    Fixture g;
    g.seed = [&](const vpps4::SeedRequest& q, vpps4::SeedImage& img) {
        if (q.kind != vpps4::SeedRequest::Kind::Depth) return false;
        z.assign(size_t(q.width) * q.height * 4, 0);
        img.data = z.data();
        return true;
    };
    g.renderer.draw(depthDraw(g));
    g.renderer.submit();
    CHECK(g.renderer.stats().seeded == 0 && g.renderer.stats().drawn == 1);
}

// A program rewritten in place without a notice of the CPU write (invalidate) is a new program: the key remembered for
// its address is checked against the code's content.
void programRewrittenInPlace() {
    Fixture f;
    Program a = load("cs_bytes"), b = load("cs_partial_store");
    const size_t words = std::max(a.code.size(), b.code.size());
    a.code.resize(words, 0xBF810000u);  // (s_endpgm padding: never reached)
    b.code.resize(words, 0xBF810000u);
    std::vector<uint32_t> program = a.code;
    vpps4::DispatchState d;
    d.cs.address = kBase + 0x82000;
    d.cs.userSgprs = 4;
    d.cs.ldsBytes = 64;
    d.cs.workgroupSize = {64, 1, 1};
    d.cs.userData[0] = uint32_t(kBase + 0x4000);
    d.cs.userData[1] = uint32_t((kBase + 0x4000) >> 32);
    d.cs.userData[2] = 0x1000;
    d.cs.code = program;
    f.renderer.dispatch(d);
    f.renderer.dispatch(d);
    const uint64_t translated = f.renderer.stats().shaders;
    CHECK(translated >= 1 && f.renderer.stats().shaderErrors == 0);
    program = b.code;  // in place, same address and size, no invalidate
    f.renderer.dispatch(d);
    CHECK(f.renderer.stats().shaders == translated + 1);
    CHECK(f.renderer.stats().shaderErrors == 0);
    // A longer program (3 KiB: past the old 1 KiB whole-hash limit, where 16 lines of 64 bytes were all that was looked at):
    // one word changed between two of the old lines is a new program as well.
    a.code.resize(768, 0xBF810000u);
    program = a.code;
    d.cs.code = program;  // (the vector grew)
    d.cs.address = kBase + 0x83000;
    f.renderer.dispatch(d);
    f.renderer.dispatch(d);
    const uint64_t longProgram = f.renderer.stats().shaders;
    program[75] = 0xBF800000u;  // s_nop in the padding after s_endpgm (byte 300: between two lines of the old sample)
    f.renderer.dispatch(d);
    CHECK(f.renderer.stats().shaders == longProgram + 1);
}

// Resolved::color[].cleared is the renderer's own fact: the pass of the draw starts with a clear that waited for the
// target (a fast clear drawn before), not the register bit of the draw.
void resolvedCleared() {
    Fixture f;
    f.renderer.draw(f.state());
    CHECK(f.resolved.back().status == vpps4::Resolved::Status::Drawn && !f.resolved.back().color[0].cleared);
    vpps4::DrawState c = f.state();
    c.primitive = vpps4::PrimitiveType::RectList;
    c.color[0].fastClear = true;
    f.renderer.draw(c);
    CHECK(!f.resolved.back().color[0].cleared);  // (the clear waits for the next pass: nothing started with it yet)
    vpps4::DrawState n = f.state();
    n.color[0].fastClear = true;                 // (the register bit alone says nothing)
    f.renderer.draw(n);
    CHECK(f.resolved.back().status == vpps4::Resolved::Status::Drawn && f.resolved.back().color[0].cleared);
    f.renderer.draw(f.state());
    CHECK(!f.resolved.back().color[0].cleared);  // consumed
    f.renderer.submit();
}

// A target's clear that waits for its pass still applies after a seed (the host took the texels before that pass): the
// pass loads with a clear and reports it; a texture's waiting clear is dropped for the seed's texels.
void seedThenColorClear() {
    Fixture f;
    std::vector<uint8_t> pixels;
    f.seed = [&](const vpps4::SeedRequest& q, vpps4::SeedImage& img) {
        if (q.kind != vpps4::SeedRequest::Kind::Color) return false;
        pixels.assign(size_t(q.width) * q.height * vpgpu::formatInfo(q.format).bytesPerBlock, 0x7f);
        img.data = pixels.data();
        return true;
    };
    vpps4::DrawState c = f.state();
    c.primitive = vpps4::PrimitiveType::RectList;
    c.color[0].fastClear = true;
    f.renderer.draw(c);
    vpps4::DrawState n = f.state();
    n.color[0].fastClear = true;
    f.renderer.draw(n);
    f.renderer.submit();
    CHECK(f.renderer.stats().seeded >= 1);
    CHECK(f.resolved.back().status == vpps4::Resolved::Status::Drawn && f.resolved.back().color[0].cleared);
    bool cleared = false;
    for (const vpgpu::RenderPassDesc& p : f.backend.passes) cleared |= p.colors[0].load == vpgpu::LoadAction::Clear;
    CHECK(cleared);
}

// A texture holding the host's seed is not looked at in memory (its source is not what it holds): a later use without a
// seed does not take the difference for a CPU write and fill it from memory.
void seededTextureNotChecked() {
    Fixture f;
    std::vector<uint8_t> pixels;
    f.seed = [&](const vpps4::SeedRequest& q, vpps4::SeedImage& img) {
        if (q.kind != vpps4::SeedRequest::Kind::Texture) return false;
        pixels.assign(size_t(q.width) * q.height * vpgpu::formatInfo(q.format).bytesPerBlock, 0x7f);
        img.data = pixels.data();
        return true;
    };
    f.renderer.draw(f.state());
    f.renderer.submit();
    CHECK(f.renderer.stats().seeded == 1 && f.renderer.stats().refilled == 0);
    f.seed = nullptr;
    *f.at(kBase + 0x10000) ^= 0xff;  // (the memory differs from what the texture holds)
    f.renderer.draw(f.state());
    CHECK(f.renderer.stats().drawn == 2 && f.renderer.stats().refilled == 0);
}

// The guest buffers (no-copy wraps of the host's windows) are given up on request, destroyed, and made afresh by the next use.
void dropGuestBuffers() {
    Fixture f;
    const std::string window = " size=" + std::to_string(kSize) + " guest=true";  // (the window over the fixture's memory)
    const auto windows = [&] {
        size_t n = 0;
        for (const std::string& c : f.backend.calls) n += c.rfind("createBuffer ", 0) == 0 && c.find(window) != std::string::npos;
        return n;
    };
    f.renderer.draw(f.state());
    f.renderer.submit();
    CHECK(windows() == 1);
    f.renderer.draw(f.state());
    f.renderer.submit();
    CHECK(windows() == 1);  // (the same one serves)
    std::string index;
    for (const std::string& c : f.backend.calls)
        if (c.rfind("createBuffer ", 0) == 0 && c.find(window) != std::string::npos) index = c.substr(13, c.find(' ', 13) - 13);
    CHECK(f.renderer.dropGuestBuffers());
    CHECK(std::count(f.backend.calls.begin(), f.backend.calls.end(), "destroyBuffer " + index) == 1);
    f.renderer.draw(f.state());
    f.renderer.submit();
    CHECK(windows() == 2);  // a new one over the memory, as after an unmap and a map at the same address
}

// The number after `key` in the last VPKENGINE_PROBE P7 line of the log (-1: no such line or key).
long long probeP7(const Fixture& f, const std::string& key) {
    for (auto it = f.log.rbegin(); it != f.log.rend(); ++it) {
        if (it->find("VPKENGINE_PROBE P7") == std::string::npos) continue;
        const size_t at = it->find(" " + key + "=");
        return at == std::string::npos ? -1 : std::stoll(it->substr(at + key.size() + 2));
    }
    return -1;
}

// A small read-only buffer the renderer already reads on the CPU (its DMAs run on the CPU) is copied to the upload ring when the
// draw is recorded: the draw does not hold the guest bytes, so a CPU DMA there right after neither submits nor waits (it waited
// for the draw before: WAR). Bound in place when the GPU wrote it last, when a label write into it is pending, or with
// VPKENGINE_BUFFER_LIVE=1.
void bufferSnapshot() {
    namespace pr = vpgpu::probe;
    pr::setEnabled(true);
    pr::registry().reset();
    const Program vt = load("vs_transform");
    const auto drawWith = [&](Sched& x, uint64_t cb, uint32_t bytes) {
        uint32_t* vtable = x.f.at(kBase + 0x100);
        const uint64_t vb = kBase + 0x1000;
        vtable[0] = uint32_t(vb);
        vtable[1] = uint32_t(vb >> 32) | (16u << 16);
        vtable[2] = 64;
        vtable[3] = (7u << 12) | (14u << 15);
        vpps4::DrawState d = x.f.state();
        d.vs.code = vt.code;
        d.vs.fetch = vt.fetch;
        d.vs.userSgprs = vt.userSgprs;
        d.vs.vgprCompCnt = 0;
        d.vs.userData = {};
        d.vs.userData[0] = uint32_t(kBase + 0x100);
        d.vs.userData[4] = uint32_t(cb);
        d.vs.userData[5] = uint32_t(cb >> 32);
        d.vs.userData[6] = bytes;
        d.vs.userData[7] = (7u << 12) | (14u << 15);
        x.f.renderer.draw(d);
        x.s.noteWork();
    };
    // Read on the CPU once (an unaligned constant buffer over it is copied), then the GPU goes idle.
    const auto readOnCpu = [&](Sched& x, uint64_t at) {
        drawWith(x, at + 8, 400);
        x.s.finish();
        CHECK(x.f.renderer.cpuReadSeen(at, 400) && !x.f.renderer.gpuMayAccess(at, 400));
    };
    // The first word of each constant buffer of `bytes` bytes the draws recorded so far are bound to (read once the frame ran).
    const auto bound = [&](Sched& x, uint32_t bytes) {
        std::vector<uint32_t> words;
        for (const vpgpu::Draw& d : x.f.backend.draws)
            for (const vpgpu::Access& a : d.bindings) {
                const auto m = x.f.backend.memory.find(a.range.index);
                if (a.range.kind != vpgpu::ResourceRange::Kind::Buffer || a.range.end - a.range.begin != bytes || m == x.f.backend.memory.end())
                    continue;
                uint32_t v;
                std::memcpy(&v, static_cast<const uint8_t*>(m->second) + a.range.begin, 4);
                words.push_back(v);
            }
        return words;
    };
    const auto snapshots = [&](Sched& x) -> long long {  // (the probe counter buffer_snapshots)
        x.f.renderer.publishProbe(1);
        std::string all;
        for (const std::string& l : pr::registry().sample(1.0))
            if (l.find("VPKENGINE_PROBE RATES ") == 0) all += l;
        const size_t at = all.find(" buffer_snapshots=");
        return at == std::string::npos ? -1 : std::stoll(all.substr(at + 18));
    };
    const uint64_t cb = kBase + 0x30000;
    {
        HostDma h;
        Sched& x = h.x;
        readOnCpu(x, cb);
        *x.f.at(cb) = 0xa1;
        *x.f.at(cb + 0x1a0) = 0xb2;
        const size_t submits = x.submits();
        const long long snaps = snapshots(x);
        drawWith(x, cb, 400);
        CHECK(!x.f.renderer.gpuMayAccess(cb, 400));
        CHECK(h.copy(cb, cb + 0x1a0, 400));  // the next record over the current one, as the title does: on the CPU, free
        CHECK(x.submits() == submits && x.f.backend.waits.size() == 1 && x.s.counters().cpuSyncs == 0);
        CHECK(*x.f.at(cb) == 0xb2);
        drawWith(x, cb, 400);  // the next draw copies the new bytes
        CHECK(!x.f.renderer.gpuMayAccess(cb, 400));
        CHECK(snapshots(x) == snaps + 2);
        x.s.finish();
        {  // (the draw before the DMA got the bytes from before it, the one after the bytes it left)
            const std::vector<uint32_t> words = bound(x, 400);
            CHECK(words.size() >= 2 && words[words.size() - 2] == 0xa1 && words[words.size() - 1] == 0xb2);
        }
        // The GPU wrote it last: in place.
        readOnCpu(x, cb + 0x1000);
        x.f.renderer.noteGpuWrite(cb + 0x1000, 4);
        drawWith(x, cb + 0x1000, 400);
        CHECK(x.f.renderer.gpuMayAccess(cb + 0x1000, 400));
        // A label write into it pending: in place.
        readOnCpu(x, cb + 0x2000);
        x.work(kBase + 0x5000);  // (the label waits for this work)
        x.s.scheduleLabel(cb + 0x2010, 4, false, [] {});
        CHECK(x.f.renderer.labelPending(cb + 0x2010, 4));
        drawWith(x, cb + 0x2000, 400);
        CHECK(x.f.renderer.gpuMayAccess(cb + 0x2000, 400));
        // Never read on the CPU: in place (its DMAs go on the GPU timeline).
        drawWith(x, cb + 0x3000, 400);
        CHECK(x.f.renderer.gpuMayAccess(cb + 0x3000, 400));
        // Up to 4 KiB; more is bound in place.
        readOnCpu(x, cb + 0x8000);
        readOnCpu(x, cb + 0x9000);
        const long long snaps2 = snapshots(x);
        drawWith(x, cb + 0x9000, 4096);
        CHECK(!x.f.renderer.gpuMayAccess(cb + 0x9000, 4096) && snapshots(x) == snaps2 + 1);
        drawWith(x, cb + 0x8000, 4097);
        CHECK(x.f.renderer.gpuMayAccess(cb + 0x8000, 4097) && snapshots(x) == snaps2 + 1);
        // Not 16-byte aligned: not a snapshot.
        readOnCpu(x, cb + 0xa000);
        *x.f.at(cb + 0xa008) = 0xc3;
        drawWith(x, cb + 0xa008, 400);
        CHECK(snapshots(x) == snaps2 + 1);
        x.s.finish();
        CHECK(!bound(x, 400).empty() && bound(x, 400).back() == 0xc3);
        // The tracker forgot its history (past its interval limit): who wrote last is not known until the work recorded before
        // has finished, so it is bound in place; later, copied again.
        readOnCpu(x, cb + 0xb000);
        for (uint64_t i = 0; i < 140000; ++i) x.f.renderer.noteGpuWrite(0x7000000000ull + i * 64, 4);
        CHECK(x.f.renderer.tracker().forgetHistoryCalls() >= 1);
        const long long snaps3 = snapshots(x);
        drawWith(x, cb + 0xb000, 400);
        CHECK(x.f.renderer.gpuMayAccess(cb + 0xb000, 400) && snapshots(x) == snaps3);
        x.s.finish();
        drawWith(x, cb + 0xb000, 400);
        CHECK(!x.f.renderer.gpuMayAccess(cb + 0xb000, 400) && snapshots(x) == snaps3 + 1);
    }
    {  // A V# buffer inside a range the command processor wrote (cpuAccessed) is copied too, even when the CPU reads are no longer
       // followed for new ranges (cpuAccessed added it to cpuRead only below 1024 of them): the pointer rule's conditions hold.
        HostDma h;
        Sched& x = h.x;
        readOnCpu(x, cb + 0x5000);  // (the draw's own CPU reads, noted before the ranges below)
        for (uint64_t i = 0; i < 1100; ++i) {
            const uint64_t far = 0x7100000000ull + i * 0x400000;
            x.f.renderer.cpuAccessed(far, 4);
            if (!x.f.renderer.cpuReadSeen(far, 4)) break;
        }
        x.f.renderer.cpuAccessed(cb + 0x6000, 0x1000);
        CHECK(!x.f.renderer.cpuReadSeen(cb + 0x6000, 256));
        const long long snaps = snapshots(x);
        drawWith(x, cb + 0x6000, 256);
        CHECK(!x.f.renderer.gpuMayAccess(cb + 0x6000, 256) && snapshots(x) == snaps + 1);
    }
    {  // A FLAT store writes anywhere in the regions its table maps: they are bound in place.
        HostDma h;
        Sched& x = h.x;
        static const uint32_t code[] = {0x7e040200u, 0x7e060201u, 0x7e080202u, 0x7e0a0203u, 0xdc300000u, 0x0a000002u,
                                        0xdc700000u, 0x00000a04u, 0xBF810000u};  // (flat_load v10 <- [s0:1]; flat_store [s2:3] <- v10)
        vpps4::DispatchState d;
        d.cs.code = code;
        d.cs.address = kBase + 0x82000;
        d.cs.userSgprs = 4;
        d.cs.workgroupSize = {64, 1, 1};
        d.cs.userData[0] = uint32_t(cb);
        d.cs.userData[1] = uint32_t(cb >> 32);
        d.cs.userData[2] = uint32_t(cb + 0x100);
        d.cs.userData[3] = uint32_t((cb + 0x100) >> 32);
        x.f.renderer.dispatch(d);
        x.f.renderer.dispatch(d);
        CHECK(x.f.renderer.stats().shaderErrors == 0 && x.f.renderer.stats().skipped.empty());
        readOnCpu(x, cb);
        const long long snaps = snapshots(x);
        drawWith(x, cb, 400);
        CHECK(x.f.renderer.gpuMayAccess(cb, 400) && snapshots(x) == snaps);
    }
    // IT_WRITE_DATA over a small V# buffer the draws of the picture before are bound to (never read on the CPU: in place). The
    // write waits for them once (Scheduler::syncWrite notes the range); the draws recorded after it copy the buffer, so the
    // next picture's write neither submits nor waits, and each draw is bound to the bytes of its place in the command stream.
    for (const bool live : {false, true}) {
        if (live) setenv("VPKENGINE_WRITE_DATA_LIVE", "1", 1);
        HostDma h;
        unsetenv("VPKENGINE_WRITE_DATA_LIVE");
        Sched& x = h.x;
        const uint64_t wd = cb + 0xc000;
        const auto write = [&](uint32_t v) {  // (the host's WriteDataOrdered)
            const bool waited = x.s.syncWrite(wd, 256);
            *x.f.at(wd) = v;
            return waited;
        };
        *x.f.at(wd) = 0x11;
        drawWith(x, wd, 256);
        CHECK(x.f.renderer.gpuMayAccess(wd, 256));
        x.s.present();
        CHECK(write(0x22));  // (the picture before reads it in place)
        drawWith(x, wd, 256);
        CHECK(x.f.renderer.gpuMayAccess(wd, 256) == live);
        x.s.present();
        const size_t submits = x.submits(), waits = x.f.backend.waits.size();
        CHECK(write(0x33) == live);
        CHECK(live || (x.submits() == submits && x.f.backend.waits.size() == waits));
        drawWith(x, wd, 256);
        if (live) continue;
        // The GPU writes the range too (a store recorded into it): memory is not the newest until that runs, so the buffer is
        // bound in place again and the write after it waits, as before.
        x.f.renderer.noteGpuWrite(wd, 4);
        drawWith(x, wd, 256);
        CHECK(x.f.renderer.gpuMayAccess(wd, 256));
        CHECK(write(0x44));
        x.s.finish();
        const std::vector<uint32_t> words = bound(x, 256);  // (the first and the last draw read memory as it is now)
        CHECK(words.size() == 4 && words[1] == 0x22 && words[2] == 0x33 && words[3] == 0x44);
    }
    {  // A write of part of the buffer is noted like one of all of it; one of more than 4 KiB is not (such a buffer is not copied).
        HostDma h;
        Sched& x = h.x;
        const uint64_t part = cb + 0xd000, big = cb + 0xe000;
        drawWith(x, part, 256);
        x.s.present();
        drawWith(x, big, 256);
        x.s.present();
        CHECK(x.s.syncWrite(part + 0x40, 16) && x.s.syncWrite(big, 8192));  // (each waits for its own frame)
        drawWith(x, part, 256);
        drawWith(x, big, 256);
        CHECK(!x.f.renderer.gpuMayAccess(part, 256) && x.f.renderer.gpuMayAccess(big, 256));
        x.s.present();
        CHECK(!x.s.syncWrite(part + 0x40, 16) && x.s.syncWrite(big, 8192));
    }
    {
        setenv("VPKENGINE_BUFFER_LIVE", "1", 1);
        HostDma h;
        unsetenv("VPKENGINE_BUFFER_LIVE");
        Sched& x = h.x;
        readOnCpu(x, cb);
        drawWith(x, cb, 400);
        CHECK(x.f.renderer.gpuMayAccess(cb, 400));
        CHECK(h.copy(cb, cb + 0x1a0, 400));
        CHECK(x.s.counters().submits[size_t(vpgpu::SubmitReason::Dma)] == 1);
    }
    pr::setEnabled(false);
}

// Constant buffers at an address that is not 16-byte aligned are copied into the upload ring. A V# says how much (its size,
// not more); a bare address gets the usual 64 KiB window. Within a submit a buffer that was not written since is bound again
// from its copy, however many others were copied in between; one written since (told to the tracker, or only seen in its
// bytes) is copied afresh, and a new submit starts the copies over.
void alignedConstantCopies() {
    const Program vt = load("vs_transform");
    // vs_transform: vertices through the V# table at s[0:1] (+0x100), constants from the V# in s[4:7].
    const auto drawWith = [&](Fixture& f, uint64_t table, uint64_t cb, uint32_t bytes) {
        uint32_t* vtable = f.at(table);
        const uint64_t vb = kBase + 0x1000;
        vtable[0] = uint32_t(vb);
        vtable[1] = uint32_t(vb >> 32) | (16u << 16);
        vtable[2] = 64;
        vtable[3] = (7u << 12) | (14u << 15);
        vpps4::DrawState d = f.state();
        d.vs.code = vt.code;
        d.vs.fetch = vt.fetch;
        d.vs.userSgprs = vt.userSgprs;
        d.vs.vgprCompCnt = 0;
        d.vs.userData = {};
        d.vs.userData[0] = uint32_t(table);
        d.vs.userData[4] = uint32_t(cb);
        d.vs.userData[5] = uint32_t(cb >> 32);  // (stride 0: `bytes` is the size)
        d.vs.userData[6] = bytes;
        d.vs.userData[7] = (7u << 12) | (14u << 15);
        f.renderer.draw(d);
    };
    const auto cbAt = [](unsigned i) { return kBase + 0x20008 + uint64_t(i) * 0x40; };
    constexpr unsigned kN = 200;  // (more than the 64 copies remembered before)
    {
        Fixture f;
        for (unsigned i = 0; i < kN; ++i) drawWith(f, kBase + 0x100, cbAt(i), 64);
        CHECK(f.renderer.stats().drawn == kN);
        CHECK(f.renderer.takeCpuTimes().alignedCopyReused == 0);
        for (unsigned i = 0; i < kN; ++i) drawWith(f, kBase + 0x100, cbAt(i), 64);
        CHECK(f.renderer.stats().drawn == 2 * kN);
        CHECK(f.renderer.takeCpuTimes().alignedCopyReused == kN);
        // Rewritten by the CPU without a word to the renderer: found in its bytes, copied afresh, then served again.
        *f.at(cbAt(7)) ^= 0xff;
        drawWith(f, kBase + 0x100, cbAt(7), 64);
        CHECK(f.renderer.takeCpuTimes().alignedCopyReused == 0);
        drawWith(f, kBase + 0x100, cbAt(7), 64);
        CHECK(f.renderer.takeCpuTimes().alignedCopyReused == 1);
        // Told to the renderer (the CPU, or the GPU through a host-side write): not served from the old copy.
        f.renderer.invalidate(cbAt(8), 4);
        drawWith(f, kBase + 0x100, cbAt(8), 64);
        CHECK(f.renderer.takeCpuTimes().alignedCopyReused == 0);
        f.renderer.noteGpuWrite(cbAt(9) + 8, 4);  // (the last bytes of the buffer)
        drawWith(f, kBase + 0x100, cbAt(9), 64);
        CHECK(f.renderer.takeCpuTimes().alignedCopyReused == 0);
        drawWith(f, kBase + 0x100, cbAt(9), 64);
        CHECK(f.renderer.takeCpuTimes().alignedCopyReused == 1);
        // Another buffer, the same address: the copy of 64 bytes does not answer for 128.
        drawWith(f, kBase + 0x100, cbAt(10), 128);
        CHECK(f.renderer.takeCpuTimes().alignedCopyReused == 0);
        // The next submit starts over (the ring is another one).
        f.renderer.submit();
        drawWith(f, kBase + 0x100, cbAt(1), 64);
        CHECK(f.renderer.takeCpuTimes().alignedCopyReused == 0);
        drawWith(f, kBase + 0x100, cbAt(1), 64);
        CHECK(f.renderer.takeCpuTimes().alignedCopyReused == 1);
    }
    // What is copied: the V#'s size; a bare address (the table pointer is unaligned here, the V# constant buffer is not): 64 KiB.
    {
        Fixture f;
        drawWith(f, kBase + 0x100, kBase + 0x20008, 256);
        f.renderer.submit();
        CHECK(probeP7(f, "aligned_copy_bytes") == 256);
        Fixture g;
        drawWith(g, kBase + 0x108, kBase + 0x20000, 256);
        g.renderer.submit();
        CHECK(probeP7(g, "aligned_copy_bytes") == 64 * 1024);
    }
}

// The probe families the renderer publishes: counts of what it did and what it holds, per picture figures, and the
// draws taken per colour target.
void probePublish() {
    namespace pr = vpgpu::probe;
    pr::setEnabled(true);
    pr::registry().reset();
    Fixture f;
    f.renderer.draw(f.state());
    f.renderer.draw(f.state());
    f.renderer.submit();
    CHECK(f.renderer.takeTargetDraws(kBase + 0x40000, 128, 128) == 2);
    CHECK(f.renderer.takeTargetDraws(kBase + 0x40000, 128, 128) == 0);
    CHECK(f.renderer.takeTargetDraws(kBase + 0x40000, 7, 7) == 0);
    f.renderer.publishProbe(1);
    const std::vector<std::string> lines = pr::registry().sample(1.0);
    const auto find = [&](const char* fam) -> std::string {  // (a family's pieces, joined)
        std::string all;
        for (const std::string& l : lines)
            if (l.find(std::string("VPKENGINE_PROBE ") + fam + " ") == 0) all += l;
        return all;
    };
    const std::string rates = find("RATES"), eng = find("ENG"), qual = find("QUAL");
    CHECK(!rates.empty() && !eng.empty() && !qual.empty());
    CHECK(rates.find(" draws=2 draws_d=2 ") != std::string::npos);
    CHECK(rates.find(" drawn=2 ") != std::string::npos);
    CHECK(rates.find(" pictures=1 ") != std::string::npos);
    CHECK(eng.find(" tex_n=1 ") != std::string::npos);
    CHECK(eng.find(" tex_2d=1 ") != std::string::npos);
    CHECK(eng.find(" rt_n=1 ") != std::string::npos);
    CHECK(eng.find(" ring_mb=") != std::string::npos);
    CHECK(qual.find(" fill_linear_upload=1 ") != std::string::npos);
    CHECK(qual.find(" refill_first=1 ") != std::string::npos);
    // A second publication with nothing new: cumulative values stay, deltas are 0, and the line differs from the first (the frame counter).
    f.renderer.publishProbe(2);
    const std::vector<std::string> again = pr::registry().sample(2.0);
    bool sameDraws = false;
    for (const std::string& l : again)
        if (l.find("VPKENGINE_PROBE RATES ") == 0 && l.find(" draws=2 draws_d=0 ") != std::string::npos) sameDraws = true;
    CHECK(sameDraws);
    pr::setEnabled(false);
    pr::registry().reset();
}

// The probe's tile histogram (QUAL tile_plan_<tile>_<dfmt>_<nfmt>_deg<0|1>) and TILE line: a 64x64 32-bit texture of thin_2d_thin
// (macro tile 128x64) is padded to a whole macro tile on level 0, not stored 1D.
void tileProbe() {
    namespace pr = vpgpu::probe;
    pr::setEnabled(true);
    pr::registry().reset();
    Fixture f;
    vpps4::DrawState d = f.state();
    uint32_t* ttable = f.at(kBase + 0x200);
    ttable[3] = (ttable[3] & ~(31u << 20)) | (14u << 20);  // thin_2d_thin
    f.renderer.draw(d);
    f.renderer.submit();
    f.renderer.publishProbe(1);
    bool tileLine = false;
    for (const std::string& l : f.log)
        if (l.find("VPKENGINE_PROBE TILE ") == 0 && l.find(" tile=14 ") != std::string::npos && l.find(" w=64 h=64 mips=1 l0=128x64 degraded=0 degraded_levels=0") != std::string::npos)
            tileLine = true;
    CHECK(tileLine);
    std::string qual;
    for (const std::string& l : pr::registry().sample(1.0))
        if (l.find("VPKENGINE_PROBE QUAL ") == 0) qual += l;
    CHECK(qual.find(" tile_plan_14_10_0_deg0=1 ") != std::string::npos);
    CHECK(qual.find(" tile_plan_14_10_0_deg1=") == std::string::npos);
    pr::setEnabled(false);
    pr::registry().reset();
}

int main(int argc, char** argv) {
    const std::string t = argc > 1 ? argv[1] : "";
    if (t == "resolved_hook") resolvedHook();
    else if (t == "seed_hook") seedHook();
    else if (t == "guest_read_only") guestReadOnly();
    else if (t == "guest_read_only_buffer_write") guestReadOnlyBufferWrite();
    else if (t == "seed_then_clear") seedThenClear();
    else if (t == "seed_depth_stencil") seedDepthStencil();
    else if (t == "seed_then_color_clear") seedThenColorClear();
    else if (t == "seeded_texture_not_checked") seededTextureNotChecked();
    else if (t == "drop_guest_buffers") dropGuestBuffers();
    else if (t == "resolved_cleared") resolvedCleared();
    else if (t == "probe_publish") probePublish();
    else if (t == "tile_probe") tileProbe();
    else if (t == "program_rewritten_in_place") programRewrittenInPlace();
    else if (t == "clear_and_draw") clearAndDraw();
    else if (t == "textured") textured();
    else if (t == "linear_upload") linearUpload();
    else if (t == "tiled_upload") tiledUpload();
    else if (t == "mip_upload") mipUpload();
    else if (t == "write_back") writeBack();
    else if (t == "indirect") indirect();
    else if (t == "gds_append") gdsAppend();
    else if (t == "flat_access") flatAccess();
    else if (t == "compute_buffer_hook") computeBufferHook();
    else if (t == "dma") dma();
    else if (t == "content_check") contentCheck();
    else if (t == "fetch_subroutine") fetchSubroutine();
    else if (t == "fan") fan();
    else if (t == "shader_error") shaderError();
    else if (t == "msaa_limit") msaaLimit();
    else if (t == "resolve") resolveCopy();
    else if (t == "resolve_required") resolveRequired();
    else if (t == "runaway") runaway();
    else if (t == "storage_alias") storageAlias();
    else if (t == "busy_ranges") busyRanges();
    else if (t == "translation_cache") translationCache();
    else if (t == "translation_cache_dedupe") translationCacheDedupe();
    else if (t == "translation_cache_evict") translationCacheEvict();
    else if (t == "upload_ring_grows") uploadRingGrows();
    else if (t == "upload_ring_bounded") uploadRingBounded();
    else if (t == "geometry_scratch_reclaim") geometryScratchReclaim();
    else if (t == "translation_cache_failures") translationCacheFailures();
    else if (t == "translation_cache_stale_pipelines") translationCacheStalePipelines();
    else if (t == "draw_bench") drawBench();
    else if (t == "poly_offset") polyOffset();
    else if (t == "viewport_rects") viewportRects();
    else if (t == "blend_factors") blendFactors();
    else if (t == "stencil_ops") stencilOps();
    else if (t == "scissors") scissors();
    else if (t == "write_mask_swap") writeMaskSwap();
    else if (t == "clip_viewport") clipAndViewport();
    else if (t == "index_prep") indexPrep();
    else if (t == "lift_exports") liftExports();
    else if (t == "state_reaches_backend") stateReachesBackend();
    else if (t == "plane_clears") planeClears();
    else if (t == "depth_copy") depthCopyDraw();
    else if (t == "restart_index") restartIndexDraw();
    else if (t == "geometry_shader") geometryShader();
    else if (t == "geometry_indexed") geometryIndexed();
    else if (t == "geometry_bad_primitive") geometryBadPrimitive();
    else if (t == "rect_list") rectList();
    else if (t == "adjacency") adjacency();
    else if (t == "patch") patch();
    else if (t == "geometry_hoist") geometryHoist();
    else if (t == "draw_bench_heavy") drawBenchHeavy();
    else if (t == "one_frame_per_image") oneFramePerImage();
    else if (t == "partial_waits") partialWaits();
    else if (t == "dma_on_gpu") dmaOnGpu();
    else if (t == "dma_stale_read") dmaStaleRead();
    else if (t == "dma_after_cpu_read") dmaAfterCpuRead();
    else if (t == "dma_sync_probe") dmaSyncProbe();
    else if (t == "staging_pool") stagingPool();
    else if (t == "unknown_extent_busy") unknownExtentBusy();
    else if (t == "dma_then_label") dmaThenLabel();
    else if (t == "label_then_dma") labelThenDma();
    else if (t == "label_supersede_consumed") labelSupersedeConsumed();
    else if (t == "scheduler_no_clock") schedulerNoClock();
    else if (t == "buffer_snapshot") bufferSnapshot();
    else if (t == "label_keep_opt_out") labelKeepOptOut();
    else if (t == "dma_on_cpu_opt_out") dmaOnCpuOptOut();
    else if (t == "index_scan_memo") indexScanMemo();
    else if (t == "abandon_owners") abandonOwners();
    else if (t == "label_ordering") labelOrdering();
    else if (t == "label_waits_for_msaa_pass") labelWaitsForMsaaPass();
    else if (t == "gpu_ordered_label_wait") gpuOrderedLabelWait();
    else if (t == "finish_all") finishAll();
    else if (t == "hold_pipeline") holdPipeline();
    else if (t == "hold_budget") holdBudget();
    else if (t == "submit_times") submitTimes();
    else if (t == "region_cache") regionCache();
    else if (t == "window_crossing") windowCrossing();
    else if (t == "guest_buffer_refused") guestBufferRefused();
    else if (t == "report_sink") reportSink();
    else if (t == "occlusion") occlusion();
    else if (t == "shadow_never_writes_guest_memory") shadowNeverWritesGuestMemory();
    else if (t == "indirect_convert") indirectConvert();
    else if (t == "texture_probe") textureProbe();
    else if (t == "texture_fail_detail") textureFailDetail();
    else if (t == "aligned_constant_copies") alignedConstantCopies();
    else return std::fprintf(stderr, "unknown test %s\n", t.c_str()), 2;
    return failures ? 1 : 0;
}
