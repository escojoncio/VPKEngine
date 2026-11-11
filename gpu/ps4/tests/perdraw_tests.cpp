// The CPU cost per draw of vpps4::Renderer, and the proof that its per-draw shortcuts change nothing.
//
// A synthetic picture shaped like the measured load of a PSVR title (about 2000 draws in some 40 passes and
// 3 submits; runs of draws with one material whose constants and vertex buffers change; about 30 pixel
// programs; small V# buffers; constant buffers aligned and not; a few hundred textures), recorded through the
// renderer on a backend that does nothing (bench) or that writes down everything it is given (equivalence).
// Usage: vpps4_perdraw_tests TEST
//   bench                  prints us/draw by phase and ms/picture of submit (VPPS4_BENCH_PICTURES, default 60;
//                          VPPS4_BENCH_OFF=1: every shortcut switched off, see kOptOuts)
//   equal_picture ...      the same script with the shortcuts on and off: the captured command stream must be
//                          the same byte for byte
//   digest                 every script once, a hash of what it left per line (VPPS4_DIGEST_OFF=1: shortcuts off)
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <span>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <sys/mman.h>
#if __has_include(<valgrind/callgrind.h>)
#include <valgrind/callgrind.h>  // (bench under callgrind --collect-atstart=no: only the measured pictures are counted)
#else
#define CALLGRIND_TOGGLE_COLLECT
#endif

#include "vpgpu/core/device.h"
#include "vpgpu/core/pipeline_cache.h"
#include "vpgpu/core/probe.h"
#include "vpmsl/msl.h"
#include "vpps4/renderer.h"

#ifndef VPPS4_LIFT_DIR
#error "VPPS4_LIFT_DIR must point at gpu/shader/gcn/tests/lift"
#endif
#ifndef VPPS4_BENCH_LIFT_DIR
#error "VPPS4_BENCH_LIFT_DIR must point at gpu/ps4/tests/lift"
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

// The environment variables that switch the per-draw shortcuts off, each one alone (renderer.cpp, core/device.cpp).
struct OptOut {
    const char* name;
    const char* value;
};
const OptOut kOptOuts[] = {{"VPKENGINE_BIND_MEMO_OFF", "1"}, {"VPKENGINE_PROGRAM_MEMO_OFF", "1"}, {"VPKENGINE_HAZARD_MEMO_OFF", "1"},
                           {"VPVK_FRAME_FOLD", "0"}};
void setOptOuts(bool off, const char* only = nullptr) {
    for (const OptOut& o : kOptOuts) {
        if (off && (!only || std::strcmp(only, o.name) == 0)) setenv(o.name, o.value, 1);
        else unsetenv(o.name);
    }
}

struct Program {
    std::vector<uint32_t> code, fetch;
    uint8_t userSgprs = 0, vgprCompCnt = 0;
    uint32_t psInputAddr = 0;
};
Program load(const char* dir, const char* name) {
    std::ifstream f(std::string(dir) + "/" + name + ".lift");
    Program p;
    for (std::string line; std::getline(f, line);) {
        std::istringstream ls(line);
        std::string key;
        ls >> key;
        if (key == "user_sgprs" || key == "vgpr_comp_cnt" || key == "ps_input_addr") {
            std::string v;
            ls >> v;
            const uint32_t n = uint32_t(std::stoul(v, nullptr, 0));
            (key == "user_sgprs" ? p.userSgprs = uint8_t(n) : key == "vgpr_comp_cnt" ? p.vgprCompCnt = uint8_t(n) : uint8_t(p.psInputAddr = n));
        }
        std::vector<uint32_t>* dst = key == "code" ? &p.code : key == "fetch" ? &p.fetch : nullptr;
        if (!dst) continue;
        for (std::string t; ls >> t;) dst->push_back(uint32_t(std::stoul(t, nullptr, 16)));
    }
    if (p.code.empty()) std::fprintf(stderr, "missing program %s\n", name), std::exit(2);
    return p;
}

// ---- Guest memory ----------------------------------------------------------------------------------------------------
// 768 MiB of address space at guest address kGuest (pages are only committed when touched), handed to the renderer as six
// mirrored regions (as the emulator does: each mapping has a mirror the buffers alias, and a generation that changes with
// the map).
constexpr uint64_t kGuest = 0x7000000000ull;
constexpr uint64_t kCode = kGuest, kStatic = kGuest + 0x01000000, kArena = kGuest + 0x02000000, kMesh = kGuest + 0x04000000,
                   kTex = kGuest + 0x08000000, kTargets = kGuest + 0x10000000, kEnd = kGuest + 0x30000000;
constexpr uint64_t kArenaBytes = 0x00800000;  // each of the two arenas of per-draw constants

struct Guest {
    uint8_t* host = nullptr;
    uint64_t generation = 1;
    uint64_t regionCalls = 0, generationCalls = 0, watchWrites = 0, watchOwned = 0;
    Guest() {
        void* p = mmap(nullptr, kEnd - kGuest, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (p == MAP_FAILED) std::fprintf(stderr, "no guest memory\n"), std::exit(2);
        host = static_cast<uint8_t*>(p);
    }
    ~Guest() { munmap(host, kEnd - kGuest); }
    Guest(const Guest&) = delete;
    template <class T = uint8_t>
    T* at(uint64_t guest) { return reinterpret_cast<T*>(host + (guest - kGuest)); }
    uint32_t* words(uint64_t guest) { return at<uint32_t>(guest); }
    bool region(uint64_t a, uint64_t s, vpps4::GuestRegion& g) {
        ++regionCalls;
        static constexpr uint64_t bounds[] = {kCode, kStatic, kArena, kMesh, kTex, kTargets, kEnd};
        for (size_t i = 0; i + 1 < std::size(bounds); ++i)
            if (a >= bounds[i] && a + s <= bounds[i + 1]) {
                g = {bounds[i], bounds[i + 1] - bounds[i], at(bounds[i]), true};
                return true;
            }
        return false;
    }
};

// Descriptors as the hardware encodes them.
void vsharp(uint32_t* w, uint64_t address, uint32_t stride, uint32_t records) {
    w[0] = uint32_t(address);
    w[1] = uint32_t(address >> 32) | (stride << 16);
    w[2] = records;
    w[3] = (7u << 12) | (14u << 15);  // float 32_32_32_32
}
void tsharp(uint32_t* w, uint64_t address, uint32_t size, uint32_t tile) {
    w[0] = uint32_t(address >> 8);
    w[1] = uint32_t(address >> 40) | (10u << 20);  // 8_8_8_8 unorm
    w[2] = (size - 1) | ((size - 1) << 14);
    w[3] = 4 | (5u << 3) | (6u << 6) | (7u << 9) | (tile << 20) | (9u << 28);  // xyzw, 2D
    w[4] = (size - 1) << 13;  // pitch
    w[5] = w[6] = w[7] = 0;
}

// ---- The scene -------------------------------------------------------------------------------------------------------

struct Step {
    enum Kind : uint8_t { Draw, Submit, Call } kind = Draw;
    vpps4::DrawState draw;
    std::function<void(vpps4::Renderer&)> call;  // (Call: a guest write, an invalidate, a dispatch, ...)
    bool probe = false;                          // (Call: only asks the renderer, see probeStep)
};
using Script = std::vector<Step>;
Step probeStep(std::vector<std::pair<uint64_t, uint64_t>> ranges);

struct Scene {
    Guest& g;
    static constexpr unsigned kPixelPrograms = 30, kMaterials = 48, kMeshes = 64, kTextures = 256, kTexSize = 64;
    static constexpr unsigned kProgramWords = 384;  // (a program binary: its code and the constant tables that follow it)
    Program vsX16 = load(VPPS4_LIFT_DIR, "vs_cbuf_x16"), vsDword = load(VPPS4_LIFT_DIR, "vs_cbuf_dword"),
            vsFetch = load(VPPS4_LIFT_DIR, "vs_fetch"), psTextured = load(VPPS4_LIFT_DIR, "ps_textured"),
            psMaterial = load(VPPS4_BENCH_LIFT_DIR, "ps_material"), csBytes = load(VPPS4_LIFT_DIR, "cs_bytes");
    uint64_t codeNext = kCode, staticNext = kStatic;
    uint32_t rng = 0x2545F491;
    uint32_t rnd() {
        rng ^= rng << 13, rng ^= rng >> 17, rng ^= rng << 5;
        return rng;
    }
    struct Placed {
        uint64_t address = 0, fetchAddress = 0;
        std::span<const uint32_t> code, fetch;
        const Program* p = nullptr;
    };
    // A program in guest memory at a page of its own, followed by words that stand for its constant tables.
    Placed place(const Program& p, uint32_t literalFrom = 0, uint32_t literalTo = 0) {
        Placed out;
        out.p = &p;
        out.address = codeNext;
        uint32_t* w = g.words(codeNext);
        std::copy(p.code.begin(), p.code.end(), w);
        if (literalFrom)
            for (size_t i = 0; i < p.code.size(); ++i)
                if (w[i] == literalFrom) w[i] = literalTo;
        const size_t words = std::max<size_t>(kProgramWords, p.code.size() + 64);
        for (size_t i = p.code.size(); i < words; ++i) w[i] = rnd();
        out.code = {w, words};
        codeNext += 0x2000;
        if (!p.fetch.empty()) {
            out.fetchAddress = codeNext;
            uint32_t* f = g.words(codeNext);
            std::copy(p.fetch.begin(), p.fetch.end(), f);
            out.fetch = {f, p.fetch.size()};
            codeNext += 0x1000;
        }
        return out;
    }
    uint64_t staticBlock(uint64_t bytes, uint64_t align = 16) {
        staticNext = (staticNext + align - 1) / align * align;
        const uint64_t a = staticNext;
        staticNext += bytes;
        return a;
    }

    std::array<Placed, 2> vs;                 // the two vertex programs of the materials
    Placed vsPost, psPost;                    // full-screen passes
    Placed cs;                                // a compute program that stores bytes through the V# in its user data
    std::array<Placed, kPixelPrograms> ps;
    struct Mesh {
        uint64_t table = 0, indices = 0;      // V# table of its three vertex streams; 16-bit indices
        uint32_t count = 0;
    };
    std::array<Mesh, kMeshes> meshes;
    struct Material {
        unsigned vs = 0, ps = 0;
        uint64_t textures = 0, psConstants = 0;  // T#/S# table; V# table of the pixel stage
        std::array<uint32_t, 4> vsConstants{};   // V# of its vertex constants (not 16-byte aligned)
        bool cullBack = true;
    };
    std::array<Material, kMaterials> materials;
    uint64_t viewTables[8] = {};  // V# table of the vertex stage's shared constants, one per view
    uint64_t postTables[4] = {};  // T#/S# table of a full-screen pass reading colour target i
    uint64_t postVertices = 0;
    uint64_t lights = 0;

    static constexpr uint64_t kEyeColor = kTargets, kEyeDepth = kTargets + 0x04000000, kShadow = kTargets + 0x08000000,
                              kPost = kTargets + 0x0C000000, kTargetStride = 0x01000000;
    static constexpr uint32_t kEyeW = 1440, kEyeH = 1536, kShadowSize = 1024, kPostSize = 256;

    explicit Scene(Guest& guest) : g(guest) {
        vs[0] = place(vsX16), vs[1] = place(vsDword);
        vsPost = place(vsFetch), psPost = place(psTextured), cs = place(csBytes);
        for (unsigned i = 0; i < kPixelPrograms; ++i) {
            const float v = 0.25f + 0.01f * float(i);
            uint32_t bits;
            std::memcpy(&bits, &v, 4);
            ps[i] = place(psMaterial, 0x3f123456, bits);
        }
        // Textures: linear 64x64 RGBA8, a page apart.
        for (unsigned i = 0; i < kTextures; ++i)
            for (unsigned k = 0; k < kTexSize * kTexSize; ++k) g.words(kTex + 0x8000ull * i)[k] = rnd();
        // Meshes: three streams of 256 vertices and 300 indices.
        for (unsigned i = 0; i < kMeshes; ++i) {
            const uint64_t base = kMesh + 0x10000ull * i;
            Mesh& m = meshes[i];
            m.table = staticBlock(48);
            for (unsigned s = 0; s < 3; ++s) vsharp(g.words(m.table) + 4 * s, base + 0x2000ull * s, 16, 256);
            m.indices = base + 0x8000;
            m.count = 150 + 6 * (i % 26);
            for (unsigned k = 0; k < m.count; ++k) g.at<uint16_t>(m.indices)[k] = uint16_t(rnd() % 256);
        }
        lights = staticBlock(8192);
        for (unsigned i = 0; i < kMaterials; ++i) {
            Material& m = materials[i];
            m.vs = i % 3 == 2 ? 1 : 0;
            m.ps = i % kPixelPrograms;
            m.cullBack = i % 5 != 0;
            m.textures = staticBlock(32 * 4);
            uint32_t* t = g.words(m.textures);
            for (unsigned k = 0; k < 3; ++k) tsharp(t + 8 * k, kTex + 0x8000ull * ((i * 7 + k * 31) % kTextures), kTexSize, 8);
            t[24] = 0, t[25] = t[26] = t[27] = 0;            // S#: wrap, point
            t[28] = 2 | (2u << 3), t[29] = t[30] = t[31] = 0;  // S#: clamp
            const uint64_t own = staticBlock(256);
            m.psConstants = staticBlock(32);
            vsharp(g.words(m.psConstants), own, 16, 16);
            vsharp(g.words(m.psConstants) + 4, lights, 16, 512);
            const uint64_t vsOwn = staticBlock(128 + 16) + 8;  // (not 16-byte aligned: copied when bound)
            vsharp(m.vsConstants.data(), vsOwn, 16, 8);
            for (unsigned k = 0; k < 32; ++k) g.words(vsOwn)[k] = rnd();
        }
        for (uint64_t& table : viewTables) {
            table = staticBlock(96);
            const uint32_t sizes[6] = {1024, 512, 512, 1024, 256, 256};
            for (unsigned k = 0; k < 6; ++k) vsharp(g.words(table) + 4 * k, staticBlock(sizes[k]), 16, sizes[k] / 16);
        }
        postVertices = staticBlock(16 * 64);
        for (unsigned i = 0; i < 4; ++i) {
            postTables[i] = staticBlock(48);
            uint32_t* t = g.words(postTables[i]);
            tsharp(t, kPost + kTargetStride * i, kPostSize, 14);
            t[8] = 0, t[9] = t[10] = t[11] = 0;
        }
    }

    // One draw of `material` with `mesh` into the current targets; its own constants at `arena` (64 bytes of tables aside).
    void stage(vpps4::Stage& st, const Placed& p) const {
        st.code = p.code;
        st.address = p.address;
        st.userSgprs = p.p->userSgprs;
        st.vgprCompCnt = p.p->vgprCompCnt;
    }
    vpps4::DrawState materialDraw(const vpps4::DrawState& pass, unsigned material, unsigned mesh, unsigned view, uint64_t arena,
                                  bool pixel) {
        const Material& m = materials[material];
        const Mesh& me = meshes[mesh];
        vpps4::DrawState d = pass;
        d.raster.cullBack = m.cullBack;
        d.indexAddress = me.indices;
        d.count = me.count;
        const Placed& v = vs[m.vs];
        stage(d.vs, v);
        d.vs.fetch = v.fetch;
        d.vs.userData[0] = uint32_t(v.fetchAddress), d.vs.userData[1] = uint32_t(v.fetchAddress >> 32);
        d.vs.userData[2] = uint32_t(me.table), d.vs.userData[3] = uint32_t(me.table >> 32);
        vsharp(&d.vs.userData[4], arena + 8, 16, 16);  // the object's constants: 256 bytes, not 16-byte aligned
        std::copy(m.vsConstants.begin(), m.vsConstants.end(), &d.vs.userData[8]);
        d.vs.userData[12] = uint32_t(viewTables[view]), d.vs.userData[13] = uint32_t(viewTables[view] >> 32);
        if (pixel) {
            stage(d.ps, ps[m.ps]);
            d.ps.userData[0] = uint32_t(m.textures), d.ps.userData[1] = uint32_t(m.textures >> 32);
            d.ps.userData[2] = uint32_t(m.psConstants), d.ps.userData[3] = uint32_t(m.psConstants >> 32);
            vsharp(&d.ps.userData[4], arena + 0x120, 16, 8);  // the object's pixel constants: 128 bytes, aligned
            d.ps.psInputAddr = ps[m.ps].p->psInputAddr;
            d.ps.exportFormat[0] = 4;  // FP16_ABGR
            d.ps.psNumInterp = 1;
        }
        return d;
    }
    // A full-screen draw into post target `target` that samples the texture of the T#/S# table at `table`.
    vpps4::DrawState postDraw(unsigned target, uint64_t table) const {
        vpps4::DrawState d = passState(kPost + kTargetStride * target, 0, kPostSize, kPostSize);
        stage(d.vs, vsPost);
        d.vs.fetch = vsPost.fetch;
        d.vs.userSgprs = 6;
        d.vs.vgprCompCnt = 3;
        const uint64_t vtable = meshes[0].table;
        d.vs.userData[4] = uint32_t(vtable), d.vs.userData[5] = uint32_t(vtable >> 32);
        stage(d.ps, psPost);
        d.ps.userSgprs = 2;
        d.ps.userData[0] = uint32_t(table), d.ps.userData[1] = uint32_t(table >> 32);
        d.ps.psInputAddr = 2;
        d.ps.exportFormat[0] = 4;
        d.count = 3;
        return d;
    }
    static vpps4::DrawState passState(uint64_t color, uint64_t depth, uint32_t w, uint32_t h) {
        vpps4::DrawState d;
        if (color) d.color[0] = {color, w, h, w, 10, 0, 0, 14};
        if (depth) d.depth = {depth, 0, w, h, 3, 0, 13, 0, 1, 1, false, false, 1.0f, 0};
        d.viewport = {float(w) / 2, float(w) / 2, -float(h) / 2, float(h) / 2, 1, 0};
        d.scissorRegs.present = true;
        d.scissorRegs.screen = d.scissorRegs.window = d.scissorRegs.generic = {0, 0, int32_t(w), int32_t(h)};
        d.primitive = vpps4::PrimitiveType::TriangleList;
        d.instances = 1;
        return d;
    }
    static vpps4::DrawState depthClear(vpps4::DrawState pass) {
        pass.primitive = vpps4::PrimitiveType::RectList;
        pass.depth.depthClear = true;
        pass.count = 3;
        return pass;
    }

    // The script of picture `n`: `draws` draws (about; 2000 is the measured load), the title's writes of its constants
    // included (Call steps), three submits.
    Script picture(unsigned n, unsigned draws = 2000) {
        Script s;
        const uint64_t arena = kArena + (n % 2) * kArenaBytes;
        uint64_t next = arena;
        rng = 0x9E3779B9u * (n + 1) | 1;
        // The title writes the frame's constants before it sends the commands: the views' and the lights'.
        for (uint64_t table : viewTables) {
            const uint64_t view = uint64_t(g.words(table)[0]) | uint64_t(g.words(table)[1] & 0xfff) << 32;
            for (unsigned k = 0; k < 16; ++k) g.words(view)[k] = rnd();
        }
        for (unsigned k = 0; k < 64; ++k) g.words(lights)[k] = rnd();
        const auto run = [&](const vpps4::DrawState& pass, unsigned count, unsigned view, bool pixel) {
            unsigned material = rnd() % kMaterials, mesh = rnd() % kMeshes, left = 0;
            for (unsigned i = 0; i < count; ++i) {
                if (!left) material = rnd() % kMaterials, left = 1 + rnd() % 8;  // runs of one material
                --left;
                if (rnd() % 2) mesh = rnd() % kMeshes;
                for (unsigned k = 0; k < 0x1a0 / 4; ++k) g.words(next)[k] = rnd();  // (the object's constants, new every picture)
                Step st;
                st.draw = materialDraw(pass, material, mesh, view, next, pixel);
                s.push_back(std::move(st));
                next += 0x1c0;
            }
        };
        const auto submit = [&] {
            const std::vector<std::pair<uint64_t, uint64_t>> asked = {
                {arena, next - arena}, {arena + 0x1c0, 0x1c0}, {next - 0x1c0, 0x1c0}, {next - 0x1c0 + 8, 64}, {lights, 8192}, {viewTables[4], 96},
                {kEyeColor, 64}, {kPost + kTargetStride, 1024}, {kShadow, 64}, {meshes[5].table, 48}, {kMesh + 0x10000ull * 5, 4096}};
            s.push_back(probeStep(asked));
            Step st;
            st.kind = Step::Submit;
            s.push_back(std::move(st));
            s.push_back(probeStep(asked));
        };
        const auto draw = [&](const vpps4::DrawState& d) {
            Step st;
            st.draw = d;
            s.push_back(std::move(st));
        };
        const unsigned unit = draws / 100;  // (the shares of the measured picture, per cent)
        // Shadow cascades: depth only.
        for (unsigned c = 0; c < 4; ++c) {
            vpps4::DrawState pass = passState(0, kShadow + kTargetStride * c, kShadowSize, kShadowSize);
            pass.depthStencil.depthTest = pass.depthStencil.depthWrite = true;
            pass.depthStencil.depthFunc = 3;
            draw(depthClear(pass));
            run(pass, 3 * unit, c, false);
        }
        submit();
        // Each eye: a depth prepass, the opaque pass, the transparent one.
        for (unsigned eye = 0; eye < 2; ++eye) {
            vpps4::DrawState pre = passState(0, kEyeDepth + kTargetStride * eye, kEyeW, kEyeH);
            pre.depthStencil.depthTest = pre.depthStencil.depthWrite = true;
            pre.depthStencil.depthFunc = 1;
            draw(depthClear(pre));
            run(pre, 14 * unit, 4 + eye, false);
            vpps4::DrawState opaque = passState(kEyeColor + kTargetStride * eye, kEyeDepth + kTargetStride * eye, kEyeW, kEyeH);
            opaque.depthStencil.depthTest = true;
            opaque.depthStencil.depthFunc = 2;
            run(opaque, 17 * unit, 4 + eye, true);
            vpps4::DrawState blend = opaque;
            blend.depthStencil.depthFunc = 3;
            blend.blend[0].enable = true;
            blend.blend[0].colorSrc = 4, blend.blend[0].colorDst = 5;
            run(blend, 7 * unit, 4 + eye, true);
            if (eye == 0) submit();
        }
        // Full-screen passes: each reads the target the one before wrote.
        for (unsigned p = 0; p < 24; ++p)
            for (unsigned k = 0; k < 1 + p % 3; ++k) draw(postDraw((p + 1) % 4, postTables[p % 4]));
        // The interface: blended, no depth.
        vpps4::DrawState ui = passState(kEyeColor, 0, kEyeW, kEyeH);
        ui.blend[0].enable = true;
        ui.blend[0].colorSrc = 4, ui.blend[0].colorDst = 5;
        run(ui, 8 * unit, 6, true);
        submit();
        return s;
    }

    // Warm-up for a run that measures: the constant arenas and the static tables become memory the renderer has read on
    // the CPU (one unaligned constant buffer of 64 KiB per draw over all of them), as they are in a title that has been
    // running for a while: small read-only buffers there are then copied when their draw is recorded.
    Script primeArenas() {
        Script s;
        vpps4::DrawState pass = passState(kEyeColor, 0, kEyeW, kEyeH);
        const auto prime = [&](uint64_t from, uint64_t to) {
            for (uint64_t a = from; a < to; a += 0x10000 - 16) {
                Step st;
                st.draw = materialDraw(pass, 0, 0, 0, a, true);
                vsharp(&st.draw.vs.userData[4], a + 8, 16, 0x10000 / 16 - 1);
                s.push_back(std::move(st));
            }
        };
        prime(kArena, kArena + 2 * kArenaBytes);
        prime(kStatic, staticNext);
        Step st;
        st.kind = Step::Submit;
        s.push_back(std::move(st));
        return s;
    }
};

vpps4::Hooks hooksOf(Guest& g, std::vector<std::string>* log) {
    vpps4::Hooks h;
    h.region = [&g](uint64_t a, uint64_t s, vpps4::GuestRegion& out) { return g.region(a, s, out); };
    h.regionGeneration = [&g] {
        ++g.generationCalls;
        return g.generation;
    };
    h.shader = [](uint64_t, vpir::Stage, const vpmsl::MslShader&) {};
    h.log = [log](const std::string& s) {
        if (log) log->push_back(s);
    };
    h.watchWrites = [&g](uint64_t, uint64_t) { ++g.watchWrites; };
    h.watchOwned = [&g](uint64_t, uint64_t, bool) { ++g.watchOwned; };
    return h;
}

// ---- Capture ---------------------------------------------------------------------------------------------------------
// Everything the renderer hands to the backend and to its host, written down in order: resources made and destroyed,
// pipelines, every command of every frame with all its fields, the bytes of the renderer's own buffers that the commands
// bind (upload rings: constants, tables, indices), the compiled frame, the shaders and the watches. Two runs that leave
// the same bytes here drew the same thing.
class Capture final : public vpgpu::Backend {
public:
    std::string bytes;
    std::vector<std::pair<size_t, const char*>> marks;  // (where each record starts: names the first difference)
    const uint8_t* guestHost = nullptr;
    uint64_t inFlight = 2;  // frames the "GPU" is behind

    void mark(const char* what) { marks.push_back({bytes.size(), what}); }
    void raw(const void* p, size_t n) { bytes.append(static_cast<const char*>(p), n); }
    template <class T>
    void v(const T& x) {
        static_assert(std::is_arithmetic_v<T> || std::is_enum_v<T>);
        raw(&x, sizeof x);
    }
    void str(const std::string& x) { v(uint64_t(x.size())), raw(x.data(), x.size()); }
    template <class Tag>
    void id(vpgpu::Handle<Tag> h) { v(h.index), v(h.generation); }

    void createBuffer(vpgpu::BufferId b, const vpgpu::BufferDesc& d) override {
        mark("createBuffer");
        id(b), v(d.size), v(d.residentByUse), str(d.label.rfind("guest ", 0) == 0 ? d.label : std::string("own"));
        const bool guest = d.label.rfind("guest ", 0) == 0;
        v(guest ? uint64_t(static_cast<const uint8_t*>(d.guestMemory) - guestHost) : uint64_t(d.guestMemory != nullptr));
        buffers[b.index] = {static_cast<const uint8_t*>(d.guestMemory), d.size, !guest};
    }
    void createTexture(vpgpu::TextureId t, const vpgpu::TextureDesc& d) override {
        mark("createTexture");
        id(t), v(d.type), v(d.format), v(d.width), v(d.height), v(d.depth), v(d.mips), v(d.layers), v(d.samples), v(d.usage);
        v(d.persistent), v(d.transient), v(d.samplesReadable), str(d.label);
    }
    void createSampler(vpgpu::SamplerId sm, const vpgpu::SamplerDesc& d) override {
        mark("createSampler");
        id(sm), v(d.minFilter), v(d.magFilter), v(d.mipFilter), v(d.u), v(d.v), v(d.w), v(d.maxAnisotropy);
        v(d.compare.has_value()), v(d.compare.value_or(vpgpu::CompareOp::Never)), v(d.border), v(d.lodMin), v(d.lodMax), v(d.lodBias);
        v(d.shaderLodBias), v(d.normalized);
    }
    void destroyBuffer(vpgpu::BufferId b) override { mark("destroyBuffer"), id(b); }
    void destroyTexture(vpgpu::TextureId t) override { mark("destroyTexture"), id(t); }
    void destroySampler(vpgpu::SamplerId sm) override { mark("destroySampler"), id(sm); }
    bool compilePipeline(const vpgpu::PipelineState& ps, std::string&) override {
        mark("pipeline");
        vpgpu::PipelineState unlabelled = ps;
        unlabelled.label.clear();
        v(ps.key().value), str(vpgpu::describePipelineState(unlabelled)), str(ps.label);
        return true;
    }
    bool takeTimings(uint64_t&, vpgpu::GpuTimings&) override { return false; }
    uint64_t completedFrames() const override { return completed; }
    bool waitFramesBelow(uint64_t frames, uint64_t) override {
        completed = std::max(completed, frames);
        return true;
    }
    bool waitAll(uint64_t) override {
        completed = submitted;
        return true;
    }

    void range(const vpgpu::ResourceRange& r) {
        v(r.kind), v(r.index), v(r.generation), v(r.begin), v(r.end), v(r.mipBegin), v(r.mipEnd), v(r.layerBegin), v(r.layerEnd);
        v(r.rect), v(r.x0), v(r.y0), v(r.x1), v(r.y1);
    }
    // The bytes of one of the renderer's own buffers (upload rings and the like) a command reads: what the CPU put there.
    // Not the ranges an earlier command of the frame fills on the GPU (a kernel's output, a copy's destination): the host
    // memory behind those holds nothing defined when the frame is handed over.
    void contents(uint32_t index, uint64_t begin, uint64_t end) {
        const auto it = buffers.find(index);
        if (it == buffers.end() || !it->second.own || !it->second.memory) return;
        end = std::min(end, it->second.size);
        for (const auto& [i, b, e] : gpuFilled)
            if (i == index && b < end && begin < e) return;
        if (begin < end) raw(it->second.memory + begin, end - begin);
    }
    void filled(uint32_t index, uint64_t begin, uint64_t end) { gpuFilled.push_back({index, begin, end}); }
    void accesses(const std::vector<vpgpu::Access>& list) {
        v(uint64_t(list.size()));
        for (const vpgpu::Access& a : list) {
            range(a.range), v(a.stages), v(a.write), v(a.pixelLocal), v(a.kind), v(a.slot);
            if (a.range.kind != vpgpu::ResourceRange::Kind::Buffer) continue;
            if (a.write) filled(a.range.index, a.range.begin, a.range.end);
            else contents(a.range.index, a.range.begin, a.range.end);
        }
    }
    void views(const std::vector<vpgpu::ViewBinding>& list) {
        v(uint64_t(list.size()));
        for (const vpgpu::ViewBinding& b : list) v(b.kind), v(b.slot), id(b.view);
    }
    void span(const vpgpu::BufferSpan& b) { id(b.buffer), v(b.offset), v(b.size), v(b.bytesPerRow), v(b.rowsPerImage); }
    void span(const vpgpu::TextureSpan& t) {
        id(t.texture), v(t.mip), v(t.layer), v(t.layerCount), v(t.x), v(t.y), v(t.z), v(t.width), v(t.height), v(t.depth), v(t.aspect);
    }
    void attachment(const vpgpu::Attachment& a) {
        id(a.texture), v(a.mip), v(a.layer), v(a.load), v(a.store);
        for (double c : a.clear.color) v(c);
        v(a.clear.depth), v(a.clear.stencil), id(a.resolve), v(a.resolveMip), v(a.resolveLayer), v(a.resolveRequired);
        v(a.depthResolve), v(a.stencilResolve), v(a.format), v(a.keepSamples), v(a.autoMsaa), v(a.needsSeed), id(a.seedSrc);
        v(a.seedMip), v(a.seedLayer);
    }
    void pass(const vpgpu::RenderPassDesc& d) {
        for (const vpgpu::Attachment& a : d.colors) attachment(a);
        attachment(d.depth);
        v(d.layerCount), v(d.width), v(d.height), v(d.samples), v(d.renderWidth), v(d.renderHeight);
        v(d.rateMap ? d.rateMap->key : uint64_t(0)), str(d.label);
    }
    void draw(const vpgpu::Draw& d) {
        mark("draw");
        v(bool(d.clear));
        if (d.clear) {
            v(uint64_t(d.clear->rects.size()));
            for (const auto& rc : d.clear->rects) v(rc.x), v(rc.y), v(rc.width), v(rc.height), v(rc.layer), v(rc.layerCount);
            v(d.clear->colorMask), raw(d.clear->colorBits.data(), sizeof d.clear->colorBits);
            v(d.clear->depth), v(d.clear->stencil), v(d.clear->depthValue), v(d.clear->stencilValue);
        }
        v(d.pipeline.value), accesses(d.bindings), v(d.primitive), v(d.first), v(d.count), v(d.instances), v(d.baseVertex);
        v(d.baseInstance), v(d.instanceViews), v(d.amplified), v(d.drawIndex), v(d.index.has_value());
        if (d.index) {
            id(d.index->buffer), v(d.index->offset), v(d.index->wide);
            const uint64_t stride = d.index->wide ? 4 : 2;
            contents(d.index->buffer.index, d.index->offset + d.first * stride, d.index->offset + (uint64_t(d.first) + d.count) * stride);
        }
        v(d.stencilRef), v(d.stencilRefBack);
        for (int k = 0; k < 2; ++k) v(d.stencilReadMask[size_t(k)]), v(d.stencilWriteMask[size_t(k)]);
        for (float c : d.blendColor) v(c);
        v(d.viewport.x), v(d.viewport.y), v(d.viewport.width), v(d.viewport.height), v(d.viewport.znear), v(d.viewport.zfar);
        v(d.scissor.x), v(d.scissor.y), v(d.scissor.width), v(d.scissor.height), v(d.frontFace), v(d.depthBias), v(d.depthBiasSlope);
        v(d.depthBiasClamp), v(d.depthClamp), v(d.wireframe), views(d.views), v(d.visibility.has_value());
        if (d.visibility) span(*d.visibility), filled(d.visibility->buffer.index, d.visibility->offset, d.visibility->offset + 8);
        v(d.indirect.has_value());
        if (d.indirect) id(d.indirect->buffer), v(d.indirect->offset);
        v(d.expandedBy[0].value), v(d.expandedBy[1].value);
    }
    void submit(uint64_t index, const vpgpu::Frame& f, const vpgpu::CompiledFrame& c) override {
        mark("frame");
        gpuFilled.clear();
        v(index), v(uint64_t(f.commands().size()));
        for (const vpgpu::Command& cmd : f.commands()) {
            if (const auto* rp = std::get_if<vpgpu::RenderPass>(&cmd)) {
                mark("pass");
                pass(rp->desc), v(uint64_t(rp->draws.size()));
                for (const vpgpu::Draw& d : rp->draws) draw(d);
            } else if (const auto* dp = std::get_if<vpgpu::Dispatch>(&cmd)) {
                mark("dispatch");
                v(dp->pipeline.value), accesses(dp->bindings), raw(dp->groups.data(), sizeof dp->groups), str(dp->label), views(dp->views);
                v(dp->indirect.has_value()), v(dp->required);
                if (dp->indirect) id(dp->indirect->buffer), v(dp->indirect->offset);
            } else {
                const vpgpu::Copy& cp = std::get<vpgpu::Copy>(cmd);
                mark("copy");
                v(cp.kind), span(cp.srcBuffer), span(cp.dstBuffer), span(cp.srcTexture), span(cp.dstTexture), v(cp.fillByte);
                v(cp.blitFilter), v(cp.blitFlipX), v(cp.blitFlipY), v(cp.blitFormat), v(cp.blitSwapRB), v(cp.mipChain), str(cp.label);
                if (cp.kind == vpgpu::CopyKind::BufferToBuffer || cp.kind == vpgpu::CopyKind::BufferToTexture)
                    contents(cp.srcBuffer.buffer.index, cp.srcBuffer.offset, cp.srcBuffer.offset + cp.srcBuffer.size);
                if (cp.kind == vpgpu::CopyKind::BufferToBuffer || cp.kind == vpgpu::CopyKind::TextureToBuffer || cp.kind == vpgpu::CopyKind::FillBuffer)
                    filled(cp.dstBuffer.buffer.index, cp.dstBuffer.offset, cp.dstBuffer.size ? cp.dstBuffer.offset + cp.dstBuffer.size : ~uint64_t(0));
            }
        }
        mark("compiled");
        v(uint64_t(c.encoders.size()));
        for (const vpgpu::Encoder& e : c.encoders) {
            v(e.kind), v(e.reason), pass(e.pass), v(uint64_t(e.commands.size()));
            for (uint32_t k : e.commands) v(k);
            v(e.passes), raw(e.tile.data(), sizeof e.tile), raw(e.memBacked.data(), sizeof e.memBacked), v(e.heavy), v(e.drawBegin);
            v(e.drawEnd), v(e.view), str(e.splitWhy ? e.splitWhy : ""), v(e.splitSig), v(e.splitViewVerts), v(e.splitHeavyAfter);
        }
        v(uint64_t(c.barriers.size()));
        for (const vpgpu::Barrier& b : c.barriers) v(b.encoder), v(b.item), v(b.after), v(b.before), v(b.hazards);
        v(uint64_t(c.badDraws.size()));
        for (const vpgpu::DrawRef& d : c.badDraws) v(d.command), v(d.draw);
        v(uint64_t(c.memAfter.size()));
        for (const auto& [t, valid] : c.memAfter) id(t), v(valid);
        v(uint64_t(c.autoAfter.size()));
        for (const auto& a : c.autoAfter) id(a.texture), id(a.src), v(a.mip), v(a.layer), v(a.valid);
        v(uint64_t(c.transientViolations.size())), v(uint64_t(c.msaaRejects.size())), v(uint64_t(c.storeObs.size()));
        const vpgpu::CompileStats& st = c.stats;
        v(st.commands), v(st.passesMerged), v(st.copiesHoisted), v(st.copiesSunk), v(st.dropped), raw(st.cuts, sizeof st.cuts);
        raw(st.hazards, sizeof st.hazards), v(st.intraBarriers), v(st.barriersComputeDraw), v(st.tileLocal), v(st.tileRejected);
        v(st.badFetch), v(st.tileBytes), v(st.storesKept), v(st.msaaStaleReads), v(st.heavyEncoders), v(st.loadsDropped);
        v(st.maxEncoderVertices), v(st.seedDraws), v(st.seedMissing);
        submitted = index + 1;
        if (submitted > inFlight) completed = std::max(completed, submitted - inFlight);
    }

private:
    struct Memory {
        const uint8_t* memory;
        uint64_t size;
        bool own;
    };
    std::map<uint32_t, Memory> buffers;
    std::vector<std::tuple<uint32_t, uint64_t, uint64_t>> gpuFilled;  // (buffer index, begin, end) of the frame being written down
    uint64_t submitted = 0, completed = 0;
};

// What one run of a script left: the capture and the renderer's own counts.
struct Outcome {
    std::string bytes;
    std::vector<std::pair<size_t, const char*>> marks;
    std::string counts;
    std::string observed;  // what probeStep() steps saw, in order
    uint64_t drawn = 0;
};

// What a host asks the renderer about a range of memory (the scheduler, the DMA paths): when the recorded or the queued work may
// still touch it, whether the CPU read it, labels waiting to be written there, what the tracker says of its writers. The caches
// of the per-draw shortcuts stand behind these answers; a script writes them down at the points that matter and a run with the
// shortcuts off must have written the same.
std::string* observed = nullptr;
void observe(vpps4::Renderer& r, uint64_t address, uint64_t size) {
    if (!observed) return;
    const vpps4::MemoryTracker& t = r.tracker();
    char line[256];
    std::snprintf(line, sizeof line, "%llx+%llx: wait=%llx recorded=%d may=%d cpuRead=%d label=%d/%llu ver=%llu gpu=%d owner=%u others=%d\n",
                  (unsigned long long)address, (unsigned long long)size, (unsigned long long)r.waitTarget(address, size, 0, 0),
                  int(r.recordedMayAccess(address, size)), int(r.gpuMayAccess(address, size)), int(r.cpuReadSeen(address, size)),
                  int(r.labelPending(address, uint32_t(size))), (unsigned long long)r.labelWaitTarget(address, uint32_t(size)),
                  (unsigned long long)t.version(address, size), int(t.gpuWritten(address, size)), t.ownerImage(address, size),
                  int(t.writtenByOthers(address, size, vpps4::MemoryTracker::kNone, 0)));
    observed->append(line);
}

struct RunOptions {
    bool watches = true;   // the host watches texture sources and owned ranges (else the renderer looks at the bytes itself)
    bool evict = false;    // residency so tight that idle images are evicted at once
};

// Plays `make(scene)` on a fresh guest and renderer. `off`: the shortcuts switched off (all, or the one named).
Outcome run(const std::function<Script(Scene&)>& make, bool off, const char* only = nullptr, RunOptions options = {}) {
    setOptOuts(off, only);
    Guest guest;
    Scene scene(guest);
    Capture capture;
    capture.guestHost = guest.host;
    std::vector<std::string> log;
    vpgpu::Device device{capture, 0, [&](const std::string& s) { log.push_back(s); }};
    device.setRefuseFailedBuffers(true);
    device.setHoldSteadyMs(12);
    vpps4::Hooks hooks = hooksOf(guest, &log);
    hooks.shader = [&](uint64_t shaderId, vpir::Stage stage, const vpmsl::MslShader& sh) {
        capture.mark("shader"), capture.v(shaderId), capture.v(stage), capture.str(sh.source);
    };
    hooks.watchWrites = [&](uint64_t a, uint64_t n) { capture.mark("watchWrites"), capture.v(a), capture.v(n); };
    hooks.watchOwned = [&](uint64_t a, uint64_t n, bool on) { capture.mark("watchOwned"), capture.v(a), capture.v(n), capture.v(on); };
    if (!options.watches) hooks.watchWrites = nullptr, hooks.watchOwned = nullptr;
    Outcome out;
    observed = &out.observed;
    {
        vpps4::Renderer renderer{device, std::move(hooks)};
        renderer.setMaxSamples(1);
        renderer.setSummaryEvery(0);
        if (options.evict) renderer.setResidency(4, 1);
        const Script script = make(scene);
        for (const Step& st : script) {
            switch (st.kind) {
                case Step::Draw: renderer.draw(st.draw); break;
                case Step::Submit:
                    capture.mark("submit");
                    renderer.submit();
                    break;
                case Step::Call: st.call(renderer); break;
            }
        }
        renderer.submit();
        const vpps4::Stats& st = renderer.stats();
        const vpps4::CpuTimes t = renderer.takeCpuTimes();
        std::ostringstream c;
        c << "draws=" << st.draws << " drawn=" << st.drawn << " dispatches=" << st.dispatches << " dispatched=" << st.dispatched
          << " passes=" << st.passes << " copies=" << st.copies << " resolves=" << st.resolves << " shaders=" << st.shaders
          << " shaderErrors=" << st.shaderErrors << " textures=" << st.textures << " uploads=" << st.uploads << " targets=" << st.targets
          << " detiled=" << st.detiled << " refilled=" << st.refilled << " retiled=" << st.retiled << " evicted=" << st.evicted
          << " viewsMade=" << st.viewsMade << " syncs=" << st.syncs << " ringChains=" << st.ringChains << " calls=" << t.calls
          << " submits=" << t.submits << " emptySubmits=" << t.emptySubmits << " userDataReused=" << t.userDataReused
          << " alignedCopyReused=" << t.alignedCopyReused << " regionHits=" << t.regionHits << " regionMisses=" << t.regionMisses;
        for (const auto& [why, n] : st.skipped) c << " skip[" << why << "]=" << n;
        for (const auto& [why, n] : st.handled) c << " handled[" << why << "]=" << n;
        for (const auto& [why, n] : st.notes) c << " note[" << why << "]=" << n;
        c << " hostRegionCalls=" << guest.regionCalls;
        out.counts = c.str();
        out.drawn = st.drawn;
        if (st.shaderErrors)
            for (const std::string& l : log)
                if (l.find("shader error") != std::string::npos) std::fprintf(stderr, "  %s\n", l.c_str());
    }
    observed = nullptr;
    out.bytes = std::move(capture.bytes);
    out.marks = std::move(capture.marks);
    setOptOuts(false);
    return out;
}

// The two outcomes are the same; if not, where the first difference is.
bool same(const Outcome& a, const Outcome& b, const char* what) {
    if (a.bytes == b.bytes && a.counts == b.counts && a.observed == b.observed) return true;
    if (a.observed != b.observed) {
        size_t at = 0;
        while (at < a.observed.size() && at < b.observed.size() && a.observed[at] == b.observed[at]) ++at;
        const size_t from = a.observed.rfind('\n', at ? at - 1 : 0) + 1;
        std::fprintf(stderr, "%s: the host's questions got other answers (probe %zu):\n  %s\n  %s\n", what,
                     size_t(std::count(a.observed.begin(), a.observed.begin() + std::ptrdiff_t(at), '\n')),
                     a.observed.substr(from, a.observed.find('\n', from) - from).c_str(),
                     b.observed.substr(from, b.observed.find('\n', from) - from).c_str());
    }
    if (a.bytes != b.bytes) {
        size_t at = 0;
        while (at < a.bytes.size() && at < b.bytes.size() && a.bytes[at] == b.bytes[at]) ++at;
        size_t record = 0;
        while (record + 1 < a.marks.size() && a.marks[record + 1].first <= at) ++record;
        size_t frames = 0, draws = 0;
        for (size_t k = 0; k <= record && k < a.marks.size(); ++k) {
            frames += std::strcmp(a.marks[k].second, "frame") == 0;
            draws += std::strcmp(a.marks[k].second, "draw") == 0;
        }
        std::fprintf(stderr, "%s: the captures differ at byte %zu of %zu / %zu: record %zu (%s, +%zu), after %zu frames and %zu draws\n", what,
                     at, a.bytes.size(), b.bytes.size(), record, a.marks.empty() ? "?" : a.marks[record].second,
                     a.marks.empty() ? size_t(0) : at - a.marks[record].first, frames, draws);
    }
    if (a.counts != b.counts) std::fprintf(stderr, "%s: the counts differ:\n  %s\n  %s\n", what, a.counts.c_str(), b.counts.c_str());
    return false;
}

// `digest`: each script is run once (VPPS4_DIGEST_OFF=1: with the shortcuts off) and only a hash of what it left is printed. The
// same binary source built against the renderer before the shortcuts existed prints the same lines.
bool digestMode = false;
uint64_t fnv(const std::string& x) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (unsigned char c : x) h = (h ^ c) * 0x100000001b3ull;
    return h;
}

// The script with every shortcut on against every shortcut off, and against each one off alone.
void equivalent(const char* what, const std::function<Script(Scene&)>& make, uint64_t minDrawn, RunOptions options = {}, bool each = true) {
    if (digestMode) {
        const Outcome o = run(make, std::getenv("VPPS4_DIGEST_OFF") != nullptr, nullptr, options);
        std::printf("%s: commands %016llx (%zu bytes) host answers %016llx drawn %llu\n", what, (unsigned long long)fnv(o.bytes), o.bytes.size(),
                    (unsigned long long)fnv(o.observed), (unsigned long long)o.drawn);
        return;
    }
    const Outcome off = run(make, true, nullptr, options);
    if (std::getenv("VPPS4_SHOW_COUNTS")) {
        std::printf("%s: %s\n", what, off.counts.c_str());
        size_t asked = 0, hits[6] = {};  // (how much the host's questions had to say: a probe that always answers no proves nothing)
        for (size_t at = 0; (at = off.observed.find('\n', at)) != std::string::npos; ++at) {
            const size_t from = off.observed.rfind('\n', at ? at - 1 : 0) + 1;
            const std::string l = off.observed.substr(from, at - from);
            asked += l.find("wait=") != std::string::npos;
            hits[0] += l.find("recorded=1") != std::string::npos, hits[1] += l.find("may=1") != std::string::npos;
            hits[2] += l.find("cpuRead=1") != std::string::npos, hits[3] += l.find("label=1") != std::string::npos;
            hits[4] += l.find("gpu=1") != std::string::npos, hits[5] += l.find("owner=0") == std::string::npos && l.find("owner=") != std::string::npos;
        }
        std::printf("  host asked %zu: recorded %zu, may %zu, cpuRead %zu, label %zu, gpuWritten %zu, owned %zu\n", asked, hits[0], hits[1],
                    hits[2], hits[3], hits[4], hits[5]);
    }
    CHECK(off.drawn >= minDrawn);
    if (off.drawn < minDrawn) std::fprintf(stderr, "%s: drawn %llu: %s\n", what, (unsigned long long)off.drawn, off.counts.c_str());
    CHECK(same(run(make, false, nullptr, options), off, what));
    CHECK(same(run(make, false, nullptr, options), run(make, false, nullptr, options), what));  // (and a run repeats itself)
    if (each)
        for (const OptOut& one : kOptOuts)
            CHECK(same(run(make, true, one.name, options), off, (std::string(what) + " without " + one.name).c_str()));
}

Step call(std::function<void(vpps4::Renderer&)> f) {
    Step st;
    st.kind = Step::Call;
    st.call = std::move(f);
    return st;
}
// A step that writes down what the renderer says of the ranges (address, size).
Step probeStep(std::vector<std::pair<uint64_t, uint64_t>> ranges) {
    Step st = call([ranges = std::move(ranges)](vpps4::Renderer& r) {
        for (const auto& [address, size] : ranges) observe(r, address, size);
        if (observed) observed->append("--\n");
    });
    st.probe = true;
    return st;
}
Step submitStep() {
    Step st;
    st.kind = Step::Submit;
    return st;
}
void append(Script& to, Script from) {
    for (Step& st : from) to.push_back(std::move(st));
}

// ---- Equivalence scripts ---------------------------------------------------------------------------------------------

// The measured picture itself: three of them after the warm-up.
void equalPicture() {
    equivalent("picture", [](Scene& sc) {
        Script s = sc.primeArenas();
        for (unsigned n = 0; n < 3; ++n) append(s, sc.picture(n, 2000));
        return s;
    }, 5000, {}, false);
    // (and with a host that does not watch memory: the renderer compares the bytes of its sources itself)
    equivalent("picture, no watches", [](Scene& sc) {
        Script s = sc.primeArenas();
        for (unsigned n = 0; n < 2; ++n) append(s, sc.picture(n, 600));
        return s;
    }, 1000, {false, false});
}

// ---- Bench -----------------------------------------------------------------------------------------------------------

int bench() {
    const bool off = std::getenv("VPPS4_BENCH_OFF") != nullptr;
    setOptOuts(off);
    const unsigned pictures = std::getenv("VPPS4_BENCH_PICTURES") ? unsigned(std::atoi(std::getenv("VPPS4_BENCH_PICTURES"))) : 60;
    const unsigned drawsPer = std::getenv("VPPS4_BENCH_DRAWS") ? unsigned(std::atoi(std::getenv("VPPS4_BENCH_DRAWS"))) : 2000;
    Guest guest;
    Scene scene(guest);
    // The GPU of the bench: it finishes a frame two submits after it was handed over (frames in flight, as on the device),
    // and the title has label writes waiting for it (two per submit).
    vpgpu::NullBackend backend;
    backend.record = false;
    std::vector<std::string> log;
    vpgpu::Device device{backend, 0, [&](const std::string& s) { log.push_back(s); }};
    device.setRefuseFailedBuffers(true);
    device.setHoldSteadyMs(12);
    vpps4::Renderer renderer{device, hooksOf(guest, &log)};
    renderer.setMaxSamples(1);
    renderer.setSummaryEvery(0);
    const auto submit = [&] {
        for (unsigned k = 0; k < 2; ++k) renderer.afterGpu(kTargets + 0x2F000000 + 16 * k, 8, [] {}, true);
        renderer.submit();
        if (const uint64_t n = device.submittedFrames(); n > 2) backend.finishFramesBelow(n - 2);
    };
    const auto warm = [&](const Script& s) {
        for (const Step& st : s) st.kind == Step::Draw ? renderer.draw(st.draw) : submit();
    };
    warm(scene.primeArenas());
    for (unsigned n = 0; n < 8; ++n) warm(scene.picture(n, drawsPer));  // (shaders, pipelines, textures, ring sizes)
    for (unsigned n = 0; n < 300; ++n) renderer.submit();  // (past the renderer's start-up diagnostics: its first 300 submits)
    renderer.takeCpuTimes();
    const vpps4::Stats before = renderer.stats();
    using Clock = std::chrono::steady_clock;
    // Per picture, so that the figures are medians (the machine is shared: a mean follows its neighbours' load).
    enum { kWall, kDraw, kShader, kBind, kPipeline, kDesc, kTex, kBuf, kCb, kUpload, kSubmitWall, kLife, kCompile, kBackend, kMetrics };
    std::array<std::vector<double>, kMetrics> m;
    uint64_t draws = 0, submits = 0, alignedReused = 0, userReused = 0, regionHits = 0, regionMisses = 0;
    vpps4::DrawState d;
    for (unsigned n = 0; n < pictures; ++n) {
        const Script s = scene.picture(8 + n, drawsPer);
        double drawWallMs = 0, submitWallMs = 0;
        uint64_t pictureDraws = 0;
        CALLGRIND_TOGGLE_COLLECT;
        for (const Step& st : s) {
            if (st.kind == Step::Draw) {
                d = st.draw;  // (as the adapter: one DrawState, filled for each draw)
                const auto t0 = Clock::now();
                renderer.draw(d);
                drawWallMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                ++pictureDraws;
            } else if (st.kind == Step::Submit) {
                const auto t0 = Clock::now();
                submit();
                submitWallMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                ++submits;
            }
        }
        CALLGRIND_TOGGLE_COLLECT;
        const vpps4::CpuTimes t = renderer.takeCpuTimes();
        const double us = 1000.0 / double(pictureDraws);  // ms of the picture -> us per draw
        m[kWall].push_back(drawWallMs * us), m[kDraw].push_back(t.drawMs * us), m[kShader].push_back(t.shaderMs * us);
        m[kBind].push_back(t.bindMs * us), m[kPipeline].push_back(t.pipelineMs * us), m[kDesc].push_back(t.bindDescMs * us);
        m[kTex].push_back(t.bindTexMs * us), m[kBuf].push_back(t.bindBufMs * us), m[kCb].push_back(t.bindCbMs * us);
        m[kUpload].push_back(t.bindUploadMs * us), m[kSubmitWall].push_back(submitWallMs), m[kLife].push_back(t.lifeMs);
        m[kCompile].push_back(t.compileMs), m[kBackend].push_back(t.backendMs);
        draws += pictureDraws, alignedReused += t.alignedCopyReused, userReused += t.userDataReused;
        regionHits += t.regionHits, regionMisses += t.regionMisses;
        if (!std::getenv("VPPS4_BENCH_STATS")) log.clear();
    }
    const vpps4::Stats& after = renderer.stats();
    const double perPicture = 1.0 / double(pictures);
    std::printf("perdraw bench (%s): %u pictures, %.0f draws and %.1f submits per picture, %.1f passes, %.2f uploads/draw\n",
                off ? "shortcuts OFF" : "shortcuts on", pictures, double(draws) * perPicture, double(submits) * perPicture,
                double(after.passes - before.passes) * perPicture, double(after.uploads - before.uploads) / double(draws));
    std::printf("  drawn %llu of %llu, aligned copies reused %.2f/draw, user data reused %.2f/draw, region cache %.1f hits/draw %.3f misses/draw\n",
                (unsigned long long)(after.drawn - before.drawn), (unsigned long long)(after.draws - before.draws),
                double(alignedReused) / double(draws), double(userReused) / double(draws), double(regionHits) / double(draws),
                double(regionMisses) / double(draws));
    for (const double q : {0.5, 0.1}) {  // the median picture, and the best tenth (the least disturbed)
        std::array<double, kMetrics> v{};
        for (int k = 0; k < kMetrics; ++k) {
            std::vector<double> sorted = m[size_t(k)];
            std::sort(sorted.begin(), sorted.end());
            v[size_t(k)] = sorted[size_t(double(sorted.size() - 1) * q)];
        }
        std::printf(" %s\n", q == 0.5 ? "median picture:" : "best tenth:");
        std::printf("  us/draw: wall %.3f | vpps4.draw %.3f = shader lookup %.3f + bind %.3f + pipeline %.3f + rest %.3f\n", v[kWall],
                    v[kDraw], v[kShader], v[kBind], v[kPipeline], v[kDraw] - v[kShader] - v[kBind] - v[kPipeline]);
        std::printf("  us/draw: bind = descriptors %.3f + textures %.3f + buffers %.3f + constants %.3f + uploads %.3f + table/other %.3f\n",
                    v[kDesc], v[kTex], v[kBuf], v[kCb], v[kUpload], v[kBind] - v[kDesc] - v[kTex] - v[kBuf] - v[kCb] - v[kUpload]);
        std::printf("  ms/picture: submit wall %.3f = life.plan %.3f + frame compile %.3f + backend %.3f + rest %.3f\n", v[kSubmitWall],
                    v[kLife], v[kCompile], v[kBackend], v[kSubmitWall] - v[kLife] - v[kCompile] - v[kBackend]);
    }
    CHECK(after.drawn - before.drawn + (after.handled.count("rect list clear") ? after.handled.at("rect list clear") : 0) -
              (before.handled.count("rect list clear") ? before.handled.at("rect list clear") : 0) == draws);
    CHECK(after.shaderErrors == 0);
    if (failures || std::getenv("VPPS4_BENCH_STATS")) {
        for (const auto& [why, n] : after.skipped) std::printf("  skip[%s]=%llu\n", why.c_str(), (unsigned long long)n);
        for (const auto& [why, n] : after.notes) std::printf("  note[%s]=%llu\n", why.c_str(), (unsigned long long)n);
        for (const auto& [why, n] : after.handled) std::printf("  handled[%s]=%llu\n", why.c_str(), (unsigned long long)n);
        size_t shown = 0;
        for (const std::string& l : log)
            if (l.rfind("gpu.", 0) != 0 && l.rfind("VPKENGINE_PROBE", 0) != 0 && shown++ < 40) std::printf("  log %s\n", l.c_str());
    }
    return failures;
}

// What the tests below share: one opaque pass of an eye and draws of two materials in it.
struct Adverse {
    Scene& sc;
    Script s;
    vpps4::DrawState pass = Scene::passState(Scene::kEyeColor, Scene::kEyeDepth, Scene::kEyeW, Scene::kEyeH);
    explicit Adverse(Scene& scene, bool prime = true) : sc(scene) {
        if (prime) s = sc.primeArenas();
        pass.depthStencil.depthTest = true;
        pass.depthStencil.depthFunc = 3;
    }
    uint64_t arena(unsigned object) const { return kArena + 0x1c0ull * object; }
    vpps4::DrawState of(unsigned material, unsigned mesh, unsigned object, bool pixel = true) {
        return sc.materialDraw(pass, material, mesh, 4, arena(object), pixel);
    }
    void draw(const vpps4::DrawState& d, unsigned times = 1) {
        for (unsigned k = 0; k < times; ++k) {
            Step st;
            st.draw = d;
            s.push_back(st);
        }
    }
    // The ranges the scripts touch: the host is asked about them before and after every call and frame cut.
    std::vector<std::pair<uint64_t, uint64_t>> watched() const {
        std::vector<std::pair<uint64_t, uint64_t>> r;
        for (unsigned k = 0; k < 4; ++k) r.push_back({arena(k), 0x1c0});
        r.push_back({arena(0) + 8, 64}), r.push_back({arena(0) + 0x120, 32}), r.push_back({arena(1) + 0x120, 128}), r.push_back({arena(2) + 8, 256});
        for (unsigned m : {3u, 4u, 33u}) {
            const Scene::Material& mat = sc.materials[m];
            r.push_back({mat.textures, 128}), r.push_back({mat.psConstants, 32});
            r.push_back({vsharpAddress(mat.psConstants), 256}), r.push_back({uint64_t(mat.vsConstants[0]) | uint64_t(mat.vsConstants[1] & 0xfff) << 32, 128});
        }
        for (unsigned v : {4u, 5u}) {
            r.push_back({sc.viewTables[v], 96});
            for (unsigned k = 0; k < 6; ++k) r.push_back({vsharpAddress(sc.viewTables[v] + 16 * k), 256});
        }
        r.push_back({sc.lights, 8192}), r.push_back({sc.lights, 64});
        for (unsigned m : {5u, 6u, 9u}) {
            r.push_back({sc.meshes[m].table, 48}), r.push_back({sc.meshes[m].indices, 600});
            for (unsigned k = 0; k < 3; ++k) r.push_back({kMesh + 0x10000ull * m + 0x2000ull * k, 4096});
        }
        const uint64_t tex = uint64_t(sc.g.words(sc.materials[3].textures)[0]) << 8;
        r.push_back({tex, 16384}), r.push_back({tex + 0x100, 128}), r.push_back({kTex + 0x8000ull * 200, 16384});
        r.push_back({Scene::kEyeColor, 64}), r.push_back({Scene::kPost + Scene::kTargetStride, 1024}), r.push_back({Scene::kEyeDepth, 64});
        r.push_back({sc.ps[3].address, 1536}), r.push_back({sc.vs[0].address, 1536}), r.push_back({sc.vs[0].fetchAddress, 64});
        return r;
    }
    void probe() {
        if (s.empty() || !s.back().probe) s.push_back(probeStep(watched()));
    }
    void then(std::function<void(vpps4::Renderer&)> f) {
        probe();
        s.push_back(call(std::move(f)));
        probe();
    }
    void submit() {
        probe();
        s.push_back(submitStep());
        probe();
    }
    Script done() {
        probe();
        return s;
    }
    // The title writes `words` words at `address` (and nobody is told).
    void write(uint64_t address, unsigned words, uint32_t seed) {
        Guest* g = &sc.g;
        then([g, address, words, seed](vpps4::Renderer&) {
            for (unsigned k = 0; k < words; ++k) g->words(address)[k] = seed * 2654435761u + k * 40503u;
        });
    }
    uint64_t vsharpAddress(uint64_t descriptor) const {
        const uint32_t* w = sc.g.words(descriptor);
        return uint64_t(w[0]) | uint64_t(w[1] & 0xfff) << 32;
    }
};

// Buffers: contents that change under the same pointer, descriptors rewritten in their tables, a frame cut, a change of
// the host's map, writes by the GPU and by the command processor, a label waiting.
void equalBuffers() {
    equivalent("buffers", [](Scene& sc) {
        Adverse t(sc);
        const vpps4::DrawState a = t.of(3, 5, 0), b = t.of(4, 5, 1), c = t.of(33, 9, 2);  // (33 shares its pixel program with 3)
        const Scene::Material& m = sc.materials[3];
        t.draw(a, 2), t.draw(b), t.draw(a), t.draw(c), t.draw(a);
        // The same pointers, other bytes: the object's constants (unaligned, then aligned), the material's, the view's, the lights.
        t.write(t.arena(0) + 8, 64, 1), t.draw(a, 2);
        t.write(t.arena(0) + 0x120, 32, 2), t.draw(a, 2), t.draw(b), t.draw(a);
        t.write(t.vsharpAddress(sc.materials[3].psConstants), 64, 3), t.draw(a), t.draw(c), t.draw(a);
        t.write(uint64_t(m.vsConstants[0]) | uint64_t(m.vsConstants[1] & 0xfff) << 32, 32, 4), t.draw(a, 2);
        t.write(t.vsharpAddress(sc.viewTables[4]), 256, 5), t.draw(a), t.draw(b), t.draw(a);
        t.write(sc.lights, 2048, 6), t.draw(a, 2);
        // Descriptors rewritten in place: another vertex buffer, another size, other constants, the pixel stage's table.
        Guest* g = &sc.g;
        const uint64_t meshTable = sc.meshes[5].table, viewTable = sc.viewTables[4], psTable = m.psConstants;
        const uint64_t otherVertices = kMesh + 0x10000ull * 7, otherConstants = t.vsharpAddress(sc.viewTables[5]);
        t.then([=](vpps4::Renderer&) { vsharp(g->words(meshTable), otherVertices, 16, 128); }), t.draw(a, 2), t.draw(b);
        t.then([=](vpps4::Renderer&) { vsharp(g->words(viewTable), otherConstants, 16, 32); }), t.draw(a, 2);
        t.then([=](vpps4::Renderer&) { vsharp(g->words(viewTable) + 4, otherConstants + 8, 16, 16); }), t.draw(a, 2);  // (now unaligned)
        t.then([=](vpps4::Renderer&) { vsharp(g->words(psTable), otherConstants + 0x200, 16, 8); }), t.draw(a, 2), t.draw(c), t.draw(a);
        t.then([=](vpps4::Renderer&) { vsharp(g->words(psTable) + 4, 0, 16, 8); }), t.draw(a, 2);  // (a null buffer)
        t.then([=](vpps4::Renderer&) { vsharp(g->words(psTable) + 4, otherConstants, 16, 600); }), t.draw(a, 2);
        // The same material after a frame cut, with and without new bytes.
        t.submit(), t.draw(a, 2), t.draw(b);
        t.write(t.arena(0) + 0x120, 32, 7), t.submit(), t.draw(a, 2);
        t.then([](vpps4::Renderer& r) { r.gpuIdle(); }), t.draw(a, 2);
        // The host's map changes; its buffers are dropped.
        t.then([g](vpps4::Renderer&) { ++g->generation; }), t.draw(a, 2), t.draw(b);
        t.then([](vpps4::Renderer& r) { r.dropGuestBuffers(); }), t.draw(a, 2);
        // The GPU writes the buffers: a host-side write, a recorded fill and a copy, a dispatch that stores into one.
        t.then([=](vpps4::Renderer& r) { r.noteGpuWrite(t.arena(0) + 0x120, 16); }), t.draw(a, 2), t.draw(b), t.draw(a);
        t.then([=](vpps4::Renderer& r) { r.invalidate(t.arena(0) + 0x120, 16); }), t.draw(a, 2);
        t.then([=](vpps4::Renderer& r) { r.fill(t.arena(1) + 0x120, 128, 0x01010101); }), t.draw(b, 2), t.draw(a);
        t.then([=](vpps4::Renderer& r) { r.copy(t.arena(1) + 8, t.arena(2) + 8, 256); }), t.draw(b, 2), t.draw(c, 2);
        const uint64_t viewConstants = t.vsharpAddress(sc.viewTables[4] + 16);
        const Scene::Placed cs = sc.cs;
        t.then([=](vpps4::Renderer& r) {
            vpps4::DispatchState d;
            d.cs.code = cs.code;
            d.cs.address = cs.address;
            d.cs.userSgprs = 4;
            d.cs.ldsBytes = 64;
            d.cs.workgroupSize = {64, 1, 1};
            d.cs.userData[0] = uint32_t(viewConstants), d.cs.userData[1] = uint32_t(viewConstants >> 32);
            d.cs.userData[2] = 0x100;
            r.dispatch(d);
        });
        t.draw(a, 2), t.draw(b), t.submit(), t.draw(a, 2);
        // The command processor writes in order with the draws (a range it had to wait for), and a label waits for the GPU.
        t.then([=](vpps4::Renderer& r) { r.cpuAccessed(meshTable, 48); }), t.draw(a, 2), t.draw(b);
        t.then([=](vpps4::Renderer& r) { r.cpuAccessed(t.arena(2), 0x1c0); }), t.draw(c, 2);
        t.then([=](vpps4::Renderer& r) { r.afterGpu(t.arena(0) + 0x120, 8, [] {}, true); }), t.draw(a, 2), t.draw(b);
        t.submit(), t.draw(a, 2), t.submit(), t.submit(), t.submit(), t.draw(a, 2), t.draw(b, 2);
        // Geometry that changes under the same state: indices, their count, an index buffer elsewhere.
        vpps4::DrawState e = a;
        e.count = 30, t.draw(e), t.draw(a);
        e.indexAddress += 64, t.draw(e, 2);
        t.write(sc.meshes[5].indices, 16, 8), t.draw(a, 2);
        return t.done();
    }, 90);
}

// Textures: rewritten with and without a word to the renderer, another T# at the same address, a render target read as a
// texture and drawn into again, a texture's memory that becomes a render target.
void equalTextures() {
    for (const bool watches : {true, false}) {
        equivalent(watches ? "textures" : "textures, no watches", [](Scene& sc) {
            Adverse t(sc);
            const vpps4::DrawState a = t.of(3, 5, 0), b = t.of(4, 6, 1), c = t.of(33, 5, 2);
            Guest* g = &sc.g;
            const uint64_t table = sc.materials[3].textures;
            const uint64_t tex = uint64_t(g->words(table)[0]) << 8;
            const uint64_t texBytes = 4ull * Scene::kTexSize * Scene::kTexSize;
            t.draw(a, 2), t.draw(b), t.draw(a), t.draw(c), t.draw(a);
            t.write(tex, 4096, 1), t.then([=](vpps4::Renderer& r) { r.invalidate(tex, texBytes); }), t.draw(a, 2), t.draw(b), t.draw(a);
            t.write(tex, 4096, 2), t.draw(a, 2), t.submit(), t.draw(a, 2), t.draw(b);
            t.write(tex + 256, 16, 3), t.then([=](vpps4::Renderer& r) { r.coherencePoint(tex, texBytes); }), t.draw(a, 2), t.submit(), t.draw(a, 2);
            t.write(tex + 8192, 16, 4), t.then([=](vpps4::Renderer& r) { r.invalidate(tex + 8192, 4); }), t.draw(a), t.draw(c), t.draw(a);
            // The T# rewritten in its table: another texture, another format of the same address, another sampler.
            t.then([=](vpps4::Renderer&) { tsharp(g->words(table), kTex + 0x8000ull * 200, Scene::kTexSize, 8); }), t.draw(a, 2), t.draw(c);
            t.then([=](vpps4::Renderer&) { tsharp(g->words(table), tex, Scene::kTexSize, 8), g->words(table)[1] |= 9u << 26; }), t.draw(a, 2);  // (sRGB)
            t.then([=](vpps4::Renderer&) { tsharp(g->words(table), tex, Scene::kTexSize, 8); }), t.draw(a, 2);
            t.then([=](vpps4::Renderer&) { g->words(table)[24] = 2 | (2u << 3); }), t.draw(a, 2), t.draw(b), t.draw(a);
            t.then([=](vpps4::Renderer&) { tsharp(g->words(table) + 8, tex, Scene::kTexSize / 2, 8); }), t.draw(a, 2);  // (a smaller picture there)
            // A render target read as a texture, drawn into again, read again; across a frame cut too.
            const vpps4::DrawState fill = sc.postDraw(1, sc.materials[7].textures), read = sc.postDraw(2, sc.postTables[1]);
            t.draw(fill), t.draw(read, 2), t.draw(fill), t.draw(read, 2), t.submit(), t.draw(read, 2), t.draw(fill, 2), t.draw(read);
            // The target's address named by a material's T#: the material reads the target.
            t.then([=](vpps4::Renderer&) { tsharp(g->words(table), Scene::kPost + Scene::kTargetStride, Scene::kPostSize, 14); });
            t.draw(a, 2), t.draw(fill), t.draw(a, 2), t.submit(), t.draw(a, 2);
            // A texture's memory becomes a render target: the material that sampled it from memory now reads what was drawn.
            t.then([=](vpps4::Renderer&) { tsharp(g->words(table), tex, Scene::kTexSize, 8); }), t.draw(a, 2);
            vpps4::DrawState into = sc.postDraw(1, sc.materials[7].textures);
            into.color[0] = {tex, Scene::kTexSize, Scene::kTexSize, Scene::kTexSize, 10, 0, 0, 8};
            into.viewport = {32, 32, -32, 32, 1, 0};
            into.scissorRegs.screen = into.scissorRegs.window = into.scissorRegs.generic = {0, 0, 64, 64};
            t.draw(into), t.draw(a, 2), t.draw(c), t.draw(into), t.draw(a, 2), t.submit(), t.draw(a, 2);
            t.write(tex, 4096, 5), t.then([=](vpps4::Renderer& r) { r.invalidate(tex, texBytes); }), t.draw(a, 2), t.submit(), t.draw(a, 2);
            return t.done();
        }, 60, {watches, false});
    }
}

// Memory that changes hands: bytes a draw reads as constants (small buffers the renderer copies when it records the draw) that a
// render target then covers whole, that the CPU rewrites whole or in part, that a GPU write splits, that a frame cut or an idle
// GPU releases; and back. The ranges are exactly the intervals the tracker holds (a CPU write, a target, a texture), so the
// shortcuts that rewrite an interval in place or remember what the tracker said of a range are asked the hard question.
void equalReuse() {
    equivalent("memory reuse", [](Scene& sc) {
        Adverse t(sc);
        Guest* g = &sc.g;
        const vpps4::DrawState a = t.of(3, 5, 0), c = t.of(33, 5, 2);
        const uint64_t tex = uint64_t(g->words(sc.materials[3].textures)[0]) << 8;
        const uint64_t texBytes = 4ull * Scene::kTexSize * Scene::kTexSize;
        vpps4::DrawState e = a, e2 = c;
        vsharp(&e.ps.userData[4], tex + 0x100, 16, 8);   // (128 bytes in the texture's memory, aligned)
        vsharp(&e2.ps.userData[4], tex + 0x200, 16, 8);
        vsharp(&e2.vs.userData[4], tex + 0x1008, 16, 16);  // (and the vertex stage's, not aligned)
        vpps4::DrawState into = sc.postDraw(1, sc.materials[7].textures);
        into.color[0] = {tex, Scene::kTexSize, Scene::kTexSize, Scene::kTexSize, 10, 0, 0, 8};
        into.viewport = {32, 32, -32, 32, 1, 0};
        into.scissorRegs.screen = into.scissorRegs.window = into.scissorRegs.generic = {0, 0, 64, 64};
        const auto both = [&](unsigned n = 2) { t.draw(e, n), t.draw(e2, n); };
        // The memory is a texture and constants; the CPU rewrites it with and without telling.
        t.draw(a, 2), both(), t.draw(a);
        t.write(tex + 0x100, 32, 1), both();
        t.write(tex + 0x100, 32, 2), t.then([=](vpps4::Renderer& r) { r.invalidate(tex + 0x100, 128); }), both();
        t.write(tex + 0x100, 32, 3), t.then([=](vpps4::Renderer& r) { r.invalidate(tex, texBytes); }), both(), t.draw(a);
        t.write(tex + 0x100, 32, 4), t.then([=](vpps4::Renderer& r) { r.invalidate(tex, texBytes); }), both(1);
        // A target covers the whole texture: what the constants read is now the picture.
        t.draw(into), both(), t.draw(a), t.draw(into), both(1);
        // The CPU takes it back, in the interval's own bounds (rewritten in place), then in a part of it.
        t.write(tex + 0x100, 32, 5), t.then([=](vpps4::Renderer& r) { r.invalidate(tex, texBytes); }), both(), t.draw(a);
        t.draw(into), t.then([=](vpps4::Renderer& r) { r.invalidate(tex, texBytes); }), both();
        t.draw(into), t.write(tex + 0x100, 32, 6), t.then([=](vpps4::Renderer& r) { r.invalidate(tex + 0x100, 128); }), both();
        // The GPU writes part of it, and all of it again; a copy and a fill land in it.
        t.then([=](vpps4::Renderer& r) { r.noteGpuWrite(tex + 0x100, 128); }), both(), t.draw(a);
        t.then([=](vpps4::Renderer& r) { r.noteGpuWrite(tex, texBytes); }), both();
        t.then([=](vpps4::Renderer& r) { r.invalidate(tex, texBytes); }), both();
        t.then([=](vpps4::Renderer& r) { r.fill(tex + 0x100, 128, 0x02020202); }), both(), t.draw(a);
        t.then([=](vpps4::Renderer& r) { r.copy(tex + 0x1000, tex + 0x100, 128); }), both();
        // Frame cuts, an idle GPU, a map that changes, buffers dropped, with the memory in each state.
        t.draw(into), t.submit(), both(), t.draw(into), t.submit(), t.submit(), both(1);
        t.then([](vpps4::Renderer& r) { r.gpuIdle(); }), both();
        t.draw(into), t.then([](vpps4::Renderer& r) { r.gpuIdle(); }), both();
        t.then([g](vpps4::Renderer&) { ++g->generation; }), both();
        t.draw(into), t.then([](vpps4::Renderer& r) { r.dropGuestBuffers(); }), both(), t.draw(a);
        // A label waiting for the GPU where the constants are, and the command processor reading them.
        t.then([=](vpps4::Renderer& r) { r.afterGpu(tex + 0x100, 8, [] {}, true); }), both(), t.submit(), both();
        t.then([=](vpps4::Renderer& r) { r.cpuAccessed(tex + 0x200, 128); }), both(), t.draw(into), both();
        t.submit(), t.submit(), t.submit(), both(), t.draw(a);
        return t.done();
    }, 60);
}

// What the caches of the shortcuts must forget or tell apart: a range touched by draws that were skipped (the frame cut that
// follows records nothing and empties the busy ranges), and the same address asked about with two sizes while the tracker's
// intervals stay as they are.
void equalCaches() {
    equivalent("skipped draws", [](Scene& sc) {
        Adverse t(sc);
        const vpps4::DrawState a = t.of(3, 5, 0);
        // (an odd index address: the index buffer is bound, its range touched, and then the draw is skipped as misaligned)
        vpps4::DrawState bad = a, ok = a;
        bad.indexAddress = sc.meshes[5].indices + 1;
        ok.indexAddress = sc.meshes[5].indices + 2, ok.count = a.count - 2;  // (inside the range `bad` touched)
        t.draw(a, 2), t.submit();
        t.draw(bad), t.submit();  // nothing was recorded: no frame, and what `bad` touched is not busy
        t.draw(ok), t.draw(a);
        t.draw(bad), t.draw(ok), t.submit(), t.draw(ok);
        return t.done();
    }, 4);
    // One address, two sizes, no change of the tracker's intervals in between: a target covers a texture, the CPU rewrites 128
    // bytes of it, and a draw reads those 128 bytes as constants, then 144 (the extra 16 are the target's again). 0x80 and 0x90 at
    // that address share a slot of Impl::quiet()'s table.
    equivalent("one address, two sizes", [](Scene& sc) {
        Adverse t(sc);
        Guest* g = &sc.g;
        const vpps4::DrawState a = t.of(3, 5, 0);
        const uint64_t tex = uint64_t(g->words(sc.materials[3].textures)[0]) << 8;
        vpps4::DrawState small = a, wide = a;
        vsharp(&small.ps.userData[4], tex + 0x100, 16, 8);
        vsharp(&wide.ps.userData[4], tex + 0x100, 16, 9);
        vpps4::DrawState into = sc.postDraw(1, sc.materials[7].textures);
        into.color[0] = {tex, Scene::kTexSize, Scene::kTexSize, Scene::kTexSize, 10, 0, 0, 8};
        into.viewport = {32, 32, -32, 32, 1, 0};
        into.scissorRegs.screen = into.scissorRegs.window = into.scissorRegs.generic = {0, 0, 64, 64};
        t.draw(a, 2), t.draw(into);
        t.write(tex + 0x100, 32, 1), t.then([=](vpps4::Renderer& r) { r.invalidate(tex + 0x100, 128); });
        t.draw(small, 2), t.draw(wide, 2), t.draw(small), t.draw(wide);
        t.draw(into), t.write(tex + 0x100, 32, 2), t.then([=](vpps4::Renderer& r) { r.invalidate(tex + 0x100, 128); });
        t.draw(small), t.draw(wide, 2), t.submit(), t.draw(small), t.draw(wide);
        return t.done();
    }, 10);
}

// Programs: rewritten where they are with and without a word to the renderer, the registers of a stage that change its
// code, another fetch subroutine, the same program after a frame cut.
void equalPrograms() {
    equivalent("programs", [](Scene& sc) {
        Adverse t(sc);
        const vpps4::DrawState a = t.of(3, 5, 0), b = t.of(4, 6, 1), c = t.of(33, 5, 2);
        Guest* g = &sc.g;
        const uint64_t ps = sc.ps[3].address, vs = sc.vs[0].address;
        size_t literal = 0;
        for (size_t i = 0; i < sc.psMaterial.code.size(); ++i)
            if (sc.psMaterial.code[i] == 0x3f123456) literal = i;
        const uint32_t first = g->words(ps)[literal];
        const auto set = [g, ps, literal](uint32_t v) { return [=](vpps4::Renderer&) { g->words(ps)[literal] = v; }; };
        t.draw(a, 2), t.draw(b), t.draw(a), t.draw(c), t.draw(a);
        t.then(set(0x3f000000)), t.draw(a, 2), t.draw(c), t.draw(a);          // rewritten, nobody told
        t.then(set(first)), t.draw(a, 2);                                       // and back
        t.then(set(0x3e800000)), t.then([=](vpps4::Renderer& r) { r.invalidate(ps, 4096); }), t.draw(a, 2), t.draw(b);
        t.then(set(0x3f000000)), t.then([=](vpps4::Renderer& r) { r.coherencePoint(ps, 4096); }), t.draw(a, 2);
        t.then(set(first)), t.submit(), t.draw(a, 2), t.draw(c);                // across a frame cut
        t.then(set(0x3e000000)), t.draw(a), t.submit(), t.draw(a), t.then(set(first)), t.draw(a, 2);
        // The words after the code (its constant tables): another program for the renderer, the same shader.
        t.then([=](vpps4::Renderer&) { g->words(ps)[Scene::kProgramWords - 1] ^= 0x10; }), t.draw(a, 2), t.draw(c);
        t.then([=](vpps4::Renderer&) { g->words(vs)[Scene::kProgramWords - 2] ^= 0x10; }), t.draw(a, 2), t.draw(b), t.draw(a);
        // Registers that are part of the program: interpolants, export format, the vertex output control.
        vpps4::DrawState e = a;
        e.ps.psNumInterp = 2, e.ps.psInputCntl[1] = 1, t.draw(e, 2), t.draw(a), t.draw(e);
        e = a, e.ps.exportFormat[0] = 9, t.draw(e, 2), t.draw(a);
        e = a, e.vs.instanceStepRate = {2, 3}, t.draw(e, 2), t.draw(a, 2);
        e = a, e.raster.clipSpaceZeroToW = false, t.draw(e, 2), t.draw(a);
        e = a, e.alphaToMask = true, t.draw(e, 2), t.draw(a);
        // Another copy of the fetch subroutine (other address, same words), and another one (other words).
        const uint64_t copy = sc.vs[0].fetchAddress + 0x800, other = sc.vs[1].fetchAddress;
        t.then([=](vpps4::Renderer&) { std::memcpy(g->words(copy), g->words(copy - 0x800), 4 * 7); });
        e = a, e.vs.userData[0] = uint32_t(copy), e.vs.userData[1] = uint32_t(copy >> 32);
        e.vs.fetch = {sc.g.words(copy), sc.vs[0].fetch.size()};
        t.draw(e, 2), t.draw(a), t.draw(e);
        const vpps4::DrawState viaCopy = e;
        e.vs.userData[0] = uint32_t(other), e.vs.userData[1] = uint32_t(other >> 32), e.vs.fetch = sc.vs[1].fetch;
        t.draw(e, 2), t.draw(a, 2);
        // The copy rewritten where it is (the draws that call it see another subroutine), with and without a word to the renderer.
        t.then([=](vpps4::Renderer&) { g->words(copy)[2] ^= 0x100; }), t.draw(viaCopy, 2), t.draw(a, 2), t.draw(viaCopy);
        t.then([=](vpps4::Renderer&) { g->words(copy)[2] ^= 0x100; }), t.draw(viaCopy, 2), t.draw(e);
        t.then([=](vpps4::Renderer&) { g->words(copy)[3] ^= 0x4; }), t.then([=](vpps4::Renderer& r) { r.invalidate(copy, 64); }), t.draw(viaCopy, 2);
        // A draw without a pixel stage between two with one (the stage's state is not the next draw's).
        t.draw(t.of(3, 5, 0, false)), t.draw(a, 2);
        return t.done();
    }, 60);
}

// Targets and textures evicted while idle and made again; a target cleared, drawn, cleared; depth targets with and without
// colour; many frame cuts in between.
void equalTargets() {
    equivalent("targets", [](Scene& sc) {
        Adverse t(sc, false);
        const vpps4::DrawState a = t.of(3, 5, 0), b = t.of(4, 6, 1);
        const vpps4::DrawState fill = sc.postDraw(1, sc.materials[7].textures), read = sc.postDraw(2, sc.postTables[1]);
        vpps4::DrawState pre = Scene::passState(0, Scene::kEyeDepth, Scene::kEyeW, Scene::kEyeH);
        pre.depthStencil.depthTest = pre.depthStencil.depthWrite = true;
        pre.depthStencil.depthFunc = 1;
        const vpps4::DrawState depthOnly = sc.materialDraw(pre, 3, 5, 4, t.arena(3), false);
        for (unsigned round = 0; round < 6; ++round) {
            t.draw(Scene::depthClear(pre)), t.draw(depthOnly, 2), t.draw(a, 2), t.draw(b), t.draw(a);
            t.draw(fill), t.draw(read, 2), t.submit();
            t.draw(a, 2), t.draw(read), t.submit();
            // Idle for long enough that everything not used is evicted (16 submits make an eviction pass), with one image kept alive.
            for (unsigned k = 0; k < 40; ++k) {
                if (round % 2) t.draw(read);
                if (round == 3 && k == 20) t.draw(a);
                t.submit();
            }
        }
        return t.done();
    }, 40, {true, true});
}

// ---- The draw cache, observed ----------------------------------------------------------------------------------------
// With the probe on the renderer makes the draw cache's keys and looks its blocks up without taking anything from them
// (RATES dc_*). Frames that are the same find their blocks good, and the blocks hold what the draws resolve; each thing
// that makes a block stale does; none of it changes what is drawn; with the probe off nothing is counted.
struct DcCounts {
    uint64_t fixedHit, fixedMiss, stageHit, stageMiss, differ, invalid[5], slow[4], texTarget, texMemory;
};
DcCounts dcCounts(vpps4::Renderer& r) {
    r.publishProbe(1);
    const auto c = [](const char* name) { return vpgpu::probe::counter("RATES", name).get(); };
    return {c("dc_fixed_hit"), c("dc_fixed_miss"), c("dc_stage_hit"), c("dc_stage_miss"), c("dc_would_differ"),
            {c("dc_invalid_frame"), c("dc_invalid_layout"), c("dc_invalid_surface"), c("dc_invalid_pass"), c("dc_invalid_clear")},
            {c("dc_slow_path_gs"), c("dc_slow_path_rectlist"), c("dc_slow_path_indirect"), c("dc_slow_path_otro")},
            c("dc_tex_target"), c("dc_tex_memory")};
}
void drawCacheFrames() {
    namespace pr = vpgpu::probe;
    enum { kFrame, kLayout, kSurface, kPass, kClear };
    constexpr unsigned kFrames = 5, kEye0 = 40, kEye1 = 24;
    std::vector<DcCounts> marks;
    const auto script = [&marks](Scene& sc) {
        Script s;
        const auto draw = [&s](const vpps4::DrawState& d) {
            Step st;
            st.draw = d;
            s.push_back(std::move(st));
        };
        const auto mark = [&] { s.push_back(call([&marks](vpps4::Renderer& r) { marks.push_back(dcCounts(r)); })); };
        vpps4::DrawState eye[2];
        for (unsigned e = 0; e < 2; ++e) {
            eye[e] = Scene::passState(Scene::kEyeColor + Scene::kTargetStride * e, Scene::kEyeDepth + Scene::kTargetStride * e, Scene::kEyeW,
                                      Scene::kEyeH);
            eye[e].depthStencil.depthTest = true;
            eye[e].depthStencil.depthFunc = 3;
        }
        // (the objects' constants are written once: every frame is the same)
        for (unsigned k = 0; k < (kEye0 + kEye1) * 0x1c0 / 4; ++k) sc.g.words(kArena)[k] = sc.rnd();
        const auto object = [&](unsigned e, unsigned i) {
            return sc.materialDraw(eye[e], 1 + i % 4, i % Scene::kMeshes, 4 + e, kArena + 0x1c0ull * (e * kEye0 + i), true);
        };
        // Identical frames: two passes of draws with one fixed state each, a mark after each frame.
        for (unsigned f = 0; f < kFrames; ++f) {
            for (unsigned i = 0; i < kEye0; ++i) draw(object(0, i));
            for (unsigned i = 0; i < kEye1; ++i) draw(object(1, i));
            s.push_back(submitStep());
            mark();
        }
        // What makes a block stale, each followed by the same draw twice (a mark after each): the first finds its block
        // stale, the second good again.
        const vpps4::DrawState a = object(0, 0);
        const auto twice = [&] {
            mark();
            draw(a), mark();
            draw(a), mark();
        };
        draw(a), draw(a);
        // 1. Another pass: a draw into other targets.
        draw(object(1, 0));
        twice();
        // 2. A clear: of the draw's depth target (a RectList; the draw after it carries the clear into its pass).
        draw(Scene::depthClear(Scene::passState(0, Scene::kEyeDepth, Scene::kEyeW, Scene::kEyeH)));
        draw(a);
        twice();
        // 3. The target given again: its memory drawn as a target of another size.
        draw(sc.materialDraw(Scene::passState(Scene::kEyeColor, 0, Scene::kEyeW / 2, Scene::kEyeH / 2), 1, 0, 4, kArena, true));
        twice();
        // 4. A texture of the draw written by the title (the host's watch reports it).
        const uint64_t texture = kTex + 0x8000ull * 7;  // (material 1's first)
        s.push_back(call([&sc](vpps4::Renderer& r) {
            for (unsigned k = 0; k < 64; ++k) sc.g.words(texture)[k] ^= 0x00FF00FF;
            r.invalidate(texture, Scene::kTexSize * Scene::kTexSize * 4);
        }));
        twice();
        // 5. The tracker's intervals: the GPU writes a range of its own.
        s.push_back(call([](vpps4::Renderer& r) { r.noteGpuWrite(kArena + kArenaBytes, 4096); }));
        twice();
        // A texture that is a render target (the second full-screen draw samples what the first drew, twice).
        draw(sc.postDraw(0, sc.postTables[1]));
        draw(sc.postDraw(1, sc.postTables[0])), draw(sc.postDraw(1, sc.postTables[0]));
        s.push_back(submitStep());
        mark();
        return s;
    };
    pr::setEnabled(true);
    pr::registry().reset();
    const Outcome on = run(script, false);
    pr::setEnabled(false);
    pr::registry().reset();
    CHECK(marks.size() == kFrames + 5 * 3 + 1);
    if (marks.size() != kFrames + 5 * 3 + 1) return;
    const auto delta = [&](size_t k) {
        DcCounts d = marks[k];
        const DcCounts& b = marks[k - 1];
        d.fixedHit -= b.fixedHit, d.fixedMiss -= b.fixedMiss, d.stageHit -= b.stageHit, d.stageMiss -= b.stageMiss, d.differ -= b.differ;
        d.texTarget -= b.texTarget, d.texMemory -= b.texMemory;
        for (unsigned i = 0; i < 5; ++i) d.invalid[i] -= b.invalid[i];
        for (unsigned i = 0; i < 4; ++i) d.slow[i] -= b.slow[i];
        return d;
    };
    if (std::getenv("VPPS4_SHOW_COUNTS"))
        for (size_t k = 0; k < marks.size(); ++k) {
            const DcCounts d = k ? delta(k) : marks[0];
            std::printf("mark %zu: fixed %llu/%llu stage %llu/%llu differ %llu invalid f%llu l%llu s%llu p%llu c%llu slow %llu %llu %llu %llu tex %llu/%llu\n", k,
                        (unsigned long long)d.fixedHit, (unsigned long long)d.fixedMiss, (unsigned long long)d.stageHit,
                        (unsigned long long)d.stageMiss, (unsigned long long)d.differ, (unsigned long long)d.invalid[0],
                        (unsigned long long)d.invalid[1], (unsigned long long)d.invalid[2], (unsigned long long)d.invalid[3],
                        (unsigned long long)d.invalid[4], (unsigned long long)d.slow[0], (unsigned long long)d.slow[1],
                        (unsigned long long)d.slow[2], (unsigned long long)d.slow[3], (unsigned long long)d.texTarget,
                        (unsigned long long)d.texMemory);
        }
    // The first frame makes the blocks: every draw is looked at, none finds a good block of an earlier frame.
    CHECK(marks[0].fixedHit + marks[0].fixedMiss == kEye0 + kEye1 && marks[0].stageHit + marks[0].stageMiss == 2 * (kEye0 + kEye1));
    CHECK(marks[0].fixedMiss >= 2 && marks[0].invalid[kFrame] == 0);
    // The frames after it: all but the first draw of each pass find their fixed block good (that one finds it of another
    // frame and pass), every stage finds its block, and no good block holds anything else than what the draw resolved.
    for (size_t k = 2; k < kFrames; ++k) {
        const DcCounts d = delta(k);
        CHECK(d.fixedHit + d.fixedMiss == kEye0 + kEye1 && d.fixedHit * 10 >= (kEye0 + kEye1) * 9);
        CHECK(d.fixedMiss == 2 && d.invalid[kFrame] == 2 && d.invalid[kPass] == 2);
        CHECK(d.stageHit == 2 * (kEye0 + kEye1) && d.stageMiss == 0);
        CHECK(d.differ == 0);
        CHECK(d.texMemory == 3 * (kEye0 + kEye1) && d.texTarget == 0);  // (three textures of guest memory per pixel stage)
        for (uint64_t n : d.slow) CHECK(n == 0);
    }
    // Each adverse case: the draw after it does not find its block good, for that reason; the same draw again does.
    const auto stale = [&](size_t at, unsigned part) {
        const DcCounts first = delta(at + 1), second = delta(at + 2);
        CHECK(first.fixedHit == 0 && first.fixedMiss == 1 && first.invalid[part] == 1);
        CHECK(second.fixedHit == 1 && second.fixedMiss == 0);
        for (unsigned i = 0; i < 5; ++i) CHECK(second.invalid[i] == 0);
        return first;
    };
    size_t at = kFrames;
    stale(at, kPass);
    {
        // (the clear itself and the draw that carries it are not draws a block can serve)
        const DcCounts before = delta(at += 3);
        CHECK(before.slow[1] == 1 && before.slow[3] == 1 && before.fixedHit + before.fixedMiss == 0);
        stale(at, kClear);
    }
    stale(at += 3, kSurface);
    {
        const DcCounts d = stale(at += 3, kLayout);
        CHECK(d.invalid[kSurface] == 0 && d.invalid[kClear] == 0);
    }
    {
        const DcCounts d = stale(at += 3, kLayout);
        CHECK(d.invalid[kSurface] == 0 && d.invalid[kClear] == 0 && d.invalid[kPass] == 0 && d.invalid[kFrame] == 0);
    }
    // A texture an image backs is not one a block holds (resolved in every draw): counted apart.
    CHECK(delta(marks.size() - 1).texTarget == 2);
    CHECK(marks.back().differ == 0);
    CHECK(marks.back().slow[0] == 0 && marks.back().slow[2] == 0);
    // Looking changes nothing of what is drawn, and with the probe off nothing is looked at.
    marks.clear();
    const Outcome off = run(script, false);
    CHECK(same(on, off, "draw cache observed"));
    CHECK(on.drawn >= kFrames * (kEye0 + kEye1));
    pr::setEnabled(true);
    for (const char* name : {"dc_fixed_hit", "dc_fixed_miss", "dc_stage_hit", "dc_stage_miss", "dc_would_differ", "dc_slow_path_rectlist",
                             "dc_slow_path_otro", "dc_invalid_frame", "dc_invalid_pass"})
        CHECK(pr::counter("RATES", name).get() == 0);
    pr::setEnabled(false);
    pr::registry().reset();
}

}  // namespace

int main(int argc, char** argv) {
    const std::string t = argc > 1 ? argv[1] : "";
    if (t == "bench") return bench() ? 1 : 0;
    else if (t == "digest") {
        digestMode = true;
        equalPicture(), equalBuffers(), equalTextures(), equalReuse(), equalPrograms(), equalTargets();
    } else if (t == "equal_picture") equalPicture();
    else if (t == "equal_buffers") equalBuffers();
    else if (t == "equal_textures") equalTextures();
    else if (t == "equal_reuse") equalReuse();
    else if (t == "equal_programs") equalPrograms();
    else if (t == "equal_targets") equalTargets();
    else if (t == "equal_caches") equalCaches();
    else if (t == "draw_cache_frames") drawCacheFrames();
    else return std::fprintf(stderr, "unknown test %s\n", t.c_str()), 2;
    return failures ? 1 : 0;
}
