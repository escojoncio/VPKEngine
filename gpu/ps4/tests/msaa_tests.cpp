// MSAA, resolve, FMASK, fast clear and load/store planning: pure functions, the planner on synthetic
// frames, the lifter on a multisampled-load shader, and the renderer on a capturing backend.
// Usage: vpps4_msaa_tests TEST
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "vp4gpu/core/backend.h"
#include "vp4gpu/core/device.h"
#include "vp4gpu/core/frame_compiler.h"
#include "vp4msl/msl.h"
#include "vpgcn/lift.h"
#include "vpps4/foveation.h"
#include "vpps4/lifetime.h"
#include "vpps4/msaa.h"
#include "vpps4/renderer.h"

#ifndef VPPS4_LIFT_DIR
#error "VPPS4_LIFT_DIR must point at gpu/ps4/engine/gcn/tests/lift"
#endif
#ifndef VPPS4_MSAA_LIFT_DIR
#error "VPPS4_MSAA_LIFT_DIR must point at gpu/ps4/tests/lift"
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

using namespace vp4gpu;

// ---- Pure functions ------------------------------------------------------------------------------

void samplePlan() {
    using vpps4::planSamples;
    {
        const uint8_t c[] = {4, 0, 0};
        auto p = planSamples(c, 4, 4);
        CHECK(p.host == 4 && p.guest == 4 && !p.eqaa && !p.reduced);
    }
    {  // EQAA: 4 colour fragments, 8 depth samples -> 4 on the host
        const uint8_t c[] = {4};
        auto p = planSamples(c, 8, 8);
        CHECK(p.host == 4 && p.guest == 8 && p.eqaa && p.reduced);
    }
    {  // the app's limit
        const uint8_t c[] = {8};
        CHECK(planSamples(c, 8, 4).host == 4);
        CHECK(planSamples(c, 8, 2).host == 2);
        CHECK(planSamples(c, 8, 3).host == 2);  // power of two
        CHECK(planSamples(c, 8, 1).host == 1);
        CHECK(planSamples(c, 8, 0).host == 1);
    }
    {  // mixed targets: the smallest count (Metal needs one per pass)
        const uint8_t c[] = {4, 2};
        CHECK(planSamples(c, 4, 8).host == 2);
    }
    {  // depth only (a shadow pass), and nothing bound
        CHECK(planSamples({}, 4, 4).host == 4);
        CHECK(planSamples({}, 0, 4).host == 1);
    }
    {  // colour 1x with a multisampled depth: 1x
        const uint8_t c[] = {1};
        auto p = planSamples(c, 4, 4);
        CHECK(p.host == 1 && p.reduced);
    }
}

void fmask() {
    using vpps4::fmaskIdentity;
    using V = vpps4::FmaskValue;
    CHECK((fmaskIdentity(4, 4) == V{0x3210, 0}));
    CHECK((fmaskIdentity(2, 2) == V{0x10, 0}));
    CHECK((fmaskIdentity(8, 8) == V{0x76543210, 0}));
    CHECK((fmaskIdentity(16, 16) == V{0x76543210, 0xfedcba98}));
    // EQAA: 8 coverage samples over 4 stored fragments
    CHECK((fmaskIdentity(8, 4) == V{0x32103210, 0}));
    // 1 fragment: every sample lives in slot 0
    CHECK((fmaskIdentity(4, 1) == V{0, 0}));
    CHECK((fmaskIdentity(0, 4) == V{0, 0}));
    CHECK(vpps4::isFmaskFormat(47) && vpps4::isFmaskFormat(56) && !vpps4::isFmaskFormat(46) && !vpps4::isFmaskFormat(57));
    // each fragment index is below the fragment count and every sample is mapped
    for (uint8_t cov : {2, 4, 8, 16})
        for (uint8_t fr : {1, 2, 4, 8}) {
            if (fr > cov) continue;
            const V v = fmaskIdentity(cov, fr);
            for (unsigned s = 0; s < cov; ++s) {
                const uint32_t nib = (s < 8 ? v.lo >> (4 * s) : v.hi >> (4 * (s - 8))) & 15;
                CHECK(nib == s % fr);
            }
        }
}

void depthResolve() {
    using vpps4::depthResolveMode;
    using M = ResolveMode;
    CHECK(depthResolveMode(true, 0, 1) == M::Sample0);   // a hardware copy of sample 0
    CHECK(depthResolveMode(false, 0, 1) == M::Min);      // LESS: nearest = smallest
    CHECK(depthResolveMode(false, 0, 3) == M::Min);      // LEQUAL
    CHECK(depthResolveMode(false, 0, 4) == M::Max);      // GREATER (reversed Z)
    CHECK(depthResolveMode(false, 0, 6) == M::Max);      // GEQUAL
    CHECK(depthResolveMode(false, 0, 7) == M::Sample0);  // ALWAYS: no test, no preference
    CHECK(depthResolveMode(true, 2, 1) == M::Min);       // another sample than 0: nearest (Metal has no sample N)
}

void colorResolve() {
    const std::array<std::array<float, 4>, 4> s = {{{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}, {1, 1, 1, 1}}};
    const auto r = vpps4::resolveColor(s);
    CHECK(r[0] == 0.5f && r[1] == 0.5f && r[2] == 0.5f && r[3] == 1.0f);
    const std::array<std::array<float, 4>, 2> two = {{{1, 0, 0, 0}, {0, 0, 0, 0}}};
    CHECK(vpps4::resolveColor(two)[0] == 0.5f);
    CHECK(vpps4::resolveColor({})[0] == 0.0f);
}

void fastClear() {
    bool exact = true;
    // 8_8_8_8 UNORM: bytes R G B A = 80 40 20 ff
    auto c = vpps4::fastClearColor(10, 0, 0, {0xff204080u, 0}, exact);
    CHECK(exact);
    CHECK(std::abs(c[0] - 128.0 / 255) < 1e-9 && std::abs(c[1] - 64.0 / 255) < 1e-9 && std::abs(c[2] - 32.0 / 255) < 1e-9 && c[3] == 1.0);
    // COMP_SWAP ALT: BGRA in memory
    c = vpps4::fastClearColor(10, 0, 1, {0xff204080u, 0}, exact);
    CHECK(std::abs(c[0] - 32.0 / 255) < 1e-9 && std::abs(c[2] - 128.0 / 255) < 1e-9);
    // 16_16_16_16 FLOAT: half 1.0 (3c00), 0.5 (3800), 0, 1
    c = vpps4::fastClearColor(12, 7, 0, {0x38003c00u, 0x3c000000u}, exact);
    CHECK(exact && c[0] == 1.0 && c[1] == 0.5 && c[2] == 0.0 && c[3] == 1.0);
    // 32 bit float, one component
    float one = 2.5f;
    uint32_t bits;
    std::memcpy(&bits, &one, 4);
    c = vpps4::fastClearColor(4, 7, 0, {bits, 0}, exact);
    CHECK(c[0] == 2.5 && c[3] == 1.0);
    // UINT
    c = vpps4::fastClearColor(1, 4, 0, {7, 0}, exact);
    CHECK(c[0] == 7.0);
    // sRGB: encoded 0.5 is linear ~0.214
    c = vpps4::fastClearColor(10, 6, 0, {0x00000080u, 0}, exact);
    CHECK(c[0] > 0.21 && c[0] < 0.22);
    // not decoded
    exact = true;
    vpps4::fastClearColor(6, 7, 0, {0, 0}, exact);
    CHECK(!exact);

    // every COMP_SWAP of 8_8_8_8: where the logical colour (R 0x80, G 0x40, B 0x20, A 0x10) sits in memory
    auto near = [](double a, double b) { return std::abs(a - b) < 1e-9; };
    const struct { uint8_t swap; uint32_t word; } rgba8[] = {
        {0, 0x10204080u},  // STD: R G B A
        {1, 0x10804020u},  // ALT: B G R A
        {2, 0x80402010u},  // STD_REV: A B G R
        {3, 0x20408010u},  // ALT_REV: A R G B
    };
    for (const auto& t : rgba8) {
        exact = true;
        c = vpps4::fastClearColor(10, 0, t.swap, {t.word, 0}, exact);
        CHECK(exact && near(c[0], 128.0 / 255) && near(c[1], 64.0 / 255) && near(c[2], 32.0 / 255) && near(c[3], 16.0 / 255));
    }
    // sRGB converts the colour, never the alpha, wherever it sits
    c = vpps4::fastClearColor(10, 6, 2, {0x80808080u, 0}, exact);
    CHECK(c[0] > 0.21 && c[0] < 0.22 && near(c[3], 128.0 / 255));
    // 2_10_10_10 (R in the low bits) and 10_10_10_2 reversed (alpha low, red on top)
    c = vpps4::fastClearColor(9, 0, 0, {1023u | (512u << 20) | (3u << 30), 0}, exact);
    CHECK(exact && c[0] == 1.0 && c[1] == 0.0 && near(c[2], 512.0 / 1023) && c[3] == 1.0);
    c = vpps4::fastClearColor(8, 0, 2, {2u | (1023u << 2) | (512u << 12), 0}, exact);  // A=2/3, B=1, G=512/1023, R=0
    CHECK(exact && near(c[3], 2.0 / 3) && c[2] == 1.0 && near(c[1], 512.0 / 1023) && c[0] == 0.0);
    // 16_16_16_16 UNORM, BGRA order: memory B G R A
    c = vpps4::fastClearColor(12, 0, 1, {0x0000ffffu, 0xffff8000u}, exact);
    CHECK(exact && c[2] == 1.0 && c[1] == 0.0 && near(c[0], 32768.0 / 65535) && c[3] == 1.0);
    // 16_16_16_16 SINT keeps the integers
    c = vpps4::fastClearColor(12, 5, 0, {0xfffe0003u, 0x00000001u}, exact);
    CHECK(exact && c[0] == 3.0 && c[1] == -2.0 && c[2] == 1.0);
    // two components: RG, RA (ALT) and AR (ALT_REV)
    c = vpps4::fastClearColor(3, 0, 1, {0x4080u, 0}, exact);
    CHECK(near(c[0], 128.0 / 255) && c[1] == 0.0 && near(c[3], 64.0 / 255));
    c = vpps4::fastClearColor(3, 0, 3, {0x4080u, 0}, exact);
    CHECK(near(c[3], 128.0 / 255) && near(c[0], 64.0 / 255));
    // alpha-only 8 bit (ALT_REV)
    c = vpps4::fastClearColor(1, 0, 3, {0xffu, 0}, exact);
    CHECK(c[0] == 0.0 && c[3] == 1.0);
    // 5_6_5
    c = vpps4::fastClearColor(14, 0, 0, {31u | (63u << 5), 0}, exact);
    CHECK(exact && c[0] == 1.0 && c[1] == 1.0 && c[2] == 0.0 && c[3] == 1.0);
    // 11_11_10 floats are not decoded
    exact = true;
    vpps4::fastClearColor(7, 7, 0, {0, 0}, exact);
    CHECK(!exact);
}

void resolveWay() {
    using vpps4::ResolveWay;
    vpps4::ResolveCase c;
    c.sourceMultisampled = true;
    c.lastPassOpenForSource = true;
    CHECK(vpps4::planResolve(c) == ResolveWay::InPass);
    c.lastPassOpenForSource = false;
    CHECK(vpps4::planResolve(c) == ResolveWay::OwnPass);
    c.sourceMultisampled = false;
    CHECK(vpps4::planResolve(c) == ResolveWay::Copy);
    c.destinationIsSource = true;
    CHECK(vpps4::planResolve(c) == ResolveWay::Skip);
    c.destinationIsSource = false;
    c.destinationMultisampled = true;
    CHECK(vpps4::planResolve(c) == ResolveWay::Skip);
    c.destinationMultisampled = false;
    c.formatsMatch = false;
    CHECK(vpps4::planResolve(c) == ResolveWay::Skip);
}

// ---- The planner on synthetic frames -----------------------------------------------------------------

struct Synth {
    ResourceTable res;
    vpps4::LifetimeTracker life;
    uint64_t frame = 0;

    TextureId color(uint8_t samples = 1) {
        TextureDesc d;
        d.format = Format::RGBA8Unorm;
        d.width = d.height = 64;
        d.samples = samples;
        d.usage = usage::sampled | usage::colorTarget;
        return res.createTexture(d);
    }
    TextureId depth(uint8_t samples = 1) {
        TextureDesc d;
        d.format = Format::D32Float;
        d.width = d.height = 64;
        d.samples = samples;
        d.usage = usage::sampled | usage::depthTarget;
        return res.createTexture(d);
    }
    static Draw drawReading(TextureId t) {
        Draw d;
        if (t) d.bindings.push_back(read(ResourceRange::wholeTexture(t), stage::fragment));
        return d;
    }
    static RenderPassDesc pass(TextureId c, TextureId z, LoadAction cl, LoadAction zl, TextureId resolve = {}) {
        RenderPassDesc p;
        if (c) {
            p.colors[0].texture = c;
            p.colors[0].load = cl;
            p.colors[0].resolve = resolve;
            p.colors[0].store = resolve ? StoreAction::StoreAndResolve : StoreAction::Store;
        }
        if (z) {
            p.depth.texture = z;
            p.depth.load = zl;
            p.depth.store = StoreAction::Store;
        }
        return p;
    }
    vpps4::LifetimeStats run(Frame& f) {
        const auto s = life.plan(f, res, frame);
        ++frame;
        return s;
    }
    static const RenderPass& rp(const Frame& f, size_t i) { return std::get<RenderPass>(f.commands()[i]); }
};

void lifetimeDepth() {
    Synth s;
    const TextureId c = s.color(), z = s.depth();
    for (int n = 0; n < 12; ++n) {
        s.life.noteRead(c, s.frame);  // the colour is shown every frame
        Frame f;
        f.beginRenderPass(Synth::pass(c, z, LoadAction::Clear, LoadAction::Clear));
        f.draw(Draw{});
        f.endRenderPass();
        const auto st = s.run(f);
        const Attachment& depth = Synth::rp(f, 0).desc.depth;
        const Attachment& color = Synth::rp(f, 0).desc.colors[0];
        CHECK(color.store == StoreAction::Store);
        if (n < 3) CHECK(depth.store == StoreAction::Store);  // warming up
        else if (n == 3) CHECK(depth.store == StoreAction::DontCare && st.storesDropped == 1);
        else CHECK(depth.store == StoreAction::DontCare);
        if (n >= 3) {
            // with the dropped store the depth buffer stays on tile
            CompiledFrame cf = compileFrame(f, &s.res);
            CHECK(cf.encoders.size() == 1);
            CHECK(cf.encoders[0].tileLocal(kDepthSlot));
            CHECK(!cf.encoders[0].tileLocal(0));
        }
    }
}

void lifetimeMiss() {
    Synth s;
    const TextureId c = s.color(), z = s.depth();
    auto cleared = [&](bool clear) {
        s.life.noteRead(c, s.frame);
        Frame f;
        f.beginRenderPass(Synth::pass(c, z, LoadAction::Clear, clear ? LoadAction::Clear : LoadAction::Load));
        f.draw(Draw{});
        f.endRenderPass();
        const auto st = s.run(f);
        return std::pair{Synth::rp(f, 0).desc.depth.store, st.misses};
    };
    for (int n = 0; n < 6; ++n) cleared(true);
    CHECK(s.life.contentsLost(z));
    auto [store, misses] = cleared(false);  // the title stopped clearing: it loads what was dropped
    CHECK(misses == 1 && store == StoreAction::Store);
    CHECK(s.life.sticky(z));
    // from now on it is stored whatever the title does
    auto [store2, misses2] = cleared(true);
    CHECK(store2 == StoreAction::Store && misses2 == 0);
    CHECK(!s.life.contentsLost(z));
}

void lifetimeSampled() {
    Synth s;
    const TextureId c = s.color(), z = s.depth();
    for (int n = 0; n < 10; ++n) {
        Frame f;
        f.beginRenderPass(Synth::pass(c, z, LoadAction::Clear, LoadAction::Clear));
        f.draw(Draw{});
        f.endRenderPass();
        // the depth is sampled by a later pass of the same frame (SSAO): kept; the colour is not
        f.beginRenderPass(Synth::pass({}, {}, LoadAction::Clear, LoadAction::Clear));
        RenderPassDesc other;
        other.colors[0].texture = s.color();
        other.colors[0].load = LoadAction::Clear;
        f.endRenderPass();
        f.beginRenderPass(other);
        f.draw(Synth::drawReading(z));
        f.endRenderPass();
        s.run(f);
        CHECK(Synth::rp(f, 0).desc.depth.store == StoreAction::Store);
        if (n >= 4) CHECK(Synth::rp(f, 0).desc.colors[0].store == StoreAction::DontCare);
    }
    // sampled across frames only (history): kept while it is being read, dropped after
    Synth h;
    const TextureId z2 = h.depth(), c2 = h.color();
    for (int n = 0; n < 14; ++n) {
        if (n < 8) h.life.noteRead(z2, h.frame);
        Frame f;
        f.beginRenderPass(Synth::pass(c2, z2, LoadAction::Clear, LoadAction::Clear));
        f.draw(Draw{});
        f.endRenderPass();
        h.run(f);
        const StoreAction st = Synth::rp(f, 0).desc.depth.store;
        if (n < 10) CHECK(st == StoreAction::Store);  // read up to frame 7, held for two frames after
        if (n >= 11) CHECK(st == StoreAction::DontCare);
    }
}

void lifetimeMsaa() {
    Synth s;
    const TextureId ms = s.color(4), one = s.color(1), z = s.depth(4);
    auto frameWith = [&](bool readCompanion, bool readSamples) {
        if (readCompanion) s.life.noteRead(one, s.frame);
        if (readSamples) s.life.noteSamplesRead(ms, s.frame);
        Frame f;
        f.beginRenderPass(Synth::pass(ms, z, LoadAction::Clear, LoadAction::Clear, one));
        f.draw(Draw{});
        f.endRenderPass();
        const auto st = s.run(f);
        return std::pair{std::move(f), st};
    };
    for (int n = 0; n < 3; ++n) {  // warm-up: everything kept
        auto [f, st] = frameWith(true, false);
        CHECK(Synth::rp(f, 0).desc.colors[0].store == StoreAction::StoreAndResolve);
    }
    {  // steady state: the picture is looked at, the samples are not: resolve only, on tile
        auto [f, st] = frameWith(true, false);
        const Attachment& a = Synth::rp(f, 0).desc.colors[0];
        CHECK(a.store == StoreAction::Resolve && a.resolve == one);
        CHECK(st.samplesDropped >= 1);
        CompiledFrame cf = compileFrame(f, &s.res);
        CHECK(cf.encoders.size() == 1 && cf.encoders[0].tileLocal(0) && cf.encoders[0].tileLocal(kDepthSlot));
        CHECK(cf.encoders[0].pass.colors[0].store == StoreAction::Resolve);
        CHECK(cf.encoders[0].pass.depth.store == StoreAction::DontCare);
    }
    {  // a shader reads the samples (texture2d_ms): kept as well
        auto [f, st] = frameWith(true, true);
        CHECK(Synth::rp(f, 0).desc.colors[0].store == StoreAction::StoreAndResolve);
        CompiledFrame cf = compileFrame(f, &s.res);
        CHECK(cf.encoders.size() == 1);
        // (explicit store of samples: the compiler keeps them in memory)
        CHECK(!cf.encoders[0].tileLocal(0));
    }
    {  // only the samples are read: nobody looks at the picture, the resolve is not done
        for (int n = 0; n < 4; ++n) frameWith(false, true);
        auto [f, st] = frameWith(false, true);
        const Attachment& a = Synth::rp(f, 0).desc.colors[0];
        CHECK(a.store == StoreAction::Store && !a.resolve);
    }
    {  // nobody reads either: the pass stores nothing
        for (int n = 0; n < 4; ++n) frameWith(false, false);
        auto [f, st] = frameWith(false, false);
        const Attachment& a = Synth::rp(f, 0).desc.colors[0];
        CHECK(a.store == StoreAction::DontCare && !a.resolve);
    }
}

void lifetimeResolveDeferred() {
    Synth s;
    const TextureId ms = s.color(4), one = s.color(1), other = s.color(1);
    // warm everything up with the same shape of frame
    for (int n = 0; n < 6; ++n) {
        s.life.noteRead(one, s.frame);
        Frame f;
        f.beginRenderPass(Synth::pass(ms, {}, LoadAction::Clear, LoadAction::Clear, one));
        f.draw(Draw{});
        f.endRenderPass();
        f.beginRenderPass(Synth::pass(other, {}, LoadAction::Clear, LoadAction::Clear));
        f.draw(Draw{});
        f.endRenderPass();
        f.beginRenderPass(Synth::pass(ms, {}, LoadAction::Load, LoadAction::Load, one));  // drawn into again
        f.draw(Draw{});
        f.endRenderPass();
        const auto st = s.run(f);
        if (n >= 4) {
            // the first pass leaves its samples for the last one and does not resolve
            const Attachment& first = Synth::rp(f, 0).desc.colors[0];
            const Attachment& last = Synth::rp(f, 2).desc.colors[0];
            CHECK(first.store == StoreAction::Store && !first.resolve);
            CHECK(last.store == StoreAction::Resolve && last.resolve == one);
            CHECK(st.resolvesDeferred == 1);
        }
    }
    // the picture is read between the two passes: the first one resolves
    Synth t;
    const TextureId ms2 = t.color(4), one2 = t.color(1);
    for (int n = 0; n < 6; ++n) {
        t.life.noteRead(one2, t.frame);
        Frame f;
        f.beginRenderPass(Synth::pass(ms2, {}, LoadAction::Clear, LoadAction::Clear, one2));
        f.draw(Draw{});
        f.endRenderPass();
        RenderPassDesc mid;
        mid.colors[0].texture = t.color();
        mid.colors[0].load = LoadAction::Clear;
        f.beginRenderPass(mid);
        f.draw(Synth::drawReading(one2));
        f.endRenderPass();
        f.beginRenderPass(Synth::pass(ms2, {}, LoadAction::Load, LoadAction::Load, one2));
        f.draw(Draw{});
        f.endRenderPass();
        t.run(f);
        if (n >= 4) {
            const Attachment& first = Synth::rp(f, 0).desc.colors[0];
            CHECK(first.resolve == one2 && first.store == StoreAction::StoreAndResolve);
        }
    }
}

// A read inside the next pass of the multisampled texture comes before that pass's resolve: the
// companion must be resolved by the earlier pass.
void lifetimeReadInNextPass() {
    Synth s;
    const TextureId ms = s.color(4), one = s.color(1);
    for (int n = 0; n < 8; ++n) {
        s.life.noteRead(one, s.frame);
        Frame f;
        f.beginRenderPass(Synth::pass(ms, {}, LoadAction::Clear, LoadAction::Clear, one));
        f.draw(Draw{});
        f.endRenderPass();
        f.beginRenderPass(Synth::pass(ms, {}, LoadAction::Load, LoadAction::Load, one));
        f.draw(Synth::drawReading(one));  // reads the companion while the pass is open
        f.endRenderPass();
        s.run(f);
        if (n >= 5) {
            const Attachment& first = Synth::rp(f, 0).desc.colors[0];
            CHECK(first.resolve == one && (first.store == StoreAction::StoreAndResolve || first.store == StoreAction::Resolve));
        }
    }
}

// One pass per layer (shadow faces, separate eyes): a clear of another layer is not an overwrite.
void lifetimeLayers() {
    Synth s;
    TextureDesc d;
    d.format = Format::D32Float;
    d.width = d.height = 64;
    d.layers = 2;
    d.usage = usage::sampled | usage::depthTarget;
    const TextureId arr = s.res.createTexture(d);
    const TextureId c = s.color();
    for (int n = 0; n < 12; ++n) {
        s.life.noteRead(c, s.frame);
        Frame f;
        for (uint16_t layer = 0; layer < 2; ++layer) {
            RenderPassDesc p;
            p.depth.texture = arr;
            p.depth.layer = layer;
            p.depth.load = LoadAction::Clear;
            p.depth.store = StoreAction::Store;
            f.beginRenderPass(p);
            f.draw(Draw{});
            f.endRenderPass();
        }
        f.beginRenderPass(Synth::pass(c, {}, LoadAction::Clear, LoadAction::Clear));
        f.draw(Synth::drawReading(arr));  // the main pass samples both layers
        f.endRenderPass();
        s.run(f);
        // both faces are read later in the frame: neither store may go
        CHECK(Synth::rp(f, 0).desc.depth.store == StoreAction::Store);
        CHECK(Synth::rp(f, 1).desc.depth.store == StoreAction::Store);
    }
    // nobody reads it: every layer's store goes after the warm-up
    Synth t;
    const TextureId arr2 = t.res.createTexture(d);
    for (int n = 0; n < 12; ++n) {
        Frame f;
        for (uint16_t layer = 0; layer < 2; ++layer) {
            RenderPassDesc p;
            p.depth.texture = arr2;
            p.depth.layer = layer;
            p.depth.load = LoadAction::Clear;
            p.depth.store = StoreAction::Store;
            f.beginRenderPass(p);
            f.draw(Draw{});
            f.endRenderPass();
        }
        t.run(f);
        if (n >= 6) {
            CHECK(Synth::rp(f, 0).desc.depth.store == StoreAction::DontCare);
            CHECK(Synth::rp(f, 1).desc.depth.store == StoreAction::DontCare);
        }
    }
}

// The companion of a multisampled target is never an attachment: a dropped resolve leaves it stale and
// reading it before the next resolve (a CPU read, a retile) must turn it sticky.
void lifetimeTwinLost() {
    Synth s;
    const TextureId ms = s.color(4), one = s.color(1);
    for (int n = 0; n < 8; ++n) {
        Frame f;
        f.beginRenderPass(Synth::pass(ms, {}, LoadAction::Clear, LoadAction::Clear, one));
        f.draw(Draw{});
        f.endRenderPass();
        s.run(f);
    }
    CHECK(s.life.contentsLost(one));  // resolve dropped: the companion is stale
    {   // a pass that reads the companion before the resolving pass
        Frame f;
        RenderPassDesc mid;
        mid.colors[0].texture = s.color();
        mid.colors[0].load = LoadAction::Clear;
        f.beginRenderPass(mid);
        f.draw(Synth::drawReading(one));
        f.endRenderPass();
        f.beginRenderPass(Synth::pass(ms, {}, LoadAction::Clear, LoadAction::Clear, one));
        f.draw(Draw{});
        f.endRenderPass();
        const auto st = s.run(f);
        CHECK(st.misses == 1 && s.life.sticky(one));
        CHECK(Synth::rp(f, 1).desc.colors[0].resolve == one);
    }
    // destroyed textures leave no entry behind
    const TextureId gone = s.color();
    s.life.noteSticky(gone, s.frame);
    CHECK(s.life.sticky(gone));
    s.res.destroy(gone);
    for (int n = 0; n < 130; ++n) {
        Frame f;
        s.run(f);
    }
    CHECK(!s.life.sticky(gone));
}

// ---- The lifter: multisampled loads and FMASK -----------------------------------------------------------

std::vector<uint32_t> hexWords(const std::string& path) {
    std::ifstream f(path);
    std::vector<uint32_t> w;
    for (std::string line; std::getline(f, line);) {
        if (line.rfind("code", 0) != 0) continue;
        std::istringstream ls(line.substr(4));
        for (std::string t; ls >> t;) w.push_back(uint32_t(std::stoul(t, nullptr, 16)));
    }
    return w;
}

void liftMsaa() {
    const std::vector<uint32_t> code = hexWords(std::string(VPPS4_MSAA_LIFT_DIR) + "/ps_msaa.lift");
    CHECK(!code.empty());
    auto lift = [&](bool multisample, bool fmaskAware) {
        vpgcn::LiftInput in;
        in.stage = vpgcn::ShaderStage::Pixel;
        in.code = code;
        in.userSgprs = 2;
        in.describe = [&](const vpgcn::DescriptorSource& s) -> std::optional<vpgcn::DescriptorInfo> {
            vpgcn::DescriptorInfo i;
            const uint32_t at = s.path.empty() ? 0 : s.path.back();
            if (at == 0x20 && fmaskAware) {
                i.fmask = true;
                i.fmaskLo = 0x3210;
            } else if (at == 0x0) {
                i.multisample = multisample;
            }
            return i;
        };
        return vpgcn::lift(in);
    };
    {
        vpgcn::LiftResult r = lift(true, true);
        CHECK(r.errors.empty());
        // one texture (the MS colour image); the FMASK binds nothing
        size_t textures = 0, ms = 0;
        for (const auto& res : r.module.resources) {
            textures += res.kind == vp4ir::ResourceKind::Texture;
            ms += res.kind == vp4ir::ResourceKind::Texture && res.multisample;
        }
        CHECK(textures == 1 && ms == 1);
        vp4msl::EmitResult e = vp4msl::emitMsl(r.module);
        CHECK(e.errors.empty());
        CHECK(e.shader.source.find("texture2d_ms<float>") != std::string::npos);
        CHECK(e.shader.source.find(".read(") != std::string::npos);
        // the constant answers the FMASK load: the fragment of sample 0 is nibble 0 of 0x3210
        CHECK(e.shader.source.find("12816") != std::string::npos || e.shader.source.find("0x3210") != std::string::npos ||
              e.shader.source.find("3210u") != std::string::npos);
    }
    {  // the host draws the surface at 1 sample: a plain texture, the sample index is dropped
        vpgcn::LiftResult r = lift(false, true);
        CHECK(r.errors.empty());
        vp4msl::EmitResult e = vp4msl::emitMsl(r.module);
        CHECK(e.errors.empty());
        CHECK(e.shader.source.find("texture2d_ms") == std::string::npos);
        CHECK(e.shader.source.find("texture2d<float>") != std::string::npos);
    }
    {  // without describe the FMASK is an ordinary (unbound-able) texture: two textures
        vpgcn::LiftInput in;
        in.stage = vpgcn::ShaderStage::Pixel;
        in.code = code;
        in.userSgprs = 2;
        vpgcn::LiftResult r = vpgcn::lift(in);
        size_t textures = 0;
        for (const auto& res : r.module.resources) textures += res.kind == vp4ir::ResourceKind::Texture;
        CHECK(textures == 2);
    }
}


// ---- The renderer on a capturing backend ----------------------------------------------------------------

class CaptureBackend final : public Backend {
public:
    void createBuffer(BufferId, const BufferDesc&) override {}
    void createTexture(TextureId id, const TextureDesc& d) override { textures.push_back({id, d}); }
    void createSampler(SamplerId, const SamplerDesc&) override {}
    void destroyBuffer(BufferId) override {}
    void destroyTexture(TextureId) override {}
    void destroySampler(SamplerId) override {}
    bool compilePipeline(const PipelineState&, std::string&) override { return true; }
    void submit(uint64_t, const Frame& f, const CompiledFrame& c) override {
        frames.push_back(f);
        compiled.push_back(c);
    }
    bool takeTimings(uint64_t&, GpuTimings&) override { return false; }
    uint64_t completedFrames() const override { return ~uint64_t(0); }

    std::vector<std::pair<TextureId, TextureDesc>> textures;
    std::vector<Frame> frames;
    std::vector<CompiledFrame> compiled;
};

constexpr uint64_t kBase = 0x10000000, kSize = 1 << 20;

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

// ---- Foveation ---------------------------------------------------------------------------------------

void foveationMap() {
    using vpps4::makeFoveationMap;
    auto m = makeFoveationMap(1440, 1536, 2);
    CHECK(m && m->layers == 2 && m->width == 1440 && m->height == 1536);
    CHECK(m->zonesX == 17 && m->qx.size() == 34 && m->qy.size() == 34);
    // 1 at the centre, symmetrical, never below the floor, monotonic towards the edge
    const float* q = m->qx.data();
    CHECK(q[8] == 1.0f || q[8] > 0.99f);
    for (int i = 0; i < 17; ++i) {
        CHECK(std::abs(q[i] - q[16 - i]) < 1e-6f);
        CHECK(q[i] >= 0.25f && q[i] <= 1.0f);
        if (i < 8) CHECK(q[i] <= q[i + 1]);
    }
    CHECK(q[0] < 0.5f);
    // both layers alike; the estimate is the mean product; the key follows the shape
    for (int i = 0; i < 17; ++i) CHECK(m->qx[17 + i] == q[i] && m->qy[i] == q[i]);
    CHECK(m->keptFraction > 0.25f * 0.25f && m->keptFraction < 1.0f);
    CHECK(makeFoveationMap(1440, 1536, 2)->key == m->key);
    CHECK(makeFoveationMap(1440, 1537, 2)->key != m->key);
    // not served
    CHECK(!makeFoveationMap(1440, 1536, 1));
    CHECK(!makeFoveationMap(8, 8, 2));
    vpps4::FoveationParams off;
    off.strength = 0;
    auto flat = makeFoveationMap(1440, 1536, 2, off);
    CHECK(flat && flat->keptFraction == 1.0f);
}

void liftViewIndexed() {
    const std::vector<uint32_t> vs = hexWords(std::string(VPPS4_MSAA_LIFT_DIR) + "/vs_layer.lift");
    CHECK(!vs.empty());
    for (bool view : {false, true}) {
        vpgcn::LiftInput in;
        in.stage = vpgcn::ShaderStage::Vertex;
        in.code = vs;
        in.viewIndexed = view;
        vpgcn::LiftResult r = vpgcn::lift(in);
        CHECK(r.errors.empty());
        vp4msl::EmitResult e = vp4msl::emitMsl(r.module);
        CHECK(e.errors.empty());
        CHECK((e.shader.source.find("[[viewport_array_index]]") != std::string::npos) == view);
        CHECK((e.shader.source.find("[[render_target_array_index]]") != std::string::npos) == view);
    }
    // the pixel side reads FragCoord through the rate map and takes the layer
    const Program ps = load("ps_textured");
    for (bool view : {false, true}) {
        vpgcn::LiftInput in;
        in.stage = vpgcn::ShaderStage::Pixel;
        in.code = ps.code;
        in.userSgprs = 2;
        in.viewIndexed = view;
        vpgcn::LiftResult r = vpgcn::lift(in);
        CHECK(r.errors.empty());
        vp4msl::EmitResult e = vp4msl::emitMsl(r.module);
        CHECK(e.errors.empty());
        CHECK((e.shader.source.find("rasterization_rate_map_data") != std::string::npos) == view);
    }
}

struct Fixture {
    void* host = std::aligned_alloc(16384, kSize);
    CaptureBackend backend;
    std::vector<std::string> log;
    Device device{backend, 0, [this](const std::string& s) { log.push_back(s); }};
    vpps4::Renderer renderer{device, hooks()};
    Program vs = load("vs_fetch"), ps = load("ps_textured");

    Fixture() {
        std::memset(host, 0, kSize);
        device.setHoldMs(0);
    }
    ~Fixture() { std::free(host); }

    vpps4::Hooks hooks() {
        vpps4::Hooks h;
        h.region = [this](uint64_t a, uint64_t s, vpps4::GuestRegion& g) {
            if (a < kBase || a + s > kBase + kSize) return false;
            g = {kBase, kSize, host};
            return true;
        };
        h.shader = [](uint64_t, vp4ir::Stage, const vp4msl::MslShader&) {};
        h.log = [this](const std::string& s) { log.push_back(s); };
        return h;
    }
    uint32_t* at(uint64_t guest) { return reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(host) + (guest - kBase)); }

    vpps4::DrawState state() {
        uint32_t* vtable = at(kBase + 0x100);
        const uint64_t vb = kBase + 0x1000;
        vtable[0] = uint32_t(vb);
        vtable[1] = uint32_t(vb >> 32) | (16u << 16);
        vtable[2] = 64;
        vtable[3] = (7u << 12) | (14u << 15);
        uint32_t* ttable = at(kBase + 0x200);
        const uint64_t tex = kBase + 0x10000;
        ttable[0] = uint32_t(tex >> 8);
        ttable[1] = uint32_t(tex >> 40) | (10u << 20) | (0u << 26);
        ttable[2] = 63 | (63u << 14);
        ttable[3] = 4 | (5u << 3) | (6u << 6) | (7u << 9) | (8u << 20) | (9u << 28);
        ttable[4] = 63u << 13;
        ttable[8] = 0;
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
        d.ps.exportFormat[0] = 4;
        return d;
    }
    const TextureDesc* desc(TextureId t) const { return device.resources().get(t); }
};

// One frame like a title's: clear colour (fast clear) and depth, draw, flip the colour. The first
// submitted frames warm up; then the multisampled colour resolves into its picture on tile.
void rendererMsaa() {
    Fixture f;
    f.renderer.setMaxSamples(4);
    constexpr int kFrames = 8;
    for (int n = 0; n < kFrames; ++n) {
        vpps4::DrawState clear = f.state();
        clear.primitive = vpps4::PrimitiveType::RectList;
        clear.color[0].samples = 4;
        clear.color[0].fastClear = true;
        clear.color[0].clearWord = {0xff204080u, 0};
        clear.depth = {kBase + 0x60000, 0, 128, 128, 3, 0, 0, 0, 1, 4, true, false, 1.0f, 0};
        f.renderer.draw(clear);
        vpps4::DrawState d = f.state();
        d.color[0].samples = 4;
        d.depth = clear.depth;
        d.depth.depthClear = false;
        d.depthStencil.depthTest = d.depthStencil.depthWrite = true;
        d.depthStencil.depthFunc = 1;
        f.renderer.draw(d);
        f.renderer.draw(d);
        f.renderer.submit();
        (void)f.renderer.colorTargetAt(kBase + 0x40000);  // the flip
    }
    CHECK(f.renderer.stats().shaderErrors == 0);
    CHECK(f.renderer.stats().drawn == 2 * kFrames);
    CHECK(f.backend.compiled.size() == size_t(kFrames));
    // Every frame is one render encoder: the clears are load actions, not passes of their own.
    for (const CompiledFrame& c : f.backend.compiled) {
        size_t render = 0;
        for (const Encoder& e : c.encoders) render += e.kind == Encoder::Kind::Render;
        CHECK(render == 1);
    }
    const CompiledFrame& last = f.backend.compiled.back();
    CHECK(last.encoders.size() == 1);
    const Encoder& e = last.encoders[0];
    const Attachment& color = e.pass.colors[0];
    const TextureDesc* ms = f.desc(color.texture);
    CHECK(ms && ms->samples == 4 && !ms->persistent);
    CHECK(color.load == LoadAction::Clear);
    CHECK(std::abs(color.clear.color[0] - 128.0 / 255) < 1e-9 && std::abs(color.clear.color[2] - 32.0 / 255) < 1e-9);
    CHECK(color.store == StoreAction::Resolve);
    const TextureDesc* one = f.desc(color.resolve);
    CHECK(one && one->samples == 1 && (one->usage & usage::storage) == 0);
    CHECK(e.tileLocal(0));
    CHECK(e.pass.depth.load == LoadAction::Clear);
    CHECK(e.pass.depth.store == StoreAction::DontCare);
    CHECK(e.tileLocal(kDepthSlot));
    // the first frames are stored in full while the planner warms up
    const CompiledFrame& warm = f.backend.compiled.front();
    for (const Encoder& first : warm.encoders)
        if (first.kind == Encoder::Kind::Render) {
            CHECK(first.pass.colors[0].store == StoreAction::StoreAndResolve);
            CHECK(!first.tileLocal(0));
        }
    if (failures)
        for (const std::string& l : f.log) std::fprintf(stderr, "  %s\n", l.c_str());
}

// MSAA off in the app: single-sample targets, same planner (depth dropped, colour kept: it is shown).
void rendererOneSample() {
    Fixture f;
    f.renderer.setMaxSamples(1);
    for (int n = 0; n < 8; ++n) {
        vpps4::DrawState clear = f.state();
        clear.primitive = vpps4::PrimitiveType::RectList;
        clear.depth = {kBase + 0x60000, 0, 128, 128, 3, 0, 0, 0, 1, 1, true, false, 1.0f, 0};
        f.renderer.draw(clear);
        vpps4::DrawState d = f.state();
        d.depth = clear.depth;
        d.depth.depthClear = false;
        d.depthStencil.depthTest = d.depthStencil.depthWrite = true;
        d.depthStencil.depthFunc = 1;
        f.renderer.draw(d);
        f.renderer.submit();
        (void)f.renderer.colorTargetAt(kBase + 0x40000);
    }
    const Encoder& e = f.backend.compiled.back().encoders[0];
    CHECK(e.pass.colors[0].store == StoreAction::Store);
    CHECK(e.pass.depth.store == StoreAction::DontCare && e.pass.depth.load == LoadAction::Clear);
    CHECK(e.tileLocal(kDepthSlot));
    CHECK(f.renderer.stats().skipped.count("RectList") == 0);
}


// Two-slice eye targets: with the option on the pass carries the rate map and both layers; off, nothing
// changes; a foveated pass keeps its stores in memory (the shadow textures need it).
void rendererFoveation() {
    for (bool on : {false, true}) {
        Fixture f;
        f.renderer.setMaxSamples(1);
        f.renderer.setFoveation(on);
        for (int n = 0; n < 6; ++n) {
            vpps4::DrawState d = f.state();
            d.color[0].sliceStart = 0;
            d.color[0].sliceCount = 2;
            d.depth = {kBase + 0x60000, 0, 128, 128, 3, 0, 0, 0, 2, 1, true, false, 1.0f, 0};
            d.depthStencil.depthTest = d.depthStencil.depthWrite = true;
            f.renderer.draw(d);
            f.renderer.submit();
            (void)f.renderer.colorTargetAt(kBase + 0x40000);
        }
        CHECK(f.renderer.stats().drawn == 6);
        const Frame& fr = f.backend.frames.back();
        const RenderPass& rp = std::get<RenderPass>(fr.commands().front());
        CHECK((rp.desc.rateMap != nullptr) == on);
        CHECK(rp.desc.layerCount == (on ? 2 : 1));
        const Encoder& e = f.backend.compiled.back().encoders.back();
        if (on) {
            CHECK(rp.desc.rateMap->width == 128 && rp.desc.rateMap->layers == 2);
            CHECK(e.pass.depth.store == StoreAction::Store && !e.tileLocal(kDepthSlot));
        } else {
            CHECK(e.pass.depth.store == StoreAction::DontCare || e.pass.depth.store == StoreAction::Store);
        }
    }
}

// A compute pass that writes a render target makes it writable on its first use; before that no
// colour target carries storage usage.
void rendererWritable() {
    Fixture f;
    f.renderer.setMaxSamples(1);
    f.renderer.draw(f.state());
    for (const auto& entry : f.backend.textures) CHECK((entry.second.usage & usage::storage) == 0);
    const Program cs = load("cs_image_atomic");
    std::array<uint32_t, 8> t{};
    const uint64_t a = kBase + 0x40000;  // the colour target's address
    t[0] = uint32_t(a >> 8);
    t[1] = uint32_t(a >> 40) | (10u << 20);
    t[2] = 63 | (63u << 14);
    t[3] = 4 | (5u << 3) | (6u << 6) | (7u << 9) | (8u << 20) | (9u << 28);
    t[4] = 63u << 13;
    vpps4::DispatchState x;
    x.cs.code = cs.code;
    x.cs.address = kBase + 0x82000;
    x.cs.userSgprs = 16;
    for (int i = 0; i < 8; ++i) x.cs.userData[i] = t[i], x.cs.userData[8 + i] = t[i];
    f.renderer.dispatch(x);
    size_t writable = 0;
    for (const auto& [id, d] : f.backend.textures) writable += (d.usage & usage::storage) != 0 && d.label.find("0x") != std::string::npos;
    // the target was recreated with storage and its contents copied over
    CHECK(writable >= 1);
    CHECK(f.renderer.stats().notes.count("colour target recreated with storage usage (written by a compute pass)") == 1);
    f.renderer.submit();
    if (failures)
        for (const std::string& l : f.log) std::fprintf(stderr, "  %s\n", l.c_str());
}

// A resolve (CB_COLOR_CONTROL.MODE = RESOLVE) once the multisampled source's picture is no longer
// looked at: done by the pass that drew the samples, into CB1, with no extra pass or copy.
void rendererResolve() {
    Fixture f;
    f.renderer.setMaxSamples(4);
    vpps4::ColorTarget dst = f.state().color[0];
    dst.address = kBase + 0x70000;
    dst.samples = 1;
    for (int n = 0; n < 6; ++n) {
        vpps4::DrawState d = f.state();
        d.color[0].samples = 4;
        f.renderer.draw(d);
        f.renderer.resolve(d.color[0], dst);
        f.renderer.submit();
        (void)f.renderer.colorTargetAt(kBase + 0x70000);
    }
    CHECK(f.renderer.stats().resolves == 6);
    CHECK(f.renderer.stats().skipped.empty());
    const CompiledFrame& last = f.backend.compiled.back();
    size_t render = 0, compute = 0;
    for (const Encoder& e : last.encoders) (e.kind == Encoder::Kind::Render ? render : compute)++;
    CHECK(render == 1 && compute == 0);
    const Attachment& a = last.encoders[0].pass.colors[0];
    CHECK(a.resolve && f.desc(a.resolve)->label.find("0x10070000") != std::string::npos);
    CHECK(a.store == StoreAction::Resolve || a.store == StoreAction::StoreAndResolve);  // (samples too: nothing clears them here)
    CHECK(f.renderer.stats().notes.count("MSAA resolve done by the pass that drew the samples") >= 1);
    // the first frames, with a young single-sample picture, copy it
    CHECK(f.renderer.stats().copies >= 1 && f.renderer.stats().copies < 6);
    if (failures) {
        for (const auto& [k, v] : f.renderer.stats().notes) std::fprintf(stderr, "  note %s x%llu\n", k.c_str(), (unsigned long long)v);
        std::fprintf(stderr, "  copies=%llu passes=%llu\n", (unsigned long long)f.renderer.stats().copies, (unsigned long long)f.renderer.stats().passes);
    }
    if (failures)
        for (const std::string& l : f.log) std::fprintf(stderr, "  %s\n", l.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    const std::string t = argc > 1 ? argv[1] : "";
    if (t == "sample_plan") samplePlan();
    else if (t == "fmask") fmask();
    else if (t == "depth_resolve") depthResolve();
    else if (t == "color_resolve") colorResolve();
    else if (t == "fast_clear") fastClear();
    else if (t == "resolve_way") resolveWay();
    else if (t == "lifetime_depth") lifetimeDepth();
    else if (t == "lifetime_miss") lifetimeMiss();
    else if (t == "lifetime_sampled") lifetimeSampled();
    else if (t == "lifetime_msaa") lifetimeMsaa();
    else if (t == "lifetime_resolve_deferred") lifetimeResolveDeferred();
    else if (t == "lifetime_read_in_next_pass") lifetimeReadInNextPass();
    else if (t == "lifetime_layers") lifetimeLayers();
    else if (t == "lifetime_twin_lost") lifetimeTwinLost();
    else if (t == "lift_msaa") liftMsaa();
    else if (t == "foveation_map") foveationMap();
    else if (t == "lift_view_indexed") liftViewIndexed();
    else if (t == "renderer_msaa") rendererMsaa();
    else if (t == "renderer_one_sample") rendererOneSample();
    else if (t == "renderer_foveation") rendererFoveation();
    else if (t == "renderer_writable") rendererWritable();
    else if (t == "renderer_resolve") rendererResolve();
    else {
        std::fprintf(stderr, "unknown test '%s'\n", t.c_str());
        return 2;
    }
    if (failures) std::fprintf(stderr, "%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
