// gpu/core unit tests on the null backend. Usage: vpgpu_core_tests [group]  (no group = all).
#include <cstdio>
#include <cstring>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <functional>
#include <random>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#include "vpgpu/core/activity_gauge.h"
#include "vpgpu/core/background.h"
#include "vpgpu/core/device.h"
#include "vpgpu/core/perf.h"
#include "vpgpu/core/scoped_pool.h"

using namespace vpgpu;

namespace {

int failures = 0;
#define CHECK(cond)                                                     \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      ++failures;                                                       \
    }                                                                   \
  } while (0)
#define CHECK_EQ(a, b)                                                                     \
  do {                                                                                     \
    const auto va = (a);                                                                   \
    const auto vb = (b);                                                                   \
    if (!(va == vb)) {                                                                     \
      std::printf("  FAIL %s:%d: %s == %s (%lld vs %lld)\n", __FILE__, __LINE__, #a, #b,   \
                  (long long)va, (long long)vb);                                           \
      ++failures;                                                                          \
    }                                                                                      \
  } while (0)

// Small fixture: a resource table plus helpers that build commands tersely.
struct Scene {
  ResourceTable res;

  TextureId rt(uint32_t w = 256, uint16_t mips = 1, uint16_t layers = 1, Format f = Format::RGBA8Unorm) {
    TextureDesc d;
    d.format = f;
    d.width = w;
    d.height = w;
    d.mips = mips;
    d.layers = layers;
    d.usage = usage::sampled | (formatInfo(f).depth() ? usage::depthTarget : usage::colorTarget);
    return res.createTexture(d);
  }
  TextureId depthRt(uint32_t w = 256) { return rt(w, 1, 1, Format::D32Float); }
  BufferId buf(uint64_t size = 4096) { return res.createBuffer({size, nullptr, ""}); }

  static RenderPassDesc pass(TextureId color, TextureId depth = {}, LoadAction load = LoadAction::Load) {
    RenderPassDesc d;
    d.colors[0].texture = color;
    d.colors[0].load = load;
    d.depth.texture = depth;
    d.depth.load = load;
    return d;
  }
  static Draw draw(std::vector<Access> bindings = {}) {
    Draw d;
    d.pipeline = {1};
    d.count = 3;
    d.bindings = std::move(bindings);
    return d;
  }
  static void passWithDraw(Frame& f, RenderPassDesc d, std::vector<Access> bindings = {}) {
    f.beginRenderPass(std::move(d));
    f.draw(draw(std::move(bindings)));
    f.endRenderPass();
  }
  static Access sample(TextureId t, uint16_t mip = 0, uint16_t layer = 0) {
    return read(ResourceRange::subresource(t, mip, 1, layer, 1), stage::fragment);
  }
  static Access vsRead(BufferId b, uint64_t off = 0, uint64_t size = 256) {
    return read(ResourceRange::bytes(b, off, size), stage::vertex);
  }
  static BufferSpan span(BufferId b, uint64_t off = 0, uint64_t size = 256) { return {b, off, size, 0, 0}; }
  static Dispatch compute(std::vector<Access> bindings) {
    Dispatch d;
    d.pipeline = {2};
    d.bindings = std::move(bindings);
    return d;
  }
};

uint32_t cuts(const CompiledFrame& c, CutReason r) { return c.stats.cuts[static_cast<int>(r)]; }
uint32_t hazards(const CompiledFrame& c, HazardKind k) { return c.stats.hazards[static_cast<int>(k)]; }

// ---------------------------------------------------------------------------------------------

void testResources() {
  ResourceTable t;
  const BufferId b = t.createBuffer({64, nullptr, "b"});
  CHECK(b && t.get(b) && t.get(b)->size == 64);
  CHECK(!t.createBuffer({0, nullptr, ""}));
  CHECK(t.destroy(b));
  CHECK(!t.get(b));
  CHECK(!t.destroy(b));
  const BufferId b2 = t.createBuffer({32, nullptr, ""});
  CHECK_EQ(b2.index, b.index);  // slot reused...
  CHECK(!(b2 == b));             // ...with a new generation, so the stale handle stays dead
  CHECK(!t.get(b));

  TextureDesc d;
  d.format = Format::RGBA8Unorm;
  d.width = 256, d.height = 128, d.mips = 9;
  CHECK(validate(d));
  d.mips = 10;
  CHECK(!validate(d));  // more mips than log2(256)+1
  d.mips = 1;
  d.type = TextureType::Cube;
  d.layers = 6;
  CHECK(!validate(d));  // cube must be square
  d.height = 256;
  CHECK(validate(d));
  d.usage = usage::depthTarget;
  CHECK(!validate(d));  // color format as depth target
  d.format = Format::D24UnormS8;
  CHECK(validate(d));
  d.format = Format::BC7;
  d.usage = usage::colorTarget;
  CHECK(!validate(d));  // compressed formats cannot be rendered to
  CHECK(formatInfo(Format::BC1).blockSize == 4 && formatInfo(Format::BC1).bytesPerBlock == 8);
  CHECK(formatInfo(Format::D32FloatS8).depth() && formatInfo(Format::D32FloatS8).stencil());
  CHECK(formatInfo(Format::RGBA8Srgb).srgb() && formatInfo(Format::R32Uint).integer());
  CHECK(std::strcmp(formatInfo(Format::RG11B10Float).name, "RG11B10Float") == 0);
}

void testCommands() {
  Scene s;
  const BufferId b = s.buf();
  const auto r1 = ResourceRange::bytes(b, 0, 100), r2 = ResourceRange::bytes(b, 100, 100);
  CHECK(!r1.overlaps(r2));
  CHECK(r1.overlaps(ResourceRange::bytes(b, 99, 1)));
  CHECK(ResourceRange::wholeBuffer(b).contains(r2));
  CHECK(ResourceRange::bytes(b, UINT64_MAX - 1, 10).end == UINT64_MAX);  // saturates, no wrap
  const TextureId t = s.rt(256, 4, 2);
  const auto m0 = ResourceRange::subresource(t, 0, 1, 0, 2), m1 = ResourceRange::subresource(t, 1, 1, 0, 2);
  CHECK(!m0.overlaps(m1));
  CHECK(!ResourceRange::subresource(t, 0, 4, 0, 1).overlaps(ResourceRange::subresource(t, 0, 4, 1, 1)));
  CHECK(ResourceRange::wholeTexture(t).contains(m1));
  CHECK(!ResourceRange::wholeTexture(t).overlaps(ResourceRange::wholeBuffer(b)));  // kinds differ

  Frame f;
  CHECK(!f.draw(Scene::draw()));  // no pass open
  f.beginRenderPass(Scene::pass(t));
  CHECK(f.draw(Scene::draw()));
  f.fillBuffer(Scene::span(b), 0);
  CHECK(!f.draw(Scene::draw()));  // a copy closes the pass
  f.clear(Scene::pass(t));
  CHECK_EQ(f.commands().size(), size_t{3});
  CHECK(std::get<RenderPass>(f.commands()[2]).desc.colors[0].load == LoadAction::Clear);
}

void testMerge() {
  Scene s;
  const TextureId c = s.rt(), d = s.depthRt(), c2 = s.rt();
  {  // three passes on the same targets -> one encoder, loads of first, stores of last
    Frame f;
    auto first = Scene::pass(c, d, LoadAction::Clear);
    first.colors[0].clear.color = {0.5f, 0, 0, 1};
    Scene::passWithDraw(f, first);
    Scene::passWithDraw(f, Scene::pass(c, d));
    auto last = Scene::pass(c, d);
    last.depth.store = StoreAction::DontCare;
    Scene::passWithDraw(f, last);
    const CompiledFrame cf = compileFrame(f);
    CHECK_EQ(cf.encoders.size(), size_t{1});
    CHECK_EQ(cf.encoders[0].passes, 3u);
    CHECK_EQ(cf.stats.passesMerged, 2u);
    CHECK(cf.encoders[0].pass.colors[0].load == LoadAction::Clear);
    CHECK(cf.encoders[0].pass.colors[0].clear.color[0] == 0.5f);
    CHECK(cf.encoders[0].pass.depth.store == StoreAction::DontCare);
    CHECK(cf.barriers.empty());  // attachment writes inside one encoder need no barrier
  }
  {  // different attachments, different mip, different layer count -> no merge
    const TextureId m = s.rt(256, 2);
    Frame f;
    Scene::passWithDraw(f, Scene::pass(c));
    Scene::passWithDraw(f, Scene::pass(c2));
    auto mip1 = Scene::pass(m);
    mip1.colors[0].mip = 1;
    Scene::passWithDraw(f, Scene::pass(m));
    Scene::passWithDraw(f, mip1);
    const CompiledFrame cf = compileFrame(f);
    CHECK_EQ(cf.encoders.size(), size_t{4});
    CHECK_EQ(cuts(cf, CutReason::AttachmentChange), 3u);
  }
  {  // a clear on the same targets cannot merge (needs a fresh load action) but a clear pass
     // followed by draws on the same targets merges into it
    Frame f;
    Scene::passWithDraw(f, Scene::pass(c));
    f.clear(Scene::pass(c));
    Scene::passWithDraw(f, Scene::pass(c));
    const CompiledFrame cf = compileFrame(f);
    CHECK_EQ(cf.encoders.size(), size_t{2});
    CHECK_EQ(cuts(cf, CutReason::ClearLoad), 1u);
    CHECK_EQ(cf.encoders[1].passes, 2u);
  }
}

void testCopies() {
  Scene s;
  const TextureId c = s.rt();
  const BufferId ubo = s.buf(), staging = s.buf(), vb = s.buf();
  {  // independent copy between passes on the same target: hoisted, passes stay merged
    Frame f;
    Scene::passWithDraw(f, Scene::pass(c), {Scene::vsRead(ubo, 0)});
    f.copyBuffer(Scene::span(staging), Scene::span(ubo, 256));  // next draw's constants
    Scene::passWithDraw(f, Scene::pass(c), {Scene::vsRead(ubo, 256)});
    const CompiledFrame cf = compileFrame(f);
    CHECK_EQ(cf.encoders.size(), size_t{2});
    CHECK(cf.encoders[0].kind == Encoder::Kind::Compute && cf.encoders[1].kind == Encoder::Kind::Render);
    CHECK_EQ(cf.encoders[1].passes, 2u);
    CHECK_EQ(cf.stats.copiesHoisted, 1u);
    CHECK_EQ(cf.barriers.size(), size_t{1});  // copy write -> vertex read of ubo[256..512)
    CHECK(cf.barriers[0].encoder == 1 && cf.barriers[0].after == stage::transfer);
  }
  {  // copy overwrites what the first pass read (WAR): cannot hoist, sunk after the encoder
    Frame f;
    Scene::passWithDraw(f, Scene::pass(c), {Scene::vsRead(ubo, 0)});
    f.copyBuffer(Scene::span(staging), Scene::span(ubo, 0));
    const CompiledFrame cf = compileFrame(f);
    CHECK_EQ(cf.encoders.size(), size_t{2});
    CHECK(cf.encoders[0].kind == Encoder::Kind::Render && cf.encoders[1].kind == Encoder::Kind::Compute);
    CHECK_EQ(cf.stats.copiesSunk, 1u);
    CHECK_EQ(hazards(cf, HazardKind::WAR), 1u);
  }
  {  // sunk copy that a later pass on the same target reads: that pass is pinned behind it
    Frame f;
    Scene::passWithDraw(f, Scene::pass(c), {Scene::vsRead(vb, 0)});
    f.copyBuffer(Scene::span(staging), Scene::span(vb, 0));
    Scene::passWithDraw(f, Scene::pass(c), {Scene::vsRead(vb, 0)});
    const CompiledFrame cf = compileFrame(f);
    CHECK_EQ(cf.encoders.size(), size_t{3});  // render, copy, render
    CHECK_EQ(cuts(cf, CutReason::PinnedCopy), 1u);
    CHECK_EQ(hazards(cf, HazardKind::WAR), 1u);
    CHECK_EQ(hazards(cf, HazardKind::RAW), 1u);
  }
  {  // sunk copy that a later pass does NOT touch: the pass merges, the copy runs after both
    Frame f;
    Scene::passWithDraw(f, Scene::pass(c), {Scene::vsRead(vb, 0)});
    f.copyBuffer(Scene::span(staging), Scene::span(vb, 0));
    Scene::passWithDraw(f, Scene::pass(c), {Scene::vsRead(ubo, 0)});
    const CompiledFrame cf = compileFrame(f);
    CHECK_EQ(cf.encoders.size(), size_t{2});
    CHECK_EQ(cf.encoders[0].passes, 2u);
  }
  {  // copy reading the render target just drawn (RAW on the attachment): must sink
    Frame f;
    Scene::passWithDraw(f, Scene::pass(c));
    Copy cp;
    cp.kind = CopyKind::TextureToBuffer;
    cp.srcTexture = {c, 0, 0, 1, 0, 0, 0, 256, 256, 1};
    cp.dstBuffer = Scene::span(staging, 0, 256 * 256 * 4);
    f.copy(cp);
    const CompiledFrame cf = compileFrame(f);
    CHECK_EQ(cf.stats.copiesSunk, 1u);
    CHECK_EQ(hazards(cf, HazardKind::RAW), 1u);
    CHECK(cf.barriers.size() == 1 && cf.barriers[0].after == stage::fragment && cf.barriers[0].before == stage::transfer);
  }
  {  // a later copy that conflicts with a sunk one must not be hoisted past it
    Frame f;
    Scene::passWithDraw(f, Scene::pass(c), {Scene::vsRead(vb, 0)});
    f.copyBuffer(Scene::span(staging, 0), Scene::span(vb, 0));    // WAR on vb -> sunk
    f.copyBuffer(Scene::span(vb, 0), Scene::span(ubo, 512));      // reads vb after that write
    const CompiledFrame cf = compileFrame(f);
    CHECK_EQ(cf.stats.copiesSunk, 2u);
    CHECK_EQ(cf.encoders.back().commands.size(), size_t{2});
    CHECK_EQ(cf.stats.intraBarriers, 1u);  // RAW between the two copies inside one encoder
  }
  {  // copies outside a render encoder batch into one compute encoder; consecutive ones without
     // hazards need no barrier
    Frame f;
    f.copyBuffer(Scene::span(staging, 0), Scene::span(ubo, 0));
    f.fillBuffer(Scene::span(vb, 0), 0);
    f.copyBuffer(Scene::span(staging, 512), Scene::span(ubo, 512));
    const CompiledFrame cf = compileFrame(f);
    CHECK_EQ(cf.encoders.size(), size_t{1});
    CHECK(cf.barriers.empty());
    CHECK_EQ(cf.stats.copiesHoisted + cf.stats.copiesSunk, 0u);
  }
}

void testHazards() {
  Scene s;
  const TextureId a = s.rt(), b = s.rt();
  const BufferId buf = s.buf();
  {  // RAW: render to a, then sample a
    Frame f;
    Scene::passWithDraw(f, Scene::pass(a));
    Scene::passWithDraw(f, Scene::pass(b), {Scene::sample(a)});
    const CompiledFrame cf = compileFrame(f);
    CHECK_EQ(cf.barriers.size(), size_t{1});
    CHECK_EQ(hazards(cf, HazardKind::RAW), 1u);
    CHECK(cf.barriers[0].encoder == 1 && cf.barriers[0].item == 0);
    CHECK(cf.barriers[0].after == stage::fragment && cf.barriers[0].before == stage::fragment);
  }
  {  // WAR: sample a, then render to a
    Frame f;
    Scene::passWithDraw(f, Scene::pass(b), {Scene::sample(a)});
    Scene::passWithDraw(f, Scene::pass(a));
    const CompiledFrame cf = compileFrame(f);
    CHECK_EQ(hazards(cf, HazardKind::WAR), 1u);
    CHECK_EQ(hazards(cf, HazardKind::RAW) + hazards(cf, HazardKind::WAW), 0u);
  }
  {  // WAW: compute writes buf, a copy overwrites it
    Frame f;
    f.dispatch(Scene::compute({write(ResourceRange::bytes(buf, 0, 256), stage::compute)}));
    f.fillBuffer(Scene::span(buf, 0), 7);
    const CompiledFrame cf = compileFrame(f);
    CHECK_EQ(cf.encoders.size(), size_t{1});  // copies and dispatches share compute encoders
    CHECK_EQ(hazards(cf, HazardKind::WAW), 1u);
    CHECK(cf.barriers.size() == 1 && cf.barriers[0].item == 1);
  }
  {  // read after read: never a barrier
    Frame f;
    Scene::passWithDraw(f, Scene::pass(a), {Scene::vsRead(buf)});
    f.dispatch(Scene::compute({read(ResourceRange::bytes(buf, 0, 256), stage::compute)}));
    Scene::passWithDraw(f, Scene::pass(b), {Scene::vsRead(buf)});
    CHECK(compileFrame(f).barriers.empty());
  }
  {  // one barrier covers a hazard for later commands: write -> barrier -> read, read
    Frame f;
    f.dispatch(Scene::compute({write(ResourceRange::bytes(buf, 0, 256), stage::compute)}));
    f.dispatch(Scene::compute({read(ResourceRange::bytes(buf, 0, 256), stage::compute)}));
    f.dispatch(Scene::compute({read(ResourceRange::bytes(buf, 0, 256), stage::compute)}));
    const CompiledFrame cf = compileFrame(f);
    CHECK_EQ(cf.barriers.size(), size_t{1});
  }
  {  // a barrier with too few stages does not cover a later reader in another stage
    Frame f;
    f.dispatch(Scene::compute({write(ResourceRange::bytes(buf, 0, 256), stage::compute)}));
    f.dispatch(Scene::compute({read(ResourceRange::bytes(buf, 0, 256), stage::compute)}));
    Scene::passWithDraw(f, Scene::pass(a), {Scene::vsRead(buf)});
    const CompiledFrame cf = compileFrame(f);
    CHECK_EQ(cf.barriers.size(), size_t{2});
    CHECK(cf.barriers[1].after == stage::compute && cf.barriers[1].before == stage::vertex);
  }
  {  // multiple hazards at one point merge into one barrier
    Frame f;
    f.dispatch(Scene::compute({write(ResourceRange::bytes(buf, 0, 256), stage::compute)}));
    Scene::passWithDraw(f, Scene::pass(a));
    Scene::passWithDraw(f, Scene::pass(b), {Scene::sample(a), Scene::vsRead(buf)});
    const CompiledFrame cf = compileFrame(f);
    CHECK_EQ(cf.barriers.size(), size_t{1});
    CHECK(cf.barriers[0].after == (stage::compute | stage::fragment));
  }
}

void testSubresources() {
  Scene s;
  const TextureId chain = s.rt(256, 4), arr = s.rt(256, 1, 2);
  const BufferId buf = s.buf();
  {  // mip chain downsample: each pass samples mip i and writes mip i+1 -> RAW only on the pair
    Frame f;
    for (uint16_t m = 0; m + 1 < 4; ++m) {
      auto d = Scene::pass(chain);
      d.colors[0].mip = m + 1;
      Scene::passWithDraw(f, d, {Scene::sample(chain, m)});
    }
    const CompiledFrame cf = compileFrame(f);
    CHECK_EQ(cf.encoders.size(), size_t{3});
    CHECK_EQ(cf.barriers.size(), size_t{2});  // first pass reads mip 0, never written this frame
    CHECK_EQ(hazards(cf, HazardKind::RAW), 2u);
    CHECK_EQ(hazards(cf, HazardKind::WAR), 0u);
  }
  {  // layers are independent
    Frame f;
    auto l1 = Scene::pass(arr);
    l1.colors[0].layer = 1;
    Scene::passWithDraw(f, Scene::pass(arr));
    Scene::passWithDraw(f, l1, {Scene::sample(arr, 0, 0)});  // reads layer 0 while writing 1
    const CompiledFrame cf = compileFrame(f);
    CHECK_EQ(hazards(cf, HazardKind::RAW), 1u);  // layer 0 written by pass 0: real
    Frame g;
    Scene::passWithDraw(g, Scene::pass(arr));
    Scene::passWithDraw(g, l1, {Scene::sample(arr, 0, 1)});  // feedback on its own layer...
    CHECK(compileFrame(g).encoders.size() == 2);
  }
  {  // disjoint byte ranges of one buffer: no hazard
    Frame f;
    f.dispatch(Scene::compute({write(ResourceRange::bytes(buf, 0, 128), stage::compute)}));
    f.dispatch(Scene::compute({write(ResourceRange::bytes(buf, 128, 128), stage::compute)}));
    f.dispatch(Scene::compute({read(ResourceRange::bytes(buf, 256, 128), stage::compute)}));
    CHECK(compileFrame(f).barriers.empty());
    f.dispatch(Scene::compute({read(ResourceRange::bytes(buf, 100, 50), stage::compute)}));  // straddles both
    const CompiledFrame cf = compileFrame(f);
    CHECK_EQ(cf.barriers.size(), size_t{1});
    CHECK_EQ(hazards(cf, HazardKind::RAW), 1u);
  }
  {  // a covering write retires older records; partial writes keep them
    Frame f;
    f.dispatch(Scene::compute({write(ResourceRange::bytes(buf, 0, 64), stage::compute)}));
    f.fillBuffer(Scene::span(buf, 0, 4096), 0);  // WAW, covers everything
    f.dispatch(Scene::compute({read(ResourceRange::bytes(buf, 0, 64), stage::compute)}));  // RAW on fill only
    const CompiledFrame cf = compileFrame(f);
    CHECK_EQ(cf.barriers.size(), size_t{2});
    CHECK(cf.barriers[1].after == stage::transfer);
  }
}

// A mip chain request (Copy::mipChain) writes the whole chain, not only mip s + 1.
void testMipChain() {
  Scene s;
  const TextureId t = s.rt(256, 5, 2), o = s.rt();
  Copy mc;
  mc.kind = CopyKind::TextureToTexture;
  mc.srcTexture = {t, 0, 0, 1, 0, 0, 0, 2, 2, 1};
  mc.dstTexture = {t, 1, 0, 1, 0, 0, 0, 1, 1, 1};
  mc.mipChain = 5;
  bool writesMip4 = false, writesMip0 = false;
  for (const Access& a : copyAccesses(mc)) {
    writesMip4 = writesMip4 || (a.write && a.range.overlaps(ResourceRange::subresource(t, 4, 1, 1, 1)));  // last mip, layer 1
    writesMip0 = writesMip0 || (a.write && a.range.overlaps(ResourceRange::subresource(t, 0, 1, 0, 2)));
  }
  CHECK(writesMip4 && !writesMip0);
  Copy plain = mc;
  plain.mipChain = 0;
  for (const Access& a : copyAccesses(plain)) CHECK(!(a.write && a.range.overlaps(ResourceRange::subresource(t, 4, 1, 0, 1))));
  {  // the chain reads mip 0 of every layer: a pass that rendered layer 1 of mip 0 before it is a RAW hazard
    Frame f;
    auto layer1 = Scene::pass(t);
    layer1.colors[0].layer = 1;
    Scene::passWithDraw(f, layer1);
    f.copy(mc);
    const CompiledFrame cf = compileFrame(f, &s.res);
    CHECK_EQ(hazards(cf, HazardKind::RAW), 1u);
  }
  {  // a pass sampling mip 3 recorded before the chain: the chain must not be hoisted above it
    Frame f;
    Scene::passWithDraw(f, Scene::pass(o), {Scene::sample(t, 3)});
    f.copy(mc);
    Scene::passWithDraw(f, Scene::pass(o));
    const CompiledFrame cf = compileFrame(f, &s.res);
    CHECK_EQ(cf.stats.copiesSunk, 1u);
  }
  {  // a pass sampling mip 3 after the chain waits for it
    Frame f;
    f.copy(mc);
    Scene::passWithDraw(f, Scene::pass(o), {Scene::sample(t, 3)});
    CHECK_EQ(hazards(compileFrame(f, &s.res), HazardKind::RAW), 1u);
  }
}

void testCuts() {
  Scene s;
  const TextureId c = s.rt(), d = s.depthRt(), other = s.rt();
  const BufferId buf = s.buf();
  Frame f;
  Scene::passWithDraw(f, Scene::pass(c, d));                                              // FrameStart
  f.dispatch(Scene::compute({write(ResourceRange::bytes(buf, 0, 256), stage::compute)})); // ComputeWork
  Scene::passWithDraw(f, Scene::pass(c, d), {Scene::vsRead(buf)});                        // ComputeBetween
  Scene::passWithDraw(f, Scene::pass(other), {Scene::sample(c)});                         // AttachmentChange
  Scene::passWithDraw(f, Scene::pass(other), {Scene::sample(other)});                     // Feedback
  f.clear(Scene::pass(other));                                                            // ClearLoad
  const CompiledFrame cf = compileFrame(f);
  CHECK_EQ(cf.encoders.size(), size_t{6});
  for (CutReason r : {CutReason::FrameStart, CutReason::ComputeWork, CutReason::ComputeBetween,
                      CutReason::AttachmentChange, CutReason::Feedback, CutReason::ClearLoad})
    CHECK_EQ(cuts(cf, r), 1u);
  uint32_t total = 0;
  for (uint32_t n : cf.stats.cuts) total += n;
  CHECK_EQ(total, static_cast<uint32_t>(cf.encoders.size()));  // every encoder has exactly one reason

  // A frame starting with copies: its first render encoder is still FrameStart.
  Frame g;
  g.fillBuffer(Scene::span(buf), 0);
  Scene::passWithDraw(g, Scene::pass(c));
  const CompiledFrame cg = compileFrame(g);
  CHECK_EQ(cuts(cg, CutReason::CopyWork), 1u);
  CHECK_EQ(cuts(cg, CutReason::FrameStart), 1u);

  // Commands on destroyed resources are dropped, not encoded.
  Frame h;
  const TextureId gone = s.rt();
  Scene::passWithDraw(h, Scene::pass(gone));
  Scene::passWithDraw(h, Scene::pass(c));
  s.res.destroy(gone);
  const CompiledFrame ch = compileFrame(h, &s.res);
  CHECK_EQ(ch.stats.dropped, 1u);
  CHECK_EQ(ch.encoders.size(), size_t{1});
}

void testDiagnostics() {
  Scene s;
  const TextureId a = s.rt(), b = s.rt();
  Frame f;
  auto p = Scene::pass(a);
  p.label = "main scene";
  Scene::passWithDraw(f, p);
  Scene::passWithDraw(f, Scene::pass(b), {Scene::sample(a)});
  const CompiledFrame cf = compileFrame(f);
  FrameReport r = makeReport(7, f, cf);
  r.gpu.frameMs = 1.5;
  const auto lines = formatReport(r);
  CHECK_EQ(lines.size(), size_t{2});
  CHECK(lines[0].starts_with("gpu.frame f=7 enc=2 render=2 compute=0 naive=2 "));
  CHECK(lines[0].find(" barriers=1 ") != std::string::npos && lines[0].find(" raw=1 ") != std::string::npos);
  CHECK(lines[0].find(" gpu_ms=1.500") != std::string::npos);
  CHECK(lines[1].find(" FrameStart=1 AttachmentChange=1 ") != std::string::npos);
  const auto enc = formatEncoders(7, f, cf);
  CHECK_EQ(enc.size(), size_t{2});
  CHECK(enc[0].find("label=main_scene") != std::string::npos);  // spaces escaped
  CHECK(enc[1].find("reason=AttachmentChange") != std::string::npos);
  CHECK(enc[1].find("barrier=RAW:2>2") != std::string::npos);
  for (const auto& l : lines) CHECK(l.find("  ") == std::string::npos);  // single-space separated
  for (const auto& l : lines) std::printf("  %s\n", l.c_str());
  for (const auto& l : enc) std::printf("  %s\n", l.c_str());
  {  // GpuPerf (vk.perf): count, sum and max of the command buffers' GPU time
    GpuPerf g;
    g.addCb(1500), g.addCb(4000), g.addCb(2500);
    CHECK_EQ(g.cbs.load(), uint64_t{3});
    CHECK_EQ(g.cbUs.load(), uint64_t{8000});
    CHECK_EQ(g.cbUsMax.load(), uint64_t{4000});
  }
}

void testPipelines() {
  NullBackend be;
  PipelineState a;
  a.vertexShader = 11, a.fragmentShader = 22;
  a.colorFormats[0] = Format::RGBA8Unorm;
  PipelineState b = a;
  b.label = "renamed";
  CHECK(a.key() == b.key());  // labels do not change the key
  b.blend[0].enabled = true;
  CHECK(!(a.key() == b.key()));
  {  // owners: the last release forgets the entry and destroys the native pipeline; a request rebuilds it
    NullBackend nb;
    PipelineCache cache(nb, 0);
    cache.release(a.key());  // never retained: ignored
    cache.retain(a.key()), cache.retain(a.key());
    CHECK(cache.request(a, Priority::Visible) == PipelineCache::Status::Ready);
    cache.release(a.key());
    CHECK(cache.status(a.key()) == PipelineCache::Status::Ready);
    cache.release(a.key());
    CHECK(cache.status(a.key()) == PipelineCache::Status::Missing);
    CHECK_EQ(nb.calls.size(), size_t{1});
    CHECK(nb.calls[0].rfind("destroyPipeline ", 0) == 0);
    CHECK(cache.request(a, Priority::Visible) == PipelineCache::Status::Ready);
  }
  {  // released while queued/compiling: the result is dropped, nothing leaks; a re-retain keeps it
    NullBackend nb;
    std::mutex m;
    std::condition_variable cv;
    bool open = false, entered = false;
    nb.onCompile = [&] {
      std::unique_lock lock(m);
      entered = true;
      cv.notify_all();
      cv.wait(lock, [&] { return open; });
    };
    auto release = [&] {
      std::lock_guard lock(m);
      open = true;
      cv.notify_all();
    };
    PipelineState q = a, kept = a;
    q.fragmentShader = 33, kept.fragmentShader = 44;
    {
      PipelineCache cache(nb, 1);
      cache.retain(a.key()), cache.retain(q.key()), cache.retain(kept.key());
      cache.request(a, Priority::Visible);  // the single worker takes it and blocks inside compile
      {
        std::unique_lock lock(m);
        cv.wait(lock, [&] { return entered; });
      }
      cache.request(q, Priority::Visible);
      CHECK(cache.status(q.key()) == PipelineCache::Status::Queued);
      cache.release(q.key());  // queued: forgotten, never compiled
      CHECK(cache.status(q.key()) == PipelineCache::Status::Missing);
      CHECK(cache.status(a.key()) == PipelineCache::Status::Compiling);
      cache.release(a.key());  // compiling: marked orphan
      CHECK(cache.status(a.key()) == PipelineCache::Status::Compiling);
      release();
      cache.waitIdle();
      CHECK(cache.status(a.key()) == PipelineCache::Status::Missing);
      CHECK_EQ(nb.calls.size(), size_t{1});
      CHECK(nb.calls[0].rfind("destroyPipeline ", 0) == 0);
      CHECK_EQ(cache.totals().compiled, 1u);  // q was never compiled
      CHECK(cache.request(a, Priority::Visible) == PipelineCache::Status::Queued);  // a later request rebuilds it
      cache.waitIdle();
      CHECK(cache.status(a.key()) == PipelineCache::Status::Ready);
    }
    {  // re-owned while compiling: kept
      PipelineCache cache(nb, 1);
      {
        std::lock_guard lock(m);
        open = false, entered = false;
      }
      cache.retain(kept.key());
      cache.request(kept, Priority::Visible);
      {
        std::unique_lock lock(m);
        cv.wait(lock, [&] { return entered; });
      }
      cache.release(kept.key());
      cache.retain(kept.key());
      release();
      cache.waitIdle();
      CHECK(cache.status(kept.key()) == PipelineCache::Status::Ready);
    }
  }
  {
    PipelineCache cache(be, 0);
    CHECK(cache.request(a, Priority::Visible) == PipelineCache::Status::Ready);
    CHECK(cache.request(a, Priority::Visible) == PipelineCache::Status::Ready);
    be.failPipeline = b.key().value;
    CHECK(cache.request(b, Priority::Visible) == PipelineCache::Status::Failed);
    const PipelineStats st = cache.takeFrameStats();
    CHECK_EQ(st.compiled, 1u);
    CHECK_EQ(st.failed, 1u);
    CHECK_EQ(st.hits, 1u);
    CHECK_EQ(st.misses, 2u);
    CHECK_EQ(cache.takeFrameStats().compiled, 0u);  // per-frame counters reset
    CHECK_EQ(cache.totals().compiled, 1u);

    const DiskCachePath path{std::filesystem::temp_directory_path() / "vpgpu_core_test_cache"};
    CHECK(cache.save(path));
    PipelineCache fresh(be, 2);
    CHECK_EQ(fresh.prefetch(path), size_t{1});  // only Ready states are saved
    fresh.waitIdle();
    CHECK(fresh.status(a.key()) == PipelineCache::Status::Ready);
    std::filesystem::remove_all(path.root);
  }
  {  // background workers drain many requests; every one ends Ready
    be.failPipeline = 0;
    PipelineCache cache(be, 3);
    std::vector<PipelineState> states(64, a);
    for (size_t i = 0; i < states.size(); ++i) {
      states[i].fragmentShader = 1000 + i;
      cache.request(states[i], i % 2 ? Priority::Prefetch : Priority::Visible);
    }
    cache.request(states[5], Priority::Blocking);  // raising priority of a queued one is harmless
    cache.waitIdle();
    for (const PipelineState& st : states) CHECK(cache.status(st.key()) == PipelineCache::Status::Ready);
    CHECK_EQ(cache.totals().compiled, 64u);
  }
  {  // boost() moves a queued job ahead of everything queued, even other Blocking jobs; compiled once
    NullBackend nb;
    std::mutex m;
    std::condition_variable cv;
    unsigned entered = 0, allowed = 0;
    nb.onCompile = [&] {  // the Nth compile starts, then waits until `allowed` reaches N
      std::unique_lock lock(m);
      const unsigned n = ++entered;
      cv.notify_all();
      cv.wait(lock, [&] { return allowed >= n; });
    };
    auto waitEntered = [&](unsigned n) {
      std::unique_lock lock(m);
      cv.wait(lock, [&] { return entered >= n; });
    };
    auto allow = [&](unsigned n) {
      std::lock_guard lock(m);
      allowed = n;
      cv.notify_all();
    };
    PipelineCache cache(nb, 1);
    std::vector<PipelineState> states(6, a);
    for (size_t i = 0; i < states.size(); ++i) states[i].fragmentShader = 2000 + i;
    cache.request(states[0], Priority::Blocking);  // the single worker takes it and waits inside compile
    waitEntered(1);
    for (size_t i = 1; i < states.size(); ++i) cache.request(states[i], Priority::Blocking);  // the game-create backlog
    cache.boost(states[5].key());
    cache.boost(states[5].key());  // a second boost must not queue it again
    allow(1);
    waitEntered(2);
    CHECK(cache.status(states[5].key()) == PipelineCache::Status::Compiling);  // next, ahead of states 1..4
    for (size_t i = 1; i < 5; ++i) CHECK(cache.status(states[i].key()) == PipelineCache::Status::Queued);
    allow(100);
    cache.waitIdle();
    CHECK_EQ(cache.totals().compiled, 6u);
    CHECK_EQ(entered, 6u);
  }
}

void testDevice() {
  NullBackend be;
  std::vector<std::string> log;
  Device dev(be, 0, [&](const std::string& l) { log.push_back(l); });
  dev.setDetailEvery(1);
  TextureDesc td;
  td.format = Format::RGBA8Unorm;
  td.usage = usage::colorTarget;
  const TextureId t = dev.createTexture(td);
  td.width = 0;
  CHECK(!dev.createTexture(td));
  CHECK(!log.empty() && log.back().starts_with("gpu.err what=texture_desc"));
  int guest[16] = {};
  const BufferId b = dev.createBuffer({sizeof guest, guest, "guest"});
  CHECK(dev.resources().get(b)->guestMemory == guest);
  Frame f;
  Scene::passWithDraw(f, Scene::pass(t), {Scene::vsRead(b, 0, 64)});
  log.clear();
  const FrameReport r = dev.submit(f);
  CHECK_EQ(r.encoders, 1u);
  CHECK(log.size() == 4 && log[0].starts_with("gpu.frame f=0") && log[2].starts_with("gpu.enc") &&
        log[3].starts_with("gpu.time f=0") && log[3].find(" t=") != std::string::npos);  // monotonic ms
  dev.destroy(b);
  dev.destroy(b);  // double destroy is ignored
  CHECK(be.calls[0] == "createTexture 0 RGBA8Unorm 1x1");
  CHECK(be.calls[1] == "createBuffer 0 size=64 guest=true");
  CHECK(be.calls[2] == "submit 0 encoders=1 barriers=0");
  CHECK(be.calls.size() == 4 && be.calls[3] == "destroyBuffer 0");
  // A pixel-local read nothing serves is a client bug: an error line, not only a counter.
  td.width = 1;
  const TextureId unrendered = dev.createTexture(td);
  Frame g;
  Scene::passWithDraw(g, Scene::pass(t), {pixelLocalRead(unrendered)});
  log.clear();
  dev.submit(g);
  CHECK(log.size() >= 3 && log[2] == "gpu.err what=bad_fetch f=1 draws=1 reads=1 first=c0.d0");
  {  // a copy of a description under the table lock; gone after destroy
    TextureDesc copy;
    CHECK(dev.textureDesc(t, copy) && copy.format == Format::RGBA8Unorm);
    const TextureId tmp = dev.createTexture(td);
    dev.destroy(tmp);
    CHECK(!dev.textureDesc(tmp, copy));
  }
}

// Deferred shading on a TBDR GPU: G-buffer + lighting in one encoder, G-buffer never in memory.
struct Deferred {
  Scene s;
  TextureId albedo, normal, material, depth, hdr, ao;
  explicit Deferred(uint32_t size = 1440) {
    albedo = s.rt(size), normal = s.rt(size, 1, 1, Format::RGBA16Float), material = s.rt(size);
    depth = s.depthRt(size), hdr = s.rt(size, 1, 1, Format::RGBA16Float), ao = s.rt(size, 1, 1, Format::R8Unorm);
  }
  // G-buffer targets declared DontCare: the client lets go of them after lighting.
  RenderPassDesc gbuffer(LoadAction load) const {
    RenderPassDesc d;
    const TextureId t[] = {albedo, normal, material};
    for (int i = 0; i < 3; ++i) d.colors[i] = {t[i], 0, 0, load, StoreAction::DontCare, {}, {}, 0, 0};
    d.depth = {depth, 0, 0, load, StoreAction::DontCare, {}, {}, 0, 0};
    d.label = "gbuffer";
    return d;
  }
  // Lighting writes hdr (slot 3, so slots 0-2 stay the G-buffer's) with read-only depth test.
  RenderPassDesc lighting() const {
    RenderPassDesc d;
    d.colors[3].texture = hdr;
    d.colors[3].load = LoadAction::Clear;
    d.depth = {depth, 0, 0, LoadAction::Load, StoreAction::DontCare, {}, {}, 0, 0};
    d.label = "lighting";
    return d;
  }
  std::vector<Access> fetches() const {
    return {pixelLocalRead(albedo), pixelLocalRead(normal), pixelLocalRead(material)};
  }
};

// Unity's final MSAA resolve: a pass without draws after the pass that rendered the MSAA colour.
void testEmptyResolve() {
  struct Setup {
    Scene s;
    TextureId ms, single, other;
    Setup() {
      TextureDesc d;
      d.format = Format::RGBA8Unorm;
      d.width = d.height = 256;
      d.samples = 2;
      d.usage = usage::colorTarget;
      ms = s.res.createTexture(d);
      single = s.rt();
      other = s.rt();
    }
    RenderPassDesc draws() const {
      RenderPassDesc p = Scene::pass(ms, {}, LoadAction::Clear);
      p.colors[0].store = StoreAction::DontCare;
      return p;
    }
    RenderPassDesc resolve(StoreAction st = StoreAction::Resolve, TextureId to = {}) const {
      RenderPassDesc p = Scene::pass(ms);
      p.colors[0].store = st;
      p.colors[0].resolve = to ? to : single;
      return p;
    }
    static void empty(Frame& f, RenderPassDesc d) {
      f.beginRenderPass(std::move(d));
      f.endRenderPass();
    }
  };
  {  // joins the encoder whose draws sample another texture; resolve last, store combined
    Setup t;
    Frame f;
    Scene::passWithDraw(f, t.draws(), {Scene::sample(t.other)});
    Setup::empty(f, t.resolve());
    const CompiledFrame cf = compileFrame(f, &t.s.res);
    CHECK_EQ(cf.encoders.size(), size_t{1});
    CHECK_EQ(cf.encoders[0].passes, 2u);
    CHECK(cf.encoders[0].pass.colors[0].load == LoadAction::Clear);
    CHECK(cf.encoders[0].pass.colors[0].store == StoreAction::Resolve);
    CHECK(cf.encoders[0].pass.colors[0].resolve == t.single);
    CHECK(cf.encoders[0].tileLocal(0));  // the MSAA colour no longer reaches memory
    Frame g;
    Scene::passWithDraw(g, t.draws(), {Scene::sample(t.other)});
    Setup::empty(g, t.resolve(StoreAction::Store));
    const CompiledFrame cg = compileFrame(g, &t.s.res);
    CHECK_EQ(cg.encoders.size(), size_t{1});
    CHECK(cg.encoders[0].pass.colors[0].store == StoreAction::Store);  // DontCare + Store = Store
    Frame h;
    Scene::passWithDraw(h, t.draws(), {Scene::sample(t.other)});
    Setup::empty(h, t.resolve(StoreAction::StoreAndResolve));
    const CompiledFrame ch = compileFrame(h, &t.s.res);  // default: msaaResolveOnTile is off, the samples are stored
    CHECK_EQ(ch.encoders.size(), size_t{1});
    CHECK(ch.encoders[0].pass.colors[0].store == StoreAction::StoreAndResolve);
    CompileOptions on;
    on.msaaResolveOnTile = true;  // opt-in: nothing reads the samples in the frame, only the resolve is kept
    CHECK(compileFrame(h, &t.s.res, on).encoders[0].pass.colors[0].store == StoreAction::Resolve);
  }
  {  // an empty pass that clears keeps its own encoder
    Setup t;
    Frame f;
    Scene::passWithDraw(f, t.draws(), {Scene::sample(t.other)});
    RenderPassDesc c = t.resolve();
    c.colors[0].load = LoadAction::Clear;
    Setup::empty(f, c);
    CHECK_EQ(compileFrame(f, &t.s.res).encoders.size(), size_t{2});
  }
  {  // resolving into a texture the encoder samples is real feedback
    Setup t;
    Frame f;
    Scene::passWithDraw(f, t.draws(), {Scene::sample(t.single)});
    Setup::empty(f, t.resolve());
    const CompiledFrame cf = compileFrame(f, &t.s.res);
    CHECK_EQ(cf.encoders.size(), size_t{2});
    CHECK_EQ(cuts(cf, CutReason::Feedback), 1u);
  }
  {  // the open encoder already resolved this slot: still cut
    Setup t;
    Frame f;
    RenderPassDesc first = t.draws();
    first.colors[0].store = StoreAction::Resolve;
    first.colors[0].resolve = t.single;
    Scene::passWithDraw(f, first);
    Setup::empty(f, t.resolve(StoreAction::Resolve, t.other));
    CHECK_EQ(compileFrame(f, &t.s.res).encoders.size(), size_t{2});
  }
  {  // a copy of the MSAA colour between the passes is sunk behind the encoder: the pass that
     // would drop/replace that store must not join it (PinnedCopy)
    Setup t;
    Frame f;
    Scene::passWithDraw(f, t.draws(), {Scene::sample(t.other)});
    Copy c;
    c.kind = CopyKind::TextureToTexture;
    c.srcTexture = {t.ms, 0, 0, 1};
    c.dstTexture = {t.other, 0, 0, 1};
    f.copy(c);
    Setup::empty(f, t.resolve());
    const CompiledFrame cf = compileFrame(f, &t.s.res);
    CHECK_EQ(cuts(cf, CutReason::PinnedCopy), 1u);
    CHECK(cf.encoders.size() >= size_t{3});
  }
  {  // a copy that reads the resolve target before the resolve pass keeps running before it
    Setup t;
    Frame f;
    Scene::passWithDraw(f, t.draws(), {Scene::sample(t.other)});
    Copy c;
    c.kind = CopyKind::TextureToTexture;
    c.srcTexture = {t.single, 0, 0, 1};
    c.dstTexture = {t.other, 0, 0, 1};
    f.copy(c);
    Setup::empty(f, t.resolve());
    const CompiledFrame cf = compileFrame(f, &t.s.res);
    size_t copyEnc = 99, resolveEnc = 99;
    for (size_t e = 0; e < cf.encoders.size(); ++e)
      for (uint32_t cmd : cf.encoders[e].commands) {
        if (cmd == 1) copyEnc = e;
        if (cmd == 2) resolveEnc = e;
      }
    CHECK(copyEnc < 99 && resolveEnc < 99 && copyEnc <= resolveEnc);
  }
}

void testTile() {
  testEmptyResolve();
  {  // MSAA colour + depth stored AND resolved, read by nothing in the frame: samples stay on tile, resolve kept
    Scene s;
    TextureDesc md;
    md.format = Format::RGBA8Unorm; md.width = md.height = 256; md.samples = 4; md.usage = usage::colorTarget;
    const TextureId mc = s.res.createTexture(md);
    md.format = Format::D32Float; md.usage = usage::depthTarget;
    const TextureId mdep = s.res.createTexture(md);
    const TextureId rc = s.rt(), rd = s.depthRt();
    auto msaa = [&]() {
      auto d = Scene::pass(mc, mdep, LoadAction::Clear);
      d.colors[0].store = StoreAction::StoreAndResolve; d.colors[0].resolve = rc;
      d.depth.store = StoreAction::StoreAndResolve; d.depth.resolve = rd;
      return d;
    };
    CompileOptions on;
    on.msaaResolveOnTile = true;  // opt-in (Device: VP_MSAA_TILE=1)
    {
      Frame f;
      Scene::passWithDraw(f, msaa());
      Scene::passWithDraw(f, Scene::pass(s.rt()), {Scene::sample(rc)});  // the resolve is what gets read
      const CompiledFrame cf = compileFrame(f, &s.res, on);
      CHECK_EQ(cf.stats.tileLocal, 2u);
      CHECK(cf.encoders[0].pass.colors[0].store == StoreAction::Resolve);
      CHECK(cf.encoders[0].pass.depth.store == StoreAction::Resolve);
      CHECK(cf.encoders[0].pass.colors[0].resolve == rc);
    }
    {  // the samples themselves read in the frame (a custom resolve): stored as asked
      Frame f;
      Scene::passWithDraw(f, msaa());
      Scene::passWithDraw(f, Scene::pass(s.rt()), {Scene::sample(mc)});
      const CompiledFrame cf = compileFrame(f, &s.res, on);
      CHECK_EQ(cf.stats.tileLocal, 1u);  // depth only
      CHECK(cf.encoders[0].pass.colors[0].store == StoreAction::StoreAndResolve);
      CHECK(cf.encoders[0].tile[0] == TileVerdict::SampledRead);
      CHECK_EQ(cf.msaaRejects.size(), 1u);  // the colour, with why it stayed in memory
      CHECK(cf.msaaRejects[0].why == TileVerdict::SampledRead && cf.msaaRejects[0].slot == 0 && cf.msaaRejects[0].sampled);
      CHECK_EQ(cf.msaaRejects[0].samples, 4u);
    }
    {  // the samples read in a frame (submit) that does not render them: counted as a stale read
      Frame f;
      Scene::passWithDraw(f, Scene::pass(s.rt()), {Scene::sample(mc)});
      const CompiledFrame cf = compileFrame(f, &s.res);
      CHECK_EQ(cf.stats.msaaStaleReads, 1u);
      Frame g;  // rendered first, read after: not stale
      Scene::passWithDraw(g, msaa());
      Scene::passWithDraw(g, Scene::pass(s.rt()), {Scene::sample(mc)});
      CHECK_EQ(compileFrame(g, &s.res).stats.msaaStaleReads, 0u);
      Frame h;  // attached twice, read between the two: the read is after the FIRST attachment, not stale
      Scene::passWithDraw(h, msaa());
      Scene::passWithDraw(h, Scene::pass(s.rt()), {Scene::sample(mc)});
      Scene::passWithDraw(h, msaa());
      CHECK_EQ(compileFrame(h, &s.res).stats.msaaStaleReads, 0u);
      Frame l;  // first attachment Loads it (nothing rendered it earlier in the frame): contents from an earlier submit
      auto loading = msaa();
      loading.colors[0].load = LoadAction::Load;
      Scene::passWithDraw(l, loading);
      CHECK_EQ(compileFrame(l, &s.res).stats.msaaStaleReads, 1u);
    }
    {  // default (option off): samples stored as asked
      Frame f;
      Scene::passWithDraw(f, msaa());
      const CompiledFrame cf = compileFrame(f, &s.res);
      CHECK_EQ(cf.stats.tileLocal, 0u);
      CHECK(cf.encoders[0].pass.colors[0].store == StoreAction::StoreAndResolve);
    }
    {  // colour-only option (Device default): the colour drops its sample store, the depth keeps it (VPKTAA reads it)
      CompileOptions col;
      col.msaaColorResolveOnTile = true;
      Frame f;
      Scene::passWithDraw(f, msaa());
      const CompiledFrame cf = compileFrame(f, &s.res, col);
      CHECK_EQ(cf.stats.tileLocal, 1u);
      CHECK(cf.encoders[0].tileLocal(0) && cf.encoders[0].pass.colors[0].store == StoreAction::Resolve);
      CHECK(cf.encoders[0].tile[kDepthSlot] == TileVerdict::None);
      CHECK(cf.encoders[0].pass.depth.store == StoreAction::StoreAndResolve);
    }
    {  // keepSamples (the translator's "the samples are needed"): the store stays under either option
      auto keeping = [&]() {
        auto d = msaa();
        d.colors[0].keepSamples = true;
        d.depth.keepSamples = true;
        return d;
      };
      CompileOptions col;
      col.msaaColorResolveOnTile = true;
      for (const CompileOptions& o : {col, on}) {
        Frame f;
        Scene::passWithDraw(f, keeping());
        const CompiledFrame cf = compileFrame(f, &s.res, o);
        CHECK_EQ(cf.stats.tileLocal, 0u);
        CHECK(cf.encoders[0].pass.colors[0].store == StoreAction::StoreAndResolve && cf.encoders[0].tile[0] == TileVerdict::None);
        CHECK(cf.encoders[0].pass.depth.store == StoreAction::StoreAndResolve);
      }
      Frame m;  // merged passes (a draw pass, then an empty pass that only stores): the flag follows the pass whose store wins
      auto drawing = msaa();
      drawing.colors[0].store = drawing.depth.store = StoreAction::DontCare;
      drawing.colors[0].resolve = drawing.depth.resolve = {};
      Scene::passWithDraw(m, drawing);
      RenderPassDesc second = keeping();
      second.colors[0].load = second.depth.load = LoadAction::Load;
      m.beginRenderPass(std::move(second));
      m.endRenderPass();
      const CompiledFrame cm = compileFrame(m, &s.res, col);
      CHECK_EQ(cm.encoders.size(), size_t{1});
      CHECK(cm.encoders[0].pass.colors[0].keepSamples && cm.encoders[0].pass.colors[0].store == StoreAction::StoreAndResolve);
    }
  }
  {  // classic deferred: 1 encoder, 3 colour targets + depth on tile
    Deferred g;
    Frame f;
    Scene::passWithDraw(f, g.gbuffer(LoadAction::Clear));
    Scene::passWithDraw(f, g.gbuffer(LoadAction::Load));
    Scene::passWithDraw(f, g.lighting(), g.fetches());
    const CompiledFrame cf = compileFrame(f, &g.s.res);
    CHECK_EQ(cf.encoders.size(), size_t{1});
    CHECK_EQ(cf.encoders[0].passes, 3u);
    CHECK_EQ(cf.stats.tileLocal, 4u);  // albedo, normal, material, depth
    CHECK_EQ(cf.stats.tileRejected, 0u);
    CHECK(cf.barriers.empty());
    const Encoder& e = cf.encoders[0];
    for (int i : {0, 1, 2, kDepthSlot}) CHECK(e.tileLocal(i));
    CHECK(e.pass.colors[0].store == StoreAction::DontCare && e.pass.colors[0].load == LoadAction::Clear);
    CHECK(e.pass.colors[3].texture == g.hdr && e.pass.colors[3].load == LoadAction::Clear);
    CHECK(e.pass.colors[3].store == StoreAction::Store && e.tile[3] == TileVerdict::None);
    // Fetch index contract: the colour attachment index of each pixel-local read.
    const Draw& light = std::get<RenderPass>(f.commands()[2]).draws[0];
    CHECK_EQ(fetchIndex(e.pass, light.bindings[1].range), 1);
    FetchInputs fi;
    CHECK(fetchInputs(light, e.pass, fi) && fi.mask == 0b111);
    const auto lines = formatReport(makeReport(0, f, cf));
    CHECK(lines[0].find(" tile_local=4 tile_rejected=0 saved_mb=39.6 ") != std::string::npos);
    const auto enc = formatEncoders(0, f, cf);
    CHECK(enc[0].find(" tile=c0,c1,c2,d ") != std::string::npos);
    for (const auto& l : lines) std::printf("  %s\n", l.c_str());
    for (const auto& l : enc) std::printf("  %s\n", l.c_str());
  }
  {  // without pixelLocal the same frame is exactly the old behaviour: Feedback cut, all stored
    Deferred g;
    Frame f;
    Scene::passWithDraw(f, g.gbuffer(LoadAction::Clear));
    Scene::passWithDraw(f, g.lighting(), {Scene::sample(g.albedo), Scene::sample(g.normal), Scene::sample(g.material)});
    const CompiledFrame cf = compileFrame(f, &g.s.res);
    CHECK_EQ(cf.encoders.size(), size_t{2});
    CHECK_EQ(cuts(cf, CutReason::AttachmentChange), 1u);
    CHECK_EQ(cf.stats.tileLocal, 0u);
    for (int i = 0; i < 3; ++i) CHECK(cf.encoders[0].pass.colors[i].store == StoreAction::Store);  // kept: read later
    CHECK_EQ(cf.stats.storesKept, 4u);  // 3 sampled targets + depth loaded by lighting
  }
  {  // SSAO samples the normal target with offsets between G-buffer and lighting
    Deferred g;
    Frame f;
    Scene::passWithDraw(f, g.gbuffer(LoadAction::Clear));
    Scene::passWithDraw(f, Scene::pass(g.ao, {}, LoadAction::DontCare), {Scene::sample(g.normal), Scene::sample(g.depth)});
    std::vector<Access> in = g.fetches();
    in.push_back(Scene::sample(g.ao));
    Scene::passWithDraw(f, g.lighting(), in);
    const CompiledFrame cf = compileFrame(f, &g.s.res);
    CHECK_EQ(cf.encoders.size(), size_t{3});  // gbuffer, ssao, lighting (G-buffer reloaded)
    CHECK_EQ(cf.stats.tileLocal, 0u);
    const Encoder &gb = cf.encoders[0], &light = cf.encoders[2];
    CHECK(gb.tile[1] == TileVerdict::SampledRead);
    CHECK(gb.tile[0] == TileVerdict::OtherEncoder);  // reloaded by lighting's fetch
    for (int i = 0; i < 3; ++i) CHECK(gb.pass.colors[i].store == StoreAction::Store);
    CHECK(gb.pass.depth.store == StoreAction::Store);
    for (int i = 0; i < 3; ++i) CHECK(light.pass.colors[i].texture && light.pass.colors[i].load == LoadAction::Load);
    CHECK_EQ(fetchIndex(light.pass, ResourceRange::subresource(g.normal, 0, 1, 0, 1)), 1);
    CHECK_EQ(cf.barriers.size(), size_t{2});  // gbuffer -> ssao (RAW); ssao -> lighting (RAW on ao, WAR on reloaded targets)
    CHECK_EQ(cf.stats.badFetch, 0u);
    for (const auto& l : formatEncoders(0, f, cf)) std::printf("  %s\n", l.c_str());
  }
  {  // TAA history is persistent: never tile-local even if only used inside one encoder
    Scene s;
    TextureDesc d;
    d.format = Format::RGBA16Float;
    d.width = d.height = 256;
    d.usage = usage::colorTarget | usage::sampled;
    d.persistent = true;
    const TextureId history = s.res.createTexture(d), out = s.rt();
    RenderPassDesc p = Scene::pass(history, {}, LoadAction::Clear);
    p.colors[0].store = StoreAction::DontCare;
    p.colors[1].texture = out;
    p.colors[1].load = LoadAction::Clear;
    Frame f;
    Scene::passWithDraw(f, p);
    p.colors[0].load = p.colors[1].load = LoadAction::Load;
    Scene::passWithDraw(f, p, {pixelLocalRead(history)});
    const CompiledFrame cf = compileFrame(f, &s.res);
    CHECK_EQ(cf.encoders.size(), size_t{1});
    CHECK(cf.encoders[0].tile[0] == TileVerdict::Persistent);
    CHECK_EQ(cf.stats.tileLocal, 0u);
  }
  {  // different sample count (MSAA light pass, 1x G-buffer): the 1x targets cannot join a 4x
     // encoder, so the fetches are bad and the draw is skipped; contents kept
    Deferred g(256);
    TextureDesc d;
    d.format = Format::RGBA8Unorm;
    d.width = d.height = 256;
    d.samples = 4;
    d.usage = usage::colorTarget;
    const TextureId msaa = g.s.res.createTexture(d);
    Frame f;
    Scene::passWithDraw(f, g.gbuffer(LoadAction::Clear));
    RenderPassDesc l;
    l.colors[3].texture = msaa;
    Scene::passWithDraw(f, l, g.fetches());
    const CompiledFrame cf = compileFrame(f, &g.s.res);
    CHECK_EQ(cf.encoders.size(), size_t{2});
    CHECK_EQ(cf.stats.badFetch, 3u);
    CHECK(cf.badDraws.size() == 1 && cf.skipped(1, 0) && !cf.skipped(0, 0));
    CHECK(!cf.encoders[1].pass.colors[0].texture);  // never added to the 4x encoder
    CHECK_EQ(cf.stats.tileLocal, 1u);  // only depth, which nothing reads afterwards
    CHECK(cf.encoders[0].tileLocal(kDepthSlot));
    CHECK(cf.encoders[0].tile[0] == TileVerdict::BadFetch && cf.encoders[0].pass.colors[0].store == StoreAction::Store);
  }
  {  // half-res G-buffer fetched by a full-res pass: size mismatch, bad fetch, nothing merged
    Scene s;
    const TextureId g = s.rt(128), hdr = s.rt(256);
    RenderPassDesc p = Scene::pass(g, {}, LoadAction::Clear);
    p.colors[0].store = StoreAction::DontCare;
    RenderPassDesc q;
    q.colors[1].texture = hdr;
    q.colors[1].load = LoadAction::Clear;
    Frame f;
    Scene::passWithDraw(f, p);
    Scene::passWithDraw(f, q, {pixelLocalRead(g)});
    const CompiledFrame cf = compileFrame(f, &s.res);
    CHECK_EQ(cf.encoders.size(), size_t{2});
    CHECK(cf.skipped(1, 0) && !cf.encoders[1].pass.colors[0].texture);
  }
  {  // a texture may sit in only one slot of an encoder: the pass rendering it at another slot cuts
    Scene s;
    const TextureId t = s.rt(), u = s.rt(), out = s.rt();
    RenderPassDesc p;
    p.colors[0] = {t, 0, 0, LoadAction::Clear, StoreAction::DontCare, {}, {}, 0, 0};
    p.colors[1] = {u, 0, 0, LoadAction::Clear, StoreAction::DontCare, {}, {}, 0, 0};
    RenderPassDesc q;
    q.colors[2].texture = t;
    q.colors[3].texture = out;
    q.colors[3].load = LoadAction::Clear;
    Frame f;
    Scene::passWithDraw(f, p);
    Scene::passWithDraw(f, q, {pixelLocalRead(u)});
    const CompiledFrame cf = compileFrame(f, &s.res);
    CHECK_EQ(cf.encoders.size(), size_t{2});
    CHECK_EQ(cuts(cf, CutReason::AttachmentChange), 1u);
    CHECK(cf.badDraws.empty());
    CHECK(cf.encoders[1].pass.colors[1].texture == u && cf.encoders[1].pass.colors[1].load == LoadAction::Load);
  }
  {  // an explicit Store is honoured even when the texture is only read pixel-locally afterwards
    Scene s;
    const TextureId a = s.rt(256, 1, 1, Format::RGBA16Float), out = s.rt();
    RenderPassDesc p = Scene::pass(a, {}, LoadAction::Clear);
    RenderPassDesc q;
    q.colors[1].texture = out;
    q.colors[1].load = LoadAction::Clear;
    Frame f;
    Scene::passWithDraw(f, p);
    Scene::passWithDraw(f, q, {pixelLocalRead(a)});
    CompiledFrame cf = compileFrame(f, &s.res);
    CHECK_EQ(cf.encoders.size(), size_t{1});
    CHECK(cf.encoders[0].tile[0] == TileVerdict::None && cf.encoders[0].pass.colors[0].store == StoreAction::Store);
    CHECK_EQ(cf.stats.tileLocal, 0u);
    // ...unless the client marks it transient.
    TextureDesc d = *s.res.get(a);
    d.transient = true;
    const TextureId b = s.res.createTexture(d);
    p.colors[0].texture = b;
    Frame g2;
    Scene::passWithDraw(g2, p);
    Scene::passWithDraw(g2, q, {pixelLocalRead(b)});
    cf = compileFrame(g2, &s.res);
    CHECK(cf.encoders[0].tileLocal(0) && cf.encoders[0].pass.colors[0].store == StoreAction::DontCare);
  }
  {  // tile budget: a fetch merge that would exceed it cuts instead
    Scene s;
    TextureId g[4];
    RenderPassDesc p;
    for (int i = 0; i < 4; ++i) {
      g[i] = s.rt(256, 1, 1, Format::RGBA32Float);  // 4 x 16 B = 64 B: at the default budget
      p.colors[i] = {g[i], 0, 0, LoadAction::Clear, StoreAction::DontCare, {}, {}, 0, 0};
    }
    RenderPassDesc q;
    q.colors[4].texture = s.rt();
    q.colors[4].load = LoadAction::Clear;
    Frame f;
    Scene::passWithDraw(f, p);
    Scene::passWithDraw(f, q, {pixelLocalRead(g[0])});
    CompiledFrame cf = compileFrame(f, &s.res);
    CHECK_EQ(cf.encoders.size(), size_t{2});
    CHECK_EQ(cuts(cf, CutReason::TileBudget), 1u);
    CHECK(cf.badDraws.empty() && cf.encoders[1].pass.colors[0].texture == g[0]);  // reloaded instead
    CompileOptions big;
    big.tileBytesPerPixel = 128;
    cf = compileFrame(f, &s.res, big);
    CHECK_EQ(cf.encoders.size(), size_t{1});
    CHECK_EQ(cf.stats.tileLocal, 4u);
  }
  {  // pixel-local read of a texture never rendered this frame: reported, its draw skipped
    Scene s;
    const TextureId a = s.rt(), b = s.rt();
    Frame f;
    f.beginRenderPass(Scene::pass(a));
    f.draw(Scene::draw());
    f.draw(Scene::draw({pixelLocalRead(b)}));
    f.endRenderPass();
    const CompiledFrame cf = compileFrame(f, &s.res);
    CHECK_EQ(cf.stats.badFetch, 1u);
    CHECK(cf.badDraws.size() == 1 && cf.skipped(0, 1) && !cf.skipped(0, 0));
    CHECK(formatReport(makeReport(0, f, cf))[0].find(" bad_fetch=1 ") != std::string::npos);
  }
  {  // MSAA G-buffer resolved at the end: samples stay on tile, only the resolve hits memory
    Scene s;
    TextureDesc d;
    d.format = Format::RGBA8Unorm;
    d.width = d.height = 256;
    d.samples = 4;
    d.usage = usage::colorTarget;
    const TextureId ms = s.res.createTexture(d), single = s.rt();
    RenderPassDesc p = Scene::pass(ms, {}, LoadAction::Clear);
    p.colors[0].store = StoreAction::StoreAndResolve;
    p.colors[0].resolve = single;
    Frame f;
    Scene::passWithDraw(f, p);
    Scene::passWithDraw(f, Scene::pass(s.rt()), {Scene::sample(single)});
    CompileOptions on;
    on.msaaResolveOnTile = true;
    const CompiledFrame cf = compileFrame(f, &s.res, on);
    // StoreAndResolve with nothing reading the samples in the frame: the resolve is kept, the samples
    // stay on tile (CompileOptions::msaaResolveOnTile, opt-in).
    CHECK(cf.encoders[0].tileLocal(0) && cf.encoders[0].pass.colors[0].store == StoreAction::Resolve);
    CHECK_EQ(hazards(cf, HazardKind::RAW), 1u);  // resolve target -> sampled
    {
      const CompiledFrame co = compileFrame(f, &s.res);  // default: off, the client asked for the samples
      CHECK(co.encoders[0].tile[0] == TileVerdict::None);
    }
    p.colors[0].store = StoreAction::Resolve;
    Frame g2;
    Scene::passWithDraw(g2, p);
    const CompiledFrame cg = compileFrame(g2, &s.res);
    CHECK(cg.encoders[0].tileLocal(0) && cg.encoders[0].pass.colors[0].store == StoreAction::Resolve);
  }
  {  // HeavyGeometry: MSAA StoreAndResolve keeps not storing its samples, with real memory (memBacked); other tile-local
     // candidates keep their DontCare store, also with real memory; limits count vertices once per layer
    Scene s;
    TextureDesc d;
    d.format = Format::RGBA8Unorm;
    d.width = d.height = 256;
    d.samples = 4;
    d.usage = usage::colorTarget;
    const TextureId ms = s.res.createTexture(d), single = s.rt();
    RenderPassDesc p = Scene::pass(ms, {}, LoadAction::Clear);
    p.colors[0].store = StoreAction::StoreAndResolve;
    p.colors[0].resolve = single;
    Frame f;
    Scene::passWithDraw(f, p);
    CompileOptions o;
    o.msaaResolveOnTile = true;
    const CompiledFrame light = compileFrame(f, &s.res, o);
    CHECK(light.encoders[0].tileLocal(0) && !light.encoders[0].memBacked[0] && light.encoders[0].heavy < 1.0f);
    o.tileMaxDraws = 0;  // one draw is already "heavy"
    const CompiledFrame heavy = compileFrame(f, &s.res, o);
    CHECK(heavy.encoders[0].tile[0] == TileVerdict::HeavyGeometry && heavy.encoders[0].memBacked[0]);
    CHECK(heavy.encoders[0].pass.colors[0].store == StoreAction::Resolve);  // samples still not written
    CHECK_EQ(heavy.stats.heavyResolve, 1u);
    CHECK(heavy.encoders[0].heavy >= 1.0f);
    o.msaaResolveOnTile = false;  // the client asked for the samples: they are stored
    const CompiledFrame kept = compileFrame(f, &s.res, o);
    CHECK(kept.encoders[0].pass.colors[0].store == StoreAction::StoreAndResolve && !kept.encoders[0].memBacked[0]);
    // A transient-style candidate (DontCare store) of a heavy encoder: HeavyGeometry, store unchanged, no memBacked flag
    const TextureId t1 = s.rt();
    RenderPassDesc q = Scene::pass(t1, {}, LoadAction::Clear);
    q.colors[0].store = StoreAction::DontCare;
    Frame g;
    Scene::passWithDraw(g, q);
    CompileOptions o2;
    o2.tileMaxDraws = 0;
    const CompiledFrame cg = compileFrame(g, &s.res, o2);
    CHECK(cg.encoders[0].tile[0] == TileVerdict::HeavyGeometry && !cg.encoders[0].memBacked[0]);
    CHECK(cg.encoders[0].pass.colors[0].store == StoreAction::DontCare);
  }
  {  // vertex limit multiplies by the pass's layers (both eyes in one pass)
    Scene s;
    TextureDesc d;
    d.format = Format::RGBA8Unorm;
    d.width = d.height = 64;
    d.layers = 2;
    d.usage = usage::colorTarget;
    const TextureId two = s.res.createTexture(d), one = s.rt();
    RenderPassDesc p2 = Scene::pass(two, {}, LoadAction::Clear);
    p2.colors[0].store = StoreAction::DontCare;
    p2.layerCount = 2;
    RenderPassDesc p1 = Scene::pass(one, {}, LoadAction::Clear);
    p1.colors[0].store = StoreAction::DontCare;
    CompileOptions o;
    o.tileMaxVertices = 5;  // the scene's draw has 3 vertices: 3 x 1 layer fits, 3 x 2 layers does not
    Frame f2, f1;
    Scene::passWithDraw(f2, p2);
    Scene::passWithDraw(f1, p1);
    CHECK(compileFrame(f2, &s.res, o).encoders[0].tile[0] == TileVerdict::HeavyGeometry);
    CHECK(compileFrame(f1, &s.res, o).encoders[0].tileLocal(0));
    // Multiview by instancing: the draw's 2 instances already are the two views (instanceViews 2): 3 x 2 = 6 vertices, not
    // 3 x 2 x 2 = 12. Vertex amplification (1 instance, instanceViews 1) still counts 3 x 2 layers = 6.
    o.tileMaxVertices = 8;
    Frame fi, fa;
    fi.beginRenderPass(p2);
    Draw di = Scene::draw();
    di.instances = 2, di.instanceViews = 2;
    fi.draw(di);
    fi.endRenderPass();
    Scene::passWithDraw(fa, p2);
    const CompiledFrame ci = compileFrame(fi, &s.res, o), ca = compileFrame(fa, &s.res, o);
    CHECK(ci.encoders[0].tileLocal(0));
    CHECK_EQ(ci.stats.maxEncoderVertices, 6u);
    // drawViewVertices counts the views the instances hold once; a pass's one view is that share of the pass total (3 here).
    const Draw da = Scene::draw();
    CHECK_EQ(drawViewVertices(di, 2), 6u);
    CHECK_EQ(drawViewVertices(da, 2), 6u);
    CHECK_EQ(drawViewVertices(da, 1), 3u);
    CompileOptions os = o;  // the one view of a pass (what a split encoder would hold) is 3 as well: not 6
    os.splitViews = true;
    CHECK_EQ(compileFrame(fi, &s.res, os).encoders[0].splitViewVerts, 3u);
    CHECK(ca.encoders[0].tileLocal(0));
    CHECK_EQ(ca.stats.maxEncoderVertices, 6u);
    o.tileMaxVertices = 5;  // both are over a limit of 5
    CHECK(compileFrame(fi, &s.res, o).encoders[0].tile[0] == TileVerdict::HeavyGeometry);
    CHECK(compileFrame(fa, &s.res, o).encoders[0].tile[0] == TileVerdict::HeavyGeometry);
  }
  {  // an indirect draw (GPU-written counts, e.g. thousands of instanced asteroids) makes its encoder heavy with the default limits
    Scene s;
    const TextureId t1 = s.rt();
    RenderPassDesc q = Scene::pass(t1, {}, LoadAction::Clear);
    q.colors[0].store = StoreAction::DontCare;
    Frame direct, ind;
    Scene::passWithDraw(direct, q);
    ind.beginRenderPass(q);
    Draw d = Scene::draw();
    d.indirect = Indirect{s.buf(), 0};
    ind.draw(d);
    ind.endRenderPass();
    const CompiledFrame cd = compileFrame(direct, &s.res), ci = compileFrame(ind, &s.res);
    CHECK(cd.encoders[0].tileLocal(0) && cd.stats.heavyIndirect == 0 && cd.encoders[0].heavy < 1.0f);
    CHECK(ci.encoders[0].tile[0] == TileVerdict::HeavyGeometry);
    CHECK(ci.encoders[0].heavy >= 1.0f);
    CHECK_EQ(ci.stats.heavyIndirect, 1u);
  }
  {  // barriers compute -> draw (vk.perf barr_c2d): a draw reading compute output, or an indirect draw's arguments
    Scene s;
    const BufferId buf = s.buf(), args = s.buf();
    const TextureId t1 = s.rt();
    Frame f, cc, ci;
    f.dispatch(Scene::compute({write(ResourceRange::bytes(buf, 0, 256), stage::compute)}));
    Scene::passWithDraw(f, Scene::pass(t1, {}, LoadAction::Clear), {Scene::vsRead(buf)});
    CHECK_EQ(compileFrame(f, &s.res).stats.barriersComputeDraw, 1u);
    cc.dispatch(Scene::compute({write(ResourceRange::bytes(buf, 0, 256), stage::compute)}));
    cc.dispatch(Scene::compute({read(ResourceRange::bytes(buf, 0, 256), stage::compute)}));
    CHECK_EQ(compileFrame(cc, &s.res).stats.barriersComputeDraw, 0u);  // compute -> compute is not counted
    ci.dispatch(Scene::compute({write(ResourceRange::bytes(args, 0, 256), stage::compute)}));
    ci.beginRenderPass(Scene::pass(t1, {}, LoadAction::Clear));
    Draw d = Scene::draw();
    d.indirect = Indirect{args, 0};
    ci.draw(d);
    ci.endRenderPass();
    CHECK_EQ(compileFrame(ci, &s.res).stats.barriersComputeDraw, 1u);
  }
  {  // a Load of memory no frame has defined (MSAA released by the previous frame) stays on tile
    Scene s;
    TextureDesc d;
    d.format = Format::RGBA8Unorm;
    d.width = d.height = 256;
    d.samples = 2;
    d.usage = usage::colorTarget;
    const TextureId ms = s.res.createTexture(d), single = s.rt();
    RenderPassDesc p = Scene::pass(ms);  // Load
    p.colors[0].store = StoreAction::Resolve;
    p.colors[0].resolve = single;
    for (int n = 0; n < 3; ++n) {  // never written, then released by each frame in turn
      Frame f;
      Scene::passWithDraw(f, p);
      const CompiledFrame cf = compileFrame(f, &s.res);
      CHECK(cf.encoders[0].tileLocal(0) && cf.encoders[0].pass.colors[0].load == LoadAction::DontCare);
      CHECK(cf.encoders[0].pass.colors[0].store == StoreAction::Resolve);
      CHECK_EQ(cf.stats.loadsDropped, 1u);
      CHECK(!s.res.memValid(ms));
      commitMemory(cf, s.res);
      CHECK(!s.res.memValid(ms) && s.res.memValid(single));  // the resolve target holds the image
    }
    // the resolve target itself was written: loading it keeps its contents
    RenderPassDesc q = Scene::pass(single);
    q.colors[0].store = StoreAction::DontCare;
    Frame g2;
    Scene::passWithDraw(g2, q);
    const CompiledFrame cg = compileFrame(g2, &s.res);
    CHECK(cg.encoders[0].tile[0] == TileVerdict::LoadsContents && cg.encoders[0].pass.colors[0].load == LoadAction::Load);
    CHECK_EQ(cg.stats.loadsDropped, 0u);
  }
  {  // Store in frame N, Load in frame N+1: the contents are in memory, so the load stays (and again after a copy)
    Scene s;
    const TextureId a = s.rt(), b = s.rt(), src = s.rt();
    RenderPassDesc w = Scene::pass(a, {}, LoadAction::Clear);  // Store
    RenderPassDesc r = Scene::pass(a);
    r.colors[0].store = StoreAction::DontCare;
    Frame f;
    Scene::passWithDraw(f, w);
    CompiledFrame cf = compileFrame(f, &s.res);
    CHECK(cf.encoders[0].tile[0] == TileVerdict::None);
    commitMemory(cf, s.res);
    CHECK(s.res.memValid(a));
    Frame g2;
    Scene::passWithDraw(g2, r);
    cf = compileFrame(g2, &s.res);
    CHECK(cf.encoders[0].tile[0] == TileVerdict::LoadsContents && cf.encoders[0].pass.colors[0].load == LoadAction::Load);
    CHECK_EQ(cf.stats.loadsDropped, 0u);
    commitMemory(cf, s.res);  // released by that frame: the next one loads garbage
    CHECK(!s.res.memValid(a));
    cf = compileFrame(g2, &s.res);
    CHECK(cf.encoders[0].tileLocal(0) && cf.stats.loadsDropped == 1u);
    RenderPassDesc rb = Scene::pass(b);
    rb.colors[0].store = StoreAction::DontCare;
    Copy c;
    c.kind = CopyKind::TextureToTexture;
    c.srcTexture = {src, 0, 0, 1};
    c.dstTexture = {b, 0, 0, 1};
    Frame h;
    h.copy(c);
    cf = compileFrame(h, &s.res);
    commitMemory(cf, s.res);
    CHECK(s.res.memValid(b));
    Frame h2;
    Scene::passWithDraw(h2, rb);
    cf = compileFrame(h2, &s.res);
    CHECK(cf.encoders[0].tile[0] == TileVerdict::LoadsContents && cf.stats.loadsDropped == 0u);
    s.res.destroy(b);
    CHECK(!s.res.memValid(b));
  }
  {  // Device::submit commits what each frame leaves in memory (the compiler reads it next frame)
    NullBackend be;
    Device dev(be, 0, [](const std::string&) {});
    TextureDesc d;
    d.format = Format::RGBA8Unorm;
    d.width = d.height = 64;
    d.usage = usage::colorTarget | usage::sampled;
    const TextureId st = dev.createTexture(d), single = dev.createTexture(d);
    d.samples = 2;
    const TextureId ms = dev.createTexture(d);
    RenderPassDesc a = Scene::pass(ms, {}, LoadAction::Clear);
    a.colors[0].store = StoreAction::Resolve;
    a.colors[0].resolve = single;
    Frame f;
    Scene::passWithDraw(f, a);
    Scene::passWithDraw(f, Scene::pass(st, {}, LoadAction::Clear));  // Store
    CHECK(!dev.resources().memValid(ms) && !dev.resources().memValid(single) && !dev.resources().memValid(st));
    dev.submit(f);
    CHECK(!dev.resources().memValid(ms));  // released: samples never reached memory
    CHECK(dev.resources().memValid(single) && dev.resources().memValid(st));
    Frame g2;
    Scene::passWithDraw(g2, Scene::pass(ms));  // Load; store Store, so the MSAA samples reach memory now
    dev.submit(g2);
    CHECK(dev.resources().memValid(ms));
  }
  {  // never-written persistent texture: its Load is never dropped
    Scene s;
    TextureDesc d;
    d.format = Format::RGBA8Unorm;
    d.width = d.height = 256;
    d.usage = usage::colorTarget | usage::sampled;
    d.persistent = true;
    const TextureId p = s.res.createTexture(d);
    RenderPassDesc q = Scene::pass(p);
    q.colors[0].store = StoreAction::DontCare;
    Frame f;
    Scene::passWithDraw(f, q);
    const CompiledFrame cf = compileFrame(f, &s.res);
    CHECK(cf.encoders[0].tile[0] == TileVerdict::Persistent && cf.encoders[0].pass.colors[0].load == LoadAction::Load);
    CHECK_EQ(cf.stats.loadsDropped, 0u);
  }
  {  // transient + explicit Store: memory is defined only if the frame really stored it
    Scene s;
    TextureDesc d;
    d.format = Format::RGBA8Unorm;
    d.width = d.height = 256;
    d.usage = usage::colorTarget | usage::sampled;
    d.transient = true;
    const TextureId kept = s.res.createTexture(d), dropped = s.res.createTexture(d);
    Frame f;
    Scene::passWithDraw(f, Scene::pass(kept, {}, LoadAction::Clear));  // Store
    Scene::passWithDraw(f, Scene::pass(s.rt()), {Scene::sample(kept)});
    CompiledFrame cf = compileFrame(f, &s.res);
    CHECK(cf.encoders[0].pass.colors[0].store == StoreAction::Store);  // sampled later: kept
    commitMemory(cf, s.res);
    CHECK(s.res.memValid(kept));
    Frame g2;
    Scene::passWithDraw(g2, Scene::pass(dropped, {}, LoadAction::Clear));  // Store, but tile-local: dropped
    cf = compileFrame(g2, &s.res);
    CHECK(cf.encoders[0].tileLocal(0));
    commitMemory(cf, s.res);
    CHECK(!s.res.memValid(dropped));
  }
  {  // partial attachments (a mip chain, one layer of an array) keep their Load
    Scene s;
    for (const TextureId t : {s.rt(256, 2, 1), s.rt(256, 1, 2)}) {
      RenderPassDesc q = Scene::pass(t);
      q.colors[0].store = StoreAction::DontCare;
      Frame f;
      Scene::passWithDraw(f, q);
      const CompiledFrame cf = compileFrame(f, &s.res);
      CHECK(cf.encoders[0].tile[0] == TileVerdict::Partial && cf.encoders[0].pass.colors[0].load == LoadAction::Load);
      CHECK_EQ(cf.stats.loadsDropped, 0u);
    }
  }
  {  // a texture that is an attachment of two encoders keeps its Load
    Scene s;
    const TextureId t = s.rt(), u = s.rt();
    RenderPassDesc q = Scene::pass(t);
    q.colors[0].store = StoreAction::DontCare;
    Frame f;
    Scene::passWithDraw(f, q);
    Scene::passWithDraw(f, Scene::pass(u, {}, LoadAction::Clear));
    Scene::passWithDraw(f, q);
    const CompiledFrame cf = compileFrame(f, &s.res);
    CHECK_EQ(cf.encoders.size(), size_t{3});
    CHECK(cf.encoders[0].tile[0] == TileVerdict::OtherEncoder && cf.encoders[0].pass.colors[0].load == LoadAction::Load);
    CHECK(cf.encoders[2].tile[0] == TileVerdict::OtherEncoder && cf.encoders[2].pass.colors[0].load == LoadAction::Load);
    CHECK_EQ(cf.stats.loadsDropped, 0u);
  }
  {  // a destroyed texture's slot is reused: the new texture starts undefined
    Scene s;
    const TextureId a = s.rt();
    s.res.setMemValid(a, true);
    CHECK(s.res.memValid(a));
    s.res.destroy(a);
    const TextureId b = s.rt();
    CHECK_EQ(b.index, a.index);
    CHECK(!(b == a) && !s.res.memValid(b) && !s.res.memValid(a));
    s.res.setMemValid(a, true);  // a stale handle changes nothing
    CHECK(!s.res.memValid(b));
  }
}

// Random frames (fixed seed) against the invariants Metal and the tile logic rely on: every render
// encoder has one size and sample count, no texture in two slots, tile-local slots never load nor
// store, released contents are not read by a later encoder, and skipped draws are exactly those
// whose pixel-local reads no colour attachment of their encoder serves.
void testFuzz() {
  std::mt19937 rng(1234);
  auto R = [&](int n) { return int(rng() % unsigned(n)); };
  uint32_t violations = 0, merged = 0, bad = 0;
  for (int it = 0; it < 4000; ++it) {
    Scene s;
    std::vector<TextureId> tex;
    const int nt = 3 + R(4);
    for (int i = 0; i < nt; ++i) {
      TextureDesc d;
      d.format = i == nt - 1 ? Format::D32Float : R(3) ? Format::RGBA8Unorm : Format::RGBA32Float;
      d.width = d.height = R(8) ? 256 : 128;
      d.samples = R(8) ? 1 : 4;
      d.persistent = !R(10);
      d.transient = !R(4);
      d.usage = usage::sampled | (i == nt - 1 ? usage::depthTarget : usage::colorTarget);
      tex.push_back(s.res.createTexture(d));
    }
    Frame f;
    for (int p = 2 + R(4); p > 0; --p) {
      if (!R(10)) {
        f.dispatch(Scene::compute({read(ResourceRange::wholeTexture(tex[R(nt - 1)]), stage::compute)}));
        continue;
      }
      // A valid client pass: distinct textures of one size and sample count.
      RenderPassDesc pd;
      const TextureDesc* ref = nullptr;
      auto fits = [&](TextureId t) {
        const TextureDesc* d = s.res.get(t);
        bool used = pd.depth.texture == t;
        for (const Attachment& o : pd.colors) used = used || o.texture == t;
        return !used && (!ref || (d->width == ref->width && d->samples == ref->samples));
      };
      for (int a = 1 + R(3); a > 0; --a) {
        Attachment& c = pd.colors[R(4)];
        const TextureId t = tex[R(nt - 1)];
        if (c.texture || !fits(t)) continue;
        ref = s.res.get(t);
        c.texture = t;
        c.load = LoadAction(R(3));
        c.store = R(2) ? StoreAction::Store : StoreAction::DontCare;
      }
      if (R(2) && fits(tex.back()))
        pd.depth = {tex.back(), 0, 0, LoadAction(R(3)), R(2) ? StoreAction::Store : StoreAction::DontCare, {}, {}, 0, 0};
      std::vector<Access> b;
      for (int k = R(3); k > 0; --k) {
        const TextureId t = tex[R(nt)];
        b.push_back(R(2) ? pixelLocalRead(t) : Scene::sample(t));
      }
      Scene::passWithDraw(f, pd, b);
    }
    const CompiledFrame cf = compileFrame(f, &s.res);
    merged += cf.stats.passesMerged;
    bad += uint32_t(cf.badDraws.size());
    const auto& cmds = f.commands();
    // sampledOnly: only bindings read as sampled (not pixel-locally, not attachments).
    auto touches = [&](uint32_t ci, TextureId t, bool sampledOnly = false) {
      bool hit = false;
      auto chk = [&](const std::vector<Access>& v) {
        for (const Access& a : v)
          hit = hit || (a.range.kind == ResourceRange::Kind::Texture && a.range.index == t.index &&
                        (!sampledOnly || !a.pixelLocal));
      };
      if (const auto* d = std::get_if<Dispatch>(&cmds[ci])) chk(d->bindings);
      if (const auto* rp = std::get_if<RenderPass>(&cmds[ci])) {
        if (!sampledOnly) chk(attachmentAccesses(rp->desc));
        for (const Draw& d : rp->draws) chk(d.bindings);
      }
      return hit;
    };
    for (uint32_t e = 0; e < cf.encoders.size(); ++e) {
      const Encoder& en = cf.encoders[e];
      if (en.kind != Encoder::Kind::Render) continue;
      std::set<uint32_t> seen;
      const TextureDesc* ref = nullptr;
      for (int slot = 0; slot <= kMaxColorAttachments; ++slot) {
        const Attachment& a = slot == kDepthSlot ? en.pass.depth : en.pass.colors[slot];
        if (!a.texture) continue;
        const TextureDesc* d = s.res.get(a.texture);
        violations += !seen.insert(a.texture.index).second;
        if (!ref) ref = d;
        violations += d->samples != ref->samples || d->width != ref->width || d->height != ref->height;
        if (en.tileLocal(slot)) violations += a.load == LoadAction::Load || a.store != StoreAction::DontCare;
        if (a.store == StoreAction::DontCare)
          for (uint32_t later = e + 1; later < cf.encoders.size(); ++later) {
            // An encoder that starts the texture clean (Clear / DontCare) reads nothing an earlier one released: what it
            // reads of it pixel-locally is its own contents. That exemption does not cover a sampled read: it would
            // see the released texture.
            const RenderPassDesc& lp = cf.encoders[later].pass;
            bool clean = lp.depth.texture == a.texture && lp.depth.load != LoadAction::Load;
            for (const Attachment& la : lp.colors)
              clean = clean || (la.texture == a.texture && la.load != LoadAction::Load);
            for (uint32_t ci : cf.encoders[later].commands) violations += touches(ci, a.texture, clean);
          }
      }
      for (uint32_t ci : en.commands) {
        const auto& draws = std::get<RenderPass>(cmds[ci]).draws;
        for (uint32_t j = 0; j < draws.size(); ++j) {
          FetchInputs fi;
          violations += fetchInputs(draws[j], en.pass, fi) == cf.skipped(ci, j);
        }
      }
    }
  }
  std::printf("  fuzz: 4000 frames, %u passes merged, %u draws skipped, %u violations\n", merged, bad, violations);
  CHECK_EQ(violations, 0u);
  CHECK(merged > 0 && bad > 0);  // the generator reaches both paths
}

void testBindings() {
  Scene s;
  const SamplerId smp = s.res.createSampler({});
  const BufferId b = s.buf();
  const TextureId t = s.rt();
  Access cb = Scene::vsRead(b);
  cb.kind = BindKind::ConstantBuffer;
  CHECK(bindKind(cb) == BindKind::ConstantBuffer);
  CHECK(bindKind(Scene::sample(t)) == BindKind::Texture);
  CHECK(bindKind(write(ResourceRange::wholeTexture(t), stage::compute)) == BindKind::StorageTexture);
  CHECK(bindKind(samplerBinding(smp, stage::fragment)) == BindKind::Sampler);
  Frame f;
  Scene::passWithDraw(f, Scene::pass(s.rt()), {Scene::sample(t), samplerBinding(smp, stage::fragment)});
  Scene::passWithDraw(f, Scene::pass(s.rt()), {Scene::sample(t), samplerBinding(smp, stage::fragment)});
  CompiledFrame cf = compileFrame(f, &s.res);
  CHECK(cf.barriers.empty());  // samplers are immutable: never a hazard
  s.res.destroy(smp);
  cf = compileFrame(f, &s.res);
  CHECK_EQ(cf.stats.dropped, 2u);  // dead sampler drops the draw like any dead resource

  PipelineState a;
  a.vertexShader = 1;
  PipelineState st = a;
  st.stencilTest = true;
  st.stencilFront.pass = StencilOp::Replace;
  CHECK(!(a.key() == st.key()));
  PipelineState fe = a;
  fe.fetch.mask = 1;
  CHECK(!(a.key() == fe.key()));
}

// A frame shaped like ASTRO BOT's per the performance report (~45 passes, copies of per-draw
// constants between passes, clears, compute post-processing), compiled vs one-encoder-per-command.
void testAstroFrame() {
  Scene s;
  const TextureId shadow = s.rt(1024, 1, 4, Format::D32Float);  // 4 cascades as layers
  const TextureId gbufA = s.rt(1440), gbufB = s.rt(1440), gbufC = s.rt(1440, 1, 1, Format::RGBA16Float);
  const TextureId depth = s.rt(1440, 1, 1, Format::D32Float);
  const TextureId hdr = s.rt(1440, 1, 1, Format::RGBA16Float), ao = s.rt(720, 1, 1, Format::R8Unorm);
  const TextureId bloom = s.rt(720, 5, 1, Format::RG11B10Float), ldr = s.rt(1440), ui = s.rt(1440);
  const BufferId constants = s.buf(1 << 20), staging = s.buf(1 << 20), particles = s.buf(1 << 16);
  uint64_t cursor = 0;
  Frame f;
  // Per-draw constants are uploaded right before each pass that uses them (fresh ring slice).
  auto uploadAndDraw = [&](RenderPassDesc d, std::vector<Access> extra = {}) {
    f.copyBuffer(Scene::span(staging, cursor, 256), Scene::span(constants, cursor, 256));
    extra.push_back(Scene::vsRead(constants, cursor, 256));
    cursor += 256;
    Scene::passWithDraw(f, std::move(d), std::move(extra));
  };
  auto shadowPass = [&](uint16_t layer, LoadAction load) {
    RenderPassDesc d;
    d.depth.texture = shadow;
    d.depth.layer = layer;
    d.depth.load = load;
    return d;
  };
  // G-buffer (albedo, normal, emissive/motion) + depth, released after lighting: DontCare stores.
  auto gbuffer = [&](LoadAction load) {
    RenderPassDesc d = Scene::pass(gbufA, depth, load);
    d.colors[1].texture = gbufB;
    d.colors[1].load = load;
    d.colors[2].texture = gbufC;
    d.colors[2].load = load;
    for (int i = 0; i < 3; ++i) d.colors[i].store = StoreAction::DontCare;
    return d;
  };

  TextureDesc hd;
  hd.format = Format::R32Float;
  hd.width = hd.height = 720;
  hd.usage = usage::colorTarget | usage::sampled;
  hd.persistent = true;
  const TextureId depthHistory = s.res.createTexture(hd);
  f.fillBuffer(Scene::span(particles, 0, 1 << 16), 0);
  f.dispatch(Scene::compute({read(ResourceRange::wholeTexture(depthHistory), stage::compute),
                             write(ResourceRange::wholeTexture(ao), stage::compute)}));  // SSAO
  f.dispatch(Scene::compute({write(ResourceRange::wholeBuffer(particles), stage::compute)}));  // particle sim
  for (uint16_t c = 0; c < 4; ++c) {  // shadow cascades: clear + 3 batches each
    f.clear(shadowPass(c, LoadAction::Clear));
    for (int k = 0; k < 3; ++k) uploadAndDraw(shadowPass(c, LoadAction::Load));
  }
  f.clear(gbuffer(LoadAction::Clear));
  for (int k = 0; k < 12; ++k) uploadAndDraw(gbuffer(LoadAction::Load));       // opaque geometry
  // Lighting on tile: G-buffer read at its own pixel; hdr in slot 3, read-only depth for the
  // light volumes. AO is half-res: it comes from last frame's depth (persistent history), so the
  // SSAO compute does not pull this frame's depth out of the tile.
  std::vector<Access> lightInputs = {pixelLocalRead(gbufA), pixelLocalRead(gbufB), pixelLocalRead(gbufC),
                                     Scene::sample(ao)};
  for (uint16_t c = 0; c < 4; ++c) lightInputs.push_back(Scene::sample(shadow, 0, c));
  RenderPassDesc light = Scene::pass({}, depth);
  light.colors[3].texture = hdr;
  light.colors[3].load = LoadAction::Clear;
  uploadAndDraw(light, lightInputs);                                           // deferred lighting
  // Forward + particles keep depth testing: depth must survive the lighting encoder.
  for (int k = 0; k < 6; ++k)                                                  // forward + particles
    uploadAndDraw(Scene::pass(hdr, depth), {read(ResourceRange::wholeBuffer(particles), stage::vertex)});
  for (uint16_t m = 0; m < 5; ++m) {                                           // bloom down chain
    RenderPassDesc d = Scene::pass(bloom);
    d.colors[0].mip = m;
    uploadAndDraw(d, {m ? Scene::sample(bloom, m - 1) : Scene::sample(hdr)});
  }
  uploadAndDraw(Scene::pass(ldr), {Scene::sample(hdr), Scene::sample(bloom, 4)});  // tonemap
  f.clear(Scene::pass(ui));
  for (int k = 0; k < 4; ++k) uploadAndDraw(Scene::pass(ui));                 // UI layers
  uploadAndDraw(Scene::pass(ldr), {Scene::sample(ui)});                       // composite

  const CompiledFrame cf = compileFrame(f, &s.res);
  const FrameReport r = makeReport(0, f, cf);
  uint32_t passes = 0;
  for (const Command& c : f.commands()) passes += std::holds_alternative<RenderPass>(c);
  std::printf("  astro-like frame: %u passes, %u commands -> naive %u encoders, core %u encoders "
              "(%u render, %u compute), %u barriers, %u passes merged, %u copies hoisted, %u sunk\n",
              passes, r.naiveEncoders, r.naiveEncoders, r.encoders, r.renderEncoders, r.computeEncoders,
              r.barriers, cf.stats.passesMerged, cf.stats.copiesHoisted, cf.stats.copiesSunk);
  for (const auto& l : formatReport(r)) std::printf("  %s\n", l.c_str());
  CHECK(passes >= 40 && passes <= 50);
  CHECK(r.encoders * 3 < r.naiveEncoders);
  CHECK_EQ(cf.stats.copiesSunk, 0u);  // every constant upload can move ahead of its encoder
  // cascades, gbuf+light (one encoder), fwd, bloom, tonemap, ui, composite
  CHECK_EQ(r.renderEncoders, 4u + 1 + 1 + 5 + 1 + 1 + 1);
  CHECK_EQ(cf.stats.tileLocal, 3u);  // the three G-buffer colour targets; depth is used by forward
  CHECK_EQ(cf.stats.badFetch, 0u);
  for (const auto& l : formatEncoders(0, f, cf))
    if (l.find("tile=") != std::string::npos || l.find("mem=") != std::string::npos) std::printf("  %s\n", l.c_str());
}

}  // namespace

// Copies to rectangles of one texture subresource (CompileOptions::copyRects) and the many-copies diagnostics.
Copy rectCopy(BufferId src, TextureId dst, uint32_t x, uint32_t y, uint32_t w = 8, uint32_t h = 8, uint16_t mip = 0,
              const char* label = "vk.upload") {
  Copy c;
  c.kind = CopyKind::BufferToTexture;
  c.srcBuffer = {src, 0, 256, 0, 0};
  c.dstTexture.texture = dst;
  c.dstTexture.mip = mip;
  c.dstTexture.x = x, c.dstTexture.y = y, c.dstTexture.width = w, c.dstTexture.height = h;
  c.label = label;
  return c;
}

void testCopyRects() {
  Scene s;
  const TextureId t = s.rt(256, 2);
  const BufferId src = s.buf();
  CompileOptions on;
  on.copyRects = true;
  {  // 5000 small copies into one subresource, disjoint rects: a WAW chain without rects, none with them
    Frame f;
    for (uint32_t i = 0; i < 5000; ++i) f.copy(rectCopy(src, t, (i % 32) * 8, (i / 32 % 32) * 8, 8, 8, uint16_t(i / 1024 % 1)));
    // 1024 distinct 8x8 cells: copies 1024.. repeat the first ones (same rect again: WAW)
    const CompiledFrame off = compileFrame(f, nullptr, {});
    CHECK_EQ(off.encoders.size(), size_t{1});
    CHECK_EQ(off.stats.intraBarriers, 4999u);  // every copy ordered after the previous one
    CHECK(bigFrame(off));
    const CompiledFrame rects = compileFrame(f, nullptr, on);
    CHECK_EQ(rects.encoders.size(), size_t{1});
    // 1024 cells repeat every 1024 copies: a barrier at each repeat of the first cell orders every earlier copy (4 of them)
    CHECK_EQ(rects.stats.intraBarriers, 4u);
    CHECK(!bigFrame(rects));
  }
  {  // strictly disjoint rects (a 16x16 grid of 8x8 cells, different mips too): no barrier at all
    Frame f;
    for (uint32_t i = 0; i < 256; ++i) f.copy(rectCopy(src, t, (i % 16) * 8, (i / 16) * 8));
    for (uint32_t i = 0; i < 16; ++i) f.copy(rectCopy(src, t, (i % 4) * 8, (i / 4) * 8, 8, 8, 1));
    const CompiledFrame rects = compileFrame(f, nullptr, on);
    CHECK_EQ(rects.encoders.size(), size_t{1});
    CHECK(rects.barriers.empty());
    CHECK_EQ(hazards(rects, HazardKind::WAW), 0u);
    CHECK_EQ(compileFrame(f, nullptr, {}).stats.intraBarriers, 270u);  // off: 255 + 15, a chain per mip (mip 1 never conflicts with mip 0)
  }
  {  // overlapping rects keep their barrier, partially or fully, and a whole-subresource access orders against all
    Frame f;
    f.copy(rectCopy(src, t, 0, 0, 16, 16));
    f.copy(rectCopy(src, t, 8, 8, 16, 16));  // overlaps the first
    f.copy(rectCopy(src, t, 100, 100));      // disjoint from both
    f.copy(rectCopy(src, t, 8, 8, 8, 8));    // inside the second
    const CompiledFrame cf = compileFrame(f, nullptr, on);
    CHECK_EQ(hazards(cf, HazardKind::WAW), 2u);  // second after first, fourth after second
    bool b1 = false, b2 = false, b3 = false;
    for (const Barrier& b : cf.barriers) b1 |= b.item == 1, b2 |= b.item == 2, b3 |= b.item == 3;
    CHECK(b1 && !b2 && b3);
    Frame g;  // a draw into the whole subresource afterwards sees both disjoint copies
    g.copy(rectCopy(src, t, 0, 0));
    g.copy(rectCopy(src, t, 64, 64));
    Scene::passWithDraw(g, Scene::pass(t));
    const CompiledFrame cg = compileFrame(g, nullptr, on);
    CHECK_EQ(cg.barriers.size(), size_t{1});  // one barrier before the pass covers both copies
    CHECK_EQ(hazards(cg, HazardKind::WAW), 1u);
  }
  {  // a rect read (copy source) against a write of a disjoint rect: no hazard; of an overlapping rect: RAW
    Frame f;
    Copy w = rectCopy(src, t, 0, 0, 16, 16);
    Copy r;
    r.kind = CopyKind::TextureToBuffer;
    r.srcTexture.texture = t, r.srcTexture.x = 64, r.srcTexture.y = 64, r.srcTexture.width = 8, r.srcTexture.height = 8;
    r.dstBuffer = {s.buf(), 0, 256, 0, 0};
    f.copy(w);
    f.copy(r);
    CHECK(compileFrame(f, nullptr, on).barriers.empty());
    CHECK(!compileFrame(f, nullptr, {}).barriers.empty());  // off: whole subresource
    f = Frame();
    r.srcTexture.x = 8, r.srcTexture.y = 8;
    f.copy(w);
    f.copy(r);
    CHECK_EQ(hazards(compileFrame(f, nullptr, on), HazardKind::RAW), 1u);
  }
  {  // a copy inside one texture (mip chain request) keeps whole-subresource ranges
    Copy c = rectCopy(src, t, 0, 0, 8, 8, 1);
    c.kind = CopyKind::TextureToTexture;
    c.srcTexture = {t, 0, 0, 1, 0, 0, 0, 16, 16, 1};
    CHECK(!copyAccesses(c, true)[1].range.rect);
    CHECK(copyAccesses(rectCopy(src, t, 4, 4), true)[1].range.rect);
    CHECK(!copyAccesses(rectCopy(src, t, 4, 4), false)[1].range.rect);
  }
  {  // ResourceRange rect algebra
    ResourceRange a = ResourceRange::subresource(t, 0, 1, 0, 1), b = a;
    a.rect = true, a.x0 = 0, a.y0 = 0, a.x1 = 16, a.y1 = 16;
    b.rect = true, b.x0 = 16, b.y0 = 0, b.x1 = 32, b.y1 = 16;
    CHECK(!a.overlaps(b) && !b.overlaps(a));
    b.x0 = 15;
    CHECK(a.overlaps(b));
    const ResourceRange whole = ResourceRange::subresource(t, 0, 1, 0, 1);
    CHECK(whole.contains(a) && !a.contains(whole));  // a rect never stands for the whole subresource
    CHECK(a.overlaps(whole) && whole.overlaps(a));
    CHECK(a.contains(a));
  }
}

void testBigFrameLog() {
  NullBackend be;
  std::vector<std::string> log;
  Device dev(be, 0, [&](const std::string& l) { log.push_back(l); });
  TextureDesc td;
  td.format = Format::RGBA8Unorm;
  td.width = td.height = 256;
  td.usage = usage::sampled | usage::colorTarget;
  td.label = "vk.linear";
  const TextureId t = dev.createTexture(td);
  td.label = "";
  const TextureId u = dev.createTexture(td);
  const BufferId src = dev.createBuffer({256, nullptr, "src"});
  Frame f;
  for (uint32_t i = 0; i < 300; ++i) f.copy(rectCopy(src, t, (i % 16) * 8, (i / 16) * 8, 8, 8, 0, i % 3 ? "vk.upload" : "vk.linear_in"));
  for (uint32_t i = 0; i < 20; ++i) f.copy(rectCopy(src, u, 0, 0, 8, 8, 0, "vk.zeroinit"));
  log.clear();
  dev.submit(f);
  std::string line;
  for (const std::string& l : log)
    if (l.starts_with("gpu.bigframe")) line = l;
  CHECK(!line.empty());
  CHECK(line.find("intra_barriers=") != std::string::npos && line.find("hazards=") != std::string::npos);
  CHECK(line.find("by_label=vk.upload:200,vk.linear_in:100,vk.zeroinit:20") != std::string::npos);
  CHECK(line.find("top1=RGBA8") != std::string::npos && line.find("tiling=linear:copies=300:rects=300") != std::string::npos);
  CHECK(line.find("top2=") != std::string::npos && line.find("tiling=optimal:copies=20:rects=1") != std::string::npos);
  // The line is capped: 24 per run, then one in 256 frames.
  size_t lines = 0;
  for (int i = 0; i < 60; ++i) {
    log.clear();
    dev.submit(f);
    for (const std::string& l : log) lines += l.starts_with("gpu.bigframe");
  }
  CHECK_EQ(lines, size_t{23});  // the first 24 frames log (one above), the rest only every 256th
  // A small frame logs nothing.
  Frame g;
  g.copy(rectCopy(src, t, 0, 0));
  log.clear();
  dev.submit(g);
  for (const std::string& l : log) CHECK(!l.starts_with("gpu.bigframe"));
}

void testRectEnv() {
  setenv("VPVK_COPY_RECT_HAZARD", "1", 1);
  {
    NullBackend be;
    Device dev(be, 0, [](const std::string&) {});
    TextureDesc td;
    td.format = Format::RGBA8Unorm;
    td.width = td.height = 64;
    td.usage = usage::sampled;
    const TextureId t = dev.createTexture(td);
    const BufferId src = dev.createBuffer({256, nullptr, "src"});
    Frame f;
    for (uint32_t i = 0; i < 64; ++i) f.copy(rectCopy(src, t, (i % 8) * 8, (i / 8) * 8));
    dev.submit(f);
    CHECK(be.calls.back() == "submit 0 encoders=1 barriers=0");
  }
  unsetenv("VPVK_COPY_RECT_HAZARD");
  NullBackend be;
  Device dev(be, 0, [](const std::string&) {});
  TextureDesc td;
  td.format = Format::RGBA8Unorm;
  td.width = td.height = 64;
  td.usage = usage::sampled;
  const TextureId t = dev.createTexture(td);
  const BufferId src = dev.createBuffer({256, nullptr, "src"});
  Frame f;
  for (uint32_t i = 0; i < 64; ++i) f.copy(rectCopy(src, t, (i % 8) * 8, (i / 8) * 8));
  dev.submit(f);
  CHECK(be.calls.back() == "submit 0 encoders=1 barriers=63");
}

// MSAA dead-store elision (CompileOptions::msaaDeadStore): a client Store nothing reads afterwards is dropped once 3
// consecutive frames showed the next frame did not need the contents.
void testDeadStore() {
  struct Ms {
    Scene s;
    TextureId ms, other;
    Ms() {
      TextureDesc d;
      d.format = Format::RGBA8Unorm;
      d.width = d.height = 64;
      d.samples = 4;
      d.usage = usage::colorTarget;
      ms = s.res.createTexture(d);
      other = s.rt(64);
    }
    // Frame of one pass on `ms` (Store), optionally followed by a pass that reads it back.
    static CompileOptions on() {
      CompileOptions o;
      o.msaaDeadStore = true;  // opt-in
      return o;
    }
    CompiledFrame run(LoadAction first, int readBack = 0, const CompileOptions& o = on()) {
      Frame f;
      RenderPassDesc w = Scene::pass(ms, {}, first);
      w.colors[0].store = StoreAction::Store;
      Scene::passWithDraw(f, w);
      if (readBack == 1) Scene::passWithDraw(f, Scene::pass(other, {}, LoadAction::Clear), {Scene::sample(ms)});
      if (readBack == 2) {  // a Load of it in an encoder of its own
        Scene::passWithDraw(f, Scene::pass(other, {}, LoadAction::Clear));
        RenderPassDesc r = Scene::pass(ms);
        r.colors[0].store = StoreAction::DontCare;
        Scene::passWithDraw(f, r, {Scene::sample(other)});
      }
      CompiledFrame cf = compileFrame(f, &s.res, o);
      commitMemory(cf, s.res);
      return cf;
    }
  };
  {  // (a) Clear every frame: 3 dead stores seen, then the store is elided and the attachment stays on tile
    Ms t;
    for (int n = 0; n < 4; ++n) {
      const CompiledFrame cf = t.run(LoadAction::Clear);
      CHECK(cf.encoders[0].tile[0] == TileVerdict::None && cf.encoders[0].pass.colors[0].store == StoreAction::Store);
      CHECK_EQ(cf.stats.storesElided, 0u);
      CHECK(t.s.res.memValid(t.ms));
    }
    for (int n = 0; n < 3; ++n) {
      const CompiledFrame cf = t.run(LoadAction::Clear);
      CHECK(cf.encoders[0].tileLocal(0) && cf.encoders[0].pass.colors[0].store == StoreAction::DontCare);
      CHECK_EQ(cf.stats.storesElided, 1u);
      CHECK_EQ(cf.stats.elideMiss, 0u);
      CHECK(!t.s.res.memValid(t.ms));
    }
  }
  {  // (a') a DontCare first load counts as dead too; a non-MSAA texture is never touched
    Ms t;
    for (int n = 0; n < 4; ++n) t.run(LoadAction::DontCare);
    CHECK(t.run(LoadAction::DontCare).encoders[0].tileLocal(0));
    RenderPassDesc w = Scene::pass(t.other, {}, LoadAction::Clear);
    for (int n = 0; n < 6; ++n) {
      Frame g;
      Scene::passWithDraw(g, w);
      const CompiledFrame cf = compileFrame(g, &t.s.res);
      CHECK(cf.encoders[0].tile[0] == TileVerdict::None && cf.stats.storesElided == 0u && cf.storeObs.empty());
      commitMemory(cf, t.s.res);
    }
  }
  {  // (b) something later in the frame reads it (sample, or a Load in another encoder): never elided
    for (int mode = 1; mode <= 2; ++mode) {
      Ms t;
      for (int n = 0; n < 10; ++n) {
        const CompiledFrame cf = t.run(LoadAction::Clear, mode);
        CHECK(cf.encoders[0].tile[0] != TileVerdict::TileLocal);
        CHECK_EQ(cf.stats.storesElided, 0u);
        CHECK(cf.encoders[0].pass.colors[0].store == StoreAction::Store);
        if (mode == 2) CHECK(cf.encoders.size() > 1);
      }
    }
  }
  {  // (c) after the elision a frame Loads it: the miss is counted once, the texture is banned from then on
    Ms t;
    for (int n = 0; n < 7; ++n) t.run(LoadAction::Clear);
    CompiledFrame cf = t.run(LoadAction::Load);
    CHECK_EQ(cf.stats.elideMiss, 1u);
    CHECK(cf.elideMisses.size() == 1 && cf.elideMisses[0] == t.ms);
    CHECK_EQ(cf.stats.storesElided, 0u);  // this frame stores for real
    CHECK(cf.encoders[0].pass.colors[0].store == StoreAction::Store);
    CHECK(t.s.res.storeHistory(t.ms).banned);
    for (int n = 0; n < 10; ++n) {
      cf = t.run(LoadAction::Clear);
      CHECK(cf.encoders[0].tile[0] == TileVerdict::None && cf.stats.storesElided == 0u && cf.stats.elideMiss == 0u);
    }
  }
  {  // (c') a Load while the store was not yet elided is not a miss: it only restarts the count
    Ms t;
    for (int n = 0; n < 3; ++n) t.run(LoadAction::Clear);
    CHECK_EQ(t.run(LoadAction::Load).stats.elideMiss, 0u);
    CHECK(!t.s.res.storeHistory(t.ms).banned && t.s.res.storeHistory(t.ms).dead == 0);
    for (int n = 0; n < 4; ++n) CHECK_EQ(t.run(LoadAction::Clear).stats.storesElided, 0u);
    CHECK_EQ(t.run(LoadAction::Clear).stats.storesElided, 1u);
  }
  {  // (d) feature off (the default): never elided
    Ms t;
    const CompileOptions off;
    CHECK(!off.msaaDeadStore);
    for (int n = 0; n < 10; ++n) {
      const CompiledFrame cf = t.run(LoadAction::Clear, 0, off);
      CHECK(cf.encoders[0].tile[0] == TileVerdict::None && cf.stats.storesElided == 0u && cf.storeObs.empty());
    }
  }
  {  // (d') Device: off unless VPVK_MSAA_DEAD_STORE=1, and the miss reaches the log once
    for (int envOn = 0; envOn < 2; ++envOn) {
      if (envOn) setenv("VPVK_MSAA_DEAD_STORE", "1", 1);
      std::vector<std::string> log;
      NullBackend be;
      Device dev(be, 0, [&](const std::string& l) { log.push_back(l); });
      unsetenv("VPVK_MSAA_DEAD_STORE");
      TextureDesc d;
      d.format = Format::RGBA8Unorm;
      d.width = d.height = 64;
      d.samples = 4;
      d.usage = usage::colorTarget;
      const TextureId ms = dev.createTexture(d);
      auto frame = [&](LoadAction load) {
        Frame f;
        RenderPassDesc w = Scene::pass(ms, {}, load);
        w.colors[0].store = StoreAction::Store;
        Scene::passWithDraw(f, w);
        return dev.submit(f);
      };
      uint32_t elided = 0, misses = 0;
      for (int n = 0; n < 8; ++n) elided += frame(LoadAction::Clear).compile.storesElided;
      misses += frame(LoadAction::Load).compile.elideMiss;
      for (int n = 0; n < 4; ++n) misses += frame(LoadAction::Clear).compile.elideMiss;
      size_t logged = 0;
      for (const std::string& l : log) logged += l.starts_with("gpu.msaa elide_miss");
      CHECK_EQ(elided, envOn ? 4u : 0u);
      CHECK_EQ(misses, envOn ? 1u : 0u);
      CHECK_EQ(logged, envOn ? size_t{1} : size_t{0});
    }
  }
  {  // (e) elided, then a frame that touches only part of it (layer 1), then a Load of all of it: banned at the partial frame
    Scene s;
    TextureDesc d;
    d.format = Format::RGBA8Unorm;
    d.width = d.height = 64;
    d.layers = 2;
    d.samples = 4;
    d.usage = usage::colorTarget;
    const TextureId ms = s.res.createTexture(d);
    auto frame = [&](LoadAction load, bool layer1Only) {
      Frame f;
      RenderPassDesc w = Scene::pass(ms, {}, load);
      w.colors[0].store = StoreAction::Store;
      if (layer1Only) w.colors[0].layer = 1;
      else w.layerCount = 2;
      Scene::passWithDraw(f, w);
      CompiledFrame cf = compileFrame(f, &s.res, Ms::on());
      commitMemory(cf, s.res);
      return cf;
    };
    for (int n = 0; n < 4; ++n) frame(LoadAction::Clear, false);
    CHECK_EQ(frame(LoadAction::Clear, false).stats.storesElided, 1u);
    CHECK(!s.res.storeHistory(ms).banned);
    const CompiledFrame part = frame(LoadAction::Clear, true);  // verdict Partial: the elided layer 0 stays undefined
    CHECK(part.encoders[0].tile[0] == TileVerdict::None && part.stats.storesElided == 0u);
    CHECK(s.res.storeHistory(ms).banned);
    frame(LoadAction::Load, false);
    for (int n = 0; n < 10; ++n) {
      const CompiledFrame cf = frame(LoadAction::Clear, false);
      CHECK(cf.encoders[0].tile[0] == TileVerdict::None && cf.stats.storesElided == 0u);
    }
  }
  {  // the frame line carries the counters
    FrameReport r;
    r.compile.storesElided = 2, r.compile.elideMiss = 1;
    CHECK(formatReport(r)[0].find(" stores_elided=2 elide_miss=1") != std::string::npos);
  }
}

// Render-target memory: attachments that live in several encoders, TRANSIENT contract violations, the gpu.mem.rt line.
void testRtMemory() {
  auto start = [](TextureId c, TextureId d, LoadAction ld) {  // colour + depth, both released at the end of the pass
    RenderPassDesc p;
    p.colors[0] = {c, 0, 0, ld, StoreAction::DontCare, {}, {}, 0, 0};
    p.depth = {d, 0, 0, ld, StoreAction::DontCare, {}, {}, 0, 0};
    return p;
  };
  {  // the same depth in two encoders that each start clean (aliased transient targets): memoryless in both
    Scene s;
    const TextureId c0 = s.rt(), c1 = s.rt(), d = s.depthRt();
    Frame f;
    Scene::passWithDraw(f, start(c0, d, LoadAction::Clear));
    Scene::passWithDraw(f, Scene::pass(c1, {}, LoadAction::Clear));
    Scene::passWithDraw(f, start(c0, d, LoadAction::Clear));
    const CompiledFrame cf = compileFrame(f, &s.res);
    CHECK_EQ(cf.encoders.size(), size_t{3});
    for (const uint32_t e : {0u, 2u}) {
      CHECK(cf.encoders[e].tileLocal(kDepthSlot) && cf.encoders[e].tileLocal(0));
      CHECK(cf.encoders[e].pass.depth.store == StoreAction::DontCare && cf.encoders[e].pass.depth.load == LoadAction::Clear);
    }
  }
  {  // ...DontCare loads too; a later encoder that Loads makes it OtherEncoder, and the first one stores
    Scene s;
    const TextureId c0 = s.rt(), c1 = s.rt(), d = s.depthRt();
    Frame f;
    Scene::passWithDraw(f, start(c0, d, LoadAction::DontCare));
    Scene::passWithDraw(f, Scene::pass(c1, {}, LoadAction::Clear));
    RenderPassDesc again = start(c0, d, LoadAction::Clear);
    again.depth.load = LoadAction::Load;
    Scene::passWithDraw(f, again);
    const CompiledFrame cf = compileFrame(f, &s.res);
    CHECK_EQ(cf.encoders.size(), size_t{3});
    CHECK(cf.encoders[0].tile[kDepthSlot] == TileVerdict::OtherEncoder && cf.encoders[2].tile[kDepthSlot] == TileVerdict::OtherEncoder);
    CHECK(cf.encoders[0].pass.depth.store == StoreAction::Store && cf.encoders[2].pass.depth.load == LoadAction::Load);
    CHECK(cf.encoders[0].tileLocal(0) && cf.encoders[2].tileLocal(0));  // the colour starts clean in both
  }
  {  // a later encoder that stores it gets memory; the earlier clean one stays memoryless and memory ends up defined
    Scene s;
    const TextureId c0 = s.rt(), c1 = s.rt(), d = s.depthRt();
    Frame f;
    Scene::passWithDraw(f, start(c0, d, LoadAction::Clear));
    Scene::passWithDraw(f, Scene::pass(c1, {}, LoadAction::Clear));
    RenderPassDesc keep = start(c0, d, LoadAction::Clear);
    keep.depth.store = StoreAction::Store;
    Scene::passWithDraw(f, keep);
    CompiledFrame cf = compileFrame(f, &s.res);
    CHECK(cf.encoders[0].tileLocal(kDepthSlot) && cf.encoders[2].tile[kDepthSlot] == TileVerdict::None);
    commitMemory(cf, s.res);
    CHECK(s.res.memValid(d));
  }
  {  // TRANSIENT contract: loaded, then sampled, then stored and read; a clean transient is not reported
    Scene s;
    TextureDesc td = *s.res.get(s.rt());
    td.transient = true;
    const TextureId sampledLater = s.res.createTexture(td), loaded = s.res.createTexture(td), clean = s.res.createTexture(td);
    Frame f;
    Scene::passWithDraw(f, Scene::pass(sampledLater, {}, LoadAction::Clear));  // Store
    Scene::passWithDraw(f, Scene::pass(s.rt(), {}, LoadAction::Clear), {Scene::sample(sampledLater)});
    Scene::passWithDraw(f, Scene::pass(clean, {}, LoadAction::Clear));  // Store, tile-local: the contract
    Scene::passWithDraw(f, Scene::pass(loaded, {}, LoadAction::Load));
    CompiledFrame cf = compileFrame(f, &s.res);
    CHECK_EQ(cf.stats.transientViolations, 2u);
    CHECK_EQ(cf.transientViolations.size(), size_t{2});
    for (const TransientViolation& v : cf.transientViolations) {
      if (v.texture == sampledLater) CHECK(v.why == TileVerdict::SampledRead);
      else if (v.texture == loaded) CHECK(v.load == LoadAction::Load);
      else CHECK(false);
    }
    CHECK(cf.encoders[0].pass.colors[0].store == StoreAction::Store);  // sampled later: kept in this frame already
    CHECK(cf.encoders[2].tileLocal(0) && cf.encoders[2].pass.colors[0].store == StoreAction::DontCare);
    commitMemory(cf, s.res);
    CHECK(s.res.transientViolated(sampledLater) && s.res.transientViolated(loaded) && !s.res.transientViolated(clean));
    // From now on their explicit Store is kept (memory), while the clean one keeps dropping it.
    Frame g;
    Scene::passWithDraw(g, Scene::pass(sampledLater, {}, LoadAction::Clear));
    Scene::passWithDraw(g, Scene::pass(clean, {}, LoadAction::Clear));
    cf = compileFrame(g, &s.res);
    CHECK_EQ(cf.encoders.size(), size_t{2});
    CHECK(cf.encoders[0].tile[0] == TileVerdict::None && cf.encoders[0].pass.colors[0].store == StoreAction::Store);
    CHECK(cf.encoders[1].tileLocal(0));
    CHECK_EQ(cf.stats.transientViolations, 0u);  // reported once
    commitMemory(cf, s.res);
    CHECK(s.res.memValid(sampledLater) && !s.res.memValid(clean));
    s.res.destroy(loaded);
    CHECK(!s.res.transientViolated(loaded));
  }
  {  // the contents a tile-local frame dropped, loaded by a later frame: reported although that load was dropped too
    Scene s;
    TextureDesc td = *s.res.get(s.rt());
    td.transient = true;
    const TextureId t = s.res.createTexture(td);
    Frame f;
    Scene::passWithDraw(f, Scene::pass(t, {}, LoadAction::Clear));
    CompiledFrame cf = compileFrame(f, &s.res);
    CHECK(cf.encoders[0].tileLocal(0) && cf.transientViolations.empty());
    commitMemory(cf, s.res);
    CHECK(!s.res.memValid(t));
    Frame g;
    Scene::passWithDraw(g, Scene::pass(t, {}, LoadAction::Load));
    cf = compileFrame(g, &s.res);
    CHECK(cf.encoders[0].tileLocal(0) && cf.encoders[0].pass.colors[0].load == LoadAction::DontCare);
    CHECK_EQ(cf.transientViolations.size(), size_t{1});
    commitMemory(cf, s.res);
    Frame h;  // now an ordinary texture: the Store the client records is kept
    Scene::passWithDraw(h, Scene::pass(t, {}, LoadAction::Clear));
    cf = compileFrame(h, &s.res);
    CHECK(cf.encoders[0].pass.colors[0].store == StoreAction::Store && !cf.encoders[0].tileLocal(0));
  }
  {  // Device: gpu.mem.rt (compiler verdicts of the attachments, only when it changes) and gpu.mem.transient_violation
    NullBackend be;
    std::vector<std::string> log;
    Device dev(be, 0, [&](const std::string& l) { log.push_back(l); });
    dev.setRtMemLogMs(0);
    auto lines = [&](const char* prefix) {
      std::vector<std::string> out;
      for (const std::string& l : log)
        if (l.starts_with(prefix)) out.push_back(l);
      return out;
    };
    TextureDesc td;
    td.format = Format::RGBA8Unorm;
    td.width = td.height = 512;
    td.usage = usage::colorTarget | usage::sampled;
    const TextureId tileColor = dev.createTexture(td), stored = dev.createTexture(td);
    td.transient = true;
    const TextureId transient = dev.createTexture(td);
    auto released = [&](TextureId t) { return Scene::pass(t, {}, LoadAction::Clear); };
    Frame f;
    {
      RenderPassDesc p = released(tileColor);
      p.colors[0].store = StoreAction::DontCare;
      Scene::passWithDraw(f, p);
    }
    Scene::passWithDraw(f, released(stored));  // client Store: needs memory
    Scene::passWithDraw(f, released(transient), {});
    dev.submit(f);
    auto rt = lines("gpu.mem.rt");
    CHECK_EQ(rt.size(), size_t{1});
    // 512x512x4 B = 1 MiB each: tileColor and the transient one are memoryless, `stored` is not (verdict None).
    CHECK(!rt.empty() && rt[0] == "gpu.mem.rt memoryless_n=2 memoryless_mib_saved=2.0 mat_n=1 mat_mib=1.0 transient_violation=0 mat_why=None:1/1.0");
    dev.submit(f);
    CHECK_EQ(lines("gpu.mem.rt").size(), size_t{1});  // unchanged: no new line
    // The transient texture is sampled by a later pass: violation, it needs memory.
    Frame g;
    Scene::passWithDraw(g, released(transient));
    Scene::passWithDraw(g, released(tileColor), {Scene::sample(transient)});
    dev.submit(g);
    const auto viol = lines("gpu.mem.transient_violation");
    CHECK_EQ(viol.size(), size_t{1});
    CHECK(!viol.empty() && viol[0].find("why=SampledRead") != std::string::npos && viol[0].find("512x512x1 s1") != std::string::npos);
    rt = lines("gpu.mem.rt");
    CHECK_EQ(rt.size(), size_t{2});
    CHECK(rt.size() == 2 && rt[1].find("transient_violation=1") != std::string::npos &&
          rt[1].find("mat_why=None:2/2.0,SampledRead:1/1.0") != std::string::npos);
    dev.destroy(transient);
    dev.destroy(stored);
    dev.destroy(tileColor);
    Frame empty;
    dev.submit(empty);
    rt = lines("gpu.mem.rt");
    CHECK(rt.back().find("memoryless_n=0") != std::string::npos && rt.back().find("mat_n=0") != std::string::npos);
  }
  {  // Device::markTransientViolated: a transient texture the client reads outside the frame (VPKTAA's eye depth) keeps its Store
    NullBackend be;
    std::vector<std::string> log;
    Device dev(be, 0, [&](const std::string& l) { log.push_back(l); });
    dev.setRtMemLogMs(0);
    TextureDesc td;
    td.format = Format::RGBA8Unorm;
    td.width = td.height = 512;
    td.usage = usage::colorTarget | usage::sampled;
    td.transient = true;
    const TextureId plain = dev.createTexture(td), marked = dev.createTexture(td);
    dev.markTransientViolated(marked);
    CHECK(dev.resources().transientViolated(marked) && !dev.resources().transientViolated(plain));
    Frame f;
    Scene::passWithDraw(f, Scene::pass(plain, {}, LoadAction::Clear));
    Scene::passWithDraw(f, Scene::pass(marked, {}, LoadAction::Clear));
    dev.submit(f);
    std::string rt;
    for (const std::string& l : log)
      if (l.starts_with("gpu.mem.rt")) rt = l;
    // `plain` stays memoryless (its Store is dropped with the tile), `marked` is an ordinary texture with memory.
    CHECK(rt.find("memoryless_n=1 ") != std::string::npos && rt.find("mat_n=1 ") != std::string::npos);
    dev.destroy(marked);
    dev.destroy(plain);
  }
}

// ScopedPool is a no-op off Apple platforms: it must still nest, and never be copied (one push, one pop).
void testScopedPool() {
  static_assert(!std::is_copy_constructible_v<ScopedPool> && !std::is_copy_assignable_v<ScopedPool>);
  ScopedPool outer;
  { ScopedPool inner; }
  CHECK(true);
}

// Background gate: commits park while the app is in the background, resume on foreground / release / shutdown.
void testBackground() {
  using namespace std::chrono_literals;
  auto until = [](auto pred) {  // polls up to 5 s
    for (int i = 0; i < 5000 && !pred(); ++i) std::this_thread::sleep_for(1ms);
    return pred();
  };
  std::mutex logMutex;
  std::vector<std::string> lines;
  auto capture = [&](BackgroundGate& g) {
    g.setLog([&](const std::string& l) {
      std::lock_guard lock(logMutex);
      lines.push_back(l);
    });
  };
  {  // foreground: never parks
    BackgroundGate g;
    g.waitForeground();
    CHECK(!g.active() && g.blockedTotal() == 0);
  }
  {  // background: a non-main thread parks, foreground releases it; one log line per transition
    BackgroundGate g;
    capture(g);
    g.set(true);  // this thread is "main"
    g.set(true);  // no second transition
    CHECK(g.active());
    std::atomic<bool> done{false};
    std::thread t([&] { g.waitForeground(), done = true; });
    CHECK(until([&] { return g.blockedTotal() == 1; }));
    std::this_thread::sleep_for(50ms);
    CHECK(!done.load());
    g.set(false);
    CHECK(until([&] { return done.load(); }));
    t.join();
    g.set(false);
    CHECK(!g.active());
    std::lock_guard lock(logMutex);
    CHECK_EQ(lines.size(), 2u);
    if (lines.size() == 2) {
      CHECK(lines[0] == "[vpgpu] bg paused");
      CHECK(lines[1] == "[vpgpu] bg resumed blocked=1");
    }
  }
  {  // shutdown releases the parked threads and the gate never parks again
    BackgroundGate g;
    capture(g);
    g.set(true);
    std::atomic<int> done{0};
    std::vector<std::thread> ts;
    for (int i = 0; i < 3; ++i) ts.emplace_back([&] { g.waitForeground(), ++done; });
    CHECK(until([&] { return g.blockedTotal() == 3; }));
    CHECK_EQ(done.load(), 0);
    g.shutdown();
    CHECK(until([&] { return done.load() == 3; }));
    for (auto& t : ts) t.join();
    CHECK(!g.active());
    g.set(false);
    g.set(true);
    std::thread late([&] { g.waitForeground(), ++done; });  // returns at once
    CHECK(until([&] { return done.load() == 4; }));
    late.join();
  }
  {  // release wakes the threads parked now only; the pause goes on
    BackgroundGate g;
    capture(g);
    g.set(true);
    std::atomic<int> done{0};
    std::thread a([&] { g.waitForeground(), ++done; });
    CHECK(until([&] { return g.blockedTotal() == 1; }));
    g.release();
    CHECK(until([&] { return done.load() == 1; }));
    a.join();
    CHECK(g.active());
    std::thread b([&] { g.waitForeground(), ++done; });
    CHECK(until([&] { return g.blockedTotal() == 2; }));
    CHECK_EQ(done.load(), 1);
    g.set(false);
    CHECK(until([&] { return done.load() == 2; }));
    b.join();
  }
  {  // the main thread is never parked: it passes and is counted
    BackgroundGate g;
    capture(g);
    g.set(true);
    g.waitForeground();
    g.waitForeground();
    CHECK(g.mainThreadPassed() == 2 && g.blockedTotal() == 0);
  }
  {  // C API on the process-wide gate
    CHECK(vpgpu_is_background() == 0);
    vpgpu_set_background(1);
    CHECK(vpgpu_is_background() == 1);
    vpgpu_set_background(0);
    CHECK(vpgpu_is_background() == 0);
    backgroundGate().waitForeground();
  }
}

// CompileOptions::splitViews: a two-layer pass of amplified draws becomes two single-view encoders.
void testViewSplit() {
  // The eye pass: 4x MSAA colour + depth, two layers, resolved into a two-layer image; `n` amplified draws of 3 vertices.
  struct Eye {
    Scene s;
    TextureId color, depth, resolve;
    RenderPassDesc pass;
    explicit Eye(uint8_t samples = 4) {
      TextureDesc d;
      d.format = Format::RGBA8Unorm;
      d.width = d.height = 64;
      d.layers = 2;
      d.samples = samples;
      d.usage = usage::colorTarget;
      color = s.res.createTexture(d);
      d.format = Format::D32Float;
      d.usage = usage::depthTarget;
      depth = s.res.createTexture(d);
      d.format = Format::RGBA8Unorm;
      d.samples = 1;
      d.usage = usage::colorTarget | usage::sampled;
      resolve = s.res.createTexture(d);
      pass = Scene::pass(color, depth, LoadAction::Clear);
      pass.layerCount = 2;
      pass.colors[0].store = StoreAction::Resolve;
      pass.colors[0].resolve = resolve;
      pass.depth.store = StoreAction::DontCare;
    }
    Frame frame(uint32_t draws, const std::function<void(Draw&)>& edit = {}) {
      Frame f;
      f.beginRenderPass(pass);
      for (uint32_t i = 0; i < draws; ++i) {
        Draw d = Scene::draw();
        d.amplified = true;
        if (edit) edit(d);
        f.draw(d);
      }
      f.endRenderPass();
      return f;
    }
  };
  CompileOptions on;
  on.splitViews = true;
  on.tileMaxVertices = 8;  // only a pass that is HeavyGeometry whole is split: 2 draws x 3 vertices x 2 layers = 12 is, 1 draw (6) is not
  {  // off: nothing changes, nothing is evaluated
    Eye e;
    const Frame f = e.frame(2);
    const CompiledFrame cf = compileFrame(f, &e.s.res);
    CHECK_EQ(cf.encoders.size(), 1u);
    CHECK(cf.encoders[0].view == -1 && !cf.encoders[0].splitWhy && !cf.stats.splitViews && cf.stats.splitEncoders == 0);
  }
  {  // on: two encoders, same commands, views 0 and 1; attachments tile-local in both; no barrier between them
    Eye e;
    const Frame f = e.frame(2);
    const CompiledFrame cf = compileFrame(f, &e.s.res, on);
    CHECK_EQ(cf.encoders.size(), 2u);
    CHECK(cf.encoders[0].view == 0 && cf.encoders[1].view == 1);
    CHECK(cf.encoders[0].commands == cf.encoders[1].commands && cf.encoders[0].passes == 1 && cf.encoders[1].passes == 1);
    CHECK(cf.encoders[1].reason == CutReason::ViewSplit && cf.encoders[0].reason == CutReason::FrameStart);
    CHECK(cf.stats.splitViews && cf.stats.splitEncoders == 1 && cuts(cf, CutReason::ViewSplit) == 1);
    CHECK(std::string(cf.encoders[0].splitWhy) == "ok" && cf.encoders[0].splitSig && cf.encoders[0].splitSig == cf.encoders[1].splitSig);
    for (const Encoder& enc : cf.encoders) {
      CHECK(enc.pass.layerCount == 2);  // the backend renders layer `view` of it
      CHECK(enc.tileLocal(0) && enc.tileLocal(kDepthSlot));  // (the unsplit pass is HeavyGeometry below)
      CHECK(enc.pass.colors[0].store == StoreAction::Resolve && enc.pass.colors[0].load == LoadAction::Clear);
    }
    CHECK(cf.barriers.empty());  // the two encoders write different layers
    CHECK_EQ(cf.stats.hazards[int(HazardKind::WAW)], 0u);
  }
  {  // the heavy test counts one view: 2 draws x 3 vertices = 6 per view, 12 for both
    Eye e;
    const Frame f = e.frame(2);
    CompileOptions o = on;
    const CompiledFrame split = compileFrame(f, &e.s.res, o);
    CHECK(split.encoders[0].tileLocal(0) && split.encoders[1].tileLocal(0));
    CHECK_EQ(split.stats.maxEncoderVertices, 6u);
    o.splitViews = false;
    const CompiledFrame whole = compileFrame(f, &e.s.res, o);
    CHECK_EQ(whole.encoders.size(), 1u);
    CHECK(whole.encoders[0].tile[0] == TileVerdict::HeavyGeometry);
    CHECK_EQ(whole.stats.maxEncoderVertices, 12u);
  }
  {  // a later pass sampling the resolve target waits for both encoders
    Eye e;
    Frame f = e.frame(2);
    Scene::passWithDraw(f, Scene::pass(e.s.rt(), {}, LoadAction::Clear), {Scene::sample(e.resolve)});
    const CompiledFrame cf = compileFrame(f, &e.s.res, on);
    CHECK_EQ(cf.encoders.size(), 3u);
    CHECK(cf.encoders[0].view == 0 && cf.encoders[1].view == 1 && cf.encoders[2].view == -1);
    CHECK_EQ(cf.barriers.size(), 1u);
    CHECK(cf.barriers[0].encoder == 2);
    CHECK_EQ(cf.stats.hazards[int(HazardKind::RAW)], 1u);
  }
  {  // not eligible: the reason is named, the encoder stays whole
    auto blocked = [&](const char* why, const std::function<void(Eye&, Draw&)>& edit, bool rateMap = false) {
      Eye e(rateMap ? 1 : 4);
      if (rateMap) e.pass.rateMap = std::make_shared<RateMap>();
      const Frame f = e.frame(2, [&](Draw& d) { edit(e, d); });
      const CompiledFrame cf = compileFrame(f, &e.s.res, on);
      CHECK_EQ(cf.encoders.size(), 1u);
      CHECK(cf.encoders[0].view == -1 && cf.stats.splitEncoders == 0);
      CHECK(cf.encoders[0].splitWhy && std::string(cf.encoders[0].splitWhy) == why);
    };
    blocked("not_amplified", [](Eye&, Draw& d) { d.amplified = false; });
    blocked("resource_write", [](Eye& e, Draw& d) { d.bindings.push_back(write(ResourceRange::bytes(e.s.buf(), 0, 64), stage::fragment)); });
    blocked("occlusion_query", [](Eye& e, Draw& d) { d.visibility = BufferSpan{e.s.buf(), 0, 8, 0, 0}; });
    blocked("framebuffer_fetch", [](Eye& e, Draw& d) {
      Access a = read(ResourceRange::subresource(e.color, 0, 1, 0, 2), stage::fragment);
      a.pixelLocal = true;
      d.bindings.push_back(a);
    });
    blocked("rate_map", [](Eye&, Draw&) {}, true);  // (a 1x foveated pass: gpu/metal2 would foveate it)
    // read-only bindings (constant buffers, sampled textures) are fine
    Eye e;
    const Frame f = e.frame(2, [&](Draw& d) { d.bindings.push_back(Scene::vsRead(e.s.buf())); });
    CHECK_EQ(compileFrame(f, &e.s.res, on).encoders.size(), 2u);
  }
  {  // a rate map with multisampled attachments is not foveated by gpu/metal2 (decline "msaa"): split like any other pass
    Eye e;
    e.pass.rateMap = std::make_shared<RateMap>();
    const Frame f = e.frame(2);
    const CompiledFrame cf = compileFrame(f, &e.s.res, on);
    CHECK_EQ(cf.encoders.size(), 2u);
    CHECK(cf.encoders[0].pass.rateMap && cf.encoders[1].pass.rateMap);
  }
  {  // a view that is still heavy by geometry is not split; by draw count alone it is (half the geometry still helps the tiler)
    Eye e;
    const Frame f = e.frame(2);
    CompileOptions o = on;
    o.tileMaxVertices = 5;  // 6 per view
    const CompiledFrame geo = compileFrame(f, &e.s.res, o);
    CHECK_EQ(geo.encoders.size(), 1u);
    CHECK(std::string(geo.encoders[0].splitWhy) == "still_heavy" && geo.encoders[0].splitViewVerts == 6 && geo.encoders[0].splitHeavyAfter);
    o.tileMaxVertices = 8;
    o.tileMaxDraws = 1;  // 2 draws: heavy whole and after (by draws), still split
    const CompiledFrame draws = compileFrame(f, &e.s.res, o);
    CHECK_EQ(draws.encoders.size(), 2u);
    CHECK(draws.encoders[0].splitHeavyAfter && draws.encoders[0].tile[0] == TileVerdict::HeavyGeometry);
    CHECK(!compileFrame(f, &e.s.res, on).encoders[0].splitHeavyAfter);
  }
  {  // in-pass clears do not block: each encoder clears the rects of its view
    Eye e;
    Frame f;
    f.beginRenderPass(e.pass);
    Draw c;
    c.clear = std::make_shared<ClearRects>();
    f.draw(c);
    for (int i = 0; i < 2; ++i) {
      Draw d = Scene::draw();
      d.amplified = true;
      f.draw(d);
    }
    f.endRenderPass();
    const CompiledFrame cf = compileFrame(f, &e.s.res, on);
    CHECK_EQ(cf.encoders.size(), 2u);
    CHECK_EQ(cf.stats.maxEncoderVertices, 6u);  // the clear counts no vertices
  }
  {  // eligible but not heavy whole (1 draw: 6 view-vertices <= 8): one encoder, the reason says so
    Eye e;
    const Frame f = e.frame(1);
    const CompiledFrame cf = compileFrame(f, &e.s.res, on);
    CHECK_EQ(cf.encoders.size(), 1u);
    CHECK(cf.encoders[0].view == -1 && cf.stats.splitEncoders == 0);
    CHECK(cf.encoders[0].splitWhy && std::string(cf.encoders[0].splitWhy) == "not_heavy");
    CHECK(cf.encoders[0].tileLocal(0));
  }
  {  // one pass of a frame that is not eligible does not stop another: only two-layer passes are looked at
    Eye e;
    Frame f = e.frame(2);
    Scene::passWithDraw(f, Scene::pass(e.s.rt(), {}, LoadAction::Clear));
    const CompiledFrame cf = compileFrame(f, &e.s.res, on);
    CHECK_EQ(cf.encoders.size(), 3u);
    CHECK(cf.encoders[2].view == -1 && !cf.encoders[2].splitWhy);
  }
  {  // attachments that cannot give a slice per view: a one-layer image under a two-layer pass, a resolve target with no room
    Eye e;
    TextureDesc d;
    d.format = Format::RGBA8Unorm;
    d.width = d.height = 64;
    d.samples = 4;
    d.usage = usage::colorTarget;
    e.pass.colors[0].texture = e.s.res.createTexture(d);  // layers = 1
    const Frame f = e.frame(1);
    const CompiledFrame cf = compileFrame(f, &e.s.res, on);
    CHECK_EQ(cf.encoders.size(), 1u);
    CHECK(std::string(cf.encoders[0].splitWhy) == "attachment_layers");
  }
  {  // Device: the run-time switch is read every frame; one gpu.split line per kind of pass, and a line when the mode changes
    NullBackend be;
    std::vector<std::string> log;
    setenv("VPVK_TILE_MAX_VERTICES", "4", 1);  // the one draw is heavy whole (6 view-vertices), not per view (3)
    Device dev(be, 0, [&](const std::string& l) { log.push_back(l); });
    unsetenv("VPVK_TILE_MAX_VERTICES");
    TextureDesc d;
    d.format = Format::RGBA8Unorm;
    d.width = d.height = 64;
    d.layers = 2;
    d.samples = 4;
    d.usage = usage::colorTarget;
    const TextureId color = dev.createTexture(d);
    RenderPassDesc p = Scene::pass(color, {}, LoadAction::Clear);
    p.layerCount = 2;
    p.colors[0].store = StoreAction::DontCare;
    p.label = "eye pass";
    Frame f;
    f.beginRenderPass(p);
    Draw dr = Scene::draw();
    dr.amplified = true;
    f.draw(dr);
    f.endRenderPass();
    auto count = [&](const char* what) {
      size_t n = 0;
      for (const std::string& l : log) n += l.find(what) != std::string::npos;
      return n;
    };
    const int before = splitViewsSwitch().load();
    splitViewsSwitch().store(0);
    FrameReport off = dev.submit(f);
    CHECK_EQ(off.encoders, 1u);
    CHECK_EQ(count("gpu.split pass="), 0u);
    splitViewsSwitch().store(1);
    FrameReport on1 = dev.submit(f);
    CHECK_EQ(on1.encoders, 2u);
    dev.submit(f);
    CHECK_EQ(count("gpu.split pass="), 1u);
    CHECK_EQ(count("eligible=1 reason=ok label=eye_pass view_verts=3 draws=1 heavy_after=0"), 1u);
    CHECK_EQ(count("gpu.split mode=1"), 1u);
    bool frameLine = false;
    for (const std::string& l : log) frameLine |= l.find("gpu.frame") == 0 && l.find(" split=1 split_enc=1") != std::string::npos;
    CHECK(frameLine);
    splitViewsSwitch().store(0);
    CHECK_EQ(dev.submit(f).encoders, 1u);
    CHECK_EQ(count("gpu.split mode=0"), 1u);  // the switch back (a Device starts with the mode off: nothing to log)
    splitViewsSwitch().store(before);
  }
}

// tileMaxVerticesSwitch(): the heavy-geometry limit changes at run time (next submit), 0 = the start value; a value above the
// start value is tried on one canary frame first.
void testTileHot() {
  NullBackend be;
  std::vector<std::string> log;
  setenv("VPVK_TILE_MAX_VERTICES", "2", 1);  // start value: the scene's 3-vertex draw is heavy
  Device dev(be, 0, [&](const std::string& l) { log.push_back(l); });
  unsetenv("VPVK_TILE_MAX_VERTICES");
  TextureDesc d;
  d.format = Format::RGBA8Unorm;
  d.width = d.height = 64;
  d.usage = usage::colorTarget;
  const TextureId color = dev.createTexture(d);
  RenderPassDesc p = Scene::pass(color, {}, LoadAction::Clear);
  p.colors[0].store = StoreAction::DontCare;
  Frame f, empty;
  f.beginRenderPass(p);
  f.draw(Scene::draw());  // 3 vertices
  f.endRenderPass();
  auto count = [&](const char* what) {
    size_t n = 0;
    for (const std::string& l : log) n += l.find(what) != std::string::npos;
    return n;
  };
  setTileMaxVertices(0);
  FrameReport r0 = dev.submit(f);  // f=0
  CHECK_EQ(r0.compile.tileMaxVertices, 2u);
  CHECK_EQ(r0.compile.heavyEncoders, 1u);
  CHECK_EQ(count("gpu.tile max_vertices="), 0u);  // (the start line says heavy_max_vertices)
  // Below the start value: stricter, applied at once, no canary.
  setTileMaxVertices(1);
  CHECK_EQ(dev.submit(f).compile.tileMaxVertices, 1u);  // f=1
  CHECK_EQ(tileProbeState().load(), int(kProbeWaiting));
  // Above it: waits for a frame the start value would call heavy; the others compile with the start value's result.
  setTileMaxVertices(5000);
  CHECK(!tileMaxVerticesReverted().load());
  FrameReport idle = dev.submit(empty);  // f=2: no pass, nothing for the raised value to change
  CHECK_EQ(idle.compile.tileRaised, 0u);
  CHECK_EQ(tileProbeState().load(), int(kProbeWaiting));
  FrameReport canary = dev.submit(f);  // f=3
  CHECK_EQ(canary.compile.tileMaxVertices, 5000u);
  CHECK_EQ(canary.compile.tileRaised, 1u);
  CHECK_EQ(canary.compile.heavyEncoders, 0u);
  CHECK_EQ(tileProbeState().load(), int(kProbeProbing));
  CHECK_EQ(tileProbeFrame().load(), 3u);
  CHECK_EQ(count("gpu.tile probe start"), 1u);
  // In flight: the following frames use the start value again.
  FrameReport after = dev.submit(f);  // f=4
  CHECK_EQ(after.compile.tileMaxVertices, 2u);
  CHECK_EQ(after.compile.tileRaised, 0u);
  CHECK_EQ(after.compile.heavyEncoders, 1u);
  // A frame other than the canary completing proves nothing; the canary completing clean accepts the value.
  CHECK(!tileProbeCompleted(4));
  CHECK(tileProbeCompleted(3) && tileProbeState().load() == int(kProbeAccepted));
  FrameReport run = dev.submit(f);  // f=5
  CHECK_EQ(run.compile.tileMaxVertices, 5000u);
  CHECK_EQ(run.compile.tileRaised, 1u);
  bool frameLine = false;
  for (const std::string& l : log)
    frameLine |= l.find("gpu.frame") == 0 && l.find(" tile_max=5000 heavy_enc=0 tile_raised=1") != std::string::npos;
  CHECK(frameLine);
  // A raised frame refused by the GPU later: back to the start value, flag set, nothing more in effect.
  CHECK_EQ(tileRaisedFailed(), int(kRaisedRevert));
  CHECK(tileMaxVerticesReverted().load() == 1 && tileMaxVerticesSwitch().load() == 0);
  CHECK_EQ(tileRaisedFailed(), int(kRaisedNone));
  CHECK_EQ(dev.submit(f).compile.tileMaxVertices, 2u);
  // A canary the GPU refuses: probe fail, the value is not applied again until the next set; that set clears the flag.
  setTileMaxVertices(5000);
  CHECK(!tileMaxVerticesReverted().load());
  dev.submit(f);
  CHECK_EQ(tileProbeState().load(), int(kProbeProbing));
  CHECK_EQ(tileRaisedFailed(), int(kRaisedProbeFail));
  CHECK_EQ(dev.submit(f).compile.tileMaxVertices, 2u);
  CHECK_EQ(tileProbeState().load(), int(kProbeFailed));
  setTileMaxVertices(0);
  CHECK_EQ(dev.submit(f).compile.tileMaxVertices, 2u);
  tileMaxVerticesReverted().store(0);
}

// ActivityGauge: background work (prewarm, archive harvest) waits for the foreground compiles to be quiet.
void testActivity() {
  using namespace std::chrono_literals;
  const auto t0 = ActivityGauge::Clock::now() + 1h;  // nonzero, deterministic times
  const auto quiet = 1500ms;
  ActivityGauge g;
  CHECK(g.idleFor(quiet, t0));  // never used: quiet
  g.enter();
  CHECK(g.active() == 1 && !g.idleFor(quiet, t0 + 1h));  // running: never idle
  g.enter();
  g.leave(t0);
  CHECK(g.active() == 1 && !g.idleFor(quiet, t0 + 1h));
  g.leave(t0 + 100ms);
  CHECK(g.active() == 0);
  CHECK(!g.idleFor(quiet, t0 + 100ms + 1499ms));  // ended too recently
  CHECK(g.idleFor(quiet, t0 + 100ms + 1500ms));
  {
    ActivityGauge::Scope off(g, false);  // does nothing
    CHECK(g.active() == 0);
    ActivityGauge::Scope on(g);
    CHECK(g.active() == 1);
  }
  CHECK(g.active() == 0 && !g.idleFor(quiet));  // the scope's end is "now"
}

int main(int argc, char** argv) {
  const std::vector<std::pair<const char*, std::function<void()>>> groups = {
      {"resources", testResources}, {"commands", testCommands}, {"merge", testMerge},
      {"copies", testCopies},       {"hazards", testHazards},   {"subresources", testSubresources},
      {"cuts", testCuts},           {"diagnostics", testDiagnostics}, {"pipelines", testPipelines},
      {"device", testDevice},       {"tile", testTile},         {"bindings", testBindings},
      {"astro", testAstroFrame},    {"fuzz", testFuzz},         {"copyrects", testCopyRects},
      {"bigframe", testBigFrameLog}, {"rectenv", testRectEnv}, {"mipchain", testMipChain},
      {"deadstore", testDeadStore}, {"rtmemory", testRtMemory}, {"scopedpool", testScopedPool},
      {"background", testBackground}, {"viewsplit", testViewSplit}, {"tilehot", testTileHot}, {"activity", testActivity}};
  bool ran = false;
  for (const auto& [name, fn] : groups) {
    if (argc > 1 && std::strcmp(argv[1], name) != 0) continue;
    std::printf("[%s]\n", name);
    fn();
    ran = true;
  }
  if (!ran) std::printf("unknown group\n");
  std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
  return failures || !ran ? 1 : 0;
}
