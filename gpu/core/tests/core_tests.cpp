// gpu/core unit tests on the null backend. Usage: vpgpu_core_tests [group]  (no group = all).
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "vpgpu/core/device.h"

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
        log[3].starts_with("gpu.time f=0"));
  dev.destroy(b);
  dev.destroy(b);  // double destroy is ignored
  CHECK(be.calls[0] == "createTexture 0 RGBA8Unorm 1x1");
  CHECK(be.calls[1] == "createBuffer 0 size=64 guest=true");
  CHECK(be.calls[2] == "submit 0 encoders=1 barriers=0");
  CHECK(be.calls.size() == 4 && be.calls[3] == "destroyBuffer 0");
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

void testTile() {
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
  {  // different sample count: no fetch merge, contents kept
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
    CHECK_EQ(cf.stats.tileLocal, 1u);  // only depth, which nothing reads afterwards
    CHECK(cf.encoders[0].tileLocal(kDepthSlot));
    CHECK(cf.encoders[0].tile[0] == TileVerdict::OtherEncoder && cf.encoders[0].pass.colors[0].store == StoreAction::Store);
  }
  {  // pixel-local read of a texture never rendered this frame: reported, treated as a plain read
    Scene s;
    const TextureId a = s.rt(), b = s.rt();
    Frame f;
    Scene::passWithDraw(f, Scene::pass(a), {pixelLocalRead(b)});
    const CompiledFrame cf = compileFrame(f, &s.res);
    CHECK_EQ(cf.stats.badFetch, 1u);
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
    const CompiledFrame cf = compileFrame(f, &s.res);
    // StoreAndResolve keeps the samples: client asked for them, not a candidate.
    CHECK(cf.encoders[0].tile[0] == TileVerdict::None);
    CHECK_EQ(hazards(cf, HazardKind::RAW), 1u);  // resolve target -> sampled
    p.colors[0].store = StoreAction::Resolve;
    Frame g2;
    Scene::passWithDraw(g2, p);
    const CompiledFrame cg = compileFrame(g2, &s.res);
    CHECK(cg.encoders[0].tileLocal(0) && cg.encoders[0].pass.colors[0].store == StoreAction::Resolve);
  }
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

int main(int argc, char** argv) {
  const std::vector<std::pair<const char*, std::function<void()>>> groups = {
      {"resources", testResources}, {"commands", testCommands}, {"merge", testMerge},
      {"copies", testCopies},       {"hazards", testHazards},   {"subresources", testSubresources},
      {"cuts", testCuts},           {"diagnostics", testDiagnostics}, {"pipelines", testPipelines},
      {"device", testDevice},       {"tile", testTile},         {"bindings", testBindings},
      {"astro", testAstroFrame}};
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
