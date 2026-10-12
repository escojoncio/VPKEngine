// Headless Metal backend tests: render offscreen through gpu/core, read back, compare with pixels
// computed here. Usage: vpgpu_metal_tests [group]  (no group = all).
// A group this GPU cannot run prints `SKIP <group>: <reason>`; if every group run was skipped the
// exit status is 77, which CTest reports as skipped (the simulator lacks tile memory, for one).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "vpgpu/core/device.h"
#include "vpgpu/metal/metal_backend.h"

using namespace vpgpu;
using namespace vpgpu::metal;

namespace {

int failures = 0;
#define CHECK(cond)                                                 \
  do {                                                              \
    if (!(cond)) {                                                  \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                   \
    }                                                               \
  } while (0)

// Set by a group that cannot run here; it returns right after.
std::string skipReason;
#define SKIP(reason)       \
  do {                     \
    skipReason = (reason); \
    return;                \
  } while (0)

// Test shaders share these structs; each stage declares (MslShader::arguments) only the entries it
// reads, and the backend fills them from the draw's bindings by (kind, slot): buffer slot 0 = verts,
// 1 = color, texture slot 0, sampler slot 0. Pixel-local bindings are [[color(n)]] inputs instead.
const char* kCommon = R"(
#include <metal_stdlib>
using namespace metal;
struct Args {
  device const float4* verts;   // xyzw clip positions
  device const float4* color;   // [0] = draw color, [1] = second target
  texture2d<float> tex;
};
struct VOut { float4 pos [[position]]; };
vertex VOut vs(uint vid [[vertex_id]], constant Args& a [[buffer(30)]]) { return {a.verts[vid]}; }
fragment float4 fs_color(constant Args& a [[buffer(30)]]) { return a.color[0]; }
struct Two { float4 c0 [[color(0)]]; float4 c1 [[color(1)]]; };
fragment Two fs_mrt(constant Args& a [[buffer(30)]]) { return {a.color[0], a.color[1]}; }
fragment float4 fs_tex(VOut in [[stage_in]], constant Args& a [[buffer(30)]]) {
  constexpr sampler s(filter::nearest, address::clamp_to_edge);
  return a.tex.sample(s, in.pos.xy / 16.0);
}
struct SArgs {
  device const float4* verts;
  device const float4* color;
  texture2d<float> tex;
  sampler smp;
};
fragment float4 fs_sampler(VOut in [[stage_in]], constant SArgs& a [[buffer(30)]]) {
  return a.tex.sample(a.smp, in.pos.xy / 16.0);
}
// Deferred: G-buffer to colours 0-2, lighting reads them at its own pixel and writes colour 3.
struct GOut { float4 albedo [[color(0)]]; float4 normal [[color(1)]]; float4 material [[color(2)]]; };
fragment GOut fs_gbuffer(constant Args& a [[buffer(30)]]) { return {a.color[0], a.color[1], a.color[2]}; }
struct CArgs { device uint* out; };
kernel void cs_fill(uint i [[thread_position_in_grid]], constant CArgs& a [[buffer(30)]]) { a.out[i] = i * 3 + 1; }
)";

// Framebuffer fetch kept in its own library: GPUs without it (the simulator) may reject the source,
// and only the deferred group, which skips there, compiles it.
const char* kLight = R"(
#include <metal_stdlib>
using namespace metal;
struct LOut { float4 c [[color(3)]]; };
fragment LOut fs_light(float4 albedo [[color(0)]], float4 normal [[color(1)]], float4 material [[color(2)]]) {
  return {float4(albedo.rgb * material.r + normal.rgb * 0.5, 1.0)};
}
)";

enum : uint64_t { kVs = 1, kFsColor, kFsMrt, kFsTex, kCsFill, kFsSampler, kFsGbuffer, kFsLight };

struct Rig {
  ShaderTable shaders;
  std::vector<std::string> errors;
  MetalBackend backend;
  Device device;

  Rig()
      : backend(registerShaders(shaders), [this](const std::string& l) { log(l); }),
        device(backend, 0, [this](const std::string& l) { log(l); }) {}

  static const ShaderProvider& registerShaders(ShaderTable& t) {
    const ShaderArgument verts{BindKind::Buffer, 0, 0}, color{BindKind::Buffer, 1, 8};
    const ShaderArgument tex{BindKind::Texture, 0, 16}, smp{BindKind::Sampler, 0, 24};
    auto shader = [](const char* entry, std::vector<ShaderArgument> args, uint8_t outputs = 1) {
      MslShader m;
      m.source = kCommon;
      m.entry = entry;
      m.arguments = std::move(args);
      m.colorOutputs = outputs;
      return m;
    };
    t.add(kVs, shader("vs", {verts}, 0));
    t.add(kFsColor, shader("fs_color", {color}));
    t.add(kFsMrt, shader("fs_mrt", {color}, 0b11));
    t.add(kFsTex, shader("fs_tex", {tex}));
    MslShader cs = shader("cs_fill", {verts}, 0);
    cs.threadsPerGroup = {64, 1, 1};
    t.add(kCsFill, cs);
    t.add(kFsSampler, shader("fs_sampler", {tex, smp}));
    t.add(kFsGbuffer, shader("fs_gbuffer", {color}, 0b111));
    MslShader light = shader("fs_light", {}, 0b1000);
    light.source = kLight;
    t.add(kFsLight, light);
    return t;
  }
  void log(const std::string& l) {
    std::printf("    %s\n", l.c_str());
    if (l.rfind("gpu.err", 0) == 0) errors.push_back(l);
  }

  PipelineKey pipeline(PipelineState s) {
    CHECK(device.pipelines().request(s, Priority::Blocking) == PipelineCache::Status::Ready);
    return s.key();
  }
  PipelineKey colorPipeline(uint64_t fs, std::vector<Format> colors, Format depth = Format::Undefined,
                            bool depthTest = false, Primitive prim = Primitive::Triangles) {
    PipelineState s;
    s.vertexShader = kVs;
    s.fragmentShader = fs;
    for (size_t i = 0; i < colors.size(); ++i) s.colorFormats[i] = colors[i];
    s.depthFormat = depth;
    s.depthTest = s.depthWrite = depthTest;
    s.depthCompare = CompareOp::Less;
    s.primitive = prim;
    return pipeline(s);
  }

  BufferId buffer(const std::vector<float>& data) {
    const BufferId b = device.createBuffer({data.size() * 4, nullptr, "data"});
    std::memcpy(backend.contents(b), data.data(), data.size() * 4);
    return b;
  }
  TextureId target(Format f, uint32_t size) {
    TextureDesc d;
    d.format = f;
    d.width = d.height = size;
    d.usage = usage::sampled | (formatInfo(f).depth() ? usage::depthTarget : usage::colorTarget);
    return device.createTexture(d);
  }

  // Submits, appends a readback copy for each texture, waits, and returns RGBA8 rows.
  std::vector<std::vector<uint8_t>> run(Frame& f, std::vector<TextureId> read, uint32_t size, FrameReport* report = nullptr) {
    std::vector<BufferId> out;
    for (TextureId t : read) {
      out.push_back(device.createBuffer({uint64_t{size} * size * 4, nullptr, "readback"}));
      Copy c;
      c.kind = CopyKind::TextureToBuffer;
      c.srcTexture = {t, 0, 0, 1, 0, 0, 0, size, size, 1};
      c.dstBuffer = {out.back(), 0, uint64_t{size} * size * 4, 0, 0};
      f.copy(c);
    }
    const FrameReport r = device.submit(f);
    if (report) *report = r;
    backend.waitIdle();
    std::vector<std::vector<uint8_t>> pixels;
    for (BufferId b : out) {
      const auto* p = static_cast<const uint8_t*>(backend.contents(b));
      pixels.emplace_back(p, p + size_t{size} * size * 4);
    }
    return pixels;
  }
};

bool near(uint8_t a, uint8_t b) { return std::abs(int(a) - int(b)) <= 1; }

// Counts pixels that differ from expect(x, y) (RGBA); prints the first few.
int compare(const std::vector<uint8_t>& px, uint32_t size, const std::function<std::array<uint8_t, 4>(uint32_t, uint32_t)>& expect,
            const std::function<bool(uint32_t, uint32_t)>& skip = nullptr) {
  int bad = 0;
  for (uint32_t y = 0; y < size; ++y)
    for (uint32_t x = 0; x < size; ++x) {
      if (skip && skip(x, y)) continue;
      const auto e = expect(x, y);
      const uint8_t* p = &px[(size_t{y} * size + x) * 4];
      if (near(p[0], e[0]) && near(p[1], e[1]) && near(p[2], e[2]) && near(p[3], e[3])) continue;
      if (bad++ < 4)
        std::printf("  pixel %u,%u = %u %u %u %u, expected %u %u %u %u\n", x, y, p[0], p[1], p[2], p[3], e[0], e[1], e[2], e[3]);
    }
  return bad;
}

uint8_t unorm(float v) { return static_cast<uint8_t>(std::lround(v * 255.0f)); }

RenderPassDesc pass(TextureId color, LoadAction load, std::array<float, 4> clear = {}) {
  RenderPassDesc d;
  d.colors[0].texture = color;
  d.colors[0].load = load;
  d.colors[0].clear.color = clear;
  return d;
}

// Buffers bound at slots 0, 1, ... in order.
Draw draw(PipelineKey p, std::vector<BufferId> buffers, uint32_t count, Primitive prim = Primitive::Triangles) {
  Draw d;
  d.pipeline = p;
  d.count = count;
  d.primitive = prim;
  for (BufferId b : buffers) {
    d.bindings.push_back(read(ResourceRange::wholeBuffer(b), stage::vertex | stage::fragment));
    d.bindings.back().slot = uint16_t(d.bindings.size() - 1);
  }
  return d;
}

constexpr uint32_t N = 64;
const std::vector<float> kLowerLeft = {-1, -1, 0, 1, 1, -1, 0, 1, -1, 1, 0, 1};  // below the y = -x diagonal
const std::vector<float> kUpperRight = {1, 1, 0, 1, -1, 1, 0, 1, 1, -1, 0, 1};
std::vector<float> fullQuad(float z) { return {-1, -1, z, 1, 1, -1, z, 1, -1, 1, z, 1, 1, 1, z, 1}; }
// Pixel centres with x == y lie exactly on the shared diagonal edge: coverage there is rule-dependent.
bool onDiagonal(uint32_t x, uint32_t y) { return x == N - 1 - y; }
bool lowerLeft(uint32_t x, uint32_t y) { return x < N - 1 - y; }

// ---------------------------------------------------------------------------------------------

void testClear(Rig& r) {
  const TextureId t = r.target(Format::RGBA8Unorm, N);
  Frame f;
  f.clear(pass(t, LoadAction::Clear, {0.25f, 0.5f, 0.75f, 1.0f}));
  const auto px = r.run(f, {t}, N);
  CHECK(compare(px[0], N, [](uint32_t, uint32_t) { return std::array<uint8_t, 4>{64, 128, 191, 255}; }) == 0);
}

void testTriangle(Rig& r) {
  const TextureId t = r.target(Format::RGBA8Unorm, N);
  const PipelineKey p = r.colorPipeline(kFsColor, {Format::RGBA8Unorm});
  const BufferId v = r.buffer(kLowerLeft), c = r.buffer({1, 0, 0, 1});
  Frame f;
  f.beginRenderPass(pass(t, LoadAction::Clear, {0, 0, 1, 1}));
  f.draw(draw(p, {v, c}, 3));
  f.endRenderPass();
  const auto px = r.run(f, {t}, N);
  CHECK(compare(px[0], N, [](uint32_t x, uint32_t y) {
          return lowerLeft(x, y) ? std::array<uint8_t, 4>{255, 0, 0, 255} : std::array<uint8_t, 4>{0, 0, 255, 255};
        }, onDiagonal) == 0);
}

void testTexturedQuad(Rig& r) {
  constexpr uint32_t S = 16;
  // 4x4 texture, each texel a distinct colour; uploaded with a buffer->texture copy.
  std::vector<uint8_t> texels(4 * 4 * 4);
  for (uint32_t i = 0; i < 16; ++i) texels[i * 4] = uint8_t(i * 16), texels[i * 4 + 1] = uint8_t(255 - i * 16), texels[i * 4 + 2] = uint8_t(i), texels[i * 4 + 3] = 255;
  const BufferId upload = r.device.createBuffer({texels.size(), nullptr, "upload"});
  std::memcpy(r.backend.contents(upload), texels.data(), texels.size());
  TextureDesc td;
  td.format = Format::RGBA8Unorm;
  td.width = td.height = 4;
  const TextureId tex = r.device.createTexture(td);
  const TextureId t = r.target(Format::RGBA8Unorm, S);
  const PipelineKey p = r.colorPipeline(kFsTex, {Format::RGBA8Unorm}, Format::Undefined, false, Primitive::TriangleStrip);
  const BufferId v = r.buffer(fullQuad(0)), c = r.buffer({0, 0, 0, 0});

  Frame f;
  Copy up;
  up.kind = CopyKind::BufferToTexture;
  up.srcBuffer = {upload, 0, texels.size(), 16, 4};
  up.dstTexture = {tex, 0, 0, 1, 0, 0, 0, 4, 4, 1};
  f.copy(up);
  f.beginRenderPass(pass(t, LoadAction::DontCare));
  Draw d = draw(p, {v, c}, 4, Primitive::TriangleStrip);
  d.bindings.push_back(read(ResourceRange::wholeTexture(tex), stage::fragment));
  f.draw(d);
  f.endRenderPass();
  // Same quad through a sampler from the argument table, bound before the texture: entries are
  // looked up by (kind, slot), so binding order does not matter.
  SamplerDesc sd;
  sd.minFilter = sd.magFilter = Filter::Nearest;
  sd.mipFilter = MipFilter::None;
  sd.u = sd.v = AddressMode::ClampToEdge;
  const SamplerId smp = r.device.createSampler(sd);
  const TextureId t2 = r.target(Format::RGBA8Unorm, S);
  const PipelineKey ps = r.colorPipeline(kFsSampler, {Format::RGBA8Unorm}, Format::Undefined, false, Primitive::TriangleStrip);
  f.beginRenderPass(pass(t2, LoadAction::DontCare));
  Draw ds = draw(ps, {v, c}, 4, Primitive::TriangleStrip);
  ds.bindings.insert(ds.bindings.begin(), samplerBinding(smp, stage::fragment));
  ds.bindings.push_back(read(ResourceRange::wholeTexture(tex), stage::fragment));
  f.draw(ds);
  f.endRenderPass();
  const auto px = r.run(f, {t, t2}, S);
  for (int i = 0; i < 2; ++i)
    CHECK(compare(px[i], S, [&](uint32_t x, uint32_t y) {
            const uint8_t* e = &texels[((y / 4) * 4 + x / 4) * 4];
            return std::array<uint8_t, 4>{e[0], e[1], e[2], e[3]};
          }) == 0);
}

void testDepth(Rig& r) {
  const TextureId t = r.target(Format::RGBA8Unorm, N), z = r.target(Format::D32Float, N);
  const PipelineKey p = r.colorPipeline(kFsColor, {Format::RGBA8Unorm}, Format::D32Float, true, Primitive::TriangleStrip);
  const PipelineKey tri = r.colorPipeline(kFsColor, {Format::RGBA8Unorm}, Format::D32Float, true);
  const BufferId mid = r.buffer(fullQuad(0.5f)), far = r.buffer(fullQuad(0.7f));
  const BufferId nearTri = r.buffer({-1, -1, 0.3f, 1, 1, -1, 0.3f, 1, -1, 1, 0.3f, 1});
  const BufferId red = r.buffer({1, 0, 0, 1}), green = r.buffer({0, 1, 0, 1}), blue = r.buffer({0, 0, 1, 1});
  RenderPassDesc d = pass(t, LoadAction::Clear);
  d.depth.texture = z;
  d.depth.load = LoadAction::Clear;
  d.depth.store = StoreAction::DontCare;
  d.depth.clear.depth = 1.0f;
  Frame f;
  f.beginRenderPass(d);
  f.draw(draw(p, {mid, red}, 4, Primitive::TriangleStrip));
  f.draw(draw(p, {far, green}, 4, Primitive::TriangleStrip));  // behind: rejected
  f.draw(draw(tri, {nearTri, blue}, 3));                       // in front: lower-left wins
  f.endRenderPass();
  const auto px = r.run(f, {t}, N);
  CHECK(compare(px[0], N, [](uint32_t x, uint32_t y) {
          return lowerLeft(x, y) ? std::array<uint8_t, 4>{0, 0, 255, 255} : std::array<uint8_t, 4>{255, 0, 0, 255};
        }, onDiagonal) == 0);
}

void testMrt(Rig& r) {
  const TextureId a = r.target(Format::RGBA8Unorm, N), b = r.target(Format::RGBA8Unorm, N);
  const PipelineKey p = r.colorPipeline(kFsMrt, {Format::RGBA8Unorm, Format::RGBA8Unorm}, Format::Undefined, false,
                                        Primitive::TriangleStrip);
  const BufferId v = r.buffer(fullQuad(0)), c = r.buffer({1, 0.5f, 0, 1, 0, 0.25f, 1, 0.5f});
  RenderPassDesc d = pass(a, LoadAction::DontCare);
  d.colors[1].texture = b;
  d.colors[1].load = LoadAction::DontCare;
  Frame f;
  f.beginRenderPass(d);
  f.draw(draw(p, {v, c}, 4, Primitive::TriangleStrip));
  f.endRenderPass();
  const auto px = r.run(f, {a, b}, N);
  CHECK(compare(px[0], N, [](uint32_t, uint32_t) { return std::array<uint8_t, 4>{255, unorm(0.5f), 0, 255}; }) == 0);
  CHECK(compare(px[1], N, [](uint32_t, uint32_t) { return std::array<uint8_t, 4>{0, unorm(0.25f), 255, unorm(0.5f)}; }) == 0);
}

void testCompute(Rig& r) {
  // Output aliases host memory in place, at a deliberately non-page-aligned address.
  constexpr uint32_t count = 256;
  void* block = std::aligned_alloc(16384, 3 * 16384);
  auto* guest = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(block) + 16384 + 100);
  std::memset(guest, 0, count * 4);
  const BufferId out = r.device.createBuffer({count * 4, guest, "guest"});
  PipelineState s;
  s.computeShader = kCsFill;
  Dispatch d;
  d.pipeline = r.pipeline(s);
  d.groups = {count / 64, 1, 1};
  d.bindings = {write(ResourceRange::wholeBuffer(out), stage::compute)};
  Frame f;
  f.dispatch(d);
  r.run(f, {}, 0);
  int bad = 0;
  for (uint32_t i = 0; i < count; ++i) bad += guest[i] != i * 3 + 1;
  CHECK(bad == 0);
  CHECK(r.backend.contents(out) == guest);
  // `block` is leaked on purpose: the backend releases the destroyed buffer frames later.
  r.device.destroy(out);
}

void testMergedPasses(Rig& r) {
  const TextureId t = r.target(Format::RGBA8Unorm, N);
  const PipelineKey p = r.colorPipeline(kFsColor, {Format::RGBA8Unorm});
  const BufferId lo = r.buffer(kLowerLeft), hi = r.buffer(kUpperRight);
  const BufferId red = r.buffer({1, 0, 0, 1}), green = r.buffer({0, 1, 0, 1});
  Frame f;
  f.beginRenderPass(pass(t, LoadAction::Clear, {0, 0, 0, 1}));
  f.draw(draw(p, {lo, red}, 3));
  f.endRenderPass();
  f.beginRenderPass(pass(t, LoadAction::Load));  // same target, no clear: joins the open encoder
  f.draw(draw(p, {hi, green}, 3));
  f.endRenderPass();
  FrameReport report;
  const auto px = r.run(f, {t}, N, &report);
  CHECK(report.renderEncoders == 1 && report.passes == 2 && report.compile.passesMerged == 1);
  CHECK(compare(px[0], N, [](uint32_t x, uint32_t y) {
          return lowerLeft(x, y) ? std::array<uint8_t, 4>{255, 0, 0, 255} : std::array<uint8_t, 4>{0, 255, 0, 255};
        }, onDiagonal) == 0);
}

// G-buffer + lighting in one encoder: G-buffer memoryless, lighting reads it by framebuffer fetch.
void testDeferred(Rig& r) {
  if (!r.backend.tileMemory()) SKIP("no tile memory (memoryless + framebuffer fetch)");
  const TextureId albedo = r.target(Format::RGBA8Unorm, N), normal = r.target(Format::RGBA16Float, N);
  const TextureId material = r.target(Format::RGBA8Unorm, N), depth = r.target(Format::D32Float, N);
  const TextureId hdr = r.target(Format::RGBA8Unorm, N);
  const std::vector<Format> formats = {Format::RGBA8Unorm, Format::RGBA16Float, Format::RGBA8Unorm, Format::RGBA8Unorm};

  RenderPassDesc gb;
  const TextureId g[] = {albedo, normal, material};
  const std::array<float, 4> clears[] = {{0.2f, 0.4f, 0.6f, 1}, {0, 0, 0, 0}, {1, 0, 0, 0}};
  for (int i = 0; i < 3; ++i) {
    gb.colors[i].texture = g[i];
    gb.colors[i].load = LoadAction::Clear;
    gb.colors[i].store = StoreAction::DontCare;
    gb.colors[i].clear.color = clears[i];
  }
  gb.depth.texture = depth;
  gb.depth.load = LoadAction::Clear;
  gb.depth.store = StoreAction::DontCare;
  RenderPassDesc light;
  light.colors[3].texture = hdr;
  light.colors[3].load = LoadAction::Clear;
  light.depth = gb.depth;
  light.depth.load = LoadAction::Load;

  // Every pipeline of the merged encoder declares all four colour formats and depth; attachments a
  // shader does not write (colorOutputs) are masked off by the backend, so the G-buffer survives
  // the lighting pipeline even with blending left on.
  PipelineState gs;
  gs.vertexShader = kVs;
  gs.fragmentShader = kFsGbuffer;
  for (size_t i = 0; i < formats.size(); ++i) gs.colorFormats[i] = formats[i];
  gs.depthFormat = Format::D32Float;
  gs.depthTest = gs.depthWrite = true;
  gs.depthCompare = CompareOp::Less;
  PipelineState ls = gs;
  ls.fragmentShader = kFsLight;
  for (int i = 0; i < 3; ++i) {
    ls.blend[i].enabled = true;
    ls.blend[i].dstColor = ls.blend[i].dstAlpha = BlendFactor::One;
  }
  ls.depthTest = ls.depthWrite = false;
  ls.primitive = Primitive::TriangleStrip;

  const BufferId tri = r.buffer({-1, -1, 0.3f, 1, 1, -1, 0.3f, 1, -1, 1, 0.3f, 1});
  const BufferId quad = r.buffer(fullQuad(0.5f));
  const BufferId gcol = r.buffer({1, 0, 0, 1, 0, 0.4f, 0.8f, 1, 0.5f, 0, 0, 1});
  Draw ld = draw({}, {quad, gcol}, 4, Primitive::TriangleStrip);
  for (TextureId t : g) ld.bindings.push_back(pixelLocalRead(t));
  CHECK(fetchInputs(ld, gb, ls.fetch) && ls.fetch.mask == 0b111);
  const PipelineKey gp = r.pipeline(gs);
  ld.pipeline = r.pipeline(ls);

  Frame f;
  f.beginRenderPass(gb);
  f.draw(draw(gp, {tri, gcol}, 3));
  f.endRenderPass();
  f.beginRenderPass(light);
  f.draw(ld);
  f.endRenderPass();
  FrameReport report;
  const auto px = r.run(f, {hdr}, N, &report);
  std::printf("  deferred: render encoders=%u tile_local=%u saved_mb=%.3f\n", report.renderEncoders,
              report.compile.tileLocal, double(report.compile.tileBytes) / (1024.0 * 1024.0));
  CHECK(report.renderEncoders == 1 && report.compile.tileLocal == 4);
  // lit = albedo * material.r + normal * 0.5
  CHECK(compare(px[0], N, [](uint32_t x, uint32_t y) {
          return lowerLeft(x, y) ? std::array<uint8_t, 4>{unorm(0.5f), unorm(0.2f), unorm(0.4f), 255}
                                 : std::array<uint8_t, 4>{unorm(0.2f), unorm(0.4f), unorm(0.6f), 255};
        }, onDiagonal) == 0);
}

// Stencil: the triangle marks 1, a full-screen quad then draws only where the mark is.
void testStencil(Rig& r) {
  const TextureId t = r.target(Format::RGBA8Unorm, N), ds = r.target(Format::D32FloatS8, N);
  PipelineState mark;
  mark.vertexShader = kVs;
  mark.fragmentShader = kFsColor;
  mark.colorFormats[0] = Format::RGBA8Unorm;
  mark.depthFormat = Format::D32FloatS8;
  mark.stencilTest = true;
  mark.stencilFront.pass = mark.stencilBack.pass = StencilOp::Replace;
  PipelineState test = mark;
  test.primitive = Primitive::TriangleStrip;
  test.stencilFront = test.stencilBack = {};
  test.stencilFront.compare = test.stencilBack.compare = CompareOp::Equal;
  const PipelineKey pm = r.pipeline(mark), pt = r.pipeline(test);
  const BufferId v = r.buffer(kLowerLeft), q = r.buffer(fullQuad(0)), red = r.buffer({1, 0, 0, 1}), green = r.buffer({0, 1, 0, 1});
  RenderPassDesc d = pass(t, LoadAction::Clear, {0, 0, 1, 1});
  d.depth.texture = ds;
  d.depth.load = LoadAction::Clear;
  d.depth.store = StoreAction::DontCare;
  Frame f;
  f.beginRenderPass(d);
  Draw dm = draw(pm, {v, red}, 3);
  dm.stencilRef = 1;
  f.draw(dm);
  Draw dt = draw(pt, {q, green}, 4, Primitive::TriangleStrip);
  dt.stencilRef = 1;
  f.draw(dt);
  f.endRenderPass();
  const auto px = r.run(f, {t}, N);
  CHECK(compare(px[0], N, [](uint32_t x, uint32_t y) {
          return lowerLeft(x, y) ? std::array<uint8_t, 4>{0, 255, 0, 255} : std::array<uint8_t, 4>{0, 0, 255, 255};
        }, onDiagonal) == 0);
}

// 4x MSAA rendered and resolved in one pass; the samples are never stored (memoryless on Apple).
void testMsaaResolve(Rig& r) {
  TextureDesc md;
  md.format = Format::RGBA8Unorm;
  md.width = md.height = N;
  md.samples = 4;
  md.usage = usage::colorTarget;
  const TextureId ms = r.device.createTexture(md), t = r.target(Format::RGBA8Unorm, N);
  PipelineState s;
  s.vertexShader = kVs;
  s.fragmentShader = kFsColor;
  s.colorFormats[0] = Format::RGBA8Unorm;
  s.samples = 4;
  const PipelineKey p = r.pipeline(s);
  const BufferId v = r.buffer(kLowerLeft), red = r.buffer({1, 0, 0, 1});
  RenderPassDesc d = pass(ms, LoadAction::Clear, {0, 0, 1, 1});
  d.colors[0].store = StoreAction::Resolve;
  d.colors[0].resolve = t;
  Frame f;
  f.beginRenderPass(d);
  f.draw(draw(p, {v, red}, 3));
  f.endRenderPass();
  FrameReport report;
  const auto px = r.run(f, {t}, N, &report);
  CHECK(report.compile.tileLocal == 1);
  // Edge pixels average their samples; every other pixel is fully inside or outside.
  CHECK(compare(px[0], N, [](uint32_t x, uint32_t y) {
          return lowerLeft(x, y) ? std::array<uint8_t, 4>{255, 0, 0, 255} : std::array<uint8_t, 4>{0, 0, 255, 255};
        }, [](uint32_t x, uint32_t y) { return x + y >= N - 2 && x + y <= N; }) == 0);
}

// Blend constant: out = src * constant.
void testBlendConstant(Rig& r) {
  const TextureId t = r.target(Format::RGBA8Unorm, N);
  PipelineState s;
  s.vertexShader = kVs;
  s.fragmentShader = kFsColor;
  s.colorFormats[0] = Format::RGBA8Unorm;
  s.primitive = Primitive::TriangleStrip;
  s.blend[0].enabled = true;
  s.blend[0].srcColor = s.blend[0].srcAlpha = BlendFactor::Constant;
  s.blend[0].dstColor = s.blend[0].dstAlpha = BlendFactor::Zero;
  const PipelineKey p = r.pipeline(s);
  const BufferId q = r.buffer(fullQuad(0)), white = r.buffer({1, 1, 1, 1});
  Frame f;
  f.beginRenderPass(pass(t, LoadAction::Clear));
  Draw d = draw(p, {q, white}, 4, Primitive::TriangleStrip);
  d.blendColor = {0.25f, 0.5f, 0.75f, 1};
  f.draw(d);
  f.endRenderPass();
  const auto px = r.run(f, {t}, N);
  CHECK(compare(px[0], N, [](uint32_t, uint32_t) { return std::array<uint8_t, 4>{64, 128, 191, 255}; }) == 0);
}

void testTimings(Rig& r) {
  const TextureId t = r.target(Format::RGBA8Unorm, N);
  Frame f;
  f.clear(pass(t, LoadAction::Clear, {1, 1, 1, 1}));
  r.run(f, {t}, N);
  // Device::submit already drained earlier frames; this frame's timings arrive after waitIdle.
  uint64_t frame = 0;
  GpuTimings g;
  bool any = false;
  while (r.backend.takeTimings(frame, g)) any = true;
  CHECK(any);
  // The simulator's command buffers report no GPU start/end times.
  if (any && g.frameMs == 0 && r.backend.caps().simulator) SKIP("no GPU frame times in the simulator");
  CHECK(g.frameMs > 0);
}

// Interop (VK_EXT_metal_objects): the exported texture is the one rendering writes to, allocated on
// request (render targets are lazy); a stale id gives null.
void testNative(Rig& r) {
  CHECK(r.backend.nativeDevice() != nullptr);
  const TextureId t = r.target(Format::RGBA8Unorm, N);
  void* tex = r.backend.nativeTexture(t);
  CHECK(tex != nullptr);
  CHECK(r.backend.nativeTexture(t) == tex);
  TextureId stale = t;
  ++stale.generation;
  CHECK(r.backend.nativeTexture(stale) == nullptr);
  Frame f;
  f.clear(pass(t, LoadAction::Clear, {1.0f, 0.0f, 0.0f, 1.0f}));
  const auto px = r.run(f, {t}, N);
  CHECK(compare(px[0], N, [](uint32_t, uint32_t) { return std::array<uint8_t, 4>{255, 0, 0, 255}; }) == 0);
  CHECK(r.backend.nativeTexture(t) == tex);
}

}  // namespace

int main(int argc, char** argv) {
  const struct {
    const char* name;
    void (*fn)(Rig&);
  } tests[] = {{"clear", testClear},   {"triangle", testTriangle}, {"textured", testTexturedQuad},
               {"depth", testDepth},   {"mrt", testMrt},           {"compute", testCompute},
               {"merged", testMergedPasses}, {"deferred", testDeferred}, {"stencil", testStencil},
               {"msaa", testMsaaResolve}, {"blendconst", testBlendConstant}, {"timings", testTimings},
               {"native", testNative}};
  Rig rig;
  if (!rig.backend.ok()) {
    std::printf("no Metal device\n");
    return 1;
  }
  // Argument tables hold GPU addresses and resource IDs: nothing here can run without them.
  if (!rig.backend.caps().bindless) {
    std::printf("SKIP all: no Metal 3 bindless\n");
    return 77;
  }
  int ran = 0, skipped = 0;
  for (const auto& t : tests) {
    if (argc > 1 && std::strcmp(argv[1], t.name) != 0) continue;
    std::printf("%s\n", t.name);
    const int before = failures;
    const size_t errs = rig.errors.size();
    skipReason.clear();
    t.fn(rig);
    if (rig.errors.size() != errs) {
      std::printf("  FAIL gpu.err lines logged\n");
      ++failures;
    }
    ++ran;
    if (!skipReason.empty() && failures == before) {
      std::printf("SKIP %s: %s\n", t.name, skipReason.c_str());
      ++skipped;
      continue;
    }
    std::printf("  %s\n", failures == before ? "ok" : "FAILED");
  }
  if (!ran) {
    std::printf("unknown test group\n");
    return 1;
  }
  if (failures) return 1;
  return skipped == ran ? 77 : 0;
}
