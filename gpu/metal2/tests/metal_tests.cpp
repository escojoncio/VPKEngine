// Headless Metal backend tests: render offscreen through gpu/core, read back, compare with pixels
// computed here. Usage: vpgpu_metal_tests [group]  (no group = all).
// A group this GPU cannot run prints `SKIP <group>: <reason>`; if every group run was skipped the
// exit status is 77, which CTest reports as skipped (the simulator lacks tile memory, for one).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include <objc/objc.h>

#include "vpgpu/core/device.h"
#include "vpgpu/metal2/metal_backend.h"

using namespace vpgpu;
using namespace vpgpu::metal2;

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

// Depth in tile: the writer also outputs its depth to colour 6 (an R32Float attachment), the reader fetches it per sample ([[sample_id]]
// makes the fragment function run per sample); the reference reader samples the stored multisampled depth instead.
const char* kTile = R"(
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; };
struct Args { device const float4* verts; device const float4* color; };
struct TArgs { device const float4* verts; device const float4* color; depth2d_ms<float> dtex; };
struct WOut { float4 c [[color(0)]]; float d [[color(6)]]; };
fragment WOut fs_dwrite(VOut in [[stage_in]], constant Args& a [[buffer(30)]]) { return {a.color[0], in.pos.z}; }
fragment float4 fs_dread(float d [[color(6)]], uint sid [[sample_id]]) { return float4(d, 0.0, 0.0, 1.0) + float4(0.0 * float(sid)); }
fragment float4 fs_dsample(VOut in [[stage_in]], uint sid [[sample_id]], constant TArgs& a [[buffer(30)]]) {
  return float4(a.dtex.read(uint2(in.pos.xy), sid), 0.0, 0.0, 1.0);
}
// Two layers (multiview by instancing: instance = layer, the way the Vulkan layer draws viewMask 0b11).
struct LOut { float4 pos [[position]]; uint layer [[render_target_array_index]]; };
struct LArgs { device const float4* verts; device const float4* color; depth2d_ms_array<float> dtex; };
vertex LOut vs_layer(uint vid [[vertex_id]], uint inst [[instance_id]], constant Args& a [[buffer(30)]]) { return {a.verts[vid], inst}; }
fragment float4 fs_dsample_l(LOut in [[stage_in]], uint sid [[sample_id]], constant LArgs& a [[buffer(30)]]) {
  return float4(a.dtex.read(uint2(in.pos.xy), in.layer, sid), 0.0, 0.0, 1.0);
}
)";

enum : uint64_t { kVs = 1, kFsColor, kFsMrt, kFsTex, kCsFill, kFsSampler, kFsGbuffer, kFsLight, kFsDepthW, kFsDepthRead, kFsDepthSample, kVsLayer, kFsDepthSampleL };

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
    MslShader dw = shader("fs_dwrite", {color}, 0b1000001);
    dw.source = kTile;
    t.add(kFsDepthW, dw);
    MslShader dr = shader("fs_dread", {}, 0b1);
    dr.source = kTile;
    t.add(kFsDepthRead, dr);
    MslShader ds = shader("fs_dsample", {tex}, 0b1);
    ds.source = kTile;
    t.add(kFsDepthSample, ds);
    MslShader vl = shader("vs_layer", {verts}, 0);
    vl.source = kTile;
    t.add(kVsLayer, vl);
    MslShader dl = shader("fs_dsample_l", {tex}, 0b1);
    dl.source = kTile;
    t.add(kFsDepthSampleL, dl);
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
  // `layers` > 1: the readback of each texture holds its layers one after the other.
  std::vector<std::vector<uint8_t>> run(Frame& f, std::vector<TextureId> read, uint32_t size, FrameReport* report = nullptr,
                                        uint32_t layers = 1) {
    std::vector<BufferId> out;
    const uint64_t layerBytes = uint64_t{size} * size * 4;
    for (TextureId t : read) {
      out.push_back(device.createBuffer({layerBytes * layers, nullptr, "readback"}));
      for (uint32_t l = 0; l < layers; ++l) {
        Copy c;
        c.kind = CopyKind::TextureToBuffer;
        c.srcTexture = {t, 0, uint16_t(l), 1, 0, 0, 0, size, size, 1};
        c.dstBuffer = {out.back(), l * layerBytes, layerBytes, 0, 0};
        f.copy(c);
      }
    }
    const FrameReport r = device.submit(f);
    if (report) *report = r;
    backend.waitIdle();
    std::vector<std::vector<uint8_t>> pixels;
    for (BufferId b : out) {
      const auto* p = static_cast<const uint8_t*>(backend.contents(b));
      pixels.emplace_back(p, p + size_t{size} * size * 4 * layers);
    }
    return pixels;
  }
};

bool near(uint8_t a, uint8_t b, int tol = 1) { return std::abs(int(a) - int(b)) <= tol; }

// Counts pixels that differ from expect(x, y) (RGBA); prints the first few.
int compare(const std::vector<uint8_t>& px, uint32_t size, const std::function<std::array<uint8_t, 4>(uint32_t, uint32_t)>& expect,
            const std::function<bool(uint32_t, uint32_t)>& skip = nullptr, int tol = 1, size_t offset = 0) {
  int bad = 0;
  for (uint32_t y = 0; y < size; ++y)
    for (uint32_t x = 0; x < size; ++x) {
      if (skip && skip(x, y)) continue;
      const auto e = expect(x, y);
      const uint8_t* p = &px[offset + (size_t{y} * size + x) * 4];
      if (near(p[0], e[0], tol) && near(p[1], e[1], tol) && near(p[2], e[2], tol) && near(p[3], e[3], tol)) continue;
      if (bad++ < 4)
        std::printf("  pixel %u,%u = %u %u %u %u, expected %u %u %u %u\n", x, y, p[0], p[1], p[2], p[3], e[0], e[1], e[2], e[3]);
    }
  return bad;
}

uint8_t unorm(float v) { return static_cast<uint8_t>(std::lround(v * 255.0f)); }

RenderPassDesc pass(TextureId color, LoadAction load, std::array<double, 4> clear = {}) {
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
  const std::array<double, 4> clears[] = {{0.2f, 0.4f, 0.6f, 1}, {0, 0, 0, 0}, {1, 0, 0, 0}};
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

// Depth in tile (Vulkan layer: VPVK_DEPTH_TILE): a 2-subpass MSAA 4x pass whose subpass 1 reads the depth of subpass 0. Fused path:
// subpass 0 also writes its depth to an R32Float attachment (colour 6), subpass 1 fetches it per sample; both subpasses are one
// encoder and neither the 4x depth nor the shadow reaches memory. Reference path: subpass 1 samples the stored 4x depth (two
// encoders). The pictures must agree (<= 1 LSB).
// `layers` 2: the multiview shape (Into The Radius 2: viewMask 0b11, one draw instanced per layer, layer = the view) with a
// per-layer reference reader; `sloped`: the triangle's depth varies across it (0.2 / 0.6 / 0.4 at its corners), so the
// pixel-centre depth the shadow holds is compared with the stored per-sample depth within a bound (<= 2 LSB inside; the
// edge pixels, where the samples straddle the triangle, are reported and not bounded).
void depthTileRun(Rig& r, uint32_t layers, bool sloped) {
  if (!r.backend.tileMemory()) SKIP("no tile memory (memoryless + framebuffer fetch)");
  std::printf("  depth_tile layers=%u sloped=%d\n", layers, int(sloped));
  auto ms = [&](Format f, uint8_t usageBits, bool transient) {
    TextureDesc d;
    d.format = f;
    d.width = d.height = N;
    d.layers = uint16_t(layers);
    d.samples = 4;
    d.usage = usageBits;
    d.transient = transient;
    return r.device.createTexture(d);
  };
  const TextureId color = ms(Format::RGBA8Unorm, usage::colorTarget, true);
  const TextureId depth = ms(Format::D32Float, usage::depthTarget, true);
  const TextureId shadow = ms(Format::R32Float, usage::colorTarget, true);
  const TextureId color2 = ms(Format::RGBA8Unorm, usage::colorTarget, true);
  const TextureId depth2 = ms(Format::D32Float, usage::depthTarget | usage::sampled, false);
  auto resolveTarget = [&] {
    TextureDesc d;
    d.format = Format::RGBA8Unorm;
    d.width = d.height = N;
    d.layers = uint16_t(layers);
    d.usage = usage::sampled | usage::colorTarget;
    return r.device.createTexture(d);
  };
  const TextureId out1 = resolveTarget(), out2 = resolveTarget();

  const BufferId tri = sloped ? r.buffer({-1, -1, 0.2f, 1, 1, -1, 0.6f, 1, -1, 1, 0.4f, 1})
                              : r.buffer({-1, -1, 0.3f, 1, 1, -1, 0.3f, 1, -1, 1, 0.3f, 1});
  const BufferId quad = r.buffer(fullQuad(0.9f));
  const BufferId blue = r.buffer({0, 0, 1, 1});

  auto pass0 = [&](TextureId c, TextureId z, bool withShadow) {
    RenderPassDesc d;
    d.colors[0].texture = c;
    d.colors[0].load = LoadAction::Clear;
    d.colors[0].store = StoreAction::Store;  // (the client's pass stores: a later subpass uses it)
    d.colors[0].clear.color = {0, 0, 0, 1};
    if (withShadow) {
      d.colors[6].texture = shadow;
      d.colors[6].load = LoadAction::Clear;
      d.colors[6].store = StoreAction::Store;
      d.colors[6].clear.color = {1, 0, 0, 0};  // the depth clear value
    }
    d.depth.texture = z;
    d.depth.load = LoadAction::Clear;
    d.depth.store = StoreAction::Store;
    d.depth.clear.depth = 1.0f;
    return d;
  };
  auto pass1 = [&](TextureId c, TextureId z, TextureId resolve, bool withShadow) {
    RenderPassDesc d;
    d.colors[0].texture = c;
    d.colors[0].load = LoadAction::Load;
    d.colors[0].store = StoreAction::Resolve;
    d.colors[0].resolve = resolve;
    if (withShadow) {
      d.colors[6].texture = shadow;
      d.colors[6].load = LoadAction::Load;
      d.colors[6].store = StoreAction::DontCare;
    }
    d.depth.texture = z;
    d.depth.load = LoadAction::Load;
    d.depth.store = StoreAction::DontCare;
    return d;
  };
  auto state = [&](uint64_t fs, bool withShadow, bool depthWrite) {
    PipelineState s;
    s.vertexShader = layers > 1 ? kVsLayer : kVs;
    s.fragmentShader = fs;
    s.colorFormats[0] = Format::RGBA8Unorm;
    if (withShadow) s.colorFormats[6] = Format::R32Float;
    s.depthFormat = Format::D32Float;
    s.depthTest = depthWrite, s.depthWrite = depthWrite;
    s.depthCompare = CompareOp::Less;
    s.primitive = depthWrite ? Primitive::Triangles : Primitive::TriangleStrip;
    s.samples = 4;
    return s;
  };

  // Fused: the writer's colour 6 is the depth; the reader's fetch input 6 is texture slot 0.
  PipelineState writer = state(kFsDepthW, true, true);
  writer.blend[6].writeMask = 1;
  PipelineState reader = state(kFsDepthRead, true, false);
  RenderPassDesc b1 = pass1(color, depth, out1, true);
  Draw rd = draw({}, {quad}, 4, Primitive::TriangleStrip);
  rd.instances = layers;
  rd.bindings.push_back(pixelLocalRead(shadow));
  CHECK(fetchInputs(rd, b1, reader.fetch) && reader.fetch.mask == 0b1000000);
  rd.pipeline = r.pipeline(reader);
  Frame f;
  f.beginRenderPass(pass0(color, depth, true));
  Draw wd = draw(r.pipeline(writer), {tri, blue}, 3);
  wd.instances = layers;
  f.draw(wd);
  f.endRenderPass();
  f.beginRenderPass(b1);
  f.draw(rd);
  f.endRenderPass();
  FrameReport fused;
  const auto px1 = r.run(f, {out1}, N, &fused, layers);
  std::printf("  depth_tile fused: encoders=%u passes=%u merged=%u tile_local=%u msaa_store_bytes=%llu\n", fused.renderEncoders,
              fused.passes, fused.compile.passesMerged, fused.compile.tileLocal, (unsigned long long)fused.compile.msaaStoreBytes);
  CHECK(fused.renderEncoders == 1 && fused.passes == 2 && fused.compile.passesMerged == 1);
  CHECK(fused.compile.tileLocal >= 3);  // colour, depth, shadow: all tile memory
  CHECK(fused.compile.msaaStoreBytes == 0 && fused.compile.msaaLoadBytes == 0);

  // Reference: subpass 1 samples the stored depth.
  PipelineState plain = state(kFsColor, false, true);
  PipelineState sampler = state(layers > 1 ? kFsDepthSampleL : kFsDepthSample, false, false);
  Draw sd = draw(r.pipeline(sampler), {quad}, 4, Primitive::TriangleStrip);
  sd.instances = layers;
  sd.bindings.push_back(read(ResourceRange::wholeTexture(depth2), stage::fragment));
  Frame g;
  g.beginRenderPass(pass0(color2, depth2, false));
  Draw pd = draw(r.pipeline(plain), {tri, blue}, 3);
  pd.instances = layers;
  g.draw(pd);
  g.endRenderPass();
  g.beginRenderPass(pass1(color2, depth2, out2, false));
  g.draw(sd);
  g.endRenderPass();
  FrameReport ref;
  const auto px2 = r.run(g, {out2}, N, &ref, layers);
  std::printf("  depth_tile reference: encoders=%u\n", ref.renderEncoders);
  CHECK(ref.renderEncoders == 2);

  // The picture, every layer: red = the depth the pixel's samples hold (the triangle's inside, the clear value 1.0 outside),
  // averaged by the resolve.
  const auto edge = [](uint32_t x, uint32_t y) { return x + y >= N - 2 && x + y <= N; };
  const auto planeZ = [&](uint32_t x, uint32_t y) {
    if (!sloped) return 0.3f;
    const float nx = (float(x) + 0.5f) / N * 2 - 1, ny = 1 - (float(y) + 0.5f) / N * 2;
    return 0.2f + 0.4f * (nx + 1) / 2 + 0.2f * (ny + 1) / 2;
  };
  const int tol = sloped ? 2 : 1;
  const size_t layerBytes = size_t{N} * N * 4;
  for (uint32_t l = 0; l < layers; ++l) {
    CHECK(compare(px1[0], N, [&](uint32_t x, uint32_t y) {
            return lowerLeft(x, y) ? std::array<uint8_t, 4>{unorm(planeZ(x, y)), 0, 0, 255} : std::array<uint8_t, 4>{255, 0, 0, 255};
          }, edge, tol, l * layerBytes) == 0);
    int differ = 0, edgeMax = 0;  // fused against reference
    for (uint32_t y = 0; y < N; ++y)
      for (uint32_t x = 0; x < N; ++x) {
        const size_t o = l * layerBytes + (size_t{y} * N + x) * 4;
        const int d = std::abs(int(px1[0][o]) - int(px2[0][o]));
        if (edge(x, y)) edgeMax = std::max(edgeMax, d);
        else differ += d > tol;
      }
    std::printf("  depth_tile layer %u: interior fused-vs-reference over %d LSB: %d pixels, edge max diff %d LSB (reported)\n", l, tol,
                differ, edgeMax);
    CHECK(differ == 0);
  }
  if (layers > 1) {  // the layers must hold the same picture (same geometry drawn once per view)
    int layerDiff = 0;
    for (size_t i = 0; i < layerBytes; ++i) layerDiff += !near(px1[0][i], px1[0][layerBytes + i], tol);
    CHECK(layerDiff == 0);
  }
}
void testDepthTile(Rig& r) {
  depthTileRun(r, 1, false);
  depthTileRun(r, 1, true);
}
void testDepthTileLayers(Rig& r) { depthTileRun(r, 2, true); }

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

// 5000 dependent fills (each overwrites the previous one: a barrier, so a blit encoder, per fill). The encoder budget
// (VPVK_MAX_ENCODERS_PER_CB, 512) must cut the frame into a bounded number of command buffers, the pending-chunk
// limit must keep submit from running ahead, and the order across the cuts must hold (the last fill wins).
void testManyEncoders(Rig& r) {
  if (std::getenv("VPVK_MAX_ENCODERS_PER_CB") || std::getenv("VPVK_MAX_PENDING_CHUNKS")) {
    skipReason = "budget overridden by the environment";
    return;
  }
  constexpr uint32_t kFills = 5000;
  const BufferId buf = r.device.createBuffer({256, nullptr, "fills"});
  Frame f;
  for (uint32_t i = 0; i < kFills; ++i) f.fillBuffer(BufferSpan{buf, 0, 256, 0, 0}, uint8_t(i & 0xff));
  auto counter = [&](const char* name) {
    std::vector<std::pair<const char*, uint64_t>> v;
    r.backend.diagCounters(v);
    for (const auto& kv : v)
      if (std::strcmp(kv.first, name) == 0) return kv.second;
    return uint64_t{0};
  };
  const uint64_t cb0 = counter("cb"), cut0 = counter("cut_enc");
  const uint64_t blit0 = counter("enc_blit");
  r.run(f, {}, 0);
  const uint64_t cbs = counter("cb") - cb0, blits = counter("enc_blit") - blit0;
  std::printf("    command buffers=%llu blit encoders=%llu cuts=%llu\n", (unsigned long long)cbs,
              (unsigned long long)blits, (unsigned long long)(counter("cut_enc") - cut0));
  CHECK(blits >= kFills / 2);  // the dependent fills really became (many) encoders
  CHECK(cbs <= blits / 512 + 3);  // bounded by the budget, not one per encoder
  CHECK(cbs >= 2);
  const auto* p = static_cast<const uint8_t*>(r.backend.contents(buf));
  int bad = 0;
  for (uint32_t i = 0; i < 256; ++i) bad += p[i] != uint8_t((kFills - 1) & 0xff);
  CHECK(bad == 0);
}

// Texture and view lifetime: guest-style churn (12 textures with a view each, destroyed in the same breath, repeated)
// must leave the backend's live texture / view gauges where they started and the create/destroy counters in step.
void testTextureLifetime(Rig& r) {
  auto counter = [&](const char* name) {
    std::vector<std::pair<const char*, uint64_t>> v;
    r.backend.diagCounters(v);
    for (const auto& kv : v)
      if (std::strcmp(kv.first, name) == 0) return kv.second;
    return uint64_t{0};
  };
  const uint64_t live0 = counter("g_tex_live"), views0 = counter("g_views");
  const uint64_t created0 = counter("tex_created"), destroyed0 = counter("tex_destroyed"), vdestroyed0 = counter("view_destroyed");
  constexpr int kCycles = 4, kPerCycle = 12;
  for (int n = 0; n < kCycles; ++n) {
    std::vector<TextureId> tex, views;
    for (int i = 0; i < kPerCycle; ++i) {
      TextureDesc d;
      d.format = Format::RGBA8Unorm;
      d.width = d.height = 16;
      d.usage = usage::sampled;
      d.persistent = true;
      const TextureId t = r.device.createTexture(d);
      MetalBackend::ViewDesc vd;
      vd.type = TextureType::Tex2D;
      vd.array = true;  // a view that differs from the texture itself, as VkImageView of a layer does
      const TextureId v = r.backend.createView(t, vd);
      CHECK(r.backend.nativeTexture(v) != nullptr);  // makes the MTLTexture view now
      tex.push_back(t), views.push_back(v);
    }
    CHECK(counter("g_tex_live") == live0 + kPerCycle && counter("g_views") == views0 + kPerCycle);
    for (const TextureId v : views) r.backend.destroyView(v);
    for (const TextureId t : tex) r.device.destroy(t);
  }
  CHECK(counter("g_tex_live") == live0 && counter("g_views") == views0);
  CHECK(counter("tex_created") - created0 == kCycles * kPerCycle && counter("tex_destroyed") - destroyed0 == kCycles * kPerCycle);
  CHECK(counter("view_destroyed") - vdestroyed0 == kCycles * kPerCycle);
}

// Weak-reference runtime entry points (<objc/objc-arc.h>): this file is plain C++, so no __weak.
extern "C" {
id objc_storeWeak(id* location, id obj);
id objc_loadWeakRetained(id* location);
void objc_destroyWeak(id* location);
void objc_release(id obj);
}

// A guest thread has no autorelease pool (long-lived pthread). create / nativeTexture / destroy there must not keep a
// destroyed texture alive: nativeTexture used to return the MTLTexture autoreleased into no pool, so none ever died.
void testNoPoolThread(Rig& r) {
  constexpr int kN = 2000;
  std::vector<id> weak(kN, nullptr);  // weak locations, nil until stored
  int tracked = 0;
  std::thread guest([&] {
    for (int i = 0; i < kN; ++i) {
      TextureDesc d;
      d.format = Format::RGBA8Unorm;
      d.width = d.height = 64;
      d.usage = usage::sampled;
      d.persistent = true;
      const TextureId t = r.device.createTexture(d);
      if (void* n = r.backend.nativeTexture(t)) {
        objc_storeWeak(&weak[size_t(i)], static_cast<id>(n));
        ++tracked;
      }
      r.device.destroy(t);
      if (i % 100 == 99) {
        std::string report;
        r.backend.drainRetired(0, report);
      }
    }
  });
  guest.join();
  // nativeTexture pins a texture: its destroy is delayed by the pinned hold (1 s, then released once the GPU is idle).
  std::this_thread::sleep_for(std::chrono::milliseconds(1200));
  std::string report;
  r.backend.drainRetired(0, report);
  int alive = 0;
  for (int i = 0; i < kN; ++i) {
    if (id o = objc_loadWeakRetained(&weak[size_t(i)])) {
      ++alive;
      objc_release(o);
    }
    objc_destroyWeak(&weak[size_t(i)]);
  }
  CHECK(tracked == kN);
  CHECK(alive == 0);
}

int main(int argc, char** argv) {
  const struct {
    const char* name;
    void (*fn)(Rig&);
  } tests[] = {{"clear", testClear},   {"triangle", testTriangle}, {"textured", testTexturedQuad},
               {"depth", testDepth},   {"mrt", testMrt},           {"compute", testCompute},
               {"merged", testMergedPasses}, {"deferred", testDeferred}, {"depth_tile", testDepthTile}, {"depth_tile_layers", testDepthTileLayers}, {"stencil", testStencil},
               {"msaa", testMsaaResolve}, {"blendconst", testBlendConstant}, {"timings", testTimings},
               {"native", testNative},   {"manyenc", testManyEncoders}, {"texlife", testTextureLifetime},
               {"nopool", testNoPoolThread}};
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
