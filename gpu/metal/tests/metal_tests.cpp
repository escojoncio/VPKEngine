// Headless Metal backend tests: render offscreen through gpu/core, read back, compare with pixels
// computed here. Usage: vpgpu_metal_tests [group]  (no group = all).
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

// Every test shader shares this argument table layout: slot i = Draw::bindings[i].
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
struct CArgs { device uint* out; };
kernel void cs_fill(uint i [[thread_position_in_grid]], constant CArgs& a [[buffer(30)]]) { a.out[i] = i * 3 + 1; }
)";

enum : uint64_t { kVs = 1, kFsColor, kFsMrt, kFsTex, kCsFill };

struct Rig {
  ShaderTable shaders;
  std::vector<std::string> errors;
  MetalBackend backend;
  Device device;

  Rig()
      : backend(registerShaders(shaders), [this](const std::string& l) { log(l); }),
        device(backend, 0, [this](const std::string& l) { log(l); }) {}

  static const ShaderProvider& registerShaders(ShaderTable& t) {
    t.add(kVs, {kCommon, "vs"});
    t.add(kFsColor, {kCommon, "fs_color"});
    t.add(kFsMrt, {kCommon, "fs_mrt"});
    t.add(kFsTex, {kCommon, "fs_tex"});
    t.add(kCsFill, {kCommon, "cs_fill", 30, {64, 1, 1}});
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

Draw draw(PipelineKey p, std::vector<BufferId> buffers, uint32_t count, Primitive prim = Primitive::Triangles) {
  Draw d;
  d.pipeline = p;
  d.count = count;
  d.primitive = prim;
  for (BufferId b : buffers) d.bindings.push_back(read(ResourceRange::wholeBuffer(b), stage::vertex | stage::fragment));
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
  const auto px = r.run(f, {t}, S);
  CHECK(compare(px[0], S, [&](uint32_t x, uint32_t y) {
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
  CHECK(g.frameMs > 0);
}

}  // namespace

int main(int argc, char** argv) {
  const struct {
    const char* name;
    void (*fn)(Rig&);
  } tests[] = {{"clear", testClear},   {"triangle", testTriangle}, {"textured", testTexturedQuad},
               {"depth", testDepth},   {"mrt", testMrt},           {"compute", testCompute},
               {"merged", testMergedPasses}, {"timings", testTimings}};
  Rig rig;
  if (!rig.backend.ok()) {
    std::printf("no Metal device\n");
    return 1;
  }
  int ran = 0;
  for (const auto& t : tests) {
    if (argc > 1 && std::strcmp(argv[1], t.name) != 0) continue;
    std::printf("%s\n", t.name);
    const int before = failures;
    const size_t errs = rig.errors.size();
    t.fn(rig);
    if (rig.errors.size() != errs) {
      std::printf("  FAIL gpu.err lines logged\n");
      ++failures;
    }
    std::printf("  %s\n", failures == before ? "ok" : "FAILED");
    ++ran;
  }
  if (!ran) {
    std::printf("unknown test group\n");
    return 1;
  }
  return failures ? 1 : 0;
}
