// GLSL ES compiler tests: each program is compiled, linked, and the SPIR-V of every stage is put
// through the real Vulkan path (spirv::parse -> spirv::lower -> vpir::verify -> MSL emitter) with a
// driver-style layout, so a test passes only if VPVulkan could actually run the shader.
// Usage: vpglsl_tests <group>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "glsl.h"
#include "vpir/text.h"
#include "vpir/verify.h"
#include "vpmsl/msl.h"
#include "vpvk/spirv_frontend.h"

namespace {

int failures = 0;
#define CHECK(c)                                                                 \
    do {                                                                         \
        if (!(c)) {                                                              \
            std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

using vpglsl::Stage;

struct Lowered {
    bool ok = false;
    std::string msl;
    spirv::SpirvInterface iface;
    vpir::Module module;
};

// Lowers one stage with the layout the GLES layer uses (vpglsl::Reflection).
Lowered lowerStage(const std::vector<uint32_t>& words, Stage stage, const vpglsl::Reflection& refl) {
    Lowered r;
    std::string error;
    spirv::SpirvModule* m = spirv::parse(words.data(), words.size(), error);
    if (!m) {
        std::fprintf(stderr, "  spirv parse: %s\n", error.c_str());
        return r;
    }
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = stage == Stage::Vertex ? vpir::Stage::Vertex : stage == Stage::Fragment ? vpir::Stage::Fragment : vpir::Stage::Compute;
    o.viewCount = refl.numViews;
    uint16_t cb = 0, tex = 0;
    auto add = [&](uint32_t binding, vpgpu::BindKind k, uint32_t count) {
        spirv::SpirvBinding b;
        b.set = 0, b.binding = binding, b.kind = k, b.arraySize = count;
        if (k == vpgpu::BindKind::ConstantBuffer) b.slot = cb, cb = uint16_t(cb + count);
        else b.slot = tex, b.samplerSlot = tex, tex = uint16_t(tex + count);
        o.layout.push_back(b);
    };
    add(vpglsl::kDefaultBlockBinding, vpgpu::BindKind::ConstantBuffer, 1);
    for (const auto& u : refl.uniforms)
        if (u.binding >= 0) add(uint32_t(u.binding), vpgpu::BindKind::Texture, uint32_t(u.arraySize));
    std::map<uint32_t, uint32_t> blockCount;
    for (const auto& b : refl.blocks) blockCount[b.binding] = std::max(blockCount[b.binding], b.element + 1);
    for (auto [binding, n] : blockCount) add(binding, vpgpu::BindKind::ConstantBuffer, n);
    if (stage == Stage::Vertex) {
        uint16_t slot = 30;
        for (const auto& a : refl.attributes) {
            spirv::VertexBufferBinding vb;
            vb.binding = uint32_t(a.location), vb.stride = 64, vb.slot = slot--;
            o.vertexBuffers.push_back(vb);
            spirv::VertexAttribute va;
            va.location = uint32_t(a.location), va.binding = uint32_t(a.location), va.offset = 0;
            va.format = vpgpu::Format::RGBA32Float;
            o.vertexAttributes.push_back(va);
            // Matrices occupy consecutive locations.
            int cols = a.glType >= 0x8B5A && a.glType <= 0x8B5C ? int(a.glType - 0x8B5A + 2) : 1;
            for (int c = 1; c < cols * a.arraySize; ++c) {
                va.location = uint32_t(a.location + c), va.binding = uint32_t(a.location);
                va.offset = uint32_t(16 * c);
                o.vertexAttributes.push_back(va);
            }
        }
    }
    bool lowered = spirv::lower(*m, o, r.module, r.iface, error);
    spirv::release(m);
    if (!lowered) {
        std::fprintf(stderr, "  lower: %s\n", error.c_str());
        return r;
    }
    auto errs = vpir::verify(r.module);
    if (!errs.empty()) {
        std::fprintf(stderr, "%s\n", vpir::print(r.module).c_str());
        for (auto& e : errs) std::fprintf(stderr, "  verify: %s\n", e.c_str());
        return r;
    }
    vpmsl::EmitResult e = vpmsl::emitMsl(r.module, {});
    if (!e.errors.empty()) {
        for (auto& x : e.errors) std::fprintf(stderr, "  msl: %s\n", x.c_str());
        return r;
    }
    r.msl = e.shader.source;
    r.ok = true;
    if (std::getenv("VPGLSL_PRINT")) std::printf("%s\n%s\n", vpir::print(r.module).c_str(), r.msl.c_str());
    return r;
}

struct Program {
    bool ok = false;
    std::string log;
    vpglsl::LinkResult link;
    Lowered vs, fs;
};

Program build(const std::string& vsrc, const std::string& fsrc, const vpglsl::LinkOptions& lo = {}) {
    Program p;
    auto v = vpglsl::parse(Stage::Vertex, vsrc);
    auto f = vpglsl::parse(Stage::Fragment, fsrc);
    p.log = v.log + f.log;
    if (!v.shader || !f.shader) {
        std::fprintf(stderr, "  compile: %s\n", p.log.c_str());
        return p;
    }
    p.link = vpglsl::link(v.shader, f.shader, lo);
    if (!p.link.ok) {
        std::fprintf(stderr, "  link: %s\n", p.link.log.c_str());
        return p;
    }
    p.vs = lowerStage(p.link.vertexSpirv, Stage::Vertex, p.link.reflection);
    p.fs = lowerStage(p.link.fragmentSpirv, Stage::Fragment, p.link.reflection);
    p.ok = p.vs.ok && p.fs.ok;
    return p;
}

bool compileFails(Stage s, const std::string& src, const char* expect) {
    auto r = vpglsl::parse(s, src);
    if (r.shader) {
        std::fprintf(stderr, "  expected a compile error (%s)\n", expect);
        return false;
    }
    if (r.log.find(expect) == std::string::npos) {
        std::fprintf(stderr, "  log lacks '%s':\n%s", expect, r.log.c_str());
        return false;
    }
    return true;
}

const vpglsl::Variable* uniform(const Program& p, const std::string& n) {
    for (const auto& u : p.link.reflection.uniforms)
        if (u.name == n) return &u;
    return nullptr;
}

// ---------------------------------------------------------------------------------------------
void testPp() {
    std::string log;
    int version = 0;
    std::string out = vpglsl::preprocess(
        "#version 300 es\n"
        "#define A 2\n"
        "#define F(x, y) ((x) * (y) + A)\n"
        "#define CAT(a, b) a##b\n"
        "#if defined(A) && A > 1\n"
        "int v = F(3, 4);\n"
        "#elif 1\n"
        "int bad;\n"
        "#else\n"
        "int bad2;\n"
        "#endif\n"
        "#ifdef GL_ES\n"
        "int CAT(x, 1) = __VERSION__;\n"
        "#endif\n"
        "#ifndef GL_OVR_multiview2\n"
        "int nomv;\n"
        "#endif\n"
        "int m = F(F(1, 2),\n 5);\n"
        "/* multi\n line */ int after;\n",
        Stage::Vertex, log, version);
    CHECK(version == 300);
    CHECK(log.empty());
    CHECK(out.find("int v = ( ( 3 ) * ( 4 ) + 2 ) ;") != std::string::npos);
    CHECK(out.find("bad") == std::string::npos);
    CHECK(out.find("int x1 = 300 ;") != std::string::npos);
    CHECK(out.find("nomv") == std::string::npos);
    CHECK(out.find("int m = ( ( ( ( 1 ) * ( 2 ) + 2 ) ) * ( 5 ) + 2 ) ;") != std::string::npos);
    CHECK(out.find("int after") != std::string::npos);
    if (failures) std::fprintf(stderr, "%s\n", out.c_str());
    log.clear();
    vpglsl::preprocess("#version 100\n#if 1\n#error boom\n#endif\n", Stage::Vertex, log, version);
    CHECK(log.find("#error boom") != std::string::npos);
    log.clear();
    vpglsl::preprocess("#if 1\nint a;\n", Stage::Vertex, log, version);
    CHECK(log.find("unterminated #if") != std::string::npos);
    log.clear();
    vpglsl::preprocess("int a;\n#version 300 es\n", Stage::Vertex, log, version);
    CHECK(log.find("#version must be the first") != std::string::npos);
}

void testParse() {
    auto r = vpglsl::parse(Stage::Vertex,
                           "#version 300 es\n"
                           "precision highp float;\n"
                           "struct Light { vec3 dir; float k[2]; };\n"
                           "uniform Light u_light;\n"
                           "in vec4 a_pos;\n"
                           "const int N = 2 * 2;\n"
                           "float arr[N];\n"
                           "float f(in float x, out float y) { y = x * 2.0; return x + 1.0; }\n"
                           "void main() {\n"
                           "  float y; float z = f(1.0, y);\n"
                           "  vec3 v = vec3(1.0, vec2(2.0));\n"
                           "  mat3 m = mat3(1.0);\n"
                           "  ivec2 iv = ivec2(3, 4) << 1;\n"
                           "  uint u = 3u;\n"
                           "  for (int i = 0; i < N; ++i) arr[i] = float(i);\n"
                           "  gl_Position = vec4(m * v * u_light.dir * u_light.k[1] + float(iv.x) + float(u), arr.length()) + a_pos;\n"
                           "}\n");
    CHECK(r.shader != nullptr);
    if (!r.shader) std::fprintf(stderr, "%s", r.log.c_str());
}

void testErrors() {
    CHECK(compileFails(Stage::Vertex, "void main() { float x = 1; }", "cannot convert"));
    CHECK(compileFails(Stage::Vertex, "void main() { undefinedThing = 1.0; }", "undeclared identifier"));
    CHECK(compileFails(Stage::Vertex, "void foo() {}", "missing main"));
    CHECK(compileFails(Stage::Fragment, "uniform float u; void main() { u = 1.0; }", "l-value"));
    CHECK(compileFails(Stage::Vertex, "void main() { vec2 v; v.xz = vec2(1.0); }", "swizzle"));
    CHECK(compileFails(Stage::Vertex, "#version 300 es\nvoid main() { float a[2]; a[2] = 1.0; }", "out of range"));
    CHECK(compileFails(Stage::Vertex, "#version 300 es\nattribute vec4 p; void main() {}", "attribute"));
    CHECK(compileFails(Stage::Fragment, "#version 300 es\nprecision mediump float; in int i; out vec4 o; void main() {}", "flat"));
    CHECK(compileFails(Stage::Vertex, "void main() { break; }", "break"));
    CHECK(compileFails(Stage::Vertex, "void main() { vec4 v = vec4(1.0, 2.0); }", "not enough data"));
    CHECK(compileFails(Stage::Vertex, "float f() { return 1; } void main() {}", "return"));
    CHECK(compileFails(Stage::Vertex, "void main() { float x = float(1u); }", "unsigned"));
    CHECK(compileFails(Stage::Vertex, "#version 300 es\nvoid main() { gl_ViewID_OVR; }", "GL_OVR_multiview"));
    CHECK(compileFails(Stage::Vertex, "float gl_x; void main() {}", "reserved"));
}

void testEs100() {
    Program p = build(
        "attribute vec4 a_pos;\n"
        "attribute vec2 a_uv;\n"
        "uniform mat4 u_mvp;\n"
        "varying vec2 v_uv;\n"
        "void main() { v_uv = a_uv; gl_Position = u_mvp * a_pos; gl_PointSize = 1.0; }\n",
        "precision mediump float;\n"
        "varying vec2 v_uv;\n"
        "uniform sampler2D u_tex;\n"
        "uniform vec4 u_col;\n"
        "void main() { gl_FragColor = texture2D(u_tex, v_uv) * u_col; }\n");
    CHECK(p.ok);
    if (!p.ok) return;
    const auto& refl = p.link.reflection;
    CHECK(refl.attributes.size() == 2);
    const auto* mvp = uniform(p, "u_mvp");
    const auto* col = uniform(p, "u_col");
    const auto* tex = uniform(p, "u_tex");
    CHECK(mvp && mvp->glType == 0x8B5C && mvp->offset == 0 && mvp->matrixStride == 16);
    CHECK(col && col->glType == 0x8B52 && col->offset == 64);
    CHECK(tex && tex->glType == 0x8B5E && tex->binding == 1);
    CHECK(refl.defaultBlockSize == 80);
    CHECK(p.fs.iface.colorOutputs == 1);
    CHECK(p.fs.msl.find("sample") != std::string::npos);
    // Clip z remap: (z + w) * 0.5 in the vertex shader.
    CHECK(p.vs.msl.find("0.5") != std::string::npos);
    // Attribute binding.
    vpglsl::LinkOptions lo;
    lo.attribBindings["a_uv"] = 5;
    Program q = build("attribute vec4 a_pos; attribute vec2 a_uv; varying vec2 v; void main() { v = a_uv; gl_Position = a_pos; }",
                      "precision mediump float; varying vec2 v; void main() { gl_FragColor = vec4(v, 0.0, 1.0); }", lo);
    CHECK(q.ok);
    for (const auto& a : q.link.reflection.attributes) {
        if (a.name == "a_uv") CHECK(a.location == 5);
        if (a.name == "a_pos") CHECK(a.location == 0);
    }
}

void testEs300() {
    Program p = build(
        "#version 300 es\n"
        "layout(location = 2) in vec3 a_pos;\n"
        "in vec3 a_n;\n"
        "uniform mat4 u_vp;\n"
        "uniform vec3 u_arr[3];\n"
        "out vec3 v_n;\n"
        "flat out int v_id;\n"
        "void main() { v_n = a_n + u_arr[gl_VertexID % 3]; v_id = gl_InstanceID; gl_Position = u_vp * vec4(a_pos, 1.0); }\n",
        "#version 300 es\n"
        "precision highp float;\n"
        "in vec3 v_n;\n"
        "flat in int v_id;\n"
        "layout(location = 0) out vec4 o0;\n"
        "layout(location = 1) out vec4 o1;\n"
        "void main() { o0 = vec4(normalize(v_n), 1.0); o1 = vec4(float(v_id)); if (v_n.x < 0.0) discard; }\n");
    CHECK(p.ok);
    if (!p.ok) return;
    const auto* arr = uniform(p, "u_arr[0]");
    CHECK(arr && arr->isArray && arr->arraySize == 3 && arr->offset == 64 && arr->arrayStride == 16);
    CHECK(p.fs.iface.colorOutputs == 3);
    for (const auto& a : p.link.reflection.attributes)
        if (a.name == "a_pos") CHECK(a.location == 2);
    // Mismatched varying types fail to link.
    auto v = vpglsl::parse(Stage::Vertex, "#version 300 es\nout vec2 x; void main() { x = vec2(0.0); gl_Position = vec4(0.0); }");
    auto f = vpglsl::parse(Stage::Fragment, "#version 300 es\nprecision mediump float; in vec3 x; out vec4 o; void main() { o = vec4(x, 1.0); }");
    CHECK(v.shader && f.shader);
    if (v.shader && f.shader) CHECK(!vpglsl::link(v.shader, f.shader, {}).ok);
}

void testMultiview() {
    Program p = build(
        "#version 300 es\n"
        "#extension GL_OVR_multiview2 : require\n"
        "layout(num_views = 2) in;\n"
        "in vec4 a_pos;\n"
        "uniform mat4 u_vp[2];\n"
        "void main() { gl_Position = u_vp[gl_ViewID_OVR] * a_pos; }\n",
        "#version 300 es\n"
        "precision mediump float;\n"
        "out vec4 o;\n"
        "void main() { o = vec4(1.0); }\n");
    CHECK(p.ok);
    if (!p.ok) return;
    CHECK(p.link.reflection.numViews == 2);
    bool layer = false;
    for (auto& out : p.vs.module.outputs) layer |= out.builtin == vpir::Builtin::Layer;
    CHECK(layer);
    CHECK(p.vs.msl.find("render_target_array_index") != std::string::npos);
}

void testBlocks() {
    Program p = build(
        "#version 300 es\n"
        "layout(std140) uniform Camera { mat4 view; vec4 tint; float exposure; } cam;\n"
        "uniform Lights { vec4 color[4]; layout(row_major) mat3 rot; };\n"
        "in vec4 a_pos;\n"
        "void main() { gl_Position = cam.view * a_pos + color[1] * cam.exposure + vec4(rot[0], 0.0); }\n",
        "#version 300 es\n"
        "precision mediump float;\n"
        "layout(std140) uniform Camera { mat4 view; vec4 tint; float exposure; } cam;\n"
        "out vec4 o;\n"
        "void main() { o = cam.tint; }\n");
    CHECK(p.ok);
    if (!p.ok) return;
    const auto& bl = p.link.reflection.blocks;
    CHECK(bl.size() == 2);
    if (bl.size() == 2) {
        const auto& cam = bl[0].name == "Camera" ? bl[0] : bl[1];
        const auto& li = bl[0].name == "Lights" ? bl[0] : bl[1];
        CHECK(cam.size == 96);
        CHECK(cam.members.size() == 3 && cam.members[2].name == "Camera.exposure" && cam.members[2].offset == 80);
        CHECK(li.members.size() == 2 && li.members[1].name == "rot" && li.members[1].offset == 64 && li.members[1].rowMajor);
        CHECK(li.size == 64 + 48);
        CHECK(cam.binding != li.binding && cam.binding >= vpglsl::kBlockBase);
        CHECK(cam.stages == 3);
    }
}

void testControl() {
    Program p = build(
        "#version 300 es\n"
        "in vec4 a_pos;\n"
        "uniform int u_n;\n"
        "uniform int u_mode;\n"
        "out float v;\n"
        "float acc(inout float s, float x) { s += x; return s; }\n"
        "void main() {\n"
        "  float s = 0.0;\n"
        "  for (int i = 0; i < 8; i++) { if (i >= u_n) break; if (i == 2) continue; acc(s, float(i)); }\n"
        "  int k = 0;\n"
        "  while (k < 3) { k++; }\n"
        "  do { s -= 1.0; } while (s > 100.0);\n"
        "  for (int j = 0; j < 4; ++j) {\n"
        "    switch (u_mode + j) {\n"
        "      case 0: s += 1.0;\n"
        "      case 1: s += 2.0; break;\n"
        "      case 2: continue;\n"
        "      default: s *= 0.5;\n"
        "    }\n"
        "  }\n"
        "  bool b = u_n > 2 && s > 1.0 || k == 3;\n"
        "  v = b ? s : -s;\n"
        "  gl_Position = a_pos;\n"
        "}\n",
        "#version 300 es\nprecision mediump float; in float v; out vec4 o; void main() { o = vec4(v); }\n");
    CHECK(p.ok);
}

void testBuiltins() {
    Program p = build(
        "#version 310 es\n"
        "in vec4 a;\n"
        "out vec4 v;\n"
        "flat out uint w;\n"
        "void main() {\n"
        "  vec3 x = a.xyz;\n"
        "  float f = dot(x, x) + length(x) + distance(x, x.zyx);\n"
        "  vec3 c = cross(x, normalize(x)) + reflect(x, x) + refract(x, x, 0.5) + faceforward(x, x, x);\n"
        "  vec4 m = mix(a, a.wzyx, 0.25) + clamp(a, 0.0, 1.0) + smoothstep(0.0, 1.0, a) + step(0.5, a);\n"
        "  m += vec4(min(f, 2.0), max(f, 1.0), mod(f, 3.0), fract(f)) + pow(abs(a), vec4(2.0)) + sign(a);\n"
        "  m += vec4(sin(f), cos(f), atan(f, 1.0), atan(f)) + exp2(a) + log2(abs(a) + 1.0) + inversesqrt(a + 2.0);\n"
        "  mat3 r = transpose(mat3(1.0)) * inverse(mat3(2.0)) * determinant(mat2(1.0));\n"
        "  mat4x3 o = outerProduct(vec3(1.0), vec4(2.0));\n"
        "  bvec4 lt = lessThan(a, m);\n"
        "  if (any(lt) && !all(not(lt))) m.x += 1.0;\n"
        "  ivec4 iv = ivec4(a) % 3 & ivec4(7) | ivec4(1) ^ ivec4(2);\n"
        "  uint u = uint(iv.x) >> 1u;\n"
        "  u += uint(floatBitsToInt(f)) + floatBitsToUint(f) + packHalf2x16(a.xy) + packUnorm4x8(a);\n"
        "  u += uint(bitCount(u)) + uint(findMSB(u)) + bitfieldExtract(u, 0, 4);\n"
        "  float whole; float frac = modf(f, whole);\n"
        "  v = m + vec4(r * c + o * a, frac + whole) + vec4(unpackHalf2x16(u), isnan(f) ? 1.0 : 0.0, trunc(f) + round(f));\n"
        "  w = u;\n"
        "  gl_Position = a;\n"
        "}\n",
        "#version 310 es\nprecision mediump float; in vec4 v; flat in uint w; out vec4 o;\n"
        "void main() { o = v + dFdx(v) + dFdy(v) + fwidth(v) + vec4(float(w)); }\n");
    CHECK(p.ok);
}

void testTextures() {
    Program p = build(
        "#version 300 es\n"
        "in vec4 a;\n"
        "uniform sampler2D u_vtex;\n"
        "out vec2 uv;\n"
        "void main() { uv = a.xy; gl_Position = a + textureLod(u_vtex, a.xy, 0.0) + texture(u_vtex, a.xy); }\n",
        "#version 300 es\n"
        "precision highp float;\n"
        "in vec2 uv;\n"
        "uniform sampler2D t2;\n"
        "uniform samplerCube tc;\n"
        "uniform highp sampler2DShadow ts;\n"
        "uniform highp sampler2DArray ta;\n"
        "uniform highp sampler3D t3;\n"
        "uniform highp isampler2D ti;\n"
        "uniform sampler2D tarr[2];\n"
        "out vec4 o;\n"
        "void main() {\n"
        "  o = texture(t2, uv) + texture(t2, uv, 1.0) + textureProj(t2, vec3(uv, 2.0)) + textureLod(t2, uv, 2.0);\n"
        "  o += textureOffset(t2, uv, ivec2(1, -1)) + textureGrad(t2, uv, vec2(0.1), vec2(0.1));\n"
        "  o += texture(tc, vec3(uv, 1.0)) + vec4(texture(ts, vec3(uv, 0.5))) + texture(ta, vec3(uv, 1.0)) + texture(t3, vec3(uv, 0.5));\n"
        "  o += texelFetch(t2, ivec2(uv * 8.0), 0) + vec4(texelFetch(ti, ivec2(0), 0)) + vec4(vec2(textureSize(t2, 0)), 0.0, 0.0);\n"
        "  o += texture(tarr[1], uv);\n"
        "}\n");
    CHECK(p.ok);
    if (!p.ok) return;
    const auto* arr = uniform(p, "tarr[0]");
    CHECK(arr && arr->arraySize == 2);
    CHECK(p.link.reflection.samplerCount == 8);
    // ES 1.00 cube + EXT lod + shadow ext.
    Program q = build(
        "attribute vec4 a; varying vec3 d; uniform sampler2D vt; void main() { d = a.xyz; gl_Position = a + texture2DLod(vt, a.xy, 0.0); }",
        "#extension GL_EXT_shader_texture_lod : enable\n"
        "#extension GL_OES_standard_derivatives : enable\n"
        "precision mediump float; varying vec3 d; uniform samplerCube c; uniform sampler2D t;\n"
        "void main() { gl_FragColor = textureCube(c, d) + texture2DProj(t, vec4(d, 1.0)) + textureCubeLodEXT(c, d, 1.0) + vec4(fwidth(d.x)); }");
    CHECK(q.ok);
}

void testLink() {
    // Unused uniforms are inactive; uniforms shared by both stages appear once with both stage bits.
    Program p = build(
        "uniform vec4 u_shared; uniform vec4 u_unused; attribute vec4 a; void main() { gl_Position = a + u_shared; }",
        "precision mediump float; uniform vec4 u_shared; uniform float u_f; void main() { gl_FragColor = u_shared * u_f; }");
    CHECK(p.ok);
    CHECK(uniform(p, "u_unused") == nullptr);
    const auto* s = uniform(p, "u_shared");
    CHECK(s && s->stages == 3);
    // Struct uniforms flatten to GL names.
    Program q = build(
        "struct L { vec3 c; float k; }; uniform L lights[2]; attribute vec4 a; void main() { gl_Position = a * lights[1].k + vec4(lights[0].c, 0.0); }",
        "precision mediump float; void main() { gl_FragColor = vec4(1.0); }");
    CHECK(q.ok);
    const auto* l1k = uniform(q, "lights[1].k");
    const auto* l0c = uniform(q, "lights[0].c");
    CHECK(l0c && l0c->offset == 0 && l1k && l1k->offset == 16 + 12);
    // A fragment input nobody writes fails.
    auto v = vpglsl::parse(Stage::Vertex, "void main() { gl_Position = vec4(0.0); }");
    auto f = vpglsl::parse(Stage::Fragment, "precision mediump float; varying vec2 x; void main() { gl_FragColor = vec4(x, 0.0, 1.0); }");
    CHECK(v.shader && f.shader && !vpglsl::link(v.shader, f.shader, {}).ok);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    const std::map<std::string, std::function<void()>> tests = {
        {"pp", testPp}, {"parse", testParse}, {"errors", testErrors}, {"es100", testEs100}, {"es300", testEs300},
        {"multiview", testMultiview}, {"blocks", testBlocks}, {"control", testControl}, {"builtins", testBuiltins},
        {"textures", testTextures}, {"link", testLink},
    };
    auto it = tests.find(argv[1]);
    if (it == tests.end()) return 2;
    it->second();
    std::printf("%s: %s\n", argv[1], failures ? "FAIL" : "ok");
    return failures ? 1 : 0;
}
