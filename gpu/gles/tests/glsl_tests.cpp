// GLSL ES compiler tests: each program is compiled, linked, and the SPIR-V of every stage is put
// through the real Vulkan path (spirv::parse -> spirv::lower -> vpir::verify -> MSL emitter) with a
// driver-style layout, so a test passes only if VPVulkan could actually run the shader.
// Usage: vpglsl_tests <group>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <functional>
#include <map>
#include <pthread.h>
#include <random>
#include <unistd.h>
#include <string>
#include <vector>

#include "glsl.h"
#include "vpir/text.h"
#include "vpir/verify.h"
#include "vpir_eval.h"
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
Lowered lowerStage(const std::vector<uint32_t>& words, Stage stage, const vpglsl::Reflection& refl, bool xfb = false) {
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
    uint16_t cb = 0, tex = 0, buf = 0, stex = 0;
    auto add = [&](uint32_t binding, vpgpu::BindKind k, uint32_t count) {
        spirv::SpirvBinding b;
        b.set = 0, b.binding = binding, b.kind = k, b.arraySize = count;
        if (k == vpgpu::BindKind::ConstantBuffer) b.slot = cb, cb = uint16_t(cb + count);
        else if (k == vpgpu::BindKind::Buffer) b.slot = buf, buf = uint16_t(buf + count);
        else if (k == vpgpu::BindKind::StorageTexture) b.slot = stex, stex = uint16_t(stex + count);
        else b.slot = tex, b.samplerSlot = tex, tex = uint16_t(tex + count);
        o.layout.push_back(b);
    };
    add(vpglsl::kDefaultBlockBinding, vpgpu::BindKind::ConstantBuffer, 1);
    for (const auto& u : refl.uniforms) {
        if (u.binding < 0) continue;
        const bool image = u.glType >= 0x904C && u.glType <= 0x906A;
        // samplerBuffer / imageBuffer are texel buffer descriptors (a Buffer binding in VPVulkan).
        const bool texel = u.glType == 0x8DC2 || u.glType == 0x8DD0 || u.glType == 0x8DD8 || u.glType == 0x9051 ||
                           u.glType == 0x905C || u.glType == 0x9067;
        add(uint32_t(u.binding), texel ? vpgpu::BindKind::Buffer : image ? vpgpu::BindKind::StorageTexture : vpgpu::BindKind::Texture,
            uint32_t(u.arraySize));
    }
    std::map<uint32_t, uint32_t> storageCount;
    for (const auto& b : refl.storageBlocks) storageCount[b.binding] = std::max(storageCount[b.binding], b.element + 1);
    for (auto [binding, n] : storageCount) add(binding, vpgpu::BindKind::Buffer, n);
    for (const auto& a : refl.atomicBuffers) add(a.binding, vpgpu::BindKind::Buffer, 1);
    if (xfb) add(vpglsl::kXfbBinding, vpgpu::BindKind::Buffer, 1);
    for (uint32_t k = 0; k < 8; ++k)
        if (refl.framebufferFetchMask & (1u << k)) add(vpglsl::kAttachmentBase + k, vpgpu::BindKind::Texture, 1);
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
void testMacroRescan();
void testPp() {
    std::string log;
    int version = 0;
    {  // #line takes macros; __LINE__ follows it, __FILE__ is its second operand
        std::string l2;
        int v2 = 0;
        std::string o2 = vpglsl::preprocess("#define A 10\n#line 234 A\nx __LINE__ __FILE__\ny __LINE__\n", Stage::Vertex, l2, v2);
        CHECK(o2.find("x 234 10") != std::string::npos && o2.find("y 235") != std::string::npos);
        o2 = vpglsl::preprocess("#define A 4\n#line A\nz __LINE__ __FILE__\n", Stage::Vertex, l2, v2);
        CHECK(o2.find("z 4 0") != std::string::npos);
    }
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
    testMacroRescan();
}

// The tokens of the preprocessed text, space separated, for comparisons.
std::string ppTokens(const std::string& src, std::string& log) {
    int version = 0;
    std::string out = vpglsl::preprocess(src, Stage::Vertex, log, version);
    std::string r;
    for (size_t i = 0; i < out.size();) {
        if (out[i] == ' ' || out[i] == '\n') { ++i; continue; }
        size_t j = out.find_first_of(" \n", i);
        if (j == std::string::npos) j = out.size();
        r += (r.empty() ? "" : " ") + out.substr(i, j - i);
        i = j;
    }
    return r;
}

// A macro whose expansion ends with the name of a function-like macro takes the "(" that follows as its call (C99 6.10.3.4: the rescan
// goes on with the rest of the source). This holds for object-like and function-like macros, without copying the rest of the input
// or recursing once per use, and every walk of the expansion counts against the step limit.
void testMacroRescan() {
    std::string log;
    const std::string defs = "#define F(x) x\n#define G F\n#define ID(x) x\n#define H(x) F\n";
    CHECK(ppTokens(defs + "G(1.0)\n", log) == "1.0" && log.empty());
    CHECK(ppTokens(defs + "ID(F)(5)\n", log) == "5" && log.empty());
    CHECK(ppTokens(defs + "H(9)(4)\n", log) == "4" && log.empty());
    CHECK(ppTokens(defs + "H(9)\n", log) == "F" && log.empty());  // no "(" follows: the name stays
    CHECK(ppTokens(defs + "ID(G)(7)\n", log) == "7" && log.empty());
    // C99 6.10.3.4 example: f(2)(9) is 2*9*g or 2*f(9); the rescan continues with g(9), whose f is not painted.
    CHECK(ppTokens("#define f(a) a*g\n#define g(a) f(a)\nf(2)(9)\n", log) == "2 * 9 * g" && log.empty());
    // A macro never expands inside its own expansion, so the name it ends with is not a call.
    CHECK(ppTokens("#define s(a) a*s\ns(1)(2)\n", log) == "1 * s ( 2 )" && log.empty());
    CHECK(ppTokens("#define S S(1)\nS\n", log) == "S ( 1 )" && log.empty());
    // The name that ends an expansion takes the "(" only if nothing follows it in the expansion, not even a macro that expands to nothing.
    CHECK(ppTokens("#define E\n#define A()\n#define F(x) x A E\nF(1)(2)\n", log) == "1 A ( 2 )" && log.empty());
    CHECK(ppTokens("#define E\n#define A()\n#define G 1 A E\nG(2)\n", log) == "1 A ( 2 )" && log.empty());
    CHECK(ppTokens("#define E\n#define f(x) x\n#define VEC f E\nVEC (2.0)\n", log) == "f ( 2.0 )" && log.empty());
    CHECK(ppTokens("#define A()\n#define F(x) x A\nF(1)()\n", log) == "1" && log.empty());  // nothing follows the name: it is called
    // An empty argument next to ## is a placemarker (C99 6.10.3.3).
    CHECK(ppTokens("#define CAT(a, b) a ## b\nCAT(, x) CAT(x, ) CAT(, ) CAT(x, y)\n", log) == "x x xy" && log.empty());
    CHECK(ppTokens("#define P(a, b) [a ## b]\nP(, )\n", log) == "[ ]" && log.empty());
    CHECK(ppTokens("#define C3(a, b, c) a ## b ## c\nC3(, x, ) C3(1, , 2) C3(, , )\n", log) == "x 12" && log.empty());
    // Nested calls in arguments go deeper than 64 levels; a hostile depth is an error, not a crash.
    {
        std::string deep = "#define f(x) x\n", hostile = deep;
        for (int i = 0; i < 300; ++i) deep += "f(";
        deep += "1";
        for (int i = 0; i < 300; ++i) deep += ")";
        CHECK(ppTokens(deep + "\n", log) == "1" && log.empty());
        for (int i = 0; i < 100000; ++i) hostile += "f(";
        hostile += "1";
        for (int i = 0; i < 100000; ++i) hostile += ")";
        log.clear();
        ppTokens(hostile + "\n", log);
        CHECK(log.find("too deep") != std::string::npos || log.find("too large") != std::string::npos);
    }
    // 20000 uses on one line: linear time, no deep recursion, no memory blow-up.
    log.clear();
    std::string many = defs;
    for (int i = 0; i < 20000; ++i) many += "G(1.0) ";
    many += "\n";
    const auto t0 = std::chrono::steady_clock::now();
    const std::string r = ppTokens(many, log);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    CHECK(log.empty() && r.size() == 20000 * 4 - 1);
    CHECK(ms < 2000);
    // Walking the arguments of a call counts as expansion work, whether or not the body uses them.
    std::string big = "#define Z(x) 0\n";
    big += "int a = ";
    for (int i = 0; i < 100; ++i) {
        big += "Z(";
        for (int k = 0; k < 20000; ++k) big += "1 ";
        big += ")+";
    }
    big += "0;\n";
    log.clear();
    ppTokens(big, log);
    CHECK(log.find("too large") != std::string::npos);
}

// Differential fuzz of the macro expansion against the system C preprocessor (small random programs of object-like, function-like and
// empty macros called with nested arguments, fixed seed). Skipped when `cpp` is not installed. What cpp rejects is not compared; what cpp accepts the compiler must expand the same way.
void testPpFuzz() {
    char path[] = "/tmp/vpglsl_ppfuzz_XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) return;
    close(fd);
    const std::string cmd = std::string("cpp -P -undef -nostdinc -w ") + path + " 2>/dev/null";
    if (std::system("cpp --version >/dev/null 2>&1") != 0) {
        std::printf("ppfuzz: no cpp, skipped\n");
        std::remove(path);
        return;
    }
    std::mt19937 rng(20241112u);
    auto pick = [&](int n) { return int(rng() % unsigned(n)); };
    const char* const names[] = {"A", "B", "C", "D", "E"};
    std::function<std::string(int, const std::vector<std::string>&)> seq;
    auto item = [&](int depth, const std::vector<std::string>& params) -> std::string {
        const int c = pick(100);
        if (c < 30) return names[pick(5)];
        if (c < 45 && !params.empty()) return params[size_t(pick(int(params.size())))];
        if (c < 60) return std::vector<std::string>{"x", "y", "1", "+"}[size_t(pick(4))];
        if (c < 75 && depth < 2) return "(" + seq(depth + 1, params) + ")";
        if (c < 90 && depth < 2) {
            std::string r = std::string(names[pick(5)]) + "(";
            for (int k = pick(3), i = 0; i < k; ++i) r += (i ? "," : "") + seq(depth + 1, params);
            return r + ")";
        }
        return pick(2) ? "x" : "2";
    };
    seq = [&](int depth, const std::vector<std::string>& params) {
        std::string r;
        for (int k = pick(4), i = 0; i < k; ++i) r += (i ? " " : "") + item(depth, params);
        return r;
    };
    int compared = 0, different = 0;
    for (int n = 0; n < 300; ++n) {
        std::string src;
        for (const char* name : names) {
            const int kind = name[0] == 'E' ? 0 : pick(6);  // 0 empty (E always: what follows a name that ends an expansion), 1-2 object, 3-5 function with 0-2 parameters
            if (kind == 0) src += std::string("#define ") + name + "\n";
            else if (kind < 3) src += std::string("#define ") + name + " " + seq(0, {}) + "\n";
            else {
                std::vector<std::string> ps = {"p", "q"};
                ps.resize(size_t(kind - 3));
                std::string list;
                for (size_t i = 0; i < ps.size(); ++i) list += (i ? "," : "") + ps[i];
                src += std::string("#define ") + name + "(" + list + ") " + seq(0, ps) + "\n";
            }
        }
        for (int k = 1 + pick(5), i = 0; i < k; ++i) src += (i ? " " : "") + item(0, {});
        src += "\n";
        if (FILE* f = std::fopen(path, "w")) std::fputs(src.c_str(), f), std::fclose(f);
        std::string want;
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) continue;
        for (int ch; (ch = std::fgetc(pipe)) != EOF;) want += char(ch);
        if (pclose(pipe) != 0) continue;
        std::string log;
        const std::string got = ppTokens(src, log);
        if (!log.empty()) {  // (cpp accepted it)
            if (++different <= 3) std::fprintf(stderr, "  ppfuzz %d: cpp accepts, the compiler says:\n%s%s", n, log.c_str(), src.c_str());
            continue;
        }
        auto squeeze = [](const std::string& t) {
            std::string r;
            for (char ch : t)
                if (ch != ' ' && ch != '\n') r += ch;
            return r;
        };
        ++compared;
        if (squeeze(got) != squeeze(want)) {
            if (++different <= 3) std::fprintf(stderr, "  ppfuzz %d differs from cpp:\n%s  cpp:  %s\n  ours: %s\n", n, src.c_str(), squeeze(want).c_str(), squeeze(got).c_str());
        }
    }
    std::remove(path);
    CHECK(compared > 100);
    CHECK(different == 0);
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
    CHECK(compileFails(Stage::Fragment, "precision mediump float; uniform float u; void main() { u = 1.0; }", "l-value"));
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
    // Keywords and reserved words cannot be identifiers; the preprocessor is strict.
    CHECK(compileFails(Stage::Vertex, "#version 300 es\nvoid main() { float else = 1.0; }", "reserved word"));
    CHECK(compileFails(Stage::Vertex, "void main() { float class = 1.0; }", "reserved word"));
    CHECK(compileFails(Stage::Vertex, "#define A 1\n#define A 2\nvoid main() {}", "redefinition"));
    CHECK(compileFails(Stage::Vertex, "#define F(a,a) a\nvoid main() {}", "duplicate"));
    CHECK(compileFails(Stage::Vertex, "#if A\n#endif\nvoid main() {}", "undefined identifier"));
    CHECK(compileFails(Stage::Vertex, "#ifdef A B\n#endif\nvoid main() {}", "extra tokens"));
    CHECK(compileFails(Stage::Vertex, "#undef __LINE__\nvoid main() {}", "predefined"));
    // __VERSION__ is 100 without #version; `0 && AAA` / `1 || AAA` never evaluate AAA; a comment spanning lines does not end its #define.
    CHECK(vpglsl::parse(Stage::Vertex, "#if __VERSION__ != 100\nfloat bad = ;\n#endif\nvoid main() { gl_Position = vec4(0.0); }").shader);
    CHECK(vpglsl::parse(Stage::Vertex, "#if 0 && AAA\nfloat bad = ;\n#endif\n#if 1 || (BBB / 0)\n#endif\nvoid main() { gl_Position = vec4(0.0); }").shader);
    CHECK(compileFails(Stage::Vertex, "#if 1 && AAA\n#endif\nvoid main() {}", "undefined identifier"));
    CHECK(vpglsl::parse(Stage::Vertex, "#define R /* a\n b */ (1.0)\n#define R /* c */ (1.0)\nvoid main() { gl_Position = vec4(R); }").shader);
    CHECK(compileFails(Stage::Vertex, "#define e +1\nvoid main() { int n = 1e; }", "numeric literal"));
    CHECK(compileFails(Stage::Vertex, "#version 300 es\nprecision mediump float;\nvoid main() {}\n#extension all : bogus\n", "behavior"));
    CHECK(compileFails(Stage::Vertex, "#version 300 es\nprecision mediump float;\nin vec4 p;\nlowp centroid out float x; void main() {}", "wrong order"));
    // Language rules a lenient driver does not enforce.
    CHECK(compileFails(Stage::Vertex, "float f(float x) { return f(x); } void main() { gl_Position = vec4(f(1.0)); }", "recursion"));
    CHECK(compileFails(Stage::Vertex, "uniform float u; void main() { const float c = u; gl_Position = vec4(c); }", "constant expression"));
    CHECK(compileFails(Stage::Vertex, "float f(const float x) { x = 1.0; return x; } void main() { gl_Position = vec4(f(1.0)); }", "const parameter"));
    CHECK(compileFails(Stage::Vertex, "void f(float x); void f(float x); void main() {}", "declared twice"));
    CHECK(compileFails(Stage::Vertex, "#version 300 es\nvoid main() { switch (1) { case 1: break; case 1: break; } }", "duplicate case"));
    CHECK(compileFails(Stage::Vertex, "#version 300 es\nvoid main() { float a[2][2]; }", "arrays of arrays"));
    CHECK(compileFails(Stage::Vertex, "#version 300 es\nuniform B { layout(foo) float x; }; void main() {}", "layout"));
    CHECK(compileFails(Stage::Fragment, "#version 300 es\nuniform float u; out mediump vec4 o; void main() { o = vec4(u); }", "no precision"));
    CHECK(compileFails(Stage::Vertex, "#version 300 es\nvoid f(int x); struct f { int a; }; void main() {}", "name already used"));
    // Link rules.
    {
        auto v = vpglsl::parse(Stage::Vertex, "uniform highp float u; void main() { gl_Position = vec4(u); }");
        auto f = vpglsl::parse(Stage::Fragment, "precision mediump float; uniform mediump float u; void main() { gl_FragColor = vec4(u); }");
        CHECK(v.shader && f.shader);
        auto l = vpglsl::link(v.shader, f.shader, {});
        CHECK(!l.ok && l.log.find("precisions") != std::string::npos);
    }
}

// Lenient mode (the default for games): shaders that break a conformance-only rule compile, and the rule is reported.
bool compilesLenient(Stage st, const std::string& src, const char* rule) {
    vpglsl::ParseResult r = vpglsl::parse(st, src);
    if (!r.shader) std::fprintf(stderr, "  lenient compile failed: %s\n", r.log.c_str());
    for (const std::string& l : r.lenient)
        if (l == rule) return r.shader != nullptr;
    std::fprintf(stderr, "  rule %s not reported (got %zu rules)\n", rule, r.lenient.size());
    return false;
}

void testLenient() {
    CHECK(vpglsl::strictMode() == false);
    CHECK(compilesLenient(Stage::Vertex, "#version 300 es\nvoid main() { float sample = 1.0; gl_Position = vec4(sample); }", "keyword_as_identifier"));
    CHECK(compilesLenient(Stage::Vertex, "void main() { float class = 1.0; gl_Position = vec4(class); }", "keyword_as_identifier"));
    CHECK(compilesLenient(Stage::Vertex, "#define A 1\n#define A 2\nvoid main() { gl_Position = vec4(A); }", "macro_redefinition"));
    CHECK(compilesLenient(Stage::Vertex, "#if A\n#endif\nvoid main() { gl_Position = vec4(0.0); }", "undefined_identifier"));
    CHECK(compilesLenient(Stage::Vertex, "#ifdef A B\n#endif\nvoid main() { gl_Position = vec4(0.0); }", "extra_tokens"));
    CHECK(compilesLenient(Stage::Fragment, "#version 310 es\nprecision mediump float;\nlayout(binding = 4000) uniform sampler2D s;\nout vec4 o;\nvoid main() { o = texture(s, vec2(0.0)); }", "binding_range"));
    CHECK(compilesLenient(Stage::Vertex, "#undef __LINE__\nvoid main() { gl_Position = vec4(0.0); }", "undef"));
    CHECK(compilesLenient(Stage::Vertex, "void main() { gl_Position = vec4(1e); }", "character"));
    CHECK(compilesLenient(Stage::Vertex, "#version 300\nvoid main() { gl_Position = vec4(0.0); }", "version_profile"));
    CHECK(compilesLenient(Stage::Vertex, "#version 300 es\nprecision mediump float;\nin vec4 p;\nlowp centroid out float x; void main() { gl_Position = p; }", "qualifier_order"));
    CHECK(compilesLenient(Stage::Vertex, "float f(float x) { return x; } float f(float x); void main() { gl_Position = vec4(f(1.0)); }", "function_redeclared"));
    CHECK(compilesLenient(Stage::Vertex, "uniform float u; void main() { const float c = u; gl_Position = vec4(c); }", "const_initializer"));
    CHECK(compilesLenient(Stage::Vertex, "#version 300 es\nvoid main() { float a[2][2]; gl_Position = vec4(a[0][0]); }", "arrays_of_arrays"));
    CHECK(compilesLenient(Stage::Fragment, "#version 300 es\nuniform float u; out mediump vec4 o; void main() { o = vec4(u); }", "unspecified_precision"));
    // Recursion cannot be compiled in either mode.
    CHECK(!vpglsl::parse(Stage::Vertex, "float f(float x) { return f(x); } void main() { gl_Position = vec4(f(1.0)); }").shader);
    // A conforming shader reports nothing, in either mode.
    vpglsl::ParseResult ok = vpglsl::parse(Stage::Vertex, "attribute vec4 p; void main() { gl_Position = p; }");
    CHECK(ok.shader && ok.lenient.empty());
    // Strict mode still rejects.
    vpglsl::setStrictMode(1);
    CHECK(compileFails(Stage::Vertex, "#define A 1\n#define A 2\nvoid main() {}", "redefinition"));
    vpglsl::setStrictMode(0);
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
    // row_major on a block reaches the matrices of struct members: mat3x2 as 2 row vec3s is 32 bytes (48 column-major).
    Program r = build(
        "#version 300 es\n"
        "struct S { highp mat3x2 m; };\n"
        "layout(std140, row_major) uniform R { highp float f; S s; highp float g; };\n"
        "in vec4 a_pos;\n"
        "void main() { gl_Position = a_pos * (f + g + s.m[0][1]); }\n",
        "#version 300 es\n"
        "precision mediump float;\n"
        "out vec4 o;\n"
        "void main() { o = vec4(1.0); }\n");
    CHECK(r.ok);
    if (r.ok && r.link.reflection.blocks.size() == 1) {
        const auto& m = r.link.reflection.blocks[0].members;
        CHECK(m.size() == 3 && m[1].name == "s.m" && m[1].rowMajor && m[1].matrixStride == 16);
        CHECK(m.size() == 3 && m[2].offset == 48);
    }
    // An instance array of blocks is indexed like an array: each element is one block (GL names "Inst[0]", "Inst[1]").
    Program q = build(
        "#version 300 es\n"
        "layout(std140) uniform Inst { highp float a; highp vec3 b; } blk[3];\n"
        "in vec4 a_pos;\n"
        "void main() { gl_Position = a_pos * (blk[0].a + blk[2].b.x); }\n",
        "#version 300 es\n"
        "precision mediump float;\n"
        "out vec4 o;\n"
        "void main() { o = vec4(1.0); }\n");
    CHECK(q.ok);
    if (q.ok) CHECK(q.link.reflection.blocks.size() == 3 && q.link.reflection.blocks[2].name == "Inst[2]");
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
    // A uniform array of structs holding samplers: every leaf is a uniform of its own, named as GL names it.
    Program st = build(
        "#version 300 es\n"
        "in vec4 a;\n"
        "void main() { gl_Position = a; }\n",
        "#version 300 es\n"
        "precision mediump float;\n"
        "struct S { sampler2D a; samplerCube b; mediump float x[2]; };\n"
        "uniform S u[2];\n"
        "out vec4 o;\n"
        "void main() { o = texture(u[1].a, vec2(0.5)) + texture(u[0].b, vec3(0.5)) + vec4(u[1].x[1]); }\n");
    CHECK(st.ok);
    if (st.ok) {
        const auto* ua = uniform(st, "u[1].a");
        const auto* ub = uniform(st, "u[0].b");
        const auto* ux = uniform(st, "u[1].x[0]");
        CHECK(ua && ua->glType == 0x8B5E && ub && ub->glType == 0x8B60 && ux && ux->arraySize == 2);
        CHECK(!uniform(st, "u[0].a"));
    }
    // ES 1.00 cube + EXT lod + shadow ext.
    Program q = build(
        "attribute vec4 a; varying vec3 d; uniform sampler2D vt; void main() { d = a.xyz; gl_Position = a + texture2DLod(vt, a.xy, 0.0); }",
        "#extension GL_EXT_shader_texture_lod : enable\n"
        "#extension GL_OES_standard_derivatives : enable\n"
        "precision mediump float; varying vec3 d; uniform samplerCube c; uniform sampler2D t;\n"
        "void main() { gl_FragColor = textureCube(c, d) + texture2DProj(t, vec4(d, 1.0)) + textureCubeLodEXT(c, d, 1.0) + vec4(fwidth(d.x)); }");
    CHECK(q.ok);
}

void testExternalSampler() {
    // ES 1.00: texture2D / texture2DProj with samplerExternalOES (sampled like a 2D texture).
    Program p = build(
        "attribute vec4 a; varying vec2 uv; void main() { uv = a.xy; gl_Position = a; }",
        "#extension GL_OES_EGL_image_external : require\n"
        "precision mediump float; varying vec2 uv; uniform samplerExternalOES vid;\n"
        "void main() { gl_FragColor = texture2D(vid, uv) + texture2DProj(vid, vec3(uv, 1.0)); }");
    CHECK(p.ok);
    if (p.ok) {
        const auto* v = uniform(p, "vid");
        CHECK(v && v->glType == 0x8D66);
    }
    // ES 3.00 with the essl3 extension: texture().
    Program q = build(
        "#version 300 es\nin vec4 a; out vec2 uv; void main() { uv = a.xy; gl_Position = a; }",
        "#version 300 es\n#extension GL_OES_EGL_image_external_essl3 : require\n"
        "precision mediump float; in vec2 uv; uniform samplerExternalOES vid; out vec4 o;\n"
        "void main() { o = texture(vid, uv) + textureProj(vid, vec3(uv, 1.0)); }");
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
    // A fragment input nobody writes links (real drivers read an undefined value; here 0): it is not a stage input
    // of the fragment shader, and the varyings that are written keep their locations.
    Program u = build(
        "#version 300 es\n"
        "in vec4 a; out vec4 v_a; out vec2 v_b;\n"
        "void main() { v_a = a; v_b = a.zw; gl_Position = a; }\n",
        "#version 300 es\n"
        "precision mediump float;\n"
        "in vec4 v_a; in vec2 v_missing; in mat2 m_missing[2]; in vec2 v_b; out vec4 o;\n"
        "void main() { o = v_a + vec4(v_missing, v_b) + vec4(m_missing[1][1], 0.0, 0.0); }\n");
    CHECK(u.ok);
    if (u.ok) {
        CHECK(!u.link.varyingDiag.empty() && u.link.varyingDiag.find("v_missing@-!unwritten") != std::string::npos);
        int ins = 0;
        for (const auto& in : u.fs.module.inputs) {
            if (in.builtin != vpir::Builtin::None) continue;
            ++ins;
            CHECK(in.location < 2);
        }
        CHECK(ins == 2);
    }
    // Varyings paired by layout(location) when their names differ; the value travels from vertex output to fragment input.
    Program w = build(
        "#version 300 es\n"
        "in vec4 a;\n"
        "layout(location = 1) out vec2 vs_uv; layout(location = 0) out vec4 vs_col; layout(location = 2) flat out int vs_id;\n"
        "void main() { vs_uv = a.xy; vs_col = a; vs_id = gl_InstanceID; gl_Position = a; }\n",
        "#version 300 es\n"
        "precision mediump float;\n"
        "layout(location = 1) in vec2 fs_uv; layout(location = 0) in vec4 fs_col; layout(location = 2) flat in int fs_id;\n"
        "out vec4 o;\n"
        "void main() { o = fs_col + vec4(fs_uv, float(fs_id), 0.0); }\n");
    CHECK(w.ok);
    if (w.ok) {
        CHECK(w.link.varyingDiag.empty());
        int paired = 0;
        for (const auto& out : w.vs.module.outputs) {
            if (out.builtin != vpir::Builtin::None) continue;
            for (const auto& in : w.fs.module.inputs) {
                if (in.builtin != vpir::Builtin::None || in.location != out.location) continue;
                CHECK(in.type == out.type);
                CHECK((in.interp == vpir::Interp::Flat) == (out.location == 2));
                ++paired;
            }
        }
        CHECK(paired == 3);
    }
    // Names differ and the locations do not match: the input is unwritten, not paired with an unrelated output.
    Program z = build(
        "#version 300 es\n"
        "in vec4 a; layout(location = 1) out vec2 vs_uv; void main() { vs_uv = a.xy; gl_Position = a; }\n",
        "#version 300 es\n"
        "precision mediump float;\n"
        "layout(location = 0) in vec2 fs_uv; out vec4 o; void main() { o = vec4(fs_uv, 0.0, 1.0); }\n");
    CHECK(z.ok && z.link.varyingDiag.find("fs_uv@0!unwritten") != std::string::npos);
    // An unused fragment input is not paired by location, so no type / interpolation check applies to it.
    Program un = build(
        "#version 300 es\n"
        "in vec4 a; layout(location = 0) out vec4 vs_i; void main() { vs_i = a; gl_Position = a; }\n",
        "#version 300 es\n"
        "precision mediump float;\n"
        "layout(location = 0) flat in vec4 fs_i; out vec4 o; void main() { o = vec4(1.0); }\n");
    CHECK(un.ok && un.link.varyingDiag.empty());
    // Automatic locations skip the explicit ones (declared later) instead of colliding with them.
    Program au = build(
        "#version 300 es\n"
        "in vec4 a; out vec4 av; layout(location = 0) out vec4 vs_b; void main() { av = a; vs_b = a; gl_Position = a; }\n",
        "#version 300 es\n"
        "precision mediump float;\n"
        "in vec4 av; layout(location = 0) in vec4 fs_b; out vec4 o; void main() { o = av + fs_b; }\n");
    CHECK(au.ok);
    if (au.ok) {
        uint32_t seen = 0;
        for (const auto& in : au.fs.module.inputs)
            if (in.builtin == vpir::Builtin::None) seen |= 1u << in.location;
        CHECK(seen == 3u);
    }
    // Two different varyings asking for the same explicit location is a link error.
    {
        auto dv = vpglsl::parse(Stage::Vertex, "#version 300 es\nin vec4 a; out vec4 p; out vec4 q; void main() { p = a; q = a; gl_Position = a; }");
        auto df = vpglsl::parse(Stage::Fragment, "#version 300 es\nprecision mediump float;\n"
                                                  "layout(location = 1) in vec4 p; layout(location = 1) in vec4 q; out vec4 o; void main() { o = p + q; }");
        CHECK(dv.shader && df.shader);
        if (dv.shader && df.shader) {
            auto r = vpglsl::link(dv.shader, df.shader, {});
            CHECK(!r.ok && r.log.find("same location") != std::string::npos);
        }
    }
    // ES 1.00: a varying the vertex shader lacks links too.
    auto v = vpglsl::parse(Stage::Vertex, "void main() { gl_Position = vec4(0.0); }");
    auto f = vpglsl::parse(Stage::Fragment, "precision mediump float; varying vec2 x; void main() { gl_FragColor = vec4(x, 0.0, 1.0); }");
    CHECK(v.shader && f.shader && vpglsl::link(v.shader, f.shader, {}).ok);
    // Same name with different types still fails, and the diagnostic names the varying.
    auto v2 = vpglsl::parse(Stage::Vertex, "#version 300 es\nout vec2 x; void main() { x = vec2(0.0); gl_Position = vec4(0.0); }");
    auto f2 = vpglsl::parse(Stage::Fragment, "#version 300 es\nprecision mediump float; in vec3 x; out vec4 o; void main() { o = vec4(x, 1.0); }");
    CHECK(v2.shader && f2.shader);
    if (v2.shader && f2.shader) {
        auto r = vpglsl::link(v2.shader, f2.shader, {});
        CHECK(!r.ok && r.varyingDiag.find("x@-!type") != std::string::npos);
    }
}

// ---------------------------------------------------------------------------------------------
// ES 3.1 / 3.2: compute, shader storage blocks, images, atomics.

struct ComputeProgram {
    bool ok = false;
    vpglsl::LinkResult link;
    Lowered cs;
};

ComputeProgram buildCompute(const std::string& src) {
    ComputeProgram p;
    auto c = vpglsl::parse(Stage::Compute, src);
    if (!c.shader) {
        std::fprintf(stderr, "  compile: %s\n", c.log.c_str());
        return p;
    }
    p.link = vpglsl::link(c.shader, nullptr, {});
    if (!p.link.ok) {
        std::fprintf(stderr, "  link: %s\n", p.link.log.c_str());
        return p;
    }
    p.cs = lowerStage(p.link.computeSpirv, Stage::Compute, p.link.reflection);
    p.ok = p.cs.ok;
    return p;
}

bool has(const std::string& hay, const char* needle) { return hay.find(needle) != std::string::npos; }

const vpglsl::Variable* member(const vpglsl::UniformBlock& b, const std::string& n) {
    for (const auto& m : b.members)
        if (m.name == n) return &m;
    return nullptr;
}

const char* kTrivialVs = "#version 310 es\nin vec4 a; out vec2 uv; void main() { uv = a.xy; gl_Position = a; }";

void testCompute() {
    // Builtins, local_size, SSBO with an unsized array, barrier, shared memory, memory barriers.
    ComputeProgram p = buildCompute(
        "#version 310 es\n"
        "layout(local_size_x = 8, local_size_y = 4) in;\n"
        "layout(std430, binding = 0) buffer Data { uint count; float values[]; } d;\n"
        "shared float tile[32];\n"
        "void main() {\n"
        "    uint i = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x;\n"
        "    tile[gl_LocalInvocationIndex] = float(i) + float(gl_LocalInvocationID.x + gl_WorkGroupID.z);\n"
        "    memoryBarrierShared();\n"
        "    barrier();\n"
        "    float s = tile[(gl_LocalInvocationIndex + 1u) % 32u];\n"
        "    groupMemoryBarrier();\n"
        "    memoryBarrier(); memoryBarrierBuffer(); memoryBarrierImage();\n"
        "    if (i < uint(d.values.length())) d.values[i] = s;\n"
        "    if (gl_LocalInvocationIndex == 0u) d.count = uint(d.values.length());\n"
        "}\n");
    CHECK(p.ok);
    if (p.ok) {
        const auto& r = p.link.reflection;
        CHECK(r.localSize[0] == 8 && r.localSize[1] == 4 && r.localSize[2] == 1);
        CHECK(r.sharedBytes == 128);
        CHECK(r.storageBlocks.size() == 1 && r.storageBlocks[0].binding == vpglsl::kStorageBase);
        CHECK(r.storageBlocks[0].initialBinding == 0);
        CHECK(has(p.cs.msl, "threadgroup"));
        CHECK(has(p.cs.msl, "threadgroup_barrier"));
        CHECK(p.link.computeSpirv.size() > 5);
    }
    // gl_WorkGroupSize in a constant context of the shader is its declared size.
    ComputeProgram q = buildCompute(
        "#version 310 es\n"
        "layout(local_size_x = 16) in;\n"
        "layout(binding = 1) buffer B { uint v[]; } b;\n"
        "void main() { b.v[gl_GlobalInvocationID.x] = gl_WorkGroupSize.x; }\n");
    CHECK(q.ok);
    // gl_WorkGroupSize.x is a constant expression (array sizes).
    ComputeProgram wgs = buildCompute(
        "#version 310 es\n"
        "layout(local_size_x = 4, local_size_y = 2) in;\n"
        "const uint kSize = gl_WorkGroupSize.x * gl_WorkGroupSize.y;\n"
        "layout(binding = 0) buffer R { bool ok[kSize]; } r;\n"
        "void main() { r.ok[gl_LocalInvocationIndex] = true; }\n");
    CHECK(wgs.ok);
    // Errors.
    CHECK(compileFails(Stage::Compute, "#version 310 es\nvoid main() {}\n", "local_size"));
    CHECK(compileFails(Stage::Compute, "#version 310 es\nlayout(local_size_x = 2000) in;\nvoid main() {}\n", "work group size"));
    CHECK(compileFails(Stage::Vertex, "#version 310 es\nshared int x;\nvoid main() { gl_Position = vec4(0.0); }\n", "shared"));
    CHECK(compileFails(Stage::Fragment,
                       "#version 310 es\nprecision mediump float;\nout vec4 o;\nvoid main() { barrier(); o = vec4(0.0); }\n",
                       "barrier"));
    CHECK(compileFails(Stage::Compute, "#version 310 es\nlayout(local_size_x = 1) in;\nout vec4 o;\nvoid main() {}\n", "compute shaders"));
    CHECK(compileFails(Stage::Compute, "#version 300 es\nlayout(local_size_x = 1) in;\nvoid main() {}\n", "#version 310"));
    // Shared memory over the Metal limit fails to link.
    auto big = vpglsl::parse(Stage::Compute, "#version 310 es\nlayout(local_size_x = 1) in;\nshared vec4 t[3000];\nvoid main() { t[0] = vec4(1.0); }\n");
    CHECK(big.shader && !vpglsl::link(big.shader, nullptr, {}).ok);
}

void testStorageBlocks() {
    // std430 layout and reflection.
    ComputeProgram p = buildCompute(
        "#version 310 es\n"
        "layout(local_size_x = 1) in;\n"
        "struct P { float a; vec3 b; };\n"
        "layout(std430, binding = 3) buffer Blk {\n"
        "    vec3 a; float b; vec4 c; mat3 m; float f3[3]; vec3 v2[2]; P ps[2];\n"
        "    readonly vec2 ro; writeonly uint wo; coherent restrict int ci; float tail[];\n"
        "} blk;\n"
        "void main() {\n"
        "    blk.a = vec3(1.0); blk.b = blk.c.x; blk.m[1] = vec3(2.0); blk.f3[2] = 1.0; blk.v2[1].y = 3.0;\n"
        "    blk.ps[1].b.z = blk.ps[0].a; blk.wo = 7u; blk.ci += 2; blk.tail[gl_GlobalInvocationID.x] = blk.ro.x;\n"
        "    blk.ps[0] = blk.ps[1];\n"
        "}\n");
    CHECK(p.ok);
    if (!p.ok) return;
    const auto& b = p.link.reflection.storageBlocks.at(0);
    CHECK(b.initialBinding == 3);
    auto off = [&](const char* n) { const auto* m = member(b, n); return m ? int(m->offset) : -1; };
    CHECK(off("Blk.a") == 0 && off("Blk.b") == 12 && off("Blk.c") == 16);
    CHECK(off("Blk.m") == 32);  // after c (16..32); mat3 stride 16, size 48
    const auto* m = member(b, "Blk.m");
    CHECK(m && m->matrixStride == 16);
    CHECK(off("Blk.f3[0]") == 80);                   // float[3] tightly packed (stride 4)
    const auto* f3 = member(b, "Blk.f3[0]");
    CHECK(f3 && f3->arrayStride == 4 && f3->arraySize == 3);
    CHECK(off("Blk.v2[0]") == 96);                   // vec3[2]: align 16, stride 16
    const auto* v2 = member(b, "Blk.v2[0]");
    CHECK(v2 && v2->arrayStride == 16);
    CHECK(off("Blk.ps[0].a") == 128 && off("Blk.ps[0].b") == 144);  // struct {float; vec3}: align 16, size 32
    // The top-level array of structs is listed through element 0 only, with its size and stride.
    CHECK(off("Blk.ps[1].a") == -1);
    const auto* ps0 = member(b, "Blk.ps[0].a");
    CHECK(ps0 && ps0->topLevelArraySize == 2 && ps0->topLevelArrayStride == 32);
    const auto* tail = member(b, "Blk.tail[0]");
    CHECK(tail && tail->arraySize == 0 && tail->arrayStride == 4);
    // The same block as std140 has the array stride rounded to 16.
    ComputeProgram q = buildCompute(
        "#version 310 es\n"
        "layout(local_size_x = 1) in;\n"
        "layout(std140, binding = 0) buffer B { float f[3]; } b;\n"
        "void main() { b.f[1] = 1.0; }\n");
    CHECK(q.ok);
    if (q.ok) {
        const auto* f = member(q.link.reflection.storageBlocks.at(0), "B.f[0]");
        CHECK(f && f->arrayStride == 16);
    }
    // Default layout statement.
    ComputeProgram r = buildCompute(
        "#version 310 es\n"
        "layout(local_size_x = 1) in;\n"
        "layout(std430) buffer;\n"
        "buffer B { float f[3]; };\n"
        "void main() { f[1] = 1.0; }\n");
    CHECK(r.ok);
    if (r.ok) {
        const auto* f = member(r.link.reflection.storageBlocks.at(0), "f[0]");
        CHECK(f && f->arrayStride == 4);
    }
    // Storage blocks in a graphics pair; a vertex shader reading, a fragment shader writing.
    Program g = build(
        "#version 310 es\nlayout(std430, binding = 0) readonly buffer In { vec4 pos[]; } src;\n"
        "void main() { gl_Position = src.pos[gl_VertexID]; }",
        "#version 310 es\nprecision mediump float;\nlayout(std430, binding = 1) buffer Out { uint hits; };\nout vec4 o;\n"
        "void main() { atomicAdd(hits, 1u); o = vec4(1.0); }");
    CHECK(g.ok);
    if (g.ok) {
        CHECK(g.link.reflection.storageBlocks.size() == 2);
        CHECK(g.link.reflection.storageBlocks[0].stages == 1u << unsigned(Stage::Vertex));
        CHECK(g.link.reflection.storageBlocks[1].stages == 1u << unsigned(Stage::Fragment));
    }
    // Errors: writing a readonly block member, an unsized array that is not last, a uniform-block misuse.
    CHECK(compileFails(Stage::Compute,
                       "#version 310 es\nlayout(local_size_x = 1) in;\nbuffer B { readonly float f; } b;\nvoid main() { b.f = 1.0; }\n",
                       "l-value"));
    CHECK(compileFails(Stage::Compute,
                       "#version 310 es\nlayout(local_size_x = 1) in;\nbuffer B { float f[]; float g; } b;\nvoid main() {}\n",
                       "unsized"));
    CHECK(compileFails(Stage::Compute,
                       "#version 310 es\nlayout(local_size_x = 1) in;\nuniform B { float f[]; } b;\nvoid main() {}\n", "unsized"));
    CHECK(compileFails(Stage::Vertex, "#version 300 es\nbuffer B { float f; } b;\nvoid main() { gl_Position = vec4(b.f); }\n",
                       "3.10"));
}

void testImages() {
    ComputeProgram p = buildCompute(
        "#version 310 es\n"
        "layout(local_size_x = 8, local_size_y = 8) in;\n"
        "layout(rgba8, binding = 0) writeonly uniform highp image2D dst;\n"
        "layout(rgba32f, binding = 1) readonly uniform highp image2D src;\n"
        "layout(r32ui, binding = 2) coherent uniform highp uimage2D counters;\n"
        "layout(r32i, binding = 3) uniform highp iimage3D vol;\n"
        "layout(rgba16f, binding = 4) readonly uniform highp image2DArray layers;\n"
        "layout(rgba8, binding = 5) writeonly uniform highp imageCube cube;\n"
        "layout(r32f, binding = 6) uniform highp image2D fl;\n"
        "void main() {\n"
        "    ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n"
        "    vec4 c = imageLoad(src, p) + imageLoad(layers, ivec3(p, 1));\n"
        "    ivec2 sz = imageSize(src); ivec3 s3 = imageSize(vol); ivec3 sl = imageSize(layers); ivec2 sc = imageSize(cube);\n"
        "    imageStore(dst, p, c * float(sz.x + s3.z + sl.z + sc.x));\n"
        "    imageStore(cube, ivec3(p, 2), c);\n"
        "    uint old = imageAtomicAdd(counters, p, 1u);\n"
        "    old += imageAtomicMax(counters, p, 5u) + imageAtomicMin(counters, p, 9u) + imageAtomicAnd(counters, p, 255u);\n"
        "    old += imageAtomicOr(counters, p, 1u) + imageAtomicXor(counters, p, 2u) + imageAtomicExchange(counters, p, old);\n"
        "    old += imageAtomicCompSwap(counters, p, 3u, 4u);\n"
        "    int v = imageAtomicAdd(vol, ivec3(p, 0), 1) + imageAtomicCompSwap(vol, ivec3(p, 1), 0, 1);\n"
        "    imageStore(fl, p, vec4(imageLoad(fl, p).x + float(v + int(old))));\n"
        "}\n");
    CHECK(p.ok);
    if (p.ok) {
        const auto& r = p.link.reflection;
        auto img = [&](const char* n) -> const vpglsl::Variable* {
            for (const auto& u : r.uniforms)
                if (u.name == n) return &u;
            return nullptr;
        };
        const auto* dst = img("dst");
        CHECK(dst && dst->glType == 0x904D && dst->unit == 0 && dst->binding >= int(vpglsl::kImageBase));
        const auto* ctr = img("counters");
        CHECK(ctr && ctr->glType == 0x9063 && ctr->unit == 2);
        const auto* vol = img("vol");
        CHECK(vol && vol->glType == 0x9059);
        const auto* lay = img("layers");
        CHECK(lay && lay->glType == 0x9053);
        const auto* cube = img("cube");
        CHECK(cube && cube->glType == 0x9050);
        CHECK(has(p.cs.msl, "write("));
        CHECK(has(p.cs.msl, "atomic"));
    }
    // Images in a fragment shader (arrays of images, passed to a function).
    Program g = build(kTrivialVs,
        "#version 310 es\nprecision mediump float;\n"
        "layout(rgba8, binding = 0) uniform mediump image2D imgs[2];\n"
        "layout(rgba8, binding = 2) readonly uniform mediump image2D one;\n"
        "in vec2 uv; out vec4 o;\n"
        "vec4 fetch(readonly layout(rgba8) mediump image2D im, ivec2 p) { return imageLoad(im, p); }\n"
        "void main() { ivec2 p = ivec2(uv); imageStore(imgs[1], p, fetch(one, p)); o = imageLoad(imgs[0], p); }\n");
    CHECK(g.ok);
    if (g.ok) {
        const vpglsl::Variable* a = nullptr;
        for (const auto& u : g.link.reflection.uniforms)
            if (u.name == "imgs[0]") a = &u;
        CHECK(a && a->isArray && a->arraySize == 2 && a->unit == 0);
    }
    // Errors.
    const std::string head = "#version 310 es\nlayout(local_size_x = 1) in;\n";
    CHECK(compileFails(Stage::Compute, head + "layout(rgba8) uniform image2D i;\nvoid main() { imageStore(i, ivec2(0), uvec4(1u)); }\n", "no matching"));
    CHECK(compileFails(Stage::Compute, head + "layout(rgba32f) uniform image2D i;\nvoid main() { imageStore(i, ivec3(0), vec4(1.0)); }\n", "no matching"));
    CHECK(compileFails(Stage::Compute, head + "layout(rgba8) uniform image2D i;\nvoid main() { imageAtomicAdd(i, ivec2(0), 1u); }\n", "no matching"));
    CHECK(compileFails(Stage::Compute, head + "layout(r32ui) uniform image2D i;\nvoid main() {}\n", "format"));
    CHECK(compileFails(Stage::Compute, head + "layout(r32f) uniform image2D i;\nvoid main() { imageAtomicExchange(i, ivec2(0), 1.0); }\n", "no matching"));
    CHECK(compileFails(Stage::Compute, head + "void main() { image2D i; }\n", "local"));
    CHECK(compileFails(Stage::Vertex, "#version 300 es\nlayout(rgba8) uniform image2D i;\nvoid main() { gl_Position = vec4(0.0); }\n", "unknown type"));
}

void testAtomics() {
    // Atomics on buffer and shared memory.
    ComputeProgram p = buildCompute(
        "#version 310 es\n"
        "layout(local_size_x = 64) in;\n"
        "layout(std430, binding = 0) buffer B { uint u; int i; uint hist[4]; uint arr[]; } b;\n"
        "shared uint sh; shared int si[4];\n"
        "void main() {\n"
        "    if (gl_LocalInvocationIndex == 0u) sh = 0u;\n"
        "    barrier();\n"
        "    uint x = atomicAdd(b.u, 1u) + atomicMin(b.u, 5u) + atomicMax(b.u, 2u) + atomicAnd(b.u, 255u) + atomicOr(b.u, 1u);\n"
        "    x += atomicXor(b.u, 3u) + atomicExchange(b.u, x) + atomicCompSwap(b.u, 7u, 8u);\n"
        "    int y = atomicAdd(b.i, -1) + atomicMax(b.i, 3) + atomicMin(b.i, 4) + atomicCompSwap(b.i, 1, 2);\n"
        "    x += atomicAdd(b.arr[gl_LocalInvocationIndex], 1u) + atomicAdd(b.hist[x & 3u], 1u);\n"
        "    x += atomicOr(sh, 1u) + atomicAdd(sh, x);\n"
        "    y += atomicAdd(si[2], y) + atomicCompSwap(si[3], 0, 1);\n"
        "    barrier();\n"
        "    if (gl_LocalInvocationIndex == 1u) b.i = int(sh) + y + int(x);\n"
        "}\n");
    CHECK(p.ok);
    if (p.ok) CHECK(has(p.cs.msl, "atomic"));
    // Atomic counters.
    ComputeProgram c = buildCompute(
        "#version 310 es\n"
        "layout(local_size_x = 1) in;\n"
        "layout(binding = 0, offset = 0) uniform atomic_uint c0;\n"
        "layout(binding = 0) uniform atomic_uint c1;\n"
        "layout(binding = 1, offset = 8) uniform atomic_uint c2[2];\n"
        "layout(std430, binding = 0) buffer O { uint v[]; } o;\n"
        "void main() {\n"
        "    uint a = atomicCounterIncrement(c0); uint b = atomicCounterDecrement(c1); uint c = atomicCounter(c0);\n"
        "    uint d = atomicCounterIncrement(c2[1]); uint e = atomicCounter(c2[a & 1u]);\n"
        "    o.v[0] = a + b + c + d + e;\n"
        "}\n");
    CHECK(c.ok);
    if (c.ok) {
        const auto& r = c.link.reflection;
        CHECK(r.atomicBuffers.size() == 2);
        CHECK(r.atomicBuffers.size() == 2 && r.atomicBuffers[0].glBinding == 0 && r.atomicBuffers[0].size == 8);
        CHECK(r.atomicBuffers.size() == 2 && r.atomicBuffers[1].glBinding == 1 && r.atomicBuffers[1].size == 16 &&
              r.atomicBuffers[1].binding == vpglsl::kAtomicBase + 1);
        auto var = [&](const char* n) -> const vpglsl::Variable* {
            for (const auto& u : r.uniforms)
                if (u.name == n) return &u;
            return nullptr;
        };
        const auto* c1 = var("c1");
        CHECK(c1 && c1->glType == 0x92DB && c1->offset == 4 && c1->atomicBinding == 0 && c1->binding == -1);
        const auto* c2 = var("c2[0]");
        CHECK(c2 && c2->offset == 8 && c2->atomicBinding == 1 && c2->arraySize == 2);
        CHECK(r.defaultBlockSize == 0);  // counters are not in the default block
    }
    // Atomic counters in a fragment shader.
    Program g = build(kTrivialVs,
        "#version 310 es\nprecision mediump float;\nlayout(binding = 0) uniform atomic_uint hits;\nin vec2 uv; out vec4 o;\n"
        "void main() { o = vec4(float(atomicCounterIncrement(hits))); }\n");
    CHECK(g.ok);
    // Errors: an atomic on a local / on a readonly member / counters as locals.
    const std::string head = "#version 310 es\nlayout(local_size_x = 1) in;\n";
    CHECK(compileFails(Stage::Compute, head + "void main() { uint x = 0u; atomicAdd(x, 1u); }\n", "buffer or shared"));
    CHECK(compileFails(Stage::Compute, head + "buffer B { readonly uint u; } b;\nvoid main() { atomicAdd(b.u, 1u); }\n", "l-value"));
    CHECK(compileFails(Stage::Compute, head + "buffer B { float f; } b;\nvoid main() { atomicAdd(b.f, 1.0); }\n", "no matching"));
    CHECK(compileFails(Stage::Compute, head + "void main() { atomic_uint c; }\n", "local"));
    CHECK(compileFails(Stage::Vertex, "#version 300 es\nvoid main() { uint x = 0u; atomicAdd(x, 1u); gl_Position = vec4(0.0); }\n", "no matching"));
    // A default offset (`layout(binding, offset) uniform atomic_uint;`) must be a multiple of 4 and moves the next counter.
    CHECK(compileFails(Stage::Compute, head + "layout(binding = 0, offset = 1) uniform atomic_uint;\nlayout(binding = 0) uniform atomic_uint c;\nvoid main() { atomicCounterIncrement(c); }\n", "multiple of 4"));
    {
        Program d = build(kTrivialVs,
            "#version 310 es\nprecision mediump float;\nlayout(binding = 0, offset = 8) uniform atomic_uint;\nlayout(binding = 0) uniform atomic_uint c;\n"
            "in vec2 uv; out vec4 o;\nvoid main() { o = vec4(float(atomicCounterIncrement(c))); }\n");
        CHECK(d.ok);
        bool at8 = false;
        for (const auto& u : d.link.reflection.uniforms) at8 |= u.name == "c" && u.offset == 8;
        CHECK(at8);
    }
    {  // A uniform shared by both stages takes the explicit location of whichever stage gives one.
        Program l = build("#version 310 es\nin vec4 pos; uniform float u;\nvoid main() { gl_Position = pos * u; }\n",
                          "#version 310 es\nprecision mediump float;\nlayout(location = 7) uniform float u;\nout vec4 o;\nvoid main() { o = vec4(u); }\n");
        CHECK(l.ok);
        int loc = -1;
        for (const auto& u : l.link.reflection.uniforms)
            if (u.name == "u") loc = u.uniformLocation;
        CHECK(loc == 7);
    }
}

void testGather() {
    // textureGather (component, offset, shadow), texelFetchOffset on 3D, cube map arrays in ES 3.20.
    Program p = build(kTrivialVs,
        "#version 320 es\nprecision mediump float;\n"
        "uniform sampler2D t2; uniform sampler2DArray ta; uniform samplerCube tc; uniform sampler2DShadow ts;\n"
        "uniform sampler3D t3; uniform isampler2D ti; uniform samplerCubeArray tca; uniform samplerCubeArrayShadow tcs;\n"
        "uniform usamplerCubeArray tcu;\n"
        "in vec2 uv; out vec4 o;\n"
        "void main() {\n"
        "    vec4 g = textureGather(t2, uv) + textureGather(t2, uv, 2) + textureGatherOffset(t2, uv, ivec2(1, -1), 1);\n"
        "    g += textureGatherOffset(t2, uv, ivec2(uv * 3.0)) + textureGather(ta, vec3(uv, 1.0), 3) + textureGather(tc, vec3(uv, 1.0));\n"
        "    g += textureGather(ts, uv, 0.5) + textureGatherOffset(ts, uv, 0.5, ivec2(0, 1));\n"
        "    g += vec4(textureGather(ti, uv, 1));\n"
        "    g += texelFetchOffset(t3, ivec3(1, 2, 3), 0, ivec3(1, 0, -1)) + texelFetchOffset(t2, ivec2(3), 1, ivec2(2, 1));\n"
        "    g += texture(tca, vec4(uv, 1.0, 2.0)) + textureLod(tca, vec4(uv, 0.0, 1.0), 1.0) + textureGrad(tca, vec4(uv, 1.0, 0.0), vec3(0.1), vec3(0.1));\n"
        "    g += textureGather(tca, vec4(uv, 1.0, 1.0), 2) + vec4(float(textureSize(tca, 0).z) + float(textureSize(tcs, 1).z));\n"
        "    g += vec4(texture(tcs, vec4(uv, 1.0, 0.0), 0.5)) + vec4(textureLod(tcs, vec4(uv, 1.0, 0.0), 0.5, 0.0)) + vec4(texture(tcu, vec4(uv, 0.0, 1.0)));\n"
        "    o = g;\n"
        "}\n");
    CHECK(p.ok);
    if (p.ok) {
        const auto* tca = uniform(p, "tca");
        CHECK(tca && tca->glType == 0x900C);
        const auto* tcs = uniform(p, "tcs");
        CHECK(tcs && tcs->glType == 0x900D);
        const auto* tcu = uniform(p, "tcu");
        CHECK(tcu && tcu->glType == 0x900F);
    }
    // Offsets beyond Metal's -8..7 (Quest 3 reports -32..31): the compiler moves the coordinate by whole texels.
    Program far = build(kTrivialVs,
        "#version 310 es\nprecision mediump float;\nuniform sampler2D t; uniform sampler2DArray ta; in vec2 uv; out vec4 o;\n"
        "void main() { o = textureGatherOffset(t, uv, ivec2(20, -31)) + textureGatherOffset(ta, vec3(uv, 1.0), ivec2(-32, 3), 1); }\n");
    CHECK(far.ok);
    if (far.ok) {
        CHECK(far.fs.msl.find("int2(20, -31)") == std::string::npos && far.fs.msl.find("int2(-32, 3)") == std::string::npos);
        CHECK(far.fs.msl.find("get_width") != std::string::npos);
    }
    // A dynamic offset (ES 3.2 / gpu_shader5): Metal's hardware offset is 4 bits wide, so offsets outside -8..7 move the
    // coordinate (the size query is in the shader) and the operand is selected to zero for those.
    Program dyn = build(kTrivialVs,
        "#version 320 es\nprecision mediump float;\nuniform sampler2D t; uniform sampler2DArray ta; uniform sampler2DShadow ts; in vec2 uv; out vec4 o;\n"
        "void main() { ivec2 d = ivec2(int(uv.x * 64.0) - 32, int(uv.y * 64.0) - 32);\n"
        "    o = textureGatherOffset(t, uv, d, 1) + textureGatherOffset(ta, vec3(uv, 1.0), d) + textureGatherOffset(ts, uv, 0.5, d); }\n");
    CHECK(dyn.ok);
    if (dyn.ok) {
        CHECK(dyn.fs.msl.find("get_width") != std::string::npos);
        CHECK(dyn.fs.msl.find("gather") != std::string::npos);
    }
    // ES 3.10: cube map arrays need the extension (and the layer's say-so); textureGather exists.
    const std::string fs310 = "#version 310 es\nprecision mediump float;\nuniform samplerCubeArray t; in vec2 uv; out vec4 o;\nvoid main() { o = texture(t, vec4(uv, 0.0, 0.0)); }\n";
    CHECK(compileFails(Stage::Fragment, fs310, "unknown type"));
    Program q = build(kTrivialVs,
        "#version 310 es\nprecision mediump float;\nuniform sampler2D t; in vec2 uv; out vec4 o;\n"
        "void main() { o = textureGather(t, uv) + textureGatherOffset(t, uv, ivec2(1)); }\n");
    CHECK(q.ok);
    // ES 3.10 arrays of arrays: `float[2] a[3]` is 3 arrays of 2; implicit inner sizes come from the constructor / initialiser.
    Program aoa = build(kTrivialVs,
        "#version 310 es\nprecision mediump float;\nin vec2 uv; out vec4 o;\nvoid main() {\n"
        "  float[2] a[3] = float[3][2](float[2](uv.x, uv.y), float[2](uv.x, uv.y), float[2](uv.x, uv.y));\n"
        "  float x[][] = float[][](float[](uv.x, uv.y), float[](uv.y, uv.x));\n"
        "  float[][] p = float[2][2](float[2](uv.x, uv.y), float[2](uv.x, uv.y)), q = float[2][3](float[3](uv.x, uv.y, 1.0), float[3](uv.x, uv.y, 1.0));\n"
        "  o = vec4(a[2][1] + x[1][1] + p[0][1] + q[1][2], float(x.length() + q[0].length() + a[0].length()), 0.0, 1.0);\n}\n");
    CHECK(aoa.ok);
    // ES 3.10: gather over a shadow array (its table has textureGather[Offset] for sampler2DArrayShadow, unlike texture/textureOffset).
    Program sa = build(kTrivialVs,
        "#version 310 es\nprecision mediump float;\nuniform highp sampler2DArrayShadow t; in vec2 uv; out vec4 o;\n"
        "void main() { o = textureGather(t, vec3(uv, 1.0), 0.5) + textureGatherOffset(t, vec3(uv, 1.0), 0.5, ivec2(-8, 7)); }\n");
    CHECK(sa.ok);
    // Not in ES 3.00.
    CHECK(compileFails(Stage::Fragment, "#version 300 es\nprecision mediump float;\nuniform sampler2D t; in vec2 uv; out vec4 o;\nvoid main() { o = textureGather(t, uv); }\n", "no matching"));
    // gl_HelperInvocation: ES 3.10 and up.
    Program h = build(kTrivialVs,
        "#version 310 es\nprecision mediump float;\nin vec2 uv; out vec4 o;\nvoid main() { o = gl_HelperInvocation ? vec4(0.0) : vec4(uv, 0.0, 1.0); }\n");
    CHECK(h.ok);
    CHECK(compileFails(Stage::Fragment, "#version 300 es\nprecision mediump float;\nout vec4 o;\nvoid main() { o = vec4(gl_HelperInvocation); }\n", "undeclared"));
    // Compute built-in constants (ES 3.10) and the compute default precisions (highp float / int).
    ComputeProgram mc = buildCompute(
        "#version 310 es\nlayout(local_size_x = 1) in;\nlayout(std430, binding = 0) buffer B { int v[]; };\n"
        "void main() { float zero = 0.0; v[0] = gl_MaxComputeWorkGroupSize.x + gl_MaxComputeWorkGroupCount.y + int(zero); }\n");
    CHECK(mc.ok);
    // With the extension caps on, cube map arrays work in 3.10.
    vpglsl::Caps saved = vpglsl::caps();
    vpglsl::Caps on = saved;
    on.cubeMapArray = true;
    vpglsl::setCaps(on);
    Program c = build(kTrivialVs,
        "#version 310 es\n#extension GL_EXT_texture_cube_map_array : require\nprecision mediump float;\n"
        "uniform samplerCubeArray t; in vec2 uv; out vec4 o;\nvoid main() { o = texture(t, vec4(uv, 0.0, 0.0)); }\n");
    CHECK(c.ok);
    vpglsl::setCaps(saved);
}

void testFramebufferFetch() {
    const std::string fs3 =
        "#version 300 es\n#extension GL_EXT_shader_framebuffer_fetch : require\nprecision mediump float;\n"
        "in vec2 uv;\nlayout(location = 0) inout vec4 color;\nlayout(location = 1) inout vec3 normal;\n"
        "void main() { color = color * 0.5 + vec4(uv, 0.0, 1.0); normal = normal + color.xyz; }\n";
    const std::string vs3 = "#version 300 es\nin vec4 a; out vec2 uv; void main() { uv = a.xy; gl_Position = a; }";
    const std::string fs1 =
        "#extension GL_EXT_shader_framebuffer_fetch : require\nprecision mediump float;\n"
        "void main() { gl_FragColor = gl_LastFragData[0] * 0.5 + gl_LastFragData[1]; }\n";
    const std::string vs1 = "attribute vec4 a; void main() { gl_Position = a; }";
    const std::string fsArm =
        "#extension GL_ARM_shader_framebuffer_fetch : require\nprecision mediump float;\n"
        "void main() { gl_FragColor = gl_LastFragColorARM * 0.5; }\n";
    // Not announced by default: the macro is absent and `require` fails.
    {
        std::string log;
        int v = 0;
        std::string out = vpglsl::preprocess("#ifdef GL_EXT_shader_framebuffer_fetch\nint yes;\n#endif\nint end;\n", Stage::Fragment, log, v);
        CHECK(!has(out, "yes") && has(out, "end"));
        CHECK(compileFails(Stage::Fragment, fs3, "not supported"));
        CHECK(compileFails(Stage::Fragment, fs1, "not supported"));
    }
    vpglsl::Caps saved = vpglsl::caps();
    vpglsl::Caps on = saved;
    on.framebufferFetch = true;
    vpglsl::setCaps(on);
    {
        std::string log;
        int v = 0;
        std::string out = vpglsl::preprocess("#ifdef GL_EXT_shader_framebuffer_fetch\nint yes;\n#endif\n", Stage::Fragment, log, v);
        CHECK(has(out, "yes"));
    }
    Program p = build(vs3, fs3);
    CHECK(p.ok);
    if (p.ok) {
        CHECK(p.link.reflection.framebufferFetchMask == 3u);
        CHECK(has(p.fs.msl, "image_fetch") || has(p.fs.msl, "read(") || p.fs.ok);
    }
    Program q = build(vs1, fs1);
    CHECK(q.ok);
    if (q.ok) CHECK(q.link.reflection.framebufferFetchMask == 3u);
    Program a = build(vs1, fsArm);
    CHECK(a.ok);
    if (a.ok) CHECK(a.link.reflection.framebufferFetchMask == 1u);
    // `inout` globals still need the extension / ES 3.00+.
    vpglsl::setCaps(saved);
    CHECK(compileFails(Stage::Fragment, "#version 300 es\nprecision mediump float;\nlayout(location = 0) inout vec4 c;\nvoid main() { c = c; }\n", "inout"));
}

void testExtensions() {
    // Extensions without an implementation are neither defined as macros nor accepted by `require`.
    const char* dead[] = {"GL_EXT_YUV_target", "GL_OES_sample_variables", "GL_OES_shader_multisample_interpolation",
                          "GL_EXT_clip_cull_distance", "GL_EXT_gpu_shader5", "GL_OES_texture_storage_multisample_2d_array",
                          "GL_EXT_shader_framebuffer_fetch", "GL_ARM_shader_framebuffer_fetch",
                          "GL_OES_texture_cube_map_array", "GL_EXT_texture_cube_map_array"};
    for (const char* e : dead) {
        std::string log;
        int v = 0;
        std::string out = vpglsl::preprocess(std::string("#version 310 es\n#ifdef ") + e + "\nint defined_ext;\n#endif\n", Stage::Fragment, log, v);
        CHECK(!has(out, "defined_ext"));
        CHECK(compileFails(Stage::Fragment,
                           std::string("#version 310 es\n#extension ") + e + " : require\nprecision mediump float;\nout vec4 o;\nvoid main() { o = vec4(1.0); }\n",
                           "not supported"));
    }
    // The implemented ones stay.
    const char* live[] = {"GL_OES_standard_derivatives", "GL_EXT_shader_texture_lod", "GL_OES_EGL_image_external",
                          "GL_OES_EGL_image_external_essl3", "GL_EXT_frag_depth", "GL_EXT_draw_buffers", "GL_OVR_multiview",
                          "GL_OVR_multiview2", "GL_EXT_shadow_samplers", "GL_OES_texture_3D", "GL_EXT_texture_buffer"};
    for (const char* e : live) {
        std::string log;
        int v = 0;
        std::string out = vpglsl::preprocess(std::string("#ifdef ") + e + "\nint defined_ext;\n#endif\n", Stage::Fragment, log, v);
        CHECK(has(out, "defined_ext"));
    }
    // I/O block extensions: `#extension` is accepted, the macro stays undefined and the blocks themselves are still rejected.
    for (const char* e : {"GL_EXT_shader_io_blocks", "GL_OES_shader_io_blocks"}) {
        std::string log;
        int v = 0;
        std::string out = vpglsl::preprocess(std::string("#version 310 es\n#ifdef ") + e + "\nint defined_ext;\n#endif\n", Stage::Fragment, log, v);
        CHECK(!has(out, "defined_ext"));
        const std::string head = std::string("#version 310 es\n#extension ") + e + " : require\nprecision mediump float;\n";
        CHECK(vpglsl::parse(Stage::Fragment, head + "out vec4 o;\nvoid main() { o = vec4(1.0); }\n").shader != nullptr);
        CHECK(compileFails(Stage::Fragment, head + "in Blk { vec4 c; } b;\nout vec4 o;\nvoid main() { o = b.c; }\n", "I/O blocks"));
    }
}


void testTextureBuffers() {
    const std::string ext = "#extension GL_EXT_texture_buffer : require\n";
    Program p = build(kTrivialVs,
        "#version 310 es\n" + ext + "precision mediump float;\n"
        "uniform highp samplerBuffer sb;\n"
        "uniform highp isamplerBuffer isb;\n"
        "layout(binding = 3) uniform highp usamplerBuffer usb[2];\n"
        "in vec2 uv; out vec4 o;\n"
        "vec4 get(highp samplerBuffer s, int i) { return texelFetch(s, i); }\n"
        "void main() {\n"
        "  o = get(sb, int(uv.x)) + vec4(texelFetch(isb, 1)) + vec4(texelFetch(usb[1], 2)) + vec4(float(textureSize(sb) + textureSize(usb[0])));\n"
        "}\n");
    CHECK(p.ok);
    if (p.ok) {
        auto glType = [&](const char* n) {
            for (const auto& u : p.link.reflection.uniforms)
                if (u.name == n) return u.glType;
            return 0u;
        };
        CHECK(glType("sb") == 0x8DC2 && glType("isb") == 0x8DD0 && glType("usb[0]") == 0x8DD8);  // GL_*SAMPLER_BUFFER
        for (const auto& u : p.link.reflection.uniforms)
            if (u.name == "usb[0]") CHECK(u.arraySize == 2 && u.unit == 3 && u.binding >= 1 && u.binding < int(vpglsl::kBlockBase));
    }
    // ES 3.20 has them without the extension; ES 3.10 needs it; no sampling.
    CHECK(vpglsl::parse(Stage::Fragment, "#version 320 es\nprecision mediump float;\nuniform samplerBuffer s;\nout vec4 o;\n"
                                          "void main() { o = texelFetch(s, 0); }\n").shader != nullptr);
    CHECK(compileFails(Stage::Fragment, "#version 310 es\nprecision mediump float;\nuniform samplerBuffer s;\nout vec4 o;\nvoid main() { o = texelFetch(s, 0); }\n", "samplerBuffer"));
    CHECK(compileFails(Stage::Fragment, "#version 300 es\n" + ext + "precision mediump float;\nuniform samplerBuffer s;\nout vec4 o;\nvoid main() { o = texelFetch(s, 0); }\n", "samplerBuffer"));
    const std::string fhead = "#version 310 es\n" + ext + "precision mediump float;\nuniform samplerBuffer s;\nout vec4 o;\n";
    CHECK(compileFails(Stage::Fragment, fhead + "void main() { o = texture(s, 0.5); }\n", "no matching"));
    CHECK(compileFails(Stage::Fragment, fhead + "void main() { o = texelFetch(s, 0, 0); }\n", "no matching"));
    CHECK(compileFails(Stage::Fragment, fhead + "void main() { o = texelFetch(s, ivec2(0)); }\n", "no matching"));
    CHECK(compileFails(Stage::Fragment, fhead + "void main() { o = vec4(textureSize(s, 0)); }\n", "no matching"));

    // imageBuffer / iimageBuffer / uimageBuffer in a compute shader.
    ComputeProgram c = buildCompute(
        "#version 310 es\n" + ext +
        "layout(local_size_x = 4) in;\n"
        "layout(rgba8, binding = 0) writeonly uniform highp imageBuffer dst;\n"
        "layout(r32f, binding = 1) readonly uniform highp imageBuffer src;\n"
        "layout(r32i, binding = 2) uniform highp iimageBuffer ii;\n"
        "layout(r32ui, binding = 3) uniform highp uimageBuffer counters;\n"
        "void main() {\n"
        "    int i = int(gl_GlobalInvocationID.x);\n"
        "    imageStore(dst, i, imageLoad(src, i) * float(imageSize(src)));\n"
        "    int a = imageAtomicAdd(ii, i, 1) + imageSize(ii);\n"
        "    imageStore(ii, i, ivec4(a));\n"
        "    uint o = imageAtomicAdd(counters, 0, 1u) + imageLoad(counters, i).x;\n"
        "    imageStore(counters, i, uvec4(o));\n"
        "}\n");
    CHECK(c.ok);
    if (c.ok) {
        auto glType = [&](const char* n) {
            for (const auto& u : c.link.reflection.uniforms)
                if (u.name == n) return u.glType;
            return 0u;
        };
        CHECK(glType("dst") == 0x9051 && glType("ii") == 0x905C && glType("counters") == 0x9067);  // GL_*IMAGE_BUFFER
    }
    // One invocation through the VPIR interpreter: imageLoad / imageStore on a r32ui texel buffer.
    ComputeProgram e = buildCompute(
        "#version 310 es\n" + ext +
        "layout(local_size_x = 1) in;\n"
        "layout(r32ui, binding = 0) uniform highp uimageBuffer b;\n"
        "void main() { uint v = imageLoad(b, 1).x; imageStore(b, 0, uvec4(v + 4u)); }\n");
    CHECK(e.ok);
    if (e.ok) {
        int res = -1;
        for (size_t i = 0; i < e.cs.module.resources.size(); ++i)
            if (e.cs.module.resources[i].kind == vpir::ResourceKind::Buffer) res = int(i);
        CHECK(res >= 0);
        if (res >= 0) {
            vpireval::Invocation inv;
            inv.memory[uint32_t(res)] = std::vector<uint8_t>(16, 0);
            const uint32_t seven = 7;
            std::memcpy(&inv.memory[uint32_t(res)][4], &seven, 4);
            std::string err;
            CHECK(vpireval::run(e.cs.module, inv, err));
            if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
            uint32_t r0 = 0;
            std::memcpy(&r0, inv.memory[uint32_t(res)].data(), 4);
            CHECK(r0 == 11u);
        }
    }
    CHECK(compileFails(Stage::Compute, "#version 310 es\n" + ext + "layout(local_size_x = 1) in;\nlayout(r32f) uniform highp imageBuffer b;\n"
                                       "void main() { imageStore(b, ivec2(0), vec4(1.0)); }\n", "no matching"));
}

// Runs a compute shader's VPIR for one invocation with the reference interpreter and checks what it
// stores into a shader storage block (std430 offsets, atomics, shared memory).
void testExecute() {
    ComputeProgram p = buildCompute(
        "#version 310 es\n"
        "layout(local_size_x = 1) in;\n"
        "layout(std430, binding = 0) buffer B { vec3 a; float b; mat3 m; float f[3]; uint counter; ivec2 iv; } b;\n"
        "shared uint sh;\n"
        "void main() {\n"
        "    b.b = b.a.x + b.a.y + b.a.z;\n"
        "    b.m[1] = b.m[0] * 2.0;\n"
        "    b.f[2] = b.f[0] + b.f[1];\n"
        "    uint old = atomicAdd(b.counter, 5u);\n"
        "    sh = 3u;\n"
        "    uint o2 = atomicAdd(sh, 4u);\n"
        "    b.iv = ivec2(int(old), int(o2 + sh));\n"
        "    atomicMax(b.counter, 12u);\n"
        "    barrier();\n"
        "}\n");
    CHECK(p.ok);
    if (!p.ok) return;
    const auto& b = p.link.reflection.storageBlocks.at(0);
    CHECK(b.size == 96);  // 88 rounded up to the block's 16-byte alignment
    int res = -1;
    for (size_t i = 0; i < p.cs.module.resources.size(); ++i)
        if (p.cs.module.resources[i].kind == vpir::ResourceKind::Buffer) res = int(i);
    CHECK(res >= 0);
    if (res < 0) return;
    std::vector<uint8_t> mem(b.size, 0);
    auto setF = [&](uint32_t off, float v) { std::memcpy(&mem[off], &v, 4); };
    auto setU = [&](uint32_t off, uint32_t v) { std::memcpy(&mem[off], &v, 4); };
    setF(0, 1.0f), setF(4, 2.0f), setF(8, 3.0f);        // a
    setF(16, 1.0f), setF(20, 2.0f), setF(24, 3.0f);     // m[0]
    setF(64, 1.5f), setF(68, 2.5f);                      // f[0], f[1]
    setU(76, 10);                                        // counter
    vpireval::Invocation inv;
    inv.memory[uint32_t(res)] = mem;
    std::string err;
    CHECK(vpireval::run(p.cs.module, inv, err));
    if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
    const auto& out = inv.memory[uint32_t(res)];
    auto getF = [&](uint32_t off) { float v; std::memcpy(&v, &out[off], 4); return v; };
    auto getU = [&](uint32_t off) { uint32_t v; std::memcpy(&v, &out[off], 4); return v; };
    CHECK(getF(12) == 6.0f);                                            // b
    CHECK(getF(32) == 2.0f && getF(36) == 4.0f && getF(40) == 6.0f);    // m[1] = m[0] * 2 (column stride 16)
    CHECK(getF(72) == 4.0f);                                            // f[2]
    CHECK(getU(76) == 15u);                                             // counter: 10 + 5, max(15, 12)
    CHECK(getU(80) == 10u && getU(84) == 10u);                          // iv = (old, o2 + sh) = (10, 3 + 7)

    // ES 3.1 integer functions: carry / borrow / extended multiplies (and frexp / ldexp compile).
    ComputeProgram q = buildCompute(
        "#version 310 es\n"
        "layout(local_size_x = 1) in;\n"
        "layout(std430, binding = 0) buffer B { uint r[8]; } b;\n"
        "void main() {\n"
        "    uint c; uint s = uaddCarry(0xFFFFFFFFu, 2u, c);\n"
        "    uint bw; uint d = usubBorrow(1u, 2u, bw);\n"
        "    uint msb, lsb; umulExtended(0x80000000u, 4u, msb, lsb);\n"
        "    int imsb, ilsb; imulExtended(-2, 3, imsb, ilsb);\n"
        "    b.r[0] = s; b.r[1] = c; b.r[2] = msb; b.r[3] = lsb; b.r[4] = uint(imsb); b.r[5] = uint(ilsb);\n"
        "    b.r[6] = d; b.r[7] = bw;\n"
        "}\n");
    CHECK(q.ok);
    if (q.ok) {
        int res2 = -1;
        for (size_t i = 0; i < q.cs.module.resources.size(); ++i)
            if (q.cs.module.resources[i].kind == vpir::ResourceKind::Buffer) res2 = int(i);
        vpireval::Invocation inv2;
        inv2.memory[uint32_t(res2)] = std::vector<uint8_t>(32, 0);
        std::string err2;
        CHECK(vpireval::run(q.cs.module, inv2, err2));
        if (!err2.empty()) std::fprintf(stderr, "  eval: %s\n", err2.c_str());
        const auto& m2 = inv2.memory[uint32_t(res2)];
        auto u = [&](int i) { uint32_t v; std::memcpy(&v, &m2[size_t(i) * 4], 4); return v; };
        CHECK(u(0) == 1u && u(1) == 1u);                 // 0xFFFFFFFF + 2 = 1, carry 1
        CHECK(u(2) == 2u && u(3) == 0u);                 // 0x80000000 * 4 = 2 : 0
        CHECK(u(4) == 0xFFFFFFFFu && u(5) == 0xFFFFFFFAu);  // -2 * 3 = -6 (msb -1)
        CHECK(u(6) == 0xFFFFFFFFu && u(7) == 1u);        // 1 - 2 = -1, borrow 1
    }
    ComputeProgram f = buildCompute(
        "#version 310 es\n"
        "layout(local_size_x = 1) in;\n"
        "layout(std430, binding = 0) buffer B { vec4 v; ivec4 e; } b;\n"
        "void main() { ivec4 e; b.v = frexp(b.v, e) + ldexp(b.v, b.e); b.e = e; }\n");
    CHECK(f.ok);
    CHECK(compileFails(Stage::Compute, "#version 310 es\nlayout(local_size_x = 1) in;\nvoid main() { uint c; uint s = uaddCarry(1, 2u, c); }\n", "no matching"));
}


// Invariant position: `invariant gl_Position;`, `#pragma STDGL invariant(all)` and an invariant varying (declared or
// listed) reach the final MSL as [[position, invariant]] (the metal2 backend turns preserveInvariance on from that text);
// a shader that declares nothing keeps the plain [[position]].
void testInvariant() {
    const char* fs3 = "#version 300 es\nprecision mediump float; in vec4 v; out vec4 o; void main() { o = v; }";
    const char* fs1 = "#version 100\nprecision mediump float; varying vec4 v; void main() { gl_FragColor = v; }";
    auto vsMsl = [&](const char* vs, const char* fs) {
        Program p = build(vs, fs);
        CHECK(p.ok);
        return p.vs.msl;
    };
    const char* kInv = "[[position, invariant]]";
    CHECK(has(vsMsl("#version 300 es\ninvariant gl_Position; in vec4 a; out vec4 v; void main() { v = a; gl_Position = a; }", fs3), kInv));
    CHECK(has(vsMsl("#version 100\ninvariant gl_Position; attribute vec4 a; varying vec4 v; void main() { v = a; gl_Position = a; }", fs1), kInv));
    CHECK(has(vsMsl("#version 100\n#pragma STDGL invariant(all)\nattribute vec4 a; varying vec4 v; void main() { v = a; gl_Position = a; }", fs1), kInv));
    CHECK(has(vsMsl("#version 300 es\n#pragma STDGL invariant(all)\nin vec4 a; out vec4 v; void main() { v = a; gl_Position = a; }", fs3), kInv));
    CHECK(has(vsMsl("#version 300 es\nin vec4 a; invariant out vec4 v; void main() { v = a; gl_Position = a; }", fs3), kInv));
    CHECK(has(vsMsl("#version 100\nattribute vec4 a; invariant varying vec4 v; void main() { v = a; gl_Position = a; }", fs1), kInv));
    CHECK(has(vsMsl("#version 300 es\nin vec4 a; out vec4 v; invariant v; void main() { v = a; gl_Position = a; }", fs3), kInv));
    // Nothing declared (and the pragma in an inactive #if group): the plain position.
    const std::string plain = vsMsl("#version 300 es\n#if 0\n#pragma STDGL invariant(all)\n#endif\nin vec4 a; out vec4 v; void main() { v = a; gl_Position = a; }", fs3);
    CHECK(!has(plain, "invariant"));
    CHECK(has(plain, "[[position]]"));
}

// Numeric semantics of the GLSL builtins as the Vulkan path lowers them: a compute shader runs in the VPIR interpreter
// (the reference semantics of the IR; the MSL itself is not executed here, only its text is checked), inputs come from
// the buffer so nothing folds. What GLSL fixes: mod = x - y * floor(x / y) (not fmod), step(edge, x) = x < edge ? 0 : 1,
// smoothstep = Hermite of the clamped ratio, integer / and % truncate toward zero (hardware / Metal; GLSL leaves a
// negative operand of % undefined), mix(x, y, bool) selects, floatBitsToInt is a bit copy (NaN payloads kept).
// round() is implementation-defined on halves; the emitter uses rint (round half to even, like roundEven and like the
// hardware instruction mobile GPUs have), which is what engines compiled for Adreno see. min / max / clamp with a NaN
// operand are undefined in GLSL; the IR gives the other operand (fmin / fmax), but the MSL `min` / `max` / `clamp`
// under Metal's fast math make no promise, so NaN-protected shaders (saturate(NaN) = 0) depend on the device.
void testSemantics() {
    ComputeProgram p = buildCompute(
        "#version 310 es\n"
        "layout(local_size_x = 1) in;\n"
        "layout(std430, binding = 0) buffer B { float a[6]; int ia[4]; float r[40]; int ir[16]; } b;\n"
        "void main() {\n"
        "    float x = b.a[0], y = b.a[1], h = b.a[2], t = b.a[3], nan = b.a[4], z = b.a[5];\n"  // -7.5 3 0.5 2.5 NaN 0
        "    b.r[0] = mod(x, y); b.r[1] = mod(-x, -y); b.r[2] = mod(vec2(x, 1.0), y).y;\n"
        "    b.r[3] = round(t); b.r[4] = round(-t); b.r[5] = roundEven(t + 1.0); b.r[6] = round(t + 0.1);\n"
        "    b.r[7] = step(h, h); b.r[8] = step(h, x); b.r[9] = step(h, t); b.r[10] = step(vec2(h, t), vec2(t, t)).x;\n"
        "    b.r[11] = smoothstep(z, 1.0, h); b.r[12] = smoothstep(z, 1.0, t); b.r[13] = smoothstep(z, 1.0, x);\n"
        "    b.r[14] = smoothstep(z, 1.0, 0.25 + z);\n"
        "    b.r[15] = min(nan, t); b.r[16] = max(nan, h); b.r[17] = clamp(nan, z, 1.0); b.r[18] = clamp(x, z, h);\n"
        "    b.r[19] = inversesqrt(t + 1.5); b.r[20] = inversesqrt(z);\n"
        "    b.r[21] = atan(h + 0.5, h + 0.5); b.r[22] = atan(1.0, -1.0 + z); b.r[23] = atan(-1.0, -1.0);\n"
        "    b.r[24] = atan(-1.0, 1.0); b.r[25] = atan(z, -1.0); b.r[26] = atan(1.0, z); b.r[27] = atan(-1.0, z);\n"
        "    b.r[28] = atan(t - 0.5); b.r[29] = atan(-h);\n"
        "    b.r[30] = sign(x); b.r[31] = sign(z); b.r[32] = sign(t);\n"
        "    b.r[33] = mix(h, t, true); b.r[34] = mix(h, t, false); b.r[35] = mix(vec2(x, y), vec2(h, t), bvec2(true, false)).x;\n"
        "    b.r[36] = mix(vec2(x, y), vec2(h, t), bvec2(true, false)).y;\n"
        "    b.r[37] = intBitsToFloat(floatBitsToInt(nan) + 0);\n"
        "    int p = b.ia[0], q = b.ia[1], u = b.ia[2], v = b.ia[3];\n"  // -7 2 7 -3
        "    b.ir[0] = p / q; b.ir[1] = p % q; b.ir[2] = u / v; b.ir[3] = u % v; b.ir[4] = sign(p);\n"
        "    b.ir[5] = floatBitsToInt(1.0); b.ir[6] = floatBitsToInt(-z); b.ir[7] = floatBitsToInt(nan);\n"
        "    b.ir[8] = int(uint(p) / 2u); b.ir[9] = -p % 4;\n"
        "}\n");
    CHECK(p.ok);
    if (!p.ok) return;
    const auto& blk = p.link.reflection.storageBlocks.at(0);
    CHECK(blk.size >= 264);  // 24 + 16 + 160 + 64 bytes
    int res = -1;
    for (size_t i = 0; i < p.cs.module.resources.size(); ++i)
        if (p.cs.module.resources[i].kind == vpir::ResourceKind::Buffer) res = int(i);
    CHECK(res >= 0);
    if (res < 0) return;
    std::vector<uint8_t> mem(blk.size, 0);
    auto setF = [&](uint32_t off, float v) { std::memcpy(&mem[off], &v, 4); };
    auto setI = [&](uint32_t off, int32_t v) { std::memcpy(&mem[off], &v, 4); };
    const float in[6] = {-7.5f, 3.0f, 0.5f, 2.5f, std::nanf(""), 0.0f};
    for (uint32_t k = 0; k < 6; ++k) setF(k * 4, in[k]);
    const int32_t iin[4] = {-7, 2, 7, -3};
    for (uint32_t k = 0; k < 4; ++k) setI(24 + k * 4, iin[k]);
    vpireval::Invocation inv;
    inv.memory[uint32_t(res)] = mem;
    std::string err;
    CHECK(vpireval::run(p.cs.module, inv, err));
    if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
    const auto& out = inv.memory[uint32_t(res)];
    auto R = [&](int k) { float v; std::memcpy(&v, &out[size_t(40 + 4 * k)], 4); return v; };
    auto I = [&](int k) { int32_t v; std::memcpy(&v, &out[size_t(200 + 4 * k)], 4); return v; };
    auto near = [&](int k, float want, float tol = 1e-5f) {
        const bool ok = std::fabs(R(k) - want) <= tol;
        if (!ok) std::fprintf(stderr, "  r[%d] = %.9g, want %.9g\n", k, R(k), want);
        return ok;
    };
    CHECK(near(0, 1.5f) && near(1, -1.5f) && near(2, 1.0f));  // not fmod (-1.5, 1.5): the sign follows y
    CHECK(near(3, 2.0f) && near(4, -2.0f) && near(5, 4.0f) && near(6, 3.0f));  // halves to even, others to nearest
    CHECK(near(7, 1.0f) && near(8, 0.0f) && near(9, 1.0f) && near(10, 1.0f));  // x == edge gives 1
    CHECK(near(11, 0.5f) && near(12, 1.0f) && near(13, 0.0f) && near(14, 0.15625f));
    CHECK(near(15, 2.5f) && near(16, 0.5f) && near(17, 0.0f) && near(18, 0.0f));  // NaN: the other operand
    CHECK(near(19, 0.5f) && std::isinf(R(20)) && R(20) > 0);
    const float pi = 3.14159265358979f;
    CHECK(near(21, pi / 4) && near(22, 3 * pi / 4) && near(23, -3 * pi / 4) && near(24, -pi / 4));
    CHECK(near(25, pi) && near(26, pi / 2) && near(27, -pi / 2));
    CHECK(near(28, std::atan(2.0f)) && near(29, std::atan(-0.5f)));
    CHECK(near(30, -1.0f) && near(31, 0.0f) && near(32, 1.0f));
    CHECK(near(33, 2.5f) && near(34, 0.5f) && near(35, 0.5f) && near(36, 3.0f));
    uint32_t nanBits;
    std::memcpy(&nanBits, &out[size_t(40 + 4 * 37)], 4);
    CHECK((nanBits & 0x7fffffffu) > 0x7f800000u);  // still a NaN after the int round trip
    CHECK(I(0) == -3 && I(1) == -1 && I(2) == -2 && I(3) == 1 && I(4) == -1);  // truncating / and %
    CHECK(I(5) == 0x3f800000 && uint32_t(I(6)) == 0x80000000u && uint32_t(I(7)) == 0x7fc00000u);  // bit copies
    CHECK(I(8) == int32_t(0xFFFFFFF9u / 2u) && I(9) == 3);  // unsigned division of the same bits; -(-7) % 4 = 3

    // The MSL of the same builtins in a fragment shader: the structure that carries the semantics above.
    Program f = build(kTrivialVs,
        "#version 310 es\nprecision highp float; precision highp int; precision highp sampler2D;\n"
        "uniform sampler2D s; uniform int lv; in vec2 uv; out vec4 o;\n"
        "void main() {\n"
        "    vec4 m = texelFetch(s, ivec2(uv), lv) + texelFetch(s, ivec2(uv), 2);\n"
        "    o = m + vec4(mod(uv.x, uv.y), round(uv.x), inversesqrt(uv.x), atan(uv.y, uv.x)) + vec4(step(uv.x, uv.y), smoothstep(0.0, 1.0, uv.y), clamp(uv.x, 0.0, 1.0), sign(uv.y));\n"
        "}\n");
    CHECK(f.ok);
    const std::string& msl = f.fs.msl;
    CHECK(has(msl, "floor(") && !has(msl, "fmod("));  // mod is x - y * floor(x / y)
    CHECK(has(msl, "rint("));                          // round: half to even
    CHECK(has(msl, "rsqrt("));
    // texelFetch with a lod reads that mip level, from the uniform and from the constant: two reads, the second one
    // with the literal 2u as its last argument (matched inside its own statement, whatever the temporaries are called).
    auto readWithLastArg = [&](const std::string& lod) {
        for (size_t at = msl.find(".read("); at != std::string::npos; at = msl.find(".read(", at + 1)) {
            const std::string stmt = msl.substr(at, msl.find(';', at) - at);
            if (stmt.size() > lod.size() + 1 && stmt.compare(stmt.size() - lod.size() - 1, lod.size() + 1, lod + ")") == 0) return true;
        }
        return false;
    };
    CHECK(readWithLastArg("2u"));
    // No assertion on how clamp / smoothstep reach the MSL (clamp or saturate): the interpreter above fixes their value
    // for finite inputs and NaN as the IR defines it, but it does NOT test Metal's NaN semantics under fast math; those
    // are measured on the headset.

    // mix(x, y, a) keeps both ends exact: the textbook x + (y - x) * a loses the small operand against a large one.
    ComputeProgram mp = buildCompute(
        "#version 310 es\n"
        "layout(local_size_x = 1) in;\n"
        "layout(std430, binding = 0) buffer B { float a[4]; float r[8]; } b;\n"
        "void main() {\n"
        "    float x = b.a[0], y = b.a[1], one = b.a[2], zero = b.a[3];\n"  // 1e8 1e-8 1 0
        "    b.r[0] = mix(x, y, one); b.r[1] = mix(x, y, zero);\n"
        "    vec2 v = mix(vec2(x, y), vec2(y, x), one); b.r[2] = v.x; b.r[3] = v.y;\n"
        "    b.r[4] = mix(x, y, 0.5 * one);\n"
        "}\n");
    CHECK(mp.ok);
    if (!mp.ok) return;
    int mres = -1;
    for (size_t i = 0; i < mp.cs.module.resources.size(); ++i)
        if (mp.cs.module.resources[i].kind == vpir::ResourceKind::Buffer) mres = int(i);
    CHECK(mres >= 0);
    if (mres < 0) return;
    std::vector<uint8_t> mmem(48, 0);
    const float mix_in[4] = {1e8f, 1e-8f, 1.0f, 0.0f};
    std::memcpy(mmem.data(), mix_in, sizeof mix_in);
    vpireval::Invocation minv;
    minv.memory[uint32_t(mres)] = mmem;
    std::string merr;
    CHECK(vpireval::run(mp.cs.module, minv, merr));
    const auto& mout = minv.memory[uint32_t(mres)];
    auto M = [&](int k) { float v; std::memcpy(&v, &mout[size_t(16 + 4 * k)], 4); return v; };
    CHECK(M(0) == 1e-8f);  // mix(x, y, 1) == y exactly (x + (y - x) * 1 would give 0)
    CHECK(M(1) == 1e8f);   // mix(x, y, 0) == x exactly
    CHECK(M(2) == 1e-8f && M(3) == 1e8f);  // vector ends, scalar weight splatted
    CHECK(M(4) == 5e7f);   // 0.5: (x + y) / 2 to float precision
    // The MSL spells it with fma, not mix().
    Program mf = build(kTrivialVs,
        "#version 310 es\nprecision highp float; in vec2 uv; out vec4 o;\n"
        "void main() { o = mix(vec4(uv, 0.0, 1.0), vec4(1.0), uv.x); }\n");
    CHECK(mf.ok);
    CHECK(has(mf.fs.msl, "fma(") && !has(mf.fs.msl, "mix("));
}

// Transform feedback: the capture variant of the vertex shader writes the varyings glTransformFeedbackVaryings named (arrays,
// array elements, matrices, integers, gl_Position as the shader wrote it, gl_PointSize, and an output no fragment shader reads)
// into the record of (instance, vertex index); the VPIR interpreter runs it.
void testXfb() {
    vpglsl::LinkOptions lo;
    lo.xfbVaryings = {"v_i", "v_f4[1]", "v_m", "gl_Position", "gl_PointSize", "v_pad", "v_f4[0]"};
    auto v = vpglsl::parse(Stage::Vertex,
        "#version 300 es\n"
        "in vec4 a_pos; in vec4 a_m;\n"
        "out vec4 v_f4[2]; flat out ivec2 v_i; out mat2x3 v_m; out mediump float v_pad; out vec2 v_used;\n"
        "void main() { v_f4[0] = a_pos; v_f4[1] = a_pos * 2.0; v_i = ivec2(a_m.xy); v_m = mat2x3(a_m.xyz, a_m.xyz * 2.0);\n"
        "  v_pad = 7.0; v_used = a_pos.xy; gl_Position = vec4(a_pos.xyz, 4.0); gl_PointSize = 3.0; }\n");
    auto f = vpglsl::parse(Stage::Fragment,
        "#version 300 es\nprecision mediump float;\nin vec2 v_used; out vec4 o;\nvoid main() { o = vec4(v_used, 0.0, 1.0); }\n");
    CHECK(v.shader && f.shader);
    if (!v.shader || !f.shader) return;
    auto l = vpglsl::link(v.shader, f.shader, lo);
    CHECK(l.ok);
    if (!l.ok) {
        std::fprintf(stderr, "  link: %s\n", l.log.c_str());
        return;
    }
    const auto& rx = l.reflection;
    CHECK(rx.xfb.size() == 7 && rx.xfbRecordWords == 2 + 4 + 6 + 4 + 1 + 1 + 4);
    CHECK(rx.xfb[0].name == "v_i" && rx.xfb[0].glType == 0x8B53 && rx.xfb[0].arraySize == 1 && rx.xfb[0].offset == 0);
    CHECK(rx.xfb[1].name == "v_f4[1]" && rx.xfb[1].glType == 0x8B52 && rx.xfb[1].arraySize == 1 && rx.xfb[1].offset == 2);
    CHECK(rx.xfb[2].glType == 0x8B65 /* mat2x3 */ && rx.xfb[2].words == 6 && rx.xfb[2].offset == 6);
    CHECK(rx.xfb[3].offset == 12 && rx.xfb[4].offset == 16 && rx.xfb[5].offset == 17 && rx.xfb[6].offset == 18);
    CHECK(!l.xfbVertexSpirv.empty());
    // The ordinary shaders stay free of the capture.
    CHECK(l.vertexSpirv.size() < l.xfbVertexSpirv.size());
    Lowered plain = lowerStage(l.vertexSpirv, Stage::Vertex, rx);
    CHECK(plain.ok);
    Lowered cap = lowerStage(l.xfbVertexSpirv, Stage::Vertex, rx, true);
    CHECK(cap.ok);
    if (!cap.ok) return;
    int res = -1;
    for (size_t i = 0; i < cap.module.resources.size(); ++i)
        if (cap.module.resources[i].kind == vpir::ResourceKind::Buffer && cap.module.resources[i].index < 29) res = int(i);
    CHECK(res >= 0);
    if (res < 0) return;
    const uint32_t W = rx.xfbRecordWords, span = 8, base = 3, vertex = 5, instance = 2;
    std::vector<uint8_t> mem((vpglsl::kXfbHeaderWords + span * 3 * W) * 4, 0xAB);
    auto putU = [&](size_t word, uint32_t x) { std::memcpy(&mem[word * 4], &x, 4); };
    auto getU = [&](size_t word) { uint32_t x; std::memcpy(&x, &mem[word * 4], 4); return x; };
    auto getF = [&](size_t word) { float x; std::memcpy(&x, &mem[word * 4], 4); return x; };
    putU(0, span), putU(1, base), putU(2, 0), putU(3, 0);
    vpireval::Invocation inv;
    inv.memory[uint32_t(res)] = mem;
    auto vec4 = [](float a, float b, float c, float d) {
        vpireval::Value x;
        x.type = vpir::Type(vpir::Scalar::F32, 4);
        const float fs[4] = {a, b, c, d};
        std::memcpy(x.bits, fs, 16);
        return x;
    };
    auto u32 = [](uint32_t a) {
        vpireval::Value x;
        x.type = vpir::Type(vpir::Scalar::U32);
        x.bits[0] = a;
        return x;
    };
    inv.builtins[vpir::Builtin::VertexId] = u32(vertex);
    inv.builtins[vpir::Builtin::InstanceId] = u32(instance);
    // The attributes come from the vertex buffers the test layout gives them (slot 30 down, stride 64, element at vertex * 64).
    uint32_t slot = 30;
    for (const auto& a : rx.attributes) {
        const float fs[4] = {a.name == "a_pos" ? 1.0f : 10.0f, a.name == "a_pos" ? 2.0f : 20.0f, a.name == "a_pos" ? 3.0f : 30.0f,
                             a.name == "a_pos" ? 9.0f : 0.0f};
        for (size_t i = 0; i < cap.module.resources.size(); ++i)
            if (cap.module.resources[i].kind == vpir::ResourceKind::Buffer && cap.module.resources[i].index == slot) {
                std::vector<uint8_t> vb((vertex + 1) * 64, 0);
                std::memcpy(&vb[vertex * 64], fs, 16);
                inv.memory[uint32_t(i)] = vb;
            }
        --slot;
    }
    std::string err;
    CHECK(vpireval::run(cap.module, inv, err));
    if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
    mem = inv.memory[uint32_t(res)];
    const size_t rec = vpglsl::kXfbHeaderWords + size_t(instance * span + (vertex - base)) * W;
    CHECK(getU(0) == span && getU(1) == base);  // the header is not touched
    CHECK(int32_t(getU(rec + 0)) == 10 && int32_t(getU(rec + 1)) == 20);                              // v_i
    CHECK(getF(rec + 2) == 2.0f && getF(rec + 3) == 4.0f && getF(rec + 4) == 6.0f && getF(rec + 5) == 18.0f);  // v_f4[1]
    CHECK(getF(rec + 6) == 10.0f && getF(rec + 7) == 20.0f && getF(rec + 8) == 30.0f);                // v_m column 0
    CHECK(getF(rec + 9) == 20.0f && getF(rec + 10) == 40.0f && getF(rec + 11) == 60.0f);              // column 1
    CHECK(getF(rec + 12) == 1.0f && getF(rec + 13) == 2.0f && getF(rec + 14) == 3.0f && getF(rec + 15) == 4.0f);  // gl_Position: before the clip-z remap
    CHECK(getF(rec + 16) == 3.0f);                                                                    // gl_PointSize
    CHECK(getF(rec + 17) == 7.0f);                                                                    // v_pad (not read by the fragment shader)
    CHECK(getF(rec + 18) == 1.0f && getF(rec + 22 - 1) == 9.0f);                                      // v_f4[0]
    CHECK(getU(rec + W) == 0xABABABABu && getU(rec - 1) == 0xABABABABu);                              // the neighbours are untouched

    // Link failures: unknown, repeated, out-of-range, element of a non-array, over the limits of the mode.
    auto linkWith = [&](std::vector<std::string> names, bool separate) {
        vpglsl::LinkOptions o;
        o.xfbVaryings = std::move(names);
        o.xfbSeparate = separate;
        return vpglsl::link(v.shader, f.shader, o).ok;
    };
    CHECK(linkWith({"v_i", "v_used"}, false));
    CHECK(!linkWith({"v_nope"}, false));
    CHECK(!linkWith({"v_i", "v_i"}, false));
    CHECK(!linkWith({"v_f4", "v_f4[1]"}, false));
    CHECK(!linkWith({"v_f4[2]"}, false));
    CHECK(!linkWith({"v_i[0]"}, false));
    CHECK(!linkWith({"v_f4[x]"}, false));
    CHECK(linkWith({"v_m"}, false) && !linkWith({"v_m"}, true));  // 6 components > 4 for one separate attribute
    CHECK(linkWith({"v_f4[0]", "v_f4[1]", "v_i", "v_used"}, true));
    CHECK(!linkWith({"v_f4[0]", "v_f4[1]", "v_i", "v_used", "v_pad"}, true));  // five separate attributes
    CHECK(!linkWith({"v_f4", "v_f4"}, false));
}

// UE4-style packed uniform arrays (pu_h / pu_m / pu_l vec4 arrays) in the default block, read with dynamic and constant
// indices; the bytes are placed exactly where glUniform*fv would put them (reflection offset + element * arrayStride).
void testPackedUniforms() {
    ComputeProgram p = buildCompute(
        "#version 310 es\n"
        "layout(local_size_x = 1) in;\n"
        "layout(std430, binding = 0) buffer B { vec4 o[8]; int idx; } b;\n"
        "uniform highp vec4 pu_h[8];\n"
        "uniform mediump vec4 pu_m[4];\n"
        "uniform lowp vec4 pu_l[2];\n"
        "uniform highp float sf[3];\n"
        "uniform highp vec3 sv[3];\n"
        "uniform highp mat4 sm[2];\n"
        "void main() {\n"
        "    int i = b.idx;\n"
        "    b.o[0] = pu_h[i];\n"
        "    b.o[1] = pu_h[7] + pu_m[i] * 2.0;\n"
        "    b.o[2] = pu_l[1] + vec4(sf[i], sf[2], sv[i].z, sv[1].x);\n"
        "    b.o[3] = sm[1] * vec4(1.0, 0.0, 0.0, 0.0) + sm[0][i];\n"
        "}\n");
    CHECK(p.ok);
    if (!p.ok) return;
    auto find = [&](const char* n) -> const vpglsl::Variable* {
        for (const auto& u : p.link.reflection.uniforms)
            if (u.name == n) return &u;
        return nullptr;
    };
    int ub = -1, sb = -1;
    for (size_t i = 0; i < p.cs.module.resources.size(); ++i) {
        if (p.cs.module.resources[i].kind == vpir::ResourceKind::ConstantBuffer) ub = int(i);
        if (p.cs.module.resources[i].kind == vpir::ResourceKind::Buffer) sb = int(i);
    }
    CHECK(ub >= 0 && sb >= 0);
    if (ub < 0 || sb < 0) return;
    std::vector<uint8_t> block(std::max<uint32_t>(16, p.link.reflection.defaultBlockSize), 0);
    auto put = [&](const char* n, int element, int col, int row, float v) {
        const auto* u = find(n);
        CHECK(u != nullptr);
        if (!u) return;
        std::memcpy(&block[u->offset + uint32_t(element) * u->arrayStride + uint32_t(col) * 16 + uint32_t(row) * 4], &v, 4);
    };
    for (int e = 0; e < 8; ++e)
        for (int c = 0; c < 4; ++c) put("pu_h[0]", e, 0, c, float(10 * e + c));
    for (int e = 0; e < 4; ++e)
        for (int c = 0; c < 4; ++c) put("pu_m[0]", e, 0, c, float(100 * e + c));
    for (int c = 0; c < 4; ++c) put("pu_l[0]", 1, 0, c, float(1000 + c));
    for (int e = 0; e < 3; ++e) put("sf[0]", e, 0, 0, float(2000 + e));
    for (int e = 0; e < 3; ++e)
        for (int c = 0; c < 3; ++c) put("sv[0]", e, 0, c, float(3000 + 10 * e + c));
    for (int e = 0; e < 2; ++e)
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r) put("sm[0]", e, c, r, float(4000 + 100 * e + 10 * c + r));
    std::vector<uint8_t> out(8 * 16 + 16, 0);
    const int32_t index = 2;
    std::memcpy(&out[128], &index, 4);
    vpireval::Invocation inv;
    inv.memory[uint32_t(ub)] = block;
    inv.memory[uint32_t(sb)] = out;
    std::string err;
    CHECK(vpireval::run(p.cs.module, inv, err));
    if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
    const auto& m = inv.memory[uint32_t(sb)];
    auto f = [&](int vec, int c) { float v; std::memcpy(&v, &m[size_t(vec) * 16 + size_t(c) * 4], 4); return v; };
    for (int c = 0; c < 4; ++c) CHECK(f(0, c) == float(20 + c));                          // pu_h[2]
    for (int c = 0; c < 4; ++c) CHECK(f(1, c) == float(70 + c) + 2.0f * float(200 + c));  // pu_h[7] + 2 * pu_m[2]
    CHECK(f(2, 0) == 1000.0f + 2002.0f && f(2, 1) == 1001.0f + 2002.0f && f(2, 2) == 1002.0f + 3022.0f && f(2, 3) == 1003.0f + 3010.0f);
    for (int c = 0; c < 4; ++c) CHECK(f(3, c) == float(4100 + c) + float(4000 + 20 + c));  // sm[1][0] + sm[0][2]

    // The same arrays in real uniform blocks (std140), the layout UE4 uses when it has UBOs: vec4 arrays, a vec3 followed by a
    // float (packs into the vec3's fourth word), float / vec3 arrays (stride 16), a mat3 and an int.
    ComputeProgram q = buildCompute(
        "#version 310 es\n"
        "layout(local_size_x = 1) in;\n"
        "layout(std430, binding = 0) buffer B { vec4 o[4]; int idx; } b;\n"
        "layout(std140) uniform Prim { highp vec4 pb0[6]; highp vec3 dir; highp float gain; highp float fa[3]; highp mat3 m3; highp int n; } u;\n"
        "void main() {\n"
        "    int i = b.idx;\n"
        "    b.o[0] = u.pb0[i];\n"
        "    b.o[1] = vec4(u.dir, u.gain);\n"
        "    b.o[2] = vec4(u.fa[i], u.fa[0], float(u.n), 0.0);\n"
        "    b.o[3] = vec4(u.m3[i], 0.0) + vec4(u.m3 * vec3(0.0, 1.0, 0.0), 0.0);\n"
        "}\n");
    CHECK(q.ok);
    if (!q.ok) return;
    const auto& blk = q.link.reflection.blocks.at(0);
    auto bm = [&](const char* n) { return member(blk, std::string("Prim.") + n); };
    CHECK(bm("pb0[0]") && bm("dir") && bm("gain") && bm("fa[0]") && bm("m3") && bm("n"));
    if (!bm("pb0[0]") || !bm("dir") || !bm("gain") || !bm("fa[0]") || !bm("m3") || !bm("n")) return;
    CHECK(bm("pb0[0]")->offset == 0 && bm("pb0[0]")->arrayStride == 16 && bm("dir")->offset == 96 && bm("gain")->offset == 108);
    CHECK(bm("fa[0]")->offset == 112 && bm("fa[0]")->arrayStride == 16 && bm("m3")->offset == 160 && bm("n")->offset == 208);
    int qu = -1, qs = -1;
    for (size_t i = 0; i < q.cs.module.resources.size(); ++i) {
        if (q.cs.module.resources[i].kind == vpir::ResourceKind::ConstantBuffer) qu = int(i);
        if (q.cs.module.resources[i].kind == vpir::ResourceKind::Buffer) qs = int(i);
    }
    CHECK(qu >= 0 && qs >= 0);
    if (qu < 0 || qs < 0) return;
    std::vector<uint8_t> ubo(blk.size, 0);
    auto setF = [&](uint32_t off, float v) { std::memcpy(&ubo[off], &v, 4); };
    for (uint32_t e = 0; e < 6; ++e)
        for (uint32_t c = 0; c < 4; ++c) setF(e * 16 + c * 4, float(10 * e + c));
    setF(96, 1.5f), setF(100, 2.5f), setF(104, 3.5f), setF(108, 4.5f);
    for (uint32_t e = 0; e < 3; ++e) setF(112 + e * 16, float(500 + e));
    for (uint32_t c = 0; c < 3; ++c)
        for (uint32_t r = 0; r < 3; ++r) setF(160 + c * 16 + r * 4, float(700 + 10 * c + r));
    const int32_t nn = 9;
    std::memcpy(&ubo[208], &nn, 4);
    std::vector<uint8_t> qo(4 * 16 + 16, 0);
    const int32_t qi = 2;
    std::memcpy(&qo[64], &qi, 4);
    vpireval::Invocation qinv;
    qinv.memory[uint32_t(qu)] = ubo;
    qinv.memory[uint32_t(qs)] = qo;
    std::string qerr;
    CHECK(vpireval::run(q.cs.module, qinv, qerr));
    if (!qerr.empty()) std::fprintf(stderr, "  eval: %s\n", qerr.c_str());
    const auto& qm = qinv.memory[uint32_t(qs)];
    auto qf = [&](int vec, int c) { float v; std::memcpy(&v, &qm[size_t(vec) * 16 + size_t(c) * 4], 4); return v; };
    for (int c = 0; c < 4; ++c) CHECK(qf(0, c) == float(20 + c));
    CHECK(qf(1, 0) == 1.5f && qf(1, 1) == 2.5f && qf(1, 2) == 3.5f && qf(1, 3) == 4.5f);
    CHECK(qf(2, 0) == 502.0f && qf(2, 1) == 500.0f && qf(2, 2) == 9.0f);
    for (int r = 0; r < 3; ++r) CHECK(qf(3, r) == float(720 + r) + float(710 + r));  // m3[2] + m3 * (0, 1, 0)
}

// Spherical-harmonic sky light of UE4 mobile (the lighting dynamic / skinned meshes get from uniforms instead of a lightmap):
// repeated-component swizzles, per-channel writes, dots against packed vec4 uniform elements, max against a scalar.
void testShLighting() {
    ComputeProgram p = buildCompute(
        "#version 310 es\n"
        "layout(local_size_x = 1) in;\n"
        "layout(std430, binding = 0) buffer B { vec4 n; vec4 o; } b;\n"
        "uniform highp vec4 pu_h[7];\n"
        "mediump vec3 SkySH(mediump vec3 n) {\n"
        "    mediump vec4 vN = vec4(n, 1.0);\n"
        "    mediump vec3 i0;\n"
        "    i0.r = dot(pu_h[0], vN); i0.g = dot(pu_h[1], vN); i0.b = dot(pu_h[2], vN);\n"
        "    mediump vec4 vB = n.xyzz * n.yzzx;\n"
        "    mediump vec3 i1;\n"
        "    i1.r = dot(pu_h[3], vB); i1.g = dot(pu_h[4], vB); i1.b = dot(pu_h[5], vB);\n"
        "    mediump float vC = n.x * n.x - n.y * n.y;\n"
        "    mediump vec3 i2 = pu_h[6].rgb * vC;\n"
        "    return max(vec3(0.0), i0 + i1 + i2);\n"
        "}\n"
        "void main() { b.o = vec4(SkySH(b.n.xyz), 1.0); }\n");
    CHECK(p.ok);
    if (!p.ok) return;
    const auto* u = [&]() -> const vpglsl::Variable* {
        for (const auto& v : p.link.reflection.uniforms)
            if (v.name == "pu_h[0]") return &v;
        return nullptr;
    }();
    int ub = -1, sb = -1;
    for (size_t i = 0; i < p.cs.module.resources.size(); ++i) {
        if (p.cs.module.resources[i].kind == vpir::ResourceKind::ConstantBuffer) ub = int(i);
        if (p.cs.module.resources[i].kind == vpir::ResourceKind::Buffer) sb = int(i);
    }
    CHECK(u && ub >= 0 && sb >= 0);
    if (!u || ub < 0 || sb < 0) return;
    float sh[7][4];
    for (int e = 0; e < 7; ++e)
        for (int c = 0; c < 4; ++c) sh[e][c] = 0.05f * float(e + 1) + 0.01f * float(c) - (e == 4 ? 0.2f : 0.0f);
    std::vector<uint8_t> block(std::max<uint32_t>(16, p.link.reflection.defaultBlockSize), 0);
    for (int e = 0; e < 7; ++e) std::memcpy(&block[u->offset + uint32_t(e) * u->arrayStride], sh[e], 16);
    float n[3] = {0.3f, -0.5f, 0.8f};
    const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    for (float& x : n) x /= len;
    std::vector<uint8_t> out(32, 0);
    std::memcpy(out.data(), n, 12);
    vpireval::Invocation inv;
    inv.memory[uint32_t(ub)] = block;
    inv.memory[uint32_t(sb)] = out;
    std::string err;
    CHECK(vpireval::run(p.cs.module, inv, err));
    if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
    const auto& m = inv.memory[uint32_t(sb)];
    auto dot4 = [](const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; };
    const float vN[4] = {n[0], n[1], n[2], 1.0f};
    const float vB[4] = {n[0] * n[1], n[1] * n[2], n[2] * n[2], n[2] * n[0]};
    const float vC = n[0] * n[0] - n[1] * n[1];
    for (int c = 0; c < 3; ++c) {
        const float want = std::max(0.0f, dot4(sh[c], vN) + dot4(sh[3 + c], vB) + sh[6][c] * vC);
        float got;
        std::memcpy(&got, &m[16 + 4 * size_t(c)], 4);
        if (std::fabs(got - want) > 1e-5f) std::fprintf(stderr, "  sh channel %d: got %g want %g\n", c, double(got), double(want));
        CHECK(std::fabs(got - want) <= 1e-5f);
    }
}

}  // namespace

// G18: a hostile shader (deep nesting, macros that double at every level) fails to compile; it does not overflow the stack or run
// away. Ordinary nesting still compiles.
void testLimits() {
    auto wrap = [](const std::string& expr) { return "void main() { float x = " + expr + "; gl_Position = vec4(x); }"; };
    const std::string deepParens = std::string(20000, '(') + "1.0" + std::string(20000, ')');
    CHECK(compileFails(Stage::Vertex, wrap(deepParens), "too deeply"));
    std::string unary;
    for (int i = 0; i < 20000; ++i) unary += "- ";
    CHECK(vpglsl::parse(Stage::Vertex, wrap(unary + "1.0")).shader == nullptr);
    std::string braces = "void main() ";
    for (int i = 0; i < 20000; ++i) braces += "{ ";
    for (int i = 0; i < 20000; ++i) braces += "} ";
    CHECK(compileFails(Stage::Vertex, braces, "too deeply"));
    // Real nesting depths are fine.
    std::string ok = std::string(60, '(') + "1.0" + std::string(60, ')');
    CHECK(vpglsl::parse(Stage::Vertex, wrap(ok)).shader != nullptr);
    std::string okStmt = "void main() { float x = 0.0; ";
    for (int i = 0; i < 40; ++i) okStmt += "if (x < 1.0) { ";
    okStmt += "x = 1.0; ";
    for (int i = 0; i < 40; ++i) okStmt += "} ";
    okStmt += "gl_Position = vec4(x); }";
    CHECK(vpglsl::parse(Stage::Vertex, okStmt).shader != nullptr);
    // #if with deep parentheses.
    CHECK(vpglsl::parse(Stage::Vertex, "#if " + std::string(20000, '(') + "1" + std::string(20000, ')') + "\nvoid main() {}\n#endif\n").shader == nullptr);
    // A macro that doubles at each of 40 levels would expand to 2^40 tokens.
    std::string macros = "#define M0 x\n";
    for (int i = 1; i <= 40; ++i) macros += "#define M" + std::to_string(i) + " M" + std::to_string(i - 1) + " M" + std::to_string(i - 1) + "\n";
    CHECK(compileFails(Stage::Vertex, macros + "void main() { float x = 1.0; M40 }\n", "too large"));
}

// Deep but legitimate nesting (machine generated shaders: else-if ladders, chained ?:, nested calls) compiles on the small stack of
// a worker thread; hostile nesting is an error, never a crash.
namespace {
size_t smallStackBytes() {
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    return 8u << 20;  // (the sanitizer's frames are several times larger)
#else
    return 512u << 10;
#endif
}
void onSmallStack(const std::function<void()>& fn, size_t bytes = 0) {
    pthread_attr_t a;
    pthread_attr_init(&a);
    pthread_attr_setstacksize(&a, bytes ? bytes : smallStackBytes());
    pthread_t t;
    CHECK(pthread_create(&t, &a, [](void* p) -> void* { (*static_cast<const std::function<void()>*>(p))(); return nullptr; }, const_cast<std::function<void()>*>(&fn)) == 0);
    pthread_join(t, nullptr);
    pthread_attr_destroy(&a);
}
const char* const kFragHead = "#version 300 es\nprecision highp float;\nflat in int id; out vec4 o;\n";
std::string elseIfChain(int n) {
    std::string s = std::string(kFragHead) + "void main() { float c = 0.0;\nif (id == 0) c = 0.0;\n";
    for (int i = 1; i < n; ++i) s += "else if (id == " + std::to_string(i) + ") c = " + std::to_string(i) + ".0;\n";
    return s + "else c = -1.0;\no = vec4(c); }\n";
}
std::string ternaryChain(int n) {
    std::string s = std::string(kFragHead) + "void main() { float c = ";
    for (int i = 0; i < n; ++i) s += "id == " + std::to_string(i) + " ? " + std::to_string(i) + ".0 : ";
    return s + "-1.0; o = vec4(c); }\n";
}
std::string nestedCalls(int n) {
    std::string s = std::string(kFragHead) + "float f(float x) { return x + 1.0; }\nvoid main() { float c = ";
    for (int i = 0; i < n; ++i) s += "f(";
    s += "0.0";
    for (int i = 0; i < n; ++i) s += ")";
    return s + "; o = vec4(c); }\n";
}
std::string nestedParens(int n) {
    return std::string(kFragHead) + "void main() { float c = " + std::string(size_t(n), '(') + "1.0" + std::string(size_t(n), ')') + "; o = vec4(c); }\n";
}
std::string nestedIfs(int n) {
    std::string s = std::string(kFragHead) + "void main() { float c = 0.0;\n";
    for (int i = 0; i < n; ++i) s += "if (id > " + std::to_string(i) + ") {\n";
    s += "c = 1.0;\n";
    for (int i = 0; i < n; ++i) s += "}\n";
    return s + "o = vec4(c); }\n";
}
std::string longSum(int n) {
    std::string s = std::string(kFragHead) + "void main() { float a = float(id); float c = a";
    for (int i = 1; i < n; ++i) s += " + a";
    return s + "; o = vec4(c); }\n";
}
}  // namespace

void testNesting() {
    const char* vs = "#version 300 es\nin vec4 p; void main() { gl_Position = p; }\n";
    // Threads with far less stack than the 512 KB of a worker: an operator chain near the limit (the tree is 1199 levels deep, the
    // parser is not) is parsed, linked, lowered, walked and destroyed there without a crash; it ends in a result or in an error.
    for (size_t kb : {44, 47, 64, 128, 256}) {
        onSmallStack([&] {
            for (int sum : {1199, 1000}) {
                auto r = vpglsl::parse(Stage::Fragment, longSum(sum));
                CHECK(r.shader != nullptr || r.log.find("too deeply") != std::string::npos);
                (void)build(vs, longSum(sum));  // (a result or an error; the point is that it returns)
            }
        }, (kb << 10) * (smallStackBytes() > (1u << 20) ? 16 : 1));
    }
    onSmallStack([&] {
        struct Case { const char* name; std::string src; } cases[] = {
            {"else-if 210", elseIfChain(210)}, {"else-if 400", elseIfChain(400)}, {"?: 250", ternaryChain(250)},
            {"calls 150", nestedCalls(150)}, {"parens 105", nestedParens(105)}, {"ifs 100", nestedIfs(100)}, {"ifs 300", nestedIfs(300)},
        };
        for (auto& c : cases) {
            auto r = vpglsl::parse(Stage::Fragment, c.src);
            if (!r.shader) std::fprintf(stderr, "  %s: %s", c.name, r.log.c_str());
            CHECK(r.shader != nullptr);
        }
        // Past any sane depth: a clean error (the same message as before), on this small stack too.
        CHECK(compileFails(Stage::Fragment, nestedParens(20000), "too deeply"));
        CHECK(compileFails(Stage::Fragment, nestedCalls(20000), "too deeply"));
        CHECK(compileFails(Stage::Fragment, nestedIfs(20000), "too deeply"));
        CHECK(compileFails(Stage::Fragment, ternaryChain(20000), "too deeply"));
        // (An else-if ladder is parsed in a loop; the SPIR-V it becomes nests once per rung, which the lowering bounds.)
        CHECK(vpglsl::parse(Stage::Fragment, elseIfChain(20000)).shader != nullptr);
        CHECK(!build(vs, elseIfChain(20000)).ok);
        // a + a + ... is a left-deep tree: every pass over it recurses once per term.
        CHECK(vpglsl::parse(Stage::Fragment, longSum(500)).shader != nullptr);
        CHECK(build(vs, longSum(500)).ok);
        CHECK(compileFails(Stage::Fragment, longSum(4000), "too deeply"));
        CHECK(compileFails(Stage::Fragment, longSum(200000), "too deeply"));
        // A sum near the limit ends in a result or in an error, not in a crash: the emitter watches the stack too.
        (void)build(vs, longSum(1100));
        // The whole way (parse, link, SPIR-V, lowering, MSL) for the ladders that generators emit.
        for (auto& c : {std::make_pair("else-if 330", elseIfChain(330)), std::make_pair("else-if 400", elseIfChain(400)),
                        std::make_pair("?: 250", ternaryChain(250)), std::make_pair("ifs 100", nestedIfs(100)),
                        std::make_pair("ifs 300", nestedIfs(300)), std::make_pair("calls 150", nestedCalls(150)),
                        std::make_pair("parens 105", nestedParens(105))}) {
            Program p = build(vs, c.second);
            if (!p.ok) std::fprintf(stderr, "  %s: build failed\n", c.first);
            CHECK(p.ok);
        }
        // The ladder stays flat in the MSL (clang stops at 256 nested braces): one do-block, not `else { if` 400 deep.
        auto braces = [](const std::string& msl) {
            int depth = 0, deepest = 0;
            for (char ch : msl) {
                if (ch == '{') deepest = std::max(deepest, ++depth);
                else if (ch == '}') --depth;
            }
            return deepest;
        };
        // (A ?: chain is lowered to branches that each copy their result after the inner ones: not a pure ladder, so the MSL nests one
        // level per link. clang refuses ~250 links; this is the limit that remains, see the note in msl.cpp ifChain.)
        Program tern = build(vs, ternaryChain(100));
        CHECK(tern.ok && braces(tern.fs.msl) < 256);
        Program p = build(vs, elseIfChain(400));
        const int deepest = braces(p.fs.msl);
        if (!p.ok || deepest >= 32) std::fprintf(stderr, "  else-if 400: ok=%d, deepest brace %d\n", int(p.ok), deepest);
        CHECK(p.ok && deepest < 32);
        CHECK(p.fs.msl.find("} while (false);") != std::string::npos);  // (the code of each rung stays where it was: one do-block)
    });
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    const std::map<std::string, std::function<void()>> tests = {
        {"pp", testPp}, {"ppfuzz", testPpFuzz}, {"parse", testParse}, {"errors", testErrors}, {"es100", testEs100}, {"es300", testEs300},
        {"multiview", testMultiview}, {"blocks", testBlocks}, {"control", testControl}, {"builtins", testBuiltins},
        {"textures", testTextures}, {"external", testExternalSampler}, {"link", testLink},
        {"compute", testCompute}, {"ssbo", testStorageBlocks}, {"images", testImages}, {"atomics", testAtomics},
        {"gather", testGather}, {"fbfetch", testFramebufferFetch}, {"exts", testExtensions}, {"texbuffer", testTextureBuffers}, {"exec", testExecute}, {"packed", testPackedUniforms}, {"shlight", testShLighting},
        {"lenient", testLenient}, {"xfb", testXfb}, {"invariant", testInvariant}, {"semantics", testSemantics}, {"limits", testLimits}, {"nesting", testNesting},
    };
    auto it = tests.find(argv[1]);
    if (it == tests.end()) return 2;
    {  // the conformance rules are on only where a test checks them (the rest of the tests build shaders with relaxed headers)
        const std::string g = argv[1];
        vpglsl::setStrictMode(g == "errors" || g == "pp" || g == "parse" || g == "fbfetch" || g == "exts" ? 1 : 0);
    }
    it->second();
    std::printf("%s: %s\n", argv[1], failures ? "FAIL" : "ok");
    return failures ? 1 : 0;
}
