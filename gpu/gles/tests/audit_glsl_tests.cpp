// Audit tests of the GLSL ES compiler against the language specification: each case is a compute shader run by the
// VPIR interpreter, the expected values come from the GLSL ES 3.10 text (not from what the compiler does).
// Usage: vpglsl_audit_tests <group>   (sem | link | pp | iface | fuzz)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
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

struct Compiled {
    bool ok = false;
    vpglsl::LinkResult link;
    vpir::Module module;
    int bufferResource = -1;
};

Compiled compileCompute(const std::string& src, bool verbose = true) {
    Compiled c;
    auto p = vpglsl::parse(Stage::Compute, src);
    if (!p.shader) {
        if (verbose) std::fprintf(stderr, "  compile: %s\n", p.log.c_str());
        return c;
    }
    c.link = vpglsl::link(p.shader, nullptr, {});
    if (!c.link.ok) {
        if (verbose) std::fprintf(stderr, "  link: %s\n", c.link.log.c_str());
        return c;
    }
    std::string error;
    spirv::SpirvModule* m = spirv::parse(c.link.computeSpirv.data(), c.link.computeSpirv.size(), error);
    if (!m) {
        std::fprintf(stderr, "  spirv parse: %s\n", error.c_str());
        return c;
    }
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = vpir::Stage::Compute;
    uint16_t cb = 0, buf = 0;
    auto add = [&](uint32_t binding, vpgpu::BindKind k, uint32_t count) {
        spirv::SpirvBinding b;
        b.set = 0, b.binding = binding, b.kind = k, b.arraySize = count;
        if (k == vpgpu::BindKind::ConstantBuffer) b.slot = cb, cb = uint16_t(cb + count);
        else b.slot = buf, buf = uint16_t(buf + count);
        o.layout.push_back(b);
    };
    add(vpglsl::kDefaultBlockBinding, vpgpu::BindKind::ConstantBuffer, 1);
    std::map<uint32_t, uint32_t> storageCount;
    for (const auto& b : c.link.reflection.storageBlocks) storageCount[b.binding] = std::max(storageCount[b.binding], b.element + 1);
    for (auto [binding, n] : storageCount) add(binding, vpgpu::BindKind::Buffer, n);
    spirv::SpirvInterface iface;
    const bool lowered = spirv::lower(*m, o, c.module, iface, error);
    spirv::release(m);
    if (!lowered) {
        std::fprintf(stderr, "  lower: %s\n", error.c_str());
        return c;
    }
    auto errs = vpir::verify(c.module);
    if (!errs.empty()) {
        for (auto& e : errs) std::fprintf(stderr, "  verify: %s\n", e.c_str());
        return c;
    }
    for (size_t i = 0; i < c.module.resources.size(); ++i)
        if (c.module.resources[i].kind == vpir::ResourceKind::Buffer) c.bufferResource = int(i);
    c.ok = c.bufferResource >= 0;
    return c;
}

// One case: `body` runs after the inputs are loaded into i0..i3, z (0), o (1) and f0..f2 (0.5, 2.0, -1.5), and
// writes b.r[k] / b.f[k]. The expected ints / floats are the first entries of each array.
struct Case {
    const char* name;
    const char* prelude;
    const char* body;
    std::vector<int32_t> ints;
    std::vector<float> floats;
    bool run = true;  // false: only compiled and lowered (the interpreter lacks some geometric built-ins)
};

constexpr uint32_t kFloatOffset = 64 * 4;

bool runCase(const Case& cs) {
    std::string src =
        "#version 310 es\nprecision highp float; precision highp int;\n"
        "layout(local_size_x = 1) in;\n"
        "layout(std430, binding = 0) buffer B { int r[64]; float f[64]; } b;\n";
    src += cs.prelude;
    src += "\nvoid main() {\n int i0 = b.r[48], i1 = b.r[49], i2 = b.r[50], i3 = b.r[51], z = b.r[52], o = b.r[53];\n"
           " float f0 = b.f[48], f1 = b.f[49], f2 = b.f[50];\n";
    src += cs.body;
    src += "\n}\n";
    Compiled c = compileCompute(src);
    if (!c.ok) {
        std::fprintf(stderr, "  [%s] does not compile/link\n", cs.name);
        ++failures;
        return false;
    }
    if (!cs.run) return true;
    std::vector<uint8_t> mem(kFloatOffset + 64 * 4, 0);
    auto setI = [&](int k, int32_t v) { std::memcpy(&mem[size_t(k) * 4], &v, 4); };
    auto setF = [&](int k, float v) { std::memcpy(&mem[kFloatOffset + size_t(k) * 4], &v, 4); };
    setI(48, 3), setI(49, 5), setI(50, -7), setI(51, 2), setI(52, 0), setI(53, 1);
    setF(48, 0.5f), setF(49, 2.0f), setF(50, -1.5f);
    vpireval::Invocation inv;
    inv.memory[uint32_t(c.bufferResource)] = mem;
    std::string err;
    if (!vpireval::run(c.module, inv, err)) {
        std::fprintf(stderr, "  [%s] eval: %s\n", cs.name, err.c_str());
        ++failures;
        return false;
    }
    const auto& out = inv.memory[uint32_t(c.bufferResource)];
    bool ok = true;
    for (size_t k = 0; k < cs.ints.size(); ++k) {
        int32_t v;
        std::memcpy(&v, &out[k * 4], 4);
        if (v != cs.ints[k]) {
            std::fprintf(stderr, "  [%s] r[%zu] = %d, want %d\n", cs.name, k, v, cs.ints[k]);
            ok = false;
        }
    }
    for (size_t k = 0; k < cs.floats.size(); ++k) {
        float v;
        std::memcpy(&v, &out[kFloatOffset + k * 4], 4);
        const float w = cs.floats[k];
        if (!(std::fabs(v - w) <= 1e-5f * std::max(1.0f, std::fabs(w))) && !(std::isnan(v) && std::isnan(w))) {
            std::fprintf(stderr, "  [%s] f[%zu] = %.9g, want %.9g\n", cs.name, k, v, w);
            ok = false;
        }
    }
    if (!ok) ++failures;
    return ok;
}

// ---------------------------------------------------------------------------------------------
void testSemantics() {
    const std::vector<Case> cases = {
        // Side effects of an l-value's index run once (GLSL ES 3.10 5.8: the l-value is evaluated once for op= and ++/--).
        {"compound_index_side_effect", "",
         "int a[4] = int[4](10, 20, 30, 40); int i = 0; a[i++] += 5; b.r[0] = i; b.r[1] = a[0]; b.r[2] = a[1];",
         {1, 15, 20}, {}},
        {"incdec_index_side_effect", "",
         "int a[4] = int[4](10, 20, 30, 40); int i = 0; int old = a[i++]++; b.r[0] = i; b.r[1] = a[0]; b.r[2] = a[1]; b.r[3] = old;",
         {1, 11, 20, 10}, {}},
        {"inout_arg_index_side_effect", "void inc(inout int x) { x += 1; }",
         "int a[4] = int[4](10, 20, 30, 40); int i = 0; inc(a[i++]); b.r[0] = i; b.r[1] = a[0]; b.r[2] = a[1];",
         {1, 11, 20}, {}},
        {"out_arg_call_index", "int n = 0; int next() { n++; return n - 1; } void put(out int x) { x = 9; }",
         "int a[4] = int[4](0, 0, 0, 0); put(a[next()]); b.r[0] = n; b.r[1] = a[0]; b.r[2] = a[1];",
         {1, 9, 0}, {}},
        {"swizzle_compound_dynamic_index", "",
         "vec4 v = vec4(1.0, 2.0, 3.0, 4.0); int k = 0; v[k++] *= 10.0; b.r[0] = k; b.f[0] = v.x; b.f[1] = v.y;",
         {1}, {10.0f, 2.0f}},
        // Short circuit and ternary evaluate one side.
        {"short_circuit", "",
         "int y = 0; bool t = (z != 0) && (y++ > 0); bool u = (o != 0) || (y++ > 0); bool w = (o == 0) ^^ (y++ > 5);"
         "b.r[0] = y; b.r[1] = t ? 1 : 0; b.r[2] = u ? 1 : 0; b.r[3] = w ? 1 : 0; int q = o != 0 ? (y += 10) : (y += 100); b.r[4] = q; b.r[5] = y;",
         {1, 0, 1, 0, 11, 11}, {}},
        // Precedence and associativity.
        {"precedence", "",
         "b.r[0] = 2 + 3 * 4 - 8 / 2 % 3; b.r[1] = 1 << 2 + 1; b.r[2] = 6 & 3 | 8 ^ 1; b.r[3] = (10 - i0) % i1; b.r[4] = i0 - i1 - i3;"
         "int x, y; x = y = i0; b.r[5] = x + y; b.r[6] = (i2 < 0) == (i0 > 0) ? 1 : 2; b.r[7] = ~i0 & 0xF; b.r[8] = !(i0 > i1) ? 7 : 8;",
         {2 + 12 - (8 / 2) % 3, 8, (6 & 3) | (8 ^ 1), 7 % 5, 3 - 5 - 2, 6, 1, (~3) & 0xF, 7}, {}},
        {"comma", "", "int c = 0; int d = (c = 3, c + 4); b.r[0] = d; b.r[1] = c;", {7, 3}, {}},
        // Integer arithmetic.
        {"int_ops", "",
         "b.r[0] = 7 / -2 + i3 - 2; b.r[1] = -8 >> 1; b.r[2] = int(0x80000000u >> 31); b.r[3] = 1 << 31; b.r[4] = int(0xFFFFFFFFu + uint(o));"
         "b.r[5] = int(uint(z) - 1u); b.r[6] = int(-1.9 + f1 * 0.0); b.r[7] = int(uint(f1 * 3.9)); b.r[8] = bool(f0) ? 1 : 0; b.r[9] = int(float(o) + 0.5);"
         "uint ux = 0x80000000u; b.r[10] = int(ux >> uint(i1)); b.r[11] = i2 >> 1; b.r[12] = int(uint(i2) >> 28u);"
         "b.r[13] = int(ux / 3u); b.r[14] = int(ux % 7u);",
         {-3, -4, 1, int32_t(0x80000000u), 0, -1, -1, 7, 1, 1, int32_t(0x80000000u >> 5), -4, 15, int32_t(0x80000000u / 3u), int32_t(0x80000000u % 7u)}, {}},
        // Constant expressions are folded the way the run-time arithmetic works.
        {"const_fold",
         "const int K = 7 / 2; const int Kn = -7 / 2; const float F = 7.0 / 2.0; const bool Bn = !true; const int Sh = 1 << 4;"
         "const ivec2 IV = ivec2(3.7, -3.7); const vec2 V = vec2(1.0, 2.0); const float Vy = V.y; const uint W = 5u - 6u; const int Bi = ~0;"
         "const float Fl = 16777217.0; const int Mod = 7 % 3; const float Lg = float(5) / 2.0;"
         "const int Sq = int(sqrt(16.0)); const float Mx = max(1.5, 2.5); const int Ci = clamp(9, 1, 4); const int Ab = abs(-4); const uint Ub = ~0u;",
         "b.r[0] = K; b.r[1] = Kn; b.f[0] = F; b.r[2] = Bn ? 1 : 0; b.r[3] = Sh; b.r[4] = IV.x; b.r[5] = IV.y; b.f[1] = Vy; b.r[6] = int(W); b.r[7] = Bi;"
         "b.f[2] = Fl; b.r[8] = Mod; b.f[3] = Lg; b.r[9] = Sq; b.f[4] = Mx; b.r[10] = Ci; b.r[11] = Ab; b.r[12] = int(Ub);",
         {3, -3, 0, 16, 3, -3, -1, -1, 1, 4, 4, 4, -1}, {3.5f, 2.0f, 16777216.0f, 2.5f, 2.5f}},
        // Matrices (column-major).
        {"matrix", "",
         "mat2 m = mat2(1.0, 2.0, 3.0, 4.0); vec2 a = m * vec2(1.0); vec2 c = vec2(1.0) * m; mat2 p = m * m; b.f[0] = a.x; b.f[1] = a.y; b.f[2] = c.x; b.f[3] = c.y;"
         "b.f[4] = p[0][0]; b.f[5] = p[1][0]; b.f[6] = m[1][0]; b.f[7] = determinant(m); mat2 iv = inverse(m); b.f[8] = iv[0][0]; b.f[9] = iv[1][0];"
         "mat3 m3 = mat3(m); b.f[10] = m3[2][2]; b.f[11] = m3[1][1]; b.f[12] = m3[0][2]; mat2 mm = mat2(vec4(5.0, 6.0, 7.0, 8.0)); b.f[13] = mm[1][0];"
         "mat2x3 t = mat2x3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0); mat3x2 tt = transpose(t); b.f[14] = tt[2][1]; b.f[15] = tt[0][1];"
         "mat2 op = outerProduct(vec2(1.0, 2.0), vec2(3.0, 4.0)); b.f[16] = op[1][0]; b.f[17] = op[0][1];"
         "mat2 mc = matrixCompMult(m, m); b.f[18] = mc[1][1]; mat2 neg = -m; b.f[19] = neg[1][0]; m[1] = vec2(9.0); m[0][1] += 1.0; b.f[20] = m[1][1]; b.f[21] = m[0][1];"
         "b.r[0] = (m == m) ? 1 : 0; b.r[1] = (m != mat2(1.0)) ? 1 : 0; mat3 d = mat3(2.0); b.f[22] = d[1][1] + d[0][1];",
         {1, 1},
         {4.0f, 6.0f, 3.0f, 7.0f, 7.0f, 15.0f, 3.0f, -2.0f, -2.0f, 1.5f, 1.0f, 4.0f, 0.0f, 7.0f, 6.0f, 4.0f, 4.0f, 6.0f, 16.0f, -3.0f, 9.0f, 3.0f, 2.0f}},
        // Arrays and structs have value semantics.
        {"arrays_structs",
         "struct S { int a; float v[2]; }; void mod(S s) { s.a = 99; s.v[0] = 99.0; } void modi(inout S s) { s.a += 1; s.v[1] = 7.0; }"
         "int[3] mk() { return int[3](4, 5, 6); } void arr(int a[3]) { a[0] = 100; }",
         "S s = S(1, float[2](1.5, 2.5)); S t = s; t.a = 2; mod(s); modi(t); b.r[0] = s.a; b.r[1] = t.a; b.f[0] = s.v[0]; b.f[1] = t.v[1];"
         "int a[3] = mk(); arr(a); b.r[2] = a[0]; b.r[3] = mk()[2]; b.r[4] = a.length(); int g[2][3]; g[1][2] = 8; b.r[5] = g[1][2]; b.r[6] = g.length() * 10 + g[0].length();"
         "S u = S(1, float[2](1.5, 2.5)); b.r[7] = (u == s) ? 1 : 0; b.r[8] = (u == t) ? 1 : 0; int k[] = int[](1, 2, 3, 4); b.r[9] = k.length();",
         {1, 3, 4, 6, 3, 8, 23, 1, 0, 4}, {1.5f, 7.0f}},
        // Function parameters.
        {"params", "void f(in int a, out int b, inout int c) { a = 50; b = a + 1; c *= 2; } int g(int x) { if (x > 2) return 1; return 2; }"
                   "void two(out int a, out int b) { a = 1; b = 2; } void incr(inout vec3 v) { v.yz += 1.0; }",
         "int a = 1, bb = 0, c = 4; f(a, bb, c); b.r[0] = a; b.r[1] = bb; b.r[2] = c; b.r[3] = g(i0) + g(1);"
         "vec3 v = vec3(1.0); incr(v); b.f[0] = v.x; b.f[1] = v.z; vec3 w = vec3(0.0); incr(w.zyx); b.f[2] = w.z;",
         {1, 51, 8, 3}, {1.0f, 2.0f, 0.0f}},
        // Control flow.
        {"control", "",
         "int s = 0; for (int i = 0; i < 6; i++) { if (i == 1) continue; if (i == 4) break; s += i; } b.r[0] = s;"
         "int d = 0; do { d++; if (d < 3) continue; break; } while (true); b.r[1] = d;"
         "int w = 10; while (w > 0) { w -= 3; } b.r[2] = w; int t = 0;"
         "for (int i = 0; i < 4; i++) { switch (i) { case 0: t += 1; case 1: t += 10; break; case 2: continue; default: t += 1000; } t += 100000; } b.r[3] = t;"
         "int u = 0; switch (i0) { default: u += 1; case 5: u += 2; break; case 3: u += 4; case 4: u += 8; } b.r[4] = u;"
         "int v = 0; switch (i1 + 1) { case 1: v = 1; break; case 6: v = 6; break; default: v = 9; } b.r[5] = v;"
         "int n = 0; for (int i = 0, j = 10; i < j; i += 2, j--) n++; b.r[6] = n;"
         "int q = 0; for (int i = 0; i < 3; i++) { for (int k = 0; k < 3; k++) { if (k == 1) continue; if (i == 2) break; q += 1; } } b.r[7] = q;"
         "int sw = 0; for (int i = 0; i < 3; i++) { switch (i) { case 1: sw += 5; continue; } sw += 1; } b.r[8] = sw;"
         "int sw2 = 0; for (int i = 0; i < 3; i++) { switch (i) { case 1: switch (i0) { case 3: continue; } sw2 += 100; break; } sw2 += 1; } b.r[9] = sw2;",
         {0 + 2 + 3, 3, -2, 301021, 4 + 8, 6, 4, 4, 1 + 5 + 1, 1 + 1}, {}},
        // Swizzles.
        {"swizzle", "",
         "vec4 v = vec4(1.0, 2.0, 3.0, 4.0); v.zx = vec2(10.0, 20.0); b.f[0] = v.x; b.f[1] = v.z; vec4 w = v; w.xyz = w.yzx; b.f[2] = w.x; b.f[3] = w.z;"
         "vec3 u = vec3(1.0, 2.0, 3.0); u.x += u.y; u.yz *= vec2(2.0, 3.0); b.f[4] = u.x; b.f[5] = u.y; b.f[6] = u.z; b.f[7] = (u + u).zy.x;"
         "ivec4 q = ivec4(1, 2, 3, 4); q.wzyx = ivec4(i0, i1, i2, i3); b.r[0] = q.x; b.r[1] = q.w; vec4 s = vec4(1.0, 2.0, 3.0, 4.0); b.f[8] = s.rgba.abgr.x; b.f[9] = s.stpq.w;"
         "vec2 p = vec2(1.0, 2.0); p.yx = p.xy; b.f[10] = p.x; b.f[11] = p.y; v[o] = 77.0; b.f[12] = v.y; v.xy.y; vec4 vv = vec4(0.0); vv.yw.y = 5.0; b.f[13] = vv.w;",
         {2, 3}, {20.0f, 10.0f, 2.0f, 20.0f, 3.0f, 4.0f, 9.0f, 18.0f, 4.0f, 4.0f, 2.0f, 1.0f, 77.0f, 5.0f}},
        // Compound assignment forms.
        {"compound_ops", "",
         "int a = 10; a -= 3; a *= 2; a /= 3; a %= 3; b.r[0] = a; uint u = 3u; u <<= 2u; u |= 1u; u ^= 5u; u >>= 1u; b.r[1] = int(u); int x = 5; x <<= 1; x &= 12; b.r[2] = x;"
         "float f = 8.0; f /= 2.0; f -= 1.0; f *= f; b.f[0] = f; vec2 v = vec2(1.0, 2.0); v *= 2.0; v /= vec2(2.0, 4.0); v += 1.0; v -= vec2(0.5); b.f[1] = v.x; b.f[2] = v.y;"
         "mat2 m = mat2(1.0); m *= 3.0; m += mat2(1.0); vec2 w = vec2(1.0, 1.0); w *= m; b.f[3] = w.x; b.f[4] = m[1][1]; int y = 3, z2 = 4; y += z2 -= 1; b.r[3] = y; b.r[4] = z2;",
         {1, int((3u << 2 | 1u) ^ 5u) >> 1, 8 & 12, 6, 3}, {9.0f, 1.5f, 1.5f, 4.0f, 4.0f}},
        // Builtin functions with the values the specification gives.
        {"builtins_float", "",
         "b.f[0] = mod(-1.5, 1.0); b.f[1] = fract(-1.25); b.f[2] = floor(-1.5); b.f[3] = ceil(-1.5); b.f[4] = trunc(-1.7); b.f[5] = round(-2.6 + f0 * 0.0); b.f[6] = sign(f2);"
         "float ip; b.f[7] = modf(-3.75, ip); b.f[8] = ip; int ex; b.f[9] = frexp(8.0, ex); b.f[10] = float(ex); b.f[11] = ldexp(0.75, 3);"
         "b.f[12] = unpackHalf2x16(packHalf2x16(vec2(1.5, -2.0))).y; b.f[13] = unpackUnorm4x8(packUnorm4x8(vec4(1.0, 0.0, 0.4, 0.2))).z; b.f[14] = unpackSnorm2x16(packSnorm2x16(vec2(-1.0, 0.5))).x;"
         "b.f[15] = 5.0; b.f[16] = dot(vec3(1.0, 2.0, 3.0), vec3(4.0, 5.0, 6.0)); b.f[17] = 1.0;"
         "b.f[18] = 7.0; b.f[19] = 0.8; b.f[20] = 1.0; b.f[21] = -1.0;"
         "b.f[22] = -1.0; b.f[23] = pow(2.0, 10.0); b.f[24] = exp2(3.0); b.f[25] = log2(8.0); b.f[26] = sqrt(16.0); b.f[27] = inversesqrt(4.0);"
         "b.f[28] = smoothstep(0.0, 2.0, 1.0); b.f[29] = step(1.0, 1.0); b.f[30] = mix(1.0, 3.0, 0.25); b.f[31] = clamp(5.0, 0.0, 1.0); b.f[32] = degrees(3.14159265358979); b.f[33] = radians(180.0);"
         "b.f[34] = atan(1.0, -1.0); b.f[35] = tanh(0.0) + sinh(0.0) + cosh(0.0); b.f[36] = float(isnan(b.f[40]) ? 1 : 0); b.f[37] = float(isinf(1.0 / (f0 * 0.0)) ? 1 : 0); b.f[38] = abs(f2); b.f[39] = min(f0, f1) + max(f0, f1);",
         {}, {0.5f, 0.75f, -2.0f, -1.0f, -1.0f, -3.0f, -1.0f, -0.75f, -3.0f, 0.5f, 4.0f, 6.0f, -2.0f, 102.0f / 255.0f, -1.0f, 5.0f, 32.0f, 1.0f, 7.0f, 0.8f,
          1.0f, -1.0f, -1.0f, 1024.0f, 8.0f, 3.0f, 4.0f, 0.5f, 0.5f, 1.0f, 1.5f, 1.0f, 180.0f, 3.14159265358979f, 3.0f * 3.14159265358979f / 4.0f, 1.0f, 0.0f, 1.0f, 1.5f, 2.5f}},
        {"builtins_int", "",
         "b.r[0] = bitCount(-1); b.r[1] = findLSB(0); b.r[2] = findLSB(8); b.r[3] = findMSB(-1); b.r[4] = findMSB(255); b.r[5] = findMSB(-256); b.r[6] = int(bitfieldReverse(1u) >> 24u) + int(bitfieldReverse(1u) & 1u);"
         "b.r[7] = bitfieldExtract(-16, 2, 3); b.r[8] = int(bitfieldExtract(0xF0u, 4, 4)); b.r[9] = bitfieldInsert(0, 5, 4, 3); b.r[10] = abs(i2); b.r[11] = sign(i2); b.r[12] = min(i0, i1) + max(i0, i1);"
         "b.r[13] = clamp(i2, -3, 3); ivec2 iv = min(ivec2(i0, i1), i3); b.r[14] = iv.x + iv.y; b.r[15] = int(any(lessThan(ivec3(1, 2, 3), ivec3(0, 0, 4)))); b.r[16] = int(all(equal(uvec2(1u, 2u), uvec2(1u, 2u))));"
         "b.r[17] = int(any(notEqual(bvec2(true, false), bvec2(true, false)))); uint c; uint s = uaddCarry(0xFFFFFFFFu, 1u, c); b.r[18] = int(s) + int(c); int hi, lo; imulExtended(i2, 100000000, hi, lo); b.r[19] = hi;"
         "uint mhi, mlo; umulExtended(0xFFFFFFFFu, 0xFFFFFFFFu, mhi, mlo); b.r[20] = int(mhi == 0xFFFFFFFEu && mlo == 1u); b.r[21] = floatBitsToInt(1.0) - 0x3f800000; b.r[22] = int(intBitsToFloat(0x40000000));",
         {32, -1, 3, -1, 7, 7, 128, -4 /* bits 4..2 of -16 = 100b, sign-extended */, 15, 80, 7, -1, 8, -3, 4, 1, 1, 0, 1, -1, 1, 0, 2}, {}},
        // Geometric built-ins: compiled and lowered only (the test interpreter has no length / normalize / cross ...).
        {"builtins_geometric", "",
         "b.f[0] = distance(vec2(1.0, 1.0), vec2(4.0, 5.0)); b.f[1] = cross(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0)).z; b.f[2] = length(vec3(2.0, 3.0, 6.0));"
         "b.f[3] = normalize(vec2(3.0, 4.0)).y; b.f[4] = reflect(vec2(1.0, -1.0), vec2(0.0, 1.0)).y; b.f[5] = faceforward(vec2(1.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 0.0)).x;"
         "b.f[6] = refract(vec2(0.0, -1.0), vec2(0.0, 1.0), 1.0).y;",
         {}, {}, false},
        // Ternary on aggregates and mixed use.
        {"ternary_aggregates", "struct P { vec2 a; int n; };",
         "P p = P(vec2(1.0, 2.0), 3), q = P(vec2(5.0, 6.0), 7); P r = (o != 0) ? p : q; b.f[0] = r.a.y; b.r[0] = r.n; vec3 v = (z != 0) ? vec3(1.0) : vec3(0.0, 1.0, 2.0); b.f[1] = v.z;"
         "int a[2] = int[2](1, 2); int c[2] = int[2](3, 4); int d[2] = (o == 0) ? a : c; b.r[1] = d[1]; mat2 m = (o != 0) ? mat2(2.0) : mat2(3.0); b.f[2] = m[1][1];",
         {3, 4}, {2.0f, 2.0f, 2.0f}},
        // Scoping and shadowing.
        {"scopes", "int g = 5; int shadow(int g) { return g + 1; } float h() { return 2.0; }",
         "int a = g; { int g = 7; a += g; { int g = 9; a += g; } a += g; } b.r[0] = a; b.r[1] = shadow(10); b.r[2] = g; for (int g = 0; g < 2; g++) { a += g; } b.r[3] = a;"
         "float h = h(); b.f[0] = h;",
         {5 + 7 + 9 + 7, 11, 5, 5 + 7 + 9 + 7 + 1}, {2.0f}},
        // Unsigned / signed conversions.
        {"conversions", "",
         "b.r[0] = int(uint(i2)); b.f[0] = float(uint(i2)); uint big = uint(3000000000.0 + f0); b.r[1] = int(big); b.f[1] = float(big); b.r[2] = int(true) + int(false) + int(2.9) + int(-2.9);"
         "vec3 v = vec3(ivec3(1, -2, 3)); b.f[2] = v.y; ivec2 w = ivec2(vec2(1.9, -1.9)); b.r[3] = w.x * 10 + w.y; bvec3 bv = bvec3(vec3(0.0, 1.0, -2.0)); b.r[4] = int(bv.x) * 100 + int(bv.y) * 10 + int(bv.z);"
         "vec4 c = vec4(ivec2(1, 2), 3.0, true); b.f[3] = c.w; b.f[4] = c.y; uvec2 uu = uvec2(5, 6u); b.r[5] = int(uu.y); float fi = float(i1) / float(i3); b.f[5] = fi;",
         {-7, int32_t(3000000000u), 1 + 0 + 2 - 2, 1 * 10 - 1, 1 * 10 + 1, 6}, {4294967289.0f, 3000000000.5f, -2.0f, 1.0f, 2.0f, 2.5f}},
        // Vector component operations.
        {"vector_ops", "",
         "vec3 a = vec3(1.0, 2.0, 3.0); vec3 c = a * 2.0 + vec3(1.0) - a / 2.0; b.f[0] = c.x; b.f[1] = c.z; ivec3 k = ivec3(7, 8, 9) % 4; b.r[0] = k.x * 100 + k.y * 10 + k.z; uvec3 u = uvec3(10u) / uvec3(3u, 4u, 5u); b.r[1] = int(u.x * 100u + u.y * 10u + u.z);"
         "bvec2 bb = bvec2(true, false); bvec2 nb = not(bb); b.r[2] = int(nb.x) * 2 + int(nb.y); vec4 m = mix(vec4(0.0), vec4(8.0), bvec4(true, false, true, false)); b.f[2] = m.x + m.y + m.z + m.w;"
         "ivec2 sh = ivec2(1, 2) << 2; b.r[3] = sh.x + sh.y; ivec2 sh2 = ivec2(16, 64) >> ivec2(2, 3); b.r[4] = sh2.x + sh2.y; vec2 q = -vec2(1.0, 2.0); b.f[3] = q.y; ivec2 qi = -ivec2(i0, i1); b.r[5] = qi.y;",
         {301, 322, 1, 4 + 8, 4 + 8, -5}, {2.5f, 5.5f, 16.0f, -2.0f}},
        // The middle operand of ?: is an `expression` (comma allowed); the last one an assignment_expression (ES 3.00 grammar).
        {"ternary_comma", "",
         "int a = 0; int k = (i0 > 0) ? (a = 5, a + 1) : 7; b.r[0] = k; b.r[1] = a;"
         "int a2 = 0; int k2 = (i0 < 0) ? a2 = 5, a2 + 1 : (a2 = 9); b.r[2] = k2; b.r[3] = a2;"
         "int k3 = (i0 > 0) ? i1, i2 : i3; b.r[4] = k3;",
         {6, 5, 9, 9, -7}, {}},
        // roundEven ties go to the even integer; round ties are the implementation's choice (ES 3.00 8.3), both neighbours pass.
        {"round_ties", "",
         "b.f[0] = roundEven(2.5); b.f[1] = roundEven(3.5); b.f[2] = roundEven(-2.5); b.f[3] = roundEven(f0); b.f[4] = round(2.6); b.f[5] = round(-2.6);"
         "float t1 = round(2.5), t2 = round(-2.5), t3 = round(f0), t4 = round(f2);"
         "b.r[0] = int(t1 == 2.0 || t1 == 3.0); b.r[1] = int(t2 == -3.0 || t2 == -2.0); b.r[2] = int(t3 == 0.0 || t3 == 1.0); b.r[3] = int(t4 == -2.0 || t4 == -1.0);",
         {1, 1, 1, 1}, {2.0f, 4.0f, -2.0f, 0.0f, 3.0f, -3.0f}},
    };
    for (const Case& c : cases) runCase(c);
}

// ---------------------------------------------------------------------------------------------
std::string pp(const std::string& src, int stage = 1) {
    std::string log;
    int version = 0;
    std::string out = vpglsl::preprocess(src, stage == 0 ? Stage::Vertex : Stage::Fragment, log, version);
    return out;
}

bool contains(const std::string& hay, const char* needle) { return hay.find(needle) != std::string::npos; }

void testPreprocessor() {
    // A macro whose body is the name of a function-like macro takes the arguments that follow it (C99 6.10.3.4).
    CHECK(contains(pp("#define BAR(x) (x+1)\n#define FOO BAR\nint a = FOO(2);"), "( 2 + 1 )"));
    CHECK(contains(pp("#define BAR(x) (x+1)\n#define FOO BAR\n#define FOO2 FOO\nint a = FOO2(7) + FOO2 (8);"), "( 7 + 1 ) + ( 8 + 1 )"));
    // ... but an object-like macro that is not followed by "(" leaves the name alone.
    CHECK(contains(pp("#define BAR(x) (x+1)\n#define FOO BAR\nint BAR = FOO;"), "int BAR = BAR ;"));
    // 64-bit corner cases of the #if evaluator do not trap (INT64_MIN / -1 is a hardware fault) and wrap like 32-bit-minded code expects.
    CHECK(contains(pp("#if 0x8000000000000000 / -1\nint a;\n#else\nint b;\n#endif"), "int a ;") ||
          contains(pp("#if 0x8000000000000000 / -1\nint a;\n#else\nint b;\n#endif"), "int b ;"));
    CHECK(contains(pp("#if (0x8000000000000000 % -1) == 0\nint a;\n#endif"), "int a ;"));
    CHECK(contains(pp("#if 0x7FFFFFFFFFFFFFFF + 1 < 0\nint a;\n#endif"), "int a ;"));
    CHECK(contains(pp("#if (1 << 63) < 0\nint a;\n#endif"), "int a ;"));
    CHECK(contains(pp("#if -9223372036854775808 < 0\nint a;\n#endif"), "int a ;"));
    CHECK(contains(pp("#if 4611686018427387904 * 4 == 0\nint a;\n#endif"), "int a ;"));
}

struct LinkResultOrError {
    bool ok = false;
    std::string log;
};

std::string compileLog(Stage st, const std::string& src, bool* ok = nullptr) {
    auto r = vpglsl::parse(st, src);
    if (ok) *ok = r.shader != nullptr;
    return r.log;
}

void testConstants() {
    // A float constant keeps float precision through float(int): float(16777217) is 16777216, and adding 1.0 rounds to even.
    runCase({"const_float_ctor", "const float F = float(16777217) + 1.0; const float G = float(16777217);",
             "b.f[0] = F; b.f[1] = G; b.f[2] = float(16777217) + f1 - 2.0;", {}, {16777216.0f, 16777216.0f, 16777216.0f}});
    // A constant vector's components are constant expressions (array sizes, case labels): ES 3.10 4.3.3.
    {
        bool ok = false;
        std::string log = compileLog(Stage::Compute,
            "#version 310 es\nlayout(local_size_x = 1) in;\nconst ivec2 K = ivec2(2, 3);\nconst vec2 V = vec2(1.0, 4.0);\n"
            "layout(std430, binding = 0) buffer B { float r[16]; } b;\nvoid main() { float a[K.y]; a[2] = 1.0; float c[int(V.y)]; c[3] = 2.0; b.r[0] = a[2] + c[3] + float(a.length() + c.length()); }\n", &ok);
        if (!ok) std::fprintf(stderr, "  vector constant as array size: %s\n", log.c_str());
        CHECK(ok);
    }
    // ... and so are the elements of a constant array indexed by a constant, and a constant vector's arithmetic.
    runCase({"const_array_vector",
             "const int A[3] = int[3](2, 5, 7); const vec2 V = vec2(1.0, 4.0) * 2.0 + vec2(1.0); const ivec3 I = -ivec3(1, 2, 3) * 2;",
             "float xa[A[1]]; float ya[int(V.y)]; float za[I.z + 20]; b.r[0] = xa.length(); b.r[1] = ya.length(); b.r[2] = za.length();"
             "switch (i0) { case A[2] - 4: b.r[3] = 1; break; default: b.r[3] = 0; }",
             {5, 9, 14, 1}, {}});
    runCase({"literals", "",
             "b.r[0] = 010; b.r[1] = 0x1F; b.r[2] = int(0xFFu); b.r[3] = 0xFFFFFFFF; b.f[0] = .5; b.f[1] = 5.; b.f[2] = 1.5e-3; b.f[3] = 1e2; b.f[4] = 1.0 / -0.0 * 0.0;"
             "b.f[5] = float(0x7FFFFFFF); b.f[6] = 0.1 + 0.2; b.r[4] = int(1.0 / (-0.0 * f0) < 0.0);",
             {8, 31, 255, -1, 1}, {0.5f, 5.0f, 1.5e-3f, 100.0f, std::nanf(""), 2147483648.0f, 0.1f + 0.2f}});
}

// ---------------------------------------------------------------------------------------------
LinkResultOrError linkPair(const std::string& vs, const std::string& fs) {
    LinkResultOrError r;
    auto v = vpglsl::parse(Stage::Vertex, vs);
    auto f = vpglsl::parse(Stage::Fragment, fs);
    if (!v.shader || !f.shader) {
        r.log = v.log + f.log;
        return r;
    }
    vpglsl::LinkResult lr = vpglsl::link(v.shader, f.shader, {});
    r.ok = lr.ok;
    r.log = lr.log;
    return r;
}

void testLinkRules() {
    const char* fsOk = "#version 300 es\nprecision mediump float; in vec4 v; out vec4 o; void main() { o = v; }";
    // Two active attributes on one location make the link fail (ES 3.00 2.11.3; strict mode: lenient drivers take the first).
    const char* vsAlias = "#version 300 es\nlayout(location = 0) in vec4 a; layout(location = 0) in vec4 b; out vec4 v; void main() { v = a + b; gl_Position = v; }";
    vpglsl::setStrictMode(1);
    {
        LinkResultOrError r = linkPair(vsAlias, fsOk);
        CHECK(!r.ok);
        if (r.ok) std::fprintf(stderr, "  two attributes on location 0 linked\n");
    }
    {  // an unused attribute does not count
        LinkResultOrError r = linkPair("#version 300 es\nlayout(location = 0) in vec4 a; layout(location = 0) in vec4 b; out vec4 v; void main() { v = a; gl_Position = v; }", fsOk);
        CHECK(r.ok);
    }
    {  // a matrix takes consecutive locations: mat2 at 0 covers 0..1
        LinkResultOrError r = linkPair("#version 300 es\nlayout(location = 0) in mat2 m; layout(location = 1) in vec4 b; out vec4 v; void main() { v = vec4(m[0], m[1]) + b; gl_Position = v; }", fsOk);
        CHECK(!r.ok);
    }
    {  // two fragment outputs on one location
        LinkResultOrError r = linkPair("#version 300 es\nin vec4 a; out vec4 v; void main() { v = a; gl_Position = a; }",
                                       "#version 300 es\nprecision mediump float; in vec4 v; layout(location = 0) out vec4 o; layout(location = 0) out vec4 p; void main() { o = v; p = v; }");
        CHECK(!r.ok);
    }
    {  // an array output covering the next output's location
        LinkResultOrError r = linkPair("#version 300 es\nin vec4 a; out vec4 v; void main() { v = a; gl_Position = a; }",
                                       "#version 300 es\nprecision mediump float; in vec4 v; layout(location = 0) out vec4 o[2]; layout(location = 1) out vec4 p; void main() { o[0] = v; o[1] = v; p = v; }");
        CHECK(!r.ok);
    }
    {  // distinct locations link
        LinkResultOrError r = linkPair("#version 300 es\nin vec4 a; out vec4 v; void main() { v = a; gl_Position = a; }",
                                       "#version 300 es\nprecision mediump float; in vec4 v; layout(location = 0) out vec4 o; layout(location = 1) out vec4 p; void main() { o = v; p = v; }");
        CHECK(r.ok);
    }
    {  // GLSL ES 1.00 (ES 2.0 2.10.4) lets glBindAttribLocation alias attributes: the strict link does not refuse it
        vpglsl::LinkOptions lo;
        lo.attribBindings = {{"a", 0}, {"b", 0}};
        auto v100 = vpglsl::parse(Stage::Vertex, "attribute vec4 a; attribute vec4 b; varying vec4 v; void main() { v = a + b; gl_Position = v; }");
        auto f100 = vpglsl::parse(Stage::Fragment, "precision mediump float; varying vec4 v; void main() { gl_FragColor = v; }");
        auto v300 = vpglsl::parse(Stage::Vertex, "#version 300 es\nin vec4 a; in vec4 b; out vec4 v; void main() { v = a + b; gl_Position = v; }");
        auto f300 = vpglsl::parse(Stage::Fragment, fsOk);
        CHECK(v100.shader && f100.shader && v300.shader && f300.shader);
        if (v100.shader && f100.shader) {
            auto l = vpglsl::link(v100.shader, f100.shader, lo);
            if (!l.ok) std::fprintf(stderr, "  ES 1.00 aliased attributes: %s\n", l.log.c_str());
            CHECK(l.ok);
        }
        if (v300.shader && f300.shader) CHECK(!vpglsl::link(v300.shader, f300.shader, lo).ok);  // (ES 3.00 still refuses)
    }
    vpglsl::setStrictMode(0);
    {  // lenient (the default): the same shaders link, the rule is only noted
        LinkResultOrError r = linkPair(vsAlias, fsOk);
        CHECK(r.ok);
    }
}

// ---------------------------------------------------------------------------------------------
// Mutation fuzz of the lexer, preprocessor, parser and linker: whatever the text, no crash, hang or sanitizer report.
struct Lcg {
    uint64_t s;
    uint32_t next() { return uint32_t((s = s * 6364136223846793005ull + 1442695040888963407ull) >> 33); }
    uint32_t below(uint32_t n) { return n ? next() % n : 0; }
};

void testFuzz() {
    static const char* seeds[] = {
        "#version 300 es\nprecision highp float; uniform sampler2D t; uniform vec4 u[4]; in vec2 uv; flat in int k; out vec4 o;\n"
        "struct S { vec3 a; float b[2]; }; S s = S(vec3(1.0), float[2](1.0, 2.0));\n"
        "float f(inout float x, out vec2 y, in int z) { y = vec2(x); x += float(z); return x * 2.0; }\n"
        "void main() { float q = 0.0; vec2 w; for (int i = 0; i < 4; i++) { if (i == 2) continue; q += f(q, w, i) * u[i].x; }\n"
        "switch (k) { case 0: q += 1.0; break; case 1: case 2: q -= 1.0; default: q *= 2.0; }\n"
        "o = texture(t, uv) * q + vec4(s.a, s.b[1]); mat3 m = mat3(o.xyz, w.xyx, vec3(1.0)); o.xyz = inverse(m) * o.xyz; }\n",
        "#version 310 es\nlayout(local_size_x = 8, local_size_y = 2) in;\nlayout(std430, binding = 0) buffer B { uint n; float v[]; } b;\n"
        "shared float tile[16]; layout(binding = 0) uniform atomic_uint c;\n"
        "void main() { uint i = gl_GlobalInvocationID.x; tile[gl_LocalInvocationIndex] = float(i); barrier();\n"
        "if (i < uint(b.v.length())) b.v[i] = tile[(gl_LocalInvocationIndex + 1u) % 16u]; atomicAdd(b.n, 1u); atomicCounterIncrement(c); }\n",
        "#version 100\nprecision mediump float; attribute vec4 a; varying vec4 v; uniform mat4 m;\n"
        "#define MUL(x, y) ((x) * (y))\n#ifdef GL_ES\n#define P 3\n#endif\n"
        "void main() { v = MUL(m, a) + float(P); gl_Position = v; }\n",
    };
    static const char* dict[] = {"(", ")", "{", "}", "[", "]", ";", ",", ".", "++", "--", "+=", "<<", "&&", "?", ":", "0", "1", "-1", "0x7fffffff",
                                 "4294967295u", "1e38", "float", "int", "uint", "vec4", "mat4", "void", "const", "in", "out", "inout", "struct",
                                 "if", "else", "for", "while", "do", "switch", "case", "default", "break", "continue", "return", "discard",
                                 "layout(location = 0)", "uniform", "#define A(x) x x\n", "#if 1\n", "#endif\n", "main", "gl_Position", "x", "y",
                                 "length()", "texture(t, uv)", "[2]", ".xyzw", "\n"};
    const int iterations = std::getenv("VPGLSL_FUZZ_ITERATIONS") ? std::atoi(std::getenv("VPGLSL_FUZZ_ITERATIONS")) : 1500;
    Lcg rng{0x9E3779B97F4A7C15ull};
    int compiled = 0, linked = 0;
    for (int it = 0; it < iterations; ++it) {
        std::string src = seeds[rng.below(3)];
        const int edits = 1 + int(rng.below(6));
        for (int e = 0; e < edits; ++e) {
            if (src.empty()) break;
            const size_t at = rng.below(uint32_t(src.size()));
            switch (rng.below(5)) {
                case 0: src.erase(at, 1 + rng.below(12)); break;
                case 1: src.insert(at, dict[rng.below(sizeof dict / sizeof *dict)]); break;
                case 2: src.insert(at, src.substr(rng.below(uint32_t(src.size())), 1 + rng.below(30))); break;
                case 3: src[at] = char(32 + rng.below(95)); break;
                default: std::swap(src[at], src[rng.below(uint32_t(src.size()))]); break;
            }
        }
        const Stage st = src.find("local_size") != std::string::npos ? Stage::Compute : (rng.below(2) ? Stage::Vertex : Stage::Fragment);
        auto r = vpglsl::parse(st, src);
        if (!r.shader) continue;
        ++compiled;
        if (st == Stage::Compute) {
            if (vpglsl::link(r.shader, nullptr, {}).ok) ++linked;
        } else {
            auto other = vpglsl::parse(st == Stage::Vertex ? Stage::Fragment : Stage::Vertex, src);
            if (other.shader) {
                auto lr = st == Stage::Vertex ? vpglsl::link(r.shader, other.shader, {}) : vpglsl::link(other.shader, r.shader, {});
                if (lr.ok) ++linked;
            }
        }
    }
    std::printf("  fuzz: %d iterations, %d compiled, %d linked\n", iterations, compiled, linked);
}

// Hostile inputs of a size and shape no game has: no stack overflow, no quadratic blow-up.
void testHostile() {
    auto fragment = [](const std::string& body) { return "#version 300 es\nprecision mediump float; out vec4 o;\n" + body; };
    const std::string deep(5000, '(');
    CHECK(!vpglsl::parse(Stage::Fragment, fragment("void main() { o = vec4" + deep + "1.0 }")).shader);
    CHECK(!vpglsl::parse(Stage::Fragment, fragment("void main() " + std::string(5000, '{') + std::string(5000, '}'))).shader);
    std::string chain;
    for (int i = 0; i < 20000; ++i) chain += "a" + std::to_string(i % 7) + " + ";
    vpglsl::parse(Stage::Fragment, fragment("float a0,a1,a2,a3,a4,a5,a6; void main() { float x = " + chain + "1.0; o = vec4(x); }"));
    std::string longIdent(100000, 'x');
    vpglsl::parse(Stage::Fragment, fragment("float " + longIdent + "; void main() { o = vec4(" + longIdent + "); }"));
    vpglsl::parse(Stage::Fragment, fragment("void main() { o = vec4(1.0e999999999999, 0x, 08, 1e, 0.0.0); }"));
    std::string many = "#define A(x) x x\n#define B(x) A(A(x))\n#define C(x) B(B(x))\nvoid main() { o = vec4(C(C(C(1.0)))); }";
    vpglsl::parse(Stage::Fragment, fragment(many));
    CHECK(!vpglsl::parse(Stage::Fragment, fragment("void main() { int a[3] = int[3](1, 2, 3); o = vec4(float(a[5])); }")).shader);
}


// ---------------------------------------------------------------------------------------------
// Vertex + fragment stages lowered the way the GLES layer does (no resources besides the attributes).
struct Stages {
    bool ok = false;
    vpglsl::LinkResult link;
    vpir::Module vs, fs;
    std::string vsMsl, fsMsl;
};

bool lowerOne(const std::vector<uint32_t>& words, vpir::Stage stage, const vpglsl::Reflection& refl, vpir::Module& out, std::string& msl) {
    std::string error;
    spirv::SpirvModule* m = spirv::parse(words.data(), words.size(), error);
    if (!m) {
        std::fprintf(stderr, "  spirv parse: %s\n", error.c_str());
        return false;
    }
    spirv::SpirvLowerOptions o;
    o.entry = "main";
    o.stage = stage;
    spirv::SpirvBinding b;
    b.set = 0, b.binding = vpglsl::kDefaultBlockBinding, b.kind = vpgpu::BindKind::ConstantBuffer, b.arraySize = 1, b.slot = 0;
    o.layout.push_back(b);
    if (stage == vpir::Stage::Vertex) {
        uint16_t slot = 30;
        for (const auto& a : refl.attributes) {
            spirv::VertexBufferBinding vb;
            vb.binding = uint32_t(a.location), vb.stride = 64, vb.slot = slot--;
            o.vertexBuffers.push_back(vb);
            spirv::VertexAttribute va;
            va.location = uint32_t(a.location), va.binding = uint32_t(a.location), va.offset = 0;
            va.format = vpgpu::Format::RGBA32Float;
            o.vertexAttributes.push_back(va);
        }
    }
    spirv::SpirvInterface iface;
    const bool lowered = spirv::lower(*m, o, out, iface, error);
    spirv::release(m);
    if (!lowered) {
        std::fprintf(stderr, "  lower: %s\n", error.c_str());
        return false;
    }
    auto errs = vpir::verify(out);
    if (!errs.empty()) {
        for (auto& e : errs) std::fprintf(stderr, "  verify: %s\n", e.c_str());
        return false;
    }
    vpmsl::EmitResult e = vpmsl::emitMsl(out, {});
    if (!e.errors.empty()) {
        for (auto& x : e.errors) std::fprintf(stderr, "  msl: %s\n", x.c_str());
        return false;
    }
    msl = e.shader.source;
    if (std::getenv("VPGLSL_PRINT")) std::printf("%s\n%s\n", vpir::print(out).c_str(), msl.c_str());
    return true;
}

Stages buildStages(const std::string& vsrc, const std::string& fsrc) {
    Stages r;
    auto v = vpglsl::parse(Stage::Vertex, vsrc);
    auto f = vpglsl::parse(Stage::Fragment, fsrc);
    if (!v.shader || !f.shader) {
        std::fprintf(stderr, "  compile: %s%s\n", v.log.c_str(), f.log.c_str());
        return r;
    }
    r.link = vpglsl::link(v.shader, f.shader, {});
    if (!r.link.ok) {
        std::fprintf(stderr, "  link: %s\n", r.link.log.c_str());
        return r;
    }
    r.ok = lowerOne(r.link.vertexSpirv, vpir::Stage::Vertex, r.link.reflection, r.vs, r.vsMsl) &&
           lowerOne(r.link.fragmentSpirv, vpir::Stage::Fragment, r.link.reflection, r.fs, r.fsMsl);
    return r;
}

// gl_PointCoord is the rasterizer's point sprite coordinate, not a constant: the fragment stage reads a point_coord input.
void testPointCoord() {
    Stages s = buildStages("#version 300 es\nin vec4 a; void main() { gl_Position = a; gl_PointSize = 8.0; }",
                           "#version 300 es\nprecision mediump float; out vec4 o; void main() { o = vec4(gl_PointCoord, 0.0, 1.0); }");
    CHECK(s.ok);
    if (!s.ok) return;
    const std::string text = vpir::print(s.fs);
    CHECK(text.find("point_coord") != std::string::npos);
    CHECK(s.fsMsl.find("[[point_coord]]") != std::string::npos);
    // Metal's [[point_coord]] has t = 0 at the top of the point, and the GLES layer keeps GL's row 0 (bottom) as memory row 0, so
    // t_GL = 1 - t_Metal: gl_PointCoord is (s, 1 - t) of the rasterizer's coordinate. (The Vulkan path, gl_PointCoord of a SPIR-V
    // module, is not touched: there row 0 is the top.)
    CHECK(s.fsMsl.find("1.0f - in_point_coord.y") != std::string::npos);
    CHECK(s.fsMsl.find("in_point_coord.x") != std::string::npos);
    {  // And by running the fragment stage: the rasterizer's (0.25, 0.125) reads as (0.25, 0.875)
        vpireval::Invocation inv;
        vpireval::Value pc;
        pc.type = vpir::kF32.withWidth(2);
        const float raster[2] = {0.25f, 0.125f};
        std::memcpy(pc.bits, raster, sizeof raster);
        inv.builtins[vpir::Builtin::PointCoord] = pc;
        std::string err;
        CHECK(vpireval::run(s.fs, inv, err));
        if (!err.empty()) std::fprintf(stderr, "  eval: %s\n", err.c_str());
        float got[4] = {};
        if (inv.outputs.count(0)) std::memcpy(got, inv.outputs[0].bits, sizeof got);
        CHECK(got[0] == 0.25f && got[1] == 0.875f && got[2] == 0.0f && got[3] == 1.0f);
    }
    // Unused: no input.
    Stages u = buildStages("#version 300 es\nin vec4 a; void main() { gl_Position = a; }",
                           "#version 300 es\nprecision mediump float; out vec4 o; void main() { o = vec4(1.0); }");
    CHECK(u.ok && u.fsMsl.find("point_coord") == std::string::npos);
}

// Structure varyings (ES 3.00 4.3.4: structs of non-array, non-bool members) link across the stages and take consecutive locations.
void testStructVaryings() {
    const char* vs =
        "#version 300 es\nstruct S { vec3 c; float w; mat2 m; };\nin vec4 a; out S s; flat out ivec2 id; out vec2 t;\n"
        "void main() { s.c = a.xyz; s.w = a.w; s.m = mat2(a.x); id = ivec2(a.xy); t = a.zw; gl_Position = a; }";
    const char* fs =
        "#version 300 es\nprecision mediump float; struct S { vec3 c; float w; mat2 m; };\nin S s; flat in ivec2 id; in vec2 t; out vec4 o;\n"
        "void main() { o = vec4(s.c, s.w) + vec4(s.m[1], 0.0, 0.0) + vec4(float(id.x), t, 0.0); }";
    Stages r = buildStages(vs, fs);
    CHECK(r.ok);
    if (!r.ok) return;
    // every fragment input has a vertex output at the same location and type
    int matched = 0;
    for (const auto& in : r.fs.inputs) {
        if (in.builtin != vpir::Builtin::None) continue;
        bool found = false;
        for (const auto& out : r.vs.outputs)
            if (out.builtin == vpir::Builtin::None && out.location == in.location && out.type == in.type) found = true;
        CHECK(found);
        matched += found;
    }
    CHECK(matched >= 6);  // c, w, m[0], m[1], id, t
    // an integer struct is flat in both stages and keeps its members flat
    Stages fl = buildStages("#version 300 es\nstruct S { ivec2 i; uint u; };\nin vec4 a; flat out S s; void main() { s.i = ivec2(a.xy); s.u = uint(a.z); gl_Position = a; }",
                            "#version 300 es\nprecision mediump float; struct S { ivec2 i; uint u; };\nflat in S s; out vec4 o; void main() { o = vec4(float(s.i.x), float(s.i.y), float(s.u), 1.0); }");
    CHECK(fl.ok);
    if (fl.ok) {
        int flat = 0;
        for (const auto& in : fl.fs.inputs) flat += in.builtin == vpir::Builtin::None && in.interp == vpir::Interp::Flat;
        CHECK(flat == 2);
    }
    Stages bad = buildStages("#version 300 es\nstruct S { vec3 c; }; in vec4 a; out S s; void main() { s.c = a.xyz; gl_Position = a; }",
                             "#version 300 es\nprecision mediump float; struct S { vec4 c; }; in S s; out vec4 o; void main() { o = s.c; }");
    CHECK(!bad.ok);  // same name, different structure: link error, not a crash
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    vpglsl::setStrictMode(0);
    const std::string g = argv[1];
    if (g == "sem") testSemantics();
    else if (g == "pp") testPreprocessor(), testConstants();
    else if (g == "link") testLinkRules();
    else if (g == "fuzz") testFuzz(), testHostile();
    else if (g == "iface") testPointCoord(), testStructVaryings();
    else return 2;
    std::printf("audit_glsl %s: %s\n", g.c_str(), failures ? "FAIL" : "ok");
    return failures ? 1 : 0;
}
