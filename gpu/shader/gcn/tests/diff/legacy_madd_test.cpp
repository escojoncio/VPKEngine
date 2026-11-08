// v_mul_legacy_f32 / v_mac_legacy_f32 / v_mad_legacy_f32 (DX9 multiply: a zero source makes the product zero even against
// infinity or NaN), lifted and run on the VPIR interpreter with directed values, bit for bit:
//   mul_legacy = (a == 0 || b == 0) ? +0 : a * b
//   mac / mad  = (a == 0 || b == 0) ? c  : a * b + c     (c itself, so -0 stays -0 and a NaN or inf c passes through)
// The MSL of the same program must not test the zero on the floats: Metal runs with fast math, which lets the compiler
// assume no NaN/inf and fold `a == 0 ? .. : a * b` back into a plain multiply (0 * inf = NaN again).
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "vpgcn/lift.h"
#include "vpir_eval.h"
#include "vpmsl/msl.h"

namespace {

// v_mov v1,s0; v_mov v2,s1; v_mov v3,s2; v_mad_legacy_f32 v4,v1,v2,v3; v_mul_legacy_f32 v5,v1,v2; v_mov v6,v3;
// v_mac_legacy_f32 v6,v1,v2; exp pos0 v1.. ; exp param0 v4,v5,v6,v6 done; s_endpgm   (llvm-mc -mcpu=bonaire)
const std::vector<uint32_t> kCode = {0x7e020200, 0x7e040201, 0x7e060202, 0xd2800004, 0x040e0501, 0x0e0a0501, 0x7e0c0303,
                                     0x0c0c0501, 0xf80000cf, 0x01010101, 0xf8000a0f, 0x06060504, 0xbf810000};

constexpr uint32_t kInf = 0x7f800000, kNInf = 0xff800000, kNaN = 0x7fc00000, kNZero = 0x80000000;

uint32_t bits(float f) { return std::bit_cast<uint32_t>(f); }
float flt(uint32_t b) { return std::bit_cast<float>(b); }

struct Case {
    uint32_t a, b, c;
};

// The expected result words: {mad, mul, mac}. NaN is any NaN.
bool same(uint32_t want, uint32_t have) {
    if (std::isnan(flt(want))) return std::isnan(flt(have));
    return want == have;
}

}  // namespace

int main() {
    vpgcn::LiftInput li;
    li.stage = vpgcn::ShaderStage::Vertex;
    li.userSgprs = 16;
    li.vgprCompCnt = 0;
    li.code = kCode;
    vpgcn::LiftResult r = vpgcn::lift(li);
    if (!r.errors.empty()) {
        std::printf("FAIL: lift error %s\n", r.errors.front().c_str());
        return 1;
    }
    int failed = 0;
    const Case cases[] = {
        {bits(0.0f), kInf, bits(5.0f)},        // 0 * inf
        {bits(0.0f), kNaN, bits(5.0f)},        // 0 * NaN
        {kInf, bits(0.0f), bits(5.0f)},        // inf * 0
        {kNaN, bits(0.0f), bits(5.0f)},        // NaN * 0
        {kNInf, kNZero, bits(5.0f)},           // -inf * -0
        {bits(0.0f), kInf, kNaN},              // c NaN passes through
        {bits(0.0f), bits(3.0f), kInf},        // c inf passes through
        {bits(0.0f), bits(3.0f), kNInf},
        {kNZero, kInf, kNZero},                // c = -0 stays -0 (0 + -0 would be +0)
        {bits(0.0f), bits(5.0f), kNZero},
        {bits(2.0f), bits(3.0f), bits(4.0f)},  // ordinary
        {bits(-2.0f), bits(3.0f), bits(4.0f)},
        {bits(-2.0f), bits(3.0f), kNZero},
        {bits(2.0f), bits(3.0f), kInf},
        {bits(2.0f), bits(3.0f), kNInf},
        {bits(2.0f), bits(3.0f), kNaN},
        {kInf, bits(2.0f), bits(1.0f)},        // non-zero times inf stays inf
        {bits(-2.0f), kInf, bits(1.0f)},
        {kInf, kInf, kNInf},                   // inf - inf = NaN, no zero involved
        {kNaN, bits(2.0f), bits(1.0f)},        // NaN times non-zero stays NaN
    };
    for (const Case& k : cases) {
        const bool zero = flt(k.a) == 0.0f || flt(k.b) == 0.0f;
        const uint32_t mulWant = zero ? 0u : bits(flt(k.a) * flt(k.b));
        const volatile float prod = flt(k.a) * flt(k.b);  // (rounded to float before the add: unfused, as on the hardware)
        const uint32_t madWant = zero ? k.c : bits(prod + flt(k.c));
        vpireval::Invocation inv;
        vpireval::Value vid;
        vid.type = vpir::kU32;
        inv.builtins[vpir::Builtin::VertexId] = vid;
        std::vector<uint8_t> ud(64, 0);
        const uint32_t w[3] = {k.a, k.b, k.c};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j) ud[size_t(i) * 4 + size_t(j)] = uint8_t(w[i] >> (8 * j));
        if (r.userDataSlot) inv.memory[*r.userDataSlot] = ud;
        std::string e;
        if (!vpireval::run(r.module, inv, e)) {
            std::printf("FAIL: eval error %s\n", e.c_str());
            return 1;
        }
        const vpireval::Value* got = nullptr;
        for (auto& [idx, val] : inv.outputs)
            if (r.module.outputs[idx].builtin == vpir::Builtin::None && r.module.outputs[idx].location == 0) got = &val;
        if (!got) {
            std::printf("FAIL: no param0 export\n");
            return 1;
        }
        const uint32_t want[3] = {madWant, mulWant, madWant};  // mad, mul, mac
        const char* name[3] = {"v_mad_legacy_f32", "v_mul_legacy_f32", "v_mac_legacy_f32"};
        for (int i = 0; i < 3; ++i)
            if (!same(want[i], got->bits[i])) {
                std::printf("FAIL: %s a=0x%08x b=0x%08x c=0x%08x: want 0x%08x, got 0x%08x\n", name[i], k.a, k.b, k.c, want[i],
                            got->bits[i]);
                ++failed;
            }
    }
    vpmsl::EmitResult em = vpmsl::emitMsl(r.module);
    if (!em.errors.empty()) {
        std::printf("FAIL: MSL emit error %s\n", em.errors.front().c_str());
        return 1;
    }
    if (em.shader.source.find("== 0.0f") != std::string::npos) {
        std::printf("FAIL: the MSL compares a float with 0.0f (fast math can fold the legacy-multiply select away):\n%s\n",
                    em.shader.source.c_str());
        ++failed;
    }
    std::printf("%d failures\n", failed);
    return failed ? 1 : 0;
}
