// Self-test of the reference interpreter: hand-checked cases per family. Encodings come from
// llvm-mc -mcpu=bonaire. Returns nonzero on any failure.
#include <bit>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "gcn_ref.h"

namespace {

using vpgcnref::Wave;

unsigned g_failures = 0;

uint32_t F(float f) { return std::bit_cast<uint32_t>(f); }
float AsF(uint32_t u) { return std::bit_cast<float>(u); }

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what.c_str());
        ++g_failures;
    }
}
void eq(uint64_t got, uint64_t want, const std::string& what) {
    if (got != want) {
        std::fprintf(stderr, "FAIL %s: got 0x%llx want 0x%llx\n", what.c_str(), (unsigned long long)got,
                     (unsigned long long)want);
        ++g_failures;
    }
}

// Decodes and executes one instruction; reports failure.
bool run(Wave& w, std::vector<uint32_t> words, const char* name) {
    vpgcn::Instruction in;
    if (!vpgcn::decode(words, in)) {
        std::fprintf(stderr, "FAIL decode %s\n", name);
        ++g_failures;
        return false;
    }
    std::string err;
    if (!vpgcnref::step(in, w, err)) {
        std::fprintf(stderr, "FAIL step %s: %s\n", name, err.c_str());
        ++g_failures;
        return false;
    }
    return true;
}

std::unique_ptr<Wave> fresh() { return std::make_unique<Wave>(); }

void salu() {
    auto w = fresh();
    Wave& W = *w;
    // s_add_u32 s0, s2, s3: carry out.
    W.s[2] = 0xFFFFFFFF, W.s[3] = 2;
    run(W, {0x80000302u}, "s_add_u32");
    eq(W.s[0], 1, "s_add_u32 value");
    check(W.scc, "s_add_u32 carry");
    // s_addc_u32 s1, s4, s5: consumes and clears the carry.
    W.s[4] = 0, W.s[5] = 0;
    run(W, {0x82010504u}, "s_addc_u32");
    eq(W.s[1], 1, "s_addc_u32 value");
    check(!W.scc, "s_addc_u32 no carry");
    // s_sub_u32 s0, s2, s3: borrow.
    W.s[2] = 1, W.s[3] = 2;
    run(W, {0x80800302u}, "s_sub_u32");
    eq(W.s[0], 0xFFFFFFFF, "s_sub_u32 value");
    check(W.scc, "s_sub_u32 borrow");
    // s_subb_u32 s1, s4, s5 with borrow in: 5 - 5 - 1.
    W.s[4] = 5, W.s[5] = 5;
    run(W, {0x82810504u}, "s_subb_u32");
    eq(W.s[1], 0xFFFFFFFF, "s_subb_u32 value");
    check(W.scc, "s_subb_u32 borrow");
    // s_add_i32: signed overflow sets SCC.
    W.s[2] = 0x7FFFFFFF, W.s[3] = 1;
    run(W, {0x81000302u}, "s_add_i32");
    eq(W.s[0], 0x80000000, "s_add_i32 value");
    check(W.scc, "s_add_i32 overflow");
    W.s[2] = 0xFFFFFFFF, W.s[3] = 1;
    run(W, {0x81000302u}, "s_add_i32 b");
    check(!W.scc, "s_add_i32 -1+1 no overflow");
    // s_min_u32: SCC = (s2 < s3).
    W.s[2] = 3, W.s[3] = 7;
    run(W, {0x83800302u}, "s_min_u32");
    eq(W.s[0], 3, "s_min_u32");
    check(W.scc, "s_min_u32 scc");
    // s_and_b64 s[0:1], s[2:3], exec: real 64-bit.
    W.s[2] = 0xF0F0F0F0, W.s[3] = 0x80000001, W.exec = 0xFFFFFFFF00000000ull;
    run(W, {0x87807E02u}, "s_and_b64");
    eq(W.s[0], 0, "s_and_b64 lo");
    eq(W.s[1], 0x80000001, "s_and_b64 hi");
    check(W.scc, "s_and_b64 scc");
    // s_lshl_b64 s[0:1], s[2:3], 33.
    W.s[2] = 0x3, W.s[3] = 0;
    run(W, {0x8F80A102u}, "s_lshl_b64");
    eq(W.s[0], 0, "s_lshl_b64 lo");
    eq(W.s[1], 6, "s_lshl_b64 hi");
    // s_ashr_i64 s[0:1], s[2:3], 4.
    W.s[2] = 0x00000010, W.s[3] = 0x80000000;
    run(W, {0x91808402u}, "s_ashr_i64");
    eq(W.s[0], 0x00000001, "s_ashr_i64 lo");
    eq(W.s[1], 0xF8000000, "s_ashr_i64 hi");
    // s_bfe_u32 / i32: offset 28, width 8 (offset + width > 32).
    W.s[2] = 0xF0000000, W.s[3] = (8u << 16) | 28;
    run(W, {0x93800302u}, "s_bfe_u32");
    eq(W.s[0], 0xF, "s_bfe_u32 off+width>32");
    run(W, {0x94000302u}, "s_bfe_i32");
    eq(W.s[0], 0xFFFFFFFF, "s_bfe_i32 off+width>32");
    W.s[2] = 0x00000780, W.s[3] = (4u << 16) | 7;  // field 0b1111 at bit 7
    run(W, {0x94000302u}, "s_bfe_i32 b");
    eq(W.s[0], 0xFFFFFFFF, "s_bfe_i32 sign");
    check(W.scc, "s_bfe_i32 scc");
    W.s[3] = 0 | 7;  // width 0
    run(W, {0x93800302u}, "s_bfe_u32 w0");
    eq(W.s[0], 0, "s_bfe_u32 width 0");
    check(!W.scc, "s_bfe_u32 width 0 scc");
    // s_bfe_u64 s[0:1], s[2:3], s4: 40-bit field at 4.
    W.s[2] = 0xFFFFFFFF, W.s[3] = 0xFFFFFFFF, W.s[4] = (40u << 16) | 4;
    run(W, {0x94800402u}, "s_bfe_u64");
    eq(W.s[0], 0xFFFFFFFF, "s_bfe_u64 lo");
    eq(W.s[1], 0xFF, "s_bfe_u64 hi");
    // s_cselect_b64 s[0:1], s[2:3], vcc.
    W.scc = false, W.vcc = 0x123456789ull, W.s[2] = 1, W.s[3] = 2;
    run(W, {0x85806A02u}, "s_cselect_b64");
    eq(W.s[0], 0x23456789, "s_cselect_b64 lo");
    eq(W.s[1], 1, "s_cselect_b64 hi");
    // s_and_saveexec_b64 s[0:1], vcc.
    W.exec = 0x00000000FFFFFFFFull, W.vcc = 0x0000000F0000000Full;
    run(W, {0xBE80246Au}, "s_and_saveexec_b64");
    eq(W.s[0], 0xFFFFFFFF, "saveexec saved lo");
    eq(W.s[1], 0, "saveexec saved hi");
    eq(W.exec, 0xF, "saveexec exec");
    check(W.scc, "saveexec scc");
    // s_xnor_saveexec_b64 s[0:1], s[2:3]: exec = ~(s ^ exec).
    W.exec = 0xF, W.s[2] = 0xF, W.s[3] = 0;
    run(W, {0xBE802B02u}, "s_xnor_saveexec_b64");
    eq(W.exec, ~0ull, "xnor_saveexec exec");
    eq(W.s[0], 0xF, "xnor_saveexec saved");
    // SOP1 bit ops.
    W.s[2] = 0x80000001, W.s[3] = 0x00000003;
    run(W, {0xBE801002u}, "s_bcnt1_i32_b64");
    eq(W.s[0], 4, "s_bcnt1_i32_b64");
    W.s[2] = 0x00000100;
    run(W, {0xBE801302u}, "s_ff1_i32_b32");
    eq(W.s[0], 8, "s_ff1");
    W.s[2] = 0;
    run(W, {0xBE801302u}, "s_ff1_i32_b32 0");
    eq(W.s[0], 0xFFFFFFFF, "s_ff1 of 0");
    W.s[2] = 0xFFFF0000;
    run(W, {0xBE801702u}, "s_flbit_i32");
    eq(W.s[0], 16, "s_flbit_i32 negative");
    W.s[2] = 0xFFFFFFFF;
    run(W, {0xBE801702u}, "s_flbit_i32 -1");
    eq(W.s[0], 0xFFFFFFFF, "s_flbit_i32 of -1");
    W.s[2] = 0x00000001;
    run(W, {0xBE800B02u}, "s_brev_b32");
    eq(W.s[0], 0x80000000, "s_brev_b32");
    W.s[2] = 0xFFFFFFFB;
    run(W, {0xBE803402u}, "s_abs_i32");
    eq(W.s[0], 5, "s_abs_i32");
    W.s[0] = 0, W.s[1] = 0;
    run(W, {0xBE801EA8u}, "s_bitset1_b64");
    eq(W.s[1], 1u << 8, "s_bitset1_b64");
    // SOPC.
    W.s[2] = 0xFFFFFFFF, W.s[3] = 1;
    run(W, {0xBF040302u}, "s_cmp_lt_i32");
    check(W.scc, "s_cmp_lt_i32 -1 < 1");
    // SOPK.
    run(W, {0xB0008000u}, "s_movk_i32");
    eq(W.s[0], 0xFFFF8000, "s_movk sign-extends");
    run(W, {0xB7807FFFu}, "s_addk_i32");
    eq(W.s[0], 0xFFFFFFFF, "s_addk");
    check(!W.scc, "s_addk no overflow");
    run(W, {0xB6808000u}, "s_cmpk_lt_u32");
    check(!W.scc, "s_cmpk_lt_u32 zero-extends");
    W.s[0] = 3;
    run(W, {0xB800FFFEu}, "s_mulk_i32");
    eq(W.s[0], 0xFFFFFFFA, "s_mulk");
    // 64-bit move of a float inline constant gives the f64 pattern.
    run(W, {0xBE8004F2u}, "s_mov_b64 1.0");
    eq((uint64_t(W.s[1]) << 32) | W.s[0], 0x3FF0000000000000ull, "s_mov_b64 1.0");
    // Unsupported op.
    vpgcn::Instruction in;
    std::string err;
    check(vpgcn::decode(std::vector<uint32_t>{0xBF840001u}, in) && !vpgcnref::step(in, W, err) &&
              err == "unsupported s_cbranch_scc0",
          "unsupported message: " + err);
}

void vint() {
    auto w = fresh();
    Wave& W = *w;
    W.exec = 0x7;  // lanes 0..2
    W.v[1][0] = 0xFFFFFFFF, W.v[2][0] = 1;
    W.v[1][1] = 1, W.v[2][1] = 1;
    W.v[1][2] = 0x80000000, W.v[2][2] = 0x80000000;
    W.v[0][5] = 0xDEAD;
    W.vcc = ~0ull;
    run(W, {0x4A000501u}, "v_add_i32");
    eq(W.v[0][0], 0, "v_add_i32 l0");
    eq(W.v[0][1], 2, "v_add_i32 l1");
    eq(W.v[0][2], 0, "v_add_i32 l2");
    eq(W.v[0][5], 0xDEAD, "v_add_i32 inactive lane kept");
    eq(W.vcc, 0x5, "v_add_i32 carry mask (inactive bits 0)");
    // v_addc_u32 v3, vcc, v4, v5, vcc
    W.v[4][0] = 0xFFFFFFFF, W.v[5][0] = 0;
    W.v[4][1] = 7, W.v[5][1] = 8;
    W.v[4][2] = 0xFFFFFFFE, W.v[5][2] = 1;
    run(W, {0x50060B04u}, "v_addc_u32");
    eq(W.v[3][0], 0, "v_addc l0");
    eq(W.v[3][1], 15, "v_addc l1");
    eq(W.v[3][2], 0, "v_addc l2");
    eq(W.vcc, 0x5, "v_addc carry");
    // v_sub_i32_e64 v0, s[4:5], v1, v2: borrow into an SGPR pair.
    W.v[1][0] = 1, W.v[2][0] = 2;
    W.v[1][1] = 2, W.v[2][1] = 1;
    W.v[1][2] = 0, W.v[2][2] = 0;
    run(W, {0xD24C0400u, 0x00020501u}, "v_sub_i32_e64");
    eq(W.v[0][0], 0xFFFFFFFF, "v_sub l0");
    eq((uint64_t(W.s[5]) << 32) | W.s[4], 1, "v_sub borrow mask");
    // v_subb_u32 v0, vcc, v1, v2, vcc
    W.vcc = 0x2;
    W.v[1][1] = 5, W.v[2][1] = 5;
    run(W, {0x52000501u}, "v_subb_u32");
    eq(W.v[0][1], 0xFFFFFFFF, "v_subb l1");
    eq(W.vcc, 0x3, "v_subb borrow");
    // v_bfe_u32 / i32 with offset + width > 32.
    W.v[1][0] = 0xF0000000, W.v[2][0] = 28, W.v[3][0] = 8;
    W.v[1][1] = 0x12345678, W.v[2][1] = 4, W.v[3][1] = 8;
    W.v[1][2] = 0x00000780, W.v[2][2] = 7 + 32, W.v[3][2] = 4 + 32;  // only [4:0] used
    run(W, {0xD2900000u, 0x040E0501u}, "v_bfe_u32");
    eq(W.v[0][0], 0xF, "v_bfe_u32 l0");
    eq(W.v[0][1], 0x67, "v_bfe_u32 l1");
    eq(W.v[0][2], 0xF, "v_bfe_u32 l2");
    run(W, {0xD2920000u, 0x040E0501u}, "v_bfe_i32");
    eq(W.v[0][0], 0xFFFFFFFF, "v_bfe_i32 l0");
    eq(W.v[0][1], 0x67, "v_bfe_i32 l1");
    eq(W.v[0][2], 0xFFFFFFFF, "v_bfe_i32 l2");
    // v_alignbit_b32 v0, v1, v2, v3
    W.v[1][0] = 0x000000AB, W.v[2][0] = 0xCD000000, W.v[3][0] = 24;
    run(W, {0xD29C0000u, 0x040E0501u}, "v_alignbit");
    eq(W.v[0][0], 0x0000ABCD, "v_alignbit");
    // v_mad_u64_u32 v[0:1], s[4:5], v1, v2, v[4:5]
    W.v[1][0] = 0xFFFFFFFF, W.v[2][0] = 0xFFFFFFFF, W.v[4][0] = 0xFFFFFFFF, W.v[5][0] = 0xFFFFFFFF;
    W.v[1][1] = 3, W.v[2][1] = 4, W.v[4][1] = 1, W.v[5][1] = 1;
    W.v[1][2] = 0, W.v[2][2] = 0, W.v[4][2] = 0, W.v[5][2] = 0;
    run(W, {0xD2EC0400u, 0x04120501u}, "v_mad_u64_u32");
    eq((uint64_t(W.v[1][0]) << 32) | W.v[0][0], 0xFFFFFFFE00000000ull, "v_mad_u64_u32 l0");
    eq((uint64_t(W.v[1][1]) << 32) | W.v[0][1], 0x10000000Dull, "v_mad_u64_u32 l1");
    eq((uint64_t(W.s[5]) << 32) | W.s[4], 1, "v_mad_u64_u32 carry");
    // v_mul_hi_i32
    W.v[1][0] = 0xFFFFFFFF, W.v[2][0] = 2;
    run(W, {0xD2D80000u, 0x00020501u}, "v_mul_hi_i32");
    eq(W.v[0][0], 0xFFFFFFFF, "v_mul_hi_i32 -1*2");
    // v_ffbh_i32
    W.v[1][0] = 0x00010000, W.v[1][1] = 0xFFFFFFFF, W.v[1][2] = 0xC0000000;
    run(W, {0x7E007701u}, "v_ffbh_i32");
    eq(W.v[0][0], 15, "v_ffbh_i32 pos");
    eq(W.v[0][1], 0xFFFFFFFF, "v_ffbh_i32 -1");
    eq(W.v[0][2], 2, "v_ffbh_i32 neg");
    // v_ashrrev_i32 v0, v1, v2: shift amount is src0.
    W.v[1][0] = 4, W.v[2][0] = 0x80000000;
    run(W, {0x30000501u}, "v_ashrrev");
    eq(W.v[0][0], 0xF8000000, "v_ashrrev");
    // v_lerp_u8 / v_sad_u32
    W.v[1][0] = 0x00FF0102, W.v[2][0] = 0x00FF0304, W.v[3][0] = 0x00000101;
    run(W, {0xD29A0000u, 0x040E0501u}, "v_lerp_u8");
    eq(W.v[0][0], 0x00FF0203, "v_lerp_u8");
    W.v[1][0] = 3, W.v[2][0] = 10, W.v[3][0] = 100;
    run(W, {0xD2BA0000u, 0x040E0501u}, "v_sad_u32");
    eq(W.v[0][0], 107, "v_sad_u32");
    // v_cvt_pk_i16_i32 saturation.
    W.v[1][0] = 0xFFFF0000, W.v[2][0] = 40000;
    run(W, {0x62000501u}, "v_cvt_pk_i16_i32");
    eq(W.v[0][0], 0x7FFF8000, "v_cvt_pk_i16_i32");
    // v_readfirstlane_b32 s0, v1: first active lane.
    W.exec = 0x6, W.v[1][1] = 0x1111, W.v[1][2] = 0x2222;
    run(W, {0x7E000501u}, "v_readfirstlane");
    eq(W.s[0], 0x1111, "v_readfirstlane");
}

void vcmp() {
    auto w = fresh();
    Wave& W = *w;
    const float nan = std::nanf("");
    W.exec = 0xF;
    float a[4] = {1.0f, 2.0f, nan, 1.0f};
    float b[4] = {2.0f, 1.0f, 1.0f, 1.0f};
    for (int l = 0; l < 4; ++l) W.v[1][l] = F(a[l]), W.v[2][l] = F(b[l]);
    W.v[1][40] = F(1.0f), W.v[2][40] = F(2.0f);  // inactive lane: must stay 0
    W.vcc = ~0ull;
    run(W, {0x7C020501u}, "v_cmp_lt_f32");
    eq(W.vcc, 0x1, "v_cmp_lt_f32 (NaN false, inactive 0)");
    run(W, {0xD01A0004u, 0x00020501u}, "v_cmp_neq_f32_e64");
    eq((uint64_t(W.s[5]) << 32) | W.s[4], 0x7, "v_cmp_neq_f32 (NaN true)");
    run(W, {0x7C100501u}, "v_cmp_u_f32");
    eq(W.vcc, 0x4, "v_cmp_u_f32");
    run(W, {0x7C140501u}, "v_cmp_nlg_f32");
    eq(W.vcc, 0xC, "v_cmp_nlg_f32");
    // v_cmpx_gt_u32: writes vcc and exec.
    for (int l = 0; l < 4; ++l) W.v[1][l] = uint32_t(l), W.v[2][l] = 1;
    run(W, {0x7DA80501u}, "v_cmpx_gt_u32");
    eq(W.vcc, 0xC, "v_cmpx_gt_u32 vcc");
    eq(W.exec, 0xC, "v_cmpx_gt_u32 exec");
    // v_cmpx_eq_u64_e64 s[4:5], v[2:3], -1 (constant sign-extended to 64 bits).
    W.exec = 0x3;
    W.v[2][0] = 0xFFFFFFFF, W.v[3][0] = 0xFFFFFFFF;
    W.v[2][1] = 0xFFFFFFFF, W.v[3][1] = 0;
    run(W, {0xD1E40004u, 0x00018302u}, "v_cmpx_eq_u64");
    eq((uint64_t(W.s[5]) << 32) | W.s[4], 0x1, "v_cmpx_eq_u64 sdst");
    eq(W.exec, 0x1, "v_cmpx_eq_u64 exec");
    // v_cmp_lt_i64 vcc, v[2:3], v[4:5]
    W.exec = 0x3;
    W.v[2][0] = 0, W.v[3][0] = 0xFFFFFFFF, W.v[4][0] = 0, W.v[5][0] = 0;  // -2^32 < 0
    W.v[2][1] = 0, W.v[3][1] = 1, W.v[4][1] = 0xFFFFFFFF, W.v[5][1] = 0;
    run(W, {0x7D420902u}, "v_cmp_lt_i64");
    eq(W.vcc, 0x1, "v_cmp_lt_i64");
    // v_cmp_class_f32 vcc, v1, v2
    W.exec = 0xF;
    W.v[1][0] = 0x7F800001, W.v[2][0] = 1u << 0;  // sNaN
    W.v[1][1] = 0x7FC00000, W.v[2][1] = 1u << 1;  // qNaN
    W.v[1][2] = 0x80000001, W.v[2][2] = 1u << 4;  // -denormal
    W.v[1][3] = 0x00000000, W.v[2][3] = 1u << 5;  // +0 asked as -0: false
    run(W, {0x7D100501u}, "v_cmp_class_f32");
    eq(W.vcc, 0x7, "v_cmp_class_f32");
    // v_cndmask_b32 v0, v1, v2, vcc: vcc set -> src1.
    W.vcc = 0x5;
    for (int l = 0; l < 4; ++l) W.v[1][l] = 10, W.v[2][l] = 20;
    run(W, {0x00000501u}, "v_cndmask");
    eq(W.v[0][0], 20, "v_cndmask l0");
    eq(W.v[0][1], 10, "v_cndmask l1");
    eq(W.v[0][2], 20, "v_cndmask l2");
}

void vfloat() {
    auto w = fresh();
    Wave& W = *w;
    const float nan = std::nanf(""), inf = INFINITY;
    W.exec = 0xF;
    // v_cvt_u32_f32 / v_cvt_i32_f32 saturation.
    float c[4] = {-1.0f, 5e9f, nan, 3.7f};
    for (int l = 0; l < 4; ++l) W.v[1][l] = F(c[l]);
    run(W, {0x7E000F01u}, "v_cvt_u32_f32");
    eq(W.v[0][0], 0, "cvt_u32 -1");
    eq(W.v[0][1], 0xFFFFFFFF, "cvt_u32 5e9");
    eq(W.v[0][2], 0, "cvt_u32 NaN");
    eq(W.v[0][3], 3, "cvt_u32 3.7");
    W.v[1][0] = F(-3e9f), W.v[1][1] = F(3e9f), W.v[1][3] = F(-3.7f);
    run(W, {0x7E001101u}, "v_cvt_i32_f32");
    eq(W.v[0][0], 0x80000000, "cvt_i32 -3e9");
    eq(W.v[0][1], 0x7FFFFFFF, "cvt_i32 3e9");
    eq(W.v[0][2], 0, "cvt_i32 NaN");
    eq(W.v[0][3], 0xFFFFFFFD, "cvt_i32 -3.7");
    // Cube: lane 0 (1, 2, -3) -z major; lane 1 (-4, 1, 2) -x major; lane 2 (0, 5, 1) +y major.
    float cx[3] = {1, -4, 0}, cy[3] = {2, 1, 5}, cz[3] = {-3, 2, 1};
    W.exec = 0x7;
    for (int l = 0; l < 3; ++l) W.v[1][l] = F(cx[l]), W.v[2][l] = F(cy[l]), W.v[3][l] = F(cz[l]);
    run(W, {0xD2880000u, 0x040E0501u}, "v_cubeid");
    eq(W.v[0][0], F(5), "cubeid -z");
    eq(W.v[0][1], F(1), "cubeid -x");
    eq(W.v[0][2], F(2), "cubeid +y");
    run(W, {0xD28A0000u, 0x040E0501u}, "v_cubesc");
    eq(W.v[0][0], F(-1), "cubesc -z");
    eq(W.v[0][1], F(2), "cubesc -x");
    eq(W.v[0][2], F(0), "cubesc +y");
    run(W, {0xD28C0000u, 0x040E0501u}, "v_cubetc");
    eq(W.v[0][0], F(-2), "cubetc -z");
    eq(W.v[0][1], F(-1), "cubetc -x");
    eq(W.v[0][2], F(1), "cubetc +y");
    run(W, {0xD28E0000u, 0x040E0501u}, "v_cubema");
    eq(W.v[0][0], F(-6), "cubema -z");
    eq(W.v[0][1], F(-8), "cubema -x");
    eq(W.v[0][2], F(10), "cubema +y");
    // v_mad_f32 v0, -v1, |v2|, v3 clamp
    W.v[1][0] = F(2), W.v[2][0] = F(-3), W.v[3][0] = F(10);   // -6 + 10 = 4 -> 1
    W.v[1][1] = F(2), W.v[2][1] = F(-3), W.v[3][1] = F(6.5f); // 0.5
    W.v[1][2] = F(inf), W.v[2][2] = F(0), W.v[3][2] = F(0);   // NaN -> 0
    run(W, {0xD2820A00u, 0x240E0501u}, "v_mad_f32");
    eq(W.v[0][0], F(1), "v_mad clamp high");
    eq(W.v[0][1], F(0.5f), "v_mad modifiers");
    eq(W.v[0][2], F(0), "v_mad clamp NaN");
    // v_mul_f32_e64 mul:2
    W.v[1][0] = F(3), W.v[2][0] = F(0.25f);
    run(W, {0xD2100000u, 0x08020501u}, "v_mul_f32 omod");
    eq(W.v[0][0], F(1.5f), "v_mul omod *2");
    // v_min_f32 with NaN returns the other operand; -0 < +0.
    W.v[1][0] = F(nan), W.v[2][0] = F(3);
    W.v[1][1] = F(-0.0f), W.v[2][1] = F(0.0f);
    W.v[1][2] = F(0.0f), W.v[2][2] = F(-0.0f);
    run(W, {0x1E000501u}, "v_min_f32");
    eq(W.v[0][0], F(3), "v_min NaN");
    eq(W.v[0][1], F(-0.0f), "v_min -0");
    eq(W.v[0][2], F(-0.0f), "v_min -0 b");
    // v_med3_f32
    W.v[1][0] = F(5), W.v[2][0] = F(1), W.v[3][0] = F(3);
    W.v[1][1] = F(nan), W.v[2][1] = F(7), W.v[3][1] = F(2);
    run(W, {0xD2AE0000u, 0x040E0501u}, "v_med3_f32");
    eq(W.v[0][0], F(3), "v_med3");
    eq(W.v[0][1], F(2), "v_med3 NaN -> min3");
    // v_mul_legacy_f32: 0 * inf = 0.
    W.v[1][0] = F(0), W.v[2][0] = F(inf);
    run(W, {0x0E000501u}, "v_mul_legacy");
    eq(W.v[0][0], F(0), "v_mul_legacy 0*inf");
    // v_mac_f32 v0, v1, v2
    W.v[0][0] = F(1), W.v[1][0] = F(2), W.v[2][0] = F(3);
    run(W, {0x3E000501u}, "v_mac_f32");
    eq(W.v[0][0], F(7), "v_mac");
    // v_madak_f32 v0, v1, v2, 10.0
    W.v[1][0] = F(2), W.v[2][0] = F(3);
    run(W, {0x42000501u, 0x41200000u}, "v_madak");
    eq(W.v[0][0], F(16), "v_madak");
    // v_subrev_f32 v0, 1.0, v1 = v1 - 1.0
    W.v[1][0] = F(5);
    run(W, {0x0A0002F2u}, "v_subrev_f32");
    eq(W.v[0][0], F(4), "v_subrev");
    // v_fract_f32
    W.v[1][0] = F(-0.25f), W.v[1][1] = F(-1e-30f), W.v[1][2] = F(inf);
    run(W, {0x7E004101u}, "v_fract");
    eq(W.v[0][0], F(0.75f), "fract -0.25");
    eq(W.v[0][1], 0x3F7FFFFF, "fract tiny negative clamps below 1");
    check(std::isnan(AsF(W.v[0][2])), "fract inf NaN");
    // v_frexp_mant / exp
    W.v[1][0] = F(12.0f);
    run(W, {0x7E008101u}, "v_frexp_mant");
    eq(W.v[0][0], F(0.75f), "frexp_mant");
    run(W, {0x7E007F01u}, "v_frexp_exp");
    eq(W.v[0][0], 4, "frexp_exp");
    // v_ldexp_f32
    W.v[1][0] = F(3), W.v[2][0] = 0xFFFFFFFE;
    run(W, {0x56000501u}, "v_ldexp");
    eq(W.v[0][0], F(0.75f), "ldexp");
    // v_cvt_pkrtz_f16_f32: round toward zero.
    W.v[1][0] = F(1.0f), W.v[2][0] = 0x3F801FFF;  // 1 + just under one f16 ulp
    W.v[1][1] = F(1e6f), W.v[2][1] = F(-0.0f);
    run(W, {0x5E000501u}, "v_cvt_pkrtz");
    eq(W.v[0][0], 0x3C003C00, "pkrtz");
    eq(W.v[0][1], 0x80007BFF, "pkrtz overflow -> max");
    // v_cvt_f16_f32 (nearest even) and back.
    W.v[1][0] = 0x3F801000;  // exactly halfway: ties to even (1.0)
    W.v[1][1] = 0x3F803000;  // halfway, odd -> up
    W.v[1][2] = F(1e6f);
    run(W, {0x7E001501u}, "v_cvt_f16_f32");
    eq(W.v[0][0], 0x3C00, "f16 tie even");
    eq(W.v[0][1], 0x3C02, "f16 tie up");
    eq(W.v[0][2], 0x7C00, "f16 overflow inf");
    W.v[1][0] = 0x0001, W.v[1][1] = 0xC000, W.v[1][2] = 0x7C00;
    run(W, {0x7E001701u}, "v_cvt_f32_f16");
    eq(W.v[0][0], F(0x1p-24f), "f16 denormal");
    eq(W.v[0][1], F(-2.0f), "f16 -2");
    eq(W.v[0][2], F(inf), "f16 inf");
    // v_cvt_pknorm_u16_f32
    W.v[1][0] = F(1.0f), W.v[2][0] = F(-1.0f);
    W.v[1][1] = F(0.5f), W.v[2][1] = F(2.0f);
    run(W, {0x5C000501u}, "v_cvt_pknorm_u16");
    eq(W.v[0][0], 0x0000FFFF, "pknorm_u16 a");
    eq(W.v[0][1], 0xFFFF8000, "pknorm_u16 b");
    // v_cvt_off_f32_i4
    W.v[1][0] = 0x8, W.v[1][1] = 0x7;
    run(W, {0x7E001D01u}, "v_cvt_off_f32_i4");
    eq(W.v[0][0], F(-0.5f), "off_i4 -8");
    eq(W.v[0][1], F(0.4375f), "off_i4 7");
    // v_cvt_f32_ubyte2
    W.v[1][0] = 0x00AB0000;
    run(W, {0x7E002701u}, "v_cvt_f32_ubyte2");
    eq(W.v[0][0], F(171.0f), "ubyte2");
    // v_max_legacy_f32 / v_mov_b32_e64 -|v1|
    W.v[1][0] = F(2.0f);
    run(W, {0xD3020100u, 0x20000101u}, "v_mov_b32_e64 -|v1|");
    eq(W.v[0][0], F(-2.0f), "v_mov neg abs");
}

}  // namespace

int main() {
    salu();
    vint();
    vcmp();
    vfloat();
    if (g_failures) {
        std::fprintf(stderr, "%u failure(s)\n", g_failures);
        return 1;
    }
    std::printf("gcn_ref selftest: OK\n");
    return 0;
}
