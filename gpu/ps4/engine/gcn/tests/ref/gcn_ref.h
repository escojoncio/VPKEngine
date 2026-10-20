// Reference interpreter for GCN 2 (Sea Islands / gfx7) scalar and vector ALU instructions.
//
// Written independently of the shader translator, from the public AMD "Sea Islands Series
// Instruction Set Architecture" reference, to differential-test it. It executes one decoded
// instruction at a time on a full 64-lane wave.
//
// Modelling choices (where the manual leaves room or the hardware has modes):
//  - f32 arithmetic is IEEE single precision, round to nearest even, denormals KEPT on input and
//    output for every op (real v_mad_f32 / v_mac_f32 / v_mad_legacy flush denormals regardless of
//    the mode register: tests should avoid denormal operands for those).
//  - NaN payloads are not modelled: a harness must treat any two NaNs as equal.
//  - VOP3 input modifiers abs/neg are sign-bit operations (abs first, then neg) applied to the
//    float sources of float ops (and v_mov_b32 / v_cndmask_b32). Output modifiers omod (*2, *4,
//    /2) then clamp to [0, 1] apply to f32 results only; clamp sends NaN to +0 (DX10 clamp mode).
//    They are ignored on integer results.
//  - min/max f32 follow IEEE minNum/maxNum (a NaN operand yields the other one) and order
//    -0 < +0. med3 with any NaN returns min3 (manual).
//  - Carry-outs and compare masks are 64-bit; bits of inactive lanes are written as 0.
//  - Inline constants of 64-bit operands: integers are sign-extended, float constants give the
//    f64 pattern; 32-bit literals are zero-extended for integer 64-bit operands and give the high
//    dword for f64 operands.
#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "vpgcn/gcn.h"

namespace vpgcnref {

constexpr unsigned kLanes = 64;
constexpr unsigned kNumSgprs = 104;
constexpr unsigned kNumVgprs = 256;

struct Wave {
    std::array<uint32_t, kNumSgprs> s{};
    uint64_t vcc = 0;
    uint64_t exec = ~0ull;
    bool scc = false;
    uint32_t m0 = 0;
    // v[reg][lane]
    std::array<std::array<uint32_t, kLanes>, kNumVgprs> v{};

    uint32_t& vgpr(unsigned reg, unsigned lane) { return v[reg][lane]; }
    uint32_t vgpr(unsigned reg, unsigned lane) const { return v[reg][lane]; }
};

// Executes one SALU/VALU instruction. On an op outside the modelled set returns false with
// err = "unsupported <mnemonic>"; on an operand it cannot model (ttmp, lds_direct, ...) returns
// false with a description. The wave is unchanged on failure.
bool step(const vpgcn::Instruction& inst, Wave& wave, std::string& err);

}  // namespace vpgcnref
