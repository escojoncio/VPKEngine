// Software IEEE-754 binary64 with GCN (gfx7) semantics, the reference of the VPIR `f64` op.
//
// The same source (src/f64_core.inc) is compiled here as C++ and emitted as Metal Shading Language for
// shaders that contain `f64` ops (mslSource()), so the evaluator, the tests and the GPU agree bit for bit.
// A double is its bit pattern (uint64_t); `mode` is the GCN MODE register:
//   bits 3:2  round mode of double arithmetic (0 nearest even, 1 +inf, 2 -inf, 3 toward zero)
//   bit 6/7   FP_DENORM double: denormal inputs used / denormal results kept (clear = flushed to zero)
//   bit 8     DX10_CLAMP (the `clamp` modifier maps NaN to 0)   bit 9  IEEE (min / max quiet signalling NaN)
//   bits 1:0, 5  (to_f32 only) round mode and denormal-result flag of the single-precision destination
// The hardware reset value, kDefaultMode, rounds to nearest even and keeps denormals.
// See docs/isa-f64.md for what is bit-exact and what is approximated.
#pragma once

#include <cstdint>

#include "vpir/ir.h"

namespace vpir::f64 {

inline constexpr uint32_t kDefaultMode = 0x1c0;
inline constexpr uint64_t kDefaultNan = 0xfff8000000000000ull;  // the NaN invalid operations produce

uint64_t add(uint64_t a, uint64_t b, uint32_t mode = kDefaultMode);
uint64_t mul(uint64_t a, uint64_t b, uint32_t mode = kDefaultMode);
uint64_t fma(uint64_t a, uint64_t b, uint64_t c, uint32_t mode = kDefaultMode);
uint64_t div(uint64_t a, uint64_t b, uint32_t mode = kDefaultMode);
uint64_t sqrt(uint64_t a, uint32_t mode = kDefaultMode);
uint64_t rcp(uint64_t a, uint32_t mode = kDefaultMode);
uint64_t rsq(uint64_t a, uint32_t mode = kDefaultMode);
uint64_t ldexp(uint64_t a, int32_t k, uint32_t mode = kDefaultMode);
uint64_t min(uint64_t a, uint64_t b, uint32_t mode = kDefaultMode);
uint64_t max(uint64_t a, uint64_t b, uint32_t mode = kDefaultMode);
bool cmp(uint64_t a, uint64_t b, uint32_t cond, uint32_t mode = kDefaultMode);
bool classify(uint64_t a, uint32_t mask);
uint64_t trunc(uint64_t a, uint32_t mode = kDefaultMode);
uint64_t floor(uint64_t a, uint32_t mode = kDefaultMode);
uint64_t ceil(uint64_t a, uint32_t mode = kDefaultMode);
uint64_t rndne(uint64_t a, uint32_t mode = kDefaultMode);
uint64_t fract(uint64_t a, uint32_t mode = kDefaultMode);
uint64_t frexpMant(uint64_t a, uint32_t mode = kDefaultMode);
int32_t frexpExp(uint64_t a, uint32_t mode = kDefaultMode);
uint64_t fromI32(int32_t v, uint32_t mode = kDefaultMode);
uint64_t fromU32(uint32_t v, uint32_t mode = kDefaultMode);
uint64_t fromF32(uint32_t f32bits);
uint32_t toF32(uint64_t a, uint32_t mode = kDefaultMode);
int32_t toI32(uint64_t a, uint32_t mode = kDefaultMode);
uint32_t toU32(uint64_t a, uint32_t mode = kDefaultMode);
uint64_t divScale(uint64_t s0, uint64_t s1, uint64_t s2, bool& vcc);
uint64_t divFmas(uint64_t a, uint64_t b, uint64_t c, bool vcc, uint32_t mode = kDefaultMode);
uint64_t divFixup(uint64_t q, uint64_t den, uint64_t num);
uint64_t trigPreop(uint64_t a, uint32_t index, uint32_t mode = kDefaultMode);
uint64_t saturate(uint64_t a, uint32_t mode = kDefaultMode);

// Evaluates one `f64` op function. `args` holds the operand dwords in order (a 'd' operand is two
// dwords, low first; the others one); `out` receives the result (d: 2 dwords, c: 3, u / b: 1).
void eval(F64Fn fn, const uint32_t* args, uint32_t* out);

// The MSL source of the library: a prelude and the code of f64_core.inc, defining vp_f64_<name>() for
// every F64Fn (uint2 for a double, uint for u32 operands, bool for b results, uint3 for div_scale).
const char* mslSource();

}  // namespace vpir::f64
