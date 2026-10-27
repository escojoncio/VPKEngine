#include "vpir/f64.h"

#include <bit>
#include <cstring>

namespace vpir::f64 {
namespace impl {

using u32 = uint32_t;
using u64 = uint64_t;
using i32 = int32_t;
struct uint2 {
    u32 x, y;
};
struct uint3 {
    u32 x, y, z;
};
inline int clz64(u64 v) { return std::countl_zero(v); }
#define VP_CLZ64(x) ::vpir::f64::impl::clz64(x)
#define VP_TABLE static const
#define VP_NOUNROLL

#include "f64_table.inc"
#include "f64_core.inc"

}  // namespace impl

using namespace impl;

uint64_t add(uint64_t a, uint64_t b, uint32_t m) { return vpf_add(a, b, m); }
uint64_t mul(uint64_t a, uint64_t b, uint32_t m) { return vpf_mul(a, b, m); }
uint64_t fma(uint64_t a, uint64_t b, uint64_t c, uint32_t m) { return vpf_fma(a, b, c, m); }
uint64_t div(uint64_t a, uint64_t b, uint32_t m) { return vpf_div(a, b, m); }
uint64_t sqrt(uint64_t a, uint32_t m) { return vpf_sqrt(a, m); }
uint64_t rcp(uint64_t a, uint32_t m) { return vpf_rcp(a, m); }
uint64_t rsq(uint64_t a, uint32_t m) { return vpf_rsq(a, m); }
uint64_t ldexp(uint64_t a, int32_t k, uint32_t m) { return vpf_ldexp(a, uint32_t(k), m); }
uint64_t min(uint64_t a, uint64_t b, uint32_t m) { return vpf_min(a, b, m); }
uint64_t max(uint64_t a, uint64_t b, uint32_t m) { return vpf_max(a, b, m); }
bool cmp(uint64_t a, uint64_t b, uint32_t cond, uint32_t m) { return vpf_cmp(a, b, cond, m); }
bool classify(uint64_t a, uint32_t mask) { return vpf_class(a, mask); }
uint64_t trunc(uint64_t a, uint32_t m) { return vpf_trunc(a, m); }
uint64_t floor(uint64_t a, uint32_t m) { return vpf_floor(a, m); }
uint64_t ceil(uint64_t a, uint32_t m) { return vpf_ceil(a, m); }
uint64_t rndne(uint64_t a, uint32_t m) { return vpf_rndne(a, m); }
uint64_t fract(uint64_t a, uint32_t m) { return vpf_fract(a, m); }
uint64_t frexpMant(uint64_t a, uint32_t m) { return vpf_frexp_mant(a, m); }
int32_t frexpExp(uint64_t a, uint32_t m) { return int32_t(vpf_frexp_exp(a, m)); }
uint64_t fromI32(int32_t v, uint32_t m) { return vpf_from_i32(uint32_t(v), m); }
uint64_t fromU32(uint32_t v, uint32_t m) { return vpf_from_u32(v, m); }
uint64_t fromF32(uint32_t f) { return vpf_from_f32(f); }
uint32_t toF32(uint64_t a, uint32_t m) { return vpf_to_f32(a, m, m & 3u, (m & 0x20u) != 0); }
int32_t toI32(uint64_t a, uint32_t m) { return int32_t(vpf_to_i32(a, m)); }
uint32_t toU32(uint64_t a, uint32_t m) { return vpf_to_u32(a, m); }
uint64_t divScale(uint64_t s0, uint64_t s1, uint64_t s2, bool& vcc) {
    vpf_scale r = vpf_div_scale(s0, s1, s2);
    vcc = r.vcc;
    return r.d;
}
uint64_t divFmas(uint64_t a, uint64_t b, uint64_t c, bool vcc, uint32_t m) { return vpf_div_fmas(a, b, c, vcc, m); }
uint64_t divFixup(uint64_t q, uint64_t den, uint64_t num) { return vpf_div_fixup(q, den, num); }
uint64_t trigPreop(uint64_t a, uint32_t idx, uint32_t m) { return vpf_trig_preop(a, idx, m); }
uint64_t saturate(uint64_t a, uint32_t m) {
    uint2 r = vp_f64_sat(uint2{uint32_t(a), uint32_t(a >> 32)}, m);
    return uint64_t(r.x) | uint64_t(r.y) << 32;
}

void eval(F64Fn fn, const uint32_t* args, uint32_t* out) {
    const char* sig = f64FnInfo(fn).args;
    // Operand i as a uint2 / u32: 'd' operands take two dwords.
    uint2 d[5] = {};
    u32 u[5] = {};
    unsigned at = 0, di = 0, ui = 0;
    for (const char* p = sig; *p; ++p) {
        if (*p == 'd') {
            d[di].x = args[at++];
            d[di].y = args[at++];
            ++di;
        } else {
            u[ui++] = args[at++];
        }
    }
    auto put = [&](uint2 v) {
        out[0] = v.x;
        out[1] = v.y;
    };
    switch (fn) {
        case F64Fn::Add: return put(vp_f64_add(d[0], d[1], u[0]));
        case F64Fn::Mul: return put(vp_f64_mul(d[0], d[1], u[0]));
        case F64Fn::Fma: return put(vp_f64_fma(d[0], d[1], d[2], u[0]));
        case F64Fn::Min: return put(vp_f64_min(d[0], d[1], u[0]));
        case F64Fn::Max: return put(vp_f64_max(d[0], d[1], u[0]));
        case F64Fn::Ldexp: return put(vp_f64_ldexp(d[0], u[0], u[1]));
        case F64Fn::Rcp: return put(vp_f64_rcp(d[0], u[0]));
        case F64Fn::RcpClamp: return put(vp_f64_rcp_clamp(d[0], u[0]));
        case F64Fn::Rsq: return put(vp_f64_rsq(d[0], u[0]));
        case F64Fn::RsqClamp: return put(vp_f64_rsq_clamp(d[0], u[0]));
        case F64Fn::Sqrt: return put(vp_f64_sqrt(d[0], u[0]));
        case F64Fn::Trunc: return put(vp_f64_trunc(d[0], u[0]));
        case F64Fn::Ceil: return put(vp_f64_ceil(d[0], u[0]));
        case F64Fn::Rndne: return put(vp_f64_rndne(d[0], u[0]));
        case F64Fn::Floor: return put(vp_f64_floor(d[0], u[0]));
        case F64Fn::Fract: return put(vp_f64_fract(d[0], u[0]));
        case F64Fn::FrexpMant: return put(vp_f64_frexp_mant(d[0], u[0]));
        case F64Fn::FrexpExp: out[0] = vp_f64_frexp_exp(d[0], u[0]); return;
        case F64Fn::FromI32: return put(vp_f64_from_i32(u[0], u[1]));
        case F64Fn::FromU32: return put(vp_f64_from_u32(u[0], u[1]));
        case F64Fn::FromF32: return put(vp_f64_from_f32(u[0], u[1]));
        case F64Fn::ToF32: out[0] = vp_f64_to_f32(d[0], u[0]); return;
        case F64Fn::ToI32: out[0] = vp_f64_to_i32(d[0], u[0]); return;
        case F64Fn::ToU32: out[0] = vp_f64_to_u32(d[0], u[0]); return;
        case F64Fn::Cmp: out[0] = vp_f64_cmp(d[0], d[1], u[0], u[1]) ? 1u : 0u; return;
        case F64Fn::Class: out[0] = vp_f64_class(d[0], u[0], u[1]) ? 1u : 0u; return;
        case F64Fn::DivScale: {
            uint3 r = vp_f64_div_scale(d[0], d[1], d[2], u[0]);
            out[0] = r.x;
            out[1] = r.y;
            out[2] = r.z;
            return;
        }
        case F64Fn::DivFmas: return put(vp_f64_div_fmas(d[0], d[1], d[2], u[0], u[1]));
        case F64Fn::DivFixup: return put(vp_f64_div_fixup(d[0], d[1], d[2], u[0]));
        case F64Fn::TrigPreop: return put(vp_f64_trig_preop(d[0], u[0], u[1]));
        case F64Fn::Sat: return put(vp_f64_sat(d[0], u[0]));
        case F64Fn::Count: break;
    }
}

const char* mslSource() {
    static const char kSource[] = {
#include "f64_msl_data.inc"
        0};
    return kSource;
}

}  // namespace vpir::f64
