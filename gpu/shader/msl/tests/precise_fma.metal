#include <metal_stdlib>
using namespace metal;

struct VpIn {
    float4 a [[user(loc0)]];
};

struct VpOut {
    float4 exact [[color(0)]];
    float4 fast [[color(1)]];
};

fragment VpOut vp_main(VpIn vp_in [[stage_in]]) {
    VpOut vp_out;
    const float4 v1 = float4(0.5f, 0.25f, 2.0f, 1.0f);
    const float4 v2 = fma(vp_in.a, v1, v1);
    const float4 v3 = precise::sqrt(v2);
    const float4 v4 = precise::divide(v3, v1);
    vp_out.exact = v4;
    const float4 v5 = fma(vp_in.a, v1, v1);
    const float4 v6 = rsqrt(v5);
    const float4 v7 = 1.0f / v6;
    vp_out.fast = v7;
    return vp_out;
}
