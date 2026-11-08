#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpIn {
    float4 a [[user(loc0)]];
};

struct VpOut {
    float4 nan_safe [[color(0)]];
    float4 plain [[color(1)]];
};

fragment VpOut vp_main(VpIn vp_in [[stage_in]]) {
    VpOut vp_out;
    const float4 v1 = float4(0.5f, 0.25f, 2.0f, 1.0f);
    const float4 v2 = precise::fmax(vp_in.a, v1);
    const float4 v3 = precise::fmin(v2, v1);
    vp_out.nan_safe = v3;
    const float4 v4 = max(vp_in.a, v1);
    const float4 v5 = min(v4, v1);
    vp_out.plain = v5;
    return vp_out;
}
