#include <metal_stdlib>
using namespace metal;

struct VpIn {
    float4 a [[user(loc0)]];
};

struct VpOut {
    float4 color [[color(0)]];
};

fragment VpOut vp_main(VpIn vp_in [[stage_in]]) {
    VpOut vp_out;
    const float v3 = quad_shuffle(vp_in.a.x, ushort(1u));
    const float v5 = quad_shuffle(vp_in.a.x, ushort(2u));
    const float v6 = v3 - vp_in.a.x;
    const float v7 = v5 - vp_in.a.x;
    const bool v9 = vp_in.a.x > 0.5f;
    const bool v10 = quad_any(v9);
    const float v13 = v10 ? 1.0f : 0.0f;
    const float4 v14 = float4(v6, v7, v13, 1.0f);
    vp_out.color = v14;
    return vp_out;
}
