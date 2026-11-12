#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpIn {
    float4 a [[user(loc0)]];
};

struct VpOut {
    float4 color [[color(0)]];
};

fragment VpOut vp_main(VpIn vp_in [[stage_in]]) {
    VpOut vp_out;
    const bool v5 = vp_in.a.x < 0.5f;
    const bool v6 = vp_in.a.y < 0.5f;
    const bool v7 = vp_in.a.z < 0.5f;
    const float4 v8 = vp_in.a + vp_in.a;
    const float4 v9 = vp_in.a * vp_in.a;
    if (v5) {
        vp_out.color = vp_in.a;
    } else if (v6) {
        vp_out.color = v8;
    } else if (v7) {
        vp_out.color = v9;
    }
    if (v5) {
        vp_out.color = vp_in.a;
    } else {
        const float4 v10 = vp_in.a - v9;
        if (v6) {
            vp_out.color = v10;
        } else if (v7) {
            vp_out.color = v8;
        } else {
            vp_out.color = v9;
        }
    }
    return vp_out;
}
