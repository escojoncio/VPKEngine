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
    float4 vp_p0_0{};
    float vp_p0_16{};
    uint4 vp_p1[2] = {};
    uint4 vp_p2[1] = {};
    vp_p0_0 = vp_in.a;
    vp_p0_0.w = vp_in.a.y;
    vp_p0_16 = vp_in.a.y;
    const float4 v5 = vp_p0_0;
    const float v6 = vp_p0_16;
    const uint v7 = as_type<uint>(vp_in.a.y);
    const uint v9 = v7 & 1u;
    const uint v10 = v9 * 16u;
    *(thread float4*)((thread uchar*)vp_p1 + v10) = vp_in.a;
    const float4 v11 = *(thread float4*)((thread uchar*)vp_p1);
    *(thread float*)((thread uchar*)vp_p2) = vp_in.a.y;
    const uint v12 = *(thread uint*)((thread uchar*)vp_p2);
    const float v13 = float(v12);
    const float4 v14 = v5 + v11;
    const float v15 = v6 + v13;
    float4 v16 = v14;
    v16.x = v15;
    vp_out.color = v16;
    return vp_out;
}
