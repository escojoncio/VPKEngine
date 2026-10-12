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
    uint4 vp_private[2] = {};
    const uint v2 = as_type<uint>(vp_in.a.x);
    const uint v4 = v2 & 3u;
    const uint v6 = v4 * 4u;
    *(thread packed_float4*)((thread uchar*)vp_private + v6) = vp_in.a;
    *(thread uint*)((thread uchar*)vp_private + 16u) = v2;
    const float4 v8 = float4(*(thread packed_float4*)((thread uchar*)vp_private + v6));
    const uint v9 = *(thread uint*)((thread uchar*)vp_private + 16u);
    const float v10 = float(v9);
    float4 v11 = v8;
    v11.w = v10;
    vp_out.color = v11;
    return vp_out;
}
