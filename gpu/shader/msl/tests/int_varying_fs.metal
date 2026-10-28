#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpIn {
    int4 id [[user(loc0), flat]];
    int4 pid [[user(loc1), flat]];
};

struct VpOut {
    float4 color [[color(0)]];
};

fragment VpOut vp_main(VpIn vp_in [[stage_in]]) {
    VpOut vp_out;
    const int v3 = vp_in.id.xy.y + vp_in.pid.x;
    const float v4 = float(v3);
    const float4 v5 = float4(v4);
    vp_out.color = v5;
    return vp_out;
}
