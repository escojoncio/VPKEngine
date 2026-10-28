#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpIn {
    float4 pos [[attribute(0)]];
    float2 uv [[attribute(1)]];
};

struct VpOut {
    float4 position [[position]];
    float2 tex_uv [[user(loc0)]];
    uint4 id [[user(loc1)]];
};

vertex VpOut vp_main(VpIn vp_in [[stage_in]],
        uint in_vid [[vertex_id]]) {
    VpOut vp_out;
    vp_out.position = vp_in.pos;
    vp_out.tex_uv = vp_in.uv;
    vp_out.id = uint4(in_vid, 0u, 0u, 0u);
    return vp_out;
}
