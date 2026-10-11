#include <metal_stdlib>
using namespace metal;

struct VpIn {
    float4 pos [[attribute(0)]];
    float2 uv [[attribute(1)]];
};

struct VpOut {
    float4 position [[position]];
    float2 tex_uv [[user(loc0)]];
    uint id [[user(loc1)]];
};

vertex VpOut vp_main(VpIn vp_in [[stage_in]],
        uint in_vid [[vertex_id]]) {
    VpOut vp_out;
    vp_out.position = vp_in.pos;
    vp_out.tex_uv = vp_in.uv;
    vp_out.id = in_vid;
    return vp_out;
}
