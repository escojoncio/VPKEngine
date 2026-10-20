#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpIn {
    float4 pos [[attribute(0)]];
};

struct VpOut {
    float4 position [[position]];
    uint layer [[render_target_array_index]];
    uint vp_index_ [[viewport_array_index]];
};

vertex VpOut vp_main(VpIn vp_in [[stage_in]],
        uint in_vid [[vertex_id]]) {
    VpOut vp_out;
    vp_out.position = vp_in.pos;
    vp_out.layer = in_vid;
    vp_out.vp_index_ = in_vid;
    return vp_out;
}
