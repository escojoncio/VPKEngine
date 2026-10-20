#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpOut {
    float4 color [[color(0)]];
};

fragment VpOut vp_main(float4 in_frag_pos [[position]],
        uint in_view [[render_target_array_index]],
        constant rasterization_rate_map_data& vp_rrm [[buffer(28)]]) {
    VpOut vp_out;
    rasterization_rate_map_decoder vp_rrmd(vp_rrm);
    float4 vp_pos = float4(vp_rrmd.map_physical_to_screen_coordinates(in_frag_pos.xy, in_view), in_frag_pos.zw);
    vp_out.color = vp_pos;
    return vp_out;
}
