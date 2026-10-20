#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpOut {
    float vp_clip [[clip_distance]] [2];
    float4 position [[position]];
    float point_size [[VP_PSZ]];
    uint layer [[render_target_array_index]];
};

vertex VpOut vp_main(uint in_vertex_id [[vertex_id]]) {
    VpOut vp_out;
    bool l0{};
    bool l1{};
    bool l2{};
    uint l3{};
    uint l4{};
    uint l5{};
    uint l6{};
    uint l7{};
    uint l8{};
    uint l9{};
    l0 = true;
    l1 = false;
    l2 = false;
    l3 = 0u;
    l4 = in_vertex_id;
    vp_out.vp_clip[0] = 1.0f;
    vp_out.vp_clip[1] = 1.0f;
    l5 = 0u;
    l6 = 1073741824u;
    l7 = 1u;
    l8 = 1056964608u;
    l9 = 3204448256u;
    const uint v13 = l5;
    const float v14 = as_type<float>(v13);
    const uint v15 = l5;
    const float v16 = as_type<float>(v15);
    const uint v17 = l5;
    const float v18 = as_type<float>(v17);
    const uint v19 = l5;
    const float v20 = as_type<float>(v19);
    const float4 v21 = float4(v14, v16, v18, v20);
    vp_out.position = v21;
    const uint v22 = l6;
    const float v23 = as_type<float>(v22);
    const uint v25 = l7;
    const float v26 = as_type<float>(v25);
    vp_out.point_size = v23;
    const uint v29 = as_type<uint>(v26);
    vp_out.layer = v29;
    const uint v30 = l8;
    const float v31 = as_type<float>(v30);
    const uint v32 = l9;
    const float v33 = as_type<float>(v32);
    vp_out.vp_clip[0] = v31;
    vp_out.vp_clip[1] = v33;
    return vp_out;
}
