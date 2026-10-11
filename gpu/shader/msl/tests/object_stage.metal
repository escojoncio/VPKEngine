#include <metal_stdlib>
using namespace metal;

struct VpArgs {
    constant uchar* params;
};

struct VpPayload {
    uint words[4];
};

[[object]] void vp_main(object_data VpPayload& vp_payload [[payload]],
        mesh_grid_properties vp_grid,
        constant VpArgs& vp_args [[buffer(0)]],
        uint3 in_group [[threadgroup_position_in_grid]]) {
    const float4 v3 = *(constant float4*)(vp_args.params);
    const float v4 = float(in_group.x);
    const float v6 = v4 * v3.x;
    const uint v7 = as_type<uint>(v6);
    *(object_data uint*)((object_data uchar*)&vp_payload) = v7;
    const uint v10 = as_type<uint>(v3.y);
    *(object_data uint*)((object_data uchar*)&vp_payload + 4u) = v10;
    vp_grid.set_threadgroups_per_grid(uint3(1u, 1u, 1u));
}
