#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpArgs {
    constant uchar* user_data;
};

[[max_total_threads_per_threadgroup(64)]] kernel void vp_main(constant VpArgs& vp_args [[buffer(0)]],
        uint3 in_workgroup_id [[threadgroup_position_in_grid]],
        uint3 in_local_id [[thread_position_in_threadgroup]],
        uint in_local_index [[thread_index_in_threadgroup]]) {
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
    uint l10{};
    l0 = true;
    l1 = false;
    l2 = false;
    l3 = 0u;
    l4 = (*(constant uint*)(vp_args.user_data));
    l5 = in_workgroup_id.x;
    l6 = in_workgroup_id.y;
    l7 = in_workgroup_id.z;
    l8 = in_local_id.x;
    l9 = in_local_id.y;
    l10 = in_local_id.z;
    const uint v20 = l4;
    const bool v21 = v20 == 0u;
    l2 = v21;
    const bool v22 = l2;
    if (v22) {
        return;
    } else {
        return;
    }
}
