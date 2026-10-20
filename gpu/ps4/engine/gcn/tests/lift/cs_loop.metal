#include <metal_stdlib>
using namespace metal;

struct VpArgs {
    device uchar* buf0;
    constant uchar* user_data;
};

[[max_total_threads_per_threadgroup(64)]] kernel void vp_main(constant VpArgs& vp_args [[buffer(0)]],
        uint3 in_workgroup_id [[threadgroup_position_in_grid]],
        uint3 in_local_id [[thread_position_in_threadgroup]],
        uint in_local_index [[thread_index_in_threadgroup]]) {
    threadgroup uint4 vp_shared[16];
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
    uint l11{};
    uint l12{};
    uint l13{};
    uint l14{};
    uint l15{};
    l0 = true;
    l1 = false;
    l2 = false;
    l3 = 0u;
    l4 = (*(constant uint*)(vp_args.user_data));
    l5 = (*(constant uint*)(vp_args.user_data + 4u));
    l6 = (*(constant uint*)(vp_args.user_data + 8u));
    l7 = (*(constant uint*)(vp_args.user_data + 12u));
    l8 = in_workgroup_id.x;
    l9 = in_workgroup_id.y;
    l10 = in_workgroup_id.z;
    l11 = in_local_id.x;
    l12 = in_local_id.y;
    l13 = in_local_id.z;
    const uint v26 = l11;
    const uint v27 = v26 << 2u;
    l12 = v27;
    const uint v28 = l12;
    const uint v29 = l11;
    *(threadgroup uint*)((threadgroup uchar*)vp_shared + v28) = v29;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    l14 = 0u;
    l13 = 0u;
    while (true) {
        const uint v34 = l12;
        const uint v37 = *(threadgroup uint*)((threadgroup uchar*)vp_shared + v34);
        l15 = v37;
        const uint v38 = l13;
        const uint v39 = l15;
        const uint v40 = v38 + v39;
        const bool v41 = v40 < v38;
        const bool v42 = l0;
        const bool v43 = v41 && v42;
        l1 = v43;
        l13 = v40;
        const uint v44 = l14;
        const uint v46 = v44 + 1u;
        const bool v47 = v46 < v44;
        l2 = v47;
        l14 = v46;
        const uint v49 = l14;
        const bool v50 = v49 < 4u;
        l2 = v50;
        const bool v51 = l2;
        if (v51) {
            continue;
        }
        break;
        break;
    }
    const uint v55 = l12;
    const uint v57 = l13;
    *(device uint*)(vp_args.buf0 + v55) = v57;
    return;
}
