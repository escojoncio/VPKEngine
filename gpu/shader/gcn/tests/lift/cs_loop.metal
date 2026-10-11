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
    uint l16{};
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
    l14 = 0u;
    while (true) {
        const uint v25 = l14;
        const bool v26 = v25 == 0u;
        if (v26) {
            const uint v30 = l11;
            const uint v31 = v30 << 2u;
            l12 = v31;
            const uint v32 = l12;
            const uint v33 = l11;
            *(threadgroup uint*)((threadgroup uchar*)vp_shared + v32) = v33;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            l15 = 0u;
            l13 = 0u;
            l14 = 1u;
            continue;
        }
        const uint v40 = l14;
        const bool v41 = v40 == 1u;
        if (v41) {
            const uint v42 = l12;
            const uint v45 = *(threadgroup uint*)((threadgroup uchar*)vp_shared + v42);
            l16 = v45;
            const uint v46 = l13;
            const uint v47 = l16;
            const uint v48 = v46 + v47;
            const bool v49 = v48 < v46;
            const bool v50 = l0;
            const bool v51 = v49 && v50;
            l1 = v51;
            l13 = v48;
            const uint v52 = l15;
            const uint v54 = v52 + 1u;
            const bool v55 = v54 < v52;
            l2 = v55;
            l15 = v54;
            const uint v57 = l15;
            const bool v58 = v57 < 4u;
            l2 = v58;
            const bool v61 = l2;
            const uint v62 = v61 ? 1u : 2u;
            l14 = v62;
            continue;
        }
        const uint v64 = l14;
        const bool v65 = v64 == 2u;
        if (v65) {
            const uint v69 = l12;
            const uint v71 = l13;
            *(device uint*)(vp_args.buf0 + v69) = v71;
            return;
        }
        break;
    }
}
