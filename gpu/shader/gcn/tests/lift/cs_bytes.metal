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
    threadgroup uint4 vp_shared[4];
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
    const uint v25 = l11;
    const uint v26 = l11;
    const uint v30 = v25 & 4294967292u;
    const uint v33 = v25 & 3u;
    const uint v34 = v33 << 3u;
    const uint v36 = 255u << v34;
    const uint v38 = v26 & 255u;
    const uint v39 = v38 << v34;
    const uint v40 = ~v36;
    atomic_fetch_and_explicit((threadgroup atomic_uint*)((threadgroup uchar*)vp_shared + v30), v40, memory_order_relaxed);
    atomic_fetch_or_explicit((threadgroup atomic_uint*)((threadgroup uchar*)vp_shared + v30), v39, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint v43 = l11;
    const uint v45 = 4294967292u & v43;
    l12 = v45;
    const uint v46 = l12;
    const uint v49 = *(threadgroup uint*)((threadgroup uchar*)vp_shared + v46);
    l13 = v49;
    const uint v53 = l11;
    const uint v55 = l11;
    const uint v57 = v53 & 4294967292u;
    const uint v60 = v53 & 3u;
    const uint v61 = v60 << 3u;
    const uint v63 = 255u << v61;
    const uint v65 = v55 & 255u;
    const uint v66 = v65 << v61;
    const uint v67 = ~v63;
    atomic_fetch_and_explicit((device atomic_uint*)(vp_args.buf0 + v57), v67, memory_order_relaxed);
    atomic_fetch_or_explicit((device atomic_uint*)(vp_args.buf0 + v57), v66, memory_order_relaxed);
    const uint v73 = l11;
    const uint v74 = v73 << 1u;
    l14 = v74;
    const uint v78 = l14;
    const uint v79 = 256u + v78;
    const uint v80 = l13;
    const uint v82 = v79 & 4294967292u;
    const uint v85 = v79 & 3u;
    const uint v86 = v85 << 3u;
    const uint v88 = 65535u << v86;
    const uint v90 = v80 & 65535u;
    const uint v91 = v90 << v86;
    const uint v92 = ~v88;
    atomic_fetch_and_explicit((device atomic_uint*)(vp_args.buf0 + v82), v92, memory_order_relaxed);
    atomic_fetch_or_explicit((device atomic_uint*)(vp_args.buf0 + v82), v91, memory_order_relaxed);
    return;
}
