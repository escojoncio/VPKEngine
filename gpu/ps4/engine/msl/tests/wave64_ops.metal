#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpArgs {
    device uchar* out;
};

[[max_total_threads_per_threadgroup(64)]] kernel void vp_main(constant VpArgs& vp_args [[buffer(0)]],
        uint in_lid [[thread_index_in_threadgroup]]) {
    threadgroup uint4 vp_shared[65];
    const uint v30 = in_lid & 63u;
    const uint v32 = v30 << 4u;
    const uint v36 = v30 >> 5u;
    const uint v2 = in_lid & 3u;
    const bool v4 = v2 == 1u;
    const uint4 v37 = uint4(as_type<uint2>(simd_vote::vote_t(simd_ballot(v4))), 0u, 0u);
    const uint v41 = v36 << 2u;
    const uint v42 = v41 + 1024u;
    *(threadgroup uint*)((threadgroup uchar*)vp_shared + v42) = v37.x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint v44 = *(threadgroup uint*)((threadgroup uchar*)vp_shared + 1024u);
    const uint v46 = *(threadgroup uint*)((threadgroup uchar*)vp_shared + 1028u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint4 v5 = uint4(v44, v46, 0u, 0u);
    const uint4 v49 = uint4(as_type<uint2>(simd_vote::vote_t(simd_ballot(v4))), 0u, 0u);
    *(threadgroup uint*)((threadgroup uchar*)vp_shared + v42) = v49.x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint v56 = *(threadgroup uint*)((threadgroup uchar*)vp_shared + 1024u);
    const uint v58 = *(threadgroup uint*)((threadgroup uchar*)vp_shared + 1028u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint v59 = v56 | v58;
    const bool v6 = v59 != 0u;
    const uint4 v62 = uint4(as_type<uint2>(simd_vote::vote_t(simd_ballot(v4))), 0u, 0u);
    *(threadgroup uint*)((threadgroup uchar*)vp_shared + v42) = v62.x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint v69 = *(threadgroup uint*)((threadgroup uchar*)vp_shared + 1024u);
    const uint v71 = *(threadgroup uint*)((threadgroup uchar*)vp_shared + 1028u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool v73 = v69 == 4294967295u;
    const bool v74 = v71 == 4294967295u;
    const bool v7 = v73 && v74;
    const uint v9 = in_lid * 7u;
    const uint v10 = v9 + 3u;
    *(threadgroup uint*)((threadgroup uchar*)vp_shared + v32) = v10;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint v12 = *(threadgroup uint*)((threadgroup uchar*)vp_shared + 80u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint v14 = 63u - in_lid;
    *(threadgroup uint*)((threadgroup uchar*)vp_shared + v32) = v10;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint v84 = v14 & 63u;
    const uint v87 = v84 << 4u;
    const uint v15 = *(threadgroup uint*)((threadgroup uchar*)vp_shared + v87);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *(threadgroup uint*)((threadgroup uchar*)vp_shared + v32) = v10;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint v16 = *(threadgroup uint*)((threadgroup uchar*)vp_shared);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint v18 = v6 ? 1u : 0u;
    const uint v19 = v7 ? 1u : 0u;
    const uint v23 = in_lid * 32u;
    const uint4 v24 = uint4(v5.x, v5.y, v18, v19);
    *(device packed_uint4*)(vp_args.out + v23) = v24;
    const uint v26 = v23 + 16u;
    const uint4 v27 = uint4(v12, v15, v16, v10);
    *(device packed_uint4*)(vp_args.out + v26) = v27;
}
