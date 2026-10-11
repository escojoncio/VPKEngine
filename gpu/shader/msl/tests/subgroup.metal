#include <metal_stdlib>
using namespace metal;

struct VpArgs {
    device uchar* out;
    const device uchar* src;
};

[[max_total_threads_per_threadgroup(32)]] kernel void vp_main(constant VpArgs& vp_args [[buffer(0)]],
        uint3 in_gid [[thread_position_in_grid]],
        uint in_lane [[thread_index_in_simdgroup]]) {
    const uint v4 = in_gid.x << 2u;
    const float v5 = *(const device float*)(vp_args.src + v4);
    const bool v7 = v5 > 0.0f;
    const uint4 v8 = uint4(as_type<uint2>(simd_vote::vote_t(simd_ballot(v7))), 0u, 0u);
    const bool v9 = simd_any(v7);
    const bool v10 = simd_all(v7);
    const float v11 = simd_broadcast_first(v5);
    const uint v13 = in_lane ^ 1u;
    const float v14 = simd_shuffle(v5, ushort(v13));
    const float v15 = simd_broadcast(v5, ushort(1u));
    const float v16 = v9 ? v11 : v14;
    const float v17 = v10 ? v15 : v16;
    const uint v19 = in_gid.x << 4u;
    *(device packed_uint4*)(vp_args.out + v19) = v8;
    const uint v21 = v19 + 16u;
    *(device float*)(vp_args.out + v21) = v17;
}
