#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpArgs {
    device uchar* out;
    const device uchar* src;
};

[[max_total_threads_per_threadgroup(32)]] kernel void vp_main(constant VpArgs& vp_args [[buffer(0)]],
        uint3 in_gid [[thread_position_in_grid]],
        uint in_lane [[thread_index_in_simdgroup]]) {
    const uint v4 = in_gid.x << 2u;
    const float v5 = *(const device float*)(vp_args.src + v4);
    const uint v6 = as_type<uint>(v5);
    const float sx7 = v5;
    const float v7 = simd_sum(sx7);
    const float sx8 = v5;
    const float v8 = simd_prefix_inclusive_sum(sx8);
    const uint sx9 = v6;
    const uint v9 = simd_prefix_exclusive_product(sx9);
    const float sx10 = v5;
    const uint sm10 = as_type<uint2>(simd_vote::vote_t(simd_active_threads_mask())).x;
    const uint sl10 = in_lane;
    float sa10_0 = as_type<float>(0x7f800000u);
    for (uint sk = 0u; sk < 32u; ++sk) {
        const float sv = simd_shuffle(sx10, ushort(sk));
        if (((sm10 >> sk) & 1u) != 0u && sk < sl10) sa10_0 = min(sa10_0, sv);
    }
    const float v10 = sa10_0;
    const uint sx11 = v6;
    const uint sm11 = as_type<uint2>(simd_vote::vote_t(simd_active_threads_mask())).x;
    const uint sl11 = in_lane;
    uint sa11_0 = 0u;
    for (uint sk = 0u; sk < 32u; ++sk) {
        const uint sv = simd_shuffle(sx11, ushort(sk));
        if (((sm11 >> sk) & 1u) != 0u && (sk >> 2u) == (sl11 >> 2u)) sa11_0 = max(sa11_0, sv);
    }
    const uint v11 = sa11_0;
    const uint sx12 = v6;
    const uint v12 = simd_and(sx12);
    const float2 v13 = float2(v5);
    const float2 sx14 = v13;
    const uint sm14 = as_type<uint2>(simd_vote::vote_t(simd_active_threads_mask())).x;
    const uint sl14 = in_lane;
    float sa14_0 = as_type<float>(0x7f800000u);
    for (uint sk = 0u; sk < 32u; ++sk) {
        const float sv = simd_shuffle(sx14.x, ushort(sk));
        if (((sm14 >> sk) & 1u) != 0u && (sk >> 1u) == (sl14 >> 1u)) sa14_0 = min(sa14_0, sv);
    }
    float sa14_1 = as_type<float>(0x7f800000u);
    for (uint sk = 0u; sk < 32u; ++sk) {
        const float sv = simd_shuffle(sx14.y, ushort(sk));
        if (((sm14 >> sk) & 1u) != 0u && (sk >> 1u) == (sl14 >> 1u)) sa14_1 = min(sa14_1, sv);
    }
    const float2 v14 = float2(sa14_0, sa14_1);
    const uint v16 = in_gid.x << 4u;
    *(device float*)(vp_args.out + v16) = v7;
    const uint v18 = v16 + 4u;
    *(device float*)(vp_args.out + v18) = v8;
    const uint v19 = v18 + 4u;
    *(device uint*)(vp_args.out + v19) = v9;
    const uint v20 = v19 + 4u;
    *(device float*)(vp_args.out + v20) = v10;
    const uint v21 = v20 + 4u;
    *(device uint*)(vp_args.out + v21) = v11;
    const uint v22 = v21 + 4u;
    *(device uint*)(vp_args.out + v22) = v12;
    const uint v23 = v22 + 4u;
    *(device packed_float2*)(vp_args.out + v23) = v14;
}
