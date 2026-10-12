#include <metal_stdlib>
using namespace metal;

struct VpArgs {
    texture2d_array<uint, access::read_write> layers;
    texture3d<int, access::read_write> vol;
};

[[max_total_threads_per_threadgroup(64)]] kernel void vp_main(constant VpArgs& vp_args [[buffer(0)]],
        uint3 in_gid [[thread_position_in_grid]]) {
    const uint v2 = vp_args.layers.atomic_fetch_add(in_gid.xy, in_gid.z, 1u).x;
    const int v3 = as_type<int>(v2);
    vp_args.vol.atomic_fetch_max(in_gid, v3);
    uint v6;
    vec<uint, 4> v6_e;
    do {
        v6_e = 0u;
    } while (!vp_args.layers.atomic_compare_exchange_weak(in_gid.xy, in_gid.z, &v6_e, v2) && v6_e.x == 0u);
    v6 = v6_e.x;
    vp_args.layers.atomic_exchange(in_gid.xy, in_gid.z, v6);
}
