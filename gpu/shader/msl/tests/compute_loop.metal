#include <metal_stdlib>
using namespace metal;

struct VpArgs {
    const device uchar* data;
    device uchar* histogram;
    constant uchar* params;
};

[[max_total_threads_per_threadgroup(64)]] kernel void vp_main(constant VpArgs& vp_args [[buffer(0)]],
        uint3 in_gid [[thread_position_in_grid]],
        uint in_lid [[thread_index_in_threadgroup]]) {
    threadgroup uint4 vp_shared[16];
    uint l0{};
    uint l1{};
    const uint v6 = *(constant uint*)(vp_args.params);
    l0 = 0u;
    l1 = 0u;
    while (true) {
        const uint v7 = l1;
        const bool v8 = v7 >= v6;
        if (v8) {
            break;
        }
        const uint v9 = in_gid.x * v6;
        const uint v10 = v9 + v7;
        const uint v11 = v10 << 2u;
        const uint v12 = *(const device uint*)(vp_args.data + v11);
        const bool v13 = v12 == 0u;
        if (v13) {
            break;
        }
        const uint v14 = l0;
        const uint v15 = v14 + v12;
        l0 = v15;
        const uint v16 = v7 + 1u;
        l1 = v16;
    }
    const uint v17 = in_lid << 2u;
    const uint v18 = l0;
    *(threadgroup uint*)((threadgroup uchar*)vp_shared + v17) = v18;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint v20 = v18 & 63u;
    const uint v21 = v20 << 2u;
    atomic_fetch_add_explicit((device atomic_uint*)(vp_args.histogram + v21), 1u, memory_order_relaxed);
    atomic_fetch_max_explicit((device atomic_uint*)(vp_args.histogram + 256u), v18, memory_order_relaxed);
    uint v26;
    do {
        v26 = 0u;
    } while (!atomic_compare_exchange_weak_explicit((device atomic_uint*)(vp_args.histogram + 260u), &v26, in_gid.x, memory_order_relaxed, memory_order_relaxed) && v26 == 0u);
}
