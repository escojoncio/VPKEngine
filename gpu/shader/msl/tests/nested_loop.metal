#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpArgs {
    device uchar* data;
};

[[max_total_threads_per_threadgroup(1)]] kernel void vp_main(constant VpArgs& vp_args [[buffer(0)]]) {
    uint vp_nest0 = 0u;
    uint vp_guard0 = 0u;
    while (true) {
        if (++vp_guard0 > 8192u) break;
        const uint v2 = *(device uint*)(vp_args.data);
        const bool v3 = v2 == 0u;
        if (v3) {
            break;
        }
        uint vp_guard1 = 0u;
        while (true) {
            if (++vp_guard1 > 8192u || ++vp_nest0 > 131072u) break;
            const uint v4 = *(device uint*)(vp_args.data + 1u);
            const bool v5 = v4 == 0u;
            if (v5) {
                break;
            }
        }
    }
}
