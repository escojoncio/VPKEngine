#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpArgs {
    const device uchar* input_index0;
    const device uchar* buf1;
    device uchar* esgs_ring2;
    constant uchar* user_data;
    constant uchar* geom_params1;
};

[[max_total_threads_per_threadgroup(64)]] kernel void vp_main(constant VpArgs& vp_args [[buffer(0)]],
        uint3 in_global_id [[thread_position_in_grid]]) {
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
    uint l17{};
    uint l18{};
    l0 = true;
    l1 = false;
    l2 = false;
    l3 = 0u;
    l4 = (*(constant uint*)(vp_args.user_data));
    l5 = (*(constant uint*)(vp_args.user_data + 4u));
    l6 = (*(constant uint*)(vp_args.user_data + 8u));
    l7 = (*(constant uint*)(vp_args.user_data + 12u));
    const uint v15 = in_global_id.x & 63u;
    const bool v20 = in_global_id.x >= (*(constant uint*)(vp_args.geom_params1));
    if (v20) {
        return;
    }
    const uint v22 = *(constant uint*)(vp_args.geom_params1 + 4u);
    const uint v23 = in_global_id.x / v22;
    const uint v24 = v23 * v22;
    const uint v25 = in_global_id.x - v24;
    l8 = v25;
    const uint v27 = *(constant uint*)(vp_args.geom_params1 + 12u);
    const bool v29 = v27 != 0u;
    if (v29) {
        const bool v31 = v27 == 2u;
        if (v31) {
            const uint v33 = v25 * 4u;
            const uint v34 = *(const device uint*)(vp_args.input_index0 + v33);
            l8 = v34;
        } else {
            const uint v37 = v25 >> 1u;
            const uint v38 = v37 << 2u;
            const uint v39 = *(const device uint*)(vp_args.input_index0 + v38);
            const uint v42 = v25 & 1u;
            const bool v43 = v42 != 0u;
            const uint v45 = v39 & 65535u;
            const uint v47 = v39 >> 16u;
            const uint v48 = v43 ? v47 : v45;
            l8 = v48;
        }
    }
    const uint v51 = l8;
    const uint v52 = v51 + (*(constant uint*)(vp_args.geom_params1 + 8u));
    const uint v55 = in_global_id.x >> 6u;
    const uint v56 = v55 * 2048u;
    l9 = v56;
    l10 = v52;
    const uint v60 = l10;
    const uint v61 = l5;
    const uint v64 = v61 >> 16u;
    const uint v65 = v64 & 16383u;
    const uint v66 = v60 * v65;
    const uint v70 = *(const device uint*)(vp_args.buf1 + v66);
    const uint v72 = v66 + 4u;
    const uint v73 = *(const device uint*)(vp_args.buf1 + v72);
    const uint v75 = v66 + 8u;
    const uint v76 = *(const device uint*)(vp_args.buf1 + v75);
    const uint v78 = v66 + 12u;
    const uint v79 = *(const device uint*)(vp_args.buf1 + v78);
    l11 = v70;
    l12 = v73;
    l13 = v76;
    l14 = v79;
    const uint v83 = l10;
    const uint v84 = l5;
    const uint v87 = v84 >> 16u;
    const uint v88 = v87 & 16383u;
    const uint v89 = v83 * v88;
    const uint v90 = 16u + v89;
    const uint v93 = *(const device uint*)(vp_args.buf1 + v90);
    const uint v95 = v90 + 4u;
    const uint v96 = *(const device uint*)(vp_args.buf1 + v95);
    const uint v98 = v90 + 8u;
    const uint v99 = *(const device uint*)(vp_args.buf1 + v98);
    const uint v101 = v90 + 12u;
    const uint v102 = *(const device uint*)(vp_args.buf1 + v101);
    l15 = v93;
    l16 = v96;
    l17 = v99;
    l18 = v102;
    const uint v103 = l9;
    const uint v111 = v15 * 4u;
    const uint v112 = v103 + v111;
    const uint v118 = l11;
    *(device uint*)(vp_args.esgs_ring2 + v112) = v118;
    const uint v123 = v112 + 256u;
    const uint v124 = l12;
    *(device uint*)(vp_args.esgs_ring2 + v123) = v124;
    const uint v129 = v112 + 512u;
    const uint v130 = l13;
    *(device uint*)(vp_args.esgs_ring2 + v129) = v130;
    const uint v135 = v112 + 768u;
    const uint v136 = l14;
    *(device uint*)(vp_args.esgs_ring2 + v135) = v136;
    const uint v137 = l9;
    const uint v146 = v137 + v111;
    const uint v151 = v146 + 1024u;
    const uint v152 = l15;
    *(device uint*)(vp_args.esgs_ring2 + v151) = v152;
    const uint v157 = v146 + 1280u;
    const uint v158 = l16;
    *(device uint*)(vp_args.esgs_ring2 + v157) = v158;
    const uint v163 = v146 + 1536u;
    const uint v164 = l17;
    *(device uint*)(vp_args.esgs_ring2 + v163) = v164;
    const uint v169 = v146 + 1792u;
    const uint v170 = l18;
    *(device uint*)(vp_args.esgs_ring2 + v169) = v170;
    return;
}
