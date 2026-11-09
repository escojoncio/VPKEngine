#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpArgs {
    const device uchar* buf0;
    constant uchar* user_data;
    constant uchar* mem1;
};

struct VpOut {
    float4 position [[position, invariant]];
    float4 param0 [[user(loc0)]];
};

vertex VpOut vp_main(constant VpArgs& vp_args [[buffer(0)]],
        uint in_vertex_id [[vertex_id]],
        uint in_instance_id [[instance_id]]) {
    VpOut vp_out;
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
    uint l19{};
    uint l20{};
    uint l21{};
    uint l22{};
    uint l23{};
    uint l24{};
    uint l25{};
    l0 = true;
    l1 = false;
    l2 = false;
    l3 = 0u;
    l4 = (*(constant uint*)(vp_args.user_data));
    l5 = (*(constant uint*)(vp_args.user_data + 4u));
    l6 = (*(constant uint*)(vp_args.user_data + 8u));
    l7 = (*(constant uint*)(vp_args.user_data + 12u));
    l8 = (*(constant uint*)(vp_args.user_data + 16u));
    l9 = (*(constant uint*)(vp_args.user_data + 20u));
    l10 = in_vertex_id;
    l11 = 0u;
    l12 = 0u;
    l13 = in_instance_id;
    l14 = (*(constant uint*)(vp_args.mem1));
    l15 = (*(constant uint*)(vp_args.mem1 + 4u));
    l16 = (*(constant uint*)(vp_args.mem1 + 8u));
    l17 = (*(constant uint*)(vp_args.mem1 + 12u));
    const uint v35 = l15;
    const uint v36 = l16;
    const uint v37 = l17;
    const uint v40 = v35 >> 16u;
    const uint v41 = v40 & 16383u;
    const uint v43 = v35 >> 31u;
    const bool v45 = v43 != 0u;
    const uint v53 = l10;
    const uint v58 = v53 * v41;
    const uint v62 = v37 >> 19u;
    const uint v63 = v62 & 3u;
    const uint v66 = v37 >> 21u;
    const uint v67 = v66 & 3u;
    const uint v69 = 2u << v63;
    const uint v71 = 8u << v67;
    const uint v73 = v67 + 3u;
    const uint v74 = v53 >> v73;
    const uint v76 = v71 - 1u;
    const uint v77 = v53 & v76;
    const uint v79 = v63 + 1u;
    const uint v80 = 0u >> v79;
    const uint v82 = v69 - 1u;
    const uint v83 = 0u & v82;
    const uint v84 = v80 * v69;
    const uint v85 = v74 * v41;
    const uint v86 = v85 + v84;
    const uint v87 = v71 * v86;
    const uint v88 = v69 * v77;
    const uint v89 = v83 + v88;
    const uint v90 = v89 + v87;
    const bool v92 = v41 != 0u;
    const bool v93 = v45 && v92;
    const uint v94 = v93 ? v90 : v58;
    const bool v98 = 0u >= v41;
    const bool v99 = v53 >= v36;
    const bool v100 = v99 || v98;
    const uint v102 = v94 + 15u;
    const bool v103 = v102 < v36;
    const bool v104 = !v100;
    const bool v105 = v92 ? v104 : v103;
    const bool v108 = v94 < v36;
    const bool v110 = v92 ? v104 : v108;
    const bool v111 = v110 && v105;
    l18 = 0u;
    l19 = 0u;
    l20 = 0u;
    l21 = 0u;
    if (v111) {
        const uint v118 = *(const device uint*)(vp_args.buf0 + v94);
        const uint v120 = v94 + 4u;
        const uint v121 = *(const device uint*)(vp_args.buf0 + v120);
        const uint v123 = v94 + 8u;
        const uint v124 = *(const device uint*)(vp_args.buf0 + v123);
        const uint v126 = v94 + 12u;
        const uint v127 = *(const device uint*)(vp_args.buf0 + v126);
        l18 = v118;
        l19 = v121;
        l20 = v124;
        l21 = v127;
    }
    const uint v135 = l18;
    const uint v136 = l19;
    const uint v137 = l20;
    const uint v138 = l21;
    l22 = v135;
    l23 = v136;
    l24 = v137;
    l25 = v138;
    const uint v139 = l22;
    const float v140 = as_type<float>(v139);
    const uint v141 = l23;
    const float v142 = as_type<float>(v141);
    const uint v143 = l24;
    const float v144 = as_type<float>(v143);
    const uint v145 = l25;
    const float v146 = as_type<float>(v145);
    const float4 v147 = float4(v140, v142, v144, v146);
    vp_out.position = v147;
    const uint v148 = l22;
    const float v149 = as_type<float>(v148);
    const uint v150 = l23;
    const float v151 = as_type<float>(v150);
    const uint v152 = l24;
    const float v153 = as_type<float>(v152);
    const uint v154 = l25;
    const float v155 = as_type<float>(v154);
    const float4 v156 = float4(v149, v151, v153, v155);
    vp_out.param0 = v156;
    return vp_out;
}
