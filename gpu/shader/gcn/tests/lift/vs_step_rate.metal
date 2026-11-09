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
    const uint v20 = in_instance_id / 3u;
    l12 = v20;
    l13 = in_instance_id;
    l14 = (*(constant uint*)(vp_args.mem1));
    l15 = (*(constant uint*)(vp_args.mem1 + 4u));
    l16 = (*(constant uint*)(vp_args.mem1 + 8u));
    l17 = (*(constant uint*)(vp_args.mem1 + 12u));
    const uint v36 = l15;
    const uint v37 = l16;
    const uint v38 = l17;
    const uint v41 = v36 >> 16u;
    const uint v42 = v41 & 16383u;
    const uint v44 = v36 >> 31u;
    const bool v46 = v44 != 0u;
    const uint v54 = l10;
    const uint v59 = v54 * v42;
    const uint v63 = v38 >> 19u;
    const uint v64 = v63 & 3u;
    const uint v67 = v38 >> 21u;
    const uint v68 = v67 & 3u;
    const uint v70 = 2u << v64;
    const uint v72 = 8u << v68;
    const uint v74 = v68 + 3u;
    const uint v75 = v54 >> v74;
    const uint v77 = v72 - 1u;
    const uint v78 = v54 & v77;
    const uint v80 = v64 + 1u;
    const uint v81 = 0u >> v80;
    const uint v83 = v70 - 1u;
    const uint v84 = 0u & v83;
    const uint v85 = v81 * v70;
    const uint v86 = v75 * v42;
    const uint v87 = v86 + v85;
    const uint v88 = v72 * v87;
    const uint v89 = v70 * v78;
    const uint v90 = v84 + v89;
    const uint v91 = v90 + v88;
    const bool v93 = v42 != 0u;
    const bool v94 = v46 && v93;
    const uint v95 = v94 ? v91 : v59;
    const bool v99 = 0u >= v42;
    const bool v100 = v54 >= v37;
    const bool v101 = v100 || v99;
    const uint v103 = v95 + 15u;
    const bool v104 = v103 < v37;
    const bool v105 = !v101;
    const bool v106 = v93 ? v105 : v104;
    const bool v109 = v95 < v37;
    const bool v111 = v93 ? v105 : v109;
    const bool v112 = v111 && v106;
    l18 = 0u;
    l19 = 0u;
    l20 = 0u;
    l21 = 0u;
    if (v112) {
        const uint v119 = *(const device uint*)(vp_args.buf0 + v95);
        const uint v121 = v95 + 4u;
        const uint v122 = *(const device uint*)(vp_args.buf0 + v121);
        const uint v124 = v95 + 8u;
        const uint v125 = *(const device uint*)(vp_args.buf0 + v124);
        const uint v127 = v95 + 12u;
        const uint v128 = *(const device uint*)(vp_args.buf0 + v127);
        l18 = v119;
        l19 = v122;
        l20 = v125;
        l21 = v128;
    }
    const uint v136 = l18;
    const uint v137 = l19;
    const uint v138 = l20;
    const uint v139 = l21;
    l22 = v136;
    l23 = v137;
    l24 = v138;
    l25 = v139;
    const uint v140 = l22;
    const float v141 = as_type<float>(v140);
    const uint v142 = l23;
    const float v143 = as_type<float>(v142);
    const uint v144 = l24;
    const float v145 = as_type<float>(v144);
    const uint v146 = l25;
    const float v147 = as_type<float>(v146);
    const float4 v148 = float4(v141, v143, v145, v147);
    vp_out.position = v148;
    const uint v149 = l22;
    const float v150 = as_type<float>(v149);
    const uint v151 = l23;
    const float v152 = as_type<float>(v151);
    const uint v153 = l24;
    const float v154 = as_type<float>(v153);
    const uint v155 = l25;
    const float v156 = as_type<float>(v155);
    const float4 v157 = float4(v150, v152, v154, v156);
    vp_out.param0 = v157;
    return vp_out;
}
