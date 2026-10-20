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
    float4 position [[position]];
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
    l11 = in_instance_id;
    l12 = in_instance_id;
    l13 = in_instance_id;
    l14 = (*(constant uint*)(vp_args.mem1));
    l15 = (*(constant uint*)(vp_args.mem1 + 4u));
    l16 = (*(constant uint*)(vp_args.mem1 + 8u));
    l17 = (*(constant uint*)(vp_args.mem1 + 12u));
    const uint v32 = l15;
    const uint v33 = l16;
    const uint v34 = l17;
    const uint v37 = v32 >> 16u;
    const uint v38 = v37 & 16383u;
    const uint v40 = v32 >> 31u;
    const bool v42 = v40 != 0u;
    const uint v51 = l10;
    const uint v56 = v51 * v38;
    const uint v60 = v34 >> 19u;
    const uint v61 = v60 & 3u;
    const uint v64 = v34 >> 21u;
    const uint v65 = v64 & 3u;
    const uint v67 = 2u << v61;
    const uint v69 = 8u << v65;
    const uint v71 = v65 + 3u;
    const uint v72 = v51 >> v71;
    const uint v74 = v69 - 1u;
    const uint v75 = v51 & v74;
    const uint v77 = v61 + 1u;
    const uint v78 = 0u >> v77;
    const uint v80 = v67 - 1u;
    const uint v81 = 0u & v80;
    const uint v82 = v78 * v67;
    const uint v83 = v72 * v38;
    const uint v84 = v83 + v82;
    const uint v85 = v69 * v84;
    const uint v86 = v67 * v75;
    const uint v87 = v81 + v86;
    const uint v88 = v87 + v85;
    const bool v90 = v38 != 0u;
    const bool v91 = v42 && v90;
    const uint v92 = v91 ? v88 : v56;
    const bool v96 = 0u >= v38;
    const bool v97 = v51 >= v33;
    const bool v98 = v97 || v96;
    const uint v100 = v92 + 15u;
    const bool v101 = v100 < v33;
    const bool v102 = !v98;
    const bool v103 = v90 ? v102 : v101;
    const bool v106 = v92 < v33;
    const bool v108 = v90 ? v102 : v106;
    const bool v109 = v108 && v103;
    l18 = 0u;
    l19 = 0u;
    l20 = 0u;
    l21 = 0u;
    if (v109) {
        const uint v116 = *(const device uint*)(vp_args.buf0 + v92);
        const uint v118 = v92 + 4u;
        const uint v119 = *(const device uint*)(vp_args.buf0 + v118);
        const uint v121 = v92 + 8u;
        const uint v122 = *(const device uint*)(vp_args.buf0 + v121);
        const uint v124 = v92 + 12u;
        const uint v125 = *(const device uint*)(vp_args.buf0 + v124);
        l18 = v116;
        l19 = v119;
        l20 = v122;
        l21 = v125;
    }
    const uint v126 = l18;
    const uint v127 = l19;
    const uint v128 = l20;
    const uint v129 = l21;
    l22 = v126;
    l23 = v127;
    l24 = v128;
    l25 = v129;
    const uint v130 = l22;
    const float v131 = as_type<float>(v130);
    const uint v132 = l23;
    const float v133 = as_type<float>(v132);
    const uint v134 = l24;
    const float v135 = as_type<float>(v134);
    const uint v136 = l25;
    const float v137 = as_type<float>(v136);
    const float4 v138 = float4(v131, v133, v135, v137);
    vp_out.position = v138;
    const uint v139 = l22;
    const float v140 = as_type<float>(v139);
    const uint v141 = l23;
    const float v142 = as_type<float>(v141);
    const uint v143 = l24;
    const float v144 = as_type<float>(v143);
    const uint v145 = l25;
    const float v146 = as_type<float>(v145);
    const float4 v147 = float4(v140, v142, v144, v146);
    vp_out.param0 = v147;
    return vp_out;
}
