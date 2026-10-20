#include <metal_stdlib>
using namespace metal;

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
    const uint v34 = l10;
    const uint v35 = l15;
    const uint v38 = v35 >> 16u;
    const uint v39 = v38 & 16383u;
    const uint v40 = v34 * v39;
    const uint v44 = *(const device uint*)(vp_args.buf0 + v40);
    const uint v46 = v40 + 4u;
    const uint v47 = *(const device uint*)(vp_args.buf0 + v46);
    const uint v49 = v40 + 8u;
    const uint v50 = *(const device uint*)(vp_args.buf0 + v49);
    const uint v52 = v40 + 12u;
    const uint v53 = *(const device uint*)(vp_args.buf0 + v52);
    l18 = v44;
    l19 = v47;
    l20 = v50;
    l21 = v53;
    const uint v54 = l18;
    const float v55 = as_type<float>(v54);
    const uint v56 = l19;
    const float v57 = as_type<float>(v56);
    const uint v58 = l20;
    const float v59 = as_type<float>(v58);
    const uint v60 = l21;
    const float v61 = as_type<float>(v60);
    const float4 v62 = float4(v55, v57, v59, v61);
    vp_out.position = v62;
    const uint v63 = l18;
    const float v64 = as_type<float>(v63);
    const uint v65 = l19;
    const float v66 = as_type<float>(v65);
    const uint v67 = l20;
    const float v68 = as_type<float>(v67);
    const uint v69 = l21;
    const float v70 = as_type<float>(v69);
    const float4 v71 = float4(v64, v66, v68, v70);
    vp_out.param0 = v71;
    return vp_out;
}
