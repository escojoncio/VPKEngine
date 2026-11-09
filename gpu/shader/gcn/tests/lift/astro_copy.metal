#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpArgs {
    const device uchar* gsvs_ring0;
    constant uchar* user_data;
    constant uchar* mem1;
};

struct VpOut {
    float4 position [[position, invariant]];
    float4 param0 [[user(loc0)]];
};

vertex VpOut vp_main(constant VpArgs& vp_args [[buffer(0)]],
        uint in_vertex_id [[vertex_id]]) {
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
    l0 = true;
    l1 = false;
    l2 = false;
    l3 = 0u;
    l4 = (*(constant uint*)(vp_args.user_data));
    l5 = (*(constant uint*)(vp_args.user_data + 4u));
    const bool v11 = in_vertex_id == 4294967295u;
    if (v11) {
        vp_out.position = float4(2.0f, 2.0f, 2.0f, 0.0f);
        return vp_out;
    }
    l6 = in_vertex_id;
    l7 = 16u;
    const uint v18 = l8;
    const uint v19 = l7;
    const uint v20 = v18 | v19;
    const bool v22 = v20 != 0u;
    l1 = v22;
    const uint v26 = l6;
    const uint v27 = v26 << 2u;
    l9 = v27;
    l4 = (*(constant uint*)(vp_args.mem1 + 128u));
    l5 = (*(constant uint*)(vp_args.mem1 + 132u));
    l10 = (*(constant uint*)(vp_args.mem1 + 136u));
    l11 = (*(constant uint*)(vp_args.mem1 + 140u));
    l6 = 0u;
    const uint v44 = l9;
    const uint v50 = *(const device uint*)(vp_args.gsvs_ring0 + v44);
    l12 = v50;
    const uint v53 = l9;
    const uint v56 = v53 + 192u;
    const uint v59 = *(const device uint*)(vp_args.gsvs_ring0 + v56);
    l13 = v59;
    const uint v62 = l9;
    const uint v65 = v62 + 384u;
    const uint v68 = *(const device uint*)(vp_args.gsvs_ring0 + v65);
    l14 = v68;
    const uint v71 = l9;
    const uint v74 = v71 + 576u;
    const uint v77 = *(const device uint*)(vp_args.gsvs_ring0 + v74);
    l15 = v77;
    const uint v80 = l9;
    const uint v83 = v80 + 768u;
    const uint v86 = *(const device uint*)(vp_args.gsvs_ring0 + v83);
    l16 = v86;
    const uint v89 = l9;
    const uint v92 = v89 + 960u;
    const uint v95 = *(const device uint*)(vp_args.gsvs_ring0 + v92);
    l9 = v95;
    const uint v96 = l12;
    const float v97 = as_type<float>(v96);
    const uint v98 = l13;
    const float v99 = as_type<float>(v98);
    const uint v100 = l14;
    const float v101 = as_type<float>(v100);
    const uint v102 = l15;
    const float v103 = as_type<float>(v102);
    const float4 v104 = float4(v97, v99, v101, v103);
    vp_out.position = v104;
    const uint v105 = l16;
    const float v106 = as_type<float>(v105);
    const uint v107 = l9;
    const float v108 = as_type<float>(v107);
    const uint v109 = l6;
    const float v110 = as_type<float>(v109);
    const uint v111 = l6;
    const float v112 = as_type<float>(v111);
    const float4 v113 = float4(v106, v108, v110, v112);
    vp_out.param0 = v113;
    return vp_out;
}
