#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpArgs {
    constant uchar* user_data;
    depth2d<float> tex0;
};

struct VpOut {
    float4 color0 [[color(0)]];
};

fragment VpOut vp_main(constant VpArgs& vp_args [[buffer(0)]],
        float4 in_frag_coord [[position]]) {
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
    l10 = (*(constant uint*)(vp_args.user_data + 24u));
    l11 = (*(constant uint*)(vp_args.user_data + 28u));
    l12 = (*(constant uint*)(vp_args.user_data + 32u));
    l13 = (*(constant uint*)(vp_args.user_data + 36u));
    l14 = (*(constant uint*)(vp_args.user_data + 40u));
    l15 = (*(constant uint*)(vp_args.user_data + 44u));
    l16 = 0u;
    l17 = 0u;
    l18 = 0u;
    const uint v43 = l19;
    const float v44 = as_type<float>(v43);
    const bool v46 = v44 >= 4294967296.0f;
    const float v49 = clamp(v44, 0.0f, 4294967040.0f);
    const uint v50 = uint(v49);
    const uint v52 = v46 ? 4294967295u : v50;
    const bool v54 = (as_type<uint>(v44) & 0x7fffffffu) > 0x7f800000u;
    const uint v55 = v54 ? 0u : v52;
    l20 = v55;
    const uint v56 = l21;
    const float v57 = as_type<float>(v56);
    const bool v59 = v57 >= 4294967296.0f;
    const float v62 = clamp(v57, 0.0f, 4294967040.0f);
    const uint v63 = uint(v62);
    const uint v65 = v59 ? 4294967295u : v63;
    const bool v67 = (as_type<uint>(v57) & 0x7fffffffu) > 0x7f800000u;
    const uint v68 = v67 ? 0u : v65;
    l22 = v68;
    const uint v69 = l20;
    const uint v70 = l22;
    const uint2 v71 = uint2(v69, v70);
    const float v72 = vp_args.tex0.read(v71);
    const uint v73 = as_type<uint>(v72);
    l23 = v73;
    const uint v74 = l23;
    const float v75 = as_type<float>(v74);
    const uint v76 = l23;
    const float v77 = as_type<float>(v76);
    const uint v78 = l23;
    const float v79 = as_type<float>(v78);
    const uint v80 = l23;
    const float v81 = as_type<float>(v80);
    const float4 v82 = float4(v75, v77, v79, v81);
    vp_out.color0 = v82;
    return vp_out;
}
