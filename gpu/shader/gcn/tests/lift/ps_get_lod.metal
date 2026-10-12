#include <metal_stdlib>
using namespace metal;

struct VpArgs {
    constant uchar* user_data;
    texture2d<float> tex0;
    sampler smp0;
};

struct VpIn {
    float4 attr0 [[user(loc0)]];
};

struct VpOut {
    float4 color0 [[color(0)]];
};

fragment VpOut vp_main(VpIn vp_in [[stage_in]],
        constant VpArgs& vp_args [[buffer(0)]],
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
    uint l24{};
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
    const uint v45 = as_type<uint>(vp_in.attr0.x);
    l19 = v45;
    const uint v48 = as_type<uint>(vp_in.attr0.y);
    l20 = v48;
    const uint v49 = l19;
    const float v50 = as_type<float>(v49);
    const uint v51 = l20;
    const float v52 = as_type<float>(v51);
    const float2 v54 = float2(v50, v52);
    const float2 v55 = float2(vp_args.tex0.calculate_clamped_lod(vp_args.smp0, v54), vp_args.tex0.calculate_unclamped_lod(vp_args.smp0, v54));
    const uint v58 = as_type<uint>(v55.x);
    l21 = v58;
    const uint v59 = as_type<uint>(v55.y);
    l22 = v59;
    l23 = 0u;
    l24 = 1065353216u;
    const uint v62 = l21;
    const float v63 = as_type<float>(v62);
    const uint v64 = l22;
    const float v65 = as_type<float>(v64);
    const uint v66 = l23;
    const float v67 = as_type<float>(v66);
    const uint v68 = l24;
    const float v69 = as_type<float>(v68);
    const float4 v70 = float4(v63, v65, v67, v69);
    vp_out.color0 = v70;
    return vp_out;
}
