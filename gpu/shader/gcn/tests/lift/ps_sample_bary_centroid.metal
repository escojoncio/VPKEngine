#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpArgs {
    constant uchar* user_data;
};

struct VpIn {
    float4 attr0 [[user(loc0), centroid_perspective]];
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
    l0 = true;
    l1 = false;
    l2 = false;
    l3 = 0u;
    l4 = (*(constant uint*)(vp_args.user_data));
    l5 = (*(constant uint*)(vp_args.user_data + 4u));
    l6 = 0u;
    l7 = 0u;
    l8 = 0u;
    const uint v25 = as_type<uint>(vp_in.attr0.x);
    l9 = v25;
    const uint v28 = as_type<uint>(vp_in.attr0.y);
    l10 = v28;
    const uint v29 = l9;
    const float v30 = as_type<float>(v29);
    const uint v31 = l10;
    const float v32 = as_type<float>(v31);
    const uint v33 = l9;
    const float v34 = as_type<float>(v33);
    const uint v35 = l10;
    const float v36 = as_type<float>(v35);
    const float4 v37 = float4(v30, v32, v34, v36);
    vp_out.color0 = v37;
    return vp_out;
}
