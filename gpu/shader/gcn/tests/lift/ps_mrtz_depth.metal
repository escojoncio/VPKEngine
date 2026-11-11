#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpArgs {
    constant uchar* user_data;
};

struct VpIn {
    float4 attr0 [[user(loc0)]];
};

struct VpOut {
    float4 color0 [[color(0)]];
#ifndef VP_NO_FDEPTH
    float depth [[depth(any)]];
#endif
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
    const uint v26 = l9;
    const float v27 = as_type<float>(v26);
    const uint v28 = l9;
    const float v29 = as_type<float>(v28);
    const uint v30 = l9;
    const float v31 = as_type<float>(v30);
    const uint v32 = l9;
    const float v33 = as_type<float>(v32);
    const float4 v34 = float4(v27, v29, v31, v33);
    vp_out.color0 = v34;
    const uint v35 = l9;
    const float v36 = as_type<float>(v35);
#ifndef VP_NO_FDEPTH
    vp_out.depth = v36;
#endif
    return vp_out;
}
