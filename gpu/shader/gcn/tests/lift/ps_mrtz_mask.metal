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
    const uint v26 = l9;
    const bool v27 = l0;
    const uint v28 = v27 ? v25 : v26;
    l9 = v28;
    const uint v29 = l9;
    const float v30 = as_type<float>(v29);
    const bool v36 = 0.0f <= v30;
    const bool v37 = l0;
    const bool v38 = v36 && v37;
    l1 = v38;
    const uint v41 = v38 ? 1u : 0u;
    l10 = v41;
    l11 = 0u;
    l0 = v38;
    const uint v43 = l9;
    const float v44 = as_type<float>(v43);
    const uint v45 = l9;
    const float v46 = as_type<float>(v45);
    const uint v47 = l9;
    const float v48 = as_type<float>(v47);
    const uint v49 = l9;
    const float v50 = as_type<float>(v49);
    const float4 v51 = float4(v44, v46, v48, v50);
    const bool v52 = l0;
    const bool v53 = !v52;
    if (v53) {
        discard_fragment();
    }
    vp_out.color0 = v51;
    const bool v65 = l0;
    const bool v66 = !v65;
    if (v66) {
        discard_fragment();
    }
    return vp_out;
}
