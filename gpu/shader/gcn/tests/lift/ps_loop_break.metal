#include <metal_stdlib>
using namespace metal;

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
        constant VpArgs& vp_args [[buffer(0)]]) {
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
    bool l12{};
    l0 = true;
    l1 = false;
    l2 = false;
    l3 = 0u;
    l4 = (*(constant uint*)(vp_args.user_data));
    l5 = (*(constant uint*)(vp_args.user_data + 4u));
    l6 = 0u;
    l7 = 0u;
    l8 = 0u;
    const uint v13 = as_type<uint>(vp_in.attr0.x);
    const uint v14 = l9;
    const bool v15 = l0;
    const uint v16 = v15 ? v13 : v14;
    l9 = v16;
    const uint v18 = l10;
    const bool v19 = l0;
    const uint v20 = v19 ? 0u : v18;
    l10 = v20;
    l11 = 0u;
    const bool v22 = l0;
    l12 = v22;
    while (true) {
        const uint v23 = l9;
        const float v24 = as_type<float>(v23);
        const uint v25 = l10;
        const float v26 = as_type<float>(v25);
        const float v27 = v26 + v24;
        const uint v28 = as_type<uint>(v27);
        const uint v29 = l10;
        const bool v30 = l0;
        const uint v31 = v30 ? v28 : v29;
        l10 = v31;
        const uint v32 = l10;
        const float v33 = as_type<float>(v32);
        const bool v39 = 1.0f < v33;
        const bool v40 = l0;
        const bool v41 = v39 && v40;
        l1 = v41;
        const bool v42 = l0;
        const bool v43 = l1;
        const bool v44 = !v43;
        const bool v45 = v42 && v44;
        l0 = v45;
        l2 = v45;
        const bool v46 = l0;
        const bool v47 = !v46;
        if (v47) {
        } else {
            const uint v48 = l11;
            const uint v50 = v48 + 1u;
            const bool v51 = v50 < v48;
            l2 = v51;
            l11 = v50;
            const uint v53 = l11;
            const bool v54 = v53 < 8u;
            l2 = v54;
            const bool v55 = l2;
            if (v55) {
                continue;
            }
        }
        break;
        break;
    }
    const bool v56 = l12;
    l0 = v56;
    const uint v57 = l10;
    const float v58 = as_type<float>(v57);
    const uint v59 = l10;
    const float v60 = as_type<float>(v59);
    const uint v61 = l10;
    const float v62 = as_type<float>(v61);
    const uint v63 = l10;
    const float v64 = as_type<float>(v63);
    const float4 v65 = float4(v58, v60, v62, v64);
    const bool v66 = l0;
    const bool v67 = !v66;
    if (v67) {
        discard_fragment();
    }
    vp_out.color0 = v65;
    return vp_out;
}
