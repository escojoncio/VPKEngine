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
    bool l10{};
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
    const uint v13 = as_type<uint>(vp_in.attr0.x);
    const uint v14 = l9;
    const bool v15 = l0;
    const uint v16 = v15 ? v13 : v14;
    l9 = v16;
    const uint v17 = l9;
    const float v18 = as_type<float>(v17);
    const bool v24 = 0.5f > v18;
    const bool v25 = l0;
    const bool v26 = v24 && v25;
    l1 = v26;
    const bool v27 = l0;
    const bool v28 = l1;
    l10 = v27;
    const bool v29 = v28 && v27;
    l0 = v29;
    l2 = v29;
    const bool v30 = l0;
    const bool v31 = l10;
    const bool v32 = v30 != v31;
    l10 = v32;
    l2 = v32;
    const bool v33 = l0;
    const bool v34 = !v33;
    if (v34) {
    } else {
        const uint v36 = l11;
        const bool v37 = l0;
        const uint v38 = v37 ? 1065353216u : v36;
        l11 = v38;
    }
    const bool v39 = l0;
    const bool v40 = l10;
    l10 = v39;
    const bool v41 = v40 || v39;
    l0 = v41;
    l2 = v41;
    const bool v42 = l0;
    const bool v43 = l10;
    const bool v44 = v42 != v43;
    l0 = v44;
    l2 = v44;
    const bool v45 = l0;
    const bool v46 = !v45;
    if (v46) {
    } else {
        const uint v48 = l11;
        const bool v49 = l0;
        const uint v50 = v49 ? 0u : v48;
        l11 = v50;
    }
    const bool v51 = l0;
    const bool v52 = l10;
    const bool v53 = v51 || v52;
    l0 = v53;
    l2 = v53;
    const uint v54 = l9;
    const float v55 = as_type<float>(v54);
    const bool v61 = 0.0f <= v55;
    const bool v62 = l0;
    const bool v63 = v61 && v62;
    l1 = v63;
    l0 = v63;
    const uint v64 = l11;
    const float v65 = as_type<float>(v64);
    const uint v66 = l11;
    const float v67 = as_type<float>(v66);
    const uint v68 = l11;
    const float v69 = as_type<float>(v68);
    const uint v70 = l11;
    const float v71 = as_type<float>(v70);
    const float4 v72 = float4(v65, v67, v69, v71);
    const bool v73 = l0;
    const bool v74 = !v73;
    if (v74) {
        discard_fragment();
    }
    vp_out.color0 = v72;
    return vp_out;
}
