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
        constant VpArgs& vp_args [[buffer(0)]],
        float4 in_frag_coord [[position]]) {
    VpOut vp_out;
    uint4 vp_private[34] = {};
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
    l10 = 0u;
    const uint v27 = l10;
    l10 = v27;
    const uint v31 = l10;
    const uint v32 = v31 << 2u;
    l10 = v32;
    l3 = 4294967295u;
    const uint v34 = l10;
    const uint v35 = l9;
    const uint v37 = v34 + 256u;
    const uint v39 = min(v37, 540u);
    *(thread uint*)((thread uchar*)vp_private + v39) = v35;
    l11 = 127u;
    const uint v41 = l10;
    const uint v42 = l11;
    const uint v44 = v41 + 512u;
    const uint v46 = min(v44, 540u);
    *(thread uint*)((thread uchar*)vp_private + v46) = v42;
    const uint v47 = l10;
    const uint v49 = v47 + 513u;
    const uint v51 = min(v49, 540u);
    const uint v53 = v51 & 4294967292u;
    const uint v56 = v51 & 3u;
    const uint v57 = v56 << 3u;
    const uint v59 = 255u << v57;
    const uint v60 = ~v59;
    const uint v61 = *(thread uint*)((thread uchar*)vp_private + v53);
    const uint v62 = v61 & v60;
    const uint v64 = l11;
    const uint v65 = v64 & 255u;
    const uint v66 = v65 << v57;
    const uint v67 = v62 | v66;
    *(thread uint*)((thread uchar*)vp_private + v53) = v67;
    l9 = 0u;
    const uint v69 = l10;
    const uint v71 = v69 + 256u;
    const uint v73 = min(v71, 540u);
    const uint v74 = *(thread uint*)((thread uchar*)vp_private + v73);
    l12 = v74;
    const uint v75 = l10;
    const uint v77 = v75 + 512u;
    const uint v79 = min(v77, 540u);
    const uint v80 = *(thread uint*)((thread uchar*)vp_private + v79);
    l13 = v80;
    const uint v81 = l13;
    const float v82 = float(v81);
    const uint v83 = as_type<uint>(v82);
    l13 = v83;
    const uint v84 = l12;
    const float v85 = as_type<float>(v84);
    const uint v86 = l12;
    const float v87 = as_type<float>(v86);
    const uint v88 = l13;
    const float v89 = as_type<float>(v88);
    const uint v90 = l13;
    const float v91 = as_type<float>(v90);
    const float4 v92 = float4(v85, v87, v89, v91);
    vp_out.color0 = v92;
    return vp_out;
}
