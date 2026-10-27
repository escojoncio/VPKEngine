#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpArgs {
    const device uchar* gsvs_ring0;
};

struct VpOut {
    float4 position [[position]];
    uint layer [[render_target_array_index]];
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
    uint l17{};
    l0 = true;
    l1 = false;
    l2 = false;
    l3 = 0u;
    const bool v7 = in_vertex_id == 4294967295u;
    if (v7) {
        vp_out.position = float4(2.0f, 2.0f, 2.0f, 0.0f);
        return vp_out;
    }
    l4 = in_vertex_id;
    l5 = 1023u;
    const uint v14 = l6;
    const uint v15 = l5;
    const uint v16 = v14 | v15;
    const bool v18 = v16 != 0u;
    l1 = v18;
    const uint v22 = l4;
    const uint v23 = v22 << 2u;
    l7 = v23;
    l8 = 0u;
    const uint v25 = l8;
    const uint v27 = l7;
    const uint v29 = v27 + v25;
    const uint v33 = *(const device uint*)(vp_args.gsvs_ring0 + v29);
    l9 = v33;
    l8 = 384u;
    const uint v35 = l8;
    const uint v37 = l7;
    const uint v39 = v37 + v35;
    const uint v43 = *(const device uint*)(vp_args.gsvs_ring0 + v39);
    l10 = v43;
    l8 = 768u;
    const uint v45 = l8;
    const uint v47 = l7;
    const uint v49 = v47 + v45;
    const uint v53 = *(const device uint*)(vp_args.gsvs_ring0 + v49);
    l11 = v53;
    l8 = 1152u;
    const uint v55 = l8;
    const uint v57 = l7;
    const uint v59 = v57 + v55;
    const uint v63 = *(const device uint*)(vp_args.gsvs_ring0 + v59);
    l12 = v63;
    l8 = 1536u;
    const uint v65 = l8;
    const uint v67 = l7;
    const uint v69 = v67 + v65;
    const uint v73 = *(const device uint*)(vp_args.gsvs_ring0 + v69);
    l13 = v73;
    l8 = 1920u;
    const uint v75 = l8;
    const uint v77 = l7;
    const uint v79 = v77 + v75;
    const uint v83 = *(const device uint*)(vp_args.gsvs_ring0 + v79);
    l14 = v83;
    l8 = 2304u;
    const uint v85 = l8;
    const uint v87 = l7;
    const uint v89 = v87 + v85;
    const uint v93 = *(const device uint*)(vp_args.gsvs_ring0 + v89);
    l15 = v93;
    l8 = 2688u;
    const uint v95 = l8;
    const uint v97 = l7;
    const uint v99 = v97 + v95;
    const uint v103 = *(const device uint*)(vp_args.gsvs_ring0 + v99);
    l16 = v103;
    l8 = 3072u;
    const uint v105 = l8;
    const uint v107 = l7;
    const uint v109 = v107 + v105;
    const uint v113 = *(const device uint*)(vp_args.gsvs_ring0 + v109);
    l17 = v113;
    const uint v114 = l9;
    const float v115 = as_type<float>(v114);
    const uint v116 = l10;
    const float v117 = as_type<float>(v116);
    const uint v118 = l11;
    const float v119 = as_type<float>(v118);
    const uint v120 = l12;
    const float v121 = as_type<float>(v120);
    const float4 v122 = float4(v115, v117, v119, v121);
    vp_out.position = v122;
    const uint v125 = l17;
    const float v126 = as_type<float>(v125);
    const uint v129 = as_type<uint>(v126);
    vp_out.layer = v129;
    const uint v130 = l13;
    const float v131 = as_type<float>(v130);
    const uint v132 = l14;
    const float v133 = as_type<float>(v132);
    const uint v134 = l15;
    const float v135 = as_type<float>(v134);
    const uint v136 = l16;
    const float v137 = as_type<float>(v136);
    const float4 v138 = float4(v131, v133, v135, v137);
    vp_out.param0 = v138;
    return vp_out;
}
