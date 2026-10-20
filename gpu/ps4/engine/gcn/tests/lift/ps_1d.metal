#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpArgs {
    constant uchar* user_data;
    texture1d<float> tex0;
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
    uint l25{};
    uint l26{};
    uint l27{};
    uint l28{};
    uint l29{};
    uint l30{};
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
    l20 = 3u;
    const uint v47 = l20;
    const int v55 = as_type<int>(v47);
    const uint v57 = as_type<uint>(v55);
    const uint v66 = v57 & 63u;
    const uint v73 = v66 << 26u;
    const int v74 = as_type<int>(v73);
    const int v75 = v74 >> 26;
    const uint v76 = as_type<uint>(v75);
    const int v82 = as_type<int>(v76);
    const uint v153 = l19;
    const float v154 = as_type<float>(v153);
    const uint v155 = vp_args.tex0.get_width();
    const float v156 = float(v155);
    const float v157 = float(v82);
    const float v158 = v157 / v156;
    const float v159 = v154 + v158;
    const float4 v160 = vp_args.tex0.sample(vp_args.smp0, v159);
    const uint v162 = as_type<uint>(v160.x);
    l21 = v162;
    const uint v164 = as_type<uint>(v160.y);
    l22 = v164;
    const uint v166 = as_type<uint>(v160.z);
    l23 = v166;
    const uint v168 = as_type<uint>(v160.w);
    l24 = v168;
    const uint v169 = l19;
    l25 = v169;
    l26 = 1065353216u;
    const uint v171 = l25;
    const float v172 = as_type<float>(v171);
    const float4 v175 = vp_args.tex0.sample(vp_args.smp0, v172);
    const uint v177 = as_type<uint>(v175.x);
    l27 = v177;
    const uint v179 = as_type<uint>(v175.y);
    l28 = v179;
    const uint v181 = as_type<uint>(v175.z);
    l29 = v181;
    const uint v183 = as_type<uint>(v175.w);
    l30 = v183;
    const uint v184 = l27;
    const float v185 = as_type<float>(v184);
    const uint v186 = l21;
    const float v187 = as_type<float>(v186);
    const float v188 = v187 + v185;
    const uint v189 = as_type<uint>(v188);
    l21 = v189;
    const uint v190 = l21;
    const float v191 = as_type<float>(v190);
    const uint v192 = l22;
    const float v193 = as_type<float>(v192);
    const uint v194 = l23;
    const float v195 = as_type<float>(v194);
    const uint v196 = l24;
    const float v197 = as_type<float>(v196);
    const float4 v198 = float4(v191, v193, v195, v197);
    vp_out.color0 = v198;
    return vp_out;
}
