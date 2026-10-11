#include <metal_stdlib>
using namespace metal;

struct VpArgs {
    constant uchar* params;
    texture2d<float> albedo;
    sampler smp;
};

struct VpIn {
    float2 uv [[user(loc0)]];
    float4 tint [[user(loc1), centroid_no_perspective]];
};

struct VpOut {
    float4 color [[color(0)]];
};

fragment VpOut vp_main(VpIn vp_in [[stage_in]],
        constant VpArgs& vp_args [[buffer(0)]]) {
    VpOut vp_out;
    const float4 v1 = vp_args.albedo.sample(vp_args.smp, vp_in.uv);
    const bool v5 = v1.w < (*(constant float*)(vp_args.params));
    if (v5) {
        discard_fragment();
    }
    const float4 v7 = v1 * vp_in.tint;
    vp_out.color = v7;
    return vp_out;
}
