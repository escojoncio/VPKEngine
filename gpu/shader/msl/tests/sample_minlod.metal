#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpArgs {
    texture2d<float> albedo;
    sampler smp;
};

struct VpIn {
    float2 uv [[user(loc0)]];
};

struct VpOut {
    float4 color [[color(0)]];
};

fragment VpOut vp_main(VpIn vp_in [[stage_in]],
        constant VpArgs& vp_args [[buffer(0)]]) {
    VpOut vp_out;
    const float4 v3 = vp_args.albedo.sample(vp_args.smp, vp_in.uv, bias(0.5f), min_lod_clamp(2.0f));
    const float2 v4 = dfdx(vp_in.uv);
    const float2 v5 = dfdy(vp_in.uv);
    const float4 v6 = vp_args.albedo.sample(vp_args.smp, vp_in.uv, gradient2d(v4, v5), min_lod_clamp(2.0f));
    const float4 v7 = v3 + v6;
    vp_out.color = v7;
    return vp_out;
}
