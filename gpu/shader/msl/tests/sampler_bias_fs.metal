#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpArgs {
    texture2d<float> albedo;
    sampler smp;
    float sbias0;
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
    const float4 v1 = vp_args.albedo.sample(vp_args.smp, vp_in.uv, bias(vp_args.sbias0));
    const float4 v3 = vp_args.albedo.sample(vp_args.smp, vp_in.uv, level(0.5f + vp_args.sbias0));
    const float4 v4 = vp_args.albedo.sample(vp_args.smp, vp_in.uv, bias(0.5f + vp_args.sbias0));
    const float4 v5 = vp_args.albedo.sample(vp_args.smp, vp_in.uv, gradient2d(vp_in.uv * exp2(vp_args.sbias0), vp_in.uv * exp2(vp_args.sbias0)));
    const float4 v6 = v1 + v3;
    const float4 v7 = v6 + v4;
    const float4 v8 = v7 + v5;
    vp_out.color = v8;
    return vp_out;
}
