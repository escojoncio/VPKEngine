#include <metal_stdlib>
using namespace metal;

struct VpArgs {
    texture2d_array<float> layers;
    texturecube_array<float> cubes;
    sampler smp;
};

struct VpIn {
    float3 uv [[user(loc0)]];
    float4 dir [[user(loc1)]];
};

struct VpOut {
    float4 color [[color(0)]];
};

fragment VpOut vp_main(VpIn vp_in [[stage_in]],
        constant VpArgs& vp_args [[buffer(0)]]) {
    VpOut vp_out;
    const float2 v2 = float2(vp_args.layers.calculate_clamped_lod(vp_args.smp, vp_in.uv.xy), vp_args.layers.calculate_unclamped_lod(vp_args.smp, vp_in.uv.xy));
    const float2 v3 = float2(vp_args.cubes.calculate_clamped_lod(vp_args.smp, vp_in.dir.xyz), vp_args.cubes.calculate_unclamped_lod(vp_args.smp, vp_in.dir.xyz));
    const float2 v4 = v2 + v3;
    const float4 v8 = float4(v4.x, v4.y, 1.0f, 1.0f);
    vp_out.color = v8;
    return vp_out;
}
