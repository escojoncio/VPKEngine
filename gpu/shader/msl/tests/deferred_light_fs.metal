#include <metal_stdlib>
using namespace metal;

struct VpArgs {
    constant uchar* light;
    texture2d<float> gbuf_albedo;
    texture2d<float> gbuf_normal;
    texture2d<float> gbuf_depth;
    sampler point_clamp;
};

struct VpOut {
    half4 lit [[color(3)]];
};

fragment VpOut vp_main(constant VpArgs& vp_args [[buffer(0)]],
        float4 in_frag_pos [[position]]) {
    VpOut vp_out;
    const float2 v1 = in_frag_pos.xy;
    const uint2 v2 = uint2(v1);
    const float4 v3 = vp_args.gbuf_albedo.read(v2);
    const float4 v4 = vp_args.gbuf_normal.read(v2);
    const uint2 v5 = uint2(vp_args.gbuf_depth.get_width(), vp_args.gbuf_depth.get_height());
    const float2 v6 = float2(v5);
    const float2 v7 = v1 / v6;
    const float4 v9 = vp_args.gbuf_depth.sample(vp_args.point_clamp, v7, level(0.0f));
    const float3 v10 = v4.xyz;
    const float3 v13 = fma(v10, float3(2.0f), float3(-1.0f));
    const float3 v14 = normalize(v13);
    const float4 v16 = *(constant float4*)(vp_args.light);
    const float3 v17 = v16.xyz;
    const float v18 = dot(v14, v17);
    const float v19 = max(v18, 0.0f);
    const float v21 = v19 * v16.w;
    const float3 v24 = v3.xyz;
    const float3 v25 = v24 * float3(*(constant packed_float3*)(vp_args.light + 16u));
    const float3 v26 = float3(v21);
    const float3 v27 = v25 * v26;
    const float4 v29 = *(constant float4*)(vp_args.light + 32u);
    const float v32 = v9.x * v29.w;
    const float v33 = -v32;
    const float v34 = exp2(v33);
    const float3 v35 = v29.xyz;
    const float3 v36 = float3(v34);
    const float3 v37 = mix(v35, v27, v36);
    const float4 v39 = float4(v37, v3.w);
    const half4 v40 = half4(v39);
    vp_out.lit = v40;
    return vp_out;
}
