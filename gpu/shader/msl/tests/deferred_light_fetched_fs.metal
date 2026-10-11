#include <metal_stdlib>
using namespace metal;

struct VpArgs {
    constant uchar* light;
};

struct VpOut {
    half4 lit [[color(3)]];
};

fragment VpOut vp_main(constant VpArgs& vp_args [[buffer(0)]],
        float4 in_frag_pos [[position]],
        float4 vp_fb0 [[color(0)]],
        float4 vp_fb1 [[color(1)]],
        float4 vp_fb2 [[color(2)]]) {
    VpOut vp_out;
    const float3 v10 = vp_fb1.xyz;
    const float3 v13 = fma(v10, float3(2.0f), float3(-1.0f));
    const float3 v14 = normalize(v13);
    const float4 v16 = *(constant float4*)(vp_args.light);
    const float3 v17 = v16.xyz;
    const float v18 = dot(v14, v17);
    const float v19 = max(v18, 0.0f);
    const float v21 = v19 * v16.w;
    const float3 v24 = vp_fb0.xyz;
    const float3 v25 = v24 * float3(*(constant packed_float3*)(vp_args.light + 16u));
    const float3 v26 = float3(v21);
    const float3 v27 = v25 * v26;
    const float4 v29 = *(constant float4*)(vp_args.light + 32u);
    const float v32 = vp_fb2.x * v29.w;
    const float v33 = -v32;
    const float v34 = exp2(v33);
    const float3 v35 = v29.xyz;
    const float3 v36 = float3(v34);
    const float3 v37 = mix(v35, v27, v36);
    const float4 v39 = float4(v37, vp_fb0.w);
    const half4 v40 = half4(v39);
    vp_out.lit = v40;
    return vp_out;
}
