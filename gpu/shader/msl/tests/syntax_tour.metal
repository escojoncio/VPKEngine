#include <metal_stdlib>
using namespace metal;

struct VpArgs {
    device uchar* counters;
    texture2d_array<half> tex;
    texturecube_array<float> cubes;
    depthcube<float> depth_map;
    sampler smp;
    texture2d<uint, access::read_write> img;
};

struct VpIn {
    interpolant<float2, interpolation::perspective> uv [[user(loc0)]];
    interpolant<float2, interpolation::no_perspective> ndc [[user(loc1)]];
    uint id [[user(loc2), flat]];
};

struct VpOut {
    float4 color [[color(0)]];
    float depth [[depth(any)]];
};

static float fn_lerp3(float v0, float v1, float v2) {
    const float v3 = mix(v0, v1, v2);
    return v3;
}

fragment VpOut vp_main(VpIn vp_in [[stage_in]],
        constant VpArgs& vp_args [[buffer(0)]],
        uint in_sid [[sample_id]],
        bool in_face [[front_facing]],
        float4 in_frag [[position]]) {
    VpOut vp_out;
    half4 l0{};
    const float2 v5 = vp_in.uv.interpolate_at_centroid();
    const float3 v7 = float3(v5, 0.5f);
    const half4 v9 = vp_args.tex.sample(vp_args.smp, v7.xy, uint(rint(v7.z)), bias(1.0f));
    const float4 v10 = float4(v7, 1.0f);
    const float3 v11 = float3(0.5f);
    const float4 v12 = vp_args.cubes.sample(vp_args.smp, v10.xyz, uint(rint(v10.w)), gradientcube(v11, v11));
    const uint2 v14 = uint2(0u);
    const uint4 v15 = vp_args.img.read(v14);
    vp_args.img.write(v15, v14);
    const uint v17 = vp_args.cubes.get_num_mip_levels();
    const uint v20 = vp_in.id << 2u;
    atomic_fetch_or_explicit((device atomic_uint*)(vp_args.counters + v20), v17, memory_order_relaxed);
    const float v23 = fn_lerp3(v12.x, 1.0f, 0.5f);
    const float4 v24 = float4(float4(-0.0f, 1e-07f, 3.4028235e+38f, as_type<float>(0x7fc00000u)).x, v12.y, float4(-0.0f, 1e-07f, 3.4028235e+38f, as_type<float>(0x7fc00000u)).z, v12.w);
    float4 v25 = v24;
    v25.w = v23;
    const bool4 v26 = (as_type<uint4>(v25) & 0x7fffffffu) > 0x7f800000u;
    const float4 v27 = select(v25, v12, v26);
    const float4 v29 = float4(v9);
    const float4 v30 = in_face ? v27 : v29;
    const float2 v31 = vp_in.ndc.interpolate_at_offset(v5 + 0.5f);
    const uint2 v32 = as_type<uint2>(v31);
    const uint v33 = extract_bits(vp_in.id, 2u, 2u);
    const uint v34 = 31u - clz(v33);
    const uint v36 = v34 + v32.y;
    const float v37 = float(v36);
    const float v38 = fwidth(v37);
    vp_out.color = v30;
    vp_out.depth = v38;
    l0 = half4(as_type<half>(ushort(0x7c00)), 0.5h, -2.0h, 65504.0h);
    const bool v41 = v38 < 0.0f;
    const bool v42 = false || v41;
    if (v42) {
        discard_fragment();
    } else {
        return vp_out;
    }
    return vp_out;
}
