#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpArgs {
    device uchar* data;
    const device uchar* src;
    ulong bs0;
    ulong bs1;
};

[[max_total_threads_per_threadgroup(64)]] kernel void vp_main(constant VpArgs& vp_args [[buffer(0)]],
        uint3 in_gid [[thread_position_in_grid]]) {
    const uint v3 = in_gid.x << 4u;
    float4 v4 = float4(0);
    if (ulong(v3) + 16ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v4 = float4(*(device packed_float4*)(vp_args.data + v3));
    else {
        if (ulong(v3) + 4ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v4.x = *(device float*)(vp_args.data + v3 + 0u);
        if (ulong(v3) + 8ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v4.y = *(device float*)(vp_args.data + v3 + 4u);
        if (ulong(v3) + 12ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v4.z = *(device float*)(vp_args.data + v3 + 8u);
        if (ulong(v3) + 16ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v4.w = *(device float*)(vp_args.data + v3 + 12u);
    }
    const uint v6 = v3 + 64u;
    float2 v7 = float2(0);
    if (ulong(v6) + 8ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v7 = float2(*(device packed_float2*)(vp_args.data + v6));
    else {
        if (ulong(v6) + 4ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v7.x = *(device float*)(vp_args.data + v6 + 0u);
        if (ulong(v6) + 8ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v7.y = *(device float*)(vp_args.data + v6 + 4u);
    }
    const uint v9 = v3 + 128u;
    uint3 v10 = uint3(0);
    if (ulong(v9) + 12ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v10 = uint3(*(device packed_uint3*)(vp_args.data + v9));
    else {
        if (ulong(v9) + 4ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v10.x = *(device uint*)(vp_args.data + v9 + 0u);
        if (ulong(v9) + 8ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v10.y = *(device uint*)(vp_args.data + v9 + 4u);
        if (ulong(v9) + 12ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v10.z = *(device uint*)(vp_args.data + v9 + 8u);
    }
    const uint v12 = v3 + 192u;
    uint v13 = uint(0);
    if (ulong(v12) + 4ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v13 = *(device uint*)(vp_args.data + v12);
    const float v17 = as_type<float>(v10.z);
    const float v18 = v4.x + v7.y;
    const float v19 = v18 + v17;
    const float v20 = as_type<float>(v13);
    const float v21 = v19 + v20;
    const float4 v22 = float4(v21);
    const uint v24 = v3 + 256u;
    int4 v25 = int4(0);
    if (ulong(v24) + 16ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v25 = int4(*(device packed_int4*)(vp_args.data + v24));
    else {
        if (ulong(v24) + 4ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v25.x = *(device int*)(vp_args.data + v24 + 0u);
        if (ulong(v24) + 8ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v25.y = *(device int*)(vp_args.data + v24 + 4u);
        if (ulong(v24) + 12ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v25.z = *(device int*)(vp_args.data + v24 + 8u);
        if (ulong(v24) + 16ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v25.w = *(device int*)(vp_args.data + v24 + 12u);
    }
    float4 v38 = float4(0);
    if (ulong(v3) + 16ul <= ((vp_args.bs1 + 3ul) & ~3ul)) v38 = float4(*(const device packed_float4*)(vp_args.src + v3));
    else {
        if (ulong(v3) + 4ul <= ((vp_args.bs1 + 3ul) & ~3ul)) v38.x = *(const device float*)(vp_args.src + v3 + 0u);
        if (ulong(v3) + 8ul <= ((vp_args.bs1 + 3ul) & ~3ul)) v38.y = *(const device float*)(vp_args.src + v3 + 4u);
        if (ulong(v3) + 12ul <= ((vp_args.bs1 + 3ul) & ~3ul)) v38.z = *(const device float*)(vp_args.src + v3 + 8u);
        if (ulong(v3) + 16ul <= ((vp_args.bs1 + 3ul) & ~3ul)) v38.w = *(const device float*)(vp_args.src + v3 + 12u);
    }
    float4 v40 = float4(0);
    if (ulong(0u) + 16ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v40 = float4(*(device packed_float4*)(vp_args.data));
    else {
        if (ulong(0u) + 4ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v40.x = *(device float*)(vp_args.data + 0u);
        if (ulong(0u) + 8ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v40.y = *(device float*)(vp_args.data + 4u);
        if (ulong(0u) + 12ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v40.z = *(device float*)(vp_args.data + 8u);
        if (ulong(0u) + 16ul <= ((vp_args.bs0 + 3ul) & ~3ul)) v40.w = *(device float*)(vp_args.data + 12u);
    }
    const float v42 = as_type<float>(v25.w);
    const float v45 = v42 + v38.y;
    const float v46 = v45 + v40.z;
    const float2 v29 = float2(v46);
    const float3 v30 = float3(v46);
    const uint3 v31 = as_type<uint3>(v30);
    const int4 v32 = int4(v25.w);
    if (ulong(v6) + 8ul <= ((vp_args.bs0 + 3ul) & ~3ul)) *(device packed_float2*)(vp_args.data + v6) = v29;
    else {
        if (ulong(v6) + 4ul <= ((vp_args.bs0 + 3ul) & ~3ul)) *(device float*)(vp_args.data + v6 + 0u) = (v29).x;
        if (ulong(v6) + 8ul <= ((vp_args.bs0 + 3ul) & ~3ul)) *(device float*)(vp_args.data + v6 + 4u) = (v29).y;
    }
    if (ulong(v9) + 12ul <= ((vp_args.bs0 + 3ul) & ~3ul)) *(device packed_uint3*)(vp_args.data + v9) = v31;
    else {
        if (ulong(v9) + 4ul <= ((vp_args.bs0 + 3ul) & ~3ul)) *(device uint*)(vp_args.data + v9 + 0u) = (v31).x;
        if (ulong(v9) + 8ul <= ((vp_args.bs0 + 3ul) & ~3ul)) *(device uint*)(vp_args.data + v9 + 4u) = (v31).y;
        if (ulong(v9) + 12ul <= ((vp_args.bs0 + 3ul) & ~3ul)) *(device uint*)(vp_args.data + v9 + 8u) = (v31).z;
    }
    if (ulong(v24) + 16ul <= ((vp_args.bs0 + 3ul) & ~3ul)) *(device packed_int4*)(vp_args.data + v24) = v32;
    else {
        if (ulong(v24) + 4ul <= ((vp_args.bs0 + 3ul) & ~3ul)) *(device int*)(vp_args.data + v24 + 0u) = (v32).x;
        if (ulong(v24) + 8ul <= ((vp_args.bs0 + 3ul) & ~3ul)) *(device int*)(vp_args.data + v24 + 4u) = (v32).y;
        if (ulong(v24) + 12ul <= ((vp_args.bs0 + 3ul) & ~3ul)) *(device int*)(vp_args.data + v24 + 8u) = (v32).z;
        if (ulong(v24) + 16ul <= ((vp_args.bs0 + 3ul) & ~3ul)) *(device int*)(vp_args.data + v24 + 12u) = (v32).w;
    }
    if (ulong(0u) + 8ul <= ((vp_args.bs0 + 3ul) & ~3ul)) *(device packed_float2*)(vp_args.data) = v29;
    else {
        if (ulong(0u) + 4ul <= ((vp_args.bs0 + 3ul) & ~3ul)) *(device float*)(vp_args.data + 0u) = (v29).x;
        if (ulong(0u) + 8ul <= ((vp_args.bs0 + 3ul) & ~3ul)) *(device float*)(vp_args.data + 4u) = (v29).y;
    }
    if (ulong(v3) + 16ul <= ((vp_args.bs0 + 3ul) & ~3ul)) *(device packed_float4*)(vp_args.data + v3) = v22;
    else {
        if (ulong(v3) + 4ul <= ((vp_args.bs0 + 3ul) & ~3ul)) *(device float*)(vp_args.data + v3 + 0u) = (v22).x;
        if (ulong(v3) + 8ul <= ((vp_args.bs0 + 3ul) & ~3ul)) *(device float*)(vp_args.data + v3 + 4u) = (v22).y;
        if (ulong(v3) + 12ul <= ((vp_args.bs0 + 3ul) & ~3ul)) *(device float*)(vp_args.data + v3 + 8u) = (v22).z;
        if (ulong(v3) + 16ul <= ((vp_args.bs0 + 3ul) & ~3ul)) *(device float*)(vp_args.data + v3 + 12u) = (v22).w;
    }
}
