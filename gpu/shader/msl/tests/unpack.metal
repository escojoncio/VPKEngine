#include <metal_stdlib>
using namespace metal;
#ifndef VP_PSZ
#define VP_PSZ point_size
#endif

struct VpOut {
    float4 color [[color(0)]];
};

fragment VpOut vp_main() {
    VpOut vp_out;
    const float4 v1 = unpack_snorm4x8_to_float(2164195073u);
    const float4 v2 = unpack_unorm4x8_to_float(2164195073u);
    const float2 v3 = unpack_unorm2x16_to_float(2164195073u);
    const float2 v4 = unpack_snorm2x16_to_float(2164195073u);
    const float4 v5 = v1 + v2;
    const float2 v6 = v3 + v4;
    const float4 v7 = float4(v6, v6);
    const float4 v8 = v5 + v7;
    vp_out.color = v8;
    return vp_out;
}
