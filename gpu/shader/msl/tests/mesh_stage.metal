#include <metal_stdlib>
using namespace metal;

struct VpVertex {
    float4 position [[position]];
    float2 uv [[user(loc0)]];
};

struct VpPrimitive {
    uint layer [[render_target_array_index]];
};

struct VpPayload {
    uint words[4];
};

[[mesh]] void vp_main(metal::mesh<VpVertex, VpPrimitive, 4, 2, topology::triangle> vp_mesh,
        const object_data VpPayload& vp_payload [[payload]],
        uint in_tid [[thread_index_in_threadgroup]]) {
    const float v2 = *(const object_data float*)((const object_data uchar*)&vp_payload);
    const float v4 = *(const object_data float*)((const object_data uchar*)&vp_payload + 4u);
    const uint v6 = in_tid & 1u;
    const float v7 = float(v6);
    const uint v8 = in_tid >> 1u;
    const float v9 = float(v8);
    const float2 v10 = float2(v7, v9);
    const float v11 = fma(v7, v4, v2);
    const float4 v13 = float4(v11, v9, float2(0.0f, 1.0f));
    vp_mesh.set_vertex(in_tid, VpVertex{v13, v10});
    const bool v15 = in_tid < 2u;
    if (v15) {
        vp_mesh.set_primitive(in_tid, VpPrimitive{0u});
        const uint v17 = in_tid * 3u;
        const uint v18 = in_tid + 1u;
        const uint v19 = in_tid + 2u;
        vp_mesh.set_index(v17, uchar(in_tid));
        const uint v20 = v17 + 1u;
        vp_mesh.set_index(v20, uchar(v18));
        const uint v21 = v17 + 2u;
        vp_mesh.set_index(v21, uchar(v19));
    }
    const bool v23 = in_tid == 0u;
    if (v23) {
        vp_mesh.set_primitive_count(2u);
    }
}
