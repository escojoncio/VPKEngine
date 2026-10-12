#version 450
#extension GL_EXT_multiview : require
// Stereo mesh: per-view matrices, instanced offsets, packed normals.
layout(set = 0, binding = 0) uniform Camera {
    mat4 viewProj[2];
    vec4 eyeOffset[2];
} cam;
layout(set = 1, binding = 0) uniform Object {
    mat4 model;
    layout(row_major) mat3x4 normalMatrix;
    float scale;
} obj;
layout(push_constant) uniform Push { vec4 tint; uint flags; } pc;
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec4 inNormal;   // snorm8x4 in the vertex buffer
layout(location = 2) in vec2 inUv;       // half2 in the vertex buffer
layout(location = 3) in vec4 inOffset;   // per instance
layout(location = 0) out vec2 uv;
layout(location = 1) out vec3 normalWs;
layout(location = 2) flat out uint view;
layout(location = 3) out vec4 tint;
void main() {
    vec4 world = obj.model * vec4(inPos * obj.scale, 1.0) + vec4(inOffset.xyz, 0.0);
    gl_Position = cam.viewProj[gl_ViewIndex] * (world + cam.eyeOffset[gl_ViewIndex]);
    normalWs = normalize((vec4(inNormal.xyz, 0.0) * obj.normalMatrix).xyz);
    uv = inUv;
    view = gl_ViewIndex;
    tint = (pc.flags & 1u) != 0u ? pc.tint : vec4(1.0);
}
