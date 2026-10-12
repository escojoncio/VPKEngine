#version 450
// Deferred resolve in the same render pass: input attachments, do-while, phis after -O.
layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput gAlbedo;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput gNormal;
layout(location = 0) out vec4 color;
layout(location = 1) out uvec4 ids;
void main() {
    vec4 al = subpassLoad(gAlbedo);
    vec3 n = subpassLoad(gNormal).xyz * 2.0 - 1.0;
    float acc = 0.0;
    int k = 0;
    do { acc += pow(max(n.z, 0.0), float(k + 1)) * atan(n.y, n.x); ++k; } while (k < 4);
    color = vec4(al.rgb * acc, 1.0);
    ids = uvec4(packHalf2x16(n.xy), packUnorm4x8(al), floatBitsToUint(acc), uint(k));
}
