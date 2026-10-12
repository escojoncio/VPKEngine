#version 450
// Forward lighting: combined samplers, a separate texture + sampler, a loop over lights with a
// break and a continue, a switch, helper functions with out parameters, and discard.
layout(set = 0, binding = 1) uniform sampler2D albedo;
layout(set = 0, binding = 2) uniform texture2D normalMap;
layout(set = 0, binding = 3) uniform sampler linearClamp;
layout(set = 0, binding = 4) uniform sampler2DArrayShadow shadowMap;
struct Light { vec4 posRange; vec4 color; };
layout(set = 0, binding = 5, std430) readonly buffer Lights { uint count; Light lights[]; } lb;
layout(set = 0, binding = 6) uniform Material { vec4 baseColor; float alphaCutoff; int mode; } mat;
layout(location = 0) in vec2 uv;
layout(location = 1) in vec3 normalWs;
layout(location = 2) flat in uint view;
layout(location = 3) in vec4 tint;
layout(location = 0) out vec4 color;
float attenuate(vec3 l, float range, out float ndl) {
    float d = length(l);
    ndl = max(dot(normalize(normalWs), l / d), 0.0);
    return clamp(1.0 - d / range, 0.0, 1.0);
}
void main() {
    vec4 a = texture(albedo, uv) * mat.baseColor * tint;
    if (a.a < mat.alphaCutoff) discard;
    vec3 n = texture(sampler2D(normalMap, linearClamp), uv * 2.0).xyz;
    vec3 sum = vec3(0.0);
    for (uint i = 0u; i < lb.count; ++i) {
        if (i >= 16u) break;
        Light L = lb.lights[i];
        if (L.color.a <= 0.0) continue;
        float ndl;
        float att = attenuate(L.posRange.xyz, L.posRange.w, ndl);
        sum += L.color.rgb * att * ndl;
    }
    float shadow = texture(shadowMap, vec4(uv, float(view), 0.5));
    switch (mat.mode) {
        case 0: color = vec4(a.rgb * sum * shadow, a.a); break;
        case 1: color = vec4(n * 0.5 + 0.5, 1.0); break;
        case 2:
        case 3: color = vec4(vec3(shadow), 1.0); break;
        default: color = a; break;
    }
}
