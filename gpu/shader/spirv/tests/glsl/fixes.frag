#version 450
// CTS regressions (bitácora 12d): textureProj with a vec4 coordinate (OpCompositeInsert operand
// order), depth compares with gradients, component-packed colour outputs, gl_ClipDistance read in
// the fragment stage and early fragment tests.
layout(early_fragment_tests) in;
layout(set = 0, binding = 0) uniform highp usampler2D tex;
layout(set = 0, binding = 1) uniform highp sampler2DShadow shadow;
layout(location = 0) in vec4 coord;
in float gl_ClipDistance[4];
layout(location = 0, component = 0) out vec2 colorXY;
layout(location = 0, component = 2) out float colorZ;
layout(location = 0, component = 3) out float colorW;
void main() {
    vec4 t = vec4(textureProjOffset(tex, coord, ivec2(7, -8)));
    float s = textureGrad(shadow, coord.xyz, vec2(coord.w, 0.0), vec2(0.0, 0.1));
    colorXY = t.xy;
    colorZ = s;
    colorW = gl_ClipDistance[2];
}
