#version 450
// CTS regressions (bitácora 12d): component-packed vertex inputs, a one-byte attribute in a
// 1-byte-stride binding, and gl_ClipDistance mirrored for the fragment stage.
layout(location = 0) in vec4 pos;
layout(location = 1, component = 0) in vec2 ab;
layout(location = 1, component = 2) in float c;
layout(location = 1, component = 3) in float d;
layout(location = 2) in uint id8;
layout(location = 0) out vec4 coord;
layout(location = 1) flat out uint idOut;
out gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[4]; };
void main() {
    coord = vec4(ab, c, d);
    idOut = id8;
    gl_Position = pos;
    for (int i = 0; i < 4; ++i) gl_ClipDistance[i] = pos[i];
}
