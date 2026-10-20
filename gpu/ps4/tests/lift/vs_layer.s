## stage vertex
## user_sgprs 0
// A vertex shader of a two-eye pass: position in pos0, the eye (render target array index) in z of pos1.
v_mov_b32 v1, 0
v_mov_b32 v2, 0
v_cvt_f32_u32 v3, v0
v_mov_b32 v4, 1.0
exp pos0 v3, v1, v1, v4
exp pos1 off, off, v3, off done
s_endpgm
