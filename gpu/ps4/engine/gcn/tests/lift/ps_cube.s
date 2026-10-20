## stage pixel
## user_sgprs 12
## ps_input_addr 0x2
## tex_dim cube
// A cube map: plain sample, with face-space gradients (exact direction derivatives) and with a texel offset.
v_interp_p1_f32 v2, v0, attr0.x
v_interp_p2_f32 v2, v1, attr0.x
v_interp_p1_f32 v3, v0, attr0.y
v_interp_p2_f32 v3, v1, attr0.y
v_mov_b32 v4, 2.0
image_sample v[8:11], v[2:4], s[0:7], s[8:11] dmask:0xf
v_mov_b32 v12, 0.01
v_mov_b32 v13, 0.02
v_mov_b32 v14, 0.03
v_mov_b32 v15, 0.04
v_mov_b32 v16, v2
v_mov_b32 v17, v3
v_mov_b32 v18, 4.0
image_sample_d v[19:22], v[12:18], s[0:7], s[8:11] dmask:0xf
v_mov_b32 v23, 0x00000102
v_mov_b32 v24, v2
v_mov_b32 v25, v3
v_mov_b32 v26, 1.0
image_sample_o v[27:30], v[23:26], s[0:7], s[8:11] dmask:0xf
s_waitcnt vmcnt(0)
v_add_f32 v8, v8, v19
v_add_f32 v8, v8, v27
exp mrt0 v8, v9, v10, v11 done vm
s_endpgm
