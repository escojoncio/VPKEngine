## stage pixel
## user_sgprs 12
## ps_input_addr 0x2
// Sampling with a lod clamp (_cl): implicit lod, with a bias, with gradients. The clamp is a minimum lod.
v_interp_p1_f32 v2, v0, attr0.x
v_interp_p2_f32 v2, v1, attr0.x
v_interp_p1_f32 v3, v0, attr0.y
v_interp_p2_f32 v3, v1, attr0.y
v_mov_b32 v4, 2.0
v_mov_b32 v8, 0.5
v_mov_b32 v9, v2
v_mov_b32 v10, v3
v_mov_b32 v11, 3.0
image_sample_cl v[12:15], v[2:4], s[0:7], s[8:11] dmask:0xf
image_sample_b_cl v[16:19], v[8:11], s[0:7], s[8:11] dmask:0xf
v_mov_b32 v20, 0.01
v_mov_b32 v21, 0.02
v_mov_b32 v22, 0.03
v_mov_b32 v23, 0.04
v_mov_b32 v24, v2
v_mov_b32 v25, v3
v_mov_b32 v26, 1.5
image_sample_d_cl v[27:30], v[20:26], s[0:7], s[8:11] dmask:0xf
s_waitcnt vmcnt(0)
v_add_f32 v12, v12, v16
v_add_f32 v12, v12, v27
exp mrt0 v12, v13, v14, v15 done vm
s_endpgm
