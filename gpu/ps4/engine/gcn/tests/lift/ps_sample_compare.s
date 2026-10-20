## stage pixel
## user_sgprs 12
## ps_input_addr 0x2
## tex_depth
// Depth compare sampling keeps its bias, its gradients and its lod clamp.
v_interp_p1_f32 v2, v0, attr0.x
v_interp_p2_f32 v2, v1, attr0.x
v_interp_p1_f32 v3, v0, attr0.y
v_interp_p2_f32 v3, v1, attr0.y
v_mov_b32 v8, 0.5
v_mov_b32 v9, 0.75
v_mov_b32 v10, v2
v_mov_b32 v11, v3
image_sample_c_b v12, v[8:11], s[0:7], s[8:11] dmask:0x1
v_mov_b32 v20, 0.01
v_mov_b32 v21, 0.02
v_mov_b32 v22, 0.03
v_mov_b32 v23, 0.04
v_mov_b32 v24, 0.5
v_mov_b32 v25, v2
v_mov_b32 v26, v3
image_sample_c_d v13, v[20:26], s[0:7], s[8:11] dmask:0x1
v_mov_b32 v30, 0.5
v_mov_b32 v31, v2
v_mov_b32 v32, v3
v_mov_b32 v33, 2.0
image_sample_c_cl v14, v[30:33], s[0:7], s[8:11] dmask:0x1
image_sample_c_lz v15, v[30:32], s[0:7], s[8:11] dmask:0x1
s_waitcnt vmcnt(0)
exp mrt0 v12, v13, v14, v15 done vm
s_endpgm
