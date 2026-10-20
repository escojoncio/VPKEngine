## stage pixel
## user_sgprs 12
## ps_input_addr 0x2
## tex_depth
// Compare gather at lod 0 and at an explicit lod (four compare samples at the texel centres).
v_interp_p1_f32 v2, v0, attr0.x
v_interp_p2_f32 v2, v1, attr0.x
v_interp_p1_f32 v3, v0, attr0.y
v_interp_p2_f32 v3, v1, attr0.y
v_mov_b32 v4, 0.5
v_mov_b32 v5, v2
v_mov_b32 v6, v3
image_gather4_c v[8:11], v[4:6], s[0:7], s[8:11] dmask:0x1
v_mov_b32 v12, 0.5
v_mov_b32 v13, v2
v_mov_b32 v14, v3
v_mov_b32 v15, 1.0
image_gather4_c_l v[16:19], v[12:15], s[0:7], s[8:11] dmask:0x1
s_waitcnt vmcnt(0)
v_add_f32 v8, v8, v16
v_add_f32 v9, v9, v17
v_add_f32 v10, v10, v18
v_add_f32 v11, v11, v19
exp mrt0 v8, v9, v10, v11 done vm
s_endpgm
