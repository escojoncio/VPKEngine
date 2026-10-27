## stage pixel
## user_sgprs 12
## ps_input_addr 0x2
// gather4 with an explicit lod (_l, _lz), with a bias (_b), with a texel offset (_l_o), and compare gather4_c_lz.
v_interp_p1_f32 v2, v0, attr0.x
v_interp_p2_f32 v2, v1, attr0.x
v_interp_p1_f32 v3, v0, attr0.y
v_interp_p2_f32 v3, v1, attr0.y
v_mov_b32 v4, 1.0
image_gather4_l v[8:11], v[2:4], s[0:7], s[8:11] dmask:0x2
image_gather4_lz v[12:15], v[2:3], s[0:7], s[8:11] dmask:0x1
v_mov_b32 v16, 0.75
v_mov_b32 v17, v2
v_mov_b32 v18, v3
image_gather4_b v[19:22], v[16:18], s[0:7], s[8:11] dmask:0x4
v_mov_b32 v23, 0x00000102
v_mov_b32 v24, v2
v_mov_b32 v25, v3
v_mov_b32 v26, 2.0
image_gather4_l_o v[27:30], v[23:26], s[0:7], s[8:11] dmask:0x1
s_waitcnt vmcnt(0)
v_add_f32 v8, v8, v12
v_add_f32 v8, v8, v19
v_add_f32 v8, v8, v27
exp mrt0 v8, v9, v10, v11 done vm
s_endpgm
