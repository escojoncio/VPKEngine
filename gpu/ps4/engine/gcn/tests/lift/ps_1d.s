## stage pixel
## user_sgprs 12
## ps_input_addr 0x2
## tex_dim d1
// A 1D texture: a texel offset moves the coordinate; an explicit lod is dropped (Metal's 1D textures take none).
v_interp_p1_f32 v3, v0, attr0.x
v_interp_p2_f32 v3, v1, attr0.x
v_mov_b32 v2, 0x00000003
image_sample_o v[8:11], v[2:3], s[0:7], s[8:11] dmask:0xf
v_mov_b32 v12, v3
v_mov_b32 v13, 1.0
image_sample_l v[14:17], v[12:13], s[0:7], s[8:11] dmask:0xf
s_waitcnt vmcnt(0)
v_add_f32 v8, v8, v14
exp mrt0 v8, v9, v10, v11 done vm
s_endpgm
