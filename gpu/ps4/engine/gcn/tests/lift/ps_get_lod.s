## stage pixel
## user_sgprs 12
## ps_input_addr 0x2
// image_get_lod: the lod a sample would use (clamped) and the computed one, as colour.
v_interp_p1_f32 v2, v0, attr0.x
v_interp_p2_f32 v2, v1, attr0.x
v_interp_p1_f32 v3, v0, attr0.y
v_interp_p2_f32 v3, v1, attr0.y
image_get_lod v[4:5], v[2:3], s[0:7], s[8:11] dmask:0x3
s_waitcnt vmcnt(0)
v_mov_b32 v6, 0
v_mov_b32 v7, 1.0
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
