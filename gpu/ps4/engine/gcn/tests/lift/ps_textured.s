## stage pixel
## user_sgprs 2
## ps_input_addr 0x2
// Interpolated UV (attr0) and tint (attr1.x), texture through a resource table at s[0:1].
s_mov_b32 m0, s2
s_load_dwordx8 s[4:11], s[0:1], 0x0
s_load_dwordx4 s[12:15], s[0:1], 0x8
v_interp_p1_f32 v2, v0, attr0.x
v_interp_p2_f32 v2, v1, attr0.x
v_interp_p1_f32 v3, v0, attr0.y
v_interp_p2_f32 v3, v1, attr0.y
v_interp_p1_f32 v4, v0, attr1.x
v_interp_p2_f32 v4, v1, attr1.x
s_waitcnt lgkmcnt(0)
image_sample v[5:8], v[2:3], s[4:11], s[12:15] dmask:0xf
s_waitcnt vmcnt(0)
v_mul_f32 v5, v5, v4
v_mul_f32 v6, v6, v4
v_cvt_pkrtz_f16_f32 v0, v5, v6
v_cvt_pkrtz_f16_f32 v1, v7, v8
exp mrt0 v0, v0, v1, v1 done compr vm
s_endpgm
