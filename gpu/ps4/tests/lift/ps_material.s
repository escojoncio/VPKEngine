## stage pixel
## user_sgprs 8
## ps_input_addr 0x2
// A material of the per-draw bench (perdraw_tests.cpp): three textures and two samplers through the resource table at s[0:1],
// a constant buffer V# in s[4:7] and two through the table at s[2:3]. The literal is what the test changes to make variants.
s_mov_b32 m0, s8
v_interp_p1_f32 v2, v0, attr0.x
v_interp_p2_f32 v2, v1, attr0.x
v_interp_p1_f32 v3, v0, attr0.y
v_interp_p2_f32 v3, v1, attr0.y
s_load_dwordx8 s[12:19], s[0:1], 0
s_load_dwordx4 s[20:23], s[0:1], 24
s_waitcnt lgkmcnt(0)
image_sample v[4:7], v[2:3], s[12:19], s[20:23] dmask:0xf
s_load_dwordx8 s[12:19], s[0:1], 8
s_load_dwordx4 s[24:27], s[0:1], 28
s_waitcnt lgkmcnt(0)
image_sample v[8:11], v[2:3], s[12:19], s[24:27] dmask:0xf
s_load_dwordx8 s[12:19], s[0:1], 16
s_waitcnt lgkmcnt(0)
image_sample v[12:15], v[2:3], s[12:19], s[20:23] dmask:0xf
s_waitcnt vmcnt(0)
s_buffer_load_dwordx4 s[28:31], s[4:7], 0x0
s_load_dwordx4 s[32:35], s[2:3], 0
s_load_dwordx4 s[36:39], s[2:3], 4
s_waitcnt lgkmcnt(0)
s_buffer_load_dwordx4 s[40:43], s[32:35], 0x0
s_buffer_load_dwordx4 s[44:47], s[36:39], 0x0
s_waitcnt lgkmcnt(0)
v_mul_f32 v4, s28, v4
v_mac_f32 v4, s40, v8
v_mac_f32 v4, s44, v12
v_mov_b32 v16, 0x3f123456
v_add_f32 v4, v4, v16
v_cvt_pkrtz_f16_f32 v0, v4, v5
v_cvt_pkrtz_f16_f32 v1, v6, v7
exp mrt0 v0, v0, v1, v1 done compr vm
s_endpgm
