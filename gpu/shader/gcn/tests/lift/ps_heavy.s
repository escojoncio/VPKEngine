## stage pixel
## user_sgprs 2
## ps_input_addr 0x2
// 12 textures and 6 buffers through a resource table at s[0:1] (bench only).
s_mov_b32 m0, s2
v_interp_p1_f32 v2, v0, attr0.x
v_interp_p2_f32 v2, v1, attr0.x
v_interp_p1_f32 v3, v0, attr0.y
v_interp_p2_f32 v3, v1, attr0.y
v_mov_b32 v12, 0
v_mov_b32 v14, 0
s_load_dwordx4 s[12:15], s[0:1], 0x60
s_load_dwordx8 s[4:11], s[0:1], 0
s_waitcnt lgkmcnt(0)
image_sample v[8:11], v[2:3], s[4:11], s[12:15] dmask:0xf
s_waitcnt vmcnt(0)
v_add_f32 v12, v8, v12
s_load_dwordx8 s[4:11], s[0:1], 8
s_waitcnt lgkmcnt(0)
image_sample v[8:11], v[2:3], s[4:11], s[12:15] dmask:0xf
s_waitcnt vmcnt(0)
v_add_f32 v12, v8, v12
s_load_dwordx8 s[4:11], s[0:1], 16
s_waitcnt lgkmcnt(0)
image_sample v[8:11], v[2:3], s[4:11], s[12:15] dmask:0xf
s_waitcnt vmcnt(0)
v_add_f32 v12, v8, v12
s_load_dwordx8 s[4:11], s[0:1], 24
s_waitcnt lgkmcnt(0)
image_sample v[8:11], v[2:3], s[4:11], s[12:15] dmask:0xf
s_waitcnt vmcnt(0)
v_add_f32 v12, v8, v12
s_load_dwordx8 s[4:11], s[0:1], 32
s_waitcnt lgkmcnt(0)
image_sample v[8:11], v[2:3], s[4:11], s[12:15] dmask:0xf
s_waitcnt vmcnt(0)
v_add_f32 v12, v8, v12
s_load_dwordx8 s[4:11], s[0:1], 40
s_waitcnt lgkmcnt(0)
image_sample v[8:11], v[2:3], s[4:11], s[12:15] dmask:0xf
s_waitcnt vmcnt(0)
v_add_f32 v12, v8, v12
s_load_dwordx8 s[4:11], s[0:1], 48
s_waitcnt lgkmcnt(0)
image_sample v[8:11], v[2:3], s[4:11], s[12:15] dmask:0xf
s_waitcnt vmcnt(0)
v_add_f32 v12, v8, v12
s_load_dwordx8 s[4:11], s[0:1], 56
s_waitcnt lgkmcnt(0)
image_sample v[8:11], v[2:3], s[4:11], s[12:15] dmask:0xf
s_waitcnt vmcnt(0)
v_add_f32 v12, v8, v12
s_load_dwordx8 s[4:11], s[0:1], 64
s_waitcnt lgkmcnt(0)
image_sample v[8:11], v[2:3], s[4:11], s[12:15] dmask:0xf
s_waitcnt vmcnt(0)
v_add_f32 v12, v8, v12
s_load_dwordx8 s[4:11], s[0:1], 72
s_waitcnt lgkmcnt(0)
image_sample v[8:11], v[2:3], s[4:11], s[12:15] dmask:0xf
s_waitcnt vmcnt(0)
v_add_f32 v12, v8, v12
s_load_dwordx8 s[4:11], s[0:1], 80
s_waitcnt lgkmcnt(0)
image_sample v[8:11], v[2:3], s[4:11], s[12:15] dmask:0xf
s_waitcnt vmcnt(0)
v_add_f32 v12, v8, v12
s_load_dwordx8 s[4:11], s[0:1], 88
s_waitcnt lgkmcnt(0)
image_sample v[8:11], v[2:3], s[4:11], s[12:15] dmask:0xf
s_waitcnt vmcnt(0)
v_add_f32 v12, v8, v12
s_load_dwordx4 s[16:19], s[0:1], 100
s_waitcnt lgkmcnt(0)
buffer_load_dword v13, v14, s[16:19], 0 idxen
s_waitcnt vmcnt(0)
v_add_f32 v12, v13, v12
s_load_dwordx4 s[16:19], s[0:1], 104
s_waitcnt lgkmcnt(0)
buffer_load_dword v13, v14, s[16:19], 0 idxen
s_waitcnt vmcnt(0)
v_add_f32 v12, v13, v12
s_load_dwordx4 s[16:19], s[0:1], 108
s_waitcnt lgkmcnt(0)
buffer_load_dword v13, v14, s[16:19], 0 idxen
s_waitcnt vmcnt(0)
v_add_f32 v12, v13, v12
s_load_dwordx4 s[16:19], s[0:1], 112
s_waitcnt lgkmcnt(0)
buffer_load_dword v13, v14, s[16:19], 0 idxen
s_waitcnt vmcnt(0)
v_add_f32 v12, v13, v12
s_load_dwordx4 s[16:19], s[0:1], 116
s_waitcnt lgkmcnt(0)
buffer_load_dword v13, v14, s[16:19], 0 idxen
s_waitcnt vmcnt(0)
v_add_f32 v12, v13, v12
s_load_dwordx4 s[16:19], s[0:1], 120
s_waitcnt lgkmcnt(0)
buffer_load_dword v13, v14, s[16:19], 0 idxen
s_waitcnt vmcnt(0)
v_add_f32 v12, v13, v12
v_cvt_pkrtz_f16_f32 v0, v12, v12
exp mrt0 v0, v0, v0, v0 done compr vm
s_endpgm
