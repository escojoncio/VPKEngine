## stage compute
## user_sgprs 12
// Implicit-lod sampling and image_get_lod in a compute shader: derivatives from the lanes of the quad.
v_cvt_f32_u32 v1, v0
v_mul_f32 v2, 0.015625, v1
v_and_b32 v3, 7, v0
v_cvt_f32_u32 v3, v3
v_mul_f32 v3, 0.125, v3
image_sample v[4:7], v[2:3], s[0:7], s[8:11] dmask:0xf
v_mov_b32 v8, 0.5
v_mov_b32 v9, v2
v_mov_b32 v10, v3
image_sample_b v[11:14], v[8:10], s[0:7], s[8:11] dmask:0xf
image_get_lod v[15:16], v[2:3], s[0:7], s[8:11] dmask:0x3
s_waitcnt vmcnt(0)
v_add_f32 v4, v4, v11
v_add_f32 v4, v4, v15
v_lshlrev_b32 v20, 2, v0
buffer_store_dword v4, v20, s[8:11], 0 offen
s_endpgm
