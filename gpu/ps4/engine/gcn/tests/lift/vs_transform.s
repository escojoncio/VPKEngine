## stage vertex
## user_sgprs 8
// Vertex buffer V# through a table at s[0:1]; constants (two matrix rows) from the V# in s[4:7].
s_load_dwordx4 s[8:11], s[0:1], 0x0
s_buffer_load_dwordx4 s[12:15], s[4:7], 0x0
s_buffer_load_dwordx4 s[16:19], s[4:7], 0x4
s_waitcnt lgkmcnt(0)
buffer_load_format_xyzw v[4:7], v0, s[8:11], 0 idxen
s_waitcnt vmcnt(0)
v_mul_f32 v8, s12, v4
v_mac_f32 v8, s13, v5
v_mac_f32 v8, s14, v6
v_mac_f32 v8, s15, v7
v_mul_f32 v9, s16, v4
v_mac_f32 v9, s17, v5
v_mac_f32 v9, s18, v6
v_mac_f32 v9, s19, v7
v_mov_b32 v10, 0
v_mov_b32 v11, 1.0
exp pos0 v8, v9, v10, v11 done
exp param0 v4, v5, v6, v7
s_endpgm
