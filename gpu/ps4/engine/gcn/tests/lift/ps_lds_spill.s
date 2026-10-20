## stage pixel
## user_sgprs 2
## ps_input_addr 0x2
// Sony's register spill to LDS in a pixel shader: address = lane index * 4 + offset; also a byte
// store into a spilled dword. Lifts to per-lane private memory.
v_interp_p1_f32 v2, v0, attr0.x
v_interp_p2_f32 v2, v1, attr0.x
v_mbcnt_lo_u32_b32_e64 v3, -1, 0
v_mbcnt_hi_u32_b32_e32 v3, -1, v3
v_lshlrev_b32_e32 v3, 2, v3
s_mov_b32 m0, -1
ds_write_b32 v3, v2 offset:256
v_mov_b32 v4, 0x7f
ds_write_b32 v3, v4 offset:512
ds_write_b8 v3, v4 offset:513
v_mov_b32 v2, 0
ds_read_b32 v5, v3 offset:256
ds_read_b32 v6, v3 offset:512
s_waitcnt lgkmcnt(0)
v_cvt_f32_u32 v6, v6
exp mrt0 v5, v5, v6, v6 done vm
s_endpgm
