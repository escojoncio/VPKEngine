## stage compute
## user_sgprs 4
// Loads of the packed small-float formats (10_11_11, 11_11_10: x from bit 0). Checked with the VPIR
// interpreter: 0x701001e0 -> 1.0, 2.0, 0.5; x 0x7c1 -> NaN, y 1 -> 2^-20 (denormal), z 0x3e0 -> inf.
tbuffer_load_format_xyz v[1:3], off, s[0:3], 0 format:[BUF_DATA_FORMAT_10_11_11,BUF_NUM_FORMAT_FLOAT]
tbuffer_load_format_xyz v[4:6], off, s[0:3], 0 format:[BUF_DATA_FORMAT_11_11_10,BUF_NUM_FORMAT_FLOAT] offset:4
s_waitcnt vmcnt(0)
v_add_f32 v1, v1, v4
v_add_f32 v2, v2, v5
v_add_f32 v3, v3, v6
v_mov_b32 v7, 0
buffer_store_dwordx3 v[1:3], v7, s[0:3], 0 offen offset:16
s_endpgm
