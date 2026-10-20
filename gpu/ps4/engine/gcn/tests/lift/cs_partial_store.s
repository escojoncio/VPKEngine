## stage compute
## user_sgprs 4
// Typed stores with fewer components than the format: only those are written (8_8_8_8 x, 16_16_16_16
// xyz, 2_10_10_10 xy). Checked with the VPIR interpreter over 0xaaaaaaaa: aaaaaaff 38003c00 aaaa4000
// aaa803ff.
v_mov_b32 v1, 1.0
v_mov_b32 v2, 0.5
v_mov_b32 v3, 2.0
tbuffer_store_format_x v1, off, s[0:3], 0 format:[BUF_DATA_FORMAT_8_8_8_8,BUF_NUM_FORMAT_UNORM]
tbuffer_store_format_xyz v[1:3], off, s[0:3], 0 format:[BUF_DATA_FORMAT_16_16_16_16,BUF_NUM_FORMAT_FLOAT] offset:4
tbuffer_store_format_xy v[1:2], off, s[0:3], 0 format:[BUF_DATA_FORMAT_2_10_10_10,BUF_NUM_FORMAT_UNORM] offset:12
s_endpgm
