## stage compute
## user_sgprs 4
// Typed stores of packed formats (8_8_8_8 unorm, 16_16 float, 8 unorm under a dword, 10_10_10_2
// unorm, 16_16_16_16 snorm). Expected bytes (checked with the VPIR interpreter): 4000ff80 3c003800
// 0000ff00 400ffe00 7fff4000 20008001.
v_mov_b32 v1, 0.5
v_mov_b32 v2, 1.0
v_mov_b32 v3, -1.0
v_mov_b32 v4, 0x3e800000
tbuffer_store_format_xyzw v[1:4], off, s[0:3], 0 format:[BUF_DATA_FORMAT_8_8_8_8,BUF_NUM_FORMAT_UNORM]
tbuffer_store_format_xy v[1:2], off, s[0:3], 0 format:[BUF_DATA_FORMAT_16_16,BUF_NUM_FORMAT_FLOAT] offset:4
tbuffer_store_format_x v2, off, s[0:3], 0 format:[BUF_DATA_FORMAT_8,BUF_NUM_FORMAT_UNORM] offset:9
tbuffer_store_format_xyzw v[1:4], off, s[0:3], 0 format:[BUF_DATA_FORMAT_10_10_10_2,BUF_NUM_FORMAT_UNORM] offset:12
tbuffer_store_format_xyzw v[1:4], off, s[0:3], 0 format:[BUF_DATA_FORMAT_16_16_16_16,BUF_NUM_FORMAT_SNORM] offset:16
s_endpgm
