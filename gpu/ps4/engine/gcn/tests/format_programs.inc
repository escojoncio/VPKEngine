// Generated with llvm-mc (-mcpu=bonaire) from the sources in the comments; patched by the test where noted.
//   v_mov_b32 v0, s0
//   v_mov_b32 v1, s1
//   v_mov_b32 v2, s2
//   v_mov_b32 v3, s3
//   tbuffer_store_format_x v[0:0], off, s[4:7], 0 format:[BUF_DATA_FORMAT_8_8_8_8,BUF_NUM_FORMAT_UNORM]
//   s_endpgm
const uint32_t kStorex[] = {0x7e000200, 0x7e020201, 0x7e040202, 0x7e060203, 0xe8540000, 0x80010000, 0xbf810000};
//   tbuffer_load_format_x v[4:4], off, s[4:7], 0 format:[BUF_DATA_FORMAT_8_8_8_8,BUF_NUM_FORMAT_UNORM]
//   s_waitcnt vmcnt(0)
//   buffer_store_dwordx4 v[4:7], off, s[8:11], 0
//   s_endpgm
const uint32_t kLoadx[] = {0xe8500000, 0x80010400, 0xbf8c0f70, 0xe0780000, 0x80020400, 0xbf810000};
//   v_mov_b32 v0, s0
//   v_mov_b32 v1, s1
//   v_mov_b32 v2, s2
//   v_mov_b32 v3, s3
//   tbuffer_store_format_xy v[0:1], off, s[4:7], 0 format:[BUF_DATA_FORMAT_8_8_8_8,BUF_NUM_FORMAT_UNORM]
//   s_endpgm
const uint32_t kStorexy[] = {0x7e000200, 0x7e020201, 0x7e040202, 0x7e060203, 0xe8550000, 0x80010000, 0xbf810000};
//   tbuffer_load_format_xy v[4:5], off, s[4:7], 0 format:[BUF_DATA_FORMAT_8_8_8_8,BUF_NUM_FORMAT_UNORM]
//   s_waitcnt vmcnt(0)
//   buffer_store_dwordx4 v[4:7], off, s[8:11], 0
//   s_endpgm
const uint32_t kLoadxy[] = {0xe8510000, 0x80010400, 0xbf8c0f70, 0xe0780000, 0x80020400, 0xbf810000};
//   v_mov_b32 v0, s0
//   v_mov_b32 v1, s1
//   v_mov_b32 v2, s2
//   v_mov_b32 v3, s3
//   tbuffer_store_format_xyz v[0:2], off, s[4:7], 0 format:[BUF_DATA_FORMAT_8_8_8_8,BUF_NUM_FORMAT_UNORM]
//   s_endpgm
const uint32_t kStorexyz[] = {0x7e000200, 0x7e020201, 0x7e040202, 0x7e060203, 0xe8560000, 0x80010000, 0xbf810000};
//   tbuffer_load_format_xyz v[4:6], off, s[4:7], 0 format:[BUF_DATA_FORMAT_8_8_8_8,BUF_NUM_FORMAT_UNORM]
//   s_waitcnt vmcnt(0)
//   buffer_store_dwordx4 v[4:7], off, s[8:11], 0
//   s_endpgm
const uint32_t kLoadxyz[] = {0xe8520000, 0x80010400, 0xbf8c0f70, 0xe0780000, 0x80020400, 0xbf810000};
//   v_mov_b32 v0, s0
//   v_mov_b32 v1, s1
//   v_mov_b32 v2, s2
//   v_mov_b32 v3, s3
//   tbuffer_store_format_xyzw v[0:3], off, s[4:7], 0 format:[BUF_DATA_FORMAT_8_8_8_8,BUF_NUM_FORMAT_UNORM]
//   s_endpgm
const uint32_t kStorexyzw[] = {0x7e000200, 0x7e020201, 0x7e040202, 0x7e060203, 0xe8570000, 0x80010000, 0xbf810000};
//   tbuffer_load_format_xyzw v[4:7], off, s[4:7], 0 format:[BUF_DATA_FORMAT_8_8_8_8,BUF_NUM_FORMAT_UNORM]
//   s_waitcnt vmcnt(0)
//   buffer_store_dwordx4 v[4:7], off, s[8:11], 0
//   s_endpgm
const uint32_t kLoadxyzw[] = {0xe8530000, 0x80010400, 0xbf8c0f70, 0xe0780000, 0x80020400, 0xbf810000};
//   v_mov_b32 v0, s0
//   v_mov_b32 v1, s1
//   v_mov_b32 v2, s2
//   v_mov_b32 v3, s3
//   exp mrt0 v0, v1, v2, v3 done vm
//   s_endpgm
const uint32_t kExport[] = {0x7e000200, 0x7e020201, 0x7e040202, 0x7e060203, 0xf800180f, 0x03020100, 0xbf810000};
//   s_load_dwordx8 s[4:11], s[0:1], 0x0
//   s_load_dwordx4 s[12:15], s[0:1], 0x8
//   v_mov_b32 v2, 0.25
//   v_mov_b32 v3, 0.75
//   s_waitcnt lgkmcnt(0)
//   image_sample v[4:7], v[2:3], s[4:11], s[12:15] dmask:0xf
//   s_waitcnt vmcnt(0)
//   exp mrt0 v4, v5, v6, v7 done vm
//   s_endpgm
const uint32_t kSample[] = {0xc0c20100, 0xc0860108, 0x7e0402ff, 0x3e800000, 0x7e0602ff, 0x3f400000, 0xbf8c007f, 0xf0800f00, 0x00610402, 0xbf8c0f70, 0xf800180f, 0x07060504, 0xbf810000};
//   s_load_dwordx8 s[4:11], s[0:1], 0x0
//   s_load_dwordx4 s[12:15], s[0:1], 0x8
//   v_mov_b32 v2, 0.25
//   v_mov_b32 v3, 0.75
//   s_waitcnt lgkmcnt(0)
//   v_mov_b32 v4, 1.0
//   image_sample_l v[4:7], v[2:4], s[4:11], s[12:15] dmask:0xf
//   s_waitcnt vmcnt(0)
//   exp mrt0 v4, v5, v6, v7 done vm
//   s_endpgm
const uint32_t kSampleL[] = {0xc0c20100, 0xc0860108, 0x7e0402ff, 0x3e800000, 0x7e0602ff, 0x3f400000, 0xbf8c007f, 0x7e0802f2, 0xf0900f00, 0x00610402, 0xbf8c0f70, 0xf800180f, 0x07060504, 0xbf810000};
//   s_load_dwordx8 s[4:11], s[0:1], 0x0
//   s_load_dwordx4 s[12:15], s[0:1], 0x8
//   v_mov_b32 v2, 0.25
//   v_mov_b32 v3, 0.75
//   s_waitcnt lgkmcnt(0)
//   image_gather4 v[4:7], v[2:3], s[4:11], s[12:15] dmask:0x2
//   s_waitcnt vmcnt(0)
//   exp mrt0 v4, v5, v6, v7 done vm
//   s_endpgm
const uint32_t kGather[] = {0xc0c20100, 0xc0860108, 0x7e0402ff, 0x3e800000, 0x7e0602ff, 0x3f400000, 0xbf8c007f, 0xf1000200, 0x00610402, 0xbf8c0f70, 0xf800180f, 0x07060504, 0xbf810000};
//   s_load_dwordx8 s[4:11], s[0:1], 0x0
//   v_mov_b32 v2, 3
//   v_mov_b32 v3, 5
//   s_waitcnt lgkmcnt(0)
//   image_load v[4:7], v[2:3], s[4:11] dmask:0xf unorm
//   s_waitcnt vmcnt(0)
//   exp mrt0 v4, v5, v6, v7 done vm
//   s_endpgm
const uint32_t kLoadImg[] = {0xc0c20100, 0x7e040283, 0x7e060285, 0xbf8c007f, 0xf0001f00, 0x00010402, 0xbf8c0f70, 0xf800180f, 0x07060504, 0xbf810000};
//   v_mov_b32 v4, s2
//   v_mov_b32 v5, s3
//   v_mov_b32 v6, s4
//   v_mov_b32 v7, s5
//   s_load_dwordx8 s[4:11], s[0:1], 0x0
//   v_mov_b32 v2, 3
//   v_mov_b32 v3, 5
//   s_waitcnt lgkmcnt(0)
//   image_store v[4:7], v[2:3], s[4:11] dmask:0xf unorm
//   s_endpgm
const uint32_t kStoreImg[] = {0x7e080202, 0x7e0a0203, 0x7e0c0204, 0x7e0e0205, 0xc0c20100, 0x7e040283, 0x7e060285, 0xbf8c007f, 0xf0201f00, 0x00010402, 0xbf810000};
