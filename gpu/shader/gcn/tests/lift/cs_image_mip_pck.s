## stage compute
## user_sgprs 20
## tex_dim d2
## tex_bits 8
## tex_num 0
// Packed (raw channel) loads and stores, and a store into a chosen mip level.
v_and_b32 v1, 7, v0
v_lshrrev_b32 v2, 3, v0
image_load_pck v[4:7], v[1:2], s[0:7] dmask:0xf unorm
image_load_pck_sgn v[8:9], v[1:2], s[0:7] dmask:0x3 unorm
s_waitcnt vmcnt(0)
v_add_i32 v4, vcc, v4, v8
image_store_pck v[4:7], v[1:2], s[0:7] dmask:0xf unorm
v_mov_b32 v3, 2
image_store_mip v[4:7], v[1:3], s[12:19] dmask:0xf unorm
s_endpgm
