## stage geometry
## user_sgprs 0
## geom esgs_item 8 gsvs_vert 8 max_vert 1 in_prim 4 out_prim 0 gs_instances 1
// GS: one point (vertex 0) for even primitive ids only.
v_and_b32 v13, 1, v2
v_cmp_eq_u32 vcc, 0, v13
s_and_saveexec_b64 s[12:13], vcc
v_lshlrev_b32 v12, 2, v0
s_movk_i32 s2, 0
buffer_load_dword v16, v12, s[4:7], s2 offen glc slc
s_movk_i32 s2, 256
buffer_load_dword v17, v12, s[4:7], s2 offen glc slc
s_movk_i32 s2, 512
buffer_load_dword v18, v12, s[4:7], s2 offen glc slc
s_movk_i32 s2, 768
buffer_load_dword v19, v12, s[4:7], s2 offen glc slc
s_movk_i32 s2, 1024
buffer_load_dword v20, v12, s[4:7], s2 offen glc slc
s_movk_i32 s2, 1280
buffer_load_dword v21, v12, s[4:7], s2 offen glc slc
s_movk_i32 s2, 1536
buffer_load_dword v22, v12, s[4:7], s2 offen glc slc
s_movk_i32 s2, 1792
buffer_load_dword v23, v12, s[4:7], s2 offen glc slc
s_waitcnt vmcnt(0)
buffer_store_dword v16, off, s[8:11], s0 offset:0 glc slc
buffer_store_dword v17, off, s[8:11], s0 offset:4 glc slc
buffer_store_dword v18, off, s[8:11], s0 offset:8 glc slc
buffer_store_dword v19, off, s[8:11], s0 offset:12 glc slc
buffer_store_dword v20, off, s[8:11], s0 offset:16 glc slc
buffer_store_dword v21, off, s[8:11], s0 offset:20 glc slc
buffer_store_dword v22, off, s[8:11], s0 offset:24 glc slc
buffer_store_dword v23, off, s[8:11], s0 offset:28 glc slc
s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
s_or_b64 exec, exec, s[12:13]
s_endpgm
