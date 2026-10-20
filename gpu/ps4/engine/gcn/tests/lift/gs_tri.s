## stage geometry
## user_sgprs 0
## geom esgs_item 8 gsvs_vert 8 max_vert 3 in_prim 4 out_prim 2 gs_instances 1
// GS: each input triangle out again (3 vertices, then a cut).
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
buffer_store_dword v17, off, s[8:11], s0 offset:12 glc slc
buffer_store_dword v18, off, s[8:11], s0 offset:24 glc slc
buffer_store_dword v19, off, s[8:11], s0 offset:36 glc slc
buffer_store_dword v20, off, s[8:11], s0 offset:48 glc slc
buffer_store_dword v21, off, s[8:11], s0 offset:60 glc slc
buffer_store_dword v22, off, s[8:11], s0 offset:72 glc slc
buffer_store_dword v23, off, s[8:11], s0 offset:84 glc slc
s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
v_lshlrev_b32 v12, 2, v1
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
buffer_store_dword v16, off, s[8:11], s0 offset:4 glc slc
buffer_store_dword v17, off, s[8:11], s0 offset:16 glc slc
buffer_store_dword v18, off, s[8:11], s0 offset:28 glc slc
buffer_store_dword v19, off, s[8:11], s0 offset:40 glc slc
buffer_store_dword v20, off, s[8:11], s0 offset:52 glc slc
buffer_store_dword v21, off, s[8:11], s0 offset:64 glc slc
buffer_store_dword v22, off, s[8:11], s0 offset:76 glc slc
buffer_store_dword v23, off, s[8:11], s0 offset:88 glc slc
s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
v_lshlrev_b32 v12, 2, v3
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
buffer_store_dword v16, off, s[8:11], s0 offset:8 glc slc
buffer_store_dword v17, off, s[8:11], s0 offset:20 glc slc
buffer_store_dword v18, off, s[8:11], s0 offset:32 glc slc
buffer_store_dword v19, off, s[8:11], s0 offset:44 glc slc
buffer_store_dword v20, off, s[8:11], s0 offset:56 glc slc
buffer_store_dword v21, off, s[8:11], s0 offset:68 glc slc
buffer_store_dword v22, off, s[8:11], s0 offset:80 glc slc
buffer_store_dword v23, off, s[8:11], s0 offset:92 glc slc
s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0)
s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)
s_endpgm
