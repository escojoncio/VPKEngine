## stage geometry
## user_sgprs 0
## geom esgs_item 8 gsvs_vert 9 max_vert 6 in_prim 4 out_prim 2 gs_instances 1
// GS: every triangle to layer 0 and layer 1, x shifted per eye (the PSVR single-pass stereo pattern).
v_mov_b32 v30, 0
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
v_add_f32 v16, -0.25, v16
v_mov_b32 v24, 0
buffer_store_dword v16, v30, s[8:11], s0 offen offset:0 glc slc
buffer_store_dword v17, v30, s[8:11], s0 offen offset:24 glc slc
buffer_store_dword v18, v30, s[8:11], s0 offen offset:48 glc slc
buffer_store_dword v19, v30, s[8:11], s0 offen offset:72 glc slc
buffer_store_dword v20, v30, s[8:11], s0 offen offset:96 glc slc
buffer_store_dword v21, v30, s[8:11], s0 offen offset:120 glc slc
buffer_store_dword v22, v30, s[8:11], s0 offen offset:144 glc slc
buffer_store_dword v23, v30, s[8:11], s0 offen offset:168 glc slc
buffer_store_dword v24, v30, s[8:11], s0 offen offset:192 glc slc
v_add_i32 v30, vcc, 4, v30
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
v_add_f32 v16, -0.25, v16
v_mov_b32 v24, 0
buffer_store_dword v16, v30, s[8:11], s0 offen offset:0 glc slc
buffer_store_dword v17, v30, s[8:11], s0 offen offset:24 glc slc
buffer_store_dword v18, v30, s[8:11], s0 offen offset:48 glc slc
buffer_store_dword v19, v30, s[8:11], s0 offen offset:72 glc slc
buffer_store_dword v20, v30, s[8:11], s0 offen offset:96 glc slc
buffer_store_dword v21, v30, s[8:11], s0 offen offset:120 glc slc
buffer_store_dword v22, v30, s[8:11], s0 offen offset:144 glc slc
buffer_store_dword v23, v30, s[8:11], s0 offen offset:168 glc slc
buffer_store_dword v24, v30, s[8:11], s0 offen offset:192 glc slc
v_add_i32 v30, vcc, 4, v30
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
v_add_f32 v16, -0.25, v16
v_mov_b32 v24, 0
buffer_store_dword v16, v30, s[8:11], s0 offen offset:0 glc slc
buffer_store_dword v17, v30, s[8:11], s0 offen offset:24 glc slc
buffer_store_dword v18, v30, s[8:11], s0 offen offset:48 glc slc
buffer_store_dword v19, v30, s[8:11], s0 offen offset:72 glc slc
buffer_store_dword v20, v30, s[8:11], s0 offen offset:96 glc slc
buffer_store_dword v21, v30, s[8:11], s0 offen offset:120 glc slc
buffer_store_dword v22, v30, s[8:11], s0 offen offset:144 glc slc
buffer_store_dword v23, v30, s[8:11], s0 offen offset:168 glc slc
buffer_store_dword v24, v30, s[8:11], s0 offen offset:192 glc slc
v_add_i32 v30, vcc, 4, v30
s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0)
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
v_add_f32 v16, 0.25, v16
v_mov_b32 v24, 1
buffer_store_dword v16, v30, s[8:11], s0 offen offset:0 glc slc
buffer_store_dword v17, v30, s[8:11], s0 offen offset:24 glc slc
buffer_store_dword v18, v30, s[8:11], s0 offen offset:48 glc slc
buffer_store_dword v19, v30, s[8:11], s0 offen offset:72 glc slc
buffer_store_dword v20, v30, s[8:11], s0 offen offset:96 glc slc
buffer_store_dword v21, v30, s[8:11], s0 offen offset:120 glc slc
buffer_store_dword v22, v30, s[8:11], s0 offen offset:144 glc slc
buffer_store_dword v23, v30, s[8:11], s0 offen offset:168 glc slc
buffer_store_dword v24, v30, s[8:11], s0 offen offset:192 glc slc
v_add_i32 v30, vcc, 4, v30
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
v_add_f32 v16, 0.25, v16
v_mov_b32 v24, 1
buffer_store_dword v16, v30, s[8:11], s0 offen offset:0 glc slc
buffer_store_dword v17, v30, s[8:11], s0 offen offset:24 glc slc
buffer_store_dword v18, v30, s[8:11], s0 offen offset:48 glc slc
buffer_store_dword v19, v30, s[8:11], s0 offen offset:72 glc slc
buffer_store_dword v20, v30, s[8:11], s0 offen offset:96 glc slc
buffer_store_dword v21, v30, s[8:11], s0 offen offset:120 glc slc
buffer_store_dword v22, v30, s[8:11], s0 offen offset:144 glc slc
buffer_store_dword v23, v30, s[8:11], s0 offen offset:168 glc slc
buffer_store_dword v24, v30, s[8:11], s0 offen offset:192 glc slc
v_add_i32 v30, vcc, 4, v30
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
v_add_f32 v16, 0.25, v16
v_mov_b32 v24, 1
buffer_store_dword v16, v30, s[8:11], s0 offen offset:0 glc slc
buffer_store_dword v17, v30, s[8:11], s0 offen offset:24 glc slc
buffer_store_dword v18, v30, s[8:11], s0 offen offset:48 glc slc
buffer_store_dword v19, v30, s[8:11], s0 offen offset:72 glc slc
buffer_store_dword v20, v30, s[8:11], s0 offen offset:96 glc slc
buffer_store_dword v21, v30, s[8:11], s0 offen offset:120 glc slc
buffer_store_dword v22, v30, s[8:11], s0 offen offset:144 glc slc
buffer_store_dword v23, v30, s[8:11], s0 offen offset:168 glc slc
buffer_store_dword v24, v30, s[8:11], s0 offen offset:192 glc slc
v_add_i32 v30, vcc, 4, v30
s_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0)
s_sendmsg sendmsg(MSG_GS, GS_OP_CUT, 0)
s_sendmsg sendmsg(MSG_GS_DONE, GS_OP_NOP)
s_endpgm
