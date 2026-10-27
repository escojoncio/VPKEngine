## stage copyvertex
## user_sgprs 0
## vgpr_comp_cnt 0
## geom esgs_item 8 gsvs_vert 8 max_vert 1 in_prim 4 out_prim 0 gs_instances 1
// Copy shader: position, colour (and layer) back from the GS->VS ring.
s_mov_b32 vcc_hi, 0x03ff
v_lshlrev_b32 v1, 2, v0
s_movk_i32 s0, 0
buffer_load_dword v4, v1, s[4:7], s0 offen glc slc
s_movk_i32 s0, 64
buffer_load_dword v5, v1, s[4:7], s0 offen glc slc
s_movk_i32 s0, 128
buffer_load_dword v6, v1, s[4:7], s0 offen glc slc
s_movk_i32 s0, 192
buffer_load_dword v7, v1, s[4:7], s0 offen glc slc
s_movk_i32 s0, 256
buffer_load_dword v8, v1, s[4:7], s0 offen glc slc
s_movk_i32 s0, 320
buffer_load_dword v9, v1, s[4:7], s0 offen glc slc
s_movk_i32 s0, 384
buffer_load_dword v10, v1, s[4:7], s0 offen glc slc
s_movk_i32 s0, 448
buffer_load_dword v11, v1, s[4:7], s0 offen glc slc
s_waitcnt vmcnt(0)
exp pos0 v4, v5, v6, v7
exp param0 v8, v9, v10, v11 done
s_endpgm
