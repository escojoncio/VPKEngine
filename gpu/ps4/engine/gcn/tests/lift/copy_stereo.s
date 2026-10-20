## stage copyvertex
## user_sgprs 0
## vgpr_comp_cnt 0
## vs_out_cntl 0x40000
## geom esgs_item 8 gsvs_vert 9 max_vert 6 in_prim 4 out_prim 2 gs_instances 1
// Copy shader: position, colour (and layer) back from the GS->VS ring.
s_mov_b32 vcc_hi, 0x03ff
v_lshlrev_b32 v1, 2, v0
s_movk_i32 s0, 0
buffer_load_dword v4, v1, s[4:7], s0 offen glc slc
s_movk_i32 s0, 1536
buffer_load_dword v5, v1, s[4:7], s0 offen glc slc
s_movk_i32 s0, 3072
buffer_load_dword v6, v1, s[4:7], s0 offen glc slc
s_movk_i32 s0, 4608
buffer_load_dword v7, v1, s[4:7], s0 offen glc slc
s_movk_i32 s0, 6144
buffer_load_dword v8, v1, s[4:7], s0 offen glc slc
s_movk_i32 s0, 7680
buffer_load_dword v9, v1, s[4:7], s0 offen glc slc
s_movk_i32 s0, 9216
buffer_load_dword v10, v1, s[4:7], s0 offen glc slc
s_movk_i32 s0, 10752
buffer_load_dword v11, v1, s[4:7], s0 offen glc slc
s_movk_i32 s0, 12288
buffer_load_dword v12, v1, s[4:7], s0 offen glc slc
s_waitcnt vmcnt(0)
exp pos0 v4, v5, v6, v7
exp pos1 off, off, v12, off
exp param0 v8, v9, v10, v11 done
s_endpgm
