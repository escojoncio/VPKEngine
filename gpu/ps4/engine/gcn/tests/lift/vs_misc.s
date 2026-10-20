## stage vertex
## user_sgprs 0
## vgpr_comp_cnt 0
## vs_out_cntl 0x450003
// Point size 2 and layer 1 in pos1, clip distances 0.5 and -0.5 in pos2.
v_mov_b32 v1, 0
v_mov_b32 v2, 2.0
v_mov_b32 v3, 1
v_mov_b32 v4, 0.5
v_mov_b32 v5, -0.5
exp pos0 v1, v1, v1, v1
exp pos1 v2, off, v3, off
exp pos2 v4, v5, off, off done
s_endpgm
