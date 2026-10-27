## stage vertex
## user_sgprs 0
## vgpr_comp_cnt 0
## rect_list 1
// Rectangle corners from the vertex id: (-1,-1) (1,-1) (-1,1); uv in param0.
v_and_b32 v1, 1, v0
v_cvt_f32_u32 v1, v1
v_lshrrev_b32 v2, 1, v0
v_cvt_f32_u32 v2, v2
v_mad_f32 v3, v1, 2.0, -1.0
v_mad_f32 v4, v2, 2.0, -1.0
v_mov_b32 v5, 0
v_mov_b32 v6, 1.0
exp pos0 v3, v4, v5, v6
exp param0 v1, v2, v5, v6 done
s_endpgm
