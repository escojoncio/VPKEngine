## stage export
## user_sgprs 4
## geom esgs_item 8
## vgpr_comp_cnt 0
// ES: position and colour from the vertex buffer to the ES->GS ring (es2gs offset = s4).
buffer_load_dwordx4 v[4:7], v0, s[0:3], 0 idxen
buffer_load_dwordx4 v[8:11], v0, s[0:3], 0 idxen offset:16
s_waitcnt vmcnt(0)
buffer_store_dwordx4 v[4:7], off, s[8:11], s4 glc slc
buffer_store_dwordx4 v[8:11], off, s[8:11], s4 offset:16 glc slc
s_endpgm
