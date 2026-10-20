## stage vertex
## user_sgprs 6
// The vertex attributes come from the fetch subroutine called through s_swappc_b64.
s_swappc_b64 s[0:1], s[2:3]
exp pos0 v4, v5, v6, v7 done
exp param0 v4, v5, v6, v7
s_endpgm
## fetch
s_load_dwordx4 s[8:11], s[4:5], 0x0
s_waitcnt lgkmcnt(0)
buffer_load_format_xyzw v[4:7], v0, s[8:11], 0 idxen
s_waitcnt vmcnt(0)
s_setpc_b64 s[0:1]
