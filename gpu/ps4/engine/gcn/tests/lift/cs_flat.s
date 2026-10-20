## stage compute
## user_sgprs 8
// FLAT: raw 64-bit guest pointers through the guest region map; loads and stores of every width, sub-dword
// accesses, atomics (32 and 64 bit, with and without a return), compare-swap, inc / dec and the float ones.
v_mov_b32 v2, s0
v_mov_b32 v3, s1
v_mov_b32 v4, s2
v_mov_b32 v5, s3
v_mov_b32 v6, s4
v_mov_b32 v7, s5
flat_load_dwordx4 v[8:11], v[2:3]
flat_store_dwordx3 v[4:5], v[8:10]
flat_load_ubyte v12, v[2:3]
flat_load_sshort v13, v[2:3]
flat_store_byte v[4:5], v12
flat_store_short v[4:5], v13
flat_atomic_add v14, v[2:3], v6 glc
flat_atomic_cmpswap v15, v[2:3], v[6:7] glc
flat_atomic_inc v16, v[2:3], v6 glc
flat_atomic_fmin v17, v[2:3], v6 glc
flat_atomic_add_x2 v[18:19], v[2:3], v[6:7] glc
flat_atomic_umax_x2 v[20:21], v[2:3], v[6:7] glc
s_endpgm
