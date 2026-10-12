## stage compute
## user_sgprs 16
// Image atomics on 2D storage textures: a counter (add, returning the old value), a compare-swap,
// and signed min on another texture (declared i32).
v_mov_b32 v2, v0
v_mov_b32 v3, 0
v_mov_b32 v1, 1
image_atomic_add v1, v[2:3], s[0:7] dmask:0x1 unorm glc
v_mov_b32 v4, 7
v_mov_b32 v5, 0
image_atomic_cmpswap v[4:5], v[2:3], s[0:7] dmask:0x3 unorm glc
s_waitcnt vmcnt(0)
v_add_i32 v6, vcc, v1, v4
image_atomic_smin v6, v[2:3], s[8:15] dmask:0x1 unorm
s_endpgm
