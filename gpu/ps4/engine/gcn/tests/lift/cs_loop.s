## stage compute
## user_sgprs 4
## lds_bytes 256
// Each lane stores its id in LDS, then sums its slot four times in a uniform loop and writes it out.
v_lshlrev_b32 v1, 2, v0
ds_write_b32 v1, v0
s_waitcnt lgkmcnt(0)
s_barrier
s_mov_b32 s8, 0
v_mov_b32 v2, 0
loop:
ds_read_b32 v3, v1
s_waitcnt lgkmcnt(0)
v_add_i32 v2, vcc, v2, v3
s_add_u32 s8, s8, 1
s_cmp_lt_u32 s8, 4
s_cbranch_scc1 loop
buffer_store_dword v2, v1, s[0:3], 0 offen
s_endpgm
