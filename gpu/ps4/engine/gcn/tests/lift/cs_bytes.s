## stage compute
## user_sgprs 4
## lds_bytes 64
// Byte and short stores: each lane writes its id as a byte into LDS and into the buffer, then a
// short of the LDS dword its byte fell into (lanes sharing a dword must not lose each other's bytes).
ds_write_b8 v0, v0
s_waitcnt lgkmcnt(0)
s_barrier
v_and_b32 v1, -4, v0
ds_read_b32 v2, v1
s_waitcnt lgkmcnt(0)
buffer_store_byte v0, v0, s[0:3], 0 offen
v_lshlrev_b32 v3, 1, v0
buffer_store_short v2, v3, s[0:3], 0 offen offset:256
s_endpgm
