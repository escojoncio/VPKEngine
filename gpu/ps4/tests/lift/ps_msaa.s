## stage pixel
## user_sgprs 2
## ps_input_addr 0x2
// A pixel shader that reads a multisampled colour image sample by sample, as a title's own MSAA resolve
// does: the FMASK (T# at +0x20) names the fragment of sample 0, the load takes it as the last address
// coordinate of the image (T# at +0x0).
s_load_dwordx8 s[4:11], s[0:1], 0x0
s_load_dwordx8 s[12:19], s[0:1], 0x8
v_mov_b32 v2, 3
v_mov_b32 v3, 4
s_waitcnt lgkmcnt(0)
image_load v[5:6], v[2:3], s[12:19] dmask:0x3 unorm
s_waitcnt vmcnt(0)
v_and_b32 v4, 15, v5
image_load v[6:9], v[2:4], s[4:11] dmask:0xf unorm
s_waitcnt vmcnt(0)
v_cvt_pkrtz_f16_f32 v0, v6, v7
v_cvt_pkrtz_f16_f32 v1, v8, v9
exp mrt0 v0, v0, v1, v1 done compr vm
s_endpgm
