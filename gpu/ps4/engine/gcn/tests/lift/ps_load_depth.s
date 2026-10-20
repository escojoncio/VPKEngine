## stage pixel
## user_sgprs 12
## ps_input_addr 0x2
## tex_depth
// image_load on a depth texture is one f32 (the verifier's texel type), copied to every selected channel.
v_cvt_u32_f32 v8, v2
v_cvt_u32_f32 v9, v3
image_load v12, v[8:9], s[0:7] dmask:0x1
s_waitcnt vmcnt(0)
exp mrt0 v12, v12, v12, v12 done vm
s_endpgm
