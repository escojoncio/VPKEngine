## stage pixel
## user_sgprs 2
## ps_input_addr 0x2
## expect_no_msl [[depth(any)]]
// MRTZ export with only the compressed sample-mask enable (en=4 compr=1, no Z): no depth output, so Metal keeps the rasterizer depth.
v_interp_p1_f32 v2, v0, attr0.x
v_interp_p2_f32 v2, v1, attr0.x
v_cmpx_le_f32 vcc, 0, v2
exp mrt0 v2, v2, v2, v2 vm
.long 0xf8001c84
.long 0x00000100
s_endpgm
