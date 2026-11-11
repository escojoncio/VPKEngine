## stage pixel
## user_sgprs 2
## ps_input_addr 0x2
## expect_msl [[depth(any)]]
## expect_msl vp_out.depth =
// MRTZ export with a real Z (en=1): the depth output is declared and assigned.
v_interp_p1_f32 v2, v0, attr0.x
v_interp_p2_f32 v2, v1, attr0.x
exp mrt0 v2, v2, v2, v2 vm
exp mrtz v2, off, off, off done vm
s_endpgm
