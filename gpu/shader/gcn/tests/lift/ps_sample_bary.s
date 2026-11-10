## stage pixel
## user_sgprs 2
## ps_input_addr 0x1
## expect_msl sample_perspective
// PERSP_SAMPLE barycentrics with the rasterizer running the shader per sample: sample-interpolated input.
v_interp_p1_f32 v2, v0, attr0.x
v_interp_p2_f32 v2, v1, attr0.x
v_interp_p1_f32 v3, v0, attr0.y
v_interp_p2_f32 v3, v1, attr0.y
exp mrt0 v2, v3, v2, v3 done vm
s_endpgm
