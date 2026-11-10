## stage pixel
## user_sgprs 2
## ps_input_addr 0x1
## ps_iter_sample 0
## expect_msl centroid_perspective
// The same program with the pixel shader running per pixel (PS_ITER_SAMPLE clear): the sample barycentrics
// are interpolated at the centroid, which does not make the host run the shader per sample.
v_interp_p1_f32 v2, v0, attr0.x
v_interp_p2_f32 v2, v1, attr0.x
v_interp_p1_f32 v3, v0, attr0.y
v_interp_p2_f32 v3, v1, attr0.y
exp mrt0 v2, v3, v2, v3 done vm
s_endpgm
