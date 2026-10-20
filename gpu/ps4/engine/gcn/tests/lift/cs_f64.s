## stage compute
## user_sgprs 8
// 64-bit float arithmetic: add / mul / fma / min / max with modifiers, the division sequence, a root,
// conversions, a compare and a class test, all through the soft-float f64 op.
v_mov_b32 v2, s0
v_mov_b32 v3, s1
v_mov_b32 v4, s2
v_mov_b32 v5, s3
v_add_f64 v[6:7], v[2:3], v[4:5]
v_mul_f64 v[8:9], |v[2:3]|, -v[4:5]
v_fma_f64 v[10:11], v[6:7], v[8:9], v[2:3] clamp
v_min_f64 v[12:13], v[6:7], v[8:9]
v_max_f64 v[14:15], v[6:7], v[8:9]
v_rcp_f64 v[16:17], v[4:5]
v_div_scale_f64 v[18:19], vcc, v[2:3], v[4:5], v[2:3]
v_div_fmas_f64 v[18:19], v[18:19], v[16:17], v[2:3]
v_div_fixup_f64 v[18:19], v[18:19], v[4:5], v[2:3]
v_sqrt_f64 v[20:21], v[18:19]
v_ldexp_f64 v[22:23], v[20:21], 3
v_cvt_f32_f64 v24, v[22:23]
v_cvt_i32_f64 v25, v[22:23]
v_cvt_f64_u32 v[26:27], v25
v_cmp_lt_f64 vcc, v[2:3], v[4:5]
v_cndmask_b32 v28, v24, v25, vcc
v_cmp_class_f64 s[10:11], v[2:3], v6
s_endpgm
