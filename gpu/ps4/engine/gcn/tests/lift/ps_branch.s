## stage pixel
## user_sgprs 2
## ps_input_addr 0x2
// Divergent if/else on an interpolated value (LLVM's saveexec idiom), then a kill.
v_interp_p1_f32 v2, v0, attr0.x
v_interp_p2_f32 v2, v1, attr0.x
v_cmp_gt_f32 vcc, 0.5, v2
s_and_saveexec_b64 s[2:3], vcc
s_xor_b64 s[2:3], exec, s[2:3]
s_cbranch_execz else_part
v_mov_b32 v3, 1.0
else_part:
s_or_saveexec_b64 s[2:3], s[2:3]
s_xor_b64 exec, exec, s[2:3]
s_cbranch_execz join
v_mov_b32 v3, 0
join:
s_or_b64 exec, exec, s[2:3]
v_cmpx_le_f32 vcc, 0, v2
exp mrt0 v3, v3, v3, v3 done vm
s_endpgm
