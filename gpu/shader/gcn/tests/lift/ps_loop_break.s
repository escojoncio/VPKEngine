## stage pixel
## user_sgprs 2
## ps_input_addr 0x2
// A divergent loop with an early exit: march up to 8 steps, lanes stop once the value passes 1.
v_interp_p1_f32 v2, v0, attr0.x
v_interp_p2_f32 v2, v1, attr0.x
v_mov_b32 v3, 0
s_mov_b32 s4, 0
s_mov_b64 s[6:7], exec
loop:
v_add_f32 v3, v3, v2
v_cmp_lt_f32 vcc, 1.0, v3
s_andn2_b64 exec, exec, vcc
s_cbranch_execz done_loop
s_add_u32 s4, s4, 1
s_cmp_lt_u32 s4, 8
s_cbranch_scc1 loop
done_loop:
s_mov_b64 exec, s[6:7]
exp mrt0 v3, v3, v3, v3 done vm
s_endpgm
