# PROGRESS isa (ps4/isa)

## Fase A (opcodes pequenos) - lifter hecho, tests diferenciales en curso
- fork/join, s_flbit_i32, s_quadmask, s_movrels/movreld, s_setvskip, s_getreg/setreg MODE, s_memtime,
  v_cvt_pkaccum_u8_f32, v_sad_*, v_msad_u8, v_qsad/mqsad, v_mullit, v_mov_fed, v_mad_i64_i32,
  v_div_scale/fmas/fixup exactos, v_clrexcp (lift.cpp).
- Auditoria: tests/audit/gcn_audit.cpp (vpgcn_audit). Estado: 528 implementados, 87 aproximados, 264 rechazados.
- Semantica recordada de memoria (PDF ISA no accesible): riesgo.

## Pendiente
- Tests diferenciales de fase A, modo wave (fase B), DS, atomicos, MIMG, f64, FLAT.
