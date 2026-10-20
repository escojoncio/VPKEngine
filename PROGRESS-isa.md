# PROGRESS isa (ps4/isa)

## Fase A (opcodes pequenos) - lifter hecho, tests diferenciales en curso
- fork/join, s_flbit_i32, s_quadmask, s_movrels/movreld, s_setvskip, s_getreg/setreg MODE, s_memtime,
  v_cvt_pkaccum_u8_f32, v_sad_*, v_msad_u8, v_qsad/mqsad, v_mullit, v_mov_fed, v_mad_i64_i32,
  v_div_scale/fmas/fixup exactos, v_clrexcp (lift.cpp).
- Auditoria: tests/audit/gcn_audit.cpp (vpgcn_audit). Estado: 528 implementados, 87 aproximados, 264 rechazados.
- Semantica recordada de memoria (PDF ISA no accesible): riesgo.

## Fase B (cross-lane, modo wave) - hecho en lift.cpp / wave.cpp
- Diseno: el lifter hace una primera pasada por lane (modelo anterior); si encuentra algo que necesita una wave real
  (readlane/writelane/readfirstlane, mbcnt con mascara parcial, mascaras como datos: s_bcnt1/ff1/quadmask/lectura de
  vcc_lo, ds_swizzle fuera del quad...) y el stage es compute con <=64 hilos, repite la elevacion en "modo wave"
  (Lifter::waveMode): los 64 hilos van en lockstep, las mascaras tienen forma dword real (ballot de la bool por lane),
  los saltos vccz/execz son de toda la wave (subgroup_any), SCC de operaciones de mascara = any. v_readfirstlane =
  primer lane con EXEC (lane 0 si ninguno). El IR usa subgroup_ballot/shuffle/any con waveSize 64 y
  vp4ir::lowerWave64 lo baja a 2 SIMD de 32 con memoria threadgroup + barreras (<=32 hilos: ops nativas).
- Pendiente de B: stages graficos y workgroups >64 siguen con aproximacion por lane (notas).
- Tests: gcn.lift.differential_wave (gen_wave.py -> programs_wave.txt, gcn_wave_diff.cpp: referencia gcn_ref wave real
  vs VPIR nativo 64 lanes y VPIR bajado a 32) + gcn_ref_ds.cpp (ds_swizzle de referencia).

## Pendiente
- Tests diferenciales de fase A, modo wave (fase B), DS, atomicos, MIMG, f64, FLAT.
