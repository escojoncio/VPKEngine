# PROGRESS isa64 (ps4/isa64, desde f21f22f)

Alcance: opcodes f64 de GCN gfx7 y FLAT.

## Hecho
- Soft-float f64 compartido C++/MSL: `gpu/ps4/engine/ir/src/f64_core.inc` (un solo texto; se compila como C++ en
  `f64.cpp` y se incrusta en el MSL con CMake como bytes: prelude + tabla 2/pi + nucleo). API C++: `vp4ir/f64.h`.
  Cubre add, mul, fma, div, sqrt, rcp, rsq, ldexp, min/max (NaN GCN, IEEE), cmp (16 condiciones), class,
  trunc/ceil/floor/rndne/fract, frexp, conversiones i32/u32/f32, div_scale/div_fmas/div_fixup, trig_preop, clamp.
  Modos: redondeo (4), denormales dentro/fuera, IEEE, DX10 clamp (registro MODE).
- VPIR: op `f64 <fn>, args...` (F64Fn en ir.h), verificador, parser/impresor (nombre de funcion), optimizador (pura),
  emisor MSL (biblioteca una vez por shader que la usa), evaluador de tests (vpir_eval).
- Tests: `vp4ir.f64.softfloat` (12.7 M comprobaciones diferenciales contra double del host en los 4 modos de redondeo),
  `vp4ir.roundtrip.f64_ops`, `vp4msl.f64.library`.

## Pendiente
- Lifter: VOP1/VOP2/VOP3/VOPC f64, FLAT, tests del lifter, docs/isa-f64.md.
