# PROGRESS isa64 (ps4/isa64, desde f21f22f)

Alcance: opcodes f64 de GCN gfx7 y las 46 instrucciones FLAT.

## Hecho
- Soft-float f64 compartido C++/MSL (`gpu/ps4/engine/ir/src/f64_core.inc`), op VPIR `f64`, emisor MSL, tests diferenciales.
- Lifter f64 (VOP1/VOP2/VOP3/VOPC) en `lift_f64.inc`; 800 programas diferenciales (795 coinciden, 5 conocidos).
- FLAT completo en `lift_flat.inc`: binding `GuestMap`, traduccion VA invitada -> direccion de dispositivo,
  carga/almacenamiento de todos los anchos, atomicos 32 y 64 bits.
- Renderer: tabla de regiones (`Impl::guestMap`), test `vpps4.flat_access` sobre el backend nulo.
- Docs: `gpu/ps4/docs/isa-f64.md`, `gpu/ps4/docs/isa-flat.md`.
- Tests: `vp4ir.f64.softfloat`, `vp4msl.f64.library`, `gcn.lift.differential_f64`, `gcn.flat.lowering`, goldens `cs_f64`/`cs_flat`.
- Auditoria: rechazados 124 (solo ds_*, del agente isa); FLAT y VOP f64 sin rechazos.

## Pendiente / notas
- 5 divergencias conocidas en cadenas aleatorias de div_scale/fmas (presupuesto en el test).
- Atomicos f64 de DS: los aporta el agente isa con las funciones de la libreria.
- Un puntero FLAT alcanzable solo desde memoria invitada no se encuentra hasta que otro binding mapee su region.
