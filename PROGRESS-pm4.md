# PROGRESS pm4 (VPKEngine side)

## Hecho
- `gpu/ps4/include/vpps4/report.h` + `src/report.cpp`: sumidero de informes (`vpps4_set_report_sink`, `vpps4::setReportSink`,
  `vpps4_report`). VPKEngine no depende de shadPS4: el host instala un callback.
- `src/renderer.cpp`: `skip()` / `note()` (ambas sobrecargas) y los errores de traduccion de shaders informan al sumidero
  (categorias `draw-skip`, `approx`, `unsupported`, `shader`).
- Test `vpps4_tests report_sink` (ctest `vpps4.report_sink`).

## Probado
- `ctest -R vpps4` (null backend).

## Pendiente
- Consultas de oclusion (Draw::visibility ya existe en el backend Metal), prediccion, indirectas por GPU para abanico/quad/loop.

## Riesgos / decisiones
- Todo lo Objective-C++ (.mm) se revisa a mano; la CI de macOS lo comprobara.
