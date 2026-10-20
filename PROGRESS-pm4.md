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

## Segunda tanda
- Oclusion: `Renderer::zpassDone/drainOcclusion/occlusionPending`. Los draws cuentan en una ranura de 64 bits de un buffer de visibilidad
  alineado con memoria del host (modo counting de Metal, `Draw::visibility`); cada ZPASS_DONE cierra la ranura y el valor acumulado se
  escribe en memoria del guest (bit 63 = valido) cuando el frame que contiene los draws ha terminado (`Device::completedFrames`) o el
  llamante espero a la GPU. Sin manejador de finalizacion propio: se consulta tras cada submit y en las esperas.
- Indirectas en la linea de tiempo de la GPU para TODOS los tipos de primitiva: un kernel de computo (MSL escrito a mano, registrado
  por `Hooks::shader`) convierte los argumentos PS4 a los de Metal, pone a cero los draws fuera del conteo (`DrawState::indirectCountAddress`,
  `drawIndex`) y genera la lista de indices de abanico/poligono/quad list/quad strip/line loop. Sin lectura por CPU. Limite de capacidad:
  16384 indices (no indexado) o 2*count+8 (<=65536) por draw convertido; el exceso no se dibuja.
- Tests: `vpps4.occlusion`, `vpps4.indirect_convert` (null backend). Verificacion unica (no ctest): el cuerpo MSL transcrito a C++ da el orden
  de `convertedOrder` y el gather de indices para todos los casos n<40.
- Lo que NO se pudo probar aqui: ejecucion real en Metal (visibilidad contada, kernel conversor); lo revisa la CI de macOS / dispositivo.
