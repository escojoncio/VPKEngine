# PROGRESS integration (VPKEngine, rama ps4/integration)

## Unido (en orden)
formats, tiling, state, coherence (GpuTiler en la costura de teselado), msaa, gs, cpucost, submit, pm4. ps4/base (81b65a7) ya contenido. isa NO unido.

## Resoluciones notables
- msaa: layer/vista del foveado dentro de `exportMisc` de gs (Layer + ViewportIndex cuando `viewIndexed`, tambien en CopyVertex).
- gs: RectList que es limpieza (fast clear / profundidad) sigue el camino de msaa; el resto se dibuja como rectangulo (vertex rectList). `Stage` unico (vsStage/psStage) con clipSpaceGL/dualSource, copia para gs.
- gs: `Scratch` de anillos ES/GS renombrado a `GeomScratch` (choca con el Scratch del GpuTiler).
- cpucost: `info_of` ahora miembro (plan de formato, FMASK, multisample) y memorizado por (dwords, texGen, muestras); snapshot de T# por etapa (6 etapas). La cache plana de texturas (`textureFast`) NO se integro: choca con el refresco por versiones del tracker de coherencia.
- submit: `life.plan(frame)` antes de `submit(std::move(frame))`; etapas de geometria grabadas con el frame retenido.
- pm4: indirectas convertidas en GPU conviven con las ramas geom/rect.
- Se quitaron del indice los ficheros `tests/Testing/Temporary` que msaa habia comiteado.

## Pruebas
`ctest -R '^(gcn|vp4|vpps4)'` en verde tras cada union (224 tests al final de pm4). `vp4gpu_core.pipelines` falla de forma intermitente solo con carga (pasa aislado).

## Cableado entre areas
- DB_ALPHA_TO_MASK -> `DrawState::alphaToMask` (un solo campo, leido por el adaptador de state): con >1 muestra `PipelineState::alphaToCoverage` (Metal); con 1 muestra `LiftInput::alphaTest` (descarta alfa<0.5 en el export 0; clave de shader incluye el flag). Test `state_reaches_backend` actualizado.
- Notas/skip de state, gs, msaa y errores de shader (6 etapas) pasan por `note()/skip()` -> sumidero de pm4 -> Census.
- `Renderer::coherencePoint(addr,size)` (nuevo): ACQUIRE_MEM/SURFACE_SYNC con invalidacion de lectores y rango acotado -> olvida hashes de shaders y fuerza recheck completo de texturas sin vigilancia. Sin escritura de vuelta (el tracker ya ordena las imagenes).
- Resultados de oclusion escritos por el host -> `tracker.markCpuWrite` del rango. Se drenan por sondeo (tras cada envio y en esperas), no por callback de finalizacion: el deque de snapshots no es seguro entre hilos.
- submit<->cpucost: `touchedLo/Hi` se reinician tambien en el envio vacio; `programGeneration` por envio (un envio por imagen); lifetime.plan antes del envio; frames retenidos con etapas de geometria.
- Hoisting de GS: `Frame::dispatchBeforeOpenPass` + `dispatchHazard` (core). ES/GS de todos los draws GS de un pase se ejecutan antes del pase; si el trabajo toca algo que el pase ya escribio, se divide (stats gsHoisted/gsSplit). Tests: `vpps4.geometry_hoist`, `vp4gpu_core.hoist`.
- `NullBackend::pipelines` con mutex (corrupcion de heap intermitente en `vp4gpu_core.pipelines`).
