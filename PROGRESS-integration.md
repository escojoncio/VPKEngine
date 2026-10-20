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
