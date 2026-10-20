# Progreso area gs (rama ps4/gs)

## Hecho
- Lifter GCN: etapas Export (ES), Geometry (GS) y CopyVertex, anillos ESGS/GSVS, `s_sendmsg` emit/cut/done, epilogo del GS con lista de indices (relleno 0xFFFFFFFF, el copy shader lo saca del volumen de recorte).
- Exports pos1..pos3 (point size, clip distances, layer, viewport index) segun PA_CL_VS_OUT_CNTL.
- RectList general: 3 ejecuciones del VS, cuarto vertice por la regla de hardware, 6 vertices por rect.
- Definiciones compartidas en `geom.h`.
- Tests Linux: `vpgcn_geom_test` (triangles, stereo16, stereo32, strip, lines, conditional, instanced, rect, misc) sobre el interprete VPIR con anillos emulados; 13 programas sintéticos en `tests/lift`.

## Pendiente
- Renderer: ruta de draw con GS (scratch, dispatch ES/GS, draw indexado, layerCount), state.h, adyacencias sin GS, Patch via Census.
- AstroVisionPro: conversion de registros ES/GS/copy en vpkengine_renderer.mm.
- Ruta rapida estereo (VS fusionado, instancias x2) con fallback.

## Riesgos
- El ABI de anillos (offsets, SGPR) no esta verificado con shaders reales de Sony.
