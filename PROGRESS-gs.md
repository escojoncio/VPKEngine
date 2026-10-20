# Progreso area gs (rama ps4/gs)

## Hecho
- Lifter GCN: etapas Export (ES), Geometry (GS) y CopyVertex, anillos ESGS/GSVS, `s_sendmsg` emit/cut/done, epilogo del GS con lista de indices (relleno 0xFFFFFFFF, el copy shader lo saca del volumen de recorte).
- Exports pos1..pos3 (point size, clip distances, layer, viewport index) segun PA_CL_VS_OUT_CNTL.
- RectList general: 3 ejecuciones del VS, cuarto vertice por la regla de hardware, 6 vertices por rect.
- Definiciones compartidas en `geom.h`.
- Tests Linux: `vpgcn_geom_test` (triangles, stereo16, stereo32, strip, lines, conditional, instanced, rect, misc) sobre el interprete VPIR con anillos emulados; 13 programas sintéticos en `tests/lift`.

- Renderer (vpps4): ruta de draw con GS (buffers scratch reutilizables, dispatch ES y GS como compute, draw indexado sobre la lista de indices del GS, layerCount por PA_CL_VS_OUT_CNTL), RectList general (6 vertices por rect, tabla de ids), adyacencia sin GS (se descartan los vertices de adyacencia), Patch omitido con nota. state.h: GeometryState, Stage::vsOutCntl, DrawState::es/gs/geom, tipos de primitiva nuevos. Tests: geometry_shader, geometry_indexed, geometry_bad_primitive, rect_list, adjacency, patch.

## Pendiente
- AstroVisionPro: conversion de registros ES/GS/copy en vpkengine_renderer.mm.
- Ruta rapida estereo (VS fusionado, instancias x2) con fallback.

## Riesgos
- El ABI de anillos (offsets, SGPR) no esta verificado con shaders reales de Sony.
