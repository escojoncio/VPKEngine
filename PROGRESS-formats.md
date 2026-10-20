# PROGRESS formats (rama ps4/formats)

## Hecho
- gcn/formats.h, formats.cpp: layouts de los 64 dfmt, referencia CPU de todas las conversiones dfmt x nfmt.
- gcn/format_emit.cpp: emisores VPIR identicos bit a bit a la referencia (tests gcn.formats, gcn.formats.lift).
- gcn/lift.cpp: loadFormat/storeFormat (dfmt 6/7/0/15, nfmt 6/7), TextureConv (nativa y manual por contenedor), ExportConv.
- vpps4 formats.*: texturePlan, colorPlan, depthFormat, composeSelect (dstSel). renderer.cpp: planFor, curExport, texture().
- ctest -R vpps4: 18/18.

## Pendiente
- Hook de capacidad BC en Renderer (setFormatCaps existe) y decodificacion BC en CPU como fallback.
- FMASK en rutas del renderer; muestreo de stencil; cuantizacion Z24.
- Tests de tabla para texturePlan/colorPlan/depthFormat.
- Backend .mm no tocado ni compilado.
