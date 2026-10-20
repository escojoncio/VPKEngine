# PROGRESS formats

## Hecho
- gpu/ps4/engine/gcn: `vpgcn/formats.h`, `src/formats.cpp` (tabla de layouts de los 64 data formats, referencia CPU de decode/encode para todos los number formats), `src/format_emit.cpp` (emisores VPIR equivalentes).
- Test `gcn.formats` (`tests/format_test.cpp`): invariantes, valores conocidos, ida y vuelta, y emisor vs referencia bit a bit para cada (dfmt,nfmt) en decode y encode.

## Pendiente
- Integrar en lift.cpp (buffers, texturas, exports) y en renderer.cpp / formats.cpp.
