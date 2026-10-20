# PROGRESS tiling (rama ps4/tiling)

## Hecho (todo con tests Linux en verde: `ctest -R vpps4`, 21/21)
- `tiling_addr.h`: direccionamiento gfx7 compartido C++/MSL (micro tiles display/thin/depth/thick, pipe P2/P8 16x16/8x16, bancos 2-16, rotaciones por slice 2D/3D, PRT, tile split con planos, muestras entrelazadas / por plano).
- `tiling.{h,cpp}`: tablas GB_TILE_MODE/GB_MACROTILE_MODE, `computeLayout` (mips, capas, cubos, volumenes, BCn, pow2, degradacion a 1D bajo un macro tile), `detileLevels/tileLevels`, `fmaskSurface`. API antigua conservada (mip 0).
- `gpu_tiler.{h,cpp}`: `vpps4::GpuTiler::detile/retile` (kernels MSL generados, un dispatch por mip, scratch lineal + copia buffer<->textura), `NotReady` mientras compila el pipeline.
- `renderer.cpp`: texturas con mips reales y capas, clave de cache con subrecurso (mips, base/last level, base array, pow2), subida por nivel (lineal directa, tilada GPU con respaldo CPU), `writeBack(ColorTarget)` por retile, `setGpuTiling`.
- `formats.{h,cpp}`: `ImageDesc::pow2Pad` (T# word3 bit 25) — cambio minimo en area formats.
- AstroVisionPro `shadps4-arm64-main/CMakeLists.txt`: gpu_tiler.cpp + texto generado de tiling_addr.h en el target.
- Tests: `tiling_tests` (referencia literal bit a bit + dorados), `gpu_tiler_tests` (emulador de kernels vs CPU), `mip_upload`, `write_back`, `tiled_upload` actualizado.

## Validacion
- Direccionamiento contrastado con la hoja del compute de detile de shadPS4 compilada como C++: 90 M de direcciones aleatorias identicas (todos los modos salvo el 18, ver riesgos); 1900 valores dorados en el repo.

## Pendiente / riesgos
- MSL no compilado aqui (sin Metal): revisado a mano y comprobado como C++ con una cabecera de sustitucion; si falla el pipeline, el renderer usa la CPU.
- Modo 18 (thin_3d_thin_prt): shadPS4 lo trata como 3D no PRT; aqui PRT 3D por nombre (pipe 16x16 como en su tabla). Sin hardware para decidir.
- Sin datos de PS4 reales: contrastado con otra implementacion, no con hardware. Offsets de mips = suma de tamanos sin alineacion extra (como shadPS4).
- Sin cubrir: micro tile rotado (ningun modo PS4), slots 27-30, bpe 12, config Neo (alt tile), DCC/CMASK/HTILE. T# base_level>0: la textura guarda la cadena completa desde la direccion.
- `writeBack` no lo llama nadie aun (coherence/present decide cuando).
- La subida GPU lee la memoria del invitado al ejecutarse en GPU (igual que la subida lineal directa): carrera conocida con escrituras de CPU posteriores al registro.
