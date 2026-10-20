# PROGRESS tiling (rama ps4/tiling)

## Hecho
- `gpu/ps4/include/vpps4/tiling_addr.h`: direccionamiento gfx7 compartido C++/MSL (micro tiles display/thin/depth/thick, pipe P2/P8 16x16/8x16, banco 2-16, rotaciones por slice 2D/3D, PRT, tile split con planos, muestras entrelazadas/por plano, kernels por elemento).
- `gpu/ps4/src/tiling.cpp` + `tiling.h`: tabla GB_TILE_MODE (27 modos validos + lineal general), GB_MACROTILE_MODE 0-15, `computeLayout` (cadena de mips con degradacion a 1D bajo un macro tile, pow2, bloques BCn, volumenes, arrays/cubos, muestras), `detileLevels/tileLevels` (CPU), FMASK como superficie (`fmaskSurface`). API antigua (`tiledBytes/tiledOffset/detile/tile`) conservada sobre mip 0.
- `gpu/ps4/src/gpu_tiler.cpp` + `gpu_tiler.h`: `vpps4::GpuTiler::detile/retile` como kernels MSL generados (uno por direccion, un dispatch por mip), con la cabecera de direccionamiento incrustada en el MSL (CMake la convierte en texto).
- Tests: `tiling_tests` (referencia literal `tests/tiling_ref.h` bit a bit para todos los modos x tamanos x mips x slices x muestras, biyeccion, ida y vuelta, valores dorados `tiling_golden.inc` de una implementacion independiente), `gpu_tiler_tests` (emulador C++ de los kernels vs ruta CPU).

## Pendiente / riesgos
- Integracion en renderer.cpp (mips reales, clave de cache con subrecurso, ruta GPU con respaldo CPU): en curso.
- MSL no compilado aqui (sin Metal): revisado a mano; si el pipeline falla el renderer cae a la CPU.
- Sin datos de PS4 reales: el direccionamiento esta contrastado con otra implementacion, no con hardware.
- No cubierto: micro tile rotado (ningun modo PS4 lo usa), slots 27-30, elementos de 12 bytes, config Neo.
