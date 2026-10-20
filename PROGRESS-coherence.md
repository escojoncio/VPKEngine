# PROGRESS coherence (rama ps4/coherence)

## Hecho (VPKEngine, gpu/ps4)
- `memory_tracker.{h,cpp}`: `vpps4::MemoryTracker`. Mapa de intervalos exactos en bytes (no falsos
  compartidos dentro de una pagina de 16 KB; `kPageBytes` es la granularidad de la deteccion de
  escrituras de CPU). Por rango: escritor (imagen / CPU / raw / DMA), version global monotona y
  `stale` (la copia GPU de una imagen va por delante de la memoria). API del BRIEF:
  `markGpuWrite`, `markCpuWrite`, `version`, `ownerImage`; ademas `writtenByOthers`,
  `staleOwners`, `markSynced`, `release`, `writerAt`, `staleAt`, `forgetHistory` (acotado a 131072 intervalos).
- `renderer.cpp` (imagenes): una sola logica para colour/depth targets y texturas (`Surface`):
  - cada imagen guarda su rango de memoria y la version que contiene; antes de usarse (`refresh`) se
    rellena desde memoria si otro escritor escribio su rango despues (CPU, escritura raw, DMA, otra imagen);
  - los pases/dispatch marcan sus escrituras al GRABAR (`markWrite` tras el draw: colour con writeMask,
    depth con depthWrite/clear/stencil; storage images por `Access::write`; buffers escribibles
    `rawAccess(write)`; DMA fill/copy; resolve destino);
  - sync perezoso en la linea temporal de la GPU: antes de un acceso raw (buffer enlazado, indices,
    argumentos indirectos, DMA origen/destino), de un rellenado de otra imagen, de un desalojo o de
    `Renderer::syncToMemory` (lectura de CPU) las imagenes por delante de la memoria se reescriben
    (`retileToMemory`), ordenadas por version; una escritura que cubre la imagen entera no la reescribe;
  - la antigua heuristica "gpuWritten" desaparece: las versiones lo cubren y no es permanente;
  - busqueda RT -> textura: direccion + nivel 0 + rango de slices + tamano + formato del mismo tamano de
    texel (vista) — si no, pasa por memoria (retile del RT + detile a la textura nueva);
  - vistas (`Hooks::view`, `Draw::views`, `Attachment::format`) en lugar de la heuristica de mismo tamano;
    sin hook cae al comportamiento anterior con nota;
  - desalojo LRU por inactividad y presupuesto de bytes (`setResidency`), con reescritura si la imagen
    iba por delante de la memoria;
  - hooks de vigilancia de paginas para regiones espejo (`GuestRegion::mirrored`, `watchWrites`, `watchOwned`).
- Costura de tiling: `detileToImage` / `retileToMemory` / `canRetile` en `Renderer::Impl` (donde entra
  `GpuTiler::detile/retile`). Hoy: lineal completo por copias de GPU; tileado = detile en CPU al grabar
  (como antes) y SIN retile (se notifica y se conserva la copia GPU).
- Tests (Linux, backend nulo con captura de comandos): `vpps4.tracker_*` y `vpps4.coherence_*`
  (rt_to_texture, raw_write_to_texture, raw_write_to_layer, texture_to_raw, dma_to_texture, cpu_write,
  alias_formats, alias_targets, raw_depth_clear, tiled_no_retile, eviction, host_watches).

## Pendiente / riesgos
- AstroVisionPro: espejo con mach_vm_remap, `ReadMemory`, `watchWrites/watchOwned`, `view` (ver abajo).
- retile de superficies tileadas (RT/depth reales son tileados) depende de `GpuTiler::retile` (area tiling).
- CPU detile al grabar lee memoria que trabajo GPU ya grabado puede escribir todavia (nota en el log).
- Los recursos con `formatView` (usage) pueden perder compresion sin perdida en GPUs Apple.
