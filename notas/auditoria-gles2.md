# Auditoria GLES, segunda ronda (G4, G6, G10, G12, G14, G15, G17)

Zona: `gpu/gles`. Tests nuevos en `gpu/gles/tests/gles_tests.cpp`, grupos `audit2_*` (backend nulo, Linux).
No se ha tocado `gpu/vulkan`. Estado: ver "Verificacion" al final; nada se da por verificado.

## G4. Subidas de texturas cortando el pase

Diseno: las subidas (`texImage`, `texSubImage`, `copyBufferToImage`) y la inicializacion de imagenes no se graban
al instante si hay un pase abierto: se encolan en `Context::xfers_` (`queueUpload`, `deferImageInit`) y se graban
en `endPass()` (`recordTransfers()`: una barrera antes, una despues, y una entre dos escrituras a la misma imagen).
Con la mantenida viva por `keep()`. El orden respecto a draws anteriores se conserva porque el pase se cierra
antes de grabar y los draws previos no usan la imagen pendiente.
Puntos de invalidacion (cuando SI hay que cerrar el pase):
- `beginPass`: attachment (o seed) con `Image::xferPending` -> `endPass()`.
- `bindResources` -> `imageUsed()` (textura muestreada o storage con subida pendiente) pone `xferHazard_`;
  `prepareDraw` hace `endPass()` y reintenta una vez (bucle en el wrapper `prepareDraw`, cuerpo en `prepareDrawOnce`).
- `blackTexture()` y `flush()` (si no hay CB y hay xfers, abre uno) graban lo pendiente.
- `readPixels`/`copyTexImage` pasan por `endPass()`/`flush()` como antes.
Test: `audit2_xfer` (10 subidas intercaladas con draws = un solo pase; textura usada -> `E U B D`; attachment; lectura).
Nota: una textura NPOT/con mips sin completar se sustituye por la negra y no cuenta como uso (comportamiento previo).

## G6. Estado dinamico por contexto

`Context::Told` guarda lo que el CB abierto ya sabe: pipeline, vertex buffers, viewport, scissor, blend, bias,
referencia/mascaras de stencil por cara, firma del descriptor set y copia del bloque por defecto. `setDynamicState`
solo emite los cambios. No se escribe sobre el `BoundState` de VPVulkan (que sigue su propio cache en
`vkCmdSet*`); es una segunda capa por encima, por eso nunca puede desincronizarse con el.
Invalidacion: `told_ = Told{}` en `cmd()` (CB nuevo) y en `flush()`. `beginPass`/`endPass` no la tocan salvo
que el dinamico cambie con el rendering (viewport/scissor se reemiten al empezar un pase: `Told` se anula alli).
Test: `audit2_dyn` (contadores de `vkCmdSet*`, `vkCmdBindPipeline/DescriptorSets/VertexBuffers` en draws repetidos
y tras cambios individuales).

## G10. Copia completa en prepareWrite

Diseno: `BufferStorage` lleva los rangos que la GPU en vuelo puede leer (`reads`, `noteRead`, `readsRange`).
`useStorage(s, lo, hi)` los anota (VBO por rango de filas, indices, UBO por `offset..offset+range`);
se limpian cuando `lastUse <= completed`. `prepareWrite(buf, discard, lo, hi, overwrites)`:
- escritura en sitio si la GPU no lee `[lo,hi)` (BufferSubData/Map de zonas no usadas por los draws en vuelo);
- si la lee: COW como antes, pero con `overwrites` (BufferSubData) no copia `[lo,hi)`.
Se descarto la copia GPU: obligaria a cortar el pase o a una barrera por draw.
`vertexRowsUnknown_` marca draws sin escaneo de indice max (`drawElements` sin `needMax`, indirectos): rango = buffer entero.
Test: `audit2_buf` (sin buffers nuevos al escribir otra zona; COW solo si solapa; contenido correcto).

## G12. LinkProgram sin el lock global

`LinkProgram` junta las entradas (`LinkInputs`: fuentes/atributos/xfb/binary hints), suelta `Device::lock`, ejecuta
`vpglsl::link` (`Context::linkAhead`, estatica, sin tocar el contexto), vuelve a tomar el lock, busca el programa de
nuevo (`c->prog(pn)`), y si las entradas no cambiaron (`operator==`) pasa el resultado a `linkProgram(..., &ahead)`;
si cambiaron (otro hilo reconfiguro) enlaza de nuevo con el lock. Aun se hace bajo lock el resto (reflexion, objetos Vulkan).
Test: `audit2_link` (un hilo enlaza un shader grande mientras otro hace >100 llamadas GL); `audit2_link2` (dos hilos, dos contextos
compartidos, enlazan a la vez programas que comparten los mismos shaders; pensado para TSan).

## G14. eglTerminate no liberaba nada

`egl.cpp` registra contextos, superficies, syncs e imagenes (`gContexts/gSurfaces/gSyncs/gImages`). `eglTerminate`
(`releaseDisplay`): borra contextos no actuales, marca `dead` los actuales (se liberan en su `MakeCurrent(NONE)`),
libera superficies sin enlazar, syncs e imagenes, espera `submitted` y llama `Device::trim()` (pools libres y chunks
de stream de reserva). Si el host reutiliza el display tras terminate, `eglInitialize` lo recrea perezosamente
(el Device y las colas viven; solo se recortan caches). Handles invalidos: `BAD_CONTEXT/BAD_SURFACE/BAD_PARAMETER`.
Test: `audit2_egl`.

## G15 (resto). glReadPixels a PBO

`readbackIsCopy` (formats.cpp): RGBA/UNSIGNED_BYTE desde R8G8B8A8 UNORM/SRGB sin swizzle de alpha = copia byte a byte.
En ese caso, con offset y fila alineados al texel, `readPixels` copia con `vkCmdCopyImageToBuffer` directo al storage
del PBO (`bufferRowLength` para el paso de fila), marca `gpuWrite = submitted+1` y vuelve sin `flush(true)`.
Cualquier acceso CPU posterior (`prepareWrite`, map, getBufferSubData) espera a `gpuWrite`. El resto de formatos y
casos desalineados siguen por el camino antiguo (flush y conversion en CPU).
Test: `audit2_pbo` (0 esperas, copia al buffer, datos al mapear, caso desalineado).

## G17 (resto). Vector convertido y copia al stream

`convertUploadTo(..., uint8_t* out)` (formats.cpp) convierte directo a memoria; `convertUpload` lo envuelve.
`texImage` (no packed depth-stencil, no `early`): `stream()` primero, convierte dentro del chunk y `queueUpload`.
Test: `audit2_upload` (una subida grande no asigna vector de >=64 KiB: `operator new` global contado).

## Dudas y riesgos conocidos
- La validacion de handles EGL puede rechazar hosts que reusan handles tras `eglTerminate`.
- G10: el caso solapado sigue copiando en CPU (COW), solo se evita en el no solapado.
- G15 solo cubre RGBA8/UBYTE con offset y fila alineados.
- Bug previo (no mio): `allocSet` puede hacer flush a mitad de `prepareDraw` (pool de 1024 sets); ahora lo cubre el
  reintento por cambio de `flushes_`.
- `clearByDraw` no mira `xferHazard_` (los attachments ya se comprueban en `beginPass`).
- `useStorage` con varios contextos compartiendo storage deduplica por contexto (previo).
- No se han lanzado CTS/dEQP. Tests de rendimiento (contadores) dependen de la ruta del backend nulo.

## Cambios en tests existentes (por diseno de G4)
- `draw` (mip streaming): las subidas ya no se graban al instante dentro de un pase abierto; se anade `glFinish()` antes de contar copias.
- `msrtt_depth`: el corte "subida de textura" (cut 2) subia a una textura ajena; ya no corta el pase. Ahora sube a la textura de color
  adjunta (que si corta). Las otras comprobaciones no cambian.
- `egl`: destruye el sync que crea (LSan lo veia como fuga del test; antes tambien lo era).
- El `operator new` global del test (cuenta asignaciones >= 64 KiB) reemplaza toda la familia (malloc/free) para que ASan no
  vea new/free cruzados.

## Verificacion (Linux, gcc, backend nulo)
- Los 7 `audit2_*` (sin `audit2_link2`, que es de carreras) fallan contra un build de `HEAD` de `gpu/gles` con el mismo fichero de tests (build -O0 aparte) y pasan con
  los cambios. `draw` y `msrtt_depth` pasaban en `HEAD` y se ajustaron como arriba.
- `ctest -R "gpu_gles|vpps4"` (Release): todo verde (incluidos `gpu_gles.audit.*` tras compilar sus ejecutables).
- ASan+UBSan (`halt_on_error`, LSan activo): los 36 grupos `gpu_gles.gl.*` verdes. TSan: los 36 verdes, incluido `audit2_link2`.
- No ejecutado: CTS/dEQP, ni GPU real (los tests de rendimiento cuentan llamadas Vulkan del backend nulo).
