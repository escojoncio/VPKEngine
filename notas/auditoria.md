# Auditoría de gestión de recursos (2026-11-11, sobre main 4593325)

Estado (2026-11-13): **CERRADA. Todos los hallazgos de abajo (gravedad A, rendimiento y el resto) están ARREGLADOS** (ver «Estado de los arreglos»); los lotes
4 y 5 (`gpu/shader`, `runtime`, `tools`, `platform`) se auditaron después (`auditoria-shader.md`, `auditoria-runtime.md`, `auditoria-misc.md`, `auditoria-x86.md`).
El texto de cada hallazgo es el original de la primera pasada (11-11), no se ha reescrito.
Todo se ejecutó en Linux con el backend nulo; los `.mm` de Metal están razonados, no ejecutados.
Repros y benchmarks: no están en el repo (sólo locales); cada hallazgo describe el escenario para reescribirlos como test.
Gravedad: A = caída, corrupción, carrera o fuga que crece; B = desperdicio real por frame, fuga
acotada o UB que hoy funciona; C = menor. "Linux" = comprobable con el backend nulo.

## Herramientas (gcc 13 ASan/UBSan/LSan/TSan; clang 18 no trae compiler-rt; clang-tidy)
- T1 [A] `gpu/vulkan/src/pending.cpp:164` (vía `vpvk_dump_pending`, `entry.cpp:4056`): `submit.unlock()` incondicional sobre un `unique_lock` tomado con `try_to_lock`; con otro hilo en `vkQueueSubmit` lanza `std::system_error` → `std::terminate`. Repro `asan-growth/repro_dump_pending.cpp` (aborta 3/3). Arreglo: `if (submit.owns_lock()) submit.unlock();`. Sin verificar si el host la llama.
- T2 [A] `tools/vpaot/image.cpp:48,54` (`unwrap_self`): phoff/phnum y tamaños de segmento del SELF sin validar → lectura fuera del búfer y escritura de 120 B en un vector de 8 (vpaot va dentro de vpconvert). Repro: SELF truncado a 128 B, `cpu-fuzz/`. Arreglo: las comprobaciones de `runtime/vp_loader.c:33-51`.
- T3 [B] `runtime/vp_loader.c:113-114→130` y `168→173`: `vaddr+memsz` y `target+8` desbordan 64 bits → memcpy fuera de la imagen, antes del fingerprint (:219). Mismo patrón en :146,153,154,157,158 sin verificar. Arreglo: comparar por resta y rechazar `vaddr+memsz < vaddr`.
- T4 [B] `gpu/core/include/vpgpu/core/device.h:83` `frame_`: escrito en `Device::submit` (`device.cpp:296`), leído sin sincronizar por `submittedFrames()` (`pipelines.cpp:813`). Arreglo: `std::atomic<uint64_t>` relaxed.
- T5 [B] `tools/vpaot/translate.cpp:1974` PMULLW, `:2051` PHADDD, `:2019` PSIGND, `:1977` PMADDWD: el C emitido desborda enteros con signo (los tests pasan). Arreglo: carriles sin signo. Cambia la salida del traductor.
- T6 [B] `runtime/vp_cpu.h:312`, `translate.cpp:2401`: `a >> (bits - n)` con n > bits (SHL de 8/16 bits, SHLD de 16) → desplazamiento ≥ 64. Arreglo: acotar n.
- T7 [C] `gpu/vulkan/src/precompile.cpp:61`: `&w[i+1]` con i+1 == size(). Arreglo: `w.data() + i + 1`.
- Limpio: 353 tests con ASan/UBSan/LSan en 3 pasadas, 0 errores de memoria; LSan 32 informes, todos handles que los tests no destruyen; TSan 27 informes con una sola raíz de test (`NullBackend::onCompile`); crecimiento del heap en 3000 frames y 3000 cargas: 0 B por iteración; clang-tidy 196 avisos, 0 reales; fuzz de ELF y `.eh_frame`: 0.
- No ejecutado: sanitizers de clang, MSan, Metal, `tools/vpconvert/vpconvert.cpp`, `tests/aot/pack` y `realbin`.

## Lote 1: core + metal2 + metal
- C1 [A, Metal] `gpu/metal2/src/metal_backend.mm:2010-2061` (llamadas en 6754, 7960): en trim, `releaseMsaaBackings` libera todo respaldo diferido no pinned con `usedFrame != frameIndex` sin exigir racha tile-local; `usedFrame` solo se fija como attachment (5400) → un RTT o sombra guardado y leído frames después se retira con contenido vivo tras `vpvk_trim_memory`. Arreglo: exigir además `tex.streak >= 1`.
- C2 [A, poco probable, Linux] `gpu/core/src/device.cpp:121-145` + metal2 `:7577,7599,7669`: `Device::destroy` suelta `resMutex_` antes de llamar al backend; otro hilo reutiliza el índice y su create llega antes → entrada viva sobrescrita sin `retire()`. Alcanzable: `vkCreateSampler` no toma `submitMutex` (`gpu/vulkan/src/entry.cpp:2678`). Repro `audit1/race.cpp`. Arreglo: `backend_.destroyX` dentro de `resMutex_`.
- C3 [B, Linux, rendimiento] `gpu/core/src/frame_compiler.cpp:583-628` `BarrierPlacer::access`: O(n²) por recurso compartido y frame, bajo `resMutex_` en el hilo de submit. Medido: 1800 encoders que leen 10 buffers = 58 ms (900 = 17, 300 = 3,2); copias desde un staging: 2000 = 23 ms, 5000 = 109 ms. Bench `audit1/`. Arreglo: registros de escritura aparte.
- C4 [B, Linux, rendimiento] `frame_compiler.cpp:414-418` `Scheduler::copy`: cada copia recorre todos los encoders previos. Medido con 1800 pases: 39 ms con copia tras cada pase, 9,8 sin copias. Arreglo: último encoder en conflicto por recurso.
- C5 [B, Metal] `gpu/metal/src/metal_backend.mm:656-663,691-698` frente a 842-846: `pipelineMutex` durante la compilación y `submit` lo toma por draw; `saveArchive` (1473-1481) igual. Arreglo: mutex propio del archivo.
- C6 [B, Metal] `gpu/metal` `:758→1280-1281` frente a 1450: `orderMutex` desde crear el CB hasta el commit, incluido `waitForeground()`; posible interbloqueo con `encodeInterop` (sin verificar).
- C7 [B, Metal] metal2 `:2357-2369`: el bloque de `prewarmShader` no abre `@autoreleasepool` (seed 2798, knock 7269 y save 9256 sí) → temporales de compilación acumulados en la ráfaga de carga. Arreglo: `@autoreleasepool {}`.
- C8 [B, Metal, rendimiento] metal2 `:5693`: `track.recs.reserve(size + encDraws)` por encoder realoca y copia todos los `DrawRec` previos, O(encoders × draws) (300 × 3000 ≈ 40 MiB copiados por frame, estimado). Arreglo: un reserve por frame o ninguno.
- C9 [B, Metal] metal2 `:2464-2546,2570-2631,2695-2740,3039-3066`: pipelines internos (clear, blit, fov, D24) compilados en el hilo de submit con `tableMutex` y fuera del binary archive → se recompilan en cada ejecución. Arreglo: cola como `seedPipeline`.
- C10 [C, Metal] metal2 `:4120-4133`: `acquireSlot` libera `slot.retired` en el hilo de submit con `tableMutex`.
- C11 [C, Metal] metal2 `:4089-4101,2196-2197`: esperas por sondeo (`sleep_for` 50 µs; prewarm 2 ms).
- C12 [C, Metal] metal2 `:3007-3017`: sombras de foveado antiguas solo se retiran con más de 8 formas; trim no las toca.
- C13 [C, Metal] metal2 `:5949-5961,5002-5017`: vista de textura nueva por encoder sembrado y por capa de blit, cada frame.
- C14 [C, Metal] `gpu/metal` `:1446-1469`: el CB de `encodeInterop` no cuenta en `inFlight`; su handler usa `impl` tras `~MetalBackend`.
- C15 [C, Metal] `gpu/metal` `:731-735,1275`: un frame sin encoders no avanza `completed`.
- C16 [C, Metal] `gpu/metal` `:1304-1393,1432-1442,1471`: entradas públicas sin pool de autorelease (sin verificar).
- C17 [C, Linux] `gpu/core/src/device.cpp:605,615`: `static slowLogged/slowSeen` sin sincronizar, compartidos entre dispositivos.
- C18 [C, Linux] `gpu/metal2/include/vpgpu/metal2/archive_index.h:53-56`: `read` acepta 2^28 claves y hace `resize` antes de leer → un `.keys` dañado reserva hasta 2 GiB.
- Dudosos: metal2 `:6904` ciclo CB → bloque → FrameTail; `:1677` `drainIdle` con solo el último slot asentado; `:6372` `carryWait` local al encoder; `:5184,6903` tiempo de GPU negativo convertido a uint64.

## Lote 2: Vulkan (rutas en `gpu/vulkan/src/`)
- V1 [A, Linux] `entry.cpp:3737-3748,3754-3763,3789-3798,2724-2733`: `liveDevice()` se copia y `liveDeviceMutex` se suelta antes de tomar `d->zeroNativeMutex`/`submitMutex`; `vkDestroyDevice` (1758-1786) acaba en `delete d` → use-after-free si otro hilo está en `vpvk_image_zero_pending`. Arreglo: tomar el mutex del device con el registro aún tomado; `vkDestroyDevice` drena `zeroNativeMutex`.
- V2 [A, Linux, confirmado] `entry.cpp:3667-3671`, `ext.cpp:57-65`: `Semaphore::pending` crece 16 B por submit con timeline semaphore (solo se poda en GetCounterValue/WaitSemaphores) y `waitMet` lo recorre en cada submit. Repro `audit2/repro_timeline.cpp` (20000 submits → 20000 entradas). Arreglo: podar en `runBatches`.
- V3 [A, con 2 devices vivos] `recipes.cpp:423-424,641-650,685-693,764-777,1019`, `entry.cpp:1701`: cada device tiene su `RecipeStore` sobre el mismo `recipes.bin`; offsets calculados desde `size_` privado con el fichero en "ab", sin comprobar `fwrite` ni el hash en `fillSource` → MSL de otro registro o registro partido. Arreglo: un store por directorio en el proceso; verificar hash; comprobar escrituras.
- V4 [A, con 2 devices] `entry.cpp:1491-1512` (`texHistTick`, statics `std::string`), `3506-3517` (static `PerfLine`), `pipelines.cpp:3588-3591` (static bool): dependen del `submitMutex` de cada device → corrupción con dos. Arreglo: un mutex estático y atomics.
- V5 [B, grande, Linux, confirmado] `pipelines.cpp:1076-1100`, `commands.h:239-242`, `pipelines.cpp:4812` (`vkTrimCommandPool` vacío), `entry.cpp:2918-2925,2980-2986` (RELEASE_RESOURCES ignorado): `UploadArena::reset` solo rebobina; los trozos mueren con el command buffer. 10 subidas 4096² → 1024² = 40 MiB residentes tras submit, reset y trim (justo en poca memoria); `Recorder::ops` (952 B por op) tampoco encoge. Repro `audit2/repro_arena.cpp`. Arreglo: liberar los trozos que pasen de uno en reset/trim.
- V6 [B] `recipes.cpp:877-941,946-1042,1044-1059`, `entry.cpp:3996-4006`: `mutex_` del store durante E/S del fichero entero (rewrite; snapshot de 100-250 MB) → bloquea `vkCreate*Pipelines`, workers y `vpvk_mem_stats` (que tiene `liveDeviceMutex`). Arreglo: escribir el tmp desde una copia fuera del lock.
- V7 [B] `commands.cpp:2358-2361` → `pipelines.cpp:1039-1048`: modo síncrono + MSAA forzado: una variante `.msN` esperada (hasta 2 s) por pipeline nuevo, en serie, en el hilo de submit. Arreglo: lote como `awaitFreshVariants`.
- V8 [B, Linux] `entry.cpp:3630-3642`: línea `vk.submit` por cada submit con draws saltados, sin tope → log sin límite y E/S en el hilo de submit. Arreglo: las N primeras y luego 1 de cada 600.
- V9 [B] `pipelines.cpp:2177-2178`: con la máscara UI activa cada pipeline retiene `sources` (SPIR-V de todas las etapas) de por vida. Arreglo: solo el fragment.
- V10 [B, medido] `pipelines.cpp:3476-3494`, `commands.cpp:2347-2372`: heap por draw (push_back sin reserve: 968 → 634 µs por frame con 3000 draws × 12 bindings; copia doble del draw con MSAA forzado).
- V11 [B, medido] `pipelines.cpp:1359-1392`: descriptor set = vector de vectores (407 ns por alloc+free; 1,2 ms por frame con 3000 sets).
- V12 [C, medido] `pipelines.cpp:1401`, `entry.cpp:2950`: `std::remove` sobre toda la lista por handle liberado (2,7 ms para 3000 sets).
- V13 [C] `commands.cpp:759,1685-1689,1748`, `pipelines.cpp:531-575`: nodo de `unordered_map` por draw de ojo; la puerta hace emplace por draw mientras dura el prewarm.
- V14 [C] `pipelines.cpp:1097,1082`: `put(nullptr)` hace memset de regiones que luego se sobrescriben enteras; `aligned_alloc` sin comprobar.
- V15 [C] `recipes.cpp:440-446`: `load()` lee el fichero en un vector creciente (~2× su tamaño); `inputs_` vive toda la sesión.
- V16 [C] `recipes.cpp:1204-1234`: el prewarm sondea `cache.status()` por clave cada 15 ms.
- V17 [C] `pipelines.cpp:3790-3835`: `noteIndirectArgs` sigue escaneando tras llegar al tope del log.
- V18 [C, foveado] `config.cpp:267-282`: `mapFor` reconstruye el RateMap por pase de ojo.
- V19 [C] `commands.cpp:855-884,1193-1217`: `linearSync` copia todos los mips antes y después de cada op de transferencia.
- V20 [C, Metal] `taa_native_metal.mm:50-60,196,238-242`: vista de textura nueva por entrada y frame.
- V21 [C, Metal] `taa_native_metal.mm:175-192,198-212,256-258`: las texturas de historia y `depthOut[8]` no se liberan al apagar TAA ni en trim.
- Dudosos: `entry.cpp:1866` asignaciones pequeñas sin poner a cero; `:1677` setenv dentro de vkCreateDevice; `pipelines.cpp:379-395` `resolvePipeline` sin back-off; `:2575-2576` sin `shaderMutex`; `entry.cpp:1926` lee `m->images` sin `submitMutex`; `ext.cpp:476-503` PrivateSlot nunca se borra.
- No leído entero: tablas de `physical_device.cpp` y de `entry.cpp` 839-1425, `vpcap.cpp`.

## Lote 3: GLES (G) y PS4 (P)
- G1 [A] `gpu/gles/src/device.cpp:169-193`: `waitSerial`/`retire` no contemplan `VK_ERROR_DEVICE_LOST` → bucle infinito al 100 % con `Device::lock` tras perder el dispositivo; con `vpgles_flush(0)` inFlight no se retira (OOM). Sin ejecutar. Arreglo: tratar r < 0 como completado.
- P1 [A, Linux] `gpu/ps4/src/renderer.cpp:628-659,767,838-856,994-1002,368-370`: targets, texturas, samplers y `translated` no se desalojan nunca → la memoria GPU crece por carga o cambio de resolución. Arreglo: desalojo por `lastUse`.
- G2 [A, raro, Linux] `gpu/gles/src/egl.cpp:507-510`: `eglWaitGL` hace `flush(true)` sin `Device::lock`. Arreglo: `lock_guard`.
- G3 [B, medido] `gles/src/device.cpp:458`: `occBegin` hace `endPass()` en cada `glBeginQuery` → 10 queries = 11 pases (más resolve y siembra con MSRTT). Arreglo: resetear los slots fuera del pase.
- G4 [B, medido] `gl_api.cpp:2153,2348,2117`, `textures.cpp:232-244`, `device.cpp:626`: `endPass()` incondicional en `glFramebufferTexture*`, `glDrawBuffers`, `glDeleteFramebuffers` sin cambios y en toda subida → 11 pases en vez de 1. Arreglo: quitarlos (`beginPass` ya compara `PassKey`); transferencias ajenas al pase en un CB previo.
- G5 [B, medido] `draw.cpp:841-842`: todo programa retiene los AST de sus etapas (FS de 14,4 KB = 716 KB; 2000 programas ≈ 1 GB). Arreglo: solo si `separable`.
- G6 [B, medido] `draw.cpp:1950-2103,1437-1641,1696-1719`, `device.cpp:277-280,600`: cada draw rehace todo (17 llamadas Vulkan, 14 redundantes, más alloc y update de un descriptor set). Arreglo: caché por contexto de lo ya emitido.
- G7 [B, medido] `draw.cpp:2330-2334,2401-2414`: `glDrawElements` recorre y copia todos los índices aunque estén en un buffer enlazable (2000 draws × 600 índices = 4,0 ms por frame).
- G8 [B, medido] `device.cpp:585-598,574-576`, `context.h:796`: los trozos de stream de 4 MiB no se reciclan (`spareChunks_` sin usar) → create + alloc por flush.
- G9 [B, medido] `device.cpp:552-556`: `flush()` sin CB abierto no suelta `keep_` → borrados retenidos hasta el siguiente comando GPU del contexto.
- G10 [B, medido] `device.cpp:613-623`, `gl_api.cpp:1002,1057-1061`: `prepareWrite` copia el buffer entero por cada escritura parcial a un buffer en uso.
- P2 [B, Linux] `gpu/ps4/src/renderer.cpp:315-324,968-979,1084-1086`: el staging de texturas consume el anillo de 16 MiB de los draws → "upload ring full" y draws saltados.
- P3 [B, medido] `gpu/ps4/src/tiling.cpp:191-242`: destesado elemento a elemento: 1080p RGBA8 = 35,6 ms por subida.
- G11 [B, medido] `draw.cpp:942-953`: `logUniformLenient` construye cadenas + mutex + map en cada `glUniform` permisivo (123 ns por llamada; 1-2 ms por frame en UE4).
- G12 [B, medido] `gl_api.cpp:2443-2455,2580-2592`: compilar y enlazar con `Device::lock` tomado (5,2 + 2,2 ms por programa).
- G13 [B] `egl.cpp:534-543`, `gl_api.cpp:3628-3641`: `ClientWaitSync` ignora `timeout` y espera con el lock global.
- P4 [B] `renderer.cpp:1568-1582`: `~Impl` no destruye guestBuffers, targets, texturas ni samplers.
- G14 [B] `egl.cpp:200-209`: `eglTerminate` no libera nada.
- P5 [B] `renderer.cpp:544-574`: `shader()` reconstruye un `std::map` por draw y etapa.
- G15 [C] `gl_api.cpp:2534,2604`, `draw.cpp:709,2845-2852`: `flush(true)` en `glDeleteProgram`/`glUseProgram`; `glReadPixels` a PBO espera a la GPU.
- G16 [C, confirmado con ASan] `gl_api.cpp:1035-1074`: `glMapBufferOES`/`MapBufferRange(len 0)` sobre un buffer sin datos desreferencia nulo (:1074). PBO pequeño y offset de XFB: sin verificar.
- G17 [C] `textures.cpp:436-444,230-231`: una subida sin comprimir = 2 vectores a cero + 3 copias.
- G18 [C, medido] `gpu/gles/glsl/parser.cpp:1794-1797,1039-1048,553-564,70`, `pp.cpp:358-392,475-488`: compilador sin límites de profundidad ni de expansión (20000 paréntesis anidados = SIGSEGV).
- P6 [C] `renderer.cpp:1044`: memset de 64 KiB por cada enlace de un V# nulo.
- Dudosos: `BufferStorage::lastUse` global con `used_` por contexto; XFB crea un buffer scratch y corta el pase por draw; `share->labels` sin borrar; `vpgles_wrap_image_as` sin `keep()`; anillo PS4 reutilizado por cuenta y no por compleción.

## Estado de los arreglos (2026-11-13)
Ciclo por zona: corrige Sonnet, revisor nuevo hasta una ronda sin bloqueantes ni importantes. Probado solo en Linux con backend nulo; nada en visor ni macOS.
- ARREGLADO T1 (`pending.cpp:164`, `unlock` solo si `owns_lock()`), T2/T3 (validación en `image.cpp` y `vp_loader.c`, ver `auditoria-runtime.md`), T4 (`frame_` atómico,
  `device.h:86`), T5/T6 (cambian la salida de `vpaot`), T7 (`precompile.cpp:61`, `w.data() + i + 1`).
- ARREGLADO Metal: C1 (trim exige `streak >= 1`), C6 (`waitForeground()` antes de `orderMutex`, `DeferredCommits`), C9 (pipelines internos en `warmQueue`), C15 (`CompletionFrontier`);
  C2-C5, C7, C8, C10-C14, C16-C18 también (ver bitácora, «Riesgos de los arreglos»). Los `.mm` nunca se han compilado.
- ARREGLADO Vulkan: V1-V4 (A), V5 (`RELEASE_RESOURCES` con histéresis: `kReleaseFrames` = 8, `commands.h:479`), V6-V21 (V7: `awaitFreshVariants` espera un solo presupuesto, `VPVK_PIPELINE_DRAW_WAIT_MS`).
- ARREGLADO GLES: G1 (`Device::retire`, `device.cpp:256`: un error negativo como `VK_ERROR_DEVICE_LOST` cuenta como completado), G2, G3 (`glBeginQuery` ya no corta el pase), G4, G6-G10
  (G7: `drawElements` con índice en sitio sin copia; G10: `glBufferSubData` parcial = copia GPU ordenada, `kOrderedPiece`, `BufferStorage::readsRange`), G11-G17, G14 (`eglTerminate` libera con el último).
- ARREGLADO G18 (compilador sin límites): topes de preprocesador y parser con error en vez de SIGSEGV (`parser.cpp` `kMaxNest` 1200 / `kMaxChain` 20000; `pp.cpp` tokens,
  fuente, salida, expansión, token pegado); valores y estado final del preprocesador en `auditoria-shader.md`, sección «Preprocesador/parser GLSL».
- ARREGLADO PS4: P1 (evicción de targets: `Hooks::targetBudgetBytes`, `enforceTargetBudget`, `renderer.cpp`; solo si el host llama a `colorTargetAt/Slice`), P2-P6 (ver `auditoria-gles3.md`
  para P5/robustBuffers).
- Dudosos de cada lote: los que se confirmaron se arreglaron; los demás no se tocaron (sin evidencia).

## Cómo seguir
La auditoría está cerrada: no queda nada abierto de esta lista. Quedan solo los «Menores aceptados» de `bitacora.md`:
- pp: macro objeto cuyo cuerpo acaba en `F (` con la llamada partida en líneas da «unterminated» (cpp acepta; ya en `main`); `#define`/`#undef` dentro de una llamada abierta se aplica al
  cerrarla (UB C99 6.10.3p11, difiere de gcc); lexer de CI con falsos negativos exóticos.
- GLES: 8 MiB enteros enlazados = 65 pases (aceptado, igual que ANGLE); el backend nulo no valida los bytes de las copias ordenadas.
- Tests intermitentes previos: `gpu_spirv.malformed` bajo TSan (tiempo), `copyhoistscale`, `host_races`.
- Sin compilar/probar nunca: todos los `.mm`, Swift, `tools/vpconvert`, rama APPLE de `gpu/vulkan/CMakeLists.txt`, `robust_vec.metal`, workflows. Sin medir en visor: latencia Vulkan,
  robustBuffers en Metal real, evicción PS4 en visionOS. Primer paso: el primer `[build]` en CI macOS.
Antiguo plan (histórico):1. Arreglar primero lo demostrado en Linux, con test que falle antes: T1, T2, T3, T4, T7, C2, V1, V2, V5, V8, G2, G16.
2. Rendimiento medido: C3, C4 (core); G3, G4, G6, G7, G8 (GLES); V10, V11 (Vulkan).
3. Metal (solo compila en CI de macOS, validar en visor): C1, C7, C8 son cambios de pocas líneas.
4. T5/T6 cambian la salida de `vpaot`: implican recompilar las traducciones.
5. Auditar los lotes 4 y 5.
