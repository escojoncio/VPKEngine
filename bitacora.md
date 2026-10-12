# Bitácora — VPEngine

Motor para juegos de PS4 en Apple Vision Pro sin JIT: traductor AOT x86-64 → C (`tools/vpaot`), runtime
(`runtime/`), conversión en el visor (`tools/vpconvert`: vpaot + clang + lld en proceso), firma/carga de paquetes
(`runtime/vp_codesign.c`, `vp_pkcs12.c`, `platform/visionos/Sources/VPGamePack.swift`, `VPConversion.swift`), motor
para shadPS4 (`integrations/shadps4/aot_guest_engine.cpp`, sustituye a FEX en la build `visionos-vpengine` de
AstroVisionPro). Plan: `docs/PLAN.md`. Bloodborne: `Bitacora BB.md`. Historial detallado: `git log -p bitacora.md`.

## Estado (2026-10-12)

| Parte | Estado |
|---|---|
| Traductor | ISA completa de Jaguar (2174 formas idénticas al hardware Intel y AMD), x87 80 bits exacto, AVX-256/F16C, atómicos LOCK, excepciones C++, fibers/corrutinas |
| Conversión en el visor | Funciona (~17–27 min el juego entero; reanudable; reutiliza objetos por hash de C) |
| Firma y carga | Aceptada por visionOS con la variante `req4k` |
| Astro en el visor | Eboot, libc y fios2 enganchados; el juego corre traducido hasta código no descubierto estáticamente |
| Traducción bajo demanda | Escrita y probada en Linux (e8a1fc7); **sin build ni prueba en el visor** |
| CPU (traducción AOT) | Aparcada: el límite de Astro es la GPU (~30 ms/imagen, ver bitácora de AstroVisionPro) |
| **GPU nativa Metal (foco)** | `gpu/`: core, VPIR, emisor MSL (con guarda de bucles), backend Metal, GCN, vpps4, VPVulkan. 122/122 tests Linux; CI visionOS en verde. En el visor (ASTRO BOT, sombra): frame 0 OK, hang de GPU en frame 1 → guarda de bucles (sin probar) |

## 2026-10-12 — VisionQuestPro logs 27/28: compilación lenta, caché Metal que no se guarda, MSAA→1x roto
- Log 28: congelaciones `gpu.hold` de hasta 1,35 s (un PSO tarda 0,3–1,3 s; antes de los slots ≤0,45 s);
  `archive_save` falla casi siempre (`cannot pack … missing 'vertex' stage in pipeline no. N`, luego
  `Nothing_to_serialize`) → pipelines nunca persistidos. Log 27 (VPKTAA + MSAA forzado a 1x): todo rosa.
- `shader/msl/src/msl.cpp` `planSlots`: slots privados con solo accesos a dirección constante, celdas sin solapes y
  mismo tipo escalar → locales tipados `T vp_pS_OFF{}` (accesos parciales como swizzle); índice dinámico, mezcla de
  tipos, cruce de celdas o atómicos → array `uint4 vp_pN[]` como antes. Slots sin uso no se declaran. Golden nuevo
  `msl/tests/private_slots.{vpir,metal}`.
- `metal2/src/metal_backend.mm`: archivos de pipelines de solo lectura (`pipelines.metalar` + `pipelines-<n>.metalar`,
  máx. 16, todos en `binaryArchives`); los nuevos de la sesión van a un archivo delta propio con copias de los
  descriptores; validación (sin vertex/compute function → fuera); fallo al serializar → reconstrucción desde los
  descriptores, culpable por el nº del error (o bisección), máx. 4 descartes; tmp + rename; `pipelines.count` v2.
  Logs `gpu.pso.cache archives= session= evicted=`, `dropped label= reason=`, `saved=… file= entries= drops=`.
- `vulkan/src/commands.cpp` `forcedResolveCopy`: MSAA del juego forzado a 1x ya no dibuja directo en el destino del
  resolve (solo valía si el mismo pase dibujaba y resolvía; con varios pases el ojo quedaba sin dibujar → rosa):
  se dibuja en la imagen 1x y se copia al destino tras el subpase (`vk.forced_resolve`). Igual en dynamic rendering
  (`OpBeginRendering::resolveCopies`).
- Comprobar: `gpu.pso.cache saved` sin `archive_save` fallidos; 2.ª sesión sin `gpu.hold` en lo ya visto; PSO más
  rápidos; con «Desactivar MSAA» imagen correcta.

## 2026-10-12 — Recetas de pipelines sobreviven a cambios del traductor (sin probar en el visor)
- Problema: cada cambio del traductor exigía subir `kRecipeVersion` → se perdía todo el historial → siguiente
  sesión compilando en vivo (300–400 ms/PSO, `missing_pipelines`).
- `vulkan/src/recipes.{h,cpp}`: formato v3 (`kRecipeVersion = 3`, los v2 se descartan: no tienen entradas).
  Registros: `W` SPIR-V (una vez por módulo, `spirvHash`), `I` entrada del traductor (`StageInput` = hash del módulo +
  `SpirvLowerOptions` completas: entry, stage, spec constants, viewCount, layout con slots, pushSlot, atributos y
  buffers de vértices) con `kTranslatorVersion` y MSL id producido, `S` MSL (igual que antes), `Q` pipeline =
  PipelineState + hashes de entrada por etapa, `D` marca de deriva.
- `kTranslatorVersion` (recipes.h, hoy 1) separado del formato: al subirlo, `RecipeStore::refresh(threads)` (llamado
  al principio de `prewarm`, tanto en vkCreateDevice como en `vpvk_prewarm` de la app) retraduce en paralelo
  (un trabajo por módulo SPIR-V, `spirv::parse` + `translateRecipeStage`), remapea los ids de shader de cada
  PipelineState (y la writeMask de salidas no escritas, como `createGraphics`), descarta lo no referenciado y
  reescribe el fichero compactado (`recipes.bin.tmp` + rename). El siguiente arranque es acierto directo.
- Deriva: misma entrada + misma versión + otro MSL en vivo → `gpu.pso.recipe drift input= old= new=` y registro `D`
  → el siguiente arranque retraduce todo (protege contra olvidar subir `kTranslatorVersion`).
- Compactación también si >256 registros muertos y >25 % del fichero.
- `pipelines.cpp`: `lowerAndEmit` + `blobOf` compartidos por `translateStage` y `translateRecipeStage`;
  `ShaderModuleObj::words/hash` solo si el device tiene recetas activas; `addStage`/`addPipeline(st, inputs)`.
- `core/src/pipeline_cache.cpp` `workerLoop`: un trabajo duplicado descartado como último de la cola no notificaba
  `idle_` → `waitIdle` colgado (salió con el test nuevo al pedir dos veces el mismo pipeline).
- Logs: `gpu.pso.recipe read shaders= pipelines= inputs= spirv= stale_inputs= drift= dead= translator= bytes=`,
  `gpu.pso.recipe refresh retranslated= failed= dropped_pipelines= ... rewritten= ms=`,
  `gpu.pso.recipe loaded shaders= pipelines= retranslated= failed= ms=`, `gpu.prewarm ... retranslated=`,
  `gpu.pso.recipe retranslate_fail input= stage= error=`.
- Test `gpu_vulkan.recipes` (grabación real vía driver, retraducción con otra versión, remapeo de id obsoleto en
  prewarm, deriva, fichero v2 descartado, sin crecimiento al repetir pipeline). 137 tests; mismos 20 fallos previos.
- Verificar en el visor: 1.ª sesión tras esta build parte de cero (v2 descartado); la 2.ª debe mostrar
  `gpu.prewarm ... hits/compiled` con recetas y `missing_pipelines` ~0; tras el próximo cambio del traductor (subiendo
  `kTranslatorVersion`) debe salir `refresh retranslated=N rewritten=1` y no perderse recetas. Vigilar `bytes=`.

## 2026-10-12 — VisionQuestPro log 26: rendimiento OK (~90 fps estables); corrupción por pipelines en compilación
- Log 26: sin frames >12 ms; corrupción breve al arrancar y en el visor de personajes = draws saltados por
  `missing_pipelines` (hasta 227, 300–400 ms por PSO; recetas vacías tras `kRecipeVersion = 2`).
- `core`: `Device::submit` retiene el frame mientras algún pipeline que usa esté Queued/Compiling
  (`PipelineCache::waitReady`, condvar `done_` notificada en `compile`), tope `VP_HOLD_MS` (def. 1500; 0 = antes).
  Log `gpu.hold f= pipelines= ms= complete= total_frames= total_ms= timeouts=` y `gpu.hold armed max_ms=`.
  El compositor del host sigue reproyectando el último frame completo (pausa en vez de agujeros).

## 2026-10-12 — VisionQuestPro log 25: 6 ms → 39 ms/frame mirando ciertas zonas (Batman)
- Log: meseta de 38–41 ms en los frames de render (cada submit de render; los impares son solo copias), ~22 fps de
  juego frente a ~95; estructura de pases idéntica en frames rápidos y lentos → coste dentro de shaders concretos.
- Causa (código): todas las variables Function/Private de SPIR-V compartían un único `uint4 vp_private[N]` direccionado
  por bytes; un solo índice dinámico (ImmCB de Unity, bucles de luces) impide a Metal llevar el alloca a registros y
  manda TODOS los locales del shader a la pila.
- Arreglo: slots por variable. `ir.h` `kTargetPrivateSlot0 + n`, `isPrivateSlot`, `privateSlotIndex`,
  `Module::privateSlots` (tamaños múltiplo de 16; texto `private_slot N` / destino `private_N`). `lower.cpp`
  `newPrivateSlot` en OpVariable y globales Private (off relativo 0). `msl.cpp`: `uint4 vp_pN[k] = {};` por slot.
  `kTargetPrivate` (arena única) sigue para GCN. verify/print/parse actualizados. Tests Linux sin cambios (20 fallos
  previos de goldens/subpass).
- `vulkan/src/recipes.cpp` `kRecipeVersion = 2` (MSL nuevo). Diagnóstico `gpu.shader.mem id= stage= pipe= slots=
  bytes= dyn_slots= dyn_bytes= loops=` (pipelines.cpp, por MSL único, todos los que tienen slots dinámicos).
- `core/src/device.cpp`: `gpu.time` de frames lentos (>12 ms) añade `enc_ms=` por encoder (1 de cada 8).
- `metal2` `saveArchiveLocked`: serializa a `pipelines.metalar.tmp` + rename (antes sobrescribía el fichero del que se
  cargó el archivo → `missing 'vertex' stage in pipeline no. N`).
- Pendiente según el log siguiente: tablas Private de solo lectura → `constant` a nivel de programa; respetar
  `EarlyFragmentTests`/`DepthGreater|Less` (hoy `depth(any)`); perfilador automático de frames pesados (bisección por
  draws, diseño en sesión 2026-10-12).

## 2026-10-12 (noche, rendimiento CPU del hilo GPU)
Log del mundo: ~8.5 fps con `gpu.time` 0.7–0.8 ms por submit → cuello en CPU/esperas, no en GPU.
- `gpu/core`: `Backend::completedFrames()` (Null: campo `completedBelow`, por defecto 0; Metal: atómico que sube
  en el completion handler) y `Device::completedFrames()`. `PipelineCache::request(state, key, prio)` (sin rehash).
- `frame_compiler.cpp` `BarrierPlacer::access`: lecturas del mismo encoder/posición sobre el mismo recurso se
  pliegan en un registro (rango unión); escrituras solo si el rango es idéntico. Antes O(n²) por recurso con ~100
  draws por encoder.
- `renderer.cpp` (vpps4): rangos busy por submit (`inFlight`, retiran con `completedFrames`, ya no solo con
  `gpuIdle`); `recordedMayAccess` / `submittedIdle` para DMA; submit vacío no llega al Device ni rota anillos;
  anillos 8×16 MB (antes 3×32 MB con 3 en vuelo: reutilización con frame posiblemente aún en GPU);
  user data idéntica reutiliza la subida (8 copias por anillo); `note/skip(const char*)` sin construir strings;
  label de pipeline solo si no está Ready; `programKey` memorizado por puntero de código dentro del submit
  (`programGeneration`, sube en submit e `invalidate`); `CpuTimes` (`takeCpuTimes()`).
- Test `vpps4.draw_bench` (1600 draws/frame, submit cada 100): 2.8–4.4 → 1.6–2.1 µs/draw en Linux x86.
  `vpps4.busy_ranges` amplía retiro por frame completado.
- Pendiente: leer la línea `VPENGINE_GPU cpu:` del siguiente log (reparto real) y decidir siguiente paso.

## 2026-10-12 (noche): imagen negra en el casco + fallos de traducción
- Negro: `renderer.cpp` `texture()` metía `usage` en la clave de la caché → la escritura storage de un compute y el
  muestreo posterior del mismo T# (p. ej. `textureFor` de los ojos en `Present::ComposeEyes`) eran texturas distintas;
  la muestreada salía de memoria guest (nunca escrita por la GPU) = negra. Ahora una sola textura por forma; si se
  escribe por primera vez se recrea con storage (la vieja se destruye kFrames después, `retiredTextures`) y queda
  `gpuWritten` (sin comprobación de contenido ni resubida). Render targets 1x con `usage::storage` (compute que
  escribe sobre un CB). **Sin probar en el visor.**
- `lift.cpp` `decodeProgram`: camino que cae en el relleno de ceros tras el programa (todos los "undecodable word
  0x00000000 at dword N" tenían N múltiplo de 16) termina ahí con nota `a path falls into zero padding (after X)`.
  Test `gcn.lift.cs_padding`.
- `lift.cpp` `loadFormat`: nfmt 6 (snorm_nz, (2c+1)/(2^w-1)) en lecturas tipadas.
- `fetch.cpp` `fetchAddress`: si el recorrido lineal perdió el par de s_swappc, se lee de user data (como shadPS4).
  Pendiente: par cargado con s_load desde puntero (necesita leer memoria guest).

## 2026-10-12 (tarde): VPEngine único renderer de AstroVisionPro
- `integrations/shadps4/gpu`: `tiling.{h,cpp}` (destesado CPU: lineal, 1D/2D thin, display/thin/depth; thick/3D/PRT y
  tile split < microtesela se registran), `fetch.{h,cpp}` (localiza el fetch shader por s_swappc y user data),
  `Renderer::textureFor/fill/copy/gds`, `Hooks::uploaded`, indirectos (`DrawState/DispatchState::indirectAddress`),
  V# sin memoria → buffer de ceros, staging propio para copias grandes. 128/128 tests Linux.
- `gpu/core`: `Indirect` en Draw/Dispatch. `gpu/metal`: draws/dispatch indirectos, orden de la cola con
  MTLSharedEvent, `encodeInterop(encode, label, completed)`. **Sin compilar aún en Apple** (va en la build de AstroVisionPro).
- `gpu/shader/gcn`: `ds_append/ds_consume` con GDS → atómico sobre `DescriptorSource::Gds`.
- Pendiente: V# dinámico de `s_buffer_load` (tabla de páginas guest→gpuAddress), GS (ES/LS → compute), VPKTAA,
  profundidad al compositor, invalidar texturas escritas por la GPU.

## SIGUIENTE BUILD (decidido): VPEngine con imagen propia en AstroVisionPro, todo en una build
Ajuste «Motor gráfico» con 3 opciones: KosmicKrisp / VPEngine en sombra / **VPEngine (imagen propia)**. Contenido:
1. Destesado propio de texturas PS4 (GB_TILE_MODE / modos de tesela de gfx7: thin1/thick, 2D/1D, micro display/thin/
   depth): pase compute o en CPU al subir; test con imágenes sintéticas teseladas (sin datos del juego).
2. Presentación: textura de cada ojo de VPEngine (`colorTargetAt` del flip HMD + `MetalBackend::nativeTexture`) al
   presentador VR de AstroVisionPro en lugar de la de KK; profundidad real al compositor; VPEngine sustituye a KK en
   esa opción (KK no dibuja o se salta su submit).
3. Draws/dispatch indirectos (argumentos leídos de memoria de guest al grabar o con buffer de argumentos Metal).
4. Pipelines con ES/LS (geometría/teselación): como mínimo GS de ASTRO BOT (~170 draws/s) → vertex+fragment con
   amplificación o compute previo; si no da tiempo, documentar el recorte.
5. Shaders: GDS `ds_append/ds_consume` (atómico en buffer propio), V# de `s_buffer_load` no trazable (lectura
   dinámica por dirección), `cs: buffer outside guest memory`.
6. VPKTAA en la ruta de VPEngine (shader de PTVisionPro, ya en este repo; jitter en el VS traducido, profundidad y
   movimiento de cámara por pose; ver «Aportes de PTVisionPro»), con opción en la app.
Antes: confirmar con la build de la guarda de bucles que la sombra corre sin hang (`vpps4.frame`/`gpu.frame`/`gpu.time`).
Si vuelve el hang con la guarda: buscar el pase (encoders del `gpu.err`) y añadir guarda de tiempo/datos donde toque.

## gpu/vulkan: foveado del juego apagado + API de configuración (2026-10-12)
- `physical_device.cpp`: `deviceExtensions()` ya no anuncia `VK_EXT_fragment_density_map`.
- `entry.cpp`: feature `fragmentDensityMap*` = FALSE; `vkCreateDevice` descarta `VK_EXT_fragment_density_map`/`_2` si
  el juego los pide (log `vk.fov ignored=device_ext`, ya no fija `Device::densityMap`); `vkCreateRenderPass`/`2`
  ignoran `VkRenderPassFragmentDensityMapCreateInfoEXT` (`logFdmIgnored`, 8 líneas máx., `densityMap` = -1).
- Nuevo `include/vpvk/config.h` + `src/config.cpp` (en `VPVK_COMMON`): foveado off/M2/M5/personalizado, VPKTAA,
  escala de render, ASW off, perfil Quest 3; acotado, generación, log `vk.fov`/`vk.cfg`. Sin consumidor aún.
- `tests/vk_tests.cpp` (grupo physical): FDM no anunciado, habilitarlo no falla, API acotada. 5/5 `gpu_vulkan.*` en Linux.
- Pendiente (docs/VULKAN.md «Foveado y VPKTAA»): rate map en pases multiview de ojo y pase VPKTAA final.

## Hecho y verificado (referencia breve)

- **Traductor** (`translate.cpp`): una función C por función invitada, bloques = etiquetas, flags perezosos, registros
  en locales (regcache), `--pic` (base en `VP_MOD.base`), `--split`, tablas de salto PIE/absolutas, landing pads de
  `.eh_frame`, puntos de reanudación por cada `call`, límites de función por `.eh_frame`, filtro de datos-como-código
  (`impossible_in_user_code`, `--keep-data`), `--roots`, `--patch`, `--fragment/--known`.
- **Carga**: ELF, SELF de PS4, PE32+ (aparcado). `VpModule` con huella FNV-1a sobre código (sin sitios de
  relocación) → enganche por huella en cualquier base; registro por referencia (`--registry`), paquetes (`--pack`).
- **Runtime** (`vp_host.c`): `vp_dispatch` con caché por hilo y generación, `vp_run` anidado, `vp_call_guest`,
  rangos de salida, hooks del embebedor (`VpEmbedderHooks`), registro de entradas perdidas (`VPENGINE_MISSING_LOG`).
- **x87**: SoftFloat 3e renombrado `vp_sf_*` en `vp_x87.c`. **TSO**: `VP_TSO` por defecto 1 fuera de x86
  (LDAPR/STLR, 128 bits y no alineados ordenados, `dmb ish` tras LOCK, `rep movs/stos` planos entre barreras).
- **Firma propia**: CodeDirectory SHA-256, CMS, requisitos de la app copiados; PKCS#12 propio (OpenSSL 3 / iloader).
- **vpconvert**: auto-prueba de clang, recuperación de fallos de LLVM, `llvm::allocate_buffer` propio, calor
  (`max_jobs`), piezas por contenido (`<mod>_u<off>.c`), `compile_key` (cabeceras SDK + flags), `vp_convert_is_current`
  (ignora la línea `roots`), `patch_image` del embebedor (parches de KnownTitle de shadPS4).
- **Tests** (`tests/aot/run.py`, CI `aot-tests.yml` x86 + ARM): diferencial 105 casos (incluidos 57 de la auditoría del
  ISA y `jaguar_sse4a_00` con golden AMD), programa, roots, ondemand, loader, módulos, hilos, atómicos, TSO (message
  passing en ARM + comprobación LDAPR/STLR), excepciones, libstdc++, pack, codesign, data_in_text, shadps4 (motor con
  veneers, callbacks, hilos, señales, fibers, bajo demanda).

## Traducción bajo demanda (e8a1fc7, sin probar en el visor)

- Motivo: el análisis estático nunca encuentra todo (punteros calculados en ejecución); en Astro cada carga nueva
  destapaba otra entrada (`eboot+0xd2df64`, `eboot+0xd86ae2` en `SoundUpdate_ATQT`) → crash, reconvertir, repetir.
- `aot_guest_engine.cpp` `TranslateOnDemand`: fallo de despacho dentro de un módulo enganchado → handler de la app
  (`runtime/vp_ondemand.h`, fuera del SDK) → `vp_convert_fragment` → `work/ondemand/<mod>_ondemand_<hex>.dylib` →
  `VPGamePack.loadFragment` (firma, dlopen) → engancha por huella → sigue. Fallidos no se reintentan.
- Swift: `missing.txt` por juego en `VPS4/VPEngine/<juego>/`; `enableOnDemand(game:)` carga los fragmentos previos;
  `onDemandActivity` para avisar en la app. La parte de AstroVisionPro (llamar a `enableOnDemand` en `startGame`,
  `vp_ondemand.h` en el bridging header, aviso en el visor) **no está en su repo**.
- Límite de diseño: cada entrada nueva para el hilo 1–3 s. Inaceptable si es frecuente.

## GPU (`gpu/`, diseño en docs/GPU.md; todo código propio, sin shadPS4/Mesa/SPIRV-Cross/MoltenVK)

Hecho (487b201, 71/71 tests en Linux con -Werror):
- `gpu/core`: grafo de pases → encoders (frame tipo Astro 93 → 15), barreras solo por riesgo real, motivo de cada corte,
  caché de pipelines con prioridades y disco, diagnóstico `gpu.frame/gpu.cut/gpu.enc/gpu.pso/gpu.err`.
  G-buffer en tesela: lecturas `pixelLocal` fusionan pases; solo `transient` o store DontCare/Resolve va a memoria de
  tesela (`persistent` veta); validación de tamaño/muestras/slots; draws con fetch imposible se saltan con `gpu.err`;
  presupuesto `CompileOptions::tileBytesPerPixel` (64; Apple8 admite 128). Fuzz 4000 frames sin violaciones.
- `gpu/shader/ir` (VPIR): IR estructurada, texto canónico, verificador, `optimize` (plegado + CSE), `toFramebufferFetch`.
- `gpu/shader/msl`: VPIR → MSL sin comprobaciones de límites, constantes directas, `[[color(n)]]`; 11 goldens.
- `gpu/metal`: backend Metal 3 (bindless + residency set; puntos `M4:` para Metal 4), memoryless, MSAA resolve,
  stencil, tablas de argumentos por etapa según `MslShader::arguments`. **Nunca compilado en Apple**.
- `gpu/shader/gcn`: decodificador + desensamblador de **todas** las codificaciones de gfx7: SOP2/SOP1/SOPC/SOPP/SOPK/
  SMRD, VOP2/VOP1/VOPC/VOP3, VINTRP, DS, MUBUF, MTBUF, MIMG, FLAT, EXP. Corpus 139 565 entradas (0 fallos) +
  `fieldChecks()` en `gcn_test.cpp` para lo que el oráculo no juzga. llvm-mc 18 no desensambla gfx7: el corpus se
  genera ensamblando (`tests/gen_corpus.py`); lo que llvm rechaza pero el hardware ejecuta (lds, r128, ttmp como
  descriptor, tfe en FLAT/atómicos de imagen, attr > 32, GWS sin gds) queda "?" (no juzgado).
  Semántica para el lifter: `Instruction::src[4]` en orden impreso; `dst` = lo que vuelve (cargas, atómicos con glc;
  cmpswap devuelve la mitad; carga con `lds` no escribe VGPR); MIMG datos = popcount(dmask) (gather4: 4) + (tfe|lwe);
  la cuenta de VGPR de dirección de MIMG es la de 2D sin array (la real depende del descriptor: la recalcula el
  lifter); EXP: fuentes apagadas = None (compr: src[0] = vsrc0, src[1] = vsrc1); MUBUF addr64 con offen/idxen se
  rechaza. GWS: el operando va en el campo addr (como lo codifica llvm-mc; el manual dice data0; sin uso en gráficos).
- CI `gpu.yml`: linux + visionos-device (SDK xros, metallib de goldens) + visionos-simulator (ctest vía
  `gpu/metal/ci/simctl-run.sh`). Primer run: 38176237184.

- **Lifter GCN → VPIR** (`gpu/shader/gcn/src/lift.cpp`, `include/vpgcn/lift.h`, lib `vpgcn_lift`): modelo un hilo por
  carril; SGPR/VGPR/M0/SCC variables, EXEC/VCC/pares de máscara bools por carril (`findMasks`), escrituras VGPR con
  `select(exec)` si el programa escribe EXEC; bucles naturales (`findLoops`: bloques h..l contiguos, una vuelta atrás,
  una salida E; anidables) → `while (true)` con `continue`/`break`; ramas → `if` anidados por post-dominador inmediato
  sin aristas de vuelta (`walk`, pasada en seco con límite de duplicación 4×); si no encaja → bucle de despacho. Descriptores: `traceOrigins`
  (user SGPR o `s_load` a offset constante de un puntero en user data) → `Binding` (slot, origen, dwords; 0 = la fuente
  es una dirección: memoria ahí). User SGPRs: constant buffer `user_data`. Fetch (`s_swappc_b64`) en línea. PS: VGPRs
  por `SPI_PS_INPUT_ADDR`, `v_interp_p2` = `load_input` (modo por el par baricéntrico), cara frontal 1.0/0. Exports:
  mrt → color N, mrtz → depth, pos0 → Position, paramN → location N, `compr` según `LiftInput::compressedExport`; PS
  descarta con `vm`/`done` y EXEC apagado. Aproximaciones → `LiftResult::notes` (readlane, mbcnt fuera de cómputo,
  swizzle, div_scale, _cl, cubos con gradientes...). Revisión adversarial (Sonnet) aplicada: ds_cmpst (DATA = comparando),
  ffbl(0) = -1, literal SMRD en dwords, bfe definido con off+w > 32, arrays de cubos (slice*8+face), *_clamp, pk_* con
  saturación, ldexp en dos pasos, rangos.
  Pruebas: `tests/lift/*.s` (fuente) → `assemble.py` (llvm-mc a objeto + llvm-objcopy: `-show-encoding` NO resuelve
  saltos) → `.lift`; `vpgcn_lift_test` exige traducción + verificación correctas y `.txt` (bindings/notas) idéntico; el `.metal`
  no se compara (su texto depende del orden de evaluación de argumentos de gcc/clang) pero el CI lo compila con Metal.
  Comprobar también con clang en local (`CXX=clang++`): el CI de Apple da error en avisos que gcc no da.
  `compile_all.sh` también compila `tests/lift/*.metal` con Metal (CI visionos-device).
- AstroVisionPro e6cf5b2: «Modo diagnóstico» vuelca el GCN de cada shader a `VPS4/VPEngine/shaders/dumps`
  (`<etapa>_<hash>.bin`, `vs_<hash>.fetch.bin`); pendiente que el usuario los suba.

- Corpus real (volcado de AstroVisionPro e6cf5b2, ASTRO BOT + KITCHEN, 174 programas: 36 cs, 75 fs, 66 vs; NUNCA al
  repo, son datos del juego): `vpgcn_lift_batch DIR [OUT]` (`tools/lift_batch.cpp`) traduce todos e histograma de
  errores/notas; infiere user SGPRs (leídos antes de escribirse, ≤16), PS = centro + posición. **174/174 a MSL válido**,
  0 errores de verificador, todos estructurados. `fs_0x...bd149c40` usa LDS en píxel como spill de registros
  (dirección = `mbcnt`·4 + offset, cada carril lee solo lo suyo) → memoria privada por invocación (abajo).
  Lo que hizo falta (patrones del compilador de Sony): base/destino SMRD solapados (`s_load_dwordx4 s[16:19], s[16:17]`),
  punteros de dos niveles (`DescriptorSource::path`), `vcc_lo/hi` y pares de máscara también como datos (doble forma:
  u32 + bool, sincronizadas), `v_cmp_*_u64` (máscaras por bit del carril / pares por dwords), `v_mov_b32` VOP3 con
  abs/neg (decoder `11/a`, corpus "?"), `v_movrel*` (cadena de select por M0), tablas de constantes tras el código
  (`s_getpc_b64` + `s_add_u32` → `DescriptorSource::Code`, offset en bytes desde el inicio del programa),
  `v_mad_u64_u32`, `ds_swizzle` sin LDS, V# construido desde una dirección cargada (`s_or_b32 hi, stride`:
  `Origin::LoadHi`, binding con dwords 0 = memoria en la dirección guardada en la fuente; `rawBuffer`).

- VPIR `quad_shuffle v, lane` / `quad_any b` (no se fusionan en optimize; MSL `quad_shuffle`/`quad_any`; test
  `msl/tests/quad_ops.vpir`). Lifter: `ds_swizzle_b32` QUAD_PERM y BITMASK dentro del quad → `quad_shuffle` con
  `quadLane` (PS: `x&1 + 2(y&1)` de la posición; CS: índice local & 3); `s_wqm_b64` de máscaras → `quad_any`.
  Limitación: dentro de ramas divergentes el vecino puede estar inactivo (en GCN se lee igual).
- **VPKTAA en MSL propio** (`gpu/visionos/shaders/vpktaa.metal`): `vpktaa_vertex` (triángulo completo) +
  `vpktaa_fragment`; texturas 0 color (render), 1 profundidad (`depth2d`), 2 historia RGBA16F (a = distancia), 3 reactivo
  opcional, 4 movimiento opcional; `VpktaaParams` (clipToPrevClip = clip actual sin jitter → clip anterior, jitter en uv
  de render, tamaños, `viewZ` (dist = 1/(x·depth+y)), nitidez, flags HISTORY_VALID/PERIPHERY/REVERSE_Z/REACTIVE/MOTION);
  salidas color(0) y historia(1). Movimiento de cámara desde profundidad (VPEngine no tiene vectores de objetos);
  distancia esperada = prev.w × distancia actual (la reproyección parte de NDC). Compilado por `compile_all.sh` (CI).
  Mismas constantes que PT build 50 (sin probar allí): afinar con su log.

- **Test diferencial** (`gcn.lift.differential`, `tests/diff/`): intérprete GCN de referencia independiente
  (`tests/ref/gcn_ref`, onda de 64 carriles, `-ffp-contract=off`, selftest `gcn.ref.selftest`) frente a intérprete
  VPIR (`tests/diff/vpir_eval`, IEEE exacto, min/max/clamp/saturate/fract como Metal: cualquier NaN, también
  señalizador, devuelve el otro operando; neg/abs de float por bit de signo) sobre el módulo traducido. NaN de
  resultado canónico (0x7fc00000) en ambos lados (salida float de la referencia; resultados float y `select` float
  del intérprete): la carga útil no está fijada ni entre GPUs ni entre compiladores host (orden de operandos), y
  reinterpretada como entero daba falsos fallos. `programs.txt` = 1000 programas aleatorios (`gen_programs.py 1000 99`,
  llvm-mc bonaire): **1000/1000**; semillas extra (3000 s31337, 4000 s4242): 0 fallos, con gcc y con clang.
  Depurar un programa: `VPGCN_DIFF_PRINT=i` (VPIR + exportaciones por carril),
  `VPGCN_DIFF_TRACE_LANE=l` (traza de valores a stderr).
  Arreglado con él: min/max legacy (select, NaN → b), med3 con NaN → min3, `bfe_i32` ancho (shift aritmético),
  ldexp (exponente acotado ±400, 4 pasos de ≤2^127, potencia desde bits `(e+127)<<23`), pkrtz (`halfRtz`), f32→i32
  saturado/NaN → 0, `cvt_u32` ≥ 2^32, pknorm NaN → 0, frexp con denormales, EXEC como operando `c` (decoder).
  Referencia: unorm/snorm16 con producto f32 (como la ALU). Generador: `s_mulk`/`s_movk` no tocan SCC.
  **Con control de flujo** (`gcn.lift.differential_cf`, `programs_cf.txt` = `gen_programs.py 400 7 cf`): if/else
  con el idioma saveexec de LLVM (anidados hasta 3, condición anidada en `s[36:37]`), bucles uniformes por SCC y
  bucles por carril que salen por EXEC (`s_andn2_b64 exec` + `s_cbranch_execnz`); el driver de `gcn_diff` ejecuta los
  saltos a nivel de onda. **400/400** (y 1000/1000 s7, 2000/2000 s2024; gcc y clang). Cuerpos
  divergentes solo con VALU que escribe VGPRs (compares/carries fuera: en GCN escriben 0 en carriles inactivos).
  64 bits (7 % de las instrucciones): `s_{lshl,lshr,ashr}_b64/i64`, `s_bfe_{u,i}64` (campo más allá del bit 63
  se extiende desde el 63), `s_bfm_b64`, `s_brev_b64`, `s_bitset{0,1}_b64`, `s_bitcmp{0,1}_b64`, `s_bcnt{0,1}_i32_b64`,
  `s_ff{0,1}_i32_b64`, `s_flbit_i32_{b64,i64}` sobre pares de datos y `v_{lshl,lshr}_b64`/`v_ashr_i64` (helper
  `shift64`); nuevos en el lifter salvo lshl/lshr/bcnt1. También `v_cmp_*_{u64,i64}` y `v_mad_u64_u32`. Lotes extra 4000 s556 + 1500 cf s557: 1 resto (±0 de med3
  que decide una rama: Metal tampoco fija el signo del cero en min/max).
  Cada programa exporta también `pos0` (no se compara) para ser un VS completo: CI (job visionos-device, paso «The MSL
  of every differential program…») compila con un `vpgcn_diff` del host macOS `--emit-msl` las 1400 MSL y las pasa por
  `xcrun -sdk xros metal` (`diff/compile_msl.sh`): el emisor con combinaciones aleatorias y control de flujo.

- VS: `LiftInput::instanceStepRate[2]` (VGT_INSTANCE_STEP_RATE_0/1): v1/v2 = instance id / tasa (0 o 1: sin dividir,
  con nota); v3 = instance id.
- `describe` (descriptores reales del cliente) no se pide para V# construidos por el programa (`rawBuffer`: la fuente
  es una dirección). El cliente de AstroVisionPro lo implementa leyendo T#/V# del draw (ver su bitácora).
- VPIR `image_atomic_{add,sub,min,max,and,or,xor,exchange,compare_exchange}` (storage read_write i32/u32, no cubo;
  MSL 3.1 `tex.atomic_fetch_*(coord[, capa], v).x`, CAS con bucle sobre `atomic_compare_exchange_weak` con vec4
  esperado; sin uso → sentencia). Lifter `image_atomic_*` (cmpswap: data[0] nuevo, data[1] comparando; glc devuelve
  el anterior); textura con atómicos declarada u32 (i32 si hay smin/smax) desde `resource()`. El host tendrá que crear
  la textura con `MTLTextureUsageShaderAtomic` y formato R32Uint/Sint. Goldens `cs_image_atomic`, `msl/tests/image_atomics`.
- VPIR `image_query_lod tex, smp, coord -> vec2<f32>` (fragment; MSL `calculate_clamped_lod` /
  `calculate_unclamped_lod` sin la capa de arrays; no se fusiona en optimize). Lifter `image_get_lod` (data[0]
  clamped, data[1] unclamped por dmask; fuera de píxel → 0 con nota). Goldens `ps_get_lod`, `msl/tests/query_lod`,
  `reject/query_lod_vertex`.
- VPIR `private` (`kTargetPrivate`, `private_bytes N`): memoria por invocación en cualquier etapa, sin atómicos
  (verificador; test `reject/private_atomic`); MSL `uint4 vp_private[..] = {}` en `thread`. Lifter: LDS fuera de
  compute → `private` solo con el idioma del spill (cada base DS = `v_mbcnt` de máscara completa, desplazado o con
  constante, último escritor en orden de programa; si no → error), tamaño = mayor offset DS + 16 (≤ 4096, `privateLds`),
  direcciones acotadas con `min` (sin UB), b8/b16 por RMW, atómicos DS → error. Golden `ps_lds_spill`. Stores `buffer_store_byte/short` y `ds_write_b8/b16` en
  compute: `atomic_and(~máscara)` + `atomic_or(campo)` sobre el dword (carriles vecinos no se pisan; recurso
  read_write). Golden `cs_bytes`. Stores tipados de todos los formatos salvo 11/10-bit float (`storeFormat`, inverso de
  `loadFormat`: unorm/snorm con RNE y NaN → 0, scaled saturado, f16; elementos < dword por `storeSubDword`; el formato
  se resuelve antes de declarar el recurso); golden `cs_typed_store` con bytes comprobados en `vpir_eval`. Loads de
  10_11_11 / 11_11_10 (antes leídos como 0): floats sin signo de exponente 5 (`smallFloat`, normales, denormales,
  inf/NaN); golden `cs_small_float`. Orden de campos (`packedLayout`): los nombres de AMD van del bit más alto, x en
  el bit 0 (2_10_10_10 = A2B10G10R10: x 10 bits; 10_10_10_2: x 2 bits; 10_11_11 = B10G11R11: x 11 bits). Antes
  10_10_10_2 y 2_10_10_10 estaban al revés (normales de vértice mal). Stores tipados parciales (menos componentes
  que el formato) escriben solo esos bits (`storeBits`: and+or atómicos; buffer read_write); golden
  `cs_partial_store`. unorm/sscaled de 32 bits acotados al mayor float convertible. Límites conocidos: short con `addr&3==3` (cruza dword) pierde el byte alto; dos
  carriles al mismo byte dan OR en vez de último-gana (haría falta bucle CAS). `vpir_eval` evalúa shared/private (una invocación).

Pendiente del lifter (sin uso en el corpus): FLAT, f64, atómicos de 64 bits, atómicos LDS fuera de compute,
`image_atomic` inc/dec/rsub/f*, RectList en VS. Validación en el visor: «Modo diagnóstico» de AstroVisionPro 3a1c945
(líneas `VPENGINE_DRAWS` / `VPENGINE_SHADER(S)`; el usuario aún no lo ha probado).

## Driver Vulkan (`gpu/vulkan`, lib `VPVulkan`; diseño en docs/VULKAN.md) — fases 1–2 parciales, sin build visionOS

Objetivo: que Klepton (`kl_vulkan.c`) cargue `libVPVulkan.dylib` en vez de MoltenVK (`KL_MVK_DYLIB=<ruta>`).
- Archivos: `gpu/vulkan/CMakeLists.txt` (registrado en `gpu/CMakeLists.txt`), `src/vpvk.h` (objetos),
  `src/physical_device.cpp` (Adreno 740/Quest 3: límites, features, tabla VkFormat→`vpgpu::Format`, extensiones),
  `src/entry.cpp` (98 `vk*` exportados + `vkGet*ProcAddr`, `vk_icd*`, alias KHR), `src/commands.{h,cpp}` (lista
  diferida → `vpgpu::Frame`), `include/vpvk/platform.h`, `src/platform_null.cpp` (Linux), `src/platform_metal.mm`
  (visionOS), `tests/vk_tests.cpp`. Cabeceras Khronos v1.3.296 en `gpu/vulkan/third_party/vulkan-headers` (THIRD_PARTY.md).
- Memoria: cada VkDeviceMemory = host alineado a 16 KiB; VkBuffer ligado = buffer core con `guestMemory = host+offset`.
  Imágenes = textura Metal propia (memoria solo contable). Tipo 3 lazily-allocated solo para TRANSIENT.
- vkQueueSubmit: todos los command buffers → 1 Frame core; vkCmdPipelineBarrier solo se cuenta. Render pass:
  subpass 0 → pase core; STORE DONT_CARE/NONE → DontCare (tesela); viewMask → layerCount=popcount.
- Fences: señal en submit, vkWaitForFences = waitIdle (grueso). Semáforos sin trabajo (cola única en orden).
- Verificado: `ctest -R gpu_vulkan` 5/5 (exports vía dlopen, physical, memory, passes multiview, submit) con gcc y clang
  -Werror; árbol gpu completo 105/105. Log `vk.instance`, `vk.device`, `vk.submit`, `vk.err`, `vk.unimpl`.
- Compila para visionOS (CI device en verde, ab2d7f0). `vkExportMetalObjectsEXT` ya devuelve MTLDevice/MTLTexture reales: `MetalBackend::nativeDevice()`/`nativeTexture(id)`
  (puntero `__bridge`, fuerza la asignación perezosa; no thread-safe: hilo de envío) y `platform_metal.mm` los usa.
  Test `gpu_metal.native`. Una textura exportada antes del primer submit ya no puede ir a memoria de tesela (correcto).
- NO verificado: carga real desde kl_vulkan.c.
- Pendiente:
  - Fase 2b: multi-subpass (input attachments → pixelLocal), vkCmdBeginRendering, CopyImage/Blit/Resolve/
    CopyImageToBuffer/ClearDepthStencil/ClearAttachments/UpdateBuffer, fill de palabra no-byte, secundarios.
  - Fase 3: descriptores, layouts, push constants, pipelines (interfaz pedida al lector SPIR-V en docs/VULKAN.md),
    draws/dispatch reales (hoy `draws_skipped`). Formatos ETC2/ASTC (Quest los usa: hoy sin features).
  - Fase 4: fences por envío (completion handler), eventos en GPU, queries/timestamps, vkGetFenceFdKHR.
  - Fase 5: multiview → vertex amplification en backend; FDM → rasterization rate map.
  - Superficie Unity (191 vk* importados): faltan p.ej. swapchain/surface Android (kl_vulkan.c los intercepta),
    pipeline cache, query pools, YCbCr, AHardwareBuffer.

## Renderer PS4 de VPEngine (`integrations/shadps4/gpu`, lib `vpps4`; diseño docs/ADAPTER.md)

Estado propio sin código de shadPS4, probado en Linux con `NullBackend` (tests `vpps4.*`). Cliente: AstroVisionPro
`src/platform/visionos/vpengine_renderer.{h,mm}` (modo en sombra: la imagen sigue siendo de KK).
- `state.h` (CB/DB/blend/DS/viewport/scissor/raster/primitiva/etapas), `formats.{h,cpp}` (CB/DB/IMG → `vpgpu::Format`,
  blend/compare/stencil, T#/V#/S#), `renderer.{h,cpp}`: memoria de guest en sitio por región, caché de shaders en dos
  niveles (lift → `emitMsl` → `Hooks::shader`), targets por dirección+formato+tamaño, texturas lineales (teseladas
  vacías), samplers, anillo de subida, fan/quad → lista, viewport GCN → Metal, RectList de profundidad = clear.
- MSAA: `setMaxSamples(n)`, una cuenta por draw = mín(targets, profundidad, límite). Ruta VPEngine sin MSAA (AA =
  VPKTAA): AstroVisionPro usa 1. `resolve(src, dst)` = copia a 1 muestra (formatos idénticos; si no, skip).
- Guarda `kRunawayCommands` = 50000 comandos sin submit → submit propio + `gpu.err what=frame_not_ended`.
- `MetalBackend`: cuenta de muestras no soportada → `gpu.err`, no abort.
- **Guarda de bucles (msl.cpp)**: cada `while (true)` emitido lleva `vp_guardN`; a las 8192 vueltas por invocación hace
  `break` (`kLoopGuard`). Motivo: log 11:54, frame 1 de VPEngine → `kIOGPUCommandBufferCallbackErrorHang` → reset de
  la GPU → KK `ErrorDeviceLost` → crash. Causa probable: bucle cuyo fin depende de datos mal traducidos (texturas
  teseladas vacías, constantes). Goldens MSL con bucle regenerados (`VPMSL_UPDATE=1`). Árbol gpu 122/122.
- Pruebas en el visor (ASTRO BOT VR): 11:31 frame nunca enviado (fin de frame solo en `Presenter::Present`; arreglado
  en AstroVisionPro); 11:54 `gpu.frame f=0` ya sale (34 encoders, 8 PSO), hang en f=1 (arriba).
- Errores de shader vistos: GDS `ds_append/ds_consume`, V# de `s_buffer_load` no trazable a user data; skip
  `cs: buffer outside guest memory`, draws con ES/LS (geometría), indirectos.
- Capturas de frame: nunca en repos ni CI; el usuario las pasa por chat (docs/ADAPTER.md).

## Aportes de PTVisionPro (revisión de d830b7b; nada portado aún)

**VPKTAA** = el TAA propio del visor. Fuente: PTVisionPro `patches/0001-visionos-menus.patch` → `shaders/vpktaa.frag`
(303 líneas) y `RecordVpkTaa` en `src/engine/render/upscale/scene_upscale.cpp`. Build 50 de PT: lanzada, **sin probar
en el visor** (último log en `ci-logs` = run 49). Antes de portarlo: leer el log de la build 50 de PT (coste `vpktaa`,
objetivo ≤ 0,7 ms por vista nítida; antes TAA viejo 0,8–1,0 + copia 1,5–1,8 ms) y las constantes que se afinen allí.
Sustituye a MetalFX temporal (en PT separaba la estereoscopia y costaba 3–4 fps). FXAA queda solo para comparar.

Algoritmo (una pasada por vista, en lugar del resolve; salidas MRT: color, profundidad, velocidad, historia siguiente):
- Actual: 3x3 de muestras con jitter → gaussiana (Blackman-Harris, `exp(-2.29·d²)`, d en píxeles de salida) + afilado
  tipo CAS contra la media del 3x3 (×1,2·nitidez, ×1,6 en movimiento, recortado al min/max; sin afilado en periferia).
- Todo en espacio comprimido reversible `c/(1+max(c))` + YCoCg; `Sane()` quita NaN/Inf (si no, quedan para siempre).
- Movimiento del más cercano del 3x3 (reverse-Z): objeto desde su vector o, si no hay, cámara calculada desde la
  profundidad con la view-proj anterior.
- Historia: Catmull-Rom de 5 taps; caja de varianza 1,25σ (plano) → 0,85σ (borde de profundidad), recortada al min/max,
  clip hacia el centro (no clamp).
- Peso de historia 0,93 quieto → 0,80 a 8 px/frame, tope 0,97; antiparpadeo por acuerdo de luma (−0,12);
  disoclusión: la historia guarda la distancia (ViewZ) en `.a` y se rechaza si guardaba algo más cercano
  (tolerancia max(5 %, 1,5× rango de profundidad del 3x3)); reactivo (partículas/transparentes) ×(1−0,75·r).
- Resolución interna 100/90/85/80 %: fases de jitter max(8, 8/r²), mip bias −0,5 + log2(1/r), confianza por distancia a la
  muestra más cercana. Historia RGBA16F en HDR plano (no comprimido: bandea cerca del blanco), una por vista, doble buffer.
- Ajustes de PT: `aa` (VPKTAA/FXAA), `taa_res`, `taa_sharpness` (def. 50), `taa_periphery`. Log `vpktaa: on RxR -> OxO view N`,
  `vpktaa: view N history starts again (motivo)`.

Lo que PT tiene gratis (motor nativo) y VPEngine debe resolver para un juego de PS4 cualquiera:
1. Jitter: sumar `jitter·w` al xy de la posición exportada (EXP pos0) en el VS traducido, por vista; mip bias en los
   samplers traducidos (VPEngine controla los descriptores).
2. Profundidad: identificar el depth target de la escena y si es Z invertida (varía por juego).
3. Movimiento: sin vectores de objetos genéricos → solo cámara: delta de pose de cabeza (la conoce el host) + proyección
   (tangentes de Compositor Services) + profundidad. Objetos móviles: los cubre la caja de color (algo más de estela).
4. Reactivo: gpu/core sabe qué draws mezclan alfa tras el opaco → marcar su cobertura (salida o stencil extra).
5. Punto de inserción: tras la escena y antes del post/HUD del juego (heurística: primer pase a pantalla completa que lee
   el color de escena). Alternativa simple pero peor: al final, en LDR, con el HUD dentro de la historia.
6. Lee vecinos (3x3, Catmull-Rom): no fusionable en memoria de tesela; pase propio con destino en memoria.

Lecciones del host de PT para el presentador `gpu/visionos`:
- visionOS mata la app tras ~2 s sin fotogramas: mantener fotogramas durante cargas.
- Composición en la misma `MTLCommandQueue` que el render (orden GPU sin esperas de CPU).
- Profundidad real al compositor (`[[depth(any)]]`) para la reproyección del sistema; HDR: EDR + P3, drawable `rgba16Float`.
- La app no elige los Hz: 45 fps con `cp_layer_renderer_set_minimum_frame_repeat_count`; medir y loguear los Hz reales.
- Ritmo: `query_next_frame` → `predict_timing` → esperar `optimal_input_time` → drawable + device anchor → `encode_present`.

Descartado en PT (no repetir a ciegas):
- Rate map (VRR) aplicado a todas las pasadas: sin ganancia y cuadrados al girar. En VPEngine solo tendría sentido en
  pases de geometría con el post consciente del mapa: medir antes de construir encima.
- MetalFX espacial sobre el ojo entero (6,7 ms de composición); resolución dinámica (imagen inaceptable); bajar la
  resolución base; luces por teselas y sonda TBDR sin ganancia.
- Foveado que sí funcionó en PT: quad views (inset nítido + vista ancha a baja resolución, compuestos con círculo y
  fundido 15 %). En un juego de PS4 exige repetir el command buffer del juego con otra proyección (vértices ×2): evaluar
  coste antes de elegirlo frente al rate map propio.
- Datos de P.T. en M2: centro 100 % → GPU 32–40 ms; domina el coste por píxel (sombras, luces), no los draws.

## Siguiente CPU (si se retoma)

1. **Intérprete de reserva** para entradas no traducidas (sin pausa: ejecuta al momento, compila en segundo plano y
   sustituye). Generarlo a partir de la semántica del traductor (formas verificadas por la auditoría).
2. Caché compartida de `.prx` de Sony entre juegos (misma huella).
3. Analizar el origen de las entradas perdidas (el log de bajo demanda da la cima de la pila) para mejorar el
   descubrimiento estático.
4. Pausa real de todos los hilos invitados durante una traducción bajo demanda.
5. Medir FPS/CPU frente a FEX con el mismo juego; si no mejora, mantener aparcado.

## Notas técnicas

- `ret` deja `cpu->rip` = dirección de retorno; el llamador hace `if (cpu->rip != next) return;`. `vp_run` termina en
  una dirección de salida (`vp_is_exit`).
- Cambiar `runtime/vp_cpu.h|vp_emit.h|vp_regs.h|vp_pack.h` cambia `compile_key` → el visor recompila **todo** (~25
  min). `vp_host.c` y `vp_ondemand.h` no van al SDK. Cambiar el traductor solo retraduce (objetos reutilizados si el
  C es igual).
- AstroVisionPro `visionos-vpengine` exige que la release `vpconvert-visionos` corresponda a VPEngine en
  `tools/vpaot|vpconvert|sdk`, `runtime`, `third_party`: tras tocarlos, dispatch de `vpconvert-visionos` antes de la IPA.
- Los runners x86 de GitHub son Intel o AMD al azar; flags indefinidos y algunos campos (MXCSR_MASK, XSTATE_BV[0],
  OF de `rcl` con cuenta > 1, NOP reservados) difieren entre fabricantes: los tests no los miran.
- Excepciones x87 sin máscara no se modelan; flags de excepción de MXCSR no se modelan; `rcpps/rsqrtps` exactos.
- Enlaces de descarga: releases `visionos-vpengine` (AstroVisionPro) y `vpaot-windows` (este repo).

## 2026-10-12 · vpps4: imágenes escritas por GPU vistas con otro formato
- `integrations/shadps4/gpu/src/renderer.cpp` `Impl::texture`: si en la dirección del T# hay una textura `gpuWritten` (storage) de mismo tipo/tamaño/capas y mismo tamaño de texel pero otro formato (p. ej. compute escribe UNORM y el presente lee sRGB), se devuelve esa en vez de crear otra subida desde memoria guest (vacía). Nota `GPU-written image sampled with another format of the same size (its own format)`. `Texture` guarda formato y forma.
- Nuevo `Renderer::describeAt(address, bytes)`: lista targets color/profundidad que solapan el rango (offset, formato, tamaño, frames sin uso) y texturas en esa dirección (storage / gpu_written). Test `vpps4.storage_alias` (pasa, 16/16 vpps4).
- Pendiente: vistas de textura en vpgpu (otro formato / capa / subrango dentro de un target mayor); write-back de targets a memoria guest.

## 2026-10-12 · vpps4: rendimiento (DMA sin espera, hash de texturas) y volcado de shaders
- Medido en log 18:39 (mundo, ~6 fps): ~15 submits por imagen, ~1600 draws/imagen, ~3,6 subidas al anillo por draw; cada DMA hacía submit + `waitIdle`.
- `renderer.cpp` `Impl::guestBuffer` registra el rango guest tocado (`touch`, mapa de intervalos fusionados `busy`, >8192 rangos → `busyAll`). API nueva `Renderer::gpuMayAccess(addr,size)` / `Renderer::gpuIdle()` (`renderer.h`): el adaptador solo sincroniza un DMA si el rango solapa trabajo GPU pendiente. Índices indirectos: rango = `count * tamaño índice` (antes 1 byte).
- `Impl::check`: hash completo de fuentes de textura limitado a 4 MiB por frame (`kFullHashBytes`); el resto espera a frames siguientes.
- `Impl::dumpProgram`: al fallar el lifter (primeros 8 programas) se vuelcan las palabras del código y del fetch (`vpps4.shader dump key=...`, 128 palabras/línea) para convertirlas en tests `.lift`.
- Test `vpps4.busy_ranges` (17/17 vpps4 pasan).
- Pendiente: 2 VS con `cannot trace the V# descriptor of s_buffer_load_dword(x16)` (24172 draws excluidos) → con el volcado del próximo log, crear `.lift` y ampliar `traceOrigins`; GS/ES/LS (VS-como-ES → anillo, GS como compute) sin implementar; tess sin implementar.

## 2026-10-12 · lift: descriptores tras ramas; slices de targets array
- `gpu/shader/gcn/src/lift.cpp` `traceOrigins`: el recorrido lineal en orden de programa pasa a flujo de datos por bloques básicos (estado de entrada = intersección de los predecesores; fetch inline en `s_swappc_b64`; cada visita recalcula `rsrcSource/sampSource/pointerSource/rawBuffer`). Antes, un `s_buffer_load_dwordx16 s[0:15]` del camino que sigue de largo borraba los SGPR de usuario s4..s7 para el bloque destino del salto.
- Tests nuevos `gcn.lift.vs_cbuf_x16` (0x5c4d08a17d1d9a24), `gcn.lift.vs_cbuf_dword` (0x69d775ec21442a0c), `gcn.lift.cs_resinfo` (0xb82099bcccf5d4fd), del volcado del log 19:05: los tres traducen.
- `integrations/shadps4/gpu/src/renderer.cpp` `Impl::target`: se reutiliza el target de la dirección con capas >= las pedidas (arrays mayores delante): dibujar a slice 1 o a slices 0-1 cae en el mismo objeto. `Impl::targetSlice` + API `Renderer::colorTargetSlice(addr, w, h, layer)`: T# en `base + k * bytes_slice` de un target array → ese target y la capa k (solo para el presente; los draws no tienen vistas de capa).
- Evidencia ojos (log 19:05): T# ojo 0 0x70723e0000, ojo 1 0x7072ce0000 (diferencia 0x900000 = 1536*1536*4, 2 slices de un array); `describeAt` sin ningún target de color en 256 MiB. Los draws con VS sin traducir sí registran su target antes de saltarse ⇒ el productor son los draws ES/LS/GS que el adaptador excluye antes de llegar al renderer.
- Pendiente: ruta GS (ES→anillo ESGS, GS con amplificación estéreo / `render_target_array_index`, copy-VS) con el volcado `VPENGINE_GEOM` del próximo log; `RenderPassDesc.layerCount = sliceCount` en draws en capas; vistas de capa para T# de draws.

## 2026-10-12 — Anisotrópico x16
- `gpu/vulkan/src/config.cpp`: `vpvk_config_set_anisotropy(int)` (defecto 16, log `vk.cfg anisotropy=N`), `vpvk::anisotropyForced()`; `config.h` declarado; `vpvk.h` prototipo.
- `entry.cpp` `vkCreateSampler`: fuerza maxAnisotropy en linear/linear con mips (`maxLod > minLod+0.25`), sin compare ni unnormalized. Features/limits ya daban samplerAnisotropy/16.
- VPKTAA: implementado en el compositor de Klepton (VisionQuestPro), `vk.cfg vpktaa ... applied=compositor(klepton)`.
