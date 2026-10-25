# Bitácora — VPKEngine

## 1. Estado actual (2026-10-20)

Motor de traducción nativo para Apple Vision Pro sin JIT. GPU → Metal (`gpu/`: core, metal2, shader gcn/spirv/ir/msl, `gpu/vulkan` = VPVulkan, `gpu/gles` = vpgles, `gpu/ps4` = `vpps4`, foveado propio, VPKTAA nativo) y CPU x86-64 → C AOT (`tools/vpaot`, `tools/vpconvert`, `runtime/`, firma/carga en `runtime/vp_codesign.c`, `vp_pkcs12.c`, `platform/visionos/Sources/VPGamePack.swift`, `VPConversion.swift`). Todo código propio: sin MoltenVK, SPIRV-Cross ni Mesa.
Clientes: VisionQuestPro (foco actual; usa VPVulkan + vpgles), AVP, AstroVisionPro (su adaptador y su plan de Bloodborne viven en ese repo). Docs: `docs/PLAN.md`, `docs/GPU.md`, `docs/VULKAN.md`.

**Verde:**
- `ctest` Linux 235/235 (gles, glsl, spirv, vulkan, metal2 con backend nulo, vpps4). TSan limpio en `gpu_core` y `gpu_vulkan` (build aparte con `-fsanitize=thread`).
- Última build en verde de VisionQuestPro con este motor: run 39402167780 (Gate completo). Último commit: `44c9785` (presupuesto de encoders). Los `.mm`/`.metal` solo los compila el CI de Mac.

**NO probado en el visor** (compila y pasa tests, nunca ejecutado en hardware): todo lo de `notas/lotes.md` posterior a `e6bb667`, es decir: puerta blanda solo para imágenes de ojo marcadas (`vpvk_mark_eye_image`), presupuesto de encoders con contrapresión (`cut_enc`, `chunk_wait`), hazard por rect, `gpu.bigframe`, `drainRetired`/trim sin tirar funciones en uso, contadores de diagnóstico, pipelines asíncronos con perfil completo, puerta blanda/dura, prewarm en segundo plano, caché estable (versiones por componente), fusión del pase vacío de resolve, hang split con MSAA, `stall_warn`/`cbtime`, trim de memoria, liberación MSAA a 120 frames, MSAA memoryless, cortes por coste, máscara UI retenida, RCAS independiente, foveado propio (interruptor apagado), VPKTAA nativo, recetas con MSL bajo demanda, caché `'F'`, vpgles ES 3.1 (storage/imagen, SSBO/atomics en vertex, FBO sin attachments, STORAGE perezoso), `[[base_vertex]]`/`buffer(29)`, perfil Adreno 1.3 con `VPVK_EXTRA_EXTENSIONS`, CTS completo. Traducción de CPU bajo demanda sin build ni prueba en visor.
**2026-10-25 (sin probar en visor):** `memValid` por textura en `ResourceTable` (`resources.h`); `placeTiles` reescribe Load→DontCare si la memoria está indefinida y el veredicto queda TileLocal (`loads_dropped=` en `gpu.frame`); `commitMemory` tras `Device::submit` (`device.cpp`); imágenes con `usage::storage` siguen persistentes (`entry.cpp` ~1340); pool memoryless de metal2 con `lastFrame` y desalojo (>16 claves, >600 frames, `retire`), `ml_pool=` en el ledger. Objetivo: MSAA del ojo de Deadpool (`why=LoadsContents`, ~284 MB) en tile. ctest 235/235.
**2026-10-25 (sin probar en visor):** portados de AVP `vpkengine-patches/` 0025, 0033, 0034 (`msaaResolveOnTile` APAGADO por defecto; `VP_MSAA_TILE=1` opt-in, inseguro hasta `Attachment::keepSamples`), 0035, 0044 (`Copy::mipChain`), 0030 (solo core: `Device::resMutex_`, `textureDesc()`), 0050 (ban memoryless por encoder culpable `memoryless_attrib`, HeavyGeometry solo si quedaría TileLocal, `VPVK_TILE_MAX_DRAWS/VERTICES`, pin `nativeTexture` con generation). 0047-0049 ya estaban. Sin portar: 0001-0024, 0026-0032 (salvo 0030 core), 0036-0043, 0045-0046 (main divergió ~42k líneas de la base AVP `19a1f44`; fusión completa pendiente, por capas). ctest 236/236.
Desconocido: si desaparece el rosa de Unity (`[magenta]`) con `VPVK_REPORT_GEOM=1` + interpoladores 128 (AC Nexus, AoT); si cesa el cuelgue de ShaderWarmup de AC.

## Últimos hechos
- 2026-10-25: bitácora compactada; detalle movido a `notas/` (arquitectura, lotes, hallazgos, build).
- 2026-10-25: `.claude/agents/coder.md`: maxTurns → 70.
- 2026-10-25: `Claude.md` línea 1: regla de cumplir siempre el rol asignado.

## Pendientes y riesgos abiertos (priorizados)
1. **Probar en visor** todo lo de §1 «NO probado». Leer: `[mtl] cut_enc`/`chunk_wait` y que desaparece el salto de IOAccelerator en AC Nexus; `gpu.bigframe` (qué labels/texturas generan barreras) para decidir `VPVK_COPY_RECT_HAZARD=1`; `gpu.hang split=on`, `gpu.cbtime`, `gpu.stall_warn`; `gpu.enc` (`ls=`, `tile=c0`, `passes=2`) en Batman para confirmar la fusión (si persiste `reason=Feedback`, el destino de resolve sí es muestreado por el encoder anterior: mirar ese binding); `vk.boot_summary`/`vk.pipeline_wait` con perfil completo; `vk.rcas active … ui_mask=yes`; `[mem] trim … VPKEngine released`; `tex_msaa`/`msaa_memoryless`; `gpu.slowcb est_cost`; `gpu.shader.msl cached=`; `[magenta]` (si no cesa: A/B con `VPVK_EXTRA_EXTENSIONS=1`, comparar `vk.caps` con el Adreno real).
2. **Contrapresión de chunks**: bloquea el hilo de submit en frames de muchos chunks (hang split y presupuesto de draws incluidos); con GPU saturada los chunks de 256 con MSAA pueden seguir pasando de 2 s (medir `gpu.cbtime`). `VPVK_MAX_PENDING_CHUNKS=0` la quita.
3. **Activar el hazard por rect** tras leer `gpu.bigframe` en visor (hoy apagado por falta de prueba de seguridad).
4. **Primer arranque sin caché**: draws sin PSO se saltan unos frames (puerta blanda) y los RTT esperan hasta 1,5 s por frame (duros); un hilo de grabación que pide un pipeline en cola espera su traducción (hasta 2 s). Si el ojo aún no se detecta (`gTaa`/`gMsaa` vacíos), un pase persistente de tamaño de ojo cuenta como duro. La puerta puede mantener `submitMutex` hasta 1,5 s. `eraseDraws` deja pases sin draws (load/clear se mantienen). Los RT de ShaderWarmup (persistentes, pequeños) son duros: sin ganancia. `RecipeStore::refresh` no es cancelable (el `join` de `vkDestroyDevice` espera). En síncrono (workers=0) no hay marca de PSO huérfano.
5. **Profundidad MSAA muestreada**: provoca corte `Feedback` y mantiene `tex_msaa` (847 MB en Batman); sin solución.
6. **int64 en VPIR**: no existen tipos de 64 bits en `ir.h`; bloquea atómicos 64, `shaderInt64`, f64.
7. **`samplerBuffer`/`imageBuffer`** en GLSL (activaría texel buffers y `EXT_texture_buffer`; `glTexBuffer` es latente). TexBuffer de tamaño fijo (`glTexBuffer`/`Range`, `VkBufferView`): revisar si el buffer se redimensiona; RGB32 rechazado.
8. **Pipelines con vs y fs de programas distintos** (`UseProgramStages`): requiere enlazado single-stage en `vpglsl::link` y propagar uniforms; hoy `INVALID_OPERATION`.
9. **`keep()` de attachments en `beginPass` con share group** (`gles/draw.cpp`): si otro contexto recrea la imagen con un pase abierto no se retiene (riesgo bajo).
10. **SampleMask no aplicada** (`pSampleMask` y `glSampleMaski` solo guardan/avisan); `sampleread`, `multiview.input_attachments`, `remaining_array_layers.single_layer_fb` sin arreglar.
11. **`waitReady` de `Missing`**: devuelve «lista» para claves inexistentes; el llamante no distingue «destruida» de «compilada».
12. `storePacked` con bool y Component en interfaces empaquetadas (`lower.cpp`) sin cubrir.
13. `UiMaskState::last` al apagar TAA en caliente: comprobar que las máscaras publicadas se liberan al desactivar el consumidor.
14. `MAX_COMBINED_IMAGE_UNIFORMS` anunciado fijo en 24 (`gl_api.cpp:140`) con `kMaxImageUnits` = 8 (`context.h`) y vertex = 0: ajustar al límite por etapa.
15. Foveado: validar en visor que `[[position]]` del fragmento es físico bajo el mapa y que la capa por `viewport_array_index` se cumple; medir plegado/desplegado; M2/M5 reales (PTVisionPro); tangentes de ojo desde XR (hoy centro geométrico); formatos de ojo no-sRGB8 sin soporte en VPKTAA nativo (`writeFormat`).
16. MSAA del juego sin máscara de UI en algunos casos (se afila todo el ojo); lineal escrita por el host y muestreada/renderizada no se sincroniza; flip en z de blits 3D ignorado; vistas texel-compatible comprimidas no soportadas; `queryLod` y multiview+occlusion sin validar; load/clear multiview con máscara irregular tocan capas fuera de máscara.
17. CTS en el visor por grupos. Fallos conocidos sin causa: `dynamic_state.monolithic.*`, `memory.pipeline_barrier.all*`, texture.mipmap projected (precisión LOD), precision.mix highp (fast-math), border_swizzle gather stencil, `legal_mismatch_ivec4_ivec2`, B4G4R4A4 (Metal solo ABGR4). Salida de fragmento int/snorm y blits 1D sin auditar; ycbcr 10/16 bit.
18. GLES/GLSL ausente: transform feedback real (stubs), geometría/tess, program binary, `EXT_disjoint_timer_query`, oclusión GL real, `sampler2DMSArray`, io blocks, `interpolateAt*`, `gl_SampleID/SampleMask`, `gl_ClipDistance`, `imageAtomicExchange` float, arrays de bloques con índice.
19. Avisos de compilación sin tocar: `builtins.cpp`, `parser.cpp`, `link.cpp`, `device.cpp`, `textures.cpp`, `draw.cpp`.
20. No viable en Metal (no declarar): variablePointers completo, LogicOp/PatchControlPoints dinámicos, availability/visibility, BDA capture/replay, inline uniform update-after-bind.
21. Lifter GCN sin uso en corpus: FLAT, f64, atómicos 64 bits, atómicos LDS fuera de compute, `image_atomic` inc/dec/rsub/f*, RectList en VS. Limitaciones: `short` con `addr&3==3` pierde el byte alto; dos carriles al mismo byte dan OR, no último-gana; no indexado + indirecto GPU usa `[[base_vertex]]`. CPU: intérprete de reserva para entradas no traducidas, pausa real de hilos durante traducción bajo demanda.

## Decisiones de diseño vigentes
- **Sin MoltenVK/SPIRV-Cross/Mesa**: traductor propio Vulkan/SPIR-V → VPIR → MSL.
- **El hang split solo se veta con dump o attachments memoryless** (no se pueden `Store`): cada corte guarda/recarga todos los attachments y partir cuesta mucho menos que un solo CB de varios segundos (AC Nexus 4x MSAA 3840×2485×2 capas: 15 cortes de 94–2492 ms antes del cambio). Cortes entre encoders (por coste/nº) son siempre seguros.
- **Chunk de 256 medido, sin escalar por coste**: en AC son 13 chunks de 256 (0,6 s); chunks menores serían ~52 cortes Store/Load de ~1 GB (hipótesis sin medir).
- **La fusión del pase vacío solo compara `resolveWrites`**: las escrituras de attachment propias del encoder no son feedback; `PinnedCopy` se mantiene completo (una copia hundida que lee el MSAA no puede ver el store adelantado).
- **Presupuesto de encoders y contrapresión**: cada encoder fija una región IOAccelerator de 64–128 KiB no contabilizada hasta que su CB completa; 512 encoders = 32–64 MiB por CB, un frame normal no se corta. El CB final del frame no se espera (no rompe el pipelining de fin de frame).
- **Fences no reutilizadas dentro del frame**: reutilizar una fence con waits pendientes de CBs previos no está demostrado seguro.
- **`VPVK_COPY_RECT_HAZARD` apagado**: texturas y buffers son `HazardTrackingModeUntracked` y un blit encoder no tiene barrera interna; con rects disjuntos la memoria es disjunta pero no está demostrado (compresión sin pérdida / RMW de tiles parciales) que sea seguro. Activar solo tras probar una escena que suba atlas por teselas.
- **`trimMemory` no tira MTLFunction/Library en uso por un pipeline vivo**; los PSO no se liberan (el caché los da por Ready).
- **Pipelines async con puerta por frame**: un draw sin PSO se salta y deja agujeros permanentes en lo que se dibuja una sola vez (RTT, sombras); la puerta espera solo ahí y salta sin esperar en warmup/ojo. Nunca se dibuja con otro shader. Async no depende de `VPVK_EXTRA_EXTENSIONS` (con «Perfil completo» daba 40 s bloqueado y congelaciones de 10,5 s).
- **La versión del traductor sube SOLO si cambia la salida** (digest del corpus); antes era el hash CMake de las fuentes y cualquier refactor invalidaba todo. `kRecipeVersion` solo ante cambio de codificación.
- **Recetas sin MSL en RAM**: se lee de `recipes.bin` bajo demanda.
- **`vertexPipelineStoresAndAtomics` FALSE** fuera de `VPVK_EXTRA_EXTENSIONS`: con stores en vertex colgaba el warmup de AC; por eso vpgles anuncia `MAX_VERTEX_SHADER_STORAGE_BLOCKS` y `MAX_VERTEX_IMAGE_UNIFORMS` = 0.
- **ES 3.1 y no 3.2**: ningún juego objetivo usa geometry/tess; sus consultas devuelven 0. Un pipeline con etapa de geometría real falla (`vk.err what=stage`).
- **STORAGE perezoso**: `STORAGE_BIT` en GPUs Apple desactiva la compresión sin pérdidas.
- **MSAA de color en tile (memoryless) y liberación a 120 frames**: un MSAA persistente costaba 605–670 MB residentes.
- **Robustez de buffers** (`robustBuffers`, se anuncia `robustBufferAccess`): descriptor nulo/OOB en warmup colgaba la GPU Apple. Coste: una comparación por acceso a SSBO.
- **Guardas de bucle** en todo `while(true)` emitido: un hang de GPU resetea la GPU y mata el proceso.
- **FDM del juego apagado, foveado propio**. Regla anti ojo deformado: Metal elige la capa del rate map por `[[viewport_array_index]]`, NO por `[[render_target_array_index]]`; en todo pase estéreo con rate map cada vista escribe `viewport_array_index = vista` y el encoder fija un viewport+scissor por vista. Comprobar con capturas de ambos ojos.
- **RCAS independiente y siempre activo, sin upscale**: el desplegado lo hace el rate map y la resolución la escala el compositor.
- **`VPVK_REPORT_GEOM=1` solo en juegos** (Unity descarta subshaders sin geometry/tess ni 128 interpoladores y pinta rosa 255,0,255); CTS en 124.
- **Cola de 256 CB y `gpu_ms` = suma de CB**: el modelo de coste es una cota superior sin calibrar (meta: ningún CB > ~100 ms; ajustar pesos con `est_cost` de `gpu.slowcb`).
- **Memoria nueva a cero** (`VPVK_ZERO_NEW_IMAGES`): semántica Android.
- **Dispositivo perdido = cierre limpio**: Metal no se recupera en proceso; el host cierra el juego con `vpvk_gpu_lost`.
- **Capturas de frames y volcados de shaders de juegos: nunca al repo ni CI**.

## Ficheros en juego
- `gpu/metal2/src/metal_backend.mm` (MSAA memoryless, pool, ledger), `gpu/core/src/frame_compiler.cpp` (`placeTiles`), `gpu/core/src/device.cpp`, `gpu/vulkan/src/entry.cpp`, `gpu/vulkan/src/physical_device.cpp`.

## Índice de notas
- `notas/arquitectura.md`: módulos, ficheros, contratos, trampas, variables `VPVK_*`, API exportada.
- `notas/lotes.md`: qué hizo cada lote reciente por módulo y la sesión de drenaje 2026-10-21.
- `notas/hallazgos.md`: mediciones de visor (Demeo, AC Nexus, Batman, Deadpool, RE4).
- `notas/build.md`: compilar, ctest, TSan, CI de VisionQuestPro.
