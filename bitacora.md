# Bitácora — VPKEngine

Motor propio para Apple Vision Pro: Vulkan/SPIR-V → VPIR → MSL → Metal (`gpu/`: core, metal2, shader, `gpu/vulkan` = VPVulkan,
`gpu/gles` = vpgles) y CPU x86-64 → C AOT (`tools/vpaot`, `runtime/`). Sin MoltenVK, SPIRV-Cross ni Mesa.
Clientes: VisionQuestPro (foco), AVP, AstroVisionPro. Licencia PolyForm Noncommercial 1.0.0.

## Last done
- 2026-11-10 Diagnóstico `vk.sub_dump` / `vk.sub_draw` (pipelines.cpp `vkCmdNextSubpass` y final de `makeDraw`; `Recorder::bound.subDump` en commands.h, puesto a 0 en el begin de entry.cpp): los 2 primeros subpases posteriores de un pase con framebuffer >= 1024 de ancho y luego 1 de cada 3000 escriben hasta 64 draws con `describePipelineState` + count/inst/indexed/vbs/stencilRef/rawDepth/cullMode. Motivo: en ITR2 falta la niebla; UE móvil 5.5 la dibuja en el subpase 1 (`r.Mobile.PixelFogQuality`, `r.Mobile.DisableVertexFog` en el binario) leyendo la depth 4x como input attachment (bajada: `fetch()` de lower.cpp, capa = vista, muestra del operando Sample). Con el volcado se sabrá si el draw de niebla llega. ctest 340/340.
- 2026-11-10 Tirones de Gundam (logs 27, DEMOSTRADO en log: en 77 s 288 `msaa_materialized` y 168 `msaa_released`, la misma textura hasta 16 veces; cada vuelta de un respaldo de 46-100 MiB = `vk.hitch backend_ms` 45-58 ms con `pso_create_ms`≈0 y `compiled_in_session=0`): `releaseMsaaBackings` (metal_backend.mm) exige `msaaReleaseFrames << min(releases, 6)` frames en tile; `Texture::releases` sube en cada liberación que no venga de aviso de memoria. Solo compilable en CI de macOS; sin probar en visor. Otros tirones medidos y sin tocar: frame con 612 copias a texturas 128x1 RGBA16F (1794 encoders, 55-82 ms, 4 veces en la escena de warmup) y `guest_gap` de 38-50 ms con subidas de 85 MiB (4096² RGBA8 con mips) repetidas.
- 2026-11-10 Lote tras logs 27 de VQP (ctest 340/340 Linux; metal_backend.mm sin tocar; sin probar en visor). (1) HLVR, DEMOSTRADO (log + binarios): el shader de preprocesado de profundidad de Meta (vertex `SV_RenderTargetArrayIndex`, `DrawProceduralNow(…, 3, 2)`) fallaba `SetPass` (`E/Unity: DrawProceduralNow requires a successful material.SetPass`) porque no anunciábamos `VK_EXT_shader_viewport_index_layer`: libunity (2021.3 y 2022.3) referencia ese nombre y `vulkan.adreno.so` del firmware de Quest 3 lo lista. Añadida a `deviceExtensions()` (physical_device.cpp); el traductor ya bajaba `BuiltIn::Layer` de vertex a `[[render_target_array_index]]` (test nuevo `kLayerFromInstanceVert` en spirv_tests.cpp); `gl_ViewportIndex` se descarta (1 viewport). (2) ITR2 (`vk.rp_dump` de logs 27: att0 color 4x y att2 D24S8 4x TRANSIENT con CLEAR/DONT_CARE, área 3552x2258 sobre imagen 3552x2260, sub1 lee att2 como input read-only sin barrier): `areaClears` trata el primer uso de un adjunto TRANSIENT como clear completo (antes Load + quads = `msaa_stale=2` por frame) y `beginPass(keepForSplit)` solo fuerza Store en un subpass que se lee a sí mismo si hay un `OpPassBarrier` por delante (`barrierAhead`); el tramo tras el último barrier recupera los stores del cliente (`selfFinal`). Queda el corte i4→i5 por Feedback (store+load 4x de color y depth una vez por frame). (3) `vk.rp_dump`: la clave incluye tamaño del framebuffer y si el pase tiene >= 16 draws; 6 formas.
- 2026-11-10 ITR2 (logs 26): `uiPass` no añade la máscara UI c7 a un pase MSAA del juego que continúa el anterior (`why=msaa_continuation`); diagnóstico `vk.rp_dump` (commands.cpp `rpDump`). La violación de transient es pegajosa (`transient = d->transient && !transientViolated`, frame_compiler.cpp placeTiles). PENDIENTE: depth como input attachment en tile (adjunto de color sombra con la z) para fusionar i4+i5; resolve de depth en tile para i6.
- 2026-11-09 Diagnósticos ITR2 (sin cambio de render; ctest 340/340 Linux; sin probar en visor): `vk.device_req exts=… names=…` y `vk.device_req features struct=… on=… unsupported=…` por petición distinta de vkCreateDevice (entry.cpp `deviceRequestLines`; `unsupported` = pedidas que el driver no reporta; `other_sType_not_decoded`); `vk.tex_hist` cada 5 s (solo si cambia, ≤240 líneas): imágenes SAMPLED por nº de mips escritos (m0, m1_2, m3_5, m6p, ASTC aparte, `partial`, `top0_unwritten`, `rt`) con `Image::mipsWritten` atómico marcado en copy/blit/copyMemoryToImage y registro global de imágenes (vkCreateImage/vkDestroyImage). Menores abiertos: recorrido del registro cada 5 s con mutex aunque lines>=240; fetch_or por región en cada copia; planos extra no contados.
- 2026-11-09 MSAA forzado ya no rechaza por `load` los pases con depth/stencil en Load si la imagen 1x es `sampled`+`formatView` (Maestro: `skip{load}` con `passes=0 up=0`, pase principal 3168x2032 eye=1): `imageDesc` (gpu/vulkan/src/entry.cpp ~205-229) da `usage::formatView` a adjuntos 1x depth+stencil no transient bajo forzado; `forceMsaa` (commands.cpp ~1353) rechaza solo si falta; el twin queda en Load y se siembra desde la imagen del juego (stencil por 8 pases, metal_backend.mm ~5929-5947); `rej_first reason=load` (vpvk.h rejFirstDone[3]). ctest 339/339 Linux; sin probar en visor. Riesgos unverified: PixelFormatView puede quitar compresión sin pérdidas; activa caminos de resolve/seed de stencil antes solo de gemelos propios (`gameAuto` commands.cpp ~1229, `splitHeavyAuto` frame_compiler.cpp ~735). Medir: `passes>0`, `seed_draws`, ausencia de `what=seed_source`. Alternativa más simple sin probar: eximir X32_Stencil8 en `sourceView` (metal_backend.mm:5947).
- 2026-11-09 MSAA forzado también en el pase de ESCENA 3D principal con target intermedio (Attack on Titan: antes 10250 pases rechazados por tamaño≠ojo y el AA no se aplicaba): `forceMsaa` (gpu/vulkan/src/commands.cpp ~1279-1300, lambda `sceneMain`: color0+depth Tex2D, mips==1, layers≤2, píxeles×2 ≥ ojo, ≥32 draws `kSceneMinDraws`); se mantienen todos los rechazos (smp/msstore/res/shape/input/fmt/load/fmt_size/area/unreadable/smask/scan_*), muestras siempre memoryless con resolve a la 1x; nuevo motivo `scene` (`rej scene WxHxL`), log `vk.msaa force= … scene{up=,rej=}`, contadores `sceneUp`/`rejScene` (vpvk.h). Test gpu_vulkan.msaa_force bloque 5b; ctest gpu_vulkan|gpu_core 109/109 (backend nulo); SIN probar en visor; heurística (32 draws / 50 % ojo) puede subir reflejos/probes grandes con color+depth (menor); sin medir memoria/tile en Apple.
## Current state
- Verde: ctest 340/340 (Linux, backend nulo); CTS rápida según arriba. Todo lo de 2026-10-27/28 sin probar en visor.
- Pendiente:
  1. Probar en visor (VQP): MSAA memoryless sin `memorylessFailure`/`msaa_materialized`, stencil tras seed, tirones por seed PSO.
  3. `bias(vp_args.sbN)` en todo muestreo implícito aunque bias = 0 (falta especialización por estado de sampler).
  4. Harvest continuo casi nunca cede con pipelines async (espera ~3 ms); vigilar `draws_skipped` y tirones.
  5. Intermitente CTS `multisample.misc.clear_attachments...ds_resolve_sample_zero.sub_framebuffer` sin causa (sync2: corregido `advance`, unverified sin CTS).
  6. Sin int64 en VPIR; `shaderInt64=0` anunciado. Quest 3 real: unverified (perfil marca "1?"); aparcado hasta confirmar.
  7. `textureCompressionBC` con identidad Quest = `platformSupportsBC()` (código/tests) vs `perfil-adreno740.md:59` = 0: discrepancia unverified (no hay vulkaninfo real de Quest 3).
  8. El backend no elimina el resolve de adjuntos `Resolve` al invalidar (solo `Store`).
- Riesgos: seed restaura depth de la muestra 0 en todas las muestras (halos en siluetas); un draw que supere solo los límites de tile no se puede partir.

## Decisions
- MSAA de juego siempre memoryless (resolve en tile + seed desde 1x); nunca materializar muestras.
- Perfil anunciado = Quest 3 real (gpuinfo) validado con CTS/dEQP; lo de nicho o a ciegas se aparca, no se anuncia falso.
- GLSL estricto solo en CI; juegos permisivos con log `gles.glsl.lenient`.
- La versión del traductor sube solo si cambia la salida (digest del corpus).
- Pipelines async con puerta: un draw sin PSO se salta, nunca se dibuja con otro shader.
- Robustez de buffers y guardas de bucle en todo `while(true)` (un hang de GPU mata el proceso).
- Memoria nueva a cero; dispositivo perdido = cierre limpio.
- ES 3.1 (sin geometry/tess). STORAGE perezoso (desactiva compresión en Apple).
- Hang split solo vetado con dump o attachments memoryless; cortes entre encoders siempre seguros.
- `VPVK_COPY_RECT_HAZARD` apagado (no demostrado seguro).
- Capturas y volcados de shaders de juegos nunca al repo ni CI.
- Ficheros propios nuevos con `SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0`; no quitar `Required Notice` de `LICENSE`.

## Files in play
- `gpu/core/src/frame_compiler.cpp`, `gpu/metal2/src/metal_backend.mm`, `gpu/vulkan/src/{commands,pipelines,entry,physical_device,ext}.cpp`.
- `gpu/gles/src/{draw,egl,device,textures,formats,gl_api}.cpp`, `gpu/shader/src/{lower,msl}.cpp`.
- `.github/workflows/{cts,gles-cts}.yml`.

## Module index
- `notas/arquitectura.md`: módulos, ficheros, contratos, trampas, variables `VPVK_*`, API exportada.
- `notas/cts.md`: cómo lanzar CTS/dEQP, estado, fallos aparcados, hallazgos de GPU Apple.
- `notas/lotes.md`: lotes anteriores por módulo (histórico).
- `notas/hallazgos.md`: mediciones de visor (Demeo, AC Nexus, Batman, Deadpool, RE4).
- `notas/build.md`: compilar, ctest, TSan, CI de VisionQuestPro.
