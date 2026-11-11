# Bitácora — VPKEngine

Motor propio para Apple Vision Pro: Vulkan/SPIR-V → VPIR → MSL → Metal (`gpu/`: core, metal2, shader, `gpu/vulkan` = VPVulkan,
`gpu/gles` = vpgles) y CPU x86-64 → C AOT (`tools/vpaot`, `runtime/`). Sin MoltenVK, SPIRV-Cross ni Mesa.
Clientes: VisionQuestPro (foco), AVP, AstroVisionPro. Licencia PolyForm Noncommercial 1.0.0.

## Last done
- 2026-11-11 Auditoría de gestión de recursos (memoria, GPU, CPU, concurrencia), **sin cambios de código y detenida a medias**. Hecho: pasada con ASan/UBSan/LSan/TSan + clang-tidy (7 defectos reales, T1-T7; 0 errores de memoria y 0 crecimiento de heap en los tests) y auditoría manual de `gpu/core` + `gpu/metal2` + `gpu/metal` (C1-C18), `gpu/vulkan` (V1-V21) y `gpu/gles` + `gpu/ps4` (G1-G18, P1-P6). Todo con fichero:línea, escenario, medición y arreglo propuesto en `notas/auditoria.md`; repros y benchmarks en la rama `auditoria-repros`. Sin auditar: `gpu/shader`, `runtime`, `tools`, `platform/visionos`, `apps`.
- 2026-11-11 VPWin (aplicaciones Windows / Steam), arranque (tests AOT 106/106 en Linux x86-64). (1) `vpaot`: leer un registro de segmento se traduce a su selector (`Image::windows`, que pone `load_pe`; PE → cs 0x33, ss/ds/es/gs 0x2b, fs 0x53; el resto → cs 0x33, ss 0x2b, los demás 0; selectores de PS4/FreeBSD sin modelar); escribir uno sigue sin soporte. `tools/vpaot/{image.h,image.cpp,translate.cpp}`, test `tests/aot/cases/mov_seg.s` + golden. (2) Medido: `vpaot` traduce el 99,989 % de los 768 PE x86-64 de Wine 11.0 (14,7 M de instrucciones); lo que falta son trampas e instrucciones privilegiadas. (3) Diseño, costura PE ↔ Unix de Wine y hitos en `notas/win.md`. Del backend (`signal_vpk.c`) no hay nada escrito; línea de trabajo aparcada.
- 2026-11-10 Lote tras logs 28 de VQP (ctest 342/342 Linux; metal2 sin tocar; sin probar en visor). (1) Presupuesto de texturas `vpvk_texture_budget(max_dim)` (config.h; entry.cpp `textureBudgetDrop`, `budgetRange`, `budgetShrink`; `Image::mipDrop` en vpvk.h): una textura muestreada simple (2D, 1x, optimal, usage ⊆ SAMPLED|TRANSFER_SRC|TRANSFER_DST, sin planos ni sparse) creada por encima del tope pierde sus niveles mayores (`desc` reducido, `info` intacto, requisitos de memoria menores). Comprimidas por bloques: solo si tienen cadena (queda >= 1 nivel). Sin comprimir de 8 bits (R8/RG8/RGBA8/BGRA8, unorm/sRGB): también con un solo nivel; la subida al nivel 0 se reduce con filtro de caja en el submit (`shrunkCopy`, commands.cpp; `OpCopyBufferToImage::shrink`). Reescritura al grabar: `vkCmdCopyBufferToImage` (entry.cpp), `vkCmdCopyImageToBuffer`, `vkCmdCopyImage` (copia a nivel descartado sin comprimir ⇒ blit escalado), `vkCmdBlitImage` (`toTop`; el paso de GenerateMips dentro de la misma imagen se omite), `vkCmdResolveImage`, `vkCmdClearColorImage`; vistas en `createViewOne`. El host solo debe apretar el tope (copias nivel a nivel entre texturas coherentes). Log `vk.tex_budget`. Test `tex_budget`. (2) `hostAlloc`/`hostFree` (entry.cpp): la memoria host de un VkDeviceMemory >= 1 MiB es un mmap anónimo propio alineado a 16K (VM tag 241 en Apple) en vez de `aligned_alloc`. (3) Diagnóstico (sustituye `vk.sub_dump`/`vk.sub_draw`): `vk.draw_census` (una línea por pipeline y subpase la primera vez que dibuja en un pase con fb >= 1024, tope 600: estado + count/inst/indexed/indirect/vbs/tb/in_att/in_ms/fb/samples), `vk.comp_census` (por pipeline de cómputo: grupos, buffers `w`/`i`, imágenes, `indirect_out`), `vk.pso.census drawn= dispatched=`, `vk.indirect_stats scanned= zero_inst=`, `vk.tex_hist … top{64=…2048p=} | new= gone=`, `vk.mem_seen`, `vk.memreq` (32 formas comprimidas + 32 del resto); `PipelineObj::census`; `r.bound.subpass = 0` en `vkCmdBeginRendering`. Test `diag_census`.
- 2026-11-10 Tirones de Gundam (logs 27, DEMOSTRADO en log: en 77 s 288 `msaa_materialized` y 168 `msaa_released`, la misma textura hasta 16 veces; cada vuelta de un respaldo de 46-100 MiB = `vk.hitch backend_ms` 45-58 ms con `pso_create_ms`≈0 y `compiled_in_session=0`): `releaseMsaaBackings` (metal_backend.mm) exige `msaaReleaseFrames << min(releases, 6)` frames en tile; `Texture::releases` sube en cada liberación que no venga de aviso de memoria. Solo compilable en CI de macOS; sin probar en visor. Otros tirones medidos y sin tocar: frame con 612 copias a texturas 128x1 RGBA16F (1794 encoders, 55-82 ms, 4 veces en la escena de warmup) y `guest_gap` de 38-50 ms con subidas de 85 MiB (4096² RGBA8 con mips) repetidas.
- 2026-11-10 Lote tras logs 27 de VQP (ctest 340/340 Linux; metal_backend.mm sin tocar; sin probar en visor). (1) HLVR, DEMOSTRADO (log + binarios): el shader de preprocesado de profundidad de Meta (vertex `SV_RenderTargetArrayIndex`, `DrawProceduralNow(…, 3, 2)`) fallaba `SetPass` (`E/Unity: DrawProceduralNow requires a successful material.SetPass`) porque no anunciábamos `VK_EXT_shader_viewport_index_layer`: libunity (2021.3 y 2022.3) referencia ese nombre y `vulkan.adreno.so` del firmware de Quest 3 lo lista. Añadida a `deviceExtensions()` (physical_device.cpp); el traductor ya bajaba `BuiltIn::Layer` de vertex a `[[render_target_array_index]]` (test nuevo `kLayerFromInstanceVert` en spirv_tests.cpp); `gl_ViewportIndex` se descarta (1 viewport). (2) ITR2 (`vk.rp_dump` de logs 27: att0 color 4x y att2 D24S8 4x TRANSIENT con CLEAR/DONT_CARE, área 3552x2258 sobre imagen 3552x2260, sub1 lee att2 como input read-only sin barrier): `areaClears` trata el primer uso de un adjunto TRANSIENT como clear completo (antes Load + quads = `msaa_stale=2` por frame) y `beginPass(keepForSplit)` solo fuerza Store en un subpass que se lee a sí mismo si hay un `OpPassBarrier` por delante (`barrierAhead`); el tramo tras el último barrier recupera los stores del cliente (`selfFinal`). Queda el corte i4→i5 por Feedback (store+load 4x de color y depth una vez por frame). (3) `vk.rp_dump`: la clave incluye tamaño del framebuffer y si el pase tiene >= 16 draws; 6 formas.
- 2026-11-10 ITR2 (logs 26): `uiPass` no añade la máscara UI c7 a un pase MSAA del juego que continúa el anterior (`why=msaa_continuation`); diagnóstico `vk.rp_dump` (commands.cpp `rpDump`). La violación de transient es pegajosa (`transient = d->transient && !transientViolated`, frame_compiler.cpp placeTiles). PENDIENTE: depth como input attachment en tile (adjunto de color sombra con la z) para fusionar i4+i5; resolve de depth en tile para i6.
- 2026-11-09 Diagnósticos ITR2 (sin cambio de render; ctest 340/340 Linux; sin probar en visor): `vk.device_req exts=… names=…` y `vk.device_req features struct=… on=… unsupported=…` por petición distinta de vkCreateDevice (entry.cpp `deviceRequestLines`; `unsupported` = pedidas que el driver no reporta; `other_sType_not_decoded`); `vk.tex_hist` cada 5 s (solo si cambia, ≤240 líneas): imágenes SAMPLED por nº de mips escritos (m0, m1_2, m3_5, m6p, ASTC aparte, `partial`, `top0_unwritten`, `rt`) con `Image::mipsWritten` atómico marcado en copy/blit/copyMemoryToImage y registro global de imágenes (vkCreateImage/vkDestroyImage). Menores abiertos: recorrido del registro cada 5 s con mutex aunque lines>=240; fetch_or por región en cada copia; planos extra no contados.
## Current state
- Verde: ctest 342/342 (Linux, backend nulo); CTS rápida según arriba. Todo lo de 2026-10-27/28 sin probar en visor.
- Pendiente:
  1. Probar en visor (VQP): MSAA memoryless sin `memorylessFailure`/`msaa_materialized`, stencil tras seed, tirones por seed PSO.
  3. `bias(vp_args.sbN)` en todo muestreo implícito aunque bias = 0 (falta especialización por estado de sampler).
  4. Harvest continuo casi nunca cede con pipelines async (espera ~3 ms); vigilar `draws_skipped` y tirones.
  5. Intermitente CTS `multisample.misc.clear_attachments...ds_resolve_sample_zero.sub_framebuffer` sin causa (sync2: corregido `advance`, unverified sin CTS).
  6. Sin int64 en VPIR; `shaderInt64=0` anunciado. Quest 3 real: unverified (perfil marca "1?"); aparcado hasta confirmar.
  7. `textureCompressionBC` con identidad Quest = `platformSupportsBC()` (código/tests) vs `perfil-adreno740.md:59` = 0: discrepancia unverified (no hay vulkaninfo real de Quest 3).
  8. El backend no elimina el resolve de adjuntos `Resolve` al invalidar (solo `Store`).
  9. `vpconvert-visionos.yml` sin relanzar tras el cambio de `tools/vpaot` del 2026-11-11.
  10. Auditoría (`notas/auditoria.md`, sección "Cómo seguir"): ningún hallazgo arreglado. Gravedad A: T1, T2, C1, C2, V1, V2, V3, V4, G1, G2, P1. Rendimiento medido: C3, C4, G3, G4, G6, G7, G8, V10, V11. Faltan los lotes 4 (`gpu/shader`) y 5 (`runtime`, `tools`, `platform`, `apps`).
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
- Aplicaciones Windows / Steam (VPWin): aparcado el 2026-11-11 por coste (meses hasta un primer juego). No retomar sin que se pida. Queda la medición y el diseño en `notas/win.md`; `runtime/win/vp_win.h` (Win32 propio, solo cabecera) sigue sin uso.

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
- `notas/auditoria.md`: auditoría de recursos del 2026-11-11: hallazgos T/C/V/G/P con fichero:línea, gravedad, medición y arreglo propuesto; qué falta por auditar.
- `notas/win.md`: VPWin (Windows/Steam), aparcado: arquitectura, cobertura medida de `vpaot` sobre Wine 11, costura PE ↔ Unix con fichero:línea, plan del backend, hitos.
