# Bitácora — VPKEngine

Motor propio para Apple Vision Pro: Vulkan/SPIR-V → VPIR → MSL → Metal (`gpu/`: core, metal2, shader, `gpu/vulkan` = VPVulkan,
`gpu/gles` = vpgles) y CPU x86-64 → C AOT (`tools/vpaot`, `runtime/`). Sin MoltenVK, SPIRV-Cross ni Mesa.
Clientes: VisionQuestPro (foco), AVP, AstroVisionPro. Licencia PolyForm Noncommercial 1.0.0.

## Last done
- 2026-11-01 `5ebfa3a`: timeline `advance` (ext.cpp:53-65) completa en orden de valor y para en la primera señal sin terminar; `waitSemaphores` elige frame acorde. drainDeferred ya era FIFO por cola. Test `timeline_order`.
- 2026-11-01 `ce0b08b`: `eyeHardWaitMs` 0 por defecto (también vpgles). Targets no-ojo redibujados >3 ticks de ojo seguidos (por vista textura+mip+capa, tolerancia 1 tick) con primer pase Clear/DontCare → espera blanda (skip); con Load o esporádicos siguen duros (`persistWaitMs`). `Device::redrawStamps` (pipelines.cpp).
- 2026-11-01 `303c644`: GLES renderbuffers MS memoryless: imagen 1x + gemelo TRANSIENT (`msTwin` con clave por slot, caché 32 protegiendo el pase actual), seed/resolve como EXT_msrtt; blit desde la 1x; invalidate sobre adjuntos implícitos marca drop (el backend no elimina resolves `Resolve`); CopyImageSubData con muestras distintas = INVALID_OPERATION. Test `msrbo`.
- 2026-11-01 `057cd69`: detección de ojo sin TAA: `Device::eyeSize` (última imagen marcada con `vpvk_mark_eye_image`) como respaldo en `taaEyePass`, `eyeSizedPass`, `msaaObservePass`; multivista sin tamaño de jitter se sigue aceptando. Test `msaa_force_marked`.
- 2026-11-01 `122fdb3`: harvest de MTLBinaryArchive cede con backlog de compilación (`compileBacklog()`, pipeline_cache.cpp) o memoria libre < `VPVK_HARVEST_MIN_FREE_MB` (512); válvula 512 entradas/20 s solo anula el backlog. `trimMemory(>=2)` vacía `archiveQueue` (`archive_trim_dropped` en `vk.pso.census`). Función pura `archiveHarvestShouldYield` (archive_harvest.h) con tests. metal_backend.mm sin compilar en Linux (unverified hasta CI).
- 2026-11-01 `4b44477`: GLES `setUniform` (draw.cpp ~952): tipo base distinto (float/int/uint, mismo nº de componentes, no matriz) copia bits crudos fuera de modo estricto (`gles.uniform_lenient`); `VPGLES_STRICT_GLSL=1` mantiene INVALID_OPERATION. Causa: UE4 sube `glUniform4iv` sobre `vec4 vc0_h[]` (Maestro morado; unverified en visor). Tests `uniform_raw`, `uniform_strict`.
- 2026-10-28 `gpu/vulkan/src/commands.cpp`+`vpvk.h`: forceMsaa cuenta las salidas silenciosas (`vk.msaa force=... rej{n,eye,cd,smp,res,desc}`) y `vk.msaa rej <motivo> WxHxL color= depth=` (12+12 combinaciones). No cambia qué pasadas se convierten. AoT: 6 twins s4 de 1116x1129; la pasada al ojo 3552x2258 s1 no aparece como twin ni rechazo (unverified por qué).
- 2026-10-28 `frame_compiler.cpp` (~1054-1059): Load sobre TRANSIENT con memoria indefinida (`!memValid`) en su primer attach ya no marca `transient_violation` (queda tile-local con DontCare) → MSAA de ITR2 sigue memoryless (antes 49 materializados, 1249M). Tests `rtmemory` reescritos. Riesgo conocido: un guest que use un transient como acumulador entre frames pierde contenido (igual que en Quest). Sin probar en visor.

## Current state
- Verde: ctest 323/323 (Linux, backend nulo); CTS rápida según arriba. Todo lo de 2026-10-27/28 sin probar en visor.
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
