# Bitácora — VPKEngine

Motor propio para Apple Vision Pro: Vulkan/SPIR-V → VPIR → MSL → Metal (`gpu/`: core, metal2, shader, `gpu/vulkan` = VPVulkan,
`gpu/gles` = vpgles) y CPU x86-64 → C AOT (`tools/vpaot`, `runtime/`). Sin MoltenVK, SPIRV-Cross ni Mesa.
Clientes: VisionQuestPro (foco), AVP, AstroVisionPro. Licencia PolyForm Noncommercial 1.0.0.

## Last done
- 2026-10-28 `frame_compiler.cpp` (~1054-1059): Load sobre TRANSIENT con memoria indefinida (`!memValid`) en su primer attach ya no marca `transient_violation` (queda tile-local con DontCare) → MSAA de ITR2 sigue memoryless (antes 49 materializados, 1249M). Tests `rtmemory` reescritos. Riesgo conocido: un guest que use un transient como acumulador entre frames pierde contenido (igual que en Quest). Sin probar en visor.
- 2026-10-28 `6f03495`: CTS rápida Vulkan 26 Fail / 0 crash (24 + 2 intermitentes), GLES 64 Fail / 0 crash. Detalle y aparcados: `notas/cts.md`.
- 2026-10-28 `69d1a00`+`f7d050f`: depth/stencil MS del framebuffer por defecto EGL memoryless (1x compañero + seed);
  regla `gameAuto` "Load de MS transient = autoMsaa" solo con `dev->internalClient` (vpgles) en `vulkan/commands.cpp`.
- 2026-10-28 `e4a81b7` (revisión adversarial): `splitHeavyAuto` parte también con stencil y depth sin resolve
  (`frame_compiler.cpp`, `kMaxTileChunks`=1024, chunks no finales resuelven D/S con Sample0); seed restaura stencil
  (8 planos de bits) y `gameAuto` mira todos los pases; seed PSO async (primer frame de firma nueva sin seed);
  `sourceView` comprueba `PixelFormatView`; TF de GLES por copia GPU sin espera si el orden coincide;
  color MS de EGL y `msTwin` transient; `floatToInt` barato con rango acotado (`kLowerVersion` 12); index_u8 cacheado,
  sin insert O(n) ni esperas de 2 s en ruta caliente.
- 2026-10-28 GLES `c38fc60..3e8338d`: copy_image, completeness, MSAA real (capas A/B, `EXT_multisampled_render_to_texture`),
  separate_shader (uniformes homónimos con prefijo `vpgl_fs_`), primitive restart, `glGetUniform` matrices no cuadradas,
  blit D/S por planos y MS vía imagen 1x temporal si cambia tamaño, stencil texturing.
- 2026-10-27/28 Vulkan perfil Quest 3 real (gpuinfo): 4 colas, subgroup arith/clustered, uint8 index, F16 constantes,
  ballot por `simd_or`, lodBias en shader; harvest de archive continuo (cola hasta 32768); `VPVK_MSAA_FORCE`,
  `VPVK_GAME_MSAA_AUTO`, `VPVK_PIPELINE_EYE_HARD_WAIT_MS`.

## Current state
- Verde: ctest 318/318; CTS rápida según arriba. Todo lo de 2026-10-27/28 sin probar en visor.
- Pendiente:
  1. Probar en visor (VQP): MSAA memoryless sin `memorylessFailure`/`msaa_materialized`, stencil tras seed, tirones por seed PSO.
  2. Renderbuffers MS de juego en GLES (`gl_api.cpp RenderbufferStorageMultisample`) siguen siendo imágenes MS almacenadas.
  3. `bias(vp_args.sbN)` en todo muestreo implícito aunque bias = 0 (falta especialización por estado de sampler).
  4. Harvest continuo casi nunca cede con pipelines async (espera ~3 ms); vigilar `draws_skipped` y tirones.
  5. `eyeHardWaitMs` de vpgles hasta 1500 ms por pipeline nuevo: puede bloquear el hilo de render al estrenar efectos.
  6. Intermitentes de Vulkan (`notas/cts.md`): el de sync2 puede ser carrera real.
  7. Sin int64 en VPIR (atómicos 64, `shaderInt64`, f64).
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
