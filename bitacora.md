# Bitácora — VPKEngine

Motor propio para Apple Vision Pro: Vulkan/SPIR-V → VPIR → MSL → Metal (`gpu/`: core, metal2, shader, `gpu/vulkan` = VPVulkan,
`gpu/gles` = vpgles) y CPU x86-64 → C AOT (`tools/vpaot`, `runtime/`). Sin MoltenVK, SPIRV-Cross ni Mesa.
Clientes: VisionQuestPro (foco), AVP, AstroVisionPro. Licencia PolyForm Noncommercial 1.0.0.

## Last done
- 2026-11-09 Maestro (UE4, GLES; causa del negro NO demostrada: siluetas correctas ⇒ posiciones/skinning bien, es sombreado): `gles.uniform_lenient` añade `count= arraySize= element= prev= new=`; `gles.vattr` (combinaciones distintas de atributo de vértice, ≤64, repite en n=10000/1000000, `mismatch=`/`cpu=`; lock global en hot path, menor); `VPVK_MSL_SAFE_MATH=1` compila MSL con MTLMathModeSafe (metal_backend.mm ~:2234, clave de caché solo cambia con el flag; solo cubre NaN/Inf, no precisión de sin/cos; `gpu.shaderfx math=fast` sigue sin reflejarlo; metal_backend.mm sin compilar en Linux); `replay.yml`/`replay-vpvk` aceptan `vpvk_env`. Refutado por código: mediump→half (todo fp32), LOD por frustum/resolución (UE4.27 usa P00/P11), decodificación 2_10_10_10/SNORM correcta. Bugs sin relación (sin tocar): draw.cpp ~2029 `rows=(instances+divisor-1)/divisor` desborda uint32; ~1802 `readComponent` sin caso HALF_FLOAT. Prueba única: captura de frames en Maestro + `gles.unwritten`/`gles.vattr`/`uniform_lenient`; luego replay.yml con `vpvk_env=VPVK_MSL_SAFE_MATH=1`. ctest 337/337, sin CTS, sin probar en visor.
- 2026-11-09 Diagnósticos (sin cambio de render): `gles.unwritten` + `unwritten=` en `gles.census` (textura comprimida completa muestreada con algún nivel sin subida/copia; relleno ASTC a cero = void-extent negro) y `gpu.msaa_stale` (por textura leída de un submit anterior: prev_store D/S/R/SR, keep, memoryless, why; tras 32 líneas solo cada 600 frames). `noteUnwritten` ignora TEXTURE_MIN/MAX_LOD (menor). Hallazgo: ITR2 renderiza en VULKAN (`vk.instance engine=UnrealEngine5.5`), NO GLES: el arreglo MSRTT de GLES (f426f82) no se ejecuta en ITR2. ctest gles/core/vulkan 159/159, sin CTS, sin probar en visor.
- 2026-11-09 GLES: depth/stencil de textura MSRTT (ITR2, pass4→pass5) ya no continúa en DONT_CARE sin siembra: `Context::beginPass` (draw.cpp ~327-345) `seeded = seedImage || tex`; el backend siembra el gemelo desde la 1x (la 1x recibe resolve SAMPLE_ZERO). Cubre D32F y D24S8 con depth+stencil; HUECO conocido (unverified en ITR2): depth-only sobre formato Vulkan con stencil (DEPTH_COMPONENT24, D24S8 solo-depth) no se siembra (`gameAuto` exige stencilResolve, commands.cpp:1210, requiere tocar VPVulkan). `MUTABLE_FORMAT` solo al adjuntar como MSRTT (`makeFormatViews`, gl_api.cpp `framebufferTexture2D`; también en solo-DEPTH D24S8 sin beneficio). Diagnósticos: `gles.incomplete` (texturas incompletas con primer nivel que falla; fallback negro 1x1), `gles.tex_realloc`, `gles.msrtt_ds` (+census `msrtt_ds_continued`). Test `msrtt_depth`; ctest gles/core/vulkan 159/159 sin sanitizers; sin CTS; sin probar en visor; metal_backend.mm no tocado.
- 2026-11-01 `ee7691f`: zero-init de ASTC copia el bloque void-extent (FC FD FF.. + 0) desde `zeroAstcBuffer` (entry.cpp ~2978); antes bytes 0 = bloque reservado = magenta en mips sin subir (causa probable del morado de Maestro, unverified en visor). GLES no estricto: `count_on_non_array` aplica el 1.º elemento, location fuera de rango se ignora; texturas mutables con BASE_LEVEL>0 sin nivel 0 (`realizeTexture` usa w<<a); max<base incompleta.
- 2026-11-01 `550898b`: blob de VkPipelineCache con identidad de receta sin MSL (`RecipeStore::recipeId`, recipes.cpp ~990); `VPVK_ASYNC_PIPELINES=0` = create síncrono sin backoff, `VPVK_CREATE_WAIT_MS` (60 s), `vk.pipeline_create_slow`; `vpvk_prewarm` hace flush completo del archive (10 min, fase 1 en `vpvk_prewarm_phase`, pausa 200 ms con poca memoria). Causa de 2400 misses por arranque: flush de 8 s (~100 entradas).
- 2026-11-01 `858a535`: MSAA forzado sube pases de ojo ya MSAA (s<N) con gemelos memoryless que resuelven en los destinos del juego; rechazos `msstore`, `smask`, unreadable; contador `up=`.
- 2026-11-01 `ce0b08b`: `eyeHardWaitMs` 0 por defecto (también vpgles). Targets no-ojo redibujados >3 ticks de ojo seguidos (por vista textura+mip+capa, tolerancia 1 tick) con primer pase Clear/DontCare → espera blanda (skip); con Load o esporádicos siguen duros (`persistWaitMs`). `Device::redrawStamps` (pipelines.cpp).
- 2026-11-01 `303c644`: GLES renderbuffers MS memoryless: imagen 1x + gemelo TRANSIENT (`msTwin` con clave por slot, caché 32 protegiendo el pase actual), seed/resolve como EXT_msrtt; blit desde la 1x; invalidate sobre adjuntos implícitos marca drop (el backend no elimina resolves `Resolve`); CopyImageSubData con muestras distintas = INVALID_OPERATION. Test `msrbo`.
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
