# CTS (Vulkan CTS y dEQP GLES) en CI

## Cómo se lanza
- Vulkan: `.github/workflows/cts.yml`. Commit con `[cts-rapida]` = subconjunto rápido (36087 casos). Focus:
  `gh api -X POST repos/escojoncio/VPKEngine/actions/workflows/cts.yml/dispatches -f ref=main -f inputs[cases]=<patrones>`.
- GLES: `.github/workflows/gles-cts.yml`, `[gles-rapida]` (7634 casos), mismo esquema de focus.
- Resultados en ramas `ci-logs-cts-report` / `ci-logs-cts-focus` / `ci-logs-gles-report` / `ci-logs-gles-focus`
  (`git fetch origin '+refs/heads/ci-logs-*:refs/remotes/origin/ci-logs-*'`); `*-fails.txt` por shard y diff contra el run anterior.
- Los focus comparten rama y se cancelan entre sí: lanzarlos en secuencia.
- Máquina: macOS 26 VM M1 (Apple5, sin Metal3 ni bindless). GLSL estricto (`VPGLES_STRICT_GLSL`) solo en CI; juegos en modo permisivo.
- Config GLES por defecto `ms0`; tests de MSAA del framebuffer por defecto con `ms4`.
- El proxy de la sesión no deja borrar ramas remotas (403): las ramas temporales las borra el usuario.

## Estado (commit `6f03495`, 2026-10-28)
- Vulkan rápida: Pass 11602, Fail 26, Crash 0, DeviceLost 0.
- GLES rápida: Pass 7189, Fail 64, NS 380, Crash 0.

## Fallos aparcados (perfil Quest 3, nicho o a ciegas)
- Vulkan: multisample misc 5, border_swizzle 2, d24s8 sampler 4, texture.mipmap projected 5, renderpasses
  input_output/ds_resolve 5, multiview input_attachments 2, culling point 1, opcompositeinsert v3f16 1, ballot.compute 2.
- Vulkan intermitentes (fallan también en `e4a81b7`, 4 de 6 runs): `multisample.misc.clear_attachments...2x.ds_resolve_sample_zero.sub_framebuffer`,
  `synchronization2.signal_order.timeline_semaphore.write_image_vertex_read_image_vertex.image_64x64x8_r32_sfloat` (posible carrera real, sin causa).
- Vulkan fuera de perfil: geometry/tess, subgroup 64, descriptor_indexing (bindless), pipelineStatistics,
  robustness2 nullDescriptor, 4444, offsets de texel fuera de -8..7, maxViewports, sparse, RT.
- GLES: GLES2 mipmap projected 5; texturas de profundidad sized/specification (swizzle de vista rompía invalidate);
  struct varyings (linker vpglsl); blit LINEAR fuera de límites; cube filtering 2 (expuestos por `ms0`);
  invariance, state_query, ubo/ssbo random, negative_api, program_interface_query, image_load_store,
  texture.multisample, sample counts; gather rgba8i level_1; nested_array usampler2D.
- GLES `multisample.stencil` (GLES2) y `multisample.default_framebuffer.stencil` (GLES3): stencil por muestra perdido
  tras readPixels a mitad de pase, coste aceptado de depth/stencil MS memoryless.

## Hallazgos de GPU Apple
- `simd_ballot`/`simd_active_threads_mask`/`simd_broadcast_first` no fiables en código divergente: usar `simd_or` de bits de carril.
- `simd_broadcast_first` cuelga en fragment. `lodBias` del sampler por KVC no tiene efecto: bias en shader (`SamplerBias`).
- Offsets de gather -8..7. Metal aborta con vista 2D de textura 3D.
