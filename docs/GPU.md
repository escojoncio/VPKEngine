# VPEngine GPU — native Metal backend for Apple Vision Pro (design)

Goal: replace the Vulkan path (shadPS4 renderer_vulkan → KosmicKrisp/MoltenVK → Metal) with a GPU layer written for
visionOS only: Metal 4, Compositor Services, no Vulkan, no other platform code. Reusable by other translation projects
(the core knows nothing about PS4).

## Why (data: AstroVisionPro bitácora, «Informe de rendimiento», ASTRO BOT, M2)
- ~30 ms GPU per frame, GPU 94–100 %. Floor 16–18 ms that does not scale with pixels or vertices (bench: no pixels
  18–20, 3 vertices per draw 16–17.5) = fixed cost of the Vulkan-on-Metal layer: ~49–55 Metal render encoders per frame
  for ~42–46 passes, `end_encoder` ALL→ALL barriers, copies through compute, RectList/QuadList through compute
  tessellation (pass split + 5 dispatches per draw), robust buffer/image access checks in every access.
- 816 ≈ 1440 per eye in the scene: pixels are not the limit today. Foveation and MetalFX only pay once the fixed cost
  is gone. Order: remove overhead first, then foveation, then temporal upscaling.
- MoltenVK 77 ms, KK 30 ms: the layer matters 2.5×.

## Layers
1. `gpu/core` (platform-neutral API inside VPEngine, PS4-agnostic):
   - Resources: buffers (shared storage, unified memory: guest memory mapped directly where possible, no staging),
     textures (formats table, swizzle/detile in compute only when needed), samplers, heaps (Metal 4 residency sets).
   - Passes: explicit render pass objects; the frontend declares attachments/loads/stores; one Metal encoder per real
     pass (merge consecutive passes with same attachments; no split for copies — blits batched before/after).
   - Barriers: tracked per resource (last writer stage) → Metal 4 `barrier` only where a hazard exists.
   - Pipelines: key → MTLRenderPipelineState/MTL4 compiled function; disk cache in VPS4; background compilation with
     priority (already proven in AstroVisionPro: 117 pipelines in 228 ms from cache).
   - Shaders: IR → MSL text (or Metal IR later) emitter; fast-math off only where the IR marks precise.
   - Diagnostics in the log from day one: per-pass GPU time (counter sample buffers), encoder count and why each was
     cut, pipeline compile times, validation of every resource binding (no Mac, no debugger: the log is the debugger).
2. `gpu/visionos`: Compositor Services presenter: game eye images rendered straight into drawable textures when the
   game's eye matches; reprojection kept; foveation = the drawable's rasterization rate map applied to the game's main
   scene passes (render target sizes from the map's physical size); MetalFX temporal upscaler (needs depth + motion
   vectors: motion from depth + head pose delta = camera motion only; moving objects may ghost → per-title switch).
3. `integrations/shadps4/metal`: frontend glue replacing `renderer_vulkan` (13.4k lines) and the Vulkan parts of
   `texture_cache`/`buffer_cache`: Liverpool PM4 processing, shader recompiler (GCN → shadPS4 IR) stay; new backend
   `shader_recompiler/backend/msl` next to `backend/spirv` (5.5k lines) emitting MSL from the same IR.
   - Geometry shaders (ASTRO BOT uses them): mesh/object shaders in Metal (object = VS+GS expansion, mesh = output),
     inside the pass. RectList/QuadList: vertex shader computes the 4th vertex (v1+v2−v0) as a triangle list, no
     tessellation. Fans/restart: index rewrite on the GPU in the same pass.

## Verification without the headset
- MSL compiles with `xcrun metal` on macOS CI runners; the macOS runner GPU (paravirtual Metal) can run small render
  tests: draw a PS4 command stream capture, read back, compare with a golden image (as tests/aot does for the CPU).
- Shader corpus: SPIR-V/IR of ASTRO BOT's shaders already dumped in the logs (`GPU_SHADER_SPIRV`) → emit MSL for all of
  them in CI, compile, and diff against SPIRV-Cross output on the same input for semantics.

## Steps (each one testable; device rounds only from 3)
1. IR → MSL emitter + corpus compile in CI (macOS runner).
2. gpu/core + headless render test on the macOS runner (triangle, textured, depth, MRT, GS via mesh, RectList).
3. shadPS4 Metal renderer behind a build option (`ENABLE_METAL_NATIVE`), KK build untouched; first device round:
   menus/logos (flat path).
4. ASTRO BOT world: correctness, then encoder/barrier counts vs KK (`KK_CENSUS`), then ms.
5. Foveation in the scene passes; 6. MetalFX temporal; 7. thermal: fps cap option.

Success criterion per step: the log numbers (GPU ms per frame, encoders per frame) against KK's 30 ms / ~50 encoders.
